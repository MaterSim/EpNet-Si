"""Differentiable local plane-wave Hamiltonian with energies in eV."""

from __future__ import annotations

import math

import torch
from torch import Tensor

# T[eV] = KINETIC_EV_A2 * |k[Å^-1]|^2  (ħ²/2m_e).
KINETIC_EV_A2 = 3.809982116151619
BOHR_ANGSTROM = 0.529177210903
HARTREE_EV = 27.211386245988
# 4π e²/(4πε0) in eV·Å, so V[eV] = COULOMB_EV_A * 4π S(G) / (Ω[Å³] |G[Å⁻¹]|²).
COULOMB_EV_A = HARTREE_EV * BOHR_ANGSTROM


def g_cutoff_angstrom(ecut_ry: float) -> float:
    """Return the |G| cutoff in Å⁻¹ for a kinetic cutoff in Rydberg."""
    if ecut_ry <= 0:
        raise ValueError("ecut_ry must be positive")
    return math.sqrt(ecut_ry) / BOHR_ANGSTROM


def pairwise_g_differences(g_indices: Tensor) -> Tensor:
    if g_indices.ndim != 2 or g_indices.shape[-1] != 3:
        raise ValueError("g_indices must have shape (n_g, 3)")
    return g_indices[:, None, :] - g_indices[None, :, :]


def pairwise_g_cartesian(g_indices: Tensor, reciprocal_lattice: Tensor) -> Tensor:
    diffs = pairwise_g_differences(g_indices).to(reciprocal_lattice)
    return diffs @ reciprocal_lattice


def pairwise_momentum_transfer(g_indices: Tensor, reciprocal_lattice: Tensor) -> Tensor:
    return pairwise_g_cartesian(g_indices, reciprocal_lattice)


def unique_g_differences(g_indices: Tensor) -> tuple[Tensor, Tensor]:
    """Unique Miller differences G_i - G_j (rows sorted lexicographically) and the (n_g, n_g) inverse map.

    Same result as torch.unique(diffs.reshape(-1, 3), dim=0, return_inverse=True), but via one int64 key per
    difference: with every component shifted to [0, S) the key (d0 S + d1) S + d2 is order-preserving, and it is
    linear, so key(G_i - G_j) = k_i - k_j + const needs no (n_g, n_g, 3) array. The 1-D unique is far faster than
    unique(dim=0) on large supercells.
    """
    g = g_indices.to(torch.int64)
    span = g.amax(0) - g.amin(0)
    size = int(2 * span.max()) + 1
    k = (g[:, 0] * size + g[:, 1]) * size + g[:, 2]
    offset = (span[0] * size + span[1]) * size + span[2]
    keys, inverse = torch.unique(k[:, None] - k[None, :] + offset, sorted=True, return_inverse=True)
    d2 = keys % size
    d1 = (keys // size) % size
    d0 = keys // (size * size)
    unique_idx = torch.stack((d0 - span[0], d1 - span[1], d2 - span[2]), dim=-1).to(g_indices.dtype)
    return unique_idx, inverse.reshape(g_indices.shape[0], g_indices.shape[0])


def unique_g_transfers(g_indices: Tensor, reciprocal_lattice: Tensor) -> tuple[Tensor, Tensor]:
    """Return unique Cartesian ΔG and the (n_g, n_g) inverse index map.

    Uniqueness is on integer Miller differences, then mapped with the reciprocal lattice.
    """
    unique_idx, inverse = unique_g_differences(g_indices)
    unique_cart = unique_idx.to(reciprocal_lattice) @ reciprocal_lattice
    return unique_cart, inverse


def structure_phases(
    g_indices: Tensor,
    fractional_positions: Tensor,
    *,
    unique_idx: Tensor | None = None,
    inverse: Tensor | None = None,
) -> Tensor:
    """Return Σ_a exp(-2πi (G_i-G_j)·r_a) with r in crystal coordinates.

    With ``unique_idx``/``inverse`` from unique_g_differences, the phases are computed once per unique difference
    and gathered (n_unique ~ 8 n_g instead of n_g^2 exponentials per atom), with the same atom chunks, so the sums
    are the same; without them, the original pair path.
    """
    n_g, n_atoms = g_indices.shape[0], fractional_positions.shape[0]
    # Large external cells (e.g. 82-atom mS164) need n_g^2 * n_atoms complex entries at once; sum over atom
    # chunks instead. Cells below the limit keep the single-shot path unchanged.
    atoms_per_chunk = max(1, int(1.5e9 // max(n_g * n_g, 1)))
    if unique_idx is not None:
        diffs_u = unique_idx.to(fractional_positions)
        total = None
        for a0 in range(0, n_atoms, atoms_per_chunk):
            dots = torch.einsum("uc,ac->ua", diffs_u, fractional_positions[a0 : a0 + atoms_per_chunk])
            part = torch.exp(-2j * math.pi * dots).sum(dim=-1)
            total = part if total is None else total + part
        return total[inverse.reshape(-1).to(torch.int64)].reshape(n_g, n_g)
    diffs = pairwise_g_differences(g_indices).to(fractional_positions)
    if n_atoms <= atoms_per_chunk:
        dots = torch.einsum("ijc,ac->ija", diffs, fractional_positions)
        return torch.exp(-2j * math.pi * dots).sum(dim=-1)
    total = None
    for a0 in range(0, n_atoms, atoms_per_chunk):
        dots = torch.einsum("ijc,ac->ija", diffs, fractional_positions[a0 : a0 + atoms_per_chunk])
        part = torch.exp(-2j * math.pi * dots).sum(dim=-1)
        total = part if total is None else total + part
    return total


def kinetic_diagonal(g_indices: Tensor, reciprocal_lattice: Tensor, k_frac: Tensor) -> Tensor:
    """Return T(k+G) in eV. ``k_frac`` is crystal coordinates, shape (3,) or (n_k, 3)."""
    g_cart = g_indices.to(reciprocal_lattice) @ reciprocal_lattice
    k = k_frac.to(reciprocal_lattice)
    if k.ndim == 1:
        k = k[None, :]
    k_cart = k @ reciprocal_lattice
    momentum = k_cart[:, None, :] + g_cart[None, :, :]
    return KINETIC_EV_A2 * (momentum * momentum).sum(dim=-1)


def local_potential_matrix(
    phases: Tensor,
    form_factor: Tensor,
    volume: Tensor | float,
    *,
    q: Tensor | None = None,
    zero_gauge: bool = True,
) -> Tensor:
    """Assemble V_{GG'} = S(G-G') v(|G-G'|) / Ω, then Hermitian-symmetrize."""
    values = form_factor.to(phases)
    if values.ndim == 1:
        matrix = phases * values
    else:
        matrix = phases * values
    matrix = matrix / float(volume)
    if zero_gauge:
        n = matrix.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=matrix.device)
        matrix = matrix.masked_fill(eye, 0)
    return 0.5 * (matrix + matrix.conj().transpose(-1, -2))


def add_multipole_potential(
    spherical: Tensor,
    multipole_values: Tensor,
    inverse: Tensor,
    volume: Tensor | float,
    q: Tensor,
    *,
    zero_gauge: bool = True,
) -> Tensor:
    """Add a unique-ΔG multipole form factor onto a spherical V_{GG'}."""
    extra = multipole_values.to(spherical)[inverse.to(dtype=torch.int64)] / float(volume)
    if zero_gauge:
        n = extra.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=extra.device)
        extra = extra.masked_fill(eye, 0)
    out = spherical + extra
    return 0.5 * (out + out.conj().transpose(-1, -2))


# Batched MKL eigvalsh beats scipy subset until n_G is well above this catalog.
# Tests pass subset_ng_min=0 to exercise the lowest-n autograd path.
SUBSET_NG_MIN = 4096


def _subset_eigh(ham: Tensor, n_bands: int) -> tuple[Tensor, Tensor]:
    """Lowest ``n_bands`` eigenpairs of a batched Hermitian matrix (CPU)."""
    import numpy as np
    import scipy.linalg as sla

    mats = ham.detach().cpu().contiguous().numpy().copy()
    n_k, n_g, _ = mats.shape
    evals = np.empty((n_k, n_bands), dtype=np.float64)
    evecs = np.empty((n_k, n_g, n_bands), dtype=mats.dtype)
    for ik in range(n_k):
        values, vectors = sla.eigh(
            mats[ik],
            subset_by_index=(0, n_bands - 1),
            driver="evr",
            check_finite=False,
            overwrite_a=True,
        )
        evals[ik] = values
        evecs[ik] = vectors
    return (
        torch.as_tensor(evals, device=ham.device, dtype=ham.real.dtype),
        torch.as_tensor(evecs, device=ham.device, dtype=ham.dtype),
    )


class _LowestEigvalsh(torch.autograd.Function):
    """Eigenvalues of the lowest ``n_bands`` Hermitian subspace, with projector backward."""

    @staticmethod
    def forward(ctx, ham: Tensor, n_bands: int) -> Tensor:
        evals, evecs = _subset_eigh(ham, int(n_bands))
        ctx.save_for_backward(evecs)
        return evals

    @staticmethod
    def backward(ctx, grad_evals: Tensor):
        (evecs,) = ctx.saved_tensors
        scaled = evecs * grad_evals.to(dtype=evecs.real.dtype).unsqueeze(-2).to(evecs)
        grad_ham = scaled @ evecs.transpose(-1, -2).conj()
        grad_ham = 0.5 * (grad_ham + grad_ham.transpose(-1, -2).conj())
        return grad_ham, None


class _TruncatedEigvalsh(torch.autograd.Function):
    """Full dense eigh, but backward only through the lowest ``n_bands`` evecs.

    The catalog loss never sees the high spectrum, so this is the exact
    ``eigvalsh`` gradient with a much cheaper ``U_s diag(g) U_s†``.
    """

    @staticmethod
    def forward(ctx, ham: Tensor, n_bands: int) -> Tensor:
        evals, evecs = torch.linalg.eigh(ham)
        n_keep = int(n_bands)
        ctx.save_for_backward(evecs[..., :n_keep].contiguous())
        return evals[..., :n_keep]

    @staticmethod
    def backward(ctx, grad_evals: Tensor):
        (evecs,) = ctx.saved_tensors
        scaled = evecs * grad_evals.to(dtype=evecs.real.dtype).unsqueeze(-2).to(evecs)
        grad_ham = scaled @ evecs.transpose(-1, -2).conj()
        grad_ham = 0.5 * (grad_ham + grad_ham.transpose(-1, -2).conj())
        return grad_ham, None


class _ScatterHermitianPotential(torch.autograd.Function):
    """``V = herm(v_unique[inverse] / Ω)`` without graphing the n_G×n_G gather."""

    @staticmethod
    def forward(ctx, v_unique: Tensor, inverse: Tensor, volume: float) -> Tensor:
        # Ampere (A40/A100) CUDA gather asserts on int32 complex index; Ada L40S does not.
        inverse = inverse.to(dtype=torch.int64)
        ctx.save_for_backward(inverse)
        ctx.n_unique = int(v_unique.shape[0])
        ctx.volume = float(volume)
        matrix = v_unique[inverse] / ctx.volume
        eye = torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
        matrix = matrix.masked_fill(eye, 0)
        return 0.5 * (matrix + matrix.conj().transpose(-1, -2))

    @staticmethod
    def backward(ctx, grad_matrix: Tensor):
        (inverse,) = ctx.saved_tensors
        g = 0.5 * (grad_matrix + grad_matrix.conj().transpose(-1, -2)) / ctx.volume
        eye = torch.eye(g.shape[0], dtype=torch.bool, device=g.device)
        g = g.masked_fill(eye, 0)
        dv = torch.zeros(ctx.n_unique, dtype=g.dtype, device=g.device)
        dv.scatter_add_(0, inverse.reshape(-1), g.reshape(-1))
        return dv, None, None


def scatter_hermitian_potential(v_unique: Tensor, inverse: Tensor, volume: float) -> Tensor:
    return _ScatterHermitianPotential.apply(v_unique, inverse, float(volume))


def band_energies(
    kinetic: Tensor,
    potential: Tensor,
    *,
    n_bands: int | None = None,
    subset_ng_min: int = SUBSET_NG_MIN,
) -> Tensor:
    if kinetic.ndim != 2:
        raise ValueError("kinetic must have shape (n_k, n_g)")
    ham = potential + torch.diag_embed(kinetic.to(dtype=potential.dtype))
    n_g = int(ham.shape[-1])
    use_subset = (
        n_bands is not None
        and 0 < int(n_bands) < n_g
        and n_g >= int(subset_ng_min)
        and ham.device.type == "cpu"
    )
    if use_subset:
        return _LowestEigvalsh.apply(ham, int(n_bands))
    if n_bands is not None and 0 < int(n_bands) < n_g and ham.requires_grad:
        return _TruncatedEigvalsh.apply(ham, int(n_bands))
    evals = torch.linalg.eigvalsh(ham)
    if n_bands is not None:
        evals = evals[:, :n_bands]
    return evals


def batched_band_energies(
    kinetic: Tensor,
    potential: Tensor,
    *,
    n_bands: int | None = None,
    k_batch: int | None = None,
    subset_ng_min: int = SUBSET_NG_MIN,
) -> Tensor:
    n_g = int(kinetic.shape[1])
    if k_batch is None:
        if kinetic.device.type == "cuda":
            # One cuSOLVER batch beats many small launches. 33×799 H is ~0.3 GB.
            k_batch = max(int(kinetic.shape[0]), 1)
        else:
            k_batch = 16 if n_g < 400 else 4
    chunks = []
    for start in range(0, kinetic.shape[0], k_batch):
        sl = slice(start, start + k_batch)
        chunk_potential = potential[sl] if potential.ndim == 3 else potential
        chunks.append(
            band_energies(
                kinetic[sl],
                chunk_potential,
                n_bands=n_bands,
                subset_ng_min=subset_ng_min,
            )
        )
    return torch.cat(chunks, dim=0)


def vbm_aligned_window_loss(
    prediction: Tensor,
    target: Tensor,
    n_occupied: int,
    *,
    half_width: float = 2.0,
    edge_weight: float = 5.0,
) -> Tensor:
    """MAE after aligning VBMs, restricted to ±half_width eV of the target midgap."""
    pred = prediction
    tgt = target
    pred_vbm = pred[:, n_occupied - 1].amax()
    tgt_vbm = tgt[:, n_occupied - 1].amax()
    aligned = pred - pred_vbm + tgt_vbm
    tgt_cbm = tgt[:, n_occupied].amin()
    mid = 0.5 * (tgt_vbm + tgt_cbm)
    mask = (tgt - mid).abs() <= half_width
    # A gap wider than 2*half_width would otherwise select no states and make
    # the window loss identically zero.  Band edges are always part of the
    # objective, irrespective of gap size.
    mask[:, n_occupied - 1] = True
    mask[:, n_occupied] = True
    weight = torch.ones_like(tgt)
    weight[:, n_occupied - 1] = edge_weight
    weight[:, n_occupied] = edge_weight
    weight = weight * mask.to(weight)
    denom = weight.sum().clamp_min(1.0)
    return ((aligned - tgt).abs() * weight).sum() / denom


def deep_occupied_band_loss(
    prediction: Tensor,
    target: Tensor,
    n_occupied: int,
    *,
    gapped: bool,
    half_width: float = 2.0,
    metal_align: str = "proxy",
) -> Tensor:
    """MAE on occupied states below the scoring window (the deep valence manifold).

    Same alignment as the matching window loss (VBM for gapped cells, the Fermi proxy for metals); only
    occupied bands whose target lies more than ``half_width`` below the window reference count. Zero when
    no such states exist.
    """
    pred, tgt = prediction, target
    if gapped:
        p_ref, t_ref = pred[:, n_occupied - 1].amax(), tgt[:, n_occupied - 1].amax()
        centre = 0.5 * (t_ref + tgt[:, n_occupied].amin())
    else:
        p_ref = 0.5 * (pred[:, n_occupied - 1].amax() + pred[:, n_occupied].amin())
        t_ref = centre = 0.5 * (tgt[:, n_occupied - 1].amax() + tgt[:, n_occupied].amin())
        if metal_align == "median":  # same shift as the median-aligned window loss
            p_ref = t_ref + (pred - tgt)[(tgt - centre).abs() <= half_width].median()
    aligned = pred[:, :n_occupied] - p_ref + t_ref
    t = tgt[:, :n_occupied]
    mask = (t < centre - half_width).to(t)
    return ((aligned - t).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def fermi_aligned_window_loss(
    prediction: Tensor,
    target: Tensor,
    n_occupied: int,
    *,
    half_width: float = 2.0,
    align: str = "proxy",
) -> Tensor:
    """MAE after aligning Fermi proxies, restricted to ±half_width eV of target E_F.

    ``align="median"`` shifts the prediction by its median offset over the window instead of matching the
    proxies (the proxy is not a Fermi level for metals and can misplace the model by up to ~1 eV).

    Used for metallic cells. The Fermi proxy is the mid-point of the highest
    occupied and lowest unoccupied eigenvalues; gap-edge band weights are not
    applied.
    """
    pred = prediction
    tgt = target
    pred_ef = 0.5 * (pred[:, n_occupied - 1].amax() + pred[:, n_occupied].amin())
    tgt_ef = 0.5 * (tgt[:, n_occupied - 1].amax() + tgt[:, n_occupied].amin())
    mask = (tgt - tgt_ef).abs() <= half_width
    if align == "median":
        aligned = pred - (pred - tgt)[mask].median()
    elif align == "proxy":
        aligned = pred - pred_ef + tgt_ef
    else:
        raise ValueError(f"align must be proxy or median, got {align!r}")
    denom = mask.to(tgt).sum().clamp_min(1.0)
    return ((aligned - tgt).abs() * mask.to(tgt)).sum() / denom
