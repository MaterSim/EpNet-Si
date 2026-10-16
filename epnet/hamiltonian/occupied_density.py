"""Eigen-decomposition with a backward pass for losses that depend only on the occupied subspace.

torch.linalg.eigh's backward divides by every eigenvalue difference, so it returns NaN when occupied bands are
degenerate (common by symmetry) even if the loss, e.g. the occupied-state density, is invariant under rotations within
the occupied subspace. For such losses the occupied-occupied (and empty-empty) terms of the eigenvector derivative are
exactly zero, so they are dropped; only occupied-empty pairs remain, whose denominators are at least the gap.
Valid only when the loss is invariant under unitary mixing of the occupied eigenvectors among themselves.
"""
from __future__ import annotations

import torch
from torch import Tensor


class _OccupiedEigh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H: Tensor, n_occ: int, gap_floor: float = 0.0):
        w, U = torch.linalg.eigh(H)
        ctx.save_for_backward(w, U)
        ctx.n_occ = int(n_occ)
        ctx.gap_floor = float(gap_floor)
        return w, U

    @staticmethod
    def backward(ctx, gw: Tensor | None, gU: Tensor | None):
        w, U = ctx.saved_tensors
        n = w.shape[-1]
        inner = torch.zeros(n, n, dtype=U.dtype, device=U.device)
        if gU is not None:
            M = U.conj().transpose(-1, -2) @ gU
            occ = torch.arange(n, device=w.device) < ctx.n_occ
            mixed = occ[:, None] ^ occ[None, :]  # occupied-empty pairs only
            dw = w[None, :] - w[:, None]
            if ctx.gap_floor > 0:  # keep |dw| >= gap_floor for occupied-empty pairs whose gap nearly closes during a fit
                dw = torch.where(dw.abs() < ctx.gap_floor, torch.sign(dw + (dw == 0)) * ctx.gap_floor, dw)
            F = torch.where(mixed, 1.0 / torch.where(mixed, dw, torch.ones_like(dw)), torch.zeros_like(dw)).to(U.dtype)
            inner = F * (M - M.conj().transpose(-1, -2)) / 2
        if gw is not None:
            inner = inner + torch.diag(gw.to(U.dtype))
        gH = U @ inner @ U.conj().transpose(-1, -2)
        return (gH + gH.conj().transpose(-1, -2)) / 2, None, None


def occupied_eigh(H: Tensor, n_occ: int, gap_floor: float = 0.0) -> tuple[Tensor, Tensor]:
    """eigh(H) with gradients restricted to occupied-empty mixing; use only for occupied-subspace-invariant losses.

    ``gap_floor`` (same units as H) bounds the occupied-empty denominators from below; 0 keeps the exact gradient.
    """
    return _OccupiedEigh.apply(H, n_occ, gap_floor)


def occupied_density(model, row: dict, gap_floor: float = 0.05) -> Tensor:
    """Valence density of the n_occ lowest bands on the row's density k-points, on the row's FFT grid (flattened).

    Uses row keys dens_k (n_k, 3), dens_cache (a PW cache for dens_k), dens_lin (basis -> flat FFT index),
    dens_shape (FFT grid). Normalised to 2 n_occ electrons per cell, in electrons per grid voxel times N/volume.
    """
    kin, pot = model.band_path(row["packed"], row["dens_k"], n_bands=row["n_occ"], cache=row["dens_cache"],
                               refresh_sphere=True, return_hamiltonian=True)
    shape = tuple(int(x) for x in row["dens_shape"]);N = int(torch.tensor(shape).prod())
    rho = None
    for i in range(kin.shape[0]):
        V = pot[i] if pot.ndim == 3 else pot
        H = torch.diag(kin[i].to(V.dtype)) + V
        _, U = occupied_eigh(H, row["n_occ"], gap_floor)
        A = torch.zeros(row["n_occ"], N, dtype=U.dtype, device=U.device)
        A[:, row["dens_lin"]] = U[:, : row["n_occ"]].T
        psi = torch.fft.ifftn(A.reshape(-1, *shape), dim=(1, 2, 3)).reshape(row["n_occ"], -1)
        part = (psi.abs() ** 2).sum(0)
        rho = part if rho is None else rho + part
    volume = float(row["packed"].volume)
    return rho / rho.sum() * (2 * row["n_occ"]) * N / volume  # electrons per A^3 on the grid
