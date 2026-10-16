"""Material-specific upper bounds for local plane-wave potentials."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from epnet.hamiltonian.torch_pw import scatter_hermitian_potential, vbm_aligned_window_loss


def _radial_shell_index(q_vectors: Tensor, shell_tolerance: float) -> Tensor:
    """Assign deterministic radial shells using CPU float64 distances."""
    q_norm_cpu = torch.linalg.vector_norm(
        q_vectors.detach().cpu().to(torch.float64), dim=-1
    )
    order = torch.argsort(q_norm_cpu)
    sorted_q = q_norm_cpu[order]
    starts = torch.ones_like(sorted_q, dtype=torch.bool)
    if sorted_q.numel() > 1:
        starts[1:] = (sorted_q[1:] - sorted_q[:-1]) > float(shell_tolerance)
    sorted_shell = torch.cumsum(starts.to(torch.int64), dim=0) - 1
    shell_index_cpu = torch.empty_like(sorted_shell)
    shell_index_cpu[order] = sorted_shell
    return shell_index_cpu.to(q_vectors.device)


def _associated_legendre(l: int, m: int, z: Tensor) -> Tensor:
    """Condon-Shortley associated Legendre polynomial P_l^m(z)."""
    p_mm = torch.ones_like(z)
    if m:
        root = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
        factor = 1.0
        for _ in range(m):
            p_mm = -factor * root * p_mm
            factor += 2.0
    if l == m:
        return p_mm
    p_m1m = (2 * m + 1) * z * p_mm
    if l == m + 1:
        return p_m1m
    previous, current = p_mm, p_m1m
    for degree in range(m + 2, l + 1):
        following = (
            (2 * degree - 1) * z * current
            - (degree + m - 1) * previous
        ) / (degree - m)
        previous, current = current, following
    return current


def _real_spherical_harmonics(q_vectors: Tensor, l_max: int) -> tuple[Tensor, Tensor]:
    """Return normalized real harmonics for every 1 <= l <= l_max."""
    if l_max < 1:
        raise ValueError("l_max must be positive")
    q_cpu = q_vectors.detach().cpu().to(torch.float64)
    q_norm = torch.linalg.vector_norm(q_cpu, dim=-1)
    direction = q_cpu / q_norm.clamp_min(1e-15).unsqueeze(-1)
    z = direction[:, 2].clamp(-1.0, 1.0)
    phi = torch.atan2(direction[:, 1], direction[:, 0])
    columns = []
    degrees = []
    for l in range(1, int(l_max) + 1):
        p_l0 = _associated_legendre(l, 0, z)
        norm_l0 = math.sqrt((2 * l + 1) / (4.0 * math.pi))
        columns.append(norm_l0 * p_l0)
        degrees.append(l)
        for m in range(1, l + 1):
            p_lm = _associated_legendre(l, m, z)
            norm = math.sqrt(
                2.0
                * (2 * l + 1)
                / (4.0 * math.pi)
                * math.factorial(l - m)
                / math.factorial(l + m)
            )
            columns.append(norm * p_lm * torch.cos(m * phi))
            columns.append(norm * p_lm * torch.sin(m * phi))
            degrees.extend((l, l))
    basis = torch.stack(columns, dim=-1).to(q_vectors)
    return basis, torch.tensor(degrees, dtype=torch.int64, device=q_vectors.device)


def aligned_band_objective(
    prediction: Tensor,
    target: Tensor,
    n_occupied: int,
    *,
    gap_weight: float = 0.25,
    edge_weight: float = 5.0,
    signed_gap: bool = False,
) -> Tensor:
    """The gapped-cell objective used by the MP training pipeline.

    ``signed_gap=True`` keeps a useful gradient when the predicted indirect
    gap is negative.  The legacy default clamps metallic overlaps to zero.
    """
    loss = vbm_aligned_window_loss(
        prediction, target, n_occupied, edge_weight=float(edge_weight)
    )
    pred_gap = prediction[:, n_occupied].amin() - prediction[:, n_occupied - 1].amax()
    target_gap = target[:, n_occupied].amin() - target[:, n_occupied - 1].amax()
    if not signed_gap:
        pred_gap = torch.clamp(pred_gap, min=0.0)
        target_gap = torch.clamp(target_gap, min=0.0)
    return loss + float(gap_weight) * (pred_gap - target_gap).abs()


class HermitianLocalCorrection(nn.Module):
    """Unrestricted cell-periodic local correction on unique delta-G values.

    Independent complex coefficients are projected to a Hermitian Toeplitz
    matrix by the same assembly operation used in EpNet.  The G=0 gauge is
    removed explicitly.
    """

    def __init__(self, q_vectors: Tensor, *, init_std: float = 0.0) -> None:
        super().__init__()
        if q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
            raise ValueError("q_vectors must have shape (n_unique, 3)")
        self.real = nn.Parameter(torch.zeros(q_vectors.shape[0], dtype=q_vectors.dtype))
        self.imag = nn.Parameter(torch.zeros(q_vectors.shape[0], dtype=q_vectors.dtype))
        self.register_buffer("nonzero", torch.linalg.vector_norm(q_vectors, dim=-1) > 1e-8)
        if init_std > 0:
            with torch.no_grad():
                self.real.normal_(0.0, float(init_std))
                self.imag.normal_(0.0, float(init_std))

    def coefficients(self) -> Tensor:
        values = torch.complex(self.real, self.imag)
        return values * self.nonzero.to(values)

    def forward(self, inverse: Tensor, volume: float) -> Tensor:
        return scatter_hermitian_potential(self.coefficients(), inverse, volume)


class SpaceGroupProjectedHermitianLocalCorrection(HermitianLocalCorrection):
    """Unrestricted local coefficients projected to space-group and real-field symmetry."""

    def __init__(
        self,
        q_vectors: Tensor,
        symmetry_indices: Tensor,
        symmetry_phases: Tensor,
        *,
        init_std: float = 0.0,
        inversion_tolerance: float = 1e-5,
    ) -> None:
        super().__init__(q_vectors, init_std=init_std)
        indices = torch.as_tensor(symmetry_indices, dtype=torch.int64)
        phases = torch.as_tensor(symmetry_phases)
        expected = (indices.shape[0], q_vectors.shape[0])
        if indices.ndim != 2 or tuple(indices.shape) != expected:
            raise ValueError("symmetry_indices must have shape (n_operations, n_q)")
        if tuple(phases.shape) != expected or not torch.is_complex(phases):
            raise ValueError("symmetry_phases must be complex and match symmetry_indices")
        if int(indices.min()) < 0 or int(indices.max()) >= q_vectors.shape[0]:
            raise ValueError("symmetry_indices are outside the q-vector range")
        distances = torch.cdist(
            q_vectors.detach().cpu().to(torch.float64),
            -q_vectors.detach().cpu().to(torch.float64),
        )
        nearest_distance, conjugate_index = distances.min(dim=1)
        if float(nearest_distance.max()) > float(inversion_tolerance):
            raise ValueError("q_vectors must be closed under inversion")
        self.register_buffer("symmetry_indices", indices)
        self.register_buffer("symmetry_phases", phases.to(torch.complex64))
        self.register_buffer("conjugate_index", conjugate_index)

    def coefficients(self) -> Tensor:
        raw = super().coefficients()
        transformed = self.symmetry_phases.to(raw) * raw.unsqueeze(0)
        projected = torch.zeros_like(raw)
        projected.scatter_add_(
            0,
            self.symmetry_indices.reshape(-1),
            transformed.reshape(-1),
        )
        projected = projected / float(self.symmetry_indices.shape[0])
        hermitian = 0.5 * (projected + projected[self.conjugate_index].conj())
        return hermitian * self.nonzero.to(hermitian)


class SpeciesRadialShellCorrection(nn.Module):
    """Species-resolved real form factors sampled freely on radial q shells.

    The correction remains atom centered and radial. It removes the smooth
    basis parameterization by assigning one real value to every species and
    distinct |delta-G| shell present in the material's plane-wave basis.
    """

    def __init__(
        self,
        q_vectors: Tensor,
        positions: Tensor,
        species: Tensor,
        n_species: int,
        *,
        shell_tolerance: float = 1e-5,
        init_std: float = 0.0,
    ) -> None:
        super().__init__()
        if q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
            raise ValueError("q_vectors must have shape (n_unique, 3)")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3)")
        if species.ndim != 1 or species.shape[0] != positions.shape[0]:
            raise ValueError("species must have shape (n_atoms,)")
        if n_species <= 0:
            raise ValueError("n_species must be positive")
        if shell_tolerance <= 0:
            raise ValueError("shell_tolerance must be positive")
        # Sorted CPU float64 clustering avoids device-dependent rounding bins.
        shell_index = _radial_shell_index(q_vectors, shell_tolerance)
        q_norm = torch.linalg.vector_norm(q_vectors, dim=-1)
        phase = torch.exp(
            -1j * torch.einsum("gc,nc->gn", q_vectors.to(positions), positions)
        )
        one_hot = F.one_hot(species.to(torch.int64), num_classes=int(n_species))
        structure_factors = phase @ one_hot.to(phase.dtype)
        self.values = nn.Parameter(
            torch.zeros(int(n_species), int(shell_index.max()) + 1, dtype=q_vectors.dtype)
        )
        self.register_buffer("shell_index", shell_index)
        self.register_buffer("structure_factors", structure_factors)
        self.register_buffer("nonzero", q_norm > 1e-8)
        if init_std > 0:
            with torch.no_grad():
                self.values.normal_(0.0, float(init_std))

    @property
    def n_shells(self) -> int:
        return int(self.values.shape[1])

    def coefficients(self) -> Tensor:
        per_q_species = self.values[:, self.shell_index].transpose(0, 1)
        total = (per_q_species.to(self.structure_factors) * self.structure_factors).sum(dim=-1)
        return total * self.nonzero.to(total)

    def forward(self, inverse: Tensor, volume: float) -> Tensor:
        return scatter_hermitian_potential(self.coefficients(), inverse, volume)


class AngularShellCorrection(nn.Module):
    r"""Cell-frame angular correction resolved independently on each radial shell.

    The deterministic radial parent supplies l=0. This module learns l>=1 real
    spherical-harmonic amplitudes. Even-l terms are real and odd-l terms are
    purely imaginary, which enforces V(-q)=V(q)* before matrix assembly.
    """

    def __init__(
        self,
        q_vectors: Tensor,
        l_max: int,
        *,
        shell_tolerance: float = 1e-5,
        init_std: float = 0.0,
    ) -> None:
        super().__init__()
        if q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
            raise ValueError("q_vectors must have shape (n_unique, 3)")
        if l_max < 1:
            raise ValueError("l_max must be positive")
        if shell_tolerance <= 0:
            raise ValueError("shell_tolerance must be positive")
        shell_index = _radial_shell_index(q_vectors, shell_tolerance)
        angular_basis, degrees = _real_spherical_harmonics(q_vectors, int(l_max))
        q_norm = torch.linalg.vector_norm(q_vectors, dim=-1)
        self.l_max = int(l_max)
        self.values = nn.Parameter(
            torch.zeros(
                int(shell_index.max()) + 1,
                int(angular_basis.shape[1]),
                dtype=q_vectors.dtype,
                device=q_vectors.device,
            )
        )
        self.register_buffer("shell_index", shell_index)
        self.register_buffer("angular_basis", angular_basis)
        self.register_buffer("degrees", degrees)
        self.register_buffer("nonzero", q_norm > 1e-8)
        if init_std > 0:
            with torch.no_grad():
                self.values.normal_(0.0, float(init_std))

    @property
    def n_shells(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_angular(self) -> int:
        return int(self.values.shape[1])

    def coefficients(self) -> Tensor:
        amplitudes = self.values[self.shell_index] * self.angular_basis
        real = amplitudes[:, self.degrees.remainder(2) == 0].sum(dim=-1)
        imag = amplitudes[:, self.degrees.remainder(2) == 1].sum(dim=-1)
        total = torch.complex(real, imag)
        return total * self.nonzero.to(total)

    def forward(self, inverse: Tensor, volume: float) -> Tensor:
        return scatter_hermitian_potential(self.coefficients(), inverse, volume)


class CellMultipoleShellCorrection(nn.Module):
    r"""Full cell-frame local potential through ``l_max`` on radial shells.

    Unlike :class:`AngularShellCorrection`, this module includes the l=0
    channel.  It is therefore suitable as a material-specific teacher on top
    of a fixed universal parent: every radial shell receives one real scalar
    coefficient plus the angular coefficients.  Even-l amplitudes are real
    and odd-l amplitudes are imaginary, enforcing ``V(-q) = V(q)*``.
    """

    def __init__(
        self,
        q_vectors: Tensor,
        l_max: int,
        *,
        shell_tolerance: float = 1e-5,
        init_std: float = 0.0,
    ) -> None:
        super().__init__()
        if q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
            raise ValueError("q_vectors must have shape (n_unique, 3)")
        if l_max < 0:
            raise ValueError("l_max must be non-negative")
        if shell_tolerance <= 0:
            raise ValueError("shell_tolerance must be positive")
        shell_index = _radial_shell_index(q_vectors, shell_tolerance)
        l0 = torch.full(
            (q_vectors.shape[0], 1),
            1.0 / math.sqrt(4.0 * math.pi),
            dtype=q_vectors.dtype,
            device=q_vectors.device,
        )
        if l_max:
            higher_basis, higher_degrees = _real_spherical_harmonics(
                q_vectors, int(l_max)
            )
            angular_basis = torch.cat((l0, higher_basis), dim=-1)
            degrees = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=q_vectors.device),
                    higher_degrees,
                )
            )
        else:
            angular_basis = l0
            degrees = torch.zeros(1, dtype=torch.int64, device=q_vectors.device)
        q_norm = torch.linalg.vector_norm(q_vectors, dim=-1)
        self.l_max = int(l_max)
        self.values = nn.Parameter(
            torch.zeros(
                int(shell_index.max()) + 1,
                int(angular_basis.shape[1]),
                dtype=q_vectors.dtype,
                device=q_vectors.device,
            )
        )
        self.register_buffer("shell_index", shell_index)
        self.register_buffer("angular_basis", angular_basis)
        self.register_buffer("degrees", degrees)
        self.register_buffer("nonzero", q_norm > 1e-8)
        if init_std > 0:
            with torch.no_grad():
                self.values.normal_(0.0, float(init_std))

    @property
    def n_shells(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_angular(self) -> int:
        return int(self.values.shape[1])

    def coefficients(self) -> Tensor:
        amplitudes = self.values[self.shell_index] * self.angular_basis
        real = amplitudes[:, self.degrees.remainder(2) == 0].sum(dim=-1)
        imag = amplitudes[:, self.degrees.remainder(2) == 1].sum(dim=-1)
        total = torch.complex(real, imag)
        return total * self.nonzero.to(total)

    def forward(self, inverse: Tensor, volume: float) -> Tensor:
        return scatter_hermitian_potential(self.coefficients(), inverse, volume)


class SpaceGroupProjectedCellMultipoleShellCorrection(CellMultipoleShellCorrection):
    """Cell multipoles projected onto a precomputed scalar space-group action."""

    def __init__(
        self,
        q_vectors: Tensor,
        l_max: int,
        symmetry_indices: Tensor,
        symmetry_phases: Tensor,
        *,
        shell_tolerance: float = 1e-5,
        init_std: float = 0.0,
    ) -> None:
        super().__init__(
            q_vectors,
            l_max,
            shell_tolerance=shell_tolerance,
            init_std=init_std,
        )
        indices = torch.as_tensor(symmetry_indices, dtype=torch.int64)
        phases = torch.as_tensor(symmetry_phases)
        expected = (indices.shape[0], q_vectors.shape[0])
        if indices.ndim != 2 or tuple(indices.shape) != expected:
            raise ValueError("symmetry_indices must have shape (n_operations, n_q)")
        if tuple(phases.shape) != expected or not torch.is_complex(phases):
            raise ValueError("symmetry_phases must be complex and match symmetry_indices")
        if int(indices.min()) < 0 or int(indices.max()) >= q_vectors.shape[0]:
            raise ValueError("symmetry_indices are outside the q-vector range")
        self.register_buffer("symmetry_indices", indices)
        self.register_buffer("symmetry_phases", phases.to(torch.complex64))

    def coefficients(self) -> Tensor:
        raw = super().coefficients()
        transformed = self.symmetry_phases.to(raw) * raw.unsqueeze(0)
        projected = torch.zeros_like(raw)
        projected.scatter_add_(
            0,
            self.symmetry_indices.reshape(-1),
            transformed.reshape(-1),
        )
        return projected / float(self.symmetry_indices.shape[0])
