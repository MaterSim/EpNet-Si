"""Invariant atom-environment corrections to a radial empirical pseudopotential."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from epnet.hamiltonian.torch_pw import batched_band_energies, scatter_hermitian_potential
from epnet.models.baselines.radial import CosineRadialEP
from epnet.models.equivariant_encoder import (
    SmoothCompactRadialBasis,
    build_periodic_neighbor_graph,
    periodic_displacements,
    smooth_coordination,
)


PHASE16_N_SPECIES = 45
PHASE16_DESCRIPTOR_RADIAL = 12
PHASE16_DESCRIPTOR_CUTOFF = 6.0
PHASE16_COORDINATION_CUTOFFS = (3.0, 5.0, 8.0)
PHASE16_DESCRIPTOR_WIDTH = 589
PHASE16_PARENT_BASIS = 24
PHASE16_ENV_PARAMETERS = 85336


def species_resolved_environment_descriptor(
    positions: Tensor,
    lattice: Tensor,
    species: Tensor,
    *,
    n_species: int = PHASE16_N_SPECIES,
    n_radial: int = PHASE16_DESCRIPTOR_RADIAL,
    cutoff: float = PHASE16_DESCRIPTOR_CUTOFF,
    coordination_cutoffs: tuple[float, ...] = PHASE16_COORDINATION_CUTOFFS,
    edge_index: Tensor | None = None,
    cell_shifts: Tensor | None = None,
) -> Tensor:
    """Return the frozen Phase 16 per-atom invariant descriptor.

    Channels are central-species one-hot, species-resolved radial neighbor
    histograms, log(volume / atom), and three smooth coordination counts.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if lattice.shape != (3, 3):
        raise ValueError("lattice must have shape (3, 3)")
    if species.ndim != 1 or len(species) != len(positions):
        raise ValueError("species must have shape (N,)")
    if not len(species):
        raise ValueError("descriptor requires at least one atom")
    species = species.to(device=positions.device, dtype=torch.long)
    if int(species.min()) < 0 or int(species.max()) >= int(n_species):
        raise ValueError("species index is outside the descriptor table")
    graph_cutoff = max(float(cutoff), *(float(value) for value in coordination_cutoffs))
    if edge_index is None:
        edge_index, cell_shifts = build_periodic_neighbor_graph(
            positions, lattice, graph_cutoff
        )
    if cell_shifts is None:
        raise ValueError("periodic descriptor requires cell shifts")
    displacement = periodic_displacements(positions, edge_index, lattice, cell_shifts)
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    radial = SmoothCompactRadialBasis(int(n_radial), float(cutoff)).to(
        device=positions.device, dtype=positions.dtype
    )(distance)
    centers = edge_index[0].to(device=positions.device, dtype=torch.long)
    neighbors = edge_index[1].to(device=positions.device, dtype=torch.long)
    neighbor_species = species.index_select(0, neighbors)
    flat_groups = centers * int(n_species) + neighbor_species
    histogram = positions.new_zeros(len(positions) * int(n_species), int(n_radial))
    histogram.index_add_(0, flat_groups, radial)
    histogram = histogram.reshape(len(positions), int(n_species) * int(n_radial))

    central = F.one_hot(species, num_classes=int(n_species)).to(dtype=positions.dtype)
    volume = torch.linalg.det(lattice.to(positions)).abs().clamp_min(
        torch.finfo(positions.dtype).tiny
    )
    log_volume_per_atom = torch.log(volume / float(len(positions))).expand(
        len(positions), 1
    )
    coordination = smooth_coordination(
        positions,
        lattice,
        coordination_cutoffs,
        edge_index=edge_index,
        cell_shifts=cell_shifts,
    )
    descriptor = torch.cat(
        (central, histogram, log_volume_per_atom, coordination), dim=-1
    )
    expected = int(n_species) + int(n_species) * int(n_radial) + 1 + len(
        coordination_cutoffs
    )
    if descriptor.shape != (len(positions), expected):
        raise AssertionError("environment descriptor width is inconsistent")
    return descriptor


def descriptor_normalization(descriptors: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Compute one atom-weighted fit-only normalization."""
    if not descriptors:
        raise ValueError("at least one descriptor tensor is required")
    width = int(descriptors[0].shape[-1])
    if any(value.ndim != 2 or value.shape[-1] != width for value in descriptors):
        raise ValueError("descriptor tensors have inconsistent shapes")
    joined = torch.cat([value.detach().to(dtype=torch.float64, device="cpu") for value in descriptors])
    mean = joined.mean(dim=0)
    scale = torch.sqrt(torch.mean((joined - mean).square(), dim=0)).clamp_min(1e-6)
    return mean.float(), scale.float()


class EnvironmentCoefficientMLP(nn.Module):
    """Phase 16 MLP mapping invariant atomic environments to radial coefficients."""

    def __init__(
        self,
        descriptor_width: int = PHASE16_DESCRIPTOR_WIDTH,
        hidden1: int = 128,
        hidden2: int = 64,
        n_basis: int = PHASE16_PARENT_BASIS,
    ) -> None:
        super().__init__()
        self.descriptor_width = int(descriptor_width)
        self.n_basis = int(n_basis)
        self.net = nn.Sequential(
            nn.Linear(self.descriptor_width, int(hidden1)),
            nn.SiLU(),
            nn.Linear(int(hidden1), int(hidden2)),
            nn.SiLU(),
            nn.Linear(int(hidden2), self.n_basis),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, descriptor: Tensor) -> Tensor:
        if descriptor.ndim != 2 or descriptor.shape[-1] != self.descriptor_width:
            raise ValueError(
                f"descriptor must have shape (N, {self.descriptor_width})"
            )
        return self.net(descriptor)


class EnvironmentConditionedRadialParent(nn.Module):
    """Apply an invariant per-atom radial correction before PW diagonalization."""

    def __init__(
        self,
        sphere: CosineRadialEP,
        descriptor_mean: Tensor | None = None,
        descriptor_scale: Tensor | None = None,
    ) -> None:
        super().__init__()
        if sphere.n_species != PHASE16_N_SPECIES:
            raise ValueError("Phase 16 requires a 45-species radial parent")
        if sphere.n_basis != PHASE16_PARENT_BASIS:
            raise ValueError("Phase 16 must preserve the 24-function parent basis")
        self.sphere = sphere
        self.sphere.coefficients.requires_grad_(False)
        self.environment = EnvironmentCoefficientMLP(n_basis=sphere.n_basis)
        mean = (
            torch.zeros(PHASE16_DESCRIPTOR_WIDTH)
            if descriptor_mean is None
            else torch.as_tensor(descriptor_mean).detach().float().clone()
        )
        scale = (
            torch.ones(PHASE16_DESCRIPTOR_WIDTH)
            if descriptor_scale is None
            else torch.as_tensor(descriptor_scale).detach().float().clone()
        )
        if mean.shape != (PHASE16_DESCRIPTOR_WIDTH,) or scale.shape != mean.shape:
            raise ValueError("invalid Phase 16 descriptor normalization")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("descriptor normalization must be finite and positive")
        self.register_buffer("descriptor_mean", mean)
        self.register_buffer("descriptor_scale", scale)
        if sum(parameter.numel() for parameter in self.environment.parameters()) != PHASE16_ENV_PARAMETERS:
            raise AssertionError("Phase 16 environment parameter count changed")

    def normalized_descriptor(self, descriptor: Tensor) -> Tensor:
        return (descriptor - self.descriptor_mean.to(descriptor)) / self.descriptor_scale.to(
            descriptor
        )

    def coefficients(self, descriptor: Tensor) -> Tensor:
        return self.environment(self.normalized_descriptor(descriptor))

    def environment_form_factor(
        self,
        descriptor: Tensor,
        positions: Tensor,
        unique_cart: Tensor,
    ) -> Tensor:
        coefficients = self.coefficients(descriptor)
        q = torch.linalg.vector_norm(unique_cart, dim=-1)
        basis = self.sphere.basis(q)
        per_atom = torch.einsum("nb,qb->nq", coefficients, basis)
        phase = torch.exp(
            -1j
            * torch.einsum(
                "qa,na->qn", unique_cart.to(device=positions.device), positions
            )
        )
        total = (per_atom.to(phase).transpose(0, 1) * phase).sum(dim=1)
        return total.masked_fill(q < 1e-8, 0)

    def potential(
        self,
        base_model: nn.Module,
        packed: SimpleNamespace,
        cache: SimpleNamespace,
        descriptor: Tensor,
    ) -> Tensor:
        base = base_model.spherical_potential(cache)
        correction = self.environment_form_factor(
            descriptor, packed.positions, cache.unique_cart
        )
        return base + scatter_hermitian_potential(
            correction, cache.inverse, cache.volume
        )

    def band_path(
        self,
        base_model: nn.Module,
        packed: SimpleNamespace,
        kpoints: Tensor,
        n_bands: int,
        *,
        descriptor: Tensor | None = None,
        cache: SimpleNamespace | None = None,
    ) -> Tensor:
        if descriptor is None:
            descriptor = species_resolved_environment_descriptor(
                packed.positions, packed.lattice, packed.species
            )
        if cache is None:
            cache = base_model.make_pw_cache(
                packed, kpoints, store_spherical=False, store_graph=False
            )
        potential = self.potential(base_model, packed, cache, descriptor)
        return batched_band_energies(
            cache.kinetic,
            potential,
            n_bands=int(n_bands),
            k_batch=base_model.k_batch,
            subset_ng_min=base_model.subset_ng_min,
        )
