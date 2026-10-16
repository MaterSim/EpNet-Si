"""Periodic reciprocal-space models for direct ordered-band prediction."""

from __future__ import annotations

import itertools

import torch
from torch import Tensor, nn

from epnet.models.baselines.descriptor_mlp import periodic_radial_descriptor
from epnet.models.equivariant_encoder import (
    SmoothCompactRadialBasis,
    build_periodic_neighbor_graph,
    periodic_displacements,
)
from epnet.models.element_encoder import PeriodicTableEmbedding


def canonical_integer_frequencies(order: int) -> Tensor:
    """Return one representative from every nonzero integer +/- pair."""
    if order < 1:
        raise ValueError("order must be positive")
    frequencies = []
    for vector in itertools.product(range(-order, order + 1), repeat=3):
        if vector == (0, 0, 0):
            continue
        first_nonzero = next(value for value in vector if value != 0)
        if first_nonzero > 0:
            frequencies.append(vector)
    return torch.tensor(frequencies, dtype=torch.float32)


class PeriodicBandResidualMLP(nn.Module):
    """Predict ordered band residuals from periodic k features and a parent."""

    def __init__(
        self,
        n_bands: int,
        *,
        harmonic_order: int = 3,
        width: int = 256,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if n_bands < 2 or width < 1 or depth < 1:
            raise ValueError("n_bands, width, and depth must be positive")
        self.n_bands = int(n_bands)
        frequencies = canonical_integer_frequencies(int(harmonic_order))
        self.register_buffer("frequencies", frequencies)
        input_width = 2 * int(frequencies.shape[0]) + self.n_bands
        layers: list[nn.Module] = [nn.Linear(input_width, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.SiLU()))
        output = nn.Linear(width, self.n_bands)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def periodic_features(self, k_frac: Tensor) -> Tensor:
        k_frac = torch.remainder(k_frac, 1.0)
        phase = 2.0 * torch.pi * (
            k_frac.to(self.frequencies) @ self.frequencies.transpose(0, 1)
        )
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)

    def forward(self, k_frac: Tensor, parent_bands: Tensor) -> Tensor:
        if parent_bands.shape[-1] != self.n_bands:
            raise ValueError("parent band count differs from model output")
        if k_frac.shape[:-1] != parent_bands.shape[:-1]:
            raise ValueError("k-point and parent leading dimensions differ")
        features = torch.cat(
            (
                self.periodic_features(k_frac).to(parent_bands),
                parent_bands,
            ),
            dim=-1,
        )
        corrected = parent_bands + self.network(features)
        return torch.sort(corrected, dim=-1).values


def invariant_structure_descriptor(
    lattice: Tensor,
    fractional_positions: Tensor,
    species: Tensor,
    *,
    n_species: int,
    n_radial: int = 8,
    cutoff: float = 5.0,
) -> Tensor:
    """Composition, pair-distance, and cell-shape invariants for one cell."""
    per_atom = periodic_radial_descriptor(
        lattice,
        fractional_positions,
        species,
        n_species=n_species,
        n_radial=n_radial,
        cutoff=cutoff,
    )
    composition = torch.nn.functional.one_hot(
        species.to(torch.long),
        n_species,
    ).to(lattice).mean(dim=0)
    volume_per_atom = torch.linalg.det(lattice).abs() / len(species)
    length_scale = volume_per_atom.clamp_min(1e-8).pow(1.0 / 3.0)
    principal_lengths = torch.linalg.eigvalsh(lattice @ lattice.T).clamp_min(1e-8).sqrt()
    cell_invariants = torch.cat(
        (
            principal_lengths / length_scale,
            torch.log(volume_per_atom.clamp_min(1e-8)).reshape(1),
            lattice.new_tensor([len(species) / 4.0]),
        )
    )
    return torch.cat((composition, per_atom.mean(dim=0), cell_invariants))


def periodic_table_descriptor_width(
    n_species: int, n_radial: int, feature_width: int
) -> int:
    if min(n_species, n_radial, feature_width) < 1:
        raise ValueError("descriptor dimensions must be positive")
    return n_species + n_species * n_radial + feature_width * (1 + n_radial) + 5


def periodic_table_structure_descriptor(
    lattice: Tensor,
    fractional_positions: Tensor,
    species: Tensor,
    element_features: Tensor,
    *,
    n_species: int,
    n_radial: int = 16,
    cutoff: float = 5.0,
) -> Tensor:
    """Augment categorical structure invariants with elemental feature moments."""
    if element_features.ndim != 2 or element_features.shape[0] != n_species:
        raise ValueError("element feature table differs from the species domain")
    if not torch.isfinite(element_features).all():
        raise ValueError("element features must be finite")
    legacy = invariant_structure_descriptor(
        lattice,
        fractional_positions,
        species,
        n_species=n_species,
        n_radial=n_radial,
        cutoff=cutoff,
    )
    feature_table = element_features.to(legacy)
    composition = legacy[:n_species]
    radial = legacy[n_species : n_species * (n_radial + 1)].reshape(
        n_species, n_radial
    )
    feature_composition = composition @ feature_table
    feature_radial = torch.einsum("sr,sf->fr", radial, feature_table).reshape(-1)
    descriptor = torch.cat(
        (
            composition,
            radial.reshape(-1),
            feature_composition,
            feature_radial,
            legacy[-5:],
        )
    )
    expected = periodic_table_descriptor_width(
        n_species, n_radial, int(feature_table.shape[1])
    )
    if descriptor.numel() != expected:
        raise RuntimeError("periodic-table descriptor width is inconsistent")
    return descriptor


class StructureConditionedBandResidualMLP(PeriodicBandResidualMLP):
    """Shared periodic band decoder conditioned on invariant cell features."""

    def __init__(
        self,
        n_bands: int,
        structure_width: int,
        *,
        harmonic_order: int = 2,
        width: int = 256,
        depth: int = 3,
        context_width: int = 128,
    ) -> None:
        nn.Module.__init__(self)
        if structure_width < 1 or context_width < 1:
            raise ValueError("structure and context widths must be positive")
        self.n_bands = int(n_bands)
        frequencies = canonical_integer_frequencies(int(harmonic_order))
        self.register_buffer("frequencies", frequencies)
        self.structure_encoder = nn.Sequential(
            nn.Linear(structure_width, width),
            nn.SiLU(),
            nn.Linear(width, context_width),
            nn.LayerNorm(context_width),
        )
        input_width = (
            2 * int(frequencies.shape[0])
            + self.n_bands
            + int(context_width)
        )
        layers: list[nn.Module] = [nn.Linear(input_width, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.SiLU()))
        output = nn.Linear(width, self.n_bands)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        k_frac: Tensor,
        parent_bands: Tensor,
        structure_descriptor: Tensor,
    ) -> Tensor:
        if structure_descriptor.ndim == 1:
            structure_descriptor = structure_descriptor.unsqueeze(0)
        if structure_descriptor.shape[0] == 1:
            structure_descriptor = structure_descriptor.expand(
                len(k_frac),
                -1,
            )
        if structure_descriptor.shape[0] != len(k_frac):
            raise ValueError("structure descriptor must be per-cell or per-k-point")
        context = self.structure_encoder(
            structure_descriptor.to(parent_bands)
        )
        features = torch.cat(
            (
                self.periodic_features(k_frac).to(parent_bands),
                parent_bands,
                context,
            ),
            dim=-1,
        )
        corrected = parent_bands + self.network(features)
        return torch.sort(corrected, dim=-1).values


class MultiHeadStructureBandResidualMLP(PeriodicBandResidualMLP):
    """Separate band-edge shifts from within-manifold dispersion residuals."""

    def __init__(
        self,
        n_bands: int,
        n_valence: int,
        structure_width: int,
        *,
        harmonic_order: int = 2,
        width: int = 256,
        depth: int = 3,
        context_width: int = 128,
    ) -> None:
        nn.Module.__init__(self)
        if not 1 <= n_valence < n_bands:
            raise ValueError("n_valence must lie strictly inside the band window")
        if structure_width < 1 or width < 1 or depth < 1 or context_width < 1:
            raise ValueError("model dimensions must be positive")
        self.n_bands = int(n_bands)
        self.n_valence = int(n_valence)
        frequencies = canonical_integer_frequencies(int(harmonic_order))
        self.register_buffer("frequencies", frequencies)
        self.structure_encoder = nn.Sequential(
            nn.Linear(structure_width, width),
            nn.SiLU(),
            nn.Linear(width, context_width),
            nn.LayerNorm(context_width),
        )
        input_width = (
            2 * int(frequencies.shape[0])
            + self.n_bands
            + int(context_width)
        )
        trunk: list[nn.Module] = [nn.Linear(input_width, width), nn.SiLU()]
        for _ in range(depth - 1):
            trunk.extend((nn.Linear(width, width), nn.SiLU()))
        self.trunk = nn.Sequential(*trunk)
        self.edge_head = nn.Linear(width, 2)
        self.dispersion_head = nn.Linear(width, self.n_bands)
        for output in (self.edge_head, self.dispersion_head):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def _features(
        self,
        k_frac: Tensor,
        parent_bands: Tensor,
        structure_descriptor: Tensor,
    ) -> Tensor:
        if parent_bands.shape[-1] != self.n_bands:
            raise ValueError("parent band count differs from model output")
        if k_frac.shape[:-1] != parent_bands.shape[:-1]:
            raise ValueError("k-point and parent leading dimensions differ")
        if structure_descriptor.ndim == 1:
            structure_descriptor = structure_descriptor.unsqueeze(0)
        if structure_descriptor.shape[0] == 1:
            structure_descriptor = structure_descriptor.expand(len(k_frac), -1)
        if structure_descriptor.shape[0] != len(k_frac):
            raise ValueError("structure descriptor must be per-cell or per-k-point")
        context = self.structure_encoder(structure_descriptor.to(parent_bands))
        return torch.cat(
            (
                self.periodic_features(k_frac).to(parent_bands),
                parent_bands,
                context,
            ),
            dim=-1,
        )

    def residual_components(
        self,
        k_frac: Tensor,
        parent_bands: Tensor,
        structure_descriptor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return two edge shifts and group-centered dispersion residuals."""
        hidden = self.trunk(
            self._features(k_frac, parent_bands, structure_descriptor)
        )
        edge_shifts = self.edge_head(hidden)
        raw_dispersion = self.dispersion_head(hidden)
        valence = raw_dispersion[..., : self.n_valence]
        conduction = raw_dispersion[..., self.n_valence :]
        dispersion = torch.cat(
            (
                valence - valence.mean(dim=-1, keepdim=True),
                conduction - conduction.mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        return edge_shifts, dispersion

    def forward(
        self,
        k_frac: Tensor,
        parent_bands: Tensor,
        structure_descriptor: Tensor,
    ) -> Tensor:
        edge_shifts, dispersion = self.residual_components(
            k_frac,
            parent_bands,
            structure_descriptor,
        )
        valence = (
            parent_bands[..., : self.n_valence]
            + dispersion[..., : self.n_valence]
            + edge_shifts[..., :1]
        )
        conduction = (
            parent_bands[..., self.n_valence :]
            + dispersion[..., self.n_valence :]
            + edge_shifts[..., 1:]
        )
        return torch.cat(
            (
                torch.sort(valence, dim=-1).values,
                torch.sort(conduction, dim=-1).values,
            ),
            dim=-1,
        )


class InvariantMessagePassingEncoder(nn.Module):
    """Learn a periodic E(3)-invariant context for one crystal structure.

    Atom states exchange messages through species embeddings and compact
    radial distance features.  Because the graph and messages depend only on
    periodic pair distances, the pooled context is invariant to rigid E(3)
    transformations and atom permutations.
    """

    def __init__(
        self,
        num_species: int,
        *,
        channels: int = 64,
        message_layers: int = 2,
        num_radial: int = 8,
        cutoff: float = 5.0,
        context_width: int = 128,
        element_features: Tensor | None = None,
        element_hidden: int = 32,
    ) -> None:
        super().__init__()
        if (
            num_species < 1
            or channels < 1
            or message_layers < 1
            or num_radial < 1
            or cutoff <= 0
            or context_width < 1
        ):
            raise ValueError("encoder dimensions and cutoff must be positive")
        self.num_species = int(num_species)
        self.channels = int(channels)
        self.cutoff = float(cutoff)
        if element_features is None:
            self.species_embedding: nn.Module = nn.Embedding(num_species, channels)
            self.element_feature_width = 0
        else:
            if element_features.ndim != 2 or element_features.shape[0] != num_species:
                raise ValueError("element feature table differs from num_species")
            self.species_embedding = PeriodicTableEmbedding(
                element_features,
                channels,
                hidden=element_hidden,
            )
            self.element_feature_width = int(element_features.shape[1])
        self.radial_basis = SmoothCompactRadialBasis(num_radial, cutoff)
        self.radial_projection = nn.Linear(num_radial, channels)
        self.message_networks = nn.ModuleList()
        self.update_networks = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        for _ in range(message_layers):
            self.message_networks.append(
                nn.Sequential(
                    nn.Linear(3 * channels, 2 * channels),
                    nn.SiLU(),
                    nn.Linear(2 * channels, channels),
                )
            )
            self.update_networks.append(
                nn.Sequential(
                    nn.Linear(2 * channels, 2 * channels),
                    nn.SiLU(),
                    nn.Linear(2 * channels, channels),
                )
            )
            self.normalizations.append(nn.LayerNorm(channels))
        pooled_width = 2 * channels + num_species + self.element_feature_width + 5
        self.readout = nn.Sequential(
            nn.Linear(pooled_width, 2 * context_width),
            nn.SiLU(),
            nn.Linear(2 * context_width, context_width),
            nn.LayerNorm(context_width),
        )

    def _cell_invariants(self, cell: Tensor, n_atoms: int) -> Tensor:
        volume_per_atom = torch.linalg.det(cell).abs() / n_atoms
        length_scale = volume_per_atom.clamp_min(1e-8).pow(1.0 / 3.0)
        lengths = torch.linalg.eigvalsh(cell @ cell.T).clamp_min(1e-8).sqrt()
        return torch.cat(
            (
                lengths / length_scale,
                torch.log(volume_per_atom.clamp_min(1e-8)).reshape(1),
                cell.new_tensor([n_atoms / 4.0]),
            )
        )

    def forward(
        self,
        positions: Tensor,
        species: Tensor,
        cell: Tensor,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> Tensor:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if cell.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")
        species = species.to(device=positions.device, dtype=torch.long)
        if species.shape != (positions.shape[0],):
            raise ValueError("species must have shape (N,)")
        if edge_index is None:
            edge_index, cell_shifts = build_periodic_neighbor_graph(
                positions,
                cell,
                self.cutoff,
            )
        edge_index = edge_index.to(device=positions.device, dtype=torch.long)
        displacement = periodic_displacements(
            positions,
            edge_index,
            cell,
            cell_shifts,
        )
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        valid = (distance > torch.finfo(distance.dtype).eps) & (
            distance < self.cutoff
        )
        keep = torch.nonzero(valid, as_tuple=False).reshape(-1)
        edge_index = edge_index.index_select(1, keep)
        distance = distance.index_select(0, keep)
        center, neighbor = edge_index

        state = self.species_embedding(species)
        radial = self.radial_projection(self.radial_basis(distance))
        n_atoms = positions.shape[0]
        for message_network, update_network, normalization in zip(
            self.message_networks,
            self.update_networks,
            self.normalizations,
        ):
            message = message_network(
                torch.cat(
                    (
                        state.index_select(0, center),
                        state.index_select(0, neighbor),
                        radial,
                    ),
                    dim=-1,
                )
            )
            aggregate = state.new_zeros(n_atoms, self.channels)
            aggregate.index_add_(0, center, message)
            degree = state.new_zeros(n_atoms)
            degree.index_add_(0, center, torch.ones_like(center, dtype=state.dtype))
            aggregate = aggregate / degree.clamp_min(1.0).sqrt().unsqueeze(-1)
            update = update_network(torch.cat((state, aggregate), dim=-1))
            state = normalization(state + update)

        mean = state.mean(dim=0)
        dispersion = torch.sqrt(((state - mean) ** 2).mean(dim=0) + 1e-12)
        composition = torch.nn.functional.one_hot(
            species,
            self.num_species,
        ).to(state).mean(dim=0)
        pooled_parts = [mean, dispersion, composition]
        if self.element_feature_width:
            if not isinstance(self.species_embedding, PeriodicTableEmbedding):
                raise RuntimeError("element feature encoder is inconsistent")
            pooled_parts.append(
                composition @ self.species_embedding.element_features.to(state)
            )
        pooled_parts.append(self._cell_invariants(cell.to(state), n_atoms))
        pooled = torch.cat(tuple(pooled_parts))
        return self.readout(pooled)


class MessagePassingBandResidualMLP(PeriodicBandResidualMLP):
    """Shared periodic band decoder with a learned crystal encoder."""

    def __init__(
        self,
        n_bands: int,
        num_species: int,
        *,
        harmonic_order: int = 2,
        width: int = 256,
        depth: int = 3,
        context_width: int = 128,
        encoder_channels: int = 64,
        message_layers: int = 2,
        num_radial: int = 8,
        cutoff: float = 5.0,
    ) -> None:
        nn.Module.__init__(self)
        if n_bands < 2 or width < 1 or depth < 1:
            raise ValueError("n_bands, width, and depth must be positive")
        self.n_bands = int(n_bands)
        frequencies = canonical_integer_frequencies(int(harmonic_order))
        self.register_buffer("frequencies", frequencies)
        self.structure_encoder = InvariantMessagePassingEncoder(
            num_species,
            channels=encoder_channels,
            message_layers=message_layers,
            num_radial=num_radial,
            cutoff=cutoff,
            context_width=context_width,
        )
        input_width = (
            2 * int(frequencies.shape[0])
            + self.n_bands
            + int(context_width)
        )
        layers: list[nn.Module] = [nn.Linear(input_width, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.SiLU()))
        output = nn.Linear(width, self.n_bands)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        k_frac: Tensor,
        parent_bands: Tensor,
        positions: Tensor,
        species: Tensor,
        cell: Tensor,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> Tensor:
        context = self.structure_encoder(
            positions,
            species,
            cell,
            edge_index,
            cell_shifts,
        ).unsqueeze(0).expand(len(k_frac), -1)
        features = torch.cat(
            (
                self.periodic_features(k_frac).to(parent_bands),
                parent_bands,
                context.to(parent_bands),
            ),
            dim=-1,
        )
        corrected = parent_bands + self.network(features)
        return torch.sort(corrected, dim=-1).values


class MultiHeadMessagePassingBandResidualMLP(PeriodicBandResidualMLP):
    """Multi-head band decoder conditioned on a learned crystal encoder."""

    def __init__(
        self,
        n_bands: int,
        n_valence: int,
        num_species: int,
        *,
        harmonic_order: int = 2,
        width: int = 256,
        depth: int = 3,
        context_width: int = 128,
        encoder_channels: int = 64,
        message_layers: int = 2,
        num_radial: int = 16,
        cutoff: float = 5.0,
    ) -> None:
        nn.Module.__init__(self)
        if not 1 <= n_valence < n_bands:
            raise ValueError("n_valence must lie strictly inside the band window")
        if width < 1 or depth < 1 or context_width < 1:
            raise ValueError("model dimensions must be positive")
        self.n_bands = int(n_bands)
        self.n_valence = int(n_valence)
        frequencies = canonical_integer_frequencies(int(harmonic_order))
        self.register_buffer("frequencies", frequencies)
        self.structure_encoder = InvariantMessagePassingEncoder(
            num_species,
            channels=encoder_channels,
            message_layers=message_layers,
            num_radial=num_radial,
            cutoff=cutoff,
            context_width=context_width,
        )
        input_width = (
            2 * int(frequencies.shape[0])
            + self.n_bands
            + int(context_width)
        )
        trunk: list[nn.Module] = [nn.Linear(input_width, width), nn.SiLU()]
        for _ in range(depth - 1):
            trunk.extend((nn.Linear(width, width), nn.SiLU()))
        self.trunk = nn.Sequential(*trunk)
        self.edge_head = nn.Linear(width, 2)
        self.dispersion_head = nn.Linear(width, self.n_bands)
        for output in (self.edge_head, self.dispersion_head):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def residual_components(
        self,
        k_frac: Tensor,
        parent_bands: Tensor,
        positions: Tensor,
        species: Tensor,
        cell: Tensor,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if parent_bands.shape[-1] != self.n_bands:
            raise ValueError("parent band count differs from model output")
        if k_frac.shape[:-1] != parent_bands.shape[:-1]:
            raise ValueError("k-point and parent leading dimensions differ")
        context = self.structure_encoder(
            positions,
            species,
            cell,
            edge_index,
            cell_shifts,
        ).unsqueeze(0).expand(len(k_frac), -1)
        hidden = self.trunk(
            torch.cat(
                (
                    self.periodic_features(k_frac).to(parent_bands),
                    parent_bands,
                    context.to(parent_bands),
                ),
                dim=-1,
            )
        )
        edge_shifts = self.edge_head(hidden)
        raw_dispersion = self.dispersion_head(hidden)
        valence = raw_dispersion[..., : self.n_valence]
        conduction = raw_dispersion[..., self.n_valence :]
        dispersion = torch.cat(
            (
                valence - valence.mean(dim=-1, keepdim=True),
                conduction - conduction.mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        return edge_shifts, dispersion

    def forward(
        self,
        k_frac: Tensor,
        parent_bands: Tensor,
        positions: Tensor,
        species: Tensor,
        cell: Tensor,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> Tensor:
        edge_shifts, dispersion = self.residual_components(
            k_frac,
            parent_bands,
            positions,
            species,
            cell,
            edge_index,
            cell_shifts,
        )
        valence = (
            parent_bands[..., : self.n_valence]
            + dispersion[..., : self.n_valence]
            + edge_shifts[..., :1]
        )
        conduction = (
            parent_bands[..., self.n_valence :]
            + dispersion[..., self.n_valence :]
            + edge_shifts[..., 1:]
        )
        return torch.cat(
            (
                torch.sort(valence, dim=-1).values,
                torch.sort(conduction, dim=-1).values,
            ),
            dim=-1,
        )


class FusedStructureBandResidualMLP(PeriodicBandResidualMLP):
    """Warm-startable fusion of fixed and learned crystal representations."""

    def __init__(
        self,
        n_bands: int,
        num_species: int,
        structure_width: int,
        *,
        harmonic_order: int = 2,
        width: int = 256,
        depth: int = 3,
        context_width: int = 128,
        encoder_channels: int = 64,
        message_layers: int = 2,
        num_radial: int = 16,
        cutoff: float = 5.0,
        gate_logit: float = -4.0,
        element_features: Tensor | None = None,
        element_hidden: int = 32,
    ) -> None:
        nn.Module.__init__(self)
        if n_bands < 2 or structure_width < 1 or width < 1 or depth < 1:
            raise ValueError("model dimensions must be positive")
        self.n_bands = int(n_bands)
        frequencies = canonical_integer_frequencies(int(harmonic_order))
        self.register_buffer("frequencies", frequencies)
        self.fixed_structure_encoder = nn.Sequential(
            nn.Linear(structure_width, width),
            nn.SiLU(),
            nn.Linear(width, context_width),
            nn.LayerNorm(context_width),
        )
        self.learned_structure_encoder = InvariantMessagePassingEncoder(
            num_species,
            channels=encoder_channels,
            message_layers=message_layers,
            num_radial=num_radial,
            cutoff=cutoff,
            context_width=context_width,
            element_features=element_features,
            element_hidden=element_hidden,
        )
        self.fusion_gate = nn.Linear(2 * context_width, context_width)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, float(gate_logit))
        input_width = (
            2 * int(frequencies.shape[0])
            + self.n_bands
            + int(context_width)
        )
        layers: list[nn.Module] = [nn.Linear(input_width, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.SiLU()))
        layers.append(nn.Linear(width, self.n_bands))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        k_frac: Tensor,
        parent_bands: Tensor,
        structure_descriptor: Tensor,
        positions: Tensor,
        species: Tensor,
        cell: Tensor,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> Tensor:
        if structure_descriptor.ndim != 1:
            raise ValueError("fusion expects one descriptor per cell")
        fixed = self.fixed_structure_encoder(
            structure_descriptor.to(parent_bands)
        )
        learned = self.learned_structure_encoder(
            positions,
            species,
            cell,
            edge_index,
            cell_shifts,
        ).to(parent_bands)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((fixed, learned))))
        context = fixed + gate * (learned - fixed)
        context = context.unsqueeze(0).expand(len(k_frac), -1)
        features = torch.cat(
            (
                self.periodic_features(k_frac).to(parent_bands),
                parent_bands,
                context,
            ),
            dim=-1,
        )
        corrected = parent_bands + self.network(features)
        return torch.sort(corrected, dim=-1).values


class MultiHeadFusedStructureBandResidualMLP(PeriodicBandResidualMLP):
    """Warm-startable multi-head decoder with fixed/learned representation fusion."""

    def __init__(
        self,
        n_bands: int,
        n_valence: int,
        num_species: int,
        structure_width: int,
        *,
        harmonic_order: int = 2,
        width: int = 256,
        depth: int = 3,
        context_width: int = 128,
        encoder_channels: int = 64,
        message_layers: int = 2,
        num_radial: int = 16,
        cutoff: float = 5.0,
        gate_logit: float = -4.0,
        element_features: Tensor | None = None,
        element_hidden: int = 32,
    ) -> None:
        nn.Module.__init__(self)
        if not 1 <= n_valence < n_bands:
            raise ValueError("n_valence must lie strictly inside the band window")
        if structure_width < 1 or width < 1 or depth < 1 or context_width < 1:
            raise ValueError("model dimensions must be positive")
        self.n_bands = int(n_bands)
        self.n_valence = int(n_valence)
        frequencies = canonical_integer_frequencies(int(harmonic_order))
        self.register_buffer("frequencies", frequencies)
        self.fixed_structure_encoder = nn.Sequential(
            nn.Linear(structure_width, width),
            nn.SiLU(),
            nn.Linear(width, context_width),
            nn.LayerNorm(context_width),
        )
        self.learned_structure_encoder = InvariantMessagePassingEncoder(
            num_species,
            channels=encoder_channels,
            message_layers=message_layers,
            num_radial=num_radial,
            cutoff=cutoff,
            context_width=context_width,
            element_features=element_features,
            element_hidden=element_hidden,
        )
        self.fusion_gate = nn.Linear(2 * context_width, context_width)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, float(gate_logit))
        input_width = (
            2 * int(frequencies.shape[0])
            + self.n_bands
            + int(context_width)
        )
        trunk: list[nn.Module] = [nn.Linear(input_width, width), nn.SiLU()]
        for _ in range(depth - 1):
            trunk.extend((nn.Linear(width, width), nn.SiLU()))
        self.trunk = nn.Sequential(*trunk)
        self.edge_head = nn.Linear(width, 2)
        self.dispersion_head = nn.Linear(width, self.n_bands)
        for output in (self.edge_head, self.dispersion_head):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def _context(
        self,
        parent_bands: Tensor,
        structure_descriptor: Tensor,
        positions: Tensor,
        species: Tensor,
        cell: Tensor,
        edge_index: Tensor | None,
        cell_shifts: Tensor | None,
    ) -> Tensor:
        if structure_descriptor.ndim != 1:
            raise ValueError("fusion expects one descriptor per cell")
        fixed = self.fixed_structure_encoder(structure_descriptor.to(parent_bands))
        learned = self.learned_structure_encoder(
            positions,
            species,
            cell,
            edge_index,
            cell_shifts,
        ).to(parent_bands)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((fixed, learned))))
        return fixed + gate * (learned - fixed)

    def residual_components(
        self,
        k_frac: Tensor,
        parent_bands: Tensor,
        structure_descriptor: Tensor,
        positions: Tensor,
        species: Tensor,
        cell: Tensor,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if parent_bands.shape[-1] != self.n_bands:
            raise ValueError("parent band count differs from model output")
        if k_frac.shape[:-1] != parent_bands.shape[:-1]:
            raise ValueError("k-point and parent leading dimensions differ")
        context = self._context(
            parent_bands,
            structure_descriptor,
            positions,
            species,
            cell,
            edge_index,
            cell_shifts,
        ).unsqueeze(0).expand(len(k_frac), -1)
        hidden = self.trunk(
            torch.cat(
                (
                    self.periodic_features(k_frac).to(parent_bands),
                    parent_bands,
                    context,
                ),
                dim=-1,
            )
        )
        edge_shifts = self.edge_head(hidden)
        raw_dispersion = self.dispersion_head(hidden)
        valence = raw_dispersion[..., : self.n_valence]
        conduction = raw_dispersion[..., self.n_valence :]
        dispersion = torch.cat(
            (
                valence - valence.mean(dim=-1, keepdim=True),
                conduction - conduction.mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        return edge_shifts, dispersion

    def forward(
        self,
        k_frac: Tensor,
        parent_bands: Tensor,
        structure_descriptor: Tensor,
        positions: Tensor,
        species: Tensor,
        cell: Tensor,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> Tensor:
        edge_shifts, dispersion = self.residual_components(
            k_frac,
            parent_bands,
            structure_descriptor,
            positions,
            species,
            cell,
            edge_index,
            cell_shifts,
        )
        valence = (
            parent_bands[..., : self.n_valence]
            + dispersion[..., : self.n_valence]
            + edge_shifts[..., :1]
        )
        conduction = (
            parent_bands[..., self.n_valence :]
            + dispersion[..., self.n_valence :]
            + edge_shifts[..., 1:]
        )
        return torch.cat(
            (
                torch.sort(valence, dim=-1).values,
                torch.sort(conduction, dim=-1).values,
            ),
            dim=-1,
        )


__all__ = [
    "FusedStructureBandResidualMLP",
    "InvariantMessagePassingEncoder",
    "MessagePassingBandResidualMLP",
    "MultiHeadFusedStructureBandResidualMLP",
    "MultiHeadMessagePassingBandResidualMLP",
    "MultiHeadStructureBandResidualMLP",
    "PeriodicBandResidualMLP",
    "StructureConditionedBandResidualMLP",
    "canonical_integer_frequencies",
    "invariant_structure_descriptor",
    "periodic_table_descriptor_width",
    "periodic_table_structure_descriptor",
]
