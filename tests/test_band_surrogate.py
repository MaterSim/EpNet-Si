import torch

from epnet.models.band_surrogate import (
    FusedStructureBandResidualMLP,
    InvariantMessagePassingEncoder,
    MessagePassingBandResidualMLP,
    MultiHeadFusedStructureBandResidualMLP,
    MultiHeadMessagePassingBandResidualMLP,
    MultiHeadStructureBandResidualMLP,
    PeriodicBandResidualMLP,
    StructureConditionedBandResidualMLP,
    canonical_integer_frequencies,
    invariant_structure_descriptor,
)


def test_canonical_integer_frequencies_has_one_sign_representative():
    frequencies = canonical_integer_frequencies(2)
    assert frequencies.shape == (62, 3)
    vectors = {tuple(int(value) for value in row) for row in frequencies}
    assert (0, 0, 0) not in vectors
    assert all(tuple(-value for value in vector) not in vectors for vector in vectors)


def test_periodic_band_surrogate_zero_attach_and_ordering():
    model = PeriodicBandResidualMLP(4, harmonic_order=2, width=8, depth=2)
    kpoints = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.0, 0.2]])
    parent = torch.tensor([[0.3, -1.0, 0.2, 1.0], [-0.8, 0.1, 0.0, 0.9]])
    prediction = model(kpoints, parent)
    torch.testing.assert_close(prediction, torch.sort(parent, dim=-1).values)
    assert torch.all(prediction[:, 1:] >= prediction[:, :-1])


def test_periodic_band_surrogate_is_reciprocal_translation_invariant():
    model = PeriodicBandResidualMLP(
        3,
        harmonic_order=3,
        width=12,
        depth=2,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
    kpoints = torch.tensor([[0.13, 0.27, 0.41]], dtype=torch.float64)
    parent = torch.tensor([[-0.8, 0.2, 1.1]], dtype=torch.float64)
    first = model(kpoints, parent)
    second = model(
        kpoints + torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float64),
        parent,
    )
    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_invariant_structure_descriptor_survives_rigid_rotation_and_translation():
    lattice = torch.diag(torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64))
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]],
        dtype=torch.float64,
    )
    species = torch.tensor([0, 1])
    angle = torch.tensor(0.47, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    first = invariant_structure_descriptor(
        lattice,
        positions,
        species,
        n_species=2,
    )
    second = invariant_structure_descriptor(
        lattice @ rotation.T,
        positions + torch.tensor([0.3, -0.2, 0.4]),
        species,
        n_species=2,
    )
    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_structure_conditioned_surrogate_zero_attach():
    model = StructureConditionedBandResidualMLP(
        4,
        7,
        harmonic_order=2,
        width=16,
        depth=2,
        context_width=8,
    )
    kpoints = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.0, 0.2]])
    parent = torch.tensor([[-1.0, 0.2, 0.3, 1.0], [-0.8, 0.0, 0.1, 0.9]])
    descriptor = torch.randn(7)
    torch.testing.assert_close(
        model(kpoints, parent, descriptor),
        parent,
    )


def test_multihead_surrogate_zero_attach_and_manifold_ordering():
    model = MultiHeadStructureBandResidualMLP(
        6,
        2,
        7,
        harmonic_order=2,
        width=16,
        depth=2,
        context_width=8,
    )
    kpoints = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.0, 0.2]])
    parent = torch.tensor(
        [[-1.0, -0.2, 0.3, 0.6, 0.8, 1.0], [-0.8, 0.0, 0.1, 0.5, 0.7, 0.9]]
    )
    descriptor = torch.randn(7)
    prediction = model(kpoints, parent, descriptor)
    torch.testing.assert_close(prediction, parent)
    assert torch.all(prediction[:, 1:2] >= prediction[:, :1])
    assert torch.all(prediction[:, 3:] >= prediction[:, 2:-1])


def test_multihead_dispersion_is_centered_and_k_periodic():
    torch.manual_seed(19)
    model = MultiHeadStructureBandResidualMLP(
        6,
        2,
        7,
        harmonic_order=2,
        width=16,
        depth=2,
        context_width=8,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
    kpoints = torch.tensor([[0.13, 0.27, 0.41]], dtype=torch.float64)
    parent = torch.tensor([[-1.0, -0.2, 0.3, 0.6, 0.8, 1.0]], dtype=torch.float64)
    descriptor = torch.randn(7, dtype=torch.float64)
    edge, dispersion = model.residual_components(kpoints, parent, descriptor)
    _, shifted_dispersion = model.residual_components(
        kpoints + torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float64),
        parent,
        descriptor,
    )
    torch.testing.assert_close(
        dispersion[..., :2].mean(dim=-1),
        torch.zeros_like(dispersion[..., 0]),
    )
    torch.testing.assert_close(
        dispersion[..., 2:].mean(dim=-1),
        torch.zeros_like(dispersion[..., 0]),
    )
    torch.testing.assert_close(dispersion, shifted_dispersion, atol=1e-10, rtol=1e-10)
    assert edge.shape == (1, 2)


def _test_crystal(dtype=torch.float64):
    cell = torch.tensor(
        [[4.2, 0.1, 0.0], [0.3, 4.8, 0.2], [0.1, 0.4, 5.3]],
        dtype=dtype,
    )
    fractional = torch.tensor(
        [[0.05, 0.12, 0.21], [0.42, 0.63, 0.74], [0.81, 0.24, 0.55]],
        dtype=dtype,
    )
    return fractional @ cell, torch.tensor([0, 1, 0]), cell


def test_message_passing_encoder_is_e3_and_permutation_invariant():
    torch.manual_seed(11)
    encoder = InvariantMessagePassingEncoder(
        2,
        channels=12,
        message_layers=2,
        num_radial=5,
        cutoff=4.0,
        context_width=9,
    ).double()
    positions, species, cell = _test_crystal()
    angle = torch.tensor(0.37, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    permutation = torch.tensor([2, 0, 1])
    first = encoder(positions, species, cell)
    second = encoder(
        positions.index_select(0, permutation) @ rotation.T
        + torch.tensor([0.7, -0.2, 0.4], dtype=torch.float64),
        species.index_select(0, permutation),
        cell @ rotation.T,
    )
    torch.testing.assert_close(first, second, atol=2e-10, rtol=2e-10)


def test_message_passing_band_surrogate_zero_attach_and_encoder_gradients():
    torch.manual_seed(13)
    model = MessagePassingBandResidualMLP(
        4,
        2,
        harmonic_order=2,
        width=16,
        depth=2,
        context_width=8,
        encoder_channels=10,
        message_layers=2,
        num_radial=4,
        cutoff=4.0,
    ).double()
    positions, species, cell = _test_crystal()
    kpoints = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.0, 0.2]], dtype=torch.float64)
    parent = torch.tensor(
        [[-1.0, 0.2, 0.3, 1.0], [-0.8, 0.0, 0.1, 0.9]],
        dtype=torch.float64,
    )
    prediction = model(kpoints, parent, positions, species, cell)
    torch.testing.assert_close(prediction, parent)
    with torch.no_grad():
        model.network[-1].weight.normal_(std=0.1)
    model(kpoints, parent, positions, species, cell).square().mean().backward()
    assert model.structure_encoder.species_embedding.weight.grad is not None
    assert model.structure_encoder.message_networks[0][0].weight.grad is not None
    assert torch.isfinite(model.structure_encoder.species_embedding.weight.grad).all()


def test_multihead_message_passing_zero_attach_and_encoder_gradients():
    torch.manual_seed(23)
    model = MultiHeadMessagePassingBandResidualMLP(
        6,
        2,
        2,
        harmonic_order=2,
        width=16,
        depth=2,
        context_width=8,
        encoder_channels=10,
        message_layers=2,
        num_radial=4,
        cutoff=4.0,
    ).double()
    positions, species, cell = _test_crystal()
    kpoints = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.0, 0.2]], dtype=torch.float64)
    parent = torch.tensor(
        [[-1.0, -0.2, 0.3, 0.6, 0.8, 1.0], [-0.8, 0.0, 0.1, 0.5, 0.7, 0.9]],
        dtype=torch.float64,
    )
    prediction = model(kpoints, parent, positions, species, cell)
    torch.testing.assert_close(prediction, parent)
    with torch.no_grad():
        model.edge_head.weight.normal_(std=0.1)
        model.dispersion_head.weight.normal_(std=0.1)
    model(kpoints, parent, positions, species, cell).square().mean().backward()
    assert model.structure_encoder.species_embedding.weight.grad is not None
    assert model.structure_encoder.message_networks[0][0].weight.grad is not None
    assert torch.isfinite(model.structure_encoder.species_embedding.weight.grad).all()


def test_fused_surrogate_closed_gate_recovers_fixed_model():
    torch.manual_seed(17)
    fixed = StructureConditionedBandResidualMLP(
        4,
        7,
        harmonic_order=2,
        width=16,
        depth=2,
        context_width=8,
    ).double()
    with torch.no_grad():
        for parameter in fixed.parameters():
            parameter.normal_(std=0.1)
    fused = FusedStructureBandResidualMLP(
        4,
        2,
        7,
        harmonic_order=2,
        width=16,
        depth=2,
        context_width=8,
        encoder_channels=10,
        message_layers=2,
        num_radial=4,
        cutoff=4.0,
        gate_logit=-40.0,
    ).double()
    mapped = {}
    for key, value in fixed.state_dict().items():
        mapped[
            "fixed_structure_encoder." + key.removeprefix("structure_encoder.")
            if key.startswith("structure_encoder.")
            else key
        ] = value
    fused.load_state_dict(mapped, strict=False)
    positions, species, cell = _test_crystal()
    kpoints = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.0, 0.2]], dtype=torch.float64)
    parent = torch.tensor(
        [[-1.0, 0.2, 0.3, 1.0], [-0.8, 0.0, 0.1, 0.9]],
        dtype=torch.float64,
    )
    descriptor = torch.randn(7, dtype=torch.float64)
    torch.testing.assert_close(
        fused(kpoints, parent, descriptor, positions, species, cell),
        fixed(kpoints, parent, descriptor),
        atol=1e-12,
        rtol=1e-12,
    )


def test_multihead_fused_closed_gate_recovers_fixed_model():
    torch.manual_seed(29)
    fixed = MultiHeadStructureBandResidualMLP(
        6,
        2,
        7,
        harmonic_order=2,
        width=16,
        depth=2,
        context_width=8,
    ).double()
    with torch.no_grad():
        for parameter in fixed.parameters():
            parameter.normal_(std=0.1)
    fused = MultiHeadFusedStructureBandResidualMLP(
        6,
        2,
        2,
        7,
        harmonic_order=2,
        width=16,
        depth=2,
        context_width=8,
        encoder_channels=10,
        message_layers=2,
        num_radial=4,
        cutoff=4.0,
        gate_logit=-40.0,
    ).double()
    mapped = {}
    for key, value in fixed.state_dict().items():
        mapped[
            "fixed_structure_encoder." + key.removeprefix("structure_encoder.")
            if key.startswith("structure_encoder.")
            else key
        ] = value
    fused.load_state_dict(mapped, strict=False)
    positions, species, cell = _test_crystal()
    kpoints = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.0, 0.2]], dtype=torch.float64)
    parent = torch.tensor(
        [[-1.0, -0.2, 0.3, 0.6, 0.8, 1.0], [-0.8, 0.0, 0.1, 0.5, 0.7, 0.9]],
        dtype=torch.float64,
    )
    descriptor = torch.randn(7, dtype=torch.float64)
    torch.testing.assert_close(
        fused(kpoints, parent, descriptor, positions, species, cell),
        fixed(kpoints, parent, descriptor),
        atol=1e-12,
        rtol=1e-12,
    )
