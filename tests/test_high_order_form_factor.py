import torch

from epnet.models.equivariant_encoder import build_periodic_neighbor_graph
from epnet.models.high_order_form_factor import HighOrderEnvironmentFormFactor


def _cell(dtype=torch.float64):
    lattice = torch.tensor(
        [[3.1, 0.2, 0.0], [0.1, 3.4, 0.3], [0.0, 0.2, 3.7]], dtype=dtype
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.8, 0.5], [0.4, 1.6, 1.1]], dtype=dtype
    )
    species = torch.tensor([0, 1, 0])
    edge_index, shifts = build_periodic_neighbor_graph(positions, lattice, 4.0)
    q = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, -0.4, 0.2], [-0.7, 0.4, -0.2]], dtype=dtype
    )
    return lattice, positions, species, edge_index, shifts, q


def test_high_order_form_factor_is_hermitian_and_differentiable():
    torch.manual_seed(4)
    model = HighOrderEnvironmentFormFactor(2, channels=5, l_max=4).to(torch.float64)
    for layer in model.readout:
        torch.nn.init.normal_(layer.weight, std=0.05)
    lattice, positions, species, edge_index, shifts, q = _cell()
    values = model(
        positions, species, lattice, q, edge_index=edge_index, cell_shifts=shifts
    )
    assert values[0] == 0
    torch.testing.assert_close(values[2], values[1].conj(), atol=1e-9, rtol=1e-9)
    values.abs().sum().backward()
    assert model.readout[4].weight.grad is not None


def test_high_order_form_factor_is_invariant_to_simultaneous_rotation():
    torch.manual_seed(8)
    model = HighOrderEnvironmentFormFactor(2, channels=4, l_max=4).to(torch.float64)
    for layer in model.readout:
        torch.nn.init.normal_(layer.weight, std=0.03)
    lattice, positions, species, edge_index, shifts, q = _cell()
    matrix = torch.randn(3, 3, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    reference = model(
        positions, species, lattice, q, edge_index=edge_index, cell_shifts=shifts
    )
    rotated = model(
        positions @ rotation.T,
        species,
        lattice @ rotation.T,
        q @ rotation.T,
        edge_index=edge_index,
        cell_shifts=shifts,
    )
    torch.testing.assert_close(rotated, reference, atol=2e-8, rtol=2e-8)


def test_global_high_order_form_factor_is_translation_invariant():
    torch.manual_seed(12)
    model = HighOrderEnvironmentFormFactor(
        2, channels=4, l_max=4, aggregation="global"
    ).to(torch.float64)
    for layer in model.readout:
        torch.nn.init.normal_(layer.weight, std=0.03)
    lattice, positions, species, edge_index, shifts, q = _cell()
    reference = model(
        positions, species, lattice, q, edge_index=edge_index, cell_shifts=shifts
    )
    translated = model(
        positions + torch.tensor([0.37, -0.22, 0.41], dtype=torch.float64),
        species,
        lattice,
        q,
        edge_index=edge_index,
        cell_shifts=shifts,
    )
    torch.testing.assert_close(translated, reference, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(translated[2], translated[1].conj(), atol=1e-9, rtol=1e-9)
