import torch

from epnet.models import EquivariantEncoder
from epnet.models.equivariant_encoder import build_periodic_neighbor_graph


def random_rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def structure():
    positions = torch.tensor(
        [[0.2, 0.4, 0.7], [1.3, 0.8, 2.1], [2.6, 1.9, 0.3], [3.1, 3.2, 2.8]],
        dtype=torch.float64,
    )
    cell = torch.tensor(
        [[4.0, 0.0, 0.0], [0.3, 3.8, 0.0], [0.1, 0.2, 4.2]], dtype=torch.float64
    )
    species = torch.tensor([0, 1, 0, 2])
    return positions, cell, species


def test_periodic_encoder_exact_rotation_equivariance():
    torch.manual_seed(4)
    positions, cell, species = structure()
    encoder = EquivariantEncoder(3, channels=7, cutoff=3.2, num_radial=5).double().eval()
    original = encoder(positions, species, cell)

    rotation = random_rotation(9)
    rotated = encoder(positions @ rotation.T, species, cell @ rotation.T)
    assert torch.allclose(rotated["scalar"], original["scalar"], atol=2e-11, rtol=2e-11)
    expected_vector = torch.einsum("ij,ncj->nci", rotation, original["vector"])
    expected_tensor = torch.einsum(
        "ik,nckl,jl->ncij", rotation, original["tensor"], rotation
    )
    assert torch.allclose(rotated["vector"], expected_vector, atol=3e-11, rtol=3e-11)
    assert torch.allclose(rotated["tensor"], expected_tensor, atol=3e-11, rtol=3e-11)
    assert torch.allclose(
        rotated["tensor"], rotated["tensor"].transpose(-1, -2), atol=1e-13
    )
    assert torch.allclose(
        rotated["tensor"].diagonal(dim1=-2, dim2=-1).sum(-1),
        torch.zeros_like(rotated["scalar"]),
        atol=2e-11,
    )


def test_periodic_encoder_permutation_equivariance():
    torch.manual_seed(5)
    positions, cell, species = structure()
    encoder = EquivariantEncoder(3, channels=5, cutoff=3.4, num_radial=4).double().eval()
    reference = encoder(positions, species, cell)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = encoder(positions[permutation], species[permutation], cell)
    for key in reference:
        assert torch.allclose(permuted[key], reference[key][permutation], atol=3e-11, rtol=3e-11)


def test_cell_shifts_reconstruct_periodic_edges():
    cell = 4.0 * torch.eye(3, dtype=torch.float64)
    positions = torch.tensor([[0.1, 0.2, 0.3], [3.9, 0.2, 0.3]], dtype=torch.float64)
    edge_index, shifts = build_periodic_neighbor_graph(positions, cell, cutoff=0.5)
    vectors = positions[edge_index[1]] + shifts.to(cell) @ cell - positions[edge_index[0]]
    assert torch.all(torch.linalg.vector_norm(vectors, dim=-1) < 0.5)
    assert torch.any(shifts != 0)
