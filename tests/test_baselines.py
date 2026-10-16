import torch

from epnet.models.baselines import CosineRadialEP, DescriptorMLP, periodic_radial_descriptor


def test_cosine_radial_cutoff_and_collocation():
    model = CosineRadialEP(1, n_basis=6, q_cutoff=4.0).double()
    q = torch.linspace(0, 3.8, 20, dtype=torch.double)
    target = torch.exp(-q)
    model.fit_collocation(0, q, target)
    error = torch.mean((model(torch.zeros_like(q, dtype=torch.long), q) - target) ** 2)
    assert error < 2e-3
    edge = torch.tensor([4.0, 4.1], dtype=torch.double)
    torch.testing.assert_close(model(torch.zeros(2, dtype=torch.long), edge), torch.zeros(2, dtype=torch.double))


def test_descriptor_translation_and_permutation_covariance():
    lattice = torch.eye(3, dtype=torch.double) * 4
    positions = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=torch.double)
    species = torch.tensor([0, 1, 0])
    descriptor = periodic_radial_descriptor(lattice, positions, species, n_species=2, cutoff=3)
    translated = periodic_radial_descriptor(lattice, positions + 0.37, species, n_species=2, cutoff=3)
    torch.testing.assert_close(descriptor, translated)
    permutation = torch.tensor([2, 0, 1])
    permuted = periodic_radial_descriptor(
        lattice, positions[permutation], species[permutation], n_species=2, cutoff=3,
    )
    torch.testing.assert_close(permuted, descriptor[permutation])


def test_descriptor_mlp_is_differentiable():
    torch.manual_seed(4)
    model = DescriptorMLP(2, n_radial=8, n_output=5, hidden=(16,))
    lattice = torch.eye(3) * 5
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]], requires_grad=True)
    output = model(lattice, positions, torch.tensor([0, 1]))
    assert output.shape == (2, 5)
    output.square().sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
