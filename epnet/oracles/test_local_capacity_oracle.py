import torch

from epnet.oracles import (
    HermitianLocalCorrection,
    SpeciesRadialShellCorrection,
    aligned_band_objective,
)


def test_local_correction_is_hermitian_and_gauge_fixed():
    q = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    inverse = torch.tensor([[0, 1], [2, 0]])
    correction = HermitianLocalCorrection(q, init_std=0.2)
    matrix = correction(inverse, volume=3.0)
    torch.testing.assert_close(matrix, matrix.mH)
    torch.testing.assert_close(matrix.diagonal(), torch.zeros(2, dtype=matrix.dtype))
    matrix.abs().sum().backward()
    assert correction.real.grad is not None
    assert correction.imag.grad is not None
    assert float(correction.coefficients()[0].abs()) == 0.0


def test_aligned_objective_is_shift_invariant_and_gap_sensitive():
    target = torch.tensor([[-2.0, -0.5, 1.0], [-1.8, -0.4, 0.8]])
    shifted = target + 7.0
    assert float(aligned_band_objective(shifted, target, 2)) < 1e-6
    narrowed = target.clone()
    narrowed[:, 2] -= 0.4
    assert float(aligned_band_objective(narrowed, target, 2)) > 0.05


def test_aligned_objective_accepts_unweighted_window():
    target = torch.tensor([[-1.5, -0.5, 1.0, 2.0], [-1.4, -0.4, 0.8, 2.2]])
    prediction = target.clone()
    prediction[:, 0] += 0.3
    weighted = aligned_band_objective(prediction, target, 2, edge_weight=5.0)
    unweighted = aligned_band_objective(prediction, target, 2, edge_weight=1.0)
    assert float(unweighted) > float(weighted)


def test_species_radial_shell_correction_is_radial_hermitian_and_differentiable():
    q = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.2, 0.1]])
    species = torch.tensor([0, 1])
    inverse = torch.tensor([[0, 1], [2, 0]])
    correction = SpeciesRadialShellCorrection(
        q, positions, species, n_species=2, init_std=0.1
    )
    assert correction.n_shells == 2
    coefficients = correction.coefficients()
    torch.testing.assert_close(coefficients[1], coefficients[2].conj())
    assert float(coefficients[0].abs()) == 0.0
    matrix = correction(inverse, volume=4.0)
    torch.testing.assert_close(matrix, matrix.mH)
    torch.testing.assert_close(matrix.diagonal(), torch.zeros(2, dtype=matrix.dtype))
    matrix.abs().sum().backward()
    assert correction.values.grad is not None


def test_species_radial_shell_clustering_does_not_use_rounding_boundaries():
    q = torch.tensor(
        [[0.0, 0.0, 0.0], [1.000004, 0.0, 0.0], [1.000011, 0.0, 0.0]]
    )
    correction = SpeciesRadialShellCorrection(
        q,
        torch.zeros(1, 3),
        torch.zeros(1, dtype=torch.long),
        n_species=1,
        shell_tolerance=1e-5,
    )
    assert correction.n_shells == 2
    assert correction.shell_index.tolist() == [0, 1, 1]
