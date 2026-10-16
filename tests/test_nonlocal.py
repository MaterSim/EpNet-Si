import torch

from epnet.physics import (
    apply_separable_nonlocal,
    separable_nonlocal_operator,
    separable_spin_nonlocal_operator,
    spin_orbit_channel_coefficients,
)


def _complex_projectors() -> torch.Tensor:
    torch.manual_seed(4)
    real = torch.randn(5, 3, dtype=torch.float64)
    imag = torch.randn(5, 3, dtype=torch.float64)
    return torch.complex(real, imag)


def test_diagonal_separable_operator_shape_and_hermiticity():
    projectors = _complex_projectors()
    coefficients = torch.tensor([0.2, -0.7, 1.1], dtype=torch.float64)
    operator = separable_nonlocal_operator(projectors, coefficients)

    assert operator.shape == (5, 5)
    torch.testing.assert_close(operator, operator.mH)


def test_full_channel_matrix_and_compact_application_agree():
    projectors = _complex_projectors()
    raw = torch.randn(3, 3, dtype=torch.float64)
    coefficients = raw + raw.mT
    states = torch.randn(5, 2, dtype=torch.complex128)

    operator = separable_nonlocal_operator(projectors, coefficients)
    applied = apply_separable_nonlocal(states, projectors, coefficients)

    torch.testing.assert_close(applied, operator @ states)
    torch.testing.assert_close(operator, operator.mH)


def test_zero_coefficients_produce_zero_correction():
    projectors = _complex_projectors()
    operator = separable_nonlocal_operator(projectors, torch.zeros(3))

    torch.testing.assert_close(operator, torch.zeros_like(operator))


def test_soc_ready_channel_blocks_and_spin_operator_are_hermitian():
    projectors = _complex_projectors()
    scalar = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)
    soc = torch.tensor(
        [[0.1, 0.2, -0.1], [0.0, -0.3, 0.2], [0.4, 0.0, 0.1]],
        dtype=torch.float64,
    )
    blocks = spin_orbit_channel_coefficients(scalar, soc)
    operator = separable_spin_nonlocal_operator(projectors, blocks)
    unflattened = separable_spin_nonlocal_operator(
        projectors, blocks, flatten=False
    )

    assert blocks.shape == (3, 2, 2)
    assert operator.shape == (10, 10)
    assert unflattened.shape == (5, 2, 5, 2)
    torch.testing.assert_close(blocks, blocks.mH)
    torch.testing.assert_close(operator, operator.mH)


def test_zero_soc_reduces_to_scalar_spin_identity():
    scalar = torch.tensor([0.5, -0.25])
    blocks = spin_orbit_channel_coefficients(scalar)

    torch.testing.assert_close(blocks[..., 0, 0], scalar.to(blocks.dtype))
    torch.testing.assert_close(blocks[..., 1, 1], scalar.to(blocks.dtype))
    torch.testing.assert_close(blocks[..., 0, 1], torch.zeros(2, dtype=blocks.dtype))
    torch.testing.assert_close(blocks[..., 1, 0], torch.zeros(2, dtype=blocks.dtype))
