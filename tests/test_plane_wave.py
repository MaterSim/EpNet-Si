import numpy as np
import pytest

from epnet.hamiltonian import assemble_hamiltonian, solve_bands


def test_free_electron_bands():
    basis = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    reciprocal = 2 * np.pi * np.eye(3)
    energies = solve_bands(basis, reciprocal, [[0, 0, 0]], {(0, 0, 0): 0})
    np.testing.assert_allclose(energies[0], [0, 2 * np.pi**2, 2 * np.pi**2])


def test_local_potential_coupling_is_hermitian():
    basis = np.array([[0, 0, 0], [1, 0, 0]])
    potential = {
        (0, 0, 0): 1.0,
        (1, 0, 0): 0.2 + 0.3j,
        (-1, 0, 0): 0.2 - 0.3j,
    }
    h = assemble_hamiltonian(basis, np.eye(3), np.zeros(3), potential)
    np.testing.assert_allclose(h, h.conj().T)
    expected = np.linalg.eigvalsh(np.array([[1, 0.2 - 0.3j], [0.2 + 0.3j, 1.5]]))
    np.testing.assert_allclose(solve_bands(basis, np.eye(3), [[0, 0, 0]], potential)[0], expected)


def test_nonhermitian_potential_is_rejected():
    basis = np.array([[0, 0, 0], [1, 0, 0]])
    bad = {(1, 0, 0): 1j, (-1, 0, 0): 1j}
    with pytest.raises(ValueError, match="non-Hermitian"):
        assemble_hamiltonian(basis, np.eye(3), np.zeros(3), bad)
