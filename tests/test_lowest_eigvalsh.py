import torch

from epnet.hamiltonian.torch_pw import band_energies


def _random_hermitian(n_k: int, n_g: int, *, complex: bool = True, seed: int = 0):
    torch.manual_seed(seed)
    kinetic = torch.randn(n_k, n_g)
    raw = torch.randn(n_g, n_g, dtype=torch.complex64 if complex else torch.float32)
    if complex:
        raw = raw + 1j * torch.randn(n_g, n_g)
    potential = 0.5 * (raw + raw.conj().transpose(-1, -2))
    return kinetic, potential


def test_lowest_eigvalsh_matches_full_solver():
    kinetic, potential = _random_hermitian(4, 32, seed=1)
    full = band_energies(kinetic, potential, n_bands=8, subset_ng_min=10**9)
    subset = band_energies(kinetic, potential, n_bands=8, subset_ng_min=0)
    torch.testing.assert_close(subset, full, atol=1e-4, rtol=1e-4)


def test_lowest_eigvalsh_gradients_match_full_solver():
    kinetic, potential = _random_hermitian(3, 24, seed=2)
    pot_full = potential.detach().clone().requires_grad_(True)
    pot_sub = potential.detach().clone().requires_grad_(True)
    loss_full = band_energies(kinetic, pot_full, n_bands=6, subset_ng_min=10**9).pow(2).mean()
    loss_sub = band_energies(kinetic, pot_sub, n_bands=6, subset_ng_min=0).pow(2).mean()
    loss_full.backward()
    loss_sub.backward()
    torch.testing.assert_close(loss_sub.detach(), loss_full.detach(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(pot_sub.grad, pot_full.grad, atol=2e-4, rtol=2e-4)


def test_truncated_eigvalsh_matches_sliced_torch_grad():
    from epnet.hamiltonian.torch_pw import _TruncatedEigvalsh

    kinetic, potential = _random_hermitian(3, 24, seed=4)
    pot_a = potential.detach().clone().requires_grad_(True)
    pot_b = potential.detach().clone().requires_grad_(True)
    ham_a = pot_a + torch.diag_embed(kinetic.to(pot_a.dtype))
    ham_b = pot_b + torch.diag_embed(kinetic.to(pot_b.dtype))
    loss_a = _TruncatedEigvalsh.apply(ham_a, 6).pow(2).mean()
    loss_b = torch.linalg.eigvalsh(ham_b)[:, :6].pow(2).mean()
    loss_a.backward()
    loss_b.backward()
    torch.testing.assert_close(loss_a.detach(), loss_b.detach(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(pot_a.grad, pot_b.grad, atol=2e-4, rtol=2e-4)


def test_scatter_hermitian_potential_grad_matches_index():
    from epnet.hamiltonian.torch_pw import scatter_hermitian_potential

    torch.manual_seed(5)
    n_g = 12
    n_unique = 20
    inverse = torch.randint(0, n_unique, (n_g, n_g))
    v_a = (torch.randn(n_unique) + 1j * torch.randn(n_unique)).requires_grad_(True)
    v_b = v_a.detach().clone().requires_grad_(True)
    volume = 42.0
    out_a = scatter_hermitian_potential(v_a, inverse, volume)
    matrix = v_b[inverse] / volume
    eye = torch.eye(n_g, dtype=torch.bool)
    matrix = matrix.masked_fill(eye, 0)
    out_b = 0.5 * (matrix + matrix.conj().transpose(-1, -2))
    (out_a.real.pow(2).mean() + out_a.imag.pow(2).mean()).backward()
    (out_b.real.pow(2).mean() + out_b.imag.pow(2).mean()).backward()
    torch.testing.assert_close(out_a.detach(), out_b.detach(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v_a.grad, v_b.grad, atol=2e-5, rtol=2e-5)


def test_batched_band_energies_adaptive_batch_matches():
    from epnet.hamiltonian.torch_pw import batched_band_energies

    kinetic, potential = _random_hermitian(9, 40, seed=3)
    a = batched_band_energies(kinetic, potential, n_bands=10, k_batch=2, subset_ng_min=0)
    b = batched_band_energies(kinetic, potential, n_bands=10, k_batch=9, subset_ng_min=10**9)
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
