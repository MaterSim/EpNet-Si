from types import SimpleNamespace

import torch

from epnet.hamiltonian.torch_pw import batched_band_energies
from epnet.data.structures import Structure
from epnet.models.si_epm import build_gated_si_epm, load_gated_si_epm_state, pack_structure
from epnet.models.sp_projector import (
    SPDFullRadialProjectorCorrection,
    SPDProjectorCorrection,
    SPProjectorCorrection,
)
from epnet.train_cells import move_pw_cache


def _packed(position=(0.2, 0.3, 0.4)):
    lattice = torch.eye(3, dtype=torch.float64) * 5.0
    positions = torch.tensor([position], dtype=torch.float64) @ lattice
    return SimpleNamespace(
        lattice=lattice,
        positions=positions,
        species=torch.tensor([0]),
        volume=torch.tensor(125.0, dtype=torch.float64),
    )


def test_sp_projector_is_zero_initialized_and_hermitian():
    model = SPProjectorCorrection(1).double()
    k = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float64)
    g = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    operator = model(_packed(), k, g)
    assert operator.shape == (2, 3, 3)
    torch.testing.assert_close(operator, torch.zeros_like(operator))
    with torch.no_grad():
        model.coefficients.copy_(torch.tensor([[[0.5, -0.2], [0.1, 0.3]]]))
    operator = model(_packed(), k, g)
    torch.testing.assert_close(operator, operator.mH)


def test_projector_operator_is_invariant_to_lattice_translation():
    model = SPProjectorCorrection(1).double()
    with torch.no_grad():
        model.coefficients.fill_(0.4)
    k = torch.tensor([[0.13, 0.07, 0.0]], dtype=torch.float64)
    g = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    first = model(_packed(), k, g)
    second = model(_packed(position=(1.2, 0.3, 0.4)), k, g)
    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_spd_projector_is_zero_initialized_and_hermitian():
    model = SPDProjectorCorrection(1, (1.0,)).double()
    k = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
    g = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    operator = model(_packed(), k, g)
    torch.testing.assert_close(operator, torch.zeros_like(operator))
    with torch.no_grad():
        model.coefficients.copy_(torch.tensor([[[0.5, -0.2, 0.3]]]))
    operator = model(_packed(), k, g)
    torch.testing.assert_close(operator, operator.mH)


def test_spd_d_multiplet_has_direction_independent_norm():
    model = SPDProjectorCorrection(1, (1.0,)).double()
    k = torch.tensor(
        [[0.1, 0.0, 0.0], [0.1 / 2**0.5, 0.1 / 2**0.5, 0.0]],
        dtype=torch.float64,
    )
    basis = model.projectors(_packed(position=(0.0, 0.0, 0.0)), k, torch.tensor([[0, 0, 0]]))
    d_norm = basis[:, 0, 4:].abs().square().sum(dim=-1)
    torch.testing.assert_close(d_norm[0], d_norm[1], atol=1e-12, rtol=1e-12)


def test_full_radial_projector_is_zero_initialized_hermitian_and_compact():
    model = SPDFullRadialProjectorCorrection(2, (0.8, 1.6)).double()
    assert model.independent_parameter_count == 18
    assert not hasattr(model, "coefficients")
    k = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
    g = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    operator = model(_packed(), k, g)
    torch.testing.assert_close(operator, torch.zeros_like(operator))
    with torch.no_grad():
        model.packed_couplings.normal_()
    operator = model(_packed(), k, g)
    torch.testing.assert_close(operator, operator.mH, atol=1e-12, rtol=1e-12)


def test_full_radial_projector_embeds_diagonal_checkpoint_by_scale():
    model = SPDFullRadialProjectorCorrection(2, (0.5, 0.8, 1.6)).double()
    coefficients = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3)
    mapped = model.initialize_from_diagonal(
        coefficients,
        torch.tensor([0.8, 1.6], dtype=torch.float64),
    )
    assert mapped == (1, 2)
    diagonal = torch.diagonal(
        model.coupling_matrices(torch.tensor([0, 1])),
        dim1=-2,
        dim2=-1,
    )
    expected = torch.zeros(2, 3, 3, dtype=torch.float64)
    expected[..., 1:] = coefficients.transpose(1, 2)
    torch.testing.assert_close(diagonal, expected)


def test_full_radial_projector_uses_off_diagonal_couplings():
    model = SPDFullRadialProjectorCorrection(1, (0.8, 1.6)).double()
    with torch.no_grad():
        model.packed_couplings.zero_()
        # Lower-triangle order for a 2x2 matrix: (0,0), (1,0), (1,1).
        model.packed_couplings[0, 0, 1] = 0.7
    k = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
    g = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    operator = model(_packed(), k, g, active_l=0)
    assert float(operator.abs().max()) > 0.0
    torch.testing.assert_close(operator, operator.mH, atol=1e-12, rtol=1e-12)


def test_full_radial_projector_is_invariant_to_lattice_translation():
    model = SPDFullRadialProjectorCorrection(1, (0.8, 1.6)).double()
    with torch.no_grad():
        model.packed_couplings.normal_()
    k = torch.tensor([[0.13, 0.07, 0.0]], dtype=torch.float64)
    g = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    first = model(_packed(), k, g)
    second = model(_packed(position=(1.2, 0.3, 0.4)), k, g)
    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_full_radial_projector_is_invariant_to_rigid_rotation():
    model = SPDFullRadialProjectorCorrection(1, (0.8, 1.6)).double()
    with torch.no_grad():
        model.packed_couplings.normal_()
    angle = torch.tensor(0.61, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    first = _packed()
    second = _packed()
    second.lattice = first.lattice @ rotation.T
    second.positions = first.positions @ rotation.T
    k = torch.tensor([[0.13, 0.07, 0.03]], dtype=torch.float64)
    g = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    torch.testing.assert_close(
        model(first, k, g),
        model(second, k, g),
        atol=1e-10,
        rtol=1e-10,
    )


def test_batched_band_energies_accepts_k_dependent_potential_chunks():
    kinetic = torch.tensor([[0.0, 1.0], [0.2, 1.2], [0.4, 1.4]])
    potential = torch.zeros(3, 2, 2, dtype=torch.complex64)
    potential[:, 0, 0] = torch.tensor([0.0, 0.1, 0.2])
    got = batched_band_energies(kinetic, potential, k_batch=2)
    expected = torch.linalg.eigvalsh(potential + torch.diag_embed(kinetic.to(potential.dtype)))
    torch.testing.assert_close(got, expected)


def test_zero_attached_projector_preserves_parent_bands():
    structure = Structure(
        lattice=torch.eye(3).numpy() * 5.0,
        species=("Si",),
        fractional_positions=torch.tensor([[0.2, 0.3, 0.4]]).numpy(),
        metadata={"n_occupied": 2},
    )
    packed = pack_structure(structure)
    k = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    parent = build_gated_si_epm(channels=4, ecut_ry=2, monopole_mode="free")
    child = build_gated_si_epm(channels=4, ecut_ry=2, monopole_mode="free", sp_nonlocal=True)
    load_gated_si_epm_state(child, parent.state_dict())
    expected = parent.band_path(packed, k, n_bands=3)
    got = child.band_path(packed, k, n_bands=3)
    torch.testing.assert_close(got, expected)


def test_two_radial_sp_channels_match_four_extra_sphere_coefficients():
    local = build_gated_si_epm(channels=4, sphere_n_basis=28)
    projector = build_gated_si_epm(channels=4, sphere_n_basis=24, sp_nonlocal=True)
    local.set_sphere_trainable(True)
    projector.set_sphere_trainable(True)
    n_local = sum(p.numel() for p in local.parameters() if p.requires_grad)
    n_projector = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    assert n_local == n_projector


def test_environment_factors_scale_projector_channels():
    model = SPProjectorCorrection(2).double()
    with torch.no_grad():
        model.coefficients.fill_(2.0)
    species = torch.tensor([0, 1])
    factors = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.0], [1.5, 2.0]]],
        dtype=torch.float64,
    )
    channels = model.channel_coefficients(species, n_k=1, environment_factors=factors)
    assert channels.shape == (1, 16)
    torch.testing.assert_close(channels[0, :4], torch.tensor([2.0, 4.0, 4.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(channels[0, 4:8], torch.tensor([6.0, 8.0, 8.0, 8.0], dtype=torch.float64))


def test_train_only_cache_keeps_projector_geometry():
    cache = SimpleNamespace(
        g_idx=torch.zeros(2, 3, dtype=torch.long),
        sp_projectors=torch.zeros(1, 2, 4, dtype=torch.complex64),
        volume=10.0,
        n_g=2,
    )
    moved = move_pw_cache(cache, torch.device("cpu"), train_only=True)
    assert moved.g_idx.shape == (2, 3)
    assert moved.sp_projectors.shape == (1, 2, 4)
