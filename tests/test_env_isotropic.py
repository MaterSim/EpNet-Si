import numpy as np
import torch

from epnet.data.structures import Structure
from epnet.models.baselines.radial import EnvironmentRadialDelta
from epnet.models.equivariant_encoder import (
    EquivariantEncoder,
    build_periodic_neighbor_graph,
    packing_gate,
    smooth_coordination,
)
from epnet.models.si_epm import (
    build_gated_si_epm,
    load_gated_si_epm_state,
    pack_structure,
)


def _diamond():
    a = 5.431
    lattice = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    frac = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    return Structure(lattice, ("Si", "Si"), frac, metadata={"n_occupied": 4})


def test_environment_delta_zero_init_is_exactly_zero():
    torch.manual_seed(0)
    layer = EnvironmentRadialDelta(6, n_basis=12, hidden=8)
    out = layer(torch.randn(4, 6))
    assert torch.equal(out, torch.zeros_like(out))


def test_zero_init_env_matches_species_sphere_bands():
    torch.manual_seed(1)
    packed = pack_structure(_diamond())
    k = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    species = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", env_isotropic=False)
    env = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", env_isotropic=True)
    env.load_state_dict(species.state_dict(), strict=False)
    cache = species.make_pw_cache(packed, k)
    n_bands = 8
    with torch.no_grad():
        e0 = species.band_path(packed, k, n_bands=n_bands, cache=cache, residual=True)
        e1 = env.band_path(packed, k, n_bands=n_bands, cache=cache, residual=True)
    torch.testing.assert_close(e1, e0, atol=1e-6, rtol=1e-6)


def test_env_delta_distinguishes_local_density():
    torch.manual_seed(2)
    lattice = np.eye(3) * 10.0
    frac = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.8, 0.8, 0.8]])
    packed = pack_structure(Structure(lattice, ("Si", "Si", "Si"), frac, metadata={"n_occupied": 6}))
    model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", cutoff=3.0, env_isotropic=True)
    with torch.no_grad():
        model.env_radial.net[-1].weight.normal_(0.0, 0.2)
        model.env_radial.net[-1].bias.normal_(0.0, 0.05)
    features = model.encode(packed)
    delta = model.env_radial(model._environment_invariants(features))
    # First two atoms are neighbors; the third is isolated inside the cutoff.
    assert torch.linalg.vector_norm(delta[0] - delta[1]) < torch.linalg.vector_norm(delta[0] - delta[2])
    assert not torch.allclose(delta[0], delta[2], atol=1e-6)


def test_env_form_factor_is_permutation_covariant():
    torch.manual_seed(3)
    lattice = np.eye(3) * 8.0
    frac = np.array([[0.1, 0.2, 0.3], [0.4, 0.15, 0.2], [0.7, 0.6, 0.55]])
    packed = pack_structure(Structure(lattice, ("Si", "Si", "Si"), frac, metadata={"n_occupied": 6}))
    model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", env_isotropic=True)
    with torch.no_grad():
        model.env_radial.net[-1].weight.normal_(0.0, 0.15)
    k = torch.zeros(1, 3)
    cache = model.make_pw_cache(packed, k)
    features = model.encode(packed, edge_index=cache.edge_index, cell_shifts=cache.cell_shifts)
    form = model.environment_form_factor(packed, cache.unique_cart, features)

    perm = torch.tensor([2, 0, 1])
    packed_p = pack_structure(Structure(lattice, ("Si", "Si", "Si"), frac[perm.numpy()], metadata={"n_occupied": 6}))
    cache_p = model.make_pw_cache(packed_p, k)
    features_p = model.encode(packed_p, edge_index=cache_p.edge_index, cell_shifts=cache_p.cell_shifts)
    form_p = model.environment_form_factor(packed_p, cache_p.unique_cart, features_p)
    torch.testing.assert_close(form_p, form, atol=2e-5, rtol=2e-5)


def test_env_isotropic_loads_species_checkpoint():
    torch.manual_seed(4)
    species = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", env_isotropic=False)
    env = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", env_isotropic=True)
    load_gated_si_epm_state(env, species.state_dict())
    packed = pack_structure(_diamond())
    k = torch.tensor([[0.0, 0.0, 0.0]])
    with torch.no_grad():
        e0 = species.band_path(packed, k, n_bands=8)
        e1 = env.band_path(packed, k, n_bands=8)
    torch.testing.assert_close(e1, e0, atol=1e-6, rtol=1e-6)
    env.band_path(packed, k, n_bands=8).sum().backward()
    assert env.env_radial.net[0].weight.grad is not None
    assert torch.isfinite(env.env_radial.net[0].weight.grad).all()


def test_smooth_coordination_isolated_and_dimer():
    lattice = torch.eye(3) * 20.0
    isolated = torch.tensor([[0.0, 0.0, 0.0]])
    cn = smooth_coordination(isolated, lattice, (3.0, 10.0))
    assert cn.shape == (1, 2)
    assert torch.count_nonzero(cn) == 0

    dimer = torch.tensor([[0.0, 0.0, 0.0], [2.35, 0.0, 0.0]])
    cn = smooth_coordination(dimer, lattice, (3.0, 10.0))
    assert torch.all(cn[:, 0] > 0.2)
    torch.testing.assert_close(cn[0], cn[1], atol=1e-6, rtol=1e-6)


def test_smooth_coordination_diamond_first_shell():
    packed = pack_structure(_diamond())
    cn = smooth_coordination(packed.positions, packed.lattice, (3.0, 10.0))
    torch.testing.assert_close(cn[0], cn[1], atol=1e-5, rtol=1e-5)
    assert 3.5 < float(cn[0, 0]) < 4.5
    assert float(cn[0, 1]) > float(cn[0, 0])


def test_smooth_coordination_open_cell_has_fewer_10a_neighbors():
    compact = pack_structure(
        Structure(np.eye(3) * 2.8, ("Si",), np.zeros((1, 3)), metadata={"n_occupied": 2})
    )
    open_cell = pack_structure(
        Structure(np.eye(3) * 12.0, ("Si",), np.zeros((1, 3)), metadata={"n_occupied": 2})
    )
    cn_c = smooth_coordination(compact.positions, compact.lattice, (3.0, 10.0))
    cn_o = smooth_coordination(open_cell.positions, open_cell.lattice, (3.0, 10.0))
    assert float(cn_c[0, 1]) > float(cn_o[0, 1]) + 1.0


def test_env_density_zero_init_matches_species_and_loads():
    torch.manual_seed(5)
    packed = pack_structure(_diamond())
    k = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    species = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free")
    env = build_gated_si_epm(
        channels=4, ecut_ry=4.0, monopole_mode="free", env_isotropic=True, env_density=True
    )
    load_gated_si_epm_state(env, species.state_dict())
    cache = species.make_pw_cache(packed, k)
    with torch.no_grad():
        e0 = species.band_path(packed, k, n_bands=8, cache=cache)
        e1 = env.band_path(packed, k, n_bands=8, cache=cache)
    torch.testing.assert_close(e1, e0, atol=1e-6, rtol=1e-6)
    assert env.env_radial.in_features == 4 * 3 + 2
    env.band_path(packed, k, n_bands=8).sum().backward()
    assert env.env_radial.net[0].weight.grad is not None


def test_env_density_features_change_when_volume_changes():
    torch.manual_seed(6)
    packed = pack_structure(_diamond())
    model = build_gated_si_epm(
        channels=4, ecut_ry=4.0, monopole_mode="free", env_isotropic=True, env_density=True
    )
    features = model.encode(packed)
    inv = model._environment_invariants(features, packed)
    packed_open = pack_structure(_diamond())
    packed_open.lattice = packed_open.lattice * 1.6
    packed_open.positions = packed_open.fractional_positions @ packed_open.lattice
    packed_open.volume = packed_open.volume * (1.6**3)
    features_open = model.encode(packed_open)
    inv_open = model._environment_invariants(features_open, packed_open)
    assert inv.shape[-1] == inv_open.shape[-1] == 4 * 3 + 2
    assert float((inv[:, -1] - inv_open[:, -1]).abs().mean()) > 0.5


def test_packing_gate_on_for_cages_off_for_compact():
    cage = packing_gate(torch.tensor([130.6, 140.0]))
    compact = packing_gate(torch.tensor([144.2, 151.1, 159.4]))
    assert torch.all(cage > 0.95)
    assert torch.all(compact < 0.05)


def test_pack_gate_requires_density():
    try:
        build_gated_si_epm(env_isotropic=True, env_pack_gate=True)
    except ValueError as exc:
        assert "env_density" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_pack_gate_silences_diamond_and_keeps_open_cell():
    torch.manual_seed(7)
    diamond = pack_structure(_diamond())
    from epnet.data.open_frameworks import clathrate_ii_primitive

    cage = pack_structure(clathrate_ii_primitive())
    model = build_gated_si_epm(
        channels=4,
        ecut_ry=4.0,
        monopole_mode="free",
        env_isotropic=True,
        env_density=True,
        env_pack_gate=True,
    )
    with torch.no_grad():
        model.env_radial.net[-1].weight.normal_(0.0, 0.3)
        model.env_radial.net[-1].bias.normal_(0.0, 0.05)
    feat_d = model.encode(diamond)
    feat_c = model.encode(cage)
    gvec = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ff_d = model.environment_form_factor(diamond, gvec, feat_d)
    gate_d = float(model._last_pack_gate.mean())
    ff_c = model.environment_form_factor(cage, gvec, feat_c)
    gate_c = float(model._last_pack_gate.mean())
    assert gate_d < 0.05
    assert gate_c > 0.95
    assert float(ff_d.abs().mean()) < 0.05 * float(ff_c.abs().mean() + 1e-6)


def _gated_residual_model():
    return build_gated_si_epm(
        channels=4,
        ecut_ry=4.0,
        monopole_mode="free",
        env_isotropic=True,
        env_density=True,
        env_pack_gate=True,
        residual_pack_gate=True,
    )


def test_residual_pack_gate_requires_density():
    try:
        build_gated_si_epm(env_isotropic=True, residual_pack_gate=True)
    except ValueError as exc:
        assert "env_density" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_residual_pack_gate_holds_diamond_and_follows_live_on_cage():
    torch.manual_seed(8)
    diamond = pack_structure(_diamond())
    from epnet.data.open_frameworks import clathrate_ii_primitive

    cage = pack_structure(clathrate_ii_primitive())
    model = _gated_residual_model()
    gvec = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with torch.no_grad():
        ff_d0 = model.residual_form_factor(diamond, gvec)
        ff_c0 = model.residual_form_factor(cage, gvec)
        for layer in model.readout.channel_readout:
            layer.weight.normal_(0.0, 1.0)
        ff_d1 = model.residual_form_factor(diamond, gvec)
        ff_c1 = model.residual_form_factor(cage, gvec)
    torch.testing.assert_close(ff_d1, ff_d0, atol=1e-5, rtol=1e-5)
    assert float((ff_c1 - ff_c0).abs().mean()) > 1e-3
    assert float(model._last_pack_gate.mean()) > 0.95


def test_residual_pack_gate_gradients_come_from_cages_only():
    torch.manual_seed(9)
    diamond = pack_structure(_diamond())
    from epnet.data.open_frameworks import clathrate_ii_primitive

    cage = pack_structure(clathrate_ii_primitive())
    model = _gated_residual_model()
    gvec = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model.zero_grad(set_to_none=True)
    model.residual_form_factor(diamond, gvec).abs().sum().backward()
    diamond_grad = torch.stack(
        [p.grad.detach().abs().max() for p in model.readout.parameters() if p.grad is not None]
    ).max()
    model.zero_grad(set_to_none=True)
    model.residual_form_factor(cage, gvec).abs().sum().backward()
    cage_grad = torch.stack(
        [p.grad.detach().abs().max() for p in model.readout.parameters() if p.grad is not None]
    ).max()
    assert float(diamond_grad) < 1e-5
    assert float(cage_grad) > 1e-4
    assert all(not p.requires_grad for p in model.readout_frozen.parameters())


def test_residual_pack_gate_snapshots_when_loading_ungated_state():
    torch.manual_seed(10)
    src = build_gated_si_epm(
        channels=4,
        ecut_ry=4.0,
        monopole_mode="free",
        env_isotropic=True,
        env_density=True,
        env_pack_gate=True,
    )
    dst = _gated_residual_model()
    load_gated_si_epm_state(dst, src.state_dict())
    for p_live, p_frozen in zip(dst.readout.parameters(), dst.readout_frozen.parameters()):
        torch.testing.assert_close(p_frozen, p_live, atol=0, rtol=0)
        assert not p_frozen.requires_grad


def test_select_valid_edges_matches_boolean_mask():
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    displacement = torch.randn(4, 3)
    distance = torch.tensor([0.4, 0.0, 3.5, 1.2])
    valid = (distance > 1e-8) & (distance < 2.0)
    got_e, got_d, got_r = EquivariantEncoder._select_valid_edges(
        edge_index, displacement, distance, valid
    )
    torch.testing.assert_close(got_e, edge_index[:, valid])
    torch.testing.assert_close(got_d, displacement[valid])
    torch.testing.assert_close(got_r, distance[valid])


def test_encoder_backward_with_cached_graph():
    torch.manual_seed(6)
    a = 5.431
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.25 * a, 0.25 * a, 0.25 * a]])
    cell = 0.5 * a * torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    species = torch.zeros(2, dtype=torch.long)
    encoder = EquivariantEncoder(1, channels=4, cutoff=3.2, num_radial=4)
    edge_index, shifts = build_periodic_neighbor_graph(positions, cell, encoder.cutoff)
    out = encoder(positions, species, cell, edge_index=edge_index, cell_shifts=shifts)
    loss = out["scalar"].pow(2).sum() + out["vector"].pow(2).sum()
    loss.backward()
    grads = [p.grad.abs().sum() for p in encoder.parameters() if p.grad is not None]
    assert grads and float(torch.stack(grads).sum()) > 0.0


def test_store_graph_caches_neighbor_list_without_env():
    packed = pack_structure(_diamond())
    k = torch.tensor([[0.0, 0.0, 0.0]])
    model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free")
    bare = model.make_pw_cache(packed, k)
    cached = model.make_pw_cache(packed, k, store_graph=True)
    assert bare.edge_index is None
    assert cached.edge_index is not None
    with torch.no_grad():
        e0 = model.band_path(packed, k, n_bands=8, cache=bare, residual=True)
        e1 = model.band_path(packed, k, n_bands=8, cache=cached, residual=True)
    torch.testing.assert_close(e0, e1, atol=1e-6, rtol=1e-6)


def test_zero_residual_readout_is_h_neutral():
    torch.manual_seed(11)
    packed = pack_structure(_diamond())
    k = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    model = _gated_residual_model()
    model.zero_residual_readout()
    model.snapshot_frozen_residual()
    gvec = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    cache = model.make_pw_cache(packed, k)
    with torch.no_grad():
        ff = model.residual_form_factor(packed, gvec)
        e_res = model.band_path(packed, k, n_bands=8, cache=cache, residual=True)
        e_sph = model.band_path(packed, k, n_bands=8, cache=cache, residual=False)
    assert float(ff.abs().max()) < 1e-6
    torch.testing.assert_close(e_res, e_sph, atol=1e-6, rtol=1e-6)
