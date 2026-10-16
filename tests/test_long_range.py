import math

import numpy as np
import torch

from epnet.data.elements import SpeciesTable
from epnet.models.si_epm import build_gated_si_epm, pack_structure
from epnet.physics import (
    neutralize_effective_charges,
    reciprocal_coulomb_potential,
)


def test_effective_charges_are_neutral_and_differentiable():
    charges = torch.tensor([1.2, -0.1, 0.4], dtype=torch.float64, requires_grad=True)
    neutral = neutralize_effective_charges(charges)

    torch.testing.assert_close(neutral.sum(), torch.zeros((), dtype=torch.float64))
    neutral.square().sum().backward()
    assert charges.grad is not None
    assert torch.isfinite(charges.grad).all()


def test_weighted_neutralization_distributes_only_the_correction():
    charges = torch.tensor([1.0, 2.0, -1.0])
    weights = torch.tensor([1.0, 0.0, 3.0])
    neutral = neutralize_effective_charges(charges, weights)

    torch.testing.assert_close(neutral.sum(), torch.tensor(0.0))
    torch.testing.assert_close(neutral[1], charges[1])


def test_reciprocal_potential_translation_and_conjugate_symmetry():
    positions = torch.tensor(
        [[0.13, 0.27, 0.39], [0.61, 0.42, 0.05]], dtype=torch.float64
    )
    charges = torch.tensor([0.8, -0.8], dtype=torch.float64)
    g = 2.0 * math.pi * torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [-1.0, -2.0, 1.0]],
        dtype=torch.float64,
    )

    potential = reciprocal_coulomb_potential(
        positions, charges, g, volume=1.0, gaussian_width=0.15
    )
    translated = reciprocal_coulomb_potential(
        positions + torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64),
        charges,
        g,
        volume=1.0,
        gaussian_width=0.15,
    )

    torch.testing.assert_close(potential, translated, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(potential[2], potential[1].conj())
    torch.testing.assert_close(potential[0], torch.zeros((), dtype=potential.dtype))


def test_zero_charges_give_zero_correction_and_gauge_is_explicit():
    positions = torch.rand(4, 3, dtype=torch.float64)
    charges = torch.zeros(4, dtype=torch.float64)
    g = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)

    potential = reciprocal_coulomb_potential(
        positions, charges, g, volume=8.0, g0_value=2.5
    )

    torch.testing.assert_close(potential[0], potential.new_tensor(2.5))
    torch.testing.assert_close(potential[1], potential.new_tensor(0.0))


def _diamond(species=("Si", "Si")):
    from epnet.data.structures import Structure

    a = 5.431
    lattice = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    frac = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    return Structure(lattice, species, frac, metadata={"n_occupied": 4})


def test_oxidation_init_keeps_elemental_si_neutral():
    from epnet.data.elements import oxidation_charges

    table = SpeciesTable.from_symbols(["Si"])
    packed = pack_structure(_diamond(), species_table=table)
    k = torch.tensor([[0.0, 0.0, 0.0]])
    sphere = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free")
    model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", long_range=True)
    model.load_state_dict(sphere.state_dict(), strict=False)
    with torch.no_grad():
        model.effective_charge.copy_(
            torch.tensor(oxidation_charges(model.species_symbols), dtype=model.effective_charge.dtype)
        )
    cache = sphere.make_pw_cache(packed, k)
    with torch.no_grad():
        ff = model.long_range_form_factor(packed, cache.unique_cart)
        e_lr = model.band_path(packed, k, n_bands=8, cache=cache, residual=False)
        e_sph = sphere.band_path(packed, k, n_bands=8, cache=cache, residual=False)
    assert float(ff.abs().max()) < 1e-6
    torch.testing.assert_close(e_lr, e_sph, atol=1e-6, rtol=1e-6)


def test_zero_init_long_range_is_h_neutral():
    table = SpeciesTable.from_symbols(["Si"])
    packed = pack_structure(_diamond(), species_table=table)
    k = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    sphere = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free")
    model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", long_range=True)
    model.load_state_dict(sphere.state_dict(), strict=False)
    cache = sphere.make_pw_cache(packed, k)
    with torch.no_grad():
        e0 = sphere.band_path(packed, k, n_bands=8, cache=cache, residual=False)
        e1 = model.band_path(packed, k, n_bands=8, cache=cache, residual=False)
    torch.testing.assert_close(e1, e0, atol=1e-6, rtol=1e-6)


def test_elemental_si_stays_neutral_for_any_species_z():
    table = SpeciesTable.from_symbols(["Si"])
    packed = pack_structure(_diamond(), species_table=table)
    k = torch.tensor([[0.0, 0.0, 0.0]])
    sphere = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free")
    model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", long_range=True)
    model.load_state_dict(sphere.state_dict(), strict=False)
    with torch.no_grad():
        model.effective_charge.fill_(3.7)
    cache = sphere.make_pw_cache(packed, k)
    with torch.no_grad():
        ff = model.long_range_form_factor(packed, cache.unique_cart)
        e_lr = model.band_path(packed, k, n_bands=8, cache=cache, residual=False)
        e_sph = sphere.band_path(packed, k, n_bands=8, cache=cache, residual=False)
    assert float(ff.abs().max()) < 1e-6
    torch.testing.assert_close(e_lr, e_sph, atol=1e-6, rtol=1e-6)


def test_binary_charges_move_bands_and_backprop():
    table = SpeciesTable.from_symbols(["Si", "O"])
    packed = pack_structure(_diamond(("Si", "O")), species_table=table)
    k = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    model = build_gated_si_epm(
        channels=4,
        ecut_ry=4.0,
        monopole_mode="free",
        long_range=True,
        n_species=2,
        species_symbols=table.symbols,
    )
    cache = model.make_pw_cache(packed, k)
    with torch.no_grad():
        e0 = model.band_path(packed, k, n_bands=8, cache=cache, residual=False)
        model.effective_charge.copy_(torch.tensor([1.2, -0.4]))
        e1 = model.band_path(packed, k, n_bands=8, cache=cache, residual=False)
    assert float((e1 - e0).abs().max()) > 1e-3
    loss = model.band_path(packed, k, n_bands=8, cache=cache, residual=False).sum()
    loss.backward()
    assert model.effective_charge.grad is not None
    assert float(model.effective_charge.grad.abs().sum()) > 0.0
