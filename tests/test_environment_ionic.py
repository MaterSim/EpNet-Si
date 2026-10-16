from types import SimpleNamespace

import torch

from epnet.data.elements import SpeciesTable
from epnet.data.structures import Structure
from epnet.models.environment_ionic import EnvironmentIonicCorrection
from epnet.models.si_epm import build_gated_si_epm, load_gated_si_epm_state, pack_structure


def _packed(dtype=torch.float64):
    lattice = torch.eye(3, dtype=dtype) * 6.0
    fractional = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=dtype)
    return SimpleNamespace(
        lattice=lattice,
        positions=fractional @ lattice,
        species=torch.tensor([0, 1]),
        volume=torch.tensor(216.0, dtype=dtype),
    )


def test_environment_ionic_has_84_parameters_for_phase3_species():
    model = EnvironmentIonicCorrection(21)
    assert sum(parameter.numel() for parameter in model.parameters()) == 84


def test_environment_ionic_zero_attach_and_exact_neutrality():
    model = EnvironmentIonicCorrection(2).double()
    packed = _packed()
    g = torch.tensor([[0.0, 0.0, 0.0], [1.1, -0.4, 0.2]], dtype=torch.float64)
    torch.testing.assert_close(model(packed, g), torch.zeros(2, dtype=torch.complex128))
    with torch.no_grad():
        model.coefficients.copy_(
            torch.tensor([[0.7, 0.1, -0.2, 0.3], [-0.4, 0.2, 0.1, -0.1]], dtype=torch.float64)
        )
    charges = model.charges(packed)
    torch.testing.assert_close(charges.sum(), torch.zeros((), dtype=torch.float64))
    assert float(charges.abs().sum()) > 0


def test_environment_ionic_translation_invariance_and_backprop():
    model = EnvironmentIonicCorrection(2).double()
    with torch.no_grad():
        model.coefficients.normal_(0.0, 0.2)
    packed = _packed()
    reciprocal = 2.0 * torch.pi * torch.linalg.inv(packed.lattice).transpose(0, 1)
    g = torch.tensor([[1, 0, 0], [-1, 0, 0]], dtype=torch.float64) @ reciprocal
    reference = model(packed, g)
    translated = _packed()
    translated.positions = translated.positions + translated.lattice[0]
    torch.testing.assert_close(model(translated, g), reference, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(reference[1], reference[0].conj(), atol=1e-10, rtol=1e-10)
    reference.real.sum().backward()
    assert model.coefficients.grad is not None
    assert torch.isfinite(model.coefficients.grad).all()


def test_phase3_arms_have_exactly_matched_capacity_and_zero_attach():
    symbols = (
        "Si", "Al", "As", "B", "Be", "Br", "C", "Ca", "Cl", "F", "Ga",
        "Ge", "K", "Li", "Mg", "N", "Na", "O", "P", "S", "Se",
    )
    common = dict(channels=8, n_species=21, species_symbols=symbols, monopole_mode="free")
    control = build_gated_si_epm(**common, sphere_n_basis=32)
    ionic = build_gated_si_epm(**common, sphere_n_basis=28, environment_ionic=True)
    projector = build_gated_si_epm(
        **common, sphere_n_basis=28, sp_nonlocal=True, sp_environment=True
    )
    combined = build_gated_si_epm(
        **common,
        sphere_n_basis=24,
        environment_ionic=True,
        sp_nonlocal=True,
        sp_environment=True,
    )
    for model in (control, ionic, projector, combined):
        model.set_sphere_trainable(True)
    counts = [sum(p.numel() for p in model.parameters() if p.requires_grad) for model in (control, ionic, projector, combined)]
    assert counts == [1912, 1912, 1912, 1912]

    structure = Structure(
        lattice=(torch.eye(3) * 5.0).numpy(),
        species=("Si", "Al"),
        fractional_positions=torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]).numpy(),
        metadata={"n_occupied": 2},
    )
    table = SpeciesTable.from_symbols(symbols)
    packed = pack_structure(structure, species_table=table)
    k = torch.tensor([[0.0, 0.0, 0.0]])
    parent = build_gated_si_epm(**common, sphere_n_basis=24, ecut_ry=2.0)
    child = build_gated_si_epm(
        **common,
        sphere_n_basis=24,
        ecut_ry=2.0,
        environment_ionic=True,
        sp_nonlocal=True,
        sp_environment=True,
    )
    load_gated_si_epm_state(child, parent.state_dict())
    expected = parent.band_path(packed, k, n_bands=3)
    got = child.band_path(packed, k, n_bands=3)
    torch.testing.assert_close(got, expected)
