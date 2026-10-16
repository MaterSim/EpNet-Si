"""Ideal tetrahedral Si polytypes at the locked catalog's cubic lattice scale."""

from __future__ import annotations

import numpy as np

from .structures import Structure


def silicon_4h(a_cubic: float = 5.431) -> Structure:
    """Eight-atom 4H Si, ABCB bilayer stacking, with ideal equal Si-Si bonds.

    ABCB stacking: https://doi.org/10.1021/acs.nanolett.8b02816.
    The lattice is ideal, not relaxed or taken from that experiment.
    """
    if not np.isfinite(a_cubic) or a_cubic <= 0:
        raise ValueError("a_cubic must be finite and positive")
    a = a_cubic / np.sqrt(2)
    c = 4 * a_cubic / np.sqrt(3)
    lattice = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
    xy = [(0, 0), (1 / 3, 2 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3)]
    positions = [[x, y, j / 4 + dz] for j, (x, y) in enumerate(xy) for dz in (0, 3 / 16)]
    return Structure(lattice, ["Si"] * 8, positions, {
        "family": "si_4h", "prototype": "si_4h", "stacking": "ABCB",
        "a_angstrom": a, "c_angstrom": c, "a_cubic_angstrom": a_cubic,
        "source": "ideal 4H Si from cubic a=5.431 Angstrom; no relaxation",
        "parent_space_group": "P6_3/mmc", "n_occupied": 16,
    })


def silicon_4h_cells() -> list[tuple[str, Structure]]:
    """Parent + 6 volume strains + 2 uniaxial + shear + 4 seeded rattles."""
    parent = silicon_4h()
    cells = []

    def add(name, lattice, positions, **metadata):
        cells.append((name, Structure(lattice, parent.species, positions, {
            **parent.metadata, **metadata, "deformation": name,
            "split": "val" if name == "rattle_0" else "train",
        })))

    add("parent", parent.lattice, parent.fractional_positions)
    for eps in (-0.06, -0.04, -0.02, 0.02, 0.04, 0.06):
        name = f"hydro_{'m' if eps < 0 else 'p'}{abs(eps):.2f}"
        add(name, parent.lattice * (1 + eps) ** (1 / 3), parent.fractional_positions, volume_strain=eps)
    for axis in (0, 2):
        lattice = parent.lattice.copy()
        lattice[axis] *= 1.04
        add(f"uniaxial_a{axis}_p0.04", lattice, parent.fractional_positions, axis=axis, strain=0.04)
    strain = np.eye(3)
    strain[0, 1] = strain[1, 0] = 0.02
    add("shear_0.04", parent.lattice @ strain, parent.fractional_positions, strain=0.04)
    for i in range(4):
        seed = 7 + 17 * i
        delta = np.random.default_rng(seed).normal(0, 0.03, (8, 3))
        delta -= delta.mean(axis=0)
        add(f"rattle_{i}", parent.lattice,
            parent.fractional_positions + delta @ np.linalg.inv(parent.lattice),
            rattle_seed=seed, rattle_std_A=0.03)
    return cells
