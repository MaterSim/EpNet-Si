"""Type-II Si clathrate prototype (the held-out open cage)."""

from __future__ import annotations

import numpy as np
from ase.spacegroup import Spacegroup
from pymatgen.core import Lattice, Structure as PMGStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from epnet.data.structures import Structure

# Literature Fd-3m origin-2 type-II Si-136 (Kasper / Dong–Sankey).
# Conventional a chosen so volume/Si matches type-I (10.2 Å, 46 atoms).
CLATHRATE_II_A = 14.62
CLATHRATE_II_SITES = (
    (0.125, 0.125, 0.125),
    (0.2176, 0.2176, 0.2176),
    (0.1825, 0.1825, 0.3703),
)


def _unique_frac(sites: list) -> np.ndarray:
    frac = np.mod(np.asarray(sites, dtype=float), 1.0)
    return np.unique(np.round(frac, 8), axis=0)


def clathrate_ii_conventional(a: float = CLATHRATE_II_A) -> PMGStructure:
    sg = Spacegroup(227, setting=2)
    sites: list = []
    for xyz in CLATHRATE_II_SITES:
        eq, _ = sg.equivalent_sites([list(xyz)])
        sites.extend(eq)
    frac = _unique_frac(sites)
    if len(frac) != 136:
        raise RuntimeError(f"expected 136 conventional sites, got {len(frac)}")
    return PMGStructure(Lattice.cubic(float(a)), ["Si"] * len(frac), frac)


def clathrate_ii_primitive(a: float = CLATHRATE_II_A) -> Structure:
    """34-atom primitive cell. Occupied bands = 68."""
    conv = clathrate_ii_conventional(a)
    prim = SpacegroupAnalyzer(conv, symprec=1e-4).get_primitive_standard_structure()
    if prim.num_sites != 34:
        raise RuntimeError(f"expected 34-atom primitive, got {prim.num_sites}")
    return Structure(
        prim.lattice.matrix,
        tuple(prim.labels),
        prim.frac_coords,
        metadata={
            "family": "clathrate_ii",
            "prototype": "clathrate_ii",
            "space_group": "Fd-3m",
            "a_angstrom": float(a),
            "n_occupied": 68,
            "source": "Type-II Si-136 clathrate Fd-3m 8a+32e+96g, primitive 34-atom cell",
        },
    )


def hydrostatic_volume_strain(structure: Structure, volume_strain: float) -> Structure:
    """Isotropic volume change: lattice scaled by (1+ε)^{1/3}."""
    scale = (1.0 + float(volume_strain)) ** (1.0 / 3.0)
    meta = dict(structure.metadata)
    meta["volume_strain"] = float(volume_strain)
    return Structure(structure.lattice * scale, structure.species, structure.fractional_positions, metadata=meta)
