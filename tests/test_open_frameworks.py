import numpy as np

from epnet.data.open_frameworks import clathrate_ii_conventional, clathrate_ii_primitive
from epnet.data.si_prototypes import HOLDOUT_FAMILIES, TRAIN_FAMILIES
from epnet.models.equivariant_encoder import (
    build_periodic_neighbor_graph,
    periodic_displacements,
    smooth_coordination,
)
from epnet.models.si_epm import pack_structure


def test_split_puts_type_i_in_train_and_type_ii_in_holdout():
    assert "clathrate_i" in TRAIN_FAMILIES
    assert "clathrate_ii" in HOLDOUT_FAMILIES
    assert "clathrate_i" not in HOLDOUT_FAMILIES


def test_clathrate_ii_is_open_tetrahedral_network():
    conv = clathrate_ii_conventional()
    assert conv.num_sites == 136
    prim = clathrate_ii_primitive()
    assert prim.n_atoms == 34
    packed = pack_structure(prim)
    cn = smooth_coordination(packed.positions, packed.lattice, (3.0, 10.0))
    edge_index, shifts = build_periodic_neighbor_graph(packed.positions, packed.lattice, 3.0)
    dist = np.linalg.norm(
        periodic_displacements(packed.positions, edge_index, packed.lattice, shifts).numpy(),
        axis=1,
    )
    assert 22.5 < prim.volume / prim.n_atoms < 23.5
    assert 3.7 < float(cn[:, 0].mean()) < 4.3
    assert float(cn[:, 1].mean()) < 140
    assert 2.25 < float(dist.min()) < 2.45
