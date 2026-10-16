import numpy as np
import torch

from epnet.data.structures import Structure
from epnet.models.multipole_form_factor import MultipoleFormFactor
from epnet.models.si_epm import build_gated_si_epm, pack_structure


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_odd_multipole_has_fourier_reality_and_survives_hermitian_assembly():
    readout = MultipoleFormFactor(channels=1, num_radial=3, l_max=1, include_monopole=False)
    with torch.no_grad():
        readout.channel_readout[1].weight.fill_(1.0)
    positions = torch.tensor([[0.17, -0.23, 0.31]])
    g = torch.tensor([[1.1, -0.4, 0.7], [-1.1, 0.4, -0.7]])
    features = {"vector": torch.tensor([[[0.8, -0.3, 0.5]]])}

    values = readout(positions, g, features)
    torch.testing.assert_close(values[1], values[0].conj(), atol=2e-6, rtol=2e-6)
    assert float(values.abs().min()) > 1e-5

    spherical = torch.zeros((2, 2), dtype=torch.complex64)
    inverse = torch.tensor([[0, 0], [1, 0]])
    from epnet.hamiltonian.torch_pw import add_multipole_potential

    assembled = add_multipole_potential(spherical, values, inverse, volume=1.0, q=g)
    assert float(assembled[0, 1].abs()) > 1e-5
    torch.testing.assert_close(assembled, assembled.mH, atol=2e-6, rtol=2e-6)


def test_non_scalar_biases_are_frozen_and_do_not_select_a_lab_frame():
    readout = MultipoleFormFactor(channels=2, num_radial=4, l_max=2, include_monopole=False)
    assert not readout.radial_bias[1].requires_grad
    assert not readout.radial_bias[2].requires_grad
    with torch.no_grad():
        readout.radial_bias[1].fill_(3.0)
        readout.radial_bias[2].fill_(5.0)
    positions = torch.zeros((1, 3))
    g = torch.tensor([[1.0, 0.2, -0.4]])
    features = {
        "vector": torch.zeros((1, 2, 3)),
        "tensor": torch.zeros((1, 2, 3, 3)),
    }
    torch.testing.assert_close(readout(positions, g, features), torch.zeros(1, dtype=torch.complex64))


def test_complete_residual_hamiltonian_is_rotation_invariant():
    torch.manual_seed(11)
    lattice = np.array([[4.8, 0.1, 0.0], [0.3, 5.1, 0.2], [0.2, 0.4, 5.4]])
    frac = np.array([[0.11, 0.23, 0.37], [0.61, 0.44, 0.18]])
    structure = Structure(lattice, ("Si", "Si"), frac, metadata={"n_occupied": 4})
    model = build_gated_si_epm(
        channels=3, cutoff=3.2, ecut_ry=2.0, monopole_mode="free", subset_ng_min=4096
    ).eval()
    with torch.no_grad():
        for layer in model.readout.channel_readout:
            layer.weight.normal_(0.0, 0.2)

    k = torch.tensor([[0.0, 0.0, 0.0], [0.17, 0.09, 0.04]])
    packed = pack_structure(structure)
    reference = model.band_path(packed, k, n_bands=6)

    rotation = _rotation(7).numpy()
    rotated_structure = Structure(lattice @ rotation.T, ("Si", "Si"), frac, metadata={"n_occupied": 4})
    rotated = model.band_path(pack_structure(rotated_structure), k, n_bands=6)
    torch.testing.assert_close(rotated, reference, atol=3e-5, rtol=3e-5)
