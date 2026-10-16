import torch

from epnet.data.kpath import strided_path_indices, virtual_band_window_margin


def test_stride_keeps_endpoint_vertex_and_band_edges():
    k = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0],
         [0.4, 0.2, 0.0], [0.4, 0.4, 0.0], [0.4, 0.6, 0.0]]
    )
    energies = torch.tensor(
        [[-1.0, 1.5], [-0.8, 1.3], [-0.2, 1.1],
         [-0.5, 0.4], [-0.7, 0.9], [-0.9, 1.2]]
    )
    idx = strided_path_indices(k, 4, energies=energies, n_occupied=1)
    assert idx.tolist() == [0, 2, 3, 4, 5]


def test_virtual_band_window_margin_detects_missing_state():
    energies = torch.tensor([[-1.0, 0.5, 1.8, 3.1], [-0.8, 0.6, 2.8, 3.4]])
    assert virtual_band_window_margin(energies, 1, 2) < 0
    assert virtual_band_window_margin(energies, 1, 3) > 0
    assert virtual_band_window_margin(energies, 1, 4) == float("inf")
