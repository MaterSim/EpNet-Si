import numpy as np

import torch

from epnet.eval import band_gap, band_mae, benchmark, equivariance_error, gap_error, subspace_distance, training_window_mae
from epnet.hamiltonian.torch_pw import fermi_aligned_window_loss, vbm_aligned_window_loss


def test_band_metrics_with_window_and_gap():
    target = np.array([[-2.0, -0.5, 1.0], [-1.8, -0.4, 0.8]])
    prediction = target + np.array([[9.0, 0.1, -0.2], [9.0, -0.1, 0.2]])
    assert np.isclose(band_mae(prediction, target, energy_window=(-0.6, 1.1)), 0.15)
    assert np.isclose(band_gap(target, 2), 1.2)
    assert gap_error(prediction, target, 2) >= 0


def test_subspace_metric_ignores_internal_gauge():
    reference = np.eye(4, dtype=complex)[:, :2]
    rotation = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    transformed = reference @ rotation
    assert subspace_distance(transformed, reference) < 2e-8
    assert subspace_distance(np.eye(4)[:, 2:], reference) == 1.0


def test_equivariance_and_timing_metrics():
    metric = equivariance_error(np.ones(3), np.ones(3) + 1e-8)
    assert metric["absolute"] > 0 and metric["relative"] > 0
    timing = benchmark(lambda: sum(range(10)), warmup=0, repeats=3)
    assert timing["repeats"] == 3 and timing["min_seconds"] >= 0


def test_reported_training_window_metric_matches_torch_loss():
    target = torch.tensor([[-2.2, -0.4, 0.9, 2.1], [-1.8, -0.5, 1.1, 2.4]])
    prediction = target + torch.tensor([[0.4, 0.2, -0.3, 0.1], [-0.2, -0.1, 0.4, -0.2]])
    gapped = training_window_mae(prediction.numpy(), target.numpy(), 2, True)
    metal = training_window_mae(prediction.numpy(), target.numpy(), 2, False)
    assert np.isclose(gapped, float(vbm_aligned_window_loss(prediction, target, 2)), atol=1e-7)
    assert np.isclose(metal, float(fermi_aligned_window_loss(prediction, target, 2)), atol=1e-7)


def test_wide_gap_still_scores_vbm_and_cbm():
    target = torch.tensor([[-5.0, 5.0], [-5.2, 5.3]])
    prediction = target + torch.tensor([[0.0, 0.4], [-0.2, -0.3]])
    loss = vbm_aligned_window_loss(prediction, target, 1)
    reported = training_window_mae(prediction.numpy(), target.numpy(), 1, True)
    assert float(loss) > 0
    assert np.isclose(reported, float(loss), atol=1e-7)
