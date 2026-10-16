"""Numerically robust experiment metrics."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def band_mae(prediction, target, *, weights=None, energy_window=None, reference=0.0) -> float:
    """Weighted band MAE, optionally restricted to a target-energy window."""
    pred, ref = np.asarray(prediction, float), np.asarray(target, float)
    if pred.shape != ref.shape or pred.ndim != 2:
        raise ValueError("bands must be matching (kpoint, band) arrays")
    mask = np.ones(ref.shape, dtype=bool)
    if energy_window is not None:
        low, high = energy_window
        mask = (ref - reference >= low) & (ref - reference <= high)
    if not mask.any():
        raise ValueError("energy window selects no bands")
    w = np.ones(pred.shape[0]) if weights is None else np.asarray(weights, float)
    if w.shape != (pred.shape[0],) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive sum")
    expanded = np.broadcast_to(w[:, None], pred.shape)
    return float(np.sum(np.abs(pred - ref)[mask] * expanded[mask]) / expanded[mask].sum())


def band_gap(energies, n_occupied: int) -> float:
    """Indirect fundamental gap, clipped at zero for band overlap."""
    values = np.asarray(energies, float)
    if values.ndim != 2 or not 0 < n_occupied < values.shape[1]:
        raise ValueError("invalid band array or occupied-band count")
    return float(max(0.0, values[:, n_occupied].min() - values[:, n_occupied - 1].max()))


def fermi_proxy(energies, n_occupied: int) -> float:
    """Mid-point of max occupied and min unoccupied eigenvalues (path E_F proxy)."""
    values = np.asarray(energies, float)
    return float(0.5 * (values[:, n_occupied - 1].max() + values[:, n_occupied].min()))


def vbm_window_mae(prediction, target, n_occupied: int, *, half_width: float = 2.0) -> float:
    pred, tgt = np.asarray(prediction, float), np.asarray(target, float)
    aligned = pred - pred[:, n_occupied - 1].max() + tgt[:, n_occupied - 1].max()
    mid = 0.5 * (tgt[:, n_occupied - 1].max() + tgt[:, n_occupied].min())
    window = np.abs(tgt - mid) <= half_width
    window[:, n_occupied - 1] = True
    window[:, n_occupied] = True
    return float(np.mean(np.abs(aligned - tgt)[window])) if window.any() else float("nan")


def fermi_window_mae(prediction, target, n_occupied: int, *, half_width: float = 2.0) -> float:
    pred, tgt = np.asarray(prediction, float), np.asarray(target, float)
    aligned = pred - fermi_proxy(pred, n_occupied) + fermi_proxy(tgt, n_occupied)
    window = np.abs(tgt - fermi_proxy(tgt, n_occupied)) <= half_width
    return float(np.mean(np.abs(aligned - tgt)[window])) if window.any() else float("nan")


def mixed_window_mae(prediction, target, n_occupied: int, gapped: bool, *, half_width: float = 2.0) -> float:
    """VBM window for gapped cells, Fermi window for metals."""
    if gapped:
        return vbm_window_mae(prediction, target, n_occupied, half_width=half_width)
    return fermi_window_mae(prediction, target, n_occupied, half_width=half_width)


def path_cartesian_k(k_frac, lattice) -> np.ndarray:
    """Cartesian k (1/Å) of fractional path points; ``lattice`` has row lattice vectors in Å."""
    recip = 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, float)).T
    return np.asarray(k_frac, float) @ recip


def path_band_slopes(target, k_cart, *, break_factor: float = 4.0) -> np.ndarray:
    """|dE/dk| (eV·Å) of each reference band along a k-path, segment by segment.

    A step longer than ``break_factor`` times the median step, or of zero length (repeated
    corner points), ends a segment, so slopes never span a path discontinuity. Points in
    segments shorter than two steps get NaN.
    """
    e = np.asarray(target, float)
    step = np.linalg.norm(np.diff(np.asarray(k_cart, float), axis=0), axis=1)
    typical = np.median(step[step > 0]) if np.any(step > 0) else 0.0
    cut = (step <= 1e-9) | (step > break_factor * typical)
    slopes = np.full(e.shape, np.nan)
    start = 0
    for end in [*(np.nonzero(cut)[0] + 1), len(e)]:
        if end - start >= 3:
            s = np.cumsum(np.concatenate([[0.0], step[start : end - 1]]))
            slopes[start:end] = np.abs(np.gradient(e[start:end], s, axis=0))
        start = end
    return slopes


def slope_aware_window_mae(
    prediction,
    target,
    n_occupied: int,
    gapped: bool,
    k_cart,
    *,
    v0: float = 1.0,
    half_width: float = 2.0,
) -> dict[str, float]:
    """Band error measured as distance between curves rather than vertically at fixed k.

    Each error is scaled by 1/sqrt(1 + (v/v0)^2), with v the reference band slope (eV·Å):
    flat bands keep their vertical error; on a steep band the error becomes v0 times the k
    offset, so 1 meV corresponds to 0.001/v0 Å^-1. Gapped cells stay VBM-aligned (as in
    ``vbm_window_mae``). Metals are aligned by the median offset in the window, since the
    path midpoint ``fermi_proxy`` is not a Fermi level. Also returns the plain vertical MAE
    under the same alignment and the median k offset of the steep points.
    """
    pred, tgt = np.asarray(prediction, float), np.asarray(target, float)
    if gapped:
        pred = pred - pred[:, n_occupied - 1].max() + tgt[:, n_occupied - 1].max()
        mid = 0.5 * (tgt[:, n_occupied - 1].max() + tgt[:, n_occupied].min())
        window = np.abs(tgt - mid) <= half_width
        window[:, n_occupied - 1] = True
        window[:, n_occupied] = True
    else:
        window = np.abs(tgt - fermi_proxy(tgt, n_occupied)) <= half_width
    slope = path_band_slopes(tgt, k_cart)
    window &= np.isfinite(slope)
    if not window.any():
        return {"slope_mae_eV": float("nan"), "aligned_mae_eV": float("nan"), "median_dk_invA": float("nan"), "align_offset_eV": float("nan")}
    offset = 0.0 if gapped else float(np.median((pred - tgt)[window]))
    err = np.abs(pred - offset - tgt)
    scaled = err / np.sqrt(1.0 + (slope / float(v0)) ** 2)
    steep = window & (slope > float(v0))
    return {
        "slope_mae_eV": float(np.mean(scaled[window])),
        "aligned_mae_eV": float(np.mean(err[window])),
        "median_dk_invA": float(np.median(err[steep] / slope[steep])) if steep.any() else float("nan"),
        "align_offset_eV": offset,
    }


def training_window_mae(
    prediction,
    target,
    n_occupied: int,
    gapped: bool,
    *,
    half_width: float = 2.0,
    edge_weight: float = 5.0,
) -> float:
    """NumPy equivalent of the aligned window term used during training."""
    pred, tgt = np.asarray(prediction, float), np.asarray(target, float)
    if pred.shape != tgt.shape or pred.ndim != 2 or not 0 < n_occupied < pred.shape[1]:
        raise ValueError("invalid band arrays or occupied-band count")
    pred_vbm = pred[:, n_occupied - 1].max()
    tgt_vbm = tgt[:, n_occupied - 1].max()
    tgt_cbm = tgt[:, n_occupied].min()
    if gapped:
        aligned = pred - pred_vbm + tgt_vbm
        reference = 0.5 * (tgt_vbm + tgt_cbm)
    else:
        pred_ef = 0.5 * (pred_vbm + pred[:, n_occupied].min())
        tgt_ef = 0.5 * (tgt_vbm + tgt_cbm)
        aligned = pred - pred_ef + tgt_ef
        reference = tgt_ef
    mask = np.abs(tgt - reference) <= half_width
    if gapped:
        mask[:, n_occupied - 1] = True
        mask[:, n_occupied] = True
    weight = np.ones_like(tgt)
    if gapped:
        weight[:, n_occupied - 1] = edge_weight
        weight[:, n_occupied] = edge_weight
    weight *= mask
    denominator = max(float(weight.sum()), 1.0)
    return float((np.abs(aligned - tgt) * weight).sum() / denominator)


def gap_error(prediction, target, n_occupied: int, *, absolute: bool = True) -> float:
    error = band_gap(prediction, n_occupied) - band_gap(target, n_occupied)
    return abs(error) if absolute else error


def subspace_distance(predicted_vectors, target_vectors, *, normalize=True) -> float:
    """Gauge/degeneracy-invariant Frobenius distance between projectors."""
    u, v = np.asarray(predicted_vectors), np.asarray(target_vectors)
    if u.shape != v.shape or u.ndim < 2:
        raise ValueError("eigenvector arrays must have matching (...,basis,state) shapes")
    overlap = np.matmul(np.swapaxes(u.conj(), -1, -2), v)
    rank = u.shape[-1]
    distance2 = np.maximum(0.0, 2.0 * rank - 2.0 * np.sum(np.abs(overlap) ** 2, axis=(-2, -1)))
    distance = np.sqrt(distance2 / (2.0 * rank) if normalize else distance2)
    return float(np.mean(distance))


def equivariance_error(reference, transformed, *, scale_floor=1e-12) -> dict[str, float]:
    """Absolute and relative max residual after applying the expected transform."""
    ref, value = np.asarray(reference), np.asarray(transformed)
    if ref.shape != value.shape:
        raise ValueError("arrays must have matching shapes")
    absolute = float(np.max(np.abs(value - ref), initial=0.0))
    scale = max(float(np.max(np.abs(ref), initial=0.0)), float(scale_floor))
    return {"absolute": absolute, "relative": absolute / scale}


def benchmark(function: Callable[[], object], *, warmup=1, repeats=10) -> dict[str, float]:
    """Measure wall time with warmup and report robust summary statistics."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    for _ in range(warmup):
        function()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        elapsed.append(time.perf_counter() - start)
    values = np.asarray(elapsed)
    return {
        "median_seconds": float(np.median(values)),
        "mean_seconds": float(values.mean()),
        "min_seconds": float(values.min()),
        "repeats": repeats,
    }
