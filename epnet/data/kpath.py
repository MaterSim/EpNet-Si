"""Correctness-preserving sampling helpers for line-mode band paths."""

from __future__ import annotations

import torch
from torch import Tensor


def strided_path_indices(
    kpoints: Tensor,
    stride: int,
    *,
    energies: Tensor | None = None,
    n_occupied: int | None = None,
    direction_tolerance: float = 1e-6,
) -> Tensor:
    """Return strided indices plus endpoints, geometric vertices, VBM, and CBM.

    Line-mode paths often contain segment junctions that do not land on a
    global ``range(0, n_k, stride)`` grid.  Dropping those points can change
    both the plotted path and the indirect gap used by the loss.
    """
    if kpoints.ndim != 2 or kpoints.shape[1] != 3 or kpoints.shape[0] < 1:
        raise ValueError("kpoints must have shape (n_k, 3) with n_k >= 1")
    step = max(int(stride), 1)
    n_k = int(kpoints.shape[0])
    keep = torch.zeros(n_k, dtype=torch.bool, device=kpoints.device)
    keep[::step] = True
    keep[0] = True
    keep[-1] = True

    if n_k > 2:
        delta = kpoints[1:] - kpoints[:-1]
        norm = torch.linalg.vector_norm(delta, dim=-1)
        prev, following = norm[:-1], norm[1:]
        denom = (prev * following).clamp_min(torch.finfo(kpoints.dtype).eps)
        cosine = (delta[:-1] * delta[1:]).sum(dim=-1) / denom
        vertex = (prev <= 1e-10) | (following <= 1e-10) | (cosine < 1.0 - direction_tolerance)
        keep[1:-1] |= vertex

    if energies is not None:
        if energies.ndim != 2 or energies.shape[0] != n_k:
            raise ValueError("energies must have shape (n_k, n_bands)")
        if n_occupied is None or not 0 < int(n_occupied) < energies.shape[1]:
            raise ValueError("n_occupied must identify valid VBM and CBM bands")
        n_occ = int(n_occupied)
        keep[int(torch.argmax(energies[:, n_occ - 1]))] = True
        keep[int(torch.argmin(energies[:, n_occ]))] = True

    return torch.nonzero(keep, as_tuple=False).flatten()


def virtual_band_window_margin(
    energies: Tensor,
    n_occupied: int,
    n_bands: int,
    *,
    half_width: float = 2.0,
) -> float:
    """Return the energy margin above the requested target window.

    A negative value means the first omitted band enters the target window.
    ``inf`` means no labeled bands were omitted.
    """
    if energies.ndim != 2 or not 0 < int(n_occupied) < energies.shape[1]:
        raise ValueError("invalid band array or occupied-band count")
    if not int(n_occupied) < int(n_bands) <= energies.shape[1]:
        raise ValueError("n_bands must include VBM and CBM and not exceed labels")
    if int(n_bands) == energies.shape[1]:
        return float("inf")
    n_occ = int(n_occupied)
    vbm = energies[:, n_occ - 1].max()
    cbm = energies[:, n_occ].min()
    midgap = 0.5 * (vbm + cbm)
    first_omitted = energies[:, int(n_bands)].min()
    return float(first_omitted - (midgap + float(half_width)))
