"""Space-group diagnostics for scalar reciprocal-space form factors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SpaceGroupAudit:
    """Residual and projection diagnostics on a symmetry-closed q set."""

    relative_rms_residual: float
    maximum_absolute_residual: float
    projection_relative_residual: float
    projection_retained_norm: float
    maximum_q_closure_error: float
    n_operations: int


@dataclass(frozen=True)
class SpaceGroupAction:
    """Permutation and phase representation of a space group on q vectors."""

    indices: NDArray[np.int64]
    phases: NDArray[np.complex128]
    maximum_q_closure_error: float


def atomic_structure_factor(
    q_vectors: ArrayLike,
    cartesian_positions: ArrayLike,
    atom_weights: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Return ``sum_a w_a exp(-i q.r_a)`` for every Cartesian q vector."""
    q = np.asarray(q_vectors, dtype=float)
    positions = np.asarray(cartesian_positions, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q_vectors must have shape (n_q, 3)")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("cartesian_positions must have shape (n_atoms, 3)")
    weights = (
        np.ones(len(positions), dtype=float)
        if atom_weights is None
        else np.asarray(atom_weights, dtype=float)
    )
    if weights.shape != (len(positions),):
        raise ValueError("atom_weights must match cartesian_positions")
    return np.exp(-1j * (q @ positions.T)) @ weights


def cartesian_rotations(
    lattice: ArrayLike, fractional_rotations: ArrayLike
) -> NDArray[np.float64]:
    """Convert spglib fractional rotations to Cartesian column rotations."""
    cell = np.asarray(lattice, dtype=float)
    rotations = np.asarray(fractional_rotations, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("lattice must have shape (3, 3)")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("fractional_rotations must have shape (n_ops, 3, 3)")
    inverse_transpose = np.linalg.inv(cell.T)
    return np.stack([cell.T @ rotation @ inverse_transpose for rotation in rotations])


def scalar_space_group_action(
    q_vectors: ArrayLike,
    lattice: ArrayLike,
    fractional_rotations: ArrayLike,
    fractional_translations: ArrayLike,
    *,
    q_tolerance: float = 2e-4,
) -> SpaceGroupAction:
    """Build the q permutation and nonsymmorphic phase for a scalar field."""
    q = np.asarray(q_vectors, dtype=float)
    cell = np.asarray(lattice, dtype=float)
    rotations = np.asarray(fractional_rotations, dtype=float)
    translations = np.asarray(fractional_translations, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q_vectors must have shape (n_q, 3)")
    if translations.shape != (len(rotations), 3):
        raise ValueError("fractional_translations must match rotations")
    if q_tolerance <= 0:
        raise ValueError("q_tolerance must be positive")
    cart_rotations = cartesian_rotations(cell, rotations)
    cart_translations = translations @ cell
    tree = cKDTree(q)
    all_indices = []
    all_phases = []
    maximum_closure = 0.0
    for rotation, translation in zip(cart_rotations, cart_translations):
        transformed_q = q @ rotation.T
        distances, indices = tree.query(transformed_q, k=1)
        maximum_closure = max(maximum_closure, float(np.max(distances)))
        if np.any(distances > q_tolerance):
            raise ValueError(
                "q set is not closed under the supplied space group: "
                f"maximum nearest distance {float(np.max(distances)):.3e}"
            )
        all_indices.append(indices)
        all_phases.append(
            np.exp(-1j * np.einsum("ij,j->i", transformed_q, translation))
        )
    indices = np.asarray(all_indices, dtype=np.int64)
    phases = np.asarray(all_phases, dtype=np.complex128)
    for operation_indices in indices:
        if len(np.unique(operation_indices)) != len(q):
            raise RuntimeError("a space-group operation did not permute the q set")
    return SpaceGroupAction(indices, phases, maximum_closure)


def audit_scalar_form_factor(
    q_vectors: ArrayLike,
    values: ArrayLike,
    lattice: ArrayLike,
    fractional_rotations: ArrayLike,
    fractional_translations: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    q_tolerance: float = 2e-4,
) -> SpaceGroupAudit:
    r"""Audit ``V(Cq)=exp[-i(Cq).t]V(q)`` and its group projection.

    The lattice vectors and q vectors are Cartesian row vectors.  Fractional
    operations follow spglib's ``x' = R x + t`` column-vector convention.
    """
    q = np.asarray(q_vectors, dtype=float)
    form_factor = np.asarray(values, dtype=np.complex128)
    cell = np.asarray(lattice, dtype=float)
    rotations = np.asarray(fractional_rotations, dtype=float)
    translations = np.asarray(fractional_translations, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q_vectors must have shape (n_q, 3)")
    if form_factor.shape != (len(q),):
        raise ValueError("values must match q_vectors")
    if translations.shape != (len(rotations), 3):
        raise ValueError("fractional_translations must match rotations")
    if q_tolerance <= 0:
        raise ValueError("q_tolerance must be positive")
    point_weights = (
        np.ones(len(q), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if point_weights.shape != (len(q),) or np.any(point_weights < 0):
        raise ValueError("weights must be a non-negative vector matching q_vectors")
    if not np.any(point_weights > 0):
        raise ValueError("at least one weight must be positive")

    action = scalar_space_group_action(
        q,
        cell,
        rotations,
        translations,
        q_tolerance=q_tolerance,
    )
    projected = np.zeros_like(form_factor)
    counts = np.zeros(len(q), dtype=np.int64)
    residual_squared = 0.0
    reference_squared = 0.0
    maximum_residual = 0.0
    for indices, phase in zip(action.indices, action.phases):
        expected = phase * form_factor
        residual = form_factor[indices] - expected
        mapped_weights = point_weights[indices]
        residual_squared += float(np.sum(mapped_weights * np.abs(residual) ** 2))
        reference_squared += float(np.sum(mapped_weights * np.abs(expected) ** 2))
        maximum_residual = max(maximum_residual, float(np.max(np.abs(residual))))
        np.add.at(projected, indices, expected)
        np.add.at(counts, indices, 1)

    if np.any(counts != len(rotations)):
        raise RuntimeError("space-group operations did not permute the q set")
    projected /= counts
    norm_squared = float(np.sum(point_weights * np.abs(form_factor) ** 2))
    projection_error_squared = float(
        np.sum(point_weights * np.abs(form_factor - projected) ** 2)
    )
    projection_norm_squared = float(np.sum(point_weights * np.abs(projected) ** 2))
    scale = max(norm_squared, np.finfo(float).tiny)
    return SpaceGroupAudit(
        relative_rms_residual=float(
            np.sqrt(residual_squared / max(reference_squared, np.finfo(float).tiny))
        ),
        maximum_absolute_residual=maximum_residual,
        projection_relative_residual=float(np.sqrt(projection_error_squared / scale)),
        projection_retained_norm=float(np.sqrt(projection_norm_squared / scale)),
        maximum_q_closure_error=action.maximum_q_closure_error,
        n_operations=int(len(rotations)),
    )
