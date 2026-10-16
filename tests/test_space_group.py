import numpy as np

from epnet.eval.space_group import atomic_structure_factor, audit_scalar_form_factor


def inversion_fixture():
    lattice = np.eye(3)
    q = 2.0 * np.pi * np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    rotations = np.stack([np.eye(3, dtype=int), -np.eye(3, dtype=int)])
    translations = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    positions = np.array([[0.25, 0.0, 0.0]])
    return lattice, q, rotations, translations, positions


def test_atomic_structure_factor_obeys_shifted_inversion():
    lattice, q, rotations, translations, positions = inversion_fixture()
    values = atomic_structure_factor(q, positions)
    audit = audit_scalar_form_factor(q, values, lattice, rotations, translations)
    assert audit.relative_rms_residual < 1e-12
    assert audit.projection_relative_residual < 1e-12
    assert abs(audit.projection_retained_norm - 1.0) < 1e-12


def test_audit_detects_and_projects_forbidden_component():
    lattice, q, rotations, translations, _ = inversion_fixture()
    values = np.array([0.0, 1.0 + 0.0j, 1.0 + 0.0j])
    audit = audit_scalar_form_factor(
        q,
        values,
        lattice,
        rotations,
        translations,
        weights=np.array([0.0, 1.0, 1.0]),
    )
    assert audit.relative_rms_residual > 1.0
    assert audit.projection_relative_residual > 0.99
    assert audit.projection_retained_norm < 1e-12


def test_audit_rejects_nonclosed_q_set():
    lattice, q, rotations, translations, positions = inversion_fixture()
    values = atomic_structure_factor(q[:-1], positions)
    with np.testing.assert_raises_regex(ValueError, "not closed"):
        audit_scalar_form_factor(
            q[:-1], values, lattice, rotations, translations, q_tolerance=1e-8
        )
