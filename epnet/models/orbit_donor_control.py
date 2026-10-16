"""Symmetry-preserving fit-bank orbit-donor control for environment parents."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

import numpy as np
import spglib
import torch
from torch import Tensor


CONTROL_VERSION = "fit-bank-orbit-donor-sham-v1"
CONTROL_SEED = 17002
DESCRIPTOR_WIDTH = 589
CENTRAL_SPECIES_WIDTH = 45
HISTOGRAM_SLICE = slice(45, 585)
VOLUME_INDEX = 585
COORDINATION_SLICE = slice(586, 589)
LOCAL_WIDTH = 543
DEFAULT_SYMPREC = 1e-5
DEFAULT_ANGLE_TOLERANCE = -1.0
DEFAULT_EQUALITY_ATOL = 1e-7
DEFAULT_EQUALITY_RTOL = 1e-7
DEFAULT_QUANTUM = 1e-7


@dataclass
class ControlClass:
    """One merged target class that the control must keep indistinguishable."""

    atom_indices: tuple[int, ...]
    species: int
    multiplicity: int
    full_mean: Tensor
    local_mean: Tensor
    class_fingerprint: str


def local_channels(descriptor: Tensor) -> Tensor:
    """Extract the 540 radial-histogram and three coordination channels."""
    if descriptor.ndim != 2 or descriptor.shape[1] != DESCRIPTOR_WIDTH:
        raise ValueError(f"descriptor must have shape (N, {DESCRIPTOR_WIDTH})")
    return torch.cat(
        (descriptor[:, HISTOGRAM_SLICE], descriptor[:, COORDINATION_SLICE]), dim=1
    )


def with_donor_local_channels(descriptor: Tensor, donor_local: Tensor) -> Tensor:
    """Keep target species and volume while replacing only local channels."""
    if donor_local.shape != (len(descriptor), LOCAL_WIDTH):
        raise ValueError(f"donor local channels must have shape (N, {LOCAL_WIDTH})")
    result = descriptor.clone()
    result[:, HISTOGRAM_SLICE] = donor_local[:, :540].to(result)
    result[:, COORDINATION_SLICE] = donor_local[:, 540:].to(result)
    if not torch.equal(
        result[:, :CENTRAL_SPECIES_WIDTH], descriptor[:, :CENTRAL_SPECIES_WIDTH]
    ):
        raise AssertionError("control changed target central-species channels")
    if not torch.equal(result[:, VOLUME_INDEX], descriptor[:, VOLUME_INDEX]):
        raise AssertionError("control changed target log-volume channel")
    return result


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _quantized(value: Tensor, quantum: float) -> list[int]:
    if quantum <= 0:
        raise ValueError("descriptor quantum must be positive")
    scaled = torch.round(value.detach().cpu().double() / float(quantum))
    if not torch.isfinite(scaled).all():
        raise ValueError("cannot fingerprint non-finite descriptors")
    return scaled.to(torch.int64).tolist()


def _multiplicity_fraction(multiplicity: int, n_atoms: int) -> str:
    if multiplicity < 1 or n_atoms < multiplicity:
        raise ValueError("invalid control-class multiplicity")
    value = Fraction(int(multiplicity), int(n_atoms))
    return f"{value.numerator}/{value.denominator}"


def _union_find_groups(means: list[Tensor], species: list[int], atol: float, rtol: float) -> list[list[int]]:
    parent = list(range(len(means)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[max(root_first, root_second)] = min(root_first, root_second)

    for first in range(len(means)):
        for second in range(first + 1, len(means)):
            if species[first] != species[second]:
                continue
            if torch.allclose(means[first], means[second], atol=atol, rtol=rtol):
                union(first, second)
    groups: dict[int, list[int]] = {}
    for index in range(len(means)):
        groups.setdefault(find(index), []).append(index)
    return [groups[key] for key in sorted(groups)]


def control_classes(
    lattice: Tensor,
    fractional_positions: Tensor,
    species: Tensor,
    descriptor: Tensor,
    *,
    symprec: float = DEFAULT_SYMPREC,
    angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE,
    equality_atol: float = DEFAULT_EQUALITY_ATOL,
    equality_rtol: float = DEFAULT_EQUALITY_RTOL,
    quantum: float = DEFAULT_QUANTUM,
) -> tuple[str, list[ControlClass], dict[str, Any]]:
    """Build crystallographic classes, merging equal same-species descriptors."""
    lattice = torch.as_tensor(lattice).detach().cpu().double()
    fractional_positions = (
        torch.as_tensor(fractional_positions).detach().cpu().double().remainder(1.0)
    )
    species = torch.as_tensor(species).detach().cpu().long()
    descriptor = torch.as_tensor(descriptor).detach().cpu()
    n_atoms = int(len(species))
    if lattice.shape != (3, 3):
        raise ValueError("lattice must have shape (3, 3)")
    if fractional_positions.shape != (n_atoms, 3) or descriptor.shape != (
        n_atoms,
        DESCRIPTOR_WIDTH,
    ):
        raise ValueError("structure arrays have inconsistent shapes")
    if n_atoms < 1 or not torch.isfinite(lattice).all() or not torch.isfinite(
        fractional_positions
    ).all():
        raise ValueError("structure must be nonempty and finite")

    cell = (
        lattice.numpy(),
        fractional_positions.numpy(),
        (species.numpy() + 1).astype(np.int32),
    )
    dataset = spglib.get_symmetry_dataset(
        cell, symprec=float(symprec), angle_tolerance=float(angle_tolerance)
    )
    if dataset is None:
        raise RuntimeError("spglib could not determine crystallographic classes")
    equivalent = np.asarray(dataset.equivalent_atoms, dtype=np.int64)
    orbit_atoms = [
        tuple(int(index) for index in np.flatnonzero(equivalent == label))
        for label in sorted(set(equivalent.tolist()))
    ]
    local = local_channels(descriptor)
    orbit_species: list[int] = []
    orbit_means: list[Tensor] = []
    for atoms in orbit_atoms:
        values = {int(species[index]) for index in atoms}
        if len(values) != 1:
            raise ValueError("spglib class mixes central species")
        orbit_species.append(values.pop())
        orbit_means.append(local[list(atoms)].double().mean(dim=0))

    merged = _union_find_groups(
        orbit_means, orbit_species, float(equality_atol), float(equality_rtol)
    )
    classes: list[ControlClass] = []
    class_payloads: list[dict[str, Any]] = []
    for group in merged:
        atoms = tuple(sorted(index for orbit in group for index in orbit_atoms[orbit]))
        class_species = {int(species[index]) for index in atoms}
        if len(class_species) != 1:
            raise AssertionError("merged control class mixes species")
        full_mean = descriptor[list(atoms)].double().mean(dim=0)
        local_mean = local[list(atoms)].double().mean(dim=0)
        payload = {
            "species": class_species.pop(),
            "multiplicity_fraction": _multiplicity_fraction(len(atoms), n_atoms),
            "descriptor_quantized": _quantized(full_mean, float(quantum)),
        }
        fingerprint = _digest(payload)
        class_payloads.append(
            {
                "species": payload["species"],
                "multiplicity_fraction": payload["multiplicity_fraction"],
                "class_fingerprint": fingerprint,
            }
        )
        classes.append(
            ControlClass(
                atom_indices=atoms,
                species=int(payload["species"]),
                multiplicity=len(atoms),
                full_mean=full_mean.to(descriptor),
                local_mean=local_mean.to(descriptor),
                class_fingerprint=fingerprint,
            )
        )
    order = sorted(range(len(classes)), key=lambda index: classes[index].class_fingerprint)
    classes = [classes[index] for index in order]
    class_payloads = [class_payloads[index] for index in order]
    if len({value.class_fingerprint for value in classes}) != len(classes):
        raise ValueError("distinct control classes have a fingerprint collision")
    structure_fingerprint = _digest(class_payloads)
    audit = {
        "spglib_version": str(spglib.__version__),
        "space_group_number": int(dataset.number),
        "space_group_symbol": str(dataset.international),
        "atoms": n_atoms,
        "spglib_orbits": len(orbit_atoms),
        "control_classes": len(classes),
        "merged_equal_descriptor_orbits": len(orbit_atoms) - len(classes),
        "structure_fingerprint": structure_fingerprint,
    }
    return structure_fingerprint, classes, audit


def donor_records(structure_fingerprint: str, classes: Iterable[ControlClass]) -> list[dict[str, Any]]:
    """Convert control classes into target-free donor-bank records."""
    records = []
    for value in classes:
        records.append(
            {
                "donor_key": f"{structure_fingerprint}:{value.class_fingerprint}",
                "structure_fingerprint": structure_fingerprint,
                "class_fingerprint": value.class_fingerprint,
                "species": int(value.species),
                "multiplicity": int(value.multiplicity),
                "local_vector": value.local_mean.detach().cpu().float().clone(),
            }
        )
    return records


def _weighted_score(
    target_structure: str,
    target_class: str,
    donor_class: str,
    multiplicity: int,
    *,
    seed: int,
    version: str,
) -> float:
    payload = f"{version}:{seed}:{target_structure}:{target_class}:{donor_class}"
    integer = int.from_bytes(hashlib.sha256(payload.encode()).digest(), "big")
    unit = (integer + 1) / (2**256 + 1)
    return -math.log(unit) / float(multiplicity)


def assign_donors(
    structure_fingerprint: str,
    classes: list[ControlClass],
    bank_records: list[dict[str, Any]],
    *,
    seed: int = CONTROL_SEED,
    version: str = CONTROL_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select deterministic fit-bank donors without using IDs, sites, or cohorts."""
    by_species: dict[int, list[dict[str, Any]]] = {}
    for donor in bank_records:
        local = donor.get("local_vector")
        if not isinstance(local, Tensor) or local.shape != (LOCAL_WIDTH,):
            raise ValueError("donor bank has an invalid local vector")
        by_species.setdefault(int(donor["species"]), []).append(donor)
    for donors in by_species.values():
        donors.sort(key=lambda value: value["donor_key"])

    used: dict[int, set[str]] = {}
    assignments: list[dict[str, Any]] = []
    reused = 0
    self_candidates_removed = 0
    for target in sorted(classes, key=lambda value: value.class_fingerprint):
        donors = by_species.get(int(target.species), [])
        if not donors:
            raise ValueError(f"no donor classes for species {target.species}")
        nonself = [
            donor
            for donor in donors
            if donor["structure_fingerprint"] != structure_fingerprint
        ]
        if nonself:
            self_candidates_removed += len(donors) - len(nonself)
            donors = nonself
        ranked = sorted(
            donors,
            key=lambda donor: (
                _weighted_score(
                    structure_fingerprint,
                    target.class_fingerprint,
                    donor["class_fingerprint"],
                    int(donor["multiplicity"]),
                    seed=int(seed),
                    version=str(version),
                ),
                donor["donor_key"],
            ),
        )
        already = used.setdefault(int(target.species), set())
        available = [donor for donor in ranked if donor["donor_key"] not in already]
        if available:
            selected = available[0]
        else:
            selected = ranked[0]
            reused += 1
        already.add(selected["donor_key"])
        assignments.append(
            {
                "target_class_fingerprint": target.class_fingerprint,
                "target_species": int(target.species),
                "donor_key": selected["donor_key"],
                "donor_structure_fingerprint": selected["structure_fingerprint"],
                "donor_class_fingerprint": selected["class_fingerprint"],
                "donor_local_vector": selected["local_vector"],
                "self_donor": selected["structure_fingerprint"] == structure_fingerprint,
            }
        )
    audit = {
        "version": str(version),
        "seed": int(seed),
        "target_classes": len(classes),
        "donor_reuses_after_exhaustion": reused,
        "self_candidates_removed": self_candidates_removed,
        "self_donor_assignments": sum(value["self_donor"] for value in assignments),
    }
    return assignments, audit


def corrected_sham_descriptor(
    descriptor: Tensor,
    classes: list[ControlClass],
    assignments: list[dict[str, Any]],
) -> Tensor:
    """Broadcast one selected donor vector throughout every target control class."""
    if len(classes) != len(assignments):
        raise ValueError("control classes and donor assignments differ in length")
    by_target = {
        value["target_class_fingerprint"]: value for value in assignments
    }
    donor_local = torch.empty(
        (len(descriptor), LOCAL_WIDTH), dtype=descriptor.dtype, device=descriptor.device
    )
    covered = torch.zeros(len(descriptor), dtype=torch.bool)
    for target in classes:
        assignment = by_target.get(target.class_fingerprint)
        if assignment is None:
            raise ValueError("a target control class lacks a donor")
        vector = assignment["donor_local_vector"].to(descriptor)
        indices = torch.tensor(target.atom_indices, dtype=torch.long, device=descriptor.device)
        donor_local.index_copy_(0, indices, vector.expand(len(indices), -1))
        covered[indices.cpu()] = True
    if not bool(covered.all()):
        raise ValueError("donor assignment does not cover every target atom")
    return with_donor_local_channels(descriptor, donor_local)


def apply_orbit_donor_control(
    lattice: Tensor,
    fractional_positions: Tensor,
    species: Tensor,
    descriptor: Tensor,
    bank_records: list[dict[str, Any]],
) -> tuple[Tensor, dict[str, Any]]:
    """Apply the complete deterministic control to one structure."""
    structure_fingerprint, classes, class_audit = control_classes(
        lattice, fractional_positions, species, descriptor
    )
    assignments, assignment_audit = assign_donors(
        structure_fingerprint, classes, bank_records
    )
    sham = corrected_sham_descriptor(descriptor, classes, assignments)
    return sham, {
        "class_audit": class_audit,
        "assignment_audit": assignment_audit,
        "assignments": [
            {key: value for key, value in row.items() if key != "donor_local_vector"}
            for row in assignments
        ],
    }
