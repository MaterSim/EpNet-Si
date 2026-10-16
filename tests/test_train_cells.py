import copy

import numpy as np
import torch

from epnet.data.structures import Structure
from epnet.models.si_epm import build_gated_si_epm, pack_structure
from epnet.train_cells import (
    _record_row_stream,
    accumulate_cell_grads,
    partition_by_work,
    pin_cell_row,
    resolve_thread_plan,
    row_host_nbytes,
    weighted_cell_loss,
)


def _cell(scale: float, family: str):
    a = 5.431 * scale
    lattice = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    frac = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    packed = pack_structure(Structure(lattice, ("Si", "Si"), frac, metadata={"n_occupied": 4}))
    k = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
    return packed, k, family


def _rows(model):
    rows = []
    for scale, family, use_gap in ((1.0, "diamond", True), (0.96, "bc8", False), (1.04, "diamond", True)):
        packed, k, fam = _cell(scale, family)
        cache = model.make_pw_cache(packed, k)
        with torch.no_grad():
            target = model.band_path(packed, k, n_bands=8, cache=cache)
            target = target + 0.05 * torch.randn_like(target)
        rows.append(
            {
                "id": f"{fam}-{scale}",
                "family": fam,
                "split": "train",
                "use_gap": use_gap,
                "n_occ": 4,
                "n_bands": 8,
                "packed": packed,
                "k": k,
                "target": target,
                "cache": cache,
            }
        )
    return rows


def test_resolve_thread_plan_splits_slurm_cpus(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    workers, threads = resolve_thread_plan(4, 0)
    assert workers == 4
    assert threads == 8


def test_parallel_cell_grads_match_serial():
    torch.manual_seed(0)
    model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", env_isotropic=True)
    rows = _rows(model)
    family_count = {"diamond": 2, "bc8": 1}
    kwargs = dict(
        gap_weight=0.25,
        env_l2=1e-3,
        refresh_sphere=False,
        n_families=2,
        family_count=family_count,
    )

    serial = copy.deepcopy(model)
    serial.zero_grad(set_to_none=True)
    raw_serial = accumulate_cell_grads(serial, rows, cell_workers=1, epoch_id=1, **kwargs)

    parallel = copy.deepcopy(model)
    parallel.zero_grad(set_to_none=True)
    raw_parallel = accumulate_cell_grads(parallel, rows, cell_workers=2, epoch_id=1, **kwargs)

    assert abs(raw_serial - raw_parallel) < 1e-4
    serial_grads = [p.grad for p in serial.parameters() if p.grad is not None]
    parallel_grads = [p.grad for p in parallel.parameters() if p.grad is not None]
    assert serial_grads and len(serial_grads) == len(parallel_grads)
    for g0, g1 in zip(serial_grads, parallel_grads):
        torch.testing.assert_close(g1, g0, atol=2e-4, rtol=2e-4)


def test_weighted_cell_loss_family_balance():
    torch.manual_seed(1)
    model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free")
    rows = _rows(model)
    family_count = {"diamond": 2, "bc8": 1}
    weights = []
    for row in rows:
        weighted, raw = weighted_cell_loss(
            model,
            row,
            gap_weight=0.0,
            env_l2=0.0,
            refresh_sphere=False,
            n_families=2,
            family_count=family_count,
        )
        weights.append(float((weighted / raw).detach()))
    assert abs(weights[0] - 0.25) < 1e-6
    assert abs(weights[1] - 0.5) < 1e-6
    assert abs(weights[2] - 0.25) < 1e-6


def test_partition_by_work_covers_and_balances():
    work = [100, 10, 10, 10, 80, 5, 5, 40]
    assign = partition_by_work(work, 4)
    assert len(assign) == len(work)
    assert set(assign) <= {0, 1, 2, 3}
    assert all(rank in assign for rank in range(4))
    loads = [0, 0, 0, 0]
    for w, rank in zip(work, assign):
        loads[rank] += w
    assert max(loads) - min(loads) <= max(work)
    assert partition_by_work(work, 1) == [0] * len(work)


def test_pin_cell_row_keeps_host_tensors():
    torch.manual_seed(2)
    model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free")
    packed, k, family = _cell(1.0, "diamond")
    cache = model.make_pw_cache(packed, k)
    row = {"packed": packed, "cache": cache, "k": k, "target": torch.zeros(3, 8)}
    pinned = pin_cell_row(row)
    assert pinned["k"].device.type == "cpu"
    assert pinned["cache"].kinetic.device.type == "cpu"
    assert row_host_nbytes(pinned) > 0


def test_record_row_stream_skips_host_tensors():
    torch.manual_seed(3)
    model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free")
    packed, k, family = _cell(1.0, "diamond")
    cache = model.make_pw_cache(packed, k)
    row = {"packed": packed, "cache": cache, "k": k, "target": torch.zeros(3, 8)}
    if torch.cuda.is_available():
        _record_row_stream(row, torch.cuda.current_stream())
    else:
        class _Dummy:
            pass
        _record_row_stream(row, _Dummy())
    assert row["k"].device.type == "cpu"
