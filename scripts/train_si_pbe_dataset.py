#!/usr/bin/env python3
"""Train GatedSiEPM jointly on the locked Si PBE catalog (no GW, no privileged parent)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epnet.data.manifest import NPZDataset  # noqa: E402
from epnet.eval.metrics import band_gap, fermi_proxy, fermi_window_mae, gap_error, mixed_window_mae, path_cartesian_k, slope_aware_window_mae, vbm_window_mae  # noqa: E402
from epnet.models.equivariant_encoder import format_gate_windows  # noqa: E402
from epnet.models.si_epm import (  # noqa: E402
    build_gated_si_epm,
    checkpoint_density_cutoffs,
    checkpoint_env_density,
    checkpoint_env_isotropic,
    checkpoint_env_pack_gate,
    checkpoint_pack_gate_alpha,
    checkpoint_pack_gate_center,
    checkpoint_pack_gate_windows,
    checkpoint_residual_anchor,
    checkpoint_quad_scale,
    checkpoint_residual_pack_gate,
    checkpoint_sp_nonlocal,
    checkpoint_sphere_density,
    load_gated_si_epm_state,
    model_payload_flags,
    normalize_gate_windows,
    pack_structure,
)
from epnet.train_cells import accumulate_cell_grads, prepare_rows_parallel, resolve_thread_plan  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/si_pbe_bands.json")
    p.add_argument("--output", type=Path, default=ROOT / "results/si_pbe_dataset_fit.json")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--ecut-ry", type=float, default=12.0)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--encoder-cutoff", type=float, default=None, help="Encoder neighbour cutoff in Å (default: checkpoint value or 5.0)")
    p.add_argument("--monopole-mode", choices=("gate", "off", "free"), default="gate")
    p.add_argument("--lr", type=float, default=3e-3, help="Residual (encoder/readout) learning rate")
    p.add_argument("--sphere-lr", type=float, default=1e-2, help="Species-sphere learning rate")
    p.add_argument("--freeze-sphere", action="store_true", help="Keep Cohen/init sphere frozen")
    p.add_argument("--gap-weight", type=float, default=0.25)
    p.add_argument(
        "--gap-mode",
        choices=("clamp", "signed"),
        default="clamp",
        help="clamp: predicted gap floored at 0 (no gradient once bands overlap); signed: keep the overlap depth",
    )
    p.add_argument("--metal-align", choices=("proxy", "median"), default="proxy", help="Metal loss alignment: E_F proxy (default) or median offset")
    p.add_argument("--density-weight", type=float, default=0.0, help="Weight of the valence-density term (0 = off; needs --density-targets)")
    p.add_argument("--density-every", type=int, default=4, help="Apply the density term on every N-th optimizer step")
    p.add_argument("--density-targets", type=Path, default=None, help="Directory of <id>.npz PBE density targets (scripts/prepare_density_targets.py)")
    p.add_argument("--density-gap-floor", type=float, default=0.05, help="Lower bound (eV) on occupied-empty gaps in the density gradient")
    p.add_argument("--deep-weight", type=float, default=0.0, help="Weight of the loss on occupied states below the +-2 eV window (0 = off)")
    p.add_argument("--k-stride", type=int, default=2, help="Use every Nth path k-point in the loss")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument(
        "--exclude-ids",
        nargs="*",
        default=None,
        help="Sample ids left out of the loss and scorecard (for example a cell with a defective label)",
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sphere-init", type=Path, default=None, help="Optional sphere checkpoint (not a crystal prefit)")
    p.add_argument("--model-init", type=Path, default=None, help="Resume full GatedSiEPM state_dict")
    p.add_argument(
        "--env-isotropic",
        action="store_true",
        help="Add per-atom cosine Δv(q) from encoder invariants (zero-init; train split only)",
    )
    p.add_argument(
        "--env-density",
        action="store_true",
        help="Append cosine-weighted 3Å/10Å coordination to the env Δv(q) features",
    )
    p.add_argument(
        "--density-cutoffs",
        type=float,
        nargs="+",
        default=None,
        help="Real-space coordination cutoffs in Å (default: 3 10)",
    )
    p.add_argument(
        "--env-l2",
        type=float,
        default=0.0,
        help="Penalty on mean(Δc²) so env Δv(q) stays small unless the loss needs it",
    )
    p.add_argument(
        "--env-pack-gate",
        action="store_true",
        help="Multiply per-atom Δv by σ(α(C0−CN10)); cages on, compact 4-fold off",
    )
    p.add_argument("--pack-gate-center", type=float, default=142.0, help="CN(10 Å) gate center C0")
    p.add_argument("--pack-gate-alpha", type=float, default=2.0, help="Packing-gate steepness α")
    p.add_argument(
        "--residual-pack-gate",
        action="store_true",
        help="Blend residual as (1-g)*frozen + g*live so l<=2 trains only on open packing",
    )
    p.add_argument(
        "--gate-window",
        action="append",
        default=None,
        metavar="SPEC",
        help=(
            "Packing-gate window such as 'cn10<142@2,cn5<17@5' (AND of terms); repeat the flag for an OR of "
            "windows. Replaces the single CN10 step for both the env Δv gate and the residual blend."
        ),
    )
    p.add_argument(
        "--sphere-density-order",
        type=int,
        default=None,
        help="Polynomial order of per-atom sphere coefficients c_n(t), t=(CN10-center)/scale; 0 = off",
    )
    p.add_argument("--sphere-density-center", type=float, default=151.0, help="CN10 of diamond Si (t=0)")
    p.add_argument("--sphere-density-scale", type=float, default=30.0, help="CN10 units per unit t")
    p.add_argument(
        "--sphere-density-lr", type=float, default=None, help="LR of the density-conditioned sphere (default: --env-lr)"
    )
    p.add_argument(
        "--residual-anchor-l2",
        type=float,
        default=0.0,
        help="Make the compact-site residual live with an L2 pull of this weight toward its snapshot (0 = hard freeze)",
    )
    p.add_argument("--compact-lr", type=float, default=None, help="LR of the anchored compact residual (default: --lr)")
    p.add_argument("--sp-nonlocal", action="store_true", help="Add zero-init species s/p separable projectors")
    p.add_argument(
        "--coordination-features",
        action="store_true",
        help="Add zero-init smooth multi-radius local densities to the encoder scalar channels",
    )
    p.add_argument(
        "--sp-gated-head",
        action="store_true",
        help="Zero-init per-atom additive s/p strengths from porous-branch invariants, times the packing gate",
    )
    p.add_argument(
        "--quad-scale-env",
        action="store_true",
        help="Per-atom exp(θ1 t + θ2 t²) scale on the l=2 residual, t=(CN10-151)/30, zero-init",
    )
    p.add_argument("--quad-scale-lr", type=float, default=None, help="LR of the quadrupole scale θ (default: --env-lr)")
    p.add_argument("--sp-projector-scales", type=float, nargs="+", default=None, help="Projector Gaussian scales (Å⁻¹)")
    p.add_argument("--projector-lr", type=float, default=0.1)
    p.add_argument(
        "--eval-splits",
        nargs="*",
        default=None,
        help="Extra splits scored without gradient every --eval-every epochs (never in the loss), e.g. holdout",
    )
    p.add_argument("--eval-families", nargs="*", default=None, help="Restrict --eval-splits to these families")
    p.add_argument("--watch-families", nargs="*", default=None, help="Train families re-scored at eval epochs")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=1,
        help=(
            "Optimizer steps per pass over the catalog. 1 = full-batch (Phase 1/2 protocol). N>1 shuffles the "
            "cells and interleaves them into N chunks; each chunk keeps the equal-family cell weights, so its "
            "gradient is an unbiased 1/N estimate of the full-batch gradient (Adam is scale-invariant)."
        ),
    )
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed for --steps-per-epoch > 1")
    p.add_argument(
        "--freeze-residual",
        action="store_true",
        help="Freeze encoder and l<=2 readout; train only env Δv(q) / sphere",
    )
    p.add_argument("--env-lr", type=float, default=None, help="Env-isotropic Δv(q) learning rate (default: --lr)")
    p.add_argument(
        "--eval-full-path",
        action="store_true",
        help="Rebuild a full-k cache for final metrics (slow). Default: score the training k-grid.",
    )
    p.add_argument(
        "--cell-workers",
        type=int,
        default=1,
        help="Parallel catalog cells per epoch (same weighted loss; default 1 keeps the serial path)",
    )
    p.add_argument(
        "--threads-per-worker",
        type=int,
        default=0,
        help="BLAS/PyTorch threads per cell worker (0: CPUs/cell-workers, uses SLURM_CPUS_PER_TASK)",
    )
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _rows(
    dataset: NPZDataset,
    splits: list[str],
    families: list[str] | None,
    max_samples: int,
    exclude_ids: list[str] | None = None,
):
    rows = []
    excluded = set(exclude_ids or ())
    for sample in dataset:
        meta = sample.structure.metadata
        if meta.get("split") not in splits:
            continue
        if families and meta.get("family") not in families:
            continue
        if sample.sample_id in excluded:
            continue
        if sample.bands is None:
            raise ValueError(f"{sample.sample_id} has no bands")
        if str(meta.get("label_source")) != "qe-pbe":
            raise ValueError(f"{sample.sample_id} is not qe-pbe")
        rows.append(sample)
        if max_samples and len(rows) >= max_samples:
            break
    if not rows:
        raise ValueError("no labeled samples matched the split/family filters")
    return rows


def _metrics(pred: np.ndarray, target: np.ndarray, n_occ: int, use_gap: bool) -> dict:
    pred_vbm = float(pred[:, n_occ - 1].max())
    tgt_vbm = float(target[:, n_occ - 1].max())
    out = {
        "window_mae_eV": mixed_window_mae(pred, target, n_occ, use_gap),
        "vbm_mae_eV": vbm_window_mae(pred, target, n_occ),
        "fermi_mae_eV": fermi_window_mae(pred, target, n_occ),
        "align": "vbm" if use_gap else "fermi",
        "gapped": use_gap,
        "vbm_shift_eV": pred_vbm - tgt_vbm,
        "fermi_shift_eV": fermi_proxy(pred, n_occ) - fermi_proxy(target, n_occ),
    }
    if use_gap:
        out["gap_eV"] = band_gap(pred, n_occ)
        out["gap_error_eV"] = gap_error(pred, target, n_occ)
    else:
        out["gap_eV"] = None
        out["gap_error_eV"] = None
    return out


def _slope_metrics(pred, row, target=None) -> dict:
    """Slope-aware band error (see epnet.eval.metrics.slope_aware_window_mae) on the row's k-path."""
    tgt = row["target"].cpu().numpy() if target is None else target
    k_cart = path_cartesian_k(row["k"].cpu().numpy(), row["packed"].lattice.cpu().numpy())
    return slope_aware_window_mae(np.asarray(pred)[:, : tgt.shape[1]], tgt, row["n_occ"], row["use_gap"], k_cart)


def _cell_extras(model, row) -> dict:
    """Geometry and gate diagnostics for one cell: the 'g per family' column."""
    out: dict = {}
    gate = getattr(model, "_last_pack_gate", None)
    if gate is not None and (model.env_pack_gate or model.residual_pack_gate):
        out["gate_mean"] = float(gate.mean())
        out["gate_max"] = float(gate.max())
    weights = getattr(model, "_last_expert_weights", None)
    if weights is not None and getattr(model, "expert_router", None) is not None:
        out["expert_mean"] = [float(w) for w in weights.mean(dim=0)]
    cn = model._coordination(row["packed"], (3.0, 5.0, 10.0))
    out["cn3_mean"] = float(cn[3.0].mean())
    out["cn5_mean"] = float(cn[5.0].mean())
    out["cn10_mean"] = float(cn[10.0].mean())
    dens = getattr(model, "_last_sphere_density_delta", None)
    if dens is not None and model.sphere_density is not None:
        out["sphere_density_delta_rms"] = float(dens.pow(2).mean().sqrt())
    return out


def _score_rows(model, rows, refresh_sphere: bool) -> list[dict]:
    """Mixed-window metrics on cached k-grids, no gradient."""
    model.eval()
    metrics = []
    with torch.no_grad():
        for row in rows:
            pred = model.band_path(
                row["packed"], row["k"], n_bands=row["n_bands"], cache=row["cache"], refresh_sphere=refresh_sphere
            )
            rec = _metrics(pred.cpu().numpy(), row["target"].cpu().numpy(), row["n_occ"], row["use_gap"])
            rec.update(_cell_extras(model, row))
            rec.update({"id": row["id"], "family": row["family"], "split": row["split"]})
            metrics.append(rec)
    return metrics


def _family_summary(metrics: list[dict], mae_gate: float = 0.10) -> dict:
    fam: dict = {}
    for rec in metrics:
        fam.setdefault(rec["family"], []).append(rec)
    out = {}
    for name in sorted(fam):
        rows = fam[name]
        maes = [float(r["window_mae_eV"]) for r in rows]
        gap_errors = [float(r["gap_error_eV"]) for r in rows if r.get("gapped") and r.get("gap_error_eV") is not None]
        gates = [float(r["gate_mean"]) for r in rows if r.get("gate_mean") is not None]
        experts = [r["expert_mean"] for r in rows if r.get("expert_mean") is not None]
        out[name] = {
            "n": len(rows),
            "mae_mean_eV": float(np.mean(maes)),
            "mae_pass": int(sum(m < mae_gate for m in maes)),
            "gap_mae_eV": float(np.mean(gap_errors)) if gap_errors else None,
            "gap_pass": sum(e < 0.05 for e in gap_errors),
            "gap_n": len(gap_errors),
            "gate_mean": float(np.mean(gates)) if gates else None,
            "expert_mean": [float(x) for x in np.mean(experts, axis=0)] if experts else None,
        }
    return out


def _format_family_summary(summary: dict) -> str:
    parts = []
    for name, rec in summary.items():
        gate = f" g={rec['gate_mean']:.2f}" if rec.get("gate_mean") is not None else ""
        if rec.get("expert_mean") is not None:
            gate += " w=" + "/".join(f"{w:.2f}" for w in rec["expert_mean"])
        gap = f" gap={rec['gap_mae_eV']:.4f} ({rec['gap_pass']}/{rec['gap_n']})" if rec.get("gap_mae_eV") is not None else ""
        parts.append(f"{name} {rec['mae_mean_eV']:.3f} ({rec['mae_pass']}/{rec['n']}){gate}{gap}")
    return "  ".join(parts)


def _prepare_row(model, sample, device, k_stride: int, density_dir: Path | None = None):
    meta = sample.structure.metadata
    packed = pack_structure(sample.structure)
    packed.positions = packed.positions.to(device)
    packed.lattice = packed.lattice.to(device)
    packed.fractional_positions = packed.fractional_positions.to(device)
    packed.species = packed.species.to(device)
    packed.volume = packed.volume.to(device)
    k_full = torch.as_tensor(sample.bands.kpoints, dtype=packed.lattice.dtype, device=device)
    e_full = torch.as_tensor(sample.bands.energies, dtype=packed.lattice.dtype, device=device)
    idx = torch.arange(0, k_full.shape[0], max(int(k_stride), 1), device=device)
    k = k_full[idx]
    target = e_full[idx]
    cache = model.make_pw_cache(packed, k)
    dens = {}
    if density_dir is not None and bool(meta.get("use_gap_loss")) and (density_dir / f"{sample.sample_id}.npz").exists():
        z = np.load(density_dir / f"{sample.sample_id}.npz")
        dk = torch.as_tensor(z["dens_k"], dtype=packed.lattice.dtype, device=device)
        dcache = model.make_pw_cache(packed, dk)
        shape = [int(x) for x in z["shape"]]
        m = dcache.g_idx.to(torch.long) % torch.as_tensor(shape, device=dcache.g_idx.device)
        assert [int(4 * x + 2) for x in dcache.g_idx.abs().amax(0).tolist()] == shape, (sample.sample_id, shape)
        dens = {"dens_k": dk, "dens_cache": dcache, "dens_shape": shape,
                "dens_lin": ((m[:, 0] * shape[1] + m[:, 1]) * shape[2] + m[:, 2]).to(device),
                "dens_idx": torch.as_tensor(z["idx"], dtype=torch.long, device=device),
                "dens_logt": torch.as_tensor(z["logt"], dtype=packed.lattice.dtype, device=device)}
    return {
        **dens,
        "id": sample.sample_id,
        "family": str(meta.get("family", "unknown")),
        "split": meta.get("split"),
        "use_gap": bool(meta.get("use_gap_loss")),
        "n_occ": int(meta.get("n_occupied") or packed.n_occupied),
        "n_bands": int(e_full.shape[1]),
        "packed": packed,
        "k": k,
        "k_full": k_full,
        "target": target,
        "e_full": e_full,
        "cache": cache,
    }


def main() -> None:
    args = parse_args()
    cell_workers, threads_per_worker = resolve_thread_plan(args.cell_workers, args.threads_per_worker)
    dataset = NPZDataset(args.manifest, allow_external=True)
    samples = _rows(dataset, args.splits, args.families, args.max_samples, exclude_ids=args.exclude_ids)
    if args.exclude_ids:
        print(f"excluded ids: {sorted(args.exclude_ids)}", flush=True)
    device = torch.device(args.device)
    env_iso = bool(args.env_isotropic)
    env_density = bool(args.env_density)
    env_pack_gate = bool(args.env_pack_gate)
    residual_pack_gate = bool(args.residual_pack_gate)
    pack_gate_center = float(args.pack_gate_center)
    pack_gate_alpha = float(args.pack_gate_alpha)
    density_cutoffs = tuple(args.density_cutoffs) if args.density_cutoffs else (3.0, 10.0)
    init_payload = None
    init_state = None
    if args.model_init:
        init_payload = torch.load(args.model_init, map_location="cpu", weights_only=False)
        init_state = (
            init_payload["state_dict"]
            if isinstance(init_payload, dict) and "state_dict" in init_payload
            else init_payload
        )
        payload_dict = init_payload if isinstance(init_payload, dict) else {}
        env_iso = env_iso or checkpoint_env_isotropic(payload_dict, init_state)
        env_density = env_density or checkpoint_env_density(payload_dict)
        env_pack_gate = env_pack_gate or checkpoint_env_pack_gate(payload_dict)
        residual_pack_gate = residual_pack_gate or checkpoint_residual_pack_gate(payload_dict)
        if args.density_cutoffs is None and checkpoint_env_density(payload_dict):
            density_cutoffs = checkpoint_density_cutoffs(payload_dict)
        if not args.env_pack_gate and checkpoint_env_pack_gate(payload_dict):
            pack_gate_center = checkpoint_pack_gate_center(payload_dict)
            pack_gate_alpha = checkpoint_pack_gate_alpha(payload_dict)
    if env_density:
        env_iso = True
    # Phase 3 channels: gate windows, density-conditioned sphere, anchored compact residual, s/p projectors.
    gate_windows = normalize_gate_windows(args.gate_window)
    sphere_density_order = args.sphere_density_order
    sphere_density_center = float(args.sphere_density_center)
    sphere_density_scale = float(args.sphere_density_scale)
    residual_anchor = float(args.residual_anchor_l2) > 0.0
    sp_nonlocal = bool(args.sp_nonlocal)
    sp_gated_head = bool(args.sp_gated_head)
    coordination_features = bool(args.coordination_features)
    quad_scale_env = bool(args.quad_scale_env)
    sp_scales = tuple(args.sp_projector_scales) if args.sp_projector_scales else (1.5, 3.5)
    if args.model_init:
        payload_dict = init_payload if isinstance(init_payload, dict) else {}
        if gate_windows is None:
            gate_windows = checkpoint_pack_gate_windows(payload_dict)
        ck_order, ck_center, ck_scale = checkpoint_sphere_density(payload_dict)
        if sphere_density_order is None and ck_order > 0:
            sphere_density_order, sphere_density_center, sphere_density_scale = ck_order, ck_center, ck_scale
        residual_anchor = residual_anchor or checkpoint_residual_anchor(payload_dict)
        sp_nonlocal = sp_nonlocal or checkpoint_sp_nonlocal(payload_dict, init_state)
        sp_gated_head = sp_gated_head or bool(payload_dict.get("sp_gated_head"))
        coordination_features = coordination_features or bool(payload_dict.get("coordination_features"))
        quad_scale_env = quad_scale_env or checkpoint_quad_scale(payload_dict, init_state)[0]
        if args.sp_projector_scales is None and payload_dict.get("sp_projector_scales"):
            sp_scales = tuple(float(x) for x in payload_dict["sp_projector_scales"])
    if sphere_density_order is None:
        sphere_density_order = 0
    if residual_anchor:
        residual_pack_gate = True
    encoder_cutoff = args.encoder_cutoff
    if encoder_cutoff is None:
        encoder_cutoff = float((init_payload or {}).get("encoder_cutoff", 5.0)) if args.model_init else 5.0
    model = build_gated_si_epm(
        channels=args.channels,
        cutoff=encoder_cutoff,
        residual_scalar_cosine_basis=int((init_payload or {}).get("residual_scalar_cosine_basis", 0)),
        residual_scalar_cosine_cutoff=float((init_payload or {}).get("residual_scalar_cosine_cutoff", 8.0)),
        residual_experts=int((init_payload or {}).get("residual_experts", 1)),
        fold_k=bool((init_payload or {}).get("fold_k", False)),
        k_centred_basis=bool((init_payload or {}).get("k_centred_basis", False)),
        residual_envelope=bool((init_payload or {}).get("residual_envelope", False)),
        smooth_cutoff=bool((init_payload or {}).get("smooth_cutoff", False)),
        smooth_buffer=float((init_payload or {}).get("smooth_buffer", 0.0)),
        ecut_ry=args.ecut_ry,
        monopole_mode=args.monopole_mode,
        env_isotropic=env_iso,
        env_density=env_density,
        density_cutoffs=density_cutoffs,
        env_pack_gate=env_pack_gate,
        residual_pack_gate=residual_pack_gate,
        pack_gate_center=pack_gate_center,
        pack_gate_alpha=pack_gate_alpha,
        pack_gate_windows=gate_windows,
        sphere_density_order=int(sphere_density_order),
        sphere_density_center=sphere_density_center,
        sphere_density_scale=sphere_density_scale,
        residual_anchor=residual_anchor,
        sp_nonlocal=sp_nonlocal,
        sp_gated_head=sp_gated_head,
        coordination_features=coordination_features,
        sp_projector_scales=sp_scales,
        quad_scale_env=quad_scale_env,
    )
    if args.model_init:
        load_gated_si_epm_state(model, init_state)
        model.set_sphere_trainable(not args.freeze_sphere)
        print(
            f"loaded model from {args.model_init}  freeze_sphere={args.freeze_sphere}  "
            f"env_isotropic={env_iso}  env_density={env_density}  "
            f"env_pack_gate={env_pack_gate}  residual_pack_gate={residual_pack_gate}  "
            f"C0={pack_gate_center}  alpha={pack_gate_alpha}  windows={format_gate_windows(model.pack_gate_windows)}  "
            f"sphere_density_order={model.sphere_density_order}  residual_anchor={model.residual_anchor}  "
            f"sp_nonlocal={model.sp_nonlocal}  quad_scale_env={model.quad_scale_env}  "
            f"scalar_cosine_basis={model.residual_scalar_cosine_basis}",
            flush=True,
        )
    elif args.sphere_init:
        model.load_sphere(args.sphere_init, freeze=args.freeze_sphere)
        print(f"loaded sphere from {args.sphere_init}  freeze={args.freeze_sphere}", flush=True)
    else:
        model.set_sphere_trainable(not args.freeze_sphere)
    if args.freeze_residual and residual_pack_gate:
        raise ValueError("residual-pack-gate needs a live residual; do not pass --freeze-residual")
    if args.freeze_residual:
        keep = ("sphere.", "env_radial.", "sphere_density.", "sp_projector.", "quad_scale.")
        for name, param in model.named_parameters():
            if not name.startswith(keep):
                param.requires_grad_(False)
    model.to(device)
    env_lr = args.env_lr if args.env_lr is not None else args.lr
    # sp_head. rides in the projector group: its output adds to the s/p strengths, so it must train at their LR.
    special = ("sphere.", "env_radial.", "encoder_frozen.", "readout_frozen.", "sphere_density.", "sp_projector.", "sp_head.", "quad_scale.")
    residual = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(special)]
    env_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("env_radial.")]
    sphere = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("sphere.")]
    density_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("sphere_density.")]
    compact_params = [
        p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("encoder_frozen.", "readout_frozen."))
    ]
    projector_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("sp_projector.", "sp_head."))]
    quad_scale_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("quad_scale.")]
    quad_scale_lr = args.quad_scale_lr if args.quad_scale_lr is not None else env_lr
    density_lr = args.sphere_density_lr if args.sphere_density_lr is not None else env_lr
    compact_lr = args.compact_lr if args.compact_lr is not None else args.lr
    param_groups = []
    if residual:
        param_groups.append({"params": residual, "lr": args.lr, "weight_decay": 1e-4})
    if env_params:
        param_groups.append({"params": env_params, "lr": env_lr, "weight_decay": 1e-4})
    if sphere:
        param_groups.append({"params": sphere, "lr": args.sphere_lr, "weight_decay": 1e-4})
    if density_params:
        param_groups.append({"params": density_params, "lr": density_lr, "weight_decay": 0.0})
    if compact_params:
        # The anchor L2 regularises this copy; no weight decay toward zero.
        param_groups.append({"params": compact_params, "lr": compact_lr, "weight_decay": 0.0})
    if projector_params:
        param_groups.append({"params": projector_params, "lr": args.projector_lr, "weight_decay": 0.0})
    if quad_scale_params:
        param_groups.append({"params": quad_scale_params, "lr": quad_scale_lr, "weight_decay": 0.0})
    if not param_groups:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(param_groups)
    refresh_sphere = any(p.requires_grad for p in model.sphere.parameters())
    print(
        f"joint train  residual_tensors={len(residual)}  env_tensors={len(env_params)}  "
        f"sphere_tensors={len(sphere)}  density_tensors={len(density_params)}  "
        f"compact_tensors={len(compact_params)}  projector_tensors={len(projector_params)}  "
        f"env_isotropic={env_iso}  env_density={env_density}  "
        f"env_pack_gate={env_pack_gate}  residual_pack_gate={residual_pack_gate}  "
        f"windows={format_gate_windows(model.pack_gate_windows)}  "
        f"sphere_density_order={model.sphere_density_order}  residual_anchor_l2={args.residual_anchor_l2}  "
        f"env_l2={args.env_l2}  freeze_residual={args.freeze_residual}  "
        f"refresh_sphere={refresh_sphere}  "
        f"cell_workers={cell_workers}  threads_per_worker={threads_per_worker}",
        flush=True,
    )

    t_cache = time.perf_counter()
    # CUDA linalg wrappers are lazily initialised and not thread-safe on first use
    # ("lazy wrapper should be called at most once"): prepare rows serially on GPU.
    prep_workers = 1 if device.type == "cuda" else cell_workers
    rows = prepare_rows_parallel(
        lambda sample: _prepare_row(model, sample, device, args.k_stride,
                                    args.density_targets if args.density_weight > 0 else None),
        samples,
        prep_workers,
    )
    if args.density_weight > 0:
        n_dens = sum(1 for r in rows if r.get("dens_logt") is not None)
        print(f"density term: weight {args.density_weight} every {args.density_every} steps, targets for {n_dens} cells", flush=True)
    print(
        f"cached {len(rows)} cells in {time.perf_counter() - t_cache:.1f}s  "
        f"k-stride={args.k_stride}  cell_workers={cell_workers}  threads_per_worker={threads_per_worker}",
        flush=True,
    )
    family_count = Counter(r["family"] for r in rows)
    n_families = len(family_count)
    n_gap = sum(1 for r in rows if r["use_gap"])
    print(f"cells {len(rows)}  gapped {n_gap}  metallic {len(rows) - n_gap}  families={n_families}", flush=True)

    eval_rows: list[dict] = []
    if args.eval_splits:
        eval_samples = _rows(dataset, args.eval_splits, args.eval_families, 0, exclude_ids=args.exclude_ids)
        eval_rows = prepare_rows_parallel(
            lambda sample: _prepare_row(model, sample, device, args.k_stride), eval_samples, prep_workers
        )
        print(
            f"eval rows {len(eval_rows)} from splits {args.eval_splits} (never in the loss): "
            f"{dict(Counter(r['family'] for r in eval_rows))}",
            flush=True,
        )
    watch = set(args.watch_families or ())
    watch_rows = [r for r in rows if r["family"] in watch]
    eval_every = max(int(args.eval_every), 1)

    def _report_eval(epoch: int, entry: dict) -> None:
        if not eval_rows and not watch_rows:
            return
        t_eval = time.perf_counter()
        if watch_rows:
            summary = _family_summary(_score_rows(model, watch_rows, refresh_sphere))
            entry["watch"] = summary
            print(f"  watch  ep {epoch:03d}  {_format_family_summary(summary)}", flush=True)
        if eval_rows:
            summary = _family_summary(_score_rows(model, eval_rows, refresh_sphere))
            entry["eval"] = summary
            print(f"  eval   ep {epoch:03d}  {_format_family_summary(summary)}", flush=True)
        entry["eval_seconds"] = time.perf_counter() - t_eval
        model.train()

    history = []
    step_zero = {"epoch": 0}
    _report_eval(0, step_zero)
    if len(step_zero) > 1:
        history.append(step_zero)
    steps = max(int(args.steps_per_epoch), 1)
    shuffle = torch.Generator().manual_seed(int(args.seed))
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        if steps == 1:
            chunks = [rows]
        else:
            order = [rows[i] for i in torch.randperm(len(rows), generator=shuffle).tolist()]
            chunks = [order[i::steps] for i in range(steps)]
        running = 0.0
        anchor_value = None
        for step_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            optimizer.zero_grad(set_to_none=True)
            global_step = (epoch - 1) * steps + step_idx + 1
            running += accumulate_cell_grads(
                model,
                chunk,
                cell_workers=cell_workers,
                gap_weight=args.gap_weight,
                gap_mode=args.gap_mode,
                deep_weight=args.deep_weight,
                metal_align=args.metal_align,
                density_weight=args.density_weight,
                density_on=args.density_weight > 0 and global_step % max(int(args.density_every), 1) == 0,
                density_gap_floor=args.density_gap_floor,
                env_l2=args.env_l2,
                refresh_sphere=refresh_sphere,
                n_families=n_families,
                family_count=family_count,
                epoch_id=(epoch - 1) * steps + step_idx + 1,
            )
            if args.residual_anchor_l2 > 0 and model.residual_anchor:
                penalty = float(args.residual_anchor_l2) * model.anchor_penalty() / steps
                penalty.backward()
                anchor_value = float(penalty) * steps
            optimizer.step()
            model.restore_frozen_sphere_rows()
        mean_loss = running / len(rows)
        entry = {"epoch": epoch, "loss": mean_loss, "seconds": time.perf_counter() - t0}
        from epnet.train_cells import DENSITY_LOG
        if DENSITY_LOG:
            entry["density_loss"] = float(np.mean(DENSITY_LOG))
            DENSITY_LOG.clear()
        if anchor_value is not None:
            entry["anchor_penalty"] = anchor_value
        history.append(entry)
        if epoch % args.log_every == 0 or epoch == 1:
            anchor_txt = f"  anchor {anchor_value:.5f}" if anchor_value is not None else ""
            if "density_loss" in entry:
                anchor_txt += f"  density {entry['density_loss']:.5f}"
            print(
                f"epoch {epoch:03d}  mean loss {mean_loss:.5f}  {entry['seconds']:.1f}s{anchor_txt}",
                flush=True,
            )
        if epoch % eval_every == 0 or epoch == args.epochs:
            _report_eval(epoch, entry)

    model.eval()
    metrics = []
    with torch.no_grad():
        for row in rows:
            if args.eval_full_path:
                cache = model.make_pw_cache(row["packed"], row["k_full"])
                pred = model.band_path(
                    row["packed"],
                    row["k_full"],
                    n_bands=row["n_bands"],
                    cache=cache,
                    refresh_sphere=refresh_sphere,
                )
                target = row["e_full"].cpu().numpy()
            else:
                pred = model.band_path(
                    row["packed"],
                    row["k"],
                    n_bands=row["n_bands"],
                    cache=row["cache"],
                    refresh_sphere=refresh_sphere,
                )
                target = row["target"].cpu().numpy()
            rec = _metrics(
                pred.cpu().numpy(),
                target,
                row["n_occ"],
                row["use_gap"],
            )
            rec.update(_cell_extras(model, row))
            rec.update({"id": row["id"], "family": row["family"], "split": row["split"]})
            metrics.append(rec)
    eval_metrics = _score_rows(model, eval_rows, refresh_sphere) if eval_rows else []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ckpt = args.output.with_suffix(".pt")
    torch.save({"state_dict": model.state_dict(), **model_payload_flags(model)}, ckpt)
    payload = {
        "schema_version": 1,
        "claim": "GatedSiEPM joint train on all Si PBE train families. Cohen v(q) is a physics init, not a crystal prefit.",
        "settings": {
            "epochs": args.epochs,
            "residual_scalar_cosine_basis": model.residual_scalar_cosine_basis,
            "residual_scalar_cosine_cutoff": model.residual_scalar_cosine_cutoff,
            "ecut_ry": args.ecut_ry,
            "channels": args.channels,
            "monopole_mode": args.monopole_mode,
            "env_isotropic": bool(model.env_isotropic),
            "env_density": bool(model.env_density),
            "density_cutoffs": list(model.density_cutoffs),
            "env_pack_gate": bool(model.env_pack_gate),
            "residual_pack_gate": bool(model.residual_pack_gate),
            "pack_gate_center": float(model.pack_gate_center),
            "pack_gate_alpha": float(model.pack_gate_alpha),
            "pack_gate_windows": model.pack_gate_windows,
            "gate_windows_text": format_gate_windows(model.pack_gate_windows),
            "sphere_density_order": int(model.sphere_density_order),
            "sphere_density_center": float(model.sphere_density_center),
            "sphere_density_scale": float(model.sphere_density_scale),
            "sphere_density_lr": density_lr if density_params else None,
            "residual_anchor_l2": float(args.residual_anchor_l2),
            "compact_lr": compact_lr if compact_params else None,
            "sp_nonlocal": bool(model.sp_nonlocal),
            "sp_gated_head": bool(model.sp_gated_head),
            "coordination_features": bool(model.coordination_features),
            "residual_experts": int(model.residual_experts),
            "gap_mode": args.gap_mode,
            "deep_weight": args.deep_weight,
            "metal_align": args.metal_align,
            "density_weight": args.density_weight,
            "density_every": args.density_every,
            "density_targets": str(args.density_targets) if args.density_targets else None,
            "density_gap_floor": args.density_gap_floor,
            "sp_projector_scales": list(sp_scales) if model.sp_nonlocal else None,
            "projector_lr": args.projector_lr if projector_params else None,
            "quad_scale_env": bool(model.quad_scale_env),
            "quad_scale_lr": quad_scale_lr if quad_scale_params else None,
            "eval_splits": args.eval_splits,
            "eval_families": args.eval_families,
            "watch_families": args.watch_families,
            "eval_every": eval_every,
            "steps_per_epoch": steps,
            "seed": int(args.seed),
            "env_l2": args.env_l2,
            "freeze_residual": bool(args.freeze_residual),
            "k_stride": args.k_stride,
            "cell_workers": cell_workers,
            "threads_per_worker": threads_per_worker,
            "n_samples": len(rows),
            "splits": args.splits,
            "families": args.families,
            "exclude_ids": sorted(args.exclude_ids) if args.exclude_ids else [],
            "sphere_init": str(args.sphere_init) if args.sphere_init else None,
            "model_init": str(args.model_init) if args.model_init else None,
            "freeze_sphere": bool(args.freeze_sphere),
            "lr": args.lr,
            "sphere_lr": args.sphere_lr,
            "env_lr": env_lr,
            "metal_loss": "fermi_window",
            "scorecard": "vbm_window if gapped else fermi_window",
            "eval_full_path": bool(args.eval_full_path),
        },
        "history": history,
        "metrics": metrics,
        "eval_metrics": eval_metrics,
        "family_summary": _family_summary(metrics),
        "eval_family_summary": _family_summary(eval_metrics) if eval_metrics else None,
        "checkpoint": str(ckpt),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"train   {_format_family_summary(payload['family_summary'])}")
    if eval_metrics:
        print(f"eval    {_format_family_summary(payload['eval_family_summary'])}")
    print(f"wrote {args.output} and {ckpt}")


if __name__ == "__main__":
    main()
