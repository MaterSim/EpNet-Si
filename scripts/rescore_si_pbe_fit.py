#!/usr/bin/env python3
"""Re-score a Si PBE checkpoint with mixed VBM/Fermi window MAE (no training)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from epnet.data.manifest import NPZDataset  # noqa: E402
from epnet.models.si_epm import load_gated_si_epm, model_payload_flags, slice_pw_cache  # noqa: E402
K_CHUNK = 16
from train_si_pbe_dataset import _cell_extras, _family_summary, _format_family_summary, _metrics, _prepare_row, _rows, _slope_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/si_pbe_bands.json")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "results/si_pbe_dataset_fit_ecut8.pt")
    p.add_argument("--output", type=Path, default=ROOT / "results/si_pbe_dataset_fit.json")
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--exclude-ids", nargs="*", default=None, help="Sample ids left out of the scorecard")
    p.add_argument("--k-stride", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-save-model", action="store_true", help="Do not write a .pt next to --output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, payload = load_gated_si_epm(args.checkpoint)
    ecut = float(payload.get("ecut_ry", 8.0))
    channels = int(payload.get("channels", 8))
    monopole = str(payload.get("monopole_mode", "free"))
    env_iso = bool(model.env_isotropic)
    env_density = bool(getattr(model, "env_density", False))
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    print(
        f"rescore {args.checkpoint}  ecut={ecut}  monopole={monopole}  "
        f"env_isotropic={env_iso}  env_density={env_density}",
        flush=True,
    )

    dataset = NPZDataset(args.manifest, allow_external=True)
    samples = _rows(dataset, args.splits, args.families, 0, exclude_ids=args.exclude_ids)
    t0 = time.perf_counter()
    rows = [_prepare_row(model, sample, device, args.k_stride) for sample in samples]
    print(f"cached {len(rows)} cells in {time.perf_counter() - t0:.1f}s", flush=True)

    metrics = []
    with torch.no_grad():
        for row in rows:
            # k in chunks: the k-centred basis gathers one (n_k, n_basis, n_basis) Hamiltonian per call
            pred = torch.cat([model.band_path(
                row["packed"],
                row["k"][i : i + K_CHUNK],
                n_bands=row["n_bands"],
                cache=slice_pw_cache(row["cache"], slice(i, i + K_CHUNK)),
                refresh_sphere=True,
            ) for i in range(0, row["k"].shape[0], K_CHUNK)])
            rec = _metrics(pred.cpu().numpy(), row["target"].cpu().numpy(), row["n_occ"], row["use_gap"])
            rec.update(_cell_extras(model, row))
            rec.update(_slope_metrics(pred.cpu().numpy(), row))
            rec.update({"id": row["id"], "family": row["family"], "split": row["split"]})
            metrics.append(rec)

    history = []
    src_json = args.checkpoint.with_suffix(".json")
    if src_json.exists():
        history = json.loads(src_json.read_text()).get("history") or []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ckpt_out = args.output.with_suffix(".pt")
    if not args.no_save_model and ckpt_out.resolve() != args.checkpoint.resolve():
        torch.save({"state_dict": model.state_dict(), **model_payload_flags(model)}, ckpt_out)
    elif args.no_save_model:
        ckpt_out = args.checkpoint
    out = {
        "schema_version": 1,
        "claim": "GatedSiEPM mixed scorecard: VBM window on gapped cells, Fermi window on metals.",
        "settings": {
            "ecut_ry": ecut,
            "channels": channels,
            "monopole_mode": monopole,
            "env_isotropic": env_iso,
            "env_density": env_density,
            "density_cutoffs": list(getattr(model, "density_cutoffs", (3.0, 10.0))),
            "k_stride": args.k_stride,
            "n_samples": len(rows),
            "splits": args.splits,
            "families": args.families,
            "exclude_ids": sorted(args.exclude_ids) if args.exclude_ids else [],
            "checkpoint": str(args.checkpoint),
            "model_flags": {k: v for k, v in model_payload_flags(model).items() if k != "species_symbols"},
            "freeze_sphere": False,
            "metal_loss": "fermi_window",
            "scorecard": "vbm_window if gapped else fermi_window",
        },
        "history": history,
        "metrics": metrics,
        "family_summary": _family_summary(metrics),
        "checkpoint": str(ckpt_out),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"scored  {_format_family_summary(out['family_summary'])}", flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
