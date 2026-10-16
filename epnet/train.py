"""Reproducible training utilities and a small tensor-dataset CLI."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch and request deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def potential_loss(
    prediction: Tensor,
    target: Tensor,
    g_vectors: Tensor,
    *,
    n_atoms: Tensor | float | None = None,
    volumes: Tensor | float | None = None,
    sample_mask: Tensor | None = None,
    kind: str = "huber",
    delta: float = 0.1,
) -> Tensor:
    """Return a sample-balanced complex potential loss.

    ``G=0`` is always excluded. Each sample is first averaged over its valid
    reciprocal vectors, then samples are averaged. If ``n_atoms`` and
    ``volumes`` are supplied, residuals are converted to the intensive
    per-atom convention ``V * volume / n_atoms`` before reduction.
    """
    if prediction.shape != target.shape or prediction.ndim not in (1, 2):
        raise ValueError("prediction and target must have matching (G,) or (sample,G) shapes")
    pred = prediction.unsqueeze(0) if prediction.ndim == 1 else prediction
    ref = target.unsqueeze(0) if target.ndim == 1 else target
    gv = g_vectors.unsqueeze(0) if g_vectors.ndim == 2 else g_vectors
    if gv.shape[-1] != 3 or gv.shape[-2] != pred.shape[-1]:
        raise ValueError("g_vectors must end in (G,3)")
    if gv.shape[0] == 1 and pred.shape[0] > 1:
        gv = gv.expand(pred.shape[0], -1, -1)
    if gv.shape[0] != pred.shape[0]:
        raise ValueError("g_vectors batch dimension does not match predictions")

    residual = pred - ref
    if (n_atoms is None) != (volumes is None):
        raise ValueError("n_atoms and volumes must be supplied together")
    if n_atoms is not None:
        atoms = torch.as_tensor(n_atoms, device=pred.device, dtype=pred.real.dtype).reshape(-1)
        volume = torch.as_tensor(volumes, device=pred.device, dtype=pred.real.dtype).reshape(-1)
        if atoms.numel() == 1:
            atoms = atoms.expand(pred.shape[0])
        if volume.numel() == 1:
            volume = volume.expand(pred.shape[0])
        if atoms.shape[0] != pred.shape[0] or volume.shape[0] != pred.shape[0]:
            raise ValueError("normalization arrays must have one value per sample")
        if torch.any(atoms <= 0) or torch.any(volume <= 0):
            raise ValueError("n_atoms and volumes must be positive")
        residual = residual * (volume / atoms)[:, None]

    magnitude = residual.abs()
    if kind == "mae":
        point_loss = magnitude
    elif kind == "mse":
        point_loss = magnitude.square()
    elif kind == "huber":
        d = torch.as_tensor(delta, dtype=magnitude.dtype, device=magnitude.device)
        if delta <= 0:
            raise ValueError("delta must be positive")
        point_loss = torch.where(magnitude <= d, 0.5 * magnitude.square() / d, magnitude - 0.5 * d)
    else:
        raise ValueError("kind must be 'mae', 'mse', or 'huber'")

    valid = (gv.square().sum(-1) > 0)
    if sample_mask is not None:
        mask = sample_mask.unsqueeze(0) if sample_mask.ndim == 1 else sample_mask
        valid = valid & mask.to(device=valid.device, dtype=torch.bool)
    counts = valid.sum(-1)
    if torch.any(counts == 0):
        raise ValueError("every sample must contain at least one valid nonzero G")
    return ((point_loss * valid).sum(-1) / counts).mean()


@dataclass
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    checkpoint_every: int = 0
    grad_clip: float | None = None


def train_model(
    model: nn.Module,
    train_batches: Iterable[Mapping[str, Tensor]],
    config: TrainConfig,
    *,
    validation_batches: Iterable[Mapping[str, Tensor]] | None = None,
    checkpoint_path: str | Path | None = None,
    forward_fn: Callable[[nn.Module, Mapping[str, Tensor]], Tensor] | None = None,
    loss_fn: Callable[[Tensor, Mapping[str, Tensor]], Tensor] | None = None,
) -> dict[str, list[float]]:
    """Train a model with deterministic ordering and resumable checkpoints."""
    seed_everything(config.seed)
    batches = list(train_batches)
    validation = list(validation_batches or [])
    if not batches:
        raise ValueError("train_batches must not be empty")
    forward = forward_fn or (lambda m, b: m(b["inputs"]))
    objective = loss_fn or (
        lambda output, batch: potential_loss(
            output, batch["targets"], batch["g_vectors"],
            n_atoms=batch.get("n_atoms"), volumes=batch.get("volumes"),
            sample_mask=batch.get("mask"),
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    start_epoch = 0
    history: dict[str, list[float]] = {"train_loss": [], "validation_loss": []}
    path = Path(checkpoint_path) if checkpoint_path else None
    if path and path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = int(state["epoch"]) + 1

    for epoch in range(start_epoch, config.epochs):
        model.train()
        losses = []
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(forward(model, batch), batch)
            loss.backward()
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        history["train_loss"].append(float(np.mean(losses)))
        if validation:
            model.eval()
            with torch.no_grad():
                values = [float(objective(forward(model, b), b)) for b in validation]
            history["validation_loss"].append(float(np.mean(values)))
        if path and config.checkpoint_every and (epoch + 1) % config.checkpoint_every == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "history": history, "config": asdict(config)}, path)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": config.epochs - 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "history": history, "config": asdict(config)}, path)
    return history


class ComplexLinear(nn.Module):
    """Small reference regressor used by the generic CLI and synthetic study."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        value = self.linear(inputs)
        return torch.complex(value[..., 0], value[..., 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON training configuration")
    parser.add_argument("--dataset", required=True, help="torch file with train/validation batch lists")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--history", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    config = TrainConfig(**raw["training"])
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    model = ComplexLinear(int(raw["model"]["n_features"]))
    history = train_model(model, dataset["train"], config, validation_batches=dataset.get("validation"),
                          checkpoint_path=args.checkpoint)
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history).write_text(json.dumps({"synthetic": bool(raw.get("synthetic", False)),
                                               "history": history, "config": raw}, indent=2) + "\n")


if __name__ == "__main__":
    main()
