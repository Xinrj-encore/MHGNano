"""Checkpoint and early-stopping helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import save_model


class EarlyStopping:
    """Stop after ``patience`` validation checks without improvement."""

    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = int(patience)
        self.delta = float(delta)
        self.best = -float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        improved = float(value) > self.best + self.delta
        if improved:
            self.best = float(value)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved


class Checkpointer:
    """Save last, periodic, and best validation checkpoints."""

    def __init__(self, out_dir: str, save_every: int = 0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(save_every) if save_every else 0
        self.best_metric = -float("inf")
        self.best_path_pt = self.out_dir / "best_model.pt"
        self.best_path_safetensors = self.out_dir / "best_model.safetensors"
        self.last_path_pt = self.out_dir / "last_model.pt"

    @staticmethod
    def _pack_state(epoch: int, meta, metrics: Dict[str, float]) -> Dict:
        return {
            "epoch": int(epoch),
            "metrics": metrics,
            "model_state_dict": meta.model.state_dict(),
            "optimizer_state_dict": meta.meta_optimizer.state_dict(),
            "hyperparams": {
                "inner_lr": meta.inner_lr,
                "inner_steps": meta.inner_steps,
                "loss_type": meta.loss_type,
            },
        }

    def save_last(self, epoch: int, meta, metrics: Dict[str, float]) -> None:
        torch.save(self._pack_state(epoch, meta, metrics), self.last_path_pt)

    def save_periodic(self, epoch: int, meta, metrics: Dict[str, float]) -> None:
        if self.save_every and epoch % self.save_every == 0:
            torch.save(
                self._pack_state(epoch, meta, metrics),
                self.out_dir / f"epoch_{epoch:05d}.pt",
            )

    def save_if_best(
        self,
        epoch: int,
        meta,
        metrics: Dict[str, float],
        monitor: str = "meta_aucpr",
    ) -> bool:
        score = float(metrics.get(monitor, -float("inf")))
        if score <= self.best_metric:
            return False
        self.best_metric = score
        payload = self._pack_state(epoch, meta, metrics)
        torch.save(payload, self.best_path_pt)
        save_model(meta.model, str(self.best_path_safetensors))
        return True

    def try_resume(self, meta, resume_path: Optional[str]) -> int:
        if not resume_path:
            return 1
        path = Path(resume_path)
        if path.is_dir():
            path = path / "last_model.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        meta.model.load_state_dict(payload["model_state_dict"])
        meta.meta_optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.best_metric = float(
            payload.get("metrics", {}).get("meta_aucpr", self.best_metric)
        )
        return int(payload.get("epoch", 0)) + 1
