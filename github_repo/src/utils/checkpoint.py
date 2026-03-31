"""
Utility functions for checkpointing and training.
"""
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import save_model


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Args:
        patience: Number of epochs to wait before stopping
        delta: Minimum change to qualify as improvement
    """

    def __init__(self, patience: int = 10, delta: float = 0):
        self.patience = patience
        self.delta = delta
        self.best_auc_pr = 0.0
        self.counter = 0
        self.early_stop = False

    def __call__(
        self, auc_pr: float, loss_last: list, loss_final: list
    ) -> bool:
        """
        Check if training should stop.

        Returns:
            True if improved, False otherwise
        """
        improved = (
            auc_pr > self.best_auc_pr
            and (
                loss_final[0] < loss_last[0]
                or loss_final[1] < loss_last[1]
                or loss_final[2] < loss_last[2]
            )
        )

        if improved:
            self.best_auc_pr = auc_pr
            self.counter = 0
        else:
            self.counter += 1
            print(f"Early stop counter: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True

        return improved


class Checkpointer:
    """
    Model checkpointing utility.

    Args:
        out_dir: Output directory for checkpoints
        save_every: Save periodic checkpoints every N epochs (0=off)
    """

    def __init__(self, out_dir: str, save_every: int = 0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(save_every) if save_every else 0
        self.best_metric = -float("inf")
        self.best_path_pt = self.out_dir / "best_model.pt"
        self.best_path_safetensors = self.out_dir / "best_model.safetensors"
        self.last_path_pt = self.out_dir / "last_model.pt"

    def _pack_state(
        self, epoch: int, meta, metrics: Dict[str, float]
    ) -> Dict:
        """Pack model state for saving."""
        return {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": meta.model.state_dict(),
            "optimizer_state_dict": meta.meta_optimizer.state_dict(),
            "hyperparams": {
                "inner_lr": meta.inner_lr,
                "inner_steps": meta.inner_steps,
            },
        }

    def save_last(self, epoch: int, meta, metrics: Dict[str, float]):
        """Save the last checkpoint."""
        payload = self._pack_state(epoch, meta, metrics)
        torch.save(payload, self.last_path_pt)

    def save_periodic(self, epoch: int, meta, metrics: Dict[str, float]):
        """Save periodic checkpoint."""
        if self.save_every and epoch % self.save_every == 0:
            p = self.out_dir / f"epoch_{epoch:05d}.pt"
            torch.save(self._pack_state(epoch, meta, metrics), p)

    def save_if_best(
        self,
        epoch: int,
        meta,
        metrics: Dict[str, float],
        monitor: str = "meta_aucpr",
    ) -> bool:
        """Save if current metric is best."""
        score = float(metrics.get(monitor, float("-inf")))
        if score > self.best_metric:
            self.best_metric = score
            torch.save(self._pack_state(epoch, meta, metrics), self.best_path_pt)
            save_model(meta.model, str(self.best_path_safetensors))
            return True
        return False

    def try_resume(self, meta, resume_path: Optional[str]) -> int:
        """
        Resume from checkpoint.

        Returns:
            Starting epoch (checkpoint_epoch + 1)
        """
        if not resume_path:
            return 1

        ckpt_path = Path(resume_path)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "last_model.pt"

        if not ckpt_path.exists():
            print(f"[Resume] No checkpoint found at: {ckpt_path}")
            return 1

        payload = torch.load(ckpt_path, map_location="cpu")
        meta.model.load_state_dict(payload["model_state_dict"])
        meta.meta_optimizer.load_state_dict(payload["optimizer_state_dict"])

        if "metrics" in payload and "meta_aucpr" in payload["metrics"]:
            self.best_metric = float(payload["metrics"]["meta_aucpr"])

        start_epoch = int(payload.get("epoch", 0)) + 1
        print(f"[Resume] Loaded checkpoint from {ckpt_path}. Resuming at epoch {start_epoch}.")
        return start_epoch
