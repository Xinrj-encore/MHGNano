from pathlib import Path
from typing import Dict, Optional
import torch
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, higher_is_better=True):
        self.patience = patience
        self.delta = delta
        self.higher_is_better = higher_is_better
        self.best = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current):
        if self.best is None:
            self.best = current
            return True

        if self.higher_is_better:
            improved = current > self.best + self.delta
        else:
            improved = current < self.best - self.delta

        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            print(f"Early stop counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return improved


class Checkpointer:
    def __init__(self, out_dir: str, save_every: int = 0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(save_every) if save_every else 0
        self.best_metric = -float("inf")
        self.best_path_pt = self.out_dir / "best_mhgnano.pt"
        self.best_path_safetensors = self.out_dir / "best_mhgnano.safetensors"
        self.last_path_pt = self.out_dir / "last_mhgnano.pt"

    def _pack_state(self, epoch: int, meta, metrics: Dict[str, float]):
        return {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": meta.model.state_dict(),
            "optimizer_state_dict": meta.meta_optimizer.state_dict(),
            "hyperparams": {
                "inner_lr": meta.inner_lr,
                "inner_steps": meta.inner_steps,
                "loss_type": meta.loss_type,
                "focal_gamma": meta.focal_gamma,
                "pairwise_lambda": meta.pairwise_lambda,
                "pairwise_margin": meta.pairwise_margin,
                "pairwise_max_pairs": meta.pairwise_max_pairs,
            },
        }

    def save_last(self, epoch: int, meta, metrics: Dict[str, float]):
        payload = self._pack_state(epoch, meta, metrics)
        torch.save(payload, self.last_path_pt)

    def save_periodic(self, epoch: int, meta, metrics: Dict[str, float]):
        if self.save_every and epoch % self.save_every == 0:
            p = self.out_dir / f"epoch_{epoch:05d}.pt"
            torch.save(self._pack_state(epoch, meta, metrics), p)

    def save_if_best(
        self, epoch: int, meta, metrics: Dict[str, float], monitor: str = "meta_aucpr"
    ):
        score = float(metrics.get(monitor, float("-inf")))
        if score > self.best_metric:
            self.best_metric = score
            torch.save(self._pack_state(epoch, meta, metrics), self.best_path_pt)
            try:
                from safetensors.torch import save_model
            except ImportError:
                print("safetensors is unavailable; saved the PyTorch checkpoint only")
            else:
                save_model(meta.model, str(self.best_path_safetensors))
            return True
        return False

    def try_resume(self, meta, resume_path: Optional[str]) -> int:
        """Load states if resume_path exists. Returns starting epoch+1 or 1 if none."""
        if not resume_path:
            return 1
        ckpt_path = Path(resume_path)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "last_mhgnano.pt"
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
