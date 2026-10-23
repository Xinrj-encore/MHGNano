"""MAML-style training and evaluation for MHGNano."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg import Hypergraph

from .hypergraph import HyperGraphMetaNetwork
from utils.metrics import evaluate_metrics


class MetaLearner(nn.Module):
    """Meta-learner with classifier-only target adaptation.

    The protein encoder and hypergraph layers remain fixed during the inner
    loop; the pair classifier is adapted from the meta-learned initialization.
    """

    def __init__(
        self,
        base_model: HyperGraphMetaNetwork,
        inner_lr: float = 1e-3,
        inner_steps: int = 5,
        meta_lr: float = 1e-4,
        loss_type: str = "bce",
    ):
        super().__init__()
        if loss_type not in {"bce", "weighted_bce"}:
            raise ValueError("loss_type must be 'bce' or 'weighted_bce'.")
        self.model = base_model
        self.inner_lr = float(inner_lr)
        self.inner_steps = int(inner_steps)
        self.loss_type = loss_type
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(meta_lr)
        )

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        labels = labels.float().reshape(-1)
        pos_weight = None
        if self.loss_type == "weighted_bce":
            positives = labels.sum().clamp_min(1.0)
            negatives = (labels.numel() - labels.sum()).clamp_min(1.0)
            pos_weight = negatives / positives
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight
        )

    @staticmethod
    def _adaptable(name: str) -> bool:
        return name.startswith("classifier.")

    def _fast_params(self) -> Dict[str, torch.Tensor]:
        return {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def _inner_loop(
        self,
        support_batch: Tuple[torch.Tensor, Hypergraph, object, object],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        X_s, hg_s, pair_s, labels_s = support_batch
        X_s = X_s.to(device)
        labels_s = torch.as_tensor(labels_s, device=device, dtype=torch.float32)
        hg_s = hg_s.to(device) if hasattr(hg_s, "to") else hg_s
        fast_params = self._fast_params()

        for _ in range(self.inner_steps):
            logits = self.model.functional_forward(
                X_s, hg_s, pair_s, params=fast_params
            )
            loss = self._loss(logits, labels_s)
            grads = torch.autograd.grad(
                loss,
                tuple(fast_params.values()),
                create_graph=False,
                allow_unused=True,
            )
            updated = {}
            for (name, parameter), gradient in zip(fast_params.items(), grads):
                if gradient is not None and self._adaptable(name):
                    updated[name] = parameter - self.inner_lr * gradient
                else:
                    updated[name] = parameter
            fast_params = updated
        return fast_params

    def _query_logits(
        self,
        task: Dict[str, Tuple],
        device: torch.device,
        adapt: bool,
    ):
        if adapt and self.inner_steps > 0:
            with torch.enable_grad():
                fast_params = self._inner_loop(task["support"], device)
        else:
            fast_params = None

        X_q, hg_q, pair_q, labels_q = task["query"]
        X_q = X_q.to(device)
        hg_q = hg_q.to(device) if hasattr(hg_q, "to") else hg_q
        labels_q = torch.as_tensor(labels_q, device=device, dtype=torch.float32)
        logits = self.model.functional_forward(
            X_q, hg_q, pair_q, params=fast_params
        )
        return logits.reshape(-1), labels_q

    def meta_train_step(
        self,
        task_batch: List[Dict[str, Tuple]],
        device: torch.device,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        self.model.train()
        self.meta_optimizer.zero_grad()
        query_losses = []
        metric_rows = []

        for task in task_batch:
            logits, labels = self._query_logits(task, device, adapt=True)
            query_losses.append(self._loss(logits, labels))
            with torch.no_grad():
                metric_rows.append(
                    evaluate_metrics(
                        labels.cpu().numpy(),
                        torch.sigmoid(logits).cpu().numpy(),
                        threshold=threshold,
                    )
                )

        meta_loss = torch.stack(query_losses).mean()
        meta_loss.backward()
        self.meta_optimizer.step()

        def mean_metric(key: str) -> float:
            return float(sum(row[key] for row in metric_rows) / len(metric_rows))

        return {
            "meta_loss": float(meta_loss.item()),
            "meta_acc": mean_metric("accuracy"),
            "meta_precision": mean_metric("precision"),
            "meta_recall": mean_metric("recall"),
            "meta_f1": mean_metric("f1"),
            "meta_auroc": mean_metric("auroc"),
            "meta_aucpr": mean_metric("auprc"),
            "threshold": float(threshold),
        }

    def meta_evaluate(
        self,
        task_batch: List[Dict[str, Tuple]],
        device: torch.device,
        threshold: float = 0.5,
        adapt: bool = True,
    ) -> Dict[str, float]:
        """Evaluate each task on its query split, optionally using support."""
        self.model.train() if adapt else self.model.eval()
        rows = []
        losses = []
        for task in task_batch:
            if adapt:
                logits, labels = self._query_logits(task, device, adapt=True)
            else:
                with torch.no_grad():
                    logits, labels = self._query_logits(task, device, adapt=False)
            losses.append(float(self._loss(logits, labels).item()))
            with torch.no_grad():
                rows.append(
                    evaluate_metrics(
                        labels.cpu().numpy(),
                        torch.sigmoid(logits).cpu().numpy(),
                        threshold=threshold,
                    )
                )

        if not rows:
            raise ValueError("No evaluation tasks were provided.")
        return {
            "loss": float(sum(losses) / len(losses)),
            "accuracy": float(sum(row["accuracy"] for row in rows) / len(rows)),
            "precision": float(sum(row["precision"] for row in rows) / len(rows)),
            "recall": float(sum(row["recall"] for row in rows) / len(rows)),
            "f1": float(sum(row["f1"] for row in rows) / len(rows)),
            "auroc": float(sum(row["auroc"] for row in rows) / len(rows)),
            "auprc": float(sum(row["auprc"] for row in rows) / len(rows)),
            "threshold": float(threshold),
            "adapt": bool(adapt),
        }
