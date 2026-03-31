"""
Model-Agnostic Meta-Learning (MAML) implementation for HyperGraph PPI prediction.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from dhg import Hypergraph
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from .hypergraph import HyperGraphMetaNetwork


class MetaLearner(nn.Module):
    """
    MAML-based meta-learner for cross-species PPI prediction.

    Implements the MAML algorithm with inner loop adaptation
    and outer loop meta-optimization.

    Args:
        base_model: The base HyperGraphMetaNetwork model
        inner_lr: Learning rate for inner loop adaptation
        inner_steps: Number of gradient steps in inner loop
        meta_lr: Learning rate for meta-optimizer
    """

    def __init__(
        self,
        base_model: HyperGraphMetaNetwork,
        inner_lr: float = 1e-2,
        inner_steps: int = 5,
        meta_lr: float = 1e-4,
    ):
        super().__init__()
        self.model = base_model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=meta_lr
        )
        self.criterion = nn.BCEWithLogitsLoss()

    @staticmethod
    def _clone_named_params(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Clone model parameters for inner loop adaptation."""
        return {
            name: p.clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    def _inner_loop(
        self,
        support_batch: Tuple[torch.Tensor, Hypergraph, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform inner loop adaptation on support set.

        Args:
            support_batch: (X_s, hg_s, index_s, index_label)
            device: Computation device

        Returns:
            Adapted parameters (fast_params)
        """
        X_s, hg_s, index_s, index_label = support_batch
        X_s = X_s.to(device)
        index_label = torch.tensor(index_label).to(device)
        if hasattr(hg_s, "to"):
            hg_s = hg_s.to(device)

        fast_params = self._clone_named_params(self.model)

        for _ in tqdm(range(self.inner_steps), desc="Inner loop"):
            logits = self.model.functional_forward(X_s, hg_s, index_s, params=fast_params)
            loss = self.criterion(logits.view_as(index_label), index_label)

            grads = torch.autograd.grad(
                loss, fast_params.values(),
                create_graph=False, retain_graph=False, allow_unused=True
            )
            grads = [
                torch.zeros_like(p) if g is None else g
                for p, g in zip(fast_params.values(), grads)
            ]

            # Update fast parameters
            fast_params = {
                name: param - self.inner_lr * g
                for (name, param), g in zip(fast_params.items(), grads)
            }

        return fast_params

    def meta_train_step(
        self,
        task_batch: List[Dict[str, Tuple]],
        device: torch.device,
    ) -> Tuple[Dict[str, float], List[float]]:
        """
        Perform one meta-training step.

        Args:
            task_batch: List of tasks, each with 'support' and 'query'
            device: Computation device

        Returns:
            metrics: Dictionary of meta-learning metrics
            before_loss: List of losses before adaptation
        """
        self.model.train()
        meta_loss = 0.0
        meta_acc = 0.0
        meta_auroc = 0.0
        meta_aucpr = 0.0
        n_tasks = len(task_batch)

        self.meta_optimizer.zero_grad()
        before_loss = []

        for task in tqdm(task_batch, desc="Processing tasks"):
            X_q, hg_q, index_q, index_label = task["query"]
            X_q = X_q.to(device)
            if hasattr(hg_q, "to"):
                hg_q = hg_q.to(device)
            index_label = torch.tensor(index_label).to(device)

            # Loss before adaptation
            logits_q = self.model.functional_forward(X_q, hg_q, index_q, params=None)
            loss_q_before = self.criterion(logits_q.view_as(index_label), index_label)

            # Inner loop adaptation
            fast_params = self._inner_loop(task["support"], device)

            # Loss after adaptation
            logits_q = self.model.functional_forward(X_q, hg_q, index_q, params=fast_params)
            loss_q = self.criterion(logits_q.view_as(index_label), index_label)

            meta_loss += loss_q
            before_loss.append(loss_q_before.item())

            # Metrics
            preds = (torch.sigmoid(logits_q) > 0.5).float()
            acc = (preds.view_as(index_label) == index_label).float().mean()
            auc_roc = roc_auc_score(
                index_label.cpu(),
                preds.view_as(index_label).detach().cpu().numpy(),
            )
            auc_pr = average_precision_score(
                index_label.cpu(),
                preds.view_as(index_label).detach().cpu().numpy(),
            )

            meta_aucpr += auc_pr
            meta_auroc += auc_roc
            meta_acc += acc.item()

        meta_loss = meta_loss / n_tasks
        meta_loss.backward()
        self.meta_optimizer.step()

        return {
            "meta_loss": meta_loss.item(),
            "meta_acc": meta_acc / n_tasks,
            "meta_auroc": meta_auroc / n_tasks,
            "meta_aucpr": meta_aucpr / n_tasks,
        }, before_loss

    def meta_evaluate(
        self,
        task_batch: List[Dict[str, Tuple]],
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate on meta-test tasks.

        Args:
            task_batch: List of test tasks
            device: Computation device

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.train()  # Keep in train mode for adaptation
        total_loss, total_acc = 0.0, 0.0

        for task in task_batch:
            X_q, hg_q, index_q, index_label = task["query"]
            X_q = X_q.to(device)
            if hasattr(hg_q, "to"):
                hg_q = hg_q.to(device)
            index_label = torch.tensor(index_label).to(device)

            # Evaluate before and after adaptation
            logits_q = self.model.functional_forward(X_q, hg_q, index_q, params=None)
            loss_q = self.criterion(logits_q.view_as(index_label), index_label)
            preds = (torch.sigmoid(logits_q) > 0.5).float()

            # Adapt and re-evaluate
            fast_params = self._inner_loop(task["support"], device)
            logits_q = self.model.functional_forward(X_q, hg_q, index_q, params=fast_params)
            loss_q = self.criterion(logits_q.view_as(index_label), index_label)
            preds = (torch.sigmoid(logits_q) > 0.5).float()
            acc = (preds.view_as(index_label) == index_label).float().mean()

            total_loss += loss_q.item()
            total_acc += acc.item()

        n = len(task_batch)
        return {"loss": total_loss / n, "acc": total_acc / n}
