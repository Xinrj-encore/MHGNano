import math
from typing import Dict, List, Tuple

import numpy as np
import torch
from dhg import Hypergraph
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from .losses import binary_loss_with_logits, pairwise_auc_loss


class MetaTrainingMixin:
    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        base_loss = binary_loss_with_logits(
            logits,
            labels,
            loss_type=self.loss_type,
            focal_gamma=self.focal_gamma,
        )
        if self.loss_type == "pairwise_auc":
            return pairwise_auc_loss(
                logits,
                labels,
                margin=self.pairwise_margin,
                max_pairs=self.pairwise_max_pairs,
            )
        if self.loss_type == "bce_pairwise":
            rank_loss = pairwise_auc_loss(
                logits,
                labels,
                margin=self.pairwise_margin,
                max_pairs=self.pairwise_max_pairs,
            )
            return base_loss + self.pairwise_lambda * rank_loss
        return base_loss

    def _inner_loop(
        self,
        support_batch: Tuple[torch.Tensor, Hypergraph, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        真·二阶 MAML 内循环：
          - 使用 create_graph=True
          - 只更新 classifier.* 参数
          - 使用动态 pos_weight，防止“全负解”
        """
        X_s, hg_s, index_s, index_label = support_batch
        X_s = X_s.to(device)
        y_s = torch.tensor(index_label, device=device, dtype=torch.float32).view(-1)

        if hasattr(hg_s, "to"):
            hg_s = hg_s.to(device)

        # 不 clone，保留计算图
        fast_params: Dict[str, torch.Tensor] = {
            name: param for name, param in self.model.named_parameters() if param.requires_grad
        }

        for step in tqdm(range(self.inner_steps), desc="Inner_loop_training"):
            logits = self.model.functional_forward(X_s, hg_s, index_s, params=fast_params)
            logits = logits.view(-1)

            loss = self._loss(logits, y_s)

            with torch.no_grad():
                pos_logits = logits[y_s == 1.0]
                neg_logits = logits[y_s == 0.0]
                if pos_logits.numel() > 0 and neg_logits.numel() > 0:
                    print(
                        f"[inner step {step}] loss={loss.item():.4f}, "
                        f"pos_logits={pos_logits.mean().item():.3f}, "
                        f"neg_logits={neg_logits.mean().item():.3f}"
                    )

            grads = torch.autograd.grad(
                loss,
                list(fast_params.values()),
                create_graph=True,      # 二阶 MAML 关键
                retain_graph=True,
                allow_unused=True,
            )

            # name_list = list(fast_params.keys())
            new_fast_params: Dict[str, torch.Tensor] = {}

            for (name, param), g in zip(fast_params.items(), grads):
                if g is None:
                    g = torch.zeros_like(param)

                if name.startswith("classifier."):
                    new_fast_params[name] = param - self.inner_lr * g
                else:
                    # encoder 不在 inner-loop 中更新
                    new_fast_params[name] = param

            fast_params = new_fast_params

        return fast_params

    def meta_train_step(
        self,
        task_batch: List[Dict[str, Tuple[torch.Tensor, Hypergraph, torch.Tensor, torch.Tensor]]],
        device: torch.device,
    ) -> Dict[str, float]:
        self.model.train()
        meta_loss = 0.0
        meta_acc = 0.0
        meta_auroc = 0.0
        meta_aucpr = 0.0
        n_tasks = len(task_batch)

        self.meta_optimizer.zero_grad()

        for task in tqdm(task_batch, desc="Processing task"):
            # 1) inner-loop on support
            fast_params = self._inner_loop(task["support"], device)

            # 2) meta-loss on query with fast_params
            X_q, hg_q, index_q, index_label = task["query"]
            X_q = X_q.to(device)
            y_q = torch.tensor(index_label, device=device, dtype=torch.float32).view(-1)

            if hasattr(hg_q, "to"):
                hg_q = hg_q.to(device)

            # BEFORE：用当前 meta 初始化直接在 query 上看一下
            with torch.no_grad():
                logits_before = self.model.functional_forward(X_q, hg_q, index_q, params=None)
                logits_before = logits_before.view(-1)
                probs_before = torch.sigmoid(logits_before)
                preds_before = (probs_before > 0.5).float()
                acc_before = (preds_before == y_q).float().mean().item()
                try:
                    auc_roc_b = roc_auc_score(y_q.cpu().numpy(), probs_before.detach().cpu().numpy())
                    auc_pr_b = average_precision_score(y_q.cpu().numpy(), probs_before.detach().cpu().numpy())
                except ValueError:
                    auc_roc_b, auc_pr_b = float("nan"), float("nan")
                print(
                    f"[Query BEFORE] acc={acc_before:.4f}, "
                    f"auc_roc={auc_roc_b:.4f}, auc_pr={auc_pr_b:.4f}"
                )

            # AFTER：使用 fast_params
            logits_q = self.model.functional_forward(X_q, hg_q, index_q, params=fast_params)
            logits_q = logits_q.view(-1)

            loss_q = self._loss(logits_q, y_q)

            meta_loss += loss_q

            with torch.no_grad():
                probs = torch.sigmoid(logits_q)
                preds = (probs > 0.5).float()
                acc = (preds == y_q).float().mean().item()
                try:
                    auc_roc = roc_auc_score(y_q.cpu().numpy(), probs.detach().cpu().numpy())
                    auc_pr = average_precision_score(y_q.cpu().numpy(), probs.detach().cpu().numpy())
                except ValueError:
                    auc_roc, auc_pr = float("nan"), float("nan")
            print(f"[Query AFTER] loss={loss_q.item():.4f}，"
                  f"auc_roc={auc_roc:.4f}, auc_pr={auc_pr:.4f}")

            meta_acc += acc
            if not math.isnan(auc_roc):
                meta_auroc += auc_roc
            if not math.isnan(auc_pr):
                meta_aucpr += auc_pr

        meta_loss = meta_loss / n_tasks
        meta_loss.backward()
        self.meta_optimizer.step()

        return {
            "meta_loss": meta_loss.item(),
            "meta_acc": meta_acc / n_tasks,
            "meta_auroc": meta_auroc / n_tasks,
            "meta_aucpr": meta_aucpr / n_tasks,
        }
