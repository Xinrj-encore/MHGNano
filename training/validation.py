import math
from typing import Dict, List, Tuple

import torch
from dhg import Hypergraph
from sklearn.metrics import average_precision_score, roc_auc_score


class MetaValidationMixin:
    def meta_validate(
        self,
        task_batch: List[Dict[str, Tuple[torch.Tensor, Hypergraph, torch.Tensor, torch.Tensor]]],
        device: torch.device,
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_auc_roc = 0.0
        total_auc_pr = 0.0
        n_tasks = len(task_batch)

        with torch.no_grad():
            for task in task_batch:
                fast_params = self._inner_loop_eval(task["support"], device)
                X_q, hg_q, index_q, index_label = task["query"]
                X_q = X_q.to(device)
                y_q = torch.as_tensor(index_label, device=device, dtype=torch.float32).view(-1)
                if hasattr(hg_q, "to"):
                    hg_q = hg_q.to(device)
                logits_q = self.model.functional_forward(X_q, hg_q, index_q, params=fast_params).view(-1)
                loss_q = self._loss(logits_q, y_q)
                probs = torch.sigmoid(logits_q)
                preds = (probs > 0.5).float()
                total_acc += (preds == y_q).float().mean().item()
                try:
                    total_auc_roc += roc_auc_score(y_q.cpu().numpy(), probs.cpu().numpy())
                    total_auc_pr += average_precision_score(y_q.cpu().numpy(), probs.cpu().numpy())
                except ValueError:
                    pass
                total_loss += loss_q.item()

        if n_tasks == 0:
            raise ValueError("Validation requires at least one task")
        return {
            "val_loss": total_loss / n_tasks,
            "val_acc": total_acc / n_tasks,
            "val_auroc": total_auc_roc / n_tasks,
            "val_aucpr": total_auc_pr / n_tasks,
        }
