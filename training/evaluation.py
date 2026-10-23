import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from dhg import Hypergraph
from tqdm import tqdm
from .metrics import safe_binary_metrics, select_threshold_by_f1
from .adaptation import InnerLoopAdaptation
from .validation import MetaValidationMixin


class MetaEvaluationMixin(InnerLoopAdaptation, MetaValidationMixin):
    def meta_evaluate(
        self,
        task_batch: List[Dict[str, Tuple[torch.Tensor, Hypergraph, torch.Tensor, torch.Tensor]]],
        device: torch.device,
        threshold: Optional[float] = 0.5,
        select_threshold: bool = False,
    ) -> Dict[str, float]:
        """
        简化版评估：对每个 task 做 inner-loop（不反向），再在 query 上评估。
        这里为了省事仍然调用 _inner_loop（里面有 create_graph=True，虽浪费一点，但能复用逻辑）。
        如果显存紧张，可以写一个不需要 create_graph 的 eval_inner_loop。
        """
        self.model.train()  # 需要 dropout / LN 的训练行为
        task_outputs = []
        before_outputs = []
        before_thresholds = []
        selected_thresholds = []

        for task in task_batch:
            # inner-loop
            with torch.enable_grad():
                fast_params = self._inner_loop_eval(task["support"], device)

            X_q, hg_q, index_q, index_label = task["query"]
            X_q = X_q.to(device)
            y_q = torch.tensor(index_label, device=device, dtype=torch.float32).view(-1)
            if hasattr(hg_q, "to"):
                hg_q = hg_q.to(device)

            logits_q = self.model.functional_forward(X_q, hg_q, index_q, params=None)
            logits_q = logits_q.view(-1)

            # pos = (y_q > 0.5).float().sum().clamp_min(1.0)
            # neg = y_q.numel() - pos
            # pos_weight = (neg / pos).detach()

            # loss_q = F.binary_cross_entropy_with_logits(logits_q, y_q, pos_weight=pos_weight)

            probs = torch.sigmoid(logits_q)
            y_np = y_q.detach().cpu().numpy()
            prob_np = probs.detach().cpu().numpy()
            if select_threshold:
                threshold_before, _ = select_threshold_by_f1(y_np, prob_np)
            else:
                threshold_before = 0.5 if threshold is None else threshold
            before_metrics = safe_binary_metrics(y_np, prob_np, threshold_before)
            before_outputs.append(before_metrics)
            before_thresholds.append(threshold_before)

            print(
                f"Before_Eval_Threshold:{before_metrics['threshold']}, "
                f"Before_Eval_ACC:{before_metrics['acc']}, "
                f"Before_Eval_Precision:{before_metrics['precision']}, "
                f"Before_Eval_Recall:{before_metrics['recall']}, "
                f"Before_Eval_F1:{before_metrics['f1']}, "
                f"Before_Eval_AUC_ROC:{before_metrics['auroc']}, "
                f"Before_Eval_AUC_PR:{before_metrics['aucpr']}"
            )
            if before_metrics["confusion_matrix"] is not None:
                print(before_metrics["confusion_matrix"])

            #-------------------After--------------
            logits_q = self.model.functional_forward(X_q, hg_q, index_q, params=fast_params)
            logits_q = logits_q.view(-1)

            loss_q = self._loss(logits_q, y_q)


            probs = torch.sigmoid(logits_q)

            y_np = y_q.detach().cpu().numpy()
            prob_np = probs.detach().cpu().numpy()
            if select_threshold:
                eval_threshold, _ = select_threshold_by_f1(y_np, prob_np)
            else:
                eval_threshold = 0.5 if threshold is None else threshold
            eval_metrics = safe_binary_metrics(y_np, prob_np, eval_threshold)
            selected_thresholds.append(eval_threshold)

            print(
                f"Eval_Threshold:{eval_metrics['threshold']}, "
                f"Eval_ACC:{eval_metrics['acc']}, "
                f"Eval_Precision:{eval_metrics['precision']}, "
                f"Eval_Recall:{eval_metrics['recall']}, "
                f"Eval_F1:{eval_metrics['f1']}, "
                f"Eval_AUC_ROC:{eval_metrics['auroc']}, "
                f"Eval_AUC_PR:{eval_metrics['aucpr']}"
            )
            if eval_metrics["confusion_matrix"] is not None:
                print(eval_metrics["confusion_matrix"])

            task_outputs.append((loss_q.item(), eval_metrics))

        n = len(task_batch)
        after_outputs = [item[1] for item in task_outputs]

        def metric_mean(items, key):
            values = []
            for item in items:
                value = float(item.get(key, float("nan")))
                if not math.isnan(value):
                    values.append(value)
            return float(np.mean(values)) if values else float("nan")

        total_loss = sum(item[0] for item in task_outputs)
        before_threshold = float(np.mean(before_thresholds)) if before_thresholds else float("nan")
        selected_threshold = float(np.mean(selected_thresholds)) if selected_thresholds else 0.5
        return {
            "val_loss": total_loss / n,
            "before_acc": metric_mean(before_outputs, "acc"),
            "before_precision": metric_mean(before_outputs, "precision"),
            "before_recall": metric_mean(before_outputs, "recall"),
            "before_f1": metric_mean(before_outputs, "f1"),
            "before_auroc": metric_mean(before_outputs, "auroc"),
            "before_aucpr": metric_mean(before_outputs, "aucpr"),
            "before_threshold": before_threshold,
            "acc": metric_mean(after_outputs, "acc"),
            "precision": metric_mean(after_outputs, "precision"),
            "recall": metric_mean(after_outputs, "recall"),
            "f1": metric_mean(after_outputs, "f1"),
            "val_auroc": metric_mean(after_outputs, "auroc"),
            "val_aucpr": metric_mean(after_outputs, "aucpr"),
            "threshold": selected_threshold,
        }
