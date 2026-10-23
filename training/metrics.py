from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
def safe_binary_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> Dict[str, object]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, object] = {
        "threshold": float(threshold),
        "acc": float((y_pred == y_true).mean()) if y_true.size else float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "auroc": float("nan"),
        "aucpr": float("nan"),
        "confusion_matrix": None,
    }

    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return metrics

    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["auroc"] = roc_auc_score(y_true, y_prob)
    metrics["aucpr"] = average_precision_score(y_true, y_prob)
    return metrics


def select_threshold_by_f1(y_true, y_prob) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5, float("nan")

    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])

