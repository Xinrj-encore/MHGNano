"""Threshold-dependent and threshold-independent binary metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Evaluate one score vector using the fixed manuscript protocol."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if not len(y_true):
        raise ValueError("Cannot evaluate an empty score vector.")

    y_pred = (y_prob >= float(threshold)).astype(int)
    has_both_classes = len(np.unique(y_true)) == 2
    auroc = roc_auc_score(y_true, y_prob) if has_both_classes else float("nan")
    auprc = (
        average_precision_score(y_true, y_prob)
        if has_both_classes
        else float("nan")
    )
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    result.update(
        {
            "acc": result["accuracy"],
            "auc_roc": result["auroc"],
            "auc_pr": result["auprc"],
        }
    )
    return result


def print_metrics(metrics: Dict[str, object], prefix: str = "") -> None:
    """Print the six reported metrics in a stable order."""
    label = f"{prefix} " if prefix else ""
    print(f"{label}Accuracy:  {float(metrics['accuracy']):.4f}")
    print(f"{label}Precision: {float(metrics['precision']):.4f}")
    print(f"{label}Recall:    {float(metrics['recall']):.4f}")
    print(f"{label}F1:        {float(metrics['f1']):.4f}")
    print(f"{label}AUROC:     {float(metrics['auroc']):.4f}")
    print(f"{label}AUPRC:     {float(metrics['auprc']):.4f}")
