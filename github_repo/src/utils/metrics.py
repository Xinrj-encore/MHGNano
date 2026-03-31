"""
Evaluation metrics for PPI prediction.
"""
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


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluate model predictions.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """Print metrics in a formatted way."""
    print(f"\n{prefix}Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
