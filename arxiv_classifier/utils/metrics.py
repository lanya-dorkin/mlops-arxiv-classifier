"""Metrics computation utilities."""

from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def compute_macro_f1(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute macro F1 score.

    Args:
        predictions: Predicted labels
        targets: Target labels

    Returns:
        Macro F1 score
    """
    return f1_score(targets, predictions, average="macro", zero_division=0)


def compute_per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Compute per-class precision, recall, and F1.

    Args:
        predictions: Predicted labels
        targets: Target labels
        class_names: Optional list of class names

    Returns:
        Dictionary with per-class metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )

    results = {}
    for idx in range(len(precision)):
        if class_names is not None and len(class_names) > idx:
            class_name = class_names[idx]
        else:
            class_name = f"class_{idx}"

        results[class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return results


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute accuracy.

    Args:
        predictions: Predicted labels
        targets: Target labels

    Returns:
        Accuracy score
    """
    return float(np.mean(predictions == targets))
