from typing import Optional

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


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def best_threshold_f1(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 200)

    best_threshold = 0.5
    best_f1 = 0.0
    # Try each threshold and keep the one with the best F1.
    for threshold in thresholds:
        f1 = f1_score(labels, probabilities >= threshold, zero_division=0)
        if f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(f1)

    return best_threshold, best_f1


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: Optional[float] = None,
) -> dict:
    labels = labels.astype(int)
    probabilities = probabilities.astype(float)
    if threshold is None:
        threshold, _ = best_threshold_f1(probabilities, labels)

    # Convert scores to predictions and count confusion matrix cells.
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(labels)) == 2
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both_classes else None,
        "average_precision": (
            float(average_precision_score(labels, probabilities)) if has_both_classes else None
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
