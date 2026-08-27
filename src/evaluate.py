from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y_true: np.ndarray, probability: np.ndarray, metric: str = "f1") -> float:
    """Choose a classification threshold on validation data only."""
    if metric != "f1":
        raise ValueError("Only threshold_metric='f1' is currently supported")
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if thresholds.size == 0:
        return 0.5
    score = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(score))])


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    top_k_fraction: float,
) -> dict[str, object]:
    """Return discrimination, calibration, threshold, and capacity metrics."""
    prediction = (probability >= threshold).astype(int)
    k = max(1, math.ceil(len(y_true) * top_k_fraction))
    top_indices = np.argsort(-probability)[:k]
    positives = int(np.sum(y_true))
    captured = int(np.sum(y_true[top_indices]))

    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
        "top_k_fraction": float(top_k_fraction),
        "precision_at_k": float(captured / k),
        "recall_at_k": float(captured / positives) if positives else 0.0,
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }

