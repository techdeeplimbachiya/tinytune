"""Evaluation metrics implemented with NumPy only (no scikit-learn needed)."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    preds = np.asarray(predictions)
    labs = np.asarray(labels)
    if preds.shape != labs.shape:
        raise ValueError("predictions and labels must have the same shape")
    if preds.size == 0:
        raise ValueError("cannot compute accuracy on empty inputs")
    return float((preds == labs).mean())


def f1_binary(predictions: Sequence[int], labels: Sequence[int], positive_label: int = 1) -> float:
    """F1 score for the positive class. Returns 0.0 when there are no true positives."""
    preds = np.asarray(predictions)
    labs = np.asarray(labels)
    if preds.shape != labs.shape:
        raise ValueError("predictions and labels must have the same shape")
    tp = int(((preds == positive_label) & (labs == positive_label)).sum())
    fp = int(((preds == positive_label) & (labs != positive_label)).sum())
    fn = int(((preds != positive_label) & (labs == positive_label)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def f1_macro(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """Unweighted mean of per-class F1 over every class seen in labels or predictions."""
    preds = np.asarray(predictions)
    labs = np.asarray(labels)
    classes = np.union1d(np.unique(preds), np.unique(labs))
    if classes.size == 0:
        raise ValueError("cannot compute F1 on empty inputs")
    return float(np.mean([f1_binary(preds, labs, positive_label=int(c)) for c in classes]))


def compute_metrics(predictions: Sequence[int], labels: Sequence[int]) -> Dict[str, float]:
    return {
        "accuracy": round(accuracy(predictions, labels), 4),
        "f1": round(f1_macro(predictions, labels), 4),
    }
