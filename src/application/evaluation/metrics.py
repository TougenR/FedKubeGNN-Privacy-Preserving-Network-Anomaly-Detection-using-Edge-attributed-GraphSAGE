"""Fixed-class metrics and summaries for scientific detection evaluation."""

from __future__ import annotations

import statistics
from typing import Iterable, Sequence

import numpy as np


def confusion_matrix(
    truth: np.ndarray, predictions: np.ndarray, *, num_classes: int
) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if truth.shape != predictions.shape:
        raise ValueError("Truth and prediction arrays must have equal shape.")
    if truth.size and (
        truth.min() < 0
        or predictions.min() < 0
        or truth.max() >= num_classes
        or predictions.max() >= num_classes
    ):
        raise ValueError("Labels are outside the fixed class vocabulary.")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (truth, predictions), 1)
    return matrix


def classification_metrics(
    matrix: np.ndarray, *, class_names: Sequence[str]
) -> dict:
    matrix = np.asarray(matrix, dtype=np.int64)
    size = len(class_names)
    if matrix.shape != (size, size):
        raise ValueError("Confusion matrix does not match class vocabulary.")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(size, dtype=float),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(size, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(size, dtype=float),
        where=(precision + recall) != 0,
    )
    total = int(matrix.sum())
    per_class = {
        name: {
            "precision": float(precision[index]) if support[index] else None,
            "recall": float(recall[index]) if support[index] else None,
            "f1": float(f1[index]) if support[index] else None,
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }
    return {
        "num_examples": total,
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1_fixed": float(f1.mean()),
        "weighted_f1": (
            float(np.dot(f1, support) / total) if total else 0.0
        ),
        "per_class": per_class,
    }


def numeric_summary(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": ordered[0],
        "mean": float(statistics.fmean(ordered)),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }
