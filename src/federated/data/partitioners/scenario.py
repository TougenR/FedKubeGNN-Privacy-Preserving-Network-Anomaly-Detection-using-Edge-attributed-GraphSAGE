"""Scenario clients with deterministic label-aware edge masks."""

from __future__ import annotations

import hashlib

import numpy as np


def deterministic_edge_masks(
    labels: np.ndarray,
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition each class independently; singleton labels always enter train."""
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) < 3:
        raise ValueError("At least three one-dimensional labels are required.")
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0.")
    masks = [np.zeros(len(y), dtype=np.bool_) for _ in range(3)]
    for label in sorted(np.unique(y), key=str):
        indices = np.flatnonzero(y == label)
        suffix = int.from_bytes(hashlib.sha256(str(label).encode()).digest()[:4], "big")
        rng = np.random.default_rng((seed + suffix) % (2**32))
        rng.shuffle(indices)
        count = len(indices)
        if count == 1:
            train_count, validation_count = 1, 0
        else:
            train_count = max(1, int(np.floor(count * train_ratio)))
            validation_count = int(np.floor(count * validation_ratio))
            # Preserve a test example for classes with enough rows.
            if count >= 3 and validation_count == 0:
                validation_count = 1
            if train_count + validation_count >= count:
                validation_count = max(0, count - train_count - 1)
        masks[0][indices[:train_count]] = True
        masks[1][indices[train_count : train_count + validation_count]] = True
        masks[2][indices[train_count + validation_count :]] = True
    coverage = masks[0].astype(np.int8) + masks[1] + masks[2]
    if not np.all(coverage == 1):  # defensive invariant
        raise RuntimeError("Split implementation failed coverage invariant.")
    return masks[0], masks[1], masks[2]
