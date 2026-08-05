"""Evaluate a portable checkpoint against one prepared split."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from src.federated.config.schema import Phase2Config
from src.federated.core.metrics import (
    aggregate_confusion_matrices,
    classification_metrics,
)
from src.federated.core.simulation import merge_personalized_state
from src.federated.experiments.factory import manifest_task
from src.federated.observability.events import NoopObserver, Observer
from src.federated.observability.run_store import atomic_json


def evaluate_checkpoint(
    config: Phase2Config,
    dataset_root: str | Path,
    checkpoint: str | Path,
    *,
    split: str = "test",
    output: str | Path | None = None,
    observer: Observer | None = None,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test.")
    observer = observer or NoopObserver()
    task = manifest_task(config, dataset_root, observer=observer)
    with np.load(checkpoint, allow_pickle=False) as archive:
        state = {name: np.asarray(archive[name]).copy() for name in archive.files}
    task.model_spec.validate_state(state)
    results = [
        task.evaluate_local(client_id, state, split=split)
        for client_id in task.client_ids
    ]
    matrix = aggregate_confusion_matrices(
        (result.confusion_matrix for result in results),
        num_classes=task.label_schema.num_classes,
    )
    metrics = classification_metrics(matrix, class_names=task.label_schema.classes)
    total = sum(result.num_examples for result in results)
    metrics["loss"] = (
        sum(result.loss * result.num_examples for result in results) / total
        if total
        else 0.0
    )
    document = {
        "dataset_id": task.metadata()["dataset_id"],
        "model_digest": task.model_spec.digest,
        "checkpoint": str(checkpoint),
        "split": split,
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
    }
    if output is not None:
        atomic_json(Path(output), document)
    observer.emit(
        "checkpoint.evaluated",
        component="evaluation",
        dataset_id=str(document["dataset_id"]),
        split=split,
        examples=int(matrix.sum()),
        macro_f1=float(metrics["macro_f1"]),
        loss=float(metrics["loss"]),
        output=str(output) if output else None,
    )
    return document


def _load_array_state(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.asarray(archive[name]).copy() for name in archive.files
        }


def evaluate_personalized_checkpoint(
    config: Phase2Config,
    dataset_root: str | Path,
    shared_checkpoint: str | Path,
    personalized_checkpoints: str | Path,
    *,
    split: str = "test",
    output: str | Path | None = None,
    observer: Observer | None = None,
) -> dict[str, Any]:
    """Evaluate each client with one shared encoder and its private head."""
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test.")
    observer = observer or NoopObserver()
    task = manifest_task(config, dataset_root, observer=observer)
    shared_state = _load_array_state(shared_checkpoint)
    personalized_root = Path(personalized_checkpoints)
    results = []
    per_client: dict[str, dict[str, Any]] = {}
    checkpoint_files: dict[str, str] = {}
    for client_id in task.client_ids:
        path = personalized_root / f"{quote(client_id, safe='')}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing personalized checkpoint for '{client_id}': {path}"
            )
        private_state = _load_array_state(path)
        state = merge_personalized_state(shared_state, private_state)
        task.model_spec.validate_state(state)
        result = task.evaluate_local(client_id, state, split=split)
        metrics = classification_metrics(
            result.confusion_matrix, class_names=task.label_schema.classes
        )
        metrics["loss"] = result.loss
        per_client[client_id] = {
            "metrics": metrics,
            "confusion_matrix": result.confusion_matrix.tolist(),
        }
        checkpoint_files[client_id] = str(path)
        results.append(result)

    matrix = aggregate_confusion_matrices(
        (result.confusion_matrix for result in results),
        num_classes=task.label_schema.num_classes,
    )
    metrics = classification_metrics(matrix, class_names=task.label_schema.classes)
    total = sum(result.num_examples for result in results)
    metrics["loss"] = (
        sum(result.loss * result.num_examples for result in results) / total
        if total
        else 0.0
    )
    document = {
        "kind": "personalized_checkpoint_evaluation",
        "dataset_id": task.metadata()["dataset_id"],
        "model_digest": task.model_spec.digest,
        "shared_checkpoint": str(shared_checkpoint),
        "personalized_checkpoints": checkpoint_files,
        "split": split,
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "per_client": per_client,
    }
    if output is not None:
        atomic_json(Path(output), document)
    observer.emit(
        "personalized_checkpoint.evaluated",
        component="evaluation",
        dataset_id=str(document["dataset_id"]),
        split=split,
        examples=int(matrix.sum()),
        macro_f1=float(metrics["macro_f1"]),
        loss=float(metrics["loss"]),
        output=str(output) if output else None,
    )
    return document
