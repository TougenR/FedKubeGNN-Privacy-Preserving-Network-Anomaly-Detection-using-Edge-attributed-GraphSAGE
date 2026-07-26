"""Observed full-participation runtime used for proof and local experiments."""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.federated.contracts.task import FederatedTask, LocalTrainConfig
from src.federated.core.aggregation import weighted_fedavg
from src.federated.core.metrics import (
    aggregate_confusion_matrices,
    classification_metrics,
)
from src.federated.core.state import copy_array_state, state_nbytes
from src.federated.observability.events import (
    CompositeObserver,
    JsonlObserver,
    NoopObserver,
    Observer,
)
from src.federated.observability.run_store import RunStore, atomic_json, atomic_text
from src.federated.strategies.fedavg import FedAvgPolicy
from src.federated.strategies.fedprox import FedProxPolicy


@dataclass(frozen=True)
class ObservedRunResult:
    run_id: str
    run_root: Path
    best_round: int
    validation_macro_f1: float
    test_metrics: Mapping[str, Any]


def _evaluate(
    task: FederatedTask, state: Mapping[str, np.ndarray], split: str
) -> tuple[dict[str, Any], np.ndarray, float]:
    task_split = "val" if split == "validation" else split
    results = [
        task.evaluate_local(client_id, state, split=task_split)
        for client_id in task.client_ids
    ]
    matrix = aggregate_confusion_matrices(
        (result.confusion_matrix for result in results),
        num_classes=task.label_schema.num_classes,
    )
    metrics = classification_metrics(matrix, class_names=task.label_schema.classes)
    total = sum(result.num_examples for result in results)
    loss = (
        sum(result.loss * result.num_examples for result in results) / total
        if total
        else 0.0
    )
    metrics["loss"] = float(loss)
    return metrics, matrix, float(loss)


def _round_commit_path(root: Path, round_number: int) -> Path:
    return root / "metrics" / "rounds" / f"round-{round_number:04d}.json"


def _load_committed_rounds(root: Path) -> list[dict[str, Any]]:
    commit_root = root / "metrics" / "rounds"
    if not commit_root.exists():
        return []
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(commit_root.glob("round-*.json"))
    ]
    for expected_round, record in enumerate(records, start=1):
        if int(record.get("round", -1)) != expected_round:
            raise RuntimeError("Round commit markers are not contiguous from round 1.")
        checkpoint = root / str(record.get("checkpoint", ""))
        if not checkpoint.is_file():
            raise RuntimeError(
                f"Committed round {expected_round} is missing checkpoint {checkpoint}."
            )
    return records


def _publish_round_views(root: Path, records: list[Mapping[str, Any]]) -> None:
    jsonl_text = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    atomic_text(root / "metrics" / "rounds.jsonl", jsonl_text)
    csv_path = root / "metrics" / "rounds.csv"
    fields = [
        "round",
        "train_loss",
        "validation_loss",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "train_examples",
        "validation_examples",
        "upload_bytes",
        "download_bytes",
        "duration_seconds",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for record in records:
        writer.writerow({key: record[key] for key in fields})
    atomic_text(csv_path, buffer.getvalue())


def _commit_round(
    store: RunStore,
    record: dict[str, Any],
    checkpoint: Path,
) -> list[dict[str, Any]]:
    committed = dict(record)
    committed["checkpoint"] = str(checkpoint.relative_to(store.root))
    atomic_json(_round_commit_path(store.root, int(record["round"])), committed)
    records = _load_committed_rounds(store.root)
    _publish_round_views(store.root, records)
    store.mark_round_committed(int(record["round"]), checkpoint)
    return records


def _reconcile_run(store: RunStore) -> list[dict[str, Any]]:
    records = _load_committed_rounds(store.root)
    status = json.loads((store.root / "run.json").read_text(encoding="utf-8"))
    committed_round = int(records[-1]["round"]) if records else 0
    status_round = int(status.get("latest_round", 0))
    if status_round > committed_round:
        raise RuntimeError(
            "run.json is ahead of durable round commits; this legacy/incomplete "
            "run cannot be resumed safely."
        )
    _publish_round_views(store.root, records)
    if records:
        best = max(records, key=lambda item: float(item["macro_f1"]))
        store.promote_best(int(best["round"]))
        if status_round != committed_round:
            checkpoint = store.root / str(records[-1]["checkpoint"])
            store.mark_round_committed(committed_round, checkpoint)
    return records


def run_observed_inprocess(
    task: FederatedTask,
    *,
    policy: FedAvgPolicy | FedProxPolicy,
    num_rounds: int,
    train_config: LocalTrainConfig,
    output_root: str | Path,
    config_digest: str,
    config_snapshot: Mapping[str, Any],
    observer: Observer | None = None,
    resume_root: str | Path | None = None,
) -> ObservedRunResult:
    """Run rounds on validation, select the best checkpoint, then test once."""
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1.")
    observer = observer or NoopObserver()
    metadata = dict(task.metadata())
    dataset_digest = str(
        metadata.get("dataset_digest", metadata.get("dataset_id", task.task_id))
    )
    if resume_root is None:
        store = RunStore.create(
            output_root,
            strategy=policy.name,
            config_digest=config_digest,
            dataset_digest=dataset_digest,
            model_digest=task.model_spec.digest,
            config_snapshot=config_snapshot,
        )
    else:
        store = RunStore.resume(
            resume_root,
            strategy=policy.name,
            config_digest=config_digest,
            dataset_digest=dataset_digest,
            model_digest=task.model_spec.digest,
        )
    observer = CompositeObserver(
        observer, JsonlObserver(store.root / "events" / "server.jsonl")
    )
    prior = _reconcile_run(store)
    status = json.loads((store.root / "run.json").read_text(encoding="utf-8"))
    latest_round = int(prior[-1]["round"]) if prior else 0
    if resume_root is not None and status.get("status") == "completed":
        raise ValueError("A completed run cannot be resumed.")
    if latest_round >= num_rounds:
        raise ValueError(
            f"Resume checkpoint round {latest_round} is not below requested rounds={num_rounds}."
        )
    state = (
        store.load_checkpoint(store.root / str(status["latest_checkpoint"]))
        if latest_round
        else task.initial_state()
    )
    task.model_spec.validate_state(state)
    payload_bytes = state_nbytes(state)
    best_round, best_f1 = 0, -1.0
    if prior:
        best = max(prior, key=lambda item: float(item["macro_f1"]))
        best_round, best_f1 = int(best["round"]), float(best["macro_f1"])
    run_started = time.perf_counter()
    observer.emit(
        "run.started",
        run_id=store.run_id,
        component="runtime",
        strategy=policy.name,
        rounds=num_rounds,
        clients=len(task.client_ids),
        graph_protocol=metadata.get("graph_protocol"),
    )
    if latest_round:
        observer.emit(
            "run.resumed",
            run_id=store.run_id,
            component="runtime",
            strategy=policy.name,
            latest_round=latest_round,
            remaining_rounds=num_rounds - latest_round,
        )
    effective_config = LocalTrainConfig(
        local_epochs=train_config.local_epochs,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        grad_clip=train_config.grad_clip,
        optimizer=train_config.optimizer,
        proximal_mu=policy.proximal_mu,
        seed=train_config.seed,
    )
    try:
        for round_number in range(latest_round + 1, num_rounds + 1):
            started = time.perf_counter()
            observer.emit(
                "round.started",
                run_id=store.run_id,
                component="runtime",
                strategy=policy.name,
                round=round_number,
                clients=len(task.client_ids),
            )
            local_results = [
                task.train_local(client_id, copy_array_state(state), effective_config)
                for client_id in task.client_ids
            ]
            state = weighted_fedavg(local_results, model_spec=task.model_spec)
            validation, matrix, validation_loss = _evaluate(task, state, "validation")
            train_examples = sum(result.num_examples for result in local_results)
            weighted_train_loss = (
                sum(
                    result.metrics.get("train_loss", 0.0) * result.num_examples
                    for result in local_results
                )
                / train_examples
            )
            upload_bytes = payload_bytes * len(task.client_ids)
            download_bytes = payload_bytes * len(task.client_ids)
            macro_f1 = float(validation["macro_f1"])
            is_best = macro_f1 > best_f1
            if is_best:
                best_round, best_f1 = round_number, macro_f1
            record = {
                "round": round_number,
                "strategy": policy.name,
                "train_loss": float(weighted_train_loss),
                "validation_loss": validation_loss,
                "accuracy": float(validation["accuracy"]),
                "macro_f1": macro_f1,
                "weighted_f1": float(validation["weighted_f1"]),
                "train_examples": train_examples,
                "validation_examples": int(matrix.sum()),
                "upload_bytes": upload_bytes,
                "download_bytes": download_bytes,
                "duration_seconds": time.perf_counter() - started,
                "confusion_matrix": matrix.tolist(),
            }
            checkpoint = store.checkpoint(
                state,
                round_number=round_number,
                mark_latest=False,
            )
            _commit_round(store, record, checkpoint)
            if is_best:
                store.promote_best(round_number)
            observer.emit(
                "round.completed",
                run_id=store.run_id,
                component="runtime",
                strategy=policy.name,
                round=round_number,
                macro_f1=macro_f1,
                loss=validation_loss,
                upload_bytes=upload_bytes,
                download_bytes=download_bytes,
                duration_seconds=record["duration_seconds"],
            )

        best_state = store.load_checkpoint(
            store.root / "checkpoints" / "best_model.npz"
        )
        test_metrics, test_matrix, _ = _evaluate(task, best_state, "test")
        summary = {
            "run_id": store.run_id,
            "strategy": policy.name,
            "best_round": best_round,
            "validation_macro_f1": best_f1,
            "test_metrics": test_metrics,
            "test_confusion_matrix": test_matrix.tolist(),
            "total_upload_bytes": payload_bytes * len(task.client_ids) * num_rounds,
            "total_download_bytes": payload_bytes * len(task.client_ids) * num_rounds,
            "duration_seconds": time.perf_counter() - run_started,
        }
        atomic_json(store.root / "metrics" / "summary.json", summary)
        store.complete(
            best_round=best_round,
            validation_macro_f1=best_f1,
            test_macro_f1=float(test_metrics["macro_f1"]),
        )
        observer.emit(
            "run.completed",
            run_id=store.run_id,
            component="runtime",
            strategy=policy.name,
            best_round=best_round,
            validation_macro_f1=best_f1,
            test_macro_f1=float(test_metrics["macro_f1"]),
            duration_seconds=summary["duration_seconds"],
        )
        return ObservedRunResult(
            store.run_id, store.root, best_round, best_f1, test_metrics
        )
    except BaseException as error:
        store.fail(error)
        observer.emit(
            "run.failed",
            level="ERROR",
            run_id=store.run_id,
            component="runtime",
            strategy=policy.name,
            error_type=type(error).__name__,
            error_message=str(error),
            duration_seconds=time.perf_counter() - run_started,
        )
        raise
