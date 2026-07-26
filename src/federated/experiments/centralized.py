"""Clean centralized reference over the exact prepared client artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.federated.adapters.phase1_iot23 import (
    Phase1IoT23Task,
    make_phase1_model_factory,
)
from src.federated.config.schema import Phase2Config
from src.federated.contracts.artifacts import ContractBundle
from src.federated.contracts.task import LocalTrainConfig
from src.federated.core.metrics import classification_metrics
from src.federated.data.manifest import PreparedDatasetManifest
from src.federated.data.storage import load_pyg_graph
from src.federated.observability.events import NoopObserver, Observer
from src.federated.observability.run_store import RunStore, atomic_json


def run_centralized_reference(
    config: Phase2Config,
    dataset_root: str | Path,
    *,
    output_root: str | Path,
    observer: Observer | None = None,
) -> Path:
    """Train one model over all six graphs using only their train masks."""
    observer = observer or NoopObserver()
    manifest = PreparedDatasetManifest.load(dataset_root, verify=True)
    bundle = ContractBundle.load(
        manifest.root / str(manifest.document["contract_path"])
    )
    if bundle.model_spec is None:
        raise ValueError("Prepared dataset contract has no model spec.")
    started = time.perf_counter()
    graphs = [
        load_pyg_graph(manifest.client_path(client_id))
        for client_id in manifest.client_ids
    ]
    for graph in graphs:
        graph.feature_dim = bundle.feature_schema.feature_dim
        graph.num_classes = bundle.label_schema.num_classes
        graph.class_to_idx = bundle.label_schema.class_to_idx
    task = Phase1IoT23Task(
        client_graphs={"centralized": tuple(graphs)},
        feature_columns=bundle.feature_schema.names,
        class_to_idx=bundle.label_schema.class_to_idx,
        model_factory=make_phase1_model_factory(
            model_name=config.components.model, cfg={"model": config.model.__dict__}
        ),
        model_family=bundle.model_spec.family,
        model_version=bundle.model_spec.model_version,
        model_hyperparameters=bundle.model_spec.hyperparameters,
        imbalance_mode=config.training.imbalance,
        source_metadata={
            "dataset_id": manifest.dataset_id,
            "dataset_digest": manifest.digest,
            "graph_protocol": manifest.document["graph_protocol"],
        },
    )
    bundle.model_spec.assert_architecture_compatible(task.model_spec)
    store = RunStore.create(
        output_root,
        strategy="centralized",
        config_digest=config.digest,
        dataset_digest=manifest.digest,
        model_digest=bundle.model_spec.digest,
        config_snapshot=config.to_dict(),
    )
    try:
        import numpy as np

        with np.load(
            manifest.root / str(manifest.document["initial_state_path"]),
            allow_pickle=False,
        ) as archive:
            shared_initial_state = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
        task.model_spec.validate_state(shared_initial_state)
        state = task.train_local(
            "centralized",
            shared_initial_state,
            LocalTrainConfig(
                local_epochs=config.training.centralized_epochs,
                learning_rate=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                grad_clip=config.training.grad_clip,
                optimizer=config.training.optimizer,
                seed=config.training.seed,
            ),
        ).state
        validation = task.evaluate_local("centralized", state, split="val")
        test = task.evaluate_local("centralized", state, split="test")
        validation_metrics = classification_metrics(
            validation.confusion_matrix, class_names=task.label_schema.classes
        )
        test_metrics = classification_metrics(
            test.confusion_matrix, class_names=task.label_schema.classes
        )
        validation_metrics["loss"], test_metrics["loss"] = validation.loss, test.loss
        store.checkpoint(
            state, round_number=config.training.centralized_epochs, best=True
        )
        summary: dict[str, Any] = {
            "run_id": store.run_id,
            "kind": "centralized_reference",
            "epochs": config.training.centralized_epochs,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "duration_seconds": time.perf_counter() - started,
            "note": "Fresh benchmark reference; historical Phase 1 score is not reused.",
        }
        atomic_json(store.root / "metrics/summary.json", summary)
        store.complete(
            validation_macro_f1=float(validation_metrics["macro_f1"]),
            test_macro_f1=float(test_metrics["macro_f1"]),
        )
        observer.emit(
            "centralized.completed",
            run_id=store.run_id,
            component="experiment",
            epochs=config.training.centralized_epochs,
            validation_macro_f1=float(validation_metrics["macro_f1"]),
            test_macro_f1=float(test_metrics["macro_f1"]),
            duration_seconds=summary["duration_seconds"],
        )
        return store.root
    except BaseException as error:
        store.fail(error)
        observer.emit(
            "centralized.failed",
            level="ERROR",
            run_id=store.run_id,
            component="experiment",
            error_type=type(error).__name__,
            error_message=str(error),
            duration_seconds=time.perf_counter() - started,
        )
        raise
