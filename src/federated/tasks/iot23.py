"""Lazy manifest-backed IoT-23 federated task."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from src.federated.adapters.phase1_iot23 import Phase1IoT23Task
from src.federated.contracts.artifacts import ContractBundle
from src.federated.contracts.schema import (
    ContractError,
    FeatureSchema,
    GraphSchema,
    LabelSchema,
    ModelSpec,
)
from src.federated.contracts.task import (
    ArrayState,
    EvaluationResult,
    LocalTrainConfig,
    LocalTrainResult,
)
from src.federated.data.manifest import PreparedDatasetManifest
from src.federated.data.storage import load_graph_arrays, load_pyg_graph
from src.federated.observability.events import NoopObserver, Observer


GraphLoader = Callable[[str | Path], Any]


class ManifestIoT23Task:
    """Load the contract at startup and only load a selected client's graph."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        model_factory: Callable[[Any], Any],
        imbalance_mode: str = "class_weight",
        class_weight_scope: str = "local",
        device: str | None = None,
        graph_loader: GraphLoader = load_pyg_graph,
        observer: Observer | None = None,
    ) -> None:
        # Server/task construction verifies the shared contract and initial
        # state but deliberately does not read every client array. A selected
        # client verifies its own checksums inside graph_loader.
        self._manifest = PreparedDatasetManifest.load(
            dataset_root, verify=True, verify_clients=False
        )
        self._bundle = ContractBundle.load(
            self._manifest.root / str(self._manifest.document["contract_path"])
        )
        if self._bundle.model_spec is None:
            raise ContractError(
                "Manifest task requires a model_spec in its contract bundle."
            )
        with np.load(
            self._manifest.root / str(self._manifest.document["initial_state_path"]),
            allow_pickle=False,
        ) as archive:
            self._initial_state = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
        self._bundle.model_spec.validate_state(self._initial_state)
        self._model_factory = model_factory
        self._imbalance_mode = imbalance_mode
        if class_weight_scope not in {"local", "global"}:
            raise ContractError("class_weight_scope must be local or global.")
        if imbalance_mode == "none" and class_weight_scope != "local":
            raise ContractError(
                "class_weight_scope=global requires imbalance_mode='class_weight'."
            )
        self._class_weight_scope = class_weight_scope
        self._global_class_weights = (
            self._compute_global_class_weights()
            if class_weight_scope == "global"
            else None
        )
        self._device = device
        self._graph_loader = graph_loader
        self._observer = observer or NoopObserver()
        self._adapters: dict[str, Phase1IoT23Task] = {}

    def _compute_global_class_weights(self) -> np.ndarray:
        counts = np.zeros(self._bundle.label_schema.num_classes, dtype=np.int64)
        for client_id in self._manifest.client_ids:
            graph = load_graph_arrays(self._manifest.client_path(client_id), verify=True)
            counts += np.bincount(
                graph.edge_label[graph.train_mask], minlength=counts.size
            )
        present = counts > 0
        if not np.any(present):
            raise ContractError("Prepared dataset has no global training labels.")
        weights = np.zeros(counts.size, dtype=np.float32)
        weights[present] = counts.sum() / (present.sum() * counts[present])
        return weights

    @property
    def task_id(self) -> str:
        return f"{self._manifest.dataset_id}-{self.model_spec.family}"

    @property
    def client_ids(self) -> tuple[str, ...]:
        return self._manifest.client_ids

    @property
    def feature_schema(self) -> FeatureSchema:
        return self._bundle.feature_schema

    @property
    def label_schema(self) -> LabelSchema:
        return self._bundle.label_schema

    @property
    def graph_schema(self) -> GraphSchema:
        return self._bundle.graph_schema

    @property
    def model_spec(self) -> ModelSpec:
        assert self._bundle.model_spec is not None
        return self._bundle.model_spec

    def initial_state(self) -> ArrayState:
        return {name: value.copy() for name, value in self._initial_state.items()}

    def _adapter(self, client_id: str) -> Phase1IoT23Task:
        if client_id in self._adapters:
            return self._adapters[client_id]
        started = time.perf_counter()
        self._manifest.verify_client_digest(client_id)
        graph = self._graph_loader(self._manifest.client_path(client_id))
        # Portable graph files intentionally omit class names; restore only the
        # non-sensitive schema metadata required by the compatibility adapter.
        graph.num_classes = self.label_schema.num_classes
        graph.class_to_idx = self.label_schema.class_to_idx
        graph.feature_dim = self.feature_schema.feature_dim
        adapter = Phase1IoT23Task(
            client_graphs={client_id: graph},
            feature_columns=self.feature_schema.names,
            class_to_idx=self.label_schema.class_to_idx,
            model_factory=self._model_factory,
            model_family=self.model_spec.family,
            model_version=self.model_spec.model_version,
            model_hyperparameters=self.model_spec.hyperparameters,
            imbalance_mode=self._imbalance_mode,
            fixed_class_weights=self._global_class_weights,
            device=self._device,
            source_metadata={"dataset_id": self._manifest.dataset_id},
        )
        self.model_spec.assert_architecture_compatible(adapter.model_spec)
        self._adapters[client_id] = adapter
        metadata = self._manifest.client_path(client_id) / "metadata.json"
        self._observer.emit(
            "client.graph_loaded",
            component="task",
            client_id=client_id,
            artifact_bytes=sum(
                path.stat().st_size
                for path in self._manifest.client_path(client_id).iterdir()
                if path.is_file()
            ),
            metadata_path=str(metadata),
            duration_seconds=time.perf_counter() - started,
        )
        return adapter

    def train_local(
        self,
        client_id: str,
        global_state: Mapping[str, np.ndarray],
        config: LocalTrainConfig,
    ) -> LocalTrainResult:
        started = time.perf_counter()
        try:
            result = self._adapter(client_id).train_local(
                client_id, global_state, config
            )
            self._observer.emit(
                "client.train_completed",
                component="task",
                client_id=client_id,
                examples=result.num_examples,
                train_loss=result.metrics.get("train_loss"),
                duration_seconds=time.perf_counter() - started,
            )
            return result
        except BaseException as error:
            self._observer.emit(
                "client.train_failed",
                level="ERROR",
                component="task",
                client_id=client_id,
                error_type=type(error).__name__,
                error_message=str(error),
                duration_seconds=time.perf_counter() - started,
            )
            raise

    def evaluate_local(
        self, client_id: str, state: Mapping[str, np.ndarray], *, split: str
    ) -> EvaluationResult:
        phase1_split = "val" if split == "validation" else split
        started = time.perf_counter()
        result = self._adapter(client_id).evaluate_local(
            client_id, state, split=phase1_split
        )
        self._observer.emit(
            "client.evaluate_completed",
            component="task",
            client_id=client_id,
            split=split,
            examples=result.num_examples,
            loss=result.loss,
            duration_seconds=time.perf_counter() - started,
        )
        return result

    def metadata(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "dataset_id": self._manifest.dataset_id,
            "dataset_digest": self._manifest.digest,
            "graph_protocol": self._manifest.document["graph_protocol"],
            "loaded_clients": sorted(self._adapters),
            "class_weight_scope": self._class_weight_scope,
            "global_class_weights": (
                self._global_class_weights.tolist()
                if self._global_class_weights is not None
                else None
            ),
        }
