"""Fail-closed adapter for the current IoT-23/E-GraphSAGE Phase 1 task.

Only this module is allowed to know the custom PyG ``Data`` fields used by
Phase 1. The federated core consumes the public ``FederatedTask`` contract.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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
from src.federated.core.metrics import (
    aggregate_confusion_matrices,
    confusion_matrix_from_predictions,
)
from src.federated.core.state import arrays_to_torch_state, torch_state_to_arrays


ModelFactory = Callable[[Any], nn.Module]


class Phase1AdapterError(RuntimeError):
    """Raised when Phase 1 cannot satisfy the explicit Phase 2 contract."""


def make_phase1_model_factory(
    *,
    model_name: str,
    cfg: Mapping[str, Any],
) -> ModelFactory:
    """Create a lazy factory so importing Phase 2 never imports PyG."""

    def factory(graph: Any) -> nn.Module:
        try:
            from src.core.model import build_model
        except (ImportError, ModuleNotFoundError) as exc:
            raise Phase1AdapterError(
                "The Phase 1 model adapter requires torch-geometric and the "
                "dependencies needed by src.model. Install the Phase 1 runtime "
                "before constructing the IoT-23 task."
            ) from exc
        return build_model(model_name, graph, dict(cfg))

    return factory


def _stable_seed(base_seed: int, client_id: str) -> int:
    suffix = int.from_bytes(
        hashlib.sha256(client_id.encode("utf-8")).digest()[:4], "big"
    )
    return (int(base_seed) + suffix) % (2**31)


def _ordered_labels(class_to_idx: Mapping[str, int]) -> tuple[str, ...]:
    indices = sorted(int(index) for index in class_to_idx.values())
    expected = list(range(len(indices)))
    if indices != expected:
        raise ContractError(
            f"class_to_idx must be contiguous [0, K); got {indices}."
        )
    inverse = {int(index): str(name) for name, index in class_to_idx.items()}
    return tuple(inverse[index] for index in expected)


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(size) for size in value.shape)


def _validate_bool_mask(mask: Any, *, name: str, num_edges: int) -> None:
    if _shape(mask) != (num_edges,):
        raise ContractError(
            f"Graph field '{name}' shape {_shape(mask)} != ({num_edges},)."
        )
    if getattr(mask, "dtype", None) != torch.bool:
        raise ContractError(f"Graph field '{name}' must have dtype torch.bool.")


class Phase1IoT23Task:
    """Adapt prepared Phase 1 scenario graphs to a reusable FL task.

    Parameters are intentionally explicit: this class does not discover or fit
    preprocessing state. A caller must provide versioned feature/label schemas
    and already split client graphs, which prevents a federation run from
    silently changing when Phase 1 is refactored.
    """

    def __init__(
        self,
        *,
        client_graphs: Mapping[str, Any | Sequence[Any]],
        feature_columns: Sequence[str],
        class_to_idx: Mapping[str, int],
        model_factory: ModelFactory,
        model_family: str = "egraphsage",
        model_version: int = 1,
        model_hyperparameters: Mapping[str, Any] | None = None,
        imbalance_mode: str = "class_weight",
        fixed_class_weights: Sequence[float] | np.ndarray | None = None,
        device: str | torch.device | None = None,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not client_graphs:
            raise ContractError("Phase1IoT23Task requires at least one client.")
        if imbalance_mode not in {"none", "class_weight"}:
            raise ContractError(
                "Phase 1 FL adapter supports imbalance_mode='none' or "
                "'class_weight'; undersampling must happen before graph build."
            )
        self._client_ids = tuple(str(client_id) for client_id in client_graphs)
        if len(self._client_ids) != len(set(self._client_ids)):
            raise ContractError("Client ids must be unique after string conversion.")
        self._graphs: dict[str, tuple[Any, ...]] = {}
        for raw_client_id, value in client_graphs.items():
            client_id = str(raw_client_id)
            graphs = (
                tuple(value)
                if isinstance(value, (list, tuple))
                else (value,)
            )
            if not graphs:
                raise ContractError(f"Client '{client_id}' has no graph.")
            self._graphs[client_id] = graphs

        self._feature_schema = FeatureSchema.from_names(feature_columns)
        self._label_schema = LabelSchema(_ordered_labels(class_to_idx))
        first_graph = self._graphs[self._client_ids[0]][0]
        node_feature_dim = int(first_graph.x.shape[1])
        self._graph_schema = GraphSchema(
            feature_schema_digest=self._feature_schema.digest,
            label_schema_digest=self._label_schema.digest,
            node_feature_dim=node_feature_dim,
        )
        self._model_factory = model_factory
        self._imbalance_mode = imbalance_mode
        self._fixed_class_weights: torch.Tensor | None = None
        if fixed_class_weights is not None:
            if imbalance_mode != "class_weight":
                raise ContractError(
                    "fixed_class_weights requires imbalance_mode='class_weight'."
                )
            fixed = np.asarray(fixed_class_weights, dtype=np.float32)
            if fixed.shape != (self._label_schema.num_classes,):
                raise ContractError(
                    "fixed_class_weights must contain one value per class."
                )
            if not np.all(np.isfinite(fixed)) or np.any(fixed < 0):
                raise ContractError(
                    "fixed_class_weights must be finite and non-negative."
                )
            self._fixed_class_weights = torch.from_numpy(fixed.copy())
        self._device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._source_metadata = dict(source_metadata or {})

        for client_id, graphs in self._graphs.items():
            for graph_index, graph in enumerate(graphs):
                self._validate_graph(
                    graph,
                    context=f"client={client_id}, graph={graph_index}",
                )

        initial_model = self._new_model(first_graph)
        state = initial_model.state_dict()
        self._model_spec = ModelSpec.from_state(
            family=model_family,
            model_version=model_version,
            feature_dim=self._feature_schema.feature_dim,
            num_classes=self._label_schema.num_classes,
            node_feature_dim=node_feature_dim,
            hyperparameters=dict(model_hyperparameters or {}),
            state=state,
        )
        self._initial_state = torch_state_to_arrays(state)

    @property
    def task_id(self) -> str:
        return f"iot23-{self._model_spec.family}-v{self._model_spec.model_version}"

    @property
    def client_ids(self) -> tuple[str, ...]:
        return self._client_ids

    @property
    def feature_schema(self) -> FeatureSchema:
        return self._feature_schema

    @property
    def label_schema(self) -> LabelSchema:
        return self._label_schema

    @property
    def graph_schema(self) -> GraphSchema:
        return self._graph_schema

    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec

    def initial_state(self) -> ArrayState:
        return {
            name: value.copy() for name, value in self._initial_state.items()
        }

    def _new_model(self, graph: Any) -> nn.Module:
        model = self._model_factory(graph)
        if not isinstance(model, nn.Module):
            raise Phase1AdapterError(
                "model_factory must return a torch.nn.Module."
            )
        return model

    def _validate_graph(self, graph: Any, *, context: str) -> None:
        for field in self._graph_schema.required_fields:
            if not hasattr(graph, field):
                raise ContractError(f"{context}: graph is missing '{field}'.")

        num_edges = int(graph.edge_index.shape[1])
        feature_dim = self._feature_schema.feature_dim
        num_classes = self._label_schema.num_classes
        node_dim = self._graph_schema.node_feature_dim
        if _shape(graph.edge_index) != (2, num_edges):
            raise ContractError(f"{context}: edge_index must have shape [2, E].")
        if _shape(graph.edge_attr) != (num_edges, feature_dim):
            raise ContractError(
                f"{context}: edge_attr shape {_shape(graph.edge_attr)} != "
                f"({num_edges}, {feature_dim})."
            )
        if _shape(graph.edge_label) != (num_edges,):
            raise ContractError(f"{context}: edge_label must have shape [E].")
        if _shape(graph.x)[1:] != (node_dim,):
            raise ContractError(
                f"{context}: node feature dim {_shape(graph.x)} does not match "
                f"{node_dim}."
            )
        if _shape(graph.edge_index_mp) != (2, 2 * num_edges):
            raise ContractError(
                f"{context}: edge_index_mp must have shape [2, 2E]."
            )
        if _shape(graph.edge_attr_mp) != (2 * num_edges, feature_dim):
            raise ContractError(
                f"{context}: edge_attr_mp must have shape [2E, F]."
            )
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            _validate_bool_mask(
                getattr(graph, mask_name), name=mask_name, num_edges=num_edges
            )
        train_mask = graph.train_mask.detach().cpu()
        val_mask = graph.val_mask.detach().cpu()
        test_mask = graph.test_mask.detach().cpu()
        if torch.any(train_mask & val_mask) or torch.any(train_mask & test_mask):
            raise ContractError(f"{context}: graph masks overlap.")
        if torch.any(val_mask & test_mask):
            raise ContractError(f"{context}: graph masks overlap.")
        if not torch.all(train_mask | val_mask | test_mask):
            raise ContractError(f"{context}: graph masks do not cover all edges.")
        labels = graph.edge_label.detach().cpu()
        if labels.numel() and (
            int(labels.min()) < 0 or int(labels.max()) >= num_classes
        ):
            raise ContractError(f"{context}: edge_label is outside [0, K).")
        if hasattr(graph, "feature_dim") and int(graph.feature_dim) != feature_dim:
            raise ContractError(f"{context}: graph.feature_dim does not match.")
        if hasattr(graph, "num_classes") and int(graph.num_classes) != num_classes:
            raise ContractError(f"{context}: graph.num_classes does not match.")
        if hasattr(graph, "class_to_idx"):
            expected = self._label_schema.class_to_idx
            actual = {
                str(name): int(index)
                for name, index in dict(graph.class_to_idx).items()
            }
            if actual != expected:
                raise ContractError(f"{context}: graph.class_to_idx does not match.")

    def _model_from_state(
        self,
        graph: Any,
        state: Mapping[str, np.ndarray],
    ) -> nn.Module:
        self._model_spec.validate_state(state)
        model = self._new_model(graph)
        template = model.state_dict()
        model.load_state_dict(arrays_to_torch_state(state, template=template))
        return model.to(self._device)

    def _client_graphs(self, client_id: str) -> tuple[Any, ...]:
        try:
            return self._graphs[client_id]
        except KeyError as exc:
            raise KeyError(f"Unknown IoT-23 client '{client_id}'.") from exc

    def _local_class_weights(self, graphs: Sequence[Any]) -> torch.Tensor | None:
        if self._imbalance_mode == "none":
            return None
        if self._fixed_class_weights is not None:
            return self._fixed_class_weights.to(self._device)
        labels = torch.cat(
            [
                graph.edge_label.detach().cpu()[graph.train_mask.detach().cpu()]
                for graph in graphs
            ]
        )
        if labels.numel() == 0:
            raise Phase1AdapterError("Client has no training edges.")
        counts = torch.bincount(
            labels, minlength=self._label_schema.num_classes
        ).float()
        present = counts > 0
        weights = torch.zeros_like(counts)
        weights[present] = labels.numel() / (present.sum() * counts[present])
        return weights.to(self._device)

    def train_local(
        self,
        client_id: str,
        global_state: Mapping[str, np.ndarray],
        config: LocalTrainConfig,
    ) -> LocalTrainResult:
        graphs = self._client_graphs(client_id)
        seed = _stable_seed(config.seed, client_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = self._model_from_state(graphs[0], global_state)
        global_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        if config.optimizer == "sgd":
            optimizer: torch.optim.Optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        device_graphs = tuple(graph.to(self._device) for graph in graphs)
        weights = self._local_class_weights(graphs)
        num_examples = sum(int(graph.train_mask.sum()) for graph in graphs)
        if num_examples <= 0:
            raise Phase1AdapterError(f"Client '{client_id}' has no train edges.")
        if weights is None:
            normalizer = float(num_examples)
        else:
            normalizer = float(
                sum(
                    weights[
                        graph.edge_label[graph.train_mask].to(self._device)
                    ].sum()
                    for graph in device_graphs
                )
            )
        if normalizer <= 0:
            raise Phase1AdapterError("Local weighted-loss normalizer is zero.")

        final_loss = 0.0
        for _ in range(config.local_epochs):
            model.train()
            optimizer.zero_grad()
            detached_total = 0.0
            for graph in device_graphs:
                mask = graph.train_mask
                labels = graph.edge_label[mask]
                logits = model(graph)[mask]
                loss_sum = F.cross_entropy(
                    logits,
                    labels,
                    weight=weights,
                    reduction="sum",
                )
                (loss_sum / normalizer).backward()
                detached_total += float(loss_sum.detach())

            proximal_value = 0.0
            if config.proximal_mu > 0:
                proximal = torch.zeros((), device=self._device)
                for name, parameter in model.named_parameters():
                    proximal = proximal + torch.sum(
                        (parameter - global_parameters[name]) ** 2
                    )
                ((config.proximal_mu / 2.0) * proximal).backward()
                proximal_value = float(proximal.detach())
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
            optimizer.step()
            final_loss = detached_total / normalizer + (
                config.proximal_mu / 2.0
            ) * proximal_value

        return LocalTrainResult(
            state=torch_state_to_arrays(model.state_dict()),
            num_examples=num_examples,
            metrics={"train_loss": float(final_loss)},
        )

    def evaluate_local(
        self,
        client_id: str,
        state: Mapping[str, np.ndarray],
        *,
        split: str,
    ) -> EvaluationResult:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test.")
        graphs = self._client_graphs(client_id)
        model = self._model_from_state(graphs[0], state)
        model.eval()
        matrices: list[np.ndarray] = []
        total_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for graph in graphs:
                device_graph = graph.to(self._device)
                mask = getattr(device_graph, f"{split}_mask")
                labels = device_graph.edge_label[mask]
                if labels.numel() == 0:
                    continue
                logits = model(device_graph)[mask]
                total_loss += float(
                    F.cross_entropy(logits, labels, reduction="sum")
                )
                predictions = logits.argmax(dim=-1)
                matrices.append(
                    confusion_matrix_from_predictions(
                        labels.detach().cpu().numpy(),
                        predictions.detach().cpu().numpy(),
                        num_classes=self._label_schema.num_classes,
                    )
                )
                total_examples += int(labels.numel())
        if total_examples == 0:
            raise Phase1AdapterError(
                f"Client '{client_id}' has no examples in split '{split}'."
            )
        matrix = aggregate_confusion_matrices(
            matrices, num_classes=self._label_schema.num_classes
        )
        return EvaluationResult(
            confusion_matrix=matrix,
            num_examples=total_examples,
            loss=total_loss / total_examples,
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "imbalance_mode": self._imbalance_mode,
            "class_weight_scope": (
                "global" if self._fixed_class_weights is not None else "local"
            ),
            "device": str(self._device),
            "client_graph_counts": {
                client_id: len(graphs)
                for client_id, graphs in self._graphs.items()
            },
            **self._source_metadata,
        }

    def contract_bundle(
        self,
        *,
        preprocessor: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContractBundle:
        """Export portable learned preprocessing values when available."""
        categories: dict[str, tuple[str, ...]] = {}
        arrays: dict[str, np.ndarray] = {}
        if preprocessor is not None:
            attribute_map = {
                "resp_port": "resp_port_categories",
                "proto": "proto_categories",
                "service": "service_categories",
                "conn_state": "conn_state_categories",
                "history_flags": "history_flag_chars",
                "numeric_columns": "numeric_columns",
                "missing_flags": "missing_flag_columns",
            }
            for output_name, attribute in attribute_map.items():
                if hasattr(preprocessor, attribute):
                    categories[output_name] = tuple(
                        str(value) for value in getattr(preprocessor, attribute)
                    )
            scaler = getattr(preprocessor, "scaler", None)
            if scaler is not None:
                for name in ("mean_", "scale_", "var_"):
                    if hasattr(scaler, name):
                        arrays[f"scaler_{name.rstrip('_')}"] = np.asarray(
                            getattr(scaler, name)
                        )
        return ContractBundle(
            feature_schema=self._feature_schema,
            label_schema=self._label_schema,
            graph_schema=self._graph_schema,
            model_spec=self._model_spec,
            categories=categories,
            learned_arrays=arrays,
            metadata={**self.metadata(), **dict(metadata or {})},
        )
