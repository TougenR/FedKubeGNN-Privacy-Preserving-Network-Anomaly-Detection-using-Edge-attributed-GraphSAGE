"""Deterministic seven-class datasets for IID versus natural non-IID diagnosis."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.federated.contracts.artifacts import ContractBundle
from src.federated.contracts.schema import ContractError, LabelSchema, ModelSpec
from src.federated.data.manifest import MANIFEST_VERSION, PreparedDatasetManifest
from src.federated.data.storage import (
    GraphArrays,
    checksum_index_digest,
    load_graph_arrays,
    sha256_file,
    write_graph_arrays,
)


SOURCE_CLASSES = (
    "Benign",
    "Attack",
    "C&C",
    "C&C-HeartBeat",
    "DDoS",
    "Okiru",
    "Okiru-Attack",
    "PartOfAHorizontalPortScan",
)
SEVEN_CLASSES = SOURCE_CLASSES[:6] + SOURCE_CLASSES[7:]
DROPPED_CLASS_INDEX = 6
OUTPUT_WEIGHT = "head.3.weight"
OUTPUT_BIAS = "head.3.bias"
NODE_NAMESPACE = "scenario_id::source_node_id"

SourceEdge = tuple[int, int]


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def remap_seven_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return retained mask and labels after dropping old class six."""
    values = np.asarray(labels)
    keep = values != DROPPED_CLASS_INDEX
    remapped = values[keep].astype(np.int64, copy=True)
    remapped[remapped > DROPPED_CLASS_INDEX] -= 1
    return keep, remapped


def project_seven_class_state(
    state: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Project the fixed eight-class E-GraphSAGE state to seven classes."""
    if OUTPUT_WEIGHT not in state or OUTPUT_BIAS not in state:
        raise ContractError("Expected E-GraphSAGE output tensors are missing.")
    weight = np.asarray(state[OUTPUT_WEIGHT])
    bias = np.asarray(state[OUTPUT_BIAS])
    if weight.ndim != 2 or weight.shape[0] != len(SOURCE_CLASSES):
        raise ContractError("Expected an eight-row E-GraphSAGE output weight.")
    if bias.shape != (len(SOURCE_CLASSES),):
        raise ContractError("Expected an eight-element E-GraphSAGE output bias.")
    projected = {
        str(name): np.asarray(value).copy() for name, value in state.items()
    }
    projected[OUTPUT_WEIGHT] = np.delete(
        weight, DROPPED_CLASS_INDEX, axis=0
    ).copy()
    projected[OUTPUT_BIAS] = np.delete(bias, DROPPED_CLASS_INDEX).copy()
    return projected


def stratified_iid_train_assignment(
    graphs: Iterable[GraphArrays],
    *,
    num_clients: int,
    seed: int,
) -> list[list[SourceEdge]]:
    """Split each retained train class into deterministic near-equal shards."""
    graph_list = tuple(graphs)
    if num_clients < 1:
        raise ValueError("num_clients must be positive.")
    assignments: list[list[SourceEdge]] = [[] for _ in range(num_clients)]
    rng = np.random.default_rng(seed)
    for class_index in range(len(SEVEN_CLASSES)):
        source_label = class_index if class_index < DROPPED_CLASS_INDEX else 7
        candidates = np.asarray(
            [
                (source_index, int(edge_index))
                for source_index, graph in enumerate(graph_list)
                for edge_index in np.flatnonzero(
                    graph.train_mask & (graph.edge_label == source_label)
                )
            ],
            dtype=np.int64,
        )
        if len(candidates) < num_clients:
            raise ContractError(
                f"Class '{SEVEN_CLASSES[class_index]}' has only "
                f"{len(candidates)} train edges for {num_clients} clients."
            )
        rng.shuffle(candidates, axis=0)
        for destination, shard in enumerate(np.array_split(candidates, num_clients)):
            assignments[destination].extend(
                (int(source), int(edge)) for source, edge in shard
            )
    for assignment in assignments:
        assignment.sort()
    return assignments


def _load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _split_name(graph: GraphArrays, edge_index: int) -> str:
    if bool(graph.train_mask[edge_index]):
        return "train"
    if bool(graph.val_mask[edge_index]):
        return "validation"
    return "test"


def _natural_graph(graph: GraphArrays) -> GraphArrays:
    keep, labels = remap_seven_labels(graph.edge_label)
    return GraphArrays(
        edge_index=graph.edge_index[:, keep].copy(),
        edge_attr=graph.edge_attr[keep].copy(),
        edge_label=labels,
        train_mask=graph.train_mask[keep].copy(),
        val_mask=graph.val_mask[keep].copy(),
        test_mask=graph.test_mask[keep].copy(),
        num_nodes=graph.num_nodes,
    )


def _iid_graph(
    graphs: tuple[GraphArrays, ...],
    source_ids: tuple[str, ...],
    records: list[SourceEdge],
) -> GraphArrays:
    records = sorted(records)
    edge_count = len(records)
    feature_dim = graphs[0].feature_dim
    edge_index = np.empty((2, edge_count), dtype=np.int64)
    edge_attr = np.empty((edge_count, feature_dim), dtype=np.float32)
    edge_label = np.empty(edge_count, dtype=np.int64)
    train_mask = np.zeros(edge_count, dtype=np.bool_)
    val_mask = np.zeros(edge_count, dtype=np.bool_)
    test_mask = np.zeros(edge_count, dtype=np.bool_)
    node_ids: dict[tuple[str, int], int] = {}

    for output_index, (source_index, source_edge) in enumerate(records):
        graph = graphs[source_index]
        source_id = source_ids[source_index]
        for endpoint in (0, 1):
            key = (source_id, int(graph.edge_index[endpoint, source_edge]))
            if key not in node_ids:
                node_ids[key] = len(node_ids)
            edge_index[endpoint, output_index] = node_ids[key]
        edge_attr[output_index] = graph.edge_attr[source_edge]
        old_label = int(graph.edge_label[source_edge])
        if old_label == DROPPED_CLASS_INDEX:
            raise ContractError("Dropped class entered an IID destination graph.")
        edge_label[output_index] = old_label - int(old_label > DROPPED_CLASS_INDEX)
        split = _split_name(graph, source_edge)
        if split == "train":
            train_mask[output_index] = True
        elif split == "validation":
            val_mask[output_index] = True
        else:
            test_mask[output_index] = True

    return GraphArrays(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=len(node_ids),
    )


def _counts(graph: GraphArrays, split: str) -> list[int]:
    mask = {
        "train": graph.train_mask,
        "validation": graph.val_mask,
        "test": graph.test_mask,
    }[split]
    return np.bincount(
        graph.edge_label[mask], minlength=len(SEVEN_CLASSES)
    ).astype(int).tolist()


def _provenance_digest(
    destinations: Mapping[str, Iterable[SourceEdge]],
    source_ids: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    rows = sorted(
        (source_ids[source_index], edge_index, destination)
        for destination, records in destinations.items()
        for source_index, edge_index in records
    )
    for source_id, edge_index, destination in rows:
        digest.update(f"{source_id}\0{edge_index}\0{destination}\n".encode())
    return digest.hexdigest()


def _validate_provenance(
    graphs: tuple[GraphArrays, ...],
    destinations: Mapping[str, list[SourceEdge]],
) -> None:
    expected = {
        (source_index, int(edge_index))
        for source_index, graph in enumerate(graphs)
        for edge_index in np.flatnonzero(
            graph.edge_label != DROPPED_CLASS_INDEX
        )
    }
    actual = [record for records in destinations.values() for record in records]
    if len(actual) != len(set(actual)):
        raise ContractError("Derived partition duplicates source flows.")
    if set(actual) != expected:
        raise ContractError("Derived partition omits or adds source flows.")


def _destination_records(
    kind: str,
    graphs: tuple[GraphArrays, ...],
    source_ids: tuple[str, ...],
    *,
    seed: int,
) -> dict[str, list[SourceEdge]]:
    if kind == "natural":
        return {
            client_id: [
                (source_index, int(edge_index))
                for edge_index in np.flatnonzero(
                    graphs[source_index].edge_label != DROPPED_CLASS_INDEX
                )
            ]
            for source_index, client_id in enumerate(source_ids)
        }
    if kind != "iid":
        raise ValueError(f"Unknown derivation kind '{kind}'.")
    assignments = stratified_iid_train_assignment(
        graphs, num_clients=len(source_ids), seed=seed
    )
    result = {
        client_id: list(assignments[index])
        for index, client_id in enumerate(source_ids)
    }
    # Validation and test retain their original scenario ownership and are not
    # inputs to the balancing algorithm.
    for source_index, client_id in enumerate(source_ids):
        graph = graphs[source_index]
        held_out = (
            (graph.val_mask | graph.test_mask)
            & (graph.edge_label != DROPPED_CLASS_INDEX)
        )
        result[client_id].extend(
            (source_index, int(edge_index))
            for edge_index in np.flatnonzero(held_out)
        )
        result[client_id].sort()
    return result


def _write_derived_dataset(
    source: PreparedDatasetManifest,
    output_root: Path,
    *,
    kind: str,
    seed: int,
) -> Path:
    source_bundle = ContractBundle.load(
        source.root / str(source.document["contract_path"])
    )
    if source_bundle.label_schema.classes != SOURCE_CLASSES:
        raise ContractError(
            "Seven-class diagnostic requires the fixed IoT-23 eight-class schema."
        )
    source_ids = source.client_ids
    graphs = tuple(
        load_graph_arrays(source.client_path(client_id), verify=True)
        for client_id in source_ids
    )
    destinations = _destination_records(
        kind, graphs, source_ids, seed=seed
    )
    _validate_provenance(graphs, destinations)

    identity = {
        "source_dataset_digest": source.digest,
        "kind": kind,
        "partition": "stratified_iid_train_v1" if kind == "iid" else "natural_v1",
        "seed": seed,
        "classes": SEVEN_CLASSES,
        "node_namespace": NODE_NAMESPACE,
    }
    dataset_id = f"iot23-seven-{kind}-{_canonical_digest(identity)[:16]}"
    final = output_root / dataset_id
    if final.exists():
        raise FileExistsError(f"Derived dataset already exists: {final}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.", dir=output_root))
    try:
        labels = LabelSchema(SEVEN_CLASSES, version=source_bundle.label_schema.version)
        graph_schema = replace(
            source_bundle.graph_schema,
            label_schema_digest=labels.digest,
            node_semantics=NODE_NAMESPACE,
        )
        source_state = _load_state(
            source.root / str(source.document["initial_state_path"])
        )
        state = project_seven_class_state(source_state)
        if source_bundle.model_spec is None:
            raise ContractError("Source contract has no model specification.")
        source_model = source_bundle.model_spec
        model = ModelSpec.from_state(
            family=source_model.family,
            model_version=source_model.model_version,
            feature_dim=source_model.feature_dim,
            num_classes=len(SEVEN_CLASSES),
            node_feature_dim=source_model.node_feature_dim,
            hyperparameters=source_model.hyperparameters,
            state=state,
        )
        shared_contract_metadata = {
            "diagnostic": "seven_class_iid_vs_natural",
            "source_dataset_digest": source.digest,
            "preprocessing": "reuse_source_train_only_global",
            "removed_class": SOURCE_CLASSES[DROPPED_CLASS_INDEX],
            "node_namespace": NODE_NAMESPACE,
        }
        contract_root = ContractBundle(
            feature_schema=source_bundle.feature_schema,
            label_schema=labels,
            graph_schema=graph_schema,
            model_spec=model,
            categories=source_bundle.categories,
            learned_arrays=source_bundle.learned_arrays,
            metadata=shared_contract_metadata,
        ).write(temporary / "contract")
        initial_path = temporary / "initial_state.npz"
        np.savez(initial_path, **state)

        client_entries: list[dict[str, Any]] = []
        client_counts: dict[str, Any] = {}
        for source_index, client_id in enumerate(source_ids):
            graph = (
                _natural_graph(graphs[source_index])
                if kind == "natural"
                else _iid_graph(
                    graphs, source_ids, destinations[client_id]
                )
            )
            relative = Path("clients") / client_id
            client_root = write_graph_arrays(
                temporary / relative,
                graph,
                metadata={
                    "client_id": client_id,
                    "graph_protocol": source.document["graph_protocol"],
                    "derivation_kind": kind,
                    "node_namespace": NODE_NAMESPACE,
                },
            )
            client_entries.append(
                {
                    "client_id": client_id,
                    "path": str(relative),
                    "num_edges": graph.num_edges,
                    "artifact_digest": checksum_index_digest(client_root),
                }
            )
            client_counts[client_id] = {
                split: _counts(graph, split)
                for split in ("train", "validation", "test")
            }

        source_retained_counts = {
            split: np.sum(
                np.asarray([_counts(_natural_graph(graph), split) for graph in graphs]),
                axis=0,
            ).astype(int).tolist()
            for split in ("train", "validation", "test")
        }
        derived_counts = {
            split: np.sum(
                np.asarray(
                    [client_counts[client_id][split] for client_id in source_ids]
                ),
                axis=0,
            ).astype(int).tolist()
            for split in ("train", "validation", "test")
        }
        if derived_counts != source_retained_counts:
            raise ContractError("Derived global split/class support changed.")
        if kind == "iid":
            train_support = np.asarray(
                [client_counts[client_id]["train"] for client_id in source_ids]
            )
            if np.any(train_support.max(axis=0) - train_support.min(axis=0) > 1):
                raise ContractError("IID train class shards differ by more than one.")

        report = {
            "dataset_id": dataset_id,
            "derivation": identity,
            "classes": list(SEVEN_CLASSES),
            "class_mapping": {
                str(index): (
                    None
                    if index == DROPPED_CLASS_INDEX
                    else index - int(index > DROPPED_CLASS_INDEX)
                )
                for index in range(len(SOURCE_CLASSES))
            },
            "client_split_class_counts": client_counts,
            "global_split_class_counts": derived_counts,
            "provenance_digest": _provenance_digest(destinations, source_ids),
            "retained_source_flows": sum(
                len(records) for records in destinations.values()
            ),
            "checks": {
                "no_duplicate_or_missing_retained_flow": True,
                "global_split_class_support_preserved": True,
                "iid_train_max_min_per_class_lte_one": kind == "iid",
                "validation_test_not_repartitioned": kind == "iid",
            },
        }
        report_path = temporary / "derivation_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "dataset_id": dataset_id,
            "config_digest": source.document.get("config_digest"),
            "graph_protocol": source.document["graph_protocol"],
            "preprocessing": "reuse_source_train_only_global",
            "contract_path": "contract",
            "contract_digest": checksum_index_digest(contract_root),
            "initial_state_path": "initial_state.npz",
            "initial_state_sha256": sha256_file(initial_path),
            "derivation_report_path": "derivation_report.json",
            "derivation_report_sha256": sha256_file(report_path),
            "source_dataset_id": source.dataset_id,
            "source_dataset_digest": source.digest,
            "derivation": identity,
            "clients": client_entries,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        PreparedDatasetManifest.load(temporary, verify=True)
        os.replace(temporary, final)
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def derive_seven_class_datasets(
    source_root: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Path]:
    """Create verified natural and stratified-IID seven-class datasets."""
    source = PreparedDatasetManifest.load(source_root, verify=True)
    destination = Path(output_root)
    return {
        kind: _write_derived_dataset(
            source, destination, kind=kind, seed=seed
        )
        for kind in ("natural", "iid")
    }
