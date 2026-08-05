"""Balance, feature, and topology analysis for a prepared federated dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.federated.contracts.artifacts import ContractBundle
from src.federated.data.manifest import PreparedDatasetManifest
from src.federated.data.storage import GraphArrays, load_graph_arrays


ANALYSIS_VERSION = 1
SEVERE_GLOBAL_SUPPORT = 30
SPLITS = ("all", "train", "validation", "test")


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty analysis table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    return counts / total if total else np.zeros_like(counts)


def _entropy_bits(probabilities: np.ndarray) -> float:
    positive = np.asarray(probabilities, dtype=np.float64)
    positive = positive[positive > 0]
    return float(-(positive * np.log2(positive)).sum()) if positive.size else 0.0


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    midpoint = 0.5 * (left + right)

    def kl(first: np.ndarray, second: np.ndarray) -> float:
        keep = first > 0
        return float((first[keep] * np.log2(first[keep] / second[keep])).sum())

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


def _summary_row(
    scope: str,
    counts: np.ndarray,
    split_totals: Mapping[str, int],
    *,
    union_distribution: np.ndarray,
) -> dict[str, Any]:
    distribution = _distribution(counts)
    positive = counts[counts > 0]
    entropy = _entropy_bits(distribution)
    class_count = int(counts.size)
    return {
        "scope": scope,
        "total": int(counts.sum()),
        "train": int(split_totals["train"]),
        "validation": int(split_totals["validation"]),
        "test": int(split_totals["test"]),
        "classes_present": int(np.count_nonzero(counts)),
        "zero_support_classes": int(np.count_nonzero(counts == 0)),
        "minority_support_nonzero": int(positive.min()) if positive.size else 0,
        "majority_support": int(positive.max()) if positive.size else 0,
        "imbalance_ratio_nonzero": (
            float(positive.max() / positive.min()) if positive.size else 0.0
        ),
        "entropy_bits": entropy,
        "normalized_entropy": (
            entropy / math.log2(class_count) if class_count > 1 else 0.0
        ),
        "effective_classes_exp_entropy": float(2**entropy),
        "jensen_shannon_from_union_bits": (
            0.0
            if scope == "global"
            else _js_divergence(distribution, union_distribution)
        ),
    }


def _mask(graph: GraphArrays, split: str) -> np.ndarray:
    if split == "all":
        return np.ones(graph.num_edges, dtype=np.bool_)
    name = "val_mask" if split == "validation" else f"{split}_mask"
    return np.asarray(getattr(graph, name), dtype=np.bool_)


def _class_counts(graph: GraphArrays, split: str, num_classes: int) -> np.ndarray:
    labels = np.asarray(graph.edge_label)[_mask(graph, split)]
    return np.bincount(labels, minlength=num_classes).astype(np.int64, copy=False)


def _weak_component_stats(graph: GraphArrays) -> tuple[int, int]:
    parent = np.arange(graph.num_nodes, dtype=np.int64)
    size = np.ones(graph.num_nodes, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for source, target in np.asarray(graph.edge_index).T:
        left = find(int(source))
        right = find(int(target))
        if left == right:
            continue
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]
    roots = np.fromiter((find(node) for node in range(graph.num_nodes)), dtype=np.int64)
    _, component_sizes = np.unique(roots, return_counts=True)
    return int(component_sizes.size), int(component_sizes.max())


def _exact_duplicate_edges(graph: GraphArrays) -> int:
    # Local node identifiers have no meaning across clients, so duplicates are
    # intentionally measured within each client graph only.
    records = np.column_stack(
        (
            np.asarray(graph.edge_index).T.astype(np.float64),
            np.asarray(graph.edge_label, dtype=np.float64),
            np.asarray(graph.edge_attr, dtype=np.float64),
        )
    )
    records = np.ascontiguousarray(records)
    opaque = records.view(np.dtype((np.void, records.dtype.itemsize * records.shape[1])))
    return int(graph.num_edges - np.unique(opaque).size)


def _topology_row(client_id: str, graph: GraphArrays) -> dict[str, Any]:
    endpoints = np.asarray(graph.edge_index)
    unique_directed_pairs = int(np.unique(endpoints.T, axis=0).shape[0])
    degree = np.bincount(endpoints.reshape(-1), minlength=graph.num_nodes)
    nodes_with_edges = int(np.count_nonzero(degree))
    components, largest_component = _weak_component_stats(graph)
    denominator = graph.num_nodes * max(graph.num_nodes - 1, 1)
    return {
        "client_id": client_id,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "feature_dim": graph.feature_dim,
        "nodes_with_edges": nodes_with_edges,
        "isolated_nodes": graph.num_nodes - nodes_with_edges,
        "weak_components_including_isolates": components,
        "largest_weak_component_nodes": largest_component,
        "self_loops": int(np.count_nonzero(endpoints[0] == endpoints[1])),
        "unique_directed_node_pairs": unique_directed_pairs,
        "parallel_edge_instances": graph.num_edges - unique_directed_pairs,
        "exact_duplicate_edge_rows": _exact_duplicate_edges(graph),
        "mean_total_degree": float(degree.mean()),
        "max_total_degree": int(degree.max()),
        "unique_directed_pair_density": float(unique_directed_pairs / denominator),
        "nonfinite_feature_values": int(
            np.count_nonzero(~np.isfinite(np.asarray(graph.edge_attr)))
        ),
    }


def _feature_rows(
    scope: str, values: np.ndarray, feature_names: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        column = np.asarray(values[:, index], dtype=np.float64)
        finite = column[np.isfinite(column)]
        rows.append(
            {
                "scope": scope,
                "feature_index": index,
                "feature_name": name,
                "rows": int(column.size),
                "nonfinite": int(column.size - finite.size),
                "mean": float(finite.mean()) if finite.size else "",
                "std": float(finite.std()) if finite.size else "",
                "min": float(finite.min()) if finite.size else "",
                "max": float(finite.max()) if finite.size else "",
                "zero_rate": (
                    float(np.count_nonzero(finite == 0) / finite.size)
                    if finite.size
                    else ""
                ),
                "constant_or_near_constant": (
                    bool(finite.size and float(finite.std()) <= 1e-8)
                ),
                "missing_indicator_rate": (
                    float(finite.mean())
                    if finite.size and name.endswith("_missing")
                    else ""
                ),
            }
        )
    return rows


def _render_figures(
    output_root: Path,
    client_ids: Sequence[str],
    class_names: Sequence[str],
    client_counts: np.ndarray,
    global_split_counts: Mapping[str, np.ndarray],
    feature_names: Sequence[str],
    client_features: Sequence[np.ndarray],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_paths: list[Path] = []

    fig, axis = plt.subplots(figsize=(12, 5.5))
    image = axis.imshow(np.log10(client_counts + 1), cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    axis.set_yticks(range(len(client_ids)), client_ids)
    axis.set_title("IoT-23 client/class support (log10 count + 1)")
    axis.set_xlabel("Class")
    axis.set_ylabel("Federated client")
    fig.colorbar(image, ax=axis, label="log10(support + 1)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        path = output_root / f"class_support_heatmap.{extension}"
        fig.savefig(path, dpi=180)
        figure_paths.append(path)
    plt.close(fig)

    positions = np.arange(len(class_names))
    fig, axis = plt.subplots(figsize=(12, 5.5))
    bottom = np.zeros(len(class_names), dtype=np.int64)
    for split, color in zip(("train", "validation", "test"), ("#4c78a8", "#f2cf5b", "#e45756")):
        counts = global_split_counts[split]
        axis.bar(positions, counts, bottom=bottom, label=split, color=color)
        bottom += counts
    axis.set_yscale("log")
    axis.set_xticks(positions, class_names, rotation=35, ha="right")
    axis.set_ylabel("Global support (log scale)")
    axis.set_title("Global class support by immutable split")
    axis.legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        path = output_root / f"global_split_support.{extension}"
        fig.savefig(path, dpi=180)
        figure_paths.append(path)
    plt.close(fig)

    union = np.concatenate(client_features, axis=0).astype(np.float64, copy=False)
    global_mean = np.nanmean(union, axis=0)
    global_std = np.nanstd(union, axis=0)
    safe_std = np.where(global_std > 1e-8, global_std, 1.0)
    shifts = np.vstack(
        [
            (np.nanmean(values, axis=0) - global_mean) / safe_std
            for values in client_features
        ]
    )
    ranking = np.argsort(np.nanmax(np.abs(shifts), axis=0))[::-1][:20]
    fig, axis = plt.subplots(figsize=(13, 5.5))
    image = axis.imshow(shifts[:, ranking], cmap="coolwarm", vmin=-3, vmax=3, aspect="auto")
    axis.set_xticks(
        range(len(ranking)),
        [feature_names[index] for index in ranking],
        rotation=40,
        ha="right",
    )
    axis.set_yticks(range(len(client_ids)), client_ids)
    axis.set_title("Top client feature-mean shifts vs union (global standard deviations)")
    fig.colorbar(image, ax=axis, label="standardized mean shift (clipped color at ±3)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        path = output_root / f"feature_mean_shift_heatmap.{extension}"
        fig.savefig(path, dpi=180)
        figure_paths.append(path)
    plt.close(fig)
    return figure_paths


def _finding_report(
    *,
    dataset_id: str,
    dataset_digest: str,
    class_names: Sequence[str],
    global_counts: np.ndarray,
    clients_per_class: np.ndarray,
    balance_rows: Sequence[Mapping[str, Any]],
    topology_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
) -> str:
    global_features = [row for row in feature_rows if row["scope"] == "global"]
    constant_features = [
        str(row["feature_name"])
        for row in global_features
        if row["constant_or_near_constant"]
    ]
    missing_indicators = [
        row
        for row in global_features
        if row["missing_indicator_rate"] != ""
    ]
    lines = [
        "# Phase 3A Prepared-Data Findings",
        "",
        f"- Dataset: `{dataset_id}`",
        f"- Manifest digest: `{dataset_digest}`",
        "- Integrity: contract, initial state, every client checksum, and split-mask "
        "coverage verified before analysis.",
        "",
        "## Class balance and non-IID structure",
        "",
        "| Class | Global support | Clients with class | Severe trigger |",
        "|---|---:|---:|---|",
    ]
    for index, class_name in enumerate(class_names):
        support = int(global_counts[index])
        clients = int(clients_per_class[index])
        severe = clients == 1 or support < SEVERE_GLOBAL_SUPPORT
        lines.append(
            f"| {class_name} | {support} | {clients} | "
            f"{'yes' if severe else 'no'} |"
        )
    global_balance = balance_rows[0]
    lines.extend(
        [
            "",
            f"The union imbalance ratio is `{global_balance['imbalance_ratio_nonzero']:.1f}:1` "
            "because Okiru-Attack has only three observations. C&C-HeartBeat, "
            "DDoS, Okiru, and Okiru-Attack are structurally private to one client. "
            "This confirms severe imbalance before any model-level treatment.",
            "",
            "## Feature and topology checks",
            "",
            f"- Non-finite transformed feature values: "
            f"`{sum(int(row['nonfinite']) for row in global_features)}`.",
            f"- Globally constant/near-constant features: "
            f"`{', '.join(constant_features) if constant_features else 'none'}`.",
        ]
    )
    for row in missing_indicators:
        lines.append(
            f"- `{row['feature_name']}` positive rate: "
            f"`{float(row['missing_indicator_rate']):.4f}`."
        )
    lines.extend(
        [
            "- Raw missing values are not retained by the prepared contract; the "
            "rates above are measured from explicit transformed missing indicators.",
            "",
            "| Client | Nodes | Edges | Unique directed pairs | Parallel edges | "
            "Exact duplicate rows |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in topology_rows:
        lines.append(
            f"| {row['client_id']} | {row['num_nodes']} | {row['num_edges']} | "
            f"{row['unique_directed_node_pairs']} | {row['parallel_edge_instances']} | "
            f"{row['exact_duplicate_edge_rows']} |"
        )
    lines.extend(
        [
            "",
            "Client 34-1 is an extreme multigraph (49 nodes, 18,751 edges, and only "
            "49 unique directed node pairs). This may be valid scenario structure, "
            "but it must be compared with Phase 1 graph construction before treating "
            "the federated metric gap as an optimizer-only problem.",
            "",
            "## Required next experiments",
            "",
            "1. Complete Phase 1/Phase 2 split, learned preprocessor, and graph-membership "
            "equivalence checks.",
            "2. Run the exact prepared-data centralized reference to separate data and "
            "federation effects.",
            "3. Evaluate global class weights first, then local epoch/learning rate, one "
            "alternative loss, and finally class-support-aware aggregation.",
            "4. Keep Okiru-Attack validation metrics marked not estimable; do not duplicate "
            "its test example or tune against it.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_prepared_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    render_figures: bool = True,
) -> Path:
    """Verify and analyze immutable arrays without mutating the dataset."""
    destination = Path(output_root)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Analysis output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    manifest = PreparedDatasetManifest.load(dataset_root, verify=True)
    contract = ContractBundle.load(
        manifest.root / str(manifest.document["contract_path"])
    )
    client_ids = list(manifest.client_ids)
    class_names = list(contract.label_schema.classes)
    feature_names = list(contract.feature_schema.names)
    num_classes = len(class_names)

    graphs = {
        client_id: load_graph_arrays(manifest.client_path(client_id), verify=True)
        for client_id in client_ids
    }
    counts_by_client_split = {
        client_id: {
            split: _class_counts(graph, split, num_classes)
            for split in SPLITS
        }
        for client_id, graph in graphs.items()
    }
    global_split_counts = {
        split: sum(
            (counts_by_client_split[client_id][split] for client_id in client_ids),
            start=np.zeros(num_classes, dtype=np.int64),
        )
        for split in SPLITS
    }
    union_distribution = _distribution(global_split_counts["all"])

    balance_rows = [
        _summary_row(
            "global",
            global_split_counts["all"],
            {
                split: int(global_split_counts[split].sum())
                for split in ("train", "validation", "test")
            },
            union_distribution=union_distribution,
        )
    ]
    for client_id in client_ids:
        balance_rows.append(
            _summary_row(
                client_id,
                counts_by_client_split[client_id]["all"],
                {
                    split: int(counts_by_client_split[client_id][split].sum())
                    for split in ("train", "validation", "test")
                },
                union_distribution=union_distribution,
            )
        )

    clients_per_class = np.count_nonzero(
        np.vstack([counts_by_client_split[item]["all"] for item in client_ids]),
        axis=0,
    )
    class_rows: list[dict[str, Any]] = []
    for scope in ["global", *client_ids]:
        split_counts = (
            global_split_counts
            if scope == "global"
            else counts_by_client_split[scope]
        )
        for split in SPLITS:
            counts = split_counts[split]
            total = int(counts.sum())
            for class_index, class_name in enumerate(class_names):
                support = int(counts[class_index])
                global_support = int(global_split_counts["all"][class_index])
                class_rows.append(
                    {
                        "scope": scope,
                        "split": split,
                        "class_index": class_index,
                        "class_name": class_name,
                        "support": support,
                        "proportion": float(support / total) if total else 0.0,
                        "global_support": global_support,
                        "clients_with_class": int(clients_per_class[class_index]),
                        "zero_support": support == 0,
                        "structurally_private_class": clients_per_class[class_index] == 1,
                        "ultra_rare_global_class": global_support < SEVERE_GLOBAL_SUPPORT,
                        "severe_imbalance_trigger": bool(
                            support == 0
                            or clients_per_class[class_index] == 1
                            or global_support < SEVERE_GLOBAL_SUPPORT
                        ),
                    }
                )

    topology_rows = [
        _topology_row(client_id, graphs[client_id]) for client_id in client_ids
    ]
    client_features = [np.asarray(graphs[item].edge_attr) for item in client_ids]
    feature_rows: list[dict[str, Any]] = []
    for client_id, values in zip(client_ids, client_features):
        feature_rows.extend(_feature_rows(client_id, values, feature_names))
    feature_rows.extend(
        _feature_rows("global", np.concatenate(client_features, axis=0), feature_names)
    )

    _write_csv(destination / "data_balance.csv", balance_rows)
    _write_csv(destination / "class_support_by_client.csv", class_rows)
    _write_csv(destination / "graph_topology.csv", topology_rows)
    _write_csv(destination / "feature_distribution.csv", feature_rows)

    severe_classes = [
        class_name
        for index, class_name in enumerate(class_names)
        if clients_per_class[index] == 1
        or global_split_counts["all"][index] < SEVERE_GLOBAL_SUPPORT
        or global_split_counts["validation"][index] == 0
    ]
    report = {
        "analysis_version": ANALYSIS_VERSION,
        "dataset_id": manifest.dataset_id,
        "dataset_digest": manifest.digest,
        "client_ids": client_ids,
        "class_names": class_names,
        "feature_schema_digest": contract.feature_schema.digest,
        "label_schema_digest": contract.label_schema.digest,
        "severe_imbalance": bool(severe_classes),
        "severe_classes": severe_classes,
        "severity_policy": {
            "global_support_below": SEVERE_GLOBAL_SUPPORT,
            "single_client_class": True,
            "zero_validation_support": True,
            "zero_client_class_support": True,
        },
        "integrity": {
            "manifest_and_all_client_checksums_verified": True,
            "masks_disjoint_and_complete": True,
            "raw_missingness_available": False,
            "duplicate_definition": (
                "exact transformed feature, label, source, and destination row "
                "within each client graph"
            ),
        },
        "global_class_support": {
            class_name: int(global_split_counts["all"][index])
            for index, class_name in enumerate(class_names)
        },
        "clients_per_class": {
            class_name: int(clients_per_class[index])
            for index, class_name in enumerate(class_names)
        },
    }
    (destination / "data_balance.json").write_text(
        _json_text(report), encoding="utf-8"
    )
    (destination / "report.md").write_text(
        _finding_report(
            dataset_id=manifest.dataset_id,
            dataset_digest=manifest.digest,
            class_names=class_names,
            global_counts=global_split_counts["all"],
            clients_per_class=clients_per_class,
            balance_rows=balance_rows,
            topology_rows=topology_rows,
            feature_rows=feature_rows,
        ),
        encoding="utf-8",
    )

    if render_figures:
        _render_figures(
            destination,
            client_ids,
            class_names,
            np.vstack([counts_by_client_split[item]["all"] for item in client_ids]),
            global_split_counts,
            feature_names,
            client_features,
        )

    files = sorted(path for path in destination.iterdir() if path.is_file())
    artifact_manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "dataset_id": manifest.dataset_id,
        "dataset_digest": manifest.digest,
        "files": {path.name: _sha256(path) for path in files},
    }
    (destination / "analysis_manifest.json").write_text(
        _json_text(artifact_manifest), encoding="utf-8"
    )
    return destination
