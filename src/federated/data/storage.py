"""Framework-neutral graph storage with integrity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.federated.contracts.schema import ContractError


GRAPH_FORMAT_VERSION = 1
ARRAY_FILES = {
    "edge_index": "edge_index.npy",
    "edge_attr": "edge_attr.npy",
    "edge_label": "edge_label.npy",
    "train_mask": "train_mask.npy",
    "val_mask": "val_mask.npy",
    "test_mask": "test_mask.npy",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_index_digest(directory: str | Path) -> str:
    """Digest a checksums index so a parent manifest can bind its contents."""
    root = Path(directory)
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    if not isinstance(checksums, dict) or not checksums:
        raise ContractError(f"Invalid or empty checksums index: {root}")
    canonical = json.dumps(
        {str(name): str(digest) for name, digest in checksums.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GraphArrays:
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_label: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    num_nodes: int

    def validate(self) -> None:
        edge_index = np.asarray(self.edge_index)
        edge_attr = np.asarray(self.edge_attr)
        labels = np.asarray(self.edge_label)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ContractError("edge_index must have shape [2, E].")
        edges = edge_index.shape[1]
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edges:
            raise ContractError("edge_attr must have shape [E, F].")
        if labels.shape != (edges,):
            raise ContractError("edge_label must have shape [E].")
        if edges < 1 or self.num_nodes < 1:
            raise ContractError("A client graph must contain nodes and edges.")
        if edge_index.dtype.kind not in "iu" or labels.dtype.kind not in "iu":
            raise ContractError("edge_index and edge_label must be integer arrays.")
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= self.num_nodes:
            raise ContractError("edge_index contains an out-of-range node id.")
        masks = []
        for name in ("train_mask", "val_mask", "test_mask"):
            mask = np.asarray(getattr(self, name))
            if mask.shape != (edges,) or mask.dtype != np.bool_:
                raise ContractError(f"{name} must be a boolean [E] array.")
            masks.append(mask)
        coverage = masks[0].astype(np.int8) + masks[1] + masks[2]
        if not np.all(coverage == 1):
            raise ContractError("Graph masks must be disjoint and cover every edge.")

    @property
    def feature_dim(self) -> int:
        return int(np.asarray(self.edge_attr).shape[1])

    @property
    def num_edges(self) -> int:
        return int(np.asarray(self.edge_index).shape[1])

    @classmethod
    def from_graph(cls, graph: Any) -> "GraphArrays":
        def array(name: str) -> np.ndarray:
            value = getattr(graph, name)
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        result = cls(
            edge_index=array("edge_index").astype(np.int64, copy=False),
            edge_attr=array("edge_attr").astype(np.float32, copy=False),
            edge_label=array("edge_label").astype(np.int64, copy=False),
            train_mask=array("train_mask").astype(np.bool_, copy=False),
            val_mask=array("val_mask").astype(np.bool_, copy=False),
            test_mask=array("test_mask").astype(np.bool_, copy=False),
            num_nodes=int(graph.num_nodes),
        )
        result.validate()
        return result


def write_graph_arrays(
    directory: str | Path,
    graph: GraphArrays,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Graph artifact directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    graph.validate()
    for name, filename in ARRAY_FILES.items():
        np.save(root / filename, np.asarray(getattr(graph, name)), allow_pickle=False)
    document = {
        "format_version": GRAPH_FORMAT_VERSION,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "feature_dim": graph.feature_dim,
        "split_counts": {
            "train": int(graph.train_mask.sum()),
            "validation": int(graph.val_mask.sum()),
            "test": int(graph.test_mask.sum()),
        },
        "class_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(graph.edge_label, return_counts=True))
        },
        "metadata": dict(metadata or {}),
    }
    (root / "metadata.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = [*ARRAY_FILES.values(), "metadata.json"]
    checksums = {name: sha256_file(root / name) for name in files}
    (root / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def load_graph_arrays(directory: str | Path, *, verify: bool = True) -> GraphArrays:
    root = Path(directory)
    if verify:
        checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
        for name, expected in checksums.items():
            actual = sha256_file(root / name)
            if actual != expected:
                raise ContractError(f"Checksum mismatch for client artifact '{name}'.")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata["format_version"]) != GRAPH_FORMAT_VERSION:
        raise ContractError(
            f"Unsupported graph format version {metadata['format_version']}."
        )
    arrays = {
        name: np.load(root / filename, allow_pickle=False)
        for name, filename in ARRAY_FILES.items()
    }
    graph = GraphArrays(num_nodes=int(metadata["num_nodes"]), **arrays)
    graph.validate()
    if graph.num_edges != int(metadata["num_edges"]) or graph.feature_dim != int(
        metadata["feature_dim"]
    ):
        raise ContractError("Graph metadata dimensions do not match arrays.")
    return graph


def load_pyg_graph(directory: str | Path) -> Any:
    """Reconstruct derived tensors only when a PyG client actually needs them."""
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError("Loading an IoT-23 graph requires torch-geometric.") from exc
    arrays = load_graph_arrays(directory)
    edge_index = torch.from_numpy(arrays.edge_index).long()
    edge_attr = torch.from_numpy(arrays.edge_attr).float()
    edge_label = torch.from_numpy(arrays.edge_label).long()
    graph = Data(
        x=torch.ones((arrays.num_nodes, 1), dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        edge_label_binary=(edge_label != 0).long(),
        edge_index_mp=torch.cat([edge_index, edge_index.flip(0)], dim=1),
        edge_attr_mp=torch.cat([edge_attr, edge_attr], dim=0),
        train_mask=torch.from_numpy(arrays.train_mask),
        val_mask=torch.from_numpy(arrays.val_mask),
        test_mask=torch.from_numpy(arrays.test_mask),
        num_nodes=arrays.num_nodes,
    )
    graph.feature_dim = arrays.feature_dim
    return graph
