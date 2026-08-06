"""Read-only loader for portable prepared client graphs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


class EvaluationGraphError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_client_graph(directory: str | Path) -> Data:
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    for filename, expected in checksums.items():
        path = root / filename
        if not path.is_file() or _sha256(path) != expected:
            raise EvaluationGraphError(f"Client graph checksum mismatch: {path}")
    arrays = {
        name: np.load(root / f"{name}.npy", allow_pickle=False)
        for name in (
            "edge_index",
            "edge_attr",
            "edge_label",
            "train_mask",
            "val_mask",
            "test_mask",
        )
    }
    edge_index = torch.from_numpy(arrays["edge_index"]).long()
    edge_attr = torch.from_numpy(arrays["edge_attr"]).float()
    labels = torch.from_numpy(arrays["edge_label"]).long()
    num_nodes = int(metadata["num_nodes"])
    if edge_index.shape != (2, int(metadata["num_edges"])):
        raise EvaluationGraphError("Client edge_index shape differs from metadata.")
    if edge_attr.shape[1] != int(metadata["feature_dim"]):
        raise EvaluationGraphError("Client feature dimension differs from metadata.")
    graph = Data(
        x=torch.ones((num_nodes, 1), dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=labels,
        edge_label_binary=(labels != 0).long(),
        edge_index_mp=torch.cat([edge_index, edge_index.flip(0)], dim=1),
        edge_attr_mp=torch.cat([edge_attr, edge_attr], dim=0),
        train_mask=torch.from_numpy(arrays["train_mask"]).bool(),
        val_mask=torch.from_numpy(arrays["val_mask"]).bool(),
        test_mask=torch.from_numpy(arrays["test_mask"]).bool(),
        num_nodes=num_nodes,
    )
    return graph
