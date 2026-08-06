"""Build an inference graph without accepting ground-truth labels."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import torch
from torch_geometric.data import Data

from src.core.preprocess import Preprocessor, clean_flows, transform


def preprocess_production_flows(
    records: list[dict], preprocessor: Preprocessor
) -> pd.DataFrame:
    """Reuse frozen preprocessing while keeping labels out of the API contract."""
    if not records:
        raise ValueError("A graph window must contain at least one flow.")
    frame = pd.DataFrame(records)
    # The shared Phase 1 cleaner preserves labels for evaluation, but labels are
    # not model features. Inject an internal placeholder after the request has
    # passed the label-forbidding production schema.
    frame["label"] = "-"
    frame["detailed-label"] = "-"
    return transform(clean_flows(frame), preprocessor)


def build_inference_graph(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    sensor_id: str,
) -> Data:
    if features.empty:
        raise ValueError("Cannot build an empty inference graph.")
    required = {"id.orig_h", "id.resp_h", *feature_columns}
    missing = sorted(required - set(features.columns))
    if missing:
        raise KeyError(f"Inference graph is missing columns: {missing}.")
    source_ids = (sensor_id + "::" + features["id.orig_h"].astype(str)).tolist()
    target_ids = (sensor_id + "::" + features["id.resp_h"].astype(str)).tolist()
    node_ids = sorted(set(source_ids) | set(target_ids))
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    sources = torch.tensor([node_index[node] for node in source_ids], dtype=torch.long)
    targets = torch.tensor([node_index[node] for node in target_ids], dtype=torch.long)
    edge_index = torch.stack((sources, targets), dim=0)
    edge_attr = torch.tensor(
        features[list(feature_columns)].astype("float32").to_numpy(),
        dtype=torch.float32,
    )
    graph = Data(
        x=torch.ones((len(node_ids), 1), dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_index_mp=torch.cat((edge_index, edge_index.flip(0)), dim=1),
        edge_attr_mp=torch.cat((edge_attr, edge_attr), dim=0),
        num_nodes=len(node_ids),
    )
    graph.node_ids = node_ids
    graph.sensor_id = sensor_id
    return graph
