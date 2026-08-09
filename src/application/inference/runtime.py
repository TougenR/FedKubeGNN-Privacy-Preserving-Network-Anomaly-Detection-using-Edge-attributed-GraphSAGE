"""Shared-encoder, trusted-routed FedPer prediction runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.application.inference.bundle_loader import FedPerServingBundle

if TYPE_CHECKING:
    from src.application.inference.fusion import FusedPrediction, MultiHeadFusionPolicy


@dataclass(frozen=True)
class RoutedPrediction:
    client_id: str
    probabilities: torch.Tensor
    predicted_indices: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True)
class MultiHeadPrediction:
    trusted: RoutedPrediction
    heads: dict[str, RoutedPrediction]
    fused: "FusedPrediction"


class CentralizedFedPerRuntime:
    def __init__(self, bundle: FedPerServingBundle) -> None:
        self.bundle = bundle

    def predict_graph(self, *, sensor_id: str, graph) -> RoutedPrediction:
        client_id = self.bundle.router.route(sensor_id)
        return self.predict_graph_for_client(client_id=client_id, graph=graph)

    def predict_graph_for_client(self, *, client_id: str, graph) -> RoutedPrediction:
        if client_id not in self.bundle.heads:
            raise KeyError(f"Unknown FedPer client '{client_id}'.")
        device_graph = graph.to(self.bundle.device)
        with torch.no_grad():
            node_embeddings = self.bundle.encoder.encode_nodes(device_graph)
            edge_repr = self.bundle.encoder.edge_representations(
                device_graph, node_embeddings
            )
            logits = self.bundle.heads[client_id](edge_repr)
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_indices = probabilities.max(dim=-1)
            entropy = -torch.sum(
                probabilities * torch.log(probabilities + 1e-12), dim=-1
            )
        return RoutedPrediction(
            client_id=client_id,
            probabilities=probabilities.cpu(),
            predicted_indices=predicted_indices.cpu(),
            confidence=confidence.cpu(),
            entropy=entropy.cpu(),
        )

    def predict_graph_all_heads(self, graph) -> dict[str, RoutedPrediction]:
        device_graph = graph.to(self.bundle.device)
        with torch.no_grad():
            node_embeddings = self.bundle.encoder.encode_nodes(device_graph)
            edge_repr = self.bundle.encoder.edge_representations(
                device_graph, node_embeddings
            )
            results: dict[str, RoutedPrediction] = {}
            for client_id, head in self.bundle.heads.items():
                probabilities = torch.softmax(head(edge_repr), dim=-1)
                confidence, predicted_indices = probabilities.max(dim=-1)
                entropy = -torch.sum(
                    probabilities * torch.log(probabilities + 1e-12), dim=-1
                )
                results[client_id] = RoutedPrediction(
                    client_id=client_id,
                    probabilities=probabilities.cpu(),
                    predicted_indices=predicted_indices.cpu(),
                    confidence=confidence.cpu(),
                    entropy=entropy.cpu(),
                )
        return results

    def predict_graph_with_fusion(
        self, *, sensor_id: str, graph, policy: "MultiHeadFusionPolicy"
    ) -> MultiHeadPrediction:
        client_id = self.bundle.router.route(sensor_id)
        predictions = self.predict_graph_all_heads(graph)
        return MultiHeadPrediction(
            trusted=predictions[client_id],
            heads=predictions,
            fused=policy.fuse(predictions),
        )
