"""Dataset-level manifest for immutable prepared clients."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.federated.contracts.artifacts import ContractBundle
from src.federated.contracts.schema import ContractError
from src.federated.data.storage import (
    checksum_index_digest,
    load_graph_arrays,
    sha256_file,
)


MANIFEST_VERSION = 2


@dataclass(frozen=True)
class PreparedDatasetManifest:
    root: Path
    document: Mapping[str, Any]

    @property
    def dataset_id(self) -> str:
        return str(self.document["dataset_id"])

    @property
    def client_ids(self) -> tuple[str, ...]:
        return tuple(str(item["client_id"]) for item in self.document["clients"])

    @property
    def digest(self) -> str:
        payload = json.dumps(self.document, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        verify: bool = True,
        verify_clients: bool = True,
    ) -> "PreparedDatasetManifest":
        root = Path(directory)
        document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if int(document.get("manifest_version", 0)) != MANIFEST_VERSION:
            raise ContractError("Unsupported prepared dataset manifest version.")
        manifest = cls(root=root, document=document)
        if (
            len(manifest.client_ids) != len(set(manifest.client_ids))
            or not manifest.client_ids
        ):
            raise ContractError("Prepared dataset must have unique clients.")
        if verify:
            manifest.validate(verify_clients=verify_clients)
        return manifest

    def client_path(self, client_id: str) -> Path:
        entries = {
            str(item["client_id"]): str(item["path"])
            for item in self.document["clients"]
        }
        try:
            relative = entries[client_id]
        except KeyError as exc:
            raise KeyError(f"Unknown prepared client '{client_id}'.") from exc
        path = (self.root / relative).resolve()
        if self.root.resolve() not in path.parents:
            raise ContractError("Client artifact path escapes dataset root.")
        return path

    def client_entry(self, client_id: str) -> Mapping[str, Any]:
        for item in self.document["clients"]:
            if str(item["client_id"]) == client_id:
                return item
        raise KeyError(f"Unknown prepared client '{client_id}'.")

    def verify_client_digest(self, client_id: str) -> None:
        entry = self.client_entry(client_id)
        expected = str(entry["artifact_digest"])
        actual = checksum_index_digest(self.client_path(client_id))
        if actual != expected:
            raise ContractError(
                f"Client '{client_id}' artifact digest mismatch: "
                f"expected={expected}, actual={actual}."
            )

    def validate_client(
        self,
        client_id: str,
        *,
        bundle: ContractBundle,
    ) -> None:
        self.verify_client_digest(client_id)
        entry = self.client_entry(client_id)
        path = self.client_path(client_id)
        graph = load_graph_arrays(path, verify=True)
        if graph.feature_dim != bundle.feature_schema.feature_dim:
            raise ContractError(
                f"Client '{client_id}' feature_dim={graph.feature_dim} does not "
                f"match shared feature_dim={bundle.feature_schema.feature_dim}."
            )
        if graph.edge_label.size and (
            int(graph.edge_label.min()) < 0
            or int(graph.edge_label.max()) >= bundle.label_schema.num_classes
        ):
            raise ContractError(
                f"Client '{client_id}' labels are outside the shared [0, K) schema."
            )
        if int(entry["num_edges"]) != graph.num_edges:
            raise ContractError(
                f"Client '{client_id}' manifest num_edges does not match artifact."
            )
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        client_metadata = dict(metadata.get("metadata", {}))
        if str(client_metadata.get("client_id")) != client_id:
            raise ContractError(
                f"Client '{client_id}' metadata identifies a different client."
            )
        if client_metadata.get("graph_protocol") != self.document.get(
            "graph_protocol"
        ):
            raise ContractError(
                f"Client '{client_id}' graph protocol differs from the manifest."
            )

    def validate(self, *, verify_clients: bool = True) -> None:
        contract_root = self.root / str(self.document["contract_path"])
        bundle = ContractBundle.load(contract_root)
        expected_contract_digest = str(self.document["contract_digest"])
        actual_contract_digest = checksum_index_digest(contract_root)
        if actual_contract_digest != expected_contract_digest:
            raise ContractError(
                "Prepared manifest contract digest does not match contract content."
            )
        initial_path = self.root / str(self.document["initial_state_path"])
        expected = str(self.document["initial_state_sha256"])
        if sha256_file(initial_path) != expected:
            raise ContractError("Initial state checksum mismatch.")
        if bundle.model_spec is None:
            raise ContractError("Prepared contract must include a model spec.")
        with np.load(initial_path, allow_pickle=False) as archive:
            initial_state = {
                name: np.asarray(archive[name]).copy()
                for name in archive.files
            }
        bundle.model_spec.validate_state(initial_state)
        if verify_clients:
            for client_id in self.client_ids:
                self.validate_client(client_id, bundle=bundle)
