"""Dataset-level manifest for immutable prepared clients."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.federated.contracts.artifacts import ContractBundle
from src.federated.contracts.schema import ContractError
from src.federated.data.storage import load_graph_arrays, sha256_file


MANIFEST_VERSION = 1


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

    def validate(self, *, verify_clients: bool = True) -> None:
        ContractBundle.load(self.root / str(self.document["contract_path"]))
        initial_path = self.root / str(self.document["initial_state_path"])
        expected = str(self.document["initial_state_sha256"])
        if sha256_file(initial_path) != expected:
            raise ContractError("Initial state checksum mismatch.")
        if verify_clients:
            for client_id in self.client_ids:
                load_graph_arrays(self.client_path(client_id), verify=True)
