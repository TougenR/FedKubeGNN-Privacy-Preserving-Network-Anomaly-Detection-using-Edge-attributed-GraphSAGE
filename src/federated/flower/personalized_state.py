"""Durable, fail-closed storage for client-owned FedPer parameters."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import quote

import numpy as np

from src.federated.contracts.task import ArrayState
from src.federated.core.state import (
    copy_array_state,
    validate_array_state_like,
)
from src.federated.observability.run_store import atomic_json


class PersonalizedStateError(RuntimeError):
    """Raised when a private client checkpoint is missing or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PersonalizedStateStore:
    """Versioned private-state store with an atomically promoted pointer."""

    def __init__(
        self,
        root: str | Path,
        *,
        client_id: str,
        run_id: str,
        model_digest: str,
        personalized_prefixes: Sequence[str],
        initial_state: Mapping[str, np.ndarray],
    ) -> None:
        self.client_id = str(client_id)
        self.run_id = str(run_id)
        self.model_digest = str(model_digest)
        self.personalized_prefixes = tuple(personalized_prefixes)
        self.initial_state = copy_array_state(initial_state)
        self.root = (
            Path(root)
            / quote(self.client_id, safe="")
            / quote(self.run_id, safe="")
            / self.model_digest
        )
        self.metadata_path = self.root / "metadata.json"

    def _metadata(self) -> dict | None:
        if not self.metadata_path.exists():
            return None
        try:
            value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PersonalizedStateError(
                f"Cannot read personalized metadata: {self.metadata_path}"
            ) from exc
        expected = {
            "client_id": self.client_id,
            "run_id": self.run_id,
            "model_digest": self.model_digest,
            "personalized_prefixes": list(self.personalized_prefixes),
        }
        mismatches = {
            key: (value.get(key), expected_value)
            for key, expected_value in expected.items()
            if value.get(key) != expected_value
        }
        if mismatches:
            raise PersonalizedStateError(
                f"Personalized metadata provenance mismatch: {mismatches}"
            )
        return value

    def load(self, *, require_ready: bool) -> tuple[ArrayState, dict]:
        metadata = self._metadata()
        if metadata is None:
            if require_ready:
                raise PersonalizedStateError(
                    f"Client '{self.client_id}' is not inference-ready: no "
                    "completed personalized training round."
                )
            return copy_array_state(self.initial_state), {
                "ready": False,
                "completed_rounds": 0,
                "cold_start": True,
            }
        if not bool(metadata.get("ready")):
            raise PersonalizedStateError(
                f"Client '{self.client_id}' personalized metadata is not ready."
            )
        state_file = self.root / str(metadata.get("state_file", ""))
        if not state_file.is_file():
            raise PersonalizedStateError(
                f"Personalized checkpoint is missing: {state_file}"
            )
        if _sha256(state_file) != metadata.get("state_sha256"):
            raise PersonalizedStateError(
                f"Personalized checkpoint digest mismatch: {state_file}"
            )
        try:
            with np.load(state_file, allow_pickle=False) as archive:
                state = {
                    name: np.asarray(archive[name]).copy()
                    for name in archive.files
                }
        except (OSError, ValueError) as exc:
            raise PersonalizedStateError(
                f"Cannot load personalized checkpoint: {state_file}"
            ) from exc
        try:
            validate_array_state_like(
                state, self.initial_state, label="personalized state"
            )
        except ValueError as exc:
            raise PersonalizedStateError(str(exc)) from exc
        return state, dict(metadata)

    def save(self, state: Mapping[str, np.ndarray]) -> dict:
        validate_array_state_like(
            state, self.initial_state, label="personalized state"
        )
        previous = self._metadata()
        completed_rounds = (
            int(previous.get("completed_rounds", 0)) + 1 if previous else 1
        )
        self.root.mkdir(parents=True, exist_ok=True)
        filename = f"head-{completed_rounds:04d}.npz"
        destination = self.root / filename
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.", dir=self.root
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    **{
                        str(name): np.asarray(value)
                        for name, value in state.items()
                    },
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        metadata = {
            "client_id": self.client_id,
            "run_id": self.run_id,
            "model_digest": self.model_digest,
            "personalized_prefixes": list(self.personalized_prefixes),
            "ready": True,
            "completed_rounds": completed_rounds,
            "cold_start": previous is None,
            "state_file": filename,
            "state_sha256": _sha256(destination),
        }
        atomic_json(self.metadata_path, metadata)
        return metadata
