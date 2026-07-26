"""Atomic run manifests and portable state checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.federated.contracts.task import ArrayState


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunStore:
    """Owns one immutable run identity and its mutable atomic status."""

    root: Path
    run_id: str
    strategy: str
    config_digest: str
    dataset_digest: str
    model_digest: str

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        *,
        strategy: str,
        config_digest: str,
        dataset_digest: str,
        model_digest: str,
        config_snapshot: Mapping[str, Any],
    ) -> "RunStore":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        identity = hashlib.sha256(
            f"{strategy}:{config_digest}:{dataset_digest}:{model_digest}:{timestamp}".encode(
                "utf-8"
            )
        ).hexdigest()[:10]
        run_id = f"{strategy}-{timestamp}-{identity}"
        root = Path(output_root) / run_id
        root.mkdir(parents=True, exist_ok=False)
        for name in ("events", "metrics", "checkpoints"):
            (root / name).mkdir()
        store = cls(
            root,
            run_id,
            strategy,
            config_digest,
            dataset_digest,
            model_digest,
        )
        atomic_json(root / "config.snapshot.json", dict(config_snapshot))
        store._write_status("running", strategy=strategy, started_at=_now())
        return store

    @classmethod
    def resume(
        cls,
        root: str | Path,
        *,
        strategy: str,
        config_digest: str,
        dataset_digest: str,
        model_digest: str,
    ) -> "RunStore":
        path = Path(root)
        manifest = json.loads((path / "run.json").read_text(encoding="utf-8"))
        expected = {
            "strategy": strategy,
            "config_digest": config_digest,
            "dataset_digest": dataset_digest,
            "model_digest": model_digest,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Cannot resume run with incompatible provenance: {mismatches}."
            )
        return cls(
            path,
            str(manifest["run_id"]),
            strategy,
            config_digest,
            dataset_digest,
            model_digest,
        )

    def _write_status(self, status: str, **fields: Any) -> None:
        current: dict[str, Any] = {}
        path = self.root / "run.json"
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
        if status == "running" and current.get("status") == "failed":
            current["previous_failure"] = current.pop("failure", "failure.json")
            current.pop("failed_at", None)
            current["resumed_at"] = _now()
        current.update(
            {
                "run_id": self.run_id,
                "status": status,
                "updated_at": _now(),
                "config_digest": self.config_digest,
                "dataset_digest": self.dataset_digest,
                "model_digest": self.model_digest,
                "strategy": self.strategy,
                **fields,
            }
        )
        atomic_json(path, current)

    def checkpoint(
        self,
        state: Mapping[str, np.ndarray],
        *,
        round_number: int,
        best: bool = False,
        mark_latest: bool = True,
    ) -> Path:
        path = self.root / "checkpoints" / f"round-{round_number:04d}.npz"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as handle:
                # Mapping order is part of ModelSpec and therefore part of the
                # transport contract; do not alphabetize checkpoint keys.
                np.savez(
                    handle,
                    **{str(key): np.asarray(value) for key, value in state.items()},
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        if best:
            _atomic_bytes(
                self.root / "checkpoints" / "best_model.npz", path.read_bytes()
            )
        if mark_latest:
            self.mark_round_committed(round_number, path)
        return path

    def promote_best(self, round_number: int) -> Path:
        source = self.root / "checkpoints" / f"round-{round_number:04d}.npz"
        if not source.is_file():
            raise FileNotFoundError(f"Cannot promote missing checkpoint: {source}")
        destination = self.root / "checkpoints" / "best_model.npz"
        _atomic_bytes(destination, source.read_bytes())
        return destination

    def mark_round_committed(self, round_number: int, checkpoint: str | Path) -> None:
        path = Path(checkpoint)
        self._write_status(
            "running",
            latest_round=round_number,
            latest_checkpoint=str(path.relative_to(self.root)),
        )

    def complete(self, **fields: Any) -> None:
        self._write_status("completed", completed_at=_now(), **fields)

    def fail(self, error: BaseException, **fields: Any) -> None:
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": _now(),
            **fields,
        }
        atomic_json(self.root / "failure.json", failure)
        self._write_status("failed", failed_at=_now(), failure="failure.json")

    def load_checkpoint(self, path: str | Path) -> ArrayState:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
