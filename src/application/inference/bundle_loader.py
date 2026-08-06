"""Fail-closed loader for immutable centralized FedPer serving bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.core.model import EGraphSAGE
from src.core.preprocess import Preprocessor
from src.application.inference.router import TrustedClientRouter


SUPPORTED_BUNDLE_SCHEMA_VERSIONS = {1, 2}


class InferenceBundleError(RuntimeError):
    """Raised when an application bundle cannot prove a safe runtime."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceBundleError(f"Cannot read bundle JSON: {path}") from exc
    if not isinstance(value, dict):
        raise InferenceBundleError(f"Expected a JSON object in {path}.")
    return value


def _safe_artifact(root: Path, document: Mapping[str, Any], label: str) -> Path:
    relative = Path(str(document.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise InferenceBundleError(f"Unsafe path for {label}: {relative}")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise InferenceBundleError(f"Missing bundle artifact for {label}: {path}")
    expected = str(document.get("sha256", ""))
    actual = _sha256(path)
    if actual != expected:
        raise InferenceBundleError(
            f"Digest mismatch for {label}: expected={expected}, actual={actual}."
        )
    return path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {
                str(name): np.asarray(archive[name]).copy()
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise InferenceBundleError(f"Cannot load NumPy state: {path}") from exc


def _validate_state(
    state: Mapping[str, np.ndarray],
    specs: Mapping[str, Mapping[str, Any]],
    names: tuple[str, ...],
    label: str,
) -> None:
    if tuple(state) != names:
        raise InferenceBundleError(f"{label} parameter names/order mismatch.")
    for name in names:
        expected_shape = tuple(int(size) for size in specs[name]["shape"])
        expected_dtype = str(specs[name]["dtype"])
        value = np.asarray(state[name])
        if value.shape != expected_shape or str(value.dtype) != expected_dtype:
            raise InferenceBundleError(
                f"{label} parameter '{name}' schema mismatch."
            )


@dataclass(frozen=True)
class FedPerServingBundle:
    root: Path
    manifest: dict[str, Any]
    preprocessor: Preprocessor
    encoder: EGraphSAGE
    heads: dict[str, torch.nn.Module]
    router: TrustedClientRouter
    class_to_idx: dict[str, int]
    device: str

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {index: name for name, index in self.class_to_idx.items()}


def load_inference_bundle(
    directory: str | Path,
    *,
    device: str = "cpu",
    require_serving_ready: bool = False,
) -> FedPerServingBundle:
    root = Path(directory).resolve()
    manifest = _read_json(root / "manifest.json")
    schema_version = int(manifest.get("bundle_schema_version", 0))
    if schema_version not in SUPPORTED_BUNDLE_SCHEMA_VERSIONS:
        raise InferenceBundleError("Unsupported bundle schema version.")
    if require_serving_ready:
        if schema_version < 2 or not manifest.get("serving_ready"):
            raise InferenceBundleError(
                "Bundle is research-only: serving graph protocol is not selected."
            )
        graph_protocol = manifest.get("graph_protocol")
        if not isinstance(graph_protocol, str) or not graph_protocol:
            raise InferenceBundleError("Serving graph protocol is missing.")
    artifacts = manifest.get("artifacts")
    heads_document = manifest.get("heads")
    if not isinstance(artifacts, dict) or not isinstance(heads_document, dict):
        raise InferenceBundleError("Bundle manifest has no artifacts/heads registry.")

    artifact_paths = {
        name: _safe_artifact(root, document, name)
        for name, document in artifacts.items()
        if isinstance(document, dict)
    }
    required = {
        "shared_encoder",
        "preprocessor",
        "model_spec",
        "feature_schema",
        "label_schema",
        "graph_schema",
    }
    if required - set(artifact_paths):
        raise InferenceBundleError(
            f"Bundle is missing required artifacts: {sorted(required - set(artifact_paths))}."
        )

    model_spec = _read_json(artifact_paths["model_spec"])
    feature_schema = _read_json(artifact_paths["feature_schema"])
    label_schema = _read_json(artifact_paths["label_schema"])
    if _canonical_digest(model_spec) != manifest.get("model_digest"):
        raise InferenceBundleError("Model spec digest does not match manifest.")
    if _canonical_digest(feature_schema) != manifest.get("feature_schema_digest"):
        raise InferenceBundleError("Feature schema digest does not match manifest.")
    if _canonical_digest(label_schema) != manifest.get("label_schema_digest"):
        raise InferenceBundleError("Label schema digest does not match manifest.")

    feature_order = tuple(str(field["name"]) for field in feature_schema["fields"])
    if feature_order != tuple(str(name) for name in manifest.get("feature_order", ())):
        raise InferenceBundleError("Feature order differs from manifest.")
    classes = tuple(str(name) for name in label_schema["classes"])
    class_to_idx = {name: index for index, name in enumerate(classes)}
    if class_to_idx != manifest.get("label_mapping"):
        raise InferenceBundleError("Label mapping differs from manifest.")

    try:
        with artifact_paths["preprocessor"].open("rb") as handle:
            preprocessor = pickle.load(handle)
    except Exception as exc:
        raise InferenceBundleError("Cannot load validated preprocessor.") from exc
    if not isinstance(preprocessor, Preprocessor):
        raise InferenceBundleError("preprocessor.pkl has an unexpected type.")
    if tuple(preprocessor.feature_columns) != feature_order:
        raise InferenceBundleError("Preprocessor feature order does not match schema.")

    parameter_items = model_spec.get("parameters", [])
    specs = {str(item["name"]): item for item in parameter_items}
    if len(specs) != len(parameter_items):
        raise InferenceBundleError("Model spec has duplicate parameters.")
    shared_names = tuple(str(name) for name in manifest.get("shared_parameter_names", ()))
    private_names = tuple(str(name) for name in manifest.get("private_parameter_names", ()))
    if set(shared_names) | set(private_names) != set(specs):
        raise InferenceBundleError("FedPer parameter boundary is incomplete.")
    if set(shared_names) & set(private_names):
        raise InferenceBundleError("FedPer parameter boundary overlaps.")

    shared_state = _load_npz(artifact_paths["shared_encoder"])
    _validate_state(shared_state, specs, shared_names, "shared encoder")
    hyperparameters = model_spec["hyperparameters"]
    encoder = EGraphSAGE(
        edge_dim=int(model_spec["feature_dim"]),
        num_classes=int(model_spec["num_classes"]),
        node_in_dim=int(model_spec["node_feature_dim"]),
        hidden_dim=int(hyperparameters["hidden_dim"]),
        num_layers=int(hyperparameters["num_layers"]),
        dropout=float(hyperparameters["dropout"]),
    )
    template = encoder.state_dict()
    shared_tensors = {
        name: torch.from_numpy(value.copy()).to(dtype=template[name].dtype)
        for name, value in shared_state.items()
    }
    incompatible = encoder.load_state_dict(shared_tensors, strict=False)
    if set(incompatible.missing_keys) != set(private_names) or incompatible.unexpected_keys:
        raise InferenceBundleError("Shared encoder cannot be loaded into model spec.")
    encoder.to(device).eval()

    client_mapping = manifest.get("client_head_mapping")
    if not isinstance(client_mapping, dict) or set(client_mapping) != set(heads_document):
        raise InferenceBundleError("Client/head mapping is incomplete.")
    heads: dict[str, torch.nn.Module] = {}
    for client_id, document in heads_document.items():
        if not isinstance(document, dict):
            raise InferenceBundleError(f"Invalid head registry for '{client_id}'.")
        if not document.get("ready") or document.get("cold_start"):
            raise InferenceBundleError(f"Head '{client_id}' is not inference-ready.")
        if int(document.get("completed_rounds", -1)) != int(manifest["best_round"]):
            raise InferenceBundleError(f"Head '{client_id}' is not at best_round.")
        path = _safe_artifact(root, document, f"head {client_id}")
        if path.relative_to(root).as_posix() != client_mapping[client_id]:
            raise InferenceBundleError(f"Head path mapping mismatch for '{client_id}'.")
        state = _load_npz(path)
        _validate_state(state, specs, private_names, f"head {client_id}")
        head = copy.deepcopy(encoder.head)
        stripped = {
            name.removeprefix("head."): torch.from_numpy(value.copy())
            for name, value in state.items()
        }
        head.load_state_dict(stripped, strict=True)
        heads[str(client_id)] = head.to(device).eval()

    router = TrustedClientRouter(manifest.get("sensor_client_mapping", {}))
    if set(router.mapping.values()) != set(heads):
        raise InferenceBundleError("Trusted routing does not cover every head.")
    return FedPerServingBundle(
        root=root,
        manifest=manifest,
        preprocessor=preprocessor,
        encoder=encoder,
        heads=heads,
        router=router,
        class_to_idx=class_to_idx,
        device=device,
    )
