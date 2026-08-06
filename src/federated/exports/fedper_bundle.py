"""Export an immutable, centralized research bundle from exact FedPer state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from src.core.preprocess import FrozenStandardScaler, Preprocessor
from src.federated.contracts.artifacts import ContractBundle


BUNDLE_SCHEMA_VERSION = 2
EXPECTED_CLIENTS = ("1-1", "3-1", "9-1", "34-1", "36-1", "39-1")
PRIVATE_PREFIX = "head."


class FedPerBundleExportError(RuntimeError):
    """Raised before publishing a bundle with inconsistent provenance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FedPerBundleExportError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise FedPerBundleExportError(f"Expected JSON object in {path}.")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FedPerBundleExportError(f"Missing NumPy checkpoint: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {
                str(name): np.asarray(archive[name]).copy()
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise FedPerBundleExportError(f"Cannot load checkpoint: {path}") from exc


def _make_bundle_read_only(root: Path) -> None:
    """Publish runtime-readable artifacts without writable bundle contents."""
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _parameter_specs(model_spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    parameters = model_spec.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        raise FedPerBundleExportError("model-spec.json has no parameters.")
    specs: dict[str, dict[str, Any]] = {}
    for item in parameters:
        if not isinstance(item, dict) or "name" not in item:
            raise FedPerBundleExportError("Invalid model parameter specification.")
        name = str(item["name"])
        if name in specs:
            raise FedPerBundleExportError(f"Duplicate model parameter '{name}'.")
        specs[name] = item
    return specs


def _validate_state(
    state: Mapping[str, np.ndarray],
    specs: Mapping[str, Mapping[str, Any]],
    *,
    expected_names: tuple[str, ...],
    label: str,
) -> None:
    if tuple(state) != expected_names:
        raise FedPerBundleExportError(
            f"{label} parameter names/order mismatch: expected={expected_names}, "
            f"actual={tuple(state)}."
        )
    for name in expected_names:
        value = np.asarray(state[name])
        expected_shape = tuple(int(size) for size in specs[name]["shape"])
        expected_dtype = str(specs[name]["dtype"])
        if value.shape != expected_shape or str(value.dtype) != expected_dtype:
            raise FedPerBundleExportError(
                f"{label} parameter '{name}' expected {expected_shape}/"
                f"{expected_dtype}, got {value.shape}/{value.dtype}."
            )


def _preprocessor_from_contract(contract: ContractBundle) -> Preprocessor:
    required_arrays = {"scaler_mean", "scaler_scale", "scaler_var"}
    missing = sorted(required_arrays - set(contract.learned_arrays))
    if missing:
        raise FedPerBundleExportError(
            f"Prepared contract is missing learned arrays: {missing}."
        )
    scaler = FrozenStandardScaler(
        mean_=contract.learned_arrays["scaler_mean"],
        scale_=contract.learned_arrays["scaler_scale"],
        var_=contract.learned_arrays["scaler_var"],
    )
    categories = contract.categories
    return Preprocessor(
        resp_port_categories=list(categories.get("resp_port", ())),
        proto_categories=list(categories.get("proto", ())),
        service_categories=list(categories.get("service", ())),
        conn_state_categories=list(categories.get("conn_state", ())),
        history_flag_chars=list(categories.get("history_flags", ())),
        numeric_columns=list(categories.get("numeric_columns", ())),
        missing_flag_columns=list(categories.get("missing_flags", ())),
        scaler=scaler,
        feature_columns=list(contract.feature_schema.names),
    )


def export_fedper_bundle(
    *,
    run_root: str | Path,
    heads_root: str | Path,
    prepared_root: str | Path,
    destination: str | Path,
    sensor_mapping: Mapping[str, str] | None = None,
) -> Path:
    """Validate exact source state and atomically publish one serving bundle."""
    run_root = Path(run_root).resolve()
    heads_root = Path(heads_root).resolve()
    prepared_root = Path(prepared_root).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Bundle destination already exists: {destination}")

    run = _read_json(run_root / "run.json")
    summary = _read_json(run_root / "metrics" / "summary.json")
    prepared_manifest = _read_json(prepared_root / "manifest.json")
    contract = ContractBundle.load(prepared_root / "contract")
    if contract.model_spec is None:
        raise FedPerBundleExportError("Prepared contract has no model spec.")
    model_spec = contract.model_spec.to_dict()
    specs = _parameter_specs(model_spec)

    if run.get("status") != "completed" or run.get("strategy") != "fedper":
        raise FedPerBundleExportError("Source run is not a completed FedPer run.")
    best_round = int(run.get("best_round", 0))
    if best_round < 1 or best_round != int(summary.get("best_round", -1)):
        raise FedPerBundleExportError("Run and summary disagree on best_round.")
    model_digest = str(run.get("model_digest", ""))
    if model_digest != contract.model_spec.digest:
        raise FedPerBundleExportError(
            "Run model digest does not match the prepared model contract."
        )
    flower_run_id = str(summary.get("flower_run_id", ""))
    if not flower_run_id:
        raise FedPerBundleExportError("Summary has no Flower run ID.")

    shared_names = tuple(name for name in specs if not name.startswith(PRIVATE_PREFIX))
    private_names = tuple(name for name in specs if name.startswith(PRIVATE_PREFIX))
    if not shared_names or not private_names:
        raise FedPerBundleExportError("Model spec does not define FedPer boundaries.")
    shared_source = run_root / "checkpoints" / "best_model.npz"
    _validate_state(
        _load_npz(shared_source),
        specs,
        expected_names=shared_names,
        label="shared encoder",
    )

    mapping = dict(sensor_mapping or {f"sensor-{client}": client for client in EXPECTED_CLIENTS})
    if set(mapping.values()) != set(EXPECTED_CLIENTS):
        raise FedPerBundleExportError(
            "Trusted sensor mapping must cover exactly the six FedPer clients."
        )
    if any(not sensor or not client for sensor, client in mapping.items()):
        raise FedPerBundleExportError("Trusted sensor mapping contains an empty ID.")

    head_sources: dict[str, tuple[Path, dict[str, Any]]] = {}
    for client_id in EXPECTED_CLIENTS:
        metadata = _read_json(heads_root / client_id / "metadata.json")
        expected_metadata = {
            "client_id": client_id,
            "run_id": flower_run_id,
            "model_digest": model_digest,
            "completed_rounds": best_round,
            "ready": True,
            "cold_start": False,
            "personalized_prefixes": [PRIVATE_PREFIX],
            "state_file": f"head-{best_round:04d}.npz",
        }
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise FedPerBundleExportError(
                f"Head metadata mismatch for '{client_id}': {mismatches}."
            )
        source = heads_root / client_id / str(metadata["state_file"])
        if _sha256(source) != metadata.get("state_sha256"):
            raise FedPerBundleExportError(f"Head digest mismatch for '{client_id}'.")
        _validate_state(
            _load_npz(source),
            specs,
            expected_names=private_names,
            label=f"head {client_id}",
        )
        head_sources[client_id] = (source, metadata)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        heads_destination = temporary / "heads"
        heads_destination.mkdir()
        shared_destination = temporary / "shared-encoder.npz"
        shutil.copy2(shared_source, shared_destination)
        copied_heads: dict[str, dict[str, Any]] = {}
        for client_id, (source, metadata) in head_sources.items():
            relative = Path("heads") / f"{client_id}.npz"
            target = temporary / relative
            shutil.copy2(source, target)
            copied_heads[client_id] = {
                "path": relative.as_posix(),
                "sha256": _sha256(target),
                "completed_rounds": int(metadata["completed_rounds"]),
                "ready": True,
                "cold_start": False,
                "source_state_file": str(metadata["state_file"]),
            }

        contract_files = {
            "model_spec": "model-spec.json",
            "feature_schema": "feature-schema.json",
            "label_schema": "label-schema.json",
            "graph_schema": "graph-schema.json",
        }
        contract_sources = {
            "model_spec": prepared_root / "contract" / "model_spec.json",
            "feature_schema": prepared_root / "contract" / "feature_schema.json",
            "label_schema": prepared_root / "contract" / "label_schema.json",
            "graph_schema": prepared_root / "contract" / "graph_schema.json",
        }
        artifacts: dict[str, dict[str, str]] = {
            "shared_encoder": {
                "path": shared_destination.name,
                "sha256": _sha256(shared_destination),
            }
        }
        for key, filename in contract_files.items():
            target = temporary / filename
            shutil.copy2(contract_sources[key], target)
            artifacts[key] = {"path": filename, "sha256": _sha256(target)}

        preprocessor_path = temporary / "preprocessor.pkl"
        _preprocessor_from_contract(contract).save(str(preprocessor_path))
        artifacts["preprocessor"] = {
            "path": preprocessor_path.name,
            "sha256": _sha256(preprocessor_path),
        }

        bundle_id = (
            f"fedper-gke-{flower_run_id}-r{best_round:04d}-"
            f"{model_digest[:12]}-b{BUNDLE_SCHEMA_VERSION:02d}"
        )
        manifest = {
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "centralized-fedper-research-demo",
            "fedper_run_id": str(run["run_id"]),
            "flower_run_id": flower_run_id,
            "dataset_id": str(prepared_manifest["dataset_id"]),
            "dataset_digest": str(run["dataset_digest"]),
            "model_digest": model_digest,
            "best_round": best_round,
            "personalized_parameter_prefixes": [PRIVATE_PREFIX],
            "shared_parameter_names": list(shared_names),
            "private_parameter_names": list(private_names),
            "shared_encoder_digest": artifacts["shared_encoder"]["sha256"],
            "head_digests": {
                client: document["sha256"]
                for client, document in copied_heads.items()
            },
            "client_head_mapping": {
                client: f"heads/{client}.npz" for client in EXPECTED_CLIENTS
            },
            "sensor_client_mapping": mapping,
            "feature_order": list(contract.feature_schema.names),
            "feature_schema_digest": contract.feature_schema.digest,
            "label_mapping": contract.label_schema.class_to_idx,
            "label_schema_digest": contract.label_schema.digest,
            "training_graph_protocol": str(prepared_manifest["graph_protocol"]),
            "graph_protocol": None,
            "serving_ready": False,
            "serving_readiness_reason": "rolling-window protocol not selected",
            "artifacts": artifacts,
            "heads": copied_heads,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _make_bundle_read_only(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--heads-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    exported = export_fedper_bundle(
        run_root=args.run_root,
        heads_root=args.heads_root,
        prepared_root=args.prepared_root,
        destination=args.destination,
    )
    print(exported)


if __name__ == "__main__":
    main()
