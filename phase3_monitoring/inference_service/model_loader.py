"""Validated model artifact loading for the Phase 3 inference PoC."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model import EGraphSAGE
from src.preprocess import Preprocessor


DEFAULT_PREPROCESSOR_PATH = (
    _REPO_ROOT
    / "artifacts"
    / "phase1_results"
    / "checkpoints"
    / "preprocessor.pkl"
)
DEFAULT_CHECKPOINT_PATH = (
    _REPO_ROOT
    / "artifacts"
    / "phase1_results"
    / "checkpoints"
    / "pooled_egraphsage_class_weight_seed42.pt"
)


class ModelContractError(RuntimeError):
    """Raised when inference artifacts cannot prove a compatible contract."""


@dataclass(frozen=True)
class RuntimeBundle:
    model: EGraphSAGE
    preprocessor: Preprocessor
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    feature_columns: tuple[str, ...]
    feature_schema_digest: str
    model_version: str
    checkpoint_path: str
    preprocessor_path: str
    device: str


def _configured_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _load_preprocessor(path: Path) -> Preprocessor:
    if not path.is_file():
        raise ModelContractError(f"Preprocessor not found: {path}")
    try:
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
    except Exception as error:
        raise ModelContractError(
            f"Could not load preprocessor {path}: {error}"
        ) from error
    # Accept the early hand-written wrapper only to provide a precise migration
    # error or load its actual Preprocessor; never fit at inference startup.
    if isinstance(loaded, dict) and "preprocessor" in loaded:
        loaded = loaded["preprocessor"]
    if not isinstance(loaded, Preprocessor):
        raise ModelContractError(
            f"{path} contains {type(loaded).__name__}, expected Preprocessor."
        )
    return loaded


def _load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    if not path.is_file():
        raise ModelContractError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except Exception as error:
        raise ModelContractError(
            f"Could not load checkpoint {path}: {error}"
        ) from error
    if not isinstance(checkpoint, dict):
        raise ModelContractError(
            f"{path} contains {type(checkpoint).__name__}, expected dict."
        )
    return checkpoint


def validate_model_contract(
    checkpoint: dict[str, Any],
    preprocessor: Preprocessor,
) -> tuple[tuple[str, ...], dict[str, int], str]:
    """Validate the complete feature and label contract before model loading."""

    required = {
        "state_dict",
        "cfg",
        "feature_dim",
        "feature_columns",
        "num_classes",
        "class_to_idx",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ModelContractError(
            "Checkpoint is missing inference contract fields "
            f"{missing}. Re-export/retrain it with the current Phase 1 or "
            "Phase 2 pipeline; dimension-only legacy checkpoints are rejected."
        )

    checkpoint_columns = tuple(str(c) for c in checkpoint["feature_columns"])
    preprocessor_columns = tuple(str(c) for c in preprocessor.feature_columns)
    feature_dim = int(checkpoint["feature_dim"])

    if len(checkpoint_columns) != feature_dim:
        raise ModelContractError(
            "Checkpoint feature_columns length "
            f"{len(checkpoint_columns)} != feature_dim {feature_dim}."
        )
    if preprocessor_columns != checkpoint_columns:
        mismatch_at = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(preprocessor_columns, checkpoint_columns)
                )
                if left != right
            ),
            min(len(preprocessor_columns), len(checkpoint_columns)),
        )
        raise ModelContractError(
            "Preprocessor/checkpoint feature schema mismatch: "
            f"preprocessor={len(preprocessor_columns)} columns, "
            f"checkpoint={len(checkpoint_columns)} columns, "
            f"first mismatch index={mismatch_at}. Position-blind padding is "
            "not permitted."
        )

    raw_mapping = checkpoint["class_to_idx"]
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ModelContractError("class_to_idx must be a non-empty mapping.")
    class_to_idx = {str(name): int(index) for name, index in raw_mapping.items()}
    expected_indices = list(range(len(class_to_idx)))
    if sorted(class_to_idx.values()) != expected_indices:
        raise ModelContractError(
            "class_to_idx values must be contiguous from 0 to K-1."
        )
    if int(checkpoint["num_classes"]) != len(class_to_idx):
        raise ModelContractError(
            "Checkpoint num_classes does not match class_to_idx."
        )

    schema_payload = json.dumps(
        {
            "feature_columns": list(checkpoint_columns),
            "class_to_idx": class_to_idx,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(schema_payload).hexdigest()
    return checkpoint_columns, class_to_idx, digest


def load_runtime_bundle(
    *,
    device: str | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    preprocessor_path: str | os.PathLike[str] | None = None,
) -> RuntimeBundle:
    """Load a model only after the preprocessing and label contracts match."""

    selected_device = device or os.environ.get("INFERENCE_DEVICE", "cpu")
    selected_checkpoint = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path
        else _configured_path("MODEL_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)
    )
    selected_preprocessor = (
        Path(preprocessor_path).expanduser().resolve()
        if preprocessor_path
        else _configured_path("PREPROCESSOR_PATH", DEFAULT_PREPROCESSOR_PATH)
    )

    preprocessor = _load_preprocessor(selected_preprocessor)
    checkpoint = _load_checkpoint(selected_checkpoint, selected_device)
    feature_columns, class_to_idx, schema_digest = validate_model_contract(
        checkpoint,
        preprocessor,
    )

    cfg_model = checkpoint.get("cfg", {}).get("model", {})
    required_model_fields = {"hidden_dim", "num_layers", "dropout"}
    missing_model_fields = sorted(required_model_fields - set(cfg_model))
    if missing_model_fields:
        raise ModelContractError(
            f"Checkpoint cfg.model is missing {missing_model_fields}."
        )

    model = EGraphSAGE(
        edge_dim=len(feature_columns),
        num_classes=len(class_to_idx),
        node_in_dim=1,
        hidden_dim=int(cfg_model["hidden_dim"]),
        num_layers=int(cfg_model["num_layers"]),
        dropout=float(cfg_model["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(selected_device)
    model.eval()

    model_version = os.environ.get(
        "MODEL_VERSION",
        f"{selected_checkpoint.stem}:{schema_digest[:12]}",
    )
    return RuntimeBundle(
        model=model,
        preprocessor=preprocessor,
        class_to_idx=class_to_idx,
        idx_to_class={index: name for name, index in class_to_idx.items()},
        feature_columns=feature_columns,
        feature_schema_digest=schema_digest,
        model_version=model_version,
        checkpoint_path=str(selected_checkpoint),
        preprocessor_path=str(selected_preprocessor),
        device=selected_device,
    )


if __name__ == "__main__":
    runtime = load_runtime_bundle()
    print(
        json.dumps(
            {
                "model_version": runtime.model_version,
                "feature_dim": len(runtime.feature_columns),
                "num_classes": len(runtime.class_to_idx),
                "feature_schema_digest": runtime.feature_schema_digest,
                "device": runtime.device,
            },
            indent=2,
        )
    )
