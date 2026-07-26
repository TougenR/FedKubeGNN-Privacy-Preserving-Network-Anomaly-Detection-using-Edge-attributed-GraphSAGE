"""Shared defaults for CLI and direct Flower execution."""

from __future__ import annotations

from typing import Any, Final, Mapping


DEFAULT_RUN_CONFIG: Final[dict[str, int | float | str | bool]] = {
    "num-server-rounds": 3,
    "fraction-evaluate": 1.0,
    "local-epochs": 1,
    "learning-rate": 0.15,
    "weight-decay": 0.0,
    "grad-clip": 1.0,
    "optimizer": "sgd",
    "strategy": "fedavg",
    "proximal-mu": 0.0,
    "seed": 42,
    "save-model": False,
    "model-output": "artifacts/phase2/toy_final_model.pt",
    "evaluate-split": "val",
    "events-output": "artifacts/phase2/flower-events",
}


def resolve_run_config(
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay runtime values on stable defaults without mutating either."""
    return {**DEFAULT_RUN_CONFIG, **dict(run_config)}
