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
    "flower-output-root": "artifacts/phase2/flower-runs",
}


def resolve_run_config(
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve toy defaults or authoritative IoT-23 benchmark settings."""
    resolved = {**DEFAULT_RUN_CONFIG, **dict(run_config)}
    if str(resolved.get("task", "toy")) != "iot23_manifest":
        return resolved

    from src.federated.config import load_phase2_config

    phase2 = load_phase2_config(
        str(resolved.get("phase2-config", "configs/phase2/iot23-federated.yaml"))
    )
    strategy = str(resolved["strategy"]).lower()
    if strategy not in phase2.federation.strategies:
        raise ValueError(
            f"Flower strategy {strategy!r} is not allowed by Phase 2 config."
        )
    resolved.update(
        {
            "num-server-rounds": phase2.training.rounds,
            "fraction-evaluate": 1.0,
            "local-epochs": phase2.training.local_epochs,
            "learning-rate": phase2.training.learning_rate,
            "weight-decay": phase2.training.weight_decay,
            "grad-clip": phase2.training.grad_clip,
            "optimizer": phase2.training.optimizer,
            "proximal-mu": (
                phase2.federation.proximal_mu if strategy == "fedprox" else 0.0
            ),
            "seed": phase2.training.seed,
            "evaluate-split": "val",
            "final-split": "test",
            "flower-output-root": phase2.observability.output_root,
            "benchmark-config-digest": phase2.digest,
        }
    )
    return resolved
