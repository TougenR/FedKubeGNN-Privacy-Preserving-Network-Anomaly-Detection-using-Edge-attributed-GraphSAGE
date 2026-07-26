"""Unified Flower task selection for toy smoke and prepared IoT-23 clients."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.federated.config import load_phase2_config
from src.federated.experiments.factory import task_from_name
from src.federated.observability import (
    CompositeObserver,
    ConsoleObserver,
    JsonlObserver,
)


def task_factory(context: Any):
    run = dict(context.run_config)
    name = str(run.get("task", "toy"))
    config = load_phase2_config(
        str(run.get("phase2-config", "configs/phase2/iot23-federated.yaml"))
    )
    dataset = run.get("dataset-root")
    return task_from_name(
        name,
        config=config,
        dataset_root=Path(str(dataset)) if dataset else None,
        observer=flower_observer(context, "task"),
    )


def flower_observer(context: Any, role: str):
    run = dict(context.run_config)
    output = Path(str(run.get("events-output", "artifacts/phase2/flower-events")))
    node = str(
        context.node_config.get(
            "partition-id", context.node_config.get("client-id", "server")
        )
    )
    filename = f"{context.run_id}-{role}-{node}.jsonl"
    return CompositeObserver(ConsoleObserver(), JsonlObserver(output / filename))
