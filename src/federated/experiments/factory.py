"""Construct configured tasks while keeping concrete imports at the boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.federated.adapters.phase1_iot23 import make_phase1_model_factory
from src.federated.config.schema import Phase2Config
from src.federated.observability.events import Observer
from src.federated.tasks.iot23 import ManifestIoT23Task
from src.federated.registry import ComponentRegistry, builtin_registry


def manifest_task(
    config: Phase2Config,
    dataset_root: str | Path,
    *,
    observer: Observer | None = None,
    device: str | None = None,
) -> ManifestIoT23Task:
    model_factory = make_phase1_model_factory(
        model_name=config.components.model,
        cfg={"model": config.model.__dict__},
    )
    return ManifestIoT23Task(
        dataset_root,
        model_factory=model_factory,
        imbalance_mode=config.training.imbalance,
        device=device,
        observer=observer,
    )


def task_from_name(
    name: str,
    *,
    config: Phase2Config,
    dataset_root: str | Path | None = None,
    observer: Observer | None = None,
    components: ComponentRegistry | None = None,
) -> Any:
    registry = components or builtin_registry()
    task_factory = registry.resolve("task", name)
    if name == "toy":
        return task_factory(seed=config.training.seed)
    if name == "iot23_manifest":
        if dataset_root is None:
            raise ValueError("task=iot23_manifest requires a prepared dataset root.")
        model_factory = make_phase1_model_factory(
            model_name=config.components.model,
            cfg={"model": config.model.__dict__},
        )
        return task_factory(
            dataset_root,
            model_factory=model_factory,
            imbalance_mode=config.training.imbalance,
            observer=observer,
        )
    raise AssertionError(name)
