"""Explicit component registry used by Phase 2 orchestration.

Registration is deliberately local and deterministic.  Configuration may pick
an implementation by name, but it cannot import arbitrary Python objects.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any


class RegistryError(ValueError):
    """Raised for unknown, duplicate, or malformed component registrations."""


Factory = Callable[..., Any]


class ComponentRegistry:
    """Registry partitioned by extension point (task, runtime, observer, ...)."""

    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Factory]] = defaultdict(dict)

    @staticmethod
    def _key(value: str, *, field: str) -> str:
        key = str(value).strip().lower()
        if not key or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in key
        ):
            raise RegistryError(f"Invalid {field} name: {value!r}.")
        return key

    def register(self, kind: str, name: str, factory: Factory) -> None:
        kind_key = self._key(kind, field="component kind")
        name_key = self._key(name, field="component")
        if not callable(factory):
            raise RegistryError(f"Factory for {kind_key}.{name_key} is not callable.")
        if name_key in self._factories[kind_key]:
            raise RegistryError(f"Component already registered: {kind_key}.{name_key}.")
        self._factories[kind_key][name_key] = factory

    def resolve(self, kind: str, name: str) -> Factory:
        kind_key = self._key(kind, field="component kind")
        name_key = self._key(name, field="component")
        try:
            return self._factories[kind_key][name_key]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories.get(kind_key, {}))) or "<none>"
            raise RegistryError(
                f"Unknown component {kind_key}.{name_key}; available: {available}."
            ) from exc

    def names(self, kind: str) -> tuple[str, ...]:
        kind_key = self._key(kind, field="component kind")
        return tuple(sorted(self._factories.get(kind_key, {})))


registry = ComponentRegistry()


def builtin_registry() -> ComponentRegistry:
    """Return a fresh registry containing supported Phase 2 implementations."""
    components = ComponentRegistry()

    def lazy(module: str, attribute: str) -> Factory:
        def resolve(*args: Any, **kwargs: Any) -> Any:
            from importlib import import_module

            return getattr(import_module(module), attribute)(*args, **kwargs)

        return resolve

    components.register(
        "data_source",
        "iot23",
        lazy("src.federated.data.sources.iot23", "read_clean_priority_sample"),
    )
    components.register(
        "partitioner",
        "scenario",
        lazy("src.federated.data.partitioners.scenario", "deterministic_edge_masks"),
    )
    components.register(
        "graph_builder", "phase1_ip_flow", lazy("src.graph_build", "build_graph")
    )
    components.register("model", "egraphsage", lazy("src.model", "build_model"))
    components.register(
        "task", "toy", lazy("src.federated.adapters.toy", "ToyFederatedTask")
    )
    components.register(
        "task", "iot23_manifest", lazy("src.federated.tasks.iot23", "ManifestIoT23Task")
    )
    components.register(
        "strategy", "fedavg", lazy("src.federated.strategies.fedavg", "FedAvgPolicy")
    )
    components.register(
        "strategy", "fedprox", lazy("src.federated.strategies.fedprox", "FedProxPolicy")
    )
    components.register(
        "runtime",
        "inprocess",
        lazy("src.federated.runtimes.inprocess", "run_observed_inprocess"),
    )
    components.register(
        "runtime", "flower", lazy("src.federated.flower.server_app", "build_server_app")
    )
    components.register(
        "observer",
        "console_jsonl",
        lazy("src.federated.observability.events", "CompositeObserver"),
    )
    return components
