"""Structured events without sensitive flow or tensor payloads."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol


_FORBIDDEN_FRAGMENTS = (
    "password",
    "secret",
    "token",
    "credential",
    "raw_ip",
    "edge_attr",
    "tensor",
    "private_key",
    "access_key",
)


class ObservabilityError(ValueError):
    """Raised when an event risks exposing sensitive or unbounded data."""


def _safe_value(value: Any, *, field: str) -> Any:
    if any(fragment in field.lower() for fragment in _FORBIDDEN_FRAGMENTS):
        raise ObservabilityError(f"Sensitive observability field rejected: {field!r}.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        if len(value) > 100:
            raise ObservabilityError(f"Observability field {field!r} is too large.")
        return [_safe_value(item, field=field) for item in value]
    if isinstance(value, Mapping):
        if len(value) > 100:
            raise ObservabilityError(f"Observability field {field!r} is too large.")
        return {
            str(key): _safe_value(item, field=str(key)) for key, item in value.items()
        }
    raise ObservabilityError(
        f"Observability field {field!r} has unsupported type {type(value).__name__}."
    )


def make_event(event: str, *, level: str = "INFO", **fields: Any) -> dict[str, Any]:
    """Create a JSON-safe event with a UTC timestamp and bounded fields."""
    name = str(event).strip()
    if not name:
        raise ObservabilityError("event must not be empty.")
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": str(level).upper(),
        "event": name,
    }
    for key, value in fields.items():
        if key in record:
            raise ObservabilityError(f"Reserved event field: {key!r}.")
        record[key] = _safe_value(value, field=key)
    return record


class Observer(Protocol):
    """Small seam for console, JSONL, OTel, or future metric exporters."""

    def emit(self, event: str, *, level: str = "INFO", **fields: Any) -> None: ...


class NoopObserver:
    def emit(self, event: str, *, level: str = "INFO", **fields: Any) -> None:
        del event, level, fields


class ConsoleObserver:
    """One compact JSON record per line for local operators."""

    def emit(self, event: str, *, level: str = "INFO", **fields: Any) -> None:
        print(json.dumps(make_event(event, level=level, **fields), sort_keys=True))


class JsonlObserver:
    """Process-local append-only JSONL sink.

    Distributed processes should write different files (for example one file per
    client).  Cross-process locking is intentionally not hidden in this class.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: str, *, level: str = "INFO", **fields: Any) -> None:
        line = (
            json.dumps(make_event(event, level=level, **fields), sort_keys=True) + "\n"
        )
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


class CompositeObserver:
    def __init__(self, *observers: Observer) -> None:
        self._observers = tuple(observers)

    def emit(self, event: str, *, level: str = "INFO", **fields: Any) -> None:
        for observer in self._observers:
            observer.emit(event, level=level, **fields)
