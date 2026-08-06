"""Deterministic event-time rolling buffer with explicit emission policy."""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class TimedFlow:
    timestamp: float
    sequence: int
    payload: dict[str, Any] = field(compare=False)


@dataclass(frozen=True)
class RollingWindowConfig:
    duration_seconds: float
    max_flows: int
    emit_stride_flows: int
    allowed_lateness_seconds: float

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0 or self.max_flows < 1:
            raise ValueError("Window duration and flow limit must be positive.")
        if self.emit_stride_flows < 1:
            raise ValueError("emit_stride_flows must be positive.")
        if self.allowed_lateness_seconds < 0:
            raise ValueError("allowed_lateness_seconds cannot be negative.")


@dataclass(frozen=True)
class WindowSnapshot:
    sensor_id: str
    window_id: str
    start_timestamp: float
    end_timestamp: float
    flows: tuple[dict[str, Any], ...]
    emission_indices: tuple[int, ...]


class RollingWindowBuffer:
    """One sensor-local buffer; no flow can create cross-sensor graph edges."""

    def __init__(self, *, sensor_id: str, config: RollingWindowConfig) -> None:
        if not sensor_id:
            raise ValueError("sensor_id is required.")
        self.sensor_id = sensor_id
        self.config = config
        self._flows: list[TimedFlow] = []
        self._sequence = 0
        self._accepted = 0
        self._emitted = 0
        self._pending_sequences: list[int] = []
        self._max_seen = float("-inf")
        self.late_drop_count = 0
        self.capacity_drop_count = 0

    def add(self, payload: dict[str, Any]) -> WindowSnapshot | None:
        timestamp = float(payload["ts"])
        watermark_before = self._max_seen - self.config.allowed_lateness_seconds
        if timestamp < watermark_before:
            self.late_drop_count += 1
            return None
        self._max_seen = max(self._max_seen, timestamp)
        self._sequence += 1
        insort(
            self._flows,
            TimedFlow(timestamp=timestamp, sequence=self._sequence, payload=dict(payload)),
        )
        self._pending_sequences.append(self._sequence)
        cutoff = self._max_seen - self.config.duration_seconds
        self._flows = [flow for flow in self._flows if flow.timestamp >= cutoff]
        if len(self._flows) > self.config.max_flows:
            excess = len(self._flows) - self.config.max_flows
            self._flows = self._flows[excess:]
            self.capacity_drop_count += excess
        self._accepted += 1
        if self._accepted % self.config.emit_stride_flows:
            return None
        return self._snapshot()

    def flush(self) -> WindowSnapshot | None:
        """Emit accepted flows not covered by the last stride boundary."""
        if not self._flows or not self._pending_sequences:
            return None
        return self._snapshot()

    def _snapshot(self) -> WindowSnapshot:
        self._emitted += 1
        pending = set(self._pending_sequences)
        snapshot = WindowSnapshot(
            sensor_id=self.sensor_id,
            window_id=f"{self.sensor_id}-window-{self._emitted:08d}",
            start_timestamp=self._flows[0].timestamp,
            end_timestamp=self._max_seen,
            flows=tuple(dict(flow.payload) for flow in self._flows),
            emission_indices=tuple(
                index for index, flow in enumerate(self._flows) if flow.sequence in pending
            ),
        )
        self._pending_sequences.clear()
        return snapshot

    @property
    def flow_drop_rate(self) -> float:
        # Expiring old context from a bounded rolling window is an eviction, not
        # an input-flow loss: the flow was accepted and evaluated on arrival.
        total = self._accepted + self.late_drop_count
        return self.late_drop_count / total if total else 0.0

    @property
    def capacity_eviction_rate(self) -> float:
        return self.capacity_drop_count / self._accepted if self._accepted else 0.0
