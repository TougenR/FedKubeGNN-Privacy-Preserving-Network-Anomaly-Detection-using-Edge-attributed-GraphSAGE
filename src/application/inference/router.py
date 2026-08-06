"""Trusted server-side sensor-to-client routing."""

from __future__ import annotations

from typing import Mapping


class TrustedRoutingError(LookupError):
    """Raised when a sensor is not assigned to an approved FedPer client."""


class TrustedClientRouter:
    def __init__(self, sensor_client_mapping: Mapping[str, str]) -> None:
        mapping = {
            str(sensor): str(client)
            for sensor, client in sensor_client_mapping.items()
        }
        if not mapping or any(not sensor or not client for sensor, client in mapping.items()):
            raise ValueError("Trusted sensor mapping must be non-empty.")
        self._mapping = mapping

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._mapping)

    def route(self, sensor_id: str) -> str:
        try:
            return self._mapping[str(sensor_id)]
        except KeyError as exc:
            raise TrustedRoutingError(
                f"Sensor '{sensor_id}' has no trusted client assignment."
            ) from exc
