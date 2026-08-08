"""Validated authority for the Phase 4 demo scenario catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


EXPECTED_SCENARIOS = {
    "benign-browsing",
    "connection-burst",
    "request-flood",
    "slow-connections",
    "port-probe",
    "periodic-beacon",
}


class ParameterBound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum: int = Field(ge=0)
    maximum: int = Field(gt=0)
    unit: str

    @model_validator(mode="after")
    def ordered(self) -> "ParameterBound":
        if self.minimum > self.maximum:
            raise ValueError("parameter minimum cannot exceed maximum")
        return self


class ScenarioDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    category: str
    summary: str
    mechanism: str
    limitation: str
    defaults: dict[str, int]
    bounds: dict[str, ParameterBound]

    @model_validator(mode="after")
    def validate_defaults(self) -> "ScenarioDefinition":
        if set(self.defaults) != set(self.bounds):
            raise ValueError(f"scenario '{self.id}' defaults and bounds differ")
        self.validate_parameters(self.defaults)
        return self

    def validate_parameters(self, raw: dict[str, Any]) -> dict[str, int]:
        if set(raw) != set(self.bounds):
            raise ValueError(
                f"scenario '{self.id}' parameters must be {sorted(self.bounds)}"
            )
        validated: dict[str, int] = {}
        for name, bound in self.bounds.items():
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"parameter '{name}' must be an integer")
            if not bound.minimum <= value <= bound.maximum:
                raise ValueError(
                    f"parameter '{name}' must be between "
                    f"{bound.minimum} and {bound.maximum}"
                )
            validated[name] = value
        return validated


class ScenarioCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    sensor_id: str
    disclaimer: str
    model_classes: list[str]
    scenarios: list[ScenarioDefinition]

    @model_validator(mode="after")
    def validate_catalog(self) -> "ScenarioCatalog":
        if self.schema_version != 1:
            raise ValueError("unsupported demo scenario schema")
        scenario_ids = [scenario.id for scenario in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("demo scenario IDs must be unique")
        if set(scenario_ids) != EXPECTED_SCENARIOS:
            raise ValueError("demo catalog must contain the six approved scenarios")
        if len(self.model_classes) != 7 or "Benign" not in self.model_classes:
            raise ValueError("demo catalog must document all seven model classes")
        return self

    def scenario(self, scenario_id: str) -> ScenarioDefinition:
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        raise KeyError(f"unknown scenario '{scenario_id}'")


def load_catalog(path: str | Path) -> ScenarioCatalog:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("demo scenario catalog must be a YAML object")
    return ScenarioCatalog.model_validate(payload)
