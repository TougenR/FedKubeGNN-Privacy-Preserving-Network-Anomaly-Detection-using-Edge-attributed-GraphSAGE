"""Fail-closed traffic-profile and private-target authority."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


ScientificStatus = Literal[
    "candidate",
    "control-not-class-equivalent",
    "blocked-target-not-ready",
    "unsupported-dataset-artifact",
]
Mechanism = Literal[
    "http-get",
    "ssh-session",
    "irc-mixed",
    "syn-only",
    "syn-only-round-robin",
    "ack-only",
]
PRIVATE_IPV4_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


class ExpectedObservables(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocols: list[str] = Field(min_length=1)
    services: list[str] = Field(min_length=1)
    connection_states: list[str] = Field(min_length=1)
    note: str = Field(min_length=1, max_length=500)


class TrafficProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    reference_class: str = Field(min_length=1)
    scientific_status: ScientificStatus
    execution_enabled: bool
    mechanism: Mechanism
    target_group: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    destination_port: int = Field(ge=1, le=65535)
    events: int = Field(ge=1, le=50)
    interval_ms: int = Field(ge=4, le=60000)
    expected_observables: ExpectedObservables

    @model_validator(mode="after")
    def executable_status_is_consistent(self) -> "TrafficProfile":
        blocked = {
            "blocked-target-not-ready",
            "unsupported-dataset-artifact",
        }
        if self.execution_enabled == (self.scientific_status in blocked):
            raise ValueError("Blocked/unsupported profiles cannot be executable.")
        if self.mechanism.endswith("round-robin") and self.events < 2:
            raise ValueError("Round-robin profiles require at least two events.")
        if (self.events - 1) * self.interval_ms > 120000:
            raise ValueError("Traffic profile runtime cannot exceed two minutes.")
        return self


class TrafficProfileCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    kind: str
    reference_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_protocol: str
    claim_boundary: str = Field(min_length=1)
    profiles: list[TrafficProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self) -> "TrafficProfileCatalog":
        if self.schema_version != 1:
            raise ValueError("Unsupported traffic-profile catalog schema.")
        if self.kind != "fixed-scientific-traffic-candidates":
            raise ValueError("Unexpected traffic-profile catalog kind.")
        ids = [profile.id for profile in self.profiles]
        classes = [profile.reference_class for profile in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("Traffic profile IDs must be unique.")
        if len(classes) != len(set(classes)) or len(classes) != 7:
            raise ValueError("Catalog must map exactly one profile to each class.")
        return self

    def profile(self, profile_id: str) -> TrafficProfile:
        for profile in self.profiles:
            if profile.id == profile_id:
                return profile
        raise KeyError(profile_id)


class TrafficTargetGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoints: list[str] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_private_endpoints(self) -> "TrafficTargetGroup":
        for endpoint in self.endpoints:
            parsed = urlparse(endpoint)
            host = parsed.hostname if parsed.scheme else endpoint
            if parsed.scheme and parsed.scheme != "http":
                raise ValueError("Only private HTTP targets are allowed.")
            if parsed.scheme and (parsed.username or parsed.password or parsed.query):
                raise ValueError("Target URL credentials and queries are forbidden.")
            try:
                address = ipaddress.ip_address(str(host))
            except ValueError as exc:
                raise ValueError("Traffic targets must be literal private IPs.") from exc
            if address.version != 4 or not any(
                address in network for network in PRIVATE_IPV4_NETWORKS
            ):
                raise ValueError("Traffic targets must be RFC1918 IPv4 addresses.")
        return self


class TrafficTargetCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    source_ipv4: str
    groups: dict[str, TrafficTargetGroup]

    @model_validator(mode="after")
    def validate_groups(self) -> "TrafficTargetCatalog":
        if self.schema_version != 1 or not self.groups:
            raise ValueError("Target catalog must use schema 1 and contain groups.")
        try:
            source = ipaddress.ip_address(self.source_ipv4)
        except ValueError as exc:
            raise ValueError("Traffic source must be an IPv4 address.") from exc
        if source.version != 4 or not any(
            source in network for network in PRIVATE_IPV4_NETWORKS
        ):
            raise ValueError("Traffic source must be an RFC1918 IPv4 address.")
        if any(not name or len(name) > 64 for name in self.groups):
            raise ValueError("Target group name is invalid.")
        return self


def load_profile_catalog(path: str | Path) -> TrafficProfileCatalog:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Traffic profile catalog must be a YAML object.")
    return TrafficProfileCatalog.model_validate(value)


def load_target_catalog(path: str | Path) -> TrafficTargetCatalog:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Traffic target catalog must be a YAML object.")
    return TrafficTargetCatalog.model_validate(value)
