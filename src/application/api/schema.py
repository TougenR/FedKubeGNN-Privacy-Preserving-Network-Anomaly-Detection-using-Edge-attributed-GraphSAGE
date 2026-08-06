"""Production request/response schemas; ground-truth labels are forbidden."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProductionFlow(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ts: float
    uid: str = "-"
    id_orig_h: str = Field(alias="id.orig_h")
    id_orig_p: int = Field(alias="id.orig_p")
    id_resp_h: str = Field(alias="id.resp_h")
    id_resp_p: int = Field(alias="id.resp_p")
    proto: str
    service: str = "-"
    duration: float = 0.0
    orig_bytes: float = 0.0
    resp_bytes: float = 0.0
    conn_state: str
    local_orig: str = "-"
    local_resp: str = "-"
    missed_bytes: float = 0.0
    history: str = "-"
    orig_pkts: float = 0.0
    orig_ip_bytes: float = 0.0
    resp_pkts: float = 0.0
    resp_ip_bytes: float = 0.0
    tunnel_parents: str = "-"


class ProductionInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sensor_id: str
    window_id: str
    flows: list[ProductionFlow]


class FlowPrediction(BaseModel):
    flow_id: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    entropy: float


class ProductionInferenceResponse(BaseModel):
    client_id: str
    model_digest: str
    head_digest: str
    schema_digest: str
    graph_protocol: str
    predictions: list[FlowPrediction]
