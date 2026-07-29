from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class FlowRecord(BaseModel):
    ts: float
    uid: Optional[str] = "-"
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
    local_orig: Optional[str] = "-"
    local_resp: Optional[str] = "-"
    missed_bytes: float = 0.0
    history: str = "-"
    orig_pkts: float = 0.0
    orig_ip_bytes: float = 0.0
    resp_pkts: float = 0.0
    resp_ip_bytes: float = 0.0
    tunnel_parents: Optional[str] = "-"
    label: Optional[str] = "-"
    detailed_label: Optional[str] = Field("-", alias="detailed-label")

class FlowBatchRequest(BaseModel):
    flows: List[FlowRecord]

class FlowPrediction(BaseModel):
    flow_id: str  # Dùng id.orig_h:id.orig_p->id.resp_h:id.resp_p hoặc uid để nhận diện
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    entropy: float

class BatchPredictionResponse(BaseModel):
    model_version: str
    feature_schema_digest: str
    graph_protocol: str
    predictions: List[FlowPrediction]
