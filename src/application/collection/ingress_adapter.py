"""Translate a target-observed HTTP request into a declared PoC flow record."""

from __future__ import annotations

import time
import uuid
from typing import Any


INGRESS_ADAPTER_PROTOCOL = "ingress-adapter-v1"


def observed_http_flow(
    *,
    source_host: str,
    source_port: int,
    target_host: str,
    target_port: int,
    request_bytes: int,
    response_bytes: int,
    started_at: float,
) -> dict[str, Any]:
    """Build a Zeek-shaped approximation; this is not a packet capture."""
    now = time.time()
    return {
        "ts": started_at,
        "uid": f"ingress-{uuid.uuid4().hex}",
        "id.orig_h": source_host,
        "id.orig_p": source_port,
        "id.resp_h": target_host,
        "id.resp_p": target_port,
        "proto": "tcp",
        "service": "http",
        "duration": max(0.0, now - started_at),
        "orig_bytes": max(0, request_bytes),
        "resp_bytes": max(0, response_bytes),
        "conn_state": "SF",
        "local_orig": "-",
        "local_resp": "-",
        "missed_bytes": 0,
        "history": "ShADadFf",
        "orig_pkts": 1,
        "orig_ip_bytes": max(0, request_bytes),
        "resp_pkts": 1,
        "resp_ip_bytes": max(0, response_bytes),
        "tunnel_parents": "-",
    }
