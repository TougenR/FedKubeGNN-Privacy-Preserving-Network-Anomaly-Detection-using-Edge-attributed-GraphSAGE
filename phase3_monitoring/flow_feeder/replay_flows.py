"""Replay labeled IoT-23 flows into the Phase 3 inference service."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests


_THIS_DIR = Path(__file__).resolve().parent
SAMPLE_FILE = Path(
    os.environ.get(
        "SAMPLE_FILE",
        _THIS_DIR / "sample_data" / "sample_conn.log.labeled",
    )
)
INFERENCE_URL = os.environ.get(
    "INFERENCE_SERVICE_URL",
    "http://localhost:8000/predict",
)
READINESS_URL = os.environ.get(
    "INFERENCE_READINESS_URL",
    INFERENCE_URL.removesuffix("/predict") + "/health/ready",
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50"))
INTERVAL_SEC = float(os.environ.get("INTERVAL_SEC", "2.0"))


def load_sample_dataframe(path: Path = SAMPLE_FILE) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Sample file not found: {path}. Run prepare_sample.py first."
        )

    canonical_map = {
        "det_label": "detailed-label",
        "detailed_label": "detailed-label",
        "label_val": "label",
    }
    field_names = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#fields"):
                field_names = [
                    canonical_map.get(field, field)
                    for field in line.rstrip("\n").split("\t")[1:]
                ]
                break
    if not field_names:
        raise ValueError(f"Could not find #fields in {path}.")

    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=field_names,
        na_values=[],
        keep_default_na=False,
        dtype=str,
    )


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def row_to_flow(row: pd.Series) -> dict:
    return {
        "ts": _as_float(row.get("ts", 0.0)),
        "uid": row.get("uid", "-"),
        "id.orig_h": row.get("id.orig_h", "-"),
        "id.orig_p": _as_int(row.get("id.orig_p", 0)),
        "id.resp_h": row.get("id.resp_h", "-"),
        "id.resp_p": _as_int(row.get("id.resp_p", 0)),
        "proto": row.get("proto", "-"),
        "service": row.get("service", "-"),
        "duration": _as_float(row.get("duration", 0.0)),
        "orig_bytes": _as_float(row.get("orig_bytes", 0.0)),
        "resp_bytes": _as_float(row.get("resp_bytes", 0.0)),
        "conn_state": row.get("conn_state", "-"),
        "local_orig": row.get("local_orig", "-"),
        "local_resp": row.get("local_resp", "-"),
        "missed_bytes": _as_float(row.get("missed_bytes", 0.0)),
        "history": row.get("history", "-"),
        "orig_pkts": _as_float(row.get("orig_pkts", 0.0)),
        "orig_ip_bytes": _as_float(row.get("orig_ip_bytes", 0.0)),
        "resp_pkts": _as_float(row.get("resp_pkts", 0.0)),
        "resp_ip_bytes": _as_float(row.get("resp_ip_bytes", 0.0)),
        "tunnel_parents": row.get("tunnel_parents", "-"),
        "label": row.get("label", "-"),
        "detailed-label": row.get("detailed-label", "-"),
    }


def wait_until_ready(max_retries: int = 30) -> None:
    print(f"Waiting for Inference Service readiness at {READINESS_URL}...")
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(READINESS_URL, timeout=2)
            if response.status_code == 200:
                metadata = response.json()
                print(
                    "Inference Service is READY "
                    f"(model={metadata.get('model_version')}, "
                    f"feature_dim={metadata.get('feature_dim')})."
                )
                return
            print(
                f"Not ready ({attempt}/{max_retries}): "
                f"HTTP {response.status_code} {response.text}"
            )
        except (requests.ConnectionError, requests.Timeout) as error:
            print(f"Waiting... ({attempt}/{max_retries}): {error}")
        time.sleep(2)
    raise RuntimeError("Inference Service did not become ready in time.")


def replay() -> None:
    dataframe = load_sample_dataframe()
    wait_until_ready()
    print(
        f"Loaded {len(dataframe)} flows. Sending batches of size {BATCH_SIZE}..."
    )

    for offset in range(0, len(dataframe), BATCH_SIZE):
        batch = dataframe.iloc[offset : offset + BATCH_SIZE]
        flows = [row_to_flow(row) for _, row in batch.iterrows()]
        batch_id = offset // BATCH_SIZE + 1
        print(f"\n--- Sending Batch {batch_id} ({len(flows)} flows) ---")
        started = time.perf_counter()
        try:
            response = requests.post(
                INFERENCE_URL,
                json={"flows": flows},
                timeout=10,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            if response.status_code != 200:
                print(
                    f"[Error] HTTP {response.status_code} | {response.text}"
                )
                continue
            payload = response.json()
            predictions = payload.get("predictions", [])
            print(
                f"[Success] Latency: {latency_ms:.1f} ms | "
                f"Predictions: {len(predictions)} | "
                f"Model: {payload.get('model_version')}"
            )
            for prediction in predictions[:3]:
                print(
                    f"  Flow {prediction['flow_id']} -> "
                    f"Label: {prediction['predicted_label']} "
                    f"(Conf: {prediction['confidence']:.2f}, "
                    f"Entropy: {prediction['entropy']:.3f})"
                )
            if len(predictions) > 3:
                print("  ...")
        except requests.RequestException as error:
            print(f"[Connection Error] {error}")

        if offset + BATCH_SIZE < len(dataframe):
            time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    try:
        replay()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"[Fatal] {error}", file=sys.stderr)
        raise SystemExit(1) from error
