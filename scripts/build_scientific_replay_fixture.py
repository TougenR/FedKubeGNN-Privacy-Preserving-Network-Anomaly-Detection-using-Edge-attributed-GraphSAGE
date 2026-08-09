"""Build deterministic, pseudonymized validation replay cases for Phase 4."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


CASES = (
    ("benign", "Benign", "Bình thường", "1-1"),
    ("attack", "Attack", "Tấn công chung", "3-1"),
    ("command-control", "C&C", "Điều khiển và chỉ huy (C&C)", "34-1"),
    ("heartbeat", "C&C-HeartBeat", "Nhịp tim C&C", "36-1"),
    ("ddos", "DDoS", "Từ chối dịch vụ phân tán (DDoS)", "34-1"),
    ("okiru", "Okiru", "Mã độc Okiru", "36-1"),
    (
        "portscan",
        "PartOfAHorizontalPortScan",
        "Quét cổng ngang",
        "1-1",
    ),
)
OCCURRENCE = 100
ALLOWED = (
    "ts",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
)
MISSING_NUMERIC = ("duration", "orig_bytes", "resp_bytes")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pseudonymize(rows: list[dict[str, Any]], case_id: str) -> list[dict[str, Any]]:
    identities: dict[str, str] = {}

    def identity(value: str) -> str:
        if value not in identities:
            identities[value] = f"{case_id}-node-{len(identities) + 1:03d}"
        return identities[value]

    flows: list[dict[str, Any]] = []
    for row in rows:
        flow = {name: row[name] for name in ALLOWED if name in row}
        flow["id.orig_h"] = identity(str(row["id.orig_h"]))
        flow["id.resp_h"] = identity(str(row["id.resp_h"]))
        flow["uid"] = f"validation-{case_id}-{int(row['source_edge_index'])}"
        for name in MISSING_NUMERIC:
            if int(row.get(f"{name}_missing", 0)) == 1:
                flow[name] = None
        flows.append(flow)
    return flows


def build(replay_root: Path) -> dict[str, Any]:
    manifest = json.loads((replay_root / "manifest.json").read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for case_id, expected_class, display_name, client_id in CASES:
        document = manifest["clients"][client_id]["validation"]
        path = replay_root / document["path"]
        if sha256(path) != document["sha256"]:
            raise ValueError(f"Replay digest mismatch for client {client_id}.")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        rows.sort(key=lambda row: (float(row["ts"]), int(row["source_edge_index"])))
        matches = [index for index, row in enumerate(rows) if row["detailed-label"] == expected_class]
        if len(matches) < OCCURRENCE:
            raise ValueError(f"Class {expected_class} has fewer than {OCCURRENCE} rows.")
        target_position = matches[OCCURRENCE - 1]
        target = rows[target_position]
        target_ts = float(target["ts"])
        context = [
            row
            for row in rows[: target_position + 1]
            if float(row["ts"]) >= target_ts - 60.0
        ][-50:]
        cases.append(
            {
                "id": case_id,
                "display_name": display_name,
                "sensor_id": f"sensor-{client_id}",
                "client_id": client_id,
                "expected_class": expected_class,
                "selection_occurrence": OCCURRENCE,
                "source_edge_index": int(target["source_edge_index"]),
                "target_index": len(context) - 1,
                "flows": pseudonymize(context, case_id),
            }
        )
    return {
        "schema_version": 1,
        "kind": "validation-only-scientific-replay",
        "selection_split": "validation",
        "selection_rule": "mẫu thứ 100 cố định của lớp; lấy tối đa 50 flow trong 60 giây trước đó",
        "dataset_digest": manifest["derived_dataset_digest"],
        "graph_protocol": "rolling-window-v1:duration=60s:max-flows=50:stride=1:lateness=1s",
        "disclaimer": (
            "Replay có nhãn chỉ dùng tập validation. Nhãn kỳ vọng nằm trong bộ đánh giá "
            "và không bao giờ được gửi vào request inference production. Đây không phải "
            "lưu lượng trực tiếp."
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build(args.replay_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
