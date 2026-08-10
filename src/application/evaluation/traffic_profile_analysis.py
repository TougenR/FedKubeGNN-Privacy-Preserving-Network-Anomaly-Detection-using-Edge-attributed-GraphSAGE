"""Derive immutable traffic-profile references from IoT-23 validation flows."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


class TrafficProfileAnalysisError(RuntimeError):
    """Raised when profile evidence cannot be derived without leakage."""


CATEGORICAL_FIELDS = (
    "proto",
    "service",
    "conn_state",
    "history",
    "id.resp_p",
)
NUMERIC_FIELDS = (
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
)
MISSING_FLAG_FIELDS = {
    "duration": "duration_missing",
    "orig_bytes": "orig_bytes_missing",
    "resp_bytes": "resp_bytes_missing",
}
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_validation_frames(
    replay_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    replay_root = replay_root.resolve()
    manifest_path = replay_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrafficProfileAnalysisError("Cannot read replay manifest.") from exc
    if not isinstance(manifest, dict) or manifest.get(
        "kind"
    ) != "labeled-scientific-evaluation-only":
        raise TrafficProfileAnalysisError("Replay manifest is not labeled evidence.")
    clients = manifest.get("clients")
    if not isinstance(clients, dict) or not clients:
        raise TrafficProfileAnalysisError("Replay manifest has no clients.")
    frames: dict[str, pd.DataFrame] = {}
    for client_id in sorted(clients):
        client = clients[client_id]
        if not isinstance(client, dict) or not isinstance(
            client.get("validation"), dict
        ):
            raise TrafficProfileAnalysisError(
                f"Client '{client_id}' has no validation evidence."
            )
        evidence = client["validation"]
        path = (replay_root / str(evidence.get("path", ""))).resolve()
        if replay_root not in path.parents or not path.is_file():
            raise TrafficProfileAnalysisError(
                f"Validation path escapes replay root for '{client_id}'."
            )
        if _sha256(path) != evidence.get("sha256"):
            raise TrafficProfileAnalysisError(
                f"Validation digest mismatch for '{client_id}'."
            )
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            frame = pd.read_json(handle, orient="records", lines=True)
        if len(frame) != int(evidence.get("rows", -1)):
            raise TrafficProfileAnalysisError(
                f"Validation row count mismatch for '{client_id}'."
            )
        frames[str(client_id)] = frame.sort_values(
            ["ts", "source_edge_index"], kind="stable"
        ).reset_index(drop=True)
    return frames, manifest


def _finite(value: float) -> float | None:
    return round(float(value), 9) if math.isfinite(float(value)) else None


def _quantile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    quantiles = np.quantile(array, QUANTILES)
    return {
        "count": int(len(array)),
        "min": _finite(array.min()),
        "p05": _finite(quantiles[0]),
        "p25": _finite(quantiles[1]),
        "p50": _finite(quantiles[2]),
        "p75": _finite(quantiles[3]),
        "p95": _finite(quantiles[4]),
        "max": _finite(array.max()),
        "mean": _finite(array.mean()),
        "std": _finite(array.std()),
    }


def _distribution(series: pd.Series, *, limit: int = 20) -> dict[str, Any]:
    values = ["<missing>" if pd.isna(value) else str(value) for value in series]
    counts = Counter(values)
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    retained = ordered[:limit]
    probabilities = [count / total for count in counts.values()] if total else []
    entropy = -sum(value * math.log2(value) for value in probabilities if value > 0)
    return {
        "support": total,
        "unique": len(counts),
        "top": [
            {"value": value, "count": count, "fraction": round(count / total, 9)}
            for value, count in retained
        ],
        "other_fraction": round(
            sum(count for _, count in ordered[limit:]) / total, 9
        )
        if total
        else 0.0,
        "entropy_bits": round(entropy, 9),
    }


def _concentration(series: pd.Series) -> float:
    counts = series.astype(str).value_counts()
    return round(float(counts.iloc[0] / counts.sum()), 9) if len(counts) else 0.0


def _rolling_observables(frame: pd.DataFrame) -> dict[str, Any]:
    """Class-conditioned 60-second/50-flow observables by source client."""
    flow_counts: list[int] = []
    unique_destinations: list[int] = []
    unique_ports: list[int] = []
    interarrival_seconds: list[float] = []
    for _, client_frame in frame.groupby("client_id", sort=True):
        ordered = client_frame.sort_values(
            ["ts", "source_edge_index"], kind="stable"
        ).reset_index(drop=True)
        timestamps = ordered["ts"].astype(float).to_numpy()
        destinations = ordered["id.resp_h"].astype(str).to_numpy()
        ports = ordered["id.resp_p"].astype(str).to_numpy()
        if len(timestamps) > 1:
            interarrival_seconds.extend(
                float(value) for value in np.diff(timestamps) if value >= 0
            )
        start = 0
        for end, timestamp in enumerate(timestamps):
            while start < end and timestamps[start] < timestamp - 60.0:
                start += 1
            bounded_start = max(start, end - 49)
            flow_counts.append(end - bounded_start + 1)
            unique_destinations.append(
                len(set(destinations[bounded_start : end + 1]))
            )
            unique_ports.append(len(set(ports[bounded_start : end + 1])))
    return {
        "protocol": {
            "duration_seconds": 60,
            "max_flows": 50,
            "class_conditioned": True,
        },
        "flow_count": _quantile_summary(flow_counts),
        "unique_destinations": _quantile_summary(unique_destinations),
        "unique_destination_ports": _quantile_summary(unique_ports),
        "interarrival_seconds": _quantile_summary(interarrival_seconds),
    }


def _candidate_reference(profile: Mapping[str, Any]) -> dict[str, Any]:
    categorical = profile["categorical"]
    rolling = profile["rolling_60s"]
    numeric = profile["numeric"]

    def top_values(field: str, cumulative: float = 0.9) -> list[str]:
        selected: list[str] = []
        observed = 0.0
        for item in categorical[field]["top"]:
            selected.append(str(item["value"]))
            observed += float(item["fraction"])
            if observed >= cumulative:
                break
        return selected

    def middle_range(summary: Mapping[str, Any]) -> list[float | None]:
        return [summary["p25"], summary["p75"]]

    return {
        "status": "reference-only-not-yet-zeek-validated",
        "timing_status": "not-authoritative-validation-is-stratified-and-thinned",
        "dominant_protocols_90pct": top_values("proto"),
        "dominant_services_90pct": top_values("service"),
        "dominant_connection_states_90pct": top_values("conn_state"),
        "dominant_destination_ports_90pct": top_values("id.resp_p"),
        "middle_50pct": {
            "flow_count_per_60s": middle_range(rolling["flow_count"]),
            "unique_destinations_per_60s": middle_range(
                rolling["unique_destinations"]
            ),
            "unique_destination_ports_per_60s": middle_range(
                rolling["unique_destination_ports"]
            ),
            "interarrival_seconds": middle_range(
                rolling["interarrival_seconds"]
            ),
            "duration_seconds": middle_range(numeric["duration"]),
            "origin_packets": middle_range(numeric["orig_pkts"]),
            "response_packets": middle_range(numeric["resp_pkts"]),
            "origin_ip_bytes": middle_range(numeric["orig_ip_bytes"]),
            "response_ip_bytes": middle_range(numeric["resp_ip_bytes"]),
        },
    }


def _profile(frame: pd.DataFrame) -> dict[str, Any]:
    numeric: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        numeric[field] = _quantile_summary(
            pd.to_numeric(frame[field], errors="coerce").to_numpy()
        )
        flag = MISSING_FLAG_FIELDS.get(field)
        if flag is not None:
            missing = pd.to_numeric(frame[flag], errors="coerce").fillna(1)
            numeric[field]["missing_fraction"] = round(float(missing.mean()), 9)
    response_present = pd.to_numeric(frame["resp_pkts"], errors="coerce").fillna(0) > 0
    topology = {
        "unique_origins": int(frame["id.orig_h"].astype(str).nunique()),
        "unique_destinations": int(frame["id.resp_h"].astype(str).nunique()),
        "unique_directed_pairs": int(
            frame[["id.orig_h", "id.resp_h"]].astype(str).drop_duplicates().shape[0]
        ),
        "origin_concentration": _concentration(frame["id.orig_h"]),
        "destination_concentration": _concentration(frame["id.resp_h"]),
        "response_present_fraction": round(float(response_present.mean()), 9),
    }
    profile: dict[str, Any] = {
        "support": int(len(frame)),
        "clients": {
            str(key): int(value)
            for key, value in sorted(
                frame["client_id"].astype(str).value_counts().items()
            )
        },
        "categorical": {
            field: _distribution(frame[field]) for field in CATEGORICAL_FIELDS
        },
        "numeric": numeric,
        "topology": topology,
        "rolling_60s": _rolling_observables(frame),
    }
    profile["candidate_reference"] = _candidate_reference(profile)
    return profile


def analyze_validation_profiles(*, replay_root: Path) -> dict[str, Any]:
    frames, manifest = _load_validation_frames(replay_root)
    if manifest.get("schema_version") != 1:
        raise TrafficProfileAnalysisError("Unsupported replay manifest schema.")
    expected_classes = [str(value) for value in manifest.get("classes", [])]
    if len(expected_classes) != 7 or len(set(expected_classes)) != 7:
        raise TrafficProfileAnalysisError("Reference requires exactly seven classes.")
    combined: list[pd.DataFrame] = []
    for client_id, frame in frames.items():
        if set(frame["evaluation_split"].astype(str)) != {"validation"}:
            raise TrafficProfileAnalysisError("Non-validation rows reached analysis.")
        owned = frame.copy()
        owned["client_id"] = client_id
        combined.append(owned)
    union = pd.concat(combined, ignore_index=True)
    observed = set(union["detailed-label"].astype(str))
    if observed != set(expected_classes):
        raise TrafficProfileAnalysisError(
            f"Reference class mismatch: expected={expected_classes}, observed={sorted(observed)}"
        )
    profiles = {
        class_name: _profile(
            union.loc[union["detailed-label"].astype(str) == class_name].copy()
        )
        for class_name in expected_classes
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "kind": "iot23-validation-traffic-profile-reference",
        "selection_split": "validation",
        "locked_test_read": False,
        "dataset": {
            "source_dataset_id": manifest["source_dataset_id"],
            "source_dataset_digest": manifest["source_dataset_digest"],
            "derived_dataset_id": manifest["derived_dataset_id"],
            "derived_dataset_digest": manifest["derived_dataset_digest"],
            "replay_manifest_sha256": _sha256(replay_root / "manifest.json"),
            "validation_rows": int(len(union)),
        },
        "graph_protocol": (
            "rolling-window-v1:duration=60s:max-flows=50:stride=1:lateness=1s"
        ),
        "classes": expected_classes,
        "feature_contract": {
            "categorical": list(CATEGORICAL_FIELDS),
            "numeric": list(NUMERIC_FIELDS),
            "rolling": [
                "flow_count",
                "unique_destinations",
                "unique_destination_ports",
                "interarrival_seconds",
            ],
            "labels_are_evaluation_only": True,
            "profile_name_never_enters_inference": True,
        },
        "sampling_contract": {
            "row_feature_authority": "exact-digest-verified-validation-sample",
            "rolling_observable_authority": "locked-evaluator-view-only",
            "generator_timing_authoritative": False,
            "reason": (
                "The deterministic stratified split thins each source timeline; "
                "inter-arrival and 60-second density describe the model's validation "
                "view, not the original packet arrival process."
            ),
        },
        "profiles": profiles,
        "limitations": [
            "This is a class-conditioned IoT-23 validation reference, not live traffic.",
            "An executable profile remains non-equivalent until its Zeek output passes the frozen comparison protocol.",
            "The selected profile must never be treated as model ground truth or used for head routing.",
            "Validation inter-arrival and density values must not directly set generator timing; contiguous source evidence is required.",
        ],
    }
    document["reference_digest"] = _canonical_digest(document)
    return document


def write_reference(*, replay_root: Path, output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Profile reference already exists: {output}")
    document = analyze_validation_profiles(replay_root=replay_root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = write_reference(replay_root=args.replay_root, output=args.output)
    print(json.dumps({
        "output": str(args.output),
        "reference_digest": document["reference_digest"],
        "validation_rows": document["dataset"]["validation_rows"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
