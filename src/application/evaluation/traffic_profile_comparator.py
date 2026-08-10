"""Compare label-free Zeek candidates with the frozen IoT-23 validation view."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.application.collection.zeek_reader import parse_zeek_json
from src.application.collection.zeek_shipper import production_flow_from_zeek
from src.application.evaluation.traffic_profile_analysis import (
    _canonical_digest,
    _load_validation_frames,
)
from src.application.traffic_agent.catalog import load_profile_catalog


class TrafficProfileComparisonError(RuntimeError):
    """Raised when candidate evidence violates the frozen comparison contract."""


CATEGORICAL_FIELDS = ("proto", "service", "conn_state", "history", "id.resp_p")
NUMERIC_FIELDS = (
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
)
BOOTSTRAP_ITERATIONS = 2000
ENVELOPE_QUANTILES = (0.025, 0.975)


def _categorical_distribution(frame: pd.DataFrame, field: str) -> dict[str, float]:
    values = frame[field].map(
        lambda value: "<missing>" if pd.isna(value) else str(value)
    )
    counts = values.value_counts()
    total = float(counts.sum())
    return {str(key): float(value / total) for key, value in counts.items()}


def _jensen_shannon(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    vocabulary = sorted(set(left) | set(right))
    divergence = 0.0
    for value in vocabulary:
        first = float(left.get(value, 0.0))
        second = float(right.get(value, 0.0))
        middle = (first + second) / 2
        if first > 0:
            divergence += 0.5 * first * math.log2(first / middle)
        if second > 0:
            divergence += 0.5 * second * math.log2(second / middle)
    return float(divergence)


def _candidate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for field in NUMERIC_FIELDS:
        values = pd.to_numeric(frame[field], errors="coerce")
        metrics[f"{field}.missing_fraction"] = float(values.isna().mean())
        finite = values.dropna()
        if len(finite):
            metrics[f"{field}.median"] = float(finite.median())
    response = pd.to_numeric(frame["resp_pkts"], errors="coerce").fillna(0)
    metrics["response_present_fraction"] = float((response > 0).mean())
    metrics["unique_destinations"] = float(frame["id.resp_h"].astype(str).nunique())
    return metrics


def _seed(reference_digest: str, profile_id: str) -> int:
    value = hashlib.sha256(f"{reference_digest}:{profile_id}".encode()).digest()
    return int.from_bytes(value[:8], "big", signed=False)


def compare_candidate_frame(
    *,
    candidate: pd.DataFrame,
    references: Mapping[str, pd.DataFrame],
    expected_class: str,
    profile_id: str,
    reference_digest: str,
    scientific_status: str,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Apply the frozen validation-only bootstrap and nearest-class protocol."""
    if not 2 <= len(candidate) <= 50:
        raise TrafficProfileComparisonError("Candidate must contain 2-50 Zeek flows.")
    if expected_class not in references or set(references) != {
        "Benign",
        "Attack",
        "C&C",
        "C&C-HeartBeat",
        "DDoS",
        "Okiru",
        "PartOfAHorizontalPortScan",
    }:
        raise TrafficProfileComparisonError(
            "Reference must contain exactly seven classes."
        )
    if bootstrap_iterations < 100:
        raise TrafficProfileComparisonError(
            "At least 100 bootstrap iterations are required."
        )
    for field in (*CATEGORICAL_FIELDS, *NUMERIC_FIELDS, "id.resp_h"):
        if field not in candidate:
            raise TrafficProfileComparisonError(f"Candidate field is missing: {field}")

    expected = references[expected_class].reset_index(drop=True)
    if len(expected) < len(candidate):
        raise TrafficProfileComparisonError(
            "Reference support is smaller than candidate."
        )
    rng = np.random.default_rng(_seed(reference_digest, profile_id))
    expected_distributions = {
        field: _categorical_distribution(expected, field)
        for field in CATEGORICAL_FIELDS
    }
    candidate_distributions = {
        field: _categorical_distribution(candidate, field)
        for field in CATEGORICAL_FIELDS
    }
    categorical_bootstrap = {field: [] for field in CATEGORICAL_FIELDS}
    numeric_bootstrap: dict[str, list[float]] = {}
    for _ in range(bootstrap_iterations):
        indices = rng.integers(0, len(expected), size=len(candidate))
        sample = expected.iloc[indices]
        for field in CATEGORICAL_FIELDS:
            categorical_bootstrap[field].append(
                _jensen_shannon(
                    _categorical_distribution(sample, field),
                    expected_distributions[field],
                )
            )
        for name, value in _candidate_metrics(sample).items():
            numeric_bootstrap.setdefault(name, []).append(value)

    categorical: dict[str, Any] = {}
    categorical_pass = True
    for field in CATEGORICAL_FIELDS:
        observed = _jensen_shannon(
            candidate_distributions[field], expected_distributions[field]
        )
        upper = float(np.quantile(categorical_bootstrap[field], 0.95))
        passed = observed <= upper + 1e-12
        categorical[field] = {
            "jensen_shannon": round(observed, 9),
            "bootstrap_p95": round(upper, 9),
            "pass": passed,
        }
        categorical_pass = categorical_pass and passed

    numeric: dict[str, Any] = {}
    numeric_pass = True
    for name, observed in sorted(_candidate_metrics(candidate).items()):
        bootstrapped = numeric_bootstrap.get(name)
        if not bootstrapped:
            continue
        lower, upper = np.quantile(bootstrapped, ENVELOPE_QUANTILES)
        passed = float(lower) - 1e-12 <= observed <= float(upper) + 1e-12
        numeric[name] = {
            "observed": round(observed, 9),
            "bootstrap_p025": round(float(lower), 9),
            "bootstrap_p975": round(float(upper), 9),
            "pass": passed,
        }
        numeric_pass = numeric_pass and passed

    nearest_scores: dict[str, float] = {}
    for class_name, frame in references.items():
        score = sum(
            _jensen_shannon(
                candidate_distributions[field],
                _categorical_distribution(frame, field),
            )
            for field in CATEGORICAL_FIELDS
        )
        nearest_scores[class_name] = round(float(score), 9)
    minimum = min(nearest_scores.values())
    nearest_classes = sorted(
        name for name, value in nearest_scores.items() if value <= minimum + 1e-9
    )
    nearest_pass = expected_class in nearest_classes
    eligible = scientific_status == "candidate"
    accepted = eligible and categorical_pass and numeric_pass and nearest_pass
    return {
        "schema_version": 1,
        "kind": "iot23-validation-candidate-comparison",
        "selection_split": "validation",
        "locked_test_read": False,
        "profile_id": profile_id,
        "expected_reference_class": expected_class,
        "scientific_status": scientific_status,
        "candidate_flows": int(len(candidate)),
        "reference_digest": reference_digest,
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed_derivation": "sha256(reference_digest:profile_id)[0:8]",
            "categorical_upper_quantile": 0.95,
            "numeric_interval_quantiles": list(ENVELOPE_QUANTILES),
        },
        "categorical": categorical,
        "numeric": numeric,
        "nearest_reference": {
            "scores": nearest_scores,
            "nearest_classes": nearest_classes,
            "pass": nearest_pass,
        },
        "eligible": eligible,
        "accepted": accepted,
        "result": (
            "accepted-reference-envelope"
            if accepted
            else "control-only"
            if scientific_status == "control-not-class-equivalent"
            else "rejected-reference-envelope"
        ),
        "claim_boundary": (
            "Acceptance concerns the frozen model validation view only; it does "
            "not establish natural malware timing or provide inference ground truth."
        ),
    }


def _load_candidate(path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(production_flow_from_zeek(parse_zeek_json(line)))
            except (KeyError, TypeError, ValueError) as exc:
                raise TrafficProfileComparisonError(
                    f"Invalid label-free Zeek JSON at line {line_number}."
                ) from exc
    return pd.DataFrame(records)


def compare_candidate_file(
    *,
    replay_root: Path,
    catalog_path: Path,
    profile_id: str,
    candidate_path: Path,
) -> dict[str, Any]:
    catalog = load_profile_catalog(catalog_path)
    profile = catalog.profile(profile_id)
    frames, manifest = _load_validation_frames(replay_root)
    if manifest.get("source_dataset_digest") != catalog.dataset_digest:
        raise TrafficProfileComparisonError("Catalog/replay dataset digest mismatch.")
    union = pd.concat(
        [frame.assign(client_id=client_id) for client_id, frame in frames.items()],
        ignore_index=True,
    )
    references = {
        class_name: union.loc[union["detailed-label"].astype(str) == class_name].copy()
        for class_name in manifest["classes"]
    }
    document = compare_candidate_frame(
        candidate=_load_candidate(candidate_path),
        references=references,
        expected_class=profile.reference_class,
        profile_id=profile.id,
        reference_digest=catalog.reference_digest,
        scientific_status=profile.scientific_status,
    )
    document["dataset_digest"] = catalog.dataset_digest
    document["comparison_digest"] = _canonical_digest(document)
    return document


def write_comparison(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Comparison report already exists: {output}")
    document = compare_candidate_file(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = write_comparison(
        output=args.output,
        replay_root=args.replay_root,
        catalog_path=args.catalog,
        profile_id=args.profile_id,
        candidate_path=args.candidate,
    )
    print(json.dumps({"output": str(args.output), "result": document["result"]}))


if __name__ == "__main__":
    main()
