"""Leakage-resistant Phase 1 E-GraphSAGE experiment runner.

The historical runner remains in :mod:`src.run_experiments`.  This module is a
separate, explicit clean path:

* row membership is fixed before learned preprocessing or imbalance handling;
* pooled/per-scenario retain full-graph transductive message passing;
* LOSO excludes held-out rows from fitting, training, and model selection;
* class weighting is fixed before the final test;
* each result is stored as a validated, run-scoped inference bundle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score

from phase3_monitoring.inference_service.model_loader import (
    ModelContractError,
    load_runtime_bundle,
    validate_model_contract,
)
from src.graph_build import build_graph
from src.imbalance import undersample_majority
from src.model import build_model
from src.preprocess import Preprocessor, fit_preprocessor, transform
from src.train import get_device, make_criterion, safe_stratified_split, set_seed


BUNDLE_SCHEMA_VERSION = 1
ROW_ID_COLUMN = "_clean_row_id"
FIXED_LABELS = (
    "Attack",
    "Benign",
    "C&C",
    "C&C-HeartBeat",
    "DDoS",
    "Okiru",
    "Okiru-Attack",
    "PartOfAHorizontalPortScan",
)
CLEAN_IMBALANCE_MODE = "class_weight"
HISTORICAL_OUTPUT = Path("artifacts/phase1_results")


class CleanProtocolError(RuntimeError):
    """Raised when a clean protocol or bundle invariant is violated."""


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fixed_class_to_idx(config: Mapping[str, Any]) -> dict[str, int]:
    """Return the externally fixed taxonomy, never a data-derived vocabulary."""

    section = config.get("phase1_clean", {})
    configured = tuple(str(label) for label in section.get("labels", ()))
    if configured != FIXED_LABELS:
        raise CleanProtocolError(
            "phase1_clean.labels must exactly match the fixed eight-label "
            f"taxonomy {list(FIXED_LABELS)}; got {list(configured)}."
        )
    return {label: index for index, label in enumerate(configured)}


def resolve_clean_imbalance_mode(config: Mapping[str, Any]) -> str:
    """Resolve the preselected clean-run mode without consulting any metric."""

    mode = str(config.get("phase1_clean", {}).get("imbalance_mode", ""))
    if mode != CLEAN_IMBALANCE_MODE:
        raise CleanProtocolError(
            "Clean reruns require preselected imbalance_mode='class_weight'; "
            f"got {mode!r}. Historical Phase A selection is not used."
        )
    return mode


def _validate_labels(
    frames: Mapping[str, pd.DataFrame],
    class_to_idx: Mapping[str, int],
) -> None:
    for scenario, frame in frames.items():
        if "detailed-label" not in frame:
            raise CleanProtocolError(
                f"Scenario {scenario!r} is missing detailed-label."
            )
        observed = set(frame["detailed-label"].astype(str).unique())
        unknown = sorted(observed - set(class_to_idx))
        if unknown:
            raise CleanProtocolError(
                f"Scenario {scenario!r} contains labels outside the fixed "
                f"taxonomy: {unknown}."
            )


def with_stable_row_ids(
    frame: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """Attach deterministic scenario/position IDs before any row transformation."""

    output = frame.reset_index(drop=True).copy()
    output[ROW_ID_COLUMN] = [
        hashlib.sha256(
            f"phase1-clean-v1:{scenario}:{position}".encode("utf-8")
        ).hexdigest()
        for position in range(len(output))
    ]
    if output[ROW_ID_COLUMN].duplicated().any():
        raise CleanProtocolError(f"Duplicate stable row IDs in {scenario!r}.")
    return output


def _frames_with_ids(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        scenario: with_stable_row_ids(frame, scenario)
        for scenario, frame in sorted(frames.items())
    }


def _digest_ids(row_ids: Iterable[str]) -> str:
    return _canonical_digest(sorted(str(row_id) for row_id in row_ids))


@dataclass(frozen=True)
class SplitPlan:
    scenario: str
    seed: int
    protocol: str
    all_ids: tuple[str, ...]
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def assert_valid(self, *, require_train: bool = True) -> None:
        all_set = set(self.all_ids)
        train_set = set(self.train_ids)
        val_set = set(self.val_ids)
        test_set = set(self.test_ids)
        if len(all_set) != len(self.all_ids):
            raise CleanProtocolError(
                f"{self.scenario}: all_ids contains duplicates."
            )
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise CleanProtocolError(
                f"{self.scenario}: train/validation/test memberships overlap."
            )
        if train_set | val_set | test_set != all_set:
            raise CleanProtocolError(
                f"{self.scenario}: split memberships do not cover all rows."
            )
        if require_train and not train_set:
            raise CleanProtocolError(f"{self.scenario}: empty training split.")

    def manifest_record(self) -> dict[str, Any]:
        self.assert_valid()
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "protocol": self.protocol,
            "counts": {
                "all": len(self.all_ids),
                "train": len(self.train_ids),
                "validation": len(self.val_ids),
                "test": len(self.test_ids),
            },
            "index_digest": {
                "all": _digest_ids(self.all_ids),
                "train": _digest_ids(self.train_ids),
                "validation": _digest_ids(self.val_ids),
                "test": _digest_ids(self.test_ids),
            },
            "disjointness_verified": True,
            "coverage_verified": True,
        }


def _label_tensor(
    frame: pd.DataFrame,
    class_to_idx: Mapping[str, int],
) -> torch.Tensor:
    labels = frame["detailed-label"].astype(str).map(class_to_idx)
    if labels.isna().any():
        raise CleanProtocolError("Cannot encode a label outside fixed taxonomy.")
    return torch.tensor(labels.to_numpy(dtype=np.int64), dtype=torch.long)


def make_transductive_split_plans(
    frames_with_ids: Mapping[str, pd.DataFrame],
    class_to_idx: Mapping[str, int],
    *,
    seed: int,
    protocol: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, SplitPlan]:
    """Create raw-row memberships before fit/transform/imbalance."""

    from src.train import split_edge_masks

    plans: dict[str, SplitPlan] = {}
    for scenario, frame in sorted(frames_with_ids.items()):
        train_mask, val_mask, test_mask = split_edge_masks(
            _label_tensor(frame, class_to_idx),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        ids = frame[ROW_ID_COLUMN].astype(str).to_numpy()
        plan = SplitPlan(
            scenario=scenario,
            seed=seed,
            protocol=protocol,
            all_ids=tuple(ids.tolist()),
            train_ids=tuple(ids[train_mask.numpy()].tolist()),
            val_ids=tuple(ids[val_mask.numpy()].tolist()),
            test_ids=tuple(ids[test_mask.numpy()].tolist()),
        )
        plan.assert_valid()
        plans[scenario] = plan
    return plans


def _train_val_plan(
    frame: pd.DataFrame,
    class_to_idx: Mapping[str, int],
    *,
    scenario: str,
    seed: int,
    val_ratio: float,
) -> SplitPlan:
    labels = _label_tensor(frame, class_to_idx).numpy()
    indices = np.arange(len(frame))
    unique, counts = np.unique(labels, return_counts=True)
    singleton_classes = unique[counts == 1]
    singleton_indices = indices[np.isin(labels, singleton_classes)]
    pool = indices[~np.isin(indices, singleton_indices)]
    if len(pool) < 2:
        raise CleanProtocolError(
            f"{scenario}: not enough non-singleton rows for train/validation."
        )
    train_pool, val_indices = safe_stratified_split(
        pool,
        labels[pool],
        test_size=val_ratio,
        seed=seed,
        context=f"phase1_clean LOSO {scenario}",
        force_into_first=singleton_indices,
    )
    train_indices = np.unique(
        np.concatenate([train_pool, singleton_indices])
    )
    ids = frame[ROW_ID_COLUMN].astype(str).to_numpy()
    plan = SplitPlan(
        scenario=scenario,
        seed=seed,
        protocol="loso",
        all_ids=tuple(ids.tolist()),
        train_ids=tuple(ids[train_indices].tolist()),
        val_ids=tuple(ids[val_indices].tolist()),
        test_ids=(),
    )
    plan.assert_valid()
    return plan


def _select_ids(frame: pd.DataFrame, row_ids: Sequence[str]) -> pd.DataFrame:
    wanted = set(str(row_id) for row_id in row_ids)
    selected = frame[frame[ROW_ID_COLUMN].astype(str).isin(wanted)].copy()
    found = set(selected[ROW_ID_COLUMN].astype(str))
    if found != wanted:
        raise CleanProtocolError(
            f"Stable row membership mismatch: missing={sorted(wanted - found)[:3]}."
        )
    return selected


def _fit_from_train_rows(
    frames: Mapping[str, pd.DataFrame],
    plans: Mapping[str, SplitPlan],
) -> Preprocessor:
    train_rows = [
        _select_ids(frames[scenario], plans[scenario].train_ids)
        for scenario in sorted(plans)
    ]
    return fit_preprocessor(
        pd.concat(train_rows, axis=0, ignore_index=True).drop(
            columns=[ROW_ID_COLUMN]
        )
    )


def fixed_class_weights(
    labels: Sequence[str],
    class_to_idx: Mapping[str, int],
) -> tuple[torch.Tensor, dict[str, int]]:
    """Balanced weights over supported train classes; zero for support=0."""

    support = {label: 0 for label in class_to_idx}
    for label in labels:
        label_string = str(label)
        if label_string not in support:
            raise CleanProtocolError(
                f"Label {label_string!r} is outside fixed taxonomy."
            )
        support[label_string] += 1
    total = sum(support.values())
    supported_classes = sum(count > 0 for count in support.values())
    if total == 0 or supported_classes == 0:
        raise CleanProtocolError("Cannot compute weights from empty train labels.")
    weights = torch.zeros(len(class_to_idx), dtype=torch.float32)
    for label, index in class_to_idx.items():
        count = support[label]
        if count > 0:
            weights[index] = float(total / (supported_classes * count))
    if not torch.isfinite(weights).all():
        raise CleanProtocolError("Class weights contain NaN or infinity.")
    return weights, support


def _transform_with_ids(
    frame: pd.DataFrame,
    preprocessor: Preprocessor,
) -> pd.DataFrame:
    transformed = transform(
        frame.drop(columns=[ROW_ID_COLUMN]),
        preprocessor,
    )
    transformed.insert(
        0,
        ROW_ID_COLUMN,
        frame[ROW_ID_COLUMN].astype(str).to_numpy(),
    )
    return transformed


def _undersample_training_membership(
    transformed: pd.DataFrame,
    plan: SplitPlan,
    *,
    seed: int,
) -> tuple[pd.DataFrame, SplitPlan]:
    """Drop only training rows; preserve validation/test IDs exactly."""

    train_frame = _select_ids(transformed, plan.train_ids)
    sampled_train = undersample_majority(
        train_frame,
        strategy="to_second_largest",
        random_state=seed,
        verbose=False,
    )
    retained_train = tuple(sampled_train[ROW_ID_COLUMN].astype(str))
    retained_set = set(retained_train)
    original_train = set(plan.train_ids)
    keep_mask = (
        ~transformed[ROW_ID_COLUMN].astype(str).isin(original_train)
        | transformed[ROW_ID_COLUMN].astype(str).isin(retained_set)
    )
    output = transformed.loc[keep_mask].copy()
    output_ids = set(output[ROW_ID_COLUMN].astype(str))
    updated = replace(
        plan,
        all_ids=tuple(output[ROW_ID_COLUMN].astype(str)),
        train_ids=tuple(
            row_id for row_id in plan.train_ids if row_id in retained_set
        ),
    )
    if not set(plan.val_ids).issubset(output_ids):
        raise CleanProtocolError("Undersampling changed validation membership.")
    if not set(plan.test_ids).issubset(output_ids):
        raise CleanProtocolError("Undersampling changed test membership.")
    updated.assert_valid()
    return output, updated


def _attach_masks_from_ids(
    graph: Any,
    row_ids: Sequence[str],
    plan: SplitPlan,
) -> None:
    ordered = np.asarray([str(row_id) for row_id in row_ids], dtype=object)
    graph.train_mask = torch.from_numpy(np.isin(ordered, plan.train_ids))
    graph.val_mask = torch.from_numpy(np.isin(ordered, plan.val_ids))
    graph.test_mask = torch.from_numpy(np.isin(ordered, plan.test_ids))
    graph.clean_row_ids = tuple(ordered.tolist())
    if torch.any(graph.train_mask & graph.val_mask):
        raise CleanProtocolError("train/validation graph masks overlap.")
    if torch.any(graph.train_mask & graph.test_mask):
        raise CleanProtocolError("train/test graph masks overlap.")
    if torch.any(graph.val_mask & graph.test_mask):
        raise CleanProtocolError("validation/test graph masks overlap.")
    if not torch.all(graph.train_mask | graph.val_mask | graph.test_mask):
        raise CleanProtocolError("Graph masks do not cover all retained rows.")


def _support_from_plans(
    frames: Mapping[str, pd.DataFrame],
    plans: Mapping[str, SplitPlan],
    class_to_idx: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    support = {
        split: {label: 0 for label in class_to_idx}
        for split in ("train", "validation", "test")
    }
    for scenario, plan in plans.items():
        frame = frames[scenario].set_index(ROW_ID_COLUMN)
        for split_name, ids in (
            ("train", plan.train_ids),
            ("validation", plan.val_ids),
            ("test", plan.test_ids),
        ):
            for label, count in (
                frame.loc[list(ids), "detailed-label"]
                .astype(str)
                .value_counts()
                .items()
                if ids
                else ()
            ):
                support[split_name][str(label)] += int(count)
    return support


@dataclass
class PreparedCleanRun:
    protocol: str
    scenarios: tuple[str, ...]
    graphs: dict[str, Any]
    frames: dict[str, pd.DataFrame]
    split_plans: dict[str, SplitPlan]
    preprocessor: Preprocessor
    class_to_idx: dict[str, int]
    class_weights: torch.Tensor
    class_support: dict[str, dict[str, int]]
    held_out: str | None = None
    held_out_frame: pd.DataFrame | None = None


def prepare_transductive_clean(
    clean_frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    *,
    seed: int,
    protocol: str = "pooled",
    split_plans: Mapping[str, SplitPlan] | None = None,
    imbalance_mode: str = CLEAN_IMBALANCE_MODE,
) -> PreparedCleanRun:
    """Prepare pooled/per-scenario graphs using raw split-before-fit."""

    if protocol not in {"pooled", "per_scenario"}:
        raise CleanProtocolError(f"Unsupported transductive protocol {protocol}.")
    class_to_idx = fixed_class_to_idx(config)
    frames = _frames_with_ids(clean_frames)
    _validate_labels(frames, class_to_idx)
    training = config.get("training", {})
    plans = dict(
        split_plans
        or make_transductive_split_plans(
            frames,
            class_to_idx,
            seed=seed,
            protocol=protocol,
            train_ratio=float(training.get("train_ratio", 0.70)),
            val_ratio=float(training.get("val_ratio", 0.10)),
            test_ratio=float(training.get("test_ratio", 0.20)),
        )
    )
    for scenario, plan in plans.items():
        if scenario not in frames:
            raise CleanProtocolError(f"Split plan has unknown scenario {scenario}.")
        plan.assert_valid()
    preprocessor = _fit_from_train_rows(frames, plans)

    train_labels = [
        str(label)
        for scenario, plan in plans.items()
        for label in _select_ids(
            frames[scenario], plan.train_ids
        )["detailed-label"]
    ]
    weights, _ = fixed_class_weights(train_labels, class_to_idx)
    graphs: dict[str, Any] = {}
    final_plans: dict[str, SplitPlan] = {}
    for scenario, frame in sorted(frames.items()):
        transformed = _transform_with_ids(frame, preprocessor)
        plan = plans[scenario]
        if imbalance_mode == "undersample":
            transformed, plan = _undersample_training_membership(
                transformed,
                plan,
                seed=seed,
            )
        elif imbalance_mode not in {"none", CLEAN_IMBALANCE_MODE}:
            raise CleanProtocolError(
                f"Unsupported clean imbalance mode {imbalance_mode!r}."
            )
        graph = build_graph(
            transformed,
            class_to_idx=class_to_idx,
            feature_columns=preprocessor.feature_columns,
        )
        _attach_masks_from_ids(
            graph,
            transformed[ROW_ID_COLUMN].astype(str),
            plan,
        )
        graphs[scenario] = graph
        final_plans[scenario] = plan

    return PreparedCleanRun(
        protocol=protocol,
        scenarios=tuple(sorted(frames)),
        graphs=graphs,
        frames=frames,
        split_plans=final_plans,
        preprocessor=preprocessor,
        class_to_idx=class_to_idx,
        class_weights=weights,
        class_support=_support_from_plans(frames, final_plans, class_to_idx),
    )


def prepare_loso_clean(
    clean_frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    *,
    held_out: str,
    seed: int,
    val_ratio: float | None = None,
    imbalance_mode: str = CLEAN_IMBALANCE_MODE,
) -> PreparedCleanRun:
    """Prepare one LOSO fold without fitting or transforming held-out values."""

    if held_out not in clean_frames:
        raise CleanProtocolError(f"Unknown held-out scenario {held_out!r}.")
    class_to_idx = fixed_class_to_idx(config)
    frames = _frames_with_ids(clean_frames)
    _validate_labels(frames, class_to_idx)
    ratio = float(
        val_ratio
        if val_ratio is not None
        else config.get("training", {}).get("val_ratio", 0.10)
    )
    train_scenarios = tuple(
        scenario for scenario in sorted(frames) if scenario != held_out
    )
    if not train_scenarios:
        raise CleanProtocolError("LOSO requires at least two scenarios.")
    plans = {
        scenario: _train_val_plan(
            frames[scenario],
            class_to_idx,
            scenario=scenario,
            seed=seed,
            val_ratio=ratio,
        )
        for scenario in train_scenarios
    }
    held_ids = tuple(frames[held_out][ROW_ID_COLUMN].astype(str))
    held_plan = SplitPlan(
        scenario=held_out,
        seed=seed,
        protocol="loso",
        all_ids=held_ids,
        train_ids=(),
        val_ids=(),
        test_ids=held_ids,
    )
    # A LOSO held-out scenario is entirely test membership by definition.
    held_plan.assert_valid(require_train=False)

    preprocessor = _fit_from_train_rows(
        {scenario: frames[scenario] for scenario in train_scenarios},
        plans,
    )
    train_labels = [
        str(label)
        for scenario, plan in plans.items()
        for label in _select_ids(
            frames[scenario], plan.train_ids
        )["detailed-label"]
    ]
    weights, _ = fixed_class_weights(train_labels, class_to_idx)

    graphs: dict[str, Any] = {}
    final_plans: dict[str, SplitPlan] = {}
    for scenario in train_scenarios:
        transformed = _transform_with_ids(frames[scenario], preprocessor)
        plan = plans[scenario]
        if imbalance_mode == "undersample":
            transformed, plan = _undersample_training_membership(
                transformed,
                plan,
                seed=seed,
            )
        elif imbalance_mode not in {"none", CLEAN_IMBALANCE_MODE}:
            raise CleanProtocolError(
                f"Unsupported clean imbalance mode {imbalance_mode!r}."
            )
        graph = build_graph(
            transformed,
            class_to_idx=class_to_idx,
            feature_columns=preprocessor.feature_columns,
        )
        _attach_masks_from_ids(
            graph,
            transformed[ROW_ID_COLUMN].astype(str),
            plan,
        )
        graphs[scenario] = graph
        final_plans[scenario] = plan

    all_plans = dict(final_plans)
    all_plans[held_out] = held_plan
    support = _support_from_plans(frames, all_plans, class_to_idx)
    return PreparedCleanRun(
        protocol="loso",
        scenarios=tuple(sorted(frames)),
        graphs=graphs,
        frames=frames,
        split_plans=all_plans,
        preprocessor=preprocessor,
        class_to_idx=class_to_idx,
        class_weights=weights,
        class_support=support,
        held_out=held_out,
        held_out_frame=frames[held_out],
    )


def build_loso_held_out_graph(prepared: PreparedCleanRun) -> Any:
    """Transform/build held-out only after model selection is complete."""

    if prepared.protocol != "loso" or prepared.held_out_frame is None:
        raise CleanProtocolError("Prepared run is not a LOSO fold.")
    transformed = _transform_with_ids(
        prepared.held_out_frame,
        prepared.preprocessor,
    )
    graph = build_graph(
        transformed,
        class_to_idx=prepared.class_to_idx,
        feature_columns=prepared.preprocessor.feature_columns,
    )
    plan = prepared.split_plans[str(prepared.held_out)]
    graph.train_mask = torch.zeros(len(transformed), dtype=torch.bool)
    graph.val_mask = torch.zeros(len(transformed), dtype=torch.bool)
    graph.test_mask = torch.ones(len(transformed), dtype=torch.bool)
    graph.clean_row_ids = tuple(transformed[ROW_ID_COLUMN].astype(str))
    if set(graph.clean_row_ids) != set(plan.test_ids):
        raise CleanProtocolError("Held-out graph membership changed at transform.")
    return graph


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_to_idx: Mapping[str, int],
) -> dict[str, Any]:
    labels = list(range(len(class_to_idx)))
    ordered_names = [
        name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]
    per_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "accuracy": float(np.mean(y_true == y_pred)),
        "per_class": {
            name: {
                "f1": float(per_f1[index]),
                "support": int(np.sum(y_true == index)),
            }
            for index, name in enumerate(ordered_names)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).astype(int).tolist(),
        "num_examples": int(len(y_true)),
    }


def _evaluate_graphs_core(
    model: torch.nn.Module,
    graphs: Mapping[str, Any],
    mask_name: str,
    class_to_idx: Mapping[str, int],
    device: torch.device,
    *,
    prediction_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    prediction_rows: list[dict[str, Any]] = []
    ordered_labels = [
        name
        for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]
    model.eval()
    with torch.no_grad():
        for scenario, graph in graphs.items():
            logits = model(graph)
            mask = getattr(graph, mask_name).to(logits.device)
            if int(mask.sum()) == 0:
                continue
            selected_logits = logits[mask].detach().cpu()
            selected_probabilities = torch.softmax(
                selected_logits, dim=-1
            )
            selected_predictions = selected_logits.argmax(dim=-1)
            selected_labels = graph.edge_label[mask].detach().cpu()
            predictions.append(selected_predictions)
            labels.append(selected_labels)
            if prediction_context is not None:
                selected_ids = [
                    row_id
                    for row_id, selected in zip(
                        graph.clean_row_ids,
                        mask.detach().cpu().tolist(),
                    )
                    if selected
                ]
                train_support = prediction_context["train_support"]
                for index, row_id in enumerate(selected_ids):
                    true_index = int(selected_labels[index].item())
                    predicted_index = int(
                        selected_predictions[index].item()
                    )
                    true_label = ordered_labels[true_index]
                    probability = selected_probabilities[index]
                    entropy = float(
                        -torch.sum(
                            probability * torch.log(probability + 1e-12)
                        ).item()
                    )
                    row: dict[str, Any] = {
                        "row_id": str(row_id),
                        "scenario": str(scenario),
                        "split": str(prediction_context["split"]),
                        "protocol": str(prediction_context["protocol"]),
                        "seed": int(prediction_context["seed"]),
                        "true_index": true_index,
                        "true_label": true_label,
                        "predicted_index": predicted_index,
                        "predicted_label": ordered_labels[predicted_index],
                        "confidence": float(probability.max().item()),
                        "entropy": entropy,
                        "true_class_train_support": int(
                            train_support.get(true_label, 0)
                        ),
                        "true_class_absent_from_train": bool(
                            train_support.get(true_label, 0) == 0
                        ),
                    }
                    for label_index, label in enumerate(ordered_labels):
                        row[f"probability::{label}"] = float(
                            probability[label_index].item()
                        )
                        row[f"logit::{label}"] = float(
                            selected_logits[index, label_index].item()
                        )
                    prediction_rows.append(row)
    if not labels:
        raise CleanProtocolError(f"No rows available for {mask_name}.")
    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(predictions).numpy()
    frame = (
        pd.DataFrame(prediction_rows)
        if prediction_context is not None
        else None
    )
    return _metrics(y_true, y_pred, class_to_idx), frame


def _evaluate_graphs(
    model: torch.nn.Module,
    graphs: Mapping[str, Any],
    mask_name: str,
    class_to_idx: Mapping[str, int],
    device: torch.device,
) -> dict[str, Any]:
    metrics, _ = _evaluate_graphs_core(
        model,
        graphs,
        mask_name,
        class_to_idx,
        device,
    )
    return metrics


def _evaluate_graphs_with_predictions(
    model: torch.nn.Module,
    graphs: Mapping[str, Any],
    mask_name: str,
    prepared: PreparedCleanRun,
    device: torch.device,
    *,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics, predictions = _evaluate_graphs_core(
        model,
        graphs,
        mask_name,
        prepared.class_to_idx,
        device,
        prediction_context={
            "seed": seed,
            "protocol": prepared.protocol,
            "split": "test",
            "train_support": prepared.class_support["train"],
        },
    )
    if predictions is None:
        raise CleanProtocolError("Final prediction export was not created.")
    return metrics, predictions


@dataclass
class CleanTrainResult:
    model: torch.nn.Module
    best_epoch: int
    validation_metric: float
    history: list[dict[str, float | int]]
    final_metrics: dict[str, Any]
    final_evaluation_calls: int
    predictions: pd.DataFrame


def train_prepared_clean(
    prepared: PreparedCleanRun,
    config: Mapping[str, Any],
    *,
    seed: int,
    epochs_override: int | None = None,
    loss_observer: Callable[[str, tuple[str, ...]], None] | None = None,
) -> CleanTrainResult:
    """Train with train masks, select on validation, then evaluate final once."""

    set_seed(seed)
    device = get_device()
    cfg = copy.deepcopy(dict(config))
    training = cfg.setdefault("training", {})
    epochs = int(
        epochs_override
        if epochs_override is not None
        else cfg.get("experiments", {}).get(
            "max_epochs", training.get("epochs", 50)
        )
    )
    patience = int(training.get("early_stop_patience", 20))
    graphs = {name: graph.to(device) for name, graph in prepared.graphs.items()}
    model = build_model("egraphsage", next(iter(graphs.values())), cfg).to(device)
    mode = resolve_clean_imbalance_mode(cfg)
    criterion = make_criterion(
        mode,
        weight_tensor=prepared.class_weights,
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    grad_clip = float(training.get("grad_clip", 1.0))

    best_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss_total = 0.0
        for scenario, graph in graphs.items():
            logits = model(graph)
            mask = graph.train_mask.to(logits.device)
            if int(mask.sum()) == 0:
                raise CleanProtocolError(f"{scenario}: empty train mask.")
            labels = graph.edge_label[mask]
            if loss_observer is not None:
                used_ids = tuple(
                    row_id
                    for row_id, selected in zip(
                        graph.clean_row_ids,
                        mask.detach().cpu().tolist(),
                    )
                    if selected
                )
                loss_observer(scenario, used_ids)
            loss = criterion(logits[mask], labels)
            loss.backward()
            loss_total += float(loss.item())
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        validation = _evaluate_graphs(
            model,
            graphs,
            "val_mask",
            prepared.class_to_idx,
            device,
        )
        validation_metric = float(validation["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / len(graphs),
                "validation_macro_f1": validation_metric,
            }
        )
        if validation_metric > best_metric:
            best_metric = validation_metric
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise CleanProtocolError("Training did not produce a best state.")
    model.load_state_dict(
        {name: value.to(device) for name, value in best_state.items()}
    )

    final_evaluation_calls = 0
    if prepared.protocol == "loso":
        held_graph = build_loso_held_out_graph(prepared).to(device)
        final_metrics, predictions = _evaluate_graphs_with_predictions(
            model,
            {str(prepared.held_out): held_graph},
            "test_mask",
            prepared,
            device,
            seed=seed,
        )
        final_evaluation_calls += 1
    else:
        final_metrics, predictions = _evaluate_graphs_with_predictions(
            model,
            graphs,
            "test_mask",
            prepared,
            device,
            seed=seed,
        )
        final_evaluation_calls += 1
    return CleanTrainResult(
        model=model,
        best_epoch=best_epoch,
        validation_metric=float(best_metric),
        history=history,
        final_metrics=final_metrics,
        final_evaluation_calls=final_evaluation_calls,
        predictions=predictions,
    )


def _git_state(repository_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_clean_bundle(
    output_dir: str | os.PathLike[str],
    prepared: PreparedCleanRun,
    result: CleanTrainResult,
    config: Mapping[str, Any],
    *,
    seed: int,
    repository_root: str | os.PathLike[str] = ".",
) -> Path:
    """Write the run-scoped clean bundle and validate it immediately."""

    output = Path(output_dir).resolve()
    historical = (Path(repository_root).resolve() / HISTORICAL_OUTPUT).resolve()
    if output == historical or historical in output.parents:
        raise CleanProtocolError(
            "Clean outputs must not overwrite artifacts/phase1_results."
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Clean bundle directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    feature_columns = list(prepared.preprocessor.feature_columns)
    schema_body = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }
    schema_document = dict(schema_body)
    schema_document["digest"] = _canonical_digest(schema_body)
    labels_body = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "labels": list(FIXED_LABELS),
        "class_to_idx": prepared.class_to_idx,
    }
    labels_document = dict(labels_body)
    labels_document["digest"] = _canonical_digest(labels_body)

    cfg = copy.deepcopy(dict(config))
    checkpoint = {
        "state_dict": {
            name: value.detach().cpu()
            for name, value in result.model.state_dict().items()
        },
        "cfg": cfg,
        "feature_dim": len(feature_columns),
        "feature_columns": feature_columns,
        "num_classes": len(prepared.class_to_idx),
        "class_to_idx": prepared.class_to_idx,
        "imbalance_mode": CLEAN_IMBALANCE_MODE,
        "val_macro_f1": result.validation_metric,
        "history_meta": {
            "best_epoch": result.best_epoch,
            "seed": seed,
            "protocol": prepared.protocol,
            "held_out": prepared.held_out,
        },
        "seed": seed,
        "protocol": prepared.protocol,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "feature_schema_digest": schema_document["digest"],
        "label_schema_digest": labels_document["digest"],
    }
    torch.save(checkpoint, output / "model.pt")
    prepared.preprocessor.save(str(output / "preprocessor.pkl"))
    _json_write(output / "schema.json", schema_document)
    _json_write(output / "labels.json", labels_document)

    split_manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "seed": seed,
        "protocol": prepared.protocol,
        "scenarios": [
            plan.manifest_record()
            if plan.train_ids
            else {
                "scenario": plan.scenario,
                "seed": plan.seed,
                "protocol": plan.protocol,
                "counts": {
                    "all": len(plan.all_ids),
                    "train": 0,
                    "validation": 0,
                    "test": len(plan.test_ids),
                },
                "index_digest": {
                    "all": _digest_ids(plan.all_ids),
                    "train": _digest_ids(()),
                    "validation": _digest_ids(()),
                    "test": _digest_ids(plan.test_ids),
                },
                "disjointness_verified": True,
                "coverage_verified": set(plan.all_ids) == set(plan.test_ids),
            }
            for _, plan in sorted(prepared.split_plans.items())
        ],
    }
    _json_write(output / "split_manifest.json", split_manifest)
    metrics = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "best_epoch": result.best_epoch,
        "validation_macro_f1": result.validation_metric,
        "final": result.final_metrics,
        "history": result.history,
        "final_evaluation_calls": result.final_evaluation_calls,
    }
    _json_write(output / "metrics.json", metrics)
    result.predictions.to_csv(output / "predictions.csv", index=False)

    commit, dirty = _git_state(Path(repository_root).resolve())
    split_counts = {
        split: int(sum(values.values()))
        for split, values in prepared.class_support.items()
    }
    metadata = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "model_name": "egraphsage",
        "protocol": prepared.protocol,
        "imbalance_mode": CLEAN_IMBALANCE_MODE,
        "seed": seed,
        "scenarios": list(prepared.scenarios),
        "held_out": prepared.held_out,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_schema_digest": schema_document["digest"],
        "label_mapping": prepared.class_to_idx,
        "label_schema_digest": labels_document["digest"],
        "row_counts": split_counts,
        "class_support": prepared.class_support,
        "best_epoch": result.best_epoch,
        "validation_metric": result.validation_metric,
        "final_test_metric": result.final_metrics["macro_f1"],
        "git_commit": commit,
        "git_dirty": dirty,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "known_limitations": [
            "Pooled/per-scenario use full-graph transductive message passing.",
            "LOSO validation edge features remain visible in training-domain graphs; validation labels are masked from gradient.",
            "A zero-support class retains an output index and weight 0; this does not imply it was learned.",
            "This closed-set softmax model is not evidence of zero-day detection.",
        ],
    }
    _json_write(output / "metadata.json", metadata)
    validate_clean_bundle(output)
    return output


def validate_clean_bundle(
    bundle_dir: str | os.PathLike[str],
) -> Any:
    """Validate external schema/labels plus the Phase 3 model contract."""

    root = Path(bundle_dir)
    required = {
        "model.pt",
        "preprocessor.pkl",
        "schema.json",
        "labels.json",
        "metadata.json",
        "metrics.json",
        "split_manifest.json",
        "predictions.csv",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise ModelContractError(f"Clean bundle is missing files: {missing}.")
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(root / "predictions.csv")
    schema_body = {
        "bundle_schema_version": schema.get("bundle_schema_version"),
        "feature_count": schema.get("feature_count"),
        "feature_columns": schema.get("feature_columns"),
    }
    labels_body = {
        "bundle_schema_version": labels.get("bundle_schema_version"),
        "labels": labels.get("labels"),
        "class_to_idx": labels.get("class_to_idx"),
    }
    if schema.get("digest") != _canonical_digest(schema_body):
        raise ModelContractError("schema.json digest mismatch.")
    if labels.get("digest") != _canonical_digest(labels_body):
        raise ModelContractError("labels.json digest mismatch.")
    checkpoint = torch.load(
        root / "model.pt", map_location="cpu", weights_only=False
    )
    preprocessor = Preprocessor.load(str(root / "preprocessor.pkl"))
    feature_columns, mapping, _ = validate_model_contract(
        checkpoint, preprocessor
    )
    if list(feature_columns) != schema.get("feature_columns"):
        raise ModelContractError("schema.json feature order does not match model.")
    if int(schema.get("feature_count", -1)) != len(feature_columns):
        raise ModelContractError("schema.json feature count does not match model.")
    if mapping != labels.get("class_to_idx"):
        raise ModelContractError("labels.json mapping does not match model.")
    if tuple(labels.get("labels", ())) != FIXED_LABELS:
        raise ModelContractError("labels.json is not the fixed taxonomy.")
    if checkpoint.get("feature_schema_digest") != schema.get("digest"):
        raise ModelContractError("Checkpoint feature schema digest mismatch.")
    if checkpoint.get("label_schema_digest") != labels.get("digest"):
        raise ModelContractError("Checkpoint label schema digest mismatch.")
    if metadata.get("feature_schema_digest") != schema.get("digest"):
        raise ModelContractError("metadata feature schema digest mismatch.")
    if metadata.get("label_schema_digest") != labels.get("digest"):
        raise ModelContractError("metadata label schema digest mismatch.")
    if metadata.get("label_mapping") != mapping:
        raise ModelContractError("metadata label mapping mismatch.")
    required_prediction_columns = {
        "row_id",
        "scenario",
        "split",
        "protocol",
        "seed",
        "true_label",
        "predicted_label",
        "entropy",
        "true_class_train_support",
        "true_class_absent_from_train",
        *(f"probability::{label}" for label in FIXED_LABELS),
        *(f"logit::{label}" for label in FIXED_LABELS),
    }
    missing_prediction_columns = sorted(
        required_prediction_columns - set(predictions.columns)
    )
    if missing_prediction_columns:
        raise ModelContractError(
            "predictions.csv is missing columns: "
            f"{missing_prediction_columns}."
        )
    if predictions.empty:
        raise ModelContractError("predictions.csv contains no final rows.")
    if set(predictions["true_label"]) - set(FIXED_LABELS):
        raise ModelContractError(
            "predictions.csv contains true labels outside fixed taxonomy."
        )
    if set(predictions["predicted_label"]) - set(FIXED_LABELS):
        raise ModelContractError(
            "predictions.csv contains predictions outside fixed taxonomy."
        )
    if set(predictions["split"].astype(str)) != {"test"}:
        raise ModelContractError(
            "predictions.csv must contain only final test rows."
        )
    if set(predictions["protocol"].astype(str)) != {
        str(metadata.get("protocol"))
    }:
        raise ModelContractError(
            "predictions.csv protocol does not match metadata."
        )
    if set(pd.to_numeric(predictions["seed"], errors="coerce")) != {
        int(metadata.get("seed"))
    }:
        raise ModelContractError(
            "predictions.csv seed does not match metadata."
        )
    if len(predictions) != int(
        metrics.get("final", {}).get("num_examples", -1)
    ):
        raise ModelContractError(
            "predictions.csv row count does not match final metrics."
        )
    probability_values = predictions[
        [f"probability::{label}" for label in FIXED_LABELS]
    ].to_numpy(dtype=float)
    if (
        not np.isfinite(probability_values).all()
        or np.any(probability_values < 0)
        or np.any(probability_values > 1)
        or not np.allclose(
            probability_values.sum(axis=1), 1.0, atol=1e-5
        )
    ):
        raise ModelContractError(
            "predictions.csv probabilities are invalid."
        )
    return load_runtime_bundle(
        device="cpu",
        checkpoint_path=root / "model.pt",
        preprocessor_path=root / "preprocessor.pkl",
    )


def run_prepared_to_bundle(
    prepared: PreparedCleanRun,
    config: Mapping[str, Any],
    *,
    seed: int,
    output_dir: str | os.PathLike[str],
    epochs_override: int | None = None,
) -> dict[str, Any]:
    result = train_prepared_clean(
        prepared,
        config,
        seed=seed,
        epochs_override=epochs_override,
    )
    bundle = write_clean_bundle(
        output_dir,
        prepared,
        result,
        config,
        seed=seed,
    )
    return {
        "bundle": str(bundle),
        "protocol": prepared.protocol,
        "held_out": prepared.held_out,
        "seed": seed,
        "best_epoch": result.best_epoch,
        "validation_macro_f1": result.validation_metric,
        "final_macro_f1": result.final_metrics["macro_f1"],
    }


def make_toy_clean_frames() -> dict[str, pd.DataFrame]:
    """Deterministic cleaned-data fixture large enough for all split paths."""

    frames: dict[str, pd.DataFrame] = {}
    scenario_labels = {
        "toy-a": ("Benign", "Attack", "C&C"),
        "toy-b": ("Benign", "DDoS", "PartOfAHorizontalPortScan"),
        "toy-c": ("Benign", "Okiru", "C&C-HeartBeat"),
    }
    for scenario_index, (scenario, labels) in enumerate(scenario_labels.items()):
        rows: list[dict[str, Any]] = []
        for index in range(72):
            label = labels[index % len(labels)]
            rows.append(
                {
                    "id.orig_h": f"10.{scenario_index}.0.{index % 12 + 1}",
                    "id.resp_h": f"172.16.{scenario_index}.{index % 9 + 1}",
                    "ts": float(index),
                    "id.orig_p": 40000 + index,
                    "id.resp_p": (22, 80, 443)[index % 3],
                    "proto": ("tcp", "udp")[index % 2],
                    "service": ("http", "dns")[index % 2],
                    "conn_state": ("SF", "S0")[index % 2],
                    "history": ("ShAD", "D")[index % 2],
                    "duration": float(index % 7 + 1),
                    "orig_bytes": float(index + 1),
                    "resp_bytes": float(index + 2),
                    "missed_bytes": 0.0,
                    "orig_pkts": float(index % 5 + 1),
                    "orig_ip_bytes": float(index + 20),
                    "resp_pkts": float(index % 4 + 1),
                    "resp_ip_bytes": float(index + 30),
                    "label": "Benign" if label == "Benign" else "Malicious",
                    "detailed-label": label,
                }
            )
        frames[scenario] = pd.DataFrame(rows)
    return frames


def run_toy_smoke(config_path: str = "config.yaml") -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config["training"]["epochs"] = 1
    config["training"]["early_stop_patience"] = 1
    frames = make_toy_clean_frames()
    with tempfile.TemporaryDirectory(prefix="phase1-clean-smoke-") as temporary:
        root = Path(temporary)
        pooled = prepare_transductive_clean(frames, config, seed=42)
        pooled_result = run_prepared_to_bundle(
            pooled,
            config,
            seed=42,
            output_dir=root / "pooled",
            epochs_override=1,
        )
        loso = prepare_loso_clean(
            frames,
            config,
            held_out="toy-c",
            seed=42,
        )
        loso_result = run_prepared_to_bundle(
            loso,
            config,
            seed=42,
            output_dir=root / "loso-toy-c",
            epochs_override=1,
        )
        validate_clean_bundle(root / "pooled")
        validate_clean_bundle(root / "loso-toy-c")
        result = {
            "pooled": pooled_result,
            "loso": loso_result,
            "temporary_bundle_validation": "passed",
        }
    return result


def _scenario_paths(
    config: Mapping[str, Any],
    overrides: Sequence[str] | None,
) -> dict[str, str]:
    if overrides:
        output: dict[str, str] = {}
        for item in overrides:
            if "=" not in item:
                raise CleanProtocolError(
                    f"Scenario override must be name=PATH, got {item!r}."
                )
            name, path = item.split("=", 1)
            output[name] = path
        return output
    return {
        str(entry["name"]): str(entry["path"])
        for entry in config.get("experiments", {}).get("scenarios", ())
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-resistant Phase 1 E-GraphSAGE clean runner."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("pooled", "per_scenario", "loso"),
    )
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--held-out")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--cap-per-class", type=int)
    parser.add_argument("--out-dir")
    parser.add_argument("--toy-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.toy_smoke:
        print(json.dumps(run_toy_smoke(args.config), indent=2, sort_keys=True))
        return 0
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    fixed_class_to_idx(config)
    resolve_clean_imbalance_mode(config)
    seed = int(
        args.seed
        if args.seed is not None
        else config.get("reproducibility", {}).get("seed", 42)
    )
    protocols = tuple(
        args.protocols
        or config.get("phase1_clean", {}).get("protocols", ("pooled", "loso"))
    )
    paths = _scenario_paths(config, args.scenarios)
    from src.multi_scenario import load_all_scenarios

    cap = (
        args.cap_per_class
        if args.cap_per_class is not None
        else config.get("experiments", {}).get("cap_per_class")
    )
    frames = load_all_scenarios(paths, cap_per_class=cap)
    output_root = Path(
        args.out_dir
        or config.get("phase1_clean", {}).get(
            "output_root", "artifacts/phase1_clean"
        )
    )
    if output_root.resolve() == HISTORICAL_OUTPUT.resolve():
        raise CleanProtocolError("Clean output cannot be historical output.")
    summaries: list[dict[str, Any]] = []
    for protocol in protocols:
        if protocol == "pooled":
            prepared = prepare_transductive_clean(
                frames, config, seed=seed, protocol="pooled"
            )
            summaries.append(
                run_prepared_to_bundle(
                    prepared,
                    config,
                    seed=seed,
                    output_dir=output_root / f"pooled-seed-{seed}",
                    epochs_override=args.epochs,
                )
            )
        elif protocol == "per_scenario":
            for scenario in sorted(frames):
                prepared = prepare_transductive_clean(
                    {scenario: frames[scenario]},
                    config,
                    seed=seed,
                    protocol="per_scenario",
                )
                summaries.append(
                    run_prepared_to_bundle(
                        prepared,
                        config,
                        seed=seed,
                        output_dir=output_root
                        / f"per-scenario-{scenario}-seed-{seed}",
                        epochs_override=args.epochs,
                    )
                )
        elif protocol == "loso":
            held_outs = (
                (args.held_out,) if args.held_out else tuple(sorted(frames))
            )
            for held_out in held_outs:
                prepared = prepare_loso_clean(
                    frames,
                    config,
                    held_out=held_out,
                    seed=seed,
                )
                summaries.append(
                    run_prepared_to_bundle(
                        prepared,
                        config,
                        seed=seed,
                        output_dir=output_root
                        / f"loso-held-out-{held_out}-seed-{seed}",
                        epochs_override=args.epochs,
                    )
                )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
