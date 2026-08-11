#!/usr/bin/env python3
"""Generate report-ready Phase 2 and Phase 3 figures from observed artifacts.

The script never invents missing experiment values. Phase 2 figures are emitted
only when completed run summaries exist. Phase 3 figures are generated from the
validation JSON/CSV artifacts that are already present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE2_ROOT = REPO_ROOT / "artifacts" / "phase2" / "runs"
DEFAULT_PHASE3_ROOT = REPO_ROOT / "artifacts" / "phase3_validation"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "report_figures"

STRATEGY_LABELS = {
    "centralized": "Tập trung",
    "centralized_reference": "Tập trung",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedper": "FedPer",
    "class_aware": "Class-aware",
    "class-aware": "Class-aware",
}
METRIC_KEYS = {
    "accuracy": ("accuracy",),
    "macro_f1": ("macro_f1", "macro-f1"),
    "weighted_f1": ("weighted_f1", "weighted-f1"),
    "loss": ("loss",),
}


@dataclass(frozen=True)
class Phase2Run:
    root: Path
    run_id: str
    strategy: str
    seed: int | None
    num_classes: int | None
    summary: Mapping[str, Any]
    test_metrics: Mapping[str, Any]
    rounds: tuple[Mapping[str, Any], ...]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric(metrics: Mapping[str, Any], name: str) -> float | None:
    for key in METRIC_KEYS[name]:
        if key in metrics:
            return _finite_float(metrics[key])
    return None


def _strategy_name(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    raw = str(
        summary.get("strategy")
        or summary.get("kind")
        or manifest.get("strategy")
        or "unknown"
    ).lower()
    return raw.replace(" ", "_")


def _strategy_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").title())


def _seed_from_config(config: Mapping[str, Any]) -> int | None:
    training = config.get("training")
    if isinstance(training, Mapping):
        value = training.get("seed")
        if value is not None:
            return int(value)
    for key in ("seed", "random_seed", "random-seed"):
        if key in config:
            return int(config[key])
    return None


def _sorted_class_names(names: Iterable[str]) -> list[str]:
    def key(name: str) -> tuple[int, int | str]:
        match = re.fullmatch(r"class[_-](\d+)", name)
        return (0, int(match.group(1))) if match else (1, name)

    return sorted({str(name) for name in names}, key=key)


def _per_class_f1(metrics: Mapping[str, Any]) -> dict[str, float]:
    per_class = metrics.get("per_class")
    if isinstance(per_class, Mapping):
        result: dict[str, float] = {}
        for name, values in per_class.items():
            if isinstance(values, Mapping):
                f1 = _finite_float(values.get("f1"))
                if f1 is not None:
                    result[str(name)] = f1
        return result

    result = {}
    for key, value in metrics.items():
        match = re.fullmatch(r"f1-class-(\d+)", str(key))
        f1 = _finite_float(value)
        if match and f1 is not None:
            result[f"class_{match.group(1)}"] = f1
    return result


def _num_classes(summary: Mapping[str, Any], metrics: Mapping[str, Any]) -> int | None:
    for key in ("num_classes", "num-classes"):
        value = metrics.get(key)
        if value is not None:
            return int(value)
    per_class = _per_class_f1(metrics)
    if per_class:
        return len(per_class)
    matrix = summary.get("test_confusion_matrix") or metrics.get("confusion-matrix")
    if isinstance(matrix, list):
        if matrix and isinstance(matrix[0], list):
            return len(matrix)
        root = int(round(math.sqrt(len(matrix))))
        return root if root * root == len(matrix) else None
    return None


def _load_rounds(run_root: Path) -> tuple[Mapping[str, Any], ...]:
    rounds_csv = run_root / "metrics" / "rounds.csv"
    rows: list[dict[str, Any]] = []
    if rounds_csv.is_file():
        with rounds_csv.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    else:
        for path in sorted((run_root / "metrics").glob("validation-round-*.json")):
            rows.append(_read_json(path))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        round_number = row.get("round")
        macro_f1 = _metric(row, "macro_f1")
        if round_number is None or macro_f1 is None:
            continue
        normalized.append(
            {
                "round": int(round_number),
                "macro_f1": macro_f1,
                "accuracy": _metric(row, "accuracy"),
                "weighted_f1": _metric(row, "weighted_f1"),
                "loss": _metric(row, "loss"),
            }
        )
    return tuple(sorted(normalized, key=lambda row: int(row["round"])))


def discover_phase2_runs(root: Path) -> list[Phase2Run]:
    runs: list[Phase2Run] = []
    if not root.exists():
        return runs
    for summary_path in sorted(root.rglob("metrics/summary.json")):
        run_root = summary_path.parent.parent
        manifest_path = run_root / "run.json"
        manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
        if manifest and manifest.get("status") not in (None, "completed"):
            continue
        summary = _read_json(summary_path)
        test_metrics = summary.get("test_metrics", {})
        if not isinstance(test_metrics, Mapping):
            continue
        config_path = run_root / "config.snapshot.json"
        config = _read_json(config_path) if config_path.is_file() else {}
        runs.append(
            Phase2Run(
                root=run_root,
                run_id=str(summary.get("run_id") or manifest.get("run_id") or run_root.name),
                strategy=_strategy_name(summary, manifest),
                seed=_seed_from_config(config),
                num_classes=_num_classes(summary, test_metrics),
                summary=summary,
                test_metrics=test_metrics,
                rounds=_load_rounds(run_root),
            )
        )
    return runs


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(
    figure: plt.Figure, output_root: Path, stem: str, formats: Sequence[str]
) -> list[str]:
    paths: list[str] = []
    for suffix in formats:
        path = output_root / f"{stem}.{suffix}"
        figure.savefig(path)
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def _protocol_suffix(num_classes: int | None) -> str:
    return f"k{num_classes}" if num_classes is not None else "taxonomy-unknown"


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def _plot_phase2_convergence(
    runs: Sequence[Phase2Run], output_root: Path, formats: Sequence[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    grouped: dict[int | None, list[Phase2Run]] = defaultdict(list)
    for run in runs:
        if run.rounds:
            grouped[run.num_classes].append(run)
    for num_classes, protocol_runs in sorted(grouped.items(), key=lambda item: str(item[0])):
        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        for strategy in sorted({run.strategy for run in protocol_runs}):
            selected = [run for run in protocol_runs if run.strategy == strategy]
            values_by_round: dict[int, list[float]] = defaultdict(list)
            for run in selected:
                for row in run.rounds:
                    values_by_round[int(row["round"])].append(float(row["macro_f1"]))
            x = np.asarray(sorted(values_by_round), dtype=int)
            mean = np.asarray([_mean_std(values_by_round[r])[0] for r in x])
            std = np.asarray([_mean_std(values_by_round[r])[1] for r in x])
            axis.plot(x, mean, marker="o", markersize=3, label=_strategy_label(strategy))
            if len(selected) > 1:
                axis.fill_between(x, mean - std, mean + std, alpha=0.16)
        axis.set_xlabel("Vòng truyền thông")
        axis.set_ylabel("Validation Macro-F1")
        axis.set_ylim(0.0, 1.03)
        axis.set_title(
            f"Hội tụ của các phương pháp học liên kết ({_protocol_suffix(num_classes)})"
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        stem = f"fig_p2_validation_macro_f1_by_round_{_protocol_suffix(num_classes)}"
        records.append(
            {
                "name": stem,
                "phase": 2,
                "caption": "Validation Macro-F1 theo vòng truyền thông; dải mờ là ±1 độ lệch chuẩn nếu có nhiều seed.",
                "paths": _save_figure(figure, output_root, stem, formats),
                "sources": [str(run.root.resolve()) for run in protocol_runs],
            }
        )
    return records


def _plot_phase2_test_metrics(
    runs: Sequence[Phase2Run], output_root: Path, formats: Sequence[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    grouped: dict[int | None, list[Phase2Run]] = defaultdict(list)
    for run in runs:
        if any(_metric(run.test_metrics, metric) is not None for metric in ("accuracy", "macro_f1", "weighted_f1")):
            grouped[run.num_classes].append(run)
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("weighted_f1", "Weighted-F1"),
    ]
    for num_classes, protocol_runs in sorted(grouped.items(), key=lambda item: str(item[0])):
        strategies = sorted({run.strategy for run in protocol_runs})
        x = np.arange(len(strategies), dtype=float)
        width = 0.24
        figure, axis = plt.subplots(figsize=(9, 5.2))
        for offset, (metric, label) in enumerate(metric_specs):
            means: list[float] = []
            stds: list[float] = []
            for strategy in strategies:
                values = [
                    value
                    for run in protocol_runs
                    if run.strategy == strategy
                    for value in [_metric(run.test_metrics, metric)]
                    if value is not None
                ]
                mean, std = _mean_std(values) if values else (math.nan, 0.0)
                means.append(mean)
                stds.append(std)
            positions = x + (offset - 1) * width
            bars = axis.bar(positions, means, width, yerr=stds, capsize=3, label=label)
            for bar, value in zip(bars, means):
                if math.isfinite(value):
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(value + 0.018, 1.015),
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        axis.set_xticks(x, [_strategy_label(strategy) for strategy in strategies])
        axis.set_ylabel("Điểm trên tập test")
        axis.set_ylim(0.0, 1.08)
        axis.set_title(f"So sánh chất lượng mô hình Phase 2 ({_protocol_suffix(num_classes)})")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, ncols=3, loc="lower center")
        figure.tight_layout()
        stem = f"fig_p2_test_metrics_{_protocol_suffix(num_classes)}"
        records.append(
            {
                "name": stem,
                "phase": 2,
                "caption": "Accuracy, Macro-F1 và Weighted-F1 trên tập test; thanh lỗi là ±1 độ lệch chuẩn giữa các seed.",
                "paths": _save_figure(figure, output_root, stem, formats),
                "sources": [str(run.root.resolve()) for run in protocol_runs],
            }
        )
    return records


def _plot_phase2_per_class(
    runs: Sequence[Phase2Run], output_root: Path, formats: Sequence[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    grouped: dict[int | None, list[Phase2Run]] = defaultdict(list)
    for run in runs:
        if _per_class_f1(run.test_metrics):
            grouped[run.num_classes].append(run)
    for num_classes, protocol_runs in sorted(grouped.items(), key=lambda item: str(item[0])):
        strategies = sorted({run.strategy for run in protocol_runs})
        class_names = _sorted_class_names(
            name
            for run in protocol_runs
            for name in _per_class_f1(run.test_metrics)
        )
        matrix = np.full((len(strategies), len(class_names)), np.nan)
        for row_index, strategy in enumerate(strategies):
            for column_index, class_name in enumerate(class_names):
                values = [
                    per_class[class_name]
                    for run in protocol_runs
                    if run.strategy == strategy
                    for per_class in [_per_class_f1(run.test_metrics)]
                    if class_name in per_class
                ]
                if values:
                    matrix[row_index, column_index] = float(np.mean(values))
        figure_width = max(8.5, 1.05 * len(class_names))
        figure, axis = plt.subplots(figsize=(figure_width, max(3.6, 0.7 * len(strategies) + 2.4)))
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        axis.set_xticks(np.arange(len(class_names)), class_names, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(strategies)), [_strategy_label(value) for value in strategies])
        axis.set_xlabel("Lớp")
        axis.set_ylabel("Phương pháp")
        axis.set_title(f"F1 theo lớp trên tập test ({_protocol_suffix(num_classes)})")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if math.isfinite(value):
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value < 0.55 else "black",
                        fontsize=8,
                    )
        figure.colorbar(image, ax=axis, label="F1-score", fraction=0.035, pad=0.02)
        figure.tight_layout()
        stem = f"fig_p2_per_class_f1_{_protocol_suffix(num_classes)}"
        records.append(
            {
                "name": stem,
                "phase": 2,
                "caption": "F1 trung bình theo lớp và phương pháp; ô trống là metric không có trong artifact.",
                "paths": _save_figure(figure, output_root, stem, formats),
                "sources": [str(run.root.resolve()) for run in protocol_runs],
            }
        )
    return records


def _plot_phase2_cost(
    runs: Sequence[Phase2Run], output_root: Path, formats: Sequence[str]
) -> list[dict[str, Any]]:
    eligible = [
        run
        for run in runs
        if _finite_float(run.summary.get("total_upload_bytes")) is not None
        or _finite_float(run.summary.get("total_download_bytes")) is not None
    ]
    if not eligible:
        return []
    records: list[dict[str, Any]] = []
    grouped: dict[int | None, list[Phase2Run]] = defaultdict(list)
    for run in eligible:
        grouped[run.num_classes].append(run)
    for num_classes, protocol_runs in sorted(grouped.items(), key=lambda item: str(item[0])):
        strategies = sorted({run.strategy for run in protocol_runs})
        upload: list[float] = []
        download: list[float] = []
        for strategy in strategies:
            selected = [run for run in protocol_runs if run.strategy == strategy]
            upload.append(
                np.mean([float(run.summary.get("total_upload_bytes", 0)) for run in selected]) / 2**20
            )
            download.append(
                np.mean([float(run.summary.get("total_download_bytes", 0)) for run in selected]) / 2**20
            )
        x = np.arange(len(strategies), dtype=float)
        width = 0.34
        figure, axis = plt.subplots(figsize=(8.5, 5.0))
        axis.bar(x - width / 2, upload, width, label="Upload")
        axis.bar(x + width / 2, download, width, label="Download")
        axis.set_xticks(x, [_strategy_label(strategy) for strategy in strategies])
        axis.set_ylabel("Dung lượng trung bình (MiB)")
        axis.set_title(f"Chi phí truyền thông Phase 2 ({_protocol_suffix(num_classes)})")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        stem = f"fig_p2_communication_cost_{_protocol_suffix(num_classes)}"
        records.append(
            {
                "name": stem,
                "phase": 2,
                "caption": "Tổng upload và download trung bình của mỗi phương pháp.",
                "paths": _save_figure(figure, output_root, stem, formats),
                "sources": [str(run.root.resolve()) for run in protocol_runs],
            }
        )
    return records


def generate_phase2_figures(
    phase2_root: Path, output_root: Path, formats: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs = discover_phase2_runs(phase2_root)
    if not runs:
        return [], [
            {
                "phase": 2,
                "status": "skipped",
                "reason": f"Không tìm thấy completed metrics/summary.json dưới {phase2_root}.",
            }
        ]
    generated = [
        *_plot_phase2_convergence(runs, output_root, formats),
        *_plot_phase2_test_metrics(runs, output_root, formats),
        *_plot_phase2_per_class(runs, output_root, formats),
        *_plot_phase2_cost(runs, output_root, formats),
    ]
    skipped: list[dict[str, Any]] = []
    if not any(record["name"].startswith("fig_p2_validation") for record in generated):
        skipped.append({"phase": 2, "status": "skipped", "reason": "Không có metric validation theo round."})
    if not any(record["name"].startswith("fig_p2_per_class") for record in generated):
        skipped.append({"phase": 2, "status": "skipped", "reason": "Không có F1 theo lớp trong test_metrics."})
    if not any(record["name"].startswith("fig_p2_communication") for record in generated):
        skipped.append({"phase": 2, "status": "skipped", "reason": "Không có total_upload_bytes/total_download_bytes."})
    return generated, skipped


def _plot_phase3_latency(
    phase3_root: Path, output_root: Path, formats: Sequence[str]
) -> dict[str, Any] | None:
    path = phase3_root / "latency.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        return None
    endpoints = sorted({str(row["endpoint"]) for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for endpoint in endpoints:
        selected = sorted(
            (row for row in rows if row["endpoint"] == endpoint),
            key=lambda row: int(row["batch_size"]),
        )
        batch = [int(row["batch_size"]) for row in selected]
        axes[0].plot(batch, [float(row["mean_ms"]) for row in selected], marker="o", label=f"{endpoint}: mean")
        axes[0].plot(batch, [float(row["p95_ms"]) for row in selected], marker="s", linestyle="--", label=f"{endpoint}: P95")
        axes[1].plot(batch, [float(row["flows_per_second"]) for row in selected], marker="o", label=endpoint)
    axes[0].set_xlabel("Batch size (flow)")
    axes[0].set_ylabel("Độ trễ HTTP (ms)")
    axes[0].set_title("Độ trễ mean và P95")
    axes[1].set_xlabel("Batch size (flow)")
    axes[1].set_ylabel("Thông lượng (flow/giây)")
    axes[1].set_title("Thông lượng suy luận")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Hiệu năng suy luận Phase 3 trên CPU local", y=1.02)
    figure.tight_layout()
    stem = "fig_p3_latency_and_throughput"
    return {
        "name": stem,
        "phase": 3,
        "caption": "Độ trễ mean/P95 và thông lượng theo batch; benchmark PoC CPU local, không đại diện production.",
        "paths": _save_figure(figure, output_root, stem, formats),
        "sources": [str(path.resolve())],
    }


def _plot_phase3_equivalence(
    phase3_root: Path, output_root: Path, formats: Sequence[str]
) -> dict[str, Any] | None:
    path = phase3_root / "equivalence_local.json"
    if not path.is_file():
        return None
    report = _read_json(path)
    values = [
        _finite_float(report.get("flow_id_agreement")),
        _finite_float(report.get("label_agreement")),
    ]
    if any(value is None for value in values):
        return None
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    bars = axis.bar(["Flow-ID", "Nhãn dự đoán"], values, width=0.55)
    for bar, value in zip(bars, values):
        assert value is not None
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value * 100:.1f}%", ha="center", va="bottom")
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Tỷ lệ trùng khớp")
    axis.set_title(
        f"Tính tương đương offline và FastAPI (n={report.get('compared_predictions', 'N/A')})"
    )
    axis.grid(axis="y", alpha=0.25)
    delta_text = (
        f"max Δ xác suất={report.get('max_probability_delta', 'N/A')}; "
        f"confidence={report.get('max_confidence_delta', 'N/A')}; "
        f"entropy={report.get('max_entropy_delta', 'N/A')}"
    )
    figure.text(0.5, 0.01, delta_text, ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    stem = "fig_p3_offline_fastapi_equivalence"
    return {
        "name": stem,
        "phase": 3,
        "caption": "Mức trùng khớp giữa suy luận trực tiếp và FastAPI trên cùng batch đầu vào.",
        "paths": _save_figure(figure, output_root, stem, formats),
        "sources": [str(path.resolve())],
    }


def _phase3_evidence_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    report = _read_json(path)
    status = str(report.get("status", "")).upper()
    return status if status else None


def _plot_phase3_coverage(
    phase3_root: Path, output_root: Path, formats: Sequence[str]
) -> dict[str, Any] | None:
    specs = [
        ("Model contract", "contract_tests.json"),
        ("Local–Minikube", "local_vs_minikube.json"),
        ("Minikube smoke", "minikube_smoke.json"),
        ("Pod recovery", "pod_recovery.json"),
        ("Latency local", "latency_summary.json"),
    ]
    labels: list[str] = []
    statuses: list[str] = []
    sources: list[str] = []
    for label, filename in specs:
        path = phase3_root / filename
        status = _phase3_evidence_status(path)
        if status:
            labels.append(label)
            statuses.append(status)
            sources.append(str(path.resolve()))
    equivalence = phase3_root / "equivalence_local.json"
    if equivalence.is_file():
        report = _read_json(equivalence)
        if _finite_float(report.get("label_agreement")) == 1.0 and _finite_float(report.get("flow_id_agreement")) == 1.0:
            labels.insert(1, "Offline–FastAPI")
            statuses.insert(1, "PASS")
            sources.append(str(equivalence.resolve()))
    if not labels:
        return None
    status_x = {"FAIL": 0, "BLOCKED": 1, "PASS": 2}
    colors = {"FAIL": "#c23b3b", "BLOCKED": "#d08a25", "PASS": "#2f855a"}
    x = [status_x.get(status, 1) for status in statuses]
    y = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.scatter(x, y, s=180, c=[colors.get(status, colors["BLOCKED"]) for status in statuses])
    for x_value, y_value, status in zip(x, y, statuses):
        axis.text(x_value + 0.08, y_value, status, va="center", fontsize=9)
    axis.set_yticks(y, labels)
    axis.set_xticks([0, 1, 2], ["Fail", "Blocked", "Pass"])
    axis.set_xlim(-0.25, 2.55)
    axis.set_ylim(-0.6, len(labels) - 0.4)
    axis.invert_yaxis()
    axis.set_title("Mức độ hoàn thành bằng chứng vận hành Phase 3")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    stem = "fig_p3_validation_coverage"
    return {
        "name": stem,
        "phase": 3,
        "caption": "Trạng thái các kiểm chứng Phase 3; Blocked phản ánh môi trường chưa sẵn sàng, không được xem là Pass.",
        "paths": _save_figure(figure, output_root, stem, formats),
        "sources": sources,
    }


def generate_phase3_figures(
    phase3_root: Path, output_root: Path, formats: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    builders = (
        _plot_phase3_latency,
        _plot_phase3_equivalence,
        _plot_phase3_coverage,
    )
    generated = [
        record
        for builder in builders
        for record in [builder(phase3_root, output_root, formats)]
        if record is not None
    ]
    skipped = [] if generated else [
        {
            "phase": 3,
            "status": "skipped",
            "reason": f"Không tìm thấy artifact Phase 3 có thể vẽ dưới {phase3_root}.",
        }
    ]
    return generated, skipped


def generate_all(
    *,
    phase2_root: Path,
    phase3_root: Path,
    output_root: Path,
    formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, Any]:
    invalid = sorted(set(formats) - {"png", "pdf", "svg"})
    if invalid:
        raise ValueError(f"Unsupported formats: {invalid}")
    output_root.mkdir(parents=True, exist_ok=True)
    _configure_style()
    phase2_figures, phase2_skipped = generate_phase2_figures(
        phase2_root, output_root, formats
    )
    phase3_figures, phase3_skipped = generate_phase3_figures(
        phase3_root, output_root, formats
    )
    manifest = {
        "policy": "Only observed artifacts are plotted; missing experiments are skipped.",
        "phase2_root": str(phase2_root.resolve()),
        "phase3_root": str(phase3_root.resolve()),
        "output_root": str(output_root.resolve()),
        "figures": [*phase2_figures, *phase3_figures],
        "skipped": [*phase2_skipped, *phase3_skipped],
    }
    manifest_path = output_root / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-runs", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--phase3-results", type=Path, default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_all(
        phase2_root=args.phase2_runs,
        phase3_root=args.phase3_results,
        output_root=args.output_dir,
        formats=args.formats,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
