"""Versioned, strict configuration for the Phase 2 benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class Phase2ConfigError(ValueError):
    """Raised when Phase 2 configuration is incomplete or ambiguous."""


def _strict(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise Phase2ConfigError(f"Unknown keys in {where}: {unknown}.")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase2ConfigError(f"{where} must be a mapping.")
    return value


@dataclass(frozen=True)
class ComponentsConfig:
    data_source: str
    partitioner: str
    graph_builder: str
    model: str
    task: str
    runtime: str
    observer: str


@dataclass(frozen=True)
class SplitConfig:
    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value <= 0 or value >= 1 for value in values):
            raise Phase2ConfigError("Every split ratio must be between 0 and 1.")
        if abs(sum(values) - 1.0) > 1e-9:
            raise Phase2ConfigError("Split ratios must sum to 1.0.")


@dataclass(frozen=True)
class ScenarioConfig:
    id: str
    path: str
    malware: str


@dataclass(frozen=True)
class DataConfig:
    prepared_root: str
    raw_root: str
    cap_per_class: int | None
    chunk_size: int
    target_column: str
    split: SplitConfig
    preprocessing: str
    graph_protocol: str
    scenarios: tuple[ScenarioConfig, ...]


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    num_layers: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    rounds: int
    local_epochs: int
    centralized_epochs: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    grad_clip: float
    imbalance: str
    class_weight_scope: str
    seed: int


@dataclass(frozen=True)
class FederationConfig:
    strategies: tuple[str, ...]
    proximal_mu: float
    participation: str
    evaluate_split: str
    final_split: str


@dataclass(frozen=True)
class ObservabilityConfig:
    output_root: str
    level: str
    console: bool
    jsonl: bool


@dataclass(frozen=True)
class Phase2Config:
    version: int
    components: ComponentsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    federation: FederationConfig
    observability: ObservabilityConfig

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_config(raw: Mapping[str, Any]) -> Phase2Config:
    top = {
        "version",
        "components",
        "data",
        "model",
        "training",
        "federation",
        "observability",
    }
    _strict(raw, top, "root")
    missing = sorted(top - set(raw))
    if missing:
        raise Phase2ConfigError(f"Missing root keys: {missing}.")
    if int(raw["version"]) != 1:
        raise Phase2ConfigError(f"Unsupported config version: {raw['version']}.")

    components_raw = _mapping(raw["components"], "components")
    component_fields = set(ComponentsConfig.__dataclass_fields__)
    _strict(components_raw, component_fields, "components")
    if component_fields - set(components_raw):
        raise Phase2ConfigError("components must select every extension point.")
    components = ComponentsConfig(
        **{key: str(components_raw[key]) for key in component_fields}
    )

    data_raw = _mapping(raw["data"], "data")
    data_fields = set(DataConfig.__dataclass_fields__)
    _strict(data_raw, data_fields, "data")
    split_raw = _mapping(data_raw.get("split"), "data.split")
    _strict(split_raw, {"train", "validation", "test"}, "data.split")
    split = SplitConfig(
        **{key: float(split_raw[key]) for key in ("train", "validation", "test")}
    )
    scenarios_raw = data_raw.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise Phase2ConfigError("data.scenarios must be a non-empty list.")
    scenarios: list[ScenarioConfig] = []
    for index, scenario_value in enumerate(scenarios_raw):
        scenario = _mapping(scenario_value, f"data.scenarios[{index}]")
        _strict(scenario, {"id", "path", "malware"}, f"data.scenarios[{index}]")
        scenarios.append(
            ScenarioConfig(
                **{key: str(scenario[key]) for key in ("id", "path", "malware")}
            )
        )
    ids = [scenario.id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise Phase2ConfigError("Scenario ids must be unique.")
    cap = data_raw.get("cap_per_class")
    data = DataConfig(
        prepared_root=str(data_raw["prepared_root"]),
        raw_root=str(data_raw["raw_root"]),
        cap_per_class=None if cap is None else int(cap),
        chunk_size=int(data_raw["chunk_size"]),
        target_column=str(data_raw["target_column"]),
        split=split,
        preprocessing=str(data_raw["preprocessing"]),
        graph_protocol=str(data_raw["graph_protocol"]),
        scenarios=tuple(scenarios),
    )
    if data.cap_per_class is not None and data.cap_per_class < 1:
        raise Phase2ConfigError("data.cap_per_class must be positive or null.")
    if data.chunk_size < 1:
        raise Phase2ConfigError("data.chunk_size must be positive.")
    if data.preprocessing != "train_only_global":
        raise Phase2ConfigError(
            "Only preprocessing=train_only_global is benchmark-authorized."
        )
    if data.graph_protocol != "transductive_edge_mask":
        raise Phase2ConfigError(
            "Only graph_protocol=transductive_edge_mask is currently implemented."
        )

    model_raw = _mapping(raw["model"], "model")
    _strict(model_raw, set(ModelConfig.__dataclass_fields__), "model")
    model = ModelConfig(
        hidden_dim=int(model_raw["hidden_dim"]),
        num_layers=int(model_raw["num_layers"]),
        dropout=float(model_raw["dropout"]),
    )
    if model.hidden_dim < 1 or model.num_layers < 1 or not 0 <= model.dropout < 1:
        raise Phase2ConfigError("Invalid model dimensions or dropout.")

    training_raw = _mapping(raw["training"], "training")
    _strict(training_raw, set(TrainingConfig.__dataclass_fields__), "training")
    training = TrainingConfig(
        **{
            "rounds": int(training_raw["rounds"]),
            "local_epochs": int(training_raw["local_epochs"]),
            "centralized_epochs": int(training_raw["centralized_epochs"]),
            "optimizer": str(training_raw["optimizer"]),
            "learning_rate": float(training_raw["learning_rate"]),
            "weight_decay": float(training_raw["weight_decay"]),
            "grad_clip": float(training_raw["grad_clip"]),
            "imbalance": str(training_raw["imbalance"]),
            "class_weight_scope": str(training_raw["class_weight_scope"]),
            "seed": int(training_raw["seed"]),
        }
    )
    if min(training.rounds, training.local_epochs, training.centralized_epochs) < 1:
        raise Phase2ConfigError("Epoch and round counts must be positive.")
    if training.optimizer not in {"sgd", "adam", "adamw"} or training.imbalance not in {
        "none",
        "class_weight",
    }:
        raise Phase2ConfigError("Unsupported optimizer or imbalance mode.")
    if training.class_weight_scope not in {"local", "global"}:
        raise Phase2ConfigError("class_weight_scope must be local or global.")
    if training.imbalance == "none" and training.class_weight_scope != "local":
        raise Phase2ConfigError(
            "class_weight_scope=global requires imbalance=class_weight."
        )

    federation_raw = _mapping(raw["federation"], "federation")
    _strict(federation_raw, set(FederationConfig.__dataclass_fields__), "federation")
    strategies = tuple(str(item).lower() for item in federation_raw["strategies"])
    if not strategies or any(item not in {"fedavg", "fedprox"} for item in strategies):
        raise Phase2ConfigError("federation.strategies must contain fedavg/fedprox.")
    federation = FederationConfig(
        strategies=strategies,
        proximal_mu=float(federation_raw["proximal_mu"]),
        participation=str(federation_raw["participation"]),
        evaluate_split=str(federation_raw["evaluate_split"]),
        final_split=str(federation_raw["final_split"]),
    )
    if (
        federation.participation != "full"
        or federation.evaluate_split != "validation"
        or federation.final_split != "test"
    ):
        raise Phase2ConfigError(
            "Benchmark requires full participation, validation per round, and final test."
        )
    if federation.proximal_mu < 0:
        raise Phase2ConfigError("federation.proximal_mu must be non-negative.")

    obs_raw = _mapping(raw["observability"], "observability")
    _strict(obs_raw, set(ObservabilityConfig.__dataclass_fields__), "observability")
    observability = ObservabilityConfig(
        output_root=str(obs_raw["output_root"]),
        level=str(obs_raw["level"]).upper(),
        console=bool(obs_raw["console"]),
        jsonl=bool(obs_raw["jsonl"]),
    )
    return Phase2Config(1, components, data, model, training, federation, observability)


def load_phase2_config(path: str | Path) -> Phase2Config:
    """Load YAML after validating every supported key and benchmark invariant."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError("Loading Phase 2 YAML requires PyYAML.") from exc
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _build_config(_mapping(raw, "root"))
