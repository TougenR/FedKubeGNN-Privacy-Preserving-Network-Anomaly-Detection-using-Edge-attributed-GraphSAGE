"""Strict Phase 2 configuration loader."""

from src.federated.config.schema import (
    Phase2Config,
    Phase2ConfigError,
    load_phase2_config,
)

__all__ = ["Phase2Config", "Phase2ConfigError", "load_phase2_config"]
