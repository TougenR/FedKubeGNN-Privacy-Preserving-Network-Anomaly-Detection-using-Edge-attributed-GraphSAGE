"""Compatibility facade for the Flower Message API runtime."""

from src.federated.flower.client_app import build_client_app
from src.federated.flower.server_app import build_server_app

__all__ = ["build_client_app", "build_server_app"]
