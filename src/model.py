"""Compatibility import for the authoritative :mod:`src.core.model` module."""

from src.core.model import *  # noqa: F401,F403
from src.core import model as _implementation


def __getattr__(name: str):
    return getattr(_implementation, name)
