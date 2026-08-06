"""Compatibility import for the authoritative :mod:`src.core.graph` module."""

from src.core.graph import *  # noqa: F401,F403
from src.core import graph as _implementation


def __getattr__(name: str):
    return getattr(_implementation, name)
