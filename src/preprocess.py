"""Compatibility import for :mod:`src.core.preprocess` during migration."""

from src.core.preprocess import *  # noqa: F401,F403
from src.core import preprocess as _implementation


def __getattr__(name: str):
    return getattr(_implementation, name)
