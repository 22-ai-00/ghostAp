"""Unified display and status metadata for all GhostAP programming backends."""

from __future__ import annotations

from src.mode.backends import (
    ALL_BACKEND_METAS,
    BACKEND_META_BY_KEY,
    VISIBLE_BACKEND_METAS,
    get_backend_meta,
    resolve_project_programming_mode,
)
from src.mode.backends import (
    BackendSpec as BackendMeta,
)

__all__ = [
    "BackendMeta",
    "ALL_BACKEND_METAS",
    "VISIBLE_BACKEND_METAS",
    "BACKEND_META_BY_KEY",
    "get_backend_meta",
    "resolve_project_programming_mode",
]
