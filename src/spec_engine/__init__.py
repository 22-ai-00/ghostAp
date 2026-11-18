"""Spec Engine — ACP-driven structured development with iterative review.

Follows spec-kit methodology: spec → plan → task → build → review,
with review-driven iteration cycles.
"""

# ---------------------------------------------------------------------------
# Inject perspective display names into engine_base (dependency direction:
# spec_engine → engine_base, never the reverse).
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

from ..engine_base import ReviewPerspective as _ReviewPerspective
from ..utils.ui_text import SPEC_UI_TEXT as _SPEC_UI_TEXT

if TYPE_CHECKING:
    from .engine import SpecEngine, SpecEngineCallbacks
    from .manager import SpecEngineManager
    from .models import (
        SpecCycle,
        SpecPhase,
        SpecProject,
        SpecProjectStatus,
        SpecTask,
        SpecTaskStatus,
    )
    from .reporter import SpecReporter
    from .tracker import PhaseTracker

_ReviewPerspective.register_display_names(
    {k: v for k, v in _SPEC_UI_TEXT.items() if k.startswith("perspective_")}
)
del _ReviewPerspective, _SPEC_UI_TEXT  # keep namespace clean

_MODEL_EXPORTS = frozenset(
    {
        "SpecCycle",
        "SpecPhase",
        "SpecProject",
        "SpecProjectStatus",
        "SpecTask",
        "SpecTaskStatus",
    }
)


def __getattr__(name: str):
    if name == "SpecEngineManager":
        from .manager import SpecEngineManager

        globals()["SpecEngineManager"] = SpecEngineManager
        return SpecEngineManager
    if name in ("SpecEngine", "SpecEngineCallbacks"):
        from .engine import SpecEngine, SpecEngineCallbacks

        globals()["SpecEngine"] = SpecEngine
        globals()["SpecEngineCallbacks"] = SpecEngineCallbacks
        return globals()[name]
    if name in _MODEL_EXPORTS:
        from . import models as _models

        for attr in _MODEL_EXPORTS:
            globals()[attr] = getattr(_models, attr)
        return globals()[name]
    if name == "SpecReporter":
        from .reporter import SpecReporter

        globals()["SpecReporter"] = SpecReporter
        return SpecReporter
    if name == "PhaseTracker":
        from .tracker import PhaseTracker

        globals()["PhaseTracker"] = PhaseTracker
        return PhaseTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Engine
    "SpecEngine",
    "SpecEngineManager",
    "SpecEngineCallbacks",
    # Models
    "SpecProject",
    "SpecProjectStatus",
    "SpecPhase",
    "SpecCycle",
    "SpecTask",
    "SpecTaskStatus",
    # Tracker
    "PhaseTracker",
    # Reporter
    "SpecReporter",
]
