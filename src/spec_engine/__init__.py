"""Spec Engine — ACP-driven structured development with iterative review.

Follows spec-kit methodology: spec → plan → task → build → review,
with review-driven iteration cycles.
"""

# ---------------------------------------------------------------------------
# Inject perspective display names into engine_base (dependency direction:
# spec_engine → engine_base, never the reverse).
# ---------------------------------------------------------------------------
from ..engine_base import ReviewPerspective as _ReviewPerspective
from ..utils.ui_text import SPEC_UI_TEXT as _SPEC_UI_TEXT
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
