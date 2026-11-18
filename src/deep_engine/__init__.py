from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import DeepEngine, DeepEngineCallbacks
    from .manager import DeepEngineManager
    from .models import DeepProject, DeepProjectStatus, EngineRunState
    from .progress import DeepProgress
    from .reporter import ProgressReporter

_MODEL_EXPORTS = frozenset({"DeepProject", "DeepProjectStatus", "EngineRunState"})


def __getattr__(name: str):
    if name == "DeepEngineManager":
        from .manager import DeepEngineManager

        globals()["DeepEngineManager"] = DeepEngineManager
        return DeepEngineManager
    if name in ("DeepEngine", "DeepEngineCallbacks"):
        from .engine import DeepEngine, DeepEngineCallbacks

        globals()["DeepEngine"] = DeepEngine
        globals()["DeepEngineCallbacks"] = DeepEngineCallbacks
        return globals()[name]
    if name in _MODEL_EXPORTS:
        from . import models as _models

        for attr in _MODEL_EXPORTS:
            globals()[attr] = getattr(_models, attr)
        return globals()[name]
    if name == "DeepProgress":
        from .progress import DeepProgress

        globals()["DeepProgress"] = DeepProgress
        return DeepProgress
    if name == "ProgressReporter":
        from .reporter import ProgressReporter

        globals()["ProgressReporter"] = ProgressReporter
        return ProgressReporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DeepEngine",
    "DeepEngineManager",
    "DeepEngineCallbacks",
    "DeepProgress",
    "DeepProject",
    "DeepProjectStatus",
    "EngineRunState",
    "ProgressReporter",
]
