from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import DeepCardState, EngineCardState
    from .shared import (
        THEMES,
        ProjectTheme,
        build_mode_buttons,
        build_responsive_layout,
        get_theme,
    )

_MODEL_EXPORTS = frozenset({"DeepCardState", "EngineCardState"})
_SHARED_EXPORTS = frozenset(
    {
        "THEMES",
        "ProjectTheme",
        "build_mode_buttons",
        "build_responsive_layout",
        "get_theme",
    }
)


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        from . import models as _models

        for attr in _MODEL_EXPORTS:
            globals()[attr] = getattr(_models, attr)
        return globals()[name]
    if name in _SHARED_EXPORTS:
        from . import shared as _shared

        for attr in _SHARED_EXPORTS:
            globals()[attr] = getattr(_shared, attr)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DeepCardState",
    "EngineCardState",
    "ProjectTheme",
    "THEMES",
    "get_theme",
    "build_mode_buttons",
    "build_responsive_layout",
]
