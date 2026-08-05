"""Card actions subpackage — action constants, dispatch registries, and routing.

Re-exports public API for backward compatibility.
"""

from src.card.actions.dispatch import *  # noqa: F401,F403 — all action_id constants
from src.card.actions.dispatch import build_common_action_registry
from src.card.actions.router import ActionRouter

__all__ = [
    "ActionRouter",
    "build_common_action_registry",
]
