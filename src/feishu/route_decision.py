"""Command routing primitives for the Feishu message pipeline.

Provides:
- RouteTarget: enum of all possible message destinations
- RouteDecision: data class representing a routing outcome
- CommandRouter: stateless command detection (consolidates _is_*_command)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from .product_catalog import is_programming_entry_command

if TYPE_CHECKING:
    from ..trust.models import EffectiveTrust


class RouteTarget(str, Enum):
    """All possible message routing destinations."""

    DEEP_ENGINE = "deep"
    SPEC_ENGINE = "spec"
    WORKFLOW_ENGINE = "workflow"
    PROGRAMMING_MODE = "programming"
    SHELL = "shell"
    SYSTEM_COMMAND = "system_command"
    EXIT_MODE = "exit_mode"
    INTENT_RECOGNITION = "intent_recognition"
    IMAGE_ONLY = "image_only"
    TOPIC_ENGINE = "topic_engine"
    IGNORE = "ignore"
    REPLY_TEXT = "reply_text"


@dataclass(frozen=True)
class RouteDecision:
    """Immutable routing decision produced by the dispatcher.

    Contains enough information for ws_client to execute the action
    without needing the dispatcher to call methods directly.
    """

    target: RouteTarget
    payload: dict = field(default_factory=dict)
    reactions: tuple[str, ...] = ()
    reply_text: Optional[str] = None
    effective_trust: EffectiveTrust | None = None


class CommandRouter:
    """Stateless command detection — replaces ws_client._is_*_command methods."""

    @staticmethod
    def is_exit_command(text: str) -> bool:
        from .handlers.system import SystemHandler
        return SystemHandler.is_exit_command(text)

    @staticmethod
    def is_deep_command(text: str) -> bool:
        from .handlers.system import SystemHandler
        return SystemHandler.is_deep_command(text)

    @staticmethod
    def is_spec_command(text: str) -> bool:
        from .handlers.system import SystemHandler
        return SystemHandler.is_spec_command(text)

    @staticmethod
    def is_workflow_command(text: str) -> bool:
        from .handlers.system import SystemHandler
        return SystemHandler.is_workflow_command(text)


    @staticmethod
    def is_programming_entry(text: str) -> bool:
        return is_programming_entry_command(text)
