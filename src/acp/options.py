"""Presentation models shared by ACP tool and model selection surfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ACPToolOption:
    """Tool option rendered by cards and command handlers."""

    name: str
    description: str = ""
    is_default: bool = False
    emoji: str = "🤖"


@dataclass(frozen=True)
class ACPModelSelectionVariant:
    """One exact model selection declared by a backend capability matrix."""

    name: str
    model: str
    profile: str | None = None
    effort: str | None = None
    is_default: bool = False


@dataclass
class ACPModelOption:
    """Backend-declared model exposed by explicit configuration commands."""

    name: str
    description: str = ""
    is_default: bool = False
    selection_variants: tuple[ACPModelSelectionVariant, ...] = ()
    reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
