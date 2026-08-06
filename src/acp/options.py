"""Presentation models shared by ACP tool and model selection surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ACPToolOption:
    """Tool option rendered by cards and command handlers."""

    name: str
    description: str = ""
    is_default: bool = False
    emoji: str = "🤖"


@dataclass(frozen=True)
class ACPModelVariantOption:
    """One explicit UI and persistence variant of a model family."""

    name: str
    profile: str
    effort: str = "default"
    display_name: str = ""
    is_variant_default: bool = False


@dataclass
class ACPModelOption:
    """Model option rendered by cards and command handlers."""

    name: str
    description: str = ""
    is_default: bool = False
    supports_1m: bool = False
    reasoning_efforts: tuple[str, ...] = ()
    adapted_reasoning_effort: Optional[str] = None
    selection_variants: tuple[ACPModelVariantOption, ...] = ()
