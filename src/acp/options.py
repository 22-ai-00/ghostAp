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

@dataclass
class ACPModelOption:
    """Backend-declared model exposed by explicit configuration commands."""

    name: str
    description: str = ""
    is_default: bool = False
