"""Typed ingress identity access settings shared by configuration and routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IngressAccessMode(str, Enum):
    """How inbound users and chats are authorized."""

    ENFORCED = "enforced"
    SHADOW = "shadow"
    LEGACY_ALLOW_ALL = "legacy_allow_all"


class AccessFindingSeverity(str, Enum):
    """Severity for ingress configuration reload findings."""

    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class AccessFinding:
    """An ingress configuration reload failure kept for diagnostics."""

    code: str
    severity: AccessFindingSeverity
    message: str
