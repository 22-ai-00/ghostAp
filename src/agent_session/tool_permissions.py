"""Least-privilege profiles for non-executing auxiliary agent sessions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class AuxiliarySessionPermissionError(RuntimeError):
    """Raised when an auxiliary session cannot enforce its tool profile."""


class AuxiliaryToolProfile(StrEnum):
    """Tool profiles allowed for coordination-only model sessions."""

    DENY_ALL = "deny_all"


def deny_all_tools(_tool_name: str, _args: dict | None = None) -> bool:
    """Reject every tool request, including unknown future tool names."""

    return False


def apply_auxiliary_tool_profile(
    session: Any,
    *,
    profile: AuxiliaryToolProfile = AuxiliaryToolProfile.DENY_ALL,
) -> None:
    """Install a fail-closed tool profile on one auxiliary session."""

    if profile is not AuxiliaryToolProfile.DENY_ALL:
        raise AuxiliarySessionPermissionError(
            f"unsupported auxiliary tool profile: {profile!s}"
        )
    set_tool_filter = getattr(session, "set_tool_filter", None)
    if not callable(set_tool_filter):
        raise AuxiliarySessionPermissionError(
            "auxiliary deny-all tool profile is unavailable"
        )
    set_tool_filter(deny_all_tools)


__all__ = [
    "AuxiliarySessionPermissionError",
    "AuxiliaryToolProfile",
    "apply_auxiliary_tool_profile",
    "deny_all_tools",
]
