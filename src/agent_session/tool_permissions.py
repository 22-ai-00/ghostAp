"""Least-privilege profiles for non-executing auxiliary agent sessions."""

from __future__ import annotations

from typing import Any


class AuxiliarySessionPermissionError(RuntimeError):
    """Raised when an auxiliary session cannot enforce its tool profile."""


def deny_all_tools(_tool_name: str, _args: dict | None = None) -> bool:
    """Reject every tool request, including unknown future tool names."""

    return False


def apply_auxiliary_tool_profile(
    session: Any,
) -> None:
    """Install a fail-closed tool profile on one auxiliary session."""
    set_tool_filter = getattr(session, "set_tool_filter", None)
    if not callable(set_tool_filter):
        raise AuxiliarySessionPermissionError(
            "auxiliary deny-all tool profile is unavailable"
        )
    set_tool_filter(deny_all_tools)


__all__ = [
    "AuxiliarySessionPermissionError",
    "apply_auxiliary_tool_profile",
    "deny_all_tools",
]
