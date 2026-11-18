"""Shared ACP and Claude CLI session boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend_resolver import (
        agent_type_for_cli_command,
        cli_command_for_agent,
        is_cli_backend,
        resolve_cwd,
    )
    from .claude_cli import SyncClaudeCLISession
    from .factory import (
        EphemeralReviewSession,
        close_session_safely,
        create_auxiliary_session,
        create_engine_session,
    )
    from .protocol import SyncSession

_BACKEND_RESOLVER_EXPORTS = frozenset(
    {
        "agent_type_for_cli_command",
        "cli_command_for_agent",
        "is_cli_backend",
        "resolve_cwd",
    }
)
_FACTORY_EXPORTS = frozenset(
    {
        "EphemeralReviewSession",
        "close_session_safely",
        "create_auxiliary_session",
        "create_engine_session",
    }
)


def __getattr__(name: str):
    if name in _BACKEND_RESOLVER_EXPORTS:
        from . import backend_resolver as _backend_resolver

        for attr in _BACKEND_RESOLVER_EXPORTS:
            globals()[attr] = getattr(_backend_resolver, attr)
        return globals()[name]
    if name == "SyncClaudeCLISession":
        from .claude_cli import SyncClaudeCLISession

        globals()["SyncClaudeCLISession"] = SyncClaudeCLISession
        return SyncClaudeCLISession
    if name in _FACTORY_EXPORTS:
        from . import factory as _factory

        for attr in _FACTORY_EXPORTS:
            globals()[attr] = getattr(_factory, attr)
        return globals()[name]
    if name == "SyncSession":
        from .protocol import SyncSession

        globals()["SyncSession"] = SyncSession
        return SyncSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SyncSession", "SyncClaudeCLISession", "EphemeralReviewSession",
    "create_engine_session", "create_auxiliary_session", "close_session_safely",
    "is_cli_backend", "resolve_cwd", "cli_command_for_agent",
    "agent_type_for_cli_command",
]
