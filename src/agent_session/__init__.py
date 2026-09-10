"""Shared ACP and Claude CLI session boundary."""

from __future__ import annotations

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

__all__ = [
    "SyncSession", "SyncClaudeCLISession", "EphemeralReviewSession",
    "create_engine_session", "create_auxiliary_session", "close_session_safely",
    "is_cli_backend", "resolve_cwd", "cli_command_for_agent",
    "agent_type_for_cli_command",
]
