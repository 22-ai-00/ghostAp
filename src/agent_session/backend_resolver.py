"""Centralized backend resolution for agent types."""

from __future__ import annotations

from typing import Literal


def resolve_backend_kind(agent_type: str) -> Literal["acp", "cli"]:
    """Determine transport backend for given agent type."""
    normalized = agent_type.lower().strip()
    if normalized == "claude":
        return "cli"
    return "acp"


def is_cli_backend(agent_type: str) -> bool:
    """Shorthand: does this agent type use CLI bridge?"""
    return resolve_backend_kind(agent_type) == "cli"


def resolve_cwd(agent_type: str, root_path: str) -> str:
    """Resolve the working directory for an agent."""
    from ..utils.path import normalize_session_cwd

    return normalize_session_cwd(root_path) or root_path
