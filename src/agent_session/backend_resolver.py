"""Centralized backend resolution for agent types."""

from __future__ import annotations


def is_cli_backend(agent_type: str) -> bool:
    """Shorthand: does this agent type use CLI bridge?"""
    return agent_type.lower().strip() == "claude"


def resolve_cwd(agent_type: str, root_path: str) -> str:
    """Resolve the working directory for an agent."""
    from ..utils.path import normalize_session_cwd

    return normalize_session_cwd(root_path) or root_path
