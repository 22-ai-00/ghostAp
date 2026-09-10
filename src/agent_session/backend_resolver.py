"""Centralized backend resolution for agent types."""

from __future__ import annotations

# Agent types that bridge to a local shell CLI instead of speaking ACP.
# ``claude``    → the ``claude`` executable
# ``claude_w``  → the separate ``claude-w`` executable (hidden /claude-w command)
CLI_BACKEND_AGENTS: frozenset[str] = frozenset({"claude", "claude_w"})

# agent_type → shell executable name used to spawn the CLI bridge.
CLI_COMMAND_FOR_AGENT: dict[str, str] = {
    "claude": "claude",
    "claude_w": "claude-w",
}


def is_cli_backend(agent_type: str) -> bool:
    """Shorthand: does this agent type use CLI bridge?"""
    return agent_type.lower().strip() in CLI_BACKEND_AGENTS


def cli_command_for_agent(agent_type: str) -> str:
    """Return the shell executable name backing a CLI-bridge agent type."""
    return CLI_COMMAND_FOR_AGENT.get(agent_type.lower().strip(), "claude")


def agent_type_for_cli_command(command: str) -> str:
    """Inverse of :func:`cli_command_for_agent` (defaults to ``claude``)."""
    target = (command or "").strip()
    for agent, executable in CLI_COMMAND_FOR_AGENT.items():
        if executable == target:
            return agent
    return "claude"


def resolve_cwd(agent_type: str, root_path: str) -> str:
    """Resolve the working directory for an agent."""
    from ..utils.path import normalize_session_cwd

    return normalize_session_cwd(root_path) or root_path
