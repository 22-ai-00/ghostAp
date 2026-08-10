"""Workflow command definitions — Single Source of Truth (SSOT).

All modules that need to recognize workflow commands should import from here
rather than maintaining their own hard-coded sets.
"""

from __future__ import annotations

# Engine identifier used in thread context mode field
ENGINE_MODE: str = "workflow"

# Display name shown in UI/cards
ENGINE_DISPLAY_NAME: str = "WF"

# --- Command sets ---

# Short-form commands (primary entry points)
SHORT_COMMANDS: frozenset[str] = frozenset({
    "/wf",
    "/wf_status",
    "/wf_help",
    "/stop_wf",
})

# Long-form commands (canonical aliases)
LONG_COMMANDS: frozenset[str] = frozenset({
    "/workflow",
    "/workflow_status",
    "/workflow_help",
    "/stop_workflow",
})

# All commands recognized as topic-engine triggers
TOPIC_ENGINE_COMMANDS: frozenset[str] = SHORT_COMMANDS | LONG_COMMANDS

# Commands that accept trailing arguments (e.g., "/wf do code review")
PREFIX_COMMANDS: frozenset[str] = frozenset({
    "/wf",
    "/workflow",
})

# All workflow prefixes for is_workflow_command matching
ALL_PREFIXES: tuple[str, ...] = tuple(sorted(TOPIC_ENGINE_COMMANDS, key=len, reverse=True))


def is_workflow_command(text: str) -> bool:
    """Return True if *text* starts with any recognized workflow command."""
    text_lower = text.lower().strip()
    return any(
        text_lower == cmd or text_lower.startswith(f"{cmd} ")
        for cmd in ALL_PREFIXES
    )
