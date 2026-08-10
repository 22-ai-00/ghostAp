"""Workflow engine utilities.

Contains subagent encouragement functionality. Roles are now dynamically
determined by the orchestrator agent based on the task requirements,
rather than being statically predefined.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Subagent encouragement constant
# ---------------------------------------------------------------------------

SUBAGENT_ENCOURAGEMENT_PROMPT: str = (
    "When a task can be decomposed, always delegate to subagents rather than "
    "doing everything yourself. Each subagent can further spawn its own "
    "subagents. When you encounter independent sub-problems "
    "during this task — such as researching a library API, validating a "
    "hypothesis, running a set of tests, or drafting an isolated component — "
    "you are strongly encouraged to delegate them to subagents. Subagents work "
    "in parallel and keep the main thread focused on orchestration and "
    "integration. Prefer spawning a subagent over doing everything sequentially "
    "yourself; the overall task will complete faster and with better separation "
    "of concerns. Each subagent should receive a clear, self-contained brief "
    "and return a structured result."
)


def get_subagent_encouragement_prompt() -> str:
    """Return the subagent encouragement paragraph, or "" if disabled via settings.

    Reads ``workflow_subagent_hint_enabled`` at call time so runtime config
    changes are honoured.  Falls back to returning the paragraph when the
    settings module is unavailable (e.g. during isolated unit tests).
    """
    try:
        from src.config import get_settings

        enabled = bool(getattr(get_settings(), "workflow_subagent_hint_enabled", True))
    except Exception:
        enabled = True
    return SUBAGENT_ENCOURAGEMENT_PROMPT if enabled else ""
