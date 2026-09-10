"""Claude (and the claude-w sibling) model selection value helpers.

Persisted selections use a single composite value shaped
``base[1m]/effort``:

- ``base`` is the raw Claude model id (``claude-sonnet-4-5``, an alias such
  as ``sonnet``, or the :data:`CLAUDE_DEFAULT_MODEL_TOKEN` sentinel);
- ``[1m]`` always stays attached to the base segment and opts into the
  1M-context beta (see :mod:`src.acp.claude_capabilities`);
- ``effort`` is one of :data:`CLAUDE_REASONING_EFFORTS`, mapped to the
  Claude Code CLI ``--effort`` flag.

The sentinel base means "do not pass ``--model`` at all" so the local CLI
(or the claude-w gateway wrapper) keeps its own default model selection.
"""

from __future__ import annotations

from typing import Optional

#: Effort levels accepted by ``claude --effort`` (verified via --help).
#: Ordered for stable card rendering.
CLAUDE_REASONING_EFFORTS: tuple[str, ...] = (
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
_CLAUDE_REASONING_EFFORT_SET = frozenset(CLAUDE_REASONING_EFFORTS)

#: Pseudo-model meaning "CLI default model": the bridge omits ``--model``
#: entirely and only forwards the chosen effort (if any).
CLAUDE_DEFAULT_MODEL_TOKEN = "default"

#: Sentinel effort shown in the card dropdown for "do not pass --effort".
CLAUDE_DEFAULT_EFFORT_TOKEN = "default"


def split_claude_model_selection(
    value: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Split a persisted ``base/effort`` Claude selection.

    Only a recognised final effort token is removed, so provider-qualified
    model identifiers containing ``/`` remain intact.  The ``[1m]`` suffix,
    when present, stays on the base segment.

    Returns ``(base, effort)``; either component may be ``None`` when absent.
    """
    selection = str(value or "").strip()
    if not selection:
        return None, None
    base, separator, suffix = selection.rpartition("/")
    if separator and suffix.lower() in _CLAUDE_REASONING_EFFORT_SET:
        return base, suffix.lower()
    return selection, None


def compose_claude_model_selection(
    base: str,
    effort: Optional[str],
) -> str:
    """Compose the stable UI/persistence value for a Claude selection."""
    model = str(base or "").strip()
    level = str(effort or "").strip().lower()
    if not model:
        return ""
    if level in _CLAUDE_REASONING_EFFORT_SET:
        return f"{model}/{level}"
    return model
