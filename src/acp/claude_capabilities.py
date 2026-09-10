"""Claude provider capability flags (1M context support).

Centralised so probe / env / provider modules agree on the same notion of
"this Anthropic model supports the 1 000 000-token context window".

Anthropic's Claude Code CLI accepts the ``[1m]`` suffix on ``--model`` to
opt into the 1M-context beta; the wrapper additionally honours the
``ANTHROPIC_BETAS=context-1m-2025-08-07`` env variable.  We use the suffix
as the primary path (preserves the raw model id for ``session/setModel``
hot-swap and persistence) and the env as a defensive fallback.
"""

from __future__ import annotations

from .claude_selection import split_claude_model_selection

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Model-id prefixes whose *base* model accepts the 1M context beta.  We
#: match by prefix so date-stamped releases (e.g. ``claude-opus-4-8-20260101``)
#: are covered without a code change.
CLAUDE_1M_PREFIXES: tuple[str, ...] = (
    "claude-sonnet-4-5",
    "claude-sonnet-4",
    "claude-opus-4-8",
    "claude-opus-4-5",
)

#: Beta token expected in ``ANTHROPIC_BETAS``.
CONTEXT_1M_BETA = "context-1m-2025-08-07"

#: Suffix Claude Code CLI uses on ``--model`` to enable the 1M variant.
SUFFIX_1M = "[1m]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_base_effort(model_id: str) -> tuple[str, str | None]:
    """Split a possible ``base/effort`` composite into its segments.

    Only a recognised Claude effort token is treated as a suffix, so other
    identifiers containing ``/`` are returned untouched.
    """
    base, effort = split_claude_model_selection(model_id)
    return (base or "", effort)


def _join_base_effort(base: str, effort: str | None) -> str:
    return f"{base}/{effort}" if effort else base


def strip_1m_suffix(model_id: str) -> str:
    """Return *model_id* with the ``[1m]`` suffix stripped, if present.

    Idempotent on inputs that don't carry the suffix.  For composite
    ``base[1m]/effort`` selections only the base segment is stripped while
    the effort segment is preserved.  Whitespace is preserved as-is so
    callers retain control over normalisation.
    """
    base, effort = _split_base_effort(str(model_id or ""))
    if base.endswith(SUFFIX_1M):
        base = base[: -len(SUFFIX_1M)]
    return _join_base_effort(base, effort)


def is_1m_variant(model_id: str) -> bool:
    """True iff *model_id* is the 1M-suffixed variant of a Claude model.

    The check runs on the base segment of a possible ``base/effort``
    composite selection.
    """
    base, _effort = _split_base_effort(str(model_id or ""))
    return base.endswith(SUFFIX_1M)


def with_1m_suffix(model_id: str) -> str:
    """Return *model_id* with ``[1m]`` appended to its base (idempotent)."""
    base, effort = _split_base_effort(str(model_id or ""))
    if not base.endswith(SUFFIX_1M):
        base = base + SUFFIX_1M
    return _join_base_effort(base, effort)


def model_supports_1m(model_id: str) -> bool:
    """True iff the *base* of *model_id* is in :data:`CLAUDE_1M_PREFIXES`.

    The ``[1m]`` suffix and any ``/effort`` segment are stripped before
    matching so callers may pass either form.  Matching is by prefix to
    cover date-stamped releases.
    """
    base, _effort = _split_base_effort(str(model_id or ""))
    base = base.removesuffix(SUFFIX_1M).strip()
    if not base:
        return False
    return any(base.startswith(p) for p in CLAUDE_1M_PREFIXES)


__all__ = [
    "CLAUDE_1M_PREFIXES",
    "CONTEXT_1M_BETA",
    "SUFFIX_1M",
    "is_1m_variant",
    "model_supports_1m",
    "strip_1m_suffix",
    "with_1m_suffix",
]
