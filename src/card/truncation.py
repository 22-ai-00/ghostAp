"""Unified truncation utilities for Feishu card content.

Aligned with pokoclaw card-truncation.ts conventions.
All default thresholds are read from styles.TRUNCATION_LIMITS.
"""

from __future__ import annotations

from src.card.thresholds import TRUNCATION_LIMITS

_NOTICE = "\n...[truncated]"


def truncate_card_string(
    value: str,
    max_chars: int | None = None,
    max_lines: int | None = None,
    notice: str | None = None,
) -> str:
    """Truncate a string by line count first, then by character count.

    Returns the original string if within both limits.
    Appends *notice* (default ``_NOTICE``) when truncation occurs.
    """
    if not value:
        return value
    if max_chars is None:
        max_chars = TRUNCATION_LIMITS["card_string_max_chars"]
    if max_lines is None:
        max_lines = TRUNCATION_LIMITS["card_string_max_lines"]
    if notice is None:
        notice = _NOTICE

    truncated = False
    result = value

    # Line truncation first
    lines = result.split("\n")
    if len(lines) > max_lines:
        result = "\n".join(lines[:max_lines])
        truncated = True

    # Then character truncation
    if len(result) > max_chars:
        result = result[:max_chars]
        truncated = True

    if truncated:
        result += notice
    return result


def truncate_bash_output(
    value: str,
    max_chars: int | None = None,
    max_lines: int | None = None,
    notice: str | None = None,
) -> str:
    """Truncate bash command output (stdout/stderr)."""
    if max_chars is None:
        max_chars = TRUNCATION_LIMITS["bash_max_chars"]
    if max_lines is None:
        max_lines = TRUNCATION_LIMITS["bash_max_lines"]
    return truncate_card_string(value, max_chars=max_chars, max_lines=max_lines, notice=notice)
