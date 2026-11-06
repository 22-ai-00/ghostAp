"""Helpers for joining streamed text chunks for card display."""
from __future__ import annotations

import re

_STRUCTURAL_LINE_RE = re.compile(
    r"^\s*(?:"
    r"#{1,6}\s+|"
    r"(?:[-*+]|\d+[.)])\s+|"
    r">|"
    r"\|.*\||"
    r"`{3,}|"
    r"-{3,}\s*$"
    r")"
)
_SENTENCE_END_CHARS = set("。！？.!?；;")


def append_stream_text(existing: str, incoming: str) -> str:
    """Append streamed text while removing token-boundary soft newlines."""
    if not existing:
        return incoming.lstrip("\n")
    if not incoming:
        return existing
    joined = soft_join_text_fragments(existing, incoming)
    if joined is None:
        return existing + incoming
    return joined


def soft_join_text_fragments(left: str, right: str) -> str | None:
    """Join adjacent streamed text fragments when they look like one paragraph."""
    if not left or not right:
        return None
    if not _should_collapse_boundary(left, right):
        return None

    left_trimmed = left.rstrip("\n")
    right_trimmed = right.lstrip("\n")
    return f"{left_trimmed}{_soft_join_separator(left_trimmed, right_trimmed)}{right_trimmed}"


def _should_collapse_boundary(left: str, right: str) -> bool:
    if _inside_fenced_code(left):
        return False

    left_line = left.rstrip("\n").rsplit("\n", 1)[-1]
    right_line = right.lstrip("\n").split("\n", 1)[0]
    if not left_line.strip() or not right_line.strip():
        return False
    if left_line.endswith("  "):
        return False
    if _is_structural_markdown_line(left_line) or _is_structural_markdown_line(right_line):
        return False

    boundary_has_break = left.endswith("\n") or right.startswith("\n")
    if boundary_has_break:
        if _ends_sentence(left_line) and (left.endswith("\n\n") or right.startswith("\n\n")):
            return False
        return True

    # No newline at boundary — chunks are contiguous token fragments.
    # Return False so append_stream_text falls through to direct concatenation.
    return False


def _inside_fenced_code(text: str) -> bool:
    return text.count("```") % 2 == 1


def _is_structural_markdown_line(line: str) -> bool:
    return bool(_STRUCTURAL_LINE_RE.match(line))


def _ends_sentence(line: str) -> bool:
    stripped = line.rstrip()
    return bool(stripped and stripped[-1] in _SENTENCE_END_CHARS)


def _soft_join_separator(left: str, right: str) -> str:
    if not left or not right or left[-1].isspace() or right[0].isspace():
        return ""
    if _needs_word_space(left[-1], right[0]):
        return " "
    return ""


def _needs_word_space(left: str, right: str) -> bool:
    left_ascii = left.isascii() and left.isalnum()
    right_ascii = right.isascii() and right.isalnum()
    if left_ascii or right_ascii:
        return True
    return False
