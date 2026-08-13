"""Pagination: split RenderAtoms into pages that fit within RenderBudget."""

from __future__ import annotations

import json
import re

from src.card.render.atoms import RenderAtom, estimate_atom_size
from src.utils.text import utf8_replace_bytes

# Approximate overhead for card config/header/footer skeleton
BASE_OVERHEAD = 500

# Fixed node overhead for elements injected after pagination:
# header/config(3) + banner(3) + footer(8) + buttons(6) = 20
FIXED_NODE_OVERHEAD = 20
_FENCE_LINE_RE = re.compile(
    r"^\s{0,3}(?P<fence>`{3,}|~{3,})(?P<tail>.*)$"
)


def split_atom(atom: RenderAtom, remaining_bytes: int) -> list[RenderAtom] | None:
    """Try to split a splittable atom.

    Split strategies (in order):
    1. By paragraph (double newline)
    2. By line (single newline)
    3. By 1600 character chunks

    Returns None if atom is not splittable.
    """
    if not atom.splittable:
        return None

    content = atom.content
    if not content:
        return None

    # Strategy 1: Split by paragraph (double newline)
    parts = _try_split_by_separator(atom, content, "\n\n", remaining_bytes)
    if parts is not None:
        return parts

    # Strategy 2: Split by line (single newline)
    parts = _try_split_by_separator(atom, content, "\n", remaining_bytes)
    if parts is not None:
        return parts

    # Strategy 3: Split by 1600 character chunks
    parts = _try_split_by_chars(atom, content, remaining_bytes)
    if parts is not None:
        return parts

    return None


def _try_split_by_separator(
    atom: RenderAtom, content: str, separator: str, remaining_bytes: int
) -> list[RenderAtom] | None:
    """Try to split content by separator, fitting first part within remaining_bytes."""
    segments = content.split(separator)
    if len(segments) < 2:
        return None

    # Find how many segments fit in remaining_bytes
    first_part_segments: list[str] = []
    for seg in segments:
        candidate = separator.join(first_part_segments + [seg])
        candidate_size = _estimate_content_bytes(atom, candidate)
        if candidate_size > remaining_bytes and first_part_segments:
            break
        first_part_segments.append(seg)

    if not first_part_segments or len(first_part_segments) == len(segments):
        # Either nothing fits or everything fits — split not useful
        if not first_part_segments:
            return None
        return None

    first_content = separator.join(first_part_segments)
    rest_content = separator.join(segments[len(first_part_segments):])

    parts = _make_split_atoms(atom, first_content, rest_content)
    # A split immediately after an opening fence can consume only the fence;
    # stabilization then prepends that fence to the remainder, making no
    # semantic progress and causing paginate_layout to loop forever. Fall
    # through to character splitting for one very long fenced line instead.
    if len(parts[1].content) >= len(content):
        return None
    if parts[0].byte_size <= remaining_bytes:
        return parts
    return None


def _try_split_by_chars(
    atom: RenderAtom,
    content: str,
    remaining_bytes: int,
) -> list[RenderAtom] | None:
    """Split at the largest character boundary that fits the current page."""
    if len(content) <= 1 or remaining_bytes <= 0:
        return None

    best_split = 0
    low = 1
    high = len(content) - 1
    while low <= high:
        split_point = (low + high) // 2
        parts = _make_split_atoms(
            atom,
            content[:split_point],
            content[split_point:],
        )
        if parts[0].byte_size <= remaining_bytes:
            best_split = split_point
            low = split_point + 1
        else:
            high = split_point - 1

    if best_split <= 0:
        return None
    return _make_split_atoms(
        atom,
        content[:best_split],
        content[best_split:],
    )


def _make_split_atoms(
    atom: RenderAtom, first_content: str, rest_content: str
) -> list[RenderAtom]:
    """Create split atom parts from content pieces."""
    if atom.kind in {"text", "reasoning", "tool_panel"}:
        first_content, rest_content = stabilize_markdown_split(first_content, rest_content)

    first_atom = RenderAtom(
        kind=atom.kind,
        block_id=atom.block_id,
        content=first_content,
        splittable=True,
        node_count=atom.node_count,
        structural_overhead=atom.structural_overhead,
    )
    first_atom.byte_size = estimate_atom_size(first_atom)

    rest_atom = RenderAtom(
        kind=atom.kind,
        block_id=atom.block_id,
        content=rest_content,
        splittable=True,
        node_count=atom.node_count,
        structural_overhead=atom.structural_overhead,
    )
    rest_atom.byte_size = estimate_atom_size(rest_atom)

    return [first_atom, rest_atom]


def stabilize_markdown_split(first_content: str, rest_content: str) -> tuple[str, str]:
    """Close and reopen Markdown spans split across independently rendered pages."""
    fence = _open_markdown_fence(first_content)
    if fence:
        first_suffix = "" if first_content.endswith("\n") else "\n"
        rest_prefix = "" if rest_content.startswith("\n") else "\n"
        return f"{first_content}{first_suffix}{fence}", f"{fence}{rest_prefix}{rest_content}"

    inline_tick = _last_unclosed_inline_code_tick(first_content)
    if inline_tick:
        return f"{first_content}{inline_tick}", f"{inline_tick}{rest_content}"

    return first_content, rest_content


def _open_markdown_fence(content: str) -> str:
    open_fence = ""
    for raw_line in str(content).splitlines():
        open_fence = _advance_markdown_fence(open_fence, raw_line)
    return open_fence


def _advance_markdown_fence(open_fence: str, raw_line: str) -> str:
    """Advance one CommonMark-style fenced-code delimiter state."""
    match = _FENCE_LINE_RE.match(raw_line)
    if not match:
        return open_fence

    fence = match.group("fence")
    if not open_fence:
        return fence

    closes_current = (
        fence[0] == open_fence[0]
        and len(fence) >= len(open_fence)
        and not match.group("tail").strip()
    )
    return "" if closes_current else open_fence


def _last_unclosed_inline_code_tick(content: str) -> str:
    last_unclosed = ""
    open_fence = ""
    for raw_line in str(content).splitlines():
        next_fence = _advance_markdown_fence(open_fence, raw_line)
        if next_fence != open_fence:
            open_fence = next_fence
            continue
        if open_fence:
            continue
        for tick in _iter_unescaped_inline_backtick_runs(raw_line):
            if not last_unclosed:
                last_unclosed = tick
            elif last_unclosed == tick:
                last_unclosed = ""
    return last_unclosed


def _iter_unescaped_inline_backtick_runs(text: str):
    i = 0
    while i < len(text):
        if text[i] != "`":
            i += 1
            continue
        escaped = i > 0 and text[i - 1] == "\\"
        j = i
        while j < len(text) and text[j] == "`":
            j += 1
        run = text[i:j]
        if not escaped and len(run) < 3:
            yield run
        i = j


def _estimate_content_bytes(atom: RenderAtom, content: str) -> int:
    """Estimate JSON byte size for content."""
    overhead = 100 + max(0, atom.structural_overhead)
    if atom.structural_overhead <= 0:
        return len(utf8_replace_bytes(content)) * 3 + overhead
    encoded_content = json.dumps(content, ensure_ascii=False)
    return len(utf8_replace_bytes(encoded_content)) + overhead
