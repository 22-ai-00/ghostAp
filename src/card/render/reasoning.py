"""Reasoning panel rendering."""

from __future__ import annotations

from src.card.render.budget import RenderBudget
from src.card.state.models import ContentBlock
from src.card.themes import PANEL_STYLES
from src.card.ui_text import UI_TEXT

REASONING_COMPACT_CHARS = 222
_MARKDOWN_LIST_PREFIXES = ("- ", "* ", "+ ")


def truncate_reasoning_for_compact(content: str, *, max_chars: int = REASONING_COMPACT_CHARS) -> str:
    """Return a compact reasoning preview capped to ``max_chars`` visible chars."""
    content = str(content or "")
    if max_chars <= 0:
        return ""
    if len(content) <= max_chars:
        return content
    if max_chars == 1:
        return "…"
    return content[: max_chars - 1] + "…"


def _format_reasoning_as_list(content: str) -> str:
    """Render non-empty process-summary lines as compact Markdown list items."""
    lines = [line.strip() for line in str(content or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(
        line if line.startswith(_MARKDOWN_LIST_PREFIXES) else f"- {line}"
        for line in lines
    )


def render_reasoning_panel(
    block: ContentBlock,
    budget: RenderBudget | None = None,
    *,
    content_override: str | None = None,
    compact: bool = False,
) -> dict:
    """Render reasoning/thinking as a tool-like collapsible panel.

    Active reasoning is expanded while completed reasoning follows tool-call
    panels and starts collapsed. Full mode preserves the complete text; compact
    mode shows a bounded preview.

    Args:
        block: The reasoning block (used for status/char_count metadata).
        budget: Render budget for truncation limits.
        content_override: If provided, use this instead of block.content.
            Needed when block_index maps to the *last* block with a shared
            block_id but the atom carries the correct per-block content.
    """
    _ = budget

    is_active = block.status == "active"
    raw_content = content_override if content_override is not None else block.content
    raw_content = raw_content or ""
    original_content = block.content or raw_content

    if is_active:
        title_text = UI_TEXT["reasoning_panel_thinking"]
    else:
        char_count = block.char_count or len(original_content)
        title_text = UI_TEXT["reasoning_panel_done"].format(char_count=char_count)

    content = _format_reasoning_as_list(raw_content)
    if compact:
        content = truncate_reasoning_for_compact(content)
    elements = [
        {
            "tag": "markdown",
            "content": content,
            "text_size": "normal",
            "text_align": "left",
        }
    ] if content else []

    return {
        "tag": "collapsible_panel",
        "expanded": is_active,
        "header": {
            "title": {"tag": "markdown", "content": title_text},
            "vertical_align": "center",
            "icon": {
                "tag": "standard_icon",
                "token": "down_outlined",
                "color": "grey",
            },
            "icon_position": "right",
            "icon_expanded_angle": -180,
        },
        "border": {"color": PANEL_STYLES["border_normal"], "corner_radius": PANEL_STYLES["corner_radius"]},
        "vertical_spacing": PANEL_STYLES["vertical_spacing"],
        "padding": PANEL_STYLES["padding_compact"] if compact else PANEL_STYLES["padding_standard"],
        "elements": elements,
    }
