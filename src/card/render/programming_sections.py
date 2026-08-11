"""Source-aware prose sections for ordinary programming cards."""

from __future__ import annotations

from src.card.state.models import CardState, ContentBlock
from src.card.text_stream import soft_join_text_fragments
from src.card.tool_display import (
    is_unhelpful_display_label,
    sanitize_tool_failure_detail,
)

_SYSTEM_TEXT_BLOCK_IDS = frozenset({
    "_error",
    "_cancelled",
    "_archived_hint",
    "_archived_nav_hint",
})


def is_programming_thought_block(block: ContentBlock) -> bool:
    """Return whether a text block belongs to the Agent prose stream."""
    return block.block_id not in _SYSTEM_TEXT_BLOCK_IDS


def render_programming_text_section(
    *,
    body_markdown: dict,
    block: ContentBlock,
    state: CardState,
) -> dict:
    """Wrap one semantic TextBlock in a legal static Card 2.0 panel."""
    title, background, border, title_color = _section_presentation(
        block,
        state,
    )
    body = dict(body_markdown)
    body.setdefault("text_size", "normal")
    return {
        "tag": "collapsible_panel",
        "expanded": True,
        "background_color": background,
        "border": {
            "color": border,
            "corner_radius": "8px",
        },
        "padding": "8px 12px",
        "vertical_spacing": "4px",
        "header": {
            "title": {
                "tag": "markdown",
                "content": (
                    f"**<font color='{title_color}'>{title}</font>**"
                ),
            },
            "background_color": background,
            "width": "fill",
            "vertical_align": "center",
            "icon": {
                "tag": "standard_icon",
                "token": "down_outlined",
                "color": "grey",
            },
            "icon_position": "right",
            "icon_expanded_angle": -180,
        },
        "elements": [body],
    }


def _section_presentation(
    block: ContentBlock,
    state: CardState,
) -> tuple[str, str, str, str]:
    if getattr(block, "source_kind", "main") == "subagent":
        label = _latest_subagent_label(block, state)
        return (
            f"子代理 · {label}",
            "grey-50",
            "grey-200",
            "grey-700",
        )

    ordinal, is_last = _main_text_position(block, state)
    if state.terminal == "completed" and is_last:
        return ("最终答复", "green-50", "green-100", "green")
    if (
        state.terminal == "running"
        and is_last
        and block.status == "active"
    ):
        return (
            "主 Agent · 当前进展",
            "blue-50",
            "blue-100",
            "blue",
        )
    return (
        f"主 Agent · 进展 {ordinal}",
        "blue-50",
        "blue-100",
        "blue",
    )


def _latest_subagent_label(
    block: ContentBlock,
    state: CardState,
) -> str:
    sequence = str(
        getattr(block, "source_sequence", "") or ""
    ).strip()
    raw_label = ""
    if sequence:
        for item in state.metadata.subagents:
            item_sequence = str(
                item.get("sequence") or item.get("card_sequence") or ""
            ).strip()
            if item_sequence == sequence:
                raw_label = str(
                    item.get("label")
                    or item.get("name")
                    or item.get("branch")
                    or ""
                )
                break
    if not raw_label:
        raw_label = str(
            getattr(block, "source_label", "") or ""
        )
    safe_label = sanitize_tool_failure_detail(
        raw_label,
        fallback="子任务",
        max_chars=60,
    )
    if is_unhelpful_display_label(safe_label):
        return "子任务"
    return safe_label


def _main_text_position(
    block: ContentBlock,
    state: CardState,
) -> tuple[int, bool]:
    main_blocks = _coalesced_main_text_blocks(state)
    for index, candidate in enumerate(main_blocks, start=1):
        if candidate.block_id == block.block_id:
            return index, index == len(main_blocks)
    return max(1, len(main_blocks)), False


def _coalesced_main_text_blocks(
    state: CardState,
) -> list[ContentBlock]:
    """Mirror render-time soft joins so visible ordinals stay contiguous."""
    semantic_text: list[tuple[ContentBlock, str]] = []
    text_is_adjacent = False
    for candidate in state.blocks:
        if candidate.kind != "text":
            text_is_adjacent = False
            continue

        content = str(candidate.content or "")
        if not content.strip():
            # Segmented text atoms omit empty placeholders before coalescing;
            # keep the numbering mirror on the same semantic stream.
            continue
        if (
            text_is_adjacent
            and semantic_text
            and _visible_text_len(semantic_text[-1][1]) == 1
            and content
            and not content[0].isspace()
            and getattr(
                semantic_text[-1][0],
                "source_ref",
                "main",
            )
            == getattr(candidate, "source_ref", "main")
        ):
            joined = soft_join_text_fragments(
                semantic_text[-1][1],
                content,
            )
            if joined is not None:
                semantic_text[-1] = (candidate, joined)
                text_is_adjacent = True
                continue

        semantic_text.append((candidate, content))
        text_is_adjacent = True

    main_blocks = [
        candidate
        for candidate, content in semantic_text
        if candidate.kind == "text"
        and getattr(candidate, "source_kind", "main") == "main"
        and candidate.block_id not in _SYSTEM_TEXT_BLOCK_IDS
        and bool(content.strip())
    ]
    return main_blocks


def _visible_text_len(text: str) -> int:
    return len("".join(str(text or "").split()))
