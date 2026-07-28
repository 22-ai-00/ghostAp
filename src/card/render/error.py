"""Collapsed error-detail rendering for task cards."""

from __future__ import annotations

import re

from src.card.themes import PANEL_STYLES
from src.utils.text import sanitize_single_line_label

_FAILED_TEST_RE = re.compile(r"\bFAILED\s+([^\s\\\"']+)")
_ERROR_HEADING = "❌ **错误摘要**"
_STATUS_HEADING = "**当前状态**"
_SUMMARY_MAX_CHARS = 96


def _error_summary(content: str) -> str:
    """Extract a short, stable summary without leaking a raw JSON payload."""
    text = str(content or "").strip()
    failed_test = _FAILED_TEST_RE.search(text)
    if failed_test:
        return sanitize_single_line_label(
            f"测试失败：{failed_test.group(1)}",
            fallback="任务执行失败",
            max_chars=_SUMMARY_MAX_CHARS,
        )

    summary_section = text
    if _ERROR_HEADING in summary_section:
        summary_section = summary_section.split(_ERROR_HEADING, 1)[1]
    if _STATUS_HEADING in summary_section:
        summary_section = summary_section.split(_STATUS_HEADING, 1)[0]
    first_line = next(
        (line.strip() for line in summary_section.splitlines() if line.strip()),
        "",
    )
    if not first_line or first_line.startswith(("{", "[")) or len(first_line) > _SUMMARY_MAX_CHARS:
        return "任务返回了较长的错误信息"
    return sanitize_single_line_label(
        first_line,
        fallback="任务执行失败",
        max_chars=_SUMMARY_MAX_CHARS,
    )


def render_collapsed_error_panel(
    *,
    detail_content: str,
    full_content: str,
) -> dict:
    """Render diagnostics in a red panel that is collapsed by default."""
    summary = _error_summary(full_content)
    return {
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {
            "title": {
                "tag": "markdown",
                "content": f"❌ **错误详情** · {summary} · 已收起",
            },
            "vertical_align": "center",
            "icon": {
                "tag": "standard_icon",
                "token": "down-small-ccm_outlined",
                "size": "16px 16px",
            },
            "icon_position": "follow_text",
            "icon_expanded_angle": -180,
        },
        "border": {
            "color": PANEL_STYLES["border_failed"],
            "corner_radius": PANEL_STYLES["corner_radius"],
        },
        "vertical_spacing": PANEL_STYLES["vertical_spacing"],
        "padding": PANEL_STYLES["padding_standard"],
        "elements": [
            {
                "tag": "markdown",
                "content": detail_content,
                "text_size": "normal",
            }
        ],
    }
