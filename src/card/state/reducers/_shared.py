"""Shared utilities for card state reducers.

Public functions here are shared infrastructure used across multiple sub-reducers.
"""
from __future__ import annotations

from src.card.themes import MODE_TEMPLATES, TERMINAL_TEMPLATES

from ...ui_text import UI_TEXT
from ..models import CardMetadata, HeaderState


def build_header(metadata: CardMetadata, terminal: str) -> HeaderState:
    """Build header state from metadata and terminal status.

    Shared across lifecycle reducers.
    """
    # Title
    if metadata.project_name:
        title = f"{metadata.mode_emoji} {metadata.project_name} · {metadata.mode_name}"
    else:
        title = UI_TEXT["card_lifecycle_mode_title"].format(
            emoji=metadata.mode_emoji, mode_name=metadata.mode_name
        )

    if metadata.unit_label:
        title = f"{title} · {metadata.unit_label}"

    if metadata.iteration_index and not (metadata.unit_label and str(metadata.iteration_index) in metadata.unit_label):
        if metadata.iteration_total and metadata.iteration_total > 1:
            iteration_label = f"第 {metadata.iteration_index}/{metadata.iteration_total} 轮"
        else:
            iteration_label = f"第 {metadata.iteration_index} 轮"
        title = f"{title} · {iteration_label}"

    # Append continuation marker for rotated cards
    if metadata.continuation_seq > 0:
        title = f"{title} (续 #{metadata.continuation_seq})"

    # Subtitle: tool/model info moved to footer, header only shows project + mode
    subtitle = None

    # Template color: terminal overrides mode
    template = TERMINAL_TEMPLATES.get(terminal) or MODE_TEMPLATES.get(metadata.mode_name, "blue")

    return HeaderState(title=title, subtitle=subtitle, template=template)
