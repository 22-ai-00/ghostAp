"""programming_adapter SectionLayout integration tests."""
from __future__ import annotations

from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card
from src.card.state.models import CardMetadata, CardState, HeaderState
from src.card.state.runtime_stats import RuntimeStats


def test_programming_direct_mode_omits_redundant_phase_banner():
    state = CardState(
        metadata=CardMetadata(
            mode_name="Programming",
            mode_emoji="💬",
            engine_type=None,
            tool_name="Coco",
        ),
        header=HeaderState(title="Programming"),
        blocks=(),
        terminal="running",
    )
    object.__setattr__(state, "runtime_stats", RuntimeStats(elapsed_seconds=32.0))

    pages = render_card(state, RenderBudget())

    assert len(pages) == 1
    body = pages[0]._card_json["body"]["elements"]
    body_text = str(body)
    assert "Programming · Coco · 进行中" not in body_text
    assert "Coco · 进行中" not in body_text
    panel_titles = [
        element.get("header", {}).get("title", {}).get("content", "")
        for element in body
        if element.get("tag") == "collapsible_panel"
    ]
    assert not any("任务列表" in title for title in panel_titles)
