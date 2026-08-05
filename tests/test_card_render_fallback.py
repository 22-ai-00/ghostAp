"""Tests for src/card/render/fallback.py — fallback card rendering."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.card.render.fallback import render_fallback_card


def _make_state(engine_type: str | None = None, title: str = "Test Task"):
    """Create a minimal CardState mock for testing."""
    state = MagicMock()
    state.header = MagicMock()
    state.header.title = title
    if engine_type:
        state.metadata = MagicMock()
        state.metadata.engine_type = engine_type
    else:
        state.metadata = None
    return state


class TestRenderFallbackCard:
    """Test render_fallback_card output structure."""

    @pytest.mark.parametrize("engine_type,expected_cmd", [
        ("deep", "/deep"),
        ("spec", "/spec"),
    ])
    def test_engine_specific_command_in_warning(self, engine_type, expected_cmd):
        """Fallback card includes the correct engine command in warning text."""
        state = _make_state(engine_type=engine_type)
        result = render_fallback_card(state, engine_type)

        assert result is not None
        assert len(result) == 1

        card_json = result[0].to_feishu_json()
        # Find warning text in body
        body_text = str(card_json["body"])
        assert expected_cmd in body_text

    def test_orange_header_template(self):
        """Fallback card uses orange header template."""
        state = _make_state(engine_type="deep")
        result = render_fallback_card(state, "deep")

        assert result is not None
        card_json = result[0].to_feishu_json()
        assert card_json["header"]["template"] == "orange"

    def test_contains_restart_button(self):
        """Fallback card contains a restart button."""
        state = _make_state(engine_type="deep")
        result = render_fallback_card(state, "deep")

        assert result is not None
        card_json = result[0].to_feishu_json()
        # Find Schema V2 column_set with button
        button_block = next(
            (el for el in card_json["body"]["elements"] if el.get("tag") == "column_set"),
            None,
        )
        assert button_block is not None
        buttons = button_block["columns"][0]["elements"]
        assert len(buttons) >= 1
        assert "重新开始" in buttons[0]["text"]["content"]
        assert buttons[0]["behaviors"] == [
            {"type": "callback", "value": buttons[0]["value"]}
        ]

    def test_none_engine_type_uses_fallback_text(self):
        """When engine_type is None, uses generic fallback text."""
        state = _make_state()
        result = render_fallback_card(state, None)

        assert result is not None
        card_json = result[0].to_feishu_json()
        body_text = str(card_json["body"])
        assert "命令" in body_text  # fallback command word

    def test_none_state_does_not_crash(self):
        """When state is None, still produces a fallback card."""
        result = render_fallback_card(None, "deep")

        assert result is not None
        card_json = result[0].to_feishu_json()
        assert card_json["header"]["title"]["content"] == "任务"

    def test_signature_is_fallback(self):
        """Rendered card has 'fallback' signature and content_hash."""
        state = _make_state(engine_type="unknown")
        result = render_fallback_card(state, "unknown")

        assert result is not None
        assert result[0].structure_signature == "fallback"
        assert result[0].content_hash == "fallback"
