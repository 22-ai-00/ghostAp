"""Tests for header/footer/buttons rendering."""
import json

import pytest

from src.card.render.buttons import render_buttons
from src.card.render.footer import render_footer
from src.card.state.models import ButtonSpec, CardMetadata, CardState, FooterState, HeaderState, TextBlock


class TestRenderFooter:
    def test_footer_thinking(self):
        """status=thinking → 💭 text"""
        state = CardState(footer=FooterState(status="thinking", status_text="💭 正在思考..."))
        result = render_footer(state)
        assert len(result) == 2  # hr + markdown
        assert result[0]["tag"] == "hr"
        assert result[1]["content"] == "💭 正在思考..."
        assert result[1]["text_size"] == "notation"

    def test_footer_tool_running(self):
        """Unified main-card execution flow hides duplicate tool footer text."""
        state = CardState(footer=FooterState(status="tool_running", status_text="🔧 执行中: bash"))
        result = render_footer(state)
        assert result == []

    def test_footer_with_progress(self):
        """Unified main-card execution flow keeps progress in the body."""
        state = CardState(footer=FooterState(
            status="tool_running",
            status_text="🔧 执行中: bash",
            progress="▰▰▰▱▱▱▱▱▱▱ 30%"
        ))
        result = render_footer(state)
        assert result == []

    def test_footer_none(self):
        """status=None → empty list"""
        state = CardState(footer=FooterState(status=None))
        result = render_footer(state)
        assert result == []


class TestRenderButtons:
    def test_no_buttons(self):
        """No buttons → empty list"""
        state = CardState(buttons=())
        result = render_buttons(state)
        assert result == []

    def test_single_button_action_block(self):
        """1 button → column_set with flex_mode 'none' (full width for mobile accessibility)"""
        state = CardState(buttons=(
            ButtonSpec(text="停止", action_id="stop", type="danger"),
        ))
        result = render_buttons(state)
        assert len(result) == 1
        assert result[0]["tag"] == "column_set"
        assert result[0]["flex_mode"] == "none"
        assert len(result[0]["columns"]) == 1
        assert result[0]["columns"][0]["width"] == "weighted"
        assert result[0]["columns"][0]["weight"] == 1
        assert result[0]["columns"][0]["elements"][0]["text"]["content"] == "停止"
        button = result[0]["columns"][0]["elements"][0]
        assert button["behaviors"] == [
            {"type": "callback", "value": button["value"]}
        ]

    def test_two_buttons_column_set(self):
        """2 buttons → column_set layout with bisect"""
        state = CardState(buttons=(
            ButtonSpec(text="停止", action_id="stop", type="danger"),
            ButtonSpec(text="继续", action_id="continue", type="primary"),
        ))
        result = render_buttons(state)
        assert len(result) == 1
        assert result[0]["tag"] == "column_set"
        assert len(result[0]["columns"]) == 2
        assert result[0]["flex_mode"] == "bisect"

    def test_many_buttons_schema_v2_layout(self):
        """3+ buttons avoid the Schema V2-incompatible action container."""
        from src.card.render.budget import RenderBudget
        budget = RenderBudget(mobile_force_vertical=True)
        # 3 buttons with mobile_force_vertical=True → vertical column_set
        state = CardState(buttons=(
            ButtonSpec(text="A", action_id="a"),
            ButtonSpec(text="B", action_id="b"),
            ButtonSpec(text="C", action_id="c"),
        ))
        result = render_buttons(state, budget=budget)
        assert len(result) == 1
        assert result[0]["tag"] == "column_set"
        assert result[0]["flex_mode"] == "none"
        assert len(result[0]["columns"]) == 1
        assert len(result[0]["columns"][0]["elements"]) == 3

        # 4 buttons → two Schema V2-compatible rows
        state4 = CardState(buttons=(
            ButtonSpec(text="A", action_id="a"),
            ButtonSpec(text="B", action_id="b"),
            ButtonSpec(text="C", action_id="c"),
            ButtonSpec(text="D", action_id="d"),
        ))
        result4 = render_buttons(state4)
        assert len(result4) == 2
        assert all(row["tag"] == "column_set" for row in result4)
        assert all(row["flex_mode"] == "bisect" for row in result4)
        assert sum(len(row["columns"]) for row in result4) == 4

    def test_button_with_confirm(self):
        """Button with confirm → confirm dialog"""
        state = CardState(buttons=(
            ButtonSpec(text="删除", action_id="delete", type="danger", confirm="确定要删除吗？"),
            ButtonSpec(text="取消", action_id="cancel"),
        ))
        result = render_buttons(state)
        # Find the delete button
        columns = result[0]["columns"]
        delete_btn = columns[0]["elements"][0]
        assert "confirm" in delete_btn
        assert delete_btn["confirm"]["text"]["content"] == "确定要删除吗？"


# ---------------------------------------------------------------------------
# Phase 5: render_progress_bar boundary tests
# ---------------------------------------------------------------------------
from src.card.render.progress import render_progress_bar


class TestRenderProgressBarBoundary:
    """Boundary value tests for render_progress_bar."""

    @pytest.mark.parametrize(
        "pct, total_segments_kwarg, expected_substrings, forbidden_substrings",
        [
            (0, None, ["▱▱▱▱▱"], ["▰"]),
            (100, None, ["▰▰▰▰▰"], ["▱"]),
            (150, None, ["▰▰▰▰▰"], []),
            (50, None, ["▰", "▱", "50%"], []),
            (50, 0, [""], []),
        ],
        ids=[
            "test_pct_zero",
            "test_pct_hundred",
            "test_pct_over_hundred_clamps",
            "test_pct_midpoint",
            "test_total_segments_zero_returns_empty",
        ],
    )
    def test_progress_bar_boundary(
        self, pct, total_segments_kwarg, expected_substrings, forbidden_substrings
    ):
        if total_segments_kwarg is None:
            result = render_progress_bar(pct)
        else:
            result = render_progress_bar(pct, total_segments=total_segments_kwarg)
        # When expected substring is "", the result must be exactly empty string
        if expected_substrings == [""]:
            assert result == ""
        else:
            for sub in expected_substrings:
                assert sub in result
        for forbidden in forbidden_substrings:
            assert forbidden not in result


# ---------------------------------------------------------------------------
# Phase 5: Footer warning_banner + progress_pct coexistence
# ---------------------------------------------------------------------------

class TestFooterWarningAndProgressCoexist:
    """Warning banner rendered in body top; footer only has status + progress."""

    def test_footer_has_no_banner_warning_type(self):
        """Banner is now in body top, not footer — footer should only have status+progress."""
        state = CardState(
            footer=FooterState(
                status="tool_running",
                status_text="⏳ 编码中",
                progress="步骤 3/6",
                progress_pct=50,
                warning_banner="注意：资源即将耗尽",
                warning_type="warning",
            ),
        )
        elements = render_footer(state)
        # Footer should NOT contain any banner div (moved to body top)
        banner_divs = [e for e in elements if e.get("tag") == "div"]
        assert len(banner_divs) == 0
        # Unified main cards render tool/progress state in the execution body.
        assert elements == []

# ---------------------------------------------------------------------------
# Banner unified position tests (all levels in body top)
# ---------------------------------------------------------------------------
from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card


class TestBannerUnifiedPosition:
    """All banner types (error/warning/info/success) render at body_elements[0]."""

    def _render_with_banner(self, warning_type: str):
        state = CardState(
            blocks=(TextBlock(kind="text", block_id="b1", content="hello", status="completed"),),
            footer=FooterState(
                status="idle",
                status_text="ready",
                warning_banner="Test banner message",
                warning_type=warning_type,
            ),
        )
        cards = render_card(state, RenderBudget())
        return cards[0]._card_json["body"]["elements"]

    @pytest.mark.parametrize(
        "warning_type, expected_style",
        [
            ("error", "red"),
            ("warning", "yellow"),
            ("info", "wathet"),
            ("success", "green"),
        ],
        ids=[
            "test_error_banner_in_body_top",
            "test_warning_banner_in_body_top",
            "test_info_banner_in_body_top",
            "test_success_banner_in_body_top",
        ],
    )
    def test_banner_in_body_top(self, warning_type, expected_style):
        elements = self._render_with_banner(warning_type)
        assert elements[0]["tag"] == "column_set"
        assert elements[0]["background_style"] == expected_style
        assert "Test banner message" in elements[0]["columns"][0]["elements"][0]["content"]

# ---------------------------------------------------------------------------
# Task 5: Multi-page banner appears on ALL pages
# ---------------------------------------------------------------------------


class TestBannerMultiPagePosition:
    """Verify that warning banner appears only on the first page in multi-page cards."""

    def test_banner_on_first_page_only_with_large_content(self):
        """Construct state with enough content to trigger pagination + warning_banner."""
        # Generate many blocks to exceed single-page budget
        blocks = tuple(
            TextBlock(
                kind="text",
                block_id=f"b_{i}",
                content=f"{'x' * 500}\n" * 5,  # ~2500 chars per block
                status="completed",
            )
            for i in range(20)  # 20 blocks × 2500 chars = ~50000 chars → multi-page
        )
        state = CardState(
            blocks=blocks,
            footer=FooterState(
                status="idle",
                warning_banner="⚠️ 注意：系统负载较高",
                warning_type="warning",
            ),
        )
        cards = render_card(state, RenderBudget())

        # Should produce multiple pages
        assert len(cards) > 1, f"Expected multi-page but got {len(cards)} page(s)"

        # Verify banner appears only on the FIRST page
        first_body = cards[0]._card_json["body"]["elements"]
        first_elem = first_body[0]
        assert first_elem["tag"] == "column_set", (
            f"Page 0: first element should be banner column_set, got {first_elem.get('tag')}"
        )
        assert first_elem["background_style"] == "yellow", (
            "Page 0: warning banner should be yellow"
        )
        # Verify banner text
        banner_text = json.dumps(first_elem, ensure_ascii=False)
        assert "注意：系统负载较高" in banner_text, (
            "Page 0: banner text not found"
        )

        # Verify banner does NOT appear on subsequent pages
        for i, card in enumerate(cards[1:], start=1):
            body_elements = card._card_json["body"]["elements"]
            first_elem = body_elements[0]
            # Should be content (markdown), not a banner div with background_style
            assert first_elem.get("background_style") != "orange", (
                f"Page {i}: banner should NOT appear on non-first pages"
            )


# ---------------------------------------------------------------------------
# AC-2: render_buttons stop intent → flex_mode == "none"
# ---------------------------------------------------------------------------


class TestToolStatusIcons:
    """Verify tool status icons use emoji style."""

    def test_status_icons_are_emoji(self):
        from src.card.render.tools import _STATUS_ICONS
        assert _STATUS_ICONS["completed"] == "✅"
        assert _STATUS_ICONS["failed"] == "❌"
        assert _STATUS_ICONS["active"] == "⏳"


class TestCriteriaPanelIcon:
    """Verify criteria panel has standard collapsible icon config."""

    def test_criteria_panel_has_icon_config(self):
        from src.card.render.atoms import RenderAtom
        from src.card.render.renderer import _render_criteria_panel
        from src.card.state.models import CardState

        atom = RenderAtom(kind="criteria", content="- [x] Passes")
        state = CardState(metadata=CardMetadata(expand_ac=True))
        result = _render_criteria_panel(atom, state)

        header = result["header"]
        assert "icon" in header
        assert header["icon"]["token"] == "down-small-ccm_outlined"
        assert header["icon_position"] == "follow_text"
        assert header["icon_expanded_angle"] == -180


class TestFooterBlockedReason:
    """Footer renders blocked reason via UI_TEXT key."""

    def test_blocked_reason_renders_in_footer(self):

        from src.card.render.footer import render_footer
        from src.card.state.models import CardState, EngineExtState, FooterState

        meta = CardMetadata(engine_type="deep", mode_name="Deep", mode_emoji="🔍")
        state = CardState(
            metadata=meta,
            terminal="blocked",
            header=HeaderState(),
            footer=FooterState(status="idle"),
            engine_ext=EngineExtState(blocked_reason="需要人工确认"),
        )
        elements = render_footer(state)
        texts = [e.get("content", "") for e in elements if e.get("tag") == "markdown"]
        assert any("需要人工确认" in t for t in texts)
        assert any("任务阻塞" in t for t in texts)
