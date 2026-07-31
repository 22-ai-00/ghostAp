"""Unit tests for Slock card UX audit changes.

Covers:
- Task board card refresh button presence and action value
- Welcome card command list completeness
"""

from __future__ import annotations

from src.slock_engine.card_templates import build_task_board_card, build_welcome_card
from src.slock_engine.models import AgentIdentity, SlockTask, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(content: str = "test task", status: TaskStatus = TaskStatus.TODO) -> SlockTask:
    return SlockTask(content=content, status=status)


def _make_agent(agent_id: str = "a1", name: str = "Dev") -> AgentIdentity:
    return AgentIdentity(agent_id=agent_id, name=name, emoji="🤖")


def _find_buttons(elements: list[dict]) -> list[dict]:
    """Recursively extract all button elements from card body elements."""
    buttons: list[dict] = []
    for el in elements:
        if el.get("tag") == "button":
            buttons.append(el)
        # Check inside column_set → columns → elements
        if el.get("tag") == "column_set":
            for col in el.get("columns", []):
                buttons.extend(_find_buttons(col.get("elements", [])))
        # Check inside responsive layouts (action elements)
        if el.get("tag") == "action":
            buttons.extend(_find_buttons(el.get("actions", [])))
        # Check nested elements list
        if "elements" in el and isinstance(el["elements"], list):
            buttons.extend(_find_buttons(el["elements"]))
    return buttons


# ---------------------------------------------------------------------------
# Task Board Refresh Button
# ---------------------------------------------------------------------------


class TestTaskBoardRefreshButton:
    """Task board card includes a refresh button with correct action."""

    def test_refresh_button_present(self):
        """build_task_board_card includes a button with text '🔄 刷新'."""
        card = build_task_board_card(
            [_make_task()], [_make_agent()], team_name="Team1", channel_id="ch123"
        )
        elements = card["body"]["elements"]
        buttons = _find_buttons(elements)
        refresh_buttons = [b for b in buttons if "刷新" in b.get("text", {}).get("content", "")]
        assert len(refresh_buttons) >= 1, f"Expected refresh button, found buttons: {buttons}"

    def test_refresh_button_action_value(self):
        """Refresh button value contains action=slock_refresh_task_board and channel_id."""
        card = build_task_board_card(
            [_make_task()], [_make_agent()], team_name="Team1", channel_id="ch-abc"
        )
        elements = card["body"]["elements"]
        buttons = _find_buttons(elements)
        refresh_buttons = [b for b in buttons if "刷新" in b.get("text", {}).get("content", "")]
        assert refresh_buttons
        btn = refresh_buttons[0]
        value = btn.get("value", {})
        assert value.get("action") == "slock_refresh_task_board"
        assert value.get("channel_id") == "ch-abc"

    def test_refresh_button_has_callback_behavior(self):
        """Refresh button uses callback behavior for interactive card schema v2."""
        card = build_task_board_card(
            [], [], team_name="T", channel_id="ch1"
        )
        elements = card["body"]["elements"]
        buttons = _find_buttons(elements)
        refresh_buttons = [b for b in buttons if "刷新" in b.get("text", {}).get("content", "")]
        assert refresh_buttons
        btn = refresh_buttons[0]
        behaviors = btn.get("behaviors", [])
        assert any(b.get("type") == "callback" for b in behaviors)

    def test_empty_channel_id_still_has_button(self):
        """Refresh button is present even when channel_id is empty string."""
        card = build_task_board_card([], [], team_name="T", channel_id="")
        elements = card["body"]["elements"]
        buttons = _find_buttons(elements)
        refresh_buttons = [b for b in buttons if "刷新" in b.get("text", {}).get("content", "")]
        assert len(refresh_buttons) >= 1


# ---------------------------------------------------------------------------
# Welcome Card Commands
# ---------------------------------------------------------------------------


class TestWelcomeCardCommands:
    """Welcome card lists all required commands."""

    def test_has_hire_command(self):
        card = build_welcome_card(team_name="Alpha")
        content = self._get_markdown_content(card)
        assert "/hire" in content

    def test_has_role_add_command(self):
        card = build_welcome_card(team_name="Alpha")
        content = self._get_markdown_content(card)
        assert "/role add" in content

    def test_has_slock_status_command(self):
        card = build_welcome_card(team_name="Alpha")
        content = self._get_markdown_content(card)
        assert "/slock status" in content

    def test_does_not_advertise_retired_goal_command(self):
        card = build_welcome_card(team_name="Alpha")
        content = self._get_markdown_content(card)
        assert "/goal" not in content

    def test_has_role_list_command(self):
        """AC: welcome card includes /role list command."""
        card = build_welcome_card(team_name="Alpha")
        content = self._get_markdown_content(card)
        assert "/role list" in content

    def test_has_task_status_command(self):
        """AC: welcome card includes task status NL example."""
        card = build_welcome_card(team_name="Alpha")
        content = self._get_markdown_content(card)
        # NL-first card uses natural language; verify task-related NL example
        assert "任务" in content or "/task assign" in content

    def test_team_name_in_header(self):
        card = build_welcome_card(team_name="MyTeam")
        header_text = card["header"]["title"]["content"]
        assert "MyTeam" in header_text

    @staticmethod
    def _get_markdown_content(card: dict) -> str:
        """Join all markdown element content from the card body."""
        parts: list[str] = []
        for el in card.get("body", {}).get("elements", []):
            if el.get("tag") == "markdown":
                parts.append(el.get("content", ""))
        return "\n".join(parts)


class TestNLICardVisual:
    """AC11: NLI confirmation card visual."""

    def test_nli_card_has_thinking_emoji(self):
        """NLI feedback card should use 🤔 emoji, not ⚠️."""
        from src.slock_engine.card_templates import build_nli_feedback_card

        card = build_nli_feedback_card(
            intent_description="创建新角色",
            channel_id="ch1",
            intent_params={"action": "new_role", "params": {}},
        )
        header_content = card["header"]["title"]["content"]
        assert "🤔" in header_content
        assert "⚠️" not in header_content

    def test_nli_card_title_is_intent_recognition(self):
        """NLI card title should be '意图识别', not '意图确认'."""
        from src.slock_engine.card_templates import build_nli_feedback_card

        card = build_nli_feedback_card(
            intent_description="查看状态",
            channel_id="ch1",
            intent_params={"action": "status", "params": {}},
        )
        header_content = card["header"]["title"]["content"]
        assert "意图识别" in header_content
        assert "意图确认" not in header_content
