"""Contract tests for Feishu card JSON schema structure.

These tests act as guardrails for future simplification of src/card/builders/
and src/card/render/. They assert structural invariants (key existence & type)
without binding to specific values, ensuring schema correctness is preserved.

Protected regression scenarios:
- Header structure must contain title.tag="plain_text" and template color string
- Body must be a dict with "elements" list
- Top-level keys: schema="2.0", config.wide_screen_mode=True
- column_set elements must contain "columns" list with column items
- Buttons must have tag/text/type/value structure
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest


@dataclass
class _FakeProject:
    """Minimal project context for card building."""

    project_id: str = "test_proj_1"
    project_name: str = "TestProject"
    root_path: str = "/tmp/test"
    bound_chat_id: str = ""
    claude_mode: bool = False
    gemini_mode: bool = False
    coco_mode: bool = False
    codex_mode: bool = False
    aiden_mode: bool = False


def _parse_card(card_tuple: tuple[str, str]) -> dict:
    """Parse the card JSON from the (msg_type, json_str) tuple."""
    msg_type, json_str = card_tuple
    assert msg_type == "interactive", f"Expected msg_type='interactive', got '{msg_type}'"
    card = json.loads(json_str)
    return card


def _assert_top_level_structure(card: dict) -> None:
    """Assert Feishu card schema 2.0 top-level invariants."""
    assert card["schema"] == "2.0"
    assert "config" in card
    assert card["config"]["wide_screen_mode"] is True
    assert "header" in card
    assert "body" in card


def _assert_header_structure(header: dict) -> None:
    """Assert header has required fields with correct types."""
    assert "title" in header
    assert header["title"]["tag"] == "plain_text"
    assert isinstance(header["title"]["content"], str)
    assert len(header["title"]["content"]) > 0
    assert "template" in header
    assert isinstance(header["template"], str)
    assert len(header["template"]) > 0


def _assert_body_structure(body: dict) -> None:
    """Assert body contains elements list."""
    assert "elements" in body
    assert isinstance(body["elements"], list)


def _assert_column_set_valid(element: dict) -> None:
    """Assert column_set element has valid structure."""
    assert "columns" in element
    assert isinstance(element["columns"], list)
    for col in element["columns"]:
        assert col["tag"] == "column"
        assert "elements" in col
        assert isinstance(col["elements"], list)


def _assert_button_valid(button: dict) -> None:
    """Assert button has minimum required fields."""
    assert button["tag"] == "button"
    assert "text" in button
    assert button["text"]["tag"] == "plain_text"
    assert isinstance(button["text"]["content"], str)
    assert "type" in button
    assert button["type"] in ("primary", "default", "danger", "text")


def _walk_elements(elements: list[dict]):
    """Recursively walk all elements and yield (element, depth)."""
    for el in elements:
        yield el
        # Recurse into column_set -> columns -> elements
        if el.get("tag") == "column_set":
            for col in el.get("columns", []):
                yield from _walk_elements(col.get("elements", []))
        # Recurse into collapsible_panel
        if el.get("tag") == "collapsible_panel":
            yield from _walk_elements(el.get("elements", []))


class TestBuildInfoCardSchema:
    """Contract tests for DeepBuilder.build_info_card output structure."""

    @pytest.fixture
    def project(self):
        return _FakeProject()

    def _build(self, project, **kwargs):
        from src.card.builder import CardBuilder
        from src.card.models import EngineCardState

        defaults = {
            "title": "测试执行中",
            "content": "正在运行测试...",
            "engine_name": "Coco",
            "is_executing": True,
            "project_id": project.project_id,
            "show_buttons": True,
        }
        defaults.update(kwargs)
        state = EngineCardState(**defaults)
        return CardBuilder.build_info_card(project, state)

    def test_top_level_structure(self, project):
        """Card must have schema=2.0, config.wide_screen_mode, header, body."""
        card = _parse_card(self._build(project))
        _assert_top_level_structure(card)

    def test_header_structure(self, project):
        """Header must have title(plain_text) and template(color string)."""
        card = _parse_card(self._build(project))
        _assert_header_structure(card["header"])

    def test_body_elements_is_list(self, project):
        """Body.elements must be a non-empty list."""
        card = _parse_card(self._build(project))
        _assert_body_structure(card["body"])
        assert len(card["body"]["elements"]) > 0

    def test_executing_card_has_buttons(self, project):
        """When is_executing=True, card should contain button elements."""
        card = _parse_card(self._build(project, is_executing=True, show_buttons=True))
        all_elements = list(_walk_elements(card["body"]["elements"]))
        buttons = [el for el in all_elements if el.get("tag") == "button"]
        assert len(buttons) > 0, "Executing card should have control buttons"
        for btn in buttons:
            _assert_button_valid(btn)

    def test_column_set_structure(self, project):
        """All column_set elements must have valid columns structure."""
        card = _parse_card(self._build(project, is_executing=True, show_buttons=True))
        all_elements = list(_walk_elements(card["body"]["elements"]))
        column_sets = [el for el in all_elements if el.get("tag") == "column_set"]
        for cs in column_sets:
            _assert_column_set_valid(cs)

    def test_completed_card_structure(self, project):
        """Completed card should still have valid schema structure."""
        card = _parse_card(self._build(
            project,
            title="已完成",
            is_executing=False,
            is_paused=False,
        ))
        _assert_top_level_structure(card)
        _assert_header_structure(card["header"])
        _assert_body_structure(card["body"])

    def test_compact_mode_structure(self, project):
        """Compact mode card still respects schema invariants."""
        card = _parse_card(self._build(project, compact=True))
        _assert_top_level_structure(card)
        _assert_header_structure(card["header"])

    def test_expanded_mode_structure(self, project):
        """Expanded mode card still respects schema invariants."""
        card = _parse_card(self._build(project, expanded=True, content="line\n" * 100))
        _assert_top_level_structure(card)


class TestBuildErrorCardSchema:
    """Contract tests for SystemBuilder.build_error_card output structure."""

    def test_error_card_top_level(self):
        from src.card.builder import CardBuilder

        result = CardBuilder.build_error_card(
            exc="TestError",
            title="操作失败",
            project=None,
        )
        card = _parse_card(result)
        _assert_top_level_structure(card)
        _assert_header_structure(card["header"])
        _assert_body_structure(card["body"])


class TestBuildProjectResponseCardSchema:
    """Contract tests for ProjectBuilder.build_project_response_card."""

    @pytest.fixture
    def project(self):
        return _FakeProject()

    def test_project_response_card_structure(self, project):
        from src.card.builder import CardBuilder

        result = CardBuilder.build_project_response_card(
            project=project,
            title="操作完成",
            content="文件已保存",
            show_buttons=True,
        )
        card = _parse_card(result)
        _assert_top_level_structure(card)
        _assert_header_structure(card["header"])
        _assert_body_structure(card["body"])
        assert len(card["body"]["elements"]) > 0


class TestBuildToolsListCardSchema:
    """Contract tests for SystemBuilder.build_tools_list_card."""

    def test_tools_list_card_structure(self):
        from src.card.builder import CardBuilder

        tools = [
            {"name": "coco", "description": "Coco AI", "emoji": "🤖", "available": True},
            {"name": "claude", "description": "Claude AI", "emoji": "🤖", "available": True},
        ]
        result = CardBuilder.build_tools_list_card(tools, project=None)
        card = _parse_card(result)
        _assert_top_level_structure(card)
        _assert_header_structure(card["header"])
        _assert_body_structure(card["body"])
