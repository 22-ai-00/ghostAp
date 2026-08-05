"""Dedicated tests for build_role_info_card.

Covers:
- Assign task form has UUID suffix (uniqueness per chat)
- Quick action buttons include correct action_type values
- Card renders without crash for all AgentStatus values
- Skill profiles sort by success_rate descending, capped at 8
"""

from __future__ import annotations

import json

from src.slock_engine.card_templates.role import build_role_info_card
from src.slock_engine.models import AgentIdentity, AgentStatus


def _make_agent(**kwargs) -> AgentIdentity:
    defaults = {"agent_id": "test-agent-1", "name": "TestAgent", "emoji": "🤖", "role": "coder"}
    defaults.update(kwargs)
    return AgentIdentity(**defaults)


def _find_form_recursive(elements: list) -> dict | None:
    """Search elements tree recursively for a form tag (may be inside collapsible_panel)."""
    for el in elements:
        if el.get("tag") == "form":
            return el
        # collapsible_panel wraps children in "elements"
        if el.get("tag") == "collapsible_panel":
            inner = el.get("elements", [])
            found = _find_form_recursive(inner)
            if found:
                return found
    return None


class TestRoleInfoCardFormUniqueness:
    """Verify form names use deterministic agent_id-based naming."""

    def test_form_name_has_agent_id(self):
        agent = _make_agent()
        card = build_role_info_card(agent, status=AgentStatus.IDLE)

        form_el = _find_form_recursive(card["body"]["elements"])
        assert form_el is not None

        name = form_el["name"]
        # Pattern: assign_task_{agent_id}
        assert name == "assign_task_test-agent-1"

class TestRoleInfoCardQuickActions:
    """Verify quick action buttons have correct action_type values."""

    def test_quick_action_buttons_present(self):
        agent = _make_agent()
        card = build_role_info_card(agent, status=AgentStatus.IDLE, channel_id="ch-1")

        serialized = json.dumps(card, ensure_ascii=False)
        assert "slock_agent_show_memory" in serialized
        assert "slock_start_discussion" in serialized

    def test_action_value_contains_agent_id_and_channel(self):
        agent = _make_agent(agent_id="agent-xyz")
        card = build_role_info_card(agent, status=AgentStatus.IDLE, channel_id="ch-abc")

        serialized = json.dumps(card, ensure_ascii=False)
        assert "agent-xyz" in serialized
        assert "ch-abc" in serialized


class TestRoleInfoCardAllStatuses:
    """Card renders without error for every AgentStatus value."""

    def test_renders_for_every_status(self):
        for status in AgentStatus:
            card = build_role_info_card(_make_agent(), status=status)
            assert "header" in card, status
            assert "body" in card, status


class TestRoleInfoCardSkillCap:
    """Skill profiles sorted desc by success_rate and capped at 8."""

    def test_skills_capped_at_8(self):
        agent = _make_agent()
        skills = [{"tag": f"skill-{i}", "success_rate": float(i * 10)} for i in range(12)]

        card = build_role_info_card(agent, status=AgentStatus.IDLE, skill_profiles=skills)
        serialized = json.dumps(card, ensure_ascii=False)

        # Skills 4-11 (top 8 by rate) should appear; skill-0..3 should not
        assert "`skill-11`" in serialized  # highest rate
        assert "`skill-4`" in serialized   # 8th highest
        assert "`skill-3`" not in serialized  # 9th — excluded


class TestDefaultPersonalityTraitsMapping:
    """Verify DEFAULT_PERSONALITY_TRAITS returns correct values for each role."""

    def test_role_defaults_match_product_copy(self):
        from src.feishu.handlers.slock import SlockHandler

        assert SlockHandler.DEFAULT_PERSONALITY_TRAITS == {
            "coder": ["严谨", "注重细节"],
            "reviewer": ["批判性思维", "追求质量"],
            "tester": ["细致", "追求覆盖"],
            "planner": ["全局视角", "有条理"],
            "architect": ["抽象思维", "系统设计"],
            "writer": ["表达清晰", "注重结构"],
            "custom": [],
        }
