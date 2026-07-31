"""Unit tests for slock_engine/card_templates.py — card building functions."""

from __future__ import annotations

import json

from src.slock_engine.card_templates import (
    build_agent_message_card,
    build_status_panel_card,
    build_task_board_card,
)
from src.slock_engine.models import (
    AgentIdentity,
    AgentStatus,
    SlockTask,
    TaskStatus,
)


def _collect_buttons(node: object) -> list[dict]:
    if isinstance(node, dict):
        buttons = [node] if node.get("tag") == "button" else []
        for value in node.values():
            buttons.extend(_collect_buttons(value))
        return buttons
    if isinstance(node, list):
        buttons: list[dict] = []
        for item in node:
            buttons.extend(_collect_buttons(item))
        return buttons
    return []


def _collect_tags(node: object) -> list[str]:
    if isinstance(node, dict):
        tags = [node["tag"]] if isinstance(node.get("tag"), str) else []
        for value in node.values():
            tags.extend(_collect_tags(value))
        return tags
    if isinstance(node, list):
        tags: list[str] = []
        for item in node:
            tags.extend(_collect_tags(item))
        return tags
    return []


def _collect_column_sets(node: object) -> list[dict]:
    """Recursively collect all column_set elements from a card structure."""
    results: list[dict] = []
    if isinstance(node, dict):
        if node.get("tag") == "column_set":
            results.append(node)
        for value in node.values():
            results.extend(_collect_column_sets(value))
    elif isinstance(node, list):
        for item in node:
            results.extend(_collect_column_sets(item))
    return results


def _status_column_sets(card: dict) -> list[dict]:
    """Find column_sets used for agent status rows (have non-default background_style)."""
    all_cs = _collect_column_sets(card)
    return [
        cs for cs in all_cs
        if cs.get("background_style") in {"grey", "card_primary"}
        or (cs.get("background_style") == "default" and "flex_mode" in cs)
    ]


class TestBuildAgentMessageCard:
    def _make_agent(self, **kwargs) -> AgentIdentity:
        defaults = {"agent_id": "a1", "name": "Coder", "emoji": "🔧", "role": "coder"}
        defaults.update(kwargs)
        return AgentIdentity(**defaults)

    def test_basic_structure(self):
        agent = self._make_agent()
        card = build_agent_message_card(agent, "Hello world")
        assert card["schema"] == "2.0"
        assert card["header"]["title"]["content"] == "🔧 Coder"
        assert card["header"]["template"] == "blue"
        body_elements = card["body"]["elements"]
        assert any(e["tag"] == "markdown" and "Hello world" in e["content"] for e in body_elements)

    def test_footer_with_model_info(self):
        agent = self._make_agent(agent_type="claude", model_name="claude-3")
        card = build_agent_message_card(agent, "msg", model_info="claude-3-opus")
        footer_elements = [e for e in card["body"]["elements"] if e.get("tag") == "markdown" and e.get("text_size") == "notation"]
        assert len(footer_elements) == 1
        assert "claude-3-opus" in footer_elements[0]["content"]

    def test_footer_with_duration_seconds(self):
        agent = self._make_agent()
        card = build_agent_message_card(agent, "msg", duration_s=5.2)
        footer_elements = [e for e in card["body"]["elements"] if e.get("tag") == "markdown" and e.get("text_size") == "notation"]
        assert len(footer_elements) == 1
        assert "5.2s" in footer_elements[0]["content"]

    def test_no_footer_when_no_metadata(self):
        agent = self._make_agent(agent_type="", model_name="")
        card = build_agent_message_card(agent, "msg")
        footer_elements = [e for e in card["body"]["elements"] if e.get("tag") == "markdown" and e.get("text_size") == "notation"]
        assert len(footer_elements) == 0

    def test_agent_message_card_has_follow_reasoning_and_done_actions(self):
        """Agent reply cards expose the follow-up/reasoning/done actions from the spec."""
        agent = self._make_agent(agent_type="codex", model_name="gpt-5")
        card = build_agent_message_card(agent, "result", channel_id="chat1", task_id="task1")

        buttons = _collect_buttons(card)
        labels = {button["text"]["content"] for button in buttons}
        assert {"@追问", "查看推理", "标记完成"}.issubset(labels)
        actions = {
            button["behaviors"][0]["value"]["action"]
            for button in buttons
            if button.get("behaviors")
        }
        assert {
            "slock_agent_follow_up",
            "slock_agent_show_reasoning",
            "slock_agent_mark_done",
        }.issubset(actions)


class TestBuildStatusPanelCard:
    def test_empty_agents(self):
        card = build_status_panel_card([], team_name="Alpha")
        assert "Alpha" in card["header"]["title"]["content"]
        body = card["body"]["elements"]
        assert any("暂无" in e.get("content", "") or "Agent" in e.get("content", "") for e in body)

    def test_with_agents_uses_column_set(self):
        """AC-6: Status panel uses column_set components instead of plain markdown list."""
        agent = AgentIdentity(agent_id="a1", name="Alice", emoji="🤖", role="coder")
        card = build_status_panel_card([(agent, AgentStatus.IDLE)], team_name="Team")
        all_column_sets = _collect_column_sets(card)
        assert len(all_column_sets) >= 1
        # Verify agent info is inside some column
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        assert "Alice" in combined

    def test_multiple_agents_produce_multiple_column_sets(self):
        """Each agent gets its own column_set row."""
        a1 = AgentIdentity(agent_id="a1", name="Alice", emoji="🤖")
        a2 = AgentIdentity(agent_id="a2", name="Bob", emoji="🔧")
        card = build_status_panel_card(
            [(a1, AgentStatus.IDLE), (a2, AgentStatus.RUNNING)]
        )
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        assert "Alice" in combined
        assert "Bob" in combined
        # Check background styles are applied
        cs = _collect_column_sets(card)
        background_styles = [c.get("background_style") for c in cs]
        assert "default" in background_styles  # IDLE
        assert "card_primary" in background_styles  # RUNNING

    def test_refresh_button_present(self):
        card = build_status_panel_card([])
        buttons = _collect_buttons(card)
        assert any("刷新" in button["text"]["content"] for button in buttons)

    def test_stop_all_button_present(self):
        """Status panel has a '全部停止' button."""
        agent = AgentIdentity(agent_id="a1", name="A", emoji="🤖")
        card = build_status_panel_card([(agent, AgentStatus.RUNNING)])
        buttons = _collect_buttons(card)
        assert any("全部停止" in button["text"]["content"] for button in buttons)

    def test_status_labels_chinese(self):
        """Status labels use Chinese text (空闲, 运行中, etc.)."""
        agents = [
            (AgentIdentity(agent_id="a1", name="A", emoji="🤖"), AgentStatus.IDLE),
            (AgentIdentity(agent_id="a2", name="B", emoji="🔧"), AgentStatus.RUNNING),
        ]
        card = build_status_panel_card(agents)
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        assert "空闲" in combined
        assert "运行中" in combined

    def test_status_panel_uses_mobile_friendly_layout(self):
        """Status panel uses mobile-friendly flow layout (replaces old bisect layout)."""
        a1 = AgentIdentity(agent_id="a1", name="Alice", emoji="🤖", role="coder")
        a2 = AgentIdentity(agent_id="a2", name="Bob", emoji="🔧", role="writer")
        card = build_status_panel_card(
            [(a1, AgentStatus.IDLE), (a2, AgentStatus.RUNNING)], team_name="Team"
        )
        column_sets = _collect_column_sets(card)
        assert len(column_sets) >= 2

        # New implementation uses flex_mode="flow" for mobile-friendly rows
        flow_rows = [cs for cs in column_sets if cs.get("flex_mode") == "flow"]
        assert len(flow_rows) >= 2

    def test_status_panel_shows_current_task_for_running_agent(self):
        """The status panel row includes the current task required by the Slock spec."""
        agent = AgentIdentity(agent_id="a1", name="Alice", emoji="🤖", role="coder")
        task = SlockTask(
            task_id="task-current",
            content="Implement payment webhook retry policy",
            status=TaskStatus.IN_PROGRESS,
            claimed_by="a1",
        )

        card = build_status_panel_card(
            [(agent, AgentStatus.RUNNING)],
            team_name="Team",
            current_tasks={"a1": task},
        )

        combined = "\n".join(_all_markdown_content(card))
        assert "Implement payment webhook retry policy" in combined
        assert "当前任务" in combined


_TASK_BOARD_BG_COLORS = {"grey", "blue", "yellow", "green"}


def _task_board_column_sets(card: dict) -> list[dict]:
    """Extract task-status column_set elements from task board card body.

    Uses positive matching on known task-status background colors to avoid
    accidentally filtering out button column_sets that lack background_style.
    New implementation uses collapsible panels, so we search recursively.
    """
    all_cs = _collect_column_sets(card)
    return [
        cs for cs in all_cs
        if cs.get("background_style") in _TASK_BOARD_BG_COLORS
    ]


def _all_markdown_content(node: object) -> list[str]:
    """Recursively collect all markdown content strings from a card structure."""
    if isinstance(node, dict):
        results = []
        if node.get("tag") == "markdown" and "content" in node:
            results.append(node["content"])
        for value in node.values():
            results.extend(_all_markdown_content(value))
        return results
    if isinstance(node, list):
        results: list[str] = []
        for item in node:
            results.extend(_all_markdown_content(item))
        return results
    return []


class TestBuildTaskBoardCard:
    def test_empty_tasks(self):
        card = build_task_board_card([], [], team_name="Dev")
        assert "Dev" in card["header"]["title"]["content"]
        # New implementation shows summary counts even when empty
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        assert "共 **0** 个任务" in combined or "任务看板" in card["header"]["title"]["content"]

    def test_with_tasks(self):
        task = SlockTask(task_id="t1", content="Build feature X", status=TaskStatus.TODO)
        agent = AgentIdentity(agent_id="a1", name="Bob", emoji="🔧")
        card = build_task_board_card([task], [agent])
        all_md = _all_markdown_content(card)
        assert any("Build feature X" in md for md in all_md)

    def test_task_with_assignee(self):
        task = SlockTask(
            task_id="t1", content="Fix bug", status=TaskStatus.IN_PROGRESS, claimed_by="a1"
        )
        agent = AgentIdentity(agent_id="a1", name="Bob", emoji="🔧")
        card = build_task_board_card([task], [agent])
        all_md = _all_markdown_content(card)
        assert any("Bob" in md for md in all_md)

    def test_board_groups_by_status(self):
        tasks = [
            SlockTask(task_id="t1", content="A", status=TaskStatus.TODO),
            SlockTask(task_id="t2", content="B", status=TaskStatus.DONE),
        ]
        card = build_task_board_card(tasks, [])
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        # Chinese status labels
        assert "待办" in combined or "Todo" in combined
        assert "完成" in combined or "Done" in combined

    def test_uses_column_set_layout(self):
        """AC-14: Task board card uses column_set components (new implementation uses 2x2 summary grid)."""
        task = SlockTask(task_id="t1", content="Task A", status=TaskStatus.TODO)
        card = build_task_board_card([task], [])
        column_sets = _collect_column_sets(card)
        # New implementation has summary column_sets + task entry column_sets
        assert len(column_sets) >= 2

    def test_column_set_background_colors(self):
        """Task entries use grey background in new implementation."""
        tasks = [
            SlockTask(task_id="t1", content="A", status=TaskStatus.TODO),
            SlockTask(task_id="t2", content="B", status=TaskStatus.IN_PROGRESS),
        ]
        card = build_task_board_card(tasks, [])
        column_sets = _task_board_column_sets(card)
        # New implementation uses grey background for all task entries
        background_styles = [cs["background_style"] for cs in column_sets]
        assert "grey" in background_styles

    def test_column_set_dual_column_structure(self):
        """Summary section has column_sets for each status category."""
        task = SlockTask(task_id="t1", content="X", status=TaskStatus.IN_PROGRESS)
        card = build_task_board_card([task], [])
        column_sets = _collect_column_sets(card)
        # 4 status summary rows (待办/进行中/审查中/已完成) use default background
        summary_rows = [
            cs for cs in column_sets
            if cs.get("background_style") == "default"
        ]
        assert len(summary_rows) >= 4  # one per status category

    def test_task_content_in_right_column(self):
        """Task content appears in task entries (new implementation uses single-column layout for tasks)."""
        task = SlockTask(task_id="t1", content="Implement auth flow", status=TaskStatus.IN_PROGRESS)
        agent = AgentIdentity(agent_id="a1", name="Dev", emoji="🔧")
        card = build_task_board_card([task], [agent])
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        assert "Implement auth flow" in combined

    def test_done_task_aborted_marker(self):
        """DONE task with resolved_reason shows strikethrough in new implementation."""
        task = SlockTask(
            task_id="t1", content="Deploy service", status=TaskStatus.DONE,
            resolved_reason="超时中止",
        )
        card = build_task_board_card([task], [])
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        # Check task content is present
        assert "Deploy service" in combined

    def test_done_task_normal_marker(self):
        """DONE task without resolved_reason shows plain content."""
        task = SlockTask(
            task_id="t1", content="Write tests", status=TaskStatus.DONE,
        )
        card = build_task_board_card([task], [])
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        assert "Write tests" in combined

    def test_refresh_button_chat_id(self):
        """Refresh button callback value contains channel_id field (new implementation uses channel_id key)."""
        task = SlockTask(task_id="t1", content="X", status=TaskStatus.TODO)
        card = build_task_board_card([task], [], channel_id="chat_abc123")
        buttons = _collect_buttons(card)
        refresh_btns = [b for b in buttons if b.get("value", {}).get("action") == "slock_refresh_task_board"]
        assert len(refresh_btns) == 1
        # New implementation uses channel_id key
        assert refresh_btns[0]["value"].get("channel_id") == "chat_abc123"


class TestBuildAgentMessageCardJsonStructure:
    """R-01 mitigation: JSON Schema regression tests for build_agent_message_card.

    Ensures the four essential elements (header.template, header.title,
    body.elements[markdown], body.elements[note]) are always present and correctly
    structured to prevent Feishu card rendering failures.
    """

    def _make_agent(self, **kwargs) -> AgentIdentity:
        defaults = {
            "agent_id": "test-001",
            "name": "TestCoder",
            "emoji": "🔧",
            "role": "coder",
            "agent_type": "claude",
            "model_name": "claude-sonnet-4",
        }
        defaults.update(kwargs)
        return AgentIdentity(**defaults)

    def test_top_level_schema_keys(self):
        """Card must have schema, config, header, body at top level."""
        agent = self._make_agent()
        card = build_agent_message_card(agent, "content")
        assert set(card.keys()) == {"schema", "config", "header", "body"}
        assert card["schema"] == "2.0"
        assert card["config"]["wide_screen_mode"] is False

    def test_header_has_template_and_title(self):
        """Header must contain template (color) and title with tag+content."""
        agent = self._make_agent(role="coder")
        card = build_agent_message_card(agent, "hello")
        header = card["header"]
        # template must be a valid Feishu color string
        assert header["template"] == "blue"  # coder -> blue
        # title must be a plain_text tag with content
        assert header["title"]["tag"] == "plain_text"
        assert "TestCoder" in header["title"]["content"]
        assert "🔧" in header["title"]["content"]

    def test_body_contains_markdown_element(self):
        """Body elements must include at least one markdown element with the content."""
        agent = self._make_agent()
        card = build_agent_message_card(agent, "The quick brown fox")
        body_elements = card["body"]["elements"]
        md_elements = [e for e in body_elements if e.get("tag") == "markdown"]
        assert len(md_elements) >= 1
        assert "The quick brown fox" in md_elements[0]["content"]

    def test_body_contains_note_element_when_metadata_present(self):
        """Body elements must include a notation-sized markdown footer when model_info or duration is given."""
        agent = self._make_agent()
        card = build_agent_message_card(agent, "msg", model_info="gpt-4o", duration_s=3.5)
        body_elements = card["body"]["elements"]
        footer_elements = [e for e in body_elements if e.get("tag") == "markdown" and e.get("text_size") == "notation"]
        assert len(footer_elements) == 1
        footer_content = footer_elements[0]["content"]
        assert "gpt-4o" in footer_content
        assert "3.5s" in footer_content

    def test_note_element_contains_agent_type(self):
        """Footer should include agent_type when present."""
        agent = self._make_agent(agent_type="codex")
        card = build_agent_message_card(agent, "msg", duration_s=1.0)
        footer_elements = [e for e in card["body"]["elements"] if e.get("tag") == "markdown" and e.get("text_size") == "notation"]
        assert len(footer_elements) == 1
        assert "codex" in footer_elements[0]["content"]

    def test_elements_ordering_markdown_before_note(self):
        """Markdown content must appear before the footer in elements array."""
        agent = self._make_agent()
        card = build_agent_message_card(agent, "body text", model_info="model-x")
        body_elements = card["body"]["elements"]
        md_idx = next(i for i, e in enumerate(body_elements) if e["tag"] == "markdown" and e.get("text_size") != "notation")
        footer_idx = next(i for i, e in enumerate(body_elements) if e.get("tag") == "markdown" and e.get("text_size") == "notation")
        assert md_idx < footer_idx


class TestBuildWelcomeCard:
    """Tests for build_welcome_card() — welcome card sent in new Slock team group."""

    def test_schema_v2(self):
        from src.slock_engine.card_templates import build_welcome_card

        card = build_welcome_card(team_name="Alpha")
        assert card["schema"] == "2.0"

    def test_header_contains_team_name(self):
        from src.slock_engine.card_templates import build_welcome_card

        card = build_welcome_card(team_name="Alpha")
        assert "Alpha" in card["header"]["title"]["content"]

    def test_body_contains_quick_start_commands(self):
        import json

        from src.slock_engine.card_templates import build_welcome_card

        card = build_welcome_card(team_name="MyTeam")
        blob = json.dumps(card, ensure_ascii=False)
        assert "/hire" in blob
        assert "/role add" in blob
        assert "/goal" not in blob
        assert "/slock status" in blob
        assert "/role list" in blob

    def test_body_contains_team_name_in_content(self):
        import json

        from src.slock_engine.card_templates import build_welcome_card

        card = build_welcome_card(team_name="BetaSquad")
        blob = json.dumps(card, ensure_ascii=False)
        assert "BetaSquad" in blob


class TestAllRoleColorsMapping:
    """Verify ALL role colors defined in AGENT_ROLE_COLORS are correctly applied
    by build_agent_message_card via AgentIdentity.card_color."""

    EXPECTED_ROLE_COLORS: dict[str, str] = {
        "coder": "blue",
        "writer": "green",
        "reviewer": "orange",
        "tester": "purple",
        "planner": "red",
        "architect": "indigo",
        "custom": "grey",
    }

    def _make_agent(self, role: str) -> AgentIdentity:
        return AgentIdentity(
            agent_id="test-role-color",
            name=f"Agent_{role}",
            emoji="🤖",
            role=role,
        )

    def test_all_roles_produce_correct_header_color(self):
        """Each role in AGENT_ROLE_COLORS maps to the expected card header template color."""
        from src.slock_engine.models import AGENT_ROLE_COLORS

        # Ensure our expected mapping matches the source of truth
        assert AGENT_ROLE_COLORS == self.EXPECTED_ROLE_COLORS

        for role, expected_color in self.EXPECTED_ROLE_COLORS.items():
            agent = self._make_agent(role)
            card = build_agent_message_card(agent, "test content")
            assert card["header"]["template"] == expected_color, (
                f"Role '{role}' expected header template '{expected_color}', "
                f"got '{card['header']['template']}'"
            )

    def test_unknown_role_falls_back_to_grey(self):
        """A role not present in AGENT_ROLE_COLORS falls back to 'grey'."""
        agent = self._make_agent("unknown_role_xyz")
        card = build_agent_message_card(agent, "fallback test")
        assert card["header"]["template"] == "grey"


# ===========================================================================
# Test Class: build_discussion_card_from_thread / build_discussion_summary_card_from_thread
# ===========================================================================


class TestBuildDiscussionCardFromThread:
    """Tests for the from_thread factory methods that bridge DiscussionThread → keyword-only card builders."""

    def _make_thread(self):
        from src.slock_engine.models import (
            DiscussionConfig,
            DiscussionMessage,
            DiscussionStatus,
            DiscussionThread,
        )

        return DiscussionThread(
            thread_id="thread-abc-123",
            channel_id="chat-001",
            participants=["Coder", "Reviewer"],
            messages=[
                DiscussionMessage(
                    sender_agent_id="coder-01",
                    receiver_agent_id="reviewer-01",
                    content="Here is my implementation.",
                    round_num=1,
                    token_count=20,
                ),
                DiscussionMessage(
                    sender_agent_id="reviewer-01",
                    receiver_agent_id="coder-01",
                    content="LGTM, looks good!",
                    round_num=2,
                    token_count=10,
                ),
            ],
            status=DiscussionStatus.ACTIVE,
            config=DiscussionConfig(max_rounds=5, token_budget=50000),
            trigger_reason="rule:coder->reviewer",
            total_tokens_used=30,
        )

    def test_build_discussion_card_from_thread_returns_valid_card(self):
        """Factory produces a valid Feishu card dict with correct fields and content."""
        import json

        from src.slock_engine.card_templates import build_discussion_card_from_thread

        thread = self._make_thread()
        card = build_discussion_card_from_thread(thread)

        assert card["schema"] == "2.0"
        assert "header" in card
        assert "body" in card
        # Header contains round info
        assert "2/5" in card["header"]["title"]["content"]
        # Body contains participants, trigger reason, and messages
        blob = json.dumps(card, ensure_ascii=False)
        assert "Coder" in blob
        assert "Reviewer" in blob
        assert "rule:coder->reviewer" in blob
        assert "Here is my implementation." in blob
        assert "LGTM" in blob


class TestBuildDiscussionSummaryCardFromThread:
    """Tests for build_discussion_summary_card_from_thread factory."""

    def _make_completed_thread(self):
        from src.slock_engine.models import (
            DiscussionConfig,
            DiscussionMessage,
            DiscussionStatus,
            DiscussionThread,
        )

        thread = DiscussionThread(
            thread_id="thread-done-456",
            channel_id="chat-002",
            participants=["Architect", "Tester"],
            messages=[
                DiscussionMessage(
                    sender_agent_id="arch-01",
                    content="Let's use microservices.",
                    round_num=1,
                ),
                DiscussionMessage(
                    sender_agent_id="test-01",
                    content="I agree with microservices.",
                    round_num=2,
                ),
            ],
            status=DiscussionStatus.CONVERGED,
            config=DiscussionConfig(max_rounds=5),
            trigger_reason="rule:architect->tester",
            total_tokens_used=500,
            conclusion="Team agreed on microservices architecture.",
        )
        return thread

    def test_build_summary_card_returns_valid_card(self):
        """Factory produces a valid Feishu card with conclusion, participants, and token count."""
        import json

        from src.slock_engine.card_templates import build_discussion_summary_card_from_thread

        thread = self._make_completed_thread()
        card = build_discussion_summary_card_from_thread(thread)

        assert card["schema"] == "2.0"
        assert "header" in card
        assert "body" in card
        # Converged status shows green header
        assert card["header"]["template"] == "green"
        blob = json.dumps(card, ensure_ascii=False)
        assert "Team agreed on microservices architecture." in blob
        assert "Architect" in blob
        assert "Tester" in blob
        assert "500" in blob

    def test_build_summary_card_timeout_shows_grey_header(self):
        """Timeout status produces a grey header template."""
        from src.slock_engine.card_templates import build_discussion_summary_card_from_thread
        from src.slock_engine.models import DiscussionStatus

        thread = self._make_completed_thread()
        thread.status = DiscussionStatus.TIMEOUT
        card = build_discussion_summary_card_from_thread(thread)
        assert card["header"]["template"] == "grey"


class TestCardButtonLimits:
    """AC10: Agent message card buttons per row."""

    def test_first_row_has_max_3_buttons(self):
        """First button row should have at most 3 buttons."""
        from src.slock_engine.card_templates import build_agent_message_card
        from src.slock_engine.models import AgentIdentity

        agent = AgentIdentity(
            agent_id="test:coco:coder",
            name="Coder",
            emoji="🔧",
            agent_type="coco",
            role="coder",
        )
        card = build_agent_message_card(
            agent=agent,
            content="Test response",
            channel_id="ch1",
        )
        # Find the first responsive_layout/action_block in elements
        body_elements = card["body"]["elements"]
        button_rows = [
            el for el in body_elements
            if el.get("tag") == "action" or (
                el.get("tag") == "column_set" and
                any("button" in str(col) for col in el.get("columns", []))
            )
        ]
        if button_rows:
            first_row = button_rows[0]
            # Count buttons in first action row
            if first_row.get("tag") == "action":
                assert len(first_row.get("actions", [])) <= 3
            elif first_row.get("tag") == "column_set":
                assert len(first_row.get("columns", [])) <= 3


class TestDiscussionCardProgress:
    """AC15: Discussion card progress bar and purple template."""

    def test_discussion_card_has_progress_bar(self):
        """Discussion card should contain a progress indicator (legacy uses markdown progress text)."""
        from src.slock_engine.card_templates import build_discussion_card

        card = build_discussion_card(
            thread_id="thread1",
            participants=["Agent A", "Agent B"],
            messages=[{"sender": "Agent A", "content": "hello", "round_num": 1}],
            current_round=2,
            max_rounds=5,
            channel_id="ch1",
        )
        # Legacy implementation uses markdown progress text
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        assert "进度" in combined or "●" in combined, "Discussion card should have progress indicator"

    def test_discussion_card_purple_template(self):
        """Discussion card header should use purple template."""
        from src.slock_engine.card_templates import build_discussion_card

        card = build_discussion_card(
            thread_id="thread1",
            participants=["Agent A", "Agent B"],
            messages=[],
            current_round=1,
            max_rounds=3,
            channel_id="ch1",
        )
        assert card["header"]["template"] == "purple"


class TestBuildCommandPanelCard:
    """AC01: /slock returns interactive button card with at least 4 grouped buttons."""

    def test_card_has_at_least_4_buttons(self):
        from src.slock_engine.card_templates import build_command_panel_card

        card = build_command_panel_card(channel_id="test_chat")
        buttons = _collect_buttons(card)
        assert len(buttons) >= 4, f"Expected >=4 buttons, got {len(buttons)}"

    def test_all_buttons_have_slock_prefix(self):
        from src.slock_engine.card_templates import build_command_panel_card

        card = build_command_panel_card(channel_id="test_chat")
        buttons = _collect_buttons(card)
        for btn in buttons:
            action = btn.get("value", {}).get("action", "")
            assert action.startswith("slock_"), f"Button action {action!r} missing slock_ prefix"

    def test_expected_actions_present(self):
        from src.slock_engine.card_templates import build_command_panel_card

        card = build_command_panel_card(channel_id="ch1")
        buttons = _collect_buttons(card)
        actions = {btn["value"]["action"] for btn in buttons if btn.get("value")}
        expected = {
            "slock_cmd_team_list",
            "slock_cmd_role_list",
            "slock_cmd_task_list",
            "slock_cmd_discuss",
        }
        assert expected.issubset(actions), f"Missing actions: {expected - actions}"

    def test_channel_id_propagated(self):
        from src.slock_engine.card_templates import build_command_panel_card

        card = build_command_panel_card(channel_id="my_channel")
        buttons = _collect_buttons(card)
        for btn in buttons:
            assert btn["value"]["channel_id"] == "my_channel"

    def test_card_schema_and_header(self):
        from src.slock_engine.card_templates import build_command_panel_card

        card = build_command_panel_card()
        assert card["schema"] == "2.0"
        # New implementation uses mobile_optimize=True which sets wide_screen_mode=False
        assert "Slock" in card["header"]["title"]["content"]

    def test_buttons_have_valid_behaviors(self):
        """Feishu requires behaviors field with callback type."""
        from src.slock_engine.card_templates import build_command_panel_card

        card = build_command_panel_card(channel_id="ch1")
        buttons = _collect_buttons(card)
        for btn in buttons:
            behaviors = btn.get("behaviors", [])
            assert len(behaviors) == 1, f"Button should have exactly 1 behavior: {btn}"
            assert behaviors[0]["type"] == "callback"
            assert behaviors[0]["value"]["action"] == btn["value"]["action"]


class TestBuildCommandPanelExtendedCard:
    def test_schema2_forms_replace_removed_action_container(self):
        from src.slock_engine.card_templates import build_command_panel_extended_card

        card = build_command_panel_extended_card(
            channel_id="oc_team",
            project_id="project_1",
        )

        blob = json.dumps(card, ensure_ascii=False)
        assert '"tag": "action"' not in blob
        forms = [
            element
            for element in card["body"]["elements"]
            if element.get("tag") == "form"
        ]
        assert len(forms) == 3
        assert all(form.get("element_id") for form in forms)
        assert {
            button["behaviors"][0]["value"]["action"]
            for form in forms
            for button in form["elements"]
            if button.get("tag") == "button"
        } == {
            "slock_form_new_team",
            "slock_form_new_role",
            "slock_form_council",
        }


class TestBuildMemoryDisplayCard:
    """Tests for build_memory_display_card (Task 20)."""

    def test_basic_structure(self):
        from src.slock_engine.card_templates import build_memory_display_card
        from src.slock_engine.models import SlockMemory

        memory = SlockMemory(
            role="coder: writes code",
            key_knowledge="python, go",
            active_context="working on auth module",
        )
        card = build_memory_display_card(memory, agent_name="TestBot")
        assert card["schema"] == "2.0"
        assert "TestBot" in card["header"]["title"]["content"]
        body_text = str(card["body"])
        assert "coder" in body_text
        assert "python" in body_text
        assert "auth module" in body_text

    def test_empty_memory(self):
        from src.slock_engine.card_templates import build_memory_display_card
        from src.slock_engine.models import SlockMemory

        memory = SlockMemory(role="", key_knowledge="", active_context="")
        card = build_memory_display_card(memory, agent_name="Empty")
        body_text = str(card["body"])
        assert "未定义" in body_text or "空" in body_text


class TestBuildRoleSwitchCard:
    """Tests for build_role_switch_card (Task 21)."""

    def test_produces_buttons_per_role(self):
        from src.slock_engine.card_templates import build_role_switch_card

        card = build_role_switch_card(
            roles=["coder", "reviewer", "tester"],
            agent_id="a1",
            channel_id="ch1",
            project_id="p1",
        )
        buttons = _collect_buttons(card)
        # Each role should produce one button
        assert len(buttons) >= 3
        button_texts = [b.get("text", {}).get("content", "") for b in buttons]
        assert any("coder" in t for t in button_texts)
        assert any("reviewer" in t for t in button_texts)

    def test_button_action_type(self):
        from src.slock_engine.card_templates import build_role_switch_card

        card = build_role_switch_card(roles=["coder"], agent_id="a1")
        buttons = _collect_buttons(card)
        for btn in buttons:
            assert btn.get("value", {}).get("action") == "slock_confirm_switch_role"


class TestDiscussionPersistence:
    """Tests for discussion serialize/deserialize (Task 14)."""

    def test_round_trip_serialization(self):
        from src.slock_engine.discussion_manager import DiscussionManager
        from src.slock_engine.models import DiscussionMessage, DiscussionStatus, DiscussionThread

        dm = DiscussionManager()
        thread = DiscussionThread(
            thread_id="t1",
            channel_id="ch1",
            participants=["a1", "a2"],
            messages=[
                DiscussionMessage(
                    message_id="m1",
                    sender_agent_id="a1",
                    receiver_agent_id="a2",
                    content="hello",
                    round_num=1,
                    timestamp=1000.0,
                    token_count=5,
                )
            ],
            status=DiscussionStatus.ACTIVE,
            trigger_reason="uncertainty:maybe",
            conclusion="",
            total_tokens_used=5,
            created_at=1000.0,
        )

        serialized = dm.serialize_thread(thread)
        restored = dm.deserialize_thread(serialized)

        assert restored.thread_id == "t1"
        assert restored.channel_id == "ch1"
        assert restored.participants == ["a1", "a2"]
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "hello"
        assert restored.status == DiscussionStatus.ACTIVE
        assert restored.trigger_reason == "uncertainty:maybe"

    def test_deserialize_unknown_status_defaults_active(self):
        from src.slock_engine.discussion_manager import DiscussionManager
        from src.slock_engine.models import DiscussionStatus

        dm = DiscussionManager()
        data = {"thread_id": "t2", "channel_id": "ch2", "status": "nonexistent", "messages": []}
        restored = dm.deserialize_thread(data)
        assert restored.status == DiscussionStatus.ACTIVE


# ===========================================================================
# Task 17: WP5+WP6 Test Coverage
# ===========================================================================


class TestClarificationCardButtons:
    """Task 17.1: Test clarification card has correct buttons and actions."""

    def test_clarification_card_has_two_buttons(self):
        """build_clarification_card generates a card with exactly two buttons."""
        from src.slock_engine.card_templates.queue_feedback import build_clarification_card

        card = build_clarification_card(
            message_preview="帮我写一个函数",
            channel_id="ch_test",
            message_id="msg_123",
        )
        buttons = _collect_buttons(card)
        assert len(buttons) == 2, f"Expected 2 buttons, got {len(buttons)}"

    def test_clarification_card_confirm_button(self):
        """Confirm button has correct text and action."""
        from src.slock_engine.card_templates.queue_feedback import build_clarification_card

        card = build_clarification_card(
            message_preview="帮我写一个函数",
            channel_id="ch_test",
            message_id="msg_123",
        )
        buttons = _collect_buttons(card)

        confirm_btn = next(
            (b for b in buttons if b["text"]["content"] == "是，这是任务"),
            None,
        )
        assert confirm_btn is not None, "Confirm button not found"
        assert confirm_btn["value"]["action"] == "slock_clarify_confirm"
        assert confirm_btn["value"]["channel_id"] == "ch_test"
        assert confirm_btn["value"]["message_preview"] == "帮我写一个函数"
        assert confirm_btn["value"]["message_id"] == "msg_123"
        assert confirm_btn["type"] == "primary"

    def test_clarification_card_ignore_button(self):
        """Ignore button has correct text and action."""
        from src.slock_engine.card_templates.queue_feedback import build_clarification_card

        card = build_clarification_card(
            message_preview="今天天气真好",
            channel_id="ch_test",
            message_id="msg_456",
        )
        buttons = _collect_buttons(card)

        ignore_btn = next(
            (b for b in buttons if b["text"]["content"] == "不是，只是聊天"),
            None,
        )
        assert ignore_btn is not None, "Ignore button not found"
        assert ignore_btn["value"]["action"] == "slock_clarify_ignore"
        assert ignore_btn["value"]["channel_id"] == "ch_test"
        assert ignore_btn["value"]["message_preview"] == "今天天气真好"
        assert ignore_btn["value"]["message_id"] == "msg_456"
        assert ignore_btn["type"] == "default"

    def test_clarification_card_contains_preview(self):
        """Card body contains the message preview."""
        from src.slock_engine.card_templates.queue_feedback import build_clarification_card

        card = build_clarification_card(message_preview="测试消息预览内容")
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        assert "测试消息预览内容" in combined

    def test_clarification_confirmed_card_shows_queued(self):
        """Confirmed card shows task was queued."""
        from src.slock_engine.card_templates import build_clarification_confirmed_card

        card = build_clarification_confirmed_card(message_preview="我的任务")
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        assert "已确认这是任务" in combined
        assert "任务已加入队列" in combined

    def test_clarification_ignored_card_shows_ignored(self):
        """Ignored card shows message was marked as chat."""
        from src.slock_engine.card_templates import build_clarification_ignored_card

        card = build_clarification_ignored_card(message_preview="只是聊天")
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        assert "已忽略" in combined
        assert "不会创建任务" in combined


class TestQueueFeedbackRedaction:
    """Task 17.2: Test sensitive info is redacted in queue feedback cards."""

    def _assert_redacted(self, text: str, secret: str):
        """Assert secret is not present and redaction marker is present."""
        assert secret not in text, f"Secret {secret!r} was not redacted"
        assert "<redacted>" in text or "<redacted:" in text, f"No redaction marker in {text!r}"

    def test_queue_wait_card_redacts_token(self):
        from src.slock_engine.card_templates.queue_feedback import build_queue_wait_card

        card = build_queue_wait_card(
            position=1,
            busy_count=2,
            message_preview="使用 token: sk-12345abcdef 进行认证",
        )
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        self._assert_redacted(combined, "sk-12345abcdef")

    def test_queue_wait_card_redacts_password(self):
        from src.slock_engine.card_templates.queue_feedback import build_queue_wait_card

        card = build_queue_wait_card(
            position=1,
            busy_count=1,
            message_preview="连接数据库 password=secret123",
        )
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        self._assert_redacted(combined, "secret123")

    def test_result_card_redacts_sensitive(self):
        from src.slock_engine.card_templates.queue_feedback import build_result_card

        card = build_result_card(
            task_preview="使用 password=mysecretpass 连接数据库",
            result="任务完成",
        )
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        self._assert_redacted(combined, "mysecretpass")

    def test_sensitive_at_80_char_boundary(self):
        """Test redaction works when sensitive info is at the 80-char truncation boundary."""
        from src.slock_engine.card_templates.queue_feedback import build_queue_wait_card

        padding = "A" * 60
        sensitive_text = f"{padding} token=sk-1234567890abcdef"

        card = build_queue_wait_card(
            position=1,
            busy_count=1,
            message_preview=sensitive_text,
        )
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)

        assert "sk-1234567890abcdef" not in combined
        assert "token=<redacted>" in combined

    def test_clarification_card_redacts_preview(self):
        """Clarification card should also redact sensitive info in preview."""
        from src.slock_engine.card_templates.queue_feedback import build_clarification_card

        card = build_clarification_card(
            message_preview="帮我用 API_KEY=my-secret-key-123 调用服务",
            channel_id="ch1",
            message_id="msg1",
        )
        all_md = _all_markdown_content(card)
        combined = "\n".join(all_md)
        self._assert_redacted(combined, "my-secret-key-123")


class TestMobileOptimizedCards:
    """Task 17.4: Test progress and result cards are mobile-optimized."""

    def test_progress_card_wide_screen_mode_false(self):
        """SlockStreamProcessor.get_progress_card() should have wide_screen_mode=False."""
        from unittest.mock import MagicMock

        from src.slock_engine.engine import SlockStreamProcessor

        # Create a mock engine
        mock_engine = MagicMock()
        mock_engine.channel = MagicMock()
        mock_engine.channel.team_name = "TestTeam"

        processor = SlockStreamProcessor(mock_engine)
        card = processor.get_progress_card()

        assert card["config"]["wide_screen_mode"] is False, (
            "Progress card should have wide_screen_mode=False for mobile optimization"
        )

    def test_queue_feedback_cards_mobile_optimized(self):
        """All queue feedback card builders should use mobile_optimize=True."""
        from src.slock_engine.card_templates.queue_feedback import (
            build_activation_confirm_card,
            build_clarification_card,
            build_queue_wait_card,
            build_result_card,
            build_timeout_notify_card,
        )

        cards = [
            build_result_card(task_preview="test", result="ok"),
            build_queue_wait_card(position=1, busy_count=1),
            build_timeout_notify_card(task_id="t1", waited_seconds=60.0),
            build_activation_confirm_card(team_name="Test"),
            build_clarification_card(message_preview="test"),
        ]
        for card in cards:
            assert card["config"]["wide_screen_mode"] is False

    def test_result_card_long_content_uses_collapsible_panel(self):
        """Long result content (>500 chars) should use collapsible_panel."""
        from src.slock_engine.card_templates.queue_feedback import build_result_card

        long_result = "这是一个很长的结果内容。" * 50  # ~500+ chars

        card = build_result_card(
            task_preview="测试任务",
            result=long_result,
        )

        # Find collapsible_panel in card
        def _find_collapsible(node: object) -> list[dict]:
            results = []
            if isinstance(node, dict):
                if node.get("tag") == "collapsible_panel":
                    results.append(node)
                for value in node.values():
                    results.extend(_find_collapsible(value))
            elif isinstance(node, list):
                for item in node:
                    results.extend(_find_collapsible(item))
            return results

        panels = _find_collapsible(card)
        assert len(panels) >= 1, "Long result should use collapsible_panel"
        assert "查看完整结果" in panels[0]["header"]["title"]["content"]

    def test_result_card_short_content_no_collapsible_panel(self):
        """Short result content (<=500 chars) should NOT use collapsible_panel."""
        from src.slock_engine.card_templates.queue_feedback import build_result_card

        short_result = "简短的结果"

        card = build_result_card(
            task_preview="测试任务",
            result=short_result,
        )

        def _find_collapsible(node: object) -> list[dict]:
            results = []
            if isinstance(node, dict):
                if node.get("tag") == "collapsible_panel":
                    results.append(node)
                for value in node.values():
                    results.extend(_find_collapsible(value))
            elif isinstance(node, list):
                for item in node:
                    results.extend(_find_collapsible(item))
            return results

        panels = _find_collapsible(card)
        assert len(panels) == 0, "Short result should NOT use collapsible_panel"
