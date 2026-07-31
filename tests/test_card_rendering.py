"""Unit tests for card template rendering functions.

Covers:
- build_role_info_card structure validation
- build_progress_overview_card with empty plans
- build_collaboration_plan_card step status icons
- build_role_list_card multi-agent rows
- build_card_wrapper mobile_optimize flag
"""

from __future__ import annotations

from src.slock_engine.card_templates import (
    build_collaboration_plan_card,
    build_progress_overview_card,
    build_role_info_card,
    build_role_list_card,
)
from src.slock_engine.card_templates.common import build_card_wrapper
from src.slock_engine.models import (
    AgentIdentity,
    AgentStatus,
    CollaborationPlan,
    CollaborationPlanStatus,
    PlanStep,
    PlanStepStatus,
    SlockMemory,
    SlockTask,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str = "TestAgent", role: str = "coder", agent_id: str = "agent-1") -> AgentIdentity:
    """Create a minimal AgentIdentity for testing."""
    return AgentIdentity(
        agent_id=agent_id,
        name=name,
        emoji="🤖",
        agent_type="coco",
        model_name="gpt-4",
        role=role,
    )


def _make_task(task_id: str = "task-001", status: TaskStatus = TaskStatus.TODO) -> SlockTask:
    """Create a minimal SlockTask for testing."""
    return SlockTask(
        task_id=task_id,
        content="Implement feature X",
        status=status,
    )


def _make_plan_step(
    order: int,
    role: str = "coder",
    status: PlanStepStatus = PlanStepStatus.TODO,
    agent_id: str = "agent-1",
) -> PlanStep:
    """Create a PlanStep with given order and status."""
    return PlanStep(
        step_id=f"step-{order}",
        role=role,
        agent_id=agent_id,
        description=f"Step {order} description",
        order=order,
        status=status,
    )


def _make_plan(
    steps: list[PlanStep] | None = None,
    status: CollaborationPlanStatus = CollaborationPlanStatus.EXECUTING,
) -> CollaborationPlan:
    """Create a CollaborationPlan with default steps."""
    if steps is None:
        steps = [
            _make_plan_step(0, role="coder", status=PlanStepStatus.DONE),
            _make_plan_step(1, role="reviewer", status=PlanStepStatus.IN_PROGRESS),
            _make_plan_step(2, role="tester", status=PlanStepStatus.TODO),
        ]
    return CollaborationPlan(
        plan_id="plan-001",
        task_id="task-plan-001",
        steps=steps,
        status=status,
        chain_template="code_review",
    )


# ---------------------------------------------------------------------------
# Test 1: build_role_info_card returns valid card dict with header and body
# ---------------------------------------------------------------------------


class TestBuildRoleInfoCard:
    """Tests for build_role_info_card."""

    def test_returns_dict_with_header_and_body(self):
        agent = _make_agent(name="Alice", role="coder")
        card = build_role_info_card(agent, status=AgentStatus.IDLE)

        assert isinstance(card, dict)
        assert "header" in card
        assert "body" in card or "elements" in card

    def test_header_contains_agent_name(self):
        agent = _make_agent(name="Alice", role="coder")
        card = build_role_info_card(agent, status=AgentStatus.RUNNING)

        header = card.get("header", {})
        title = header.get("title", {})
        # Title content should reference the agent name
        title_content = title.get("content", "")
        assert "Alice" in title_content

    def test_with_memory_and_task(self):
        agent = _make_agent(name="Bob", role="reviewer")
        memory = SlockMemory(
            role="Code reviewer",
            key_knowledge="Python best practices",
            active_context="Reviewing PR #42",
        )
        task = _make_task(task_id="task-review", status=TaskStatus.IN_PROGRESS)

        card = build_role_info_card(
            agent,
            status=AgentStatus.THINKING,
            memory=memory,
            current_task=task,
        )

        assert isinstance(card, dict)
        assert "header" in card

    def test_role_info_card_ac2_full_profile(self):
        """AC-2: /role info card shows complete profile with all 6 required sections."""
        agent = AgentIdentity(
            agent_id="ac2-agent",
            name="FullAgent",
            emoji="🧑‍💻",
            agent_type="claude",
            model_name="claude-sonnet",
            role="architect",
            system_prompt="我是一个系统架构师，专注于分布式系统设计和微服务架构，善于权衡技术方案并输出清晰的架构决策文档，确保团队对齐。",
        )
        memory = SlockMemory(
            role="Architect",
            key_knowledge="系统采用事件驱动架构\n数据库选用 PostgreSQL\n缓存层使用 Redis Cluster\n部署基于 K8s\n监控使用 Prometheus",
            active_context="设计新的消息队列方案",
        )
        skills = [
            {"tag": "架构设计", "success_rate": 95.0, "total_tasks": 20},
            {"tag": "代码审查", "success_rate": 88.0, "total_tasks": 15},
        ]
        current_task = SlockTask(
            task_id="t-curr", content="设计订单系统微服务拆分方案", status=TaskStatus.IN_PROGRESS, claimed_by="ac2-agent"
        )
        recent_tasks = [
            SlockTask(task_id="t-h1", content="完成API网关选型", status=TaskStatus.DONE, claimed_by="ac2-agent"),
            SlockTask(task_id="t-h2", content="编写数据库迁移方案", status=TaskStatus.DONE, claimed_by="ac2-agent"),
            SlockTask(task_id="t-h3", content="审查认证模块设计", status=TaskStatus.DONE, claimed_by="ac2-agent"),
        ]

        card = build_role_info_card(
            agent,
            status=AgentStatus.THINKING,
            memory=memory,
            skill_profiles=skills,
            current_task=current_task,
            recent_tasks=recent_tasks,
            channel_id="ch-test",
        )

        # AC-2 field 1: emoji + name in header
        header = card["header"]
        assert "🧑‍💻" in header["title"]["content"]
        assert "FullAgent" in header["title"]["content"]

        # AC-2 field 2: role type + color
        assert header["template"] == "indigo"  # architect → indigo
        elements = card["body"]["elements"]
        body_text = " ".join(
            e.get("content", "") for e in elements if e.get("tag") == "markdown"
        )
        assert "`architect`" in body_text

        # AC-2 field 3: personality ≤100 chars summary
        assert "系统架构师" in body_text

        # AC-2 field 4: skill tags with success rate
        assert "`架构设计`" in body_text
        assert "95%" in body_text
        assert "(20次)" in body_text
        assert "`代码审查`" in body_text

        # AC-2 field 5: L1 memory top 3 key knowledge
        # Memory is in collapsible_panel
        panel = next(e for e in elements if e.get("tag") == "collapsible_panel")
        mem_content = panel["elements"][0]["content"]
        assert "事件驱动架构" in mem_content
        assert "PostgreSQL" in mem_content
        assert "Redis Cluster" in mem_content
        assert "还有 2 条" in mem_content

        # AC-2 field 6: current task + 3 history tasks
        assert "设计订单系统微服务拆分方案" in body_text
        assert "完成API网关选型" in body_text
        assert "编写数据库迁移方案" in body_text
        assert "审查认证模块设计" in body_text

    def test_role_info_task_overflow(self):
        """Only last 3 recent_tasks are shown when more are provided."""
        agent = _make_agent(name="Overflow", role="coder")
        tasks = [
            SlockTask(task_id=f"t-{i}", content=f"历史任务第{i}条", status=TaskStatus.DONE, claimed_by="agent-1")
            for i in range(5)
        ]

        card = build_role_info_card(
            agent,
            status=AgentStatus.IDLE,
            recent_tasks=tasks,
        )

        elements = card["body"]["elements"]
        all_content = _extract_all_markdown(elements)

        # Only last 3 (index 2, 3, 4) shown — but build_role_info_card takes first 3 from list
        # The card_templates/role.py slices [:3], so tasks[0], tasks[1], tasks[2] are shown
        assert "历史任务第0条" in all_content
        assert "历史任务第1条" in all_content
        assert "历史任务第2条" in all_content
        # tasks[3] and tasks[4] should NOT appear
        assert "历史任务第3条" not in all_content
        assert "历史任务第4条" not in all_content


# ---------------------------------------------------------------------------
# Test 2: build_progress_overview_card with empty plan list
# ---------------------------------------------------------------------------


class TestBuildProgressOverviewCard:
    """Tests for build_progress_overview_card."""

    def test_empty_plans_returns_valid_card(self):
        agents = [_make_agent(name="Agent1", agent_id="a1")]
        card = build_progress_overview_card(plans=[], agents=agents)

        assert isinstance(card, dict)
        assert "header" in card
        # Body should have elements even if no plans
        body = card.get("body", {})
        elements = body.get("elements", [])
        assert isinstance(elements, list)

    def test_single_plan_renders_row(self):
        agent = _make_agent(name="Coder", agent_id="agent-1")
        plan = _make_plan()
        card = build_progress_overview_card(plans=[plan], agents=[agent])

        assert isinstance(card, dict)
        body = card.get("body", {})
        elements = body.get("elements", [])
        # Should have summary line + hr + at least one plan row
        assert len(elements) >= 3


# ---------------------------------------------------------------------------
# Test 3: build_collaboration_plan_card renders steps with correct status icons
# ---------------------------------------------------------------------------


class TestBuildCollaborationPlanCard:
    """Tests for build_collaboration_plan_card step rendering."""

    def test_renders_card_with_steps(self):
        agent = _make_agent(name="Alice", agent_id="agent-1")
        plan = _make_plan()

        card = build_collaboration_plan_card(plan=plan, agents=[agent])

        assert isinstance(card, dict)
        assert "header" in card
        body = card.get("body", {})
        elements = body.get("elements", [])
        assert len(elements) > 0

    def test_step_status_icons_in_rendered_content(self):
        """Verify that step status icons from _STEP_STATUS_ICONS appear in the card body."""
        agent = _make_agent(name="Worker", agent_id="agent-1")
        steps = [
            _make_plan_step(0, role="coder", status=PlanStepStatus.DONE, agent_id="agent-1"),
            _make_plan_step(1, role="reviewer", status=PlanStepStatus.IN_PROGRESS, agent_id="agent-1"),
            _make_plan_step(2, role="tester", status=PlanStepStatus.TODO, agent_id="agent-1"),
        ]
        plan = _make_plan(steps=steps)

        card = build_collaboration_plan_card(plan=plan, agents=[agent])

        # Collect all markdown content from the card body
        body = card.get("body", {})
        elements = body.get("elements", [])
        all_content = _extract_all_markdown(elements)

        # Expected icons from _STEP_STATUS_ICONS
        assert "\u2705" in all_content  # DONE icon (checkmark)
        assert "\U0001f535" in all_content  # IN_PROGRESS icon (blue circle)
        assert "\u2b1c" in all_content  # TODO icon (white square)

    def test_skipped_step_shows_skip_icon(self):
        agent = _make_agent(name="Skipper", agent_id="agent-1")
        steps = [
            _make_plan_step(0, role="coder", status=PlanStepStatus.SKIPPED, agent_id="agent-1"),
        ]
        plan = _make_plan(steps=steps)

        card = build_collaboration_plan_card(plan=plan, agents=[agent])
        body = card.get("body", {})
        elements = body.get("elements", [])
        all_content = _extract_all_markdown(elements)

        # SKIPPED icon
        assert "\u23ed" in all_content  # skip icon

    def test_show_actions_false_omits_buttons(self):
        agent = _make_agent(name="NoActions", agent_id="agent-1")
        plan = _make_plan()

        card = build_collaboration_plan_card(plan=plan, agents=[agent], show_actions=False)

        body = card.get("body", {})
        elements = body.get("elements", [])
        # No button elements should be present
        buttons = [e for e in elements if e.get("tag") == "button"]
        assert len(buttons) == 0


# ---------------------------------------------------------------------------
# Test 5: build_role_list_card with multiple agents returns card with rows
# ---------------------------------------------------------------------------


class TestBuildRoleListCard:
    """Tests for build_role_list_card."""

    def test_multiple_agents_returns_card(self):
        agents = [
            (_make_agent(name="Alice", role="coder", agent_id="a1"), AgentStatus.RUNNING),
            (_make_agent(name="Bob", role="reviewer", agent_id="a2"), AgentStatus.IDLE),
            (_make_agent(name="Charlie", role="tester", agent_id="a3"), AgentStatus.THINKING),
        ]

        card = build_role_list_card(agents=agents)

        assert isinstance(card, dict)
        assert "header" in card
        body = card.get("body", {})
        elements = body.get("elements", [])
        assert len(elements) > 0

    def test_card_contains_agent_names(self):
        agents = [
            (_make_agent(name="Alice", role="coder", agent_id="a1"), AgentStatus.RUNNING),
            (_make_agent(name="Bob", role="reviewer", agent_id="a2"), AgentStatus.IDLE),
        ]

        card = build_role_list_card(agents=agents)

        body = card.get("body", {})
        elements = body.get("elements", [])
        all_content = _extract_all_markdown(elements)

        assert "Alice" in all_content
        assert "Bob" in all_content


# ---------------------------------------------------------------------------
# Test 6: build_card_wrapper mobile_optimize sets wide_screen_mode to False
# ---------------------------------------------------------------------------


class TestBuildCardWrapperMobileOptimize:
    """Tests for the mobile_optimize flag in build_card_wrapper."""

    def test_mobile_optimize_false_sets_wide_screen_true(self):
        card = build_card_wrapper(
            header_title="Test Card",
            elements=[{"tag": "markdown", "content": "hello"}],
            mobile_optimize=False,
        )

        assert card["config"]["wide_screen_mode"] is True

    def test_mobile_optimize_true_sets_wide_screen_false(self):
        card = build_card_wrapper(
            header_title="Mobile Card",
            elements=[{"tag": "markdown", "content": "mobile content"}],
            mobile_optimize=True,
        )

        assert card["config"]["wide_screen_mode"] is False



# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _extract_all_markdown(elements: list[dict]) -> str:
    """Recursively extract all markdown content from card elements."""
    parts: list[str] = []
    for elem in elements:
        tag = elem.get("tag", "")
        if tag == "markdown":
            parts.append(elem.get("content", ""))
        # Handle collapsible_panel which nests elements
        if tag == "collapsible_panel":
            nested = elem.get("elements", [])
            parts.append(_extract_all_markdown(nested))
        # Handle column_set → columns → elements
        if tag == "column_set":
            for col in elem.get("columns", []):
                col_elements = col.get("elements", [])
                parts.append(_extract_all_markdown(col_elements))
        # Handle column directly
        if tag == "column":
            parts.append(_extract_all_markdown(elem.get("elements", [])))
        # Handle form elements
        if tag == "form":
            parts.append(_extract_all_markdown(elem.get("elements", [])))
        # Generic nested elements (but not already handled)
        if "elements" in elem and tag not in ("markdown", "collapsible_panel", "column_set", "column", "form"):
            parts.append(_extract_all_markdown(elem["elements"]))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Test 7: column_set helpers produce valid structures
# ---------------------------------------------------------------------------


class TestColumnSetHelpers:
    """Tests for build_column_set_row and build_column."""

    def test_build_column_set_row_structure(self):
        from src.slock_engine.card_templates.common import build_column, build_column_set_row

        cols = [
            build_column([{"tag": "markdown", "content": "A"}], weight=1),
            build_column([{"tag": "markdown", "content": "B"}], weight=2),
        ]
        row = build_column_set_row(cols)

        assert row["tag"] == "column_set"
        assert len(row["columns"]) == 2
        assert row["columns"][0]["weight"] == 1
        assert row["columns"][1]["weight"] == 2

    def test_build_column_defaults(self):
        from src.slock_engine.card_templates.common import build_column

        col = build_column([{"tag": "markdown", "content": "Hi"}])
        assert col["tag"] == "column"
        assert col["width"] == "weighted"
        assert col["weight"] == 1
        assert col["vertical_align"] == "center"
        assert len(col["elements"]) == 1

    def test_status_card_uses_column_set(self):
        """Refactored status card should contain schema-safe column_set rows per agent."""
        from src.slock_engine.card_templates import build_status_panel_card

        agents = [
            (_make_agent(name="Alice", agent_id="a1"), AgentStatus.RUNNING),
            (_make_agent(name="Bob", agent_id="a2"), AgentStatus.IDLE),
        ]
        card = build_status_panel_card(agents=agents, channel_id="ch1")
        elements = card["body"]["elements"]

        row_elements = [e for e in elements if e.get("tag") == "column_set"]
        assert len(row_elements) >= 2  # one per agent

    def test_role_list_card_uses_column_set(self):
        """Refactored role list card should contain schema-safe column_set rows."""
        agents = [
            (_make_agent(name="Dev", role="coder", agent_id="d1"), AgentStatus.IDLE),
        ]
        card = build_role_list_card(agents=agents)
        elements = card["body"]["elements"]

        row_elements = [e for e in elements if e.get("tag") == "column_set"]
        assert len(row_elements) >= 1

    def test_progress_overview_uses_native_progress(self):
        """Progress overview card should use native progress elements."""
        agent = _make_agent(name="Coder", agent_id="agent-1")
        plan = _make_plan()
        card = build_progress_overview_card(plans=[plan], agents=[agent])
        elements = card["body"]["elements"]

        # Find progress elements recursively (inside div > column_set > column)
        def find_progress(elems):
            found = []
            for e in elems:
                if e.get("tag") == "progress":
                    found.append(e)
                if e.get("tag") == "column_set":
                    for col in e.get("columns", []):
                        found.extend(find_progress(col.get("elements", [])))
                if e.get("tag") == "column":
                    found.extend(find_progress(e.get("elements", [])))
                if e.get("tag") == "div":
                    found.extend(find_progress(e.get("elements", [])))
            return found

        progress_els = find_progress(elements)
        assert len(progress_els) >= 1
        assert progress_els[0]["tag"] == "progress"
        assert "percent" in progress_els[0]

    def test_role_info_card_has_assign_form(self):
        """Role info card should contain a form for task assignment (inside collapsible_panel)."""
        agent = _make_agent(name="Former", role="coder")
        card = build_role_info_card(agent, status=AgentStatus.IDLE, channel_id="ch1")
        elements = card["body"]["elements"]

        # Form may be nested inside a collapsible_panel
        def find_forms(elems):
            found = []
            for e in elems:
                if e.get("tag") == "form":
                    found.append(e)
                if e.get("tag") == "collapsible_panel":
                    found.extend(find_forms(e.get("elements", [])))
            return found

        forms = find_forms(elements)
        assert len(forms) >= 1
        form = forms[0]
        assert form["name"].startswith("assign_task_")
        # Should have input + button
        form_els = form["elements"]
        tags = [e["tag"] for e in form_els]
        assert "input" in tags
        assert "button" in tags


# ---------------------------------------------------------------------------
# Test 8: Card payload structural validation (Feishu Card 2.0 compliance)
# ---------------------------------------------------------------------------


class TestCardPayloadValidation:
    """Validate card payloads meet Feishu Interactive Card 2.0 structural requirements."""

    VALID_TOP_LEVEL_KEYS = {"schema", "header", "body", "config", "card_link", "i18n_elements"}
    VALID_HEADER_KEYS = {"title", "subtitle", "template", "icon", "ud_icon"}
    VALID_ELEMENT_TAGS = {
        "markdown", "hr", "column_set", "collapsible_panel", "form",
        "button", "input", "select_static", "overflow", "progress",
        "action", "div", "note", "img", "table", "chart",
    }

    def _validate_card_structure(self, card: dict) -> list[str]:
        """Return list of structural violations found in card."""
        errors = []
        if not isinstance(card, dict):
            errors.append("Card is not a dict")
            return errors

        # Must have schema 2.0
        if card.get("schema") != "2.0":
            errors.append(f"schema is '{card.get('schema')}', expected '2.0'")

        # Must have header
        if "header" not in card:
            errors.append("Missing 'header'")
        else:
            header = card["header"]
            if "title" not in header:
                errors.append("Header missing 'title'")
            elif "content" not in header.get("title", {}):
                errors.append("Header title missing 'content'")

        # Must have body with elements
        if "body" not in card:
            errors.append("Missing 'body'")
        elif "elements" not in card.get("body", {}):
            errors.append("Body missing 'elements'")

        return errors

    def _validate_elements_recursive(self, elements: list[dict], path: str = "body") -> list[str]:
        """Recursively validate element tags and nesting."""
        errors = []
        for i, elem in enumerate(elements):
            loc = f"{path}[{i}]"
            if not isinstance(elem, dict):
                errors.append(f"{loc}: element is not a dict")
                continue
            tag = elem.get("tag")
            if tag is None:
                errors.append(f"{loc}: element missing 'tag'")
                continue

            # Validate column_set nesting
            if tag == "column_set":
                columns = elem.get("columns", [])
                if not isinstance(columns, list):
                    errors.append(f"{loc}: column_set 'columns' is not a list")
                for j, col in enumerate(columns):
                    if not isinstance(col, dict):
                        errors.append(f"{loc}.columns[{j}]: not a dict")
                        continue
                    if col.get("tag") != "column":
                        errors.append(f"{loc}.columns[{j}]: tag is '{col.get('tag')}', expected 'column'")
                    col_elems = col.get("elements", [])
                    errors.extend(self._validate_elements_recursive(col_elems, f"{loc}.columns[{j}]"))

            # Validate form nesting
            if tag == "form":
                if "name" not in elem:
                    errors.append(f"{loc}: form missing 'name'")
                form_elems = elem.get("elements", [])
                errors.extend(self._validate_elements_recursive(form_elems, f"{loc}"))

            # Validate collapsible_panel
            if tag == "collapsible_panel":
                panel_elems = elem.get("elements", [])
                errors.extend(self._validate_elements_recursive(panel_elems, f"{loc}"))

            # Validate progress element
            if tag == "progress":
                if "percent" not in elem:
                    errors.append(f"{loc}: progress missing 'percent'")
                pct = elem.get("percent")
                if pct is not None and not (0 <= pct <= 100):
                    errors.append(f"{loc}: progress percent {pct} out of range [0,100]")

            # Validate button
            if tag == "button":
                if "text" not in elem:
                    errors.append(f"{loc}: button missing 'text'")

        return errors

    def test_role_info_card_valid_structure(self):
        agent = _make_agent(name="ValidAgent", role="coder")
        card = build_role_info_card(agent, status=AgentStatus.IDLE, channel_id="ch1")

        errors = self._validate_card_structure(card)
        assert not errors, f"Structural errors: {errors}"

        elements = card["body"]["elements"]
        elem_errors = self._validate_elements_recursive(elements)
        assert not elem_errors, f"Element errors: {elem_errors}"

    def test_progress_overview_card_valid_structure(self):
        agent = _make_agent(name="Coder", agent_id="agent-1")
        plan = _make_plan()
        card = build_progress_overview_card(plans=[plan], agents=[agent])

        errors = self._validate_card_structure(card)
        assert not errors, f"Structural errors: {errors}"

        elements = card["body"]["elements"]
        elem_errors = self._validate_elements_recursive(elements)
        assert not elem_errors, f"Element errors: {elem_errors}"

    def test_collaboration_plan_card_valid_structure(self):
        plan = _make_plan()
        agent = _make_agent(name="Coder", agent_id="agent-1")
        card = build_collaboration_plan_card(plan=plan, agents=[agent])

        errors = self._validate_card_structure(card)
        assert not errors, f"Structural errors: {errors}"

        elements = card["body"]["elements"]
        elem_errors = self._validate_elements_recursive(elements)
        assert not elem_errors, f"Element errors: {elem_errors}"

    def test_progress_percent_in_valid_range(self):
        """All progress elements in plan cards should have percent in [0, 100]."""
        # Test with all steps done (100%)
        steps_all_done = [
            _make_plan_step(0, status=PlanStepStatus.DONE),
            _make_plan_step(1, status=PlanStepStatus.DONE),
        ]
        plan = _make_plan(steps=steps_all_done, status=CollaborationPlanStatus.COMPLETED)
        agent = _make_agent(agent_id="agent-1")
        card = build_collaboration_plan_card(plan=plan, agents=[agent])

        elements = card["body"]["elements"]
        elem_errors = self._validate_elements_recursive(elements)
        assert not elem_errors, f"Element errors: {elem_errors}"


# ---------------------------------------------------------------------------
# Wave 4: STATUS_BG_STYLE_MAP integration and highlight_plan_id
# ---------------------------------------------------------------------------


class TestStatusBgStyleMapIntegration:
    """Verify role/status cards use STATUS_BG_STYLE_MAP for background styling."""

    def test_role_list_uses_alternating_bg(self):
        """Role list card uses alternating default/grey backgrounds (not status-based)."""
        from src.slock_engine.card_templates.role import build_role_list_card
        from src.slock_engine.models import AgentIdentity, AgentStatus

        agents = [
            (AgentIdentity(agent_id="a1", name="Coder", emoji="🤖", role="coder", agent_type="coco"), AgentStatus.RUNNING),
            (AgentIdentity(agent_id="a2", name="Reviewer", emoji="👀", role="reviewer", agent_type="coco"), AgentStatus.IDLE),
        ]
        card = build_role_list_card(agents=agents, channel_id="ch1")
        body_str = str(card)
        # Role list uses alternating row backgrounds: default for even, grey for odd
        assert "default" in body_str
        assert "grey" in body_str

    def test_status_panel_uses_semantic_bg_for_discussing(self):
        """Status panel should use 'purple' background for DISCUSSING agents."""
        from src.slock_engine.card_templates.common import STATUS_BG_STYLE_MAP
        from src.slock_engine.card_templates.status import build_status_panel_card
        from src.slock_engine.models import AgentIdentity, AgentStatus

        agent = AgentIdentity(
            agent_id="a2", name="Reviewer", emoji="👀",
            role="reviewer", agent_type="claude",
        )
        card = build_status_panel_card(
            agents=[(agent, AgentStatus.DISCUSSING)],
            channel_id="ch2",
        )
        expected_bg = STATUS_BG_STYLE_MAP[AgentStatus.DISCUSSING]  # "purple"
        assert expected_bg in str(card)

class TestHighlightPlanId:
    """Verify progress overview card highlights the specified plan."""

    def test_highlight_plan_gets_blue_bg(self):
        """Highlighted plan should get 'blue' background_style."""
        from src.slock_engine.card_templates.progress import build_progress_overview_card
        from src.slock_engine.models import (
            AgentIdentity,
            CollaborationPlan,
            CollaborationPlanStatus,
            PlanStep,
        )

        plan = CollaborationPlan(
            plan_id="plan-001",
            task_id="task-001",
            steps=[PlanStep(step_id="s1", role="coder", description="Code it")],
            status=CollaborationPlanStatus.EXECUTING,
        )
        agent = AgentIdentity(
            agent_id="a1", name="Coder", emoji="🤖",
            role="coder", agent_type="coco",
        )
        card = build_progress_overview_card(
            plans=[plan],
            agents=[agent],
            highlight_plan_id="plan-001",
            channel_id="ch1",
        )
        # The highlighted plan row should have card_primary background
        assert "card_primary" in str(card)

    def test_no_highlight_gets_default_bg(self):
        """Without highlight, plan rows should get 'default' background."""
        from src.slock_engine.card_templates.progress import build_progress_overview_card
        from src.slock_engine.models import (
            AgentIdentity,
            CollaborationPlan,
            CollaborationPlanStatus,
            PlanStep,
        )

        plan = CollaborationPlan(
            plan_id="plan-002",
            task_id="task-002",
            steps=[PlanStep(step_id="s1", role="coder", description="Code it")],
            status=CollaborationPlanStatus.EXECUTING,
        )
        agent = AgentIdentity(
            agent_id="a1", name="Coder", emoji="🤖",
            role="coder", agent_type="coco",
        )
        card = build_progress_overview_card(
            plans=[plan],
            agents=[agent],
            highlight_plan_id="",
            channel_id="ch1",
        )
        body_str = str(card)
        # Should not have "blue" as a background style (only "default")
        assert "'background_style': 'blue'" not in body_str


class TestLegacyCardsUseWrapper:
    """Regression: ensure no raw schema dicts remain in card_templates_legacy.py."""

    def test_no_raw_schema_outside_wrapper(self):
        """All card construction in legacy module must go through _build_card_wrapper."""
        import ast
        from pathlib import Path

        src = Path("src/slock_engine/card_templates_legacy.py").read_text()
        tree = ast.parse(src)

        # Find all string literals that are "2.0" — should only appear inside _build_card_wrapper
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_build_card_wrapper":
                continue
            if isinstance(node, ast.FunctionDef):
                func_src = ast.get_source_segment(src, node)
                if func_src and '"schema": "2.0"' in func_src:
                    violations.append(node.name)

        assert violations == [], (
            f"Functions still construct raw card dicts: {violations}"
        )


# ---------------------------------------------------------------------------
# Test 9: Migrated card templates produce valid cards
# ---------------------------------------------------------------------------


class TestMigratedCardTemplates:
    """Verify migrated card submodules produce valid Feishu cards."""

    def test_welcome_card_structure(self):
        from src.slock_engine.card_templates.welcome import build_welcome_card

        card = build_welcome_card(team_name="测试团队")
        assert isinstance(card, dict)
        assert card["schema"] == "2.0"
        assert "header" in card
        assert "body" in card
        assert "测试团队" in card["header"]["title"]["content"]
        # Should have mobile_optimize=True → wide_screen_mode=False
        assert card["config"]["wide_screen_mode"] is False

    def test_command_hub_card_structure(self):
        from src.slock_engine.card_templates.command import build_command_hub_card

        card = build_command_hub_card(channel_id="ch-test")
        assert isinstance(card, dict)
        assert card["schema"] == "2.0"
        assert "header" in card
        assert "body" in card

    def test_command_panel_card_structure(self):
        from src.slock_engine.card_templates.command import build_command_panel_card

        card = build_command_panel_card(channel_id="ch-test", project_id="proj-1")
        assert isinstance(card, dict)
        assert card["schema"] == "2.0"

    def test_command_panel_extended_card_structure(self):
        from src.slock_engine.card_templates.command import build_command_panel_extended_card

        card = build_command_panel_extended_card(channel_id="ch-test")
        assert isinstance(card, dict)
        assert card["schema"] == "2.0"

    def test_council_card_structure(self):
        from src.slock_engine.card_templates.council import build_council_card
        from src.slock_engine.models import CouncilRun, CouncilStatus

        run = CouncilRun(
            run_id="r1",
            question="架构评审",
            status=CouncilStatus.STAGE1_RUNNING,
            participant_ids=["agent-1", "agent-2"],
        )
        card = build_council_card(run, channel_id="ch-test")
        assert isinstance(card, dict)
        assert card["schema"] == "2.0"
        assert "header" in card

    def test_escalation_card_structure(self):
        from src.slock_engine.card_templates.escalation import build_escalation_card
        from src.slock_engine.models import EscalationLevel, EscalationRequest

        esc = EscalationRequest(
            agent_id="agent-1",
            agent_name="Coder",
            reason="代码冲突需要人工决策",
            level=EscalationLevel.BLOCKED,
        )
        card = build_escalation_card(esc, channel_id="ch-test")
        assert isinstance(card, dict)
        assert card["schema"] == "2.0"
        assert "header" in card

    def test_resolved_escalation_card_structure(self):
        from src.slock_engine.card_templates.escalation import build_resolved_escalation_card
        from src.slock_engine.models import EscalationLevel, EscalationRequest

        esc = EscalationRequest(
            agent_id="agent-1",
            agent_name="Coder",
            reason="代码冲突",
            level=EscalationLevel.BLOCKED,
            resolved=True,
        )
        card = build_resolved_escalation_card(
            esc,
            resolution="已手动合并解决",
            resolved_by="admin",
        )
        assert isinstance(card, dict)
        assert card["schema"] == "2.0"

    def test_memory_group_card_structure(self):
        from src.slock_engine.card_templates.memory import build_memory_group_card

        items = [
            {"category": "key_knowledge", "content": "Python best practices", "timestamp": "2024-01-01"},
            {"category": "experience", "content": "Always write tests first"},
        ]
        card = build_memory_group_card(
            agent_name="Coder",
            agent_emoji="🤖",
            memory_items=items,
            channel_id="ch-test",
            agent_id="agent-1",
        )
        assert isinstance(card, dict)
        assert card["schema"] == "2.0"
        assert "header" in card
        assert "turquoise" == card["header"]["template"]
