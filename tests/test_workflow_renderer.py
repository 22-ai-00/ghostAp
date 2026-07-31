"""Tests for WorkflowProgressRenderer column_set layout (Task 10/16).

Validates:
- render_progress_card produces valid card structure
- Phase sections use column_set elements
- Budget section uses column_set
- render_compact_status produces expected format
- Pagination works for large agent lists
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

from src.workflow_engine.models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.renderer import (
    _AGENT_OUTPUT_FORBIDDEN_MARKERS,
    WorkflowProgressRenderer,
    _card_text_for_agent_output,
    _column_set,
    _format_duration,
    _format_tokens,
    _md_element,
    _pct,
    render_completion_card,
    render_script_preview,
)


class TestHelperFunctions(unittest.TestCase):
    """Test renderer helper functions."""

    def test_format_duration_seconds(self):
        self.assertEqual(_format_duration(30), "30s")

    def test_format_duration_minutes(self):
        self.assertEqual(_format_duration(90), "1m30s")

    def test_format_duration_hours(self):
        self.assertEqual(_format_duration(3700), "1h1m")

    def test_format_duration_sub_second(self):
        self.assertEqual(_format_duration(0.5), "<1s")

    def test_format_tokens_small(self):
        self.assertEqual(_format_tokens(500), "500")

    def test_format_tokens_k(self):
        self.assertEqual(_format_tokens(5000), "5K")

    def test_format_tokens_m(self):
        self.assertEqual(_format_tokens(2_500_000), "2.5M")

    def test_pct_empty(self):
        pct = _pct(0, 100)
        self.assertIn("0%", pct)

    def test_pct_full(self):
        pct = _pct(100, 100)
        self.assertIn("100%", pct)

    def test_pct_zero_total(self):
        pct = _pct(0, 0)
        self.assertIn("0%", pct)

    def test_column_set_structure(self):
        cs = _column_set([{"tag": "column"}], flex_mode="bisect")
        self.assertEqual(cs["tag"], "column_set")
        self.assertEqual(cs["flex_mode"], "bisect")
        self.assertEqual(len(cs["columns"]), 1)


class TestWorkflowProgressRenderer(unittest.TestCase):
    """Test the full renderer output structure."""

    def _make_project(self, n_agents=3):
        """Create a WorkflowProject with some progress."""
        project = WorkflowProject(
            name="test-workflow",
            status=WorkflowStatus.RUNNING,
            started_at=time.time() - 60,
        )
        phase = PhaseProgress(
            title="Code Analysis",
            started_at=time.time() - 60,
        )
        for i in range(n_agents):
            agent = AgentProgress(
                label=f"agent_{i}",
                tool="coco",
                status=AgentStatus.DONE if i < n_agents - 1 else AgentStatus.RUNNING,
                duration_s=5.0 if i < n_agents - 1 else 0.0,
                token_usage=10000,
            )
            phase.agents.append(agent)
        project.phases.append(phase)
        project.metrics.total_agents = n_agents
        project.metrics.completed_agents = n_agents - 1
        return project

    def test_render_progress_card_has_header_and_elements(self):
        project = self._make_project()
        renderer = WorkflowProgressRenderer(project)
        card = renderer.render_progress_card()

        self.assertIn("header", card)
        self.assertIn("elements", card)
        self.assertIsInstance(card["elements"], list)
        self.assertGreater(len(card["elements"]), 0)

    def test_render_progress_card_uses_markdown(self):
        """Phase rendering should produce markdown elements (mobile-friendly)."""
        project = self._make_project()
        renderer = WorkflowProgressRenderer(project)
        card = renderer.render_progress_card()

        # Find markdown elements
        md_elements = [e for e in card["elements"] if e.get("tag") == "markdown"]
        self.assertGreater(len(md_elements), 0, "Expected at least one markdown in card")

    def test_metrics_footer_uses_clock_format_with_days_as_largest_unit(self):
        project = WorkflowProject(
            name="long-workflow",
            status=WorkflowStatus.COMPLETED,
            started_at=100.0,
            finished_at=90_161.0,
        )

        footer = WorkflowProgressRenderer(project)._render_metrics_footer()
        rendered = json.dumps(footer, ensure_ascii=False)

        self.assertIn("**耗时:** 1天 01:01:01", rendered)

    def test_render_compact_status_format(self):
        project = self._make_project()
        renderer = WorkflowProgressRenderer(project)
        status = renderer.render_compact_status()

        self.assertIn("任务:", status)
        self.assertIn("test-workflow", status)
        self.assertIn("阶段", status)
        self.assertIn("代理", status)

    def test_pagination_large_agent_list(self):
        """With >20 agents, pagination notice should appear."""
        project = self._make_project(n_agents=25)
        renderer = WorkflowProgressRenderer(project)
        card = renderer.render_progress_card()

        # Look for pagination/overflow notice in any top-level markdown element.
        # The renderer now emits "共 N 条已完成/缓存（已折叠）" for hidden done/cached agents.
        md_elements = [e for e in card["elements"] if e.get("tag") == "markdown"]
        md_texts = [e["content"] for e in md_elements]
        has_pagination = any(("代理" in t and "..." in t) or ("条已完成/缓存" in t) for t in md_texts)
        self.assertTrue(has_pagination, "Expected pagination notice for 25 agents")

    def test_phase_section_uses_collapsible_panels(self):
        """Phase rendering should use collapsible_panel grouped by agent status."""
        project = self._make_project()
        # Ensure at least one agent of each major status
        project.phases[0] if project.phases else None
        renderer = WorkflowProgressRenderer(project)
        card = renderer.render_progress_card()

        # Collect all elements (including nested inside collapsible_panel)
        all_tags = [e.get("tag") for e in card["elements"]]
        # Phase section now wraps agents in collapsible_panel, so it shouldn't have any top-level markdown for those
        self.assertIn("collapsible_panel", all_tags)

    def test_header_template_reflects_status(self):
        project = self._make_project()
        renderer = WorkflowProgressRenderer(project)

        # Running → blue
        header = renderer._render_header()
        self.assertEqual(header["template"], "blue")

        # Completed → green
        project.status = WorkflowStatus.COMPLETED
        header = renderer._render_header()
        self.assertEqual(header["template"], "green")

        # Failed → red
        project.status = WorkflowStatus.FAILED
        header = renderer._render_header()
        self.assertEqual(header["template"], "red")

    def test_progress_card_collapsible_headers_do_not_use_template(self):
        """Feishu Schema 2.0 rejects collapsible_panel.header.template."""
        project = self._make_project()
        renderer = WorkflowProgressRenderer(project)
        card = renderer.render_progress_card()

        violations = [
            node
            for node in self._walk_card_nodes(card.get("elements", []))
            if node.get("tag") == "collapsible_panel"
            and isinstance(node.get("header"), dict)
            and "template" in node["header"]
        ]

        self.assertEqual(violations, [])

    def test_cancelled_agents_in_mixed_state_render_correctly(self):
        """CANCELLED agents must appear in a "已取消" grey group and NOT
        alongside RUNNING agents in the "执行中" group."""
        project = WorkflowProject(
            name="mixed-cancel-test",
            status=WorkflowStatus.RUNNING,
            started_at=time.time() - 60,
        )
        phase = PhaseProgress(title="Race Phase", started_at=time.time() - 60)

        # 2 RUNNING
        phase.agents.append(
            AgentProgress(
                label="runner-1",
                tool="coco",
                status=AgentStatus.RUNNING,
            )
        )
        phase.agents.append(
            AgentProgress(
                label="runner-2",
                tool="claude",
                status=AgentStatus.RUNNING,
            )
        )
        # 2 DONE
        phase.agents.append(
            AgentProgress(
                label="finisher-1",
                tool="coco",
                status=AgentStatus.DONE,
                duration_s=10.0,
            )
        )
        phase.agents.append(
            AgentProgress(
                label="finisher-2",
                tool="claude",
                status=AgentStatus.DONE,
                duration_s=12.0,
            )
        )
        # 1 FAILED
        phase.agents.append(
            AgentProgress(
                label="flaky-agent",
                tool="coco",
                status=AgentStatus.FAILED,
                duration_s=3.0,
                error="boom",
            )
        )
        # 2 CANCELLED
        phase.agents.append(
            AgentProgress(
                label="slow-agent",
                tool="coco",
                status=AgentStatus.CANCELLED,
                duration_s=5.0,
            )
        )
        phase.agents.append(
            AgentProgress(
                label="race-loser",
                tool="claude",
                status=AgentStatus.CANCELLED,
                duration_s=2.0,
            )
        )

        project.phases.append(phase)
        project.metrics.total_agents = 7
        project.metrics.completed_agents = 2

        renderer = WorkflowProgressRenderer(project)
        card = renderer.render_progress_card()

        all_text = self._extract_all_text(card.get("elements", []))

        # CANCELLED group header must be present
        self.assertIn("已取消", all_text)
        # CANCELLED agent labels must appear somewhere in the card
        self.assertIn("slow-agent", all_text)
        self.assertIn("race-loser", all_text)
        # RUNNING group header must be present (there ARE running agents)
        self.assertIn("执行中", all_text)
        # RUNNING agent labels must appear
        self.assertIn("runner-1", all_text)
        self.assertIn("runner-2", all_text)

        # To verify CANCELLED agents are NOT in the RUNNING group, we inspect
        # each collapsible_panel individually: the panel whose header contains
        # "执行中" must not mention the CANCELLED agent labels.
        found_running_panel = False
        for node in self._walk_card_nodes(card.get("elements", [])):
            if node.get("tag") != "collapsible_panel":
                continue
            header_text = self._panel_header_text(node)
            if "执行中" in header_text:
                found_running_panel = True
                panel_text = self._extract_all_text(node.get("elements", []))
                self.assertIn("runner-1", panel_text)
                self.assertIn("runner-2", panel_text)
                self.assertNotIn("slow-agent", panel_text)
                self.assertNotIn("race-loser", panel_text)

        self.assertTrue(found_running_panel, "Expected a RUNNING collapsible panel")

        # Also verify the CANCELLED panel contains the cancelled agents
        found_cancelled_panel = False
        for node in self._walk_card_nodes(card.get("elements", [])):
            if node.get("tag") != "collapsible_panel":
                continue
            header_text = self._panel_header_text(node)
            if "已取消" in header_text:
                found_cancelled_panel = True
                panel_text = self._extract_all_text(node.get("elements", []))
                self.assertIn("slow-agent", panel_text)
                self.assertIn("race-loser", panel_text)
                self.assertNotIn("runner-1", panel_text)

        self.assertTrue(found_cancelled_panel, "Expected a CANCELLED collapsible panel")

    @staticmethod
    def _panel_header_text(panel):
        """Extract the header title text from a collapsible_panel element."""
        header = panel.get("header", {})
        title = header.get("title", {})
        if isinstance(title, dict):
            return title.get("content", "")
        return ""

    @staticmethod
    def _extract_all_text(elements):
        """Recursively extract all text content from card elements
        (markdown content, plain_text fields, and collapsible_panel headers)."""
        out = []
        for node in TestWorkflowProgressRenderer._walk_card_nodes(elements):
            if node.get("tag") == "markdown":
                out.append(node.get("content", ""))
            if node.get("tag") == "plain_text":
                out.append(node.get("text", "") or node.get("content", ""))
            title = node.get("title")
            if isinstance(title, dict) and title.get("tag") == "plain_text":
                out.append(title.get("content", ""))
            # Also pull header.title.content from collapsible_panel
            if node.get("tag") == "collapsible_panel":
                header = node.get("header", {})
                htitle = header.get("title", {})
                if isinstance(htitle, dict) and htitle.get("tag") == "plain_text":
                    out.append(htitle.get("content", ""))
        return "\n".join(out)

    @staticmethod
    def _walk_card_nodes(nodes):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            yield node
            nested = node.get("elements")
            if isinstance(nested, list):
                yield from TestWorkflowProgressRenderer._walk_card_nodes(nested)
            columns = node.get("columns")
            if isinstance(columns, list):
                for column in columns:
                    if isinstance(column, dict):
                        yield from TestWorkflowProgressRenderer._walk_card_nodes(column.get("elements", []))


class TestConfirmCardStructure(unittest.TestCase):
    """Test _build_confirm_card output structure from WorkflowHandler."""

    def _make_handler(self):
        from unittest.mock import MagicMock

        from src.feishu.handlers.workflow import WorkflowHandler

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = MagicMock()
        return handler

    def _make_meta(self, phases=None, tools=None):
        return {
            "name": "test-workflow",
            "description": "A test workflow",
            "phases": phases or [],
            "tools": tools or ["coco"],
        }

    def _build_card(self, meta=None, requirement="test requirement", is_fallback=False, selected_tools=None):
        handler = self._make_handler()
        return handler._build_confirm_card(
            meta=meta or self._make_meta(),
            requirement=requirement,
            engine_session_key="test-session-key-123",
            chat_id="chat_001",
            project_id="proj_001",
            is_fallback=is_fallback,
            selected_tools=selected_tools,
        )

    def _get_elements(self, card):
        return card["body"]["elements"]

    def test_confirm_card_has_phases_list(self):
        """Card should contain markdown listing all phase titles.

        Phases live inside a collapsible_panel (not a top-level markdown element), so we must
        search inside collapsible_panel elements too.
        """
        phases = [
            {"title": "Analysis", "detail": "Analyze codebase"},
            {"title": "Implementation", "detail": "Write code"},
            {"title": "Testing", "detail": "Run tests"},
        ]
        meta = self._make_meta(phases=phases)
        card = self._build_card(meta=meta)
        elements = self._get_elements(card)

        # Recursively collect markdown text from the card (including inside collapsible_panel)
        def flatten_md(els: list[dict]) -> str:
            out: list[str] = []
            for e in els:
                if e.get("tag") == "markdown":
                    out.append(e.get("content", ""))
                if e.get("tag") == "collapsible_panel":
                    out.append(flatten_md(e.get("elements", [])))
            return "\n".join(out)

        all_text = flatten_md(elements)
        for p in phases:
            self.assertIn(p["title"], all_text)

    def test_confirm_card_has_tool_buttons(self):
        """Card should contain tool toggle buttons (in column_set for Schema 2.0 compliance)."""
        meta = self._make_meta(tools=["coco", "claude"])
        card = self._build_card(meta=meta)
        elements = self._get_elements(card)

        # Collect all buttons from both legacy action containers and Schema 2.0 column_set
        all_buttons = []
        for e in elements:
            if e.get("tag") == "action":
                all_buttons.extend(e.get("actions", []))
            if e.get("tag") == "column_set":
                for col in e.get("columns", []):
                    for col_el in col.get("elements", []):
                        if col_el.get("tag") == "button":
                            all_buttons.append(col_el)
            # Also check inside collapsible_panel for "更多工具" buttons
            if e.get("tag") == "collapsible_panel":
                for inner in e.get("elements", []):
                    if inner.get("tag") == "action":
                        all_buttons.extend(inner.get("actions", []))
                    if inner.get("tag") == "column_set":
                        for col in inner.get("columns", []):
                            for col_el in col.get("elements", []):
                                if col_el.get("tag") == "button":
                                    all_buttons.append(col_el)

        tool_buttons = [btn for btn in all_buttons if btn.get("value", {}).get("action") == "workflow_select_tool"]
        self.assertGreater(len(tool_buttons), 0, "Expected at least one tool selection button")
        button_tool_names = [btn["value"]["tool_name"] for btn in tool_buttons]
        self.assertIn("coco", button_tool_names)
        self.assertIn("claude", button_tool_names)

    def test_confirm_card_has_confirm_cancel_buttons(self):
        """Card should have confirm and cancel buttons (in column_set for Schema 2.0 compliance)."""
        card = self._build_card()
        elements = self._get_elements(card)

        # Collect all buttons from both legacy action containers and Schema 2.0 column_set
        all_buttons = []
        for e in elements:
            if e.get("tag") == "action":
                all_buttons.extend(e.get("actions", []))
            if e.get("tag") == "column_set":
                for col in e.get("columns", []):
                    for col_el in col.get("elements", []):
                        if col_el.get("tag") == "button":
                            all_buttons.append(col_el)

        button_actions = [btn["value"]["action"] for btn in all_buttons]
        self.assertIn("workflow_confirm_start", button_actions)
        self.assertIn("workflow_cancel", button_actions)

    def test_confirm_card_fallback_shows_warning(self):
        """Card should show fallback warning when is_fallback=True."""
        card = self._build_card(is_fallback=True)
        elements = self._get_elements(card)

        warning_elements = [e for e in elements if e.get("tag") == "markdown" and "默认模板" in e.get("content", "")]
        self.assertGreater(len(warning_elements), 0, "Expected a markdown element with fallback warning")

    def test_confirm_card_uses_no_schema_v2_note_tags(self) -> None:
        """Feishu Schema 2.0 rejects ``tag=note`` in workflow confirmation cards."""
        cards = [
            self._build_card(),
            self._build_card(is_fallback=True),
            self._build_card(
                meta=self._make_meta(tools=["coco", "claude"]),
                selected_tools=["coco"],
            ),
            self._build_card(meta={**self._make_meta(), "workflow_refs": []}),
        ]

        for card in cards:
            tags = [node.get("tag") for node in self._walk_elements(self._get_elements(card))]
            self.assertNotIn("note", tags)

    def test_confirm_card_phase_tool_row_uses_two_columns(self) -> None:
        """Phase and tool counts should share a two-column (bisect) row so
        the screen space is used efficiently."""
        card = self._build_card(requirement="mobile layout audit")
        two_column_rows = [
            node
            for node in self._walk_elements(self._get_elements(card))
            if node.get("tag") == "column_set" and len(node.get("columns", [])) == 2
        ]
        self.assertTrue(
            two_column_rows,
            "expected at least one two-column stats row for phases/tools",
        )

    def test_confirm_card_collapsible_headers_do_not_use_template(self) -> None:
        """Feishu Schema 2.0 rejects collapsible_panel.header.template."""
        phases = [
            {"title": "Analysis", "detail": "Analyze the target"},
        ]
        card = self._build_card(
            meta=self._make_meta(phases=phases, tools=["coco"]),
            selected_tools=["coco"],
        )
        elements = self._get_elements(card)

        violations = [
            node
            for node in self._walk_elements(elements)
            if node.get("tag") == "collapsible_panel"
            and isinstance(node.get("header"), dict)
            and "template" in node["header"]
        ]

        self.assertEqual(violations, [])

    @staticmethod
    def _walk_elements(elements):
        for node in elements:
            yield node
            for key in ("columns", "elements"):
                nested = node.get(key)
                if isinstance(nested, list):
                    yield from TestConfirmCardStructure._walk_elements(nested)


class TestRenderScriptPreview(unittest.TestCase):
    """Test render_script_preview truncation and formatting."""

    def test_short_script_complete(self):
        """A script under 80 lines and 2000 chars should be shown in full."""
        script = "const x = 1;\nconsole.log(x);\n"
        result = render_script_preview(script)

        self.assertIn("```javascript", result)
        self.assertIn("const x = 1;", result)
        self.assertIn("console.log(x);", result)
        self.assertNotIn("_(脚本已截断，完整内容将在执行时使用)_", result)

    def test_long_script_truncated_by_lines(self):
        """A script with 100 lines should only show first 80 lines with truncation note."""
        lines = [f"// line {i}" for i in range(100)]
        script = "\n".join(lines)
        result = render_script_preview(script)

        self.assertIn("```javascript", result)
        self.assertIn("// line 0", result)
        self.assertIn("// line 79", result)
        self.assertNotIn("// line 80", result)
        self.assertIn("_(脚本已截断，完整内容将在执行时使用)_", result)

    def test_long_script_truncated_by_chars(self):
        """A script where total chars >2000 but <80 lines should be truncated at 2000 chars."""
        # Each line is ~100 chars, 30 lines = ~3000 chars total > 2000
        lines = [f"var longVariable_{i} = " + "x" * 80 + ";" for i in range(30)]
        script = "\n".join(lines)
        self.assertGreater(len(script), 2000)
        self.assertLess(len(lines), 80)

        result = render_script_preview(script)

        self.assertIn("```javascript", result)
        self.assertIn("_(脚本已截断，完整内容将在执行时使用)_", result)
        # The body inside the fence should be at most 2000 chars
        fence_start = result.index("```javascript\n") + len("```javascript\n")
        fence_end = result.index("\n```")
        body = result[fence_start:fence_end]
        self.assertLessEqual(len(body), 2000)

    def test_empty_script_returns_empty(self):
        """Empty string or whitespace-only string should return empty string."""
        self.assertEqual(render_script_preview(""), "")
        self.assertEqual(render_script_preview("   "), "")
        self.assertEqual(render_script_preview("\n\n"), "")

    def test_custom_limits(self):
        """Passing custom max_lines=5 should truncate at 5 lines."""
        lines = [f"line {i}" for i in range(10)]
        script = "\n".join(lines)
        result = render_script_preview(script, max_lines=5)

        self.assertIn("line 0", result)
        self.assertIn("line 4", result)
        self.assertNotIn("line 5", result)
        self.assertIn("_(脚本已截断，完整内容将在执行时使用)_", result)


class TestRenderCompletionCard(unittest.TestCase):
    """Tests for render_completion_card module-level function."""

    def _make_project(self, status=WorkflowStatus.COMPLETED, **kwargs):
        defaults = {
            "name": "code-audit",
            "requirement": "Audit the repository for security issues",
            "status": status,
            "started_at": time.time() - 120,
            "finished_at": time.time(),
            "phases": [
                PhaseProgress(
                    title="Analysis",
                    agents=[
                        AgentProgress(label="scan", tool="coco", status=AgentStatus.DONE, duration_s=10.0),
                        AgentProgress(label="verify", tool="claude", status=AgentStatus.DONE, duration_s=15.0),
                    ],
                ),
            ],
            "result": "Found 3 issues.",
        }
        defaults.update(kwargs)
        return WorkflowProject(**defaults)

    def test_returns_header_and_elements(self):
        project = self._make_project()
        card = render_completion_card(project)
        self.assertIn("header", card)
        self.assertIn("elements", card)
        self.assertIsInstance(card["elements"], list)
        self.assertGreater(len(card["elements"]), 0)

    def test_completed_status_green_header(self):
        project = self._make_project(status=WorkflowStatus.COMPLETED)
        card = render_completion_card(project)
        self.assertEqual(card["header"]["template"], "green")
        self.assertIn("完成", card["header"]["title"]["content"])

    def test_failed_status_red_header(self):
        project = self._make_project(status=WorkflowStatus.FAILED, error="timeout")
        card = render_completion_card(project)
        self.assertEqual(card["header"]["template"], "red")
        self.assertIn("失败", card["header"]["title"]["content"])

    def test_cancelled_status_grey_header(self):
        project = self._make_project(status=WorkflowStatus.CANCELLED)
        card = render_completion_card(project)
        self.assertEqual(card["header"]["template"], "grey")

    def test_elements_contain_metrics(self):
        project = self._make_project()
        card = render_completion_card(project)

        # Recursively extract all markdown content (including from column_set/columns)
        def extract_markdown(elements):
            content = []
            for e in elements:
                if e.get("tag") == "markdown":
                    content.append(e.get("content", ""))
                elif e.get("tag") == "column_set":
                    for col in e.get("columns", []):
                        content.extend(extract_markdown(col.get("elements", [])))
            return content

        all_content = " ".join(extract_markdown(card["elements"]))
        self.assertIn("耗时", all_content)
        self.assertIn("阶段", all_content)
        self.assertIn("验证", all_content)

    def test_completion_card_keeps_clock_format_for_final_elapsed(self):
        project = self._make_project(
            started_at=100.0,
            finished_at=90_161.0,
        )

        card = render_completion_card(project)
        all_content = self._extract_all_text(card["elements"])

        self.assertIn("1天 01:01:01", all_content)

    def test_elements_contain_phase_summary(self):
        project = self._make_project()
        card = render_completion_card(project)
        all_content = self._extract_all_text(card["elements"])
        self.assertIn("Analysis", all_content)

    def test_unstructured_result_uses_neutral_conclusion(self):
        project = self._make_project(result="Found 3 issues.")
        card = render_completion_card(project)
        all_content = self._extract_all_text(card["elements"])
        self.assertIn("任务已完成，完整结果见报告", all_content)
        self.assertNotIn("Found 3 issues", all_content)

    def test_completed_card_without_result_still_has_report_notice(self):
        project = self._make_project(result="")
        card = render_completion_card(project)
        all_content = self._extract_all_text(card["elements"])

        self.assertIn("结论", all_content)
        self.assertIn("任务已完成，完整结果见报告", all_content)

    def test_failed_shows_error_message(self):
        project = self._make_project(status=WorkflowStatus.FAILED, error="Runtime timeout exceeded")
        card = render_completion_card(project)
        all_content = " ".join(e.get("content", "") for e in card["elements"] if e.get("tag") == "markdown")
        self.assertIn("Runtime timeout", all_content)

    def test_completion_card_surfaces_result_brief_and_process_summary(self):
        result = {
            "card_summary": {
                "verdict": "passed",
                "conclusion": "已增加目标完成度监控，并展示任务结果简报。",
                "findings": [
                    {"severity": "medium", "text": "晚到进度可能覆盖终态结果。"},
                ],
                "verification": [
                    {"status": "passed", "text": "结果、过程和风险回归通过。"},
                ],
                "deliverables": [
                    {"type": "code", "text": "Workflow 完成卡实现。"},
                ],
                "next_steps": ["观察真实飞书移动端阅读体验。"],
            },
        }
        project = self._make_project(
            result=json.dumps(result, ensure_ascii=False),
            phases=[
                PhaseProgress(
                    title="Routing",
                    finished_at=time.time() - 90,
                    agents=[],
                ),
                PhaseProgress(
                    title="Execution",
                    finished_at=time.time() - 30,
                    agents=[
                        AgentProgress(
                            label="execute-traex",
                            tool="traex",
                            status=AgentStatus.DONE,
                            duration_s=30.0,
                        )
                    ],
                ),
                PhaseProgress(
                    title="Verification",
                    finished_at=time.time(),
                    agents=[
                        AgentProgress(
                            label="verify-output",
                            tool="traex",
                            status=AgentStatus.DONE,
                            duration_s=20.0,
                        )
                    ],
                ),
            ],
        )

        card = render_completion_card(project)
        all_content = self._extract_all_text(card["elements"])

        self.assertIn("已增加目标完成度监控", all_content)
        self.assertIn("晚到进度可能覆盖终态结果", all_content)
        self.assertIn("结果、过程和风险回归通过", all_content)
        self.assertIn("Workflow 完成卡实现", all_content)
        self.assertIn("观察真实飞书移动端阅读体验", all_content)
        self.assertIn("阶段", all_content)
        self.assertIn("Routing", all_content)
        self.assertIn("Execution", all_content)
        self.assertIn("Verification", all_content)

    def test_completion_card_with_html_report_shows_complete_brief_without_truncation(self):
        sentinel = "FINAL_SENTINEL_AFTER_LONG_CONTENT"
        result = {
            "card_summary": {
                "verdict": "needs_attention",
                "conclusion": "两项事实错误必须修正。",
                "findings": [
                    {"severity": "high", "text": "Freshness Gate 已有三段式重试闭环。"},
                    {"severity": "low", "text": ("完整长发现" * 1200) + sentinel},
                ],
                "verification": [{"status": "failed", "text": "评审未通过。"}],
                "next_steps": ["修正事实错误后重新评审。"],
            },
            "final_report": ("完整报告 " * 1200) + sentinel,
        }
        project = self._make_project(result=json.dumps(result, ensure_ascii=False))

        card = render_completion_card(
            project,
            report_status={
                "generated": True,
                "attachment_sent": True,
                "html_path": "/tmp/report.html",
            },
        )
        all_content = self._extract_all_text(card["elements"])

        self.assertIn("完整 HTML 报告已发送", all_content)
        self.assertIn("两项事实错误必须修正", all_content)
        self.assertIn("Freshness Gate 已有三段式重试闭环", all_content)
        self.assertIn("评审未通过", all_content)
        self.assertIn("修正事实错误后重新评审", all_content)
        self.assertIn("另有 1 条", all_content)
        self.assertNotIn("内容已截断", all_content)
        self.assertNotIn(sentinel, all_content)

    def test_completion_card_keeps_result_before_collapsed_process(self):
        project = self._make_project(
            result=json.dumps(
                {"card_summary": {"verdict": "passed", "conclusion": "目标已完成。"}},
                ensure_ascii=False,
            )
        )

        card = render_completion_card(project)

        conclusion_index = next(
            index for index, element in enumerate(card["elements"])
            if "目标已完成" in str(element)
        )
        process_index = next(
            index for index, element in enumerate(card["elements"])
            if element.get("tag") == "collapsible_panel" and "执行过程" in str(element)
        )
        self.assertLess(conclusion_index, process_index)
        self.assertFalse(card["elements"][process_index]["expanded"])

    def test_completion_card_stays_under_payload_limit_without_slicing_result_items(self):
        findings = [
            {"severity": "medium", "text": f"完整发现 {index}: " + ("证据" * 120)}
            for index in range(100)
        ]
        project = self._make_project(
            result=json.dumps(
                {
                    "card_summary": {
                        "verdict": "needs_attention",
                        "conclusion": "需要处理。",
                        "findings": findings,
                    }
                },
                ensure_ascii=False,
            )
        )

        card = render_completion_card(
            project,
            report_status={"generated": True, "attachment_sent": True},
        )
        payload = json.dumps(card, ensure_ascii=False).encode("utf-8")
        all_content = self._extract_all_text(card["elements"])

        self.assertLessEqual(len(payload), 28_000)
        self.assertIn("需要处理", all_content)
        self.assertIn("详见报告", all_content)
        self.assertNotIn("内容已截断", all_content)

    def test_workflow_report_files_preserve_full_result_and_escape_html(self):
        """Full HTML/Markdown artifacts should carry untruncated result content safely."""
        from src.workflow_engine.reporting import write_workflow_report_files

        sentinel = "FINAL_SENTINEL_AFTER_LONG_CONTENT"
        result = {
            "final_report": ("完整报告 " * 1200) + sentinel,
            "verification": {
                "summary": "验证通过",
                "raw": "<script>alert('x')</script>",
            },
            "agent_outputs": [
                {
                    "label": "auditor",
                    "output": "agent raw output " * 200,
                }
            ],
        }
        project = self._make_project(result=json.dumps(result, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            cache_root = Path(tmpdir) / "cache"
            project_root.mkdir()
            files = write_workflow_report_files(
                project,
                root_path=str(project_root),
                cache_root=str(cache_root),
            )
            html = Path(files.html_path).read_text(encoding="utf-8")
            markdown = Path(files.markdown_path).read_text(encoding="utf-8")

        self.assertIn(f"{cache_root}", files.html_path)
        self.assertIn("workflow_reports", files.html_path)
        self.assertNotIn(".ghostap", files.html_path)
        self.assertIn(sentinel, html)
        self.assertIn(sentinel, markdown)
        self.assertIn("原始结果 JSON", html)
        self.assertIn("agent raw output", html)
        self.assertIn("&lt;script&gt;alert", html)
        self.assertNotIn("<script>alert", html)
        self.assertIn("function toggleSection", html)

    @staticmethod
    def _extract_all_text(elements):
        out = []
        for node in TestWorkflowProgressRenderer._walk_card_nodes(elements):
            if node.get("tag") == "markdown":
                out.append(node.get("content", ""))
            if node.get("tag") == "plain_text":
                out.append(node.get("text", "") or node.get("content", ""))
            title = node.get("title")
            if isinstance(title, dict) and title.get("tag") == "plain_text":
                out.append(title.get("content", ""))
        return "\n".join(out)


class TestAgentOutputDefensiveCheck(unittest.TestCase):
    """Test the AC4 defensive gate that trips when a sentinel appears in card text."""

    SENTINEL = "AC4_SENTINEL_OUTPUT_XYZ"

    def _make_project(self, **kwargs):
        defaults = {
            "name": "audit",
            "status": WorkflowStatus.RUNNING,
            "started_at": time.time() - 60,
        }
        defaults.update(kwargs)
        return WorkflowProject(**defaults)

    # ------------------------------------------------------------------
    # _card_text_for_agent_output unit behaviour
    # ------------------------------------------------------------------

    def test_empty_markers_is_noop(self):
        """An empty marker tuple must never raise, even if elements exist."""
        elements = [_md_element("plain content")]
        # Should not raise.
        _card_text_for_agent_output(elements, ())

    def test_raises_on_content_field_match(self):
        elements = [_md_element(f"prefix {self.SENTINEL} suffix")]
        with self.assertRaises(RuntimeError) as ctx:
            _card_text_for_agent_output(elements, (self.SENTINEL,))
        self.assertIn("card leaked agent output", str(ctx.exception))

    def test_raises_on_text_field_match(self):
        elements = [{"tag": "note", "elements": [{"tag": "plain_text", "text": self.SENTINEL}]}]
        with self.assertRaises(RuntimeError):
            _card_text_for_agent_output(elements, (self.SENTINEL,))

    def test_no_raise_without_match(self):
        elements = [
            _md_element("safe line 1"),
            {"tag": "column_set", "columns": [{"tag": "column", "elements": [_md_element("safe line 2")]}]},
        ]
        # Should not raise.
        _card_text_for_agent_output(elements, (self.SENTINEL,))

    def test_marker_constant_starts_empty(self):
        # Production default is an empty tuple so the gate is a no-op.
        self.assertEqual(_AGENT_OUTPUT_FORBIDDEN_MARKERS, ())

    # ------------------------------------------------------------------
    # Integration: monkey-patch the module-level constant and verify
    # render_progress_card / render_completion_card trip the gate.
    # ------------------------------------------------------------------

    def test_progress_card_trips_gate_on_leaked_result(self):
        import src.workflow_engine.renderer as renderer_mod

        original = getattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS")
        try:
            setattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS", (self.SENTINEL,))
            # Inject the sentinel into an agent label so it leaks into the
            # rendered phase section text, simulating an accidental
            # agent-output leak.
            project = self._make_project()
            phase = PhaseProgress(
                title="Analysis",
                started_at=time.time() - 60,
            )
            phase.agents.append(
                AgentProgress(
                    label=f"leaked-{self.SENTINEL}-label",
                    tool="coco",
                    status=AgentStatus.DONE,
                    duration_s=5.0,
                    token_usage=1000,
                )
            )
            project.phases.append(phase)
            project.metrics.total_agents = 1
            project.metrics.completed_agents = 1
            renderer = WorkflowProgressRenderer(project)
            with self.assertRaises(RuntimeError) as ctx:
                renderer.render_progress_card()
            self.assertIn("card leaked agent output", str(ctx.exception))
        finally:
            setattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS", original)

    def test_progress_card_clean_project_passes(self):
        import src.workflow_engine.renderer as renderer_mod

        original = getattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS")
        try:
            setattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS", (self.SENTINEL,))
            project = self._make_project(
                requirement="normal requirement",
                result="normal result",
            )
            renderer = WorkflowProgressRenderer(project)
            card = renderer.render_progress_card()
            self.assertIn("elements", card)
            self.assertIsInstance(card["elements"], list)
        finally:
            setattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS", original)

    def test_completion_card_trips_gate_on_leaked_result(self):
        import src.workflow_engine.renderer as renderer_mod

        original = getattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS")
        try:
            setattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS", (self.SENTINEL,))
            project = self._make_project(
                status=WorkflowStatus.COMPLETED,
                result=json.dumps(
                    {
                        "card_summary": {
                            "verdict": "passed",
                            "conclusion": f"leaked {self.SENTINEL} here",
                        }
                    }
                ),
            )
            with self.assertRaises(RuntimeError) as ctx:
                render_completion_card(project)
            self.assertIn("card leaked agent output", str(ctx.exception))
        finally:
            setattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS", original)

    def test_completion_card_clean_project_passes(self):
        import src.workflow_engine.renderer as renderer_mod

        original = getattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS")
        try:
            setattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS", (self.SENTINEL,))
            project = self._make_project(
                status=WorkflowStatus.COMPLETED,
                requirement="audit the repo",
                result="found no issues",
            )
            card = render_completion_card(project)
            self.assertIn("elements", card)
            self.assertIsInstance(card["elements"], list)
        finally:
            setattr(renderer_mod, "_AGENT_OUTPUT_FORBIDDEN_MARKERS", original)


class TestMiddleEllipsisLabelSafety(unittest.TestCase):
    """WorkflowProgressRenderer must keep long phase/agent labels readable
    on mobile by emitting a middle-ellipsis form rather than raw text that
    would overflow the card width."""

    LONG_PHASE = "payment-gateway: migrate-checkout-flow-and-verify-3ds2-compliance-phase"
    LONG_AGENT = "agent:generate-migration-script-for-checkout-payment-methods-upgrade"

    def _make_project_with_long_labels(self):
        from src.workflow_engine.models import (
            AgentProgress,
            PhaseProgress,
            WorkflowProject,
            WorkflowStatus,
        )

        project = WorkflowProject(
            workflow_id="w1",
            status=WorkflowStatus.RUNNING,
            name="audit",
        )
        project.phases = [
            PhaseProgress(
                index=1,
                title=self.LONG_PHASE,
                started_at=1_700_000_000.0,
                agents=[
                    AgentProgress(
                        label=self.LONG_AGENT,
                        tool="coco",
                        status=AgentStatus.RUNNING,
                        started_at=1_700_000_000.0,
                    ),
                ],
            ),
        ]
        return project

    def test_progress_summary_truncates_long_labels(self) -> None:
        from src.workflow_engine.renderer import WorkflowProgressRenderer

        project = self._make_project_with_long_labels()
        renderer = WorkflowProgressRenderer(project)
        summary = renderer._render_summary_section()
        self.assertIsNotNone(summary)
        content = _markdown_content(summary) or ""
        # Middle ellipsis must appear: no raw 60+ char title on the card.
        self.assertIn("…", content)
        # Head of each label must still be visible so the operator can
        # disambiguate phases/agents.
        self.assertIn("payment-gateway", content)
        self.assertIn("agent:generate", content)

    def test_progress_card_truncates_phase_and_agent_labels(self) -> None:
        import json as _json

        from src.workflow_engine.renderer import WorkflowProgressRenderer

        project = self._make_project_with_long_labels()
        renderer = WorkflowProgressRenderer(project)
        card = renderer.render_progress_card()
        raw = _json.dumps(card, ensure_ascii=False)

        # The raw title must NOT appear verbatim — it would otherwise spill
        # off a mobile card.
        self.assertNotIn(self.LONG_PHASE, raw)
        self.assertNotIn(self.LONG_AGENT, raw)
        # But a truncated form (head + …) should remain readable.
        self.assertIn("payment-gateway", raw)
        self.assertIn("agent:generate", raw)
        self.assertIn("…", raw)


class TestWorkflowTerminalProgressRendering(unittest.TestCase):
    """Terminal progress cards must not look like still-running workflows."""

    def test_failed_summary_does_not_say_currently_running(self) -> None:
        project = WorkflowProject(
            workflow_id="w-terminal",
            status=WorkflowStatus.FAILED,
            name="terminal",
            error="Execution: agent_call timed out",
        )
        project.phases = [
            PhaseProgress(
                title="Execution",
                started_at=1_700_000_000.0,
                finished_at=1_700_000_120.0,
                agents=[
                    AgentProgress(
                        label="execute-traex",
                        tool="traex",
                        status=AgentStatus.FAILED,
                        error="agent_call timed out",
                        started_at=1_700_000_000.0,
                        finished_at=1_700_000_120.0,
                    ),
                ],
            )
        ]

        renderer = WorkflowProgressRenderer(project)
        summary = renderer._render_summary_section()
        self.assertIsNotNone(summary)
        content = _markdown_content(summary) or ""
        self.assertIn("执行已失败", content)
        self.assertNotIn("当前执行中", content)

    def test_finished_empty_phase_is_not_rendered_as_waiting(self) -> None:
        project = WorkflowProject(
            workflow_id="w-empty-phase",
            status=WorkflowStatus.FAILED,
            name="empty-phase",
        )
        project.phases = [
            PhaseProgress(
                title="Routing",
                started_at=1_700_000_000.0,
                finished_at=1_700_000_001.0,
                agents=[],
            )
        ]

        renderer = WorkflowProgressRenderer(project)
        card = renderer.render_progress_card()
        import json as _json

        raw = _json.dumps(card, ensure_ascii=False)
        self.assertIn("已完成 0/0", raw)
        self.assertNotIn("等待中", raw)


class TestRunningAgentElapsed(unittest.TestCase):
    """Task B1: RUNNING agents show a live elapsed-time counter derived from
    ``started_at`` without leaking any agent output."""

    def _make_running_project(self, started_at):
        """Build a RUNNING project with a single RUNNING agent."""
        project = WorkflowProject(
            name="elapsed-test",
            status=WorkflowStatus.RUNNING,
            started_at=time.time() - 200,
        )
        phase = PhaseProgress(
            title="Execution",
            started_at=time.time() - 200,
        )
        phase.agents.append(
            AgentProgress(
                label="long-runner",
                tool="coco",
                status=AgentStatus.RUNNING,
                started_at=started_at,
            )
        )
        project.phases.append(phase)
        project.metrics.total_agents = 1
        project.metrics.completed_agents = 0
        return project

    def test_running_agent_renders_duration_token(self):
        """A RUNNING agent with started_at ~125s ago shows a "2m" token in the
        RUNNING panel text and keeps the trailing ellipsis."""
        project = self._make_running_project(started_at=time.time() - 125)
        renderer = WorkflowProgressRenderer(project)
        card = renderer.render_progress_card()

        all_text = TestWorkflowProgressRenderer._extract_all_text(card.get("elements", []))
        # 125s → "2m5s" (approx); assert the minute token appears in a
        # "执行中 <dur>…" fragment.
        self.assertIn("执行中 2m", all_text)
        self.assertIn("…", all_text)

    def test_running_agent_without_started_at_does_not_crash(self):
        """A RUNNING agent with started_at=None must render the plain
        "执行中…" text with no duration and must not raise."""
        project = self._make_running_project(started_at=None)
        renderer = WorkflowProgressRenderer(project)
        card = renderer.render_progress_card()

        all_text = TestWorkflowProgressRenderer._extract_all_text(card.get("elements", []))
        self.assertIn("执行中…", all_text)
        # No duration token should be attached to the running line. The RUNNING
        # panel header itself renders as "执行中 (1)", so match specifically on
        # a digit following "执行中 " to avoid that false positive.
        import re as _re

        self.assertIsNone(_re.search(r"执行中 \d", all_text))

    def test_summary_shows_running_elapsed_line(self):
        """The top "当前执行中" summary shows a "已运行" elapsed line for a
        genuinely running agent."""
        project = self._make_running_project(started_at=time.time() - 125)
        renderer = WorkflowProgressRenderer(project)
        summary = renderer._render_summary_section()
        self.assertIsNotNone(summary)
        content = _markdown_content(summary) or ""
        self.assertIn("已运行", content)
        self.assertIn("2m", content)

    def test_summary_no_running_elapsed_in_terminal_state(self):
        """Terminal workflows must NOT show the "已运行" line even if the last
        agent still carries a started_at."""
        project = self._make_running_project(started_at=time.time() - 125)
        # Flip to a terminal state and mark the agent finished.
        project.status = WorkflowStatus.COMPLETED
        project.finished_at = time.time()
        agent = project.phases[0].agents[0]
        agent.status = AgentStatus.DONE
        agent.finished_at = time.time()
        agent.duration_s = 125.0

        renderer = WorkflowProgressRenderer(project)
        summary = renderer._render_summary_section()
        self.assertIsNotNone(summary)
        content = _markdown_content(summary) or ""
        self.assertNotIn("已运行", content)


def _markdown_content(element) -> str | None:
    """Best-effort extract of the markdown text inside a render element."""
    if not isinstance(element, dict):
        return None
    if element.get("tag") == "markdown":
        return element.get("content", "")
    for value in element.values():
        if isinstance(value, list):
            pieces = [_markdown_content(item) for item in value]
            joined = "\n".join(p for p in pieces if p)
            if joined:
                return joined
        if isinstance(value, dict):
            inner = _markdown_content(value)
            if inner:
                return inner
    return None


if __name__ == "__main__":
    unittest.main()
