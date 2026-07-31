from unittest.mock import MagicMock

from src.project.context import ProjectContext
from src.worktree_engine.manager import WorktreeManager
from src.worktree_engine.models import (
    WorktreeJourneyStatus,
    WorktreeSelectionItem,
    WorktreeUnit,
)
from src.worktree_engine.reporter import WorktreeReporter
from src.worktree_engine.review_adapter import (
    WorktreeReviewAdapter,
    WorktreeReviewOutcome,
    WorktreeReviewVerdict,
)
from src.worktree_engine.selection import WorktreeToolOption


def _passing_review_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.plan_roles.side_effect = WorktreeReviewAdapter().plan_roles
    adapter.review_units.return_value = WorktreeReviewOutcome(
        verdict=WorktreeReviewVerdict.PASS,
        summary="review passed",
        findings=[
            {
                "severity": "observation",
                "message": "verified",
                "evidence": "targeted tests passed",
            }
        ],
        tests=[
            {
                "command": "uv run pytest -q",
                "passed": True,
                "verified": True,
                "evidence": "passed",
            }
        ],
        observations=[
            {
                "severity": "observation",
                "message": "verified",
                "evidence": "targeted tests passed",
            }
        ],
    )
    return adapter


def test_reporter_builds_unit_summary_and_merge_notes():
    project = ProjectContext(project_id="p1", project_name="P1", root_path="/tmp/p1")
    state = project.worktree_state
    state.base_branch = "main"
    state.units = [
        WorktreeUnit(
            unit_id="u1",
            selection_key="acp:coco:model-a",
            provider="acp",
            tool_name="coco",
            display_name="Coco",
            model_name="model-a",
            branch_name="ghostap/wt/01-acp-coco-model-a",
            worktree_path="/tmp/w1",
            status="completed",
            task_title="分析与方案",
            summary="分析完成",
            has_changes=True,
        ),
        WorktreeUnit(
            unit_id="u2",
            selection_key="ttadk:codex:model-b",
            provider="ttadk",
            tool_name="codex",
            display_name="Codex",
            model_name="model-b",
            branch_name="ghostap/wt/02-ttadk-codex-model-b",
            worktree_path="/tmp/w2",
            status="completed",
            task_title="审查与汇总",
            summary="审查完成",
            has_changes=False,
        ),
    ]
    state.journey.status = WorktreeJourneyStatus.COMPLETED
    state.review_outcome = {"verdict": "PASS", "passed": True}

    reporter = WorktreeReporter()
    refreshed = reporter.refresh_state(state)

    assert refreshed.merge_entry_ready is True
    assert len(refreshed.summary_lines) == 2
    assert "Coco" in refreshed.summary_lines[0]
    assert "有代码变更" in refreshed.summary_lines[0]
    assert len(refreshed.merge_notes) == 2
    assert refreshed.merge_notes[0]["branch"] == "ghostap/wt/01-acp-coco-model-a"
    assert refreshed.merge_notes[0]["status"] == "ready"
    assert "main" in refreshed.merge_notes[0]["summary"]


def test_execute_goal_refreshes_summary_lines_and_merge_notes(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    project = ProjectContext(project_id="p2", project_name="P2", root_path=str(project_root))
    manager = WorktreeManager(project_manager=None)
    manager._review_adapter = _passing_review_adapter()

    manager.start_selection(project)
    manager.select_tool(project, WorktreeToolOption(provider="acp", tool_name="coco", display_name="Coco"))
    manager.add_pending_item(project, model_name="model-a")
    manager.finalize_selection(project)
    project.worktree_state.units = [
        WorktreeUnit(
            unit_id="u1",
            selection_key="acp:coco:model-a",
            provider="acp",
            tool_name="coco",
            display_name="Coco",
            model_name="model-a",
            branch_name="ghostap/wt/01-acp-coco-model-a",
            worktree_path=str(project_root / "wt1"),
            status="completed",
            task_title="分析与方案",
            summary="分析完成",
        )
    ]

    original_execute_units = manager._dispatcher.execute_units

    def _fake_execute_units(units, timeout=None, max_workers=None, on_unit_update=None):
        for unit in units:
            unit.status = "completed"
            unit.summary = unit.summary or "执行完成"
        return list(units)

    manager._dispatcher.execute_units = _fake_execute_units
    state = manager.execute_goal(project, "实现 worktree 汇总")
    manager._dispatcher.execute_units = original_execute_units

    assert state.summary_lines
    assert state.merge_notes
    assert state.merge_entry_ready is True


def test_reporter_timeout_unit_shows_timeout_summary():
    """Failed unit with timeout error → summary line includes timeout info."""
    reporter = WorktreeReporter()
    units = [
        WorktreeUnit(
            unit_id="u1",
            selection_key="acp:coco:d",
            provider="acp",
            tool_name="coco",
            display_name="Coco",
            status="failed",
            task_title="分析与方案",
            error="执行超时 (30s)",
        ),
        WorktreeUnit(
            unit_id="u2",
            selection_key="acp:codex:d",
            provider="acp",
            tool_name="codex",
            display_name="Codex",
            status="completed",
            task_title="实现与修改",
            summary="实现完成",
            has_changes=True,
        ),
    ]
    lines = reporter.build_unit_summary_lines(units)

    # Failed timeout unit shows timeout error text with new prefix
    assert "🔍 失败原因：执行超时 (30s)" in lines[0]
    assert "❌" in lines[0]
    # Completed unit shows normally
    assert "✅" in lines[1]
    assert "实现完成" in lines[1]


def test_get_unit_display_name_mapping():
    reporter = WorktreeReporter()

    # 1. Explicit display_name
    u1 = WorktreeUnit(unit_id="wt-01", display_name="My Tool")
    assert reporter._get_unit_display_name(u1) == "My Tool"

    # 2. Standard wt-01 maps to 工作空间 A
    u2 = WorktreeUnit(unit_id="wt-01")
    assert reporter._get_unit_display_name(u2) == "工作空间 A"

    # 3. Standard wt-26 maps to 工作空间 Z
    u3 = WorktreeUnit(unit_id="wt-26")
    assert reporter._get_unit_display_name(u3) == "工作空间 Z"

    # 4. Out of range wt-27 falls back to 单元 27
    u4 = WorktreeUnit(unit_id="wt-27")
    assert reporter._get_unit_display_name(u4) == "单元 27"

    # 5. Unknown ID maps to 自动分配中
    u5 = WorktreeUnit(unit_id="unknown")
    assert reporter._get_unit_display_name(u5) == "自动分配中"

    # 6. Suffix but not wt- prefix fallback
    u6 = WorktreeUnit(unit_id="custom-02")
    assert reporter._get_unit_display_name(u6) == "单元 02"


class TestWorktreeRendererGC:
    """Verify WorktreeRenderer proactive GC via on_terminal hook."""

    def test_session_removed_on_terminal(self):
        """When a session fires on_terminal, it's removed from _sessions dict."""
        from unittest.mock import MagicMock

        from src.feishu.renderers.worktree_renderer import WorktreeRenderer, _WorktreeGCHook

        handler = MagicMock()
        handler._handler = handler
        handler.delivery = MagicMock()

        renderer = WorktreeRenderer.__new__(WorktreeRenderer)
        renderer._sessions = {}
        renderer._lock = __import__("threading").Lock()

        # Simulate adding a session
        mock_session = MagicMock()
        mock_session.closed = False
        renderer._sessions["proj1"] = mock_session

        # Create and fire the GC hook
        hook = _WorktreeGCHook(renderer, "proj1")
        hook.on_terminal(state=None, reason="completed")

        assert "proj1" not in renderer._sessions

    def test_gc_hook_idempotent(self):
        """Firing on_terminal twice doesn't crash."""
        from unittest.mock import MagicMock

        from src.feishu.renderers.worktree_renderer import WorktreeRenderer, _WorktreeGCHook

        renderer = WorktreeRenderer.__new__(WorktreeRenderer)
        renderer._sessions = {"proj1": MagicMock()}
        renderer._lock = __import__("threading").Lock()

        hook = _WorktreeGCHook(renderer, "proj1")
        hook.on_terminal(state=None, reason="ttl_expired")
        hook.on_terminal(state=None, reason="ttl_expired")  # second call is no-op

        assert "proj1" not in renderer._sessions


# ---------------------------------------------------------------------------
# Tests merged from test_worktree_review_adapter.py
# ---------------------------------------------------------------------------


def test_review_adapter_derives_programming_roles_from_goal_and_diff():
    adapter = WorktreeReviewAdapter()

    plan = adapter.plan_roles(
        goal="修复 WT 路由",
        changed_files=["src/feishu/dispatcher.py", "tests/test_feishu_dispatcher.py"],
    )

    role_ids = {role.role_id for role in plan.roles}
    assert {"architect", "tester", "integration", "product"} <= role_ids


def test_review_adapter_keeps_blocker_without_evidence_fail_closed():
    adapter = WorktreeReviewAdapter()

    outcome = adapter.aggregate([
        {"role_id": "tester", "severity": "blocker", "evidence": "", "message": "bad"}
    ])

    assert outcome.verdict is WorktreeReviewVerdict.FAIL
    assert outcome.blockers
    assert outcome.passed is False


def test_execute_goal_records_worktree_review_metadata():
    project = ProjectContext("p-review", "Review", "/tmp/review")
    manager = WorktreeManager(project_manager=None)
    state = manager.get_state(project)
    state.units = [
        WorktreeUnit(
            unit_id="u1",
            worktree_path="/tmp/review-wt",
            has_changes=True,
        )
    ]
    state.selection.selected_items = [
        WorktreeSelectionItem(provider="acp", tool_name="coco", display_name="Coco")
    ]
    manager._dispatcher = MagicMock()
    manager._dispatcher.plan_user_goal.side_effect = lambda goal, units, items: units
    manager._dispatcher.execute_units.side_effect = lambda units, **kwargs: [
        setattr(unit, "status", "completed") or unit for unit in units
    ]
    manager._review_adapter = _passing_review_adapter()

    state = manager.execute_goal(project, "修复 auth token 检查")

    role_ids = {role["role_id"] for role in state.review_plan["roles"]}
    assert "security" in role_ids
    assert state.review_outcome["verdict"] == "PASS"
    assert state.review_outcome["passed"] is True
