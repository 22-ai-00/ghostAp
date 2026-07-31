from __future__ import annotations

from unittest.mock import MagicMock

from src.card.render.worktree import _render_worktree_merge
from src.project.context import ProjectContext
from src.worktree_engine.manager import WorktreeManager
from src.worktree_engine.models import (
    WorktreeJourneyStatus,
    WorktreeSelectionItem,
    WorktreeUnit,
    WorktreeUnitStatus,
)
from src.worktree_engine.review_adapter import (
    WorktreeReviewOutcome,
    WorktreeReviewPlan,
    WorktreeReviewVerdict,
)


def _project_with_units(*units: WorktreeUnit) -> tuple[ProjectContext, WorktreeManager]:
    project = ProjectContext("p-terminal", "Terminal", "/tmp/terminal")
    manager = WorktreeManager(project_manager=None)
    state = manager.get_state(project)
    state.units = list(units)
    state.base_branch = "main"
    state.git_root = "/tmp/terminal"
    state.selection.selected_items = [
        WorktreeSelectionItem(provider="acp", tool_name="coco", display_name="Coco")
    ]
    manager._dispatcher = MagicMock()
    manager._dispatcher.plan_user_goal.side_effect = lambda _goal, planned, _items: planned
    manager._review_adapter = MagicMock()
    return project, manager


def test_failed_unit_prevents_journey_completed_and_merge():
    completed = WorktreeUnit(
        unit_id="ok",
        status=WorktreeUnitStatus.COMPLETED,
        has_changes=True,
        branch_name="ghostap/wt/ok",
    )
    failed = WorktreeUnit(
        unit_id="failed",
        status=WorktreeUnitStatus.FAILED,
        error="tests failed",
        branch_name="ghostap/wt/failed",
    )
    project, manager = _project_with_units(completed, failed)
    manager._dispatcher.execute_units.return_value = [completed, failed]

    state = manager.execute_goal(project, "implement safely")

    assert state.journey.status is WorktreeJourneyStatus.FAILED
    assert state.merge_entry_ready is False
    assert "failed" in state.last_error.lower() or "失败" in state.last_error
    manager._review_adapter.review_units.assert_not_called()

    manager._git = MagicMock()
    _state, merge_results = manager.merge_to_base(project)
    manager._git.merge_branch.assert_not_called()
    assert merge_results
    assert all(result["success"] is False for result in merge_results)


def test_cancelled_unit_prevents_journey_completed():
    cancelled = WorktreeUnit(
        unit_id="cancelled",
        status=WorktreeUnitStatus.CANCELLED,
        error="pool_timeout",
    )
    project, manager = _project_with_units(cancelled)
    manager._dispatcher.execute_units.return_value = [cancelled]

    state = manager.execute_goal(project, "implement safely")

    assert state.journey.status is WorktreeJourneyStatus.FAILED
    assert state.merge_entry_ready is False
    manager._review_adapter.review_units.assert_not_called()


def test_failed_review_prevents_journey_completed_and_merge():
    completed = WorktreeUnit(
        unit_id="reviewed",
        status=WorktreeUnitStatus.COMPLETED,
        branch_name="ghostap/wt/reviewed",
    )
    project, manager = _project_with_units(completed)
    manager._dispatcher.execute_units.return_value = [completed]
    manager._review_adapter.plan_roles.return_value = WorktreeReviewPlan()
    manager._review_adapter.review_units.return_value = WorktreeReviewOutcome(
        verdict=WorktreeReviewVerdict.FAIL,
        summary="verification failed",
        findings=[
            {
                "severity": "blocker",
                "message": "tests failed",
                "evidence": "1 failed",
            }
        ],
        tests=[
            {
                "command": "uv run pytest -q",
                "passed": False,
                "evidence": "1 failed",
            }
        ],
        blockers=[
            {
                "severity": "blocker",
                "message": "tests failed",
                "evidence": "1 failed",
            }
        ],
        error_code="tests_failed",
    )

    state = manager.execute_goal(project, "implement safely")

    assert state.journey.status is WorktreeJourneyStatus.FAILED
    assert state.review_outcome["verdict"] == "FAIL"
    assert state.merge_entry_ready is False

    manager._git = MagicMock()
    manager.merge_to_base(project)
    manager._git.merge_branch.assert_not_called()


def test_merge_note_discloses_worktree_branch_wins_conflict_rule():
    unit = WorktreeUnit(
        unit_id="ok",
        status=WorktreeUnitStatus.COMPLETED,
        has_changes=True,
        branch_name="ghostap/wt/ok",
    )
    project, manager = _project_with_units(unit)
    state = manager._reporter.refresh_state(manager.get_state(project))

    assert state.merge_notes
    assert state.merge_notes[0]["conflict_policy"] == "worktree_branch_wins"
    assert "Worktree 分支" in state.merge_notes[0]["summary"]
    rendered = _render_worktree_merge(
        {
            "merge_notes": state.merge_notes,
            "base_branch": state.base_branch,
        }
    )
    assert "冲突时自动优先采用 Worktree 分支变更" in rendered["content"]
