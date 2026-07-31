"""Tests for worktree execute_goal flow — parallel execution and progress callbacks."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

from src.project.context import ProjectContext
from src.worktree_engine.dispatcher import WorktreeDispatcher
from src.worktree_engine.manager import WorktreeManager
from src.worktree_engine.models import WorktreeUnit
from src.worktree_engine.review_adapter import (
    WorktreeReviewOutcome,
    WorktreeReviewPlan,
    WorktreeReviewVerdict,
)

# ---------------------------------------------------------------------------
# Fake session for testing
# ---------------------------------------------------------------------------

@dataclass
class FakePromptResult:
    stop_reason: str
    text: str


class FakeSession:
    def __init__(self, *, provider, tool_name, working_dir, model_name=None, ttadk_use_pty=False):
        self.provider = provider
        self.tool_name = tool_name
        self.working_dir = working_dir
        self.model_name = model_name

    def start(self, startup_timeout=60):
        return "session"

    def send_prompt(self, text, on_event=None, timeout=None):
        Path(self.working_dir, f"{self.tool_name}.txt").write_text(text, encoding="utf-8")
        return FakePromptResult(stop_reason="end_turn", text=f"done:{self.tool_name}")

    def close(self):
        return None


def _enable_passing_review(manager: WorktreeManager) -> None:
    manager._review_adapter = MagicMock()
    manager._review_adapter.plan_roles.return_value = WorktreeReviewPlan()
    manager._review_adapter.review_units.return_value = WorktreeReviewOutcome(
        verdict=WorktreeReviewVerdict.PASS,
        summary="verified",
        findings=[
            {
                "severity": "observation",
                "message": "verified",
                "evidence": "fake test evidence",
            }
        ],
        tests=[
            {
                "command": "fake tests",
                "passed": True,
                "verified": True,
                "evidence": "passed",
            }
        ],
    )


# ---------------------------------------------------------------------------
# T3: parallel execution with 3 units receiving correct prompts
# ---------------------------------------------------------------------------


def test_execute_goal_creates_sessions_and_runs_parallel(tmp_path):
    """3 units execute in parallel, each receives its own prompt."""
    dirs = [tmp_path / f"wt{i}" for i in range(3)]
    for d in dirs:
        d.mkdir()

    project = ProjectContext(project_id="p1", project_name="P1", root_path=str(tmp_path))

    units = [
        WorktreeUnit(
            unit_id=f"u{i}",
            selection_key=f"acp:{name}:default",
            provider="acp",
            tool_name=name,
            display_name=name.title(),
            worktree_path=str(dirs[i]),
            status="ready",
        )
        for i, name in enumerate(["claude", "codex", "gemini"])
    ]
    project.worktree_state.units = units
    project.worktree_state.enabled = True

    manager = WorktreeManager(project_manager=None)
    manager._dispatcher = WorktreeDispatcher(session_factory=lambda **kw: FakeSession(**kw))
    _enable_passing_review(manager)

    state = manager.execute_goal(project, "实现 worktree 功能")

    # All units should complete
    assert all(u.status == "completed" for u in state.units), [u.status for u in state.units]
    # Each unit wrote its own file in its own worktree dir
    for i, name in enumerate(["claude", "codex", "gemini"]):
        assert (dirs[i] / f"{name}.txt").exists()
    # No cross-contamination
    assert not (dirs[0] / "codex.txt").exists()
    assert not (dirs[1] / "claude.txt").exists()


# ---------------------------------------------------------------------------
# T5: on_unit_update callback fires per unit status change
# ---------------------------------------------------------------------------


def test_execute_goal_progress_callback_fires_per_unit(tmp_path):
    """on_unit_update callback fires at least once per unit (running + completed)."""
    dirs = [tmp_path / f"wt{i}" for i in range(2)]
    for d in dirs:
        d.mkdir()

    project = ProjectContext(project_id="p2", project_name="P2", root_path=str(tmp_path))

    units = [
        WorktreeUnit(
            unit_id=f"u{i}",
            selection_key=f"acp:{name}:default",
            provider="acp",
            tool_name=name,
            display_name=name.title(),
            worktree_path=str(dirs[i]),
            status="ready",
        )
        for i, name in enumerate(["coco", "codex"])
    ]
    project.worktree_state.units = units
    project.worktree_state.enabled = True

    callback_log: list[tuple[str, str]] = []
    log_lock = threading.Lock()

    def on_update(unit):
        with log_lock:
            callback_log.append((unit.unit_id, unit.status))

    manager = WorktreeManager(project_manager=None)
    manager._dispatcher = WorktreeDispatcher(session_factory=lambda **kw: FakeSession(**kw))
    _enable_passing_review(manager)

    manager.execute_goal(project, "测试回调", on_unit_update=on_update)

    # Each unit should have at least 2 callback fires: running + completed
    unit_ids_seen = {uid for uid, _ in callback_log}
    assert "u0" in unit_ids_seen
    assert "u1" in unit_ids_seen

    statuses_by_unit = {}
    for uid, status in callback_log:
        statuses_by_unit.setdefault(uid, []).append(status)

    for uid in ["u0", "u1"]:
        assert "running" in statuses_by_unit[uid]
        assert "completed" in statuses_by_unit[uid]
