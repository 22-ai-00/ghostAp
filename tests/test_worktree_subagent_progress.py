from __future__ import annotations

from dataclasses import dataclass

from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
from src.worktree_engine.dispatcher import WorktreeDispatcher
from src.worktree_engine.models import WorktreeSelectionItem, WorktreeUnit
from src.worktree_engine.review_adapter import WorktreeReviewAdapter
from src.worktree_engine.subagent_progress import WorktreeSubagentProgress

_SOURCE_ID = "019c15f8-683e-7770-a94d-c77bb6503654"


def _collaboration_event(
    *,
    call_id: str,
    status: str,
    message: str,
) -> ACPEvent:
    return ACPEvent(
        event_type=ACPEventType.TOOL_CALL_UPDATE,
        tool_call=ToolCallInfo(
            id=call_id,
            title="spawn_agent",
            kind="other",
            status="in_progress",
            collaboration_tool="spawn_agent",
            collaboration_receivers=(_SOURCE_ID,),
            subagent_states=(
                {
                    "source_id": _SOURCE_ID,
                    "status": status,
                    "message": message,
                },
            ),
        ),
    )


def _activity_event(
    *,
    call_id: str,
    activity: str = "started",
    status: str = "completed",
) -> ACPEvent:
    return ACPEvent(
        event_type=ACPEventType.TOOL_CALL_DONE,
        source_id=_SOURCE_ID,
        tool_call=ToolCallInfo(
            id=call_id,
            title=f"subagent {activity}",
            kind="other",
            status=status,
            subagent_source_id=_SOURCE_ID,
            subagent_activity=activity,
        ),
    )


@dataclass
class _PromptResult:
    text: str
    stop_reason: str = "end_turn"
    tool_results: list[dict] | None = None


def test_dispatcher_projects_safe_stable_subagent_progress_to_unit(tmp_path):
    updates: list[str] = []
    sessions: list[object] = []

    class _Session:
        def __init__(self, **_kwargs):
            self.received_on_event = None
            sessions.append(self)

        def start(self):
            return "ok"

        def send_prompt(self, _prompt, *, on_event=None, timeout=None):
            del timeout
            self.received_on_event = on_event
            if on_event is not None:
                on_event(
                    _collaboration_event(
                        call_id="spawn-call-1",
                        status="running",
                        message=(
                            f"正在核查 {_SOURCE_ID} "
                            "API_TOKEN=sk-0123456789abcdef"
                        ),
                    )
                )
                # The activity operation completed, but the child itself is
                # still running. A different call id must not create #2.
                on_event(_activity_event(call_id="activity-call-2"))
                on_event(
                    _collaboration_event(
                        call_id="wait-call-3",
                        status="completed",
                        message="核查完成，相关测试通过",
                    )
                )
            return _PromptResult(text="unit done")

        def close(self):
            return None

    unit = WorktreeUnit(unit_id="u1", worktree_path=str(tmp_path))
    tool = WorktreeSelectionItem(
        provider="acp",
        tool_name="codex",
        display_name="Codex",
    )
    dispatcher = WorktreeDispatcher(session_factory=lambda **kwargs: _Session(**kwargs))
    planned = dispatcher.plan_user_goal("inspect cards", [unit], [tool])

    result = dispatcher.execute_units(
        planned,
        max_workers=1,
        on_unit_update=lambda current: updates.append(current.summary),
    )

    assert sessions[0].received_on_event is not None
    assert any("子Agent #1" in update and "正在核查" in update for update in updates)
    activity_update = next(update for update in updates if "已启动" in update)
    assert "执行中" in activity_update
    assert "已完成" not in activity_update
    assert "正在核查" in activity_update
    assert not any("子Agent #2" in update for update in updates)
    assert any(
        "子Agent #1" in update
        and "已完成" in update
        and "相关测试通过" in update
        for update in updates
    )
    rendered_updates = "\n".join(updates)
    assert _SOURCE_ID not in rendered_updates
    assert "spawn-call-1" not in rendered_updates
    assert "activity-call-2" not in rendered_updates
    assert "sk-0123456789abcdef" not in rendered_updates
    assert result[0].summary == "unit done"


def test_failed_interrupt_activity_does_not_cancel_worktree_child():
    updates: list[str] = []
    unit = WorktreeUnit(unit_id="u1")
    progress = WorktreeSubagentProgress(
        unit,
        on_unit_update=lambda current: updates.append(current.summary),
    )

    progress.on_event(
        _collaboration_event(
            call_id="spawn-call",
            status="running",
            message="正在核查卡片",
        )
    )
    progress.on_event(
        _activity_event(
            call_id="interrupt-call",
            activity="interrupted",
            status="failed",
        )
    )

    assert updates[-1].startswith("子Agent #1 执行中")
    assert "中断未完成" in updates[-1]
    assert "已中断" not in updates[-1]


def test_worktree_progress_hides_peer_ids_paths_and_markdown():
    source_b = "019c15f8-bbbb-7770-a94d-c77bb6503654"
    updates: list[str] = []
    unit = WorktreeUnit(unit_id="u1")
    progress = WorktreeSubagentProgress(
        unit,
        on_unit_update=lambda current: updates.append(current.summary),
    )
    progress.on_event(ACPEvent(
        event_type=ACPEventType.TOOL_CALL_UPDATE,
        tool_call=ToolCallInfo(
            id="call-private-progress",
            title="spawn_agent",
            kind="other",
            status="in_progress",
            collaboration_tool="spawn_agent",
            collaboration_receivers=(_SOURCE_ID, source_b),
            subagent_states=(
                {
                    "source_id": _SOURCE_ID,
                    "status": "running",
                    "message": (
                        f"检查 {source_b} /data00/home/user/private.py:12 "
                        "[详情](file:///tmp/private.md)"
                    ),
                },
                {
                    "source_id": source_b,
                    "status": "running",
                    "message": "等待执行",
                },
            ),
        ),
    ))

    visible = "\n".join(updates)
    assert _SOURCE_ID not in visible
    assert source_b not in visible
    assert "/data00/home/user/private.py" not in visible
    assert "](file:" not in visible


def test_review_projects_subagent_progress_without_losing_execution_summary(tmp_path):
    updates: list[str] = []
    sessions: list[object] = []
    response = """{
      "verdict": "PASS",
      "summary": "review passed",
      "tests": [{"command": "uv run pytest -q", "passed": true, "evidence": "12 passed"}],
      "findings": [{"severity": "observation", "message": "verified", "evidence": "tests passed"}]
    }"""

    class _ReviewSession:
        def __init__(self):
            self.received_on_event = None
            sessions.append(self)

        def send_prompt(self, _prompt, *, on_event=None, timeout=None):
            del timeout
            self.received_on_event = on_event
            if on_event is not None:
                on_event(
                    _collaboration_event(
                        call_id="review-spawn-call",
                        status="running",
                        message=f"复核边界 {_SOURCE_ID}",
                    )
                )
            return _PromptResult(
                text=response,
                tool_results=[
                    {
                        "kind": "execute",
                        "data": {
                            "command": "uv run pytest -q",
                            "exit_code": 0,
                        },
                    }
                ],
            )

        def close(self):
            if self.received_on_event is not None:
                self.received_on_event(
                    _collaboration_event(
                        call_id="late-review-call",
                        status="running",
                        message="迟到的评审事件",
                    )
                )
            return None

    adapter = WorktreeReviewAdapter(
        session_factory=lambda **_kwargs: _ReviewSession()
    )
    unit = WorktreeUnit(
        unit_id="review-unit",
        provider="acp",
        tool_name="codex",
        worktree_path=str(tmp_path),
        summary="implementation complete",
    )

    outcome = adapter.review_unit(
        goal="inspect cards",
        unit=unit,
        timeout=5,
        on_unit_update=lambda current: updates.append(current.summary),
    )

    assert sessions[0].received_on_event is not None
    assert any(
        "implementation complete" in update
        and "评审子Agent #1" in update
        and "复核边界" in update
        for update in updates
    )
    assert _SOURCE_ID not in "\n".join(updates)
    assert unit.summary == "implementation complete"
    assert outcome.passed is True
