"""Regression coverage for bounded ordinary ACP prompt continuation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import patch

import pytest

import src.acp as acp
from src.acp.models import (
    ACPEvent,
    ACPGoalInfo,
    PlanEntryInfo,
    PlanInfo,
    PromptResult,
    ToolCallInfo,
)
from src.acp.outcome import PromptOutcome


@dataclass(frozen=True)
class _PromptCall:
    text: str
    on_event: Callable[[ACPEvent], None] | None
    timeout: float | int | None


class _FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeSession:
    def __init__(
        self,
        *results: PromptResult | BaseException,
        on_send: Callable[[int], None] | None = None,
    ) -> None:
        self._results = list(results)
        self._on_send = on_send
        self.calls: list[_PromptCall] = []
        self.continuation_calls: list[_PromptCall] = []
        self._force_dead = False

    def send_prompt(
        self,
        text: str,
        on_event: Callable[[ACPEvent], None] | None = None,
        timeout: float | int | None = None,
    ) -> PromptResult:
        self.calls.append(
            _PromptCall(text=text, on_event=on_event, timeout=timeout)
        )
        if self._on_send is not None:
            self._on_send(len(self.calls))
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def send_continuation_prompt(
        self,
        text: str,
        on_event: Callable[[ACPEvent], None] | None = None,
        timeout: float | int | None = None,
    ) -> PromptResult:
        call = _PromptCall(
            text=text,
            on_event=on_event,
            timeout=timeout,
        )
        self.continuation_calls.append(call)
        return self.send_prompt(
            text,
            on_event=on_event,
            timeout=timeout,
        )


def _pending_result(
    *,
    stop_reason: str = "end_turn",
    active_tool: bool = False,
    pending_count: int = 1,
) -> PromptResult:
    tool_calls = (
        [
            ToolCallInfo(
                id="tool-active",
                title="pytest",
                kind="execute",
                status="in_progress",
            )
        ]
        if active_tool
        else []
    )
    return PromptResult(
        stop_reason=stop_reason,
        text="partial",
        plan=PlanInfo(
            entries=[
                PlanEntryInfo(content="implementation", status="completed"),
                *[
                    PlanEntryInfo(
                        content=f"remaining-{index}",
                        status="pending",
                    )
                    for index in range(pending_count)
                ],
            ]
        ),
        tool_calls=tool_calls,
    )


def _complete_result(text: str = "done") -> PromptResult:
    return PromptResult(
        stop_reason="end_turn",
        text=text,
        plan=PlanInfo(
            entries=[
                PlanEntryInfo(content="implementation", status="completed"),
                PlanEntryInfo(content="verification", status="completed"),
            ]
        ),
    )


def _runner():
    runner = getattr(acp, "run_prompt_with_continuation", None)
    assert callable(runner), "bounded ACP prompt continuation is not implemented"
    return runner


def test_pending_plan_continues_once_on_the_same_session() -> None:
    runner = _runner()
    first_result = _pending_result()
    first_result.modified_files.add("src/first.py")
    final_result = _complete_result()
    final_result.modified_files.add("src/second.py")
    order: list[str] = []
    session = _FakeSession(
        first_result,
        final_result,
        on_send=lambda call_number: order.append(f"send:{call_number}"),
    )
    events: list[ACPEvent] = []
    on_event = events.append
    continuation_starts: list[str] = []
    clock = _FakeClock()

    def mark_continuation() -> None:
        order.append("boundary")
        continuation_starts.append("continuing")

    with patch("src.acp.continuation._monotonic", clock):
        execution = runner(
            session,
            "original task",
            on_event=on_event,
            timeout_s=90,
            finalization_reserve_s=30,
            on_continuation_start=mark_continuation,
        )

    assert execution.result.text == "done"
    assert execution.result.modified_files == {
        "src/first.py",
        "src/second.py",
    }
    assert execution.assessment.outcome is PromptOutcome.COMPLETED
    assert execution.automatic_continuations == 1
    assert len(session.calls) == 2
    assert [call.timeout for call in session.calls] == [60, 60]
    assert session.calls[0].text == "original task"
    assert "自动续做指令" in session.calls[1].text
    assert all(call.on_event is on_event for call in session.calls)
    assert continuation_starts == ["continuing"]
    assert order == ["send:1", "boundary", "send:2"]


def test_pending_plan_stops_after_three_automatic_continuations() -> None:
    runner = _runner()
    session = _FakeSession(*(_pending_result() for _ in range(4)))
    continuation_starts: list[str] = []
    clock = _FakeClock()

    with patch("src.acp.continuation._monotonic", clock):
        execution = runner(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            on_continuation_start=lambda: continuation_starts.append(
                "continuing"
            ),
        )

    assert len(session.calls) == 4
    assert continuation_starts == ["continuing"] * 3
    assert session.continuation_calls == session.calls[1:]
    assert execution.automatic_continuations == 3
    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE
    assert execution.assessment.pending_plan_entries == 1






def test_safe_choice_uses_default_and_continues_without_user_input() -> None:
    runner = _runner()
    session = _FakeSession(
        PromptResult(stop_reason="end_turn", text="请选择 A 或 B 实现方案"),
        _complete_result(),
    )

    execution = runner(
        session,
        "implement the feature",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert len(session.calls) == 2
    assert "自动续做默认决策" in session.calls[1].text
    assert execution.automatic_continuations == 1
    assert execution.assessment.outcome is PromptOutcome.COMPLETED


def test_dangerous_confirmation_never_auto_authorizes_or_reports_success() -> None:
    runner = _runner()
    session = _FakeSession(
        PromptResult(
            stop_reason="end_turn",
            text="请确认是否部署到生产环境并删除旧数据",
        ),
        _complete_result("已拒绝高风险操作并完成其余工作"),
    )

    execution = runner(
        session,
        "prepare the change",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert len(session.calls) == 2
    assert "自动安全决策" in session.calls[1].text
    assert execution.automatic_continuations == 1
    assert execution.assessment.outcome is PromptOutcome.COMPLETED


def test_unresolved_child_gets_a_bounded_reconciliation_turn() -> None:
    runner = _runner()
    first_result = PromptResult(
        stop_reason="end_turn",
        text="reviewer final answer arrived asynchronously",
        tool_calls=[
            ToolCallInfo(
                id="wait-before-final-answer",
                title="wait_agent",
                kind="other",
                status="completed",
                subagent_states=(
                    {"source_id": "reviewer", "status": "running"},
                ),
            )
        ],
    )
    reconciled_result = PromptResult(
        stop_reason="end_turn",
        text="all reviewers are terminal",
        tool_calls=[
            ToolCallInfo(
                id="list-after-final-answer",
                title="list_agents",
                kind="other",
                status="completed",
                subagent_states=(
                    {"source_id": "reviewer", "status": "completed"},
                ),
            )
        ],
    )
    session = _FakeSession(first_result, reconciled_result)
    continuation_starts: list[str] = []
    clock = _FakeClock()

    with patch("src.acp.continuation._monotonic", clock):
        execution = runner(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            on_continuation_start=lambda: continuation_starts.append(
                "reconciling"
            ),
        )

    assert len(session.calls) == 2
    assert "子代理状态对账指令" in session.calls[1].text
    assert "MESSAGE 不代表子代理已进入终态" in session.calls[1].text
    assert "wait_agent -> list_agents" in session.calls[1].text
    assert continuation_starts == ["reconciling"]
    assert execution.automatic_continuations == 1
    assert execution.assessment.outcome is PromptOutcome.COMPLETED
    assert execution.assessment.incomplete_tool_calls == 0


def test_child_reconciliation_applies_session_terminal_evidence_once() -> None:
    runner = _runner()
    running = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="list-running",
                title="list_agents",
                kind="other",
                status="completed",
                collaboration_tool="list_agents",
                subagent_states=(
                    {"source_id": "reviewer", "status": "running"},
                ),
            )
        ],
    )

    class EnrichingSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__(running, running)
            self.enrichment_calls = 0

        def enrich_child_reconciliation_result(
            self,
            result: PromptResult,
            *,
            started_at: float,
            ended_at: float,
            logical_task_started_at: float,
            on_event: Callable[[ACPEvent], None] | None = None,
        ) -> PromptResult:
            assert started_at > 0
            assert ended_at >= started_at
            assert 0 < logical_task_started_at <= started_at
            assert on_event is None
            self.enrichment_calls += 1
            return PromptResult(
                stop_reason=result.stop_reason,
                text=result.text,
                tool_calls=[
                    *result.tool_calls,
                    ToolCallInfo(
                        id="rollout-list-agents",
                        title="list_agents",
                        kind="other",
                        status="completed",
                        collaboration_tool="list_agents",
                        subagent_states=(
                            {
                                "source_id": "reviewer",
                                "status": "completed",
                            },
                        ),
                    ),
                ],
            )

    session = EnrichingSession()
    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert len(session.calls) == 2
    assert session.enrichment_calls == 1
    assert execution.automatic_continuations == 1
    assert execution.assessment.outcome is PromptOutcome.COMPLETED
    assert execution.assessment.unresolved_child_tool_calls == 0




def test_unresolved_child_still_fails_closed_after_bounded_reconciliation() -> None:
    runner = _runner()

    def running_snapshot(tool_id: str) -> PromptResult:
        return PromptResult(
            stop_reason="end_turn",
            tool_calls=[
                ToolCallInfo(
                    id=tool_id,
                    title="list_agents",
                    kind="other",
                    status="completed",
                    subagent_states=(
                        {"source_id": "reviewer", "status": "running"},
                    ),
                )
            ],
        )

    session = _FakeSession(
        running_snapshot("list-before-reconciliation"),
        running_snapshot("list-after-reconciliation"),
        AssertionError("duplicate child reconciliation"),
    )

    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert len(session.calls) == 2
    assert execution.automatic_continuations == 1
    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE
    assert execution.assessment.incomplete_tool_calls == 1


def test_completed_goal_allows_child_state_reconciliation() -> None:
    runner = _runner()
    completed_goal = ACPGoalInfo(objective="review", status="completed")
    first_result = PromptResult(
        stop_reason="end_turn",
        goal=completed_goal,
        tool_calls=[
            ToolCallInfo(
                id="list-before-final-answer",
                title="list_agents",
                kind="other",
                status="completed",
                subagent_states=(
                    {"source_id": "reviewer", "status": "running"},
                ),
            )
        ],
    )
    reconciled_result = PromptResult(
        stop_reason="end_turn",
        goal=completed_goal,
        tool_calls=[
            ToolCallInfo(
                id="list-after-final-answer",
                title="list_agents",
                kind="other",
                status="completed",
                subagent_states=(
                    {"source_id": "reviewer", "status": "completed"},
                ),
            )
        ],
    )
    session = _FakeSession(first_result, reconciled_result)

    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert len(session.calls) == 2
    assert execution.automatic_continuations == 1
    assert execution.assessment.outcome is PromptOutcome.COMPLETED
    assert execution.assessment.incomplete_tool_calls == 0


def test_plan_then_child_reconciliation_each_get_one_bounded_turn() -> None:
    runner = _runner()
    plan_completed_with_running_child = PromptResult(
        stop_reason="end_turn",
        plan=PlanInfo(
            entries=[
                PlanEntryInfo(content="implementation", status="completed"),
            ]
        ),
        tool_calls=[
            ToolCallInfo(
                id="review-running",
                title="list_agents",
                kind="other",
                status="completed",
                subagent_states=(
                    {"source_id": "reviewer", "status": "running"},
                ),
            )
        ],
    )
    reviewer_completed = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="review-completed",
                title="list_agents",
                kind="other",
                status="completed",
                subagent_states=(
                    {"source_id": "reviewer", "status": "completed"},
                ),
            )
        ],
    )
    session = _FakeSession(
        _pending_result(),
        plan_completed_with_running_child,
        reviewer_completed,
    )
    clock = _FakeClock()

    with patch("src.acp.continuation._monotonic", clock):
        execution = runner(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
        )

    assert len(session.calls) == 3
    assert "自动续做指令" in session.calls[1].text
    assert "子代理状态对账指令" in session.calls[2].text
    assert [call.timeout for call in session.calls] == [60, 60, 63]
    assert execution.automatic_continuations == 2
    assert execution.assessment.outcome is PromptOutcome.COMPLETED








def test_structured_child_reconciliation_timeout_retires_without_third_turn() -> None:
    runner = _runner()
    child_running = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="review-running",
                title="list_agents",
                kind="other",
                status="completed",
                subagent_states=(
                    {"source_id": "reviewer", "status": "running"},
                ),
            )
        ],
    )
    session = _FakeSession(
        child_running,
        PromptResult(stop_reason="timeout", text="still reconciling"),
        _complete_result(),
    )
    retirements: list[tuple[object, float]] = []

    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
        retire_finalization_session=lambda active, budget: (
            retirements.append((active, budget))
        ),
    )

    assert len(session.calls) == 2
    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE
    assert execution.assessment.stop_reason == "timeout"
    assert execution.entered_finalization is False
    assert retirements and retirements[0][0] is session
    assert session._force_dead is True










@pytest.mark.parametrize("status", ["paused", "blocked"])
def test_provider_goal_gets_one_bounded_recovery(status: str) -> None:
    runner = _runner()
    result = _pending_result()
    result.goal = ACPGoalInfo(objective="finish", status=status)
    session = _FakeSession(result, _complete_result())

    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert len(session.calls) == 2
    assert "自动恢复指令" in session.calls[1].text
    assert execution.automatic_continuations == 1
    assert execution.assessment.outcome is PromptOutcome.COMPLETED
    assert execution.entered_finalization is False






def test_running_child_snapshot_is_replaced_by_authoritative_terminal_update() -> None:
    from src.acp.continuation import _merge_tool_calls

    previous = ToolCallInfo(
        id="wait-1",
        title="wait_agent",
        kind="other",
        status="completed",
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
            {"source_id": "child-b", "status": "pending"},
        ),
    )
    current = ToolCallInfo(
        id="wait-1",
        title="wait_agent",
        kind="other",
        status="completed",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )

    merged = _merge_tool_calls([previous], [current])

    assert merged[0].subagent_states == (
        {"source_id": "child-a", "status": "completed"},
        {"source_id": "child-b", "status": "pending"},
    )




def test_terminal_child_snapshot_is_sticky_with_same_outer_call_id() -> None:
    from src.acp.continuation import _merge_tool_calls

    previous = ToolCallInfo(
        id="list-1",
        title="list_agents",
        kind="other",
        status="completed",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    current = ToolCallInfo(
        id="list-1",
        title="list_agents",
        kind="other",
        status="completed",
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )

    merged = _merge_tool_calls([previous], [current])

    assert merged[0].subagent_states == previous.subagent_states


def test_same_outer_id_explicit_followup_starts_new_child_generation() -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    previous = ToolCallInfo(
        id="provider-reused-id",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    current = ToolCallInfo(
        id="provider-reused-id",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )

    merged = _merge_tool_calls([previous], [current])
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert assessment.outcome is PromptOutcome.INCOMPLETE
    assert assessment.unresolved_child_tool_calls == 1
















def test_stale_same_followup_update_does_not_reopen_newer_terminal() -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    followup_running = ToolCallInfo(
        id="follow-x",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )
    list_terminal = ToolCallInfo(
        id="list-y",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )

    merged = _merge_tool_calls(
        [],
        [followup_running, list_terminal, followup_running],
    )
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert [tool.id for tool in merged] == ["follow-x", "list-y"]
    assert assessment.outcome is PromptOutcome.COMPLETED














def test_timeout_exception_does_not_start_ordinary_continuation() -> None:
    runner = _runner()
    session = _FakeSession(TimeoutError("primary deadline"))

    with pytest.raises(TimeoutError, match="primary deadline"):
        runner(
            session,
            "original task",
            timeout_s=60,
            finalization_reserve_s=0,
        )

    assert len(session.calls) == 1


def test_provider_cancelled_recovers_once_on_same_session() -> None:
    runner = _runner()
    session = _FakeSession(
        PromptResult(stop_reason="cancelled", text="interrupted"),
        _complete_result(),
    )

    execution = runner(
        session,
        "original task",
        timeout_s=60,
        finalization_reserve_s=0,
    )

    assert execution.assessment.outcome is PromptOutcome.COMPLETED
    assert execution.automatic_continuations == 1
    assert session.continuation_calls == session.calls[1:]
    assert "本指令不新增任何权限" in session.calls[1].text
    assert "sandbox" in session.calls[1].text


def test_repeated_provider_cancelled_stops_incomplete_after_one_recovery() -> None:
    runner = _runner()
    session = _FakeSession(
        PromptResult(stop_reason="cancelled", text="first interruption"),
        PromptResult(stop_reason="cancelled", text="second interruption"),
    )

    execution = runner(
        session,
        "original task",
        timeout_s=60,
        finalization_reserve_s=0,
    )

    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE
    assert execution.automatic_continuations == 1
    assert len(session.calls) == 2


def test_permission_denied_cancelled_gets_one_safe_recovery() -> None:
    runner = _runner()
    denied = PromptResult(
        stop_reason="cancelled",
        tool_results=[
            {
                "kind": "permission",
                "data": {
                    "outcome": "cancelled",
                    "reason": "dangerous_execute",
                },
            }
        ],
    )
    session = _FakeSession(denied, _complete_result())

    execution = runner(
        session,
        "original task",
        timeout_s=60,
        finalization_reserve_s=0,
    )

    assert execution.assessment.outcome is PromptOutcome.COMPLETED
    assert execution.automatic_continuations == 1


def test_manager_marked_user_cancel_never_continues() -> None:
    runner = _runner()
    session = _FakeSession(
        PromptResult(
            stop_reason="cancelled",
            cancellation_source="user",
        )
    )

    execution = runner(
        session,
        "original task",
        timeout_s=60,
        finalization_reserve_s=0,
    )

    assert execution.assessment.outcome is PromptOutcome.CANCELLED
    assert execution.automatic_continuations == 0
    assert len(session.calls) == 1


def test_continuation_uses_only_the_original_deadline_remaining_budget() -> None:
    runner = _runner()
    clock = _FakeClock()
    session = _FakeSession(
        _pending_result(),
        _complete_result(),
        on_send=lambda call_number: (
            clock.advance(10.0) if call_number == 1 else None
        ),
    )
    continuation_starts: list[str] = []

    def close_first_turn() -> None:
        continuation_starts.append("continuing")
        clock.advance(10.0)

    with patch("src.acp.continuation._monotonic", clock):
        execution = runner(
            session,
            "original task",
            timeout_s=120,
            finalization_reserve_s=30,
            on_continuation_start=close_first_turn,
        )

    assert execution.assessment.outcome is PromptOutcome.COMPLETED
    assert continuation_starts == ["continuing"]
    assert [call.timeout for call in session.calls] == [90, 70]












def test_continuation_prompt_uses_safe_defaults_without_adding_authority() -> None:
    runner = _runner()
    session = _FakeSession(
        _pending_result(pending_count=2),
        _complete_result(),
    )
    clock = _FakeClock()

    with patch("src.acp.continuation._monotonic", clock):
        runner(
            session,
            "original task explicitly authorizes commit and push",
            timeout_s=90,
            finalization_reserve_s=30,
        )

    continuation_prompt = session.calls[1].text
    assert "结构化计划仍有 2 项未完成" in continuation_prompt
    assert "继续完成所有其他已获授权且在原任务范围内的工作" in continuation_prompt
    assert "文档明确推荐的选项" in continuation_prompt
    assert "最小安全默认值" in continuation_prompt
    assert "记录该选择" in continuation_prompt
    assert "本指令不新增任何权限" in continuation_prompt
    assert "保留原始用户请求已经明确授予的精确权限" in continuation_prompt
    assert "不得新推断" in continuation_prompt
    assert "凭据" in continuation_prompt
    assert "部署或发布" in continuation_prompt
    assert "删除" in continuation_prompt
    assert "不可逆外部副作用" in continuation_prompt
    assert "ACP、sandbox 或工具权限" in continuation_prompt
    assert "不得绕过" in continuation_prompt
