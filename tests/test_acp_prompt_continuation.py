"""Regression coverage for bounded ordinary ACP prompt continuation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
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


def _complete_result() -> PromptResult:
    return PromptResult(
        stop_reason="end_turn",
        text="done",
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
    assert execution.awaiting_user_input is False
    assert len(session.calls) == 2
    assert [call.timeout for call in session.calls] == [60, 60]
    assert session.calls[0].text == "original task"
    assert "自动续做指令" in session.calls[1].text
    assert all(call.on_event is on_event for call in session.calls)
    assert continuation_starts == ["continuing"]
    assert order == ["send:1", "boundary", "send:2"]


def test_second_pending_plan_stops_after_one_continuation() -> None:
    runner = _runner()
    session = _FakeSession(_pending_result(), _pending_result())
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

    assert len(session.calls) == 2
    assert continuation_starts == ["continuing"]
    assert session.continuation_calls == [session.calls[1]]
    assert execution.automatic_continuations == 1
    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE
    assert execution.assessment.pending_plan_entries == 1
    assert execution.awaiting_user_input is True


def test_missing_second_turn_plan_keeps_prior_plan_pending() -> None:
    runner = _runner()
    first_result = _pending_result()
    prior_tool = ToolCallInfo(
        id="shared-tool",
        title="pytest",
        kind="execute",
        status="completed",
    )
    first_result.tool_calls = [prior_tool]
    first_result.tool_results = [{"kind": "read_file", "data": {"path": "a.py"}}]
    first_result.modified_files = {"src/first.py"}
    first_result.output_tokens = 7
    text_only_result = PromptResult(
        stop_reason="end_turn",
        text="continued without a plan update",
        tool_calls=[
            ToolCallInfo(
                id="shared-tool",
                title="pytest",
                kind="execute",
                status="completed",
            ),
            ToolCallInfo(
                id="second-tool",
                title="ruff",
                kind="execute",
                status="completed",
            ),
        ],
        tool_results=[{"kind": "write_file", "data": {"path": "b.py"}}],
        modified_files={"src/second.py"},
        output_tokens=11,
    )
    first_tool_calls = list(first_result.tool_calls)
    first_tool_results = list(first_result.tool_results)
    first_modified_files = set(first_result.modified_files)
    second_tool_calls = list(text_only_result.tool_calls)
    second_tool_results = list(text_only_result.tool_results)
    second_modified_files = set(text_only_result.modified_files)
    session = _FakeSession(first_result, text_only_result)
    clock = _FakeClock()

    with patch("src.acp.continuation._monotonic", clock):
        execution = runner(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
        )

    assert len(session.calls) == 2
    assert text_only_result.plan is None
    assert execution.result is not text_only_result
    assert execution.result.plan is first_result.plan
    assert execution.result.text == text_only_result.text
    assert execution.result.stop_reason == text_only_result.stop_reason
    assert [tool.id for tool in execution.result.tool_calls] == [
        "shared-tool",
        "second-tool",
    ]
    assert execution.result.tool_calls[0] is text_only_result.tool_calls[0]
    assert execution.result.tool_results == [
        *first_tool_results,
        *second_tool_results,
    ]
    assert execution.result.modified_files == {
        "src/first.py",
        "src/second.py",
    }
    assert execution.result.output_tokens == 18
    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE
    assert execution.assessment.pending_plan_entries == 1
    assert execution.awaiting_user_input is True
    assert first_result.tool_calls == first_tool_calls
    assert first_result.tool_calls[0] is prior_tool
    assert first_result.tool_results == first_tool_results
    assert first_result.modified_files == first_modified_files
    assert first_result.output_tokens == 7
    assert text_only_result.tool_calls == second_tool_calls
    assert text_only_result.tool_results == second_tool_results
    assert text_only_result.modified_files == second_modified_files
    assert text_only_result.output_tokens == 11


def test_pending_plan_with_active_tool_is_not_continued() -> None:
    runner = _runner()
    session = _FakeSession(_pending_result(active_tool=True))
    continuation_starts: list[str] = []

    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
        on_continuation_start=lambda: continuation_starts.append("continuing"),
    )

    assert len(session.calls) == 1
    assert continuation_starts == []
    assert execution.automatic_continuations == 0
    assert execution.assessment.pending_plan_entries == 1
    assert execution.assessment.incomplete_tool_calls == 1
    assert execution.awaiting_user_input is False


def test_explicit_user_confirmation_preempts_child_reconciliation() -> None:
    runner = _runner()
    session = _FakeSession(
        PromptResult(
            stop_reason="end_turn",
            text=(
                "检测到其他进程正在修改核心文件。请确认一种处理方式：\n"
                "1. 由我接管\n2. 等其他进程完成"
            ),
            tool_calls=[
                ToolCallInfo(
                    id="list-before-confirmation",
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
    )

    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert len(session.calls) == 1
    assert execution.automatic_continuations == 0
    assert execution.awaiting_user_input is True
    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE


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
    assert execution.awaiting_user_input is False


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
            on_event: Callable[[ACPEvent], None] | None = None,
        ) -> PromptResult:
            assert started_at > 0
            assert ended_at >= started_at
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


def test_unresolved_child_does_not_reconcile_again_after_stale_running_snapshot() -> None:
    runner = _runner()

    def snapshot(tool_id: str, status: str) -> PromptResult:
        return PromptResult(
            stop_reason="end_turn",
            tool_calls=[
                ToolCallInfo(
                    id=tool_id,
                    title="list_agents",
                    kind="other",
                    status="completed",
                    subagent_states=(
                        {"source_id": "reviewer", "status": status},
                    ),
                )
            ],
        )

    session = _FakeSession(
        snapshot("list-before-reconciliation", "running"),
        snapshot("list-still-stale", "running"),
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
    assert execution.assessment.unresolved_child_tool_calls == 1


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
    assert execution.awaiting_user_input is False


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
    assert execution.awaiting_user_input is False


def test_child_reconciliation_then_plan_each_get_one_bounded_turn() -> None:
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
    child_completed_with_plan = _pending_result()
    child_completed_with_plan.tool_calls = [
        ToolCallInfo(
            id="review-completed",
            title="list_agents",
            kind="other",
            status="completed",
            subagent_states=(
                {"source_id": "reviewer", "status": "completed"},
            ),
        )
    ]
    session = _FakeSession(
        child_running,
        child_completed_with_plan,
        _complete_result(),
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
    assert "子代理状态对账指令" in session.calls[1].text
    assert "自动续做指令" in session.calls[2].text
    assert [call.timeout for call in session.calls] == [60, 63, 60]
    assert execution.automatic_continuations == 2
    assert execution.assessment.outcome is PromptOutcome.COMPLETED
    assert execution.awaiting_user_input is False


def test_unresolved_child_overrides_paused_goal_after_reconciliation() -> None:
    runner = _runner()

    def running_snapshot(*, goal: ACPGoalInfo | None = None) -> PromptResult:
        return PromptResult(
            stop_reason="end_turn",
            goal=goal,
            tool_calls=[
                ToolCallInfo(
                    id="list-agents",
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
        running_snapshot(),
        running_snapshot(
            goal=ACPGoalInfo(objective="review", status="paused")
        ),
    )

    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert len(session.calls) == 2
    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE
    assert execution.assessment.unresolved_child_tool_calls == 1
    assert execution.awaiting_user_input is False


def test_child_reconciliation_timeout_does_not_start_finalization_or_replace_session() -> None:
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
        TimeoutError("reconciliation deadline"),
        _complete_result(),
    )
    finalization_starts: list[str] = []
    replacements: list[float] = []
    retirements: list[tuple[object, float]] = []

    with pytest.raises(TimeoutError, match="reconciliation deadline"):
        runner(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            on_finalization_start=lambda: finalization_starts.append(
                "finalizing"
            ),
            replace_dead_session=lambda budget: (
                replacements.append(budget) or session
            ),
            retire_finalization_session=lambda active, budget: (
                retirements.append((active, budget))
            ),
        )

    assert len(session.calls) == 2
    assert finalization_starts == []
    assert replacements == []
    assert retirements and retirements[0][0] is session
    assert session._force_dead is True


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


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_non_natural_stop_after_reconciliation_never_waits_on_paused_goal(
    stop_reason: str,
) -> None:
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
    non_natural = PromptResult(
        stop_reason=stop_reason,
        goal=ACPGoalInfo(objective="review", status="paused"),
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
    session = _FakeSession(child_running, non_natural)

    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert len(session.calls) == 2
    assert execution.assessment.stop_reason == stop_reason
    assert execution.awaiting_user_input is False


def test_active_outer_tool_does_not_trigger_child_reconciliation() -> None:
    runner = _runner()
    session = _FakeSession(
        PromptResult(
            stop_reason="end_turn",
            tool_calls=[
                ToolCallInfo(
                    id="outer-still-running",
                    title="exec",
                    kind="execute",
                    status="in_progress",
                )
            ],
        )
    )

    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert len(session.calls) == 1
    assert execution.automatic_continuations == 0
    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE


@pytest.mark.parametrize(
    "stop_reason",
    ["cancelled", "refusal", "max_tokens", "max_turn_requests", "timeout"],
)
def test_non_natural_stop_with_pending_plan_is_not_continued(
    stop_reason: str,
) -> None:
    runner = _runner()
    session = _FakeSession(_pending_result(stop_reason=stop_reason))

    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=0,
    )

    assert len(session.calls) == 1
    assert execution.automatic_continuations == 0
    assert execution.awaiting_user_input is False


def test_timeout_finalization_result_is_not_ordinary_continuation() -> None:
    runner = _runner()
    session = _FakeSession(
        PromptResult(stop_reason="timeout", text="partial"),
        _pending_result(),
    )
    transitions: list[str] = []
    clock = _FakeClock()

    with patch("src.acp.continuation._monotonic", clock):
        execution = runner(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            on_finalization_start=lambda: transitions.append("finalizing"),
        )

    assert len(session.calls) == 2
    assert [call.timeout for call in session.calls] == [60, 30]
    assert "运行时收尾指令" in session.calls[1].text
    assert "自动续做指令" not in session.calls[1].text
    assert transitions == ["finalizing"]
    assert execution.automatic_continuations == 0
    assert execution.awaiting_user_input is False


@pytest.mark.parametrize(
    ("status", "awaiting_user_input"),
    [
        ("active", False),
        ("paused", True),
        ("blocked", True),
        ("completed", False),
        ("future-provider-state", False),
    ],
)
def test_provider_goal_disables_ordinary_continuation(
    status: str,
    awaiting_user_input: bool,
) -> None:
    runner = _runner()
    result = _pending_result()
    result.goal = ACPGoalInfo(objective="finish", status=status)
    session = _FakeSession(result)

    with patch(
        "src.acp.continuation._build_continuation_prompt"
    ) as build_continuation:
        execution = runner(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
        )

    assert len(session.calls) == 1
    build_continuation.assert_not_called()
    assert execution.automatic_continuations == 0
    assert execution.awaiting_user_input is awaiting_user_input
    assert execution.entered_finalization is False


@pytest.mark.parametrize("status", ["paused", "blocked"])
def test_provider_goal_after_finalization_does_not_await_retired_session(
    status: str,
) -> None:
    runner = _runner()
    final_result = _pending_result()
    final_result.goal = ACPGoalInfo(objective="finish", status=status)
    session = _FakeSession(
        PromptResult(stop_reason="timeout", text="partial"),
        final_result,
    )

    with patch(
        "src.acp.continuation._build_continuation_prompt"
    ) as build_continuation:
        execution = runner(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
        )

    assert len(session.calls) == 2
    build_continuation.assert_not_called()
    assert execution.awaiting_user_input is False
    assert execution.entered_finalization is True


def test_running_child_snapshot_survives_outer_update_without_children() -> None:
    from src.acp.continuation import _merge_tool_calls

    previous = ToolCallInfo(
        id="wait-1",
        title="wait_agent",
        kind="other",
        status="completed",
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )
    current = ToolCallInfo(
        id="wait-1",
        title="wait_agent",
        kind="other",
        status="completed",
    )

    merged = _merge_tool_calls([previous], [current])

    assert merged[0].subagent_states == previous.subagent_states


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


def test_running_child_snapshot_retains_unknown_update_fail_closed() -> None:
    from src.acp.continuation import _merge_tool_calls

    previous_child = {
        "source_id": "child-a",
        "status": "running",
        "message": "still working",
    }
    previous = ToolCallInfo(
        id="wait-1",
        title="wait_agent",
        kind="other",
        status="completed",
        subagent_states=(previous_child,),
    )
    current = ToolCallInfo(
        id="wait-1",
        title="wait_agent",
        kind="other",
        status="completed",
        subagent_states=(
            {"source_id": "child-a", "status": "future-state"},
        ),
    )

    merged = _merge_tool_calls([previous], [current])

    assert merged[0].subagent_states == (
        previous_child,
        {"source_id": "child-a", "status": "future-state"},
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


def test_same_followup_terminal_snapshot_is_sticky_with_same_outer_id() -> None:
    from src.acp.collaboration import merge_tool_call_snapshot
    from src.acp.outcome import classify_prompt_result

    previous = ToolCallInfo(
        id="followup-1",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    current = ToolCallInfo(
        id="followup-1",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )

    merged = [merge_tool_call_snapshot(previous, current)]
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert merged[0].subagent_states == previous.subagent_states
    assert assessment.outcome is PromptOutcome.COMPLETED


def test_same_followup_id_reused_across_turns_starts_new_generation() -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    previous = ToolCallInfo(
        id="provider-reused-followup",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    current = ToolCallInfo(
        id="provider-reused-followup",
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

    assert merged[-1].subagent_states == current.subagent_states
    assert assessment.outcome is PromptOutcome.INCOMPLETE
    assert assessment.unresolved_child_tool_calls == 1


@pytest.mark.parametrize(
    "malformed_previous",
    [
        ToolCallInfo(
            id="same-call",
            title="list_agents",
            kind="other",
            status="completed",
            collaboration_tool="list_agents",
            collaboration_receivers=("child-a", 123),  # type: ignore[arg-type]
            subagent_states=(
                {"source_id": "child-a", "status": "running"},
            ),
        ),
        ToolCallInfo(
            id="same-call",
            title="Start child",
            kind="other",
            status="completed",
            subagent_source_id=123,  # type: ignore[arg-type]
            subagent_activity="started",
        ),
        ToolCallInfo(
            id="same-call",
            title="Start child",
            kind="other",
            status="completed",
            subagent_source_id="child-a",
            subagent_activity="future-activity",
        ),
        ToolCallInfo(
            id="same-call",
            title="list_agents",
            kind="other",
            status="completed",
            collaboration_receivers=0,  # type: ignore[arg-type]
        ),
        ToolCallInfo(
            id="same-call",
            title="Child activity",
            kind="other",
            status="completed",
            subagent_source_id=0,  # type: ignore[arg-type]
        ),
        ToolCallInfo(
            id="same-call",
            title="collaboration",
            kind="other",
            status="completed",
            collaboration_tool=0,  # type: ignore[arg-type]
        ),
        ToolCallInfo(
            id="same-call",
            title="list_agents",
            kind="other",
            status="completed",
            subagent_states=0,  # type: ignore[arg-type]
        ),
        ToolCallInfo(
            id="same-call",
            title="list_agents",
            kind="other",
            status="completed",
            subagent_states={},  # type: ignore[arg-type]
        ),
        ToolCallInfo(
            id="same-call",
            title="list_agents",
            kind="other",
            status="completed",
            child_metadata_malformed=None,  # type: ignore[arg-type]
        ),
    ],
)
def test_same_id_merge_preserves_prior_malformed_child_metadata(
    malformed_previous: ToolCallInfo,
) -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    current = ToolCallInfo(
        id="same-call",
        title="clean terminal update",
        kind="other",
        status="completed",
    )

    merged = _merge_tool_calls([malformed_previous], [current])
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert assessment.outcome is PromptOutcome.INCOMPLETE
    assert assessment.unresolved_child_tool_calls == 1


@pytest.mark.parametrize(
    "current",
    [
        ToolCallInfo(
            id="same-call",
            title="future_restart_agent",
            kind="other",
            status="completed",
            collaboration_tool="future_restart_agent",
            collaboration_receivers=("child-a",),
            subagent_states=(
                {"source_id": "child-a", "status": "running"},
            ),
        ),
        ToolCallInfo(
            id="same-call",
            title="Start child",
            kind="other",
            status="completed",
            subagent_source_id="child-a",
            subagent_activity="started",
        ),
        ToolCallInfo(
            id="same-call",
            title="Child interaction",
            kind="other",
            status="completed",
            subagent_source_id="child-a",
            subagent_activity="interacted",
        ),
        ToolCallInfo(
            id="same-call",
            title="Failed child interruption",
            kind="other",
            status="failed",
            subagent_source_id="child-a",
            subagent_activity="interrupted",
        ),
    ],
)
def test_same_id_ambiguous_restart_cannot_reuse_prior_terminal(
    current: ToolCallInfo,
) -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    previous = ToolCallInfo(
        id="same-call",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )

    merged = _merge_tool_calls([previous], [current])
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert assessment.outcome is PromptOutcome.INCOMPLETE
    assert assessment.unresolved_child_tool_calls == 1


def test_merge_preserves_anonymous_terminal_before_named_followup() -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    terminal = ToolCallInfo(
        id="",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    followup = ToolCallInfo(
        id="followup-1",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )

    merged = _merge_tool_calls([terminal], [followup])
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert [tool.title for tool in merged] == ["list_agents", "followup_task"]
    assert assessment.outcome is PromptOutcome.INCOMPLETE
    assert assessment.unresolved_child_tool_calls == 1


def test_merge_orders_reused_named_call_by_its_latest_snapshot() -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    reused_terminal = ToolCallInfo(
        id="reused",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    later_terminal = ToolCallInfo(
        id="later-old-terminal",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    reused_followup = ToolCallInfo(
        id="reused",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )

    merged = _merge_tool_calls(
        [reused_terminal, later_terminal],
        [reused_followup],
    )
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert [tool.id for tool in merged] == [
        "reused",
        "later-old-terminal",
        "reused",
    ]
    assert assessment.outcome is PromptOutcome.INCOMPLETE
    assert assessment.unresolved_child_tool_calls == 1


def test_stale_same_action_update_keeps_original_lifecycle_position() -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    list_terminal = ToolCallInfo(
        id="list-x",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    followup_running = ToolCallInfo(
        id="follow-y",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )

    merged = _merge_tool_calls(
        [],
        [list_terminal, followup_running, list_terminal],
    )
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert [tool.id for tool in merged] == ["list-x", "follow-y"]
    assert assessment.outcome is PromptOutcome.INCOMPLETE


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


def test_cross_turn_reused_passive_call_moves_later_terminal_snapshot() -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    reused_terminal = ToolCallInfo(
        id="reused-list",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    followup_running = ToolCallInfo(
        id="follow-running",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )

    merged = _merge_tool_calls(
        [reused_terminal, followup_running],
        [reused_terminal],
    )
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert [tool.id for tool in merged] == [
        "reused-list",
        "follow-running",
        "reused-list",
    ]
    assert assessment.outcome is PromptOutcome.COMPLETED


def test_cross_turn_reused_passive_without_state_does_not_inherit_terminal() -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    reused_terminal = ToolCallInfo(
        id="reused-list",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    followup_running = ToolCallInfo(
        id="follow-running",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )
    empty_refresh = ToolCallInfo(
        id="reused-list",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
    )

    merged = _merge_tool_calls(
        [reused_terminal, followup_running],
        [empty_refresh],
    )
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert len(merged) == 3
    assert merged[-1].subagent_states == ()
    assert assessment.outcome is PromptOutcome.INCOMPLETE


def test_cross_turn_unrelated_id_collision_does_not_inherit_child_state() -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    reused_terminal = ToolCallInfo(
        id="reused-id",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    followup_running = ToolCallInfo(
        id="follow-running",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )
    unrelated_current = ToolCallInfo(
        id="reused-id",
        title="read",
        kind="read",
        status="completed",
    )

    merged = _merge_tool_calls(
        [reused_terminal, followup_running],
        [unrelated_current],
    )
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert [tool.title for tool in merged] == [
        "list_agents",
        "followup_task",
        "read",
    ]
    assert merged[-1].subagent_states == ()
    assert assessment.outcome is PromptOutcome.INCOMPLETE


@pytest.mark.parametrize("new_action", ["wait_agent", "followup_task"])
def test_same_turn_changed_action_without_state_does_not_inherit_terminal(
    new_action: str,
) -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    reused_terminal = ToolCallInfo(
        id="reused-action",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    followup_running = ToolCallInfo(
        id="follow-running",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )
    changed_action = ToolCallInfo(
        id="reused-action",
        title=new_action,
        kind="other",
        status="completed",
        collaboration_tool=new_action,
    )

    merged = _merge_tool_calls(
        [],
        [reused_terminal, followup_running, changed_action],
    )
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert len(merged) == 3
    assert merged[-1].subagent_states == ()
    assert assessment.outcome is PromptOutcome.INCOMPLETE


def test_same_turn_passive_metadata_enrichment_keeps_original_position() -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    unclassified_terminal = ToolCallInfo(
        id="enriched-list",
        title="child snapshot",
        kind="other",
        status="completed",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    enriched_terminal = ToolCallInfo(
        id="enriched-list",
        title="list_agents",
        kind="other",
        status="completed",
        collaboration_tool="list_agents",
        subagent_states=(
            {"source_id": "child-a", "status": "completed"},
        ),
    )
    followup_running = ToolCallInfo(
        id="follow-running",
        title="followup_task",
        kind="other",
        status="completed",
        collaboration_tool="followup_task",
        collaboration_receivers=("child-a",),
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )

    merged = _merge_tool_calls(
        [],
        [unclassified_terminal, followup_running, enriched_terminal],
    )
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert [tool.id for tool in merged] == ["enriched-list", "follow-running"]
    assert assessment.outcome is PromptOutcome.INCOMPLETE


def test_malformed_child_container_is_not_reinterpreted_during_merge() -> None:
    from src.acp.continuation import _merge_tool_calls
    from src.acp.outcome import classify_prompt_result

    previous = ToolCallInfo(
        id="list-1",
        title="list_agents",
        kind="other",
        status="completed",
        subagent_states=(
            {"source_id": "child-a", "status": "running"},
        ),
    )
    current = ToolCallInfo(
        id="list-1",
        title="list_agents",
        kind="other",
        status="completed",
        subagent_states={  # type: ignore[arg-type]
            "source_id": "child-a",
            "status": "completed",
        },
    )

    merged = _merge_tool_calls([previous], [current])
    assessment = classify_prompt_result(
        PromptResult(stop_reason="end_turn", tool_calls=merged)
    )

    assert assessment.outcome is PromptOutcome.INCOMPLETE
    assert assessment.unresolved_child_tool_calls == 1


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


def test_first_turn_uses_budget_remaining_from_outer_deadline() -> None:
    runner = _runner()
    clock = _FakeClock()
    session = _FakeSession(_complete_result())
    real_float = float

    def normalize_timeout(value: float | int) -> float:
        clock.advance(10.0)
        return real_float(value)

    with (
        patch("src.acp.continuation._monotonic", clock),
        patch(
            "src.acp.continuation.float",
            side_effect=normalize_timeout,
            create=True,
        ),
    ):
        execution = runner(
            session,
            "original task",
            timeout_s=120,
            finalization_reserve_s=30,
        )

    assert execution.assessment.outcome is PromptOutcome.COMPLETED
    assert [call.timeout for call in session.calls] == [80]


def test_boundary_hook_exhausting_budget_does_not_send_continuation() -> None:
    runner = _runner()
    clock = _FakeClock()
    first_result = _pending_result(pending_count=2)
    session = _FakeSession(first_result)
    boundary_calls = 0

    def exhaust_budget_at_boundary() -> None:
        nonlocal boundary_calls
        boundary_calls += 1
        clock.advance(90.0)

    with patch("src.acp.continuation._monotonic", clock):
        execution = runner(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            on_continuation_start=exhaust_budget_at_boundary,
        )

    assert boundary_calls == 1
    assert len(session.calls) == 1
    assert execution.result is first_result
    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE
    assert execution.assessment.pending_plan_entries == 2
    assert execution.automatic_continuations == 0
    assert execution.awaiting_user_input is False


def test_continuation_callback_failure_does_not_lose_execution() -> None:
    runner = _runner()
    session = _FakeSession(_pending_result(), _complete_result())
    callback_calls = 0

    def fail_callback() -> None:
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("card turn transition failed")

    clock = _FakeClock()
    with patch("src.acp.continuation._monotonic", clock):
        execution = runner(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
            on_continuation_start=fail_callback,
        )

    assert callback_calls == 1
    assert len(session.calls) == 2
    assert execution.assessment.outcome is PromptOutcome.COMPLETED


@pytest.mark.parametrize("failing_send", [1, 2])
def test_generic_send_exception_propagates(failing_send: int) -> None:
    runner = _runner()
    error = RuntimeError(f"send {failing_send} failed")
    results: tuple[PromptResult | BaseException, ...]
    if failing_send == 1:
        results = (error,)
    else:
        results = (_pending_result(), error)
    session = _FakeSession(*results)
    clock = _FakeClock()

    with (
        patch("src.acp.continuation._monotonic", clock),
        pytest.raises(RuntimeError, match=rf"send {failing_send} failed"),
    ):
        runner(
            session,
            "original task",
            timeout_s=90,
            finalization_reserve_s=30,
        )

    assert len(session.calls) == failing_send


@pytest.mark.parametrize("timeout_s", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_validation_is_preserved(timeout_s: float) -> None:
    runner = _runner()
    session = _FakeSession(_complete_result())
    clock = _FakeClock()

    with (
        patch("src.acp.continuation._monotonic", clock),
        pytest.raises(
            ValueError,
            match="prompt timeout must be a finite positive number",
        ),
    ):
        runner(
            session,
            "original task",
            timeout_s=timeout_s,
            finalization_reserve_s=30,
        )

    assert session.calls == []


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


def test_continuation_result_is_immutable_and_exported() -> None:
    runner = _runner()
    result_type = getattr(acp, "PromptContinuationResult", None)
    assert result_type is not None
    session = _FakeSession(_complete_result())

    execution = runner(
        session,
        "original task",
        timeout_s=90,
        finalization_reserve_s=30,
    )

    assert isinstance(execution, result_type)
    with pytest.raises(FrozenInstanceError):
        execution.awaiting_user_input = True
