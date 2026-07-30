"""Regression coverage for bounded ordinary ACP prompt continuation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
from unittest.mock import patch

import pytest

import src.acp as acp
from src.acp.models import (
    ACPEvent,
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
