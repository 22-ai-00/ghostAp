"""Task-level execution window rollover contracts."""

from __future__ import annotations

from collections.abc import Callable

from src.acp.continuation import PromptContinuationResult
from src.acp.execution_windows import run_prompt_across_execution_windows
from src.acp.models import PlanEntryInfo, PlanInfo, PromptResult
from src.acp.outcome import PromptOutcome, classify_prompt_result


class _Session:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


def _execution(
    *,
    completed: bool,
    finalized: bool,
    text: str,
    modified_file: str,
) -> PromptContinuationResult:
    status = "completed" if completed else "in_progress"
    result = PromptResult(
        stop_reason="end_turn",
        text=text,
        plan=PlanInfo(
            entries=[PlanEntryInfo(content="finish task", status=status)]
        ),
        modified_files={modified_file},
    )
    return PromptContinuationResult(
        result=result,
        assessment=classify_prompt_result(result),
        automatic_continuations=0,
        entered_finalization=finalized,
    )


def _queued_executor(
    *executions: PromptContinuationResult,
) -> tuple[
    Callable[[_Session, str], PromptContinuationResult],
    list[tuple[str, str]],
]:
    pending = list(executions)
    calls: list[tuple[str, str]] = []

    def execute(session: _Session, prompt: str) -> PromptContinuationResult:
        calls.append((session.session_id, prompt))
        return pending.pop(0)

    return execute, calls


def test_timeout_finalized_incomplete_result_opens_fresh_window() -> None:
    execute, calls = _queued_executor(
        _execution(
            completed=False,
            finalized=True,
            text="first checkpoint",
            modified_file="first.py",
        ),
        _execution(
            completed=True,
            finalized=False,
            text="done",
            modified_file="second.py",
        ),
    )
    initial = _Session("provider-session-1")
    resumed = _Session("provider-session-1")
    resume_calls: list[tuple[_Session, str]] = []

    def resume(old: _Session, session_id: str) -> _Session:
        resume_calls.append((old, session_id))
        return resumed

    execution = run_prompt_across_execution_windows(
        initial,
        "original task",
        max_windows=4,
        execute_window=execute,
        resume_window=resume,
    )

    assert execution.assessment.outcome is PromptOutcome.COMPLETED
    assert execution.execution_windows == 2
    assert execution.window_limit_reached is False
    assert execution.result.modified_files == {"first.py", "second.py"}
    assert resume_calls == [(initial, "provider-session-1")]
    assert calls[0] == ("provider-session-1", "original task")
    assert calls[1][0] == "provider-session-1"
    assert "第 2/4 个执行窗口" in calls[1][1]
    assert "original task" in calls[1][1]
    assert "GhostAP 不增加二次权限或风险判断" in calls[1][1]


def test_natural_incomplete_result_does_not_open_new_window() -> None:
    execute, _calls = _queued_executor(
        _execution(
            completed=False,
            finalized=False,
            text="blocked without deadline",
            modified_file="first.py",
        )
    )
    resumes: list[str] = []

    execution = run_prompt_across_execution_windows(
        _Session("provider-session-1"),
        "original task",
        max_windows=4,
        execute_window=execute,
        resume_window=lambda _old, session_id: (
            resumes.append(session_id) or _Session(session_id)
        ),
    )

    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE
    assert execution.execution_windows == 1
    assert execution.window_limit_reached is False
    assert resumes == []


def test_cancelled_result_does_not_open_new_window() -> None:
    cancelled = PromptResult(
        stop_reason="cancelled",
        text="stopped",
        cancellation_source="user",
    )
    execute, _calls = _queued_executor(
        PromptContinuationResult(
            result=cancelled,
            assessment=classify_prompt_result(cancelled),
            automatic_continuations=0,
            entered_finalization=True,
        )
    )
    resumes: list[str] = []

    execution = run_prompt_across_execution_windows(
        _Session("provider-session-1"),
        "original task",
        max_windows=4,
        execute_window=execute,
        resume_window=lambda _old, session_id: (
            resumes.append(session_id) or _Session(session_id)
        ),
    )

    assert execution.assessment.outcome is PromptOutcome.CANCELLED
    assert execution.execution_windows == 1
    assert resumes == []


def test_execution_window_ceiling_is_explicit_terminal_evidence() -> None:
    execute, calls = _queued_executor(
        *[
            _execution(
                completed=False,
                finalized=True,
                text=f"checkpoint {index}",
                modified_file=f"file-{index}.py",
            )
            for index in range(1, 5)
        ]
    )
    rollover_boundaries: list[tuple[int, int]] = []

    execution = run_prompt_across_execution_windows(
        _Session("provider-session-1"),
        "original task",
        max_windows=4,
        execute_window=execute,
        resume_window=lambda _old, session_id: _Session(session_id),
        on_window_rollover=lambda current, total: rollover_boundaries.append(
            (current, total)
        ),
    )

    assert execution.assessment.outcome is PromptOutcome.INCOMPLETE
    assert execution.execution_windows == 4
    assert execution.window_limit_reached is True
    assert len(calls) == 4
    assert rollover_boundaries == [(2, 4), (3, 4), (4, 4)]


def test_invalid_execution_window_limit_is_rejected() -> None:
    execute, _calls = _queued_executor(
        _execution(
            completed=True,
            finalized=False,
            text="done",
            modified_file="done.py",
        )
    )

    try:
        run_prompt_across_execution_windows(
            _Session("provider-session-1"),
            "original task",
            max_windows=0,
            execute_window=execute,
            resume_window=lambda old, _session_id: old,
        )
    except ValueError as exc:
        assert "max_windows" in str(exc)
    else:  # pragma: no cover - the assertion above is the contract
        raise AssertionError("max_windows=0 must fail closed")


def test_rollover_scope_uses_raw_task_not_injected_initial_prompt() -> None:
    execute, calls = _queued_executor(
        _execution(
            completed=False,
            finalized=True,
            text="checkpoint",
            modified_file="first.py",
        ),
        _execution(
            completed=True,
            finalized=False,
            text="done",
            modified_file="second.py",
        ),
    )

    run_prompt_across_execution_windows(
        _Session("provider-session-1"),
        "BRIDGE CONTEXT\ninjected initial prompt",
        task_scope="raw user task",
        max_windows=2,
        execute_window=execute,
        resume_window=lambda _old, session_id: _Session(session_id),
    )

    assert calls[0][1] == "BRIDGE CONTEXT\ninjected initial prompt"
    assert "raw user task" in calls[1][1]
    assert "BRIDGE CONTEXT" not in calls[1][1]
