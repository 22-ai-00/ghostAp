"""Task-level rollover across bounded ACP execution windows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Protocol, TypeVar

from .continuation import PromptContinuationResult, merge_prompt_results
from .outcome import PromptOutcome, classify_prompt_result

_TASK_EXCERPT_CHARS = 12_000


class _WindowSession(Protocol):
    session_id: str


SessionT = TypeVar("SessionT", bound=_WindowSession)


def _task_excerpt(original_task: str) -> str:
    task = str(original_task or "").strip()
    if len(task) <= _TASK_EXCERPT_CHARS:
        return task
    return task[:_TASK_EXCERPT_CHARS] + "\n…（原任务已截断）"


def _build_window_continuation_prompt(
    original_task: str,
    *,
    window_index: int,
    max_windows: int,
    reason: str,
) -> str:
    return (
        "[GhostAP 跨执行窗口自动续做指令]\n"
        f"单个执行窗口的时间预算已用尽，现已进入第 {window_index}/{max_windows} "
        "个执行窗口。此前成果已安全收尾并保存在工作区；请从当前工作区、"
        "结构化计划和会话历史继续，不要重复已经完成的工作。\n"
        f"上一窗口未完成原因：{reason}\n"
        "继续采用推荐默认值；GhostAP 不增加二次权限或风险判断，provider 自身"
        "的权限交互由 provider 处理。\n"
        "完成后必须回读原始任务和结构化计划；仍有未完成项时继续执行，不要仅因"
        "单个窗口耗尽而宣告失败。\n\n"
        "[原始用户任务]\n"
        f"{_task_excerpt(original_task)}"
    )


def _resume_session_id(session: SessionT) -> str:
    session_id = str(getattr(session, "session_id", "") or "").strip()
    if not session_id:
        raise RuntimeError("ACP 执行窗口无法恢复：缺少 provider session_id")
    return session_id


def _merge_executions(
    previous: PromptContinuationResult | None,
    current: PromptContinuationResult,
    *,
    execution_windows: int,
) -> PromptContinuationResult:
    if previous is None:
        return replace(current, execution_windows=execution_windows)
    merged_result = merge_prompt_results(previous.result, current.result)
    merged_assessment = classify_prompt_result(merged_result)
    # ``run_prompt_with_continuation`` also treats an explicit user question as
    # incomplete. Preserve that stricter judgment if the structural classifier
    # alone would otherwise call the latest result complete.
    if (
        current.assessment.outcome is PromptOutcome.INCOMPLETE
        and merged_assessment.outcome is PromptOutcome.COMPLETED
    ):
        merged_assessment = current.assessment
    return PromptContinuationResult(
        result=merged_result,
        assessment=merged_assessment,
        automatic_continuations=(
            previous.automatic_continuations
            + current.automatic_continuations
        ),
        entered_finalization=(
            previous.entered_finalization or current.entered_finalization
        ),
        execution_windows=execution_windows,
        window_limit_reached=False,
    )


def run_prompt_across_execution_windows(
    session: SessionT,
    initial_prompt: str,
    *,
    task_scope: str | None = None,
    max_windows: int,
    execute_window: Callable[[SessionT, str], PromptContinuationResult],
    resume_window: Callable[[SessionT, str], SessionT],
    on_window_rollover: Callable[[int, int], None] | None = None,
) -> PromptContinuationResult:
    """Execute one logical task across fresh, resumed ACP transports.

    Rollover is intentionally narrow: only an incomplete result that actually
    entered timeout finalization gets another window. Cancellation, ordinary
    incomplete turns and completed work keep their existing terminal meaning.
    """
    if not isinstance(max_windows, int) or isinstance(max_windows, bool) or max_windows < 1:
        raise ValueError("max_windows must be a positive integer")

    current_session = session
    original_task = initial_prompt if task_scope is None else task_scope
    prompt = initial_prompt
    aggregate: PromptContinuationResult | None = None

    for window_index in range(1, max_windows + 1):
        current = execute_window(current_session, prompt)
        aggregate = _merge_executions(
            aggregate,
            current,
            execution_windows=window_index,
        )
        if current.assessment.outcome is not PromptOutcome.INCOMPLETE:
            return aggregate
        if not current.entered_finalization:
            return aggregate
        if window_index >= max_windows:
            return replace(aggregate, window_limit_reached=True)

        next_window = window_index + 1
        if on_window_rollover is not None:
            on_window_rollover(next_window, max_windows)
        current_session = resume_window(
            current_session,
            _resume_session_id(current_session),
        )
        prompt = _build_window_continuation_prompt(
            original_task,
            window_index=next_window,
            max_windows=max_windows,
            reason=current.assessment.detail,
        )

    raise AssertionError("execution window loop exhausted unexpectedly")


__all__ = ["run_prompt_across_execution_windows"]
