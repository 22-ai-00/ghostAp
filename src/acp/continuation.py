"""Bounded same-session continuation for ordinary ACP pending plans."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar

from .finalization import run_prompt_with_finalization
from .models import ACPEvent, PromptResult, ToolCallInfo
from .outcome import (
    PromptAssessment,
    PromptOutcome,
    classify_prompt_result,
)

logger = logging.getLogger(__name__)

MAX_ORDINARY_CONTINUATIONS = 1
_monotonic = time.monotonic


class _PromptSession(Protocol):
    _force_dead: bool

    def send_prompt(
        self,
        text: str,
        on_event: Callable[[ACPEvent], None] | None = None,
        timeout: float | int | None = None,
    ) -> PromptResult: ...


SessionT = TypeVar("SessionT", bound=_PromptSession)


@dataclass(frozen=True)
class PromptContinuationResult:
    """Final result and completion metadata for a bounded prompt execution."""

    result: PromptResult
    assessment: PromptAssessment
    automatic_continuations: int
    awaiting_user_input: bool


def _build_continuation_prompt(pending_plan_entries: int) -> str:
    return (
        "[GhostAP 自动续做指令]\n"
        f"上一轮自然结束，但结构化计划仍有 {pending_plan_entries} 项未完成。"
        "请保持原任务范围，在同一会话中继续执行。\n"
        "对于原任务范围内普通、安全、可逆的设计或实现选择，优先采用"
        "文档明确推荐的选项；如果没有明确推荐，采用最小安全默认值，"
        "记录该选择及理由后继续，不要仅因这类选择停下来询问。\n"
        "本指令不新增任何权限；保留原始用户请求已经明确授予的精确权限，"
        "但不得扩大。对于凭据、部署或发布、删除数据、不可逆外部副作用，"
        "只有原始用户请求已明确、精确授权时才能执行；不得新推断、扩大或"
        "替代授权。ACP、sandbox 或工具权限仍然有效，任何情况下都不得绕过。\n"
        "只推迟确实需要新权限或新外部授权的项目，并清楚记录原因；"
        "继续完成所有其他已获授权且在原任务范围内的工作。"
        "本轮结束时如实说明已完成、验证和仍需用户决定的事项。"
    )


def _notify_continuation_start(callback: Callable[[], None] | None) -> None:
    """Close the current stream turn without claiming another send occurred."""
    if callback is None:
        return
    try:
        callback()
    except Exception:
        logger.warning("prompt continuation callback failed", exc_info=True)


def _merge_tool_calls(
    previous: list[ToolCallInfo],
    current: list[ToolCallInfo],
) -> list[ToolCallInfo]:
    merged: dict[str, ToolCallInfo] = {}
    anonymous: list[ToolCallInfo] = []
    for tool_call in [*previous, *current]:
        if tool_call.id:
            merged[tool_call.id] = tool_call
        else:
            anonymous.append(tool_call)
    return [*merged.values(), *anonymous]


def _merge_prompt_results(
    previous: PromptResult,
    current: PromptResult,
) -> PromptResult:
    """Carry cross-turn evidence forward without mutating either turn result."""
    output_tokens: int | None = None
    if previous.output_tokens is not None or current.output_tokens is not None:
        output_tokens = (previous.output_tokens or 0) + (
            current.output_tokens or 0
        )
    return replace(
        current,
        tool_calls=_merge_tool_calls(
            previous.tool_calls,
            current.tool_calls,
        ),
        tool_results=[*previous.tool_results, *current.tool_results],
        plan=current.plan if current.plan is not None else previous.plan,
        modified_files={
            *previous.modified_files,
            *current.modified_files,
        },
        output_tokens=output_tokens,
    )


def _eligible_for_ordinary_continuation(
    assessment: PromptAssessment,
    *,
    entered_finalization: bool,
) -> bool:
    return (
        not entered_finalization
        and assessment.outcome is PromptOutcome.INCOMPLETE
        and assessment.stop_reason == "end_turn"
        and assessment.pending_plan_entries > 0
        and assessment.incomplete_tool_calls == 0
    )


def run_prompt_with_continuation(
    session: SessionT,
    text: str,
    *,
    on_event: Callable[[ACPEvent], None] | None = None,
    timeout_s: float | int,
    finalization_reserve_s: float | int,
    finalization_task_text: str | None = None,
    on_finalization_start: Callable[[], None] | None = None,
    on_continuation_start: Callable[[], None] | None = None,
    replace_dead_session: Callable[[float], SessionT] | None = None,
    retire_finalization_session: Callable[[SessionT, float], None] | None = None,
) -> PromptContinuationResult:
    """Run one prompt and at most one safe same-session continuation.

    ``on_continuation_start`` is a best-effort structural boundary hook for
    closing streamed blocks. Its invocation does not mean the continuation was
    sent. If the hook consumes the remaining deadline, the first incomplete
    result is returned with zero automatic continuations so the caller can use
    its ordinary incomplete/timeout handling.
    """
    deadline = _monotonic() + float(timeout_s)
    finalization_scope = (
        text if finalization_task_text is None else finalization_task_text
    )

    def run_turn(
        prompt: str,
        turn_timeout_s: float | int,
    ) -> tuple[PromptResult, bool]:
        entered_finalization = False

        def mark_finalization_start() -> None:
            nonlocal entered_finalization
            entered_finalization = True
            if on_finalization_start is not None:
                on_finalization_start()

        result = run_prompt_with_finalization(
            session,
            prompt,
            on_event=on_event,
            timeout_s=turn_timeout_s,
            finalization_reserve_s=finalization_reserve_s,
            finalization_task_text=finalization_scope,
            on_finalization_start=mark_finalization_start,
            replace_dead_session=replace_dead_session,
            retire_finalization_session=retire_finalization_session,
        )
        return result, entered_finalization

    first_turn_budget = deadline - _monotonic()
    result, entered_finalization = run_turn(text, first_turn_budget)
    assessment = classify_prompt_result(result)
    automatic_continuations = 0

    while (
        automatic_continuations < MAX_ORDINARY_CONTINUATIONS
        and _eligible_for_ordinary_continuation(
            assessment,
            entered_finalization=entered_finalization,
        )
    ):
        remaining_budget = deadline - _monotonic()
        if remaining_budget <= 0:
            break
        continuation_prompt = _build_continuation_prompt(
            assessment.pending_plan_entries
        )
        _notify_continuation_start(on_continuation_start)
        remaining_budget = deadline - _monotonic()
        if remaining_budget <= 0:
            break
        next_result, entered_finalization = run_turn(
            continuation_prompt,
            remaining_budget,
        )
        automatic_continuations += 1
        result = _merge_prompt_results(result, next_result)
        assessment = classify_prompt_result(result)

    awaiting_user_input = (
        automatic_continuations == MAX_ORDINARY_CONTINUATIONS
        and _eligible_for_ordinary_continuation(
            assessment,
            entered_finalization=entered_finalization,
        )
    )
    return PromptContinuationResult(
        result=result,
        assessment=assessment,
        automatic_continuations=automatic_continuations,
        awaiting_user_input=awaiting_user_input,
    )


__all__ = [
    "MAX_ORDINARY_CONTINUATIONS",
    "PromptContinuationResult",
    "run_prompt_with_continuation",
]
