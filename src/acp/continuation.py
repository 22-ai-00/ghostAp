"""Bounded same-session continuation for incomplete ACP prompt state."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar

from .collaboration import merge_tool_call_sequence
from .finalization import (
    _primary_timeout_with_cleanup_reserve,
    run_prompt_with_finalization,
)
from .models import ACPEvent, PromptResult, ToolCallInfo
from .outcome import (
    PromptAssessment,
    PromptOutcome,
    classify_prompt_result,
)

logger = logging.getLogger(__name__)

MAX_ORDINARY_CONTINUATIONS = 3
MAX_CHILD_RECONCILIATIONS = 1
MAX_AUTOMATIC_DECISIONS = 1
MAX_GOAL_RECOVERIES = 1
_monotonic = time.monotonic
_PLAN_CONTINUATION = "plan"
_CHILD_RECONCILIATION = "child_reconciliation"
_GOAL_RECOVERY = "goal_recovery"
_USER_INPUT_MARKERS = (
    "请确认",
    "请选择",
    "请回复",
    "需要你确认",
    "等待你的确认",
    "please confirm",
    "please choose",
    "please reply",
    "need your confirmation",
    "can i ",
    "may i ",
    "should i ",
    "do you want me to",
    "would you like me to",
    "are you sure",
)
_USER_INPUT_DANGEROUS_MARKERS = (
    "凭据",
    "凭证",
    "api key",
    "apikey",
    "credential",
    "secret",
    "密码",
    "password",
    "token",
    "权限",
    "permission",
    "授权",
    "authorization",
    "authenticate",
    "其他进程",
    "并发修改",
    "接管",
    "部署",
    "deploy",
    "production",
    "发布",
    "发布到",
    "publish",
    "release to",
    "删除",
    "delete",
    "清空",
    "wipe",
    "purge",
    "销毁",
    "destroy",
    "格式化",
    "format disk",
    "移除",
    "remove data",
    "清理",
    "付费",
    "billing",
    "charge",
    "pay",
    "购买",
    "purchase",
    "rm -",
    "drop ",
    "truncate",
)


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
    entered_finalization: bool
    execution_windows: int = 1
    window_limit_reached: bool = False


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


def _build_child_reconciliation_prompt(
    unresolved_child_tool_calls: int,
) -> str:
    return (
        "[GhostAP 子代理状态对账指令]\n"
        "上一轮自然结束，但结构化结果中仍有 "
        f"{unresolved_child_tool_calls} 个协作工具携带未终态子代理状态。"
        "这可能是异步 FINAL_ANSWER 已到达、而协作快照尚未刷新的竞态。\n"
        "请先调用 list_agents 获取权威最新状态；对仍为 running/pending "
        "的子代理，使用 wait_agent 等待并接收其最终结果。wait_agent 可能被普通 "
        "MESSAGE 提前唤醒，MESSAGE 不代表子代理已进入终态；每次 wait_agent 返回后"
        "必须再次调用 list_agents。若仍有 running/pending 子代理，在本轮内继续执行 "
        "wait_agent -> list_agents，直到权威状态全部为 completed/failed/cancelled，"
        "或本轮达到工具期限。不要仅凭已收到文本或 end_turn 推断子代理完成，也不要"
        "重复已经完成的实现。\n"
        "本指令不新增任何权限，不得扩大原任务范围，也不得中止或取消子代理。"
        "所有子代理进入 completed/failed/cancelled 终态后，整合现有结果并"
        "如实给出最终答复。"
    )


def _notify_continuation_start(callback: Callable[[], None] | None) -> None:
    """Close the current stream turn without claiming another send occurred."""
    if callback is None:
        return
    try:
        callback()
    except Exception:
        logger.warning("prompt continuation callback failed", exc_info=True)


def _confirmation_context(result: PromptResult) -> str:
    """Collect visible and structured evidence relevant to authorization."""
    parts = [str(result.text or "")]
    if result.plan is not None:
        parts.extend(str(entry.content or "") for entry in result.plan.entries)
    for tool_call in result.tool_calls:
        parts.extend((str(tool_call.title or ""), str(tool_call.kind or "")))
    parts.extend(str(item) for item in result.tool_results)
    return "\n".join(parts).strip().casefold()


def _requests_explicit_user_input(result: PromptResult) -> bool:
    text = str(result.text or "").strip().casefold()
    if not text:
        return False
    if any(marker in text for marker in _USER_INPUT_MARKERS):
        return True
    return text.endswith(("?", "？")) and _has_dangerous_confirmation_requirement(
        _confirmation_context(result)
    )


def _has_dangerous_confirmation_requirement(text: str) -> bool:
    normalized = (text or "").strip().casefold()
    if not normalized:
        return False
    return any(marker in normalized for marker in _USER_INPUT_DANGEROUS_MARKERS)


def _normalize_user_input_assessment(
    result: PromptResult,
    assessment: PromptAssessment,
) -> PromptAssessment:
    """Never report a turn that explicitly asks a question as completed."""
    if (
        assessment.outcome is PromptOutcome.COMPLETED
        and assessment.stop_reason == "end_turn"
        and _requests_explicit_user_input(result)
    ):
        return replace(
            assessment,
            outcome=PromptOutcome.INCOMPLETE,
            detail="模型仍在请求选择或授权，任务尚未完成。",
        )
    return assessment


def _build_confirmation_default_prompt() -> str:
    return (
        "[GhostAP 自动续做默认决策]\n"
        "上一步出现了“请选择/请确认”等提示，但未出现明确新增权限、发布部署、"
        "删除数据、不可逆外部副作用的高风险诉求。\n"
        "本次按“文档推荐选项 + 最小可逆本地默认值”自动继续，不要再次询问。"
        "这不构成新增授权：不得猜测凭据或权限，不得发布、部署、付费、删除数据，"
        "也不得执行不可逆外部操作；若确实需要这些授权，请只说明精确阻塞项。"
    )


def _build_risk_denial_prompt() -> str:
    return (
        "[GhostAP 自动安全决策]\n"
        "上一步请求了新增凭据/权限，或部署、发布、删除数据等高风险操作。"
        "该操作未获原始请求的明确精确授权，本次自动拒绝并跳过，不要再次询问，"
        "也不得寻找绕过方式。请继续完成原任务中其余已授权、安全、可逆的工作；"
        "若被拒绝的操作是完成任务的必要条件，请保留已有结果并明确报告失败原因。"
    )


def _build_goal_recovery_prompt(status: str) -> str:
    return (
        "[GhostAP 自动恢复指令]\n"
        f"结构化 Goal 当前为 {status}，但任务不能等待用户交互。请在同一会话中"
        "采用推荐的安全可逆默认值继续；需要新增权限、凭据或不可逆副作用的步骤"
        "一律拒绝并跳过。不要再次暂停、阻塞或询问用户。若无法安全完成，请完整"
        "保留已有结果并明确报告失败原因。"
    )


def _send_child_reconciliation_prompt(
    session: SessionT,
    text: str,
    *,
    on_event: Callable[[ACPEvent], None] | None,
    timeout: float | int,
) -> PromptResult:
    method = getattr(session, "send_reconciliation_prompt", None)
    if callable(method):
        return method(text, on_event=on_event, timeout=timeout)
    return session.send_prompt(text, on_event=on_event, timeout=timeout)


def _retire_reconciliation_timeout(
    session: SessionT,
    callback: Callable[[SessionT, float], None] | None,
    *,
    deadline: float,
    error: BaseException | None,
) -> None:
    """Poison and retire a timed-out reconciliation without another turn."""
    try:
        setattr(session, "_force_dead", True)
        if callback is not None:
            callback(session, max(0.0, deadline - _monotonic()))
    except Exception as retirement_error:
        if error is not None:
            raise ExceptionGroup(
                "ACP child reconciliation timeout and retirement both failed",
                [error, retirement_error],
            ) from None
        raise


def _merge_tool_calls(
    previous: list[ToolCallInfo],
    current: list[ToolCallInfo],
) -> list[ToolCallInfo]:
    return merge_tool_call_sequence(
        [*previous, *current],
        generation_boundary_index=len(previous),
    )


def merge_prompt_results(
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


def _continuation_kind(
    result: PromptResult,
    assessment: PromptAssessment,
    *,
    entered_finalization: bool,
) -> str | None:
    if (
        entered_finalization
        or assessment.outcome is not PromptOutcome.INCOMPLETE
        or assessment.stop_reason != "end_turn"
    ):
        return None
    if _requests_explicit_user_input(result):
        return None
    goal = result.goal
    goal_status = (
        str(goal.status or "").strip().casefold()
        if goal is not None
        else ""
    )
    if (
        assessment.pending_plan_entries == 0
        and assessment.incomplete_tool_calls > 0
        and assessment.incomplete_outer_tool_calls == 0
        and assessment.incomplete_tool_calls
        == assessment.unresolved_child_tool_calls
    ):
        return _CHILD_RECONCILIATION
    if (
        goal_status in {"paused", "blocked"}
        and assessment.incomplete_tool_calls == 0
    ):
        return _GOAL_RECOVERY
    if (
        (goal is None or goal_status in {"active", "completed"})
        and assessment.pending_plan_entries > 0
        and assessment.incomplete_tool_calls == 0
    ):
        return _PLAN_CONTINUATION
    return None


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
    """Run a prompt with bounded plan continuation and child reconciliation.

    ``on_continuation_start`` is a best-effort structural boundary hook for
    closing streamed blocks for either plan continuation or child-state
    reconciliation. Its invocation does not mean the follow-up was sent. If the
    hook consumes the remaining deadline, the first incomplete result is returned
    with zero automatic continuations so the caller can use its ordinary
    incomplete/timeout handling.
    """
    deadline = _monotonic() + float(timeout_s)
    logical_task_started_at = time.time()
    finalization_scope = (
        text if finalization_task_text is None else finalization_task_text
    )

    def run_turn(
        prompt: str,
        turn_timeout_s: float | int,
        *,
        replay_deferred_child_events: bool = False,
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
            replay_deferred_child_events=replay_deferred_child_events,
        )
        return result, entered_finalization

    first_turn_budget = deadline - _monotonic()
    result, entered_finalization = run_turn(text, first_turn_budget)
    ever_entered_finalization = entered_finalization
    assessment = _normalize_user_input_assessment(
        result,
        classify_prompt_result(result),
    )
    automatic_continuations = 0
    plan_continuations = 0
    child_reconciliations = 0
    automatic_decisions = 0
    goal_recoveries = 0

    while True:
        continuation_kind = _continuation_kind(
            result,
            assessment,
            entered_finalization=ever_entered_finalization,
        )

        if continuation_kind is None:
            if (
                not ever_entered_finalization
                and automatic_decisions < MAX_AUTOMATIC_DECISIONS
                and assessment.outcome is PromptOutcome.INCOMPLETE
                and assessment.stop_reason == "end_turn"
                and _requests_explicit_user_input(result)
            ):
                _notify_continuation_start(on_continuation_start)
                remaining_budget = deadline - _monotonic()
                if remaining_budget <= 0:
                    break
                decision_prompt = (
                    _build_risk_denial_prompt()
                    if _has_dangerous_confirmation_requirement(
                        _confirmation_context(result)
                    )
                    else _build_confirmation_default_prompt()
                )
                next_result, entered_finalization = run_turn(
                    decision_prompt,
                    remaining_budget,
                    replay_deferred_child_events=True,
                )
                automatic_decisions += 1
                ever_entered_finalization = (
                    ever_entered_finalization or entered_finalization
                )
                automatic_continuations += 1
                result = merge_prompt_results(result, next_result)
                assessment = _normalize_user_input_assessment(
                    result,
                    classify_prompt_result(result),
                )
                continue
            break

        if (
            continuation_kind == _PLAN_CONTINUATION
            and plan_continuations >= MAX_ORDINARY_CONTINUATIONS
        ) or (
            continuation_kind == _CHILD_RECONCILIATION
            and child_reconciliations >= MAX_CHILD_RECONCILIATIONS
        ) or (
            continuation_kind == _GOAL_RECOVERY
            and goal_recoveries >= MAX_GOAL_RECOVERIES
        ):
            break

        raw_goal_status = getattr(result.goal, "status", "")
        goal_status = str(
            getattr(raw_goal_status, "value", raw_goal_status) or ""
        ).strip().casefold()
        continuation_prompt = (
            _build_continuation_prompt(assessment.pending_plan_entries)
            if continuation_kind == _PLAN_CONTINUATION
            else _build_goal_recovery_prompt(goal_status)
            if continuation_kind == _GOAL_RECOVERY
            else _build_child_reconciliation_prompt(
                assessment.unresolved_child_tool_calls
            )
        )
        _notify_continuation_start(on_continuation_start)
        remaining_budget = deadline - _monotonic()
        if remaining_budget <= 0:
            break
        reconciliation_started_at: float | None = None
        reconciliation_finished_at: float | None = None
        if continuation_kind in {_PLAN_CONTINUATION, _GOAL_RECOVERY}:
            next_result, entered_finalization = run_turn(
                continuation_prompt,
                remaining_budget,
                replay_deferred_child_events=True,
            )
            if continuation_kind == _PLAN_CONTINUATION:
                plan_continuations += 1
            else:
                goal_recoveries += 1
        else:
            # Reconciliation is itself the single safe cleanup turn. Running it
            # through the ordinary timeout-finalization wrapper could send a
            # third prompt or replace the session, defeating that bound.
            reconciliation_timeout = _primary_timeout_with_cleanup_reserve(
                remaining_budget
            )
            reconciliation_started_at = time.time()
            try:
                next_result = _send_child_reconciliation_prompt(
                    session,
                    continuation_prompt,
                    on_event=on_event,
                    timeout=reconciliation_timeout,
                )
                reconciliation_finished_at = time.time()
            except TimeoutError as timeout_error:
                _retire_reconciliation_timeout(
                    session,
                    retire_finalization_session,
                    deadline=deadline,
                    error=timeout_error,
                )
                raise
            if (
                str(next_result.stop_reason or "").strip().casefold()
                == "timeout"
            ):
                _retire_reconciliation_timeout(
                    session,
                    retire_finalization_session,
                    deadline=deadline,
                    error=None,
                )
            entered_finalization = False
            child_reconciliations += 1
        ever_entered_finalization = (
            ever_entered_finalization or entered_finalization
        )
        automatic_continuations += 1
        result = merge_prompt_results(result, next_result)
        if (
            reconciliation_started_at is not None
            and reconciliation_finished_at is not None
        ):
            enrich_result = getattr(
                session,
                "enrich_child_reconciliation_result",
                None,
            )
            if callable(enrich_result):
                try:
                    enriched = enrich_result(
                        result,
                        started_at=reconciliation_started_at,
                        ended_at=reconciliation_finished_at,
                        logical_task_started_at=logical_task_started_at,
                        on_event=on_event,
                    )
                    if isinstance(enriched, PromptResult):
                        result = enriched
                except Exception:
                    logger.warning(
                        "Codex child reconciliation evidence enrichment failed",
                        exc_info=True,
                    )
        assessment = _normalize_user_input_assessment(
            result,
            classify_prompt_result(result),
        )

    return PromptContinuationResult(
        result=result,
        assessment=assessment,
        automatic_continuations=automatic_continuations,
        entered_finalization=ever_entered_finalization,
    )


__all__ = [
    "MAX_CHILD_RECONCILIATIONS",
    "MAX_ORDINARY_CONTINUATIONS",
    "PromptContinuationResult",
    "merge_prompt_results",
    "run_prompt_with_continuation",
]
