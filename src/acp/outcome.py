"""Fail-closed completion classification for agent prompt results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PromptOutcome(str, Enum):
    """User-work outcome, distinct from a transport returning normally."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class PromptAssessment:
    """Normalized outcome and a user-facing diagnostic."""

    outcome: PromptOutcome
    stop_reason: str
    detail: str
    pending_plan_entries: int = 0
    incomplete_tool_calls: int = 0


_CANCELLED_REASONS = frozenset({"cancelled", "canceled"})
_TERMINAL_TOOL_STATUSES = frozenset({"completed", "failed"})
_TERMINAL_CHILD_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _status(value: Any) -> str:
    return str(getattr(value, "status", "") or "").strip().casefold()


def _has_unresolved_child(tool_call: object) -> bool:
    for child in getattr(tool_call, "subagent_states", None) or ():
        if not isinstance(child, Mapping):
            return True
        status = str(child.get("status") or "").strip().casefold()
        if status not in _TERMINAL_CHILD_STATUSES:
            return True
    return False


def classify_prompt_result(result: object) -> PromptAssessment:
    """Classify whether a prompt result proves that requested work is complete.

    ``end_turn`` is necessary but not sufficient: a result with pending plan
    entries or any non-terminal tool call is incomplete. Failed tool attempts are
    terminal and may be followed by a successful recovery within the same prompt;
    treating their history as pending would make every TDD/debugging turn fail.
    Unknown states fail closed so backend additions cannot silently become success.
    """

    stop_reason = str(getattr(result, "stop_reason", "") or "").strip().casefold()
    plan = getattr(result, "plan", None)
    pending_plan = [
        entry
        for entry in (getattr(plan, "entries", None) or ())
        if _status(entry) != "completed"
    ]
    incomplete_tools = [
        tool
        for tool in (getattr(result, "tool_calls", None) or ())
        if (
            _status(tool) not in _TERMINAL_TOOL_STATUSES
            or _has_unresolved_child(tool)
        )
    ]
    counts = {
        "pending_plan_entries": len(pending_plan),
        "incomplete_tool_calls": len(incomplete_tools),
    }

    if stop_reason in _CANCELLED_REASONS:
        return PromptAssessment(
            outcome=PromptOutcome.CANCELLED,
            stop_reason=stop_reason,
            detail=f"ACP 停止原因：{stop_reason}",
            **counts,
        )
    if stop_reason != "end_turn":
        reason = stop_reason or "missing_stop_reason"
        return PromptAssessment(
            outcome=PromptOutcome.INCOMPLETE,
            stop_reason=reason,
            detail=f"ACP 停止原因：{reason}",
            **counts,
        )

    goal = getattr(result, "goal", None)
    goal_status = _status(goal)
    if goal is not None and goal_status != "completed":
        detail = {
            "active": "Codex Goal 仍在执行",
            "paused": "Codex Goal 已暂停",
            "blocked": "Codex Goal 已阻塞",
        }.get(
            goal_status,
            f"Codex Goal 状态未知：{goal_status or 'missing'}",
        )
        return PromptAssessment(
            PromptOutcome.INCOMPLETE,
            stop_reason,
            detail,
            **counts,
        )

    if pending_plan:
        return PromptAssessment(
            outcome=PromptOutcome.INCOMPLETE,
            stop_reason=stop_reason,
            detail=f"仍有 {len(pending_plan)} 个计划项未完成",
            **counts,
        )

    if incomplete_tools:
        return PromptAssessment(
            outcome=PromptOutcome.INCOMPLETE,
            stop_reason=stop_reason,
            detail=f"仍有 {len(incomplete_tools)} 个工具调用未进入终态",
            **counts,
        )

    return PromptAssessment(
        outcome=PromptOutcome.COMPLETED,
        stop_reason=stop_reason,
        detail="ACP 已正常结束且没有未决计划或非终态工具调用",
        **counts,
    )


__all__ = [
    "PromptAssessment",
    "PromptOutcome",
    "classify_prompt_result",
]
