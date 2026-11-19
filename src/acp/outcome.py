"""Fail-closed completion classification for agent prompt results."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .child_lifecycle import (
    ChildLifecycleReducer,
    normalize_provider_child_status,
)


class PromptOutcome(str, Enum):
    """User-work outcome, distinct from a transport returning normally."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class IncompleteToolTrace:
    """Privacy-safe correlation evidence for one unresolved tool call."""

    id_sha256: str
    diagnostic: str
    child_source_present: bool
    context_compaction: bool
    child_metadata_malformed: bool


@dataclass(frozen=True)
class PromptAssessment:
    """Normalized outcome and a user-facing diagnostic."""

    outcome: PromptOutcome
    stop_reason: str
    detail: str
    pending_plan_entries: int = 0
    incomplete_tool_calls: int = 0
    incomplete_outer_tool_calls: int = 0
    unresolved_child_tool_calls: int = 0
    incomplete_tool_diagnostics: tuple[str, ...] = ()
    # Opaque diagnostics for structured logs only; never render these on cards.
    incomplete_tool_traces: tuple[IncompleteToolTrace, ...] = ()


_CANCELLED_REASONS = frozenset({"cancelled", "canceled"})
_TERMINAL_TOOL_STATUSES = frozenset({"completed", "failed"})
_CONTEXT_COMPACTION_TRANSIENT_STATUSES = frozenset(
    {"pending", "in_progress", "running"}
)
_KNOWN_DIAGNOSTIC_TOOLS = frozenset(
    {
        "followup_task",
        "interrupt_agent",
        "list_agents",
        "send_message",
        "spawn_agent",
        "wait_agent",
    }
)
_KNOWN_DIAGNOSTIC_KINDS = frozenset(
    {
        "agent",
        "delete",
        "edit",
        "execute",
        "fetch",
        "other",
        "read",
        "search",
        "think",
    }
)
_KNOWN_DIAGNOSTIC_OUTER_STATUSES = frozenset(
    {"completed", "failed", "in_progress", "pending", "running"}
)
_MAX_TOOL_DIAGNOSTICS = 8


def _status(value: Any) -> str:
    return str(getattr(value, "status", "") or "").strip().casefold()


def _is_nonblocking_context_compaction(tool: object) -> bool:
    """Exclude only an explicitly tagged internal Codex bookkeeping item.

    The Codex adapter may emit a context-compaction start without its matching
    completion.  It does not represent user work, so it cannot by itself keep
    a completed prompt open.  Require both the exact boolean marker and the
    provider's ``other`` kind; every normal in-progress tool remains blocking.
    """
    return (
        getattr(tool, "is_context_compaction", False) is True
        and str(getattr(tool, "kind", "") or "").strip().casefold()
        == "other"
        and _status(tool) in _CONTEXT_COMPACTION_TRANSIENT_STATUSES
    )


def _tool_incompleteness_details(
    tool_calls: list[object],
) -> tuple[list[object], int, int, bool]:
    """Fold child lifecycles while keeping malformed evidence fail-closed."""
    incomplete_outer = {
        index
        for index, tool in enumerate(tool_calls)
        if (
            _status(tool) not in _TERMINAL_TOOL_STATUSES
            and not _is_nonblocking_context_compaction(tool)
        )
    }
    reducer = ChildLifecycleReducer()
    for index, tool in enumerate(tool_calls):
        reducer.apply(tool, index=index)

    transient_child_indexes = reducer.transient_indexes
    unresolved_children = reducer.invalid_indexes | transient_child_indexes
    incomplete_indexes = incomplete_outer | unresolved_children
    return (
        [tool_calls[index] for index in sorted(incomplete_indexes)],
        len(incomplete_outer),
        len(unresolved_children),
        bool(transient_child_indexes),
    )


def _tool_incompleteness(
    tool_calls: list[object],
) -> tuple[list[object], int, int]:
    incomplete, outer_count, child_count, _ = _tool_incompleteness_details(
        tool_calls
    )
    return incomplete, outer_count, child_count


def has_transient_child_lifecycle(tool_calls: Iterable[object]) -> bool:
    """Whether a child is still pending/running after stable-source folding."""
    return _tool_incompleteness_details(list(tool_calls))[3]


def _allowlisted_token(
    value: object,
    *,
    allowed: frozenset[str],
    missing: str = "missing",
) -> str:
    if not isinstance(value, str) or not value.strip():
        return missing
    normalized = value.strip().casefold()
    return normalized if normalized in allowed else "unknown"


def _tool_diagnostic(tool: object) -> str:
    collaboration_tool = getattr(tool, "collaboration_tool", None)
    if collaboration_tool:
        name = _allowlisted_token(
            collaboration_tool,
            allowed=_KNOWN_DIAGNOSTIC_TOOLS,
        )
    else:
        name = _allowlisted_token(
            getattr(tool, "kind", None),
            allowed=_KNOWN_DIAGNOSTIC_KINDS,
        )
    outer_status = _allowlisted_token(
        getattr(tool, "status", None),
        allowed=_KNOWN_DIAGNOSTIC_OUTER_STATUSES,
    )
    child_statuses: list[str] = []
    if getattr(tool, "child_metadata_malformed", False) is not False:
        child_statuses.append("malformed")
    raw_children = getattr(tool, "subagent_states", None)
    if isinstance(raw_children, Iterable) and not isinstance(
        raw_children,
        (str, bytes, bytearray, Mapping),
    ):
        for child in raw_children:
            if isinstance(child, Mapping):
                raw_status = child.get("status")
                normalized = normalize_provider_child_status(raw_status)
                status = (
                    normalized
                    if normalized is not None
                    else (
                        "missing"
                        if not isinstance(raw_status, str)
                        or not raw_status.strip()
                        else "unknown"
                    )
                )
            else:
                status = "malformed"
            if status not in child_statuses:
                child_statuses.append(status)
    elif raw_children is not None:
        child_statuses.append("malformed")
    if not child_statuses:
        raw_receivers = getattr(tool, "collaboration_receivers", ())
        if raw_receivers is not None and (
            isinstance(raw_receivers, (str, bytes, bytearray, Mapping))
            or not isinstance(raw_receivers, Iterable)
        ):
            child_statuses.append("malformed")
        elif raw_receivers:
            child_statuses.append("pending")
    if not child_statuses:
        activity = getattr(tool, "subagent_activity", None)
        if isinstance(activity, str) and activity.strip().casefold() in {
            "started",
            "interacted",
        }:
            child_statuses.append("running")
    child_summary = ",".join(child_statuses) or "-"
    return f"{name}:{outer_status}[{child_summary}]"


def _tool_id_sha256(value: object) -> str:
    """Return a bounded opaque correlation value without logging the raw id."""
    if not isinstance(value, str) or not value:
        return "missing"
    if len(value) > 512:
        return "invalid"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _incomplete_tool_trace(tool: object) -> IncompleteToolTrace:
    raw_source = getattr(tool, "subagent_source_id", None)
    return IncompleteToolTrace(
        id_sha256=_tool_id_sha256(getattr(tool, "id", None)),
        diagnostic=_tool_diagnostic(tool),
        child_source_present=bool(
            isinstance(raw_source, str) and raw_source.strip()
        ),
        context_compaction=(
            getattr(tool, "is_context_compaction", False) is True
        ),
        child_metadata_malformed=(
            getattr(tool, "child_metadata_malformed", False) is not False
        ),
    )


def classify_prompt_result(result: object) -> PromptAssessment:
    """Classify whether a prompt result proves that requested work is complete.

    ``end_turn`` is necessary but not sufficient: a result with pending plan
    entries or any non-terminal tool call is incomplete. Child snapshots are
    reconciled by stable source id across collaboration calls so stale history
    cannot override later terminal evidence. Failed tool attempts are terminal
    and may be followed by a successful recovery within the same prompt;
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
    tool_calls = list(getattr(result, "tool_calls", None) or ())
    (
        incomplete_tools,
        incomplete_outer_tool_calls,
        unresolved_child_tool_calls,
    ) = _tool_incompleteness(tool_calls)
    counts = {
        "pending_plan_entries": len(pending_plan),
        "incomplete_tool_calls": len(incomplete_tools),
        "incomplete_outer_tool_calls": incomplete_outer_tool_calls,
        "unresolved_child_tool_calls": unresolved_child_tool_calls,
        "incomplete_tool_diagnostics": tuple(
            _tool_diagnostic(tool) for tool in incomplete_tools
        )[:_MAX_TOOL_DIAGNOSTICS],
        "incomplete_tool_traces": tuple(
            _incomplete_tool_trace(tool) for tool in incomplete_tools
        )[:_MAX_TOOL_DIAGNOSTICS],
    }

    if stop_reason in _CANCELLED_REASONS:
        cancellation_source = str(
            getattr(result, "cancellation_source", "") or ""
        ).strip().casefold()
        return PromptAssessment(
            outcome=(
                PromptOutcome.CANCELLED
                if cancellation_source == "user"
                else PromptOutcome.INCOMPLETE
            ),
            stop_reason=stop_reason,
            detail=(
                "用户已明确取消当前 ACP 任务"
                if cancellation_source == "user"
                else (
                    f"ACP 停止原因：{stop_reason}"
                    f"（中断来源：{cancellation_source or 'provider'}）"
                )
            ),
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
    "IncompleteToolTrace",
    "PromptAssessment",
    "PromptOutcome",
    "classify_prompt_result",
    "has_transient_child_lifecycle",
]
