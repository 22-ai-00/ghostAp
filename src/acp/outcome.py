"""Fail-closed completion classification for agent prompt results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .collaboration import (
    TERMINAL_CHILD_STATUSES,
    TRANSIENT_CHILD_STATUSES,
    strict_child_status,
    strict_source_id,
)


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
    incomplete_outer_tool_calls: int = 0
    unresolved_child_tool_calls: int = 0
    incomplete_tool_diagnostics: tuple[str, ...] = ()


_CANCELLED_REASONS = frozenset({"cancelled", "canceled"})
_TERMINAL_TOOL_STATUSES = frozenset({"completed", "failed"})
_CHILD_GENERATION_ACTIONS = frozenset({"followup_task", "spawn_agent"})
_PASSIVE_CHILD_ACTIONS = frozenset(
    {"interrupt_agent", "list_agents", "send_message", "wait_agent"}
)
_KNOWN_CHILD_ACTIONS = _CHILD_GENERATION_ACTIONS | _PASSIVE_CHILD_ACTIONS
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


def _tool_incompleteness_details(
    tool_calls: list[object],
) -> tuple[list[object], int, int, bool]:
    """Fold child lifecycles while keeping malformed evidence fail-closed."""
    incomplete_outer = {
        index
        for index, tool in enumerate(tool_calls)
        if _status(tool) not in _TERMINAL_TOOL_STATUSES
    }
    lifecycle_by_source: dict[str, tuple[str, int]] = {}
    invalid_indexes: set[int] = set()

    for index, tool in enumerate(tool_calls):
        malformed_marker = getattr(
            tool,
            "child_metadata_malformed",
            False,
        )
        if malformed_marker is not False:
            invalid_indexes.add(index)
        observations: list[tuple[str, str]] = []
        represented_sources: set[str] = set()
        normalized_activity = ""
        activity_source: str | None = None
        raw_children = getattr(tool, "subagent_states", None)
        if raw_children is not None and (
            isinstance(raw_children, (str, bytes, bytearray, Mapping))
            or not isinstance(raw_children, Iterable)
        ):
            invalid_indexes.add(index)
            child_items: Iterable[object] = ()
        else:
            child_items = raw_children or ()
        for child in child_items:
            if not isinstance(child, Mapping):
                invalid_indexes.add(index)
                continue
            source_id = strict_source_id(child.get("source_id"))
            child_status = strict_child_status(child.get("status"))
            if source_id is None or child_status is None:
                invalid_indexes.add(index)
                continue
            represented_sources.add(source_id)
            observations.append((source_id, child_status))

        raw_receivers = getattr(tool, "collaboration_receivers", ())
        if raw_receivers is not None and (
            isinstance(raw_receivers, (str, bytes, bytearray, Mapping))
            or not isinstance(raw_receivers, Iterable)
        ):
            invalid_indexes.add(index)
            receiver_items: Iterable[object] = ()
        else:
            receiver_items = raw_receivers or ()
        receivers: list[str] = []
        for raw_receiver in receiver_items:
            receiver = strict_source_id(raw_receiver)
            if receiver is None:
                invalid_indexes.add(index)
                continue
            if receiver not in receivers:
                receivers.append(receiver)
            if receiver not in represented_sources:
                observations.append((receiver, "pending"))

        raw_activity = getattr(tool, "subagent_activity", None)
        raw_activity_source = getattr(tool, "subagent_source_id", None)
        if raw_activity not in (None, "") or raw_activity_source not in (
            None,
            "",
        ):
            activity_source = strict_source_id(raw_activity_source)
            normalized_activity = (
                raw_activity.strip().casefold()
                if isinstance(raw_activity, str)
                else ""
            )
            if activity_source is None:
                invalid_indexes.add(index)
            elif normalized_activity not in {
                "started",
                "interacted",
                "interrupted",
            }:
                invalid_indexes.add(index)
            elif activity_source not in represented_sources:
                if normalized_activity in {"started", "interacted"}:
                    observations.append((activity_source, "running"))
                else:
                    observations.append(
                        (
                            activity_source,
                            "cancelled"
                            if _status(tool) == "completed"
                            else "running",
                        )
                    )

        collaboration_tool = getattr(tool, "collaboration_tool", None)
        if isinstance(collaboration_tool, str):
            normalized_collaboration_tool = (
                collaboration_tool.strip().casefold()
            )
        else:
            normalized_collaboration_tool = ""
            if collaboration_tool is not None:
                invalid_indexes.add(index)
        if not normalized_collaboration_tool:
            title = getattr(tool, "title", None)
            normalized_title = (
                title.strip().casefold() if isinstance(title, str) else ""
            )
            if normalized_title in _KNOWN_CHILD_ACTIONS:
                normalized_collaboration_tool = normalized_title
        if (
            normalized_collaboration_tool
            in {"spawn_agent", "followup_task"}
            and _status(tool) == "completed"
            and not observations
        ):
            invalid_indexes.add(index)
        transient_sources = {
            source_id
            for source_id, child_status in observations
            if child_status in TRANSIENT_CHILD_STATUSES
        }
        activity_is_transient = (
            normalized_activity in {"started", "interacted"}
            or (
                normalized_activity == "interrupted"
                and _status(tool) != "completed"
            )
        )
        reset_sources: set[str] = set()
        if normalized_collaboration_tool in _CHILD_GENERATION_ACTIONS:
            if _status(tool) == "completed":
                reset_sources.update(receivers)
                reset_sources.update(
                    source_id for source_id, _ in observations
                )
            else:
                # A failed outer action that already exposes a running child is
                # ambiguous: treat it as a new generation until a later
                # terminal snapshot resolves it.
                reset_sources.update(transient_sources)
        elif activity_is_transient and activity_source is not None:
            reset_sources.add(activity_source)
        elif (
            normalized_collaboration_tool
            and normalized_collaboration_tool not in _PASSIVE_CHILD_ACTIONS
        ):
            # An unknown lifecycle action may start a new generation. Never let
            # an older terminal snapshot silently prove its transient child done.
            reset_sources.update(transient_sources)
        for source_id in reset_sources:
            lifecycle_by_source.pop(source_id, None)

        for source_id, child_status in observations:
            prior = lifecycle_by_source.get(source_id)
            prior_status = prior[0] if prior is not None else None
            if child_status in TERMINAL_CHILD_STATUSES:
                lifecycle_by_source[source_id] = (child_status, index)
                continue
            if prior_status in TERMINAL_CHILD_STATUSES:
                # A passive delayed snapshot cannot reopen a terminal child.
                continue
            lifecycle_by_source[source_id] = (child_status, index)
        if (
            normalized_collaboration_tool in _CHILD_GENERATION_ACTIONS
            and _status(tool) != "completed"
            and normalized_activity
            in {"started", "interacted", "interrupted"}
            and activity_source is not None
        ):
            # A failed start/follow-up can report the new activity alongside a
            # stale terminal state for the same stable child id.  Keep that
            # generation unresolved until a later independent terminal snapshot.
            lifecycle_by_source[activity_source] = ("running", index)

    transient_child_indexes = {
        index
        for status, index in lifecycle_by_source.values()
        if status in TRANSIENT_CHILD_STATUSES
    }
    unresolved_children = {
        *invalid_indexes,
        *transient_child_indexes,
    }
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
                status = _allowlisted_token(
                    raw_status,
                    allowed=(
                        TRANSIENT_CHILD_STATUSES | TERMINAL_CHILD_STATUSES
                    ),
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
    "PromptAssessment",
    "PromptOutcome",
    "classify_prompt_result",
    "has_transient_child_lifecycle",
]
