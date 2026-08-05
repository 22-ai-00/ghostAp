"""Card event dataclass and factory methods."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from src.utils.text import sanitize_single_line_label

from .types import CardEventType

if TYPE_CHECKING:
    from src.acp.models import ACPEvent

    from .payloads import (
        BlockedPayload,
        CardSplitPayload,
        CompletedPayload,
        CriteriaUpdatedPayload,
        CycleDonePayload,
        CycleStartedPayload,
        FailedPayload,
        ImagePayload,
        PhaseDonePayload,
        PhaseStartedPayload,
        PlanUpdatedPayload,
        ProgressPayload,
        ReasoningBlockPayload,
        ReviewResultUpdatedPayload,
        ReviewRetryPayload,
        SpecPlanUpdatedPayload,
        SpecTasksUpdatedPayload,
        TextBlockPayload,
        ToolDeltaPayload,
        ToolDonePayload,
        ToolFailedPayload,
        ToolModelChangedPayload,
        ToolStartedPayload,
        WarningPayload,
    )


# Whether to run expensive per-element payload validation (only in DEBUG/test mode).
# In production this is False to skip O(n) list-traversal checks in event factories.
# Convention: top-level isinstance() type guards stay unconditional (cheap);
# per-element structural checks (for-loop validators) MUST be wrapped with:
#   if VALIDATE_PAYLOAD:
VALIDATE_PAYLOAD = (
    os.environ.get("DEBUG", "") == "1"
    or "pytest" in sys.modules
)

# Payload type variable — used purely for type-checker narrowing.
# At runtime, payload is always Mapping[str, Any].
P = TypeVar("P", bound=Mapping[str, Any])


@dataclass(frozen=True)
class CardEvent(Generic[P]):
    """Immutable card event dispatched to reducer.

    Generic parameter P narrows the payload type for type-checkers.
    At runtime, payload is always ``Mapping[str, Any]`` — no overhead.
    """
    type: CardEventType
    payload: Mapping[str, Any] = field(default_factory=dict)

    # --- Factory methods ---
    # NOTE: @overload signatures live in .pyi stubs or are guarded by TYPE_CHECKING
    # to avoid runtime overhead. The concrete implementations below serve both roles.

    @classmethod
    def started(cls) -> CardEvent[Mapping[str, Any]]:
        """Signal that the engine session has started.

        Payload: {} (empty)
        Triggered when: Engine begins processing a user request.
        """
        return cls(type=CardEventType.STARTED)

    @classmethod
    def completed(
        cls,
        summary: str = "",
        *,
        duration_seconds: float | None = None,
    ) -> CardEvent[CompletedPayload]:
        """Signal successful completion of the engine session.

        Payload: {summary?: str} — optional completion summary text.
        Triggered when: Engine finishes all work successfully.
        """
        if not isinstance(summary, str):
            raise TypeError(f"summary must be str, got {type(summary).__name__}")
        payload = {}
        if summary:
            payload["summary"] = summary
        if duration_seconds is not None:
            if (
                isinstance(duration_seconds, bool)
                or not isinstance(duration_seconds, (int, float))
                or not math.isfinite(float(duration_seconds))
                or duration_seconds < 0
            ):
                raise ValueError("duration_seconds must be a finite non-negative number")
            payload["duration_seconds"] = float(duration_seconds)
        return cls(type=CardEventType.COMPLETED, payload=payload)

    @classmethod
    def failed(
        cls,
        error: str = "",
        *,
        details: str = "",
        detail_action: Mapping[str, Any] | None = None,
        retry_action: Mapping[str, Any] | None = None,
        duration_seconds: float | None = None,
    ) -> CardEvent[FailedPayload]:
        """Signal that the engine session has failed.

        Payload: {error: str} — error description text.
        Triggered when: Engine encounters an unrecoverable error.
        """
        if not isinstance(error, str):
            raise TypeError(f"error must be str, got {type(error).__name__}")
        payload: dict[str, Any] = {"error": error}
        if details:
            payload["details"] = str(details)
        if detail_action:
            action_payload = dict(detail_action)
            if "diagnostic_token" not in action_payload:
                from src.card.error_diagnostics import register_error_diagnostic

                action_payload = {
                    key: value
                    for key, value in action_payload.items()
                    if key not in {"details", "detail", "stderr", "stdout", "error", "exception", "traceback"}
                }
                action_payload["diagnostic_token"] = register_error_diagnostic(
                    title="错误详情",
                    summary=error,
                    details=str(details or error),
                    chat_id=action_payload.get("chat_id"),
                    origin_message_id=action_payload.get("origin_message_id"),
                    request_id=action_payload.get("request_id"),
                    trace_id=action_payload.get("trace_id"),
                )
            payload["detail_action"] = action_payload
        if retry_action:
            payload["retry_action"] = dict(retry_action)
        if duration_seconds is not None:
            if (
                isinstance(duration_seconds, bool)
                or not isinstance(duration_seconds, (int, float))
                or not math.isfinite(float(duration_seconds))
                or duration_seconds < 0
            ):
                raise ValueError("duration_seconds must be a finite non-negative number")
            payload["duration_seconds"] = float(duration_seconds)
        return cls(type=CardEventType.FAILED, payload=payload)

    @classmethod
    def cancelled(cls, *, reason: str | None = None) -> CardEvent[Mapping[str, Any]]:
        """Signal that the user has cancelled the engine session.

        Payload: {reason?: str} — optional cancellation reason (e.g. 'ttl_expired').
        Triggered when: User explicitly cancels via button or command, or TTL expires.
        """
        payload = {"reason": reason} if reason else {}
        return cls(type=CardEventType.CANCELLED, payload=payload)

    @classmethod
    def archived(
        cls,
        summary: str = "",
        sequence: int = 0,
        new_message_id: str = "",
        bridge_phrase: str | None = None,
        *,
        append_hint: bool = True,
    ) -> CardEvent[Mapping[str, Any]]:
        """Signal that the session has been archived (rotated out by SessionRotator).

        Payload: {summary?: str, sequence?: int, new_message_id?: str,
        append_hint: bool}
        Triggered when: SessionRotator replaces active session with a new one.
        new_message_id: the message_id of the new (replacement) card, for navigation URL.
        append_hint: whether lifecycle reduction may add navigation content blocks.
        """
        payload = {"append_hint": append_hint}
        if summary:
            payload["summary"] = summary
        if sequence:
            payload["sequence"] = sequence
        if new_message_id:
            payload["new_message_id"] = new_message_id
        if bridge_phrase:
            payload["bridge_phrase"] = bridge_phrase
        return cls(type=CardEventType.ARCHIVED, payload=payload)

    @classmethod
    def blocked(cls, reason: str = "") -> CardEvent[BlockedPayload]:
        """Signal that the engine session is blocked and cannot proceed.

        Payload: {reason?: str} — optional reason explaining why the task is blocked.
        Triggered when: Engine encounters a blocking condition (e.g. lock conflict, awaiting external input).
        """
        if not isinstance(reason, str):
            raise TypeError(f"reason must be str, got {type(reason).__name__}")
        payload = {}
        if reason:
            payload["reason"] = reason
        return cls(type=CardEventType.BLOCKED, payload=payload)

    @classmethod
    def text_started(
        cls,
        block_id: str,
        *,
        source_kind: str = "main",
        source_sequence: str | None = None,
        source_label: str | None = None,
        source_ref: str = "main",
    ) -> CardEvent[TextBlockPayload]:
        """Signal the start of a new text content block.

        Payload: {block_id, source_kind?, source_sequence?, source_label?,
        source_ref?}. Source attribution is optional and defaults to the main
        Agent so existing engine/static callers remain unchanged.
        Triggered when: Engine begins streaming a new text segment.
        """
        if not block_id:
            raise ValueError("block_id is required for text_started")
        if source_kind not in {"main", "subagent"}:
            raise ValueError("source_kind must be 'main' or 'subagent'")
        payload: TextBlockPayload = {"block_id": block_id}
        if source_kind != "main":
            payload["source_kind"] = "subagent"
        if source_sequence:
            payload["source_sequence"] = source_sequence
        if source_label:
            payload["source_label"] = source_label
        if source_ref != "main":
            payload["source_ref"] = source_ref
        return cls(type=CardEventType.TEXT_STARTED, payload=payload)

    @classmethod
    def text_delta(cls, block_id: str, text: str) -> CardEvent[TextBlockPayload]:
        """Append text content to an active text block.

        Payload: {block_id: str, text: str} — incremental text chunk.
        Triggered when: Engine streams a text chunk for the active block.
        """
        if not block_id:
            raise ValueError("block_id is required for text_delta")
        return cls(type=CardEventType.TEXT_DELTA, payload={"block_id": block_id, "text": text})

    @classmethod
    def text_done(cls, block_id: str) -> CardEvent[TextBlockPayload]:
        """Signal that a text block has finished streaming.

        Payload: {block_id: str}
        Triggered when: Engine completes the current text segment.
        """
        if not block_id:
            raise ValueError("block_id is required for text_done")
        return cls(type=CardEventType.TEXT_DONE, payload={"block_id": block_id})

    @classmethod
    def reasoning_started(cls, block_id: str) -> CardEvent[ReasoningBlockPayload]:
        """Signal the start of a reasoning/thinking block.

        Payload: {block_id: str}
        Triggered when: Model enters thinking/reasoning mode.
        """
        if not block_id:
            raise ValueError("block_id is required for reasoning_started")
        return cls(type=CardEventType.REASONING_STARTED, payload={"block_id": block_id})

    @classmethod
    def reasoning_delta(cls, block_id: str, text: str) -> CardEvent[ReasoningBlockPayload]:
        """Append content to an active reasoning block.

        Payload: {block_id: str, text: str}
        Triggered when: Model streams a reasoning chunk.
        """
        if not block_id:
            raise ValueError("block_id is required for reasoning_delta")
        return cls(type=CardEventType.REASONING_DELTA, payload={"block_id": block_id, "text": text})

    @classmethod
    def reasoning_done(cls, block_id: str) -> CardEvent[ReasoningBlockPayload]:
        """Signal that a reasoning block has finished.

        Payload: {block_id: str}
        Triggered when: Model exits thinking/reasoning mode.
        """
        if not block_id:
            raise ValueError("block_id is required for reasoning_done")
        return cls(type=CardEventType.REASONING_DONE, payload={"block_id": block_id})

    @classmethod
    def tool_started(cls, block_id: str, tool_name: str, tool_input: str = "") -> CardEvent[ToolStartedPayload]:
        """Signal the start of a tool call execution.

        Payload: {block_id: str, tool_name: str, tool_input: str}
        Triggered when: Agent invokes a tool (file read, shell command, etc.).
        """
        if not block_id:
            raise ValueError("block_id is required for tool_started")
        if not isinstance(tool_name, str):
            raise TypeError(f"tool_name must be str, got {type(tool_name).__name__}")
        return cls(type=CardEventType.TOOL_STARTED, payload={
            "block_id": block_id, "tool_name": tool_name, "tool_input": tool_input,
        })

    @classmethod
    def tool_delta(cls, block_id: str, content: str) -> CardEvent[ToolDeltaPayload]:
        """Append streaming output to an active tool call block.

        Payload: {block_id: str, content: str}
        Triggered when: Tool produces incremental output.
        """
        if not block_id:
            raise ValueError("block_id is required for tool_delta")
        return cls(type=CardEventType.TOOL_DELTA, payload={"block_id": block_id, "content": content})

    @classmethod
    def tool_done(cls, block_id: str, tool_output: str = "", tool_summary: str = "") -> CardEvent[ToolDonePayload]:
        """Signal successful completion of a tool call.

        Payload: {block_id: str, tool_output: str, tool_summary: str}
        Triggered when: Tool finishes execution successfully.
        """
        if not block_id:
            raise ValueError("block_id is required for tool_done")
        return cls(type=CardEventType.TOOL_DONE, payload={
            "block_id": block_id, "tool_output": tool_output, "tool_summary": tool_summary,
        })

    @classmethod
    def tool_failed(cls, block_id: str, error: str = "") -> CardEvent[ToolFailedPayload]:
        """Signal that a tool call has failed.

        Payload: {block_id: str, error: str}
        Triggered when: Tool execution encounters an error.
        """
        if not block_id:
            raise ValueError("block_id is required for tool_failed")
        return cls(type=CardEventType.TOOL_FAILED, payload={"block_id": block_id, "error": error})

    @classmethod
    def image_added(
        cls,
        image_id: str,
        image_key: str,
        alt: str = "任务图片",
    ) -> CardEvent[ImagePayload]:
        """Add a successfully uploaded image artifact to the card."""
        if not image_id:
            raise ValueError("image_id is required for image_added")
        if not image_key:
            raise ValueError("image_key is required for image_added")
        return cls(
            type=CardEventType.IMAGE_ADDED,
            payload={
                "image_id": image_id,
                "image_key": image_key,
                "alt": sanitize_single_line_label(
                    alt,
                    fallback="任务图片",
                    max_chars=120,
                ),
            },
        )

    @classmethod
    def image_failed(
        cls,
        image_id: str,
        alt: str = "任务图片",
    ) -> CardEvent[ImagePayload]:
        """Record a non-fatal image publication failure."""
        if not image_id:
            raise ValueError("image_id is required for image_failed")
        return cls(
            type=CardEventType.IMAGE_FAILED,
            payload={
                "image_id": image_id,
                "alt": sanitize_single_line_label(
                    alt,
                    fallback="任务图片",
                    max_chars=120,
                ),
            },
        )

    @classmethod
    def plan_updated(cls, content: str) -> CardEvent[PlanUpdatedPayload]:
        """Update the plan/checklist display in the card.

        Payload: {content: str} — formatted plan text (markdown checklist).
        Triggered when: Agent updates its execution plan.
        """
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__}")
        return cls(type=CardEventType.PLAN_UPDATED, payload={"content": content})

    @classmethod
    def tool_model_changed(
        cls,
        tool_name: str | None = None,
        model_name: str | None = None,
        *,
        unit_label: str | None = None,
        live_ticker_frame: str | None = None,
        subagents: tuple[dict, ...] | None = None,
    ) -> CardEvent[ToolModelChangedPayload]:
        """Update metadata displayed in the card header/footer.

        Payload: {tool_name?: str | None, model_name?: str | None, unit_label?: str | None}
        Triggered when: User switches tool/model or a task card gets a better label.
        """
        payload = {}
        if tool_name is not None:
            payload["tool_name"] = tool_name
        if model_name is not None:
            payload["model_name"] = model_name
        if unit_label is not None:
            payload["unit_label"] = unit_label
        if live_ticker_frame is not None:
            payload["live_ticker_frame"] = live_ticker_frame
        if subagents is not None:
            payload["subagents"] = subagents
        return cls(type=CardEventType.TOOL_MODEL_CHANGED, payload=payload)

    @classmethod
    def progress_updated(cls, current: int, total: int, label: str = "") -> CardEvent[ProgressPayload]:
        """Update the progress indicator in the card footer.

        Payload: {current: int, total: int, label?: str}
        Triggered when: Engine completes a discrete step (tool call, file operation, etc.).
        """
        if not isinstance(current, int):
            raise TypeError(f"current must be int, got {type(current).__name__}")
        if not isinstance(total, int):
            raise TypeError(f"total must be int, got {type(total).__name__}")
        return cls(type=CardEventType.PROGRESS_UPDATED, payload={
            "current": current, "total": total, "label": label,
        })

    @classmethod
    def card_split(cls, reason: str, hint: str = "", bridge_phrase: str | None = None) -> CardEvent[CardSplitPayload]:
        """Request a semantic split into a continuation card.

        Payload: {reason: str, hint: str, bridge_phrase?: str}
        Triggered when an engine reaches a semantic boundary (task/round/cycle).
        """
        if not isinstance(reason, str):
            raise TypeError(f"reason must be str, got {type(reason).__name__}")
        if not reason:
            raise ValueError("reason is required for card_split")
        if not isinstance(hint, str):
            raise TypeError(f"hint must be str, got {type(hint).__name__}")
        payload = {"reason": reason, "hint": hint}
        if bridge_phrase is not None:
            if not isinstance(bridge_phrase, str):
                raise TypeError(f"bridge_phrase must be str, got {type(bridge_phrase).__name__}")
            payload["bridge_phrase"] = bridge_phrase
        return cls(type=CardEventType.CARD_SPLIT, payload=payload)

    @classmethod
    def cycle_started(cls, cycle_num: int, max_cycles: int) -> CardEvent[CycleStartedPayload]:
        """Signal the start of an engine iteration cycle.

        Payload: {cycle_num: int, max_cycles: int}
        Triggered when: Spec engine begins a new iteration cycle.
        """
        if not isinstance(cycle_num, int):
            raise TypeError(f"cycle_num must be int, got {type(cycle_num).__name__}")
        if not isinstance(max_cycles, int):
            raise TypeError(f"max_cycles must be int, got {type(max_cycles).__name__}")
        return cls(type=CardEventType.CYCLE_STARTED, payload={
            "cycle_num": cycle_num, "max_cycles": max_cycles,
        })

    @classmethod
    def cycle_done(cls, cycle_num: int, status: str = "completed") -> CardEvent[CycleDonePayload]:
        """Signal the completion of an engine iteration cycle.

        Payload: {cycle_num: int, status: str}
        Triggered when: Spec engine finishes one cycle (regardless of outcome).
        """
        if not isinstance(cycle_num, int):
            raise TypeError(f"cycle_num must be int, got {type(cycle_num).__name__}")
        if not isinstance(status, str):
            raise TypeError(f"status must be str, got {type(status).__name__}")
        return cls(type=CardEventType.CYCLE_DONE, payload={
            "cycle_num": cycle_num, "status": status,
        })

    @classmethod
    def phase_started(
        cls,
        cycle_num: int,
        phase: str,
        *,
        subtitle: str | None = None,
        content: str | None = None,
    ) -> CardEvent[PhaseStartedPayload]:
        """Signal that a named phase has begun within a cycle.

        Payload: {cycle_num: int, phase: str, subtitle?: str}
        Triggered when: Engine enters a phase (e.g. "planning", "coding", "review").
        If subtitle is provided, the header subtitle is updated to this value.
        """
        if not isinstance(cycle_num, int):
            raise TypeError(f"cycle_num must be int, got {type(cycle_num).__name__}")
        if not phase:
            raise ValueError("phase is required for phase_started")
        payload: dict = {"cycle_num": cycle_num, "phase": phase}
        if subtitle is not None:
            payload["subtitle"] = subtitle
        if content is not None:
            payload["content"] = content
        return cls(type=CardEventType.PHASE_STARTED, payload=payload)

    @classmethod
    def phase_done(
        cls,
        cycle_num: int,
        phase: str,
        output: str = "",
        *,
        subtitle: str | None = None,
    ) -> CardEvent[PhaseDonePayload]:
        """Signal that a named phase has completed.

        Payload: {cycle_num: int, phase: str, output: str}
        Triggered when: Engine completes a phase; output contains phase summary.
        """
        if not isinstance(cycle_num, int):
            raise TypeError(f"cycle_num must be int, got {type(cycle_num).__name__}")
        if not phase:
            raise ValueError("phase is required for phase_done")
        payload: dict = {
            "cycle_num": cycle_num, "phase": phase, "output": output,
        }
        if subtitle is not None:
            payload["subtitle"] = subtitle
        return cls(type=CardEventType.PHASE_DONE, payload=payload)

    @classmethod
    def review_result_updated(cls, cycle_num: int, roles: list[dict]) -> CardEvent[ReviewResultUpdatedPayload]:
        """Render Spec multi-role review details as one panel per role."""
        if not isinstance(cycle_num, int):
            raise TypeError(f"cycle_num must be int, got {type(cycle_num).__name__}")
        if not isinstance(roles, list):
            raise TypeError(f"roles must be list, got {type(roles).__name__}")
        return cls(
            type=CardEventType.REVIEW_RESULT_UPDATED,
            payload={"cycle_num": cycle_num, "roles": [dict(role) for role in roles]},
        )

    @classmethod
    def spec_plan_updated(cls, cycle_num: int, plan: Mapping[str, Any]) -> CardEvent[SpecPlanUpdatedPayload]:
        """Render Spec PLAN output as a structured plan panel."""
        if not isinstance(cycle_num, int):
            raise TypeError(f"cycle_num must be int, got {type(cycle_num).__name__}")
        if not isinstance(plan, Mapping):
            raise TypeError(f"plan must be mapping, got {type(plan).__name__}")
        return cls(
            type=CardEventType.SPEC_PLAN_UPDATED,
            payload={"cycle_num": cycle_num, "plan": dict(plan)},
        )

    @classmethod
    def spec_tasks_updated(cls, cycle_num: int, tasks: list[dict]) -> CardEvent[SpecTasksUpdatedPayload]:
        """Render Spec TASK output as one complete panel per task."""
        if not isinstance(cycle_num, int):
            raise TypeError(f"cycle_num must be int, got {type(cycle_num).__name__}")
        if not isinstance(tasks, list):
            raise TypeError(f"tasks must be list, got {type(tasks).__name__}")
        return cls(
            type=CardEventType.SPEC_TASKS_UPDATED,
            payload={"cycle_num": cycle_num, "tasks": [dict(task) for task in tasks if isinstance(task, Mapping)]},
        )

    @classmethod
    def review_retry(cls, cycle_num: int, attempt: int, max_attempts: int, status: str = "executing", delay_sec: float = 0) -> CardEvent[ReviewRetryPayload]:
        """Signal a review retry attempt within the Spec engine.

        Payload: {cycle_num, attempt, max_attempts, status, delay_sec}
        Status values: "waiting" (delay), "executing" (in-flight), "exhausted" (gave up).
        Triggered when: Spec engine retries criteria review after failure.
        """
        if not isinstance(cycle_num, int):
            raise TypeError(f"cycle_num must be int, got {type(cycle_num).__name__}")
        if not isinstance(attempt, int):
            raise TypeError(f"attempt must be int, got {type(attempt).__name__}")
        if not isinstance(max_attempts, int):
            raise TypeError(f"max_attempts must be int, got {type(max_attempts).__name__}")
        return cls(type=CardEventType.REVIEW_RETRY, payload={
            "cycle_num": cycle_num, "attempt": attempt, "max_attempts": max_attempts,
            "status": status, "delay_sec": delay_sec,
        })

    @classmethod
    def criteria_updated(cls, content: str, satisfied_count: int = 0, total_count: int = 0) -> CardEvent[CriteriaUpdatedPayload]:
        """Update the acceptance criteria display in the card.

        Payload: {content: str, satisfied_count: int, total_count: int}
        Triggered when: Spec engine re-evaluates criteria satisfaction.
        """
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__}")
        if not isinstance(satisfied_count, int):
            raise TypeError(f"satisfied_count must be int, got {type(satisfied_count).__name__}")
        if not isinstance(total_count, int):
            raise TypeError(f"total_count must be int, got {type(total_count).__name__}")
        return cls(type=CardEventType.CRITERIA_UPDATED, payload={
            "content": content, "satisfied_count": satisfied_count, "total_count": total_count,
        })

    @classmethod
    def warning_updated(cls, warning: str, *, show_keep_alive_btn: bool = False, keep_alive_minutes: int = 0) -> CardEvent[WarningPayload]:
        """Update or clear the warning banner in the card footer.

        Payload: {warning: str, show_keep_alive_btn: bool, keep_alive_minutes: int} — empty string clears the banner.
        Triggered when: Engine detects approaching limits (token, time, cost).
        """
        return cls(type=CardEventType.WARNING_UPDATED, payload={"warning": warning, "show_keep_alive_btn": show_keep_alive_btn, "keep_alive_minutes": keep_alive_minutes})

    # --- UI control ---

    @classmethod
    def mode_toggled(cls, compact: bool) -> CardEvent[Mapping[str, Any]]:
        """Toggle card display mode (full ↔ compact).

        Args:
            compact: Target state — True means switch to compact view.
        """
        return cls(type=CardEventType.MODE_TOGGLED, payload={"compact": compact})

    @classmethod
    def stop_escalated(cls) -> CardEvent[Mapping[str, Any]]:
        """Escalate a pending stop to force-stop after timeout."""
        return cls(type=CardEventType.STOP_ESCALATED)

    @classmethod
    def from_acp(cls, acp_event: "ACPEvent") -> CardEvent[Mapping[str, Any]]:
        """Convert an ACPEvent to a CardEvent.

        Delegates to :func:`src.card.events.acp_adapter.card_event_from_acp`.
        """
        from .acp_adapter import card_event_from_acp

        return card_event_from_acp(acp_event)
