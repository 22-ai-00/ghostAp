"""Approval sub-reducer."""
from __future__ import annotations

from dataclasses import replace

from ...events import CardEvent, CardEventType
from ...ui_text import UI_TEXT
from ..button_intent import ButtonIntent
from ..models import ButtonSpec, CardState, FooterState
from ._shared import build_header
from .lifecycle import _compute_duration

# Reuse retry actions for restart button on rejection
_RETRY_ACTIONS: dict[str, str] = {
    "deep": ButtonIntent.DEEP_RESUME,
    "spec": ButtonIntent.SPEC_RESUME,
}


def reduce_approval(state: CardState, event: CardEvent) -> CardState:
    """Handle APPROVAL_REQUESTED / APPROVAL_RESOLVED events."""
    if event.type == CardEventType.APPROVAL_REQUESTED:
        header = build_header(state.metadata, "awaiting_approval")
        tool_name = event.payload.get("tool_name", "")
        event.payload.get("description", "")
        status_text = "⏳ 等待审批"
        if tool_name:
            status_text += f": {tool_name}"
        buttons = (
            ButtonSpec(text="✅ 批准", action_id=ButtonIntent.APPROVE, type="primary"),
            ButtonSpec(text="❌ 拒绝", action_id=ButtonIntent.REJECT, type="danger"),
        )
        return replace(
            state,
            terminal="awaiting_approval",
            header=header,
            footer=replace(state.footer, status="waiting_approval", status_text=status_text),
            buttons=buttons,
        )

    elif event.type == CardEventType.APPROVAL_RESOLVED:
        approved = event.payload.get("approved", False)
        if approved:
            # Resume running state
            header = build_header(state.metadata, "running")
            return replace(
                state,
                terminal="running",
                header=header,
                footer=FooterState(status="thinking", status_text="💭 正在思考..."),
                buttons=(),
            )
        else:
            # Approval rejected → cancelled with restart button
            header = build_header(state.metadata, "cancelled")
            reject_buttons: tuple[ButtonSpec, ...] = ()
            engine_type = state.metadata.engine_type
            if engine_type and engine_type in _RETRY_ACTIONS:
                reject_buttons = (ButtonSpec(
                    text=UI_TEXT["card_lifecycle_restart"],
                    action_id=_RETRY_ACTIONS[engine_type],
                    type="primary",
                ),)
            return replace(
                state,
                terminal="cancelled",
                terminal_reason="rejected",
                header=header,
                footer=FooterState(
                    status_text=UI_TEXT["card_lifecycle_cancelled_status"],
                    duration_seconds=_compute_duration(
                        state,
                        event.payload.get("_now"),
                    ),
                ),
                buttons=reject_buttons,
            )

    return state
