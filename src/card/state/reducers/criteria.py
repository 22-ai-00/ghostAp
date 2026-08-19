"""Criteria/warning/review-retry sub-reducer for Spec engine."""
from __future__ import annotations

import logging
from dataclasses import replace

from ...events import CardEvent, CardEventType
from ...ui_text import UI_TEXT
from ..button_intent import ButtonIntent
from ..models import ButtonSpec, CardState, CriteriaBlock

logger = logging.getLogger(__name__)


def reduce_criteria(state: CardState, event: CardEvent) -> CardState:
    """Handle CRITERIA_UPDATED / WARNING_UPDATED / REVIEW_RETRY events."""
    # Warnings are shared card chrome, not Spec-only state.  Handle them before
    # the engine extension guard so ordinary programming cards can surface
    # transport degradation without manufacturing an EngineExtState.
    if event.type is CardEventType.WARNING_UPDATED:
        warning = event.payload.get("warning", "")
        explicit_type = event.payload.get("warning_type")
        if explicit_type in {"success", "warning", "error", "info"}:
            warning_type = explicit_type
        elif not warning:
            warning_type = None
        elif warning.startswith("✅"):
            warning_type = "success"
        elif warning.startswith("❌"):
            warning_type = "error"
        elif warning.startswith("ℹ️"):
            warning_type = "info"
        else:
            warning_type = "warning"
        footer = replace(
            state.footer,
            warning_banner=warning or None,
            warning_type=warning_type,
        )
        # Keep-alive is a singleton derived from the current warning.  Strip
        # any prior instance first so repeated TTL warnings update in place,
        # and clearing/replacing the warning cannot leave a stale action.
        buttons_without_keep_alive = tuple(
            button
            for button in state.buttons
            if button.action_id != "ttl_keep_alive"
        )
        buttons = buttons_without_keep_alive
        if warning and event.payload.get("show_keep_alive_btn"):
            minutes = event.payload.get("keep_alive_minutes", 7)
            keep_alive_btn = ButtonSpec(
                text=UI_TEXT["ttl_keep_alive_btn"].format(minutes=minutes),
                action_id="ttl_keep_alive",
                type="primary",
            )
            buttons = (keep_alive_btn,) + buttons_without_keep_alive
        return replace(state, footer=footer, buttons=buttons)

    if state.engine_ext is None:
        logger.warning("reduce_criteria called with engine_ext=None, event=%s", event.type)
        return state
    match event.type:
        case CardEventType.CRITERIA_UPDATED:
            content = event.payload.get("content", "")
            satisfied = event.payload.get("satisfied_count", state.engine_ext.criteria_satisfied)
            total = event.payload.get("total_count", state.engine_ext.criteria_total)
            ext = replace(state.engine_ext,
                          criteria_section=content,
                          criteria_satisfied=satisfied,
                          criteria_total=total)
            # Insert or update the criteria ContentBlock in state.blocks
            # so the render pipeline (atoms.py → renderer.py) can pick it up.
            criteria_block = CriteriaBlock(
                block_id="criteria_section", content=content,
            )
            existing_ids = [b.block_id for b in state.blocks]
            if "criteria_section" in existing_ids:
                blocks = tuple(
                    criteria_block if b.block_id == "criteria_section" else b
                    for b in state.blocks
                )
            else:
                blocks = state.blocks + (criteria_block,)
            return replace(state, engine_ext=ext, blocks=blocks)

        case CardEventType.REVIEW_RETRY:
            attempt = event.payload.get("attempt", 1)
            max_attempts = event.payload.get("max_attempts", 1)
            status = event.payload.get("status", "executing")
            delay_sec = event.payload.get("delay_sec", 0)

            # Build status text
            if status == "waiting":
                status_text = UI_TEXT["card_retry_waiting"].format(
                    delay_sec=int(delay_sec or 0), attempt=attempt, max_attempts=max_attempts)
            elif status == "executing":
                status_text = UI_TEXT["card_retry_executing"].format(
                    attempt=attempt, max_attempts=max_attempts)
            elif status == "exhausted":
                status_text = UI_TEXT["card_retry_exhausted"].format(max_attempts=max_attempts)
            else:
                status_text = UI_TEXT["card_retry_skip"]

            footer = replace(state.footer, status_text=status_text)

            # Add retry action buttons based on status
            buttons: tuple[ButtonSpec, ...] = ()
            if status in ("waiting", "executing"):
                buttons = (
                    ButtonSpec(text=UI_TEXT["card_btn_stop"], action_id=ButtonIntent.SPEC_STOP, type="default"),
                )
            return replace(state, footer=footer, buttons=buttons)

    return state
