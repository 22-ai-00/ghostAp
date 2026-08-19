"""Lifecycle sub-reducer."""
from __future__ import annotations

from dataclasses import replace
from typing import cast

from src.card.themes import TERMINAL_TEMPLATES

from ...engine_meta import ENGINE_CMD_MAP
from ...events import CardEvent, CardEventType
from ...events.payloads import CompletedPayload, FailedPayload
from ...ui_text import UI_TEXT
from ..button_intent import ButtonIntent
from ..models import ButtonSpec, CardState, EngineExtState, FooterState, HeaderState, TextBlock
from ._shared import build_header

# Engine type → user-facing command — imported from engine_meta
_ENGINE_CMD = ENGINE_CMD_MAP

# Stop button action IDs per engine type
_STOP_ACTIONS: dict[str, str] = {
    "deep": ButtonIntent.ENGINE_STOP,
    "spec": ButtonIntent.ENGINE_STOP,
}

# Engine-specific retry button CTA text (differentiated by engine_type)
_RETRY_CTA_TEXT: dict[str, str] = {
    "deep": "🔁 重新执行",
    "spec": "🔁 重新 Build",
}


def _compute_duration(
    state: CardState,
    now: float | None,
    authoritative_duration: object = None,
) -> float | None:
    """Compute terminal total duration from the card session start to now.

    Args:
        state: Current card state.
        now: Monotonic timestamp injected by CardSession (via event payload '_now').
    """
    if (
        not isinstance(authoritative_duration, bool)
        and isinstance(authoritative_duration, (int, float))
    ):
        return max(0.0, float(authoritative_duration))

    started_at = state.metadata.session_started_at
    if started_at is not None and now is not None:
        return max(0.0, float(now) - float(started_at))

    started_at = state.footer.progress_started_at
    if started_at is None or now is None:
        return None
    return max(0.0, float(now) - float(started_at))


def _format_unified_error_block(
    error: str,
    *,
    engine_cmd: str,
    has_details: bool,
) -> str:
    """Unified user-visible error card content for engine/session failures."""
    summary = (error or "").strip() or UI_TEXT["card_lifecycle_error_fallback"].format(engine_cmd=engine_cmd)
    current_status = (
        "当前操作已停止；可查看脱敏诊断并按提示重新发起。\n\n"
        f"{UI_TEXT['card_lifecycle_details_collapsed']}"
        if has_details
        else "当前操作已停止；可点击“查看状态”确认当前会话状态后重新发起。"
    )
    return (
        "❌ **错误摘要**\n"
        f"{summary}\n\n"
        "**当前状态**\n"
        f"{current_status}"
    )


def _with_failed_title(header: HeaderState) -> HeaderState:
    """Make terminal error cards explicit in the title, while preserving context."""
    title = header.title or UI_TEXT["system_error_prompt_title"]
    if "错误" not in title:
        title = f"❌ 错误 · {title}"
    return replace(header, title=title)


def _compute_frozen_elapsed(state: CardState, now: float | None) -> float | None:
    """Compute final elapsed for a frozen/archived card with monotonic clocks."""
    duration = _compute_duration(state, now)
    if duration is not None:
        return duration
    started_at = state.metadata.session_started_at
    if started_at is not None and now is not None:
        return max(0.0, float(now) - float(started_at))
    return state.footer.duration_seconds


def _archived_status_text(payload: dict) -> str:
    bridge = str(payload.get("bridge_phrase") or "").strip()
    if bridge:
        return f"本卡已停止更新 · {bridge}"
    sequence = payload.get("sequence")
    if sequence:
        try:
            return f"本卡已停止更新 · 续接 #{int(sequence) + 1} ↓"
        except (TypeError, ValueError):
            return "本卡已停止更新"
    return UI_TEXT["card_lifecycle_archived"]


def _running_buttons(engine_type: str | None, compact: bool = False) -> tuple[ButtonSpec, ...]:
    """Build buttons for running state (stop button + mode toggle).

    Mode toggle button shows target mode with current state context:
    - In compact mode: button says "切换完整模式" (switch TO full)
    - In full mode: button says "切换精简模式" (switch TO compact)
    """
    if not engine_type or engine_type not in _STOP_ACTIONS:
        return ()
    mode_btn = ButtonSpec(
        text=UI_TEXT["card_btn_mode_full"] if compact else UI_TEXT["card_btn_mode_compact"],
        action_id=ButtonIntent.MODE_FULL if compact else ButtonIntent.MODE_COMPACT,
        type="default",
    )
    stop_btn = ButtonSpec(
        text=UI_TEXT["card_btn_stop"],
        action_id=_STOP_ACTIONS[engine_type],
        type="default",
        confirm=UI_TEXT["card_btn_stop_confirm"],
    )
    return (mode_btn, stop_btn)


def _stopping_buttons(engine_type: str | None) -> tuple[ButtonSpec, ...]:
    """Build disabled stop button for STOPPING intermediate state."""
    if not engine_type or engine_type not in _STOP_ACTIONS:
        return ()
    return (ButtonSpec(
        text=UI_TEXT["card_btn_stop"],
        action_id=_STOP_ACTIONS[engine_type],
        type="default",
        disabled=True,
        disabled_text=UI_TEXT.get("card_btn_stopping", "正在停止…"),
    ),)


def reduce_lifecycle(state: CardState, event: CardEvent) -> CardState:
    """Handle lifecycle events: STARTED, COMPLETED, FAILED, CANCELLED, PAUSED, RESUMED."""
    payload = event.payload or {}
    now = payload.get("_now") if payload else None
    match event.type:
        case CardEventType.STARTED:
            header = build_header(state.metadata, "running")
            return replace(state, terminal="running", header=header,
                           footer=FooterState(status="thinking", status_text=UI_TEXT["card_lifecycle_thinking"]),
                           buttons=_running_buttons(state.metadata.engine_type, state.metadata.compact))

        case CardEventType.STOPPING:
            # Intermediate state: user clicked stop, awaiting engine acknowledgement
            # Short-circuit if already terminal (COMPLETED/FAILED/CANCELLED arrived first)
            if state.terminal and state.terminal != "running":
                return state
            # Show stopping banner in card body for mobile visibility
            _stopping_status = UI_TEXT["card_lifecycle_stopping"]
            _escalation_hint = UI_TEXT["card_stop_escalation_countdown"]
            footer = replace(
                state.footer,
                status_text=f"{_stopping_status}\n{_escalation_hint}",
                warning_banner=_stopping_status,
                warning_type="info",
            )
            return replace(state, buttons=_stopping_buttons(state.metadata.engine_type), footer=footer)

        case CardEventType.COMPLETED:
            # Idempotency guard: prevent duplicate terminal transitions
            if state.terminal and state.terminal != "running":
                return state
            # For engines that own the header, preserve it and only change color
            if state.header.header_source == "engine" and state.header.title:
                header = replace(state.header, template=TERMINAL_TEMPLATES.get("completed", "green"))
            else:
                header = build_header(state.metadata, "completed")
            # Append summary content block if provided
            payload_c = cast(CompletedPayload, payload)
            summary = payload_c.get("summary")
            blocks = state.blocks
            if summary:
                blocks = blocks + (TextBlock(
                    block_id="_summary", content=summary
                ),)
            engine_cmd = _ENGINE_CMD.get(state.metadata.engine_type or "", "").lstrip("/")
            cta_text = UI_TEXT["card_completed_cta_fmt"].format(engine_cmd=engine_cmd) if engine_cmd else ""
            detail_action = payload_c.get("detail_action")
            detail_action_id = (
                str(detail_action.get("action") or "")
                if isinstance(detail_action, dict)
                else ""
            )
            buttons = (
                ButtonSpec(
                    text=UI_TEXT["card_lifecycle_show_details"],
                    action_id=detail_action_id,
                    value=dict(detail_action),
                ),
            ) if detail_action_id else ()
            warning = str(payload_c.get("warning") or "").strip()
            return replace(state, terminal="completed", terminal_reason="completed",
                           header=header, footer=FooterState(
                               progress_pct=100,
                               duration_seconds=_compute_duration(
                                   state,
                                   now,
                                   payload_c.get("duration_seconds"),
                               ),
                               status_text=cta_text,
                               warning_banner=warning or None,
                               warning_type="warning" if warning else None,
                               persistent_warning=bool(warning),
                           ),
                           buttons=buttons, blocks=blocks)

        case CardEventType.FAILED:
            # Idempotency guard: prevent duplicate terminal transitions
            if state.terminal and state.terminal != "running":
                return state
            # For engines that own the header, preserve it and only change color
            if state.header.header_source == "engine" and state.header.title:
                header = replace(state.header, template=TERMINAL_TEMPLATES.get("failed", "red"))
            else:
                header = build_header(state.metadata, "failed")
            header = _with_failed_title(header)
            # Insert error message as visible content block
            payload_f = cast(FailedPayload, payload)
            error = payload_f.get("error", "")
            blocks = state.blocks
            engine_cmd = _ENGINE_CMD.get(state.metadata.engine_type or "", UI_TEXT["card_session_fallback_cmd"])
            if not error:
                # Select engine-specific error fallback text
                _fallback_key = {
                    "spec": "card_lifecycle_error_fallback_spec",
                }.get(state.metadata.engine_type or "", "card_lifecycle_error_fallback")
                error = UI_TEXT[_fallback_key].format(engine_cmd=engine_cmd)
            detail_action = payload_f.get("detail_action") if isinstance(payload_f, dict) else None
            detail_action_id = None
            if isinstance(detail_action, dict):
                detail_action_id = str(detail_action.get("action") or "") or None
            error_text = _format_unified_error_block(
                error,
                engine_cmd=engine_cmd,
                has_details=bool(detail_action_id),
            )
            blocks = blocks + (TextBlock(
                block_id="_error", content=error_text
            ),)
            # Inject retry button based on engine type with differentiated CTA text
            buttons_list: list[ButtonSpec] = []
            engine_type = state.metadata.engine_type
            retry_action = payload_f.get("retry_action") if isinstance(payload_f, dict) else None
            retry_action_id = None
            if isinstance(retry_action, dict):
                retry_action_id = str(retry_action.get("action") or "") or None
            if retry_action_id:
                retry_text = _RETRY_CTA_TEXT.get(engine_type, UI_TEXT["card_lifecycle_retry_failed"])
                buttons_list.append(ButtonSpec(
                    text=retry_text,
                    action_id=retry_action_id,
                    type="primary",
                    confirm=UI_TEXT["card_btn_confirm_retry_body"],
                    value=dict(retry_action) if isinstance(retry_action, dict) else None,
                ))
            if detail_action_id:
                buttons_list.append(ButtonSpec(
                    text=UI_TEXT["card_lifecycle_show_details"],
                    action_id=detail_action_id,
                    value=dict(detail_action) if isinstance(detail_action, dict) else None,
                ))
            else:
                buttons_list.append(ButtonSpec(
                    text=UI_TEXT.get("card_btn_show_status", "📋 查看状态"),
                    action_id=ButtonIntent.SHOW_STATUS,
                ))
            buttons = tuple(buttons_list)
            return replace(state, terminal="failed", terminal_reason="failed",
                           header=header, footer=FooterState(duration_seconds=_compute_duration(
                               state,
                               now,
                               payload_f.get("duration_seconds"),
                           )),
                           buttons=buttons, blocks=blocks)

        case CardEventType.CANCELLED:
            # Idempotency guard: prevent duplicate terminal transitions
            if state.terminal and state.terminal != "running":
                return state
            header = build_header(state.metadata, "cancelled")
            blocks = state.blocks + (TextBlock(
                block_id="_cancelled", content=UI_TEXT["card_lifecycle_cancelled_block"]
            ),)
            # Determine terminal reason: ttl_expired if specified in payload, else cancelled
            reason = payload.get("reason", "cancelled") if payload else "cancelled"
            # A normal cancellation supersedes transient running-state
            # warnings (including COT cutover's "task continues" banner).
            # TTL closure and explicitly persistent warnings remain visible.
            preserve_warning = reason == "ttl_expired" or (
                payload.get("persistent_warning") is True
            )
            warning_banner = (
                state.footer.warning_banner if preserve_warning else None
            )
            warning_type = state.footer.warning_type if warning_banner else None
            footer = FooterState(
                status_text=UI_TEXT["card_lifecycle_cancelled_status"],
                warning_banner=warning_banner,
                warning_type=warning_type,
                persistent_warning=bool(warning_banner),
                duration_seconds=_compute_duration(state, now),
            )
            cancel_buttons_list: list[ButtonSpec] = []
            # For TTL-expired cancellations, add a unified "show status" button
            if reason == "ttl_expired":
                cancel_buttons_list.append(ButtonSpec(
                    text=UI_TEXT.get("card_btn_show_status", "📋 查看状态"),
                    action_id=ButtonIntent.SHOW_STATUS,
                    type="default",
                ))
            cancel_buttons = tuple(cancel_buttons_list)
            return replace(state, terminal="cancelled", terminal_reason=reason,
                           header=header, footer=footer, buttons=cancel_buttons, blocks=blocks)

        case CardEventType.ARCHIVED:
            frozen_elapsed = _compute_frozen_elapsed(state, now)
            frozen_metadata = replace(
                state.metadata,
                frozen=True,
                frozen_total_elapsed=frozen_elapsed,
                final_state_for_freeze=state,
                bridge_phrase=None,
            )
            header = build_header(frozen_metadata, "archived")
            # Append sequence number to title if provided (e.g., "#1 已归档")
            sequence = (event.payload or {}).get("sequence", 0)
            if sequence:
                header = HeaderState(
                    title=f"{header.title} (#{sequence} 已归档)",
                    subtitle=header.subtitle,
                    template=header.template,
                )
            # Capacity rollover can archive a card that is already exactly at
            # the block limit.  In that path the frozen header/footer/button
            # provide navigation without mutating the preserved content.
            append_hint = bool((event.payload or {}).get("append_hint", True))
            blocks = state.blocks
            if append_hint:
                blocks = blocks + (TextBlock(
                    block_id="_archived_hint", content=UI_TEXT["card_lifecycle_archived_hint"]
                ),)
            footer = FooterState(
                status="idle",
                status_text=_archived_status_text(event.payload or {}),
                duration_seconds=_compute_duration(state, now),
            )
            # Add URL button to navigate to the new card if message_id available
            archived_buttons: tuple[ButtonSpec, ...] = ()
            new_message_id = (event.payload or {}).get("new_message_id", "")
            if new_message_id:
                archived_buttons = (ButtonSpec(
                    text=UI_TEXT.get("card_archived_navigate_btn", "📍 查看最新卡片"),
                    action_id="url_navigate",
                    type="default",
                    url=f"https://applink.feishu.cn/client/message/link?msgId={new_message_id}",
                ),)
            elif append_hint:
                # Graceful degradation: no message_id available, show text hint
                blocks = blocks + (TextBlock(
                    block_id="_archived_nav_hint",
                    content=UI_TEXT.get("card_archived_scroll_hint", "💡 请在对话底部查看最新卡片"),
                ),)
            return replace(state, terminal="archived", terminal_reason="archived",
                           metadata=frozen_metadata, header=header, footer=footer, buttons=archived_buttons, blocks=blocks)

        case CardEventType.PAUSED:
            header = build_header(state.metadata, "paused")
            return replace(state, terminal="paused", header=header,
                           footer=FooterState(status="idle", status_text=UI_TEXT["card_lifecycle_paused"]),
                           buttons=())

        case CardEventType.RESUMED:
            header = build_header(state.metadata, "running")
            return replace(state, terminal="running", header=header,
                           footer=FooterState(status="thinking", status_text=UI_TEXT["card_lifecycle_thinking"]),
                           buttons=_running_buttons(state.metadata.engine_type, state.metadata.compact))

        case CardEventType.BLOCKED:
            header = build_header(state.metadata, "blocked")
            reason = payload.get("reason", "") if payload else ""
            # Store reason in engine_ext for rendering layer to access
            ext = state.engine_ext or EngineExtState()
            ext = replace(ext, blocked_reason=reason or None)
            footer = FooterState(
                status="idle",
                status_text=UI_TEXT.get("card_lifecycle_blocked", "任务已阻塞"),
                duration_seconds=_compute_duration(state, now),
            )
            return replace(state, terminal="blocked", terminal_reason="blocked",
                           header=header,
                           footer=footer, buttons=(), engine_ext=ext)

        case CardEventType.MODE_TOGGLED:
            # Toggle compact mode and rebuild running buttons
            compact = bool(payload.get("compact", not state.metadata.compact))
            new_meta = replace(state.metadata, compact=compact)
            # Only update buttons if currently running (non-terminal or running terminal)
            if state.terminal == "running":
                new_buttons = _running_buttons(new_meta.engine_type, compact)
            else:
                new_buttons = state.buttons
            return replace(state, metadata=new_meta, buttons=new_buttons)

        case CardEventType.STOP_ESCALATED:
            # Escalate: replace buttons with danger force-stop button
            engine_type = state.metadata.engine_type
            if not engine_type or engine_type not in _STOP_ACTIONS:
                return state
            force_buttons = (ButtonSpec(
                text=UI_TEXT.get("card_btn_force_stop", "⚠️ 强制停止"),
                action_id=_STOP_ACTIONS[engine_type],
                type="danger",
                confirm=UI_TEXT.get("card_btn_confirm_stop_title_danger", "强制停止当前任务"),
            ),)
            return replace(state, buttons=force_buttons)

    return state
