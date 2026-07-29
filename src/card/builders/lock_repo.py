"""Repo-lock card builders.

Card/message builders for repository-level lock conflicts, force-release
confirmation, and reclaim notifications.
"""

from __future__ import annotations

import logging
import os
import time as _time
from typing import Optional

from ..ui_text import UI_TEXT
from .lock_common import (
    MAX_COMMAND_TEXT_LENGTH,
    _build_p2p_multi_url,
    _compute_command_sig,
    format_elapsed_ago,
    sign_command,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_repo_lock_card",
    "build_force_release_confirm_card",
    "build_lock_reclaim_notify_card",
]


def build_repo_lock_card(
    root_path: str,
    locked_since_monotonic: float,
    *,
    is_admin: bool = False,
    command_text: str = "",
    repo_token: str = "",
    last_active_time_monotonic: float = 0.0,
    app_id: str = "",
    is_same_sender: bool = False,
    retry_count: int = 0,
    idle_timeout_seconds: Optional[int] = None,
    chat_id: str = "",
) -> tuple[str, list[dict]]:
    """Build a Markdown prompt + action buttons for a repo lock conflict.

    Returns ``(markdown_content, buttons)`` where *buttons* may contain:
    - a "retry" button (when *command_text* is provided)
    - a "force release" button (when *is_admin* is ``True``)
    - a "go to private chat" button (when *app_id* is provided)

    The card intentionally does **not** expose the holder's chat_id or the
    full filesystem path to avoid cross-group information leakage.
    *repo_token* is an opaque identifier used in the force-release button
    instead of the raw path.
    """
    elapsed = _time.monotonic() - locked_since_monotonic
    duration = format_elapsed_ago(elapsed)

    repo_name = os.path.basename(root_path.rstrip(os.sep)) or root_path

    # Read idle timeout from settings for the auto-release hint
    if idle_timeout_seconds is not None:
        idle_timeout_min = idle_timeout_seconds // 60
    else:
        idle_timeout_min = 5

    # Differentiate "holder actively working" vs "holder idle for N min"
    now = _time.monotonic()
    active_threshold = 60  # seconds
    if last_active_time_monotonic > 0 and (now - last_active_time_monotonic) < active_threshold:
        _active_secs = int(now - last_active_time_monotonic)
        last_active_ago = f"{_active_secs} 秒前" if _active_secs > 0 else "刚刚"
        status_hint = UI_TEXT["repo_lock_hint_active"].format(timeout_min=idle_timeout_min, last_active_ago=last_active_ago)
    else:
        if last_active_time_monotonic > 0:
            idle_secs = now - last_active_time_monotonic
            idle_min = int(idle_secs // 60)
            remaining_min = max(0, idle_timeout_min - idle_min)
            if remaining_min > 0:
                status_hint = UI_TEXT["repo_lock_hint_idle_countdown"].format(idle_min=idle_min, remaining_min=remaining_min)
            else:
                status_hint = UI_TEXT["repo_lock_hint_idle_imminent"].format(idle_min=idle_min)
        else:
            status_hint = UI_TEXT["repo_lock_hint_idle_default"].format(timeout_min=idle_timeout_min)

    # Private chat guidance
    _has_retry_button = bool(command_text and len(command_text) <= MAX_COMMAND_TEXT_LENGTH)
    if app_id:
        p2p_hint = UI_TEXT["repo_lock_p2p_hint_link"].format(app_id=app_id)
        if not _has_retry_button:
            p2p_hint += "\n" + UI_TEXT["repo_lock_p2p_fallback_note"]
    else:
        logger.warning("app_id not configured; lock conflict card will lack P2P deep-link button")
        # Emphasize the hint when the "go to p2p" button is absent (app_id not configured)
        p2p_hint = f"**💬 {UI_TEXT['repo_lock_p2p_hint_plain']}**"

    # Same-sender hint: when the conflicting user is also the lock holder
    same_sender_line = UI_TEXT["repo_lock_same_sender_hint"] if is_same_sender else ""

    # Retry-still-occupied hint (only on retries, not first conflict)
    retry_hint_line = ""
    if retry_count > 0:
        # Build countdown hint for retry feedback
        _countdown = ""
        if last_active_time_monotonic > 0:
            _idle_s = int(now - last_active_time_monotonic)
            _remaining_s = max(0, (idle_timeout_seconds or 300) - _idle_s)
            _remaining_m = _remaining_s // 60
            if _idle_s >= 60:
                _countdown = f"对方已空闲 {_idle_s // 60} 分钟，预计 {_remaining_m} 分钟后自动释放" if _remaining_m > 0 else "锁即将自动释放"
            else:
                _countdown = f"对方最近 {_idle_s} 秒前有操作"
        retry_hint_line = f"\n⚠️ {UI_TEXT['repo_lock_retry_still_occupied'].format(countdown_hint=_countdown)}\n"

    markdown = (
        f"{UI_TEXT['repo_lock_title']}\n\n"
        f"{retry_hint_line}"
        f"{UI_TEXT['repo_lock_occupied'].format(repo_name=repo_name)}\n"
        f"{UI_TEXT['repo_lock_duration_line'].format(duration=duration)}\n\n"
        f"{same_sender_line}"
        f"{status_hint}\n"
        f"{p2p_hint}\n"
        f"{UI_TEXT['repo_lock_retry_hint']}"
        f"\n\n---\n{UI_TEXT['repo_lock_concept_note']}"
    )

    buttons = []
    if command_text and len(command_text) <= MAX_COMMAND_TEXT_LENGTH:
        _next_count = retry_count + 1
        _retry_label = UI_TEXT["repo_lock_btn_retry"]
        # Use v2 signing (nonce+exp+chat_id) when chat_id is available
        _sig = sign_command(command_text, chat_id) if chat_id else _compute_command_sig(command_text)
        buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": _retry_label},
            "type": "primary",
            "value": {
                "action": "retry_command",
                "_t": command_text,
                "_s": _sig,
                "_rc": _next_count,
            },
        })
    elif command_text:
        # Command too long for card action payload; hint user to resend manually
        markdown += f"\n💡 {UI_TEXT['repo_lock_long_command_hint']}"
    if app_id:
        buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": UI_TEXT["repo_lock_btn_go_p2p"]},
            "type": "default",
            "multi_url": _build_p2p_multi_url(app_id),
        })
    if is_admin and repo_token:
        buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": UI_TEXT["repo_lock_btn_force_release"]},
            "type": "danger",
            "value": {
                "action": "force_release_repo_lock",
                "_tk": repo_token,
            },
        })

    return markdown, buttons


def build_force_release_confirm_card(
    repo_token: str,
    repo_name: str,
    *,
    holder_hint: str = "",
    holder_chat_id: str = "",
    confirm_timeout: int | None = None,
) -> tuple[str, list[dict]]:
    """Build a confirmation card before force-releasing a repo lock.

    Returns ``(markdown_content, buttons)`` with confirm/cancel buttons.
    The confirm button embeds ``repo_token`` and a ``timestamp`` for expiry
    validation (same pattern as :func:`build_lock_confirm_card`).

    Parameters
    ----------
    repo_token : str
        Opaque token identifying the repository (from ``RepoLockManager``).
    repo_name : str
        Human-readable repo name (e.g. basename of root_path).
    holder_hint : str
        Optional hint about the current holder (e.g. idle info).
    holder_chat_id : str
        Chat ID of the current lock holder at the time the confirmation card
        is built.  Embedded as ``_hcid`` in the confirm button value so that
        ``handle_confirm_force_release`` can detect holder changes between
        confirmation and execution (race guard).
    """
    import time as _t

    if confirm_timeout is not None:
        _confirm_timeout_sec = confirm_timeout
    else:
        _confirm_timeout_sec = 120
    _confirm_minutes = max(1, _confirm_timeout_sec // 60)

    lines = [
        UI_TEXT["lock_force_release_confirm_title"],
        UI_TEXT["lock_force_release_confirm_occupied"].format(repo_name=repo_name),
        UI_TEXT["lock_force_release_confirm_warn"],
    ]
    if holder_hint:
        lines.append(holder_hint)
    lines.append(UI_TEXT["lock_force_release_confirm_expiry"].format(minutes=_confirm_minutes))

    markdown = "\n".join(lines)
    now = _t.time()
    _confirm_value: dict = {
        "action": "confirm_force_release",
        "_tk": repo_token,
        "_ts": now,
    }
    if holder_chat_id:
        _confirm_value["_hcid"] = holder_chat_id
    buttons = [
        {
            "tag": "button",
            "text": {"tag": "plain_text", "content": UI_TEXT["lock_force_release_btn_confirm"]},
            "type": "danger",
            "value": _confirm_value,
        },
        {
            "tag": "button",
            "text": {"tag": "plain_text", "content": UI_TEXT["lock_btn_cancel"]},
            "type": "default",
            "value": {
                "action": "cancel_force_release",
            },
        },
    ]
    return markdown, buttons


def build_lock_reclaim_notify_card(
    repo_name: str,
    *,
    reason: str = "hard_timeout",
    hard_timeout_seconds: int | None = None,
) -> str:
    """Build a notification message for lock reclaim events.

    Parameters
    ----------
    repo_name : str
        Display name of the repository whose lock was reclaimed.
    reason : str
        ``"hard_timeout"`` — system auto-reclaim after the renewable lease
        stops receiving activity heartbeats.
        ``"force_release"`` — admin force-released the lock from another chat.

    Returns a Markdown string suitable for ``send_message()``.
    """
    if reason == "force_release":
        return UI_TEXT["lock_force_release_notify_holder"].format(repo_name=repo_name)
    if hard_timeout_seconds is not None:
        max_hours = max(1, hard_timeout_seconds // 3600)
    else:
        max_hours = 1
    return UI_TEXT["lock_hard_timeout_reclaim_notify"].format(repo_name=repo_name, max_hours=max_hours)
