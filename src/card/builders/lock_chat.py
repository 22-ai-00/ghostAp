"""Chat-lock card builders.

Card/message builders for chat-level (group) lock intercepts, success
notifications, confirmation flows, and help sections.
"""

from __future__ import annotations

import logging
import time as _t_wall
from typing import Optional

from ..ui_text import UI_TEXT
from .lock_common import (
    _build_p2p_multi_url,
    _compute_command_sig,
    format_undo_window,
    sign_command,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_chat_lock_card",
    "build_lock_success_card",
    "build_lock_help_body",
]


def build_chat_lock_card(
    locked_by: Optional[str] = None,
    locked_by_name: str = "",
    admin_name: str = "",
    *,
    app_id: str = "",
    locked_at_wall: Optional[float] = None,
    max_duration_seconds: Optional[int] = None,
    allowed_commands_display: Optional[str] = None,
    chat_id: str = "",
) -> tuple[str, list[dict]]:
    """Build a Markdown prompt + action buttons for a chat-level lock.

    Returns ``(markdown_content, buttons)`` where *buttons* may contain:
    - a "查看状态" button (action=retry_command, command_text='/status')
    - a "去私聊" jump button (when *app_id* is provided)

    Display priority for the locker identity:
    1. *locked_by_name* (human-readable display name)
    2. Feishu ``<at>`` mention via *locked_by* open_id
    3. Truncated *locked_by* ID (fallback)
    """
    locker_line = ""
    if locked_by_name:
        locker_line = UI_TEXT["chat_lock_locker_name"].format(name=locked_by_name)
    elif locked_by:
        # Never expose raw open_id to non-admin users; show generic label.
        locker_line = UI_TEXT["chat_lock_locker_id"].format(safe_id=UI_TEXT.get("ws_fallback_admin_name", "Bot 管理员"))

    # Locked-at time line
    locked_at_line = ""
    if locked_at_wall is not None:
        from datetime import datetime
        try:
            _dt = datetime.fromtimestamp(locked_at_wall)
            if _dt.date() == datetime.now().date():
                locked_at_line = UI_TEXT["chat_lock_locked_at"].format(time=_dt.strftime("%H:%M"))
            else:
                locked_at_line = UI_TEXT["chat_lock_locked_at"].format(time=_dt.strftime("%m-%d %H:%M"))
        except Exception:
            logger.debug("Failed to format locked_at_wall timestamp", exc_info=True)

    # Build contact line: provide actionable path for non-admin users
    if admin_name:
        contact_line = UI_TEXT["chat_lock_contact_named"].format(name=admin_name)
    else:
        contact_line = UI_TEXT["chat_lock_contact_admin"]
    # When locked_by_name is empty, suppress the separate locker line —
    # the contact line alone is sufficient and avoids exposing raw IDs.
    if not locked_by_name:
        locker_line = ""

    # Show a concise list of representative commands dynamically from
    # the domain layer (ChatLockManager.get_allowed_commands_display).
    if allowed_commands_display is not None:
        cmd_list = allowed_commands_display
    else:
        try:
            from ...chat_lock import ChatLockManager
            cmd_list = ChatLockManager.get_allowed_commands_display()
        except Exception:
            cmd_list = "`/help` `/status`"

    # Compute auto-unlock remaining time
    auto_unlock_line = ""
    if locked_at_wall is not None:
        try:
            from .lock_common import format_friendly_duration
            max_dur = max_duration_seconds if max_duration_seconds is not None else 86400
            remaining = max_dur - (_t_wall.time() - locked_at_wall)
            if remaining > 60:
                _time_str = format_friendly_duration(remaining)
                auto_unlock_line = f"\n{UI_TEXT['chat_lock_auto_unlock_hint'].format(time=_time_str)}\n"
        except Exception:
            logger.debug("Failed to compute auto-unlock remaining time", exc_info=True)

    markdown = (
        f"{UI_TEXT['chat_lock_card_title']}\n\n"
        f"{UI_TEXT['chat_lock_card_desc']}\n"
        f"{locker_line}"
        f"{locked_at_line}"
        f"{contact_line}"
        f"{UI_TEXT['chat_lock_allowed_cmds'].format(cmd_list=cmd_list)}"
        f"{auto_unlock_line}"
        f"\n\n---\n{UI_TEXT['chat_lock_concept_note']}"
        f"\n{UI_TEXT['chat_lock_p2p_guide']}"
    )

    _status_sig = sign_command("/status", chat_id) if chat_id else _compute_command_sig("/status")
    _status_val = {
        "action": "retry_command",
        "_t": "/status",
        "_s": _status_sig,
    }
    buttons: list[dict] = [
        {
            "tag": "button",
            "text": {"tag": "plain_text", "content": UI_TEXT["chat_lock_btn_status"]},
            "type": "default",
            "value": _status_val,
            "behaviors": [{"type": "callback", "value": dict(_status_val)}],
        },
    ]
    if app_id:
        buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": UI_TEXT["chat_lock_btn_go_p2p"]},
            "type": "default",
            "multi_url": _build_p2p_multi_url(app_id),
        })
    else:
        logger.warning("app_id not configured; chat lock card will lack P2P deep-link button")

    return markdown, buttons


def build_lock_success_card(
    action: str, message: str = "", *, variant: str = "reply", locker_name: str = "", app_id: str = "",
    lock_undo_window_seconds: int = 300, chat_id: str = "",
) -> str | tuple[str, list[dict]]:
    """Build a response card after a successful /lock or /unlock command.

    Parameters
    ----------
    action : str
        ``"lock"`` or ``"unlock"``.
    message : str
        Optional idempotent-hint message (e.g. "群已处于锁定状态").
    variant : str
        ``"reply"`` (default) — reply to the admin who ran the command.
        ``"broadcast"`` — group-wide notification (neutral tone, no admin hints).
    locker_name : str
        Display name of the operator, shown in broadcast cards.
    app_id : str
        Feishu app ID for the "go to private chat" deeplink button.

    Returns
    -------
    str
        Plain markdown for most variants.
    tuple[str, list[dict]]
        ``(markdown, buttons)`` for broadcast lock cards (includes actionable buttons).
    """
    if action == "lock":
        if variant == "broadcast":
            by_line = UI_TEXT["lock_success_lock_broadcast_by"].format(name=locker_name) if locker_name else ""
            md = UI_TEXT["lock_success_lock_broadcast"].format(by_line=by_line)
            md += UI_TEXT["lock_success_lock_broadcast_hint"]
            _status_sig = sign_command("/status", chat_id) if chat_id else _compute_command_sig("/status")
            _status_val = {"action": "retry_command", "_t": "/status", "_s": _status_sig}
            buttons: list[dict] = [{
                "tag": "button",
                "text": {"tag": "plain_text", "content": UI_TEXT["chat_lock_btn_status"]},
                "type": "default",
                "value": _status_val,
                "behaviors": [{"type": "callback", "value": dict(_status_val)}],
            }]
            if app_id:
                buttons.append({
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["chat_lock_btn_go_p2p"]},
                    "type": "default",
                    "multi_url": _build_p2p_multi_url(app_id),
                })
            return md, buttons
        if message:
            return UI_TEXT["lock_success_lock_idempotent"].format(message=message)
        # Non-idempotent lock reply: include an undo button (configurable window)
        _undo_display = format_undo_window(lock_undo_window_seconds)
        if _undo_display:
            md = UI_TEXT["lock_success_lock_reply"].format(lock_undo_window_display=_undo_display)
            # Use v2 signing for /unlock when chat_id is available
            _unlock_sig = sign_command("/unlock", chat_id) if chat_id else _compute_command_sig("/unlock")
            _undo_val = {
                "action": "retry_command",
                "_t": "/unlock",
                "_s": _unlock_sig,
                "_ul": True,
                "_ue": int(_t_wall.time()) + lock_undo_window_seconds,
            }
            undo_buttons: list[dict] = [{
                "tag": "button",
                "text": {"tag": "plain_text", "content": UI_TEXT["lock_success_lock_undo_btn"]},
                "type": "default",
                "value": _undo_val,
                "behaviors": [{"type": "callback", "value": dict(_undo_val)}],
            }]
        else:
            if locker_name:
                _tpl = UI_TEXT.get("lock_success_lock_reply_no_undo", "🔒 已锁定，如需解锁请使用 /unlock")
                md = _tpl.format(locker_name=locker_name)
            else:
                md = UI_TEXT.get("lock_success_lock_reply_no_undo_anonymous", "**🔒 已锁定**\n\n非 Bot 管理员的指令将不被 Bot 执行。\nBot 管理员可发送 `/unlock` 解锁。")
            undo_buttons = []
        return md, undo_buttons
    # unlock
    if variant == "broadcast":
        by_line = UI_TEXT["lock_success_lock_broadcast_by"].format(name=locker_name) if locker_name else ""
        md = UI_TEXT["lock_success_unlock_broadcast"].format(by_line=by_line)
        _status_sig = sign_command("/status", chat_id) if chat_id else _compute_command_sig("/status")
        _status_val = {"action": "retry_command", "_t": "/status", "_s": _status_sig}
        buttons: list[dict] = [{
            "tag": "button",
            "text": {"tag": "plain_text", "content": UI_TEXT["chat_lock_btn_status"]},
            "type": "default",
            "value": _status_val,
            "behaviors": [{"type": "callback", "value": dict(_status_val)}],
        }]
        return md, buttons
    if message:
        return UI_TEXT["lock_success_unlock_idempotent"].format(message=message)
    return UI_TEXT["lock_success_unlock_reply"]


def build_lock_help_body(is_admin: bool = False, chat_id: str = "", *, chat_lock_manager=None, allowed_commands_display: Optional[str] = None, lock_undo_window_seconds: int = 300) -> str:
    """Generate lock help section body dynamically from authoritative command sets.

    When *chat_id* is provided and the chat is locked, non-admin users see
    the lock holder's name for transparency.

    Args:
        chat_lock_manager: Optional ``ChatLockManager`` instance.  When provided,
            it is used directly instead of fetching the global singleton.  This
            enables dependency-injection for testing.
    """
    if allowed_commands_display is not None:
        _user_cmds_display = allowed_commands_display
    else:
        from src.chat_lock import ChatLockManager
        _user_cmds_display = ChatLockManager.get_allowed_commands_display()
    lines: list[str] = []
    if is_admin:
        _undo_display = format_undo_window(lock_undo_window_seconds)
        lines.append(UI_TEXT["lock_dual_concept_explain"])
        lines.append(UI_TEXT["lock_help_admin_group_mgmt"])
        if _undo_display:
            lines.append(UI_TEXT["lock_help_admin_lock_cmd"].format(lock_undo_window_display=_undo_display))
        else:
            lines.append(UI_TEXT["lock_help_admin_lock_cmd_no_undo"])
        lines.append(UI_TEXT["lock_help_admin_unlock_cmd"])
        lines.append(UI_TEXT["lock_help_admin_group_info"])
        lines.append(UI_TEXT["lock_help_admin_exempt_cmds"].format(cmd_list=_user_cmds_display))
        lines.append(UI_TEXT["lock_help_admin_repo_lock_hint"])
        # Dynamic lock status for admin
        if chat_id:
            try:
                _mgr = chat_lock_manager
                if _mgr is None:
                    from src.chat_lock import get_chat_lock_manager
                    _mgr = get_chat_lock_manager()
                _info = _mgr.get_lock_info(chat_id)
                if _info:
                    lines.append(UI_TEXT["lock_help_admin_chat_locked"])
                else:
                    lines.append(UI_TEXT["lock_help_admin_chat_unlocked"])
            except Exception:
                logger.debug("Failed to check lock status for admin help body", exc_info=True)
    else:
        lines.append(UI_TEXT["lock_dual_concept_explain_nonadmin"])
        # Show lock holder info when the chat is currently locked
        _locker_hint = ""
        if chat_id:
            try:
                _mgr = chat_lock_manager
                if _mgr is None:
                    from src.chat_lock import get_chat_lock_manager
                    _mgr = get_chat_lock_manager()
                _lock_info = _mgr.get_lock_info(chat_id)
                if _lock_info and _lock_info.locked_by_name:
                    _locker_hint = UI_TEXT["lock_help_locked_by_named"].format(name=_lock_info.locked_by_name)
                elif _lock_info:
                    _locker_hint = UI_TEXT["lock_help_locked_by_admin"]
            except Exception:
                logger.debug("Failed to check lock holder for non-admin help body", exc_info=True)
        lines.append(UI_TEXT["lock_help_nonadmin_contact"])
        lines.append(UI_TEXT["lock_help_nonadmin_repo_hint"])
        lines.append(UI_TEXT["lock_help_nonadmin_exempt_cmds"].format(cmd_list=_user_cmds_display))
        if _locker_hint:
            lines.append(_locker_hint)
    return "\n".join(lines)
