"""Lock-related command handlers extracted from SystemHandler (God Class split)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...card.ui_text import UI_TEXT

if TYPE_CHECKING:
    from ...chat_lock import ChatLockResult

logger = logging.getLogger(__name__)


class LockCommandsMixin:
    """Mixin providing /lock, /unlock, force-release and related card-action handlers.

    Intended for use with ``SystemHandler(LockCommandsMixin, ..., BaseHandler)``.
    All ``self.*`` helper methods (``reply_text``, ``reply_card``, ``update_card``,
    ``send_card_to_chat``, ``send_text_to_chat``, ``settings``, ``ctx``) are resolved via MRO from ``BaseHandler``.
    """

    @staticmethod
    def resolve_lock_message(result: "ChatLockResult") -> str:
        """Map a structured ChatLockCode to a UI text string.

        Pure function — safe for concurrent use and independently testable.
        """
        if result.code is None:
            return ""
        template = UI_TEXT.get(result.code.value, "")
        return template.format(**result.format_params) if result.format_params else template

    def _handle_lock_command(self, message_id: str, chat_id: str, action: str) -> None:
        """Handle /lock and /unlock commands.

        Both /lock and /unlock execute directly (symmetric behaviour).
        """
        from ...card.builders.lock import build_lock_success_card
        from ...config import get_settings as _get_settings
        from ...thread import get_current_is_p2p, get_current_sender_id, get_current_sender_name

        _settings = _get_settings()
        _lock_undo_window = _settings.lock_undo_window_seconds
        _app_id = getattr(_settings, "feishu_app_id", "") or ""

        # Lock/unlock is only meaningful in group chats.
        if get_current_is_p2p():
            self.reply_error(message_id, UI_TEXT["lock_cmd_p2p_only"])
            return

        chat_lock_manager = getattr(self.ctx, "chat_lock_manager", None)
        if chat_lock_manager is None:
            self.reply_error(message_id, UI_TEXT["lock_cmd_not_enabled"])
            return

        sender_id = get_current_sender_id() or ""
        if not sender_id:
            self.reply_error(message_id, UI_TEXT["lock_cmd_unknown_sender"])
            return

        sender_name = get_current_sender_name() or sender_id[:8]

        if action == "lock":
            # Execute directly — lock_chat() handles admin check and idempotency.
            result = chat_lock_manager.lock_chat(chat_id, sender_id, sender_name=sender_name)
            if result.success:
                idempotent_msg = self.resolve_lock_message(result) if result.idempotent else ""
                if result.idempotent:
                    # Enrich with current locker info
                    _lock_info = chat_lock_manager.get_lock_info(chat_id)
                    if _lock_info and _lock_info.locked_by_name:
                        idempotent_msg += f"（锁定者：{_lock_info.locked_by_name}）"
                    # Info-style card for idempotent response (no new action taken)
                    from ...card.builders.project import ProjectBuilder as _PB
                    _md = build_lock_success_card("lock", message=idempotent_msg)
                    _mt, _card = _PB.build_project_response_card(
                        project=None, title=UI_TEXT["lock_card_title_hint"], content=_md, show_buttons=False)
                    self.reply_card(message_id, _card)
                else:
                    _reply = build_lock_success_card("lock", message=idempotent_msg, lock_undo_window_seconds=_lock_undo_window, chat_id=chat_id)
                    if isinstance(_reply, tuple):
                        _md, _btns = _reply
                        from ...card.builders.project import ProjectBuilder as _PB2
                        _msg_type, _card = _PB2.build_project_response_card(
                            project=None, title=UI_TEXT["chat_locked_title"],
                            content=_md, show_buttons=False, extra_buttons=_btns,
                        )
                        self.reply_card(message_id, _card)
                    else:
                        self.reply_text(message_id, _reply)
                if not result.idempotent:
                    _broadcast = build_lock_success_card("lock", variant="broadcast", locker_name=sender_name, app_id=_app_id, chat_id=chat_id)
                    if isinstance(_broadcast, tuple):
                        from ...card.builders.project import ProjectBuilder
                        _md, _btns = _broadcast
                        _msg_type, _card = ProjectBuilder.build_project_response_card(
                            project=None, title=UI_TEXT["chat_locked_title"],
                            content=_md, show_buttons=False, extra_buttons=_btns,
                        )
                        self.send_card_to_chat(chat_id, _card)
                    else:
                        self.send_text_to_chat(chat_id, _broadcast)
            else:
                self._reply_lock_permission_error(message_id, result)
            return

        # /unlock — execute directly (symmetric with /lock)
        result = chat_lock_manager.unlock_chat(chat_id, sender_id)
        if result.success:
            idempotent_msg = self.resolve_lock_message(result) if result.idempotent else ""
            if result.idempotent:
                from ...card.builders.project import ProjectBuilder as _PB2
                _md = build_lock_success_card(action, message=idempotent_msg)
                _mt, _card = _PB2.build_project_response_card(
                    project=None, title=UI_TEXT["lock_card_title_hint"], content=_md, show_buttons=False)
                self.reply_card(message_id, _card)
            else:
                self.reply_text(message_id, build_lock_success_card(action, message=idempotent_msg))
            if not result.idempotent:
                _unlock_broadcast = build_lock_success_card("unlock", variant="broadcast", locker_name=sender_name, chat_id=chat_id)
                if isinstance(_unlock_broadcast, tuple):
                    from ...card.builders.project import ProjectBuilder as _PB3
                    _md, _btns = _unlock_broadcast
                    _msg_type, _card = _PB3.build_project_response_card(
                        project=None, title=UI_TEXT["chat_unlocked_title"],
                        content=_md, show_buttons=False, extra_buttons=_btns,
                    )
                    self.send_card_to_chat(chat_id, _card)
                else:
                    self.send_text_to_chat(chat_id, _unlock_broadcast)
        else:
            self._reply_lock_permission_error(message_id, result)

    def _reply_lock_permission_error(self, message_id: str, result) -> None:
        """Reply with an actionable card for lock/unlock permission failures."""
        from ...chat_lock import ChatLockCode

        _err_msg = self.resolve_lock_message(result)
        # Append admin names for non-admin callers
        if result.code == ChatLockCode.CONTACT_ADMIN_TO_LOCK:
            try:
                _admin_ids = self.settings.admin_user_ids
                if _admin_ids:
                    from ..user_cache import resolve_display_name
                    _names = [resolve_display_name(uid) or uid[:8] for uid in list(_admin_ids)[:3]]
                    _admin_line = "、".join(_names)
                    if len(_admin_ids) > 3:
                        _admin_line += f" 等 {len(_admin_ids)} 人"
                    _err_msg += f"\n\n👤 Bot 管理员: {_admin_line}"
            except Exception:
                logger.debug("Failed to append admin list to lock error", exc_info=True)
        _non_admin_codes = {
            ChatLockCode.CONTACT_ADMIN_TO_LOCK,
            ChatLockCode.NO_ADMIN_CONFIG_USER,
            ChatLockCode.CONTACT_NAMED_UNLOCK,
            ChatLockCode.CONTACT_ADMIN_UNLOCK,
        }
        if result.code in _non_admin_codes:
            try:
                from ...card.builders.project import ProjectBuilder as _PB3
                _buttons: list[dict] = [{
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["chat_lock_btn_status"]},
                    "type": "default",
                    "value": {"action": "retry_command", "_t": "/status"},
                }]
                _app_id = getattr(self.settings, "app_id", "") or ""
                if _app_id:
                    from ...card.builders.lock import _build_p2p_multi_url
                    _buttons.append({
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": UI_TEXT["chat_lock_btn_go_p2p"]},
                        "type": "default",
                        "multi_url": _build_p2p_multi_url(_app_id),
                    })
                _mt, _card = _PB3.build_project_response_card(
                    project=None, title=UI_TEXT["lock_card_title_no_permission"], content=_err_msg,
                    show_buttons=False, extra_buttons=_buttons,
                )
                self.reply_card(message_id, _card)
                return
            except Exception:
                logger.debug("Failed to build lock permission error card", exc_info=True)
        self.reply_error(message_id, _err_msg)

    def _check_confirm_card_expiry(
        self,
        message_id: str,
        value: dict,
        *,
        retry_button_label: str,
        retry_button_type: str = "primary",
        retry_button_value: dict,
        expired_title_key: str,
    ) -> bool:
        """Check if a confirm card has expired and send an expiry reply if so.

        Returns True if expired (caller should return early), False otherwise.
        """
        from ...card.builders.project import ProjectBuilder
        from ...config import get_settings

        timestamp = value.get("_ts") or value.get("timestamp", 0)
        try:
            timeout = get_settings().lock_confirm_timeout
        except Exception:
            timeout = 120
        if not timestamp or (time.time() - timestamp) <= timeout:
            return False

        _timeout_min = max(1, timeout // 60)
        _timeout_md = UI_TEXT["lock_cmd_confirm_expired_msg"].format(minutes=_timeout_min)
        _timeout_buttons = [{
            "tag": "button",
            "text": {"tag": "plain_text", "content": retry_button_label},
            "type": retry_button_type,
            "value": retry_button_value,
        }]
        msg_type, card_json = ProjectBuilder.build_project_response_card(
            project=None,
            title=UI_TEXT[expired_title_key],
            content=_timeout_md,
            show_buttons=False,
            extra_buttons=_timeout_buttons,
        )
        self.reply_card(message_id, card_json)
        return True

    def handle_confirm_lock(
        self, message_id: str, chat_id: str, project_id: Optional[str] = None, value: dict | None = None
    ) -> None:
        """Card action callback: deprecated — /lock now executes directly.

        Old confirmation cards may still be floating in chats; clicking
        "confirm" on them should gracefully reply with a hint to use /lock.
        """
        self.reply_text(message_id, UI_TEXT["lock_confirm_deprecated"])

    def handle_cancel_lock(
        self, message_id: str, chat_id: str, project_id: Optional[str] = None, value: dict | None = None
    ) -> None:
        """Card action callback: deprecated — /lock no longer uses confirmation cards."""
        self.reply_text(message_id, UI_TEXT["lock_confirm_deprecated"])

    def handle_force_release_repo_lock(
        self, message_id: str, chat_id: str, project_id: Optional[str] = None, value: dict | None = None
    ) -> None:
        """Card action: admin requests force-release — show confirmation card first."""
        from ...thread import get_current_sender_id

        sender_id = get_current_sender_id() or ""
        chat_lock_manager = getattr(self.ctx, "chat_lock_manager", None)
        if chat_lock_manager is None or not chat_lock_manager.is_admin(sender_id):
            self.reply_error(message_id, UI_TEXT["lock_force_release_admin_only"])
            return

        value = value or {}
        repo_lock_mgr = getattr(self.ctx, "repo_lock_manager", None)
        if repo_lock_mgr is None:
            self.reply_error(message_id, UI_TEXT["lock_repo_mgr_not_init"])
            return

        repo_token = value.get("_tk") or value.get("repo_token", "")
        root_path = ""
        if repo_token:
            root_path = repo_lock_mgr.token_to_path(repo_token) or ""
        if not root_path:
            root_path = value.get("root_path", "")
        if not root_path:
            if repo_token:
                self.reply_error(message_id, UI_TEXT["lock_repo_already_released"])
            else:
                self.reply_error(message_id, UI_TEXT["lock_repo_path_not_found"])
            return

        # Ensure token exists for the confirm card payload
        if not repo_token:
            repo_token = repo_lock_mgr.path_to_token(root_path)

        repo_name = Path(root_path).name or root_path

        # Build holder context hint for the confirmation card
        holder_hint = ""
        _holder_chat_id = ""
        try:
            from ...card.builders.lock import format_elapsed_ago
            lock_info = repo_lock_mgr.get_lock_info(root_path) if hasattr(repo_lock_mgr, "get_lock_info") else None
            if lock_info:
                _hcid_raw = getattr(lock_info, "chat_id", None)
                _holder_chat_id = str(_hcid_raw) if isinstance(_hcid_raw, str) else ""
                if getattr(lock_info, "acquired_at", None):
                    import time as _t
                    elapsed = _t.monotonic() - lock_info.acquired_at
                    duration = format_elapsed_ago(elapsed)
                    holder_hint = UI_TEXT.get(
                        "lock_force_release_holder_hint",
                        "持有者已锁定 {duration}",
                    ).format(duration=duration)
        except Exception:
            logger.debug("failed to get repo lock holder info", exc_info=True)

        # F-22: Show confirmation card instead of releasing immediately
        from ...card.builders.lock import build_force_release_confirm_card
        from ...card.builders.project import ProjectBuilder

        confirm_content, confirm_buttons = build_force_release_confirm_card(
            repo_token, repo_name, holder_hint=holder_hint,
            holder_chat_id=_holder_chat_id,
        )
        msg_type, card_json = ProjectBuilder.build_project_response_card(
            project=None,
            title=UI_TEXT["lock_force_release_confirm_card_title"],
            content=confirm_content,
            show_buttons=False,
            extra_buttons=confirm_buttons,
        )
        self.reply_card(message_id, card_json)

    def handle_confirm_force_release(
        self, message_id: str, chat_id: str, project_id: Optional[str] = None, value: dict | None = None
    ) -> None:
        """Card action callback: execute the actual force-release after admin confirms."""
        from ...thread import get_current_sender_id

        value = value or {}
        sender_id = get_current_sender_id() or ""

        chat_lock_manager = getattr(self.ctx, "chat_lock_manager", None)
        if chat_lock_manager is None or not chat_lock_manager.is_admin(sender_id):
            self.reply_error(message_id, UI_TEXT["lock_force_release_admin_only"])
            return

        # Check expiry
        if self._check_confirm_card_expiry(
            message_id, value,
            retry_button_label=UI_TEXT["lock_btn_retry_force_release"],
            retry_button_type="danger",
            retry_button_value={
                "action": "force_release_repo_lock",
                "_tk": value.get("_tk") or value.get("repo_token", ""),
            },
            expired_title_key="lock_force_release_expired_title",
        ):
            return

        repo_lock_mgr = getattr(self.ctx, "repo_lock_manager", None)
        if repo_lock_mgr is None:
            self.reply_error(message_id, UI_TEXT["lock_repo_mgr_not_init"])
            return

        repo_token = value.get("_tk") or value.get("repo_token", "")
        root_path = repo_lock_mgr.token_to_path(repo_token) if repo_token else ""
        if not root_path:
            if repo_token:
                self.reply_error(message_id, UI_TEXT["lock_repo_already_released"])
            else:
                self.reply_error(message_id, UI_TEXT["lock_repo_path_not_found"])
            return

        # Capture holder before releasing, for notification
        _lock_info = repo_lock_mgr.get_lock_info(root_path)
        _holder_chat_id = _lock_info.chat_id if _lock_info else None

        # F-01: Race guard — verify that the lock holder hasn't changed since
        # the confirmation card was built.  If _hcid is present in the payload
        # and differs from the current holder, abort with a warning.
        # Old cards without _hcid are allowed through for backward compat.
        _expected_hcid = value.get("_hcid", "")
        if _expected_hcid and _holder_chat_id and _expected_hcid != _holder_chat_id:
            repo_name = Path(root_path).name or root_path
            self.reply_error(
                message_id,
                UI_TEXT["lock_force_release_holder_changed"].format(repo_name=repo_name),
            )
            return

        repo_lock_mgr.force_release(root_path)
        repo_name = Path(root_path).name or root_path
        self.reply_text(message_id, UI_TEXT["lock_force_release_success"].format(repo_name=repo_name))

        # Fire-and-forget: notify the original holder chat
        if _holder_chat_id and _holder_chat_id != chat_id:
            try:
                from ...card.builders.lock import build_lock_reclaim_notify_card
                self.send_text_to_chat(
                    _holder_chat_id,
                    build_lock_reclaim_notify_card(repo_name, reason="force_release"),
                )
            except Exception as notify_err:
                logger.warning("Failed to notify original lock holder chat=%s: %s", _holder_chat_id[:12], notify_err)

    def handle_cancel_force_release(
        self, message_id: str, chat_id: str, project_id: Optional[str] = None, value: dict | None = None
    ) -> None:
        """Card action callback: cancel the force-release confirmation."""
        self.reply_text(message_id, UI_TEXT["lock_cmd_cancel_force_release"])
