"""Main Bot identity lookup and optional Slash command sync mixin."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import threading
import time

from .slash_commands import SlashPermissionRequired

logger = logging.getLogger(__name__)


def _ws_client_module():
    """Return the ws_client module for runtime-patchable symbols."""
    from . import ws_client as _ws_mod

    return _ws_mod


class WSSlashIdentityMixin:
    """Mixin providing Main Bot Open ID lookup and Slash command reconciliation."""

    def _sync_main_slash_commands(self) -> None:
        """Retry optional Slash discovery convergence without gating service."""

        ws_mod = _ws_client_module()
        while not self._closed:
            self._sync_main_bot_identity()
            try:
                reconcile_fn = getattr(ws_mod, "reconcile_main_agent_slash_commands")
                verified = asyncio.run(
                    reconcile_fn(
                        self._get_api_client(),
                        app_id=self.settings.app_id,
                    )
                )
            except SlashPermissionRequired as exc:
                with self._slash_permission_notice_lock:
                    previous_permission = self._pending_slash_permission
                    self._pending_slash_permission = exc
                if (
                    previous_permission is None
                    or previous_permission.authorization_url
                    != exc.authorization_url
                ):
                    logger.warning(
                        "Optional Main Agent Slash discovery is unavailable; "
                        "messaging, Shell, and programming tools remain available. "
                        "Grant and publish permission, then background recovery will "
                        "retry automatically: %s",
                        exc.authorization_url,
                    )
                self._notify_main_slash_permission_required(exc)
            except Exception as exc:
                logger.warning(
                    "Optional Main Agent Slash discovery sync failed (%s); "
                    "messaging remains available and background recovery will retry",
                    type(exc).__name__,
                )
            else:
                with self._slash_permission_notice_lock:
                    self._pending_slash_permission = None
                logger.info(
                    "Main Agent Slash Commands ready: total=%d created=%d "
                    "updated=%d deleted=%d",
                    len(verified.observed),
                    len(verified.created),
                    len(verified.updated),
                    len(verified.deleted),
                )
                try:
                    import linecache

                    from ..utils.memory_tuning import trim_process_heap

                    linecache.clearcache()
                    gc.collect()
                    trim_process_heap()
                except Exception:
                    pass
                return

            retry_seconds = getattr(ws_mod, "_MAIN_SLASH_SYNC_RETRY_SECONDS", 60.0)
            self._slash_command_sync_wakeup.wait(retry_seconds)
            self._slash_command_sync_wakeup.clear()
            if self._closed:
                return

    def _current_main_slash_permission(
        self,
    ) -> SlashPermissionRequired | None:
        """Return any pending Slash permission grant requirement."""
        with self._slash_permission_notice_lock:
            return self._pending_slash_permission

    @staticmethod
    def _slash_notice_admin_ids(settings: object) -> tuple[str, ...]:
        """Extract canonical admin Open IDs eligible for Slash notices."""
        raw_admin_ids = getattr(settings, "admin_user_ids", ()) or ()
        if isinstance(raw_admin_ids, str):
            return tuple(
                item.strip()
                for item in raw_admin_ids.split(",")
                if item.strip().startswith("ou_")
            )
        return tuple(
            sorted(
                item
                for item in raw_admin_ids
                if isinstance(item, str) and item.startswith("ou_")
            )
        )

    def _register_slash_permission_notice_recipient(
        self,
        *,
        sender_id: str,
        chat_type: str,
    ) -> None:
        """Wake the worker for the first admitted private user when no admin exists."""

        if (
            chat_type != "p2p"
            or not isinstance(sender_id, str)
            or not sender_id.startswith("ou_")
        ):
            return
        if self._slash_notice_admin_ids(self.settings):
            return
        lock = getattr(self, "_slash_permission_notice_lock", None)
        recipients = getattr(self, "_slash_permission_notice_recipients", None)
        if lock is None or recipients is None:
            return
        with lock:
            if recipients:
                return
            recipients.add(sender_id)
            should_wake = self._pending_slash_permission is not None
        if should_wake:
            self._slash_command_sync_wakeup.set()

    def _notify_main_slash_permission_required(
        self,
        required: SlashPermissionRequired,
    ) -> None:
        """Send an audited, deduplicated grant notice to an eligible recipient."""

        admin_ids = self._slash_notice_admin_ids(self.settings)
        with self._slash_permission_notice_lock:
            recipient_ids = admin_ids or tuple(
                sorted(self._slash_permission_notice_recipients)
            )
        if not recipient_ids:
            with self._slash_permission_notice_lock:
                should_log = (
                    required.authorization_url
                    not in self._slash_permission_unroutable_logs
                )
                self._slash_permission_unroutable_logs.add(
                    required.authorization_url
                )
            if should_log:
                logger.warning(
                    "Slash permission grant link could not be sent in Feishu because "
                    "ADMIN_USER_IDS is empty; use /setadmin, then automatic retry will "
                    "notify the administrator: %s",
                    required.authorization_url,
                )
            return

        handler = self._handler_ctx.handlers.get("coco")
        im_client = getattr(handler, "im_client", None)
        send_message = getattr(im_client, "send_message", None)
        if not callable(send_message):
            logger.warning(
                "Slash permission notice transport is unavailable; background "
                "recovery remains active"
            )
            return

        notice = (
            "⚠️ GhostAP 的“/ 命令发现面板”缺少可选权限，"
            "消息、Shell 和编程工具仍可正常使用。\n"
            "请应用管理员点击下方链接申请并发布权限；GhostAP 会在后台"
            "自动重试，无需重启：\n"
            f"{required.authorization_url}\n"
            f"缺失权限：{', '.join(required.scopes)}"
        )
        content = json.dumps({"text": notice}, ensure_ascii=False)
        for recipient_id in recipient_ids:
            notification_key = (recipient_id, required.authorization_url)
            with self._slash_permission_notice_lock:
                if notification_key in self._slash_permission_notifications:
                    continue
            try:
                response = send_message(
                    "open_id",
                    recipient_id,
                    content,
                    msg_type="text",
                )
                if response is not None and response.success():
                    with self._slash_permission_notice_lock:
                        self._slash_permission_notifications.add(notification_key)
                else:
                    logger.warning(
                        "Slash permission notice delivery failed for recipient=%s",
                        recipient_id[:12],
                    )
            except Exception:
                logger.warning(
                    "Slash permission notice delivery raised for recipient=%s",
                    recipient_id[:12],
                    exc_info=True,
                )

    def _ensure_main_bot_identity_sync_state(self) -> threading.Condition:
        """Return the Condition guarding main Bot identity lookup state."""
        lock = getattr(self, "_main_bot_identity_lock", None)
        if lock is None:
            lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
            self._main_bot_identity_lock = lock
        cond = getattr(self, "_main_bot_identity_cond", None)
        if cond is None:
            cond = threading.Condition(lock)
            self._main_bot_identity_cond = cond
        if not hasattr(self, "_main_bot_open_id"):
            self._main_bot_open_id = ""
        if not hasattr(self, "_main_bot_identity_next_retry_at"):
            self._main_bot_identity_next_retry_at = 0.0
        if not hasattr(self, "_main_bot_identity_inflight"):
            self._main_bot_identity_inflight = False
        return cond

    def _sync_main_bot_identity(self) -> str:
        """Resolve the main app's bot Open ID outside the WS ACK path."""

        cond = self._ensure_main_bot_identity_sync_state()
        with cond:
            while self._main_bot_identity_inflight:
                cond.wait()
            if self._main_bot_open_id:
                return self._main_bot_open_id
            now = time.monotonic()
            if now < self._main_bot_identity_next_retry_at:
                return ""
            # Hydrate lark_oapi.core while holding the sync guard so concurrent
            # callers never race between core.enum and core.model.__init__.
            import lark_oapi.core  # noqa: F401
            from lark_oapi.core.enum import AccessTokenType, HttpMethod
            from lark_oapi.core.model.base_request import BaseRequest
            from lark_oapi.core.model.base_response import BaseResponse

            self._main_bot_identity_inflight = True

        resolved_open_id: str | None = None
        try:
            request = (
                BaseRequest.builder()
                .http_method(HttpMethod.GET)
                .uri("/open-apis/bot/v3/info")
                .token_types({AccessTokenType.TENANT})
                .paths({})
                .body(None)
                .build()
            )
            response = self._get_api_client().request(request)
            raw = response.raw if isinstance(response, BaseResponse) else None
            payload = (
                json.loads(raw.content)
                if raw is not None and isinstance(raw.content, bytes)
                else None
            )
            bot = payload.get("bot") if isinstance(payload, dict) else None
            open_id = bot.get("open_id") if isinstance(bot, dict) else None
            app_id = bot.get("app_id") if isinstance(bot, dict) else None
            if (
                not isinstance(response, BaseResponse)
                or not response.success()
                or raw is None
                or not isinstance(raw.status_code, int)
                or not 200 <= raw.status_code < 300
                or not isinstance(payload, dict)
                or payload.get("code") != 0
                or not isinstance(open_id, str)
                or not open_id.startswith("ou_")
                or len(open_id) > 256
                # `/bot/v3/info` is already scoped by the tenant token
                # minted from this configured app.  The official response
                # currently omits `app_id`; if a future response includes
                # it, retain the extra binding check.
                or (app_id is not None and app_id != self.settings.app_id)
            ):
                raise ValueError("invalid main Bot identity response")
            resolved_open_id = open_id
        except Exception:
            logger.warning("Main Bot identity lookup failed closed", exc_info=True)
        finally:
            retry_seconds = getattr(
                _ws_client_module(),
                "_MAIN_BOT_IDENTITY_RETRY_SECONDS",
                1.0,
            )
            with cond:
                if resolved_open_id is not None:
                    self._main_bot_open_id = resolved_open_id
                    self._main_bot_identity_next_retry_at = 0.0
                else:
                    self._main_bot_identity_next_retry_at = (
                        time.monotonic() + retry_seconds
                    )
                self._main_bot_identity_inflight = False
                cond.notify_all()
        return resolved_open_id or ""

    def _start_main_slash_command_sync(self) -> None:
        """Start at most one non-blocking Slash reconciliation worker."""

        if self._slash_command_sync_thread is not None:
            return
        thread = threading.Thread(
            target=self._sync_main_slash_commands,
            name="main-slash-command-sync",
            daemon=True,
        )
        self._slash_command_sync_thread = thread
        thread.start()
