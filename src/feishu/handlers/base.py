"""Base handler providing shared messaging, reaction, and context utilities.

Every concrete handler (Coco, Claude, Deep, Project, System, Diagnostics)
inherits from ``BaseHandler`` so that it can reply to messages, add reactions,
manage streaming cards, and interact with the unified context system without
duplicating the underlying Feishu API calls.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass as _dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from ...card.ui_text import UI_TEXT
from ...utils.engine_identity import resolve_engine_identity
from ...utils.errors import get_error_detail
from ..im_client import FeishuIMClient
from ..ws_card_action_handler import bind_managed_trust_revisions


@_dataclass
class CardActionContext:
    """Encapsulates the context needed for dispatching card actions."""
    open_message_id: str
    open_chat_id: str
    action_type: str
    value: dict
    prefix: str
    action_map: "dict[str, Callable]"
    toggle_log_method: "Optional[Callable]" = None
    toggle_ac_method: "Optional[Callable]" = None
    switch_mode_method: "Optional[Callable]" = None
    project: "Optional[Any]" = None

if TYPE_CHECKING:
    from ...acp.manager import ACPSessionManager
    from ...project import ContextSourceMode, ProjectContext
    from ...repo_lock import LockConflictError
    from ..handler_context import HandlerContext

logger = logging.getLogger(__name__)


class BaseHandler:
    """Shared utilities available to every handler."""

    def __init__(self, ctx: "HandlerContext") -> None:
        self.ctx = ctx
        self._card_delivery = None  # Lazy-init singleton CardDelivery
        self.im_client = FeishuIMClient(
            ctx.api_client_factory,
            ctx.settings,
            outbound_audit=ctx.main_bot_outbound_audit,
            outbound_audit_failure=ctx.main_bot_outbound_audit_failure,
            tenant_key_resolver=ctx.tenant_key_resolver,
        )
        # Lock helper (composition — keeps BaseHandler focused on messaging)
        from .lock_helper import LockHelper
        self.lock_helper = LockHelper(self)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def settings(self):
        return self.ctx.settings

    @property
    def project_manager(self):
        return self.ctx.project_manager

    @property
    def mode_manager(self):
        return self.ctx.mode_manager

    @property
    def context_manager(self):
        return self.ctx.context_manager

    @property
    def scheduler(self):
        return self.ctx.scheduler

    def get_handler(self, key: str) -> Optional[Any]:
        """Get a handler from the registry."""
        return self.ctx.handlers.get(key)

    def get_manager(self, key: str) -> Optional["ACPSessionManager"]:
        """Get a manager from the registry."""
        return self.ctx.managers.get(key)

    def upload_acp_image(self, image) -> str | None:
        """Validate and upload a typed ACP image without logging its payload."""
        from ...acp.client import (
            MAX_ACP_IMAGE_BYTES,
            SUPPORTED_ACP_IMAGE_MIME_TYPES,
            detect_acp_image_mime,
        )

        try:
            mime_type = str(image.mime_type or "").strip().casefold()
            if mime_type not in SUPPORTED_ACP_IMAGE_MIME_TYPES:
                raise ValueError("Unsupported ACP image MIME type")
            compact = "".join(str(image.data or "").split())
            max_encoded_chars = ((MAX_ACP_IMAGE_BYTES + 2) // 3) * 4
            if not compact or len(compact) > max_encoded_chars:
                raise ValueError("ACP image encoding is empty or too large")
            payload = base64.b64decode(compact, validate=True)
            if not payload or len(payload) > MAX_ACP_IMAGE_BYTES:
                raise ValueError("ACP image payload is empty or too large")
            if detect_acp_image_mime(payload) != mime_type:
                raise ValueError("ACP image bytes do not match MIME type")
            return self.im_client.upload_image_bytes(payload)
        except Exception as exc:
            logger.warning(
                "ACP 图片产物上传失败: %s",
                type(exc).__name__,
            )
            return None

    def send_error_card(
        self,
        chat_id: str,
        exc: Exception | str,
        title: str = "",
        origin_message_id: Optional[str] = None,
        reply_in_thread: Optional[bool] = None,
        *,
        details: Optional[str] = None,
        detail_action: Optional[dict] = None,
        retry_action: Optional[dict] = None,
    ):
        """Send a structured error card (schema 2.0) with QuickActions if available."""
        from ...card import CardBuilder

        if not title:
            title = UI_TEXT["system_error_title"]

        try:
            detail_value = detail_action
            if detail_value is None:
                detail_value = {
                    "action": "show_error_details",
                    "chat_id": chat_id,
                    "origin_message_id": origin_message_id or "",
                    "title": title,
                    "summary": get_error_detail(exc),
                }
            _, card_json_str = CardBuilder.build_error_card(
                exc,
                title=title,
                details=details or f"错误上下文：chat={chat_id}，message={origin_message_id or '未绑定原消息'}",
                detail_action=detail_value,
                retry_action=retry_action,
            )

            if origin_message_id:
                self.reply_card(
                    origin_message_id, card_json_str, reply_in_thread=reply_in_thread
                )
            else:
                self.send_card_to_chat(chat_id, card_json_str)
        except Exception as e:
            logger.error("发送错误卡片失败: %s", e, exc_info=True)
            # Fallback to simple text reply
            fallback_detail = get_error_detail(exc)
            if origin_message_id:
                self.reply_text(origin_message_id, f"❌ {title}: {fallback_detail}")
            else:
                self.send_text_to_chat(chat_id, f"❌ {title}: {fallback_detail}")

    def reply_error(
        self,
        message_id: str,
        exc: Exception | str,
        title: str = "",
        chat_id: Optional[str] = None,
    ):
        """Convenience wrapper for send_error_card to reply to a message."""
        self.send_error_card(chat_id=chat_id or "unknown", exc=exc, title=title, origin_message_id=message_id)

    # ------------------------------------------------------------------
    # Unified messaging API
    # ------------------------------------------------------------------

    def create_static_card_session(
        self,
        chat_id: str,
        *,
        reply_to: str | None = None,
        session_id: str | None = None,
    ):
        """Create a lightweight static card session for pre-built card JSON delivery.

        Wraps CardDelivery directly without reduce/render pipeline.
        Used by diagnostics handler for pre-built static cards.
        """
        from ...card.session.static import StaticCardSession

        delivery = self.get_card_delivery()
        return StaticCardSession(
            delivery, chat_id, reply_to=reply_to, session_id=session_id
        )

    def get_card_delivery(self):
        """Get a shared CardDelivery instance (lazy-init singleton per handler)."""
        if self._card_delivery is None:
            from ...card.delivery.factory import create_card_delivery
            from ...card.delivery.feishu_client import FeishuCardAPIClient

            api_client = FeishuCardAPIClient(
                self.ctx.api_client_factory(),
                outbound_audit=self.ctx.main_bot_outbound_audit,
                outbound_audit_failure=self.ctx.main_bot_outbound_audit_failure,
                tenant_key_resolver=self.ctx.tenant_key_resolver,
                outbound_target_aliases=lambda target: self._reply_audit_aliases(
                    self._resolve_origin(target)
                ),
            )
            self._card_delivery = create_card_delivery(
                api_client,
                payload_transform=lambda chat_id, payload: (
                    self._bind_managed_card_revisions(
                        payload,
                        chat_id=chat_id,
                    )
                ),
            )
        return self._card_delivery

    def _resolve_origin(self, message_id: str) -> str:
        """Best-effort resolve origin message for linking."""
        try:
            origin = self.ctx.message_linker.resolve_origin(reply_message_id=message_id)
            return origin or message_id
        except Exception:
            return message_id

    def _link_reply_response(self, origin_message_id: str, reply_id: Optional[str]) -> None:
        """Best-effort link a reply response to its origin."""
        if reply_id and origin_message_id:
            try:
                self.ctx.message_linker.link_reply(origin_message_id, reply_id)
            except Exception as e:
                logger.warning("Failed to link reply %s → origin %s: %s", reply_id, origin_message_id, str(e))

    def _reply_audit_aliases(self, origin_message_id: str) -> tuple[str, ...]:
        """Resolve canonical recipient coordinates from trusted origin provenance."""

        try:
            provenance = self.ctx.message_linker.query(origin_message_id)
        except Exception:
            return ()
        if not isinstance(provenance, Mapping):
            return ()
        chat_id = provenance.get("chat_id")
        if not isinstance(chat_id, str) or not chat_id:
            return ()
        aliases = [chat_id]
        sender_id = provenance.get("sender_id")
        if (
            provenance.get("chat_type") == "p2p"
            and isinstance(sender_id, str)
            and sender_id
            and sender_id != chat_id
        ):
            aliases.append(sender_id)
        return tuple(aliases)

    def _bind_managed_card_revisions(
        self,
        card_content: Any,
        *,
        chat_id: str = "",
        origin_message_id: str = "",
    ) -> Any:
        registry = getattr(self.ctx, "managed_group_registry", None)
        if registry is None:
            return card_content
        if not chat_id and origin_message_id:
            try:
                provenance = self.ctx.message_linker.query(origin_message_id)
            except Exception:
                provenance = None
            if isinstance(provenance, Mapping):
                candidate = provenance.get("chat_id")
                chat_id = candidate if isinstance(candidate, str) else ""
        if not chat_id:
            return card_content
        try:
            group, grant = registry.trust_snapshot(chat_id)
        except Exception:
            logger.warning("managed card trust snapshot unavailable", exc_info=True)
            return card_content
        if group is None or grant is None:
            return card_content
        raw = card_content
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                return card_content
        if not isinstance(raw, dict):
            return card_content
        return bind_managed_trust_revisions(
            raw,
            group_revision=group.revision,
            grant_revision=grant.revision,
        )

    # ------------------------------------------------------------------
    # Messaging API
    # ------------------------------------------------------------------

    def reply_text(
        self,
        message_id: str,
        text: str,
        *,
        reply_in_thread: Optional[bool] = None,
        idempotency_key: str | None = None,
    ) -> Optional[str]:
        """Reply with plain text to *message_id*.

        Args:
            message_id: The Feishu message ID to reply to.
            text: Plain text content to send. Must not be None or empty.
            reply_in_thread: Whether to reply in thread. If None, uses
                ``settings.default_reply_mode`` to determine.
            idempotency_key: Optional stable Feishu UUID for safe retries.

        Returns:
            The sent message's message_id on success, or None on failure.

        Raises:
            No exceptions raised; errors are logged and None is returned.
        """
        if text is None or text == "":
            logger.warning("reply_text 收到空内容 (text=%r)，跳过发送", text)
            return None
        origin = self._resolve_origin(message_id)
        audit_aliases = self._reply_audit_aliases(origin)
        request_id = self.ensure_request_id(origin)
        ref_note = self.format_ref_note(origin, request_id, None)
        text_val = str(text)
        if ref_note:
            text_val = f"{text_val}\n\n{ref_note}"
        content_str = json.dumps({"text": text_val})

        if reply_in_thread is None:
            reply_in_thread = self.settings.default_reply_mode == "thread"

        try:
            response = self.im_client.reply_message(
                message_id, content_str, msg_type="text",
                reply_in_thread=reply_in_thread,
                idempotency_key=idempotency_key,
                audit_aliases=audit_aliases,
            )
            if response and response.success() and response.data and response.data.message_id:
                reply_id = response.data.message_id
                self._link_reply_response(origin, reply_id)
                return reply_id
            return None
        except Exception as e:
            logger.error("reply_text 异常: %s", e, exc_info=True)
            return None

    def reply_card(
        self,
        message_id: str,
        card_content,
        *,
        reply_in_thread: Optional[bool] = None,
    ) -> Optional[str]:
        """Reply with an interactive card to *message_id*.

        Args:
            message_id: The Feishu message ID to reply to.
            card_content: Card JSON string or dict (CardKit v2 format with ``body.elements``
                array). Will be normalized and have ref-note injected automatically.
            reply_in_thread: Whether to reply in thread. If None, uses
                ``settings.default_reply_mode`` to determine.

        Returns:
            The sent message's message_id on success, or None on failure.

        Raises:
            No exceptions raised; errors are logged and None is returned.
        """
        origin = self._resolve_origin(message_id)
        audit_aliases = self._reply_audit_aliases(origin)
        request_id = self.ensure_request_id(origin)
        ref_note = self.format_ref_note(origin, request_id, None)
        card_content = self._bind_managed_card_revisions(
            card_content,
            origin_message_id=origin,
        )
        content_str = self._inject_ref_note(card_content, "interactive", ref_note)

        if reply_in_thread is None:
            reply_in_thread = self.settings.default_reply_mode == "thread"

        try:
            response = self.im_client.reply_message(
                message_id, content_str, msg_type="interactive",
                reply_in_thread=reply_in_thread,
                audit_aliases=audit_aliases,
            )
            if response and response.success() and response.data and response.data.message_id:
                reply_id = response.data.message_id
                self._link_reply_response(origin, reply_id)
                return reply_id
            return None
        except Exception as e:
            logger.error("reply_card 异常: %s", e, exc_info=True)
            return None

    def update_card(self, message_id: str, card_content) -> bool:
        """Update an existing card message in-place.

        Args:
            message_id: The message_id of the card to update.
            card_content: Card JSON string or dict (CardKit v2 format).

        Returns:
            True on success, False on failure.

        Raises:
            No exceptions raised; errors are logged and False is returned.
        """
        try:
            origin = self._resolve_origin(message_id)
            card_content = self._bind_managed_card_revisions(
                card_content,
                origin_message_id=origin,
            )
            content_str = self._normalize_interactive_card_content(card_content)
            response = self.im_client.patch_message(
                message_id,
                content_str,
                audit_aliases=self._reply_audit_aliases(origin),
            )
            return bool(response and response.success())
        except Exception as e:
            logger.error("update_card 异常: %s", e, exc_info=True)
            return False

    def send_card_to_chat(
        self,
        chat_id: str,
        card_content,
        *,
        origin_message_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[str]:
        """Send a card to *chat_id* (not a reply).

        Args:
            chat_id: The Feishu chat_id to send the card to.
            card_content: Card JSON string or dict (CardKit v2 format with ``body.elements``
                array). Will have ref-note injected automatically.
            origin_message_id: Optional original message_id for response linking.
            request_id: Optional request_id for ref-note generation.

        Returns:
            The sent message's message_id on success, or None on failure.

        Raises:
            No exceptions raised; errors are logged and None is returned.
        """
        try:
            card_content = self._bind_managed_card_revisions(
                card_content,
                chat_id=chat_id,
                origin_message_id=origin_message_id or "",
            )
            ref_note = self.format_ref_note(origin_message_id, request_id, None)
            content_str = self._inject_ref_note(card_content, "interactive", ref_note)
            response = self.im_client.send_message(
                "chat_id", chat_id, content_str, msg_type="interactive",
            )
            if response and response.success() and response.data and response.data.message_id:
                mid = response.data.message_id
                if origin_message_id:
                    self._link_reply_response(origin_message_id, mid)
                return mid
            return None
        except Exception as e:
            logger.error("send_card_to_chat 异常: %s", e, exc_info=True)
            return None

    def send_text_to_chat(self, chat_id: str, text: str) -> Optional[str]:
        """Send plain text to *chat_id* (not a reply).

        Args:
            chat_id: The Feishu chat_id to send text to.
            text: Plain text content to send.

        Returns:
            The sent message's message_id on success, or None on failure.

        Raises:
            No exceptions raised; errors are logged and None is returned.
        """
        try:
            content_str = json.dumps({"text": str(text)})
            response = self.im_client.send_message(
                "chat_id", chat_id, content_str, msg_type="text",
            )
            if response and response.success() and response.data and response.data.message_id:
                return response.data.message_id
            return None
        except Exception as e:
            logger.error("send_text_to_chat 异常: %s", e, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Ref-note injection (shared by reply_text / reply_card / send_card_to_chat)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_interactive_card_content(content_str) -> str:
        """Normalize outgoing interactive card JSON to Feishu-accepted shape.

        Accepts both str (JSON) and dict inputs. Always returns a JSON string.
        """
        if isinstance(content_str, dict):
            card = content_str
        elif isinstance(content_str, str):
            try:
                card = json.loads(content_str)
            except Exception:
                return content_str
        else:
            return json.dumps(content_str, ensure_ascii=False) if content_str is not None else ""

        if not isinstance(card, dict):
            return json.dumps(card, ensure_ascii=False) if card is not None else ""
        if str(card.get("schema") or "").strip() != "2.0":
            return json.dumps(card, ensure_ascii=False)

        root_elements = card.get("elements")
        if isinstance(root_elements, list):
            body = card.get("body")
            if not isinstance(body, dict):
                body = {}
                card["body"] = body
            if not isinstance(body.get("elements"), list):
                body["elements"] = root_elements
            card.pop("elements", None)

        return json.dumps(card, ensure_ascii=False)

    @staticmethod
    def _inject_ref_note(content_str, msg_type: str, ref_note: str) -> str:
        """Best-effort inject ref_note into interactive/post content. Returns modified content_str."""
        if msg_type == "interactive":
            content_str = BaseHandler._normalize_interactive_card_content(content_str)

        if not ref_note:
            return content_str

        if msg_type == "interactive" and isinstance(content_str, str):
            try:
                card = json.loads(content_str)
                body = card.get("body") if isinstance(card, dict) else None
                if isinstance(body, dict) and isinstance(body.get("elements"), list):
                    body["elements"].append(
                        {
                            "tag": "markdown",
                            "text_size": "normal",
                            "content": ref_note,
                        }
                    )
                    return json.dumps(card, ensure_ascii=False)
            except Exception as e:
                logger.warning("Failed to inject ref note into interactive card: %s", e, exc_info=True)
        elif msg_type == "post" and isinstance(content_str, str):
            try:
                post = json.loads(content_str)
                if isinstance(post, dict):
                    lang = post.get("zh_cn")
                    if isinstance(lang, dict) and isinstance(lang.get("content"), list):
                        blocks = lang.get("content")
                        injected = False
                        for row in reversed(blocks):
                            if not isinstance(row, list):
                                continue
                            for node in reversed(row):
                                if (
                                    isinstance(node, dict)
                                    and node.get("tag") == "md"
                                    and isinstance(node.get("text"), str)
                                ):
                                    node["text"] = f"{node['text']}\n\n---\n{ref_note}"
                                    injected = True
                                    break
                            if injected:
                                break
                        if not injected:
                            blocks.append([{"tag": "md", "text": f"---\n{ref_note}"}])
                        return json.dumps(post, ensure_ascii=False)
            except Exception as e:
                logger.warning("Failed to inject ref note into post content: %s", e, exc_info=True)

        return content_str

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------
    def add_reaction(self, message_id: str, emoji_type: str):
        from ..emoji import EmojiReaction

        if not EmojiReaction.should_send(emoji_type):
            logger.debug("跳过非保留自动表情: %s", emoji_type)
            return
        try:
            self.im_client.add_reaction(message_id, emoji_type)
        except Exception as e:
            logger.warning("添加表情异常: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Card Action Dispatcher (Template Method)
    # ------------------------------------------------------------------
    def _dispatch_standard_card_action(self, ctx: CardActionContext) -> bool:
        """
        Dispatch standard card actions (pause, resume, stop, expand, collapse, mode_full, mode_compact).
        Returns True if action was handled, False otherwise.
        """
        # 1. Lifecycle actions (pause, resume, stop)
        if ctx.action_type in ctx.action_map:
            ctx.action_map[ctx.action_type](ctx.open_message_id, ctx.open_chat_id, project=ctx.project)
            return True

        # Common extraction for UI state actions
        # Note: 'deep_project_id' is the convention used in card buttons for Deep/Spec/Worktree engines
        engine_project_id = ctx.value.get("deep_project_id", "")

        # 2. Log expansion
        if ctx.action_type in (f"{ctx.prefix}_expand", f"{ctx.prefix}_collapse"):
            if ctx.toggle_log_method:
                expanded = ctx.action_type == f"{ctx.prefix}_expand"
                ctx.toggle_log_method(ctx.open_message_id, ctx.open_chat_id, ctx.project, engine_project_id, expanded)
                return True

        # 3. View mode
        if ctx.action_type in (f"{ctx.prefix}_mode_full", f"{ctx.prefix}_mode_compact"):
            if ctx.switch_mode_method:
                compact = ctx.action_type == f"{ctx.prefix}_mode_compact"
                ctx.switch_mode_method(ctx.open_message_id, ctx.open_chat_id, ctx.project, engine_project_id, compact)
                return True

        # 4. Acceptance-criteria expansion (AC)
        if ctx.action_type in (f"{ctx.prefix}_expand_ac", f"{ctx.prefix}_collapse_ac"):
            if ctx.toggle_ac_method:
                expand_ac = ctx.action_type == f"{ctx.prefix}_expand_ac"
                ctx.toggle_ac_method(ctx.open_message_id, ctx.open_chat_id, ctx.project, engine_project_id, expand_ac)
                return True

        return False

    # ------------------------------------------------------------------
    # Project ↔ message registration
    # ------------------------------------------------------------------
    def register_message_project(self, message_id: str, project: "ProjectContext"):
        self.ctx.message_mapper.register(message_id, project.project_id)

    # ------------------------------------------------------------------
    # Working directory
    # ------------------------------------------------------------------
    def get_working_dir(self, chat_id: str) -> str:
        with self.ctx.working_dir_lock:
            return self.ctx.working_dirs.get(chat_id, os.getcwd())

    def set_working_dir(self, chat_id: str, path: str) -> tuple[bool, str]:
        expanded_path = os.path.expanduser(path)
        if not os.path.isabs(expanded_path):
            current_dir = self.get_working_dir(chat_id)
            expanded_path = os.path.normpath(os.path.join(current_dir, expanded_path))

        # 路径范围校验 — 使用 realpath 解析 symlink，避免 /home/x 与 /data00/home/x 不匹配
        allowed_roots = self.settings.project_allowed_roots
        if allowed_roots:
            real_path = os.path.realpath(expanded_path)
            norm_path = os.path.normpath(expanded_path)
            if not any(
                real_path.startswith(os.path.realpath(root))
                or norm_path.startswith(os.path.normpath(root))
                for root in allowed_roots
            ):
                return False, f"目录不在允许范围内: {expanded_path}"

        if os.path.isdir(expanded_path):
            with self.ctx.working_dir_lock:
                self.ctx.working_dirs[chat_id] = expanded_path
                if len(self.ctx.working_dirs) > 500:
                    self.ctx.working_dirs.popitem(last=False)  # 淘汰最旧条目
            return True, expanded_path
        else:
            return False, f"目录不存在: {expanded_path}"

    # ------------------------------------------------------------------
    # Request-id / ref-note helpers
    # ------------------------------------------------------------------
    def ensure_request_id(
        self, message_id: Optional[str], chat_id: Optional[str] = None, project_id: Optional[str] = None
    ) -> Optional[str]:
        if not message_id:
            return None
        rid = None
        try:
            rid = self.ctx.message_linker.get_request_id(message_id)
        except Exception as e:
            logger.debug("Failed to get existing request_id for message %s: %s", message_id, get_error_detail(e))
            rid = None
        if rid:
            return rid
        rid = uuid.uuid4().hex[:10]
        try:
            self.ctx.message_linker.register_origin(message_id, request_id=rid, chat_id=chat_id, project_id=project_id)
        except Exception as e:
            logger.warning("Failed to register origin for message %s: %s", message_id, e, exc_info=True)
        return rid

    def format_ref_note(
        self, origin_message_id: Optional[str], request_id: Optional[str], run_id: Optional[str] = None
    ) -> str:
        if not self.settings.ref_note_enabled:
            return ""
        origin_message_id = origin_message_id or ""
        request_id = request_id or ""
        run_id = run_id or ""
        parts = []
        if origin_message_id:
            parts.append(f"origin={origin_message_id}")
        if request_id:
            parts.append(f"req={request_id}")
        if run_id:
            parts.append(f"run={run_id}")
        if not parts:
            return ""
        return UI_TEXT["system_ref_note_prefix"] + " • ".join(parts)

    # ------------------------------------------------------------------
    # Unified context utilities
    # ------------------------------------------------------------------
    @staticmethod
    def mode_to_context_source(mode) -> "ContextSourceMode":
        """Map InteractionMode → ContextSourceMode."""
        from ...mode import InteractionMode
        from ...project import ContextSourceMode

        mapping = {
            InteractionMode.SMART: ContextSourceMode.SMART,
            InteractionMode.COCO: ContextSourceMode.COCO,
            InteractionMode.CLAUDE: ContextSourceMode.CLAUDE,
            InteractionMode.AIDEN: ContextSourceMode.AIDEN,
            InteractionMode.CODEX: ContextSourceMode.CODEX,
            InteractionMode.GEMINI: ContextSourceMode.GEMINI,
            InteractionMode.TRAEX: ContextSourceMode.TRAEX,
            InteractionMode.TTADK: ContextSourceMode.TTADK,
            InteractionMode.TUI2ACP: ContextSourceMode.TUI2ACP,
        }
        return mapping.get(mode, ContextSourceMode.SMART)

    def record_mode_transition(self, project_id: str, from_mode, to_mode, reason: str = "", *, chat_id: str = ""):
        """Record a mode switch into the unified context and build a bridge summary."""
        from_source = self.mode_to_context_source(from_mode)
        to_source = self.mode_to_context_source(to_mode)
        logger.info(
            "[模式切换] project=%s: %s -> %s, reason=%s", project_id, from_source.value, to_source.value, reason
        )
        self.ctx.context_manager.update_context(
            project_id,
            mode_transition={
                "from_mode": from_source.value,
                "to_mode": to_source.value,
                "reason": reason,
            },
            chat_id=chat_id,
        )
        ctx = self.ctx.context_manager.store.get(project_id, chat_id=chat_id)
        if ctx:
            ctx.build_bridge_summary(from_source, to_source)
            ctx.create_version(
                reason=f"mode_transition: {from_source.value} -> {to_source.value}",
                source_mode=from_source,
            )

    def inject_bridge_context(self, text: str, project: Optional["ProjectContext"], *, chat_id: str = "") -> str:
        """Consume and prepend bridge summary to the user prompt (one-shot)."""
        if not project:
            return text
        ctx = self.ctx.context_manager.store.get(project.project_id, chat_id=chat_id)
        if not ctx:
            return text
        bridge = ctx.consume_bridge_summary()
        if not bridge:
            return text
        injection = bridge.to_injection_prompt()
        if not injection:
            return text
        logger.info(
            "[Bridge注入] project=%s: %s -> %s, 注入 %d 字符到 prompt",
            project.project_name,
            bridge.from_mode.value,
            bridge.to_mode.value,
            len(injection),
        )
        return f"{injection}\n\n{text}"

    # ------------------------------------------------------------------
    # Shared engine callback factories
    # ------------------------------------------------------------------
    def create_rate_limit_callback(
        self,
        chat_id: str,
        message_id: str,
        project: Optional["ProjectContext"],
        engine_name: str,
        request_id: Optional[str] = None,
    ):
        """Create a reusable _on_rate_limit callback for any engine handler."""
        from ...card import CardBuilder

        def _on_rate_limit(wait_seconds: int):
            try:
                msg_type, card_content = CardBuilder.build_info_card(
                    project=project,
                    title=UI_TEXT["system_rate_limit_title"],
                    content=UI_TEXT["system_rate_limit_content"].format(wait_seconds=wait_seconds),
                    engine_name=engine_name,
                    show_buttons=False,
                )
                self.send_card_to_chat(chat_id, card_content, origin_message_id=message_id, request_id=request_id)
            except Exception as e:
                logger.error("Failed to send rate limit notification: %s", e, exc_info=True)

        return _on_rate_limit

    # ------------------------------------------------------------------
    # Engine name helper
    # ------------------------------------------------------------------
    def get_engine_name(self, chat_id: str, project_id: str | None = None) -> str:
        """Return engine display name based on unified identity mapping."""
        current_mode = self.ctx.mode_manager.get_mode(chat_id, project_id=project_id)
        project = None
        if project_id:
            try:
                project = self.project_manager.get_project_for_chat(project_id, chat_id)
            except Exception:
                project = None
        identity = resolve_engine_identity(
            mode=current_mode,
            ttadk_tool_name=getattr(project, "ttadk_tool_name", None) if project else None,
            ttadk_model_name=getattr(project, "ttadk_model_name", None) if project else None,
            acp_tool_name=getattr(project, "acp_tool_name", None) if project else None,
            acp_model_name=getattr(project, "acp_model_name", None) if project else None,
        )
        return identity.engine_name

    # ------------------------------------------------------------------
    # Repo lock helper (delegated to LockHelper)
    # ------------------------------------------------------------------

    def _with_repo_lock(self, root_path, chat_id, body_func):
        return self.lock_helper._with_repo_lock(root_path, chat_id, body_func)

    def _acquire_repo_lock(self, root_path, chat_id):
        return self.lock_helper._acquire_repo_lock(root_path, chat_id)

    def _release_repo_lock(self, root_path, chat_id, repo_lock_mgr=None):
        return self.lock_helper._release_repo_lock(root_path, chat_id, repo_lock_mgr)

    def send_lock_conflict_card(
        self,
        e: "LockConflictError",
        message_id: str,
        command_text: str,
        *,
        retry_count: int = 0,
        chat_id: str = "",
    ):
        return self.lock_helper.send_lock_conflict_card(
            e,
            message_id,
            command_text,
            retry_count=retry_count,
            chat_id=chat_id,
        )

    def _collect_lock_conflict_context(self, e):
        return self.lock_helper._collect_lock_conflict_context(e)

    def send_chat_lock_intercept_card(self, message_id, chat_id, chat_lock_manager):
        return self.lock_helper.send_chat_lock_intercept_card(message_id, chat_id, chat_lock_manager)

    def send_chat_lock_throttled_reply(self, message_id, chat_id, chat_lock_manager):
        return self.lock_helper.send_chat_lock_throttled_reply(message_id, chat_id, chat_lock_manager)
