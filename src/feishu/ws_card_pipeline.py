"""Card callback advisory dispatch and async action pipeline for FeishuWSClient."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from lark_channel.event.callback.model.p2_card_action_trigger import (
        P2CardActionTrigger,
        P2CardActionTriggerResponse,
    )

    from ..trust.models import EffectiveTrust

from ..tasking import (
    TaskPriority,
    TaskQueueFullError,
    TaskSpec,
)
from ..thread import (
    get_current_tenant_key,
    set_current_tenant_key,
)
from ..trust.models import TrustZone
from ..trust.registry import ManagedGroupRegistry
from ..utils.circuit_breaker import CircuitBreakerOpenException
from ..utils.errors import get_error_detail
from ..utils.rate_limit import RateLimitExceededException
from .ws_card_action_handler import (
    _extract_behavior_value,
    bind_managed_trust_revisions,
)

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("ghostap.audit")

_MAX_PENDING_CARD_ADVISORIES = 128


def _ws_client_module():
    """Return the ws_client module for runtime-patchable symbols."""
    from . import ws_client as _ws_mod

    return _ws_mod


class _CardAdvisoryDispatcher:
    """Bound callback-scoped advisory work until the typed handler returns.

    The SDK callback adapter opens one thread-local scope, calls the typed card
    handler, and only then drains the captured scheduler submissions.  A
    process-wide reservation count keeps concurrent callback scopes bounded.
    """

    def __init__(self, *, max_pending: int) -> None:
        self._max_pending = max_pending
        self._pending_count = 0
        self._closed = False
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._active: contextvars.ContextVar[
            list[Callable[[], object]] | None
        ] = contextvars.ContextVar(
            f"ghostap_card_advisory_active_{id(self)}",
            default=None,
        )
        self._sealed: contextvars.ContextVar[
            tuple[Callable[[], object], ...] | None
        ] = contextvars.ContextVar(
            f"ghostap_card_advisory_sealed_{id(self)}",
            default=None,
        )

    def begin(self) -> list[Callable[[], object]]:
        """Open a callback-scoped advisory capture list."""
        if self._active.get() is not None or self._sealed.get() is not None:
            raise RuntimeError("nested card advisory scope")
        pending: list[Callable[[], object]] = []
        self._active.set(pending)
        return pending

    def defer(self, callback: Callable[[], object]) -> bool:
        """Capture one bounded advisory callback for post-response execution."""
        pending = self._active.get()
        if pending is None:
            return False
        with self._lock:
            if self._closed or self._pending_count >= self._max_pending:
                return False
            self._pending_count += 1
        pending.append(callback)
        return True

    def seal(self, pending: list[Callable[[], object]]) -> None:
        """Freeze the captured advisory callbacks for the current async context."""
        if self._active.get() is not pending:
            raise RuntimeError("card advisory scope identity mismatch")
        self._active.set(None)
        self._sealed.set(tuple(pending))

    def finish_response(self, *, written: bool) -> None:
        """Drain sealed advisories after the callback response write completes."""
        pending = self._sealed.get()
        if pending is None:
            return
        self._sealed.set(None)
        for callback in pending:
            with self._lock:
                self._pending_count = max(0, self._pending_count - 1)
                should_dispatch = written and not self._closed
            if should_dispatch:
                try:
                    callback()
                except Exception:
                    logger.warning(
                        "card advisory post-response handoff failed",
                        exc_info=True,
                    )

    def discard(self, pending: list[Callable[[], object]]) -> None:
        """Drop unsealed advisories when the callback raises before returning."""
        if self._active.get() is pending:
            self._active.set(None)
        with self._lock:
            self._pending_count = max(0, self._pending_count - len(pending))

    def close(self) -> None:
        """Prevent new advisory dispatch during shutdown."""
        with self._lock:
            self._closed = True


class WSCardPipelineMixin:
    """Mixin handling card advisory dispatch, action execution, and bot deletion."""

    def _refresh_managed_card_revisions(
        self,
        message_id: str,
        chat_id: str,
        trust: EffectiveTrust,
    ) -> bool:
        """Refresh a stale managed card without dispatching its submitted action."""

        if (
            not message_id
            or trust.zone is not TrustZone.MANAGED_AGENT_GROUP
            or trust.managed_group is None
            or trust.managed_group.chat_id != chat_id
            or not self._current_trust_can_dispatch(trust)
        ):
            return False
        try:
            from lark_oapi.api.im.v1 import GetMessageRequest

            request = (
                GetMessageRequest.builder()
                .message_id(message_id)
                .user_id_type("open_id")
                .card_msg_content_type("user_card_content")
                .build()
            )
            response = self._get_api_client().im.v1.message.get(request)
            if not response or not response.success() or response.data is None:
                return False
            items = getattr(response.data, "items", None)
            if not isinstance(items, (list, tuple)) or len(items) != 1:
                return False
            item = items[0]
            if (
                getattr(item, "message_id", None) != message_id
                or getattr(item, "chat_id", None) != chat_id
            ):
                return False
            content = getattr(getattr(item, "body", None), "content", None)
            if isinstance(content, str):
                content = json.loads(content)
            if not isinstance(content, dict):
                return False
            refreshed = bind_managed_trust_revisions(
                content,
                group_revision=trust.group_revision,
                grant_revision=trust.grant_revision,
            )
            return (
                self._handler_ctx.handlers["system"].update_card(
                    message_id, refreshed
                )
                is True
            )
        except (AttributeError, RuntimeError, OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("managed stale card refresh failed closed", exc_info=True)
            return False

    def _submit_card_advisory_follow_up(
        self,
        *,
        chat_id: str,
        message_id: str,
        name: str,
        callback: Callable[[], object],
        project_id: str | None = None,
        tenant_key: str = "",
    ) -> bool:
        """Submit advisory remote I/O after the typed callback has returned.

        These jobs use the scheduler's reserved system lane, but they never
        change the allow/deny result of the submitted card action.  A fenced
        or full lane therefore drops only the advisory work and must not fall
        back to synchronous network I/O on the three-second callback path.
        """

        def _run(_task_ctx):
            previous_tenant_key = get_current_tenant_key()
            set_current_tenant_key(tenant_key or None)
            try:
                return callback()
            finally:
                set_current_tenant_key(previous_tenant_key)

        spec = TaskSpec(
            chat_id=chat_id,
            name=name,
            task_type="card_advisory_follow_up",
            message_id=message_id,
            origin_message_id=message_id,
            project_id=project_id,
            priority=TaskPriority.HIGH,
            is_system_command=True,
            tenant_key=tenant_key,
        )
        try:
            self._scheduler.submit(spec, _run)
            return True
        except (
            TaskQueueFullError,
            RateLimitExceededException,
            CircuitBreakerOpenException,
            RuntimeError,
        ) as exc:
            logger.warning(
                "card advisory follow-up dropped: name=%s chat=%s reason=%s",
                name,
                self._access_identifier_hash(chat_id),
                get_error_detail(exc),
            )
            return False

    def _enqueue_card_advisory_follow_up(
        self,
        *,
        chat_id: str,
        message_id: str,
        name: str,
        callback: Callable[[], object],
        project_id: str | None = None,
        tenant_key: str = "",
    ) -> bool:
        """Capture one bounded advisory for the outer SDK callback adapter."""

        dispatcher = getattr(self, "_card_advisory_dispatcher", None)
        if dispatcher is None:
            logger.warning("card advisory follow-up dropped without callback scope")
            return False
        accepted = dispatcher.defer(
            lambda: self._submit_card_advisory_follow_up(
                chat_id=chat_id,
                message_id=message_id,
                name=name,
                callback=callback,
                project_id=project_id,
                tenant_key=tenant_key,
            )
        )
        if not accepted:
            logger.warning(
                "card advisory follow-up dropped before handoff: name=%s chat=%s",
                name,
                self._access_identifier_hash(chat_id),
            )
        return accepted

    def _handle_card_action_callback(
        self,
        data: P2CardActionTrigger,
    ) -> P2CardActionTriggerResponse:
        """SDK adapter that hands advisories off after typed ACK creation."""

        dispatcher = getattr(self, "_card_advisory_dispatcher", None)
        if dispatcher is None:
            ws_mod = _ws_client_module()
            dispatcher_cls = getattr(ws_mod, "_CardAdvisoryDispatcher", _CardAdvisoryDispatcher)
            max_pending = getattr(ws_mod, "_MAX_PENDING_CARD_ADVISORIES", _MAX_PENDING_CARD_ADVISORIES)
            dispatcher = dispatcher_cls(max_pending=max_pending)
            self._card_advisory_dispatcher = dispatcher
        pending = dispatcher.begin()
        try:
            response = self._handle_card_action(data)
        except BaseException:
            dispatcher.discard(pending)
            raise
        # Seal the batch in this async Context.  ObservedLarkWSClient releases
        # it only after the typed ACK bytes have been written successfully.
        dispatcher.seal(pending)
        return response

    def _finish_card_advisories(self, written: bool) -> None:
        """Flush or discard sealed advisory callbacks after response write."""
        dispatcher = getattr(self, "_card_advisory_dispatcher", None)
        if dispatcher is not None:
            dispatcher.finish_response(written=written)

    def _resolve_card_is_p2p(
        self,
        *,
        origin_message_id: str | None,
        open_chat_id: str,
        operator_id: str,
    ) -> bool:
        """Resolve card DM provenance without trusting callback payload fields.

        Card callbacks do not carry a structural chat type, so only metadata
        captured from ``im.message.receive_v1`` may grant P2P provenance. The
        callback's three-second response path must not perform a remote Chat
        API lookup; a missing or mismatched local origin is denied.
        """
        if not origin_message_id:
            return False
        try:
            origin = self._message_linker.query(origin_message_id)
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            logger.warning("failed to query card origin provenance", exc_info=True)
            return False

        if origin is not None:
            stored_origin_message_id = origin.get("origin_message_id")
            origin_chat_id = origin.get("chat_id")
            origin_sender_id = origin.get("sender_id")
            origin_chat_type = origin.get("chat_type")
            if not all(
                isinstance(value, str) and bool(value)
                for value in (
                    stored_origin_message_id,
                    origin_chat_id,
                    origin_sender_id,
                    origin_chat_type,
                )
            ):
                logger.warning("incomplete card origin provenance: chat=%s", str(open_chat_id)[:12])
                return False
            if stored_origin_message_id != origin_message_id:
                logger.warning("card origin identity mismatch: chat=%s", str(open_chat_id)[:12])
                return False
            if origin_chat_id != open_chat_id:
                logger.warning(
                    "card origin chat mismatch: origin_chat=%s callback_chat=%s",
                    str(origin_chat_id)[:12],
                    str(open_chat_id)[:12],
                )
                return False

            if origin_sender_id != operator_id:
                logger.warning("card origin operator mismatch: chat=%s", str(open_chat_id)[:12])
                return False
            return origin_chat_type == "p2p"

        return False

    def _get_chat_mode(self, chat_id: str) -> str | None:
        """Read structural chat mode from the official Chat API.

        ``chat_type`` on this API is group visibility (private/public), so it
        must never participate in DM authorization.
        """
        if not chat_id or chat_id == "unknown":
            return None
        try:
            from lark_oapi.api.im.v1 import GetChatRequest

            request = GetChatRequest.builder().chat_id(chat_id).build()
            response = self._get_api_client().im.v1.chat.get(request)
            if not response or not response.success() or not response.data:
                logger.warning(
                    "failed to resolve chat mode: chat=%s code=%s",
                    chat_id[:12],
                    getattr(response, "code", None),
                )
                return None
            chat_mode = getattr(response.data, "chat_mode", None)
            return chat_mode if chat_mode in {"p2p", "group", "topic"} else None
        except Exception:
            logger.warning("chat mode lookup failed: chat=%s", chat_id[:12], exc_info=True)
            return None

    def _process_card_action_async(
        self,
        data: Any,
        task_ctx=None,
        effective_trust: EffectiveTrust | None = None,
    ):
        """卡片动作处理逻辑（第二阶段实现）。

        该方法会把 `action.value` normalize 为 dict，提取 `action/project_id`，并通过
        `ActionDispatcher` 做 exact/prefix 路由。
        """
        from ..thread import (
            set_current_is_p2p,
            set_current_sender_id,
            set_current_sender_name,
            set_current_sender_union_id,
            set_current_tenant_key,
            set_current_thread_id,
        )

        # Scheduler workers may retain a ContextVar copied from an earlier task.
        # A card callback must start unscoped and restore only payload context
        # that resolves to the callback's authoritative chat/project coordinates.
        set_current_thread_id(None)
        try:
            start_time = time.perf_counter()
            action = data.event.action
            # Schema 2.0: prefer behaviors[0].value, fallback to legacy action.value
            value_raw = action.value
            behaviors = getattr(action, "behaviors", None)
            if isinstance(behaviors, list) and behaviors:
                first_behavior = behaviors[0]
                behavior_value = _extract_behavior_value(first_behavior)
                if behavior_value is not None:
                    value_raw = behavior_value
            operator = data.event.operator
            open_message_id = data.event.context.open_message_id
            open_chat_id = data.event.context.open_chat_id

            # sender_id is carried in task_ctx.spec (set at submit time);
            # fall back to event operator extraction only when task_ctx is unavailable.
            _operator_id = (
                task_ctx.spec.sender_id
                if task_ctx and hasattr(task_ctx, "spec") and task_ctx.spec.sender_id
                else (
                    getattr(operator, "open_id", None)
                    or getattr(operator, "user_id", None)
                    or getattr(operator, "union_id", None)
                    or ""
                )
            )
            _card_is_p2p = (
                task_ctx.spec.is_p2p
                if task_ctx and hasattr(task_ctx, "spec")
                else getattr(getattr(data.event, "context", None), "chat_type", None) == "p2p"
            )
            current_trust = self._resolve_effective_trust(
                sender_id=_operator_id,
                chat_id=open_chat_id,
                chat_type=("p2p" if _card_is_p2p else "group"),
            )
            current_decision = self._managed_trust_access_decision(current_trust)
            if current_decision is not None and not current_decision.allowed:
                return
            if effective_trust is not None and current_trust != effective_trust:
                audit_logger.warning(
                    "MANAGED_CARD_CURRENT_REVISION_DENIED chat_hash=%s",
                    self._access_identifier_hash(open_chat_id),
                )
                if (
                    current_trust is not None
                    and current_trust.zone is TrustZone.MANAGED_AGENT_GROUP
                ):
                    self._refresh_managed_card_revisions(
                        open_message_id,
                        open_chat_id,
                        current_trust,
                    )
                return
            set_current_sender_id(_operator_id)
            _operator_union_id = (
                task_ctx.spec.sender_union_id
                if task_ctx and hasattr(task_ctx, "spec")
                else (getattr(operator, "union_id", None) or "")
            )
            set_current_sender_union_id(_operator_union_id or None)
            from .user_cache import resolve_display_name_nonblocking as _resolve_name
            _op_name = _resolve_name(_operator_id, self._get_api_client) if _operator_id else ""
            set_current_sender_name(_op_name or (_operator_id[:8] if _operator_id else ""))
            set_current_is_p2p(_card_is_p2p)
            _tenant_key = (
                task_ctx.spec.tenant_key
                if task_ctx and hasattr(task_ctx, "spec")
                else ""
            )
            set_current_tenant_key(_tenant_key or None)

            logger.debug(
                "卡片回调上下文: operator_open_id=%s, operator_user_id=%s, value_raw_type=%s",
                getattr(operator, "open_id", None),
                getattr(operator, "user_id", None),
                type(value_raw).__name__,
            )

            if isinstance(value_raw, dict):
                value = dict(value_raw)
            elif isinstance(value_raw, str):
                try:
                    value = json.loads(value_raw)
                    if not isinstance(value, dict):
                        value = {"action": value_raw}
                except (json.JSONDecodeError, TypeError):
                    logger.warning("卡片 value 解析失败: value_raw=%s", value_raw[:500])
                    value = {"action": value_raw}
            else:
                value = {"action": str(value_raw)}

            # --- 注入交互组件的额外返回值 ---
            try:
                if getattr(action, "option", None) is not None:
                    value["_option"] = action.option
                if getattr(action, "options", None) is not None:
                    value["_options"] = action.options
                if getattr(action, "form_value", None) is not None:
                    value["_form_value"] = action.form_value
                if getattr(action, "input_value", None) is not None:
                    value["_input_value"] = action.input_value
            except Exception:
                logger.debug("failed to extract action input_value", exc_info=True)

            action_type = value.get("action", "")
            project_id = (
                current_trust.managed_group.project_id
                if current_trust is not None
                and current_trust.managed_group is not None
                else value.get("project_id", "")
            )
            if project_id:
                value["project_id"] = project_id

            card_thread_id = value.get("thread_root_id")
            if (
                isinstance(card_thread_id, str)
                and card_thread_id
                and self.settings.thread_programming_enabled
            ):
                thread_ctx = self._thread_manager.get(card_thread_id)
                if (
                    thread_ctx is not None
                    and thread_ctx.chat_id == open_chat_id
                    and thread_ctx.project_id == project_id
                ):
                    set_current_thread_id(thread_ctx.thread_root_id)

            if task_ctx and project_id:
                try:
                    self._scheduler.update_project_id(task_ctx.run_id, project_id)
                except Exception as e:
                    logger.debug("update_project_id失败(card_action): run_id=%s, err=%s", task_ctx.run_id, get_error_detail(e))

            logger.info(
                "卡片按钮点击: action=%s, project_id=%s, value_keys=%s",
                action_type,
                project_id,
                list(value.keys()),
            )

            # --- Chat lock interception for card actions (fail-close) ---
            if self._chat_lock_gate.check_card_action(
                open_chat_id, _operator_id, open_message_id,
                action_type=action_type,
            ):
                return

            # --- Exact dispatch against the authoritative action table ---
            if not self._current_trust_can_dispatch(current_trust):
                return
            handler = self._action_handlers.get(action_type)
            if handler is None:
                logger.warning("未注册的卡片动作: action=%s, message_id=%s", action_type, open_message_id)
                self._handler_ctx.handlers["coco"].reply_text(open_message_id, f"⚠️ 未识别的操作: {action_type}")
            else:
                handler(open_message_id, open_chat_id, project_id, value)

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.debug("卡片回调处理耗时: %dms", elapsed_ms)

        except asyncio.TimeoutError as e:
            logger.warning("处理卡片动作超时: %s", get_error_detail(e))
            _mid = locals().get("open_message_id", "unknown")
            _action = locals().get("action_type", "unknown")
            try:
                if _mid != "unknown":
                    self._handler_ctx.handlers["coco"].reply_text(_mid, f"⏳ 操作超时 ({_action}): {get_error_detail(e)}")
            except Exception:
                logger.debug("failed to reply timeout action error", exc_info=True)
        except Exception as e:
            logger.error("处理卡片动作异常: %s", e, exc_info=True)
            # 发送错误提示给用户
            _mid = locals().get("open_message_id", "unknown")
            _cid = locals().get("open_chat_id", "unknown")
            _action = locals().get("action_type", "unknown")
            try:
                if _mid != "unknown":
                    self._handler_ctx.handlers["coco"].reply_text(_mid, f"❌ 操作失败 ({_action}): {get_error_detail(e)}")
            except Exception:
                logger.debug("failed to reply action failure error", exc_info=True)
        finally:
            set_current_thread_id(None)
            set_current_sender_id(None)
            set_current_sender_union_id(None)
            set_current_sender_name("")
            set_current_is_p2p(False)
            set_current_tenant_key(None)

    def _handle_bot_deleted(self, data):
        """Durably revoke trust before retiring a remotely deleted group."""

        event = getattr(data, "event", None)
        chat_id = getattr(event, "chat_id", "")
        if not isinstance(chat_id, str) or not chat_id.startswith("oc_"):
            return
        try:
            registry = getattr(self, "_managed_group_registry", None)
            if type(registry) is not ManagedGroupRegistry:
                return
            record = registry.record(chat_id)
            revoke_required = (
                record is not None
                and getattr(record.status, "value", "") == "active"
            )
            if revoke_required:
                registry.begin_revoke(
                    chat_id,
                    requested_at=datetime.now(UTC),
                )
            project_manager = getattr(self, "_project_manager", None)
            if project_manager is None or not project_manager.revoke_managed_chat(chat_id):
                raise OSError("project managed-chat revocation was not persisted")
            if revoke_required:
                registry.complete_revoke(chat_id)
        except Exception:
            logger.exception(
                "failed to revoke deleted managed group chat=%s",
                chat_id[:12],
            )
            raise
