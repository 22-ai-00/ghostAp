"""Slock Engine handler — multi-Agent mouthpiece collaboration engine."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import threading
from datetime import UTC, datetime
from numbers import Real
from typing import TYPE_CHECKING, Optional

from ...acp.helper import fetch_acp_models
from ...autonomous.team import TeamAdmissionError, TeamTarget
from ...card import CardBuilder
from ...card.actions import dispatch as action_ids
from ...card.shared import build_responsive_layout
from ...model_selection import is_default_model_option
from ...slock_engine.exceptions import ExecutorQueueFullError, TaskQueueFullError
from ...slock_engine.models import SlockChannel
from ...slock_engine.slash_commands import (
    SlockCommandAction,
    is_slock_command,
    parse_slock_command,
)
from ...utils.errors import safe_error_message
from ...utils.redact import redact_sensitive
from ..emoji import EmojiReaction
from ..user_cache import resolve_display_name_nonblocking
from .base import CardActionContext
from .engine_base import BaseEngineHandler

if TYPE_CHECKING:
    from ...project import ProjectContext
    from ..handler_context import HandlerContext

logger = logging.getLogger(__name__)

_EXISTING_HIRE_APP_ID_RE = re.compile(r"cli_[A-Za-z0-9_-]{3,128}")
_EXPLICIT_TEAM_TASK_RE = re.compile(
    r"(?:实现|修复|优化|排查|分析|编写|补充|测试|重构|部署|发布|更新|修改|"
    r"创建|生成|完成|执行|运行|评审|审查|检查|迁移|设计|开发|整理|总结|"
    r"自我介绍|做个.{0,8}介绍|"
    r"\b(?:implement|fix|optimize|debug|analy[sz]e|write|test|refactor|deploy|"
    r"review|create|generate|migrate|design|develop)\b)",
    re.IGNORECASE,
)
_SLOCK_CONTROL_INTENT_RE = re.compile(
    r"^(?:(?:请|麻烦)(?:帮我)?|帮我)?(?:创建|新建)(?:一个|个|一名)?"
    r"(?:角色|员工|团队|任务)(?:吧|吗)?[！!。.?？]?$|"
    r"^(?:please\s+)?(?:create|new)\s+(?:a\s+)?"
    r"(?:role|employee|team|task)[.!?]?$",
    re.IGNORECASE,
)
_NON_IMPERATIVE_TEAM_TASK_RE = re.compile(
    r"^(?:如何|怎么|怎样|为什么|是否|能否|可否|请问|不要|别|无需|不必|先别)|"
    r"(?:吗|么|了吗|好了没)[？?]?$",
    re.IGNORECASE,
)
_LARK_MD_INLINE_SPECIAL_RE = re.compile(r"([\\`*_~\[\]()<>|#!])")


def _normalize_existing_hire_app_id(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("existing hire App ID must be text")
    normalized = value.strip()
    if normalized and _EXISTING_HIRE_APP_ID_RE.fullmatch(normalized) is None:
        raise ValueError("invalid existing hire App ID")
    return normalized


def _positive_seconds(value: object, default: float) -> float:
    """Return a positive timeout value, falling back for mocks or invalid config."""
    if isinstance(value, bool):
        return default
    if isinstance(value, Real):
        seconds = float(value)
    elif isinstance(value, str):
        try:
            seconds = float(value.strip())
        except ValueError:
            return default
    else:
        return default
    return seconds if seconds > 0 else default


def _confidence_threshold(value: object, default: float) -> float:
    """Return a valid confidence threshold in [0, 1], falling back for mocks."""
    if isinstance(value, bool):
        return default
    if isinstance(value, Real):
        threshold = float(value)
    elif isinstance(value, str):
        try:
            threshold = float(value.strip())
        except ValueError:
            return default
    else:
        return default
    return threshold if 0.0 <= threshold <= 1.0 else default


def _escape_lark_md_inline(value: object) -> str:
    """Render one dynamic value as a bounded, single-line Markdown literal."""

    normalized = " ".join(str(value or "").splitlines()).strip()
    return _LARK_MD_INLINE_SPECIAL_RE.sub(r"\\\1", normalized)


def _membership_is_degraded(value: object) -> bool:
    """Fail closed unless membership health is the exact boolean ``False``."""

    return value is not False


def _get_nli_loop():
    """Return the shared loop used for async NLI classification."""
    from src.utils.async_helpers import _get_bridge_loop

    return _get_bridge_loop()


from .slock_roles import SlockRoleMixin  # noqa: E402
from .slock_tasks import SlockTaskMixin  # noqa: E402


class SlockHandler(SlockRoleMixin, SlockTaskMixin, BaseEngineHandler):
    """Manages the full lifecycle of Slock Engine (multi-Agent mouthpiece) tasks."""

    def __init__(self, ctx: "HandlerContext") -> None:
        super().__init__(ctx)
        # Rate-limit tracker: key = "chat_id:sender_id" → list of timestamps
        self._rate_limit_tracker: dict[str, list[float]] = {}
        # Singleton IntentRouter (AC-10: only instantiated once per handler lifecycle)
        from src.slock_engine.intent_router import IntentRouter

        # Use the bounded shared compute pool for one-shot ACP classification.
        from src.utils.thread_pools import get_compute_pool

        self._nli_executor = get_compute_pool()
        self._intent_router = IntentRouter(
            confidence_threshold=_confidence_threshold(
                getattr(ctx.settings, "slock_nli_confidence_threshold", 0.7), 0.7
            ),
            timeout=_positive_seconds(getattr(ctx.settings, "slock_nli_timeout", 5), 5.0),
            executor=self._nli_executor,
        )

    def cleanup(self) -> None:
        """Release resources held by this handler instance.

        Called by the dispatcher when the handler is being rebuilt or shut down.
        The NLI executor is shared (compute pool) so we don't shut it down here.
        """
        pass

    # ------------------------------------------------------------------
    # Reply mode override: Slock uses slock_reply_mode (default "direct")
    # ------------------------------------------------------------------

    def reply_text(self, message_id: str, text: str, *, reply_in_thread: Optional[bool] = None) -> Optional[str]:
        """Override: default to slock_reply_mode instead of default_reply_mode."""
        if reply_in_thread is None:
            reply_in_thread = self.ctx.settings.slock_reply_mode == "thread"
        return super().reply_text(message_id, text, reply_in_thread=reply_in_thread)

    def reply_card(self, message_id: str, card_content: str, *, reply_in_thread: Optional[bool] = None) -> Optional[str]:
        """Override: default to slock_reply_mode instead of default_reply_mode."""
        if reply_in_thread is None:
            reply_in_thread = self.ctx.settings.slock_reply_mode == "thread"
        return super().reply_card(message_id, card_content, reply_in_thread=reply_in_thread)

    # ------------------------------------------------------------------
    # BaseEngineHandler abstract method implementations
    # ------------------------------------------------------------------

    def _get_engine_manager(self):
        return self.ctx.slock_engine_manager

    def _get_global_registry(self):
        """Get a global AgentRegistry for /hire without requiring an active engine."""
        from ...slock_engine.agent_registry import AgentRegistry
        if not hasattr(self, '_global_registry'):
            self._global_registry = AgentRegistry.legacy()
        return self._global_registry

    def _get_engine_name_prefix(self) -> str:
        return "Slock"

    def _get_task_type(self) -> str:
        return "slock_engine"

    def _show_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        self.show_slock_status(message_id, chat_id, project)

    def _create_callbacks(
        self, message_id: str, chat_id: str, project: Optional["ProjectContext"], engine_name: str, root_path: str
    ):
        from ...slock_engine.card_channel import SlockCardChannel
        from ...slock_engine.engine import SlockEngineCallbacks

        # Unified card channel — routes through CardDelivery for payload guard + retry
        _slock_thread = self.ctx.settings.slock_reply_mode == "thread"
        channel = SlockCardChannel(self.get_card_delivery(), chat_id, reply_in_thread=_slock_thread)

        def on_agent_wake(agent):
            logger.debug("Slock agent waking: %s in chat %s", agent.name, chat_id)

        def on_agent_running(agent, msg):
            logger.debug("Slock agent running: %s task=%s", agent.name, msg[:80])

        def on_agent_done(agent, result):
            logger.debug("Slock agent done: %s result_len=%d", agent.name, len(result))
            # Broadcast agent result as identity card when broadcast mode enabled
            if getattr(self.ctx.settings, 'slock_discussion_broadcast_rounds', True) and result.strip():
                from ...slock_engine.mouthpiece import Mouthpiece
                mouthpiece = Mouthpiece()
                try:
                    card = mouthpiece.format_card(agent, result, channel_id=chat_id)
                    channel.send_card(card)
                except Exception as bcast_exc:
                    logger.debug("on_agent_done broadcast failed: %s", bcast_exc)

        def on_error(err_msg):
            logger.error("Slock engine error in chat %s: %s", chat_id, err_msg)

        def on_escalation(esc):
            """Send escalation card to chat and write back message_id."""
            manager = self._get_engine_manager()
            engine = manager.get_active_engine(chat_id)
            if not engine and hasattr(manager, "get_activated_engine"):
                engine = manager.get_activated_engine(chat_id)
            if not engine:
                logger.warning("on_escalation: engine not found for chat %s", chat_id)
                return
            card = engine.get_escalation_card(esc)
            if not card:
                logger.warning("on_escalation: failed to build card for esc %s", esc.escalation_id)
                return
            sent_msg_id = channel.send_card(card, reply_to=message_id)
            if sent_msg_id:
                esc.card_message_id = sent_msg_id
            else:
                logger.warning("on_escalation: send_card_to_chat returned None for esc %s", esc.escalation_id)

        def on_card_send(card):
            """Send a card via unified channel (payload guard + retry)."""
            return channel.send_card(card, reply_to=message_id)

        def on_card_update(msg_id, card):
            """Update an existing card via unified channel."""
            channel.update_card(msg_id, card)

        return SlockEngineCallbacks(
            on_agent_wake=on_agent_wake,
            on_agent_running=on_agent_running,
            on_agent_done=on_agent_done,
            on_escalation=on_escalation,
            on_error=on_error,
            on_card_send=on_card_send,
            on_card_update=on_card_update,
        )

    # ------------------------------------------------------------------
    # Command router
    # ------------------------------------------------------------------

    def handle_slock_command(
        self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None
    ):
        """Route slock-related commands to the appropriate handler method."""
        cmd = parse_slock_command(text)
        is_hire = text.strip().lower().startswith("/hire") or text.strip().lower().startswith("/h ")

        dispatch: dict[SlockCommandAction, callable] = {
            SlockCommandAction.ACTIVATE: lambda: self.activate_slock(message_id, chat_id, cmd.args, project),
            SlockCommandAction.STATUS: lambda: self.show_slock_status(message_id, chat_id, project),
            SlockCommandAction.STOP: lambda: self.stop_slock_engine(message_id, chat_id, project),
            SlockCommandAction.HELP: lambda: self.show_slock_help(message_id),
            SlockCommandAction.NEW_TEAM: lambda: self.create_team(message_id, chat_id, cmd.args, project),
            SlockCommandAction.TEAM_LIST: lambda: self.list_teams(message_id, chat_id, project),
            SlockCommandAction.TEAM_STATUS: lambda: self.show_team_status(message_id, chat_id, cmd.target, project),
            SlockCommandAction.TEAM_DISSOLVE: lambda: self.dissolve_team(message_id, chat_id, cmd.target, project),
            SlockCommandAction.NEW_ROLE: lambda: self.create_role(message_id, chat_id, cmd.args, project, global_hire=is_hire),
            SlockCommandAction.FIRE_EMPLOYEE: lambda: self.fire_employee(message_id, chat_id, cmd.args),
            SlockCommandAction.ROLE_LIST: lambda: self.list_roles(message_id, chat_id, project),
            SlockCommandAction.ROLE_REMOVE: lambda: self.remove_role(message_id, chat_id, cmd.target, project),
            SlockCommandAction.ROLE_MOVE: lambda: self.move_role(message_id, chat_id, cmd.target, cmd.args, project),
            SlockCommandAction.ROLE_ADD: lambda: self.add_role_to_group(message_id, chat_id, cmd.target, project),
            SlockCommandAction.ROLE_INFO: lambda: self.show_role_info(message_id, chat_id, cmd.target, project),
            SlockCommandAction.TASK_LIST: lambda: self.list_tasks(message_id, chat_id, project),
            SlockCommandAction.TASK_STATUS: lambda: self.show_task_status(message_id, chat_id, project),
            SlockCommandAction.TASK_ASSIGN: lambda: self.assign_task(message_id, chat_id, cmd.args, cmd.target, project),
            SlockCommandAction.DISCUSSION: lambda: self._trigger_nli_discussion(message_id, chat_id, cmd.args, {}, project),
            SlockCommandAction.STOP_DISCUSSION: lambda: self.stop_discussion(message_id, chat_id, project),
            SlockCommandAction.DISCUSSION_HISTORY: lambda: self.show_discussion_history(message_id, chat_id, cmd.target, project),
            SlockCommandAction.DISCUSSION_LIST: lambda: self.list_discussions(message_id, chat_id, project),
            SlockCommandAction.COUNCIL: lambda: self.run_council(message_id, chat_id, cmd.args, project),
            SlockCommandAction.MEMORY: lambda: self.show_agent_memory(message_id, chat_id, cmd.target, project),
            SlockCommandAction.MEMORY_LIST: lambda: self.show_memory_list(message_id, chat_id, project),
            SlockCommandAction.MEMORY_GROUP: lambda: self.show_memory_group(message_id, chat_id, project),
            SlockCommandAction.EMPLOYEE_HISTORY: lambda: self.show_employee_history(
                message_id, chat_id, cmd.target
            ),
            SlockCommandAction.PLAN_LIST: lambda: self.list_plans(message_id, chat_id, project),
            SlockCommandAction.PLAN_DETAIL: lambda: self.show_plan_detail(message_id, chat_id, cmd.target, project),
            SlockCommandAction.EMPLOYEE_LIST: lambda: self.list_employees_roster(message_id, chat_id),
        }

        handler = dispatch.get(cmd.action)
        if handler:
            handler()
        else:
            self.show_slock_help(message_id)

    # ------------------------------------------------------------------
    # Async execution helper (shared by handle_message & _submit_task_execution)
    # ------------------------------------------------------------------

    def _execute_async(
        self,
        *,
        engine,
        execute_fn,
        placeholder_card_json: str,
        result_card_fn,
        error_card_fn,
        empty_card_fn,
        busy_card_fn,
        message_id: str,
        chat_id: str,
    ) -> None:
        """Submit async execution with placeholder→update card pattern.

        This method encapsulates the common pattern:
          1. Send placeholder card immediately
          2. Submit execute_fn to BoundedExecutor
          3. On success: update card via result_card_fn(result, duration)
          4. On empty result: update via empty_card_fn()
          5. On exception: update via error_card_fn(exception)
          6. On queue full: update via busy_card_fn()
          7. On queue wait timeout: update with timeout card

        Callbacks:
          execute_fn() -> Optional[str] — the actual work
          result_card_fn(result: str, duration: float) -> str — JSON card for success
          error_card_fn(exc: Exception) -> str — JSON card for error
          empty_card_fn() -> str — JSON card for empty result
          busy_card_fn() -> str — JSON card for busy/full
        """
        card_message_id = self.send_card_to_chat(
            chat_id, placeholder_card_json, origin_message_id=message_id
        )

        executor = engine._get_executor()

        def _async_work():
            import time as _time

            start_time = _time.time()

            # Check queue wait timeout
            if self._check_queue_wait_timeout(future, start_time, card_message_id, chat_id):
                return

            try:
                result = execute_fn()
            except Exception as e:
                from src.utils.errors import redact_sensitive
                logger.error("Slock _execute_async error: %s", redact_sensitive(repr(e)), exc_info=True)
                error_card = error_card_fn(e)
                if card_message_id:
                    self.update_card(card_message_id, error_card)
                return

            if result is None:
                empty_card = empty_card_fn()
                if card_message_id:
                    self.update_card(card_message_id, empty_card)
                return

            duration = _time.time() - start_time

            if not result:
                # Empty string — agent ran successfully but produced no output
                result = "✅ 处理完成"

            success_card = result_card_fn(result, duration)
            if card_message_id:
                self.update_card(card_message_id, success_card)
            else:
                self.send_card_to_chat(chat_id, success_card, origin_message_id=message_id)

        future = None  # Assigned by executor.submit(); closure reads after thread start
        try:
            future = executor.submit(_async_work)
        except (ExecutorQueueFullError, RuntimeError) as e:
            logger.warning("Slock executor submit rejected for chat %s: %s", chat_id, repr(e))
            busy_card = busy_card_fn()
            if card_message_id:
                self.update_card(card_message_id, busy_card)

    # ------------------------------------------------------------------
    # Message routing (non-command messages in slock-active chats)
    # ------------------------------------------------------------------

    def handle_message(
        self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None
    ):
        """Route a non-command message in a slock-active chat.

        Routing priority (FS-01):
        1. /task keyword → redirect to handle_slock_command
        2. @AgentName → precise route to named agent
        3. NLI intent detection → if high confidence, execute as command
        4. Normal text → engine.execute() smart routing

        NLI fallback: UNKNOWN or low confidence falls through to smart routing (FS-02).
        Execution is submitted asynchronously to the engine's thread pool.
        """
        import re

        # Priority 1: /task keyword → redirect to command handler
        if text and text.strip().lower().startswith("/task"):
            self.handle_slock_command(message_id, chat_id, text, project)
            return

        # Priority 2: @AgentName precise routing (must be before NLI per AC-01)
        from ...thread.manager import get_current_mentioned_names

        mention_names = tuple(
            dict.fromkeys(
                (*get_current_mentioned_names(), *re.findall(r"@([\w\-]+)", text or ""))
            )
        )
        at_match = re.search(r"@([\w\-]+)", text or "")
        target_agent = None

        if mention_names:
            membership = getattr(self.ctx, "employee_membership_service", None)
            team_service = getattr(self.ctx, "employee_team_service", None)
            if membership is not None and team_service is not None:
                from ...autonomous.ingress import (
                    GroupRouteKind,
                    GroupRouteRequest,
                    decide_group_route,
                )
                from ...thread.manager import (
                    get_current_sender_id,
                    get_current_tenant_key,
                )

                tenant_key = get_current_tenant_key() or ""
                requester = get_current_sender_id() or ""
                employees = []
                for name in mention_names:
                    employee = membership.find_employee_by_name(tenant_key, name)
                    if (
                        employee is not None
                        and chat_id in employee.member_groups
                        and not _membership_is_degraded(
                            membership.is_degraded(employee.agent_id, chat_id)
                        )
                    ):
                        employees.append(employee)
                employees = list(
                    {employee.agent_id: employee for employee in employees}.values()
                )
                if employees and requester and tenant_key:
                    decision = decide_group_route(
                        GroupRouteRequest(
                            tenant_key=tenant_key,
                            chat_id=chat_id,
                            sender_principal_id=requester,
                            sender_type="user",
                            sender_tenant_key=tenant_key,
                            text=text.strip(),
                            mentioned_agent_ids=tuple(
                                employee.agent_id for employee in employees
                            ),
                        )
                    )
                    if decision.kind is GroupRouteKind.TEAM_TASK:
                        self._start_visible_employee_team_task(
                            message_id=message_id,
                            chat_id=chat_id,
                            text=text.strip(),
                            team_service=team_service,
                        )
                        return
                    employee = employees[0]
                    try:
                        team_service.dispatch_direct(
                            target=TeamTarget(
                                employee.agent_id,
                                employee.name,
                                employee.role,
                                tuple(employee.capabilities),
                            ),
                            tenant_key=tenant_key,
                            chat_id=chat_id,
                            message_id=message_id,
                            requester_principal_id=requester,
                            instruction=text.strip(),
                        )
                    except Exception:
                        logger.exception("Visible employee direct dispatch failed")
                        self.reply_text(message_id, "⚠️ 该员工当前无法安全受理任务，请稍后重试。")
                    else:
                        self.reply_text(
                            message_id,
                            f"📨 已直接交给 {employee.name}；其他员工不会被唤醒。",
                        )
                    return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)

        if at_match and engine:
            agent_name = at_match.group(1)
            target_agent = engine.registry.find_by_name(agent_name, channel_id=chat_id)
            if target_agent:
                # Direct route to mentioned agent — skip NLI entirely
                self._execute_routed_message(
                    engine, message_id, chat_id, text, project, target_agent
                )
                return

        # Priority 3: NLI Intent Detection (via dedicated event loop thread)
        # Gate: use TaskClassifier confidence to decide if NLI LLM fallback is needed
        from src.slock_engine.task_classifier import TaskClassifier

        is_chitchat, confidence = TaskClassifier.classify(text or "")
        if is_chitchat and confidence >= 0.7:
            # The main ingress has already placed this message in the durable
            # group ledger. Ambient chat is context, not a model wake signal.
            return

        intent_result = None
        coro = None
        future = None
        scheduled = False
        try:
            # First try synchronous fast path (no LLM, no async)
            fast_result = self._intent_router.fast_classify(text or "")
            if fast_result is not None:
                intent_result = fast_result
            elif not is_chitchat or confidence < 0.7:
                # A managed-chat task can be admitted using deterministic local
                # classification.  Do not put the LLM NLI call in front of the
                # durable employee Team path.
                if self._is_explicit_team_task(text or "") and self._try_start_autonomous_collaboration_task(
                    message_id=message_id,
                    chat_id=chat_id,
                    text=text,
                    project=project,
                    engine=engine,
                    target_agent=target_agent,
                ):
                    return
                # Ambiguous or likely-task — invoke LLM NLI for better classification
                nli_loop = _get_nli_loop()
                timeout_s = _positive_seconds(getattr(self.ctx.settings, "slock_nli_timeout", 5), 5.0)
                coro = self._classify_with_timeout(text or "", timeout_s=timeout_s)
                future = asyncio.run_coroutine_threadsafe(coro, nli_loop)
                from concurrent.futures import Future
                scheduled = isinstance(future, Future)
                intent_result = future.result(timeout=timeout_s + 0.2)
                if coro is not None and not scheduled:
                    coro.close()
                    coro = None
        except Exception as nli_exc:
            if coro is not None and not scheduled:
                coro.close()
            if future is not None and scheduled:
                future.cancel()
            logger.debug("NLI classification skipped (timeout/error): %s", nli_exc)
            # Fall through to smart routing on any NLI failure

        # Handle activate intent even when engine is not active (FS-03, AC-03)
        if intent_result and intent_result.action == SlockCommandAction.ACTIVATE:
            if not engine:
                self.activate_slock(message_id, chat_id, text or "", project)
                return
            # Engine already active — dispatch activate as status
            self._dispatch_nli_intent(message_id, chat_id, text, project, intent_result)
            return

        if (
            not engine
            and intent_result
            and intent_result.action == SlockCommandAction.EMPLOYEE_LIST
            and intent_result.confidence >= 0.85
            and intent_result.confidence
            >= _confidence_threshold(
                getattr(
                    self.ctx.settings,
                    "slock_nli_confidence_threshold",
                    0.7,
                ),
                0.7,
            )
        ):
            # Journal-backed employees do not depend on the legacy Slock
            # engine.  A current-team query remains available before or after
            # legacy engine activation.
            self._dispatch_nli_intent(
                message_id,
                chat_id,
                text,
                project,
                intent_result,
            )
            return

        if intent_result and intent_result.action == SlockCommandAction.CHITCHAT:
            logger.debug("Skipping smart routing: high-confidence chitchat")
            return

        # If engine not activated and no activate intent → send hint (FS-03)
        if not engine:
            if intent_result and intent_result.action != SlockCommandAction.UNKNOWN:
                # User has an intent but engine isn't active
                self._send_no_engine_hint(message_id, chat_id)
            return

        # NLI dispatch: only for known actions above threshold (FS-02: UNKNOWN falls through)
        settings = self.ctx.settings
        if (
            intent_result
            and intent_result.action != SlockCommandAction.UNKNOWN
            and intent_result.confidence >= settings.slock_nli_confidence_threshold
        ):
            if intent_result.confidence >= 0.85:
                # High confidence: execute directly
                self._dispatch_nli_intent(message_id, chat_id, text, project, intent_result)
                return
            else:
                # Medium confidence: show confirmation card
                from ...slock_engine.card_templates import build_nli_feedback_card

                action_value = intent_result.action.value if hasattr(intent_result.action, "value") else str(intent_result.action)
                description = self._NLI_ACTION_DESCRIPTIONS.get(action_value, action_value)
                intent_params = {
                    "action": action_value,
                    "params": intent_result.params,
                    "original_text": text,
                }
                feedback_card = build_nli_feedback_card(
                    intent_description=description,
                    channel_id=chat_id,
                    intent_params=intent_params,
                )
                self.reply_card(message_id, json.dumps(feedback_card, ensure_ascii=False))
                return

        # Priority 3.5: If text looks like a failed command, show error suggestion
        if text and text.strip().startswith("/") and engine:
            from ...slock_engine.card_templates import build_error_suggestion_card

            suggestions = [
                "`/role` — 角色管理",
                "`/task` — 任务管理",
                "`/team` — 团队管理",
                "`/council` — 发起评审",
                "`/slock help` — 查看帮助",
            ]
            error_card = build_error_suggestion_card(text, suggestions, channel_id=chat_id)
            self.reply_card(message_id, json.dumps(error_card, ensure_ascii=False))
            return

        if self._try_start_autonomous_collaboration_task(
            message_id=message_id,
            chat_id=chat_id,
            text=text,
            project=project,
            engine=engine,
            target_agent=target_agent,
        ):
            return

        # Priority 4: Smart routing — engine.execute() (fallback for UNKNOWN/low confidence)
        self._execute_routed_message(engine, message_id, chat_id, text, project, target_agent=None)

    def _try_start_autonomous_collaboration_task(
        self,
        *,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional["ProjectContext"],
        engine,
        target_agent,
    ) -> bool:
        """Turn ordinary task messages into planned multi-role work when enabled."""
        if not self._autonomous_task_planning_enabled():
            return False
        if target_agent is not None:
            return False
        if not engine or not text or not text.strip():
            return False
        if self._looks_like_shell_text(text):
            return False

        from src.slock_engine.task_classifier import TaskClassifier

        classification, _confidence = TaskClassifier.classify_with_uncertainty(
            text,
            managed_chat=True,
        )
        if classification != "task":
            return False

        team_service = getattr(self.ctx, "employee_team_service", None)
        if team_service is not None:
            return self._start_visible_employee_team_task(
                message_id=message_id,
                chat_id=chat_id,
                text=text.strip(),
                team_service=team_service,
            )

        self.assign_task(message_id, chat_id, text.strip(), "", project)
        return True

    def _start_visible_employee_team_task(
        self,
        *,
        message_id: str,
        chat_id: str,
        text: str,
        team_service,
    ) -> bool:
        from ...thread.manager import (
            get_current_sender_id,
            get_current_tenant_key,
        )

        tenant_key = get_current_tenant_key() or ""
        requester_principal_id = get_current_sender_id() or ""
        if not tenant_key or not requester_principal_id:
            self.reply_text(message_id, "⚠️ 无法确认团队任务的租户或发起人身份。")
            return True
        try:
            run = team_service.start_task(
                tenant_key=tenant_key,
                message_id=message_id,
                chat_id=chat_id,
                requester_principal_id=requester_principal_id,
                task=text,
            )
        except TeamAdmissionError as exc:
            if exc.error_code == "no_active_team_employee":
                self.reply_text(
                    message_id,
                    "⚠️ 当前群没有可执行员工，团队任务尚未受理。"
                    "服务会自动恢复本地员工 Channel；请稍后重试，"
                    "若持续失败请检查该员工是否已通过 `/role add <名字>` 加入本群。",
                )
            else:
                self.reply_text(
                    message_id,
                    "ℹ️ 该消息对应的团队任务已经结束，请发送一条新消息重试。",
                )
            return True
        except Exception:
            logger.exception("Visible employee team admission failed")
            self.reply_text(message_id, "⚠️ 团队任务未能安全入队，请检查员工与群成员状态。")
            return True
        self.reply_text(
            message_id,
            f"👥 团队任务已受理（`{run.run_id[:20]}…`），员工将依次执行、交接、评审并汇总。",
        )
        return True

    def _autonomous_task_planning_enabled(self) -> bool:
        """Return True only for an explicit boolean True setting."""
        return getattr(self.ctx.settings, "slock_autonomous_task_planning_enabled", False) is True

    @staticmethod
    def _is_explicit_team_task(text: str) -> bool:
        """Allow the no-LLM Team path only for explicit action requests."""
        normalized = text.strip()
        return bool(
            normalized
            and not normalized.startswith("/")
            and _SLOCK_CONTROL_INTENT_RE.search(normalized) is None
            and _NON_IMPERATIVE_TEAM_TASK_RE.search(normalized) is None
            and _EXPLICIT_TEAM_TASK_RE.search(normalized)
        )

    @staticmethod
    def _looks_like_shell_text(text: str) -> bool:
        """Use the shared shell detector so task planning cannot steal commands."""
        try:
            from src.agent.intent_recognizer import IntentRecognizer

            return IntentRecognizer.looks_like_shell(text)
        except Exception:
            logger.debug("Shell-like check failed for Slock autonomous task planning", exc_info=True)
            return False

    async def _classify_with_timeout(self, text: str, *, timeout_s: float | None = None):
        """Run NLI classification with timeout protection.

        Runs inside the shared slock event loop (src/slock_engine/engine.py).
        Uses safe_wait_for for timeout; TimeoutError propagates to the caller.
        """
        from src.utils.async_helpers import safe_wait_for

        effective_timeout_s = _positive_seconds(timeout_s, 5.0)
        return await safe_wait_for(
            self._intent_router.classify_intent(text),
            timeout=effective_timeout_s,
            action="NLI 意图分类",
        )

    def _execute_routed_message(
        self,
        engine,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional["ProjectContext"],
        target_agent,
    ):
        """Execute a message routed to a specific agent or via smart routing.

        Implements fallback chain:
        - If no IDLE agent: wait up to 60s for one to become available
        - If execution fails: retry with an alternative agent once
        """
        from ...slock_engine.task_router import RoutingStatus

        agent_used = None

        def _execute():
            nonlocal agent_used
            callbacks = self._create_callbacks(message_id, chat_id, project, engine.engine_name, engine.root_path)
            if target_agent:
                agent_used = target_agent
                return engine._execute_agent(target_agent, text, callbacks)
            channel_id = engine.channel.channel_id if engine.channel else chat_id
            agents = engine.registry.list_agents(channel_id=channel_id)

            # Use fallback-aware routing
            routing_result = engine.router.route_message_with_fallback(text, agents)

            if routing_result.status == RoutingStatus.ASSIGNED:
                agent_used = routing_result.agent
                if callbacks and callbacks.on_message_routed:
                    callbacks.on_message_routed(text, agent_used)
                return engine._execute_agent(agent_used, text, callbacks)

            if routing_result.status == RoutingStatus.QUEUE_WAIT:
                # Non-blocking enqueue: hand off to dispatch loop
                from ...slock_engine.task_queue import QueuedTask
                logger.info(
                    "All %d agents busy, enqueuing message %s for dispatch loop",
                    routing_result.busy_count, message_id,
                )
                callbacks = self._create_callbacks(
                    message_id, chat_id, project, engine.engine_name, engine.root_path
                )

                # Create final result delivery callback
                def _deliver_final_result(task_id: str, result: str, card_msg_id: Optional[str]) -> None:
                    """Deliver final result to user via reply or card update."""
                    if not result:
                        return
                    try:
                        if card_msg_id:
                            # Update existing card
                            from ...slock_engine.card_templates import build_result_card
                            result_card = build_result_card(
                                task_preview=text[:100],
                                result=result,
                            )
                            self.update_card(card_msg_id, json.dumps(result_card, ensure_ascii=False))
                        else:
                            # Reply to original message
                            self.reply_text(message_id, result)
                    except Exception as e:
                        logger.warning("Failed to deliver final result for task %s: %s", task_id, str(e))

                queued = QueuedTask(
                    task_id=f"msg:{message_id}",
                    text=text,
                    chat_id=chat_id,
                    message_id=message_id,
                    callbacks=callbacks,
                    engine=engine,
                    project=project,
                    handler=self,
                    origin_message_id=message_id,
                    final_result_callback=_deliver_final_result,
                )
                try:
                    position = engine.enqueue_task(queued)
                except TaskQueueFullError:
                    logger.warning("Task queue full, rejecting message %s", message_id)
                    return None

                # Send queue position feedback card and track card_message_id
                try:
                    from ...slock_engine.card_templates import build_queue_wait_card
                    queue_card = build_queue_wait_card(
                        position=position,
                        busy_count=routing_result.busy_count,
                        message_preview=text,
                    )
                    card_msg_id = self.send_card_to_chat(chat_id, json.dumps(queue_card, ensure_ascii=False))
                    queued.card_message_id = card_msg_id
                except Exception:
                    pass  # Non-critical: don't fail execution for card issues

                # Return immediately — dispatch loop will consume when agent is idle
                return "queued"

            # Compatibility: older/mocked routers may not implement the fallback
            # result object. Fall back to the direct route_message contract.
            if routing_result.status not in (
                RoutingStatus.ASSIGNED,
                RoutingStatus.QUEUE_WAIT,
                RoutingStatus.NO_MATCH,
            ):
                direct_agent = engine.router.route_message(text, agents)
                if direct_agent:
                    agent_used = direct_agent
                    if callbacks and callbacks.on_message_routed:
                        callbacks.on_message_routed(text, agent_used)
                    return engine._execute_agent(agent_used, text, callbacks)

            # NO_MATCH — no agents at all
            return None

        def _execute_with_retry():
            """Wrap _execute with one retry using an alternative agent on failure.

            Treats both exceptions AND None return (agent failed silently) as
            retryable failures — tries an alternative agent once before giving up.
            """
            nonlocal agent_used
            try:
                result = _execute()
            except Exception as first_err:
                logger.warning(
                    "Primary agent execution failed for message %s: %s, attempting retry",
                    message_id, first_err,
                )
                result = None
                _first_err = first_err
            else:
                if result is not None:
                    return result
                _first_err = None
                logger.warning(
                    "Primary agent returned None for message %s (agent=%s), attempting retry",
                    message_id, agent_used.name if agent_used else "?",
                )

            # Retry: try a different agent
            _failed_agent_name = agent_used.name if agent_used else "unknown"
            try:
                channel_id = engine.channel.channel_id if engine.channel else chat_id
                agents = engine.registry.list_agents(channel_id=channel_id)
                fallback_agents = [a for a in agents if a != agent_used] if agent_used else agents

                # Filter out agents whose agent_type is known to be ACP-incompatible
                # (avoids wasting ~100s retrying claude when it can't serve ACP)
                if fallback_agents:
                    try:
                        from ...acp.sync_adapter import resolve_agent_spec
                        _acp_compatible = []
                        for a in fallback_agents:
                            try:
                                resolve_agent_spec(a.agent_type, model_name=a.model_name or None)
                                _acp_compatible.append(a)
                            except (RuntimeError, Exception):
                                logger.debug(
                                    "Retry skip agent %s (%s): ACP incompatible",
                                    a.name, a.agent_type,
                                )
                        if _acp_compatible:
                            fallback_agents = _acp_compatible
                    except ImportError:
                        pass

                if fallback_agents:
                    alt_agent = engine.router.route_message(text, fallback_agents)
                    if alt_agent:
                        # Send retry swap feedback card
                        try:
                            from ...slock_engine.card_templates import build_retry_swap_card
                            swap_card = build_retry_swap_card(
                                failed_agent_name=_failed_agent_name,
                                new_agent_name=alt_agent.name,
                                error_hint=str(_first_err)[:120] if _first_err else "",
                            )
                            self.send_card_to_chat(chat_id, json.dumps(swap_card, ensure_ascii=False))
                        except Exception:
                            pass
                        agent_used = alt_agent
                        callbacks = self._create_callbacks(
                            message_id, chat_id, project, engine.engine_name, engine.root_path
                        )
                        return engine._execute_agent(alt_agent, text, callbacks)
            except Exception as retry_err:
                logger.error(
                    "Retry also failed for message %s: %s",
                    message_id, retry_err,
                )
            # Re-raise original error if retry didn't help
            if _first_err:
                raise _first_err
            return None

        def _result_card(result: str, duration: float) -> str:
            if agent_used:
                try:
                    engine.memory.write_agent_reasoning_snapshot(
                        agent_used.agent_id,
                        f"message:{message_id}",
                        prompt_summary=text[:1000],
                        result_summary=result[:2000],
                        tool_name=agent_used.agent_type,
                        model_name=agent_used.model_name,
                    )
                except Exception:
                    logger.warning(
                        "Failed to persist Slock reasoning snapshot for message %s agent %s",
                        message_id,
                        agent_used.agent_id,
                        exc_info=True,
                    )
                card_data = engine._mouthpiece.format_card(
                    agent_used,
                    result,
                    model_info=agent_used.agent_type,
                    duration_s=duration,
                    channel_id=chat_id,
                    task_id=f"message:{message_id}",
                )
                # Append confirmation buttons for result feedback
                try:
                    from ...slock_engine.card_templates.queue_feedback import build_result_confirmation_buttons
                    confirm_elements = build_result_confirmation_buttons(
                        channel_id=chat_id,
                        message_id=message_id,
                        task_id=f"message:{message_id}",
                    )
                    if isinstance(card_data, dict):
                        body = card_data.get("body", card_data.get("card", {}).get("body", {}))
                        elements = body.get("elements", [])
                        elements.extend(confirm_elements)
                except Exception:
                    pass  # Non-critical: don't break result delivery for button rendering
                return json.dumps(card_data, ensure_ascii=False)
            from ...slock_engine.card_templates.common import build_card_wrapper  # noqa: F811
            return json.dumps(build_card_wrapper(
                header_title="💬 Agent 回复",
                header_template="blue",
                elements=[{"tag": "markdown", "content": result}],
                mobile_optimize=True,
            ), ensure_ascii=False)

        def _error_card(exc: Exception) -> str:
            from ...slock_engine.card_templates.common import build_card_wrapper
            return json.dumps(build_card_wrapper(
                header_title="❌ 执行出错",
                header_template="red",
                elements=[{"tag": "markdown", "content": f"Agent 执行出错: {safe_error_message(exc)}"}],
                mobile_optimize=True,
            ), ensure_ascii=False)

        def _empty_card() -> str:
            from ...slock_engine.card_templates.common import build_card_wrapper
            return json.dumps(build_card_wrapper(
                header_title="⚠️ 执行无结果",
                header_template="orange",
                elements=[{"tag": "markdown", "content": (
                    "Agent 执行完成但未产出结果。可能原因：\n"
                    "- 所有角色正在忙碌\n"
                    "- 执行超时被自动中止\n"
                    "- 任务内容不明确\n\n"
                    "请稍后重试或补充更多描述。"
                )}],
                mobile_optimize=True,
            ), ensure_ascii=False)

        def _busy_card() -> str:
            from ...slock_engine.card_templates.common import build_card_wrapper
            return json.dumps(build_card_wrapper(
                header_title="⚠️ 团队繁忙",
                header_template="orange",
                elements=[{"tag": "markdown", "content": "当前所有角色均在忙碌中，请稍后重试。"}],
                mobile_optimize=True,
            ), ensure_ascii=False)

        from ...slock_engine.card_templates.queue_feedback import build_task_ack_card
        ack_card = build_task_ack_card(
            message_preview=text or "",
            status="received",
        )
        placeholder_card = json.dumps(ack_card, ensure_ascii=False)

        self._execute_async(
            engine=engine,
            execute_fn=_execute_with_retry,
            placeholder_card_json=placeholder_card,
            result_card_fn=_result_card,
            error_card_fn=_error_card,
            empty_card_fn=_empty_card,
            busy_card_fn=_busy_card,
            message_id=message_id,
            chat_id=chat_id,
        )

    # ------------------------------------------------------------------
    # NLI and routing helpers
    # ------------------------------------------------------------------

    _NLI_ACTION_DESCRIPTIONS: dict[str, str] = {
        "status": "查看团队状态",
        "stop": "停止 Slock 引擎",
        "help": "查看帮助信息",
        "new_role": "创建新角色",
        "role_list": "查看角色列表",
        "employee_list": "查看当前群团队成员",
        "task_list": "查看任务列表",
        "task_assign": "分配任务",
        "task_status": "查看任务状态",
        "activate": "启动 Slock",
        "discussion": "发起讨论",
    }

    def _send_no_engine_hint(self, message_id: str, chat_id: str) -> None:
        """Send a friendly hint card when slock engine is not activated."""
        hint_card = json.dumps({
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "💡 Slock 未激活"},
                "template": "wathet",
            },
            "body": {"elements": [
                {"tag": "markdown", "content": (
                    "当前群聊尚未启用 Slock 协作模式。\n\n"
                    "**直接在群里发任务即可，Agent 自动处理。** 无需任何前置命令。\n\n"
                    "---\n"
                    "📎 *也可用命令*: `/slock`、`/new-team 团队名`、`/slock help`"
                )},
            ]},
        }, ensure_ascii=False)
        self.reply_card(message_id, hint_card)

    def _dispatch_nli_intent(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional["ProjectContext"],
        intent_result,
    ) -> None:
        """Dispatch a high-confidence NLI intent by calling handler methods directly.

        Avoids command string assembly — passes structured params to internal
        handler methods, eliminating quoting/escaping issues.
        """
        from src.slock_engine.slash_commands import SlockCommandAction

        action = intent_result.action

        # Param whitelist filtering: only allow known keys per action to prevent injection
        _ALLOWED_PARAM_KEYS: dict[SlockCommandAction, set[str]] = {
            SlockCommandAction.NEW_ROLE: {"name", "tool", "role"},
            SlockCommandAction.DISCUSSION: {"participants", "topic"},
            SlockCommandAction.COUNCIL: {"topic"},
            SlockCommandAction.NEW_TEAM: {"name"},
            SlockCommandAction.ROLE_REMOVE: {"target"},
            SlockCommandAction.ROLE_INFO: {"target"},
            SlockCommandAction.ACTIVATE: {"requirement"},
        }
        allowed_keys = _ALLOWED_PARAM_KEYS.get(action, set())
        params = {k: v for k, v in intent_result.params.items() if k in allowed_keys} if allowed_keys else {}

        # --- Simple no-param actions: direct method calls ---
        if action == SlockCommandAction.STOP:
            self.stop_slock_engine(message_id, chat_id, project)
        elif action == SlockCommandAction.STATUS:
            self.show_slock_status(message_id, chat_id, project)
        elif action == SlockCommandAction.HELP:
            self.show_slock_help(message_id)
        elif action == SlockCommandAction.TASK_LIST:
            self.list_tasks(message_id, chat_id, project)
        elif action == SlockCommandAction.TASK_STATUS:
            self.show_task_status(message_id, chat_id, project)
        elif action == SlockCommandAction.ROLE_LIST:
            self.list_roles(message_id, chat_id, project)
        elif action == SlockCommandAction.EMPLOYEE_LIST:
            self.list_current_team_members(message_id, chat_id)
        elif action == SlockCommandAction.TEAM_LIST:
            self.list_teams(message_id, chat_id, project)
        elif action == SlockCommandAction.ACTIVATE:
            self.activate_slock(message_id, chat_id, params.get("requirement", ""), project)

        # --- Parameterized actions: structured params ---
        elif action == SlockCommandAction.NEW_ROLE:
            name = params.get("name", "Agent")
            # Build args string for create_role (it handles shlex parsing internally)
            args_parts = [name]
            if params.get("tool"):
                args_parts.append(f"--tool {params['tool']}")
            if params.get("role"):
                args_parts.append(f"--role {params['role']}")
            self.create_role(message_id, chat_id, " ".join(args_parts), project)
        elif action == SlockCommandAction.NEW_TEAM:
            team_name = params.get("name", "Team")
            self.create_team(message_id, chat_id, team_name, project)
        elif action == SlockCommandAction.ROLE_REMOVE:
            target = params.get("target", "")
            self.remove_role(message_id, chat_id, target, project)
        elif action == SlockCommandAction.ROLE_INFO:
            target = params.get("target", "")
            self.show_role_info(message_id, chat_id, target, project)

        # --- Discussion action: trigger inter-agent discussion ---
        elif action == SlockCommandAction.DISCUSSION:
            self._trigger_nli_discussion(message_id, chat_id, text, params, project)
        elif action == SlockCommandAction.COUNCIL:
            self.run_council(message_id, chat_id, params.get("topic") or text, project)

        else:
            # Unhandled intent — fallback to reply
            self.reply_text(message_id, f"🤔 理解为：{action.value}，但暂不支持自然语言执行此操作。请使用对应的 / 命令。")

    def _trigger_nli_discussion(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        params: dict,
        project: Optional["ProjectContext"],
    ) -> None:
        """Handle DISCUSSION intent: resolve participant names and trigger engine discussion.

        Extracts participant agent names from NLI params (or action_hint fallback),
        resolves them to agent IDs via the engine registry, and triggers the
        discussion flow through the engine's executor (bounded concurrency + watchdog).

        Permission: admin/owner can always trigger; other members are rate-limited
        to 3 discussion triggers per 5 minutes.
        """
        import json as _json

        from src.slock_engine.models import AgentStatus, DiscussionStatus, DiscussionThread

        # Permission gate: check discussion trigger rate-limit for non-admin users
        if not self._check_discussion_permission(message_id, chat_id):
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self._send_no_engine_hint(message_id, chat_id)
            return

        # Extract participant names from params
        participants_raw: list[str] = []
        if "participants" in params:
            participants_raw = params["participants"]
        elif "participant_a" in params and "participant_b" in params:
            participants_raw = [params["participant_a"], params["participant_b"]]

        if len(participants_raw) < 2:
            self.reply_text(
                message_id,
                "💬 发起讨论需要至少两个参与者。\n\n示例：「让 coder 和 reviewer 讨论一下代码方案」",
            )
            return

        # Resolve names to agent IDs
        participant_ids: list[str] = []
        unresolved: list[str] = []
        for name in participants_raw:
            agent = engine.find_agent_by_name(name) if hasattr(engine, "find_agent_by_name") else None
            if agent is None:
                # Fallback: try get_agent directly (if name is already an ID)
                agent = engine.get_agent(name)
            if agent:
                participant_ids.append(agent.agent_id)
            else:
                unresolved.append(name)

        if unresolved:
            self.reply_text(
                message_id,
                f"❌ 未找到以下角色：{'、'.join(unresolved)}\n\n请确认角色名称后重试，或使用 `/role list` 查看可用角色。",
            )
            return

        if len(participant_ids) < 2:
            self.reply_text(message_id, "💬 需要至少两个有效参与者才能发起讨论。")
            return

        # Build discussion thread with config from settings
        topic = params.get("topic", text)
        config = engine.build_discussion_config_from_settings()
        thread = DiscussionThread(
            channel_id=chat_id,
            participants=participant_ids,
            trigger_reason=f"NLI discussion: {topic[:100]}",
            config=config,
        )

        # Check capacity and add
        if not engine._add_discussion(chat_id, thread):
            self.reply_text(
                message_id,
                "⏳ 当前频道讨论数已达上限，请等待现有讨论结束后重试。",
            )
            return

        # Send starting card
        from src.slock_engine.card_templates import (
            build_discussion_card_from_thread,
            build_discussion_summary_card_from_thread,
        )

        try:
            card = build_discussion_card_from_thread(thread)
            card_json = _json.dumps(card, ensure_ascii=False)
            discussion_card_msg_id = self.send_card_to_chat(
                chat_id, card_json, origin_message_id=message_id
            )
        except Exception:
            discussion_card_msg_id = None

        # Run discussion via engine executor (bounded concurrency) with watchdog
        settings = self.ctx.settings
        watchdog_timeout = settings.slock_discussion_timeout

        def _run():
            from src.slock_engine.discussion_manager import DiscussionManager

            # Watchdog timer: force-terminate if discussion exceeds timeout
            watchdog = threading.Timer(watchdog_timeout, lambda: _watchdog_trigger())
            watchdog_fired = threading.Event()

            def _watchdog_trigger():
                watchdog_fired.set()
                thread.status = DiscussionStatus.TIMEOUT
                logger.warning("Discussion watchdog fired: %s (timeout=%ds)", thread.thread_id, watchdog_timeout)

            watchdog.daemon = True
            watchdog.start()

            try:
                dm = DiscussionManager(engine=engine, memory_manager=engine._memory, config=config)

                def on_round_complete(updated_thread):
                    if watchdog_fired.is_set():
                        return
                    if discussion_card_msg_id:
                        try:
                            updated_card = build_discussion_card_from_thread(updated_thread)
                            self.update_card(discussion_card_msg_id, _json.dumps(updated_card, ensure_ascii=False))
                        except Exception:
                            pass

                completed = dm.run_discussion(thread, topic, on_round_complete=on_round_complete)

                # Send summary card
                try:
                    summary_card = build_discussion_summary_card_from_thread(completed)
                    self.send_card_to_chat(chat_id, _json.dumps(summary_card, ensure_ascii=False))
                except Exception:
                    pass

            except Exception as exc:
                from src.utils.errors import redact_sensitive, safe_error_message
                logger.error("NLI discussion failed: %s", redact_sensitive(str(exc)), exc_info=True)
                self.reply_text(message_id, f"❌ 讨论执行失败：{safe_error_message(exc)}")
            finally:
                watchdog.cancel()
                engine._remove_discussion(chat_id, thread.thread_id)
                # Reset participant agent states to IDLE
                for pid in participant_ids:
                    try:
                        engine.set_agent_status(pid, AgentStatus.IDLE)
                    except Exception:
                        pass

        # Submit to engine executor for bounded concurrency
        try:
            executor = engine._get_executor()
            executor.submit(_run)
        except Exception as exc:
            from src.utils.errors import redact_sensitive
            logger.warning("Failed to submit discussion to executor: %s", redact_sensitive(str(exc)))
            engine._remove_discussion(chat_id, thread.thread_id)
            self.reply_text(message_id, "⏳ 执行队列已满，请稍后再试。")

    def run_council(
        self,
        message_id: str,
        chat_id: str,
        question: str = "",
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Run a Slock Council: independent opinions, anonymous review, final synthesis."""
        if not question.strip():
            self.reply_text(message_id, "请提供 Council 议题\n\n用法: `/council <要评审的问题或方案>`")
            return

        if not self._check_discussion_permission(message_id, chat_id):
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self._send_no_engine_hint(message_id, chat_id)
            return

        from src.slock_engine.card_templates import build_council_card
        from src.slock_engine.models import AgentStatus, CouncilRun, CouncilStatus

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agents = list(engine.registry.list_agents(channel_id=channel_id))
        idle_agents = [
            agent for agent in agents
            if engine.get_agent_status(agent.agent_id) == AgentStatus.IDLE
        ]
        if len(idle_agents) < 2:
            self.reply_text(message_id, "Council 至少需要两个空闲角色。可用 `/role list` 查看角色状态。")
            return

        rank_agents = getattr(engine.router, "rank_agents_for_claim", None)
        ranked_agents = rank_agents(question, idle_agents) if callable(rank_agents) else idle_agents
        if not isinstance(ranked_agents, list) or len(ranked_agents) < 2:
            ranked_agents = idle_agents

        max_agents = max(2, int(getattr(self.ctx.settings, "slock_max_parallel_agents", 4)))
        participants = ranked_agents[:max_agents]
        chairman = self._select_council_chairman(idle_agents, participants)

        initial_run = CouncilRun(
            channel_id=channel_id,
            question=question,
            participant_ids=[agent.agent_id for agent in participants],
            chairman_agent_id=chairman.agent_id if chairman else "",
            status=CouncilStatus.STARTING,
        )
        card_message_id = self.send_card_to_chat(
            chat_id,
            json.dumps(build_council_card(initial_run, channel_id=channel_id), ensure_ascii=False),
            origin_message_id=message_id,
        )

        def _update_stage(run) -> None:
            if not card_message_id:
                return
            try:
                self.update_card(
                    card_message_id,
                    json.dumps(build_council_card(run, channel_id=channel_id), ensure_ascii=False),
                )
            except Exception as exc:
                logger.debug("Council card update failed: %s", str(exc))

        def _run() -> None:
            try:
                run = engine.run_council(
                    question,
                    participants=participants,
                    chairman=chairman,
                    on_stage=_update_stage,
                    timeout=float(getattr(self.ctx.settings, "slock_discussion_timeout", 300)),
                )
                final_card = json.dumps(build_council_card(run, channel_id=channel_id), ensure_ascii=False)
                if card_message_id:
                    self.update_card(card_message_id, final_card)
                else:
                    self.send_card_to_chat(chat_id, final_card, origin_message_id=message_id)
            except Exception as exc:
                logger.error("Slock Council failed: %s", exc, exc_info=True)
                failed = CouncilRun(
                    channel_id=channel_id,
                    question=question,
                    participant_ids=[agent.agent_id for agent in participants],
                    chairman_agent_id=chairman.agent_id if chairman else "",
                    status=CouncilStatus.FAILED,
                    error=safe_error_message(exc),
                )
                failed_card = json.dumps(build_council_card(failed, channel_id=channel_id), ensure_ascii=False)
                if card_message_id:
                    self.update_card(card_message_id, failed_card)
                else:
                    self.send_card_to_chat(chat_id, failed_card, origin_message_id=message_id)

        try:
            engine._get_executor().submit(_run)
        except (ExecutorQueueFullError, RuntimeError) as exc:
            logger.warning("Failed to submit Slock Council: %s", repr(exc))
            busy = CouncilRun(
                channel_id=channel_id,
                question=question,
                participant_ids=[agent.agent_id for agent in participants],
                chairman_agent_id=chairman.agent_id if chairman else "",
                status=CouncilStatus.FAILED,
                error="执行队列已满，请稍后重试。",
            )
            if card_message_id:
                self.update_card(
                    card_message_id,
                    json.dumps(build_council_card(busy, channel_id=channel_id), ensure_ascii=False),
                )
            else:
                self.reply_text(message_id, "⏳ 执行队列已满，请稍后重试。")

    @staticmethod
    def _select_council_chairman(agents: list, participants: list):
        """Prefer a chair/architect/planner/reviewer, otherwise reuse the first participant."""
        preferred_roles = ("chair", "architect", "planner", "reviewer")
        for role in preferred_roles:
            for agent in agents:
                if getattr(agent, "role", "") == role:
                    return agent
        return participants[0] if participants else None

    # ------------------------------------------------------------------
    # Slock activation
    # ------------------------------------------------------------------

    def activate_slock(
        self,
        message_id: str,
        chat_id: str,
        requirement: str = "",
        project: Optional["ProjectContext"] = None,
        *,
        skip_guard_check: bool = False,
    ) -> bool:
        """Activate slock mode for the current chat.

        Args:
            skip_guard_check: If True, skip the ActivationGuard permission/rate-limit
                check. Used when the guard has already been checked by the caller
                (e.g., ws_client._auto_activate_slock) to avoid double consumption
                of rate-limit budget.

        Returns:
            True if activation succeeded (or was already active), False otherwise.
        """
        project = self._ensure_project(message_id, chat_id, project)
        if not project:
            return False

        # ActivationGuard permission check — deny early if sender lacks permission
        if not skip_guard_check:
            from ...slock_engine.activation_guard import get_activation_guard
            from ...thread.manager import get_current_sender_id

            guard = get_activation_guard()
            sender_id = get_current_sender_id() or ""
            allowed, reason = guard.can_auto_activate(sender_id, chat_id, self.ctx.settings)
            if not allowed:
                from ...slock_engine.card_templates.queue_feedback import build_activation_denied_card

                card = build_activation_denied_card(
                    reason=reason,
                    hint="当前用户无权激活 Slock 团队，请联系管理员配置白名单或使用 allow_all 策略。",
                )
                self.send_card_to_chat(chat_id, json.dumps(card, ensure_ascii=False))
                return False

        root_path = project.root_path if project else self.get_working_dir(chat_id)
        manager = self._get_engine_manager()

        # Check if already active — show interactive command panel
        existing = manager.get_activated_engine(chat_id)
        if existing:
            from ...slock_engine.card_templates import build_command_panel_card

            panel_card = build_command_panel_card(channel_id=chat_id)
            self.reply_card(message_id, json.dumps(panel_card, ensure_ascii=False))
            return True

        from ...thread.manager import get_current_sender_id

        sender_open_id = get_current_sender_id() or ""

        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
        # Create and activate channel
        channel = SlockChannel(
            channel_id=chat_id,
            name=project.project_name if project else "slock",
            team_name=project.project_name if project else "Team",
            owner_id=sender_open_id,
        )
        engine = manager.get_or_create_activated(
            chat_id,
            root_path,
            channel,
            engine_name=engine_name,
        )

        # Bootstrap default roles asynchronously (non-blocking)
        self._bootstrap_default_roles_if_configured(engine, channel.channel_id, chat_id)

        # Wire UI callbacks for escalation timeout notifications
        engine.set_escalation_ui_callbacks(
            update_card_fn=self.update_card,
            send_text_fn=self.send_text_to_chat,
        )

        # Wire card delivery callbacks for progress tracking
        def _send_card(card: dict) -> "Optional[str]":
            card_json = json.dumps(card, ensure_ascii=False)
            return self.send_card_to_chat(chat_id, card_json)

        def _update_card(msg_id: str, card: dict) -> bool:
            card_json = json.dumps(card, ensure_ascii=False)
            return self.update_card(msg_id, card_json)

        engine.set_card_callbacks(send_fn=_send_card, update_fn=_update_card)

        self.add_reaction(message_id, EmojiReaction.on_multi_task_start())

        # Use unified welcome card with team/channel info prepended
        from ...slock_engine.card_templates import build_welcome_card

        welcome_card = build_welcome_card(team_name=channel.team_name)
        # Prepend team/channel metadata to the welcome card body
        team_info_element = {
            "tag": "markdown",
            "content": (
                f"**团队**: {channel.team_name}\n"
                f"**频道**: {channel.name}"
            ),
        }
        welcome_card["body"]["elements"].insert(0, team_info_element)
        welcome_card["header"]["title"]["content"] = "🎭 Slock 协作模式已激活"

        # Task 13: detect synthetic message_id → use send_card_to_chat instead of reply
        if message_id.startswith("passive-activate-"):
            card_json = json.dumps(welcome_card, ensure_ascii=False)
            self.send_card_to_chat(chat_id, card_json)
        else:
            session = self.create_static_card_session(chat_id, reply_to=message_id)
            session.send(welcome_card)
            session.close()

        # Task 12: Enqueue first message atomically if requirement is provided
        if requirement:
            import uuid as _uuid

            from ...slock_engine.task_queue import QueuedTask

            # Create final result delivery callback for first message
            def _deliver_first_result(task_id: str, result: str, card_msg_id: Optional[str]) -> None:
                """Deliver final result for the first message."""
                if not result:
                    return
                try:
                    if card_msg_id:
                        from ...slock_engine.card_templates import build_result_card
                        result_card = build_result_card(
                            task_preview=requirement[:100],
                            result=result,
                        )
                        self.update_card(card_msg_id, json.dumps(result_card, ensure_ascii=False))
                    else:
                        if message_id.startswith("passive-activate-"):
                            self.send_text_to_chat(chat_id, result)
                        else:
                            self.reply_text(message_id, result)
                except Exception as e:
                    logger.warning("Failed to deliver first message result: %s", str(e))

            task = QueuedTask(
                task_id=f"first-msg-{_uuid.uuid4().hex[:8]}",
                text=requirement,
                chat_id=chat_id,
                message_id=message_id,
                bootstrap_pending=True,
                origin_message_id=message_id,
                final_result_callback=_deliver_first_result,
            )
            try:
                engine._task_queue.enqueue(task)
                logger.info("First message enqueued for chat=%s: %s", chat_id, requirement[:60])
            except Exception as enqueue_err:
                from ...slock_engine.card_templates.queue_feedback import build_queue_full_card
                from ...slock_engine.exceptions import TaskQueueFullError

                is_queue_full = isinstance(enqueue_err, TaskQueueFullError)
                logger.warning(
                    "Failed to enqueue first message for chat=%s (queue_full=%s): %s",
                    chat_id, is_queue_full, enqueue_err,
                )
                if is_queue_full:
                    try:
                        card = build_queue_full_card(
                            message_preview=requirement,
                            max_size=getattr(engine._task_queue, "_max_size", 8),
                        )
                        if message_id.startswith("passive-activate-"):
                            self.send_card_to_chat(chat_id, json.dumps(card, ensure_ascii=False))
                        else:
                            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
                    except Exception as card_err:
                        logger.warning("Failed to send queue-full card: %s", card_err)

        return True

    def _bootstrap_default_roles_if_configured(
        self,
        engine: "object",
        channel_id: str,
        chat_id: str,
    ) -> None:
        """Bootstrap default roles with exception safety and async execution.

        - Never blocks the caller (uses asyncio.create_task if a loop is running,
          otherwise falls back to threading).
        - Never raises — bootstrap failure is logged and a degradation notice is
          sent to the chat so the user knows some roles may be missing.
        """
        _default_roles_cfg = getattr(self.ctx.settings, "slock_default_roles", "")
        if not _default_roles_cfg:
            # No roles to bootstrap — leave bootstrap-ready state as-is (default: ready)
            return

        # Signal dispatch loop to wait for bootstrap via public API
        engine.prepare_bootstrap()

        def _do_bootstrap() -> None:
            try:
                from ...slock_engine.role_bootstrap import bootstrap_default_roles
                created = bootstrap_default_roles(engine, channel_id, _default_roles_cfg)
                if created:
                    logger.info(
                        "_bootstrap_default_roles_if_configured: created %d role(s) in %s",
                        len(created), channel_id,
                    )
            except Exception as e:
                logger.error(
                    "_bootstrap_default_roles_if_configured: bootstrap failed for channel %s: %s",
                    channel_id, e, exc_info=True,
                )
                # Send degradation notice to the chat
                try:
                    self.send_text_to_chat(
                        chat_id,
                        "⚠️ 部分预置角色创建失败，可使用 /role list 检查",
                    )
                except Exception:
                    logger.debug("Failed to send bootstrap degradation notice")
            finally:
                # Always signal bootstrap complete via public API so dispatch loop can proceed
                engine.finish_bootstrap()

        # Execute asynchronously to avoid blocking the response path
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _do_bootstrap)
        except RuntimeError:
            # No running event loop — fall back to thread
            threading.Thread(
                target=_do_bootstrap,
                name="slock-bootstrap",
                daemon=True,
            ).start()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def show_slock_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Show persistent employee runtime, falling back to legacy Slock status."""
        from ...thread.manager import get_current_tenant_key

        tenant_key = get_current_tenant_key() or ""
        ctx = getattr(self, "ctx", None)
        runtime_facade = getattr(ctx, "employee_runtime_facade", None)
        list_runtime = getattr(runtime_facade, "list_employee_runtime_statuses", None)
        if tenant_key and callable(list_runtime):
            try:
                runtime_views = tuple(list_runtime(tenant_key))
            except Exception:
                logger.exception("employee runtime overview facade failed")
            else:
                if runtime_views:
                    from ...autonomous.manager.cards import (
                        build_employee_runtime_overview_card,
                    )

                    self.reply_card(
                        message_id,
                        json.dumps(
                            build_employee_runtime_overview_card(runtime_views),
                            ensure_ascii=False,
                        ),
                    )
                    return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))

        if not engine:
            _msg_type, card_content = CardBuilder.build_info_card(
                project=project,
                title="🎭 Slock 状态",
                content="当前没有活跃的 Slock 协作团队\n\n发送 `/slock` 激活协作模式",
                engine_name=engine_name,
                show_buttons=False,
            )
            self.reply_card(message_id, card_content)
            return

        # Build status panel card with clear sections:
        # 1) Team overview (name + member count)
        # 2) Role status table (emoji | name | status | current task)
        # 3) Action buttons (refresh, stop all, per-agent stop)
        from ...slock_engine.card_templates import build_status_panel_card
        from ...slock_engine.models import AgentStatus as AgentStatusEnum
        from ...slock_engine.models import TaskStatus

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        team_name = engine.channel.team_name if engine.channel else ""

        agents: list[tuple] = []
        current_tasks: dict = {}
        skill_profiles: dict = {}
        for agent in engine.registry.list_agents(channel_id=channel_id):
            status = engine.get_agent_status(agent.agent_id) or AgentStatusEnum.IDLE
            agents.append((agent, status))
            # Find active task claimed by this agent
            for task in engine.tasks:
                if task.claimed_by == agent.agent_id and task.status == TaskStatus.IN_PROGRESS:
                    current_tasks[agent.agent_id] = task
                    break
            # Collect skill profiles
            profiles = engine.memory.read_skill_profiles(agent.agent_id)
            if profiles:
                skill_profiles[agent.agent_id] = [
                    {"tag": p.tag, "success_rate": p.success_rate, "total_tasks": p.total_tasks}
                    for p in profiles
                ]

        status_card = build_status_panel_card(
            agents=agents,
            team_name=team_name,
            channel_id=channel_id,
            current_tasks=current_tasks,
            skill_profiles=skill_profiles,
            tasks_summary={
                "total": len(engine.tasks),
                "todo": sum(1 for t in engine.tasks if t.status == TaskStatus.TODO),
                "in_progress": sum(1 for t in engine.tasks if t.status == TaskStatus.IN_PROGRESS),
                "in_review": sum(1 for t in engine.tasks if t.status == TaskStatus.IN_REVIEW),
                "done": sum(1 for t in engine.tasks if t.status == TaskStatus.DONE),
            },
        )
        self.reply_card(message_id, json.dumps(status_card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def show_slock_help(self, message_id: str):
        """Show slock help information using the command panel card."""
        from ...slock_engine.card_templates import build_command_panel_card

        panel_card = build_command_panel_card()
        self.reply_card(message_id, json.dumps(panel_card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Team management
    # ------------------------------------------------------------------

    def create_team(
        self, message_id: str, chat_id: str, name: str = "", project: Optional["ProjectContext"] = None
    ):
        """Create a new Feishu group and activate slock runtime inside it.

        Flow: validate → create group → promote sender → init engine →
              activate channel (workspace) → register managed chat →
              send welcome in new group → send jump card in original group.
        On failure: roll back local state and retain the remote group untrusted.
        """
        if not name:
            self.reply_text(message_id, "请提供团队名称\n\n用法: `/new-team <团队名称>`")
            return

        from ...project_chat.lark_chat_client import LarkChatClient
        from ...thread.manager import get_current_sender_id

        sender_open_id = get_current_sender_id() or ""
        if not sender_open_id:
            self.reply_text(message_id, "❌ 无法获取发送者信息，请重试")
            return

        settings = self.ctx.settings
        group_name = self._format_slock_group_name(name, getattr(settings, "slock_team_name_suffix", "[Slock]"))
        manager = self._get_engine_manager()
        retained_chat_id = manager.reserve_retained_team_name(name)
        if retained_chat_id is None and not manager.reserve_team_name(name):
            self.reply_text(message_id, f"❌ 团队名称 **{name}** 已存在或正在创建，请换一个名称。")
            return

        root_path = project.root_path if project else self.get_working_dir(chat_id)
        project_id = project.project_id if project else f"team:{name}"
        managed_groups = self._managed_group_registry()
        provision_id = (
            f"new-team:{self._managed_group_owner_id()}:"
            f"{name.casefold()}:{root_path}"
        )
        persisted = True
        if managed_groups is not None:
            from ...trust.models import ManagedGroupOrigin

            owner_id = self._managed_group_owner_id()
            receiving_bot_ref = self._managed_group_receiving_bot_ref()
            if not owner_id or not receiving_bot_ref:
                self.reply_text(
                    message_id,
                    "❌ 受管群 Owner 或接收 Bot 身份未配置，无法创建团队。",
                )
                manager.release_team_name(name)
                return
            try:
                managed_groups.begin_provision(
                    provision_id=provision_id,
                    owner_id=owner_id,
                    origin=ManagedGroupOrigin.GHOSTAP_CREATED,
                    receiving_bot_ref=receiving_bot_ref,
                    project_id=project_id,
                    canonical_root_ref=root_path,
                    created_at=datetime.now(UTC),
                )
                recovered_chat_id = managed_groups.provision_chat_id(provision_id)
            except Exception:
                logger.exception("create_team: managed group provision intent failed")
                self.reply_text(message_id, "❌ 受管团队登记准备失败，请重试")
                manager.release_team_name(name)
                return
        else:
            recovered_chat_id = None

        # Step 1: Create Feishu group
        lark_client = LarkChatClient(api_client_factory=self.ctx.api_client_factory)
        if retained_chat_id is not None:
            if recovered_chat_id not in (None, retained_chat_id):
                self.reply_text(
                    message_id,
                    "⚠️ 保留群与 Registry operation 记录不一致，已保持阻断。",
                )
                manager.release_team_name(name)
                return
            recovered_chat_id = retained_chat_id
        if recovered_chat_id is not None:
            from ...project_chat.lark_chat_client import ManagedChatValidation

            validation = lark_client.validate_managed_chat(
                recovered_chat_id,
                self._managed_group_owner_id(),
            )
            if (
                retained_chat_id is not None
                and validation is ManagedChatValidation.INVALID
            ):
                if not self._abandon_managed_group_provision(
                    managed_groups,
                    provision_id,
                ):
                    self.reply_text(
                        message_id,
                        "⚠️ 保留团队群校验失败，但 Registry 创建意图未能持久清理；"
                        "已保留恢复记录并失败关闭。",
                    )
                    return
                consumed = manager.consume_retained_team_name(
                    name,
                    retained_chat_id,
                )
                if consumed:
                    manager.release_team_name(name)
                    self.reply_text(
                        message_id,
                        "⚠️ 上次保留团队群已无法通过 Owner/接收 Bot 校验，"
                        "恢复记录已清理；请重试原命令明确新建。",
                    )
                else:
                    self.reply_text(
                        message_id,
                        "⚠️ 保留团队群校验失败，且恢复记录未能持久清理；"
                        "当前进程已失败关闭。",
                    )
                return
            if validation is not ManagedChatValidation.VALID:
                if retained_chat_id is not None:
                    self.reply_text(
                        message_id,
                        "⚠️ 无法确认上次保留团队群的 Owner/接收 Bot 身份，"
                        "已继续阻止新建；请稍后重试。",
                    )
                    manager.release_team_name(name)
                    return
                persisted = manager.block_team_name_for_cleanup(
                    name,
                    recovered_chat_id,
                    "recovered_chat_invalid",
                )
                self.reply_text(
                    message_id,
                    "⚠️ 无法确认重试群仍存在且 Owner/接收 Bot 均在群内，已停止激活。"
                    + (
                        ""
                        if persisted
                        else " 恢复记录持久化失败，当前进程已失败关闭。"
                    ),
                )
                if persisted:
                    manager.release_team_name(name)
                return
            if retained_chat_id is not None and managed_groups is not None:
                try:
                    managed_groups.bind_provision_chat(
                        provision_id,
                        retained_chat_id,
                    )
                except Exception:
                    logger.exception(
                        "create_team: retained Registry bind failed name=%s",
                        name,
                    )
                    self.reply_text(
                        message_id,
                        "⚠️ 保留团队群已通过身份校验，但 Registry 绑定仍未确认；"
                        "未新建群，请稍后重试。",
                    )
                    manager.release_team_name(name)
                    return
            new_chat_id = recovered_chat_id
        else:
            if managed_groups is not None and not managed_groups.prepare_create_dispatch(
                provision_id,
                dispatched_at=datetime.now(UTC),
            ):
                persisted = manager.block_team_name_for_cleanup(
                    name,
                    "outcome_unknown",
                    "create_window_expired",
                )
                self.reply_text(
                    message_id,
                    "⚠️ 上次建群结果仍不确定且去重窗口已过，已阻止重复建群；请人工核对飞书群。"
                    + (
                        ""
                        if persisted
                        else " 恢复记录持久化失败，当前进程已失败关闭。"
                    ),
                )
                if persisted:
                    manager.release_team_name(name)
                return
            try:
                result = lark_client.create_chat(
                    name=group_name,
                    description=f"Slock 协作团队: {name}",
                    user_id_list=[sender_open_id],
                    operation_id=provision_id,
                )
                new_chat_id = result.chat_id
                if managed_groups is not None:
                    managed_groups.bind_provision_chat(provision_id, new_chat_id)
            except Exception as e:
                from src.utils.errors import redact_sensitive

                from ...project_chat.errors import (
                    CreateChatError,
                    CreateChatFailureDisposition,
                )
                logger.error("create_team: 建群/绑定失败 name=%s err=%s", name, redact_sensitive(str(e)))
                from ...trust.registry import RegistryCommitUncertainError

                if (
                    isinstance(e, RegistryCommitUncertainError)
                    and e.committed
                    and "new_chat_id" in locals()
                ):
                    persisted = manager.block_team_name_for_cleanup(
                        name,
                        new_chat_id,
                        "registry_bind_uncertain",
                    )
                    self.reply_text(
                        message_id,
                        "⚠️ 受管团队远端绑定结果不确定，已保留飞书群且不会执行删除；"
                        + (
                            "服务恢复核对后可安全重试。"
                            if persisted
                            else "恢复记录持久化失败，当前进程已失败关闭。"
                        ),
                    )
                    if persisted:
                        manager.release_team_name(name)
                    return
                if "new_chat_id" in locals():
                    persisted = manager.block_team_name_for_cleanup(
                        name,
                        new_chat_id,
                        "untrusted_retained",
                    )
                    self.reply_text(
                        message_id,
                        f"⚠️ 创建团队群失败: {safe_error_message(e)}；"
                        "为避免竞态删错群，已保留远端群但不授予信任，"
                        + (
                            "请由 Owner 核对。"
                            if persisted
                            else "本地恢复记录持久化失败，当前进程已失败关闭。"
                        ),
                    )
                elif (
                    managed_groups is not None
                    and isinstance(e, CreateChatError)
                    and e.disposition
                    is CreateChatFailureDisposition.OUTCOME_UNKNOWN
                ):
                    managed_groups.mark_create_outcome_unknown(provision_id)
                    persisted = manager.block_team_name_for_cleanup(
                        name,
                        "outcome_unknown",
                        "create_outcome_unknown",
                    )
                    self.reply_text(
                        message_id,
                        "⚠️ 建群结果不确定，已保持失败关闭；"
                        + (
                            "请在去重窗口内重试原命令。"
                            if persisted
                            else "恢复记录持久化失败，当前进程不会释放同名保留。"
                        ),
                    )
                else:
                    self.reply_text(
                        message_id,
                        f"❌ 创建团队群失败: {safe_error_message(e)}",
                    )
                    self._abandon_managed_group_provision(
                        managed_groups, provision_id
                    )
                if persisted:
                    manager.release_team_name(name)
                return

        engine = None
        registry_committed = False
        release_reservation = True

        try:
            # Step 2: Promote sender to group manager
            lark_client.add_managers(new_chat_id, [sender_open_id])

            # Step 3: Initialize slock engine for the new group
            engine_name = self.get_engine_name(
                new_chat_id, project_id=(project.project_id if project else None)
            )
            # Step 4: Activate channel (creates workspace directory + marker)
            channel = SlockChannel(
                channel_id=new_chat_id,
                name=group_name,
                team_name=name,
                owner_id=sender_open_id,
            )
            engine = manager.get_or_create_activated(
                new_chat_id,
                root_path,
                channel,
                engine_name=engine_name,
            )

            # Wire UI callbacks for escalation timeout notifications
            engine.set_escalation_ui_callbacks(
                update_card_fn=self.update_card,
                send_text_fn=self.send_text_to_chat,
            )

            if managed_groups is not None:
                managed_groups.activate(
                    provision_id=provision_id,
                    chat_id=new_chat_id,
                    project_id=project_id,
                    canonical_root_ref=root_path,
                )
                registry_committed = True
                if retained_chat_id is not None and not (
                    manager.consume_retained_team_name(name, new_chat_id)
                ):
                    release_reservation = False
                    self.reply_text(
                        message_id,
                        f"⚠️ 团队 **{name}** 已获得信任，但恢复记录未能持久清理；"
                        "请重试原命令完成核对。",
                    )
                    return

            # Runtime bootstrap follows durable Team bind + Registry ACTIVE.
            self._bootstrap_default_roles_if_configured(
                engine, channel.channel_id, new_chat_id
            )

            # Step 6: Send welcome card in the new group
            from ...slock_engine.card_templates import build_welcome_card

            welcome_card = build_welcome_card(team_name=name)
            try:
                welcome_message_id = self.send_card_to_chat(
                    new_chat_id, json.dumps(welcome_card, ensure_ascii=False)
                )
            except Exception:
                logger.exception("create_team: welcome card delivery raised chat=%s", new_chat_id)
                welcome_message_id = None
            if not welcome_message_id:
                self.send_text_to_chat(
                    new_chat_id,
                    f"⚠️ 欢迎卡发送失败，但团队 **{name}** 已创建。可发送 `/slock` 打开控制台。",
                )

            # Step 7: Send confirmation with jump link in original group
            from ...slock_engine.card_templates import build_team_created_card

            card = build_team_created_card(
                team_name=name,
                group_name=group_name,
                channel_id=new_chat_id,
            )
            try:
                confirmation_message_id = self.reply_card(
                    message_id, json.dumps(card, ensure_ascii=False)
                )
            except Exception:
                logger.exception("create_team: confirmation card delivery raised chat=%s", new_chat_id)
                confirmation_message_id = None
            if not confirmation_message_id:
                self.reply_text(
                    message_id,
                    f"✅ 团队已创建：**{name}**（群 ID: `{new_chat_id}`）。确认卡发送失败，请在飞书群列表中打开该群。",
                )

        except Exception as e:
            from src.utils.errors import redact_sensitive

            from ...trust.registry import RegistryCommitUncertainError

            if registry_committed:
                logger.exception(
                    "create_team: post-ACTIVE bootstrap/delivery degraded chat=%s",
                    new_chat_id,
                )
                self.reply_text(
                    message_id,
                    f"⚠️ 团队 **{name}** 已创建并获得信任，但初始化未完整完成；请在团队群内重试 `/slock`。",
                )
                return
            if isinstance(e, RegistryCommitUncertainError) and e.committed:
                logger.error(
                    "create_team: registry commit uncertain chat=%s; "
                    "preserving remote group and stopping local runtime",
                    new_chat_id,
                )
                if engine is not None:
                    try:
                        manager.archive_managed_chat_marker(new_chat_id)
                        engine.deactivate()
                        manager.unregister_managed_chat(new_chat_id)
                        manager.remove(new_chat_id, root_path)
                    except Exception:
                        logger.exception(
                            "create_team: uncertain-commit local stop failed chat=%s",
                            new_chat_id,
                        )
                release_reservation = False
                self.reply_text(
                    message_id,
                    f"⚠️ 团队 **{name}** 登记结果不确定，已停止继续操作且不会删除飞书群；"
                    "服务恢复核对前该群不获得信任。",
                )
                return
            logger.error("create_team: 激活失败, 回滚本地状态 chat=%s err=%s", new_chat_id, redact_sensitive(str(e)))
            local_rollback_ok = True
            if engine is not None:
                try:
                    manager.archive_managed_chat_marker(new_chat_id)
                    engine.deactivate()
                    manager.unregister_managed_chat(new_chat_id)
                    manager.remove(new_chat_id, root_path)
                except Exception:
                    local_rollback_ok = False
                    logger.exception("create_team: 本地回滚失败 chat=%s", new_chat_id)
            detail = safe_error_message(e)
            local_state = "本地状态已回滚" if local_rollback_ok else "本地状态回滚失败"
            persisted = manager.block_team_name_for_cleanup(
                name, new_chat_id, "untrusted_retained"
            )
            release_reservation = persisted
            self.reply_text(
                message_id,
                f"⚠️ 团队激活失败，{local_state}；"
                f"远端群已保留但不授予信任，"
                + (
                    f"请由 Owner 核对: {detail}"
                    if persisted
                    else f"恢复记录持久化失败，当前进程已失败关闭: {detail}"
                ),
            )
        finally:
            if release_reservation:
                manager.release_team_name(name)

    def _managed_group_registry(self):
        """Return only the Registry explicitly injected on shared context."""

        registry = vars(self.ctx).get("managed_group_registry")
        if registry is None:
            return None
        required = (
            "begin_provision",
            "bind_provision_chat",
            "provision_chat_id",
            "activate",
            "begin_revoke",
            "cancel_revoke",
            "complete_revoke",
            "pending_revokes",
            "record",
            "tombstone",
        )
        if all(callable(getattr(registry, name, None)) for name in required):
            return registry
        return None

    def _managed_group_owner_id(self) -> str:
        explicit = vars(self.ctx).get("managed_group_owner_id")
        if isinstance(explicit, str) and explicit:
            return explicit
        from ...trust.registry import single_owner_id

        return single_owner_id(getattr(self.ctx.settings, "admin_user_ids", ""))

    def _managed_group_receiving_bot_ref(self) -> str:
        explicit = vars(self.ctx).get("managed_group_receiving_bot_ref")
        if isinstance(explicit, str) and explicit:
            return explicit
        value = getattr(self.ctx.settings, "app_id", "")
        return value if isinstance(value, str) else ""

    @staticmethod
    def _abandon_managed_group_provision(registry, provision_id: str) -> bool:
        if registry is None:
            return False
        try:
            return bool(registry.abandon_provision(provision_id))
        except Exception:
            logger.exception("failed to abandon Team managed-group provision")
            return False

    @staticmethod
    def _format_slock_group_name(name: str, suffix: str = "[Slock]") -> str:
        """Format Slock team group names with a visible suffix marker."""
        clean_name = (name or "").strip()
        clean_suffix = (suffix or "").strip()
        if not clean_suffix:
            return clean_name
        if clean_name.casefold().endswith(clean_suffix.casefold()):
            return clean_name
        separator = "" if suffix.startswith((" ", "-", "_", "/", "|", "｜", ":", "：")) else " "
        return f"{clean_name}{separator}{suffix}"

    def list_teams(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """List all active Slock teams."""
        manager = self._get_engine_manager()
        engines = manager.list_activated_engines()

        if not engines:
            self.reply_text(message_id, "当前没有活跃的团队\n\n发送 `/slock` 激活协作模式")
            return

        teams = []
        for engine in sorted(engines, key=lambda item: (item.channel.team_name if item.channel else "")):
            channel = engine.channel
            if not channel:
                continue
            agents = engine.registry.list_agents(channel_id=channel.channel_id)
            agent_count = len(agents)
            task_count = len(engine.tasks)
            teams.append(
                {
                    "team_name": channel.team_name or channel.name or channel.channel_id,
                    "name": channel.name,
                    "channel_id": channel.channel_id,
                    "agent_count": agent_count,
                    "task_count": task_count,
                }
            )

        from ...slock_engine.card_templates import build_team_list_card

        self.reply_card(message_id, json.dumps(build_team_list_card(teams), ensure_ascii=False))

    def show_team_status(
        self, message_id: str, chat_id: str, name: str = "", project: Optional["ProjectContext"] = None
    ):
        """Show status of a specific team."""
        manager = self._get_engine_manager()
        engine = manager.find_team(name) if name else manager.get_activated_engine(chat_id)
        if not engine or not engine.channel:
            self.reply_text(message_id, f"未找到团队: **{redact_sensitive(name)}**" if name else "当前没有活跃的团队")
            return

        team_name = engine.channel.team_name if engine.channel else ""
        status_card = engine.get_status_card(team_name=team_name)
        sent_msg_id = self.reply_card(message_id, json.dumps(status_card, ensure_ascii=False))
        # Task 30: Register status card message_id for auto-refresh
        if sent_msg_id and engine.channel:
            engine._status_card_msg_ids[engine.channel.channel_id] = sent_msg_id

    def dissolve_team(
        self, message_id: str, chat_id: str, name: str = "", project: Optional["ProjectContext"] = None
    ):
        """Dissolve (stop) a team."""
        manager = self._get_engine_manager()
        engine = manager.find_team(name) if name else manager.get_activated_engine(chat_id)
        if not engine or not engine.channel:
            self.reply_text(message_id, f"未找到团队: **{redact_sensitive(name)}**" if name else "当前没有活跃的团队")
            return

        if not self._check_slock_permission(engine, message_id, chat_id):
            return

        target_chat_id = engine.channel.channel_id
        if not manager.claim_dissolve(target_chat_id):
            self.reply_text(message_id, "⚠️ 该团队正在解散，请勿重复提交。")
            return
        try:
            self._dissolve_team_claimed(message_id, manager, engine)
        finally:
            manager.release_dissolve(target_chat_id)

    def _dissolve_team_claimed(self, message_id: str, manager, engine) -> None:
        """Run one serialized local+Feishu dissolve transaction."""
        target_chat_id = engine.channel.channel_id
        team_name = engine.channel.team_name or engine.channel.name or target_chat_id
        channel_snapshot = engine.channel
        root_path = engine.root_path
        registry = self._managed_group_registry()
        revoke_started = False
        if registry is not None and registry.record(target_chat_id) is not None:
            try:
                registry.begin_revoke(
                    target_chat_id,
                    requested_at=datetime.now(UTC),
                )
                revoke_started = True
            except Exception:
                logger.exception(
                    "dissolve_team: durable revoke intent failed chat=%s",
                    target_chat_id,
                )
                self.reply_text(
                    message_id,
                    f"❌ 团队 **{team_name}** 未解散：受管群撤销意图无法持久化，请重试。",
                )
                return

        def _cancel_revoke() -> bool:
            if not revoke_started:
                return True
            try:
                return bool(registry.cancel_revoke(target_chat_id))
            except Exception:
                logger.exception(
                    "dissolve_team: failed to cancel revoke chat=%s",
                    target_chat_id,
                )
                return False

        try:
            archived_marker = manager.archive_managed_chat_marker(target_chat_id)
        except (OSError, ValueError) as e:
            logger.error(
                "dissolve_team: 归档本地 marker 失败 chat=%s err=%s",
                target_chat_id,
                redact_sensitive(str(e)),
            )
            marker_restored = bool(getattr(e, "restored", True))
            cancelled = _cancel_revoke() if marker_restored else False
            self.reply_text(
                message_id,
                (
                    f"❌ 团队 **{team_name}** 未解散：本地状态归档失败，撤销意图已取消，请重试。"
                    if cancelled
                    else f"⚠️ 团队 **{team_name}** 未解散，但 active marker 恢复或撤销取消失败；该群保持失败关闭，请人工检查。"
                ),
            )
            return
        if archived_marker is None:
            cancelled = _cancel_revoke()
            self.reply_text(
                message_id,
                (
                    f"❌ 团队 **{team_name}** 缺少活动 marker，撤销意图已取消，未执行删群。"
                    if cancelled
                    else f"⚠️ 团队 **{team_name}** 缺少活动 marker，且撤销取消无法持久化；该群保持失败关闭。"
                ),
            )
            return

        def _restore_local_team() -> bool:
            try:
                if archived_marker:
                    manager.restore_archived_chat_marker(target_chat_id, archived_marker)
                manager.discard_engine_for_recovery(target_chat_id, root_path)
                manager.get_or_create_activated(
                    target_chat_id,
                    root_path,
                    channel_snapshot,
                    engine_name="Slock",
                )
                return True
            except Exception:
                logger.exception("dissolve_team: 本地团队补偿恢复失败 chat=%s", target_chat_id)
                return False

        try:
            engine.deactivate()
            manager.unregister_managed_chat(target_chat_id)
            manager.remove(target_chat_id, root_path)
        except Exception as e:
            restored = _restore_local_team()
            if restored:
                _cancel_revoke()
            logger.error(
                "dissolve_team: 本地运行时拆除失败 chat=%s err=%s",
                target_chat_id,
                redact_sensitive(str(e)),
            )
            self.reply_text(message_id, f"❌ 团队 **{team_name}** 未完整解散，请重试。")
            return
        from ...project_chat.lark_chat_client import LarkChatClient

        lark_client = LarkChatClient(api_client_factory=self.ctx.api_client_factory)
        delete_result = lark_client.delete_chat(target_chat_id)
        if delete_result is False:
            restored = _restore_local_team()
            if restored:
                _cancel_revoke()
            self.reply_text(
                message_id,
                (
                    f"❌ 团队 **{team_name}** 未解散：飞书删群失败，本地团队已恢复，请重试。"
                    if restored
                    else f"⚠️ 团队 **{team_name}** 飞书删群失败，且本地恢复失败，请人工处理。"
                ),
            )
            return
        if delete_result is None:
            manager = self._get_engine_manager()
            residual_persisted = manager.block_team_name_for_cleanup(
                team_name,
                target_chat_id,
                "unknown",
            )
            if not self._tombstone_managed_team(
                message_id, target_chat_id, team_name
            ):
                return
            self.reply_text(
                message_id,
                f"⚠️ 团队 **{team_name}** 的飞书删群结果未知；本地已停止且不会自动恢复，请人工确认群状态。\n"
                + (
                    "同名团队创建已持久阻止，直到人工处理完成。"
                    if residual_persisted
                    else "同名团队当前进程已阻止，但恢复记录持久化失败。"
                ),
            )
            return

        if not self._tombstone_managed_team(
            message_id, target_chat_id, team_name
        ):
            return

        self.reply_text(message_id, f"✅ 团队 **{team_name}** 已解散并归档本地状态")

    def _tombstone_managed_team(
        self,
        message_id: str,
        chat_id: str,
        team_name: str,
    ) -> bool:
        registry = self._managed_group_registry()
        if registry is None or registry.record(chat_id) is None:
            return True
        try:
            if chat_id in registry.pending_revokes():
                registry.complete_revoke(chat_id)
            else:
                registry.tombstone(chat_id)
            return True
        except Exception:
            logger.exception(
                "dissolve_team: Registry tombstone failed chat=%s", chat_id
            )
            self.reply_text(
                message_id,
                f"⚠️ 团队 **{team_name}** 已停止，但受管群撤销记录失败；请立即重试。",
            )
            return False

    # ------------------------------------------------------------------
    # Role / Agent management
    # ------------------------------------------------------------------

    # tool_type → default role inference mapping
    TOOL_TYPE_ROLE_MAP: dict[str, str] = {
        "traex": "coder",
        "codex": "coder",
        "claude": "reviewer",
        "coco": "writer",
        "aiden": "coder",
        "gemini": "coder",
    }

    # role → default personality traits (used when --traits not specified)
    DEFAULT_PERSONALITY_TRAITS: dict[str, list[str]] = {
        "coder": ["严谨", "注重细节"],
        "reviewer": ["批判性思维", "追求质量"],
        "tester": ["细致", "追求覆盖"],
        "planner": ["全局视角", "有条理"],
        "architect": ["抽象思维", "系统设计"],
        "writer": ["表达清晰", "注重结构"],
        "custom": [],
    }
    TOOL_SELECT_OPTIONS: tuple[dict[str, str], ...] = (
        {"name": "traex", "label": "Traex", "emoji": "🚀", "description": "默认编程工具"},
        {"name": "coco", "label": "Coco", "emoji": "🥥", "description": "协作工具"},
        {"name": "codex", "label": "Codex", "emoji": "🧠", "description": "代码实现"},
        {"name": "aiden", "label": "Aiden", "emoji": "🎯", "description": "AI 编程助手"},
        {"name": "claude", "label": "Claude", "emoji": "📝", "description": "评审与长文"},
        {"name": "gemini", "label": "Gemini", "emoji": "💎", "description": "多模态与代码"},
    )

    def create_role(
        self, message_id: str, chat_id: str, name: str = "", project: Optional["ProjectContext"] = None,
        *, global_hire: bool = False,
    ):
        """Create a Slock virtual role or dispatch an independent employee hire.

        Supports parameter syntax:
            /hire <name> [--app-id <cli_...>] [--tool <type>] [--model <model>]
            /new-role <name> [--tool <type>] [--model <model>] [--emoji <e>] [--role <role>] [--prompt <text>]

        ``global_hire=True`` never writes the legacy registry. It dispatches to
        the injected employee provisioning service or fails closed.
        """
        if not name:
            self.reply_text(
                message_id,
                "请提供员工名称\n\n用法: `/hire <名称>` [--tool codex] [--model o3-pro] "
                "[--app-id cli_...] [--emoji 🔧] [--role coder] [--prompt <约束描述>]",
            )
            return

        engine = None
        if global_hire:
            if self._visible_hire_requester(message_id) is None:
                return
        else:
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(chat_id)
            if not engine:
                self.reply_text(message_id, "请先激活 Slock 模式: `/slock`")
                return
            if not self._check_slock_permission(engine, message_id, chat_id):
                return

        # Parse optional arguments from the name/args string
        try:
            tokens = shlex.split(name)
        except ValueError:
            tokens = name.split()

        role_name = tokens[0] if tokens else name

        tool_type = "traex"
        model_name = ""
        emoji = "🤖"
        system_prompt = ""
        explicit_role: str | None = None
        template_name = ""
        fork_name = ""
        explicit_traits: str | None = None
        profile = "standard"
        effort = "default"
        tool_explicit = False
        model_explicit = False
        emoji_explicit = False
        prompt_explicit = False
        existing_app_id = ""

        i = 1
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--tool" and i + 1 < len(tokens):
                tool_type = tokens[i + 1]
                tool_explicit = True
                i += 2
            elif tok == "--model" and i + 1 < len(tokens):
                model_name = tokens[i + 1]
                model_explicit = True
                i += 2
            elif tok == "--effort" and i + 1 < len(tokens):
                effort = tokens[i + 1]
                i += 2
            elif tok == "--profile" and i + 1 < len(tokens):
                profile = tokens[i + 1]
                i += 2
            elif tok == "--emoji" and i + 1 < len(tokens):
                emoji = tokens[i + 1]
                emoji_explicit = True
                i += 2
            elif tok == "--role" and i + 1 < len(tokens):
                explicit_role = tokens[i + 1]
                i += 2
            elif tok == "--prompt" and i + 1 < len(tokens):
                system_prompt = tokens[i + 1]
                prompt_explicit = True
                i += 2
            elif tok in {"--app-id", "--existing-app-id"}:
                if i + 1 >= len(tokens):
                    self.reply_text(message_id, "请在 `--app-id` 后提供有效的飞书 App ID（cli_...）。")
                    return
                try:
                    existing_app_id = _normalize_existing_hire_app_id(tokens[i + 1])
                except ValueError:
                    self.reply_text(message_id, "飞书 App ID 无效；应为 `cli_` 开头的开放平台应用 ID。")
                    return
                i += 2
            elif tok == "--template" and i + 1 < len(tokens):
                template_name = tokens[i + 1]
                i += 2
            elif tok == "--fork" and i + 1 < len(tokens):
                fork_name = tokens[i + 1]
                i += 2
            elif tok == "--traits" and i + 1 < len(tokens):
                explicit_traits = tokens[i + 1]
                i += 2
            else:
                i += 1

        if existing_app_id and not global_hire:
            self.reply_text(message_id, "`--app-id` 仅用于 `/hire` 独立飞书员工。")
            return

        app_only_selection = len(tokens) == 3 and tokens[1] in {
            "--app-id",
            "--existing-app-id",
        }
        if len(tokens) == 1 or (global_hire and app_only_selection):
            self.show_new_role_tool_selection(
                message_id,
                role_name,
                project,
                global_hire=global_hire,
                existing_app_id=existing_app_id,
            )
            return

        if global_hire:
            if prompt_explicit:
                self.reply_text(
                    message_id,
                    "`/hire` 不接受任意 `--prompt`；员工工作风格由受控角色模板生成。",
                )
                return
            traits = tuple(
                value.strip()
                for value in (explicit_traits or "").split(",")
                if value.strip()
            )
            self._start_visible_employee_hire(
                message_id=message_id,
                chat_id=chat_id,
                employee_name=role_name,
                tool=tool_type,
                model=model_name,
                profile=profile,
                effort=effort,
                role=explicit_role or "",
                personality_traits=traits,
                existing_app_id=existing_app_id,
            )
            return

        # Validate tool_type against whitelist
        from ...slock_engine.models import AGENT_ROLE_COLORS, AgentIdentity, SlockMemory

        template_data: dict = {}
        if template_name:
            raw_template = engine.memory.read_agent_template(template_name)
            template_data = raw_template if isinstance(raw_template, dict) else {}
            if not template_data:
                self.reply_text(message_id, f"❌ 未找到 Agent 模板: `{template_name}`")
                return
            if not tool_explicit:
                tool_type = template_data.get("tool_type", tool_type)
            if not model_explicit:
                model_name = template_data.get("model_name", model_name)
            if not emoji_explicit:
                emoji = template_data.get("emoji", emoji)
            if explicit_role is None:
                explicit_role = template_data.get("role") or explicit_role
            if not prompt_explicit:
                system_prompt = template_data.get("system_prompt", system_prompt)

        fork_source: AgentIdentity | None = None
        if fork_name:
            registry = engine.registry if engine else self._get_global_registry()
            fork_source = registry.find_by_name(fork_name, channel_id=chat_id) or registry.find_by_name(fork_name)
            if not isinstance(fork_source, AgentIdentity):
                self.reply_text(message_id, f"❌ 未找到可 fork 的角色: `{fork_name}`")
                return
            if not tool_explicit:
                tool_type = fork_source.agent_type
            if not model_explicit:
                model_name = fork_source.model_name
            if not emoji_explicit:
                emoji = fork_source.emoji
            if explicit_role is None:
                explicit_role = fork_source.role
            if not prompt_explicit:
                system_prompt = fork_source.system_prompt

        VALID_TOOLS = set(self.TOOL_TYPE_ROLE_MAP.keys())
        if tool_type not in VALID_TOOLS:
            self.reply_text(
                message_id,
                f"❌ 无效的 tool_type: `{tool_type}`\n"
                f"合法值: {', '.join(sorted(VALID_TOOLS))}",
            )
            return

        # Validate explicit role against whitelist
        VALID_ROLES = set(AGENT_ROLE_COLORS.keys())
        if explicit_role and explicit_role not in VALID_ROLES:
            self.reply_text(
                message_id,
                f"❌ 无效的 role: `{explicit_role}`\n"
                f"合法值: {', '.join(sorted(VALID_ROLES))}",
            )
            return

        # Determine role: explicit --role takes priority, otherwise infer from tool_type
        if explicit_role:
            role = explicit_role
        else:
            role = self.TOOL_TYPE_ROLE_MAP.get(tool_type, "custom")

        # Assign unique emoji if not explicitly provided
        if not emoji_explicit:
            from ...slock_engine.role_bootstrap import pick_unique_emoji
            registry = engine.registry if engine else self._get_global_registry()
            used_emojis = {a.emoji for a in registry.list_agents() if hasattr(a, 'emoji')}
            emoji = pick_unique_emoji(role, used_emojis)

        runtime_agent_type = self._resolve_slock_runtime_agent_type(tool_type)
        agent_id = f"{runtime_agent_type}:{model_name or 'default'}:{role_name}"
        registry = engine.registry if engine else self._get_global_registry()
        existing_raw = registry.get(agent_id)
        existing_agent = existing_raw if isinstance(existing_raw, AgentIdentity) else None
        if existing_agent and not system_prompt:
            system_prompt = existing_agent.system_prompt
        if not system_prompt:
            system_prompt = self._build_default_directive(
                role_name=role_name,
                role=role,
                tool_type=tool_type,
                model_name=model_name,
                team_name=(engine.channel.team_name if engine and engine.channel else chat_id),
            )

        # Determine personality traits: explicit --traits overrides default role mapping
        if explicit_traits:
            personality_traits = [t.strip() for t in explicit_traits.replace("，", ",").split(",") if t.strip()]
        else:
            personality_traits = list(self.DEFAULT_PERSONALITY_TRAITS.get(role, []))

        agent = AgentIdentity(
            agent_id=agent_id,
            name=role_name,
            emoji=emoji,
            agent_type=runtime_agent_type,
            model_name=model_name,
            system_prompt=system_prompt,
            role=role,
            owner_group=chat_id,
            member_groups=[chat_id],
            personality_traits=personality_traits,
        )
        workspace_paths = engine.memory.initialize_agent_workspace(agent.agent_id)
        agent.memory_path = workspace_paths.get("memory_path") or engine.memory.agent_memory_path(agent.agent_id)
        agent.notes_path = workspace_paths.get("notes_path") or engine.memory.agent_notes_path(agent.agent_id)
        agent.workspace_path = workspace_paths.get("workspace_path") or engine.memory.agent_workspace_path(agent.agent_id)
        if not existing_agent:
            if fork_source is not None:
                source_memory = engine.memory.read_agent_memory(fork_source.agent_id)
                active_context = source_memory.active_context
                fork_entry = f"Forked from {fork_source.agent_id} into Slock team {chat_id}."
                active_context = f"{active_context}\n{fork_entry}".strip() if active_context else fork_entry
                engine.memory.write_agent_memory(
                    agent.agent_id,
                    SlockMemory(
                        role=source_memory.role or system_prompt,
                        key_knowledge=source_memory.key_knowledge,
                        active_context=active_context,
                    ),
                )
                source_profiles = engine.memory.read_skill_profiles(fork_source.agent_id)
                engine.memory.write_skill_profiles(agent.agent_id, source_profiles)
            else:
                key_knowledge = template_data.get("key_knowledge") or (
                    f"tool_type={tool_type}\nruntime_agent_type={runtime_agent_type}\n"
                    f"model={model_name or 'default'}\nrole={role}"
                )
                engine.memory.write_agent_memory(
                    agent.agent_id,
                    SlockMemory(
                        role=system_prompt,
                        key_knowledge=key_knowledge,
                        active_context=f"Created in Slock team {chat_id}.",
                    ),
                )
        else:
            engine.memory.update_agent_context(agent.agent_id, f"Joined Slock team {chat_id}.")
        engine.registry.register(agent)

        self.reply_text(
            message_id,
            f"✅ 角色 **{agent.emoji} {agent.name}** 已创建 (ID: `{agent.agent_id[:8]}`)\n"
            f"   工具: `{tool_type}` | 运行时: `{runtime_agent_type}` | 模型: `{model_name or '默认'}` | "
            f"角色: `{role}` | Emoji: {emoji}",
        )

    @staticmethod
    def _resolve_slock_runtime_agent_type(tool_type: str) -> str:
        """Map role creation tool choice to an executable agent session backend."""
        return tool_type

    def _selectable_role_tool_options(self) -> list[dict[str, str]]:
        """Return role-creation tools executable in the current environment."""
        try:
            from ...workflow_engine.tool_registry import get_available_tools

            available = set(get_available_tools(require_available=True).keys())
        except Exception:
            logger.debug("failed to resolve selectable role tools", exc_info=True)
            available = set()
        if not available:
            return []
        return [
            option
            for option in self.TOOL_SELECT_OPTIONS
            if str(option.get("name") or "") in available
        ]

    def show_new_role_tool_selection(
        self,
        message_id: str,
        role_name: str,
        project: Optional["ProjectContext"] = None,
        *,
        global_hire: bool = False,
        existing_app_id: str = "",
    ) -> None:
        """Show unified /hire card: tool dropdown + model cascade in one card."""

        tool_options = self._selectable_role_tool_options()
        if not tool_options:
            self.reply_text(message_id, "当前环境未检测到可用的 Slock 编程工具，请安装 Traex/Claude/Codex 等 CLI 后重试。")
            return

        default_tool = tool_options[0]["name"]
        cwd = getattr(project, "root_path", None) or self.get_working_dir(None)
        models = fetch_acp_models(default_tool, cwd=cwd, current_model=None)

        value_extra: dict = {"global_hire": True} if global_hire else {}
        if global_hire and existing_app_id:
            value_extra["existing_app_id"] = existing_app_id
        _, card_content = CardBuilder.build_slock_role_unified_select_card(
            role_name=role_name,
            tools=tool_options,
            models=models,
            tool_name=default_tool,
            project_id=(project.project_id if project else None),
            profile_markdown=self._visible_employee_profile_markdown(
                role_name,
                default_tool,
                "",
            ) if global_hire else None,
            value_extra=value_extra,
        )
        self.reply_card(message_id, card_content)

    @staticmethod
    def _visible_employee_profile_markdown(
        employee_name: str,
        tool: str,
        model: str,
    ) -> str:
        from ...autonomous.manager.cards import build_employee_profile_markdown
        from ...autonomous.provisioning.hire_port import recommended_employee_profile

        return build_employee_profile_markdown(
            employee_name=employee_name,
            tool=tool,
            model=model,
            profile=recommended_employee_profile(tool),
        )

    def _start_visible_employee_hire(
        self,
        *,
        message_id: str,
        chat_id: str,
        employee_name: str,
        tool: str,
        model: str,
        profile: str,
        effort: str,
        role: str = "",
        personality_traits: tuple[str, ...] = (),
        existing_app_id: str = "",
    ) -> None:
        from ...autonomous.provisioning.hire_port import (
            EmployeeHireRequest,
            recommended_employee_profile,
        )
        from ...autonomous.provisioning.hire_service import HireAdmissionError
        from ...thread.manager import (
            get_current_sender_union_id,
            get_current_tenant_key,
        )
        requester_id = self._visible_hire_requester(message_id)
        if requester_id is None:
            return
        try:
            existing_app_id = _normalize_existing_hire_app_id(existing_app_id)
        except ValueError:
            self.reply_text(message_id, "飞书 App ID 无效；应为 `cli_` 开头的开放平台应用 ID。")
            return
        if tool not in self.TOOL_TYPE_ROLE_MAP:
            self.reply_text(message_id, f"❌ 无效的员工工具: `{tool}`")
            return
        readiness_blockers = self._visible_hire_readiness_blockers()
        if readiness_blockers:
            blocker_labels = {
                "visible_employee_limit": "autonomous_visible_employee_limit=0（可见员工发布开关未放行）",
                "release_evidence": "QA release evidence（独立发布证明未就绪）",
                "registration_notifier": "注册链接通知通道未就绪",
                "main_bot_send_audit": "主 Bot 零代发审计未就绪",
                "main_bot_activation_fence": "主 Bot 激活并发围栏未就绪",
                "production_anchor": "生产 Journal anchor 未就绪",
                "worker_sandbox": "员工进程 sandbox 证明未就绪",
                "durable_configuration": "Journal/Vault 持久化配置未就绪",
                "runtime_composition": "员工运行时组装失败",
                "runtime_recovery": "员工运行时恢复尚未完成",
                "credential_keyring": "员工凭证 Keyring 未就绪",
                "admission_closed": "员工创建入口已关闭",
                "closed": "员工运行时已关闭",
                "not_composed": "员工运行时尚未组装",
                "readiness_unavailable": "readiness 诊断未注入",
                "readiness_probe": "readiness 诊断执行失败",
                "readiness_invalid": "readiness 诊断返回无效",
                "readiness_not_ready": "readiness 未就绪且未提供具体原因",
            }
            details = "；".join(
                blocker_labels.get(blocker, f"{blocker}（未满足）")
                for blocker in readiness_blockers
            )
            self.reply_text(
                message_id,
                f"独立飞书智能体创建被安全门禁阻断：{details}。已拒绝降级创建本地虚拟角色。",
            )
            return
        service = getattr(self.ctx, "employee_hire_service", None)
        if service is None:
            self.reply_text(
                message_id,
                "独立飞书智能体创建尚未接入生产服务；已拒绝降级创建本地虚拟角色。",
            )
            return
        requester_union_id = get_current_sender_union_id() or ""
        if not requester_union_id:
            self.reply_text(
                message_id,
                "无法确认跨应用用户身份（union_id 缺失），已拒绝创建员工。请从飞书主应用重新发起 /hire。",
            )
            return
        template = recommended_employee_profile(tool, role)
        request = EmployeeHireRequest(
            employee_name=employee_name,
            tool=tool,
            model=model,
            profile=profile,
            effort=effort,
            chat_id=chat_id,
            message_id=message_id,
            requester_principal_id=requester_id,
            requester_union_id=requester_union_id,
            tenant_key=get_current_tenant_key() or "",
            role=template.role,
            persona=template.persona,
            personality_traits=personality_traits or template.personality_traits,
            capabilities=template.capabilities,
            permissions=template.permissions,
            existing_app_id=existing_app_id,
        )
        try:
            service.start_hire(request)
        except HireAdmissionError as exc:
            reason = str(exc)
            logger.warning(
                "visible employee hire admission rejected: %s",
                reason,
            )
            if reason == "hire name conflict":
                self.reply_text(
                    message_id,
                    f"同名员工 `{employee_name}` 已存在或尚未完成退役。请先执行 `/roster`；"
                    f"若仍显示待处理，请再次执行 `/fire {employee_name}`。确认归档后再重新 `/hire`。",
                )
            elif reason == "visible_employee_limit capacity reached":
                self.reply_text(
                    message_id,
                    "可见员工数量已达到上限。请先执行 `/roster`，并用 `/fire <员工名称>` "
                    "完成不再需要员工的退役后重试。",
                )
            elif reason == "existing app already assigned":
                self.reply_text(
                    message_id,
                    f"飞书应用 `{existing_app_id}` 已绑定其他未归档员工。请先执行 `/roster`，"
                    "并用 `/fire <员工名称>` 完成原员工退役后再重试。",
                )
            else:
                self.reply_text(
                    message_id,
                    "独立飞书智能体创建被准入层安全拒绝。请先执行 `/roster` 查看员工状态；"
                    "不要连续重复提交 `/hire`。",
                )
        except Exception as exc:
            logger.error("visible employee hire dispatch failed: %s", type(exc).__name__)
            self.reply_text(message_id, "独立飞书智能体创建启动失败，请查看安全日志后重试。")

    def list_current_team_members(self, message_id: str, chat_id: str) -> None:
        """List persistent employees with confirmed membership in this chat."""

        from ...autonomous.domain.enums import EmployeeState, WorkerType
        from ...thread.manager import get_current_tenant_key

        tenant_key = get_current_tenant_key() or ""
        if not tenant_key:
            self.reply_text(message_id, "⚠️ 无法确认租户身份。")
            return

        membership_service = getattr(
            self.ctx,
            "employee_membership_service",
            None,
        )

        hire_service = getattr(self.ctx, "employee_hire_service", None)
        if hire_service is None:
            self.reply_text(message_id, "员工管理服务尚未接入，无法查询当前团队。")
            return

        try:
            projection = hire_service.synchronize_projection()
            employees = [
                employee
                for employee in projection.employees.values()
                if employee.tenant_key == tenant_key
                and employee.worker_type is WorkerType.VISIBLE
                and employee.state is not EmployeeState.ARCHIVED
                and chat_id in getattr(employee, "member_groups", ())
            ]
        except Exception:
            logger.exception("current team employee membership query failed")
            self.reply_text(
                message_id,
                "⚠️ 无法确认当前群团队成员状态，请稍后重试。",
            )
            return

        if not employees:
            self.reply_text(
                message_id,
                "当前群没有可确认的员工成员。"
                "使用 `/role add` 将员工加入本群，或 `/roster` 查看全局花名册。",
            )
            return

        membership_degraded: dict[str, bool] = {}
        batch_health = (
            getattr(type(membership_service), "degraded_for", None)
            if membership_service is not None
            else None
        )
        if callable(batch_health):
            try:
                batch_result = membership_service.degraded_for(
                    (employee.agent_id for employee in employees),
                    chat_id,
                )
                if not isinstance(batch_result, dict):
                    raise TypeError("invalid membership health snapshot")
            except Exception:
                logger.exception(
                    "current team employee membership health snapshot failed"
                )
                batch_result = {}
            membership_degraded = {
                employee.agent_id: _membership_is_degraded(
                    batch_result.get(employee.agent_id, True)
                )
                for employee in employees
            }
        else:
            for employee in employees:
                if membership_service is None:
                    membership_degraded[employee.agent_id] = True
                    continue
                try:
                    membership_degraded[
                        employee.agent_id
                    ] = _membership_is_degraded(
                        membership_service.is_degraded(
                            employee.agent_id,
                            chat_id,
                        )
                    )
                except Exception:
                    logger.exception(
                        "current team employee membership health query failed"
                    )
                    membership_degraded[employee.agent_id] = True

        runtime_views: dict[str, object] = {}
        runtime_facade = getattr(self.ctx, "employee_runtime_facade", None)
        list_runtime = getattr(
            runtime_facade,
            "list_employee_runtime_statuses",
            None,
        )
        if callable(list_runtime):
            try:
                runtime_views = {
                    view.agent_id: view
                    for view in list_runtime(tenant_key)
                }
            except Exception:
                logger.exception("current team employee runtime query failed")

        lines: list[str] = []
        for employee in sorted(employees, key=lambda item: item.name):
            tool = _escape_lark_md_inline(
                getattr(employee, "tool", "") or "未配置"
            )
            model = _escape_lark_md_inline(getattr(employee, "model", "") or "")
            tool_model = f"{tool}/{model}" if model else tool
            emoji = _escape_lark_md_inline(
                getattr(employee, "emoji", "") or "🤖"
            )
            name = _escape_lark_md_inline(employee.name)
            membership_label = (
                "⚠️ 群关系待确认"
                if membership_degraded.get(employee.agent_id, True)
                else "✅ 已确认在群"
            )
            line = (
                f"{emoji} **员工**：{name}　{tool_model}　"
                f"{membership_label}"
            )
            role = _escape_lark_md_inline(
                getattr(employee, "role", "") or "未设置"
            )
            capabilities = ", ".join(
                _escape_lark_md_inline(capability)
                for capability in getattr(employee, "capabilities", ())
                if str(capability)
            ) or "未声明"
            if employee.state is EmployeeState.ACTION_REQUIRED:
                lifecycle = "⚠️ 待恢复"
            elif employee.state is EmployeeState.ACTIVE:
                lifecycle = "✅ 在职"
            elif employee.state is EmployeeState.RETIRING:
                lifecycle = "🔄 退休中"
            else:
                lifecycle = "⏳ 配置中"
            line += (
                f"　{lifecycle}\n  角色：{role}\n  能力：{capabilities}"
            )
            runtime_view = runtime_views.get(employee.agent_id)
            if runtime_view is not None:
                bot_state = str(
                    getattr(runtime_view, "bot_state", "unknown")
                ).upper()
                actor_state = str(
                    getattr(runtime_view, "actor_state", "unknown")
                ).upper()
                admission = (
                    "可接任务"
                    if not membership_degraded.get(employee.agent_id, True)
                    and employee.state is EmployeeState.ACTIVE
                    and getattr(runtime_view, "can_accept", False) is True
                    else "不可接任务"
                )
                line += (
                    f"\n  Bot {bot_state} / Agent {actor_state}　"
                    f"{admission}"
                )
            else:
                line += "\n  运行状态未知　暂不可确认接单"
            lines.append(line)

        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"当前团队成员（{len(employees)}人）",
                },
                "template": "green",
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": "\n\n".join(lines),
                    },
                },
            ],
        }
        self.reply_card(message_id, card)

    def list_employees_roster(self, message_id: str, chat_id: str) -> None:
        """List all tenant employees with state and group membership count."""

        from ...autonomous.domain.enums import EmployeeState, WorkerType
        from ...autonomous.provisioning.lark_app import (
            current_registration_manifest,
        )
        from ...thread.manager import (
            get_current_is_p2p,
            get_current_sender_id,
            get_current_tenant_key,
        )

        tenant_key = get_current_tenant_key() or ""
        if not tenant_key:
            self.reply_text(message_id, "⚠️ 无法确认租户身份。")
            return

        requester_id = get_current_sender_id() or ""
        show_admin_details = (
            get_current_is_p2p()
            and requester_id
            in getattr(self.ctx.settings, "admin_user_ids", frozenset())
        )
        runtime_facade = getattr(self.ctx, "employee_runtime_facade", None)
        runtime_views: dict[str, object] = {}
        list_runtime = getattr(runtime_facade, "list_employee_runtime_statuses", None)
        if callable(list_runtime):
            try:
                runtime_views = {
                    view.agent_id: view for view in list_runtime(tenant_key)
                }
            except Exception:
                logger.exception("employee runtime status facade failed")

        hire_service = getattr(self.ctx, "employee_hire_service", None)
        if hire_service is None:
            self.reply_text(message_id, "员工管理服务尚未接入，无法查询花名册。")
            return

        try:
            projection = hire_service.synchronize_projection()
        except Exception:
            self.reply_text(message_id, "⚠️ 员工投影同步失败，请稍后重试。")
            return

        tenant_employees = [
            emp for emp in projection.employees.values()
            if emp.tenant_key == tenant_key and emp.worker_type is WorkerType.VISIBLE
        ]
        archived_count = sum(
            emp.state is EmployeeState.ARCHIVED for emp in tenant_employees
        )
        employees = [
            emp for emp in tenant_employees
            if emp.state is not EmployeeState.ARCHIVED
        ]

        if not employees:
            history = f"（历史归档 {archived_count} 人）" if archived_count else ""
            self.reply_text(
                message_id,
                f"当前租户没有在职员工{history}。使用 `/hire <名字>` 开始雇佣。",
            )
            return

        _STATE_LABELS = {
            EmployeeState.ACTIVE: "✅ 就绪",
            EmployeeState.DRAFT: "📝 草稿",
            EmployeeState.PROVISIONING_APP: "⏳ 配置中",
            EmployeeState.STORING_CREDENTIAL: "⏳ 配置中",
            EmployeeState.CONFIGURING: "⏳ 配置中",
            EmployeeState.VALIDATING: "⏳ 验证中",
            EmployeeState.READY_PENDING_VERIFICATION: "⏳ 待验证",
            EmployeeState.ACTION_REQUIRED: "⚠️ 待处理",
            EmployeeState.RETIRING: "🔄 退休中",
        }

        hire_states: dict[str, object] = {}
        pending_fire_states: dict[str, list[object]] = {}
        if show_admin_details:
            list_hires = getattr(hire_service, "list_states", None)
            if callable(list_hires):
                for state in list_hires():
                    if getattr(state, "tenant_key", "") == tenant_key:
                        hire_states[getattr(state, "agent_id", "")] = state
            fire_service = getattr(self.ctx, "employee_fire_service", None)
            list_fires = getattr(fire_service, "list_states", None)
            if callable(list_fires):
                from ...autonomous.provisioning.fire_state import (
                    FireCleanupMode,
                    FirePhase,
                )

                for state in list_fires():
                    if (
                        getattr(state, "tenant_key", "") == tenant_key
                        and getattr(state, "phase", None)
                        is FirePhase.ACTION_REQUIRED
                        and getattr(state, "cleanup_mode", None)
                        is FireCleanupMode.EXTERNAL_UNKNOWN
                        and getattr(state, "error_code", "")
                        == "external_cleanup_authority_unavailable"
                        and getattr(
                            state,
                            "external_disposition_confirmed",
                            False,
                        )
                        is False
                    ):
                        pending_fire_states.setdefault(
                            getattr(state, "agent_id", ""),
                            [],
                        ).append(state)

        lines: list[str] = []
        manifest_actions: list[dict] = []
        current_manifest_hash = current_registration_manifest().fingerprint()
        trusted_manifest_evidence = "lark_oapi.aregister_app/exact_app_id"

        def short_manifest_hash(value: str) -> str:
            digest = value.removeprefix("sha256:")
            return f"sha256:{digest[:10]}…"

        for emp in sorted(employees, key=lambda e: e.name):
            state_label = _STATE_LABELS.get(emp.state, f"❓ {emp.state.value}")
            group_count = len(emp.member_groups)
            tool_model = emp.tool or "未配置"
            if emp.model:
                tool_model = f"{emp.tool}/{emp.model}"
            line = (
                f"{emp.emoji} **{emp.name}**　{tool_model}　"
                f"{state_label}　群×{group_count}"
            )
            runtime_view = runtime_views.get(emp.agent_id)
            if runtime_view is not None:
                bot_state = str(getattr(runtime_view, "bot_state", "unknown")).upper()
                actor_state = str(
                    getattr(runtime_view, "actor_state", "unknown")
                ).upper()
                admission = (
                    "可接任务"
                    if getattr(runtime_view, "can_accept", False) is True
                    else "不可接任务"
                )
                line += f"\n  Bot {bot_state} / Agent {actor_state}　{admission}"
                manifest_actions.extend(
                    build_responsive_layout(
                        [
                            {
                                "tag": "button",
                                "text": {
                                    "tag": "plain_text",
                                    "content": f"查看 {emp.name} 运行时",
                                },
                                "type": "default",
                                "value": {
                                    "action": "employee_runtime_show_status",
                                    "agent_id": emp.agent_id,
                                },
                            }
                        ]
                    )
                )
            if show_admin_details:
                details = [f"Employee ID: `{emp.agent_id}`"]
                principal = getattr(projection, "bot_principals", {}).get(
                    getattr(emp, "bot_principal_id", "")
                )
                app_id = ""
                if (
                    principal is not None
                    and getattr(principal, "tenant_key", "") == tenant_key
                    and getattr(principal, "agent_id", "") == emp.agent_id
                ):
                    app_id = str(getattr(principal, "app_id", "") or "")
                if not app_id:
                    hire_state = hire_states.get(emp.agent_id)
                    app_id = str(
                        getattr(hire_state, "app_id", "")
                        or getattr(hire_state, "existing_app_id", "")
                        or ""
                    )
                pending = pending_fire_states.get(emp.agent_id, [])
                if len(pending) == 1:
                    pending_ref = str(
                        getattr(pending[0], "app_id", "")
                        or "NO_APP_FOUND"
                    )
                    if pending_ref == "NO_APP_FOUND":
                        details.append("App ID: `未识别`（请先确认开放平台未创建应用）")
                    else:
                        app_id = pending_ref
                if app_id:
                    details.append(f"App ID: `{app_id}`")
                if principal is not None:
                    recorded_desired_hash = str(
                        getattr(principal, "desired_manifest_hash", "") or ""
                    )
                    observed_hash = str(
                        getattr(principal, "observed_manifest_hash", "") or ""
                    )
                    evidence_source = str(
                        getattr(principal, "manifest_evidence_source", "") or ""
                    )
                    has_trusted_evidence = (
                        evidence_source == trusted_manifest_evidence
                    )
                    if not has_trusted_evidence:
                        details.append(
                            "权限证据：❔ 未知（本地期望 "
                            f"`{short_manifest_hash(current_manifest_hash)}`；"
                            "远端未知，尚无可信官方回执）"
                        )
                    elif (
                        recorded_desired_hash == current_manifest_hash
                        and observed_hash == current_manifest_hash
                    ):
                        details.append(
                            "权限证据：✅ 当前 manifest "
                            f"`{short_manifest_hash(current_manifest_hash)}`"
                            "（飞书官方原 App 授权回执已锚定）"
                        )
                    else:
                        details.append(
                            "权限证据：⚠️ 漂移 "
                            f"`{short_manifest_hash(observed_hash)}` → "
                            f"`{short_manifest_hash(current_manifest_hash)}`\n"
                            "  请为原 App 原地重新授权并发布，无需 `/fire`；"
                            "旧回执不能证明当前远端状态，未验证飞书远端状态。"
                        )
                    if (
                        emp.state is EmployeeState.ACTIVE
                        and not (
                            has_trusted_evidence
                            and recorded_desired_hash == current_manifest_hash
                            and observed_hash == current_manifest_hash
                        )
                    ):
                        manifest_actions.extend(
                            build_responsive_layout(
                                [
                                    {
                                        "tag": "button",
                                        "text": {
                                            "tag": "plain_text",
                                            "content": f"为 {emp.name} 原地授权并发布",
                                        },
                                        "type": "primary",
                                        "value": {
                                            "action": (
                                                "slock_reauthorize_employee_app"
                                            ),
                                            "agent_id": emp.agent_id,
                                        },
                                    }
                                ]
                            )
                        )
                if len(pending) == 1:
                    pending_ref = str(
                        getattr(pending[0], "app_id", "")
                        or "NO_APP_FOUND"
                    )
                    details.append(
                        "核验开放平台处置结果后执行："
                        f"`/fire {emp.agent_id} --confirm-app-disposed "
                        f"{pending_ref}`"
                    )
                elif len(pending) > 1:
                    details.append("⚠️ 退役记录不一致，请查看安全审计日志。")
                line = f"{line}\n  " + "　".join(details)
            lines.append(line)

        card = {
            "config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": f"员工花名册（{len(employees)}人）"}, "template": "blue"},
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(lines)}},
                *manifest_actions,
            ],
        }
        self.reply_card(message_id, card)

    def fire_employee(self, message_id: str, chat_id: str, args: str) -> None:
        """Retire a visible employee through the production Journal-backed service."""

        from ...autonomous.provisioning.fire_service import (
            EmployeeFireRequest,
            FireServiceError,
        )
        from ...autonomous.provisioning.fire_state import (
            FireCleanupMode,
            FireEffectState,
            FirePhase,
        )
        from ...thread.manager import (
            get_current_is_p2p,
            get_current_sender_id,
            get_current_tenant_key,
        )

        requester_id = get_current_sender_id() or ""
        if (
            not get_current_is_p2p()
            or not requester_id
            or requester_id not in getattr(self.ctx.settings, "admin_user_ids", frozenset())
        ):
            self.reply_text(message_id, "⛔ `/fire` 仅允许配置管理员在主 Bot 私聊中使用。")
            return
        tokens = args.split()
        drain = "--drain" in tokens
        confirmation_ref = ""
        confirmation_flag = "--confirm-app-disposed"
        if confirmation_flag in tokens:
            index = tokens.index(confirmation_flag)
            if index + 1 >= len(tokens):
                self.reply_text(
                    message_id,
                    "用法: `/fire <员工名称或ID> --confirm-app-disposed <AppID|NO_APP_FOUND>`",
                )
                return
            confirmation_ref = tokens[index + 1]
            tokens = tokens[:index] + tokens[index + 2 :]
        names = [token for token in tokens if token != "--drain"]
        if len(names) != 1:
            self.reply_text(message_id, "用法: `/fire <员工名称或ID> [--drain]`")
            return
        request = EmployeeFireRequest(
            employee=names[0],
            tenant_key=get_current_tenant_key() or "",
            message_id=message_id,
            chat_id=chat_id,
            requester_principal_id=requester_id,
            drain=drain,
        )
        service = getattr(self.ctx, "employee_fire_service", None)
        if service is None:
            self.reply_text(message_id, "员工退役服务未就绪；未执行任何删除操作。")
            return
        try:
            state = (
                service.confirm_external_disposition(request, confirmation_ref)
                if confirmation_ref
                else service.start_fire(request)
            )
        except FireServiceError as exc:
            reason = str(exc)
            logger.warning("visible employee fire rejected: %s", reason)
            if reason == "employee already archived":
                self.reply_text(
                    message_id,
                    f"员工 `{names[0]}` 已经归档，无需重复 `/fire`；现在可以重新 `/hire {names[0]}`。",
                )
            elif reason == "retirable employee was not uniquely resolved":
                self.reply_text(
                    message_id,
                    "未找到唯一的在职员工。请执行 `/roster` 核对名称或改用员工 ID。",
                )
            elif reason == "external disposition reference mismatch":
                self.reply_text(
                    message_id,
                    "外部应用处置确认与待处理记录不匹配。请使用 `/roster` 核对员工，"
                    "并提交记录中的精确 App ID；若开放平台确认从未创建应用，使用 `NO_APP_FOUND`。",
                )
            else:
                self.reply_text(
                    message_id,
                    "员工退役被安全门禁拒绝；未执行或误报任何清理操作。",
                )
            return
        except Exception as exc:
            logger.error("visible employee fire failed: %s", type(exc).__name__)
            self.reply_text(message_id, "员工退役未能安全推进；请查看审计日志后处理。")
            return
        if state.phase is FirePhase.ARCHIVED:
            if getattr(state, "external_disposition_confirmed", False):
                self.reply_text(
                    message_id,
                    "✅ 已记录管理员的开放平台处置确认，员工现已完成归档，名称和容量已释放。",
                )
            elif (
                getattr(state, "cleanup_mode", None)
                is FireCleanupMode.SAFE_ABORT
            ):
                if getattr(state, "app_id", ""):
                    self.reply_text(
                        message_id,
                        "✅ 员工创建已在应用注册执行前安全中止并完成归档，名称和容量已释放。"
                        f"GhostAP 未修改或删除已有应用 `{state.app_id}`；是否继续保留或由管理员"
                        "手动停用，请按实际用途决定。",
                    )
                else:
                    self.reply_text(
                        message_id,
                        "✅ 员工创建已在应用注册执行前安全中止并完成归档，名称和容量已释放。"
                        "未创建或绑定飞书应用，也没有员工凭证、Slash 命令或 Channel 需要清理。",
                    )
            elif (
                getattr(state, "cleanup_mode", None)
                is FireCleanupMode.EXTERNAL_UNKNOWN
            ):
                self.reply_text(
                    message_id,
                    "⚠️ 退役 Journal 出现缺少外部处置确认的异常归档；请查看安全审计日志，"
                    "GhostAP 未声称应用已删除。",
                )
            else:
                self.reply_text(
                    message_id,
                    "✅ 员工已停止托管、清理命令和本地凭证并完成归档。"
                    "开放平台应用仍需管理员手动停用或删除，GhostAP 未声称已删除该应用。",
                )
        else:
            effects = dict(getattr(state, "effects", ()) or ())
            if (
                state.error_code
                == "external_cleanup_authority_unavailable"
            ):
                self.reply_text(
                    message_id,
                    "⚠️ 员工已关闭新任务入口，但应用注册已开始且本地没有可验证的完整凭据，"
                    "因此 GhostAP 拒绝误报退役成功。请在飞书开放平台核对并停用或删除对应应用，"
                    f"然后执行 `/fire {names[0]} --confirm-app-disposed "
                    f"{state.app_id or 'NO_APP_FOUND'}`。完成确认前该员工名称和应用仍保持占用。",
                )
            elif effects.get("slash_cleanup") is FireEffectState.ACTION_REQUIRED:
                employee_name = str(getattr(state, "employee_name", "") or names[0])
                self.reply_text(
                    message_id,
                    "⚠️ 员工已关闭新任务入口，但 Slash 命令清理结果未确认。"
                    f"请再次执行 `/fire {employee_name}` 重新核验；系统不会盲目重复删除。"
                    "若仍未确认，请先在开放平台完成命令清理后再次核验；"
                    "完成归档前不要重新 `/hire` 同名员工。",
                )
            else:
                self.reply_text(
                    message_id,
                    f"⚠️ 员工已关闭新任务入口，但退役需要人工处理：{state.error_code or 'unknown'}。"
                    "凭证与归档不会被误报为已完成。",
                )

    def _visible_hire_readiness_blockers(self) -> tuple[str, ...]:
        provider = getattr(self.ctx, "employee_hire_readiness", None)
        if not callable(provider):
            return ("readiness_unavailable",)
        try:
            readiness = provider()
        except Exception:
            logger.warning("visible employee readiness probe failed", exc_info=True)
            return ("readiness_probe",)
        blockers = getattr(readiness, "blockers", None)
        if not isinstance(blockers, (tuple, list)):
            return ("readiness_invalid",)
        normalized = tuple(
            blocker
            for blocker in blockers
            if isinstance(blocker, str) and blocker.strip()
        )
        if len(normalized) != len(blockers):
            return ("readiness_invalid",)
        if getattr(readiness, "ready", None) is not True:
            return normalized or ("readiness_not_ready",)
        return normalized

    def _visible_hire_requester(self, message_id: str) -> str | None:
        from ...thread.manager import get_current_is_p2p, get_current_sender_id

        requester_id = get_current_sender_id() or ""
        admin_ids = getattr(self.ctx.settings, "admin_user_ids", frozenset())
        if not get_current_is_p2p() or not requester_id or requester_id not in admin_ids:
            self.reply_text(message_id, "⛔ `/hire` 仅允许配置管理员在主 Bot 私聊中使用。")
            return None
        return requester_id

    def handle_new_role_tool_dropdown_change(
        self,
        message_id: str,
        chat_id: str,
        value: dict,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Repaint the unified /hire card when the user changes the tool dropdown."""

        role_name = str(value.get("role_name") or "").strip()
        tool_name = str(value.get("_option") or "").strip().lower()
        if not role_name or not tool_name:
            self.reply_text(message_id, "请选择有效的工具")
            return
        if tool_name not in self.TOOL_TYPE_ROLE_MAP:
            self.reply_text(message_id, f"请选择有效的工具: `{tool_name}`")
            return

        is_global = bool(value.get("global_hire"))
        try:
            existing_app_id = (
                _normalize_existing_hire_app_id(value.get("existing_app_id"))
                if is_global
                else ""
            )
        except ValueError:
            self.reply_text(message_id, "飞书 App ID 无效；请重新执行 `/hire`。")
            return
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id) if not is_global else None

        cwd = (
            getattr(project, "root_path", None)
            or (getattr(engine, "root_path", None) if engine else None)
            or self.get_working_dir(chat_id)
        )
        models = fetch_acp_models(tool_name, cwd=cwd, current_model=None)
        tool_options = self._selectable_role_tool_options()
        value_extra: dict = {"global_hire": is_global}
        if existing_app_id:
            value_extra["existing_app_id"] = existing_app_id

        _, card_content = CardBuilder.build_slock_role_unified_select_card(
            role_name=role_name,
            tools=tool_options,
            models=models,
            tool_name=tool_name,
            project_id=(project.project_id if project else value.get("project_id")),
            profile_markdown=self._visible_employee_profile_markdown(
                role_name,
                tool_name,
                "",
            ) if is_global else None,
            value_extra=value_extra,
        )
        self.update_card(message_id, card_content)

    def handle_new_role_select_tool(
        self,
        message_id: str,
        chat_id: str,
        value: dict,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Handle `/hire` or `/new-role` tool selection and show the ACP model picker.

        Uses cascade card (with effort/thinking dropdown) when the tool has
        enough model variants (same UX as /traex model selection).
        Falls back to paginated buttons for simpler tools.
        """

        role_name = str(value.get("role_name") or "").strip()
        tool_name = str(value.get("tool_name") or "").strip().lower()
        if not role_name or not tool_name:
            self.reply_text(message_id, "请选择有效的工具")
            return
        if tool_name not in self.TOOL_TYPE_ROLE_MAP:
            self.reply_text(message_id, f"请选择有效的工具: `{tool_name}`")
            return

        is_global = bool(value.get("global_hire"))
        try:
            existing_app_id = (
                _normalize_existing_hire_app_id(value.get("existing_app_id"))
                if is_global
                else ""
            )
        except ValueError:
            self.reply_text(message_id, "飞书 App ID 无效；请重新执行 `/hire`。")
            return
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id) if not is_global else None
        if not is_global and not engine:
            self.reply_text(message_id, "请先激活 Slock 模式: `/slock`")
            return

        cwd = getattr(project, "root_path", None) or (getattr(engine, "root_path", None) if engine else None) or self.get_working_dir(chat_id)
        try:
            model_page = int(value.get("model_page", 0) or 0)
        except (TypeError, ValueError):
            model_page = 0
        models = fetch_acp_models(tool_name, cwd=cwd, current_model=None)

        # Try cascade card (with effort/thinking dropdown) for tools with many variants
        value_extra = {"role_name": role_name, "global_hire": is_global}
        if existing_app_id:
            value_extra["existing_app_id"] = existing_app_id

        # has_cascade_variants expects dicts; models are ACPModelOption objects
        # Use build_acp_model_cascade_card which handles both formats internally
        has_variants = any(
            getattr(m, "selection_variants", None) or getattr(m, "reasoning_efforts", None)
            for m in (models or [])
        )
        app_context = (
            f" · 复用应用 `{existing_app_id}`" if existing_app_id else ""
        )

        if has_variants:
            _, card_content = CardBuilder.build_acp_model_cascade_card(
                models,
                tool_name,
                project_id=(project.project_id if project else value.get("project_id")),
                context_markdown=(
                    f"员工: **{role_name}** · 选择模型和思考深度{app_context}"
                ),
                group_action=action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_GROUP,
                profile_action=action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_PROFILE,
                effort_action=action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_EFFORT,
                select_action=action_ids.SLOCK_NEW_ROLE_SELECT_MODEL,
                refresh_action=action_ids.SLOCK_NEW_ROLE_SELECT_TOOL,
                value_extra=value_extra,
            )
        else:
            _, card_content = CardBuilder.build_acp_model_select_card(
                models,
                tool_name,
                project_id=(project.project_id if project else value.get("project_id")),
                action_name=action_ids.SLOCK_NEW_ROLE_SELECT_MODEL,
                value_extra=value_extra,
                context_markdown=f"员工: **{role_name}**{app_context}",
                refresh_action_name=action_ids.SLOCK_NEW_ROLE_SELECT_TOOL,
                model_page=model_page,
            )
        # When the click carries a page change, update the existing card in
        # place; the initial tool selection (page 0) replies a fresh card.
        if model_page and self.update_card(message_id, card_content):
            return
        self.reply_card(message_id, card_content)

    def handle_new_role_model_cascade_select(
        self,
        message_id: str,
        chat_id: str,
        value: dict,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Repaint the employee model cascade without finalizing the role."""

        role_name = str(value.get("role_name") or "").strip()
        tool_name = str(value.get("tool_name") or "").strip().lower()
        if not role_name or tool_name not in self.TOOL_TYPE_ROLE_MAP:
            self.reply_text(message_id, "请选择有效的员工工具和模型")
            return

        action = str(value.get("action") or "")
        selected = str(value.get("_option") or "").strip()
        pending_group = value.get("model_group")
        pending_profile = value.get("model_profile")
        pending_effort = value.get("model_effort")
        if action == action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_GROUP:
            pending_group = selected or pending_group
            pending_profile = None
            pending_effort = None
        elif action == action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_PROFILE:
            pending_profile = selected or pending_profile
            pending_effort = None
        elif action == action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_EFFORT:
            pending_effort = selected or pending_effort
        else:
            self.reply_text(message_id, "请选择有效的模型选项")
            return

        is_global = bool(value.get("global_hire"))
        try:
            existing_app_id = (
                _normalize_existing_hire_app_id(value.get("existing_app_id"))
                if is_global
                else ""
            )
        except ValueError:
            self.reply_text(message_id, "飞书 App ID 无效；请重新执行 `/hire`。")
            return
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id) if not is_global else None
        if not is_global and not engine:
            self.reply_text(message_id, "请先激活 Slock 模式: `/slock`")
            return

        cwd = (
            getattr(project, "root_path", None)
            or (getattr(engine, "root_path", None) if engine else None)
            or self.get_working_dir(chat_id)
        )
        models = fetch_acp_models(tool_name, cwd=cwd, current_model=None)
        value_extra = {"role_name": role_name, "global_hire": is_global}
        if existing_app_id:
            value_extra["existing_app_id"] = existing_app_id
        tool_options = self._selectable_role_tool_options()
        _, card_content = CardBuilder.build_slock_role_unified_select_card(
            role_name=role_name,
            tools=tool_options,
            models=models,
            tool_name=tool_name,
            project_id=(project.project_id if project else value.get("project_id")),
            pending_group=pending_group,
            pending_profile=pending_profile,
            pending_effort=pending_effort,
            profile_markdown=self._visible_employee_profile_markdown(
                role_name,
                tool_name,
                str(pending_group or ""),
            ) if is_global else None,
            value_extra=value_extra,
        )
        self.update_card(message_id, card_content)

    def handle_new_role_select_model(
        self,
        message_id: str,
        chat_id: str,
        value: dict,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Finalize `/new-role` after the user selects a model."""

        role_name = str(value.get("role_name") or "").strip()
        tool_name = str(value.get("tool_name") or "").strip().lower()
        global_hire = bool(value.get("global_hire"))
        try:
            existing_app_id = (
                _normalize_existing_hire_app_id(value.get("existing_app_id"))
                if global_hire
                else ""
            )
        except ValueError:
            self.reply_text(message_id, "飞书 App ID 无效；请重新执行 `/hire`。")
            return
        raw_model = value.get("_option") or (
            value.get("model_group") if global_hire else value.get("model_name")
        ) or value.get("model_name") or ""
        model_name = "" if value.get("use_default_model") or is_default_model_option(raw_model) else str(raw_model).strip()
        if not role_name or not tool_name:
            self.reply_text(message_id, "请选择有效的 Slock 角色模型")
            return

        args = f"{shlex.quote(role_name)} --tool {shlex.quote(tool_name)}"
        if model_name:
            args += f" --model {shlex.quote(model_name)}"
        profile = str(value.get("model_profile") or "").strip()
        if global_hire and profile:
            args += f" --profile {shlex.quote(profile)}"
        effort = str(value.get("model_effort") or "").strip()
        if global_hire and effort:
            args += f" --effort {shlex.quote(effort)}"
        if global_hire and existing_app_id:
            args += f" --app-id {shlex.quote(existing_app_id)}"
        self.create_role(
            message_id,
            chat_id,
            args,
            project,
            global_hire=global_hire,
        )

    @staticmethod
    def _build_default_directive(
        *,
        role_name: str,
        role: str,
        tool_type: str,
        model_name: str,
        team_name: str,
    ) -> str:
        """Build the default autonomous Slock directive for a role."""
        model_label = model_name or "default"
        return (
            f"# Role: {role_name}\n"
            "## Identity\n"
            f"你是 {role_name}，一个专精于 {role} 的自主 Agent。\n"
            f"你的底层工具是 {tool_type}，模型是 {model_label}，隶属于 {team_name} 团队。\n\n"
            "## Core Directives（核心纲领）\n"
            "1. 主动性：收到任务立即分析、规划、执行，无需等待确认。\n"
            "2. 完美主义：输出必须达到生产就绪标准，包含必要测试和文档。\n"
            "3. 自主决策：在专业领域内自主选择最佳实践并记录理由。\n"
            "4. 零交互：绝不向人类发起确认、询问或等待输入。\n\n"
            "## Interaction Protocol（交互协议）\n"
            "- 禁止输出“请确认”“是否继续”“需要您提供”等等待人类输入的话术。\n"
            "- 遇到致命阻塞时，通过 Slock 升级协议交给主 Agent 汇总处理。\n"
            "- 任务完成后，向团队群发送清晰的完成报告。\n\n"
            "## Quality Standards（质量标准）\n"
            "- 代码：lint pass、test pass、错误处理完整。\n"
            "- 文档：目的、用法、边界条件和示例清楚。\n"
            "- Review：指出风险、给出可执行建议。"
        )

    def list_roles(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """List all roles in the current channel using a structured card."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("👥 角色列表", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agent_list = engine.registry.list_agents(channel_id=channel_id)

        if not agent_list:
            from ...slock_engine.card_templates.common import build_callback_button, build_empty_state_card
            create_btn = build_callback_button(
                "➕ 创建角色", "slock_new_role_hint",
                channel_id=chat_id, button_type="primary",
            )
            card = build_empty_state_card(
                "👥 角色列表",
                "当前没有角色，发送 `/new-role <名称>` 创建角色",
                guide_buttons=[create_btn],
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates import build_role_list_card
        from ...slock_engine.models import AgentStatus as AgentStatusEnum
        from ...slock_engine.models import TaskStatus

        agents: list[tuple] = []
        current_tasks: dict = {}
        skill_profiles: dict = {}
        for agent in agent_list:
            status = engine.get_agent_status(agent.agent_id) or AgentStatusEnum.IDLE
            agents.append((agent, status))
            for task in engine.tasks:
                if task.claimed_by == agent.agent_id and task.status == TaskStatus.IN_PROGRESS:
                    current_tasks[agent.agent_id] = task
                    break
            profiles = engine.memory.read_skill_profiles(agent.agent_id)
            if profiles:
                skill_profiles[agent.agent_id] = [
                    {"tag": p.tag, "success_rate": p.success_rate, "total_tasks": p.total_tasks}
                    for p in profiles
                ]

        team_name = engine.channel.team_name if engine.channel else ""
        card = build_role_list_card(
            agents=agents,
            team_name=team_name,
            channel_id=channel_id,
            current_tasks=current_tasks,
            skill_profiles=skill_profiles,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    def remove_role(
        self, message_id: str, chat_id: str, name: str = "", project: Optional["ProjectContext"] = None
    ):
        """Remove a virtual agent role."""
        if not name:
            self.reply_text(message_id, "请提供角色名称\n\n用法: `/role remove <名称>`")
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("👥 角色管理", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        if not self._check_slock_permission(engine, message_id, chat_id):
            return

        service = getattr(self.ctx, "employee_membership_service", None)
        if service is not None:
            from ...thread.manager import get_current_tenant_key

            tenant_key = get_current_tenant_key() or ""
            if not tenant_key:
                self.reply_text(message_id, "⚠️ 无法确认租户身份，员工群成员关系未变更。")
                return
            try:
                employee = service.find_employee_by_name(tenant_key, name)
            except Exception:
                self.reply_text(message_id, "⚠️ 无法唯一确认该员工，群成员关系未变更。")
                return
            if employee is not None:
                self._change_employee_membership(
                    message_id,
                    chat_id,
                    employee,
                    operation="remove",
                )
                return

        agent = engine.registry.find_by_name(name, channel_id=chat_id)
        if not agent:
            self.reply_text(message_id, f"未找到角色: **{redact_sensitive(name)}**")
            return

        engine.registry.remove(agent.agent_id)
        self.reply_text(message_id, f"✅ 角色 **{agent.emoji} {agent.name}** 已移除")

    def add_role_to_group(
        self,
        message_id: str,
        chat_id: str,
        agent_name: str = "",
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Add an existing employee to the current group.

        If agent_name is empty, show a dropdown of available agents.
        """
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.reply_text(message_id, "请先激活 Slock 模式: `/slock`")
            return

        if not self._check_slock_permission(engine, message_id, chat_id):
            return

        service = getattr(self.ctx, "employee_membership_service", None)
        tenant_key = ""
        if service is not None:
            from ...thread.manager import get_current_tenant_key

            tenant_key = get_current_tenant_key() or ""
            if not tenant_key:
                self.reply_text(message_id, "⚠️ 无法确认租户身份，员工群成员关系未变更。")
                return

        if not agent_name:
            # Show dropdown of all agents not already in this group
            registry = self._get_global_registry() if service is None else None
            all_agents = service.list_employees(tenant_key) if service is not None else registry.list_agents()
            available = [a for a in all_agents if chat_id not in getattr(a, "member_groups", ())]

            if not available:
                self.reply_text(message_id, "没有可加入的员工。使用 `/hire <名字>` 先雇佣。")
                return

            options = [
                {"text": {"tag": "plain_text", "content": f"{a.emoji} {a.name} ({getattr(a, 'tool', getattr(a, 'agent_type', 'agent'))})"}, "value": a.agent_id}
                for a in available[:20]
            ]
            card = {
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "选择要加入的员工"}, "template": "blue"},
                "elements": [
                    {
                        "tag": "select_static",
                        "placeholder": {"tag": "plain_text", "content": "选择员工"},
                        "options": options,
                        "value": {"action": "slock_role_add_pick", "chat_id": chat_id},
                    },
                ],
            }
            self.reply_card(message_id, card)
            return


        if service is not None:
            try:
                employee = service.find_employee_by_name(tenant_key, agent_name)
            except Exception:
                self.reply_text(message_id, "⚠️ 无法唯一确认该员工，群成员关系未变更。")
                return
            if employee is not None:
                self._change_employee_membership(
                    message_id,
                    chat_id,
                    employee,
                    operation="add",
                )
                return

        # Find agent by name
        registry = self._get_global_registry()
        agent = registry.find_by_name(agent_name)
        if not agent:
            self.reply_text(message_id, f"未找到员工: `{agent_name}`\n使用 `/hire {agent_name}` 先雇佣")
            return

        # Add to current group
        if chat_id not in agent.member_groups:
            agent.member_groups.append(chat_id)
        engine.registry.register(agent)
        self.reply_text(message_id, f"✅ **{agent.emoji} {agent.name}** 已加入当前群")

    def _change_employee_membership(
        self,
        message_id: str,
        chat_id: str,
        employee: object,
        *,
        operation: str,
    ) -> None:
        """Mutate a canonical employee membership and report only proven state."""

        from ...autonomous.membership import (
            MembershipAuthorizationError,
            MembershipBindingError,
            MembershipMutationRequest,
            MembershipOperation,
            MembershipState,
        )
        from ...thread.manager import get_current_sender_id, get_current_tenant_key

        service = getattr(self.ctx, "employee_membership_service", None)
        tenant_key = get_current_tenant_key() or ""
        requester = get_current_sender_id() or ""
        if service is None or not tenant_key or not requester:
            self.reply_text(message_id, "⚠️ 无法确认操作身份，员工群成员关系未变更。")
            return
        try:
            outcome = service.mutate(
                MembershipMutationRequest(
                    tenant_key=tenant_key,
                    chat_id=chat_id,
                    agent_id=str(getattr(employee, "agent_id", "")),
                    requester_principal_id=requester,
                    operation=MembershipOperation(operation),
                )
            )
        except MembershipAuthorizationError:
            self.reply_text(message_id, "⚠️ 权限不足，仅管理员或团队创建者可变更员工群成员关系。")
            return
        except MembershipBindingError:
            self.reply_text(message_id, "⚠️ 员工身份绑定不可用，群成员关系未变更。")
            return
        except Exception:
            logger.exception("canonical employee membership mutation failed")
            self.reply_text(message_id, "⚠️ 员工群成员关系变更失败，未记录为成功。")
            return
        emoji = str(getattr(employee, "emoji", "🤖"))
        name = redact_sensitive(str(getattr(employee, "name", "员工")))
        desired = MembershipState.ACTIVE if operation == "add" else MembershipState.ABSENT
        if outcome.confirmed and outcome.state is desired:
            action = "加入" if operation == "add" else "移出"
            self.reply_text(message_id, f"✅ **{emoji} {name}** 已{action}当前群")
            return
        self.reply_text(
            message_id,
            f"⚠️ **{emoji} {name}** 的远端群成员状态无法确认；已进入降级状态并暂停任务路由。",
        )

    def move_role(
        self,
        message_id: str,
        chat_id: str,
        name: str = "",
        target_team_name: str = "",
        project: Optional["ProjectContext"] = None,
    ):
        """Move an agent from the current team to a target team.

        Flow: validate → permission(source+target) → try_lock_for_move (atomic
        IDLE→MOVING) → registry.move_agent → unlock → refresh target cache →
        notify target (best-effort) → append L1 context → reply confirm card.

        Design: 'move first, notify second' — registry persistence is the
        authoritative operation; notification card is best-effort UI.  If the
        notification fails, the move still stands and source group receives a
        degraded warning instead of a rollback.
        """
        import time as _time

        from ...slock_engine.card_templates import (
            build_agent_move_confirm_card,
            build_agent_move_departure_card,
            build_agent_move_notification_card,
        )
        from ...thread.manager import get_current_sender_id

        if not name or not target_team_name:
            self.reply_text(
                message_id,
                "请提供角色名称和目标团队\n\n用法: `/role move <角色名> <目标团队名>`",
            )
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("👥 角色管理", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        # Permission check: source group
        if not self._check_slock_permission(engine, message_id, chat_id):
            return

        agent = engine.registry.find_by_name(name, channel_id=chat_id)
        if not agent:
            self.reply_text(message_id, f"未找到角色: **{redact_sensitive(name)}**")
            return

        # Find target team
        target_engine = manager.find_team(target_team_name)
        if not target_engine or not target_engine.channel:
            self.reply_text(message_id, f"未找到目标团队: **{redact_sensitive(target_team_name)}**")
            return

        target_channel_id = target_engine.channel.channel_id
        if target_channel_id == chat_id:
            self.reply_text(message_id, "⚠️ 目标团队与当前团队相同，无需移动")
            return

        # Permission check: target group (operator must be global admin or target owner)
        if not self._check_slock_permission(target_engine, message_id, chat_id):
            return

        # Atomic: check IDLE and lock agent in MOVING state
        if not engine.try_lock_for_move(agent.agent_id):
            status = engine.get_agent_status(agent.agent_id)
            self.reply_text(
                message_id,
                f"⚠️ 角色 **{agent.name}** 当前状态为 {status.value}，仅 IDLE 状态可移动",
            )
            return

        source_team_display = (engine.channel.team_name if engine.channel else chat_id) or chat_id
        target_team_display = target_engine.channel.team_name or target_team_name
        operator_id = get_current_sender_id() or ""

        try:
            # Step 1: Perform atomic move in registry (authoritative operation)
            outcome = engine.registry.move_agent(agent.agent_id, chat_id, target_channel_id)
            if not outcome.success:
                from ...slock_engine.agent_registry import MoveResult

                if outcome.status == MoveResult.NOT_FOUND:
                    err_msg = f"❌ 移动失败：角色 **{redact_sensitive(name)}** 未找到"
                elif outcome.status == MoveResult.NOT_IN_SOURCE:
                    err_msg = "❌ 移动失败，请确认角色属于当前团队"
                elif outcome.status == MoveResult.DUPLICATE_NAME:
                    err_msg = "❌ 移动失败：目标团队已存在同名角色，请先修改角色名称"
                elif outcome.status == MoveResult.PERSIST_FAILED:
                    err_msg = "❌ 迁移失败：数据持久化异常，角色仍留在原团队，请稍后重试"
                else:
                    err_msg = "❌ 移动失败，请确认角色属于当前团队"
                logger.warning(
                    "slock move_role: registry move_agent failed | agent=%s source=%s target=%s operator=%s status=%s",
                    agent.agent_id, chat_id, target_channel_id, operator_id, outcome.status.value,
                )
                self.reply_text(message_id, err_msg)
                return

            # Step 1.5: Redact active_context to prevent source-group history leakage
            import hashlib as _hashlib

            try:
                _pre_memory = engine.memory.read_agent_memory(agent.agent_id)
                _ctx_len = len(_pre_memory.active_context) if _pre_memory.active_context else 0
                _ctx_md5 = _hashlib.md5(_pre_memory.active_context.encode()).hexdigest() if _pre_memory.active_context else ""
                logger.info(
                    "slock move_role: pre-redact audit | agent=%s context_chars=%d md5=%s",
                    agent.agent_id, _ctx_len, _ctx_md5,
                )
                engine.memory.redact_active_context_for_move(agent.agent_id, chat_id, target_channel_id)
            except Exception as exc:
                logger.warning(
                    "slock move_role: redact_active_context failed (non-fatal) | agent=%s error=%s",
                    agent.agent_id, exc,
                )
        finally:
            engine.unlock_after_move(agent.agent_id)

        # Refresh target engine's registry cache so it can discover the agent
        target_engine.registry.refresh_agent(agent.agent_id)

        # Verify L1 memory integrity after move (user-facing degradation notice)
        _l1_memory_degraded = False
        try:
            _moved_memory = target_engine.memory.read_agent_memory(agent.agent_id)
            if agent.system_prompt and not _moved_memory.role and not _moved_memory.key_knowledge:
                logger.error(
                    "slock move_role: L1 memory empty after move — persona may be inconsistent | agent=%s",
                    agent.agent_id,
                )
                _l1_memory_degraded = True
        except Exception as exc:
            logger.error(
                "slock move_role: L1 memory read failed after move | agent=%s error=%s",
                agent.agent_id, exc,
            )
            _l1_memory_degraded = True

        # Resolve operator display name for notification card
        operator_display = (
            resolve_display_name_nonblocking(operator_id, self.ctx.api_client_factory)
            if operator_id
            else ""
        )

        # Step 2: Send notification card to target group (best-effort)
        notification_card = build_agent_move_notification_card(
            agent=agent,
            source_team=source_team_display,
            target_team=target_team_display,
            operator_display=operator_display,
        )
        sent_msg_id = self.send_card_to_chat(
            target_channel_id, json.dumps(notification_card, ensure_ascii=False)
        )
        if not sent_msg_id:
            logger.warning(
                "slock move_role: notification card send failed (non-fatal) | agent=%s source=%s target=%s operator=%s",
                agent.agent_id, chat_id, target_channel_id, operator_id,
            )
            self.reply_text(
                message_id,
                "✅ 移动成功，但目标群通知发送失败。目标群可通过 /role list 查看新成员。",
            )
            # Do NOT return — continue to append context and send confirm card

        # Step 2.5: Send departure notification card to source group (best-effort)
        try:
            departure_card = build_agent_move_departure_card(
                agent=agent,
                target_team=target_team_display,
            )
            self.send_card_to_chat(chat_id, json.dumps(departure_card, ensure_ascii=False))
        except Exception as exc:
            logger.warning(
                "slock move_role: departure card send failed (non-fatal) | agent=%s source=%s error=%s",
                agent.agent_id, chat_id, exc,
            )

        # Step 3: Append migration record to L1 active context
        context_record = (
            f"[{_time.strftime('%Y-%m-%d %H:%M')}] "
            f"Moved from {source_team_display} to {target_team_display}"
        )
        try:
            engine.memory.update_agent_context(agent.agent_id, context_record)
        except Exception as exc:
            logger.warning(
                "slock move_role: update_agent_context failed (non-fatal) | agent=%s error=%s",
                agent.agent_id, exc,
            )

        # Step 4: Reply confirm card to source group with jump button
        confirm_card = build_agent_move_confirm_card(
            agent=agent,
            source_team=source_team_display,
            target_team=target_team_display,
            target_channel_id=target_channel_id,
        )
        self.reply_card(message_id, json.dumps(confirm_card, ensure_ascii=False))

        if _l1_memory_degraded:
            self.reply_text(
                message_id,
                "⚠️ 注意：角色记忆加载异常，人格一致性可能受影响。请在目标群执行 /role info 确认。",
            )

        logger.info(
            "slock move_role: success | agent=%s source=%s target=%s operator=%s",
            agent.agent_id, chat_id, target_channel_id, operator_id,
        )

    def show_role_info(
        self, message_id: str, chat_id: str, name: str = "", project: Optional["ProjectContext"] = None
    ):
        """Show detailed info about a role using structured card.

        Permission-aware: admin/owner sees full details including Active Context
        and permissions; regular members see only non-sensitive identity info.
        """
        if not name:
            from ...slock_engine.card_templates.common import build_usage_hint_card
            card = build_usage_hint_card(
                "/role info <名称>",
                ["/role info coder", "/role info reviewer", "/r info tester"],
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("👤 角色详情", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        agent = engine.registry.find_by_name(name, channel_id=chat_id)
        if not agent:
            self.reply_text(message_id, f"未找到角色: **{redact_sensitive(name)}**")
            return

        from ...slock_engine.card_templates import build_role_info_card
        from ...slock_engine.models import AgentStatus as AgentStatusEnum
        from ...slock_engine.models import TaskStatus

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        status = engine.get_agent_status(agent.agent_id) or AgentStatusEnum.IDLE
        memory = engine.memory.read_agent_memory(agent.agent_id)
        profiles = engine.memory.read_skill_profiles(agent.agent_id)
        skill_profiles = [
            {"tag": p.tag, "success_rate": p.success_rate, "total_tasks": p.total_tasks}
            for p in profiles
        ]

        # Current and recent tasks
        current_task = None
        recent_tasks: list = []
        for task in engine.tasks:
            if task.claimed_by == agent.agent_id:
                if task.status == TaskStatus.IN_PROGRESS:
                    current_task = task
                elif task.status == TaskStatus.DONE:
                    recent_tasks.append(task)
        recent_tasks = recent_tasks[-3:]  # Last 3 completed

        card = build_role_info_card(
            agent,
            status=status,
            memory=memory,
            skill_profiles=skill_profiles,
            current_task=current_task,
            recent_tasks=recent_tasks,
            channel_id=channel_id,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def list_tasks(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """List all tasks in the current slock session."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("📋 任务列表", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        tasks = engine.tasks
        if not tasks:
            from ...slock_engine.card_templates.common import build_callback_button, build_empty_state_card
            guide_btn = build_callback_button(
                "➕ 分配任务", "slock_assign_task_hint",
                channel_id=chat_id, button_type="primary",
            )
            card = build_empty_state_card(
                "📋 任务列表",
                "当前没有任务。\n\n发送 `/task assign <任务描述>` 或点击下方按钮分配任务。",
                guide_buttons=[guide_btn],
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates import build_task_board_card

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agents = engine.registry.list_agents(channel_id=channel_id)
        team_name = engine.channel.team_name if engine.channel else ""
        card = build_task_board_card(engine.tasks, agents, team_name=team_name, channel_id=channel_id)
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    def assign_task(
        self,
        message_id: str,
        chat_id: str,
        content: str = "",
        role_name: str = "",
        project: Optional["ProjectContext"] = None,
    ):
        """Assign a task to a specific role and execute it.

        If role_name is provided: claim → execute → complete/rollback.
        If no role_name: create task only (unassigned).
        """
        if not content:
            self.reply_text(message_id, "请提供任务内容\n\n用法: `/task assign <任务内容> <角色名称>`")
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.reply_text(message_id, "请先激活 Slock 模式: `/slock`")
            return

        # Rate-limit check for non-admin/non-owner users
        if not self._check_assign_rate_limit(engine, message_id, chat_id):
            return

        task = engine.add_task(content)
        if task is None:
            self.reply_text(
                message_id,
                "❌ 任务创建失败（任务队列已满或内部错误），请稍后重试",
            )
            return

        if role_name:
            channel_id = engine.channel.channel_id if engine.channel else chat_id
            agent = engine.registry.find_by_name(role_name, channel_id=channel_id)
            if not agent:
                self.reply_text(
                    message_id,
                    f"⚠️ 任务已创建但未找到角色 **{redact_sensitive(role_name)}**\n"
                    f"• 任务 ID: `{task.task_id[:8]}`\n"
                    f"• 发送 `/role list` 查看可用角色",
                )
                return

            # Claim the task
            if not engine.claim_task(task.task_id, agent.agent_id):
                self.reply_text(message_id, f"❌ 任务 claim 失败，{agent.name} 可能正在执行其他任务")
                return

            # Submit async execution — send placeholder, update on completion
            self._submit_task_execution(
                engine, task, agent, message_id, chat_id, content, project
            )
            return

        # No role specified — use skill-based automatic assignment.
        channel_id = engine.channel.channel_id if engine.channel else chat_id

        # Auto-detect multi-role collaboration: if a chain template matches,
        # create a collaboration plan instead of single-agent execution.
        chain = engine._chain_manager.find_chain_for_task(content)
        if chain and len(chain.roles) > 1:
            plan = engine._collaboration_orchestrator.create_plan(task, channel_id)
            if plan:
                from ...slock_engine.card_templates.progress import build_collaboration_plan_card
                agents = list(engine.registry.list_agents(channel_id=channel_id))
                card = build_collaboration_plan_card(plan, agents, channel_id=channel_id)
                card_msg_id = self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
                # Register overview message_id for in-place card updates
                if card_msg_id and hasattr(engine, '_progress_tracker'):
                    engine._progress_tracker.set_overview_message_id(plan.plan_id, card_msg_id)
                # Confirm delivery — reset auto-start timer from this point
                engine._collaboration_orchestrator.confirm_plan_delivery(plan.plan_id)
                return

        agents = list(engine.registry.list_agents(channel_id=channel_id))
        if not agents:
            self.reply_text(
                message_id,
                f"✅ 任务已创建（等待分配）\n"
                f"• ID: `{task.task_id[:8]}`\n"
                f"• 内容: {redact_sensitive(content[:80])}\n"
                f"• 当前没有可用角色，发送 `/new-role <名称>` 创建角色",
            )
            return

        rank_agents = getattr(engine.router, "rank_agents_for_claim", None)
        ranked_agents = rank_agents(content, agents) if callable(rank_agents) else []
        if not isinstance(ranked_agents, list):
            selected = engine.router.route_message(content, agents)
            ranked_agents = [selected] if selected else []
        if not ranked_agents:
            self.reply_text(
                message_id,
                f"✅ 任务已创建（等待分配）\n"
                f"• ID: `{task.task_id[:8]}`\n"
                f"• 内容: {redact_sensitive(content[:80])}\n"
                f"• 暂无匹配角色，发送 `/role list` 查看可用角色",
            )
            return

        claimed_agent = None
        for candidate in ranked_agents:
            if engine.claim_task(task.task_id, candidate.agent_id):
                claimed_agent = candidate
                break

        if claimed_agent is None:
            self.reply_text(message_id, "❌ 任务 claim 失败，所有匹配角色可能都在执行其他任务")
            return

        self._submit_task_execution(
            engine, task, claimed_agent, message_id, chat_id, content, project, auto_routed=True
        )

    def _submit_task_execution(
        self,
        engine,
        task,
        agent,
        message_id: str,
        chat_id: str,
        content: str,
        project: Optional["ProjectContext"],
        *,
        auto_routed: bool = False,
    ) -> None:
        """Submit task execution to the engine's thread pool asynchronously.

        Sends a placeholder card immediately, then updates it with the result
        or error when execution completes.
        """
        prefix = "自动分配给" if auto_routed else "已分配给"

        def _execute():
            callbacks = self._create_callbacks(message_id, chat_id, project, engine.engine_name, engine.root_path)
            return engine.execute_task(task.task_id, agent.agent_id, callbacks)

        def _result_card(result: str, duration: float) -> str:
            try:
                engine.memory.write_agent_reasoning_snapshot(
                    agent.agent_id,
                    task.task_id,
                    prompt_summary=content[:1000],
                    result_summary=result[:2000],
                    tool_name=agent.agent_type,
                    model_name=agent.model_name,
                )
            except Exception:
                logger.warning(
                    "Failed to persist Slock reasoning snapshot for task %s agent %s",
                    task.task_id,
                    agent.agent_id,
                    exc_info=True,
                )
            card_data = engine._mouthpiece.format_card(
                agent,
                result,
                model_info=agent.agent_type,
                duration_s=duration,
                channel_id=chat_id,
                task_id=task.task_id,
            )
            return json.dumps(card_data, ensure_ascii=False)

        def _error_card(exc: Exception) -> str:
            return json.dumps({
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "❌ 任务执行失败"}, "template": "red"},
                "body": {"elements": [{"tag": "markdown", "content": (
                    f"Agent: {agent.emoji} {agent.name}\n"
                    f"任务: {content[:60]}\n"
                    f"错误: {safe_error_message(exc)}\n\n"
                    "任务已回退为 TODO，可重新分配"
                )}]},
            }, ensure_ascii=False)

        def _empty_card() -> str:
            return json.dumps({
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "⚠️ 任务完成无输出"}, "template": "orange"},
                "body": {"elements": [{"tag": "markdown", "content": (
                    f"Agent: {agent.emoji} {agent.name}\n"
                    f"任务: {content[:60]}\n\n"
                    "任务已回退为 TODO，可重新分配"
                )}]},
            }, ensure_ascii=False)

        def _busy_card() -> str:
            return json.dumps({
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "⚠️ 团队繁忙"}, "template": "orange"},
                "body": {"elements": [{"tag": "markdown", "content": "当前所有角色均在忙碌中，请稍后重试。"}]},
            }, ensure_ascii=False)

        placeholder_card = json.dumps({
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"⏳ 任务{prefix} {agent.emoji} {agent.name}"},
                "template": "indigo",
            },
            "body": {"elements": [{"tag": "markdown", "content": (
                f"**任务**: {content[:80]}\n"
                f"**ID**: `{task.task_id[:8]}`\n\n"
                "Agent 正在执行中..."
            )}]},
        }, ensure_ascii=False)

        self._execute_async(
            engine=engine,
            execute_fn=_execute,
            placeholder_card_json=placeholder_card,
            result_card_fn=_result_card,
            error_card_fn=_error_card,
            empty_card_fn=_empty_card,
            busy_card_fn=_busy_card,
            message_id=message_id,
            chat_id=chat_id,
        )

    def show_task_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Show task board with status summary."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("📊 任务状态", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates import build_task_board_card

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agents = engine.registry.list_agents(channel_id=channel_id)
        team_name = engine.channel.team_name if engine.channel else ""
        card = build_task_board_card(engine.tasks, agents, team_name=team_name, channel_id=channel_id)
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def show_memory_group(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Show L2 shared group memory for the current slock team."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("🧠 群组记忆", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        group_memory = engine.memory.read_group_memory(channel_id)
        if not group_memory:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card(
                "🧠 群组共享记忆",
                "当前群组暂无共享记忆。\n\nAgent 协作过程中会自动积累群组知识。",
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates.memory import build_memory_group_card

        # Parse group memory text into structured memory_items
        lines = [line.strip() for line in group_memory.strip().split("\n") if line.strip()]
        memory_items: list[dict] = []
        current_category = "context"
        for line in lines[:50]:  # NFR-6: limit to 50 items per read
            if line.startswith("## ") or line.startswith("# "):
                current_category = line.lstrip("#").strip().lower() or "context"
            else:
                memory_items.append({
                    "category": current_category,
                    "content": line[:200],
                    "timestamp": None,
                })

        total_lines = len(lines)
        team_name = engine.channel.team_name if engine.channel else "群组"

        card = build_memory_group_card(
            agent_name=f"{team_name} (共享)",
            agent_emoji="🧠",
            memory_items=memory_items,
            channel_id=channel_id,
        )

        # Inject meta-info row at the top of elements (after header)
        truncated = total_lines > 50
        meta_text = f"📊 共 {total_lines} 条记录"
        if truncated:
            meta_text += " | ⚠️ 内容已截断，仅显示前 50 条"
        if card.get("body", {}).get("elements"):
            card["body"]["elements"].insert(0, {"tag": "markdown", "content": meta_text})

        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Agent Memory (per-agent L1 memory)
    # ------------------------------------------------------------------

    def _resolve_canonical_employee(self, target: str):
        """Resolve a durable workforce employee without consulting legacy roles."""

        context = getattr(self, "ctx", None)
        service = getattr(context, "employee_hire_service", None)
        if service is None or not isinstance(target, str) or not target.strip():
            return None
        from ...thread import get_current_tenant_key

        tenant_key = get_current_tenant_key() or ""
        projection = service.synchronize_projection()
        employees = getattr(projection, "employees", None)
        if not isinstance(employees, dict):
            return None
        folded = target.strip().casefold()
        matches = [
            employee
            for employee in employees.values()
            if (not tenant_key or employee.tenant_key == tenant_key)
            and (
                employee.agent_id.casefold() == folded
                or employee.name.casefold() == folded
            )
        ]
        return matches[0] if len(matches) == 1 else None

    def _employee_data_request(self, *, employee, chat_id: str):
        from ...autonomous.data.query import AuthenticatedDataRequest
        from ...thread import (
            get_current_is_p2p,
            get_current_sender_id,
            get_current_tenant_key,
            get_current_thread_id,
        )

        return AuthenticatedDataRequest(
            principal_id=get_current_sender_id() or "",
            tenant_key=get_current_tenant_key() or "",
            receiving_bot_app_id=getattr(self.ctx.settings, "app_id", ""),
            chat_id=chat_id,
            chat_type="p2p" if get_current_is_p2p() else "group",
            thread_root_id=get_current_thread_id() or "",
            requested_agent_id=employee.agent_id,
        )

    def _show_authoritative_employee_memory(self, message_id: str, chat_id: str, employee) -> None:
        data = self.ctx.employee_data_composition
        if data is None:
            self.reply_text(message_id, "员工数据服务当前不可用。")
            return
        from ...autonomous.data.ports import MemoryQuerySpec
        from ...autonomous.data.query import QueryDeniedError

        try:
            result = data.memory_query.query(
                self._employee_data_request(employee=employee, chat_id=chat_id),
                MemoryQuerySpec(agent_id=employee.agent_id, full_l1=True),
            )
        except QueryDeniedError:
            self.reply_text(message_id, "权限不足：完整员工记忆仅限主 Bot 管理员私聊读取。")
            return
        except Exception:
            logger.exception("authoritative employee memory read failed")
            self.reply_text(message_id, "员工记忆暂不可用，请稍后重试。")
            return
        from ...slock_engine.card_templates.common import build_card_wrapper

        content = result.content or "当前员工暂无 L1 记忆。"
        card = build_card_wrapper(
            header_title=f"🧠 {employee.name} · 权威记忆",
            header_template="turquoise",
            elements=[{"tag": "markdown", "content": content[:20_000]}],
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    def _show_authoritative_employee_history(self, message_id: str, chat_id: str, employee) -> None:
        data = self.ctx.employee_data_composition
        if data is None:
            self.reply_text(message_id, "员工数据服务当前不可用。")
            return
        from ...thread import get_current_is_p2p, get_current_sender_id

        admins = frozenset(getattr(self.ctx.settings, "admin_user_ids", ()) or ())
        if not get_current_is_p2p() or (get_current_sender_id() or "") not in admins:
            self.reply_text(message_id, "权限不足：员工完整历史仅限主 Bot 管理员私聊读取。")
            return
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        from ...autonomous.data.ports import HistoryQuerySpec
        from ...autonomous.data.query import QueryDeniedError

        end = datetime.now(ZoneInfo(data.service.shard_timezone)).date()
        try:
            result = data.query.query(
                self._employee_data_request(employee=employee, chat_id=chat_id),
                HistoryQuerySpec(
                    start_day=(end - timedelta(days=6)).isoformat(),
                    end_day=end.isoformat(),
                    page_size=50,
                ),
            )
        except QueryDeniedError:
            self.reply_text(message_id, "权限不足：员工完整历史仅限主 Bot 管理员私聊读取。")
            return
        except Exception:
            logger.exception("authoritative employee history read failed")
            self.reply_text(message_id, "员工历史暂不可用，请稍后重试。")
            return
        from ...slock_engine.card_templates.common import build_card_wrapper

        rows = [
            (
                f"| {record.ended_at[:19]} | {record.status} | "
                f"{record.safe_summary_text[:160]} |"
            )
            for record in result.records
        ]
        content = (
            "最近 7 天暂无执行记录。"
            if not rows
            else "| 时间 | 状态 | 摘要 |\n| --- | --- | --- |\n" + "\n".join(rows)
        )
        card = build_card_wrapper(
            header_title=f"📚 {employee.name} · 权威历史",
            header_template="blue",
            elements=[{"tag": "markdown", "content": content}],
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    def show_employee_history(
        self,
        message_id: str,
        chat_id: str,
        employee_name: str = "",
    ) -> None:
        """Read canonical execution history; discussion history stays separate."""

        if not employee_name:
            self.reply_text(message_id, "用法：`/history <员工名称>`")
            return
        employee = SlockHandler._resolve_canonical_employee(self, employee_name)
        if employee is None:
            self.reply_text(message_id, f"未找到正式员工 `{employee_name}`。")
            return
        SlockHandler._show_authoritative_employee_history(
            self,
            message_id,
            chat_id,
            employee,
        )

    def show_agent_memory(self, message_id: str, chat_id: str, agent_name: str = "", project: Optional["ProjectContext"] = None):
        """Show L1 memory for a specific agent."""
        if not agent_name:
            from ...slock_engine.card_templates.common import build_usage_hint_card
            card = build_usage_hint_card(
                "/memory <agent_name>",
                ["/memory coder", "/memory list"],
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        employee = SlockHandler._resolve_canonical_employee(self, agent_name)
        if employee is not None:
            SlockHandler._show_authoritative_employee_memory(
                self, message_id, chat_id, employee
            )
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("🧠 角色记忆", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agent = engine.registry.find_by_name(agent_name, channel_id=channel_id)
        if not agent:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("🧠 角色记忆", f"未找到角色 `{agent_name}`")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates.memory import build_memory_group_card

        memory = engine.memory.read_agent_memory(agent.agent_id)
        memory_items: list[dict] = []

        # Add key_knowledge entries
        if memory.key_knowledge:
            for line in memory.key_knowledge.strip().split("\n"):
                line = line.strip()
                if line:
                    memory_items.append({"category": "key_knowledge", "content": line[:200], "timestamp": None})

        # Add active_context entries
        if memory.active_context:
            for line in memory.active_context.strip().split("\n"):
                line = line.strip()
                if line:
                    memory_items.append({"category": "context", "content": line[:200], "timestamp": None})

        # Add role definition
        if memory.role:
            memory_items.append({"category": "role", "content": memory.role[:200], "timestamp": None})

        if not memory_items:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("🧠 角色记忆", f"{agent.display_name} 暂无记忆数据")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        card = build_memory_group_card(
            agent_name=agent.name,
            agent_emoji=agent.emoji,
            memory_items=memory_items,
            channel_id=channel_id,
            agent_id=agent.agent_id,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    def show_memory_list(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Show memory summary for all agents in the channel."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("🧠 记忆列表", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agents = engine.registry.list_agents(channel_id=channel_id)

        if not agents:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("🧠 记忆列表", "当前团队暂无角色")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates.common import build_card_wrapper

        rows: list[str] = []
        for agent in agents:
            memory = engine.memory.read_agent_memory(agent.agent_id)
            item_count = 0
            if memory.key_knowledge:
                item_count += len([line for line in memory.key_knowledge.split("\n") if line.strip()])
            if memory.active_context:
                item_count += len([line for line in memory.active_context.split("\n") if line.strip()])
            rows.append(f"| {agent.emoji} | {agent.name} | {item_count} 条 |")

        table_md = "| 头像 | 角色 | 记忆条数 |\n| --- | --- | --- |\n" + "\n".join(rows)
        elements: list[dict] = [{"tag": "markdown", "content": table_md}]

        card = build_card_wrapper(
            header_title="🧠 团队记忆概览",
            header_template="turquoise",
            elements=elements,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Discussion management (stop/history/list)
    # ------------------------------------------------------------------

    def stop_discussion(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Stop the active discussion in the current channel."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("💬 停止讨论", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id

        # Find active discussion threads
        active_thread = None
        with engine._discussions_lock:
            threads = engine._active_discussions.get(channel_id, [])
            if threads:
                active_thread = threads[0]

        if not active_thread:
            self.reply_text(message_id, "💬 当前无进行中的讨论")
            return

        # Stop the discussion via DiscussionManager

        if engine._discussion_manager is not None:
            stopped_thread = engine._discussion_manager.stop_discussion(active_thread)
            engine._remove_discussion(channel_id, stopped_thread.thread_id)

            from ...slock_engine.card_templates.discussion import build_discussion_conclusion_card
            card = build_discussion_conclusion_card(
                thread_id=stopped_thread.thread_id,
                participants=getattr(stopped_thread, "participants", []),
                conclusion=getattr(stopped_thread, "conclusion", "") or "讨论已手动停止",
                total_rounds=getattr(stopped_thread, "current_round", 0),
                total_tokens=getattr(stopped_thread, "total_tokens", 0),
                status="manually_stopped",
                channel_id=channel_id,
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
        else:
            self.reply_text(message_id, "💬 讨论管理器未初始化，无法停止讨论")

    def show_discussion_history(self, message_id: str, chat_id: str, thread_id: str = "", project: Optional["ProjectContext"] = None):
        """Show discussion history for a specific thread."""
        if not thread_id:
            from ...slock_engine.card_templates.common import build_usage_hint_card
            card = build_usage_hint_card(
                "/discussion history <thread_id>",
                ["/discussion list", "/discussion history abc123"],
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("💬 讨论历史", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id

        # Try to find in active discussions first
        thread = engine.find_active_discussion(channel_id, thread_id)

        # Try loading from persisted discussions if not active
        if not thread and engine._discussion_manager is not None:
            persisted = engine._discussion_manager.load_discussions(channel_id)
            for t in persisted:
                if getattr(t, "thread_id", "") == thread_id:
                    thread = t
                    break

        if not thread:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("💬 讨论历史", f"未找到讨论 `{thread_id[:12]}`")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates import build_discussion_history_card

        # Convert thread to history format expected by the card builder
        messages = getattr(thread, "messages", [])
        history_entry = {
            "topic_hash": getattr(thread, "thread_id", "")[:8],
            "title": getattr(thread, "topic", "") or "讨论",
            "participants": getattr(thread, "participants", []),
            "time": getattr(thread, "created_at", 0),
            "conclusion": getattr(thread, "conclusion", "") or "",
            "messages": [
                {"speaker": getattr(m, "speaker", ""), "content": getattr(m, "content", "")[:200]}
                for m in messages[-20:]
            ],
        }
        card = build_discussion_history_card(history=[history_entry], channel_id=channel_id)
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    def list_discussions(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """List all discussions (active + recent persisted) in the channel."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("💬 讨论列表", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id

        # Collect active threads
        all_threads = []
        with engine._discussions_lock:
            active = engine._active_discussions.get(channel_id, [])
            all_threads.extend(active)

        # Load persisted threads
        if engine._discussion_manager is not None:
            persisted = engine._discussion_manager.load_discussions(channel_id)
            existing_ids = {getattr(t, "thread_id", "") for t in all_threads}
            for t in persisted:
                if getattr(t, "thread_id", "") not in existing_ids:
                    all_threads.append(t)

        if not all_threads:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("💬 讨论列表", "当前暂无讨论记录")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        import time as _time

        from ...slock_engine.card_templates.common import build_card_wrapper

        rows: list[str] = []
        for thread in all_threads[:20]:
            tid = getattr(thread, "thread_id", "?")[:8]
            status = getattr(thread, "status", "unknown")
            status_label = status.value if hasattr(status, "value") else str(status)
            participants = getattr(thread, "participants", [])
            p_names = ", ".join(participants[:3]) if participants else "-"
            created = getattr(thread, "created_at", 0)
            ts = _time.strftime("%m-%d %H:%M", _time.localtime(created)) if created else "-"
            rows.append(f"| `{tid}` | {status_label} | {p_names} | {ts} |")

        table_md = "| ID | 状态 | 参与者 | 创建时间 |\n| --- | --- | --- | --- |\n" + "\n".join(rows)
        elements: list[dict] = [{"tag": "markdown", "content": table_md}]

        card = build_card_wrapper(
            header_title="💬 讨论列表",
            header_template="blue",
            elements=elements,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Plan management
    # ------------------------------------------------------------------

    def list_plans(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """List all collaboration plans in the current slock session."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("📋 协作计划", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates.progress import build_progress_overview_card

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        plans = engine.collaboration_orchestrator.list_active_plans(channel_id=channel_id)
        agents = engine.registry.list_agents(channel_id=channel_id)
        if not plans:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card(
                "📋 协作计划",
                "当前没有协作计划。\n\n任务创建后会自动触发协作规划。",
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        card = build_progress_overview_card(
            plans=plans,
            agents=agents,
            team_name=engine.channel.team_name if engine.channel else "",
            channel_id=channel_id,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    def show_plan_detail(self, message_id: str, chat_id: str, plan_id: str = "", project: Optional["ProjectContext"] = None):
        """Show detailed view of a specific collaboration plan."""
        if not plan_id:
            from ...slock_engine.card_templates.common import build_usage_hint_card
            card = build_usage_hint_card(
                "/plan <plan_id>",
                ["/plan list", "/plan abc123"],
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("📋 协作计划", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates.progress import build_task_overview_card
        from ...slock_engine.models import AgentStatus as AgentStatusEnum

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        plan = engine.collaboration_orchestrator.get_plan(plan_id)
        if not plan:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("📋 协作计划", f"未找到计划 `{plan_id[:12]}`，可能已结束或不存在")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        # Build agents with status for the task overview card
        agents: list[tuple] = []
        for agent in engine.registry.list_agents(channel_id=channel_id):
            status = engine.get_agent_status(agent.agent_id) or AgentStatusEnum.IDLE
            agents.append((agent, status))

        # Extract latest output from the last completed step
        latest_output_summary = ""
        if plan.steps:
            completed_steps = [s for s in plan.steps if s.get("status") == "DONE"]
            if completed_steps:
                last_step = completed_steps[-1]
                latest_output_summary = last_step.get("output_summary", "")[:200]

        # Gather discussion entries (last 5) from active discussions
        discussion_entries: list[dict] = []
        try:
            with engine._discussions_lock:
                active_threads = engine._active_discussions.get(channel_id, [])
            for thread in active_threads[:5]:
                for msg in getattr(thread, "messages", [])[-5:]:
                    discussion_entries.append({
                        "speaker": getattr(msg, "speaker", ""),
                        "content": getattr(msg, "content", "")[:100],
                        "timestamp": getattr(msg, "timestamp", 0),
                    })
        except Exception:
            pass
        discussion_entries = discussion_entries[-5:]

        # Gather timeline events (last 10) from plan step history
        timeline_events: list[dict] = []
        if plan.steps:
            for step in plan.steps:
                if step.get("started_at"):
                    timeline_events.append({
                        "event_type": "step_started",
                        "agent_id": step.get("agent_id", ""),
                        "timestamp": step.get("started_at", 0),
                        "detail": step.get("name", ""),
                    })
                if step.get("completed_at"):
                    timeline_events.append({
                        "event_type": "step_completed",
                        "agent_id": step.get("agent_id", ""),
                        "timestamp": step.get("completed_at", 0),
                        "detail": step.get("name", ""),
                    })
        timeline_events.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        timeline_events = timeline_events[:10]

        card = build_task_overview_card(
            plan=plan,
            agents=agents,
            channel_id=channel_id,
            latest_output_summary=latest_output_summary,
            discussion_entries=discussion_entries or None,
            timeline_events=timeline_events or None,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop_slock_engine(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Stop the slock engine, unregister managed chat, and clean up."""
        manager = self._get_engine_manager()

        # Deactivate the engine if it exists
        engine = manager.get_activated_engine(chat_id)
        if engine:
            if not self._check_slock_permission(engine, message_id, chat_id):
                return
            engine.deactivate()

        # Unregister managed chat so dispatcher stops routing to this engine
        manager.unregister_managed_chat(chat_id)

        self._safe_lifecycle_action(
            lambda: self._stop_engine_generic(message_id, chat_id, project),
            "stop", chat_id, message_id, project,
        )

    def _stop_single_agent(self, message_id: str, chat_id: str, value: dict):
        """Stop a single agent by agent_id from card action value.

        Falls back to full engine stop if agent_id is not provided (backwards compat).
        """
        agent_id = value.get("agent_id", "")

        if not agent_id:
            # Fallback: no agent_id means legacy card — stop the whole engine
            self.stop_slock_engine(message_id, chat_id)
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式。")
            return

        if not self._check_slock_permission(engine, message_id, chat_id):
            return

        stopped = engine.stop_agent(agent_id)
        if stopped:
            self.send_text_to_chat(chat_id, "⏹ Agent 已停止，状态已重置为 IDLE。")
        else:
            self.send_text_to_chat(chat_id, "⚠️ 未找到该 Agent 或其已处于空闲状态。")

    def _expand_discussion(self, chat_id: str, value: dict):
        """Show full discussion thread content in response to expand button."""
        thread_id = value.get("thread_id", "")
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式。")
            return

        thread = SlockHandler._find_discussion_thread(engine, chat_id, thread_id)
        if not thread:
            self.send_text_to_chat(chat_id, "ℹ️ 讨论线程已结束或不存在。")
            return

        # Format all messages for display
        lines: list[str] = [f"💬 **讨论详情** (thread: `{thread_id[:12]}...`)\n"]
        for msg in thread.messages:
            lines.append(f"**{msg.sender_agent_id}** (R{msg.round_num}):\n{msg.content}\n")
        self.send_text_to_chat(chat_id, "\n".join(lines)[:4000])

    def _stop_discussion(self, chat_id: str, value: dict):
        """Manually stop an active discussion thread."""
        thread_id = value.get("thread_id", "")
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式。")
            return

        dm = getattr(engine, "_discussion_manager", None)
        if dm and thread_id:
            thread_obj = SlockHandler._find_discussion_thread(engine, chat_id, thread_id)
            if thread_obj:
                dm.stop_discussion(thread_obj)
                self.send_text_to_chat(chat_id, "⏹ 讨论已手动终止。")
                return
        self.send_text_to_chat(chat_id, "ℹ️ 讨论线程已结束或不存在。")

    @staticmethod
    def _find_discussion_thread(engine, chat_id: str, thread_id: str):
        """Resolve an active discussion thread across real engine and old test-double shapes."""
        finder = getattr(engine, "find_active_discussion", None)
        if callable(finder):
            try:
                candidate = finder(chat_id, thread_id)
            except Exception:
                candidate = None
            if getattr(candidate, "thread_id", None) == thread_id:
                return candidate

        active_discussions = getattr(engine, "_active_discussions", {})
        candidates = active_discussions.get(chat_id, [])
        if not isinstance(candidates, list):
            candidates = [candidates]
        for candidate in candidates:
            if getattr(candidate, "thread_id", None) == thread_id:
                return candidate
        return None

    # ------------------------------------------------------------------
    # Card action handler
    # ------------------------------------------------------------------

    def _refresh_status_card(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Rebuild and update the status panel card in-place."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式，无法刷新状态。")
            return
        team_name = engine.channel.team_name if engine.channel else ""
        status_card = engine.get_status_card(team_name=team_name)
        card_content = json.dumps(status_card, ensure_ascii=False)
        self.update_card(message_id, card_content)

    def _refresh_task_board_card(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Rebuild and update the task board card in-place."""
        from ...slock_engine.card_templates import build_task_board_card

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式，无法刷新任务看板。")
            return
        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agents = engine.registry.list_agents(channel_id=channel_id)
        team_name = engine.channel.team_name if engine.channel else ""
        card = build_task_board_card(engine.tasks, agents, team_name=team_name, channel_id=channel_id)
        self.update_card(message_id, json.dumps(card, ensure_ascii=False))

    def _has_slock_permission(self, engine) -> bool:
        """Pure boolean check: is current operator admin or channel owner? No side effects.

        When ADMIN_USER_IDS is empty (bootstrap state), all users are permitted —
        consistent with the one-way /setadmin bootstrap contract.
        """
        from ...config import get_settings
        from ...thread.manager import get_current_sender_id

        operator_id = get_current_sender_id() or ""
        settings = get_settings()
        admin_ids = settings.admin_user_ids if hasattr(settings, "admin_user_ids") else frozenset()

        # Bootstrap state: no admin configured → permissive
        if not admin_ids:
            return True

        channel_owner_id = ""
        if engine.channel:
            channel_owner_id = getattr(engine.channel, "owner_id", "") or ""

        return (
            (operator_id and operator_id in admin_ids)
            or (operator_id and channel_owner_id and operator_id == channel_owner_id)
        )

    def _check_slock_permission(self, engine, message_id: str, chat_id: str) -> bool:
        """Check if current operator is admin or channel owner. Returns True if authorized."""
        if not self._has_slock_permission(engine):
            perm_msg = "⚠️ 权限不足，仅管理员或团队创建者可执行此操作。"
            if not self.reply_text(message_id, perm_msg):
                self.send_text_to_chat(chat_id, perm_msg)
            return False
        return True

    def _require_slock_permission(self, engine, action_type: str, *, allow_assignee: bool = False, task_id: str = "") -> dict | None:
        """Check permission and return rejection card if denied, None if allowed.

        Authorized: admin_user_ids | channel_owner | (task claimed_by if allow_assignee).
        When ADMIN_USER_IDS is empty (bootstrap state), all users are permitted.
        """
        from ...config import get_settings
        from ...thread.manager import get_current_sender_id

        settings = get_settings()
        sender_id = get_current_sender_id() or ""

        # Check admin
        admin_ids = settings.admin_user_ids if hasattr(settings, "admin_user_ids") else frozenset()

        # Bootstrap state: no admin configured → permissive
        if not admin_ids:
            return None  # allowed

        if sender_id and sender_id in admin_ids:
            return None  # allowed

        # Check channel owner
        if engine and engine.channel:
            if sender_id and sender_id == (getattr(engine.channel, "owner_id", "") or ""):
                return None  # allowed

        # Check task assignee
        if allow_assignee and task_id and engine:
            for task in engine.tasks:
                if task.task_id == task_id and task.claimed_by == sender_id:
                    return None  # allowed

        # Denied - return rejection toast
        return {"toast": {"type": "error", "content": "\u26d4 权限不足：仅管理员、群主或任务负责人可执行此操作"}}

    def _check_assign_rate_limit(self, engine, message_id: str, chat_id: str) -> bool:
        """Check rate-limit for task assignment. Admin/owner bypass. Returns True if allowed."""
        import time as _time

        from ...config import get_settings
        from ...thread.manager import get_current_sender_id

        operator_id = get_current_sender_id() or ""
        settings = get_settings()
        admin_ids = settings.admin_user_ids if hasattr(settings, "admin_user_ids") else frozenset()
        channel_owner_id = ""
        if engine.channel:
            channel_owner_id = getattr(engine.channel, "owner_id", "") or ""

        # Admin and owner bypass rate-limit
        is_privileged = (
            (operator_id and operator_id in admin_ids)
            or (operator_id and channel_owner_id and operator_id == channel_owner_id)
        )
        if is_privileged:
            return True

        # Rate-limit for regular users: sliding window of 60s
        rate_limit = settings.slock_assign_rate_limit
        tracker_key = f"{chat_id}:{operator_id}"
        now = _time.time()
        window = 60.0

        self._prune_assign_rate_limit_tracker(now, window)
        timestamps = self._rate_limit_tracker.get(tracker_key, [])

        if len(timestamps) >= rate_limit:
            self.reply_text(
                message_id,
                f"⚠️ 任务提交频率超限（每分钟最多 {rate_limit} 次），请稍后重试。",
            )
            self._rate_limit_tracker[tracker_key] = timestamps
            return False

        timestamps.append(now)
        self._rate_limit_tracker[tracker_key] = timestamps
        return True

    def _prune_assign_rate_limit_tracker(self, now: float, window: float) -> None:
        """Remove expired assign-rate-limit entries for inactive chat/sender pairs."""
        _GLOBAL_TTL = 3600.0  # Remove keys inactive for > 1 hour
        for key, timestamps in list(self._rate_limit_tracker.items()):
            if not timestamps or now - max(timestamps) > _GLOBAL_TTL:
                # Global TTL: no activity for over 1 hour, remove entirely
                self._rate_limit_tracker.pop(key, None)
                continue
            active = [ts for ts in timestamps if now - ts < window]
            if active:
                self._rate_limit_tracker[key] = active
            else:
                self._rate_limit_tracker.pop(key, None)

    def _check_discussion_permission(self, message_id: str, chat_id: str) -> bool:
        """Check permission for triggering a discussion.

        Admin/owner: always allowed (no rate-limit).
        Regular members: rate-limited to 3 discussion triggers per 5 minutes.
        Returns True if allowed, False if denied (with user notification).
        """
        import time as _time

        from ...config import get_settings
        from ...thread.manager import get_current_sender_id

        operator_id = get_current_sender_id() or ""
        settings = get_settings()
        admin_ids = settings.admin_user_ids if hasattr(settings, "admin_user_ids") else frozenset()

        # Try to get channel owner from engine
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        channel_owner_id = ""
        if engine and engine.channel:
            channel_owner_id = getattr(engine.channel, "owner_id", "") or ""

        # Admin and owner bypass rate-limit
        is_privileged = (
            (operator_id and operator_id in admin_ids)
            or (operator_id and channel_owner_id and operator_id == channel_owner_id)
        )
        if is_privileged:
            return True

        # Rate-limit for regular users: sliding window of 5 minutes, max 3 triggers
        DISCUSSION_WINDOW = 300.0  # 5 minutes
        DISCUSSION_MAX = 3
        tracker_key = f"disc:{chat_id}:{operator_id}"
        now = _time.time()

        timestamps = self._rate_limit_tracker.get(tracker_key, [])
        timestamps = [t for t in timestamps if now - t < DISCUSSION_WINDOW]

        if len(timestamps) >= DISCUSSION_MAX:
            self.reply_text(
                message_id,
                f"⚠️ 讨论触发频率超限（每 5 分钟最多 {DISCUSSION_MAX} 次），请稍后重试。",
            )
            self._rate_limit_tracker[tracker_key] = timestamps
            return False

        timestamps.append(now)
        self._rate_limit_tracker[tracker_key] = timestamps
        return True

    def _check_queue_wait_timeout(self, future, start_time: float, card_message_id: str, chat_id: str) -> bool:
        """Check if task waited too long in queue. Returns True if timed out (caller should abort).

        If future is None (e.g. synchronous executor in tests where the work runs
        before submit() returns), skip timeout detection entirely — the task is
        already executing, so queue-wait is irrelevant.
        """
        import json as _json
        import time

        from ...config import get_settings as _get_settings

        if future is None:
            return False

        _settings = _get_settings()
        _enqueue_time = getattr(future, "enqueue_time", None)
        if _enqueue_time is None:
            _enqueue_time = start_time
        enqueue_elapsed = time.time() - _enqueue_time
        if enqueue_elapsed > _settings.slock_queue_wait_timeout:
            logger.warning(
                "Slock queue wait timeout (%.1fs > %ds) for chat %s",
                enqueue_elapsed, _settings.slock_queue_wait_timeout, chat_id,
            )
            timeout_card = _json.dumps({
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "⏱️ 排队超时"}, "template": "orange"},
                "body": {"elements": [{"tag": "markdown", "content": "消息在队列中等待过久，已自动取消。请稍后重试。"}]},
            }, ensure_ascii=False)
            if card_message_id:
                self.update_card(card_message_id, timeout_card)
            return True
        return False

    def _resolve_escalation(self, message_id: str, chat_id: str, value: dict):
        """Handle admin clicking Retry/Skip/Abort on an escalation card.

        Enhanced flow:
        (a) Get operator via thread-local
        (b) Permission check via _check_slock_permission
        (c) Resolution whitelist validation
        (d) Resolve via engine
        (e) Update card to resolved state
        (f) Fallback to text if card update fails
        (g) Handle duplicate clicks gracefully
        """
        from ...slock_engine.card_templates import build_resolved_escalation_card
        from ...thread.manager import get_current_sender_id

        escalation_id = value.get("escalation_id", "")
        resolution = value.get("resolution", "")

        if not escalation_id or not resolution:
            logger.warning("Escalation resolve missing params: escalation_id=%s resolution=%s", escalation_id, resolution)
            self.send_text_to_chat(chat_id, "⚠️ 升级请求处理参数缺失，请重试或联系管理员。")
            return

        # (a) Get operator identity
        operator_id = get_current_sender_id() or ""
        operator_display = (
            resolve_display_name_nonblocking(operator_id, self.ctx.api_client_factory)
            if operator_id
            else ""
        )

        # (b) Permission check — admin or team owner (unified)
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            logger.warning("No active engine for escalation resolve in chat %s", chat_id)
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式，无法处理升级请求。")
            return

        if not self._check_slock_permission(engine, message_id, chat_id):
            return

        # (c) Resolution whitelist validation
        escalation = engine.get_escalation(escalation_id)
        if not escalation:
            self.send_text_to_chat(chat_id, f"⚠️ Escalation `{escalation_id}` 不存在。")
            return

        # Handle duplicate clicks — already resolved
        if escalation.resolved:
            self.send_text_to_chat(chat_id, f"ℹ️ 此升级请求已处理（{escalation.resolution}），无需重复操作。")
            return

        allowed_options = escalation.options or ["重试", "跳过", "中止"]
        resolution_stripped = resolution.strip()
        if resolution_stripped not in allowed_options:
            logger.warning(
                "Escalation resolve invalid resolution: '%s' not in %s",
                resolution, allowed_options,
            )
            self.send_text_to_chat(
                chat_id,
                f"⚠️ 无效的解决选项 `{resolution}`，允许的选项: {', '.join(allowed_options)}",
            )
            return

        # (d) Execute resolve
        resolved = engine.resolve_escalation(escalation_id, resolution_stripped)
        if not resolved:
            # Race condition: resolved between our check and this call
            self.send_text_to_chat(chat_id, "ℹ️ 此升级请求已处理，无需重复操作。")
            return

        # (e) Build resolved card and update in-place
        resolved_card = build_resolved_escalation_card(
            escalation,
            resolved_by=operator_display or operator_id,
            resolution=resolution_stripped,
            resolved_at=resolved.resolved_at,
            channel_id=chat_id,
        )
        card_json = json.dumps(resolved_card, ensure_ascii=False)
        card_updated = self.update_card(message_id, card_json)

        # (f) Fallback to text confirmation if card update fails
        if not card_updated:
            logger.error(
                "Failed to update escalation card: message_id=%s chat_id=%s",
                message_id, chat_id,
            )
            confirm_text = (
                f"✅ Escalation resolved: **{resolved.agent_name}** — {resolution_stripped}\n"
                f"Reason: {redact_sensitive(resolved.reason)}"
            )
            self.send_text_to_chat(chat_id, confirm_text)

        # (g) Trigger agent recovery based on resolution
        engine.resume_after_escalation(resolved)

    def _handle_clarify_confirm(self, open_message_id: str, open_chat_id: str, value: dict) -> None:
        """Handle "是，这是任务" button click from clarification card.

        Triggers task enqueue logic similar to auto-activate path.
        Validates that the action performer is the original message sender.
        """
        import json as _json

        from ...slock_engine.card_templates.queue_feedback import build_clarification_confirmed_card
        from ...thread.manager import get_current_sender_id

        message_preview = str(value.get("message_preview") or "")
        original_message_id = str(value.get("message_id") or "")
        original_sender_id = str(value.get("sender_id") or "")
        current_sender_id = get_current_sender_id() or ""

        # Sender verification: only original sender can confirm
        if original_sender_id and current_sender_id and original_sender_id != current_sender_id:
            logger.warning(
                "Clarify confirm rejected: sender mismatch (original=%s, current=%s)",
                original_sender_id, current_sender_id,
            )
            self.reply_text(open_message_id, "⚠️ 仅消息发送者可确认此操作。")
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(open_chat_id)

        # Build confirmation card first (will be shown regardless of path)
        confirm_card = build_clarification_confirmed_card(message_preview=message_preview)
        confirm_card_json = _json.dumps(confirm_card, ensure_ascii=False)

        if engine:
            # Engine already active — route the message to the engine
            logger.info(
                "Clarify confirm: engine already active, routing message chat=%s preview=%s",
                open_chat_id, message_preview[:50],
            )
            # Update card first to show confirmation
            self.update_card(open_message_id, confirm_card_json)
            # Then route the message to the engine
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            # Use original message_id if available, otherwise use the card message_id
            target_message_id = original_message_id or open_message_id
            self.handle_message(target_message_id, open_chat_id, message_preview, project)
        else:
            # Engine not active — trigger auto-activate path
            logger.info(
                "Clarify confirm: engine not active, triggering auto-activate chat=%s preview=%s",
                open_chat_id, message_preview[:50],
            )
            # Activate slock with the message as requirement
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            success = self.activate_slock(
                message_id=open_message_id,
                chat_id=open_chat_id,
                requirement=message_preview,
                project=project,
            )
            if not success:
                # If activation failed, at least show the confirmation card
                self.update_card(open_message_id, confirm_card_json)

    def _handle_clarify_ignore(self, open_message_id: str, open_chat_id: str, value: dict) -> None:
        """Handle "不是，只是聊天" button click from clarification card.

        Updates the card to "已忽略" state without creating any task.
        Validates that the action performer is the original message sender.
        """
        import json as _json

        from ...slock_engine.card_templates.queue_feedback import build_clarification_ignored_card
        from ...thread.manager import get_current_sender_id

        message_preview = str(value.get("message_preview") or "")
        original_sender_id = str(value.get("sender_id") or "")
        current_sender_id = get_current_sender_id() or ""

        # Sender verification: only original sender can ignore
        if original_sender_id and current_sender_id and original_sender_id != current_sender_id:
            logger.warning(
                "Clarify ignore rejected: sender mismatch (original=%s, current=%s)",
                original_sender_id, current_sender_id,
            )
            self.reply_text(open_message_id, "⚠️ 仅消息发送者可忽略此操作。")
            return

        logger.info(
            "Clarify ignore: user marked message as chat chat=%s preview=%s",
            open_chat_id, message_preview[:50],
        )

        # Build and send ignored card
        ignored_card = build_clarification_ignored_card(message_preview=message_preview)
        ignored_card_json = _json.dumps(ignored_card, ensure_ascii=False)
        self.update_card(open_message_id, ignored_card_json)

    def handle_card_action(self, open_message_id: str, open_chat_id: str, action_type: str, value: dict):
        """Handle slock_* card actions."""
        employee_runtime_actions = {
            "employee_runtime_show_status",
            "employee_runtime_recycle_session",
            "employee_runtime_rebuild_workspace",
            "employee_runtime_lint_knowledge",
            "employee_runtime_retry_review",
        }
        if action_type in employee_runtime_actions:
            from ...autonomous.manager.cards import build_employee_runtime_status_card
            from ...thread.manager import (
                get_current_is_p2p,
                get_current_sender_id,
                get_current_tenant_key,
            )

            facade = getattr(self.ctx, "employee_runtime_facade", None)
            tenant_key = get_current_tenant_key() or ""
            agent_id = str(value.get("agent_id") or "")
            sender_id = get_current_sender_id() or ""
            is_admin = (
                get_current_is_p2p()
                and sender_id
                in frozenset(getattr(self.ctx.settings, "admin_user_ids", ()) or ())
            )
            if facade is None or not tenant_key or not agent_id:
                self.send_text_to_chat(open_chat_id, "⚠️ 员工运行时状态当前不可用。")
                return
            if action_type != "employee_runtime_show_status" and not is_admin:
                self.send_text_to_chat(
                    open_chat_id,
                    "⛔ 员工恢复动作仅允许配置管理员在主 Bot 私聊中执行。",
                )
                return
            try:
                if action_type == "employee_runtime_recycle_session":
                    facade.recycle_employee_session(tenant_key, agent_id)
                    self.send_text_to_chat(open_chat_id, "✅ 模型会话已回收；下次任务将冷启动。")
                elif action_type == "employee_runtime_rebuild_workspace":
                    facade.rebuild_employee_workspace(tenant_key, agent_id)
                    self.send_text_to_chat(open_chat_id, "✅ Workspace 已从 canonical projection 重建。")
                elif action_type == "employee_runtime_lint_knowledge":
                    report = facade.lint_employee_knowledge(tenant_key, agent_id)
                    issues = tuple(getattr(report, "issues", ()) or ())
                    if issues:
                        codes = "、".join(
                            str(getattr(issue, "code", "unknown")) for issue in issues
                        )
                        self.send_text_to_chat(
                            open_chat_id,
                            f"⚠️ Knowledge 检查发现 {len(issues)} 项：{codes}",
                        )
                    else:
                        self.send_text_to_chat(open_chat_id, "✅ Knowledge 检查通过。")
                elif action_type == "employee_runtime_retry_review":
                    review_id = str(value.get("review_id") or "")
                    if not review_id:
                        raise ValueError("review item is missing")
                    facade.retry_employee_knowledge_review(
                        tenant_key,
                        agent_id,
                        review_id,
                    )
                    self.send_text_to_chat(
                        open_chat_id,
                        "✅ Review Item 已重新入队；原始来源与权限边界保持不变。",
                    )
                view = facade.get_employee_runtime_status(tenant_key, agent_id)
            except KeyError:
                self.send_text_to_chat(open_chat_id, "⚠️ 员工或 Review Item 已失效，请刷新 `/roster`。")
                return
            except Exception:
                logger.exception("employee runtime administration failed")
                self.send_text_to_chat(
                    open_chat_id,
                    "⚠️ 恢复动作未能安全完成；状态未被伪造为 READY。",
                )
                return
            self.update_card(
                open_message_id,
                json.dumps(
                    build_employee_runtime_status_card(view, admin=is_admin),
                    ensure_ascii=False,
                ),
            )
            return

        if action_type == "slock_reauthorize_employee_app":
            from ...autonomous.provisioning.hire_service import HireAdmissionError
            from ...thread.manager import (
                get_current_is_p2p,
                get_current_sender_id,
                get_current_tenant_key,
            )

            sender_id = get_current_sender_id() or ""
            tenant_key = get_current_tenant_key() or ""
            admins = frozenset(
                getattr(self.ctx.settings, "admin_user_ids", ()) or ()
            )
            if not get_current_is_p2p() or not sender_id or sender_id not in admins:
                self.send_text_to_chat(
                    open_chat_id,
                    "⛔ 原地授权仅允许配置管理员在主 Bot 私聊中执行。",
                )
                return
            service = getattr(self.ctx, "employee_hire_service", None)
            agent_id = str(value.get("agent_id") or "")
            if service is None or not tenant_key or not agent_id:
                self.send_text_to_chat(
                    open_chat_id,
                    "⚠️ 员工授权服务当前不可用，请稍后重试。",
                )
                return
            try:
                operation = service.request_manifest_reauthorization(
                    tenant_key=tenant_key,
                    agent_id=agent_id,
                    request_id=open_message_id,
                )
            except HireAdmissionError:
                self.send_text_to_chat(
                    open_chat_id,
                    "⚠️ 原地授权未能安全启动；远端状态仍为未知，请刷新 `/roster` 后重试。",
                )
                return
            except Exception:
                logger.exception("employee manifest reauthorization dispatch failed")
                self.send_text_to_chat(
                    open_chat_id,
                    "⚠️ 原地授权未能安全启动；远端状态仍为未知。",
                )
                return
            phase = str(getattr(getattr(operation, "phase", None), "value", ""))
            if phase == "committed":
                self.send_text_to_chat(
                    open_chat_id,
                    "✅ 该员工的当前 manifest 已有可信飞书官方回执，无需重复授权。",
                )
                return
            self.send_text_to_chat(
                open_chat_id,
                "已锚定原地授权请求。请按随后发送的飞书官方授权链接完成授权和发布；"
                "只有官方同 App 回执成功后，`/roster` 才会显示当前。",
            )
            return

        if action_type in ("slock_role_add_pick", "slock_role_add_select"):
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine:
                self.send_text_to_chat(open_chat_id, "⚠️ 当前没有活跃的 Slock 团队。")
                return
            if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                return
            from ...thread.manager import get_current_tenant_key

            tenant_key = get_current_tenant_key() or ""
            agent_id = str(value.get("_option") or "")
            service = getattr(self.ctx, "employee_membership_service", None)
            employee = service.get_employee(tenant_key, agent_id) if service and tenant_key and agent_id else None
            if employee is None:
                self.send_text_to_chat(open_chat_id, "⚠️ 员工选择已失效，请重新打开列表。")
                return
            emoji = str(getattr(employee, "emoji", "🤖"))
            name = str(getattr(employee, "name", "员工"))
            confirm_card = {
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "确认添加员工"}, "template": "orange"},
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"将把 **{emoji} {name}** 加入本群"}},
                    *build_responsive_layout([
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "确认加入"},
                            "type": "primary",
                            "value": {"action": "slock_role_add_confirm", "chat_id": open_chat_id, "agent_id": agent_id},
                        },
                    ]),
                ],
            }
            confirm_card_json = json.dumps(confirm_card, ensure_ascii=False)
            self.update_card(open_message_id, confirm_card_json)
            return

        if action_type == "slock_role_add_confirm":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine:
                self.send_text_to_chat(open_chat_id, "⚠️ 当前没有活跃的 Slock 团队。")
                return
            if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                return
            from ...thread.manager import get_current_tenant_key

            tenant_key = get_current_tenant_key() or ""
            agent_id = str(value.get("agent_id") or "")
            service = getattr(self.ctx, "employee_membership_service", None)
            employee = service.get_employee(tenant_key, agent_id) if service and tenant_key and agent_id else None
            if employee is None:
                self.send_text_to_chat(open_chat_id, "⚠️ 员工选择已失效，请重新打开列表。")
                return
            self._change_employee_membership(
                open_message_id,
                open_chat_id,
                employee,
                operation="add",
            )
            return

        # Escalation resolve needs extra params from value — handle before standard dispatch
        if action_type == "slock_escalation_resolve":
            self._resolve_escalation(open_message_id, open_chat_id, value)
            return

        # Per-agent stop needs agent_id from value — handle before standard dispatch
        if action_type == "slock_stop_agent":
            self._stop_single_agent(open_message_id, open_chat_id, value)
            return

        if action_type == "slock_agent_follow_up":
            agent_name = value.get("agent_name") or value.get("agent_id") or "Agent"
            self.send_text_to_chat(open_chat_id, f"可以直接在群里 @ {agent_name} 继续追问。")
            return

        if action_type == "slock_agent_show_reasoning":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id) or manager.get_active_engine(open_chat_id)
            agent_id = str(value.get("agent_id") or "")
            task_id = str(value.get("task_id") or "")
            rejection = self._require_slock_permission(engine, action_type, allow_assignee=True, task_id=task_id)
            if rejection:
                return rejection
            snapshot = engine.memory.read_agent_reasoning_snapshot(agent_id, task_id) if engine and agent_id and task_id else {}
            if snapshot:
                tool_label = snapshot.get("tool_name") or "unknown"
                model_label = snapshot.get("model_name") or "default"
                prompt_summary = redact_sensitive(str(snapshot.get("prompt_summary") or ""))[:1200]
                result_summary = redact_sensitive(str(snapshot.get("result_summary") or ""))[:1800]
                self.send_text_to_chat(
                    open_chat_id,
                    "🧠 **执行摘要**\n\n"
                    f"• Agent: `{agent_id[:8]}`\n"
                    f"• 工具/模型: `{tool_label}` / `{model_label}`\n\n"
                    f"**输入摘要**\n{prompt_summary or '(空)'}\n\n"
                    f"**输出摘要**\n{result_summary or '(空)'}",
                )
                return
            self.send_text_to_chat(open_chat_id, "当前 Agent 回复未保存可展示的执行摘要；可查看任务执行卡和消息归档。")
            return

        if action_type == "slock_agent_mark_done":
            task_id = value.get("task_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            rejection = self._require_slock_permission(engine, action_type, allow_assignee=True, task_id=task_id)
            if rejection:
                if isinstance(rejection, dict):
                    content = rejection.get("toast", {}).get("content", "权限不足")
                    self.send_text_to_chat(open_chat_id, content)
                    return rejection
                if not self._has_slock_permission(engine):
                    self.send_text_to_chat(open_chat_id, "⚠️ 权限不足，仅管理员或团队创建者可执行此操作。")
                    return rejection
            if not self._has_slock_permission(engine) and isinstance(rejection, dict):
                return rejection
            if task_id and engine:
                engine._force_complete_task(task_id)
            self.send_text_to_chat(open_chat_id, "✅ 已标记完成。")
            return

        # --- Task 20: Show agent memory (L1 snapshot) ---
        if action_type == "slock_agent_show_memory":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            agent_id = str(value.get("agent_id") or "")
            rejection = self._require_slock_permission(engine, action_type)
            if rejection:
                return rejection
            if engine and agent_id:
                memory = engine.memory.read_agent_memory(agent_id)
                if memory:
                    import json as _json

                    from src.slock_engine.card_templates import build_memory_display_card
                    agent = engine.registry.get(agent_id)
                    agent_name = agent.display_name if agent else agent_id[:8]
                    card = build_memory_display_card(memory, agent_name=agent_name)
                    self.send_card_to_chat(open_chat_id, _json.dumps(card, ensure_ascii=False))
                    return
            self.send_text_to_chat(open_chat_id, "该 Agent 暂无记忆记录。")
            return

        # --- Task 21: Show role switch card ---
        if action_type == "slock_agent_switch_role":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            agent_id = str(value.get("agent_id") or "")
            rejection = self._require_slock_permission(engine, action_type)
            if rejection:
                if isinstance(rejection, dict):
                    content = rejection.get("toast", {}).get("content", "权限不足")
                    self.send_text_to_chat(open_chat_id, content)
                    return rejection
                if not self._has_slock_permission(engine):
                    self.send_text_to_chat(open_chat_id, "⚠️ 权限不足，仅管理员或团队创建者可执行此操作。")
                    return rejection
            if not self._has_slock_permission(engine) and isinstance(rejection, dict):
                return rejection
            if engine and agent_id:
                channel_id = engine.channel.channel_id if engine.channel else ""
                agents = engine.registry.list_agents(channel_id=channel_id)
                available_roles = sorted({a.role for a in agents if a.role})
                # Add standard roles if not already present
                for std_role in ("coder", "reviewer", "writer", "tester", "planner", "architect"):
                    if std_role not in available_roles:
                        available_roles.append(std_role)
                import json as _json

                from src.slock_engine.card_templates import build_role_switch_card
                project_id = value.get("project_id", "")
                card = build_role_switch_card(
                    roles=available_roles,
                    agent_id=agent_id,
                    channel_id=channel_id,
                    project_id=project_id,
                )
                self.send_card_to_chat(open_chat_id, _json.dumps(card, ensure_ascii=False))
            else:
                self.send_text_to_chat(open_chat_id, "⚠️ 未找到活跃引擎或 Agent。")
            return

        # --- Task 22: Confirm role switch ---
        if action_type == "slock_confirm_switch_role":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            agent_id = str(value.get("agent_id") or "")
            target_role = str(value.get("target_role") or "")
            rejection = self._require_slock_permission(engine, action_type)
            if rejection:
                return rejection
            if engine and agent_id and target_role:
                agent = engine.registry.get(agent_id)
                if agent:
                    old_role = agent.role
                    agent.role = target_role
                    engine.registry.update(agent)
                    # Update memory role field
                    memory = engine.memory.read_agent_memory(agent_id)
                    if memory:
                        memory.role = f"{target_role}: {agent.system_prompt[:200]}" if agent.system_prompt else target_role
                        engine.memory.write_agent_memory(agent_id, memory)
                    self.send_text_to_chat(
                        open_chat_id,
                        f"🎭 **{agent.display_name}** 角色已切换: `{old_role}` → `{target_role}`",
                    )
                    return
            self.send_text_to_chat(open_chat_id, "⚠️ 角色切换失败，请重试。")
            return

        if action_type == "slock_confirm_discussion":
            thread_id = str(value.get("thread_id") or "")
            trust_type = str(value.get("trust_type") or "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine:
                self.send_text_to_chat(open_chat_id, "⚠️ 未找到活跃 Slock 引擎。")
                return
            engine.confirm_discussion(thread_id, trust_type=trust_type)
            self.send_text_to_chat(open_chat_id, "✅ 讨论已启动。")
            return

        if action_type == "slock_cancel_discussion":
            thread_id = str(value.get("thread_id") or "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine:
                self.send_text_to_chat(open_chat_id, "⚠️ 未找到活跃 Slock 引擎。")
                return
            engine.cancel_discussion(thread_id)
            self.send_text_to_chat(open_chat_id, "✅ 已取消讨论。")
            return

        # --- Tasks 26-28: Dissolve confirmation & undo ---
        if action_type == "slock_confirm_dissolve":
            team_name = str(value.get("team_name") or "")
            target_chat_id = str(value.get("channel_id") or "")
            manager = self._get_engine_manager()
            engine = (
                manager.get_activated_engine(target_chat_id)
                if target_chat_id and target_chat_id == open_chat_id
                else None
            )
            if engine and engine.channel:
                if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                    return
                self.dissolve_team(open_message_id, open_chat_id, target_chat_id)
            else:
                self.send_text_to_chat(open_chat_id, "⚠️ 未找到目标团队。")
            return

        if action_type == "slock_undo_dissolve":
            self.send_text_to_chat(open_chat_id, "⚠️ 解散会删除飞书群，当前操作不可撤销。")
            return

        # --- Task 24: Form submissions from command panel ---
        if action_type == "slock_form_new_team":
            team_name = str(value.get("team_name") or "").strip()
            if not team_name:
                self.send_text_to_chat(open_chat_id, "⚠️ 请输入团队名称。")
                return
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            self.create_team(open_message_id, open_chat_id, team_name, project)
            return

        if action_type == "slock_form_new_role":
            role_name = str(value.get("role_name") or "").strip()
            role_type = str(value.get("role_type") or "coder").strip()
            agent_type = str(value.get("agent_type") or "coco").strip()
            if not role_name:
                self.send_text_to_chat(open_chat_id, "⚠️ 请输入角色名称。")
                return
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            # Build params string for create_role
            params_str = f"{role_name} --type={role_type} --agent={agent_type}"
            self.create_role(open_message_id, open_chat_id, params_str, project)
            return

        if action_type == "slock_form_council":
            topic = str(value.get("topic") or "").strip()
            if not topic:
                self.send_text_to_chat(open_chat_id, "⚠️ 请输入评审议题。")
                return
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            self.run_council(open_message_id, open_chat_id, topic, project)
            return

        if action_type == "slock_form_discuss":
            topic = str(value.get("topic") or "").strip()
            if not topic:
                self.send_text_to_chat(open_chat_id, "⚠️ 请输入讨论主题。")
                return
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            self._trigger_nli_discussion(open_message_id, open_chat_id, topic, {}, project)
            return

        if action_type == "slock_discussion_expand":
            self._expand_discussion(open_chat_id, value)
            return

        if action_type == "slock_discussion_stop":
            self._stop_discussion(open_chat_id, value)
            return

        # --- Tasks 19+20: Collaboration plan actions & user intervention ---
        if action_type == "slock_plan_approve":
            plan_id = value.get("plan_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and plan_id:
                if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                    return
                success = engine.collaboration_orchestrator.approve_plan(plan_id)
                if success:
                    import json as _json

                    from src.slock_engine.card_templates.progress import build_collaboration_plan_card
                    plan = engine.collaboration_orchestrator.get_plan(plan_id)
                    if plan:
                        channel_id = engine.channel.channel_id if engine.channel else ""
                        agents = engine.registry.list_agents(channel_id=channel_id)
                        card = build_collaboration_plan_card(plan, agents, channel_id=channel_id)
                        self.update_card(open_message_id, _json.dumps(card, ensure_ascii=False))
            return

        if action_type == "slock_plan_cancel":
            plan_id = value.get("plan_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and plan_id:
                if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                    return
                engine.collaboration_orchestrator.cancel_plan(plan_id)
            return

        if action_type == "slock_progress_refresh":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine:
                import json as _json

                from src.slock_engine.card_templates.progress import build_progress_overview_card
                channel_id = engine.channel.channel_id if engine.channel else ""
                plans = engine.collaboration_orchestrator.list_active_plans(channel_id)
                agents = engine.registry.list_agents(channel_id=channel_id)
                card = build_progress_overview_card(plans, agents, channel_id=channel_id)
                self.send_card_to_chat(open_chat_id, _json.dumps(card, ensure_ascii=False))
            return

        if action_type == "slock_user_intervention":
            plan_id = value.get("plan_id", "")
            message = value.get("message", "") or value.get("supplement_content", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine or not plan_id:
                self.send_text_to_chat(open_chat_id, "⚠️ 引擎未激活或缺少 plan_id。")
                return
            if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                return

            plan = engine.collaboration_orchestrator.get_plan(plan_id)
            if not plan:
                self.send_text_to_chat(open_chat_id, "⚠️ 未找到该计划。")
                return

            # If no message provided, pause the plan and prompt user for input
            if not message:
                engine.collaboration_orchestrator.pause_plan(plan_id)
                self.send_text_to_chat(
                    open_chat_id,
                    "⏸ 计划已暂停。请在讨论面板中输入补充信息后点击发送，或使用 ▶️ 恢复 按钮继续。",
                )
                return

            # Inject intervention message into current step's agent context
            current = plan.current_step
            if current and current.agent_id:
                engine.memory.update_agent_context(
                    current.agent_id,
                    f"[用户干预] {message[:200]}",
                )
                self.send_text_to_chat(
                    open_chat_id,
                    f"✅ 已将干预信息注入 Agent `{current.agent_id[:8]}` 上下文。",
                )
                # Resume if paused
                from ...slock_engine.models import CollaborationPlanStatus
                if plan.status == CollaborationPlanStatus.PAUSED:
                    engine.collaboration_orchestrator.resume_plan(plan_id)
                    self.send_text_to_chat(open_chat_id, "▶️ 计划已恢复执行。")
            else:
                self.send_text_to_chat(open_chat_id, "⚠️ 当前步骤无执行中的 Agent，无法注入干预信息。")
            return

        if action_type == "slock_plan_supplement":
            plan_id = value.get("plan_id", "")
            # 兼容读取: 优先 _form_value 子字典，回退顶层
            form_value = value.get("_form_value", {})
            content = (form_value.get("supplement_content") or value.get("supplement_content", "")).strip()
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine or not plan_id:
                return
            if not content:
                return {"toast": {"type": "error", "content": "请输入补充信息内容"}}
            # Permission check
            rejection = self._require_slock_permission(engine, action_type)
            if rejection:
                return rejection
            # Text length check
            if len(content) > 2000:
                return {"toast": {"type": "error", "content": "补充内容过长，请限制在 2000 字符以内"}}
            # Sensitive content filter
            import re as _re
            if _re.search(r'(token|key|secret|password|credential)[=:]\s*\S{8,}', content, _re.I):
                return {"toast": {"type": "error", "content": "检测到疑似敏感凭据信息，请移除后重试"}}
            plan = engine.collaboration_orchestrator.get_plan(plan_id)
            if not plan:
                return
            # Inject to all collaborating agents in this plan
            agent_ids = {step.agent_id for step in plan.steps if step.agent_id}
            if agent_ids:
                supplement_text = f"[用户补充] {content[:500]}"
                for agent_id in agent_ids:
                    engine.memory.update_agent_context(agent_id, supplement_text)
                self.send_text_to_chat(
                    open_chat_id,
                    f"✅ 补充信息已注入 {len(agent_ids)} 个协作 Agent 上下文。",
                )
            else:
                self.send_text_to_chat(open_chat_id, "⚠️ 计划中暂无已分配的 Agent。")
            return

        # --- Collaboration plan detail / pause / resume ---
        if action_type == "slock_show_plan_detail":
            plan_id = value.get("plan_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and plan_id:
                from ...slock_engine.card_templates.progress import build_collaboration_plan_card
                plan = engine.collaboration_orchestrator.get_plan(plan_id)
                if plan:
                    channel_id = engine.channel.channel_id if engine.channel else ""
                    agents = engine.registry.list_agents(channel_id=channel_id)
                    card = build_collaboration_plan_card(plan, agents, channel_id=channel_id)
                    self.send_card_to_chat(open_chat_id, json.dumps(card, ensure_ascii=False))
                    return
            self.send_text_to_chat(open_chat_id, "⚠️ 未找到该计划。")
            return

        if action_type == "slock_pause_plan":
            plan_id = value.get("plan_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and plan_id:
                if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                    return
                success = engine.collaboration_orchestrator.pause_plan(plan_id)
                if success:
                    self.send_text_to_chat(open_chat_id, f"⏸ 计划 `{plan_id[:8]}` 已暂停。")
                    return
            self.send_text_to_chat(open_chat_id, "⚠️ 暂停失败，计划不存在或状态不允许。")
            return

        if action_type == "slock_resume_plan":
            plan_id = value.get("plan_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and plan_id:
                if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                    return
                success = engine.collaboration_orchestrator.resume_plan(plan_id)
                if success:
                    self.send_text_to_chat(open_chat_id, f"▶️ 计划 `{plan_id[:8]}` 已恢复执行。")
                    return
            self.send_text_to_chat(open_chat_id, "⚠️ 恢复失败，计划不存在或状态不允许。")
            return

        # --- Role info from card button ---
        if action_type == "slock_role_info":
            agent_id = value.get("agent_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and agent_id:
                agent = engine.registry.get(agent_id)
                if agent:
                    self.show_role_info(open_message_id, open_chat_id, agent.name)
                    return
            self.send_text_to_chat(open_chat_id, "⚠️ 未找到该角色。")
            return

        # --- Task board actions ---
        if action_type == "slock_new_task":
            content = value.get("content", "")
            if not content:
                self.send_text_to_chat(open_chat_id, "⚠️ 请输入任务内容。")
                return
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine:
                self.assign_task(open_message_id, open_chat_id, content)
            return

        if action_type == "slock_dispatch_tasks":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine:
                engine.dispatch_pending_tasks()
                self.send_text_to_chat(open_chat_id, "✅ 已派发待处理任务。")
            return

        if action_type == "slock_show_task_board":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine:
                from ...slock_engine.card_templates import build_task_board_card
                channel_id = engine.channel.channel_id if engine.channel else ""
                agents = engine.registry.list_agents(channel_id=channel_id)
                card = build_task_board_card(
                    tasks=engine.tasks,
                    agents=agents,
                    channel_id=channel_id,
                )
                self.send_card_to_chat(open_chat_id, json.dumps(card, ensure_ascii=False))
            return

        if action_type == "slock_assign_task_to_agent":
            task_id = value.get("task_id", "")
            agent_id = value.get("agent_id", "")
            task_content = value.get("task_content", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine or not agent_id:
                self.send_text_to_chat(open_chat_id, "⚠️ 引擎未激活或缺少 agent_id。")
                return

            # Case 1: Form submission with task_content → create + assign
            if task_content and not task_id:
                new_task = engine.create_and_assign_task(task_content.strip(), agent_id)
                if new_task:
                    self.send_text_to_chat(
                        open_chat_id,
                        f"✅ 任务已创建并分配给 `{agent_id[:8]}`。",
                    )
                else:
                    self.send_text_to_chat(open_chat_id, "⚠️ 任务创建失败。")
                return

            # Case 2: Existing task_id → reassign
            if task_id:
                success = engine.assign_task_to_agent(task_id, agent_id)
                if success:
                    self.send_text_to_chat(open_chat_id, f"✅ 任务已分配给 `{agent_id[:8]}`。")
                else:
                    self.send_text_to_chat(open_chat_id, "⚠️ 任务分配失败。")
                return

            self.send_text_to_chat(open_chat_id, "⚠️ 缺少任务内容或任务 ID。")
            return

        # --- Task 9: Clarification card button handlers ---
        if action_type == "slock_clarify_confirm":
            self._handle_clarify_confirm(open_message_id, open_chat_id, value)
            return

        if action_type == "slock_clarify_ignore":
            self._handle_clarify_ignore(open_message_id, open_chat_id, value)
            return

        # --- Hub card button routing (slock_hub_cmd) ---
        if action_type == "slock_hub_cmd":
            cmd_text = str(value.get("cmd") or "").strip()
            if not cmd_text:
                self.send_text_to_chat(open_chat_id, "⚠️ 无效的命令。")
                return
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            self.handle_slock_command(open_message_id, open_chat_id, cmd_text, project)
            return

        # --- Command panel button routing (slock_cmd_* prefix) ---
        if action_type.startswith("slock_cmd_"):
            self._dispatch_cmd_panel_action(open_message_id, open_chat_id, action_type, value)
            return

        project_id = value.get("project_id", "")
        target_project = (
            self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
        )

        slock_actions = {
            "slock_stop": self.stop_slock_engine,
            "slock_refresh_status": self._refresh_status_card,
            "slock_refresh_task_board": self._refresh_task_board_card,
        }

        if action_type not in slock_actions:
            logger.warning("Unhandled slock card action: %s", action_type)

        self._dispatch_standard_card_action(CardActionContext(
            open_message_id=open_message_id,
            open_chat_id=open_chat_id,
            action_type=action_type,
            value=value,
            prefix="slock",
            action_map=slock_actions,
            toggle_log_method=self._toggle_log,
            switch_mode_method=self._switch_card_mode,
            toggle_ac_method=self._toggle_ac,
            project=target_project,
        ))

    # ------------------------------------------------------------------
    # Command panel button dispatch
    # ------------------------------------------------------------------

    def _dispatch_cmd_panel_action(
        self, message_id: str, chat_id: str, action_type: str, value: dict
    ) -> None:
        """Route slock_cmd_* button actions to existing handler methods."""
        project = None
        project_id = value.get("project_id", "")
        if project_id:
            project = self.project_manager.get_project_for_chat(project_id, chat_id)

        # --- Task 25: Permission check for destructive actions ---
        _PERM_REQUIRED = {"slock_cmd_dissolve_team", "slock_cmd_stop"}
        if action_type in _PERM_REQUIRED:
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(chat_id)
            if engine and not self._check_slock_permission(engine, message_id, chat_id):
                return

        # --- Task 26: Dissolve with confirmation card ---
        if action_type == "slock_cmd_dissolve_team":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(chat_id)
            if not engine or not engine.channel:
                self.send_text_to_chat(chat_id, "⚠️ 当前没有活跃团队可解散。")
                return
            team_name = engine.channel.team_name or engine.channel.name or "当前团队"
            channel_id = engine.channel.channel_id
            import json as _json
            buttons = [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "确认解散"},
                    "type": "danger",
                    "behaviors": [{
                        "type": "callback",
                        "value": {
                            "action": "slock_confirm_dissolve",
                            "team_name": team_name,
                            "channel_id": channel_id,
                            "project_id": project_id,
                        },
                    }],
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "取消"},
                    "type": "default",
                    "behaviors": [{
                        "type": "callback",
                        "value": {"action": "slock_noop", "channel_id": channel_id},
                    }],
                },
            ]
            confirm_card = {
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "⚠️ 确认解散团队"}, "template": "red"},
                "body": {"elements": [
                    {"tag": "markdown", "content": f"即将解散团队 **{team_name}**，此操作将：\n- 停止所有 Agent\n- 删除飞书群\n- 清除运行时状态\n\n确认继续？"},
                    *build_responsive_layout(buttons),
                ]},
            }
            sent_message_id = self.send_card_to_chat(
                chat_id, _json.dumps(confirm_card, ensure_ascii=False)
            )
            if not sent_message_id:
                self.send_text_to_chat(
                    chat_id,
                    "⚠️ 解散确认卡发送失败，未执行解散。请重新打开 `/slock` 控制台后再试。",
                )
            return

        # --- Task 23: Discuss routing ---
        if action_type == "slock_cmd_discuss":
            topic = str(value.get("topic") or "").strip()
            if not topic:
                # Task 29: send hint for empty params
                self.send_text_to_chat(chat_id, "💡 请输入讨论主题，例如: `/slock discuss 方案对比`")
                return
            self._trigger_nli_discussion(message_id, chat_id, topic, {}, project)
            return

        # --- Task 23: Council routing (empty topic → hint) ---
        if action_type == "slock_cmd_council":
            topic = str(value.get("topic") or "").strip()
            if not topic:
                self.send_text_to_chat(chat_id, "💡 请输入评审议题，例如: `/council 方案是否可行`")
                return
            self.run_council(message_id, chat_id, topic, project)
            return

        # --- Task 22: Extended panel routing ---
        if action_type == "slock_cmd_panel_extended":
            import json as _json

            from ...slock_engine.card_templates import build_command_panel_extended_card
            extended_card = build_command_panel_extended_card(channel_id=chat_id, project_id=project_id)
            self.send_card_to_chat(chat_id, _json.dumps(extended_card, ensure_ascii=False))
            return

        # --- Task 21: Memory routing (empty target → hint) ---
        if action_type == "slock_cmd_memory":
            target = str(value.get("target") or "").strip()
            if not target:
                self.send_text_to_chat(chat_id, "💡 请指定角色，例如: `/memory @coder` 或 `/memory coder`")
                return
            # Delegate to handle_slock_command with /memory syntax
            self.handle_slock_command(message_id, chat_id, f"/memory {target}", project)
            return

        # --- Task 20: Parameter-required actions with hints ---
        if action_type == "slock_cmd_task_assign":
            self.send_text_to_chat(chat_id, "💡 请使用命令分配任务，例如: `/task assign 修复登录bug @coder`")
            return
        if action_type == "slock_cmd_role_info":
            self.send_text_to_chat(chat_id, "💡 请指定角色，例如: `/role info coder`")
            return
        if action_type == "slock_cmd_role_remove":
            self.send_text_to_chat(chat_id, "💡 请指定要移除的角色，例如: `/role remove coder`")
            return
        if action_type == "slock_cmd_team_status":
            self.send_text_to_chat(chat_id, "💡 请指定团队，例如: `/team status 前端团队` 或直接使用 `/slock status`")
            return

        # --- Command fix suggestion: execute the suggested command ---
        if action_type == "slock_cmd_fix":
            fix_command = str(value.get("fix_command") or "").strip()
            if fix_command:
                self.handle_slock_command(message_id, chat_id, fix_command, project)
            else:
                self.send_text_to_chat(chat_id, "⚠️ 修正命令为空，请手动输入命令")
            return

        _CMD_DISPATCH: dict[str, callable] = {
            "slock_cmd_team_list": lambda: self.list_teams(message_id, chat_id, project),
            "slock_cmd_new_team": lambda: self.create_team(message_id, chat_id, "", project),
            "slock_cmd_role_list": lambda: self.list_roles(message_id, chat_id, project),
            "slock_cmd_new_role": lambda: self.create_role(message_id, chat_id, "", project),
            "slock_cmd_task_list": lambda: self.list_tasks(message_id, chat_id, project),
            "slock_cmd_task_status": lambda: self.show_task_status(message_id, chat_id, project),
            "slock_cmd_status": lambda: self._refresh_status_card(message_id, chat_id, value),
        }

        handler = _CMD_DISPATCH.get(action_type)
        if handler:
            handler()
        else:
            self.send_text_to_chat(chat_id, f"未知的面板操作: {action_type}")

    # ------------------------------------------------------------------
    # Static command detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_slock_command(text: str, chat_id: str | None = None, manager=None) -> bool:
        """Check if text is any slock-related command."""
        return is_slock_command(text, chat_id=chat_id, manager=manager)
