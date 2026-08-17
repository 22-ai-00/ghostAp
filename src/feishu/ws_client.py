"""Feishu WebSocket 客户端（核心路由枢纽）。

职责概览：
- 接收飞书 WS 事件（消息、卡片动作、反应等）并做基础校验/去重。
- 解码并准入事件，将消息和卡片动作交给权威 handler。
- 通过 `TaskScheduler` 提供：按项目串行、全局并发限制、系统命令快通道、背压与熔断。

关键设计点：
- 兼容性：部分 lark-oapi 版本不包含完整的 callback model 类型；这里对仅用于类型标注的符号做了降级处理。
"""

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

import lark_oapi as lark
from lark_channel import EventDispatcherHandler as ChannelEventDispatcherHandler
from lark_channel import FeishuChannel
from lark_channel import LogLevel as ChannelLogLevel
from lark_channel import TransportConfig as ChannelTransportConfig
from lark_channel.event.callback.model.p2_card_action_trigger import (
    P2CardActionTrigger,
    P2CardActionTriggerResponse,
)
from lark_oapi.api.im.v1 import GetChatRequest

# NOTE: lark-oapi 的 event callback models 在不同版本中并不完整。
# 本项目仅将 P2ImMessageReceiveV1 用于类型标注；运行时缺失不应导致 import 失败。
try:  # pragma: no cover
    from lark_channel.event.callback.model.p2_im_message_receive_v1 import P2ImMessageReceiveV1  # type: ignore
except (ImportError, AttributeError):  # pragma: no cover
    P2ImMessageReceiveV1 = Any  # type: ignore

from ..access_control import (
    AccessDecision,
    AccessOperation,
    IngressAccessPolicy,
    IngressAccessPolicyProvider,
    IngressAccessRequest,
    are_canonical_ingress_facts,
    build_ingress_access_policy,
)
from ..acp.manager import ACPSessionManager
from ..acp.telemetry import build_idle_health_config_for_manager
from ..agent.intent_recognizer import IntentRecognizer
from ..autonomous.provisioning.notification_state import (
    hire_notification_message_uuid,
)
from ..card.actions.dispatch import WORKFLOW_AGENT_SELECTION_ACTIONS
from ..card.ui_text import UI_TEXT
from ..config import IngressAccessMode, get_settings
from ..config.env_file_store import AtomicEnvFileStore
from ..deep_engine import DeepEngineManager, ProgressReporter
from ..mode import PROGRAMMING_MODE_VALUES, PROGRAMMING_MODES
from ..project import (
    MessageLinker,
    MessageProjectMapper,
    ProjectContext,
    ProjectContextManager,
    ProjectManager,
)
from ..spec_engine import SpecEngineManager, SpecReporter
from ..tasking import (
    TaskPriority,
    TaskQueueFullError,
    TaskScheduler,
    TaskSpec,
)
from ..thread import (
    get_current_tenant_key,
    get_current_thread_id,
    get_thread_manager,
    set_current_tenant_key,
)
from ..trust.action_matrix import ActionMatrix, can_dispatch
from ..trust.models import (
    ActionDecision as TrustActionDecision,
)
from ..trust.models import (
    ActionKind,
    ActionRequest,
    ActionTargetKind,
    ActorKind,
    EffectiveTrust,
    TrustZone,
)
from ..trust.registry import ManagedGroupRegistry, single_owner_id
from ..trust.resolver import TrustZoneResolver
from ..utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException
from ..utils.errors import get_error_detail
from ..utils.rate_limit import RateLimiter, RateLimitExceededException
from ..utils.restart_gate import RestartGate
from ..utils.trace import TraceContext, configure_logging_with_trace
from .card_callback_contract import (
    build_card_action_response,
    empty_card_action_response,
    validate_card_action_trigger,
)
from .emoji import EmojiReaction
from .handler_context import HandlerContext
from .handlers import (
    AidenModeHandler,
    ClaudeModeHandler,
    CocoModeHandler,
    CodexModeHandler,
    DeepHandler,
    DiagnosticsHandler,
    DSHModeHandler,
    GeminiModeHandler,
    GrokModeHandler,
    ProjectHandler,
    SpecHandler,
    SystemHandler,
    TraexModeHandler,
    WorkflowHandler,
)
from .image_handler import FeishuImageHandler
from .main_slash_commands import reconcile_main_agent_slash_commands
from .message_cache import MessageCache
from .renderers.deep_renderer import DeepRenderer
from .renderers.spec_renderer import SpecRenderer
from .slash_command_parser import CommandMatch, SlashCommandParser
from .ws_card_action_handler import (
    CardActionInspector,
    _extract_behavior_value,
    bind_managed_trust_revisions,
    classify_card_action_error,
)
from .ws_event_router import MessageIngressGuard, WSErrorAction, classify_ws_error
from .ws_health import WSHealthMonitor
from .ws_lifecycle import ObservedLarkWSClient
from .ws_resource_manager import EngineResourceGroup


def _build_managed_group_registry(
    project_manager: ProjectManager,
) -> ManagedGroupRegistry | None:
    """Compose the one Registry beside the injected Project storage.

    Some unit tests replace ProjectManager with an unconfigured mock.  Those
    compatibility contexts intentionally receive no registry instead of
    writing through a synthesized mock path.
    """

    project_storage = getattr(project_manager, "_storage_path", None)
    if not isinstance(project_storage, Path):
        return None
    return ManagedGroupRegistry(project_storage.parent / "managed-groups.json")


def _configured_managed_group_owner_id(settings: Any) -> str:
    return single_owner_id(getattr(settings, "admin_user_ids", ""))

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("ghostap.audit")
_LARK_MENTION_PLACEHOLDER_RE = re.compile(r"@_user_[0-9]+|@all")
_EMPLOYEE_HANDOFF_GRACE_SECONDS = 0.25
_EMPLOYEE_HANDOFF_MAX_SECONDS = 3.0
_EMPLOYEE_WARNING_RETRY_DELAYS = (0.0, 0.05, 0.2)
_MAIN_BOT_IDENTITY_RETRY_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class _MentionedGroupCommand:
    tenant_key: str
    chat_id: str
    message_id: str
    bot_open_id: str


@dataclass(frozen=True, slots=True)
class _EmployeeTargetedCommand:
    tenant_key: str
    chat_id: str
    message_id: str
    agent_id: str
    bot_principal_id: str
    app_id: str
    bot_open_id: str
    channel_generation: int
    connection_id: str


@dataclass(frozen=True, slots=True)
class _MainBotWarningReplyTransport:
    """Narrow recovered-reply adapter with explicit durable coordinates."""

    handler: Any
    main_app_id: str

    def send_warning(
        self,
        *,
        message_id: str,
        tenant_key: str,
        chat_id: str,
        text: str,
        idempotency_key: str,
    ) -> str:
        from ..autonomous.acceptance.main_bot_warning_outbox import (
            MainBotWarningPermanentDeliveryError,
            MainBotWarningRetryableDeliveryError,
        )
        from .handlers.base import (
            DurableMainBotReplyError,
            DurableMainBotReplyPermanentError,
        )

        try:
            return self.handler.reply_durable_text(
                message_id=message_id,
                tenant_key=tenant_key,
                chat_id=chat_id,
                text=text,
                idempotency_key=idempotency_key,
            )
        except DurableMainBotReplyPermanentError as exc:
            raise MainBotWarningPermanentDeliveryError(exc.error_code) from exc
        except DurableMainBotReplyError as exc:
            raise MainBotWarningRetryableDeliveryError(str(exc)) from exc


class _MessageIngressReservation:
    """Track whether an admitted scheduler callback ever started."""

    __slots__ = (
        "_guard",
        "_lock",
        "_message_id",
        "_owner",
        "_started",
        "_terminal_claimed",
    )

    def __init__(
        self,
        *,
        guard: MessageIngressGuard,
        message_id: str,
        owner: str,
    ) -> None:
        self._guard = guard
        self._message_id = message_id
        self._owner = owner
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._started = False
        self._terminal_claimed = False

    @property
    def message_id(self) -> str:
        return self._message_id

    @property
    def owner(self) -> str:
        return self._owner

    def owns(self) -> bool:
        return self._guard.owns(self._message_id, self._owner)

    def mark_started(self) -> bool:
        with self._lock:
            if self._terminal_claimed:
                return False
            self._started = True
            return True

    def claim_unstarted_terminal(self) -> bool:
        """Claim one terminal-before-start disposition exactly once."""

        with self._lock:
            if self._started or self._terminal_claimed:
                return False
            self._terminal_claimed = True
            return True

    def commit(self) -> bool:
        return self._guard.commit(self._message_id, self._owner)

    def release(self) -> bool:
        return self._guard.release(self._message_id, self._owner)

    def release_if_unstarted(self) -> bool:
        with self._lock:
            if self._started:
                return False
            return self.release()


def _employee_hire_status_text(employee_name: str, status: str) -> str | None:
    if status == "polling":
        return (
            "独立飞书智能体注册请求已提交，正在等待你在上方链接中完成授权确认。"
            "确认前注册接口会持续返回 400 authorization_pending，这是设备授权"
            "流程的正常等待状态；请按链接完成授权，期间请勿重复发送 /hire。"
        )
    if status == "ready":
        return (
            f"✅ 员工 **{employee_name}** 基础配置已就绪，系统将自动完成启用，"
            "无需手动激活；完成后会再次通知。"
        )
    if status == "active":
        return f"✅ 员工 **{employee_name}** 已激活，可以加入 Team 群协作。"
    if status == "action_required":
        return (
            f"⚠️ 员工 **{employee_name}** 创建未能自动收敛，已转为人工处理；"
            f"可使用 `/fire {employee_name}` 清理后重试。"
        )
    return None


def _employee_hire_status_uuid(intent_id: str, status: str) -> str:
    return hire_notification_message_uuid(intent_id, status)


def _employee_warning_uuid(message_id: str, label: str) -> str:
    digest = hashlib.sha256(f"{message_id}\0{label}".encode("utf-8")).hexdigest()
    return "employee-warning-" + digest[:32]


def _ignore_ws_event(_data: object) -> None:
    """Acknowledge subscribed events that require no product action."""


def _unavailable_main_bot_outbound_audit(
    _tenant_key: str,
    _operation: str,
    _target: str,
) -> None:
    raise RuntimeError("main Bot outbound audit is unavailable")


def _main_bot_outbound_wiring(
    runtime: object | None,
    *,
    required: bool,
) -> tuple[
    Callable[[str, str, str], None] | None,
    Callable[[Exception], None] | None,
]:
    """Bind all main-Bot mutations to the same audit used by activation."""

    try:
        audit = getattr(runtime, "main_bot_outbound_audit", None)
    except Exception as exc:
        logger.error(
            "main Bot outbound audit lookup failed: %s",
            type(exc).__name__,
        )
        audit = None
    if audit is not None:
        record_attempt = getattr(audit, "record_attempt", None)
        mark_incomplete = getattr(audit, "mark_incomplete", None)
        if callable(record_attempt) and callable(mark_incomplete):
            return record_attempt, mark_incomplete
    if required:
        return _unavailable_main_bot_outbound_audit, None
    return None, None


def _visible_employee_runtime_requires_outbound_audit(settings: object) -> bool:
    """Fail closed when an injected settings object has an invalid limit shape."""

    limit = getattr(settings, "autonomous_visible_employee_limit", 0)
    if type(limit) is not int:
        logger.error(
            "Invalid autonomous_visible_employee_limit type: %s",
            type(limit).__name__,
        )
        return True
    return limit > 0

_READONLY_CARD_ACTIONS = {
    "deep_expand", "deep_collapse", "deep_mode_full", "deep_mode_compact", "deep_expand_ac", "deep_collapse_ac",
    "spec_expand", "spec_collapse", "spec_mode_full", "spec_mode_compact", "spec_expand_ac", "spec_collapse_ac",
}

_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
_SHUTDOWN_SCHEDULER_DRAIN_S = 5.0
_SHUTDOWN_DELEGATED_DRAIN_S = 5.0
_MAX_PENDING_CARD_ADVISORIES = 128
# The managed-trust cutover first shipped at 2026-07-31 20:25 UTC.  Archives
# created after this boundary may be the startup regression repaired by the
# legacy Team migration; older dissolved markers remain permanently retired.


def _build_task_scheduler(
    settings: Any,
    *,
    project_dir: Path | None = None,
) -> TaskScheduler:
    """Build the production scheduler with a checkout-scoped restart guard."""

    configured_gate_dir = getattr(settings, "restart_gate_dir", "")
    # A number of focused tests and integrations use partial proxy settings
    # objects.  Only the Settings contract's concrete string is meaningful;
    # an auto-created MagicMock attribute must not become a filesystem path.
    if not isinstance(configured_gate_dir, str):
        configured_gate_dir = ""
    gate = RestartGate.for_project(
        project_dir or _CHECKOUT_ROOT,
        override=configured_gate_dir or None,
    )

    def positive_int_setting(name: str, default: int) -> int:
        value = getattr(settings, name, default)
        return value if type(value) is int and value > 0 else default

    def positive_float_setting(name: str, default: float) -> float:
        value = getattr(settings, name, default)
        if type(value) not in {int, float}:
            return default
        normalized = float(value)
        return normalized if 0 < normalized < float("inf") else default

    scheduler = TaskScheduler(
        max_concurrent=settings.task_scheduler_max_concurrent,
        per_key_concurrency=settings.task_scheduler_per_key_concurrency,
        system_concurrency=settings.system_command_concurrency,
        max_pending_normal=positive_int_setting(
            "task_scheduler_max_pending_normal",
            1000,
        ),
        max_pending_system=positive_int_setting(
            "task_scheduler_max_pending_system",
            100,
        ),
        max_terminal_history=positive_int_setting(
            "task_scheduler_max_terminal_history",
            5000,
        ),
        thread_name_prefix="ghost_worker",
        run_guard=gate.task_guard,
        run_guard_timeout_s=positive_float_setting(
            "restart_gate_timeout",
            7200.0,
        ),
    )
    scheduler._restart_gate = gate
    return scheduler


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
        if self._active.get() is not None or self._sealed.get() is not None:
            raise RuntimeError("nested card advisory scope")
        pending: list[Callable[[], object]] = []
        self._active.set(pending)
        return pending

    def defer(self, callback: Callable[[], object]) -> bool:
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
        if self._active.get() is not pending:
            raise RuntimeError("card advisory scope identity mismatch")
        self._active.set(None)
        self._sealed.set(tuple(pending))

    def finish_response(self, *, written: bool) -> None:
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
        if self._active.get() is pending:
            self._active.set(None)
        with self._lock:
            self._pending_count = max(0, self._pending_count - len(pending))

    def close(self) -> None:
        with self._lock:
            self._closed = True


class FeishuWSClient:
    """Feishu WS Client 的服务端运行态。

    该类面向"长连接服务"场景：
    - 内部会初始化 scheduler / handler / cache，并在 `start()` 后进入事件循环。
    - `close()` 提供 best-effort 资源回收（线程/缓存/调度器等）。
    """

    def __init__(
        self,
        message_callback: Callable[[str, str, str, Optional[str]], None],
    ):
        self._employee_department_runtime = None
        self._employee_runtime_init_cleanup_done = False
        try:
            self._initialize(message_callback)
        except BaseException:
            self._close_scheduler_after_initialization_failure()
            self._close_employee_runtime_after_initialization_failure()
            raise

    def _initialize(
        self,
        message_callback: Callable[[str, str, str, Optional[str]], None],
    ) -> None:
        self.settings = get_settings()
        self._ingress_access_policy_provider = IngressAccessPolicyProvider(
            build_ingress_access_policy(self.settings)
        )
        self._ingress_env_store = AtomicEnvFileStore(".env")
        self.message_callback = message_callback
        self._client: Optional[ObservedLarkWSClient] = None
        self._closed = False
        self._message_ingress_binding_cv = threading.Condition(
            threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        )
        self._message_ingress_bindings_inflight = 0
        self._message_ingress_bindings_fenced = False
        self._api_client: Optional[lark.Client] = None
        self._main_bot_open_id = ""
        self._main_bot_identity_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._main_bot_identity_next_retry_at = 0.0
        self._channel_client: Optional[FeishuChannel] = None
        self._slash_command_sync_thread: Optional[threading.Thread] = None
        self._employee_runtime_recovery_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._employee_runtime_recovery_thread: Optional[threading.Thread] = None
        self._employee_runtime_recovery_started = False
        self._employee_runtime_recovery_error: Exception | None = None
        self._main_ws_connected = False
        self._restart_readiness_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._restart_readiness_blocker: str | None = None
        self._channel_client_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # ACPSessionManager: IdleHealth 相关协作者统一通过 IdleHealthConfig 注入，
        # 避免在构造函数中直接依赖具体 Telemetry/Service 实现。
        idle_health_cfg = build_idle_health_config_for_manager()

        manager_timeouts = {
            "coco": self.settings.coco_session_timeout,
            "claude": self.settings.claude_session_timeout,
            "aiden": self.settings.coco_session_timeout,
            "codex": self.settings.coco_session_timeout,
            "gemini": self.settings.coco_session_timeout,
            "traex": self.settings.coco_session_timeout,
            "grok": self.settings.coco_session_timeout,
            "dsh": self.settings.coco_session_timeout,
        }
        acp_managers = {
            name: ACPSessionManager(
                name,
                session_timeout=session_timeout,
                keepalive_interval=self.settings.acp_keepalive_interval,
                idle_healthcheck_s=self.settings.acp_session_idle_healthcheck_s,
                idle_health_config=idle_health_cfg,
            )
            for name, session_timeout in manager_timeouts.items()
        }
        self._intent_recognizer = IntentRecognizer()
        self._message_cache = MessageCache(ttl=self.settings.message_cache_ttl, max_size=self.settings.message_cache_max_size, cleanup_interval=60)
        self._message_ingress_guard = MessageIngressGuard(
            message_cache=self._message_cache,
            message_expire_seconds=self.settings.message_expire_seconds,
        )
        self._card_event_cache = MessageCache(ttl=self.settings.message_cache_ttl, max_size=self.settings.message_cache_max_size, cleanup_interval=60)
        # Card action dedupe (user rapid clicks): short TTL, per-action key.
        self._card_action_dedup_cache = MessageCache(ttl=self.settings.card.action_dedup_ttl, max_size=self.settings.card.action_dedup_max_size, cleanup_interval=30)
        self._card_advisory_dispatcher = _CardAdvisoryDispatcher(
            max_pending=_MAX_PENDING_CARD_ADVISORIES
        )
        # Chat lock gate: initialized after handler_ctx is available (see below).
        self._chat_lock_gate = None  # type: ignore[assignment]
        self._scheduler = _build_task_scheduler(self.settings)
        self._restart_gate = self._scheduler._restart_gate
        self._restart_participation_id: str | None = None
        self._restart_generation: str | None = None
        # Spec Engine limits: e.g. 50 calls per second, max 100 capacity
        self._scheduler.register_policy(
            "spec_command",
            rate_limiter=RateLimiter(capacity=self.settings.spec_rate_limit_capacity, fill_rate=self.settings.spec_rate_limit_fill_rate),
            circuit_breaker=CircuitBreaker(failure_threshold=self.settings.spec_circuit_breaker_threshold, recovery_timeout=self.settings.spec_circuit_breaker_recovery),
        )
        self._WORKING_DIRS_MAX_SIZE = 500
        self._working_dirs: OrderedDict[str, str] = OrderedDict()
        self._working_dir_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        self._project_manager = ProjectManager()
        self._project_manager.on_eviction = self._on_project_evicted
        # Registry replay completes during composition, before any handler or
        # ingress subscriber can observe a managed group.
        self._managed_group_registry = _build_managed_group_registry(
            self._project_manager
        )
        self._managed_group_owner_id = _configured_managed_group_owner_id(
            self.settings
        )
        app_id = getattr(self.settings, "app_id", "")
        self._managed_group_receiving_bot_ref = (
            app_id if isinstance(app_id, str) else ""
        )
        self._message_mapper = MessageProjectMapper()
        self._message_linker = MessageLinker()

        from ..mode import ModeManager

        self._mode_manager = ModeManager()
        # Inject mode_manager into project_manager so LRU eviction
        # automatically cleans up stale _project_modes entries (AC-R01).
        self._project_manager.mode_manager = self._mode_manager
        self._thread_manager = get_thread_manager()
        self._thread_manager._on_evict = self._on_thread_evicted

        self._image_handler: Optional[FeishuImageHandler] = None
        self._pending_image_keys: dict[str, list[str]] = {}
        self._pending_image_only: set[str] = set()  # message_ids that are image-only (no user text)
        self._pending_image_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._enable_streaming = self.settings.streaming_enabled

        self._ws_health_monitor = WSHealthMonitor(self, self.settings)

        self._deep_engine_manager = DeepEngineManager()
        self._progress_reporter = ProgressReporter()
        self._spec_engine_manager = SpecEngineManager()
        self._spec_reporter = SpecReporter()
        from ..autonomous.team.runtime import TeamRuntime
        self._team_runtime = TeamRuntime()

        from ..workflow_engine.manager import WorkflowEngineManager
        self._workflow_engine_manager = WorkflowEngineManager()

        self._context_manager = ProjectContextManager()

        # Initialize lock managers before HandlerContext construction
        _repo_lock_mgr = None
        try:
            from ..repo_lock import get_repo_lock_manager
            _repo_lock_mgr = get_repo_lock_manager()
        except Exception:
            logger.warning("RepoLockManager initialization failed", exc_info=True)

        _chat_lock_mgr = None
        try:
            from ..chat_lock import get_chat_lock_manager
            _chat_lock_mgr = get_chat_lock_manager()
        except Exception:
            logger.warning("ChatLockManager initialization failed", exc_info=True)

        self._employee_department_runtime = None
        try:
            from ..autonomous.gateway.env_scope import (
                local_employee_environment,
            )
            from ..autonomous.provisioning.composition import (
                EmployeeDepartmentRuntime,
            )

            self._employee_department_runtime = EmployeeDepartmentRuntime.from_settings(
                self.settings,
                managed_group_registry=self._managed_group_registry,
                managed_group_owner_id=self._managed_group_owner_id,
                team_runtime=self._team_runtime,
                employee_environment_provider=lambda authority: local_employee_environment(
                    authority,
                    traex_auth_home=getattr(
                        self.settings,
                        "autonomous_employee_traex_auth_home",
                        "~/.trae",
                    ),
                ),
                manager_client_factory=self._get_api_client,
                notification_link=lambda state, url, expire_in: self._reply_employee_hire_message(
                    state,
                    f"请在 {expire_in} 秒内完成独立飞书智能体注册：{url}",
                ),
                notification_status=self._reply_employee_hire_status,
                team_notification=(
                    lambda message_id,
                    chat_id,
                    result,
                    idempotency_key="",
                    tenant_key="",
                    requester_principal_id="": self._reply_employee_team_message(
                        message_id,
                        chat_id,
                        result,
                        tenant_key=tenant_key,
                        requester_principal_id=requester_principal_id,
                        idempotency_key=idempotency_key or None,
                    )
                ),
                recover_immediately=False,
            )
        except Exception as exc:
            logger.error(
                "Employee Department composition failed closed: %s",
                type(exc).__name__,
            )

        main_bot_outbound_audit, main_bot_outbound_audit_failure = (
            _main_bot_outbound_wiring(
                self._employee_department_runtime,
                required=_visible_employee_runtime_requires_outbound_audit(
                    self.settings,
                ),
            )
        )

        # ------------------------------------------------------------------
        # Handler infrastructure
        # ------------------------------------------------------------------
        self._handler_ctx = HandlerContext(
            settings=self.settings,
            api_client_factory=self._get_api_client,
            message_callback=self.message_callback,
            coco_manager=acp_managers["coco"],
            claude_manager=acp_managers["claude"],
            aiden_manager=acp_managers["aiden"],
            codex_manager=acp_managers["codex"],
            gemini_manager=acp_managers["gemini"],
            traex_manager=acp_managers["traex"],
            grok_manager=acp_managers["grok"],
            dsh_manager=acp_managers["dsh"],
            intent_recognizer=self._intent_recognizer,
            scheduler=self._scheduler,
            project_manager=self._project_manager,
            message_mapper=self._message_mapper,
            message_linker=self._message_linker,
            mode_manager=self._mode_manager,
            context_manager=self._context_manager,
            deep_engine_manager=self._deep_engine_manager,
            progress_reporter=self._progress_reporter,
            spec_engine_manager=self._spec_engine_manager,
            spec_reporter=self._spec_reporter,
            workflow_engine_manager=self._workflow_engine_manager,
            thread_manager=self._thread_manager,

            image_handler_factory=self._get_image_handler,
            working_dirs=self._working_dirs,
            working_dir_lock=self._working_dir_lock,
            pending_image_keys=self._pending_image_keys,
            pending_image_lock=self._pending_image_lock,
            enable_streaming=self._enable_streaming,
            repo_lock_manager=_repo_lock_mgr,
            chat_lock_manager=_chat_lock_mgr,
            employee_hire_service=(
                self._employee_department_runtime.hire_service
                if self._employee_department_runtime is not None
                else None
            ),
            employee_fire_service=(
                self._employee_department_runtime.fire_service
                if self._employee_department_runtime is not None
                else None
            ),
            employee_hire_readiness=(
                self._employee_department_runtime.readiness
                if self._employee_department_runtime is not None
                else None
            ),
            employee_membership_service=(
                self._employee_department_runtime.membership_service
                if self._employee_department_runtime is not None
                else None
            ),
            employee_data_composition=(
                self._employee_department_runtime.data_composition
                if self._employee_department_runtime is not None
                else None
            ),
            employee_team_service=(
                self._employee_department_runtime.team_service
                if self._employee_department_runtime is not None
                else None
            ),
            employee_runtime_facade=self._employee_department_runtime,
            main_bot_outbound_audit=main_bot_outbound_audit,
            main_bot_outbound_audit_failure=main_bot_outbound_audit_failure,
            tenant_key_resolver=get_current_tenant_key,
            channel_client_factory=self._get_channel_client,
            ingress_access_policy_provider=(
                self._ingress_access_policy_provider
            ),
            ingress_env_store=self._ingress_env_store,
            managed_group_registry=self._managed_group_registry,
            managed_group_owner_id=self._managed_group_owner_id,
            managed_group_receiving_bot_ref=(
                self._managed_group_receiving_bot_ref
            ),
            managed_group_bot_rotation=self.rotate_main_managed_group_bot,
        )

        handler_types = {
            "coco": CocoModeHandler,
            "claude": ClaudeModeHandler,
            "aiden": AidenModeHandler,
            "codex": CodexModeHandler,
            "gemini": GeminiModeHandler,
            "traex": TraexModeHandler,
            "grok": GrokModeHandler,
            "dsh": DSHModeHandler,
            "deep": DeepHandler,
            "spec": SpecHandler,
            "project": ProjectHandler,
            "system": SystemHandler,
            "diagnostics": DiagnosticsHandler,
            "workflow": WorkflowHandler,
        }
        handlers = {name: cls(self._handler_ctx) for name, cls in handler_types.items()}
        handlers["deep"].renderer = DeepRenderer(handlers["deep"])
        handlers["spec"].renderer = SpecRenderer(handlers["spec"])
        self._handler_ctx.managers.update(acp_managers)
        self._handler_ctx.handlers.update(handlers)
        if self._employee_department_runtime is not None:
            bind_warning_transport = getattr(
                self._employee_department_runtime,
                "bind_main_bot_warning_transport",
                None,
            )
            if not callable(bind_warning_transport):
                raise RuntimeError("main Bot durable warning transport is unavailable")
            bind_warning_transport(
                _MainBotWarningReplyTransport(
                    handler=handlers["coco"],
                    main_app_id=self.settings.app_id,
                )
            )
        system_handler = handlers["system"]

        # Subscribe to hard-timeout reclaim events on RepoLockManager
        # (fire-and-forget notification to the displaced lock holder chat).
        repo_lock_mgr = self._handler_ctx.repo_lock_manager
        if repo_lock_mgr is not None:
            _send_card = system_handler.send_card_to_chat  # narrow reference

            def _notify_hard_timeout_reclaim(root_path: str, holder_chat_id: str) -> None:
                try:
                    from pathlib import Path as _Path

                    from ..card.builders.lock import build_lock_reclaim_notify_card
                    repo_name = _Path(root_path).name or root_path
                    _send_card(
                        holder_chat_id,
                        build_lock_reclaim_notify_card(
                            repo_name, reason="hard_timeout",
                            hard_timeout_seconds=getattr(self.settings, "repo_lock_hard_timeout", None),
                        ),
                    )
                except Exception as _exc:
                    logger.warning(
                        "Failed to notify hard-timeout reclaim to chat=%s: %s",
                        holder_chat_id[:12], _exc,
                    )

            repo_lock_mgr.on_reclaim.subscribe(_notify_hard_timeout_reclaim)

            # Subscribe to lock release events — notify previously blocked chats.
            def _notify_lock_released(root_path: str, blocked_chat_ids: set) -> None:
                try:
                    import json as _json
                    from pathlib import Path as _Path

                    from ..card.builders.lock_common import _compute_command_sig
                    from ..card.ui_text import UI_TEXT
                    repo_name = _Path(root_path).name or root_path
                    _text = UI_TEXT["repo_lock_released_notify"].format(repo_name=repo_name)
                    _btn = {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "📊 查看状态"},
                        "type": "default",
                        "value": {"action": "retry_command", "_t": "/status", "_s": _compute_command_sig("/status")},
                    }
                    _card = _json.dumps({
                        "schema": "2.0",
                        "config": {"wide_screen_mode": True},
                        "body": {"elements": [
                            {"tag": "markdown", "content": _text},
                            {
                                "tag": "column_set",
                                "flex_mode": "none",
                                "background_style": "default",
                                "columns": [
                                    {
                                        "tag": "column",
                                        "width": "weighted",
                                        "weight": 1,
                                        "elements": [_btn],
                                    }
                                ],
                            },
                        ]},
                    }, ensure_ascii=False)
                    for _cid in blocked_chat_ids:
                        try:
                            _send_card(_cid, _card)
                        except Exception as _inner:
                            logger.debug("Failed to notify release to chat=%s: %s", _cid[:12], _inner)
                except Exception as _exc:
                    logger.warning("Failed to send lock release notifications: %s", _exc)

            repo_lock_mgr.on_release.subscribe(_notify_lock_released)

        # Initialize ChatLockGate (ingress-level chat-lock interception).
        from .chat_lock_gate import ChatLockGate
        _clm = getattr(self._handler_ctx, "chat_lock_manager", None)
        _lock_dedup = MessageCache(ttl=30, max_size=10_000, cleanup_interval=60)
        self._chat_lock_gate = ChatLockGate(
            _clm, _lock_dedup, handler=system_handler,
        )

        # ------------------------------------------------------------------
        # Control-plane (deferred /exit, system command gate)
        # ------------------------------------------------------------------
        from .control_plane import ControlPlane
        self._control_plane = ControlPlane(
            scheduler=self._scheduler,
            project_manager=self._project_manager,
            exit_handler_fn=system_handler.exit_current_mode,
        )
        # --- Message Dispatcher ---
        from .dispatcher import MessageDispatcher
        self._message_dispatcher = MessageDispatcher(self)

        # --- Action Dispatcher ---
        from .action_registry import init_action_registry
        self._action_handlers = init_action_registry(self)

        # Configure trace logging
        configure_logging_with_trace()
        self._register_scheduler_listeners()

    def _recover_employee_runtime_after_handler_binding(
        self,
        *,
        close_on_failure: bool = True,
    ) -> object | None:
        """Start durable recovery only after the main-Bot reply path exists."""

        runtime = self._employee_department_runtime
        if runtime is None:
            return None
        try:
            if "coco" not in self._handler_ctx.handlers:
                raise RuntimeError("main Bot reply transport is not bound")
            return runtime.recover()
        except Exception:
            if close_on_failure:
                self._close_employee_runtime_after_initialization_failure()
            raise

    def _run_employee_runtime_recovery(self) -> None:
        """Recover Employee state after main WS readiness without weakening admission."""

        try:
            for recovery_attempt in range(2):
                try:
                    recovery_summary = (
                        self._recover_employee_runtime_after_handler_binding(
                            close_on_failure=False
                        )
                    )
                    break
                except Exception:
                    if recovery_attempt:
                        raise
                    logger.warning(
                        "Employee runtime background recovery failed once; "
                        "retrying once",
                        exc_info=True,
                    )
        except Exception as exc:
            self._employee_runtime_recovery_error = exc
            runtime = self._employee_department_runtime
            fail_recovery = getattr(runtime, "fail_recovery", None)
            if callable(fail_recovery):
                try:
                    fail_recovery("background_recovery")
                except Exception:
                    logger.exception(
                        "Employee runtime failed to close admission after recovery error"
                    )
            logger.exception(
                "Employee runtime background recovery failed; "
                "main Bot remains available and Employee admission stays closed"
            )
            return
        logger.info(
            "Employee runtime background recovery complete "
            "eligible=%d recovered=%d skipped=%d failed=%d",
            int(getattr(recovery_summary, "eligible", 0) or 0),
            int(getattr(recovery_summary, "recovered", 0) or 0),
            int(getattr(recovery_summary, "skipped", 0) or 0),
            int(getattr(recovery_summary, "failed", 0) or 0),
        )
        self._try_publish_restart_readiness()

    def _start_employee_runtime_recovery(self) -> None:
        """Start at most one post-connection Employee recovery worker."""

        if self._employee_department_runtime is None:
            return
        with self._employee_runtime_recovery_lock:
            if self._closed or self._employee_runtime_recovery_started:
                return
            self._employee_runtime_recovery_started = True
            thread = threading.Thread(
                target=self._run_employee_runtime_recovery,
                name="employee-runtime-recovery",
                daemon=True,
            )
            self._employee_runtime_recovery_thread = thread
            thread.start()

    def _wait_for_employee_runtime_recovery(self, timeout: float) -> bool:
        """Wait for background recovery before closing its durable resources."""

        lock = getattr(self, "_employee_runtime_recovery_lock", None)
        if lock is None:
            thread = getattr(self, "_employee_runtime_recovery_thread", None)
        else:
            with lock:
                thread = getattr(self, "_employee_runtime_recovery_thread", None)
        if thread is None or thread is threading.current_thread():
            return True
        if not thread.is_alive():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _register_scheduler_listeners(self) -> None:
        """Attach scheduler-owned callbacks only after construction succeeds."""

        self._scheduler.add_listener(self._control_plane.on_scheduler_event)

    def _detach_scheduler_listeners(self) -> None:
        """Release scheduler references after its callbacks have quiesced."""

        scheduler = getattr(self, "_scheduler", None)
        remove_listener = getattr(scheduler, "remove_listener", None)
        if not callable(remove_listener):
            return

        control_plane = getattr(self, "_control_plane", None)
        control_listener = getattr(control_plane, "on_scheduler_event", None)
        if not callable(control_listener):
            return
        try:
            remove_listener(control_listener)
        except Exception:
            logger.debug(
                "failed to detach scheduler listener",
                exc_info=True,
            )

    def _close_scheduler_after_initialization_failure(self) -> None:
        """Stop a partially constructed scheduler without masking the cause."""

        control_plane = getattr(self, "_control_plane", None)
        stop_control_plane = getattr(control_plane, "stop", None)
        if callable(stop_control_plane):
            try:
                stop_control_plane()
            except BaseException:
                logger.error(
                    "ControlPlane cleanup failed after initialization error",
                    exc_info=True,
                )

        try:
            self._detach_scheduler_listeners()
        except BaseException:
            logger.error(
                "scheduler listener cleanup failed after initialization error",
                exc_info=True,
            )

        scheduler = getattr(self, "_scheduler", None)
        stop_scheduler = getattr(scheduler, "stop", None)
        if callable(stop_scheduler):
            try:
                stop_scheduler(wait=True, shutdown_executor=True)
            except BaseException:
                logger.error(
                    "scheduler cleanup failed after initialization error",
                    exc_info=True,
                )

    def _close_employee_runtime_after_initialization_failure(self) -> None:
        """Close a composed runtime once without masking the init failure."""

        runtime = getattr(self, "_employee_department_runtime", None)
        if runtime is None or getattr(
            self,
            "_employee_runtime_init_cleanup_done",
            False,
        ):
            return
        self._employee_runtime_init_cleanup_done = True
        try:
            runtime.close()
        except BaseException:
            logger.error(
                "Employee Department cleanup failed after initialization error",
                exc_info=True,
            )

    def _begin_message_ingress_binding(self) -> bool:
        """Join the handler-to-completion-binding shutdown barrier."""

        with self._message_ingress_binding_cv:
            if self._message_ingress_bindings_fenced:
                return False
            self._message_ingress_bindings_inflight += 1
            return True

    def _finish_message_ingress_binding(self) -> None:
        with self._message_ingress_binding_cv:
            if self._message_ingress_bindings_inflight <= 0:
                raise RuntimeError("message ingress binding barrier underflow")
            self._message_ingress_bindings_inflight -= 1
            if self._message_ingress_bindings_inflight == 0:
                self._message_ingress_binding_cv.notify_all()

    def _fence_message_ingress_bindings(self) -> None:
        with self._message_ingress_binding_cv:
            self._message_ingress_bindings_fenced = True
            self._message_ingress_binding_cv.notify_all()

    def _wait_for_message_ingress_bindings(self, *, deadline: float) -> bool:
        """Wait for already-dispatched handlers to bind terminal callbacks."""

        with self._message_ingress_binding_cv:
            while self._message_ingress_bindings_inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._message_ingress_binding_cv.wait(timeout=remaining)
            return True

    def close(self) -> bool:
        """Fence intake, drain work, then clean up dependencies.

        ``False`` means a worker or card delivery did not drain within the
        bounded shutdown budget. In that case shared dependencies are left
        intact for the still-running callback and the process-level caller must
        not tear down lock managers underneath it.
        """
        self._closed = True
        self._fence_message_ingress_bindings()
        dispatcher = getattr(self, "_card_advisory_dispatcher", None)
        if dispatcher is not None:
            dispatcher.close()

        self._ws_health_monitor.stop_watchdog()

        # Stop external intake before fencing the in-process scheduler.  This
        # prevents a reconnect callback from admitting fresh work during drain.
        try:
            self._ws_health_monitor.disconnect()
        except Exception:
            logger.debug("failed to disconnect primary WS intake", exc_info=True)
        try:
            if self._channel_client is not None:
                self._channel_client.stop()
        except Exception as e:
            logger.debug("停止普通模式 Channel SDK 客户端失败: %s", get_error_detail(e))

        try:
            self._scheduler.fence_admission()
        except Exception:
            logger.debug("failed to fence scheduler admission", exc_info=True)

        try:
            self._control_plane.stop()
        except Exception:
            logger.debug("failed to stop control_plane", exc_info=True)

        scheduler_drain_deadline = (
            time.monotonic() + _SHUTDOWN_SCHEDULER_DRAIN_S
        )
        scheduler_idle = False
        try:
            scheduler_idle = self._scheduler.wait_for_idle(
                max(0.0, scheduler_drain_deadline - time.monotonic())
            )
            if not scheduler_idle:
                self._scheduler.cancel_active()
                scheduler_idle = self._scheduler.wait_for_idle(
                    max(0.0, scheduler_drain_deadline - time.monotonic())
                )
        except Exception:
            logger.debug("failed to drain scheduler before cleanup", exc_info=True)
            scheduler_idle = False

        if not scheduler_idle:
            logger.error(
                "scheduler did not reach idle; preserving callback dependencies"
            )
            try:
                self._scheduler.stop(wait=True, shutdown_executor=False)
            except Exception:
                logger.debug("failed to stop scheduler dispatcher", exc_info=True)
            return False

        ingress_bindings_idle = False
        try:
            ingress_bindings_idle = self._wait_for_message_ingress_bindings(
                deadline=scheduler_drain_deadline
            )
        except Exception:
            logger.debug(
                "failed to drain message ingress bindings before cleanup",
                exc_info=True,
            )
        if not ingress_bindings_idle:
            logger.error(
                "message ingress completion bindings did not drain; "
                "preserving callback dependencies"
            )
            try:
                self._scheduler.stop(wait=True, shutdown_executor=False)
            except Exception:
                logger.debug("failed to stop scheduler dispatcher", exc_info=True)
            return False

        completion_callbacks_idle = False
        try:
            completion_callbacks_idle = self._scheduler.wait_for_completion_callbacks(
                timeout=max(
                    0.0,
                    scheduler_drain_deadline - time.monotonic(),
                )
            )
        except Exception:
            logger.debug(
                "failed to drain scheduler completion callbacks before cleanup",
                exc_info=True,
            )
        if not completion_callbacks_idle:
            logger.error(
                "scheduler completion callbacks did not drain; preserving callback dependencies"
            )
            try:
                self._scheduler.stop(wait=True, shutdown_executor=False)
            except Exception:
                logger.debug("failed to stop scheduler dispatcher", exc_info=True)
            return False

        self._detach_scheduler_listeners()

        # Ask every delegated execution surface to stop before destroying ACP
        # sessions.  Employee runtime close waits for its Team executor; Team
        # tasks are delegated to a separate bounded executor, so explicitly
        # pause their sessions and observe that executor as well.
        deep_resources = EngineResourceGroup("deep_engine", self._deep_engine_manager)
        spec_resources = EngineResourceGroup("spec_engine", self._spec_engine_manager)
        workflow_resources = EngineResourceGroup(
            "workflow_engine",
            self._workflow_engine_manager,
        )
        deep_engines = deep_resources.stop_running_engines()
        spec_engines = spec_resources.stop_running_engines()
        workflow_engines = workflow_resources.stop_running_engines()

        if not self._wait_for_employee_runtime_recovery(
            _SHUTDOWN_DELEGATED_DRAIN_S
        ):
            logger.error(
                "Employee recovery worker did not drain; preserving durable resources"
            )
            try:
                self._scheduler.stop(wait=True, shutdown_executor=False)
            except Exception:
                logger.debug("failed to stop scheduler dispatcher", exc_info=True)
            return False

        delegated_idle = True
        try:
            if self._employee_department_runtime is not None:
                self._employee_department_runtime.close()
        except Exception:
            logger.debug("Employee Department shutdown skipped", exc_info=True)
            delegated_idle = False

        delegated_idle = (
            EngineResourceGroup.wait_stopped(deep_engines)
            and delegated_idle
        )
        delegated_idle = (
            EngineResourceGroup.wait_stopped(spec_engines)
            and delegated_idle
        )
        delegated_idle = (
            EngineResourceGroup.wait_stopped(workflow_engines)
            and delegated_idle
        )
        if not delegated_idle:
            logger.error(
                "delegated engine work did not drain; preserving shared dependencies"
            )
            try:
                self._scheduler.stop(wait=True, shutdown_executor=False)
            except Exception:
                logger.debug("failed to stop scheduler dispatcher", exc_info=True)
            return False

        from ..card.delivery.registry import delivery_registry

        # ``shutdown_all`` fences, drains, and shuts down every registered
        # delivery under one shared deadline.  Calling ``drain_in_flight``
        # first would give the same delivery two independent timeout budgets
        # and leave its eviction worker alive after a successful drain.
        if not delivery_registry.shutdown_all(
            timeout=_SHUTDOWN_DELEGATED_DRAIN_S
        ):
            logger.error(
                "card delivery did not shut down; preserving shared dependencies"
            )
            try:
                self._scheduler.stop(wait=True, shutdown_executor=False)
            except Exception:
                logger.debug("failed to stop scheduler dispatcher", exc_info=True)
            return False

        try:
            self._message_cache.stop_cleanup_thread()
        except Exception as e:
            logger.debug("停止message_cache清理线程失败: %s", get_error_detail(e))

        try:
            self._card_event_cache.stop_cleanup_thread()
        except Exception as e:
            logger.debug("停止card_event_cache清理线程失败: %s", get_error_detail(e))

        try:
            self._card_action_dedup_cache.stop_cleanup_thread()
        except Exception as e:
            logger.debug("停止card_action_dedup_cache清理线程失败: %s", get_error_detail(e))

        deep_resources.cleanup_all()
        spec_resources.cleanup_all()
        workflow_resources.cleanup_all()

        # Only after execution surfaces have quiesced may their shared ACP
        # sessions be destroyed.
        for name, mgr in self._handler_ctx.managers.items():
            try:
                mgr.cleanup_all()
            except Exception as e:
                logger.debug("清理%s_session_manager失败: %s", name, get_error_detail(e))

        try:
            self._thread_manager.close()
        except Exception as e:
            logger.debug("清理thread_manager失败: %s", get_error_detail(e))

        try:
            self._scheduler.stop(wait=True, shutdown_executor=True)
        except Exception as e:
            logger.debug("停止scheduler失败: %s", get_error_detail(e))

        # Stop chat-lock dedup only after callbacks can no longer use it.
        try:
            self._chat_lock_gate.close()
        except Exception:
            logger.debug("failed to close chat_lock_gate", exc_info=True)

        # Best-effort shutdown lock-manager daemon threads so non-Application
        # callers (e.g. tests) do not leak background threads.
        try:
            from ..chat_lock import shutdown_if_active as _chat_sd
            _chat_sd()
        except Exception:
            logger.debug("ChatLockManager shutdown in close() skipped", exc_info=True)
        try:
            from ..repo_lock import shutdown_if_active as _repo_sd
            _repo_sd()
        except Exception:
            logger.debug("RepoLockManager shutdown in close() skipped", exc_info=True)
        return True



    def _on_thread_evicted(self, ctx) -> None:
        for mgr in self._handler_ctx.managers.values():
            try:
                mgr.end_session(ctx.chat_id, project_id=ctx.project_id, thread_id=ctx.thread_root_id)
            except Exception:
                logger.debug("failed to end ACP session during cleanup", exc_info=True)

    def _on_project_evicted(self, evicted_chat_id: str, project_name: str, project_id: str) -> None:
        """Notify a chat that its project binding was evicted due to LRU capacity.

        Convergence: cleans up ACP sessions for the evicted project, then sends
        a rebind notification card.  Both run in a daemon thread to avoid blocking
        ProjectManager's critical section (which holds _lock when calling
        this callback).
        """
        def _send_notification():
            # Phase 1: Clean up ACP sessions for the evicted project.
            for mgr in self._handler_ctx.managers.values():
                try:
                    mgr.end_session(evicted_chat_id, project_id=project_id)
                except Exception:
                    logger.debug("failed to end session for evicted project", exc_info=True)

            # Phase 2: Send rebind notification card.
            try:
                from ..card.builders.project import ProjectBuilder
                content = UI_TEXT["eviction_notify_body"].format(name=project_name)
                buttons = [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": UI_TEXT["eviction_notify_btn_rebind"]},
                        "type": "primary",
                        "value": {"action": "show_board"},
                    }
                ]
                msg_type, card_json = ProjectBuilder.build_project_response_card(
                    project=None,
                    title=UI_TEXT["eviction_notify_title"],
                    content=content,
                    show_buttons=False,
                    extra_buttons=buttons,
                )
                if msg_type == "text":
                    self._handler_ctx.handlers["coco"].send_text_to_chat(
                        evicted_chat_id, card_json
                    )
                else:
                    self._handler_ctx.handlers["coco"].send_card_to_chat(
                        evicted_chat_id, card_json
                    )
            except Exception as send_err:
                # Fallback to plain text
                try:
                    msg = UI_TEXT["ws_project_eviction_notify"].format(name=project_name)
                    self._handler_ctx.handlers["coco"].send_text_to_chat(
                        evicted_chat_id, msg
                    )
                except Exception:
                    logger.debug("failed to send eviction fallback notification", exc_info=True)
                logger.warning("Failed to send LRU eviction notification to %s: %s", evicted_chat_id[:12], send_err)

        threading.Thread(target=_send_notification, daemon=True).start()

    def _get_api_client(self) -> lark.Client:
        """延迟构造 `lark_oapi.Client`（用于调用消息/卡片 API）。"""
        if self._api_client is None:
            self._api_client = (
                lark.Client.builder()
                .app_id(self.settings.app_id)
                .app_secret(self.settings.app_secret)
                .log_level(lark.LogLevel.INFO)
                .timeout(30)  # 30s timeout for all API calls (card delivery protection)
                .build()
            )
        return self._api_client

    def _get_channel_client(self) -> FeishuChannel:
        """Return the process-shared Channel SDK client for programming cards.

        Inbound events use the dedicated raw Channel WebSocket client.  This
        capability instance is outbound-only, so webhook transport prevents
        accidental creation of a second WebSocket connection while retaining
        the SDK's async CardKit and message APIs.
        """
        with self._channel_client_lock:
            if self._channel_client is None:
                timeout = float(self.settings.card.delivery_api_timeout)
                self._channel_client = FeishuChannel(
                    app_id=self.settings.app_id,
                    app_secret=self.settings.app_secret,
                    log_level=ChannelLogLevel.WARNING,
                    transport=ChannelTransportConfig(
                        kind="webhook",
                        http_timeout_seconds=timeout,
                    ),
                )
            return self._channel_client

    def _sync_main_slash_commands(self) -> None:
        """Best-effort convergence of the main Bot's Slash discovery panel."""

        self._sync_main_bot_identity()
        try:
            verified = asyncio.run(
                reconcile_main_agent_slash_commands(self._get_api_client())
            )
        except Exception as exc:
            logger.warning(
                "Main Agent Slash Command sync skipped (%s); grant and publish "
                "application:app_slash_command:read and "
                "application:app_slash_command:write",
                type(exc).__name__,
            )
            return

        logger.info(
            "Main Agent Slash Commands ready: total=%d created=%d updated=%d "
            "deleted=%d",
            len(verified.observed),
            len(verified.created),
            len(verified.updated),
            len(verified.deleted),
        )

    def _sync_main_bot_identity(self) -> str:
        """Resolve the main app's bot Open ID outside the WS ACK path."""

        from lark_oapi.core.enum import AccessTokenType, HttpMethod
        from lark_oapi.core.model.base_request import BaseRequest
        from lark_oapi.core.model.base_response import BaseResponse

        with self._main_bot_identity_lock:
            if self._main_bot_open_id:
                return self._main_bot_open_id
            now = time.monotonic()
            if now < self._main_bot_identity_next_retry_at:
                return ""
            request = (
                BaseRequest.builder()
                .http_method(HttpMethod.GET)
                .uri("/open-apis/bot/v3/info")
                .token_types({AccessTokenType.TENANT})
                .paths({})
                .body(None)
                .build()
            )
            try:
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
            except Exception:
                self._main_bot_identity_next_retry_at = (
                    time.monotonic() + _MAIN_BOT_IDENTITY_RETRY_SECONDS
                )
                logger.warning("Main Bot identity lookup failed closed", exc_info=True)
                return ""
            self._main_bot_open_id = open_id
            self._main_bot_identity_next_retry_at = 0.0
            return open_id

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



    def _get_image_handler(self) -> FeishuImageHandler:
        """获取/创建图片处理器（解析 + 下载 + 生成引用文本）。"""
        if self._image_handler is None:
            self._image_handler = FeishuImageHandler(self._get_api_client, self.settings)
        return self._image_handler

    # ==================================================================
    # Core routing — these remain in ws_client.py
    # ==================================================================

    def _resolve_project_from_message(
        self, message_id: str, chat_id: str, parent_id: Optional[str] = None
    ) -> tuple[Optional[ProjectContext], Optional[str]]:
        """根据消息引用（parent/root）解析项目上下文。

        返回：
        - `project`: 最终解析到的 ProjectContext（或当前 active project）。
        - `auto_enter_mode`: 若该消息是回复某个编程会话/项目卡片，允许自动进入对应编程模式。
        """
        auto_enter_mode = None

        if parent_id:
            project_id = self._message_mapper.get_project_id(parent_id)
            if not isinstance(project_id, str):
                project_id = None
            if project_id:
                project = self._project_manager.get_project_for_chat(project_id, chat_id)
                if project:
                    self._project_manager.set_active_project(chat_id, project_id)
                    logger.info("通过消息引用切换到项目: %s", project.project_name)

                    # Resolve mode from ModeManager (single source of truth).
                    _proj_mode = self._mode_manager.get_mode(chat_id, project_id=project_id)
                    if _proj_mode in PROGRAMMING_MODES:
                        auto_enter_mode = _proj_mode.value

                    if auto_enter_mode:
                        logger.info("自动进入 %s 模式 (回复编程消息)", auto_enter_mode)

                    return project, auto_enter_mode

        bound_project = self._project_manager.find_by_bound_chat_id(chat_id)
        if bound_project is not None:
            return bound_project, None

        return self._project_manager.get_active_project(chat_id), None

    @staticmethod
    def _access_identifier_hash(value: str) -> str:
        return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]

    def _current_ingress_access_policy(self) -> IngressAccessPolicy:
        provider = getattr(self, "_ingress_access_policy_provider", None)
        if isinstance(provider, IngressAccessPolicyProvider):
            return provider.current
        # Direct unit construction historically bypasses _initialize. Keep that
        # path secure by deriving a fresh immutable snapshot from its settings.
        return build_ingress_access_policy(self.settings)

    def _decide_ingress_access(
        self,
        *,
        message_id: str,
        sender_id: str,
        chat_id: str,
        chat_type: str,
        command_match: CommandMatch | None,
        policy_snapshot: IngressAccessPolicy | None = None,
    ) -> AccessDecision:
        try:
            policy = (
                policy_snapshot
                if policy_snapshot is not None
                else self._current_ingress_access_policy()
            )
            decision = policy.decide(
                IngressAccessRequest(
                    message_id=message_id,
                    sender_id=sender_id,
                    chat_id=chat_id,
                    chat_type=chat_type,
                    command_match=command_match,
                )
            )
        except Exception:
            logger.critical(
                "INGRESS_ACCESS_POLICY_EVALUATION_FAILED",
                exc_info=True,
            )
            policy = IngressAccessPolicy(
                admin_ids=frozenset(),
                allowed_user_ids=frozenset(),
                allowed_chat_ids=frozenset(),
                mode=IngressAccessMode.ENFORCED,
                admin_bootstrap_scope="p2p_only",
            )
            decision = AccessDecision(
                allowed=False,
                operation=AccessOperation.NORMAL_MESSAGE,
                reason_code="access_policy_error",
                prospective_allowed=False,
            )

        if not decision.allowed or policy.mode.value in {
            "shadow",
            "legacy_allow_all",
        }:
            audit_logger.warning(
                "INGRESS_ACCESS_DECISION mode=%s operation=%s reason=%s "
                "sender_hash=%s chat_hash=%s prospective_allowed=%s",
                policy.mode.value,
                decision.operation.value,
                decision.reason_code,
                self._access_identifier_hash(sender_id),
                self._access_identifier_hash(chat_id),
                decision.prospective_allowed,
            )
        return decision

    def _resolve_effective_trust(
        self,
        *,
        sender_id: str,
        chat_id: str,
        chat_type: str,
        message_id: str = "",
    ) -> EffectiveTrust | None:
        """Resolve current Registry trust, or ``None`` for legacy test wiring.

        A real Registry is authoritative.  Runtime lookup failure only removes
        Employee actors; it can never turn an unknown sender into an Owner.
        """

        registry = getattr(self, "_managed_group_registry", None)
        if type(registry) is not ManagedGroupRegistry:
            return None
        employee_bot_ids: frozenset[str] = frozenset()
        runtime = getattr(self, "_employee_department_runtime", None)
        employee_ids = getattr(runtime, "trusted_employee_bot_open_ids", None)
        if callable(employee_ids):
            try:
                resolved = employee_ids()
                if isinstance(resolved, (tuple, frozenset)) and all(
                    isinstance(value, str) and value.startswith("ou_")
                    for value in resolved
                ):
                    employee_bot_ids = frozenset(resolved)
            except Exception:
                logger.warning(
                    "managed ingress Employee identity snapshot unavailable",
                    exc_info=True,
                )
        try:
            groups = registry.managed_groups()
            group = next(
                (record for record in groups if record.chat_id == chat_id),
                None,
            )
            if group is not None:
                current_group, current_grant = registry.trust_snapshot(chat_id)
                groups = (() if current_group is None else (current_group,))
                grants = (() if current_grant is None else (current_grant,))
            else:
                grants = ()
            resolver = TrustZoneResolver(
                owner_id=getattr(self, "_managed_group_owner_id", ""),
                managed_groups=groups,
                project_grants=grants,
                employee_bot_ids=employee_bot_ids,
            )
        except Exception:
            logger.critical(
                "MANAGED_INGRESS_REGISTRY_SNAPSHOT_FAILED",
                exc_info=True,
            )
            resolver = TrustZoneResolver(
                owner_id=getattr(self, "_managed_group_owner_id", ""),
                managed_groups=(),
                project_grants=(),
                employee_bot_ids=frozenset(),
            )
        trust = resolver.resolve(
            sender_id=sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
        )
        if trust.actor is ActorKind.EMPLOYEE:
            validator = getattr(runtime, "is_valid_employee_continuation", None)
            if not message_id or not callable(validator):
                return resolver.resolve(
                    sender_id="",
                    chat_id=chat_id,
                    chat_type=chat_type,
                )
            try:
                valid = validator(
                    sender_open_id=sender_id,
                    chat_id=chat_id,
                    message_id=message_id,
                )
            except Exception:
                valid = False
            if valid is not True:
                return resolver.resolve(
                    sender_id="",
                    chat_id=chat_id,
                    chat_type=chat_type,
                )
        return trust

    @staticmethod
    def _managed_trust_access_decision(
        trust: EffectiveTrust | None,
    ) -> AccessDecision | None:
        """Return an authoritative decision for group trust zones.

        Owner P2P and clients without the durable Registry retain the existing
        access-policy path.  Managed actors bypass legacy enrollment; unknown
        group actors are denied before content parsing.
        """

        if trust is None:
            return None
        if trust.zone is TrustZone.OWNER_P2P:
            return AccessDecision(
                allowed=True,
                operation=AccessOperation.NORMAL_MESSAGE,
                reason_code="owner_p2p_trust",
                prospective_allowed=True,
                effective_trust=trust,
            )
        allowed = (
            trust.zone is TrustZone.MANAGED_AGENT_GROUP
            and trust.actor in {ActorKind.OWNER, ActorKind.EMPLOYEE}
        )
        return AccessDecision(
            allowed=allowed,
            operation=AccessOperation.NORMAL_MESSAGE,
            reason_code=("managed_trust" if allowed else "external_or_unknown"),
            prospective_allowed=allowed,
            effective_trust=trust,
        )

    def _managed_ingress_action_allowed(
        self,
        trust: EffectiveTrust | None,
        *,
        text: str,
        command_match: CommandMatch | None,
    ) -> bool:
        if trust is None:
            return True
        action: ActionKind | None = None
        if self._is_known_host_shell_invocation(text):
            action = ActionKind.HOST_SHELL
        elif command_match is not None and command_match.command in {
            "/access",
            "/setadmin",
            "/hire",
            "/fire",
            "/employee-role",
        }:
            action = ActionKind.GRANT_ADMIN
        elif command_match is not None and command_match.command in {
            "/employees",
            "/history",
            "/employee-memory",
        }:
            action = ActionKind.SYSTEM_ADMIN
        if action is None:
            return trust.zone is not TrustZone.EXTERNAL_OR_UNKNOWN_GROUP
        return ActionMatrix().decide(
            ActionRequest(
                trust=trust,
                action=action,
                target=ActionTargetKind.HOST_GLOBAL,
            )
        ) is TrustActionDecision.ALLOW

    @staticmethod
    def _is_known_host_shell_invocation(text: str) -> bool:
        """Recognize only deterministic shell forms at the intake fence.

        The broader SMART heuristic intentionally treats unknown English verbs
        as possible executables.  It is useful for routing but is not an
        authorization fact, so managed ingress only rejects the explicit
        command whitelist and local executable paths here.  The Dispatcher
        applies the ActionMatrix again when intent recognition confirms Shell.
        """

        from ..agent.intent_recognizer import IntentRecognizer

        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        first_word = normalized.split()[0]
        return (
            first_word == "cd"
            or first_word in IntentRecognizer.SHELL_COMMANDS
            or IntentRecognizer._looks_like_local_executable_path(first_word)
        )

    def _current_trust_can_dispatch(
        self,
        trust: EffectiveTrust | None,
        *,
        project=None,
    ) -> bool:
        if trust is None:
            return True
        current_group_revision = None
        current_grant_revision = None
        if trust.managed_group is not None:
            registry = getattr(self, "_managed_group_registry", None)
            if type(registry) is not ManagedGroupRegistry:
                return False
            current_group, current_grant = registry.trust_snapshot(
                trust.managed_group.chat_id
            )
            current_group_revision = (
                current_group.revision if current_group is not None else None
            )
            current_grant_revision = (
                current_grant.revision if current_grant is not None else None
            )
            if (
                project is not None
                and (
                    current_group is None
                    or getattr(project, "project_id", None) != current_group.project_id
                    or getattr(project, "root_path", None)
                    != current_group.canonical_root_ref
                )
            ):
                return False
        return can_dispatch(
            trust,
            current_group_revision=current_group_revision,
            current_grant_revision=current_grant_revision,
            killed=False,
            paused=False,
        ) is TrustActionDecision.ALLOW

    @staticmethod
    def _extract_canonical_ingress_facts(
        data: P2ImMessageReceiveV1,
    ) -> tuple[str, str, str, str] | None:
        """Read and validate the event trust root without parsing content."""

        try:
            message = data.event.message
            message_id = message.message_id
            chat_id = message.chat_id
            chat_type = message.chat_type
            sender_id = data.event.sender.sender_id.open_id
        except (AttributeError, TypeError):
            return None
        if not are_canonical_ingress_facts(
            message_id=message_id,
            sender_id=sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
        ):
            return None
        return message_id, chat_id, chat_type, sender_id

    def _parse_uniquely_mentioned_group_command(
        self,
        data: P2ImMessageReceiveV1,
    ) -> _MentionedGroupCommand | None:
        """Parse one uniquely mentioned group slash without consulting runtime state."""

        try:
            message = data.event.message
            event_tenant_key = data.header.tenant_key
        except (AttributeError, TypeError):
            return None
        chat_id = getattr(message, "chat_id", None)
        if message.chat_type != "group" or message.message_type != "text":
            return None
        mentions = getattr(message, "mentions", None)
        if not isinstance(mentions, (list, tuple)) or len(mentions) != 1:
            return None
        mention = mentions[0]
        identity = getattr(mention, "id", None)
        mention_key = getattr(mention, "key", None)
        target_open_id = getattr(identity, "open_id", None)
        mention_tenant_key = getattr(mention, "tenant_key", None)
        if (
            not isinstance(event_tenant_key, str)
            or not event_tenant_key
            or not isinstance(chat_id, str)
            or not chat_id
            or mention_tenant_key != event_tenant_key
            or not isinstance(mention_key, str)
            or not mention_key.startswith("@_user_")
            or not mention_key.removeprefix("@_user_").isdigit()
            or not isinstance(target_open_id, str)
            or not target_open_id.startswith("ou_")
        ):
            return None
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            return None
        try:
            decoded = json.loads(content)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        raw_text = decoded.get("text") if isinstance(decoded, dict) else None
        if (
            not isinstance(raw_text, str)
            or _LARK_MENTION_PLACEHOLDER_RE.findall(raw_text) != [mention_key]
        ):
            return None
        addressed = raw_text.lstrip()
        if not addressed.startswith(mention_key):
            return None
        command_text = addressed[len(mention_key) :]
        if not command_text[:1].isspace():
            return None
        if SlashCommandParser.parse(command_text.lstrip()) is None:
            return None
        message_id = getattr(message, "message_id", None)
        if not isinstance(message_id, str) or not message_id:
            return None
        return _MentionedGroupCommand(
            tenant_key=event_tenant_key,
            chat_id=chat_id,
            message_id=message_id,
            bot_open_id=target_open_id,
        )

    def _resolve_ready_employee_target(
        self,
        candidate: _MentionedGroupCommand,
    ) -> _EmployeeTargetedCommand | None:
        """Freeze one READY employee authority on a scheduler worker."""

        from ..autonomous.provisioning.composition import (
            EmployeeTargetResolutionUnknownError,
        )

        runtime = getattr(self, "_employee_department_runtime", None)
        resolve_target = getattr(runtime, "resolve_ready_employee_bot_target", None)
        if not callable(resolve_target):
            raise EmployeeTargetResolutionUnknownError(
                "employee target resolver is unavailable"
            )
        try:
            target = resolve_target(
                tenant_key=candidate.tenant_key,
                chat_id=candidate.chat_id,
                bot_open_id=candidate.bot_open_id,
            )
        except EmployeeTargetResolutionUnknownError:
            logger.warning("employee Bot identity lookup failed closed", exc_info=True)
            raise
        except (OSError, TimeoutError) as exc:
            logger.warning("employee Bot identity lookup failed closed", exc_info=True)
            raise EmployeeTargetResolutionUnknownError(
                "employee target lookup dependency is unavailable"
            ) from exc
        if target is None:
            return None
        agent_id = getattr(target, "agent_id", None)
        bot_principal_id = getattr(target, "bot_principal_id", None)
        app_id = getattr(target, "app_id", None)
        channel_generation = getattr(target, "channel_generation", None)
        connection_id = getattr(target, "connection_id", None)
        if (
            getattr(target, "tenant_key", None) != candidate.tenant_key
            or getattr(target, "chat_id", None) != candidate.chat_id
            or getattr(target, "bot_open_id", None) != candidate.bot_open_id
            or not isinstance(agent_id, str)
            or not agent_id.startswith("agt_")
            or not isinstance(bot_principal_id, str)
            or not bot_principal_id.startswith("bot_")
            or not isinstance(app_id, str)
            or not app_id.startswith("cli_")
            or type(channel_generation) is not int
            or channel_generation <= 0
            or not isinstance(connection_id, str)
            or not connection_id.startswith("conn_")
        ):
            raise EmployeeTargetResolutionUnknownError(
                "employee target structure is invalid"
            )
        return _EmployeeTargetedCommand(
            tenant_key=candidate.tenant_key,
            chat_id=candidate.chat_id,
            message_id=candidate.message_id,
            agent_id=agent_id,
            bot_principal_id=bot_principal_id,
            app_id=app_id,
            bot_open_id=candidate.bot_open_id,
            channel_generation=channel_generation,
            connection_id=connection_id,
        )

    def _employee_handoff_timeout(self) -> float:
        configured = getattr(
            self.settings,
            "autonomous_employee_ingress_ack_timeout_seconds",
            1.5,
        )
        if (
            isinstance(configured, bool)
            or not isinstance(configured, (int, float))
            or configured <= 0
        ):
            configured = 1.5
        return min(
            _EMPLOYEE_HANDOFF_MAX_SECONDS,
            float(configured) + _EMPLOYEE_HANDOFF_GRACE_SECONDS,
        )

    def _bind_message_ingress_reservation(
        self,
        handle: object,
        reservation: _MessageIngressReservation,
        *,
        tenant_key: str,
        chat_id: str,
    ) -> None:
        """Durably report an exact task that terminates before its callback starts."""

        if not reservation.owns():
            return

        add_done_callback = getattr(handle, "add_done_callback", None)
        if not callable(add_done_callback):
            reservation.release_if_unstarted()
            raise TypeError("scheduler task handle lacks completion callbacks")

        def report_if_unstarted(_event: object) -> None:
            if not reservation.claim_unstarted_terminal():
                return
            anchored = False
            try:
                runtime = getattr(self, "_employee_department_runtime", None)
                queue_warning = getattr(runtime, "queue_main_bot_warning", None)
                if callable(queue_warning):
                    anchored = queue_warning(
                        tenant_key=tenant_key,
                        chat_id=chat_id,
                        message_id=reservation.message_id,
                        text=UI_TEXT["ws_message_prestart_terminal"],
                    ) is True
                else:
                    logger.error(
                        "pre-start terminal durable warning queue is unavailable"
                    )
            except Exception:
                logger.error(
                    "pre-start terminal durable warning prepare failed",
                    exc_info=True,
                )
            finally:
                try:
                    if anchored:
                        reservation.commit()
                    else:
                        # The scheduler terminal proves this callback never
                        # started, so platform redelivery cannot duplicate work.
                        reservation.release()
                except Exception:
                    logger.error(
                        "message ingress terminal disposition failed",
                        exc_info=True,
                    )

        try:
            add_done_callback(report_if_unstarted)
        except Exception:
            reservation.release_if_unstarted()
            raise

    def _run_reserved_message(
        self,
        reservation: _MessageIngressReservation,
        data: P2ImMessageReceiveV1,
        *,
        task_ctx: object,
        shell_fast_tracked: bool,
        effective_trust: EffectiveTrust | None,
        employee_candidate: _MentionedGroupCommand | None,
    ) -> None:
        if not reservation.mark_started():
            return
        self._process_message_async(
            data,
            task_ctx=task_ctx,
            shell_fast_tracked=shell_fast_tracked,
            effective_trust=effective_trust,
            employee_candidate=employee_candidate,
            message_reservation_id=reservation.message_id,
            message_reservation_owner=reservation.owner,
        )

    def _employee_target_handoff_confirmed(
        self,
        target: _EmployeeTargetedCommand,
    ) -> bool | None:
        from ..autonomous.provisioning.composition import (
            EmployeeMessageHandoffUnknownError,
        )

        runtime = getattr(self, "_employee_department_runtime", None)
        wait_for_handoff = getattr(
            runtime,
            "wait_for_employee_message_handoff",
            None,
        )
        if not callable(wait_for_handoff):
            logger.warning(
                "employee targeted command handoff proof API unavailable"
            )
            return None
        try:
            result = wait_for_handoff(
                tenant_key=target.tenant_key,
                agent_id=target.agent_id,
                bot_principal_id=target.bot_principal_id,
                app_id=target.app_id,
                channel_generation=target.channel_generation,
                connection_id=target.connection_id,
                chat_id=target.chat_id,
                message_id=target.message_id,
                timeout=self._employee_handoff_timeout(),
            )
        except (EmployeeMessageHandoffUnknownError, OSError, TimeoutError):
            logger.warning(
                "employee targeted command acceptance proof unavailable",
                exc_info=True,
            )
            return None
        if result is True:
            return True
        if result is False:
            return False
        logger.warning(
            "employee targeted command handoff proof returned invalid result"
        )
        return None

    def _complete_employee_target_handoff(
        self,
        target: _EmployeeTargetedCommand,
    ) -> bool:
        """Return whether this origin must remain fenced after handoff."""

        confirmed = self._employee_target_handoff_confirmed(target)
        if confirmed is True:
            return True
        if confirmed is False:
            return self._reply_employee_handoff_unconfirmed(target)
        # UNKNOWN may mean the Employee already executed.  Warning persistence
        # failure therefore cannot authorize a competing main-Bot retry.
        self._reply_employee_handoff_unknown(target)
        return True

    def _reply_employee_handoff_unknown(
        self,
        target: _EmployeeTargetedCommand | _MentionedGroupCommand,
    ) -> bool:
        return self._queue_employee_handoff_warning(
            target,
            UI_TEXT["ws_employee_handoff_unknown"],
            label="employee targeted command handoff unknown",
        )

    def _reply_employee_handoff_unconfirmed(
        self,
        target: _EmployeeTargetedCommand | _MentionedGroupCommand,
    ) -> bool:
        return self._queue_employee_handoff_warning(
            target,
            UI_TEXT["ws_employee_handoff_unconfirmed"],
            label="employee targeted command handoff warning",
        )

    def _queue_employee_handoff_warning(
        self,
        target: _EmployeeTargetedCommand | _MentionedGroupCommand,
        text: str,
        *,
        label: str,
    ) -> bool:
        """Prepare one warning without performing transport I/O on this task."""

        from ..autonomous.provisioning.composition import (
            MainBotWarningPreparationError,
        )

        runtime = getattr(self, "_employee_department_runtime", None)
        queue_warning = getattr(runtime, "queue_main_bot_warning", None)
        if not callable(queue_warning):
            logger.error("%s durable queue is unavailable", label)
            return False
        try:
            anchored = queue_warning(
                tenant_key=target.tenant_key,
                chat_id=target.chat_id,
                message_id=target.message_id,
                text=text,
            )
        except MainBotWarningPreparationError:
            logger.error("%s durable prepare failed", label, exc_info=True)
            return False
        if anchored is not True:
            logger.error("%s durable queue returned invalid result", label)
            return False
        return True

    def _reply_text_with_bounded_retry(
        self,
        message_id: str,
        text: str,
        *,
        label: str,
    ) -> bool:
        """Deliver one user-visible terminal warning with bounded recovery."""

        idempotency_key = _employee_warning_uuid(message_id, label)
        for attempt, delay in enumerate(_EMPLOYEE_WARNING_RETRY_DELAYS, start=1):
            if delay:
                time.sleep(delay)
            try:
                reply_id = self._handler_ctx.handlers["coco"].reply_text(
                    message_id,
                    text,
                    idempotency_key=idempotency_key,
                )
                if isinstance(reply_id, str) and reply_id:
                    return True
            except (RuntimeError, OSError, TimeoutError, TypeError, ValueError):
                logger.debug(
                    "%s delivery attempt %d raised",
                    label,
                    attempt,
                    exc_info=True,
                )
            if attempt < len(_EMPLOYEE_WARNING_RETRY_DELAYS):
                logger.debug("%s delivery retry %d", label, attempt)
                continue
            logger.warning("%s delivery failed", label)
        return False

    def _schedule_employee_handoff_warning(
        self,
        candidate: _MentionedGroupCommand,
        *,
        reservation: _MessageIngressReservation,
        request_id: str,
        sender_id: str,
        sender_union_id: str,
        tenant_key: str,
    ) -> bool:
        """Transfer one owned ingress reservation to the system warning lane."""

        spec = TaskSpec(
            chat_id=candidate.chat_id,
            name="employee_handoff_warning",
            task_type="employee_handoff_warning",
            message_id=candidate.message_id,
            origin_message_id=candidate.message_id,
            request_id=request_id,
            priority=TaskPriority.HIGH,
            is_system_command=True,
            is_p2p=False,
            sender_id=sender_id,
            sender_union_id=sender_union_id,
            tenant_key=tenant_key,
        )

        def warn(_ctx) -> None:
            if not reservation.mark_started():
                return
            if not reservation.owns():
                return
            try:
                main_bot_open_id = (
                    self._main_bot_open_id or self._sync_main_bot_identity()
                )
                if not main_bot_open_id:
                    delivered = self._reply_text_with_bounded_retry(
                        candidate.message_id,
                        UI_TEXT["ws_main_bot_identity_unavailable"],
                        label="main Bot identity warning",
                    )
                elif candidate.bot_open_id == main_bot_open_id:
                    delivered = self._reply_text_with_bounded_retry(
                        candidate.message_id,
                        UI_TEXT["ws_backpressure_generic"],
                        label="main Bot backpressure warning",
                    )
                else:
                    from ..autonomous.provisioning.composition import (
                        EmployeeTargetResolutionUnknownError,
                    )

                    try:
                        target = self._resolve_ready_employee_target(candidate)
                    except EmployeeTargetResolutionUnknownError:
                        try:
                            self._reply_employee_handoff_unknown(candidate)
                        finally:
                            # UNKNOWN may conceal completed Employee execution.
                            reservation.commit()
                        return
                    if target is not None:
                        confirmed = self._employee_target_handoff_confirmed(target)
                        if confirmed is False:
                            if self._reply_employee_handoff_unconfirmed(candidate):
                                reservation.commit()
                        elif confirmed is None:
                            try:
                                self._reply_employee_handoff_unknown(candidate)
                            finally:
                                # UNKNOWN may conceal completed Employee execution.
                                reservation.commit()
                        else:
                            reservation.commit()
                        return
                    delivered = self._reply_employee_handoff_unconfirmed(candidate)
                if delivered:
                    reservation.commit()
            finally:
                reservation.release()

        try:
            handle = self._scheduler.submit(spec, warn)
        except (
            RateLimitExceededException,
            CircuitBreakerOpenException,
            TaskQueueFullError,
            RuntimeError,
            OSError,
            TimeoutError,
            TypeError,
            ValueError,
        ):
            reservation.release()
            logger.warning(
                "employee handoff warning admission failed closed",
                exc_info=True,
            )
            return False
        self._bind_message_ingress_reservation(
            handle,
            reservation,
            tenant_key=tenant_key,
            chat_id=candidate.chat_id,
        )
        return True

    def _handle_message(self, data: P2ImMessageReceiveV1):
        """飞书消息事件入口：只做轻量前置判断，然后交给 scheduler 异步处理。"""
        ingress_facts = self._extract_canonical_ingress_facts(data)
        if ingress_facts is None:
            audit_logger.warning("INGRESS_MALFORMED_EVENT_REJECTED phase=intake")
            return
        message_id, chat_id, chat_type, _sender_id = ingress_facts
        is_p2p = chat_type == "p2p"
        employee_candidate = self._parse_uniquely_mentioned_group_command(data)
        if employee_candidate is not None:
            main_bot_open_id = self._main_bot_open_id
            if (
                not isinstance(main_bot_open_id, str)
                or not main_bot_open_id.startswith("ou_")
            ):
                audit_logger.warning(
                    "MENTIONED_COMMAND_TARGET_IDENTITY_UNAVAILABLE chat_hash=%s",
                    self._access_identifier_hash(chat_id),
                )
                return
            if employee_candidate.bot_open_id == main_bot_open_id:
                # A mention is routing metadata, not an Employee authority
                # fact.  Self-addressed commands retain the ordinary trust,
                # project, origin, and control-queue path.
                employee_candidate = None

        configured_owner = getattr(self, "_managed_group_owner_id", "")
        defer_employee_target_trust = bool(
            employee_candidate is not None
            and isinstance(configured_owner, str)
            and configured_owner
            and _sender_id == configured_owner
        )
        if (
            employee_candidate is not None
            and isinstance(configured_owner, str)
            and configured_owner
            and not defer_employee_target_trust
        ):
            return

        message = data.event.message
        causal_message_id = (
            getattr(message, "parent_id", None)
            or getattr(message, "root_id", None)
            or ""
        )
        effective_trust = (
            None
            if employee_candidate is not None
            else self._resolve_effective_trust(
                sender_id=_sender_id,
                chat_id=chat_id,
                chat_type=chat_type,
                message_id=causal_message_id,
            )
        )
        trust_decision = self._managed_trust_access_decision(effective_trust)
        if trust_decision is not None and not trust_decision.allowed:
            return

        _raw_sender_union_id = getattr(
            getattr(getattr(data.event, "sender", None), "sender_id", None),
            "union_id", None,
        )
        _sender_union_id = (
            _raw_sender_union_id if isinstance(_raw_sender_union_id, str) else ""
        )
        _raw_tenant_key = getattr(getattr(data, "header", None), "tenant_key", None)
        tenant_key = _raw_tenant_key if isinstance(_raw_tenant_key, str) else ""

        # Ordinary routes have already resolved their current trust.  A
        # uniquely Employee-addressed command from the configured Owner uses
        # bounded deferred admission so the WS callback never blocks on the
        # durable Registry; the worker rechecks that authority before target
        # lookup, warning delivery, or handoff.
        text = self._extract_text_from_message(data)
        command_match = SlashCommandParser.parse(text)
        ingress_decision = (
            AccessDecision(
                allowed=True,
                operation=AccessOperation.NORMAL_MESSAGE,
                reason_code="employee_target_deferred_trust",
                prospective_allowed=True,
            )
            if defer_employee_target_trust
            else trust_decision
            or self._decide_ingress_access(
                message_id=message_id,
                sender_id=_sender_id,
                chat_id=chat_id,
                chat_type=chat_type,
                command_match=command_match,
            )
        )
        if not ingress_decision.allowed:
            return
        if employee_candidate is None and not self._managed_ingress_action_allowed(
            effective_trust,
            text=text,
            command_match=command_match,
        ):
            return

        managed_group = (
            effective_trust.managed_group
            if effective_trust is not None
            and effective_trust.zone is TrustZone.MANAGED_AGENT_GROUP
            else None
        )
        project_id = managed_group.project_id if managed_group is not None else None
        thread_root_id = None
        if employee_candidate is None:
            try:
                parent_id = getattr(data.event.message, "parent_id", None)
                root_id = getattr(data.event.message, "root_id", None)
                thread_root_id = root_id
                thread_ctx = None

                if managed_group is None and root_id and self.settings.thread_programming_enabled:
                    thread_ctx = self._thread_manager.get(root_id)
                    if thread_ctx:
                        project_id = thread_ctx.project_id
                        thread_root_id = thread_ctx.thread_root_id
                        logger.debug(
                            "[Thread] _handle_message hit: msg_root=%s canonical=%s mode=%s",
                            root_id[:12] if root_id else "N", thread_ctx.thread_root_id[:12], thread_ctx.mode,
                        )
                    else:
                        logger.debug("[Thread] _handle_message miss: msg_root=%s", root_id[:12] if root_id else "N")

                if managed_group is None and not project_id:
                    for ref in (parent_id, root_id):
                        if ref:
                            project_id = self._message_mapper.get_project_id(ref)
                            if not isinstance(project_id, str):
                                project_id = None
                            if project_id:
                                break
            except (AttributeError, KeyError, TypeError):
                project_id = None

        if employee_candidate is None and managed_group is None and not project_id:
            try:
                active = self._project_manager.get_active_project(chat_id)
                project_id = active.project_id if active else None
                if not isinstance(project_id, str):
                    project_id = None
            except (AttributeError, KeyError):
                project_id = None

        is_system = bool(
            text and (text.startswith("/") or SystemHandler.is_exit_command(text))
        )
        is_shell_fast = (
            False
            if is_system
            else SystemHandler.is_likely_shell_command(text)
        )
        is_spec = SystemHandler.is_spec_command(text) if text else False

        # For likely shell commands, route to a separate shell queue so they
        # don't block behind long-running programming tasks on the project queue.
        shell_queue_key = None
        if is_shell_fast:
            queue_suffix = project_id or "default"
            if thread_root_id:
                queue_suffix = f"{queue_suffix}:t:{thread_root_id}"
            shell_queue_key = f"{chat_id}:shell:{queue_suffix}"

        control_queue_key = self._build_control_queue_key(chat_id=chat_id, project_id=project_id, text=text)
        queue_key = shell_queue_key or control_queue_key
        if employee_candidate is not None:
            queue_key = f"{chat_id}:employee-handoff"
        if not queue_key and thread_root_id and self.settings.thread_programming_enabled:
            queue_suffix = project_id or "default"
            queue_key = f"{chat_id}:{queue_suffix}:t:{thread_root_id}"

        request_id = self._handler_ctx.handlers["coco"].ensure_request_id(message_id, chat_id=chat_id, project_id=project_id)
        try:
            self._message_linker.register_origin(
                message_id,
                request_id=request_id,
                chat_id=chat_id,
                project_id=project_id,
                chat_type=chat_type,
                sender_id=_sender_id,
                tenant_key=tenant_key,
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as e:
            logger.debug(
                "register trusted message origin failed: message_id=%s, err=%s",
                message_id,
                get_error_detail(e),
            )

        with TraceContext(request_id):
            task_type = (
                "employee_target_handoff"
                if employee_candidate is not None
                else ("spec_command" if is_spec else "feishu_message")
            )
            if employee_candidate is None and is_system:
                tl = (text or "").strip().lower()
                if tl in {"/help", "/帮助"}:
                    task_type = "system_help"
                elif tl in {"/exit", "/quit"}:
                    task_type = "system_exit"
            spec = TaskSpec(
                chat_id=chat_id,
                name="process_message",
                task_type=task_type,
                message_id=message_id,
                project_id=project_id,
                origin_message_id=message_id,
                request_id=request_id,
                priority=TaskPriority.HIGH if is_system else TaskPriority.NORMAL,
                is_system_command=is_system,
                is_p2p=is_p2p,
                sender_id=_sender_id,
                sender_union_id=_sender_union_id,
                tenant_key=tenant_key,
                queue_key=queue_key,
            )
            reservation_owner = self._message_ingress_guard.reserve(message_id)
            if reservation_owner is None:
                return
            reservation = _MessageIngressReservation(
                guard=self._message_ingress_guard,
                message_id=message_id,
                owner=reservation_owner,
            )
            try:
                handle = self._scheduler.submit(
                    spec,
                    lambda ctx,
                    _sf=is_shell_fast,
                    _trust=effective_trust,
                    _candidate=employee_candidate,
                    _reservation=reservation: self._run_reserved_message(
                        _reservation,
                        data,
                        task_ctx=ctx,
                        shell_fast_tracked=_sf,
                        effective_trust=_trust,
                        employee_candidate=_candidate,
                    ),
                )
            except (
                RateLimitExceededException,
                CircuitBreakerOpenException,
                TaskQueueFullError,
            ) as e:
                logger.warning(f"Backpressure applied: {get_error_detail(e)}")
                if employee_candidate is not None:
                    if not self._schedule_employee_handoff_warning(
                        employee_candidate,
                        reservation=reservation,
                        request_id=request_id,
                        sender_id=_sender_id,
                        sender_union_id=_sender_union_id,
                        tenant_key=tenant_key,
                    ):
                        raise
                else:
                    warning_text = (
                        UI_TEXT["ws_backpressure_spec"]
                        if is_spec
                        else UI_TEXT["ws_backpressure_generic"]
                    )
                    delivered: object = None
                    try:
                        delivered = self._handler_ctx.handlers["coco"].reply_text(
                            message_id,
                            warning_text,
                        )
                    finally:
                        if isinstance(delivered, str) and delivered:
                            reservation.commit()
                        else:
                            reservation.release()
                return
            except (RuntimeError, OSError, TimeoutError, TypeError, ValueError):
                reservation.release()
                raise
            self._bind_message_ingress_reservation(
                handle,
                reservation,
                tenant_key=tenant_key,
                chat_id=chat_id,
            )
            try:
                if message_id:
                    self._message_linker.link_task(message_id, handle.run_id)
            except (KeyError, AttributeError, RuntimeError) as e:
                logger.debug("link_task失败(message): message_id=%s, run_id=%s, err=%s", message_id, handle.run_id, get_error_detail(e))

    def _extract_text_from_message(self, data: P2ImMessageReceiveV1) -> str:
        """Extract message text with the same parser used by async dispatch."""
        try:
            message = data.event.message
            content_str = message.content
            if not content_str:
                return ""
            parsed = self._get_image_handler().parse_message(
                message.message_type,
                content_str,
            )
            return self._clean_at_text(parsed.text)
        except (AttributeError, KeyError, TypeError, ValueError):
            return ""

    def _build_control_queue_key(self, *, chat_id: str, project_id: Optional[str], text: str) -> Optional[str]:
        """为编程初始化与 spec 命令构造串行控制队列 key。"""
        from .product_catalog import is_programming_entry_command

        normalized = (text or "").strip()
        if not normalized:
            return None
        if not (
            SystemHandler.is_spec_command(normalized)
            or is_programming_entry_command(normalized)
            or SystemHandler.is_workflow_command(normalized)
        ):
            return None
        queue_suffix = project_id or "default"
        return f"{chat_id}:control:{queue_suffix}"

    def _process_message_async(
        self,
        data: P2ImMessageReceiveV1,
        task_ctx=None,
        shell_fast_tracked: bool = False,
        effective_trust: EffectiveTrust | None = None,
        employee_candidate: _MentionedGroupCommand | None = None,
        message_reservation_id: str | None = None,
        message_reservation_owner: str | None = None,
    ):
        """消息处理主逻辑（运行在 scheduler 线程池中）。

        大致流程：校验 → 解析文本/图片 → 解析项目上下文 → 路由到对应模式/引擎。
        """
        from ..thread import (
            set_current_is_p2p,
            set_current_mentioned_names,
            set_current_sender_id,
            set_current_sender_name,
            set_current_sender_union_id,
            set_current_tenant_key,
            set_current_thread_id,
        )

        def release_message_reservation() -> None:
            if (
                message_reservation_id is not None
                and message_reservation_owner is not None
            ):
                self._message_ingress_guard.release(
                    message_reservation_id,
                    message_reservation_owner,
                )

        message_id = ""
        try:
            ingress_facts = self._extract_canonical_ingress_facts(data)
            if ingress_facts is None:
                audit_logger.warning(
                    "INGRESS_MALFORMED_EVENT_REJECTED phase=worker"
                )
                return
            message_id, chat_id, chat_type, _sender_id = ingress_facts
            event = data.event
            message = event.message

            event_sender_union_id = getattr(
                getattr(getattr(event, "sender", None), "sender_id", None),
                "union_id",
                "",
            )
            _sender_union_id = (
                event_sender_union_id
                if isinstance(event_sender_union_id, str)
                else ""
            )
            _is_p2p = chat_type == "p2p"
            event_tenant_key = getattr(
                getattr(data, "header", None),
                "tenant_key",
                "",
            )
            _tenant_key = (
                event_tenant_key if isinstance(event_tenant_key, str) else ""
            )

            causal_message_id = (
                getattr(message, "parent_id", None)
                or getattr(message, "root_id", None)
                or ""
            )
            current_trust = self._resolve_effective_trust(
                sender_id=_sender_id,
                chat_id=chat_id,
                chat_type=chat_type,
                message_id=causal_message_id,
            )
            trust_decision = self._managed_trust_access_decision(current_trust)
            if trust_decision is not None and not trust_decision.allowed:
                return
            if effective_trust is not None and current_trust != effective_trust:
                audit_logger.warning(
                    "MANAGED_INGRESS_STALE_TRUST_DENIED chat_hash=%s",
                    self._access_identifier_hash(chat_id),
                )
                return

            # Parsing is allowed only after the current event facts pass their
            # independent worker check. No downstream fact is trusted from the
            # queued TaskSpec.
            image_handler = self._get_image_handler()
            parse_result = image_handler.parse_message(message.message_type, message.content)
            text = self._clean_at_text(parse_result.text)

            # Slash parsing is request-scoped: parse once and reuse.
            # This match becomes the single source of truth for downstream
            # slash consumers (gate/system/engines).
            try:
                command_match = SlashCommandParser.parse(text)
            except Exception:
                command_match = None

            ingress_decision = trust_decision or self._decide_ingress_access(
                message_id=message_id,
                sender_id=_sender_id,
                chat_id=chat_id,
                chat_type=chat_type,
                command_match=command_match,
            )
            if not ingress_decision.allowed:
                return
            request_id = self._handler_ctx.handlers["coco"].ensure_request_id(message_id, chat_id=chat_id)

            # Validation follows authorization so denied traffic cannot mutate
            # duplicate/expiry state or trigger unsupported-content replies.
            if not self._validate_message(
                message,
                request_id,
                message_reservation_owner=message_reservation_owner,
            ):
                return

            employee_target = None
            if employee_candidate is not None:
                if (
                    employee_candidate.tenant_key != _tenant_key
                    or employee_candidate.chat_id != chat_id
                    or employee_candidate.message_id != message_id
                ):
                    audit_logger.warning(
                        "EMPLOYEE_TARGET_CANDIDATE_EVENT_MISMATCH chat_hash=%s",
                        self._access_identifier_hash(chat_id),
                    )
                    return
                main_bot_open_id = (
                    self._main_bot_open_id or self._sync_main_bot_identity()
                )
                if not main_bot_open_id:
                    set_current_sender_id(_sender_id)
                    set_current_sender_union_id(_sender_union_id or None)
                    set_current_is_p2p(_is_p2p)
                    set_current_tenant_key(_tenant_key or None)
                    delivered = self._reply_text_with_bounded_retry(
                        employee_candidate.message_id,
                        UI_TEXT["ws_main_bot_identity_unavailable"],
                        label="main Bot identity warning",
                    )
                    if not delivered:
                        release_message_reservation()
                    return
                if employee_candidate.bot_open_id != main_bot_open_id:
                    from ..autonomous.provisioning.composition import (
                        EmployeeTargetResolutionUnknownError,
                    )

                    try:
                        employee_target = self._resolve_ready_employee_target(
                            employee_candidate
                        )
                    except EmployeeTargetResolutionUnknownError:
                        set_current_sender_id(_sender_id)
                        set_current_sender_union_id(_sender_union_id or None)
                        set_current_is_p2p(_is_p2p)
                        set_current_tenant_key(_tenant_key or None)
                        self._reply_employee_handoff_unknown(employee_candidate)
                        return
                    if employee_target is None:
                        set_current_sender_id(_sender_id)
                        set_current_sender_union_id(_sender_union_id or None)
                        set_current_is_p2p(_is_p2p)
                        set_current_tenant_key(_tenant_key or None)
                        anchored = self._reply_employee_handoff_unconfirmed(
                            employee_candidate
                        )
                        if not anchored:
                            # A definite no-READY result cannot conceal
                            # Employee execution, so a failed warning prepare
                            # may safely allow the platform to retry this origin.
                            release_message_reservation()
                        return

            if employee_target is None and not self._managed_ingress_action_allowed(
                current_trust,
                text=text,
                command_match=command_match,
            ):
                return

            if employee_target is not None:
                if (
                    employee_target.tenant_key != _tenant_key
                    or employee_target.chat_id != chat_id
                    or employee_target.message_id != message_id
                ):
                    audit_logger.warning(
                        "EMPLOYEE_TARGET_HANDOFF_EVENT_MISMATCH chat_hash=%s",
                        self._access_identifier_hash(chat_id),
                    )
                    return
                set_current_sender_id(_sender_id)
                set_current_sender_union_id(_sender_union_id or None)
                set_current_is_p2p(_is_p2p)
                set_current_tenant_key(_tenant_key or None)
                must_remain_fenced = self._complete_employee_target_handoff(
                    employee_target
                )
                if not must_remain_fenced:
                    # Only a durable handoff denial reaches this branch.
                    release_message_reservation()
                return

            if ingress_decision.operation in {
                AccessOperation.BOOTSTRAP_ADMIN,
                AccessOperation.ENROL_CURRENT_CHAT,
            }:
                if command_match is None:
                    logger.critical(
                        "INGRESS_ENROLMENT_COMMAND_MATCH_MISSING operation=%s",
                        ingress_decision.operation.value,
                    )
                    return
                # Enrollment is its own narrow route. It must not touch chat
                # locks, GroupLedger, image downloads, Shell, project lookup,
                # or any non-system handler.
                set_current_sender_id(_sender_id)
                set_current_sender_union_id(_sender_union_id or None)
                set_current_is_p2p(_is_p2p)
                set_current_tenant_key(_tenant_key or None)
                self._handler_ctx.handlers["system"].handle_intercepted_command(
                    message_id,
                    chat_id,
                    text,
                    command_match=command_match,
                )
                return

            # Propagate authorized identity to downstream request-local state.
            set_current_sender_id(_sender_id)
            set_current_sender_union_id(_sender_union_id or None)
            set_current_is_p2p(_is_p2p)
            set_current_tenant_key(_tenant_key or None)

            from .user_cache import resolve_display_name_nonblocking

            _display_name = (
                resolve_display_name_nonblocking(_sender_id, self._get_api_client)
                if _sender_id
                else ""
            )
            set_current_sender_name(
                _display_name or (_sender_id[:8] if _sender_id else "")
            )
            structured_mentions = tuple(
                dict.fromkeys(
                    name
                    for mention in (getattr(message, "mentions", None) or ())
                    if isinstance((name := getattr(mention, "name", None)), str)
                    and name
                    and name == name.strip()
                    and (
                        not getattr(mention, "tenant_key", None)
                        or not _tenant_key
                        or getattr(mention, "tenant_key", None) == _tenant_key
                    )
                )
            )
            set_current_mentioned_names(structured_mentions)

            root_id = getattr(message, "root_id", None)
            if (
                root_id
                and self.settings.thread_programming_enabled
                and not (
                    current_trust is not None
                    and current_trust.zone is TrustZone.MANAGED_AGENT_GROUP
                )
            ):
                thread_ctx = self._thread_manager.get(root_id)
                if thread_ctx:
                    set_current_thread_id(thread_ctx.thread_root_id)
                    logger.debug(
                        "[Thread] _process_async hit: msg_root=%s canonical=%s",
                        root_id[:12],
                        thread_ctx.thread_root_id[:12],
                    )

            # Chat lock interception (fail-close: non-admin blocked on exception).
            # Use the request-scoped CommandMatch instead of re-parsing raw text.
            if self._chat_lock_gate.check(
                chat_id,
                _sender_id,
                message_id,
                command_match=command_match,
            ):
                return

            # Publish every authorized main-Bot group message before any
            # programming/engine mode can bypass SMART routing.  When the
            # canonical ledger is composed, failure is fail-closed so the
            # current event can never execute without its durable context.
            if not _is_p2p and text:
                runtime = getattr(self, "_employee_department_runtime", None)
                record_group_event = getattr(runtime, "record_group_event", None)
                if callable(record_group_event):
                    record_group_event(
                        tenant_key=_tenant_key,
                        chat_id=chat_id,
                        thread_id=getattr(message, "thread_id", None) or "",
                        message_id=message_id,
                        sender_id=_sender_id,
                        text=text,
                    )

            # 3. Handle Images (if any)
            is_image_only = False
            trusted_project = None
            if (
                current_trust is not None
                and current_trust.managed_group is not None
            ):
                managed_group = current_trust.managed_group
                trusted_project = self._project_manager.get_project_for_chat(
                    managed_group.project_id,
                    chat_id,
                )
                if trusted_project is None:
                    return
                if (
                    getattr(trusted_project, "project_id", None)
                    != managed_group.project_id
                    or getattr(trusted_project, "root_path", None)
                    != managed_group.canonical_root_ref
                ):
                    return
            if parse_result.image_keys:
                image_kwargs = {}
                if trusted_project is not None:
                    image_kwargs["trusted_project"] = trusted_project
                project, auto_enter_mode, text, is_image_only = self._handle_image_content(
                    message,
                    parse_result.image_keys,
                    text,
                    request_id,
                    task_ctx,
                    **image_kwargs,
                )
                # Downloaded image references are part of the effective prompt.
                # Refresh slash args so engine consumers receive the
                # same evidence-rich goal as text-based engine handlers.
                if command_match is not None:
                    enriched_match = SlashCommandParser.parse(text)
                    if enriched_match is not None and enriched_match.command == command_match.command:
                        command_match = enriched_match
                if trusted_project is not None:
                    project = trusted_project
            else:
                # 4. Resolve Context (if no images to drive it)
                if trusted_project is not None:
                    project, auto_enter_mode = trusted_project, None
                else:
                    project, auto_enter_mode = self._resolve_message_context(message)

            # 4b. Safety net: if auto_enter_mode is still None but we are in a
            # registered thread, force-set mode from thread context so that the
            # message never accidentally falls through to SMART / intent recognition.
            if not auto_enter_mode and trusted_project is None:
                _root = getattr(message, "root_id", None)
                if _root and self.settings.thread_programming_enabled:
                    _tctx = self._thread_manager.get(_root)
                    if _tctx and _tctx.mode and _tctx.mode != "smart":
                        auto_enter_mode = _tctx.mode
                        set_current_thread_id(_tctx.thread_root_id)
                        thread_project = self._project_manager.get_project_for_chat(
                            _tctx.project_id,
                            chat_id,
                        )
                        if _tctx.mode in {"deep", "spec", "workflow"}:
                            project = thread_project
                        elif not project:
                            project = thread_project or self._project_manager.get_active_project(chat_id)
                        logger.info(
                            "[Thread] Safety-net resolved mode: root=%s canonical=%s mode=%s",
                            _root[:12], _tctx.thread_root_id[:12], auto_enter_mode,
                        )

            # 5. Handle Context Updates (Task Scheduler)
            if task_ctx and project:
                self._update_task_project(task_ctx, project.project_id)

            # 6. Dispatch Logic
            if not self._current_trust_can_dispatch(current_trust, project=project):
                return
            if not text and not is_image_only:
                # Special case: handle empty text (e.g. unsupported content that parsed to empty)
                # But wait, if image_keys exist, text might be empty but valid (image only).
                # _handle_image_content handles text augmentation.
                # If we are here and text is empty, check if we should show help or dispatch to mode.
                _root_id = getattr(message, "root_id", None)
                self._dispatch_empty_text(message_id, chat_id, project, task_ctx, root_id=_root_id)
                return

            self._dispatch_message_logic(
                message_id,
                chat_id,
                text,
                project,
                auto_enter_mode,
                command_match=command_match,
                is_image_only=is_image_only,
                shell_fast_tracked=shell_fast_tracked,
                chat_type=chat_type,
                effective_trust=current_trust,
            )

        except asyncio.TimeoutError as e:
            logger.warning("处理消息超时: %s", get_error_detail(e))
            try:
                self._handler_ctx.handlers["coco"].reply_text(message_id, UI_TEXT["ws_message_timeout"])
            except (RuntimeError, OSError, TimeoutError, TypeError, ValueError):
                classify_ws_error(RuntimeError("reply timeout failed"), phase="dispatch")
                logger.debug("failed to reply timeout message", exc_info=True)
        except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as e:
            classification = classify_ws_error(e, phase="dispatch")
            if classification.action == WSErrorAction.REPLY_INTERNAL_ERROR:
                logger.error("处理消息异常: %s", get_error_detail(e), exc_info=True)
                try:
                    self._handler_ctx.handlers["coco"].reply_text(message_id, UI_TEXT["ws_message_internal_error"])
                except (RuntimeError, OSError, TimeoutError, TypeError, ValueError):
                    classify_ws_error(RuntimeError("reply internal error failed"), phase="best_effort_notify")
                    logger.debug("failed to reply internal error message", exc_info=True)
            elif classification.action == WSErrorAction.LOG_AND_CONTINUE:
                logger.debug("处理消息 best-effort 失败: %s", get_error_detail(e), exc_info=True)
            elif classification.action == WSErrorAction.PROPAGATE:
                logger.error("处理消息异常: %s", get_error_detail(e), exc_info=True)
                raise
        finally:
            set_current_thread_id(None)
            set_current_sender_id(None)
            set_current_sender_union_id(None)
            set_current_sender_name("")
            set_current_is_p2p(False)
            set_current_tenant_key(None)
            if message_id:
                with self._pending_image_lock:
                    self._pending_image_keys.pop(message_id, None)
                    self._pending_image_only.discard(message_id)
            if (
                message_reservation_id is not None
                and message_reservation_owner is not None
            ):
                self._message_ingress_guard.commit(
                    message_reservation_id,
                    message_reservation_owner,
                )

    def _validate_message(
        self,
        message,
        request_id: str,
        *,
        message_reservation_owner: str | None = None,
    ) -> bool:
        """校验消息是否需要处理（过期/重复/类型不支持等）。"""
        if message.create_time and self._message_ingress_guard.is_message_expired(
            int(message.create_time)
        ):
            logger.debug("跳过过期消息: %s", message.message_id)
            return False

        if message_reservation_owner is not None:
            if not self._message_ingress_guard.owns(
                message.message_id,
                message_reservation_owner,
            ):
                logger.debug("跳过无主消息占位: %s", message.message_id)
                return False
        elif self._message_ingress_guard.is_duplicate_message(message.message_id):
            logger.debug("跳过重复消息: %s", message.message_id)
            return False

        supported_types = {"text", "image", "post"}
        if message.message_type not in supported_types:
            self._handler_ctx.handlers["coco"].reply_text(message.message_id, UI_TEXT["ws_unsupported_msg_type"])
            return False
        return True

    def _clean_at_text(self, text: str) -> str:
        """移除 '@机器人' 前缀，得到用户真实输入文本。"""
        text = text.strip()
        if text.startswith("@"):
            parts = text.split(None, 1)
            if len(parts) > 1:
                return parts[1].strip()
            return ""
        return text

    def _handle_image_content(
        self,
        message,
        image_keys,
        text,
        request_id,
        task_ctx,
        *,
        trusted_project=None,
    ):
        """处理图片消息：下载并把图片引用文本拼接回 prompt。

        返回 `(project, auto_enter_mode, text, is_image_only)`。
        """
        message_id = message.message_id
        chat_id = message.chat_id
        getattr(message, "parent_id", None)
        getattr(message, "root_id", None)

        with self._pending_image_lock:
            self._pending_image_keys[message_id] = image_keys

        if trusted_project is not None:
            project, auto_enter_mode = trusted_project, None
        else:
            project, auto_enter_mode = self._resolve_message_context(message)

        try:
            if project:
                self._message_linker.register_origin(
                    message_id, request_id=request_id, chat_id=chat_id, project_id=project.project_id
                )
        except Exception as e:
            logger.debug("register_origin失败(image_msg): message_id=%s, err=%s", message_id, get_error_detail(e))

        if task_ctx and project:
            self._update_task_project(task_ctx, project.project_id)

        save_dir = FeishuImageHandler.get_image_save_dir(
            project.root_path if project else None,
            self._handler_ctx.handlers["coco"].get_working_dir(chat_id),
        )

        image_handler = self._get_image_handler()
        download_result = image_handler.download_images(message_id, image_keys, save_dir)

        is_image_only = False
        if download_result.saved_paths:
            is_image_only = not text
            ref_text = FeishuImageHandler.build_image_reference_text(download_result.saved_paths)
            if text:
                text += ref_text
            else:
                text = UI_TEXT["ws_image_only_prefix"] + ref_text

        if download_result.failed_keys:
            logger.warning("部分图片下载失败: %s", download_result.failed_keys)

        if is_image_only:
            with self._pending_image_lock:
                self._pending_image_only.add(message_id)

        return project, auto_enter_mode, text, is_image_only

    def _resolve_message_context(self, message):
        """从 message 的 parent/root 引用恢复项目上下文。

        优先级：话题上下文 > 消息引用上下文 > 当前 active project。
        """
        message_id = message.message_id
        chat_id = message.chat_id
        parent_id = getattr(message, "parent_id", None)
        root_id = getattr(message, "root_id", None)

        if root_id and self.settings.thread_programming_enabled:
            thread_ctx = self._thread_manager.get(root_id)
            if thread_ctx:
                from ..thread import set_current_thread_id
                set_current_thread_id(thread_ctx.thread_root_id)
                auto_enter_mode = thread_ctx.mode if thread_ctx.mode != "smart" else None
                project = self._project_manager.get_project_for_chat(thread_ctx.project_id, chat_id)
                if not project and thread_ctx.mode not in {"deep", "spec", "workflow"}:
                    project = self._project_manager.get_active_project(chat_id)
                logger.info(
                    "[Thread] Resolved context: msg_root=%s canonical=%s project=%s mode=%s project_found=%s",
                    root_id[:12], thread_ctx.thread_root_id[:12], thread_ctx.project_id, thread_ctx.mode, project is not None,
                )
                return project, auto_enter_mode

        return self._resolve_project_from_message(message_id, chat_id, parent_id or root_id)

    def _get_effective_mode(self, chat_id: str, project_id: Optional[str] = None):
        from ..mode import InteractionMode
        thread_id = get_current_thread_id()
        if thread_id:
            thread_ctx = self._thread_manager.get(thread_id)
            if thread_ctx and thread_ctx.mode != "smart":
                try:
                    return InteractionMode(thread_ctx.mode), True
                except ValueError:
                    logger.debug(
                        "thread mode is engine-only, not InteractionMode: %s",
                        thread_ctx.mode,
                    )
                    return InteractionMode.SMART, False
        return (
            self._mode_manager.get_mode(chat_id, project_id=project_id),
            self._mode_manager.is_programming_mode(chat_id, project_id=project_id),
        )

    def _is_topic_engine_context(self) -> bool:
        """Return True when the current Feishu topic is owned by an engine."""
        thread_id = get_current_thread_id()
        if not thread_id or not self.settings.thread_programming_enabled:
            return False
        thread_ctx = self._thread_manager.get(thread_id)
        return bool(thread_ctx and thread_ctx.mode in {"deep", "spec", "workflow"})

    def _get_mode_handler(self, mode):
        return self._handler_ctx.handlers.get(getattr(mode, "value", ""))

    def _find_active_thread(self, chat_id):
        if not self.settings.thread_programming_enabled:
            return None
        contexts = self._thread_manager.get_by_chat(chat_id)
        for ctx in contexts:
            if ctx.mode and ctx.mode != "smart":
                return ctx
        return None

    def _update_task_project(self, task_ctx, project_id):
        """将调度任务与 project_id 关联（便于任务看板/诊断）。"""
        try:
            self._scheduler.update_project_id(task_ctx.run_id, project_id)
        except Exception as e:
            logger.debug("update_project_id失败: run_id=%s, err=%s", task_ctx.run_id, get_error_detail(e))

    def _dispatch_empty_text(self, message_id, chat_id, project, task_ctx, *, root_id=None):
        """处理"文本为空"的情况：在编程模式下仍转发（保持会话），否则展示帮助。"""

        _pid = project.project_id if project else None
        if not _pid and task_ctx and task_ctx.spec.project_id:
            _pid = task_ctx.spec.project_id

        current_mode, _is_prog = self._get_effective_mode(chat_id, project_id=_pid)
        if current_mode in PROGRAMMING_MODES:
            if project is None:
                project = self._project_manager.get_active_project(chat_id)

            handler = self._get_mode_handler(current_mode)
            if handler:
                handler.handle_message(message_id, chat_id, "", project)
        else:
            self._handler_ctx.handlers["system"].show_help(message_id, chat_id)

    def _dispatch_message_logic(
        self,
        message_id,
        chat_id,
        text,
        project,
        auto_enter_mode,
        *,
        command_match,
        is_image_only=False,
        shell_fast_tracked=False,
        chat_type: str = "group",
        effective_trust: EffectiveTrust | None = None,
    ):
        """Apply topic/project safety gates, then use the one dispatcher."""
        handlers = self._handler_ctx.handlers

        def dispatch(target_project=project) -> None:
            if not self._current_trust_can_dispatch(
                effective_trust, project=target_project
            ):
                return
            self._message_dispatcher.process_with_intent(
                message_id,
                chat_id,
                text,
                target_project,
                command_match=command_match,
                shell_fast_tracked=shell_fast_tracked,
                chat_type=chat_type,
                effective_trust=effective_trust,
            )

        engine_modes = {"deep", "spec", "workflow"}
        if auto_enter_mode in engine_modes:
            command = getattr(command_match, "command", "")
            args = str(getattr(command_match, "args", "")).lower()
            is_exit = SystemHandler.is_exit_command(text)
            safe_without_project = command in {
                "/help", "/projects", "/project", "/status", "/exit", "/quit",
            } or (
                command in {"/stop_deep", "/deep_status"}
                and args in {"all", "-a", "--all"}
            )
            if project is None:
                if is_exit and self._current_trust_can_dispatch(effective_trust):
                    handlers["system"].exit_current_mode(
                        message_id, chat_id, project=None
                    )
                elif safe_without_project:
                    dispatch()
                else:
                    handlers["coco"].reply_text(
                        message_id, UI_TEXT["ws_topic_project_unavailable"]
                    )
                return

            requested = None
            if command in {"/deep", "/deep_update", "/deep_status", "/stop_deep"}:
                requested = "deep"
            elif command.startswith("/spec") or command == "/stop_spec":
                requested = "spec"
            else:
                from ..workflow_engine.commands import TOPIC_ENGINE_COMMANDS

                if command in TOPIC_ENGINE_COMMANDS:
                    requested = "workflow"
            if requested and requested != auto_enter_mode:
                handlers["coco"].reply_text(
                    message_id,
                    UI_TEXT["topic_engine_switch_blocked"].format(
                        current=auto_enter_mode.upper(),
                        requested=requested.upper(),
                    ),
                )
                return
            if command_match is None:
                if not self._current_trust_can_dispatch(
                    effective_trust, project=project
                ):
                    return
                handlers["coco"].add_reaction(
                    message_id, EmojiReaction.on_processing()
                )
                if auto_enter_mode == "workflow":
                    handlers["workflow"].handle_message(
                        message_id, chat_id, text, project
                    )
                else:
                    getattr(
                        handlers[auto_enter_mode],
                        f"start_{auto_enter_mode}_engine",
                    )(message_id, chat_id, text, project)
                return

        managed = (
            effective_trust is not None
            and effective_trust.zone is TrustZone.MANAGED_AGENT_GROUP
        )
        if (
            auto_enter_mode is None
            and not managed
            and command_match is None
            and not is_image_only
            and text
            and not self._intent_recognizer.looks_like_shell(text)
        ):
            bound_project = self._project_manager.find_by_bound_chat_id(chat_id)
            if bound_project is not None:
                project_id = getattr(bound_project, "project_id", None)
                _, is_programming = self._get_effective_mode(
                    chat_id, project_id=project_id
                )
                if is_programming:
                    dispatch(bound_project)
                    return
                tool = str(
                    getattr(bound_project, "acp_tool_name", None)
                    or getattr(self.settings, "default_acp_tool", None)
                    or "coco"
                ).strip().lower()
                handler = handlers[
                    tool if tool in PROGRAMMING_MODE_VALUES else "coco"
                ]
                if not self._current_trust_can_dispatch(
                    effective_trust, project=bound_project
                ):
                    return
                handlers["coco"].add_reaction(
                    message_id, EmojiReaction.on_coco_mode()
                )
                handlers["coco"].add_reaction(
                    message_id, EmojiReaction.on_processing()
                )
                handler.enter_mode(
                    message_id, chat_id, silent=True, project=bound_project
                )
                handler.handle_message(message_id, chat_id, text, bound_project)
                return
        dispatch()

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
            dispatcher = _CardAdvisoryDispatcher(
                max_pending=_MAX_PENDING_CARD_ADVISORIES
            )
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
        dispatcher = getattr(self, "_card_advisory_dispatcher", None)
        if dispatcher is not None:
            dispatcher.finish_response(written=written)

    def _handle_card_action(
        self,
        data: P2CardActionTrigger,
    ) -> P2CardActionTriggerResponse:
        """Consume one latest Card 2.0 callback and enqueue its business action."""
        try:
            validate_card_action_trigger(data)
        except ValueError as exc:
            logger.warning("拒绝非最新版飞书卡片回调: %s", get_error_detail(exc))
            return empty_card_action_response()

        try:
            open_message_id = data.event.context.open_message_id
            open_chat_id = data.event.context.open_chat_id
            operator = data.event.operator
            operator_id = (
                getattr(operator, "open_id", None)
                or getattr(operator, "user_id", None)
                or getattr(operator, "union_id", None)
                or ""
            )
        except (AttributeError, TypeError):
            return empty_card_action_response()

        _raw_tenant_key = getattr(getattr(data, "header", None), "tenant_key", None)
        tenant_key = _raw_tenant_key if isinstance(_raw_tenant_key, str) else ""

        effective_trust = self._resolve_effective_trust(
            sender_id=operator_id,
            chat_id=open_chat_id,
            chat_type="group",
        )
        trust_decision = self._managed_trust_access_decision(effective_trust)
        card_is_p2p = False
        if trust_decision is not None and not trust_decision.allowed:
            registry = getattr(self, "_managed_group_registry", None)
            active_group = (
                registry.active_record(open_chat_id)
                if type(registry) is ManagedGroupRegistry
                else None
            )
            # A current managed group with an unknown actor is never a DM.
            # Only the configured Owner may use durable origin provenance to
            # distinguish an actual P2P callback from an external group.
            if (
                active_group is None
                and operator_id == getattr(self, "_managed_group_owner_id", "")
            ):
                try:
                    origin = self._message_linker.resolve_origin(
                        reply_message_id=open_message_id
                    ) or open_message_id
                    card_is_p2p = self._resolve_card_is_p2p(
                        origin_message_id=origin,
                        open_chat_id=open_chat_id,
                        operator_id=operator_id,
                    )
                except (RuntimeError, OSError, TypeError, ValueError):
                    card_is_p2p = False
                if card_is_p2p:
                    effective_trust = self._resolve_effective_trust(
                        sender_id=operator_id,
                        chat_id=open_chat_id,
                        chat_type="p2p",
                    )
                    trust_decision = self._managed_trust_access_decision(
                        effective_trust
                    )
            if not card_is_p2p:
                return empty_card_action_response()

        if (
            effective_trust is not None
            and effective_trust.zone is TrustZone.MANAGED_AGENT_GROUP
        ):
            submitted_group_revision, submitted_grant_revision = (
                CardActionInspector.trust_revisions(data.event.action)
            )
            if (
                submitted_group_revision is None
                or submitted_grant_revision is None
                or submitted_group_revision != effective_trust.group_revision
            ) or (
                submitted_grant_revision != effective_trust.grant_revision
            ):
                audit_logger.warning(
                    "MANAGED_CARD_STALE_REVISION_DENIED chat_hash=%s",
                    self._access_identifier_hash(open_chat_id),
                )
                self._enqueue_card_advisory_follow_up(
                    chat_id=open_chat_id,
                    message_id=open_message_id,
                    name="refresh_stale_managed_card",
                    project_id=effective_trust.managed_group.project_id,
                    tenant_key=tenant_key,
                    callback=lambda _message_id=open_message_id,
                    _chat_id=open_chat_id,
                    _trust=effective_trust: self._refresh_managed_card_revisions(
                        _message_id,
                        _chat_id,
                        _trust,
                    ),
                )
                return empty_card_action_response()

        try:
            header = data.header
            event_id = header.event_id
            if self._card_event_cache.is_duplicate(event_id):
                logger.warning("跳过重复卡片回调事件: %s", event_id)
                return empty_card_action_response()

            event = data.event
            action = event.action
            context = event.context
            value_preview = action.value
            if isinstance(value_preview, str):
                value_preview = value_preview[:500]
            else:
                try:
                    value_preview = json.dumps(value_preview, ensure_ascii=False)[:500]
                except (TypeError, ValueError):
                    value_preview = str(value_preview)[:500]
            logger.debug(
                "卡片回调收到: event_id=%s, event_type=%s, open_message_id=%s, open_chat_id=%s, "
                "action_tag=%s, action_name=%s, value_type=%s, value_preview=%s",
                header.event_id,
                header.event_type,
                context.open_message_id,
                context.open_chat_id,
                action.tag,
                action.name,
                type(action.value).__name__,
                value_preview,
            )
        except (AttributeError, TypeError, KeyError) as e:
            logger.warning("卡片回调基础信息解析失败: %s", get_error_detail(e))
        action_type_preview = ""
        try:
            action_type_preview = CardActionInspector.action_type(data.event.action)
        except (AttributeError, TypeError, ValueError):
            classify_card_action_error(RuntimeError("action preview failed"), phase="payload_parse")
            action_type_preview = ""

        try:
            if (
                self._control_plane.is_system_cmd_inflight(open_chat_id)
                and action_type_preview not in _READONLY_CARD_ACTIONS
            ):
                if open_message_id:
                    self._enqueue_card_advisory_follow_up(
                        chat_id=open_chat_id,
                        message_id=open_message_id,
                        name="notify_system_command_gate",
                        tenant_key=tenant_key,
                        callback=lambda _message_id=open_message_id: self._handler_ctx.handlers[
                            "coco"
                        ].reply_text(
                            _message_id,
                            UI_TEXT["ws_system_cmd_gate_blocked"],
                        ),
                    )
                return empty_card_action_response()
        except (RuntimeError, OSError, TypeError, ValueError):
            classify_card_action_error(RuntimeError("system command gate failed"), phase="dispatch")
            logger.debug("failed to check system command gate", exc_info=True)
            return empty_card_action_response()

        operator_union_id = ""
        try:
            operator = data.event.operator
            raw_operator_union_id = getattr(operator, "union_id", None)
            operator_union_id = (
                raw_operator_union_id
                if isinstance(raw_operator_union_id, str)
                else ""
            )
        except (AttributeError, TypeError):
            operator_id = ""

        if open_message_id and action_type_preview:
            dedupe_fingerprint = CardActionInspector.dedup_fingerprint(data.event.action)
            dedupe_key = f"{open_chat_id}:{open_message_id}:{operator_id}:{action_type_preview}:{dedupe_fingerprint}"
            try:
                if self._card_action_dedup_cache.is_duplicate(dedupe_key):
                    return build_card_action_response(
                        toast_type="info",
                        toast_content=UI_TEXT["card_session_toast_dedup"],
                    )


            except (RuntimeError, OSError, TypeError, ValueError):
                classify_card_action_error(RuntimeError("dedup failed"), phase="dedup")
                # best-effort only
                logger.debug("failed to ack card action", exc_info=True)

        # Synchronous undo-lock expiry check: return toast if window has passed
        try:
            value_raw = data.event.action.value
            _val = value_raw if isinstance(value_raw, dict) else (
                json.loads(value_raw) if isinstance(value_raw, str) else None
            )
            if isinstance(_val, dict) and _val.get("_ul"):
                undo_expires = _val.get("_ue", 0)
                if undo_expires and time.time() > undo_expires:
                    return build_card_action_response(
                        toast_type="warning",
                        toast_content="撤销窗口已过期，请使用 /unlock 解锁",
                    )
        except (json.JSONDecodeError, TypeError, ValueError):
            classify_card_action_error(RuntimeError("undo payload parse failed"), phase="payload_parse")

        project_id = (
            effective_trust.managed_group.project_id
            if effective_trust is not None
            and effective_trust.managed_group is not None
            else None
        )
        if project_id is None:
            try:
                project_id = CardActionInspector.project_id(data.event.action)
            except (AttributeError, TypeError, ValueError):
                classify_card_action_error(RuntimeError("project id parse failed"), phase="payload_parse")
                project_id = None

        if not project_id:
            try:
                active = self._project_manager.get_active_project(open_chat_id)
                project_id = active.project_id if active else None
            except (RuntimeError, OSError, TypeError, ValueError):
                project_id = None

        origin_message_id = None
        origin_lookup_failed = False
        try:
            origin_message_id = self._message_linker.resolve_origin(reply_message_id=open_message_id)
        except (RuntimeError, OSError, TypeError, ValueError):
            origin_lookup_failed = True
        origin_message_id = origin_message_id or open_message_id
        if not card_is_p2p and not origin_lookup_failed:
            card_is_p2p = self._resolve_card_is_p2p(
                origin_message_id=origin_message_id,
                open_chat_id=open_chat_id,
                operator_id=operator_id,
            )
        # Provenance has already been resolved above. Request bookkeeping must
        # not overwrite its trusted chat/operator fields after a rejected card.
        request_id = self._handler_ctx.handlers["coco"].ensure_request_id(origin_message_id, project_id=project_id)

        is_system = CardActionInspector.is_system_action(data.event.action)

        with TraceContext(request_id):
            spec = TaskSpec(
                chat_id=open_chat_id,
                name="process_card_action",
                task_type="feishu_card_action",
                message_id=open_message_id,
                project_id=project_id,
                origin_message_id=origin_message_id,
                request_id=request_id,
                priority=TaskPriority.HIGH if is_system else TaskPriority.NORMAL,
                is_system_command=is_system,
                is_p2p=card_is_p2p,
                sender_id=operator_id,
                sender_union_id=operator_union_id,
                tenant_key=tenant_key,
            )
            handle = self._scheduler.submit(
                spec,
                lambda ctx, _trust=effective_trust: self._process_card_action_async(
                    data,
                    task_ctx=ctx,
                    effective_trust=_trust,
                ),
            )
            try:
                self._message_linker.link_task(origin_message_id, handle.run_id)
            except (KeyError, AttributeError, RuntimeError) as e:
                logger.debug(
                    "link_task失败(card_action): origin=%s, run_id=%s, err=%s", origin_message_id, handle.run_id, e
                )
        if action_type_preview in WORKFLOW_AGENT_SELECTION_ACTIONS:
            return build_card_action_response(
                toast_type="info",
                toast_content=UI_TEXT["workflow_selection_toast_processing"],
            )
        return empty_card_action_response()

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

    def _reply_employee_hire_status(self, state: object, status: str) -> object | None:
        text = _employee_hire_status_text(
            str(getattr(state, "employee_name", "")),
            status,
        )
        if text is None:
            return None
        intent_id = getattr(state, "intent_id", "")
        if not isinstance(intent_id, str) or not intent_id:
            return None
        return self._reply_employee_hire_message(
            state,
            text,
            idempotency_key=_employee_hire_status_uuid(intent_id, status),
        )

    def _reply_employee_hire_message(
        self,
        state: object,
        text: str,
        *,
        idempotency_key: str | None = None,
    ) -> object | None:
        """Restore durable requester scope before a recovered hire reply."""

        if not self._restore_employee_hire_origin(state):
            logger.warning("employee hire notification recipient scope is unavailable")
            return None
        message_id = getattr(state, "message_id", "")
        tenant_key = getattr(state, "tenant_key", "")
        previous_tenant_key = get_current_tenant_key()
        set_current_tenant_key(tenant_key or None)
        try:
            return self._handler_ctx.handlers["coco"].reply_text(
                message_id,
                text,
                idempotency_key=idempotency_key,
            )
        finally:
            set_current_tenant_key(previous_tenant_key)

    def _reply_employee_team_message(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        *,
        tenant_key: str,
        requester_principal_id: str,
        idempotency_key: str | None = None,
    ) -> object | None:
        """Restore durable Team Run requester scope before a recovered reply."""

        if not all(
            isinstance(value, str) and bool(value)
            for value in (
                message_id,
                chat_id,
                tenant_key,
                requester_principal_id,
            )
        ):
            logger.warning("employee team notification recipient scope is unavailable")
            raise RuntimeError("employee team notification recipient scope is unavailable")
        if not self._restore_durable_message_origin(
            message_id=message_id,
            chat_id=chat_id,
            sender_id=requester_principal_id,
            tenant_key=tenant_key,
        ):
            logger.warning("employee team notification recipient scope is unavailable")
            raise RuntimeError("employee team notification recipient scope is unavailable")
        previous_tenant_key = get_current_tenant_key()
        set_current_tenant_key(tenant_key)
        try:
            reply_id = self._handler_ctx.handlers["coco"].reply_text(
                message_id,
                text,
                idempotency_key=idempotency_key,
            )
            if not reply_id:
                raise RuntimeError("employee team notification delivery failed")
            return reply_id
        finally:
            set_current_tenant_key(previous_tenant_key)

    def _restore_employee_hire_origin(self, state: object) -> bool:
        message_id = getattr(state, "message_id", "")
        chat_id = getattr(state, "chat_id", "")
        sender_id = getattr(state, "requester_principal_id", "")
        tenant_key = getattr(state, "tenant_key", "")
        if not all(
            isinstance(value, str) and bool(value)
            for value in (message_id, chat_id, sender_id)
        ) or not isinstance(tenant_key, str):
            return False
        return self._restore_durable_message_origin(
            message_id=message_id,
            chat_id=chat_id,
            sender_id=sender_id,
            tenant_key=tenant_key,
        )

    def _restore_durable_message_origin(
        self,
        *,
        message_id: str,
        chat_id: str,
        sender_id: str,
        tenant_key: str,
    ) -> bool:
        """Rebuild trusted in-memory provenance from anchored durable facts."""

        try:
            origin = self._message_linker.query(message_id)
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            return False
        if origin is not None:
            chat_type = origin.get("chat_type")
            if chat_type not in {"p2p", "group", "topic_group"}:
                return False
        else:
            chat_mode = self._get_chat_mode(chat_id)
            if chat_mode is None:
                return False
            chat_type = "topic_group" if chat_mode == "topic" else chat_mode
        try:
            return (
                self._message_linker.register_trusted_origin_with_tenant(
                    message_id,
                    chat_id=chat_id,
                    sender_id=sender_id,
                    chat_type=chat_type,
                    tenant_key=tenant_key,
                )
                is True
            )
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            return False

    def _get_chat_mode(self, chat_id: str) -> str | None:
        """Read structural chat mode from the official Chat API.

        ``chat_type`` on this API is group visibility (private/public), so it
        must never participate in DM authorization.
        """
        if not chat_id or chat_id == "unknown":
            return None
        try:
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

    def _reconcile_managed_groups_before_team_start(self) -> None:
        """Reconcile pending revokes and validated legacy Project candidates."""

        from ..project_chat.lark_chat_client import (
            LarkChatClient,
            ManagedChatValidation,
        )

        # Resolve an anchored pre/post-replace state before interpreting grants
        # or running any compensation against remote chats.
        self._managed_group_registry.reconcile_uncertain_commit()
        remote = LarkChatClient(api_client_factory=self._get_api_client)

        # Project binding sagas are authoritative lifecycle work, not legacy
        # migration candidates. Consume them before scanning bound projects so
        # OWNER_ADOPTED provenance cannot be rewritten as GHOSTAP_CREATED.
        from ..trust.registry import RegistryCommitUncertainError

        for saga in self._project_manager.pending_managed_chat_binding_sagas():
            project = self._project_manager.get_project_for_diagnostics(
                saga.project_id
            )
            active = self._managed_group_registry.active_record(saga.chat_id)
            grant = self._managed_group_registry.grant_for_chat(saga.chat_id)
            expected_origin = saga.expected_origin or (
                "owner_adopted"
                if saga.operation_id.startswith("adopt:")
                else "ghostap_created"
            )
            expected_owner = saga.expected_owner_id or self._managed_group_owner_id
            expected_bot = (
                saga.expected_receiving_bot_ref
                or self._managed_group_receiving_bot_ref
            )
            expected_root = saga.expected_root_ref or (
                project.root_path if project is not None else ""
            )
            active_matches = (
                project is not None
                and project.bound_chat_id == saga.chat_id
                and project.root_path == expected_root
                and active is not None
                and grant is not None
                and active.project_id == saga.project_id
                and active.canonical_root_ref == expected_root
                and active.origin.value == expected_origin
                and active.owner_id == expected_owner
                and active.receiving_bot_ref == expected_bot
                and grant.managed_group_id == saga.chat_id
                and grant.owner_id == expected_owner
                and grant.project_id == saga.project_id
                and grant.canonical_root_ref == expected_root
                and not grant.backend_binding_ids
                and not grant.connected_target_refs
            )
            if saga.expected is None:
                if active_matches:
                    self._project_manager.complete_exact_active_legacy_saga(
                        saga.operation_id
                    )
                else:
                    self._project_manager.mark_legacy_saga_resolution_required(
                        saga.operation_id
                    )
                continue
            if not self._project_manager.validate_managed_chat_binding_saga(
                saga.operation_id
            ):
                continue
            if active_matches:
                self._project_manager.complete_managed_chat_binding_saga(
                    saga.operation_id
                )
                continue
            if active is not None:
                self._project_manager.quarantine_bound_chat(saga.chat_id)
                continue

            binding = self._managed_group_registry.provision_binding(
                saga.operation_id
            )
            if (
                project is not None
                and binding is not None
                and binding.chat_id == saga.chat_id
                and binding.project_id == saga.project_id
                and binding.canonical_root_ref == expected_root
                and binding.origin.value == expected_origin
                and binding.owner_id == expected_owner
                and binding.receiving_bot_ref == expected_bot
            ):
                try:
                    self._managed_group_registry.activate(
                        provision_id=saga.operation_id,
                        chat_id=saga.chat_id,
                        project_id=saga.project_id,
                        canonical_root_ref=expected_root,
                    )
                except RegistryCommitUncertainError as exc:
                    if exc.committed:
                        continue
                    raise
                except Exception:
                    logger.exception(
                        "managed group binding saga activation failed operation=%s",
                        saga.operation_id,
                    )
                else:
                    self._project_manager.complete_managed_chat_binding_saga(
                        saga.operation_id
                    )
                    continue

            restored = self._project_manager.restore_managed_chat_binding_saga(
                saga.operation_id
            )
            if restored:
                try:
                    self._managed_group_registry.abandon_provision(
                        saga.operation_id
                    )
                except RegistryCommitUncertainError:
                    logger.exception(
                        "managed group binding saga cleanup uncertain operation=%s",
                        saga.operation_id,
                    )
            else:
                self._project_manager.quarantine_bound_chat(saga.chat_id)

        for chat_id in self._managed_group_registry.pending_revokes():
            result = remote.delete_chat(chat_id)
            if result is False:
                continue
            if not self._project_manager.revoke_managed_chat(chat_id):
                continue
            self._managed_group_registry.complete_revoke(chat_id)

        pending_saga_chat_ids = {
            saga.chat_id
            for saga in self._project_manager.pending_managed_chat_binding_sagas()
        }
        candidates_by_chat: dict[str, list[dict]] = {}
        for project in self._project_manager.get_all_projects(
            sort_by_recent=False,
            chat_id=None,
        ):
            candidate = project.managed_group_migration_candidate(
                owner_id=self._managed_group_owner_id,
                receiving_bot_ref=self._managed_group_receiving_bot_ref,
            )
            if candidate is None:
                continue
            chat_id = candidate["chat_id"]
            if chat_id in pending_saga_chat_ids:
                continue
            candidates_by_chat.setdefault(chat_id, []).append(candidate)

        for chat_id, candidates in sorted(candidates_by_chat.items()):
            if self._managed_group_registry.record(chat_id) is not None:
                continue
            distinct_bindings = {
                (candidate["project_id"], candidate["canonical_root_ref"])
                for candidate in candidates
            }
            if len(distinct_bindings) != 1:
                project_ids = ",".join(
                    sorted({candidate["project_id"] for candidate in candidates})
                )
                self._managed_group_registry.record_migration_disposition(
                    chat_id,
                    project_id=project_ids,
                    status="ambiguous",
                )
                logger.warning(
                    "managed group migration is ambiguous chat=%s projects=%s",
                    chat_id[:12],
                    project_ids,
                )
                continue
            candidate = candidates[0]
            validation = remote.validate_managed_chat(
                chat_id,
                self._managed_group_owner_id,
            )
            if validation is ManagedChatValidation.VALID:
                self._managed_group_registry.import_candidate(
                    candidate,
                    validator=lambda _facts: True,
                )
            else:
                existing = {
                    item[0]: (item[1], item[2])
                    for item in self._managed_group_registry.migration_dispositions()
                }
                disposition = (candidate["project_id"], validation.value)
                self._managed_group_registry.record_migration_disposition(
                    chat_id,
                    project_id=candidate["project_id"],
                    status=validation.value,
                )
                if existing.get(chat_id) != disposition:
                    logger.warning(
                        "managed group migration requires Owner review chat=%s disposition=%s",
                        chat_id[:12],
                        validation.value,
                    )

        for chat_id, project_id, status in (
            self._managed_group_registry.unreported_migration_dispositions()
        ):
            if status != "ambiguous":
                continue
            sent = remote.send_text_to_open_id(
                self._managed_group_owner_id,
                "受管群迁移需要人工处理："
                f"群 {chat_id}，项目 {project_id}，状态 {status}。"
                "请在 Owner 私聊中使用 /access migration-status 查看。",
            )
            if sent:
                self._managed_group_registry.mark_migration_reported(chat_id)


    def rotate_main_managed_group_bot(
        self,
        expected_bot_ref: str,
    ) -> tuple[int, int]:
        """CAS-rotate records for the explicitly identified former main Bot."""

        from ..project_chat.lark_chat_client import (
            LarkChatClient,
            ManagedChatValidation,
        )

        if not expected_bot_ref or expected_bot_ref == self._managed_group_receiving_bot_ref:
            return 0, 0
        remote = LarkChatClient(api_client_factory=self._get_api_client)
        rotated = 0
        rejected = 0
        for record in self._managed_group_registry.managed_groups():
            if (
                getattr(record.status, "value", "") != "active"
                or record.receiving_bot_ref != expected_bot_ref
            ):
                continue
            validation = remote.validate_managed_chat(
                record.chat_id,
                self._managed_group_owner_id,
            )
            if validation is not ManagedChatValidation.VALID:
                rejected += 1
                continue
            self._managed_group_registry.rotate_receiving_bot(
                chat_id=record.chat_id,
                expected_bot_ref=expected_bot_ref,
                new_bot_ref=self._managed_group_receiving_bot_ref,
            )
            rotated += 1
        return rotated, rejected

    # ==================================================================
    # WebSocket lifecycle
    # ==================================================================
    def _main_bot_reply_chain_readiness(self) -> tuple[bool, str]:
        """Return whether the connected main Bot can emit an audited reply."""

        handler_ctx = getattr(self, "_handler_ctx", None)
        handlers = getattr(handler_ctx, "handlers", None)
        try:
            reply_handler_ready = handlers is not None and "coco" in handlers
        except Exception:
            reply_handler_ready = False
        if not reply_handler_ready:
            return False, "main_bot_reply_handler"

        if not _visible_employee_runtime_requires_outbound_audit(
            getattr(self, "settings", None)
        ):
            return True, ""

        audit = getattr(handler_ctx, "main_bot_outbound_audit", None)
        audit_failure = getattr(
            handler_ctx,
            "main_bot_outbound_audit_failure",
            None,
        )
        if audit is _unavailable_main_bot_outbound_audit or not callable(audit):
            return False, "main_bot_outbound_audit"
        if not callable(audit_failure):
            return False, "main_bot_outbound_audit_failure"
        return True, ""

    def _try_publish_restart_readiness(self) -> bool:
        """Publish restart readiness once transport and reply dependencies are live."""

        if (
            getattr(self, "_closed", False)
            or not getattr(self, "_main_ws_connected", False)
        ):
            return False
        ready, blocker = self._main_bot_reply_chain_readiness()
        if not ready:
            if getattr(self, "_restart_readiness_blocker", None) != blocker:
                self._restart_readiness_blocker = blocker
                logger.error(
                    "GhostAP restart readiness withheld: blocker=%s",
                    blocker,
                )
            return False

        lock = getattr(self, "_restart_readiness_lock", None)
        if lock is None:
            lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
            self._restart_readiness_lock = lock
        with lock:
            if getattr(self, "_restart_generation", None) is not None:
                return True
            self._restart_generation = self._restart_gate.mark_ready(
                service_pid=os.getpid()
            )
            self._restart_readiness_blocker = None
            logger.info(
                "GhostAP restart readiness published generation=%s",
                self._restart_generation,
            )
        return True

    def _record_ws_activity(self, kind: str) -> None:
        """Record transport health and publish only reply-capable readiness."""

        self._ws_health_monitor.record_activity(kind)
        if kind == "disconnected":
            self._main_ws_connected = False
            return
        if kind != "connected":
            return
        self._main_ws_connected = True
        self._start_employee_runtime_recovery()
        self._try_publish_restart_readiness()

    def _publish_restart_participation(self) -> str:
        """Bind restart identity only when the real service starts intake."""

        if self._restart_participation_id is None:
            self._restart_participation_id = (
                self._restart_gate.publish_participation(
                    service_pid=os.getpid(),
                )
            )
        return self._restart_participation_id

    def _build_event_handler(self):
        """Build the main Bot dispatcher with every subscribed event consumed."""

        event_builder = (
            ChannelEventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._handle_message)
            .register_p2_im_message_reaction_created_v1(_ignore_ws_event)
            .register_p2_im_message_reaction_deleted_v1(_ignore_ws_event)
            .register_p2_im_chat_member_bot_added_v1(_ignore_ws_event)
            .register_p2_im_chat_member_bot_deleted_v1(self._handle_bot_deleted)
            .register_p2_im_message_message_read_v1(_ignore_ws_event)
            .register_p2_card_action_trigger(self._handle_card_action_callback)
        )
        # lark-channel-sdk 1.1.0 omits the typed p2p-chat-entered registrar,
        # while Feishu still delivers the subscribed event. Falling back to a
        # customized processor preserves the previous no-op/ACK behavior and
        # prevents the SDK from returning 500 (which triggers event retries).
        register_chat_entered = getattr(
            event_builder,
            "register_p2_im_chat_access_event_bot_p2p_chat_entered_v1",
            None,
        )
        if callable(register_chat_entered):
            event_builder = register_chat_entered(_ignore_ws_event)
        else:
            event_builder = event_builder.register_p2_customized_event(
                "im.chat.access_event.bot_p2p_chat_entered_v1",
                _ignore_ws_event,
            )
        return event_builder.build()

    def _restore_trusted_ingress_dependencies(self, root: str) -> None:
        """Restore managed-group authority before WS admission opens."""

        try:
            self._reconcile_managed_groups_before_team_start()
        except Exception:
            logger.exception("Managed-group startup reconciliation failed")
            raise

    def start(self):
        """启动 WS 长连接并进入重连循环。

        注意：该方法是阻塞的；通常在主线程调用。
        """
        self._publish_restart_participation()
        event_handler = self._build_event_handler()

        self._message_cache.start_cleanup_thread()
        self._card_event_cache.start_cleanup_thread()
        self._ws_health_monitor.start_watchdog()
        self._start_main_slash_command_sync()

        # Registry reconciliation must precede marker-based Team restore.
        import os
        _root = os.getcwd()
        self._restore_trusted_ingress_dependencies(_root)

        logger.info("正在建立飞书长连接...")
        logger.info("多项目管理已启用")

        reconnect_delay = getattr(self.settings, "feishu_ws_reconnect_delay_s", 5.0)

        while not self._closed:
            if not self._main_bot_open_id and not self._sync_main_bot_identity():
                if self._closed:
                    break
                logger.warning(
                    "Main Bot identity unavailable; WS intake remains closed, "
                    "retrying in %.1fs",
                    reconnect_delay,
                )
                time.sleep(reconnect_delay)
                continue
            self._client = ObservedLarkWSClient(
                self.settings.app_id,
                self.settings.app_secret,
                event_handler=event_handler,
                # The SDK logs its credential-bearing connection URL at
                # INFO/DEBUG. Lifecycle health is observed through hooks, so
                # WARNING keeps diagnostics without persisting access_key or
                # ticket query parameters.
                log_level=ChannelLogLevel.WARNING,
                source="ghostap",
                on_activity=self._record_ws_activity,
                on_response_written=self._finish_card_advisories,
                on_handler_scheduled=self._begin_message_ingress_binding,
                on_handler_finished=self._finish_message_ingress_binding,
            )
            try:
                self._client.start()
            except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as e:
                classify_ws_error(e, phase="dispatch")
                if self._closed:
                    break
                logger.exception("飞书 WS 连接异常退出")

            if self._closed:
                break

            logger.warning("飞书 WS 连接已断开，%.1fs 后重连...", reconnect_delay)
            time.sleep(reconnect_delay)
