"""Feishu WebSocket 客户端（核心路由枢纽）。

职责概览：
- 接收飞书 WS 事件（消息、卡片动作、反应等）并做基础校验/去重。
- 解码并准入事件，将消息和卡片动作交给权威 handler。
- 通过 `TaskScheduler` 提供：按项目串行、全局并发限制、系统命令快通道、背压与熔断。

关键设计点：
- 兼容性：部分 lark-oapi 版本不包含完整的 callback model 类型；这里对仅用于类型标注的符号做了降级处理。
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import lark_oapi as lark

if TYPE_CHECKING:
    from lark_channel import FeishuChannel
    from lark_channel.event.callback.model.p2_card_action_trigger import (
        P2CardActionTrigger,
        P2CardActionTriggerResponse,
    )

    from .slash_commands import SlashPermissionRequired
    from .ws_lifecycle import ObservedLarkWSClient

    try:  # pragma: no cover
        from lark_channel.event.callback.model.p2_im_message_receive_v1 import P2ImMessageReceiveV1  # type: ignore
    except (ImportError, AttributeError):  # pragma: no cover
        P2ImMessageReceiveV1 = Any  # type: ignore

from ..access_control import (
    AccessOperation,
    IngressAccessPolicyProvider,
    build_ingress_access_policy,
)
from ..acp.manager import ACPSessionManager
from ..acp.telemetry import build_idle_health_config_for_manager
from ..agent.intent_recognizer import IntentRecognizer
from ..card.actions.dispatch import (
    WF_CONFIG_ACTIONS,
    WF_START_WORKFLOW,
    WORKFLOW_STOP_RUNNING,
)
from ..card.ui_text import UI_TEXT
from ..config import get_settings
from ..config.env_file_store import AtomicEnvFileStore
from ..deep_engine import DeepEngineManager, ProgressReporter
from ..mode import PROGRAMMING_MODE_VALUES
from ..mode.backends import ALL_BACKEND_METAS
from ..project import (
    MessageLinker,
    MessageProjectMapper,
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
    get_thread_manager,
    set_current_is_p2p,
    set_current_sender_id,
    set_current_sender_name,
    set_current_sender_union_id,
    set_current_tenant_key,
    set_current_thread_id,
)
from ..trust.models import TrustZone
from ..trust.registry import ManagedGroupRegistry, single_owner_id
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
from .emoji import EmojiReaction as EmojiReaction
from .handler_context import HandlerContext
from .handlers import (
    PROGRAMMING_HANDLER_TYPES,
    DeepHandler,
    DiagnosticsHandler,
    ProjectHandler,
    SpecHandler,
    SystemHandler,
    WorkflowHandler,
)
from .image_handler import FeishuImageHandler
from .message_cache import MessageCache
from .slash_command_parser import SlashCommandParser
from .ws_card_action_handler import (
    CardActionInspector,
    _extract_behavior_value,
    bind_managed_trust_revisions,
    classify_card_action_error,
)
from .ws_card_pipeline import (
    _MAX_PENDING_CARD_ADVISORIES,
    WSCardPipelineMixin,
    _CardAdvisoryDispatcher,
)
from .ws_event_router import MessageIngressGuard, classify_ws_error
from .ws_health import WSDisconnectOutcome, WSHealthMonitor
from .ws_managed_groups import WSManagedGroupsMixin
from .ws_message_pipeline import WSMessagePipelineMixin
from .ws_resource_manager import EngineResourceGroup
from .ws_slash_identity import WSSlashIdentityMixin
from .ws_trust_access import (
    _BOOTSTRAP_ACCESS_OPERATIONS,
    WSTrustAccessMixin,
)


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
_MAIN_WARNING_RETRY_DELAYS = (0.0, 0.05, 0.2)
_MAIN_BOT_IDENTITY_RETRY_SECONDS = 1.0
_MAIN_SLASH_SYNC_RETRY_SECONDS = 60.0


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


def _main_warning_uuid(message_id: str, label: str) -> str:
    digest = hashlib.sha256(f"{message_id}\0{label}".encode("utf-8")).hexdigest()
    return "main-warning-" + digest[:32]


def _ignore_ws_event(_data: object) -> None:
    """Acknowledge subscribed events that require no product action."""


_READONLY_CARD_ACTIONS = {
    "deep_expand", "deep_collapse", "deep_mode_full", "deep_mode_compact", "deep_expand_ac", "deep_collapse_ac",
    "spec_expand", "spec_collapse", "spec_mode_full", "spec_mode_compact", "spec_expand_ac", "spec_collapse_ac",
}

_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
_SHUTDOWN_SCHEDULER_DRAIN_S = 5.0
_SHUTDOWN_DELEGATED_DRAIN_S = 5.0
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


class FeishuWSClient(
    WSSlashIdentityMixin,
    WSTrustAccessMixin,
    WSMessagePipelineMixin,
    WSCardPipelineMixin,
    WSManagedGroupsMixin,
):
    """Feishu WS Client 的服务端运行态。

    该类面向"长连接服务"场景：
    - 内部会初始化 scheduler / handler / cache，并在 `start()` 后进入事件循环。
    - `close()` 提供 best-effort 资源回收（线程/缓存/调度器等）。
    """

    def __init__(
        self,
        message_callback: Callable[[str, str, str, Optional[str]], None],
    ):
        try:
            self._initialize(message_callback)
        except BaseException:
            self._close_scheduler_after_initialization_failure()
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
        self._api_client_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._main_bot_open_id = ""
        self._main_bot_identity_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._main_bot_identity_cond = threading.Condition(self._main_bot_identity_lock)
        self._main_bot_identity_inflight = False
        self._main_bot_identity_next_retry_at = 0.0
        self._channel_client: Optional["FeishuChannel"] = None
        self._slash_command_sync_thread: Optional[threading.Thread] = None
        self._slash_command_sync_wakeup = threading.Event()
        self._slash_permission_notice_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._slash_permission_notice_recipients: set[str] = set()
        self._slash_permission_notifications: set[tuple[str, str]] = set()
        self._slash_permission_unroutable_logs: set[str] = set()
        self._pending_slash_permission: SlashPermissionRequired | None = None
        self._main_ws_connected = False
        self._restart_readiness_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._restart_readiness_blocker: str | None = None
        self._channel_client_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # ACPSessionManager: IdleHealth 相关协作者统一通过 IdleHealthConfig 注入，
        # 避免在构造函数中直接依赖具体 Telemetry/Service 实现。
        idle_health_cfg = build_idle_health_config_for_manager()

        acp_managers = {
            spec.mode_key: ACPSessionManager(
                spec.mode_key,
                session_timeout=getattr(self.settings, spec.session_timeout_attr),
                keepalive_interval=self.settings.acp_keepalive_interval,
                idle_healthcheck_s=self.settings.acp_session_idle_healthcheck_s,
                idle_health_config=idle_health_cfg,
            )
            for spec in ALL_BACKEND_METAS
        }
        self._intent_recognizer = IntentRecognizer()
        self._message_cache = MessageCache(
            ttl=self.settings.message_cache_ttl,
            max_size=self.settings.message_cache_max_size,
            cleanup_interval=60,
            lazy_cleanup=True,
        )
        self._message_ingress_guard = MessageIngressGuard(
            message_cache=self._message_cache,
            message_expire_seconds=self.settings.message_expire_seconds,
        )
        self._card_event_cache = MessageCache(
            ttl=self.settings.message_cache_ttl,
            max_size=self.settings.message_cache_max_size,
            cleanup_interval=60,
            lazy_cleanup=True,
        )
        # Card action dedupe (user rapid clicks): short TTL, per-action key.
        self._card_action_dedup_cache = MessageCache(
            ttl=self.settings.card.action_dedup_ttl,
            max_size=self.settings.card.action_dedup_max_size,
            cleanup_interval=30,
            lazy_cleanup=True,
        )
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
        self._team_runtime = None

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

        # ------------------------------------------------------------------
        # Handler infrastructure
        # ------------------------------------------------------------------
        self._handler_ctx = HandlerContext(
            settings=self.settings,
            api_client_factory=self._get_api_client,
            message_callback=self.message_callback,
            managers=acp_managers,
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
            slash_permission_provider=self._current_main_slash_permission,
        )

        handler_types = {
            **PROGRAMMING_HANDLER_TYPES,
            "deep": DeepHandler,
            "spec": SpecHandler,
            "project": ProjectHandler,
            "system": SystemHandler,
            "diagnostics": DiagnosticsHandler,
            "workflow": WorkflowHandler,
        }
        handlers = {name: cls(self._handler_ctx) for name, cls in handler_types.items()}
        self._handler_ctx.handlers.update(handlers)
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

                    from ..card.builders.lock_common import sign_command
                    from ..card.ui_text import UI_TEXT
                    repo_name = _Path(root_path).name or root_path
                    _text = UI_TEXT["repo_lock_released_notify"].format(repo_name=repo_name)
                    for _cid in blocked_chat_ids:
                        try:
                            _btn_val = {"action": "retry_command", "_t": "/status", "_s": sign_command("/status", _cid)}
                            _btn = {
                                "tag": "button",
                                "text": {"tag": "plain_text", "content": "📊 查看状态"},
                                "type": "default",
                                "value": _btn_val,
                                "behaviors": [{"type": "callback", "value": dict(_btn_val)}],
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
                            _send_card(_cid, _card)
                        except Exception as _inner:
                            logger.debug("Failed to notify release to chat=%s: %s", _cid[:12], _inner)
                except Exception as _exc:
                    logger.warning("Failed to send lock release notifications: %s", _exc)

            repo_lock_mgr.on_release.subscribe(_notify_lock_released)

        # Initialize ChatLockGate (ingress-level chat-lock interception).
        from .chat_lock_gate import ChatLockGate
        _clm = getattr(self._handler_ctx, "chat_lock_manager", None)
        _lock_dedup = MessageCache(ttl=30, max_size=1_000, cleanup_interval=60, lazy_cleanup=True)
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

    def _disconnect_primary_ws_intake(self) -> tuple[bool, WSDisconnectOutcome]:
        """Disconnect the primary WS client and classify whether intake quiesced."""
        monitor = self._ws_health_monitor
        disconnect_with_outcome = getattr(monitor, "disconnect_with_outcome", None)
        if callable(disconnect_with_outcome):
            try:
                outcome = disconnect_with_outcome()
                if isinstance(outcome, WSDisconnectOutcome):
                    return (
                        outcome in {
                            WSDisconnectOutcome.DISCONNECTED,
                            WSDisconnectOutcome.NO_CONNECTION,
                        },
                        outcome,
                    )
            except Exception:
                logger.debug("failed to disconnect primary WS intake", exc_info=True)
                return False, WSDisconnectOutcome.FAILED
        try:
            quiesced = bool(monitor.disconnect())
            outcome = (
                WSDisconnectOutcome.DISCONNECTED
                if quiesced
                else WSDisconnectOutcome.FAILED
            )
            return quiesced, outcome
        except Exception:
            logger.debug("failed to disconnect primary WS intake", exc_info=True)
            return False, WSDisconnectOutcome.FAILED

    def _stop_active_programming_runs(self) -> None:
        """Request cooperative cancellation for any active programming runs."""
        handlers = getattr(getattr(self, "_handler_ctx", None), "handlers", None)
        if not isinstance(handlers, dict):
            return
        for handler in list(handlers.values()):
            runs_map = getattr(handler, "_active_programming_runs", None)
            if not isinstance(runs_map, dict) or not runs_map:
                continue
            runs_lock = getattr(handler, "_active_programming_runs_lock", None)
            if runs_lock is not None:
                with runs_lock:
                    active_runs = list(runs_map.values())
            else:
                active_runs = list(runs_map.values())
            for active_run in active_runs:
                prog_session = getattr(active_run, "prog_session", None)
                mark_stopping = getattr(prog_session, "mark_stopping", None)
                if callable(mark_stopping):
                    try:
                        mark_stopping()
                    except Exception:
                        logger.debug(
                            "Failed to mark active programming card stopping",
                            exc_info=True,
                        )
                run_session = getattr(active_run, "session", None)
                cancel_fn = getattr(run_session, "cancel", None)
                if callable(cancel_fn):
                    try:
                        try:
                            cancel_fn(wait=False)
                        except TypeError:
                            cancel_fn()
                    except Exception:
                        logger.debug(
                            "Failed to cancel active programming run session",
                            exc_info=True,
                        )

    @staticmethod
    def _collect_new_stopped_engines(
        group: EngineResourceGroup,
        existing: list[Any],
        seen_ids: set[int],
    ) -> list[Any]:
        """Stop any engines registered after the initial shutdown stop pass."""
        try:
            current = list(group.manager.list_engines())
        except Exception:
            logger.debug("failed to list %s engines", group.name, exc_info=True)
            return existing
        merged = list(existing)
        for engine in current:
            eid = id(engine)
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            merged.append(engine)
            try:
                if engine and getattr(engine, "is_running", False):
                    engine.stop()
            except Exception:
                logger.debug("failed to stop %s engine instance", group.name, exc_info=True)
        return merged

    def close(self) -> bool:
        """Fence intake, drain work, then clean up dependencies.

        ``False`` means a worker, WS disconnect, or card delivery did not drain
        within the bounded shutdown budget. In that case shared dependencies
        are left intact for the still-running callback and retry ``close()``,
        and the process-level caller must not tear down lock managers
        underneath it.
        """
        scheduler = getattr(self, "_scheduler", None)
        in_cb = getattr(scheduler, "is_in_completion_callback", None)
        if callable(in_cb) and in_cb() is True:
            raise RuntimeError(
                "FeishuWSClient.close() cannot be called from a scheduler completion callback"
            )

        self._closed = True
        slash_wakeup = getattr(self, "_slash_command_sync_wakeup", None)
        if slash_wakeup is not None:
            slash_wakeup.set()
        self._fence_message_ingress_bindings()
        dispatcher = getattr(self, "_card_advisory_dispatcher", None)
        if dispatcher is not None:
            dispatcher.close()

        self._ws_health_monitor.stop_watchdog()

        # Stop external intake before fencing the in-process scheduler.  This
        # prevents a reconnect callback from admitting fresh work during drain.
        ws_intake_quiesced, ws_disconnect_outcome = (
            self._disconnect_primary_ws_intake()
        )
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

        # Immediately request cooperative cancellation for running engines,
        # active programming runs, and scheduler workers BEFORE spending the
        # bounded drain window waiting for them to go idle.
        deep_resources = EngineResourceGroup("deep_engine", self._deep_engine_manager)
        spec_resources = EngineResourceGroup("spec_engine", self._spec_engine_manager)
        workflow_resources = EngineResourceGroup(
            "workflow_engine",
            self._workflow_engine_manager,
        )
        deep_engines = deep_resources.stop_running_engines()
        spec_engines = spec_resources.stop_running_engines()
        workflow_engines = workflow_resources.stop_running_engines()
        self._stop_active_programming_runs()
        seen_engine_ids = {
            id(engine)
            for engine in (*deep_engines, *spec_engines, *workflow_engines)
        }
        try:
            self._scheduler.cancel_active()
        except Exception:
            logger.debug("failed to cancel active scheduler tasks", exc_info=True)

        scheduler_drain_deadline = (
            time.monotonic() + _SHUTDOWN_SCHEDULER_DRAIN_S
        )
        scheduler_idle = False
        try:
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

        # Sweep any delegated engine registered during drain and wait under one
        # bounded deadline before destroying ACP sessions.
        deep_engines = self._collect_new_stopped_engines(
            deep_resources, deep_engines, seen_engine_ids
        )
        spec_engines = self._collect_new_stopped_engines(
            spec_resources, spec_engines, seen_engine_ids
        )
        workflow_engines = self._collect_new_stopped_engines(
            workflow_resources, workflow_engines, seen_engine_ids
        )
        delegated_deadline = time.monotonic() + _SHUTDOWN_DELEGATED_DRAIN_S
        delegated_idle = True
        delegated_idle = (
            EngineResourceGroup.wait_stopped(
                deep_engines,
                timeout_s=max(0.0, delegated_deadline - time.monotonic()),
            )
            and delegated_idle
        )
        delegated_idle = (
            EngineResourceGroup.wait_stopped(
                spec_engines,
                timeout_s=max(0.0, delegated_deadline - time.monotonic()),
            )
            and delegated_idle
        )
        delegated_idle = (
            EngineResourceGroup.wait_stopped(
                workflow_engines,
                timeout_s=max(0.0, delegated_deadline - time.monotonic()),
            )
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

        if not ws_intake_quiesced:
            logger.error(
                "primary WS disconnect did not quiesce (outcome=%s); "
                "preserving shared dependencies for retry",
                ws_disconnect_outcome.value,
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
            hooks_mod = sys.modules.get("src.card.hooks")
            hook_mgr = (
                getattr(hooks_mod, "_hook_executor_manager", None)
                if hooks_mod is not None
                else None
            )
            shutdown_hooks = getattr(hook_mgr, "shutdown", None)
            if callable(shutdown_hooks):
                shutdown_hooks()
        except Exception:
            logger.debug("hook executor shutdown in close() skipped", exc_info=True)

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
        self._client = None
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
                        "behaviors": [{"type": "callback", "value": {"action": "show_board"}}],
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
        client = getattr(self, "_api_client", None)
        if client is not None:
            return client
        lock = getattr(self, "_api_client_lock", None)
        if lock is None:
            lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
            self._api_client_lock = lock
        with lock:
            client = getattr(self, "_api_client", None)
            if client is None:
                import lark_oapi.core  # noqa: F401

                client = (
                    lark.Client.builder()
                    .app_id(self.settings.app_id)
                    .app_secret(self.settings.app_secret)
                    .log_level(lark.LogLevel.INFO)
                    .timeout(30)  # 30s timeout for all API calls (card delivery protection)
                    .build()
                )
                self._api_client = client
            return client

    def _get_channel_client(self) -> "FeishuChannel":
        """Return the process-shared Channel SDK client for programming cards.

        Inbound events use the dedicated raw Channel WebSocket client.  This
        capability instance is outbound-only, so webhook transport prevents
        accidental creation of a second WebSocket connection while retaining
        the SDK's async CardKit and message APIs.
        """
        with self._channel_client_lock:
            if self._channel_client is None:
                mod = sys.modules[__name__]
                channel_cls = getattr(mod, "FeishuChannel")
                log_level_cls = getattr(mod, "ChannelLogLevel")
                transport_cfg_cls = getattr(mod, "ChannelTransportConfig")
                timeout = float(self.settings.card.delivery_api_timeout)
                self._channel_client = channel_cls(
                    app_id=self.settings.app_id,
                    app_secret=self.settings.app_secret,
                    log_level=log_level_cls.WARNING,
                    transport=transport_cfg_cls(
                        kind="webhook",
                        http_timeout_seconds=timeout,
                    ),
                )
            return self._channel_client

    def _get_image_handler(self) -> FeishuImageHandler:
        """获取/创建图片处理器（解析 + 下载 + 生成引用文本）。"""
        if self._image_handler is None:
            self._image_handler = FeishuImageHandler(self._get_api_client, self.settings)
        return self._image_handler

    # ==================================================================
    # Core routing — these remain in ws_client.py
    # ==================================================================

    def _handle_message(self, data: P2ImMessageReceiveV1):
        """飞书消息事件入口：只做轻量前置判断，然后交给 scheduler 异步处理。"""
        ingress_facts = self._extract_canonical_ingress_facts(data)
        if ingress_facts is None:
            audit_logger.warning("INGRESS_MALFORMED_EVENT_REJECTED phase=intake")
            return
        message_id, chat_id, chat_type, _sender_id = ingress_facts
        is_p2p = chat_type == "p2p"
        mentioned_command = self._parse_uniquely_mentioned_group_command(data)
        if mentioned_command is not None:
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
            if mentioned_command.bot_open_id != main_bot_open_id:
                return

        effective_trust = self._resolve_effective_trust(
            sender_id=_sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
        )
        trust_decision = self._message_trust_access_decision(
            effective_trust,
            chat_type=chat_type,
        )
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

        text = self._extract_text_from_message(data)
        command_match = SlashCommandParser.parse(text)
        ingress_decision = trust_decision or self._decide_ingress_access(
            message_id=message_id,
            sender_id=_sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
            command_match=command_match,
        )
        if not ingress_decision.allowed:
            return
        if ingress_decision.operation is not AccessOperation.BOOTSTRAP_HELP:
            self._register_slash_permission_notice_recipient(
                sender_id=_sender_id,
                chat_type=chat_type,
            )
        if (
            ingress_decision.operation not in _BOOTSTRAP_ACCESS_OPERATIONS
            and not self._managed_ingress_action_allowed(
                effective_trust,
                text=text,
                command_match=command_match,
            )
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
        thread_ctx = None
        if ingress_decision.operation not in _BOOTSTRAP_ACCESS_OPERATIONS:
            try:
                parent_id = getattr(data.event.message, "parent_id", None)
                root_id = getattr(data.event.message, "root_id", None)
                thread_root_id = root_id

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

        if (
            ingress_decision.operation not in _BOOTSTRAP_ACCESS_OPERATIONS
            and managed_group is None
            and not project_id
        ):
            try:
                active = self._project_manager.get_active_project(chat_id)
                project_id = active.project_id if active else None
                if not isinstance(project_id, str):
                    project_id = None
            except (AttributeError, KeyError):
                project_id = None

        in_programming_mode = False
        if managed_group is None:
            if thread_ctx is not None and getattr(thread_ctx, "mode", None) in PROGRAMMING_MODE_VALUES:
                in_programming_mode = True
            else:
                try:
                    mode_check = self._mode_manager.is_programming_mode(
                        chat_id,
                        project_id=project_id,
                    )
                    in_programming_mode = mode_check is True
                except (AttributeError, KeyError, TypeError):
                    in_programming_mode = False

        has_leading_slash = bool(
            text and (text.lstrip().startswith("/") or SystemHandler.is_exit_command(text))
        )
        is_system = bool(
            text
            and SystemHandler.is_fast_lane_system_command(
                text,
                in_programming_mode=in_programming_mode,
            )
        )
        is_shell_fast = (
            False
            if has_leading_slash
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
            task_type = "spec_command" if is_spec else "feishu_message"
            if is_system:
                tl = (text or "").strip().lower()
                if tl in {"/help", "/帮助"}:
                    task_type = "system_help"
                elif tl in {"/exit", "/quit"}:
                    task_type = "system_exit"
                elif tl == "/stop":
                    task_type = "system_stop"
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
                    _reservation=reservation: self._run_reserved_message(
                        _reservation,
                        data,
                        task_ctx=ctx,
                        shell_fast_tracked=_sf,
                        effective_trust=_trust,
                    ),
                )
            except (
                RateLimitExceededException,
                CircuitBreakerOpenException,
                TaskQueueFullError,
            ) as e:
                logger.warning(f"Backpressure applied: {get_error_detail(e)}")
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

        if (
            action_type_preview in WF_CONFIG_ACTIONS
            and action_type_preview != WF_START_WORKFLOW
        ):
            if not self._current_trust_can_dispatch(effective_trust):
                return empty_card_action_response()
            if self._chat_lock_gate.check_card_action(
                open_chat_id,
                operator_id,
                open_message_id,
                action_type=action_type_preview,
            ):
                return empty_card_action_response()
            wf_handler = self._handler_ctx.handlers.get("workflow")
            sync_fn = (
                getattr(wf_handler, "handle_workflow_config_callback_sync", None)
                if wf_handler is not None
                else None
            )
            if callable(sync_fn):
                action_obj = data.event.action
                value_raw = getattr(action_obj, "value", None)
                behaviors = getattr(action_obj, "behaviors", None)
                if isinstance(behaviors, list) and behaviors:
                    behavior_value = _extract_behavior_value(behaviors[0])
                    if behavior_value is not None:
                        value_raw = behavior_value
                if isinstance(value_raw, dict):
                    sync_value = dict(value_raw)
                elif isinstance(value_raw, str):
                    try:
                        parsed_val = json.loads(value_raw)
                        sync_value = (
                            parsed_val
                            if isinstance(parsed_val, dict)
                            else {"action": value_raw}
                        )
                    except (json.JSONDecodeError, TypeError):
                        sync_value = {"action": value_raw}
                else:
                    sync_value = {"action": str(value_raw)}
                for attr, key in (
                    ("option", "_option"),
                    ("options", "_options"),
                    ("form_value", "_form_value"),
                    ("input_value", "_input_value"),
                ):
                    extra_val = getattr(action_obj, attr, None)
                    if extra_val is not None:
                        sync_value[key] = extra_val
                if project_id:
                    sync_value["project_id"] = project_id

                set_current_thread_id(None)
                try:
                    set_current_sender_id(operator_id)
                    set_current_sender_union_id(operator_union_id or None)
                    set_current_sender_name(operator_id[:8] if operator_id else "")
                    set_current_is_p2p(card_is_p2p)
                    set_current_tenant_key(tenant_key or None)
                    card_thread_id = sync_value.get("thread_root_id")
                    if (
                        isinstance(card_thread_id, str)
                        and card_thread_id
                        and self.settings.thread_programming_enabled
                    ):
                        thread_ctx = self._thread_manager.get(card_thread_id)
                        if (
                            thread_ctx is not None
                            and thread_ctx.chat_id == open_chat_id
                            and thread_ctx.project_id == (project_id or "")
                        ):
                            set_current_thread_id(thread_ctx.thread_root_id)
                    with TraceContext(request_id):
                        updated_card, toast_type, toast_content = sync_fn(
                            open_message_id,
                            open_chat_id,
                            project_id,
                            sync_value,
                        )
                except (
                    AttributeError,
                    KeyError,
                    OSError,
                    RuntimeError,
                    TimeoutError,
                    TypeError,
                    ValueError,
                ) as exc:
                    classification = classify_card_action_error(exc, phase="dispatch")
                    logger.error(
                        "Workflow 配置卡片同步回调失败 (action=%s): %s",
                        classification.action.value,
                        exc,
                        exc_info=True,
                    )
                    return build_card_action_response(
                        toast_type="error",
                        toast_content=f"配置更新失败: {get_error_detail(exc)}"[:120],
                    )
                finally:
                    set_current_thread_id(None)
                    set_current_sender_id(None)
                    set_current_sender_union_id(None)
                    set_current_sender_name("")
                    set_current_is_p2p(False)
                    set_current_tenant_key(None)

                if (
                    updated_card is not None
                    and effective_trust is not None
                    and effective_trust.zone is TrustZone.MANAGED_AGENT_GROUP
                ):
                    updated_card = bind_managed_trust_revisions(
                        updated_card,
                        group_revision=effective_trust.group_revision,
                        grant_revision=effective_trust.grant_revision,
                    )
                return build_card_action_response(
                    toast_type=toast_type,
                    toast_content=toast_content,
                    card=updated_card,
                )

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
        if action_type_preview == WF_START_WORKFLOW:
            return build_card_action_response(
                toast_type="info",
                toast_content="正在启动 Workflow…",
            )
        if action_type_preview == WORKFLOW_STOP_RUNNING:
            return build_card_action_response(
                toast_type="info",
                toast_content="正在停止 Workflow…",
            )
        return empty_card_action_response()


    # ==================================================================
    # WebSocket lifecycle
    # ==================================================================

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
        published_now = False
        with lock:
            if getattr(self, "_restart_generation", None) is not None:
                return True
            self._restart_generation = self._restart_gate.mark_ready(
                service_pid=os.getpid()
            )
            self._restart_readiness_blocker = None
            published_now = True
            logger.info(
                "GhostAP restart readiness published generation=%s",
                self._restart_generation,
            )
        if published_now:
            try:
                import linecache

                from ..utils.memory_tuning import trim_process_heap

                linecache.clearcache()
                gc.collect()
                trim_process_heap()
            except Exception:
                pass
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

        dispatcher_cls = getattr(sys.modules[__name__], "ChannelEventDispatcherHandler")
        event_builder = (
            dispatcher_cls.builder("", "")
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
        # Hydrate lark_oapi.core and lark_channel.core synchronously before spawning background threads
        # (core <-> core.model is unsafe under first-import races).
        import lark_channel.core  # noqa: F401
        import lark_oapi.core  # noqa: F401

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
            log_level_cls = getattr(sys.modules[__name__], "ChannelLogLevel")
            observed_cls = getattr(sys.modules[__name__], "ObservedLarkWSClient")
            self._client = observed_cls(
                self.settings.app_id,
                self.settings.app_secret,
                event_handler=event_handler,
                # The SDK logs its credential-bearing connection URL at
                # INFO/DEBUG. Lifecycle health is observed through hooks, so
                # WARNING keeps diagnostics without persisting access_key or
                # ticket query parameters.
                log_level=log_level_cls.WARNING,
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


_LAZY_CHANNEL_ATTRS = frozenset(
    {
        "FeishuChannel",
        "ChannelEventDispatcherHandler",
        "ChannelLogLevel",
        "ChannelTransportConfig",
    }
)


def __getattr__(name: str) -> Any:
    if name == "ObservedLarkWSClient":
        from .ws_lifecycle import ObservedLarkWSClient

        globals()["ObservedLarkWSClient"] = ObservedLarkWSClient
        return ObservedLarkWSClient
    if name == "reconcile_main_agent_slash_commands":
        from .main_slash_commands import reconcile_main_agent_slash_commands

        globals()["reconcile_main_agent_slash_commands"] = reconcile_main_agent_slash_commands
        return reconcile_main_agent_slash_commands
    if name == "ChannelEventDispatcherHandler":
        from lark_channel import EventDispatcherHandler as ChannelEventDispatcherHandler

        globals()["ChannelEventDispatcherHandler"] = ChannelEventDispatcherHandler
        return ChannelEventDispatcherHandler
    if name == "ChannelLogLevel":
        from lark_channel import LogLevel as ChannelLogLevel

        globals()["ChannelLogLevel"] = ChannelLogLevel
        return ChannelLogLevel
    if name == "ChannelTransportConfig":
        from lark_channel import TransportConfig as ChannelTransportConfig

        globals()["ChannelTransportConfig"] = ChannelTransportConfig
        return ChannelTransportConfig
    if name == "FeishuChannel":
        from lark_channel import FeishuChannel

        globals()["FeishuChannel"] = FeishuChannel
        return FeishuChannel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
