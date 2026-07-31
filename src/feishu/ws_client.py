"""Feishu WebSocket 客户端（核心路由枢纽）。

职责概览：
- 接收飞书 WS 事件（消息、卡片动作、反应等）并做基础校验/去重。
- 将用户消息路由到不同 handler（SMART/COCO/CLAUDE/SHELL/TTADK 以及 Deep/Spec 引擎）。
- 通过 `TaskScheduler` 提供：按项目串行、全局并发限制、系统命令快通道、背压与熔断。

关键设计点：
- `_FORWARDING_MAP` + `__getattr__`：把不同 mode 的实现解耦到 handlers 中，同时保持 ws_client 的调用面稳定。
- 兼容性：部分 lark-oapi 版本不包含完整的 callback model 类型；这里对仅用于类型标注的符号做了降级处理。
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
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
from ..agent.intent_recognizer import IntentRecognizer, IntentResult, TaskStep
from ..autonomous.provisioning.notification_state import (
    hire_notification_message_uuid,
)
from ..card.ui_text import UI_TEXT
from ..config import IngressAccessMode, get_settings
from ..config.env_file_store import AtomicEnvFileStore
from ..deep_engine import DeepEngineManager, ProgressReporter
from ..project import (
    ContextSourceMode,
    MessageLinker,
    MessageProjectMapper,
    ProjectContext,
    ProjectContextManager,
    ProjectManager,
)
from ..slock_engine import SlockEngineManager
from ..spec_engine import SpecEngineManager, SpecReporter
from ..tasking import TaskPriority, TaskScheduler, TaskSpec
from ..thread import (
    get_current_tenant_key,
    get_current_thread_id,
    get_thread_manager,
    set_current_tenant_key,
)
from ..trust.registry import ManagedGroupRegistry, single_owner_id
from ..utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException
from ..utils.errors import get_error_detail
from ..utils.rate_limit import RateLimiter, RateLimitExceededException
from ..utils.restart_gate import RestartGate
from ..utils.trace import TraceContext, configure_logging_with_trace
from ..worktree_engine.manager import WorktreeManager
from .action_dispatcher import ActionDispatcher
from .emoji import EmojiReaction
from .handler_context import HandlerContext
from .handlers import (
    AidenModeHandler,
    ClaudeModeHandler,
    CocoModeHandler,
    CodexModeHandler,
    DeepHandler,
    DiagnosticsHandler,
    GeminiModeHandler,
    ProjectHandler,
    SlockHandler,
    SpecHandler,
    SystemHandler,
    TraexModeHandler,
    TTADKModeHandler,
    Tui2acpModeHandler,
    WorkflowHandler,
)
from .handlers.worktree import WorktreeHandler
from .image_handler import FeishuImageHandler
from .main_slash_commands import reconcile_main_agent_slash_commands
from .message_cache import MessageCache
from .renderers.deep_renderer import DeepRenderer
from .renderers.spec_renderer import SpecRenderer
from .renderers.worktree_renderer import WorktreeRenderer
from .slash_command_parser import CommandMatch, SlashCommandParser
from .ws_card_action_handler import (
    CardActionInspector,
    _extract_behavior_value,
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


def _employee_hire_status_text(employee_name: str, status: str) -> str | None:
    if status == "polling":
        return (
            "独立飞书智能体注册请求已提交，正在等待你在上方链接中完成授权确认。"
            "确认前注册接口会持续返回 400 authorization_pending，这是设备授权"
            "流程的正常等待状态；请按链接完成授权，期间请勿重复发送 /hire。"
        )
    if status == "ready":
        return (
            f"✅ 员工 **{employee_name}** 配置完成，正在等待激活。"
            "请先私聊该员工发送 `/status`，激活成功后再将其加入 Slock 群。"
        )
    if status == "active":
        return f"✅ 员工 **{employee_name}** 已激活，可以加入 Slock 群协作。"
    if status == "action_required":
        return (
            f"⚠️ 员工 **{employee_name}** 创建未能自动收敛，已转为人工处理；"
            f"可使用 `/fire {employee_name}` 清理后重试。"
        )
    return None


def _employee_hire_status_uuid(intent_id: str, status: str) -> str:
    return hire_notification_message_uuid(intent_id, status)


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
    except Exception:
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

# Sentinel used to distinguish "caller didn't provide command_match" from
# "caller provided command_match=None". This ensures request-scoped SSOT:
# parse exactly once at WS ingress, then thread the result through.
_COMMAND_MATCH_MISSING: object = object()


_READONLY_CARD_ACTIONS = {
    "deep_expand", "deep_collapse", "deep_mode_full", "deep_mode_compact", "deep_expand_ac", "deep_collapse_ac",
    "spec_expand", "spec_collapse", "spec_mode_full", "spec_mode_compact", "spec_expand_ac", "spec_collapse_ac",
}

# Selection-flow actions where duplicate clicks are silently dropped (no toast).
# Rapid clicking during model/tool selection is normal UX — showing a toast
# every time is disruptive.
_SILENT_DEDUP_ACTIONS = {
    "workflow_select_tool", "workflow_orchestrator_select_tool",
    "workflow_orchestrator_select_model_group",
    "workflow_orchestrator_select_model_profile",
    "workflow_orchestrator_select_model_effort",
    "workflow_orchestrator_select_model", "workflow_review_select_tool",
    "workflow_review_select_model_group",
    "workflow_review_select_model_profile",
    "workflow_review_select_model_effort",
    "workflow_review_select_model", "worktree_select_tool",
    "worktree_select_model", "spec_review_select_tool",
    "spec_review_select_model", "select_ttadk_tool",
    "select_ttadk_model", "select_ttadk_combined",
    "select_ttadk_combined_tool", "select_acp_tool",
    "select_acp_model", "slock_new_role_select_tool",
    "slock_new_role_select_model",
    "slock_role_add_pick", "slock_role_add_confirm",
    "slock_role_add_select",
    "slock_new_role_select_tool_dropdown",
}

_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
_SHUTDOWN_SCHEDULER_DRAIN_S = 5.0
_SHUTDOWN_DELEGATED_DRAIN_S = 5.0


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
    scheduler = TaskScheduler(
        max_concurrent=settings.task_scheduler_max_concurrent,
        per_key_concurrency=settings.task_scheduler_per_key_concurrency,
        system_concurrency=settings.system_command_concurrency,
        thread_name_prefix="ghost_worker",
        run_guard=gate.task_guard,
    )
    scheduler._restart_gate = gate
    return scheduler


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
        self._api_client: Optional[lark.Client] = None
        self._channel_client: Optional[FeishuChannel] = None
        self._slash_command_sync_thread: Optional[threading.Thread] = None
        self._channel_client_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # ACPSessionManager: IdleHealth 相关协作者统一通过 IdleHealthConfig 注入，
        # 避免在构造函数中直接依赖具体 Telemetry/Service 实现。
        idle_health_cfg = build_idle_health_config_for_manager()

        self._coco_manager = ACPSessionManager(
            "coco",
            session_timeout=self.settings.coco_session_timeout,
            keepalive_interval=self.settings.acp_keepalive_interval,
            idle_healthcheck_s=self.settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self._claude_manager = ACPSessionManager(
            "claude",
            session_timeout=self.settings.claude_session_timeout,
            keepalive_interval=self.settings.acp_keepalive_interval,
            idle_healthcheck_s=self.settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self._aiden_manager = ACPSessionManager(
            "aiden",
            session_timeout=self.settings.coco_session_timeout,
            keepalive_interval=self.settings.acp_keepalive_interval,
            idle_healthcheck_s=self.settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self._codex_manager = ACPSessionManager(
            "codex",
            session_timeout=self.settings.coco_session_timeout,
            keepalive_interval=self.settings.acp_keepalive_interval,
            idle_healthcheck_s=self.settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self._gemini_manager = ACPSessionManager(
            "gemini",
            session_timeout=self.settings.coco_session_timeout,
            keepalive_interval=self.settings.acp_keepalive_interval,
            idle_healthcheck_s=self.settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self._traex_manager = ACPSessionManager(
            "traex",
            session_timeout=self.settings.coco_session_timeout,
            keepalive_interval=self.settings.acp_keepalive_interval,
            idle_healthcheck_s=self.settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self._ttadk_manager = ACPSessionManager(
            "ttadk",
            session_timeout=self.settings.coco_session_timeout,
            keepalive_interval=self.settings.acp_keepalive_interval,
            idle_healthcheck_s=self.settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self._tui2acp_manager = ACPSessionManager(
            "tui2acp",
            session_timeout=self.settings.coco_session_timeout,
            keepalive_interval=self.settings.acp_keepalive_interval,
            idle_healthcheck_s=self.settings.acp_session_idle_healthcheck_s,
            idle_health_config=idle_health_cfg,
        )
        self._intent_recognizer = IntentRecognizer()
        self._message_cache = MessageCache(ttl=self.settings.message_cache_ttl, max_size=self.settings.message_cache_max_size, cleanup_interval=60)
        self._message_ingress_guard = MessageIngressGuard(
            message_cache=self._message_cache,
            message_expire_seconds=self.settings.message_expire_seconds,
        )
        self._card_event_cache = MessageCache(ttl=self.settings.message_cache_ttl, max_size=self.settings.message_cache_max_size, cleanup_interval=60)
        # Card action dedupe (user rapid clicks): short TTL, per-action key.
        self._card_action_dedup_cache = MessageCache(ttl=self.settings.card.action_dedup_ttl, max_size=self.settings.card.action_dedup_max_size, cleanup_interval=30)
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
        self._slock_engine_manager = SlockEngineManager()

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
                slock_engine_manager=self._slock_engine_manager,
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
            coco_manager=self._coco_manager,
            claude_manager=self._claude_manager,
            aiden_manager=self._aiden_manager,
            codex_manager=self._codex_manager,
            gemini_manager=self._gemini_manager,
            traex_manager=self._traex_manager,
            ttadk_manager=self._ttadk_manager,
            tui2acp_manager=self._tui2acp_manager,
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
            slock_engine_manager=self._slock_engine_manager,
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

        # Instantiate handlers (temp locals for registry population)
        coco_handler = CocoModeHandler(self._handler_ctx)
        claude_handler = ClaudeModeHandler(self._handler_ctx)
        aiden_handler = AidenModeHandler(self._handler_ctx)
        codex_handler = CodexModeHandler(self._handler_ctx)
        gemini_handler = GeminiModeHandler(self._handler_ctx)
        traex_handler = TraexModeHandler(self._handler_ctx)
        ttadk_handler = TTADKModeHandler(self._handler_ctx)
        tui2acp_handler = Tui2acpModeHandler(self._handler_ctx)
        deep_handler = DeepHandler(self._handler_ctx)
        deep_handler.renderer = DeepRenderer(deep_handler)
        spec_handler = SpecHandler(self._handler_ctx)
        spec_handler.renderer = SpecRenderer(spec_handler)
        project_handler = ProjectHandler(self._handler_ctx)
        system_handler = SystemHandler(self._handler_ctx)
        worktree_handler = WorktreeHandler(self._handler_ctx)
        worktree_handler._renderer = WorktreeRenderer(worktree_handler)
        diagnostics_handler = DiagnosticsHandler(self._handler_ctx)
        slock_handler = SlockHandler(self._handler_ctx)
        workflow_handler = WorkflowHandler(self._handler_ctx)

        # ------------------------------------------------------------------
        # Populate registry containers in context
        # ------------------------------------------------------------------
        # Bind handlers directly on instance for backward compatibility (especially for tests)
        self._coco_handler = coco_handler
        self._claude_handler = claude_handler
        self._aiden_handler = aiden_handler
        self._codex_handler = codex_handler
        self._gemini_handler = gemini_handler
        self._traex_handler = traex_handler
        self._ttadk_handler = ttadk_handler
        self._tui2acp_handler = tui2acp_handler
        self._deep_handler = deep_handler
        self._spec_handler = spec_handler
        self._project_handler = project_handler
        self._system_handler = system_handler
        self._worktree_handler = worktree_handler
        self._diagnostics_handler = diagnostics_handler
        self._slock_handler = slock_handler
        self._workflow_handler = workflow_handler

        self._handler_ctx.managers.update({
            "coco": self._coco_manager,
            "claude": self._claude_manager,
            "aiden": self._aiden_manager,
            "codex": self._codex_manager,
            "gemini": self._gemini_manager,
            "traex": self._traex_manager,
            "ttadk": self._ttadk_manager,
            "tui2acp": self._tui2acp_manager,
        })
        self._handler_ctx.handlers.update({
            "coco": coco_handler,
            "claude": claude_handler,
            "aiden": aiden_handler,
            "codex": codex_handler,
            "gemini": gemini_handler,
            "traex": traex_handler,
            "ttadk": ttadk_handler,
            "tui2acp": tui2acp_handler,
            "deep": deep_handler,
            "spec": spec_handler,
            "project": project_handler,
            "system": system_handler,
            "worktree": worktree_handler,
            "diagnostics": diagnostics_handler,
            "slock": slock_handler,
            "workflow": workflow_handler,
        })

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
                        "config": {"wide_screen_mode": True},
                        "elements": [
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
                        ],
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
        self._chat_lock_gate = ChatLockGate(_clm, _lock_dedup, host=self)

        # Bind forwarding methods directly on instance (replaces __getattr__ dispatch)
        from .router import bind_forwarding_methods
        bind_forwarding_methods(self, self._handler_ctx)
        self._recover_employee_runtime_after_handler_binding()

        # ------------------------------------------------------------------
        # Control-plane (deferred /exit, system command gate)
        # ------------------------------------------------------------------
        from .control_plane import ControlPlane
        self._control_plane = ControlPlane(
            scheduler=self._scheduler,
            project_manager=self._project_manager,
            exit_handler_fn=lambda *a, **kw: self._exit_current_mode(*a, **kw),
        )
        self._scheduler.add_listener(self._control_plane.on_scheduler_event)
        # Backward-compat aliases for tests
        self._system_cmd_gate_lock = self._control_plane._system_cmd_gate_lock
        self._system_cmd_inflight_by_chat = self._control_plane._system_cmd_inflight_by_chat

        # --- Message Dispatcher ---
        from .dispatcher import MessageDispatcher
        self._message_dispatcher = MessageDispatcher(self)

        # --- Action Dispatcher ---
        from .action_registry import init_action_registry
        self._action_dispatcher = ActionDispatcher()
        init_action_registry(self)

        # Configure trace logging
        configure_logging_with_trace()

    def _recover_employee_runtime_after_handler_binding(self) -> None:
        """Start durable recovery only after the main-Bot reply path exists."""

        runtime = self._employee_department_runtime
        if runtime is None:
            return
        try:
            if not callable(getattr(self, "_reply_text", None)):
                raise RuntimeError("main Bot reply transport is not bound")
            runtime.recover()
        except Exception:
            self._close_employee_runtime_after_initialization_failure()
            raise

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

    def _register_action(self, handler: Callable, exact: Optional[str] = None, prefix: Optional[str] = None):
        """Register a card action handler."""
        self._action_dispatcher.register(handler, exact, prefix)

    def close(self) -> bool:
        """Fence intake, drain work, then clean up dependencies.

        ``False`` means a worker or card delivery did not drain within the
        bounded shutdown budget. In that case shared dependencies are left
        intact for the still-running callback and the process-level caller must
        not tear down lock managers underneath it.
        """
        self._closed = True

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

        scheduler_idle = False
        try:
            scheduler_idle = self._scheduler.wait_for_idle(
                _SHUTDOWN_SCHEDULER_DRAIN_S
            )
            if not scheduler_idle:
                self._scheduler.cancel_active()
                scheduler_idle = self._scheduler.wait_for_idle(
                    _SHUTDOWN_SCHEDULER_DRAIN_S
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

        # Ask every delegated execution surface to stop before destroying ACP
        # sessions.  Employee runtime close waits for its Team executor; Slock
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
        slock_engines = self._quiesce_slock_activities()

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
        delegated_idle = (
            self._wait_slock_activities(
                slock_engines,
                timeout_s=_SHUTDOWN_DELEGATED_DRAIN_S,
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

        from ..card.delivery.registry import delivery_registry

        if not delivery_registry.drain_in_flight(
            timeout=_SHUTDOWN_DELEGATED_DRAIN_S
        ):
            logger.error(
                "card delivery did not drain; preserving shared dependencies"
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
        try:
            self._slock_engine_manager.cleanup_all()
        except Exception as e:
            logger.debug("清理slock_engine_manager失败: %s", get_error_detail(e))

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

    def _quiesce_slock_activities(self) -> list[Any]:
        """Cancel Slock work delegated beyond the message scheduler."""

        try:
            engines = list(self._slock_engine_manager.list_engines())
        except Exception:
            logger.debug("failed to list Slock engines during shutdown", exc_info=True)
            return []
        for engine in engines:
            try:
                pause = getattr(engine, "pause", None)
                stop = getattr(engine, "stop", None)
                if callable(pause):
                    pause()
                elif callable(stop):
                    stop()
            except Exception:
                logger.debug("failed to quiesce Slock engine", exc_info=True)
            seen_executors: set[int] = set()
            for name in ("_executor", "_discussion_executor"):
                executor = getattr(engine, name, None)
                if executor is None or id(executor) in seen_executors:
                    continue
                seen_executors.add(id(executor))
                shutdown = getattr(executor, "shutdown", None)
                if not callable(shutdown):
                    continue
                try:
                    shutdown(wait=False, cancel_futures=True)
                except Exception:
                    logger.debug(
                        "failed to fence Slock executor %s",
                        name,
                        exc_info=True,
                    )
        return engines

    @staticmethod
    def _wait_slock_activities(
        engines: list[Any],
        *,
        timeout_s: float,
    ) -> bool:
        """Boundedly observe Slock delegated executors after cancellation."""

        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            pending = 0
            for engine in engines:
                for name in ("_executor", "_discussion_executor"):
                    executor = getattr(engine, name, None)
                    count = getattr(executor, "pending_count", 0)
                    if isinstance(count, int) and count > 0:
                        pending += count
            if pending == 0:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error(
                    "Slock delegated activities did not drain pending=%s",
                    pending,
                )
                return False
            time.sleep(min(0.05, remaining))

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
                self.reply(evicted_chat_id, card_json, msg_type=msg_type, chat_id=evicted_chat_id)
            except Exception as send_err:
                # Fallback to plain text
                try:
                    msg = UI_TEXT["ws_project_eviction_notify"].format(name=project_name)
                    self.reply(evicted_chat_id, msg, msg_type="text", chat_id=evicted_chat_id)
                except Exception:
                    logger.debug("failed to send eviction fallback notification", exc_info=True)
                logger.warning("Failed to send LRU eviction notification to %s: %s", evicted_chat_id[:12], send_err)

        threading.Thread(target=_send_notification, daemon=True).start()

    def _is_message_expired(self, create_time: int) -> bool:
        """判断消息是否过期。

        飞书历史消息可能会被 WS 重放；这里通过 `create_time` 过滤掉过旧消息，
        避免触发重复执行（尤其是 shell/编程任务）。
        """
        return self._message_ingress_guard.is_message_expired(create_time)

    def _is_duplicate_message(self, message_id: str) -> bool:
        """消息去重：基于 `MessageCache` 判断是否重复处理。"""
        return self._message_ingress_guard.is_duplicate_message(message_id)

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
    # Handler forwarding dispatch
    # ==================================================================
    # Maps ``client._xxx(...)`` calls to the corresponding handler
    # method.  This replaces 50+ one-liner stubs with a single
    # ``__getattr__`` lookup, keeping backward compatibility with tests
    # that mock ``client._enter_coco_mode`` etc.
    # ------------------------------------------------------------------

    def reply(self, message_id: str, content, msg_type: str = "text", chat_id: Optional[str] = None):
        """轻量回复封装：兼容旧调用路径，按 msg_type 委托到对应的新 API。"""
        if chat_id is not None:
            logger.warning("chat_id 参数已废弃且不再生效，请移除该参数")
        if msg_type == "text":
            self._reply_text(message_id, content)
        else:
            self._reply_card(message_id, content)

    def add_reaction(self, message_id: str, emoji_type: str):
        """轻量表情反馈封装：委托到 handler 的 `add_reaction`。"""
        self._add_reaction(message_id, emoji_type)

    def send_lock_conflict_card(
        self,
        e,
        message_id: str,
        command_text: str,
        *,
        retry_count: int = 0,
        chat_id: str = "",
    ) -> None:
        """Public facade: send a repo-lock conflict card via the system handler.

        Delegates to ``SystemHandler.send_lock_conflict_card`` obtained via
        ``_get_handler("system")``, consistent with other handler access
        patterns (e.g. ``_switch_project``).
        """
        handler = self._get_handler("system")
        if handler:
            handler.send_lock_conflict_card(
                e,
                message_id,
                command_text,
                retry_count=retry_count,
                chat_id=chat_id,
            )
        else:
            from .handlers.lock_helper import logger as _lock_logger
            _lock_logger.warning("send_lock_conflict_card: _system_handler unavailable, cannot notify user")
            # Fallback: send plain text notification
            self._reply_text(message_id, f"🔒 {str(e) or 'lock conflict'}")

    def _get_handler(self, key: str) -> Any:
        return self._handler_ctx.handlers.get(key)

    def _switch_project(self, message_id: str, chat_id: str, name: str, auto_enter_coco: bool = True):
        """切换当前 chat 的 active project，并可选自动进入 Coco 模式。"""
        project_handler = self._get_handler("project")
        if project_handler:
            project_handler.switch_project(
                message_id,
                chat_id,
                name,
                auto_enter_coco=auto_enter_coco,
                coco_handler=self._get_handler("coco"),
                claude_handler=self._get_handler("claude"),
            )

    @staticmethod
    def _is_exit_command(text: str) -> bool:
        """判断是否为“退出当前编程模式”的命令（跨模式一致）。"""
        return SystemHandler.is_exit_command(text)

    @staticmethod
    def _is_deep_command(text: str) -> bool:
        """判断是否为 Deep Engine 命令。"""
        return SystemHandler.is_deep_command(text)

    @staticmethod
    def _is_spec_command(text: str) -> bool:
        """判断是否为 Spec Engine 命令。"""
        return SystemHandler.is_spec_command(text)

    @staticmethod
    def _is_workflow_command(text: str) -> bool:
        """判断是否为 Workflow Engine 命令。"""
        return SystemHandler.is_workflow_command(text)

    def _is_slock_command(self, text: str, chat_id: str = "") -> "bool | str":
        """判断是否为 Slock Engine 命令。"""
        from ..slock_engine.slash_commands import is_slock_command
        manager = getattr(self, '_slock_engine_manager', None)
        return is_slock_command(text, chat_id=chat_id, manager=manager)

    def _is_slock_active(self, chat_id: str) -> bool:
        """Check if a chat has an active slock engine."""
        manager = getattr(self, '_slock_engine_manager', None)
        if manager is None:
            return False
        return manager.is_slock_active(chat_id)

    def _is_slock_managed_chat(self, chat_id: str) -> bool:
        """Check if a chat is registered as managed by the slock engine."""
        manager = getattr(self, '_slock_engine_manager', None)
        if manager is None:
            return False
        return manager.is_managed_chat(chat_id)

    # ------------------------------------------------------------------
    # Passive mode auto-activate helpers
    # ------------------------------------------------------------------

    _chat_locks: dict[str, threading.Lock] = {}
    _chat_locks_meta: dict[str, float] = {}  # chat_id → last_used timestamp
    _chat_locks_guard = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def _get_chat_lock(self, chat_id: str) -> threading.Lock:
        """Get or create a per-chat activation lock."""
        with self._chat_locks_guard:
            if chat_id not in self._chat_locks:
                self._chat_locks[chat_id] = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
            self._chat_locks_meta[chat_id] = time.time()
            return self._chat_locks[chat_id]

    @classmethod
    def _gc_chat_locks(cls, max_age: float = 300.0) -> int:
        """Remove chat locks unused for more than max_age seconds."""
        now = time.time()
        removed = 0
        with cls._chat_locks_guard:
            stale = [
                cid for cid, ts in cls._chat_locks_meta.items()
                if now - ts > max_age
            ]
            for cid in stale:
                # Only remove if lock is not currently held
                lock = cls._chat_locks.get(cid)
                if lock and not lock.locked():
                    del cls._chat_locks[cid]
                    del cls._chat_locks_meta[cid]
                    removed += 1
        return removed

    def _should_auto_activate_slock(self, chat_id: str, text: str, *, chat_type: str = "group") -> bool:
        """Check if message should trigger slock auto-activation.

        Short-circuits for already-managed chats to avoid redundant classification
        overhead. Only performs task classification for unmanaged group chats.

        Returns True if:
        - Chat is already managed by slock (short-circuit), or
        - Chat is an unmanaged group AND text is classified as a task
        """
        from .slock_dispatch import should_auto_activate

        return should_auto_activate(
            chat_id,
            text,
            chat_type=chat_type,
            is_managed=self._is_slock_managed_chat(chat_id),
        )

    def _auto_activate_slock(
        self, chat_id: str, text: str, project: "Optional[ProjectContext]" = None
    ) -> tuple[bool, str]:
        """Auto-activate slock for an unmanaged chat on first valid task message.

        Returns a tuple of (success, reason):
        - success: True if activation succeeded, False if denied/failed.
        - reason: A string indicating the result. One of:
            - ACTIVATION_ALLOWED: activation succeeded
            - ACTIVATION_DENIED_RATE_LIMIT: rate limit exceeded
            - ACTIVATION_DENIED_ADMIN_REQUIRED: admin-only policy
            - ACTIVATION_DENIED_NOT_WHITELISTED: not in whitelist
            - "error": activation failed with exception

        Idempotent: if the chat becomes managed between check and call, the
        slock handler's activate_slock will detect the existing engine and
        short-circuit. Uses _auto_activate_lock to prevent concurrent bootstrap
        for the same chat.

        Guarded by ActivationGuard for permission and rate-limit checks.
        """
        from .slock_dispatch import try_passive_activation

        return try_passive_activation(
            chat_id,
            text,
            project=project,
            settings=self.settings,
            is_managed_fn=lambda: self._is_slock_managed_chat(chat_id),
            get_chat_lock_fn=lambda: self._get_chat_lock(chat_id),
            slock_handler=self._slock_handler,
            card_sender=self._system_handler,
        )

    @staticmethod
    def _is_interceptable_command_match(command_match: CommandMatch | None) -> bool:
        """SSOT variant: decide based on request-scoped CommandMatch."""
        return SystemHandler.is_interceptable_command_match(command_match)

    def _is_worktree_awaiting_goal(self, project: "ProjectContext") -> bool:
        """Return True when worktree journey is awaiting a goal.

        具体判定逻辑下沉到 ``WorktreeManager.is_awaiting_goal``，避免在
        WS 层拼装布尔条件，统一依赖 WorktreeRuntimeState / journey 状态机。
        """

        if not getattr(project, "project_id", None):
            return WorktreeManager.is_awaiting_goal(getattr(project, "worktree_state", None))
        try:
            state = self._worktree_handler._worktree_manager().get_state(project)
        except Exception:
            state = getattr(project, "worktree_state", None)
        return WorktreeManager.is_awaiting_goal(state)

    @staticmethod
    def _mode_to_context_source(mode) -> ContextSourceMode:
        """将 `InteractionMode` 映射到 `ContextSourceMode`（用于统一上下文记录）。"""
        from ..mode import InteractionMode

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
                    if _proj_mode.value in {"coco", "claude", "aiden", "codex", "gemini", "traex", "ttadk"}:
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

    def _handle_message(self, data: P2ImMessageReceiveV1):
        """飞书消息事件入口：只做轻量前置判断，然后交给 scheduler 异步处理。"""
        ingress_facts = self._extract_canonical_ingress_facts(data)
        if ingress_facts is None:
            audit_logger.warning("INGRESS_MALFORMED_EVENT_REJECTED phase=intake")
            return
        message_id, chat_id, chat_type, _sender_id = ingress_facts
        is_p2p = chat_type == "p2p"

        _raw_sender_union_id = getattr(
            getattr(getattr(data.event, "sender", None), "sender_id", None),
            "union_id", None,
        )
        _sender_union_id = (
            _raw_sender_union_id if isinstance(_raw_sender_union_id, str) else ""
        )
        _raw_tenant_key = getattr(getattr(data, "header", None), "tenant_key", None)
        tenant_key = _raw_tenant_key if isinstance(_raw_tenant_key, str) else ""

        # The event trust root was validated before this content/image parser.
        # Authorization still precedes project/thread lookup, origin linking,
        # scheduler submission, image download, Shell, and all handlers.
        text = self._extract_text_from_message(data)
        command_match = SlashCommandParser.parse(text)
        ingress_decision = self._decide_ingress_access(
            message_id=message_id,
            sender_id=_sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
            command_match=command_match,
        )
        if not ingress_decision.allowed:
            return

        project_id = None
        thread_root_id = None
        try:
            parent_id = getattr(data.event.message, "parent_id", None)
            root_id = getattr(data.event.message, "root_id", None)
            thread_root_id = root_id
            thread_ctx = None

            if root_id and self.settings.thread_programming_enabled:
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

            if not project_id:
                for ref in (parent_id, root_id):
                    if ref:
                        project_id = self._message_mapper.get_project_id(ref)
                        if not isinstance(project_id, str):
                            project_id = None
                        if project_id:
                            break
        except (AttributeError, KeyError, TypeError):
            project_id = None

        if not project_id:
            try:
                active = self._project_manager.get_active_project(chat_id)
                project_id = active.project_id if active else None
                if not isinstance(project_id, str):
                    project_id = None
            except (AttributeError, KeyError):
                project_id = None

        is_system = bool(
            text and (text.startswith("/") or self._is_exit_command(text))
        )
        is_shell_fast = (
            False
            if is_system
            else SystemHandler.is_likely_shell_command(text)
        )
        is_spec = self._is_spec_command(text) if text else False

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

        request_id = self._ensure_request_id(message_id, chat_id=chat_id, project_id=project_id)
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
            try:
                handle = self._scheduler.submit(
                    spec,
                    lambda ctx, _sf=is_shell_fast: self._process_message_async(
                        data,
                        task_ctx=ctx,
                        shell_fast_tracked=_sf,
                    ),
                )
            except (RateLimitExceededException, CircuitBreakerOpenException) as e:
                logger.warning(f"Backpressure applied: {get_error_detail(e)}")
                if is_spec:
                    self._reply_text(message_id, UI_TEXT["ws_backpressure_spec"])
                else:
                    self._reply_text(message_id, UI_TEXT["ws_backpressure_generic"])
                return
            try:
                if message_id:
                    self._message_linker.link_task(message_id, handle.run_id)
            except (KeyError, AttributeError, RuntimeError) as e:
                logger.debug("link_task失败(message): message_id=%s, run_id=%s, err=%s", message_id, handle.run_id, get_error_detail(e))

    def _is_system_command_message(self, data: P2ImMessageReceiveV1) -> bool:
        """Check if the message is a system command that should bypass project queue.

        All slash commands (``/xxx``) are system commands: they should never
        block behind long-running Coco/Claude programming tasks on the project
        queue.  This includes ``/stop_deep``, ``/exit``, ``/spec_status``, etc.
        """
        text = self._extract_text_from_message(data)
        if not text:
            return False
        return text.startswith("/") or self._is_exit_command(text)

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

    @staticmethod
    def _is_programming_entry_command(text: str) -> bool:
        """是否为编程模式初始化命令（用于与 /spec 串行化控制面执行）。"""
        from .product_catalog import is_programming_entry_command

        return is_programming_entry_command(text)

    def _build_control_queue_key(self, *, chat_id: str, project_id: Optional[str], text: str) -> Optional[str]:
        """为编程初始化与 spec 命令构造串行控制队列 key。"""
        normalized = (text or "").strip()
        if not normalized:
            return None
        if not (self._is_spec_command(normalized) or self._is_programming_entry_command(normalized) or self._is_workflow_command(normalized)):
            return None
        queue_suffix = project_id or "default"
        return f"{chat_id}:control:{queue_suffix}"

    def _is_likely_shell_command_message(self, data: P2ImMessageReceiveV1) -> bool:
        """Check if the message looks like a shell command for early routing."""
        text = self._extract_text_from_message(data)
        return SystemHandler.is_likely_shell_command(text) if text else False

    def _process_message_async(
        self,
        data: P2ImMessageReceiveV1,
        task_ctx=None,
        shell_fast_tracked: bool = False,
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

            # Parsing is allowed only after the current event facts pass their
            # independent worker check. No downstream fact is trusted from the
            # queued TaskSpec.
            image_handler = self._get_image_handler()
            parse_result = image_handler.parse_message(message.message_type, message.content)
            text = self._clean_at_text(parse_result.text)

            # Slash parsing is request-scoped: parse once and reuse.
            # This match becomes the single source of truth for downstream
            # slash consumers (gate/system/worktree).
            try:
                command_match = SlashCommandParser.parse(text)
            except Exception:
                command_match = None

            ingress_decision = self._decide_ingress_access(
                message_id=message_id,
                sender_id=_sender_id,
                chat_id=chat_id,
                chat_type=chat_type,
                command_match=command_match,
            )
            if not ingress_decision.allowed:
                return

            request_id = self._ensure_request_id(message_id, chat_id=chat_id)

            # Validation follows authorization so denied traffic cannot mutate
            # duplicate/expiry state or trigger unsupported-content replies.
            if not self._validate_message(message, request_id):
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
                self._system_handler.handle_intercepted_command(
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
            if root_id and self.settings.thread_programming_enabled:
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
            if parse_result.image_keys:
                project, auto_enter_mode, text, is_image_only = self._handle_image_content(
                    message, parse_result.image_keys, text, request_id, task_ctx
                )
                # Downloaded image references are part of the effective prompt.
                # Refresh slash args so consumers such as Worktree receive the
                # same evidence-rich goal as text-based engine handlers.
                if command_match is not None:
                    enriched_match = SlashCommandParser.parse(text)
                    if enriched_match is not None and enriched_match.command == command_match.command:
                        command_match = enriched_match
            else:
                # 4. Resolve Context (if no images to drive it)
                project, auto_enter_mode = self._resolve_message_context(message)

            # 4b. Safety net: if auto_enter_mode is still None but we are in a
            # registered thread, force-set mode from thread context so that the
            # message never accidentally falls through to SMART / intent recognition.
            if not auto_enter_mode:
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
                        if _tctx.mode in {"worktree", "deep", "spec", "workflow"}:
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
            )

        except asyncio.TimeoutError as e:
            logger.warning("处理消息超时: %s", get_error_detail(e))
            try:
                self._reply_text(message_id, UI_TEXT["ws_message_timeout"])
            except (RuntimeError, OSError, TimeoutError, TypeError, ValueError):
                classify_ws_error(RuntimeError("reply timeout failed"), phase="dispatch")
                logger.debug("failed to reply timeout message", exc_info=True)
        except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as e:
            classification = classify_ws_error(e, phase="dispatch")
            if classification.action == WSErrorAction.REPLY_INTERNAL_ERROR:
                logger.error("处理消息异常: %s", get_error_detail(e), exc_info=True)
                try:
                    self._reply_text(message_id, UI_TEXT["ws_message_internal_error"])
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

    def _validate_message(self, message, request_id: str) -> bool:
        """校验消息是否需要处理（过期/重复/类型不支持等）。"""
        if message.create_time and self._is_message_expired(int(message.create_time)):
            logger.debug("跳过过期消息: %s", message.message_id)
            return False

        if self._is_duplicate_message(message.message_id):
            logger.debug("跳过重复消息: %s", message.message_id)
            return False

        supported_types = {"text", "image", "post"}
        if message.message_type not in supported_types:
            self._reply_text(message.message_id, UI_TEXT["ws_unsupported_msg_type"])
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

    def _handle_image_content(self, message, image_keys, text, request_id, task_ctx):
        """处理图片消息：下载并把图片引用文本拼接回 prompt。

        返回 `(project, auto_enter_mode, text, is_image_only)`。
        """
        message_id = message.message_id
        chat_id = message.chat_id
        getattr(message, "parent_id", None)
        getattr(message, "root_id", None)

        with self._pending_image_lock:
            self._pending_image_keys[message_id] = image_keys

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
            self._get_working_dir(chat_id),
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
                if not project and thread_ctx.mode not in {"worktree", "deep", "spec", "workflow"}:
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
        """Return True when the current Feishu topic is owned by Deep/Spec/WT."""
        thread_id = get_current_thread_id()
        if not thread_id or not self.settings.thread_programming_enabled:
            return False
        thread_ctx = self._thread_manager.get(thread_id)
        return bool(thread_ctx and thread_ctx.mode in {"worktree", "deep", "spec", "workflow"})

    def _get_mode_handler(self, mode):
        from ..mode import InteractionMode
        _map = {
            InteractionMode.COCO: self._coco_handler,
            InteractionMode.CLAUDE: self._claude_handler,
            InteractionMode.AIDEN: self._aiden_handler,
            InteractionMode.CODEX: self._codex_handler,
            InteractionMode.GEMINI: self._gemini_handler,
            InteractionMode.TRAEX: self._traex_handler,
            InteractionMode.TTADK: self._ttadk_handler,
            InteractionMode.TUI2ACP: self._tui2acp_handler,
        }
        return _map.get(mode)

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
        from ..mode import InteractionMode

        _pid = project.project_id if project else None
        if not _pid and task_ctx and task_ctx.spec.project_id:
            _pid = task_ctx.spec.project_id

        current_mode, _is_prog = self._get_effective_mode(chat_id, project_id=_pid)
        if current_mode in {
            InteractionMode.COCO,
            InteractionMode.CLAUDE,
            InteractionMode.AIDEN,
            InteractionMode.CODEX,
            InteractionMode.GEMINI,
            InteractionMode.TRAEX,
            InteractionMode.TTADK,
            InteractionMode.TUI2ACP,
        }:
            if project is None:
                project = self._project_manager.get_active_project(chat_id)

            handler = self._get_mode_handler(current_mode)
            if handler:
                handler.handle_message(message_id, chat_id, "", project)
        else:
            self._show_help(message_id, chat_id)

    def _dispatch_message_logic(
        self,
        message_id,
        chat_id,
        text,
        project,
        auto_enter_mode,
        *,
        command_match=_COMMAND_MATCH_MISSING,
        is_image_only=False,
        shell_fast_tracked=False,
        chat_type: str = "group",
    ):
        """根据 auto-enter 与当前模式，将消息路由到对应编程模式或 SMART 处理路径。"""
        # Compatibility: some unit tests call _dispatch_message_logic directly.
        # In the real message ingress path, command_match is always provided.
        if command_match is _COMMAND_MATCH_MISSING:
            try:
                command_match = SlashCommandParser.parse(text)
            except Exception:
                command_match = None

        missing_topic_project_safe_commands = {
            "/help",
            "/projects",
            "/project",
            "/status",
            "/exit",
            "/quit",
        }
        command = command_match.command if command_match is not None else ""
        args = command_match.args.lower() if command_match is not None else ""
        is_global_deep_stop = command == "/stop_deep" and args in {"all", "-a", "--all"}
        is_global_deep_status = command == "/deep_status" and args in {"all", "-a", "--all"}
        is_missing_topic_exit = self._is_exit_command(text)
        can_recover_missing_topic = (
            command in missing_topic_project_safe_commands
            or is_global_deep_stop
            or is_global_deep_status
            or is_missing_topic_exit
        )
        if (
            auto_enter_mode in {"worktree", "deep", "spec", "workflow"}
            and project is None
            and not can_recover_missing_topic
        ):
            self._reply_text(message_id, UI_TEXT["ws_topic_project_unavailable"])
            return
        if (
            auto_enter_mode in {"worktree", "deep", "spec", "workflow"}
            and project is None
            and is_missing_topic_exit
        ):
            self._exit_current_mode(message_id, chat_id, project=None)
            return

        if auto_enter_mode:
            if self._reply_if_topic_engine_switch_blocked(
                message_id,
                auto_enter_mode,
                command_match=command_match,
            ):
                return
            if self._is_exit_command(text):
                self._add_reaction(message_id, EmojiReaction.on_coco_mode())
                _pid = project.project_id if project else None
                if self._control_plane.should_defer_exit(chat_id=chat_id, project_id=_pid):
                    self._control_plane.request_deferred_exit(message_id=message_id, chat_id=chat_id, project_id=_pid)
                    self._reply_text(message_id, UI_TEXT["ws_exit_deferred_msg"])
                    return
                self._exit_current_mode(message_id, chat_id, project=project)
                return
            # Interceptable system commands (/wt, /worktree, /help, /status, /codex, etc.)
            # must be routed to the system handler even inside thread programming mode,
            # otherwise they can be hidden behind same-mode/topic-hint handling.
            if self._is_interceptable_command_match(command_match):
                self._process_with_intent(
                    message_id,
                    chat_id,
                    text,
                    project,
                    command_match=command_match,
                    shell_fast_tracked=shell_fast_tracked,
                    chat_type=chat_type,
                )
                return
            from .product_catalog import is_same_programming_mode_entry

            if is_same_programming_mode_entry(auto_enter_mode, text):
                self._reply_text(
                    message_id,
                    UI_TEXT["ws_topic_hint_msg"],
                )
                return
            if self._is_programming_entry_command(text):
                self._reply_text(
                    message_id,
                    UI_TEXT["ws_topic_hint_msg"],
                )
                return
            if (
                self._is_deep_command(text)
                or self._is_spec_command(text)
                or self._is_workflow_command(text)
            ):
                self._process_with_intent(
                    message_id,
                    chat_id,
                    text,
                    project,
                    command_match=command_match,
                    shell_fast_tracked=shell_fast_tracked,
                    chat_type=chat_type,
                )
                return
        if auto_enter_mode in {"worktree", "deep", "spec", "workflow"}:
            if command_match is not None:
                self._process_with_intent(
                    message_id,
                    chat_id,
                    text,
                    project,
                    command_match=command_match,
                    shell_fast_tracked=shell_fast_tracked,
                    chat_type=chat_type,
                )
                return
            self._add_reaction(message_id, EmojiReaction.on_processing())
            if auto_enter_mode == "worktree":
                self._handle_worktree_execute(message_id, chat_id, text, project)
            elif auto_enter_mode == "deep":
                self._start_deep_engine(message_id, chat_id, text, project)
            elif auto_enter_mode == "workflow":
                self._workflow_handler.handle_message(message_id, chat_id, text, project)
            else:
                self._start_spec_engine(message_id, chat_id, text, project)
            return

        if auto_enter_mode and auto_enter_mode in {"coco", "claude", "aiden", "codex", "gemini", "traex", "ttadk"}:
            from ..mode import InteractionMode
            handler = self._get_mode_handler(InteractionMode(auto_enter_mode))
            if handler:
                self._add_reaction(message_id, EmojiReaction.on_coco_mode())
                self._add_reaction(message_id, EmojiReaction.on_processing())
                handler.handle_message(message_id, chat_id, text, project)
            else:
                self._process_with_intent(
                    message_id,
                    chat_id,
                    text,
                    project,
                    command_match=command_match,
                    shell_fast_tracked=shell_fast_tracked,
                    chat_type=chat_type,
                )
        else:
            # Project-chat default: when the chat is bound to a project via
            # /new-chat and the message is neither a slash command, a shell-like
            # invocation, nor an image-only message, route free-form text into
            # the Coco programming flow (model-select card + pending prompt).
            # Slash commands (command_match is not None) always fall through to
            # _process_with_intent so that /coco, /help, /deep, /wt, /exit, ...
            # keep their highest priority.
            if (
                command_match is None
                and not is_image_only
                and text
                and not self._intent_recognizer.looks_like_shell(text)
            ):
                bound_project = self._project_manager.find_by_bound_chat_id(chat_id)
                if bound_project is not None:
                    bound_project_id = getattr(bound_project, "project_id", None)
                    current_mode, is_programming = self._get_effective_mode(
                        chat_id, project_id=bound_project_id
                    )
                    if is_programming:
                        self._process_with_intent(
                            message_id,
                            chat_id,
                            text,
                            bound_project,
                            command_match=command_match,
                            shell_fast_tracked=shell_fast_tracked,
                            chat_type=chat_type,
                        )
                        return

                    default_tool = str(
                        getattr(bound_project, "acp_tool_name", None)
                        or getattr(self.settings, "default_acp_tool", None)
                        or "coco"
                    ).strip().lower()
                    saved_tool = str(getattr(bound_project, "acp_tool_name", None) or "").strip().lower()
                    self._add_reaction(message_id, EmojiReaction.on_coco_mode())
                    self._add_reaction(message_id, EmojiReaction.on_processing())
                    if saved_tool in {"coco", "claude", "aiden", "codex", "gemini", "traex"}:
                        self._system_handler.handle_enter_acp_saved_selection(
                            message_id,
                            chat_id,
                            saved_tool,
                            bound_project,
                            pending_prompt=text,
                        )
                    elif default_tool == "coco":
                        self._message_dispatcher._handle_enter_coco(
                            message_id, chat_id, bound_project, pending_prompt=text,
                        )
                    elif default_tool in {"codex", "traex"}:
                        self._message_dispatcher._handle_enter_acp_mode(
                            default_tool, message_id, chat_id, bound_project, pending_prompt=text,
                        )
                    else:
                        self._message_dispatcher._handle_enter_coco(
                            message_id, chat_id, bound_project, pending_prompt=text,
                        )
                    return
            self._process_with_intent(
                message_id,
                chat_id,
                text,
                project,
                command_match=command_match,
                shell_fast_tracked=shell_fast_tracked,
                chat_type=chat_type,
            )

    @staticmethod
    def _requested_topic_engine(command_match) -> Optional[str]:
        command = getattr(command_match, "command", None)
        if command in {"/worktree", "/wt"}:
            return "worktree"
        if command in {"/deep", "/deep_update", "/deep_status", "/stop_deep"}:
            return "deep"
        if command in {
            "/spec",
            "/spec_status",
            "/spec_history",
            "/spec_metrics",
            "/spec_config",
            "/spec_export",
            "/spec_save",
            "/spec_pause",
            "/spec_resume",
            "/spec_recover",
            "/spec_guide",
            "/stop_spec",
        }:
            return "spec"
        from ..workflow_engine.commands import TOPIC_ENGINE_COMMANDS as _WF_CMDS
        if command in _WF_CMDS:
            return "workflow"
        return None

    @staticmethod
    def _engine_display_name(engine: str) -> str:
        return {
            "worktree": "WT",
            "deep": "Deep",
            "spec": "Spec",
            "workflow": "WF",
        }.get(engine, engine)

    def _reply_if_topic_engine_switch_blocked(
        self,
        message_id: str,
        current_engine: str,
        *,
        command_match=None,
    ) -> bool:
        requested = self._requested_topic_engine(command_match)
        if not requested or requested == current_engine:
            return False
        if current_engine not in {"worktree", "deep", "spec", "workflow"}:
            return False
        self._reply_text(
            message_id,
            UI_TEXT["topic_engine_switch_blocked"].format(
                current=self._engine_display_name(current_engine),
                requested=self._engine_display_name(requested),
            ),
        )
        return True

    def _handle_card_action(self, data: P2CardActionTrigger) -> Optional[P2CardActionTriggerResponse]:
        """飞书卡片回调入口：做去重 + 任务入队（system action 走快通道）。"""
        try:
            header = data.header
            event_id = header.event_id
            if self._card_event_cache.is_duplicate(event_id):
                logger.warning("跳过重复卡片回调事件: %s", event_id)
                return None

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
        try:
            open_message_id = data.event.context.open_message_id
            open_chat_id = data.event.context.open_chat_id
        except (AttributeError, TypeError):
            open_message_id = None
            open_chat_id = "unknown"

        _raw_tenant_key = getattr(getattr(data, "header", None), "tenant_key", None)
        tenant_key = _raw_tenant_key if isinstance(_raw_tenant_key, str) else ""

        action_type_preview = ""
        try:
            action_type_preview = CardActionInspector.action_type(data.event.action)
        except (AttributeError, TypeError, ValueError):
            classify_card_action_error(RuntimeError("action preview failed"), phase="payload_parse")
            action_type_preview = ""

        try:
            with self._system_cmd_gate_lock:
                inflight = int(self._system_cmd_inflight_by_chat.get(open_chat_id, 0) or 0)
            if inflight > 0 and action_type_preview not in _READONLY_CARD_ACTIONS:
                if open_message_id:
                    self._reply_text(open_message_id, UI_TEXT["ws_system_cmd_gate_blocked"])
                return None
        except (RuntimeError, OSError, TypeError, ValueError):
            classify_card_action_error(RuntimeError("system command gate failed"), phase="dispatch")
            logger.debug("failed to check system command gate", exc_info=True)

        operator_id = ""
        operator_union_id = ""
        try:
            operator = data.event.operator
            operator_id = (
                getattr(operator, "open_id", None)
                or getattr(operator, "user_id", None)
                or getattr(operator, "union_id", None)
                or ""
            )
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
                    if action_type_preview in _SILENT_DEDUP_ACTIONS:
                        return {}
                    return {"toast": {"type": "info", "content": UI_TEXT["card_session_toast_dedup"]}}


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
                    return {"toast": {"type": "warning", "content": "撤销窗口已过期，请使用 /unlock 解锁"}}
        except (json.JSONDecodeError, TypeError, ValueError):
            classify_card_action_error(RuntimeError("undo payload parse failed"), phase="payload_parse")

        project_id = None
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
        card_is_p2p = False
        if not origin_lookup_failed:
            card_is_p2p = self._resolve_card_is_p2p(
                origin_message_id=origin_message_id,
                open_chat_id=open_chat_id,
                operator_id=operator_id,
            )
        # Provenance has already been resolved above. Request bookkeeping must
        # not overwrite its trusted chat/operator fields after a rejected card.
        request_id = self._ensure_request_id(origin_message_id, project_id=project_id)

        is_system = self._is_system_card_action(data)

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
            handle = self._scheduler.submit(spec, lambda ctx: self._process_card_action_async(data, task_ctx=ctx))
            try:
                self._message_linker.link_task(origin_message_id, handle.run_id)
            except (KeyError, AttributeError, RuntimeError) as e:
                logger.debug(
                    "link_task失败(card_action): origin=%s, run_id=%s, err=%s", origin_message_id, handle.run_id, e
                )
        return None

    def _resolve_card_is_p2p(
        self,
        *,
        origin_message_id: str | None,
        open_chat_id: str,
        operator_id: str,
    ) -> bool:
        """Resolve card DM provenance without trusting callback payload fields.

        Card callbacks do not carry a structural chat type. Prefer metadata
        captured from ``im.message.receive_v1``; after a process restart, fall
        back to the Chat API's ``chat_mode`` field. Any provenance mismatch or
        lookup failure is denied.
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

        if not isinstance(operator_id, str) or not operator_id:
            return False
        chat_mode = self._get_chat_mode(open_chat_id)
        if chat_mode is None:
            return False
        try:
            registered = self._message_linker.register_trusted_origin_if_absent(
                origin_message_id,
                chat_id=open_chat_id,
                chat_type="topic_group" if chat_mode == "topic" else chat_mode,
                sender_id=operator_id,
            )
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            logger.warning("failed to persist Chat API provenance", exc_info=True)
            return False
        return registered is True and chat_mode == "p2p"

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
            return self._reply_text(
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
            reply_id = self._reply_text(
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

    @classmethod
    def _card_action_dedup_fingerprint(cls, action: Any) -> str:
        """Return a stable fingerprint for the concrete card interaction payload."""
        return CardActionInspector.dedup_fingerprint(action)

    @staticmethod
    def _normalize_card_action_dedup_value(value: Any) -> Any:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, TypeError, ValueError):
                return value
            return parsed
        return value

    def _is_system_card_action(self, data: P2CardActionTrigger) -> bool:
        """Check if the card action is a system action that should bypass project queue."""
        try:
            return CardActionInspector.is_system_action(data.event.action)
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
            return False

    def _process_card_action_async(self, data: Any, task_ctx=None):
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
            project_id = value.get("project_id", "")

            card_thread_id = value.get("thread_root_id")
            if card_thread_id and self.settings.thread_programming_enabled:
                thread_ctx = self._thread_manager.get(card_thread_id)
                if thread_ctx:
                    set_current_thread_id(card_thread_id)

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

            # --- Dispatch via ActionDispatcher ---
            matched = self._action_dispatcher.dispatch(action_type, open_message_id, open_chat_id, project_id, value)

            if not matched:
                logger.warning("未注册的卡片动作: action=%s, message_id=%s", action_type, open_message_id)
                self._reply_text(open_message_id, f"⚠️ 未识别的操作: {action_type}")

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.debug("卡片回调处理耗时: %dms", elapsed_ms)

        except asyncio.TimeoutError as e:
            logger.warning("处理卡片动作超时: %s", get_error_detail(e))
            _mid = locals().get("open_message_id", "unknown")
            _action = locals().get("action_type", "unknown")
            try:
                if _mid != "unknown":
                    if str(_action).startswith("ttadk") or _action in {
                        "show_ttadk_menu",
                        "select_ttadk_tool",
                        "select_ttadk_model",
                        "refresh_ttadk_models",
                        "toggle_ttadk_yolo",
                    }:
                        try:
                            from ..card import CardBuilder

                            _pid = locals().get("project_id")
                            msg_type, card_content = CardBuilder.build_ttadk_soft_failure_card_for(
                                "操作超时",
                                project_id=_pid or None,
                            )
                            self._reply_card(_mid, card_content)
                        except Exception:
                            self._reply_text(_mid, "⏳ 操作超时，请稍后重试或发送 /ttadk 重新进入")
                    else:
                        self._reply_text(_mid, f"⏳ 操作超时 ({_action}): {get_error_detail(e)}")
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
                    if str(_action).startswith("ttadk") or _action in {
                        "show_ttadk_menu",
                        "select_ttadk_tool",
                        "select_ttadk_model",
                        "refresh_ttadk_models",
                        "toggle_ttadk_yolo",
                    }:
                        try:
                            from ..card import CardBuilder

                            project_id = locals().get("project_id")
                            msg_type, card_content = CardBuilder.build_ttadk_soft_failure_card_for(
                                "操作未完成",
                                project_id=project_id or None,
                            )
                            self._reply_card(_mid, card_content)
                        except Exception:
                            self._reply_text(_mid, "⚠️ 操作未完成，请稍后重试或发送 /ttadk 重新进入")
                    else:
                        self._reply_text(_mid, f"❌ 操作失败 ({_action}): {get_error_detail(e)}")
            except Exception:
                logger.debug("failed to reply action failure error", exc_info=True)
        finally:
            set_current_thread_id(None)
            set_current_sender_id(None)
            set_current_sender_union_id(None)
            set_current_sender_name("")
            set_current_is_p2p(False)
            set_current_tenant_key(None)

    def _process_with_intent(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional[ProjectContext] = None,
        *,
        command_match=_COMMAND_MATCH_MISSING,
        shell_fast_tracked: bool = False,
        chat_type: str = "group",
    ):
        """SMART 模式下的主路由：控制命令优先，其次进入意图识别/多任务执行。"""
        # Compatibility: allow callers outside ws message ingress to omit command_match.
        if command_match is _COMMAND_MATCH_MISSING:
            try:
                command_match = SlashCommandParser.parse(text)
            except Exception:
                command_match = None
        from .dispatcher import FeishuRequestContext

        self._message_dispatcher.process_request(
            FeishuRequestContext(
                message_id=message_id,
                chat_id=chat_id,
                text=text,
                project=project,
                command_match=command_match,
                shell_fast_tracked=shell_fast_tracked,
                chat_type=chat_type,
            )
        )

    def _execute_multi_tasks(
        self, message_id: str, chat_id: str, intent_result: IntentResult, project: Optional[ProjectContext] = None
    ):
        """执行多任务计划（逐步执行；遇到失败停止后续步骤）。"""
        self._message_dispatcher.execute_multi_tasks(message_id, chat_id, intent_result, project=project)

    def _execute_single_task(
        self,
        message_id: str,
        chat_id: str,
        task: Optional[TaskStep],
        original_text: str,
        project: Optional[ProjectContext] = None,
        *,
        shell_fast_tracked: bool = False,
    ):
        """执行单一任务步骤（模式切换/系统命令/引擎命令/执行 shell 等）。"""
        self._message_dispatcher.execute_single_task(
            message_id, chat_id, task, original_text, project=project, shell_fast_tracked=shell_fast_tracked
        )

    def _execute_task_step(
        self,
        message_id: str,
        chat_id: str,
        task: TaskStep,
        step_num: int,
        total_steps: int,
        project: Optional[ProjectContext] = None,
    ) -> bool:
        """执行一个 TaskStep，并返回是否成功。"""
        return self._message_dispatcher.execute_task_step(
            message_id, chat_id, task, step_num, total_steps, project=project
        )

    def _get_task_description(self, task: TaskStep) -> str:
        """为 TaskStep 生成可读描述（用于多任务计划展示）。"""
        return self._message_dispatcher.get_task_description(task)

    # ==================================================================
    # Event stubs (no-op)
    # ==================================================================
    def _handle_reaction_created(self, data):
        """飞书 reaction 事件回调（当前无需处理，保留占位）。"""
        pass

    def _handle_chat_entered(self, data):
        """飞书 chat entered 事件回调（当前无需处理，保留占位）。"""
        pass

    def _handle_message_read(self, data):
        """飞书 message read 事件回调（当前无需处理，保留占位）。"""
        pass

    def _handle_bot_deleted(self, data):
        """Durably revoke trust before retiring a remotely deleted group."""

        event = getattr(data, "event", None)
        chat_id = getattr(event, "chat_id", "")
        if not isinstance(chat_id, str) or not chat_id.startswith("oc_"):
            return
        manager = self._slock_engine_manager
        try:
            record = self._managed_group_registry.record(chat_id)
            revoke_required = (
                record is not None
                and getattr(record.status, "value", "") == "active"
            )
            if revoke_required:
                self._managed_group_registry.begin_revoke(
                    chat_id,
                    requested_at=datetime.now(UTC),
                )
            manager.retire_deleted_chat(chat_id)
            if not self._project_manager.revoke_managed_chat(chat_id):
                raise OSError("project managed-chat revocation was not persisted")
            if revoke_required:
                self._managed_group_registry.complete_revoke(chat_id)
        except Exception:
            logger.exception(
                "failed to retire deleted Slock chat=%s",
                chat_id[:12],
            )
            raise

    def _reconcile_managed_groups_before_slock_restore(self) -> None:
        """Reconcile pending revokes and validated legacy Project candidates."""

        from ..project_chat.lark_chat_client import (
            LarkChatClient,
            ManagedChatValidation,
        )

        # Resolve an anchored pre/post-replace state before interpreting grants
        # or running any compensation against remote chats.
        self._managed_group_registry.reconcile_uncertain_commit()
        remote = LarkChatClient(api_client_factory=self._get_api_client)
        for chat_id in self._managed_group_registry.pending_revokes():
            marker_disposition = (
                self._slock_engine_manager.prepare_revoke_marker(chat_id)
            )
            result = remote.delete_chat(chat_id)
            if result is False:
                if self._slock_engine_manager.restore_revoke_marker(
                    marker_disposition
                ):
                    self._managed_group_registry.cancel_revoke(chat_id)
                continue
            if not self._project_manager.revoke_managed_chat(chat_id):
                continue
            self._managed_group_registry.complete_revoke(chat_id)

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
    def _record_ws_activity(self, kind: str) -> None:
        """Record transport health and publish readiness on a real connection."""

        self._ws_health_monitor.record_activity(kind)
        if kind != "connected":
            return
        self._restart_generation = self._restart_gate.mark_ready(
            service_pid=os.getpid()
        )
        logger.info(
            "GhostAP restart readiness published generation=%s",
            self._restart_generation,
        )

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
            .register_p2_im_message_reaction_created_v1(self._handle_reaction_created)
            .register_p2_im_chat_member_bot_deleted_v1(self._handle_bot_deleted)
            .register_p2_im_message_message_read_v1(self._handle_message_read)
            .register_p2_card_action_trigger(self._handle_card_action)
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
            event_builder = register_chat_entered(self._handle_chat_entered)
        else:
            event_builder = event_builder.register_p2_customized_event(
                "im.chat.access_event.bot_p2p_chat_entered_v1",
                self._handle_chat_entered,
            )
        return event_builder.build()

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

        # Registry reconciliation must precede marker-based Slock restore.
        import os
        _root = os.getcwd()
        try:
            self._reconcile_managed_groups_before_slock_restore()
        except Exception:
            logger.exception("Managed-group startup reconciliation failed")
            raise

        # Restore slock engines from persisted marker files
        try:
            restored = self._slock_engine_manager.restore_from_disk(
                _root,
                managed_group_active=lambda chat_id: (
                    self._managed_group_registry.active_record(chat_id) is not None
                ),
            )
            if restored:
                logger.info("Restored %d slock engine(s) from disk", restored)
        except (OSError, ValueError, KeyError):
            logger.warning("Failed to restore slock engines from disk", exc_info=True)

        runtime = self._employee_department_runtime
        membership = (
            getattr(runtime, "membership_service", None)
            if runtime is not None
            else None
        )
        reconcile_memberships = getattr(
            membership,
            "reconcile_projected_memberships",
            None,
        )
        if callable(reconcile_memberships):
            summary = reconcile_memberships()
            removed = int(getattr(summary, "removed", 0) or 0)
            degraded = int(getattr(summary, "degraded", 0) or 0)
            if removed or degraded:
                logger.warning(
                    "Employee membership startup audit reconciled "
                    "removed=%d degraded=%d",
                    removed,
                    degraded,
                )

        logger.info("正在建立飞书长连接...")
        logger.info("多项目管理已启用")

        reconnect_delay = getattr(self.settings, "feishu_ws_reconnect_delay_s", 5.0)

        while not self._closed:
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
