"""ACP Session Manager — manages per-chat, per-project ACP sessions.

Sessions are keyed by (chat_id, project_id) to ensure full isolation between
projects within the same chat.  When project_id is not provided, a default
suffix is used for backward compatibility.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from .. import agent_session as _agent_session_mod
from ..agent_session import SyncClaudeCLISession, SyncSession, SyncTTADKCLISession
from ..config import get_settings
from ..utils.errors import get_error_detail
from . import startup_utils as _startup_utils
from .diagnostics import (
    get_diagnostics_config,
    redact_text,
    truncate_text,
)
from .helper import SessionKeyCodec
from .session import ACPResumeRejected
from .sync_adapter import SyncACPSession, build_startup_diagnostics
from .telemetry import (
    IdleHealthConfig,
    IdleHealthTelemetryContext,
    TelemetryAdapter,
    classify_manager_idle_health,
    resolve_idle_health_collaborators_for_manager,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionReplacementResult:
    """Outcome of an identity-guarded session replacement."""

    session: SyncSession | None
    created: bool


@dataclass
class _ClosingSessionState:
    """Per-key close completion and sticky failure tombstone."""

    completed: threading.Event
    error: BaseException | None = None


def _session_matches_requested_model(
    session: object,
    model_name: str,
) -> bool:
    """Compare model state without assuming it is encoded in process argv."""
    requested = str(model_name or "").strip()
    if not requested:
        return True
    active_model = getattr(session, "_model_name", None)
    if isinstance(active_model, str) and active_model.strip():
        return active_model.strip() == requested
    existing_args = getattr(session, "_agent_args", None)
    return requested in " ".join(existing_args or [])


def _normalize_manager_acp_model(agent_type: str, model_name: Optional[str]) -> Optional[str]:
    agent = (agent_type or "").strip().lower()
    if (
        not model_name
        or agent in {"claude", "traex"}
        or agent.startswith("ttadk_")
    ):
        return model_name
    try:
        from .providers import normalize_acp_model_name

        normalized = normalize_acp_model_name(agent, model_name)
        if normalized != model_name:
            logger.info(
                "[ACP:%s] normalized selected model for backend: selected=%s backend=%s",
                agent.upper(),
                model_name,
                normalized,
            )
        return normalized
    except Exception:
        logger.debug("ACPSessionManager model normalization failed", exc_info=True)
        return model_name


if TYPE_CHECKING:
    # 仅用于类型检查：避免在运行时将内部协议/实现暴露为公开 API。
    from ..utils.time_ago import IdleHealth, TimeAgoBucket
    from .telemetry import _IdleHealthServiceProtocol as IdleHealthServiceProtocol
    from .telemetry import _IdleHealthTelemetry as IdleHealthTelemetry


# Preserve the original TTADK CLI session class so that we can
# distinguish between tests patching src.acp.manager.SyncTTADKCLISession
# and those patching src.agent_session.SyncTTADKCLISession.
try:  # best-effort, never raise during import
    _ORIG_TTADK_CLI_SESSION = _agent_session_mod.SyncTTADKCLISession
except Exception:  # pragma: no cover - extremely unlikely
    _ORIG_TTADK_CLI_SESSION = None


def _format_ttadk_startup_attempts(diagnostics: object, *, per_item_limit: int = 300, total_limit: int = 1600) -> str:
    """TTADK 启动 attempts 摘要（compat wrapper）。

    说明：脱敏/截断/配置读取的 SSOT 在 `src.acp.diagnostics`，本函数仅做 best-effort 薄封装，
    以保持现有日志调用点与函数名稳定。
    """
    try:
        from .diagnostics import format_attempts_summary

        diag = diagnostics if isinstance(diagnostics, dict) else {}
        attempts = (diag.get("attempts") or []) if isinstance(diag, dict) else []
        return format_attempts_summary(
            attempts, per_item_limit=per_item_limit, total_limit=total_limit, get_settings_fn=get_settings
        )
    except Exception:
        logger.warning("Error while formatting TTADK startup attempts", exc_info=True)
        return ""


def _sanitize_startup_detail(text: str) -> str:
    """Redact and truncate startup detail for safe logging/user-facing errors."""
    s = str(text or "")
    if not s:
        return ""
    try:
        cfg = get_diagnostics_config(get_settings_fn=get_settings)
        if bool(getattr(cfg, "redact_enabled", True)):
            s = redact_text(
                s,
                list(getattr(cfg, "redact_patterns", []) or []),
                str(getattr(cfg, "redact_replacement", "***REDACTED***") or "***REDACTED***"),
            )
        lim = int(getattr(cfg, "snippet_limit", 240) or 240)
        s = truncate_text(s, max(1, lim))
    except (AttributeError, TypeError, ValueError):
        logger.warning("Error while sanitizing startup detail", exc_info=True)
    return s


def _build_startup_diagnostics(
    session: Optional[SyncSession],
    agent_type: str,
    cwd: str,
    model_name: Optional[str],
    timeout: float,
    error: Exception,
) -> dict:
    """兼容入口：收敛到 SSOT（src.acp.sync_adapter.build_startup_diagnostics）。"""
    try:
        return build_startup_diagnostics(
            agent_type=agent_type,
            cwd=cwd,
            model_name=model_name,
            session=session,
            error=error,
            timeout_s=float(timeout or 0),
        )
    except Exception:
        logger.warning("Error while building startup diagnostics", exc_info=True)
        # 极端兜底：保证返回可序列化 dict
        return {
            "agent_type": agent_type or "",
            "cwd": cwd or "",
            "model": model_name or "",
            "timeout_s": float(timeout or 0),
            "error_type": type(error).__name__,
            "error": get_error_detail(error) if isinstance(error, Exception) else "(empty)",
            "cmd": "",
            "args": [],
            "rc": None,
            "stdout_snippet": "",
            "stderr_snippet": "",
        }


# `SessionKeyCodec` 的默认 project 占位符应作为 session_key 协议的 SSOT；
# 为保持历史常量名稳定，这里仅作为引用别名保留。
_DEFAULT_PROJECT = SessionKeyCodec.DEFAULT_PROJECT_PLACEHOLDER
_RESUME_CLOSE_HEADROOM_S = 17.0

# 标准化后的 session_key 解析结果类型：
# (chat_id, project_id, thread_id)
SessionKeyParts = tuple[str, Optional[str], Optional[str]]
class ACPSessionManager:
    """Manages per-chat, per-project sessions for a specific agent type.

    - Coco: ACP backend (SyncACPSession)
    - Claude: CLI backend (SyncClaudeCLISession)
    """

    def __init__(
        self,
        agent_type: str,
        session_timeout: int = 86400,
        session_starter: Optional[Callable[..., tuple[SyncSession, str, dict]]] = None,
        keepalive_interval: int = 0,
        idle_healthcheck_s: float = 120.0,
        idle_health_telemetry: IdleHealthTelemetry | None = None,
        session_telemetry: TelemetryAdapter | None = None,
        idle_health_service: IdleHealthServiceProtocol | None = None,
        idle_health_config: IdleHealthConfig | None = None,
    ):
        """Initialize a per-agent ACPSessionManager instance.

        IdleHealth 相关参数使用约定：

        - **新代码唯一推荐入口**：优先通过 ``idle_health_config=`` 注入
          :class:`IdleHealthConfig` 实例。业务侧通常应调用
          :func:`src.acp.telemetry.build_idle_health_config_for_manager` 构造该
          配置对象，并在需要时通过 ``session_telemetry=...`` 覆盖会话级
          Telemetry；
        - ``idle_health_config``: 推荐的 IdleHealth 协作者注入入口；调用方应优先
          通过 :class:`IdleHealthConfig`（或
          :func:`src.acp.telemetry.build_idle_health_config_for_manager` 的返回值）
          集中声明 IdleHealthTelemetry / SessionTelemetry / IdleHealthService 等协作
          者，以避免构造函数参数膨胀；
        - ``idle_health_telemetry`` / ``session_telemetry`` / ``idle_health_service``:
          [DEPRECATED] 仅为兼容历史调用点保留的显式协作者注入参数。新代码不应
          直接依赖这些参数，而应通过 ``idle_health_config`` 进行配置收口；当同
          时给定 ``idle_health_config`` 与上述显式参数时，显式参数仍然具有更高
          优先级，其行为由 :func:`resolve_idle_health_collaborators_for_manager` 统一解析。

        注意：本构造函数不自行推导 IdleHealth 默认行为，而是将「显式参数 /
        IdleHealthConfig / 默认工厂」的组合解析委托给 Telemetry 模块中的
        :func:`resolve_idle_health_collaborators_for_manager`，以确保 IdleHealth 相关配置的
        单一事实来源（SSOT）位于 Telemetry 层。
        """
        self._agent_type = agent_type  # "coco" / "claude"
        self._sessions: dict[str, SyncSession] = {}  # key = _session_key(...)
        self._closing_sessions: dict[str, _ClosingSessionState] = {}
        self._session_timeout = session_timeout
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._key_locks: dict[str, list] = {}  # per-session-key: [Lock, refcount]
        self._key_locks_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._session_starter = session_starter
        self._keepalive_interval = keepalive_interval
        self._idle_healthcheck_s = idle_healthcheck_s
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None
        # Telemetry 与 IdleHealth 协作者：统一通过 Telemetry 公共 facade
        # + 显式参数解析，解析优先级为「显式参数 > IdleHealthConfig 字段 > 默认工厂/适配器」。

        cfg: IdleHealthConfig | None = idle_health_config

        # 兼容性提示：当调用方显式使用 idle_health_* 协作者参数时，打印一次
        # 软 deprecate 日志，提示迁移到 idle_health_config 收口路径。为避免在
        # 高频路径产生副作用，日志输出采用 best-effort 策略，不影响主流程。
        try:
            if any(
                x is not None
                for x in (idle_health_telemetry, session_telemetry, idle_health_service)
            ):
                logger.warning(
                    "[DEPRECATED] ACPSessionManager.__init__ 的 idle_health_telemetry/"
                    "session_telemetry/idle_health_service 参数仅为兼容旧调用点保留；"
                    "请优先通过 idle_health_config=IdleHealthConfig(...) 或"
                    "build_idle_health_config_for_manager(...) 注入 IdleHealth 协作者。"
                )
        except Exception:
            logger.warning("Error while logging deprecated parameters warning", exc_info=True)
            # 不因日志问题影响核心逻辑。
            pass

        (
            self._idle_health_telemetry,
            self._session_telemetry,
            self._idle_health_service,
        ) = resolve_idle_health_collaborators_for_manager(
            config=cfg,
            idle_health_telemetry=idle_health_telemetry,
            session_telemetry=session_telemetry,
            idle_health_service=idle_health_service,
        )
        if keepalive_interval > 0:
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, daemon=True, name=f"acp-keepalive-{agent_type}"
            )
            self._keepalive_thread.start()

    @contextlib.contextmanager
    def _acquire_lock(self, timeout: float = 30.0):
        """Context manager to acquire lock with timeout, preventing deadlocks."""
        if not self._lock.acquire(timeout=timeout):
            msg = f"[ACP:{self._agent_type.upper()}] Failed to acquire lock within {timeout}s (deadlock detected)"
            logger.error(msg)
            raise TimeoutError(msg)
        try:
            yield
        finally:
            self._lock.release()

    def _get_key_lock(self, key: str) -> threading.Lock:
        """Get or create a per-session-key lock, incrementing reference count."""
        with self._key_locks_lock:
            entry = self._key_locks.get(key)
            if entry is None:
                entry = [threading.Lock(), 0]  # leaf lock: never held while acquiring a LockLevel lock
                self._key_locks[key] = entry
            entry[1] += 1  # increment refcount
            return entry[0]

    def _release_key_lock(self, key: str) -> None:
        """Decrement reference count for a per-session-key lock; remove when no references."""
        with self._key_locks_lock:
            entry = self._key_locks.get(key)
            if entry is None:
                return
            entry[1] -= 1
            if entry[1] <= 0:
                self._key_locks.pop(key, None)

    @staticmethod
    def _remaining_before(deadline: float, operation: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"ACP session {operation} timeout")
        return remaining

    def _close_session_before(
        self,
        session: SyncSession,
        *,
        key: str,
        deadline: float,
    ) -> None:
        """Run close synchronously from the caller's perspective, with a deadline."""
        with self._acquire_lock(
            timeout=self._remaining_before(deadline, "close registration")
        ):
            state = self._register_closing_session_unlocked(key)
        self._launch_session_close(key, session, state)

        remaining = deadline - time.monotonic()
        if remaining <= 0 or not state.completed.wait(timeout=remaining):
            raise TimeoutError("ACP session close timeout")
        if state.error is not None:
            raise state.error

    def _register_closing_session_unlocked(
        self,
        key: str,
    ) -> _ClosingSessionState:
        """Register one close while ``self._lock`` is held."""
        previous = self._closing_sessions.get(key)
        if previous is not None:
            if previous.error is not None:
                raise RuntimeError(
                    "ACP previous session close failed"
                ) from previous.error
            if not previous.completed.is_set():
                raise RuntimeError("ACP session is already closing")
        state = _ClosingSessionState(completed=threading.Event())
        self._closing_sessions[key] = state
        return state

    def _launch_session_close(
        self,
        key: str,
        session: SyncSession,
        state: _ClosingSessionState,
    ) -> None:
        """Start the close worker for a previously registered state."""
        def _close() -> None:
            try:
                session.close()
            except BaseException as exc:
                state.error = exc
            finally:
                state.completed.set()
                with self._lock:
                    if (
                        self._closing_sessions.get(key) is state
                        and state.error is None
                    ):
                        self._closing_sessions.pop(key, None)

        worker = threading.Thread(
            target=_close,
            daemon=True,
            name=f"acp-retire-close-{str(session.session_id or 'none')[:8]}",
        )
        worker.start()

    def _wait_for_closing_session(
        self,
        key: str,
        *,
        deadline: float,
    ) -> None:
        """Fence starts until a previously detached session really closes."""
        while True:
            with self._acquire_lock(
                timeout=self._remaining_before(deadline, "closing gate")
            ):
                state = self._closing_sessions.get(key)
            if state is None:
                return
            if state.error is not None:
                raise RuntimeError(
                    "ACP previous session close failed"
                ) from state.error
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not state.completed.wait(timeout=remaining)
            ):
                raise TimeoutError("ACP session is still closing")
            with self._acquire_lock(
                timeout=self._remaining_before(deadline, "closing gate cleanup")
            ):
                if state.error is not None:
                    raise RuntimeError(
                        "ACP previous session close failed"
                    ) from state.error
                if (
                    self._closing_sessions.get(key) is state
                    and state.completed.is_set()
                ):
                    self._closing_sessions.pop(key, None)

    def _build_startup_coordinator(self) -> _startup_utils.SessionStartupCoordinator:
        return _startup_utils.SessionStartupCoordinator(
            manager_agent_type=self._agent_type,
            session_starter=self._session_starter,
            session_telemetry=self._session_telemetry,
            sync_acp_session_cls=SyncACPSession,
            sync_claude_cli_session_cls=SyncClaudeCLISession,
            sync_ttadk_cli_session_cls=SyncTTADKCLISession,
            agent_session_module=_agent_session_mod,
            original_ttadk_cli_session_cls=_ORIG_TTADK_CLI_SESSION,
            get_settings_fn=get_settings,
        )

    def _remove_key_lock(self, key: str) -> None:
        """Compatibility wrapper for historical callers.

        Per-session-key locks are transient startup leases owned exclusively by
        ``start_session()``.  Session ending, keepalive cleanup, and rebind flows
        do not own that lease and must not decrement its reference count.
        """
        lock = getattr(self, "_key_locks_lock", None)
        if lock is None:
            return
        return

    def _keepalive_loop(self) -> None:
        while not self._keepalive_stop.wait(timeout=self._keepalive_interval):
            try:
                # Take snapshot under lock, then release — iteration is lock-free
                with self._acquire_lock():
                    snapshot = list(self._sessions.items())
                # Lock released here; safe to iterate without blocking session ops
                now = time.time()
                for key, session in snapshot:
                    try:
                        idle = now - session.last_active
                        # Always check sessions that have been force-marked dead
                        # (e.g. after terminal-state errors); skip idle threshold.
                        force_dead = getattr(session, "_force_dead", False)
                        if not force_dead and idle <= self._idle_healthcheck_s:
                            continue
                        alive = session.is_server_running()
                        if not alive:
                            # Re-acquire lock independently for mutation
                            with self._acquire_lock():
                                if self._sessions.get(key) is session:
                                    logger.info(
                                        "[ACP:%s] Keepalive cleaning dead session: key=%s, session=%s",
                                        self._agent_type.upper(),
                                        key[-16:],
                                        (session.session_id or "none")[:8],
                                    )
                                    self._end_session_unlocked(key)
                    except Exception:
                        logger.debug("[ACP:%s] Keepalive check error for key=%s", self._agent_type.upper(), key[-16:], exc_info=True)
            except Exception:
                logger.debug("[ACP:%s] Keepalive loop iteration error", self._agent_type.upper(), exc_info=True)

    @staticmethod
    def _compute_idle_bucket(seconds: float) -> "TimeAgoBucket":
        """将 idle 秒数转换为 :class:`TimeAgoBucket` 结构的纯语义函数。

        说明：
        - 仅负责「秒数 → 语义化时间区间」的映射，不产出任何 UI 文案；
        - 内部委托 `src.utils.time_ago.compute_time_ago_bucket` 作为 SSOT，
          保证与全局 TimeAgo 语义保持一致；
        - 供会话管理与诊断逻辑使用，上层如 Feishu Handler / CardBuilder
          需要展示「多久之前」文本时，应基于返回的 ``TimeAgoBucket``
          调用文案层 helper（例如 ``src.utils.text.format_time_ago_from_bucket``）。
        """

        from src.utils.time_ago import compute_time_ago_bucket

        return compute_time_ago_bucket(seconds)

    def _classify_idle_health_with_telemetry(
        self,
        bucket: "TimeAgoBucket",
        context: IdleHealthTelemetryContext | None = None,
    ) -> "IdleHealth":
        """基于 TimeAgoBucket 对会话 idle 状态做粗粒度健康分类（实例入口）。

        说明：
        - 委托 Telemetry 模块的公共入口 `classify_manager_idle_health`；
        - 通过实例级 ``self._idle_health_telemetry`` 注入可观测性实现；
        - 保持 UNKNOWN 回退语义与历史实现等价。
        """

        # NOTE: 具体实现收敛在 Telemetry 模块内部，manager 仅负责提供 bucket/context
        # 与注入好的 telemetry 实例，避免在此处重复维护 UNKNOWN 回退策略。
        return classify_manager_idle_health(
            bucket,
            context=context,
            telemetry=self._idle_health_telemetry,
        )

    @staticmethod
    def classify_idle_health(bucket: "TimeAgoBucket", context: IdleHealthTelemetryContext | None = None) -> "IdleHealth":
        """兼容静态入口：保持历史调用点 API 不变。

        说明：
        - 静态入口统一委托 Telemetry 模块的
          `classify_manager_idle_health`；
        - Telemetry 实例的选择与 UNKNOWN 回退策略完全收敛在 Telemetry
          模块内部，manager 只负责提供 bucket/context；
        - 调用方签名与返回语义保持不变，兼容历史测试与调用点。
        """

        return classify_manager_idle_health(
            bucket,
            context=context,
        )

    @staticmethod
    def _session_key(chat_id: str, project_id: Optional[str] = None, thread_id: Optional[str] = None) -> str:
        """Compute the opaque ``session_key`` used as the internal dict key.

        说明：
        - 具体编码协议已集中到 :class:`SessionKeyCodec` 中，本方法仅作为
          兼容入口，委托给协作者以避免协议散落在多个模块；
        - 历史上关于 chat/project/thread 段落与占位符的约束在迁移过程中
          由 SessionKeyCodec 保持，与现有行为等价；
        - 调用方仍应将返回值视为不透明字符串，仅通过
          :meth:`_parse_session_key` 或 SessionKeyCodec.decode 进行解析。
        """

        return SessionKeyCodec.encode(chat_id, project_id=project_id, thread_id=thread_id)

    @staticmethod
    def _parse_session_key(key: str) -> SessionKeyParts:
        """Parse a ``session_key`` back into ``(chat_id, project_id, thread_id)``.

        说明：
        - 解析协议同样集中在 :class:`SessionKeyCodec` 中，本方法仅作为
          兼容入口，保证历史静态调用点与测试无需修改即可复用新实现；
        - SessionKeyCodec.decode 已实现对旧格式与异常输入的宽容解析逻辑，
          包括 ``("", None, None)`` 与 ``(key, None, None)`` 等兜底分支。
        """

        return SessionKeyCodec.decode(key)

    def start_session(
        self,
        chat_id: str,
        cwd: str = "",
        session_id: Optional[str] = None,
        startup_timeout: float = 60,
        project_id: Optional[str] = None,
        agent_type_override: Optional[str] = None,
        model_name: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> SyncSession:
        """Start a new session for a chat/project."""
        key = self._session_key(chat_id, project_id, thread_id=thread_id)
        deadline = time.monotonic() + float(startup_timeout)
        # Per-key lock serializes concurrent start_session calls for the same key,
        # preventing TOCTOU race where two threads both create sessions and one leaks.
        key_lock = self._get_key_lock(key)
        try:
            lock_timeout = self._remaining_before(deadline, "startup lock")
        except Exception:
            self._release_key_lock(key)
            raise
        if not key_lock.acquire(timeout=lock_timeout):
            self._release_key_lock(key)
            raise TimeoutError("会话启动超时：当前会话正忙，请稍后重试")
        try:
            self._wait_for_closing_session(key, deadline=deadline)
            return self._start_session_inner(
                key,
                chat_id,
                cwd,
                session_id,
                self._remaining_before(deadline, "startup"),
                project_id, agent_type_override, model_name, thread_id,
            )
        finally:
            key_lock.release()
            self._release_key_lock(key)

    def _start_session_inner(
        self,
        key: str,
        chat_id: str,
        cwd: str,
        session_id: Optional[str],
        startup_timeout: float,
        project_id: Optional[str],
        agent_type_override: Optional[str],
        model_name: Optional[str],
        thread_id: Optional[str],
    ) -> SyncSession:
        """Inner implementation of start_session (called under per-key lock)."""
        startup_deadline = time.monotonic() + float(startup_timeout)
        # Detach and fully close an existing session before a replacement can
        # start.  The closing gate remains visible if close exceeds this call's
        # budget, so a later start still cannot overlap it.
        with self._acquire_lock(
            timeout=self._remaining_before(startup_deadline, "startup detach")
        ):
            existing, _snapshot = self._detach_forced_session_unlocked(
                key,
                expected_session=self._sessions.get(key),
            )
        if existing is not None:
            self._close_session_before(
                existing,
                key=key,
                deadline=startup_deadline,
            )
        # ``end_session()`` and keepalive cleanup intentionally do not acquire
        # the startup key lock. They may therefore install a close tombstone
        # after start_session()'s first gate check but before the detach above.
        # Re-check immediately before spawning a backend so the old transport
        # and its replacement can never overlap.
        self._wait_for_closing_session(
            key,
            deadline=startup_deadline,
        )
        startup_timeout = self._remaining_before(
            startup_deadline,
            "backend startup",
        )

        settings = get_settings()
        retries = int(getattr(settings, "acp_startup_retries", 2) or 2)
        retries = max(1, retries)
        effective_agent_type = (agent_type_override or self._agent_type).lower()
        model_name = _normalize_manager_acp_model(effective_agent_type, model_name)
        startup_result = self._build_startup_coordinator().start(
            _startup_utils.SessionStartupRequest(
                key=key,
                cwd=cwd,
                startup_timeout=startup_timeout,
                project_id=project_id,
                session_id=session_id,
                effective_agent_type=effective_agent_type,
                model_name=model_name,
                retries=retries,
                deadline_monotonic=startup_deadline,
            )
        )
        session = startup_result.session
        actual_id = startup_result.actual_id
        effective_agent_type = startup_result.effective_agent_type
        model_name = startup_result.model_name

        # If caller wants a specific session_id (resume), load it
        if session_id:
            try:
                resume_remaining = self._remaining_before(
                    startup_deadline,
                    "resume load",
                )
                resume_close_reserve = min(
                    _RESUME_CLOSE_HEADROOM_S,
                    resume_remaining / 2.0,
                )
                resume_load_timeout = resume_remaining - resume_close_reserve
                session.load_session(
                    session_id,
                    timeout=resume_load_timeout,
                )
                session.session_id = session_id
                session.is_resumed = True
            except ACPResumeRejected as e:
                logger.warning(
                    "[ACP:%s] Failed to load session %s, using new: %s", effective_agent_type.upper(), session_id[:8], e
                )
            except BaseException as e:
                setattr(session, "_force_dead", True)
                try:
                    self._close_session_before(
                        session,
                        key=key,
                        deadline=startup_deadline,
                    )
                except BaseException as close_error:
                    raise BaseExceptionGroup(
                        "ACP resume and uncertain session retirement both failed",
                        [e, close_error],
                    ) from None
                raise

        # Load local persisted history (best-effort)
        try:
            session.load_local_history(session.session_id)
        except Exception:
            logger.warning("Error while loading local history", exc_info=True)
            pass

        with self._acquire_lock():
            self._sessions[key] = session
        # 会话成功启动后触发 Telemetry 事件（best-effort）。
        try:
            from ..agent_session.backend_resolver import is_cli_backend
            backend_kind = "cli" if is_cli_backend(effective_agent_type) else "acp"
            self._session_telemetry.on_session_start(
                manager_agent_type=self._agent_type,
                session_key=key,
                session_id=session.session_id or actual_id,
                backend_kind=backend_kind,
                model_name=model_name,
            )
        except Exception:
            logger.debug("[ACP:%s] session telemetry on_session_start error", self._agent_type.upper(), exc_info=True)
        return session

    def ensure_session(
        self,
        chat_id: str,
        cwd: str = "",
        session_id: Optional[str] = None,
        startup_timeout: float = 60,
        project_id: Optional[str] = None,
        agent_type_override: Optional[str] = None,
        model_name: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> SyncSession:
        """Ensure a session exists and it is ready.

        1) Detect whether current backend is alive/healthy (if applicable).
        2) If not alive / missing / timed out, auto-start a new session.
        3) Optionally load a given session_id (resume) after startup.
        """
        key = self._session_key(chat_id, project_id, thread_id=thread_id)
        effective_agent_for_model = (agent_type_override or self._agent_type).lower()
        model_name = _normalize_manager_acp_model(effective_agent_for_model, model_name)

        # Helper: safely end session under lock with double-check
        def _safe_end_session(check_fn) -> bool:
            """End session under lock if check_fn returns True. Returns True if ended."""
            with self._acquire_lock():
                s = self._sessions.get(key)
                if s is not None and check_fn(s):
                    self._end_session_unlocked(key)
                    return True
                return False

        with self._acquire_lock():
            existing = self._sessions.get(key)
        if existing and getattr(existing, "_force_dead", False) is True:
            logger.warning(
                "[ACP:%s] Refusing force-dead session during ensure: key=%s session=%s",
                self._agent_type.upper(),
                key[-16:],
                (existing.session_id or "none")[:8],
            )
            self.retire_session(
                chat_id,
                project_id=project_id,
                thread_id=thread_id,
                expected_session=existing,
            )
            with self._acquire_lock():
                current = self._sessions.get(key)
            existing = current if current is not existing else None
        if existing:
            # Timeout check (reuse get_session semantics)
            if time.time() - existing.last_active > self._session_timeout:
                logger.info("[ACP:%s] Session timeout before ensure: key=%s", self._agent_type.upper(), key[-16:])
                _safe_end_session(lambda _: True)
                existing = None

        # Agent type / model mismatch for dynamic backends (e.g., TTADK)
        if existing and agent_type_override:
            existing_agent = getattr(existing, "_agent_type", "")
            if existing_agent and existing_agent.lower() != agent_type_override.lower():
                logger.info(
                    "[ACP:%s] Agent type changed (%s -> %s), restarting: key=%s",
                    self._agent_type.upper(),
                    existing_agent,
                    agent_type_override,
                    key[-16:],
                )
                _safe_end_session(lambda _: True)
                existing = None
            elif model_name:
                # TTADK: model_name 可能是"意图/友好名"，未必会透传 -m；仅当能解析出 validated 的真实模型名时才做一致性重启。
                from ..agent_session.backend_resolver import is_ttadk_type
                if is_ttadk_type(agent_type_override):
                    # 若该 session 已因 TTADK 启动失败降级（例如降级到 coco ACP），则不要再因 model mismatch 触发重启，
                    # 否则在 TTADK 不可用时会产生"每次 ensure 都重启→再失败→再降级"的抖动。
                    target_model: Optional[str] = None
                    if not getattr(existing, "_degraded_to", ""):
                        try:
                            target_model = self._build_startup_coordinator().resolve_ttadk_target_model_for_existing_session(
                                agent_type=agent_type_override,
                                cwd=cwd or ".",
                                model_name=model_name,
                            )
                        except Exception:
                            logger.warning("Error while prechecking TTADK startup model", exc_info=True)
                            target_model = None

                    if target_model:
                        existing_args = getattr(existing, "_agent_args", None)
                        args_text = " ".join(existing_args or [])
                        if target_model not in args_text:
                            logger.info(
                                "[ACP:%s] TTADK model changed (missing %s), restarting: key=%s",
                                self._agent_type.upper(),
                                target_model,
                                key[-16:],
                            )
                            _safe_end_session(lambda _: True)
                            existing = None
                else:
                    if not _session_matches_requested_model(existing, model_name):
                        logger.info(
                            "[ACP:%s] Model changed (%s), restarting: key=%s",
                            self._agent_type.upper(),
                            model_name,
                            key[-16:],
                        )
                        _safe_end_session(lambda _: True)
                        existing = None

        if existing:
            idle = time.time() - existing.last_active
            # Quick process-alive check first (no RPC); full health only after prolonged idle
            if not existing.is_server_running():
                logger.warning(
                    "[ACP:%s] Detected dead ACP server, restarting: key=%s session=%s",
                    self._agent_type.upper(),
                    key[-16:],
                    (existing.session_id or "none")[:8],
                )
                _safe_end_session(lambda s: s is existing)
                existing = None
            elif idle > 30.0:
                health_to = float(getattr(get_settings(), "acp_healthcheck_timeout", 2.0) or 2.0)
                if not existing.is_server_healthy(healthcheck_timeout=health_to):
                    logger.warning(
                        "[ACP:%s] Detected unhealthy ACP server, restarting: key=%s session=%s",
                        self._agent_type.upper(),
                        key[-16:],
                        (existing.session_id or "none")[:8],
                    )
                    _safe_end_session(lambda s: s is existing)
                    existing = None

        # Model mismatch check for non-TTADK sessions (when no agent_type_override).
        # Ensures that calling ensure_session() with a different model_name triggers a restart.
        if existing and not agent_type_override and model_name:
            if not _session_matches_requested_model(existing, model_name):
                logger.info(
                    "[ACP:%s] Model changed (%s), restarting: key=%s",
                    self._agent_type.upper(),
                    model_name,
                    key[-16:],
                )
                _safe_end_session(lambda _: True)
                existing = None

        if existing and session_id and existing.session_id != session_id:
            # Different target session requested; restart to load requested session.
            _safe_end_session(lambda _: True)
            existing = None

        if existing:
            return existing

        return self.start_session(
            chat_id,
            cwd=cwd,
            session_id=session_id,
            startup_timeout=startup_timeout,
            project_id=project_id,
            agent_type_override=agent_type_override,
            model_name=model_name,
            thread_id=thread_id,
        )

    def resume_session(
        self, chat_id: str, session_id: str, cwd: str = "", project_id: Optional[str] = None, thread_id: Optional[str] = None,
    ) -> SyncSession:
        """Resume an existing session by session_id."""
        return self.start_session(chat_id, cwd=cwd, session_id=session_id, project_id=project_id, thread_id=thread_id)

    def replace_session(
        self,
        chat_id: str,
        *,
        cwd: str = "",
        expected_session: SyncSession,
        startup_timeout: float = 60,
        project_id: Optional[str] = None,
        agent_type_override: Optional[str] = None,
        model_name: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> SessionReplacementResult:
        """Atomically retire ``expected_session`` and install its replacement.

        The per-key startup lease is held across compare, close, and start.
        Replacement is a strict compare-and-swap: only the exact
        ``expected_session`` may be detached.  A stale caller receives
        ``created=False`` with the current session (or ``None`` if it was
        already removed).
        """
        timeout = float(startup_timeout)
        if timeout <= 0:
            raise TimeoutError("ACP session replacement timeout")
        deadline = time.monotonic() + timeout
        key = self._session_key(chat_id, project_id, thread_id=thread_id)
        key_lock = self._get_key_lock(key)
        try:
            if not key_lock.acquire(
                timeout=self._remaining_before(deadline, "replacement lock")
            ):
                raise TimeoutError("ACP session replacement lock timeout")
            try:
                self._wait_for_closing_session(key, deadline=deadline)
                with self._acquire_lock(
                    timeout=self._remaining_before(deadline, "replacement")
                ):
                    current = self._sessions.get(key)
                    if current is not expected_session:
                        return SessionReplacementResult(
                            session=current,
                            created=False,
                        )
                    detached, _snapshot = self._detach_forced_session_unlocked(
                        key,
                        expected_session=expected_session,
                    )
                if detached is not None:
                    self._close_session_before(
                        detached,
                        key=key,
                        deadline=deadline,
                    )
                remaining = self._remaining_before(deadline, "replacement start")
                return SessionReplacementResult(
                    session=self._start_session_inner(
                        key,
                        chat_id,
                        cwd,
                        None,
                        remaining,
                        project_id,
                        agent_type_override,
                        model_name,
                        thread_id,
                    ),
                    created=True,
                )
            finally:
                key_lock.release()
        finally:
            self._release_key_lock(key)

    def get_session(self, chat_id: str, project_id: Optional[str] = None, thread_id: Optional[str] = None) -> Optional[SyncSession]:
        """Get active session for a chat/project (with timeout check).

        Health check is only performed when the session has been idle for a while
        (> 30s) to avoid costly RPC round-trips on every call.  For recently-active
        sessions the send_prompt watchdog already handles crash detection.
        """
        key = self._session_key(chat_id, project_id, thread_id=thread_id)
        with self._acquire_lock():
            session = self._sessions.get(key)
        if session:
            if getattr(session, "_force_dead", False) is True:
                logger.warning(
                    "[ACP:%s] Evicting force-dead session: key=%s session=%s",
                    self._agent_type.upper(),
                    key[-16:],
                    (session.session_id or "none")[:8],
                )
                self.retire_session(
                    chat_id,
                    project_id=project_id,
                    thread_id=thread_id,
                    expected_session=session,
                )
                with self._acquire_lock():
                    current = self._sessions.get(key)
                return current if current is not session else None
            now = time.time()
            idle = now - session.last_active
            if idle > self._session_timeout:
                logger.info("[ACP:%s] Session timeout: key=%s", self._agent_type.upper(), key[-16:])
                # Use _end_session_unlocked under lock to avoid race window
                with self._acquire_lock():
                    # Double-check: session may have been replaced by another thread
                    current = self._sessions.get(key)
                    if current is session:
                        self._end_session_unlocked(key)
                        current = None
                return current
            # Only do expensive RPC health check after prolonged idle (>30s).
            # Recently active sessions are protected by the send_prompt watchdog.
            if idle > 30.0:
                if not session.is_server_running():
                    logger.warning(
                        "[ACP:%s] Session server dead: key=%s session=%s",
                        self._agent_type.upper(),
                        key[-16:],
                        (session.session_id or "none")[:8],
                    )
                    with self._acquire_lock():
                        current = self._sessions.get(key)
                        if current is session:
                            self._end_session_unlocked(key)
                            current = None
                    return current
        return session

    def cancel_session(
        self,
        chat_id: str,
        *,
        project_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        wait: bool = False,
        timeout: float = 2.0,
    ) -> bool:
        """Cancel the currently selected session without changing mode state.

        This is intentionally narrower than ``end_session``: the caller keeps
        the session routing key and lifecycle ownership while the underlying
        transport receives its native cancellation request.
        """
        session = self.get_session(
            chat_id,
            project_id=project_id,
            thread_id=thread_id,
        )
        if session is None:
            return False
        result = session.cancel(wait=wait, timeout=timeout)
        return result is not False

    def _end_session_unlocked(self, key: str, *, remove_key_lock: bool = False) -> Optional[dict]:
        """End a session without acquiring lock (caller must hold _lock)."""
        if key in self._sessions:
            session = self._sessions[key]
            logger.info(
                "[ACP:%s] Session ended: key=%s, session=%s, msgs=%d",
                self._agent_type.upper(),
                key[-16:],
                session.session_id[:8] if session.session_id else "none",
                session.message_count,
            )
            try:
                snapshot = session.to_snapshot()
            except Exception:
                snapshot = None
                logger.warning(
                    "[ACP:%s] session end snapshot unavailable: key=%s session=%s",
                    self._agent_type.upper(),
                    key[-16:],
                    str(session.session_id or "none")[:8],
                    exc_info=True,
                )
            # Best-effort Telemetry：会话结束事件
            try:
                self._session_telemetry.on_session_end(
                    manager_agent_type=self._agent_type,
                    session_key=key,
                    session_id=session.session_id or "",
                    message_count=session.message_count,
                    reason=None,
                    extra=None,
                )
            except Exception:
                logger.debug("[ACP:%s] session telemetry on_session_end error", self._agent_type.upper(), exc_info=True)
            closing_state = self._register_closing_session_unlocked(key)
            del self._sessions[key]
            if remove_key_lock:
                # Historical compatibility only: key-lock leases are owned by
                # start_session(), so end/keepalive/rebind callers must not
                # release or remove the startup lock registry entry.
                self._remove_key_lock(key)

            # Offload closing while leaving a per-key tombstone. Any later
            # start/replace waits until this worker confirms close success.
            self._launch_session_close(key, session, closing_state)
            return snapshot
        return None

    def _detach_forced_session_unlocked(
        self,
        key: str,
        *,
        expected_session: SyncSession | None,
    ) -> tuple[SyncSession | None, dict | None]:
        """Detach an exact session under ``self._lock`` for bounded closing."""
        session = self._sessions.get(key)
        if session is None:
            return None, None
        if expected_session is not None and session is not expected_session:
            return None, None
        logger.info(
            "[ACP:%s] Session synchronously retired: key=%s, session=%s, msgs=%d",
            self._agent_type.upper(),
            key[-16:],
            str(session.session_id or "none")[:8],
            session.message_count,
        )
        try:
            snapshot = session.to_snapshot()
        except Exception:
            snapshot = None
            logger.warning(
                "[ACP:%s] forced retirement snapshot unavailable: key=%s session=%s",
                self._agent_type.upper(),
                key[-16:],
                str(session.session_id or "none")[:8],
                exc_info=True,
            )
        try:
            self._session_telemetry.on_session_end(
                manager_agent_type=self._agent_type,
                session_key=key,
                session_id=session.session_id or "",
                message_count=session.message_count,
                reason="forced_retirement",
                extra=None,
            )
        except Exception:
            logger.debug(
                "[ACP:%s] session telemetry on forced retirement error",
                self._agent_type.upper(),
                exc_info=True,
            )
        del self._sessions[key]
        return session, snapshot

    def end_session(self, chat_id: str, project_id: Optional[str] = None, thread_id: Optional[str] = None) -> Optional[dict]:
        """End a session and return its snapshot."""
        key = self._session_key(chat_id, project_id, thread_id=thread_id)
        with self._acquire_lock():
            return self._end_session_unlocked(key)

    def retire_session(
        self,
        chat_id: str,
        project_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        *,
        expected_session: SyncSession | None = None,
        timeout: float = 30.0,
    ) -> Optional[dict]:
        """Synchronously retire one exact session before allowing replacement.

        Unlike ``end_session()``, this emergency path waits for ``close()`` and
        uses an identity guard so a late timeout callback cannot remove a
        concurrently installed healthy replacement.
        """
        timeout = float(timeout)
        if timeout <= 0:
            raise TimeoutError("ACP session retirement timeout")
        deadline = time.monotonic() + timeout
        key = self._session_key(chat_id, project_id, thread_id=thread_id)
        key_lock = self._get_key_lock(key)
        try:
            if not key_lock.acquire(
                timeout=self._remaining_before(deadline, "retirement lock")
            ):
                raise TimeoutError(
                    f"[ACP:{self._agent_type.upper()}] session retirement lock timeout"
                )
            try:
                self._wait_for_closing_session(key, deadline=deadline)
                with self._acquire_lock(
                    timeout=self._remaining_before(deadline, "retirement")
                ):
                    session, snapshot = self._detach_forced_session_unlocked(
                        key,
                        expected_session=expected_session,
                    )
                if session is None:
                    return None
                self._close_session_before(
                    session,
                    key=key,
                    deadline=deadline,
                )
            finally:
                key_lock.release()
            return snapshot
        finally:
            self._release_key_lock(key)

    def rebind_thread(
        self,
        chat_id: str,
        project_id: str,
        thread_id: str,
        *,
        timeout: float = 30.0,
    ) -> bool:
        """Move a chat-level session to a thread without overlapping transports."""
        old_key = self._session_key(chat_id, project_id, thread_id=None)
        new_key = self._session_key(chat_id, project_id, thread_id=thread_id)
        timeout = float(timeout)
        if timeout <= 0:
            raise TimeoutError("ACP session rebind timeout")
        deadline = time.monotonic() + timeout
        keyed_locks = [
            (key, self._get_key_lock(key))
            for key in sorted({old_key, new_key})
        ]
        acquired: list[tuple[str, threading.Lock]] = []
        try:
            for key, key_lock in keyed_locks:
                if not key_lock.acquire(
                    timeout=self._remaining_before(deadline, "rebind lock")
                ):
                    raise TimeoutError("ACP session rebind lock timeout")
                acquired.append((key, key_lock))

            for key, _key_lock in keyed_locks:
                self._wait_for_closing_session(key, deadline=deadline)

            with self._acquire_lock(
                timeout=self._remaining_before(deadline, "rebind detach")
            ):
                session = self._sessions.get(old_key)
                if session is None:
                    return False
                existing = self._sessions.get(new_key)
                if existing is session:
                    del self._sessions[old_key]
                    return True
                detached: SyncSession | None = None
                if existing is not None:
                    detached, _snapshot = self._detach_forced_session_unlocked(
                        new_key,
                        expected_session=existing,
                    )

            if detached is not None:
                self._close_session_before(
                    detached,
                    key=new_key,
                    deadline=deadline,
                )

            with self._acquire_lock(
                timeout=self._remaining_before(deadline, "rebind commit")
            ):
                if self._sessions.get(old_key) is not session:
                    return False
                if self._sessions.get(new_key) is not None:
                    return False
                self._sessions[new_key] = session
                del self._sessions[old_key]
            return True
        finally:
            for _key, key_lock in reversed(acquired):
                key_lock.release()
            for key, _key_lock in keyed_locks:
                self._release_key_lock(key)

    def has_active_session(self, chat_id: str, project_id: Optional[str] = None, thread_id: Optional[str] = None) -> bool:
        return self.get_session(chat_id, project_id=project_id, thread_id=thread_id) is not None

    def get_session_info(self, chat_id: str, project_id: Optional[str] = None, thread_id: Optional[str] = None) -> Optional[str]:
        """Return human-readable session info."""
        session = self.get_session(chat_id, project_id=project_id, thread_id=thread_id)
        if not session:
            return None
        return session.get_session_info()

    def cleanup_all(self, *, timeout: float = 30.0) -> None:
        """Close all sessions and wait for every transport within one deadline."""
        timeout = float(timeout)
        if timeout <= 0:
            raise TimeoutError("ACP session cleanup timeout")
        deadline = time.monotonic() + timeout
        failures: list[BaseException] = []
        self._keepalive_stop.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(
                timeout=min(
                    5.0,
                    self._remaining_before(deadline, "cleanup keepalive"),
                )
            )
            if self._keepalive_thread.is_alive():
                failures.append(
                    TimeoutError("ACP keepalive thread did not stop")
                )
            else:
                self._keepalive_thread = None
        try:
            with self._acquire_lock(
                timeout=self._remaining_before(deadline, "cleanup listing")
            ):
                keys = list(self._sessions.keys())
        except BaseException as exc:
            keys = []
            failures.append(exc)
        for key in keys:
            try:
                with self._acquire_lock(
                    timeout=self._remaining_before(deadline, "cleanup detach")
                ):
                    self._end_session_unlocked(key)
            except BaseException as exc:
                failures.append(exc)
                logger.error(
                    "Error detaching ACP session for %s: %s",
                    key[-16:],
                    get_error_detail(exc),
                )

        try:
            with self._acquire_lock(
                timeout=self._remaining_before(deadline, "cleanup close listing")
            ):
                closing = list(self._closing_sessions.items())
        except BaseException as exc:
            closing = []
            failures.append(exc)

        for key, state in closing:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not state.completed.wait(timeout=remaining):
                failures.append(
                    TimeoutError(
                        f"ACP session close timeout during cleanup: {key[-16:]}"
                    )
                )
                continue
            if state.error is not None:
                failures.append(state.error)
                continue
            try:
                with self._acquire_lock(
                    timeout=self._remaining_before(
                        deadline,
                        "cleanup close commit",
                    )
                ):
                    if self._closing_sessions.get(key) is state:
                        self._closing_sessions.pop(key, None)
            except BaseException as exc:
                failures.append(exc)

        if failures:
            raise BaseExceptionGroup(
                "ACP session cleanup failed",
                failures,
            )

    def list_active_sessions(self, chat_id: Optional[str] = None) -> list[dict]:
        """Return lightweight snapshots for currently tracked sessions.

        When *chat_id* is given, only sessions belonging to that chat are returned.
        """
        now = time.time()
        out: list[dict] = []
        with self._acquire_lock():
            items = list(self._sessions.items())

        for key, session in items:
            try:
                # Chat-level isolation: skip sessions not belonging to the requested chat
                if chat_id is not None:
                    key_chat_id, _, _ = SessionKeyCodec.decode(key)
                    if key_chat_id != chat_id:
                        continue

                sid = str(getattr(session, "session_id", "") or "")
                last_active = float(getattr(session, "last_active", 0.0) or 0.0)
                message_count = int(getattr(session, "message_count", 0) or 0)

                idle_health, idle_bucket, idle_s, _ctx = self._idle_health_service.classify_session_idle_health(
                    manager_agent_type=self._agent_type,
                    session_key=key,
                    session_id=sid,
                    last_active=last_active,
                    now=now,
                    message_count=message_count,
                )

                out.append(
                    {
                        "manager_agent_type": self._agent_type,
                        "session_key": key,
                        "session_id": sid,
                        "last_active": last_active,
                        "message_count": message_count,
                        "idle_seconds": idle_s,
                        "idle_bucket": idle_bucket,
                        "idle_health": idle_health,
                    }
                )
            except Exception:
                logger.warning("Error while building session status", exc_info=True)
                continue
        return out


class AgentSessionManager(ACPSessionManager):
    """Semantically clearer alias for ACP+CLI session routing manager."""
