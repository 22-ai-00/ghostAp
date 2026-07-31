"""Session factory functions and helpers."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Optional

from ..acp.providers import normalize_acp_model_name
from ..acp.sync_adapter import SyncACPSession
from ..config import get_settings
from ..utils.errors import get_error_detail
from .claude_cli import SyncClaudeCLISession
from .protocol import SyncSession
from .tool_permissions import (
    AuxiliarySessionPermissionError,
    apply_auxiliary_tool_profile,
)
from .ttadk_cli import SyncTTADKCLISession
from .wrappers import ModelFailureAwareSession, RateLimitAwareSession

logger = logging.getLogger(__name__)
_EMPLOYEE_SESSION_ENV: ContextVar[dict[str, str] | None] = ContextVar(
    "employee_session_env",
    default=None,
)


@contextmanager
def employee_session_environment(env: dict[str, str]):
    """Scope one explicit env until the synchronous session factory captures it."""

    if _EMPLOYEE_SESSION_ENV.get() is not None:
        raise RuntimeError("nested employee session environment is forbidden")
    if not isinstance(env, dict) or not env or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in env.items()
    ):
        raise ValueError("employee session environment must be explicit")
    token = _EMPLOYEE_SESSION_ENV.set(dict(env))
    try:
        yield
    finally:
        _EMPLOYEE_SESSION_ENV.reset(token)


def current_employee_session_environment() -> dict[str, str] | None:
    """Return a copy for immediate synchronous capture by the factory."""

    value = _EMPLOYEE_SESSION_ENV.get()
    return None if value is None else dict(value)


def _normalize_acp_startup_model(agent_type: str, model_name: Optional[str]) -> Optional[str]:
    """Normalize ACP model values before startup/protocol use.

    Some providers expose UI-facing values that are not valid backend model IDs
    when passed back to their CLI/ACP protocol. Keep this at the session-factory
    boundary so Deep/Spec/Review/Slock share the same normalization.
    """
    agent = (agent_type or "").strip().lower()
    if (
        not model_name
        or agent in {"claude", "traex"}
        or agent.startswith("ttadk_")
    ):
        return model_name
    normalized = normalize_acp_model_name(agent, model_name)
    if normalized != model_name:
        logger.info(
            "[SessionFactory] normalized ACP model: agent=%s selected_model=%s backend_model=%s",
            agent,
            model_name,
            normalized,
        )
    return normalized


def close_session_safely(session: Optional[SyncSession]) -> None:
    """Close an ACP/CLI session, ignoring errors."""
    if session:
        try:
            session.close()
        except Exception as e:
            logger.debug("关闭旧ACP session失败: %s", get_error_detail(e))


def resolve_ttadk_engine_startup_model(
    *,
    agent_type: str,
    cwd: str,
    model_intent: Optional[str],
) -> dict:
    """为 Deep/Spec 引擎统一解析 TTADK 启动模型。

    注意：该函数仅做"启动阶段预校验"，不做执行阶段强校验/纠错。
    统一收敛到 `src.ttadk.startup_common.precheck_ttadk_startup_model()`，避免多处实现漂移。
    """
    from ..ttadk.startup_common import precheck_ttadk_startup_model
    from ..utils.path import normalize_ttadk_cwd

    raw_cwd = cwd
    norm_cwd = normalize_ttadk_cwd(raw_cwd)
    cwd = norm_cwd or raw_cwd
    try:
        if bool(getattr(get_settings(), "ttadk_cwd_debug_enabled", False)):
            logger.debug(
                "[TTADK:CWD] where=%s raw_cwd=%r normalized_cwd=%r",
                "agent_session.resolve_ttadk_engine_startup_model",
                raw_cwd,
                norm_cwd,
            )
    except Exception:
        logger.debug("resolve_ttadk_engine_startup_model: cwd debug logging failed", exc_info=True)

    info = precheck_ttadk_startup_model(agent_type=agent_type, cwd=cwd, model_intent=model_intent)
    # 兼容旧调用方字段名：resolved_model
    # 说明：startup_common 已输出 resolved_model；这里仅做 best-effort 兜底，不覆盖其语义。
    if "resolved_model" not in info:
        info["resolved_model"] = info.get("model")
    # 透出诊断（用于引擎日志/排障，不参与逻辑判断）
    if "diagnostics" not in info:
        info["diagnostics"] = {}
    return info


def create_sync_session(agent_type: str, cwd: str, model_name: Optional[str] = None) -> SyncSession:
    """Factory for creating a sync session by backend.

    - coco/default: ACP backend
    - claude: CLI backend
    - ttadk_*: CLI backend（强隔离：TTADK 前缀不允许拉起 ACP Server）
    """
    from ..coco_model import get_coco_model_manager
    from ..utils.path import normalize_ttadk_cwd

    agent_type = (agent_type or "").lower()
    raw_cwd = cwd
    norm_cwd = normalize_ttadk_cwd(raw_cwd)
    cwd = norm_cwd or raw_cwd
    try:
        if bool(getattr(get_settings(), "ttadk_cwd_debug_enabled", False)):
            logger.debug(
                "[TTADK:CWD] where=%s raw_cwd=%r normalized_cwd=%r",
                "agent_session.create_sync_session",
                raw_cwd,
                norm_cwd,
            )
    except Exception:
        logger.debug("create_sync_session: cwd debug logging failed", exc_info=True)
    if agent_type == "claude":
        return SyncClaudeCLISession(cwd=cwd, model_name=model_name)

    effective_model = model_name
    if not effective_model and agent_type in ("coco", ""):
        effective_model = get_coco_model_manager().get_current_model()

    if agent_type.startswith("ttadk_"):
        # 该工厂只负责构造 session：启动阶段预校验下沉到统一 helper，validated 才透传 -m。
        try:
            from ..ttadk.startup_common import precheck_ttadk_startup_model

            info = precheck_ttadk_startup_model(agent_type=agent_type, cwd=cwd, model_intent=model_name)
            model_name = info.get("model")
            logger.info(
                "[SessionFactory] ttadk precheck(startup): tool=%s input_model=%s model=%s validated=%s source=%s decision=%s fail_phase=%s warnings=%s",
                info.get("tool") or "",
                info.get("input_model") or "",
                (model_name or "(auto)"),
                bool(info.get("validated")),
                info.get("source") or "unknown",
                info.get("decision") or "",
                info.get("fail_phase") or "",
                list(info.get("warnings") or []),
            )
        except Exception:
            logger.debug("create_sync_session: model resolution failed", exc_info=True)
            model_name = None
        # Switch to CLI backend
        return SyncTTADKCLISession(agent_type=agent_type, cwd=cwd, model_name=model_name)

    effective_model = _normalize_acp_startup_model(agent_type or "coco", effective_model)
    return SyncACPSession(agent_type=agent_type or "coco", cwd=cwd, model_name=effective_model)


def create_engine_session(
    agent_type: str,
    cwd: str,
    on_rate_limit: Optional[Callable[[int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    model_name: Optional[str] = None,
    *,
    thread_id: Optional[str] = None,
    auto_approve: bool = False,
    require_tool_filter: bool = False,
    startup_timeout: Optional[float] = None,
    startup_retries: Optional[int] = None,
    startup_log_failures: Optional[bool] = None,
) -> SyncSession:
    """Create and start a session for Deep/Spec/Slock engines.

    - Claude: CLI backend (no ACP retry needed)
    - ttadk_*: CLI backend (no ACP retry needed)
    - Others: ACP backend with retry and progressive timeout

    If rate_limit_retry_enabled is True in settings, the returned session
    is wrapped with RateLimitAwareSession for automatic retry on throttling.

    Keyword args:
        thread_id: Optional isolation key for concurrent sessions (e.g. slock agents).
        auto_approve: If True, suppress interactive confirmation prompts (slock mode).
        require_tool_filter: If True, choose a backend that exposes set_tool_filter.
        startup_timeout: Optional ACP startup budget override.
        startup_retries: Optional ACP startup attempt override.
        startup_log_failures: Override startup diagnostics logging for expected
            best-effort callers such as the one-shot NLI classifier.
    """
    from ..acp.sync_adapter import start_session_with_retry
    from ..coco_model import get_coco_model_manager
    from ..utils.path import normalize_ttadk_cwd

    settings = get_settings()
    agent_type = (agent_type or "").lower()
    employee_env = current_employee_session_environment()
    # TTADK/引擎侧 cwd 归一化：避免传入 "." 导致 TTADK 项目级缓存不落盘。
    raw_cwd = cwd
    norm_cwd = normalize_ttadk_cwd(raw_cwd)
    cwd = norm_cwd or raw_cwd
    try:
        if bool(getattr(get_settings(), "ttadk_cwd_debug_enabled", False)):
            logger.debug(
                "[TTADK:CWD] where=%s raw_cwd=%r normalized_cwd=%r",
                "agent_session.create_engine_session",
                raw_cwd,
                norm_cwd,
            )
    except Exception:
        logger.debug("create_engine_session: cwd debug logging failed", exc_info=True)

    # 日志语义：
    # - TTADK: 传入的可能是"友好名/意图"，并不等于最终透传 -m 的真实模型名；避免用 `model=` 误导。
    # - 非 TTADK: 依旧输出 `model=` 便于排障。
    if agent_type.startswith("ttadk_"):
        logger.info(
            "[SessionFactory] create_engine_session: agent=%s cwd=%s input_model=%s (CLI mode)",
            agent_type or "coco",
            cwd,
            model_name,
        )
    else:
        logger.info(
            "[SessionFactory] create_engine_session: agent=%s cwd=%s model=%s",
            agent_type or "coco",
            cwd,
            model_name,
        )

    if agent_type == "claude" and (not require_tool_filter or employee_env is not None):
        session: SyncSession = SyncClaudeCLISession(
            cwd=cwd,
            model_name=model_name,
            employee_process_env=employee_env,
        )
        session.start()
    elif agent_type.startswith("ttadk_"):
        # TTADK CLI mode: precheck model then use CLI session
        # Switch to CLI backend as requested, replacing the previous ACP startup coordinator.
        try:
            from ..ttadk.startup_common import precheck_ttadk_startup_model

            # 1. Precheck to resolve model name
            info = precheck_ttadk_startup_model(agent_type=agent_type, cwd=cwd, model_intent=model_name)

            resolved_model = info.get("model")  # Validated model ID or None (auto)

            logger.info(
                "[SessionFactory] ttadk cli startup: tool=%s input_model=%s model=%s validated=%s source=%s warnings=%s",
                info.get("tool") or "",
                info.get("input_model") or "",
                (resolved_model or "(auto)"),
                bool(info.get("validated")),
                info.get("source") or "unknown",
                list(info.get("warnings") or []),
            )

            # 2. Create CLI session
            session = SyncTTADKCLISession(
                agent_type=agent_type,
                cwd=cwd,
                model_name=resolved_model,
                employee_process_env=employee_env,
            )
            session.start()

        except Exception:
            raise
    else:
        effective_model = model_name
        if not effective_model and agent_type in ("coco", ""):
            effective_model = get_coco_model_manager().get_current_model()
        elif not effective_model and agent_type == "traex":
            try:
                from ..acp.providers import get_providers, tool_registry
                get_providers()
                provider = tool_registry.get_provider("traex")
                if provider and hasattr(provider, "get_default_model"):
                    effective_model = provider.get_default_model()
                    logger.info("[SessionFactory] traex default model resolved: %s", effective_model)
            except Exception:
                pass

        effective_model = _normalize_acp_startup_model(agent_type or "coco", effective_model)
        startup_kwargs: dict[str, object] = {}
        if startup_retries is not None:
            startup_kwargs["retries"] = startup_retries
        if startup_log_failures is not None:
            startup_kwargs["log_failures"] = startup_log_failures
        if employee_env is not None:
            startup_kwargs["env"] = employee_env
        session = start_session_with_retry(
            agent_type=agent_type or "coco",
            cwd=cwd,
            startup_timeout=(
                settings.acp_startup_timeout
                if startup_timeout is None
                else startup_timeout
            ),
            model_name=effective_model,
            **startup_kwargs,
        )

    if settings.rate_limit_retry_enabled:
        session = RateLimitAwareSession(
            inner=session,
            on_rate_limit=on_rate_limit,
            cancel_event=cancel_event,
        )

    # Model failure (compaction/loop/failover) auto-repair wrapper.
    # 说明：该 wrapper 只在 send_prompt 阶段生效，不影响启动时 TTADK/ACP 的既有重试逻辑。
    try:
        session = ModelFailureAwareSession(
            inner=session,
            on_rate_limit=on_rate_limit,
            cancel_event=cancel_event,
        )
    except Exception:
        # best-effort: wrapper 失败不应影响正常会话创建
        logger.debug("create_engine_session: ModelFailureAwareSession wrapper failed", exc_info=True)

    return session


def create_auxiliary_session(
    agent_type: str,
    cwd: str,
    model_name: Optional[str] = None,
    *,
    thread_id: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    startup_timeout: Optional[float] = None,
    startup_retries: Optional[int] = None,
    startup_log_failures: Optional[bool] = None,
) -> SyncSession:
    """Create a text-only coordination/classification session with no tools.

    Auxiliary sessions deliberately skip ``ModelFailureAwareSession`` because
    that wrapper may replace its inner session during repair. A replacement
    would otherwise lose the deny-all filter. Rate-limit retry remains safe
    because it never replaces the filtered inner session.
    """

    from ..acp.sync_adapter import start_session_with_retry
    from ..coco_model import get_coco_model_manager
    from ..utils.path import normalize_ttadk_cwd

    settings = get_settings()
    normalized_agent = (agent_type or "coco").strip().lower()
    if normalized_agent.startswith("ttadk_"):
        raise AuxiliarySessionPermissionError(
            "TTADK CLI cannot enforce the auxiliary deny-all tool profile"
        )

    normalized_cwd = normalize_ttadk_cwd(cwd) or cwd
    effective_model = model_name
    if not effective_model and normalized_agent == "coco":
        effective_model = get_coco_model_manager().get_current_model()
    elif not effective_model and normalized_agent == "traex":
        try:
            from ..acp.providers import get_providers, tool_registry

            get_providers()
            provider = tool_registry.get_provider("traex")
            if provider and hasattr(provider, "get_default_model"):
                effective_model = provider.get_default_model()
        except Exception:
            logger.debug(
                "create_auxiliary_session: traex default model resolution failed",
                exc_info=True,
            )
    effective_model = _normalize_acp_startup_model(
        normalized_agent,
        effective_model,
    )

    startup_kwargs: dict[str, object] = {}
    if startup_retries is not None:
        startup_kwargs["retries"] = startup_retries
    if startup_log_failures is not None:
        startup_kwargs["log_failures"] = startup_log_failures

    logger.info(
        "[SessionFactory] create_auxiliary_session: agent=%s cwd=%s "
        "model=%s thread=%s profile=deny_all",
        normalized_agent,
        normalized_cwd,
        effective_model,
        thread_id or "",
    )
    session: SyncSession = start_session_with_retry(
        agent_type=normalized_agent,
        cwd=normalized_cwd,
        startup_timeout=(
            settings.acp_startup_timeout
            if startup_timeout is None
            else startup_timeout
        ),
        model_name=effective_model,
        **startup_kwargs,
    )
    if settings.rate_limit_retry_enabled:
        session = RateLimitAwareSession(
            inner=session,
            cancel_event=cancel_event,
        )
    try:
        apply_auxiliary_tool_profile(session)
    except Exception:
        close_session_safely(session)
        raise
    return session


def create_review_session(
    agent_type: str,
    cwd: str,
    model_name: Optional[str] = None,
    startup_timeout: Optional[float] = None,
) -> SyncSession:
    """Create a short-lived session dedicated to review prompts.

    Differs from `create_engine_session` in two ways:
    - Skips `RateLimitAwareSession` / `ModelFailureAwareSession` wrappers.
      Review is best-effort — on failure the pipeline falls back to other
      strategies (lint, skip) instead of burning retries.
    - Caller is expected to close the session after use; see
      `EphemeralReviewSession` for a context-managed convenience.

    Allows `agent_type` to differ from the build agent (heterogeneous review).

    Args:
        startup_timeout: Optional override for the ACP startup timeout.
            When None, falls back to ``settings.acp_startup_timeout``.
    """
    from ..acp.sync_adapter import start_session_with_retry
    from ..coco_model import get_coco_model_manager
    from ..utils.path import normalize_ttadk_cwd

    settings = get_settings()
    agent_type = (agent_type or "coco").lower()
    cwd = normalize_ttadk_cwd(cwd) or cwd
    effective_startup_timeout = startup_timeout if startup_timeout is not None else settings.acp_startup_timeout

    logger.info(
        "[SessionFactory] create_review_session: agent=%s cwd=%s model=%s startup_timeout=%s",
        agent_type, cwd, model_name, effective_startup_timeout,
    )

    if agent_type == "claude":
        session: SyncSession = SyncClaudeCLISession(cwd=cwd)
        session.start()
        return session

    if agent_type.startswith("ttadk_"):
        from ..ttadk.startup_common import precheck_ttadk_startup_model
        info = precheck_ttadk_startup_model(
            agent_type=agent_type, cwd=cwd, model_intent=model_name
        )
        session = SyncTTADKCLISession(
            agent_type=agent_type, cwd=cwd, model_name=info.get("model")
        )
        session.start()
        return session

    effective_model = model_name
    if not effective_model and agent_type in ("coco", ""):
        effective_model = get_coco_model_manager().get_current_model()
    effective_model = _normalize_acp_startup_model(agent_type, effective_model)
    return start_session_with_retry(
        agent_type=agent_type,
        cwd=cwd,
        startup_timeout=float(effective_startup_timeout),
        model_name=effective_model,
    )


class EphemeralReviewSession:
    """Context manager: fresh review session per `with` block; auto-close on exit.

    Use to isolate review from the build session so review prompts run on a
    clean, small ACP context. Create anew per cycle — do not reuse across cycles.

    Attributes:
        startup_elapsed_s: Wall-clock seconds spent inside create_review_session
            during __enter__. Set even on failure so callers can distinguish
            startup-time failures from prompt-time failures.
    """

    def __init__(
        self,
        agent_type: str,
        cwd: str,
        model_name: Optional[str] = None,
        startup_timeout: Optional[float] = None,
    ):
        self._agent_type = agent_type
        self._cwd = cwd
        self._model_name = model_name
        self._startup_timeout = startup_timeout
        self._session: Optional[SyncSession] = None
        self.startup_elapsed_s: float = 0.0
        self.session_started: bool = False

    def __enter__(self) -> SyncSession:
        import time
        t0 = time.perf_counter()
        try:
            self._session = create_review_session(
                self._agent_type,
                self._cwd,
                self._model_name,
                startup_timeout=self._startup_timeout,
            )
            self.session_started = True
            return self._session
        finally:
            self.startup_elapsed_s = time.perf_counter() - t0

    def __exit__(self, *exc) -> None:
        if self._session is None:
            return
        try:
            close = getattr(self._session, "close", None)
            if callable(close):
                close()
        except Exception as e:
            logger.debug("[EphemeralReviewSession] close failed: %s", repr(e))
        finally:
            self._session = None


def create_sync_session_for_worktree(
    *,
    provider: str = "",
    tool_name: str = "",
    working_dir: str,
    model_name: Optional[str] = None,
) -> SyncSession:
    """Create a sync session for a worktree unit.

    Maps *provider*/*tool_name* to an ``agent_type`` understood by
    :func:`create_sync_session` and uses *working_dir* as the session cwd
    so the agent operates inside its dedicated worktree directory.

    Provider mapping:
    - ``"ttadk"`` → ``"ttadk_{tool_name}"``  (TTADK CLI bridge)
    - ``"cli"``   → ``tool_name`` as-is, typically ``"claude"``
    - ``"acp"``   → ``tool_name`` as-is, e.g. ``"coco"``/``"codex"``
    - fallback    → ``tool_name or "coco"``
    """
    provider = (provider or "").strip().lower()
    tool_name = (tool_name or "").strip().lower()

    if provider == "ttadk":
        agent_type = f"ttadk_{tool_name}" if tool_name else "ttadk_coco"
    elif provider == "cli":
        agent_type = tool_name or "claude"
    else:
        agent_type = tool_name or "coco"

    return create_sync_session(agent_type=agent_type, cwd=working_dir, model_name=model_name)
