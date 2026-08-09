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
    apply_auxiliary_tool_profile,
)
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
    boundary so Deep/Spec/Review/employee share the same normalization.
    """
    agent = (agent_type or "").strip().lower()
    if (
        not model_name
        or agent in {"claude", "traex"}
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



def create_sync_session(agent_type: str, cwd: str, model_name: Optional[str] = None) -> SyncSession:
    """Factory for creating a sync session by backend.

    - coco/default: ACP backend
    - claude: CLI backend
    """
    from ..coco_model import get_coco_model_manager
    from ..utils.path import normalize_session_cwd

    agent_type = (agent_type or "").lower()
    cwd = normalize_session_cwd(cwd) or cwd
    if agent_type == "claude":
        return SyncClaudeCLISession(cwd=cwd, model_name=model_name)

    effective_model = model_name
    if not effective_model and agent_type in ("coco", ""):
        effective_model = get_coco_model_manager().get_current_model()

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
    """Create and start a session for Deep/Spec/employee engines.

    - Claude: CLI backend (no ACP retry needed)
    - Others: ACP backend with retry and progressive timeout

    If rate_limit_retry_enabled is True in settings, the returned session
    is wrapped with RateLimitAwareSession for automatic retry on throttling.

    Keyword args:
        thread_id: Optional isolation key for concurrent sessions (e.g. employee agents).
        auto_approve: If True, suppress interactive confirmation prompts (employee mode).
        require_tool_filter: If True, choose a backend that exposes set_tool_filter.
        startup_timeout: Optional ACP startup budget override.
        startup_retries: Optional ACP startup attempt override.
        startup_log_failures: Override startup diagnostics logging for expected
            best-effort callers such as the one-shot NLI classifier.
    """
    from ..acp.sync_adapter import start_session_with_retry
    from ..coco_model import get_coco_model_manager
    from ..utils.path import normalize_session_cwd

    settings = get_settings()
    agent_type = (agent_type or "").lower()
    cwd = normalize_session_cwd(cwd) or cwd
    employee_env = current_employee_session_environment()
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
    # The wrapper only affects prompt execution, not startup retries.
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
    from ..utils.path import normalize_session_cwd

    settings = get_settings()
    normalized_agent = (agent_type or "coco").strip().lower()
    normalized_cwd = normalize_session_cwd(cwd) or cwd
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
    from ..utils.path import normalize_session_cwd

    settings = get_settings()
    agent_type = (agent_type or "coco").lower()
    cwd = normalize_session_cwd(cwd) or cwd
    effective_startup_timeout = startup_timeout if startup_timeout is not None else settings.acp_startup_timeout

    logger.info(
        "[SessionFactory] create_review_session: agent=%s cwd=%s model=%s startup_timeout=%s",
        agent_type, cwd, model_name, effective_startup_timeout,
    )

    if agent_type == "claude":
        session: SyncSession = SyncClaudeCLISession(
            cwd=cwd,
            model_name=model_name,
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
