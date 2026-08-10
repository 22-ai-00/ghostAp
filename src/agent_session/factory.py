"""Session construction for all programming backends."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Optional

from ..acp.providers import normalize_acp_model_name
from ..config import get_settings
from ..utils.errors import get_error_detail
from .backend_resolver import is_cli_backend
from .claude_cli import SyncClaudeCLISession
from .protocol import SyncSession
from .tool_permissions import apply_auxiliary_tool_profile
from .wrappers import ModelFailureAwareSession, RateLimitAwareSession

logger = logging.getLogger(__name__)
_EMPLOYEE_SESSION_ENV: ContextVar[dict[str, str] | None] = ContextVar(
    "employee_session_env", default=None
)


@contextmanager
def employee_session_environment(env: dict[str, str]):
    """Expose one validated employee environment to the synchronous factory."""
    if _EMPLOYEE_SESSION_ENV.get() is not None:
        raise RuntimeError("nested employee session environment is forbidden")
    if not isinstance(env, dict) or not env or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in env.items()
    ):
        raise ValueError("employee session environment must be explicit")
    token = _EMPLOYEE_SESSION_ENV.set(dict(env))
    try:
        yield
    finally:
        _EMPLOYEE_SESSION_ENV.reset(token)


def current_employee_session_environment() -> dict[str, str] | None:
    value = _EMPLOYEE_SESSION_ENV.get()
    return None if value is None else dict(value)


def _normalize_model(agent_type: str, model_name: str | None) -> str | None:
    if not model_name or agent_type in {"claude", "traex"}:
        return model_name
    normalized = normalize_acp_model_name(agent_type, model_name)
    if normalized != model_name:
        logger.info(
            "[SessionFactory] normalized ACP model: agent=%s selected_model=%s backend_model=%s",
            agent_type,
            model_name,
            normalized,
        )
    return normalized


def _resolve_inputs(
    agent_type: str,
    cwd: str,
    model_name: str | None,
) -> tuple[str, str, str | None]:
    """Normalize backend, cwd and provider default model once for every lane."""
    from ..utils.path import normalize_session_cwd

    agent = (agent_type or "coco").strip().lower()
    normalized_cwd = normalize_session_cwd(cwd) or cwd
    model = (model_name or "").strip() or None
    if model is None and agent == "coco":
        from ..coco_model import get_coco_model_manager

        model = get_coco_model_manager().get_current_model()
    elif model is None and agent == "traex":
        try:
            from ..acp.providers import get_providers, tool_registry

            get_providers()
            provider = tool_registry.get_provider("traex")
            if provider and hasattr(provider, "get_default_model"):
                model = provider.get_default_model()
        except Exception:
            logger.debug("Traex default model resolution failed", exc_info=True)
    return agent, normalized_cwd, _normalize_model(agent, model)


def _start_base_session(
    agent_type: str,
    cwd: str,
    model_name: str | None,
    *,
    allow_cli: bool,
    auto_approve: bool | None = None,
    employee_env: dict[str, str] | None = None,
    startup_timeout: float | None = None,
    startup_retries: int | None = None,
    startup_log_failures: bool | None = None,
) -> SyncSession:
    """Start exactly one CLI or ACP transport with shared startup arguments."""
    if agent_type == "claude" and allow_cli:
        session: SyncSession = SyncClaudeCLISession(
            cwd=cwd,
            model_name=model_name,
            employee_process_env=employee_env,
        )
        session.start()
        return session

    from ..acp.sync_adapter import start_session_with_retry

    kwargs: dict[str, object] = {}
    if startup_retries is not None:
        kwargs["retries"] = startup_retries
    if startup_log_failures is not None:
        kwargs["log_failures"] = startup_log_failures
    if employee_env is not None:
        kwargs["env"] = employee_env
    return start_session_with_retry(
        agent_type=agent_type,
        cwd=cwd,
        startup_timeout=(
            get_settings().acp_startup_timeout
            if startup_timeout is None
            else startup_timeout
        ),
        model_name=model_name,
        auto_approve=auto_approve,
        **kwargs,
    )


def close_session_safely(session: SyncSession | None) -> None:
    if session is None:
        return
    try:
        session.close()
    except Exception as exc:
        logger.debug("关闭旧 session 失败: %s", get_error_detail(exc))


def create_engine_session(
    agent_type: str,
    cwd: str,
    on_rate_limit: Optional[Callable[[int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    model_name: Optional[str] = None,
    *,
    thread_id: Optional[str] = None,
    auto_approve: bool | None = None,
    require_tool_filter: bool = False,
    startup_timeout: Optional[float] = None,
    startup_retries: Optional[int] = None,
    startup_log_failures: Optional[bool] = None,
) -> SyncSession:
    """Create the long-lived Deep/Spec/Workflow/employee session chain."""
    agent, normalized_cwd, model = _resolve_inputs(agent_type, cwd, model_name)
    employee_env = current_employee_session_environment()
    logger.info(
        "[SessionFactory] engine agent=%s cwd=%s model=%s thread=%s auto=%s",
        agent,
        normalized_cwd,
        model,
        thread_id or "",
        auto_approve,
    )
    session = _start_base_session(
        agent,
        normalized_cwd,
        model,
        allow_cli=not require_tool_filter or employee_env is not None,
        auto_approve=auto_approve,
        employee_env=employee_env,
        startup_timeout=startup_timeout,
        startup_retries=startup_retries,
        startup_log_failures=startup_log_failures,
    )
    settings = get_settings()
    if settings.rate_limit_retry_enabled:
        session = RateLimitAwareSession(session, on_rate_limit, cancel_event)
    try:
        return ModelFailureAwareSession(
            session,
            on_rate_limit=on_rate_limit,
            cancel_event=cancel_event,
        )
    except Exception:
        logger.debug("Model failure wrapper unavailable", exc_info=True)
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
    """Create a coordination session with a mandatory deny-all tool filter."""
    if is_cli_backend(agent_type):
        raise RuntimeError(
            "Claude CLI backend does not support auxiliary ACP transport"
        )
    agent, normalized_cwd, model = _resolve_inputs(agent_type, cwd, model_name)
    logger.info(
        "[SessionFactory] auxiliary agent=%s cwd=%s model=%s thread=%s profile=deny_all",
        agent,
        normalized_cwd,
        model,
        thread_id or "",
    )
    session = _start_base_session(
        agent,
        normalized_cwd,
        model,
        allow_cli=False,
        startup_timeout=startup_timeout,
        startup_retries=startup_retries,
        startup_log_failures=startup_log_failures,
    )
    if get_settings().rate_limit_retry_enabled:
        session = RateLimitAwareSession(session, cancel_event=cancel_event)
    try:
        apply_auxiliary_tool_profile(session)
    except Exception:
        close_session_safely(session)
        raise
    return session


class EphemeralReviewSession:
    """Fresh unwrapped review session, closed at context exit."""

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
        self._session: SyncSession | None = None
        self.startup_elapsed_s = 0.0
        self.session_started = False

    def __enter__(self) -> SyncSession:
        started = time.perf_counter()
        try:
            agent, cwd, model = _resolve_inputs(
                self._agent_type,
                self._cwd,
                self._model_name,
            )
            self._session = _start_base_session(
                agent,
                cwd,
                model,
                allow_cli=True,
                startup_timeout=self._startup_timeout,
            )
            self.session_started = True
            return self._session
        finally:
            self.startup_elapsed_s = time.perf_counter() - started

    def __exit__(self, *exc) -> None:
        close_session_safely(self._session)
        self._session = None
