"""Session construction for all programming backends."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any, Callable, Optional, cast

from ..acp.agy_selection import requires_agy_cli_orchestrator_session
from ..acp.providers import normalize_acp_model_name
from ..config import get_settings
from ..utils.errors import get_error_detail
from .agy_cli import SyncAGYCLISession
from .backend_resolver import (
    cli_command_for_agent,
    is_cli_backend,
    is_codex_cli_backend,
)
from .claude_cli import ClaudeCLIConfig, SyncClaudeCLISession
from .codex_cli import CodexCLIConfig, SyncCodexCLISession
from .protocol import SyncSession
from .wrappers import ModelFailureAwareSession, RateLimitAwareSession

logger = logging.getLogger(__name__)


def _normalize_model(agent_type: str, model_name: str | None) -> str | None:
    if not model_name or agent_type in {"claude", "claude_w", "codex_w", "traex"}:
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
    if model is None and agent == "traex":
        try:
            from ..acp.providers import get_providers, tool_registry

            get_providers()
            provider = tool_registry.get_provider("traex")
            if provider and hasattr(provider, "get_default_model"):
                model = provider.get_default_model()
        except Exception:
            logger.debug("Traex default model resolution failed", exc_info=True)
    return agent, normalized_cwd, _normalize_model(agent, model)


def _apply_runtime_settings(
    session: object,
    runtime_settings: Mapping[str, object] | None,
) -> None:
    """Apply runtime settings to a session when the backend exposes ``set_runtime_settings``."""
    if runtime_settings is None:
        return
    setter = getattr(session, "set_runtime_settings", None)
    if callable(setter):
        setter(dict(runtime_settings))


def _start_base_session(
    agent_type: str,
    cwd: str,
    model_name: str | None,
    *,
    capture_full_tool_content: bool = False,
    startup_timeout: float | None = None,
    startup_retries: int | None = None,
    startup_log_failures: bool | None = None,
    runtime_settings: Mapping[str, object] | None = None,
) -> SyncSession:
    """Start exactly one CLI or ACP transport with shared startup arguments."""
    if agent_type in {"agy", "agy_b"} and requires_agy_cli_orchestrator_session(model_name):
        if agent_type == "agy_b":
            from .agy_cli import AGYCLIConfig

            session: SyncSession = SyncAGYCLISession(
                cwd=cwd,
                config=AGYCLIConfig(command="agy-b"),
                model_name=model_name,
                agent_type="agy_b",
            )
        else:
            session = SyncAGYCLISession(
                cwd=cwd,
                model_name=model_name,
            )
        _apply_runtime_settings(session, runtime_settings)
        session.start()
        return session

    if is_cli_backend(agent_type):
        command = cli_command_for_agent(agent_type)
        if is_codex_cli_backend(agent_type):
            session = SyncCodexCLISession(
                cwd=cwd,
                model_name=model_name,
                config=CodexCLIConfig(command=command),
            )
        elif command != "claude":
            session = SyncClaudeCLISession(
                cwd=cwd,
                model_name=model_name,
                config=ClaudeCLIConfig(command=command),
            )
        else:
            session = SyncClaudeCLISession(
                cwd=cwd,
                model_name=model_name,
            )
        _apply_runtime_settings(session, runtime_settings)
        session.start()
        return session

    from ..acp.sync_adapter import start_session_with_retry

    acp_kwargs: dict[str, Any] = {}
    if startup_retries is not None:
        acp_kwargs["retries"] = startup_retries
    if startup_log_failures is not None:
        acp_kwargs["log_failures"] = startup_log_failures
    if capture_full_tool_content:
        acp_kwargs["capture_full_tool_content"] = True
    acp_session = start_session_with_retry(
        agent_type=agent_type,
        cwd=cwd,
        startup_timeout=(
            get_settings().acp_startup_timeout
            if startup_timeout is None
            else startup_timeout
        ),
        model_name=model_name,
        **acp_kwargs,
    )
    _apply_runtime_settings(acp_session, runtime_settings)
    return cast(SyncSession, acp_session)


def close_session_safely(session: SyncSession | None) -> None:
    if session is None:
        return
    try:
        session.close()
    except Exception as exc:
        logger.debug("关闭旧 session 失败: %s", get_error_detail(exc))


def _call_start_base_session(
    agent: str,
    normalized_cwd: str,
    model: str | None,
    *,
    capture_full_tool_content: bool = False,
    startup_timeout: float | None = None,
    startup_retries: int | None = None,
    startup_log_failures: bool | None = None,
    runtime_settings: Mapping[str, object] | None = None,
) -> SyncSession:
    """Invoke ``_start_base_session``, forwarding ``runtime_settings`` only when non-None."""
    if runtime_settings is not None:
        return _start_base_session(
            agent,
            normalized_cwd,
            model,
            capture_full_tool_content=capture_full_tool_content,
            startup_timeout=startup_timeout,
            startup_retries=startup_retries,
            startup_log_failures=startup_log_failures,
            runtime_settings=runtime_settings,
        )
    return _start_base_session(
        agent,
        normalized_cwd,
        model,
        capture_full_tool_content=capture_full_tool_content,
        startup_timeout=startup_timeout,
        startup_retries=startup_retries,
        startup_log_failures=startup_log_failures,
    )


def create_programming_session(
    agent_type: str,
    cwd: str,
    model_name: Optional[str] = None,
    *,
    capture_full_tool_content: bool = False,
    startup_timeout: Optional[float] = None,
    startup_retries: Optional[int] = None,
    startup_log_failures: Optional[bool] = None,
    runtime_settings: dict[str, object] | None = None,
) -> SyncSession:
    """Create a direct programming session and apply optional runtime settings."""
    agent, normalized_cwd, model = _resolve_inputs(agent_type, cwd, model_name)
    logger.info(
        "[SessionFactory] programming agent=%s cwd=%s model=%s",
        agent,
        normalized_cwd,
        model,
    )
    return _call_start_base_session(
        agent,
        normalized_cwd,
        model,
        capture_full_tool_content=capture_full_tool_content,
        startup_timeout=startup_timeout,
        startup_retries=startup_retries,
        startup_log_failures=startup_log_failures,
        runtime_settings=runtime_settings,
    )


def create_engine_session(
    agent_type: str,
    cwd: str,
    on_rate_limit: Optional[Callable[[int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    model_name: Optional[str] = None,
    *,
    thread_id: Optional[str] = None,
    capture_full_tool_content: bool = False,
    startup_timeout: Optional[float] = None,
    startup_retries: Optional[int] = None,
    startup_log_failures: Optional[bool] = None,
    runtime_settings: dict[str, object] | None = None,
) -> SyncSession:
    """Create the long-lived Deep/Spec/Workflow session chain."""
    agent, normalized_cwd, model = _resolve_inputs(agent_type, cwd, model_name)
    logger.info(
        "[SessionFactory] engine agent=%s cwd=%s model=%s thread=%s",
        agent,
        normalized_cwd,
        model,
        thread_id or "",
    )
    session = _call_start_base_session(
        agent,
        normalized_cwd,
        model,
        capture_full_tool_content=capture_full_tool_content,
        startup_timeout=startup_timeout,
        startup_retries=startup_retries,
        startup_log_failures=startup_log_failures,
        runtime_settings=runtime_settings,
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
    runtime_settings: dict[str, object] | None = None,
) -> SyncSession:
    """Create an unprivileged-by-convention coordination session."""
    agent, normalized_cwd, model = _resolve_inputs(agent_type, cwd, model_name)
    logger.info(
        "[SessionFactory] auxiliary agent=%s cwd=%s model=%s thread=%s",
        agent,
        normalized_cwd,
        model,
        thread_id or "",
    )
    session = _call_start_base_session(
        agent,
        normalized_cwd,
        model,
        startup_timeout=startup_timeout,
        startup_retries=startup_retries,
        startup_log_failures=startup_log_failures,
        runtime_settings=runtime_settings,
    )
    if get_settings().rate_limit_retry_enabled:
        session = RateLimitAwareSession(session, cancel_event=cancel_event)
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
                startup_timeout=self._startup_timeout,
            )
            self.session_started = True
            return self._session
        finally:
            self.startup_elapsed_s = time.perf_counter() - started

    def __exit__(self, *exc) -> None:
        close_session_safely(self._session)
        self._session = None
