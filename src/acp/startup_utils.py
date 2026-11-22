"""Shared ACP startup orchestration and diagnostics helpers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StartupOperationalError(RuntimeError):
    """Expected startup/runtime boundary failure that may retry or fallback."""


class StartupFatalError(RuntimeError):
    """Unexpected startup failure that must not be treated as operational."""


STARTUP_OPERATIONAL_EXCEPTIONS = (
    StartupOperationalError,
    TimeoutError,
    OSError,
    ConnectionError,
    EOFError,
)


class StartupBackend(str, Enum):
    """Backend family selected for a session startup request."""

    ACP = "acp"
    CLI = "cli"


@dataclass(frozen=True)
class RetryPlan:
    """ACP/CLI startup retry plan with progressive timeout semantics."""

    attempts: int
    startup_timeout: float

    def timeout_for_attempt(self, attempt: int) -> float:
        safe_attempt = max(1, int(attempt or 1))
        return float(self.startup_timeout or 60) * (1.0 + 0.5 * (safe_attempt - 1))


@dataclass(frozen=True)
class SessionStartupRequest:
    key: str
    cwd: str
    startup_timeout: float
    project_id: Optional[str]
    session_id: Optional[str]
    effective_agent_type: str
    model_name: Optional[str]
    retries: int
    deadline_monotonic: float | None = None


@dataclass(frozen=True)
class SessionStartupResult:
    session: Any
    actual_id: str
    effective_agent_type: str
    model_name: Optional[str] = None


def select_startup_backend(
    agent_type: str,
    *,
    model_name: Optional[str] = None,
) -> StartupBackend:
    from ..acp.agy_selection import requires_agy_cli_orchestrator_session
    from ..agent_session.backend_resolver import is_cli_backend

    normalized = (agent_type or "").lower().strip()
    if is_cli_backend(normalized):
        return StartupBackend.CLI
    if normalized == "agy" and requires_agy_cli_orchestrator_session(model_name):
        return StartupBackend.CLI
    return StartupBackend.ACP


def normalize_startup_cwd(cwd: str, *, normalize_fn: Callable[[str], str | None]) -> str:
    raw_cwd = cwd or "."
    try:
        return normalize_fn(raw_cwd) or raw_cwd
    except (OSError, RuntimeError, TypeError, ValueError):
        logger.warning("Error while normalizing startup cwd", exc_info=True)
        return raw_cwd


def build_retry_plan(
    agent_type: str,
    *,
    retries: int,
    startup_timeout: float,
    model_name: Optional[str] = None,
) -> RetryPlan:
    attempts = (
        1
        if select_startup_backend(agent_type, model_name=model_name) == StartupBackend.CLI
        else max(1, int(retries or 1))
    )
    return RetryPlan(attempts=attempts, startup_timeout=float(startup_timeout or 60))


def run_startup_operation(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run an external startup adapter call with a narrow fail-fast boundary."""
    try:
        return fn(*args, **kwargs)
    except (TypeError, AssertionError, *STARTUP_OPERATIONAL_EXCEPTIONS):
        raise
    except Exception as exc:
        raise StartupFatalError(str(exc) or type(exc).__name__) from exc


class AcpRetryStarter:
    """Session construction/closing helpers kept for bridge/startup call sites."""

    @staticmethod
    def _build_session(
        *,
        backend: StartupBackend,
        agent_type: str,
        cwd: str,
        model_name: Optional[str],
        acp_session_cls: Any,
        cli_session_cls: Any,
    ) -> Any:
        if backend == StartupBackend.CLI:
            if (agent_type or "").lower().strip() == "agy":
                from ..agent_session.agy_cli import SyncAGYCLISession

                return SyncAGYCLISession(
                    cwd=cwd or ".",
                    model_name=model_name,
                )
            from ..agent_session.backend_resolver import cli_command_for_agent, is_codex_cli_backend

            command = cli_command_for_agent(agent_type)
            if is_codex_cli_backend(agent_type):
                from ..agent_session.codex_cli import CodexCLIConfig, SyncCodexCLISession

                return SyncCodexCLISession(
                    cwd=cwd or ".",
                    config=CodexCLIConfig(command=command),
                    model_name=model_name,
                )
            from ..agent_session.claude_cli import ClaudeCLIConfig

            return cli_session_cls(
                cwd=cwd or ".",
                config=ClaudeCLIConfig(command=command),
                model_name=model_name,
            )
        if model_name:
            try:
                return acp_session_cls(agent_type=agent_type, cwd=cwd or ".", model_name=model_name)
            except TypeError:
                return acp_session_cls(agent_type=agent_type, cwd=cwd or ".")
        return acp_session_cls(agent_type=agent_type, cwd=cwd or ".")

    @staticmethod
    def _describe_session(session: Any) -> str:
        try:
            return session.describe_agent()
        except (AttributeError, TypeError):
            logger.warning("Error while describing agent", exc_info=True)
            return ""

    @staticmethod
    def _close_session(session: Any) -> None:
        try:
            if session:
                session.close()
        except (*STARTUP_OPERATIONAL_EXCEPTIONS, AttributeError):
            logger.warning("Error while closing session", exc_info=True)


def _default_sync_acp_session_cls() -> Any:
    from .sync_adapter import SyncACPSession

    return SyncACPSession


def _default_sync_claude_cli_session_cls() -> Any:
    from ..agent_session.claude_cli import SyncClaudeCLISession

    return SyncClaudeCLISession


def _default_get_settings() -> Any:
    from ..config import get_settings

    return get_settings()


class SessionStartupCoordinator:
    """Coordinates backend session startup within the ACP startup boundary."""

    def __init__(
        self,
        *,
        manager_agent_type: str,
        session_starter: Optional[Callable[..., tuple[Any, str, dict]]] = None,
        session_telemetry: Any,
        sync_acp_session_cls: Any | None = None,
        sync_claude_cli_session_cls: Any | None = None,
        get_settings_fn: Callable[[], Any] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        monotonic_fn: Callable[[], float] | None = None,
    ) -> None:
        self._manager_agent_type = manager_agent_type
        self._session_starter = session_starter
        self._session_telemetry = session_telemetry
        self._sync_acp_session_cls = sync_acp_session_cls
        self._sync_claude_cli_session_cls = sync_claude_cli_session_cls
        self._get_settings = get_settings_fn or _default_get_settings
        self._sleep = sleep_fn or time.sleep
        self._monotonic = monotonic_fn or time.monotonic

    def _try_injected_starter(
        self,
        *,
        request: SessionStartupRequest,
        retries: int,
        remaining: float,
    ) -> tuple[Any, str, dict[str, Any] | None] | None:
        agent_type = request.effective_agent_type
        if (
            not callable(self._session_starter)
            or select_startup_backend(agent_type, model_name=request.model_name) != StartupBackend.ACP
        ):
            return None
        try:
            return self._session_starter(
                agent_type=agent_type,
                retries=retries,
                cwd=request.cwd or ".",
                startup_timeout=remaining,
                model_name=request.model_name,
                session_id=request.session_id,
                project_id=request.project_id,
            )
        except STARTUP_OPERATIONAL_EXCEPTIONS:
            return None

    def _resume_requested_session(
        self,
        result: SessionStartupResult,
        *,
        request: SessionStartupRequest,
        deadline: float,
    ) -> SessionStartupResult:
        target_session_id = str(request.session_id or "").strip()
        if not target_session_id:
            return result
        current_session_id = str(getattr(result.session, "session_id", "") or "").strip()
        if current_session_id == target_session_id:
            return replace(result, actual_id=target_session_id)

        remaining = deadline - self._monotonic()
        if remaining <= 0:
            AcpRetryStarter._close_session(result.session)
            raise TimeoutError("ACP resume deadline exhausted")
        load_session = getattr(result.session, "load_session", None)
        if not callable(load_session):
            AcpRetryStarter._close_session(result.session)
            raise RuntimeError("ACP session backend does not support resume")
        try:
            load_session(target_session_id, timeout=remaining)
        except Exception:
            AcpRetryStarter._close_session(result.session)
            raise
        if str(getattr(result.session, "session_id", "") or "").strip() != target_session_id:
            AcpRetryStarter._close_session(result.session)
            raise RuntimeError("ACP session resume returned a different session_id")
        return replace(result, actual_id=target_session_id)

    def _start_backend_with_retry(
        self,
        *,
        request: SessionStartupRequest,
        backend: StartupBackend,
        cwd: str,
        retry_plan: RetryPlan,
        deadline: float,
    ) -> SessionStartupResult:
        from .diagnostics import format_startup_failure_log_line
        from .sync_adapter import build_startup_diagnostics

        agent_type = request.effective_agent_type
        model_name = request.model_name
        last_err: Exception | None = None
        last_spec = ""
        acp_cls = self._sync_acp_session_cls or _default_sync_acp_session_cls()
        cli_cls = self._sync_claude_cli_session_cls or _default_sync_claude_cli_session_cls()

        for attempt in range(1, int(retry_plan.attempts) + 1):
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                last_err = TimeoutError("ACP startup deadline exhausted")
                break
            session = AcpRetryStarter._build_session(
                backend=backend,
                agent_type=agent_type,
                cwd=cwd or ".",
                model_name=model_name,
                acp_session_cls=acp_cls,
                cli_session_cls=cli_cls,
            )
            try:
                last_spec = AcpRetryStarter._describe_session(session)
                effective_timeout = min(retry_plan.timeout_for_attempt(attempt), remaining)
                actual_id = run_startup_operation(session.start, startup_timeout=effective_timeout)
                if self._monotonic() >= deadline:
                    last_err = TimeoutError("ACP startup deadline exhausted")
                    AcpRetryStarter._close_session(session)
                    break
                logger.info(
                    "[ACP:%s] Session started: key=%s, session=%s, agent=%s",
                    agent_type.upper(),
                    request.key[-16:],
                    actual_id[:8],
                    last_spec or agent_type,
                )
                return SessionStartupResult(
                    session=session,
                    actual_id=actual_id,
                    effective_agent_type=agent_type,
                    model_name=model_name,
                )
            except STARTUP_OPERATIONAL_EXCEPTIONS as exc:
                last_err = exc
                diag = build_startup_diagnostics(
                    agent_type=agent_type,
                    cwd=cwd or ".",
                    model_name=model_name,
                    session=session,
                    error=exc,
                    attempt=int(attempt),
                    retries=int(retry_plan.attempts),
                    timeout_s=float(min(retry_plan.timeout_for_attempt(attempt), remaining) or 0),
                )
                if isinstance(diag, dict):
                    error_text = str(diag.get("error_text") or "").strip()
                    if error_text and ((not (str(exc) or "").strip()) or (str(exc) or "").strip() in ("(empty)", "None")):
                        exc.args = (error_text,)
                logger.warning(
                    format_startup_failure_log_line(
                        agent_type=agent_type,
                        event="Session start failed",
                        attempt=int(attempt),
                        retries=int(retry_plan.attempts),
                        error=exc,
                        diag=diag if isinstance(diag, dict) else None,
                        get_settings_fn=self._get_settings,
                    )
                )
                AcpRetryStarter._close_session(session)
                if attempt < int(retry_plan.attempts):
                    retry_delay = min(2.0, 0.3 * attempt, max(0.0, deadline - self._monotonic()))
                    if retry_delay > 0:
                        self._sleep(retry_delay)

        detail = str(last_err) if last_err else "unknown"
        spec = f" ({last_spec})" if last_spec else ""
        try:
            self._session_telemetry.on_session_start_failed(
                manager_agent_type=self._manager_agent_type,
                session_key=request.key,
                backend_kind=backend.value,
                error=last_err or RuntimeError(detail),
                diagnostics=None,
            )
        except (RuntimeError, OSError, TypeError, ValueError):
            logger.debug("[ACP:%s] session telemetry on_session_start_failed error", agent_type.upper(), exc_info=True)
        label = f"{agent_type} 会话" if backend == StartupBackend.CLI else f"{agent_type} ACP Server "
        raise RuntimeError(f"启动 {label}失败{spec}（已重试 {int(retry_plan.attempts)} 次）: {detail}".replace("  失败", " 失败"))

    def start(self, request: SessionStartupRequest) -> SessionStartupResult:
        deadline = request.deadline_monotonic
        if deadline is None:
            deadline = self._monotonic() + float(request.startup_timeout or 60)
            request = replace(request, deadline_monotonic=deadline)
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise TimeoutError("ACP startup deadline exhausted")

        agent_type = request.effective_agent_type
        backend = select_startup_backend(agent_type, model_name=request.model_name)
        retry_plan = build_retry_plan(
            agent_type,
            retries=int(request.retries or 1),
            startup_timeout=float(request.startup_timeout or 60),
            model_name=request.model_name,
        )

        injected = self._try_injected_starter(request=request, retries=retry_plan.attempts, remaining=remaining)
        if injected is not None:
            session, actual_id, _diag = injected
            if session and actual_id:
                if self._monotonic() >= deadline:
                    AcpRetryStarter._close_session(session)
                    raise TimeoutError("ACP startup deadline exhausted")
                AcpRetryStarter._describe_session(session)
                logger.info(
                    "[ACP:%s] Session started via injected starter: key=%s, session=%s",
                    agent_type.upper(),
                    request.key[-16:],
                    actual_id[:8],
                )
                return self._resume_requested_session(
                    SessionStartupResult(
                        session=session,
                        actual_id=actual_id,
                        effective_agent_type=agent_type,
                        model_name=request.model_name,
                    ),
                    request=request,
                    deadline=deadline,
                )

        if self._monotonic() >= deadline:
            raise TimeoutError("ACP startup deadline exhausted")
        from ..utils.path import normalize_session_cwd

        cwd = normalize_startup_cwd(request.cwd, normalize_fn=normalize_session_cwd)
        if self._monotonic() >= deadline:
            raise TimeoutError("ACP startup deadline exhausted")
        result = self._start_backend_with_retry(
            request=request,
            backend=backend,
            cwd=cwd,
            retry_plan=retry_plan,
            deadline=deadline,
        )
        return self._resume_requested_session(result, request=request, deadline=deadline)


def safe_float_or_none(value: object) -> Optional[float]:
    """Best-effort float conversion that never raises."""
    if value is None:
        return None
    try:
        if isinstance(value, (bool, int, float)):
            return float(value)
        s = str(value).strip()
        return float(s) if s else None
    except Exception:
        return None


def initial_startup_diagnostics(
    *,
    agent_type: str,
    cwd: str,
    model_name: Optional[str],
    error: Exception,
    attempt: Optional[int],
    retries: Optional[int],
    timeout_s: Optional[float],
) -> dict:
    """Create the stable startup diagnostics container before enrichment."""
    err_name = type(error).__name__ if error is not None else ""
    return {
        "agent_type": (agent_type or ""),
        "cwd": (cwd or ""),
        "model": (model_name or ""),
        "attempt": int(attempt) if isinstance(attempt, int) else attempt,
        "retries": int(retries) if isinstance(retries, int) else retries,
        "timeout_s": safe_float_or_none(timeout_s),
        "error_type": err_name,
        "exception_type": err_name,
        "error_text": "",
        "error": "",
        "error_repr": "",
        "cmd": "",
        "args": [],
        "rc": None,
        "stdout_snippet": "",
        "stderr_snippet": "",
        "fail_reason": "",
        "spec": "",
        "agent_spec": "",
    }
