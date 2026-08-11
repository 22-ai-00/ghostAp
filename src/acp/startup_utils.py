"""Shared ACP startup orchestration and diagnostics helpers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class StartupOperationalError(RuntimeError):
    """Expected startup/runtime boundary failure that may retry or fallback."""


class StartupFatalError(RuntimeError):
    """Unexpected startup failure that must not be treated as operational."""


# Startup crosses process, stdio and adapter boundaries.  Only explicitly
# expected operational failures participate in retry/degrade semantics; ordinary
# programming errors (TypeError/AssertionError) and unknown business exceptions
# must fail fast so they are not disguised as recoverable startup failures.
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


class StartupErrorKind(str, Enum):
    """User-visible startup failure category."""

    RECOVERABLE = "recoverable"
    DEGRADED = "degraded"
    FATAL = "fatal"


class StartupErrorAction(str, Enum):
    """Next action associated with a classified startup failure."""

    FALLBACK = "fallback"
    RETRY = "retry"
    FAIL_WITH_DIAGNOSTICS = "fail_with_diagnostics"
    REQUIRES_USER_ACTION = "requires_user_action"
    RAISE = "raise"


@dataclass(frozen=True)
class RetryPlan:
    """ACP/CLI startup retry plan with progressive timeout semantics."""

    attempts: int
    startup_timeout: float

    def timeout_for_attempt(self, attempt: int) -> float:
        safe_attempt = max(1, int(attempt or 1))
        return float(self.startup_timeout or 60) * (1.0 + 0.5 * (safe_attempt - 1))


@dataclass(frozen=True)
class StartupErrorClassification:
    kind: StartupErrorKind
    action: StartupErrorAction
    phase: str
    user_reachable: bool = True


@dataclass(frozen=True)
class AcpRetryStartResult:
    session: Any | None
    actual_id: str
    last_spec: str
    last_err: Exception | None
    effective_timeout: float


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


def select_startup_backend(agent_type: str) -> StartupBackend:
    if agent_type == "claude":
        return StartupBackend.CLI
    return StartupBackend.ACP


def should_try_injected_starter(agent_type: str, *, starter: object) -> bool:
    return callable(starter) and select_startup_backend(agent_type) == StartupBackend.ACP


class InjectedStarterFallback:
    """Run the optional injected ACP starter and classify fallback failures."""

    def __init__(self, starter: object) -> None:
        self._starter = starter

    def should_try(self, agent_type: str) -> bool:
        return should_try_injected_starter(agent_type, starter=self._starter)

    def start(self, *, agent_type: str, retries: int, **kwargs: Any) -> tuple[Any, str, dict[str, Any] | None] | None:
        if not self.should_try(agent_type):
            return None
        try:
            session, actual_id, diagnostics = self._starter(agent_type=agent_type, **kwargs)  # type: ignore[misc]
        except STARTUP_OPERATIONAL_EXCEPTIONS as exc:
            classification = classify_startup_error(
                exc,
                phase="injected_starter",
                attempt=1,
                retries=retries,
            )
            if classification.action != StartupErrorAction.FALLBACK:
                raise
            return None
        return session, actual_id, diagnostics


class AcpRetryStarter:
    """Run ACP/CLI startup retries and diagnostics outside the coordinator."""

    phase = "acp_retry"

    def start(
        self,
        *,
        backend: StartupBackend,
        agent_type: str,
        cwd: str,
        model_name: Optional[str],
        retry_plan: RetryPlan,
        request_key: str,
        acp_session_cls: Any,
        cli_session_cls: Any,
        start_operation: Callable[..., Any],
        diagnostics_fn: Callable[..., dict[str, Any]],
        format_log_line_fn: Callable[..., str],
        get_settings_fn: Callable[[], Any],
        sleep_fn: Callable[[float], None],
        deadline_monotonic: float | None = None,
        monotonic_fn: Callable[[], float] | None = None,
    ) -> AcpRetryStartResult:
        last_err: Exception | None = None
        last_spec = ""
        effective_timeout = float(retry_plan.startup_timeout or 60)
        clock = monotonic_fn or time.monotonic

        for attempt in range(1, int(retry_plan.attempts) + 1):
            if deadline_monotonic is not None:
                remaining = deadline_monotonic - clock()
                if remaining <= 0:
                    last_err = TimeoutError("ACP startup deadline exhausted")
                    break
            else:
                remaining = float("inf")
            session = self._build_session(
                backend=backend,
                agent_type=agent_type,
                cwd=cwd,
                model_name=model_name,
                acp_session_cls=acp_session_cls,
                cli_session_cls=cli_session_cls,
            )
            try:
                last_spec = self._describe_session(session)
                effective_timeout = min(
                    retry_plan.timeout_for_attempt(attempt),
                    remaining,
                )
                actual_id = start_operation(session.start, startup_timeout=effective_timeout)
                if (
                    deadline_monotonic is not None
                    and clock() >= deadline_monotonic
                ):
                    last_err = TimeoutError("ACP startup deadline exhausted")
                    self._close_session(session)
                    break
                return AcpRetryStartResult(
                    session=session,
                    actual_id=actual_id,
                    last_spec=last_spec,
                    last_err=None,
                    effective_timeout=effective_timeout,
                )
            except STARTUP_OPERATIONAL_EXCEPTIONS as exc:
                last_err = exc
                classification = classify_startup_error(
                    exc,
                    phase=self.phase,
                    attempt=int(attempt),
                    retries=int(retry_plan.attempts),
                )
                diag = diagnostics_fn(
                    agent_type=agent_type,
                    cwd=cwd or ".",
                    model_name=model_name,
                    session=session,
                    error=exc,
                    attempt=int(attempt),
                    retries=int(retry_plan.attempts),
                    timeout_s=float(effective_timeout or 0),
                )
                if isinstance(diag, dict):
                    error_text = str(diag.get("error_text") or "").strip()
                    if error_text and ((not (str(exc) or "").strip()) or (str(exc) or "").strip() in ("(empty)", "None")):
                        exc.args = (error_text,)
                logger.warning(
                    format_log_line_fn(
                        agent_type=agent_type,
                        event="Session start failed",
                        attempt=int(attempt),
                        retries=int(retry_plan.attempts),
                        error=exc,
                        diag=diag if isinstance(diag, dict) else None,
                        attempts=(diag.get("attempts") if isinstance(diag, dict) else None),
                        get_settings_fn=get_settings_fn,
                    )
                )
                self._close_session(session)
                if classification.action == StartupErrorAction.RETRY and attempt < int(retry_plan.attempts):
                    retry_delay = min(2.0, 0.3 * attempt)
                    if deadline_monotonic is not None:
                        retry_delay = min(
                            retry_delay,
                            max(0.0, deadline_monotonic - clock()),
                        )
                    if retry_delay > 0:
                        sleep_fn(retry_delay)
                continue

        return AcpRetryStartResult(
            session=None,
            actual_id="",
            last_spec=last_spec,
            last_err=last_err,
            effective_timeout=effective_timeout,
        )

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
            return cli_session_cls(cwd=cwd or ".", model_name=model_name)
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


class StartupFailureReporter:
    """Report startup failures to telemetry through a narrow collaborator."""

    def __init__(self, *, telemetry: Any, manager_agent_type: str) -> None:
        self._telemetry = telemetry
        self._manager_agent_type = manager_agent_type

    def record(
        self,
        *,
        session_key: str,
        backend_kind: str,
        error: Exception,
        diagnostics: Optional[dict[str, Any]],
    ) -> None:
        record_startup_failure(
            telemetry=self._telemetry,
            manager_agent_type=self._manager_agent_type,
            session_key=session_key,
            backend_kind=backend_kind,
            error=error,
            diagnostics=diagnostics,
        )


def _default_sync_acp_session_cls() -> Any:
    from .sync_adapter import SyncACPSession

    return SyncACPSession


def _default_sync_claude_cli_session_cls() -> Any:
    from ..agent_session import SyncClaudeCLISession

    return SyncClaudeCLISession


class StartupBackendStrategy(Protocol):
    """Backend-specific startup strategy selected by the coordinator."""

    backend: StartupBackend

    def start(
        self,
        *,
        request: SessionStartupRequest,
        cwd: str,
        model_name: Optional[str],
        retry_plan: RetryPlan,
    ) -> SessionStartupResult:
        ...


class AcpStartupBackendStrategy:
    """Start ACP-direct sessions through retry/diagnostics collaborator."""

    backend = StartupBackend.ACP


    def __init__(
        self,
        *,
        retry_starter: AcpRetryStarter,
        failure_reporter: StartupFailureReporter,
        sync_acp_session_cls: Any | None,
        sync_claude_cli_session_cls: Any | None,
        get_settings_fn: Callable[[], Any],
        sleep_fn: Callable[[float], None],
        monotonic_fn: Callable[[], float],
    ) -> None:
        self._retry_starter = retry_starter
        self._failure_reporter = failure_reporter
        self._sync_acp_session_cls = sync_acp_session_cls
        self._sync_claude_cli_session_cls = sync_claude_cli_session_cls
        self._get_settings = get_settings_fn
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn

    def start(
        self,
        *,
        request: SessionStartupRequest,
        cwd: str,
        model_name: Optional[str],
        retry_plan: RetryPlan,
    ) -> SessionStartupResult:
        started = self._start_with_retry(
            request=request,
            backend=self.backend,
            cwd=cwd,
            model_name=model_name,
            retry_plan=retry_plan,
        )
        if started.session and started.actual_id:
            return SessionStartupResult(
                session=started.session,
                actual_id=started.actual_id,
                effective_agent_type=request.effective_agent_type,
                model_name=model_name,
            )
        self._raise_startup_failure(
            request=request,
            retry_plan=retry_plan,
            last_spec=started.last_spec,
            last_err=started.last_err,
        )

    def _start_with_retry(
        self,
        *,
        request: SessionStartupRequest,
        backend: StartupBackend,
        cwd: str,
        model_name: Optional[str],
        retry_plan: RetryPlan,
    ) -> AcpRetryStartResult:
        from .diagnostics import format_startup_failure_log_line
        from .sync_adapter import build_startup_diagnostics

        return self._retry_starter.start(
            backend=backend,
            agent_type=request.effective_agent_type,
            cwd=cwd or ".",
            model_name=model_name,
            retry_plan=retry_plan,
            request_key=request.key,
            acp_session_cls=self._sync_acp_session_cls or _default_sync_acp_session_cls(),
            cli_session_cls=self._sync_claude_cli_session_cls or _default_sync_claude_cli_session_cls(),
            start_operation=run_startup_operation,
            diagnostics_fn=lambda **kwargs: build_manager_startup_diagnostics(
                diagnostics_fn=build_startup_diagnostics,
                **kwargs,
            ),
            format_log_line_fn=format_startup_failure_log_line,
            get_settings_fn=self._get_settings,
            sleep_fn=self._sleep,
            deadline_monotonic=request.deadline_monotonic,
            monotonic_fn=self._monotonic,
        )

    def _raise_startup_failure(
        self,
        *,
        request: SessionStartupRequest,
        retry_plan: RetryPlan,
        last_spec: str,
        last_err: Exception | None,
    ) -> None:
        detail = str(last_err) if last_err else "unknown"
        spec = f" ({last_spec})" if last_spec else ""
        try:
            self._failure_reporter.record(
                session_key=request.key,
                backend_kind="acp",
                error=last_err or RuntimeError(detail),
                diagnostics=None,
            )
        except (RuntimeError, OSError, TypeError, ValueError):
            logger.debug("[ACP:%s] session telemetry on_session_start_failed error", request.effective_agent_type.upper(), exc_info=True)
        raise RuntimeError(
            f"启动 {request.effective_agent_type} ACP Server 失败{spec}（已重试 {int(retry_plan.attempts)} 次）: {detail}"
        )


class CliStartupBackendStrategy(AcpStartupBackendStrategy):
    """Start shell CLI bridge sessions without ACP-specific branching in coordinator."""

    backend = StartupBackend.CLI

    def _raise_startup_failure(
        self,
        *,
        request: SessionStartupRequest,
        retry_plan: RetryPlan,
        last_spec: str,
        last_err: Exception | None,
    ) -> None:
        detail = str(last_err) if last_err else "unknown"
        spec = f" ({last_spec})" if last_spec else ""
        try:
            self._failure_reporter.record(
                session_key=request.key,
                backend_kind="cli",
                error=last_err or RuntimeError(detail),
                diagnostics=None,
            )
        except (RuntimeError, OSError, TypeError, ValueError):
            logger.debug("[ACP:%s] session telemetry on_session_start_failed error", request.effective_agent_type.upper(), exc_info=True)
        raise RuntimeError(
            f"启动 {request.effective_agent_type} 会话失败{spec}（已重试 {int(retry_plan.attempts)} 次）: {detail}"
        )


def _coco_acp_args(model_name: Optional[str]) -> list[str]:
    args: list[str] = ["acp", "serve"]
    if model_name:
        args.extend(["-c", f"model.name={model_name}"])
    return args


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
        self._injected_starter = InjectedStarterFallback(session_starter)
        self._acp_retry_starter = AcpRetryStarter()
        self._failure_reporter = StartupFailureReporter(
            telemetry=session_telemetry,
            manager_agent_type=manager_agent_type,
        )
        self._sync_acp_session_cls = sync_acp_session_cls
        self._sync_claude_cli_session_cls = sync_claude_cli_session_cls
        self._get_settings = get_settings_fn or self._default_get_settings
        self._sleep = sleep_fn or self._default_sleep
        self._monotonic = monotonic_fn or time.monotonic
        self._backend_strategies: dict[StartupBackend, StartupBackendStrategy] = {
            StartupBackend.ACP: AcpStartupBackendStrategy(
                retry_starter=self._acp_retry_starter,
                failure_reporter=self._failure_reporter,
                sync_acp_session_cls=self._sync_acp_session_cls,
                sync_claude_cli_session_cls=self._sync_claude_cli_session_cls,
                get_settings_fn=self._get_settings,
                sleep_fn=self._sleep,
                monotonic_fn=self._monotonic,
            ),
            StartupBackend.CLI: CliStartupBackendStrategy(
                retry_starter=self._acp_retry_starter,
                failure_reporter=self._failure_reporter,
                sync_acp_session_cls=self._sync_acp_session_cls,
                sync_claude_cli_session_cls=self._sync_claude_cli_session_cls,
                get_settings_fn=self._get_settings,
                sleep_fn=self._sleep,
                monotonic_fn=self._monotonic,
            ),
        }

    @staticmethod
    def _default_get_settings() -> Any:
        from ..config import get_settings

        return get_settings()

    @staticmethod
    def _default_sleep(seconds: float) -> None:
        import time

        time.sleep(seconds)

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
        current_session_id = str(
            getattr(result.session, "session_id", "") or ""
        ).strip()
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
        loaded_session_id = str(
            getattr(result.session, "session_id", "") or ""
        ).strip()
        if loaded_session_id != target_session_id:
            AcpRetryStarter._close_session(result.session)
            raise RuntimeError("ACP session resume returned a different session_id")
        return replace(result, actual_id=target_session_id)

    def start(self, request: SessionStartupRequest) -> SessionStartupResult:
        deadline = request.deadline_monotonic
        if deadline is None:
            deadline = self._monotonic() + float(request.startup_timeout or 60)
            request = replace(request, deadline_monotonic=deadline)
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise TimeoutError("ACP startup deadline exhausted")
        effective_agent_type = request.effective_agent_type
        backend = select_startup_backend(effective_agent_type)
        retry_plan = build_retry_plan(
            effective_agent_type,
            retries=int(request.retries or 1),
            startup_timeout=float(request.startup_timeout or 60),
        )
        retries = int(retry_plan.attempts)
        cwd = request.cwd
        model_name = request.model_name

        injected_result = self._injected_starter.start(
            agent_type=effective_agent_type,
            retries=retries,
            cwd=cwd or ".",
            startup_timeout=remaining,
            model_name=model_name,
            session_id=request.session_id,
            project_id=request.project_id,
        )
        if injected_result is not None:
            session, actual_id, _diag = injected_result
            if session and actual_id:
                if self._monotonic() >= deadline:
                    AcpRetryStarter._close_session(session)
                    raise TimeoutError("ACP startup deadline exhausted")
                try:
                    session.describe_agent()
                except (AttributeError, TypeError):
                    logger.warning("Error while describing agent", exc_info=True)
                logger.info(
                    "[ACP:%s] Session started via injected starter: key=%s, session=%s",
                    effective_agent_type.upper(),
                    request.key[-16:],
                    actual_id[:8],
                )
                return self._resume_requested_session(
                    SessionStartupResult(
                        session=session,
                        actual_id=actual_id,
                        effective_agent_type=effective_agent_type,
                        model_name=model_name,
                    ),
                    request=request,
                    deadline=deadline,
                )

        if self._monotonic() >= deadline:
            raise TimeoutError("ACP startup deadline exhausted")
        from ..utils.path import normalize_session_cwd

        cwd = normalize_startup_cwd(cwd, normalize_fn=normalize_session_cwd)
        if self._monotonic() >= deadline:
            raise TimeoutError("ACP startup deadline exhausted")
        result = self._backend_strategies[backend].start(
            request=request,
            cwd=cwd,
            model_name=model_name,
            retry_plan=retry_plan,
        )
        return self._resume_requested_session(
            result,
            request=request,
            deadline=deadline,
        )


def normalize_startup_cwd(cwd: str, *, normalize_fn: Callable[[str], str | None]) -> str:
    raw_cwd = cwd or "."
    try:
        normalized = normalize_fn(raw_cwd)
    except (OSError, RuntimeError, TypeError, ValueError):
        logger.warning("Error while normalizing startup cwd", exc_info=True)
        return raw_cwd
    return normalized or raw_cwd


def build_retry_plan(agent_type: str, *, retries: int, startup_timeout: float) -> RetryPlan:
    attempts = 1 if select_startup_backend(agent_type) == StartupBackend.CLI else max(1, int(retries or 1))
    return RetryPlan(attempts=attempts, startup_timeout=float(startup_timeout or 60))


def classify_startup_error(
    error: Exception,
    *,
    phase: str,
    attempt: int,
    retries: int,
) -> StartupErrorClassification:
    if phase == "injected_starter":
        return StartupErrorClassification(
            kind=StartupErrorKind.RECOVERABLE,
            action=StartupErrorAction.FALLBACK,
            phase=phase,
        )
    if phase == "acp_retry":
        if int(attempt or 0) < int(retries or 1):
            return StartupErrorClassification(
                kind=StartupErrorKind.RECOVERABLE,
                action=StartupErrorAction.RETRY,
                phase=phase,
            )
        return StartupErrorClassification(
            kind=StartupErrorKind.DEGRADED,
            action=StartupErrorAction.REQUIRES_USER_ACTION,
            phase=phase,
        )
    return StartupErrorClassification(
        kind=StartupErrorKind.FATAL,
        action=StartupErrorAction.RAISE,
        phase=phase,
    )


def build_manager_startup_diagnostics(
    *,
    diagnostics_fn: Callable[..., dict[str, Any]],
    agent_type: str,
    cwd: str,
    model_name: Optional[str],
    session: Any,
    error: Exception,
    attempt: int,
    retries: int,
    timeout_s: float,
) -> dict[str, Any]:
    return diagnostics_fn(
        agent_type=agent_type,
        cwd=cwd or ".",
        model_name=model_name,
        session=session,
        error=error,
        attempt=int(attempt),
        retries=int(retries),
        timeout_s=float(timeout_s or 0),
    )


def record_startup_failure(
    *,
    telemetry: Any,
    manager_agent_type: str,
    session_key: str,
    backend_kind: str,
    error: Exception,
    diagnostics: Optional[dict[str, Any]],
) -> None:
    telemetry.on_session_start_failed(
        manager_agent_type=manager_agent_type,
        session_key=session_key,
        backend_kind=backend_kind,
        error=error,
        diagnostics=diagnostics,
    )


def run_startup_operation(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run an external startup adapter call with a narrow fail-fast boundary.

    Adapter/test-double code may raise custom ``Exception`` subclasses for
    process/handshake failures.  Programming errors remain fatal and propagate.
    """

    try:
        return fn(*args, **kwargs)
    except (TypeError, AssertionError):
        raise
    except STARTUP_OPERATIONAL_EXCEPTIONS:
        raise
    except Exception as exc:
        raise StartupFatalError(str(exc) or type(exc).__name__) from exc


def safe_float_or_none(value: object) -> Optional[float]:
    """Best-effort float conversion that never raises."""
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        logger.debug("safe_float_or_none: conversion failed for %r", value, exc_info=True)
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
    return {
        "agent_type": (agent_type or ""),
        "cwd": (cwd or ""),
        "model": (model_name or ""),
        "attempt": int(attempt) if isinstance(attempt, int) else attempt,
        "retries": int(retries) if isinstance(retries, int) else retries,
        "timeout_s": safe_float_or_none(timeout_s),
        "error_type": type(error).__name__ if error is not None else "",
        "exception_type": type(error).__name__ if error is not None else "",
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
