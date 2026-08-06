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
    TTADK_CLI = "ttadk_cli"


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
class TtadkCliStartResult:
    session: Any
    actual_id: str
    resolved_model: Optional[str]


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
    if (agent_type or "").startswith("ttadk_"):
        return StartupBackend.TTADK_CLI
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


class CwdNormalizer:
    """Normalize startup cwd without leaking path utility details to the coordinator."""

    def normalize(self, cwd: str, *, normalize_fn: Callable[[str], str | None]) -> str:
        return normalize_startup_cwd(cwd, normalize_fn=normalize_fn)


class TtadkCliStarter:
    """Start TTADK CLI sessions without leaking class resolution into manager."""

    phase = "ttadk_cli"

    def start(
        self,
        *,
        agent_type: str,
        cwd: str,
        model_name: Optional[str],
        manager_factory: Callable[[], Any],
        precheck_fn: Callable[..., dict[str, Any]],
        manager_session_cls: Any,
        agent_session_cls: Any,
        original_session_cls: Any,
        start_operation: Callable[..., Any],
        deadline_monotonic: float,
        monotonic_fn: Callable[[], float],
    ) -> TtadkCliStartResult:
        session_cls = self._resolve_session_class(
            manager_session_cls=manager_session_cls,
            agent_session_cls=agent_session_cls,
            original_session_cls=original_session_cls,
        )
        precheck = precheck_ttadk_cli_startup(
            agent_type=agent_type,
            cwd=cwd or ".",
            model_name=model_name,
            manager_factory=manager_factory,
            precheck_fn=precheck_fn,
        )
        resolved_model = precheck.get("model")
        remaining = deadline_monotonic - monotonic_fn()
        if remaining <= 0:
            raise TimeoutError("ACP startup deadline exhausted during TTADK precheck")
        session = session_cls(agent_type=agent_type, cwd=cwd or ".", model_name=resolved_model)
        actual_id = start_operation(session.start, startup_timeout=remaining)
        if monotonic_fn() >= deadline_monotonic:
            AcpRetryStarter._close_session(session)
            raise TimeoutError("ACP startup deadline exhausted during TTADK startup")
        return TtadkCliStartResult(session=session, actual_id=actual_id, resolved_model=resolved_model)

    @staticmethod
    def _resolve_session_class(*, manager_session_cls: Any, agent_session_cls: Any, original_session_cls: Any) -> Any:
        effective_cls = manager_session_cls
        if original_session_cls is not None:
            if manager_session_cls is not None and manager_session_cls is not original_session_cls:
                return manager_session_cls
            if agent_session_cls is not None and agent_session_cls is not original_session_cls:
                return agent_session_cls
        elif agent_session_cls is not None:
            effective_cls = agent_session_cls
        return effective_cls


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


def _default_agent_session_module() -> Any:
    from .. import agent_session as agent_session_mod

    return agent_session_mod


def _default_sync_acp_session_cls() -> Any:
    from .sync_adapter import SyncACPSession

    return SyncACPSession


def _default_sync_claude_cli_session_cls() -> Any:
    from ..agent_session import SyncClaudeCLISession

    return SyncClaudeCLISession


def _default_sync_ttadk_cli_session_cls() -> Any:
    from ..agent_session import SyncTTADKCLISession

    return SyncTTADKCLISession


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


class TtadkCliStartupBackendStrategy:
    """Start TTADK-wrapped tools through CLI bridge only."""

    backend = StartupBackend.TTADK_CLI

    def __init__(
        self,
        *,
        ttadk_cli_starter: TtadkCliStarter,
        failure_reporter: StartupFailureReporter,
        sync_ttadk_cli_session_cls: Any | None,
        agent_session_module: Any | None,
        original_ttadk_cli_session_cls: Any | None,
        get_settings_fn: Callable[[], Any],
        monotonic_fn: Callable[[], float],
    ) -> None:
        self._ttadk_cli_starter = ttadk_cli_starter
        self._failure_reporter = failure_reporter
        self._sync_ttadk_cli_session_cls = sync_ttadk_cli_session_cls
        self._agent_session_module = agent_session_module
        self._original_ttadk_cli_session_cls = original_ttadk_cli_session_cls
        self._get_settings = get_settings_fn
        self._monotonic = monotonic_fn

    def start(
        self,
        *,
        request: SessionStartupRequest,
        cwd: str,
        model_name: Optional[str],
        retry_plan: RetryPlan,
    ) -> SessionStartupResult:
        from ..ttadk import get_ttadk_manager
        from ..ttadk.startup_common import precheck_ttadk_startup_model

        agent_session_mod = self._agent_session_module or _default_agent_session_module()
        try:
            agent_cls = getattr(agent_session_mod, "SyncTTADKCLISession", None)
        except (AttributeError, TypeError):
            logger.warning("Error while getting TTADK CLI session class", exc_info=True)
            agent_cls = None

        try:
            started = self._ttadk_cli_starter.start(
                agent_type=request.effective_agent_type,
                cwd=cwd or ".",
                model_name=model_name,
                manager_factory=get_ttadk_manager,
                precheck_fn=precheck_ttadk_startup_model,
                manager_session_cls=self._sync_ttadk_cli_session_cls or _default_sync_ttadk_cli_session_cls(),
                agent_session_cls=agent_cls,
                original_session_cls=self._original_ttadk_cli_session_cls,
                start_operation=run_startup_operation,
                deadline_monotonic=float(request.deadline_monotonic or 0),
                monotonic_fn=self._monotonic,
            )
            logger.info(
                "[ACP:%s] TTADK CLI Session started: key=%s, session=%s, model=%s",
                request.effective_agent_type.upper(),
                request.key[-16:],
                started.actual_id[:8],
                started.resolved_model,
            )
            return SessionStartupResult(
                session=started.session,
                actual_id=started.actual_id,
                effective_agent_type=request.effective_agent_type,
                model_name=model_name,
            )
        except STARTUP_OPERATIONAL_EXCEPTIONS as exc:
            classify_startup_error(exc, phase="ttadk_cli", attempt=1, retries=1)
            safe_detail = self._safe_ttadk_error_detail(exc)
            logger.warning("Error while starting TTADK CLI: %s", safe_detail)
            try:
                self._failure_reporter.record(
                    session_key=request.key,
                    backend_kind="cli",
                    error=exc or RuntimeError(safe_detail),
                    diagnostics=None,
                )
            except (RuntimeError, OSError, TypeError, ValueError):
                logger.debug(
                    "[ACP:%s] session telemetry on_session_start_failed error",
                    request.effective_agent_type.upper(),
                    exc_info=True,
                )
            raise RuntimeError(f"启动 {request.effective_agent_type} CLI 失败: {safe_detail}")

    def _safe_ttadk_error_detail(self, exc: Exception) -> str:
        detail = str(exc or "").strip()
        if not detail and exc is not None:
            for key in ("stderr_snippet", "stdout_snippet", "stderr", "stdout"):
                try:
                    value = str(getattr(exc, key, "") or "").strip()
                    if value:
                        detail = value
                        break
                except (AttributeError, TypeError):
                    logger.warning("Error while extracting error detail from attribute %s", key, exc_info=True)
                    continue
        if not detail:
            _, err_repr = _format_error_type_and_repr(exc)
            detail = err_repr or "unknown"
        return _sanitize_startup_detail(detail, get_settings_fn=self._get_settings) or "start_failed"


def _format_error_type_and_repr(err: object) -> tuple[str, str]:
    try:
        err_type = type(err).__name__
    except Exception:
        logger.warning("Error while formatting error type", exc_info=True)
        err_type = "Exception"
    try:
        err_repr = repr(err)
    except Exception:
        logger.warning("Error while formatting error representation", exc_info=True)
        err_repr = ""
    if not (err_repr or "").strip():
        err_repr = f"<{err_type}>"
    return (err_type or "Exception", err_repr)


def _sanitize_startup_detail(text: str, *, get_settings_fn: Callable[[], Any]) -> str:
    """Redact and truncate startup detail for safe logging/user-facing errors."""

    s = str(text or "")
    if not s:
        return ""
    try:
        from .diagnostics import get_diagnostics_config, redact_text, truncate_text

        cfg = get_diagnostics_config(get_settings_fn=get_settings_fn)
        if bool(getattr(cfg, "redact_enabled", True)):
            s = redact_text(
                s,
                list(getattr(cfg, "redact_patterns", []) or []),
                str(getattr(cfg, "redact_replacement", "***REDACTED***") or "***REDACTED***"),
            )
        lim = int(getattr(cfg, "snippet_limit", 240) or 240)
        s = truncate_text(s, max(1, lim))
    except (AttributeError, TypeError, ValueError, ImportError):
        logger.warning("Error while sanitizing startup detail", exc_info=True)
    return s


def _coco_acp_args(model_name: Optional[str]) -> list[str]:
    args: list[str] = ["acp", "serve"]
    if model_name:
        args.extend(["-c", f"model.name={model_name}"])
    return args


def resolve_ttadk_target_model_for_existing_session(
    *,
    agent_type: str,
    cwd: str,
    model_name: Optional[str],
    manager_factory: Callable[[], Any],
    precheck_fn: Callable[..., dict[str, Any]],
) -> Optional[str]:
    """Resolve a validated TTADK real model name for existing-session reuse checks."""

    precheck = precheck_ttadk_cli_startup(
        agent_type=agent_type,
        cwd=cwd or ".",
        model_name=model_name,
        manager_factory=manager_factory,
        precheck_fn=precheck_fn,
    )
    if bool(precheck.get("validated")):
        return str(precheck.get("model") or "").strip() or None
    return None


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
        sync_ttadk_cli_session_cls: Any | None = None,
        agent_session_module: Any | None = None,
        original_ttadk_cli_session_cls: Any | None = None,
        get_settings_fn: Callable[[], Any] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        monotonic_fn: Callable[[], float] | None = None,
    ) -> None:
        self._manager_agent_type = manager_agent_type
        self._session_starter = session_starter
        self._session_telemetry = session_telemetry
        self._injected_starter = InjectedStarterFallback(session_starter)
        self._cwd_normalizer = CwdNormalizer()
        self._ttadk_cli_starter = TtadkCliStarter()
        self._acp_retry_starter = AcpRetryStarter()
        self._failure_reporter = StartupFailureReporter(
            telemetry=session_telemetry,
            manager_agent_type=manager_agent_type,
        )
        self._sync_acp_session_cls = sync_acp_session_cls
        self._sync_claude_cli_session_cls = sync_claude_cli_session_cls
        self._sync_ttadk_cli_session_cls = sync_ttadk_cli_session_cls
        self._agent_session_module = agent_session_module
        self._original_ttadk_cli_session_cls = original_ttadk_cli_session_cls
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
            StartupBackend.TTADK_CLI: TtadkCliStartupBackendStrategy(
                ttadk_cli_starter=self._ttadk_cli_starter,
                failure_reporter=self._failure_reporter,
                sync_ttadk_cli_session_cls=self._sync_ttadk_cli_session_cls,
                agent_session_module=self._agent_session_module,
                original_ttadk_cli_session_cls=self._original_ttadk_cli_session_cls,
                get_settings_fn=self._get_settings,
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

    def _normalize_cwd(self, cwd: str) -> str:
        try:
            from ..utils.path import normalize_ttadk_cwd

            raw_cwd = cwd
            normalized = self._cwd_normalizer.normalize(raw_cwd, normalize_fn=normalize_ttadk_cwd)
            try:
                if bool(getattr(self._get_settings(), "ttadk_cwd_debug_enabled", False)):
                    logger.debug(
                        "[TTADK:CWD] where=%s raw_cwd=%r normalized_cwd=%r",
                        "acp.startup.ensure_session",
                        raw_cwd,
                        normalized,
                    )
            except (AttributeError, TypeError):
                logger.warning("Error while checking TTADK CWD debug flag", exc_info=True)
            return normalized
        except (ImportError, AttributeError, TypeError):
            logger.warning("Error while normalizing TTADK CWD", exc_info=True)
            return cwd

    def resolve_ttadk_target_model_for_existing_session(
        self,
        *,
        agent_type: str,
        cwd: str,
        model_name: Optional[str],
    ) -> Optional[str]:
        from ..ttadk import get_ttadk_manager
        from ..ttadk.startup_common import precheck_ttadk_startup_model

        return resolve_ttadk_target_model_for_existing_session(
            agent_type=agent_type,
            cwd=cwd or ".",
            model_name=model_name,
            manager_factory=get_ttadk_manager,
            precheck_fn=precheck_ttadk_startup_model,
        )

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
                return SessionStartupResult(
                    session=session,
                    actual_id=actual_id,
                    effective_agent_type=effective_agent_type,
                    model_name=model_name,
                )

        if self._monotonic() >= deadline:
            raise TimeoutError("ACP startup deadline exhausted")
        cwd = self._normalize_cwd(cwd)
        if self._monotonic() >= deadline:
            raise TimeoutError("ACP startup deadline exhausted")
        return self._backend_strategies[backend].start(
            request=request,
            cwd=cwd,
            model_name=model_name,
            retry_plan=retry_plan,
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


def precheck_ttadk_cli_startup(
    *,
    agent_type: str,
    cwd: str,
    model_name: Optional[str],
    manager_factory: Callable[[], Any],
    precheck_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    manager = manager_factory()
    return precheck_fn(
        agent_type=agent_type,
        cwd=cwd or ".",
        model_intent=model_name,
        manager=manager,
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
