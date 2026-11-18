"""IdleHealth telemetry and IdleHealthConfig helpers for ACPSessionManager."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, TypedDict

from ..utils.time_ago import IdleHealth, compute_time_ago_bucket

if TYPE_CHECKING:
    from ..utils.time_ago import TimeAgoBucket

__all__ = [
    "TelemetryAdapter",
    "DefaultSessionTelemetryAdapter",
    "IdleHealthConfig",
    "build_idle_health_config_for_manager",
    "resolve_idle_health_collaborators_for_manager",
    "classify_manager_idle_health",
    "IdleHealthTelemetryContext",
]

logger = logging.getLogger(__name__)


class IdleHealthContext(TypedDict, total=False):
    manager_agent_type: str
    session_key: str
    chat_id: str
    project_id: Optional[str]
    thread_id: Optional[str]
    session_id: str
    idle_seconds: float
    idle_bucket: TimeAgoBucket


IdleHealthTelemetryContext = IdleHealthContext


class TelemetryAdapter(Protocol):
    """Session lifecycle telemetry adapter protocol."""

    def on_session_start(
        self,
        *,
        manager_agent_type: str,
        session_key: str,
        session_id: str,
        backend_kind: str,
        model_name: str | None,
    ) -> None: ...

    def on_session_start_failed(
        self,
        *,
        manager_agent_type: str,
        session_key: str,
        backend_kind: str,
        error: Exception,
        diagnostics: dict | None = None,
    ) -> None: ...

    def on_session_end(
        self,
        *,
        manager_agent_type: str,
        session_key: str,
        session_id: str,
        message_count: int,
        reason: str | None = None,
        extra: dict | None = None,
    ) -> None: ...


class DefaultSessionTelemetryAdapter:
    """Default no-op TelemetryAdapter implementation."""

    def on_session_start(
        self,
        *,
        manager_agent_type: str,
        session_key: str,
        session_id: str,
        backend_kind: str,
        model_name: str | None,
    ) -> None:
        return None

    def on_session_start_failed(
        self,
        *,
        manager_agent_type: str,
        session_key: str,
        backend_kind: str,
        error: Exception,
        diagnostics: dict | None = None,
    ) -> None:
        return None

    def on_session_end(
        self,
        *,
        manager_agent_type: str,
        session_key: str,
        session_id: str,
        message_count: int,
        reason: str | None = None,
        extra: dict | None = None,
    ) -> None:
        return None


class _DefaultIdleHealthTelemetry:
    """Default IdleHealth fallback logger."""

    def record_idle_health_fallback_metric(self, *, error_type: str) -> None:
        return None

    def log_idle_health_classification_fallback(
        self,
        *,
        bucket: "TimeAgoBucket",
        error: Exception,
        context: IdleHealthContext | None = None,
    ) -> None:
        try:
            error_type = type(error).__name__
            parts: list[str] = []
            if context:
                for key, label in (
                    ("manager_agent_type", "agent_type"),
                    ("session_key", "session_key"),
                    ("chat_id", "chat_id"),
                    ("thread_id", "thread_id"),
                    ("session_id", "session_id"),
                ):
                    val = (context.get(key) or "").strip()  # type: ignore[arg-type]
                    if val:
                        parts.append(f"{label}={val}")
                project_id = context.get("project_id")
                if project_id is not None:
                    parts.append(f"project_id={str(project_id or '').strip() or '_default_'}")
            ctx_suffix = (" " + " ".join(parts)) if parts else ""
            logger.warning(
                "[ACP] IdleHealth classification fallback to UNKNOWN due to input error: bucket=%r, error_type=%s, error=%s%s",
                bucket,
                error_type,
                error,
                ctx_suffix,
            )
        except Exception:
            return


def _classify_idle_health_with_fallback(
    bucket: "TimeAgoBucket",
    context: IdleHealthContext | None = None,
    telemetry: Any | None = None,
) -> IdleHealth:
    eff_telemetry = telemetry or _DefaultIdleHealthTelemetry()
    try:
        import src.utils.time_ago as time_ago_mod

        return time_ago_mod.classify_idle_health_from_bucket(bucket)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        try:
            eff_telemetry.log_idle_health_classification_fallback(
                bucket=bucket,
                error=exc,
                context=context,
            )
            eff_telemetry.record_idle_health_fallback_metric(error_type=type(exc).__name__)
        except Exception:
            logger.debug("telemetry recording failed", exc_info=True)
        return IdleHealth.UNKNOWN


class _IdleHealthService:
    """Compute IdleHealth and routing context for manager session snapshots."""

    def __init__(self, telemetry: Any | None = None) -> None:
        self._telemetry = telemetry or _DefaultIdleHealthTelemetry()

    def classify_idle_health(
        self,
        bucket: "TimeAgoBucket",
        *,
        context: IdleHealthTelemetryContext | None = None,
    ) -> IdleHealth:
        return _classify_idle_health_with_fallback(bucket, context=context, telemetry=self._telemetry)

    def classify_session_idle_health(
        self,
        *,
        manager_agent_type: str,
        session_key: str,
        session_id: str,
        last_active: float,
        now: Optional[float] = None,
        message_count: Optional[int] = None,
    ) -> tuple[IdleHealth, "TimeAgoBucket", float, IdleHealthTelemetryContext]:
        eff_now = float(now if now is not None else time.time())
        try:
            last_active_f = float(last_active or 0.0)
        except Exception:
            last_active_f = 0.0

        idle_seconds = max(0.0, eff_now - last_active_f) if last_active_f > 0 else 0.0
        idle_bucket = compute_time_ago_bucket(idle_seconds)

        try:
            from .session_keys import SessionKeyCodec

            chat_id, project_id, thread_id = SessionKeyCodec.decode(session_key or "")
        except Exception:
            chat_id, project_id, thread_id = "", None, None

        ctx: IdleHealthTelemetryContext = {
            "manager_agent_type": manager_agent_type,
            "session_key": session_key,
            "session_id": session_id,
            "idle_seconds": idle_seconds,
            "idle_bucket": idle_bucket,
        }
        if chat_id:
            ctx["chat_id"] = chat_id
        if project_id is not None:
            ctx["project_id"] = project_id
        if thread_id:
            ctx["thread_id"] = thread_id

        health = self.classify_idle_health(idle_bucket, context=ctx)
        return health, idle_bucket, idle_seconds, ctx


@dataclass
class IdleHealthConfig:
    """High-level telemetry/idle-health configuration container for ACPSessionManager."""

    idle_health_telemetry: Any | None = None
    session_telemetry: TelemetryAdapter | None = None
    idle_health_service: Any | None = None


def resolve_idle_health_collaborators_for_manager(
    *,
    config: IdleHealthConfig | None = None,
    idle_health_telemetry: Any | None = None,
    session_telemetry: TelemetryAdapter | None = None,
    idle_health_service: Any | None = None,
) -> tuple[Any, TelemetryAdapter, Any]:
    eff_idle_telemetry = (
        idle_health_telemetry
        or (config.idle_health_telemetry if config is not None else None)
        or _DefaultIdleHealthTelemetry()
    )
    eff_service = (
        idle_health_service
        or (config.idle_health_service if config is not None else None)
        or _IdleHealthService(telemetry=eff_idle_telemetry)
    )
    eff_session_telemetry = (
        session_telemetry
        or (config.session_telemetry if config is not None else None)
        or DefaultSessionTelemetryAdapter()
    )
    return eff_idle_telemetry, eff_session_telemetry, eff_service


def classify_manager_idle_health(
    bucket: "TimeAgoBucket",
    context: IdleHealthContext | None = None,
    telemetry: Any | None = None,
) -> IdleHealth:
    return _classify_idle_health_with_fallback(bucket, context=context, telemetry=telemetry)


def build_idle_health_config_for_manager(
    *,
    idle_health_telemetry: Any | None = None,
    session_telemetry: TelemetryAdapter | None = None,
    idle_health_service: Any | None = None,
) -> IdleHealthConfig:
    eff_telemetry = idle_health_telemetry or _DefaultIdleHealthTelemetry()
    eff_service = idle_health_service or _IdleHealthService(telemetry=eff_telemetry)
    return IdleHealthConfig(
        idle_health_telemetry=eff_telemetry,
        session_telemetry=session_telemetry,
        idle_health_service=eff_service,
    )
