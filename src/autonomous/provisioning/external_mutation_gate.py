"""Per-employee fence for hire side effects racing retirement."""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum

EXTERNAL_TASK_NAME_PREFIX = "employee-external-mutation:"


def external_task_name(operation: str) -> str:
    """Return the marker used to keep real external work alive at shutdown."""

    if not isinstance(operation, str) or not operation:
        raise ValueError("external task operation is required")
    return f"{EXTERNAL_TASK_NAME_PREFIX}{operation}"


def is_external_task(task: object) -> bool:
    """Identify a Task whose completion is part of an active mutation lease."""

    get_name = getattr(task, "get_name", None)
    return callable(get_name) and str(get_name()).startswith(
        EXTERNAL_TASK_NAME_PREFIX
    )


class ExternalMutationFenced(RuntimeError):
    """A hire-side mutation was rejected after retirement won admission."""


class ExternalTaskTerminalError(RuntimeError):
    """An external child task terminated without returning a value."""

    def __init__(
        self,
        *,
        caller_cancelled: bool,
        child_cancelled: bool,
    ) -> None:
        super().__init__(
            "external child task was cancelled"
            if child_cancelled
            else "external child task failed"
        )
        self.caller_cancelled = caller_cancelled
        self.child_cancelled = child_cancelled


class ExternalMutationKind(str, Enum):
    APP_REGISTRATION = "app_registration"
    CREDENTIAL_PUT = "credential_put"
    SLASH_RECONCILIATION = "slash_reconciliation"
    CHANNEL_START = "channel_start"
    MEMBERSHIP_ADD = "membership_add"


@dataclass(slots=True)
class EmployeeExternalMutationLease:
    """One process-local capability to finish an already-admitted mutation."""

    _gate: EmployeeExternalMutationGate
    tenant_key: str
    agent_id: str
    kind: ExternalMutationKind
    token: str
    _released: bool = False

    def require_active(self) -> None:
        self._gate._require_active(self)

    def release(self) -> None:
        if self._released:
            return
        self._gate._release(self)
        self._released = True


class EmployeeExternalMutationGate:
    """Linearize hire-side external mutations before a permanent fire fence.

    The condition lock protects only small in-memory bookkeeping. Callers must
    never hold it across Journal I/O or an external operation.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._fenced: set[tuple[str, str]] = set()
        self._active: dict[
            tuple[str, str],
            dict[str, ExternalMutationKind],
        ] = {}

    def acquire(
        self,
        tenant_key: str,
        agent_id: str,
        kind: ExternalMutationKind,
    ) -> EmployeeExternalMutationLease:
        identity = self._identity(tenant_key, agent_id)
        try:
            normalized_kind = ExternalMutationKind(kind)
        except (TypeError, ValueError):
            raise ValueError("invalid external mutation kind") from None
        with self._condition:
            if identity in self._fenced:
                raise ExternalMutationFenced("employee is retiring")
            token = uuid.uuid4().hex
            active = self._active.setdefault(identity, {})
            while token in active:
                token = uuid.uuid4().hex
            active[token] = normalized_kind
            return EmployeeExternalMutationLease(
                self,
                tenant_key,
                agent_id,
                normalized_kind,
                token,
            )

    def begin_retirement(
        self,
        tenant_key: str,
        agent_id: str,
        *,
        timeout_seconds: float,
    ) -> bool:
        """Permanently fence an employee and wait boundedly for old leases."""

        identity = self._identity(tenant_key, agent_id)
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout_seconds must be finite and non-negative")
        deadline = time.monotonic() + timeout
        with self._condition:
            self._fenced.add(identity)
            while self._active.get(identity):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def restore_retirement_fence(self, tenant_key: str, agent_id: str) -> None:
        identity = self._identity(tenant_key, agent_id)
        with self._condition:
            self._fenced.add(identity)
            self._condition.notify_all()

    def is_fenced(self, tenant_key: str, agent_id: str) -> bool:
        identity = self._identity(tenant_key, agent_id)
        with self._condition:
            return identity in self._fenced

    def _require_active(self, lease: EmployeeExternalMutationLease) -> None:
        identity = (lease.tenant_key, lease.agent_id)
        with self._condition:
            active = self._active.get(identity, {})
            if lease._released or active.get(lease.token) is not lease.kind:
                raise ExternalMutationFenced("external mutation lease is inactive")

    def _release(self, lease: EmployeeExternalMutationLease) -> None:
        identity = (lease.tenant_key, lease.agent_id)
        with self._condition:
            active = self._active.get(identity)
            if active is None or active.get(lease.token) is not lease.kind:
                return
            del active[lease.token]
            if not active:
                self._active.pop(identity, None)
            self._condition.notify_all()

    @staticmethod
    def _identity(tenant_key: str, agent_id: str) -> tuple[str, str]:
        for field_name, value in (
            ("tenant_key", tenant_key),
            ("agent_id", agent_id),
        ):
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(f"{field_name} is required")
        return tenant_key, agent_id


__all__ = [
    "EXTERNAL_TASK_NAME_PREFIX",
    "EmployeeExternalMutationGate",
    "EmployeeExternalMutationLease",
    "ExternalMutationFenced",
    "ExternalMutationKind",
    "ExternalTaskTerminalError",
    "external_task_name",
    "is_external_task",
]
