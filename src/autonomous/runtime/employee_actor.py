"""Serialized, Journal-anchored logical employee actor."""

from __future__ import annotations

import hashlib
import queue
import threading
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

from ..journal.frame import JournalEvent
from ..journal.writer import CommitState, JournalWriter
from .employee_session import EmployeeSessionBootstrap


class EmployeeActorStatus(StrEnum):
    RECOVERING = "recovering"
    READY_COLD = "ready_cold"
    STARTING_SESSION = "starting_session"
    READY_WARM = "ready_warm"
    BUSY = "busy"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class EmployeeAssignment:
    assignment_id: str
    bootstrap: EmployeeSessionBootstrap
    prompt: str = field(repr=False)
    timeout_seconds: float
    payload_ref: str = ""
    session_factory: Callable[[EmployeeSessionBootstrap], Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.assignment_id or not self.assignment_id.strip():
            raise ValueError("assignment_id is required")
        if not self.prompt:
            raise ValueError("assignment prompt is required")
        if not 0 < float(self.timeout_seconds):
            raise ValueError("assignment timeout must be positive")


@dataclass(frozen=True, slots=True)
class EmployeeAssignmentTerminal:
    assignment_id: str
    status: str
    output: str = ""
    error_code: str = ""


@dataclass(frozen=True, slots=True)
class EmployeeCancellationOutcome:
    assignment_id: str
    status: str
    changed: bool


@dataclass(frozen=True, slots=True)
class EmployeeActorInspection:
    status: EmployeeActorStatus
    mailbox_depth: int
    active_assignment_id: str = ""


class EmployeeActor:
    """One employee mailbox; no two assignments execute concurrently."""

    def __init__(
        self,
        agent_id: str,
        *,
        session_factory: Callable[[EmployeeSessionBootstrap], Any] | None,
        terminal_sink: Callable[[EmployeeAssignmentTerminal], None],
        writer: JournalWriter | None = None,
        idle_ttl_seconds: float = 900.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not agent_id:
            raise ValueError("agent_id is required")
        if idle_ttl_seconds <= 0:
            raise ValueError("idle_ttl_seconds must be positive")
        self.agent_id = agent_id
        self._factory = session_factory
        self._sink = terminal_sink
        self._writer = writer
        self._idle_ttl = float(idle_ttl_seconds)
        self._monotonic = monotonic
        self._queue: queue.Queue[EmployeeAssignment | None] = queue.Queue()
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._known: set[str] = set()
        self._terminal: set[str] = set()
        self._canceled: set[str] = set()
        self._active_id = ""
        self._session: Any = None
        self._session_key: object = None
        self._status = EmployeeActorStatus.READY_COLD
        self._closed = False
        self._last_activity = self._monotonic()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"employee-{agent_id}",
        )
        self._thread.start()

    @property
    def status(self) -> EmployeeActorStatus:
        self.reap_idle()
        with self._lock:
            return self._status

    @property
    def mailbox_depth(self) -> int:
        return self._queue.qsize()

    @property
    def active_assignment_id(self) -> str:
        with self._lock:
            return self._active_id

    def inspect_snapshot(self) -> EmployeeActorInspection:
        """Return a coarse process-local view without reaping the live session."""

        with self._lock:
            return EmployeeActorInspection(
                status=self._status,
                mailbox_depth=self._queue.qsize(),
                active_assignment_id=self._active_id,
            )

    def submit(self, assignment: EmployeeAssignment, *, recovered: bool = False) -> str:
        if assignment.bootstrap.session_key.agent_id != self.agent_id:
            raise ValueError("assignment employee mismatch")
        with self._lock:
            if self._closed:
                raise RuntimeError("employee actor is closed")
            if assignment.assignment_id in self._known:
                return assignment.assignment_id
            if not recovered:
                self._commit(
                    assignment.assignment_id,
                    "employee.actor.assignment_queued",
                    {
                        "agent_id": self.agent_id,
                        "payload_ref": assignment.payload_ref,
                        "prompt_digest": hashlib.sha256(
                            assignment.prompt.encode("utf-8")
                        ).hexdigest(),
                        "timeout_seconds": float(assignment.timeout_seconds),
                        "session_key": {
                            "tenant_key": assignment.bootstrap.session_key.tenant_key,
                            "agent_id": self.agent_id,
                            "project_root": assignment.bootstrap.session_key.project_root,
                            "backend": assignment.bootstrap.session_key.backend,
                            "model": assignment.bootstrap.session_key.model,
                            "profile": assignment.bootstrap.session_key.profile,
                            "effort": assignment.bootstrap.session_key.effort,
                            "identity_version": assignment.bootstrap.session_key.identity_version,
                            "instruction_digest": assignment.bootstrap.session_key.instruction_digest,
                        },
                    },
                )
            self._known.add(assignment.assignment_id)
            self._queue.put(assignment)
        return assignment.assignment_id

    def cancel(self, assignment_id: str) -> EmployeeCancellationOutcome:
        with self._lock:
            if assignment_id in self._terminal:
                return EmployeeCancellationOutcome(assignment_id, "terminal", False)
            if assignment_id not in self._known:
                return EmployeeCancellationOutcome(assignment_id, "not_found", False)
            changed = assignment_id not in self._canceled
            self._canceled.add(assignment_id)
            if changed:
                self._commit(
                    assignment_id,
                    "employee.actor.cancel_requested",
                    {"agent_id": self.agent_id},
                )
            session = self._session if self._active_id == assignment_id else None
        cancel = getattr(session, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
        return EmployeeCancellationOutcome(assignment_id, "cancel_requested", changed)

    def drain(self) -> None:
        self._queue.join()

    def reap_idle(self) -> bool:
        with self._lock:
            if (
                self._session is None
                or self._active_id
                or self._monotonic() - self._last_activity < self._idle_ttl
            ):
                return False
            self._close_session_locked()
            if not self._closed:
                self._status = EmployeeActorStatus.READY_COLD
            return True

    def recycle(self, _reason: str) -> None:
        with self._lock:
            if self._active_id:
                raise RuntimeError("cannot recycle a busy employee actor")
            self._close_session_locked()
            if not self._closed:
                self._status = EmployeeActorStatus.READY_COLD

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._status = EmployeeActorStatus.STOPPING
            self._canceled.update(self._known - self._terminal)
            session = self._session
            self._queue.put(None)
        cancel = getattr(session, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
        self._thread.join(timeout=5)
        with self._lock:
            self._close_session_locked()
            self._status = EmployeeActorStatus.STOPPED

    def _run(self) -> None:
        while True:
            assignment = self._queue.get()
            try:
                if assignment is None:
                    return
                self._execute(assignment)
            finally:
                self._queue.task_done()

    def _execute(self, assignment: EmployeeAssignment) -> None:
        with self._lock:
            self._active_id = assignment.assignment_id
            canceled = assignment.assignment_id in self._canceled
        if canceled:
            self._finish(
                EmployeeAssignmentTerminal(
                    assignment.assignment_id,
                    "canceled",
                    error_code="employee_assignment_canceled",
                )
            )
            return
        try:
            self._commit_effect(assignment.assignment_id, "prepared")
            self._commit_effect(assignment.assignment_id, "executing")
            with self._lock:
                healthy = callable(
                    getattr(self._session, "is_server_healthy", None)
                ) and self._session.is_server_healthy()
                if self._session_key != assignment.bootstrap.session_key or not healthy:
                    self._close_session_locked()
                    self._status = EmployeeActorStatus.STARTING_SESSION
                    factory = assignment.session_factory or self._factory
                    if factory is None:
                        raise RuntimeError("employee session factory is unavailable")
                    self._session = factory(assignment.bootstrap)
                    self._session_key = assignment.bootstrap.session_key
                self._status = EmployeeActorStatus.BUSY
                session = self._session
            result = session.send_prompt(
                assignment.bootstrap.wrap_prompt(assignment.prompt),
                timeout=assignment.timeout_seconds,
            )
            with self._lock:
                canceled = assignment.assignment_id in self._canceled
            if canceled:
                terminal = EmployeeAssignmentTerminal(
                    assignment.assignment_id,
                    "canceled",
                    error_code="employee_assignment_canceled",
                )
            else:
                output = getattr(result, "text", "")
                if not isinstance(output, str) or not output:
                    raise RuntimeError("employee session returned no output")
                terminal = EmployeeAssignmentTerminal(
                    assignment.assignment_id,
                    "completed",
                    output=output,
                )
        except TimeoutError:
            terminal = EmployeeAssignmentTerminal(
                assignment.assignment_id,
                "timeout",
                error_code="employee_session_timeout",
            )
            with self._lock:
                self._close_session_locked()
        except CancelledError:
            terminal = EmployeeAssignmentTerminal(
                assignment.assignment_id,
                "canceled",
                error_code="employee_assignment_canceled",
            )
            with self._lock:
                self._close_session_locked()
        except Exception:
            terminal = EmployeeAssignmentTerminal(
                assignment.assignment_id,
                "action_required",
                error_code="employee_session_failed",
            )
            with self._lock:
                self._close_session_locked()
                self._status = EmployeeActorStatus.DEGRADED
        self._finish(terminal)

    def _finish(self, terminal: EmployeeAssignmentTerminal) -> None:
        with self._lock:
            if terminal.assignment_id in self._terminal:
                return
            try:
                self._commit(
                    terminal.assignment_id,
                    "employee.actor.assignment_terminal",
                    {
                        "agent_id": self.agent_id,
                        "status": terminal.status,
                        "error_code": terminal.error_code,
                        "output_digest": hashlib.sha256(
                            terminal.output.encode("utf-8")
                        ).hexdigest()
                        if terminal.output
                        else "",
                    },
                )
            except Exception:
                terminal = EmployeeAssignmentTerminal(
                    terminal.assignment_id,
                    "action_required",
                    error_code="employee_result_anchor_failed",
                )
            self._terminal.add(terminal.assignment_id)
            self._active_id = ""
            self._last_activity = self._monotonic()
            if not self._closed and self._status is not EmployeeActorStatus.DEGRADED:
                self._status = (
                    EmployeeActorStatus.READY_WARM
                    if self._session is not None
                    else EmployeeActorStatus.READY_COLD
                )
        self._sink(terminal)

    def _commit_effect(self, assignment_id: str, state: str) -> None:
        self._commit(
            assignment_id,
            f"employee.actor.effect_{state}",
            {"agent_id": self.agent_id, "effect_type": "backend_prompt"},
        )

    def _commit(self, assignment_id: str, event_type: str, payload: dict[str, Any]) -> None:
        if self._writer is None:
            return
        aggregate_id = f"employee-assignment:{assignment_id}"
        event = JournalEvent(
            event_type=event_type,
            aggregate_id=aggregate_id,
            payload=payload,
        )
        with self._writer.transaction_guard():
            last = self._writer.get_last_frame()
            result = self._writer.commit(
                (event,),
                self._writer.get_aggregate_versions((aggregate_id,)),
                expected_head_sequence=0 if last is None else last.sequence,
                expected_head_hash="" if last is None else last.frame_hash,
            )
        if result.state is not CommitState.ANCHORED:
            raise RuntimeError("employee actor event was not anchored")

    def _close_session_locked(self) -> None:
        session, self._session = self._session, None
        self._session_key = None
        close = getattr(session, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


__all__ = [
    "EmployeeActor",
    "EmployeeActorInspection",
    "EmployeeActorStatus",
    "EmployeeAssignment",
    "EmployeeAssignmentTerminal",
    "EmployeeCancellationOutcome",
]
