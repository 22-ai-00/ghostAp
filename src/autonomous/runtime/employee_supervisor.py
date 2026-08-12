"""Process-local owner and recovery facade for employee actors."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from ..journal.frame import JournalEvent
from ..journal.writer import CommitState, JournalWriter
from .employee_actor import (
    EmployeeActor,
    EmployeeActorStatus,
    EmployeeAssignment,
    EmployeeAssignmentTerminal,
    EmployeeCancellationOutcome,
)

AssignmentLoader = Callable[[str, dict[str, object]], EmployeeAssignment | None]


@dataclass(frozen=True, slots=True)
class EmployeeActorSnapshot:
    agent_id: str
    status: EmployeeActorStatus
    mailbox_depth: int
    active_assignment_id: str = ""


class EmployeeRuntimeSupervisor:
    def __init__(
        self,
        *,
        session_factory: Callable[[Any], Any] | None = None,
        terminal_sink: Callable[[EmployeeAssignmentTerminal], None] | None = None,
        writer: JournalWriter | None = None,
        assignment_loader: AssignmentLoader | None = None,
        idle_ttl_seconds: float = 900.0,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._factory = session_factory
        self._external_sink = terminal_sink or (lambda _terminal: None)
        self._writer = writer
        self._loader = assignment_loader
        self._idle_ttl = idle_ttl_seconds
        self._monotonic = monotonic
        self._actors: dict[str, EmployeeActor] = {}
        self._assignment_owner: dict[str, str] = {}
        self._terminals: dict[str, EmployeeAssignmentTerminal] = {}
        self._terminal_sink_emitted: set[str] = set()
        self._conditions: dict[str, threading.Event] = {}
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._closed = False
        self._retired: set[str] = set()
        self._recovering = False

    def ensure_employee(self, agent_id: str) -> EmployeeActor:
        with self._lock:
            if self._closed:
                raise RuntimeError("employee supervisor is closed")
            if agent_id in self._retired:
                raise RuntimeError("employee actor is retired")
            actor = self._actors.get(agent_id)
            if actor is None:
                kwargs: dict[str, object] = {}
                if self._monotonic is not None:
                    kwargs["monotonic"] = self._monotonic
                actor = EmployeeActor(
                    agent_id,
                    session_factory=self._factory,
                    terminal_sink=self._record_terminal,
                    writer=self._writer,
                    idle_ttl_seconds=self._idle_ttl,
                    **kwargs,
                )
                self._actors[agent_id] = actor
            return actor

    def recover(self) -> int:
        if self._writer is None:
            return len(self._actors)
        with self._lock:
            if self._recovering:
                raise RuntimeError("employee supervisor recovery is already running")
            if any(
                assignment_id not in self._terminals
                for assignment_id in self._assignment_owner
            ):
                raise RuntimeError(
                    "employee supervisor recovery conflicts with live assignments"
                )
            self._recovering = True
        try:
            return self._recover_from_journal()
        finally:
            with self._lock:
                self._recovering = False

    def _recover_from_journal(self) -> int:
        assert self._writer is not None
        records: dict[str, dict[str, object]] = {}
        for frame in self._writer.replay():
            for event in frame.events:
                if not event.aggregate_id.startswith("employee-assignment:"):
                    continue
                assignment_id = event.aggregate_id.removeprefix("employee-assignment:")
                record = records.setdefault(assignment_id, {})
                if event.event_type == "employee.actor.assignment_queued":
                    record.update(event.payload)
                elif event.event_type == "employee.actor.assignment_terminal":
                    terminal_payload = dict(event.payload)
                    existing = record.get("terminal_payload")
                    if existing is not None and existing != terminal_payload:
                        raise RuntimeError("employee actor terminal replay conflicts")
                    record["terminal_payload"] = terminal_payload
        recovered = 0
        for assignment_id, record in records.items():
            terminal_payload = record.get("terminal_payload")
            if isinstance(terminal_payload, dict):
                agent_id = terminal_payload.get("agent_id")
                queued_agent_id = record.get("agent_id")
                status = terminal_payload.get("status")
                error_code = terminal_payload.get("error_code")
                if (
                    not isinstance(agent_id, str)
                    or not agent_id
                    or (
                        isinstance(queued_agent_id, str)
                        and queued_agent_id
                        and queued_agent_id != agent_id
                    )
                    or status not in {"completed", "timeout", "canceled", "action_required"}
                    or not isinstance(error_code, str)
                ):
                    raise RuntimeError("employee actor terminal replay is invalid")
                self._restore_terminal(
                    agent_id,
                    EmployeeAssignmentTerminal(
                        assignment_id,
                        status,
                        error_code=error_code,
                    ),
                )
                continue
            agent_id = record.get("agent_id")
            payload_ref = record.get("payload_ref", "")
            assignment = None
            if (
                isinstance(agent_id, str)
                and isinstance(payload_ref, str)
                and payload_ref
                and self._loader is not None
            ):
                assignment = self._loader(payload_ref, dict(record))
            if assignment is not None:
                with self._lock:
                    self._assignment_owner[assignment_id] = agent_id
                    self._conditions.setdefault(assignment_id, threading.Event())
                self.ensure_employee(agent_id).submit(assignment, recovered=True)
            elif isinstance(agent_id, str):
                terminal = EmployeeAssignmentTerminal(
                    assignment_id,
                    "action_required",
                    error_code="employee_recovery_payload_unavailable",
                )
                self._commit_recovery_terminal(agent_id, terminal)
                self._record_terminal(terminal)
            recovered += 1
        return recovered

    def status(self, agent_id: str) -> EmployeeActorStatus:
        return self.ensure_employee(agent_id).status

    def inspect(self, agent_id: str) -> EmployeeActorSnapshot:
        """Read status without allocating a cold actor as a side effect."""

        with self._lock:
            actor = self._actors.get(agent_id)
            retired = agent_id in self._retired
        if actor is None:
            return EmployeeActorSnapshot(
                agent_id,
                EmployeeActorStatus.STOPPED if retired else EmployeeActorStatus.READY_COLD,
                0,
            )
        inspected = actor.inspect_snapshot()
        return EmployeeActorSnapshot(
            agent_id,
            inspected.status,
            inspected.mailbox_depth,
            inspected.active_assignment_id,
        )

    def submit(self, assignment: EmployeeAssignment) -> str:
        assignment_id = assignment.assignment_id
        agent_id = assignment.bootstrap.session_key.agent_id
        with self._lock:
            if self._recovering:
                raise RuntimeError("employee supervisor recovery is running")
            owner = self._assignment_owner.get(assignment_id)
            if owner is not None and owner != agent_id:
                raise ValueError("assignment identity belongs to another employee")
            self._assignment_owner[assignment_id] = agent_id
            self._conditions.setdefault(assignment_id, threading.Event())
        return self.ensure_employee(agent_id).submit(assignment)

    def terminal(self, assignment_id: str) -> EmployeeAssignmentTerminal | None:
        """Return one live or Journal-restored terminal without waiting."""

        if (
            not isinstance(assignment_id, str)
            or not assignment_id
            or assignment_id != assignment_id.strip()
        ):
            raise ValueError("assignment_id is required")
        with self._lock:
            return self._terminals.get(assignment_id)

    def wait_terminal(
        self,
        assignment_id: str,
        *,
        timeout: float | None = None,
    ) -> EmployeeAssignmentTerminal:
        with self._lock:
            terminal = self._terminals.get(assignment_id)
            event = self._conditions.setdefault(assignment_id, threading.Event())
        if terminal is None and not event.wait(timeout):
            raise TimeoutError("employee assignment terminal wait timed out")
        with self._lock:
            return self._terminals[assignment_id]

    def cancel(self, assignment_id: str) -> EmployeeCancellationOutcome:
        with self._lock:
            owner = self._assignment_owner.get(assignment_id)
            terminal = self._terminals.get(assignment_id)
            actor = self._actors.get(owner or "")
        if terminal is not None:
            return EmployeeCancellationOutcome(assignment_id, "terminal", False)
        if actor is None:
            return EmployeeCancellationOutcome(assignment_id, "not_found", False)
        return actor.cancel(assignment_id)

    def recycle(self, agent_id: str, reason: str) -> None:
        self.ensure_employee(agent_id).recycle(reason)

    def retire_employee(self, agent_id: str) -> None:
        """Close mailbox/session and fence recreation before authority removal."""

        with self._lock:
            self._retired.add(agent_id)
            actor = self._actors.pop(agent_id, None)
        if actor is not None:
            actor.close()

    def is_retired(self, agent_id: str) -> bool:
        """Return whether the process-local recreation fence is installed."""

        with self._lock:
            return agent_id in self._retired

    def sweep_idle(self) -> int:
        with self._lock:
            actors = tuple(self._actors.values())
        return sum(actor.reap_idle() for actor in actors)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            actors = tuple(self._actors.values())
        for actor in actors:
            actor.close()

    def _restore_terminal(
        self,
        agent_id: str,
        terminal: EmployeeAssignmentTerminal,
    ) -> None:
        """Index a replayed fact without re-emitting its process-local sink."""

        with self._lock:
            owner = self._assignment_owner.get(terminal.assignment_id)
            if owner is not None and owner != agent_id:
                raise RuntimeError("employee actor terminal owner conflicts")
            existing = self._terminals.get(terminal.assignment_id)
            if existing is not None:
                terminal = self._merge_terminal(existing, terminal)
            self._assignment_owner[terminal.assignment_id] = agent_id
            self._terminals[terminal.assignment_id] = terminal
            event = self._conditions.setdefault(
                terminal.assignment_id,
                threading.Event(),
            )
            event.set()

    def _record_terminal(self, terminal: EmployeeAssignmentTerminal) -> None:
        with self._lock:
            existing = self._terminals.get(terminal.assignment_id)
            if existing is not None:
                terminal = self._merge_terminal(existing, terminal)
            self._terminals[terminal.assignment_id] = terminal
            event = self._conditions.setdefault(terminal.assignment_id, threading.Event())
            event.set()
            if terminal.assignment_id in self._terminal_sink_emitted:
                return
            self._terminal_sink_emitted.add(terminal.assignment_id)
        self._external_sink(terminal)

    @staticmethod
    def _merge_terminal(
        existing: EmployeeAssignmentTerminal,
        incoming: EmployeeAssignmentTerminal,
    ) -> EmployeeAssignmentTerminal:
        """Merge the Journal placeholder with a matching live terminal."""

        if (
            existing.assignment_id != incoming.assignment_id
            or existing.status != incoming.status
            or existing.error_code != incoming.error_code
        ):
            raise RuntimeError("employee actor terminal result conflicts")
        if existing.output and incoming.output and existing.output != incoming.output:
            raise RuntimeError("employee actor terminal output conflicts")
        if incoming.output and not existing.output:
            return incoming
        return existing

    def _commit_recovery_terminal(
        self,
        agent_id: str,
        terminal: EmployeeAssignmentTerminal,
    ) -> None:
        assert self._writer is not None
        aggregate_id = f"employee-assignment:{terminal.assignment_id}"
        event = JournalEvent(
            event_type="employee.actor.assignment_terminal",
            aggregate_id=aggregate_id,
            payload={
                "agent_id": agent_id,
                "status": terminal.status,
                "error_code": terminal.error_code,
                "output_digest": "",
            },
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
            raise RuntimeError("employee recovery terminal was not anchored")


__all__ = ["EmployeeActorSnapshot", "EmployeeRuntimeSupervisor"]
