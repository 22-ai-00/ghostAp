"""Coordinator-only facade for visible employee collaboration runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Protocol

from ..journal.blob_store import BlobStore
from ..journal.writer import JournalWriter
from .coordinator import DecisionProvider, TeamCoordinatorActor
from .models import TeamRunPhase

_MAX_TASK_CHARS = 4_000


class TeamServiceError(RuntimeError):
    """A team run could not safely progress."""


@dataclass(frozen=True, slots=True)
class TeamTarget:
    agent_id: str
    name: str
    role: str = ""
    capabilities: tuple[str, ...] = ()
    runtime_status: str = "ready"
    mailbox_load: int = 0


@dataclass(frozen=True, slots=True)
class TeamAttemptResult:
    status: str
    output: str = ""
    history_record_id: str = ""
    error_code: str = ""
    retry_allowed: bool = True


@dataclass(frozen=True, slots=True)
class TeamRunState:
    run_id: str
    tenant_key: str
    message_id: str
    chat_id: str
    requester_principal_id: str
    task_digest: str
    status: str = "running"
    result: str = ""


class TeamAdmissionError(TeamServiceError):
    """A team run was rejected before durable admission."""

    def __init__(self, error_code: str) -> None:
        if not isinstance(error_code, str) or not error_code.strip():
            raise ValueError("team admission error_code is required")
        super().__init__(error_code)
        self.error_code = error_code


class TeamBackend(Protocol):
    def list_active(self, tenant_key: str, chat_id: str) -> tuple[TeamTarget, ...]: ...

    def submit(
        self,
        *,
        run_id: str,
        step_id: str,
        target: TeamTarget,
        tenant_key: str,
        chat_id: str,
        message_id: str,
        requester_principal_id: str,
        instruction: str,
        deadline_at: str,
    ) -> str: ...

    def result(self, acceptance_id: str) -> TeamAttemptResult | None: ...

    def cancel(
        self,
        acceptance_id: str,
        *,
        run_id: str,
        step_id: str,
    ) -> TeamAttemptResult: ...

    def notify(
        self,
        message_id: str,
        chat_id: str,
        result: str,
        *,
        idempotency_key: str = "",
        tenant_key: str = "",
        requester_principal_id: str = "",
    ) -> None: ...

    def submit_direct(
        self,
        *,
        target: TeamTarget,
        tenant_key: str,
        chat_id: str,
        message_id: str,
        requester_principal_id: str,
        instruction: str,
    ) -> str: ...


class EmployeeTeamService:
    """Expose the single durable TeamCoordinator execution path."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        backend: TeamBackend,
        blob_store: BlobStore | None,
        active_key_id: str,
        attempt_timeout_seconds: float = 600.0,
        poll_seconds: float = 0.1,
        clock: Callable[[], datetime] | None = None,
        coordinator_tool: str = "coco",
        coordinator_model: str = "",
        coordinator_profile: str = "",
        coordinator_effort: str = "",
        coordinator_decision_provider: DecisionProvider | None = None,
    ) -> None:
        if blob_store is None or not active_key_id:
            raise ValueError("Team Coordinator requires encrypted Blob storage")
        self._backend = backend
        self._coordinator = TeamCoordinatorActor(
            writer=writer,
            blob_store=blob_store,
            active_key_id=active_key_id,
            backend=backend,
            coordinator_tool=coordinator_tool,
            coordinator_model=coordinator_model,
            coordinator_profile=coordinator_profile,
            coordinator_effort=coordinator_effort,
            attempt_timeout_seconds=attempt_timeout_seconds,
            poll_seconds=poll_seconds,
            clock=clock,
            decision_provider=coordinator_decision_provider,
        )

    def start_task(
        self,
        *,
        tenant_key: str,
        message_id: str,
        chat_id: str,
        requester_principal_id: str,
        task: str,
    ) -> TeamRunState:
        values = (tenant_key, message_id, chat_id, requester_principal_id, task)
        if not all(isinstance(value, str) and value.strip() == value and value for value in values):
            raise ValueError("team task coordinates are required")
        if len(task) > _MAX_TASK_CHARS:
            raise ValueError("team task exceeds maximum length")
        run_id = self._coordinator.task_run_id(
            tenant_key=tenant_key,
            chat_id=chat_id,
            message_id=message_id,
        )
        existing = self._coordinator.projection().runs.get(run_id)
        if existing is not None:
            state = self._adapt(existing)
            if state.status != "running":
                raise TeamAdmissionError(f"team_run_{state.status}")
            return state
        if not tuple(self._backend.list_active(tenant_key, chat_id)):
            raise TeamAdmissionError("no_active_team_employee")
        run, created = self._coordinator.admit_task(
            tenant_key=tenant_key,
            message_id=message_id,
            chat_id=chat_id,
            requester_principal_id=requester_principal_id,
            task=task,
        )
        state = self._adapt(run)
        if not created and state.status != "running":
            raise TeamAdmissionError(f"team_run_{state.status}")
        return state

    def get_run(self, run_id: str) -> TeamRunState | None:
        run = self._coordinator.projection().runs.get(run_id)
        return None if run is None else self._adapt(run)

    def dispatch_direct(
        self,
        *,
        target: TeamTarget,
        tenant_key: str,
        chat_id: str,
        message_id: str,
        requester_principal_id: str,
        instruction: str,
    ) -> str:
        submit = getattr(self._backend, "submit_direct", None)
        if not callable(submit):
            raise TeamServiceError("direct employee dispatch is unavailable")
        return submit(
            target=target,
            tenant_key=tenant_key,
            chat_id=chat_id,
            message_id=message_id,
            requester_principal_id=requester_principal_id,
            instruction=instruction,
        )

    def record_collaboration_event(self, **coordinates: str) -> bool:
        return self._coordinator.record_collaboration_event(**coordinates)

    def recover(self) -> int:
        return self._coordinator.recover()

    def close(self) -> None:
        self._coordinator.close()

    @staticmethod
    def _adapt(run: object) -> TeamRunState:
        phase = run.phase
        status = (
            "completed"
            if phase is TeamRunPhase.COMPLETED
            else "action_required"
            if phase is TeamRunPhase.BLOCKED
            else "canceled"
            if phase is TeamRunPhase.CANCELED
            else "running"
        )
        return TeamRunState(
            run_id=run.run_id,
            tenant_key=run.tenant_key,
            message_id=run.message_id,
            chat_id=run.chat_id,
            requester_principal_id=run.requester_principal_id,
            task_digest=run.task_ref.payload_hash,
            status=status,
        )


__all__ = [
    "EmployeeTeamService",
    "TeamAdmissionError",
    "TeamAttemptResult",
    "TeamBackend",
    "TeamRunState",
    "TeamServiceError",
    "TeamTarget",
]
