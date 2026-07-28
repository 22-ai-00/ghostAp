"""Production-shaped durable admission core for visible employee hiring."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import threading
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from ...utils.async_helpers import safe_wait_for
from ..domain import EmployeeState, WorkerType
from ..journal.frame import JournalEvent, TransactionFrame
from ..journal.projections import (
    ProjectionError,
    ProjectionRepository,
    ProjectionState,
    apply_frame,
)
from ..journal.writer import AnchorMismatchError, CommitState, JournalWriter
from ..workforce.projection import (
    commit_workforce_events_unlocked,
    is_workforce_event,
    validate_workforce_events,
    workforce_projection_guard,
)
from .callback_bridge import AsyncCallbackBridge
from .hire_port import EmployeeHireRequest, complete_employee_hire_request
from .hire_state import (
    ACTIVATION_CHALLENGE_RENEWAL_LEAD_SECONDS,
    DurableHireState,
    HireEffectState,
    HirePhase,
    HireProjection,
)
from .lark_app import (
    MANIFEST_EVIDENCE_SOURCE,
    RegistrationRequest,
    RegistrationResult,
    current_registration_manifest,
)
from .manifest_reauthorization import (
    ManifestReauthorizationPhase,
    ManifestReauthorizationState,
    apply_manifest_reauthorization_event,
    is_manifest_reauthorization_event,
    rebuild_manifest_reauthorizations,
)
from .verification import (
    VerificationChallenge,
    VerificationDecision,
    VerificationOutcome,
)

_EFFECT_METADATA_KEYS = frozenset(
    {
        "app_id",
        "credential_ref",
        "slash_spec_hash",
        "slash_observed_hash",
        "slash_verified_at",
        "generation",
        "identity_app_id",
        "connection_id",
        "channel_verified_at",
        "send_request_id",
        "ingress_event_id",
        "reply_app_id",
        "reply_message_id",
        "main_bot_send_count",
        "error_code",
    }
)


class HireAdmissionError(RuntimeError):
    """A visible employee request was rejected before external activity."""


@dataclass(frozen=True)
class HireReadiness:
    ready: bool
    blockers: tuple[str, ...]


class AppRegistrarPort(Protocol):
    async def register(
        self,
        request: RegistrationRequest,
        *,
        on_link: Callable[[str, int], None],
        on_status: Callable[[str], None] | None = None,
    ) -> RegistrationResult: ...


class CredentialVaultPort(Protocol):
    def put(
        self,
        agent_id: str,
        app_id: str,
        app_secret: str,
        hire_intent_id: str,
        attempt_id: str,
    ) -> Any: ...

    def find_orphan_receipts(self, live_credential_refs: set[str]) -> list[Any]: ...


RegistrationLinkCallback = Callable[
    [DurableHireState, str, int], object | Awaitable[object]
]
RegistrationStatusCallback = Callable[
    [DurableHireState, str], object | Awaitable[object]
]


def _stable_id(prefix: str, tenant_key: str, message_id: str) -> str:
    canonical = json.dumps(
        [prefix, tenant_key, message_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(canonical).hexdigest()}"


def _frame_matches(
    frame: TransactionFrame,
    expected: tuple[tuple[str, str, Mapping[str, object]], ...],
) -> bool:
    return len(frame.events) == len(expected) and all(
        event.event_type == event_type
        and event.aggregate_id == aggregate_id
        and dict(event.payload) == dict(payload)
        for event, (event_type, aggregate_id, payload) in zip(
            frame.events,
            expected,
            strict=True,
        )
    )


class ProductionEmployeeHireService:
    """Admit, execute, and recover the Journal/Vault hire Saga."""

    def __init__(
        self,
        writer: JournalWriter,
        projection_state: ProjectionState,
        *,
        visible_employee_limit: int = 0,
        release_evidence_ready: bool = False,
        credential_keyring_ready: bool = False,
        registrar: AppRegistrarPort | None = None,
        credential_vault: CredentialVaultPort | None = None,
        on_registration_link: RegistrationLinkCallback | None = None,
        on_registration_status: RegistrationStatusCallback | None = None,
        provisioning_submitter: Callable[[str], object] | None = None,
        manifest_reauthorization_submitter: Callable[[str], object] | None = None,
        manifest_reauthorization_timeout_seconds: float = 300.0,
        runtime_recovery_ready: bool = True,
        workspace_projector: Any = None,
    ) -> None:
        if (
            isinstance(visible_employee_limit, bool)
            or not isinstance(visible_employee_limit, int)
            or visible_employee_limit < 0
        ):
            raise ValueError("visible_employee_limit must be a non-negative integer")
        self._writer = writer
        self._projection_state = projection_state
        self._visible_employee_limit = visible_employee_limit
        self._release_evidence_ready = release_evidence_ready is True
        self._credential_keyring_ready = credential_keyring_ready is True
        self._registrar = registrar
        self._credential_vault = credential_vault
        self._on_registration_link = on_registration_link
        self._on_registration_status = on_registration_status
        self._provisioning_submitter = provisioning_submitter
        self._manifest_reauthorization_submitter = (
            manifest_reauthorization_submitter
        )
        if (
            isinstance(manifest_reauthorization_timeout_seconds, bool)
            or not isinstance(manifest_reauthorization_timeout_seconds, (int, float))
            or not math.isfinite(float(manifest_reauthorization_timeout_seconds))
            or manifest_reauthorization_timeout_seconds <= 0
        ):
            raise ValueError(
                "manifest_reauthorization_timeout_seconds must be positive"
            )
        self._manifest_reauthorization_timeout_seconds = float(
            manifest_reauthorization_timeout_seconds
        )
        self._runtime_recovery_ready = runtime_recovery_ready is True
        self._workspace_projector = workspace_projector
        self._mutex = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._activities: dict[str, asyncio.Task[DurableHireState]] = {}
        self._manifest_activities: dict[
            str, asyncio.Task[ManifestReauthorizationState]
        ] = {}
        self._admission_closed = False
        self._closed = False
        self._hire_projection = HireProjection.empty()
        self._manifest_reauthorizations: dict[
            str, ManifestReauthorizationState
        ] = {}
        self.recover()

    @property
    def projection_state(self) -> ProjectionState:
        return self._projection_state

    @contextmanager
    def employee_dispatch_guard(self):
        """Hold workforce authority then hire state, without taking Journal."""

        with workforce_projection_guard(), self._mutex:
            yield

    def synchronize_projection(self) -> ProjectionState:
        """Advance the read model across frames committed by sibling domains."""
        with self._mutex:
            self._synchronize_projection_to_journal_locked()
            return self._projection_state

    def synchronize_projection_unlocked(self) -> ProjectionState:
        """Advance views while the caller owns ``employee_dispatch_guard``."""

        self._synchronize_projection_to_journal_locked()
        return self._projection_state

    def get_manifest_reauthorization(
        self,
        operation_id: str,
    ) -> ManifestReauthorizationState | None:
        with self._mutex:
            self._synchronize_projection_to_journal_locked()
            return self._manifest_reauthorizations.get(operation_id)

    def request_manifest_reauthorization(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        request_id: str,
    ) -> ManifestReauthorizationState:
        """Durably request an in-place official authorization flow for an ACTIVE App."""

        for name, value in (
            ("tenant_key", tenant_key),
            ("agent_id", agent_id),
            ("request_id", request_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise HireAdmissionError(f"{name} is required")
        if self._registrar is None or self._manifest_reauthorization_submitter is None:
            raise HireAdmissionError("manifest reauthorization dependencies unavailable")
        manifest = current_registration_manifest()
        desired_hash = manifest.fingerprint()
        operation_id = _stable_id(
            "manifestreauth",
            tenant_key,
            f"{request_id}:{agent_id}:{desired_hash}",
        )
        submit = False
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            self._synchronize_projection_to_journal_locked()
            if self._closed or self._admission_closed:
                raise HireAdmissionError("closed")
            existing = self._manifest_reauthorizations.get(operation_id)
            if existing is not None:
                if existing.phase is ManifestReauthorizationPhase.ACTION_REQUIRED:
                    raise HireAdmissionError(
                        "manifest reauthorization requires a fresh request"
                    )
                return existing
            employee = self._projection_state.employees.get(agent_id)
            if (
                employee is None
                or employee.tenant_key != tenant_key
                or employee.state is not EmployeeState.ACTIVE
                or not employee.bot_principal_id
            ):
                raise HireAdmissionError("active employee is required")
            principal = self._projection_state.bot_principals.get(
                employee.bot_principal_id
            )
            if (
                principal is None
                or principal.tenant_key != tenant_key
                or principal.agent_id != agent_id
                or not principal.app_id
            ):
                raise HireAdmissionError("bot principal binding is unavailable")
            active = next(
                (
                    item
                    for item in self._manifest_reauthorizations.values()
                    if item.tenant_key == tenant_key
                    and item.bot_principal_id == principal.bot_principal_id
                    and item.app_id == principal.app_id
                    and item.desired_manifest_hash == desired_hash
                    and item.phase
                    in {
                        ManifestReauthorizationPhase.PREPARED,
                        ManifestReauthorizationPhase.EXECUTING,
                        ManifestReauthorizationPhase.COMMITTED,
                    }
                ),
                None,
            )
            if active is not None:
                return active
            events = (
                JournalEvent(
                    event_type="bot_principal.manifest_desired",
                    aggregate_id=principal.bot_principal_id,
                    payload={
                        "desired_manifest_hash": desired_hash,
                        "scopes": list(manifest.tenant_scopes),
                    },
                ),
                JournalEvent(
                    event_type="manifest.reauthorization.prepared",
                    aggregate_id=operation_id,
                    payload={
                        "tenant_key": tenant_key,
                        "agent_id": agent_id,
                        "bot_principal_id": principal.bot_principal_id,
                        "app_id": principal.app_id,
                        "desired_manifest_hash": desired_hash,
                        "message_id": request_id,
                        "employee_name": employee.name,
                    },
                ),
            )
            state = self._commit_manifest_reauthorization_events_locked(events)
            submit = True
        if submit:
            try:
                self._manifest_reauthorization_submitter(operation_id)
            except Exception:
                with self.employee_dispatch_guard(), self._writer.transaction_guard():
                    self._synchronize_projection_to_journal_locked()
                    self._commit_manifest_reauthorization_events_locked(
                        (
                            JournalEvent(
                                event_type=(
                                    "manifest.reauthorization.action_required"
                                ),
                                aggregate_id=operation_id,
                                payload={"error_code": "activity_submit_failed"},
                            ),
                        )
                    )
                raise HireAdmissionError(
                    "manifest reauthorization requires manual action"
                ) from None
        return state

    async def run_manifest_reauthorization(
        self,
        operation_id: str,
    ) -> ManifestReauthorizationState:
        """Run one deduplicated official existing-App registration activity."""

        if not isinstance(operation_id, str) or not operation_id:
            raise HireAdmissionError("operation_id is required")
        with self._mutex:
            if self._closed:
                raise HireAdmissionError("closed")
            task = self._manifest_activities.get(operation_id)
            if task is None:
                task = asyncio.create_task(
                    self._run_manifest_reauthorization_activity(operation_id)
                )
                self._manifest_activities[operation_id] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                with self._mutex:
                    if self._manifest_activities.get(operation_id) is task:
                        self._manifest_activities.pop(operation_id, None)

    async def _run_manifest_reauthorization_activity(
        self,
        operation_id: str,
    ) -> ManifestReauthorizationState:
        state = self.get_manifest_reauthorization(operation_id)
        if state is None:
            raise HireAdmissionError("unknown manifest reauthorization")
        if state.phase is ManifestReauthorizationPhase.COMMITTED:
            return state
        if state.phase is not ManifestReauthorizationPhase.PREPARED:
            raise HireAdmissionError("manifest reauthorization requires manual action")
        state = self._commit_manifest_reauthorization_event(
            JournalEvent(
                event_type="manifest.reauthorization.executing",
                aggregate_id=operation_id,
                payload={},
            )
        )
        bridge = AsyncCallbackBridge()
        link_callback = bridge.callback(self._on_registration_link, state)
        status_callback = bridge.callback(self._on_registration_status, state)

        async def register_and_drain() -> RegistrationResult:
            if self._registrar is None:
                raise HireAdmissionError("manifest registrar unavailable")
            result: RegistrationResult | None = None
            registration_error: Exception | None = None
            try:
                result = await self._registrar.register(
                    RegistrationRequest(
                        name=state.employee_name,
                        description="GhostAP employee manifest authorization",
                        existing_app_id=state.app_id,
                    ),
                    on_link=link_callback,
                    on_status=status_callback,
                )
            except Exception as exc:
                registration_error = exc
            try:
                await bridge.drain()
            except Exception as exc:
                registration_error = exc
            if registration_error is not None or result is None:
                raise HireAdmissionError("official registration failed")
            return result

        result: RegistrationResult | None = None
        try:
            result = await safe_wait_for(
                register_and_drain(),
                timeout=self._manifest_reauthorization_timeout_seconds,
                action="employee manifest reauthorization",
            )
            if (
                result.app_id != state.app_id
                or result.manifest_hash != state.desired_manifest_hash
                or result.evidence_source != MANIFEST_EVIDENCE_SOURCE
            ):
                raise HireAdmissionError("manifest authorization receipt mismatch")
        except Exception as exc:
            error_code = (
                "registration_timeout"
                if isinstance(exc, TimeoutError)
                else "registration_failed"
            )
            self._commit_manifest_reauthorization_event(
                JournalEvent(
                    event_type="manifest.reauthorization.action_required",
                    aggregate_id=operation_id,
                    payload={"error_code": error_code},
                )
            )
            raise HireAdmissionError(
                "manifest reauthorization requires manual action"
            ) from None
        if result is None:  # pragma: no cover - guarded by the successful await above
            raise HireAdmissionError("manifest reauthorization requires manual action")
        evidence_source = result.evidence_source
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            self._synchronize_projection_to_journal_locked()
            current = self._manifest_reauthorizations.get(operation_id)
            employee = self._projection_state.employees.get(state.agent_id)
            principal = self._projection_state.bot_principals.get(
                state.bot_principal_id
            )
            if (
                current is None
                or current.phase is not ManifestReauthorizationPhase.EXECUTING
                or employee is None
                or employee.state is not EmployeeState.ACTIVE
                or employee.tenant_key != state.tenant_key
                or principal is None
                or principal.app_id != state.app_id
                or principal.agent_id != state.agent_id
                or principal.desired_manifest_hash != state.desired_manifest_hash
            ):
                if current is not None and current.phase is (
                    ManifestReauthorizationPhase.EXECUTING
                ):
                    self._commit_manifest_reauthorization_events_locked(
                        (
                            JournalEvent(
                                event_type=(
                                    "manifest.reauthorization.action_required"
                                ),
                                aggregate_id=operation_id,
                                payload={"error_code": "binding_changed"},
                            ),
                        )
                    )
                raise HireAdmissionError(
                    "manifest reauthorization requires manual action"
                )
            return self._commit_manifest_reauthorization_events_locked(
                (
                    JournalEvent(
                        event_type="manifest.reauthorization.committed",
                        aggregate_id=operation_id,
                        payload={
                            "observed_manifest_hash": state.desired_manifest_hash,
                            "evidence_source": evidence_source,
                        },
                    ),
                    JournalEvent(
                        event_type="bot_principal.manifest_observed",
                        aggregate_id=state.bot_principal_id,
                        payload={
                            "observed_manifest_hash": state.desired_manifest_hash,
                            "evidence_source": evidence_source,
                        },
                    ),
                )
            )

    def recover_manifest_reauthorizations(self) -> tuple[str, ...]:
        """Fail-close interrupted calls and return safe PREPARED work to resume."""

        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            self._synchronize_projection_to_journal_locked()
            active_keys: set[tuple[str, str, str, str]] = {
                (
                    state.tenant_key,
                    state.bot_principal_id,
                    state.app_id,
                    state.desired_manifest_hash,
                )
                for state in self._manifest_reauthorizations.values()
                if state.phase
                in {
                    ManifestReauthorizationPhase.EXECUTING,
                    ManifestReauthorizationPhase.COMMITTED,
                }
            }
            interrupted = tuple(
                state.operation_id
                for state in self._manifest_reauthorizations.values()
                if state.phase is ManifestReauthorizationPhase.EXECUTING
            )
            for operation_id in interrupted:
                self._commit_manifest_reauthorization_events_locked(
                    (
                        JournalEvent(
                            event_type="manifest.reauthorization.action_required",
                            aggregate_id=operation_id,
                            payload={"error_code": "interrupted_remote_outcome"},
                        ),
                    )
                )
            pending: list[str] = []
            prepared = tuple(
                state
                for state in self._manifest_reauthorizations.values()
                if state.phase is ManifestReauthorizationPhase.PREPARED
            )
            for state in prepared:
                key = (
                    state.tenant_key,
                    state.bot_principal_id,
                    state.app_id,
                    state.desired_manifest_hash,
                )
                if key not in active_keys:
                    active_keys.add(key)
                    pending.append(state.operation_id)
                    continue
                self._commit_manifest_reauthorization_events_locked(
                    (
                        JournalEvent(
                            event_type=(
                                "manifest.reauthorization.action_required"
                            ),
                            aggregate_id=state.operation_id,
                            payload={"error_code": "duplicate_active_operation"},
                        ),
                    )
                )
            return tuple(pending)

    def _commit_manifest_reauthorization_event(
        self,
        event: JournalEvent,
    ) -> ManifestReauthorizationState:
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            self._synchronize_projection_to_journal_locked()
            return self._commit_manifest_reauthorization_events_locked((event,))

    def _commit_manifest_reauthorization_events_locked(
        self,
        events: tuple[JournalEvent, ...],
    ) -> ManifestReauthorizationState:
        manifest_events = tuple(
            event
            for event in events
            if is_manifest_reauthorization_event(event.event_type)
        )
        if len(manifest_events) != 1:
            raise HireAdmissionError(
                "manifest reauthorization transaction requires one state event"
            )
        event = manifest_events[0]
        apply_manifest_reauthorization_event(
            self._manifest_reauthorizations.get(event.aggregate_id),
            event,
        )
        workforce_events = tuple(
            item for item in events if is_workforce_event(item.event_type)
        )
        if workforce_events:
            validate_workforce_events(self._projection_state, workforce_events)
        last = self._writer.get_last_frame()
        writer_sequence = 0 if last is None else last.sequence
        writer_hash = "" if last is None else last.frame_hash
        result = self._writer.commit(
            events,
            self._writer.get_aggregate_versions(
                {item.aggregate_id for item in events}
            ),
            expected_head_sequence=writer_sequence,
            expected_head_hash=writer_hash,
        )
        if result.state is not CommitState.ANCHORED:
            raise AnchorMismatchError(
                "manifest reauthorization event was not anchored"
            )
        apply_frame(self._projection_state, result.frame)
        frames = tuple(self._writer.replay())
        self._hire_projection = HireProjection.rebuild(frames)
        self._manifest_reauthorizations = rebuild_manifest_reauthorizations(frames)
        return self._manifest_reauthorizations[event.aggregate_id]

    def apply_committed_frame_unlocked(self, frame: TransactionFrame) -> None:
        """Advance workforce and hire views for a sibling-domain anchored frame."""

        from ..journal.projections import apply_frame

        if not isinstance(frame, TransactionFrame) or not frame.committed:
            raise TypeError("frame must be a committed TransactionFrame")
        if frame.sequence != self._projection_state.cursor_sequence + 1:
            raise HireAdmissionError("hire frame sequence is not continuous")
        apply_frame(self._projection_state, frame)
        frames = tuple(self._writer.replay())
        self._hire_projection = HireProjection.rebuild(frames)
        self._manifest_reauthorizations = rebuild_manifest_reauthorizations(frames)

    def readiness(self) -> HireReadiness:
        blockers: list[str] = []
        if self._closed:
            blockers.append("closed")
        elif self._admission_closed:
            blockers.append("admission_closed")
        if self._visible_employee_limit == 0:
            blockers.append("visible_employee_limit")
        if not self._release_evidence_ready:
            blockers.append("release_evidence")
        if not self._credential_keyring_ready:
            blockers.append("credential_keyring")
        if getattr(self._writer.anchor, "production_safe", False) is not True:
            blockers.append("production_anchor")
        if not self._runtime_recovery_ready:
            blockers.append("runtime_recovery")
        return HireReadiness(ready=not blockers, blockers=tuple(blockers))

    def recover(self) -> HireProjection:
        with self.employee_dispatch_guard():
            if self._closed:
                raise HireAdmissionError("closed")
            if self._admission_closed:
                raise HireAdmissionError("admission closed")
            frames = tuple(self._writer.replay())
            repository = ProjectionRepository()
            rebuilt = repository.rebuild(iter(frames))
            if frames:
                self._projection_state = rebuilt
            self._hire_projection = HireProjection.rebuild(frames)
            self._manifest_reauthorizations = rebuild_manifest_reauthorizations(
                frames
            )
        self._reconcile_recovered_hires()
        with self.employee_dispatch_guard():
            return self._hire_projection

    def start_hire(self, request: EmployeeHireRequest) -> DurableHireState:
        try:
            request = complete_employee_hire_request(request)
        except (TypeError, ValueError):
            raise HireAdmissionError("invalid employee profile") from None
        self._validate_request(request)
        intent_id = _stable_id("hire", request.tenant_key, request.message_id)
        submit_after_commit = False
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            self._synchronize_projection_to_journal_locked()
            if self._closed:
                raise HireAdmissionError("closed")
            if self._admission_closed:
                raise HireAdmissionError("admission closed")
            existing = self._hire_projection.get(intent_id)
            if existing is not None:
                if not self._matches_request(existing, request):
                    raise HireAdmissionError("hire idempotency conflict")
                submit_after_commit = (
                    self._provisioning_submitter is not None
                    and self.readiness().ready
                    and existing.phase
                    in {
                        HirePhase.PROVISIONING_APP,
                        HirePhase.STORING_CREDENTIAL,
                        HirePhase.CONFIGURING,
                        HirePhase.VALIDATING,
                    }
                )
                admitted = existing
            else:
                readiness = self.readiness()
                if not readiness.ready:
                    raise HireAdmissionError(",".join(readiness.blockers))
                requested_app_assigned = (
                    bool(request.existing_app_id)
                    and self._app_id_assigned_locked(
                        request.existing_app_id,
                    )
                )
                if requested_app_assigned:
                    raise HireAdmissionError("existing app already assigned")
                visible_count = sum(
                    employee.worker_type is WorkerType.VISIBLE
                    and employee.state is not EmployeeState.ARCHIVED
                    for employee in self._projection_state.employees.values()
                )
                if visible_count >= self._visible_employee_limit:
                    raise HireAdmissionError("visible_employee_limit capacity reached")
                agent_id = _stable_id("agt", request.tenant_key, request.message_id)
                bot_principal_id = _stable_id(
                    "bot",
                    request.tenant_key,
                    request.message_id,
                )
                attempt_id = _stable_id(
                    "attempt",
                    request.tenant_key,
                    request.message_id,
                )
                event = JournalEvent(
                    event_type="employee.created",
                    aggregate_id=agent_id,
                    payload={
                        "agent_id": agent_id,
                        "tenant_key": request.tenant_key,
                        "owner_principal_id": request.requester_principal_id,
                        "requester_union_id": request.requester_union_id,
                        "name": request.employee_name,
                        "tool": request.tool,
                        "model": request.model,
                        "profile": request.profile,
                        "effort": request.effort,
                        "role": request.role,
                        "persona": request.persona,
                        "personality_traits": list(request.personality_traits),
                        "capabilities": list(request.capabilities),
                        "permissions": list(request.permissions),
                        "existing_app_id": request.existing_app_id,
                        "worker_type": WorkerType.VISIBLE.value,
                        "state": EmployeeState.PROVISIONING_APP.value,
                        "hire_schema_version": 2,
                        "hire_intent_id": intent_id,
                        "hire_message_id": request.message_id,
                        "hire_chat_id": request.chat_id,
                        "planned_bot_principal_id": bot_principal_id,
                        "provisioning_attempt_id": attempt_id,
                    },
                )
                name_owner_id = self._projection_state.employee_name_keys.get(
                    (request.tenant_key, request.employee_name.casefold())
                )
                name_owner = (
                    self._projection_state.employees.get(name_owner_id)
                    if name_owner_id is not None
                    else None
                )
                release_events = (
                    (
                        JournalEvent(
                            event_type="employee.name_released",
                            aggregate_id=name_owner.agent_id,
                            payload={"name": name_owner.name},
                        ),
                    )
                    if name_owner is not None
                    and name_owner.state is EmployeeState.ARCHIVED
                    else ()
                )
                try:
                    commit_workforce_events_unlocked(
                        self._writer,
                        self._projection_state,
                        (*release_events, event),
                    )
                except ProjectionError as exc:
                    detail = (
                        "name" if "name" in str(exc).casefold() else "projection"
                    )
                    raise HireAdmissionError(f"hire {detail} conflict") from exc
                self._hire_projection = HireProjection.rebuild(self._writer.replay())
                admitted = self._hire_projection.get(intent_id)
                if admitted is None:
                    raise HireAdmissionError("anchored hire admission did not replay")
                submit_after_commit = self._provisioning_submitter is not None
        if submit_after_commit and self._provisioning_submitter is not None:
            try:
                self._provisioning_submitter(admitted.intent_id)
            except Exception:
                raise HireAdmissionError(
                    "provisioning submission failed after durable admission"
                ) from None
        return admitted

    def commit_effect_transition(
        self,
        intent_id: str,
        *,
        effect_id: str,
        effect_type: str,
        next_state: HireEffectState,
        metadata: Mapping[str, str] | None = None,
    ) -> DurableHireState:
        """Commit one effect fact without performing the external operation."""

        if not all(
            isinstance(value, str) and value
            for value in (intent_id, effect_id, effect_type)
        ):
            raise HireAdmissionError("effect identifiers are required")
        event_types = {
            HireEffectState.PREPARED: "hire.effect.prepared",
            HireEffectState.EXECUTING: "hire.effect.executing",
            HireEffectState.COMMITTED: "hire.effect.committed",
            HireEffectState.ACTION_REQUIRED: "hire.effect.action_required",
        }
        event_type = event_types.get(next_state)
        if event_type is None:
            raise HireAdmissionError("invalid effect transition")
        allowed_previous = {
            HireEffectState.PREPARED: {None, HireEffectState.PLANNED},
            HireEffectState.EXECUTING: {HireEffectState.PREPARED},
            HireEffectState.COMMITTED: {HireEffectState.EXECUTING},
            HireEffectState.ACTION_REQUIRED: {
                HireEffectState.PREPARED,
                HireEffectState.EXECUTING,
            },
        }
        metadata_value = dict(metadata or {})
        if not set(metadata_value).issubset(_EFFECT_METADATA_KEYS):
            raise HireAdmissionError("invalid effect metadata")
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not value
            for key, value in metadata_value.items()
        ):
            raise HireAdmissionError("invalid effect metadata")
        if "credential_ref" in metadata_value and "app_id" not in metadata_value:
            raise HireAdmissionError("invalid effect metadata")
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            self._synchronize_projection_to_journal_locked()
            current = self._hire_projection.get(intent_id)
            if current is None:
                raise HireAdmissionError("unknown hire intent")
            if current.effect_state(effect_id) not in allowed_previous[next_state]:
                raise HireAdmissionError("invalid effect transition")
            previous_type = dict(current.effect_types).get(effect_id)
            if previous_type not in (None, effect_type):
                raise HireAdmissionError("invalid effect transition")
            previous_metadata = dict(current.metadata_for(effect_id))
            if any(
                key in previous_metadata and previous_metadata[key] != value
                for key, value in metadata_value.items()
            ):
                raise HireAdmissionError("invalid effect metadata")
            last_frame = self._writer.get_last_frame()
            writer_sequence = 0 if last_frame is None else last_frame.sequence
            writer_hash = "" if last_frame is None else last_frame.frame_hash
            if (
                self._projection_state.cursor_sequence != writer_sequence
                or self._projection_state.cursor_hash != writer_hash
            ):
                raise HireAdmissionError("hire projection is stale")
            payload: dict[str, object] = {
                "effect_id": effect_id,
                "effect_type": effect_type,
            }
            if metadata_value:
                payload["metadata"] = metadata_value
            event = JournalEvent(
                event_type=event_type,
                aggregate_id=intent_id,
                payload=payload,
            )
            expected_versions = self._writer.get_aggregate_versions((intent_id,))
            result = self._writer.commit(
                (event,),
                expected_versions,
                expected_head_sequence=writer_sequence,
                expected_head_hash=writer_hash,
            )
            if result.state is not CommitState.ANCHORED:
                raise AnchorMismatchError("hire effect commit was not anchored")
            apply_frame(self._projection_state, result.frame)
            self._hire_projection = HireProjection.rebuild(self._writer.replay())
            updated = self._hire_projection.get(intent_id)
            if updated is None:
                raise HireAdmissionError("anchored hire effect did not replay")
            return updated

    def begin_activation_verification(
        self,
        challenge: VerificationChallenge,
    ) -> DurableHireState:
        """Persist a challenge only after exact Slash and Channel evidence."""
        if not isinstance(challenge, VerificationChallenge):
            raise HireAdmissionError("invalid verification challenge")
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            state = self._require_hire(challenge.hire_intent_id)
            slash_effect_id = self._latest_slash_effect_id(
                state,
                challenge.generation,
            )
            if slash_effect_id is None:
                raise HireAdmissionError("configuration evidence is incomplete")
            slash_metadata = dict(state.metadata_for(slash_effect_id))
            channel_effect_id = f"channel-start:{challenge.generation}"
            channel_metadata = dict(state.metadata_for(channel_effect_id))
            if (
                state.phase not in {HirePhase.CONFIGURING, HirePhase.VALIDATING}
                or state.effect_state(slash_effect_id) is not HireEffectState.COMMITTED
                or slash_metadata.get("slash_spec_hash")
                != challenge.expected_slash_spec_hash
                or slash_metadata.get("slash_observed_hash")
                != challenge.expected_slash_spec_hash
                or state.slash_verified_at <= 0
                or state.effect_state(channel_effect_id)
                is not HireEffectState.COMMITTED
                or channel_metadata.get("app_id") != state.app_id
                or channel_metadata.get("identity_app_id") != state.app_id
                or channel_metadata.get("generation") != str(challenge.generation)
                or not channel_metadata.get("connection_id")
                or state.channel_verified_at < state.slash_verified_at
            ):
                raise HireAdmissionError("configuration evidence is incomplete")
            if (
                challenge.tenant_key != state.tenant_key
                or challenge.app_id != state.app_id
                or challenge.agent_id != state.agent_id
                or challenge.requester_principal_id
                != state.requester_principal_id
                or challenge.requester_union_id != state.requester_union_id
            ):
                raise HireAdmissionError("verification challenge binding mismatch")
            if state.phase is HirePhase.CONFIGURING:
                state = self._commit_phase_transition_locked(
                    state,
                    HirePhase.VALIDATING,
                )
            state = self._commit_hire_event(
                JournalEvent(
                    event_type="hire.verification.challenge_issued",
                    aggregate_id=state.intent_id,
                    payload={
                        "tenant_key": challenge.tenant_key,
                        "app_id": challenge.app_id,
                        "agent_id": challenge.agent_id,
                        "generation": challenge.generation,
                        "requester_principal_id": challenge.requester_principal_id,
                        "requester_union_id": challenge.requester_union_id,
                        "expected_slash_spec_hash": challenge.expected_slash_spec_hash,
                        "nonce": challenge.nonce,
                        "issued_at": challenge.issued_at,
                        "expires_at": challenge.expires_at,
                    },
                )
            )
            return self._commit_phase_transition_locked(
                state,
                HirePhase.READY_PENDING_VERIFICATION,
            )

    def renew_activation_verification(
        self,
        challenge: VerificationChallenge,
    ) -> DurableHireState:
        """Replace an expiring unconsumed challenge without changing generation."""

        if not isinstance(challenge, VerificationChallenge):
            raise HireAdmissionError("invalid verification challenge")
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            state = self._require_hire(challenge.hire_intent_id)
            if (
                state.phase is not HirePhase.READY_PENDING_VERIFICATION
                or state.verification_consumed
                or challenge.nonce == state.verification_nonce
                or challenge.issued_at
                < state.verification_expires_at
                - ACTIVATION_CHALLENGE_RENEWAL_LEAD_SECONDS
                or challenge.expires_at <= state.verification_expires_at
                or challenge.tenant_key != state.tenant_key
                or challenge.app_id != state.app_id
                or challenge.agent_id != state.agent_id
                or challenge.generation != state.channel_generation
                or challenge.requester_principal_id
                != state.requester_principal_id
                or challenge.requester_union_id != state.requester_union_id
                or challenge.expected_slash_spec_hash != state.slash_spec_hash
            ):
                raise HireAdmissionError("verification challenge renewal rejected")
            return self._commit_hire_event_locked(
                JournalEvent(
                    event_type="hire.verification.challenge_issued",
                    aggregate_id=state.intent_id,
                    payload={
                        "tenant_key": challenge.tenant_key,
                        "app_id": challenge.app_id,
                        "agent_id": challenge.agent_id,
                        "generation": challenge.generation,
                        "requester_principal_id": challenge.requester_principal_id,
                        "requester_union_id": challenge.requester_union_id,
                        "expected_slash_spec_hash": challenge.expected_slash_spec_hash,
                        "nonce": challenge.nonce,
                        "issued_at": challenge.issued_at,
                        "expires_at": challenge.expires_at,
                    },
                )
            )

    def reissue_activation_verification_after_audit_collision(
        self,
        challenge: VerificationChallenge,
    ) -> DurableHireState:
        """Start a fresh window after a conflicting main-Bot audit fact."""

        if not isinstance(challenge, VerificationChallenge):
            raise HireAdmissionError("invalid verification challenge")
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            state = self._require_hire(challenge.hire_intent_id)
            if (
                state.phase is not HirePhase.READY_PENDING_VERIFICATION
                or state.verification_consumed
                or challenge.nonce == state.verification_nonce
                or challenge.issued_at <= state.verification_issued_at
                or challenge.expires_at <= challenge.issued_at
                or challenge.tenant_key != state.tenant_key
                or challenge.app_id != state.app_id
                or challenge.agent_id != state.agent_id
                or challenge.generation != state.channel_generation
                or challenge.requester_principal_id
                != state.requester_principal_id
                or challenge.requester_union_id != state.requester_union_id
                or challenge.expected_slash_spec_hash != state.slash_spec_hash
            ):
                raise HireAdmissionError("verification challenge reissue rejected")
            return self._commit_hire_event_locked(
                JournalEvent(
                    event_type="hire.verification.challenge_reissued",
                    aggregate_id=state.intent_id,
                    payload={
                        "tenant_key": challenge.tenant_key,
                        "app_id": challenge.app_id,
                        "agent_id": challenge.agent_id,
                        "generation": challenge.generation,
                        "requester_principal_id": challenge.requester_principal_id,
                        "requester_union_id": challenge.requester_union_id,
                        "expected_slash_spec_hash": challenge.expected_slash_spec_hash,
                        "nonce": challenge.nonce,
                        "issued_at": challenge.issued_at,
                        "expires_at": challenge.expires_at,
                    },
                )
            )

    def select_slash_reconcile_effect(
        self,
        intent_id: str,
        *,
        generation: int,
        force_refresh: bool,
    ) -> str:
        """Select a durable, resumable Slash attempt for one Channel generation."""
        if isinstance(generation, bool) or generation <= 0:
            raise HireAdmissionError("invalid Channel generation")
        with self._mutex:
            state = self._require_hire(intent_id)
            attempts = self._slash_effect_attempts(state, generation)
            if not attempts:
                return f"slash-reconcile:{generation}:1"
            attempt, effect_id = attempts[-1]
            if (
                force_refresh
                and state.effect_state(effect_id) is HireEffectState.COMMITTED
            ):
                return f"slash-reconcile:{generation}:{attempt + 1}"
            return effect_id

    def consume_once(
        self,
        challenge: VerificationChallenge,
        *,
        consumed_at: float,
    ) -> bool:
        """Journal-backed atomic nonce consumer used by VerificationRouter."""
        if not isinstance(challenge, VerificationChallenge):
            return False
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            state = self._hire_projection.get(challenge.hire_intent_id)
            if (
                state is None
                or state.phase is not HirePhase.READY_PENDING_VERIFICATION
                or state.verification_consumed
                or state.verification_nonce != challenge.nonce
                or state.channel_generation != challenge.generation
                or state.tenant_key != challenge.tenant_key
                or state.app_id != challenge.app_id
                or state.agent_id != challenge.agent_id
                or isinstance(consumed_at, bool)
                or not isinstance(consumed_at, (int, float))
                or not (
                    state.verification_issued_at
                    <= float(consumed_at)
                    <= state.verification_expires_at
                )
            ):
                return False
            self._commit_hire_event_locked(
                JournalEvent(
                    event_type="hire.verification.nonce_consumed",
                    aggregate_id=state.intent_id,
                    payload={
                        "nonce": challenge.nonce,
                        "consumed_at": float(consumed_at),
                    },
                )
            )
            return True

    def commit_activation(
        self,
        decision: VerificationDecision,
        *,
        reply_effect_id: str = "",
        reply_effect_metadata: Mapping[str, str] | None = None,
    ) -> DurableHireState:
        """Serialize a READY decision and only then enter ACTIVE."""
        if (
            not isinstance(decision, VerificationDecision)
            or decision.outcome is not VerificationOutcome.READY
            or decision.activation_evidence is None
        ):
            raise HireAdmissionError("activation decision is not ready")
        evidence = decision.activation_evidence
        coordinates = evidence.coordinates
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            state = self._require_hire(coordinates.hire_intent_id)
            if (
                state.phase is not HirePhase.READY_PENDING_VERIFICATION
                or coordinates.tenant_key != state.tenant_key
                or coordinates.app_id != state.app_id
                or coordinates.agent_id != state.agent_id
                or coordinates.generation != state.channel_generation
                or coordinates.nonce != state.verification_nonce
                or state.verification_consumed
                or evidence.slash_spec_hash != state.slash_spec_hash
                or evidence.channel_connection_id != state.channel_connection_id
                or evidence.reply_app_id != state.app_id
                or evidence.main_bot_send_count != 0
                or evidence.sender_union_id != state.requester_union_id
                or any(
                    not isinstance(value, str) or not value
                    for value in (
                        evidence.ingress_event_id,
                        evidence.ingress_message_id,
                        evidence.employee_send_request_id,
                    )
                )
                or isinstance(evidence.verified_at, bool)
                or not isinstance(evidence.verified_at, (int, float))
                or not math.isfinite(float(evidence.verified_at))
                or not (
                    state.verification_issued_at
                    <= float(evidence.verified_at)
                    <= state.verification_expires_at
                )
            ):
                raise HireAdmissionError("activation evidence binding mismatch")
            activation_event = JournalEvent(
                event_type="hire.activation.verified",
                aggregate_id=state.intent_id,
                payload={
                    "tenant_key": coordinates.tenant_key,
                    "app_id": coordinates.app_id,
                    "agent_id": coordinates.agent_id,
                    "generation": coordinates.generation,
                    "nonce": coordinates.nonce,
                    "slash_spec_hash": evidence.slash_spec_hash,
                    "channel_connection_id": evidence.channel_connection_id,
                    "ingress_event_id": evidence.ingress_event_id,
                    "ingress_message_id": evidence.ingress_message_id,
                    "employee_send_request_id": evidence.employee_send_request_id,
                    "reply_app_id": evidence.reply_app_id,
                    "main_bot_send_count": evidence.main_bot_send_count,
                    "sender_union_id": evidence.sender_union_id,
                    "verified_at": evidence.verified_at,
                },
            )
            reply_metadata = dict(reply_effect_metadata or {})
            expected_effect_id = (
                f"verification-status-reply:{evidence.ingress_event_id}"
            )
            if (
                reply_effect_id != expected_effect_id
                or state.effect_state(reply_effect_id)
                is not HireEffectState.EXECUTING
                or dict(state.effect_types).get(reply_effect_id)
                != "employee_status_reply"
                or not reply_metadata
                or not set(reply_metadata).issubset(_EFFECT_METADATA_KEYS)
                or any(
                    not isinstance(key, str)
                    or not isinstance(value, str)
                    or not value
                    for key, value in reply_metadata.items()
                )
                or reply_metadata.get("send_request_id")
                != evidence.employee_send_request_id
                or reply_metadata.get("ingress_event_id")
                != evidence.ingress_event_id
                or reply_metadata.get("reply_app_id") != evidence.reply_app_id
                or reply_metadata.get("main_bot_send_count") != "0"
            ):
                raise HireAdmissionError("activation reply effect binding mismatch")
            if any(
                effect_state
                in {HireEffectState.PREPARED, HireEffectState.EXECUTING}
                and effect_id != reply_effect_id
                for effect_id, effect_state in state.effects
            ):
                raise HireAdmissionError("activation has unresolved effects")
            events = (
                JournalEvent(
                    event_type="hire.effect.committed",
                    aggregate_id=state.intent_id,
                    payload={
                        "effect_id": reply_effect_id,
                        "effect_type": "employee_status_reply",
                        "metadata": reply_metadata,
                    },
                ),
                JournalEvent(
                    event_type="hire.verification.nonce_consumed",
                    aggregate_id=state.intent_id,
                    payload={
                        "nonce": coordinates.nonce,
                        "consumed_at": evidence.verified_at,
                    },
                ),
                activation_event,
                JournalEvent(
                    event_type="employee.state_changed",
                    aggregate_id=state.agent_id,
                    payload={"state": HirePhase.ACTIVE.value},
                ),
            )
            self._prepare_workspace_for_activation(state)
            self._synchronize_projection_to_journal_locked()
            state = self._require_hire(state.intent_id)
            last_frame = self._writer.get_last_frame()
            writer_sequence = 0 if last_frame is None else last_frame.sequence
            writer_hash = "" if last_frame is None else last_frame.frame_hash
            expected_versions = self._writer.get_aggregate_versions(
                (state.intent_id, state.agent_id)
            )
            try:
                validate_workforce_events(self._projection_state, (events[-1],))
            except ProjectionError as exc:
                raise HireAdmissionError(
                    "activation workforce transition rejected"
                ) from exc
            result = self._writer.commit(
                events,
                expected_versions,
                expected_head_sequence=writer_sequence,
                expected_head_hash=writer_hash,
            )
            if result.state is not CommitState.ANCHORED:
                raise AnchorMismatchError("atomic activation commit was not anchored")
            apply_frame(self._projection_state, result.frame)
            self._hire_projection = HireProjection.rebuild(self._writer.replay())
            activated = self._require_hire(state.intent_id)
            if activated.phase is not HirePhase.ACTIVE:
                raise HireAdmissionError("atomic activation did not enter ACTIVE")
            return activated

    def commit_automatic_activation(
        self,
        intent_id: str,
        *,
        activated_at: float,
    ) -> DurableHireState:
        """Enter ACTIVE from anchored local Slash and Channel READY evidence."""

        if (
            isinstance(activated_at, bool)
            or not isinstance(activated_at, (int, float))
            or not math.isfinite(float(activated_at))
        ):
            raise HireAdmissionError("automatic activation time is invalid")
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            self._synchronize_projection_to_journal_locked()
            state = self._require_hire(intent_id)
            slash_effect_id = self._latest_slash_effect_id(
                state,
                state.channel_generation,
            )
            channel_effect_id = f"channel-start:{state.channel_generation}"
            slash_metadata = (
                dict(state.metadata_for(slash_effect_id))
                if slash_effect_id is not None
                else {}
            )
            channel_metadata = dict(state.metadata_for(channel_effect_id))
            if (
                state.phase is not HirePhase.READY_PENDING_VERIFICATION
                or state.verification_consumed
                or not state.credential_ref
                or not state.requester_principal_id
                or slash_effect_id is None
                or state.effect_state(slash_effect_id) is not HireEffectState.COMMITTED
                or slash_metadata.get("slash_spec_hash") != state.slash_spec_hash
                or slash_metadata.get("slash_observed_hash") != state.slash_spec_hash
                or state.effect_state(channel_effect_id) is not HireEffectState.COMMITTED
                or channel_metadata.get("app_id") != state.app_id
                or channel_metadata.get("identity_app_id") != state.app_id
                or channel_metadata.get("generation") != str(state.channel_generation)
                or channel_metadata.get("connection_id") != state.channel_connection_id
                or state.channel_identity_app_id != state.app_id
                or not state.channel_connection_id
                or state.slash_verified_at <= 0
                or state.channel_verified_at < state.slash_verified_at
                or float(activated_at) < state.channel_verified_at
                or any(
                    effect_state in {
                        HireEffectState.PREPARED,
                        HireEffectState.EXECUTING,
                    }
                    for _effect_id, effect_state in state.effects
                )
            ):
                raise HireAdmissionError("automatic activation evidence is incomplete")
            events = (
                JournalEvent(
                    event_type="hire.activation.automatic",
                    aggregate_id=state.intent_id,
                    payload={
                        "tenant_key": state.tenant_key,
                        "app_id": state.app_id,
                        "agent_id": state.agent_id,
                        "generation": state.channel_generation,
                        "slash_spec_hash": state.slash_spec_hash,
                        "channel_connection_id": state.channel_connection_id,
                        "requester_principal_id": state.requester_principal_id,
                        "requester_union_id": state.requester_union_id,
                        "source": "channel_ready",
                        "activated_at": float(activated_at),
                    },
                ),
                JournalEvent(
                    event_type="employee.state_changed",
                    aggregate_id=state.agent_id,
                    payload={"state": HirePhase.ACTIVE.value},
                ),
            )
            self._prepare_workspace_for_activation(state)
            last_frame = self._writer.get_last_frame()
            writer_sequence = 0 if last_frame is None else last_frame.sequence
            writer_hash = "" if last_frame is None else last_frame.frame_hash
            expected_versions = self._writer.get_aggregate_versions(
                (state.intent_id, state.agent_id)
            )
            try:
                validate_workforce_events(self._projection_state, (events[-1],))
            except ProjectionError as exc:
                raise HireAdmissionError(
                    "automatic activation workforce transition rejected"
                ) from exc
            result = self._writer.commit(
                events,
                expected_versions,
                expected_head_sequence=writer_sequence,
                expected_head_hash=writer_hash,
            )
            if result.state is not CommitState.ANCHORED:
                raise AnchorMismatchError(
                    "automatic activation commit was not anchored"
                )
            apply_frame(self._projection_state, result.frame)
            self._hire_projection = HireProjection.rebuild(self._writer.replay())
            activated = self._require_hire(state.intent_id)
            if activated.phase is not HirePhase.ACTIVE:
                raise HireAdmissionError("automatic activation did not enter ACTIVE")
            return activated

    def _commit_hire_event(self, event: JournalEvent) -> DurableHireState:
        """Commit one already-validated hire-only fact and advance all cursors."""
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            return self._commit_hire_event_locked(event)

    def _commit_hire_event_locked(self, event: JournalEvent) -> DurableHireState:
        """Commit after the caller acquired workforce, hire, then Journal."""

        self._synchronize_projection_to_journal_locked()
        last_frame = self._writer.get_last_frame()
        writer_sequence = 0 if last_frame is None else last_frame.sequence
        writer_hash = "" if last_frame is None else last_frame.frame_hash
        if (
            self._projection_state.cursor_sequence != writer_sequence
            or self._projection_state.cursor_hash != writer_hash
        ):
            raise HireAdmissionError("hire projection is stale")
        expected_versions = self._writer.get_aggregate_versions(
            (event.aggregate_id,)
        )
        result = self._writer.commit(
            (event,),
            expected_versions,
            expected_head_sequence=writer_sequence,
            expected_head_hash=writer_hash,
        )
        if result.state is not CommitState.ANCHORED:
            raise AnchorMismatchError("hire event commit was not anchored")
        apply_frame(self._projection_state, result.frame)
        self._hire_projection = HireProjection.rebuild(self._writer.replay())
        return self._require_hire(event.aggregate_id)

    def _synchronize_projection_to_journal_locked(self) -> None:
        """Advance workforce/hire views across frames written by other domains."""
        last = self._writer.get_last_frame()
        sequence = 0 if last is None else last.sequence
        logical_hash = "" if last is None else last.frame_hash
        if (
            self._projection_state.cursor_sequence == sequence
            and self._projection_state.cursor_hash == logical_hash
        ):
            return
        frames = tuple(self._writer.replay())
        self._projection_state = ProjectionRepository().rebuild(iter(frames))
        self._hire_projection = HireProjection.rebuild(frames)
        self._manifest_reauthorizations = rebuild_manifest_reauthorizations(frames)

    async def run_provisioning(self, intent_id: str) -> DurableHireState:
        """Run one deduplicated asynchronous provisioning activity."""

        if not isinstance(intent_id, str) or not intent_id:
            raise HireAdmissionError("intent_id is required")
        with self._mutex:
            if self._closed:
                raise HireAdmissionError("closed")
            if self._hire_projection.get(intent_id) is None:
                raise HireAdmissionError("unknown hire intent")
            task = self._activities.get(intent_id)
            if task is None:
                task = asyncio.create_task(self._run_provisioning_activity(intent_id))
                self._activities[intent_id] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                with self._mutex:
                    if self._activities.get(intent_id) is task:
                        self._activities.pop(intent_id, None)

    async def _run_provisioning_activity(
        self,
        intent_id: str,
    ) -> DurableHireState:
        state = self._require_hire(intent_id)
        if state.phase is HirePhase.CONFIGURING:
            return state
        if state.phase is HirePhase.ACTION_REQUIRED:
            raise HireAdmissionError("hire requires manual action")
        if self._registrar is None or self._credential_vault is None:
            raise HireAdmissionError("hire provisioning dependencies unavailable")

        register_state = state.effect_state("register-app")
        if register_state is None:
            state = self.commit_effect_transition(
                intent_id,
                effect_id="register-app",
                effect_type="app_registration",
                next_state=HireEffectState.PREPARED,
            )
            register_state = HireEffectState.PREPARED
        if register_state is not HireEffectState.PREPARED:
            await self._mark_action_required(
                intent_id,
                effect_id="register-app",
                effect_type="app_registration",
            )
            raise HireAdmissionError("app registration outcome requires manual action")
        state = self.commit_effect_transition(
            intent_id,
            effect_id="register-app",
            effect_type="app_registration",
            next_state=HireEffectState.EXECUTING,
        )

        bridge = AsyncCallbackBridge()
        link_callback = bridge.callback(self._on_registration_link, state)
        status_callback = bridge.callback(self._on_registration_status, state)
        registration_result: RegistrationResult | None = None
        registration_error: Exception | None = None
        try:
            registration_result = await self._registrar.register(
                RegistrationRequest(
                    name=state.employee_name,
                    description=state.role or state.persona or "GhostAP employee",
                    existing_app_id=getattr(state, "existing_app_id", "") or "",
                ),
                on_link=link_callback,
                on_status=status_callback,
            )
        except Exception as exc:
            registration_error = exc
        try:
            await bridge.drain()
        except Exception as exc:
            registration_error = exc
        if registration_error is not None or registration_result is None:
            await self._mark_action_required(
                intent_id,
                effect_id="register-app",
                effect_type="app_registration",
            )
            raise HireAdmissionError("app registration outcome requires manual action") from None

        app_id = registration_result.app_id
        app_secret = registration_result.app_secret
        try:
            state = self._commit_registered_app(intent_id, app_id)
        except HireAdmissionError:
            await self._mark_action_required(
                intent_id,
                effect_id="register-app",
                effect_type="app_registration",
            )
            raise HireAdmissionError(
                "app registration outcome requires manual action"
            ) from None
        state = self.commit_effect_transition(
            intent_id,
            effect_id="store-credential",
            effect_type="credential_vault_put",
            next_state=HireEffectState.PREPARED,
            metadata={"app_id": app_id},
        )
        state = self.commit_effect_transition(
            intent_id,
            effect_id="store-credential",
            effect_type="credential_vault_put",
            next_state=HireEffectState.EXECUTING,
            metadata={"app_id": app_id},
        )
        receipt: object | None = None
        for _attempt in range(2):
            try:
                receipt = await asyncio.to_thread(
                    self._credential_vault.put,
                    state.agent_id,
                    app_id,
                    app_secret,
                    state.intent_id,
                    state.attempt_id,
                )
                break
            except Exception:
                continue
        del app_secret, registration_result
        if receipt is None:
            await self._mark_action_required(
                intent_id,
                effect_id="store-credential",
                effect_type="credential_vault_put",
            )
            raise HireAdmissionError("credential storage requires manual action")
        try:
            credential_ref = self._validate_receipt(state, app_id, receipt)
        except HireAdmissionError:
            await self._mark_action_required(
                intent_id,
                effect_id="store-credential",
                effect_type="credential_vault_put",
            )
            raise HireAdmissionError("credential storage requires manual action") from None
        state = self.commit_effect_transition(
            intent_id,
            effect_id="store-credential",
            effect_type="credential_vault_put",
            next_state=HireEffectState.COMMITTED,
            metadata={"app_id": app_id, "credential_ref": credential_ref},
        )
        return self._bind_principal(state, app_id, credential_ref)

    async def _mark_action_required(
        self,
        intent_id: str,
        *,
        effect_id: str,
        effect_type: str,
    ) -> DurableHireState:
        return self._mark_action_required_sync(
            self._require_hire(intent_id),
            effect_id=effect_id,
            effect_type=effect_type,
        )

    def mark_recovery_action_required(
        self,
        intent_id: str,
        *,
        error_code: str,
    ) -> DurableHireState:
        """Terminalize every unresolved hire effect after bounded recovery.

        Recovery must not keep global hire admission closed forever because one
        employee cannot be reconciled.  All unresolved effects are disposed
        durably before the hire itself enters ``ACTION_REQUIRED``.
        """

        if not isinstance(error_code, str) or not error_code:
            raise HireAdmissionError("recovery error code is required")
        state = self._require_hire(intent_id)
        unresolved = [
            (effect_id, dict(state.effect_types).get(effect_id, ""))
            for effect_id, effect_state in state.effects
            if effect_state in {HireEffectState.PREPARED, HireEffectState.EXECUTING}
        ]
        for effect_id, effect_type in unresolved:
            if not effect_type:
                raise HireAdmissionError("unresolved hire effect type is missing")
            state = self.commit_effect_transition(
                intent_id,
                effect_id=effect_id,
                effect_type=effect_type,
                next_state=HireEffectState.ACTION_REQUIRED,
                metadata={
                    **dict(state.metadata_for(effect_id)),
                    "error_code": error_code,
                },
            )
        if state.phase is not HirePhase.ACTION_REQUIRED:
            state = self._commit_phase_transition(state, HirePhase.ACTION_REQUIRED)
        return state

    def _commit_phase_transition(
        self,
        state: DurableHireState,
        phase: HirePhase,
    ) -> DurableHireState:
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            return self._commit_phase_transition_locked(state, phase)

    def _commit_phase_transition_locked(
        self,
        state: DurableHireState,
        phase: HirePhase,
    ) -> DurableHireState:
        """Transition after the caller acquired workforce, hire, then Journal."""

        self._synchronize_projection_to_journal_locked()
        state = self._require_hire(state.intent_id)
        if phase in {
            HirePhase.ACTIVE,
            HirePhase.ACTION_REQUIRED,
            HirePhase.ARCHIVED,
        } and any(
            effect_state in {HireEffectState.PREPARED, HireEffectState.EXECUTING}
            for _effect_id, effect_state in state.effects
        ):
            raise HireAdmissionError("terminal hire phase has unresolved effects")
        if state.phase is phase:
            return state
        if phase is HirePhase.ACTIVE:
            self._prepare_workspace_for_activation(state)
        try:
            commit_workforce_events_unlocked(
                self._writer,
                self._projection_state,
                (
                    JournalEvent(
                        event_type="employee.state_changed",
                        aggregate_id=state.agent_id,
                        payload={"state": phase.value},
                    ),
                ),
            )
        except ProjectionError as exc:
            raise HireAdmissionError("hire phase transition rejected") from exc
        self._hire_projection = HireProjection.rebuild(self._writer.replay())
        return self._require_hire(state.intent_id)

    def _prepare_workspace_for_activation(self, state: DurableHireState) -> None:
        projector = self._workspace_projector
        if projector is None:
            return
        try:
            snapshot = projector.rebuild(state.tenant_key, state.agent_id)
            projector.verify(snapshot)
        except Exception:
            raise HireAdmissionError("employee workspace projection failed") from None

    def _has_exact_phase_only_channel_shutdown_history(
        self,
        state: DurableHireState,
        frames: tuple[TransactionFrame, ...],
    ) -> bool:
        generation = state.channel_generation
        channel_effect_id = f"channel-start:{generation}"
        channel_metadata = dict(state.metadata_for(channel_effect_id))
        slash_effect_id = self._latest_slash_effect_id(state, generation)
        slash_metadata = (
            dict(state.metadata_for(slash_effect_id))
            if slash_effect_id is not None
            else {}
        )
        if (
            state.phase is not HirePhase.ACTION_REQUIRED
            or generation <= 0
            or not state.app_id
            or not state.credential_ref
            or not state.verification_nonce
            or state.verification_consumed is not True
            or state.activation_source != "channel_ready"
            or state.activation_verified_at <= 0
            or not math.isfinite(state.activation_verified_at)
            or not state.slash_spec_hash
            or state.slash_spec_hash != state.slash_observed_hash
            or state.channel_identity_app_id != state.app_id
            or not state.channel_connection_id
            or state.effect_state(channel_effect_id)
            is not HireEffectState.COMMITTED
            or channel_metadata.get("app_id") != state.app_id
            or channel_metadata.get("identity_app_id") != state.app_id
            or channel_metadata.get("generation") != str(generation)
            or channel_metadata.get("connection_id")
            != state.channel_connection_id
            or slash_effect_id is None
            or state.effect_state(slash_effect_id) is not HireEffectState.COMMITTED
            or slash_metadata.get("slash_spec_hash") != state.slash_spec_hash
            or slash_metadata.get("slash_observed_hash") != state.slash_spec_hash
            or any(
                effect_state is not HireEffectState.COMMITTED
                for _effect_id, effect_state in state.effects
            )
        ):
            return False

        related_frames = tuple(
            frame
            for frame in frames
            if any(
                event.aggregate_id in {state.intent_id, state.agent_id}
                for event in frame.events
            )
        )
        if len(related_frames) < 5:
            return False
        crashed, validating, pending, action_required = related_frames[-4:]
        activation_payload = {
            "tenant_key": state.tenant_key,
            "app_id": state.app_id,
            "agent_id": state.agent_id,
            "generation": generation,
            "slash_spec_hash": state.slash_spec_hash,
            "channel_connection_id": state.channel_connection_id,
            "requester_principal_id": state.requester_principal_id,
            "requester_union_id": state.requester_union_id,
            "source": "channel_ready",
            "activated_at": state.activation_verified_at,
        }
        activation_frame = next(
            (
                frame
                for frame in reversed(related_frames[:-4])
                if _frame_matches(
                    frame,
                    (
                        (
                            "hire.activation.automatic",
                            state.intent_id,
                            activation_payload,
                        ),
                        (
                            "employee.state_changed",
                            state.agent_id,
                            {"state": HirePhase.ACTIVE.value},
                        ),
                    ),
                )
            ),
            None,
        )
        old_writer_epoch = crashed.writer_epoch
        restart_writer_epoch = pending.writer_epoch
        return (
            activation_frame is not None
            and state.last_sequence == action_required.sequence
            and validating.writer_epoch == old_writer_epoch
            and restart_writer_epoch != old_writer_epoch
            # A second crash can happen after the restart wrote PENDING but
            # before it disposes the consumed nonce as ACTION_REQUIRED.  The
            # exact four-frame chain is still phase-only evidence even when
            # those two recovery transitions have distinct writer epochs.
            and action_required.writer_epoch != old_writer_epoch
            and _frame_matches(
                crashed,
                (
                    (
                        "hire.channel.crashed",
                        state.intent_id,
                        {"generation": generation},
                    ),
                ),
            )
            and _frame_matches(
                validating,
                (
                    (
                        "employee.state_changed",
                        state.agent_id,
                        {"state": HirePhase.VALIDATING.value},
                    ),
                ),
            )
            and _frame_matches(
                pending,
                (
                    (
                        "employee.state_changed",
                        state.agent_id,
                        {"state": HirePhase.READY_PENDING_VERIFICATION.value},
                    ),
                ),
            )
            and _frame_matches(
                action_required,
                (
                    (
                        "employee.state_changed",
                        state.agent_id,
                        {"state": HirePhase.ACTION_REQUIRED.value},
                    ),
                ),
            )
        )

    def _reopen_exact_phase_only_channel_shutdown(
        self,
        state: DurableHireState,
    ) -> DurableHireState:
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            self._synchronize_projection_to_journal_locked()
            state = self._require_hire(state.intent_id)
            frames = tuple(self._writer.replay())
            if not self._has_exact_phase_only_channel_shutdown_history(
                state,
                frames,
            ):
                return state
            events = (
                JournalEvent(
                    event_type="hire.channel.phase_only_recovery",
                    aggregate_id=state.intent_id,
                    payload={"generation": state.channel_generation},
                ),
                JournalEvent(
                    event_type="employee.state_changed",
                    aggregate_id=state.agent_id,
                    payload={"state": HirePhase.VALIDATING.value},
                ),
            )
            last_frame = self._writer.get_last_frame()
            writer_sequence = 0 if last_frame is None else last_frame.sequence
            writer_hash = "" if last_frame is None else last_frame.frame_hash
            expected_versions = self._writer.get_aggregate_versions(
                (state.intent_id, state.agent_id)
            )
            try:
                validate_workforce_events(self._projection_state, (events[-1],))
            except ProjectionError as exc:
                raise HireAdmissionError(
                    "phase-only channel recovery transition rejected"
                ) from exc
            result = self._writer.commit(
                events,
                expected_versions,
                expected_head_sequence=writer_sequence,
                expected_head_hash=writer_hash,
            )
            if result.state is not CommitState.ANCHORED:
                raise AnchorMismatchError(
                    "phase-only channel recovery was not anchored"
                )
            apply_frame(self._projection_state, result.frame)
            self._hire_projection = HireProjection.rebuild(self._writer.replay())
            reopened = self._require_hire(state.intent_id)
            if reopened.phase is not HirePhase.VALIDATING:
                raise HireAdmissionError(
                    "phase-only channel recovery did not enter VALIDATING"
                )
            return reopened

    def _has_pending_phase_only_channel_recovery(
        self,
        state: DurableHireState,
    ) -> bool:
        recovery_sequence = 0
        recovery_generation = 0
        fresh_challenge_sequence = 0
        for frame in self._writer.replay():
            for event in frame.events:
                if (
                    event.aggregate_id == state.intent_id
                    and event.event_type == "hire.channel.phase_only_recovery"
                ):
                    generation = event.payload.get("generation")
                    if isinstance(generation, bool) or not isinstance(
                        generation,
                        int,
                    ):
                        continue
                    recovery_sequence = frame.sequence
                    recovery_generation = generation
                    fresh_challenge_sequence = 0
                elif (
                    recovery_sequence > 0
                    and frame.sequence > recovery_sequence
                    and event.aggregate_id == state.intent_id
                    and event.event_type
                    in {
                        "hire.verification.challenge_issued",
                        "hire.verification.challenge_reissued",
                    }
                ):
                    generation = event.payload.get("generation")
                    if (
                        not isinstance(generation, bool)
                        and isinstance(generation, int)
                        and generation > recovery_generation
                    ):
                        fresh_challenge_sequence = frame.sequence
        return recovery_sequence > fresh_challenge_sequence

    def _bind_principal(
        self,
        state: DurableHireState,
        app_id: str,
        credential_ref: str,
    ) -> DurableHireState:
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            self._synchronize_projection_to_journal_locked()
            state = self._require_hire(state.intent_id)
            existing = self._projection_state.bot_principals.get(
                state.bot_principal_id
            )
            if existing is not None:
                if (
                    existing.agent_id != state.agent_id
                    or existing.app_id != app_id
                    or existing.credential_ref != credential_ref
                ):
                    raise HireAdmissionError("bot principal binding conflict")
                return self._require_hire(state.intent_id)
            manifest = current_registration_manifest()
            manifest_hash = manifest.fingerprint()
            events = (
                JournalEvent(
                    event_type="employee.bot_principal_bound",
                    aggregate_id=state.agent_id,
                    payload={
                        "agent_id": state.agent_id,
                        "bot_principal_id": state.bot_principal_id,
                    },
                ),
                JournalEvent(
                    event_type="bot_principal.bound",
                    aggregate_id=state.bot_principal_id,
                    payload={
                        "bot_principal_id": state.bot_principal_id,
                        "tenant_key": state.tenant_key,
                        "agent_id": state.agent_id,
                        "app_id": app_id,
                        "credential_ref": credential_ref,
                        "scopes": list(manifest.tenant_scopes),
                        "desired_manifest_hash": manifest_hash,
                        "observed_manifest_hash": "",
                    },
                ),
                JournalEvent(
                    event_type="employee.state_changed",
                    aggregate_id=state.agent_id,
                    payload={"state": HirePhase.CONFIGURING.value},
                ),
            )
            try:
                commit_workforce_events_unlocked(
                    self._writer,
                    self._projection_state,
                    events,
                )
            except ProjectionError as exc:
                raise HireAdmissionError(
                    "bot principal binding rejected"
                ) from exc
            self._hire_projection = HireProjection.rebuild(
                self._writer.replay()
            )
            return self._require_hire(state.intent_id)

    @staticmethod
    def _validate_receipt(
        state: DurableHireState,
        app_id: str,
        receipt: object,
    ) -> str:
        expected = {
            "agent_id": state.agent_id,
            "app_id": app_id,
            "hire_intent_id": state.intent_id,
            "attempt_id": state.attempt_id,
        }
        if any(getattr(receipt, key, None) != value for key, value in expected.items()):
            raise HireAdmissionError("credential receipt identity mismatch")
        credential_ref = getattr(receipt, "credential_ref", None)
        if not isinstance(credential_ref, str) or not credential_ref:
            raise HireAdmissionError("credential receipt is invalid")
        return credential_ref

    def _reconcile_recovered_hires(self) -> None:
        for state in tuple(self._hire_projection.states.values()):
            effect_types = dict(state.effect_types)
            pending_group_replies = [
                effect_id
                for effect_id, effect_state in state.effects
                if effect_state is HireEffectState.EXECUTING
                and effect_types.get(effect_id)
                == "employee_activation_required_reply"
            ]
            for effect_id in pending_group_replies:
                # EXECUTING belongs to an old Channel generation after process
                # recovery, so its external outcome cannot be proven. Dispose
                # only the effect; PREPARED is still safe for composition to
                # dispatch on the replacement Channel.
                state = self.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="employee_activation_required_reply",
                    next_state=HireEffectState.ACTION_REQUIRED,
                    metadata={
                        **dict(state.metadata_for(effect_id)),
                        "error_code": (
                            "activation_required_reply_outcome_unknown"
                        ),
                    },
                )
            if state.phase is HirePhase.ACTION_REQUIRED:
                state = self._reopen_exact_phase_only_channel_shutdown(state)
                # ACTION_REQUIRED is otherwise terminal.  A guarded reopen must
                # remain VALIDATING until composition launches a fresh,
                # generation-fenced Channel configuration.
                continue
            if state.phase is HirePhase.VALIDATING and state.verification_nonce:
                effect_types = dict(state.effect_types)
                if any(
                    effect_state
                    in {HireEffectState.PREPARED, HireEffectState.EXECUTING}
                    and effect_types.get(effect_id) == "employee_status_reply"
                    for effect_id, effect_state in state.effects
                ):
                    # The ingress replay owns this anchored phase-status reply.
                    # Advancing first would reinterpret the same /status as an
                    # activation attempt after restart.
                    continue
                if (
                    state.verification_consumed
                    and self._has_pending_phase_only_channel_recovery(state)
                ):
                    continue
                self._commit_phase_transition(
                    state,
                    HirePhase.READY_PENDING_VERIFICATION,
                )
                continue
            if (
                state.phase is HirePhase.READY_PENDING_VERIFICATION
                and state.activation_ingress_event_id
            ):
                self._commit_phase_transition(state, HirePhase.ACTIVE)
                continue
            if (
                state.phase is HirePhase.READY_PENDING_VERIFICATION
                and state.verification_consumed
            ):
                effect_types = dict(state.effect_types)
                for effect_id, effect_state in state.effects:
                    if effect_state not in {
                        HireEffectState.PREPARED,
                        HireEffectState.EXECUTING,
                    }:
                        continue
                    effect_type = effect_types.get(effect_id)
                    if not effect_type:
                        raise HireAdmissionError(
                            "consumed activation effect type is missing"
                        )
                    state = self.commit_effect_transition(
                        state.intent_id,
                        effect_id=effect_id,
                        effect_type=effect_type,
                        next_state=HireEffectState.ACTION_REQUIRED,
                        metadata={
                            **dict(state.metadata_for(effect_id)),
                            "error_code": "activation_nonce_consumed_without_commit",
                        },
                    )
                state = self._commit_phase_transition(
                    state,
                    HirePhase.ACTION_REQUIRED,
                )
                self._reopen_exact_phase_only_channel_shutdown(state)
                continue
            if state.phase is HirePhase.READY_PENDING_VERIFICATION:
                effect_types = dict(state.effect_types)
                reply_effects = [
                    (effect_id, effect_state)
                    for effect_id, effect_state in state.effects
                    if effect_types.get(effect_id) == "employee_status_reply"
                ]
                if reply_effects:
                    effect_id, effect_state = reply_effects[-1]
                    if effect_state in {
                        HireEffectState.PREPARED,
                        HireEffectState.EXECUTING,
                    }:
                        if dict(state.metadata_for(effect_id)).get(
                            "error_code", ""
                        ).startswith("phase_status:"):
                            continue
                        self.commit_effect_transition(
                            state.intent_id,
                            effect_id=effect_id,
                            effect_type="employee_status_reply",
                            next_state=HireEffectState.ACTION_REQUIRED,
                            metadata={
                                **dict(state.metadata_for(effect_id)),
                                "error_code": "activation_reply_outcome_unknown",
                            },
                        )
                    continue
            if state.phase in {
                HirePhase.CONFIGURING,
                HirePhase.VALIDATING,
                HirePhase.READY_PENDING_VERIFICATION,
                HirePhase.ACTIVE,
                HirePhase.ARCHIVED,
            }:
                continue
            register_state = state.effect_state("register-app")
            if register_state is HireEffectState.ACTION_REQUIRED:
                if state.phase is not HirePhase.ACTION_REQUIRED:
                    self._commit_phase_transition(state, HirePhase.ACTION_REQUIRED)
                continue
            if register_state is HireEffectState.EXECUTING and not state.app_id:
                self._mark_action_required_sync(
                    state,
                    effect_id="register-app",
                    effect_type="app_registration",
                )
                continue
            if not state.app_id:
                continue
            vault_state = state.effect_state("store-credential")
            if state.credential_ref:
                self._bind_principal(state, state.app_id, state.credential_ref)
                continue
            receipt = self._matching_orphan_receipt(state)
            if receipt is None:
                self._mark_action_required_sync(
                    state,
                    effect_id=(
                        "store-credential"
                        if vault_state in {
                            HireEffectState.PREPARED,
                            HireEffectState.EXECUTING,
                        }
                        else "register-app"
                    ),
                    effect_type=(
                        "credential_vault_put"
                        if vault_state in {
                            HireEffectState.PREPARED,
                            HireEffectState.EXECUTING,
                        }
                        else "app_registration"
                    ),
                )
                continue
            credential_ref = self._validate_receipt(state, state.app_id, receipt)
            if vault_state is HireEffectState.EXECUTING:
                state = self.commit_effect_transition(
                    state.intent_id,
                    effect_id="store-credential",
                    effect_type="credential_vault_put",
                    next_state=HireEffectState.COMMITTED,
                    metadata={
                        "app_id": state.app_id,
                        "credential_ref": credential_ref,
                    },
                )
            elif vault_state is not HireEffectState.COMMITTED:
                self._mark_action_required_sync(
                    state,
                    effect_id="register-app",
                    effect_type="app_registration",
                )
                continue
            self._bind_principal(state, state.app_id, credential_ref)

    def _matching_orphan_receipt(self, state: DurableHireState) -> object | None:
        if self._credential_vault is None:
            return None
        live_refs = {
            principal.credential_ref
            for principal in self._projection_state.bot_principals.values()
            if principal.credential_ref
        }
        receipts = self._credential_vault.find_orphan_receipts(live_refs)
        matches = [
            receipt
            for receipt in receipts
            if getattr(receipt, "agent_id", None) == state.agent_id
            and getattr(receipt, "app_id", None) == state.app_id
            and getattr(receipt, "hire_intent_id", None) == state.intent_id
            and getattr(receipt, "attempt_id", None) == state.attempt_id
        ]
        if len(matches) > 1:
            raise HireAdmissionError("ambiguous credential orphan receipts")
        return matches[0] if matches else None

    def _mark_action_required_sync(
        self,
        state: DurableHireState,
        *,
        effect_id: str,
        effect_type: str,
    ) -> DurableHireState:
        effect_state = state.effect_state(effect_id)
        if effect_state in {HireEffectState.PREPARED, HireEffectState.EXECUTING}:
            state = self.commit_effect_transition(
                state.intent_id,
                effect_id=effect_id,
                effect_type=effect_type,
                next_state=HireEffectState.ACTION_REQUIRED,
                metadata=dict(state.metadata_for(effect_id)),
            )
        if state.phase is not HirePhase.ACTION_REQUIRED:
            state = self._commit_phase_transition(state, HirePhase.ACTION_REQUIRED)
        return state

    def _require_hire(self, intent_id: str) -> DurableHireState:
        state = self._hire_projection.get(intent_id)
        if state is None:
            raise HireAdmissionError("unknown hire intent")
        return state

    def get_state(self, intent_id: str) -> DurableHireState | None:
        """Return the current immutable replay state without running recovery."""
        with self._mutex:
            return self._hire_projection.get(intent_id)

    def list_states(self) -> tuple[DurableHireState, ...]:
        with self._mutex:
            return tuple(self._hire_projection.states.values())

    def begin_channel_revalidation(
        self,
        intent_id: str,
        *,
        observed_generation: int,
    ) -> DurableHireState:
        """Fence a crashed Channel generation before any replacement launch."""
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            state = self._require_hire(intent_id)
            if (
                state.phase
                not in {
                    HirePhase.CONFIGURING,
                    HirePhase.VALIDATING,
                    HirePhase.ACTIVE,
                    HirePhase.READY_PENDING_VERIFICATION,
                }
                or state.channel_generation != observed_generation
                or observed_generation <= 0
            ):
                raise HireAdmissionError("Channel revalidation binding mismatch")
            state = self._commit_hire_event_locked(
                JournalEvent(
                    event_type="hire.channel.crashed",
                    aggregate_id=state.intent_id,
                    payload={"generation": observed_generation},
                )
            )
            return self._commit_phase_transition_locked(
                state,
                HirePhase.VALIDATING,
            )

    @staticmethod
    def _slash_effect_attempts(
        state: DurableHireState,
        generation: int,
    ) -> list[tuple[int, str]]:
        prefix = f"slash-reconcile:{generation}"
        attempts: list[tuple[int, str]] = []
        for effect_id, _effect_state in state.effects:
            if effect_id == prefix:
                attempts.append((0, effect_id))
                continue
            if not effect_id.startswith(prefix + ":"):
                continue
            attempt_text = effect_id.removeprefix(prefix + ":")
            if attempt_text.isdigit():
                attempt = int(attempt_text)
                if attempt > 0 and str(attempt) == attempt_text:
                    attempts.append((attempt, effect_id))
        attempts.sort()
        return attempts

    @classmethod
    def _latest_slash_effect_id(
        cls,
        state: DurableHireState,
        generation: int,
    ) -> str | None:
        attempts = cls._slash_effect_attempts(state, generation)
        return attempts[-1][1] if attempts else None

    def stop_admission(self) -> None:
        """Reject new hires while allowing already submitted activities to drain."""
        with self._mutex:
            self._admission_closed = True

    def mark_runtime_recovered(self) -> None:
        """Open admission only after composition finishes process recovery."""
        with self._mutex:
            if self._closed:
                raise HireAdmissionError("closed")
            self._runtime_recovery_ready = True

    def close(self) -> None:
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            if self._closed:
                return
            self._admission_closed = True
            self._closed = True
            self._writer.close()

    @staticmethod
    def _validate_request(request: EmployeeHireRequest) -> None:
        if not isinstance(request, EmployeeHireRequest):
            raise HireAdmissionError("invalid hire request")
        required = {
            "tenant_key": request.tenant_key,
            "employee_name": request.employee_name,
            "tool": request.tool,
            "model": request.model,
            "effort": request.effort,
            "profile": request.profile,
            "chat_id": request.chat_id,
            "message_id": request.message_id,
            "requester_principal_id": request.requester_principal_id,
        }
        for field_name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                raise HireAdmissionError(f"{field_name} is required")
        if not isinstance(request.role, str) or not isinstance(request.persona, str):
            raise HireAdmissionError("role and persona must be strings")
        if (
            not isinstance(request.personality_traits, tuple)
            or not isinstance(request.capabilities, tuple)
            or not isinstance(request.permissions, tuple)
        ):
            raise HireAdmissionError("employee profile collections must be tuples")
        if (
            not isinstance(request.existing_app_id, str)
            or (
                request.existing_app_id
                and re.fullmatch(
                    r"cli_[A-Za-z0-9_-]{3,128}",
                    request.existing_app_id,
                )
                is None
            )
        ):
            raise HireAdmissionError("existing_app_id is invalid")
        if (
            not isinstance(request.requester_union_id, str)
            or request.requester_union_id != request.requester_union_id.strip()
        ):
            raise HireAdmissionError("requester_union_id is invalid")
        from src.acp.employee_selection import validate_employee_model_components

        try:
            validate_employee_model_components(
                request.tool,
                request.model,
                request.profile,
                request.effort,
            )
        except ValueError:
            raise HireAdmissionError("invalid employee model selection") from None

    @staticmethod
    def _matches_request(
        state: DurableHireState,
        request: EmployeeHireRequest,
    ) -> bool:
        return (
            state.tenant_key == request.tenant_key
            and state.message_id == request.message_id
            and state.chat_id == request.chat_id
            and state.requester_principal_id == request.requester_principal_id
            and state.requester_union_id == request.requester_union_id
            and state.employee_name == request.employee_name
            and state.tool == request.tool
            and state.model == request.model
            and state.effort == request.effort
            and state.profile == request.profile
            and state.role == request.role
            and state.persona == request.persona
            and state.personality_traits == request.personality_traits
            and state.capabilities == request.capabilities
            and state.permissions == request.permissions
            and state.existing_app_id == request.existing_app_id
        )

    def _app_id_assigned_locked(
        self,
        app_id: str,
        *,
        exclude_intent_id: str = "",
    ) -> bool:
        """Check every durable/canonical app claim while admission locks are held."""

        excluded = self._hire_projection.get(exclude_intent_id)
        excluded_agent_id = excluded.agent_id if excluded is not None else ""

        def employee_is_live(agent_id: str) -> bool:
            employee = self._projection_state.employees.get(agent_id)
            return employee is None or employee.state is not EmployeeState.ARCHIVED

        for state in self._hire_projection.states.values():
            if state.intent_id == exclude_intent_id or not employee_is_live(
                state.agent_id
            ):
                continue
            reserved = {
                state.existing_app_id,
                state.app_id,
                dict(state.metadata_for("register-app")).get("app_id", ""),
            }
            if app_id in reserved:
                return True
        return any(
            principal.agent_id != excluded_agent_id
            and principal.app_id == app_id
            and employee_is_live(principal.agent_id)
            for principal in self._projection_state.bot_principals.values()
        )

    def _commit_registered_app(
        self,
        intent_id: str,
        app_id: str,
    ) -> DurableHireState:
        """Atomically reserve a registrar result and commit its external effect."""

        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            self._synchronize_projection_to_journal_locked()
            current = self._require_hire(intent_id)
            if current.effect_state("register-app") is not HireEffectState.EXECUTING:
                raise HireAdmissionError("invalid effect transition")
            if self._app_id_assigned_locked(
                app_id,
                exclude_intent_id=intent_id,
            ):
                raise HireAdmissionError("registered app already assigned")
            return self.commit_effect_transition(
                intent_id,
                effect_id="register-app",
                effect_type="app_registration",
                next_state=HireEffectState.COMMITTED,
                metadata={"app_id": app_id},
            )

    def bind_requester_union_id(
        self,
        intent_id: str,
        requester_union_id: str,
    ) -> DurableHireState:
        """Durably attach a cross-app identity to a legacy pending hire."""

        if (
            not isinstance(requester_union_id, str)
            or not requester_union_id.strip()
            or requester_union_id != requester_union_id.strip()
        ):
            raise HireAdmissionError("requester_union_id is required")
        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            state = self._require_hire(intent_id)
            if state.requester_union_id:
                if state.requester_union_id != requester_union_id:
                    raise HireAdmissionError("requester identity binding mismatch")
                return state
            return self._commit_hire_event_locked(
                JournalEvent(
                    event_type="hire.requester_identity_bound",
                    aggregate_id=state.intent_id,
                    payload={"requester_union_id": requester_union_id},
                )
            )


__all__ = [
    "HireAdmissionError",
    "HireReadiness",
    "ProductionEmployeeHireService",
]
