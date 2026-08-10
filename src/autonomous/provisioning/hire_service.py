"""Production-shaped durable admission core for visible employee hiring."""

from __future__ import annotations

import asyncio
import math
import re
import threading
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from ..domain import EmployeeDefinition, EmployeeState, WorkerType
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
    validate_workforce_events,
    workforce_projection_guard,
)
from .callback_bridge import AsyncCallbackBridge
from .hire_port import EmployeeHireRequest, EmployeeRoleUpdateRequest
from .hire_state import (
    DurableHireState,
    HireEffectState,
    HirePhase,
    HireProjection,
)
from .lark_app import (
    RegistrationRequest,
    RegistrationResult,
    current_registration_manifest,
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
        runtime_recovery_ready: bool = True,
        workspace_projector: Any = None,
        admin_principal_ids_provider: Callable[[], frozenset[str]] | None = None,
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
        self._runtime_recovery_ready = runtime_recovery_ready is True
        self._workspace_projector = workspace_projector
        self._admin_principal_ids_provider = (
            admin_principal_ids_provider or (lambda: frozenset())
        )
        self._mutex = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._activities: dict[str, asyncio.Task[DurableHireState]] = {}
        self._admission_closed = False
        self._closed = False
        self._hire_projection = HireProjection.empty()
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

    def list_employee_roster(
        self,
        tenant_key: str,
        include_archived: bool = True,
    ) -> list[EmployeeDefinition]:
        """List visible employees without granting them dispatch authority."""

        projection = self.synchronize_projection()
        employees = (
            employee
            for employee in projection.employees.values()
            if employee.tenant_key == tenant_key
            and employee.worker_type is WorkerType.VISIBLE
            and (
                include_archived
                or employee.state is not EmployeeState.ARCHIVED
            )
        )
        return sorted(
            employees,
            key=lambda employee: (employee.name.casefold(), employee.agent_id),
        )

    def update_employee_role(
        self,
        request: EmployeeRoleUpdateRequest,
    ) -> EmployeeDefinition:
        """Commit one tenant-scoped role change through the workforce Journal."""

        if not isinstance(request, EmployeeRoleUpdateRequest):
            raise HireAdmissionError("invalid employee role update request")
        required = (
            request.tenant_key,
            request.employee,
            request.requester_principal_id,
            request.message_id,
        )
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in required
        ):
            raise HireAdmissionError("invalid employee role update request")
        if not isinstance(request.role, str):
            raise HireAdmissionError("employee role is invalid")
        normalized_role = " ".join(request.role.split())
        if not normalized_role or len(normalized_role) > 80:
            raise HireAdmissionError("employee role is invalid")

        with self.employee_dispatch_guard(), self._writer.transaction_guard():
            self._synchronize_projection_to_journal_locked()
            try:
                raw_admins = self._admin_principal_ids_provider()
                if not isinstance(raw_admins, (frozenset, set, tuple, list)):
                    raise TypeError("invalid admin provider result")
                admins = frozenset(raw_admins)
                if any(
                    not isinstance(admin, str)
                    or not admin
                    or admin != admin.strip()
                    for admin in admins
                ):
                    raise TypeError("invalid admin provider result")
            except Exception:
                raise HireAdmissionError(
                    "employee role update is not authorized"
                ) from None
            if request.requester_principal_id not in admins:
                raise HireAdmissionError("employee role update is not authorized")

            matches = tuple(
                employee
                for employee in self._projection_state.employees.values()
                if employee.tenant_key == request.tenant_key
                and employee.worker_type is WorkerType.VISIBLE
                and employee.state is not EmployeeState.ARCHIVED
                and request.employee in {employee.agent_id, employee.name}
            )
            if len(matches) != 1:
                raise HireAdmissionError("employee role target is unavailable or ambiguous")
            employee = matches[0]
            if employee.role == normalized_role:
                return employee

            event = JournalEvent(
                event_type="employee.profile_changed",
                aggregate_id=employee.agent_id,
                payload={"role": normalized_role},
            )
            commit_workforce_events_unlocked(
                self._writer,
                self._projection_state,
                (event,),
            )
            return self._projection_state.employees[employee.agent_id]

    def synchronize_projection_unlocked(self) -> ProjectionState:
        """Advance views while the caller owns ``employee_dispatch_guard``."""

        self._synchronize_projection_to_journal_locked()
        return self._projection_state








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
        self._reconcile_recovered_hires()
        with self.employee_dispatch_guard():
            return self._hire_projection


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
                state.phase
                not in {
                    HirePhase.CONFIGURING,
                    HirePhase.VALIDATING,
                    HirePhase.READY_PENDING_VERIFICATION,
                }
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
            retired_replies = [
                (effect_id, effect_type)
                for effect_id, effect_state in state.effects
                if effect_state
                in {HireEffectState.PREPARED, HireEffectState.EXECUTING}
                and (effect_type := effect_types.get(effect_id, ""))
                in {
                    "employee_activation_required_reply",
                    "employee_status_reply",
                }
            ]
            for effect_id, effect_type in retired_replies:
                metadata = dict(state.metadata_for(effect_id))
                metadata.setdefault("error_code", "manual_activation_retired")
                state = self.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type=effect_type,
                    next_state=HireEffectState.ACTION_REQUIRED,
                    metadata=metadata,
                )
            if (
                state.phase is HirePhase.READY_PENDING_VERIFICATION
                and state.activation_ingress_event_id
            ):
                self._commit_phase_transition(state, HirePhase.ACTIVE)
                continue
            if state.phase in {
                HirePhase.ACTION_REQUIRED,
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

__all__ = [
    "HireAdmissionError",
    "HireReadiness",
    "ProductionEmployeeHireService",
]
