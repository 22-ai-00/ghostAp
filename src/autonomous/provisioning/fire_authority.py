"""Journal/workforce authority for employee retirement admission and terminal facts."""

from __future__ import annotations

from datetime import UTC, datetime

from ..domain import EmployeeState, WorkerType
from ..ingress.service import EmployeeIngressService
from ..journal.frame import JournalEvent
from ..journal.writer import CommitState, JournalWriter
from ..workforce.projection import validate_workforce_events
from .fire_service import (
    EmployeeFireRequest,
    EmployeeFireTarget,
    FireServiceError,
)
from .fire_state import (
    DurableFireState,
    FireCleanupMode,
    FirePhase,
    rebuild_fire_projection,
)
from .hire_state import DurableHireState, HireEffectState


class JournalFireAuthority:
    """Authorize admins and commit retirement facts into the shared Journal."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        hire_service: object,
        ingress_service: EmployeeIngressService,
        admin_principal_ids: frozenset[str],
    ) -> None:
        self._writer = writer
        self._hire = hire_service
        self._ingress = ingress_service
        self._admins = frozenset(admin_principal_ids)

    def resolve(self, request: EmployeeFireRequest) -> EmployeeFireTarget:
        self.authorize_request(request)
        projection = self._hire.synchronize_projection()
        return self._resolve_from_projection(request, projection)

    def authorize_request(self, request: EmployeeFireRequest) -> None:
        if request.requester_principal_id not in self._admins:
            raise FireServiceError("fire is not authorized")

    def ensure_no_unresolved_external_mutation(
        self,
        target: EmployeeFireTarget,
    ) -> None:
        """Reject a crashed external mutation regardless of fence presence."""

        with self._hire.employee_dispatch_guard():
            state = self._hire_state(target.agent_id)
            if state is None:
                return
            effect_types = dict(state.effect_types)
            if any(
                effect_state is HireEffectState.EXECUTING
                and effect_types.get(effect_id)
                in {"slash_reconciliation", "employee_channel_start"}
                for effect_id, effect_state in state.effects
            ):
                raise FireServiceError("external mutation outcome is unresolved")

    def _resolve_from_projection(
        self,
        request: EmployeeFireRequest,
        projection: object,
    ) -> EmployeeFireTarget:
        identity_matches = tuple(
            employee
            for employee in projection.employees.values()
            if employee.tenant_key == request.tenant_key
            and employee.worker_type is WorkerType.VISIBLE
            and request.employee in {employee.agent_id, employee.name}
        )
        matches = tuple(
            employee
            for employee in identity_matches
            if employee.state
            in {
                EmployeeState.PROVISIONING_APP,
                EmployeeState.STORING_CREDENTIAL,
                EmployeeState.CONFIGURING,
                EmployeeState.VALIDATING,
                EmployeeState.READY_PENDING_VERIFICATION,
                EmployeeState.ACTIVE,
                EmployeeState.ACTION_REQUIRED,
            }
        )
        if len(matches) != 1:
            if not matches and any(
                employee.state is EmployeeState.ARCHIVED
                for employee in identity_matches
            ):
                raise FireServiceError("employee already archived")
            raise FireServiceError("retirable employee was not uniquely resolved")
        employee = matches[0]
        if not employee.bot_principal_id:
            hire_state = self._hire_state(employee.agent_id)
            if hire_state is None:
                return EmployeeFireTarget(
                    tenant_key=employee.tenant_key,
                    agent_id=employee.agent_id,
                    employee_name=employee.name,
                    bot_principal_id="",
                    app_id="",
                    credential_ref="",
                    cleanup_mode=FireCleanupMode.EXTERNAL_UNKNOWN,
                )
            credential_metadata = dict(hire_state.metadata_for("store-credential"))
            if (
                hire_state.effect_state("store-credential")
                is HireEffectState.COMMITTED
                and credential_metadata.get("app_id")
                and credential_metadata.get("credential_ref")
                and hire_state.bot_principal_id
            ):
                return EmployeeFireTarget(
                    tenant_key=employee.tenant_key,
                    agent_id=employee.agent_id,
                    employee_name=employee.name,
                    bot_principal_id=hire_state.bot_principal_id,
                    app_id=credential_metadata["app_id"],
                    credential_ref=credential_metadata["credential_ref"],
                    cleanup_mode=FireCleanupMode.RECOVERABLE,
                )
            register_state = hire_state.effect_state("register-app")
            cleanup_mode = (
                FireCleanupMode.SAFE_ABORT
                if register_state
                in {None, HireEffectState.PLANNED, HireEffectState.PREPARED}
                else FireCleanupMode.EXTERNAL_UNKNOWN
            )
            return EmployeeFireTarget(
                tenant_key=employee.tenant_key,
                agent_id=employee.agent_id,
                employee_name=employee.name,
                bot_principal_id="",
                app_id=self._prebinding_app_id(hire_state),
                credential_ref="",
                cleanup_mode=cleanup_mode,
            )
        principal = projection.bot_principals.get(employee.bot_principal_id)
        if (
            principal is None
            or principal.tenant_key != employee.tenant_key
            or principal.agent_id != employee.agent_id
            or not principal.app_id
            or not principal.credential_ref
        ):
            raise FireServiceError("employee credential authority is unavailable")
        return EmployeeFireTarget(
            tenant_key=employee.tenant_key,
            agent_id=employee.agent_id,
            employee_name=employee.name,
            bot_principal_id=employee.bot_principal_id,
            app_id=principal.app_id,
            credential_ref=principal.credential_ref,
        )

    def admit(
        self,
        request: EmployeeFireRequest,
        target: EmployeeFireTarget,
        intent_id: str,
    ) -> EmployeeFireTarget:
        with self._hire.employee_dispatch_guard(), self._ingress.employee_dispatch_guard(), self._writer.transaction_guard():
            self._ingress.synchronize_projection_unlocked()
            projection = self._hire.synchronize_projection_unlocked()
            # Re-resolve under the same locks that commit retirement. Provisioning
            # may have bound a principal after the optimistic resolve above.
            resolved_target = self._resolve_from_projection(request, projection)
            if (
                resolved_target.tenant_key != target.tenant_key
                or resolved_target.agent_id != target.agent_id
            ):
                raise FireServiceError(
                    "employee retirement authority changed during admission"
                )
            target = resolved_target
            live_requests = [
                state
                for state in rebuild_fire_projection(
                    tuple(self._writer.replay())
                ).values()
                if state.tenant_key == target.tenant_key
                and state.agent_id == target.agent_id
                and state.phase
                in {FirePhase.RETIRING, FirePhase.ACTION_REQUIRED}
            ]
            if live_requests:
                raise FireServiceError(
                    "employee retirement already in progress"
                )
            ingress_aggregate = (
                f"employee-ingress:{target.tenant_key}:{target.agent_id}"
            )
            requested = JournalEvent(
                event_type="fire.requested",
                aggregate_id=intent_id,
                payload={
                    "intent_id": intent_id,
                    "tenant_key": request.tenant_key,
                    "message_id": request.message_id,
                    "chat_id": request.chat_id,
                    "requester_principal_id": request.requester_principal_id,
                    "agent_id": target.agent_id,
                    "employee_name": target.employee_name,
                    "bot_principal_id": target.bot_principal_id,
                    "app_id": target.app_id,
                    "credential_ref": target.credential_ref,
                    "drain": request.drain,
                    "cleanup_mode": target.cleanup_mode.value,
                },
            )
            retiring = JournalEvent(
                event_type="employee.state_changed",
                aggregate_id=target.agent_id,
                payload={"state": EmployeeState.RETIRING.value},
            )
            ingress_closed = JournalEvent(
                event_type="employee.ingress.closed",
                aggregate_id=ingress_aggregate,
                payload={
                    "tenant_key": target.tenant_key,
                    "agent_id": target.agent_id,
                    "reason_code": "retiring",
                    "closed_at": datetime.now(UTC).isoformat(),
                },
            )
            hire_effect_dispositions = self._hire_effect_dispositions(
                target.agent_id,
            )
            validate_workforce_events(projection, (retiring,))
            frame = self._commit_unlocked(
                (
                    requested,
                    *hire_effect_dispositions,
                    retiring,
                    ingress_closed,
                )
            )
            self._hire.apply_committed_frame_unlocked(frame)
            self._ingress.apply_committed_frame_unlocked(frame)
            return target

    def _hire_effect_dispositions(
        self,
        agent_id: str,
    ) -> tuple[JournalEvent, ...]:
        list_states = getattr(self._hire, "list_states", None)
        if not callable(list_states):
            return ()
        matches = [state for state in list_states() if state.agent_id == agent_id]
        if len(matches) > 1:
            raise FireServiceError("employee hire authority is ambiguous")
        if not matches:
            return ()
        state = matches[0]
        effect_types = dict(state.effect_types)
        unresolved_external = tuple(
            effect_id
            for effect_id, effect_state in state.effects
            if effect_state is HireEffectState.EXECUTING
            and effect_types.get(effect_id)
            in {"slash_reconciliation", "employee_channel_start"}
        )
        if unresolved_external:
            # These calls can outlive the process that dispatched them.  A
            # fresh in-memory gate therefore cannot prove they stopped.  Keep
            # the durable fence, but do not retire or clean up until runtime
            # recovery has observed/disposed the exact effects.  The fence
            # prevents new calls; it is not evidence that an old call ended.
            raise FireServiceError("external mutation outcome is unresolved")
        events: list[JournalEvent] = []
        for effect_id, effect_state in state.effects:
            if effect_state.value not in {"prepared", "executing"}:
                continue
            effect_type = effect_types.get(effect_id)
            if not effect_type:
                raise FireServiceError("employee hire effect type is unavailable")
            metadata = {
                **dict(state.metadata_for(effect_id)),
                "error_code": "retirement_requested",
            }
            events.append(
                JournalEvent(
                    event_type="hire.effect.action_required",
                    aggregate_id=state.intent_id,
                    payload={
                        "effect_id": effect_id,
                        "effect_type": effect_type,
                        "metadata": metadata,
                    },
                )
            )
        return tuple(events)

    def _hire_state(self, agent_id: str) -> DurableHireState | None:
        list_states = getattr(self._hire, "list_states", None)
        if not callable(list_states):
            return None
        matches = [state for state in list_states() if state.agent_id == agent_id]
        if len(matches) > 1:
            raise FireServiceError("employee hire authority is ambiguous")
        if not matches:
            return None
        return matches[0]

    @staticmethod
    def _prebinding_app_id(state: DurableHireState) -> str:
        return (
            dict(state.metadata_for("register-app")).get("app_id", "")
            or state.app_id
            or state.existing_app_id
        )

    def mark_action_required(self, agent_id: str) -> None:
        self._commit_employee_state(agent_id, EmployeeState.ACTION_REQUIRED)

    def confirm_external_disposition(
        self,
        request: EmployeeFireRequest,
        state: DurableFireState,
        disposition_ref: str,
    ) -> None:
        if request.requester_principal_id not in self._admins:
            raise FireServiceError("fire is not authorized")
        if (
            request.tenant_key != state.tenant_key
            or request.employee not in {state.agent_id, state.employee_name}
            or disposition_ref != (state.app_id or "NO_APP_FOUND")
        ):
            raise FireServiceError("external disposition reference mismatch")
        with self._hire.employee_dispatch_guard(), self._writer.transaction_guard():
            current_fire = rebuild_fire_projection(
                tuple(self._writer.replay())
            ).get(state.intent_id)
            if current_fire is None:
                raise FireServiceError("external cleanup request is unavailable")
            if current_fire.external_disposition_confirmed:
                if current_fire.external_disposition_ref != disposition_ref:
                    raise FireServiceError(
                        "external disposition reference mismatch"
                    )
                return
            if (
                current_fire.phase is not FirePhase.ACTION_REQUIRED
                or current_fire.cleanup_mode
                is not FireCleanupMode.EXTERNAL_UNKNOWN
                or current_fire.error_code
                != "external_cleanup_authority_unavailable"
                or disposition_ref
                != (current_fire.app_id or "NO_APP_FOUND")
            ):
                raise FireServiceError(
                    "external cleanup request is not awaiting confirmation"
                )
            projection = self._hire.synchronize_projection_unlocked()
            current = projection.employees.get(state.agent_id)
            if (
                current is None
                or current.tenant_key != state.tenant_key
                or current.state is not EmployeeState.ACTION_REQUIRED
            ):
                raise FireServiceError(
                    "external cleanup request is not awaiting confirmation"
                )
            event = JournalEvent(
                event_type="fire.external_disposition_confirmed",
                aggregate_id=state.intent_id,
                payload={
                    "disposition_ref": disposition_ref,
                    "disposed_by": request.requester_principal_id,
                    "disposed_at": datetime.now(UTC).isoformat(),
                    "confirmation_message_id": request.message_id,
                },
            )
            frame = self._commit_unlocked((event,))
            self._hire.apply_committed_frame_unlocked(frame)

    def mark_credential_destroyed(self, target: EmployeeFireTarget) -> None:
        event = JournalEvent(
            event_type="credential.destroyed",
            aggregate_id=target.bot_principal_id,
            payload={"credential_ref": target.credential_ref},
        )
        with self._hire.employee_dispatch_guard(), self._writer.transaction_guard():
            projection = self._hire.synchronize_projection_unlocked()
            principal = projection.bot_principals.get(target.bot_principal_id)
            if principal is None:
                hire_state = self._hire_state(target.agent_id)
                metadata = (
                    {}
                    if hire_state is None
                    else dict(hire_state.metadata_for("store-credential"))
                )
                if (
                    target.cleanup_mode is FireCleanupMode.RECOVERABLE
                    and hire_state is not None
                    and hire_state.bot_principal_id == target.bot_principal_id
                    and metadata.get("app_id") == target.app_id
                    and metadata.get("credential_ref") == target.credential_ref
                    and hire_state.effect_state("store-credential")
                    is HireEffectState.COMMITTED
                ):
                    return
                raise FireServiceError("employee bot principal is unavailable")
            if principal.credential_ref == "":
                return
            validate_workforce_events(projection, (event,))
            frame = self._commit_unlocked((event,))
            self._hire.apply_committed_frame_unlocked(frame)

    def mark_archived(
        self,
        agent_id: str,
        intent_id: str,
        *,
        external_disposition_confirmed: bool,
    ) -> None:
        archived = JournalEvent(
            event_type="employee.state_changed",
            aggregate_id=agent_id,
            payload={"state": EmployeeState.ARCHIVED.value},
        )
        completed = JournalEvent(
            event_type="fire.completed",
            aggregate_id=intent_id,
            payload={
                "external_app_disposition": (
                    "manual_disposition_confirmed"
                    if external_disposition_confirmed
                    else "manual_deletion_required"
                )
            },
        )
        with self._hire.employee_dispatch_guard(), self._writer.transaction_guard():
            projection = self._hire.synchronize_projection_unlocked()
            current = projection.employees.get(agent_id)
            current_fire = rebuild_fire_projection(
                tuple(self._writer.replay())
            ).get(intent_id)
            if (
                current is not None
                and current.state is EmployeeState.ARCHIVED
                and current_fire is not None
                and current_fire.phase is FirePhase.ARCHIVED
            ):
                return
            validate_workforce_events(projection, (archived,))
            frame = self._commit_unlocked((archived, completed))
            self._hire.apply_committed_frame_unlocked(frame)

    def _commit_employee_state(self, agent_id: str, state: EmployeeState) -> None:
        event = JournalEvent(
            event_type="employee.state_changed",
            aggregate_id=agent_id,
            payload={"state": state.value},
        )
        with self._hire.employee_dispatch_guard(), self._writer.transaction_guard():
            projection = self._hire.synchronize_projection_unlocked()
            current = projection.employees.get(agent_id)
            if (
                current is None
                or current.state is state
                or current.state is EmployeeState.ARCHIVED
            ):
                return
            validate_workforce_events(projection, (event,))
            frame = self._commit_unlocked((event,))
            self._hire.apply_committed_frame_unlocked(frame)

    def _commit_unlocked(self, events: tuple[JournalEvent, ...]):
        last = self._writer.get_last_frame()
        sequence = 0 if last is None else last.sequence
        frame_hash = "" if last is None else last.frame_hash
        aggregate_ids = {event.aggregate_id for event in events}
        result = self._writer.commit(
            events,
            self._writer.get_aggregate_versions(aggregate_ids),
            expected_head_sequence=sequence,
            expected_head_hash=frame_hash,
        )
        if result.state is not CommitState.ANCHORED:
            raise FireServiceError("fire authority fact was not anchored")
        return result.frame


__all__ = ["JournalFireAuthority"]
