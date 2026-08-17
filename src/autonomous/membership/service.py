"""Journal-backed membership mutation, reconciliation, and Router health."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from ..domain import EmployeeState, WorkerType
from ..journal.frame import JournalEvent
from ..journal.writer import JournalWriter
from ..provisioning.external_mutation_gate import (
    EmployeeExternalMutationGate,
    EmployeeExternalMutationLease,
    ExternalMutationFenced,
    ExternalMutationKind,
)
from ..workforce.projection import commit_workforce_events_unlocked
from .lark import MembershipRemoteRejected, MembershipRemoteUnknown
from .models import (
    MembershipEffect,
    MembershipEffectState,
    MembershipOperation,
    MembershipState,
    membership_effect_id,
)
from .projection import (
    EFFECT_ACTION_REQUIRED,
    EFFECT_COMMITTED,
    EFFECT_EXECUTING,
    EFFECT_PREPARED,
    MembershipProjectionState,
    MembershipRecord,
    rebuild_membership_projection,
    reduce_membership_frame,
)

_AUDITABLE_EMPLOYEE_STATES = frozenset(
    {
        EmployeeState.ACTIVE,
        EmployeeState.VALIDATING,
        EmployeeState.READY_PENDING_VERIFICATION,
    }
)


class MembershipServiceError(RuntimeError):
    pass


class MembershipAuthorizationError(MembershipServiceError):
    pass


class MembershipBindingError(MembershipServiceError):
    pass


@dataclass(frozen=True, slots=True)
class MembershipMutationRequest:
    tenant_key: str
    chat_id: str
    agent_id: str
    requester_principal_id: str
    operation: MembershipOperation

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value and value == value.strip()
            for value in (
                self.tenant_key,
                self.chat_id,
                self.agent_id,
                self.requester_principal_id,
            )
        ):
            raise ValueError("membership mutation coordinates are required")
        try:
            object.__setattr__(
                self,
                "operation",
                MembershipOperation(self.operation),
            )
        except (TypeError, ValueError):
            raise ValueError("invalid membership operation") from None


@dataclass(frozen=True, slots=True)
class MembershipMutationOutcome:
    state: MembershipState
    confirmed: bool
    changed: bool
    effect_id: str = ""
    error_code: str = ""


@dataclass(frozen=True, slots=True)
class MembershipAuditSummary:
    checked: int = 0
    confirmed: int = 0
    removed: int = 0
    degraded: int = 0


@dataclass(frozen=True, slots=True)
class _Authority:
    app_id: str
    credential_ref: str
    member_groups: tuple[str, ...]


@dataclass(slots=True)
class _ExternalMutationProgress:
    effect_id: str = ""
    external_started: bool = False


class EmployeeMembershipService:
    """Own canonical employee membership; legacy registry is never a fallback."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        hire_service: Any,
        remote: Any,
        admin_principal_ids: frozenset[str],
        team_owner_resolver: Callable[[str], str],
        team_active_resolver: Callable[[str], bool],
        external_mutation_gate: EmployeeExternalMutationGate | None = None,
    ) -> None:
        if not isinstance(writer, JournalWriter):
            raise TypeError("writer must be JournalWriter")
        if not callable(team_owner_resolver):
            raise TypeError("team_owner_resolver is required")
        if not callable(team_active_resolver):
            raise TypeError("team_active_resolver is required")
        self._writer = writer
        self._hire = hire_service
        self._remote = remote
        self._admins = frozenset(admin_principal_ids)
        self._team_owner_resolver = team_owner_resolver
        self._team_active_resolver = team_active_resolver
        self._external_mutation_gate = external_mutation_gate
        self._state = MembershipProjectionState()
        self._mutex = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._chat_locks: dict[str, threading.RLock] = {}
        self._chat_locks_guard = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self.rebuild_projection()

    @property
    def state(self) -> MembershipProjectionState:
        with self._mutex:
            return self._state.clone()

    def get(
        self,
        tenant_key: str,
        chat_id: str,
        agent_id: str,
    ) -> MembershipRecord | None:
        self.rebuild_projection()
        with self._mutex:
            return self._state.records.get((tenant_key, chat_id, agent_id))

    def get_employee(self, tenant_key: str, agent_id: str) -> Any | None:
        """Return one projected employee without exposing a writable registry."""

        projection = self._hire.synchronize_projection()
        employee = projection.employees.get(agent_id)
        if employee is None or employee.tenant_key != tenant_key:
            return None
        return employee

    def find_employee_by_name(self, tenant_key: str, name: str) -> Any | None:
        """Resolve an exact tenant-scoped visible employee name."""

        normalized = name.casefold()
        matches = [
            employee
            for employee in self.list_employees(tenant_key)
            if employee.name.casefold() == normalized
        ]
        if len(matches) > 1:
            raise MembershipBindingError("employee name is ambiguous")
        return matches[0] if matches else None

    def resolve_member_bot(
        self,
        tenant_key: str,
        chat_id: str,
        bot_principal_id: str,
    ) -> Any | None:
        """Bind a collaboration event to one active employee Bot membership."""

        if not all((tenant_key, chat_id, bot_principal_id)):
            return None
        matches = [
            employee
            for employee in self.list_employees(tenant_key)
            if employee.bot_principal_id == bot_principal_id
            and chat_id in employee.member_groups
        ]
        if len(matches) != 1:
            return None
        employee = matches[0]
        return (
            None
            if self.is_degraded(employee.agent_id, chat_id)
            else employee
        )

    def list_employees(self, tenant_key: str) -> list[Any]:
        projection = self._hire.synchronize_projection()
        return [
            employee
            for employee in projection.employees.values()
            if employee.tenant_key == tenant_key
            and employee.state is EmployeeState.ACTIVE
            and employee.worker_type is WorkerType.VISIBLE
        ]

    def is_degraded(self, agent_id: str, team_id: str) -> bool:
        """Deny unless Journal proves ACTIVE membership for this exact chat."""

        return self.degraded_for((agent_id,), team_id).get(agent_id, True)

    def degraded_for(
        self,
        agent_ids: Iterable[str],
        team_id: str,
    ) -> dict[str, bool]:
        """Resolve many membership health checks from one Journal snapshot."""

        unique_agent_ids = tuple(dict.fromkeys(agent_ids))
        self.rebuild_projection()
        projection = self._hire.synchronize_projection()
        health: dict[str, bool] = {}
        with self._mutex:
            for agent_id in unique_agent_ids:
                employee = projection.employees.get(agent_id)
                if employee is None or team_id not in employee.member_groups:
                    health[agent_id] = True
                    continue
                record = self._state.records.get(
                    (employee.tenant_key, team_id, agent_id)
                )
                health[agent_id] = (
                    record is None
                    or record.state is not MembershipState.ACTIVE
                )
        return health

    def mutate(
        self,
        request: MembershipMutationRequest,
    ) -> MembershipMutationOutcome:
        if not isinstance(request, MembershipMutationRequest):
            raise TypeError("request must be MembershipMutationRequest")
        with self._chat_lock(request.chat_id):
            authority = self._resolve_authority(request)
            lease = self._acquire_add_lease(request)
            progress = _ExternalMutationProgress()
            completed = False
            try:
                outcome = self._mutate_with_authority(
                    request,
                    authority,
                    progress=progress,
                )
                completed = True
                return outcome
            except BaseException:
                if progress.external_started and progress.effect_id:
                    self._best_effort_terminal_disposition(progress.effect_id)
                raise
            finally:
                if lease is not None and (
                    completed
                    or not progress.external_started
                    or self._effect_is_terminal(progress.effect_id)
                ):
                    lease.release()

    def _acquire_add_lease(
        self,
        request: MembershipMutationRequest,
    ) -> EmployeeExternalMutationLease | None:
        gate = self._external_mutation_gate
        if gate is None:
            return None
        try:
            return gate.acquire(
                request.tenant_key,
                request.agent_id,
                ExternalMutationKind.MEMBERSHIP_ADD,
            )
        except ExternalMutationFenced:
            raise MembershipBindingError("employee is retiring") from None

    def _mutate_with_authority(
        self,
        request: MembershipMutationRequest,
        authority: _Authority,
        *,
        progress: _ExternalMutationProgress | None = None,
    ) -> MembershipMutationOutcome:
        progress = progress or _ExternalMutationProgress()
        desired = request.operation is MembershipOperation.ADD
        desired_state = (
            MembershipState.ACTIVE if desired else MembershipState.ABSENT
        )
        projected = request.chat_id in authority.member_groups
        record = self.get(
            request.tenant_key,
            request.chat_id,
            request.agent_id,
        )
        if (
            projected is desired
            and record is not None
            and record.state is desired_state
            and record.confirmed_state is desired_state
        ):
            return MembershipMutationOutcome(
                state=desired_state,
                confirmed=True,
                changed=False,
            )

        if projected is desired:
            try:
                observed = self._observe(request, authority)
            except MembershipRemoteUnknown:
                effect = self._prepare(request, authority)
                progress.effect_id = effect.effect_id
                self._mark_executing(effect.effect_id)
                return self._mark_action_required(
                    effect.effect_id,
                    (
                        "remote_unknown"
                        if record is not None
                        and record.state is MembershipState.DEGRADED
                        else "idempotency_observation_unknown"
                    ),
                )
            if observed is desired:
                effect = self._prepare(request, authority)
                progress.effect_id = effect.effect_id
                self._mark_executing(effect.effect_id)
                return self._commit_confirmed(effect.effect_id, observed)

        effect = self._prepare(request, authority)
        progress.effect_id = effect.effect_id
        self._mark_executing(effect.effect_id)
        mutation_error = ""
        mutation_confirmed = False
        try:
            progress.external_started = True
            mutation_confirmed = self._remote.mutate(
                request.operation,
                chat_id=request.chat_id,
                app_id=authority.app_id,
            ) is True
            if not mutation_confirmed:
                mutation_error = "remote_unknown"
        except MembershipRemoteRejected:
            mutation_error = "remote_rejected"
        except Exception:
            mutation_error = "remote_unknown"
        try:
            observed = self._observe(request, authority)
        except MembershipRemoteUnknown:
            # A strict successful create/delete response (including empty
            # invalid/pending lists) is direct remote evidence.  The
            # follow-up employee probe is defense in depth and can be
            # temporarily unavailable while an existing app's scopes are
            # being upgraded.
            if mutation_confirmed:
                return self._commit_confirmed(effect.effect_id, desired)
            return self._mark_action_required(
                effect.effect_id,
                mutation_error or "remote_unknown",
            )
        if observed is desired:
            return self._commit_confirmed(effect.effect_id, observed)
        return self._mark_action_required(
            effect.effect_id,
            mutation_error or "observation_mismatch",
        )

    def _effect_is_terminal(self, effect_id: str) -> bool:
        if not effect_id:
            return False
        try:
            self.rebuild_projection()
            effect = self._state.effects.get(effect_id)
            return effect is not None and effect.state.terminal
        except Exception:
            return False

    def _best_effort_terminal_disposition(self, effect_id: str) -> None:
        if self._effect_is_terminal(effect_id):
            return
        try:
            self._mark_action_required(effect_id, "terminal_anchor_failed")
        except Exception:
            # Without a durable terminal fact the active lease intentionally
            # remains held.  Restart replay will retain the EXECUTING cleanup
            # obligation instead of allowing retirement to miss it.
            return

    def retire_all(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        requester_principal_id: str,
    ) -> tuple[MembershipMutationOutcome, ...]:
        """Remove a RETIRING employee from every known team; never permits add."""

        if requester_principal_id not in self._admins:
            raise MembershipAuthorizationError("retirement membership cleanup is not authorized")
        projection = self._hire.synchronize_projection()
        employee = projection.employees.get(agent_id)
        if (
            employee is None
            or employee.tenant_key != tenant_key
            or employee.state not in {EmployeeState.RETIRING, EmployeeState.ACTION_REQUIRED}
            or employee.worker_type is not WorkerType.VISIBLE
        ):
            raise MembershipBindingError("retiring employee membership authority unavailable")
        self.rebuild_projection()
        with self._mutex:
            historical_targets = {
                record.chat_id
                for record in self._state.records.values()
                if record.tenant_key == tenant_key
                and record.agent_id == agent_id
            }
        targets = tuple(sorted(set(employee.member_groups) | historical_targets))
        outcomes: list[MembershipMutationOutcome] = []
        for chat_id in targets:
            self._dispose_pending_retirement_effect(
                tenant_key=tenant_key,
                chat_id=chat_id,
                agent_id=agent_id,
            )
            outcomes.append(
                self._force_retirement_remove(
                    MembershipMutationRequest(
                        tenant_key=tenant_key,
                        chat_id=chat_id,
                        agent_id=agent_id,
                        requester_principal_id=requester_principal_id,
                        operation=MembershipOperation.REMOVE,
                    )
                )
            )
        return tuple(outcomes)

    def _force_retirement_remove(
        self,
        request: MembershipMutationRequest,
    ) -> MembershipMutationOutcome:
        """Dispatch an idempotent REMOVE and anchor fresh retirement evidence."""

        with self._chat_lock(request.chat_id):
            authority = self._resolve_authority(request)
            effect = self._prepare(request, authority)
            self._mark_executing(effect.effect_id)
            try:
                self._remote.mutate(
                    MembershipOperation.REMOVE,
                    chat_id=request.chat_id,
                    app_id=authority.app_id,
                )
            except MembershipRemoteRejected:
                pass
            except Exception:
                pass
            try:
                observed = self._observe(request, authority)
            except MembershipRemoteUnknown:
                return self._mark_action_required(
                    effect.effect_id,
                    "retirement_remove_unknown",
                )
            if observed is False:
                return self._commit_confirmed(effect.effect_id, False)
            return self._mark_action_required(
                effect.effect_id,
                "retirement_remove_unconfirmed",
            )

    def _dispose_pending_retirement_effect(
        self,
        *,
        tenant_key: str,
        chat_id: str,
        agent_id: str,
    ) -> None:
        with self._chat_lock(chat_id):
            self.rebuild_projection()
            record = self._state.records.get((tenant_key, chat_id, agent_id))
            effect = (
                self._state.effects.get(record.current_effect_id)
                if record is not None
                else None
            )
            if effect is not None and not effect.state.terminal:
                self._mark_action_required(
                    effect.effect_id,
                    "retirement_requested",
                )

    def retirement_confirmed_empty(
        self,
        *,
        tenant_key: str,
        agent_id: str,
    ) -> bool:
        """Prove every projected or historical team target is absent."""

        self.rebuild_projection()
        projection = self._hire.synchronize_projection()
        employee = projection.employees.get(agent_id)
        if employee is None or employee.tenant_key != tenant_key:
            return False
        if employee.member_groups:
            return False
        retirement_started = employee.state in {
            EmployeeState.RETIRING,
            EmployeeState.ACTION_REQUIRED,
            EmployeeState.ARCHIVED,
        }
        if not retirement_started:
            return False
        retirement_sequence = self._retirement_start_sequence(
            tenant_key,
            agent_id,
        )
        effect_sequences = self._membership_effect_terminal_sequences(
            tenant_key,
            agent_id,
        )
        effect_last_sequences = self._membership_effect_last_sequences(
            tenant_key,
            agent_id,
        )
        with self._mutex:
            effects = tuple(
                effect
                for effect in self._state.effects.values()
                if effect.tenant_key == tenant_key
                and effect.agent_id == agent_id
            )
            records = tuple(
                record
                for record in self._state.records.values()
                if record.tenant_key == tenant_key
                and record.agent_id == agent_id
            )
            for record in records:
                effect = self._state.effects.get(record.current_effect_id)
                if (
                    effect is None
                    or effect.operation is not MembershipOperation.REMOVE
                    or effect.state is not MembershipEffectState.COMMITTED
                    or record.state is not MembershipState.ABSENT
                    or record.confirmed_state is not MembershipState.ABSENT
                ):
                    return False
                if (
                    retirement_sequence is None
                    or effect_sequences.get(effect.effect_id, 0)
                    <= retirement_sequence
                ):
                    return False
            for add_effect in effects:
                if add_effect.operation is not MembershipOperation.ADD:
                    continue
                if not self._add_effect_causally_settled(add_effect):
                    return False
                cutoff = max(
                    retirement_sequence or 0,
                    effect_last_sequences.get(add_effect.effect_id, 0),
                )
                if not any(
                    remove_effect.operation is MembershipOperation.REMOVE
                    and remove_effect.chat_id == add_effect.chat_id
                    and remove_effect.state is MembershipEffectState.COMMITTED
                    and effect_sequences.get(remove_effect.effect_id, 0) > cutoff
                    for remove_effect in effects
                ):
                    return False
        return True

    def _add_effect_causally_settled(self, effect: MembershipEffect) -> bool:
        """Prove an ADD cannot become effective after retirement's REMOVE."""

        if effect.state is MembershipEffectState.COMMITTED:
            return True
        executed = any(
            event.event_type == EFFECT_EXECUTING
            and event.aggregate_id == effect.effect_id
            for frame in self._writer.replay()
            for event in frame.events
        )
        if not executed:
            return True
        # These EXECUTING records are observation-only probes or an explicit
        # remote rejection; no asynchronous ADD request was accepted.
        return effect.state is MembershipEffectState.ACTION_REQUIRED and (
            effect.error_code
            in {
                "event_observation_unknown",
                "event_observed_absent",
                "idempotency_observation_unknown",
                "prepared_recovery_unknown",
                "remote_rejected",
            }
        )

    def _retirement_start_sequence(
        self,
        tenant_key: str,
        agent_id: str,
    ) -> int | None:
        sequence: int | None = None
        for frame in self._writer.replay():
            if any(
                (
                    event.event_type == "employee.external_mutation_fenced"
                    and event.payload.get("tenant_key") == tenant_key
                    and event.payload.get("agent_id") == agent_id
                )
                or (
                    event.event_type == "employee.state_changed"
                    and event.aggregate_id == agent_id
                    and event.payload.get("state") == EmployeeState.RETIRING.value
                )
                for event in frame.events
            ):
                sequence = frame.sequence if sequence is None else min(sequence, frame.sequence)
        return sequence

    def _membership_effect_terminal_sequences(
        self,
        tenant_key: str,
        agent_id: str,
    ) -> dict[str, int]:
        identities = {
            effect.effect_id
            for effect in self._state.effects.values()
            if effect.tenant_key == tenant_key and effect.agent_id == agent_id
        }
        sequences: dict[str, int] = {}
        for frame in self._writer.replay():
            for event in frame.events:
                effect_id = event.payload.get("effect_id")
                if (
                    event.event_type == EFFECT_COMMITTED
                    and effect_id in identities
                ):
                    sequences[effect_id] = frame.sequence
        return sequences

    def _membership_effect_last_sequences(
        self,
        tenant_key: str,
        agent_id: str,
    ) -> dict[str, int]:
        identities = {
            effect.effect_id
            for effect in self._state.effects.values()
            if effect.tenant_key == tenant_key and effect.agent_id == agent_id
        }
        sequences: dict[str, int] = {}
        for frame in self._writer.replay():
            for event in frame.events:
                if event.aggregate_id in identities:
                    sequences[event.aggregate_id] = frame.sequence
        return sequences

    def reconcile_event(
        self,
        *,
        tenant_key: str,
        chat_id: str,
        agent_id: str,
        app_id: str,
        observed_is_member: bool,
    ) -> MembershipMutationOutcome:
        """Use an app-bound durable event only to trigger live observation."""

        with self._chat_lock(chat_id):
            if self._team_active_resolver(chat_id) is not True:
                raise MembershipBindingError("membership team is not active")
            projection = self._hire.synchronize_projection()
            employee = projection.employees.get(agent_id)
            if (
                employee is None
                or employee.tenant_key != tenant_key
                or employee.state is not EmployeeState.ACTIVE
                or employee.worker_type is not WorkerType.VISIBLE
                or not employee.bot_principal_id
            ):
                raise MembershipBindingError("employee membership authority unavailable")
            principal = projection.bot_principals.get(employee.bot_principal_id)
            if (
                principal is None
                or principal.tenant_key != tenant_key
                or principal.agent_id != agent_id
                or not principal.app_id
                or not principal.credential_ref
            ):
                raise MembershipBindingError("employee principal authority unavailable")
            if (
                type(observed_is_member) is not bool
                or not app_id
                or principal.app_id != app_id
            ):
                raise MembershipBindingError("membership event evidence is not bound")
            authority = _Authority(
                app_id=principal.app_id,
                credential_ref=principal.credential_ref,
                member_groups=employee.member_groups,
            )
            probe = MembershipMutationRequest(
                tenant_key=tenant_key,
                chat_id=chat_id,
                agent_id=agent_id,
                requester_principal_id="system_membership_event",
                operation=(
                    MembershipOperation.ADD
                    if observed_is_member
                    else MembershipOperation.REMOVE
                ),
            )
            # The live observation may contradict the unordered event and
            # prove an ADD.  Acquire the ADD capability before probing, then
            # hold it until any resulting durable membership fact is anchored.
            lease = self._acquire_add_lease(
                replace(probe, operation=MembershipOperation.ADD)
            )
            prior_record = self.get(tenant_key, chat_id, agent_id)
            effect: MembershipEffect | None = None
            try:
                # Anchor the possible positive fact before the live probe.  If
                # observation discovers membership, Fire already has a durable
                # chat cleanup target before this lease can be released.
                add_probe = replace(probe, operation=MembershipOperation.ADD)
                effect = self._prepare(add_probe, authority)
                self._mark_executing(effect.effect_id)
                try:
                    observed = self._observe(probe, authority)
                except MembershipRemoteUnknown:
                    stable_state = (
                        MembershipState.ACTIVE
                        if observed_is_member
                        else MembershipState.ABSENT
                    )
                    if (
                        prior_record is not None
                        and prior_record.state is stable_state
                        and prior_record.confirmed_state is stable_state
                        and (chat_id in employee.member_groups)
                        is observed_is_member
                    ):
                        committed = self._commit_confirmed(
                            effect.effect_id,
                            observed_is_member,
                        )
                        return replace(committed, changed=False)
                    return self._mark_action_required(
                        effect.effect_id,
                        "event_observation_unknown",
                    )
                if observed:
                    return self._commit_confirmed(effect.effect_id, True)
                self._mark_action_required(
                    effect.effect_id,
                    "event_observed_absent",
                )
                removal = replace(probe, operation=MembershipOperation.REMOVE)
                effect = self._prepare(removal, authority)
                self._mark_executing(effect.effect_id)
                return self._commit_confirmed(
                    effect.effect_id,
                    False,
                )
            except BaseException:
                if effect is not None:
                    self._best_effort_terminal_disposition(effect.effect_id)
                raise
            finally:
                # Event reconciliation only observes remote state; it
                # never dispatches an ADD side effect.
                if lease is not None:
                    lease.release()

    def rebuild_projection(self) -> MembershipProjectionState:
        with self._mutex:
            self._state = rebuild_membership_projection(
                self._writer.replay()
            )
            return self._state

    def recover_pending(self) -> int:
        """Observe incomplete effects without replaying an external mutation."""

        self.rebuild_projection()
        pending = tuple(
            effect
            for effect in self.state.effects.values()
            if not effect.state.terminal
        )
        recovered = 0
        for snapshot in pending:
            with self._chat_lock(snapshot.chat_id):
                self.rebuild_projection()
                effect = self._state.effects.get(snapshot.effect_id)
                if effect is None or effect.state.terminal:
                    continue
                if effect.state is MembershipEffectState.PREPARED:
                    self._mark_action_required(
                        effect.effect_id,
                        "prepared_recovery_unknown",
                    )
                    recovered += 1
                    continue
                lease: EmployeeExternalMutationLease | None = None
                retirement_fenced = (
                    effect.operation is MembershipOperation.ADD
                    and self._external_mutation_gate is not None
                    and self._external_mutation_gate.is_fenced(
                        effect.tenant_key,
                        effect.agent_id,
                    )
                )
                try:
                    authority = self._authority_for_effect(
                        effect,
                        allow_retiring_add_observation=retirement_fenced,
                    )
                    request = MembershipMutationRequest(
                        tenant_key=effect.tenant_key,
                        chat_id=effect.chat_id,
                        agent_id=effect.agent_id,
                        requester_principal_id=effect.requester_principal_id,
                        operation=effect.operation,
                    )
                    if not retirement_fenced:
                        try:
                            lease = self._acquire_add_lease(request)
                        except MembershipBindingError:
                            gate = self._external_mutation_gate
                            if (
                                effect.operation is not MembershipOperation.ADD
                                or gate is None
                                or not gate.is_fenced(
                                    effect.tenant_key,
                                    effect.agent_id,
                                )
                            ):
                                raise
                            retirement_fenced = True
                            authority = self._authority_for_effect(
                                effect,
                                allow_retiring_add_observation=True,
                            )
                    observed = self._observe(request, authority)
                except (MembershipBindingError, MembershipRemoteUnknown):
                    self._mark_action_required(
                        effect.effect_id,
                        (
                            "retirement_reconciliation_required"
                            if retirement_fenced
                            else "recovery_observation_unknown"
                        ),
                    )
                else:
                    desired = effect.operation is MembershipOperation.ADD
                    if retirement_fenced:
                        # A fence forbids restoring an ADD projection from a
                        # stale or late observation.  Preserve the chat as a
                        # durable retirement cleanup target instead.
                        self._mark_action_required(
                            effect.effect_id,
                            "retirement_reconciliation_required",
                        )
                    elif observed is desired:
                        self._commit_confirmed(effect.effect_id, observed)
                    else:
                        self._mark_action_required(
                            effect.effect_id,
                            "recovery_observation_mismatch",
                        )
                finally:
                    # Recovery performs observation only.  If its Journal
                    # disposition fails, the original pending effect remains
                    # a durable cleanup obligation.
                    if lease is not None:
                        lease.release()
                recovered += 1
        return recovered

    def reconcile_projected_memberships(self) -> MembershipAuditSummary:
        """Observe every projected ACTIVE membership without mutating Feishu."""

        projection = self._hire.synchronize_projection()
        coordinates = tuple(
            sorted(
                (
                    employee.tenant_key,
                    chat_id,
                    employee.agent_id,
                )
                for employee in projection.employees.values()
                if employee.state in _AUDITABLE_EMPLOYEE_STATES
                and employee.worker_type is WorkerType.VISIBLE
                for chat_id in employee.member_groups
            )
        )
        confirmed = 0
        removed = 0
        degraded = 0
        for tenant_key, chat_id, agent_id in coordinates:
            with self._chat_lock(chat_id):
                projection = self._hire.synchronize_projection()
                employee = projection.employees.get(agent_id)
                if (
                    employee is None
                    or employee.tenant_key != tenant_key
                    or employee.state not in _AUDITABLE_EMPLOYEE_STATES
                    or employee.worker_type is not WorkerType.VISIBLE
                    or chat_id not in employee.member_groups
                    or not employee.bot_principal_id
                ):
                    continue
                principal = projection.bot_principals.get(
                    employee.bot_principal_id
                )
                if (
                    principal is None
                    or principal.tenant_key != tenant_key
                    or principal.agent_id != agent_id
                    or not principal.app_id
                    or not principal.credential_ref
                ):
                    degraded += 1
                    continue
                authority = _Authority(
                    app_id=principal.app_id,
                    credential_ref=principal.credential_ref,
                    member_groups=employee.member_groups,
                )
                request = MembershipMutationRequest(
                    tenant_key=tenant_key,
                    chat_id=chat_id,
                    agent_id=agent_id,
                    requester_principal_id="system_membership_recovery",
                    operation=MembershipOperation.ADD,
                )
                try:
                    lease = self._acquire_add_lease(request)
                except MembershipBindingError:
                    degraded += 1
                    continue
                try:
                    result = self._audit_projected_membership(
                        request,
                        authority,
                    )
                finally:
                    # Startup audit performs observation only; the lease
                    # prevents a stale positive result from crossing a Fire
                    # fence while its Journal disposition is committed.
                    if lease is not None:
                        lease.release()
                if result == "confirmed":
                    confirmed += 1
                elif result == "removed":
                    removed += 1
                else:
                    degraded += 1
        return MembershipAuditSummary(
            checked=len(coordinates),
            confirmed=confirmed,
            removed=removed,
            degraded=degraded,
        )

    def _audit_projected_membership(
        self,
        request: MembershipMutationRequest,
        authority: _Authority,
    ) -> str:
        try:
            observed = self._observe(request, authority)
        except MembershipRemoteUnknown:
            record = self.get(
                request.tenant_key,
                request.chat_id,
                request.agent_id,
            )
            if not (
                record is not None
                and record.state is MembershipState.DEGRADED
                and record.error_code == "recovery_observation_unknown"
            ):
                effect = self._prepare(request, authority)
                self._mark_executing(effect.effect_id)
                self._mark_action_required(
                    effect.effect_id,
                    "recovery_observation_unknown",
                )
            return "degraded"
        if observed:
            record = self.get(
                request.tenant_key,
                request.chat_id,
                request.agent_id,
            )
            if not (
                record is not None
                and record.state is MembershipState.ACTIVE
                and record.confirmed_state is MembershipState.ACTIVE
            ):
                effect = self._prepare(request, authority)
                self._mark_executing(effect.effect_id)
                self._commit_confirmed(effect.effect_id, True)
            return "confirmed"
        removal = replace(request, operation=MembershipOperation.REMOVE)
        effect = self._prepare(removal, authority)
        self._mark_executing(effect.effect_id)
        self._commit_confirmed(effect.effect_id, False)
        return "removed"

    def _authority_for_effect(
        self,
        effect: MembershipEffect,
        *,
        allow_retiring_add_observation: bool = False,
    ) -> _Authority:
        projection = self._hire.synchronize_projection()
        employee = projection.employees.get(effect.agent_id)
        if (
            employee is None
            or employee.tenant_key != effect.tenant_key
            or (
                employee.state is not EmployeeState.ACTIVE
                and not (
                    effect.operation is MembershipOperation.REMOVE
                    and employee.state in {EmployeeState.RETIRING, EmployeeState.ACTION_REQUIRED}
                )
                and not (
                    allow_retiring_add_observation
                    and effect.operation is MembershipOperation.ADD
                    and employee.state
                    in {EmployeeState.RETIRING, EmployeeState.ACTION_REQUIRED}
                )
            )
            or employee.worker_type is not WorkerType.VISIBLE
            or not employee.bot_principal_id
        ):
            raise MembershipBindingError("employee membership authority unavailable")
        principal = projection.bot_principals.get(employee.bot_principal_id)
        if (
            principal is None
            or principal.agent_id != effect.agent_id
            or principal.app_id != effect.app_id
            or not principal.credential_ref
        ):
            raise MembershipBindingError("employee principal authority unavailable")
        return _Authority(
            app_id=principal.app_id,
            credential_ref=principal.credential_ref,
            member_groups=employee.member_groups,
        )

    def _resolve_authority(
        self,
        request: MembershipMutationRequest,
    ) -> _Authority:
        projection = self._hire.synchronize_projection()
        employee = projection.employees.get(request.agent_id)
        if (
            employee is None
            or employee.tenant_key != request.tenant_key
            or (
                employee.state is not EmployeeState.ACTIVE
                and not (
                    request.operation is MembershipOperation.REMOVE
                    and employee.state in {EmployeeState.RETIRING, EmployeeState.ACTION_REQUIRED}
                    and request.requester_principal_id in self._admins
                )
            )
            or employee.worker_type is not WorkerType.VISIBLE
            or not employee.bot_principal_id
        ):
            raise MembershipBindingError("employee membership authority unavailable")
        principal = projection.bot_principals.get(employee.bot_principal_id)
        if (
            principal is None
            or principal.tenant_key != request.tenant_key
            or principal.agent_id != request.agent_id
            or not principal.app_id
            or not principal.credential_ref
        ):
            raise MembershipBindingError("employee principal authority unavailable")
        retiring_remove = (
            request.operation is MembershipOperation.REMOVE
            and employee.state in {EmployeeState.RETIRING, EmployeeState.ACTION_REQUIRED}
            and request.requester_principal_id in self._admins
        )
        owner = self._team_owner_resolver(request.chat_id)
        if not retiring_remove and self._team_active_resolver(request.chat_id) is not True:
            raise MembershipBindingError("membership team is not active")
        if request.requester_principal_id not in self._admins and (
            not owner or request.requester_principal_id != owner
        ):
            raise MembershipAuthorizationError("membership mutation is not authorized")
        return _Authority(
            app_id=principal.app_id,
            credential_ref=principal.credential_ref,
            member_groups=employee.member_groups,
        )

    def _observe(
        self,
        request: MembershipMutationRequest,
        authority: _Authority,
    ) -> bool:
        try:
            value = self._remote.is_member(
                chat_id=request.chat_id,
                agent_id=request.agent_id,
                app_id=authority.app_id,
                credential_ref=authority.credential_ref,
            )
        except MembershipRemoteUnknown:
            raise
        except Exception:
            raise MembershipRemoteUnknown("membership_observation_unknown") from None
        if type(value) is not bool:
            raise MembershipRemoteUnknown("membership_observation_unknown")
        return value

    def _prepare(
        self,
        request: MembershipMutationRequest,
        authority: _Authority,
    ) -> MembershipEffect:
        with self._hire.employee_dispatch_guard(), self._mutex, self._writer.transaction_guard():
            self._synchronize_unlocked()
            key = (request.tenant_key, request.chat_id, request.agent_id)
            current = self._state.records.get(key)
            epoch = 1 if current is None else current.membership_epoch + 1
            effect = MembershipEffect(
                schema_version=1,
                effect_id=membership_effect_id(
                    request.tenant_key,
                    request.chat_id,
                    request.agent_id,
                    request.operation,
                    epoch,
                ),
                tenant_key=request.tenant_key,
                chat_id=request.chat_id,
                agent_id=request.agent_id,
                app_id=authority.app_id,
                requester_principal_id=request.requester_principal_id,
                operation=request.operation,
                state=MembershipEffectState.PREPARED,
                membership_epoch=epoch,
                error_code="",
            )
            self._commit_unlocked(
                JournalEvent(
                    event_type=EFFECT_PREPARED,
                    aggregate_id=effect.effect_id,
                    payload={"effect": effect.to_dict()},
                ),
            )
            return effect

    def _mark_executing(self, effect_id: str) -> None:
        with self._hire.employee_dispatch_guard(), self._mutex, self._writer.transaction_guard():
            self._synchronize_unlocked()
            self._commit_unlocked(
                JournalEvent(
                    event_type=EFFECT_EXECUTING,
                    aggregate_id=effect_id,
                    payload={"effect_id": effect_id},
                ),
            )

    def _commit_confirmed(
        self,
        effect_id: str,
        observed: bool,
    ) -> MembershipMutationOutcome:
        with self._hire.employee_dispatch_guard(), self._mutex, self._writer.transaction_guard():
            self._synchronize_unlocked()
            effect = self._state.effects[effect_id]
            employee = self._hire.projection_state.employees[effect.agent_id]
            groups = [
                group
                for group in employee.member_groups
                if group != effect.chat_id
            ]
            if observed:
                groups.append(effect.chat_id)
            events = (
                JournalEvent(
                    event_type=EFFECT_COMMITTED,
                    aggregate_id=effect_id,
                    payload={
                        "effect_id": effect_id,
                        "observed_is_member": observed,
                    },
                ),
                JournalEvent(
                    event_type="employee.membership_changed",
                    aggregate_id=effect.agent_id,
                    payload={"member_groups": list(dict.fromkeys(groups))},
                ),
            )
            self._commit_unlocked(*events)
            record = self._state.records[
                (effect.tenant_key, effect.chat_id, effect.agent_id)
            ]
            return MembershipMutationOutcome(
                state=record.state,
                confirmed=True,
                changed=True,
                effect_id=effect_id,
            )

    def _mark_action_required(
        self,
        effect_id: str,
        error_code: str,
    ) -> MembershipMutationOutcome:
        with self._hire.employee_dispatch_guard(), self._mutex, self._writer.transaction_guard():
            self._synchronize_unlocked()
            self._commit_unlocked(
                JournalEvent(
                    event_type=EFFECT_ACTION_REQUIRED,
                    aggregate_id=effect_id,
                    payload={
                        "effect_id": effect_id,
                        "error_code": error_code,
                    },
                ),
            )
            effect = self._state.effects[effect_id]
            record = self._state.records[
                (effect.tenant_key, effect.chat_id, effect.agent_id)
            ]
            return MembershipMutationOutcome(
                state=record.state,
                confirmed=False,
                changed=False,
                effect_id=effect_id,
                error_code=error_code,
            )

    def _synchronize_unlocked(self) -> None:
        self._hire.synchronize_projection_unlocked()
        last = self._writer.get_last_frame()
        head = (0, "") if last is None else (last.sequence, last.frame_hash)
        if (self._state.cursor_sequence, self._state.cursor_hash) != head:
            self._state = rebuild_membership_projection(
                self._writer.replay()
            )

    def _commit_unlocked(self, *events: JournalEvent) -> None:
        result = commit_workforce_events_unlocked(
            self._writer,
            self._hire.projection_state,
            events,
        )
        reduce_membership_frame(self._state, result.frame)

    def _chat_lock(self, chat_id: str) -> threading.RLock:
        with self._chat_locks_guard:
            return self._chat_locks.setdefault(chat_id, threading.RLock())  # leaf lock: never held while acquiring a LockLevel lock


__all__ = [
    "EmployeeMembershipService",
    "MembershipAuthorizationError",
    "MembershipAuditSummary",
    "MembershipBindingError",
    "MembershipMutationOutcome",
    "MembershipMutationRequest",
    "MembershipServiceError",
]
