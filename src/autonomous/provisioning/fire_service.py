"""Journal-backed fail-closed employee retirement orchestration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from ..journal.frame import JournalEvent
from ..journal.writer import CommitState, JournalWriter
from .fire_state import (
    FIRE_EFFECT_ORDER,
    DurableFireState,
    FireCleanupMode,
    FireEffectState,
    FirePhase,
    rebuild_fire_projection,
)


class FireServiceError(RuntimeError):
    """Retirement could not safely progress."""


@dataclass(frozen=True, slots=True)
class EmployeeFireRequest:
    employee: str
    tenant_key: str
    message_id: str
    chat_id: str
    requester_principal_id: str
    drain: bool = False

    def __post_init__(self) -> None:
        for name in (
            "employee", "tenant_key", "message_id", "chat_id", "requester_principal_id"
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} is required")
        if type(self.drain) is not bool:
            raise ValueError("drain must be bool")


@dataclass(frozen=True, slots=True)
class EmployeeFireTarget:
    tenant_key: str
    agent_id: str
    employee_name: str
    bot_principal_id: str
    app_id: str
    credential_ref: str
    cleanup_mode: FireCleanupMode = FireCleanupMode.BOUND

    @property
    def pre_binding(self) -> bool:
        return self.cleanup_mode is not FireCleanupMode.BOUND


class FireAuthority(Protocol):
    def authorize_request(self, request: EmployeeFireRequest) -> None: ...

    def resolve(self, request: EmployeeFireRequest) -> EmployeeFireTarget: ...

    def admit(
        self,
        request: EmployeeFireRequest,
        target: EmployeeFireTarget,
        intent_id: str,
    ) -> EmployeeFireTarget: ...

    def mark_action_required(self, agent_id: str) -> None: ...

    def confirm_external_disposition(
        self,
        request: EmployeeFireRequest,
        state: DurableFireState,
        disposition_ref: str,
    ) -> None: ...

    def mark_credential_destroyed(self, target: EmployeeFireTarget) -> None: ...

    def mark_archived(
        self,
        agent_id: str,
        intent_id: str,
        *,
        external_disposition_confirmed: bool,
    ) -> None: ...


class FireEffectPort(Protocol):
    def execute(self, state: DurableFireState) -> None: ...

    def observe(self, state: DurableFireState) -> bool | None: ...


class EmployeeFireService:
    """Run one-way cleanup with anchored effect transitions and safe recovery."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        authority: FireAuthority,
        effects: dict[str, FireEffectPort],
    ) -> None:
        if set(effects) != set(FIRE_EFFECT_ORDER):
            raise ValueError("all fire effects must be configured exactly once")
        self._writer = writer
        self._authority = authority
        self._effects = dict(effects)
        self._mutex = RLock()
        self._pending_drains: set[str] = set()

    def start_fire(self, request: EmployeeFireRequest) -> DurableFireState:
        with self._mutex:
            state = self._start_fire(request)
            self._track_drain(state)
            return state

    def _start_fire(self, request: EmployeeFireRequest) -> DurableFireState:
        if not isinstance(request, EmployeeFireRequest):
            raise TypeError("request must be EmployeeFireRequest")
        self._authority.authorize_request(request)
        intent_id = self._intent_id(request)
        state = self._states().get(intent_id)
        if state is not None:
            if not self._matches_existing(state, request):
                raise FireServiceError("fire idempotency conflict")
            if state.phase is FirePhase.SUPERSEDED:
                raise FireServiceError("fire request was superseded")
            return self.resume(intent_id)
        existing = self._coalesce_live_requests(request)
        if existing is not None:
            if existing.phase is FirePhase.ARCHIVED:
                raise FireServiceError("employee already archived")
            return self._retry_action_required(existing.intent_id)
        target = self._authority.resolve(request)
        try:
            target = self._authority.admit(request, target, intent_id)
        except FireServiceError as exc:
            if str(exc) != "employee retirement already in progress":
                raise
            existing = self._coalesce_live_requests(request)
            if existing is None:
                raise
            if existing.phase is FirePhase.ARCHIVED:
                raise FireServiceError("employee already archived")
            return self._retry_action_required(existing.intent_id)
        state = self._require(intent_id)
        if not self._matches(state, request, target):
            raise FireServiceError("fire idempotency conflict")
        return self.resume(intent_id)

    def resume(self, intent_id: str) -> DurableFireState:
        with self._mutex:
            state = self._resume(intent_id)
            self._track_drain(state)
            return state

    def _resume(self, intent_id: str) -> DurableFireState:
        state = self._require(intent_id)
        if state.phase is FirePhase.ARCHIVED:
            return state
        if state.phase is FirePhase.SUPERSEDED:
            raise FireServiceError("fire request was superseded")
        if state.phase is FirePhase.ACTION_REQUIRED:
            return state
        target = EmployeeFireTarget(
            tenant_key=state.tenant_key,
            agent_id=state.agent_id,
            employee_name=state.employee_name,
            bot_principal_id=state.bot_principal_id,
            app_id=state.app_id,
            credential_ref=state.credential_ref,
            cleanup_mode=state.cleanup_mode,
        )
        if (
            state.cleanup_mode is FireCleanupMode.EXTERNAL_UNKNOWN
            and not state.external_disposition_confirmed
        ):
            effect_type = "credential_destroy"
            if state.effect_state(effect_type) is None:
                self._transition(intent_id, effect_type, FireEffectState.PREPARED)
                self._transition(intent_id, effect_type, FireEffectState.EXECUTING)
                return self._action_required(
                    self._require(intent_id),
                    effect_type,
                    "external_cleanup_authority_unavailable",
                )
            return self._require(intent_id)
        for effect_type in FIRE_EFFECT_ORDER:
            state = self._require(intent_id)
            effect_state = state.effect_state(effect_type)
            if (
                state.cleanup_mode
                in {FireCleanupMode.SAFE_ABORT, FireCleanupMode.EXTERNAL_UNKNOWN}
                and effect_type != "archive_move"
            ):
                if effect_state is FireEffectState.ACTION_REQUIRED:
                    return state
                if effect_state is None:
                    self._transition(
                        intent_id,
                        effect_type,
                        FireEffectState.PREPARED,
                    )
                    effect_state = FireEffectState.PREPARED
                if effect_state is FireEffectState.PREPARED:
                    self._transition(
                        intent_id,
                        effect_type,
                        FireEffectState.EXECUTING,
                    )
                    effect_state = FireEffectState.EXECUTING
                if effect_state is FireEffectState.EXECUTING:
                    self._transition(
                        intent_id,
                        effect_type,
                        FireEffectState.COMMITTED,
                    )
                continue
            if effect_state is FireEffectState.COMMITTED:
                if effect_type == "credential_destroy":
                    self._authority.mark_credential_destroyed(target)
                continue
            if effect_state is FireEffectState.EXECUTING:
                return self._reconcile_executing(state, effect_type)
            if effect_state is FireEffectState.ACTION_REQUIRED:
                return state
            if effect_state is None:
                self._transition(intent_id, effect_type, FireEffectState.PREPARED)
            self._transition(intent_id, effect_type, FireEffectState.EXECUTING)
            state = self._require(intent_id)
            try:
                self._effects[effect_type].execute(state)
            except Exception:
                pass
            observed = self._observe(state, effect_type)
            if observed is not True:
                if state.drain and effect_type == "execution_quiesce":
                    # A drain is an asynchronous wait, not an ambiguous side
                    # effect.  Keep the anchored EXECUTING frame recoverable;
                    # the department loop will reconcile it after the active
                    # assignment reaches a terminal state.
                    return self._require(intent_id)
                return self._action_required(state, effect_type, "outcome_unknown")
            self._transition(intent_id, effect_type, FireEffectState.COMMITTED)
            if effect_type == "credential_destroy":
                self._authority.mark_credential_destroyed(target)
        state = self._require(intent_id)
        self._authority.mark_archived(
            state.agent_id,
            intent_id,
            external_disposition_confirmed=state.external_disposition_confirmed,
        )
        return self._require(intent_id)

    def confirm_external_disposition(
        self,
        request: EmployeeFireRequest,
        disposition_ref: str,
    ) -> DurableFireState:
        with self._mutex:
            return self._confirm_external_disposition(
                request,
                disposition_ref,
            )

    def _confirm_external_disposition(
        self,
        request: EmployeeFireRequest,
        disposition_ref: str,
    ) -> DurableFireState:
        if not isinstance(request, EmployeeFireRequest):
            raise TypeError("request must be EmployeeFireRequest")
        if (
            not isinstance(disposition_ref, str)
            or not disposition_ref
            or disposition_ref != disposition_ref.strip()
        ):
            raise FireServiceError("external disposition reference is required")
        self._authority.authorize_request(request)
        candidates = [
            state
            for state in self._states().values()
            if state.tenant_key == request.tenant_key
            and request.employee in {state.agent_id, state.employee_name}
            and state.cleanup_mode is FireCleanupMode.EXTERNAL_UNKNOWN
            and state.phase is not FirePhase.SUPERSEDED
            and (
                state.external_disposition_confirmed
                or (
                    state.phase is FirePhase.ACTION_REQUIRED
                    and state.error_code
                    == "external_cleanup_authority_unavailable"
                )
            )
        ]
        pending = [
            state
            for state in candidates
            if not state.external_disposition_confirmed
        ]
        if pending:
            agent_ids = {state.agent_id for state in pending}
            if len(agent_ids) > 1:
                referenced = [
                    state
                    for state in pending
                    if (state.app_id or "NO_APP_FOUND") == disposition_ref
                ]
                agent_ids = {state.agent_id for state in referenced}
            if len(agent_ids) != 1:
                raise FireServiceError(
                    "external cleanup request was not uniquely resolved"
                )
            agent_id = next(iter(agent_ids))
            matches = [
                state for state in candidates if state.agent_id == agent_id
            ]
        else:
            referenced = [
                state
                for state in candidates
                if (state.app_id or "NO_APP_FOUND") == disposition_ref
            ]
            matches = referenced or candidates
        if len(matches) > 1:
            canonical = self._coalesce_equivalent(matches)
            matches = [canonical]
        if len(matches) != 1:
            raise FireServiceError(
                "external cleanup request was not uniquely resolved"
            )
        state = matches[0]
        expected_ref = state.app_id or "NO_APP_FOUND"
        if disposition_ref != expected_ref:
            raise FireServiceError("external disposition reference mismatch")
        if not state.external_disposition_confirmed:
            self._authority.confirm_external_disposition(
                request,
                state,
                disposition_ref,
            )
        return self.resume(state.intent_id)

    def _coalesce_live_requests(
        self,
        request: EmployeeFireRequest,
    ) -> DurableFireState | None:
        states = tuple(self._states().values())
        live = [
            state
            for state in states
            if state.tenant_key == request.tenant_key
            and request.employee in {state.agent_id, state.employee_name}
            and state.phase in {FirePhase.RETIRING, FirePhase.ACTION_REQUIRED}
        ]
        if not live:
            return None
        agent_ids = {state.agent_id for state in live}
        if len(agent_ids) != 1:
            raise FireServiceError("employee retirement authority is ambiguous")
        agent_id = next(iter(agent_ids))
        matches = [
            state
            for state in states
            if state.tenant_key == request.tenant_key
            and state.agent_id == agent_id
            and state.phase is not FirePhase.SUPERSEDED
        ]
        return self._coalesce_equivalent(matches)

    def _retry_action_required(self, intent_id: str) -> DurableFireState:
        state = self._require(intent_id)
        if state.phase is not FirePhase.ACTION_REQUIRED:
            return self.resume(intent_id)
        if state.error_code == "external_cleanup_authority_unavailable":
            return state
        failed_effects = [
            effect_type
            for effect_type, effect_state in state.effects
            if effect_state is FireEffectState.ACTION_REQUIRED
        ]
        if len(failed_effects) != 1:
            raise FireServiceError("fire recovery effect is ambiguous")
        effect_type = failed_effects[0]
        if state.drain and effect_type == "execution_quiesce":
            # Historical builds marked a long drain ACTION_REQUIRED.  Once
            # work is terminal, execute the idempotent actor retirement fence
            # before reconciling that durable effect as committed.
            try:
                self._effects[effect_type].execute(state)
            except Exception:
                return state
            if self._observe(state, effect_type) is not True:
                return state
        elif self._observe(state, effect_type) is not True:
            return state
        self._commit(
            JournalEvent(
                event_type="fire.effect.reconciled",
                aggregate_id=intent_id,
                payload={
                    "effect_type": effect_type,
                    "resolution_code": "observed_committed",
                },
            )
        )
        return self.resume(intent_id)

    def reconcile_draining(self) -> tuple[DurableFireState, ...]:
        """Advance drains whose active assignments have naturally finished.

        Waiting drains remain unchanged and are omitted from the result so a
        caller can poll this method without turning an idle worker into a hot
        loop.  The Journal state is the cursor, so restart recovery preserves
        exactly the same bounded operation.
        """

        with self._mutex:
            if not self._pending_drains:
                return ()
            progressed: list[DurableFireState] = []
            for intent_id in tuple(sorted(self._pending_drains)):
                state = self._require(intent_id)
                if not state.drain or state.phase not in {
                    FirePhase.RETIRING,
                    FirePhase.ACTION_REQUIRED,
                }:
                    self._pending_drains.discard(intent_id)
                    continue
                effect_state = state.effect_state("execution_quiesce")
                if effect_state not in {
                    FireEffectState.EXECUTING,
                    FireEffectState.ACTION_REQUIRED,
                }:
                    continue
                if effect_state is FireEffectState.ACTION_REQUIRED:
                    current = self._retry_action_required(state.intent_id)
                else:
                    current = self.resume(state.intent_id)
                if current.last_sequence != state.last_sequence:
                    progressed.append(current)
            return tuple(progressed)

    def _track_drain(self, state: DurableFireState) -> None:
        if state.drain and state.phase in {
            FirePhase.RETIRING,
            FirePhase.ACTION_REQUIRED,
        }:
            self._pending_drains.add(state.intent_id)
        else:
            self._pending_drains.discard(state.intent_id)

    def _coalesce_equivalent(
        self,
        states: list[DurableFireState],
    ) -> DurableFireState:
        coordinates = {
            (
                state.tenant_key,
                state.agent_id,
                state.bot_principal_id,
                state.app_id,
                state.credential_ref,
                state.cleanup_mode,
            )
            for state in states
        }
        if len(coordinates) != 1:
            raise FireServiceError("employee retirement authority is ambiguous")
        archived = [
            state for state in states if state.phase is FirePhase.ARCHIVED
        ]
        canonical = min(
            archived or states,
            key=lambda state: (state.requested_sequence, state.intent_id),
        )
        for duplicate in states:
            if (
                duplicate.intent_id == canonical.intent_id
                or duplicate.phase
                not in {FirePhase.RETIRING, FirePhase.ACTION_REQUIRED}
            ):
                continue
            self._commit(
                JournalEvent(
                    event_type="fire.superseded",
                    aggregate_id=duplicate.intent_id,
                    payload={"canonical_intent_id": canonical.intent_id},
                )
            )
        return self._require(canonical.intent_id)

    def recover(self) -> tuple[DurableFireState, ...]:
        with self._mutex:
            recovered: list[DurableFireState] = []
            grouped: dict[tuple[str, str], list[DurableFireState]] = {}
            self._pending_drains.clear()
            for state in self._states().values():
                if state.phase is FirePhase.SUPERSEDED:
                    continue
                grouped.setdefault(
                    (state.tenant_key, state.agent_id),
                    [],
                ).append(state)
            for identity in sorted(grouped):
                state = self._coalesce_equivalent(grouped[identity])
                self._track_drain(state)
                if state.phase is FirePhase.RETIRING:
                    current = self.resume(state.intent_id)
                    self._track_drain(current)
                    recovered.append(current)
                elif state.phase is FirePhase.ACTION_REQUIRED:
                    if state.drain:
                        current = self._retry_action_required(state.intent_id)
                        self._track_drain(current)
                        if current.last_sequence != state.last_sequence:
                            recovered.append(current)
                    else:
                        self._authority.mark_action_required(state.agent_id)
            return tuple(recovered)

    def list_states(self) -> tuple[DurableFireState, ...]:
        """Return a stable read-only snapshot for admin diagnostics."""

        with self._mutex:
            return tuple(
                sorted(
                    self._states().values(),
                    key=lambda state: (state.requested_sequence, state.intent_id),
                )
            )

    def _reconcile_executing(
        self,
        state: DurableFireState,
        effect_type: str,
    ) -> DurableFireState:
        if state.drain and effect_type == "execution_quiesce":
            try:
                self._effects[effect_type].execute(state)
            except Exception:
                return self._action_required(
                    state,
                    effect_type,
                    "recovery_outcome_unknown",
                )
            if self._observe(state, effect_type) is not True:
                return state
        elif self._observe(state, effect_type) is not True:
            return self._action_required(state, effect_type, "recovery_outcome_unknown")
        self._transition(state.intent_id, effect_type, FireEffectState.COMMITTED)
        return self.resume(state.intent_id)

    def _observe(self, state: DurableFireState, effect_type: str) -> bool | None:
        try:
            value = self._effects[effect_type].observe(state)
        except Exception:
            return None
        return value if type(value) is bool else None

    def _action_required(
        self,
        state: DurableFireState,
        effect_type: str,
        error_code: str,
    ) -> DurableFireState:
        self._transition(
            state.intent_id,
            effect_type,
            FireEffectState.ACTION_REQUIRED,
            error_code=error_code,
        )
        self._authority.mark_action_required(state.agent_id)
        return self._require(state.intent_id)

    def _transition(
        self,
        intent_id: str,
        effect_type: str,
        state: FireEffectState,
        *,
        error_code: str = "",
    ) -> None:
        payload = {"effect_type": effect_type}
        if error_code:
            payload["error_code"] = error_code
        self._commit(
            JournalEvent(
                event_type=f"fire.effect.{state.value}",
                aggregate_id=intent_id,
                payload=payload,
            )
        )

    def _commit(self, event: JournalEvent) -> None:
        with self._writer.transaction_guard():
            last = self._writer.get_last_frame()
            sequence = 0 if last is None else last.sequence
            frame_hash = "" if last is None else last.frame_hash
            result = self._writer.commit(
                (event,),
                self._writer.get_aggregate_versions((event.aggregate_id,)),
                expected_head_sequence=sequence,
                expected_head_hash=frame_hash,
            )
        if result.state is not CommitState.ANCHORED:
            raise FireServiceError("fire transition was not anchored")
        self._states()

    def _states(self):
        return rebuild_fire_projection(tuple(self._writer.replay()))

    def _require(self, intent_id: str) -> DurableFireState:
        state = self._states().get(intent_id)
        if state is None:
            raise FireServiceError("unknown fire request")
        return state

    @staticmethod
    def _intent_id(request: EmployeeFireRequest) -> str:
        raw = "\x00".join((request.tenant_key, request.message_id))
        return f"fire_{hashlib.sha256(raw.encode()).hexdigest()}"

    @staticmethod
    def _matches(
        state: DurableFireState,
        request: EmployeeFireRequest,
        target: EmployeeFireTarget,
    ) -> bool:
        return (
            state.tenant_key == request.tenant_key
            and state.message_id == request.message_id
            and state.chat_id == request.chat_id
            and state.requester_principal_id == request.requester_principal_id
            and state.drain is request.drain
            and state.agent_id == target.agent_id
            and state.bot_principal_id == target.bot_principal_id
            and state.app_id == target.app_id
            and state.credential_ref == target.credential_ref
            and state.cleanup_mode is target.cleanup_mode
        )

    @staticmethod
    def _matches_existing(
        state: DurableFireState,
        request: EmployeeFireRequest,
    ) -> bool:
        return (
            state.tenant_key == request.tenant_key
            and state.message_id == request.message_id
            and state.chat_id == request.chat_id
            and state.requester_principal_id == request.requester_principal_id
            and state.drain is request.drain
            and request.employee in {state.agent_id, state.employee_name}
        )


__all__ = [
    "EmployeeFireRequest",
    "EmployeeFireService",
    "EmployeeFireTarget",
    "FireAuthority",
    "FireEffectPort",
    "FireServiceError",
]
