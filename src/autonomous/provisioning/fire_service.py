"""Journal-backed fail-closed employee retirement orchestration."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Protocol

from ..journal.frame import JournalEvent
from ..journal.writer import CommitState, JournalWriter
from .external_mutation_gate import EmployeeExternalMutationGate
from .fire_state import (
    FIRE_EFFECT_ORDER,
    DurableFireState,
    FireCleanupMode,
    FireEffectState,
    FirePhase,
    rebuild_fire_projection,
)

_EXTERNAL_MUTATION_FENCE_EVENT = "employee.external_mutation_fenced"
_UNSET = object()


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


@dataclass(frozen=True, slots=True)
class _PendingExternalMutationFence:
    request: EmployeeFireRequest
    target: EmployeeFireTarget
    intent_id: str


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
        drain_reconcile_interval_seconds: float = 0.5,
        monotonic: Callable[[], float] | None = None,
        external_mutation_gate: EmployeeExternalMutationGate | None = None,
        external_mutation_wait_seconds: float = 5.0,
    ) -> None:
        if set(effects) != set(FIRE_EFFECT_ORDER):
            raise ValueError("all fire effects must be configured exactly once")
        interval = float(drain_reconcile_interval_seconds)
        if not 0.5 <= interval <= 1.0:
            raise ValueError(
                "drain reconcile interval must be between 0.5 and 1.0 seconds"
            )
        if monotonic is not None and not callable(monotonic):
            raise TypeError("monotonic must be callable")
        mutation_wait = float(external_mutation_wait_seconds)
        if not math.isfinite(mutation_wait) or mutation_wait < 0:
            raise ValueError(
                "external mutation wait must be finite and non-negative"
            )
        self._writer = writer
        self._authority = authority
        self._effects = dict(effects)
        self._external_mutation_gate = external_mutation_gate
        self._external_mutation_wait_seconds = mutation_wait
        self._mutex = RLock()
        self._pending_drains: set[str] = set()
        self._drain_reconcile_interval = interval
        self._monotonic = monotonic or time.monotonic
        self._next_drain_reconcile_at = 0.0
        self._restore_external_mutation_fences()

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
            if state.phase is FirePhase.ACTION_REQUIRED:
                return self._retry_action_required(intent_id)
            return self.resume(intent_id)
        existing = self._coalesce_live_requests(request)
        if existing is not None:
            if existing.phase is FirePhase.ARCHIVED:
                raise FireServiceError("employee already archived")
            return self._retry_action_required(existing.intent_id)
        target = self._authority.resolve(request)
        if self._external_mutation_gate is not None:
            pending = self._anchor_external_mutation_fence(
                request,
                target,
                intent_id,
            )
            request = pending.request
            target = pending.target
            intent_id = pending.intent_id
            settled = self._external_mutation_gate.begin_retirement(
                target.tenant_key,
                target.agent_id,
                timeout_seconds=self._external_mutation_wait_seconds,
            )
            if not settled:
                raise FireServiceError("external mutation is still active")
            preflight = getattr(
                self._authority,
                "ensure_no_unresolved_external_mutation",
                None,
            )
            if callable(preflight):
                try:
                    preflight(target)
                except FireServiceError as exc:
                    if str(exc) == "external mutation outcome is unresolved":
                        self._authority.mark_action_required(target.agent_id)
                    raise
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
                if (
                    effect_type == "membership_cleanup"
                    and self._observe(state, effect_type) is not True
                ):
                    claimed = self._transition(
                        intent_id,
                        effect_type,
                        FireEffectState.EXECUTING,
                        expected_previous=FireEffectState.COMMITTED,
                    )
                    if not claimed:
                        return self._resume(intent_id)
                    return self._reconcile_executing(
                        self._require(intent_id),
                        effect_type,
                    )
                if effect_type == "credential_destroy":
                    self._authority.mark_credential_destroyed(target)
                continue
            if effect_state is FireEffectState.EXECUTING:
                return self._reconcile_executing(state, effect_type)
            if effect_state is FireEffectState.ACTION_REQUIRED:
                return state
            if effect_state is None:
                self._transition(intent_id, effect_type, FireEffectState.PREPARED)
            claimed = self._transition(
                intent_id,
                effect_type,
                FireEffectState.EXECUTING,
            )
            if not claimed:
                return self._resume(intent_id)
            state = self._require(intent_id)
            try:
                self._effects[effect_type].execute(state)
            except Exception:
                pass
            observed = self._observe(state, effect_type)
            if observed is not True:
                if effect_type == "execution_quiesce":
                    # Quiescing is an asynchronous retirement fence, not an
                    # ambiguous side effect. Keep EXECUTING recoverable until
                    # work and its owed responses are terminal.
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
        if effect_type in {"execution_quiesce", "membership_cleanup"}:
            # Both operations are idempotent retirement fences.  Re-execute
            # them so a transient actor wait or remote membership REMOVE can
            # converge instead of leaving an ACTION_REQUIRED tombstone with
            # no automated recovery path.
            try:
                self._effects[effect_type].execute(state)
            except Exception:
                return state
            if self._observe(state, effect_type) is not True:
                if effect_type == "execution_quiesce":
                    return state
                return self._action_required(
                    state,
                    effect_type,
                    "recovery_outcome_unknown",
                )
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
        """Advance pending retirement fences whose work has settled.

        Waiting fences remain unchanged and are omitted from the result so a
        caller can poll this method without turning an idle worker into a hot
        loop.  The Journal state is the cursor, so restart recovery preserves
        exactly the same bounded operation.
        """

        with self._mutex:
            if not self._pending_drains:
                return ()
            now = float(self._monotonic())
            if not math.isfinite(now):
                raise FireServiceError("drain reconcile clock is invalid")
            if now < self._next_drain_reconcile_at:
                return ()
            self._next_drain_reconcile_at = (
                now + self._drain_reconcile_interval
            )
            progressed: list[DurableFireState] = []
            for intent_id in tuple(sorted(self._pending_drains)):
                state = self._require(intent_id)
                if state.phase not in {
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
                    self._pending_drains.discard(intent_id)
                    continue
                if effect_state is FireEffectState.ACTION_REQUIRED:
                    current = self._retry_action_required(state.intent_id)
                else:
                    current = self.resume(state.intent_id)
                if current.last_sequence != state.last_sequence:
                    progressed.append(current)
            return tuple(progressed)

    def _track_drain(self, state: DurableFireState) -> None:
        if (
            state.phase in {FirePhase.RETIRING, FirePhase.ACTION_REQUIRED}
            and state.effect_state("execution_quiesce")
            in {FireEffectState.EXECUTING, FireEffectState.ACTION_REQUIRED}
        ):
            if state.intent_id not in self._pending_drains:
                self._next_drain_reconcile_at = 0.0
            self._pending_drains.add(state.intent_id)
        else:
            self._pending_drains.discard(state.intent_id)
            if not self._pending_drains:
                self._next_drain_reconcile_at = 0.0

    def _restore_committed_drain_fence(
        self,
        state: DurableFireState,
    ) -> None:
        if (
            state.phase not in {FirePhase.RETIRING, FirePhase.ACTION_REQUIRED}
            or state.effect_state("execution_quiesce")
            is not FireEffectState.COMMITTED
        ):
            return
        try:
            self._effects["execution_quiesce"].execute(state)
        except Exception as exc:
            raise FireServiceError(
                "committed drain actor fence could not be restored"
            ) from exc
        if self._observe(state, "execution_quiesce") is not True:
            raise FireServiceError("committed drain actor fence is unavailable")

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
            for pending in self._pending_external_mutation_fences():
                self._recover_pending_external_mutation_fence(pending)
            grouped: dict[tuple[str, str], list[DurableFireState]] = {}
            self._pending_drains.clear()
            self._next_drain_reconcile_at = 0.0
            for state in self._states().values():
                if state.phase is FirePhase.SUPERSEDED:
                    continue
                grouped.setdefault(
                    (state.tenant_key, state.agent_id),
                    [],
                ).append(state)
            for identity in sorted(grouped):
                state = self._coalesce_equivalent(grouped[identity])
                self._restore_committed_drain_fence(state)
                self._track_drain(state)
                if state.phase is FirePhase.RETIRING:
                    current = self.resume(state.intent_id)
                    self._track_drain(current)
                    recovered.append(current)
                elif state.phase is FirePhase.ACTION_REQUIRED:
                    if (
                        state.effect_state("execution_quiesce")
                        is FireEffectState.ACTION_REQUIRED
                        or state.effect_state("membership_cleanup")
                        is FireEffectState.ACTION_REQUIRED
                    ):
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
        if effect_type in {"execution_quiesce", "membership_cleanup"}:
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
        expected_previous: FireEffectState | None | object = _UNSET,
    ) -> bool:
        payload = {"effect_type": effect_type}
        if error_code:
            payload["error_code"] = error_code
        if expected_previous is _UNSET:
            expected_previous = {
                FireEffectState.PREPARED: None,
                FireEffectState.EXECUTING: FireEffectState.PREPARED,
                FireEffectState.COMMITTED: FireEffectState.EXECUTING,
                FireEffectState.ACTION_REQUIRED: FireEffectState.EXECUTING,
            }[state]
        return self._commit(
            JournalEvent(
                event_type=f"fire.effect.{state.value}",
                aggregate_id=intent_id,
                payload=payload,
            ),
            expected_effect_state=expected_previous,
        )

    def _commit(
        self,
        event: JournalEvent,
        *,
        expected_effect_state: FireEffectState | None | object = _UNSET,
    ) -> bool:
        with self._writer.transaction_guard():
            frames = tuple(self._writer.replay())
            current = rebuild_fire_projection(frames).get(event.aggregate_id)
            if event.event_type == "fire.effect.reconciled":
                effect_type = event.payload.get("effect_type")
                if current is None or effect_type not in FIRE_EFFECT_ORDER:
                    raise FireServiceError("invalid reconciled fire effect")
                actual = current.effect_state(effect_type)
                if actual is FireEffectState.COMMITTED:
                    return False
                if (
                    current.phase is not FirePhase.ACTION_REQUIRED
                    or actual is not FireEffectState.ACTION_REQUIRED
                ):
                    return False
            elif event.event_type in {
                f"fire.effect.{value.value}" for value in FireEffectState
            }:
                effect_type = event.payload.get("effect_type")
                try:
                    desired = FireEffectState(
                        event.event_type.removeprefix("fire.effect.")
                    )
                except ValueError as exc:
                    raise FireServiceError("invalid fire effect transition") from exc
                if current is None or effect_type not in FIRE_EFFECT_ORDER:
                    raise FireServiceError("invalid fire effect transition")
                actual = current.effect_state(effect_type)
                if actual is desired:
                    return False
                if (
                    current.phase in {FirePhase.ARCHIVED, FirePhase.SUPERSEDED}
                    or expected_effect_state is _UNSET
                    or actual is not expected_effect_state
                ):
                    return False
            elif event.event_type == "fire.superseded":
                if current is None:
                    raise FireServiceError("invalid fire supersession")
                if current.phase in {FirePhase.ARCHIVED, FirePhase.SUPERSEDED}:
                    return False
            last = frames[-1] if frames else None
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
        return True

    def _anchor_external_mutation_fence(
        self,
        request: EmployeeFireRequest,
        target: EmployeeFireTarget,
        intent_id: str,
    ) -> _PendingExternalMutationFence:
        payload = {
            "tenant_key": target.tenant_key,
            "agent_id": target.agent_id,
            "intent_id": intent_id,
            "employee": request.employee,
            "message_id": request.message_id,
            "chat_id": request.chat_id,
            "requester_principal_id": request.requester_principal_id,
            "drain": request.drain,
            "employee_name": target.employee_name,
            "bot_principal_id": target.bot_principal_id,
            "app_id": target.app_id,
            "credential_ref": target.credential_ref,
            "cleanup_mode": target.cleanup_mode.value,
        }
        aggregate_id = self._external_mutation_fence_aggregate_id(
            target.tenant_key,
            target.agent_id,
        )
        event = JournalEvent(
            event_type=_EXTERNAL_MUTATION_FENCE_EVENT,
            aggregate_id=aggregate_id,
            payload=payload,
        )
        with self._writer.transaction_guard():
            frames = tuple(self._writer.replay())
            existing = tuple(
                parsed
                for frame in frames
                for existing_event in frame.events
                if existing_event.event_type == _EXTERNAL_MUTATION_FENCE_EVENT
                and self._external_mutation_fence_identity(existing_event)
                == (target.tenant_key, target.agent_id)
                if (
                    parsed := self._external_mutation_fence_request(
                        existing_event
                    )
                )
                is not None
            )
            if existing:
                canonical = existing[0]
                if any(item != canonical for item in existing[1:]):
                    raise FireServiceError(
                        "external mutation fence authority is ambiguous"
                    )
                return canonical
            last = frames[-1] if frames else None
            sequence = 0 if last is None else last.sequence
            frame_hash = "" if last is None else last.frame_hash
            result = self._writer.commit(
                (event,),
                self._writer.get_aggregate_versions((aggregate_id,)),
                expected_head_sequence=sequence,
                expected_head_hash=frame_hash,
            )
        if result.state is not CommitState.ANCHORED:
            raise FireServiceError(
                "external mutation fence was not anchored"
            )
        return _PendingExternalMutationFence(
            request=request,
            target=target,
            intent_id=intent_id,
        )

    def _pending_external_mutation_fence_for(
        self,
        target: EmployeeFireTarget,
    ) -> _PendingExternalMutationFence | None:
        matches = tuple(
            pending
            for pending in self._pending_external_mutation_fences()
            if (
                pending.target.tenant_key,
                pending.target.agent_id,
            )
            == (target.tenant_key, target.agent_id)
        )
        if len(matches) > 1:
            raise FireServiceError("external mutation fence authority is ambiguous")
        return matches[0] if matches else None

    def _pending_external_mutation_fences(
        self,
    ) -> tuple[_PendingExternalMutationFence, ...]:
        active_identities = {
            (state.tenant_key, state.agent_id) for state in self._states().values()
        }
        pending: dict[tuple[str, str], _PendingExternalMutationFence] = {}
        for frame in self._writer.replay():
            for event in frame.events:
                if event.event_type != _EXTERNAL_MUTATION_FENCE_EVENT:
                    continue
                parsed = self._external_mutation_fence_request(event)
                if parsed is None:
                    continue
                identity = (parsed.target.tenant_key, parsed.target.agent_id)
                if identity in active_identities:
                    continue
                prior = pending.get(identity)
                if prior is not None and prior != parsed:
                    raise FireServiceError(
                        "external mutation fence authority is ambiguous"
                    )
                pending[identity] = parsed
        return tuple(pending[identity] for identity in sorted(pending))

    def _recover_pending_external_mutation_fence(
        self,
        pending: _PendingExternalMutationFence,
    ) -> bool:
        gate = self._external_mutation_gate
        if gate is None or not gate.begin_retirement(
            pending.target.tenant_key,
            pending.target.agent_id,
            timeout_seconds=self._external_mutation_wait_seconds,
        ):
            return False
        preflight = getattr(
            self._authority,
            "ensure_no_unresolved_external_mutation",
            None,
        )
        try:
            if callable(preflight):
                preflight(pending.target)
        except FireServiceError as exc:
            if str(exc) != "external mutation outcome is unresolved":
                raise
            self._authority.mark_action_required(pending.target.agent_id)
            return False
        try:
            self._authority.admit(
                pending.request,
                pending.target,
                pending.intent_id,
            )
        except FireServiceError as exc:
            if str(exc) == "employee retirement already in progress":
                return True
            if str(exc) != "external mutation outcome is unresolved":
                raise
            self._authority.mark_action_required(pending.target.agent_id)
            return False
        return True

    def _restore_external_mutation_fences(self) -> None:
        gate = self._external_mutation_gate
        if gate is None:
            return
        self.restore_external_mutation_fences(self._writer, gate)

    @classmethod
    def restore_external_mutation_fences(
        cls,
        writer: JournalWriter,
        gate: EmployeeExternalMutationGate,
    ) -> None:
        """Restore permanent fences before any mutator recovery can run."""

        identities: set[tuple[str, str]] = set()
        frames = tuple(writer.replay())
        for frame in frames:
            for event in frame.events:
                if event.event_type != _EXTERNAL_MUTATION_FENCE_EVENT:
                    continue
                identities.add(
                    cls._external_mutation_fence_identity(event)
                )
        for tenant_key, agent_id in identities:
            gate.restore_retirement_fence(tenant_key, agent_id)
        # Legacy retirement streams predate the explicit pre-admission marker.
        # Their requested fact is nevertheless a durable permanent fence.
        legacy_identities = {
            (state.tenant_key, state.agent_id)
            for state in rebuild_fire_projection(frames).values()
        }
        for tenant_key, agent_id in legacy_identities - identities:
            gate.restore_retirement_fence(tenant_key, agent_id)

    @staticmethod
    def _external_mutation_fence_identity(
        event: JournalEvent,
    ) -> tuple[str, str]:
        allowed_shapes = {
            frozenset({"tenant_key", "agent_id"}),
            frozenset(
                {
                    "tenant_key",
                    "agent_id",
                    "intent_id",
                    "employee",
                    "message_id",
                    "chat_id",
                    "requester_principal_id",
                    "drain",
                    "employee_name",
                    "bot_principal_id",
                    "app_id",
                    "credential_ref",
                    "cleanup_mode",
                }
            ),
        }
        if frozenset(event.payload) not in allowed_shapes:
            raise FireServiceError(
                "external mutation fence record is invalid"
            )
        tenant_key = event.payload.get("tenant_key")
        agent_id = event.payload.get("agent_id")
        if (
            not isinstance(tenant_key, str)
            or not tenant_key
            or tenant_key != tenant_key.strip()
            or not isinstance(agent_id, str)
            or not agent_id
            or agent_id != agent_id.strip()
        ):
            raise FireServiceError(
                "external mutation fence record is invalid"
            )
        return tenant_key, agent_id

    @classmethod
    def _external_mutation_fence_request(
        cls,
        event: JournalEvent,
    ) -> _PendingExternalMutationFence | None:
        tenant_key, agent_id = cls._external_mutation_fence_identity(event)
        if set(event.payload) == {"tenant_key", "agent_id"}:
            return None
        try:
            request = EmployeeFireRequest(
                employee=event.payload["employee"],
                tenant_key=tenant_key,
                message_id=event.payload["message_id"],
                chat_id=event.payload["chat_id"],
                requester_principal_id=event.payload["requester_principal_id"],
                drain=event.payload["drain"],
            )
            target = EmployeeFireTarget(
                tenant_key=tenant_key,
                agent_id=agent_id,
                employee_name=event.payload["employee_name"],
                bot_principal_id=event.payload["bot_principal_id"],
                app_id=event.payload["app_id"],
                credential_ref=event.payload["credential_ref"],
                cleanup_mode=FireCleanupMode(event.payload["cleanup_mode"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FireServiceError(
                "external mutation fence record is invalid"
            ) from exc
        intent_id = event.payload.get("intent_id")
        target_identity = (
            target.employee_name,
            target.bot_principal_id,
            target.app_id,
            target.credential_ref,
        )
        if (
            not isinstance(intent_id, str)
            or intent_id != cls._intent_id(request)
            or any(not isinstance(value, str) for value in target_identity)
            or request.employee not in {target.agent_id, target.employee_name}
            or not target.employee_name
            or target.employee_name != target.employee_name.strip()
            or (
                target.cleanup_mode
                in {FireCleanupMode.BOUND, FireCleanupMode.RECOVERABLE}
                and any(not value for value in target_identity[1:])
            )
            or (
                target.cleanup_mode
                in {FireCleanupMode.SAFE_ABORT, FireCleanupMode.EXTERNAL_UNKNOWN}
                and (target.bot_principal_id or target.credential_ref)
            )
            or event.aggregate_id
            != cls._external_mutation_fence_aggregate_id(tenant_key, agent_id)
        ):
            raise FireServiceError("external mutation fence record is invalid")
        return _PendingExternalMutationFence(
            request=request,
            target=target,
            intent_id=intent_id,
        )

    @staticmethod
    def _external_mutation_fence_aggregate_id(
        tenant_key: str,
        agent_id: str,
    ) -> str:
        raw = "\x00".join((tenant_key, agent_id))
        return (
            "employee-external-mutation-fence:"
            f"{hashlib.sha256(raw.encode()).hexdigest()}"
        )

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
