"""Pure Journal projection for employee execution attempts."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

from ..authorization import EmployeeAuthorizationScope
from ..journal.frame import GENESIS_HASH, JournalEvent, TransactionFrame
from .models import DispatchBinding, GatewayExecutionStatus

ATTEMPT_BOUND = "employee.execution_attempt.bound"
ATTEMPT_DISPATCH_COMMITTED = "employee.execution_attempt.dispatch_committed"
ATTEMPT_CANCEL_REQUESTED = "employee.execution_attempt.cancel_requested"
ATTEMPT_TERMINAL = "employee.execution_attempt.terminal"
_GATEWAY_EVENTS = frozenset(
    {
        ATTEMPT_BOUND,
        ATTEMPT_DISPATCH_COMMITTED,
        ATTEMPT_CANCEL_REQUESTED,
        ATTEMPT_TERMINAL,
    }
)
_CURRENT_TEAM_BINDING_FIELDS = frozenset(
    {"team_chat_id", "team_identity", "team_root_identity"}
)
_LEGACY_SLOCK_BINDING_FIELDS = frozenset(
    {"slock_chat_id", "slock_engine_identity", "slock_root_identity"}
)
_CURRENT_BINDING_FIELDS = frozenset(DispatchBinding.__dataclass_fields__)
_CURRENT_SLOCK_BINDING_FIELDS = (
    _CURRENT_BINDING_FIELDS - _CURRENT_TEAM_BINDING_FIELDS
) | _LEGACY_SLOCK_BINDING_FIELDS
_SCOPE_BINDING_FIELDS = frozenset(
    {"authorization_scope", "source_requester_principal_id"}
)
_LEGACY_TEAM_BINDING_FIELDS = (
    _CURRENT_BINDING_FIELDS - _SCOPE_BINDING_FIELDS
)
_LEGACY_SLOCK_BINDING_FIELDS_EXACT = (
    _LEGACY_TEAM_BINDING_FIELDS - _CURRENT_TEAM_BINDING_FIELDS
) | _LEGACY_SLOCK_BINDING_FIELDS


class GatewayProjectionError(RuntimeError):
    """Anchored attempt history violates the exact lifecycle contract."""


@dataclass(frozen=True, slots=True)
class AttemptLifecycleRecord:
    binding: DispatchBinding
    bound_sequence: int = 0
    dispatch_committed: bool = False
    dispatch_sequence: int = 0
    cancel_requested: bool = False
    cancel_epoch: int = 0
    cancel_requester_principal_id: str = ""
    cancel_command_acceptance_id: str = ""
    cancel_requested_at: str = ""
    cancel_sequence: int = 0
    terminal_status: str = ""
    terminal_epoch: int = 0
    result_digest: str = ""
    history_record_id: str = ""
    ended_at: str = ""
    terminal_sequence: int = 0


@dataclass(slots=True)
class GatewayProjectionState:
    attempts: dict[str, AttemptLifecycleRecord] = field(default_factory=dict)
    attempt_by_acceptance_id: dict[str, str] = field(default_factory=dict)
    attempt_by_permit_id: dict[str, str] = field(default_factory=dict)
    attempt_by_ingress_identity: dict[tuple[str, str], str] = field(
        default_factory=dict
    )
    frame_hashes: dict[int, str] = field(default_factory=dict)
    cursor_sequence: int = 0
    cursor_hash: str = ""

    def clone(self) -> GatewayProjectionState:
        return copy.deepcopy(self)


def is_gateway_event(event_type: str) -> bool:
    return event_type in _GATEWAY_EVENTS


def _dispatch_binding_from_journal(value: object) -> DispatchBinding:
    """Normalize the one authenticated pre-team binding schema during replay."""

    if isinstance(value, dict) and set(value) in {
        _CURRENT_SLOCK_BINDING_FIELDS,
        _LEGACY_SLOCK_BINDING_FIELDS_EXACT,
    }:
        normalized = dict(value)
        normalized["team_chat_id"] = normalized.pop("slock_chat_id")
        normalized["team_identity"] = normalized.pop("slock_engine_identity")
        normalized["team_root_identity"] = normalized.pop("slock_root_identity")
        value = normalized
    if isinstance(value, dict) and set(value) == _LEGACY_TEAM_BINDING_FIELDS:
        normalized = dict(value)
        normalized["authorization_scope"] = (
            EmployeeAuthorizationScope.MANAGED_GROUP.value
        )
        normalized["source_requester_principal_id"] = normalized[
            "requester_principal_id"
        ]
        value = normalized
    return DispatchBinding.from_dict(value)


def _reduce_gateway_event(
    state: GatewayProjectionState,
    event: JournalEvent,
    *,
    frame_sequence: int = 0,
) -> None:
    if event.event_type == ATTEMPT_BOUND:
        _reduce_bound(state, event, frame_sequence)
    elif event.event_type == ATTEMPT_DISPATCH_COMMITTED:
        _reduce_dispatch_committed(state, event, frame_sequence)
    elif event.event_type == ATTEMPT_CANCEL_REQUESTED:
        _reduce_cancel_requested(state, event, frame_sequence)
    elif event.event_type == ATTEMPT_TERMINAL:
        _reduce_terminal(state, event, frame_sequence)
    else:
        raise GatewayProjectionError(f"unknown gateway event: {event.event_type}")


def reduce_gateway_frame(
    state: GatewayProjectionState,
    frame: TransactionFrame,
) -> None:
    """Apply one authenticated frame with dispatch and terminal atomicity."""

    if not isinstance(frame, TransactionFrame) or not frame.committed:
        raise GatewayProjectionError("gateway projection requires committed frame")
    known_hash = state.frame_hashes.get(frame.sequence)
    if known_hash is not None:
        if known_hash == frame.frame_hash:
            return
        raise GatewayProjectionError("conflicting replayed gateway frame")
    if frame.sequence != state.cursor_sequence + 1:
        raise GatewayProjectionError("gateway frame sequence is not contiguous")
    expected_previous = state.cursor_hash or GENESIS_HASH
    if frame.previous_hash != expected_previous:
        raise GatewayProjectionError("gateway frame hash chain mismatch")
    event_values = frame.events
    bound = [event for event in event_values if event.event_type == ATTEMPT_BOUND]
    dispatched = [
        event
        for event in event_values
        if event.event_type == ATTEMPT_DISPATCH_COMMITTED
    ]
    terminal = [event for event in event_values if event.event_type == ATTEMPT_TERMINAL]
    if len(bound) != len({event.aggregate_id for event in bound}) or len(
        dispatched
    ) != len({event.aggregate_id for event in dispatched}) or len(terminal) != len(
        {event.aggregate_id for event in terminal}
    ):
        raise GatewayProjectionError("attempt frame contains duplicate lifecycle event")
    if {event.aggregate_id for event in bound} != {
        event.aggregate_id for event in dispatched
    }:
        raise GatewayProjectionError(
            "new attempt binding and dispatch commit require the same frame"
        )
    _validate_cross_domain_atomicity(state, event_values, bound)
    isolated = state.clone()
    for event in event_values:
        if is_gateway_event(event.event_type):
            _reduce_gateway_event(
                isolated,
                event,
                frame_sequence=frame.sequence,
            )
    state.attempts = isolated.attempts
    state.attempt_by_acceptance_id = isolated.attempt_by_acceptance_id
    state.attempt_by_permit_id = isolated.attempt_by_permit_id
    state.attempt_by_ingress_identity = isolated.attempt_by_ingress_identity
    state.cursor_sequence = frame.sequence
    state.cursor_hash = frame.frame_hash
    state.frame_hashes[frame.sequence] = frame.frame_hash


def _validate_cross_domain_atomicity(
    state: GatewayProjectionState,
    events: tuple[JournalEvent, ...],
    bound: list[JournalEvent],
) -> None:
    for event in bound:
        try:
            binding = _dispatch_binding_from_journal(event.payload["binding"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GatewayProjectionError("invalid attempt binding") from exc
        router_dispatch = [
            candidate
            for candidate in events
            if candidate.event_type == "employee.ingress.router_dispatching"
            and candidate.aggregate_id == binding.ingress_aggregate_id
            and candidate.payload == {"acceptance_id": binding.acceptance_id}
        ]
        if len(router_dispatch) != 1:
            raise GatewayProjectionError(
                "attempt dispatch requires one Router dispatching event"
            )
    for event in events:
        if event.event_type != ATTEMPT_TERMINAL:
            continue
        attempt_id = event.aggregate_id
        record = state.attempts.get(attempt_id)
        if record is None:
            raise GatewayProjectionError("terminal references unknown attempt")
        status = event.payload.get("status")
        history = [
            candidate
            for candidate in events
            if candidate.event_type == "employee.history.recorded"
            and candidate.payload.get("attempt_id") == attempt_id
            and candidate.aggregate_id == event.payload.get("history_record_id")
            and candidate.payload.get("record_id")
            == event.payload.get("history_record_id")
        ]
        router_terminal = [
            candidate
            for candidate in events
            if candidate.event_type == "employee.ingress.router_terminal"
            and candidate.aggregate_id == record.binding.ingress_aggregate_id
            and candidate.payload
            == {
                "acceptance_id": record.binding.acceptance_id,
                "reason_code": status,
            }
        ]
        if len(history) != 1 or len(router_terminal) != 1:
            raise GatewayProjectionError(
                "attempt terminal requires history and Router terminal in same frame"
            )


def _reduce_bound(
    state: GatewayProjectionState,
    event: JournalEvent,
    frame_sequence: int,
) -> None:
    if set(event.payload) != {"binding"}:
        raise GatewayProjectionError("attempt binding must use exact schema")
    try:
        binding = _dispatch_binding_from_journal(event.payload["binding"])
    except (TypeError, ValueError) as exc:
        raise GatewayProjectionError("invalid attempt binding") from exc
    if binding.attempt_id != event.aggregate_id:
        raise GatewayProjectionError("attempt binding aggregate mismatch")
    existing = state.attempts.get(binding.attempt_id)
    if existing is not None:
        if existing.binding == binding:
            return
        raise GatewayProjectionError("conflicting attempt binding")
    reverse_values = (
        (state.attempt_by_acceptance_id, binding.acceptance_id),
        (state.attempt_by_permit_id, binding.permit_id),
        (
            state.attempt_by_ingress_identity,
            (binding.ingress_aggregate_id, binding.envelope_id),
        ),
    )
    if any(key in index for index, key in reverse_values):
        raise GatewayProjectionError("attempt reuses a dispatch identity")
    state.attempts[binding.attempt_id] = AttemptLifecycleRecord(
        binding=binding,
        bound_sequence=frame_sequence,
    )
    state.attempt_by_acceptance_id[binding.acceptance_id] = binding.attempt_id
    state.attempt_by_permit_id[binding.permit_id] = binding.attempt_id
    state.attempt_by_ingress_identity[
        (binding.ingress_aggregate_id, binding.envelope_id)
    ] = binding.attempt_id


def _reduce_dispatch_committed(
    state: GatewayProjectionState,
    event: JournalEvent,
    frame_sequence: int,
) -> None:
    if set(event.payload) != {"attempt_id", "permit_id"}:
        raise GatewayProjectionError("dispatch commit must use exact schema")
    attempt_id = event.payload.get("attempt_id")
    permit_id = event.payload.get("permit_id")
    record = state.attempts.get(str(attempt_id))
    if record is None or event.aggregate_id != attempt_id:
        raise GatewayProjectionError("dispatch commit references unknown attempt")
    if permit_id != record.binding.permit_id:
        raise GatewayProjectionError("dispatch permit binding mismatch")
    if record.terminal_status:
        raise GatewayProjectionError("dispatch commit follows terminal")
    if record.dispatch_committed:
        return
    state.attempts[attempt_id] = AttemptLifecycleRecord(
        binding=record.binding,
        bound_sequence=record.bound_sequence,
        dispatch_committed=True,
        dispatch_sequence=frame_sequence,
    )


def _reduce_terminal(
    state: GatewayProjectionState,
    event: JournalEvent,
    frame_sequence: int,
) -> None:
    expected = {
        "attempt_id",
        "terminal_epoch",
        "status",
        "result_digest",
        "history_record_id",
        "ended_at",
    }
    if set(event.payload) != expected:
        raise GatewayProjectionError("attempt terminal must use exact schema")
    payload = event.payload
    attempt_id = payload.get("attempt_id")
    record = state.attempts.get(str(attempt_id))
    if record is None or event.aggregate_id != attempt_id:
        raise GatewayProjectionError("terminal references unknown attempt")
    if not record.dispatch_committed:
        raise GatewayProjectionError("terminal precedes dispatch commit")
    try:
        status = GatewayExecutionStatus(payload["status"])
    except (TypeError, ValueError) as exc:
        raise GatewayProjectionError("invalid attempt terminal status") from exc
    epoch = payload["terminal_epoch"]
    if type(epoch) is not int or epoch != 1:
        raise GatewayProjectionError("invalid attempt terminal epoch")
    values = (
        status.value,
        epoch,
        payload["result_digest"],
        payload["history_record_id"],
        payload["ended_at"],
    )
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(payload["result_digest"])) is None
        or re.fullmatch(r"hist_[A-Za-z0-9][A-Za-z0-9_-]*", str(payload["history_record_id"]))
        is None
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
            str(payload["ended_at"]),
        )
        is None
    ):
        raise GatewayProjectionError("invalid attempt terminal metadata")
    if record.terminal_status:
        existing = (
            record.terminal_status,
            record.terminal_epoch,
            record.result_digest,
            record.history_record_id,
            record.ended_at,
        )
        if existing == values:
            return
        raise GatewayProjectionError("conflicting attempt terminal")
    state.attempts[attempt_id] = AttemptLifecycleRecord(
        binding=record.binding,
        bound_sequence=record.bound_sequence,
        dispatch_committed=True,
        dispatch_sequence=record.dispatch_sequence,
        cancel_requested=record.cancel_requested,
        cancel_epoch=record.cancel_epoch,
        cancel_requester_principal_id=record.cancel_requester_principal_id,
        cancel_command_acceptance_id=record.cancel_command_acceptance_id,
        cancel_requested_at=record.cancel_requested_at,
        cancel_sequence=record.cancel_sequence,
        terminal_status=status.value,
        terminal_epoch=epoch,
        result_digest=payload["result_digest"],
        history_record_id=payload["history_record_id"],
        ended_at=payload["ended_at"],
        terminal_sequence=frame_sequence,
    )


def _reduce_cancel_requested(
    state: GatewayProjectionState,
    event: JournalEvent,
    frame_sequence: int,
) -> None:
    expected = {
        "attempt_id",
        "cancel_epoch",
        "requester_principal_id",
        "command_acceptance_id",
        "requested_at",
    }
    if set(event.payload) != expected:
        raise GatewayProjectionError("attempt cancellation must use exact schema")
    payload = event.payload
    attempt_id = payload.get("attempt_id")
    record = state.attempts.get(str(attempt_id))
    if record is None or event.aggregate_id != attempt_id:
        raise GatewayProjectionError("cancellation references unknown attempt")
    if not record.dispatch_committed or record.terminal_status:
        raise GatewayProjectionError("cancellation requires a live committed attempt")
    values = (
        payload.get("cancel_epoch"),
        payload.get("requester_principal_id"),
        payload.get("command_acceptance_id"),
        payload.get("requested_at"),
    )
    if (
        values[0] != 1
        or not isinstance(values[1], str)
        or not values[1]
        or not isinstance(values[2], str)
        or not values[2].startswith("acc_")
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
            str(values[3]),
        )
        is None
    ):
        raise GatewayProjectionError("invalid attempt cancellation metadata")
    if record.cancel_requested:
        existing = (
            record.cancel_epoch,
            record.cancel_requester_principal_id,
            record.cancel_command_acceptance_id,
            record.cancel_requested_at,
        )
        if existing == values:
            return
        raise GatewayProjectionError("attempt cancellation already requested")
    state.attempts[attempt_id] = AttemptLifecycleRecord(
        binding=record.binding,
        bound_sequence=record.bound_sequence,
        dispatch_committed=True,
        dispatch_sequence=record.dispatch_sequence,
        cancel_requested=True,
        cancel_epoch=1,
        cancel_requester_principal_id=str(values[1]),
        cancel_command_acceptance_id=str(values[2]),
        cancel_requested_at=str(values[3]),
        cancel_sequence=frame_sequence,
    )


__all__ = [
    "ATTEMPT_BOUND",
    "ATTEMPT_CANCEL_REQUESTED",
    "ATTEMPT_DISPATCH_COMMITTED",
    "ATTEMPT_TERMINAL",
    "AttemptLifecycleRecord",
    "GatewayProjectionError",
    "GatewayProjectionState",
    "is_gateway_event",
    "reduce_gateway_frame",
]
