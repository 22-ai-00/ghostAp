"""Journal-backed projection for durable employee ingress metadata."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

from ..journal.blob_store import BlobRef
from ..journal.frame import JournalEvent
from .models import EmployeeIngressMetadata, IngressAcceptance, IngressDisposition


class IngressProjectionError(RuntimeError):
    """The ingress Journal history is inconsistent or malformed."""


@dataclass(frozen=True, slots=True)
class IngressRecord:
    """Safe durable metadata for one canonical employee ingress acceptance."""

    aggregate_id: str
    metadata: EmployeeIngressMetadata
    acceptance: IngressAcceptance
    blob_ref: BlobRef
    transport_message_proof: bool = False
    disposition: IngressDisposition | None = None
    payload_tombstoned: bool = False

    @property
    def employee_key(self) -> tuple[str, str]:
        return (self.metadata.tenant_key, self.metadata.agent_id)

    @property
    def terminal(self) -> bool:
        return self.disposition is not None and self.disposition.state in {
            "ignored",
            "rejected",
            "terminal",
        }


@dataclass
class IngressProjectionState:
    """Replayable ingress indexes plus recovery-only closed employee state."""

    by_dedup_key: dict[str, IngressRecord] = field(default_factory=dict)
    by_acceptance_id: dict[str, IngressRecord] = field(default_factory=dict)
    closed_employees: set[tuple[str, str]] = field(default_factory=set)
    cursor_sequence: int = 0
    cursor_hash: str = ""

    def clone(self) -> IngressProjectionState:
        return copy.deepcopy(self)


_INGRESS_EVENT_TYPES = frozenset(
    {
        "employee.ingress.accepted",
        "employee.ingress.closed",
        "employee.ingress.dispositioned",
        "employee.ingress.payload_tombstoned",
    }
)


def is_ingress_event(event_type: str) -> bool:
    return event_type in _INGRESS_EVENT_TYPES


def reduce_ingress_event(
    state: IngressProjectionState,
    event: JournalEvent,
    *,
    frame_sequence: int,
    frame_hash: str,
) -> None:
    """Apply one ingress event using only authenticated frame coordinates."""

    if event.event_type == "employee.ingress.accepted":
        _reduce_accepted(state, event, frame_sequence, frame_hash)
    elif event.event_type == "employee.ingress.closed":
        _reduce_closed(state, event)
    elif event.event_type == "employee.ingress.dispositioned":
        _reduce_dispositioned(state, event, frame_sequence, frame_hash)
    elif event.event_type == "employee.ingress.payload_tombstoned":
        _reduce_payload_tombstoned(state, event)
    else:
        raise IngressProjectionError(f"unknown ingress event: {event.event_type}")


def _reduce_closed(state: IngressProjectionState, event: JournalEvent) -> None:
    payload = event.payload
    if set(payload) != {"tenant_key", "agent_id", "reason_code", "closed_at"}:
        raise IngressProjectionError("invalid employee.ingress.closed payload")
    tenant_key = payload.get("tenant_key")
    agent_id = payload.get("agent_id")
    reason_code = payload.get("reason_code")
    closed_at = payload.get("closed_at")
    if (
        not isinstance(tenant_key, str)
        or not tenant_key
        or not isinstance(agent_id, str)
        or not agent_id
        or event.aggregate_id != f"employee-ingress:{tenant_key}:{agent_id}"
        or not isinstance(reason_code, str)
        or not reason_code
        or not isinstance(closed_at, str)
        or not closed_at
    ):
        raise IngressProjectionError("invalid employee.ingress.closed payload")
    state.closed_employees.add((tenant_key, agent_id))


def _reduce_accepted(
    state: IngressProjectionState,
    event: JournalEvent,
    frame_sequence: int,
    frame_hash: str,
) -> None:
    payload = event.payload
    current_fields = {
        "metadata",
        "acceptance_id",
        "accepted_at",
        "blob_ref",
        "transport_message_proof",
    }
    legacy_fields = current_fields - {"transport_message_proof"}
    if frozenset(payload) not in {
        frozenset(current_fields),
        frozenset(legacy_fields),
    }:
        raise IngressProjectionError("invalid employee.ingress.accepted payload")
    transport_message_proof = payload.get("transport_message_proof", False)
    if type(transport_message_proof) is not bool:
        raise IngressProjectionError("invalid employee ingress transport proof")
    try:
        metadata = EmployeeIngressMetadata.from_dict(payload["metadata"])
        blob_ref = BlobRef.from_dict(payload["blob_ref"])
        acceptance = IngressAcceptance(
            schema_version=1,
            acceptance_id=payload["acceptance_id"],
            envelope_id=metadata.envelope_id,
            dedup_key=metadata.dedup_key,
            semantic_digest=metadata.semantic_digest,
            journal_sequence=frame_sequence,
            journal_frame_hash=frame_hash,
            accepted_at=payload["accepted_at"],
        )
    except (TypeError, ValueError) as exc:
        raise IngressProjectionError("invalid employee ingress acceptance") from exc
    if event.aggregate_id != metadata.dedup_key:
        raise IngressProjectionError("ingress aggregate does not match dedup key")
    if metadata.dedup_key in state.by_dedup_key:
        raise IngressProjectionError(f"duplicate ingress dedup key: {metadata.dedup_key}")
    if acceptance.acceptance_id in state.by_acceptance_id:
        raise IngressProjectionError(
            f"duplicate ingress acceptance: {acceptance.acceptance_id}"
        )
    record = IngressRecord(
        aggregate_id=event.aggregate_id,
        metadata=metadata,
        acceptance=acceptance,
        blob_ref=blob_ref,
        transport_message_proof=transport_message_proof,
    )
    state.by_dedup_key[metadata.dedup_key] = record
    state.by_acceptance_id[acceptance.acceptance_id] = record


def _reduce_dispositioned(
    state: IngressProjectionState,
    event: JournalEvent,
    frame_sequence: int,
    frame_hash: str,
) -> None:
    payload = event.payload
    if set(payload) != {
        "acceptance_id",
        "disposition_id",
        "state",
        "reason_code",
        "recorded_at",
    }:
        raise IngressProjectionError("invalid employee.ingress.dispositioned payload")
    record = state.by_acceptance_id.get(payload.get("acceptance_id", ""))
    if record is None:
        raise IngressProjectionError("disposition references unknown acceptance")
    if event.aggregate_id != record.aggregate_id:
        raise IngressProjectionError("disposition aggregate mismatch")
    if record.disposition is not None:
        raise IngressProjectionError("acceptance already has a disposition")
    try:
        disposition = IngressDisposition(
            schema_version=1,
            disposition_id=payload["disposition_id"],
            acceptance_id=payload["acceptance_id"],
            state=payload["state"],
            reason_code=payload["reason_code"],
            journal_sequence=frame_sequence,
            journal_frame_hash=frame_hash,
            recorded_at=payload["recorded_at"],
        )
    except (TypeError, ValueError) as exc:
        raise IngressProjectionError("invalid ingress disposition") from exc
    updated = replace(record, disposition=disposition)
    state.by_dedup_key[record.metadata.dedup_key] = updated
    state.by_acceptance_id[record.acceptance.acceptance_id] = updated


def _reduce_payload_tombstoned(
    state: IngressProjectionState,
    event: JournalEvent,
) -> None:
    payload = event.payload
    if set(payload) != {"acceptance_id", "tombstoned_at"}:
        raise IngressProjectionError("invalid ingress payload tombstone")
    record = state.by_acceptance_id.get(payload.get("acceptance_id", ""))
    if record is None:
        raise IngressProjectionError("tombstone references unknown acceptance")
    if event.aggregate_id != record.aggregate_id:
        raise IngressProjectionError("tombstone aggregate mismatch")
    if not record.terminal:
        raise IngressProjectionError("nonterminal ingress payload cannot be tombstoned")
    if record.payload_tombstoned:
        raise IngressProjectionError("ingress payload already tombstoned")
    updated = replace(record, payload_tombstoned=True)
    state.by_dedup_key[record.metadata.dedup_key] = updated
    state.by_acceptance_id[record.acceptance.acceptance_id] = updated
