"""Journal-backed projection for durable employee ingress metadata."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime

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


MessageLogicalKey = tuple[str, str, str, str, str, str, str]
MessageTransportKey = tuple[str, str, str, str, str, str, str, int, str]


@dataclass(frozen=True, slots=True)
class MessageAcceptanceDenial:
    """Durable execution fence for one logical transport message."""

    aggregate_id: str
    tenant_key: str
    agent_id: str
    bot_principal_id: str
    app_id: str
    event_type: str
    chat_id: str
    message_id: str
    channel_generation: int
    connection_id: str
    reason_code: str
    denied_at: str

    @property
    def logical_key(self) -> MessageLogicalKey:
        return message_logical_key(
            tenant_key=self.tenant_key,
            agent_id=self.agent_id,
            bot_principal_id=self.bot_principal_id,
            app_id=self.app_id,
            event_type=self.event_type,
            chat_id=self.chat_id,
            message_id=self.message_id,
        )

    @property
    def transport_key(self) -> MessageTransportKey:
        return message_transport_key(
            tenant_key=self.tenant_key,
            agent_id=self.agent_id,
            bot_principal_id=self.bot_principal_id,
            app_id=self.app_id,
            event_type=self.event_type,
            chat_id=self.chat_id,
            message_id=self.message_id,
            channel_generation=self.channel_generation,
            connection_id=self.connection_id,
        )


@dataclass
class IngressProjectionState:
    """Replayable ingress indexes plus recovery-only closed employee state."""

    by_dedup_key: dict[str, IngressRecord] = field(default_factory=dict)
    by_acceptance_id: dict[str, IngressRecord] = field(default_factory=dict)
    message_acceptance_winners: dict[MessageLogicalKey, str] = field(
        default_factory=dict
    )
    message_transport_witnesses: dict[MessageTransportKey, str] = field(
        default_factory=dict
    )
    message_acceptance_denials: dict[
        MessageLogicalKey, MessageAcceptanceDenial
    ] = field(default_factory=dict)
    message_denied_acceptances: dict[MessageLogicalKey, str] = field(
        default_factory=dict
    )
    closed_employees: set[tuple[str, str]] = field(default_factory=set)
    cursor_sequence: int = 0
    cursor_hash: str = ""

    def clone(self) -> IngressProjectionState:
        return copy.deepcopy(self)


_INGRESS_EVENT_TYPES = frozenset(
    {
        "employee.ingress.accepted",
        "employee.ingress.denied_acceptance",
        "employee.ingress.invalid_transport_acceptance",
        "employee.ingress.message_acceptance_denied",
        "employee.ingress.message_redelivery_accepted",
        "employee.ingress.closed",
        "employee.ingress.dispositioned",
        "employee.ingress.payload_tombstoned",
    }
)

_SAFE_MESSAGE_INDEX = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")


def message_logical_key(
    *,
    tenant_key: str,
    agent_id: str,
    bot_principal_id: str,
    app_id: str,
    event_type: str,
    chat_id: str,
    message_id: str,
) -> MessageLogicalKey:
    values = (
        tenant_key,
        agent_id,
        bot_principal_id,
        app_id,
        event_type,
        chat_id,
        message_id,
    )
    if any(
        not isinstance(value, str)
        or not value
        or _SAFE_MESSAGE_INDEX.fullmatch(value) is None
        for value in values
    ):
        raise ValueError("message acceptance coordinates are invalid")
    if (
        not agent_id.startswith("agt_")
        or not bot_principal_id.startswith("bot_")
        or not app_id.startswith("cli_")
        or len(chat_id) != 67
        or not chat_id.startswith("oc_")
        or len(message_id) != 67
        or not message_id.startswith("om_")
        or any(character not in "0123456789abcdef" for character in chat_id[3:])
        or any(character not in "0123456789abcdef" for character in message_id[3:])
    ):
        raise ValueError("message acceptance binding is invalid")
    return values


def message_transport_key(
    *,
    tenant_key: str,
    agent_id: str,
    bot_principal_id: str,
    app_id: str,
    event_type: str,
    chat_id: str,
    message_id: str,
    channel_generation: int,
    connection_id: str,
) -> MessageTransportKey:
    logical = message_logical_key(
        tenant_key=tenant_key,
        agent_id=agent_id,
        bot_principal_id=bot_principal_id,
        app_id=app_id,
        event_type=event_type,
        chat_id=chat_id,
        message_id=message_id,
    )
    if (
        type(channel_generation) is not int
        or channel_generation <= 0
        or not isinstance(connection_id, str)
        or not connection_id.startswith("conn_")
        or _SAFE_MESSAGE_INDEX.fullmatch(connection_id) is None
    ):
        raise ValueError("message acceptance transport is invalid")
    return (*logical, channel_generation, connection_id)


def message_logical_key_from_metadata(
    metadata: EmployeeIngressMetadata,
) -> MessageLogicalKey:
    return message_logical_key(
        tenant_key=metadata.tenant_key,
        agent_id=metadata.agent_id,
        bot_principal_id=metadata.bot_principal_id,
        app_id=metadata.app_id,
        event_type=metadata.event_type,
        chat_id=metadata.chat_id,
        message_id=metadata.message_id,
    )


def message_transport_key_from_metadata(
    metadata: EmployeeIngressMetadata,
) -> MessageTransportKey:
    return message_transport_key(
        tenant_key=metadata.tenant_key,
        agent_id=metadata.agent_id,
        bot_principal_id=metadata.bot_principal_id,
        app_id=metadata.app_id,
        event_type=metadata.event_type,
        chat_id=metadata.chat_id,
        message_id=metadata.message_id,
        channel_generation=metadata.channel_generation,
        connection_id=metadata.connection_id,
    )


def message_deny_aggregate_id(transport_key: MessageTransportKey) -> str:
    canonical = json.dumps(
        list(transport_key),
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "employee-ingress-deny:" + hashlib.sha256(canonical).hexdigest()


def _validate_timestamp(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be UTC")
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    if parsed.utcoffset() is None:
        raise ValueError(f"{name} must be UTC")
    return value


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
    elif event.event_type == "employee.ingress.denied_acceptance":
        _reduce_denied_acceptance(state, event, frame_sequence, frame_hash)
    elif event.event_type == "employee.ingress.invalid_transport_acceptance":
        _reduce_invalid_transport_acceptance(
            state,
            event,
            frame_sequence,
            frame_hash,
        )
    elif event.event_type == "employee.ingress.message_acceptance_denied":
        _reduce_message_acceptance_denied(state, event)
    elif event.event_type == "employee.ingress.message_redelivery_accepted":
        _reduce_message_redelivery_accepted(state, event)
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
    if transport_message_proof:
        _bind_message_acceptance(state, record)
    state.by_dedup_key[metadata.dedup_key] = record
    state.by_acceptance_id[acceptance.acceptance_id] = record


def _bind_message_acceptance(
    state: IngressProjectionState,
    record: IngressRecord,
) -> None:
    metadata = record.metadata
    try:
        logical_key = message_logical_key_from_metadata(metadata)
        transport_key = message_transport_key_from_metadata(metadata)
    except ValueError as exc:
        raise IngressProjectionError("invalid message acceptance binding") from exc
    if logical_key in state.message_acceptance_denials:
        raise IngressProjectionError(
            "message acceptance conflicts with durable denial"
        )
    winner = state.message_acceptance_winners.get(logical_key)
    if winner is not None and winner != record.acceptance.acceptance_id:
        raise IngressProjectionError("logical message has multiple acceptances")
    if winner is None:
        winner = record.acceptance.acceptance_id
    existing_witness = state.message_transport_witnesses.get(transport_key)
    if existing_witness is not None and existing_witness != winner:
        raise IngressProjectionError("message transport witness owner mismatch")
    state.message_acceptance_winners.setdefault(logical_key, winner)
    state.message_transport_witnesses.setdefault(transport_key, winner)


def _reduce_message_acceptance_denied(
    state: IngressProjectionState,
    event: JournalEvent,
) -> None:
    fields = {
        "tenant_key",
        "agent_id",
        "bot_principal_id",
        "app_id",
        "event_type",
        "chat_id",
        "message_id",
        "channel_generation",
        "connection_id",
        "reason_code",
        "denied_at",
    }
    if set(event.payload) != fields:
        raise IngressProjectionError(
            "invalid employee.ingress.message_acceptance_denied payload"
        )
    try:
        transport_key = message_transport_key(
            tenant_key=event.payload["tenant_key"],
            agent_id=event.payload["agent_id"],
            bot_principal_id=event.payload["bot_principal_id"],
            app_id=event.payload["app_id"],
            event_type=event.payload["event_type"],
            chat_id=event.payload["chat_id"],
            message_id=event.payload["message_id"],
            channel_generation=event.payload["channel_generation"],
            connection_id=event.payload["connection_id"],
        )
        reason_code = event.payload["reason_code"]
        denied_at = event.payload["denied_at"]
        if transport_key[4] != "im.message.receive_v1":
            raise ValueError("invalid message acceptance denial event type")
        if reason_code != "handoff_unconfirmed":
            raise ValueError("invalid message acceptance denial reason")
        _validate_timestamp(denied_at, "denied_at")
    except (TypeError, ValueError) as exc:
        raise IngressProjectionError(
            "invalid employee message acceptance denial"
        ) from exc
    if event.aggregate_id != message_deny_aggregate_id(transport_key):
        raise IngressProjectionError("message acceptance denial aggregate mismatch")
    logical_key = transport_key[:7]
    if logical_key in state.message_acceptance_winners:
        raise IngressProjectionError(
            "message acceptance denial conflicts with accepted witness"
        )
    if logical_key in state.message_acceptance_denials:
        raise IngressProjectionError("duplicate message acceptance denial")
    state.message_acceptance_denials[logical_key] = MessageAcceptanceDenial(
        aggregate_id=event.aggregate_id,
        tenant_key=transport_key[0],
        agent_id=transport_key[1],
        bot_principal_id=transport_key[2],
        app_id=transport_key[3],
        event_type=transport_key[4],
        chat_id=transport_key[5],
        message_id=transport_key[6],
        channel_generation=transport_key[7],
        connection_id=transport_key[8],
        reason_code=reason_code,
        denied_at=denied_at,
    )


def _reduce_message_redelivery_accepted(
    state: IngressProjectionState,
    event: JournalEvent,
) -> None:
    fields = {
        "acceptance_id",
        "tenant_key",
        "agent_id",
        "bot_principal_id",
        "app_id",
        "event_type",
        "chat_id",
        "message_id",
        "channel_generation",
        "connection_id",
        "witnessed_at",
    }
    if set(event.payload) != fields:
        raise IngressProjectionError(
            "invalid employee.ingress.message_redelivery_accepted payload"
        )
    acceptance_id = event.payload.get("acceptance_id")
    record = state.by_acceptance_id.get(
        acceptance_id if isinstance(acceptance_id, str) else ""
    )
    if record is None or event.aggregate_id != record.aggregate_id:
        raise IngressProjectionError("message redelivery references unknown acceptance")
    try:
        transport_key = message_transport_key(
            tenant_key=event.payload["tenant_key"],
            agent_id=event.payload["agent_id"],
            bot_principal_id=event.payload["bot_principal_id"],
            app_id=event.payload["app_id"],
            event_type=event.payload["event_type"],
            chat_id=event.payload["chat_id"],
            message_id=event.payload["message_id"],
            channel_generation=event.payload["channel_generation"],
            connection_id=event.payload["connection_id"],
        )
        _validate_timestamp(event.payload["witnessed_at"], "witnessed_at")
    except (TypeError, ValueError) as exc:
        raise IngressProjectionError("invalid message redelivery witness") from exc
    logical_key = transport_key[:7]
    winner = state.message_acceptance_winners.get(logical_key)
    if (
        not record.transport_message_proof
        or winner != record.acceptance.acceptance_id
        or logical_key in state.message_acceptance_denials
    ):
        raise IngressProjectionError("message redelivery witness owner mismatch")
    existing = state.message_transport_witnesses.get(transport_key)
    if existing is not None:
        raise IngressProjectionError("duplicate message redelivery witness")
    state.message_transport_witnesses[transport_key] = winner


def _reduce_denied_acceptance(
    state: IngressProjectionState,
    event: JournalEvent,
    frame_sequence: int,
    frame_hash: str,
) -> None:
    fields = {
        "metadata",
        "acceptance_id",
        "accepted_at",
        "blob_ref",
        "transport_message_proof",
        "disposition_id",
        "reason_code",
        "recorded_at",
    }
    if set(event.payload) != fields:
        raise IngressProjectionError(
            "invalid employee.ingress.denied_acceptance payload"
        )
    if event.payload.get("transport_message_proof") is not True:
        raise IngressProjectionError("denied acceptance lacks transport proof")
    try:
        metadata = EmployeeIngressMetadata.from_dict(event.payload["metadata"])
        blob_ref = BlobRef.from_dict(event.payload["blob_ref"])
        acceptance = IngressAcceptance(
            schema_version=1,
            acceptance_id=event.payload["acceptance_id"],
            envelope_id=metadata.envelope_id,
            dedup_key=metadata.dedup_key,
            semantic_digest=metadata.semantic_digest,
            journal_sequence=frame_sequence,
            journal_frame_hash=frame_hash,
            accepted_at=event.payload["accepted_at"],
        )
        disposition = IngressDisposition(
            schema_version=1,
            disposition_id=event.payload["disposition_id"],
            acceptance_id=event.payload["acceptance_id"],
            state="terminal",
            reason_code=event.payload["reason_code"],
            journal_sequence=frame_sequence,
            journal_frame_hash=frame_hash,
            recorded_at=event.payload["recorded_at"],
        )
    except (TypeError, ValueError) as exc:
        raise IngressProjectionError("invalid denied ingress acceptance") from exc
    if event.aggregate_id != metadata.dedup_key:
        raise IngressProjectionError("denied acceptance aggregate mismatch")
    logical_key = message_logical_key_from_metadata(metadata)
    denial = state.message_acceptance_denials.get(logical_key)
    if denial is None or denial.reason_code != disposition.reason_code:
        raise IngressProjectionError("denied acceptance lacks matching fence")
    if logical_key in state.message_acceptance_winners:
        raise IngressProjectionError("denied acceptance conflicts with accepted winner")
    if (
        metadata.dedup_key in state.by_dedup_key
        or acceptance.acceptance_id in state.by_acceptance_id
        or logical_key in state.message_denied_acceptances
    ):
        raise IngressProjectionError("duplicate denied ingress acceptance")
    record = IngressRecord(
        aggregate_id=event.aggregate_id,
        metadata=metadata,
        acceptance=acceptance,
        blob_ref=blob_ref,
        transport_message_proof=True,
        disposition=disposition,
    )
    state.by_dedup_key[metadata.dedup_key] = record
    state.by_acceptance_id[acceptance.acceptance_id] = record
    state.message_denied_acceptances[logical_key] = acceptance.acceptance_id


def _reduce_invalid_transport_acceptance(
    state: IngressProjectionState,
    event: JournalEvent,
    frame_sequence: int,
    frame_hash: str,
) -> None:
    """Project a normally ACKed public message that cannot prove its indexes."""

    fields = {
        "metadata",
        "acceptance_id",
        "accepted_at",
        "blob_ref",
        "transport_message_proof",
        "disposition_id",
        "reason_code",
        "recorded_at",
    }
    if set(event.payload) != fields:
        raise IngressProjectionError(
            "invalid employee.ingress.invalid_transport_acceptance payload"
        )
    if event.payload.get("transport_message_proof") is not False:
        raise IngressProjectionError("invalid transport acceptance grants proof")
    try:
        metadata = EmployeeIngressMetadata.from_dict(event.payload["metadata"])
        blob_ref = BlobRef.from_dict(event.payload["blob_ref"])
        acceptance = IngressAcceptance(
            schema_version=1,
            acceptance_id=event.payload["acceptance_id"],
            envelope_id=metadata.envelope_id,
            dedup_key=metadata.dedup_key,
            semantic_digest=metadata.semantic_digest,
            journal_sequence=frame_sequence,
            journal_frame_hash=frame_hash,
            accepted_at=event.payload["accepted_at"],
        )
        disposition = IngressDisposition(
            schema_version=1,
            disposition_id=event.payload["disposition_id"],
            acceptance_id=event.payload["acceptance_id"],
            state="terminal",
            reason_code=event.payload["reason_code"],
            journal_sequence=frame_sequence,
            journal_frame_hash=frame_hash,
            recorded_at=event.payload["recorded_at"],
        )
    except (TypeError, ValueError) as exc:
        raise IngressProjectionError(
            "invalid terminal transport acceptance"
        ) from exc
    if (
        event.aggregate_id != metadata.dedup_key
        or metadata.event_type != "im.message.receive_v1"
        or disposition.reason_code != "invalid_transport_proof"
    ):
        raise IngressProjectionError("invalid transport acceptance binding")
    if (
        metadata.dedup_key in state.by_dedup_key
        or acceptance.acceptance_id in state.by_acceptance_id
    ):
        raise IngressProjectionError("duplicate invalid transport acceptance")
    record = IngressRecord(
        aggregate_id=event.aggregate_id,
        metadata=metadata,
        acceptance=acceptance,
        blob_ref=blob_ref,
        transport_message_proof=False,
        disposition=disposition,
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
