"""Canonical Journal-backed group event ledger and partial context builder."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass

from ..journal.blob_store import BlobRef, BlobStore
from ..journal.frame import GENESIS_HASH, JournalEvent
from ..journal.writer import CommitState, JournalWriter
from .models import (
    AssembledContext,
    AuthorizedContextRequest,
    ContextLayer,
    ContextMessage,
    ContextQuality,
    ContextUnavailableError,
    ContextUnavailableReason,
    ContextWarning,
    MessageRevision,
    ThreadContextConfig,
    ThreadWatermark,
)


class GroupLedgerError(RuntimeError):
    pass


_FEISHU_THREAD_ID_PATTERN = re.compile(r"omt_[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_FEISHU_MESSAGE_ID_PATTERN = re.compile(r"om_[A-Za-z0-9][A-Za-z0-9_-]*\Z")


@dataclass(frozen=True, slots=True)
class GroupEventPayload:
    sender_id: str
    sender_id_type: str
    sender_type: str
    sender_tenant_key: str
    text: str
    timestamp: float
    message_type: str = "text"

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "message_type": self.message_type,
                "sender_id": self.sender_id,
                "sender_id_type": self.sender_id_type,
                "sender_tenant_key": self.sender_tenant_key,
                "sender_type": self.sender_type,
                "text": self.text,
                "timestamp": self.timestamp,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> "GroupEventPayload":
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {
            "message_type",
            "sender_id",
            "sender_id_type",
            "sender_tenant_key",
            "sender_type",
            "text",
            "timestamp",
        }:
            raise GroupLedgerError("invalid group event payload")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class GroupEventRecord:
    event_id: str
    tenant_key: str
    chat_id: str
    thread_id: str
    message_id: str
    transport_principal_id: str
    journal_sequence: int
    payload_ref: BlobRef
    dedup_key: str
    causal_event_id: str = ""


@dataclass(frozen=True, slots=True)
class CanonicalGroupContext:
    records: tuple[GroupEventRecord, ...]
    quality: ContextQuality
    warnings: tuple[ContextWarning, ...]


class GroupContextLedger:
    """Deduplicate transport observations without trusting payload authority."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        blob_store: BlobStore,
        active_key_id: str,
        config: ThreadContextConfig | None = None,
        blob_retainer: Callable[[str], None] | None = None,
        blob_releaser: Callable[[str], None] | None = None,
    ) -> None:
        self._writer = writer
        self._blobs = blob_store
        self._key = active_key_id
        self._config = config or ThreadContextConfig()
        self._blob_retainer = blob_retainer
        self._blob_releaser = blob_releaser
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._records: dict[str, GroupEventRecord] = {}
        self.rebuild_projection()

    def publish(
        self,
        *,
        tenant_key: str,
        chat_id: str,
        thread_id: str,
        message_id: str,
        transport_principal_id: str,
        transport_event_id: str,
        payload: GroupEventPayload,
        causal_event_id: str = "",
    ) -> GroupEventRecord:
        for value, name in (
            (tenant_key, "tenant_key"),
            (chat_id, "chat_id"),
            (message_id, "message_id"),
            (transport_principal_id, "transport_principal_id"),
            (transport_event_id, "transport_event_id"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} is required")
        raw = payload.to_bytes()
        identity = "\0".join((tenant_key, chat_id, message_id, causal_event_id))
        dedup_key = hashlib.sha256(identity.encode()).hexdigest()
        aggregate = f"group-event:{dedup_key}"
        with self._lock, self._writer.transaction_guard():
            self.rebuild_projection()
            duplicate = self._records.get(dedup_key)
            if duplicate is not None:
                existing_payload = GroupEventPayload.from_bytes(
                    self._blobs.read(duplicate.payload_ref)
                )
                if (
                    existing_payload.sender_id,
                    existing_payload.sender_id_type,
                    existing_payload.sender_type,
                    existing_payload.sender_tenant_key,
                    existing_payload.text,
                    existing_payload.message_type,
                ) != (
                    payload.sender_id,
                    payload.sender_id_type,
                    payload.sender_type,
                    payload.sender_tenant_key,
                    payload.text,
                    payload.message_type,
                ):
                    raise GroupLedgerError("group event idempotency conflict")
                return duplicate
            ref = self._blobs.stage_and_publish(
                raw,
                {
                    "tenant_key": tenant_key,
                    "chat_id": chat_id,
                    "kind": "group_event",
                },
                self._key,
            )
            if self._blob_retainer is not None:
                self._blob_retainer(ref.blob_id)
            event = JournalEvent(
                event_type="group.event.recorded",
                aggregate_id=aggregate,
                payload={
                    "event_id": transport_event_id,
                    "tenant_key": tenant_key,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "message_id": message_id,
                    "transport_principal_id": transport_principal_id,
                    "payload_ref": ref.to_dict(),
                    "dedup_key": dedup_key,
                    "causal_event_id": causal_event_id,
                },
            )
            last = self._writer.get_last_frame()
            try:
                result = self._writer.commit(
                    (event,),
                    self._writer.get_aggregate_versions((aggregate,)),
                    expected_head_sequence=0 if last is None else last.sequence,
                    expected_head_hash="" if last is None else last.frame_hash,
                )
            except BaseException:
                if self._blob_releaser is not None:
                    self._blob_releaser(ref.blob_id)
                raise
            if result.state is not CommitState.ANCHORED:
                if self._blob_releaser is not None:
                    self._blob_releaser(ref.blob_id)
                raise GroupLedgerError("group event was not anchored")
            record = self._record_from_event(event, result.frame.sequence)
            self._records[dedup_key] = record
            return record

    def rebuild_projection(self) -> int:
        records: dict[str, GroupEventRecord] = {}
        anchored = self._writer.anchor.read()
        last_hash = GENESIS_HASH
        for frame in self._writer.replay():
            if frame.sequence > anchored.sequence:
                break
            last_hash = frame.frame_hash
            for event in frame.events:
                if event.event_type != "group.event.recorded":
                    continue
                record = self._record_from_event(event, frame.sequence)
                existing = records.get(record.dedup_key)
                if existing is not None and existing != record:
                    raise GroupLedgerError("duplicate canonical group event")
                records[record.dedup_key] = record
        if last_hash != anchored.frame_hash:
            raise GroupLedgerError("group ledger anchor mismatch")
        self._records = records
        if self._blob_retainer is not None:
            for record in records.values():
                self._blob_retainer(record.payload_ref.blob_id)
        return len(records)

    def window(
        self,
        *,
        tenant_key: str,
        chat_id: str,
        current_message_id: str,
        causal_event_id: str = "",
    ) -> CanonicalGroupContext:
        with self._lock:
            candidates = sorted(
                (
                    record
                    for record in self._records.values()
                    if record.tenant_key == tenant_key and record.chat_id == chat_id
                ),
                key=lambda item: (item.journal_sequence, item.dedup_key),
            )
        current = next(
            (
                item
                for item in reversed(candidates)
                if item.message_id == current_message_id
                and item.causal_event_id == causal_event_id
            ),
            None,
        )
        if current is None:
            raise ContextUnavailableError(ContextUnavailableReason.CURRENT_MESSAGE)
        records = tuple(
            item
            for item in candidates
            if item.journal_sequence <= current.journal_sequence
        )[-self._config.max_group_messages :]
        return CanonicalGroupContext(
            records=records,
            quality=ContextQuality.CANONICAL_PARTIAL,
            warnings=(ContextWarning("order_unavailable", "lark"),),
        )

    def assemble_partial(
        self,
        request: AuthorizedContextRequest,
        *,
        warning_reason: ContextUnavailableReason,
        causal_event_id: str = "",
        l1_summary: str = "",
        l2_summary: str = "",
    ) -> AssembledContext:
        canonical = self.window(
            tenant_key=request.tenant_key,
            chat_id=request.chat_id,
            current_message_id=request.current_message_id,
            causal_event_id=causal_event_id,
        )
        messages: list[ContextMessage] = []
        legacy_thread_record_omitted = False
        for record in canonical.records:
            is_current = record is canonical.records[-1]
            payload = GroupEventPayload.from_bytes(self._blobs.read(record.payload_ref))
            if payload.sender_tenant_key != request.tenant_key:
                raise ContextUnavailableError(ContextUnavailableReason.SCOPE)
            thread_id = record.thread_id
            if thread_id == request.thread_root_message_id:
                # Older main-Bot observations stored the topic root message
                # (``om_``) in the Feishu thread field.  Recover only the
                # authority-bound current topic root with a real ``omt_``;
                # all other malformed ledger coordinates fail as Context.
                if not request.feishu_thread_id:
                    raise ContextUnavailableError(
                        ContextUnavailableReason.ROOT_THREAD_BINDING
                    )
                thread_id = request.feishu_thread_id
            elif thread_id and _FEISHU_THREAD_ID_PATTERN.fullmatch(thread_id) is None:
                # An older producer stored ``om_`` topic roots in this field.
                # A non-current record is only group history, so omit it when
                # its real ``omt_`` coordinate cannot be authority-bound.  The
                # current record and arbitrary malformed values still fail.
                if (
                    not is_current
                    and _FEISHU_MESSAGE_ID_PATTERN.fullmatch(thread_id) is not None
                ):
                    legacy_thread_record_omitted = True
                    continue
                raise ContextUnavailableError(
                    ContextUnavailableReason.ROOT_THREAD_BINDING
                )
            messages.append(
                ContextMessage(
                    message_id=record.message_id,
                    sender_id=payload.sender_id,
                    sender_type=payload.sender_type,
                    text=payload.text,
                    timestamp=payload.timestamp,
                    is_current=is_current,
                    chat_id=record.chat_id,
                    thread_id=thread_id,
                    root_id=(
                        ""
                        if record.message_id == request.thread_root_message_id
                        else request.thread_root_message_id
                    ),
                    parent_id="",
                    sender_id_type=payload.sender_id_type,
                    sender_tenant_key=payload.sender_tenant_key,
                    msg_type=payload.message_type,
                )
            )
        current = messages[-1]
        if (
            current.message_id != request.current_message_id
            or current.sender_id != request.requester_principal_id
            or current.sender_id_type != "open_id"
        ):
            raise ContextUnavailableError(ContextUnavailableReason.CURRENT_MESSAGE)
        revision_digest = hashlib.sha256(
            "".join(MessageRevision.from_message(item).digest for item in messages).encode()
        ).hexdigest()
        watermark = ThreadWatermark(
            thread_root_id=request.thread_root_message_id,
            last_message_id=current.message_id,
            last_timestamp=current.timestamp,
            message_count=1,
            tenant_key=request.tenant_key,
            chat_id=request.chat_id,
            feishu_thread_id=request.feishu_thread_id,
            revision_digest=revision_digest,
        )
        group = messages[:-1]
        chars = sum(len(item.text) for item in messages) + len(l1_summary) + len(l2_summary)
        tokens = math.ceil(chars * self._config.tokens_per_char) + request.system_prompt_token_reserve
        if chars > self._config.max_context_chars or tokens > self._config.max_context_tokens:
            raise ContextUnavailableError(ContextUnavailableReason.BUDGET)
        warning_code = (
            "order_unavailable"
            if warning_reason is ContextUnavailableReason.ORDERING
            else f"{warning_reason.value}_unavailable"
        )
        warnings = [ContextWarning(warning_code, "lark")]
        if legacy_thread_record_omitted:
            warnings.append(
                ContextWarning("legacy_thread_record_omitted", "ledger")
            )
        snapshot_hash = hashlib.sha256(
            (
                revision_digest
                + "".join(item.code for item in warnings)
                + request.constraints_digest
            ).encode()
        ).hexdigest()
        return AssembledContext(
            thread_messages=(current,),
            group_messages=tuple(group),
            l1_summary=l1_summary,
            l2_summary=l2_summary,
            total_tokens_estimate=tokens,
            watermark=watermark,
            layers_used=(ContextLayer.THREAD_FULL, ContextLayer.GROUP_RECENT),
            total_chars=chars,
            snapshot_hash=snapshot_hash,
            system_prompt_tokens_reserved=request.system_prompt_token_reserve,
            constraints_digest=request.constraints_digest,
            tokens_per_char=self._config.tokens_per_char,
            quality=ContextQuality.CANONICAL_PARTIAL,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _record_from_event(event: JournalEvent, sequence: int) -> GroupEventRecord:
        payload = event.payload
        required = {
            "event_id",
            "tenant_key",
            "chat_id",
            "thread_id",
            "message_id",
            "transport_principal_id",
            "payload_ref",
            "dedup_key",
            "causal_event_id",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise GroupLedgerError("invalid group event record")
        ref = BlobRef.from_dict(payload["payload_ref"])
        return GroupEventRecord(
            event_id=str(payload["event_id"]),
            tenant_key=str(payload["tenant_key"]),
            chat_id=str(payload["chat_id"]),
            thread_id=str(payload["thread_id"]),
            message_id=str(payload["message_id"]),
            transport_principal_id=str(payload["transport_principal_id"]),
            journal_sequence=sequence,
            payload_ref=ref,
            dedup_key=str(payload["dedup_key"]),
            causal_event_id=str(payload["causal_event_id"]),
        )


__all__ = [
    "CanonicalGroupContext",
    "GroupContextLedger",
    "GroupEventPayload",
    "GroupEventRecord",
    "GroupLedgerError",
]
