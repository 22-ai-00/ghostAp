"""Tests for EmployeeDataService and data projection replay."""

from __future__ import annotations

import hashlib
import secrets
from pathlib import Path

import pytest

from src.autonomous.data.models import (
    DataKind,
    EmployeeDataDocumentV1,
    ExecutionAttemptContext,
    ExecutionHistoryPayloadV1,
    ExecutionHistoryRecordV1,
    SafeExecutionSummary,
    ToolUsageV1,
)
from src.autonomous.data.projection import (
    DataProjectionState,
    JournalHead,
    is_data_event,
)
from src.autonomous.data.service import (
    DataConflictError,
    EmployeeDataService,
)
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import CommitState, JournalWriter


class _InMemoryAnchor:
    def __init__(self) -> None:
        self._sequence = 0
        self._hash = "0" * 64

    def read(self):
        from src.autonomous.journal.anchor import AnchorState
        return AnchorState(self._sequence, self._hash)

    def compare_and_swap(
        self,
        expected_sequence: int,
        expected_hash: str,
        new_sequence: int,
        new_hash: str,
    ) -> bool:
        if self._sequence == expected_sequence and self._hash == expected_hash:
            self._sequence = new_sequence
            self._hash = new_hash
            return True
        return False


def _key() -> bytes:
    return secrets.token_bytes(32)


def _hmac_key() -> bytes:
    return secrets.token_bytes(32)


@pytest.fixture
def service(tmp_path: Path):
    key = _key()
    provider = AesGcmEncryptionProvider(lambda _ref: key)
    blob_store = BlobStore(tmp_path / "blobs", provider)
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=_InMemoryAnchor(),
        hmac_key=_hmac_key(),
    )
    state = DataProjectionState()
    svc = EmployeeDataService(
        writer=writer,
        blob_store=blob_store,
        data_state=state,
        active_key_id="k1",
        shard_timezone="UTC",
    )
    yield svc
    blob_store.close()
    writer.close()


def _context(attempt_id: str = "attempt_1") -> ExecutionAttemptContext:
    return ExecutionAttemptContext(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        owner_principal_id="principal_owner",
        requester_principal_id="principal_requester",
        task_id="task_1",
        run_id="run_1",
        attempt_id=attempt_id,
        message_id="message_1",
        thread_root_id="thread_1",
        chat_id="chat_1",
        tool="codex",
        model="gpt-test",
        effort="high",
        started_at="2026-07-12T00:30:00+00:00",
        terminal_epoch=1,
    )


def _record(context: ExecutionAttemptContext | None = None) -> ExecutionHistoryRecordV1:
    ctx = context or _context()
    return ExecutionHistoryRecordV1.from_attempt(
        ctx,
        ended_at="2026-07-12T01:30:00+00:00",
        status="completed",
        safe_summary=SafeExecutionSummary.build(status="completed", tool_count=1, attachment_count=0),
        prompt_tokens=100,
        completion_tokens=50,
        tool_usage=(ToolUsageV1("shell", 1, 120, "completed"),),
        predecessor_sequence=0,
        predecessor_hash="",
        shard_timezone="UTC",
    )


def _payload(record: ExecutionHistoryRecordV1) -> ExecutionHistoryPayloadV1:
    return ExecutionHistoryPayloadV1(
        record_id=record.record_id,
        occurrence_key=record.occurrence_key,
        request_text="please do work",
        result_text="done",
        error_detail="",
    )


class TestStartAttempt:
    def test_anchors_immutable_binding(self, service: EmployeeDataService) -> None:
        ctx = _context()
        result = service.start_attempt(ctx)
        assert result.commit_result.state == CommitState.ANCHORED
        assert ctx.attempt_id in service.state.execution_attempts
        binding = service.state.execution_attempts[ctx.attempt_id]
        assert binding.tenant_key == "tenant_1"
        assert binding.agent_id == "agt_alpha"
        assert binding.publish_sequence == 1

    def test_rejects_duplicate_attempt(self, service: EmployeeDataService) -> None:
        service.start_attempt(_context())
        with pytest.raises(DataConflictError):
            service.start_attempt(_context())


class TestRecordHistory:
    def test_publishes_encrypted_blob_and_indexes(self, service: EmployeeDataService) -> None:
        record = _record()
        payload = _payload(record)
        result = service.record_history(record, payload)
        assert result.commit_result.state == CommitState.ANCHORED
        assert record.record_id in service.state.history_records
        metadata = service.state.history_records[record.record_id]
        assert metadata.status == "completed"
        assert metadata.publish_sequence == 1
        day_key = ("tenant_1", "agt_alpha", "2026-07-12")
        assert day_key in service.state.history_by_employee_day
        occ_key = ("tenant_1", "agt_alpha", record.occurrence_key)
        assert occ_key in service.state.history_by_occurrence

    def test_idempotent_retry_returns_existing(self, service: EmployeeDataService) -> None:
        record = _record()
        payload = _payload(record)
        service.record_history(record, payload)
        second = service.record_history(record, payload)
        assert second.record == record

    def test_conflicting_payload_raises(self, service: EmployeeDataService) -> None:
        record = _record()
        payload = _payload(record)
        service.record_history(record, payload)
        different = ExecutionHistoryPayloadV1(
            record_id=record.record_id,
            occurrence_key=record.occurrence_key,
            request_text="different request",
            result_text="different",
            error_detail="",
        )
        with pytest.raises(DataConflictError):
            service.record_history(record, different)

    def test_id_mismatch_raises(self, service: EmployeeDataService) -> None:
        record = _record()
        ctx2 = _context(attempt_id="attempt_2")
        record2 = _record(ctx2)
        payload2 = _payload(record2)
        with pytest.raises(ValueError, match="mismatch"):
            service.record_history(record, payload2)


class TestPublishDocument:
    def test_publishes_and_indexes_document(self, service: EmployeeDataService) -> None:
        content = b"# Employee memory"
        content_hash = hashlib.sha256(content).hexdigest()
        doc = EmployeeDataDocumentV1(
            document_id="data_0123456789abcdef",
            tenant_key="tenant_1",
            agent_id="agt_alpha",
            owner_principal_id="principal_owner",
            kind=DataKind.L1_MEMORY,
            version=1,
            source_id="l1_memory",
            created_at="2026-07-12T01:00:00+00:00",
            predecessor_sequence=0,
            predecessor_hash="",
            content_type="text/markdown",
            content_hash=content_hash,
        )
        result = service.publish_document(doc, content)
        assert result.commit_result.state == CommitState.ANCHORED
        assert doc.document_id in service.state.employee_documents
        latest_key = ("tenant_1", "agt_alpha", "l1_memory", "l1_memory")
        assert service.state.latest_employee_document[latest_key] == doc.document_id

    def test_content_hash_mismatch_raises(self, service: EmployeeDataService) -> None:
        doc = EmployeeDataDocumentV1(
            document_id="data_0123456789abcdef",
            tenant_key="tenant_1",
            agent_id="agt_alpha",
            owner_principal_id="principal_owner",
            kind=DataKind.L1_MEMORY,
            version=1,
            source_id="l1_memory",
            created_at="2026-07-12T01:00:00+00:00",
            predecessor_sequence=0,
            predecessor_hash="",
            content_type="text/markdown",
            content_hash="a" * 64,
        )
        with pytest.raises(ValueError, match="content hash"):
            service.publish_document(doc, b"wrong content")

    def test_duplicate_document_raises(self, service: EmployeeDataService) -> None:
        content = b"# Memory"
        content_hash = hashlib.sha256(content).hexdigest()
        doc = EmployeeDataDocumentV1(
            document_id="data_0123456789abcdef",
            tenant_key="tenant_1",
            agent_id="agt_alpha",
            owner_principal_id="principal_owner",
            kind=DataKind.L1_MEMORY,
            version=1,
            source_id="l1_memory",
            created_at="2026-07-12T01:00:00+00:00",
            predecessor_sequence=0,
            predecessor_hash="",
            content_type="text/markdown",
            content_hash=content_hash,
        )
        service.publish_document(doc, content)
        with pytest.raises(DataConflictError):
            service.publish_document(doc, content)


class TestProjectionReplay:
    def test_fresh_replay_rebuilds_state(self, tmp_path: Path) -> None:
        key = _key()
        provider = AesGcmEncryptionProvider(lambda _ref: key)
        blob_store = BlobStore(tmp_path / "blobs", provider)
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=_InMemoryAnchor(),
            hmac_key=_hmac_key(),
        )
        state = DataProjectionState()
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=state,
            active_key_id="k1",
        )
        ctx = _context()
        svc.start_attempt(ctx)
        record = _record(ctx)
        payload = _payload(record)
        svc.record_history(record, payload)

        fresh = DataProjectionState()
        svc.replay_into(fresh)
        assert ctx.attempt_id in fresh.execution_attempts
        assert record.record_id in fresh.history_records
        assert fresh.cursor_sequence == state.cursor_sequence

        blob_store.close()
        writer.close()


class TestJournalHead:
    def test_genesis_head(self) -> None:
        head = JournalHead(0, "")
        assert head.sequence == 0

    def test_invalid_genesis_raises(self) -> None:
        with pytest.raises(ValueError):
            JournalHead(0, "abc")
        with pytest.raises(ValueError):
            JournalHead(1, "")


class TestDataEventClassification:
    def test_data_events_detected(self) -> None:
        assert is_data_event("employee.history.recorded")
        assert is_data_event("employee.data.published")
        assert not is_data_event("goal.created")
        assert not is_data_event("employee.created")
