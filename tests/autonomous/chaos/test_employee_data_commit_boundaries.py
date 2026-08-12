"""Chaos tests: blob publish failure, anchor failure, and head race boundaries."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from src.autonomous.data.models import (
    DataKind,
    EmployeeDataDocumentV1,
    ExecutionAttemptContext,
    ExecutionHistoryPayloadV1,
    ExecutionHistoryRecordV1,
    SafeExecutionSummary,
)
from src.autonomous.data.projection import DataProjectionState
from src.autonomous.data.service import (
    DataBlobError,
    DataWriteDisabledError,
    EmployeeDataService,
)
from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.blob_store import (
    AesGcmEncryptionProvider,
    BlobPublishError,
    BlobStore,
)
from src.autonomous.journal.writer import CommitState, JournalWriter


class _InMemoryAnchor:
    def __init__(self, *, fail_after: int = -1) -> None:
        self._sequence = 0
        self._hash = "0" * 64
        self._call_count = 0
        self._fail_after = fail_after

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
        self._call_count += 1
        if self._fail_after >= 0 and self._call_count > self._fail_after:
            return False
        if self._sequence == expected_sequence and self._hash == expected_hash:
            self._sequence = new_sequence
            self._hash = new_hash
            return True
        return False


def _key() -> bytes:
    return secrets.token_bytes(32)


def _context() -> ExecutionAttemptContext:
    return ExecutionAttemptContext(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        owner_principal_id="principal_owner",
        requester_principal_id="principal_requester",
        task_id="task_1",
        run_id="run_1",
        attempt_id="attempt_1",
        message_id="message_1",
        thread_root_id="thread_1",
        chat_id="chat_1",
        tool="codex",
        model="gpt-test",
        effort="high",
        started_at="2026-07-12T00:30:00+00:00",
        terminal_epoch=1,
    )


def _record() -> ExecutionHistoryRecordV1:
    return ExecutionHistoryRecordV1.from_attempt(
        _context(),
        ended_at="2026-07-12T01:30:00+00:00",
        status="completed",
        safe_summary=SafeExecutionSummary.build(status="completed", tool_count=1, attachment_count=0),
        prompt_tokens=10,
        completion_tokens=5,
        tool_usage=(),
        predecessor_sequence=0,
        predecessor_hash="",
        shard_timezone="UTC",
    )


def _payload(record: ExecutionHistoryRecordV1) -> ExecutionHistoryPayloadV1:
    return ExecutionHistoryPayloadV1(
        record_id=record.record_id,
        occurrence_key=record.occurrence_key,
        request_text="test",
        result_text="ok",
        error_detail="",
    )


class TestAnchorFailureDisablesWrites:
    def test_anchor_failure_disables_subsequent_writes(self, tmp_path: Path) -> None:
        anchor = _InMemoryAnchor(fail_after=0)
        key = _key()
        provider = AesGcmEncryptionProvider(lambda _ref: key)
        blob_store = BlobStore(tmp_path / "blobs", provider)
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=anchor,
            hmac_key=secrets.token_bytes(32),
        )
        state = DataProjectionState()
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=state,
            active_key_id="k1",
        )
        with pytest.raises(DataWriteDisabledError):
            svc.start_attempt(_context())
        blob_store.close()
        writer.close()

    def test_ambiguous_anchor_preserves_history_blob_for_hygiene_and_restart(
        self,
        tmp_path: Path,
    ) -> None:
        durable_anchor = FileAnchor(tmp_path / "anchor.json")

        class _RaiseAfterAnchor:
            production_safe = True

            def read(self):
                return durable_anchor.read()

            def compare_and_swap(self, *args) -> bool:
                assert durable_anchor.compare_and_swap(*args) is True
                raise OSError("anchor directory fsync outcome unknown")

        key = _key()
        hmac_key = secrets.token_bytes(32)
        provider = AesGcmEncryptionProvider(lambda _ref: key)
        blob_store = BlobStore(tmp_path / "blobs", provider)
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=_RaiseAfterAnchor(),
            hmac_key=hmac_key,
        )
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=DataProjectionState(),
            active_key_id="k1",
        )
        record = _record()
        payload = _payload(record)

        with pytest.raises(DataWriteDisabledError, match="not anchored"):
            svc.record_history(record, payload)

        assert len(blob_store.iter_blob_ids()) == 1
        assert tuple((blob_store.root / "quarantine").glob("*.blob")) == ()
        assert svc.quarantine_unreferenced_blobs() == 0
        assert len(blob_store.iter_blob_ids()) == 1
        svc.close()
        writer.close()

        recovered_store = BlobStore(tmp_path / "blobs", provider)
        recovered_writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=durable_anchor,
            hmac_key=hmac_key,
            writer_epoch=1,
        )
        recovered = EmployeeDataService(
            writer=recovered_writer,
            blob_store=recovered_store,
            data_state=DataProjectionState(),
            active_key_id="k1",
        )
        try:
            recovered.rebuild_projection()
            recovered.verify_live_blobs()
            assert recovered.get_history_payload(record.record_id) == payload
        finally:
            recovered.close()
            recovered_writer.close()

    def test_anchored_history_apply_failure_preserves_blob_for_restart(
        self,
        tmp_path: Path,
    ) -> None:
        durable_anchor = FileAnchor(tmp_path / "anchor.json")
        key = _key()
        hmac_key = secrets.token_bytes(32)
        provider = AesGcmEncryptionProvider(lambda _ref: key)
        blob_store = BlobStore(tmp_path / "blobs", provider)
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=durable_anchor,
            hmac_key=hmac_key,
        )
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=DataProjectionState(),
            active_key_id="k1",
        )
        record = _record()
        payload = _payload(record)

        with (
            patch.object(
                svc,
                "_apply_frame",
                side_effect=RuntimeError("projection apply fault"),
            ),
            pytest.raises(RuntimeError, match="projection apply fault"),
        ):
            svc.record_history(record, payload)

        assert len(blob_store.iter_blob_ids()) == 1
        svc.close()
        writer.close()

        recovered_store = BlobStore(tmp_path / "blobs", provider)
        recovered_writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=durable_anchor,
            hmac_key=hmac_key,
            writer_epoch=1,
        )
        recovered = EmployeeDataService(
            writer=recovered_writer,
            blob_store=recovered_store,
            data_state=DataProjectionState(),
            active_key_id="k1",
        )
        try:
            recovered.rebuild_projection()
            recovered.verify_live_blobs()
            assert recovered.get_history_payload(record.record_id) == payload
        finally:
            recovered.close()
            recovered_writer.close()

    def test_hygiene_serializes_live_snapshot_through_quarantine(
        self,
        tmp_path: Path,
    ) -> None:
        key = _key()
        blob_store = BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(lambda _ref: key),
        )
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=_InMemoryAnchor(),
            hmac_key=secrets.token_bytes(32),
        )
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=DataProjectionState(),
            active_key_id="k1",
        )
        content = b"# GC-safe memory"
        document = EmployeeDataDocumentV1(
            document_id="data_0123456789abcdef",
            tenant_key="tenant_1",
            agent_id="agt_alpha",
            owner_principal_id="principal_owner",
            kind=DataKind.L1_MEMORY,
            version=1,
            source_id="l1_memory",
            created_at="2026-07-12T01:30:00+00:00",
            predecessor_sequence=0,
            predecessor_hash="",
            content_type="text/markdown",
            content_hash=hashlib.sha256(content).hexdigest(),
        )
        scan_entered = threading.Event()
        release_scan = threading.Event()
        publish_done = threading.Event()
        errors: list[Exception] = []
        gc_results: list[int] = []
        original_iter = blob_store.iter_blob_ids

        def blocked_iter():
            scan_entered.set()
            assert release_scan.wait(5)
            return original_iter()

        def collect() -> None:
            try:
                gc_results.append(svc.quarantine_unreferenced_blobs())
            except Exception as exc:
                errors.append(exc)

        def publish() -> None:
            try:
                svc.publish_document(document, content)
            except Exception as exc:
                errors.append(exc)
            finally:
                publish_done.set()

        with patch.object(blob_store, "iter_blob_ids", side_effect=blocked_iter):
            gc_thread = threading.Thread(target=collect)
            publish_thread = threading.Thread(target=publish)
            gc_thread.start()
            assert scan_entered.wait(5)
            publish_thread.start()
            assert not publish_done.wait(0.1)
            release_scan.set()
            gc_thread.join(5)
            publish_thread.join(5)

        assert not gc_thread.is_alive()
        assert not publish_thread.is_alive()
        assert errors == []
        assert gc_results == [0]
        svc.verify_live_blobs()
        blob_store.close()
        writer.close()

    def test_hygiene_retains_history_staged_before_commit(
        self,
        tmp_path: Path,
    ) -> None:
        key = _key()
        blob_store = BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(lambda _ref: key),
        )
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=_InMemoryAnchor(),
            hmac_key=secrets.token_bytes(32),
        )
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=DataProjectionState(),
            active_key_id="k1",
        )
        record = _record()
        payload = _payload(record)
        stage_done = threading.Event()
        release_stage = threading.Event()
        errors: list[Exception] = []
        original_stage = svc.stage_history_payload

        def blocked_stage(*args, **kwargs):
            staged = original_stage(*args, **kwargs)
            stage_done.set()
            assert release_stage.wait(5)
            return staged

        def publish() -> None:
            try:
                svc.record_history(record, payload)
            except Exception as exc:
                errors.append(exc)

        with patch.object(svc, "stage_history_payload", side_effect=blocked_stage):
            thread = threading.Thread(target=publish)
            thread.start()
            try:
                assert stage_done.wait(5)
                assert svc.quarantine_unreferenced_blobs() == 0
                assert len(blob_store.iter_blob_ids()) == 1
            finally:
                release_stage.set()
                thread.join(5)

        assert not thread.is_alive()
        assert errors == []
        assert svc.get_history_payload(record.record_id) == payload
        blob_store.close()
        writer.close()

    def test_history_publication_is_reserved_before_hygiene_can_scan(
        self,
        tmp_path: Path,
    ) -> None:
        key = _key()
        blob_store = BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(lambda _ref: key),
        )
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=_InMemoryAnchor(),
            hmac_key=secrets.token_bytes(32),
        )
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=DataProjectionState(),
            active_key_id="k1",
        )
        record = _record()
        payload = _payload(record)
        published = threading.Event()
        release_readback = threading.Event()
        hygiene_done = threading.Event()
        errors: list[BaseException] = []
        original_read = blob_store.read

        def blocked_readback(ref):
            published.set()
            assert release_readback.wait(5)
            return original_read(ref)

        def publish() -> None:
            try:
                svc.record_history(record, payload)
            except BaseException as exc:
                errors.append(exc)

        def collect() -> None:
            try:
                svc.quarantine_unreferenced_blobs()
            except BaseException as exc:
                errors.append(exc)
            finally:
                hygiene_done.set()

        with patch.object(blob_store, "read", side_effect=blocked_readback):
            publish_thread = threading.Thread(target=publish)
            hygiene_thread = threading.Thread(target=collect)
            publish_thread.start()
            try:
                assert published.wait(5)
                hygiene_thread.start()
                assert not hygiene_done.wait(0.1)
            finally:
                release_readback.set()
                publish_thread.join(5)
                hygiene_thread.join(5)

        assert not publish_thread.is_alive()
        assert not hygiene_thread.is_alive()
        assert errors == []
        assert svc.get_history_payload(record.record_id) == payload
        blob_store.close()
        writer.close()


class TestBlobPublishFailure:
    def test_blob_failure_does_not_commit_event(self, tmp_path: Path) -> None:
        key = _key()
        provider = AesGcmEncryptionProvider(lambda _ref: key)
        blob_store = BlobStore(tmp_path / "blobs", provider)
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=_InMemoryAnchor(),
            hmac_key=secrets.token_bytes(32),
        )
        state = DataProjectionState()
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=state,
            active_key_id="k1",
        )
        record = _record()
        payload = _payload(record)
        with patch.object(blob_store, "stage_and_publish", side_effect=BlobPublishError("disk full")):
            with pytest.raises(BlobPublishError):
                svc.record_history(record, payload)
        assert record.record_id not in state.history_records
        assert state.cursor_sequence == 0
        blob_store.close()
        writer.close()

    def test_stage_and_readback_run_outside_data_and_writer_locks(
        self,
        tmp_path: Path,
    ) -> None:
        key = _key()
        blob_store = BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(lambda _ref: key),
        )
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=_InMemoryAnchor(),
            hmac_key=secrets.token_bytes(32),
        )
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=DataProjectionState(),
            active_key_id="k1",
        )
        held = threading.local()

        class _TrackingRLock:
            def __init__(self):
                self._lock = threading.RLock()

            def __enter__(self):
                self._lock.acquire()
                held.count = getattr(held, "count", 0) + 1
                return self

            def __exit__(self, *_args):
                held.count -= 1
                self._lock.release()

        svc._mutex = _TrackingRLock()  # type: ignore[assignment]  # noqa: SLF001
        writer._transaction_mutex = _TrackingRLock()  # type: ignore[assignment]  # noqa: SLF001
        original_stage = blob_store.stage_and_publish
        original_read = blob_store.read

        def stage(*args, **kwargs):
            assert getattr(held, "count", 0) == 0
            return original_stage(*args, **kwargs)

        def read(*args, **kwargs):
            assert getattr(held, "count", 0) == 0
            return original_read(*args, **kwargs)

        with (
            patch.object(blob_store, "stage_and_publish", side_effect=stage),
            patch.object(blob_store, "read", side_effect=read),
        ):
            svc.record_history(_record(), _payload(_record()))
        blob_store.close()
        writer.close()

    def test_failed_readback_quarantines_only_its_concurrent_blob(
        self,
        tmp_path: Path,
    ) -> None:
        key = _key()
        blob_store = BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(lambda _ref: key),
        )
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=_InMemoryAnchor(),
            hmac_key=secrets.token_bytes(32),
        )
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=DataProjectionState(),
            active_key_id="k1",
        )
        second_context = replace(
            _context(),
            task_id="task_2",
            run_id="run_2",
            attempt_id="attempt_2",
        )
        second_record = ExecutionHistoryRecordV1.from_attempt(
            second_context,
            ended_at="2026-07-12T01:30:00+00:00",
            status="completed",
            safe_summary=SafeExecutionSummary.build(status="completed"),
            prompt_tokens=0,
            completion_tokens=0,
            predecessor_sequence=0,
            predecessor_hash="",
            shard_timezone="UTC",
        )
        records = (_record(), second_record)
        payloads = tuple(_payload(record) for record in records)
        bad_hash = hashlib.sha256(
            json.dumps(
                payloads[0].to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        barrier = threading.Barrier(2)
        original_read = blob_store.read

        def readback(ref):
            barrier.wait(timeout=3)
            if ref.content_hash == bad_hash:
                return b"corrupt readback"
            return original_read(ref)

        def stage(index):
            try:
                return svc.stage_history_payload(records[index], payloads[index])
            except DataBlobError:
                return None

        with patch.object(blob_store, "read", side_effect=readback):
            with ThreadPoolExecutor(max_workers=2) as pool:
                staged = tuple(pool.map(stage, range(2)))

        assert staged[0] is None
        assert staged[1] is not None
        assert staged[1].blob_ref.blob_id in set(blob_store.iter_blob_ids())
        blob_store.close()
        writer.close()


class TestHeadRaceRetry:
    def test_stale_head_causes_integrity_error(self, tmp_path: Path) -> None:
        key = _key()
        provider = AesGcmEncryptionProvider(lambda _ref: key)
        blob_store = BlobStore(tmp_path / "blobs", provider)
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=_InMemoryAnchor(),
            hmac_key=secrets.token_bytes(32),
        )
        state = DataProjectionState()
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=state,
            active_key_id="k1",
        )
        svc.start_attempt(_context())
        state.cursor_sequence = 0
        state.cursor_hash = ""
        ctx2 = ExecutionAttemptContext(
            tenant_key="tenant_1",
            agent_id="agt_alpha",
            owner_principal_id="principal_owner",
            requester_principal_id="principal_requester",
            task_id="task_2",
            run_id="run_2",
            attempt_id="attempt_2",
            message_id="message_2",
            thread_root_id="thread_2",
            chat_id="chat_2",
            tool="codex",
            model="gpt-test",
            effort="high",
            started_at="2026-07-12T00:30:00+00:00",
            terminal_epoch=1,
        )
        from src.autonomous.journal.frame import JournalIntegrityError
        with pytest.raises(JournalIntegrityError, match="head mismatch"):
            svc.start_attempt(ctx2)
        blob_store.close()
        writer.close()


class TestMultipleTerminalStatuses:
    @pytest.mark.parametrize("status", ["completed", "failed", "canceled", "timeout", "action_required"])
    def test_all_terminal_statuses_commit(self, tmp_path: Path, status: str) -> None:
        key = _key()
        provider = AesGcmEncryptionProvider(lambda _ref: key)
        blob_store = BlobStore(tmp_path / "blobs", provider)
        writer = JournalWriter.open(
            tmp_path / "journal",
            anchor=_InMemoryAnchor(),
            hmac_key=secrets.token_bytes(32),
        )
        state = DataProjectionState()
        svc = EmployeeDataService(
            writer=writer,
            blob_store=blob_store,
            data_state=state,
            active_key_id="k1",
        )
        ctx = ExecutionAttemptContext(
            tenant_key="tenant_1",
            agent_id="agt_alpha",
            owner_principal_id="principal_owner",
            requester_principal_id="principal_requester",
            task_id="task_1",
            run_id="run_1",
            attempt_id=f"attempt_{status}",
            message_id="message_1",
            thread_root_id="",
            chat_id="chat_1",
            tool="codex",
            model="gpt-test",
            effort="high",
            started_at="2026-07-12T00:30:00+00:00",
            terminal_epoch=1,
        )
        record = ExecutionHistoryRecordV1.from_attempt(
            ctx,
            ended_at="2026-07-12T01:30:00+00:00",
            status=status,
            safe_summary=SafeExecutionSummary.build(status=status, tool_count=0, attachment_count=0),
            prompt_tokens=0,
            completion_tokens=0,
            tool_usage=(),
            predecessor_sequence=0,
            predecessor_hash="",
            shard_timezone="UTC",
        )
        payload = ExecutionHistoryPayloadV1(
            record_id=record.record_id,
            occurrence_key=record.occurrence_key,
            request_text="",
            result_text="",
            error_detail="" if status == "completed" else "timed out",
        )
        result = svc.record_history(record, payload)
        assert result.commit_result.state == CommitState.ANCHORED
        metadata = state.history_records[record.record_id]
        assert metadata.status == status
        blob_store.close()
        writer.close()
