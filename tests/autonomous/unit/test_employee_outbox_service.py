from __future__ import annotations

from dataclasses import replace

import pytest

from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.outbox.models import (
    EmployeeCardState,
    EmployeeOutboxSnapshot,
    employee_outbox_id,
)
from src.autonomous.outbox.projection import OutboxProjectionState
from src.autonomous.outbox.service import (
    EmployeeOutboxService,
    OutboxClosedError,
    OutboxConflictError,
)


def _snapshot(**overrides: object) -> EmployeeOutboxSnapshot:
    values: dict[str, object] = {
        "schema_version": 1,
        "outbox_id": employee_outbox_id("tenant-a", "agt_alpha", "attempt-a"),
        "tenant_key": "tenant-a",
        "agent_id": "agt_alpha",
        "attempt_id": "attempt-a",
        "chat_id": "oc_team",
        "thread_root_message_id": "om_root",
        "version": 1,
        "state": EmployeeCardState.QUEUED,
        "title": "修复登录回调",
        "summary": "任务已进入员工队列",
        "progress_percent": 0,
        "card_json": {"schema": "2.0", "body": {"elements": []}},
        "created_at": "2026-07-14T00:00:00Z",
        "terminal_version": 0,
    }
    values.update(overrides)
    return EmployeeOutboxSnapshot(**values)


def _runtime(tmp_path):
    anchor = MemoryAnchor()
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=b"j" * 32,
        writer_epoch=1,
    )
    store = BlobStore(
        tmp_path / "outbox-blobs",
        AesGcmEncryptionProvider(lambda _ref: b"b" * 32),
    )
    service = EmployeeOutboxService(
        writer=writer,
        blob_store=store,
        outbox_state=OutboxProjectionState(),
        active_key_id="k1",
    )
    return service, writer, anchor


def test_append_encrypts_before_anchor_and_duplicate_is_idempotent(tmp_path) -> None:
    service, writer, _anchor = _runtime(tmp_path)
    try:
        snapshot = _snapshot()
        first = service.append_snapshot(snapshot)
        duplicate = service.append_snapshot(snapshot)

        assert first == duplicate
        assert service.get_snapshot(snapshot.outbox_id) == snapshot
        assert len(service.state.by_outbox_id) == 1
        assert len(service.blob_store.iter_blob_ids()) == 1
        frame = tuple(writer.replay())[-1]
        event = frame.events[0]
        assert event.event_type == "employee.outbox.snapshot_appended"
        assert "card_json" not in event.payload
        assert "summary" not in event.payload
        assert service.blob_store.read(first.blob_ref) == snapshot.canonical_bytes
        cloned = service.state.clone()
        assert cloned.by_outbox_id == service.state.by_outbox_id
        assert cloned.by_outbox_id is not service.state.by_outbox_id
    finally:
        service.close()
        writer.close()


def test_ambiguous_anchor_preserves_referenced_snapshot_blob(tmp_path) -> None:
    durable_anchor = MemoryAnchor()

    class RaiseAfterAnchor:
        production_safe = True

        @staticmethod
        def read():
            return durable_anchor.read()

        @staticmethod
        def compare_and_swap(*args):
            assert durable_anchor.compare_and_swap(*args) is True
            raise OSError("directory fsync outcome unknown")

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=RaiseAfterAnchor(),
        hmac_key=b"j" * 32,
        writer_epoch=1,
    )
    service = EmployeeOutboxService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "outbox-blobs",
            AesGcmEncryptionProvider(lambda _ref: b"b" * 32),
        ),
        outbox_state=OutboxProjectionState(),
        active_key_id="k1",
    )
    snapshot = _snapshot()
    try:
        with pytest.raises(Exception):
            service.append_snapshot(snapshot)
        assert len(service.blob_store.iter_blob_ids()) == 1
    finally:
        service.close()
        writer.close()

    writer2 = JournalWriter.open(
        tmp_path / "journal",
        anchor=durable_anchor,
        hmac_key=b"j" * 32,
        writer_epoch=2,
    )
    service2 = EmployeeOutboxService(
        writer=writer2,
        blob_store=BlobStore(
            tmp_path / "outbox-blobs",
            AesGcmEncryptionProvider(lambda _ref: b"b" * 32),
        ),
        outbox_state=OutboxProjectionState(),
        active_key_id="k1",
    )
    try:
        assert service2.get_snapshot(snapshot.outbox_id) == snapshot
    finally:
        service2.close()
        writer2.close()


def test_conflicting_duplicate_and_late_terminal_progress_are_rejected(tmp_path) -> None:
    service, writer, _anchor = _runtime(tmp_path)
    try:
        queued = _snapshot()
        service.append_snapshot(queued)
        with pytest.raises(OutboxConflictError):
            service.append_snapshot(replace(queued, summary="conflict"))
        running = replace(
            queued,
            version=2,
            state=EmployeeCardState.RUNNING,
            progress_percent=50,
        )
        terminal = replace(
            running,
            version=3,
            state=EmployeeCardState.COMPLETED,
            progress_percent=100,
            terminal_version=3,
        )
        service.append_snapshot(running)
        service.append_snapshot(terminal)
        with pytest.raises(OutboxConflictError, match="terminal"):
            service.append_snapshot(replace(terminal, version=4, terminal_version=4, summary="late"))
    finally:
        service.close()
        writer.close()


def test_collaboration_publication_is_bound_to_employee_app_generation(tmp_path) -> None:
    service, writer, _anchor = _runtime(tmp_path)
    try:
        queued = _snapshot()
        service.append_snapshot(queued)
        running = replace(
            queued,
            version=2,
            state=EmployeeCardState.RUNNING,
            progress_percent=50,
        )
        service.append_snapshot(running)
        terminal = replace(
            running,
            version=3,
            state=EmployeeCardState.COMPLETED,
            progress_percent=100,
            terminal_version=3,
        )
        service.append_snapshot(terminal)
        effect = service.prepare_delivery(terminal.outbox_id, terminal.version)
        service.mark_effect_executing(effect.effect_id)
        service.commit_delivery(
            effect.effect_id,
            app_id="cli_alpha",
            generation=3,
            connection_id="conn_alpha",
            message_id="om_employee_contribution",
        )
        assert service.record_collaboration_publication(
            outbox_id=terminal.outbox_id,
            team_run_id="teamrun2_x",
            assignment_id="teamrun2_x:assignment:1",
            causal_event_id="cause_x",
        )
        event = tuple(writer.replay())[-1].events[0]
        assert event.payload["agent_id"] == "agt_alpha"
        assert event.payload["app_id"] == "cli_alpha"
        assert event.payload["generation"] == 3
        assert service.record_collaboration_publication(
            outbox_id=terminal.outbox_id,
            team_run_id="teamrun2_x",
            assignment_id="teamrun2_x:assignment:1",
            causal_event_id="cause_x",
        )
    finally:
        service.close()
        writer.close()


def test_restart_replays_snapshots_and_missing_blob_closes_only_employee(tmp_path) -> None:
    service, writer, anchor = _runtime(tmp_path)
    snapshot = _snapshot()
    record = service.append_snapshot(snapshot)
    service.close()
    writer.close()

    blob_path = tmp_path / "outbox-blobs" / f"{record.blob_ref.blob_id}.blob"
    blob_path.unlink()
    writer2 = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=b"j" * 32,
        writer_epoch=2,
    )
    service2 = EmployeeOutboxService(
        writer=writer2,
        blob_store=BlobStore(
            tmp_path / "outbox-blobs",
            AesGcmEncryptionProvider(lambda _ref: b"b" * 32),
        ),
        outbox_state=OutboxProjectionState(),
        active_key_id="k1",
    )
    try:
        assert ("tenant-a", "agt_alpha") in service2.state.closed_employees
        with pytest.raises(OutboxClosedError):
            service2.append_snapshot(
                _snapshot(
                    outbox_id=employee_outbox_id("tenant-a", "agt_alpha", "attempt-b"),
                    attempt_id="attempt-b",
                )
            )

        other = _snapshot(
            outbox_id=employee_outbox_id("tenant-a", "agt_beta", "attempt-b"),
            agent_id="agt_beta",
            attempt_id="attempt-b",
        )
        service2.append_snapshot(other)
        assert service2.get_snapshot(other.outbox_id) == other
    finally:
        service2.close()
        writer2.close()


def test_recovery_quarantines_orphans_and_gc_tombstones_superseded_versions(
    tmp_path,
) -> None:
    service, writer, _anchor = _runtime(tmp_path)
    try:
        orphan = service.blob_store.stage_and_publish(
            b"orphan",
            {"schema": "employee-outbox-v1", "employee": "agt_orphan"},
            "k1",
        )
        assert orphan.blob_id in service.blob_store.iter_blob_ids()

        queued = _snapshot()
        running = replace(
            queued,
            version=2,
            state=EmployeeCardState.RUNNING,
            progress_percent=50,
        )
        terminal = replace(
            running,
            version=3,
            state=EmployeeCardState.FAILED,
            terminal_version=3,
        )
        service.append_snapshot(queued)
        service.append_snapshot(running)
        service.append_snapshot(terminal)

        assert service.gc_superseded_snapshots() == 0
        effect = service.prepare_delivery(terminal.outbox_id, terminal.version)
        service.mark_effect_executing(effect.effect_id)
        service.commit_delivery(
            effect.effect_id,
            app_id="cli_employee",
            generation=1,
            connection_id="conn_employee",
            message_id="om_employee_card",
        )

        assert service.quarantine_unreferenced_blobs() == 1
        assert service.gc_superseded_snapshots() == 2
        assert service.gc_superseded_snapshots() == 0
        record = service.state.by_outbox_id[queued.outbox_id]
        assert record.tombstoned_versions == frozenset({1, 2})
        assert service.get_snapshot(queued.outbox_id) == terminal
        assert set(service.blob_store.iter_blob_ids()) == {record.snapshots[3].blob_ref.blob_id}
    finally:
        service.close()
        writer.close()
