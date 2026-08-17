from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

import src.autonomous.journal.writer as writer_module
from src.autonomous.acceptance.main_bot_audit import MainBotSendAuditLog
from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.frame import JournalIntegrityError
from src.autonomous.journal.writer import JournalWriter


def _audit(tmp_path: Path) -> MainBotSendAuditLog:
    writer = JournalWriter.open(
        tmp_path / "audit",
        anchor=FileAnchor(tmp_path / "audit.anchor"),
        hmac_key=b"a" * 32,
        writer_epoch=1,
    )
    return MainBotSendAuditLog(writer)


def test_main_bot_audit_counts_matching_and_unknown_tenant_attempts(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    audit.record_attempt("tenant-a", "reply", "om_message", attempted_at=100.0)
    audit.record_attempt("", "create", "oc_chat", attempted_at=101.0)
    audit.record_attempt("tenant-b", "patch", "om_other", attempted_at=102.0)

    assert audit.count_attempts("tenant-a", 99.0, 102.0) == 2
    assert audit.count_attempts("tenant-b", 99.0, 103.0) == 2
    assert audit.count_attempts("tenant-a", 102.0, 103.0) == 0
    raw = audit.writer.journal_path.read_text(encoding="utf-8")
    assert "om_message" not in raw
    assert "tenant-a" not in raw
    audit.close()


def test_main_bot_target_audit_ignores_unrelated_messages(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    audit.record_attempt("tenant-a", "reply", "om_ready_notice", attempted_at=100.0)
    audit.record_attempt("tenant-a", "reply", "om_employee_status", attempted_at=101.0)

    target_hash = hashlib.sha256(b"om_employee_status").hexdigest()

    assert audit.count_target_attempts("tenant-a", target_hash, 99.0, 102.0) == 1
    audit.close()


def test_main_bot_audit_reopens_from_anchored_tail_without_history_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "audit"
    anchor_path = tmp_path / "audit.anchor"
    audit = MainBotSendAuditLog.open(
        directory,
        anchor_path=anchor_path,
        hmac_key=b"a" * 32,
        writer_epoch=1,
    )
    audit.record_attempt("tenant-a", "reply", "om_1", attempted_at=100.0)
    audit.record_attempt("tenant-a", "patch", "om_1", attempted_at=101.0)
    audit.close()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            writer_module.JournalWriter,
            "_load_and_recover",
            lambda *_args, **_kwargs: pytest.fail("audit startup replayed history"),
        )
        reopened = MainBotSendAuditLog.open(
            directory,
            anchor_path=anchor_path,
            hmac_key=b"a" * 32,
            writer_epoch=2,
        )
        try:
            assert reopened.activation_fence_ready is True
            reopened.record_attempt(
                "tenant-a",
                "reply",
                "om_2",
                attempted_at=102.0,
            )
        finally:
            reopened.close()

    verified = MainBotSendAuditLog.open(
        directory,
        anchor_path=anchor_path,
        hmac_key=b"a" * 32,
        writer_epoch=3,
    )
    try:
        assert verified.count_attempts("tenant-a", 99.0, 103.0) == 3
    finally:
        verified.close()


def test_history_replay_failure_marks_fast_opened_audit_incomplete(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "audit"
    anchor_path = tmp_path / "audit.anchor"
    audit = MainBotSendAuditLog.open(
        directory,
        anchor_path=anchor_path,
        hmac_key=b"a" * 32,
        writer_epoch=1,
    )
    audit.record_attempt("tenant-a", "reply", "om_1", attempted_at=100.0)
    audit.record_attempt("tenant-a", "reply", "om_2", attempted_at=101.0)
    audit.close()

    reopened = MainBotSendAuditLog.open(
        directory,
        anchor_path=anchor_path,
        hmac_key=b"a" * 32,
        writer_epoch=2,
    )
    records = (directory / "journal.jsonl").read_bytes().splitlines(keepends=True)
    first = json.loads(records[0])
    digest = first["hmac_digest"]
    first["hmac_digest"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    records[0] = json.dumps(
        first,
        sort_keys=True,
        separators=(",", ":"),
    ).encode() + b"\n"
    (directory / "journal.jsonl").write_bytes(b"".join(records))

    try:
        with pytest.raises(JournalIntegrityError, match="hmac"):
            reopened.count_attempts("tenant-a", 99.0, 102.0)
        assert reopened.activation_fence_ready is False
    finally:
        reopened.close()


def test_activation_fence_serializes_outbound_record_until_commit_window_closes(
    tmp_path: Path,
) -> None:
    audit = _audit(tmp_path)
    target = "om_employee_status"
    target_hash = hashlib.sha256(target.encode()).hexdigest()
    attempted = threading.Event()
    completed = threading.Event()

    def record() -> None:
        attempted.set()
        audit.record_attempt(
            "tenant-a",
            "reply",
            target,
            attempted_at=100.0,
        )
        completed.set()

    with audit.activation_fence("tenant-a", (target_hash,)):
        worker = threading.Thread(target=record)
        worker.start()
        assert attempted.wait(1)
        assert not completed.wait(0.1)

    assert completed.wait(1)
    worker.join(timeout=1)
    assert audit.count_target_attempts("tenant-a", target_hash, 99.0, 101.0) == 1
    audit.close()


def test_activation_fence_timestamps_default_attempt_after_fence_release(
    tmp_path: Path,
) -> None:
    audit = _audit(tmp_path)
    target = "om_employee_status"
    target_hash = hashlib.sha256(target.encode()).hexdigest()
    attempted = threading.Event()
    completed = threading.Event()

    def record() -> None:
        attempted.set()
        audit.record_attempt("tenant-a", "reply", target)
        completed.set()

    with audit.activation_fence("tenant-a", (target_hash,)):
        worker = threading.Thread(target=record)
        worker.start()
        assert attempted.wait(1)
        assert not completed.wait(0.1)
        fence_release_floor = time.time()

    assert completed.wait(1)
    worker.join(timeout=1)
    attempt_event = next(
        event
        for frame in audit.writer.replay()
        for event in frame.events
        if event.event_type == "main_bot.send_attempted"
    )
    assert attempt_event.payload["attempted_at"] >= fence_release_floor
    audit.close()


def test_external_audit_without_activation_fence_fails_closed(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    audit.external_audit = object()

    assert audit.activation_fence_ready is False
    with pytest.raises(RuntimeError, match="activation fence"):
        with audit.activation_fence("tenant-a", ("b" * 64,)):
            raise AssertionError("fence body must not run")

    audit.close()


def test_main_bot_audit_rejects_invalid_query_and_fails_after_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _audit(tmp_path)
    with pytest.raises(ValueError, match="window"):
        audit.count_attempts("tenant-a", 2.0, 1.0)
    with pytest.raises(ValueError, match="target hash"):
        audit.count_target_attempts("tenant-a", "not-a-hash", 1.0, 2.0)

    monkeypatch.setattr(audit.writer, "commit", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        audit.record_attempt("tenant-a", "reply", "om_message", attempted_at=100.0)
    with pytest.raises(RuntimeError, match="incomplete"):
        audit.count_attempts("tenant-a", 99.0, 101.0)
    audit.close()


def test_main_bot_audit_uses_external_cross_replica_ledger_as_authority(
    tmp_path: Path,
) -> None:
    class ExternalAudit:
        def __init__(self) -> None:
            self.records: list[dict[str, object]] = []
            self.count = 3

        def record_main_bot_send_attempt(self, **kwargs) -> None:
            self.records.append(kwargs)

        def count_main_bot_send_attempts(self, tenant_key, start, end) -> int:
            assert (tenant_key, start, end) == ("tenant-a", 99.0, 101.0)
            return self.count

        def count_main_bot_target_send_attempts(
            self,
            tenant_key,
            target_hash,
            start,
            end,
        ) -> int:
            assert tenant_key == "tenant-a"
            assert len(target_hash) == 64
            assert (start, end) == (99.0, 101.0)
            return self.count

    external = ExternalAudit()
    audit = _audit(tmp_path)
    audit.external_audit = external
    audit.record_attempt("tenant-a", "reply", "om_message", attempted_at=100.0)

    assert external.records[0]["tenant_hash"] != "tenant-a"
    assert external.records[0]["target_hash"] != "om_message"
    assert audit.count_attempts("tenant-a", 99.0, 101.0) == 3
    assert (
        audit.count_target_attempts(
            "tenant-a",
            external.records[0]["target_hash"],
            99.0,
            101.0,
        )
        == 3
    )

    external.count = 0
    with pytest.raises(RuntimeError, match="behind local"):
        audit.count_attempts("tenant-a", 99.0, 101.0)
    audit.close()
