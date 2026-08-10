"""Integration test: full employee data composition end-to-end."""

from __future__ import annotations

import base64
import json
import secrets
import threading
from pathlib import Path

import pytest
from pydantic import SecretStr

from src.autonomous.data.composition import (
    EmployeeDataComposition,
    LegacyEmployeeDataUnsupportedError,
    build_employee_data_composition,
)
from src.autonomous.data.models import DataKind, ExecutionAttemptContext
from src.autonomous.data.ports import (
    AuthenticatedExecutionTerminal,
    PublishEmployeeDocumentCommand,
)
from src.autonomous.data.query import AuthenticatedDataRequest, HistoryQuerySpec
from src.autonomous.journal.writer import JournalWriter


class _InMemoryAnchor:
    def __init__(self) -> None:
        self._sequence = 0
        self._hash = "0" * 64

    def read(self):
        from src.autonomous.journal.anchor import AnchorState
        return AnchorState(self._sequence, self._hash)

    def compare_and_swap(self, es, eh, ns, nh) -> bool:
        if self._sequence == es and self._hash == eh:
            self._sequence = ns
            self._hash = nh
            return True
        return False


class _FakeSettings:
    def __init__(self, tmp_path: Path) -> None:
        key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        self.autonomous_data_keys = SecretStr(
            json.dumps({"version": 1, "keys": {"k1": key}})
        )
        self.autonomous_data_active_key_id = "k1"
        self.autonomous_data_blob_dir = str(tmp_path / "data-blobs")
        self.autonomous_history_timezone = "UTC"
        self.autonomous_history_max_range_days = 31
        self.autonomous_history_page_size = 50


@pytest.fixture
def composition(tmp_path: Path) -> EmployeeDataComposition:
    settings = _FakeSettings(tmp_path)
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=_InMemoryAnchor(),
        hmac_key=secrets.token_bytes(32),
    )
    comp = build_employee_data_composition(
        settings=settings,
        writer=writer,
        admin_principal_ids=frozenset({"admin_1"}),
        main_bot_app_id="main_bot",
        agents_root=tmp_path / "agents",
    )
    yield comp
    comp.service._blob_store.close()
    writer.close()


class TestFullComposition:
    def test_production_composition_has_independent_canonical_authority(
        self,
        composition: EmployeeDataComposition,
    ) -> None:
        authority = composition.state.data_authority

        assert authority.mode == "canonical"
        assert authority.epoch == 1
        assert authority.cutover_sequence > 0

    def test_start_attempt_record_terminal_and_query(
        self, composition: EmployeeDataComposition
    ) -> None:
        composition.service._shard_timezone = "Asia/Shanghai"
        ctx = ExecutionAttemptContext(
            tenant_key="tenant_1",
            agent_id="agt_alpha",
            owner_principal_id="principal_owner",
            requester_principal_id="principal_requester",
            task_id="task_1",
            run_id="run_1",
            attempt_id="attempt_1",
            message_id="msg_1",
            thread_root_id="",
            chat_id="chat_1",
            tool="codex",
            model="gpt-test",
            effort="high",
            started_at="2026-07-12T00:30:00+00:00",
            terminal_epoch=1,
        )
        composition.service.start_attempt(ctx)
        terminal = AuthenticatedExecutionTerminal(
            attempt_id="attempt_1",
            status="completed",
            request_text="do something",
            result_text="done",
            error_detail="",
            prompt_tokens=100,
            completion_tokens=50,
        )
        composition.record_terminal(terminal)
        assert len(composition.state.history_records) == 1
        history_event = next(
            event
            for frame in composition.service._writer.replay()
            for event in frame.events
            if event.event_type == "employee.history.recorded"
        )
        assert history_event.payload["shard_timezone"] == "Asia/Shanghai"
        request = AuthenticatedDataRequest(
            principal_id="admin_1",
            tenant_key="tenant_1",
            receiving_bot_app_id="main_bot",
            chat_id="dm",
            chat_type="p2p",
            thread_root_id="",
            requested_agent_id="agt_alpha",
        )
        today = next(iter(composition.state.history_records.values())).shard_day
        result = composition.query.query(
            request,
            HistoryQuerySpec(start_day=today, end_day=today),
        )
        assert result.total_available == 1
        assert result.records[0].status == "completed"
        audits = tuple(composition.state.data_read_audits.values())
        assert len(audits) == 1
        assert audits[0].operation == "history_query"
        assert audits[0].outcome == "granted"

    def test_unmigrated_legacy_data_is_explicitly_unsupported(
        self, tmp_path: Path
    ) -> None:
        settings = _FakeSettings(tmp_path)
        legacy = tmp_path / "manual-agents" / "agt_alpha" / "memory"
        legacy.mkdir(parents=True)
        (legacy / "MEMORY.md").write_text("legacy")
        writer = JournalWriter.open(
            tmp_path / "manual-journal",
            anchor=_InMemoryAnchor(),
            hmac_key=secrets.token_bytes(32),
        )
        with pytest.raises(
            LegacyEmployeeDataUnsupportedError,
            match="offline migration",
        ):
            build_employee_data_composition(
                settings=settings,
                writer=writer,
                admin_principal_ids=frozenset(),
                main_bot_app_id="main_bot",
                agents_root=tmp_path / "manual-agents",
            )
        writer.close()

    def test_publish_document_materializes_file(
        self, composition: EmployeeDataComposition
    ) -> None:
        command = PublishEmployeeDocumentCommand(
            agent_id="agt_alpha",
            tenant_key="tenant_1",
            owner_principal_id="principal_owner",
            kind=DataKind.L1_MEMORY,
            source_id="l1_memory",
            content=b"# Memory content",
            content_type="text/markdown",
        )
        composition.publish_document(command)
        assert len(composition.state.employee_documents) == 1
        l1 = composition.memory_facade.read_l1("agt_alpha", "tenant_1")
        assert l1 == "# Memory content"

    def test_unanchored_attempt_raises(
        self, composition: EmployeeDataComposition
    ) -> None:
        terminal = AuthenticatedExecutionTerminal(
            attempt_id="nonexistent_attempt",
            status="failed",
            request_text="",
            result_text="",
            error_detail="no such attempt",
        )
        with pytest.raises(ValueError, match="no anchored attempt"):
            composition.record_terminal(terminal)

    def test_memory_summary_publishes(
        self, composition: EmployeeDataComposition
    ) -> None:
        command = PublishEmployeeDocumentCommand(
            agent_id="agt_alpha",
            tenant_key="tenant_1",
            owner_principal_id="principal_owner",
            kind=DataKind.MEMORY_SUMMARY,
            source_id="",
            content=b"# Chat summary",
            content_type="text/markdown",
            chat_id="chat_1",
            thread_root_id="",
        )
        composition.publish_document(command)
        summary = composition.memory_facade.read_memory_summary(
            "agt_alpha", "tenant_1", "chat_1"
        )
        assert summary == "# Chat summary"

    def test_memory_summary_versions_and_retries_are_idempotent(
        self,
        composition: EmployeeDataComposition,
    ) -> None:
        def command(content: bytes, key: str) -> PublishEmployeeDocumentCommand:
            return PublishEmployeeDocumentCommand(
                agent_id="agt_alpha",
                tenant_key="tenant_1",
                owner_principal_id="principal_owner",
                kind=DataKind.MEMORY_SUMMARY,
                source_id="",
                content=content,
                content_type="text/markdown",
                chat_id="chat_1",
                thread_root_id="root_1",
                idempotency_key=key,
            )

        composition.publish_document(command(b"first", "attempt_1"))
        composition.publish_document(command(b"first", "attempt_1"))
        composition.publish_document(command(b"second", "attempt_2"))

        records = sorted(
            composition.state.employee_documents.values(),
            key=lambda item: item.version,
        )
        assert [record.version for record in records] == [1, 2]
        assert records[1].previous_document_id == records[0].document_id
        assert composition.memory_facade.read_memory_summary(
            "agt_alpha",
            "tenant_1",
            "chat_1",
            "root_1",
        ) == "second"
        with pytest.raises(ValueError, match="idempotency conflict"):
            composition.publish_document(command(b"different", "attempt_2"))

    def test_reasoning_preserves_task_source_and_versions(
        self,
        composition: EmployeeDataComposition,
    ) -> None:
        def command(content: bytes, key: str) -> PublishEmployeeDocumentCommand:
            return PublishEmployeeDocumentCommand(
                agent_id="agt_alpha",
                tenant_key="tenant_1",
                owner_principal_id="principal_owner",
                kind=DataKind.REASONING,
                source_id="task_alpha",
                content=content,
                content_type="application/json",
                idempotency_key=key,
            )

        composition.publish_document(command(b'{"attempt":1}', "attempt_1"))
        composition.publish_document(command(b'{"attempt":2}', "attempt_2"))

        records = sorted(
            composition.state.employee_documents.values(),
            key=lambda item: item.version,
        )
        assert [record.source_id for record in records] == ["task_alpha", "task_alpha"]
        assert [record.version for record in records] == [1, 2]

    def test_gc_serializes_with_inflight_publish_before_quarantine(
        self,
        composition: EmployeeDataComposition,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        commit_entered = threading.Event()
        release_commit = threading.Event()
        gc_done = threading.Event()
        errors: list[Exception] = []
        gc_results: list[int] = []
        original_commit = composition.service._writer.commit

        def blocked_commit(*args, **kwargs):
            commit_entered.set()
            assert release_commit.wait(5)
            return original_commit(*args, **kwargs)

        monkeypatch.setattr(composition.service._writer, "commit", blocked_commit)
        command = PublishEmployeeDocumentCommand(
            agent_id="agt_alpha",
            tenant_key="tenant_1",
            owner_principal_id="principal_owner",
            kind=DataKind.L1_MEMORY,
            source_id="l1_memory",
            content=b"# GC-safe memory",
            content_type="text/markdown",
        )

        def publish() -> None:
            try:
                composition.publish_document(command)
            except Exception as exc:
                errors.append(exc)

        def collect() -> None:
            try:
                gc_results.append(composition.gc_unreferenced_blobs())
            except Exception as exc:
                errors.append(exc)
            finally:
                gc_done.set()

        publish_thread = threading.Thread(target=publish)
        gc_thread = threading.Thread(target=collect)
        publish_thread.start()
        assert commit_entered.wait(5)
        gc_thread.start()
        assert not gc_done.wait(0.1)
        release_commit.set()
        publish_thread.join(5)
        gc_thread.join(5)

        assert not publish_thread.is_alive()
        assert not gc_thread.is_alive()
        assert errors == []
        assert gc_results == [0]
        assert composition.memory_facade.read_l1(
            "agt_alpha",
            "tenant_1",
        ) == "# GC-safe memory"
