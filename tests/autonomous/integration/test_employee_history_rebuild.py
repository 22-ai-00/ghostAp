"""Integration tests for daily history rebuild and ACL-gated queries."""

from __future__ import annotations

import json
import secrets
from pathlib import Path

import pytest

from src.autonomous.data.materializer import DailyHistoryMaterializer
from src.autonomous.data.models import (
    ExecutionAttemptContext,
    ExecutionHistoryPayloadV1,
    ExecutionHistoryRecordV1,
    SafeExecutionSummary,
)
from src.autonomous.data.projection import DataProjectionState
from src.autonomous.data.query import (
    AuthenticatedDataRequest,
    EmployeeDataRequestContextFactory,
    HistoryQuerySpec,
    HistoryRangeQuery,
    QueryDeniedError,
)
from src.autonomous.data.service import EmployeeDataService
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
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


def _service(tmp_path: Path):
    key = secrets.token_bytes(32)
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
    return svc, state, writer, blob_store


def _publish_record(svc, *, status="completed", day_offset=0, attempt_suffix=""):
    ended = f"2026-07-{12 + day_offset:02d}T01:30:00+00:00"
    ctx = ExecutionAttemptContext(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        owner_principal_id="principal_owner",
        requester_principal_id="principal_requester",
        task_id="task_1",
        run_id="run_1",
        attempt_id=f"attempt_{status}{attempt_suffix}",
        message_id="message_1",
        thread_root_id="",
        chat_id="chat_1",
        tool="codex",
        model="gpt-test",
        effort="high",
        started_at=f"2026-07-{12 + day_offset:02d}T00:30:00+00:00",
        terminal_epoch=1,
    )
    record = ExecutionHistoryRecordV1.from_attempt(
        ctx,
        ended_at=ended,
        status=status,
        safe_summary=SafeExecutionSummary.build(status=status, tool_count=0, attachment_count=0),
        prompt_tokens=10,
        completion_tokens=5,
        tool_usage=(),
        predecessor_sequence=0,
        predecessor_hash="",
        shard_timezone="UTC",
    )
    payload = ExecutionHistoryPayloadV1(
        record_id=record.record_id,
        occurrence_key=record.occurrence_key,
        request_text="test",
        result_text="ok",
        error_detail="",
    )
    svc.record_history(record, payload)
    return record


class TestDailyHistoryMaterializer:
    def test_materialize_day_writes_deterministic_jsonl(self, tmp_path: Path) -> None:
        svc, state, writer, blob_store = _service(tmp_path)
        _publish_record(svc)
        mat = DailyHistoryMaterializer(tmp_path / "agents")
        manifest = mat.materialize_day(state, "tenant_1", "agt_alpha", "2026-07-12")
        assert manifest is not None
        assert manifest.row_count == 1
        assert manifest.day == "2026-07-12"
        shard_path = tmp_path / "agents" / "agt_alpha" / "history" / "2026-07-12.jsonl"
        assert shard_path.exists()
        rows = shard_path.read_bytes().strip().split(b"\n")
        assert len(rows) == 1
        parsed = json.loads(rows[0])
        assert parsed["status"] == "completed"
        assert "request_text" not in parsed
        blob_store.close()
        writer.close()

    def test_materialize_all_rebuilds_multiple_days(self, tmp_path: Path) -> None:
        svc, state, writer, blob_store = _service(tmp_path)
        _publish_record(svc, day_offset=0, attempt_suffix="_d0")
        _publish_record(svc, status="failed", day_offset=1, attempt_suffix="_d1")
        mat = DailyHistoryMaterializer(tmp_path / "agents")
        manifests = mat.materialize_all(state)
        assert len(manifests) == 2
        assert manifests[0].day == "2026-07-12"
        assert manifests[1].day == "2026-07-13"
        blob_store.close()
        writer.close()

    def test_empty_day_returns_none(self, tmp_path: Path) -> None:
        state = DataProjectionState()
        mat = DailyHistoryMaterializer(tmp_path / "agents")
        assert mat.materialize_day(state, "tenant_1", "agt_alpha", "2026-07-20") is None


class TestACLHistoryQuery:
    def test_nonempty_chat_without_trusted_membership_is_denied(
        self,
        tmp_path: Path,
    ) -> None:
        svc, state, writer, blob_store = _service(tmp_path)
        _publish_record(svc)
        factory = EmployeeDataRequestContextFactory(
            admin_principal_ids=frozenset(),
            main_bot_app_id="main_bot",
        )
        query = HistoryRangeQuery(state=state, context_factory=factory)
        request = AuthenticatedDataRequest(
            principal_id="random_user",
            tenant_key="tenant_1",
            receiving_bot_app_id="employee_bot",
            chat_id="chat_1",
            chat_type="group",
            thread_root_id="",
            requested_agent_id="agt_alpha",
        )

        with pytest.raises(QueryDeniedError):
            query.query(
                request,
                HistoryQuerySpec(start_day="2026-07-12", end_day="2026-07-12"),
            )
        blob_store.close()
        writer.close()

    def test_admin_sees_all_records(self, tmp_path: Path) -> None:
        svc, state, writer, blob_store = _service(tmp_path)
        _publish_record(svc, attempt_suffix="_1")
        _publish_record(svc, status="failed", attempt_suffix="_2")
        factory = EmployeeDataRequestContextFactory(
            admin_principal_ids=frozenset({"admin_1"}),
            main_bot_app_id="main_bot",
        )
        query = HistoryRangeQuery(
            state=state,
            context_factory=factory,
            max_range_days=31,
        )
        request = AuthenticatedDataRequest(
            principal_id="admin_1",
            tenant_key="tenant_1",
            receiving_bot_app_id="main_bot",
            chat_id="dm_chat",
            chat_type="p2p",
            thread_root_id="",
            requested_agent_id="agt_alpha",
        )
        result = query.query(request, HistoryQuerySpec(start_day="2026-07-12", end_day="2026-07-12"))
        assert len(result.records) == 2
        assert result.total_available == 2
        blob_store.close()
        writer.close()

    def test_non_admin_non_owner_denied(self, tmp_path: Path) -> None:
        svc, state, writer, blob_store = _service(tmp_path)
        _publish_record(svc)
        factory = EmployeeDataRequestContextFactory(
            admin_principal_ids=frozenset({"admin_1"}),
            main_bot_app_id="main_bot",
        )
        query = HistoryRangeQuery(
            state=state,
            context_factory=factory,
        )
        request = AuthenticatedDataRequest(
            principal_id="random_user",
            tenant_key="tenant_1",
            receiving_bot_app_id="main_bot",
            chat_id="",
            chat_type="group",
            thread_root_id="",
            requested_agent_id="agt_alpha",
        )
        with pytest.raises(QueryDeniedError):
            query.query(request, HistoryQuerySpec(start_day="2026-07-12", end_day="2026-07-12"))
        blob_store.close()
        writer.close()

    def test_cross_tenant_denied(self, tmp_path: Path) -> None:
        svc, state, writer, blob_store = _service(tmp_path)
        _publish_record(svc)
        factory = EmployeeDataRequestContextFactory(
            admin_principal_ids=frozenset({"admin_1"}),
            main_bot_app_id="main_bot",
        )
        query = HistoryRangeQuery(
            state=state,
            context_factory=factory,
        )
        request = AuthenticatedDataRequest(
            principal_id="admin_1",
            tenant_key="tenant_OTHER",
            receiving_bot_app_id="main_bot",
            chat_id="dm_chat",
            chat_type="p2p",
            thread_root_id="",
            requested_agent_id="agt_alpha",
        )
        with pytest.raises(QueryDeniedError):
            query.query(request, HistoryQuerySpec(start_day="2026-07-12", end_day="2026-07-12"))
        blob_store.close()
        writer.close()

    def test_range_exceeds_max_denied(self, tmp_path: Path) -> None:
        state = DataProjectionState()
        factory = EmployeeDataRequestContextFactory(
            admin_principal_ids=frozenset({"admin_1"}),
            main_bot_app_id="main_bot",
        )
        query = HistoryRangeQuery(
            state=state,
            context_factory=factory,
            max_range_days=7,
        )
        request = AuthenticatedDataRequest(
            principal_id="admin_1",
            tenant_key="tenant_1",
            receiving_bot_app_id="main_bot",
            chat_id="dm",
            chat_type="p2p",
            thread_root_id="",
            requested_agent_id="agt_alpha",
        )
        with pytest.raises(QueryDeniedError, match="range"):
            query.query(request, HistoryQuerySpec(start_day="2026-07-01", end_day="2026-07-31"))

    def test_pagination_with_cursor(self, tmp_path: Path) -> None:
        svc, state, writer, blob_store = _service(tmp_path)
        for i in range(5):
            _publish_record(svc, attempt_suffix=f"_{i}")
        factory = EmployeeDataRequestContextFactory(
            admin_principal_ids=frozenset({"admin_1"}),
            main_bot_app_id="main_bot",
        )
        query = HistoryRangeQuery(
            state=state,
            context_factory=factory,
        )
        request = AuthenticatedDataRequest(
            principal_id="admin_1",
            tenant_key="tenant_1",
            receiving_bot_app_id="main_bot",
            chat_id="dm",
            chat_type="p2p",
            thread_root_id="",
            requested_agent_id="agt_alpha",
        )
        page1 = query.query(request, HistoryQuerySpec(start_day="2026-07-12", end_day="2026-07-12", page_size=2))
        assert len(page1.records) == 2
        assert page1.next_cursor is not None
        page2 = query.query(request, HistoryQuerySpec(start_day="2026-07-12", end_day="2026-07-12", page_size=2, cursor=page1.next_cursor))
        assert len(page2.records) == 2
        page3 = query.query(request, HistoryQuerySpec(start_day="2026-07-12", end_day="2026-07-12", page_size=2, cursor=page2.next_cursor))
        assert len(page3.records) == 1
        assert page3.next_cursor is None
        blob_store.close()
        writer.close()

    def test_owner_sees_own_records(self, tmp_path: Path) -> None:
        svc, state, writer, blob_store = _service(tmp_path)
        _publish_record(svc)
        factory = EmployeeDataRequestContextFactory(
            admin_principal_ids=frozenset(),
            main_bot_app_id="main_bot",
        )
        query = HistoryRangeQuery(
            state=state,
            context_factory=factory,
        )
        request = AuthenticatedDataRequest(
            principal_id="principal_owner",
            tenant_key="tenant_1",
            receiving_bot_app_id="employee_bot",
            chat_id="some_chat",
            chat_type="group",
            thread_root_id="",
            requested_agent_id="agt_alpha",
        )
        result = query.query(request, HistoryQuerySpec(start_day="2026-07-12", end_day="2026-07-12"))
        assert len(result.records) == 1
        blob_store.close()
        writer.close()
