"""Unit tests for Reporter outbox."""

from __future__ import annotations

import pytest

from src.autonomous.domain import ProgressSnapshot, RunState
from src.autonomous.reporter.reporter import (
    DeliveryState,
    OutboxEntry,
    Reporter,
    ReportType,
)


class TestReporter:
    @pytest.mark.asyncio
    async def test_enqueue_returns_entry_id(self) -> None:
        reporter = Reporter(deliver_fn=lambda t, p: True)
        entry_id = await reporter.enqueue(
            ReportType.PROGRESS_UPDATE, "channel_1", {"msg": "hi"}
        )
        assert entry_id != ""
        assert entry_id.startswith("out_")

    @pytest.mark.asyncio
    async def test_flush_delivers_pending(self) -> None:
        delivered_payloads: list[dict] = []

        def deliver(target: str, payload: dict) -> bool:
            delivered_payloads.append(payload)
            return True

        reporter = Reporter(deliver_fn=deliver)
        await reporter.enqueue(ReportType.RUN_STARTED, "ch1", {"run_id": "r1"})
        await reporter.enqueue(ReportType.PROGRESS_UPDATE, "ch1", {"step": 1})

        count = await reporter.flush()
        assert count == 2
        assert len(delivered_payloads) == 2

    @pytest.mark.asyncio
    async def test_flush_retries_failed(self) -> None:
        call_count = 0

        def deliver(target: str, payload: dict) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count > 1  # Fail first, succeed second

        reporter = Reporter(deliver_fn=deliver, backoff_base=0.0)
        await reporter.enqueue(ReportType.RUN_COMPLETED, "ch1", {"run_id": "r1"})

        # First flush fails
        count1 = await reporter.flush()
        assert count1 == 0
        assert len(reporter.get_pending()) == 1

        # Second flush succeeds
        count2 = await reporter.flush()
        assert count2 == 1
        assert len(reporter.get_pending()) == 0

    @pytest.mark.asyncio
    async def test_dead_letter_after_max_attempts(self) -> None:
        reporter = Reporter(deliver_fn=lambda t, p: False, max_retries=2, backoff_base=0.0)
        await reporter.enqueue(ReportType.RUN_FAILED, "ch1", {"run_id": "r1"})

        await reporter.flush()  # attempt 1
        await reporter.flush()  # attempt 2 -> dead letter

        assert len(reporter.get_pending()) == 0
        assert len(reporter.get_dead_letters()) == 1

    @pytest.mark.asyncio
    async def test_idempotency_key_dedup(self) -> None:
        reporter = Reporter(deliver_fn=lambda t, p: True)
        await reporter.enqueue(
            ReportType.PROGRESS_UPDATE, "ch1", {"msg": "a"}, idempotency_key="key_1"
        )
        await reporter.flush()

        # Second enqueue with same key should be skipped
        id2 = await reporter.enqueue(
            ReportType.PROGRESS_UPDATE, "ch1", {"msg": "b"}, idempotency_key="key_1"
        )
        assert id2 == ""  # Skipped

    @pytest.mark.asyncio
    async def test_report_progress_convenience(self) -> None:
        reporter = Reporter(deliver_fn=lambda t, p: True)
        snapshot = ProgressSnapshot(
            run_id="run_1",
            run_state=RunState.EXECUTING,
            completed_steps=2,
            total_steps=5,
        )
        entry_id = await reporter.report_progress("ch1", snapshot)
        assert entry_id.startswith("out_")

    @pytest.mark.asyncio
    async def test_report_completion(self) -> None:
        reporter = Reporter(deliver_fn=lambda t, p: True)
        entry_id = await reporter.report_completion("ch1", "run_1", {"output": "done"})
        assert entry_id.startswith("out_")

    @pytest.mark.asyncio
    async def test_report_failure(self) -> None:
        reporter = Reporter(deliver_fn=lambda t, p: True)
        entry_id = await reporter.report_failure("ch1", "run_1", "timeout")
        assert entry_id.startswith("out_")

    @pytest.mark.asyncio
    async def test_request_approval(self) -> None:
        reporter = Reporter(deliver_fn=lambda t, p: True)
        entry_id = await reporter.request_approval("ch1", "appr_1", {"plan": "..."})
        assert entry_id.startswith("out_")

    @pytest.mark.asyncio
    async def test_get_delivered(self) -> None:
        reporter = Reporter(deliver_fn=lambda t, p: True)
        await reporter.enqueue(ReportType.RUN_STARTED, "ch1", {})
        await reporter.flush()
        assert len(reporter.get_delivered()) == 1

    def test_outbox_entry_serialization(self) -> None:
        entry = OutboxEntry(
            entry_id="out_test",
            report_type=ReportType.PROGRESS_UPDATE,
            target="ch1",
            payload={"key": "val"},
            state=DeliveryState.PENDING,
        )
        d = entry.to_dict()
        restored = OutboxEntry.from_dict(d)
        assert restored.entry_id == "out_test"
        assert restored.report_type == ReportType.PROGRESS_UPDATE
        assert restored.target == "ch1"
        assert restored.payload == {"key": "val"}

    @pytest.mark.asyncio
    async def test_exception_in_deliver_fn(self) -> None:
        def deliver(target: str, payload: dict) -> bool:
            raise RuntimeError("network error")

        reporter = Reporter(deliver_fn=deliver, backoff_base=0.0)
        await reporter.enqueue(ReportType.BLOCKED, "ch1", {})
        count = await reporter.flush()
        assert count == 0
        pending = reporter.get_pending()
        assert len(pending) == 1
        assert "network error" in pending[0].error
