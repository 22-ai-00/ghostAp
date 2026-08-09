"""Unit tests for Finalization saga."""

from __future__ import annotations

import time

import pytest

from src.autonomous.domain import Effect, EffectState
from src.autonomous.reporter.finalization import (
    DispositionType,
    EffectDisposition,
    Finalization,
    FinalizationRecord,
    FinalizationState,
)


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def write_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


@pytest.fixture
def journal() -> FakeJournal:
    return FakeJournal()


@pytest.fixture
def finalization(journal: FakeJournal) -> Finalization:
    return Finalization(journal=journal)


def _make_effects(count: int = 3) -> list[Effect]:
    return [
        Effect(effect_id=f"eff_{i}", state=EffectState.COMMITTED)
        for i in range(count)
    ]


class TestFinalization:
    @pytest.mark.asyncio
    async def test_start_finalization(self, finalization: Finalization, journal: FakeJournal) -> None:
        effects = _make_effects(3)
        record_id = await finalization.start_finalization("run_1", effects)
        assert record_id.startswith("fin_")

        record = finalization.get_record(record_id)
        assert record is not None
        assert record.run_id == "run_1"
        assert record.total_effects == 3
        assert record.state == FinalizationState.IN_PROGRESS

        event_types = [e[0] for e in journal.events]
        assert "finalization.started" in event_types

    @pytest.mark.asyncio
    async def test_record_disposition(self, finalization: Finalization) -> None:
        effects = _make_effects(2)
        record_id = await finalization.start_finalization("run_1", effects)

        ok = await finalization.record_disposition(
            record_id, "eff_0", DispositionType.COMMITTED
        )
        assert ok is True

        record = finalization.get_record(record_id)
        assert record is not None
        assert record.dispositioned_effects == 1
        assert record.state == FinalizationState.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_auto_complete_on_all_dispositioned(
        self, finalization: Finalization, journal: FakeJournal
    ) -> None:
        effects = _make_effects(2)
        record_id = await finalization.start_finalization("run_1", effects)

        await finalization.record_disposition(record_id, "eff_0", DispositionType.COMMITTED)
        await finalization.record_disposition(record_id, "eff_1", DispositionType.RELEASED)

        record = finalization.get_record(record_id)
        assert record is not None
        assert record.state == FinalizationState.COMPLETE
        assert record.completed_at is not None

        event_types = [e[0] for e in journal.events]
        assert "finalization.complete" in event_types

    @pytest.mark.asyncio
    async def test_duplicate_disposition_rejected(self, finalization: Finalization) -> None:
        effects = _make_effects(2)
        record_id = await finalization.start_finalization("run_1", effects)

        ok1 = await finalization.record_disposition(record_id, "eff_0", DispositionType.COMMITTED)
        assert ok1 is True

        # Duplicate
        ok2 = await finalization.record_disposition(record_id, "eff_0", DispositionType.COMPENSATED)
        assert ok2 is False

    @pytest.mark.asyncio
    async def test_disposition_on_nonexistent_record(self, finalization: Finalization) -> None:
        ok = await finalization.record_disposition(
            "nonexistent_record", "eff_0", DispositionType.COMMITTED
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_check_complete_true(self, finalization: Finalization) -> None:
        effects = _make_effects(1)
        record_id = await finalization.start_finalization("run_1", effects)
        await finalization.record_disposition(record_id, "eff_0", DispositionType.COMMITTED)

        assert await finalization.check_complete("run_1") is True

    @pytest.mark.asyncio
    async def test_check_complete_false(self, finalization: Finalization) -> None:
        effects = _make_effects(2)
        await finalization.start_finalization("run_1", effects)

        assert await finalization.check_complete("run_1") is False

    @pytest.mark.asyncio
    async def test_check_complete_no_record(self, finalization: Finalization) -> None:
        # No finalization record means nothing to finalize -> True
        assert await finalization.check_complete("run_no_record") is True

    @pytest.mark.asyncio
    async def test_timeout_detection(self, journal: FakeJournal) -> None:
        fin = Finalization(journal=journal)
        effects = _make_effects(2)
        record_id = await fin.start_finalization("run_1", effects, timeout_seconds=0.0)

        # Force timeout by making started_at in the past
        record = fin.get_record(record_id)
        assert record is not None
        record.started_at = time.time() - 1000

        assert fin.is_timed_out(record_id) is True
        assert await fin.check_complete("run_1") is False

    @pytest.mark.asyncio
    async def test_disposition_rejected_after_complete(self, finalization: Finalization) -> None:
        effects = _make_effects(1)
        record_id = await finalization.start_finalization("run_1", effects)
        await finalization.record_disposition(record_id, "eff_0", DispositionType.COMMITTED)

        # Record is now COMPLETE, further dispositions rejected
        ok = await finalization.record_disposition(
            record_id, "eff_extra", DispositionType.ABANDONED
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_get_records_for_run(self, finalization: Finalization) -> None:
        await finalization.start_finalization("run_1", _make_effects(1))
        await finalization.start_finalization("run_2", _make_effects(2))

        records = finalization.get_records_for_run("run_1")
        assert len(records) == 1
        assert records[0].run_id == "run_1"

    def test_effect_disposition_serialization(self) -> None:
        disp = EffectDisposition(
            effect_id="eff_1",
            disposition=DispositionType.COMPENSATED,
            evidence_hash="abc123",
            reason="rollback",
        )
        d = disp.to_dict()
        restored = EffectDisposition.from_dict(d)
        assert restored.effect_id == "eff_1"
        assert restored.disposition == DispositionType.COMPENSATED
        assert restored.evidence_hash == "abc123"

    def test_finalization_record_serialization(self) -> None:
        record = FinalizationRecord(
            record_id="fin_test",
            run_id="run_1",
            total_effects=3,
        )
        d = record.to_dict()
        assert d["record_id"] == "fin_test"
        assert d["run_id"] == "run_1"
        assert d["total_effects"] == 3
        assert d["state"] == "in_progress"

    @pytest.mark.asyncio
    async def test_finalization_no_journal(self) -> None:
        """Finalization works without a journal (journal is optional)."""
        fin = Finalization(journal=None)
        effects = _make_effects(1)
        record_id = await fin.start_finalization("run_1", effects)
        assert record_id.startswith("fin_")
        await fin.record_disposition(record_id, "eff_0", DispositionType.COMMITTED)
        assert await fin.check_complete("run_1") is True
