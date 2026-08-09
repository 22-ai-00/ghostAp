"""Chaos test: revocation race conditions between prepare and execute."""

from __future__ import annotations

import pytest

from src.autonomous.broker.dispatch_gate import (
    DispatchGate,
    DispatchGateClosed,
    StaleEpoch,
)
from src.autonomous.domain.effects import CapabilityDescriptor
from src.autonomous.domain.enums import RiskLevel


def _make_gate() -> tuple[DispatchGate, list]:
    frames: list[tuple[str, dict]] = []
    seq_counter = [0]

    def record(event_type: str, data: dict) -> int:
        seq_counter[0] += 1
        frames.append((event_type, data))
        return seq_counter[0]

    def anchor(seq: int, ref: str) -> bool:
        return True

    gate = DispatchGate(anchor_fn=anchor, record_frame_fn=record)
    return gate, frames


def _cap() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="cap_write",
        name="Write File",
        version="1.0.0",
        risk_level=RiskLevel.R3,
    )


@pytest.mark.asyncio
async def test_concurrent_revocation_and_execute() -> None:
    """Gate closes while execute is waiting for lock."""
    gate, frames = _make_gate()
    cap = _cap()
    gate.open_gate("run_1", plan_epoch=1)

    prep = gate.prepare(
        run_id="run_1",
        capability=cap,
        parameters={"data": "test"},
        attempt_id="att_1",
    )

    # First acquire the lock to simulate contention
    async with gate._lock:
        # Close gate while lock is held
        gate.close_gate("run_1", reason="concurrent_revocation")

    # Now try to execute — gate is closed
    async def adapter(params: dict) -> dict:
        return {"ok": True}

    with pytest.raises(DispatchGateClosed):
        await gate.execute(prep.effect.effect_instance_id, adapter, {})


@pytest.mark.asyncio
async def test_epoch_advancement_revokes_old_preparations() -> None:
    """New epoch invalidates all old preparations."""
    gate, frames = _make_gate()
    cap = _cap()

    epoch1 = gate.open_gate("run_1", plan_epoch=1)

    prep_old = gate.prepare(
        run_id="run_1",
        capability=cap,
        parameters={"epoch": 1},
        attempt_id="att_old",
    )

    # Epoch advance: close old, open new
    gate.close_gate("run_1", reason="replan")
    epoch2 = gate.open_gate("run_1", plan_epoch=2)

    assert epoch2.gate_epoch > epoch1.gate_epoch

    async def adapter(params: dict) -> dict:
        return {}

    # Old preparation rejected
    with pytest.raises(StaleEpoch):
        await gate.execute(prep_old.effect.effect_instance_id, adapter, {})

    # New preparation on new epoch succeeds
    prep_new = gate.prepare(
        run_id="run_1",
        capability=cap,
        parameters={"epoch": 2},
        attempt_id="att_new",
    )
    result = await gate.execute(prep_new.effect.effect_instance_id, adapter, {})
    assert result.success


@pytest.mark.asyncio
async def test_drain_count_tracks_pending_and_active() -> None:
    gate, frames = _make_gate()
    cap = _cap()
    gate.open_gate("run_1", plan_epoch=1)

    assert gate.drain_count("run_1") == 0

    gate.prepare(
        run_id="run_1",
        capability=cap,
        parameters={},
        attempt_id="att_1",
    )
    gate.prepare(
        run_id="run_1",
        capability=cap,
        parameters={},
        attempt_id="att_2",
    )

    assert gate.drain_count("run_1") == 2


@pytest.mark.asyncio
async def test_adapter_exception_records_unknown_state() -> None:
    gate, frames = _make_gate()
    cap = _cap()
    gate.open_gate("run_1", plan_epoch=1)

    prep = gate.prepare(
        run_id="run_1",
        capability=cap,
        parameters={},
        attempt_id="att_1",
    )

    async def failing_adapter(params: dict) -> dict:
        raise ConnectionError("network timeout")

    result = await gate.execute(prep.effect.effect_instance_id, failing_adapter, {})
    assert not result.success
    assert result.needs_reconciliation
    assert "network timeout" in result.error

    # Verify unknown frame was recorded
    unknown_frames = [(et, d) for et, d in frames if et == "effect.unknown"]
    assert len(unknown_frames) == 1
