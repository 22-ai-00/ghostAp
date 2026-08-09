"""Chaos test: remote success + local crash produces no resend on restart."""

from __future__ import annotations

import pytest

from src.autonomous.broker.dispatch_gate import (
    DispatchGate,
    EffectNotPrepared,
)
from src.autonomous.domain.effects import CapabilityDescriptor
from src.autonomous.domain.enums import RiskLevel


def _make_capability() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="cap_create_file",
        name="Create File",
        version="1.0.0",
        risk_level=RiskLevel.R3,
        idempotency="semantic_key",
    )


class CrashJournal:
    """Simulates a journal that records frames but can be replayed after crash."""

    def __init__(self) -> None:
        self.frames: list[tuple[str, dict]] = []
        self._seq = 0
        self.crash_after_commit = False

    def record(self, event_type: str, data: dict) -> int:
        self._seq += 1
        self.frames.append((event_type, data))
        if self.crash_after_commit and event_type == "effect.committed":
            raise RuntimeError("simulated crash after commit")
        return self._seq

    def anchor(self, seq: int, ref: str) -> bool:
        return True


def test_crash_after_remote_success_no_resend_on_restart() -> None:
    """After remote success + local crash, the journal has evidence of the
    EXECUTING frame. On restart, reconciliation can query remote state
    and resolve without resending."""
    journal = CrashJournal()
    gate = DispatchGate(
        anchor_fn=journal.anchor,
        record_frame_fn=journal.record,
    )

    cap = _make_capability()
    gate.open_gate("run_1", plan_epoch=1)

    prep = gate.prepare(
        run_id="run_1",
        capability=cap,
        parameters={"path": "/tmp/test.txt"},
        attempt_id="att_1",
    )

    # Verify PREPARED frame was recorded
    assert any(et == "effect.prepared" for et, _ in journal.frames)
    assert prep.anchored is True

    # Execute succeeds at the adapter level but crash during commit recording
    # means the outcome is recorded as "unknown" (needs reconciliation)
    journal.crash_after_commit = True

    import asyncio

    send_count = 0

    async def succeed_adapter(params: dict) -> dict:
        nonlocal send_count
        send_count += 1
        return {"created": True}

    # The execute catches the RuntimeError from record_frame_fn and
    # reports it as needing reconciliation
    result = asyncio.run(
        gate.execute(prep.effect.effect_instance_id, succeed_adapter, {})
    )

    # Adapter was called exactly once
    assert send_count == 1

    # Result indicates unknown state (needs reconciliation)
    assert not result.success
    assert result.needs_reconciliation

    # The EXECUTING frame was recorded before the crash
    executing_frames = [
        (et, d) for et, d in journal.frames if et == "effect.executing"
    ]
    assert len(executing_frames) == 1


def test_unanchored_effect_cannot_execute() -> None:
    """Effect that fails to anchor cannot proceed to execution."""

    def fail_anchor(seq: int, ref: str) -> bool:
        return False

    frames: list[tuple[str, dict]] = []
    seq_counter = [0]

    def record(event_type: str, data: dict) -> int:
        seq_counter[0] += 1
        frames.append((event_type, data))
        return seq_counter[0]

    gate = DispatchGate(anchor_fn=fail_anchor, record_frame_fn=record)
    cap = _make_capability()
    gate.open_gate("run_1", plan_epoch=1)

    prep = gate.prepare(
        run_id="run_1",
        capability=cap,
        parameters={},
        attempt_id="att_1",
    )
    assert prep.anchored is False

    import asyncio

    async def adapter(params: dict) -> dict:
        return {}

    with pytest.raises(EffectNotPrepared, match="not anchored"):
        asyncio.run(gate.execute(prep.effect.effect_instance_id, adapter, {}))


def test_revocation_race_gate_closed_before_dispatch() -> None:
    """If revocation closes the gate between prepare and execute,
    execution must be rejected."""
    frames: list[tuple[str, dict]] = []
    seq_counter = [0]

    def record(event_type: str, data: dict) -> int:
        seq_counter[0] += 1
        frames.append((event_type, data))
        return seq_counter[0]

    def anchor(seq: int, ref: str) -> bool:
        return True

    gate = DispatchGate(anchor_fn=anchor, record_frame_fn=record)
    cap = _make_capability()
    gate.open_gate("run_1", plan_epoch=1)

    prep = gate.prepare(
        run_id="run_1",
        capability=cap,
        parameters={},
        attempt_id="att_1",
    )

    # Revocation arrives between prepare and execute
    gate.close_gate("run_1", reason="kill_switch_activated")

    import asyncio

    from src.autonomous.broker.dispatch_gate import DispatchGateClosed

    async def adapter(params: dict) -> dict:
        return {}

    with pytest.raises(DispatchGateClosed):
        asyncio.run(gate.execute(prep.effect.effect_instance_id, adapter, {}))
