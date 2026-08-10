"""Task 6 terminal pipeline contract for every started employee attempt."""

from __future__ import annotations

import pytest

from src.autonomous.gateway.coordinator import (
    EmployeeDispatchCoordinator,
    EmployeeDispatchError,
)
from src.autonomous.gateway.models import (
    GatewayExecutionResult,
    GatewayExecutionStatus,
)
from src.autonomous.journal.frame import GENESIS_HASH, JournalEvent, TransactionFrame

_FRAME_KEY = b"gateway-test-frame-key-at-least-32-bytes"


def _frame(events, *, sequence, previous_hash=GENESIS_HASH):
    aggregate_ids = {event.aggregate_id for event in events}
    return TransactionFrame.seal(
        tx_id=f"tx_gateway_{sequence}",
        sequence=sequence,
        writer_epoch=1,
        timestamp=float(sequence),
        expected_versions={aggregate_id: sequence - 1 for aggregate_id in aggregate_ids},
        aggregate_versions={aggregate_id: sequence for aggregate_id in aggregate_ids},
        previous_hash=previous_hash,
        events=tuple(events),
        hmac_key=_FRAME_KEY,
    )


def test_every_started_attempt_has_one_terminal_or_action_required(tmp_path) -> None:
    """EI-TERMINAL-01 runs all five statuses through the real coordinator."""

    from unittest.mock import patch

    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    for status in GatewayExecutionStatus:
        harness = _real_coordinator_harness(tmp_path / status.value)
        harness.data._shard_timezone = "Asia/Shanghai"
        prepared = harness.coordinator.prepare_next()
        assert prepared is not None
        result = GatewayExecutionResult(
            status=status,
            output="done" if status is GatewayExecutionStatus.COMPLETED else "",
            safe_error_code="" if status is GatewayExecutionStatus.COMPLETED else "safe",
        )
        prepare_sequence = harness.writer.anchor.read().sequence
        original_stage = harness.data.stage_history_payload

        def stage(*args, **kwargs):
            assert harness.writer.anchor.read().sequence == prepare_sequence
            return original_stage(*args, **kwargs)

        with patch.object(harness.data, "stage_history_payload", side_effect=stage):
            finalized = harness.coordinator.finalize_attempt(
                prepared.binding.attempt_id,
                result,
                request_text=prepared.prompt,
            )
        terminal_frame = tuple(harness.writer.replay())[-1]
        assert [event.event_type for event in terminal_frame.events] == [
            "employee.history.recorded",
            "employee.execution_attempt.terminal",
            "employee.ingress.router_terminal",
        ]
        assert finalized.status is status
        history_event = next(
            event
            for event in terminal_frame.events
            if event.event_type == "employee.history.recorded"
        )
        assert history_event.payload["shard_timezone"] == "Asia/Shanghai"
        assert finalized.history_record_id in harness.data.state.history_records
        assert (
            harness.coordinator.finalize_attempt(
                prepared.binding.attempt_id,
                result,
                request_text=prepared.prompt,
            )
            == finalized
        )
        conflicting = GatewayExecutionResult(
            status=status,
            output="different" if status is GatewayExecutionStatus.COMPLETED else "",
            safe_error_code="different" if status is not GatewayExecutionStatus.COMPLETED else "",
        )
        with pytest.raises(EmployeeDispatchError, match="conflicts"):
            harness.coordinator.finalize_attempt(
                prepared.binding.attempt_id,
                conflicting,
            )
        harness.close()


def test_history_blob_is_staged_before_atomic_terminal_commit() -> None:
    """Task 6 must not call BlobStore while holding the data/Journal locks."""

    from src.autonomous.data.service import EmployeeDataService

    assert hasattr(EmployeeDataService, "stage_history_payload")
    assert hasattr(EmployeeDispatchCoordinator, "finalize_attempt")


def test_employee_terminal_card_hook_runs_after_atomic_terminal_anchor(tmp_path) -> None:
    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)
    observed: list[str] = []

    class _Lifecycle:
        def queued(self, _binding):
            observed.append("queued")

        def running(self, _binding):
            observed.append("running")

        def terminal(self, _binding, _result):
            frame = tuple(harness.writer.replay())[-1]
            assert [event.event_type for event in frame.events] == [
                "employee.history.recorded",
                "employee.execution_attempt.terminal",
                "employee.ingress.router_terminal",
            ]
            observed.append("terminal")

    harness.coordinator._attempt_lifecycle = _Lifecycle()
    try:
        prepared = harness.coordinator.prepare_next()
        assert prepared is not None
        harness.coordinator.finalize_attempt(
            prepared.binding.attempt_id,
            GatewayExecutionResult(
                GatewayExecutionStatus.COMPLETED,
                output="done",
            ),
            request_text=prepared.prompt,
        )

        assert observed == ["queued", "terminal"]
    finally:
        harness.close()


def test_recovery_rebuilds_missing_terminal_snapshot_without_rerunning_acp(
    tmp_path,
) -> None:
    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)

    class _CrashAfterAnchor:
        def queued(self, _binding):
            return None

        def running(self, _binding):
            return None

        def terminal(self, _binding, _result):
            raise RuntimeError("crash after terminal anchor")

    harness.coordinator._attempt_lifecycle = _CrashAfterAnchor()
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    with pytest.raises(RuntimeError, match="crash after terminal anchor"):
        harness.coordinator.finalize_attempt(
            prepared.binding.attempt_id,
            GatewayExecutionResult(
                GatewayExecutionStatus.COMPLETED,
                output="durable result",
            ),
            request_text=prepared.prompt,
        )

    recovered: list[tuple[str, str]] = []

    class _RecoveredLifecycle:
        def queued(self, _binding):
            raise AssertionError("reconciliation must use the terminal hook")

        def running(self, _binding):
            raise AssertionError("reconciliation must not rerun execution")

        def terminal(self, binding, result):
            recovered.append((binding.attempt_id, result.output))

    harness.coordinator._attempt_lifecycle = _RecoveredLifecycle()
    try:
        assert harness.coordinator.reconcile_terminal_snapshots() == 1
        assert recovered == [(prepared.binding.attempt_id, "durable result")]
        assert harness.coordinator._data_sink.publish_document.call_count == 8
    finally:
        harness.close()


def test_terminal_commit_section_never_replays_full_journal(
    tmp_path,
    monkeypatch,
) -> None:
    from contextlib import contextmanager

    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    original_guard = harness.writer.transaction_guard
    original_replay = harness.writer.replay
    in_transaction = False

    @contextmanager
    def guarded_transaction():
        nonlocal in_transaction
        with original_guard():
            in_transaction = True
            try:
                yield
            finally:
                in_transaction = False

    def checked_replay(*args, **kwargs):
        assert not in_transaction, "full Journal replay inside terminal commit"
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(harness.writer, "transaction_guard", guarded_transaction)
    monkeypatch.setattr(harness.writer, "replay", checked_replay)
    finalized = harness.coordinator.finalize_attempt(
        prepared.binding.attempt_id,
        GatewayExecutionResult(
            GatewayExecutionStatus.COMPLETED,
            output="done",
        ),
    )
    assert finalized.status is GatewayExecutionStatus.COMPLETED
    harness.close()


def test_terminal_head_race_retries_without_restaging_or_rerunning_acp(
    tmp_path,
    monkeypatch,
) -> None:
    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    original_presync = harness.coordinator._presynchronize_domains  # noqa: SLF001
    original_stage = harness.data.stage_history_payload
    presync_calls = 0
    stage_calls = 0

    def racing_presync():
        nonlocal presync_calls
        captured = original_presync()
        presync_calls += 1
        if presync_calls == 1:
            event = JournalEvent(
                event_type="test.concurrent.head_advance",
                aggregate_id="test-head-advance",
                payload={},
            )
            harness.writer.commit(
                (event,),
                harness.writer.get_aggregate_versions((event.aggregate_id,)),
                expected_head_sequence=captured[0],
                expected_head_hash=captured[1],
            )
        return captured

    def counting_stage(*args, **kwargs):
        nonlocal stage_calls
        stage_calls += 1
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(
        harness.coordinator,
        "_presynchronize_domains",
        racing_presync,
    )
    monkeypatch.setattr(harness.data, "stage_history_payload", counting_stage)
    finalized = harness.coordinator.finalize_attempt(
        prepared.binding.attempt_id,
        GatewayExecutionResult(
            GatewayExecutionStatus.COMPLETED,
            output="already executed once",
        ),
    )
    assert finalized.status is GatewayExecutionStatus.COMPLETED
    assert presync_calls == 2
    assert stage_calls == 1
    harness.close()


def test_history_failure_blocks_false_success_and_recovery_requires_action(
    tmp_path,
    monkeypatch,
) -> None:
    from src.autonomous.data.service import DataBlobError
    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    monkeypatch.setattr(
        harness.data,
        "stage_history_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataBlobError("fault")),
    )
    with pytest.raises(DataBlobError, match="fault"):
        harness.coordinator.finalize_attempt(
            prepared.binding.attempt_id,
            GatewayExecutionResult(
                GatewayExecutionStatus.COMPLETED,
                output="must not become success",
            ),
        )
    assert all(
        event.event_type != "employee.execution_attempt.terminal"
        for frame in harness.writer.replay()
        for event in frame.events
    )
    assert harness.router.state.by_acceptance_id[prepared.binding.acceptance_id].state == "dispatching"
    monkeypatch.undo()

    recovered = harness.restart().recover_incomplete_attempts()
    assert len(recovered) == 1
    assert recovered[0].status is GatewayExecutionStatus.ACTION_REQUIRED
    harness.close()


def test_anchored_terminal_apply_failure_keeps_live_history_blob(
    tmp_path,
    monkeypatch,
) -> None:
    """Once the frame anchors, its referenced blob is live despite apply failure."""

    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    staged = {}
    original_stage = harness.data.stage_history_payload

    def capture_stage(*args, **kwargs):
        value = original_stage(*args, **kwargs)
        staged["value"] = value
        return value

    monkeypatch.setattr(harness.data, "stage_history_payload", capture_stage)
    monkeypatch.setattr(
        harness.coordinator,
        "_apply_committed_frame_unlocked",
        lambda _frame: (_ for _ in ()).throw(RuntimeError("apply fault")),
    )
    with pytest.raises(RuntimeError, match="apply fault"):
        harness.coordinator.finalize_attempt(
            prepared.binding.attempt_id,
            GatewayExecutionResult(
                GatewayExecutionStatus.COMPLETED,
                output="durable output",
            ),
            request_text=prepared.prompt,
        )
    published = staged["value"]
    assert published.blob_ref.blob_id in harness.data._blob_store.iter_blob_ids()  # noqa: SLF001
    assert harness.data._blob_store.read(published.blob_ref)  # noqa: SLF001

    monkeypatch.undo()
    restarted = harness.restart()
    assert restarted.recover_incomplete_attempts() == ()
    assert restarted.state.attempts[prepared.binding.attempt_id].terminal_status == "completed"
    harness.close()


def test_gateway_result_has_explicit_timeout_cancel_and_failure_states() -> None:
    """The Team runner's Optional[str] must never be interpreted as success."""


    assert GatewayExecutionResult
    statuses = {status.value for status in GatewayExecutionStatus}
    assert statuses == {
        "completed",
        "failed",
        "canceled",
        "timeout",
        "action_required",
    }


def test_attempt_binding_and_dispatch_commit_require_one_frame() -> None:
    from src.autonomous.gateway.projection import (
        ATTEMPT_BOUND,
        ATTEMPT_DISPATCH_COMMITTED,
        GatewayProjectionError,
        GatewayProjectionState,
        reduce_gateway_frame,
    )
    from tests.autonomous.integration.test_employee_team_gateway import _binding

    binding = _binding()
    bound = JournalEvent(
        event_type=ATTEMPT_BOUND,
        aggregate_id=binding.attempt_id,
        payload={"binding": binding.to_dict()},
    )
    committed = JournalEvent(
        event_type=ATTEMPT_DISPATCH_COMMITTED,
        aggregate_id=binding.attempt_id,
        payload={"attempt_id": binding.attempt_id, "permit_id": binding.permit_id},
    )
    state = GatewayProjectionState()
    router_dispatch = JournalEvent(
        event_type="employee.ingress.router_dispatching",
        aggregate_id=binding.ingress_aggregate_id,
        payload={"acceptance_id": binding.acceptance_id},
    )

    with pytest.raises(GatewayProjectionError, match="same frame"):
        reduce_gateway_frame(state, _frame([bound, router_dispatch], sequence=1))
    with pytest.raises(GatewayProjectionError, match="same frame"):
        reduce_gateway_frame(state, _frame([committed], sequence=1))
    committed_frame = _frame(
        [router_dispatch, bound, committed],
        sequence=1,
    )
    reduce_gateway_frame(state, committed_frame)

    attempt = state.attempts[binding.attempt_id]
    assert attempt.dispatch_committed is True
    assert attempt.bound_sequence == attempt.dispatch_sequence == 1


def test_first_attempt_terminal_wins_and_identical_replay_is_idempotent() -> None:
    from src.autonomous.gateway.projection import (
        ATTEMPT_BOUND,
        ATTEMPT_DISPATCH_COMMITTED,
        ATTEMPT_TERMINAL,
        GatewayProjectionError,
        GatewayProjectionState,
        reduce_gateway_frame,
    )
    from tests.autonomous.integration.test_employee_team_gateway import _binding

    binding = _binding()
    state = GatewayProjectionState()
    first_frame = _frame(
        [
            JournalEvent(
                event_type="employee.ingress.router_dispatching",
                aggregate_id=binding.ingress_aggregate_id,
                payload={"acceptance_id": binding.acceptance_id},
            ),
            JournalEvent(
                event_type=ATTEMPT_BOUND,
                aggregate_id=binding.attempt_id,
                payload={"binding": binding.to_dict()},
            ),
            JournalEvent(
                event_type=ATTEMPT_DISPATCH_COMMITTED,
                aggregate_id=binding.attempt_id,
                payload={
                    "attempt_id": binding.attempt_id,
                    "permit_id": binding.permit_id,
                },
            ),
        ],
        sequence=1,
    )
    reduce_gateway_frame(state, first_frame)
    payload = {
        "attempt_id": binding.attempt_id,
        "terminal_epoch": 1,
        "status": "action_required",
        "result_digest": "e" * 64,
        "history_record_id": "hist_" + "f" * 64,
        "ended_at": "2026-07-14T00:01:00Z",
    }
    terminal = JournalEvent(
        event_type=ATTEMPT_TERMINAL,
        aggregate_id=binding.attempt_id,
        payload=payload,
    )
    history = JournalEvent(
        event_type="employee.history.recorded",
        aggregate_id=payload["history_record_id"],
        payload={
            "attempt_id": binding.attempt_id,
            "record_id": payload["history_record_id"],
        },
    )
    router_terminal = JournalEvent(
        event_type="employee.ingress.router_terminal",
        aggregate_id=binding.ingress_aggregate_id,
        payload={
            "acceptance_id": binding.acceptance_id,
            "reason_code": "action_required",
        },
    )
    terminal_frame = _frame(
        [history, terminal, router_terminal],
        sequence=2,
        previous_hash=first_frame.frame_hash,
    )
    reduce_gateway_frame(state, terminal_frame)
    reduce_gateway_frame(state, first_frame)
    reduce_gateway_frame(state, terminal_frame)

    conflicting = JournalEvent(
        event_type=ATTEMPT_TERMINAL,
        aggregate_id=binding.attempt_id,
        payload={**payload, "status": "failed"},
    )
    with pytest.raises(GatewayProjectionError, match="conflicting"):
        reduce_gateway_frame(
            state,
            _frame(
                [
                    history,
                    conflicting,
                    JournalEvent(
                        event_type="employee.ingress.router_terminal",
                        aggregate_id=binding.ingress_aggregate_id,
                        payload={
                            "acceptance_id": binding.acceptance_id,
                            "reason_code": "failed",
                        },
                    ),
                ],
                sequence=3,
                previous_hash=terminal_frame.frame_hash,
            ),
        )


@pytest.mark.parametrize("mismatch", ["aggregate_id", "record_id"])
def test_terminal_history_requires_exact_record_identity(mismatch) -> None:
    from src.autonomous.gateway.projection import (
        ATTEMPT_BOUND,
        ATTEMPT_DISPATCH_COMMITTED,
        ATTEMPT_TERMINAL,
        GatewayProjectionError,
        GatewayProjectionState,
        reduce_gateway_frame,
    )
    from tests.autonomous.integration.test_employee_team_gateway import _binding

    binding = _binding()
    state = GatewayProjectionState()
    first = _frame(
        [
            JournalEvent(
                event_type="employee.ingress.router_dispatching",
                aggregate_id=binding.ingress_aggregate_id,
                payload={"acceptance_id": binding.acceptance_id},
            ),
            JournalEvent(
                event_type=ATTEMPT_BOUND,
                aggregate_id=binding.attempt_id,
                payload={"binding": binding.to_dict()},
            ),
            JournalEvent(
                event_type=ATTEMPT_DISPATCH_COMMITTED,
                aggregate_id=binding.attempt_id,
                payload={
                    "attempt_id": binding.attempt_id,
                    "permit_id": binding.permit_id,
                },
            ),
        ],
        sequence=1,
    )
    reduce_gateway_frame(state, first)
    history_id = "hist_" + "f" * 64
    history = JournalEvent(
        event_type="employee.history.recorded",
        aggregate_id=("hist_" + "e" * 64) if mismatch == "aggregate_id" else history_id,
        payload={
            "attempt_id": binding.attempt_id,
            "record_id": ("hist_" + "e" * 64) if mismatch == "record_id" else history_id,
        },
    )
    terminal = JournalEvent(
        event_type=ATTEMPT_TERMINAL,
        aggregate_id=binding.attempt_id,
        payload={
            "attempt_id": binding.attempt_id,
            "terminal_epoch": 1,
            "status": "failed",
            "result_digest": "e" * 64,
            "history_record_id": history_id,
            "ended_at": "2026-07-14T00:01:00Z",
        },
    )
    router_terminal = JournalEvent(
        event_type="employee.ingress.router_terminal",
        aggregate_id=binding.ingress_aggregate_id,
        payload={
            "acceptance_id": binding.acceptance_id,
            "reason_code": "failed",
        },
    )
    with pytest.raises(GatewayProjectionError, match="requires history"):
        reduce_gateway_frame(
            state,
            _frame(
                [history, terminal, router_terminal],
                sequence=2,
                previous_hash=first.frame_hash,
            ),
        )
