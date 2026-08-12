from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime


class _Projection:
    def __init__(self) -> None:
        self.state = SimpleNamespace(by_acceptance_id={})

    def rebuild_projection(self) -> None:
        return None


class _DispatchFailure:
    """Narrow dispatch port that preserves the three durable crash boundaries."""

    employee_runtime = None

    def __init__(self, fail_stage: str) -> None:
        self.fail_stage = fail_stage
        self.fail_once = True
        self.calls: list[str] = []
        self.recover_calls = 0
        self.reconcile_calls = 0
        self.acp_calls = 0
        self.attempt_committed = False
        self.attempt_terminal = False
        self.outbox_queued = False
        self.outbox_running = False
        self.outbox_terminal = False
        self.raise_recover = False
        self._prepared = object()

    @property
    def recovery_converged(self) -> bool:
        return self.attempt_terminal and self.outbox_terminal

    def prepare_next(self) -> object | None:
        self.calls.append("prepare")
        if self.attempt_committed:
            return None
        # The real coordinator anchors the attempt before publishing QUEUED.
        self.attempt_committed = True
        self.calls.append("attempt:committed")
        self._append_snapshot("queued")
        return self._prepared

    def execute_prepared(self, prepared: object) -> object:
        assert prepared is self._prepared
        # RUNNING must be durable before the external ACP effect starts.
        self._append_snapshot("running")
        self.acp_calls += 1
        self.calls.append("acp")
        # The execution terminal is durable before its Outbox projection.
        self.attempt_terminal = True
        self.calls.append("attempt:terminal")
        self._append_snapshot("terminal")
        return object()

    def recover_incomplete_attempts(self) -> tuple[object, ...]:
        self.recover_calls += 1
        self.calls.append("recover")
        if self.raise_recover:
            raise RuntimeError("attempt recovery unavailable")
        if not self.attempt_committed or self.attempt_terminal:
            return ()
        # Recovery terminalizes the uncertain dispatch without rerunning ACP.
        self.attempt_terminal = True
        self.calls.append("attempt:terminal:recovered")
        self._append_recovered_terminal()
        return (object(),)

    def reconcile_terminal_snapshots(self) -> int:
        self.reconcile_calls += 1
        self.calls.append("reconcile")
        if not self.attempt_terminal:
            return 0
        self._append_recovered_terminal()
        return 1

    def _append_recovered_terminal(self) -> None:
        if self.outbox_terminal:
            return
        if not self.outbox_queued:
            self._append_snapshot("queued")
        self._append_snapshot("terminal")

    def _append_snapshot(self, stage: str) -> None:
        self.calls.append(f"outbox:{stage}")
        if self.fail_once and self.fail_stage == stage:
            self.fail_once = False
            raise RuntimeError(f"{stage} Outbox append unavailable")
        if stage == "queued":
            self.outbox_queued = True
        elif stage == "running":
            self.outbox_running = True
        elif stage == "terminal":
            self.outbox_terminal = True
        else:  # pragma: no cover - fixture misuse guard
            raise AssertionError(f"unknown Outbox stage: {stage}")


class _Ingress(_Projection):
    def __init__(self, dispatch: _DispatchFailure) -> None:
        super().__init__()
        self._dispatch = dispatch
        self.gc_observations: list[bool] = []

    def gc_terminal_payloads(self) -> int:
        self.gc_observations.append(self._dispatch.recovery_converged)
        return 0


class _Outbox:
    def __init__(self, dispatch: _DispatchFailure) -> None:
        self._dispatch = dispatch
        self.gc_observations: list[bool] = []

    def gc_superseded_snapshots(self) -> int:
        self.gc_observations.append(self._dispatch.recovery_converged)
        return 0


def _runtime_for(
    dispatch: _DispatchFailure,
) -> tuple[EmployeeDepartmentRuntime, _Ingress, _Outbox, list[bool]]:
    runtime = EmployeeDepartmentRuntime()
    ingress = _Ingress(dispatch)
    outbox = _Outbox(dispatch)
    terminal_dispositions: list[bool] = []
    runtime._ingress = ingress  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = _Projection()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = dispatch  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox = outbox  # type: ignore[assignment]  # noqa: SLF001

    def reconcile_terminal_ingress() -> int:
        terminal_dispositions.append(dispatch.recovery_converged)
        return int(dispatch.recovery_converged)

    runtime._reconcile_terminal_ingress = reconcile_terminal_ingress  # type: ignore[method-assign]  # noqa: SLF001
    return runtime, ingress, outbox, terminal_dispositions


@pytest.mark.parametrize(
    ("fail_stage", "expected_acp_calls"),
    [
        pytest.param("queued", 0, id="queued-append-after-attempt-anchor"),
        pytest.param("running", 0, id="running-append-before-acp"),
        pytest.param("terminal", 1, id="terminal-append-after-terminal-anchor"),
    ],
)
def test_next_dispatch_tick_recovers_outbox_append_failure_without_reexecuting_acp(
    fail_stage: str,
    expected_acp_calls: int,
) -> None:
    dispatch = _DispatchFailure(fail_stage)
    runtime, ingress, outbox, terminal_dispositions = _runtime_for(dispatch)

    with pytest.raises(RuntimeError, match=rf"{fail_stage} Outbox append unavailable"):
        runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert dispatch.acp_calls == expected_acp_calls
    assert terminal_dispositions == []
    assert ingress.gc_observations == []
    assert outbox.gc_observations == []
    recover_calls = dispatch.recover_calls
    reconcile_calls = dispatch.reconcile_calls
    dispatch.calls.clear()

    runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert dispatch.recover_calls == recover_calls + 1
    assert dispatch.reconcile_calls == reconcile_calls + 1
    assert dispatch.calls.index("recover") < dispatch.calls.index("reconcile")
    assert dispatch.recovery_converged is True
    assert dispatch.acp_calls == expected_acp_calls
    assert terminal_dispositions == [True]
    assert ingress.gc_observations == [True]
    assert outbox.gc_observations == [True]


def test_reconcile_is_attempted_after_recover_raises_and_terminal_cleanup_stays_fenced(
) -> None:
    dispatch = _DispatchFailure("queued")
    runtime, ingress, outbox, terminal_dispositions = _runtime_for(dispatch)

    with pytest.raises(RuntimeError, match="queued Outbox append unavailable"):
        runtime._drain_employee_dispatch_once()  # noqa: SLF001

    dispatch.raise_recover = True
    dispatch.calls.clear()
    recover_calls = dispatch.recover_calls
    reconcile_calls = dispatch.reconcile_calls

    with pytest.raises(
        RuntimeError,
        match="employee dispatch reporting recovery failed",
    ) as exc_info:
        runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "attempt recovery unavailable"

    assert dispatch.recover_calls > recover_calls
    assert dispatch.reconcile_calls == reconcile_calls + 1
    assert dispatch.calls.index("recover") < dispatch.calls.index("reconcile")
    assert dispatch.recovery_converged is False
    # The programming/contract error remains fatal, but the independent
    # terminal-ingress pass still runs and observes an unproven (fenced)
    # dispatch state instead of silently reopening or garbage-collecting it.
    assert terminal_dispositions == [False]
    assert ingress.gc_observations == []
    assert outbox.gc_observations == []
    assert runtime._dispatch_recovery_pending is True  # noqa: SLF001


def test_real_outbox_lifecycle_blob_failure_does_not_block_later_attempt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.autonomous.gateway.models import (
        EmployeeDispatchReportingDeferredError,
    )
    from src.autonomous.journal.blob_store import (
        AesGcmEncryptionProvider,
        BlobPublishError,
        BlobStore,
    )
    from src.autonomous.outbox.lifecycle import EmployeeOutboxLifecycle
    from src.autonomous.outbox.models import employee_outbox_id
    from src.autonomous.outbox.projection import OutboxProjectionState
    from src.autonomous.outbox.service import EmployeeOutboxService
    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path, second_candidate=True)
    first = harness.coordinator.prepare_next()
    second = harness.coordinator.prepare_next()
    assert first is not None and second is not None
    outbox = EmployeeOutboxService(
        writer=harness.writer,
        blob_store=BlobStore(
            tmp_path / "reporting-outbox-blobs",
            AesGcmEncryptionProvider(lambda _key_ref: b"o" * 32),
        ),
        outbox_state=OutboxProjectionState(),
        active_key_id="reporting-key",
    )
    lifecycle = EmployeeOutboxLifecycle(outbox)
    harness.coordinator._attempt_lifecycle = lifecycle  # noqa: SLF001
    stage_and_publish = outbox.blob_store.stage_and_publish

    def publish_with_one_poison(payload, labels, key_ref):
        if labels["attempt_id"] == first.binding.attempt_id:
            raise BlobPublishError("temporary blob publication failure")
        return stage_and_publish(payload, labels, key_ref)

    monkeypatch.setattr(outbox.blob_store, "stage_and_publish", publish_with_one_poison)
    try:
        with pytest.raises(EmployeeDispatchReportingDeferredError) as raised:
            harness.coordinator.recover_incomplete_attempts()

        assert raised.value.failed_attempt_ids == (first.binding.attempt_id,)
        assert raised.value.repaired_count == 1
        assert outbox.get_snapshot(
            employee_outbox_id(
                second.binding.tenant_key,
                second.binding.agent_id,
                second.binding.attempt_id,
            )
        ).state.terminal
        assert all(
            harness.coordinator.state.attempts[attempt_id].terminal_status
            == "action_required"
            for attempt_id in (
                first.binding.attempt_id,
                second.binding.attempt_id,
            )
        )
    finally:
        outbox.close()
        harness.close()


@pytest.mark.parametrize("missing", [False, True], ids=["read-error", "missing"])
def test_real_outbox_lifecycle_read_failure_does_not_block_later_attempt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    missing: bool,
) -> None:
    from src.autonomous.gateway.models import (
        EmployeeDispatchReportingDeferredError,
    )
    from src.autonomous.journal.blob_store import (
        AesGcmEncryptionProvider,
        BlobMissingError,
        BlobReadError,
        BlobStore,
    )
    from src.autonomous.outbox.lifecycle import EmployeeOutboxLifecycle
    from src.autonomous.outbox.models import employee_outbox_id
    from src.autonomous.outbox.projection import OutboxProjectionState
    from src.autonomous.outbox.service import EmployeeOutboxService
    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path, second_candidate=True)
    first = harness.coordinator.prepare_next()
    second = harness.coordinator.prepare_next()
    assert first is not None and second is not None
    assert len(harness.coordinator.recover_incomplete_attempts()) == 2
    outbox = EmployeeOutboxService(
        writer=harness.writer,
        blob_store=BlobStore(
            tmp_path / "reporting-read-outbox-blobs",
            AesGcmEncryptionProvider(lambda _key_ref: b"r" * 32),
        ),
        outbox_state=OutboxProjectionState(),
        active_key_id="reporting-read-key",
    )
    lifecycle = EmployeeOutboxLifecycle(outbox)
    harness.coordinator._attempt_lifecycle = lifecycle  # noqa: SLF001
    assert harness.coordinator.reconcile_terminal_snapshots() == 2
    read_blob = outbox.blob_store.read

    def read_with_one_poison(blob_ref):
        if blob_ref.labels["attempt_id"] == first.binding.attempt_id:
            error_type = BlobMissingError if missing else BlobReadError
            raise error_type("record-local Outbox read failed")
        return read_blob(blob_ref)

    monkeypatch.setattr(outbox.blob_store, "read", read_with_one_poison)
    try:
        with pytest.raises(EmployeeDispatchReportingDeferredError) as raised:
            harness.coordinator.reconcile_terminal_snapshots()

        assert raised.value.failed_attempt_ids == (first.binding.attempt_id,)
        assert raised.value.repaired_count == 1
        assert outbox.get_snapshot(
            employee_outbox_id(
                second.binding.tenant_key,
                second.binding.agent_id,
                second.binding.attempt_id,
            )
        ).state.terminal
    finally:
        outbox.close()
        harness.close()


def test_real_outbox_lifecycle_integrity_failure_remains_fatal(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.autonomous.journal.blob_store import (
        AesGcmEncryptionProvider,
        BlobIntegrityError,
        BlobStore,
    )
    from src.autonomous.outbox.lifecycle import EmployeeOutboxLifecycle
    from src.autonomous.outbox.projection import OutboxProjectionState
    from src.autonomous.outbox.service import (
        EmployeeOutboxService,
        OutboxBlobError,
    )
    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path, second_candidate=True)
    first = harness.coordinator.prepare_next()
    second = harness.coordinator.prepare_next()
    assert first is not None and second is not None
    outbox = EmployeeOutboxService(
        writer=harness.writer,
        blob_store=BlobStore(
            tmp_path / "reporting-integrity-outbox-blobs",
            AesGcmEncryptionProvider(lambda _key_ref: b"i" * 32),
        ),
        outbox_state=OutboxProjectionState(),
        active_key_id="reporting-integrity-key",
    )
    lifecycle = EmployeeOutboxLifecycle(outbox)
    harness.coordinator._attempt_lifecycle = lifecycle  # noqa: SLF001
    read_blob = outbox.blob_store.read

    def read_with_integrity_failure(blob_ref):
        if blob_ref.labels["attempt_id"] == first.binding.attempt_id:
            raise BlobIntegrityError("Outbox payload verification failed")
        return read_blob(blob_ref)

    monkeypatch.setattr(outbox.blob_store, "read", read_with_integrity_failure)
    try:
        with pytest.raises(OutboxBlobError, match="publication failed") as raised:
            harness.coordinator.recover_incomplete_attempts()

        assert isinstance(raised.value.__cause__, BlobIntegrityError)
        assert not harness.coordinator.state.attempts[
            second.binding.attempt_id
        ].terminal_status
    finally:
        outbox.close()
        harness.close()
