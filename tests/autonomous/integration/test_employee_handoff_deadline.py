from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.autonomous.ingress.router import (
    DurableEmployeeIngressRouter,
    RouterProjectionState,
    RouterWriteDisabledError,
)
from src.autonomous.ingress.service import (
    IngressWriteDisabledError,
    MessageAcceptanceOutcome,
)
from src.autonomous.provisioning import composition as employee_composition
from src.autonomous.provisioning.composition import (
    EmployeeDepartmentRuntime,
    EmployeeMessageHandoffUnknownError,
)
from tests.autonomous.chaos.test_employee_ingress_recovery import (
    _bound_message,
    _service,
)
from tests.autonomous.integration.test_employee_ingress_payload_retry import _stack
from tests.autonomous.integration.test_employee_targeted_handoff_terminal import (
    _CHANNEL_GENERATION,
    _accepted_targeted_ingress,
    _runtime_for_handoff,
    _terminal_record,
)
from tests.autonomous.unit.test_journal_writer import BlockingFsOps


def test_handoff_timeout_durably_denies_before_reporting_nonexecution(
    tmp_path: Path,
) -> None:
    service, writer, _store = _service(tmp_path)
    payload, metadata = _bound_message(
        raw_chat_id="oc_handoff_timeout",
        raw_message_id="om_handoff_timeout",
    )
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = service
    runtime._router = SimpleNamespace(
        record_snapshot=lambda _acceptance_id: None,
    )
    runtime._closing = False

    try:
        try:
            result = runtime.wait_for_employee_message_handoff(
                tenant_key=metadata.tenant_key,
                agent_id=metadata.agent_id,
                bot_principal_id=metadata.bot_principal_id,
                app_id=metadata.app_id,
                channel_generation=metadata.channel_generation,
                connection_id=metadata.connection_id,
                chat_id="oc_handoff_timeout",
                message_id="om_handoff_timeout",
                timeout=0.25,
            )
        except TypeError as exc:
            pytest.fail(f"composition is not wired to the exact ingress API: {exc}")

        assert result is False

        late = service.accept(metadata, payload, request_id="req_late_handoff")
        record = service.record_snapshot(late.acceptance.acceptance_id)
        assert record is not None
        assert record.disposition is not None
        assert record.disposition.reason_code == "handoff_unconfirmed"
    finally:
        service.close()
        writer.close()


def test_public_handoff_deadline_detaches_durable_denial_fsync(
    tmp_path: Path,
) -> None:
    fs_ops = BlockingFsOps()
    service, writer, _store = _service(tmp_path, fs_ops=fs_ops)
    _payload, metadata = _bound_message(
        raw_chat_id="oc_handoff_fsync_deadline",
        raw_message_id="om_handoff_fsync_deadline",
    )
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = service  # noqa: SLF001
    runtime._writer = writer  # noqa: SLF001
    runtime._router = SimpleNamespace(  # noqa: SLF001
        record_snapshot=lambda _acceptance_id: None,
    )
    outcome: list[BaseException | bool] = []

    def wait_for_handoff() -> None:
        try:
            outcome.append(
                runtime.wait_for_employee_message_handoff(
                    tenant_key=metadata.tenant_key,
                    agent_id=metadata.agent_id,
                    bot_principal_id=metadata.bot_principal_id,
                    app_id=metadata.app_id,
                    channel_generation=metadata.channel_generation,
                    connection_id=metadata.connection_id,
                    chat_id="oc_handoff_fsync_deadline",
                    message_id="om_handoff_fsync_deadline",
                    timeout=0.05,
                )
            )
        except BaseException as exc:
            outcome.append(exc)

    waiter = employee_composition.threading.Thread(target=wait_for_handoff)
    waiter.start()
    completed_before_fsync = False
    try:
        assert fs_ops.write_started.wait(1.0)
        waiter.join(timeout=0.15)
        completed_before_fsync = not waiter.is_alive()
    finally:
        fs_ops.allow_fsync.set()
        waiter.join(timeout=2.0)

    try:
        assert completed_before_fsync
        assert len(outcome) == 1
        assert isinstance(outcome[0], EmployeeMessageHandoffUnknownError)
        denied = service.wait_for_anchored_message_acceptance(
            tenant_key=metadata.tenant_key,
            agent_id=metadata.agent_id,
            bot_principal_id=metadata.bot_principal_id,
            app_id=metadata.app_id,
            event_type=metadata.event_type,
            chat_id=metadata.chat_id,
            message_id=metadata.message_id,
            channel_generation=metadata.channel_generation,
            connection_id=metadata.connection_id,
            timeout=1.0,
        )
        assert denied is not None
        assert denied.status == "denied"
    finally:
        runtime.close()


def test_public_handoff_deadline_detaches_durable_abandon_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(tmp_path)
    original_read = stack.ingress.blob_store.read
    monkeypatch.setattr(
        stack.ingress.blob_store,
        "read",
        MagicMock(side_effect=OSError("temporary payload read failure")),
    )
    assert stack.router.route(stack.acceptance_id).state == "accepted"
    monkeypatch.setattr(stack.ingress.blob_store, "read", original_read)

    fs_ops = BlockingFsOps()
    stack.writer._fs_ops = fs_ops  # noqa: SLF001 - block the abandon fsync
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = stack.ingress  # noqa: SLF001
    runtime._router = stack.router  # noqa: SLF001
    runtime._writer = stack.writer  # noqa: SLF001
    outcome: list[BaseException | bool] = []

    def wait_for_handoff() -> None:
        try:
            outcome.append(
                runtime.wait_for_employee_message_handoff(
                    tenant_key="tenant_1",
                    agent_id="agt_alpha",
                    bot_principal_id="bot_alpha",
                    app_id="cli_alpha",
                    channel_generation=3,
                    connection_id="conn_alpha",
                    chat_id="oc_team",
                    message_id="om_1",
                    timeout=0.05,
                )
            )
        except BaseException as exc:
            outcome.append(exc)

    waiter = employee_composition.threading.Thread(target=wait_for_handoff)
    waiter.start()
    completed_before_fsync = False
    try:
        assert fs_ops.write_started.wait(1.0)
        waiter.join(timeout=0.15)
        completed_before_fsync = not waiter.is_alive()
    finally:
        fs_ops.allow_fsync.set()
        waiter.join(timeout=2.0)

    try:
        assert completed_before_fsync
        assert len(outcome) == 1
        assert isinstance(outcome[0], EmployeeMessageHandoffUnknownError)
        assert runtime._shutdown_employee_handoff_finalization_executor(  # noqa: SLF001
            timeout=1.0,
        )
        abandoned = stack.router.record_snapshot(stack.acceptance_id)
        assert abandoned is not None
        assert abandoned.state == "terminal"
        assert abandoned.reason_code == "handoff_unconfirmed"
    finally:
        runtime.close()


def test_repeated_handoff_waits_share_one_pending_durable_finalizer() -> None:
    denial_started = employee_composition.threading.Event()
    release_denial = employee_composition.threading.Event()
    denial_calls = 0

    def deny_message_acceptance(**_kwargs):
        nonlocal denial_calls
        denial_calls += 1
        denial_started.set()
        release_denial.wait(1.0)
        return MessageAcceptanceOutcome(
            status="denied",
            acceptance=None,
            channel_generation=3,
            connection_id="conn_alpha",
        )

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = SimpleNamespace(  # noqa: SLF001
        wait_for_anchored_message_acceptance=MagicMock(return_value=None),
        deny_message_acceptance=deny_message_acceptance,
    )
    runtime._router = object()  # noqa: SLF001
    coordinates = {
        "tenant_key": "tenant_test",
        "agent_id": "agt_alpha",
        "bot_principal_id": "bot_alpha",
        "app_id": "cli_alpha",
        "channel_generation": 3,
        "connection_id": "conn_alpha",
        "chat_id": "oc_singleflight",
        "message_id": "om_singleflight",
        "timeout": 0,
    }

    try:
        with pytest.raises(EmployeeMessageHandoffUnknownError):
            runtime.wait_for_employee_message_handoff(**coordinates)
        assert denial_started.wait(1.0)
        with pytest.raises(EmployeeMessageHandoffUnknownError):
            runtime.wait_for_employee_message_handoff(**coordinates)
        assert denial_calls == 1
    finally:
        release_denial.set()
        assert runtime._shutdown_employee_handoff_finalization_executor(
            timeout=1.0,
        )

    assert denial_calls == 1


def test_close_reports_blocked_handoff_finalizer_without_waiting_on_ingress_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        employee_composition,
        "_EMPLOYEE_HANDOFF_FINALIZATION_CLOSE_SECONDS",
        0.05,
        raising=False,
    )
    fs_ops = BlockingFsOps()
    service, writer, _store = _service(tmp_path, fs_ops=fs_ops)
    _payload, metadata = _bound_message(
        raw_chat_id="oc_handoff_close_fsync",
        raw_message_id="om_handoff_close_fsync",
    )
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = service  # noqa: SLF001
    runtime._writer = writer  # noqa: SLF001
    runtime._router = SimpleNamespace(  # noqa: SLF001
        record_snapshot=lambda _acceptance_id: None,
    )
    handoff_outcome: list[BaseException | bool] = []
    close_outcome: list[BaseException] = []

    def wait_for_handoff() -> None:
        try:
            handoff_outcome.append(
                runtime.wait_for_employee_message_handoff(
                    tenant_key=metadata.tenant_key,
                    agent_id=metadata.agent_id,
                    bot_principal_id=metadata.bot_principal_id,
                    app_id=metadata.app_id,
                    channel_generation=metadata.channel_generation,
                    connection_id=metadata.connection_id,
                    chat_id="oc_handoff_close_fsync",
                    message_id="om_handoff_close_fsync",
                    timeout=0.05,
                )
            )
        except BaseException as exc:
            handoff_outcome.append(exc)

    def close_runtime() -> None:
        try:
            runtime.close()
        except BaseException as exc:
            close_outcome.append(exc)

    waiter = employee_composition.threading.Thread(target=wait_for_handoff)
    closer = employee_composition.threading.Thread(target=close_runtime)
    waiter.start()
    close_completed_before_fsync = False
    dependencies_preserved = False
    pending_finalizer_preserved = False
    try:
        assert fs_ops.write_started.wait(1.0)
        waiter.join(timeout=0.15)
        closer.start()
        closer.join(timeout=0.15)
        close_completed_before_fsync = not closer.is_alive()
        dependencies_preserved = not service._admission_closed and not service._closed  # noqa: SLF001
        with runtime._employee_handoff_finalization_lock:  # noqa: SLF001
            tracked = tuple(
                runtime._employee_handoff_finalization_futures.values()  # noqa: SLF001
            )
        pending_finalizer_preserved = (
            len(tracked) == 1
            and not tracked[0].done()
            and not tracked[0].cancelled()
        )
    finally:
        fs_ops.allow_fsync.set()
        waiter.join(timeout=2.0)
        closer.join(timeout=2.0)

    try:
        assert len(handoff_outcome) == 1
        assert isinstance(handoff_outcome[0], EmployeeMessageHandoffUnknownError)
        assert close_completed_before_fsync
        assert dependencies_preserved
        assert len(close_outcome) == 1
        assert isinstance(close_outcome[0], RuntimeError)
        assert "handoff_finalization" in str(close_outcome[0])
        assert runtime._close_incomplete is True  # noqa: SLF001
        assert pending_finalizer_preserved
        runtime.close()
        assert runtime._close_incomplete is False  # noqa: SLF001
        assert service._closed is True  # noqa: SLF001
    finally:
        runtime.close()


def test_handoff_deny_commit_failure_remains_unknown(tmp_path: Path) -> None:
    service, writer, _store = _service(tmp_path)
    _payload, metadata = _bound_message(
        raw_chat_id="oc_handoff_unknown",
        raw_message_id="om_handoff_unknown",
    )
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = service
    runtime._router = SimpleNamespace(
        record_snapshot=lambda _acceptance_id: None,
    )
    runtime._closing = False

    try:
        with (
            patch.object(
                service,
                "deny_message_acceptance",
                side_effect=IngressWriteDisabledError("anchor unavailable"),
            ),
            pytest.raises(EmployeeMessageHandoffUnknownError),
        ):
            runtime.wait_for_employee_message_handoff(
                tenant_key=metadata.tenant_key,
                agent_id=metadata.agent_id,
                bot_principal_id=metadata.bot_principal_id,
                app_id=metadata.app_id,
                channel_generation=metadata.channel_generation,
                connection_id=metadata.connection_id,
                chat_id="oc_handoff_unknown",
                message_id="om_handoff_unknown",
                timeout=0.25,
            )
    finally:
        service.close()
        writer.close()


class _AdvancingEvent:
    def __init__(self, clock: list[float]) -> None:
        self.clock = clock
        self.waits: list[float] = []

    def clear(self) -> None:
        return None

    def wait(self, timeout: float) -> bool:
        self.waits.append(timeout)
        self.clock[0] += timeout
        return False


class _ReacquireTrapEvent(employee_composition.threading.Event):
    def __init__(self) -> None:
        super().__init__()
        self.wait_entered = employee_composition.threading.Event()

    def wait(self, timeout: float | None = None) -> bool:
        self.wait_entered.set()
        return super().wait(timeout)


def test_handoff_deadline_does_not_reacquire_contended_condition_after_wait() -> None:
    acceptance = SimpleNamespace(acceptance_id="acc_condition_reacquire")
    outcome = MessageAcceptanceOutcome(
        status="accepted",
        acceptance=acceptance,
        channel_generation=3,
        connection_id="conn_alpha",
    )
    progress = _ReacquireTrapEvent()
    revision_lock = employee_composition.threading.Lock()
    holder_acquired = employee_composition.threading.Event()
    release_holder = employee_composition.threading.Event()
    result: list[BaseException | bool] = []

    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = SimpleNamespace(
        wait_for_anchored_message_acceptance=MagicMock(return_value=outcome),
    )
    runtime._router = object()
    runtime._closing = False
    runtime._employee_handoff_revision = 0
    runtime._employee_handoff_event = progress
    runtime._employee_handoff_revision_lock = revision_lock
    runtime._employee_handoff_projection_result = MagicMock(
        return_value=employee_composition._EmployeeHandoffProjection.PENDING,
    )
    runtime._abandon_unconfirmed_employee_handoff = MagicMock(
        side_effect=EmployeeMessageHandoffUnknownError("handoff deadline expired"),
    )

    def hold_condition_after_wait_starts() -> None:
        if not progress.wait_entered.wait(0.5):
            return
        with revision_lock:
            holder_acquired.set()
            release_holder.wait(1.0)

    def wait_for_handoff() -> None:
        try:
            result.append(
                runtime.wait_for_employee_message_handoff(
                    tenant_key="tenant_test",
                    agent_id="agt_alpha",
                    bot_principal_id="bot_alpha",
                    app_id="cli_alpha",
                    channel_generation=3,
                    connection_id="conn_alpha",
                    chat_id="oc_condition_reacquire",
                    message_id="om_condition_reacquire",
                    timeout=0.05,
                )
            )
        except BaseException as exc:  # captured for the calling test thread
            result.append(exc)

    holder = employee_composition.threading.Thread(
        target=hold_condition_after_wait_starts,
    )
    waiter = employee_composition.threading.Thread(target=wait_for_handoff)
    started = employee_composition.time.monotonic()
    holder.start()
    waiter.start()
    try:
        progress.wait_entered.wait(0.1)
        if progress.wait_entered.is_set():
            assert holder_acquired.wait(0.1)
        waiter.join(timeout=0.15)
        completed_before_release = not waiter.is_alive()
    finally:
        release_holder.set()
        holder.join(timeout=1.0)
        waiter.join(timeout=1.0)

    assert completed_before_release
    assert employee_composition.time.monotonic() - started < 0.5
    assert len(result) == 1
    assert isinstance(result[0], EmployeeMessageHandoffUnknownError)


def test_handoff_notification_rotates_epoch_and_wakes_all_existing_waiters() -> None:
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._employee_handoff_revision = 0
    runtime._employee_handoff_event = employee_composition.threading.Event()
    runtime._employee_handoff_revision_lock = employee_composition.threading.Lock()

    old_epoch, _lock = runtime._employee_handoff_wait_state()
    started = employee_composition.threading.Barrier(3)
    awakened: list[int] = []

    def wait_on_old_epoch(index: int) -> None:
        started.wait()
        if old_epoch.wait(0.5):
            awakened.append(index)

    waiters = [
        employee_composition.threading.Thread(target=wait_on_old_epoch, args=(index,))
        for index in range(2)
    ]
    for waiter in waiters:
        waiter.start()
    started.wait()
    runtime._notify_employee_handoff_progress()
    for waiter in waiters:
        waiter.join(timeout=1.0)
    new_epoch, _lock = runtime._employee_handoff_wait_state()

    assert sorted(awakened) == [0, 1]
    assert old_epoch.wait(0)
    assert new_epoch is not old_epoch
    assert not new_epoch.wait(0)
    assert runtime._employee_handoff_revision == 1


def test_handoff_public_wait_uses_one_deadline_but_finalization_is_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    acceptance = SimpleNamespace(acceptance_id="acc_shared_deadline")

    def accepted_after_most_of_budget(**kwargs):
        assert kwargs["timeout"] == pytest.approx(0.75)
        clock[0] += 0.7
        return MessageAcceptanceOutcome(
            status="accepted",
            acceptance=acceptance,
            channel_generation=3,
            connection_id="conn_alpha",
        )

    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = SimpleNamespace(
        wait_for_anchored_message_acceptance=accepted_after_most_of_budget,
    )
    runtime._router = object()
    runtime._closing = False
    runtime._employee_handoff_revision = 0
    progress = _AdvancingEvent(clock)
    runtime._employee_handoff_event = progress
    runtime._employee_handoff_revision_lock = employee_composition.threading.Lock()
    runtime._employee_handoff_projection_result = MagicMock(
        return_value=employee_composition._EmployeeHandoffProjection.PENDING,
    )
    runtime._abandon_unconfirmed_employee_handoff = MagicMock(return_value=False)
    monkeypatch.setattr(employee_composition.time, "monotonic", lambda: clock[0])

    assert (
        runtime.wait_for_employee_message_handoff(
            tenant_key="tenant_test",
            agent_id="agt_alpha",
            bot_principal_id="bot_alpha",
            app_id="cli_alpha",
            channel_generation=3,
            connection_id="conn_alpha",
            chat_id="oc_shared_deadline",
            message_id="om_shared_deadline",
            timeout=1.0,
        )
        is False
    )

    assert progress.waits == [pytest.approx(0.05)]
    assert (
        runtime._abandon_unconfirmed_employee_handoff.call_args.kwargs[
            "deadline"
        ]
        is None
    )


def test_handoff_revision_storm_cannot_starve_expired_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [200.0]
    acceptance = SimpleNamespace(acceptance_id="acc_revision_storm")
    outcome = MessageAcceptanceOutcome(
        status="accepted",
        acceptance=acceptance,
        channel_generation=3,
        connection_id="conn_alpha",
    )
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = SimpleNamespace(
        wait_for_anchored_message_acceptance=MagicMock(return_value=outcome),
    )
    runtime._router = object()
    runtime._closing = False
    runtime._employee_handoff_revision = 0
    runtime._employee_handoff_event = _AdvancingEvent(clock)
    runtime._employee_handoff_revision_lock = employee_composition.threading.Lock()
    projections = 0

    def noisy_projection(**_kwargs):
        nonlocal projections
        projections += 1
        if projections > 3:
            raise AssertionError("expired deadline was starved by revisions")
        clock[0] += 0.6
        runtime._employee_handoff_revision += 1
        return employee_composition._EmployeeHandoffProjection.PENDING

    runtime._employee_handoff_projection_result = noisy_projection
    runtime._abandon_unconfirmed_employee_handoff = MagicMock(return_value=False)
    monkeypatch.setattr(employee_composition.time, "monotonic", lambda: clock[0])

    with pytest.raises(
        EmployeeMessageHandoffUnknownError,
        match="completed after the public deadline",
    ):
        runtime.wait_for_employee_message_handoff(
            tenant_key="tenant_test",
            agent_id="agt_alpha",
            bot_principal_id="bot_alpha",
            app_id="cli_alpha",
            channel_generation=3,
            connection_id="conn_alpha",
            chat_id="oc_revision_storm",
            message_id="om_revision_storm",
            timeout=1.0,
        )

    assert projections == 2
    runtime._abandon_unconfirmed_employee_handoff.assert_called_once()


def test_handoff_projection_programming_assertion_propagates() -> None:
    acceptance = SimpleNamespace(acceptance_id="acc_projection_assertion")
    outcome = MessageAcceptanceOutcome(
        status="accepted",
        acceptance=acceptance,
        channel_generation=3,
        connection_id="conn_alpha",
    )
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = SimpleNamespace(
        wait_for_anchored_message_acceptance=MagicMock(return_value=outcome),
    )
    runtime._router = object()
    runtime._closing = False
    runtime._employee_handoff_projection_result = MagicMock(
        side_effect=AssertionError("projection invariant failed"),
    )

    with pytest.raises(AssertionError, match="projection invariant failed"):
        runtime.wait_for_employee_message_handoff(
            tenant_key="tenant_test",
            agent_id="agt_alpha",
            bot_principal_id="bot_alpha",
            app_id="cli_alpha",
            channel_generation=3,
            connection_id="conn_alpha",
            chat_id="oc_projection_assertion",
            message_id="om_projection_assertion",
            timeout=0.25,
        )


def test_handoff_acceptance_winner_without_exact_alias_remains_unknown(
    tmp_path: Path,
) -> None:
    service, writer, _store = _service(tmp_path)
    payload, original = _bound_message(
        raw_chat_id="oc_alias_unknown",
        raw_message_id="om_alias_unknown",
    )
    service.accept(original, payload, request_id="req_original_alias")
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = service
    runtime._router = object()
    runtime._closing = False

    try:
        with pytest.raises(
            EmployeeMessageHandoffUnknownError,
            match="durable finalization is still pending",
        ):
            runtime.wait_for_employee_message_handoff(
                tenant_key=original.tenant_key,
                agent_id=original.agent_id,
                bot_principal_id=original.bot_principal_id,
                app_id=original.app_id,
                channel_generation=original.channel_generation + 1,
                connection_id="conn_new_alias",
                chat_id="oc_alias_unknown",
                message_id="om_alias_unknown",
                timeout=0.25,
            )
    finally:
        service.close()
        assert runtime._shutdown_employee_handoff_finalization_executor(
            timeout=1.0,
        )
        writer.close()


def test_handoff_deny_race_uses_the_exact_winning_alias() -> None:
    acceptance = SimpleNamespace(acceptance_id="acc_alias_race")
    canonical = MessageAcceptanceOutcome(
        status="accepted",
        acceptance=acceptance,
        channel_generation=3,
        connection_id="conn_original",
    )
    exact = MessageAcceptanceOutcome(
        status="accepted",
        acceptance=acceptance,
        channel_generation=4,
        connection_id="conn_redelivery",
    )
    ingress = SimpleNamespace(
        wait_for_anchored_message_acceptance=MagicMock(
            side_effect=(None, exact),
        ),
        deny_message_acceptance=MagicMock(return_value=canonical),
    )
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = ingress
    runtime._router = object()
    runtime._closing = False
    runtime._employee_handoff_projection_result = MagicMock(
        return_value=employee_composition._EmployeeHandoffProjection.OWNED,
    )

    assert runtime.wait_for_employee_message_handoff(
        tenant_key="tenant_test",
        agent_id="agt_alpha",
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
        channel_generation=4,
        connection_id="conn_redelivery",
        chat_id="oc_alias_race",
        message_id="om_alias_race",
        timeout=0.25,
    )
    ingress.deny_message_acceptance.assert_called_once()


def test_handoff_deny_lock_contention_returns_then_finishes_durable_denial(
    tmp_path: Path,
) -> None:
    service, writer, _store = _service(tmp_path)
    _payload, metadata = _bound_message(
        raw_chat_id="oc_handoff_lock",
        raw_message_id="om_handoff_lock",
    )
    entered = employee_composition.threading.Event()
    release = employee_composition.threading.Event()

    def occupy_ingress() -> None:
        with service._mutex:  # noqa: SLF001 - deterministic lock contention
            entered.set()
            release.wait(1)

    holder = employee_composition.threading.Thread(target=occupy_ingress)
    holder.start()
    assert entered.wait(1)
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = service
    runtime._router = object()
    runtime._closing = False
    before = tuple(writer.replay())

    try:
        started = employee_composition.time.monotonic()
        with pytest.raises(EmployeeMessageHandoffUnknownError):
            runtime.wait_for_employee_message_handoff(
                tenant_key=metadata.tenant_key,
                agent_id=metadata.agent_id,
                bot_principal_id=metadata.bot_principal_id,
                app_id=metadata.app_id,
                channel_generation=metadata.channel_generation,
                connection_id=metadata.connection_id,
                chat_id="oc_handoff_lock",
                message_id="om_handoff_lock",
                timeout=0.05,
            )
        assert employee_composition.time.monotonic() - started < 0.2
    finally:
        release.set()
        holder.join(1)

    assert runtime._shutdown_employee_handoff_finalization_executor(timeout=1.0)
    denied = service.observe_anchored_message_acceptance(
        tenant_key=metadata.tenant_key,
        agent_id=metadata.agent_id,
        bot_principal_id=metadata.bot_principal_id,
        app_id=metadata.app_id,
        event_type=metadata.event_type,
        chat_id=metadata.chat_id,
        message_id=metadata.message_id,
        channel_generation=metadata.channel_generation,
        connection_id=metadata.connection_id,
    )
    assert denied is not None
    assert denied.status == "denied"
    assert tuple(writer.replay()) != before
    service.close()
    writer.close()


@pytest.mark.parametrize("router_state", ("queued", "terminal"))
def test_exact_reconnect_alias_uses_canonical_router_ownership(
    tmp_path: Path,
    router_state: str,
) -> None:
    with _accepted_targeted_ingress(
        tmp_path,
        marker=f"canonical_alias_{router_state}",
    ) as accepted:
        redelivery_generation = _CHANNEL_GENERATION + 1
        redelivery_connection = "conn_reconnected_alias"
        redelivery = replace(
            accepted.metadata,
            channel_generation=redelivery_generation,
            connection_id=redelivery_connection,
            event_id=f"evt_alias_{router_state}",
        )
        duplicate = accepted.ingress.accept(
            redelivery,
            accepted.payload,
            request_id=f"req_alias_{router_state}",
        )
        assert duplicate.duplicate is True
        record = _terminal_record(
            accepted,
            reason_code="completed",
            queued_sequence=31,
        )
        if router_state == "queued":
            record = replace(record, state="queued", reason_code="")
        runtime = _runtime_for_handoff(record, accepted.ingress)

        assert (
            runtime.wait_for_employee_message_handoff(
                tenant_key=accepted.metadata.tenant_key,
                agent_id=accepted.metadata.agent_id,
                bot_principal_id=accepted.metadata.bot_principal_id,
                app_id=accepted.metadata.app_id,
                channel_generation=redelivery_generation,
                connection_id=redelivery_connection,
                chat_id=str(
                    accepted.payload.normalized_parts[0]["remote_chat_id"]
                ),
                message_id=str(
                    accepted.payload.normalized_parts[0]["remote_message_id"]
                ),
                timeout=0.25,
            )
            is True
        )


def test_expired_prequeue_acceptance_abandon_failure_remains_unknown() -> None:
    acceptance = SimpleNamespace(acceptance_id="acc_prequeue_expired")
    outcome = MessageAcceptanceOutcome(
        status="accepted",
        acceptance=acceptance,
        channel_generation=3,
        connection_id="conn_alpha",
    )
    routed = SimpleNamespace(state="accepted")
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = SimpleNamespace(
        wait_for_anchored_message_acceptance=MagicMock(return_value=outcome),
        record_snapshot=lambda _acceptance_id: None,
    )
    runtime._router = SimpleNamespace(
        record_snapshot=lambda _acceptance_id: routed,
        abandon_message_handoff=MagicMock(
            side_effect=RouterWriteDisabledError("deadline expired")
        ),
    )
    runtime._closing = False

    with pytest.raises(EmployeeMessageHandoffUnknownError):
        runtime.wait_for_employee_message_handoff(
            tenant_key="tenant_test",
            agent_id="agt_alpha",
            bot_principal_id="bot_alpha",
            app_id="cli_alpha",
            channel_generation=3,
            connection_id="conn_alpha",
            chat_id="oc_prequeue_expired",
            message_id="om_prequeue_expired",
            timeout=0,
        )


def test_closing_runtime_returns_unknown_after_exact_acceptance() -> None:
    acceptance = SimpleNamespace(acceptance_id="acc_closing")
    outcome = MessageAcceptanceOutcome(
        status="accepted",
        acceptance=acceptance,
        channel_generation=3,
        connection_id="conn_alpha",
    )
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = SimpleNamespace(
        wait_for_anchored_message_acceptance=MagicMock(return_value=outcome),
    )
    runtime._router = object()
    runtime._closing = True

    with pytest.raises(EmployeeMessageHandoffUnknownError, match="closing"):
        runtime.wait_for_employee_message_handoff(
            tenant_key="tenant_test",
            agent_id="agt_alpha",
            bot_principal_id="bot_alpha",
            app_id="cli_alpha",
            channel_generation=3,
            connection_id="conn_alpha",
            chat_id="oc_closing",
            message_id="om_closing",
            timeout=0,
        )


@pytest.mark.parametrize("router_state", ("queued", "terminal_denied"))
def test_zero_timeout_observes_real_current_router_projection(
    tmp_path: Path,
    router_state: str,
) -> None:
    with _accepted_targeted_ingress(
        tmp_path,
        marker=f"zero_timeout_{router_state}",
    ) as accepted:
        record = _terminal_record(
            accepted,
            reason_code=("" if router_state == "queued" else "authority_denied"),
            queued_sequence=(31 if router_state == "queued" else 0),
        )
        if router_state == "queued":
            record = replace(record, state="queued")
        else:
            ingress_record = accepted.ingress.record_snapshot(accepted.acceptance_id)
            assert ingress_record is not None
            record = replace(
                record,
                authority=None,
                team_id=accepted.metadata.chat_id,
                requester_principal_id=accepted.metadata.sender_principal_id,
            )
        router = object.__new__(DurableEmployeeIngressRouter)
        router._mutex = employee_composition.threading.RLock()
        router._state = RouterProjectionState(
            by_acceptance_id={accepted.acceptance_id: record},
        )
        runtime = object.__new__(EmployeeDepartmentRuntime)
        runtime._ingress = accepted.ingress
        runtime._router = router
        runtime._closing = False
        runtime._employee_handoff_revision = 0
        runtime._employee_handoff_event = employee_composition.threading.Event()
        runtime._employee_handoff_revision_lock = (
            employee_composition.threading.Lock()
        )
        runtime._employee_handoff_revision_lock.acquire()

        try:
            result = runtime.wait_for_employee_message_handoff(
                tenant_key=accepted.metadata.tenant_key,
                agent_id=accepted.metadata.agent_id,
                bot_principal_id=accepted.metadata.bot_principal_id,
                app_id=accepted.metadata.app_id,
                channel_generation=accepted.metadata.channel_generation,
                connection_id=accepted.metadata.connection_id,
                chat_id=str(
                    accepted.payload.normalized_parts[0]["remote_chat_id"]
                ),
                message_id=str(
                    accepted.payload.normalized_parts[0]["remote_message_id"]
                ),
                timeout=0,
            )
        finally:
            runtime._employee_handoff_revision_lock.release()

        assert result is (router_state == "queued")
