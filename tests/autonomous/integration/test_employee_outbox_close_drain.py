from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from src.autonomous.outbox.delivery import (
    EmployeeDeliveryAuthority,
    EmployeeOutboxDeliveryCoordinator,
    EmployeeOutboxDrainDeadlineExceeded,
    EmployeeOutboxDrainResult,
    EmployeeOutboxItemDeliveryError,
)
from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
from src.autonomous.supervisor.employee_channels import ChannelSendReceipt
from tests.autonomous.integration.test_employee_outbox_fair_delivery import (
    _runtime,
    _snapshot,
)


@dataclass
class _DeadlineChannel:
    calls: list[str]
    deadlines: list[float | None]

    def send(
        self,
        agent_id,
        *,
        generation,
        target,
        message,
        options=None,
        deadline=None,
    ):
        del target, message, options
        self.calls.append(agent_id)
        self.deadlines.append(deadline)
        return ChannelSendReceipt(
            request_id=f"send_{agent_id}",
            success=True,
            app_id=f"cli_{agent_id}",
            generation=generation,
            connection_id=f"conn_{agent_id}",
            message_id=f"om_{agent_id}",
        )

    def update_card(
        self,
        agent_id,
        *,
        generation,
        message_id,
        card,
        deadline=None,
    ):
        del card
        self.calls.append(agent_id)
        self.deadlines.append(deadline)
        return ChannelSendReceipt(
            request_id=f"update_{agent_id}",
            success=True,
            app_id=f"cli_{agent_id}",
            generation=generation,
            connection_id=f"conn_{agent_id}",
            message_id=message_id,
        )


def test_close_scans_past_full_failed_batch_before_declaring_unresolved(
    tmp_path,
) -> None:
    service, writer = _runtime(tmp_path)
    poison_ids = {f"agt_poison_{index:02d}" for index in range(16)}
    for index, agent_id in enumerate(sorted(poison_ids)):
        service.append_snapshot(
            _snapshot(
                agent_id=agent_id,
                attempt_id=f"attempt-poison-{index:02d}",
                created_at=f"2026-08-12T00:00:{index:02d}Z",
            )
        )
    healthy = _snapshot(
        agent_id="agt_healthy_17",
        attempt_id="attempt-healthy-17",
        created_at="2026-08-12T00:00:16Z",
    )
    service.append_snapshot(healthy)
    channel = _DeadlineChannel(calls=[], deadlines=[])

    def resolve(record):
        if record.agent_id in poison_ids:
            raise EmployeeOutboxItemDeliveryError("authority unavailable")
        return EmployeeDeliveryAuthority(
            app_id=f"cli_{record.agent_id}",
            generation=1,
            connection_id=f"conn_{record.agent_id}",
        )

    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = service  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = EmployeeOutboxDeliveryCoordinator(  # noqa: SLF001
        outbox=service,
        channels=channel,
        authority_resolver=resolve,
    )
    deadline = time.monotonic() + 2.0
    try:
        with pytest.raises(EmployeeOutboxItemDeliveryError, match="unresolved"):
            runtime._drain_employee_outbox_until_idle(  # noqa: SLF001
                max_batches=3,
                deadline=deadline,
            )

        assert channel.calls == [healthy.agent_id]
        assert channel.deadlines == [deadline]
        assert service.get_record(healthy.outbox_id).binding is not None
    finally:
        service.close()
        writer.close()


def test_close_batch_budget_uses_read_only_final_probe() -> None:
    results = iter(
        (
            EmployeeOutboxDrainResult(
                pending_count=2,
                attempted_outbox_ids=("out_1",),
                delivered_outbox_ids=("out_1",),
                failed_outbox_ids=(),
            ),
            EmployeeOutboxDrainResult(
                pending_count=1,
                attempted_outbox_ids=("out_2",),
                delivered_outbox_ids=("out_2",),
                failed_outbox_ids=(),
            ),
        )
    )
    calls: list[tuple[int, float | None]] = []
    probes = 0
    deadline = time.monotonic() + 1.0

    class _Delivery:
        @staticmethod
        def deliver_pending(*, max_items, deadline=None):
            calls.append((max_items, deadline))
            return next(results)

    class _Outbox:
        @staticmethod
        def list_pending_delivery_records(*, deadline=None):
            nonlocal probes
            probes += 1
            assert deadline is not None
            return ()

    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = _Outbox()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = _Delivery()  # type: ignore[assignment]  # noqa: SLF001

    runtime._drain_employee_outbox_until_idle(  # noqa: SLF001
        max_batches=2,
        deadline=deadline,
    )

    assert calls == [(16, deadline), (1, deadline)]
    assert probes == 1


def test_expired_close_deadline_stops_before_delivery() -> None:
    calls = 0

    class _Delivery:
        @staticmethod
        def deliver_pending(**_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("expired deadline must stop before delivery")

    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = _Delivery()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(EmployeeOutboxDrainDeadlineExceeded):
        runtime._drain_employee_outbox_until_idle(  # noqa: SLF001
            max_batches=2,
            deadline=time.monotonic() - 1.0,
        )

    assert calls == 0


def test_close_deadline_bounds_wait_for_runtime_delivery_lock() -> None:
    calls = 0

    class _Delivery:
        @staticmethod
        def deliver_pending(**_kwargs: object) -> EmployeeOutboxDrainResult:
            nonlocal calls
            calls += 1
            return EmployeeOutboxDrainResult(0, (), (), ())

    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = _Delivery()  # type: ignore[assignment]  # noqa: SLF001
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_delivery_lock() -> None:
        with runtime._employee_outbox_delivery_lock:  # noqa: SLF001
            lock_held.set()
            assert release_lock.wait(2.0)

    owner = threading.Thread(target=hold_delivery_lock)
    owner.start()
    assert lock_held.wait(1.0)
    delayed_release = threading.Timer(0.3, release_lock.set)
    delayed_release.start()
    started = time.monotonic()
    try:
        with pytest.raises(EmployeeOutboxDrainDeadlineExceeded):
            runtime._deliver_employee_outbox_batch(  # noqa: SLF001
                max_items=1,
                deadline=started + 0.05,
            )
        assert time.monotonic() - started < 0.2
        assert calls == 0
    finally:
        release_lock.set()
        delayed_release.cancel()
        owner.join(timeout=1.0)
    assert not owner.is_alive()


def test_runtime_close_bounds_blocking_outbox_io_and_reuses_drain_future(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.autonomous.provisioning.composition._EMPLOYEE_OUTBOX_CLOSE_DRAIN_SECONDS",
        0.05,
    )
    drain_started = threading.Event()
    release_io = threading.Event()
    drain_finished = threading.Event()
    caller_thread_id = threading.get_ident()
    delivery_calls = 0
    delivery_thread_ids: list[int] = []

    class _Outbox:
        stop_calls = 0
        close_calls = 0

        def stop_admission(self) -> None:
            self.stop_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    class _BlockingDelivery:
        @staticmethod
        def deliver_pending(**_kwargs: object) -> EmployeeOutboxDrainResult:
            nonlocal delivery_calls
            delivery_calls += 1
            delivery_thread_ids.append(threading.get_ident())
            drain_started.set()
            assert release_io.wait(2.0)
            drain_finished.set()
            return EmployeeOutboxDrainResult(0, (), (), ())

    outbox = _Outbox()
    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = outbox  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = _BlockingDelivery()  # type: ignore[assignment]  # noqa: SLF001
    watchdog = threading.Timer(0.5, release_io.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="outbox_drain:EmployeeOutboxDrainDeadlineExceeded"):
            runtime.close()

        assert time.monotonic() - started < 0.2
        assert drain_started.is_set()
        assert not release_io.is_set()
        assert outbox.close_calls == 0
        first_future = runtime._employee_outbox_close_drain_future  # noqa: SLF001
        assert first_future is not None

        release_io.set()
        assert drain_finished.wait(1.0)
        first_future.result(timeout=1.0)

        runtime.close()

        assert runtime._employee_outbox_close_drain_future is first_future  # noqa: SLF001
        assert delivery_calls == 1
        assert all(thread_id != caller_thread_id for thread_id in delivery_thread_ids)
        assert outbox.close_calls == 1
        assert runtime._close_incomplete is False  # noqa: SLF001
    finally:
        release_io.set()
        watchdog.cancel()


def test_runtime_close_retries_terminal_safe_drain_deadline_failure() -> None:
    delivery_calls = 0

    class _Outbox:
        close_calls = 0

        @staticmethod
        def stop_admission() -> None:
            return None

        def close(self) -> None:
            self.close_calls += 1

    class _Delivery:
        @staticmethod
        def deliver_pending(**_kwargs: object) -> EmployeeOutboxDrainResult:
            nonlocal delivery_calls
            delivery_calls += 1
            if delivery_calls == 1:
                raise EmployeeOutboxDrainDeadlineExceeded(
                    "deadline expired before external delivery"
                )
            return EmployeeOutboxDrainResult(0, (), (), ())

    outbox = _Outbox()
    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = outbox  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = _Delivery()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(
        RuntimeError,
        match="outbox_drain:EmployeeOutboxDrainDeadlineExceeded",
    ):
        runtime.close()
    first_future = runtime._employee_outbox_close_drain_future  # noqa: SLF001
    assert first_future is not None and first_future.done()
    assert outbox.close_calls == 0

    runtime.close()

    assert runtime._employee_outbox_close_drain_future is not first_future  # noqa: SLF001
    assert delivery_calls == 2
    assert outbox.close_calls == 1


def test_runtime_close_surfaces_outbox_drain_failure() -> None:
    stopped = False
    delivery_calls = 0

    class _Outbox:
        @staticmethod
        def stop_admission() -> None:
            nonlocal stopped
            stopped = True

        @staticmethod
        def list_pending_delivery_records() -> tuple[object, ...]:
            return (object(),)

    class _Delivery:
        @staticmethod
        def deliver_pending(**_kwargs: object) -> EmployeeOutboxDrainResult:
            nonlocal delivery_calls
            delivery_calls += 1
            raise EmployeeOutboxItemDeliveryError("transport unavailable")

    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = _Outbox()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = _Delivery()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(RuntimeError, match="outbox_drain"):
        runtime.close()
    first_future = runtime._employee_outbox_close_drain_future  # noqa: SLF001

    with pytest.raises(RuntimeError, match="outbox_drain"):
        runtime.close()

    assert stopped is True
    assert delivery_calls == 1
    assert runtime._employee_outbox_close_drain_future is first_future  # noqa: SLF001
    assert runtime._closing is True  # noqa: SLF001
    assert runtime._close_incomplete is True  # noqa: SLF001


def test_runtime_close_does_not_retry_terminal_unknown_timeout() -> None:
    delivery_calls = 0

    class _Outbox:
        @staticmethod
        def stop_admission() -> None:
            return None

    class _Delivery:
        @staticmethod
        def deliver_pending(**_kwargs: object) -> EmployeeOutboxDrainResult:
            nonlocal delivery_calls
            delivery_calls += 1
            raise TimeoutError("external outcome is unknown")

    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = _Outbox()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = _Delivery()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(RuntimeError, match="outbox_drain:TimeoutError"):
        runtime.close()
    first_future = runtime._employee_outbox_close_drain_future  # noqa: SLF001

    with pytest.raises(RuntimeError, match="outbox_drain:TimeoutError"):
        runtime.close()

    assert runtime._employee_outbox_close_drain_future is first_future  # noqa: SLF001
    assert delivery_calls == 1


def test_runtime_close_surfaces_worker_join_timeout() -> None:
    class _StuckWorker:
        @staticmethod
        def join(*, timeout: float) -> None:
            assert timeout == 5.0

        @staticmethod
        def is_alive() -> bool:
            return True

    runtime = EmployeeDepartmentRuntime()
    runtime._dispatch_thread = _StuckWorker()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(RuntimeError, match="dispatch_worker:TimeoutError"):
        runtime.close()

    assert runtime._closing is True  # noqa: SLF001
    assert runtime._close_incomplete is True  # noqa: SLF001
