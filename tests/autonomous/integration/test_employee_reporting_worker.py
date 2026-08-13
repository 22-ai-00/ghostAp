from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from types import SimpleNamespace

import pytest

from src.autonomous.acceptance.main_bot_warning_outbox import (
    MainBotWarningRetryableDeliveryError,
)
from src.autonomous.ingress.service import IngressBlobRetryableError
from src.autonomous.outbox.delivery import (
    EmployeeOutboxDrainResult,
    EmployeeOutboxItemDeliveryError,
)
from src.autonomous.outbox.projection import OutboxProjectionError
from src.autonomous.provisioning.composition import (
    EmployeeDepartmentRuntime,
    EmployeeDispatchReportingDeferredError,
)
from src.trust.models import ActorKind, TrustZone


class _EmptyIngress:
    def __init__(self) -> None:
        self.state = SimpleNamespace(by_acceptance_id={})

    def rebuild_projection(self) -> None:
        return None

    def gc_terminal_payloads(self) -> int:
        return 0


class _EmptyRouter:
    def __init__(self) -> None:
        self.state = SimpleNamespace(by_acceptance_id={})

    def rebuild_projection(self) -> None:
        return None


class _BlockingDispatch:
    employee_runtime = None

    def __init__(self) -> None:
        self.prepared = False
        self.execute_started = threading.Event()
        self.release_execute = threading.Event()
        self.latest_stage = ""

    def prepare_next(self) -> object | None:
        if self.prepared:
            return None
        self.prepared = True
        self.latest_stage = "queued"
        return self

    def execute_prepared(self, prepared: object) -> object:
        assert prepared is self
        self.latest_stage = "running"
        self.execute_started.set()
        assert self.release_execute.wait(5.0)
        self.latest_stage = "terminal"
        return object()

    def recover_incomplete_attempts(self) -> tuple[object, ...]:
        return ()

    def reconcile_terminal_snapshots(self) -> int:
        return 0


def _blocking_runtime() -> tuple[EmployeeDepartmentRuntime, _BlockingDispatch]:
    runtime = EmployeeDepartmentRuntime()
    dispatch = _BlockingDispatch()
    runtime._ingress = _EmptyIngress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = _EmptyRouter()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = dispatch  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        gc_superseded_snapshots=lambda: 0,
    )
    runtime._outbox_delivery = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._fire = None  # noqa: SLF001
    return runtime, dispatch


def _stop_runtime_workers(
    runtime: EmployeeDepartmentRuntime,
    dispatch: _BlockingDispatch,
) -> None:
    dispatch.release_execute.set()
    runtime._dispatch_stop.set()  # noqa: SLF001
    reporting_stop = getattr(runtime, "_reporting_stop", None)
    if isinstance(reporting_stop, threading.Event):
        reporting_stop.set()
    reporting_wakeup = getattr(runtime, "_reporting_wakeup", None)
    if isinstance(reporting_wakeup, threading.Event):
        reporting_wakeup.set()
    for name in ("_dispatch_thread", "_reporting_thread"):
        worker = getattr(runtime, name, None)
        if isinstance(worker, threading.Thread):
            worker.join(timeout=5.0)


def test_reporting_worker_delivers_running_card_while_execution_is_blocked() -> None:
    runtime, dispatch = _blocking_runtime()
    running_delivered = threading.Event()

    runtime._reconcile_terminal_ingress = lambda: 0  # type: ignore[method-assign]  # noqa: SLF001

    def drain_outbox() -> bool:
        if dispatch.latest_stage == "running":
            running_delivered.set()
            return True
        return False

    runtime._drain_employee_outbox_once = drain_outbox  # type: ignore[method-assign]  # noqa: SLF001

    runtime._start_dispatch_worker()  # noqa: SLF001
    try:
        assert dispatch.execute_started.wait(2.0)
        assert running_delivered.wait(2.0)
        assert not dispatch.release_execute.is_set()
    finally:
        _stop_runtime_workers(runtime, dispatch)


def test_warning_retry_backoff_survives_unrelated_reporting_wakeups() -> None:
    runtime, dispatch = _blocking_runtime()
    warning_attempts: list[float] = []
    employee_ticks = 0
    first_warning = threading.Event()

    def warning() -> bool:
        warning_attempts.append(time.monotonic())
        first_warning.set()
        raise MainBotWarningRetryableDeliveryError("projection deadline")

    def drain_employee() -> bool:
        nonlocal employee_ticks
        employee_ticks += 1
        return False

    runtime._drain_main_bot_warning_outbox_once = warning  # type: ignore[method-assign]  # noqa: SLF001
    runtime._reconcile_terminal_ingress = lambda: 0  # type: ignore[method-assign]  # noqa: SLF001
    runtime._recover_retirement_delivery_channels = lambda: ()  # type: ignore[method-assign]  # noqa: SLF001
    runtime._drain_employee_outbox_once = drain_employee  # type: ignore[method-assign]  # noqa: SLF001
    runtime._start_reporting_worker()  # noqa: SLF001
    try:
        assert first_warning.wait(1.0)
        for _ in range(20):
            runtime._reporting_wakeup.set()  # noqa: SLF001
            time.sleep(0.005)
        assert len(warning_attempts) == 1
        assert employee_ticks > 1
    finally:
        _stop_runtime_workers(runtime, dispatch)


def test_reporting_worker_starts_after_actor_recovery_fails() -> None:
    delivered = threading.Event()
    outbox_rebuilds: list[None] = []

    class _Service:
        projection_state = SimpleNamespace()

        def recover(self) -> None:
            return None

        def recover_replay_safe_action_required(self) -> None:
            return None

        def list_states(self) -> tuple[()]:
            return ()

        def mark_runtime_recovered(self) -> None:
            return None

    class _ActorRuntime:
        def recover(self) -> None:
            raise RuntimeError("unrelated actor replay failed")

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = _Service()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        employee_runtime=_ActorRuntime(),
    )
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        rebuild_projection=lambda: outbox_rebuilds.append(None),
    )
    runtime._outbox_delivery = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._refresh_context_bindings = lambda _projection: True  # type: ignore[method-assign]  # noqa: SLF001
    runtime._drain_employee_reporting_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: delivered.set() or True
    )

    try:
        runtime._recover_once()  # noqa: SLF001

        assert runtime._execution_blockers == ("employee_actor_recovery",)  # noqa: SLF001
        assert outbox_rebuilds == [None]
        assert delivered.wait(2.0)
    finally:
        runtime._reporting_stop.set()  # noqa: SLF001
        runtime._reporting_wakeup.set()  # noqa: SLF001
        if runtime._reporting_thread is not None:  # noqa: SLF001
            runtime._reporting_thread.join(timeout=5.0)  # noqa: SLF001


def test_reporting_worker_concurrent_start_publishes_one_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = object()  # type: ignore[assignment]  # noqa: SLF001
    callers_ready = threading.Barrier(3)
    second_constructor_entered = threading.Event()
    constructed: list[_FakeReportingThread] = []

    class _FakeReportingThread:
        def __init__(self, **_kwargs: object) -> None:
            self.alive = False
            self.starts = 0
            constructed.append(self)
            if len(constructed) == 1:
                second_constructor_entered.wait(0.5)
            else:
                second_constructor_entered.set()

        def start(self) -> None:
            self.starts += 1
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

    def start_reporting() -> None:
        callers_ready.wait(2.0)
        runtime._start_reporting_worker()  # noqa: SLF001

    real_thread = threading.Thread
    starters = [real_thread(target=start_reporting) for _ in range(2)]
    monkeypatch.setattr(
        "src.autonomous.provisioning.composition.threading.Thread",
        _FakeReportingThread,
    )
    for starter in starters:
        starter.start()
    callers_ready.wait(2.0)
    for starter in starters:
        starter.join(timeout=2.0)

    assert all(not starter.is_alive() for starter in starters)
    assert len(constructed) == 1
    assert constructed[0].starts == 1
    assert runtime._reporting_thread is constructed[0]  # noqa: SLF001


def test_reporting_worker_start_and_close_publish_one_stopped_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = EmployeeDepartmentRuntime()
    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    close_finished = threading.Event()
    close_errors: list[BaseException] = []
    constructed: list[_FakeReportingThread] = []

    class _Outbox:
        @staticmethod
        def stop_admission() -> None:
            return None

        @staticmethod
        def list_pending_delivery_records() -> tuple[object, ...]:
            return ()

        @staticmethod
        def close() -> None:
            return None

    class _Delivery:
        @staticmethod
        def deliver_pending(**_kwargs: object) -> EmployeeOutboxDrainResult:
            return EmployeeOutboxDrainResult(0, (), (), ())

    class _FakeReportingThread:
        def __init__(self, **_kwargs: object) -> None:
            self.alive = False
            self.starts = 0
            self.joins = 0
            constructed.append(self)
            constructor_entered.set()
            assert release_constructor.wait(2.0)

        def start(self) -> None:
            self.starts += 1
            self.alive = True

        def join(self, *, timeout: float) -> None:
            assert timeout == 5.0
            self.joins += 1
            self.alive = False

        def is_alive(self) -> bool:
            return self.alive

    def close_runtime() -> None:
        try:
            runtime.close()
        except BaseException as exc:  # pragma: no cover - assertion reports fault
            close_errors.append(exc)
        finally:
            close_finished.set()

    runtime._outbox = _Outbox()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = _Delivery()  # type: ignore[assignment]  # noqa: SLF001
    real_thread = threading.Thread
    starter = real_thread(target=runtime._start_reporting_worker)  # noqa: SLF001
    closer = real_thread(target=close_runtime)

    class _ThreadingProxy:
        def __getattr__(self, name: str) -> object:
            if name == "Thread":
                return _FakeReportingThread
            return getattr(threading, name)

    monkeypatch.setattr(
        "src.autonomous.provisioning.composition.threading",
        _ThreadingProxy(),
    )

    starter.start()
    assert constructor_entered.wait(1.0)
    closer.start()
    close_overtook_publication = close_finished.wait(0.2)
    release_constructor.set()
    starter.join(timeout=2.0)
    closer.join(timeout=2.0)

    assert not close_overtook_publication
    assert not close_errors
    assert not starter.is_alive()
    assert not closer.is_alive()
    assert len(constructed) == 1
    assert constructed[0].starts == 1
    assert constructed[0].joins == 1
    assert not constructed[0].is_alive()
    assert runtime._reporting_stop.is_set()  # noqa: SLF001


def test_reporting_worker_delivers_rebalanced_target_failure_while_other_execution_blocks() -> None:
    runtime, dispatch = _blocking_runtime()
    target_terminal = threading.Event()
    target_failure_anchored = threading.Event()
    target_failure_delivered = threading.Event()

    def reconcile_terminal() -> int:
        if target_terminal.is_set() and not target_failure_anchored.is_set():
            target_failure_anchored.set()
            return 1
        return 0

    def drain_outbox() -> bool:
        if target_failure_anchored.is_set():
            target_failure_delivered.set()
            return True
        return False

    runtime._reconcile_terminal_ingress = reconcile_terminal  # type: ignore[method-assign]  # noqa: SLF001
    runtime._drain_employee_outbox_once = drain_outbox  # type: ignore[method-assign]  # noqa: SLF001

    runtime._start_dispatch_worker()  # noqa: SLF001
    try:
        assert dispatch.execute_started.wait(2.0)
        target_terminal.set()
        assert target_failure_delivered.wait(2.0)
        assert not dispatch.release_execute.is_set()
    finally:
        _stop_runtime_workers(runtime, dispatch)


def test_admission_singleflight_counts_one_transient_blob_failure_per_acceptance() -> None:
    acceptance_id = "acc_singleflight_transient_blob"
    start = threading.Barrier(2)
    second_blob_reader = threading.Event()
    counts_lock = threading.Lock()
    get_payload_calls = 0

    record = SimpleNamespace(disposition=None, metadata=SimpleNamespace())

    class _Ingress:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: record})

        def get_payload(self, _acceptance_id: str) -> object:
            nonlocal get_payload_calls
            with counts_lock:
                get_payload_calls += 1
                if get_payload_calls >= 2:
                    second_blob_reader.set()
            second_blob_reader.wait(0.25)
            raise IngressBlobRetryableError("transient encrypted Inbox read")

    class _Router:
        def __init__(self) -> None:
            self.state = SimpleNamespace(by_acceptance_id={})
            self.eligible = True
            self.defer_calls = 0

        def is_inbox_candidate_eligible(self, _acceptance_id: str) -> bool:
            return self.eligible

        def defer_inbox_candidate(self, _acceptance_id: str) -> None:
            self.defer_calls += 1
            self.eligible = False

    router = _Router()
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = router  # type: ignore[assignment]  # noqa: SLF001
    results: list[bool] = []
    errors: list[BaseException] = []

    def admit() -> None:
        try:
            start.wait(2.0)
            results.append(runtime._admit_employee_ingress_once(acceptance_id))  # noqa: SLF001
        except BaseException as exc:  # pragma: no cover - assertion reports worker faults
            errors.append(exc)

    workers = [threading.Thread(target=admit) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3.0)

    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    assert sorted(results) == [False, True]
    assert get_payload_calls == 1
    assert router.defer_calls == 1
    # Keep the bounded wait honest: the serialized reader must have waited for
    # the competing call rather than racing through before both threads started.
    assert not second_blob_reader.is_set()
    assert time.monotonic() > 0


def test_admission_singleflight_does_not_disposition_after_peer_queues_same_acceptance() -> None:
    acceptance_id = "acc_singleflight_queue_wins"
    start = threading.Barrier(2)
    route_started = threading.Event()
    release_route = threading.Event()
    disposition_calls: list[str] = []
    route_calls = 0
    state_lock = threading.Lock()

    metadata = SimpleNamespace()
    record = SimpleNamespace(disposition=None, metadata=metadata)
    payload = SimpleNamespace(
        normalized_parts=({"type": "message", "chat_type": "group"},)
    )

    class _Ingress:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: record})

        def get_payload(self, _acceptance_id: str) -> object:
            return payload

        def record_disposition(
            self,
            _acceptance_id: str,
            *,
            state: str,
            reason_code: str,
        ) -> None:
            disposition_calls.append(f"{state}:{reason_code}")

    class _Router:
        def __init__(self) -> None:
            self.state = SimpleNamespace(by_acceptance_id={})

        def is_inbox_candidate_eligible(self, _acceptance_id: str) -> bool:
            return True

        def route(self, _acceptance_id: str) -> None:
            nonlocal route_calls
            with state_lock:
                route_calls += 1
            route_started.set()
            assert release_route.wait(2.0)
            self.state.by_acceptance_id[acceptance_id] = SimpleNamespace(
                state="queued"
            )

    router = _Router()
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = router  # type: ignore[assignment]  # noqa: SLF001
    runtime._owner_p2p_requester = lambda *_args: None  # type: ignore[method-assign]  # noqa: SLF001
    runtime._authorized_targeted_group_task = lambda *_args: None  # type: ignore[method-assign]  # noqa: SLF001
    runtime._managed_employee_ingress_trust = lambda *_args: SimpleNamespace(  # type: ignore[method-assign]  # noqa: SLF001
        zone=TrustZone.MANAGED_AGENT_GROUP,
        actor=ActorKind.OWNER,
    )
    runtime._handle_control_ingress = lambda *_args, **_kwargs: False  # type: ignore[method-assign]  # noqa: SLF001
    runtime._handle_main_bot_group_command_ingress = lambda *_args, **_kwargs: False  # type: ignore[method-assign]  # noqa: SLF001
    runtime._record_employee_ingress_group_event = lambda *_args: True  # type: ignore[method-assign]  # noqa: SLF001

    # B must enter while A is still inside route(). Without the keyed
    # singleflight, both callers cross the same pre-route projection snapshot.
    results: list[bool] = []
    errors: list[BaseException] = []

    def admit() -> None:
        try:
            start.wait(2.0)
            results.append(runtime._admit_employee_ingress_once(acceptance_id))  # noqa: SLF001
        except BaseException as exc:  # pragma: no cover - assertion reports worker faults
            errors.append(exc)

    workers = [threading.Thread(target=admit) for _ in range(2)]
    for worker in workers:
        worker.start()
    assert route_started.wait(2.0)
    time.sleep(0.05)
    release_route.set()
    for worker in workers:
        worker.join(timeout=3.0)

    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    assert sorted(results) == [False, True]
    assert route_calls == 1
    assert disposition_calls == []


def test_reporting_repair_failure_does_not_block_healthy_outbox_delivery() -> None:
    events: list[str] = []

    class _BrokenRecoveryDispatch:
        employee_runtime = None

        @staticmethod
        def recover_incomplete_attempts() -> tuple[object, ...]:
            events.append("repair")
            raise EmployeeDispatchReportingDeferredError("one deferred attempt")

        @staticmethod
        def reconcile_terminal_snapshots() -> int:
            events.append("snapshot")
            raise EmployeeDispatchReportingDeferredError("same deferred attempt")

    runtime = EmployeeDepartmentRuntime()
    runtime._dispatch = _BrokenRecoveryDispatch()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch_recovery_pending = True  # noqa: SLF001
    runtime._ingress = _EmptyIngress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        gc_superseded_snapshots=lambda: 0,
    )
    runtime._reconcile_terminal_ingress = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("terminal") or 0
    )
    runtime._drain_employee_outbox_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("healthy_outbox") or True
    )

    assert runtime._drain_employee_reporting_once() is True  # noqa: SLF001
    assert events == ["repair", "snapshot", "terminal", "healthy_outbox"]
    assert runtime._dispatch_recovery_pending is True  # noqa: SLF001


def test_reporting_repair_backoff_keeps_gc_fenced_but_runs_independent_work() -> None:
    events: list[str] = []

    class _BrokenRecoveryDispatch:
        @staticmethod
        def recover_incomplete_attempts() -> tuple[object, ...]:
            events.append("repair")
            raise EmployeeDispatchReportingDeferredError("one deferred attempt")

        @staticmethod
        def reconcile_terminal_snapshots() -> int:
            events.append("snapshot")
            raise EmployeeDispatchReportingDeferredError("same deferred attempt")

    runtime = EmployeeDepartmentRuntime()
    runtime._dispatch = _BrokenRecoveryDispatch()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch_recovery_pending = True  # noqa: SLF001
    runtime._reconcile_terminal_ingress = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("terminal") or 0
    )
    runtime._drain_employee_outbox_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("outbox") or True
    )
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        gc_superseded_snapshots=lambda: events.append("outbox_gc") or 0,
    )
    runtime._ingress = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        gc_terminal_payloads=lambda: events.append("ingress_gc") or 0,
    )

    assert runtime._drain_employee_reporting_once() is True  # noqa: SLF001
    assert runtime._drain_employee_reporting_once() is True  # noqa: SLF001

    assert events == [
        "repair",
        "snapshot",
        "terminal",
        "outbox",
        "terminal",
        "outbox",
    ]


def test_live_reporting_worker_is_woken_without_synchronous_control_drain() -> None:
    runtime = EmployeeDepartmentRuntime()
    runtime._reporting_thread = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        is_alive=lambda: True,
    )
    runtime._drain_employee_outbox_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: pytest.fail("control path must not perform production delivery")
    )

    runtime._request_employee_outbox_delivery()  # noqa: SLF001

    assert runtime._reporting_wakeup.is_set()  # noqa: SLF001


def test_control_delivery_keeps_synchronous_fallback_without_reporting_worker() -> None:
    events: list[str] = []
    runtime = EmployeeDepartmentRuntime()
    runtime._reporting_thread = None  # noqa: SLF001
    runtime._drain_employee_outbox_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("drain") or True
    )

    runtime._request_employee_outbox_delivery()  # noqa: SLF001

    assert events == ["drain"]


def test_close_drain_accepts_idle_probe_after_exact_iteration_budget() -> None:
    attempts = 0
    probes = 0
    runtime = EmployeeDepartmentRuntime()

    class _Delivery:
        @staticmethod
        def deliver_pending(**_kwargs: object) -> EmployeeOutboxDrainResult:
            nonlocal attempts
            attempts += 1
            return EmployeeOutboxDrainResult(
                pending_count=1,
                attempted_outbox_ids=(f"out_{attempts}",),
                delivered_outbox_ids=(f"out_{attempts}",),
                failed_outbox_ids=(),
            )

    def pending(*, deadline: float | None = None) -> tuple[object, ...]:
        nonlocal probes
        assert isinstance(deadline, float)
        probes += 1
        return ()

    runtime._outbox_delivery = _Delivery()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        list_pending_delivery_records=pending,
    )

    runtime._drain_employee_outbox_until_idle(max_batches=2)  # noqa: SLF001

    assert attempts == 2
    assert probes == 1


def test_close_drain_rejects_work_remaining_after_final_idle_probe() -> None:
    attempts = 0
    probes = 0
    runtime = EmployeeDepartmentRuntime()

    class _Delivery:
        @staticmethod
        def deliver_pending(**_kwargs: object) -> EmployeeOutboxDrainResult:
            nonlocal attempts
            attempts += 1
            return EmployeeOutboxDrainResult(
                pending_count=1,
                attempted_outbox_ids=(f"out_{attempts}",),
                delivered_outbox_ids=(f"out_{attempts}",),
                failed_outbox_ids=(),
            )

    def pending(*, deadline: float | None = None) -> tuple[object, ...]:
        nonlocal probes
        assert isinstance(deadline, float)
        probes += 1
        return (object(),)

    runtime._outbox_delivery = _Delivery()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        list_pending_delivery_records=pending,
    )

    with pytest.raises(RuntimeError, match="did not drain"):
        runtime._drain_employee_outbox_until_idle(max_batches=2)  # noqa: SLF001

    assert attempts == 2
    assert probes == 1


def test_reporting_projection_failure_stops_outbox_and_gc_for_current_tick() -> None:
    events: list[str] = []
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _EmptyIngress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        gc_superseded_snapshots=lambda: events.append("outbox_gc") or 0,
    )

    def fail_projection() -> int:
        events.append("terminal_projection")
        raise OutboxProjectionError("anchored projection is invalid")

    runtime._reconcile_terminal_ingress = fail_projection  # type: ignore[method-assign]  # noqa: SLF001
    runtime._drain_employee_outbox_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("outbox_delivery") or True
    )

    with pytest.raises(OutboxProjectionError, match="projection is invalid"):
        runtime._drain_employee_reporting_once()  # noqa: SLF001

    assert events == ["terminal_projection"]
    assert not hasattr(runtime, "_employee_dispatch_next_gc_at")


def test_reporting_repair_programming_failure_propagates_after_terminal_fence() -> None:
    events: list[str] = []

    class _BrokenDispatch:
        @staticmethod
        def recover_incomplete_attempts() -> tuple[object, ...]:
            events.append("repair")
            raise NameError("undefined repair dependency")

        @staticmethod
        def reconcile_terminal_snapshots() -> int:
            events.append("snapshot")
            return 0

    runtime = EmployeeDepartmentRuntime()
    runtime._dispatch = _BrokenDispatch()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch_recovery_pending = True  # noqa: SLF001
    runtime._reconcile_terminal_ingress = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("terminal") or 0
    )
    runtime._drain_employee_outbox_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("outbox") or True
    )

    with pytest.raises(RuntimeError, match="reporting recovery failed") as raised:
        runtime._drain_employee_reporting_once()  # noqa: SLF001

    assert isinstance(raised.value.__cause__, NameError)
    assert events == ["repair", "snapshot", "terminal"]


def test_reporting_repair_programming_failure_wins_over_deferred_peer() -> None:
    events: list[str] = []

    class _BrokenDispatch:
        @staticmethod
        def recover_incomplete_attempts() -> tuple[object, ...]:
            events.append("repair")
            raise EmployeeDispatchReportingDeferredError("one deferred attempt")

        @staticmethod
        def reconcile_terminal_snapshots() -> int:
            events.append("snapshot")
            raise NameError("undefined snapshot dependency")

    runtime = EmployeeDepartmentRuntime()
    runtime._dispatch = _BrokenDispatch()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch_recovery_pending = True  # noqa: SLF001
    runtime._reconcile_terminal_ingress = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("terminal") or 0
    )
    runtime._drain_employee_outbox_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("outbox") or True
    )

    with pytest.raises(RuntimeError, match="reporting recovery failed") as raised:
        runtime._drain_employee_reporting_once()  # noqa: SLF001

    assert isinstance(raised.value.__cause__, NameError)
    assert events == ["repair", "snapshot", "terminal"]


def test_reporting_gc_failure_does_not_advance_retry_deadline() -> None:
    events: list[str] = []
    runtime = EmployeeDepartmentRuntime()
    runtime._reconcile_terminal_ingress = lambda: 0  # type: ignore[method-assign]  # noqa: SLF001
    runtime._drain_employee_outbox_once = lambda: False  # type: ignore[method-assign]  # noqa: SLF001
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        gc_superseded_snapshots=lambda: events.append("outbox_gc") or 0,
    )

    class _Ingress(_EmptyIngress):
        def gc_terminal_payloads(self) -> int:
            events.append("inbox_gc")
            raise OSError("blob quarantine unavailable")

    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(OSError, match="blob quarantine unavailable"):
        runtime._drain_employee_reporting_once()  # noqa: SLF001

    assert events == ["outbox_gc", "inbox_gc"]
    assert not hasattr(runtime, "_employee_dispatch_next_gc_at")


def test_runtime_fair_drain_raises_when_every_attempt_is_deferred() -> None:
    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        deliver_pending=lambda **_kwargs: SimpleNamespace(
            attempted_outbox_ids=("out_bad",),
            delivered_outbox_ids=(),
            failed_outbox_ids=("out_bad",),
        )
    )

    with pytest.raises(EmployeeOutboxItemDeliveryError):
        runtime._drain_employee_outbox_once()  # noqa: SLF001


def test_runtime_fair_drain_reports_mixed_batch_progress() -> None:
    observed: list[int] = []
    runtime = EmployeeDepartmentRuntime()
    runtime._outbox = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_delivery = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        deliver_pending=lambda *, max_items: observed.append(max_items)
        or SimpleNamespace(
            attempted_outbox_ids=("out_unavailable", "out_healthy"),
            delivered_outbox_ids=("out_healthy",),
            failed_outbox_ids=("out_unavailable",),
        )
    )

    assert runtime._drain_employee_outbox_once() is True  # noqa: SLF001
    assert observed == [16]


@pytest.mark.parametrize("action_identity", ("", "unexpected-action"))
def test_legacy_unproved_message_is_disposed_before_router_eligibility(
    action_identity: str,
) -> None:
    legacy_id = "acc_legacy_unproved"
    healthy_id = "acc_healthy_proved"
    dispositions: list[tuple[str, str, str]] = []
    legacy = SimpleNamespace(
        disposition=None,
        transport_message_proof=False,
        metadata=SimpleNamespace(
            event_type="im.message.receive_v1",
            action_identity=action_identity,
        ),
    )
    healthy = SimpleNamespace(
        disposition=None,
        transport_message_proof=True,
        metadata=SimpleNamespace(
            event_type="im.message.receive_v1",
            action_identity="",
        ),
    )

    class _Ingress:
        state = SimpleNamespace(
            by_acceptance_id={legacy_id: legacy, healthy_id: healthy}
        )

        @staticmethod
        def get_payload(acceptance_id: str) -> object:
            assert acceptance_id == healthy_id
            return SimpleNamespace(
                normalized_parts=({"type": "message", "chat_type": "group"},)
            )

        @staticmethod
        def record_disposition(
            acceptance_id: str,
            *,
            state: str,
            reason_code: str,
        ) -> None:
            dispositions.append((acceptance_id, state, reason_code))
            legacy.disposition = SimpleNamespace(
                state=state,
                reason_code=reason_code,
            )

    class _Router:
        def __init__(self) -> None:
            self.state = SimpleNamespace(by_acceptance_id={})
            self.routed: list[str] = []

        @staticmethod
        def is_inbox_candidate_eligible(acceptance_id: str) -> bool:
            assert acceptance_id != legacy_id
            return True

        def route(self, acceptance_id: str) -> None:
            self.routed.append(acceptance_id)
            self.state.by_acceptance_id[acceptance_id] = SimpleNamespace(
                state="queued"
            )

    router = _Router()
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = router  # type: ignore[assignment]  # noqa: SLF001
    runtime._owner_p2p_requester = lambda *_args: "ou_owner"  # type: ignore[method-assign]  # noqa: SLF001
    runtime._authorized_targeted_group_task = lambda *_args: None  # type: ignore[method-assign]  # noqa: SLF001
    runtime._managed_employee_ingress_trust = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda *_args: runtime._unknown_employee_ingress_trust()  # noqa: SLF001
    )
    runtime._handle_control_ingress = lambda *_args, **_kwargs: False  # type: ignore[method-assign]  # noqa: SLF001

    assert runtime._admit_employee_ingress_once(legacy_id) is True  # noqa: SLF001
    assert runtime._admit_employee_ingress_once(healthy_id) is True  # noqa: SLF001

    assert dispositions == [
        (legacy_id, "terminal", "invalid_transport_proof")
    ]
    assert router.routed == [healthy_id]


def test_channel_admission_uses_runtime_executor_when_default_executor_is_saturated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = EmployeeDepartmentRuntime()
    admitted = threading.Event()

    runtime._admit_employee_ingress_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda _acceptance_id: admitted.set() or True
    )

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        original_run_in_executor = loop.run_in_executor

        def reject_default_executor(executor, func, *args):
            if executor is None:
                raise RuntimeError("default executor is saturated")
            return original_run_in_executor(executor, func, *args)

        monkeypatch.setattr(loop, "run_in_executor", reject_default_executor)
        admission = asyncio.create_task(
            runtime._handle_channel_event(  # noqa: SLF001
                "hire_alpha",
                1,
                {
                    "event": "durableIngressAccepted",
                    "data": {"acceptance_id": "acc_dedicated_executor"},
                },
            )
        )
        deadline = time.monotonic() + 2.0
        while not admitted.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert admitted.is_set()
        await admission

    try:
        asyncio.run(scenario())
    finally:
        runtime._shutdown_employee_admission_executor(timeout=2.0)  # noqa: SLF001


def test_saturated_admission_executor_falls_back_to_durable_dispatch_scan() -> None:
    runtime = EmployeeDepartmentRuntime()
    runtime._employee_admission_max_pending = 1  # noqa: SLF001
    runtime._employee_admission_futures = {  # type: ignore[assignment]  # noqa: SLF001
        concurrent.futures.Future()
    }

    asyncio.run(
        runtime._handle_channel_event(  # noqa: SLF001
            "hire_alpha",
            1,
            {
                "event": "durableIngressAccepted",
                "data": {"acceptance_id": "acc_saturated_queue"},
            },
        )
    )

    assert runtime._dispatch_wakeup.is_set()  # noqa: SLF001
    assert runtime._employee_admission_executor is None  # noqa: SLF001


def test_admission_shutdown_initiates_executor_close_after_timeout() -> None:
    release = threading.Event()
    runtime = EmployeeDepartmentRuntime()
    runtime._admit_employee_ingress_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda _acceptance_id: release.wait(2.0)
    )
    future = runtime._submit_employee_admission("acc_slow_shutdown")  # noqa: SLF001
    assert future is not None
    deadline = time.monotonic() + 2.0
    while not future.running() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert runtime._shutdown_employee_admission_executor(timeout=0.01) is False  # noqa: SLF001
    assert runtime._employee_admission_executor is None  # noqa: SLF001
    with pytest.raises(RuntimeError, match="executor is closed"):
        runtime._submit_employee_admission("acc_after_shutdown")  # noqa: SLF001

    release.set()
    future.result(timeout=2.0)
