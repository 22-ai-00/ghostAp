from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import pytest

from src.autonomous.supervisor.channel_models import (
    ChannelProcessState,
    EmployeeChannelOutboundTimeout,
)
from src.autonomous.supervisor.employee_channels import (
    ChannelProcessStatus,
    EmployeeChannelSupervisor,
    _write_all,
)


def _ready_supervisor(
    *,
    send_timeout: float = 10.0,
) -> tuple[EmployeeChannelSupervisor, object]:
    supervisor = EmployeeChannelSupervisor(
        secret_resolver=lambda *_args: "unused",
        worker_path=__file__,
        sandbox_prefix=(),
        send_timeout=send_timeout,
    )
    runtime = SimpleNamespace(
        status=ChannelProcessStatus(
            agent_id="agt_employee",
            app_id="cli_employee",
            generation=3,
            pid=1234,
            state=ChannelProcessState.READY,
            identity={"app_id": "cli_employee", "open_id": "ou_employee"},
            ready_metadata={"connection_id": "conn_employee"},
        ),
        pending_lock=threading.Lock(),
        pending_sends={},
        control_fd=-1,
        control_lock=threading.Lock(),
        outbound_sequence=0,
    )
    supervisor._runtimes["agt_employee"] = runtime  # type: ignore[assignment]
    return supervisor, runtime


@pytest.mark.parametrize(
    "send_timeout",
    [True, 0, -0.1, float("nan"), float("inf"), "soon"],
)
def test_supervisor_rejects_non_positive_or_non_finite_send_timeout(
    send_timeout: object,
) -> None:
    with pytest.raises(ValueError, match="send_timeout"):
        EmployeeChannelSupervisor(
            secret_resolver=lambda *_args: "unused",
            worker_path=__file__,
            sandbox_prefix=(),
            send_timeout=send_timeout,  # type: ignore[arg-type]
        )


def test_outbound_deadline_bounds_supervisor_lock_contention_without_pending_leak() -> None:
    supervisor, runtime = _ready_supervisor()
    holder_ready = threading.Event()
    release_holder = threading.Event()
    result: list[BaseException | object] = []

    def hold_supervisor_lock() -> None:
        with supervisor._lock:
            holder_ready.set()
            release_holder.wait(timeout=1.0)

    def send_with_short_deadline() -> None:
        try:
            result.append(
                supervisor.send(
                    "agt_employee",
                    generation=3,
                    target="oc_team",
                    message={"text": "hello"},
                    deadline=time.monotonic() + 0.03,
                )
            )
        except BaseException as exc:  # captured for the controlling test thread
            result.append(exc)

    holder = threading.Thread(target=hold_supervisor_lock)
    sender = threading.Thread(target=send_with_short_deadline)
    holder.start()
    assert holder_ready.wait(timeout=1.0)
    started = time.monotonic()
    sender.start()
    sender.join(timeout=0.15)
    completed_in_budget = not sender.is_alive()
    try:
        assert completed_in_budget
        assert time.monotonic() - started < 0.5
        assert len(result) == 1
        assert isinstance(result[0], EmployeeChannelOutboundTimeout)
        assert runtime.pending_sends == {}
    finally:
        release_holder.set()
        holder.join(timeout=1.0)
        sender.join(timeout=1.0)
        if holder.is_alive() or sender.is_alive():
            raise AssertionError("employee Channel lock contention threads did not stop")


@pytest.mark.parametrize("receipt_succeeds", [False, True], ids=["timeout", "success"])
def test_receipt_cleanup_does_not_wait_behind_another_full_pipe_write(
    receipt_succeeds: bool,
) -> None:
    supervisor, runtime = _ready_supervisor(send_timeout=1.0)
    blocker = SimpleNamespace(
        status=ChannelProcessStatus(
            agent_id="agt_blocker",
            app_id="cli_blocker",
            generation=4,
            pid=5678,
            state=ChannelProcessState.READY,
            identity={"app_id": "cli_blocker", "open_id": "ou_blocker"},
            ready_metadata={"connection_id": "conn_blocker"},
        ),
        pending_lock=threading.Lock(),
        pending_sends={},
        control_fd=-1,
        control_lock=threading.Lock(),
        outbound_sequence=0,
    )
    supervisor._runtimes["agt_blocker"] = blocker  # type: ignore[assignment]
    receipt_read_fd, receipt_write_fd = os.pipe()
    blocker_read_fd, blocker_write_fd = os.pipe()
    runtime.control_fd = receipt_write_fd
    blocker.control_fd = blocker_write_fd
    os.set_blocking(blocker_write_fd, False)
    while True:
        try:
            os.write(blocker_write_fd, b"x" * 4096)
        except BlockingIOError:
            break
    os.set_blocking(blocker_write_fd, True)

    original_send_control = supervisor._send_control
    receipt_sent = threading.Event()
    blocker_write_started = threading.Event()
    completed_wait_observed = threading.Event()
    release_completed_wait = threading.Event()
    receipt_result: list[BaseException | object] = []
    blocker_result: list[BaseException | object] = []

    class _CompletedBeforeCleanup:
        """Pause a completed receipt exactly before caller-side cleanup."""

        @staticmethod
        def is_set() -> bool:
            return True

        @staticmethod
        def set() -> None:
            return None

        @staticmethod
        def wait(_timeout: float | None = None) -> bool:
            completed_wait_observed.set()
            release_completed_wait.wait(timeout=1.0)
            return True

    def observed_send_control(runtime_arg, frame_type, payload, *, deadline=None):
        if runtime_arg is runtime:
            result = original_send_control(
                runtime_arg,
                frame_type,
                payload,
                deadline=deadline,
            )
            if receipt_succeeds:
                pending = runtime.pending_sends[payload["request_id"]]
                pending.success = True
                pending.app_id = "cli_employee"
                pending.generation = 3
                pending.connection_id = "conn_employee"
                pending.message_id = "om_employee"
                pending.completed = _CompletedBeforeCleanup()
            receipt_sent.set()
            return result
        blocker_write_started.set()
        return original_send_control(
            runtime_arg,
            frame_type,
            payload,
            deadline=deadline,
        )

    supervisor._send_control = observed_send_control  # type: ignore[method-assign]

    def send_receipt_candidate() -> None:
        try:
            receipt_result.append(
                supervisor.send(
                    "agt_employee",
                    generation=3,
                    target="oc_team",
                    message={"text": "receipt"},
                    deadline=time.monotonic() + (0.8 if receipt_succeeds else 0.15),
                )
            )
        except BaseException as exc:  # captured for the controlling test thread
            receipt_result.append(exc)

    def block_other_runtime_pipe() -> None:
        try:
            blocker_result.append(
                supervisor.send(
                    "agt_blocker",
                    generation=4,
                    target="oc_team",
                    message={"text": "blocked"},
                    deadline=time.monotonic() + 1.0,
                )
            )
        except BaseException as exc:  # captured for the controlling test thread
            blocker_result.append(exc)

    receipt_sender = threading.Thread(target=send_receipt_candidate)
    blocker_sender = threading.Thread(target=block_other_runtime_pipe)
    receipt_sender.start()
    assert receipt_sent.wait(timeout=1.0)
    if receipt_succeeds:
        assert completed_wait_observed.wait(timeout=1.0)
    blocker_sender.start()
    assert blocker_write_started.wait(timeout=1.0)
    if receipt_succeeds:
        release_completed_wait.set()

    started = time.monotonic()
    receipt_sender.join(timeout=0.35)
    completed_in_budget = not receipt_sender.is_alive()
    try:
        assert completed_in_budget
        assert time.monotonic() - started < 0.5
        assert len(receipt_result) == 1
        if receipt_succeeds:
            assert not isinstance(receipt_result[0], BaseException)
        else:
            assert isinstance(receipt_result[0], EmployeeChannelOutboundTimeout)
    finally:
        release_completed_wait.set()
        os.close(blocker_read_fd)
        blocker_read_fd = -1
        blocker_sender.join(timeout=1.0)
        receipt_sender.join(timeout=1.0)
        if blocker_sender.is_alive() or receipt_sender.is_alive():
            raise AssertionError("employee Channel outbound threads did not stop")
        os.close(receipt_read_fd)
        if runtime.control_fd >= 0:
            os.close(runtime.control_fd)
            runtime.control_fd = -1
        if blocker.control_fd >= 0:
            os.close(blocker.control_fd)
            blocker.control_fd = -1
        if blocker_read_fd >= 0:
            os.close(blocker_read_fd)

    assert runtime.pending_sends == {}
    assert blocker.pending_sends == {}
    assert blocker.control_fd == -1
    assert len(blocker_result) == 1


def test_outbound_receipt_wait_uses_absolute_deadline() -> None:
    supervisor, _runtime = _ready_supervisor()
    observed_deadlines: list[float | None] = []

    def send_control(_runtime, _frame_type, _payload, *, deadline=None):
        observed_deadlines.append(deadline)
        return True

    supervisor._send_control = send_control  # type: ignore[method-assign]
    deadline = time.monotonic() + 0.03
    started = time.monotonic()

    with pytest.raises(EmployeeChannelOutboundTimeout):
        supervisor.send(
            "agt_employee",
            generation=3,
            target="oc_team",
            message={"text": "hello"},
            deadline=deadline,
        )

    assert time.monotonic() - started < 0.5
    assert observed_deadlines == [deadline]


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf"), "soon"])
def test_outbound_rejects_invalid_absolute_deadline(deadline: object) -> None:
    supervisor, _runtime = _ready_supervisor()

    with pytest.raises(ValueError, match="deadline is invalid"):
        supervisor.send(
            "agt_employee",
            generation=3,
            target="oc_team",
            message={"text": "hello"},
            deadline=deadline,  # type: ignore[arg-type]
        )


def test_expired_outbound_deadline_stops_before_control_write() -> None:
    supervisor, _runtime = _ready_supervisor()
    supervisor._send_control = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: pytest.fail(
            "expired request must not write a control frame"
        )
    )

    with pytest.raises(EmployeeChannelOutboundTimeout, match="deadline exceeded"):
        supervisor.send(
            "agt_employee",
            generation=3,
            target="oc_team",
            message={"text": "hello"},
            deadline=time.monotonic() - 1.0,
        )


def test_deadline_aware_pipe_write_does_not_block_on_full_pipe() -> None:
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    try:
        while True:
            try:
                os.write(write_fd, b"x" * 4096)
            except BlockingIOError:
                break
        os.set_blocking(write_fd, True)
        started = time.monotonic()

        with pytest.raises(TimeoutError, match="timed out"):
            _write_all(
                write_fd,
                b"blocked",
                deadline=time.monotonic() + 0.03,
            )

        assert time.monotonic() - started < 0.5
        assert os.get_blocking(write_fd) is True
    finally:
        os.close(write_fd)
        os.close(read_fd)


def test_outbound_without_caller_deadline_bounds_full_pipe_by_send_timeout() -> None:
    supervisor, runtime = _ready_supervisor(send_timeout=0.03)
    read_fd, write_fd = os.pipe()
    runtime.control_fd = write_fd
    os.set_blocking(write_fd, False)
    while True:
        try:
            os.write(write_fd, b"x" * 4096)
        except BlockingIOError:
            break
    os.set_blocking(write_fd, True)
    result: list[BaseException | object] = []

    def send_without_deadline() -> None:
        try:
            result.append(
                supervisor.send(
                    "agt_employee",
                    generation=3,
                    target="oc_team",
                    message={"text": "hello"},
                )
            )
        except BaseException as exc:  # captured for the controlling test thread
            result.append(exc)

    sender = threading.Thread(target=send_without_deadline)
    started = time.monotonic()
    sender.start()
    sender.join(timeout=0.15)
    completed_in_budget = not sender.is_alive()
    try:
        assert completed_in_budget
        assert time.monotonic() - started < 0.5
        assert len(result) == 1
        assert isinstance(result[0], EmployeeChannelOutboundTimeout)
        assert runtime.control_fd == -1
    finally:
        if sender.is_alive():
            os.close(read_fd)
            sender.join(timeout=1.0)
            read_fd = -1
        if runtime.control_fd >= 0:
            os.close(runtime.control_fd)
            runtime.control_fd = -1
        if read_fd >= 0:
            os.close(read_fd)


def test_timed_out_pipe_write_invalidates_channel_stream() -> None:
    supervisor, runtime = _ready_supervisor()
    read_fd, write_fd = os.pipe()
    runtime.control_fd = write_fd
    os.set_blocking(write_fd, False)
    try:
        while True:
            try:
                os.write(write_fd, b"x" * 4096)
            except BlockingIOError:
                break
        os.set_blocking(write_fd, True)

        with pytest.raises(EmployeeChannelOutboundTimeout):
            supervisor.send(
                "agt_employee",
                generation=3,
                target="oc_team",
                message={"text": "hello"},
                deadline=time.monotonic() + 0.03,
            )

        assert runtime.control_fd == -1
        with pytest.raises(OSError):
            os.fstat(write_fd)
    finally:
        if runtime.control_fd >= 0:
            os.close(runtime.control_fd)
        os.close(read_fd)
