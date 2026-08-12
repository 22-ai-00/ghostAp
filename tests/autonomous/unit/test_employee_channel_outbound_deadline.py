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


def _ready_supervisor() -> tuple[EmployeeChannelSupervisor, object]:
    supervisor = EmployeeChannelSupervisor(
        secret_resolver=lambda *_args: "unused",
        worker_path=__file__,
        sandbox_prefix=(),
        send_timeout=10.0,
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
        pending_sends={},
        control_fd=-1,
        control_lock=threading.Lock(),
        outbound_sequence=0,
    )
    supervisor._runtimes["agt_employee"] = runtime  # type: ignore[assignment]
    return supervisor, runtime


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
