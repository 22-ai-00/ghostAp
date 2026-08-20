"""Typed employee Channel failures distinguish retryable transport from defects."""

from __future__ import annotations

import ast
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from lark_oapi.core.exception import (
    AccessTokenException,
    NoAuthorizationException,
    ObtainAccessTokenException,
)

import src.autonomous.provisioning.channel_worker as channel_worker_module
from src.autonomous.provisioning.channel_protocol import (
    ChannelFrame,
    FrameType,
    ProtocolError,
    decode_frame,
    encode_frame,
)
from src.autonomous.provisioning.channel_worker import _handle_low_level_outbound
from src.autonomous.provisioning.lark_outbound import EmployeeOutboundError
from src.autonomous.supervisor.channel_models import (
    ChannelProcessState,
    EmployeeChannelOutboundError,
)
from src.autonomous.supervisor.employee_channels import (
    ChannelProcessStatus,
    EmployeeChannelSupervisor,
    _Runtime,
)


def _outbound_frame(frame_type: FrameType) -> ChannelFrame:
    return ChannelFrame(
        frame_type,
        "agt_employee",
        3,
        1,
        (
            {
                "request_id": "send_employee",
                "target": "oc_team",
                "message": {"text": "hello"},
                "options": None,
            }
            if frame_type is FrameType.SEND
            else {
                "request_id": "update_employee",
                "message_id": "om_card",
                "card": {"schema": "2.0"},
            }
        ),
    )


def test_worker_defers_outbound_sdk_imports_until_after_process_hardening() -> None:
    tree = ast.parse(Path(channel_worker_module.__file__).read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imported_modules.update(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module is not None:
            imported_modules.add(statement.module)

    assert "lark_oapi.core.exception" not in imported_modules
    assert "requests.exceptions" not in imported_modules
    assert (
        "src.autonomous.provisioning.lark_outbound" not in imported_modules
    )


def _worker_failure(
    frame_type: FrameType,
    error: BaseException,
) -> dict[str, object]:
    emitted: list[tuple[FrameType, dict[str, object]]] = []

    class _Outbound:
        def send(self, *_args, **_kwargs):
            raise error

        def update_card(self, *_args, **_kwargs):
            raise error

    _handle_low_level_outbound(
        _outbound_frame(frame_type),
        SimpleNamespace(app_id="cli_employee", generation=3),
        _Outbound(),
        SimpleNamespace(
            wait_snapshot=lambda **_kwargs: (3, "conn_employee")
        ),
        SimpleNamespace(
            emit=lambda kind, payload: emitted.append((kind, payload))
        ),
    )

    assert len(emitted) == 1
    assert emitted[0][0] is FrameType.HEALTH
    return emitted[0][1]


@pytest.mark.parametrize("frame_type", [FrameType.SEND, FrameType.UPDATE_CARD])
@pytest.mark.parametrize(
    "error",
    [RuntimeError("boom"), ValueError("boom"), AssertionError(), AttributeError()],
)
def test_worker_marks_unexpected_outbound_exceptions_as_internal(
    frame_type: FrameType,
    error: BaseException,
) -> None:
    failure = _worker_failure(frame_type, error)

    assert failure == {
        "operation": "send" if frame_type is FrameType.SEND else "update_card",
        "request_id": (
            "send_employee"
            if frame_type is FrameType.SEND
            else "update_employee"
        ),
        "success": False,
        "failure_kind": "internal",
        "error_code": "outbound-internal-error",
    }


@pytest.mark.parametrize("frame_type", [FrameType.SEND, FrameType.UPDATE_CARD])
def test_worker_marks_lark_rejection_as_remote(
    frame_type: FrameType,
) -> None:
    failure = _worker_failure(
        frame_type,
        EmployeeOutboundError("remote rejected"),
    )

    assert failure["success"] is False
    assert failure["failure_kind"] == "remote"
    assert failure["error_code"] == "remote-rejected"


@pytest.mark.parametrize("frame_type", [FrameType.SEND, FrameType.UPDATE_CARD])
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),
        ConnectionError(),
        requests.exceptions.Timeout(),
        requests.exceptions.ConnectionError(),
    ],
)
def test_worker_marks_connection_failures_as_transport(
    frame_type: FrameType,
    error: BaseException,
) -> None:
    failure = _worker_failure(frame_type, error)

    assert failure["success"] is False
    assert failure["failure_kind"] == "transport"
    assert failure["error_code"] == "transport-unavailable"


@pytest.mark.parametrize(
    "error",
    [
        AccessTokenException(401, 99991663, "invalid token", "rejected"),
        ObtainAccessTokenException("token rejected", 99991663, "rejected"),
    ],
)
def test_worker_marks_lark_access_token_rejection_as_remote(
    error: BaseException,
) -> None:
    failure = _worker_failure(FrameType.SEND, error)

    assert failure["failure_kind"] == "remote"
    assert failure["error_code"] == "remote-rejected"


def test_worker_keeps_local_authorization_configuration_failure_internal() -> None:
    failure = _worker_failure(
        FrameType.SEND,
        NoAuthorizationException("missing local authorization context"),
    )

    assert failure["failure_kind"] == "internal"
    assert failure["error_code"] == "outbound-internal-error"


def _nack_payload(failure_kind: str) -> dict[str, object]:
    return {
        "operation": "send",
        "request_id": "send_employee",
        "success": False,
        "failure_kind": failure_kind,
        "error_code": f"{failure_kind}-failure",
    }


@pytest.mark.parametrize("failure_kind", ["transport", "remote", "internal"])
def test_outbound_nack_protocol_round_trips_only_the_typed_failure_shape(
    failure_kind: str,
) -> None:
    frame = ChannelFrame(
        FrameType.HEALTH,
        "agt_employee",
        3,
        2,
        _nack_payload(failure_kind),
    )

    assert decode_frame(encode_frame(frame)) == frame


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("failure_kind"),
        lambda payload: payload.update({"failure_kind": "future-kind"}),
        lambda payload: payload.update({"failure_kind": 7}),
        lambda payload: payload.update({"unknown": True}),
        lambda payload: payload.update({"error_code": ""}),
    ],
)
def test_outbound_nack_protocol_rejects_missing_unknown_or_extra_fields(
    mutation,
) -> None:
    payload = _nack_payload("internal")
    mutation(payload)

    with pytest.raises(ProtocolError, match="health"):
        encode_frame(
            ChannelFrame(
                FrameType.HEALTH,
                "agt_employee",
                3,
                2,
                payload,
            )
        )


def _supervisor_nack(
    failure_kind: str,
    *,
    followed_by_success: bool = False,
) -> BaseException:
    supervisor = EmployeeChannelSupervisor(
        secret_resolver=lambda *_args: "unused",
        worker_path=__file__,
    )
    runtime = SimpleNamespace(
        status=ChannelProcessStatus(
            agent_id="agt_employee",
            app_id="cli_employee",
            generation=3,
            pid=1234,
            state=ChannelProcessState.READY,
            identity={
                "app_id": "cli_employee",
                "open_id": "ou_employee",
            },
            ready_metadata={"connection_id": "conn_employee"},
        ),
        pending_lock=threading.Lock(),
        pending_sends={},
    )
    supervisor._runtimes["agt_employee"] = runtime  # type: ignore[assignment]

    def emit_nack(_runtime, _frame_type, payload, *, deadline=None):
        assert deadline is not None
        supervisor._accept_frame(  # type: ignore[arg-type]
            runtime,
            ChannelFrame(
                FrameType.HEALTH,
                "agt_employee",
                3,
                2,
                {
                    "operation": "send",
                    "request_id": payload["request_id"],
                    "success": False,
                    "failure_kind": failure_kind,
                    "error_code": f"{failure_kind}-failure",
                },
            ),
        )
        if followed_by_success:
            supervisor._accept_frame(  # type: ignore[arg-type]
                runtime,
                ChannelFrame(
                    FrameType.HEALTH,
                    "agt_employee",
                    3,
                    3,
                    {
                        "operation": "send",
                        "request_id": payload["request_id"],
                        "success": True,
                        "app_id": "cli_employee",
                        "generation": 3,
                        "connection_id": "conn_employee",
                        "message_id": "om_employee",
                    },
                ),
            )
        return True

    supervisor._send_control = emit_nack  # type: ignore[method-assign]
    try:
        supervisor.send(
            "agt_employee",
            generation=3,
            target="oc_team",
            message={"text": "hello"},
        )
    except BaseException as exc:
        return exc
    raise AssertionError("employee Channel NACK must raise")


@pytest.mark.parametrize("failure_kind", ["transport", "remote"])
def test_supervisor_exposes_only_external_nacks_as_retryable(
    failure_kind: str,
) -> None:
    error = _supervisor_nack(failure_kind)

    assert type(error) is EmployeeChannelOutboundError


def test_supervisor_propagates_internal_nack_as_non_retryable_integrity_error() -> None:
    error = _supervisor_nack("internal")

    assert isinstance(error, RuntimeError)
    assert not isinstance(error, EmployeeChannelOutboundError)
    assert type(error).__name__ == "EmployeeChannelOutboundIntegrityError"


def test_supervisor_keeps_first_terminal_nack_when_late_success_arrives() -> None:
    error = _supervisor_nack("internal", followed_by_success=True)

    assert isinstance(error, RuntimeError)
    assert not isinstance(error, EmployeeChannelOutboundError)
    assert type(error).__name__ == "EmployeeChannelOutboundIntegrityError"


def test_malformed_worker_nack_wakes_pending_send_as_integrity_failure() -> None:
    supervisor = EmployeeChannelSupervisor(
        secret_resolver=lambda *_args: "unused",
        worker_path=__file__,
        send_timeout=0.1,
    )
    event_r, event_w = os.pipe()
    runtime = _Runtime(
        process=SimpleNamespace(poll=lambda: 1),
        control_fd=-1,
        event_fd=event_r,
        status=ChannelProcessStatus(
            agent_id="agt_employee",
            app_id="cli_employee",
            generation=3,
            pid=1234,
            state=ChannelProcessState.READY,
            identity={
                "app_id": "cli_employee",
                "open_id": "ou_employee",
            },
            ready_metadata={"connection_id": "conn_employee"},
        ),
        on_event=lambda _event: None,
    )
    supervisor._runtimes["agt_employee"] = runtime

    def emit_malformed_nack(_runtime, _frame_type, payload, *, deadline=None):
        assert deadline is not None
        raw = {
            "v": 1,
            "type": "HEALTH",
            "agent_id": "agt_employee",
            "generation": 3,
            "sequence": 1,
            "payload": {
                "operation": "send",
                "request_id": payload["request_id"],
                "success": False,
                "failure_kind": "future-kind",
                "error_code": "future-error",
            },
        }
        os.write(
            event_w,
            (json.dumps(raw, separators=(",", ":")) + "\n").encode(),
        )
        return True

    supervisor._send_control = emit_malformed_nack  # type: ignore[method-assign]
    reader = threading.Thread(target=supervisor._read_frames, args=(runtime,))
    reader.start()
    try:
        with pytest.raises(RuntimeError) as raised:
            supervisor.send(
                "agt_employee",
                generation=3,
                target="oc_team",
                message={"text": "hello"},
            )
        assert type(raised.value).__name__ == (
            "EmployeeChannelOutboundIntegrityError"
        )
        assert not isinstance(raised.value, EmployeeChannelOutboundError)
        assert "protocol-error" in str(raised.value)
    finally:
        os.close(event_w)
        reader.join(timeout=1)
        if reader.is_alive():
            raise AssertionError("employee Channel reader did not stop")
