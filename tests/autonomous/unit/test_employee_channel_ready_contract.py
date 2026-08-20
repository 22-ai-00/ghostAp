"""Fail-closed READY authority and immutable supervisor status snapshots."""

from __future__ import annotations

import copy
import threading
from types import SimpleNamespace

import pytest

from src.autonomous.provisioning.channel_protocol import (
    ChannelFrame,
    FrameType,
    ProtocolError,
    decode_frame,
    encode_frame,
)
from src.autonomous.supervisor.channel_models import ChannelProcessState
from src.autonomous.supervisor.employee_channels import (
    ChannelProcessStatus,
    EmployeeChannelSupervisor,
)


def _ready_payload(*, app_id: str = "cli_alpha") -> dict[str, object]:
    return {
        "identity": {
            "app_id": app_id,
            "open_id": "ou_employee_alpha",
        },
        "connection_id": "conn_employee_alpha",
        "connection": {
            "observed": True,
            "sdk_connection_id": "sdk-connection-alpha",
            "service_id": "service-alpha",
            "secure": True,
        },
    }


def _status() -> ChannelProcessStatus:
    payload = _ready_payload()
    return ChannelProcessStatus(
        agent_id="agt_alpha",
        app_id="cli_alpha",
        generation=3,
        pid=1234,
        state=ChannelProcessState.READY,
        identity=payload["identity"],  # type: ignore[arg-type]
        ready_metadata={
            "connection_id": payload["connection_id"],
            "connection": payload["connection"],
        },
    )


def _runtime(status: ChannelProcessStatus) -> SimpleNamespace:
    return SimpleNamespace(
        process=SimpleNamespace(poll=lambda: None),
        status=status,
        stopping=False,
        requires_observed_connection=False,
        ready=threading.Event(),
    )


def _supervisor() -> EmployeeChannelSupervisor:
    return EmployeeChannelSupervisor(
        secret_resolver=lambda *_args: "unused",
        worker_path=__file__,
    )


def test_status_returns_a_deep_snapshot_without_live_ready_aliases() -> None:
    supervisor = _supervisor()
    runtime = _runtime(_status())
    supervisor._runtimes["agt_alpha"] = runtime  # type: ignore[assignment]

    first = supervisor.status("agt_alpha")
    assert first is not None
    first.identity["app_id"] = "cli_attacker"
    first.ready_metadata["connection_id"] = "conn_attacker"
    first.ready_metadata["connection"]["sdk_connection_id"] = "sdk-attacker"

    current = supervisor.status("agt_alpha")

    assert current is not None
    assert current.identity == {
        "app_id": "cli_alpha",
        "open_id": "ou_employee_alpha",
    }
    assert current.ready_metadata["connection_id"] == "conn_employee_alpha"
    assert current.ready_metadata["connection"]["sdk_connection_id"] == (
        "sdk-connection-alpha"
    )


def test_ready_protocol_round_trips_exact_authority_schema() -> None:
    frame = ChannelFrame(
        FrameType.READY,
        "agt_alpha",
        3,
        1,
        _ready_payload(),
    )

    assert decode_frame(encode_frame(frame)) == frame


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value["identity"].pop("app_id"),
        lambda value: value["identity"].pop("open_id"),
        lambda value: value["identity"].update({"unknown": True}),
        lambda value: value["identity"].update({"app_id": 7}),
        lambda value: value["identity"].update({"open_id": 7}),
        lambda value: value["identity"].update({"open_id": "user_alpha"}),
        lambda value: value["identity"].update({"open_id": "ou_alpha\nother"}),
        lambda value: value.update({"connection_id": 7}),
        lambda value: value.update({"connection_id": "connection_alpha"}),
        lambda value: value["connection"].update({"unknown": True}),
        lambda value: value["connection"].update({"observed": False}),
        lambda value: value["connection"].update({"secure": 1}),
        lambda value: value["connection"].update({"sdk_connection_id": ""}),
        lambda value: value["connection"].update({"service_id": None}),
    ],
)
def test_ready_protocol_rejects_unknown_or_wrong_typed_authority(
    mutation,
) -> None:
    payload = copy.deepcopy(_ready_payload())
    mutation(payload)
    frame = ChannelFrame(FrameType.READY, "agt_alpha", 3, 1, payload)

    with pytest.raises(ProtocolError, match="ready"):
        encode_frame(frame)


def test_supervisor_rejects_ready_identity_for_another_runtime_app() -> None:
    supervisor = _supervisor()
    runtime = _runtime(
        ChannelProcessStatus(
            agent_id="agt_alpha",
            app_id="cli_alpha",
            generation=3,
            pid=1234,
            state=ChannelProcessState.STARTING,
        )
    )

    supervisor._accept_frame(  # type: ignore[arg-type]
        runtime,
        ChannelFrame(
            FrameType.READY,
            "agt_alpha",
            3,
            1,
            _ready_payload(app_id="cli_other"),
        ),
    )

    assert runtime.status.state is ChannelProcessState.STARTING
    assert runtime.status.error_code == "invalid-ready"
    assert not runtime.ready.is_set()


def test_supervisor_copies_valid_ready_payload_before_publishing_authority() -> None:
    supervisor = _supervisor()
    runtime = _runtime(
        ChannelProcessStatus(
            agent_id="agt_alpha",
            app_id="cli_alpha",
            generation=3,
            pid=1234,
            state=ChannelProcessState.STARTING,
        )
    )
    payload = _ready_payload()

    supervisor._accept_frame(  # type: ignore[arg-type]
        runtime,
        ChannelFrame(FrameType.READY, "agt_alpha", 3, 1, payload),
    )
    payload["identity"]["app_id"] = "cli_attacker"
    payload["connection"]["sdk_connection_id"] = "sdk-attacker"

    assert runtime.status.state is ChannelProcessState.READY
    assert runtime.status.identity["app_id"] == "cli_alpha"
    assert runtime.status.ready_metadata["connection"]["sdk_connection_id"] == (
        "sdk-connection-alpha"
    )
    assert runtime.ready.is_set()
