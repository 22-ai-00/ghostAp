from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from src.autonomous.outbox.delivery import (
    EmployeeDeliveryAuthority,
    EmployeeOutboxDeliveryCoordinator,
    EmployeeOutboxDrainDeadlineExceeded,
)
from src.autonomous.supervisor.employee_channels import ChannelSendReceipt
from tests.autonomous.integration.test_employee_outbox_fair_delivery import (
    _runtime,
    _snapshot,
)


@dataclass
class _DeadlineChannel:
    deadlines: list[float | None] = field(default_factory=list)

    def send(
        self,
        agent_id: str,
        *,
        generation: int,
        target: str,
        message: object,
        options: object = None,
        deadline: float | None = None,
    ) -> ChannelSendReceipt:
        del target, message, options
        self.deadlines.append(deadline)
        return ChannelSendReceipt(
            request_id=f"send_{agent_id}",
            success=True,
            app_id=f"cli_{agent_id}",
            generation=generation,
            connection_id=f"conn_{agent_id}",
            message_id=f"om_{agent_id}",
        )

    def update_card(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("new snapshot must use send")


def _coordinator(service, channel: _DeadlineChannel):
    return EmployeeOutboxDeliveryCoordinator(
        outbox=service,
        channels=channel,
        authority_resolver=lambda record: EmployeeDeliveryAuthority(
            app_id=f"cli_{record.agent_id}",
            generation=1,
            connection_id=f"conn_{record.agent_id}",
        ),
    )


def test_outbox_forwards_one_absolute_deadline_to_channel(tmp_path) -> None:
    service, writer = _runtime(tmp_path)
    service.append_snapshot(
        _snapshot(
            agent_id="agt_deadline",
            attempt_id="attempt-deadline",
            created_at="2026-08-12T00:00:00Z",
        )
    )
    channel = _DeadlineChannel()
    deadline = time.monotonic() + 1.0
    try:
        result = _coordinator(service, channel).deliver_pending(
            deadline=deadline,
        )

        assert len(result.delivered_outbox_ids) == 1
        assert channel.deadlines == [deadline]
    finally:
        service.close()
        writer.close()


def test_expired_outbox_deadline_stops_before_channel_call(tmp_path) -> None:
    service, writer = _runtime(tmp_path)
    service.append_snapshot(
        _snapshot(
            agent_id="agt_expired",
            attempt_id="attempt-expired",
            created_at="2026-08-12T00:00:00Z",
        )
    )
    channel = _DeadlineChannel()
    try:
        with pytest.raises(EmployeeOutboxDrainDeadlineExceeded):
            _coordinator(service, channel).deliver_pending(
                deadline=time.monotonic() - 1.0,
            )

        assert channel.deadlines == []
    finally:
        service.close()
        writer.close()


def test_outbox_deadline_bounds_projection_lock_contention(tmp_path) -> None:
    service, writer = _runtime(tmp_path)
    channel = _DeadlineChannel()
    held = threading.Event()
    release = threading.Event()

    def hold_projection_lock() -> None:
        with service._mutex:  # noqa: SLF001 - deterministic fault injection
            held.set()
            assert release.wait(2)

    holder = threading.Thread(target=hold_projection_lock)
    holder.start()
    try:
        assert held.wait(1)
        started = time.monotonic()
        with pytest.raises(EmployeeOutboxDrainDeadlineExceeded):
            _coordinator(service, channel).deliver_pending(
                deadline=started + 0.03,
            )
        assert time.monotonic() - started < 0.5
        assert channel.deadlines == []
    finally:
        release.set()
        holder.join(timeout=1)
        service.close()
        writer.close()
