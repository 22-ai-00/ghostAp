from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.card.ui_text import UI_TEXT
from src.feishu.message_cache import MessageCache
from src.feishu.ws_client import FeishuWSClient, _MessageIngressReservation
from src.feishu.ws_event_router import MessageIngressGuard
from src.tasking import TaskStatus


class _DeferredTerminalHandle:
    run_id = "run_prestart_terminal"

    def __init__(self) -> None:
        self.callback = None

    def add_done_callback(self, callback) -> None:
        self.callback = callback

    def fire(self, *, status: TaskStatus = TaskStatus.FAILED) -> None:
        assert self.callback is not None
        self.callback(
            SimpleNamespace(
                run_id=self.run_id,
                status=status,
            )
        )


class _ReplayedTerminalHandle:
    run_id = "run_prestart_terminal_replayed"

    def add_done_callback(self, callback) -> None:
        callback(
            SimpleNamespace(
                run_id=self.run_id,
                status=TaskStatus.CANCELED,
            )
        )


def _reservation(message_id: str):
    guard = MessageIngressGuard(
        message_cache=MessageCache(ttl=300, max_size=10),
        message_expire_seconds=30,
    )
    owner = guard.reserve(message_id)
    assert owner is not None
    return guard, _MessageIngressReservation(
        guard=guard,
        message_id=message_id,
        owner=owner,
    )


def _client(queue_main_bot_warning: MagicMock) -> FeishuWSClient:
    client = object.__new__(FeishuWSClient)
    client._employee_department_runtime = SimpleNamespace(
        queue_main_bot_warning=queue_main_bot_warning,
    )
    return client


def test_unstarted_terminal_prepares_durable_warning_and_fences_origin() -> None:
    message_id = "om_prestart_terminal"
    guard, reservation = _reservation(message_id)
    queue_warning = MagicMock(return_value=True)
    client = _client(queue_warning)
    handle = _DeferredTerminalHandle()

    client._bind_message_ingress_reservation(
        handle,
        reservation,
        tenant_key="tenant-a",
        chat_id="oc_group",
    )
    handle.fire()
    handle.fire()

    queue_warning.assert_called_once_with(
        tenant_key="tenant-a",
        chat_id="oc_group",
        message_id=message_id,
        text=UI_TEXT["ws_message_prestart_terminal"],
    )
    assert guard.reserve(message_id) is None


def test_terminal_before_bind_replay_prepares_warning_without_network_io() -> None:
    message_id = "om_prestart_terminal_replay"
    guard, reservation = _reservation(message_id)
    queue_warning = MagicMock(return_value=True)
    client = _client(queue_warning)

    client._bind_message_ingress_reservation(
        _ReplayedTerminalHandle(),
        reservation,
        tenant_key="tenant-a",
        chat_id="oc_group",
    )

    queue_warning.assert_called_once()
    assert guard.reserve(message_id) is None


@pytest.mark.parametrize(
    "prepare_result",
    (False, OSError("warning journal unavailable")),
    ids=("invalid-result", "exception"),
)
def test_unstarted_terminal_prepare_failure_releases_for_platform_retry(
    prepare_result: bool | Exception,
) -> None:
    message_id = "om_prestart_terminal_prepare_failed"
    guard, reservation = _reservation(message_id)
    queue_warning = MagicMock(
        side_effect=prepare_result if isinstance(prepare_result, Exception) else None,
        return_value=prepare_result if isinstance(prepare_result, bool) else True,
    )
    client = _client(queue_warning)
    handle = _DeferredTerminalHandle()

    client._bind_message_ingress_reservation(
        handle,
        reservation,
        tenant_key="tenant-a",
        chat_id="oc_group",
    )
    handle.fire()

    retry_owner = guard.reserve(message_id)
    assert retry_owner is not None
    assert guard.release(message_id, retry_owner)


def test_started_terminal_does_not_prepare_prestart_warning() -> None:
    message_id = "om_started_terminal"
    guard, reservation = _reservation(message_id)
    queue_warning = MagicMock(return_value=True)
    client = _client(queue_warning)
    handle = _DeferredTerminalHandle()
    reservation.mark_started()

    client._bind_message_ingress_reservation(
        handle,
        reservation,
        tenant_key="tenant-a",
        chat_id="oc_group",
    )
    handle.fire(status=TaskStatus.SUCCEEDED)

    queue_warning.assert_not_called()
    assert reservation.owns()
    assert reservation.release()
    assert guard.reserve(message_id) is not None
