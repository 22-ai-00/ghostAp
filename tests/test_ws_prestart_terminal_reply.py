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


def _client(reply_warning: MagicMock) -> FeishuWSClient:
    client = object.__new__(FeishuWSClient)
    client._reply_text_with_bounded_retry = reply_warning
    return client


def test_unstarted_terminal_delivers_warning_and_fences_origin() -> None:
    message_id = "om_prestart_terminal"
    guard, reservation = _reservation(message_id)
    reply_warning = MagicMock(return_value=True)
    client = _client(reply_warning)
    handle = _DeferredTerminalHandle()

    client._bind_message_ingress_reservation(
        handle,
        reservation,
        tenant_key="tenant-a",
        chat_id="oc_group",
    )
    handle.fire()
    handle.fire()

    reply_warning.assert_called_once_with(
        message_id,
        UI_TEXT["ws_message_prestart_terminal"],
        label="pre-start terminal warning",
    )
    assert guard.reserve(message_id) is None


def test_terminal_before_bind_replay_delivers_warning() -> None:
    message_id = "om_prestart_terminal_replay"
    guard, reservation = _reservation(message_id)
    reply_warning = MagicMock(return_value=True)
    client = _client(reply_warning)

    client._bind_message_ingress_reservation(
        _ReplayedTerminalHandle(),
        reservation,
        tenant_key="tenant-a",
        chat_id="oc_group",
    )

    reply_warning.assert_called_once_with(
        message_id,
        UI_TEXT["ws_message_prestart_terminal"],
        label="pre-start terminal warning",
    )
    assert guard.reserve(message_id) is None


@pytest.mark.parametrize(
    "prepare_result",
    (False, OSError("warning delivery unavailable")),
    ids=("invalid-result", "exception"),
)
def test_unstarted_terminal_delivery_failure_releases_for_platform_retry(
    prepare_result: bool | Exception,
) -> None:
    message_id = "om_prestart_terminal_prepare_failed"
    guard, reservation = _reservation(message_id)
    reply_warning = MagicMock(
        side_effect=prepare_result if isinstance(prepare_result, Exception) else None,
        return_value=prepare_result if isinstance(prepare_result, bool) else True,
    )
    client = _client(reply_warning)
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


def test_started_terminal_does_not_deliver_prestart_warning() -> None:
    message_id = "om_started_terminal"
    guard, reservation = _reservation(message_id)
    reply_warning = MagicMock(return_value=True)
    client = _client(reply_warning)
    handle = _DeferredTerminalHandle()
    reservation.mark_started()

    client._bind_message_ingress_reservation(
        handle,
        reservation,
        tenant_key="tenant-a",
        chat_id="oc_group",
    )
    handle.fire(status=TaskStatus.SUCCEEDED)

    reply_warning.assert_not_called()
    assert reservation.owns()
    assert reservation.release()
    assert guard.reserve(message_id) is not None
