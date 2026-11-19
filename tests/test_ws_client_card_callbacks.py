import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.feishu.ws_client import FeishuWSClient
from src.tasking import TaskQueueFullError
from src.thread import set_current_thread_id
from src.trust.models import TrustZone
from tests.helpers.ws_routing_harness import (
    _card_action_data,
    _direct_card_action_data,
    _observe_card_thread_context,
    mock_ws_client,
)

__all__ = ["mock_ws_client"]


def test_card_callback_clears_inherited_thread_without_payload(
    mock_ws_client: FeishuWSClient,
):
    set_current_thread_id("om_stale_thread")
    try:
        observed = _observe_card_thread_context(
            mock_ws_client,
            _direct_card_action_data(chat_id="chat_456", project_id="proj_1"),
        )
    finally:
        set_current_thread_id(None)

    assert observed is None


def test_card_callback_restores_canonical_matching_thread(
    mock_ws_client: FeishuWSClient,
):
    mock_ws_client._thread_manager.register(
        "om_canonical_root",
        "chat_456",
        "proj_1",
        alias_keys=["om_callback_alias"],
    )
    try:
        observed = _observe_card_thread_context(
            mock_ws_client,
            _direct_card_action_data(
                chat_id="chat_456",
                project_id="proj_1",
                thread_root_id="om_callback_alias",
            ),
        )
    finally:
        mock_ws_client._thread_manager.remove("om_canonical_root")

    assert observed == "om_canonical_root"


@pytest.mark.parametrize(
    ("callback_chat_id", "callback_project_id"),
    [("chat_other", "proj_1"), ("chat_456", "proj_other")],
)
def test_card_callback_rejects_cross_coordinate_thread(
    mock_ws_client: FeishuWSClient,
    callback_chat_id: str,
    callback_project_id: str,
):
    mock_ws_client._thread_manager.register(
        "om_coordinate_root",
        "chat_456",
        "proj_1",
        alias_keys=["om_coordinate_alias"],
    )
    set_current_thread_id("om_stale_thread")
    try:
        observed = _observe_card_thread_context(
            mock_ws_client,
            _direct_card_action_data(
                chat_id=callback_chat_id,
                project_id=callback_project_id,
                thread_root_id="om_coordinate_alias",
            ),
        )
    finally:
        set_current_thread_id(None)
        mock_ws_client._thread_manager.remove("om_coordinate_root")

    assert observed is None


def test_card_action_rejects_old_callback_before_scheduling(
    mock_ws_client: FeishuWSClient,
) -> None:
    data = _card_action_data(
        event_id="evt_old_callback",
        message_id="msg_old_card",
        chat_id="chat_old",
        operator_id="ou_admin",
    )
    data.schema = "1.0"

    response = mock_ws_client._handle_card_action(data)

    assert response.__class__.__name__ == "P2CardActionTriggerResponse"
    mock_ws_client._scheduler.submit.assert_not_called()


def test_stale_managed_card_refresh_runs_only_in_system_follow_up(
    mock_ws_client: FeishuWSClient,
) -> None:
    trust = SimpleNamespace(
        zone=TrustZone.MANAGED_AGENT_GROUP,
        managed_group=SimpleNamespace(chat_id="chat_managed", project_id="project_1"),
        group_revision=7,
        grant_revision=11,
    )
    mock_ws_client._resolve_effective_trust = MagicMock(return_value=trust)
    mock_ws_client._managed_trust_access_decision = MagicMock(return_value=None)
    mock_ws_client._refresh_managed_card_revisions = MagicMock(return_value=True)
    submitted: list[tuple[object, object]] = []

    def submit(spec, callback):
        submitted.append((spec, callback))
        return SimpleNamespace(run_id="follow-up-1")

    mock_ws_client._scheduler.submit.side_effect = submit
    data = _card_action_data(
        event_id="evt_stale_managed",
        message_id="msg_stale_managed",
        chat_id="chat_managed",
        operator_id="ou_admin",
    )
    data.event.action.value.update({"group_revision": 6, "grant_revision": 11})

    response = mock_ws_client._handle_card_action_callback(data)

    assert response.__class__.__name__ == "P2CardActionTriggerResponse"
    mock_ws_client._refresh_managed_card_revisions.assert_not_called()
    assert submitted == []
    mock_ws_client._finish_card_advisories(True)
    assert len(submitted) == 1
    spec, callback = submitted[0]
    assert spec.name == "refresh_stale_managed_card"
    assert spec.task_type == "card_advisory_follow_up"
    assert spec.chat_id == "chat_managed"
    assert spec.is_system_command is True
    callback(SimpleNamespace())
    mock_ws_client._refresh_managed_card_revisions.assert_called_once_with(
        "msg_stale_managed", "chat_managed", trust
    )


def test_registered_card_callback_submits_advisory_only_after_inner_ack_returns(
    mock_ws_client: FeishuWSClient,
) -> None:
    """A real scheduler worker cannot race the stale refresh ahead of ACK creation."""
    from src.tasking import TaskScheduler

    trust = SimpleNamespace(
        zone=TrustZone.MANAGED_AGENT_GROUP,
        managed_group=SimpleNamespace(chat_id="chat_managed", project_id="project_1"),
        group_revision=7,
        grant_revision=11,
    )
    mock_ws_client._resolve_effective_trust = MagicMock(return_value=trust)
    mock_ws_client._managed_trust_access_decision = MagicMock(return_value=None)
    remote_started = threading.Event()

    def refresh(*_args):
        remote_started.set()
        return True

    mock_ws_client._refresh_managed_card_revisions = refresh
    real_scheduler = TaskScheduler(
        max_concurrent=1,
        system_concurrency=1,
        max_pending_normal=4,
        max_pending_system=4,
    )
    original_scheduler = mock_ws_client._scheduler
    mock_ws_client._scheduler = real_scheduler
    from lark_channel.ws.pb.pbbp2_pb2 import Frame

    payload = {
        "schema": "2.0",
        "header": {
            "event_id": "evt_stale_real_worker",
            "event_type": "card.action.trigger",
            "tenant_key": "tenant_card",
        },
        "event": {
            "context": {
                "open_message_id": "msg_stale_real_worker",
                "open_chat_id": "chat_managed",
            },
            "operator": {"open_id": "ou_admin"},
            "action": {
                "tag": "button",
                "value": {
                    "action": "show_status",
                    "group_revision": 6,
                    "grant_revision": 11,
                },
            },
        },
    }
    frame = Frame()
    frame.SeqID = 1
    frame.LogID = 1
    frame.service = 1
    frame.method = 1
    for key, value in (
        ("type", "card"),
        ("message_id", "msg-stale-real-worker"),
        ("trace_id", "trace-stale-real-worker"),
        ("sum", "1"),
        ("seq", "0"),
    ):
        header = frame.headers.add()
        header.key = key
        header.value = value
    frame.payload = json.dumps(payload).encode()

    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    observed = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
    observed._on_activity = lambda _kind: None
    observed._on_response_written = mock_ws_client._finish_card_advisories
    observed._event_handler = mock_ws_client._build_event_handler()
    observed._conn_id = ""

    try:
        async def exercise() -> None:
            write_started = asyncio.Event()
            release_write = asyncio.Event()

            async def write_message(_raw):
                write_started.set()
                await release_write.wait()

            observed._write_message = write_message
            callback_task = asyncio.create_task(observed._handle_data_frame(frame))
            await asyncio.wait_for(write_started.wait(), timeout=2)
            assert remote_started.is_set() is False
            release_write.set()
            await asyncio.wait_for(callback_task, timeout=2)
            deadline = asyncio.get_running_loop().time() + 2
            while not remote_started.is_set() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert remote_started.is_set()

        asyncio.run(exercise())
    finally:
        real_scheduler.stop(wait=True)
        mock_ws_client._scheduler = original_scheduler


@pytest.mark.parametrize(
    "admission_error",
    [
        TaskQueueFullError("system", 1),
        RuntimeError("TaskScheduler admission is fenced"),
    ],
    ids=["system-lane-full", "admission-fenced"],
)
def test_stale_managed_card_follow_up_rejection_never_refreshes_synchronously(
    mock_ws_client: FeishuWSClient,
    admission_error: Exception,
) -> None:
    trust = SimpleNamespace(
        zone=TrustZone.MANAGED_AGENT_GROUP,
        managed_group=SimpleNamespace(chat_id="chat_managed", project_id="project_1"),
        group_revision=7,
        grant_revision=11,
    )
    mock_ws_client._resolve_effective_trust = MagicMock(return_value=trust)
    mock_ws_client._managed_trust_access_decision = MagicMock(return_value=None)
    mock_ws_client._refresh_managed_card_revisions = MagicMock(return_value=True)
    mock_ws_client._scheduler.submit.side_effect = admission_error
    data = _card_action_data(
        event_id="evt_stale_full",
        message_id="msg_stale_full",
        chat_id="chat_managed",
        operator_id="ou_admin",
    )
    data.event.action.value.update({"group_revision": 6, "grant_revision": 11})

    response = mock_ws_client._handle_card_action_callback(data)

    assert response.__class__.__name__ == "P2CardActionTriggerResponse"
    mock_ws_client._refresh_managed_card_revisions.assert_not_called()
    mock_ws_client._finish_card_advisories(True)
    assert mock_ws_client._scheduler.submit.call_count == 1


def test_card_action_restores_p2p_from_trusted_message_origin(mock_ws_client: FeishuWSClient):
    """A DM selection flow must remain a DM after the callback hop."""
    mock_ws_client._message_linker.register_origin(
        "msg_origin",
        request_id="req_hire",
        chat_id="chat_dm",
        chat_type="p2p",
        sender_id="ou_admin",
    )
    mock_ws_client._message_linker.link_reply("msg_origin", "msg_card")
    data = _card_action_data(
        event_id="evt_hire_select",
        message_id="msg_card",
        chat_id="chat_dm",
        operator_id="ou_admin",
    )

    mock_ws_client._handle_card_action(data)

    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.origin_message_id == "msg_origin"
    assert spec.is_p2p is True


def test_card_action_after_restart_fails_closed_without_chat_api_lookup(
    mock_ws_client: FeishuWSClient,
):
    """The three-second callback path never performs a remote provenance lookup."""
    response = MagicMock()
    response.success.return_value = True
    response.data.chat_mode = "p2p"
    response.data.chat_type = "public"
    api_client = MagicMock()
    api_client.im.v1.chat.get.return_value = response
    mock_ws_client._get_api_client = MagicMock(return_value=api_client)

    data = _card_action_data(
        event_id="evt_after_restart",
        message_id="msg_card_after_restart",
        chat_id="chat_dm",
        operator_id="ou_admin",
    )
    mock_ws_client._handle_card_action(data)

    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.is_p2p is False
    api_client.im.v1.chat.get.assert_not_called()


def test_card_action_ignores_non_contract_callback_chat_type(mock_ws_client: FeishuWSClient):
    """An injected callback context.chat_type must not grant DM privileges."""
    response = MagicMock()
    response.success.return_value = True
    response.data.chat_mode = "group"
    response.data.chat_type = "private"
    api_client = MagicMock()
    api_client.im.v1.chat.get.return_value = response
    mock_ws_client._get_api_client = MagicMock(return_value=api_client)
    data = _card_action_data(
        event_id="evt_group",
        message_id="msg_group_card",
        chat_id="chat_group",
        operator_id="ou_admin",
    )
    data.event.context.chat_type = "p2p"

    mock_ws_client._handle_card_action(data)

    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.is_p2p is False


def test_card_action_cross_operator_provenance_fails_closed(mock_ws_client: FeishuWSClient):
    mock_ws_client._message_linker.register_origin(
        "msg_origin",
        request_id="req_hire",
        chat_id="chat_dm",
        chat_type="p2p",
        sender_id="ou_original_admin",
    )
    mock_ws_client._message_linker.link_reply("msg_origin", "msg_card")
    mock_ws_client._get_api_client = MagicMock()
    data = _card_action_data(
        event_id="evt_other_operator",
        message_id="msg_card",
        chat_id="chat_dm",
        operator_id="ou_other_admin",
    )

    mock_ws_client._handle_card_action(data)

    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.is_p2p is False
    mock_ws_client._get_api_client.assert_not_called()


def test_card_action_partial_origin_provenance_fails_closed(mock_ws_client: FeishuWSClient):
    mock_ws_client._message_linker.register_origin(
        "msg_partial",
        request_id="req_partial",
        chat_id="chat_dm",
    )
    mock_ws_client._get_api_client = MagicMock()

    assert mock_ws_client._resolve_card_is_p2p(
        origin_message_id="msg_partial",
        open_chat_id="chat_dm",
        operator_id="ou_admin",
    ) is False
    mock_ws_client._get_api_client.assert_not_called()


def test_card_action_origin_query_error_fails_closed(mock_ws_client: FeishuWSClient):
    mock_ws_client._message_linker.query = MagicMock(side_effect=RuntimeError("cache unavailable"))
    mock_ws_client._get_api_client = MagicMock()

    assert mock_ws_client._resolve_card_is_p2p(
        origin_message_id="msg_origin",
        open_chat_id="chat_dm",
        operator_id="ou_admin",
    ) is False
    mock_ws_client._get_api_client.assert_not_called()


def test_card_action_origin_resolution_error_does_not_become_api_miss(
    mock_ws_client: FeishuWSClient,
):
    mock_ws_client._message_linker.resolve_origin = MagicMock(
        side_effect=OSError("origin index unavailable")
    )
    mock_ws_client._get_api_client = MagicMock()
    data = _card_action_data(
        event_id="evt_origin_index_error",
        message_id="msg_card",
        chat_id="chat_dm",
        operator_id="ou_admin",
    )

    mock_ws_client._handle_card_action(data)

    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.is_p2p is False
    mock_ws_client._get_api_client.assert_not_called()


def test_card_action_api_fallback_rejects_empty_operator(mock_ws_client: FeishuWSClient):
    mock_ws_client._get_api_client = MagicMock()

    assert mock_ws_client._resolve_card_is_p2p(
        origin_message_id="msg_card",
        open_chat_id="chat_dm",
        operator_id="",
    ) is False
    mock_ws_client._get_api_client.assert_not_called()


def test_card_action_missing_origin_never_attempts_remote_provenance_write(
    mock_ws_client: FeishuWSClient,
):
    mock_ws_client._get_api_client = MagicMock()
    mock_ws_client._message_linker = MagicMock()
    mock_ws_client._message_linker.query.return_value = None
    mock_ws_client._message_linker.register_trusted_origin_if_absent.return_value = None

    assert mock_ws_client._resolve_card_is_p2p(
        origin_message_id="msg_card",
        open_chat_id="chat_dm",
        operator_id="ou_admin",
    ) is False
    mock_ws_client._get_api_client.assert_not_called()
    mock_ws_client._message_linker.register_trusted_origin_if_absent.assert_not_called()


def test_card_action_rejects_provenance_for_different_origin(
    mock_ws_client: FeishuWSClient,
):
    mock_ws_client._message_linker = MagicMock()
    mock_ws_client._message_linker.query.return_value = {
        "origin_message_id": "msg_other_origin",
        "chat_id": "chat_dm",
        "sender_id": "ou_admin",
        "chat_type": "p2p",
    }
    mock_ws_client._get_api_client = MagicMock()

    assert mock_ws_client._resolve_card_is_p2p(
        origin_message_id="msg_expected_origin",
        open_chat_id="chat_dm",
        operator_id="ou_admin",
    ) is False
    mock_ws_client._get_api_client.assert_not_called()


def test_rejected_cross_chat_callback_cannot_rewrite_trusted_provenance(
    mock_ws_client: FeishuWSClient,
):
    assert mock_ws_client._message_linker.register_trusted_origin_if_absent(
        "msg_origin",
        chat_id="chat_dm",
        sender_id="ou_admin",
        chat_type="p2p",
    ) is True
    mock_ws_client._message_linker.link_reply("msg_origin", "msg_card")
    mock_ws_client._get_api_client = MagicMock()

    for event_id in ("evt_cross_chat_1", "evt_cross_chat_2"):
        data = _card_action_data(
            event_id=event_id,
            message_id="msg_card",
            chat_id="chat_other",
            operator_id="ou_admin",
        )
        mock_ws_client._handle_card_action(data)
        spec, _ = mock_ws_client._scheduler.submit.call_args.args
        assert spec.is_p2p is False
        assert mock_ws_client._message_linker.query("msg_origin")["chat_id"] == "chat_dm"

    mock_ws_client._get_api_client.assert_not_called()
