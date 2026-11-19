"""Shared fixtures and builders for WebSocket client routing tests."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.ws_client import FeishuWSClient
from src.thread import get_current_thread_id

__all__ = [
    "_card_action_data",
    "_direct_card_action_data",
    "_mentioned_group_command",
    "_observe_card_thread_context",
    "_pending_task_handle",
    "_sdk_message_mention",
    "create_mock_message",
    "mock_ws_client",
]


@pytest.fixture
def mock_ws_client():
    """Build the real routing graph without external sessions or worker threads."""
    scheduler = MagicMock()
    scheduler._restart_gate = MagicMock()

    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client._build_task_scheduler", return_value=scheduler),
        patch("src.feishu.ws_client.ACPSessionManager"),
        patch("src.feishu.ws_client.IntentRecognizer"),
        patch("src.feishu.ws_client.ProjectManager"),
        patch("src.feishu.ws_client.MessageProjectMapper"),
        patch("src.feishu.ws_client.DeepEngineManager"),
        patch("src.feishu.ws_client.ProgressReporter"),
        patch("src.feishu.ws_client.SpecEngineManager"),
        patch("src.feishu.ws_client.SpecReporter"),
        patch("src.mode.ModeManager"),
        patch("src.workflow_engine.manager.WorkflowEngineManager"),
    ):
        settings = MagicMock()
        settings.app_id = "test_app_id"
        settings.app_secret = "test_app_secret"
        settings.admin_user_ids = ""
        settings.allowed_user_ids = ""
        settings.allowed_chat_ids = ""
        settings.ingress_access_mode = "legacy_allow_all"
        settings.admin_bootstrap_scope = "p2p_only"
        settings.streaming_enabled = False
        settings.task_scheduler_max_concurrent = 2
        settings.task_scheduler_per_key_concurrency = 1
        settings.system_command_concurrency = 10
        settings.message_cache_ttl = 300
        settings.message_cache_max_size = 1000
        settings.message_expire_seconds = 30
        settings.card.action_dedup_ttl = 1
        settings.card.action_dedup_max_size = 5000
        settings.spec_rate_limit_capacity = 100
        settings.spec_rate_limit_fill_rate = 50.0
        settings.spec_circuit_breaker_threshold = 10
        settings.spec_circuit_breaker_recovery = 5.0
        mock_get_settings.return_value = settings

        client = FeishuWSClient(MagicMock())
        client._main_bot_open_id = "ou_main_bot"
        client._project_manager.get_active_project.return_value = None
        try:
            yield client
        finally:
            client.close()


def create_mock_message(text: str, message_id="om_123", chat_id="oc_456", message_type="text"):
    data = MagicMock()
    data.header.tenant_key = "tenant_test"
    data.event.message.message_id = message_id
    data.event.message.chat_id = chat_id
    data.event.message.content = json.dumps({"text": text})
    data.event.message.message_type = message_type
    data.event.message.create_time = str(int(time.time() * 1000))
    # Reset parent/root
    data.event.message.parent_id = None
    data.event.message.root_id = None
    data.event.message.thread_id = None
    data.event.message.chat_type = "group"
    data.event.sender.sender_id.open_id = "ou_test"
    data.event.sender.sender_id.union_id = "on_test"
    return data


def _sdk_message_mention(
    *,
    key: str,
    open_id: str,
    name: str,
    tenant_key: str = "tenant_test",
):
    """Build the exact mention shape emitted by lark-channel-sdk 1.2.0."""

    from lark_channel.api.im.v1.model.mention_event import MentionEvent
    from lark_channel.api.im.v1.model.user_id import UserId

    return (
        MentionEvent.builder()
        .key(key)
        .id(UserId.builder().open_id(open_id).build())
        .name(name)
        .tenant_key(tenant_key)
        .build()
    )


def _mentioned_group_command(
    text: str,
    *mentions,
    message_id: str = "om_mentioned_command",
):
    message = create_mock_message(text, message_id=message_id)
    message.event.message.mentions = tuple(mentions)
    return message


def _pending_task_handle(run_id: str) -> SimpleNamespace:
    return SimpleNamespace(run_id=run_id, add_done_callback=MagicMock())


def _direct_card_action_data(
    *,
    chat_id: str,
    project_id: str,
    thread_root_id: str | None = None,
):
    value = {"action": "observe_thread_context", "project_id": project_id}
    if thread_root_id is not None:
        value["thread_root_id"] = thread_root_id
    return SimpleNamespace(
        event=SimpleNamespace(
            action=SimpleNamespace(
                value=value,
                behaviors=None,
                option=None,
                options=None,
                form_value=None,
                input_value=None,
            ),
            operator=SimpleNamespace(
                open_id="ou_card_operator",
                user_id=None,
                union_id=None,
            ),
            context=SimpleNamespace(
                open_message_id="om_card_callback",
                open_chat_id=chat_id,
                chat_type="group",
            ),
        )
    )


def _observe_card_thread_context(
    mock_ws_client: FeishuWSClient,
    data,
) -> str | None:
    observed: list[str | None] = []
    mock_ws_client._resolve_effective_trust = MagicMock(return_value=None)
    mock_ws_client._current_trust_can_dispatch = MagicMock(return_value=True)
    mock_ws_client._chat_lock_gate.check_card_action = MagicMock(return_value=False)
    mock_ws_client._action_handlers["observe_thread_context"] = (
        lambda *_args: observed.append(get_current_thread_id())
    )
    mock_ws_client._process_card_action_async(data)
    assert len(observed) == 1
    return observed[0]


def _card_action_data(*, event_id: str, message_id: str, chat_id: str, operator_id: str):
    data = MagicMock()
    data.schema = "2.0"
    data.header.event_id = event_id
    data.header.event_type = "card.action.trigger"
    data.header.tenant_key = "tenant_card"
    data.event.context.open_message_id = message_id
    data.event.context.open_chat_id = chat_id
    # The official card callback context has no chat_type field.
    del data.event.context.chat_type
    data.event.action.tag = "button"
    data.event.action.name = ""
    data.event.action.value = {"action": "show_status"}
    data.event.operator.open_id = operator_id
    data.event.operator.user_id = None
    data.event.operator.union_id = None
    return data
