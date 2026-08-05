from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.feishu.ws_client import FeishuWSClient


def _make_card_action_data(
    *,
    open_message_id: str,
    open_chat_id: str,
    action: str,
    project_id: str = "p1",
    value_extra: dict | None = None,
):
    value = {"action": action, "project_id": project_id}
    if value_extra:
        value.update(value_extra)
    return SimpleNamespace(
        event=SimpleNamespace(
            action=SimpleNamespace(value=value, tag="button", name=action),
            operator=SimpleNamespace(open_id="ou_x", user_id="u_x"),
            context=SimpleNamespace(open_message_id=open_message_id, open_chat_id=open_chat_id),
        )
    )


def test_button_is_blocked_while_system_command_inflight():
    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client.ACPSessionManager"),
        patch("src.feishu.ws_client.IntentRecognizer"),
        patch("src.feishu.ws_client.ProjectManager"),
        patch("src.feishu.ws_client.MessageProjectMapper"),
        patch("src.feishu.ws_client.DeepEngineManager"),
        patch("src.feishu.ws_client.ProgressReporter"),
        patch("src.mode.ModeManager"),
    ):
        mock_settings = MagicMock()
        mock_settings.app_id = "test_app_id"
        mock_settings.app_secret = "test_app_secret"
        mock_settings.streaming_enabled = False
        mock_settings.task_scheduler_max_concurrent = 2
        mock_settings.task_scheduler_per_key_concurrency = 1
        mock_settings.message_cache_ttl = 300
        mock_settings.message_cache_max_size = 1000
        mock_settings.card.action_dedup_ttl = 1
        mock_settings.card.action_dedup_max_size = 5000
        mock_settings.system_command_concurrency = 10
        mock_settings.spec_rate_limit_capacity = 100
        mock_settings.spec_rate_limit_fill_rate = 50.0
        mock_settings.spec_circuit_breaker_threshold = 10
        mock_settings.spec_circuit_breaker_recovery = 5.0
        mock_settings.message_expire_seconds = 30
        mock_settings.autonomous_visible_employee_limit = 0
        mock_get_settings.return_value = mock_settings

        client = FeishuWSClient(MagicMock())
        client._scheduler = MagicMock()
        client._reply_text = MagicMock()

        with client._system_cmd_gate_lock:
            client._system_cmd_inflight_by_chat["oc_1"] = 1

        data = _make_card_action_data(open_message_id="om_1", open_chat_id="oc_1", action="enter_coco")
        client._handle_card_action(data)

        client._reply_text.assert_called_once()
        args, _ = client._reply_text.call_args
        assert args[0] == "om_1"
        assert "系统指令处理中" in args[1]

        client._scheduler.submit.assert_not_called()

        client.close()


def test_same_spec_review_action_with_different_tool_payloads_is_not_deduped():
    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client.ACPSessionManager"),
        patch("src.feishu.ws_client.IntentRecognizer"),
        patch("src.feishu.ws_client.ProjectManager"),
        patch("src.feishu.ws_client.MessageProjectMapper"),
        patch("src.feishu.ws_client.DeepEngineManager"),
        patch("src.feishu.ws_client.ProgressReporter"),
        patch("src.mode.ModeManager"),
    ):
        mock_settings = MagicMock()
        mock_settings.app_id = "test_app_id"
        mock_settings.app_secret = "test_app_secret"
        mock_settings.streaming_enabled = False
        mock_settings.task_scheduler_max_concurrent = 2
        mock_settings.task_scheduler_per_key_concurrency = 1
        mock_settings.message_cache_ttl = 300
        mock_settings.message_cache_max_size = 1000
        mock_settings.card.action_dedup_ttl = 1
        mock_settings.card.action_dedup_max_size = 5000
        mock_settings.system_command_concurrency = 10
        mock_settings.spec_rate_limit_capacity = 100
        mock_settings.spec_rate_limit_fill_rate = 50.0
        mock_settings.spec_circuit_breaker_threshold = 10
        mock_settings.spec_circuit_breaker_recovery = 5.0
        mock_settings.message_expire_seconds = 30
        mock_settings.autonomous_visible_employee_limit = 0
        mock_get_settings.return_value = mock_settings

        client = FeishuWSClient(MagicMock())
        client._scheduler = MagicMock()
        client._reply_text = MagicMock()
        client._get_streaming_manager = MagicMock(
            return_value=SimpleNamespace(get_card=lambda _mid: None, set_sticky_message=lambda *_a, **_k: None)
        )

        aiden = _make_card_action_data(
            open_message_id="om_1",
            open_chat_id="oc_1",
            action="spec_review_select_tool",
            value_extra={"tool_name": "aiden", "provider": "acp"},
        )
        coco = _make_card_action_data(
            open_message_id="om_1",
            open_chat_id="oc_1",
            action="spec_review_select_tool",
            value_extra={"tool_name": "coco", "provider": "acp"},
        )

        client._handle_card_action(aiden)
        second_result = client._handle_card_action(coco)
        duplicate_result = client._handle_card_action(coco)

        assert client._scheduler.submit.call_count == 2
        assert second_result is None
        assert duplicate_result == {}

        client.close()


def test_same_spec_review_tool_after_selection_change_is_not_deduped():
    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client.ACPSessionManager"),
        patch("src.feishu.ws_client.IntentRecognizer"),
        patch("src.feishu.ws_client.ProjectManager"),
        patch("src.feishu.ws_client.MessageProjectMapper"),
        patch("src.feishu.ws_client.DeepEngineManager"),
        patch("src.feishu.ws_client.ProgressReporter"),
        patch("src.mode.ModeManager"),
    ):
        mock_settings = MagicMock()
        mock_settings.app_id = "test_app_id"
        mock_settings.app_secret = "test_app_secret"
        mock_settings.streaming_enabled = False
        mock_settings.task_scheduler_max_concurrent = 2
        mock_settings.task_scheduler_per_key_concurrency = 1
        mock_settings.message_cache_ttl = 300
        mock_settings.message_cache_max_size = 1000
        mock_settings.card.action_dedup_ttl = 1
        mock_settings.card.action_dedup_max_size = 5000
        mock_settings.system_command_concurrency = 10
        mock_settings.spec_rate_limit_capacity = 100
        mock_settings.spec_rate_limit_fill_rate = 50.0
        mock_settings.spec_circuit_breaker_threshold = 10
        mock_settings.spec_circuit_breaker_recovery = 5.0
        mock_settings.message_expire_seconds = 30
        mock_settings.autonomous_visible_employee_limit = 0
        mock_get_settings.return_value = mock_settings

        client = FeishuWSClient(MagicMock())
        client._scheduler = MagicMock()
        client._reply_text = MagicMock()
        client._get_streaming_manager = MagicMock(
            return_value=SimpleNamespace(get_card=lambda _mid: None, set_sticky_message=lambda *_a, **_k: None)
        )

        coco_before_selection = _make_card_action_data(
            open_message_id="om_1",
            open_chat_id="oc_1",
            action="spec_review_select_tool",
            value_extra={"tool_name": "coco", "provider": "acp", "_selection_sig": "empty"},
        )
        coco_after_selection = _make_card_action_data(
            open_message_id="om_1",
            open_chat_id="oc_1",
            action="spec_review_select_tool",
            value_extra={"tool_name": "coco", "provider": "acp", "_selection_sig": "coco-model-a"},
        )

        first_result = client._handle_card_action(coco_before_selection)
        second_result = client._handle_card_action(coco_after_selection)
        duplicate_result = client._handle_card_action(coco_after_selection)

        assert client._scheduler.submit.call_count == 2
        assert first_result is None
        assert second_result is None
        assert duplicate_result == {}

        client.close()


def test_button_rapid_clicks_are_deduped_and_only_one_task_is_submitted():
    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client.ACPSessionManager"),
        patch("src.feishu.ws_client.IntentRecognizer"),
        patch("src.feishu.ws_client.ProjectManager"),
        patch("src.feishu.ws_client.MessageProjectMapper"),
        patch("src.feishu.ws_client.DeepEngineManager"),
        patch("src.feishu.ws_client.ProgressReporter"),
        patch("src.mode.ModeManager"),
    ):
        mock_settings = MagicMock()
        mock_settings.app_id = "test_app_id"
        mock_settings.app_secret = "test_app_secret"
        mock_settings.streaming_enabled = False
        mock_settings.task_scheduler_max_concurrent = 2
        mock_settings.task_scheduler_per_key_concurrency = 1
        mock_settings.message_cache_ttl = 300
        mock_settings.message_cache_max_size = 1000
        mock_settings.card.action_dedup_ttl = 1
        mock_settings.card.action_dedup_max_size = 5000
        mock_settings.system_command_concurrency = 10
        mock_settings.spec_rate_limit_capacity = 100
        mock_settings.spec_rate_limit_fill_rate = 50.0
        mock_settings.spec_circuit_breaker_threshold = 10
        mock_settings.spec_circuit_breaker_recovery = 5.0
        mock_settings.message_expire_seconds = 30
        mock_settings.autonomous_visible_employee_limit = 0
        mock_get_settings.return_value = mock_settings

        client = FeishuWSClient(MagicMock())
        client._scheduler = MagicMock()
        client._reply_text = MagicMock()
        client._get_streaming_manager = MagicMock(
            return_value=SimpleNamespace(get_card=lambda _mid: None, set_sticky_message=lambda *_a, **_k: None)
        )

        data = _make_card_action_data(open_message_id="om_1", open_chat_id="oc_1", action="enter_coco")
        client._handle_card_action(data)
        client._handle_card_action(data)

        # first click -> submit once; second click within TTL -> ignored
        assert client._scheduler.submit.call_count == 1
        assert client._reply_text.call_count == 0

        client.close()
