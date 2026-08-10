from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.feishu.ws_client import FeishuWSClient
from src.tasking import TaskStatus


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
        reply_text = MagicMock()
        client._handler_ctx.handlers["coco"].reply_text = reply_text

        client._control_plane.on_scheduler_event(
            SimpleNamespace(
                task_type="system_help",
                status=TaskStatus.RUNNING,
                chat_id="oc_1",
                run_id="run_system_1",
                project_id=None,
            )
        )
        assert client._control_plane.is_system_cmd_inflight("oc_1") is True

        data = _make_card_action_data(open_message_id="om_1", open_chat_id="oc_1", action="enter_coco")
        client._handle_card_action(data)

        reply_text.assert_called_once()
        args, _ = reply_text.call_args
        assert args[0] == "om_1"
        assert "系统指令处理中" in args[1]

        client._scheduler.submit.assert_not_called()

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
        reply_text = MagicMock()
        client._handler_ctx.handlers["coco"].reply_text = reply_text
        client._get_streaming_manager = MagicMock(
            return_value=SimpleNamespace(get_card=lambda _mid: None, set_sticky_message=lambda *_a, **_k: None)
        )

        data = _make_card_action_data(open_message_id="om_1", open_chat_id="oc_1", action="enter_coco")
        client._handle_card_action(data)
        client._handle_card_action(data)

        # first click -> submit once; second click within TTL -> ignored
        assert client._scheduler.submit.call_count == 1
        assert reply_text.call_count == 0

        client.close()
