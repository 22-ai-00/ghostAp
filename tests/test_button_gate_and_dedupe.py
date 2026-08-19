import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.feishu.ws_client import FeishuWSClient
from src.tasking import TaskSpec


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
        schema="2.0",
        header=SimpleNamespace(
            event_id=f"evt_{open_message_id}",
            event_type="card.action.trigger",
            tenant_key="tenant_test",
        ),
        event=SimpleNamespace(
            action=SimpleNamespace(value=value, tag="button", name=action),
            operator=SimpleNamespace(
                open_id="ou_x",
                user_id="u_x",
                union_id=None,
            ),
            context=SimpleNamespace(open_message_id=open_message_id, open_chat_id=open_chat_id),
        )
    )


def test_button_is_blocked_while_system_command_inflight(tmp_path):
    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client._CHECKOUT_ROOT", tmp_path),
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
        reply_text = MagicMock()
        client._handler_ctx.handlers["coco"].reply_text = reply_text
        system_started = threading.Event()
        release_system = threading.Event()
        original_submit = client._scheduler.submit
        try:
            original_submit(
                TaskSpec(
                    chat_id="oc_1",
                    name="system-help",
                    task_type="system_help",
                    is_system_command=True,
                ),
                lambda _ctx: (
                    system_started.set(),
                    release_system.wait(timeout=3),
                ),
            )
            assert system_started.wait(timeout=1)
            assert client._control_plane.is_system_cmd_inflight("oc_1") is True

            data = _make_card_action_data(
                open_message_id="om_1",
                open_chat_id="oc_1",
                action="enter_coco",
            )
            submitted: list[tuple[object, object]] = []

            def submit(spec, callback):
                submitted.append((spec, callback))
                return SimpleNamespace(run_id="gate-follow-up")

            client._scheduler.submit = MagicMock(side_effect=submit)
            response = client._handle_card_action_callback(data)

            assert response.__class__.__name__ == "P2CardActionTriggerResponse"
            reply_text.assert_not_called()
            assert submitted == []
            client._finish_card_advisories(True)
            assert len(submitted) == 1
            spec, callback = submitted[0]
            assert spec.name == "notify_system_command_gate"
            assert spec.task_type == "card_advisory_follow_up"
            assert spec.chat_id == "oc_1"
            assert spec.is_system_command is True

            callback(SimpleNamespace())
            reply_text.assert_called_once()
            args, _ = reply_text.call_args
            assert args[0] == "om_1"
            assert "系统指令处理中" in args[1]
        finally:
            client._scheduler.submit = original_submit
            release_system.set()
            client.close()


def test_system_gate_lookup_failure_acks_and_never_schedules_business_action(tmp_path):
    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client._CHECKOUT_ROOT", tmp_path),
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
        client._control_plane.is_system_cmd_inflight = MagicMock(
            side_effect=RuntimeError("gate unavailable")
        )
        data = _make_card_action_data(
            open_message_id="om_gate_failure",
            open_chat_id="oc_gate_failure",
            action="enter_coco",
        )

        response = client._handle_card_action(data)

        assert response.__class__.__name__ == "P2CardActionTriggerResponse"
        client._scheduler.submit.assert_not_called()
        client.close()



def test_button_rapid_clicks_dedupe_same_state_and_accept_changed_state(tmp_path):
    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client._CHECKOUT_ROOT", tmp_path),
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

        data = _make_card_action_data(
            open_message_id="om_1",
            open_chat_id="oc_1",
            action="workflow_add_agent",
            value_extra={"_selection_sig": "draft-a"},
        )
        first = client._handle_card_action(data)
        data.header.event_id = "evt_rapid_click_2"
        duplicate = client._handle_card_action(data)

        changed = _make_card_action_data(
            open_message_id="om_1",
            open_chat_id="oc_1",
            action="workflow_add_agent",
            value_extra={"_selection_sig": "draft-b"},
        )
        changed.header.event_id = "evt_changed_selection"
        changed_result = client._handle_card_action(changed)

        # Same-state replay is ignored; a callback from a newly rendered
        # selection state is a distinct intent and must still be submitted.
        assert client._scheduler.submit.call_count == 2
        assert reply_text.call_count == 0
        assert first.toast.type == "info"
        assert first.toast.content == "正在应用选择，卡片将自动更新"
        assert duplicate.toast.type == "info"
        assert duplicate.toast.content == "操作正在处理中，请稍候"
        assert changed_result.toast.type == "info"
        assert changed_result.toast.content == "正在应用选择，卡片将自动更新"

        client.close()
