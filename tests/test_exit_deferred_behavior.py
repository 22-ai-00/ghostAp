import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handlers.system import SystemHandler
from src.feishu.slash_command_parser import SlashCommandParser
from src.feishu.ws_client import FeishuWSClient
from src.mode import InteractionMode
from src.tasking import TaskSpec


@pytest.mark.parametrize(
    ("mode", "handler_name"),
    [
        (InteractionMode.COCO, "coco"),
        (InteractionMode.CLAUDE, "claude"),
        (InteractionMode.AIDEN, "aiden"),
        (InteractionMode.CODEX, "codex"),
        (InteractionMode.GEMINI, "gemini"),
        (InteractionMode.TRAEX, "traex"),
        (InteractionMode.GROK, "grok"),
        (InteractionMode.DSH, "dsh"),
    ],
)
def test_exit_current_mode_delegates_for_every_programming_backend(
    mode,
    handler_name,
):
    ctx = MagicMock()
    ctx.mode_manager.get_mode.return_value = mode
    programming_handlers = {
        name: MagicMock()
        for name in ("coco", "claude", "aiden", "codex", "gemini", "traex", "grok", "dsh")
    }
    ctx.handlers.get.side_effect = programming_handlers.get
    handler = SystemHandler(ctx)
    handler.reply_text = MagicMock()
    project = SimpleNamespace(project_id="project-1")

    with patch("src.thread.get_current_thread_id", return_value=None):
        handler.exit_current_mode("message-1", "chat-1", project)

    programming_handlers[handler_name].exit_mode.assert_called_once_with(
        "message-1",
        "chat-1",
        project,
    )
    for name, programming_handler in programming_handlers.items():
        if name != handler_name:
            programming_handler.exit_mode.assert_not_called()
    handler.reply_text.assert_not_called()


def test_exit_is_deferred_until_running_task_finishes():
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
        mock_settings.task_scheduler_max_concurrent = 1
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

        started = threading.Event()
        unblock = threading.Event()
        finished = threading.Event()
        finished_time = {"t": None}
        exit_time = {"t": None}

        def long_task(_ctx):
            started.set()
            unblock.wait(timeout=2)
            finished_time["t"] = time.time()
            finished.set()
            return "ok"

        def on_exit(*args, **kwargs):
            exit_time["t"] = time.time()
            return True

        exit_current_mode = MagicMock(side_effect=on_exit)
        system_handler = client._handler_ctx.handlers["system"]
        system_handler.exit_current_mode = exit_current_mode
        client._control_plane._exit_handler_fn = system_handler.exit_current_mode

        client._scheduler.submit(
            TaskSpec(chat_id="chat", name="normal", task_type="feishu_message", project_id="p1"),
            long_task,
        )
        assert started.wait(timeout=1)

        client._control_plane.request_deferred_exit(message_id="m_exit", chat_id="chat", project_id="p1")

        # Should not exit while task is still running
        time.sleep(0.1)
        exit_current_mode.assert_not_called()

        unblock.set()
        assert finished.wait(timeout=2)

        # Wait for control-plane thread to schedule deferred exit
        deadline = time.time() + 2
        while time.time() < deadline and not exit_current_mode.called:
            time.sleep(0.01)

        exit_current_mode.assert_called_once()
        assert finished.is_set()
        assert finished_time["t"] is not None
        assert exit_time["t"] is not None
        assert exit_time["t"] >= finished_time["t"]

        client.close()


def test_exit_is_immediate_when_no_running_task():
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
        mock_settings.task_scheduler_max_concurrent = 1
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

        def short_task(_ctx):
            return "done"

        completed = threading.Event()

        def mark_completed(event):
            if event.name == "short" and event.status.name == "SUCCEEDED":
                completed.set()

        client._scheduler.add_listener(mark_completed)
        client._scheduler.submit(
            TaskSpec(chat_id="chat", name="short", task_type="feishu_message", project_id="p1"),
            short_task,
        )
        assert completed.wait(timeout=2)

        assert client._control_plane.should_defer_exit(chat_id="chat", project_id="p1") is False

        exit_current_mode = MagicMock()
        client._handler_ctx.handlers["system"].exit_current_mode = exit_current_mode
        client._handler_ctx.handlers["coco"].add_reaction = MagicMock()
        client._get_effective_mode = MagicMock(
            return_value=(InteractionMode.COCO, True)
        )
        project = SimpleNamespace(project_id="p1")

        client._message_dispatcher.process_with_intent(
            "m_exit",
            "chat",
            "/exit",
            project,
            command_match=SlashCommandParser.parse("/exit"),
        )

        exit_current_mode.assert_called_once_with(
            "m_exit", "chat", project=project
        )

        client.close()
