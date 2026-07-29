import asyncio
import hashlib
import inspect
import threading
import time
from types import SimpleNamespace

import pytest

from src.feishu.ws_client import (
    _employee_hire_status_text,
    _employee_hire_status_uuid,
)


def test_pending_employee_notification_does_not_claim_ready() -> None:
    text = _employee_hire_status_text("Atlas", "ready")

    assert text is not None
    assert "配置完成，正在等待激活" in text
    assert "已就绪" not in text
    assert "/status" in text
    assert _employee_hire_status_text("Atlas", "active") == (
        "✅ 员工 **Atlas** 已激活，可以加入 Slock 群协作。"
    )


def test_employee_hire_status_uuid_is_stable_per_intent_and_status() -> None:
    expected = hashlib.sha256(
        b"employee-hire-status:hire-intent-1:active"
    ).hexdigest()[:50]

    assert _employee_hire_status_uuid("hire-intent-1", "active") == expected
    assert _employee_hire_status_uuid("hire-intent-1", "active") == expected
    assert _employee_hire_status_uuid("hire-intent-1", "ready") != expected


def test_ws_client_start_reconnects_if_underlying_start_returns(monkeypatch):
    """Ensure WS client doesn't stop the whole service when lark client exits.

    We simulate lark-channel WS client's `.start()` returning unexpectedly. GhostAP
    should reconnect (create client again) until `close()` is called.
    """

    from src.feishu import ws_client as ws

    fake_settings = SimpleNamespace(
        app_id="test_app_id",
        app_secret="test_secret",
        coco_session_timeout=60,
        claude_session_timeout=60,
        acp_keepalive_interval=10,
        acp_session_idle_healthcheck_s=0,
        task_scheduler_max_concurrent=1,
        task_scheduler_per_key_concurrency=1,
        message_cache_ttl=300,
        message_cache_max_size=1000,
        card=SimpleNamespace(action_dedup_ttl=1, action_dedup_max_size=5000),
        system_command_concurrency=10,
        spec_rate_limit_capacity=100,
        spec_rate_limit_fill_rate=50.0,
        spec_circuit_breaker_threshold=10,
        spec_circuit_breaker_recovery=5.0,
        message_expire_seconds=30,
        streaming_enabled=False,
        thread_programming_enabled=False,
        feishu_ws_reconnect_delay_s=0.02,
        feishu_ws_watchdog_interval=999,
    )
    monkeypatch.setattr(ws, "get_settings", lambda: fake_settings)

    created = []

    class DummyClient:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)

        def start(self):
            # Simulate immediate exit (disconnect / internal error).
            time.sleep(0.01)

        async def _disconnect(self):
            return None

    # Avoid background watchdog behavior in this unit test.
    monkeypatch.setattr(ws, "ObservedLarkWSClient", DummyClient)
    from src.feishu.ws_health import WSHealthMonitor
    monkeypatch.setattr(WSHealthMonitor, "start_watchdog", lambda self: None)
    monkeypatch.setattr(WSHealthMonitor, "stop_watchdog", lambda self: None)

    # Avoid starting extra cache threads here; close() logic is covered elsewhere.
    monkeypatch.setattr(ws.MessageCache, "start_cleanup_thread", lambda self: None)
    monkeypatch.setattr(ws.MessageCache, "stop_cleanup_thread", lambda self: None)
    monkeypatch.setattr(
        ws.SlockEngineManager,
        "restore_from_disk",
        lambda self, root: 0,
    )
    slash_sync_starts = []
    monkeypatch.setattr(
        ws.FeishuWSClient,
        "_start_main_slash_command_sync",
        lambda self: slash_sync_starts.append(True),
        raising=False,
    )

    client = ws.FeishuWSClient(message_callback=lambda *a, **k: None)
    membership_audits = []
    client._employee_department_runtime = SimpleNamespace(
        membership_service=SimpleNamespace(
            reconcile_projected_memberships=lambda: membership_audits.append(True)
        ),
        close=lambda: None,
    )

    t = threading.Thread(target=client.start, daemon=True)
    t.start()

    # Wait until at least one reconnect attempt happens.
    deadline = time.time() + 1.0
    while time.time() < deadline and len(created) < 2:
        time.sleep(0.01)

    client.close()
    t.join(timeout=1.0)
    assert len(created) >= 2
    assert membership_audits == [True]
    assert slash_sync_starts == [True]
    assert all(item["log_level"] == ws.ChannelLogLevel.WARNING for item in created)
    assert all(item["source"] == "ghostap" for item in created)
    assert not t.is_alive()


# ---------------------------------------------------------------------------
# WS lifecycle tests (merged from test_ws_lifecycle.py)
# ---------------------------------------------------------------------------


def test_ws_lifecycle_helpers_are_extracted_from_ws_client():
    from src.feishu.ws_lifecycle import ObservedLarkWSClient, frame_header_value

    frame = SimpleNamespace(
        headers=[
            SimpleNamespace(key="irrelevant", value="x"),
            SimpleNamespace(key="type", value="pong"),
        ]
    )

    assert ObservedLarkWSClient.__name__ == "ObservedLarkWSClient"
    assert frame_header_value(frame, "type") == "pong"
    assert frame_header_value(frame, "missing") is None


def test_observed_ws_client_uses_official_channel_sdk() -> None:
    from lark_channel.ws import Client as ChannelWSClient

    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    assert issubclass(ObservedLarkWSClient, ChannelWSClient)


def test_observed_ws_disconnect_forwards_expected_connection(monkeypatch) -> None:
    from lark_channel.ws import Client as ChannelWSClient

    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    expected = object()
    forwarded = []
    activity = []

    async def fake_disconnect(_self, *, expected_conn=None):
        forwarded.append(expected_conn)
        return True

    monkeypatch.setattr(ChannelWSClient, "_disconnect", fake_disconnect)
    client = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
    client._on_activity = activity.append

    result = asyncio.run(client._disconnect(expected_conn=expected))

    assert result is True
    assert forwarded == [expected]
    assert activity == ["disconnected"]


def test_observed_ws_disconnect_ignores_stale_expected_connection(monkeypatch) -> None:
    from lark_channel.ws import Client as ChannelWSClient

    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    async def stale_disconnect(_self, *, expected_conn=None):
        return False

    monkeypatch.setattr(ChannelWSClient, "_disconnect", stale_disconnect)
    activity = []
    client = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
    client._on_activity = activity.append

    assert asyncio.run(client._disconnect(expected_conn=object())) is False
    assert activity == []


def test_channel_client_factory_returns_one_outbound_webhook_client(monkeypatch) -> None:
    import src.feishu.ws_client as ws_client_module

    class _FakeChannel:
        def __init__(self, **kwargs) -> None:
            self.config = SimpleNamespace(transport=kwargs["transport"])

    monkeypatch.setattr(ws_client_module, "FeishuChannel", _FakeChannel)
    FeishuWSClient = ws_client_module.FeishuWSClient

    client = FeishuWSClient.__new__(FeishuWSClient)
    client.settings = SimpleNamespace(
        app_id="cli_test",
        app_secret="secret",
        card=SimpleNamespace(delivery_api_timeout=17.5),
    )
    client._channel_client = None
    client._channel_client_lock = threading.Lock()

    first = client._get_channel_client()
    second = client._get_channel_client()

    assert first is second
    assert first.config.transport.kind == "webhook"
    assert first.config.transport.http_timeout_seconds == 17.5


def test_main_ws_acknowledges_bot_deleted_events() -> None:
    from src.feishu.ws_client import FeishuWSClient

    source = inspect.getsource(FeishuWSClient.start)

    assert "register_p2_im_chat_member_bot_deleted_v1" in source


def test_main_bot_deleted_event_retires_persisted_slock_marker() -> None:
    from src.feishu.ws_client import FeishuWSClient

    calls: list[tuple[str, ...]] = []

    class _Manager:
        def retire_deleted_chat(self, chat_id: str):
            calls.append(("retire", chat_id))

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._slock_engine_manager = _Manager()

    client._handle_bot_deleted(
        SimpleNamespace(event=SimpleNamespace(chat_id="oc_deleted"))
    )

    assert calls == [("retire", "oc_deleted")]


def test_main_bot_deleted_event_retires_marker_without_loaded_engine() -> None:
    from src.feishu.ws_client import FeishuWSClient

    calls: list[tuple[str, ...]] = []

    class _Manager:
        def retire_deleted_chat(self, chat_id: str):
            calls.append(("retire", chat_id))

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._slock_engine_manager = _Manager()

    client._handle_bot_deleted(
        SimpleNamespace(event=SimpleNamespace(chat_id="oc_marker_only"))
    )

    assert calls == [("retire", "oc_marker_only")]


def test_main_bot_deleted_event_propagates_marker_archive_failure() -> None:
    from src.feishu.ws_client import FeishuWSClient

    class _Manager:
        def retire_deleted_chat(self, _chat_id: str):
            raise OSError("disk unavailable")

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._slock_engine_manager = _Manager()

    with pytest.raises(OSError, match="disk unavailable"):
        client._handle_bot_deleted(
            SimpleNamespace(event=SimpleNamespace(chat_id="oc_archive_failure"))
        )


def test_recovered_hire_notification_restores_trusted_recipient_scope() -> None:
    from unittest.mock import MagicMock

    from src.feishu.ws_client import FeishuWSClient
    from src.project.mapper import MessageLinker
    from src.thread.manager import get_current_tenant_key

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._message_linker = MessageLinker()
    client._get_chat_mode = MagicMock(return_value="p2p")
    client._reply_text = MagicMock(return_value="om_status")
    state = SimpleNamespace(
        intent_id="hire_1",
        tenant_key="tenant-a",
        message_id="om_hire",
        chat_id="oc_requester_dm",
        requester_principal_id="ou_requester",
        employee_name="Atlas",
    )

    assert client._reply_employee_hire_status(state, "ready") == "om_status"

    origin = client._message_linker.query("om_hire")
    assert origin is not None
    assert origin["chat_id"] == "oc_requester_dm"
    assert origin["sender_id"] == "ou_requester"
    assert origin["chat_type"] == "p2p"
    assert origin["tenant_key"] == "tenant-a"
    client._reply_text.assert_called_once()
    assert get_current_tenant_key() is None


def test_recovered_hire_notification_rejects_conflicting_recipient_scope() -> None:
    from unittest.mock import MagicMock

    from src.feishu.ws_client import FeishuWSClient
    from src.project.mapper import MessageLinker

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._message_linker = MessageLinker()
    client._message_linker.register_trusted_origin_if_absent(
        "om_hire",
        chat_id="oc_other",
        sender_id="ou_other",
        chat_type="p2p",
    )
    client._get_chat_mode = MagicMock(return_value="p2p")
    client._reply_text = MagicMock()
    state = SimpleNamespace(
        intent_id="hire_1",
        tenant_key="tenant-a",
        message_id="om_hire",
        chat_id="oc_requester_dm",
        requester_principal_id="ou_requester",
        employee_name="Atlas",
    )

    assert client._reply_employee_hire_status(state, "ready") is None
    client._reply_text.assert_not_called()
    client._get_chat_mode.assert_not_called()


def test_recovered_team_notification_restores_trusted_recipient_scope() -> None:
    from unittest.mock import MagicMock

    from src.feishu.ws_client import FeishuWSClient
    from src.project.mapper import MessageLinker
    from src.thread.manager import get_current_tenant_key

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._message_linker = MessageLinker()
    client._get_chat_mode = MagicMock(return_value="group")
    client._reply_text = MagicMock(return_value="om_team_result")

    assert client._reply_employee_team_message(
        "om_team_task",
        "oc_team",
        "团队任务已完成",
        tenant_key="tenant-a",
        requester_principal_id="ou_requester",
        idempotency_key="team-notify-key",
    ) == "om_team_result"

    origin = client._message_linker.query("om_team_task")
    assert origin is not None
    assert origin["chat_id"] == "oc_team"
    assert origin["sender_id"] == "ou_requester"
    assert origin["chat_type"] == "group"
    assert origin["tenant_key"] == "tenant-a"
    client._reply_text.assert_called_once_with(
        "om_team_task",
        "团队任务已完成",
        idempotency_key="team-notify-key",
    )
    assert get_current_tenant_key() is None


def test_recovered_team_notification_transport_failure_is_retryable() -> None:
    from unittest.mock import MagicMock

    from src.feishu.ws_client import FeishuWSClient
    from src.project.mapper import MessageLinker
    from src.thread.manager import get_current_tenant_key

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._message_linker = MessageLinker()
    client._get_chat_mode = MagicMock(return_value="group")
    client._reply_text = MagicMock(return_value=None)

    with pytest.raises(RuntimeError, match="delivery failed"):
        client._reply_employee_team_message(
            "om_team_task",
            "oc_team",
            "团队任务已完成",
            tenant_key="tenant-a",
            requester_principal_id="ou_requester",
            idempotency_key="team-notify-key",
        )

    assert get_current_tenant_key() is None


def test_recovered_team_notification_scope_conflict_is_not_committed() -> None:
    from unittest.mock import MagicMock

    from src.feishu.ws_client import FeishuWSClient
    from src.project.mapper import MessageLinker

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._message_linker = MessageLinker()
    client._message_linker.register_trusted_origin_if_absent(
        "om_team_task",
        chat_id="oc_other",
        sender_id="ou_other",
        chat_type="group",
    )
    client._get_chat_mode = MagicMock(return_value="group")
    client._reply_text = MagicMock()

    with pytest.raises(RuntimeError, match="recipient scope"):
        client._reply_employee_team_message(
            "om_team_task",
            "oc_team",
            "团队任务已完成",
            tenant_key="tenant-a",
            requester_principal_id="ou_requester",
            idempotency_key="team-notify-key",
        )

    client._reply_text.assert_not_called()
    client._get_chat_mode.assert_not_called()


def test_employee_runtime_recovery_requires_bound_reply_transport() -> None:
    from unittest.mock import MagicMock

    from src.feishu.ws_client import FeishuWSClient

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._employee_department_runtime = SimpleNamespace(
        recover=MagicMock(),
        close=MagicMock(),
    )

    with pytest.raises(RuntimeError, match="reply transport"):
        client._recover_employee_runtime_after_handler_binding()
    client._employee_department_runtime.recover.assert_not_called()
    client._employee_department_runtime.close.assert_called_once_with()

    client._reply_text = MagicMock()
    client._employee_department_runtime.close.reset_mock()
    client._recover_employee_runtime_after_handler_binding()

    client._employee_department_runtime.recover.assert_called_once_with()
    client._employee_department_runtime.close.assert_not_called()


def test_employee_runtime_recovery_failure_closes_runtime_once() -> None:
    from unittest.mock import MagicMock

    from src.feishu.ws_client import FeishuWSClient

    runtime = SimpleNamespace(
        recover=MagicMock(side_effect=RuntimeError("journal replay failed")),
        close=MagicMock(),
    )
    client = FeishuWSClient.__new__(FeishuWSClient)
    client._employee_department_runtime = runtime
    client._reply_text = MagicMock()

    with pytest.raises(RuntimeError, match="journal replay failed"):
        client._recover_employee_runtime_after_handler_binding()

    runtime.close.assert_called_once_with()


def test_ws_client_init_failure_after_employee_runtime_composition_closes_once(
    monkeypatch,
) -> None:
    from unittest.mock import MagicMock

    from src.autonomous.provisioning.composition import (
        EmployeeDepartmentRuntime,
    )
    from src.feishu import ws_client as ws

    runtime = SimpleNamespace(close=MagicMock())
    initialization_error = RuntimeError("handler construction failed")
    monkeypatch.setattr(
        EmployeeDepartmentRuntime,
        "from_settings",
        lambda *_args, **_kwargs: runtime,
    )

    def fail_after_runtime(*_args, **_kwargs):
        raise initialization_error

    monkeypatch.setattr(ws, "_main_bot_outbound_wiring", fail_after_runtime)

    with pytest.raises(RuntimeError) as caught:
        ws.FeishuWSClient(message_callback=lambda *_args: None)

    assert caught.value is initialization_error
    runtime.close.assert_called_once_with()


def test_lifecycle_fatal_errors_are_not_silently_swallowed():
    from src.feishu.ws_lifecycle import WSLifecycleAction, classify_lifecycle_error

    disconnect = classify_lifecycle_error(RuntimeError("disconnect cleanup"), phase="disconnect")
    assert disconnect.action == WSLifecycleAction.RECORD_ACTIVITY_AND_CONTINUE

    data = classify_lifecycle_error(RuntimeError("bad frame"), phase="data_frame")
    assert data.action == WSLifecycleAction.PROPAGATE

    startup = classify_lifecycle_error(RuntimeError("auth failed"), phase="startup")
    assert startup.action == WSLifecycleAction.PROPAGATE


# ---------------------------------------------------------------------------
# WS resource manager tests (merged from test_ws_resource_manager.py)
# ---------------------------------------------------------------------------


class _Engine:
    def __init__(self, running: bool):
        self.is_running = running
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True
        self.is_running = False


class _Manager:
    def __init__(self, engines):
        self._engines = engines
        self.cleaned = False

    def list_engines(self):
        return self._engines

    def cleanup_all(self):
        self.cleaned = True


def test_engine_resource_group_stops_running_engines_and_cleans_manager():
    from src.feishu.ws_resource_manager import EngineResourceGroup

    running = _Engine(True)
    stopped = _Engine(False)
    manager = _Manager([running, stopped])

    group = EngineResourceGroup("test", manager)

    engines = group.stop_running_engines()
    group.wait_stopped(engines, timeout_s=0.1, interval_s=0.001)
    group.cleanup_all()

    assert running.stopped is True
    assert stopped.stopped is False
    assert manager.cleaned is True
