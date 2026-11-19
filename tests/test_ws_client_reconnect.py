import asyncio
import json
import os
import threading
from types import SimpleNamespace

import pytest


def test_ws_health_logs_recovery_duration_after_disconnect(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from src.feishu.ws_health import WSHealthMonitor

    now = iter((100.0, 110.0, 112.5))
    monkeypatch.setattr(
        "src.feishu.ws_health.time",
        SimpleNamespace(time=lambda: next(now)),
    )
    monitor = WSHealthMonitor(SimpleNamespace(), SimpleNamespace())

    with caplog.at_level("INFO", logger="src.feishu.ws_health"):
        monitor.record_activity("connected")
        monitor.record_activity("disconnected")
        monitor.record_activity("connected")

    assert "WS重连成功: outage_seconds=2.500" in caplog.text
    assert "[METRIC] ws_reconnected outage_seconds=2.500" in caplog.text


def test_lark_sdk_logging_is_single_and_downgrades_recoverable_disconnect(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from src.feishu.sdk_logging import configure_lark_sdk_logging

    sdk_logger = logging.getLogger("Lark")
    previous_handlers = list(sdk_logger.handlers)
    previous_filters = list(sdk_logger.filters)
    previous_propagate = sdk_logger.propagate
    first = logging.StreamHandler()
    second = logging.StreamHandler()
    sdk_logger.handlers = [first, second]
    sdk_logger.filters = []
    sdk_logger.propagate = False
    try:
        configure_lark_sdk_logging()
        with caplog.at_level(logging.WARNING):
            sdk_logger.error(
                "receive message loop exit, err: no close frame received or sent"
            )
        assert sdk_logger.handlers == []
        assert sdk_logger.propagate is True
        records = [record for record in caplog.records if record.name == "Lark"]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert records[0].levelname == "WARNING"
    finally:
        sdk_logger.handlers = previous_handlers
        sdk_logger.filters = previous_filters
        sdk_logger.propagate = previous_propagate


def _bind_reply_transport(client, *, return_value=None):
    from unittest.mock import MagicMock

    reply_text = MagicMock(return_value=return_value)
    client._handler_ctx = SimpleNamespace(
        handlers={"coco": SimpleNamespace(reply_text=reply_text)}
    )
    return reply_text


def test_scheduler_factory_does_not_publish_service_identity_during_construction(
    tmp_path,
    monkeypatch,
) -> None:
    from contextlib import nullcontext

    from src.feishu import ws_client as ws

    events = []

    class FakeGate:
        def task_guard(self):
            return nullcontext()

        def publish_participation(self, *, service_pid):
            events.append(("participating", service_pid))
            return "I" * 24

    gate = FakeGate()

    class FakeScheduler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        ws.RestartGate,
        "for_project",
        lambda *_args, **_kwargs: gate,
    )
    monkeypatch.setattr(ws, "TaskScheduler", FakeScheduler)
    settings = SimpleNamespace(
        task_scheduler_max_concurrent=1,
        task_scheduler_per_key_concurrency=1,
        system_command_concurrency=1,
        restart_gate_dir="",
    )

    scheduler = ws._build_task_scheduler(settings, project_dir=tmp_path)

    assert events == []
    assert scheduler._restart_gate is gate
    assert scheduler.kwargs["run_guard"].__self__ is gate


def test_service_start_phase_publishes_participation_once() -> None:
    from src.feishu import ws_client as ws

    events = []

    class FakeGate:
        def publish_participation(self, *, service_pid):
            events.append(("participating", service_pid))
            return "I" * 24

    client = object.__new__(ws.FeishuWSClient)
    client._restart_gate = FakeGate()
    client._restart_participation_id = None

    assert client._publish_restart_participation() == "I" * 24
    assert client._publish_restart_participation() == "I" * 24
    assert events == [("participating", os.getpid())]


def test_connected_activity_marks_the_participating_instance_ready() -> None:
    from src.feishu import ws_client as ws

    events = []

    class FakeHealthMonitor:
        def record_activity(self, kind):
            events.append(("activity", kind))

    class FakeGate:
        def mark_ready(self, *, service_pid):
            events.append(("ready", service_pid))
            return "G" * 24

    client = object.__new__(ws.FeishuWSClient)
    client._ws_health_monitor = FakeHealthMonitor()
    client._restart_gate = FakeGate()
    client._handler_ctx = type("HandlerContext", (), {"handlers": {"coco": object()}})()

    client._record_ws_activity("pong")
    client._record_ws_activity("connected")

    assert events == [
        ("activity", "pong"),
        ("activity", "connected"),
        ("ready", os.getpid()),
    ]


def test_connected_readiness_waits_for_required_reply_wiring() -> None:
    from src.feishu import ws_client as ws

    class FakeHealthMonitor:
        def record_activity(self, _kind):
            return None

    class FakeGate:
        def __init__(self) -> None:
            self.ready = threading.Event()

        def mark_ready(self, *, service_pid):
            assert service_pid == os.getpid()
            self.ready.set()
            return "G" * 24

    client = object.__new__(ws.FeishuWSClient)
    client._ws_health_monitor = FakeHealthMonitor()
    client._restart_gate = FakeGate()
    client._closed = False
    client._handler_ctx = SimpleNamespace(handlers={})

    client._record_ws_activity("connected")

    assert not client._restart_gate.ready.is_set()
    assert client._restart_readiness_blocker == "main_bot_reply_handler"
    _bind_reply_transport(client)
    assert client._try_publish_restart_readiness() is True
    assert client._restart_gate.ready.is_set()


def test_connected_marks_real_restart_gate_ready(tmp_path) -> None:
    from src.feishu import ws_client as ws
    from src.utils.restart_gate import RestartGate

    project = tmp_path / "project"
    project.mkdir()
    gate = RestartGate.for_project(project, override=tmp_path / "gate")
    gate.publish_participation(service_pid=os.getpid())

    client = object.__new__(ws.FeishuWSClient)
    client.settings = SimpleNamespace()
    client._ws_health_monitor = SimpleNamespace(record_activity=lambda _kind: None)
    client._restart_gate = gate
    client._restart_generation = None
    client._closed = False
    _bind_reply_transport(client)

    client._record_ws_activity("connected")

    assert gate.ready_generation(service_pid=os.getpid()) == client._restart_generation


def test_main_dispatcher_acknowledges_p2p_chat_entered_event() -> None:
    """Subscribed informational events are explicitly consumed as no-ops."""
    from src.feishu.ws_client import FeishuWSClient

    client = object.__new__(FeishuWSClient)
    client._handle_message = lambda _event: None
    client._handle_bot_deleted = lambda _event: None
    client._handle_card_action = lambda _event: None

    event_handler = client._build_event_handler()
    payload = {
        "schema": "2.0",
        "header": {
            "event_id": "evt-p2p-entered",
            "event_type": "im.chat.access_event.bot_p2p_chat_entered_v1",
            "create_time": "1785500978668",
            "token": "",
            "app_id": "cli_test",
            "tenant_key": "tenant_test",
        },
        "event": {"operator_id": {"open_id": "ou_test"}},
    }

    result = event_handler._do_without_validation(
        json.dumps(payload).encode("utf-8")
    )

    assert result is None


@pytest.mark.parametrize(
    "event_type,event",
    [
        ("im.chat.member.bot.added_v1", {"chat_id": "oc_added"}),
        (
            "im.message.reaction.deleted_v1",
            {
                "message_id": "om_reacted",
                "reaction_type": {"emoji_type": "OK"},
                "operator_type": "user",
                "user_id": {"open_id": "ou_actor"},
            },
        ),
    ],
)
def test_main_dispatcher_acknowledges_subscribed_informational_events(
    event_type: str,
    event: dict[str, object],
) -> None:
    from src.feishu.ws_client import FeishuWSClient

    client = object.__new__(FeishuWSClient)
    client._handle_message = lambda _event: None
    client._handle_bot_deleted = lambda _event: None
    client._handle_card_action = lambda _event: None
    event_handler = client._build_event_handler()
    payload = {
        "schema": "2.0",
        "header": {
            "event_id": f"evt-{event_type}",
            "event_type": event_type,
            "create_time": "1785500978668",
            "token": "",
            "app_id": "cli_test",
            "tenant_key": "tenant_test",
        },
        "event": event,
    }

    result = event_handler._do_without_validation(
        json.dumps(payload).encode("utf-8")
    )

    assert result is None




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


def test_observed_ws_requires_handler_lifecycle_hooks_as_a_pair() -> None:
    from unittest.mock import patch

    from lark_channel.ws import Client as ChannelWSClient

    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    with (
        patch.object(ChannelWSClient, "__init__", return_value=None),
        pytest.raises(ValueError, match="must be configured together"),
    ):
        ObservedLarkWSClient(
            "cli_test",
            "secret",
            on_activity=lambda _kind: None,
            on_handler_scheduled=lambda: True,
        )


def test_observed_ws_client_dispatches_latest_card_callback_frames() -> None:
    import base64

    from lark_channel.core.json import JSON
    from lark_channel.ws.pb.pbbp2_pb2 import Frame

    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    payload = {
        "schema": "2.0",
        "header": {
            "event_id": "evt-card",
            "event_type": "card.action.trigger",
        },
        "event": {"action": {"value": {"action": "workflow_select_tool"}}},
    }
    dispatched = []
    written = []

    class _Dispatcher:
        def _do_without_validation(self, raw):
            dispatched.append(json.loads(raw))
            return {"toast": {"type": "info", "content": "accepted"}}

    frame = Frame()
    frame.SeqID = 1
    frame.LogID = 1
    frame.service = 1
    frame.method = 1
    for key, value in (
        ("type", "card"),
        ("message_id", "msg-card"),
        ("trace_id", "trace-card"),
        ("sum", "1"),
        ("seq", "0"),
    ):
        header = frame.headers.add()
        header.key = key
        header.value = value
    frame.payload = json.dumps(payload).encode()

    client = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
    client._on_activity = lambda _kind: None
    client._event_handler = _Dispatcher()
    client._conn_id = ""

    async def _write_message(raw):
        written.append(raw)

    client._write_message = _write_message

    asyncio.run(client._handle_data_frame(frame))

    assert dispatched == [payload]
    assert len(written) == 1
    response = JSON.unmarshal(frame.payload.decode(), dict)
    assert response["code"] == 200
    assert json.loads(base64.b64decode(response["data"])) == {
        "toast": {"type": "info", "content": "accepted"}
    }


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

    client = object.__new__(FeishuWSClient)
    deleted = []
    client._handle_message = lambda _event: None
    client._handle_reaction_created = lambda _event: None
    client._handle_bot_deleted = deleted.append
    client._handle_message_read = lambda _event: None
    client._handle_card_action = lambda _event: None
    client._handle_chat_entered = lambda _event: None
    event_handler = client._build_event_handler()
    payload = {
        "schema": "2.0",
        "header": {
            "event_id": "evt-bot-deleted",
            "event_type": "im.chat.member.bot.deleted_v1",
            "create_time": "1785500978668",
            "token": "",
            "app_id": "cli_test",
            "tenant_key": "tenant_test",
        },
        "event": {"chat_id": "oc_deleted"},
    }

    result = event_handler._do_without_validation(
        json.dumps(payload).encode("utf-8")
    )

    assert result is None
    assert len(deleted) == 1
    assert deleted[0].event.chat_id == "oc_deleted"








def test_ws_client_init_failure_does_not_leave_scheduler_holding_client(
    monkeypatch,
    tmp_path,
) -> None:
    from unittest.mock import MagicMock

    from src.feishu import ws_client as ws

    settings = ws.get_settings().model_copy(update={"restart_gate_dir": ""})
    monkeypatch.setattr(ws, "get_settings", lambda: settings)
    monkeypatch.setattr(ws, "_CHECKOUT_ROOT", tmp_path)
    monkeypatch.setattr(ws, "ProjectManager", MagicMock())
    initialization_error = RuntimeError("handler construction failed")
    schedulers = []
    build_scheduler = ws._build_task_scheduler

    def capture_scheduler(*args, **kwargs):
        scheduler = build_scheduler(*args, **kwargs)
        schedulers.append(scheduler)
        return scheduler

    monkeypatch.setattr(ws, "_build_task_scheduler", capture_scheduler)
    monkeypatch.setattr(
        ws,
        "HandlerContext",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(initialization_error),
    )

    try:
        with pytest.raises(RuntimeError) as caught:
            ws.FeishuWSClient(message_callback=lambda *_args: None)

        assert caught.value is initialization_error
        assert len(schedulers) == 1
        scheduler = schedulers[0]
        assert not scheduler._dispatcher.is_alive()
        with scheduler._lock:
            listener_owners = {
                getattr(listener, "__self__", None)
                for listener in scheduler._listeners
            }
        assert not any(
            isinstance(owner, ws.FeishuWSClient) for owner in listener_owners
        )
    finally:
        for scheduler in schedulers:
            scheduler.stop(wait=True, shutdown_executor=True)
            with scheduler._lock:
                scheduler._listeners.clear()


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


def test_ws_health_disconnect_outcomes(monkeypatch) -> None:
    """B083: distinguish NO_CONNECTION, DISCONNECTED, TIMED_OUT, and FAILED."""
    from unittest.mock import MagicMock

    from src.feishu.ws_health import WSDisconnectOutcome, WSHealthMonitor

    holder = SimpleNamespace(_client=None)
    monitor = WSHealthMonitor(holder, SimpleNamespace())

    # 1. No client / no _conn -> NO_CONNECTION
    assert monitor.disconnect_with_outcome() is WSDisconnectOutcome.NO_CONNECTION
    assert monitor.disconnect() is True
    assert monitor._trigger_disconnect(reason="test") is False

    holder._client = SimpleNamespace(_conn=None)
    assert monitor.disconnect_with_outcome() is WSDisconnectOutcome.NO_CONNECTION
    assert monitor.disconnect() is True

    # 2. Active connection + running loop -> DISCONNECTED
    running_loop = SimpleNamespace(is_closed=lambda: False, is_running=lambda: True)
    monkeypatch.setattr("lark_channel.ws.client.loop", running_loop)

    def fake_coro():
        async def _noop():
            return True

        return _noop()

    holder._client = SimpleNamespace(_conn=object(), _disconnect=fake_coro)
    ok_future = MagicMock()
    ok_future.result.return_value = True

    def _run_ok(coro, loop):
        coro.close()
        assert loop is running_loop
        return ok_future

    monkeypatch.setattr("src.feishu.ws_health.asyncio.run_coroutine_threadsafe", _run_ok)
    assert monitor.disconnect_with_outcome() is WSDisconnectOutcome.DISCONNECTED
    assert monitor.disconnect() is True
    assert monitor._trigger_disconnect(reason="test") is True

    # 3. 5s wait timeout -> TIMED_OUT and cancels future
    timeout_future = MagicMock()
    timeout_future.result.side_effect = TimeoutError("5s elapsed")

    def _run_timeout(coro, _loop):
        coro.close()
        return timeout_future

    monkeypatch.setattr("src.feishu.ws_health.asyncio.run_coroutine_threadsafe", _run_timeout)
    assert monitor.disconnect_with_outcome() is WSDisconnectOutcome.TIMED_OUT
    timeout_future.cancel.assert_called_once_with()
    assert monitor.disconnect() is False
    assert monitor._trigger_disconnect(reason="test") is False

    # 4. _disconnect returns False with _conn still set -> FAILED; cleared -> NO_CONNECTION
    false_future = MagicMock()
    false_future.result.return_value = False
    monkeypatch.setattr(
        "src.feishu.ws_health.asyncio.run_coroutine_threadsafe",
        lambda coro, _loop: (coro.close(), false_future)[1],
    )
    assert monitor.disconnect_with_outcome() is WSDisconnectOutcome.FAILED
    assert monitor.disconnect() is False

    holder._client = SimpleNamespace(_conn=None, _disconnect=fake_coro)
    assert monitor.disconnect_with_outcome() is WSDisconnectOutcome.NO_CONNECTION

    # 5. Exception during disconnect -> FAILED
    holder._client = SimpleNamespace(_conn=object(), _disconnect=fake_coro)
    err_future = MagicMock()
    err_future.result.side_effect = RuntimeError("socket error")
    err_future.done.return_value = True
    monkeypatch.setattr(
        "src.feishu.ws_health.asyncio.run_coroutine_threadsafe",
        lambda coro, _loop: (coro.close(), err_future)[1],
    )
    assert monitor.disconnect_with_outcome() is WSDisconnectOutcome.FAILED
    assert monitor.disconnect() is False
    assert monitor._trigger_disconnect(reason="test") is False
