import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.card.ui_text import UI_TEXT
from src.feishu.ws_client import FeishuWSClient
from src.tasking import TaskScheduler, TaskSpec, TaskStatus
from tests.helpers.ws_routing_harness import (
    _pending_task_handle,
    create_mock_message,
    mock_ws_client,
)

__all__ = ["mock_ws_client"]


def test_close_shuts_down_card_registry_with_one_bounded_wave(
    mock_ws_client: FeishuWSClient,
) -> None:
    with (
        patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ) as shutdown_all,
        patch(
            "src.card.delivery.registry.delivery_registry.drain_in_flight",
            return_value=True,
        ) as drain_in_flight,
    ):
        assert mock_ws_client.close() is True

    drain_in_flight.assert_not_called()
    shutdown_all.assert_called_once()
    assert shutdown_all.call_args.kwargs["timeout"] > 0


def test_only_control_plane_scheduler_listener_registers_and_detaches_after_idle_close(
    mock_ws_client: FeishuWSClient,
) -> None:
    scheduler = mock_ws_client._scheduler
    control_plane_callbacks = [
        call.args[0]
        for call in scheduler.add_listener.call_args_list
        if getattr(call.args[0], "__self__", None) is mock_ws_client._control_plane
    ]

    assert len(control_plane_callbacks) == 1
    assert scheduler.add_listener.call_count == 1
    scheduler.remove_listener.reset_mock()

    with patch(
        "src.card.delivery.registry.delivery_registry.shutdown_all",
        return_value=True,
    ):
        assert mock_ws_client.close() is True

    scheduler.remove_listener.assert_called_once_with(control_plane_callbacks[0])


def test_close_keeps_scheduler_listener_until_running_callbacks_drain(
    mock_ws_client: FeishuWSClient,
) -> None:
    scheduler = mock_ws_client._scheduler
    scheduler.wait_for_idle.return_value = False
    scheduler.remove_listener.reset_mock()

    assert mock_ws_client.close() is False

    scheduler.remove_listener.assert_not_called()


def test_close_waits_for_completion_callbacks_before_manager_cleanup(
    mock_ws_client: FeishuWSClient,
) -> None:
    scheduler = mock_ws_client._scheduler
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    scheduler.wait_for_idle.return_value = True
    cleanup.reset_mock()

    def wait_for_completions(*, timeout: float) -> bool:
        assert timeout > 0
        cleanup.assert_not_called()
        return True

    scheduler.wait_for_completion_callbacks.side_effect = wait_for_completions
    with patch(
        "src.card.delivery.registry.delivery_registry.shutdown_all",
        return_value=True,
    ):
        assert mock_ws_client.close() is True

    scheduler.wait_for_completion_callbacks.assert_called_once()
    assert cleanup.call_count == len(mock_ws_client._handler_ctx.managers)


def test_close_preserves_manager_cleanup_when_completion_callbacks_do_not_drain(
    mock_ws_client: FeishuWSClient,
) -> None:
    scheduler = mock_ws_client._scheduler
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    scheduler.wait_for_idle.return_value = True
    scheduler.wait_for_completion_callbacks.return_value = False
    cleanup.reset_mock()

    with patch(
        "src.card.delivery.registry.delivery_registry.shutdown_all",
        return_value=True,
    ) as shutdown_all:
        assert mock_ws_client.close() is False

    shutdown_all.assert_not_called()
    cleanup.assert_not_called()


def test_close_waits_for_terminal_replay_to_anchor_prestart_warning(
    mock_ws_client: FeishuWSClient,
) -> None:
    from src.feishu.ws_client import _MessageIngressReservation

    original_scheduler = mock_ws_client._scheduler
    scheduler = TaskScheduler(max_concurrent=1, system_concurrency=1)
    mock_ws_client._scheduler = scheduler
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    cleanup.reset_mock()
    message_id = "om_terminal_replay_shutdown"
    reservation_owner = mock_ws_client._message_ingress_guard.reserve(message_id)
    assert reservation_owner is not None
    reservation = _MessageIngressReservation(
        guard=mock_ws_client._message_ingress_guard,
        message_id=message_id,
        owner=reservation_owner,
    )
    terminal = threading.Event()
    warning_started = threading.Event()
    release_warning = threading.Event()
    warning_anchored = threading.Event()
    close_wait_started = threading.Event()
    cleanup_started = threading.Event()
    close_results: list[bool] = []

    scheduler.add_listener(
        lambda event: terminal.set()
        if event.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED}
        else None
    )
    handle = scheduler.submit(
        TaskSpec(chat_id="oc_group", name="prestart-terminal"),
        lambda _ctx: None,
    )
    assert terminal.wait(timeout=1)

    def reply_warning(*_args, **_kwargs) -> bool:
        warning_started.set()
        release_warning.wait(timeout=3)
        warning_anchored.set()
        return True

    mock_ws_client._reply_text_with_bounded_retry = reply_warning
    cleanup.side_effect = cleanup_started.set
    wait_for_completions = scheduler.wait_for_completion_callbacks

    def observe_completion_wait(*, timeout: float) -> bool:
        close_wait_started.set()
        return wait_for_completions(timeout=timeout)

    scheduler.wait_for_completion_callbacks = observe_completion_wait  # type: ignore[method-assign]
    registration_thread = threading.Thread(
        target=mock_ws_client._bind_message_ingress_reservation,
        args=(handle, reservation),
        kwargs={"tenant_key": "tenant-a", "chat_id": "oc_group"},
    )
    close_thread = threading.Thread(
        target=lambda: close_results.append(mock_ws_client.close())
    )

    try:
        with patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ):
            registration_thread.start()
            assert warning_started.wait(timeout=1)
            close_thread.start()
            assert close_wait_started.wait(timeout=1)
            assert cleanup_started.wait(timeout=0.2) is False

            release_warning.set()
            registration_thread.join(timeout=1)
            close_thread.join(timeout=2)

        assert warning_anchored.is_set()
        assert close_results == [True]
        assert cleanup.call_count == len(mock_ws_client._handler_ctx.managers)
        assert mock_ws_client._message_ingress_guard.reserve(message_id) is None
    finally:
        release_warning.set()
        registration_thread.join(timeout=1)
        close_thread.join(timeout=2)
        mock_ws_client._scheduler = original_scheduler
        scheduler.stop(wait=True, shutdown_executor=True)


def test_close_waits_for_dispatched_message_to_bind_prestart_callback(
    mock_ws_client: FeishuWSClient,
) -> None:
    """A Channel-dispatched handler must join shutdown before teardown."""

    class ReplayedTerminalHandle:
        run_id = "run_submit_bind_shutdown"

        @staticmethod
        def add_done_callback(callback) -> None:
            callback(
                SimpleNamespace(
                    run_id="run_submit_bind_shutdown",
                    status=TaskStatus.CANCELED,
                )
            )

    scheduler = mock_ws_client._scheduler
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    scheduler.submit.return_value = ReplayedTerminalHandle()
    scheduler.wait_for_idle.return_value = True
    scheduler.wait_for_completion_callbacks.return_value = True
    reply_warning = MagicMock(return_value=True)
    mock_ws_client._reply_text_with_bounded_retry = reply_warning
    cleanup.reset_mock()

    bind_entered = threading.Event()
    release_bind = threading.Event()
    cleanup_started = threading.Event()
    close_results: list[bool] = []
    original_bind = mock_ws_client._bind_message_ingress_reservation

    def block_between_submit_and_bind(*args, **kwargs) -> None:
        bind_entered.set()
        assert release_bind.wait(timeout=2)
        original_bind(*args, **kwargs)

    mock_ws_client._bind_message_ingress_reservation = block_between_submit_and_bind  # type: ignore[method-assign]
    cleanup.side_effect = cleanup_started.set
    handler_thread = threading.Thread(
        target=mock_ws_client._handle_message,
        args=(
            create_mock_message(
                "hello",
                message_id="om_submit_bind_shutdown",
                chat_id="oc_submit_bind_shutdown",
            ),
        ),
    )
    close_thread = threading.Thread(
        target=lambda: close_results.append(mock_ws_client.close())
    )

    try:
        assert mock_ws_client._begin_message_ingress_binding()
        with patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ):
            handler_thread.start()
            assert bind_entered.wait(timeout=1)
            close_thread.start()
            assert cleanup_started.wait(timeout=0.2) is False

            release_bind.set()
            handler_thread.join(timeout=1)
            mock_ws_client._finish_message_ingress_binding()
            close_thread.join(timeout=2)

        assert handler_thread.is_alive() is False
        assert close_thread.is_alive() is False
        assert close_results == [True]
        reply_warning.assert_called_once_with(
            "om_submit_bind_shutdown",
            UI_TEXT["ws_message_prestart_terminal"],
            label="pre-start terminal warning",
        )
        assert cleanup.call_count == len(mock_ws_client._handler_ctx.managers)
    finally:
        release_bind.set()
        handler_thread.join(timeout=1)
        if mock_ws_client._message_ingress_bindings_inflight:
            mock_ws_client._finish_message_ingress_binding()
        close_thread.join(timeout=2)


def test_close_timeout_preserves_cleanup_during_submit_bind_window(
    mock_ws_client: FeishuWSClient,
) -> None:
    scheduler = mock_ws_client._scheduler
    scheduler.submit.return_value = _pending_task_handle("run_bind_timeout")
    scheduler.wait_for_idle.return_value = True
    scheduler.wait_for_completion_callbacks.return_value = True
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    cleanup.reset_mock()

    bind_entered = threading.Event()
    release_bind = threading.Event()
    original_bind = mock_ws_client._bind_message_ingress_reservation

    def block_between_submit_and_bind(*args, **kwargs) -> None:
        bind_entered.set()
        assert release_bind.wait(timeout=2)
        original_bind(*args, **kwargs)

    mock_ws_client._bind_message_ingress_reservation = block_between_submit_and_bind  # type: ignore[method-assign]
    handler_thread = threading.Thread(
        target=mock_ws_client._handle_message,
        args=(
            create_mock_message(
                "hello",
                message_id="om_bind_timeout",
                chat_id="oc_bind_timeout",
            ),
        ),
    )

    try:
        assert mock_ws_client._begin_message_ingress_binding()
        handler_thread.start()
        assert bind_entered.wait(timeout=1)
        started = time.monotonic()
        with (
            patch("src.feishu.ws_client._SHUTDOWN_SCHEDULER_DRAIN_S", 0.05),
            patch(
                "src.card.delivery.registry.delivery_registry.shutdown_all",
                return_value=True,
            ) as shutdown_all,
        ):
            assert mock_ws_client.close() is False
        assert time.monotonic() - started < 0.5
        cleanup.assert_not_called()
        shutdown_all.assert_not_called()
    finally:
        release_bind.set()
        handler_thread.join(timeout=1)
        if mock_ws_client._message_ingress_bindings_inflight:
            mock_ws_client._finish_message_ingress_binding()


def test_observed_ws_tracks_scheduled_handler_before_business_entry(
    mock_ws_client: FeishuWSClient,
) -> None:
    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    scheduler = mock_ws_client._scheduler
    scheduler.submit.return_value = SimpleNamespace(
        run_id="run_scheduled_before_entry",
        add_done_callback=lambda callback: callback(
            SimpleNamespace(
                run_id="run_scheduled_before_entry",
                status=TaskStatus.CANCELED,
            )
        ),
    )
    scheduler.wait_for_idle.return_value = True
    scheduler.wait_for_completion_callbacks.return_value = True
    reply_warning = MagicMock(return_value=True)
    mock_ws_client._reply_text_with_bounded_retry = reply_warning
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    cleanup.reset_mock()

    observed = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
    observed._handler_semaphore = None
    observed._on_handler_scheduled = mock_ws_client._begin_message_ingress_binding
    observed._on_handler_finished = mock_ws_client._finish_message_ingress_binding
    allow_business_entry: asyncio.Event
    cleanup_started = threading.Event()
    close_results: list[bool] = []
    cleanup.side_effect = cleanup_started.set
    message = create_mock_message(
        "hello",
        message_id="om_scheduled_before_entry",
        chat_id="oc_scheduled_before_entry",
    )

    async def exercise() -> None:
        nonlocal allow_business_entry
        allow_business_entry = asyncio.Event()

        async def delayed_handler(_raw) -> None:
            await allow_business_entry.wait()
            mock_ws_client._handle_message(message)

        observed._handle_message = delayed_handler  # type: ignore[method-assign]
        await observed._schedule_handle_message(b"raw-frame")
        close_thread = threading.Thread(
            target=lambda: close_results.append(mock_ws_client.close())
        )
        with patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ):
            close_thread.start()
            deadline = asyncio.get_running_loop().time() + 0.2
            while not cleanup_started.is_set() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert cleanup_started.is_set() is False
            allow_business_entry.set()
            deadline = asyncio.get_running_loop().time() + 2
            while close_thread.is_alive() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
        assert close_thread.is_alive() is False

    asyncio.run(exercise())

    assert close_results == [True]
    reply_warning.assert_called_once_with(
        "om_scheduled_before_entry",
        UI_TEXT["ws_message_prestart_terminal"],
        label="pre-start terminal warning",
    )
    assert cleanup.call_count == len(mock_ws_client._handler_ctx.managers)


def test_close_stops_engines_before_busy_scheduler_drain_and_skips_cleanup_on_timeout(
    mock_ws_client: FeishuWSClient,
) -> None:
    events: list[str] = []
    running_spec = MagicMock()
    running_spec.is_running = True
    running_spec.stop.side_effect = lambda: events.append("spec_stop")
    mock_ws_client._spec_engine_manager.list_engines.return_value = [running_spec]
    mock_ws_client._deep_engine_manager.list_engines.return_value = []
    mock_ws_client._workflow_engine_manager.list_engines.return_value = []

    scheduler = mock_ws_client._scheduler
    scheduler.cancel_active.side_effect = lambda: events.append("cancel_active")
    scheduler.wait_for_idle.side_effect = lambda _timeout: (
        events.append("wait_for_idle"),
        False,
    )[1]
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    cleanup.reset_mock()

    with patch(
        "src.card.delivery.registry.delivery_registry.shutdown_all",
        return_value=True,
    ) as shutdown_all:
        assert mock_ws_client.close() is False

    assert events.index("spec_stop") < events.index("wait_for_idle")
    assert events.index("cancel_active") < events.index("wait_for_idle")
    cleanup.assert_not_called()
    shutdown_all.assert_not_called()
    scheduler.stop.assert_called_once_with(wait=True, shutdown_executor=False)


def test_observed_ws_rejects_handler_scheduled_after_shutdown_fence(
    mock_ws_client: FeishuWSClient,
) -> None:
    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    observed = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
    observed._handler_semaphore = None
    observed._on_handler_scheduled = mock_ws_client._begin_message_ingress_binding
    observed._on_handler_finished = mock_ws_client._finish_message_ingress_binding
    handled = MagicMock()

    async def handle_message(_raw) -> None:
        handled()

    observed._handle_message = handle_message  # type: ignore[method-assign]
    mock_ws_client._fence_message_ingress_bindings()

    asyncio.run(observed._schedule_handle_message(b"late-frame"))

    handled.assert_not_called()
    assert mock_ws_client._message_ingress_bindings_inflight == 0


def test_observed_ws_immediate_cancel_releases_handler_tracking() -> None:
    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    scheduled = MagicMock(return_value=True)
    finished = MagicMock()
    handled = MagicMock()

    async def exercise() -> None:
        observed = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
        observed._handler_semaphore = None
        observed._on_handler_scheduled = scheduled
        observed._on_handler_finished = finished

        async def handle_message(_raw) -> None:
            handled()

        observed._handle_message = handle_message  # type: ignore[method-assign]
        before = asyncio.all_tasks()
        await observed._schedule_handle_message(b"cancel-before-start")
        created = asyncio.all_tasks() - before
        assert len(created) == 1
        task = created.pop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())

    scheduled.assert_called_once_with()
    finished.assert_called_once_with()
    handled.assert_not_called()


def test_observed_ws_immediate_cancel_releases_handler_semaphore() -> None:
    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    scheduled = MagicMock(return_value=True)
    finished = MagicMock()
    handled = MagicMock()

    async def exercise() -> None:
        observed = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
        observed._handler_semaphore = asyncio.Semaphore(1)
        observed._on_handler_scheduled = scheduled
        observed._on_handler_finished = finished

        async def handle_message(_raw) -> None:
            handled()

        observed._handle_message = handle_message  # type: ignore[method-assign]
        before = asyncio.all_tasks()
        await observed._schedule_handle_message(b"cancel-before-start")
        created = asyncio.all_tasks() - before
        assert len(created) == 1
        task = created.pop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(observed._handler_semaphore.acquire(), timeout=0.1)

    asyncio.run(exercise())

    scheduled.assert_called_once_with()
    finished.assert_called_once_with()
    handled.assert_not_called()


def test_close_preserves_dependencies_when_card_registry_shutdown_times_out(
    mock_ws_client: FeishuWSClient,
) -> None:
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    cleanup.reset_mock()
    with patch(
        "src.card.delivery.registry.delivery_registry.shutdown_all",
        return_value=False,
    ):
        assert mock_ws_client.close() is False

    cleanup.assert_not_called()
