"""Regression tests for B083, B084, B090 shutdown race, and Spec total SLA."""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.ws_client import FeishuWSClient
from src.feishu.ws_health import WSDisconnectOutcome
from src.tasking import TaskScheduler, TaskSpec, TaskStatus


def _build_minimal_ws_client(scheduler=None) -> FeishuWSClient:
    """Build a mocked-collaborator FeishuWSClient suitable for close() unit tests."""
    sched = scheduler if scheduler is not None else MagicMock()
    if not hasattr(sched, "_restart_gate"):
        sched._restart_gate = MagicMock()
    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client._build_task_scheduler", return_value=sched),
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
        return FeishuWSClient(MagicMock())


class TestWSCloseReentrantAndDisconnectOutcome:
    """B084 & B083 close() contract tests."""

    def test_scheduler_is_in_completion_callback_and_stop_self_join_guard(self) -> None:
        scheduler = TaskScheduler(max_concurrent=1, system_concurrency=1)
        seen_in_cb: list[bool] = []
        stop_errors: list[BaseException] = []
        done = threading.Event()
        try:
            assert scheduler.is_in_completion_callback() is False
            handle = scheduler.submit(
                TaskSpec(chat_id="oc_cb", name="cb-guard"),
                lambda _ctx: "ok",
            )

            def on_done(_event) -> None:
                seen_in_cb.append(scheduler.is_in_completion_callback())
                try:
                    scheduler.stop(wait=True, shutdown_executor=False)
                except BaseException as exc:
                    stop_errors.append(exc)
                finally:
                    done.set()

            handle.add_done_callback(on_done)
            assert done.wait(timeout=3)
            assert seen_in_cb == [True]
            assert stop_errors == []
            assert scheduler.is_in_completion_callback() is False
        finally:
            scheduler.stop(wait=True, shutdown_executor=True)

    def test_close_from_completion_callback_raises_before_setting_closed(self) -> None:
        scheduler = TaskScheduler(max_concurrent=1, system_concurrency=1)
        client = _build_minimal_ws_client(scheduler=scheduler)
        callback_errors: list[BaseException] = []
        done = threading.Event()
        try:
            handle = scheduler.submit(
                TaskSpec(chat_id="oc_reentrant", name="reentrant-close"),
                lambda _ctx: "ok",
            )

            def on_done(ev) -> None:
                assert ev.status == TaskStatus.SUCCEEDED
                try:
                    client.close()
                except BaseException as exc:
                    callback_errors.append(exc)
                finally:
                    done.set()

            handle.add_done_callback(on_done)
            assert done.wait(timeout=3)
            assert len(callback_errors) == 1
            assert isinstance(callback_errors[0], RuntimeError)
            assert client._closed is False

            with patch(
                "src.card.delivery.registry.delivery_registry.shutdown_all",
                return_value=True,
            ):
                assert client.close() is True
            assert client._closed is True
        finally:
            scheduler.stop(wait=True, shutdown_executor=True)

    @pytest.mark.parametrize(
        "outcome",
        [WSDisconnectOutcome.TIMED_OUT, WSDisconnectOutcome.FAILED],
    )
    def test_close_preserves_deps_and_allows_retry_when_ws_disconnect_fails(
        self,
        outcome: WSDisconnectOutcome,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = _build_minimal_ws_client()
        raw_ws_client = SimpleNamespace(_conn=object())
        client._client = raw_ws_client
        running_spec = MagicMock()
        running_spec.is_running = True
        running_spec.stop.side_effect = lambda: setattr(running_spec, "is_running", False)
        client._spec_engine_manager.list_engines.return_value = [running_spec]
        client._deep_engine_manager.list_engines.return_value = []
        client._workflow_engine_manager.list_engines.return_value = []
        client._scheduler.wait_for_idle.return_value = True
        client._scheduler.wait_for_completion_callbacks.return_value = True
        client._thread_manager = MagicMock()
        acp_cleanup = client._handler_ctx.managers["coco"].cleanup_all
        acp_cleanup.reset_mock()

        outcomes = iter([outcome, WSDisconnectOutcome.DISCONNECTED])
        client._ws_health_monitor.disconnect_with_outcome = lambda: next(outcomes)

        with (
            caplog.at_level(logging.ERROR),
            patch(
                "src.card.delivery.registry.delivery_registry.shutdown_all",
                return_value=True,
            ) as shutdown_all,
        ):
            assert client.close() is False

        assert outcome.value in caplog.text
        running_spec.stop.assert_called_once_with()
        client._scheduler.wait_for_idle.assert_called()
        client._scheduler.stop.assert_called_once_with(
            wait=True, shutdown_executor=False
        )
        acp_cleanup.assert_not_called()
        client._thread_manager.close.assert_not_called()
        shutdown_all.assert_not_called()
        assert client._client is raw_ws_client

        # Retry succeeds once WS disconnect completes.
        client._scheduler.stop.reset_mock()
        with patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ) as shutdown_all:
            assert client.close() is True

        shutdown_all.assert_called_once()
        assert acp_cleanup.call_count == len(client._handler_ctx.managers)
        client._thread_manager.close.assert_called_once_with()
        client._scheduler.stop.assert_called_once_with(
            wait=True, shutdown_executor=True
        )
        assert client._client is None

    def test_close_falls_back_to_mock_disconnect_boolean(self) -> None:
        client = _build_minimal_ws_client()
        client._scheduler.wait_for_idle.return_value = True
        client._scheduler.wait_for_completion_callbacks.return_value = True
        client._deep_engine_manager.list_engines.return_value = []
        client._spec_engine_manager.list_engines.return_value = []
        client._workflow_engine_manager.list_engines.return_value = []
        client._ws_health_monitor = SimpleNamespace(
            stop_watchdog=MagicMock(),
            disconnect=MagicMock(return_value=False),
        )
        acp_cleanup = client._handler_ctx.managers["coco"].cleanup_all
        acp_cleanup.reset_mock()

        with patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ):
            assert client.close() is False
        acp_cleanup.assert_not_called()

        client._ws_health_monitor.disconnect.return_value = True
        with patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ):
            assert client.close() is True
        assert acp_cleanup.call_count == len(client._handler_ctx.managers)
