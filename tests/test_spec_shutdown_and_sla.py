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


class TestSpecShutdownRaceAndGuards:
    """B090: active Spec (in build and parallel review) exits cleanly on FeishuWSClient.close()."""

    def test_active_spec_in_build_cancels_cleanly_on_ws_close(self, tmp_path, monkeypatch) -> None:
        from src.acp.models import ACPEvent, ACPEventType
        from src.spec_engine.engine import SpecEngine
        from src.spec_engine.models import SpecProjectStatus

        scheduler = TaskScheduler(max_concurrent=1, system_concurrency=1)
        client = _build_minimal_ws_client(scheduler=scheduler)
        build_entered = threading.Event()
        prompt_cancelled = threading.Event()
        criteria_called = threading.Event()
        spec_json = (
            '{"goals":["G"],"functional_spec":["F"],"acceptance_criteria":["C1"],'
            '"required_experts":['
            '{"role":"R1","purpose":"P1","focus":["F1"],"checks":["K1"]},'
            '{"role":"R2","purpose":"P2","focus":["F2"],"checks":["K2"]},'
            '{"role":"R3","purpose":"P3","focus":["F3"],"checks":["K3"]}]}'
        )
        plan_json = '{"architecture":"A","steps":["S"]}'

        class _BuildSession:
            def __init__(self) -> None:
                self._call = 0

            def send_prompt(self, _prompt, on_event=None, timeout=None):
                self._call += 1
                responses = [spec_json, plan_json, "1. T1 (依赖: 无)"]
                if self._call <= len(responses):
                    if on_event:
                        on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=responses[self._call - 1]))
                    return SimpleNamespace(stop_reason="end_turn")
                build_entered.set()
                assert prompt_cancelled.wait(timeout=3)
                raise RuntimeError("cancelled during build")

            def send_prompt_with_retry(self, prompt, on_event=None, timeout=None, **kw):
                return self.send_prompt(prompt, on_event=on_event, timeout=timeout)

            def cancel(self, wait=False, timeout=2.0):
                prompt_cancelled.set()
                return True

            def close(self):
                return None

        settings = SimpleNamespace(
            spec_max_cycles=2,
            spec_max_cycles_limit=10,
            spec_execution_timeout=120,
            spec_convergence_window=2,
            spec_min_cycles=1,
            spec_review_enabled=False,
            spec_discovery_enabled=False,
            spec_persist_phase_artifacts=False,
            spec_persist_every_phase=True,
            spec_cycle_tasks_max=10,
            spec_cycle_output_max_chars=4000,
            spec_state_filename=".spec_engine_state.json",
            spec_artifacts_dirname=".spec_engine",
            spec_history_log_filename="history.jsonl",
            spec_phase_output_persist_max_chars=4000,
            spec_cycle_artifact_retention=10,
            spec_generated_specs_retention=10,
            spec_state_cycles_tail=10,
            spec_state_work_items_tail=10,
            spec_state_metrics_tail=10,
            spec_max_retries=0,
            spec_max_consecutive_failures=3,
            spec_infinite_mode=False,
            spec_disable_convergence=False,
            spec_disable_early_stop=False,
            spec_objective_verify_enabled=False,
            spec_rebuild_session_between_cycles=False,
            spec_cache_root=str(tmp_path / "cache"),
        )
        monkeypatch.setattr("src.engine_base.get_settings", lambda: settings)
        engine = SpecEngine(
            chat_id="oc_spec_build",
            root_path=str(tmp_path),
            create_session_fn=lambda **_kw: _BuildSession(),
        )
        monkeypatch.setattr(
            engine,
            "_evaluate_criteria",
            lambda *_a, **_kw: (criteria_called.set(), {"all_satisfied": False})[1],
        )
        client._spec_engine_manager.list_engines.return_value = [engine]
        client._deep_engine_manager.list_engines.return_value = []
        client._workflow_engine_manager.list_engines.return_value = []

        result_holder: list[object] = []
        handle = scheduler.submit(
            TaskSpec(chat_id="oc_spec_build", name="spec-build"),
            lambda _ctx: result_holder.append(engine.execute("build feature")),
        )
        assert handle is not None
        assert build_entered.wait(timeout=3)

        try:
            with patch(
                "src.card.delivery.registry.delivery_registry.shutdown_all",
                return_value=True,
            ):
                assert client.close() is True
            assert len(result_holder) == 1
            assert engine.project.status == SpecProjectStatus.CANCELLED
            assert criteria_called.is_set() is False
        finally:
            prompt_cancelled.set()
            scheduler.stop(wait=True, shutdown_executor=True)

    def test_active_spec_in_parallel_review_cancels_sessions_immediately(self, tmp_path, monkeypatch) -> None:
        from src.acp.models import ACPEvent, ACPEventType
        from src.spec_engine.engine import SpecEngine
        from src.spec_engine.models import SpecProjectStatus

        scheduler = TaskScheduler(max_concurrent=1, system_concurrency=1)
        client = _build_minimal_ws_client(scheduler=scheduler)
        review_entered = threading.Event()
        review_cancelled = threading.Event()
        eval_called = threading.Event()
        spec_json = (
            '{"goals":["G"],"functional_spec":["F"],"acceptance_criteria":["C1"],'
            '"required_experts":['
            '{"role":"R1","purpose":"P1","focus":["F1"],"checks":["K1"]},'
            '{"role":"R2","purpose":"P2","focus":["F2"],"checks":["K2"]},'
            '{"role":"R3","purpose":"P3","focus":["F3"],"checks":["K3"]}]}'
        )
        plan_json = '{"architecture":"A","steps":["S"]}'

        class _MainSession:
            def __init__(self) -> None:
                self._call = 0

            def send_prompt(self, _prompt, on_event=None, timeout=None):
                self._call += 1
                responses = [spec_json, plan_json, "1. T1 (依赖: 无)", "build ok"]
                text = responses[self._call - 1] if self._call <= len(responses) else "ok"
                if on_event:
                    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
                return SimpleNamespace(stop_reason="end_turn")

            def send_prompt_with_retry(self, prompt, on_event=None, timeout=None, **kw):
                return self.send_prompt(prompt, on_event=on_event, timeout=timeout)

            def cancel(self, wait=False, timeout=2.0):
                return True

            def close(self):
                return None

        class _BlockingReviewActiveSession:
            def __init__(self) -> None:
                self.cancelled = threading.Event()

            def send_prompt(self, _prompt, on_event=None, timeout=None):
                review_entered.set()
                assert self.cancelled.wait(timeout=3)
                review_cancelled.set()
                raise RuntimeError("review session cancelled")

            def cancel(self, wait=False, timeout=2.0):
                self.cancelled.set()
                return True

        class _FakeEphemeralReviewSession:
            def __init__(self, *_args, **_kwargs) -> None:
                self.session_started = True
                self.startup_elapsed_s = 0.01
                self._session = _BlockingReviewActiveSession()

            def __enter__(self):
                return self._session

            def __exit__(self, *_exc):
                return None

        settings = SimpleNamespace(
            spec_max_cycles=2,
            spec_max_cycles_limit=10,
            spec_execution_timeout=120,
            spec_convergence_window=2,
            spec_min_cycles=1,
            spec_review_enabled=True,
            spec_review_timeout=240,
            spec_review_role_timeout_multipliers={},
            spec_review_max_parallel=2,
            spec_review_total_roles_max=4,
            spec_review_retry_max_attempts=1,
            spec_review_retry_base_delay=2.0,
            spec_review_retry_max_delay=30,
            spec_review_startup_timeout=30,
            spec_review_failure_circuit_enabled=False,
            spec_completion_gate_enabled=True,
            spec_discovery_enabled=False,
            spec_persist_phase_artifacts=False,
            spec_persist_every_phase=True,
            spec_cycle_tasks_max=10,
            spec_cycle_output_max_chars=4000,
            spec_state_filename=".spec_engine_state.json",
            spec_artifacts_dirname=".spec_engine",
            spec_history_log_filename="history.jsonl",
            spec_phase_output_persist_max_chars=4000,
            spec_cycle_artifact_retention=10,
            spec_generated_specs_retention=10,
            spec_state_cycles_tail=10,
            spec_state_work_items_tail=10,
            spec_state_metrics_tail=10,
            spec_max_retries=0,
            spec_max_consecutive_failures=3,
            spec_infinite_mode=False,
            spec_disable_convergence=False,
            spec_disable_early_stop=False,
            spec_objective_verify_enabled=False,
            spec_rebuild_session_between_cycles=False,
            spec_cache_root=str(tmp_path / "cache"),
        )
        monkeypatch.setattr("src.engine_base.get_settings", lambda: settings)
        monkeypatch.setattr("src.spec_engine.review.EphemeralReviewSession", _FakeEphemeralReviewSession)
        engine = SpecEngine(
            chat_id="oc_spec_review",
            root_path=str(tmp_path),
            create_session_fn=lambda **_kw: _MainSession(),
        )
        monkeypatch.setattr(
            engine,
            "_evaluate_criteria",
            lambda *_a, **_kw: (eval_called.set(), {"all_satisfied": False})[1],
        )
        client._spec_engine_manager.list_engines.return_value = [engine]
        client._deep_engine_manager.list_engines.return_value = []
        client._workflow_engine_manager.list_engines.return_value = []

        scheduler.submit(
            TaskSpec(chat_id="oc_spec_review", name="spec-review"),
            lambda _ctx: engine.execute("review feature"),
        )
        assert review_entered.wait(timeout=3)

        try:
            with patch(
                "src.card.delivery.registry.delivery_registry.shutdown_all",
                return_value=True,
            ):
                assert client.close() is True
            assert review_cancelled.is_set()
            assert engine.project.status == SpecProjectStatus.CANCELLED
            assert eval_called.is_set() is False
        finally:
            scheduler.stop(wait=True, shutdown_executor=True)

    def test_delivery_pool_and_timer_scheduler_guard_post_shutdown_submit(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from src.card.delivery.pool import _ShutdownSafePoolExecutor
        from src.card.timers.scheduler import TimerScheduler

        pool = _ShutdownSafePoolExecutor(max_workers=1, thread_name_prefix="test-pool")
        pool.shutdown(wait=True)
        ran: list[str] = []
        with caplog.at_level(logging.DEBUG):
            fut = pool.submit(lambda: ran.append("degraded") or "ok")
        assert fut.result(timeout=1) == "ok"
        assert ran == ["degraded"]
        assert "delivery pool submit skipped after shutdown" in caplog.text

        timer_sched = TimerScheduler(callback_workers=1, callback_queue_size=4)
        try:
            timer_sched._callback_executor.shutdown(wait=True)
            handle = timer_sched.schedule(0.01, lambda: None, session_id="s-shutdown")
            # Wait briefly for timer thread to attempt submit and hit RuntimeError
            import time as _t

            deadline = _t.monotonic() + 1.0
            while not handle.cancelled and _t.monotonic() < deadline:
                _t.sleep(0.01)
            assert handle.cancelled is True
        finally:
            timer_sched.shutdown(timeout=1.0)

    def test_close_cancels_active_programming_runs_before_scheduler_drain(self) -> None:
        client = _build_minimal_ws_client()
        events: list[str] = []
        prog_session = SimpleNamespace(
            mark_stopping=lambda: events.append("mark_stopping")
        )
        run_session = SimpleNamespace(
            cancel=lambda wait=False: events.append(f"cancel:{wait}")
        )
        coco_handler = client._handler_ctx.handlers["coco"]
        with coco_handler._active_programming_runs_lock:
            coco_handler._active_programming_runs[("oc_1", "p_1", "")] = SimpleNamespace(
                prog_session=prog_session,
                session=run_session,
            )
        client._scheduler.wait_for_idle.side_effect = lambda _timeout: (
            events.append("wait_for_idle"),
            True,
        )[1]
        client._scheduler.wait_for_completion_callbacks.return_value = True

        with patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ):
            assert client.close() is True

        assert events == ["mark_stopping", "cancel:False", "wait_for_idle"]
