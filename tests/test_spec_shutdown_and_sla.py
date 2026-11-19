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


_VALID_SPEC_JSON = """{
  "goal": "Implement feature",
  "acceptance_criteria": ["passes tests"],
  "required_experts": [
    {"role": "Architect", "purpose": "Design boundaries", "focus": ["contracts"], "checks": ["boundary check"]},
    {"role": "QA Engineer", "purpose": "Verify regressions", "focus": ["tests"], "checks": ["test check"]},
    {"role": "Security Reviewer", "purpose": "Check safety", "focus": ["auth"], "checks": ["auth check"]}
  ]
}"""

_VALID_PLAN_JSON = """{
  "summary": "Plan",
  "steps": ["Step 1"]
}"""


def _make_sla_settings(tmp_path, **overrides) -> SimpleNamespace:
    base = dict(
        spec_max_cycles=3,
        spec_max_cycles_limit=10,
        spec_execution_timeout=10,
        acp_startup_timeout=30,
        spec_convergence_window=2,
        spec_min_cycles=1,
        spec_review_enabled=False,
        spec_review_timeout=120,
        spec_review_role_timeout_multipliers={},
        spec_review_max_parallel=2,
        spec_review_total_roles_max=4,
        spec_review_retry_max_attempts=1,
        spec_review_retry_base_delay=0.01,
        spec_review_retry_max_delay=0.05,
        spec_review_startup_timeout=30,
        spec_review_failure_circuit_enabled=False,
        spec_completion_gate_enabled=True,
        spec_discovery_enabled=False,
        spec_discovery_max_questions=5,
        spec_discovery_force_nonempty=False,
        spec_generated_specs_per_cycle=2,
        spec_discovery_gate_on_satisfied=True,
        spec_discovery_max_pending=5,
        spec_discovery_cooldown_cycles=3,
        spec_persist_phase_artifacts=False,
        spec_persist_every_phase=False,
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
        spec_objective_verify_timeout=300,
        spec_rebuild_session_between_cycles=False,
        spec_success_ignore_backlog=True,
        engine_eval_prompt_timeout=90,
        spec_cache_root=str(tmp_path / "cache"),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


def _patch_sla_clocks(monkeypatch, clock: _FakeClock) -> None:
    for target in (
        "src.spec_engine.engine.time.monotonic",
        "src.spec_engine.criteria.time.monotonic",
        "src.spec_engine.discovery.time.monotonic",
        "src.spec_engine.review.time.monotonic",
    ):
        monkeypatch.setattr(target, clock.monotonic)


class TestSpecMonotonicTotalSLA:
    """B090: monotonic SPEC_EXECUTION_TIMEOUT SLA across phases, retries, verify, and review."""

    def test_total_sla_propagates_without_cycle_digestion_or_extra_cycles(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.acp import ACPEvent, ACPEventType
        from src.spec_engine.engine import SpecEngine, SpecEngineCallbacks
        from src.spec_engine.models import SpecProjectStatus
        from src.utils.retry import RetryPolicy

        clock = _FakeClock(1000.0)
        _patch_sla_clocks(monkeypatch, clock)
        settings = _make_sla_settings(tmp_path, spec_execution_timeout=10, spec_max_cycles=5)
        monkeypatch.setattr("src.engine_base.get_settings", lambda: settings)

        seen_timeouts: list[float] = []

        class _SlowPlanSession:
            def send_prompt(self, _prompt, on_event=None, timeout=None):
                return SimpleNamespace(stop_reason="end_turn")

            def send_prompt_with_retry(self, _prompt, on_event=None, timeout=None, **_kw):
                seen_timeouts.append(float(timeout))
                if len(seen_timeouts) == 1:
                    clock.advance(4.0)
                    if on_event:
                        on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=_VALID_SPEC_JSON))
                    return SimpleNamespace(stop_reason="end_turn")
                clock.advance(6.5)
                raise TimeoutError("plan phase timed out")

            def close(self):
                return None

        cycle_starts: list[int] = []
        errors: list[str] = []
        engine = SpecEngine(
            chat_id="oc_sla_prop",
            root_path=str(tmp_path),
            retry_policy=RetryPolicy(max_retries=0, retry_delay=0.0, backoff_multiplier=1.0),
            create_session_fn=lambda **_kw: _SlowPlanSession(),
        )
        project = engine.execute(
            "build feature",
            callbacks=SpecEngineCallbacks(
                on_cycle_start=lambda c, _m: cycle_starts.append(c),
                on_error=errors.append,
            ),
        )

        assert project.status == SpecProjectStatus.FAILED
        assert "Spec执行超时" in (project.error or "")
        assert any("Spec执行超时" in msg for msg in errors)
        assert cycle_starts == [1]
        assert len(project.cycles) == 1
        assert seen_timeouts[0] == pytest.approx(6.0, abs=0.01)
        assert seen_timeouts[1] == pytest.approx(6.0, abs=0.01)
        assert engine.sla_deadline is None

    def test_phase_timeout_while_sla_remaining_still_digests_cycle(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.acp import ACPEvent, ACPEventType
        from src.spec_engine.engine import SpecEngine
        from src.spec_engine.models import SpecProjectStatus
        from src.utils.retry import RetryPolicy

        clock = _FakeClock(1000.0)
        _patch_sla_clocks(monkeypatch, clock)
        settings = _make_sla_settings(tmp_path, spec_execution_timeout=300, spec_max_cycles=2)
        monkeypatch.setattr("src.engine_base.get_settings", lambda: settings)

        calls = 0

        class _TransientPhaseTimeoutSession:
            def send_prompt_with_retry(self, _prompt, on_event=None, timeout=None, **_kw):
                nonlocal calls
                calls += 1
                if calls == 1:
                    clock.advance(0.1)
                    if on_event:
                        on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=_VALID_SPEC_JSON))
                    return SimpleNamespace(stop_reason="end_turn")
                if calls == 2:
                    raise TimeoutError("transient single phase timeout")
                payloads = {
                    3: _VALID_SPEC_JSON,
                    4: _VALID_PLAN_JSON,
                    5: "1. T1",
                    6: "built",
                }
                if on_event:
                    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=payloads[calls]))
                return SimpleNamespace(stop_reason="end_turn")

            def close(self):
                return None

        engine = SpecEngine(
            chat_id="oc_digest",
            root_path=str(tmp_path),
            retry_policy=RetryPolicy(max_retries=0, retry_delay=0.0, backoff_multiplier=1.0),
            create_session_fn=lambda **_kw: _TransientPhaseTimeoutSession(),
        )
        monkeypatch.setattr(engine, "_try_switch_model", lambda *_a, **_kw: False)
        monkeypatch.setattr(engine, "_recreate_session_best_effort", lambda: None)
        monkeypatch.setattr(engine, "_evaluate_criteria", lambda *_a, **_kw: {"all_satisfied": True})
        project = engine.execute("build feature")

        assert len(project.cycles) == 2
        assert project.status == SpecProjectStatus.COMPLETED

    def test_one_second_sla_keeps_float_phase_budgets_and_caps_build(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.acp import ACPEvent, ACPEventType
        from src.spec_engine.engine import SpecEngine
        from src.spec_engine.models import SpecProjectStatus
        from src.utils.retry import RetryPolicy

        clock = _FakeClock(1000.0)
        _patch_sla_clocks(monkeypatch, clock)
        settings = _make_sla_settings(
            tmp_path,
            spec_execution_timeout=1,
            acp_startup_timeout=30,
            spec_max_cycles=1,
        )
        monkeypatch.setattr("src.engine_base.get_settings", lambda: settings)

        startup_timeouts: list[float] = []
        phase_timeouts: list[float] = []

        class _OneSecondSession:
            def __init__(self) -> None:
                self._n = 0

            def send_prompt_with_retry(self, _prompt, on_event=None, timeout=None, **_kw):
                self._n += 1
                phase_timeouts.append(float(timeout))
                clock.advance(0.05)
                outputs = [_VALID_SPEC_JSON, _VALID_PLAN_JSON, "1. T1", "built"]
                if on_event:
                    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=outputs[self._n - 1]))
                return SimpleNamespace(stop_reason="end_turn")

            def close(self):
                return None

        def _create_session(**kw):
            startup_timeouts.append(float(kw["startup_timeout"]))
            return _OneSecondSession()

        engine = SpecEngine(
            chat_id="oc_one_sec",
            root_path=str(tmp_path),
            retry_policy=RetryPolicy(max_retries=0, retry_delay=0.0, backoff_multiplier=1.0),
            create_session_fn=_create_session,
        )
        monkeypatch.setattr(engine, "_evaluate_criteria", lambda *_a, **_kw: {"all_satisfied": True})
        project = engine.execute("tiny sla")

        assert project.status == SpecProjectStatus.COMPLETED
        assert len(startup_timeouts) == 1
        assert 0.0 < startup_timeouts[0] <= 1.0
        assert len(phase_timeouts) == 4
        # task phase (0.5 * 1) must NOT truncate to int(0) == infinite, and build must be <= 1.0
        assert all(t > 0.0 and t <= 1.0 for t in phase_timeouts)
        assert phase_timeouts[2] <= 0.5
        assert phase_timeouts[3] <= 1.0

    def test_retry_policy_and_spec_repair_round_clamp_against_remaining_sla(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.acp import ACPEvent, ACPEventType
        from src.spec_engine.engine import SpecEngine
        from src.spec_engine.models import SpecProjectStatus
        from src.utils.retry import RetryPolicy

        clock = _FakeClock(1000.0)
        _patch_sla_clocks(monkeypatch, clock)
        settings = _make_sla_settings(
            tmp_path, spec_execution_timeout=10, spec_max_retries=2, spec_max_cycles=1
        )
        monkeypatch.setattr("src.engine_base.get_settings", lambda: settings)

        calls: list[dict[str, float]] = []
        recreate_called = 0
        switch_called = 0

        class _RepairRetrySession:
            def send_prompt_with_retry(
                self, _prompt, on_event=None, timeout=None, retry_policy=None, before_retry=None, total_timeout=None
            ):
                calls.append(
                    {
                        "timeout": float(timeout),
                        "policy_total": float(retry_policy.total_timeout),
                        "total_timeout": float(total_timeout),
                    }
                )
                if len(calls) == 1:
                    # Missing required_experts -> triggers SPEC repair round after spending 6.0s
                    clock.advance(6.0)
                    if on_event:
                        on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text='{"goal": "x"}'))
                    return SimpleNamespace(stop_reason="end_turn")
                # Repair round: spend remaining 4.5s and invoke before_retry then fail
                clock.advance(4.5)
                if before_retry is not None:
                    before_retry(1, TimeoutError("repair timeout"))
                raise TimeoutError("repair failed after SLA")

            def close(self):
                return None

        engine = SpecEngine(
            chat_id="oc_repair_sla",
            root_path=str(tmp_path),
            retry_policy=RetryPolicy(max_retries=2, retry_delay=0.0, backoff_multiplier=1.0),
            create_session_fn=lambda **_kw: _RepairRetrySession(),
        )

        def _fake_recreate(*_a, **_kw):
            nonlocal recreate_called
            recreate_called += 1

        def _fake_switch(*_a, **_kw):
            nonlocal switch_called
            switch_called += 1
            return False

        monkeypatch.setattr(engine, "_recreate_session_best_effort", _fake_recreate)
        monkeypatch.setattr(engine, "_try_switch_model", _fake_switch)

        project = engine.execute("repair feature")

        assert len(calls) == 2
        assert calls[1]["timeout"] == pytest.approx(4.0, abs=0.01)
        assert calls[1]["policy_total"] == pytest.approx(4.0, abs=0.01)
        assert calls[1]["total_timeout"] == pytest.approx(4.0, abs=0.01)
        assert recreate_called == 0
        assert switch_called == 0
        assert project.status == SpecProjectStatus.FAILED

    def test_objective_verify_and_command_executor_clamp_to_remaining_sla(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.acp import ACPEvent, ACPEventType
        from src.command_executor import CommandExecutor
        from src.spec_engine.engine import SpecEngine
        from src.spec_engine.models import SpecProjectStatus
        from src.utils.retry import RetryPolicy

        seen_subproc_timeouts: list[float] = []

        class _FakeSubprocessExecutor:
            def run(self, _cmd, **kwargs):
                seen_subproc_timeouts.append(float(kwargs["timeout"]))
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        executor = CommandExecutor(
            SimpleNamespace(command_timeout=300, command_max_output_length=4000),
            subprocess_executor=_FakeSubprocessExecutor(),
        )
        exec_res = executor.execute("echo ok", cwd=str(tmp_path), timeout=1.25)
        assert exec_res.success is True
        assert seen_subproc_timeouts == [pytest.approx(1.25)]

        clock = _FakeClock(1000.0)
        _patch_sla_clocks(monkeypatch, clock)
        settings = _make_sla_settings(
            tmp_path,
            spec_execution_timeout=10,
            spec_objective_verify_enabled=True,
            spec_objective_verify_timeout=300,
            spec_review_enabled=True,
        )
        monkeypatch.setattr("src.engine_base.get_settings", lambda: settings)

        class _FourPhaseSession:
            def __init__(self) -> None:
                self._n = 0

            def send_prompt_with_retry(self, _prompt, on_event=None, timeout=None, **_kw):
                self._n += 1
                outputs = [_VALID_SPEC_JSON, _VALID_PLAN_JSON, "1. T1", "built"]
                if self._n == 4:
                    clock.advance(7.5)
                if on_event:
                    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=outputs[self._n - 1]))
                return SimpleNamespace(stop_reason="end_turn")

            def close(self):
                return None

        verify_timeouts: list[float] = []
        review_started = False

        def _fake_objective_verify(_cmd, *, cwd: str, timeout: float = 300):
            verify_timeouts.append(float(timeout))
            clock.advance(3.0)
            return True, "ok"

        engine = SpecEngine(
            chat_id="oc_verify_sla",
            root_path=str(tmp_path),
            retry_policy=RetryPolicy(max_retries=0, retry_delay=0.0, backoff_multiplier=1.0),
            create_session_fn=lambda **_kw: _FourPhaseSession(),
        )
        monkeypatch.setattr("src.spec_engine.criteria.run_objective_verify", _fake_objective_verify)

        orig_review = engine._run_review_phase

        def _tracked_review(*args, **kwargs):
            nonlocal review_started
            review_started = True
            return orig_review(*args, **kwargs)

        monkeypatch.setattr(engine, "_run_review_phase", _tracked_review)
        # Ensure project has verify_command once initialized
        orig_init = engine._initialize_model_context

        def _set_verify():
            orig_init()
            engine._project.verify_command = "pytest -q"

        monkeypatch.setattr(engine, "_initialize_model_context", _set_verify)
        project = engine.execute("verify clamped")

        assert verify_timeouts == [pytest.approx(2.5, abs=0.01)]
        assert review_started is False
        assert project.status == SpecProjectStatus.FAILED

    def test_criteria_and_discovery_clamp_and_propagate_total_timeout(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.spec_engine.criteria import evaluate_criteria
        from src.spec_engine.discovery import (
            discover_optimization_questions,
            generate_specs_from_discovery,
        )
        from src.spec_engine.models import SpecProject
        from src.spec_engine.session_utils import SpecTotalTimeoutError

        clock = _FakeClock(1000.0)
        _patch_sla_clocks(monkeypatch, clock)
        settings = _make_sla_settings(tmp_path, engine_eval_prompt_timeout=90)
        project = SpecProject.create(name="demo", root_path=str(tmp_path))

        # Expired at entry fails immediately without issuing prompts
        called = 0

        def _should_not_run(*_a, **_kw):
            nonlocal called
            called += 1
            return SimpleNamespace(stop_reason="end_turn")

        with pytest.raises(SpecTotalTimeoutError):
            evaluate_criteria(
                session=object(),
                criteria=["c1"],
                cycle=1,
                project=project,
                send_prompt_fn=_should_not_run,
                settings=settings,
                sla_deadline=999.0,
            )
        with pytest.raises(SpecTotalTimeoutError):
            discover_optimization_questions(
                project=project,
                session=object(),
                send_prompt_fn=_should_not_run,
                last_review=None,
                cycle_num=1,
                settings=settings,
                sla_deadline=999.0,
            )
        with pytest.raises(SpecTotalTimeoutError):
            generate_specs_from_discovery(
                project=project,
                session=object(),
                send_prompt_fn=_should_not_run,
                root_path=str(tmp_path),
                settings=settings,
                cycle_num=1,
                discovery=[{"question": "q", "priority": "high"}],
                sla_deadline=999.0,
            )
        assert called == 0

        # Active deadline clamps prompt timeout and propagates SpecTotalTimeoutError cleanly
        recorded_timeouts: list[float] = []

        def _timeout_after_expiry(_prompt, on_event=None, timeout=None, **_kw):
            recorded_timeouts.append(float(timeout))
            clock.advance(2.0)
            raise TimeoutError("prompt timed out")

        for fn_call in (
            lambda: evaluate_criteria(
                session=object(),
                criteria=["c1"],
                cycle=1,
                project=project,
                send_prompt_fn=_timeout_after_expiry,
                settings=settings,
                sla_deadline=clock.now + 1.5,
            ),
            lambda: discover_optimization_questions(
                project=project,
                session=object(),
                send_prompt_fn=_timeout_after_expiry,
                last_review=None,
                cycle_num=1,
                settings=settings,
                sla_deadline=clock.now + 1.5,
            ),
            lambda: generate_specs_from_discovery(
                project=project,
                session=object(),
                send_prompt_fn=_timeout_after_expiry,
                root_path=str(tmp_path),
                settings=settings,
                cycle_num=1,
                discovery=[{"question": "q", "priority": "high"}],
                sla_deadline=clock.now + 1.5,
            ),
        ):
            with pytest.raises(SpecTotalTimeoutError):
                fn_call()

        assert recorded_timeouts == [
            pytest.approx(1.5),
            pytest.approx(1.5),
            pytest.approx(1.5),
        ]

    def test_parallel_review_sla_expiration_cancels_sessions_and_fails_run(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.acp import ACPEvent, ACPEventType
        from src.spec_engine.engine import SpecEngine, SpecEngineCallbacks
        from src.spec_engine.models import SpecProjectStatus
        from src.utils.retry import RetryPolicy

        cancelled_sessions: list[bool] = []
        session_entered = threading.Event()

        class _MainFastSession:
            def __init__(self) -> None:
                self._n = 0

            def send_prompt_with_retry(self, _prompt, on_event=None, timeout=None, **_kw):
                self._n += 1
                outputs = [_VALID_SPEC_JSON, _VALID_PLAN_JSON, "1. T1", "built"]
                if on_event:
                    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=outputs[self._n - 1]))
                return SimpleNamespace(stop_reason="end_turn")

            def close(self):
                return None

        class _HungReviewSession:
            def __init__(self) -> None:
                self._ev = threading.Event()

            def send_prompt(self, _prompt, on_event=None, timeout=None):
                session_entered.set()
                self._ev.wait(timeout=3.0)
                raise TimeoutError("review cancelled by SLA")

            def cancel(self, wait=False, timeout=2.0):
                cancelled_sessions.append(True)
                self._ev.set()
                return True

        class _FakeEphemeralReview:
            def __init__(self, *_a, **_kw) -> None:
                self.session_started = True
                self.startup_elapsed_s = 0.01
                self._session = _HungReviewSession()

            def __enter__(self):
                return self._session

            def __exit__(self, *_exc):
                return None

        settings = _make_sla_settings(
            tmp_path,
            spec_execution_timeout=0.35,
            spec_review_enabled=True,
            spec_review_timeout=120,
            spec_review_max_parallel=2,
            spec_review_retry_max_attempts=1,
            spec_max_cycles=3,
        )
        monkeypatch.setattr("src.engine_base.get_settings", lambda: settings)
        monkeypatch.setattr("src.spec_engine.review.EphemeralReviewSession", _FakeEphemeralReview)

        errors: list[str] = []
        engine = SpecEngine(
            chat_id="oc_review_sla",
            root_path=str(tmp_path),
            retry_policy=RetryPolicy(max_retries=0, retry_delay=0.0, backoff_multiplier=1.0),
            create_session_fn=lambda **_kw: _MainFastSession(),
        )
        project = engine.execute(
            "review sla feature",
            callbacks=SpecEngineCallbacks(on_error=errors.append),
        )

        assert session_entered.is_set()
        assert cancelled_sessions
        assert project.status == SpecProjectStatus.FAILED
        assert "Spec执行超时" in (project.error or "")
        assert any("Spec执行超时" in err for err in errors)

    def test_spec_stream_processor_rollover_ttl_clamped_to_remaining_sla(self) -> None:
        import time as _time

        from src.card.render.budget import RenderBudget
        from src.card.state.models import CardMetadata
        from src.feishu.renderers._spec_stream_processor import SpecStreamProcessor

        created_ttls: list[float] = []
        engine_holder: list[object] = [SimpleNamespace(sla_deadline=_time.monotonic() + 18.0)]

        renderer = MagicMock()
        renderer.settings = SimpleNamespace(
            spec_execution_timeout=600,
            streaming_flush_interval=0.5,
            streaming_max_chars=4000,
            feishu_card_max_bytes=20000,
        )
        renderer.ctx = SimpleNamespace(
            spec_engine_manager=SimpleNamespace(get=lambda _c, _r: engine_holder[0])
        )

        def _fake_create_session(_chat_id, _msg_id, meta, **_kw):
            created_ttls.append(float(_kw["ttl_seconds"]))
            return MagicMock(session_started_at=meta.session_started_at)

        renderer.create_session.side_effect = _fake_create_session
        initial_session = MagicMock(session_started_at=123.0)
        rotator = MagicMock(current=initial_session)
        meta = CardMetadata(project_name="Spec", session_started_at=123.0)

        proc = SpecStreamProcessor(
            rotator=rotator,
            reporter=MagicMock(),
            metadata=meta,
            hooks=(),
            budget=RenderBudget(),
            spec_project_id="proj-1",
            message_id="om_1",
            chat_id="oc_roll",
            renderer=renderer,
            project_root_path="/tmp/proj",
        )
        next_meta = CardMetadata(project_name="Spec", session_started_at=123.0)
        proc._rotator._session_factory(next_meta)
        assert len(created_ttls) == 1
        assert 1.0 < created_ttls[0] <= 19.0

        engine_holder[0] = SimpleNamespace(sla_deadline=_time.monotonic() - 5.0)
        proc._rotator._session_factory(next_meta)
        assert created_ttls[1] == pytest.approx(1.0)
