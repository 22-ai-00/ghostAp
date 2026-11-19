"""Tests for _HookExecutorManager timeout recovery and executor rebuild."""

import threading
from unittest.mock import MagicMock, patch

from src.card.hooks import _MAX_CONSECUTIVE_TIMEOUTS, _HookExecutorManager


class TestHookExecutorManagerBasic:
    """Basic executor manager operations."""

    def test_lazy_executor_not_created_before_submit(self):
        """Executor stays None until the first submit()."""
        mgr = _HookExecutorManager()
        assert mgr._executor is None
        assert mgr._terminal_executor is None
        mgr.shutdown()

    def test_submit_executes_callable(self):
        """submit() runs the callable in the thread pool."""
        mgr = _HookExecutorManager()
        try:
            result = mgr.submit(lambda: 42)
            assert result is not None
            assert result.result(timeout=5) == 42
        finally:
            mgr.shutdown()

    def test_record_success_resets_counter(self):
        """record_success() resets consecutive timeout counter."""
        mgr = _HookExecutorManager()
        mgr._consecutive_timeouts = 1
        mgr.record_success()
        assert mgr._consecutive_timeouts == 0


class TestHookExecutorTimeout:
    """Timeout tracking and executor rebuild."""

    def test_single_timeout_does_not_rebuild(self):
        """A single timeout increments counter but doesn't rebuild."""
        mgr = _HookExecutorManager()
        try:
            mgr.submit(lambda: None).result(timeout=5)
            original_executor = mgr._executor
            assert original_executor is not None

            mgr.record_timeout()

            assert mgr._consecutive_timeouts == 1
            assert mgr._executor is original_executor
        finally:
            mgr.shutdown()

    def test_max_timeouts_triggers_rebuild(self):
        """Reaching _MAX_CONSECUTIVE_TIMEOUTS rebuilds the executor and semaphore."""
        mgr = _HookExecutorManager()
        try:
            mgr.submit(lambda: None).result(timeout=5)
            original_gen = mgr._generation
            original_executor = mgr._executor
            assert original_gen is not None
            assert original_executor is not None

            for _ in range(_MAX_CONSECUTIVE_TIMEOUTS):
                mgr.record_timeout()

            # Executor and generation semaphore should have been rebuilt
            assert mgr._executor is not None
            assert mgr._executor is not original_executor
            assert mgr._generation is not original_gen
            assert mgr._semaphore is not original_gen.semaphore
            # Counter should be reset
            assert mgr._consecutive_timeouts == 0
        finally:
            mgr.shutdown()

    def test_rebuild_shuts_down_old_executor(self):
        """Old executor is shut down (non-blocking) after rebuild."""
        mgr = _HookExecutorManager()
        try:
            mgr.submit(lambda: None).result(timeout=5)
            old_executor = mgr._executor
            assert old_executor is not None

            with patch.object(old_executor, 'shutdown') as mock_shutdown:
                for _ in range(_MAX_CONSECUTIVE_TIMEOUTS):
                    mgr.record_timeout()
                mock_shutdown.assert_called_once_with(wait=False)
        finally:
            mgr.shutdown()

    def test_success_after_timeout_resets_counter(self):
        """A success after a timeout resets the counter, preventing rebuild."""
        mgr = _HookExecutorManager()
        try:
            mgr.submit(lambda: None).result(timeout=5)
            original_executor = mgr._executor
            assert original_executor is not None

            mgr.record_timeout()
            assert mgr._consecutive_timeouts == 1

            mgr.record_success()
            assert mgr._consecutive_timeouts == 0

            # Another timeout shouldn't rebuild yet
            mgr.record_timeout()
            assert mgr._executor is original_executor
        finally:
            mgr.shutdown()

    def test_thread_safety_of_record_timeout(self):
        """Concurrent record_timeout calls don't corrupt state."""
        mgr = _HookExecutorManager()
        try:
            mgr.submit(lambda: None).result(timeout=5)
            num_threads = 10

            def record_many():
                for _ in range(5):
                    mgr.record_timeout()
                    mgr.record_success()

            threads = [threading.Thread(target=record_many) for _ in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should not crash and counter should be valid
            assert mgr._consecutive_timeouts >= 0
        finally:
            mgr.shutdown()


class TestHookTimeoutEndToEnd:
    """End-to-end test: _fire_hooks_terminal with a slow hook does NOT block the pipeline."""

    def test_slow_hook_does_not_block_pipeline(self, monkeypatch):
        """A hook sleeping longer than HOOK_TIMEOUT_SECONDS should be timed out gracefully."""
        import threading
        import time

        import src.card.hooks as hooks_mod
        from src.card.delivery.engine import CardDelivery
        from src.card.session import CardSession
        from src.card.session.config import SessionConfig
        from src.card.state.models import CardMetadata

        # Speed up test by reducing the timeout constant
        monkeypatch.setattr(hooks_mod, "HOOK_TIMEOUT_SECONDS", 1.0)
        FAST_TIMEOUT = 1.0

        cancel = threading.Event()

        class SlowHook:
            def on_dispatched(self, event, state):
                pass

            def on_terminal(self, state, reason):
                # Wait much longer than timeout (cancellable for fast teardown)
                cancel.wait(timeout=FAST_TIMEOUT + 2)

        client = MagicMock()
        delivery = CardDelivery(client)
        config = SessionConfig(metadata=CardMetadata(engine_type="deep"))
        session = CardSession(
            delivery=delivery,
            chat_id="test_timeout",
            config=config,
            hooks=(SlowHook(),),
        )

        # Dispatch a STARTED event to get state initialized
        from src.card.events import CardEvent, CardEventType
        session.dispatch(CardEvent(type=CardEventType.STARTED, payload={}))

        # Now fire terminal — this should complete within HOOK_TIMEOUT_SECONDS + buffer
        # not the full sleep duration of the hook
        start = time.monotonic()
        session.dispatch(CardEvent(type=CardEventType.COMPLETED, payload={}))
        elapsed = time.monotonic() - start

        # Should return much faster than the hook's sleep
        assert elapsed < FAST_TIMEOUT + 1.5, f"Pipeline blocked for {elapsed:.1f}s"
        cancel.set()  # Allow hook thread to exit immediately
        session.close()


class TestHookExecutorTwoPhaseShutdown:
    """Two-phase shutdown: graceful wait then forced cancel."""

    def test_shutdown_waits_for_inflight_tasks(self):
        """Phase 1 waits for submitted tasks to complete."""
        import time
        results = []

        mgr = _HookExecutorManager()
        # Submit a task that takes a moment
        mgr.submit(lambda: (time.sleep(0.1), results.append("done")))

        # shutdown should wait for it
        mgr.shutdown()
        assert "done" in results

    def test_shutdown_idempotent(self):
        """Calling shutdown() multiple times does not raise."""
        mgr = _HookExecutorManager()
        mgr.shutdown()
        # Second call should not raise
        mgr.shutdown()

    def test_shutdown_phase2_on_exception(self):
        """If shutdown() raises, it's caught and doesn't propagate."""
        from unittest.mock import patch

        mgr = _HookExecutorManager()
        mgr.submit(lambda: None).result(timeout=5)
        original_executor = mgr._executor
        assert original_executor is not None

        # Make shutdown call raise
        def mock_shutdown(wait=False, cancel_futures=True):
            raise RuntimeError("simulated interpreter shutdown")

        try:
            with patch.object(original_executor, 'shutdown', side_effect=mock_shutdown):
                mgr.shutdown()  # Should not raise
        finally:
            original_executor.shutdown(wait=False)
