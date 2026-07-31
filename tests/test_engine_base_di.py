"""Verify BaseEngine accepts injectable settings parameter and thread safety."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from src.engine_base import (
    BaseEngine,
    BaseEngineManager,
    EngineRunState,
    ReviewPerspective,
)


class TestBaseEngineDI:

    def test_injected_settings_used(self):
        mock_settings = MagicMock(name="injected_settings")
        engine = BaseEngine(
            chat_id="c1",
            root_path="/tmp/test",
            settings=mock_settings,
        )
        assert engine.settings is mock_settings

    def test_default_falls_back_to_get_settings(self):
        sentinel = MagicMock(name="global_settings")
        with patch("src.engine_base.get_settings", return_value=sentinel):
            engine = BaseEngine(chat_id="c1", root_path="/tmp/test")
        assert engine.settings is sentinel

    def test_subclass_inherits_injection(self):
        class DummyEngine(BaseEngine):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        mock_settings = MagicMock(name="sub_settings")
        engine = DummyEngine(
            chat_id="c1",
            root_path="/tmp/test",
            settings=mock_settings,
        )
        assert engine.settings is mock_settings


class TestBaseEngineOnStopHook:
    """AC-R18: stop() calls _on_stop() exactly once."""

    def test_stop_calls_on_stop_once(self):
        """Verify that calling stop() invokes _on_stop() exactly once."""
        mock_settings = MagicMock(name="settings")

        class TrackingEngine(BaseEngine):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.on_stop_call_count = 0

            def _on_stop(self):
                self.on_stop_call_count += 1

        engine = TrackingEngine(chat_id="c1", root_path="/tmp/test", settings=mock_settings)
        engine.stop()
        assert engine.on_stop_call_count == 1

    def test_stop_calls_on_stop_even_without_session(self):
        """_on_stop() is called even when no session is set."""
        mock_settings = MagicMock(name="settings")

        class TrackingEngine(BaseEngine):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.on_stop_called = False

            def _on_stop(self):
                self.on_stop_called = True

        engine = TrackingEngine(chat_id="c1", root_path="/tmp/test", settings=mock_settings)
        assert engine._session is None
        engine.stop()
        assert engine.on_stop_called is True

    def test_stop_calls_on_stop_after_session_cancel(self):
        """_on_stop() is called after session.cancel(), not before."""
        mock_settings = MagicMock(name="settings")
        call_order = []

        class OrderTrackingEngine(BaseEngine):
            def _on_stop(self):
                call_order.append("_on_stop")

        engine = OrderTrackingEngine(chat_id="c1", root_path="/tmp/test", settings=mock_settings)
        mock_session = MagicMock()
        mock_session.cancel = lambda: call_order.append("session_cancel")
        engine._session = mock_session

        engine.stop()
        assert call_order == ["session_cancel", "_on_stop"]

    def test_multiple_stops_call_on_stop_each_time(self):
        """Calling stop() multiple times calls _on_stop() each time."""
        mock_settings = MagicMock(name="settings")

        class TrackingEngine(BaseEngine):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.on_stop_call_count = 0

            def _on_stop(self):
                self.on_stop_call_count += 1

        engine = TrackingEngine(chat_id="c1", root_path="/tmp/test", settings=mock_settings)
        engine.stop()
        engine.stop()
        assert engine.on_stop_call_count == 2


class TestConcurrentStopCleanup:
    """Concurrent stop()+cleanup() must not corrupt state or raise."""

    def test_concurrent_stop_cleanup_no_error(self):
        """Fire stop() and cleanup() from two threads simultaneously."""
        mock_settings = MagicMock(name="settings")
        errors: list[Exception] = []

        class SlowEngine(BaseEngine):
            def _on_stop(self):
                pass

        for _ in range(50):
            engine = SlowEngine(chat_id="c1", root_path="/tmp/test", settings=mock_settings)
            engine._run_state = EngineRunState.RUNNING
            mock_session = MagicMock()
            engine._session = mock_session

            barrier = threading.Barrier(2)

            def do_stop():
                try:
                    barrier.wait(timeout=2)
                    engine.stop()
                except Exception as exc:
                    errors.append(exc)

            def do_cleanup():
                try:
                    barrier.wait(timeout=2)
                    engine.cleanup()
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=do_stop)
            t2 = threading.Thread(target=do_cleanup)
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

        assert not errors, f"Concurrent stop/cleanup raised: {errors}"

    def test_cleanup_idle_closes_session(self):
        """cleanup() on IDLE engine closes (not cancels) the session."""
        mock_settings = MagicMock(name="settings")
        engine = BaseEngine(chat_id="c1", root_path="/tmp/test", settings=mock_settings)
        mock_session = MagicMock()
        engine._session = mock_session

        engine.cleanup()

        mock_session.close.assert_called_once()
        mock_session.cancel.assert_not_called()

    def test_cleanup_running_cancels_session(self):
        """cleanup() on RUNNING engine cancels (not closes) the session."""
        mock_settings = MagicMock(name="settings")
        engine = BaseEngine(chat_id="c1", root_path="/tmp/test", settings=mock_settings)
        engine._run_state = EngineRunState.RUNNING
        mock_session = MagicMock()
        engine._session = mock_session

        engine.cleanup()

        mock_session.cancel.assert_called_once()
        mock_session.close.assert_not_called()


class TestLockOrderingViolation:
    """Verify ordered_rlock detects out-of-order acquisition."""

    def test_reverse_order_logs_warning(self):
        """Acquiring a lower-level lock while holding a higher-level one triggers warning."""
        from src.utils.lock_order import (
            LockLevel,
            disable_lock_order_check,
            enable_lock_order_check,
            ordered_rlock,
        )

        enable_lock_order_check()
        try:
            outer = ordered_rlock(LockLevel.REPO_LOCK, name="test_inner")
            inner = ordered_rlock(LockLevel.ENGINE_INSTANCE, name="test_outer")

            with patch("src.utils.lock_order.logger") as mock_logger:
                with outer:
                    with inner:
                        pass
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args
                assert "Lock ordering violation" in call_args[0][0]
        finally:
            disable_lock_order_check()

    def test_correct_order_no_warning(self):
        """Acquiring locks in correct ascending order does not trigger warning."""
        from src.utils.lock_order import (
            LockLevel,
            disable_lock_order_check,
            enable_lock_order_check,
            ordered_rlock,
        )

        enable_lock_order_check()
        try:
            outer = ordered_rlock(LockLevel.ENGINE_INSTANCE, name="test_correct_outer")
            inner = ordered_rlock(LockLevel.REPO_LOCK, name="test_correct_inner")

            with patch("src.utils.lock_order.logger") as mock_logger:
                with outer:
                    with inner:
                        pass
                mock_logger.warning.assert_not_called()
        finally:
            disable_lock_order_check()


class TestRunStateUnderLock:
    """AC-T20: _run_state transitions are always protected by _lock.

    Verifies that concurrent stop() calls during simulated execute/resume
    finally-blocks do not corrupt _run_state.
    """

    def test_concurrent_stop_during_finally_idle_transition(self):
        """Many threads calling stop() while one thread sets IDLE in finally."""
        mock_settings = MagicMock(name="settings")
        errors: list[str] = []

        for _ in range(30):
            engine = BaseEngine(chat_id="c1", root_path="/tmp/test", settings=mock_settings)
            with engine._lock:
                engine._run_state = EngineRunState.RUNNING

            barrier = threading.Barrier(3)

            def do_stop():
                try:
                    barrier.wait(timeout=2)
                    engine.stop()
                except Exception as exc:
                    errors.append(str(exc))

            def do_finally():
                try:
                    barrier.wait(timeout=2)
                    with engine._lock:
                        engine._run_state = EngineRunState.IDLE
                except Exception as exc:
                    errors.append(str(exc))

            threads = [
                threading.Thread(target=do_stop),
                threading.Thread(target=do_stop),
                threading.Thread(target=do_finally),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            # After all complete, state must be either IDLE or STOPPING (not corrupt)
            state = engine.run_state
            assert state in (EngineRunState.IDLE, EngineRunState.STOPPING), (
                f"Unexpected state: {state}"
            )

        assert not errors, f"Errors: {errors}"

    def test_run_state_property_is_atomic(self):
        """run_state property reads under lock — concurrent writes don't tear."""
        mock_settings = MagicMock(name="settings")
        engine = BaseEngine(chat_id="c1", root_path="/tmp/test", settings=mock_settings)
        results: list[EngineRunState] = []
        stop_event = threading.Event()

        def writer():
            states = [EngineRunState.RUNNING, EngineRunState.IDLE, EngineRunState.STOPPING]
            i = 0
            while not stop_event.is_set():
                with engine._lock:
                    engine._run_state = states[i % 3]
                i += 1

        def reader():
            for _ in range(200):
                state = engine.run_state
                results.append(state)

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_read.join(timeout=5)
        stop_event.set()
        t_write.join(timeout=5)

        valid_states = {EngineRunState.RUNNING, EngineRunState.IDLE, EngineRunState.STOPPING}
        for s in results:
            assert s in valid_states, f"Invalid/torn state: {s}"


# ---------------------------------------------------------------------------
# Step-04/05: ReviewPerspective display_name injection
# ---------------------------------------------------------------------------


class TestReviewPerspectiveDisplayName:
    """AC-R01/AC-R02: display_name injection and degradation."""

    def test_degrade_without_injection(self):
        """Before injection, display_name returns the raw enum value."""
        saved = dict(ReviewPerspective._display_names)
        ReviewPerspective._display_names.clear()
        try:
            assert ReviewPerspective.ARCHITECT.display_name == "architect"
            assert ReviewPerspective.PRODUCT.display_name == "product"
        finally:
            ReviewPerspective._display_names.update(saved)

    def test_injection_returns_chinese(self):
        """After injection, display_name returns localised text."""
        saved = dict(ReviewPerspective._display_names)
        ReviewPerspective._display_names.clear()
        ReviewPerspective.register_display_names({"perspective_architect": "架构师"})
        try:
            assert ReviewPerspective.ARCHITECT.display_name == "架构师"
            # Non-injected member still degrades
            assert ReviewPerspective.USER.display_name == "user"
        finally:
            ReviewPerspective._display_names.clear()
            ReviewPerspective._display_names.update(saved)

    def test_no_spec_engine_import_in_engine_base(self):
        """AC-R01: engine_base.py must not import from spec_engine."""
        import pathlib
        import re
        content = pathlib.Path("src/engine_base.py").read_text()
        # Match actual import statements referencing spec_engine
        imports = re.findall(r"^\s*(from\s+\S*spec_engine|import\s+\S*spec_engine)", content, re.MULTILINE)
        assert imports == [], (
            f"engine_base.py still imports spec_engine: {imports}"
        )


def test_cleanup_all_retains_engine_that_has_not_quiesced():
    """An explicit False cleanup result is a manager tombstone."""

    class DummyManager(BaseEngineManager[BaseEngine]):
        def _create_engine(
            self,
            chat_id,
            root_path,
            agent_type,
            engine_name,
            model_name,
        ):
            raise AssertionError("test installs its engine directly")

    manager = DummyManager()
    engine = MagicMock()
    engine.cleanup.return_value = False
    engine.is_running = False
    key = "chat_1:/repo"
    manager._engines[key] = engine
    manager._chat_keys["chat_1"] = {key}

    manager.cleanup_all()

    assert manager.get("chat_1", "/repo") is engine
    assert manager.list_engines("chat_1") == [engine]
