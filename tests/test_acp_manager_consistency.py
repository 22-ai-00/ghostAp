import asyncio
import concurrent.futures
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.acp.manager import ACPSessionManager
from src.acp.sync_adapter import SyncACPSession


class TestACPSessionManagerConsistency(unittest.TestCase):
    def test_retire_session_closes_expected_session_before_returning(self):
        manager = ACPSessionManager(agent_type="coco")
        session = MagicMock()
        session.session_id = "retire-session"
        session.message_count = 2
        session.to_snapshot.return_value = {"session_id": session.session_id}
        key = manager._session_key("chat-retire")
        manager._sessions[key] = session
        retire = getattr(manager, "retire_session", None)

        self.assertTrue(callable(retire), "synchronous retirement is missing")
        snapshot = retire(
            "chat-retire",
            expected_session=session,
        )

        self.assertEqual(snapshot, {"session_id": "retire-session"})
        self.assertNotIn(key, manager._sessions)
        session.close.assert_called_once_with()

    def test_retire_session_does_not_remove_concurrent_replacement(self):
        manager = ACPSessionManager(agent_type="coco")
        timed_out = MagicMock()
        timed_out.session_id = "old"
        replacement = MagicMock()
        replacement.session_id = "new"
        key = manager._session_key("chat-retire-race")
        manager._sessions[key] = replacement
        retire = getattr(manager, "retire_session", None)

        self.assertTrue(callable(retire), "identity-safe retirement is missing")
        snapshot = retire(
            "chat-retire-race",
            expected_session=timed_out,
        )

        self.assertIsNone(snapshot)
        self.assertIs(manager._sessions[key], replacement)
        timed_out.close.assert_not_called()
        replacement.close.assert_not_called()

    def test_retire_session_snapshot_failure_does_not_block_recovery(self):
        manager = ACPSessionManager(agent_type="coco")
        session = MagicMock()
        session.session_id = "snapshot-failure"
        session.message_count = 2
        session.to_snapshot.side_effect = ValueError("snapshot unavailable")
        key = manager._session_key("chat-retire-snapshot-failure")
        manager._sessions[key] = session

        snapshot = manager.retire_session(
            "chat-retire-snapshot-failure",
            expected_session=session,
        )

        self.assertIsNone(snapshot)
        self.assertNotIn(key, manager._sessions)
        session.close.assert_called_once_with()

        replacement = MagicMock()
        with patch.object(
            manager,
            "start_session",
            return_value=replacement,
        ):
            self.assertIs(
                manager.ensure_session(
                    "chat-retire-snapshot-failure",
                    cwd="/tmp",
                ),
                replacement,
            )

    def test_retire_session_close_is_bounded_by_shared_timeout(self):
        manager = ACPSessionManager(agent_type="coco")
        close_started = threading.Event()
        allow_close = threading.Event()

        class BlockingCloseSession:
            session_id = "blocking-close"
            message_count = 1

            def to_snapshot(self):
                return {"session_id": self.session_id}

            def close(self):
                close_started.set()
                allow_close.wait(timeout=2)

        session = BlockingCloseSession()
        key = manager._session_key("chat-bounded-retire")
        manager._sessions[key] = session
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(TimeoutError, "close timeout"):
                manager.retire_session(
                    "chat-bounded-retire",
                    expected_session=session,
                    timeout=0.05,
                )
            elapsed = time.monotonic() - started
            self.assertTrue(close_started.wait(timeout=0.2))
            self.assertLess(elapsed, 0.3)
            self.assertNotIn(key, manager._sessions)
        finally:
            allow_close.set()

    def test_new_session_waits_for_timed_out_close_to_really_finish(self):
        manager = ACPSessionManager(agent_type="coco")
        close_started = threading.Event()
        allow_close = threading.Event()

        class BlockingCloseSession:
            session_id = "still-closing"
            message_count = 1

            def to_snapshot(self):
                return {"session_id": self.session_id}

            def close(self):
                close_started.set()
                allow_close.wait(timeout=2)

        session = BlockingCloseSession()
        key = manager._session_key("chat-close-gate")
        manager._sessions[key] = session
        try:
            with self.assertRaisesRegex(TimeoutError, "close timeout"):
                manager.retire_session(
                    "chat-close-gate",
                    expected_session=session,
                    timeout=0.05,
                )
            self.assertTrue(close_started.wait(timeout=0.2))

            with (
                patch.object(manager, "_start_session_inner") as start_inner,
                self.assertRaisesRegex(TimeoutError, "closing"),
            ):
                manager.start_session(
                    "chat-close-gate",
                    cwd="/tmp",
                    startup_timeout=0.05,
                )
            start_inner.assert_not_called()

            allow_close.set()
            replacement = MagicMock()
            with patch.object(
                manager,
                "_start_session_inner",
                return_value=replacement,
            ) as start_inner:
                observed = manager.start_session(
                    "chat-close-gate",
                    cwd="/tmp",
                    startup_timeout=1,
                )
            self.assertIs(observed, replacement)
            start_inner.assert_called_once()
        finally:
            allow_close.set()

    def test_start_rechecks_close_gate_after_concurrent_normal_end(self):
        first_gate_checked = threading.Event()
        allow_start_to_continue = threading.Event()
        close_started = threading.Event()
        allow_close = threading.Event()
        starter_called = threading.Event()
        start_errors: list[BaseException] = []

        def starter(**_kwargs):
            starter_called.set()
            raise AssertionError("backend must not start while old close is running")

        manager = ACPSessionManager(
            agent_type="coco",
            session_starter=starter,
        )

        class BlockingCloseSession:
            session_id = "normal-end-race"
            message_count = 1

            def to_snapshot(self):
                return {"session_id": self.session_id}

            def close(self):
                close_started.set()
                allow_close.wait(timeout=2)

        key = manager._session_key("chat-normal-end-race")
        manager._sessions[key] = BlockingCloseSession()
        original_wait = manager._wait_for_closing_session
        wait_calls = 0

        def wait_with_barrier(*args, **kwargs):
            nonlocal wait_calls
            wait_calls += 1
            result = original_wait(*args, **kwargs)
            if wait_calls == 1:
                first_gate_checked.set()
                allow_start_to_continue.wait(timeout=1)
            return result

        def start() -> None:
            try:
                manager.start_session(
                    "chat-normal-end-race",
                    cwd="/tmp",
                    startup_timeout=0.15,
                )
            except BaseException as exc:
                start_errors.append(exc)

        with patch.object(
            manager,
            "_wait_for_closing_session",
            side_effect=wait_with_barrier,
        ):
            worker = threading.Thread(target=start)
            worker.start()
            try:
                self.assertTrue(first_gate_checked.wait(timeout=0.2))
                manager.end_session("chat-normal-end-race")
                self.assertTrue(close_started.wait(timeout=0.2))
                allow_start_to_continue.set()
                worker.join(timeout=1)
                self.assertFalse(worker.is_alive())
                self.assertEqual(len(start_errors), 1)
                self.assertIsInstance(start_errors[0], TimeoutError)
                self.assertFalse(starter_called.is_set())
            finally:
                allow_start_to_continue.set()
                allow_close.set()

    def test_close_failure_leaves_sticky_gate_against_replacement(self):
        manager = ACPSessionManager(agent_type="coco")
        session = MagicMock()
        session.session_id = "close-failed"
        session.message_count = 1
        session.to_snapshot.return_value = {"session_id": session.session_id}
        session.close.side_effect = RuntimeError("transport still alive")
        key = manager._session_key("chat-close-failed")
        manager._sessions[key] = session

        with self.assertRaisesRegex(RuntimeError, "transport still alive"):
            manager.retire_session(
                "chat-close-failed",
                expected_session=session,
                timeout=1,
            )

        with (
            patch.object(manager, "_start_session_inner") as start_inner,
            self.assertRaisesRegex(RuntimeError, "close failed"),
        ):
            manager.start_session(
                "chat-close-failed",
                cwd="/tmp",
                startup_timeout=1,
            )
        start_inner.assert_not_called()

    def test_production_sync_close_timeout_leaves_sticky_replacement_gate(self):
        manager = ACPSessionManager(agent_type="codex")
        session = SyncACPSession.__new__(SyncACPSession)
        session.session_id = "production-close-timeout"
        session.message_count = 1
        session._agent_type = "codex"
        session._force_dead = False
        session._active_future = None
        session._stop_watchdog = lambda: None
        session.to_snapshot = MagicMock(
            return_value={"session_id": session.session_id}
        )
        backend = SimpleNamespace(close=lambda: asyncio.sleep(0))
        session._acp_session = backend

        class _Loop:
            def stop(self):
                return None

            def call_soon_threadsafe(self, _callback):
                return None

            def close(self):
                return None

        loop = _Loop()
        session._loop = loop
        session._loop_thread = None
        session._drain_loop_before_close = lambda: None
        key = manager._session_key("chat-production-close-timeout")
        manager._sessions[key] = session

        close_future: concurrent.futures.Future[None] = (
            concurrent.futures.Future()
        )
        close_future.set_exception(TimeoutError("transport close timed out"))

        def submit(coro, _loop):
            coro.close()
            return close_future

        with patch(
            "src.acp.sync_adapter.asyncio.run_coroutine_threadsafe",
            side_effect=submit,
        ):
            with self.assertRaisesRegex(
                TimeoutError,
                "transport close timed out",
            ):
                manager.retire_session(
                    "chat-production-close-timeout",
                    expected_session=session,
                    timeout=1,
                )

        self.assertTrue(session._force_dead)
        self.assertIs(session._acp_session, backend)
        self.assertIs(session._loop, loop)
        with (
            patch.object(manager, "_start_session_inner") as start_inner,
            self.assertRaisesRegex(RuntimeError, "close failed"),
        ):
            manager.start_session(
                "chat-production-close-timeout",
                cwd="/tmp",
                startup_timeout=1,
            )
        start_inner.assert_not_called()

    def test_normal_end_session_also_gates_replacement_until_close(self):
        manager = ACPSessionManager(agent_type="coco")
        close_started = threading.Event()
        allow_close = threading.Event()

        class BlockingCloseSession:
            session_id = "normal-close"
            message_count = 1

            def to_snapshot(self):
                return {"session_id": self.session_id}

            def close(self):
                close_started.set()
                allow_close.wait(timeout=2)

        session = BlockingCloseSession()
        key = manager._session_key("chat-normal-close")
        manager._sessions[key] = session
        try:
            manager.end_session("chat-normal-close")
            self.assertTrue(close_started.wait(timeout=0.2))
            with (
                patch.object(manager, "_start_session_inner") as start_inner,
                self.assertRaisesRegex(TimeoutError, "closing"),
            ):
                manager.start_session(
                    "chat-normal-close",
                    cwd="/tmp",
                    startup_timeout=0.05,
                )
            start_inner.assert_not_called()
        finally:
            allow_close.set()

    def test_end_session_snapshot_failure_still_detaches_and_closes(self):
        manager = ACPSessionManager(agent_type="coco")
        session = MagicMock()
        session.session_id = "snapshot-failed-on-end"
        session.message_count = 1
        session.to_snapshot.side_effect = ValueError("broken snapshot")
        key = manager._session_key("chat-end-snapshot-failure")
        manager._sessions[key] = session

        snapshot = manager.end_session("chat-end-snapshot-failure")

        self.assertIsNone(snapshot)
        self.assertNotIn(key, manager._sessions)
        session.close.assert_called_once_with()

    def test_cleanup_all_waits_for_session_close_before_returning(self):
        manager = ACPSessionManager(agent_type="coco")
        close_started = threading.Event()
        allow_close = threading.Event()
        completed = threading.Event()
        errors: list[BaseException] = []

        class BlockingCloseSession:
            session_id = "cleanup-blocking"
            message_count = 1

            def to_snapshot(self):
                return {"session_id": self.session_id}

            def close(self):
                close_started.set()
                allow_close.wait(timeout=2)

        key = manager._session_key("chat-cleanup-drain")
        manager._sessions[key] = BlockingCloseSession()

        def cleanup() -> None:
            try:
                manager.cleanup_all(timeout=1)
            except BaseException as exc:  # pragma: no cover - assertion reports it
                errors.append(exc)
            finally:
                completed.set()

        worker = threading.Thread(target=cleanup)
        worker.start()
        try:
            self.assertTrue(close_started.wait(timeout=0.2))
            self.assertFalse(completed.wait(timeout=0.05))
            allow_close.set()
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
        finally:
            allow_close.set()

    def test_cleanup_all_does_not_silence_close_failure(self):
        manager = ACPSessionManager(agent_type="coco")
        session = MagicMock()
        session.session_id = "cleanup-close-failed"
        session.message_count = 1
        session.to_snapshot.return_value = {"session_id": session.session_id}
        session.close.side_effect = RuntimeError("cleanup transport alive")
        key = manager._session_key("chat-cleanup-failed")
        manager._sessions[key] = session

        with self.assertRaisesRegex(BaseExceptionGroup, "ACP session cleanup failed"):
            manager.cleanup_all(timeout=1)

    def test_rebind_waits_for_destination_close_before_moving_source(self):
        manager = ACPSessionManager(agent_type="coco")
        close_started = threading.Event()
        allow_close = threading.Event()
        observed: list[bool] = []
        errors: list[BaseException] = []

        source = MagicMock()
        source.session_id = "source"
        source.message_count = 1

        class BlockingDestination:
            session_id = "destination"
            message_count = 1

            def to_snapshot(self):
                return {"session_id": self.session_id}

            def close(self):
                close_started.set()
                allow_close.wait(timeout=2)

        old_key = manager._session_key("chat-rebind-fenced", "project")
        new_key = manager._session_key(
            "chat-rebind-fenced",
            "project",
            thread_id="thread",
        )
        manager._sessions[old_key] = source
        manager._sessions[new_key] = BlockingDestination()

        def rebind() -> None:
            try:
                observed.append(
                    manager.rebind_thread(
                        "chat-rebind-fenced",
                        "project",
                        "thread",
                        timeout=1,
                    )
                )
            except BaseException as exc:  # pragma: no cover - assertion reports it
                errors.append(exc)

        worker = threading.Thread(target=rebind)
        worker.start()
        try:
            self.assertTrue(close_started.wait(timeout=0.2))
            self.assertTrue(worker.is_alive())
            self.assertIs(manager._sessions[old_key], source)
            self.assertNotIn(new_key, manager._sessions)
            allow_close.set()
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(observed, [True])
            self.assertNotIn(old_key, manager._sessions)
            self.assertIs(manager._sessions[new_key], source)
        finally:
            allow_close.set()

    def test_atomic_replace_preserves_concurrent_healthy_session(self):
        manager = ACPSessionManager(agent_type="coco")
        timed_out = MagicMock()
        timed_out.session_id = "old"
        healthy = MagicMock()
        healthy.session_id = "healthy"
        healthy._force_dead = False
        key = manager._session_key("chat-atomic-replace")
        manager._sessions[key] = healthy

        replace = getattr(manager, "replace_session", None)
        self.assertTrue(callable(replace), "atomic session replacement is missing")
        with patch.object(manager, "_start_session_inner") as start_inner:
            observed = replace(
                "chat-atomic-replace",
                cwd="/tmp",
                expected_session=timed_out,
                startup_timeout=1,
            )

        self.assertIs(observed.session, healthy)
        self.assertFalse(observed.created)
        healthy.close.assert_not_called()
        timed_out.close.assert_not_called()
        start_inner.assert_not_called()

    def test_atomic_replace_rejects_absent_expected_session_without_starting(self):
        manager = ACPSessionManager(agent_type="coco")
        timed_out = MagicMock()
        timed_out.session_id = "old"
        key = manager._session_key("chat-atomic-replace-absent")

        with patch.object(manager, "_start_session_inner") as start_inner:
            observed = manager.replace_session(
                "chat-atomic-replace-absent",
                cwd="/tmp",
                expected_session=timed_out,
                startup_timeout=1,
            )

        self.assertIsNone(observed.session)
        self.assertFalse(observed.created)
        self.assertNotIn(key, manager._sessions)
        timed_out.close.assert_not_called()
        start_inner.assert_not_called()

    def test_atomic_replace_preserves_concurrent_force_dead_session(self):
        manager = ACPSessionManager(agent_type="coco")
        timed_out = MagicMock()
        timed_out.session_id = "old"
        concurrent = MagicMock()
        concurrent.session_id = "concurrent-dead"
        concurrent._force_dead = True
        key = manager._session_key("chat-atomic-replace-concurrent-dead")
        manager._sessions[key] = concurrent

        with patch.object(manager, "_start_session_inner") as start_inner:
            observed = manager.replace_session(
                "chat-atomic-replace-concurrent-dead",
                cwd="/tmp",
                expected_session=timed_out,
                startup_timeout=1,
            )

        self.assertIs(observed.session, concurrent)
        self.assertFalse(observed.created)
        self.assertIs(manager._sessions[key], concurrent)
        timed_out.close.assert_not_called()
        concurrent.close.assert_not_called()
        start_inner.assert_not_called()

    def test_atomic_replace_closes_old_session_before_starting_new(self):
        manager = ACPSessionManager(agent_type="coco")
        events: list[str] = []
        timed_out = MagicMock()
        timed_out.session_id = "old"
        timed_out.message_count = 1
        timed_out.to_snapshot.return_value = {"session_id": "old"}
        timed_out.close.side_effect = lambda: events.append("close")
        replacement = MagicMock()
        key = manager._session_key("chat-atomic-replace-old")
        manager._sessions[key] = timed_out

        def start_inner(*_args, **_kwargs):
            events.append("start")
            manager._sessions[key] = replacement
            return replacement

        with patch.object(manager, "_start_session_inner", side_effect=start_inner):
            observed = manager.replace_session(
                "chat-atomic-replace-old",
                cwd="/tmp",
                expected_session=timed_out,
                startup_timeout=1,
            )

        self.assertIs(observed.session, replacement)
        self.assertTrue(observed.created)
        self.assertEqual(events, ["close", "start"])

    def test_get_session_evicts_recent_force_dead_session(self):
        manager = ACPSessionManager(agent_type="coco")
        session = MagicMock()
        session._force_dead = True
        session.last_active = time.time()
        session.session_id = "dead-session"
        session.message_count = 1
        session.to_snapshot.return_value = {"session_id": session.session_id}
        key = manager._session_key("chat-force-dead")
        manager._sessions[key] = session

        self.assertIsNone(manager.get_session("chat-force-dead"))
        self.assertNotIn(key, manager._sessions)

    def test_ensure_session_never_reuses_force_dead_session(self):
        manager = ACPSessionManager(agent_type="coco")
        dead = MagicMock()
        dead._force_dead = True
        dead.last_active = time.time()
        dead.session_id = "dead-session"
        dead.message_count = 1
        dead.to_snapshot.return_value = {"session_id": dead.session_id}
        replacement = MagicMock()
        key = manager._session_key("chat-force-dead-ensure")
        manager._sessions[key] = dead

        with patch.object(
            manager,
            "start_session",
            return_value=replacement,
        ) as start:
            observed = manager.ensure_session(
                "chat-force-dead-ensure",
                cwd="/tmp",
            )

        self.assertIs(observed, replacement)
        dead.close.assert_called_once_with()
        start.assert_called_once()

    def test_get_session_idle_expiry_returns_concurrent_replacement(self):
        manager = ACPSessionManager(agent_type="coco", session_timeout=50)
        last_active_read = threading.Event()
        allow_last_active = threading.Event()

        class ExpiredSession:
            _force_dead = False
            session_id = "expired"
            message_count = 1

            @property
            def last_active(self):
                last_active_read.set()
                allow_last_active.wait(timeout=2)
                return 0

        healthy = MagicMock()
        healthy._force_dead = False
        healthy.last_active = 100
        healthy.session_id = "healthy"
        healthy.message_count = 1
        key = manager._session_key("chat-idle-race")
        manager._sessions[key] = ExpiredSession()
        observed = []

        with patch("src.acp.manager.time.time", return_value=100):
            worker = threading.Thread(
                target=lambda: observed.append(
                    manager.get_session("chat-idle-race")
                )
            )
            worker.start()
            self.assertTrue(last_active_read.wait(timeout=1))
            with manager._acquire_lock():
                manager._sessions[key] = healthy
            allow_last_active.set()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertIs(manager._sessions[key], healthy)
        self.assertEqual(observed, [healthy])

    def test_get_session_dead_healthcheck_returns_concurrent_replacement(self):
        manager = ACPSessionManager(agent_type="coco", session_timeout=1000)
        healthcheck_started = threading.Event()
        allow_healthcheck = threading.Event()

        class DeadSession:
            _force_dead = False
            last_active = 0
            session_id = "dead"
            message_count = 1

            def is_server_running(self):
                healthcheck_started.set()
                allow_healthcheck.wait(timeout=2)
                return False

        healthy = MagicMock()
        healthy._force_dead = False
        healthy.last_active = 100
        healthy.session_id = "healthy"
        healthy.message_count = 1
        key = manager._session_key("chat-health-race")
        manager._sessions[key] = DeadSession()
        observed = []

        with patch("src.acp.manager.time.time", return_value=100):
            worker = threading.Thread(
                target=lambda: observed.append(
                    manager.get_session("chat-health-race")
                )
            )
            worker.start()
            self.assertTrue(healthcheck_started.wait(timeout=1))
            with manager._acquire_lock():
                manager._sessions[key] = healthy
            allow_healthcheck.set()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertIs(manager._sessions[key], healthy)
        self.assertEqual(observed, [healthy])

    def test_ensure_session_restarts_on_agent_mismatch(self):
        manager = ACPSessionManager(agent_type="coco")

        # Mock session 1
        mock_session_1 = MagicMock()
        mock_session_1._agent_type = "coco"
        mock_session_1._agent_args = ["acp", "serve"]
        mock_session_1.last_active = 1000
        mock_session_1.is_server_running.return_value = True

        # Mock session 2
        mock_session_2 = MagicMock()
        mock_session_2._agent_type = "claude"
        mock_session_2.session_id = "new_sid"

        with patch.object(manager, "start_session") as mock_start:
            # First, manually add session 1
            key = manager._session_key("chat1")
            manager._sessions[key] = mock_session_1

            # Now ensure_session with agent_type_override="claude"
            # It should detect mismatch and call start_session
            mock_start.return_value = mock_session_2

            with patch("time.time", return_value=1005):
                session = manager.ensure_session("chat1", agent_type_override="claude")

            self.assertEqual(session, mock_session_2)
            mock_start.assert_called()
            # Verify that the old session was cleaned up (not in _sessions if mock_start is real, but here we mock it)
            # In our implementation, start_session will overwrite it.

    def test_ensure_session_restarts_on_model_mismatch(self):
        manager = ACPSessionManager(agent_type="coco")

        # Mock session with model A
        mock_session = MagicMock()
        mock_session._agent_type = "coco"
        mock_session._agent_args = ["acp", "serve", "-c", "model.name=gpt-4"]
        mock_session.last_active = 1000
        mock_session.is_server_running.return_value = True

        with patch.object(manager, "start_session") as mock_start:
            key = manager._session_key("chat1")
            manager._sessions[key] = mock_session

            # ensure_session with model_name="claude-3"
            with patch("time.time", return_value=1005):
                manager.ensure_session("chat1", model_name="claude-3")

            # Should restart because "claude-3" is not in args
            mock_start.assert_called()

    def test_ensure_session_no_warm_restart_on_server_death(self):
        manager = ACPSessionManager(agent_type="coco")

        # Mock dead session
        mock_session_1 = MagicMock()
        mock_session_1._agent_type = "coco"
        mock_session_1.last_active = 1000
        mock_session_1.is_server_running.return_value = False # Dead

        # Mock new session
        mock_session_2 = MagicMock()
        mock_session_2.warm_restart_msg = ""

        with patch.object(manager, "start_session") as mock_start:
            key = manager._session_key("chat1")
            manager._sessions[key] = mock_session_1
            mock_start.return_value = mock_session_2

            with patch("time.time", return_value=1005):
                session = manager.ensure_session("chat1")

            self.assertEqual(session, mock_session_2)
            self.assertEqual(session.warm_restart_msg, "")

    def test_start_session_inner_delegates_backend_start_to_coordinator(self):
        from src.acp import startup_utils

        manager = ACPSessionManager(agent_type="coco")
        fake_session = MagicMock()
        fake_session.session_id = "sid"
        fake_session.load_local_history.return_value = []

        fake_coordinator = MagicMock()
        fake_coordinator.start.return_value = startup_utils.SessionStartupResult(
            session=fake_session,
            actual_id="sid",
            effective_agent_type="coco",
            model_name="gpt-test",
        )

        with patch.object(startup_utils, "SessionStartupCoordinator", return_value=fake_coordinator):
            session = manager._start_session_inner(
                key=manager._session_key("chat1", "project1"),
                chat_id="chat1",
                cwd="/tmp",
                session_id=None,
                startup_timeout=0.1,
                project_id="project1",
                agent_type_override=None,
                model_name="gpt-test",
                thread_id=None,
            )

        self.assertIs(session, fake_session)
        fake_coordinator.start.assert_called_once()
        start_request = fake_coordinator.start.call_args.args[0]
        self.assertEqual(start_request.key, manager._session_key("chat1", "project1"))
        self.assertEqual(start_request.effective_agent_type, "coco")
        self.assertEqual(start_request.cwd, "/tmp")
        self.assertEqual(start_request.model_name, "gpt-test")
        self.assertIs(manager._sessions[manager._session_key("chat1", "project1")], fake_session)

    def test_start_session_inner_fatal_startup_has_no_success_side_effects(self):
        from src.acp import startup_utils

        telemetry = MagicMock()
        manager = ACPSessionManager(agent_type="coco", session_telemetry=telemetry)
        fake_coordinator = MagicMock()
        fake_coordinator.start.side_effect = RuntimeError("fatal startup")
        key = manager._session_key("chat1", "project1")

        with patch.object(startup_utils, "SessionStartupCoordinator", return_value=fake_coordinator):
            with self.assertRaises(RuntimeError):
                manager._start_session_inner(
                    key=key,
                    chat_id="chat1",
                    cwd="/tmp",
                    session_id=None,
                    startup_timeout=0.1,
                    project_id="project1",
                    agent_type_override=None,
                    model_name="gpt-test",
                    thread_id=None,
                )

        self.assertNotIn(key, manager._sessions)
        telemetry.on_session_start.assert_not_called()
        telemetry.on_session_start_failed.assert_not_called()

    def test_start_session_releases_key_lock_after_success(self):
        manager = ACPSessionManager(agent_type="coco")
        fake_session = MagicMock()

        with patch.object(manager, "_start_session_inner", return_value=fake_session):
            assert manager.start_session("chat-lock", startup_timeout=0.1) is fake_session

        self.assertEqual(manager._key_locks, {})

    def test_start_session_releases_key_lock_after_failure(self):
        manager = ACPSessionManager(agent_type="coco")

        with patch.object(manager, "_start_session_inner", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                manager.start_session("chat-lock", startup_timeout=0.1)

        self.assertEqual(manager._key_locks, {})

    def test_start_session_releases_key_lock_ref_after_acquire_timeout(self):
        manager = ACPSessionManager(agent_type="coco")
        key = manager._session_key("chat-timeout")
        held_lock = manager._get_key_lock(key)
        held_lock.acquire()

        try:
            with self.assertRaises(TimeoutError):
                manager.start_session("chat-timeout", startup_timeout=0.01)

            self.assertIn(key, manager._key_locks)
            self.assertEqual(manager._key_locks[key][1], 1)
        finally:
            held_lock.release()
            manager._release_key_lock(key)

        self.assertEqual(manager._key_locks, {})

    def test_start_session_releases_key_lock_after_concurrent_starts(self):
        manager = ACPSessionManager(agent_type="coco")
        entered = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        def fake_start(*args, **kwargs):
            calls.append(args[1])
            entered.set()
            release.wait(timeout=2)
            return MagicMock()

        with patch.object(manager, "_start_session_inner", side_effect=fake_start):
            first = threading.Thread(target=lambda: manager.start_session("chat-lock", startup_timeout=1), daemon=True)
            first.start()
            self.assertTrue(entered.wait(timeout=1))
            second_result: list[object] = []
            second = threading.Thread(
                target=lambda: second_result.append(manager.start_session("chat-lock", startup_timeout=1)),
                daemon=True,
            )
            second.start()
            time.sleep(0.05)
            release.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(second_result), 1)
        self.assertEqual(manager._key_locks, {})

    def test_manager_uses_public_idle_health_facade_only(self):
        root = Path(__file__).resolve().parents[1]
        manager_source = (root / "src" / "acp" / "manager.py").read_text(encoding="utf-8")

        assert "_classify_idle_health_for_manager" not in manager_source
        assert "classify_manager_idle_health" in manager_source
        assert "IdleHealthConfig._resolve_for_manager" not in manager_source


    def test_public_idle_health_facade_matches_legacy_classification(self):
        from src.acp import telemetry
        from src.utils.time_ago import TimeAgoBucket

        bucket = TimeAgoBucket(label="刚刚", seconds=1.0, level="fresh")

        assert telemetry.classify_manager_idle_health(bucket) == telemetry._classify_idle_health_for_manager(bucket)

if __name__ == "__main__":
    unittest.main()
