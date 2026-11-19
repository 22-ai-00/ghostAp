"""Tests for card hook executor backpressure, isolation, lazy startup, and telemetry."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock, patch

from src.card.events import CardEvent, CardEventType
from src.card.hooks import (
    _MAX_CONSECUTIVE_TIMEOUTS,
    SLOW_HOOK_THRESHOLD_MS,
    ContextPersistenceHook,
    EmojiHook,
    HookFirer,
    _HookExecutorManager,
    _log_hook_timing,
)
from src.card.state.models import CardMetadata, CardState


def _make_state() -> CardState:
    return CardState(
        metadata=CardMetadata(mode_name="Test", tool_name="coco", model_name="m1"),
    )


class TestNoopDispatchedHooksSkipped:
    """Built-in no-op on_dispatched hooks bypass executor submission completely."""

    def test_2048_event_replay_skips_default_hooks_with_zero_submissions(self, caplog):
        """2048-event replay with EmojiHook + ContextPersistenceHook submits 0 tasks."""
        mgr = _HookExecutorManager()
        submit_spy = MagicMock(wraps=mgr.submit)
        mgr.submit = submit_spy  # type: ignore[method-assign]

        emoji = EmojiHook(add_reaction=MagicMock(), message_id="om_123", chat_id="oc_secret")
        ctx_hook = ContextPersistenceHook(update_fn=MagicMock(), chat_id="oc_secret")
        firer = HookFirer(hooks=(emoji, ctx_hook), session_id="replay-2048", executor=mgr)
        state = _make_state()

        events = [
            CardEvent.text_delta("blk_1", f"secret message chunk {i}")
            for i in range(2048)
        ]

        try:
            with caplog.at_level(logging.WARNING, logger="src.card.hooks"):
                for ev in events:
                    firer.fire_dispatched(ev, state)

            assert submit_spy.call_count == 0
            assert mgr._executor is None
            assert mgr._terminal_executor is None
            assert firer.drain_dispatched(timeout=0.05) is True
            assert "backpressure" not in caplog.text.lower()
        finally:
            mgr.shutdown()

    def test_unoverridden_subclasses_also_skipped(self):
        """Subclasses of EmojiHook/ContextPersistenceHook without overrides still skip."""

        class SubEmojiHook(EmojiHook):
            pass

        class SubContextHook(ContextPersistenceHook):
            pass

        mgr = _HookExecutorManager()
        submit_spy = MagicMock(wraps=mgr.submit)
        mgr.submit = submit_spy  # type: ignore[method-assign]

        hooks = (
            SubEmojiHook(add_reaction=MagicMock(), message_id="om_1"),
            SubContextHook(update_fn=MagicMock()),
        )
        firer = HookFirer(hooks=hooks, session_id="sub-noop", executor=mgr)
        try:
            firer.fire_dispatched(CardEvent.started(), _make_state())
            assert submit_spy.call_count == 0
            assert mgr._executor is None
        finally:
            mgr.shutdown()

    def test_subclasses_overriding_on_dispatched_still_execute(self):
        """Subclasses overriding on_dispatched execute normally in the executor."""
        called = []
        done = threading.Event()

        class CustomEmojiHook(EmojiHook):
            def on_dispatched(self, event: CardEvent, state: CardState) -> None:
                called.append(("emoji", event.type))
                if len(called) == 2:
                    done.set()

        class CustomContextHook(ContextPersistenceHook):
            def on_dispatched(self, event: CardEvent, state: CardState) -> None:
                called.append(("ctx", event.type))
                if len(called) == 2:
                    done.set()

        mgr = _HookExecutorManager()
        add_reaction = MagicMock()
        update_fn = MagicMock()
        hooks = (
            CustomEmojiHook(add_reaction=add_reaction, message_id="om_sub"),
            CustomContextHook(update_fn=update_fn),
        )
        firer = HookFirer(hooks=hooks, session_id="sub-override", executor=mgr)
        state = _make_state()
        try:
            firer.fire_dispatched(CardEvent.started(), state)
            assert done.wait(timeout=2.0)
            assert firer.drain_dispatched(timeout=2.0) is True
            assert ("emoji", CardEventType.STARTED) in called
            assert ("ctx", CardEventType.STARTED) in called

            firer.fire_terminal(state, "completed")
            add_reaction.assert_called_once_with("om_sub", EmojiHook.SUCCESS_EMOJI_DEFAULT)
            update_fn.assert_called_once_with(state)
        finally:
            mgr.shutdown()

    def test_instance_and_class_patch_overrides_still_execute(self):
        """Instance-level or class-patched on_dispatched overrides are not skipped."""
        mgr = _HookExecutorManager()
        emoji = EmojiHook(add_reaction=MagicMock(), message_id="om_inst")
        inst_done = threading.Event()
        emoji.on_dispatched = lambda _ev, _st: inst_done.set()  # type: ignore[method-assign]

        firer = HookFirer(hooks=(emoji,), session_id="inst-override", executor=mgr)
        try:
            firer.fire_dispatched(CardEvent.started(), _make_state())
            assert inst_done.wait(timeout=2.0)
        finally:
            mgr.shutdown()

        mgr2 = _HookExecutorManager()
        ctx_hook = ContextPersistenceHook(update_fn=MagicMock())
        cls_done = threading.Event()
        firer2 = HookFirer(hooks=(ctx_hook,), session_id="cls-patch", executor=mgr2)
        try:
            with patch.object(
                ContextPersistenceHook,
                "on_dispatched",
                side_effect=lambda _ev, _st: cls_done.set(),
            ):
                firer2.fire_dispatched(CardEvent.started(), _make_state())
                assert cls_done.wait(timeout=2.0)
        finally:
            mgr2.shutdown()

    def test_builtin_on_dispatched_remains_callable(self):
        """Direct calls to built-in on_dispatched remain valid callable no-ops."""
        emoji = EmojiHook(add_reaction=MagicMock(), message_id="om_direct")
        ctx_hook = ContextPersistenceHook(update_fn=MagicMock())
        ev = CardEvent.started()
        st = _make_state()
        assert emoji.on_dispatched(ev, st) is None
        assert ctx_hook.on_dispatched(ev, st) is None


class TestTerminalIsolationAndAdmission:
    """on_terminal is isolated from on_dispatched saturation and logs failures clearly."""

    def test_on_terminal_not_starved_by_dispatched_backpressure(self, caplog):
        """Saturating all 24 dispatched slots + workers never blocks EmojiHook/ContextPersistenceHook on_terminal."""
        mgr = _HookExecutorManager()
        release_dispatched = threading.Event()
        workers_running = threading.Barrier(7, timeout=5.0)  # 6 worker threads + test thread

        class BlockingDispatchedHook:
            def on_dispatched(self, event: CardEvent, state: CardState) -> None:
                # First 6 calls synchronize on barrier so all 6 dispatched workers are actively pinned
                try:
                    workers_running.wait(timeout=0.05)
                except threading.BrokenBarrierError:
                    pass
                release_dispatched.wait(timeout=5.0)

        dispatched_firer = HookFirer(
            hooks=(BlockingDispatchedHook(),),
            session_id="burst-dispatched",
            executor=mgr,
        )
        add_reaction = MagicMock()
        update_fn = MagicMock()
        terminal_firer = HookFirer(
            hooks=(
                EmojiHook(add_reaction=add_reaction, message_id="om_term", chat_id="oc_hidden"),
                ContextPersistenceHook(update_fn=update_fn, chat_id="oc_hidden"),
            ),
            session_id="terminal-session",
            executor=mgr,
        )
        state = _make_state()
        ev = CardEvent.text_delta("b1", "user secret payload")

        try:
            with caplog.at_level(logging.WARNING, logger="src.card.hooks"):
                # Fill 24 dispatched slots + trigger 6 backpressure rejections
                for _ in range(30):
                    dispatched_firer.fire_dispatched(ev, state)

                dispatched_gen = mgr._generation
                assert dispatched_gen is not None
                assert dispatched_gen.in_flight == 24
                assert "backpressure triggered" in caplog.text
                assert "hook_name=BlockingDispatchedHook" in caplog.text
                assert "phase=on_dispatched" in caplog.text
                assert "event_type=text_delta" in caplog.text
                assert "in_flight=24/24" in caplog.text
                assert "capacity=24" in caplog.text

                # While all 24 dispatched slots and 6 dispatched workers remain blocked,
                # terminal hooks must execute immediately without waiting on release_dispatched.
                t0 = time.monotonic()
                terminal_firer.fire_terminal(state, "completed")
                elapsed = time.monotonic() - t0

                assert elapsed < 1.5
                add_reaction.assert_called_once_with("om_term", EmojiHook.SUCCESS_EMOJI_DEFAULT)
                update_fn.assert_called_once_with(state)
        finally:
            release_dispatched.set()
            mgr.shutdown()

    def test_terminal_saturation_logs_backpressure_and_rejection_with_hook_name(self, caplog):
        """If the 24-slot terminal lane itself saturates, a clear WARNING with hook_name is logged."""
        mgr = _HookExecutorManager()
        blocker = threading.Event()
        try:
            for _ in range(24):
                fut = mgr.submit(lambda: blocker.wait(timeout=5.0), phase="on_terminal")
                assert fut is not None

            emoji = EmojiHook(add_reaction=MagicMock(), message_id="om_sat", chat_id="oc_private")
            firer = HookFirer(hooks=(emoji,), session_id="sat-term", executor=mgr)
            with caplog.at_level(logging.WARNING, logger="src.card.hooks"):
                firer.fire_terminal(_make_state(), "completed")

            assert "backpressure triggered" in caplog.text
            assert "on_terminal submission rejected" in caplog.text
            assert "hook_name=EmojiHook" in caplog.text
            assert "phase=on_terminal" in caplog.text
            assert "in_flight=24/24" in caplog.text
            assert "capacity=24" in caplog.text
            assert "oc_private" not in caplog.text
        finally:
            blocker.set()
            mgr.shutdown()

    def test_terminal_rejection_logs_warning_with_sanitized_hook_name(self, caplog):
        """When terminal submission is rejected or fails, a WARNING with hook_name is logged."""
        mgr = _HookExecutorManager()
        mgr.shutdown()

        emoji = EmojiHook(add_reaction=MagicMock(), message_id="om_rej", chat_id="oc_private")
        ctx_hook = ContextPersistenceHook(update_fn=MagicMock(), chat_id="oc_private")
        firer = HookFirer(hooks=(emoji, ctx_hook), session_id="rej-term", executor=mgr)

        with caplog.at_level(logging.WARNING, logger="src.card.hooks"):
            firer.fire_terminal(_make_state(), "completed")

        assert "hook_name=EmojiHook" in caplog.text
        assert "hook_name=ContextPersistenceHook" in caplog.text
        assert "phase=on_terminal" in caplog.text
        assert "oc_private" not in caplog.text

    def test_terminal_submit_exception_logs_warning_with_hook_name(self, caplog):
        """Unexpected exception from executor.submit during fire_terminal logs hook_name WARNING."""
        mock_exec = MagicMock()
        mock_exec.submit.side_effect = ValueError("submit failed")
        emoji = EmojiHook(add_reaction=MagicMock(), message_id="om_exc", chat_id="oc_private")
        firer = HookFirer(hooks=(emoji,), session_id="exc-term", executor=mock_exec)

        with caplog.at_level(logging.WARNING, logger="src.card.hooks"):
            firer.fire_terminal(_make_state(), "failed")

        assert "hook_name=EmojiHook" in caplog.text
        assert "phase=on_terminal" in caplog.text
        assert "event_type=failed" in caplog.text
        assert "oc_private" not in caplog.text


class TestGenerationalRebuildAndShutdownSafety:
    """Generational semaphore isolation on rebuild, cancellation, and shutdown RuntimeError."""

    def test_timeout_rebuild_replenishes_full_24_slot_capacity(self):
        """Rebuilding on consecutive timeouts gives a fresh 24-slot semaphore isolated from stuck old tasks."""
        mgr = _HookExecutorManager()
        old_gate = threading.Event()
        new_gate = threading.Event()
        old_finished = threading.Event()
        old_done_count = 0
        count_lock = threading.Lock()

        def stuck_task() -> None:
            nonlocal old_done_count
            old_gate.wait(timeout=5.0)
            with count_lock:
                old_done_count += 1
                if old_done_count == 24:
                    old_finished.set()

        try:
            # Fill all 24 slots in generation 1
            old_futures = [mgr.submit(stuck_task, phase="on_dispatched") for _ in range(24)]
            assert all(f is not None for f in old_futures)
            old_gen = mgr._generation
            assert old_gen is not None
            assert old_gen.capacity == 24
            assert old_gen.in_flight == 24
            # 25th rejected due to full capacity
            assert mgr.submit(lambda: None, phase="on_dispatched") is None

            # Trigger rebuild via consecutive timeouts while old_gate is still blocked
            for _ in range(_MAX_CONSECUTIVE_TIMEOUTS):
                mgr.record_timeout(phase="on_dispatched")

            new_gen = mgr._generation
            assert new_gen is not None
            assert new_gen is not old_gen
            assert new_gen.semaphore is not old_gen.semaphore
            assert new_gen.capacity == 24
            assert new_gen.in_flight == 0

            # Full 24-slot capacity must be available in generation 2 immediately
            new_futures = [
                mgr.submit(lambda: new_gate.wait(timeout=5.0), phase="on_dispatched")
                for _ in range(24)
            ]
            assert all(f is not None for f in new_futures)
            assert new_gen.in_flight == 24
            assert mgr.submit(lambda: None, phase="on_dispatched") is None

            # Releasing old tasks only releases old_gen.semaphore, never new_gen.semaphore
            old_gate.set()
            assert old_finished.wait(timeout=5.0)
            for f in old_futures:
                assert f is not None
                f.result(timeout=5.0)
            assert old_gen.in_flight == 0
            assert new_gen.in_flight == 24

            new_gate.set()
            for f in new_futures:
                assert f is not None
                f.result(timeout=5.0)
            assert new_gen.in_flight == 0
        finally:
            old_gate.set()
            new_gate.set()
            mgr.shutdown()

    def test_runtime_error_after_executor_shutdown_releases_permit_and_returns_none(self, caplog):
        """RuntimeError on executor.submit releases generation semaphore permit and returns None."""
        mgr = _HookExecutorManager()
        try:
            mgr.submit(lambda: None, phase="on_dispatched").result(timeout=5.0)
            gen = mgr._generation
            assert gen is not None
            gen.executor.shutdown(wait=True)

            with caplog.at_level(logging.WARNING, logger="src.card.hooks"):
                fut = mgr.submit(
                    lambda: None,
                    phase="on_dispatched",
                    hook_name="LateHook",
                    event_type="tool_done",
                )

            assert fut is None
            assert gen.in_flight == 0
            acquired, in_flight, capacity = gen.try_acquire()
            assert acquired is True
            assert in_flight == 1
            assert capacity == 24
            gen.release()
            assert "cannot schedule" in caplog.text.lower() or "after shutdown" in caplog.text.lower()
            assert "hook_name=LateHook" in caplog.text
        finally:
            mgr.shutdown()

    def test_cancelled_queued_futures_release_semaphore_permits(self):
        """Cancelling queued futures releases their generation permits without leakage."""
        mgr = _HookExecutorManager()
        blocker = threading.Event()
        try:
            # 6 tasks occupy the 6 workers; tasks 7..24 remain queued
            running = [mgr.submit(lambda: blocker.wait(timeout=5.0), phase="on_terminal") for _ in range(6)]
            queued = [mgr.submit(lambda: None, phase="on_terminal") for _ in range(18)]
            assert all(f is not None for f in running + queued)

            gen = mgr._terminal_generation
            assert gen is not None
            assert gen.in_flight == 24

            cancelled_count = sum(1 for f in queued if f is not None and f.cancel())
            assert cancelled_count > 0
            assert gen.in_flight == 24 - cancelled_count

            blocker.set()
            for f in running + queued:
                if f is not None and not f.cancelled():
                    f.result(timeout=5.0)
            assert gen.in_flight == 0
        finally:
            blocker.set()
            mgr.shutdown()

    def test_concurrent_dispatched_terminal_and_timeout_rebuilds_safe(self):
        """Concurrent fire_dispatched, fire_terminal, and record_timeout never deadlock or error."""
        mgr = _HookExecutorManager()
        errors: list[Exception] = []

        class QuickHook:
            def on_dispatched(self, event: CardEvent, state: CardState) -> None:
                pass

            def on_terminal(self, state: CardState, reason: str) -> None:
                pass

        state = _make_state()
        ev = CardEvent.text_delta("b", "ok")
        barrier = threading.Barrier(6, timeout=5.0)

        def worker(idx: int) -> None:
            try:
                barrier.wait()
                for j in range(25):
                    firer = HookFirer(hooks=(QuickHook(),), session_id=f"conc-{idx}-{j}", executor=mgr)
                    firer.fire_dispatched(ev, state)
                    if j % 5 == 0:
                        firer.fire_terminal(state, "completed")
                    if j % 7 == 0:
                        mgr.record_timeout(phase="on_dispatched")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
            assert not errors
        finally:
            mgr.shutdown()


class TestSanitizedHookTelemetry:
    """Sanitized hook latency, slow-hook, and backpressure logging."""

    def test_slow_hook_warning_and_normal_debug_without_sensitive_content(self, caplog):
        """Logs include hook_name/phase/event_type/elapsed_ms and never leak chat_id/open_id/body."""
        secret_body = "SECRET_MESSAGE_BODY_XYZ"
        secret_chat = "oc_secret_chat_12345"
        secret_open = "ou_secret_open_67890"

        class CustomObservedHook:
            def on_dispatched(self, event: CardEvent, state: CardState) -> None:
                pass

            def on_terminal(self, state: CardState, reason: str) -> None:
                pass

        mgr = _HookExecutorManager()
        emoji = EmojiHook(add_reaction=MagicMock(), message_id="", chat_id=secret_chat)
        firer = HookFirer(
            hooks=(CustomObservedHook(), emoji),
            session_id="telemetry-sess",
            executor=mgr,
        )
        event = CardEvent.text_delta("b1", f"{secret_body} {secret_open}")
        state = _make_state()

        try:
            # 1. Normal fast completion -> DEBUG with elapsed_ms
            with caplog.at_level(logging.DEBUG, logger="src.card.hooks"):
                firer.fire_dispatched(event, state)
                assert firer.drain_dispatched(timeout=2.0) is True
                firer.fire_terminal(state, "completed")

            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG and "Hook completed:" in r.message]
            assert any(
                "hook_name=CustomObservedHook" in r.message
                and "phase=on_dispatched" in r.message
                and "event_type=text_delta" in r.message
                and "elapsed_ms=" in r.message
                for r in debug_records
            )
            assert any(
                "hook_name=EmojiHook" in r.message
                and "phase=on_terminal" in r.message
                and "event_type=completed" in r.message
                and "elapsed_ms=" in r.message
                for r in debug_records
            )
            assert secret_body not in caplog.text
            assert secret_chat not in caplog.text
            assert secret_open not in caplog.text

            caplog.clear()

            # 2. Slow hook >= 1000ms -> WARNING with elapsed_ms for both phases
            assert SLOW_HOOK_THRESHOLD_MS == 1000.0
            dispatched_done = threading.Event()

            class RealSlowHook:
                def on_dispatched(self, _ev: CardEvent, _st: CardState) -> None:
                    time.sleep(1.05)
                    dispatched_done.set()

                def on_terminal(self, _st: CardState, _rs: str) -> None:
                    time.sleep(1.05)

            with caplog.at_level(logging.WARNING, logger="src.card.hooks"):
                slow_firer = HookFirer(
                    hooks=(RealSlowHook(),),
                    session_id="slow-sess",
                    executor=mgr,
                )
                slow_firer.fire_dispatched(event, state)
                slow_firer.fire_terminal(state, "completed")
                assert dispatched_done.wait(timeout=3.0)
                assert slow_firer.drain_dispatched(timeout=3.0) is True

            slow_warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "Hook slow:" in r.message]
            dispatched_slow = [r for r in slow_warnings if "phase=on_dispatched" in r.message]
            terminal_slow = [r for r in slow_warnings if "phase=on_terminal" in r.message]
            assert len(dispatched_slow) == 1
            assert len(terminal_slow) == 1
            for rec in (dispatched_slow[0], terminal_slow[0]):
                msg = rec.message
                assert "hook_name=RealSlowHook" in msg
                assert "elapsed_ms=" in msg
                elapsed_val = float(msg.split("elapsed_ms=")[1].split()[0])
                assert elapsed_val >= 1000.0
            assert "event_type=text_delta" in dispatched_slow[0].message
            assert "event_type=completed" in terminal_slow[0].message
            assert secret_body not in caplog.text
            assert secret_chat not in caplog.text
            assert secret_open not in caplog.text

            # Threshold boundary check: 999.9ms -> DEBUG, 1000.0ms -> WARNING
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger="src.card.hooks"):
                _log_hook_timing(
                    hook_name="BoundaryHook",
                    phase="on_dispatched",
                    event_type="tool_done",
                    elapsed_ms=999.9,
                )
                _log_hook_timing(
                    hook_name="BoundaryHook",
                    phase="on_dispatched",
                    event_type="tool_done",
                    elapsed_ms=1000.0,
                )
            assert any(
                r.levelno == logging.DEBUG and "elapsed_ms=999.9" in r.message
                for r in caplog.records
            )
            assert any(
                r.levelno == logging.WARNING and "elapsed_ms=1000.0" in r.message
                for r in caplog.records
            )
        finally:
            mgr.shutdown()


class TestLazyExecutorStartup:
    """Importing src.card.hooks spawns zero worker threads until actual submission."""

    def test_import_spawns_zero_threads(self):
        """Fresh interpreter importing src.card.hooks has 1 thread and no hook pools."""
        code = (
            "import threading; "
            "import src.card.hooks as h; "
            "names = [t.name for t in threading.enumerate()]; "
            "assert len(names) == 1, f'unexpected threads: {names}'; "
            "assert h._hook_executor_manager._executor is None; "
            "assert h._hook_executor_manager._terminal_executor is None; "
            "print('OK')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert proc.stdout.strip() == "OK"
