"""Tests for RepoLockHeartbeat."""

from __future__ import annotations

import threading
import time

from src.utils.heartbeat import RepoLockHeartbeat


class TestRepoLockHeartbeatBasic:
    """Core functionality."""

    def test_touch_fn_called_multiple_times(self):
        counter = {"n": 0}

        def touch():
            counter["n"] += 1

        stop = threading.Event()
        hb = RepoLockHeartbeat(stop, touch, interval=0.02, max_beats=None)
        hb.start()
        deadline = time.monotonic() + 1.5
        while counter["n"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        hb.join(timeout=1)
        assert counter["n"] >= 2

    def test_stop_event_terminates(self):
        counter = {"n": 0}

        def touch():
            counter["n"] += 1

        stop = threading.Event()
        hb = RepoLockHeartbeat(stop, touch, interval=0.02)
        hb.start()
        deadline = time.monotonic() + 1.5
        while counter["n"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        hb.join(timeout=1)
        after_stop = counter["n"]
        assert after_stop >= 2
        time.sleep(0.06)  # verify no more beats
        assert counter["n"] == after_stop

    def test_max_beats_stops_loop(self):
        counter = {"n": 0}

        def touch():
            counter["n"] += 1

        stop = threading.Event()
        hb = RepoLockHeartbeat(stop, touch, interval=0.01, max_beats=3)
        hb.start()
        hb.join(timeout=2)
        assert counter["n"] == 3
        assert not hb.is_alive

    def test_daemon_thread(self):
        stop = threading.Event()
        hb = RepoLockHeartbeat(stop, lambda: None, interval=100)
        hb.start()
        assert hb._thread.daemon is True
        stop.set()
        hb.join()


class TestRepoLockHeartbeatExceptionSafety:
    """touch_fn exceptions must not kill the heartbeat thread."""

    def test_exception_does_not_stop_heartbeat(self):
        counter = {"n": 0}

        def touch():
            counter["n"] += 1
            if counter["n"] == 1:
                raise RuntimeError("boom")

        stop = threading.Event()
        hb = RepoLockHeartbeat(stop, touch, interval=0.02, max_beats=5)
        hb.start()
        hb.join(timeout=2)
        # beat 1 raises but loop continues; beats 2-5 succeed
        assert counter["n"] == 5


class TestRepoLockHeartbeatMaxBeatsNone:
    """max_beats=None means unlimited (until stop_event)."""

    def test_unlimited_beats(self):
        counter = {"n": 0}

        def touch():
            counter["n"] += 1

        stop = threading.Event()
        hb = RepoLockHeartbeat(stop, touch, interval=0.01, max_beats=None)
        hb.start()
        deadline = time.monotonic() + 1.5
        while counter["n"] < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        hb.join(timeout=1)
        assert counter["n"] >= 3
