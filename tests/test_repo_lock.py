"""Unit tests for RepoLockManager."""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.repo_lock import AcquireResult, LockConflictError, RepoLockInfo, RepoLockManager

# `RepoLockManager` 内部用 `os.path.realpath` 归一化 key，macOS 上 `/tmp` 是 `/private/tmp`
# 的符号链接。测试硬编码 `/tmp/...` 会与归一化结果错位，因此统一通过 `_TMP` 拼接。
_TMP = os.path.realpath("/tmp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def mgr():
    """Create an isolated RepoLockManager (no global singleton side-effects)."""
    m = RepoLockManager(idle_timeout=300, cleanup_interval=9999)
    yield m
    m.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRepoLockAcquire:

    def test_acquire_success(self, mgr: RepoLockManager):
        result = mgr.acquire(f"{_TMP}/repo1", "chat_A")
        assert result.success is True

    def test_acquire_reentrant(self, mgr: RepoLockManager):
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        result = mgr.acquire(f"{_TMP}/repo1", "chat_A")
        assert result.success is True
        # Check refcount via list_locks
        locks = mgr.list_locks()
        assert len(locks) == 1
        assert locks[0].refcount == 2

    def test_same_chat_different_request_owner_conflicts(
        self,
        mgr: RepoLockManager,
    ):
        mgr.acquire(
            f"{_TMP}/repo1",
            "chat_A",
            owner_id="run-1",
        )

        result = mgr.acquire(
            f"{_TMP}/repo1",
            "chat_A",
            owner_id="run-2",
        )

        assert result.success is False
        assert result.holder_chat_id == "chat_A"
        assert mgr.list_locks()[0].refcount == 1

    def test_same_request_owner_remains_reentrant(
        self,
        mgr: RepoLockManager,
    ):
        mgr.acquire(
            f"{_TMP}/repo1",
            "chat_A",
            owner_id="run-1",
        )

        result = mgr.acquire(
            f"{_TMP}/repo1",
            "chat_A",
            owner_id="run-1",
        )

        assert result.success is True
        assert mgr.list_locks()[0].refcount == 2

    def test_different_request_cannot_touch_or_release_holder(
        self,
        mgr: RepoLockManager,
    ):
        mgr.acquire(
            f"{_TMP}/repo1",
            "chat_A",
            owner_id="run-1",
        )
        before = mgr.get_lock_info(f"{_TMP}/repo1")
        assert before is not None

        time.sleep(0.001)
        mgr.touch(
            f"{_TMP}/repo1",
            "chat_A",
            owner_id="run-2",
        )
        mgr.release(
            f"{_TMP}/repo1",
            "chat_A",
            owner_id="run-2",
        )

        after = mgr.get_lock_info(f"{_TMP}/repo1")
        assert after is not None
        assert after.refcount == 1
        assert after.last_active_time == before.last_active_time

    def test_late_old_request_cannot_mutate_new_same_chat_holder(
        self,
        mgr: RepoLockManager,
    ):
        path = f"{_TMP}/repo1"
        mgr.acquire(path, "chat_A", owner_id="run-old")
        mgr.release(path, "chat_A", owner_id="run-old")
        mgr.acquire(path, "chat_A", owner_id="run-new")
        before = mgr.get_lock_info(path)
        assert before is not None

        time.sleep(0.001)
        mgr.touch(path, "chat_A", owner_id="run-old")
        mgr.release(path, "chat_A", owner_id="run-old")

        after = mgr.get_lock_info(path)
        assert after is not None
        assert after.refcount == 1
        assert after.last_active_time == before.last_active_time

    def test_acquire_conflict(self, mgr: RepoLockManager):
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is False
        assert result.holder_chat_id == "chat_A"
        assert result.locked_since is not None

    def test_acquire_p2p_bypass(self, mgr: RepoLockManager):
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B", is_p2p=True)
        assert result.success is True

    def test_release_and_reacquire(self, mgr: RepoLockManager):
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.acquire(f"{_TMP}/repo1", "chat_A")  # refcount = 2
        mgr.release(f"{_TMP}/repo1", "chat_A")  # refcount = 1
        # Still held by chat_A
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is False

        mgr.release(f"{_TMP}/repo1", "chat_A")  # refcount = 0 → removed
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is True

    def test_force_release(self, mgr: RepoLockManager):
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.force_release(f"{_TMP}/repo1")
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is True

    def test_path_normalization(self, mgr: RepoLockManager):
        """~/repo, $HOME/repo, $HOME/./repo should all resolve to the same key."""
        home = os.path.expanduser("~")
        real_home = os.path.realpath(home)

        mgr.acquire(f"{home}/test_repo_norm", "chat_A")
        # Same path via ~/
        result = mgr.acquire("~/test_repo_norm", "chat_A")
        assert result.success is True  # reentrant, same key

        # Different chat, same resolved path → conflict
        result2 = mgr.acquire(f"{real_home}/./test_repo_norm", "chat_B")
        assert result2.success is False

    def test_list_locks(self, mgr: RepoLockManager):
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.acquire(f"{_TMP}/repo2", "chat_B")
        locks = mgr.list_locks()
        assert len(locks) == 2
        paths = {l.root_path for l in locks}
        assert f"{_TMP}/repo1" in paths
        assert f"{_TMP}/repo2" in paths

    def test_idle_timeout_cleanup(self, mgr: RepoLockManager):
        mgr._idle_timeout = 1  # 1 second timeout for testing
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        # Simulate a leaked lock entry with refcount=0 (e.g. bug in caller)
        with mgr._mu:
            mgr._locks[f"{_TMP}/repo1"].refcount = 0
        time.sleep(1.5)  # Wait for lock to become idle
        mgr._cleanup_idle()
        # Lock should be cleaned up
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is True

    def test_cleanup_skips_active_refcount_on_idle(self, mgr: RepoLockManager):
        """H-1 fix: _cleanup_idle must NOT evict entries with refcount > 0 via idle_timeout.
        Also verifies single refcount case (test_cleanup_idle_skips_active_refcount)."""
        mgr._idle_timeout = 0  # everything is "idle"
        mgr._hard_timeout = 999999  # hard_timeout won't trigger

        # Case 1: refcount=2
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.acquire(f"{_TMP}/repo1", "chat_A")  # refcount = 2
        mgr._cleanup_idle()
        locks = mgr.list_locks()
        assert len(locks) == 1
        assert locks[0].refcount == 2
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is False

        # Cleanup for case 2
        mgr.force_release(f"{_TMP}/repo1")

        # Case 2: refcount=1
        mgr.acquire(f"{_TMP}/repo2", "chat_C")  # refcount = 1
        mgr._cleanup_idle()
        locks = mgr.list_locks()
        assert len(locks) == 1
        assert locks[0].refcount == 1
        assert locks[0].chat_id == "chat_C"

    def test_cleanup_evicts_zero_refcount(self, mgr: RepoLockManager):
        """F-01: _cleanup_idle evicts entries with refcount <= 0 and idle."""
        mgr._idle_timeout = 0
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.release(f"{_TMP}/repo1", "chat_A")  # refcount = 0, removed by release
        # Re-acquire to get refcount=1, then force refcount=0 without release
        mgr.acquire(f"{_TMP}/repo2", "chat_B")
        # Manually set refcount to 0 for testing
        with mgr._mu:
            mgr._locks[f"{_TMP}/repo2"].refcount = 0
        mgr._cleanup_idle()
        assert mgr.list_locks() == []

    def test_release_and_force_release_clean_token_mapping(self, mgr: RepoLockManager):
        """release() and force_release() clean up token mappings when lock is removed."""
        # release() path
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        token1 = mgr.path_to_token(f"{_TMP}/repo1")
        assert token1
        mgr.release(f"{_TMP}/repo1", "chat_A")
        assert mgr.token_to_path(token1) is None
        with mgr._mu:
            assert f"{_TMP}/repo1" not in mgr._path_to_token

        # force_release() path
        mgr.acquire(f"{_TMP}/repo2", "chat_B")
        token2 = mgr.path_to_token(f"{_TMP}/repo2")
        assert token2
        mgr.force_release(f"{_TMP}/repo2")
        assert mgr.token_to_path(token2) is None
        with mgr._mu:
            assert f"{_TMP}/repo2" not in mgr._path_to_token

    def test_cleanup_idle_skips_recently_touched(self, mgr: RepoLockManager):
        """AC-18: _cleanup_idle must NOT evict entries that are recently touched."""
        mgr._idle_timeout = 1  # 1 second timeout for testing
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.acquire(f"{_TMP}/repo1", "chat_A")  # refcount = 2

        time.sleep(0.6)
        mgr.touch(f"{_TMP}/repo1", "chat_A")  # refresh last_active_time
        time.sleep(0.6)
        # Total elapsed > 1s from acquire, but touch was recent (0.6s ago)
        mgr._cleanup_idle()

        # Lock should still be held because touch refreshed last_active_time
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is False  # still held by chat_A

    def test_acquire_none_path(self, mgr: RepoLockManager):
        result = mgr.acquire("", "chat_A")
        assert result.success is True  # empty path → no-op success

    def test_release_wrong_chat_noop(self, mgr: RepoLockManager):
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.release(f"{_TMP}/repo1", "chat_B")  # wrong chat → no-op
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is False  # still held by chat_A

    def test_cleanup_hard_timeout_evicts_active_lock(self, mgr: RepoLockManager, caplog):
        """Leaked active lock with an expired lease is eventually reclaimed."""
        import logging

        mgr._idle_timeout = 999999  # idle won't trigger
        mgr._hard_timeout = 10
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.acquire(f"{_TMP}/repo1", "chat_A")  # refcount = 2
        with mgr._mu:
            entry = mgr._locks[f"{_TMP}/repo1"]
            entry.acquired_at -= 20
            entry.last_active_time -= 20

        with caplog.at_level(logging.CRITICAL, logger="src.repo_lock"):
            mgr._cleanup_idle()

        # Lock should be force-reclaimed by hard_timeout
        assert mgr.list_locks() == []
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is True

        # Verify CRITICAL log emitted for hard-timeout force-reclaim
        critical_msgs = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert any("hard-timeout force-reclaimed" in r.message for r in critical_msgs)

    def test_hard_timeout_does_not_reclaim_renewed_long_running_lock(
        self,
        mgr: RepoLockManager,
    ):
        """A heartbeat renews the lease even after total runtime exceeds the cap."""
        mgr._idle_timeout = 300
        mgr._hard_timeout = 3600
        path = f"{_TMP}/repo-long-running"
        mgr.acquire(path, "chat_A", owner_id="run-long")
        with mgr._mu:
            entry = mgr._locks[path]
            entry.acquired_at -= 7200
            entry.last_active_time = time.monotonic()

        mgr._cleanup_idle()

        lock = mgr.get_lock_info(path)
        assert lock is not None
        assert lock.refcount == 1
        conflict = mgr.acquire(
            path,
            "chat_B",
            owner_id="run-other",
        )
        assert conflict.success is False

    def test_heartbeat_interval_is_bounded_by_one_third_of_lease(self):
        short_lease = RepoLockManager(
            idle_timeout=300,
            cleanup_interval=9999,
            hard_timeout=3,
        )
        default_lease = RepoLockManager(
            idle_timeout=300,
            cleanup_interval=9999,
            hard_timeout=3600,
        )
        try:
            assert 0 < short_lease.heartbeat_interval <= 1
            assert default_lease.heartbeat_interval == 30
        finally:
            short_lease.shutdown()
            default_lease.shutdown()

    def test_cleanup_phase3_warning_log(self, mgr: RepoLockManager, caplog):
        """Phase 3: active lock idle without touch emits WARNING but is NOT evicted."""
        import logging

        mgr._idle_timeout = 0  # everything is "idle"
        mgr._hard_timeout = 999999  # hard_timeout won't trigger
        mgr.acquire(f"{_TMP}/repo1", "chat_A")  # refcount = 1

        with caplog.at_level(logging.WARNING, logger="src.repo_lock"):
            mgr._cleanup_idle()

        # Lock must NOT be evicted (Phase 3 only warns)
        locks = mgr.list_locks()
        assert len(locks) == 1
        assert locks[0].refcount == 1

        # Verify WARNING log about idle active lock
        warn_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("active lock idle without touch" in r.message for r in warn_msgs)
        assert any("not evicted" in r.message.lower() for r in warn_msgs)

    def test_cleanup_hard_timeout_cleans_token_mapping(self, mgr: RepoLockManager):
        """Phase 2 hard_timeout eviction must also clean path↔token mappings."""
        mgr._idle_timeout = 999999
        mgr._hard_timeout = 0  # everything exceeds hard_timeout
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        token = mgr.path_to_token(f"{_TMP}/repo1")
        assert token
        assert mgr.token_to_path(token) == f"{_TMP}/repo1"

        mgr._cleanup_idle()

        # Token mapping should be cleaned up after hard-timeout eviction
        assert mgr.token_to_path(token) is None
        with mgr._mu:
            assert f"{_TMP}/repo1" not in mgr._path_to_token


class TestPathToken:
    """Tests for RepoLockManager.path_to_token / token_to_path."""

    def test_path_token_roundtrip_and_edge_cases(self, mgr: RepoLockManager):
        # Basic roundtrip
        token = mgr.path_to_token(f"{_TMP}/repo1")
        assert token  # non-empty
        assert len(token) == 32  # sha256 hex prefix
        assert mgr.token_to_path(token) == f"{_TMP}/repo1"

        # Deterministic
        assert mgr.path_to_token(f"{_TMP}/repo1") == token

        # Different paths give different tokens
        t2 = mgr.path_to_token(f"{_TMP}/repo2")
        assert t2 != token

        # Edge cases
        assert mgr.token_to_path("nonexistent") is None
        assert mgr.path_to_token("") == ""
        assert mgr.token_to_path("") is None


class TestGetLockInfo:
    """Tests for RepoLockManager.get_lock_info O(1) lookup."""

    def test_get_lock_info_lifecycle(self, mgr: RepoLockManager):
        # Not locked -> None
        assert mgr.get_lock_info(f"{_TMP}/repo1") is None
        assert mgr.get_lock_info("") is None

        # After acquire
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        info = mgr.get_lock_info(f"{_TMP}/repo1")
        assert info is not None
        assert info.chat_id == "chat_A"
        assert info.refcount == 1
        assert info.root_path == f"{_TMP}/repo1"
        assert info.idle_seconds >= 0

        # After release
        mgr.release(f"{_TMP}/repo1", "chat_A")
        assert mgr.get_lock_info(f"{_TMP}/repo1") is None

    def test_get_lock_info_normalizes_path(self, mgr: RepoLockManager):
        home = os.path.expanduser("~")
        mgr.acquire(f"{home}/test_info_repo", "chat_A")
        info = mgr.get_lock_info("~/test_info_repo")
        assert info is not None
        assert info.chat_id == "chat_A"


# ---------------------------------------------------------------------------
# is_p2p thread-local → RepoLock propagation tests
# ---------------------------------------------------------------------------


class TestIsP2PThreadLocalPropagation:
    """Verify that the is_p2p thread-local value propagates correctly through
    handler methods to RepoLockManager.hold / acquire, solving the broken
    parameter chain issue.
    """

    def test_programming_handle_message_reads_threadlocal_p2p(self):
        """handle_message() without explicit is_p2p should read thread-local."""
        from unittest.mock import MagicMock

        from src.thread import set_current_is_p2p

        set_current_is_p2p(True)
        try:
            mock_ctx = MagicMock()
            mock_ctx.settings.thread_programming_enabled = False
            mock_ctx.settings.coco_execution_timeout = 30
            mock_ctx.settings.card.collapsible_enabled = False

            from src.feishu.handlers.programming import CocoModeHandler
            handler = CocoModeHandler(mock_ctx)

            # Stub out methods that are not under test
            handler.reply_text = MagicMock()
            handler.reply_card = MagicMock(return_value="reply_1")
            handler.add_reaction = MagicMock()
            handler.register_message_project = MagicMock()
            handler.record_mode_transition = MagicMock()
            handler.inject_bridge_context = MagicMock(side_effect=lambda t, p, **kw: t)
            handler.get_working_dir = MagicMock(return_value=f"{_TMP}/test")
            handler.ensure_request_id = MagicMock(return_value="req-1")

            # Mock project with root_path
            mock_project = MagicMock()
            mock_project.project_id = "proj-1"
            mock_project.root_path = f"{_TMP}/test"

            # Mock repo_lock_manager on ctx
            mock_lock_mgr = MagicMock()
            mock_ctx.repo_lock_manager = mock_lock_mgr

            # Mock session so handle_message doesn't try to enter_mode
            mock_session = MagicMock()
            mock_session.session_id = "sess-1"
            mock_session.message_count = 1
            mock_session.last_query = "test"
            mock_ctx.coco_manager.get_session.return_value = mock_session

            # Call handle_message WITHOUT is_p2p — should read True from thread-local
            handler.handle_message("msg-1", "chat-1", "hello", project=mock_project)

            # is_p2p=True → lock acquire is skipped entirely (guard: `not is_p2p`)
            mock_lock_mgr.acquire.assert_not_called()
        finally:
            set_current_is_p2p(False)

    def test_programming_handle_message_threadlocal_false(self):
        """Thread-local is_p2p=False should call repo_lock_mgr.acquire()."""
        from unittest.mock import MagicMock

        from src.thread import set_current_is_p2p

        # is_p2p is now purely thread-local; no explicit parameter on handle_message
        set_current_is_p2p(False)
        try:
            mock_ctx = MagicMock()
            mock_ctx.settings.thread_programming_enabled = False
            mock_ctx.settings.coco_execution_timeout = 30
            mock_ctx.settings.card.collapsible_enabled = False

            from src.feishu.handlers.programming import CocoModeHandler
            handler = CocoModeHandler(mock_ctx)

            handler.reply_text = MagicMock()
            handler.reply_card = MagicMock(return_value="reply_1")
            handler.add_reaction = MagicMock()
            handler.register_message_project = MagicMock()
            handler.record_mode_transition = MagicMock()
            handler.inject_bridge_context = MagicMock(side_effect=lambda t, p, **kw: t)
            handler.get_working_dir = MagicMock(return_value=f"{_TMP}/test")
            handler.ensure_request_id = MagicMock(return_value="req-1")

            mock_project = MagicMock()
            mock_project.project_id = "proj-1"
            mock_project.root_path = f"{_TMP}/test"

            mock_lock_mgr = MagicMock()
            mock_lock_result = MagicMock()
            mock_lock_result.success = True
            mock_lock_mgr.acquire.return_value = mock_lock_result
            mock_ctx.repo_lock_manager = mock_lock_mgr

            mock_session = MagicMock()
            mock_session.session_id = "sess-1"
            mock_session.message_count = 1
            mock_session.last_query = "test"
            mock_ctx.coco_manager.get_session.return_value = mock_session

            handler.handle_message("msg-1", "chat-1", "hello", project=mock_project)

            # is_p2p=False → acquire should be called
            mock_lock_mgr.acquire.assert_called_once()
        finally:
            set_current_is_p2p(False)

    def test_engine_base_reads_threadlocal_p2p(self):
        """_safe_execute_engine() without explicit is_p2p should read thread-local."""
        from unittest.mock import MagicMock

        from src.thread import set_current_is_p2p

        set_current_is_p2p(True)
        try:
            mock_ctx = MagicMock()

            from src.feishu.handlers.engine_base import BaseEngineHandler
            handler = BaseEngineHandler(mock_ctx)

            mock_project = MagicMock()
            mock_project.root_path = f"{_TMP}/test"

            mock_lock_mgr = MagicMock()
            mock_lock_result = MagicMock()
            mock_lock_result.success = True
            mock_lock_mgr.hold.return_value.__enter__ = MagicMock(return_value=mock_lock_result)
            mock_lock_mgr.hold.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.repo_lock_manager = mock_lock_mgr

            handler.reply_text = MagicMock()

            executor = MagicMock()

            # Call without is_p2p — should read True from thread-local
            handler._safe_execute_engine(
                executor_func=executor,
                task_id="task-1",
                chat_id="chat-1",
                message_id="msg-1",
                project=mock_project,
                engine_name="TestEngine",
                reporter=MagicMock(),
                request_id="req-1",
            )

            mock_lock_mgr.hold.assert_called_once()
            call_kwargs = mock_lock_mgr.hold.call_args
            assert call_kwargs[1].get("is_p2p") is True
        finally:
            set_current_is_p2p(False)

    def test_system_worktree_execute_reads_threadlocal_p2p(self):
        """handle_worktree_execute() without explicit is_p2p should read thread-local."""
        from unittest.mock import MagicMock

        from src.thread import set_current_is_p2p

        set_current_is_p2p(True)
        try:
            mock_ctx = MagicMock()
            mock_ctx.settings.thread_programming_enabled = False

            from src.feishu.handlers.worktree import WorktreeHandler
            handler = WorktreeHandler(mock_ctx)

            mock_project = MagicMock()
            mock_project.project_id = "proj-1"
            mock_project.root_path = f"{_TMP}/test"
            mock_ctx.project_manager.get_active_project.return_value = mock_project

            mock_lock_mgr = MagicMock()
            mock_lock_result = MagicMock()
            mock_lock_result.success = True
            mock_lock_mgr.hold.return_value.__enter__ = MagicMock(return_value=mock_lock_result)
            mock_lock_mgr.hold.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.repo_lock_manager = mock_lock_mgr

            handler.reply_text = MagicMock()
            handler.send_card_to_chat = MagicMock(return_value="progress-mid")
            handler.update_card = MagicMock()

            # Mock worktree manager
            mock_wt_mgr = MagicMock()
            mock_state = MagicMock()
            mock_state.units = []
            mock_state.last_error = None
            mock_state.merge_entry_ready = False
            mock_wt_mgr.get_state.return_value = mock_state
            mock_wt_mgr.execute_goal.return_value = mock_state
            handler._worktree_manager = MagicMock(return_value=mock_wt_mgr)

            # Call without is_p2p — should read True from thread-local
            handler.handle_worktree_execute("msg-1", "chat-1", "fix bug", project=mock_project)

            mock_lock_mgr.hold.assert_called_once()
            call_kwargs = mock_lock_mgr.hold.call_args
            assert call_kwargs[1].get("is_p2p") is True
        finally:
            set_current_is_p2p(False)

    def test_default_threadlocal_is_false(self):
        """When thread-local is not set, is_p2p should default to False."""
        from unittest.mock import MagicMock

        from src.thread import set_current_is_p2p
        from src.thread.manager import _current_is_p2p

        # Ensure thread-local is cleared
        if hasattr(_current_is_p2p, "value"):
            delattr(_current_is_p2p, "value")
        try:
            mock_ctx = MagicMock()

            from src.feishu.handlers.engine_base import BaseEngineHandler
            handler = BaseEngineHandler(mock_ctx)

            mock_project = MagicMock()
            mock_project.root_path = f"{_TMP}/test"

            mock_lock_mgr = MagicMock()
            mock_lock_result = MagicMock()
            mock_lock_result.success = True
            mock_lock_mgr.hold.return_value.__enter__ = MagicMock(return_value=mock_lock_result)
            mock_lock_mgr.hold.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.repo_lock_manager = mock_lock_mgr

            handler.reply_text = MagicMock()

            handler._safe_execute_engine(
                executor_func=MagicMock(),
                task_id="task-1",
                chat_id="chat-1",
                message_id="msg-1",
                project=mock_project,
                engine_name="TestEngine",
                reporter=MagicMock(),
                request_id="req-1",
            )

            mock_lock_mgr.hold.assert_called_once()
            call_kwargs = mock_lock_mgr.hold.call_args
            assert call_kwargs[1].get("is_p2p") is False
        finally:
            set_current_is_p2p(False)

    def test_sandbox_executor_no_independent_lock(self):
        """SandboxExecutor.execute() should NOT acquire repo lock independently.

        After the refactor, SandboxExecutor only logs a warning sentinel if the
        lock is held by a different chat — it no longer calls hold().
        """
        from unittest.mock import MagicMock

        from src.sandbox.executor import SandboxExecutor

        mock_lock_mgr = MagicMock()
        mock_lock_mgr.get_lock_info.return_value = None  # no lock held

        with patch("src.sandbox.executor.get_repo_lock_manager", return_value=mock_lock_mgr, create=True):
            with patch("src.repo_lock.get_repo_lock_manager", return_value=mock_lock_mgr):
                mock_subprocess = MagicMock()
                mock_process = MagicMock()
                mock_process.returncode = 0
                mock_process.stdout = ""
                mock_process.stderr = ""
                mock_subprocess.run.return_value = mock_process

                executor = SandboxExecutor(subprocess_executor=mock_subprocess)
                result = executor.execute("echo test", cwd=f"{_TMP}/test", chat_id="chat-1")

                # hold() should NOT be called — no independent lock acquisition
                mock_lock_mgr.hold.assert_not_called()
                assert result.success is True


# ---------------------------------------------------------------------------
# hold() context manager tests (replaces former RepoLockGuard tests)
# ---------------------------------------------------------------------------


class TestRepoLockHold:
    """Tests for RepoLockManager.hold() — context-manager lock coordination."""

    def test_hold_no_root_path(self, mgr: RepoLockManager):
        """When root_path is empty, hold() still succeeds (p2p-like bypass)."""
        # hold() with empty path should not raise; body can execute freely.
        body = MagicMock(return_value="ok")
        with mgr.hold("", "chat_A") as lock_result:
            body() if lock_result.success else None
        # Empty path → acquire returns success immediately (no-op lock)
        assert body.called

    def test_hold_success(self, mgr: RepoLockManager):
        """Body executes under lock via hold() and lock is released on exit."""
        body = MagicMock(return_value=42)
        with mgr.hold(f"{_TMP}/repo", "chat_A") as lock_result:
            assert lock_result.success
            result = body()
        assert result == 42
        body.assert_called_once()
        # Lock should be released after context exit
        assert mgr.list_locks() == []

    def test_hold_conflict(self, mgr: RepoLockManager):
        """When another chat holds the lock, hold() raises LockConflictError with correct attributes."""
        mgr.acquire(f"{_TMP}/repo", "chat_A")
        body = MagicMock()
        with pytest.raises(LockConflictError) as exc_info:
            with mgr.hold(f"{_TMP}/repo", "chat_B"):
                body()
        body.assert_not_called()
        assert exc_info.value.holder_chat_id == "chat_A"
        assert exc_info.value.root_path == f"{_TMP}/repo"

    def test_hold_p2p_bypass(self, mgr: RepoLockManager):
        """With is_p2p=True, lock conflict is bypassed via hold()."""
        mgr.acquire(f"{_TMP}/repo", "chat_A")
        body = MagicMock(return_value="bypassed")
        with mgr.hold(f"{_TMP}/repo", "chat_B", is_p2p=True) as lock_result:
            assert lock_result.success
            result = body()
        assert result == "bypassed"
        body.assert_called_once()

    def test_hold_body_exception_releases_lock(self, mgr: RepoLockManager):
        """When body raises, lock is still released on context exit."""
        with pytest.raises(ValueError, match="boom"):
            with mgr.hold(f"{_TMP}/repo", "chat_A") as lock_result:
                assert lock_result.success
                raise ValueError("boom")
        # Lock must be released despite the exception
        assert mgr.list_locks() == []


# ---------------------------------------------------------------------------
# Lazy cleanup thread tests
# ---------------------------------------------------------------------------


class TestLazyCleanupThread:
    """Task 23: RepoLockManager cleanup thread is lazily started."""

    def test_lazy_thread_lifecycle(self):
        m = RepoLockManager(idle_timeout=999, cleanup_interval=999)
        try:
            # No thread on construction
            assert m._cleanup_thread is None

            # Thread starts on first acquire
            m.acquire(f"{_TMP}/repo", "chat_A")
            assert m._cleanup_thread is not None
            assert m._cleanup_thread.is_alive()
        finally:
            # Shutdown clears thread ref
            m.shutdown()
            assert m._cleanup_thread is None
# Concurrency tests
# ---------------------------------------------------------------------------


class TestRepoLockConcurrency:
    """Thread-safety tests for RepoLockManager under concurrent access."""

    def test_concurrent_acquire_same_path(self, mgr: RepoLockManager):
        """Only one chat wins when multiple threads race to acquire the same path."""
        results: dict[str, AcquireResult] = {}
        barrier = threading.Barrier(10)

        def try_acquire(chat_id):
            barrier.wait()
            results[chat_id] = mgr.acquire(f"{_TMP}/contested", chat_id)

        threads = [threading.Thread(target=try_acquire, args=(f"chat_{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [cid for cid, r in results.items() if r.success]
        losers = [cid for cid, r in results.items() if not r.success]
        assert len(winners) == 1
        assert len(losers) == 9

    def test_concurrent_acquire_release(self, mgr: RepoLockManager):
        """Rapid acquire/release cycles from multiple threads don't corrupt state."""
        errors = []

        def cycle(chat_id, path):
            try:
                for _ in range(50):
                    r = mgr.acquire(path, chat_id)
                    if r.success:
                        mgr.release(path, chat_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=cycle, args=(f"chat_{i}", f"{_TMP}/repo_{i % 3}"))
            for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors: {errors}"

    def test_concurrent_hold_execute(self, mgr: RepoLockManager):
        """hold() from multiple threads on different paths is safe."""
        results = []

        def run(idx):
            with mgr.hold(f"{_TMP}/repo_{idx}", f"chat_{idx}"):
                results.append(idx)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == list(range(10))

    def test_cleanup_concurrent_with_acquire_release(self, mgr: RepoLockManager):
        """_cleanup_idle running concurrently with acquire/release must not corrupt state."""
        mgr._idle_timeout = 0
        mgr._hard_timeout = 0
        errors = []

        def cleanup_loop():
            try:
                for _ in range(50):
                    mgr._cleanup_idle()
            except Exception as e:
                errors.append(e)

        def acquire_release_loop(chat_id, path):
            try:
                for _ in range(50):
                    r = mgr.acquire(path, chat_id)
                    if r.success:
                        mgr.touch(path, chat_id)
                        mgr.release(path, chat_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cleanup_loop)]
        threads += [
            threading.Thread(target=acquire_release_loop, args=(f"chat_{i}", f"{_TMP}/repo_{i % 3}"))
            for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors: {errors}"


class TestRepoLockInfoFrozen:
    """AC-R03: RepoLockInfo must be immutable (frozen dataclass)."""

    def test_frozen_raises_on_assignment(self):
        info = RepoLockInfo(
            root_path=f"{_TMP}/test",
            chat_id="chat1",
            refcount=1,
            acquired_at=100.0,
            last_active_time=200.0,
            idle_seconds=5.0,
        )
        with pytest.raises(AttributeError):
            info.chat_id = "chat2"
        with pytest.raises(AttributeError):
            info.refcount = 99


class TestCleanupLoopResilience:
    """_cleanup_loop must survive exceptions from _cleanup_idle."""

    def test_cleanup_loop_survives_and_logs_exception(self, mgr: RepoLockManager, caplog):
        """If _cleanup_idle raises, the loop continues and the error is logged."""
        import logging

        call_count = 0
        original_cleanup_idle = mgr._cleanup_idle

        def flaky_cleanup():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            mgr._stop_event.set()
            original_cleanup_idle()

        mgr._cleanup_interval = 0.01
        mgr._cleanup_idle = flaky_cleanup  # type: ignore[assignment]

        with caplog.at_level(logging.ERROR, logger="src.repo_lock"):
            mgr._cleanup_loop()

        assert call_count == 2, f"Expected 2 calls, got {call_count}"
        assert any("RepoLock: cleanup error" in r.message for r in caplog.records)
        assert any("boom" in (r.exc_text or "") for r in caplog.records)


# ---------------------------------------------------------------------------
# Hard-timeout callback tests
# ---------------------------------------------------------------------------


class TestHardTimeoutCallback:
    """Verify on_hard_timeout_reclaim callback invocation and exception safety."""

    def test_callback_invoked_on_hard_timeout(self):
        """Phase 2 hard-timeout eviction invokes the registered callback with (path, chat_id)."""
        reclaimed: list[tuple[str, str]] = []

        def on_reclaim(path: str, chat_id: str) -> None:
            reclaimed.append((path, chat_id))

        m = RepoLockManager(
            idle_timeout=999999,
            cleanup_interval=9999,
            hard_timeout=0,
            on_hard_timeout_reclaim=on_reclaim,
        )
        try:
            m.acquire(f"{_TMP}/repo_cb", "chat_X")
            m._cleanup_idle()

            assert len(reclaimed) == 1
            assert reclaimed[0] == (f"{_TMP}/repo_cb", "chat_X")
            # Lock should be evicted
            assert m.list_locks() == []
        finally:
            m.shutdown()

    def test_callback_exception_does_not_crash_cleanup(self, caplog):
        """If the callback raises, cleanup still completes and the error is logged."""
        import logging

        def bad_callback(path: str, chat_id: str) -> None:
            raise RuntimeError("callback exploded")

        m = RepoLockManager(
            idle_timeout=999999,
            cleanup_interval=9999,
            hard_timeout=0,
            on_hard_timeout_reclaim=bad_callback,
        )
        try:
            m.acquire(f"{_TMP}/repo_cb2", "chat_Y")

            with caplog.at_level(logging.ERROR, logger="src.repo_lock"):
                m._cleanup_idle()  # must not raise

            # Lock should still be evicted despite callback failure
            assert m.list_locks() == []

            # Error should be logged
            assert any("SimpleEvent callback failed" in r.message for r in caplog.records)
        finally:
            m.shutdown()


# ---------------------------------------------------------------------------
# AC-R02: Orphan token cleanup + capacity limit
# ---------------------------------------------------------------------------

class TestOrphanTokenCleanup:
    """Verify _cleanup_idle removes orphan token mappings (Phase 4)."""

    def test_cleanup_removes_orphan_tokens_preserves_active(self):
        """Orphan tokens are removed; tokens for active locks survive."""
        m = RepoLockManager(idle_timeout=999, cleanup_interval=999)
        try:
            # Active lock with token
            m.acquire(f"{_TMP}/active_repo", "chatA")
            active_token = m.path_to_token(f"{_TMP}/active_repo")

            # Orphan token (no acquire)
            orphan_token = m.path_to_token(f"{_TMP}/orphan_repo")
            assert orphan_token
            assert m.token_to_path(orphan_token) is not None

            m._cleanup_idle()

            # Active token survives
            assert m.token_to_path(active_token) is not None
            # Orphan is removed
            assert m.token_to_path(orphan_token) is None
            with m._mu:
                assert f"{_TMP}/orphan_repo" not in m._path_to_token
        finally:
            m.shutdown()


class TestTokenMappingCapacityLimit:
    """Verify path_to_token enforces _TOKEN_MAP_CAPACITY."""

    def test_capacity_limit_purges_orphans_and_respects_active_locks(self):
        m = RepoLockManager(idle_timeout=999, cleanup_interval=999)
        m._TOKEN_MAP_CAPACITY = 10
        try:
            # Fill up with orphan tokens (no acquire)
            for i in range(10):
                m.path_to_token(f"{_TMP}/cap_repo_{i}")
            with m._mu:
                assert len(m._token_to_path) == 10

            # Next token should trigger orphan purge (all are orphans)
            new_token = m.path_to_token(f"{_TMP}/cap_repo_new")
            assert new_token
            with m._mu:
                assert len(m._token_to_path) == 1

            # With active locks: capacity full of non-orphans
            m._TOKEN_MAP_CAPACITY = 5
            for i in range(5):
                m.acquire(f"{_TMP}/locked_repo_{i}", "chatA")
                m.path_to_token(f"{_TMP}/locked_repo_{i}")

            # Adding one more with full capacity - token returned but not cached
            extra_token = m.path_to_token(f"{_TMP}/extra_repo")
            assert extra_token
            assert m.token_to_path(extra_token) is None
        finally:
            m.shutdown()


# ======================================================================
# FS-24: RepoLock double-release tests
# ======================================================================


class TestRepoLockDoubleRelease:
    """Verify that double-releasing a repo lock is safe (no exception, no side-effects)."""

    def test_double_release_and_wrong_chat_are_noop(self):
        m = RepoLockManager(idle_timeout=999, cleanup_interval=999, hard_timeout=9999)
        try:
            # Double release same chat
            result = m.acquire(f"{_TMP}/dr_repo", "chat_a")
            assert result.success
            m.release(f"{_TMP}/dr_repo", "chat_a")
            m.release(f"{_TMP}/dr_repo", "chat_a")  # no-op
            assert m.get_lock_info(f"{_TMP}/dr_repo") is None

            # Wrong chat release is noop
            m.acquire(f"{_TMP}/dr_repo2", "chat_a")
            m.release(f"{_TMP}/dr_repo2", "chat_b")
            info = m.get_lock_info(f"{_TMP}/dr_repo2")
            assert info is not None
            assert info.chat_id == "chat_a"

            # Release nonexistent path is noop
            m.release(f"{_TMP}/never_acquired", "chat_x")

            # Double force_release is safe
            m.acquire(f"{_TMP}/fr_repo", "chat_a")
            m.force_release(f"{_TMP}/fr_repo")
            m.force_release(f"{_TMP}/fr_repo")
            assert m.get_lock_info(f"{_TMP}/fr_repo") is None
        finally:
            m.shutdown()


# ---------------------------------------------------------------------------
# AC-R1: is_held_by() tests
# ---------------------------------------------------------------------------


class TestIsHeldBy:
    """AC-R1: is_held_by returns correct bool and is thread-safe."""

    def test_is_held_by_basic_and_edge_cases(self, mgr):
        """acquire -> True; release -> False; invalid/empty -> False."""
        # Edge cases
        assert mgr.is_held_by("", "chat_x") is False
        assert mgr.is_held_by(f"{_TMP}/nonexistent_path_xyz", "chat_x") is False

        # Normal lifecycle
        mgr.acquire(f"{_TMP}/held_repo", "chat_x")
        assert mgr.is_held_by(f"{_TMP}/held_repo", "chat_x") is True
        assert mgr.is_held_by(f"{_TMP}/held_repo", "chat_y") is False
        mgr.release(f"{_TMP}/held_repo", "chat_x")
        assert mgr.is_held_by(f"{_TMP}/held_repo", "chat_x") is False

    def test_is_held_by_thread_safety(self, mgr):
        """100 concurrent threads calling is_held_by produce no exceptions."""
        mgr.acquire(f"{_TMP}/ts_repo", "chat_ts")
        barrier = threading.Barrier(100)
        errors = []

        def worker():
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    mgr.is_held_by(f"{_TMP}/ts_repo", "chat_ts")
                    mgr.is_held_by(f"{_TMP}/ts_repo", "chat_other")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread safety errors: {errors}"
        mgr.release(f"{_TMP}/ts_repo", "chat_ts")


class TestRepoLockE2ELifecycle:
    """End-to-end lock lifecycle integration test with a real RepoLockManager."""

    def test_e2e_lock_lifecycle(self, mgr: RepoLockManager):
        """acquire(A) → list_locks visible → release(A) → acquire(B) succeeds."""
        # Step 1: chat_A acquires
        result_a = mgr.acquire(f"{_TMP}/e2e_repo", "chat_A")
        assert result_a.success is True

        # Step 2: lock is visible in list_locks
        locks = mgr.list_locks()
        assert any(lock.chat_id == "chat_A" and f"{_TMP}/e2e_repo" in lock.root_path for lock in locks)

        # Step 3: chat_B cannot acquire while A holds
        result_b_conflict = mgr.acquire(f"{_TMP}/e2e_repo", "chat_B")
        assert result_b_conflict.success is False

        # Step 4: chat_A releases
        mgr.release(f"{_TMP}/e2e_repo", "chat_A")

        # Step 5: lock is no longer visible
        locks_after = mgr.list_locks()
        assert not any(f"{_TMP}/e2e_repo" in lock.root_path for lock in locks_after)

        # Step 6: chat_B can now acquire
        result_b_ok = mgr.acquire(f"{_TMP}/e2e_repo", "chat_B")
        assert result_b_ok.success is True

        # Cleanup
        mgr.release(f"{_TMP}/e2e_repo", "chat_B")

    def test_shutdown_during_active_hold(self):
        """shutdown() while a thread holds a lock must not deadlock or raise."""
        m = RepoLockManager(idle_timeout=300, cleanup_interval=9999)
        errors = []

        def hold_lock():
            try:
                m.acquire(f"{_TMP}/shutdown_repo", "chat_hold")
                time.sleep(2)
                m.release(f"{_TMP}/shutdown_repo", "chat_hold")
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=hold_lock)
        t.start()
        time.sleep(0.1)  # Ensure thread has acquired the lock

        # shutdown() should not deadlock
        m.shutdown()
        t.join(timeout=5)

        assert not t.is_alive(), "Thread is still alive — possible deadlock"
        assert not errors, f"Unexpected errors during hold: {errors}"


# ---------------------------------------------------------------------------
# Task 15: Path normalization edge cases
# ---------------------------------------------------------------------------


class TestNormalizePathEdgeCases:
    """Edge cases for normalize_repo_path: ../, trailing slash, symlinks."""

    def test_dot_dot_relative_path(self, mgr: RepoLockManager):
        """../bar and the resolved equivalent must be the same lock key."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # tmpdir/foo/../bar == tmpdir/bar
            path_with_dotdot = os.path.join(tmpdir, "foo", "..", "bar")
            path_resolved = os.path.join(tmpdir, "bar")

            mgr.acquire(path_resolved, "chat_A")
            result = mgr.acquire(path_with_dotdot, "chat_B")
            assert result.success is False
            assert result.holder_chat_id == "chat_A"

    def test_trailing_slash_normalization(self, mgr: RepoLockManager):
        """Trailing slash variants of a repo path must resolve to the same lock key."""
        mgr.acquire(f"{_TMP}/trailing_slash_repo", "chat_A")
        result = mgr.acquire(f"{_TMP}/trailing_slash_repo/", "chat_B")
        assert result.success is False

    def test_symlink_resolution(self, mgr: RepoLockManager):
        """Symlinked path resolves to the same key as the real target."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = os.path.join(tmpdir, "real_repo")
            os.makedirs(real_dir)
            link_path = os.path.join(tmpdir, "link_repo")
            os.symlink(real_dir, link_path)

            mgr.acquire(real_dir, "chat_A")
            result = mgr.acquire(link_path, "chat_B")
            assert result.success is False
            assert result.holder_chat_id == "chat_A"


# ---------------------------------------------------------------------------
# Task 16: Blocked chats notification tests
# ---------------------------------------------------------------------------


class TestBlockedChatsNotification:
    """on_release fires with correct blocked chat_ids when lock is released."""

    def test_release_and_force_release_notify_blocked_chats(self, mgr: RepoLockManager):
        """Both release() and force_release() fire on_release with blocked chats."""
        notifications: list[tuple] = []

        def handler(path, chats):
            notifications.append((path, chats))

        mgr.on_release.subscribe(handler)

        # Normal release
        mgr.acquire(f"{_TMP}/notify_repo", "chat_A")
        mgr.acquire(f"{_TMP}/notify_repo", "chat_B")
        mgr.acquire(f"{_TMP}/notify_repo", "chat_C")
        mgr.release(f"{_TMP}/notify_repo", "chat_A")

        assert len(notifications) == 1
        path, chats = notifications[0]
        assert f"{_TMP}/notify_repo" in path or path == os.path.realpath(f"{_TMP}/notify_repo")
        assert "chat_B" in chats
        assert "chat_C" in chats

        # Force release
        notifications.clear()
        mgr.acquire(f"{_TMP}/force_notify", "chat_A")
        mgr.acquire(f"{_TMP}/force_notify", "chat_B")  # blocked
        mgr.force_release(f"{_TMP}/force_notify")

        assert len(notifications) == 1
        assert "chat_B" in notifications[0][1]

    def test_blocked_chats_bounded(self, mgr: RepoLockManager):
        """At most 10 blocked chat_ids are tracked per lock."""
        mgr.acquire(f"{_TMP}/bounded_repo", "holder_chat")

        for i in range(15):
            mgr.acquire(f"{_TMP}/bounded_repo", f"blocked_{i}")

        key = os.path.realpath(f"{_TMP}/bounded_repo")
        with mgr._mu:
            blocked = mgr._blocked_chats.get(key, set())
            assert len(blocked) <= 10


# ---------------------------------------------------------------------------
# Task 17: Singleton thread safety
# ---------------------------------------------------------------------------


class TestSingletonThreadSafety:
    """get_repo_lock_manager() must return the same instance from concurrent threads."""

    def test_concurrent_get_returns_same_instance(self):
        from src.repo_lock import _reset_repo_lock_manager_for_testing, get_repo_lock_manager

        _reset_repo_lock_manager_for_testing()
        try:
            instances: list = []
            barrier = threading.Barrier(10)

            def getter():
                barrier.wait()
                instances.append(id(get_repo_lock_manager()))

            threads = [threading.Thread(target=getter) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All threads should get the same instance
            assert len(set(instances)) == 1
        finally:
            _reset_repo_lock_manager_for_testing()


# ---------------------------------------------------------------------------
# Task 18: Lock after shutdown
# ---------------------------------------------------------------------------


class TestLockAfterShutdown:
    """acquire() after shutdown() must not crash."""

    def test_acquire_after_shutdown_no_crash(self):
        m = RepoLockManager(idle_timeout=300, cleanup_interval=9999)
        m.acquire(f"{_TMP}/pre_shutdown", "chat_A")
        m.shutdown()

        # acquire after shutdown should still work (creates new thread)
        result = m.acquire(f"{_TMP}/post_shutdown", "chat_B")
        assert result.success is True
        m.shutdown()


# ---------------------------------------------------------------------------
# Task 37: shutdown_if_active idempotent module-level function
# ---------------------------------------------------------------------------


class TestRepoShutdownIfActive:
    """shutdown_if_active() is idempotent and safe when no instance exists."""

    def test_idempotent_and_no_instance(self):
        from src import repo_lock as _mod
        _orig = _mod._instance
        try:
            # No instance is noop
            _mod._instance = None
            _mod.shutdown_if_active()  # should not raise

            # Double call is idempotent
            m = RepoLockManager(idle_timeout=300, cleanup_interval=9999)
            _mod._instance = m
            _mod.shutdown_if_active()
            _mod.shutdown_if_active()  # second call should not raise
        finally:
            _mod._instance = _orig
