"""Core unit tests for RepoLockManager acquire, hold, lookup, and P2P propagation."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.repo_lock import LockConflictError, RepoLockInfo, RepoLockManager

# `RepoLockManager` 内部用 `os.path.realpath` 归一化 key，macOS 上 `/tmp` 是 `/private/tmp`
# 的符号链接。测试硬编码 `/tmp/...` 会与归一化结果错位，因此统一通过 `_TMP` 拼接。
_TMP = os.path.realpath("/tmp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def isolated_lock_manager(**kwargs: object) -> Iterator[RepoLockManager]:
    """Create an isolated RepoLockManager and shut it down cleanly on exit."""
    kwargs.setdefault("idle_timeout", 300)
    kwargs.setdefault("cleanup_interval", 9999)
    manager = RepoLockManager(**kwargs)  # type: ignore[arg-type]
    try:
        yield manager
    finally:
        manager.shutdown()


@pytest.fixture()
def mgr() -> Iterator[RepoLockManager]:
    """Create an isolated RepoLockManager (no global singleton side-effects)."""
    with isolated_lock_manager() as manager:
        yield manager


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

    def test_acquire_none_path(self, mgr: RepoLockManager):
        result = mgr.acquire("", "chat_A")
        assert result.success is True  # empty path → no-op success

    def test_release_wrong_chat_noop(self, mgr: RepoLockManager):
        mgr.acquire(f"{_TMP}/repo1", "chat_A")
        mgr.release(f"{_TMP}/repo1", "chat_B")  # wrong chat → no-op
        result = mgr.acquire(f"{_TMP}/repo1", "chat_B")
        assert result.success is False  # still held by chat_A


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

    def test_command_executor_has_no_independent_lock(self):
        """CommandExecutor.execute() does not acquire the routing repo lock."""
        from unittest.mock import MagicMock

        from src.command_executor import CommandExecutor

        mock_lock_mgr = MagicMock()
        mock_lock_mgr.get_lock_info.return_value = None  # no lock held

        with patch("src.repo_lock.get_repo_lock_manager", return_value=mock_lock_mgr):
            mock_subprocess = MagicMock()
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = ""
            mock_process.stderr = ""
            mock_subprocess.run.return_value = mock_process

            executor = CommandExecutor(subprocess_executor=mock_subprocess)
            result = executor.execute(
                "echo test", cwd=f"{_TMP}/test", chat_id="chat-1"
            )

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


# ======================================================================
# FS-24: RepoLock double-release tests
# ======================================================================


class TestRepoLockDoubleRelease:
    """Verify that double-releasing a repo lock is safe (no exception, no side-effects)."""

    def test_double_release_and_wrong_chat_are_noop(self):
        with isolated_lock_manager(hard_timeout=9999, idle_timeout=999, cleanup_interval=999) as m:
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
