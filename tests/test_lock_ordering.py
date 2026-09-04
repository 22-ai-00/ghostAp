"""FS-07 / FS-01: Tests for src/utils/lock_order.py — lock ordering detection."""

from __future__ import annotations

import logging

import pytest

from src.utils.lock_order import (
    LockLevel,
    _get_held,
    disable_lock_order_check,
    enable_lock_order_check,
    ordered_lock,
    ordered_rlock,
)


@pytest.fixture(autouse=True)
def _enable_and_cleanup():
    """Enable lock-order checking and clear thread-local state."""
    enable_lock_order_check()
    _get_held().clear()
    yield
    _get_held().clear()
    disable_lock_order_check()


class TestLockLevel:
    def test_ordering_values(self):
        assert LockLevel.ACP_ACTIVATION < LockLevel.PROJECT_MANAGER
        assert LockLevel.PROJECT_MANAGER < LockLevel.CHAT_LOCK_CTX
        assert LockLevel.CHAT_LOCK_CTX < LockLevel.CHAT_LOCK_MGR
        assert LockLevel.CHAT_LOCK_MGR < LockLevel.REPO_LOCK

    def test_all_levels_distinct(self):
        values = [int(v) for v in LockLevel]
        assert len(values) == len(set(values))


class TestOrderedLock:
    def test_basic_acquire_release(self):
        lock = ordered_lock(LockLevel.REPO_LOCK)
        lock.acquire()
        assert lock.locked()
        lock.release()
        assert not lock.locked()

    def test_context_manager(self):
        lock = ordered_lock(LockLevel.REPO_LOCK)
        with lock:
            assert lock.locked()
        assert not lock.locked()

    def test_correct_order_no_warning(self, caplog):
        outer = ordered_lock(LockLevel.PROJECT_MANAGER, name="outer")
        inner = ordered_lock(LockLevel.REPO_LOCK, name="inner")
        with caplog.at_level(logging.WARNING):
            with outer:
                with inner:
                    pass
        assert "Lock ordering violation" not in caplog.text

    def test_reverse_order_logs_warning(self, caplog):
        outer = ordered_lock(LockLevel.REPO_LOCK, name="inner_first")
        inner = ordered_lock(LockLevel.PROJECT_MANAGER, name="outer_second")
        with caplog.at_level(logging.WARNING):
            with outer:
                with inner:
                    pass
        assert "Lock ordering violation" in caplog.text
        assert "outer_second" in caplog.text

    def test_same_level_logs_warning(self, caplog):
        lock_a = ordered_lock(LockLevel.REPO_LOCK, name="a")
        lock_b = ordered_lock(LockLevel.REPO_LOCK, name="b")
        with caplog.at_level(logging.WARNING):
            with lock_a:
                with lock_b:
                    pass
        assert "Lock ordering violation" in caplog.text


class TestOrderedRLock:
    def test_reentrant_no_double_warning(self, caplog):
        rlock = ordered_rlock(LockLevel.PROJECT_MANAGER, name="pm")
        with caplog.at_level(logging.WARNING):
            with rlock:
                with rlock:
                    pass
        assert "Lock ordering violation" not in caplog.text

    def test_rlock_then_inner_correct_order(self, caplog):
        outer = ordered_rlock(LockLevel.PROJECT_MANAGER, name="pm")
        inner = ordered_lock(LockLevel.REPO_LOCK, name="repo")
        with caplog.at_level(logging.WARNING):
            with outer:
                with inner:
                    pass
        assert "Lock ordering violation" not in caplog.text


class TestDisabledChecking:
    def test_no_warning_when_disabled(self, caplog):
        disable_lock_order_check()
        outer = ordered_lock(LockLevel.REPO_LOCK, name="inner_first")
        inner = ordered_lock(LockLevel.PROJECT_MANAGER, name="outer_second")
        with caplog.at_level(logging.WARNING):
            with outer:
                with inner:
                    pass
        assert "Lock ordering violation" not in caplog.text


class TestIntegrationWithRealManagers:
    def test_project_manager_lock_level(self):
        """ProjectManager._lock should use its declared ordered level."""
        from src.project.manager import ProjectManager
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("GHOSTAP_LOCK_ORDER_CHECK", "1")
            pm = ProjectManager()
            assert pm._lock._level == int(LockLevel.PROJECT_MANAGER)

    def test_acp_activation_lock_is_outside_project_manager(self):
        """ACP lifecycle commits may safely enter ProjectManager."""
        activation = ordered_rlock(
            LockLevel.ACP_ACTIVATION,
            name="activation",
        )
        project = ordered_rlock(
            LockLevel.PROJECT_MANAGER,
            name="project",
        )
        enable_lock_order_check(strict=True)
        with activation:
            with project:
                pass

    def test_repo_lock_manager_lock_level(self):
        """RepoLockManager._mu should use the innermost ordered level."""
        from src.repo_lock import RepoLockManager
        try:
            mgr = RepoLockManager(idle_timeout=999, cleanup_interval=999, hard_timeout=9999)
            assert mgr._mu._level == int(LockLevel.REPO_LOCK)
        finally:
            mgr.shutdown()


class TestEngineManagerLockLevel:
    """AC-R32: ENGINE_MANAGER is the outermost lock level (-1)."""

    def test_engine_manager_is_outermost(self):
        assert LockLevel.ENGINE_MANAGER == -1
        assert LockLevel.ENGINE_MANAGER < LockLevel.ENGINE_INSTANCE
        # ENGINE_MANAGER should be the minimum of all levels
        assert LockLevel.ENGINE_MANAGER == min(LockLevel)

    def test_engine_manager_to_inner_no_warning(self, caplog):
        """Acquiring ENGINE_MANAGER then inner lock should not warn."""
        import logging
        outer = ordered_lock(LockLevel.ENGINE_MANAGER, name="em")
        inner = ordered_rlock(LockLevel.ENGINE_INSTANCE, name="ei")
        with caplog.at_level(logging.WARNING):
            with outer:
                with inner:
                    pass
        assert "Lock ordering violation" not in caplog.text

    def test_inner_to_engine_manager_warns(self, caplog):
        """Acquiring inner lock then ENGINE_MANAGER should warn."""
        import logging
        inner = ordered_rlock(LockLevel.ENGINE_INSTANCE, name="ei")
        outer = ordered_lock(LockLevel.ENGINE_MANAGER, name="em")
        with caplog.at_level(logging.WARNING):
            with inner:
                with outer:
                    pass
        assert "Lock ordering violation" in caplog.text


class TestLeafLockAnnotationScan:
    """AC-T23: Every threading.Lock()/RLock() in src/ has the leaf lock comment.

    Excludes src/utils/lock_order.py (internal infrastructure).
    """

    def test_all_leaf_locks_annotated(self):
        """Scan all .py files under src/ for un-annotated leaf locks."""
        import re
        from pathlib import Path

        src_dir = Path(__file__).resolve().parent.parent / "src"
        lock_pattern = re.compile(r"threading\.(Lock|RLock)\(\)")
        comment = "# leaf lock: never held while acquiring a LockLevel lock"

        missing = []
        for py_file in sorted(src_dir.rglob("*.py")):
            # Exclude the lock_order infrastructure itself
            if py_file.name == "lock_order.py":
                continue
            lines = py_file.read_text().splitlines()
            for i, line in enumerate(lines, 1):
                if lock_pattern.search(line) and comment not in line:
                    rel = py_file.relative_to(src_dir.parent)
                    missing.append(f"{rel}:{i}")

        assert not missing, (
            f"Found {len(missing)} unannotated leaf lock(s):\n" + "\n".join(missing)
        )


# ---------------------------------------------------------------------------
# Task 19: Strict mode raises RuntimeError on violation
# ---------------------------------------------------------------------------


class TestStrictModeRaises:
    """GHOSTAP_LOCK_ORDER_CHECK=strict raises RuntimeError on ordering violation."""

    def test_strict_mode_raises_on_violation(self):
        """enable_lock_order_check(strict=True) → violation raises RuntimeError."""
        enable_lock_order_check(strict=True)
        try:
            outer = ordered_lock(LockLevel.REPO_LOCK, name="inner_first")
            inner = ordered_lock(LockLevel.PROJECT_MANAGER, name="outer_second")
            with outer:
                with pytest.raises(RuntimeError, match="Lock ordering violation"):
                    inner.acquire()
                # If we got here, the acquire raised (good). Clean up outer.
        finally:
            _get_held().clear()
            disable_lock_order_check()

    def test_strict_mode_correct_order_no_raise(self):
        """Correct ordering with strict=True does not raise."""
        enable_lock_order_check(strict=True)
        try:
            outer = ordered_lock(LockLevel.PROJECT_MANAGER, name="outer")
            inner = ordered_lock(LockLevel.REPO_LOCK, name="inner")
            with outer:
                with inner:
                    pass
        finally:
            _get_held().clear()
            disable_lock_order_check()
