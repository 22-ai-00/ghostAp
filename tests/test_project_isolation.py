"""Tests for ProjectManager multi-chat project isolation.

Validates that:
- Projects are scoped to their creator's chat_id via allowed_chat_ids.
- Legacy projects (empty allowed_chat_ids) remain visible to all chats.
- set_active_project auto-adds chat_id to allowed_chat_ids.
- Query methods (get_all_projects, find_project_by_name, search_projects)
  respect chat_id filtering.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from src.project.context import ProjectContext, ProjectStatus
from src.project.manager import ProjectManager


@pytest.fixture()
def pm(tmp_path):
    """Create a ProjectManager with isolated temp storage."""
    storage = str(tmp_path / "projects.json")
    return ProjectManager(storage_path=storage)


class TestProjectIsolation:
    """Core isolation: projects are only visible to allowed chats."""

    def test_create_project_binds_chat_id(self, pm):
        ok, _, ctx = pm.create_project(None, "alpha", "/tmp/alpha_iso", chat_id="chatA")
        assert ok
        assert ctx.owner_chat_id == "chatA"
        assert ctx._chat_id_set() == {"chatA"}

    def test_create_project_without_chat_id(self, pm):
        ok, _, ctx = pm.create_project(None, "beta", "/tmp/beta_iso")
        assert ok
        assert ctx.owner_chat_id == ""
        assert ctx._chat_id_set() == set()

    def test_get_all_projects_isolation(self, pm):
        pm.create_project(None, "projA", "/tmp/projA_iso", chat_id="chatA")
        pm.create_project(None, "projB", "/tmp/projB_iso", chat_id="chatB")

        assert [p.project_name for p in pm.get_all_projects(chat_id="chatA")] == ["projA"]
        assert [p.project_name for p in pm.get_all_projects(chat_id="chatB")] == ["projB"]

    def test_get_all_projects_no_filter(self, pm):
        pm.create_project(None, "projA", "/tmp/projA_iso2", chat_id="chatA")
        pm.create_project(None, "projB", "/tmp/projB_iso2", chat_id="chatB")

        names = sorted(p.project_name for p in pm.get_all_projects())
        assert names == ["projA", "projB"]

    def test_find_project_by_name_isolation(self, pm):
        pm.create_project(None, "secret", "/tmp/secret_iso", chat_id="chatA")

        assert pm.find_project_by_name("secret", chat_id="chatA") is not None
        assert pm.find_project_by_name("secret", chat_id="chatB") is None

    def test_search_projects_isolation(self, pm):
        pm.create_project(None, "myapp", "/tmp/myapp_iso", chat_id="chatA")

        assert len(pm.search_projects("myapp", chat_id="chatA")) == 1
        assert len(pm.search_projects("myapp", chat_id="chatB")) == 0


class TestLegacyProjectVisibility:
    """Legacy projects (empty allowed_chat_ids) are visible to all chats."""

    def test_legacy_project_visible_to_all(self, pm):
        pm.create_project(None, "legacy", "/tmp/legacy_iso")
        ctx = pm.get_project_for_diagnostics("legacy")
        assert ctx._chat_id_set() == set()

        assert pm.find_project_by_name("legacy", chat_id="chatX") is not None
        assert len(pm.get_all_projects(chat_id="chatX")) == 1


class TestGetProjectForChat:
    """Tests for ProjectManager.get_project_for_chat isolation boundary."""

    def test_visible_to_owner_chat(self, pm):
        ok, _, ctx = pm.create_project(None, "alpha", "/tmp/alpha_fc", chat_id="chatA")
        assert ok
        result = pm.get_project_for_chat(ctx.project_id, "chatA")
        assert result is not None
        assert result.project_name == "alpha"

    def test_invisible_to_other_chat(self, pm):
        ok, _, ctx = pm.create_project(None, "alpha", "/tmp/alpha_fc2", chat_id="chatA")
        assert ok
        result = pm.get_project_for_chat(ctx.project_id, "chatB")
        assert result is None

    def test_legacy_visible_to_any_chat(self, pm):
        ok, _, ctx = pm.create_project(None, "legacy", "/tmp/legacy_fc")
        assert ok
        assert pm.get_project_for_chat(ctx.project_id, "chatX") is not None
        assert pm.get_project_for_chat(ctx.project_id, "chatY") is not None

    def test_none_chat_id_always_visible(self, pm):
        ok, _, ctx = pm.create_project(None, "scoped", "/tmp/scoped_fc", chat_id="chatA")
        assert ok
        # chat_id=None bypasses isolation — useful for internal/admin lookups
        result = pm.get_project_for_chat(ctx.project_id, None)
        assert result is not None

    def test_nonexistent_project(self, pm):
        result = pm.get_project_for_chat("nonexistent_id", "chatA")
        assert result is None


class TestCrossChatAccess:
    """set_active_project auto-adds chat_id to allowed_chat_ids."""

    def test_set_active_project_adds_chat_id(self, pm):
        pm.create_project(None, "shared", "/tmp/shared_iso", chat_id="chatA")
        ctx = pm.get_project_for_diagnostics("shared")
        assert "chatB" not in ctx._chat_id_set()

        pm.set_active_project("chatB", "shared")
        assert "chatB" in ctx._chat_id_set()
        assert "chatA" in ctx._chat_id_set()

    def test_after_join_both_chats_see_project(self, pm):
        pm.create_project(None, "collab", "/tmp/collab_iso", chat_id="chatA")
        pm.set_active_project("chatB", "collab")

        assert pm.find_project_by_name("collab", chat_id="chatA") is not None
        assert pm.find_project_by_name("collab", chat_id="chatB") is not None

    def test_identical_active_project_refresh_skips_snapshot_rewrite(self, pm):
        pm.create_project(None, "hot", "/tmp/hot_iso", chat_id="chatA")
        ctx = pm.get_project_for_diagnostics("hot")
        prior_chat_touch = ctx.allowed_chat_ids["chatA"]
        prior_project_touch = ctx.last_active
        pm._save_projects = MagicMock(return_value=True)

        ok, _ = pm.set_active_project("chatA", "hot")

        assert ok is True
        pm._save_projects.assert_not_called()
        assert ctx.allowed_chat_ids["chatA"] >= prior_chat_touch
        assert ctx.last_active >= prior_project_touch

    def test_identical_active_binding_repairs_missing_membership_and_persists(self, pm):
        pm.create_project(None, "repair", "/tmp/repair_iso", chat_id="chatOwner")
        ctx = pm.get_project_for_diagnostics("repair")
        pm._active_project["chatMissing"] = "repair"
        ctx.status = ProjectStatus.ACTIVE
        pm._save_projects = MagicMock(return_value=True)

        ok, _ = pm.set_active_project("chatMissing", "repair")

        assert ok is True
        assert "chatMissing" in ctx.allowed_chat_ids
        pm._save_projects.assert_called_once_with()


class TestSnapshotPersistence:
    """Isolation fields survive save/load cycle."""

    def test_roundtrip_persistence(self, tmp_path):
        storage = str(tmp_path / "projects.json")
        pm1 = ProjectManager(storage_path=storage)
        pm1.create_project(None, "persist", "/tmp/persist_iso", chat_id="chatA")
        pm1.set_active_project("chatB", "persist")

        # Reload from disk
        pm2 = ProjectManager(storage_path=storage)
        ctx = pm2.get_project_for_diagnostics("persist")
        assert ctx is not None
        assert ctx.owner_chat_id == "chatA"
        assert ctx._chat_id_set() == {"chatA", "chatB"}


class TestAddChatIdLRU:
    """add_chat_id respects max_allowed_chat_ids and evicts non-owner entries."""

    def test_add_chat_id_basic(self):
        from unittest.mock import MagicMock, patch
        ctx = ProjectContext(project_id="p1", project_name="t", root_path="/tmp/t",
                             owner_chat_id="owner", allowed_chat_ids=[("owner", 1.0)])

        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 50
        with patch("src.config.get_settings", return_value=mock_settings):
            ctx.add_chat_id("chatB")
        assert "chatB" in ctx._chat_id_set()

    def test_add_chat_id_noop_if_present(self):
        ctx = ProjectContext(project_id="p1", project_name="t", root_path="/tmp/t",
                             owner_chat_id="owner", allowed_chat_ids=[("owner", 1.0)])
        # Should not call get_settings at all if already present (move-to-end)
        ctx.add_chat_id("owner")
        assert ctx._chat_id_set() == {"owner"}

    def test_add_chat_id_evicts_when_full(self):
        from unittest.mock import MagicMock, patch
        ctx = ProjectContext(project_id="p1", project_name="t", root_path="/tmp/t",
                             owner_chat_id="owner",
                             allowed_chat_ids=[("owner", 1.0), ("c1", 2.0), ("c2", 3.0)])

        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 3  # already at limit
        with patch("src.config.get_settings", return_value=mock_settings):
            ctx.add_chat_id("c3")

        # Should have evicted one non-owner entry to make room
        assert len(ctx.allowed_chat_ids) == 3
        assert "owner" in ctx._chat_id_set()
        assert "c3" in ctx._chat_id_set()

    def test_add_chat_id_preserves_owner(self):
        from unittest.mock import MagicMock, patch
        ctx = ProjectContext(project_id="p1", project_name="t", root_path="/tmp/t",
                             owner_chat_id="owner", allowed_chat_ids=[("owner", 1.0)])

        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 2
        with patch("src.config.get_settings", return_value=mock_settings):
            ctx.add_chat_id("c1")
            ctx.add_chat_id("c2")  # should evict c1, not owner

        assert "owner" in ctx._chat_id_set()
        assert "c2" in ctx._chat_id_set()
        assert len(ctx.allowed_chat_ids) == 2




class TestLegacyProjectBackfill:
    """AC-R06: set_active_project backfills empty allowed_chat_ids for legacy projects."""

    def test_legacy_backfill_on_set_active(self, pm, tmp_path):
        # Create a legacy project (no chat_id → empty allowed_chat_ids)
        pm.create_project(None, "old_proj", str(tmp_path / "old"))
        ctx = pm.get_project_for_diagnostics("old_proj")
        assert len(ctx.allowed_chat_ids) == 0

        # set_active_project should trigger backfill
        pm.set_active_project("chatX", "old_proj")
        assert "chatX" in ctx._chat_id_set()
        assert ctx.owner_chat_id == "chatX"

    def test_non_legacy_not_backfilled(self, pm, tmp_path):
        pm.create_project(None, "new_proj", str(tmp_path / "new"), chat_id="chatA")
        ctx = pm.get_project_for_diagnostics("new_proj")
        original_owner = ctx.owner_chat_id

        pm.set_active_project("chatB", "new_proj")
        # owner_chat_id should NOT change for non-legacy projects
        assert ctx.owner_chat_id == original_owner


class TestHintEvictedVsNeverBound:
    """AC-R19: find_project_by_name_with_hint distinguishes evicted vs never-bound."""

    def test_never_bound_hint(self, pm, tmp_path):
        pm.create_project(None, "proj1", str(tmp_path / "p1"), chat_id="chatA")
        _, hint = pm.find_project_by_name_with_hint("proj1", chat_id="chatB")
        assert hint is not None
        assert "/new" in hint
        assert "已自动解绑" not in hint

    def test_evicted_hint(self, pm, tmp_path):
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch
        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 2

        pm.create_project(None, "proj2", str(tmp_path / "p2"), chat_id="chatOwner")
        with mock_patch("src.config.get_settings", return_value=mock_settings):
            pm.set_active_project("chatB", "proj2")
            # At limit=2: [chatOwner, chatB], adding chatC evicts chatB
            pm.set_active_project("chatC", "proj2")

        _, hint = pm.find_project_by_name_with_hint("proj2", chat_id="chatB")
        assert hint is not None
        assert "已自动解绑" in hint


class TestOnEvictionCallback:
    """Tests for ProjectManager.on_eviction fire-and-forget callback."""

    def test_on_eviction_called_on_lru_eviction(self, pm, tmp_path):
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 2

        callback = MagicMock()
        pm.on_eviction = callback

        pm.create_project(None, "proj_ev", str(tmp_path / "ev"), chat_id="chatOwner")
        with mock_patch("src.config.get_settings", return_value=mock_settings):
            pm.set_active_project("chatB", "proj_ev")
            # chatOwner + chatB at limit; adding chatC evicts chatOwner
            pm.set_active_project("chatC", "proj_ev")

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "chatB"       # evicted chat_id (oldest non-owner)
        assert args[1] == "proj_ev"     # project_name
        assert args[2] == "proj_ev"     # project_id

    def test_on_eviction_not_called_when_no_eviction(self, pm, tmp_path):
        from unittest.mock import MagicMock

        callback = MagicMock()
        pm.on_eviction = callback

        pm.create_project(None, "proj_noev", str(tmp_path / "noev"), chat_id="chat1")
        pm.set_active_project("chat1", "proj_noev")

        callback.assert_not_called()

    def test_on_eviction_error_does_not_propagate(self, pm, tmp_path):
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 2

        pm.on_eviction = MagicMock(side_effect=RuntimeError("boom"))

        pm.create_project(None, "proj_err", str(tmp_path / "err"), chat_id="chatOwner")
        with mock_patch("src.config.get_settings", return_value=mock_settings):
            pm.set_active_project("chatB", "proj_err")
            # This should NOT raise even though callback raises
            pm.set_active_project("chatC", "proj_err")


class TestGetActiveProjectAfterEviction:
    """AC-15: get_active_project returns None for an evicted chat_id."""

    def test_evicted_chat_returns_none(self, pm, tmp_path):
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 2

        pm.create_project(None, "proj_ev", str(tmp_path / "ev"), chat_id="chatOwner")
        with mock_patch("src.config.get_settings", return_value=mock_settings):
            pm.set_active_project("chatB", "proj_ev")
            # At limit=2: [chatOwner, chatB], adding chatC evicts chatB
            pm.set_active_project("chatC", "proj_ev")

        # chatB was evicted — get_active_project should return None
        result = pm.get_active_project("chatB")
        assert result is None

    def test_non_evicted_chat_still_visible(self, pm, tmp_path):
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 3

        pm.create_project(None, "proj_vis", str(tmp_path / "vis"), chat_id="chatOwner")
        with mock_patch("src.config.get_settings", return_value=mock_settings):
            pm.set_active_project("chatB", "proj_vis")

        # chatB not evicted — should still be visible
        result = pm.get_active_project("chatB")
        assert result is not None
        assert result.project_name == "proj_vis"


class TestProjectManagerConcurrency:
    """Thread-safety tests for ProjectManager under concurrent access."""

    def test_concurrent_set_active_project(self, pm, tmp_path):
        """Multiple threads calling set_active_project on the same project concurrently."""
        pm.create_project(None, "shared", str(tmp_path / "shared"), chat_id="owner")
        errors = []

        def activate(chat_id):
            try:
                pm.set_active_project(chat_id, "shared")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=activate, args=(f"chat_{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors: {errors}"
        ctx = pm.get_project_for_diagnostics("shared")
        assert ctx is not None

    def test_concurrent_get_all_projects(self, pm, tmp_path):
        """get_all_projects reads are safe while set_active_project writes happen."""
        for i in range(5):
            pm.create_project(None, f"proj_{i}", str(tmp_path / f"p{i}"), chat_id="owner")

        results = []
        errors = []

        def read_projects():
            try:
                for _ in range(20):
                    projects = pm.get_all_projects(chat_id="owner")
                    results.append(len(projects))
            except Exception as e:
                errors.append(e)

        def write_projects():
            try:
                for i in range(20):
                    pm.set_active_project(f"chat_w{i}", f"proj_{i % 5}")
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=read_projects) for _ in range(3)]
            + [threading.Thread(target=write_projects) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors: {errors}"
        assert all(r >= 5 for r in results), f"Expected at least 5 projects per read: {results}"

    def test_concurrent_create_and_search(self, pm, tmp_path):
        """create_project and search_projects running concurrently don't crash."""
        errors = []

        def create_loop():
            try:
                for i in range(10):
                    pm.create_project(None, f"conc_{i}", str(tmp_path / f"c{i}"), chat_id="owner")
            except Exception as e:
                errors.append(e)

        def search_loop():
            try:
                for _ in range(20):
                    pm.search_projects("conc", chat_id="owner")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_loop),
            threading.Thread(target=search_loop),
            threading.Thread(target=search_loop),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors: {errors}"


class TestSetActiveProjectRollback:
    """Rollback in set_active_project must preserve original old project status."""

    def test_rollback_preserves_idle_status(self, pm, tmp_path):
        """AC-R07: When old project was IDLE, rollback must NOT promote it to ACTIVE."""
        from src.project.context import ProjectStatus

        # Create two projects owned by different chats
        pm.create_project(None, "old_proj", str(tmp_path / "old"), chat_id="chatA")
        pm.create_project(None, "new_proj", str(tmp_path / "new"), chat_id="owner_only")

        # Activate old_proj for chatA, then deactivate it so status is IDLE
        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 50
        with patch("src.config.get_settings", return_value=mock_settings):
            pm.set_active_project("chatA", "old_proj")
        old_ctx = pm.get_project_for_diagnostics("old_proj")
        assert old_ctx.status == ProjectStatus.ACTIVE

        # Manually set old_proj to IDLE (simulating another chat switching away)
        old_ctx.status = ProjectStatus.IDLE

        # Now make new_proj reject chatA by setting limit=1 and owner_only as sole entry
        new_ctx = pm.get_project_for_diagnostics("new_proj")
        new_ctx.owner_chat_id = "owner_only"
        # Ensure capacity is full with only owner entries
        mock_settings_tight = MagicMock()
        mock_settings_tight.max_allowed_chat_ids = 1
        with patch("src.config.get_settings", return_value=mock_settings_tight):
            success, msg = pm.set_active_project("chatA", "new_proj")

        # set_active_project should fail (REJECTED)
        assert success is False
        assert "已满" in msg

        # Critical assertion: old_proj status must still be IDLE, not ACTIVE
        assert old_ctx.status == ProjectStatus.IDLE

    def test_rollback_preserves_active_status(self, pm, tmp_path):
        """When old project was ACTIVE, rollback restores it to ACTIVE."""
        from src.project.context import ProjectStatus

        pm.create_project(None, "old_proj", str(tmp_path / "old"), chat_id="chatA")
        pm.create_project(None, "new_proj", str(tmp_path / "new"), chat_id="owner_only")

        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 50
        with patch("src.config.get_settings", return_value=mock_settings):
            pm.set_active_project("chatA", "old_proj")
        old_ctx = pm.get_project_for_diagnostics("old_proj")
        assert old_ctx.status == ProjectStatus.ACTIVE

        new_ctx = pm.get_project_for_diagnostics("new_proj")
        new_ctx.owner_chat_id = "owner_only"
        mock_settings_tight = MagicMock()
        mock_settings_tight.max_allowed_chat_ids = 1
        with patch("src.config.get_settings", return_value=mock_settings_tight):
            success, _ = pm.set_active_project("chatA", "new_proj")

        assert success is False
        # old_proj was ACTIVE before, should be restored to ACTIVE after rollback
        assert old_ctx.status == ProjectStatus.ACTIVE


class TestValidateProjectPathIsolation:
    """validate_project_path must hold _lock and respect chat visibility."""

    def test_invisible_chat_rejected(self, pm, tmp_path):
        """AC-R01: Passing a chat_id that is not in allowed_chat_ids returns rejection."""
        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 50
        with patch("src.config.get_settings", return_value=mock_settings):
            pm.create_project(None, "secret", str(tmp_path / "s"), chat_id="chatOwner")
            pm.set_active_project("chatOwner", "secret")

        ok, msg = pm.validate_project_path("secret", chat_id="chatStranger")
        assert ok is False
        assert "无权" in msg

    def test_visible_chat_allowed(self, pm, tmp_path):
        """Owner chat can validate its own project."""
        mock_settings = MagicMock()
        mock_settings.max_allowed_chat_ids = 50
        with patch("src.config.get_settings", return_value=mock_settings):
            pm.create_project(None, "mine", str(tmp_path / "m"), chat_id="chatOwner")
            pm.set_active_project("chatOwner", "mine")

        ok, path = pm.validate_project_path("mine", chat_id="chatOwner")
        assert ok is True

    def test_no_chat_id_backward_compatible(self, pm, tmp_path):
        """Not passing chat_id keeps the original behaviour (no visibility check)."""
        pm.create_project(None, "pub", str(tmp_path / "p"), chat_id="chatOwner")
        ok, path = pm.validate_project_path("pub")
        assert ok is True

    def test_concurrent_validate(self, pm, tmp_path):
        """10 threads concurrently calling validate_project_path must not crash."""
        pm.create_project(None, "conc", str(tmp_path / "c"), chat_id="owner")
        errors = []

        def worker():
            try:
                for _ in range(20):
                    pm.validate_project_path("conc", chat_id="owner")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Unexpected errors: {errors}"


class TestGetProjectUncheckedThreadSafe:
    """get_project_for_diagnostics must be safe under concurrent reads and writes."""

    def test_concurrent_read_write(self, pm, tmp_path):
        """AC-R02: Concurrent get_project_for_diagnostics reads + create_project writes don't crash."""
        pm.create_project(None, "base", str(tmp_path / "b"), chat_id="owner")
        errors = []

        def read_loop():
            try:
                for _ in range(50):
                    pm.get_project_for_diagnostics("base")
                    pm.get_project_for_diagnostics("nonexistent")
            except Exception as e:
                errors.append(e)

        def write_loop():
            try:
                for i in range(10):
                    pm.create_project(None, f"w_{i}", str(tmp_path / f"w{i}"), chat_id="owner")
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=read_loop) for _ in range(5)]
            + [threading.Thread(target=write_loop)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_rlock_reentrant(self, pm, tmp_path):
        """RLock allows get_project_for_diagnostics inside already-locked context (no deadlock)."""
        pm.create_project(None, "reent", str(tmp_path / "r"), chat_id="owner")
        # Simulate a caller already holding _lock, then calling get_project_for_diagnostics
        with pm._lock:
            ctx = pm.get_project_for_diagnostics("reent")
        assert ctx is not None
        assert ctx.project_name == "reent"
