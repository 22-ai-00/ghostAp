"""Tests for ModeManager cross-chat project mode isolation.

Validates that the composite key ``{chat_id}:{project_id}`` ensures
different chats operating on the same project maintain independent modes.
"""

from __future__ import annotations

from src.mode.manager import InteractionMode, ModeManager


class TestModeManagerIsolation:

    def test_cross_chat_mode_isolation(self):
        mm = ModeManager()
        mm.set_mode("chat_A", InteractionMode.COCO, project_id="proj1")
        mm.set_mode("chat_B", InteractionMode.CLAUDE, project_id="proj1")

        assert mm.get_mode("chat_A", "proj1") == InteractionMode.COCO
        assert mm.get_mode("chat_B", "proj1") == InteractionMode.CLAUDE

    def test_clear_project_mode_scoped(self):
        mm = ModeManager()
        mm.set_mode("chat_A", InteractionMode.COCO, project_id="proj1")
        mm.set_mode("chat_B", InteractionMode.CLAUDE, project_id="proj1")

        # Clearing chat_A's project mode should NOT affect chat_B
        mm.clear_project_mode("chat_A", "proj1")
        assert mm.get_project_mode("chat_A", "proj1") is None
        assert mm.get_project_mode("chat_B", "proj1") == InteractionMode.CLAUDE

    def test_get_project_mode_isolated(self):
        mm = ModeManager()
        mm.set_mode("chat_A", InteractionMode.TRAEX, project_id="proj1")

        assert mm.get_project_mode("chat_A", "proj1") == InteractionMode.TRAEX
        # chat_B has no mode set for proj1
        assert mm.get_project_mode("chat_B", "proj1") is None

    def test_exit_to_smart_only_affects_own_chat(self):
        mm = ModeManager()
        mm.set_mode("chat_A", InteractionMode.COCO, project_id="proj1")
        mm.set_mode("chat_B", InteractionMode.CLAUDE, project_id="proj1")

        mm.exit_to_smart("chat_A", project_id="proj1")
        assert mm.get_mode("chat_A", "proj1") == InteractionMode.SMART
        assert mm.get_mode("chat_B", "proj1") == InteractionMode.CLAUDE


class TestModeManagerBoundary:
    """Boundary tests for ModeManager isolation edge cases."""

    def test_same_chat_different_projects(self):
        """Same chat_id with different project_ids must maintain independent modes."""
        mm = ModeManager()
        mm.set_mode("chat_A", InteractionMode.COCO, project_id="proj1")
        mm.set_mode("chat_A", InteractionMode.CLAUDE, project_id="proj2")

        assert mm.get_mode("chat_A", "proj1") == InteractionMode.COCO
        assert mm.get_mode("chat_A", "proj2") == InteractionMode.CLAUDE

    def test_no_project_id_falls_back_to_chat_mode(self):
        """get_mode without project_id returns the chat-level mode, not project mode."""
        mm = ModeManager()
        mm.set_mode("chat_A", InteractionMode.COCO, project_id="proj1")

        # Chat-level mode (no project) defaults to SMART
        assert mm.get_mode("chat_A") == InteractionMode.SMART
        # With project_id, returns the project mode
        assert mm.get_mode("chat_A", "proj1") == InteractionMode.COCO

    def test_clear_project_mode_does_not_affect_other_projects(self):
        """Clearing mode on one project must not affect other projects on same chat."""
        mm = ModeManager()
        mm.set_mode("chat_A", InteractionMode.COCO, project_id="proj1")
        mm.set_mode("chat_A", InteractionMode.CLAUDE, project_id="proj2")

        mm.clear_project_mode("chat_A", "proj1")
        assert mm.get_project_mode("chat_A", "proj1") is None
        assert mm.get_project_mode("chat_A", "proj2") == InteractionMode.CLAUDE

    def test_is_programming_mode_isolated(self):
        """is_programming_mode must check only the specified chat+project combination."""
        mm = ModeManager()
        mm.set_mode("chat_A", InteractionMode.COCO, project_id="proj1")
        mm.set_mode("chat_B", InteractionMode.SMART, project_id="proj1")

        assert mm.is_programming_mode("chat_A", "proj1") is True
        assert mm.is_programming_mode("chat_B", "proj1") is False


class TestClearModesForChat:
    """Tests for ModeManager.clear_modes_for_chat (AC-R01)."""

    def test_clears_single_project(self):
        mm = ModeManager()
        mm.set_mode("chat_X", InteractionMode.COCO, project_id="p1")
        removed = mm.clear_modes_for_chat("chat_X")
        assert removed == 1
        assert mm.get_project_mode("chat_X", "p1") is None

    def test_clears_multiple_projects(self):
        mm = ModeManager()
        mm.set_mode("chat_X", InteractionMode.COCO, project_id="p1")
        mm.set_mode("chat_X", InteractionMode.CLAUDE, project_id="p2")
        mm.set_mode("chat_X", InteractionMode.AIDEN, project_id="p3")
        removed = mm.clear_modes_for_chat("chat_X")
        assert removed == 3
        assert mm.get_project_mode("chat_X", "p1") is None
        assert mm.get_project_mode("chat_X", "p2") is None
        assert mm.get_project_mode("chat_X", "p3") is None

    def test_does_not_affect_other_chats(self):
        mm = ModeManager()
        mm.set_mode("chat_X", InteractionMode.COCO, project_id="p1")
        mm.set_mode("chat_Y", InteractionMode.CLAUDE, project_id="p1")
        mm.clear_modes_for_chat("chat_X")
        assert mm.get_project_mode("chat_Y", "p1") == InteractionMode.CLAUDE

    def test_returns_zero_when_no_entries(self):
        mm = ModeManager()
        assert mm.clear_modes_for_chat("nonexistent") == 0

    def test_bulk_eviction_bounded_size(self):
        """After evicting 1000 chats, _project_modes size stays bounded."""
        mm = ModeManager()
        # Populate 1000 chats × 2 projects each
        for i in range(1000):
            mm.set_mode(f"chat_{i}", InteractionMode.COCO, project_id="p1")
            mm.set_mode(f"chat_{i}", InteractionMode.CLAUDE, project_id="p2")
        assert len(mm._project_modes) == 2000

        # Evict all 1000 chats
        for i in range(1000):
            mm.clear_modes_for_chat(f"chat_{i}")
        assert len(mm._project_modes) == 0
