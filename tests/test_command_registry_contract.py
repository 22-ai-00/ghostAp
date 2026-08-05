"""Contract tests for command routing registry snapshot.

These tests freeze the set of registered commands so that future simplification
of src/feishu/handlers/ and src/mode/ cannot accidentally remove a command path.

Protected regression scenarios:
- Exact command set in SystemHandler._exact_handlers must not shrink
- Prefix command set in SystemHandler._prefix_handlers must not shrink
- Deep engine commands (/deep, /deep_status, /stop_deep) must remain routable
- Spec engine commands (/spec, /spec_status, /stop_spec, ...) must remain routable
- Exit commands must cover all tool variants (coco/claude/aiden/codex/gemini/traex/ttadk)
- Removed Worktree commands must not hide the surviving engine routes
"""

from __future__ import annotations

from src.feishu.handlers.system import SystemHandler
from src.feishu.slash_command_parser import SlashCommandParser

# ============================================================
# Frozen command snapshots — update ONLY when intentionally
# adding/removing commands (never as a side-effect of refactoring)
# ============================================================

EXPECTED_EXACT_COMMANDS = frozenset({
    "/help",
    "/帮助",
    "/coco",
    "/enter_coco",
    "/claude",
    "/enter_claude",
    "/aiden",
    "/enter_aiden",
    "/codex",
    "/enter_codex",
    "/gemini",
    "/enter_gemini",
    "/traex",
    "/enter_traex",
    "/exit",
    "/quit",
    "/end_coco",
    "/exit_coco",
    "/end_claude",
    "/exit_claude",
    "/end_aiden",
    "/exit_aiden",
    "/end_codex",
    "/exit_codex",
    "/end_gemini",
    "/exit_gemini",
    "/end_traex",
    "/exit_traex",
    "/end_ttadk",
    "/exit_ttadk",
    "/coco_status",
    "/coco_info",
    "/claude_info",
    "/aiden_info",
    "/codex_info",
    "/gemini_info",
    "/traex_info",
    "/projects",
    "/project",
    "/switch",
    "/new-chat",
    "/ttadk",
    "/enter_ttadk",
    "/acp",
    "/ttadk_info",
    "/ttadk_refresh",
    "/menu",
    "/tools",
    "/tools_status",
    "/model",
    "/lock",
    "/unlock",
    "/setadmin",
})

EXPECTED_PREFIX_COMMANDS = frozenset({
    "/status",
    "/tasks",
    "/diff",
    "/trace",
    "/model",
})

EXPECTED_DEEP_COMMANDS = frozenset({
    "/deep",
    "/deep_status",
    "/deep_update",
    "/stop_deep",
})

EXPECTED_SPEC_COMMANDS = frozenset({
    "/spec",
    "/spec_recover",
    "/spec_status",
    "/spec_history",
    "/spec_metrics",
    "/spec_config",
    "/spec_export",
    "/spec_save",
    "/stop_spec",
    "/spec_pause",
    "/spec_resume",
    "/spec_guide",
})

EXPECTED_EXIT_COMMANDS = frozenset({
    "/exit",
    "/quit",
    "/end_coco",
    "/exit_coco",
    "/end_claude",
    "/exit_claude",
    "/end_aiden",
    "/exit_aiden",
    "/end_codex",
    "/exit_codex",
    "/end_gemini",
    "/exit_gemini",
    "/end_traex",
    "/exit_traex",
    "/end_ttadk",
    "/exit_ttadk",
})

EXPECTED_INTERCEPTABLE_EXACT = frozenset({
    "/help", "/帮助",
    "/coco", "/enter_coco",
    "/claude", "/enter_claude",
    "/aiden", "/enter_aiden",
    "/codex", "/enter_codex",
    "/gemini", "/enter_gemini",
    "/traex", "/enter_traex",
    "/enter_ttadk",
    "/exit", "/quit",
    "/end_coco", "/exit_coco",
    "/end_claude", "/exit_claude",
    "/end_aiden", "/exit_aiden",
    "/end_codex", "/exit_codex",
    "/end_gemini", "/exit_gemini",
    "/end_traex", "/exit_traex",
    "/end_ttadk", "/exit_ttadk",
    "/coco_status",
    "/coco_info", "/claude_info", "/aiden_info", "/codex_info", "/gemini_info", "/traex_info",
    "/ttadk_info",
    "/projects", "/status", "/project", "/switch", "/new-chat",
    "/tasks", "/diff", "/trace",
    "/ttadk", "/acp",
    "/ttadk_refresh",
    "/menu", "/tools", "/tools_status",
    "/model", "/lock", "/unlock", "/setadmin", "/btw",
})

EXPECTED_INTERCEPTABLE_PREFIX = frozenset({
    "/switch", "/new", "/new-chat", "/close",
    "/tasks", "/diff", "/trace", "/status", "/model", "/btw", "/setadmin",
})


class TestExactCommandRegistry:
    """Verify exact command handlers are not accidentally removed."""

    def test_all_expected_exact_commands_are_interceptable(self):
        """Every expected exact command must be interceptable without args."""
        for cmd in EXPECTED_EXACT_COMMANDS | EXPECTED_INTERCEPTABLE_EXACT:
            match = SlashCommandParser.parse(cmd)
            assert SystemHandler.is_interceptable_command_match(match), (
                f"Command '{cmd}' should be interceptable but is not"
            )


class TestPrefixCommandRegistry:
    """Verify prefix command handlers are not accidentally removed."""

    def test_prefix_commands_with_args_are_interceptable(self):
        """Prefix commands with args must be interceptable."""
        for cmd in sorted(EXPECTED_INTERCEPTABLE_PREFIX):
            match = SlashCommandParser.parse(f"{cmd} some_arg")
            assert SystemHandler.is_interceptable_command_match(match), (
                f"Prefix command '{cmd} some_arg' should be interceptable"
            )


class TestDeepCommandRouting:
    """Verify deep engine command routing is preserved."""

    def test_deep_commands_recognized(self):
        """Deep commands must be recognized by is_deep_command."""
        commands = (
            "/deep hello",
            "/deep_status",
            "/stop_deep",
            "/deep_update some context",
        )
        for cmd in commands:
            assert SystemHandler.is_deep_command(cmd), (
                f"'{cmd}' should be recognized as deep command"
            )

    def test_non_deep_not_recognized(self):
        """Non-deep commands must not be falsely recognized."""
        assert not SystemHandler.is_deep_command("/spec hello")
        assert not SystemHandler.is_deep_command("/help")


class TestSpecCommandRouting:
    """Verify spec engine command routing is preserved."""

    def test_spec_commands_recognized(self):
        """All spec commands must be recognized by is_spec_command."""
        for cmd in sorted(EXPECTED_SPEC_COMMANDS):
            assert SystemHandler.is_spec_command(cmd), (
                f"'{cmd}' should be recognized as spec command"
            )

    def test_spec_with_args_recognized(self):
        """Spec commands with arguments must still be recognized."""
        assert SystemHandler.is_spec_command("/spec implement feature X")
        assert SystemHandler.is_spec_command("/spec_guide focus on tests")

    def test_non_spec_not_recognized(self):
        """Non-spec commands must not be falsely recognized."""
        assert not SystemHandler.is_spec_command("/deep hello")
        assert not SystemHandler.is_spec_command("/help")


class TestExitCommandRouting:
    """Verify exit command routing covers all tool variants."""

    def test_exit_commands_recognized(self):
        """All exit commands must be recognized by is_exit_command."""
        for cmd in sorted(EXPECTED_EXIT_COMMANDS):
            assert SystemHandler.is_exit_command(cmd), (
                f"'{cmd}' should be recognized as exit command"
            )

    def test_non_exit_not_recognized(self):
        """Non-exit commands must not trigger exit."""
        assert not SystemHandler.is_exit_command("/help")
        assert not SystemHandler.is_exit_command("/coco")
        assert not SystemHandler.is_exit_command("/deep hello")


def test_removed_worktree_commands_do_not_hide_active_engines():
    from src.feishu.product_catalog import ExecutionLane, get_public_actions, resolve_command

    assert "/worktree" not in {action.command for action in get_public_actions()}
    assert {lane.value for lane in ExecutionLane}.isdisjoint({"worktree"})
    for token in ("/worktree", "/wt"):
        assert resolve_command(token, "legacy goal") is None
    for token in ("/deep", "/spec", "/wf"):
        resolved = resolve_command(token, "keep working")
        assert resolved is not None
        assert resolved.action.command == token
