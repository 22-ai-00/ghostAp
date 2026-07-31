"""Product contract for execution-lane visibility and completion labels."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.card.builders.system import SystemBuilder
from src.feishu import product_catalog
from src.feishu.dispatcher import MessageDispatcher
from src.feishu.main_slash_commands import MAIN_AGENT_COMMANDS
from src.feishu.product_catalog import (
    CompletionLabel,
    ExecutionLane,
    RuntimeHealth,
    get_execution_action,
    get_owner_actions,
    is_explicit_protected_command,
)
from src.feishu.route_decision import CommandRouter
from src.feishu.slash_command_parser import SlashCommandParser
from src.feishu.ws_client import FeishuWSClient
from src.slock_engine.card_templates import build_welcome_card

# Captured from the pre-catalog main Slash registration.  This fixture protects
# current Owner-visible commands from accidental hiding during catalog changes.
MAIN_SLASH_COMMAND_BASELINE = frozenset({
    "/help", "/menu", "/tools", "/tools_status", "/coco", "/claude",
    "/aiden", "/codex", "/gemini", "/traex", "/ttadk", "/tui2acp",
    "/acp", "/model", "/exit", "/btw", "/coco_status", "/coco_info",
    "/claude_info", "/aiden_info", "/codex_info", "/gemini_info",
    "/traex_info", "/ttadk_info", "/tui2acp_info", "/ttadk_refresh",
    "/projects", "/new", "/new-chat", "/switch", "/close", "/status",
    "/tasks", "/diff", "/trace", "/lock", "/unlock", "/setadmin",
    "/deep", "/deep_status", "/deep_update", "/stop_deep", "/spec",
    "/spec_status", "/spec_history", "/spec_metrics", "/spec_config",
    "/spec_export", "/spec_save", "/spec_pause", "/spec_resume",
    "/spec_guide", "/spec_recover", "/stop_spec", "/worktree", "/wf",
    "/wf_status", "/wf_help", "/stop_wf", "/wf_save", "/wf_list",
    "/wf_delete", "/wf_history", "/slock", "/slocks", "/new-team",
    "/new-role", "/hire", "/fire", "/employees", "/history",
    "/employee-memory", "/role", "/task", "/team", "/council",
    "/discuss", "/memory", "/plan",
})

DIRECT_COMPATIBILITY_ALIASES = (
    ("/enter_coco", "/coco"),
    ("/enter_claude", "/claude"),
    ("/enter_aiden", "/aiden"),
    ("/enter_codex", "/codex"),
    ("/enter_gemini", "/gemini"),
    ("/enter_traex", "/traex"),
    ("/enter_ttadk", "/ttadk"),
    ("/enter_tui2acp", "/tui2acp"),
)


def test_direct_deep_spec_are_mature_by_default() -> None:
    """Protected existing lanes retain the mature product expectation."""
    for lane in (ExecutionLane.DIRECT, ExecutionLane.DEEP, ExecutionLane.SPEC):
        assert get_execution_action(lane).completion is CompletionLabel.MATURE


def test_implemented_developing_lane_is_visible_to_owner() -> None:
    """A developing contract describes support status without hiding the entry."""
    action = get_execution_action(ExecutionLane.WORKFLOW)

    assert action.completion is CompletionLabel.DEVELOPING
    assert action.runtime_health is RuntimeHealth.AVAILABLE
    assert action.blocking_reason
    assert action in get_owner_actions()


def test_completion_label_does_not_gate_owner_access() -> None:
    """Every implemented lane remains directly available to the single Owner."""
    owner_actions = get_owner_actions()

    assert {action.lane for action in owner_actions} == set(ExecutionLane)
    assert all(action.owner_accessible for action in owner_actions)
    assert all(action.completion is not CompletionLabel.NOT_IMPLEMENTED for action in owner_actions)


def test_owner_menu_renders_every_execution_lane_with_its_completion() -> None:
    """The live menu exposes the catalog instead of hiding developing lanes."""
    _, card_json = SystemBuilder.build_command_menu_card()
    rendered = json.loads(card_json)
    content = json.dumps(rendered, ensure_ascii=False)

    for action in get_owner_actions():
        assert action.command in content
        assert action.completion.value in content


def test_developing_lane_never_auto_activates_over_explicit_command() -> None:
    """Automatic SMART/Slock paths yield to explicit protected commands."""
    assert is_explicit_protected_command("/codex")
    assert is_explicit_protected_command("/deep deliver the migration")
    assert is_explicit_protected_command("/spec make the contract explicit")
    assert not is_explicit_protected_command("implement the migration")


@pytest.mark.parametrize(
    "command",
    tuple(canonical for _, canonical in DIRECT_COMPATIBILITY_ALIASES)
    + tuple(alias for alias, _ in DIRECT_COMPATIBILITY_ALIASES),
)
def test_slock_detection_cannot_capture_an_explicit_direct_command(command: str) -> None:
    """An explicit Direct command stays on the system path before Slock routing."""
    client = MagicMock()
    client.settings.slock_passive_mode = True
    client._get_effective_mode.return_value = ("smart", False)
    client._is_topic_engine_context.return_value = False
    client._is_deep_command.return_value = False
    client._is_spec_command.return_value = False
    client._is_workflow_command.return_value = False
    client._is_slock_command.return_value = True
    client._is_interceptable_command_match.return_value = True
    dispatcher = MessageDispatcher(client)
    command_match = SlashCommandParser.parse(command)

    dispatcher.process_with_intent(
        "om_direct",
        "oc_owner",
        command,
        command_match=command_match,
    )

    client._handle_intercepted_command.assert_called_once_with(
        "om_direct",
        "oc_owner",
        command,
        None,
        command_match=command_match,
    )
    client._handle_slock_command.assert_not_called()
    client._is_slock_command.assert_not_called()


def test_catalog_exposes_typed_owner_and_compatibility_projections() -> None:
    """Main Slash registration is projected from one typed catalog."""
    assert hasattr(product_catalog, "ProductRole")
    assert hasattr(product_catalog, "ProductScope")
    assert hasattr(product_catalog, "CompatibilityBehavior")
    assert hasattr(product_catalog, "ResolvedProductCommand")
    assert hasattr(product_catalog, "get_public_actions")
    assert hasattr(product_catalog, "resolve_command")
    assert {
        action.command for action in product_catalog.get_public_actions()
    } == MAIN_SLASH_COMMAND_BASELINE
    assert tuple(
        (command.name, command.description, command.usage_hint)
        for command in MAIN_AGENT_COMMANDS
    ) == tuple(
        (action.command, action.description, action.usage)
        for action in product_catalog.get_public_actions()
    )


def test_every_retired_token_resolves_to_the_fail_closed_action() -> None:
    """Compatibility projections include canonical retired commands and aliases."""
    expected = frozenset({
        "/goal",
        "/goals",
        "/run",
        "/runs",
        "/approve",
        "/approvals",
        "/decisions",
    })

    assert product_catalog.COMPATIBILITY_TOKENS == expected
    for token in product_catalog.COMPATIBILITY_TOKENS:
        resolved = product_catalog.resolve_command(token, "refactor auth")
        assert resolved is not None
        assert (
            resolved.action.compatibility
            is product_catalog.CompatibilityBehavior.RETIRED_MESSAGE
        )


def test_lane_projection_derives_command_metadata_from_public_actions() -> None:
    """Lane summaries cannot independently drift from their canonical command."""
    for lane in ExecutionLane:
        lane_action = get_execution_action(lane)
        canonical = product_catalog.resolve_command(lane_action.command)

        assert canonical is not None
        assert lane_action.description == canonical.action.description


def test_execution_preview_matches_live_menu_lane_markers() -> None:
    """The static preview keeps every live lane's visible core metadata."""
    preview = (
        Path(__file__).resolve().parents[1]
        / "ux"
        / "execution-lanes-menu-preview.html"
    ).read_text(encoding="utf-8")
    _, card_json = SystemBuilder.build_command_menu_card()
    production = json.dumps(json.loads(card_json), ensure_ascii=False)

    for action in get_owner_actions():
        command_display = " · ".join((action.command, *action.aliases))
        for marker in (
            action.label,
            command_display,
            action.completion.value,
            action.runtime_health.value,
        ):
            assert marker in preview
            assert marker in production
        if action.blocking_reason:
            assert action.blocking_reason in preview
            assert action.blocking_reason in production


@pytest.mark.parametrize("alias, canonical", DIRECT_COMPATIBILITY_ALIASES)
def test_direct_compatibility_aliases_share_catalog_parser_and_mode_contract(
    alias: str,
    canonical: str,
) -> None:
    """Every existing Direct alias remains a passthrough programming entry."""
    resolved = product_catalog.resolve_command(alias)
    match = SlashCommandParser.parse(alias)

    assert resolved is not None
    assert resolved.action.command == canonical
    assert match is not None
    assert match.command == alias
    assert CommandRouter.is_programming_entry(alias)
    assert FeishuWSClient._is_programming_entry_command(alias)


def test_retired_goal_command_is_not_advertised() -> None:
    """The retired Manager entry is absent from public actions and welcome copy."""
    advertised_commands = {
        command
        for action in get_owner_actions()
        for command in (action.command, *action.aliases)
    }

    assert "/goal" not in advertised_commands
    assert "/goal" not in json.dumps(build_welcome_card(team_name="Alpha"), ensure_ascii=False)
