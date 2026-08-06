"""Tests for the shared model cascade (family × profile × effort) rendering
and its integration into normal-mode ACP model selection.

Covers:
* pure algorithm (split / dimensions / groups / reverse-solve default)
* build_acp_model_cascade_card renders 3 dropdowns for Traex-style lists
* default selection reverse-solved from current_model (remember last model)
* non-variant small lists fall back to the plain button card
* the cascade redraw handler repaints without entering the mode; only the
  final select commits acp_model_name.
"""

from __future__ import annotations

import json

from src.card.actions import dispatch as action_ids
from src.card.builders.system import SystemBuilder
from src.card.render import model_cascade


def _walk_selects(card: dict) -> list[dict]:
    found: list[dict] = []

    def walk(el):
        if isinstance(el, dict):
            if el.get("tag") == "select_static":
                found.append(el)
            for v in el.values():
                walk(v)
        elif isinstance(el, list):
            for v in el:
                walk(v)

    walk(card)
    return found


# ---------------------------------------------------------------------------
# Pure algorithm
# ---------------------------------------------------------------------------


def test_split_model_variant_peels_trailing_variant_tokens():
    assert model_cascade.split_model_variant("openrouter-3o/max/high") == (
        "openrouter-3o",
        ("max", "high"),
    )
    assert model_cascade.split_model_variant("openrouter-3o/max") == ("openrouter-3o", ("max",))
    # No variant suffix -> unchanged
    assert model_cascade.split_model_variant("openrouter-3o") == ("openrouter-3o", ())
    # Provider-qualified plain name must NOT be split
    assert model_cascade.split_model_variant("anthropic/claude-sonnet") == (
        "anthropic/claude-sonnet",
        (),
    )


def test_dimensions_from_tokens():
    assert model_cascade.dimensions_from_tokens(()) == ("standard", "default")
    assert model_cascade.dimensions_from_tokens(("high",)) == ("standard", "high")
    assert model_cascade.dimensions_from_tokens(("max",)) == ("max", "default")
    assert model_cascade.dimensions_from_tokens(("max", "high")) == ("max", "high")


def test_resolve_default_selection_reverse_solves_current_model():
    models = [
        {"name": "openrouter-3o"},
        {"name": "openrouter-3o/max"},
        {"name": "openrouter-3o/max/high"},
        {"name": "c_o_new_thinking/high"},
    ]
    groups = model_cascade.build_model_groups(models)
    assert model_cascade.resolve_default_selection(groups, "openrouter-3o/max/high") == (
        "openrouter-3o",
        "max",
        "high",
    )
    # Empty / unknown current model -> no default
    assert model_cascade.resolve_default_selection(groups, "") == (None, None, None)
    assert model_cascade.resolve_default_selection(groups, "does/not/exist") == (None, None, None)


# ---------------------------------------------------------------------------
# Cascade card rendering
# ---------------------------------------------------------------------------


def _traex_models() -> list[dict]:
    return [
        {"name": "openrouter-3o"},
        {"name": "openrouter-3o/max"},
        {"name": "openrouter-3o/max/high"},
        {"name": "openrouter-3o/high"},
        {"name": "c_o_new_thinking/high"},
    ]


def test_cascade_card_renders_three_level_dropdowns():
    _, card_json = SystemBuilder.build_acp_model_cascade_card(
        _traex_models(),
        "traex",
        project_id="p1",
        current_model="openrouter-3o/max/high",
        thread_root_id="r1",
    )
    card = json.loads(card_json)
    selects = _walk_selects(card)
    actions = {s["value"].get("action") for s in selects}
    assert action_ids.SELECT_ACP_MODEL_GROUP in actions
    assert action_ids.SELECT_ACP_MODEL_PROFILE in actions
    assert action_ids.SELECT_ACP_MODEL_EFFORT in actions
    # tool/project/thread carried on every dropdown value for stateless redraw
    for s in selects:
        assert s["value"]["tool_name"] == "traex"
        assert s["value"]["project_id"] == "p1"
        assert s["value"]["thread_root_id"] == "r1"


def test_cascade_default_selection_reverse_solved_from_current_model():
    _, card_json = SystemBuilder.build_acp_model_cascade_card(
        _traex_models(),
        "traex",
        project_id="p1",
        current_model="openrouter-3o/max/high",
    )
    card = json.loads(card_json)
    by_action = {s["value"]["action"]: s.get("initial_option") for s in _walk_selects(card)}
    assert by_action[action_ids.SELECT_ACP_MODEL_GROUP] == "openrouter-3o"
    assert by_action[action_ids.SELECT_ACP_MODEL_PROFILE] == "max"
    assert by_action[action_ids.SELECT_ACP_MODEL_EFFORT] == "high"


def test_cascade_confirm_button_uses_select_acp_model_action():
    _, card_json = SystemBuilder.build_acp_model_cascade_card(
        _traex_models(),
        "traex",
        project_id="p1",
        current_model="openrouter-3o/max/high",
    )
    # The confirm button commits via SELECT_ACP_MODEL with the resolved model.
    assert action_ids.SELECT_ACP_MODEL in card_json
    card = json.loads(card_json)
    confirm_models: list[str] = []

    def walk(el):
        if isinstance(el, dict):
            if el.get("tag") == "button":
                val = el.get("value", {})
                if val.get("action") == action_ids.SELECT_ACP_MODEL and val.get("model_name"):
                    confirm_models.append(val["model_name"])
            for v in el.values():
                walk(v)
        elif isinstance(el, list):
            for v in el:
                walk(v)

    walk(card)
    assert "openrouter-3o/max/high" in confirm_models


def test_cascade_falls_back_to_button_card_for_non_variant_models():
    models = [{"name": "test-o-new"}, {"name": "deepseek-v4-pro"}]
    _, card_json = SystemBuilder.build_acp_model_cascade_card(
        models, "coco", project_id="p1", current_model="deepseek-v4-pro"
    )
    card = json.loads(card_json)
    # No cascade dropdowns; plain selectable buttons instead.
    assert _walk_selects(card) == []
    assert "deepseek-v4-pro" in card_json
    assert action_ids.SELECT_ACP_MODEL in card_json


def test_cascade_pending_overrides_default_selection():
    # Explicit pending group beats current_model reverse-solve.
    _, card_json = SystemBuilder.build_acp_model_cascade_card(
        _traex_models(),
        "traex",
        project_id="p1",
        current_model="openrouter-3o/max/high",
        pending_group="c_o_new_thinking",
    )
    card = json.loads(card_json)
    by_action = {s["value"]["action"]: s.get("initial_option") for s in _walk_selects(card)}
    assert by_action[action_ids.SELECT_ACP_MODEL_GROUP] == "c_o_new_thinking"


def _codex_models() -> list[dict]:
    return [
        {
            "name": "gpt-5.5",
            "display_name": "GPT-5.5",
            "reasoning_efforts": ("low", "medium", "high", "xhigh"),
            "adapted_reasoning_effort": "high",
        },
        {
            "name": "gpt-5.6-sol",
            "display_name": "GPT-5.6-Sol",
            "reasoning_efforts": (
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
                "ultra",
            ),
            "adapted_reasoning_effort": "high",
        },
    ]


def test_codex_capabilities_render_effort_dropdown_without_profile():
    _, card_json = SystemBuilder.build_acp_model_cascade_card(
        _codex_models(),
        "codex",
        project_id="p1",
        current_model="gpt-5.6-sol/max",
    )

    card = json.loads(card_json)
    by_action = {
        select["value"]["action"]: select
        for select in _walk_selects(card)
    }

    assert action_ids.SELECT_ACP_MODEL_GROUP in by_action
    assert action_ids.SELECT_ACP_MODEL_PROFILE not in by_action
    effort_select = by_action[action_ids.SELECT_ACP_MODEL_EFFORT]
    assert effort_select["initial_option"] == "max"
    assert [option["value"] for option in effort_select["options"]] == [
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    ]


def test_codex_low_reasoning_efforts_are_ordered_before_low():
    groups = model_cascade.build_model_groups(
        [
            {
                "name": "gpt-5.4-mini",
                "reasoning_efforts": ("high", "low", "minimal", "none"),
                "adapted_reasoning_effort": "minimal",
            }
        ]
    )

    assert model_cascade.ordered_unique(
        (variant["effort"] for variant in groups[0]["variants"]),
        kind="effort",
    ) == ["none", "minimal", "low", "high"]


def test_codex_max_is_effort_and_old_bare_model_uses_adapter_default_effort():
    groups = model_cascade.build_model_groups(_codex_models())

    assert model_cascade.resolve_default_selection(
        groups,
        "gpt-5.6-sol/max",
    ) == ("gpt-5.6-sol", "standard", "max")
    assert model_cascade.resolve_default_selection(
        groups,
        "gpt-5.6-sol",
    ) == ("gpt-5.6-sol", "standard", "high")


def test_codex_pending_group_uses_that_models_default_effort():
    _, card_json = SystemBuilder.build_acp_model_cascade_card(
        _codex_models(),
        "codex",
        project_id="p1",
        current_model="gpt-5.6-sol/max",
        pending_group="gpt-5.5",
    )

    card = json.loads(card_json)
    by_action = {
        select["value"]["action"]: select
        for select in _walk_selects(card)
    }
    assert by_action[action_ids.SELECT_ACP_MODEL_GROUP]["initial_option"] == "gpt-5.5"
    assert by_action[action_ids.SELECT_ACP_MODEL_EFFORT]["initial_option"] == "high"


def test_codex_stale_current_selection_falls_back_to_adapter_default():
    models = _codex_models()
    models[1]["is_default"] = True

    _, card_json = SystemBuilder.build_acp_model_cascade_card(
        models,
        "codex",
        project_id="p1",
        current_model="removed-model/max",
    )

    card = json.loads(card_json)
    by_action = {
        select["value"]["action"]: select
        for select in _walk_selects(card)
    }
    assert by_action[action_ids.SELECT_ACP_MODEL_GROUP]["initial_option"] == "gpt-5.6-sol"
    assert by_action[action_ids.SELECT_ACP_MODEL_EFFORT]["initial_option"] == "high"


# ---------------------------------------------------------------------------
# Redraw handler: dropdown change repaints, does NOT enter mode
# ---------------------------------------------------------------------------


def _make_handler(monkeypatch):
    from unittest.mock import MagicMock

    from src.feishu.handlers.system import SystemHandler

    ctx = MagicMock()
    ctx.project_manager.get_active_project.return_value = None
    ctx.project_manager.get_project_for_chat.return_value = None
    handler = SystemHandler(ctx)
    handler.get_working_dir = MagicMock(return_value="/tmp")
    handler.update_card = MagicMock(return_value=True)
    handler.reply_card = MagicMock()
    handler.reply_error = MagicMock()
    handler._fetch_acp_models = MagicMock(return_value=_traex_models())
    monkeypatch.setattr("src.thread.get_current_thread_id", lambda: "r1", raising=False)
    return handler


def test_cascade_redraw_does_not_enter_mode(monkeypatch):
    handler = _make_handler(monkeypatch)
    entered: list = []
    # Guard: entering the mode must never happen on a dropdown change.
    handler._enter_mode_with_acp_model = lambda *a, **k: entered.append(a)

    handler.handle_acp_model_cascade_select(
        "msg1",
        "chat1",
        "traex",
        "p1",
        {
            "action": action_ids.SELECT_ACP_MODEL_GROUP,
            "tool_name": "traex",
            "project_id": "p1",
            "_option": "c_o_new_thinking",
        },
    )
    # Repainted the card (update_card), no mode entry.
    assert entered == []
    handler.update_card.assert_called_once()
    _, repainted = handler.update_card.call_args[0]
    card = json.loads(repainted)
    by_action = {s["value"]["action"]: s.get("initial_option") for s in _walk_selects(card)}
    # The changed group is now reflected as the selected group.
    assert by_action[action_ids.SELECT_ACP_MODEL_GROUP] == "c_o_new_thinking"


def test_cascade_redraw_resets_downstream_dropdowns_on_group_change(monkeypatch):
    handler = _make_handler(monkeypatch)
    handler.handle_acp_model_cascade_select(
        "msg1",
        "chat1",
        "traex",
        "p1",
        {
            "action": action_ids.SELECT_ACP_MODEL_GROUP,
            "tool_name": "traex",
            "project_id": "p1",
            # stale downstream selections from a previous group
            "model_profile": "max",
            "model_effort": "high",
            "_option": "c_o_new_thinking",
        },
    )
    _, repainted = handler.update_card.call_args[0]
    card = json.loads(repainted)
    # c_o_new_thinking only has one variant (high) → group changes cleanly
    # without carrying the stale max/high from openrouter-3o.
    selects = _walk_selects(card)
    group_sel = next(s for s in selects if s["value"]["action"] == action_ids.SELECT_ACP_MODEL_GROUP)
    assert group_sel["initial_option"] == "c_o_new_thinking"


def _explicit_traex_models() -> list[dict]:
    return [
        {
            "name": "c_o_new_thinking",
            "display_name": "Test-O-New-Thinking",
            "is_default": True,
            "selection_variants": [
                {
                    "name": f"c_o_new_thinking/{profile}/{effort}",
                    "display_name": (
                        f"Test-O-New-Thinking · {profile} · {effort}"
                    ),
                    "profile": profile,
                    "effort": effort,
                    "is_variant_default": effort == "high",
                }
                for profile in ("standard", "max")
                for effort in ("low", "medium", "high", "max")
            ],
        }
    ]


def test_explicit_traex_variants_render_three_level_dropdowns():
    _, card_json = SystemBuilder.build_acp_model_cascade_card(
        _explicit_traex_models(),
        "traex",
        project_id="p1",
        current_model="c_o_new_thinking/max/max",
    )

    card = json.loads(card_json)
    by_action = {
        select["value"]["action"]: select
        for select in _walk_selects(card)
    }
    assert by_action[action_ids.SELECT_ACP_MODEL_GROUP]["initial_option"] == (
        "c_o_new_thinking"
    )
    assert by_action[action_ids.SELECT_ACP_MODEL_PROFILE]["initial_option"] == "max"
    assert by_action[action_ids.SELECT_ACP_MODEL_EFFORT]["initial_option"] == "max"
    assert "c_o_new_thinking/max/max" in card_json


def test_button_model_card_expands_explicit_traex_variants():
    from src.acp.options import ACPModelOption, ACPModelVariantOption

    model = ACPModelOption(
        name="c_o_new_thinking",
        selection_variants=(
            ACPModelVariantOption(
                name="c_o_new_thinking/standard/high",
                profile="standard",
                effort="high",
            ),
            ACPModelVariantOption(
                name="c_o_new_thinking/max/max",
                profile="max",
                effort="max",
            ),
        ),
    )

    _, card_json = SystemBuilder.build_acp_model_select_card(
        [model],
        "traex",
        refresh_action_name=None,
    )

    assert "c_o_new_thinking/standard/high" in card_json
    assert "c_o_new_thinking/max/max" in card_json
