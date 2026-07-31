"""Tests for ``SelectionFlowController`` (orchestrator + review selection).

The controller is a pure-Python state machine; tests here cover:

* Initialization and step switching.
* ``toggle_tool_expand``: expands/collapses the inline model panel.
* ``add_or_update_selection`` / ``remove_selection`` / ``clear_selections``.
* ``validate_non_empty``: orchestrator must be non-empty; review is non-empty
  when ``review_auto_mode`` is off.
* ``finish_step``: returns next step and captures current step snapshot.
* ``build_orchestrator_combined_card`` / ``build_review_combined_card``:
  produced cards have the expected shape and button actions.
* ``review_auto_mode`` toggling and the downstream effect on validation.
* ``snapshot`` / ``restore`` preserve selections and panel state.
"""

from __future__ import annotations

import pytest

from src.workflow_engine.selection_flow import (
    SelectionFlowController,
    SelectionItem,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic_controller(step: int = 1) -> SelectionFlowController:
    return SelectionFlowController(step=step)


def _find_actions(card: dict, action: str) -> list[dict]:
    """Walk a card dict and collect button values matching ``action``."""
    matches: list[dict] = []

    def _walk(obj):
        if isinstance(obj, dict):
            value = obj.get("value")
            if isinstance(value, dict) and value.get("action") == action:
                matches.append(value)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(card)
    return matches


def _find_tags(card: dict, tag: str) -> list[dict]:
    """Walk a card dict and collect nodes matching ``tag``."""
    matches: list[dict] = []

    def _walk(obj):
        if isinstance(obj, dict):
            if obj.get("tag") == tag:
                matches.append(obj)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(card)
    return matches


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestControllerInitialization:
    def test_init_default_step_is_1(self):
        ctrl = SelectionFlowController()
        assert ctrl.step == 1

    def test_init_step_must_be_1_or_2_or_3(self):
        for bad in [0, 4, "1", None]:
            with pytest.raises((ValueError, TypeError)):
                SelectionFlowController(step=bad)

    def test_init_pending_tool_name_stored(self):
        ctrl = SelectionFlowController(step=1, pending_tool_name="coco")
        assert ctrl.pending_tool_name == "coco"

    def test_init_empty_selections(self):
        ctrl = _basic_controller()
        assert ctrl.orchestrator_selections == {}
        assert ctrl.review_selections == {}
        assert ctrl.review_auto_mode is False


# ---------------------------------------------------------------------------
# Step navigation
# ---------------------------------------------------------------------------


class TestStepNavigation:
    def test_set_step_valid(self):
        ctrl = _basic_controller()
        ctrl.set_step(2)
        assert ctrl.step == 2
        ctrl.set_step(1)
        assert ctrl.step == 1
        ctrl.set_step(3)
        assert ctrl.step == 3

    def test_review_step_adds_two_distinct_traex_models_and_finishes(self):
        ctrl = SelectionFlowController(step=2)
        available_tools = [{"tool_name": "traex", "display_name": "Traex"}]
        available_models = [
            {
                "name": "DeepSeek-V4-Flash/standard/high",
                "display_name": "DeepSeek-V4-Flash/standard/high",
            },
            {
                "name": "DeepSeek-V4-Flash/standard/low",
                "display_name": "DeepSeek-V4-Flash/standard/low",
            },
        ]
        ctrl.select_tool("traex", is_review=True)
        ctrl.set_model_group("traex", "DeepSeek-V4-Flash", is_review=True)
        ctrl.set_model_effort("traex", "high", is_review=True)

        first_card = ctrl.build_review_combined_card(
            available_tools=available_tools,
            available_models=available_models,
        )
        first_model = next(
            action
            for action in _find_actions(first_card, "workflow_review_select_model")
            if action.get("model_name") == "DeepSeek-V4-Flash/standard/high"
        )
        ctrl.set_step(2)
        ctrl.add_or_update_selection(first_model, is_review=True, keep_panel_open=True)

        assert ctrl.pending_tool_name == "traex"
        assert ctrl.pending_model_group == "DeepSeek-V4-Flash"
        ctrl.set_model_effort("traex", "low", is_review=True)
        second_card = ctrl.build_review_combined_card(
            available_tools=available_tools,
            available_models=available_models,
        )
        second_model = next(
            action
            for action in _find_actions(second_card, "workflow_review_select_model")
            if action.get("model_name") == "DeepSeek-V4-Flash/standard/low"
        )
        ctrl.set_step(2)
        ctrl.add_or_update_selection(second_model, is_review=True, keep_panel_open=True)

        assert {
            selection["model_name"] for selection in ctrl.review_selections.values()
        } == {
            "DeepSeek-V4-Flash/standard/high",
            "DeepSeek-V4-Flash/standard/low",
        }
        ok, error = ctrl.validate_non_empty(is_review=True)
        assert ok is True
        assert error == ""
        finish_card = ctrl.build_review_combined_card(
            available_tools=available_tools,
            available_models=available_models,
        )
        assert _find_actions(finish_card, "workflow_review_finish")
        next_step, snapshot = ctrl.finish_step()
        assert next_step == 3
        assert len(snapshot) == 2

    def test_set_step_invalid_raises(self):
        ctrl = _basic_controller()
        with pytest.raises(ValueError):
            ctrl.set_step(4)

    def test_finish_step_from_1_advances_to_2(self):
        ctrl = _basic_controller(step=1)
        ctrl.add_or_update_selection(
            {"tool_name": "coco", "model_name": "gpt-4"},
            is_review=False,
        )
        next_step, snapshot = ctrl.finish_step()
        assert next_step == 2
        # snapshot should capture the orchestrator selection
        assert isinstance(snapshot, dict)
        assert len(snapshot) == 1
        assert next(iter(snapshot.values()))["tool_name"] == "coco"

    def test_finish_step_from_2_advances_to_3(self):
        ctrl = _basic_controller(step=2)
        ctrl.add_or_update_selection(
            {"tool_name": "claude", "model_name": "opus"},
            is_review=True,
        )
        next_step, snapshot = ctrl.finish_step()
        assert next_step == 3  # Now advances to step 3 instead of looping back
        assert len(snapshot) == 1
        assert next(iter(snapshot.values()))["tool_name"] == "claude"

    def test_is_complete_without_orchestrator_is_false(self):
        ctrl = _basic_controller()
        assert ctrl.is_complete() is False

    def test_is_complete_with_orchestrator_only_is_false(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        assert ctrl.is_complete() is False

    def test_is_complete_with_review(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        ctrl.add_or_update_selection({"tool_name": "claude"}, is_review=True)
        assert ctrl.is_complete() is True

    def test_is_complete_with_auto_mode(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        ctrl.set_review_auto_mode(True)
        assert ctrl.is_complete() is True


# ---------------------------------------------------------------------------
# Toggle tool expand
# ---------------------------------------------------------------------------


class TestToggleToolExpand:
    def test_first_toggle_expands(self):
        ctrl = _basic_controller()
        ctrl.toggle_tool_expand("coco", is_review=False)
        assert ctrl.pending_tool_name == "coco"

    def test_same_tool_again_collapses(self):
        ctrl = _basic_controller()
        ctrl.toggle_tool_expand("coco", is_review=False)
        ctrl.toggle_tool_expand("coco", is_review=False)
        assert ctrl.pending_tool_name is None

    def test_different_tool_switches(self):
        ctrl = _basic_controller()
        ctrl.toggle_tool_expand("coco", is_review=False)
        ctrl.toggle_tool_expand("claude", is_review=False)
        assert ctrl.pending_tool_name == "claude"

    def test_empty_tool_name_clears(self):
        ctrl = _basic_controller()
        ctrl.toggle_tool_expand("coco", is_review=False)
        ctrl.toggle_tool_expand("", is_review=False)
        assert ctrl.pending_tool_name is None

    def test_is_review_flag_does_not_affect_panel_state_storage(self):
        """The pending_tool_name is just a scalar on the controller - there is
        only one panel at a time because the handler drives one step."""
        ctrl = _basic_controller()
        ctrl.toggle_tool_expand("coco", is_review=True)
        assert ctrl.pending_tool_name == "coco"


# ---------------------------------------------------------------------------
# Selection mutation
# ---------------------------------------------------------------------------


class TestSelectionMutation:
    def test_add_or_update_creates_entry(self):
        ctrl = _basic_controller()
        key = ctrl.add_or_update_selection(
            {"tool_name": "coco", "display_name": "Coco", "model_name": "gpt-4"},
            is_review=False,
        )
        assert key in ctrl.orchestrator_selections
        entry = ctrl.orchestrator_selections[key]
        assert entry["tool_name"] == "coco"
        assert entry["model_name"] == "gpt-4"
        assert entry["selection_key"] == key

    def test_add_or_update_clears_pending_tool(self):
        ctrl = _basic_controller()
        ctrl.toggle_tool_expand("coco", is_review=False)
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        assert ctrl.pending_tool_name is None

    def test_add_or_update_without_selection_key_generates_one(self):
        ctrl = _basic_controller()
        key1 = ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        key2 = ctrl.add_or_update_selection({"tool_name": "claude"}, is_review=False)
        assert key1 and key2 and key1 != key2

    def test_add_or_update_with_existing_selection_key_updates(self):
        ctrl = _basic_controller()
        payload = {"selection_key": "abc", "tool_name": "coco", "model_name": "gpt-4"}
        ctrl.add_or_update_selection(payload, is_review=False)
        payload2 = {"selection_key": "abc", "tool_name": "coco", "model_name": "gpt-5"}
        ctrl.add_or_update_selection(payload2, is_review=False)
        assert len(ctrl.orchestrator_selections) == 1
        assert ctrl.orchestrator_selections["abc"]["model_name"] == "gpt-5"

    def test_remove_selection(self):
        ctrl = _basic_controller()
        key = ctrl.add_or_update_selection({"tool_name": "claude"}, is_review=True)
        ctrl.remove_selection(key, is_review=True)
        assert key not in ctrl.review_selections

    def test_remove_selection_missing_is_noop(self):
        ctrl = _basic_controller()
        ctrl.remove_selection("nope", is_review=False)  # must not raise

    def test_clear_selections(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "a"}, is_review=False)
        ctrl.add_or_update_selection({"tool_name": "b"}, is_review=False)
        ctrl.clear_selections(is_review=False)
        assert ctrl.orchestrator_selections == {}

    def test_review_and_orchestrator_are_isolated(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        ctrl.add_or_update_selection({"tool_name": "claude"}, is_review=True)
        assert "coco" in [v["tool_name"] for v in ctrl.orchestrator_selections.values()]
        assert "claude" in [v["tool_name"] for v in ctrl.review_selections.values()]
        ctrl.clear_selections(is_review=False)
        assert ctrl.orchestrator_selections == {}
        assert ctrl.review_selections, "clearing orchestrator must not touch review"


# ---------------------------------------------------------------------------
# Review auto-mode
# ---------------------------------------------------------------------------


class TestReviewAutoMode:
    def test_set_auto_mode_true(self):
        ctrl = _basic_controller()
        ctrl.set_review_auto_mode(True)
        assert ctrl.review_auto_mode is True

    def test_set_auto_mode_clears_review_selections(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "claude"}, is_review=True)
        ctrl.set_review_auto_mode(True)
        assert ctrl.review_selections == {}

    def test_set_auto_mode_closes_explicit_review_picker(self):
        ctrl = _basic_controller(step=2)
        ctrl.select_tool("traex", is_review=True)
        ctrl.set_model_group("traex", "DeepSeek-V4-Flash", is_review=True)
        ctrl.set_model_profile("traex", "standard", is_review=True)
        ctrl.set_model_effort("traex", "high", is_review=True)
        ctrl.model_page = 2

        ctrl.set_review_auto_mode(True)

        assert ctrl.review_auto_mode is True
        assert ctrl.pending_tool_name is None
        assert ctrl.pending_model_group is None
        assert ctrl.pending_model_profile is None
        assert ctrl.pending_model_effort is None
        assert ctrl.model_page == 0
        card = ctrl.build_review_combined_card(
            available_tools=[{"tool_name": "traex", "display_name": "Traex"}],
            available_models=[
                {
                    "name": "DeepSeek-V4-Flash/standard/high",
                    "display_name": "DeepSeek-V4-Flash/standard/high",
                }
            ],
        )
        assert not _find_actions(card, "workflow_review_select_model")

    def test_explicit_review_selection_disables_auto_mode(self):
        ctrl = _basic_controller(step=2)
        ctrl.set_review_auto_mode(True)

        ctrl.add_or_update_selection(
            {
                "tool_name": "traex",
                "display_name": "Traex",
                "model_name": "DeepSeek-V4-Flash/standard/high",
            },
            is_review=True,
            keep_panel_open=True,
        )

        assert ctrl.review_auto_mode is False
        assert len(ctrl.review_selections) == 1

    @pytest.mark.parametrize("method_name", ["select_tool", "toggle_tool_expand"])
    def test_opening_review_tool_disables_auto_mode(self, method_name):
        ctrl = _basic_controller(step=2)
        ctrl.set_review_auto_mode(True)

        getattr(ctrl, method_name)("traex", is_review=True)

        assert ctrl.review_auto_mode is False
        assert ctrl.pending_tool_name == "traex"

    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("set_model_page", ("traex", 1)),
            ("set_model_group", ("traex", "DeepSeek-V4-Flash")),
            ("set_model_profile", ("traex", "standard")),
            ("set_model_effort", ("traex", "high")),
        ],
    )
    def test_changing_review_picker_disables_auto_mode(self, method_name, args):
        ctrl = _basic_controller(step=2)
        ctrl.set_review_auto_mode(True)

        getattr(ctrl, method_name)(*args, is_review=True)

        assert ctrl.review_auto_mode is False
        assert ctrl.pending_tool_name == "traex"

    def test_set_auto_mode_false_keeps_state(self):
        ctrl = _basic_controller()
        ctrl.set_review_auto_mode(True)
        ctrl.set_review_auto_mode(False)
        assert ctrl.review_auto_mode is False

    def test_auto_mode_allows_empty_review_validation(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        ctrl.set_review_auto_mode(True)
        ok, msg = ctrl.validate_non_empty(is_review=True)
        assert ok is True
        assert msg == ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateNonEmpty:
    def test_orchestrator_empty_rejected(self):
        ctrl = _basic_controller()
        ok, msg = ctrl.validate_non_empty(is_review=False)
        assert ok is False
        assert msg

    def test_orchestrator_non_empty_ok(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        ok, msg = ctrl.validate_non_empty(is_review=False)
        assert ok is True
        assert msg == ""

    def test_review_empty_rejected_when_no_auto_mode(self):
        ctrl = _basic_controller()
        ok, msg = ctrl.validate_non_empty(is_review=True)
        assert ok is False
        assert msg

    def test_review_non_empty_ok(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "claude"}, is_review=True)
        ok, msg = ctrl.validate_non_empty(is_review=True)
        assert ok is True

    def test_review_auto_mode_short_circuits(self):
        ctrl = _basic_controller()
        ctrl.set_review_auto_mode(True)
        ok, msg = ctrl.validate_non_empty(is_review=True)
        assert ok is True
        assert msg == ""


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_snapshot_includes_all_state(self):
        ctrl = _basic_controller(step=2)
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        ctrl.add_or_update_selection({"tool_name": "claude"}, is_review=True)
        ctrl.set_review_auto_mode(False)
        ctrl.toggle_tool_expand("coco", is_review=True)
        snap = ctrl.snapshot()
        assert snap["step"] == 2
        assert snap["pending_tool_name"] == "coco"
        assert len(snap["orchestrator_selections"]) == 1
        assert len(snap["review_selections"]) == 1
        assert snap["review_auto_mode"] is False

    def test_restore_replaces_state(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        snap = ctrl.snapshot()
        ctrl.clear_selections(is_review=False)
        ctrl.set_step(2)
        ctrl.set_review_auto_mode(True)
        ctrl.restore(snap)
        assert ctrl.step == 1
        assert len(ctrl.orchestrator_selections) == 1
        assert ctrl.review_auto_mode is False

    def test_restore_prefers_explicit_reviewers_over_legacy_auto_conflict(self):
        ctrl = _basic_controller(step=2)

        ctrl.restore({
            "step": 2,
            "review_auto_mode": True,
            "review_selections": {
                "review_1": {
                    "selection_key": "review_1",
                    "tool_name": "traex",
                    "model_name": "DeepSeek-V4-Flash/standard/high",
                }
            },
        })

        assert ctrl.review_auto_mode is False
        assert list(ctrl.review_selections) == ["review_1"]

    def test_restore_auto_mode_closes_legacy_picker_state(self):
        ctrl = _basic_controller(step=2)

        ctrl.restore({
            "step": 2,
            "review_auto_mode": True,
            "pending_tool_name": "traex",
            "pending_model_group": "DeepSeek-V4-Flash",
            "pending_model_profile": "standard",
            "pending_model_effort": "high",
            "model_page": 2,
        })

        assert ctrl.review_auto_mode is True
        assert ctrl.pending_tool_name is None
        assert ctrl.pending_model_group is None
        assert ctrl.pending_model_profile is None
        assert ctrl.pending_model_effort is None
        assert ctrl.model_page == 0

    def test_snapshot_is_detached_copy(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        snap = ctrl.snapshot()
        ctrl.add_or_update_selection({"tool_name": "claude"}, is_review=False)
        # Mutating controller after snapshot must not affect snapshot
        assert "claude" not in [v["tool_name"] for v in snap["orchestrator_selections"].values()]


# ---------------------------------------------------------------------------
# Card builders - orchestrator
# ---------------------------------------------------------------------------


class TestOrchestratorCombinedCard:
    def test_card_has_header_and_elements(self):
        ctrl = _basic_controller()
        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "coco", "display_name": "Coco"}],
            available_models=[{"name": "gpt-4", "display_name": "GPT-4"}],
            requirement="build a feature",
            session_key="sess_abc",
        )
        assert isinstance(card, dict)
        assert "header" in card
        # Card is wrapped by CardBuilder._wrap_card, so elements are under 'body'
        assert "body" in card
        assert "elements" in card["body"]
        assert isinstance(card["body"]["elements"], list)

    def test_card_contains_select_tool_action(self):
        ctrl = _basic_controller()
        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "coco", "display_name": "Coco"}],
            session_key="sess_abc",
        )
        tool_actions = _find_actions(card, "workflow_orchestrator_select_tool")
        assert tool_actions, "expected orchestrator_select_tool action"
        assert tool_actions[0]["tool_name"] == "coco"

    def test_card_contains_finish_action(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "coco", "display_name": "Coco"}],
            session_key="sess_abc",
        )
        finish_actions = _find_actions(card, "workflow_orchestrator_finish")
        assert finish_actions, "expected orchestrator_finish action when a tool is selected"
        assert finish_actions[0]["engine_session_key"] == "sess_abc"

    def test_card_shows_selected_item_with_remove_button(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection(
            {"tool_name": "coco", "display_name": "Coco", "model_name": "gpt-4"},
            is_review=False,
        )
        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "coco", "display_name": "Coco"}],
            session_key="sess_abc",
        )
        remove_actions = _find_actions(card, "workflow_orchestrator_remove")
        assert remove_actions, "expected orchestrator_remove action"
        # The remove button must include a selection_key so the handler can undo it
        assert remove_actions[0].get("selection_key"), "remove button missing selection_key"

    def test_card_has_clear_button_when_selections_exist(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "coco"}, is_review=False)
        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "coco", "display_name": "Coco"}],
            session_key="sess_abc",
        )
        clear_actions = _find_actions(card, "workflow_orchestrator_clear")
        assert clear_actions

    def test_pending_tool_expands_model_panel(self):
        ctrl = _basic_controller()
        ctrl.toggle_tool_expand("coco", is_review=False)
        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "coco", "display_name": "Coco"}],
            available_models=[{"name": "gpt-4", "display_name": "GPT-4"}],
            session_key="sess_abc",
        )
        model_actions = _find_actions(card, "workflow_orchestrator_select_model")
        assert len(model_actions) >= 2, "expected default + specific model buttons"
        # default-model button uses use_default_model, specific button uses model_name
        has_default = any(a.get("use_default_model") for a in model_actions)
        has_specific = any(a.get("model_name") == "gpt-4" for a in model_actions)
        assert has_default
        assert has_specific

    def test_pending_tool_model_panel_keeps_columns_under_feishu_element_limit(self):
        ctrl = _basic_controller()
        ctrl.toggle_tool_expand("traex", is_review=False)
        models = [
            {"name": f"traex-model-{i}", "display_name": f"Traex Model {i}"}
            for i in range(18)
        ]
        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "traex", "display_name": "Traex"}],
            available_models=models,
            session_key="sess_abc",
        )

        columns = _find_tags(card, "column")
        assert max(len(col.get("elements", [])) for col in columns) <= 20
        from src.card.render.payload_truncator import count_tagged_nodes
        assert count_tagged_nodes(card) <= 180

        model_actions = _find_actions(card, "workflow_orchestrator_select_model")
        model_names = {a.get("model_name") for a in model_actions if a.get("model_name")}
        assert model_names == {"traex-model-0"}
        assert any(a.get("use_default_model") for a in model_actions)

        group_select = next(
            s for s in _find_tags(card, "select_static")
            if s.get("value", {}).get("action") == "workflow_orchestrator_select_model_group"
        )
        option_values = {opt.get("value") for opt in group_select.get("options", [])}
        assert option_values == {f"traex-model-{i}" for i in range(18)}

    def test_pending_tool_model_panel_uses_select_for_large_model_lists(self):
        ctrl = _basic_controller()
        ctrl.toggle_tool_expand("traex", is_review=False)
        models = [
            {"name": f"traex-model-{i}", "display_name": f"Traex Model {i}"}
            for i in range(90)
        ]

        first_page = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "traex", "display_name": "Traex"}],
            available_models=models,
            session_key="sess_abc",
        )
        first_model_actions = _find_actions(first_page, "workflow_orchestrator_select_model")
        first_model_names = {
            a.get("model_name") for a in first_model_actions if a.get("model_name")
        }
        assert first_model_names == {"traex-model-0"}

        group_select = next(
            s for s in _find_tags(first_page, "select_static")
            if s.get("value", {}).get("action") == "workflow_orchestrator_select_model_group"
        )
        option_values = {opt.get("value") for opt in group_select.get("options", [])}
        assert len(option_values) == 90
        assert "traex-model-0" in option_values
        assert "traex-model-89" in option_values

        from src.card.render.payload_truncator import count_tagged_nodes
        assert count_tagged_nodes(first_page) <= 180

    def test_large_variant_models_render_as_cascading_selects(self):
        ctrl = _basic_controller()
        ctrl.select_tool("traex", is_review=False)
        ctrl.set_model_group("traex", "openrouter-3o", is_review=False)
        models = [
            {"name": "openrouter-3o/low", "display_name": "openrouter-3o/low"},
            {"name": "openrouter-3o/medium", "display_name": "openrouter-3o/medium"},
            {"name": "openrouter-3o/high", "display_name": "openrouter-3o/high"},
            {"name": "openrouter-3o/max", "display_name": "openrouter-3o/max"},
            {"name": "openrouter-3o/max/low", "display_name": "openrouter-3o/max/low"},
            {"name": "openrouter-3o/max/high", "display_name": "openrouter-3o/max/high"},
            {"name": "openrouter-2o/low", "display_name": "openrouter-2o/low"},
            {"name": "openrouter-2o/high", "display_name": "openrouter-2o/high"},
        ]

        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "traex", "display_name": "Traex"}],
            available_models=models,
            session_key="sess_abc",
            chat_id="chat_1",
            project_id="proj_1",
        )

        selects = _find_tags(card, "select_static")
        select_actions = {s.get("value", {}).get("action") for s in selects}
        assert "workflow_orchestrator_select_tool" in select_actions
        assert "workflow_orchestrator_select_model_group" in select_actions
        assert "workflow_orchestrator_select_model_profile" in select_actions
        assert "workflow_orchestrator_select_model_effort" in select_actions

        group_select = next(
            s for s in selects
            if s.get("value", {}).get("action") == "workflow_orchestrator_select_model_group"
        )
        group_values = {opt.get("value") for opt in group_select.get("options", [])}
        assert group_values == {"openrouter-3o", "openrouter-2o"}

        model_actions = _find_actions(card, "workflow_orchestrator_select_model")
        concrete_model_actions = [a for a in model_actions if a.get("model_name")]
        concrete_model_names = {a["model_name"] for a in concrete_model_actions}
        assert concrete_model_names == {"openrouter-3o/low"}

    def test_codex_reasoning_capabilities_render_workflow_effort_dropdown(self):
        ctrl = _basic_controller()
        ctrl.select_tool("codex", is_review=False)
        ctrl.set_model_group("codex", "gpt-5.6-sol", is_review=False)
        ctrl.set_model_effort("codex", "ultra", is_review=False)
        models = [
            {
                "name": "gpt-5.6-sol",
                "display_name": "GPT-5.6-Sol",
                "reasoning_efforts": [
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                    "max",
                    "ultra",
                ],
                "adapted_reasoning_effort": "high",
            }
        ]

        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "codex", "display_name": "Codex"}],
            available_models=models,
            session_key="sess_abc",
            chat_id="chat_1",
            project_id="proj_1",
        )

        effort_select = next(
            select
            for select in _find_tags(card, "select_static")
            if select.get("value", {}).get("action")
            == "workflow_orchestrator_select_model_effort"
        )
        assert effort_select["initial_option"] == "ultra"

        model_actions = _find_actions(card, "workflow_orchestrator_select_model")
        selected = next(action for action in model_actions if action.get("model_name"))
        assert selected["model_name"] == "gpt-5.6-sol/ultra"
        assert selected["model_profile"] == "standard"
        assert selected["model_effort"] == "ultra"

    def test_codex_workflow_uses_adapter_default_model_and_effort(self):
        ctrl = _basic_controller()
        ctrl.select_tool("codex", is_review=False)
        models = [
            {
                "name": "gpt-5.5",
                "display_name": "GPT-5.5",
                "reasoning_efforts": ["low", "medium", "high", "xhigh"],
                "adapted_reasoning_effort": "high",
                "is_default": False,
            },
            {
                "name": "gpt-5.6-sol",
                "display_name": "GPT-5.6-Sol",
                "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"],
                "adapted_reasoning_effort": "high",
                "is_default": True,
            },
        ]

        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "codex", "display_name": "Codex"}],
            available_models=models,
            session_key="sess_abc",
            chat_id="chat_1",
            project_id="proj_1",
        )

        selects = _find_tags(card, "select_static")
        group_select = next(
            select
            for select in selects
            if select.get("value", {}).get("action")
            == "workflow_orchestrator_select_model_group"
        )
        effort_select = next(
            select
            for select in selects
            if select.get("value", {}).get("action")
            == "workflow_orchestrator_select_model_effort"
        )
        assert group_select["initial_option"] == "gpt-5.6-sol"
        assert effort_select["initial_option"] == "high"

        model_actions = _find_actions(card, "workflow_orchestrator_select_model")
        selected = next(action for action in model_actions if action.get("model_name"))
        assert selected["model_name"] == "gpt-5.6-sol/high"

    def test_codex_workflow_changed_group_uses_group_default_effort(self):
        ctrl = _basic_controller()
        ctrl.select_tool("codex", is_review=False)
        ctrl.set_model_group("codex", "gpt-5.5", is_review=False)
        models = [
            {
                "name": "gpt-5.5",
                "display_name": "GPT-5.5",
                "reasoning_efforts": ["low", "medium", "high", "xhigh"],
                "adapted_reasoning_effort": "high",
                "is_default": False,
            },
            {
                "name": "gpt-5.6-sol",
                "display_name": "GPT-5.6-Sol",
                "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"],
                "adapted_reasoning_effort": "high",
                "is_default": True,
            },
        ]

        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "codex", "display_name": "Codex"}],
            available_models=models,
            session_key="sess_abc",
            chat_id="chat_1",
            project_id="proj_1",
        )

        effort_select = next(
            select
            for select in _find_tags(card, "select_static")
            if select.get("value", {}).get("action")
            == "workflow_orchestrator_select_model_effort"
        )
        assert effort_select["initial_option"] == "high"

        model_actions = _find_actions(card, "workflow_orchestrator_select_model")
        selected = next(action for action in model_actions if action.get("model_name"))
        assert selected["model_name"] == "gpt-5.5/high"

    def test_pending_tool_model_panel_stays_under_payload_budget_with_many_tools(self):
        import json

        from src.card.render.payload_truncator import count_tagged_nodes
        from src.card.thresholds import THRESHOLDS

        ctrl = _basic_controller()
        ctrl.toggle_tool_expand("traex", is_review=False)
        models = [
            {"name": f"traex-model-{i}", "display_name": f"Traex Model {i}"}
            for i in range(90)
        ]
        tools = [
            {"tool_name": name, "display_name": name.title(), "description": "AI 编程工具"}
            for name in ["traex", "claude", "codex", "aiden", "gemini", "coco"]
        ]

        card = ctrl.build_orchestrator_combined_card(
            available_tools=tools,
            available_models=models,
            requirement="检查 Traex 模型列表分页",
            session_key="sess_abc",
            chat_id="chat_1",
            project_id="proj_1",
        )

        assert count_tagged_nodes(card) <= THRESHOLDS["CARD_NODE_BUDGET"]
        assert len(json.dumps(card, ensure_ascii=False).encode("utf-8")) <= THRESHOLDS["CARD_BYTE_BUDGET"]

    def test_requirement_is_rendered_when_given(self):
        ctrl = _basic_controller()
        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "coco"}],
            requirement="build a feature",
        )
        text = str(card)
        assert "build a feature" in text

    def test_error_message_is_rendered(self):
        ctrl = _basic_controller()
        ctrl.error_message = "请至少选择一个 Agent"
        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "coco"}],
        )
        assert "请至少选择一个 Agent" in str(card)


# ---------------------------------------------------------------------------
# Card builders - review
# ---------------------------------------------------------------------------


class TestReviewCombinedCard:
    def test_review_card_has_auto_mode_button(self):
        ctrl = _basic_controller()
        card = ctrl.build_review_combined_card(
            available_tools=[{"tool_name": "claude"}],
            session_key="sess_abc",
        )
        auto_actions = _find_actions(card, "workflow_review_toggle_auto")
        assert auto_actions, "review card should expose toggle_auto"

    def test_auto_mode_makes_review_card_valid(self):
        ctrl = _basic_controller()
        ctrl.set_review_auto_mode(True)
        card = ctrl.build_review_combined_card(
            available_tools=[{"tool_name": "claude"}],
            session_key="sess_abc",
        )
        # When auto mode is on, the finish button should be present (validate ok)
        finish_actions = _find_actions(card, "workflow_review_finish")
        assert finish_actions, "auto mode should enable review finish action"

    def test_review_card_uses_review_actions(self):
        ctrl = _basic_controller()
        ctrl.add_or_update_selection({"tool_name": "claude"}, is_review=True)
        card = ctrl.build_review_combined_card(
            available_tools=[{"tool_name": "claude"}],
            session_key="sess_abc",
        )
        assert _find_actions(card, "workflow_review_select_tool")
        assert _find_actions(card, "workflow_review_remove")
        assert _find_actions(card, "workflow_review_clear")
        assert _find_actions(card, "workflow_review_finish")

    def test_review_card_does_not_mention_orchestrator_actions(self):
        ctrl = _basic_controller()
        card = ctrl.build_review_combined_card(
            available_tools=[{"tool_name": "claude"}],
            session_key="sess_abc",
        )
        assert not _find_actions(card, "workflow_orchestrator_select_tool")

    def test_review_card_uses_review_cascade_actions(self):
        ctrl = _basic_controller(step=2)
        ctrl.select_tool("traex", is_review=True)
        ctrl.set_model_group("traex", "openrouter-3o", is_review=True)
        card = ctrl.build_review_combined_card(
            available_tools=[{"tool_name": "traex", "display_name": "Traex"}],
            available_models=[
                {"name": "openrouter-3o/low", "display_name": "openrouter-3o/low"},
                {"name": "openrouter-3o/max/high", "display_name": "openrouter-3o/max/high"},
            ],
            session_key="sess_abc",
        )

        selects = _find_tags(card, "select_static")
        select_actions = {s.get("value", {}).get("action") for s in selects}
        assert "workflow_review_select_tool" in select_actions
        assert "workflow_review_select_model_group" in select_actions
        assert "workflow_review_select_model_profile" in select_actions

        model_actions = _find_actions(card, "workflow_review_select_model")
        model_names = {a.get("model_name") for a in model_actions if a.get("model_name")}
        assert model_names == {"openrouter-3o/low"}
        assert not _find_actions(card, "workflow_orchestrator_select_model_group")


# ---------------------------------------------------------------------------
# SelectionItem - data model sanity
# ---------------------------------------------------------------------------


class TestSelectionItem:
    def test_to_dict_round_trip(self):
        item = SelectionItem(
            selection_key="abc",
            tool_name="coco",
            provider="workflow",
            display_name="Coco",
            supports_model=True,
            model_name="gpt-4",
        )
        data = item.to_dict()
        assert data["tool_name"] == "coco"
        assert data["model_name"] == "gpt-4"
        assert data["selection_key"] == "abc"

    def test_label_with_model(self):
        item = SelectionItem(
            selection_key="k",
            tool_name="coco",
            display_name="Coco",
            model_name="gpt-4",
        )
        label = item.label()
        assert "Coco" in label
        assert "gpt-4" in label

    def test_label_with_default_model(self):
        item = SelectionItem(
            selection_key="k",
            tool_name="coco",
            display_name="Coco",
            use_default_model=True,
        )
        assert "默认模型" in item.label()


# ---------------------------------------------------------------------------
# Cancel button validation
# ---------------------------------------------------------------------------


class TestCancelButtonValidation:
    """Tests for cancel button presence and correctness in selection cards."""

    def test_orchestrator_card_has_cancel_button(self):
        """Orchestrator selection card must contain cancel button."""
        ctrl = _basic_controller()
        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "coco"}],
            session_key="sess_abc",
            chat_id="chat_1",
            project_id="proj_1",
        )
        cancel_actions = _find_actions(card, "workflow_cancel")
        assert cancel_actions, "orchestrator card should have cancel button"

    def test_review_card_has_cancel_button(self):
        """Review selection card must contain cancel button."""
        ctrl = _basic_controller()
        card = ctrl.build_review_combined_card(
            available_tools=[{"tool_name": "claude"}],
            session_key="sess_abc",
            chat_id="chat_1",
            project_id="proj_1",
        )
        cancel_actions = _find_actions(card, "workflow_cancel")
        assert cancel_actions, "review card should have cancel button"

    def test_cancel_button_has_correct_payload(self):
        """Cancel button must contain correct session_key, chat_id, and project_id."""
        ctrl = _basic_controller()
        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "coco"}],
            session_key="sess_123",
            chat_id="chat_456",
            project_id="proj_789",
        )
        cancel_actions = _find_actions(card, "workflow_cancel")
        # There may be multiple cancel buttons (one in header, one in footer)
        assert len(cancel_actions) >= 1

        # Check the first cancel button payload
        payload = cancel_actions[0]
        assert payload.get("chat_id") == "chat_456"
        assert payload.get("project_id") == "proj_789"
        assert payload.get("engine_session_key") == "sess_123"


class TestSchemaV2Compatibility:
    def test_selection_error_uses_no_note_tag(self):
        """Feishu Schema 2.0 rejects ``tag=note`` in workflow selection cards."""
        ctrl = _basic_controller()
        ctrl.error_message = "请选择一个主编排 Agent"

        card = ctrl.build_orchestrator_combined_card(
            available_tools=[{"tool_name": "coco"}],
            session_key="sess_123",
            chat_id="chat_456",
            project_id="proj_789",
        )

        assert not _find_tags(card, "note")
        markdown = "\n".join(node.get("content", "") for node in _find_tags(card, "markdown"))
        assert "请选择一个主编排 Agent" in markdown
