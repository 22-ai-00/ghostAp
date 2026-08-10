"""Contracts for explicit ACP model-selection and activation cards."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterable

from src.card.actions import dispatch as action_ids
from src.card.builders.system import SystemBuilder


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _card(result: tuple[str, str]) -> dict[str, Any]:
    message_type, content = result
    assert message_type == "interactive"
    return json.loads(content)


def _nodes(card: dict[str, Any], tag: str) -> list[dict[str, Any]]:
    return [node for node in _walk(card) if node.get("tag") == tag]


def _callback_value(node: dict[str, Any]) -> dict[str, Any]:
    value = node.get("value")
    assert isinstance(value, dict)
    behaviors = node.get("behaviors")
    assert behaviors == [{"type": "callback", "value": value}]
    return value


def _variant(
    name: str,
    model: str,
    *,
    profile: str | None = None,
    effort: str | None = None,
    is_default: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        model=model,
        profile=profile,
        effort=effort,
        is_default=is_default,
    )


def _models() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            name="c_o_new",
            description="Primary coding family",
            is_default=True,
            selection_variants=(
                _variant(
                    "c_o_new/standard/high",
                    "c_o_new",
                    profile="standard",
                    effort="high",
                    is_default=True,
                ),
                _variant(
                    "c_o_new/max/high",
                    "c_o_new",
                    profile="max",
                    effort="high",
                ),
                _variant(
                    "c_o_new/max/xhigh",
                    "c_o_new",
                    profile="max",
                    effort="xhigh",
                ),
            ),
            reasoning_efforts=(),
            default_reasoning_effort=None,
        ),
        SimpleNamespace(
            name="openrouter-3o",
            description="Alternate family",
            is_default=False,
            selection_variants=(
                _variant(
                    "openrouter-3o/standard/medium",
                    "openrouter-3o",
                    profile="standard",
                    effort="medium",
                    is_default=True,
                ),
            ),
            reasoning_efforts=(),
            default_reasoning_effort=None,
        ),
    ]


def test_explicit_acp_model_action_ids_are_stable() -> None:
    assert getattr(action_ids, "SELECT_ACP_MODEL", None) == "select_acp_model"
    assert getattr(action_ids, "REFRESH_ACP_MODELS", None) == "refresh_acp_models"
    assert getattr(action_ids, "SELECT_ACP_MODEL_GROUP", None) == "select_acp_model_group"
    assert getattr(action_ids, "SELECT_ACP_MODEL_PROFILE", None) == "select_acp_model_profile"
    assert getattr(action_ids, "SELECT_ACP_MODEL_EFFORT", None) == "select_acp_model_effort"


def test_cascade_card_restores_saved_selection_and_exact_callback_context() -> None:
    build = getattr(SystemBuilder, "build_acp_model_cascade_card", None)
    assert callable(build)

    card = _card(
        build(
            _models(),
            "traex",
            project_id="project-1",
            current_model="c_o_new/max/xhigh",
        )
    )
    selects = {node.get("name"): node for node in _nodes(card, "select_static")}
    assert set(selects) == {"model_group", "model_profile", "model_effort"}
    assert selects["model_group"]["initial_option"] == "c_o_new"
    assert selects["model_profile"]["initial_option"] == "max"
    assert selects["model_effort"]["initial_option"] == "xhigh"

    expected_actions = {
        "model_group": action_ids.SELECT_ACP_MODEL_GROUP,
        "model_profile": action_ids.SELECT_ACP_MODEL_PROFILE,
        "model_effort": action_ids.SELECT_ACP_MODEL_EFFORT,
    }
    for name, select in selects.items():
        value = _callback_value(select)
        assert value == {
            "action": expected_actions[name],
            "tool_name": "traex",
            "project_id": "project-1",
            "current_model": "c_o_new/max/xhigh",
            "model_group": "c_o_new",
            "model_profile": "max",
            "model_effort": "xhigh",
        }


def test_cascade_card_static_selects_omit_unsupported_width() -> None:
    card = _card(
        SystemBuilder.build_acp_model_cascade_card(
            _models(),
            "traex",
            project_id="project-1",
            current_model="c_o_new/max/xhigh",
        )
    )

    selects = _nodes(card, "select_static")
    assert selects
    assert all("width" not in select for select in selects)


def test_pending_cascade_dimensions_override_saved_selection_and_confirm_exact_value() -> None:
    build = getattr(SystemBuilder, "build_acp_model_cascade_card", None)
    assert callable(build)

    card = _card(
        build(
            _models(),
            "traex",
            project_id="project-1",
            current_model="c_o_new/max/xhigh",
            pending_group="openrouter-3o",
            pending_profile="standard",
            pending_effort="medium",
        )
    )
    selects = {node.get("name"): node for node in _nodes(card, "select_static")}
    assert selects["model_group"]["initial_option"] == "openrouter-3o"
    assert selects["model_profile"]["initial_option"] == "standard"
    assert selects["model_effort"]["initial_option"] == "medium"

    callbacks = [_callback_value(button) for button in _nodes(card, "button")]
    confirm = next(
        value
        for value in callbacks
        if value["action"] == action_ids.SELECT_ACP_MODEL
        and value.get("use_default_model") is False
    )
    assert confirm == {
        "action": action_ids.SELECT_ACP_MODEL,
        "tool_name": "traex",
        "project_id": "project-1",
        "current_model": "c_o_new/max/xhigh",
        "model_group": "openrouter-3o",
        "model_profile": "standard",
        "model_effort": "medium",
        "model_name": "openrouter-3o/standard/medium",
        "use_default_model": False,
    }

    refresh = next(value for value in callbacks if value["action"] == action_ids.REFRESH_ACP_MODELS)
    assert refresh["model_group"] == "openrouter-3o"
    assert refresh["model_profile"] == "standard"
    assert refresh["model_effort"] == "medium"


def test_plain_model_card_has_default_refresh_and_current_model_actions() -> None:
    build = getattr(SystemBuilder, "build_acp_model_cascade_card", None)
    assert callable(build)
    models = [
        SimpleNamespace(
            name="model-a",
            description="A",
            is_default=False,
            selection_variants=(),
            reasoning_efforts=(),
            default_reasoning_effort=None,
        ),
        SimpleNamespace(
            name="model-b",
            description="B",
            is_default=True,
            selection_variants=(),
            reasoning_efforts=(),
            default_reasoning_effort=None,
        ),
    ]

    card = _card(
        build(models, "coco", project_id="project-2", current_model="model-a")
    )
    buttons = _nodes(card, "button")
    callbacks = [_callback_value(button) for button in buttons]
    default = next(value for value in callbacks if value.get("use_default_model") is True)
    assert default["action"] == action_ids.SELECT_ACP_MODEL
    assert default["tool_name"] == "coco"
    current = next(value for value in callbacks if value.get("model_name") == "model-a")
    assert current["model_group"] == "model-a"
    current_button = buttons[callbacks.index(current)]
    assert current_button["type"] == "primary"
    assert any(value["action"] == action_ids.REFRESH_ACP_MODELS for value in callbacks)


def test_activation_cards_cover_initializing_ready_and_retryable_failure() -> None:
    initializing = getattr(SystemBuilder, "build_acp_programming_initializing_card", None)
    ready = getattr(SystemBuilder, "build_acp_programming_ready_card", None)
    failed = getattr(SystemBuilder, "build_acp_programming_failed_card", None)
    assert callable(initializing) and callable(ready) and callable(failed)

    loading_card = _card(initializing("codex", "gpt-5.6-sol/high", "project-3", None))
    assert "gpt-5.6-sol/high" in json.dumps(loading_card, ensure_ascii=False)
    assert not _nodes(loading_card, "button")

    ready_card = _card(ready("codex", "gpt-5.6-sol/high", "project-3", None))
    ready_actions = [_callback_value(button) for button in _nodes(ready_card, "button")]
    assert ready_actions == [
        {
            "action": action_ids.REFRESH_ACP_MODELS,
            "tool_name": "codex",
            "project_id": "project-3",
            "current_model": "gpt-5.6-sol/high",
        }
    ]

    failed_card = _card(
        failed(
            "codex",
            "gpt-5.6-sol/high",
            "startup rejected",
            "project-3",
            None,
            model_group="gpt-5.6-sol",
            model_profile=None,
            model_effort="high",
        )
    )
    assert "startup rejected" in json.dumps(failed_card, ensure_ascii=False)
    failed_actions = [_callback_value(button) for button in _nodes(failed_card, "button")]
    assert failed_actions[0] == {
        "action": action_ids.SELECT_ACP_MODEL,
        "tool_name": "codex",
        "project_id": "project-3",
        "model_group": "gpt-5.6-sol",
        "model_profile": None,
        "model_effort": "high",
        "model_name": "gpt-5.6-sol/high",
        "use_default_model": False,
    }
    assert failed_actions[1] == {
        "action": action_ids.REFRESH_ACP_MODELS,
        "tool_name": "codex",
        "project_id": "project-3",
        "current_model": "gpt-5.6-sol/high",
    }

    default_failed = _card(
        failed("codex", None, "default startup rejected", "project-3", None)
    )
    default_retry = _callback_value(_nodes(default_failed, "button")[0])
    assert default_retry == {
        "action": action_ids.SELECT_ACP_MODEL,
        "tool_name": "codex",
        "project_id": "project-3",
        "model_group": None,
        "model_profile": None,
        "model_effort": None,
        "model_name": None,
        "use_default_model": True,
    }


def test_model_discovery_loading_and_error_frames_keep_retry_coordinates() -> None:
    loading = getattr(SystemBuilder, "build_acp_model_loading_card", None)
    error = getattr(SystemBuilder, "build_acp_model_error_card", None)
    assert callable(loading) and callable(error)

    loading_card = _card(loading("codex", "project-4", "thread-4"))
    assert not _nodes(loading_card, "button")
    error_card = _card(error("codex", "project-4", "thread-4"))
    retry = _callback_value(_nodes(error_card, "button")[0])
    assert retry == {
        "action": action_ids.REFRESH_ACP_MODELS,
        "tool_name": "codex",
        "project_id": "project-4",
        "thread_root_id": "thread-4",
    }
