import pytest

from src.card.shared import build_mode_buttons
from src.mode.manager import InteractionMode


def _actions(buttons: list[dict]) -> list[str]:
    return [button["behaviors"][0]["value"]["action"] for button in buttons]


@pytest.mark.parametrize(
    "mode",
    [
        None,
        InteractionMode.SMART,
        InteractionMode.CODEX,
        InteractionMode.AIDEN,
    ],
)
def test_inactive_or_unhandled_mode_footer_has_no_cross_tool_entry_guidance(
    mode: InteractionMode | None,
) -> None:
    actions = _actions(
        build_mode_buttons(
            mode,
            "project-1",
            thread_root_id="thread-1",
        )
    )

    assert not [action for action in actions if action.startswith("enter_")]


@pytest.mark.parametrize(
    ("mode", "exit_action"),
    [
        (InteractionMode.COCO, "exit_coco"),
        (InteractionMode.CLAUDE, "exit_claude"),
        (InteractionMode.GEMINI, "exit_gemini"),
        (InteractionMode.TRAEX, "exit_traex"),
    ],
)
def test_active_mode_footer_keeps_exit_and_project_switch(
    mode: InteractionMode,
    exit_action: str,
) -> None:
    buttons = build_mode_buttons(
        mode,
        "project-1",
        thread_root_id="thread-1",
    )

    assert _actions(buttons) == [exit_action, "switch_project"]
    exit_value = buttons[0]["behaviors"][0]["value"]
    assert exit_value["project_id"] == "project-1"
    assert exit_value["thread_root_id"] == "thread-1"

