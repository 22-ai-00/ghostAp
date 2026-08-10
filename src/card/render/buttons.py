"""Render CardSession buttons with one responsive layout implementation."""

from __future__ import annotations

import logging
import re

from src.card.actions.dispatch import (
    ENGINE_STOP,
    MODE_COMPACT,
    MODE_FULL,
    SHOW_STATUS,
    SPEC_STOP,
)
from src.card.render.budget import RenderBudget
from src.card.state.button_intent import ButtonIntent
from src.card.state.models import ButtonSpec, CardState
from src.card.ui_text import UI_TEXT

logger = logging.getLogger(__name__)

INTENT_TO_ACTION_ID: dict[str, str] = {
    ButtonIntent.SPEC_STOP: SPEC_STOP,
    ButtonIntent.ENGINE_STOP: ENGINE_STOP,
    ButtonIntent.MODE_FULL: MODE_FULL,
    ButtonIntent.MODE_COMPACT: MODE_COMPACT,
    ButtonIntent.SHOW_STATUS: SHOW_STATUS,
}

_missing_intents = {intent.value for intent in ButtonIntent} - set(INTENT_TO_ACTION_ID)
if _missing_intents:
    raise RuntimeError(f"Button intents without action mapping: {sorted(_missing_intents)}")
del _missing_intents

_DESTRUCTIVE_ACTIONS = frozenset({ENGINE_STOP, SPEC_STOP})
_CONFIRM_TITLE_MAP = {
    ButtonIntent.ENGINE_STOP: "card_btn_confirm_stop_title_normal",
    ButtonIntent.SPEC_STOP: "card_btn_confirm_stop_title_normal",
}
_LEADING_EMOJI_RE = re.compile(
    r"^[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*"
)


def _get_confirm_title(action_id: str, button_text: str = "") -> str:
    """Return an exact action title, then a label-derived generic title."""
    ui_key = _CONFIRM_TITLE_MAP.get(action_id)
    if ui_key:
        return UI_TEXT.get(
            ui_key,
            UI_TEXT.get("card_btn_confirm_default_title", "确认操作？"),
        )
    clean_text = _LEADING_EMOJI_RE.sub("", button_text).strip()
    if clean_text:
        return UI_TEXT.get("btn_confirm_template", "确认「{text}」？").format(
            text=clean_text
        )
    return UI_TEXT.get("card_btn_confirm_default_title", "确认操作？")


def _resolve_action_id(spec: ButtonSpec) -> str | None:
    action_id = spec.action_id
    if not action_id.startswith("intent."):
        return action_id
    resolved = INTENT_TO_ACTION_ID.get(action_id)
    if resolved is None:
        logger.warning(
            "Unknown ButtonIntent '%s', rendering as disabled button",
            action_id,
        )
    return resolved


def _render_button(
    spec: ButtonSpec,
    *,
    engine_type: str | None = None,
    budget: RenderBudget | None = None,
) -> dict:
    """Render one URL, disabled, or callback button."""
    size = budget.button_size if budget else "medium"
    text = {"tag": "plain_text", "content": spec.text}
    if spec.url:
        return {
            "tag": "button",
            "text": text,
            "type": spec.type,
            "size": size,
            "behaviors": [{"type": "open_url", "default_url": spec.url}],
        }

    action_id = _resolve_action_id(spec)
    if action_id is None:
        return {
            "tag": "button",
            "text": text,
            "type": "default",
            "size": size,
            "disabled": True,
            "disabled_tips": {
                "tag": "plain_text",
                "content": UI_TEXT["card_btn_disabled_tips"],
            },
        }
    if spec.disabled:
        button = {
            "tag": "button",
            "text": {
                "tag": "plain_text",
                "content": spec.disabled_text or spec.text,
            },
            "type": spec.type,
            "size": size,
            "disabled": True,
        }
        if spec.disabled_text:
            button["disabled_tips"] = {
                "tag": "plain_text",
                "content": spec.disabled_text,
            }
        return button

    value = dict(spec.value or {})
    value.setdefault("action", action_id)
    if action_id == ENGINE_STOP and engine_type:
        value["engine_type"] = engine_type
    if action_id in {MODE_FULL, MODE_COMPACT} and engine_type:
        value["action"] = f"{engine_type}_{action_id}"

    button = {
        "tag": "button",
        "text": text,
        "type": spec.type,
        "value": value,
        "behaviors": [{"type": "callback", "value": value}],
        "size": size,
    }
    confirm_text = spec.confirm
    if confirm_text is None and action_id in _DESTRUCTIVE_ACTIONS:
        confirm_text = UI_TEXT.get(
            "card_btn_confirm_default_text",
            "此操作不可撤销，确认继续？",
        )
    if confirm_text is not None:
        button["confirm"] = {
            "title": {
                "tag": "plain_text",
                "content": _get_confirm_title(spec.action_id, spec.text),
            },
            "text": {"tag": "plain_text", "content": confirm_text},
        }
    return button


def _column(buttons: list[dict], *, vertical: bool = False) -> dict:
    column = {
        "tag": "column",
        "width": "weighted",
        "weight": 1,
        "elements": buttons,
    }
    if vertical:
        column["vertical_align"] = "top"
    return column


def _column_set(
    columns: list[dict],
    *,
    flex_mode: str,
    horizontal_spacing: str,
) -> dict:
    return {
        "tag": "column_set",
        "flex_mode": flex_mode,
        "background_style": "default",
        "horizontal_spacing": horizontal_spacing,
        "columns": columns,
    }


def build_responsive_button_row(
    buttons: list[dict],
    *,
    mobile_force_vertical: bool = False,
    horizontal_spacing: str = "8px",
) -> list[dict]:
    """Arrange buttons into mobile-safe Card 2.0 column sets."""
    if not buttons:
        return []
    if mobile_force_vertical:
        if len(buttons) == 2:
            return [
                _column_set(
                    [_column([button])],
                    flex_mode="none",
                    horizontal_spacing=horizontal_spacing,
                )
                for button in buttons
            ]
        return [
            _column_set(
                [_column(buttons, vertical=len(buttons) >= 3)],
                flex_mode="none",
                horizontal_spacing=horizontal_spacing,
            )
        ]
    if len(buttons) <= 3:
        return [
            _column_set(
                [_column([button]) for button in buttons]
                if len(buttons) > 1
                else [_column(buttons)],
                flex_mode="bisect" if len(buttons) == 2 else "none",
                horizontal_spacing=horizontal_spacing,
            )
        ]
    return [
        _column_set(
            [_column([button]) for button in buttons[index : index + 2]],
            flex_mode="bisect" if len(buttons[index : index + 2]) == 2 else "none",
            horizontal_spacing=horizontal_spacing,
        )
        for index in range(0, len(buttons), 2)
    ]


def render_buttons(
    state: CardState,
    budget: RenderBudget | None = None,
) -> list[dict]:
    if not state.buttons:
        return []
    engine_type = state.metadata.engine_type if state.metadata else None
    buttons = [
        _render_button(spec, engine_type=engine_type, budget=budget)
        for spec in state.buttons
    ]
    force_vertical = bool(budget and budget.mobile_force_vertical)
    if len(buttons) == 3:
        force_vertical = force_vertical or any(
            len(str(button.get("text", {}).get("content", ""))) > 8
            for button in buttons
        )
    return build_responsive_button_row(
        buttons,
        mobile_force_vertical=force_vertical,
        horizontal_spacing=(
            budget.button_horizontal_spacing
            if budget and budget.button_horizontal_spacing
            else "8px"
        ),
    )
