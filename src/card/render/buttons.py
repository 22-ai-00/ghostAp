"""Button group rendering.

Handles ButtonIntent → action_id mapping and dynamic flex_mode selection.
"""

from __future__ import annotations

import logging
import re

from src.card.actions.dispatch import (
    APPROVE_ACTION,
    DEEP_RESUME,
    DEEP_STOP,
    ENGINE_STOP,
    MODE_COMPACT,
    MODE_FULL,
    REJECT_ACTION,
    SHOW_STATUS,
    SHOW_WORKFLOW_MENU,
    SPEC_RESUME,
    SPEC_SKIP_RETRY,
    SPEC_STOP,
    WORKFLOW_CANCEL,
    WORKFLOW_CONFIRM_START,
    WORKFLOW_CONFIRM_TOOLS,
    WORKFLOW_LIST_TEMPLATES,
    WORKFLOW_ORCHESTRATOR_FINISH,
    WORKFLOW_ORCHESTRATOR_SELECT_MODEL,
    WORKFLOW_ORCHESTRATOR_SELECT_TOOL,
    WORKFLOW_REGENERATE_SCRIPT,
    WORKFLOW_REVIEW_FINISH,
    WORKFLOW_REVIEW_SELECT_MODEL,
    WORKFLOW_REVIEW_SELECT_TOOL,
    WORKFLOW_SELECT_TOOL,
    WORKFLOW_SHOW_HELP,
)
from src.card.engine_meta import ENGINE_LABEL_DEFAULT, ENGINE_LABELS
from src.card.render.budget import RenderBudget
from src.card.state.button_intent import ButtonIntent
from src.card.state.models import ButtonSpec, CardState
from src.card.ui_text import UI_TEXT

# ---------------------------------------------------------------------------
# ButtonIntent → action_id mapping (single source of truth)
# ---------------------------------------------------------------------------
INTENT_TO_ACTION_ID: dict[str, str] = {
    # Deep engine
    ButtonIntent.DEEP_RESUME: DEEP_RESUME,
    ButtonIntent.DEEP_STOP: DEEP_STOP,

    # Spec engine
    ButtonIntent.SPEC_RESUME: SPEC_RESUME,
    ButtonIntent.SPEC_STOP: SPEC_STOP,
    ButtonIntent.SPEC_SKIP_RETRY: SPEC_SKIP_RETRY,

    # Engine stop (generic, routed by engine_type at dispatch time)
    ButtonIntent.ENGINE_STOP: ENGINE_STOP,

    # View mode toggle
    ButtonIntent.MODE_FULL: MODE_FULL,
    ButtonIntent.MODE_COMPACT: MODE_COMPACT,

    # Global
    ButtonIntent.SHOW_STATUS: SHOW_STATUS,

    # Approval
    ButtonIntent.APPROVE: APPROVE_ACTION,
    ButtonIntent.REJECT: REJECT_ACTION,
}

# Import-time consistency check: every ButtonIntent must have a mapping.
_intent_enum_members = set(ButtonIntent)
_mapped_keys = set(INTENT_TO_ACTION_ID.keys())
if _intent_enum_members != _mapped_keys:
    raise RuntimeError(
        f"ButtonIntent ↔ INTENT_TO_ACTION_ID out of sync.\n"
        f"  In enum but not mapped: {_intent_enum_members - _mapped_keys}\n"
        f"  In mapping but not enum: {_mapped_keys - _intent_enum_members}"
    )
del _intent_enum_members, _mapped_keys

logger = logging.getLogger(__name__)

# Resolved action_ids that require confirm dialog (destructive/irreversible actions)
_DESTRUCTIVE_ACTIONS = frozenset({
    ENGINE_STOP, DEEP_STOP, SPEC_STOP,
    WORKFLOW_CANCEL,
    APPROVE_ACTION,
})

# Intents that represent "stop/cancel" actions
_STOP_INTENTS = frozenset({
    "intent.engine.stop", "intent.deep.stop",
    "intent.spec.stop",
    WORKFLOW_CANCEL,
})

# White-list mapping: action_id (exact match) → confirm dialog title UI_TEXT key
# Keys MUST be ButtonIntent enum values or registered action_ids.
_CONFIRM_TITLE_MAP: dict[str, str] = {
    # Stop intents — normal (gentle question for first-stage stop)
    ButtonIntent.ENGINE_STOP: "card_btn_confirm_stop_title_normal",
    ButtonIntent.DEEP_STOP: "card_btn_confirm_stop_title_normal",
    ButtonIntent.SPEC_STOP: "card_btn_confirm_stop_title_normal",
    # Retry/resume intents
    ButtonIntent.DEEP_RESUME: "card_btn_confirm_retry_title",
    ButtonIntent.SPEC_RESUME: "card_btn_confirm_retry_title",
    # Workflow
    WORKFLOW_CANCEL: "workflow_btn_confirm_cancel_title",
    # Approval
    ButtonIntent.APPROVE: "card_btn_confirm_approve_title",
    ButtonIntent.REJECT: "card_btn_confirm_reject_title",
}

# Import-time assertion: every key in _CONFIRM_TITLE_MAP must be a ButtonIntent
# member value or a registered action_id, preventing silent fallback bugs.
# Workflow action_ids are handled directly by WorkflowHandler (not via CardSession
# event pipeline), so they are not in INTENT_TO_ACTION_ID but are still valid keys.
_valid_keys = (
    {m.value for m in ButtonIntent}
    | set(INTENT_TO_ACTION_ID.values())
    | {
        WORKFLOW_CANCEL,
        WORKFLOW_CONFIRM_TOOLS,
        WORKFLOW_CONFIRM_START,
        WORKFLOW_SELECT_TOOL,
        WORKFLOW_REGENERATE_SCRIPT,
        WORKFLOW_ORCHESTRATOR_SELECT_TOOL,
        WORKFLOW_ORCHESTRATOR_SELECT_MODEL,
        WORKFLOW_ORCHESTRATOR_FINISH,
        WORKFLOW_REVIEW_SELECT_TOOL,
        WORKFLOW_REVIEW_SELECT_MODEL,
        WORKFLOW_REVIEW_FINISH,
        SHOW_WORKFLOW_MENU,
        WORKFLOW_LIST_TEMPLATES,
        WORKFLOW_SHOW_HELP,
    }
)
_invalid_keys = set(_CONFIRM_TITLE_MAP.keys()) - _valid_keys
if _invalid_keys:
    import warnings
    warnings.warn(
        f"_CONFIRM_TITLE_MAP contains keys that are neither ButtonIntent members "
        f"nor registered action_ids: {sorted(_invalid_keys)}",
        RuntimeWarning,
        stacklevel=2,
    )
del _invalid_keys

# Regex to match leading emoji sequences (emoji presentation + optional VS16 + ZWJ sequences)
_LEADING_EMOJI_RE = re.compile(
    r"^[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+\s*"
)


def _strip_leading_emoji(text: str) -> str:
    """Strip leading emoji characters and optional trailing space from text."""
    return _LEADING_EMOJI_RE.sub("", text)


def _get_confirm_title(action_id: str, button_text: str = "") -> str:
    """Get semantic confirm dialog title based on action_id white-list mapping.

    Uses exact match only — no substring fallback. Unregistered actions
    fall back to a generic template or default title.
    """
    ui_key = _CONFIRM_TITLE_MAP.get(action_id)
    if ui_key:
        return UI_TEXT.get(ui_key, UI_TEXT.get("card_btn_confirm_default_title", "确认操作？"))
    # Use button label as fallback context (strip leading emoji for cleanliness)
    if button_text:
        clean_text = _strip_leading_emoji(button_text)
        if clean_text:
            return UI_TEXT.get("btn_confirm_template", "确认「{text}」？").format(text=clean_text)
    return UI_TEXT.get("card_btn_confirm_default_title", "确认操作？")


def _resolve_action_id(spec: ButtonSpec) -> str | None:
    """Resolve the action_id from a ButtonSpec.

    If the action_id is a ButtonIntent (starts with 'intent.'), maps it
    via INTENT_TO_ACTION_ID. Otherwise uses the raw action_id string directly.

    Returns None if the intent is not registered (graceful degradation).
    """
    action_id = spec.action_id
    if action_id.startswith("intent."):
        resolved = INTENT_TO_ACTION_ID.get(action_id)
        if resolved is None:
            logger.warning("Unknown ButtonIntent '%s', rendering as disabled button", action_id)
        return resolved
    return action_id


def _is_stop_intent(spec: ButtonSpec) -> bool:
    """Check if a button represents a stop/cancel action."""
    return spec.action_id in _STOP_INTENTS


def _render_button(spec: ButtonSpec, *, engine_type: str | None = None, budget: RenderBudget | None = None) -> dict:
    """Render a single button element. Returns disabled button for unresolvable intents."""
    button_size = budget.button_size if budget else "medium"

    # URL button: opens a link instead of triggering a callback
    if spec.url:
        return {
            "tag": "button",
            "text": {"tag": "plain_text", "content": spec.text},
            "type": spec.type,
            "size": button_size,
            "behaviors": [{"type": "open_url", "default_url": spec.url}],
        }

    action_id = _resolve_action_id(spec)
    if action_id is None:
        # Graceful degradation: render as disabled button with no action
        btn: dict = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": spec.text},
            "type": "default",
            "size": button_size,
            "disabled": True,
            "disabled_tips": {
                "tag": "plain_text",
                "content": UI_TEXT["card_btn_disabled_tips"],
            },
        }
        return btn

    # Explicitly disabled button (e.g. STOPPING intermediate state)
    if spec.disabled:
        btn = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": spec.disabled_text or spec.text},
            "type": spec.type,
            "size": button_size,
            "disabled": True,
        }
        if spec.disabled_text:
            btn["disabled_tips"] = {"tag": "plain_text", "content": spec.disabled_text}
        return btn
    value: dict = dict(spec.value or {})
    value.setdefault("action", action_id)
    # Inject engine_type for ENGINE_STOP so the dispatcher can route correctly
    if action_id == ENGINE_STOP and engine_type:
        value["engine_type"] = engine_type
    # Prefix mode toggle actions with engine_type for correct prefix routing
    # (ws_client dispatcher expects "{engine_type}_mode_compact" / "{engine_type}_mode_full")
    if action_id in (MODE_FULL, MODE_COMPACT) and engine_type:
        value["action"] = f"{engine_type}_{action_id}"

    btn = {
        "tag": "button",
        "text": {"tag": "plain_text", "content": spec.text},
        "type": spec.type,
        "value": value,
        "behaviors": [{"type": "callback", "value": value}],
        "size": button_size,
    }

    if spec.confirm is not None:
        btn["confirm"] = {
            "title": {"tag": "plain_text", "content": _get_confirm_title(spec.action_id, spec.text)},
            "text": {"tag": "plain_text", "content": spec.confirm},
        }
    elif action_id in _DESTRUCTIVE_ACTIONS:
        btn["confirm"] = {
            "title": {"tag": "plain_text", "content": _get_confirm_title(spec.action_id, spec.text)},
            "text": {"tag": "plain_text", "content": UI_TEXT.get("card_btn_confirm_default_text", "此操作不可撤销，确认继续？")},
        }

    return btn


def render_buttons(state: CardState, budget: RenderBudget | None = None) -> list[dict]:
    """Generate button group elements.

    Layout rules:
    - No buttons → empty list
    - 1 button → column_set with flex_mode 'none' (full width for mobile accessibility)
    - 2 buttons → column_set with flex_mode 'bisect' (equal width)
    - ≥3 buttons with card_mobile_force_vertical → vertical column_set
    - 3 buttons otherwise → column_set with flex_mode 'none' (equal trisect)
    - >3 buttons → two-column column_set rows
    """
    if not state.buttons:
        return []

    engine_type = state.metadata.engine_type if state.metadata else None
    buttons = [_render_button(spec, engine_type=engine_type, budget=budget) for spec in state.buttons]

    elements = _layout_buttons(buttons, budget=budget)

    return elements


def _layout_buttons(buttons: list[dict], *, budget: RenderBudget | None = None) -> list[dict]:
    """Internal: arrange buttons into layout elements."""

    horizontal_spacing = budget.button_horizontal_spacing if budget and budget.button_horizontal_spacing else '8px'

    # Detect whether any button has a long label (>8 chars). Used as a
    # conservative heuristic to force vertical stacking on narrower widths
    # even when the caller did not set mobile_force_vertical=True.
    def _label_length(b: dict) -> int:
        text = b.get("text", {})
        if isinstance(text, dict):
            return len(text.get("content", ""))
        return 0

    has_long_label = any(_label_length(b) > 8 for b in buttons)

    mobile_force_vertical = bool(budget.mobile_force_vertical) if budget else False

    def _vertical_column(button: dict) -> dict:
        return {
            "tag": "column",
            "width": "weighted",
            "weight": 1,
            "elements": [button],
        }

    def _vertical_column_set(column: dict) -> dict:
        return {
            "tag": "column_set",
            "flex_mode": "none",
            "background_style": "default",
            "horizontal_spacing": horizontal_spacing,
            "columns": [column],
        }

    if len(buttons) == 1:
        # Single button: full width for mobile accessibility (Apple HIG)
        return [
            {
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "default",
                "horizontal_spacing": horizontal_spacing,
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": buttons,
                    },
                ],
            }
        ]

    if len(buttons) == 2:
        # Two buttons: vertical stack when mobile_force_vertical=True, so
        # long labels like "确认角色并生成脚本 →" stay readable on narrow
        # screens and the tap target stays wide enough.
        if mobile_force_vertical:
            rows: list[dict] = []
            for b in buttons:
                rows.append(_vertical_column_set(_vertical_column(b)))
            return rows

        columns = [
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "elements": [btn],
            }
            for btn in buttons
        ]
        return [
            {
                "tag": "column_set",
                "flex_mode": "bisect",
                "background_style": "default",
                "horizontal_spacing": horizontal_spacing,
                "columns": columns,
            }
        ]

    # ≥3 buttons: vertical stack when mobile_force_vertical=True, or when
    # any button label is too long to fit comfortably on a narrow screen.
    if mobile_force_vertical or (len(buttons) == 3 and has_long_label):
        return [
            {
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "default",
                "horizontal_spacing": horizontal_spacing,
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "vertical_align": "top",
                        "elements": buttons,
                    },
                ],
            }
        ]

    if len(buttons) == 3:
        # Three buttons: equal trisect via weighted columns
        columns = [
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "elements": [btn],
            }
            for btn in buttons
        ]
        return [
            {
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "default",
                "horizontal_spacing": horizontal_spacing,
                "columns": columns,
            }
        ]

    # >3 buttons: render as two-column rows to stay Schema 2.0 compatible
    # without creating oversized single-row button groups.
    rows: list[dict] = []
    for idx in range(0, len(buttons), 2):
        pair = buttons[idx:idx + 2]
        rows.append({
            "tag": "column_set",
            "flex_mode": "bisect" if len(pair) == 2 else "none",
            "background_style": "default",
            "horizontal_spacing": horizontal_spacing,
            "columns": [
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "elements": [btn],
                }
                for btn in pair
            ],
        })
    return rows


def build_responsive_button_row(
    buttons: list[dict],
    *,
    mobile_force_vertical: bool = False,
    horizontal_spacing: str = '8px',
    vertical_spacing: str = '8px',
) -> list[dict]:
    """Build a responsive button row layout from a list of button dicts.

    Replaces the deprecated ``{tag: 'action', actions: [...]}`` container with
    Schema 2.0 compliant ``column_set`` layouts. Handles 1, 2, 3, and >3 buttons
    with appropriate flex modes for mobile responsiveness.

    Args:
        buttons: List of button dicts (each must have tag='button' and
            contain value/behaviors fields).
        mobile_force_vertical: If True and there are >=3 buttons, stack them
            vertically in a single column (full width each) for better mobile
            experience. Defaults to False for backward compatibility.
        horizontal_spacing: Horizontal spacing between columns in a row.
            Defaults to '8px'.
        vertical_spacing: Vertical spacing for multi-row button groups.
            Defaults to '8px'.

    Returns:
        List of layout elements (column_set containers) ready to be appended
        to a card's elements array.
    """
    if not buttons:
        return []

    if len(buttons) == 1:
        return [
            {
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "default",
                "horizontal_spacing": horizontal_spacing,
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": buttons,
                    },
                ],
            }
        ]

    if len(buttons) == 2:
        if mobile_force_vertical:
            # Two long action buttons on narrow screens (e.g. "取消" +
            # "确认工具并自动执行 →") would truncate when bisected. Stack
            # them as separate column_sets so each button takes the full
            # width individually and the label stays readable.
            return [
                {
                    "tag": "column_set",
                    "flex_mode": "none",
                    "background_style": "default",
                    "horizontal_spacing": horizontal_spacing,
                    "columns": [
                        {
                            "tag": "column",
                            "width": "weighted",
                            "weight": 1,
                            "elements": [btn],
                        }
                    ],
                }
                for btn in buttons
            ]
        return [
            {
                "tag": "column_set",
                "flex_mode": "bisect",
                "background_style": "default",
                "horizontal_spacing": horizontal_spacing,
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": [btn],
                    }
                    for btn in buttons
                ],
            }
        ]

    # >=3 buttons: check for mobile force vertical stacking
    if mobile_force_vertical and len(buttons) >= 3:
        return [
            {
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "default",
                "horizontal_spacing": horizontal_spacing,
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "vertical_align": "top",
                        "elements": buttons,
                    },
                ],
            }
        ]

    if len(buttons) == 3:
        return [
            {
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "default",
                "horizontal_spacing": horizontal_spacing,
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": [btn],
                    }
                    for btn in buttons
                ],
            }
        ]

    # >3 buttons: two-column rows
    rows: list[dict] = []
    for idx in range(0, len(buttons), 2):
        pair = buttons[idx:idx + 2]
        rows.append({
            "tag": "column_set",
            "flex_mode": "bisect" if len(pair) == 2 else "none",
            "background_style": "default",
            "horizontal_spacing": horizontal_spacing,
            "columns": [
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "elements": [btn],
                }
                for btn in pair
            ],
        })
    return rows


def build_restart_button(engine_type: str, budget: RenderBudget | None = None) -> dict:
    """Build a '重新开始' action button for terminal/TTL-expired cards.

    The button triggers ENGINE_RESTART action which is routed by engine_type
    to re-send the appropriate engine command.
    """
    from src.card.actions.dispatch import ENGINE_RESTART

    label = ENGINE_LABELS.get(engine_type, ENGINE_LABEL_DEFAULT)
    button_size = budget.button_size if budget else "medium"
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": "default",
        "width": "default",
        "size": button_size,
        "confirm": {
            "title": {"tag": "plain_text", "content": UI_TEXT.get("card_btn_confirm_retry_title", "确认重新开始？")},
            "text": {"tag": "plain_text", "content": UI_TEXT.get("card_btn_confirm_default_text", "此操作不可撤销，确认继续？")},
        },
        "behaviors": [
            {
                "type": "callback",
                "value": {"action_id": ENGINE_RESTART, "engine_type": engine_type},
            }
        ],
    }
