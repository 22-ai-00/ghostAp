"""Shared card element builders used by CardBuilder and renderers."""

from typing import Optional

from src.mode.manager import InteractionMode

from ..buttons_config import BUTTON_CONFIG
from ..themes import THEMES, ProjectTheme
from ..ui_text import UI_TEXT


def get_theme(color: str) -> ProjectTheme:
    return THEMES.get(color, THEMES["green"])


def apply_compact_style(
    button: dict,
    *,
    button_size: str = "medium",
    button_count: int | None = None,
    skip_compact: bool = False,
) -> dict:
    """Apply compact button styling (small size) for mobile friendliness."""
    if isinstance(button, dict) and button.get("tag") == "button":
        resolved_size = button_size or "medium"
        if not skip_compact and button_count is not None and button_count > 2:
            resolved_size = "small"
        button.setdefault("size", resolved_size)
    return button


def _create_mode_button(
    key: str,
    action: str,
    project_id: Optional[str] = None,
    thread_root_id: Optional[str] = None,
    *,
    button_size: str = "medium",
) -> dict:
    """Create a button from config with dynamic action value."""
    config = BUTTON_CONFIG.get(key)
    if not config:
        return {}

    value = {"action": action}
    if project_id:
        value["project_id"] = project_id
    if thread_root_id:
        value["thread_root_id"] = thread_root_id

    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": config["text"]},
        "type": config["type"],
        "size": button_size or "medium",
        "behaviors": [{"type": "callback", "value": value}],
    }


def build_mode_buttons(
    mode: Optional[InteractionMode] = None,
    project_id: Optional[str] = None,
    thread_root_id: Optional[str] = None,
    *,
    button_size: str = "medium",
) -> list[dict]:
    """Build mode-specific footer buttons (exit/enter mode + switch project)."""
    buttons = []

    if mode == InteractionMode.CLAUDE:
        buttons.append(_create_mode_button("exit_claude", "exit_claude", project_id, thread_root_id, button_size=button_size))
        buttons.append(_create_mode_button("switch_project", "switch_project", button_size=button_size))
    elif mode == InteractionMode.COCO:
        buttons.append(_create_mode_button("exit_coco", "exit_coco", project_id, thread_root_id, button_size=button_size))
        buttons.append(_create_mode_button("switch_project", "switch_project", button_size=button_size))
    elif mode == InteractionMode.GEMINI:
        buttons.append(_create_mode_button("exit_gemini", "exit_gemini", project_id, thread_root_id, button_size=button_size))
        buttons.append(_create_mode_button("switch_project", "switch_project", button_size=button_size))
    elif mode == InteractionMode.TRAEX:
        buttons.append(_create_mode_button("exit_traex", "exit_traex", project_id, thread_root_id, button_size=button_size))
        buttons.append(_create_mode_button("switch_project", "switch_project", button_size=button_size))
    else:
        buttons.append(_create_mode_button("enter_coco", "enter_coco", project_id, thread_root_id, button_size=button_size))
        buttons.append(_create_mode_button("enter_claude", "enter_claude", project_id, thread_root_id, button_size=button_size))
        buttons.append(_create_mode_button("enter_gemini", "enter_gemini", project_id, thread_root_id, button_size=button_size))
        buttons.append(_create_mode_button("enter_traex", "enter_traex", project_id, thread_root_id, button_size=button_size))

    return [b for b in buttons if b]


def build_responsive_layout(
    buttons: list[dict],
    *,
    layout: str = "responsive",
    mobile_force_vertical: bool = True,
    force_vertical: bool = False,
) -> list[dict]:
    """Build responsive button layout based on config setting.

    Parameters
    ----------
    buttons : list[dict]
        Button elements to lay out.
    layout : str
        Layout mode: "desktop", "mobile", "flow", or "responsive" (default).
    mobile_force_vertical : bool
        When True and layout is "responsive", force vertical for >2 buttons.

    - desktop: native action layout (horizontal)
    - mobile: forced vertical stack for better touch targets
    - responsive: <=2 buttons use action, >2 use column_set grid
    - flow: use flow layout
    """
    if not buttons:
        return []

    if force_vertical:
        return _build_button_vertical(buttons)

    effective_layout = (layout or "responsive").strip().lower()

    if effective_layout == "desktop":
        return _build_button_row_action(buttons)
    if effective_layout == "mobile":
        return _build_button_vertical(buttons)
    if effective_layout == "flow":
        return _build_button_flow(buttons)

    # responsive (default)

    # Force vertical for >2 buttons if mobile optimization is enabled
    # This helps avoid clutter on small screens while keeping 2 buttons side-by-side
    if mobile_force_vertical and len(buttons) > 2:
        return _build_button_vertical(buttons)

    if len(buttons) <= 2:
        return _build_button_row_action(buttons)
    return _build_button_grid(buttons)


def resolve_title_and_template(
    project_name: Optional[str],
    mode: Optional[InteractionMode] = None,
    theme_color: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve card title and header template based on mode and project name."""
    if mode == InteractionMode.CLAUDE:
        mode_icon, header_template = "🔮", "purple"
    elif mode == InteractionMode.COCO:
        mode_icon, header_template = "🤖", "blue"
    elif mode == InteractionMode.GEMINI:
        mode_icon, header_template = "✨", "turquoise"
    elif mode == InteractionMode.TRAEX:
        mode_icon, header_template = "🚀", "blue"
    else:
        mode_icon, header_template = "🧠", "turquoise"

    # If a theme_color is provided (from project), use it for the template
    if theme_color and mode not in [InteractionMode.CLAUDE, InteractionMode.COCO, InteractionMode.GEMINI, InteractionMode.TRAEX]:
        header_template = get_theme(theme_color).header_template

    if project_name:
        if mode == InteractionMode.CLAUDE:
            title = f"🔮 {project_name} · Claude"
        elif mode == InteractionMode.COCO:
            title = f"🤖 {project_name} · Coco"
        elif mode == InteractionMode.GEMINI:
            title = f"✨ {project_name} · Gemini"
        elif mode == InteractionMode.TRAEX:
            title = f"🚀 {project_name} · Traex"
        else:
            title = f"🧠 {project_name}"
    else:
        if mode == InteractionMode.CLAUDE:
            mode_name = UI_TEXT["mode_name_claude"]
        elif mode == InteractionMode.COCO:
            mode_name = UI_TEXT["mode_name_coco"]
        elif mode == InteractionMode.GEMINI:
            mode_name = UI_TEXT["mode_name_gemini"]
        elif mode == InteractionMode.TRAEX:
            mode_name = UI_TEXT["mode_name_traex"]
        else:
            mode_name = UI_TEXT["mode_name_smart"]
        title = f"{mode_icon} {mode_name}"

    return title, header_template

# ---- Internal helpers ----


def _build_button_row_action(buttons: list[dict]) -> list[dict]:
    """Build button row using column_set (schema 2.0 compatible).

    Schema 2.0 does not support the 'action' tag; use column_set grid instead.
    """
    if not buttons:
        return []
    return _build_button_grid(buttons, columns=len(buttons))


def _build_button_grid(buttons: list[dict], columns: int = 2) -> list[dict]:
    """Build two-column grid layout for buttons."""
    if not buttons:
        return []
    if columns <= 0:
        columns = 2

    styled = [apply_compact_style(b, button_count=len(buttons)) for b in buttons]
    rows: list[dict] = []

    for i in range(0, len(styled), columns):
        chunk = styled[i : i + columns]

        column_objs = []
        for j in range(columns):
            btn = chunk[j] if j < len(chunk) else None
            # Only create column if it has content OR if we need to preserve grid spacing?
            # Feishu schema 2.0: empty columns are allowed.
            column_objs.append(
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "elements": [btn] if btn else [],
                }
            )

        rows.append(
            {
                "tag": "column_set",
                "flex_mode": "stretch",
                "background_style": "default",
                "columns": column_objs,
            }
        )

    return rows


def build_quick_actions(actions: list[str], context: dict = None) -> list[dict]:
    """Build standardized QuickAction buttons.

    Supported actions: confirm, retry, cancel, continue, next, stop
    """
    buttons = []
    context = context or {}

    for action in actions:
        action = action.lower()
        if action == "confirm":
            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["qa_btn_confirm"]},
                    "type": "primary",
                    "value": {"action": "confirm", **context},
                }
            )
        elif action == "retry":
            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["qa_btn_retry"]},
                    "type": "primary",
                    "value": {"action": "retry_command", **context},
                }
            )
        elif action == "cancel":
            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["qa_btn_cancel"]},
                    "type": "default",
                    "value": {"action": "cancel", **context},
                }
            )
        elif action == "continue":
            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["qa_btn_continue"]},
                    "type": "primary",
                    "value": {"action": "continue", **context},
                }
            )
        elif action == "next":
            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["qa_btn_next"]},
                    "type": "primary",
                    "value": {"action": "next", **context},
                }
            )
        elif action == "stop":
            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["qa_btn_stop"]},
                    "type": "danger",
                    "value": {"action": "stop", **context},
                }
            )
        elif action == "new_project_prompt":
            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["qa_btn_new_project"]},
                    "type": "primary",
                    "value": {"action": "new_project_prompt", **context},
                }
            )
        elif action == "list_projects":
            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["qa_btn_list_projects"]},
                    "type": "default",
                    "value": {"action": "show_board", **context},
                }
            )

    return [apply_compact_style(b) for b in buttons]


def _build_button_vertical(buttons: list[dict]) -> list[dict]:
    """Build vertical stack layout for buttons (best for mobile)."""
    if not buttons:
        return []

    styled = [apply_compact_style(b, button_count=len(buttons)) for b in buttons]
    rows = []
    for btn in styled:
        rows.append(
            {
                "tag": "column_set",
                "flex_mode": "none",
                "columns": [{"tag": "column", "width": "weighted", "weight": 1, "elements": [btn]}],
            }
        )
    return rows


def _build_button_flow(buttons: list[dict]) -> list[dict]:
    """Build flow layout for buttons."""
    if not buttons:
        return []

    styled = [apply_compact_style(b, button_count=len(buttons)) for b in buttons]

    return [
        {
            "tag": "column_set",
            "flex_mode": "flow",
            "columns": [{"tag": "column", "width": "auto", "elements": [btn]} for btn in styled],
        }
    ]
