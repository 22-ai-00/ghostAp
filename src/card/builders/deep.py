from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from ..models import EngineCardState
from ..shared import get_theme
from ..themes import ENGINE_STYLES
from .core import CoreBuilder

if TYPE_CHECKING:
    from src.project.context import ProjectContext


class DeepBuilder:
    """Static info-card builder for Deep/Spec/Diagnostics notifications."""

    @staticmethod
    def _resolve_style(engine_name: str) -> dict:
        name_lower = (engine_name or "").strip().lower()
        for key, style in ENGINE_STYLES.items():
            if key != "default" and name_lower.startswith(key):
                return style
        return ENGINE_STYLES["default"]

    @staticmethod
    def _pick_deep_template(
        engine_name: str, status: str = "running", terminal_state: str | None = None
    ) -> str:
        if terminal_state:
            terminal_colors = {
                "running": "blue",
                "completed": "green",
                "failed": "red",
                "cancelled": "orange",
                "blocked": "grey",
            }
            if terminal_state in terminal_colors:
                return terminal_colors[terminal_state]
        status_colors = {
            "error": "red",
            "completed": "green",
            "paused": "orange",
            "planning": "blue",
        }
        return status_colors.get(status.lower(), DeepBuilder._resolve_style(engine_name)["color"])

    @staticmethod
    def build_info_card(
        project: Optional[ProjectContext],
        state: EngineCardState,
    ) -> tuple[str, str]:
        style = DeepBuilder._resolve_style(state.engine_name)
        header_template = DeepBuilder._pick_deep_template(
            state.engine_name,
            terminal_state=getattr(state, "terminal_state", None),
        )
        theme = get_theme(header_template)

        icon = style["icon"]
        if state.title:
            header_title = state.title if icon in state.title else f"{icon} {state.title}"
        else:
            label = style.get("label_static") or style["label_format"].format(name=state.engine_name)
            header_title = f"{icon} {project.project_name} · {label}" if project else f"{icon} {label}"
        if not state.is_read:
            header_title = f"🔴 {header_title}"

        working_path = project.root_path if project else (state.working_dir or "~")
        display_content = state.content if isinstance(state.content, str) else str(state.content or "")
        elements: list[dict] = [
            {
                "tag": "markdown",
                "content": f"📁 `{working_path}`",
                "text_size": "notation",
            },
            {"tag": "hr"},
            {
                "tag": "markdown",
                "content": display_content,
                "text_size": "normal",
            },
        ]
        if state.footer_note:
            elements.append(
                {"tag": "markdown", "content": state.footer_note, "text_size": "notation"}
            )
        card = CoreBuilder._wrap_card(header_title, theme.header_template, elements, subtitle=state.subtitle)
        return "interactive", json.dumps(card, ensure_ascii=False)
