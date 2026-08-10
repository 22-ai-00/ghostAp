from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from ..buttons_config import BUTTON_CONFIG
from ..models import EngineCardState
from ..shared import (
    apply_compact_style,
    build_responsive_layout,
    get_theme,
)
from ..themes import ENGINE_STYLES
from ..thresholds import THRESHOLDS
from ..ui_text import UI_TEXT
from .core import CoreBuilder

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.project.context import ProjectContext


class DeepBuilder:
    """Deep engine card building utilities."""

    @staticmethod
    def _resolve_style(engine_name: str) -> dict:
        name_lower = (engine_name or "").strip().lower()
        for key, style in ENGINE_STYLES.items():
            if key != "default" and name_lower.startswith(key):
                return style
        return ENGINE_STYLES["default"]

    @staticmethod
    def _build_deep_header_title(
        project: Optional[ProjectContext],
        engine_name: str = "Coco",
    ) -> str:
        style = DeepBuilder._resolve_style(engine_name)
        icon = style["icon"]

        if "label_static" in style:
            label = style["label_static"]
        else:
            label = style["label_format"].format(name=engine_name)

        if project:
            return f"{icon} {project.project_name} · {label}"
        return f"{icon} {label}"

    @staticmethod
    def _pick_deep_template(engine_name: str, status: str = "running", terminal_state: str | None = None) -> str:
        """Pick header template color based on engine, status, and optional terminal_state."""
        # terminal_state takes priority when present
        if terminal_state:
            TERMINAL_COLORS = {
                "running": "blue",
                "completed": "green",
                "failed": "red",
                "cancelled": "orange",
                "blocked": "grey",
            }
            if terminal_state in TERMINAL_COLORS:
                return TERMINAL_COLORS[terminal_state]

        status = status.lower()
        STATUS_COLORS = {
            "error": "red",
            "completed": "green",
            "paused": "orange",
            "planning": "blue",
        }
        if status in STATUS_COLORS:
            return STATUS_COLORS[status]

        style = DeepBuilder._resolve_style(engine_name)
        return style["color"]

    @staticmethod
    def _create_button(action_key: str, state: EngineCardState, action_suffix: str = None) -> dict:
        """Helper to create a button based on config and state."""
        config = BUTTON_CONFIG.get(action_key)
        if not config:
            return {}

        # Determine the engine-prefixed action string.
        if action_suffix:
            action_name = f"{state.action_prefix}_{action_suffix}"
        else:
            action_name = f"{state.action_prefix}_{action_key}"

        btn = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": config["text"]},
            "type": config["type"],
            "value": {
                "action": action_name,
                "project_id": state.project_id or state.engine_project_id,
                "deep_project_id": state.engine_project_id,
            },
        }

        # Add confirm dialog for danger buttons (only stop_danger requires confirmation)
        if action_key == "stop_danger":
            engine_cmd = f"/{state.action_prefix}" if state.action_prefix else "/deep"
            btn["confirm"] = {
                "title": {"tag": "plain_text", "content": UI_TEXT["card_btn_confirm_stop_title_danger"]},
                "text": {"tag": "plain_text", "content": UI_TEXT["card_btn_confirm_stop_danger_body"].format(engine_cmd=engine_cmd)},
            }

        return btn

    @staticmethod
    def _build_deep_buttons(state: EngineCardState) -> list[dict]:
        """Build a flat list of deep engine buttons (for backward compatibility with tests)."""
        buttons = []

        # Control buttons
        if state.is_executing:
            if state.action_prefix != "deep":
                buttons.append(DeepBuilder._create_button("pause", state))
            buttons.append(DeepBuilder._create_button("stop", state))
        elif state.is_paused:
            if state.action_prefix != "deep":
                buttons.append(DeepBuilder._create_button("resume", state))
            buttons.append(DeepBuilder._create_button("stop", state))

        # View buttons
        lines = (state.content or "").split("\n")
        threshold = THRESHOLDS["COMPACT_LINE_THRESHOLD"] if state.compact else THRESHOLDS["FULL_LINE_THRESHOLD"]
        if len(lines) > threshold:
            action_key = "collapse" if state.expanded else "expand"
            buttons.append(DeepBuilder._create_button(action_key, state))

        # Mode switch
        mode_key = "mode_full" if state.compact else "mode_compact"
        buttons.append(DeepBuilder._create_button(mode_key, state))

        # Feature-specific buttons
        style = DeepBuilder._resolve_style(state.engine_name)
        features = style.get("features", {})
        if features.get("history_button"):
            buttons.append(DeepBuilder._create_button("history", state))

        # Extra buttons (custom actions like retry)
        if state.extra_buttons:
            for b in state.extra_buttons:
                if b:
                    buttons.append(b)

        return [apply_compact_style(b) for b in buttons if b]

    @staticmethod
    def _build_grouped_layout(state: EngineCardState) -> list[dict]:
        """Build layout with grouped control buttons (Pause/Stop side-by-side)."""
        control_buttons = []
        other_buttons = []
        style = DeepBuilder._resolve_style(state.engine_name)
        features = style.get("features", {})

        # 1. Collect Control Buttons (Pause/Resume, Stop)
        if state.is_executing:
            if state.action_prefix != "deep":
                control_buttons.append(DeepBuilder._create_button("pause", state))
            control_buttons.append(DeepBuilder._create_button("stop_danger", state, action_suffix="stop"))
        elif state.is_paused:
            if state.action_prefix != "deep":
                control_buttons.append(DeepBuilder._create_button("resume", state))
            control_buttons.append(DeepBuilder._create_button("stop_danger", state, action_suffix="stop"))

        # 2. Collect View/Mode Buttons
        # Only show these auxiliary buttons if we don't have high-priority extra buttons (like Retry)
        # This keeps the user focused on the recovery action when an error occurs
        if not state.extra_buttons:
            # Log Expand/Collapse
            lines = (state.content or "").split("\n")
            threshold = THRESHOLDS["COMPACT_LINE_THRESHOLD"] if state.compact else THRESHOLDS["FULL_LINE_THRESHOLD"]
            if len(lines) > threshold:
                action_key = "collapse" if state.expanded else "expand"
                other_buttons.append(DeepBuilder._create_button(action_key, state))

            # AC Expand/Collapse
            if state.criteria_section:
                ac_lines = state.criteria_section.split("\n")
                if len(ac_lines) > THRESHOLDS["AC_LINE_THRESHOLD"]:
                    ac_action_key = "collapse_ac" if state.expand_ac else "expand_ac"
                    other_buttons.append(DeepBuilder._create_button(ac_action_key, state))

            # Mode Switch
            mode_key = "mode_full" if state.compact else "mode_compact"
            other_buttons.append(DeepBuilder._create_button(mode_key, state))

        # Feature-specific Buttons (History)
        if features.get("history_button"):
            other_buttons.append(DeepBuilder._create_button("history", state))

        # Extra buttons (custom actions like retry) - priority high
        if state.extra_buttons:
            for b in reversed(state.extra_buttons): # Insert at beginning of other_buttons
                if b:
                    other_buttons.insert(0, b)

        # 3. Build Layout Elements
        elements = []

        # Apply styling
        control_buttons = [apply_compact_style(b) for b in control_buttons if b]
        other_buttons = [apply_compact_style(b) for b in other_buttons if b]

        # Group controls in a specific column set (force side-by-side)
        if control_buttons:
            cols = []
            for btn in control_buttons:
                cols.append({"tag": "column", "width": "weighted", "weight": 1, "elements": [btn]})
            elements.append(
                {"tag": "column_set", "flex_mode": "stretch", "background_style": "default", "columns": cols}
            )

        # Group others using standard responsive layout
        if other_buttons:
            elements.extend(build_responsive_layout(other_buttons))

        return elements

    @staticmethod
    def build_info_card(
        project: Optional[ProjectContext],
        state: EngineCardState,
    ) -> tuple[str, str]:
        from ..models import CardLayoutSpec
        from .layout import UnifiedCardLayout

        # ---- 1. 状态判断（用于颜色映射） ----
        status_key = "running"
        title_lower = state.title.lower()
        if "error" in title_lower or UI_TEXT["deep_status_failed_zh"] in state.title:
            status_key = "error"
        elif (
            UI_TEXT["deep_status_completed_zh"] in state.title
            or UI_TEXT["deep_status_finished_zh"] in state.title
            or "completed" in title_lower
            or "finished" in title_lower
            or "success" in title_lower
        ):
            status_key = "completed"
        elif state.is_paused:
            status_key = "paused"
        elif (
            UI_TEXT["deep_status_planning_zh"] in state.title
            or UI_TEXT["deep_status_analyzing_zh"] in state.title
            or "planning" in title_lower
            or "analyzing" in title_lower
        ):
            status_key = "planning"

        header_template = DeepBuilder._pick_deep_template(state.engine_name, status_key, terminal_state=getattr(state, 'terminal_state', None))
        theme = get_theme(header_template)

        # ---- 2. 标题 ----
        if not state.title:
            header_title = DeepBuilder._build_deep_header_title(project, state.engine_name)
        else:
            style = DeepBuilder._resolve_style(state.engine_name)
            icon = style["icon"]
            if icon not in state.title:
                header_title = f"{icon} {state.title}"
            else:
                header_title = state.title

        # 未读标记
        if not state.is_read:
            header_title = f"🔴 {header_title}"

        # ---- 3. 内容截断（compact/expanded/full 模式） ----
        display_content = state.content if isinstance(state.content, str) else str(state.content or "")

        if state.expanded:
            pass  # 全展开，不截断
        elif state.compact:
            is_error = status_key == "error"
            if is_error:
                if not display_content:
                    if state.action_prefix and state.action_prefix.strip():
                        display_content = UI_TEXT["deep_error_no_detail"].format(engine_cmd=f"/{state.action_prefix}")
                    else:
                        display_content = UI_TEXT["deep_error_fallback_no_prefix"]
                else:
                    lines = display_content.split("\n")
                    if len(lines) > THRESHOLDS["COMPACT_LINE_THRESHOLD"]:
                        display_content = "\n".join(lines[:THRESHOLDS["COMPACT_LINE_THRESHOLD"]]) + UI_TEXT["deep_error_expand_hint"]
            else:
                if not display_content:
                    display_content = UI_TEXT["deep_executing_placeholder"]
                else:
                    lines = display_content.split("\n")
                    if len(lines) > THRESHOLDS["COMPACT_LINE_THRESHOLD"]:
                        display_content = "…\n" + "\n".join(lines[-THRESHOLDS["COMPACT_LINE_THRESHOLD"]:]) + UI_TEXT["deep_content_expand_hint"]
                    elif len(display_content) > THRESHOLDS["COMPACT_CHAR_FALLBACK"]:
                        display_content = "…" + display_content[-THRESHOLDS["COMPACT_CHAR_FALLBACK"]:]
        else:
            if display_content:
                lines = display_content.split("\n")
                if len(lines) > THRESHOLDS["FULL_LINE_THRESHOLD"]:
                    display_content = UI_TEXT["deep_folded_lines_hint"].format(count=len(lines) - THRESHOLDS["FULL_LINE_THRESHOLD"]) + "\n".join(
                        lines[-THRESHOLDS["FULL_LINE_THRESHOLD"]:]
                    )

        # ---- 4. 验收标准截断 ----
        criteria = None
        if state.criteria_section and not state.compact:
            criteria = state.criteria_section
            if not state.expand_ac:
                ac_lines = criteria.split("\n")
                if len(ac_lines) > THRESHOLDS["AC_LINE_THRESHOLD"]:
                    criteria = "\n".join(ac_lines[:THRESHOLDS["AC_LINE_THRESHOLD"]]) + UI_TEXT["deep_ac_expand_hint"]

        # ---- 5. 按钮构建（保留 grouped layout 逻辑） ----
        btn_elements: list[dict] | None = None
        if state.show_buttons:
            if state.is_executing or state.is_paused:
                btn_elements = DeepBuilder._build_grouped_layout(state)
            else:
                buttons = []
                from src.mode.manager import InteractionMode
                base_buttons = CoreBuilder._build_footer_buttons(project, mode=InteractionMode.SMART)

                if state.extra_buttons:
                    for b in state.extra_buttons:
                        if b:
                            buttons.append(apply_compact_style(b))

                if not state.extra_buttons:
                    lines = (state.content or "").split("\n")
                    threshold = THRESHOLDS["COMPACT_LINE_THRESHOLD"] if state.compact else THRESHOLDS["FULL_LINE_THRESHOLD"]
                    if len(lines) > threshold:
                        action_key = "collapse" if state.expanded else "expand"
                        buttons.append(apply_compact_style(DeepBuilder._create_button(action_key, state)))

                    mode_key = "mode_full" if state.compact else "mode_compact"
                    mode_btn = apply_compact_style(DeepBuilder._create_button(mode_key, state))
                    if mode_btn:
                        buttons.append(mode_btn)

                buttons.extend(base_buttons)
                if buttons:
                    btn_elements = build_responsive_layout(buttons)

        # ---- 6. 通过 UnifiedCardLayout 组装 ----
        # 解析引擎的 meta_separator
        style = DeepBuilder._resolve_style(state.engine_name)
        meta_sep = style.get("meta_separator", " · ")

        # 项目路径
        working_path = None
        if project:
            working_path = project.root_path
        elif state.working_dir:
            working_path = state.working_dir

        # 结构化内容（折叠面板）：非 compact 模式下，如果有 rendered_content，
        # 使用 to_elements(collapsible=True) 替代纯 markdown
        content_elements = None
        use_content_markdown = display_content
        if state.rendered_content is not None and not state.compact:
            try:
                els = state.rendered_content.to_elements(collapsible=True)
                if els:  # 仅当结构化内容非空时才替代 markdown
                    content_elements = els
                    use_content_markdown = None
                # else: 保留 markdown 内容作为 fallback
            except Exception:
                logger.debug("failed to render structured content, falling back to markdown", exc_info=True)

        spec = CardLayoutSpec(
            project_path=working_path,
            progress_bar=state.progress_bar,
            status_line=state.status_line,
            duration_line=state.duration_line,
            engine_meta_separator=meta_sep,
            warning_banner=state.warning_banner,
            content_markdown=use_content_markdown,
            content_elements=content_elements,
            criteria_section=criteria,
            footer_note=state.footer_note,
            button_elements=btn_elements,
            footer_status=state.footer_status,
            terminal_state=state.terminal_state,
        )
        elements = UnifiedCardLayout.build(spec)

        card = CoreBuilder._wrap_card(header_title, theme.header_template, elements, subtitle=state.subtitle)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_history_list_card(
        project: Optional[ProjectContext],
        title: str,
        content: str,
        history_buttons: list[dict],
        page: int,
        has_next: bool,
        deep_project_id: Optional[str] = None,
        engine_name: str = "Coco",
    ) -> tuple[str, str]:
        """Build a history list card with pagination."""
        header_template = DeepBuilder._pick_deep_template(engine_name, "running")
        theme = get_theme(header_template)

        # Consistent title using helper (or similar logic)
        header_title = f"📜 {project.project_name if project else engine_name}{UI_TEXT['system_history_record_title']}"

        elements = [
            {"tag": "markdown", "content": f"**{title}**\n\n{content}"},
            {"tag": "hr"},
        ]

        # History Items (as buttons grid)
        elements.extend(build_responsive_layout(history_buttons))

        # Pagination Controls
        nav_buttons = []
        if page > 1:
            nav_buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["system_btn_prev_page"]},
                    "type": "default",
                    "value": {
                        "action": "history_page",
                        "page": page - 1,
                        "project_id": project.project_id if project else None,
                        "deep_project_id": deep_project_id,
                    },
                }
            )

        if has_next:
            nav_buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["system_btn_next_page"]},
                    "type": "default",
                    "value": {
                        "action": "history_page",
                        "page": page + 1,
                        "project_id": project.project_id if project else None,
                        "deep_project_id": deep_project_id,
                    },
                }
            )

        # Back to Status
        nav_buttons.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": UI_TEXT["system_btn_back_status"]},
                "type": "primary",
                "value": {
                    "action": "back_to_list",
                    "project_id": project.project_id if project else None,
                    "deep_project_id": deep_project_id,
                },
            }
        )

        if nav_buttons:
            elements.append({"tag": "hr"})
            elements.extend(build_responsive_layout([apply_compact_style(b) for b in nav_buttons]))

        card = CoreBuilder._wrap_card(header_title, theme.header_template, elements)
        return "interactive", json.dumps(card, ensure_ascii=False)
