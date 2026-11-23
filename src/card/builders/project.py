from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote

from src.mode.manager import InteractionMode
from src.project.context import (
    get_project_acp_model_name,
    get_project_acp_tool_name,
    get_project_programming_mode,
    get_project_programming_snapshot,
)

from ..backend_meta import ALL_BACKEND_METAS, get_backend_meta, resolve_project_programming_mode
from ..shared import (
    apply_compact_style,
    build_callback_button,
    build_responsive_layout,
    get_theme,
)
from ..ui_text import UI_TEXT
from .core import CoreBuilder

if TYPE_CHECKING:
    from src.project.context import ProjectContext


class ProjectBuilder:
    """Project-related card building utilities."""

    @staticmethod
    def _build_project_chat_multi_url(chat_id: str) -> dict:
        safe_chat_id = quote(str(chat_id or "").strip(), safe="")
        https = f"https://applink.feishu.cn/client/chat/open?openChatId={safe_chat_id}"
        native = f"lark://applink/client/chat/open?openChatId={safe_chat_id}"
        return {
            "url": https,
            "pc_url": https,
            "android_url": native,
            "ios_url": native,
        }

    @staticmethod
    def _build_project_group_jump_button(project: Optional[ProjectContext]) -> Optional[dict]:
        chat_id = str(getattr(project, "bound_chat_id", "") or "").strip() if project else ""
        if not chat_id:
            return None
        return apply_compact_style(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": UI_TEXT["project_btn_open_group"]},
                "type": "default",
                "multi_url": ProjectBuilder._build_project_chat_multi_url(chat_id),
            }
        )


    @staticmethod
    def build_project_info_content(project: ProjectContext, global_working_dir: str) -> str:
        """Build the Markdown content for project info."""
        lines = [UI_TEXT["project_current_header"].format(name=project.project_name), ""]

        lines.append(UI_TEXT["project_id_label"].format(id=project.project_id))
        lines.append(UI_TEXT["project_dir_label"].format(path=project.root_path))
        lines.append(UI_TEXT["project_dir_info_cwd"].format(cwd=global_working_dir))
        lines.append(
            UI_TEXT["project_status_label"].format(
                emoji=project.get_status_emoji(), status=project.status.value
            )
        )
        lines.append(
            UI_TEXT["project_coco_status"].format(
                status=f"🤖 {UI_TEXT['system_on']}" if project.coco_mode else UI_TEXT["system_off"]
            )
        )
        lines.append(
            UI_TEXT["project_claude_status"].format(
                status=f"🔮 {UI_TEXT['system_on']}" if project.claude_mode else UI_TEXT["system_off"]
            )
        )

        return "\n".join(lines)

    @staticmethod
    def build_current_project_card(project: ProjectContext, global_working_dir: str) -> tuple[str, str]:
        """Build the full card for current project info."""
        content = ProjectBuilder.build_project_info_content(project, global_working_dir)
        return ProjectBuilder.build_project_response_card(
            project, UI_TEXT["project_info_card_title"], content, show_buttons=True
        )

    @staticmethod
    def build_project_status_report_card(
        project: ProjectContext,
        global_working_dir: str,
        *,
        chat_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build the full card for project status report."""
        del global_working_dir  # status card keeps one directory row for compactness

        theme = get_theme(project.theme_color)
        status = UI_TEXT["project_status_label"].format(
            emoji=project.get_status_emoji(), status=project.status.value
        )
        last_active = UI_TEXT["project_last_active_label"].format(
            time_ago=CoreBuilder._format_time_ago(project.last_active)
        )
        group_chat_id = str(getattr(project, "bound_chat_id", "") or "").strip()
        group_name = str(getattr(project, "bound_chat_name", "") or "").strip()
        group_line = (
            UI_TEXT["project_status_group_label"].format(name=group_name or group_chat_id)
            if group_chat_id
            else UI_TEXT["project_status_no_group"]
        )

        lines = [
            f"**{UI_TEXT['project_status_card_title']}**",
            status,
            last_active,
            group_line,
        ]

        for meta in ALL_BACKEND_METAS:
            if get_project_programming_mode(project, meta.mode_key, chat_id):
                snap = get_project_programming_snapshot(project, meta.mode_key, chat_id)
                if snap:
                    lines.append(
                        f"• {meta.display_name}: `{snap.session_id}` · {UI_TEXT['project_session_count_label'].format(count=snap.query_count).lstrip('• ')}"
                    )

        buttons = [
            apply_compact_style(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["project_board_btn_switch"]},
                    "type": "default",
                    "behaviors": [
                        {
                            "type": "callback",
                            "value": {"action": "switch_project", "project_id": project.project_id},
                        }
                    ],
                }
            )
        ]
        group_button = ProjectBuilder._build_project_group_jump_button(project)
        if group_button:
            buttons.append(group_button)

        elements = [
            CoreBuilder._build_directory_element(project),
            {"tag": "hr"},
            {"tag": "markdown", "content": "\n".join(lines)},
        ]
        elements.extend(build_responsive_layout(buttons))

        header_title = CoreBuilder._build_header_title(project)
        card = CoreBuilder._wrap_card(header_title, theme.header_template, elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_project_not_found_content(name: str, suggestions: Optional[list[ProjectContext]] = None) -> str:
        """Build the Markdown content for project not found error."""
        header = UI_TEXT["project_not_found"].format(name=name)
        content = f"{header}"
        if suggestions:
            similar_header = UI_TEXT["project_similar_header"]
            suggestion_lines = "\n".join([f"• {p.project_name}" for p in suggestions[:5]])
            content += f"{similar_header}\n{suggestion_lines}"
        return content

    @staticmethod
    def build_project_response_card(
        project: Optional[ProjectContext],
        title: str,
        content: str,
        show_buttons: bool = True,
        extra_buttons: Optional[list[dict]] = None,
        footer: Optional[str] = None,
        image_keys: Optional[list[str]] = None,
        banner: Optional[dict] = None,
        working_dir: Optional[str] = None,
    ) -> tuple[str, str]:
        effective_mode = resolve_project_programming_mode(project)

        return ProjectBuilder._build_response_card_inner(
            project,
            title,
            content,
            working_dir=working_dir,
            show_buttons=show_buttons,
            mode=effective_mode,
            extra_buttons=extra_buttons,
            footer=footer,
            image_keys=image_keys,
            banner=banner,
        )

    @staticmethod
    def build_smart_response_card(
        project: Optional[ProjectContext],
        title: str,
        content: str,
        working_dir: Optional[str] = None,
        show_buttons: bool = True,
    ) -> tuple[str, str]:
        return ProjectBuilder._build_response_card_inner(
            project,
            title,
            content,
            working_dir,
            show_buttons,
            mode=InteractionMode.SMART,
        )

    @staticmethod
    def _build_response_card_inner(
        project: Optional[ProjectContext],
        title: str,
        content: str,
        working_dir: Optional[str] = None,
        show_buttons: bool = True,
        mode: Optional[InteractionMode] = None,
        extra_buttons: Optional[list[dict]] = None,
        footer: Optional[str] = None,
        image_keys: Optional[list[str]] = None,
        banner: Optional[dict] = None,
    ) -> tuple[str, str]:
        effective_mode = mode

        theme_color = getattr(project, "theme_color", None) if project else None
        if not theme_color:
            theme_color = "turquoise" if effective_mode == InteractionMode.GEMINI else "blue"
        theme = get_theme(theme_color)

        header_title = CoreBuilder._build_header_title(
            project,
            mode=effective_mode,
        )

        elements = [
            CoreBuilder._build_directory_element(project, working_dir),
            {"tag": "hr"},
        ]

        if banner:
            elements.append(banner)
            elements.append({"tag": "hr"})

        if image_keys:
            elements.extend(CoreBuilder._build_image_elements(image_keys))
            elements.append({"tag": "hr"})

        elements.append(CoreBuilder._build_content_element(content, title))

        if footer:
            elements.append(
                {
                    "tag": "markdown",
                    "content": footer,
                    "text_size": "notation",
                }
            )
        else:
            footer_note = CoreBuilder._build_footer_note(project, working_dir)
            if footer_note:
                elements.append(footer_note)

        if show_buttons:
            buttons = CoreBuilder._build_footer_buttons(
                project,
                mode=effective_mode,
            )
            if extra_buttons:
                buttons.extend([apply_compact_style(b) for b in extra_buttons])
            elements.extend(build_responsive_layout(buttons))
        elif extra_buttons:
            # Render extra buttons without the standard footer buttons
            elements.extend(build_responsive_layout(
                [apply_compact_style(b) for b in extra_buttons]
            ))

        card = CoreBuilder._wrap_card(header_title, theme.header_template, elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_status_board_card(
        projects: list[ProjectContext],
        current_project_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 5,
        *,
        chat_id: Optional[str] = None,
    ) -> tuple[str, str]:
        board_title = UI_TEXT["project_board_title"]
        if not projects:
            empty_elements = [
                {"tag": "markdown", "content": UI_TEXT["project_board_empty_content"]},
                *build_responsive_layout(
                    [
                        build_callback_button(
                            UI_TEXT["project_board_btn_new"],
                            {"action": "new_project_prompt"},
                            button_type="primary",
                            compact=True,
                        )
                    ]
                ),
            ]
            card = CoreBuilder._wrap_card(board_title, "blue", empty_elements)
            return "interactive", json.dumps(card, ensure_ascii=False)

        # Sort: Current project first, then by last_active descending
        def _sort_key(p):
            is_cur = p.project_id == current_project_id
            return (not is_cur, -(p.last_active or 0))

        sorted_projects = sorted(projects, key=_sort_key)
        total_projects = len(sorted_projects)
        total_pages = (total_projects + page_size - 1) // page_size

        # Clamp page
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_projects = sorted_projects[start_idx:end_idx]

        elements = [{"tag": "markdown", "content": UI_TEXT["project_board_total_projects"].format(total=total_projects)}, {"tag": "hr"}]

        for project in page_projects:
            is_current = project.project_id == current_project_id
            status_emoji = project.get_status_emoji()
            current_marker = UI_TEXT["project_board_current_marker"] if is_current else ""

            mode_info = ""
            active_mode = resolve_project_programming_mode(project, chat_id=chat_id)
            if active_mode is not None:
                meta = get_backend_meta(active_mode)
                query_count = 0
                if meta is not None:
                    snap = get_project_programming_snapshot(project, meta.mode_key, chat_id)
                    if snap:
                        query_count = getattr(snap, "query_count", 0) or 0
                if active_mode == InteractionMode.CLAUDE:
                    mode_info = UI_TEXT["project_board_claude_info"].format(count=query_count)
                elif active_mode == InteractionMode.COCO:
                    mode_info = UI_TEXT["project_board_coco_info"].format(count=query_count)
                elif meta is not None:
                    mode_info = f" · {meta.emoji} {meta.display_name}({query_count})"
            else:
                saved_tool = get_project_acp_tool_name(project, chat_id)
                meta = get_backend_meta(saved_tool) if saved_tool else None
                if meta is not None:
                    saved_model = get_project_acp_model_name(
                        project,
                        chat_id,
                        tool_name=saved_tool,
                    )
                    model_suffix = f"/{saved_model}" if saved_model else ""
                    mode_info = f" · {meta.emoji} {meta.display_name}{model_suffix}"

            last_active = CoreBuilder._format_time_ago(project.last_active)

            project_content = (
                f"{status_emoji} **{project.project_name}**{current_marker}\n"
                f"└─ 📁 `{project.root_path}`\n"
                f"└─ ⏱️ {last_active}{mode_info}"
            )

            elements.append({"tag": "markdown", "content": project_content})

            buttons = []
            if not is_current:
                buttons.append(
                    build_callback_button(
                        UI_TEXT["project_board_btn_switch"],
                        {"action": "switch_to", "project_id": project.project_id},
                        button_type="primary",
                    )
                )
            else:
                buttons.append(
                    build_callback_button(
                        UI_TEXT["project_board_btn_continue"],
                        {"action": "continue_dev", "project_id": project.project_id},
                        button_type="primary",
                    )
                )

            buttons.append(
                build_callback_button(
                    UI_TEXT["project_board_btn_detail"],
                    {"action": "show_detail", "project_id": project.project_id},
                )
            )
            group_button = ProjectBuilder._build_project_group_jump_button(project)
            if group_button:
                buttons.append(group_button)

            elements.extend(build_responsive_layout(buttons))
            elements.append({"tag": "hr"})

        # Remove last hr
        if elements and elements[-1].get("tag") == "hr":
            elements.pop()

        # Pagination & Global Actions
        pagination_buttons = []
        if page > 1:
            pagination_buttons.append(
                build_callback_button(
                    UI_TEXT["project_board_btn_prev"],
                    {"action": "switch_board_page", "page": page - 1},
                )
            )

        if page < total_pages:
            pagination_buttons.append(
                build_callback_button(
                    UI_TEXT["project_board_btn_next"],
                    {"action": "switch_board_page", "page": page + 1},
                )
            )

        if pagination_buttons:
            elements.append({"tag": "hr"})
            elements.extend(build_responsive_layout(pagination_buttons))
            if total_pages > 1:
                elements.append(
                    {
                        "tag": "markdown",
                        "content": UI_TEXT["project_board_page_info"].format(page=page, total=total_pages),
                        "text_align": "center",
                        "text_size": "notation",
                    }
                )

        elements.append({"tag": "hr"})
        elements.extend(
            build_responsive_layout(
                [
                    build_callback_button(
                        UI_TEXT["project_board_btn_new"],
                        {"action": "new_project_prompt"},
                    ),
                    build_callback_button(
                        UI_TEXT["project_board_btn_refresh"],
                        {"action": "refresh_board"},
                    ),
                ]
            )
        )

        card = CoreBuilder._wrap_card(board_title, "blue", elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_project_chat_jump_card(project: ProjectContext) -> tuple[str, str]:
        """Build the /new-chat ready card with a direct jump button."""
        theme = get_theme(project.theme_color)
        chat_name = str(getattr(project, "bound_chat_name", "") or "").strip() or "项目群"
        content = (
            f"✅ 项目 **{project.project_name}** 已就绪\n"
            f"📂 `{project.root_path}`\n"
            f"💬 **{chat_name}**"
        )
        buttons = []
        group_button = ProjectBuilder._build_project_group_jump_button(project)
        if group_button:
            buttons.append(group_button)
        buttons.append(
            build_callback_button(
                UI_TEXT["project_board_title"],
                {"action": "show_board"},
                compact=True,
            )
        )

        elements = [CoreBuilder._build_content_element(content)]
        elements.extend(build_responsive_layout(buttons))
        card = CoreBuilder._wrap_card("项目群已就绪", theme.header_template, elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_project_created_card(
        project: ProjectContext,
    ) -> tuple[str, str]:
        theme = get_theme(project.theme_color)

        content = UI_TEXT["project_create_success"].format(
            name=project.project_name, path=project.root_path
        ) + "\n• " + UI_TEXT["project_id_label"].format(id=project.project_id) + \
            f"\n• {UI_TEXT['system_theme_color_label']}{project.emoji_prefix} {project.theme_color}"

        buttons = [
            build_callback_button(
                UI_TEXT["project_btn_start_coco"],
                {"action": "enter_coco", "project_id": project.project_id},
                button_type="primary",
                compact=True,
            ),
            build_callback_button(
                UI_TEXT["project_btn_start_claude"],
                {"action": "enter_claude", "project_id": project.project_id},
                compact=True,
            ),
            build_callback_button(
                UI_TEXT["project_board_title"],
                {"action": "show_board"},
                compact=True,
            ),
        ]

        elements = [
            CoreBuilder._build_content_element(content),
        ]
        elements.extend(build_responsive_layout(buttons))

        card = CoreBuilder._wrap_card(UI_TEXT["project_created_title"], theme.header_template, elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_project_switch_notification_card(
        project: ProjectContext,
        restore_info: dict,
    ) -> tuple[str, str]:
        """Build a switch notice without forcing a resume/new-session choice."""
        context_info = ""
        if restore_info.get("has_context"):
            context_info = UI_TEXT["project_restore_info"].format(
                count=restore_info["entry_count"]
            )
            if restore_info.get("last_mode"):
                context_info += UI_TEXT["project_restore_last_mode"].format(
                    mode=restore_info["last_mode"]
                )
        content = UI_TEXT["project_switched_content"].format(
            name=project.project_name,
            root=project.root_path,
            context_info=context_info,
        )
        return ProjectBuilder.build_project_response_card(
            project,
            UI_TEXT["project_switch_title"],
            content,
            show_buttons=True,
        )
