from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote

from src.mode.manager import InteractionMode
from src.project.context import ProjectStatus

from ..shared import (
    apply_compact_style,
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
    def build_project_status_content(project: ProjectContext, global_working_dir: str) -> str:
        """Build the Markdown content for project status report."""
        lines = []

        lines.append(
            UI_TEXT["project_status_label"].format(
                emoji=project.get_status_emoji(), status=project.status.value
            )
        )
        lines.append(UI_TEXT["project_dir_label"].format(path=project.root_path))
        lines.append(UI_TEXT["project_dir_info_cwd"].format(cwd=global_working_dir))
        lines.append(
            UI_TEXT["project_last_active_label"].format(
                time_ago=CoreBuilder._format_time_ago(project.last_active)
            )
        )

        if project.coco_mode and project.coco_session_snapshot:
            snap = project.coco_session_snapshot
            lines.append(UI_TEXT["project_coco_session_header"])
            lines.append(UI_TEXT["project_session_id_label"].format(id=snap.session_id))
            lines.append(UI_TEXT["project_session_count_label"].format(count=snap.query_count))

        if project.claude_mode and project.claude_session_snapshot:
            snap = project.claude_session_snapshot
            lines.append(UI_TEXT["project_claude_session_header"])
            lines.append(UI_TEXT["project_session_id_label"].format(id=snap.session_id))
            lines.append(UI_TEXT["project_session_count_label"].format(count=snap.query_count))

        return "\n".join(lines)

    @staticmethod
    def build_current_project_card(project: ProjectContext, global_working_dir: str) -> tuple[str, str]:
        """Build the full card for current project info."""
        content = ProjectBuilder.build_project_info_content(project, global_working_dir)
        return ProjectBuilder.build_project_response_card(
            project, UI_TEXT["project_info_card_title"], content, show_buttons=True
        )

    @staticmethod
    def build_project_status_report_card(project: ProjectContext, global_working_dir: str) -> tuple[str, str]:
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

        if project.coco_mode and project.coco_session_snapshot:
            snap = project.coco_session_snapshot
            lines.append(
                f"• Coco: `{snap.session_id}` · {UI_TEXT['project_session_count_label'].format(count=snap.query_count).lstrip('• ')}"
            )
        if project.claude_mode and project.claude_session_snapshot:
            snap = project.claude_session_snapshot
            lines.append(
                f"• Claude: `{snap.session_id}` · {UI_TEXT['project_session_count_label'].format(count=snap.query_count).lstrip('• ')}"
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
    def build_restore_info_content(restore_info: dict) -> str:
        """Build the Markdown content for context restoration info."""
        if not restore_info.get("has_context"):
            return ""

        content = UI_TEXT["project_restore_info"].format(
            count=restore_info['entry_count']
        )
        if restore_info.get("last_mode"):
            content += UI_TEXT["project_restore_last_mode"].format(
                mode=restore_info['last_mode']
            )
        return content

    @staticmethod
    def build_project_switch_card(project: ProjectContext, context_info: str = "") -> tuple[str, str]:
        """Build a notification card for project switch."""
        content = UI_TEXT["project_switched_content"].format(
            name=project.project_name, root=project.root_path, context_info=context_info
        )
        return ProjectBuilder.build_project_response_card(
            project, UI_TEXT["project_switch_title"], content, show_buttons=True
        )

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
    ) -> tuple[str, str]:
        effective_mode = None
        if project:
            if getattr(project, "claude_mode", False):
                effective_mode = InteractionMode.CLAUDE
            elif getattr(project, "gemini_mode", False):
                effective_mode = InteractionMode.GEMINI
            elif getattr(project, "traex_mode", False):
                effective_mode = InteractionMode.TRAEX
            elif getattr(project, "coco_mode", False):
                effective_mode = InteractionMode.COCO

        return ProjectBuilder._build_response_card_inner(
            project,
            title,
            content,
            show_buttons=show_buttons,
            mode=effective_mode,
            extra_buttons=extra_buttons,
            footer=footer,
            image_keys=image_keys,
            banner=banner,
        )

    @staticmethod
    def build_coco_response_card(
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
            mode=InteractionMode.COCO,
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
        # Determine actual mode from project if not provided directly
        effective_mode = mode
        if effective_mode is None and project:
            if getattr(project, "claude_mode", False):
                effective_mode = InteractionMode.CLAUDE
            elif getattr(project, "gemini_mode", False):
                effective_mode = InteractionMode.GEMINI
            elif getattr(project, "traex_mode", False):
                effective_mode = InteractionMode.TRAEX
            elif getattr(project, "coco_mode", False):
                effective_mode = InteractionMode.COCO

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
    ) -> tuple[str, str]:
        board_title = UI_TEXT["project_board_title"]
        if not projects:
            empty_elements = [
                {"tag": "markdown", "content": UI_TEXT["project_board_empty_content"]},
                *build_responsive_layout(
                    [
                        apply_compact_style(
                            {
                                "tag": "button",
                                "text": {"tag": "plain_text", "content": UI_TEXT["project_board_btn_new"]},
                                "type": "primary",
                                "value": {"action": "new_project_prompt"},
                            }
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
            if project.claude_mode:
                query_count = 0
                if project.claude_session_snapshot:
                    query_count = project.claude_session_snapshot.query_count
                mode_info = UI_TEXT["project_board_claude_info"].format(count=query_count)
            elif project.coco_mode:
                query_count = 0
                if project.coco_session_snapshot:
                    query_count = project.coco_session_snapshot.query_count
                mode_info = UI_TEXT["project_board_coco_info"].format(count=query_count)
            elif project.status == ProjectStatus.BUSY and project.current_task:
                mode_info = UI_TEXT["project_board_busy_info"].format(task_type=project.current_task.task_type)

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
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": UI_TEXT["project_board_btn_switch"]},
                        "type": "primary",
                        "value": {"action": "switch_to", "project_id": project.project_id},
                    }
                )
            else:
                buttons.append(
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": UI_TEXT["project_board_btn_continue"]},
                        "type": "primary",
                        "value": {"action": "continue_dev", "project_id": project.project_id},
                    }
                )

            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["project_board_btn_detail"]},
                    "type": "default",
                    "value": {"action": "show_detail", "project_id": project.project_id},
                }
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
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["project_board_btn_prev"]},
                    "type": "default",
                    "value": {"action": "switch_board_page", "page": page - 1},
                }
            )

        if page < total_pages:
            pagination_buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["project_board_btn_next"]},
                    "type": "default",
                    "value": {"action": "switch_board_page", "page": page + 1},
                }
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
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": UI_TEXT["project_board_btn_new"]},
                        "type": "default",
                        "value": {"action": "new_project_prompt"},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": UI_TEXT["project_board_btn_refresh"]},
                        "type": "default",
                        "value": {"action": "refresh_board"},
                    },
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
            apply_compact_style(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["project_board_title"]},
                    "type": "default",
                    "value": {"action": "show_board"},
                }
            )
        )

        elements = [CoreBuilder._build_content_element(content)]
        elements.extend(build_responsive_layout(buttons))
        card = CoreBuilder._wrap_card("项目群已就绪", theme.header_template, elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_notification_card(
        project: ProjectContext,
        notification_type: str,
        title: str,
        content: str,
        suggestions: Optional[list[str]] = None,
        buttons: Optional[list[dict]] = None,
    ) -> tuple[str, str]:
        theme = get_theme(project.theme_color)

        type_emoji = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️",
            "task_complete": "🎉",
        }.get(notification_type, "📢")

        header_title = f"{type_emoji} {title}"

        elements = [
            CoreBuilder._build_directory_element(project),
            {"tag": "hr"},
            CoreBuilder._build_content_element(content),
        ]

        if suggestions:
            suggestion_text = UI_TEXT["project_notif_suggestion_header"] + "\n" + "\n".join(f"• {s}" for s in suggestions)
            elements.append({"tag": "markdown", "content": suggestion_text})

        if buttons:
            elements.extend(build_responsive_layout(buttons[:4]))
        else:
            effective_mode = None
            if project:
                if getattr(project, "claude_mode", False):
                    effective_mode = InteractionMode.CLAUDE
                elif getattr(project, "gemini_mode", False):
                    effective_mode = InteractionMode.GEMINI
                elif getattr(project, "traex_mode", False):
                    effective_mode = InteractionMode.TRAEX
                elif getattr(project, "coco_mode", False):
                    effective_mode = InteractionMode.COCO
            elements.extend(
                build_responsive_layout(
                    CoreBuilder._build_footer_buttons(
                        project,
                        mode=effective_mode,
                    )
                )
            )

        card = CoreBuilder._wrap_card(header_title, theme.header_template, elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def _build_resume_card(
        project: ProjectContext,
        mode: str,
    ) -> tuple[str, str]:
        theme = get_theme(project.theme_color)
        mode_names = {
            "coco": "Coco",
            "claude": "Claude",
            "aiden": "Aiden",
            "codex": "Codex",
            "gemini": "Gemini",
            "traex": "Traex",
            "tui2acp": "Tui2ACP",
        }
        mode_name = mode_names.get(mode, mode.title())
        snapshot = project.get_programming_snapshot(mode)

        if not snapshot:
            return ProjectBuilder.build_project_response_card(
                project, f"{mode_name}{UI_TEXT['system_mode_label']}", UI_TEXT["project_resume_no_session"], show_buttons=True
            )

        content = (
            UI_TEXT["project_resume_detected"].format(mode=mode_name) + "\n\n"
            "• " + UI_TEXT["project_resume_session_id"].format(id=snapshot.session_id) + "\n"
            "• " + UI_TEXT["project_resume_query_count"].format(count=snapshot.query_count) + "\n"
            "• " + UI_TEXT["project_resume_last_query"].format(query=snapshot.last_query)
        )

        resume_action = f"resume_{mode}"
        new_action = f"new_{mode}"

        buttons = [
            apply_compact_style(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["project_resume_btn_resume"]},
                    "type": "primary",
                    "value": {
                        "action": resume_action,
                        "project_id": project.project_id,
                        "session_id": snapshot.session_id,
                    },
                }
            ),
            apply_compact_style(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["project_resume_btn_new"]},
                    "type": "default",
                    "value": {
                        "action": new_action,
                        "project_id": project.project_id,
                    },
                }
            ),
        ]

        try:
            effective_mode = InteractionMode(mode)
        except ValueError:
            effective_mode = None

        header_title = CoreBuilder._build_header_title(project, mode=effective_mode)

        elements = [
            CoreBuilder._build_directory_element(project),
            {"tag": "hr"},
        ]
        elements.append(CoreBuilder._build_content_element(content))
        elements.extend(build_responsive_layout(buttons))

        card = CoreBuilder._wrap_card(header_title, theme.header_template, elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_coco_resume_card(project: ProjectContext) -> tuple[str, str]:
        return ProjectBuilder._build_resume_card(project, "coco")

    @staticmethod
    def build_claude_resume_card(project: ProjectContext) -> tuple[str, str]:
        return ProjectBuilder._build_resume_card(project, "claude")

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
            apply_compact_style(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["project_btn_start_coco"]},
                    "type": "primary",
                    "value": {"action": "enter_coco", "project_id": project.project_id},
                }
            ),
            apply_compact_style(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["project_btn_start_claude"]},
                    "type": "default",
                    "value": {"action": "enter_claude", "project_id": project.project_id},
                }
            ),
            apply_compact_style(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["project_board_title"]},
                    "type": "default",
                    "value": {"action": "show_board"},
                }
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
        """Consolidates switch message, context restoration info, and resume session logic."""
        context_info = ProjectBuilder.build_restore_info_content(restore_info)

        if project.coco_session_snapshot and project.coco_session_snapshot.is_resumable:
            return ProjectBuilder.build_coco_resume_card(project)
        elif project.claude_session_snapshot and project.claude_session_snapshot.is_resumable:
            return ProjectBuilder.build_claude_resume_card(project)
        else:
            return ProjectBuilder.build_project_switch_card(project, context_info)
