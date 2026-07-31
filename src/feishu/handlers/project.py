"""Project management handler — create, switch, close, status, context preserve/restore."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ...card import CardBuilder
from ...card.ui_text import UI_TEXT
from ...project import ContextEntryType, ContextSourceMode
from .base import BaseHandler

if TYPE_CHECKING:
    from ...project import ProjectContext

logger = logging.getLogger(__name__)


class ProjectHandler(BaseHandler):
    """Handles project CRUD and context preserve/restore."""

    def create_project(self, message_id: str, chat_id: str, name: str, path: str):
        success, msg, project = self.project_manager.create_project(
            project_id=None,
            project_name=name,
            root_path=path,
            chat_id=chat_id,
        )

        if success and project:
            msg_type, card_content = CardBuilder.build_project_created_card(project)
            response_id = self.reply_card(message_id, card_content)
            if response_id:
                self.register_message_project(response_id, project)
        else:
            self.reply_error(message_id, UI_TEXT["project_create_error"].format(error=msg))

    def show_project_board(self, message_id: str, chat_id: str, origin_message_id: Optional[str] = None, page: int = 1):
        projects = self.project_manager.get_all_projects(chat_id=chat_id)
        active_project = self.project_manager.get_active_project(chat_id)
        current_id = active_project.project_id if active_project else None

        msg_type, content = CardBuilder.build_status_board_card(projects, current_id, page=page)

        if origin_message_id:
            if self.update_card(origin_message_id, content):
                if active_project:
                    self.register_message_project(origin_message_id, active_project)
                return

        response_id = self.reply_card(message_id, content)

        if response_id and active_project:
            self.register_message_project(response_id, active_project)

    def show_current_project(self, message_id: str, chat_id: str, project: Optional["ProjectContext"]):
        if not project:
            self.reply_text(
                message_id,
                UI_TEXT["project_board_empty"],
            )
            return

        global_working_dir = self.get_working_dir(chat_id)
        msg_type, card_content = CardBuilder.build_current_project_card(project, global_working_dir)
        response_id = self.reply_card(message_id, card_content)
        if response_id:
            self.register_message_project(response_id, project)

    def show_project_status(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"],
        origin_message_id: Optional[str] = None,
    ):
        if not project:
            self.show_project_board(message_id, chat_id)
            return

        global_working_dir = self.get_working_dir(chat_id)
        msg_type, card_content = CardBuilder.build_project_status_report_card(project, global_working_dir)

        if origin_message_id:
            if self.update_card(origin_message_id, card_content):
                self.register_message_project(origin_message_id, project)
                return

        response_id = self.reply_card(message_id, card_content)
        if response_id:
            self.register_message_project(response_id, project)

    # ------------------------------------------------------------------
    # Context preserve / restore
    # ------------------------------------------------------------------
    def preserve_project_context(self, chat_id: str, project: "ProjectContext"):
        from ...mode import InteractionMode

        pid = project.project_id
        current_mode = self.mode_manager.get_mode(chat_id, project_id=pid)
        logger.info(
            "[%s] 保留项目上下文: project=%s, mode=%s",
            chat_id,
            project.project_name,
            current_mode.value if hasattr(current_mode, "value") else current_mode,
        )

        if current_mode == InteractionMode.COCO:
            session = self.ctx.coco_manager.get_session(chat_id, project_id=pid)
            if session:
                project.update_coco_snapshot(query=session.last_query, query_count=session.message_count)
                self.context_manager.update_context(
                    pid,
                    session_snapshot={"data": session.to_snapshot(), "source_mode": ContextSourceMode.COCO.value},
                    chat_id=chat_id,
                )
        elif current_mode == InteractionMode.CLAUDE:
            session = self.ctx.claude_manager.get_session(chat_id, project_id=pid)
            if session:
                project.update_claude_snapshot(
                    query=session.last_query, query_count=session.message_count, session_id=session.session_id
                )
                self.context_manager.update_context(
                    pid,
                    session_snapshot={"data": session.to_snapshot(), "source_mode": ContextSourceMode.CLAUDE.value},
                    chat_id=chat_id,
                )

    def restore_project_context(self, project: "ProjectContext", *, chat_id: str = "") -> dict:
        ctx = self.context_manager.store.get(project.project_id, chat_id=chat_id)
        if ctx is None:
            logger.debug("[恢复上下文] 项目 %s 无已有上下文", project.project_name)
            return {"has_context": False, "entry_count": 0, "version_count": 0, "last_mode": None, "has_bridge": False}

        last_mode = None
        transitions = ctx.get_entries_by_type(ContextEntryType.MODE_TRANSITION)
        if transitions:
            last_transition = transitions[-1]
            last_mode = last_transition.metadata.get("to_mode")

        info = {
            "has_context": True,
            "entry_count": ctx.entry_count,
            "version_count": len(ctx.versions),
            "last_mode": last_mode,
            "has_bridge": ctx.last_bridge_summary is not None,
        }
        logger.info(
            "[恢复上下文] 项目 %s: %d 条记录, %d 版本, 上次模式=%s, 有桥接=%s",
            project.project_name,
            info["entry_count"],
            info["version_count"],
            info["last_mode"],
            info["has_bridge"],
        )
        return info

    # ------------------------------------------------------------------
    # Switch / Close
    # ------------------------------------------------------------------
    def switch_project(
        self,
        message_id: str,
        chat_id: str,
        name: str,
        auto_enter_coco: bool = True,
        coco_handler=None,
        claude_handler=None,
    ):
        """Switch active project.

        *coco_handler* and *claude_handler* are the programming-mode handlers
        used to exit the current mode safely when switching projects.
        """
        project, hint = self.project_manager.find_project_by_name_with_hint(name, chat_id=chat_id)
        if not project:
            if hint:
                self.reply_error(message_id, hint)
                return
            results = self.project_manager.search_projects(name, chat_id=chat_id)
            content = CardBuilder.build_project_not_found_content(name, results)
            self.reply_error(message_id, content, title=UI_TEXT["project_not_found_title"])
            return

        valid, path_msg = self.project_manager.validate_project_path(project.project_id)
        if not valid:
            self.reply_error(
                message_id, UI_TEXT["project_dir_not_exist"].format(path=path_msg)
            )
            return

        old_project = self.project_manager.get_active_project(chat_id)
        if old_project and old_project.project_id != project.project_id:
            self.preserve_project_context(chat_id, old_project)

            from ...mode import InteractionMode

            current_mode = self.mode_manager.get_mode(chat_id, project_id=old_project.project_id)
            if current_mode == InteractionMode.COCO and coco_handler:
                coco_handler.exit_mode(message_id, chat_id, project=old_project)
            elif current_mode == InteractionMode.CLAUDE and claude_handler:
                claude_handler.exit_mode(message_id, chat_id, project=old_project)

            old_ctx = self.context_manager.store.get(old_project.project_id, chat_id=chat_id)
            if old_ctx:
                old_ctx.create_version(
                    reason=f"project_switch: {old_project.project_name} -> {project.project_name}",
                    source_mode=ContextSourceMode.SMART,
                    summary=f"Switched to project {project.project_name}",
                )

        success, msg = self.project_manager.set_active_project(chat_id, project.project_id)
        if not success:
            self.reply_error(message_id, UI_TEXT["project_switch_error"].format(error=msg))
            return

        restore_info = self.restore_project_context(project, chat_id=chat_id)
        self.context_manager.store.get_or_create(project.project_id, chat_id=chat_id)

        if auto_enter_coco and coco_handler:
            coco_handler.enter_mode(message_id, chat_id, project=project)
        else:
            msg_type, card_content = CardBuilder.build_project_switch_notification_card(project, restore_info)

            response_id = self.reply_card(message_id, card_content)
            if response_id:
                self.register_message_project(response_id, project)

    def close_project(self, message_id: str, chat_id: str, name: str):
        project, hint = self.project_manager.find_project_by_name_with_hint(name, chat_id=chat_id)
        if not project:
            self.reply_error(message_id, hint or UI_TEXT["project_not_found"].format(name=name))
            return

        success, msg = self.project_manager.close_project(project.project_id)
        if success:
            self.reply_text(message_id, UI_TEXT["project_close_success"].format(name=name))
        else:
            self.reply_error(message_id, UI_TEXT["project_close_error"].format(error=msg))

    def handle_new_chat_project(self, message_id: str, chat_id: str, data: dict) -> None:
        """Handle /new-chat command: create project + dedicated Feishu group."""
        from ...project_chat import ProjectChatService
        from ...project_chat.lark_chat_client import LarkChatClient
        from ...thread.manager import get_current_sender_id

        sender_open_id = get_current_sender_id() or ""
        if not sender_open_id:
            self.reply_error(message_id, "无法获取发送者信息")
            return

        # Lazy-init service
        if not hasattr(self, "_project_chat_service"):
            lark_client = LarkChatClient(api_client_factory=self.ctx.api_client_factory)
            shared_registry = vars(self.ctx).get("managed_group_registry")
            owner_id = vars(self.ctx).get("managed_group_owner_id", "")
            receiving_bot_ref = vars(self.ctx).get(
                "managed_group_receiving_bot_ref", ""
            )
            self._project_chat_service = ProjectChatService(
                project_manager=self.project_manager,
                lark_chat_client=lark_client,
                reply_fn=lambda mid, text, msg_type: (
                    self.reply_card(mid, text) if msg_type == "interactive"
                    else self.reply_text(mid, text)
                ),
                send_to_chat_fn=lambda cid, msg_type, text, _: (
                    self.send_card_to_chat(cid, text) if msg_type == "interactive"
                    else self.send_text_to_chat(cid, text)
                ),
                managed_group_registry=shared_registry,
                owner_id=owner_id,
                receiving_bot_ref=receiving_bot_ref,
            )

        # Inject shell working directory so service uses the same default as /new
        if "path" not in data or not data["path"]:
            data["path"] = self.get_working_dir(chat_id)

        self._project_chat_service.handle(message_id, chat_id, sender_open_id, data)
