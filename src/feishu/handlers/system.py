"""System handler — help, exit mode, shell commands, directory switching, intercepted commands."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ...acp.claude_capabilities import (
    is_1m_variant,
    model_supports_1m,
    strip_1m_suffix,
)
from ...acp.helper import (
    fetch_acp_models,
    invalidate_acp_model_cache,
    is_programming_tool_available,
    list_acp_tools,
)
from ...acp.providers import get_providers
from ...card.builders.project import ProjectBuilder
from ...card.builders.system import SystemBuilder
from ...card.render.model_cascade import compose_model_selection, parse_model_selection
from ...card.ui_text import UI_TEXT
from ...coco_model import get_coco_model_manager
from ...tasking import TaskPriority, TaskSpec
from ...utils.errors import safe_error_message
from ..emoji import EmojiReaction
from ..message_formatter import FeishuMessageFormatter as fmt
from ..product_catalog import retired_command_tokens
from ..slash_command_parser import CommandMatch, SlashCommandParser
from .base import BaseHandler
from .employee import EmployeeHandler
from .lock_commands import LockCommandsMixin

if TYPE_CHECKING:
    from ...project import ProjectContext
    from ..handler_context import HandlerContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SystemSubcommands:
    """Minimal delegator view for a SystemHandler responsibility group."""
    _owner: "SystemHandler"
    _method_names: tuple[str, ...]

    def __getattr__(self, name: str):
        if name in self._method_names:
            return getattr(self._owner, name)
        raise AttributeError(name)


class SystemHandler(LockCommandsMixin, BaseHandler):
    """Help, exit, shell, directory, and intercepted-command handling."""

    # `/status` remains the established Deep/Spec diagnostics command. The
    # Standalone v5 Manager was superseded by the Journal-backed employee/Team
    # runtime. Keep its old spellings fail-closed with an explicit migration
    # message instead of reviving the obsolete process-local command surface.
    _RETIRED_AUTONOMOUS_MANAGER_COMMANDS = retired_command_tokens()

    def __init__(self, ctx: "HandlerContext") -> None:
        super().__init__(ctx)
        self.employee = EmployeeHandler(ctx)
        self._init_command_registry()
        self.help_commands = _SystemSubcommands(self, ("show_help", "show_full_help", "handle_help_category", "handle_menu_command"))
        self.shell_commands = _SystemSubcommands(self, ("submit_shell_command", "execute_shell_and_reply", "change_directory"))
        self.lock_commands = _SystemSubcommands(self, ("handle_force_release_repo_lock", "handle_confirm_lock", "handle_cancel_lock", "handle_confirm_force_release", "handle_cancel_force_release"))

    @staticmethod
    def _project_id(project: Optional["ProjectContext"]) -> Optional[str]:
        project_id = getattr(project, "project_id", None) if project else None
        return project_id if isinstance(project_id, str) else None

    def _init_command_registry(self):
        """Initialize the command dispatch registry."""
        # Exact match handlers: command -> handler_func(message_id, chat_id, text, project)
        self._exact_handlers = {
            "/help": lambda m, c, t, p: self.show_full_help(m, c, p),
            "/帮助": lambda m, c, t, p: self.show_full_help(m, c, p),
            "/coco": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "coco", p),
            "/enter_coco": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "coco", p),
            "/claude": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "claude", p),
            "/enter_claude": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "claude", p),
            "/aiden": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "aiden", p),
            "/enter_aiden": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "aiden", p),
            "/codex": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "codex", p),
            "/enter_codex": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "codex", p),
            "/gemini": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "gemini", p),
            "/enter_gemini": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "gemini", p),
            "/traex": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "traex", p),
            "/enter_traex": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "traex", p),
            "/grok": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "grok", p),
            "/enter_grok": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "grok", p),
            "/exit": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/quit": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_coco": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_coco": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_claude": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_claude": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_aiden": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_aiden": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_codex": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_codex": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_gemini": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_gemini": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_traex": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_traex": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_grok": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_grok": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/coco_status": lambda m, c, t, p: self.show_coco_status(m, c),
            "/coco_info": lambda m, c, t, p: self.get_handler("coco").show_info(m, c, p),
            "/claude_info": lambda m, c, t, p: self.get_handler("claude").show_info(m, c, p),
            "/aiden_info": lambda m, c, t, p: self.get_handler("aiden").show_info(m, c, p),
            "/codex_info": lambda m, c, t, p: self.get_handler("codex").show_info(m, c, p),
            "/gemini_info": lambda m, c, t, p: self.get_handler("gemini").show_info(m, c, p),
            "/traex_info": lambda m, c, t, p: self.get_handler("traex").show_info(m, c, p),
            "/grok_info": lambda m, c, t, p: self.get_handler("grok").show_info(m, c, p),
            "/projects": lambda m, c, t, p: self.get_handler("project").show_project_board(m, c),
            "/project": lambda m, c, t, p: self.get_handler("project").show_project_board(m, c),
            "/switch": lambda m, c, t, p: self.get_handler("project").show_project_board(m, c),
            "/new-chat": lambda m, c, t, p: self._handle_new_chat_project_args(m, c, ""),
            "/acp": lambda m, c, t, p: self.handle_acp_command(m, c, t, p),
            "/menu": lambda m, c, t, p: self.handle_menu_command(m, c, p),
            "/tools": lambda m, c, t, p: self.show_tools_list(m, c, p),
            "/tools_status": lambda m, c, t, p: self.show_tools_status(m, c, p),
            "/model": lambda m, c, t, p: self.handle_model_command(m, c, t, p),
            "/lock": lambda m, c, t, p: self._handle_lock_command(m, c, "lock"),
            "/unlock": lambda m, c, t, p: self._handle_lock_command(m, c, "unlock"),
            "/setadmin": lambda m, c, t, p: self._handle_setadmin_command(m, c, ""),
            "/access": lambda m, c, t, p: self._handle_access_command(m, c, ""),
            "/employees": lambda m, c, t, p: self.employee.list_employees_roster(m, c),
            "/employee-role": lambda m, c, t, p: self.employee.update_employee_role(
                m, c, ""
            ),
            "/hire": lambda m, c, t, p: self.employee.hire_employee(m, c, ""),
            "/fire": lambda m, c, t, p: self.employee.fire_employee(m, c, ""),
            "/history": lambda m, c, t, p: self.employee.show_employee_history(
                m, c, ""
            ),
            "/employee-memory": lambda m, c, t, p: self.employee.show_employee_memory(
                m, c, ""
            ),
        }

        # Prefix match handlers: prefix -> handler_func(message_id, chat_id, text, project)
        # Note: Order matters if prefixes overlap (not the case here yet)
        self._prefix_handlers = [
            ("/status", lambda m, c, t, p: self.get_handler("diagnostics").show_unified_status(m, c, t, p)),
            ("/tasks", lambda m, c, t, p: self.get_handler("diagnostics").show_task_board(m, c, t, p)),
            ("/diff", lambda m, c, t, p: self.get_handler("diagnostics").show_context_diff(m, c, t, p)),
            ("/trace", lambda m, c, t, p: self.get_handler("diagnostics").show_message_trace(m, c, t, p)),
            ("/acp", self.handle_acp_command),
            ("/model", self.handle_model_command),
            (
                "/employee-role",
                lambda m, c, t, p: self.employee.update_employee_role(
                    m,
                    c,
                    (SlashCommandParser.parse(t).args if SlashCommandParser.parse(t) else ""),
                ),
            ),
            (
                "/hire",
                lambda m, c, t, p: self.employee.hire_employee(
                    m,
                    c,
                    (SlashCommandParser.parse(t).args if SlashCommandParser.parse(t) else ""),
                ),
            ),
            (
                "/fire",
                lambda m, c, t, p: self.employee.fire_employee(
                    m,
                    c,
                    (SlashCommandParser.parse(t).args if SlashCommandParser.parse(t) else ""),
                ),
            ),
            (
                "/history",
                lambda m, c, t, p: self.employee.show_employee_history(
                    m,
                    c,
                    (SlashCommandParser.parse(t).args if SlashCommandParser.parse(t) else ""),
                ),
            ),
            (
                "/employee-memory",
                lambda m, c, t, p: self.employee.show_employee_memory(
                    m,
                    c,
                    (SlashCommandParser.parse(t).args if SlashCommandParser.parse(t) else ""),
                ),
            ),
        ]

    def _handle_direct_mode_enter(
        self,
        message_id: str,
        chat_id: str,
        mode_key: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        self.show_explicit_acp_model_selection(
            message_id,
            chat_id,
            mode_key,
            project,
        )

    def _handle_switch_args(self, message_id: str, chat_id: str, args: str) -> None:
        name = (args or "").strip()
        if name:
            self.get_handler("project").switch_project(
                message_id,
                chat_id,
                name,
                coco_handler=self.get_handler("coco"),
                claude_handler=self.get_handler("claude"),
            )
        else:
            self.get_handler("project").show_project_board(message_id, chat_id)


    def _handle_new_project_args(self, message_id: str, chat_id: str, args: str) -> None:
        parts = (args or "").strip().split(None, 1)
        name = parts[0] if parts else ""
        path = parts[1] if len(parts) > 1 else self.get_working_dir(chat_id)
        if name:
            self.get_handler("project").create_project(message_id, chat_id, name, path)
        else:
            self.reply_error(
                message_id, UI_TEXT["system_new_project_usage"], title=UI_TEXT["system_arg_error"]
            )

    def _handle_new_chat_project_args(self, message_id: str, chat_id: str, args: str) -> None:
        parts = (args or "").strip().split(None, 3)
        data: dict[str, str] = {}
        if len(parts) >= 1 and parts[0]:
            data["name"] = parts[0]
        if len(parts) >= 2 and parts[1]:
            data["suffix"] = parts[1]
        if len(parts) >= 3 and parts[2]:
            data["path"] = parts[2]
        if len(parts) >= 4 and parts[3]:
            data["path"] = f"{data.get('path', '')} {parts[3]}".strip()
        self.get_handler("project").handle_new_chat_project(message_id, chat_id, data)

    def _handle_close_args(self, message_id: str, chat_id: str, args: str) -> None:
        name = (args or "").strip()
        if name:
            self.get_handler("project").close_project(message_id, chat_id, name)
        else:
            self.reply_error(
                message_id,
                UI_TEXT["system_close_project_usage"],
                title=UI_TEXT["system_arg_error"],
            )

    # ------------------------------------------------------------------
    # Command predicates
    # ------------------------------------------------------------------
    @staticmethod
    def is_exit_command(text: str) -> bool:
        text_lower = text.lower().strip()
        exit_commands = {
            "/exit",
            "/quit",
            "/end_coco",
            "/exit_coco",
            "/end_claude",
            "/exit_claude",
            "/end_aiden",
            "/exit_aiden",
            "/end_codex",
            "/exit_codex",
            "/end_gemini",
            "/exit_gemini",
            "/end_traex",
            "/exit_traex",
            "/end_grok",
            "/exit_grok",
        }
        exit_keywords = {
            "退出模式",
            "退出编程模式",
            "退出编程",
            "结束编程",
            "退出claude",
            "退出coco",
            "退出aiden",
            "退出grok",
            "退出codex",
            "退出gemini",
            "退出traex",
        }
        if text_lower in exit_commands:
            return True
        return any(kw in text_lower for kw in exit_keywords)

    @staticmethod
    def is_deep_command(text: str) -> bool:
        match = SlashCommandParser.parse(text)
        return bool(match and match.command in {"/deep", "/deep_status", "/deep_update", "/stop_deep"})

    @staticmethod
    def is_spec_command(text: str) -> bool:
        match = SlashCommandParser.parse(text)
        spec_commands = {
            "/spec",
            "/stop_spec",
            "/spec_status",
            "/spec_history",
            "/spec_metrics",
            "/spec_config",
            "/spec_save",
            "/spec_guide",
            "/spec_export",
        }
        return bool(match and match.command in spec_commands)

    @staticmethod
    def is_workflow_command(text: str) -> bool:
        from ...workflow_engine.commands import TOPIC_ENGINE_COMMANDS

        match = SlashCommandParser.parse(text)
        return bool(match and match.command in TOPIC_ENGINE_COMMANDS)

    @staticmethod
    def is_likely_shell_command(text: str) -> bool:
        """Heuristic check for common shell commands.

        Used for early routing in _handle_message to prevent shell commands
        from blocking behind long-running programming tasks on the project queue.
        """
        from ...agent.intent_recognizer import IntentRecognizer

        return IntentRecognizer.looks_like_shell(text)

    @staticmethod
    def is_interceptable_command_match(command_match: CommandMatch | None) -> bool:
        """Return True when *command_match* should be routed to SystemHandler.

        NOTE: This is the request-scoped SSOT variant (no parsing).
        """
        m = command_match
        if not m:
            return False
        cmd = m.command

        exact_commands = {
            "/help",
            "/帮助",
            "/coco",
            "/enter_coco",
            "/claude",
            "/enter_claude",
            "/aiden",
            "/enter_aiden",
            "/codex",
            "/enter_codex",
            "/gemini",
            "/enter_gemini",
            "/traex",
            "/enter_traex",
            "/grok",
            "/enter_grok",
            "/exit",
            "/quit",
            "/end_coco",
            "/exit_coco",
            "/end_claude",
            "/exit_claude",
            "/end_aiden",
            "/exit_aiden",
            "/end_codex",
            "/exit_codex",
            "/end_gemini",
            "/exit_gemini",
            "/end_traex",
            "/exit_traex",
            "/end_grok",
            "/exit_grok",
            "/coco_status",
            "/coco_info",
            "/claude_info",
            "/aiden_info",
            "/codex_info",
            "/gemini_info",
            "/traex_info",
            "/grok_info",
            "/projects",
            "/status",
            "/project",
            "/switch",
            "/new-chat",
            "/tasks",
            "/diff",
            "/trace",
            "/acp",
            "/menu",
            "/tools",
            "/tools_status",
            "/model",
            "/lock",
            "/unlock",
            "/setadmin",
            "/access",
            "/employees",
            "/employee-role",
            "/hire",
            "/fire",
            "/history",
            "/employee-memory",
            "/btw",
            "/goals",
            "/runs",
        }
        exact_commands.update(SystemHandler._RETIRED_AUTONOMOUS_MANAGER_COMMANDS)
        if not m.has_args and cmd in exact_commands:
            return True
        prefix_commands = {
            "/switch",
            "/new",
            "/new-chat",
            "/close",
            "/tasks",
            "/diff",
            "/trace",
            "/status",
            "/model",
            "/btw",
            "/setadmin",
            "/access",
            "/employee-role",
            "/hire",
            "/fire",
            "/history",
            "/employee-memory",
            "/goal",
            "/approve",
            "/runs",
        }
        prefix_commands.update(SystemHandler._RETIRED_AUTONOMOUS_MANAGER_COMMANDS)
        return cmd in prefix_commands

    # ------------------------------------------------------------------
    # Intercepted command router
    # ------------------------------------------------------------------
    def handle_intercepted_command(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional["ProjectContext"] = None,
        *,
        command_match: CommandMatch | None = None,
    ):
        m = command_match
        if not m:
            # SSOT: intercepted commands must carry request-scoped CommandMatch.
            self.reply_error(message_id, UI_TEXT["system_slash_parse_missing"], title=UI_TEXT["system_internal_error"])
            return
        # Use the canonical command as the routing key while preserving the
        # original text for handlers that still need it for legacy parsing.
        text_lower = m.command

        if text_lower == "/btw":
            self._handle_btw_command(message_id, chat_id, m, project)
            return
        if text_lower == "/setadmin":
            self._handle_setadmin_command(message_id, chat_id, m.args)
            return
        if text_lower == "/access":
            self._handle_access_command(message_id, chat_id, m.args)
            return

        # 1. Try exact match
        if not m.has_args:
            handler = self._exact_handlers.get(text_lower)
            if handler:
                handler(message_id, chat_id, text, project)
                return

        # 1b. Prefix commands that historically used text slicing: route with parsed args.
        if text_lower == "/switch":
            self._handle_switch_args(message_id, chat_id, m.args)
            return
        if text_lower == "/new":
            self._handle_new_project_args(message_id, chat_id, m.args)
            return
        if text_lower == "/new-chat":
            self._handle_new_chat_project_args(message_id, chat_id, m.args)
            return
        if text_lower == "/close":
            self._handle_close_args(message_id, chat_id, m.args)
            return

        # 2. Try prefix match
        for prefix, handler in self._prefix_handlers:
            if text_lower == prefix:
                handler(message_id, chat_id, text, project)
                return

        # 3. Autonomous system commands
        if text_lower in self._RETIRED_AUTONOMOUS_MANAGER_COMMANDS:
            self._handle_retired_autonomous_manager_command(
                message_id,
                chat_id,
                m,
                project,
            )
            return

        self.reply_text(
            message_id,
            UI_TEXT["system_unknown_slash_command"].format(command=m.raw_command or text_lower),
        )

    def _handle_retired_autonomous_manager_command(
        self,
        message_id: str,
        chat_id: str,
        m: "CommandMatch",
        project: "Optional[ProjectContext]" = None,
    ) -> None:
        """Fail closed for the retired standalone Autonomous Manager surface."""
        del chat_id, project
        if m.command == "/goal" and m.args:
            self.reply_text(
                message_id,
                UI_TEXT["system_autonomous_manager_retired_goal"].format(task=m.args),
            )
            return
        self.reply_text(message_id, UI_TEXT["system_autonomous_manager_retired"])

    def _handle_setadmin_command(self, message_id: str, chat_id: str, args: str = "") -> None:
        from ...thread import get_current_is_p2p, get_current_sender_id

        sender_id = get_current_sender_id() or ""
        chat_type = "p2p" if get_current_is_p2p() else "group"
        result = self._admin_bootstrap_service().set_admin(
            sender_id,
            args,
            chat_type=chat_type,
            chat_id=chat_id,
            message_id=message_id,
        )
        if result.success:
            if result.code == "bootstrap":
                self.reply_text(message_id, UI_TEXT["system_setadmin_bootstrap_success"])
            else:
                self.reply_text(
                    message_id,
                    UI_TEXT["system_setadmin_update_success"].format(admin_id=result.target_id),
                )
            return

        if result.code == "missing_sender":
            self.reply_error(message_id, UI_TEXT["system_setadmin_missing_sender"])
        elif result.code == "invalid_target":
            self.reply_error(message_id, UI_TEXT["system_setadmin_invalid_target"])
        elif result.code == "bootstrap_requires_p2p":
            self.reply_error(message_id, UI_TEXT["system_setadmin_requires_p2p"])
        elif result.code == "rate_limited":
            self.reply_error(message_id, UI_TEXT["system_setadmin_rate_limited"])
        elif result.code == "persistence_failed":
            self.reply_error(
                message_id,
                "管理员配置持久化失败，在线访问策略未改变。",
            )
        elif result.code == "commit_uncertain":
            self.reply_error(
                message_id,
                "管理员配置文件替换已发生，但目录耐久性确认失败；"
                "在线策略已按磁盘现状重新对齐，并记录了安全阻断项。",
            )
        elif result.code == "commit_uncertain_unreconciled":
            self.reply_error(
                message_id,
                "管理员配置替换状态不确定，且无法读取磁盘现状；"
                "在线策略保持原快照，已记录安全阻断项。",
            )
        elif result.code == "commit_uncertain_refresh_failed":
            self.reply_error(
                message_id,
                "管理员配置替换状态不确定；已读取磁盘现状，但运行时刷新失败，"
                "并已记录安全阻断项。",
            )
        elif result.code == "commit_cleanup_failed":
            self.reply_error(
                message_id,
                "管理员配置已持久化并在线生效，但事务资源清理失败；"
                "已记录安全阻断项，请检查文件系统状态。",
            )
        elif result.code == "commit_cleanup_refresh_failed":
            self.reply_error(
                message_id,
                "管理员配置已持久化，但事务资源清理和运行时刷新均失败；"
                "已记录安全阻断项，请修复后重启。",
            )
        elif result.code == "policy_refresh_failed":
            self.reply_error(
                message_id,
                "管理员配置已写入磁盘，但在线访问策略刷新失败；"
                "当前进程仍使用旧策略，请修复配置后重启。",
            )
        elif result.code == "settings_mirror_failed":
            self.reply_error(
                message_id,
                "管理员配置已写入磁盘，但运行时设置镜像失败；"
                "在线策略保持旧快照并已记录安全阻断项。",
            )
        else:
            self.reply_text(message_id, UI_TEXT["system_setadmin_denied"])

    def _handle_access_command(
        self,
        message_id: str,
        chat_id: str,
        args: str = "",
    ) -> None:
        from ...thread import get_current_is_p2p, get_current_sender_id

        access_args = (args or "").strip()
        if access_args.casefold().startswith("rotate-main-bot "):
            sender_id = get_current_sender_id() or ""
            owner_id = vars(self.ctx).get("managed_group_owner_id", "")
            rotate = vars(self.ctx).get("managed_group_bot_rotation")
            expected_bot_ref = access_args.split(maxsplit=1)[1].strip()
            if (
                not get_current_is_p2p()
                or not owner_id
                or sender_id != owner_id
                or not callable(rotate)
                or not expected_bot_ref
            ):
                self.reply_error(message_id, "仅配置的 Owner 可在私聊中执行主 Bot 轮换。")
                return
            rotated, rejected = rotate(expected_bot_ref)
            self.reply_text(
                message_id,
                f"主 Bot 轮换完成：已更新 {rotated} 个群，远端校验未通过 {rejected} 个群。",
            )
            return

        if access_args.casefold() == "migration-status":
            sender_id = get_current_sender_id() or ""
            owner_id = vars(self.ctx).get("managed_group_owner_id", "")
            registry = vars(self.ctx).get("managed_group_registry")
            if (
                not get_current_is_p2p()
                or not owner_id
                or sender_id != owner_id
                or registry is None
            ):
                self.reply_error(message_id, "仅配置的 Owner 可在私聊中查看迁移待办。")
                return
            dispositions = registry.migration_dispositions()
            if not dispositions:
                self.reply_text(message_id, "✅ 当前没有待人工处理的受管群迁移项。")
                return
            lines = ["受管群迁移待办："]
            lines.extend(
                f"- `{chat}` → `{project}`：{status}"
                for chat, project, status in dispositions
            )
            self.reply_text(message_id, "\n".join(lines))
            return

        if access_args.casefold().startswith("adopt-chat "):
            parts = access_args.split(maxsplit=2)
            sender_id = get_current_sender_id() or ""
            owner_id = vars(self.ctx).get("managed_group_owner_id", "")
            if not get_current_is_p2p() or not owner_id or sender_id != owner_id:
                self.reply_error(
                    message_id,
                    "仅配置的 Owner 可在与 Bot 的私聊中收养已有群。",
                )
                return
            if (
                len(parts) != 3
                or not parts[1].startswith("oc_")
                or not parts[2].strip()
            ):
                self.reply_error(
                    message_id,
                    "用法：`/access adopt-chat <oc_chat_id> <project-id-or-exact-name>`",
                )
                return
            project_handler = self.ctx.handlers.get("project")
            adopt = getattr(project_handler, "adopt_managed_chat", None)
            if not callable(adopt):
                self.reply_error(message_id, "受管群收养服务未就绪。")
                return
            adopt(message_id, parts[1], parts[2].strip())
            return

        if access_args.casefold() != "allow-chat":
            self.reply_text(
                message_id,
                "用法：目标群内发送 `/access allow-chat`，或由 Owner 私聊发送 "
                "`/access adopt-chat <oc_chat_id> <project-id-or-exact-name>`。",
            )
            return

        sender_id = get_current_sender_id() or ""
        chat_type = "p2p" if get_current_is_p2p() else "group"
        result = self._admin_bootstrap_service().allow_current_chat(
            sender_id,
            chat_id,
            chat_type=chat_type,
            message_id=message_id,
        )
        if result.success:
            self.reply_text(message_id, "✅ 当前群已加入访问白名单，立即生效。")
            return
        if result.code == "access_requires_group":
            self.reply_error(
                message_id,
                "请在需要授权的目标群内发送 `/access allow-chat`。",
            )
        elif result.code == "persistence_failed":
            self.reply_error(
                message_id,
                "群授权持久化失败，在线访问策略未改变。",
            )
        elif result.code == "commit_uncertain":
            self.reply_error(
                message_id,
                "群授权配置文件替换已发生，但目录耐久性确认失败；"
                "在线策略已按磁盘现状重新对齐，并记录了安全阻断项。",
            )
        elif result.code == "commit_uncertain_unreconciled":
            self.reply_error(
                message_id,
                "群授权配置替换状态不确定，且无法读取磁盘现状；"
                "在线策略保持原快照，已记录安全阻断项。",
            )
        elif result.code == "commit_uncertain_refresh_failed":
            self.reply_error(
                message_id,
                "群授权配置替换状态不确定；已读取磁盘现状，但运行时刷新失败，"
                "并已记录安全阻断项。",
            )
        elif result.code == "commit_cleanup_failed":
            self.reply_error(
                message_id,
                "群授权配置已持久化并在线生效，但事务资源清理失败；"
                "已记录安全阻断项，请检查文件系统状态。",
            )
        elif result.code == "commit_cleanup_refresh_failed":
            self.reply_error(
                message_id,
                "群授权配置已持久化，但事务资源清理和运行时刷新均失败；"
                "已记录安全阻断项，请修复后重启。",
            )
        elif result.code == "policy_refresh_failed":
            self.reply_error(
                message_id,
                "群授权已写入磁盘，但在线访问策略刷新失败；"
                "当前进程仍拒绝该群，请修复配置后重启。",
            )
        elif result.code == "settings_mirror_failed":
            self.reply_error(
                message_id,
                "群授权配置已写入磁盘，但运行时设置镜像失败；"
                "在线策略保持旧快照并已记录安全阻断项。",
            )
        else:
            self.reply_text(message_id, "当前账号无权授权此群。")

    def _admin_bootstrap_service(self):
        from ...admin_bootstrap import AdminBootstrapService

        return AdminBootstrapService(
            settings_getter=lambda: self.settings,
            policy_provider=self.ctx.ingress_access_policy_provider,
            env_store=self.ctx.ingress_env_store,
        )

    def _handle_btw_command(
        self,
        message_id: str,
        chat_id: str,
        command_match: CommandMatch,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        if not command_match.has_args:
            self.reply_text(message_id, UI_TEXT["system_btw_usage"])
            return

        mode_key = self._resolve_active_programming_mode_key(chat_id, project)
        if not mode_key:
            self.reply_text(message_id, UI_TEXT["system_btw_no_active_session"])
            return

        handler = self.get_handler(mode_key)
        if not handler:
            self.reply_text(message_id, UI_TEXT["system_btw_no_active_session"])
            return

        handler.handle_message(message_id, chat_id, command_match.normalized_text, project)

    def _resolve_active_programming_mode_key(
        self,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
    ) -> Optional[str]:
        from ...mode import InteractionMode
        from ...thread import get_current_thread_id

        programming_modes = {
            InteractionMode.COCO,
            InteractionMode.CLAUDE,
            InteractionMode.AIDEN,
            InteractionMode.CODEX,
            InteractionMode.GEMINI,
            InteractionMode.TRAEX,
            InteractionMode.GROK,
        }
        thread_id = get_current_thread_id()
        if thread_id:
            thread_ctx = self.ctx.thread_manager.get(thread_id)
            if thread_ctx and thread_ctx.mode:
                try:
                    mode = InteractionMode(thread_ctx.mode)
                except ValueError:
                    mode = None
                if mode in programming_modes:
                    return mode.value

        project_id = self._project_id(project)
        mode = self.mode_manager.get_mode(chat_id, project_id=project_id)
        if mode in programming_modes and self.mode_manager.is_programming_mode(
            chat_id,
            project_id=project_id,
        ):
            return mode.value
        return None

    def handle_menu_command(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        msg_type, card_content = SystemBuilder.build_command_menu_card(project)
        self.reply_card(message_id, card_content)

    def handle_help_category(
        self,
        message_id: str,
        chat_id: str,
        category: str,
        project: Optional["ProjectContext"] = None,
        origin_message_id: Optional[str] = None,
    ):
        current_mode = self.mode_manager.get_mode(chat_id)
        current_dir = self.get_working_dir(chat_id)

        # Determine admin status for conditional help content
        is_admin = False
        lock_enabled = True  # F-20: Always show lock section in /help for discoverability
        chat_lock_mgr = getattr(self.ctx, "chat_lock_manager", None)
        if chat_lock_mgr is not None:
            from ...thread import get_current_sender_id
            sender_id = get_current_sender_id() or ""
            if sender_id:
                is_admin = chat_lock_mgr.is_admin(sender_id)

        # FS-09: Inject guidance when ADMIN_USER_IDS is empty
        no_admin_configured = False
        try:
            from ...config import get_settings as _gs
            _settings = _gs()
            no_admin_configured = not _settings.admin_user_ids
        except Exception:
            logger.debug("failed to check admin config", exc_info=True)
            _settings = None

        msg_type, card_content = SystemBuilder.build_help_card(
            project, category, current_dir, current_mode,
            is_admin=is_admin, lock_enabled=lock_enabled, chat_id=chat_id,
            no_admin_configured=no_admin_configured,
        )

        if origin_message_id:
            if self.update_card(origin_message_id, card_content):
                return

        self.reply_card(message_id, card_content)

    def handle_deep_prompt(self, message_id: str, chat_id: str):
        self.reply_text(
            message_id,
            UI_TEXT["system_help_deep_prompt"],
        )

    # ------------------------------------------------------------------
    # ACP command handling
    # ------------------------------------------------------------------
    def _enter_mode_with_acp_model(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        model_name: Optional[str],
        project: Optional["ProjectContext"] = None,
        thread_id: Optional[str] = None,
    ) -> bool:
        target_project = project or self.project_manager.get_active_project(chat_id)

        _TOOL_HANDLER_MAP = [
            ("coco",   "is_coco_mode"),
            ("claude", "is_claude_mode"),
            ("aiden",  "is_aiden_mode"),
            ("codex",  "is_codex_mode"),
            ("gemini", "is_gemini_mode"),
            ("traex", "is_traex_mode"),
            ("grok", "is_grok_mode"),
        ]
        for _tool, _mode_check in _TOOL_HANDLER_MAP:
            if tool_name != _tool:
                continue
            handler = self.get_handler(_tool)
            if not handler:
                break
            if hasattr(handler, "current_model"):
                handler.current_model = model_name
            # If already in this mode, switch model on the active session instead of
            # calling enter_mode() which would return early with an "already in mode" warning.
            _project_id = target_project.project_id if target_project else None
            mode_checker = getattr(self.mode_manager, _mode_check, None)
            if callable(mode_checker) and mode_checker(chat_id, project_id=_project_id) and hasattr(handler, "switch_model"):
                return bool(
                    handler.switch_model(
                        message_id,
                        chat_id,
                        model_name,
                        project=target_project,
                    )
                )
            else:
                # silent=True: model selection card already informs the user, no need for redundant "已开启" notification
                enter_kwargs = {"project": target_project, "silent": True}
                if target_project is not None:
                    enter_kwargs.update(
                        model_override=model_name,
                        commit_project_state=False,
                        activate_mode=False,
                        exit_opposite_mode=False,
                    )
                if thread_id is not None:
                    enter_kwargs["thread_id"] = thread_id
                return bool(handler.enter_mode(message_id, chat_id, **enter_kwargs))

        self.reply_error(message_id, UI_TEXT["system_acp_unsupported_tool"].format(tool_name=tool_name))
        return False

    def _configuration_project(
        self,
        chat_id: str,
        project: Optional["ProjectContext"],
    ) -> Optional["ProjectContext"]:
        if project is not None:
            return project
        active = self.project_manager.get_active_project(chat_id)
        if active is not None:
            return active
        try:
            active, _created = self.project_manager.get_or_create_project_for_path(
                self.get_working_dir(chat_id), chat_id
            )
            return active
        except Exception:
            logger.exception("failed to resolve project for ACP configuration")
            return None

    @staticmethod
    def _model_names(models: list) -> list[str]:
        names: list[str] = []
        for item in models or []:
            variants = getattr(item, "selection_variants", ()) or ()
            for candidate in variants or (item,):
                name = str(
                    candidate.get("name", "")
                    if isinstance(candidate, dict)
                    else getattr(candidate, "name", candidate)
                ).strip()
                if name and name not in names:
                    names.append(name)
        return names

    def handle_acp_command(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Show or atomically update the saved project ACP tool."""
        project = self._configuration_project(chat_id, project)
        names = [
            str(getattr(item, "name", "") or "").strip()
            for item in list_acp_tools()
        ]
        names = [name for name in names if name]
        current = str(getattr(project, "acp_tool_name", "") or "").strip() or "未设置"
        parts = (text or "").strip().split(maxsplit=1)
        requested = parts[1].strip().lower() if len(parts) > 1 else ""
        if not requested:
            self.reply_text(
                message_id,
                UI_TEXT["system_acp_config_summary"].format(
                    current=current,
                    available=" · ".join(f"`{name}`" for name in names) or "无",
                ),
            )
            return
        if requested not in names:
            self.reply_error(message_id, UI_TEXT["system_acp_unsupported_tool"].format(tool_name=requested))
            return
        if project is None:
            self.reply_error(message_id, UI_TEXT["system_acp_config_save_failed"])
            return
        previous_tool = str(getattr(project, "acp_tool_name", "") or "").strip()
        model_name = getattr(project, "acp_model_name", None) if previous_tool == requested else None
        if not self.project_manager.commit_acp_configuration(
            project, tool_name=requested, model_name=model_name
        ):
            self.reply_error(message_id, UI_TEXT["system_acp_config_save_failed"])
            return
        self.reply_text(
            message_id,
            UI_TEXT["system_acp_config_saved"].format(
                tool=requested, model=model_name or "default"
            ),
        )

    def _fetch_acp_models(
        self,
        tool_name: str,
        *,
        cwd: Optional[str] = None,
        current_model: Optional[str] = None,
    ) -> list:
        """Fetch the compact model catalog used by the text command."""
        return fetch_acp_models(tool_name, cwd=cwd, current_model=current_model)

    @staticmethod
    def _selected_option(value: dict) -> Optional[str]:
        """Return the trusted select option injected by the Feishu callback."""
        option = value.get("_option")
        if isinstance(option, dict):
            option = option.get("value")
        selected = str(option or "").strip()
        return selected or None

    def _action_project(
        self,
        chat_id: str,
        project_id: Optional[str],
    ) -> Optional["ProjectContext"]:
        if project_id:
            return self.project_manager.get_project_for_chat(project_id, chat_id)
        return self.project_manager.get_active_project(chat_id)

    def _available_acp_tool_names(self) -> set[str]:
        return {
            name
            for raw_name in get_providers()
            if (name := str(raw_name or "").strip().lower())
        }

    def show_explicit_acp_model_selection(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        project: Optional["ProjectContext"] = None,
        *,
        origin_message_id: Optional[str] = None,
        pending_group: Optional[str] = None,
        pending_profile: Optional[str] = None,
        pending_effort: Optional[str] = None,
        show_loading: bool = True,
    ) -> None:
        """Show the explicit model configuration surface for one backend.

        Task-bearing routes deliberately do not call this method. They keep using
        the saved project selection (or the backend default) and continue
        automatically.
        """
        tool = str(tool_name or "").strip().lower()
        if not tool or tool not in self._available_acp_tool_names():
            self.reply_error(
                message_id,
                UI_TEXT["system_acp_unsupported_tool"].format(tool_name=tool),
            )
            return

        target_project = self._configuration_project(chat_id, project)
        if target_project is None:
            self.reply_error(message_id, UI_TEXT["system_acp_config_save_failed"])
            return
        current_model = (
            getattr(target_project, "acp_model_name", None)
            if getattr(target_project, "acp_tool_name", None) == tool
            else None
        )
        card_message_id = origin_message_id
        if show_loading:
            _msg_type, loading_content = SystemBuilder.build_acp_model_loading_card(
                tool,
                project_id=target_project.project_id,
            )
            if card_message_id:
                if not self.update_card(card_message_id, loading_content):
                    replacement_id = self.reply_card(message_id, loading_content)
                    card_message_id = (
                        replacement_id
                        if isinstance(replacement_id, str) and replacement_id
                        else None
                    )
            else:
                replacement_id = self.reply_card(message_id, loading_content)
                card_message_id = (
                    replacement_id
                    if isinstance(replacement_id, str) and replacement_id
                    else None
                )
        try:
            models = self._fetch_acp_models(
                tool,
                cwd=target_project.root_path,
                current_model=current_model,
            )
        except Exception:
            logger.exception(
                "[ACP] explicit model discovery failed tool=%s project=%s",
                tool,
                target_project.project_id,
            )
            _msg_type, error_content = SystemBuilder.build_acp_model_error_card(
                tool,
                project_id=target_project.project_id,
            )
            if card_message_id and self.update_card(card_message_id, error_content):
                return
            self.reply_card(message_id, error_content)
            return
        _msg_type, card_content = SystemBuilder.build_acp_model_cascade_card(
            models,
            tool,
            project_id=target_project.project_id,
            current_model=current_model,
            pending_group=pending_group,
            pending_profile=pending_profile,
            pending_effort=pending_effort,
        )
        if card_message_id and self.update_card(card_message_id, card_content):
            return
        self.reply_card(message_id, card_content)

    def handle_acp_model_cascade_select(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str],
        value: dict,
    ) -> None:
        """Redraw a selector after one server-recognized cascade dimension."""
        project = self._action_project(chat_id, project_id)
        if project is None:
            self.reply_error(message_id, UI_TEXT["system_acp_config_save_failed"])
            return
        action = str(value.get("action") or "").strip()
        selected = self._selected_option(value)
        if selected is None:
            self.reply_error(message_id, UI_TEXT["system_acp_select_model_prompt"])
            return

        from ...card.actions import dispatch as action_ids

        group = str(value.get("model_group") or "").strip() or None
        profile = str(value.get("model_profile") or "").strip() or None
        effort = str(value.get("model_effort") or "").strip() or None
        if action == action_ids.SELECT_ACP_MODEL_GROUP:
            group, profile, effort = selected, None, None
        elif action == action_ids.SELECT_ACP_MODEL_PROFILE:
            profile, effort = selected, None
        elif action == action_ids.SELECT_ACP_MODEL_EFFORT:
            effort = selected
        else:
            self.reply_error(message_id, UI_TEXT["system_acp_select_model_prompt"])
            return

        self.show_explicit_acp_model_selection(
            message_id,
            chat_id,
            str(value.get("tool_name") or ""),
            project,
            origin_message_id=message_id,
            pending_group=group,
            pending_profile=profile,
            pending_effort=effort,
            show_loading=False,
        )

    def handle_refresh_acp_models(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str],
        value: dict,
    ) -> None:
        """Invalidate the backend catalog before redrawing the explicit picker."""
        project = self._action_project(chat_id, project_id)
        if project is None:
            self.reply_error(message_id, UI_TEXT["system_acp_config_save_failed"])
            return
        tool = str(value.get("tool_name") or "").strip().lower()
        invalidate_acp_model_cache(tool, project.root_path)
        self.show_explicit_acp_model_selection(
            message_id,
            chat_id,
            tool,
            project,
            origin_message_id=message_id,
            pending_group=str(value.get("model_group") or "").strip() or None,
            pending_profile=str(value.get("model_profile") or "").strip() or None,
            pending_effort=str(value.get("model_effort") or "").strip() or None,
        )

    def handle_select_acp_model(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str],
        value: dict,
    ) -> None:
        """Validate a final callback against a freshly probed capability matrix."""
        project = self._action_project(chat_id, project_id)
        if project is None:
            self.reply_error(message_id, UI_TEXT["system_acp_config_save_failed"])
            return
        tool = str(value.get("tool_name") or "").strip().lower()
        if not tool or tool not in self._available_acp_tool_names():
            self.reply_error(
                message_id,
                UI_TEXT["system_acp_unsupported_tool"].format(tool_name=tool),
            )
            return

        invalidate_acp_model_cache(tool, project.root_path)
        models = self._fetch_acp_models(
            tool,
            cwd=project.root_path,
            current_model=(
                getattr(project, "acp_model_name", None)
                if getattr(project, "acp_tool_name", None) == tool
                else None
            ),
        )
        if value.get("use_default_model") is True:
            model_name = None
            model_group = None
            model_profile = None
            model_effort = None
        else:
            group = str(value.get("model_group") or "").strip()
            model_name = compose_model_selection(
                models,
                model=group,
                profile=str(value.get("model_profile") or "").strip() or None,
                effort=str(value.get("model_effort") or "").strip() or None,
            )
            if model_name is None:
                self.reply_error(
                    message_id,
                    UI_TEXT["system_acp_unknown_model"].format(model=group),
                )
                return
            canonical = parse_model_selection(models, model_name)
            if canonical is None:
                self.reply_error(
                    message_id,
                    UI_TEXT["system_acp_unknown_model"].format(model=group),
                )
                return
            model_group = canonical.model
            model_profile = canonical.profile
            model_effort = canonical.effort

        self._activate_acp_selection(
            message_id,
            chat_id,
            tool,
            model_name,
            project,
            explicit_card=True,
            model_group=model_group,
            model_profile=model_profile,
            model_effort=model_effort,
        )

    def handle_enter_acp_saved_selection(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        project: Optional["ProjectContext"],
        *,
        pending_prompt: Optional[str] = None,
    ) -> None:
        """Enter an ACP tool using the project's stored tool/model selection.

        Project chats use this path for free-form messages after a tool has
        already been chosen once, so normal follow-up work does not show the
        model selection card again.
        """
        tool = (tool_name or "").strip().lower()
        if not tool:
            self.reply_error(message_id, UI_TEXT["system_acp_select_tool_prompt"])
            return

        stored_model = None
        if project and getattr(project, "acp_tool_name", "") == tool:
            stored_model = getattr(project, "acp_model_name", None)
        model_name = str(stored_model).strip() if stored_model else None
        self._activate_acp_selection(
            message_id,
            chat_id,
            tool,
            model_name,
            project,
            pending_prompt=pending_prompt,
        )

    def _activate_acp_selection(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        model_name: Optional[str],
        project: Optional["ProjectContext"] = None,
        *,
        pending_prompt: Optional[str] = None,
        explicit_card: bool = False,
        model_group: Optional[str] = None,
        model_profile: Optional[str] = None,
        model_effort: Optional[str] = None,
    ) -> None:
        tool = (tool_name or "").strip().lower()
        use_default_model = model_name is None
        model = None if use_default_model else (model_name or "").strip()
        if not tool or (not use_default_model and not model):
            self.reply_error(message_id, UI_TEXT["system_acp_select_model_prompt"])
            return
        if (
            tool == "claude"
            and model
            and is_1m_variant(model)
            and not model_supports_1m(strip_1m_suffix(model))
        ):
            self.reply_error(
                message_id,
                UI_TEXT["system_acp_unsupported_1m_model"].format(
                    model=strip_1m_suffix(model),
                ),
            )
            return

        target_project = project or self.project_manager.get_active_project(chat_id)
        handler = self.get_handler(tool)

        if explicit_card:
            # Explicit direct-programming configuration is project-persistent.
            # Never inherit a topic coordinate from the card callback worker.
            thread_root_id = None
        else:
            from ...thread import get_current_thread_id

            raw_thread_id = get_current_thread_id()
            thread_root_id = (
                raw_thread_id.strip()
                if isinstance(raw_thread_id, str) and raw_thread_id.strip()
                else None
            )
        project_id = self._project_id(target_project)
        status_message_id = message_id

        def _replace_status_card(card_content: str) -> None:
            nonlocal status_message_id
            if self.update_card(status_message_id, card_content):
                return
            replacement_id = self.reply_card(status_message_id, card_content)
            if isinstance(replacement_id, str) and replacement_id:
                status_message_id = replacement_id

        def _publish_failure(reason: str) -> None:
            failure_reason = reason or UI_TEXT["system_acp_activation_failed_safe"]
            if not explicit_card:
                self.reply_error(message_id, failure_reason)
                return
            _msg_type, failed_content = SystemBuilder.build_acp_programming_failed_card(
                tool,
                model,
                failure_reason,
                project_id,
                None,
                model_group=model_group,
                model_profile=model_profile,
                model_effort=model_effort,
            )
            _replace_status_card(failed_content)

        if explicit_card:
            _msg_type, initializing_content = (
                SystemBuilder.build_acp_programming_initializing_card(
                    tool,
                    model,
                    project_id,
                    None,
                )
            )
            _replace_status_card(initializing_content)

        spec = TaskSpec(
            chat_id=chat_id,
            name="activate_acp_model",
            task_type="acp_model_activation",
            project_id=project_id,
            message_id=message_id,
            origin_message_id=message_id,
            priority=TaskPriority.HIGH,
        )

        def _run_activation(_ctx) -> bool:
            previous_handler_model = (
                handler.current_model
                if handler and hasattr(handler, "current_model")
                else None
            )
            if handler and hasattr(handler, "current_model"):
                # ``enter_mode`` receives the same explicit override.  This
                # assignment keeps existing handler display/context behavior
                # while the explicit input prevents a later selection from
                # changing an already-running callback's startup model.
                handler.current_model = model
            try:
                entered = self._enter_mode_with_acp_model(
                    message_id,
                    chat_id,
                    tool,
                    model,
                    target_project,
                    thread_id=thread_root_id,
                )
            except Exception as exc:
                logger.exception(
                    "[ACP] background model activation failed chat=%s project=%s tool=%s model=%s",
                    chat_id,
                    project_id or "-",
                    tool,
                    model or "<default>",
                )
                if handler and hasattr(handler, "current_model"):
                    handler.current_model = previous_handler_model
                _publish_failure(safe_error_message(exc))
                return False

            if not entered:
                if handler and hasattr(handler, "current_model"):
                    handler.current_model = previous_handler_model
                _publish_failure(UI_TEXT["system_acp_activation_failed_safe"])
                return False

            try:
                if target_project:
                    manager = getattr(handler, "_get_session_manager", lambda: None)()
                    session = (
                        manager.get_session(
                            chat_id,
                            project_id=project_id,
                            thread_id=thread_root_id,
                        )
                        if manager
                        else None
                    )
                    if session is None:
                        if handler and hasattr(handler, "current_model"):
                            handler.current_model = previous_handler_model
                        _publish_failure(UI_TEXT["system_acp_activation_failed_safe"])
                        return False
                    committed = self.project_manager.commit_acp_programming_activation(
                        target_project,
                        tool_name=tool,
                        model_name=model,
                        session_id=session.session_id,
                        query_count=session.message_count,
                        activate_mode=thread_root_id is None,
                    )
                    if not committed:
                        if handler and hasattr(handler, "current_model"):
                            handler.current_model = previous_handler_model
                        _publish_failure(UI_TEXT["system_acp_config_save_failed"])
                        return False
                    if not thread_root_id:
                        # Preserve the normal successful-switch cleanup only after
                        # the project selection is durably committed.
                        handler._exit_opposite_mode(
                            message_id,
                            chat_id,
                            project=target_project,
                            silent=True,
                        )
                        handler._enter_mode_on_manager(chat_id, project_id=project_id)

                if pending_prompt and handler and hasattr(handler, "handle_message"):
                    handler.handle_message(
                        message_id,
                        chat_id,
                        pending_prompt,
                        target_project,
                    )
                if explicit_card:
                    _msg_type, ready_content = SystemBuilder.build_acp_programming_ready_card(
                        tool,
                        model,
                        project_id,
                        None,
                    )
                    _replace_status_card(ready_content)
            except Exception as exc:
                logger.exception(
                    "[ACP] model activation finalization failed chat=%s project=%s tool=%s model=%s",
                    chat_id,
                    project_id or "-",
                    tool,
                    model or "<default>",
                )
                if handler and hasattr(handler, "current_model"):
                    handler.current_model = previous_handler_model
                _publish_failure(safe_error_message(exc))
                return False
            return True

        try:
            self.scheduler.submit(spec, _run_activation)
        except Exception as exc:
            logger.exception(
                "[ACP] failed to schedule model activation chat=%s project=%s tool=%s",
                chat_id,
                project_id or "-",
                tool,
            )
            _publish_failure(safe_error_message(exc))

    # ------------------------------------------------------------------
    # /model command — list/switch models for current ACP tool
    # ------------------------------------------------------------------
    def _resolve_current_acp_tool(self, chat_id: str, project: Optional["ProjectContext"] = None) -> str:
        """Resolve the ACP tool name relevant to the current context.

        Priority:
        1. project.acp_tool_name (explicit tool set on active project)
        2. Current interaction mode (coco/aiden/codex/gemini/claude)
        3. Default: "coco"
        """
        if project and getattr(project, "acp_tool_name", ""):
            from ...project.context import ProjectContext

            tool_name = ProjectContext.normalize_acp_tool_name(
                project.acp_tool_name
            )
            if tool_name:
                project.acp_tool_name = tool_name
                return tool_name
            project.acp_tool_name = None
            project.acp_model_name = None

        mode_to_tool = {
            "coco": "coco",
            "aiden": "aiden",
            "codex": "codex",
            "gemini": "gemini",
            "claude": "claude",
            "traex": "traex",
            "grok": "grok",
        }
        for mode_check, tool in mode_to_tool.items():
            checker = getattr(self.mode_manager, f"is_{mode_check}_mode", None)
            project_id = project.project_id if project else None
            if callable(checker) and checker(chat_id, project_id=project_id):
                return tool

        return "coco"

    def handle_model_command(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Show or atomically update the saved project model."""
        parts = (text or "").strip().split(maxsplit=1)
        requested = parts[1].strip() if len(parts) > 1 else ""
        project = self._configuration_project(chat_id, project)
        tool_name = self._resolve_current_acp_tool(chat_id, project)
        if not requested:
            self.show_explicit_acp_model_selection(
                message_id,
                chat_id,
                tool_name,
                project,
            )
            return
        cwd = (project.root_path if project else None) or self.get_working_dir(chat_id)
        current_model: Optional[str] = None
        if project and getattr(project, "acp_tool_name", "") == tool_name:
            current_model = getattr(project, "acp_model_name", None)
        model_names = self._model_names(
            self._fetch_acp_models(
                tool_name,
                cwd=cwd,
                current_model=current_model,
            )
        )
        model_name = None if requested.lower() == "default" else requested
        if model_name and model_names and model_name not in model_names:
            self.reply_error(
                message_id,
                UI_TEXT["system_acp_unknown_model"].format(model=model_name),
            )
            return
        if (
            tool_name == "claude"
            and model_name
            and is_1m_variant(model_name)
            and not model_supports_1m(strip_1m_suffix(model_name))
        ):
            self.reply_error(
                message_id,
                UI_TEXT["system_acp_unsupported_1m_model"].format(
                    model=strip_1m_suffix(model_name),
                ),
            )
            return
        if project is None or not self.project_manager.commit_acp_configuration(
            project,
            tool_name=tool_name,
            model_name=model_name,
        ):
            self.reply_error(message_id, UI_TEXT["system_acp_config_save_failed"])
            return
        self.reply_text(
            message_id,
            UI_TEXT["system_model_config_saved"].format(
                tool=tool_name,
                model=model_name or "default",
            ),
        )

    # ------------------------------------------------------------------
    # Exit current mode
    # ------------------------------------------------------------------
    def exit_current_mode(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        from ...mode import InteractionMode
        from ...thread import get_current_thread_id, get_thread_manager, set_current_thread_id

        _pid = project.project_id if project else None
        current_mode = self.mode_manager.get_mode(chat_id, project_id=_pid)

        thread_id = get_current_thread_id()
        if thread_id:
            thread_ctx = get_thread_manager().get(thread_id)
            if thread_ctx and thread_ctx.mode in {"deep", "spec", "workflow"}:
                removed = get_thread_manager().remove(thread_ctx.thread_root_id)
                set_current_thread_id(None)
                engine_name = {
                    "deep": "Deep",
                    "spec": "Spec",
                    "workflow": "WF",
                }.get(thread_ctx.mode, thread_ctx.mode)
                if removed:
                    self.reply_text(
                        message_id,
                        UI_TEXT["topic_engine_exit_msg"].format(engine=engine_name),
                    )
                else:
                    self.reply_text(message_id, UI_TEXT["system_already_in_mode"])
                return
            if thread_ctx and thread_ctx.mode != "smart" and current_mode == InteractionMode.SMART:
                try:
                    current_mode = InteractionMode(thread_ctx.mode)
                except ValueError:
                    logger.debug("invalid InteractionMode value: %s", thread_ctx.mode, exc_info=True)

        programming_handler = {
            InteractionMode.COCO: "coco",
            InteractionMode.CLAUDE: "claude",
            InteractionMode.AIDEN: "aiden",
            InteractionMode.CODEX: "codex",
            InteractionMode.GEMINI: "gemini",
            InteractionMode.TRAEX: "traex",
            InteractionMode.GROK: "grok",
        }.get(current_mode)
        if programming_handler:
            self.get_handler(programming_handler).exit_mode(
                message_id,
                chat_id,
                project,
            )
            return
        self.reply_text(message_id, UI_TEXT["system_already_in_mode"])

    # ------------------------------------------------------------------
    # Shell command submission
    # ------------------------------------------------------------------
    def execute_shell_and_reply(
        self,
        message_id: str,
        chat_id: str,
        cmd: str,
        working_dir: Optional[str],
        project: Optional["ProjectContext"] = None,
    ):
        """Execute a shell command via SandboxExecutor and reply with the result."""
        from ...repo_lock import LockConflictError
        from ...sandbox import SandboxExecutor

        if project is not None:
            lock_root_path = getattr(project, "root_path", None) or working_dir
        else:
            lock_root_path = self.lock_helper.resolve_git_lock_root(working_dir)

        def _run_shell():
            executor = SandboxExecutor()
            # Smart mode shell execution: disable interactive mode to avoid .bashrc noise and job control errors
            result = executor.execute(cmd, cwd=working_dir, interactive=False, chat_id=chat_id)
            msg_type, card_content = SystemBuilder.build_shell_result_card(
                cmd,
                result,
                working_dir,
                project,
            )
            self.reply_card(message_id, card_content)
            if result.success:
                self.add_reaction(message_id, EmojiReaction.on_shell_executed())
            else:
                self.add_reaction(message_id, EmojiReaction.on_error())
            return result

        try:
            return self.lock_helper._with_repo_lock_strict(
                lock_root_path,
                chat_id,
                _run_shell,
            )
        except LockConflictError as err:
            self.send_lock_conflict_card(
                err,
                message_id,
                cmd,
                chat_id=chat_id,
            )
            return None

    def submit_shell_command(
        self,
        message_id: str,
        chat_id: str,
        cmd: str,
        working_dir: Optional[str],
        project: Optional["ProjectContext"] = None,
        origin_message_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        project_id = project.project_id if project else None
        origin_message_id = origin_message_id or message_id
        queue_suffix = project_id or (working_dir or "cwd")

        spec = TaskSpec(
            chat_id=chat_id,
            queue_key=f"{chat_id}:shell:{queue_suffix}",
            name="shell_command",
            task_type="shell",
            project_id=project_id,
            message_id=message_id,
            origin_message_id=origin_message_id,
            request_id=request_id,
            priority=TaskPriority.NORMAL,
        )

        def _run(_ctx):
            return self.execute_shell_and_reply(message_id, chat_id, cmd, working_dir, project)

        handle = self.scheduler.submit(spec, _run)
        try:
            self.ctx.message_linker.link_task(origin_message_id, handle.run_id)
        except Exception:
            logger.debug("failed to link task message", exc_info=True)
        return handle

    # ------------------------------------------------------------------
    # Directory change
    # ------------------------------------------------------------------
    def change_directory(self, message_id: str, chat_id: str, path: str, project: Optional["ProjectContext"] = None):
        current_dir = self.get_working_dir(chat_id)

        if not path:
            self.add_reaction(message_id, EmojiReaction.on_dir_changed())
            if project:
                content = ProjectBuilder.build_project_info_content(project, current_dir)
                msg_type, card_content = ProjectBuilder.build_project_response_card(
                    project,
                    UI_TEXT["project_dir_info_title"],
                    content,
                    show_buttons=True,
                )
                response_id = self.reply_card(message_id, card_content)
                if response_id:
                    self.register_message_project(response_id, project)
            else:
                self.reply_text(message_id, fmt.format_current_dir(current_dir))
            return

        success, result = self.set_working_dir(chat_id, path)
        if success:
            self.add_reaction(message_id, EmojiReaction.on_dir_changed())
            card_res = SystemBuilder.build_directory_change_card(project, result, success=True)
            if card_res:
                msg_type, card_content = card_res
                response_id = self.reply_card(message_id, card_content)
                if response_id and project:
                    self.register_message_project(response_id, project)
            else:
                self.reply_text(message_id, fmt.format_dir_change(result, True))
        else:
            self.add_reaction(message_id, EmojiReaction.on_error())
            card_res = SystemBuilder.build_directory_change_card(project, result, success=False)
            if card_res:
                msg_type, card_content = card_res
                self.reply_card(message_id, card_content)
            else:
                self.reply_text(message_id, fmt.format_error(result))

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------
    def show_help(self, message_id: str, chat_id: str):
        project = self.project_manager.get_active_project(chat_id)
        self.show_full_help(message_id, chat_id, project)

    def show_full_help(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        self.handle_help_category(message_id, chat_id, "main", project)

    def show_coco_status(self, message_id: str, chat_id: str):
        manager = get_coco_model_manager()
        current_model = manager.get_current_model()
        models = manager.get_models().models

        content = SystemBuilder.build_coco_status_content(current_model, models)
        self.reply_text(message_id, content)

    def show_tools_list(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Show a list of all available ACP tools with quick access buttons."""
        # Define tool names
        names = ["coco", "claude", "aiden", "codex", "gemini", "traex", "grok"]
        emojis = {
            "coco": "🤖",
            "claude": "🔮",
            "aiden": "🎯",
            "codex": "💻",
            "gemini": "✨",
            "traex": "🚀",
            "grok": "🌌",
        }

        # Cached-first availability check: avoid blocking user-path on external probe.
        tools = []
        for name in names:
            is_available = is_programming_tool_available(
                name,
                allow_sync_probe=False,
                trigger_async_probe=True,
            )
            desc = UI_TEXT[f"system_acp_tool_desc_{name}"]
            tools.append(
                {
                    "name": name,
                    "emoji": emojis.get(name, "🤖"),
                    "description": desc,
                    "available": is_available,
                }
            )

        msg_type, card = SystemBuilder.build_tools_list_card(tools, project)
        self.reply_interactive_card(message_id, card, msg_type=msg_type)

    def show_tools_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Show detailed status of all tools with availability and session info."""
        # Define tool metadata
        tool_defs = [
            {"name": "coco", "emoji": "🤖", "manager": self.ctx.coco_manager},
            {"name": "claude", "emoji": "🔮", "manager": self.ctx.claude_manager},
            {"name": "aiden", "emoji": "🎯", "manager": self.ctx.aiden_manager},
            {"name": "codex", "emoji": "💻", "manager": self.ctx.codex_manager},
            {"name": "gemini", "emoji": "✨", "manager": self.ctx.gemini_manager},
            {"name": "traex", "emoji": "🚀", "manager": self.ctx.traex_manager},
            {"name": "grok", "emoji": "🌌", "manager": self.ctx.grok_manager},
        ]

        def _format_last_used(ts: float) -> str:
            """格式化最近使用时间，基于共享 TimeAgo 语义层。

            语义边界（秒 → bucket）交给 ``compute_time_ago_bucket`` 处理，
            本函数只负责结合现有 UI_TEXT 模板渲染具体文案，以保持系统
            状态卡片的既有风格。
            """

            try:
                raw_ts = float(ts or 0.0)
            except Exception:
                return UI_TEXT["system_unknown"]

            if raw_ts <= 0.0:
                return UI_TEXT["system_never_used"]

            try:
                idle_seconds = max(0, int(time.time() - raw_ts))
            except Exception:
                return UI_TEXT["system_unknown"]

            from src.utils.time_ago import compute_time_ago_bucket

            bucket = compute_time_ago_bucket(idle_seconds)
            kind = bucket["kind"]
            value = int(bucket["value"])

            # seconds 区间：保持原有「X 秒前」样式（使用实际 idle 秒数）
            if kind == "seconds":
                return UI_TEXT["time_secs_ago"].format(seconds=idle_seconds)

            # minutes 区间：使用 bucket 的分钟值 + 余下秒数，保留原有模板
            if kind == "minutes":
                m = value
                s = max(0, idle_seconds - m * 60)
                return UI_TEXT["time_mins_secs_ago"].format(minutes=m, seconds=s)

            # hours/days 统归为「X 小时 Y 分钟前」风格，避免新增文案 key
            total_minutes = idle_seconds // 60
            h, m = divmod(total_minutes, 60)
            return UI_TEXT["time_hours_mins_ago"].format(hours=h, minutes=m)

        # Gather availability + real session activity from ACP managers.
        tools = []
        active_sessions: dict[str, dict] = {}
        for meta in tool_defs:
            name = meta["name"]
            manager = meta["manager"]
            is_available = is_programming_tool_available(
                name,
                allow_sync_probe=False,
                trigger_async_probe=True,
            )

            sessions = []
            try:
                sessions = manager.list_active_sessions(chat_id=chat_id)
            except Exception:
                sessions = []

            last_active_ts = 0.0
            if sessions:
                try:
                    last_active_ts = max(float(s.get("last_active", 0.0) or 0.0) for s in sessions)
                except Exception:
                    last_active_ts = 0.0

            tools.append(
                {
                    "name": name,
                    "emoji": meta["emoji"],
                    "available": is_available,
                    "last_used": _format_last_used(last_active_ts),
                }
            )
            if sessions:
                # Card expects one active summary line; provide latest session in that tool.
                latest = None
                try:
                    latest = max(sessions, key=lambda s: float(s.get("last_active", 0.0) or 0.0))
                except Exception:
                    latest = sessions[0]
                if latest:
                    # chat_id 由 ACPSessionManager.list_active_sessions 统一解析并暴露，避免外部再做手工 split
                    session_chat_id = str(latest.get("chat_id") or "") or "N/A"
                    active_sessions[name] = {
                        "chat_id": session_chat_id,
                        "session_id": str(latest.get("session_id", "") or ""),
                        "message_count": int(latest.get("message_count", 0) or 0),
                    }

        msg_type, card = SystemBuilder.build_tools_status_card(tools, active_sessions, project)
        self.reply_interactive_card(message_id, card, msg_type=msg_type)
