"""System handler — help, exit mode, shell commands, directory switching, intercepted commands."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ...acp.helper import (
    fetch_acp_models,
    invalidate_acp_model_cache,
    list_acp_tools,
)
from ...acp.providers import tool_registry
from ...card import CardBuilder
from ...card.actions import dispatch as action_ids
from ...card.builders.project import ProjectBuilder
from ...card.ui_text import UI_TEXT
from ...coco_model import get_coco_model_manager
from ...tasking import TaskPriority, TaskSpec
from ...utils.errors import safe_error_message
from ..emoji import EmojiReaction
from ..message_formatter import FeishuMessageFormatter as fmt
from ..product_catalog import retired_command_tokens
from ..slash_command_parser import CommandMatch, SlashCommandParser
from .base import BaseHandler
from .lock_commands import LockCommandsMixin
from .ttadk_commands import TTADKCommandsMixin

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


class SystemHandler(LockCommandsMixin, TTADKCommandsMixin, BaseHandler):
    """Help, exit, shell, directory, and intercepted-command handling."""

    # `/status` remains the established Deep/Spec diagnostics command. The
    # Standalone v5 Manager was superseded by the Journal-backed employee/Slock
    # runtime. Keep its old spellings fail-closed with an explicit migration
    # message instead of reviving the obsolete process-local command surface.
    _RETIRED_AUTONOMOUS_MANAGER_COMMANDS = retired_command_tokens()

    def __init__(self, ctx: "HandlerContext") -> None:
        super().__init__(ctx)
        self._init_command_registry()
        self._ttadk_flow_start_times: dict[str, float] = {}
        self._ttadk_flow_last_duration_ms: OrderedDict[str, int] = OrderedDict()
        self._TTADK_FLOW_DURATION_MAX_SIZE = 200
        # Pending programming prompts stashed when showing the ACP model-select
        # card. After the user picks a model and enters the mode, the stashed
        # prompt is forwarded to the mode handler as the first requirement.
        # Keyed by f"{chat_id}:{tool_name}".
        self._pending_prompts: "OrderedDict[str, str]" = OrderedDict()
        self._PENDING_PROMPTS_MAX_SIZE = 256
        # ACP model cards can be answered out of order.  Ownership is scoped
        # to this handler request, never inferred from mutable project choice.
        self._acp_activation_lock = threading.RLock()
        self._acp_activation_tokens: dict[str, object] = {}
        self.help_commands = _SystemSubcommands(self, ("show_help", "show_full_help", "handle_help_category", "handle_menu_command"))
        self.shell_commands = _SystemSubcommands(self, ("submit_shell_command", "execute_shell_and_reply", "change_directory"))
        self.acp_commands = _SystemSubcommands(self, ("handle_acp_command", "handle_select_acp_tool", "handle_select_acp_model", "handle_refresh_acp_models", "handle_model_command"))
        self.ttadk_commands = _SystemSubcommands(self, ("handle_ttadk_command", "handle_select_ttadk_tool", "handle_select_ttadk_model", "handle_refresh_ttadk_models", "handle_toggle_ttadk_yolo", "handle_select_ttadk_combined", "handle_select_ttadk_combined_tool"))
        self.lock_commands = _SystemSubcommands(self, ("handle_force_release_repo_lock", "handle_confirm_lock", "handle_cancel_lock", "handle_confirm_force_release", "handle_cancel_force_release"))

    @staticmethod
    def _project_id(project: Optional["ProjectContext"]) -> Optional[str]:
        project_id = getattr(project, "project_id", None) if project else None
        return project_id if isinstance(project_id, str) else None

    @staticmethod
    def _pending_prompt_key(chat_id: str, tool_name: str) -> str:
        return f"{chat_id}:{(tool_name or '').lower()}"

    def _stash_pending_prompt(self, chat_id: str, tool_name: str, prompt: str) -> None:
        if not prompt or not chat_id or not tool_name:
            return
        key = self._pending_prompt_key(chat_id, tool_name)
        self._pending_prompts[key] = prompt
        self._pending_prompts.move_to_end(key)
        while len(self._pending_prompts) > self._PENDING_PROMPTS_MAX_SIZE:
            self._pending_prompts.popitem(last=False)

    def _pop_pending_prompt(self, chat_id: str, tool_name: str) -> Optional[str]:
        if not chat_id or not tool_name:
            return None
        key = self._pending_prompt_key(chat_id, tool_name)
        return self._pending_prompts.pop(key, None)

    @staticmethod
    def _acp_activation_scope(chat_id: str, project_id: Optional[str]) -> str:
        return f"{chat_id}:{project_id or '-'}"

    def _claim_acp_activation(self, chat_id: str, project_id: Optional[str]) -> tuple[str, object]:
        scope = self._acp_activation_scope(chat_id, project_id)
        token = object()
        with self._acp_activation_lock:
            self._acp_activation_tokens[scope] = token
        return scope, token

    def _owns_acp_activation(self, scope: str, token: object) -> bool:
        with self._acp_activation_lock:
            return self._acp_activation_tokens.get(scope) is token

    def _release_acp_activation(self, scope: str, token: object) -> None:
        with self._acp_activation_lock:
            if self._acp_activation_tokens.get(scope) is token:
                del self._acp_activation_tokens[scope]

    def _commit_acp_activation_if_owned(
        self,
        scope: str,
        token: object,
        project: "ProjectContext",
        *,
        tool_name: str,
        model_name: Optional[str],
        session_id: str,
        query_count: int,
        activate_mode: bool,
    ) -> bool:
        """CAS the request token and project commit under one lock boundary."""
        with self._acp_activation_lock:
            if self._acp_activation_tokens.get(scope) is not token:
                return False
            return bool(self.project_manager.commit_acp_programming_activation(
                project,
                tool_name=tool_name,
                model_name=model_name,
                session_id=session_id,
                query_count=query_count,
                activate_mode=activate_mode,
            ))

    def _init_command_registry(self):
        """Initialize the command dispatch registry."""
        # Exact match handlers: command -> handler_func(message_id, chat_id, text, project)
        self._exact_handlers = {
            "/help": lambda m, c, t, p: self.show_full_help(m, c, p),
            "/帮助": lambda m, c, t, p: self.show_full_help(m, c, p),
            "/coco": lambda m, c, t, p: self.handle_select_acp_tool(m, c, "coco", project_id=self._project_id(p)),
            "/enter_coco": lambda m, c, t, p: self.handle_select_acp_tool(m, c, "coco", project_id=self._project_id(p)),
            "/claude": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "claude", p),
            "/enter_claude": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "claude", p),
            "/aiden": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "aiden", p),
            "/enter_aiden": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "aiden", p),
            "/codex": lambda m, c, t, p: self.handle_select_acp_tool(m, c, "codex", project_id=self._project_id(p)),
            "/enter_codex": lambda m, c, t, p: self.handle_select_acp_tool(m, c, "codex", project_id=self._project_id(p)),
            "/gemini": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "gemini", p),
            "/enter_gemini": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "gemini", p),
            "/traex": lambda m, c, t, p: self.handle_select_acp_tool(m, c, "traex", project_id=self._project_id(p)),
            "/enter_traex": lambda m, c, t, p: self.handle_select_acp_tool(m, c, "traex", project_id=self._project_id(p)),
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
            "/end_ttadk": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_ttadk": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/coco_status": lambda m, c, t, p: self.show_coco_status(m, c),
            "/coco_info": lambda m, c, t, p: self.get_handler("coco").show_info(m, c, p),
            "/claude_info": lambda m, c, t, p: self.get_handler("claude").show_info(m, c, p),
            "/aiden_info": lambda m, c, t, p: self.get_handler("aiden").show_info(m, c, p),
            "/codex_info": lambda m, c, t, p: self.get_handler("codex").show_info(m, c, p),
            "/gemini_info": lambda m, c, t, p: self.get_handler("gemini").show_info(m, c, p),
            "/traex_info": lambda m, c, t, p: self.get_handler("traex").show_info(m, c, p),
            "/projects": lambda m, c, t, p: self.get_handler("project").show_project_board(m, c),
            "/project": lambda m, c, t, p: self.get_handler("project").show_project_board(m, c),
            "/switch": lambda m, c, t, p: self.get_handler("project").show_project_board(m, c),
            "/new-chat": lambda m, c, t, p: self._handle_new_chat_project_args(m, c, ""),
            "/ttadk": lambda m, c, t, p: self.handle_ttadk_command(m, c, p),
            "/enter_ttadk": lambda m, c, t, p: self.handle_ttadk_command(m, c, p),
            "/tui2acp": lambda m, c, t, p: self.handle_tui2acp_command(m, c, p),
            "/enter_tui2acp": lambda m, c, t, p: self.handle_tui2acp_command(m, c, p),
            "/exit_tui2acp": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_tui2acp": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/tui2acp_info": lambda m, c, t, p: self.get_handler("tui2acp").show_info(m, c, p),
            "/acp": lambda m, c, t, p: self.handle_acp_command(m, c, p),
            # Worktree: canonical command is /worktree (aliases like /wt are normalized by SlashCommandParser)
            "/worktree": lambda m, c, t, p: self.get_handler("worktree").handle_worktree_command(m, c, p),
            "/ttadk_info": lambda m, c, t, p: self.show_ttadk_info(m, c),
            "/ttadk_refresh": lambda m, c, t, p: self.refresh_ttadk_models(m, c, p),
            "/menu": lambda m, c, t, p: self.handle_menu_command(m, c, p),
            "/tools": lambda m, c, t, p: self.show_tools_list(m, c, p),
            "/tools_status": lambda m, c, t, p: self.show_tools_status(m, c, p),
            "/model": lambda m, c, t, p: self.handle_model_command(m, c, t, p),
            "/lock": lambda m, c, t, p: self._handle_lock_command(m, c, "lock"),
            "/unlock": lambda m, c, t, p: self._handle_lock_command(m, c, "unlock"),
            "/setadmin": lambda m, c, t, p: self._handle_setadmin_command(m, c, ""),
            "/access": lambda m, c, t, p: self._handle_access_command(m, c, ""),
        }

        # Prefix match handlers: prefix -> handler_func(message_id, chat_id, text, project)
        # Note: Order matters if prefixes overlap (not the case here yet)
        self._prefix_handlers = [
            ("/status", lambda m, c, t, p: self.get_handler("diagnostics").show_unified_status(m, c, t, p)),
            ("/tasks", lambda m, c, t, p: self.get_handler("diagnostics").show_task_board(m, c, t, p)),
            ("/diff", lambda m, c, t, p: self.get_handler("diagnostics").show_context_diff(m, c, t, p)),
            ("/trace", lambda m, c, t, p: self.get_handler("diagnostics").show_message_trace(m, c, t, p)),
            ("/model", self.handle_model_command),
        ]

    def _handle_direct_mode_enter(
        self,
        message_id: str,
        chat_id: str,
        mode_key: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        handler = self.get_handler(mode_key)
        if not handler:
            self.reply_error(
                message_id,
                UI_TEXT["system_acp_unsupported_tool"].format(tool_name=mode_key),
                title=UI_TEXT["system_internal_error"],
            )
            return
        handler.enter_mode(message_id, chat_id, project=project)

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
            "/end_ttadk",
            "/exit_ttadk",
        }
        exit_keywords = {
            "退出模式",
            "退出编程模式",
            "退出编程",
            "结束编程",
            "退出claude",
            "退出coco",
            "退出aiden",
            "退出codex",
            "退出gemini",
            "退出traex",
            "退出ttadk",
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
            "/spec_pause",
            "/spec_resume",
            "/spec_recover",
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
            "/enter_ttadk",
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
            "/end_ttadk",
            "/exit_ttadk",
            "/tui2acp",
            "/enter_tui2acp",
            "/exit_tui2acp",
            "/end_tui2acp",
            "/tui2acp_info",
            "/coco_status",
            "/coco_info",
            "/claude_info",
            "/aiden_info",
            "/codex_info",
            "/gemini_info",
            "/traex_info",
            "/ttadk_info",
            "/projects",
            "/status",
            "/project",
            "/switch",
            "/new-chat",
            "/tasks",
            "/diff",
            "/trace",
            "/ttadk",
            "/acp",
            "/worktree",
            "/ttadk_refresh",
            "/menu",
            "/tools",
            "/tools_status",
            "/model",
            "/lock",
            "/unlock",
            "/setadmin",
            "/access",
            "/btw",
            "/goals",
            "/runs",
        }
        exact_commands.update(SystemHandler._RETIRED_AUTONOMOUS_MANAGER_COMMANDS)
        if not m.has_args and cmd in exact_commands:
            return True
        prefix_commands = {
            "/worktree",
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
        # Use canonical command as routing key, but keep original text for handlers
        # that still need it for legacy parsing (non-worktree).
        text_lower = m.command

        # Worktree is special: route directly with parsed goal to avoid handler-side slicing.
        if text_lower == "/worktree" and m.has_args:
            wt = self.get_handler("worktree")
            if wt:
                # Prefer passing the parsed CommandMatch through the chain.
                if hasattr(wt, "handle_worktree_command_match"):
                    wt.handle_worktree_command_match(message_id, chat_id, m, project=project)
                else:
                    wt.handle_worktree_command(message_id, chat_id, project, goal=m.args)
                return

        # ACP model-select mode enter: /codex, /traex, ...
        if not m.has_args and text_lower in {"/codex", "/enter_codex", "/traex", "/enter_traex"}:
            _pid = self._project_id(project)
            tool = text_lower.rsplit("_", 1)[-1].lstrip("/") if text_lower.startswith("/enter_") else text_lower.lstrip("/")
            self.handle_select_acp_tool(message_id, chat_id, tool, project_id=_pid)
            return

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

        if (args or "").strip().casefold() != "allow-chat":
            self.reply_text(
                message_id,
                "用法：请由管理员在目标群内发送 `/access allow-chat`。",
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
            InteractionMode.TTADK,
            InteractionMode.TUI2ACP,
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
        msg_type, card_content = CardBuilder.build_command_menu_card(project)
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

        msg_type, card_content = CardBuilder.build_help_card(
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

    def handle_acp_command(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        project_id = project.project_id if project else None
        current_tool = project.acp_tool_name if project else None
        tools = list_acp_tools()
        if not tools:
            self.reply_error(message_id, UI_TEXT["system_acp_no_available_tools"])
            return
        msg_type, card_content = CardBuilder.build_acp_tool_select_card(tools, project_id, current_tool=current_tool)
        self.reply_card(message_id, card_content)

    def _fetch_acp_models(
        self,
        tool_name: str,
        *,
        cwd: Optional[str] = None,
        current_model: Optional[str] = None,
    ) -> list:
        """Thin wrapper around ``fetch_acp_models`` for easier testing.

        Unit tests can monkeypatch this method on ``SystemHandler`` instances
        without needing to stub the global helper import.
        """
        return fetch_acp_models(tool_name, cwd=cwd, current_model=current_model)

    def _show_acp_model_selection_flow(
        self,
        *,
        message_id: str,
        chat_id: str,
        tool: str,
        project_id: Optional[str],
        cwd: str,
        current_model: Optional[str],
        thread_root_id: Optional[str],
        update_existing: bool = False,
        model_page: int = 0,
    ) -> None:
        """Show ACP model discovery and selection as one progressively patched card."""
        _, loading_card = CardBuilder.build_acp_model_loading_card(
            tool,
            project_id=project_id,
            thread_root_id=thread_root_id,
        )
        progress_message_id: Optional[str] = None
        if update_existing:
            if self.update_card(message_id, loading_card):
                progress_message_id = message_id
            else:
                progress_message_id = self.reply_card(message_id, loading_card)
        else:
            progress_message_id = self.reply_card(message_id, loading_card)

        models = self._fetch_acp_models(tool, cwd=cwd, current_model=current_model)
        if not models:
            _, error_card = CardBuilder.build_acp_model_error_card(
                tool,
                project_id=project_id,
                thread_root_id=thread_root_id,
            )
            if progress_message_id and self.update_card(progress_message_id, error_card):
                return
            self.reply_error(message_id, UI_TEXT["system_acp_get_models_failed"].format(tool_name=tool))
            return

        _, model_card = CardBuilder.build_acp_model_cascade_card(
            models,
            tool,
            project_id,
            current_model=current_model,
            thread_root_id=thread_root_id,
        )
        if progress_message_id and self.update_card(progress_message_id, model_card):
            return
        self.reply_card(message_id, model_card)

    def handle_select_acp_tool(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        project_id: Optional[str] = None,
        *,
        pending_prompt: Optional[str] = None,
    ):
        tool = (tool_name or "").strip().lower()
        if not tool:
            self.reply_error(message_id, UI_TEXT["system_acp_select_tool_prompt"])
            return

        project = self.project_manager.get_project_for_chat(project_id, chat_id) if project_id else self.project_manager.get_active_project(chat_id)
        cwd = (project.root_path if project else None) or self.get_working_dir(chat_id)

        current_model = None
        if project and getattr(project, "acp_tool_name", "") == tool:
            current_model = getattr(project, "acp_model_name", None)

        logger.info("[ACP] selecting tool=%s project_id=%s cwd=%s", tool, project_id or "-", cwd)

        # Stash the pending prompt (if provided) so that after the user picks a
        # model, the mode handler picks it up as the first programming request.
        if pending_prompt:
            self._stash_pending_prompt(chat_id, tool, pending_prompt)

        from ...thread import get_current_thread_id
        self._show_acp_model_selection_flow(
            message_id=message_id,
            chat_id=chat_id,
            tool=tool,
            project_id=project_id,
            cwd=cwd,
            current_model=current_model,
            thread_root_id=get_current_thread_id(),
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
        if pending_prompt:
            self._stash_pending_prompt(chat_id, tool, pending_prompt)
        self.handle_select_acp_model(message_id, chat_id, tool, model_name, project)

    def handle_refresh_acp_models(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        project_id: Optional[str] = None,
        value: Optional[dict] = None,
    ):
        tool = (tool_name or "").strip().lower()
        if not tool:
            self.reply_error(message_id, UI_TEXT["system_acp_select_tool_prompt"])
            return
        project = self.project_manager.get_project_for_chat(project_id, chat_id) if project_id else self.project_manager.get_active_project(chat_id)
        cwd = (project.root_path if project else None) or self.get_working_dir(chat_id)
        current_model = None
        if project and getattr(project, "acp_tool_name", "") == tool:
            current_model = getattr(project, "acp_model_name", None)
        from ...thread import get_current_thread_id
        thread_root_id = str((value or {}).get("thread_root_id") or "").strip() or get_current_thread_id()
        try:
            model_page = int((value or {}).get("model_page", 0) or 0)
        except (TypeError, ValueError):
            model_page = 0
        if "model_page" not in (value or {}):
            invalidate_acp_model_cache(tool, cwd)
        self._show_acp_model_selection_flow(
            message_id=message_id,
            chat_id=chat_id,
            tool=tool,
            project_id=project_id,
            cwd=cwd,
            current_model=current_model,
            thread_root_id=thread_root_id,
            update_existing=True,
            model_page=model_page,
        )

    def handle_acp_model_cascade_select(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        project_id: Optional[str] = None,
        value: Optional[dict] = None,
    ) -> None:
        """Re-render the ACP model cascade after a family/profile/effort change.

        This only updates the pending dropdown selection and repaints the card;
        it does NOT enter the mode. The user must click the final "confirm"
        button (``SELECT_ACP_MODEL``) to actually commit the model choice.

        The pending group/profile/effort are carried statelessly in the button
        ``value`` (plus the Feishu-injected ``_option`` for the just-changed
        dropdown), so no per-chat selection state is stored.
        """
        value = value or {}
        tool = (tool_name or "").strip().lower()
        if not tool:
            self.reply_error(message_id, UI_TEXT["system_acp_select_tool_prompt"])
            return

        action = str(value.get("action") or "")
        selected = str(value.get("_option") or "").strip()

        pending_group = value.get("model_group")
        pending_profile = value.get("model_profile")
        pending_effort = value.get("model_effort")
        # Apply the just-changed dropdown's new value. Downstream dropdowns
        # reset so the cascade recomputes their defaults from the new parent.
        if action == action_ids.SELECT_ACP_MODEL_GROUP:
            pending_group = selected or pending_group
            pending_profile = None
            pending_effort = None
        elif action == action_ids.SELECT_ACP_MODEL_PROFILE:
            pending_profile = selected or pending_profile
            pending_effort = None
        elif action == action_ids.SELECT_ACP_MODEL_EFFORT:
            pending_effort = selected or pending_effort

        project = (
            self.project_manager.get_project_for_chat(project_id, chat_id)
            if project_id
            else self.project_manager.get_active_project(chat_id)
        )
        cwd = (project.root_path if project else None) or self.get_working_dir(chat_id)
        current_model = None
        if project and getattr(project, "acp_tool_name", "") == tool:
            current_model = getattr(project, "acp_model_name", None)

        from ...thread import get_current_thread_id
        thread_root_id = str(value.get("thread_root_id") or "").strip() or get_current_thread_id()

        models = self._fetch_acp_models(tool, cwd=cwd, current_model=current_model)
        if not models:
            _, error_card = CardBuilder.build_acp_model_error_card(
                tool, project_id=project_id, thread_root_id=thread_root_id
            )
            self.update_card(message_id, error_card)
            return

        _, model_card = CardBuilder.build_acp_model_cascade_card(
            models,
            tool,
            project_id,
            current_model=current_model,
            thread_root_id=thread_root_id,
            pending_group=pending_group,
            pending_profile=pending_profile,
            pending_effort=pending_effort,
        )
        self.update_card(message_id, model_card)

    def handle_select_acp_model(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        model_name: Optional[str],
        project: Optional["ProjectContext"] = None,
        *,
        run_immediately: bool = False,
    ) -> None:
        tool = (tool_name or "").strip().lower()
        use_default_model = model_name is None
        model = None if use_default_model else (model_name or "").strip()
        if not tool or (not use_default_model and not model):
            self.reply_error(message_id, UI_TEXT["system_acp_select_model_prompt"])
            return

        target_project = project or self.project_manager.get_active_project(chat_id)
        handler = self.get_handler(tool)

        from ...thread import get_current_thread_id

        raw_thread_id = get_current_thread_id()
        thread_root_id = raw_thread_id.strip() if isinstance(raw_thread_id, str) and raw_thread_id.strip() else None
        project_id = self._project_id(target_project)
        activation_scope, activation_token = self._claim_acp_activation(
            chat_id,
            project_id,
        )
        _, initializing_card = CardBuilder.build_acp_programming_initializing_card(
            tool,
            model,
            project_id=project_id,
            thread_root_id=thread_root_id,
        )
        self.update_card(message_id, initializing_card)

        spec = TaskSpec(
            chat_id=chat_id,
            name="activate_acp_model",
            task_type="acp_model_activation",
            project_id=project_id,
            message_id=message_id,
            origin_message_id=message_id,
            priority=TaskPriority.HIGH,
        )

        def _run_activation_impl(_ctx) -> bool:
            if not self._owns_acp_activation(activation_scope, activation_token):
                logger.info(
                    "[ACP] skip stale model activation chat=%s project=%s tool=%s model=%s",
                    chat_id,
                    project_id or "-",
                    tool,
                    model or "<default>",
                )
                self._release_acp_activation(activation_scope, activation_token)
                return False

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
                failure_reason = UI_TEXT["system_acp_activation_failed_safe"]
            except Exception as exc:
                logger.exception(
                    "[ACP] background model activation failed chat=%s project=%s tool=%s model=%s",
                    chat_id,
                    project_id or "-",
                    tool,
                    model or "<default>",
                )
                entered = False
                failure_reason = safe_error_message(exc)

            # A later selection owns the card and project state. Never let this
            # older task overwrite it or forward a prompt under the wrong model.
            if not self._owns_acp_activation(activation_scope, activation_token):
                self._release_acp_activation(activation_scope, activation_token)
                return False

            if not entered:
                if handler and hasattr(handler, "current_model"):
                    handler.current_model = previous_handler_model
                _, failed_card = CardBuilder.build_acp_programming_failed_card(
                    tool,
                    model,
                    failure_reason,
                    project_id=project_id,
                    thread_root_id=thread_root_id,
                )
                self.update_card(message_id, failed_card)
                self._release_acp_activation(activation_scope, activation_token)
                return False

            if target_project:
                manager = getattr(handler, "_get_session_manager", lambda: None)()
                session = manager.get_session(
                    chat_id,
                    project_id=project_id,
                    thread_id=thread_root_id,
                ) if manager else None
                if session is None:
                    if handler and hasattr(handler, "current_model"):
                        handler.current_model = previous_handler_model
                    self._release_acp_activation(activation_scope, activation_token)
                    return False
                try:
                    committed = self._commit_acp_activation_if_owned(
                        activation_scope,
                        activation_token,
                        target_project,
                        tool_name=tool,
                        model_name=model,
                        session_id=session.session_id,
                        query_count=session.message_count,
                        activate_mode=thread_root_id is None,
                    )
                except Exception:
                    self._release_acp_activation(activation_scope, activation_token)
                    raise
                if not committed:
                    if handler and hasattr(handler, "current_model"):
                        handler.current_model = previous_handler_model
                    self._release_acp_activation(activation_scope, activation_token)
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

            _, ready_card = CardBuilder.build_acp_programming_ready_card(
                tool,
                model,
                project_id=project_id,
                thread_root_id=thread_root_id,
            )
            self.update_card(message_id, ready_card)

            # Project-chat selection may carry the user's first requirement.
            # The activation task owns the popped value and forwards it once,
            # only after the selected session is actually ready.
            pending = self._pop_pending_prompt(chat_id, tool)
            if pending and handler and hasattr(handler, "handle_message"):
                try:
                    handler.handle_message(
                        message_id,
                        chat_id,
                        pending,
                        target_project,
                    )
                except Exception:
                    logger.exception("forwarding pending prompt failed after ACP activation")
            self._release_acp_activation(activation_scope, activation_token)
            return True

        def _run_activation(_ctx) -> bool:
            try:
                return _run_activation_impl(_ctx)
            finally:
                self._release_acp_activation(activation_scope, activation_token)

        if run_immediately:
            _run_activation(None)
            return

        try:
            self.scheduler.submit(spec, _run_activation)
        except Exception as exc:
            logger.exception(
                "[ACP] failed to schedule model activation chat=%s project=%s tool=%s",
                chat_id,
                project_id or "-",
                tool,
            )
            _, failed_card = CardBuilder.build_acp_programming_failed_card(
                tool,
                model,
                safe_error_message(exc),
                project_id=project_id,
                thread_root_id=thread_root_id,
            )
            self.update_card(message_id, failed_card)
            self._release_acp_activation(activation_scope, activation_token)

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
            return str(project.acp_tool_name).lower()

        mode_to_tool = {
            "coco": "coco",
            "aiden": "aiden",
            "codex": "codex",
            "gemini": "gemini",
            "claude": "claude",
            "traex": "traex",
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
        """Handle /model [list|<name>|switch <name>] command.

        /model              — show model selection card for current ACP tool
        /model list         — same as above
        /model <name>       — switch directly to <name>
        /model switch <name>— same as /model <name>
        """
        text_stripped = (text or "").strip()
        parts = text_stripped.split(None, 2)
        # parts[0] == "/model" (case-insensitive)
        subcommand = parts[1].lower() if len(parts) >= 2 else ""

        # Resolve project if not provided
        if project is None:
            project = self.project_manager.get_active_project(chat_id)

        tool_name = self._resolve_current_acp_tool(chat_id, project)
        cwd = (project.root_path if project else None) or self.get_working_dir(chat_id)
        project_id = project.project_id if project else None

        current_model: Optional[str] = None
        if project and getattr(project, "acp_tool_name", "") == tool_name:
            current_model = getattr(project, "acp_model_name", None)

        if subcommand in ("", "list", "ls"):
            # Show interactive model selection card
            from ...thread import get_current_thread_id
            self._show_acp_model_selection_flow(
                message_id=message_id,
                chat_id=chat_id,
                tool=tool_name,
                project_id=project_id,
                cwd=cwd,
                current_model=current_model,
                thread_root_id=get_current_thread_id(),
            )
            return

        # Direct switch: /model <name> or /model switch <name>
        if subcommand == "switch":
            model_name = parts[2].strip() if len(parts) >= 3 else ""
        else:
            model_name = parts[1].strip() if len(parts) >= 2 else ""

        if not model_name:
            self.reply_error(
                message_id,
                UI_TEXT["system_acp_specify_model_prompt"].format(
                    example=UI_TEXT["system_model_usage_example"]
                ),
            )
            return

        self.handle_select_acp_model(
            message_id,
            chat_id,
            tool_name,
            model_name,
            project,
            run_immediately=True,
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
            if thread_ctx and thread_ctx.mode in {"worktree", "deep", "spec", "workflow"}:
                removed = get_thread_manager().remove(thread_ctx.thread_root_id)
                set_current_thread_id(None)
                engine_name = {
                    "worktree": "WT",
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

        if current_mode == InteractionMode.COCO:
            self.get_handler("coco").exit_mode(message_id, chat_id, project)
        elif current_mode == InteractionMode.CLAUDE:
            self.get_handler("claude").exit_mode(message_id, chat_id, project)
        elif current_mode == InteractionMode.AIDEN:
            self.get_handler("aiden").exit_mode(message_id, chat_id, project)
        elif current_mode == InteractionMode.CODEX:
            self.get_handler("codex").exit_mode(message_id, chat_id, project)
        elif current_mode == InteractionMode.GEMINI:
            self.get_handler("gemini").exit_mode(message_id, chat_id, project)
        elif current_mode == InteractionMode.TTADK:
            self.get_handler("ttadk").exit_mode(message_id, chat_id, project)
        elif current_mode == InteractionMode.TUI2ACP:
            self.get_handler("tui2acp").exit_mode(message_id, chat_id, project)
        else:
            self.reply_text(message_id, UI_TEXT["system_already_in_mode"])

    # ------------------------------------------------------------------
    # Tui2ACP adapter selection
    # ------------------------------------------------------------------

    _TUI2ACP_ADAPTERS = [
        {"name": "coco", "emoji": "🤖", "description": "Coco (TraeCLI)"},
        {"name": "claude", "emoji": "🔮", "description": "Claude Code"},
        {"name": "codex", "emoji": "⚡", "description": "OpenAI Codex CLI"},
    ]

    def handle_tui2acp_command(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        from_card: bool = False,
    ):
        """Show tui2acp adapter selection card. Auto-installs if missing."""
        import shutil
        import subprocess

        def _find_tui2acp() -> str | None:
            """Find tui2acp in PATH + common npm-global locations."""
            found = shutil.which("tui2acp")
            if found:
                return found
            import os
            home = os.path.expanduser("~")
            extra_dirs = [
                os.path.join(home, ".npm-global", "bin"),
                os.path.join(home, ".local", "bin"),
                "/opt/homebrew/bin",
                "/usr/local/bin",
            ]
            for d in extra_dirs:
                candidate = os.path.join(d, "tui2acp")
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return candidate
            return None

        if not _find_tui2acp():
            if not shutil.which("npm"):
                self.reply_text(
                    message_id,
                    "❌ tui2acp 未安装且未找到 npm，无法自动安装。请手动安装 Node.js 和 npm 后重试。",
                )
                return
            self.reply_text(message_id, "⏳ tui2acp 未安装，正在自动安装...")
            try:
                subprocess.run(
                    ["npm", "install", "-g", "tui2acp"],
                    capture_output=True, text=True, timeout=120,
                )
            except Exception:
                pass
            if not _find_tui2acp():
                self.reply_text(
                    message_id,
                    "❌ tui2acp 自动安装失败。请手动运行 `npm install -g tui2acp`。",
                )
                return

        current_adapter = None
        if project:
            current_adapter = getattr(project, "tui2acp_adapter_name", None)

        _, card_content = CardBuilder.build_tui2acp_adapter_select_card(
            self._TUI2ACP_ADAPTERS,
            project_id=project.project_id if project else None,
            current_adapter=current_adapter,
        )
        if from_card:
            self.update_card(message_id, card_content)
        else:
            self.reply_card(message_id, card_content)

    def handle_select_tui2acp_adapter(
        self,
        message_id: str,
        chat_id: str,
        adapter_name: str,
        project_id: Optional[str] = None,
    ):
        """Handle card callback when user picks a tui2acp adapter."""
        adapter = (adapter_name or "").strip().lower()
        if not adapter:
            self.reply_text(message_id, "❌ 请选择一个 adapter")
            return

        project = self.project_manager.get_project_for_chat(project_id, chat_id) if project_id else self.project_manager.get_active_project(chat_id)

        if project:
            project.tui2acp_adapter_name = adapter
            project.acp_tool_name = "tui2acp"

        handler = self.get_handler("tui2acp")
        if handler:
            handler.current_adapter = adapter
            handler.enter_mode(message_id, chat_id, project=project)

    def handle_select_tui2acp_custom_command(
        self,
        message_id: str,
        chat_id: str,
        custom_command: str,
        project_id: Optional[str] = None,
    ):
        """Handle form submission when user enters a custom tui2acp command."""
        command = (custom_command or "").strip()
        if not command:
            self.reply_text(message_id, "❌ 请输入工具启动命令")
            return

        project = self.project_manager.get_project_for_chat(project_id, chat_id) if project_id else self.project_manager.get_active_project(chat_id)

        if project:
            project.tui2acp_adapter_name = f"custom:{command}"
            project.acp_tool_name = "tui2acp"

        handler = self.get_handler("tui2acp")
        if handler:
            handler.current_adapter = f"custom:{command}"
            handler.enter_mode(message_id, chat_id, project=project)

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
            msg_type, card_content = CardBuilder.build_shell_result_card(
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
                msg_type, card_content = CardBuilder.build_project_response_card(
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
            card_res = CardBuilder.build_directory_change_card(project, result, success=True)
            if card_res:
                msg_type, card_content = card_res
                response_id = self.reply_card(message_id, card_content)
                if response_id and project:
                    self.register_message_project(response_id, project)
            else:
                self.reply_text(message_id, fmt.format_dir_change(result, True))
        else:
            self.add_reaction(message_id, EmojiReaction.on_error())
            card_res = CardBuilder.build_directory_change_card(project, result, success=False)
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

        content = CardBuilder.build_coco_status_content(current_model, models)
        self.reply_text(message_id, content)

    def show_tools_list(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Show a list of all available ACP tools with quick access buttons."""
        # Define tool names
        names = ["coco", "claude", "aiden", "codex", "gemini", "traex"]
        emojis = {
            "coco": "🤖",
            "claude": "🔮",
            "aiden": "🎯",
            "codex": "💻",
            "gemini": "✨",
            "traex": "🚀",
        }

        # Cached-first availability check: avoid blocking user-path on external probe.
        tools = []
        for name in names:
            is_available = tool_registry.get_availability(name, allow_sync_probe=False, trigger_async_probe=True)
            desc = UI_TEXT[f"system_acp_tool_desc_{name}"]
            tools.append(
                {
                    "name": name,
                    "emoji": emojis.get(name, "🤖"),
                    "description": desc,
                    "available": is_available,
                }
            )

        msg_type, card = CardBuilder.build_tools_list_card(tools, project)
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
            is_available = tool_registry.get_availability(name, allow_sync_probe=False, trigger_async_probe=True)

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

        msg_type, card = CardBuilder.build_tools_status_card(tools, active_sessions, project)
        self.reply_interactive_card(message_id, card, msg_type=msg_type)
