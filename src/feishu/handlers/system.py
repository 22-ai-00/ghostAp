"""System handler — help, exit mode, shell commands, directory switching, intercepted commands."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ...acp.helper import (
    invalidate_acp_model_cache,  # noqa: F401
    is_programming_tool_available,
    list_acp_tools,  # noqa: F401
)
from ...acp.native_commands import (
    DEFAULT_TOOL_PRIORITY,
    STATIC_TOOL_NATIVE_COMMANDS,
    get_tool_native_commands,
    is_blocked_native_command,
    normalize_native_tool_key,
    parse_slash_command,
    resolve_smart_mode_native_command,
)
from ...acp.providers import get_providers  # noqa: F401
from ...card.backend_meta import VISIBLE_BACKEND_METAS
from ...card.builders.project import ProjectBuilder
from ...card.builders.system import SystemBuilder
from ...card.ui_text import UI_TEXT
from ...coco_model import get_coco_model_manager
from ...mode import PROGRAMMING_MODE_VALUES, PROGRAMMING_MODES, InteractionMode
from ...tasking import TaskPriority, TaskSpec
from ...utils.lock_order import LockLevel, ordered_rlock
from ..emoji import EmojiReaction
from ..message_formatter import FeishuMessageFormatter as fmt
from ..product_catalog import retired_command_tokens
from ..slash_command_parser import CommandMatch, SlashCommandParser
from .access_admin import AccessAdminMixin
from .acp_activation import ACPActivationMixin
from .base import BaseHandler
from .employee import EmployeeHandler
from .lock_commands import LockCommandsMixin

if TYPE_CHECKING:
    from ...project import ProjectContext
    from ..handler_context import HandlerContext

logger = logging.getLogger(__name__)


@dataclass
class _ACPActivationOwner:
    """One visible activation writer for an exact session coordinate."""

    generation: int
    fence_epoch: int
    origin_message_id: str
    status_message_id: str
    explicit_card: bool
    tool_name: str
    model_name: Optional[str]
    terminal: bool = False
    committed: bool = False


@dataclass(frozen=True)
class _ACPActivationEffect:
    """Exact session result and whether this activation mutated its lifecycle."""

    session: object
    changed: bool


@dataclass(frozen=True)
class _SystemSubcommands:
    """Minimal delegator view for a SystemHandler responsibility group."""
    _owner: "SystemHandler"
    _method_names: tuple[str, ...]

    def __getattr__(self, name: str):
        if name in self._method_names:
            return getattr(self._owner, name)
        raise AttributeError(name)


class SystemHandler(ACPActivationMixin, AccessAdminMixin, LockCommandsMixin, BaseHandler):
    """Help, exit, shell, directory, and intercepted-command handling."""

    # `/status` remains the established Deep/Spec diagnostics command. The
    # Standalone v5 Manager was superseded by the Journal-backed employee/Team
    # runtime. Keep its old spellings fail-closed with an explicit migration
    # message instead of reviving the obsolete process-local command surface.
    _RETIRED_AUTONOMOUS_MANAGER_COMMANDS = retired_command_tokens()
    _DIRECT_TOOL_COMMAND_TO_MODE: dict[str, str] = {
        "/coco": "coco",
        "/enter_coco": "coco",
        "/claude": "claude",
        "/enter_claude": "claude",
        "/claude-w": "claude_w",
        "/enter_claude_w": "claude_w",
        "/aiden": "aiden",
        "/enter_aiden": "aiden",
        "/codex": "codex",
        "/enter_codex": "codex",
        "/codex-w": "codex_w",
        "/enter_codex_w": "codex_w",
        "/gemini": "gemini",
        "/enter_gemini": "gemini",
        "/traex": "traex",
        "/enter_traex": "traex",
        "/grok": "grok",
        "/enter_grok": "grok",
        "/dsh": "dsh",
        "/enter_dsh": "dsh",
        "/agy": "agy",
        "/enter_agy": "agy",
    }
    _FAST_EXACT_CONTROL_COMMANDS: frozenset[str] = frozenset(
        {
            "/help",
            "/帮助",
            "/menu",
            "/tools",
            "/tools_status",
            "/projects",
            "/project",
            "/coco_status",
            "/coco_info",
            "/claude_info",
            "/aiden_info",
            "/codex_info",
            "/gemini_info",
            "/traex_info",
            "/grok_info",
            "/dsh_info",
            "/agy_info",
            "/employees",
            "/exit",
            "/quit",
            "/end_coco",
            "/exit_coco",
            "/end_claude",
            "/exit_claude",
            "/end_claude_w",
            "/exit_claude_w",
            "/end_aiden",
            "/exit_aiden",
            "/end_codex",
            "/exit_codex",
            "/end_codex_w",
            "/exit_codex_w",
            "/end_gemini",
            "/exit_gemini",
            "/end_traex",
            "/exit_traex",
            "/end_grok",
            "/exit_grok",
            "/end_dsh",
            "/exit_dsh",
            "/end_agy",
            "/exit_agy",
            *_DIRECT_TOOL_COMMAND_TO_MODE.keys(),
        }
    )
    _FAST_PREFIX_CONTROL_COMMANDS: frozenset[str] = frozenset(
        {
            "/cmds",
            "/acp",
            "/model",
            "/status",
            "/tasks",
            "/diff",
            "/trace",
            "/switch",
            "/new",
            "/new-chat",
            "/close",
            "/lock",
            "/unlock",
            "/setadmin",
            "/access",
            "/employee-role",
            "/hire",
            "/fire",
            "/history",
            "/employee-memory",
            "/btw",
            "/info",
            "/stop",
            "/clear",
            "/sh",
        }
    )

    def __init__(self, ctx: "HandlerContext") -> None:
        super().__init__(ctx)
        self.employee = EmployeeHandler(ctx)
        # All state transitions for one activation generation share this
        # re-entrant lock.  The critical sections are deliberately limited to
        # generation/card writes and the short durable commit + mode transition;
        # ACP startup and prompt execution remain outside it.
        self._acp_activation_lock = ordered_rlock(
            LockLevel.ACP_ACTIVATION,
            name="SystemHandler._acp_activation_lock",
        )
        self._acp_activation_generations: dict[tuple[str, str, str], int] = {}
        self._acp_activation_fence_epochs: dict[tuple[str, str, str], int] = {}
        self._acp_activation_owners: dict[
            tuple[str, str, str], _ACPActivationOwner
        ] = {}
        self._acp_activation_task_owners: dict[
            tuple[str, str, str], dict[int, _ACPActivationOwner]
        ] = {}
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
            "/claude-w": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "claude_w", p),
            "/enter_claude_w": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "claude_w", p),
            "/aiden": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "aiden", p),
            "/enter_aiden": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "aiden", p),
            "/codex": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "codex", p),
            "/enter_codex": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "codex", p),
            "/codex-w": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "codex_w", p),
            "/enter_codex_w": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "codex_w", p),
            "/gemini": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "gemini", p),
            "/enter_gemini": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "gemini", p),
            "/traex": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "traex", p),
            "/enter_traex": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "traex", p),
            "/grok": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "grok", p),
            "/enter_grok": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "grok", p),
            "/dsh": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "dsh", p),
            "/enter_dsh": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "dsh", p),
            "/agy": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "agy", p),
            "/enter_agy": lambda m, c, t, p: self._handle_direct_mode_enter(m, c, "agy", p),
            "/exit": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/quit": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_coco": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_coco": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_claude": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_claude": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_claude_w": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_claude_w": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_aiden": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_aiden": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_codex": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_codex": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_codex_w": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_codex_w": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_gemini": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_gemini": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_traex": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_traex": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_grok": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_grok": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_dsh": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_dsh": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/end_agy": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/exit_agy": lambda m, c, t, p: self.exit_current_mode(m, c, p),
            "/coco_status": lambda m, c, t, p: self.show_coco_status(m, c),
            "/coco_info": lambda m, c, t, p: self.get_handler("coco").show_info(m, c, p),
            "/claude_info": lambda m, c, t, p: self.get_handler("claude").show_info(m, c, p),
            "/aiden_info": lambda m, c, t, p: self.get_handler("aiden").show_info(m, c, p),
            "/codex_info": lambda m, c, t, p: self.get_handler("codex").show_info(m, c, p),
            "/gemini_info": lambda m, c, t, p: self.get_handler("gemini").show_info(m, c, p),
            "/traex_info": lambda m, c, t, p: self.get_handler("traex").show_info(m, c, p),
            "/grok_info": lambda m, c, t, p: self.get_handler("grok").show_info(m, c, p),
            "/dsh_info": lambda m, c, t, p: self.get_handler("dsh").show_info(m, c, p),
            "/agy_info": lambda m, c, t, p: self.get_handler("agy").show_info(m, c, p),
            "/projects": lambda m, c, t, p: self.get_handler("project").show_project_board(m, c),
            "/project": lambda m, c, t, p: self.get_handler("project").show_project_board(m, c),
            "/switch": lambda m, c, t, p: self.get_handler("project").show_project_board(m, c),
            "/new-chat": lambda m, c, t, p: self._handle_new_chat_project_args(m, c, ""),
            "/acp": lambda m, c, t, p: self.handle_acp_command(m, c, t, p),
            "/cmds": lambda m, c, t, p: self.show_tool_commands(m, c, t, p),
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
            ("/cmds", lambda m, c, t, p: self.show_tool_commands(m, c, t, p)),
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

    def _handle_direct_mode_enter_with_args(
        self,
        message_id: str,
        chat_id: str,
        mode_key: str,
        args: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Execute ``/<tool> <task or /native_cmd>`` in one step without card friction."""
        task_text = str(args or "").strip()
        if not task_text:
            self._handle_direct_mode_enter(message_id, chat_id, mode_key, project)
            return
        if is_blocked_native_command(task_text):
            parsed = parse_slash_command(task_text)
            blocked_name = parsed[0] if parsed else task_text.lstrip("/").split(maxsplit=1)[0].casefold()
            self.reply_error(
                message_id,
                UI_TEXT["system_native_cmd_blocked"].format(command=blocked_name),
            )
            return

        active_mode_key = self._resolve_active_programming_mode_key(chat_id, project)
        if active_mode_key == mode_key:
            handler = self.get_handler(mode_key)
            if handler is not None and hasattr(handler, "handle_message"):
                self.add_reaction(message_id, EmojiReaction.on_processing())
                handler.handle_message(message_id, chat_id, task_text, project)
                return

        self.add_reaction(message_id, EmojiReaction.on_processing())
        self.handle_enter_acp_saved_selection(
            message_id,
            chat_id,
            mode_key,
            project,
            pending_prompt=task_text,
        )

    def _resolve_live_commands_for_tool(
        self,
        chat_id: str,
        tool_name: str,
        project: Optional["ProjectContext"] = None,
        *,
        thread_root_id: Optional[str] = None,
    ) -> Optional[list]:
        try:
            handler = self.get_handler(tool_name)
        except Exception:
            return None
        manager_getter = getattr(handler, "_get_session_manager", None) if handler is not None else None
        manager = manager_getter() if callable(manager_getter) else None
        if manager is None or not hasattr(manager, "get_session"):
            return None
        try:
            session = manager.get_session(
                chat_id,
                project_id=self._project_id(project),
                thread_id=thread_root_id,
            )
        except Exception:
            return None
        getter = getattr(session, "get_available_commands", None) if session is not None else None
        if not callable(getter):
            return None
        try:
            cmds = getter()
            return cmds if isinstance(cmds, (list, tuple)) else None
        except Exception:
            return None

    def show_tool_commands(
        self,
        message_id: str,
        chat_id: str,
        text_or_tool: str = "",
        project: Optional["ProjectContext"] = None,
        *,
        origin_message_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> None:
        """Display the native slash commands and skills card for the active or requested tool."""
        raw_input = str(text_or_tool or "").strip()
        if raw_input.startswith("/"):
            parsed = SlashCommandParser.parse(raw_input)
            raw_arg = (parsed.args if parsed else "").strip()
        else:
            raw_arg = raw_input

        if raw_arg:
            token = raw_arg.split(maxsplit=1)[0]
            tool_name = normalize_native_tool_key(token)
            if tool_name not in STATIC_TOOL_NATIVE_COMMANDS and tool_name not in PROGRAMMING_MODE_VALUES:
                self.reply_error(
                    message_id,
                    UI_TEXT["system_acp_unsupported_tool"].format(tool_name=token),
                )
                return
        else:
            tool_name = (
                self._resolve_active_programming_mode_key(chat_id, project)
                or self._resolve_current_acp_tool(chat_id, project)
            )

        live_cmds = self._resolve_live_commands_for_tool(
            chat_id,
            tool_name,
            project,
            thread_root_id=thread_root_id,
        )
        commands, source = get_tool_native_commands(tool_name, live_commands=live_cmds)
        msg_type, card = SystemBuilder.build_tool_native_commands_card(
            tool_name,
            commands,
            source=source,
            project_id=self._project_id(project),
            thread_root_id=thread_root_id,
        )
        self.reply_interactive_card(origin_message_id or message_id, card, msg_type=msg_type)

    def handle_show_tool_commands(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str],
        value: dict,
    ) -> None:
        """Card callback opening the `/cmds` card for the specified tool."""
        project = (self.project_manager.get_project(project_id) if project_id else None) or (
            self.project_manager.get_active_project(chat_id)
        )
        tool_name = str((value or {}).get("tool_name") or "").strip()
        thread_root_id = (value or {}).get("thread_root_id")
        self.show_tool_commands(
            message_id,
            chat_id,
            tool_name,
            project,
            thread_root_id=thread_root_id,
        )

    def handle_run_native_tool_command(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str],
        value: dict,
    ) -> None:
        """Card callback executing a zero-arg native slash command on the target tool."""
        tool_name = normalize_native_tool_key(str((value or {}).get("tool_name") or ""))
        command = str((value or {}).get("command") or "").strip()
        if command and not command.startswith("/"):
            command = f"/{command}"
        if not tool_name or not command:
            return
        if is_blocked_native_command(command):
            parsed = parse_slash_command(command)
            blocked_name = parsed[0] if parsed else command.lstrip("/").split(maxsplit=1)[0].casefold()
            self.reply_error(
                message_id,
                UI_TEXT["system_native_cmd_blocked"].format(command=blocked_name),
            )
            return
        project = (self.project_manager.get_project(project_id) if project_id else None) or (
            self.project_manager.get_active_project(chat_id)
        )
        self._handle_direct_mode_enter_with_args(
            message_id,
            chat_id,
            tool_name,
            command,
            project,
        )

    def try_dispatch_smart_native_command(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional["ProjectContext"] = None,
    ) -> bool:
        """In SMART mode, auto-route unambiguous/preferred native slash commands to their tool."""
        if parse_slash_command(text) is None:
            return False

        cfg_project = self._configuration_project(chat_id, project)
        preferred_tool = getattr(cfg_project, "acp_tool_name", None) if cfg_project else None

        available_tools = [
            tool
            for tool in DEFAULT_TOOL_PRIORITY
            if is_programming_tool_available(
                tool,
                allow_sync_probe=False,
                trigger_async_probe=True,
            )
        ]
        live_map: dict[str, list] = {}
        for tool in DEFAULT_TOOL_PRIORITY:
            live_cmds = self._resolve_live_commands_for_tool(chat_id, tool, project)
            if live_cmds:
                live_map[tool] = live_cmds

        result = resolve_smart_mode_native_command(
            text,
            preferred_tool=preferred_tool,
            available_tools=available_tools,
            live_commands_by_tool=live_map,
        )
        if result.status == "unavailable" and result.matched_tools:
            # Retry sync probe only for matching candidates when cache was cold.
            probed_available = set(available_tools)
            for candidate in result.matched_tools:
                if is_programming_tool_available(
                    candidate,
                    allow_sync_probe=True,
                    trigger_async_probe=False,
                ):
                    probed_available.add(candidate)
            result = resolve_smart_mode_native_command(
                text,
                preferred_tool=preferred_tool,
                available_tools=probed_available,
                live_commands_by_tool=live_map,
            )

        if result.status == "none":
            return False
        if result.status == "blocked":
            self.reply_error(
                message_id,
                UI_TEXT["system_native_cmd_blocked"].format(command=result.command_name),
            )
            return True
        if result.status == "unavailable":
            self.reply_error(
                message_id,
                UI_TEXT["system_native_cmd_tool_unavailable"].format(
                    command=result.command_name,
                    tools=", ".join(result.matched_tools),
                ),
            )
            return True

        self.add_reaction(message_id, EmojiReaction.on_processing())
        self.handle_enter_acp_saved_selection(
            message_id,
            chat_id,
            result.tool_name,
            project,
            pending_prompt=text,
        )
        return True

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
            "/end_claude_w",
            "/exit_claude_w",
            "/end_aiden",
            "/exit_aiden",
            "/end_codex",
            "/exit_codex",
            "/end_codex_w",
            "/exit_codex_w",
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
        return any(kw in text_lower for kw in exit_keywords) or any(
            f"退出{mode}" in text_lower for mode in PROGRAMMING_MODE_VALUES
        )

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
            "/claude-w",
            "/enter_claude_w",
            "/aiden",
            "/enter_aiden",
            "/codex",
            "/enter_codex",
            "/codex-w",
            "/enter_codex_w",
            "/gemini",
            "/enter_gemini",
            "/traex",
            "/enter_traex",
            "/grok",
            "/enter_grok",
            "/dsh",
            "/enter_dsh",
            "/agy",
            "/enter_agy",
            "/exit",
            "/quit",
            "/end_coco",
            "/exit_coco",
            "/end_claude",
            "/exit_claude",
            "/end_claude_w",
            "/exit_claude_w",
            "/end_aiden",
            "/exit_aiden",
            "/end_codex",
            "/exit_codex",
            "/end_codex_w",
            "/exit_codex_w",
            "/end_gemini",
            "/exit_gemini",
            "/end_traex",
            "/exit_traex",
            "/end_grok",
            "/exit_grok",
            "/end_dsh",
            "/exit_dsh",
            "/end_agy",
            "/exit_agy",
            "/coco_status",
            "/coco_info",
            "/claude_info",
            "/aiden_info",
            "/codex_info",
            "/gemini_info",
            "/traex_info",
            "/grok_info",
            "/dsh_info",
            "/agy_info",
            "/projects",
            "/status",
            "/project",
            "/switch",
            "/new-chat",
            "/tasks",
            "/diff",
            "/trace",
            "/acp",
            "/cmds",
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
            "/acp",
            "/cmds",
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
            *SystemHandler._DIRECT_TOOL_COMMAND_TO_MODE.keys(),
        }
        prefix_commands.update(SystemHandler._RETIRED_AUTONOMOUS_MANAGER_COMMANDS)
        return cmd in prefix_commands

    @classmethod
    def is_ghostap_control_command(
        cls,
        command_match: CommandMatch | None,
        *,
        in_programming_mode: bool = False,
    ) -> bool:
        """Return True when *command_match* is a GhostAP control command rather than a tool native command."""
        if not command_match:
            return False
        if in_programming_mode and command_match.command == "/goal":
            return False
        return cls.is_interceptable_command_match(command_match)

    @classmethod
    def is_fast_lane_system_command(
        cls,
        text: str | None,
        *,
        in_programming_mode: bool = False,
    ) -> bool:
        """Classify lightweight GhostAP control commands vs task-bearing slash prompts.

        Prompts starting with ``/<tool> <task>``, ``/wf``, ``/deep``, ``/spec``, or native tool
        slash commands (``/swarm``, ``/compact``, ``/review``) belong on the normal ordered lane
        instead of overtaking queued work in ``system_fast``.
        """
        raw = str(text or "").strip()
        if not raw:
            return False
        if cls.is_exit_command(raw):
            return True
        m = SlashCommandParser.parse(raw)
        if not m:
            return False
        if in_programming_mode and m.command == "/goal":
            return False
        if m.command in cls._FAST_PREFIX_CONTROL_COMMANDS:
            return True
        if m.command in cls._FAST_EXACT_CONTROL_COMMANDS and not m.has_args:
            return True
        if m.command in cls._RETIRED_AUTONOMOUS_MANAGER_COMMANDS:
            return True
        return False

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
        if text_lower in self._DIRECT_TOOL_COMMAND_TO_MODE:
            self._handle_direct_mode_enter_with_args(
                message_id,
                chat_id,
                self._DIRECT_TOOL_COMMAND_TO_MODE[text_lower],
                m.args,
                project,
            )
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

        handle_btw = getattr(handler, "handle_btw", None)
        if callable(handle_btw):
            handle_btw(message_id, chat_id, command_match.args, project)
            return

        # Compatibility for custom programming handlers: /btw belongs to the
        # GhostAP control plane, so only the question reaches the provider.
        handler.handle_message(message_id, chat_id, command_match.args, project)

    def _resolve_active_programming_mode_key(
        self,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
    ) -> Optional[str]:
        from ...thread import get_current_thread_id

        thread_id = get_current_thread_id()
        if thread_id:
            thread_ctx = self.ctx.thread_manager.get(thread_id)
            if thread_ctx and thread_ctx.mode:
                try:
                    mode = InteractionMode(thread_ctx.mode)
                except ValueError:
                    mode = None
                if mode in PROGRAMMING_MODES:
                    return mode.value

        project_id = self._project_id(project)
        mode = self.mode_manager.get_mode(chat_id, project_id=project_id)
        if mode in PROGRAMMING_MODES and self.mode_manager.is_programming_mode(
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
                self._reply_slash_permission_hint(message_id)
                return

        self.reply_card(message_id, card_content)
        self._reply_slash_permission_hint(message_id)


    def handle_deep_prompt(self, message_id: str, chat_id: str):
        self.reply_text(
            message_id,
            UI_TEXT["system_help_deep_prompt"],
        )

    # ------------------------------------------------------------------
    # ACP command handling
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # /model command — list/switch models for current ACP tool
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Exit current mode
    # ------------------------------------------------------------------
    def exit_current_mode(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        from ...thread import get_current_thread_id, get_thread_manager, set_current_thread_id

        _pid = project.project_id if project else None
        thread_id = get_current_thread_id()
        thread_coordinate = self._canonical_acp_thread_id(thread_id)
        # The exit and activation commit share one lifecycle boundary.  Whichever
        # acquires it first becomes authoritative: a completed activation is then
        # exited normally, while an already-invalidated activation cannot commit
        # or re-enter the mode afterward.
        with self._acp_activation_lock:
            invalidated_owners = self._invalidate_acp_activation(
                chat_id,
                _pid,
                thread_coordinate,
            )
            self._finish_invalidated_acp_cards(invalidated_owners)
            current_mode = self.mode_manager.get_mode(chat_id, project_id=_pid)

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
                if (
                    thread_ctx
                    and thread_ctx.mode != "smart"
                    and current_mode == InteractionMode.SMART
                ):
                    try:
                        current_mode = InteractionMode(thread_ctx.mode)
                    except ValueError:
                        logger.debug(
                            "invalid InteractionMode value: %s",
                            thread_ctx.mode,
                            exc_info=True,
                        )

            if current_mode in PROGRAMMING_MODES:
                self.get_handler(current_mode.value).exit_mode(
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
        """Execute a host shell command directly and reply with the result."""
        from ...command_executor import CommandExecutor
        from ...repo_lock import LockConflictError

        if project is not None:
            lock_root_path = getattr(project, "root_path", None) or working_dir
        else:
            lock_root_path = self.lock_helper.resolve_git_lock_root(working_dir)

        def _run_shell():
            executor = CommandExecutor()
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
        # Only user-visible programming tools; hidden bridges (e.g. claude-w)
        # stay out of the discovery surfaces even though they are full modes.
        tools = []
        for meta in VISIBLE_BACKEND_METAS:
            name = meta.mode_key
            is_available = is_programming_tool_available(
                name,
                allow_sync_probe=False,
                trigger_async_probe=True,
            )
            desc = UI_TEXT[f"system_acp_tool_desc_{name}"]
            tools.append(
                {
                    "name": name,
                    "emoji": meta.emoji,
                    "description": desc,
                    "available": is_available,
                }
            )

        msg_type, card = SystemBuilder.build_tools_list_card(tools, project)
        self.reply_interactive_card(message_id, card, msg_type=msg_type)

    def show_tools_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Show detailed status of all tools with availability and session info."""
        # Define tool metadata from unified SSOT
        tool_defs = [
            {
                "name": meta.mode_key,
                "emoji": meta.emoji,
                "manager": getattr(self.ctx, f"{meta.mode_key}_manager", None),
            }
            for meta in VISIBLE_BACKEND_METAS
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
