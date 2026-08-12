import logging
import os
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..agent.intent_recognizer import TaskStep
    from ..project import ProjectContext
    from ..trust.models import EffectiveTrust


from ..agent.intent_recognizer import IntentType
from ..card.ui_text import UI_TEXT
from ..utils.errors import get_error_detail
from .emoji import EmojiReaction
from .message_formatter import FeishuMessageFormatter as fmt
from .route_decision import CommandRouter
from .slash_command_parser import CommandMatch

logger = logging.getLogger(__name__)


class MessageDispatcher:
    """Handles the dispatching of user messages and intents to appropriate engines/modes."""

    def __init__(self, client: Any):
        self.client = client
        self.handlers = client._handler_ctx.handlers
        self.base = self.handlers["coco"]
        self.system = self.handlers["system"]
        self.project = self.handlers["project"]
        self._router = CommandRouter()

    def process_with_intent(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional['ProjectContext'] = None,
        *,
        command_match: CommandMatch | None = None,
        shell_fast_tracked: bool = False,
        chat_type: str = "group",
        effective_trust: Optional['EffectiveTrust'] = None,
    ):
        """SMART mode routing logic."""
        if effective_trust is not None:
            from ..trust.models import TrustZone

            if effective_trust.zone is TrustZone.EXTERNAL_OR_UNKNOWN_GROUP:
                return

        def current_dispatch_allowed() -> bool:
            if effective_trust is None:
                return True
            gate = getattr(self.client, "_current_trust_can_dispatch", None)
            return bool(
                callable(gate)
                and gate(effective_trust, project=project) is True
            )
        from .slash_command_parser import SlashCommandParser

        if command_match is None and (text or "").strip().startswith("/"):
            command_match = SlashCommandParser.parse(text)
        if (
            command_match is not None
            and command_match.command
            in {"/access", "/setadmin", "/hire", "/fire", "/employee-role"}
            and not self._action_matrix_allows(
                effective_trust,
                action_name="grant_admin",
            )
        ):
            return
        if (
            command_match is not None
            and command_match.command in {"/employees", "/history", "/employee-memory"}
            and not self._action_matrix_allows(
                effective_trust,
                action_name="system_admin",
            )
        ):
            return

        _pid = project.project_id if project else None
        current_mode, is_in_programming = self.client._get_effective_mode(chat_id, project_id=_pid)
        is_topic_engine_context = (
            getattr(self.client, "_is_topic_engine_context", lambda: False)() is True
        )

        # Control-plane commands: handle consistently in all modes
        if self._router.is_deep_command(text):
            if not current_dispatch_allowed():
                return
            self.base.add_reaction(message_id, EmojiReaction.on_smart_mode())
            self.base.add_reaction(message_id, EmojiReaction.on_processing())
            self.handlers["deep"].handle_deep_command(message_id, chat_id, text, project)
            return

        if self._router.is_spec_command(text):
            if not current_dispatch_allowed():
                return
            self.base.add_reaction(message_id, EmojiReaction.on_smart_mode())
            self.base.add_reaction(message_id, EmojiReaction.on_processing())
            self.handlers["spec"].handle_spec_command(message_id, chat_id, text, project)
            return

        if self._router.is_workflow_command(text):
            if not current_dispatch_allowed():
                return
            # Workflow mode entry: /wf is the top-level orchestrator agent entry point
            # (agent → tools → roles → script confirm → execute), no prior /coco needed
            self.base.add_reaction(message_id, EmojiReaction.on_smart_mode())
            self.base.add_reaction(message_id, EmojiReaction.on_processing())
            self.handlers["workflow"].handle_workflow_command(message_id, chat_id, text, project)
            return

        # Topic engine context routing: workflow mode free-text goes to WorkflowHandler
        if is_topic_engine_context:
            from src.thread import get_current_thread_id
            thread_id = get_current_thread_id()
            if thread_id:
                thread_ctx = self.client._thread_manager.get(thread_id)
                if thread_ctx and thread_ctx.mode == "workflow":
                    # Free-text messages in workflow topic context go to WorkflowHandler
                    # Consistent with ws_client.py:1558-1590 routing logic
                    if command_match is None and project is not None:
                        if not current_dispatch_allowed():
                            return
                        self.base.add_reaction(message_id, EmojiReaction.on_processing())
                        self.handlers["workflow"].handle_message(message_id, chat_id, text, project)
                        return

        if is_in_programming and self._router.is_exit_command(text):
            if not current_dispatch_allowed():
                return
            self.base.add_reaction(message_id, EmojiReaction.on_coco_mode())
            if self.client._control_plane.should_defer_exit(chat_id=chat_id, project_id=_pid):
                self.client._control_plane.request_deferred_exit(message_id=message_id, chat_id=chat_id, project_id=_pid)
                self.base.reply_text(message_id, UI_TEXT["ws_exit_deferred_msg"])
                return
            self.system.exit_current_mode(message_id, chat_id, project=project)
            return

        # Request-scoped slash parsing: direct system commands must not fall
        # through to intent recognition, which can be slow or ambiguous.
        if self.system.is_interceptable_command_match(command_match):
            if not current_dispatch_allowed():
                return
            self.system.handle_intercepted_command(message_id, chat_id, text, project, command_match=command_match)
            return

        if command_match is not None:
            if not current_dispatch_allowed():
                return
            self.system.handle_intercepted_command(
                message_id,
                chat_id,
                text,
                project,
                command_match=command_match,
            )
            return

        # Programming mode: exit or forward to active session
        if is_in_programming:
            if not current_dispatch_allowed():
                return
            self.base.add_reaction(message_id, EmojiReaction.on_coco_mode())
            self.base.add_reaction(message_id, EmojiReaction.on_processing())
            handler = self.client._get_mode_handler(current_mode)
            if handler:
                handler.handle_message(message_id, chat_id, text, project)
            else:
                self.system.show_help(message_id, chat_id)
            return

        # SMART mode: image-only messages bypass intent recognition
        with self.client._pending_image_lock:
            is_image_only = message_id in self.client._pending_image_only
        if is_image_only:
            if not current_dispatch_allowed():
                return
            self.base.add_reaction(message_id, EmojiReaction.on_coco_mode())
            self.base.add_reaction(message_id, EmojiReaction.on_processing())
            self.base.handle_message(message_id, chat_id, text, project)
            return

        # SMART mode: intent recognition
        self.base.add_reaction(message_id, EmojiReaction.on_smart_mode())
        self.base.add_reaction(message_id, EmojiReaction.on_processing())

        try:
            intent_result = self.client._intent_recognizer.recognize(text, current_mode.value)
        except (RuntimeError, TimeoutError, ValueError, TypeError) as e:
            if not self._action_matrix_allows(
                effective_trust,
                action_name="host_shell",
            ):
                return
            if not current_dispatch_allowed():
                return
            logger.warning("意图识别异常，回退到 shell: %s", get_error_detail(e))
            working_dir = self.base.get_working_dir(chat_id)
            self.system.submit_shell_command(message_id, chat_id, text, working_dir, project)
            return

        logger.info(
            "意图识别: %s (置信度: %.2f, 任务数: %d)",
            intent_result.primary_intent.value,
            intent_result.confidence,
            len(intent_result.tasks),
        )

        tasks = intent_result.tasks
        if any(task.intent is IntentType.SHELL_COMMAND for task in tasks):
            if not self._action_matrix_allows(effective_trust, action_name="host_shell"):
                return
        if not tasks:
            self.execute_single_task(message_id, chat_id, None, text, project)
            return
        for task in tasks:
            if not current_dispatch_allowed():
                return
            self.execute_single_task(
                message_id,
                chat_id,
                task,
                text,
                project,
                shell_fast_tracked=shell_fast_tracked,
            )
            if task.intent.name.startswith("ENTER_") and task.intent.name[6:].lower() in self._PROGRAMMING_MODES:
                break

    @staticmethod
    def _action_matrix_allows(
        effective_trust: Optional['EffectiveTrust'],
        *,
        action_name: str,
    ) -> bool:
        if effective_trust is None:
            return True
        from ..trust.action_matrix import ActionMatrix
        from ..trust.models import (
            ActionDecision,
            ActionKind,
            ActionRequest,
            ActionTargetKind,
        )

        action = {
            "grant_admin": ActionKind.GRANT_ADMIN,
            "host_shell": ActionKind.HOST_SHELL,
            "system_admin": ActionKind.SYSTEM_ADMIN,
        }[action_name]
        return ActionMatrix().decide(
            ActionRequest(
                trust=effective_trust,
                action=action,
                target=ActionTargetKind.HOST_GLOBAL,
            )
        ) is ActionDecision.ALLOW

    _PROGRAMMING_MODES = frozenset({"coco", "claude", "aiden", "codex", "gemini", "traex", "grok"})

    _ENGINE_ENTER_MAP: dict = {
        IntentType.ENTER_DEEP: ("deep", "start_deep_engine"),
        IntentType.ENTER_SPEC: ("spec", "start_spec_engine"),
    }

    _SIMPLE_ENGINE_DISPATCH: dict = {
        IntentType.SPEC_STATUS: ("spec", "show_spec_status"),
        IntentType.STOP_SPEC: ("spec", "stop_spec_engine"),
    }

    _ENGINE_GUIDE_MAP: dict = {
        IntentType.DEEP_UPDATE: ("deep", "update_deep_context", "📝 请提供上下文信息\n\n用法: `/deep_update <上下文描述>`"),
        IntentType.SPEC_GUIDE: ("spec", "update_spec_guidance", "📝 请提供引导信息\n\n用法: `/spec_guide <引导描述>`"),
    }

    _PROJECT_INTENTS: set = {
        IntentType.CREATE_PROJECT, IntentType.SWITCH_PROJECT,
        IntentType.LIST_PROJECTS, IntentType.CLOSE_PROJECT,
        IntentType.PROJECT_STATUS, IntentType.NEW_CHAT_PROJECT,
    }

    def execute_single_task(
        self,
        message_id: str,
        chat_id: str,
        task: Optional['TaskStep'],
        original_text: str,
        project: Optional['ProjectContext'] = None,
        *,
        shell_fast_tracked: bool = False,
    ):
        """执行单一任务步骤（模式切换/系统命令/引擎命令/执行 shell 等）。"""
        if not task:
            self._handle_no_task(message_id, chat_id)
            return

        intent = task.intent
        data = task.data
        intent_name = intent.name

        if intent_name.startswith("ENTER_") and intent_name[6:].lower() in self._PROGRAMMING_MODES:
            mode = intent_name[6:].lower()
            if data.get("auto_forward") is True:
                self._auto_enter_and_forward(mode, message_id, chat_id, original_text, project)
            elif mode == "coco":
                self._handle_enter_coco(message_id, chat_id, project)
            else:
                self._handle_enter_acp_mode(mode, message_id, chat_id, project)
            return
        if intent_name.startswith("EXIT_") and intent_name[5:].lower() in self._PROGRAMMING_MODES:
            self.handlers[intent_name[5:].lower()].exit_mode(message_id, chat_id, project=project)
            return
        if intent_name.endswith("_MESSAGE") and intent_name[:-8].lower() in self._PROGRAMMING_MODES:
            self._handle_mode_message(intent_name[:-8].lower(), data, message_id, chat_id, original_text, project)
            return

        if intent == IntentType.EXIT_MODE:
            self.system.exit_current_mode(message_id, chat_id, project=project)
        elif intent == IntentType.CHANGE_DIR:
            self.system.change_directory(message_id, chat_id, data.get("path", ""), project)
        elif intent == IntentType.SHOW_HELP:
            self.system.show_full_help(message_id, chat_id, project)
        elif intent == IntentType.SHOW_TOOLS:
            self.system.show_tools_list(message_id, chat_id, project)
        elif intent == IntentType.TOOLS_STATUS:
            self.system.show_tools_status(message_id, chat_id, project)
        elif intent == IntentType.LIST_EMPLOYEES:
            self.system.employee.list_employees_roster(
                message_id,
                chat_id,
                project,
            )
        # Project commands
        elif intent in self._PROJECT_INTENTS:
            self._dispatch_project(intent, data, message_id, chat_id, project)
        # Engine enter
        elif intent in self._ENGINE_ENTER_MAP:
            requirement = data.get("requirement") or original_text
            handler, method = self._ENGINE_ENTER_MAP[intent]
            getattr(self.handlers[handler], method)(message_id, chat_id, requirement, project)
        # Engine status/control (simple)
        elif intent in self._SIMPLE_ENGINE_DISPATCH:
            handler, method = self._SIMPLE_ENGINE_DISPATCH[intent]
            getattr(self.handlers[handler], method)(message_id, chat_id, project)
        # Deep status/stop (arg parsing)
        elif intent in (IntentType.DEEP_STATUS, IntentType.STOP_DEEP):
            self._handle_deep_status_or_stop(intent, data, message_id, chat_id, project)
        # Engine guide/update
        elif intent in self._ENGINE_GUIDE_MAP:
            self._handle_engine_guide(intent, data, message_id, chat_id, project)
        # Shell
        elif intent == IntentType.SHELL_COMMAND:
            self._dispatch_shell(data, message_id, chat_id, original_text, project, shell_fast_tracked)
        # Unknown
        elif intent == IntentType.UNKNOWN:
            self.base.reply_text(message_id, fmt.format_unknown_intent())

    # ------------------------------------------------------------------
    # Extracted helpers for execute_single_task
    # ------------------------------------------------------------------

    def _handle_no_task(self, message_id: str, chat_id: str):
        from ..thread import get_current_thread_id
        if self.client.settings.thread_programming_enabled and not get_current_thread_id():
            active_thread = self.client._find_active_thread(chat_id)
            if active_thread:
                mode_display = active_thread.mode.upper() if active_thread.mode else "编程"
                self.base.reply_text(
                    message_id,
                    UI_TEXT["ws_active_topic_msg"].format(name=mode_display),
                )
                return
        self.base.reply_text(message_id, "🤔 无法理解你的意图")

    def _handle_enter_coco(self, message_id: str, chat_id: str, project, *, pending_prompt: Optional[str] = None):
        if pending_prompt is None:
            self.system.show_explicit_acp_model_selection(
                message_id,
                chat_id,
                "coco",
                project,
            )
            return
        _pid = project.project_id if project else None
        if self.client._mode_manager.is_coco_mode(chat_id, project_id=_pid):
            if pending_prompt:
                handler = self.client._get_mode_handler(
                    self.client._mode_manager.get_mode(chat_id, project_id=_pid)
                )
                if handler:
                    handler.handle_message(message_id, chat_id, pending_prompt, project)
            return
        self.base.enter_mode(message_id, chat_id, project=project)
        if pending_prompt:
            self.base.handle_message(
                message_id, chat_id, pending_prompt, project
            )

    def _handle_enter_acp_mode(self, mode: str, message_id: str, chat_id: str, project, *, pending_prompt: Optional[str] = None):
        if pending_prompt is None:
            self.system.show_explicit_acp_model_selection(
                message_id,
                chat_id,
                mode,
                project,
            )
            return
        _pid = project.project_id if project else None
        mode_checker = getattr(self.client._mode_manager, f"is_{mode}_mode", None)
        if callable(mode_checker) and mode_checker(chat_id, project_id=_pid):
            if pending_prompt:
                handler = self.client._get_mode_handler(
                    self.client._mode_manager.get_mode(chat_id, project_id=_pid)
                )
                if handler:
                    handler.handle_message(message_id, chat_id, pending_prompt, project)
            return

        enter_fn = getattr(self.handlers[mode], "enter_mode", None)
        if enter_fn:
            enter_fn(message_id, chat_id, project=project)
        if pending_prompt:
            handle_fn = getattr(self.handlers[mode], "handle_message", None)
            if handle_fn:
                handle_fn(message_id, chat_id, pending_prompt, project)

    def _auto_enter_and_forward(self, mode: str, message_id: str, chat_id: str, text: str, project):
        """Auto-enter programming mode and forward message (default ACP tool)."""
        enter_fn = getattr(self.handlers[mode], "enter_mode", None)
        if enter_fn:
            enter_fn(message_id, chat_id, silent=True, project=project)
        handle_fn = getattr(self.handlers[mode], "handle_message", None)
        if handle_fn:
            handle_fn(message_id, chat_id, text, project)
        else:
            logger.warning("默认工具模式 %s 无消息处理器", mode)

    def _handle_mode_message(self, mode: str, data: dict, message_id: str, chat_id: str, original_text: str, project):
        if data.get("command") == "info":
            self.handlers[mode].show_info(message_id, chat_id, project)
        else:
            self.handlers[mode].handle_message(message_id, chat_id, original_text, project)

    def _dispatch_project(self, intent, data: dict, message_id: str, chat_id: str, project):
        if intent == IntentType.CREATE_PROJECT:
            name = data.get("name", "")
            path = data.get("path", "")
            working_dir = self.base.get_working_dir(chat_id)
            if not path:
                path = working_dir
            if not name:
                name = os.path.basename(os.path.normpath(path))
                if not name or name in (".", "/", "~"):
                    name = f"project_{int(time.time())}"
            self.project.create_project(message_id, chat_id, name, path)
        elif intent == IntentType.SWITCH_PROJECT:
            name = data.get("name", "")
            if name:
                self.project.switch_project(
                    message_id,
                    chat_id,
                    name,
                    coco_handler=self.base,
                    claude_handler=self.handlers["claude"],
                )
            else:
                self.project.show_project_board(message_id, chat_id)
        elif intent == IntentType.LIST_PROJECTS:
            self.project.show_project_board(message_id, chat_id)
        elif intent == IntentType.CLOSE_PROJECT:
            name = data.get("name", "")
            if name:
                self.project.close_project(message_id, chat_id, name)
            else:
                self.base.reply_text(message_id, "❌ 请指定要关闭的项目名称")
        elif intent == IntentType.PROJECT_STATUS:
            self.project.show_project_status(message_id, chat_id, project)
        elif intent == IntentType.NEW_CHAT_PROJECT:
            self.project.handle_new_chat_project(message_id, chat_id, data)

    def _handle_deep_status_or_stop(self, intent, data: dict, message_id: str, chat_id: str, project):
        arg = (data.get("arg") or "").strip().lower()
        is_all = arg in ("all", "-a", "--all")
        if intent == IntentType.DEEP_STATUS:
            if is_all:
                self.handlers["deep"].show_deep_board(message_id, chat_id)
            else:
                self.handlers["deep"].show_deep_status(message_id, chat_id, project)
        else:  # STOP_DEEP
            if is_all:
                self.handlers["deep"].stop_all_deep_engines(message_id, chat_id)
            else:
                self.handlers["deep"].stop_deep_engine(message_id, chat_id, project)

    def _handle_engine_guide(self, intent, data: dict, message_id: str, chat_id: str, project):
        handler, method_name, hint = self._ENGINE_GUIDE_MAP[intent]
        guide_message = data.get("message")
        if guide_message:
            getattr(self.handlers[handler], method_name)(message_id, chat_id, guide_message, project)
        else:
            self.base.reply_text(message_id, hint)

    def _dispatch_shell(self, data: dict, message_id: str, chat_id: str, original_text: str, project, shell_fast_tracked: bool):
        working_dir = self.base.get_working_dir(chat_id)
        cmd = data.get("command") or original_text
        if shell_fast_tracked:
            self.system.execute_shell_and_reply(message_id, chat_id, cmd, working_dir, project)
        else:
            self.system.submit_shell_command(message_id, chat_id, cmd, working_dir, project)
        if project:
            project.add_conversation("user", cmd, message_id)
            self.client._context_manager.update_context(
                project.project_id,
                conversation={"role": "user", "content": cmd, "source_mode": "shell", "message_id": message_id},
                chat_id=chat_id,
            )
