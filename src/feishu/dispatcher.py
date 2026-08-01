import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..agent.intent_recognizer import IntentResult, TaskStep
    from ..project import ProjectContext
    from ..trust.models import EffectiveTrust


from ..agent.intent_recognizer import IntentRecognizer, IntentType
from ..card.ui_text import UI_TEXT
from ..slock_engine.gateway import NEEDS_ACTIVATION
from ..utils.errors import get_error_detail
from .emoji import EmojiReaction
from .message_formatter import FeishuMessageFormatter as fmt
from .product_catalog import is_explicit_protected_command
from .request_context import RequestContext
from .route_decision import CommandRouter, RouteDecision, RouteTarget
from .slash_command_parser import CommandMatch

logger = logging.getLogger(__name__)


class DispatchErrorAction(str, Enum):
    FALLBACK_TO_SHELL = "fallback_to_shell"
    FALLBACK_TO_DIRECT_ENTER = "fallback_to_direct_enter"
    LOG_AND_CONTINUE = "log_and_continue"
    STOP_MULTI_TASK = "stop_multi_task"
    RAISE = "raise"


@dataclass(frozen=True)
class DispatchErrorClassification:
    action: DispatchErrorAction
    phase: str
    user_reachable: bool = True


def classify_dispatch_error(error: Exception, *, phase: str) -> DispatchErrorClassification:
    if phase == "intent_recognition":
        return DispatchErrorClassification(DispatchErrorAction.FALLBACK_TO_SHELL, phase, True)
    if phase == "coco_model_card":
        return DispatchErrorClassification(DispatchErrorAction.FALLBACK_TO_DIRECT_ENTER, phase, True)
    if phase == "pending_prompt_forward":
        return DispatchErrorClassification(DispatchErrorAction.LOG_AND_CONTINUE, phase, False)
    if phase == "multi_task_step":
        return DispatchErrorClassification(DispatchErrorAction.STOP_MULTI_TASK, phase, True)
    return DispatchErrorClassification(DispatchErrorAction.RAISE, phase, False)


@dataclass(frozen=True)
class FeishuRequestContext:
    """Request-scoped message context for dispatcher core paths."""
    message_id: str
    chat_id: str
    text: str
    project: Optional['ProjectContext'] = None
    command_match: CommandMatch | None = None
    shell_fast_tracked: bool = False
    chat_type: str = "group"
    effective_trust: Optional['EffectiveTrust'] = None


class MessageDispatcher:
    """Handles the dispatching of user messages and intents to appropriate engines/modes."""

    def __init__(self, client: Any):
        self.client = client
        self._router = CommandRouter()

    # ------------------------------------------------------------------
    # Pure routing: produces RouteDecision without side effects
    # ------------------------------------------------------------------

    def resolve_command_route(self, ctx: RequestContext) -> Optional[RouteDecision]:
        """Determine routing target from command detection alone.

        Returns a RouteDecision if a control-plane command matches,
        or None if the message should proceed to intent recognition.
        This method is side-effect-free and independently testable.
        """
        text = ctx.text

        # Control-plane engine commands
        if CommandRouter.is_deep_command(text):
            return RouteDecision(target=RouteTarget.DEEP_ENGINE)
        if CommandRouter.is_spec_command(text):
            return RouteDecision(target=RouteTarget.SPEC_ENGINE)
        if CommandRouter.is_workflow_command(text):
            return RouteDecision(target=RouteTarget.WORKFLOW_ENGINE)

        # Slock command
        slock_result = CommandRouter.is_slock_command(
            text, ctx.chat_id,
            manager=getattr(self.client, '_slock_engine_manager', None),
        )
        if slock_result and slock_result is not NEEDS_ACTIVATION:
            return RouteDecision(target=RouteTarget.SLOCK_COMMAND)
        if slock_result == NEEDS_ACTIVATION and ctx.slock_context_allowed:
            return RouteDecision(
                target=RouteTarget.SLOCK_AUTO_ACTIVATE,
                payload={"needs_activation": True},
            )

        # Exit command while in programming mode
        if ctx.is_in_programming and CommandRouter.is_exit_command(text):
            return RouteDecision(target=RouteTarget.EXIT_MODE)

        # Interceptable slash command
        if ctx.command_match is not None:
            return RouteDecision(
                target=RouteTarget.SYSTEM_COMMAND,
                payload={"command_match": ctx.command_match},
            )

        # Worktree awaiting goal
        if ctx.has_project and self.client._is_worktree_awaiting_goal(ctx.project):
            return RouteDecision(target=RouteTarget.WORKTREE_GOAL)

        # Active programming mode
        if ctx.is_in_programming:
            return RouteDecision(target=RouteTarget.PROGRAMMING_MODE)

        return None  # Proceed to intent recognition

    def process_request(self, request: FeishuRequestContext):
        """Dispatch using a request context while preserving legacy entrypoint behavior."""
        return self.process_with_intent(
            request.message_id,
            request.chat_id,
            request.text,
            request.project,
            command_match=request.command_match,
            shell_fast_tracked=request.shell_fast_tracked,
            chat_type=request.chat_type,
            effective_trust=request.effective_trust,
        )

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
            and command_match.command in {"/access", "/setadmin", "/hire", "/fire"}
            and not self._action_matrix_allows(
                effective_trust,
                action_name="grant_admin",
            )
        ):
            return

        _pid = project.project_id if project else None
        current_mode, is_in_programming = self.client._get_effective_mode(chat_id, project_id=_pid)
        is_topic_engine_context = (
            getattr(self.client, "_is_topic_engine_context", lambda: False)() is True
        )
        slock_context_allowed = not is_in_programming and not is_topic_engine_context
        slock_auto_activate_allowed = slock_context_allowed and project is None

        # Control-plane commands: handle consistently in all modes
        if self.client._is_deep_command(text):
            if not current_dispatch_allowed():
                return
            self.client._add_reaction(message_id, EmojiReaction.on_smart_mode())
            self.client._add_reaction(message_id, EmojiReaction.on_processing())
            self.client._handle_deep_command(message_id, chat_id, text, project)
            return

        if self.client._is_spec_command(text):
            if not current_dispatch_allowed():
                return
            self.client._add_reaction(message_id, EmojiReaction.on_smart_mode())
            self.client._add_reaction(message_id, EmojiReaction.on_processing())
            self.client._handle_spec_command(message_id, chat_id, text, project)
            return

        if self.client._is_workflow_command(text):
            if not current_dispatch_allowed():
                return
            # Workflow mode entry: /wf is the top-level orchestrator agent entry point
            # (agent → tools → roles → script confirm → execute), no prior /coco needed
            self.client._add_reaction(message_id, EmojiReaction.on_smart_mode())
            self.client._add_reaction(message_id, EmojiReaction.on_processing())
            self.client._handle_workflow_command(message_id, chat_id, text, project)
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
                        self.client._add_reaction(message_id, EmojiReaction.on_processing())
                        self.client._workflow_handler.handle_message(message_id, chat_id, text, project)
                        return

        # Protected explicit entries are routed before any Slock detection.
        # This is intentionally independent from catalog completion labels.
        _is_protected_explicit_command = is_explicit_protected_command(text)
        _slock_result = (
            False
            if _is_protected_explicit_command
            else self.client._is_slock_command(text, chat_id)
        )
        if _slock_result and not _is_protected_explicit_command:
            if not current_dispatch_allowed():
                return
            self.client._add_reaction(message_id, EmojiReaction.on_smart_mode())
            self.client._add_reaction(message_id, EmojiReaction.on_processing())
            self.client._handle_slock_command(message_id, chat_id, text, project)
            return
        if (
            _slock_result == NEEDS_ACTIVATION
            and slock_context_allowed
            and not _is_protected_explicit_command
        ):
            self.client._add_reaction(message_id, EmojiReaction.on_smart_mode())
            # In passive mode, auto-activate instead of asking user to run /slock
            _passive_mode = getattr(self.client.settings, "slock_passive_mode", True)
            _is_explicit_slash_command = (text or "").lstrip().startswith("/")
            if (
                _passive_mode
                and slock_auto_activate_allowed
                and not _is_explicit_slash_command
                and self.client._should_auto_activate_slock(chat_id, text, chat_type=chat_type)
            ):
                activated, _ = self.client._auto_activate_slock(chat_id, text, project)
                if activated:
                    self.client._add_reaction(message_id, EmojiReaction.on_processing())
                    return
            # Fallback: user-friendly message without /slock requirement
            self.client._reply_text(message_id, "💡 **直接在群里发任务即可，Agent 自动处理。** 无需任何前置命令。")
            return

        # Shell is a SMART control/data-plane route of its own.  Detect it from
        # the shared intent SSOT before passive Slock can classify, resolve, or
        # clarify it.  The normal intent path below still distinguishes `cd`
        # from executable shell commands and preserves their existing handling.
        _is_smart_shell = (
            slock_context_allowed
            and command_match is None
            and IntentRecognizer.looks_like_shell(text or "")
        )

        # Slock active chat: route non-command messages to slock engine
        # Passive mode (default): only managed-chat check is needed.
        # Legacy mode: both activation AND managed-chat registration required.
        # Auto-activate: in passive mode, if chat is not yet managed but message
        # looks like a valid task, auto-activate slock for this chat (idempotent).
        _is_managed = self.client._is_slock_managed_chat(chat_id)
        _passive_mode = getattr(self.client.settings, "slock_passive_mode", True)
        # Global command check BEFORE slock routing: valid global commands must not
        # be intercepted by Slock in passive managed chat mode.
        _is_global_command = (
            command_match is not None
            or self.client._is_interceptable_command_match(command_match)
        )
        if _passive_mode and slock_context_allowed and not _is_smart_shell:
            if _is_managed and not _is_global_command:
                # Passive mode: managed chat alone is sufficient for routing
                # BUT skip if this is a valid global command
                from src.slock_engine.task_classifier import TaskClassifier

                is_ambient, ambient_confidence = TaskClassifier.classify(
                    text or "", managed_chat=True
                )
                if not current_dispatch_allowed():
                    return
                if is_ambient and ambient_confidence >= 0.7:
                    # The durable group ledger above remains the only effect.
                    return
                self.client._add_reaction(message_id, EmojiReaction.on_processing())
                self.client._handle_slock_message(message_id, chat_id, text, project)
                return
            # Auto-activate: use 3-way classification (task/chat/uncertain)
            # Skip auto-activate for explicit slash commands
            _is_command_intent = command_match is not None or (text or "").lstrip().startswith("/")
            if slock_auto_activate_allowed and not _is_command_intent:
                from src.slock_engine.gateway import (
                    SlockMessageClass,
                    attempt_autonomous_resolve,
                    build_activation_denied_card,
                    build_clarification_card,
                    classify_message,
                )

                _is_already_managed = _is_managed
                result = classify_message(text or "", managed_chat=_is_already_managed)

                if result.label == SlockMessageClass.CHAT:
                    logger.debug("Slock: message classified as chat, ignoring: chat=%s", chat_id)
                elif result.label == SlockMessageClass.UNCERTAIN:
                    logger.info("Slock: message classification uncertain, attempting autonomous resolution: chat=%s", chat_id)
                    import json

                    from src.thread.manager import get_current_sender_id

                    resolve_outcome = attempt_autonomous_resolve(text or "", message_id, chat_id)

                    if resolve_outcome.resolved:
                        logger.info("Slock: autonomous resolution succeeded, treating as task")
                        resolved_text = resolve_outcome.resolved_text or text or ""
                        if self.client._should_auto_activate_slock(chat_id, resolved_text, chat_type=chat_type):
                            activated, reason = self.client._auto_activate_slock(chat_id, resolved_text, project)
                            if activated:
                                self.client._add_reaction(message_id, EmojiReaction.on_processing())
                                return
                            else:
                                card = build_activation_denied_card(
                                    reason=reason,
                                    hint="自动协作模式激活受限。",
                                )
                                self.client._reply_card(
                                    message_id,
                                    json.dumps(card, ensure_ascii=False),
                                )
                                return

                    # Resolution failed or needs clarification — ask user
                    logger.info("Slock: autonomous resolution inconclusive, asking user clarification: chat=%s", chat_id)
                    sender_id = get_current_sender_id() or ""
                    card = build_clarification_card(
                        message_preview=text or "",
                        channel_id=chat_id,
                        message_id=message_id,
                        sender_id=sender_id,
                    )
                    self.client._reply_card(
                        message_id,
                        json.dumps(card, ensure_ascii=False),
                    )
                    return
                elif result.label == SlockMessageClass.TASK and self.client._should_auto_activate_slock(chat_id, text, chat_type=chat_type):
                    activated, reason = self.client._auto_activate_slock(chat_id, text, project)
                    if activated:
                        self.client._add_reaction(message_id, EmojiReaction.on_processing())
                        return
                    else:
                        logger.info(
                            "Slock auto-activation denied for chat=%s, reason=%s",
                            chat_id, reason,
                        )
                        import json

                        card = build_activation_denied_card(
                            reason=reason,
                            hint="自动协作模式激活受限。",
                        )
                        self.client._reply_card(
                            message_id,
                            json.dumps(card, ensure_ascii=False),
                        )
                        return
        elif slock_context_allowed and not _is_smart_shell:
            # Legacy mode: require both active AND managed
            if self.client._is_slock_active(chat_id) and _is_managed:
                from src.slock_engine.task_classifier import TaskClassifier

                is_ambient, ambient_confidence = TaskClassifier.classify(
                    text or "", managed_chat=True
                )
                if not current_dispatch_allowed():
                    return
                if is_ambient and ambient_confidence >= 0.7:
                    return
                self.client._add_reaction(message_id, EmojiReaction.on_processing())
                self.client._handle_slock_message(message_id, chat_id, text, project)
                return

        if is_in_programming and self.client._is_exit_command(text):
            if not current_dispatch_allowed():
                return
            self.client._add_reaction(message_id, EmojiReaction.on_coco_mode())
            if self.client._control_plane.should_defer_exit(chat_id=chat_id, project_id=_pid):
                self.client._control_plane.request_deferred_exit(message_id=message_id, chat_id=chat_id, project_id=_pid)
                self.client._reply_text(message_id, UI_TEXT["ws_exit_deferred_msg"])
                return
            self.client._exit_current_mode(message_id, chat_id, project=project)
            return

        # Request-scoped slash parsing: direct system commands must not fall
        # through to intent recognition, which can be slow or ambiguous.
        if self.client._is_interceptable_command_match(command_match):
            if not current_dispatch_allowed():
                return
            self.client._handle_intercepted_command(message_id, chat_id, text, project, command_match=command_match)
            return

        if command_match is not None:
            if not current_dispatch_allowed():
                return
            self.client._handle_intercepted_command(
                message_id,
                chat_id,
                text,
                project,
                command_match=command_match,
            )
            return

        # Worktree mode
        if project and self.client._is_worktree_awaiting_goal(project):
            if not current_dispatch_allowed():
                return
            self.client._add_reaction(message_id, EmojiReaction.on_processing())
            self.client._handle_worktree_execute(message_id, chat_id, text, project)
            return

        # Programming mode (Coco / Claude / TTADK): exit or forward to active session
        if is_in_programming:
            if not current_dispatch_allowed():
                return
            self.client._add_reaction(message_id, EmojiReaction.on_coco_mode())
            self.client._add_reaction(message_id, EmojiReaction.on_processing())
            handler = self.client._get_mode_handler(current_mode)
            if handler:
                handler.handle_message(message_id, chat_id, text, project)
            else:
                self.client._show_help(message_id, chat_id)
            return

        # SMART mode: image-only messages bypass intent recognition
        with self.client._pending_image_lock:
            is_image_only = message_id in self.client._pending_image_only
        if is_image_only:
            if not current_dispatch_allowed():
                return
            self.client._add_reaction(message_id, EmojiReaction.on_coco_mode())
            self.client._add_reaction(message_id, EmojiReaction.on_processing())
            self.client._handle_coco_message(message_id, chat_id, text, project)
            return

        # SMART mode: intent recognition
        self.client._add_reaction(message_id, EmojiReaction.on_smart_mode())
        self.client._add_reaction(message_id, EmojiReaction.on_processing())

        try:
            intent_result = self.client._intent_recognizer.recognize(text, current_mode.value)
        except (RuntimeError, TimeoutError, ValueError, TypeError) as e:
            classification = classify_dispatch_error(e, phase="intent_recognition")
            if classification.action == DispatchErrorAction.FALLBACK_TO_SHELL:
                if not self._action_matrix_allows(
                    effective_trust,
                    action_name="host_shell",
                ):
                    return
                if not current_dispatch_allowed():
                    return
                logger.warning("意图识别异常，回退到 shell: %s", get_error_detail(e))
                working_dir = self.client._get_working_dir(chat_id)
                self.client._submit_shell_command(message_id, chat_id, text, working_dir, project)
                return
            elif classification.action == DispatchErrorAction.LOG_AND_CONTINUE:
                logger.debug("意图识别 best-effort 失败: %s", get_error_detail(e))
                return
            elif classification.action == DispatchErrorAction.RAISE:
                logger.error("意图识别异常: %s", get_error_detail(e), exc_info=True)
                raise
            # Default: log and fallback to shell
            logger.error("意图识别异常: %s", get_error_detail(e))
            if not self._action_matrix_allows(
                effective_trust,
                action_name="host_shell",
            ):
                return
            working_dir = self.client._get_working_dir(chat_id)
            if not current_dispatch_allowed():
                return
            self.client._submit_shell_command(message_id, chat_id, text, working_dir, project)
            return

        logger.info(
            "意图识别: %s (置信度: %.2f, 任务数: %d)",
            intent_result.primary_intent.value,
            intent_result.confidence,
            len(intent_result.tasks),
        )

        if intent_result.is_multi_task:
            if not current_dispatch_allowed():
                return
            if any(
                task.intent is IntentType.SHELL_COMMAND
                for task in intent_result.tasks
            ) and not self._action_matrix_allows(
                effective_trust,
                action_name="host_shell",
            ):
                return
            multi_kwargs = {}
            if effective_trust is not None:
                multi_kwargs["effective_trust"] = effective_trust
            self.execute_multi_tasks(
                message_id,
                chat_id,
                intent_result,
                project,
                **multi_kwargs,
            )
        else:
            task = intent_result.tasks[0] if intent_result.tasks else None
            if (
                task is not None
                and task.intent is IntentType.SHELL_COMMAND
                and not self._action_matrix_allows(
                    effective_trust,
                    action_name="host_shell",
                )
            ):
                return
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
        }[action_name]
        return ActionMatrix().decide(
            ActionRequest(
                trust=effective_trust,
                action=action,
                target=ActionTargetKind.HOST_GLOBAL,
            )
        ) is ActionDecision.ALLOW

    def execute_multi_tasks(
        self,
        message_id: str,
        chat_id: str,
        intent_result: 'IntentResult',
        project: Optional['ProjectContext'] = None,
        *,
        effective_trust: Optional['EffectiveTrust'] = None,
    ):
        """执行多任务计划（逐步执行；遇到失败停止后续步骤）。"""
        tasks = intent_result.tasks

        task_list = [{"description": task.description or self.get_task_description(task)} for task in tasks]
        plan_msg = fmt.format_multi_task_plan(task_list)
        self.client._reply_text(message_id, plan_msg)

        self.client._add_reaction(message_id, EmojiReaction.on_multi_task_start())

        all_success = True
        for i, task in enumerate(tasks, 1):
            if effective_trust is not None:
                gate = getattr(self.client, "_current_trust_can_dispatch", None)
                if not callable(gate) or gate(
                    effective_trust,
                    project=project,
                ) is not True:
                    all_success = False
                    break
            success = self.execute_task_step(
                message_id, chat_id, task, step_num=i, total_steps=len(tasks), project=project
            )

            if task.intent in {
                IntentType.ENTER_COCO,
                IntentType.ENTER_CLAUDE,
                IntentType.ENTER_AIDEN,
                IntentType.ENTER_CODEX,
                IntentType.ENTER_GEMINI,
                IntentType.ENTER_TRAEX,
                IntentType.TTADK_MESSAGE,
            }:
                break

            if not success:
                all_success = False
                self.client._reply_text(message_id, UI_TEXT["multi_task_step_failed"].format(step=i))
                break

        if all_success:
            self.client._add_reaction(message_id, EmojiReaction.on_multi_task_done())
        else:
            self.client._add_reaction(message_id, EmojiReaction.on_error())

    # ------------------------------------------------------------------
    # Dict-driven dispatch tables for execute_single_task
    # ------------------------------------------------------------------
    _MODE_ENTER_MAP: dict = {
        IntentType.ENTER_CLAUDE: "claude",
        IntentType.ENTER_AIDEN: "aiden",
        IntentType.ENTER_CODEX: "codex",
        IntentType.ENTER_GEMINI: "gemini",
        IntentType.ENTER_TRAEX: "traex",
    }

    _MODE_EXIT_MAP: dict = {
        IntentType.EXIT_COCO: "coco",
        IntentType.EXIT_CLAUDE: "claude",
        IntentType.EXIT_AIDEN: "aiden",
        IntentType.EXIT_CODEX: "codex",
        IntentType.EXIT_GEMINI: "gemini",
        IntentType.EXIT_TRAEX: "traex",
    }

    _MODE_MESSAGE_MAP: dict = {
        IntentType.COCO_MESSAGE: "coco",
        IntentType.CLAUDE_MESSAGE: "claude",
        IntentType.AIDEN_MESSAGE: "aiden",
        IntentType.CODEX_MESSAGE: "codex",
        IntentType.GEMINI_MESSAGE: "gemini",
        IntentType.TRAEX_MESSAGE: "traex",
    }

    _MODEL_SELECT_ENTER_MODES: set[str] = {"codex", "traex"}

    _ENGINE_ENTER_MAP: dict = {
        IntentType.ENTER_DEEP: "_start_deep_engine",
        IntentType.ENTER_SPEC: "_start_spec_engine",
    }

    _SIMPLE_ENGINE_DISPATCH: dict = {
        IntentType.SPEC_STATUS: "_show_spec_status",
        IntentType.STOP_SPEC: "_stop_spec_engine",
        IntentType.SPEC_PAUSE: "_pause_spec_engine",
        IntentType.SPEC_RESUME: "_resume_spec_engine",
    }

    _ENGINE_GUIDE_MAP: dict = {
        IntentType.DEEP_UPDATE: ("_update_deep_context", "📝 请提供上下文信息\n\n用法: `/deep_update <上下文描述>`"),
        IntentType.SPEC_GUIDE: ("_update_spec_guidance", "📝 请提供引导信息\n\n用法: `/spec_guide <引导描述>`"),
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

        # Mode enter
        if intent == IntentType.ENTER_COCO:
            if data.get("auto_forward") is True:
                self._auto_enter_and_forward("coco", message_id, chat_id, original_text, project)
            else:
                self._handle_enter_coco(message_id, chat_id, project)
        elif intent in self._MODE_ENTER_MAP:
            mode = self._MODE_ENTER_MAP[intent]
            if data.get("auto_forward") is True:
                self._auto_enter_and_forward(mode, message_id, chat_id, original_text, project)
            elif mode in self._MODEL_SELECT_ENTER_MODES:
                self._handle_enter_acp_mode(mode, message_id, chat_id, project)
            else:
                getattr(self.client, f"_enter_{mode}_mode")(message_id, chat_id, project=project)
        # Mode exit
        elif intent == IntentType.EXIT_MODE:
            self.client._exit_current_mode(message_id, chat_id, project=project)
        elif intent in self._MODE_EXIT_MAP:
            mode = self._MODE_EXIT_MAP[intent]
            getattr(self.client, f"_exit_{mode}_mode")(message_id, chat_id, project=project)
        # Mode message
        elif intent in self._MODE_MESSAGE_MAP:
            self._handle_mode_message(self._MODE_MESSAGE_MAP[intent], data, message_id, chat_id, original_text, project)
        elif intent == IntentType.TTADK_MESSAGE:
            self._handle_ttadk_message(data, message_id, chat_id, original_text, project)
        # System commands
        elif intent == IntentType.CHANGE_DIR:
            self.client._change_directory(message_id, chat_id, data.get("path", ""), project)
        elif intent == IntentType.SHOW_HELP:
            self.client._show_full_help(message_id, chat_id, project)
        elif intent == IntentType.SHOW_TOOLS:
            self.client._system_handler.show_tools_list(message_id, chat_id, project)
        elif intent == IntentType.TOOLS_STATUS:
            self.client._system_handler.show_tools_status(message_id, chat_id, project)
        # Project commands
        elif intent in self._PROJECT_INTENTS:
            self._dispatch_project(intent, data, message_id, chat_id, project)
        # Engine enter
        elif intent in self._ENGINE_ENTER_MAP:
            requirement = data.get("requirement") or original_text
            getattr(self.client, self._ENGINE_ENTER_MAP[intent])(message_id, chat_id, requirement, project)
        # Engine status/control (simple)
        elif intent in self._SIMPLE_ENGINE_DISPATCH:
            getattr(self.client, self._SIMPLE_ENGINE_DISPATCH[intent])(message_id, chat_id, project)
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
            self.client._reply_text(message_id, fmt.format_unknown_intent())

    # ------------------------------------------------------------------
    # Extracted helpers for execute_single_task
    # ------------------------------------------------------------------

    def _handle_no_task(self, message_id: str, chat_id: str):
        from ..thread import get_current_thread_id
        if self.client.settings.thread_programming_enabled and not get_current_thread_id():
            active_thread = self.client._find_active_thread(chat_id)
            if active_thread:
                mode_display = active_thread.mode.upper() if active_thread.mode else "编程"
                self.client._reply_text(
                    message_id,
                    UI_TEXT["ws_active_topic_msg"].format(name=mode_display),
                )
                return
        self.client._reply_text(message_id, "🤔 无法理解你的意图")

    def _handle_enter_coco(self, message_id: str, chat_id: str, project, *, pending_prompt: Optional[str] = None):
        # If already in coco mode, skip model selection — just forward the pending prompt
        _pid = project.project_id if project else None
        if self.client._mode_manager.is_coco_mode(chat_id, project_id=_pid):
            if pending_prompt:
                handler = self.client._get_mode_handler(
                    self.client._mode_manager.get_mode(chat_id, project_id=_pid)
                )
                if handler:
                    handler.handle_message(message_id, chat_id, pending_prompt, project)
            return

        try:
            self.client._system_handler.handle_select_acp_tool(
                message_id, chat_id, "coco", project_id=_pid,
                pending_prompt=pending_prompt,
            )
        except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as e:
            classify_dispatch_error(e, phase="coco_model_card")
            logger.warning("展示 Coco 模型选择卡失败，回退直接进入: %s", get_error_detail(e))
            self.client._enter_coco_mode(message_id, chat_id, project=project)
            # Best-effort: forward the pending prompt after fallback entry.
            if pending_prompt:
                handle_fn = getattr(self.client, "_handle_coco_message", None)
                if handle_fn:
                    try:
                        handle_fn(message_id, chat_id, pending_prompt, project)
                    except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as fwd_err:
                        classify_dispatch_error(fwd_err, phase="pending_prompt_forward")
                        logger.warning("fallback 转发 pending prompt 失败: %s", str(fwd_err))

    def _handle_enter_acp_mode(self, mode: str, message_id: str, chat_id: str, project, *, pending_prompt: Optional[str] = None):
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

        try:
            self.client._system_handler.handle_select_acp_tool(
                message_id, chat_id, mode, project_id=_pid,
                pending_prompt=pending_prompt,
            )
        except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as e:
            classify_dispatch_error(e, phase=f"{mode}_model_card")
            logger.warning("展示 %s 模型选择卡失败，回退直接进入: %s", mode, get_error_detail(e))
            enter_fn = getattr(self.client, f"_enter_{mode}_mode", None)
            if enter_fn:
                enter_fn(message_id, chat_id, project=project)
            if pending_prompt:
                handle_fn = getattr(self.client, f"_handle_{mode}_message", None)
                if handle_fn:
                    try:
                        handle_fn(message_id, chat_id, pending_prompt, project)
                    except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as fwd_err:
                        classify_dispatch_error(fwd_err, phase="pending_prompt_forward")
                        logger.warning("fallback 转发 pending prompt 失败: %s", str(fwd_err))

    def _auto_enter_and_forward(self, mode: str, message_id: str, chat_id: str, text: str, project):
        """Auto-enter programming mode and forward message (default ACP tool)."""
        enter_fn = getattr(self.client, f"_enter_{mode}_mode", None)
        if enter_fn:
            enter_fn(message_id, chat_id, silent=True, project=project)
        handle_fn = getattr(self.client, f"_handle_{mode}_message", None)
        if handle_fn:
            handle_fn(message_id, chat_id, text, project)
        else:
            logger.warning("默认工具模式 %s 无消息处理器", mode)

    def _handle_mode_message(self, mode: str, data: dict, message_id: str, chat_id: str, original_text: str, project):
        if data.get("command") == "info":
            getattr(self.client, f"_show_{mode}_info")(message_id, chat_id, project)
        else:
            getattr(self.client, f"_handle_{mode}_message")(message_id, chat_id, original_text, project)

    def _handle_ttadk_message(self, data: dict, message_id: str, chat_id: str, original_text: str, project):
        from ..mode import InteractionMode
        if data.get("command") == "info":
            self.client._show_ttadk_info(message_id, chat_id, project)
        elif str(original_text or "").strip().lower() in {"/ttadk", "/enter_ttadk"}:
            self.client._handle_ttadk_command(message_id, chat_id, project)
        else:
            _pid = project.project_id if project else None
            mode = self.client._mode_manager.get_mode(chat_id, project_id=_pid)
            if mode != InteractionMode.TTADK:
                self.client._enter_ttadk_mode(message_id, chat_id, project=project)
            self.client._ttadk_handler.handle_message(message_id, chat_id, original_text, project)

    def _dispatch_project(self, intent, data: dict, message_id: str, chat_id: str, project):
        if intent == IntentType.CREATE_PROJECT:
            name = data.get("name", "")
            path = data.get("path", "")
            working_dir = self.client._get_working_dir(chat_id)
            if not path:
                path = working_dir
            if not name:
                name = os.path.basename(os.path.normpath(path))
                if not name or name in (".", "/", "~"):
                    name = f"project_{int(time.time())}"
            self.client._create_project(message_id, chat_id, name, path)
        elif intent == IntentType.SWITCH_PROJECT:
            name = data.get("name", "")
            if name:
                self.client._switch_project(message_id, chat_id, name)
            else:
                self.client._show_project_board(message_id, chat_id)
        elif intent == IntentType.LIST_PROJECTS:
            self.client._show_project_board(message_id, chat_id)
        elif intent == IntentType.CLOSE_PROJECT:
            name = data.get("name", "")
            if name:
                self.client._close_project(message_id, chat_id, name)
            else:
                self.client._reply_text(message_id, "❌ 请指定要关闭的项目名称")
        elif intent == IntentType.PROJECT_STATUS:
            self.client._show_project_status(message_id, chat_id, project)
        elif intent == IntentType.NEW_CHAT_PROJECT:
            self.client._handle_new_chat_project(message_id, chat_id, data)

    def _handle_deep_status_or_stop(self, intent, data: dict, message_id: str, chat_id: str, project):
        arg = (data.get("arg") or "").strip().lower()
        is_all = arg in ("all", "-a", "--all")
        if intent == IntentType.DEEP_STATUS:
            if is_all:
                self.client._show_deep_board(message_id, chat_id)
            else:
                self.client._show_deep_status(message_id, chat_id, project)
        else:  # STOP_DEEP
            if is_all:
                self.client._stop_all_deep_engines(message_id, chat_id)
            else:
                self.client._stop_deep_engine(message_id, chat_id, project)

    def _handle_engine_guide(self, intent, data: dict, message_id: str, chat_id: str, project):
        method_name, hint = self._ENGINE_GUIDE_MAP[intent]
        guide_message = data.get("message")
        if guide_message:
            getattr(self.client, method_name)(message_id, chat_id, guide_message, project)
        else:
            self.client._reply_text(message_id, hint)

    def _dispatch_shell(self, data: dict, message_id: str, chat_id: str, original_text: str, project, shell_fast_tracked: bool):
        working_dir = self.client._get_working_dir(chat_id)
        cmd = data.get("command") or original_text
        if shell_fast_tracked:
            self.client._system_handler.execute_shell_and_reply(message_id, chat_id, cmd, working_dir, project)
        else:
            self.client._submit_shell_command(message_id, chat_id, cmd, working_dir, project)
        if project:
            project.add_conversation("user", cmd, message_id)
            self.client._context_manager.update_context(
                project.project_id,
                conversation={"role": "user", "content": cmd, "source_mode": "shell", "message_id": message_id},
                chat_id=chat_id,
            )

    def execute_task_step(
        self,
        message_id: str,
        chat_id: str,
        task: 'TaskStep',
        step_num: int,
        total_steps: int,
        project: Optional['ProjectContext'] = None,
    ) -> bool:
        """执行一个 TaskStep，并返回是否成功。"""
        intent = task.intent
        data = task.data
        desc = task.description or self.get_task_description(task)

        logger.info("执行步骤 %d/%d: %s", step_num, total_steps, desc)

        try:
            if intent == IntentType.ENTER_COCO:
                self.client._enter_coco_mode(message_id, chat_id, silent=True, project=project)
                self.client._reply_text(message_id, UI_TEXT["multi_task_step_success"].format(step=step_num, desc="已进入 Coco 模式"))
                return True

            elif intent == IntentType.EXIT_COCO:
                success = self.client._coco_manager.end_session(chat_id)
                if success:
                    self.client._reply_text(message_id, UI_TEXT["multi_task_step_success"].format(step=step_num, desc="已退出 Coco 模式"))
                return success

            elif intent == IntentType.CHANGE_DIR:
                path = data.get("path", "")
                if not path:
                    current_dir = self.client._get_working_dir(chat_id)
                    self.client._reply_text(message_id, f"✅ 步骤 {step_num}: 当前目录 {current_dir}")
                    return True

                success, result = self.client._set_working_dir(chat_id, path)
                if success:
                    self.client._reply_text(message_id, UI_TEXT["multi_task_step_success"].format(step=step_num, desc=f"已切换到 {result}"))
                else:
                    self.client._reply_text(message_id, UI_TEXT["multi_task_step_error"].format(step=step_num, error=result))
                return success

            elif intent == IntentType.CREATE_PROJECT:
                name = data.get("name", "")
                path = data.get("path", "")
                if not name:
                    name = f"project_{int(time.time())}"
                if not path:
                    path = self.client._get_working_dir(chat_id)
                project_id = name.lower().replace(" ", "_").replace("-", "_")
                success, msg, new_project = self.client._project_manager.create_project(
                    project_id=project_id, project_name=name, root_path=path, chat_id=chat_id
                )
                if success:
                    self.client._reply_text(message_id, f"✅ 步骤 {step_num}: 已创建项目 {name}")
                    project = new_project
                else:
                    self.client._reply_text(message_id, f"❌ 步骤 {step_num}: {msg}")
                return success

            elif intent == IntentType.SWITCH_PROJECT:
                name = data.get("name", "")
                if name:
                    found_project = self.client._project_manager.find_project_by_name(name, chat_id=chat_id)
                    if found_project:
                        success, msg = self.client._project_manager.set_active_project(chat_id, found_project.project_id)
                        if success:
                            self.client._reply_text(message_id, f"✅ 步骤 {step_num}: 已切换到项目 {name}")
                        return success
                return False

            elif intent == IntentType.SHELL_COMMAND:
                cmd = data.get("command", task.description)
                if cmd:
                    working_dir = self.client._get_working_dir(chat_id)
                    self.client._system_handler.execute_shell_and_reply(message_id, chat_id, cmd, working_dir, project)
                return True
            elif intent == IntentType.TTADK_MESSAGE:
                self.client._enter_ttadk_mode(message_id, chat_id, silent=True, project=project)
                self.client._reply_text(message_id, UI_TEXT["multi_task_step_success"].format(step=step_num, desc="已进入 TTADK 模式"))
                return True

            else:
                return False

        except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as e:
            classify_dispatch_error(e, phase="multi_task_step")
            logger.error("执行步骤 %d 异常: %s", step_num, get_error_detail(e))
            return False

    _TASK_DESCRIPTIONS: dict = {
        IntentType.ENTER_COCO: "进入 Coco 编程模式",
        IntentType.ENTER_CLAUDE: "进入 Claude 编程模式",
        IntentType.ENTER_AIDEN: "进入 Aiden 编程模式",
        IntentType.ENTER_CODEX: "进入 Codex 编程模式",
        IntentType.ENTER_GEMINI: "进入 Gemini 编程模式",
        IntentType.TTADK_MESSAGE: "进入 TTADK 编程模式",
        IntentType.EXIT_COCO: "退出 Coco 模式",
        IntentType.EXIT_CLAUDE: "退出 Claude 模式",
        IntentType.EXIT_AIDEN: "退出 Aiden 模式",
        IntentType.EXIT_CODEX: "退出 Codex 模式",
        IntentType.EXIT_GEMINI: "退出 Gemini 模式",
        IntentType.LIST_PROJECTS: "查看项目列表",
        IntentType.PROJECT_STATUS: "查看项目状态",
        IntentType.SHELL_COMMAND: "执行命令",
    }

    # Intents whose description uses ``data["name"]`` or ``data["path"]``
    _TASK_DESC_WITH_DATA: dict = {
        IntentType.CHANGE_DIR: ("path", "切换到目录: {v}", "查看当前目录"),
        IntentType.CREATE_PROJECT: ("name", "创建项目: {v}", "创建新项目"),
        IntentType.SWITCH_PROJECT: ("name", "切换到项目: {v}", "切换项目"),
        IntentType.CLOSE_PROJECT: ("name", "关闭项目: {v}", "关闭项目"),
    }

    def get_task_description(self, task: 'TaskStep') -> str:
        """为 TaskStep 生成可读描述（用于多任务计划展示）。"""
        intent = task.intent
        desc = self._TASK_DESCRIPTIONS.get(intent)
        if desc:
            return desc
        tpl = self._TASK_DESC_WITH_DATA.get(intent)
        if tpl:
            key, fmt, fallback = tpl
            v = task.data.get(key, "")
            return fmt.format(v=v) if v else fallback
        return UI_TEXT["task_desc_unknown"]
