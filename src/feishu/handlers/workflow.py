"""Workflow Engine handler — /wf, /workflow, /stop_wf, /wf_status, /wf_help commands."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from src.card.render.buttons import build_responsive_button_row

from ...utils.text import generate_task_id
from ..emoji import EmojiReaction
from .engine_base import BaseEngineHandler

if TYPE_CHECKING:
    from ...project import ProjectContext
    from ...workflow_engine.selection_flow import SelectionFlowController
    from ..handler_context import HandlerContext

logger = logging.getLogger(__name__)

_WORKFLOW_MODEL_CACHE_TTL_S = 5 * 60
_WORKFLOW_MODEL_PREHEAT_LIMIT = 2
_WORKFLOW_ACP_MODEL_TOOLS = {"coco", "claude", "aiden", "codex", "gemini", "traex"}


@dataclass(frozen=True, slots=True)
class _WorkflowLifecycleOwner:
    """Linearization token shared by generation, card delivery, and queued start."""

    session_key: str
    initiator_user_id: str = ""
    stop_event: threading.Event = field(default_factory=threading.Event)
    heartbeat_stop_event: threading.Event = field(default_factory=threading.Event)
    delivery_lock: Any = field(default_factory=threading.RLock)
    source_script_path: str | None = None
    execution_script_path: str | None = None
    done_event: threading.Event = field(default_factory=threading.Event)
    claimed_event: threading.Event = field(default_factory=threading.Event)
    worker_started_event: threading.Event = field(
        default_factory=threading.Event,
    )
    worker_thread_id: int | None = field(
        default=None,
        compare=False,
    )


class _WorkflowGenerationCancelled(RuntimeError):
    """Internal control flow: a superseded generator must not write fallback."""


def _workflow_pending_statuses():
    """States that own a pending Workflow card/session rather than a runtime run."""
    from ...workflow_engine.models import WorkflowStatus

    return {
        WorkflowStatus.GENERATING_SCRIPT,
        WorkflowStatus.AWAITING_AGENT_SELECT,
        WorkflowStatus.AWAITING_TOOL_SELECT,
        WorkflowStatus.AWAITING_CONFIRM,
    }


from .workflow_script import WorkflowScriptMixin  # noqa: E402
from .workflow_selection import (  # noqa: E402
    WorkflowSelectionMixin,
    workflow_model_name_is_available,
)


class WorkflowHandler(WorkflowSelectionMixin, WorkflowScriptMixin, BaseEngineHandler):
    """Manages the full lifecycle of Workflow Engine tasks.

    Commands:
        /wf <requirement>       — Start a new workflow (AI generates script)
        /wf <template> [args]   — Start from a built-in/saved template
        /stop_wf                — Cancel the running workflow
        /wf_status              — Show current workflow progress
    """

    def __init__(self, ctx: "HandlerContext") -> None:
        super().__init__(ctx)
        # Workflow uses its own renderer (card JSON comes from WorkflowProgressRenderer)
        from ...workflow_engine.renderer import WorkflowProgressRenderer  # noqa: F401

    # ------------------------------------------------------------------
    # Topic-engine free-text entry point
    # ------------------------------------------------------------------
    def handle_message(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        """Handle free-text messages when auto_enter_mode == 'workflow'.

        Treats the entire text as a workflow requirement and starts generation.
        """
        text_stripped = text.strip()
        if not text_stripped:
            return
        self.start_workflow(message_id, chat_id, text_stripped, project)

    # ------------------------------------------------------------------
    # Command router
    # ------------------------------------------------------------------
    def handle_workflow_command(
        self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None
    ):
        """Route /wf, /workflow, /stop_wf, /wf_status commands."""
        from ...utils.command_parser import CommandParser

        cmd = CommandParser.parse_basic(text)
        command = cmd.command

        if command in ("/stop_wf", "/stop_workflow"):
            self.stop_workflow(message_id, chat_id, project)
        elif command in ("/wf_status", "/workflow_status"):
            self.show_workflow_status(message_id, chat_id, project)
        elif command in ("/wf_help", "/workflow_help"):
            self.show_workflow_help(message_id)
        elif command in ("/wf_save", "/workflow_save"):
            self._handle_wf_save(message_id, chat_id, cmd.args, project)
        elif command in ("/wf_list", "/workflow_list"):
            self._handle_wf_list(message_id, chat_id, project)
        elif command in ("/wf_delete", "/workflow_delete"):
            self._handle_wf_delete(message_id, chat_id, cmd.args, project)
        elif command in ("/wf_history", "/workflow_history"):
            self._handle_wf_history(message_id, chat_id, project)
        elif command in ("/wf", "/workflow"):
            # /wf <需求> — 主编排 Agent 入口：①选择主编排Agent（工具+模型）→ ②选择评审Agent（工具+模型，可多选或Auto）→ 自动生成并执行
            arg = cmd.args
            if arg:
                self.start_workflow(message_id, chat_id, arg, project)
            else:
                self._show_workflow_entry_card(message_id, chat_id, project)
        else:
            self._reply_workflow_error(
                message_id,
                "invalid_argument",
                detail=(
                    "未知命令。可用命令列表:\n"
                    "• `/wf <需求>` — 基于需求生成编排脚本并执行\n"
                    "• `/wf <模板名> [key=value ...]` — 使用已保存的模板\n"
                    "• `/wf_save <名称> [--global]` — 保存脚本为模板\n"
                    "• `/wf_list` — 列出可用模板\n"
                    "• `/wf_delete <名称> [--global]` — 删除模板\n"
                    "• `/wf_history` — 查看执行历史\n"
                    "• `/wf_status` — 查看当前 Workflow 进度\n"
                    "• `/wf_help` — 查看完整帮助\n"
                    "• `/stop_wf` — 停止正在执行的 Workflow\n"
                    "\n发送 `/wf_help` 查看完整说明。"
                ),
            )

    def _get_engine_manager(self):
        return self.ctx.workflow_engine_manager

    def _get_engine_name_prefix(self) -> str:
        return "Workflow"

    def _build_workflow_stepper(self, current: int, total: int = 3) -> dict[str, Any]:
        """Build a stepper element for the three-step orchestration flow
        (select main agent → select review agents → confirm/execution)."""
        from ...card.ui_text import UI_TEXT

        steps = [
            UI_TEXT["workflow_stepper_step_main_agent"],
            UI_TEXT["workflow_stepper_step_review_agents"],
            UI_TEXT["workflow_stepper_step_confirm"],
        ]
        clamped = max(1, min(total, current))
        lines = []
        for idx, label in enumerate(steps, start=1):
            if idx < clamped:
                lines.append(f"✓ {idx}. ~~{label}~~")
            elif idx == clamped:
                lines.append(f"▶ {idx}. **{label}**")
            else:
                lines.append(f"○ {idx}. {label}")
        header = UI_TEXT["workflow_stepper_current_label"].format(current=clamped, total=total)
        return {
            "tag": "markdown",
            "content": f"**{header}**\n" + "\n".join(lines),
        }

    def _get_task_type(self) -> str:
        return "workflow_engine"

    def _show_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        self.show_workflow_status(message_id, chat_id, project)

    def _create_callbacks(
        self, message_id: str, chat_id: str, project: Optional["ProjectContext"], engine_name: str, root_path: str
    ):
        return self._build_workflow_callbacks(message_id, chat_id, project)

    # ------------------------------------------------------------------
    # Entry card
    # ------------------------------------------------------------------

    def _show_workflow_entry_card(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Show the Workflow entry help card when user sends `/wf` without arguments.

        The primary "开始编排任务" button carries an empty `requirement` field
        so that its handler can prompt the user to type a task description inline
        before launching the three-step orchestration flow. The previous dead-end
        behavior (sending users back to the chat to type `/wf <需求>`) has been
        replaced with an inline requirement entry.
        """
        from ...card import CardBuilder
        from ...card.actions.dispatch import (
            SHOW_WORKFLOW_MENU,
            WORKFLOW_LIST_TEMPLATES,
            WORKFLOW_SHOW_HELP,
        )
        from ...card.ui_text import UI_TEXT

        project_id = project.project_id if project else ""

        start_value = {
            "action": SHOW_WORKFLOW_MENU,
            "chat_id": chat_id,
            "project_id": project_id,
            "requirement": "",
        }
        templates_value = {
            "action": WORKFLOW_LIST_TEMPLATES,
            "chat_id": chat_id,
            "project_id": project_id,
        }
        help_value = {
            "action": WORKFLOW_SHOW_HELP,
            "chat_id": chat_id,
            "project_id": project_id,
        }

        buttons = [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": UI_TEXT["workflow_entry_btn_start"]},
                "type": "primary",
                "value": start_value,
                "behaviors": [{"type": "callback", "value": start_value}],
            },
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": UI_TEXT["workflow_entry_btn_templates"]},
                "type": "default",
                "value": templates_value,
                "behaviors": [{"type": "callback", "value": templates_value}],
            },
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": UI_TEXT["workflow_entry_btn_help"]},
                "type": "default",
                "value": help_value,
                "behaviors": [{"type": "callback", "value": help_value}],
            },
        ]

        card_content = CardBuilder._wrap_card(
            header_title=UI_TEXT["workflow_entry_title"],
            header_template="turquoise",
            elements=[
                {"tag": "markdown", "content": UI_TEXT["workflow_entry_body"]},
                {"tag": "hr"},
                *build_responsive_button_row(buttons, mobile_force_vertical=True),
            ],
        )
        self.reply_card(message_id, card_content)

    def _show_requirement_input_card(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"],
        root_path: str,
    ) -> None:
        """Prompt the user to enter a Workflow task description.

        Unlike earlier versions (which only printed guidance text), we
        now bind the topic to ``workflow`` mode so the user's very next
        free-text message is routed directly to :meth:`start_workflow`.
        We also emit an ``entry-cancel`` session key so the cancel
        handler can distinguish this prompt from a real pending session
        and update the card instead of reporting ``session_expired``.
        """
        from ...card import CardBuilder
        from ...card.actions.dispatch import WORKFLOW_CANCEL
        from ...card.render.buttons import build_responsive_button_row
        from ...thread import get_current_sender_id

        # Ensure a project exists for the topic binding
        if project is None:
            project = self._ensure_project(message_id, chat_id, None)
            if project is None:
                self._reply_workflow_error(message_id, "invalid_argument")
                return

        # Bind topic engine context so the *next* free-text message in
        # this chat is routed through WorkflowHandler.handle_message.
        # Users can still cancel the entry flow through the cancel
        # button below; the topic remains bound so subsequent messages
        # keep going through the workflow handler.
        self._ensure_topic_engine_context(
            mode="workflow",
            message_id=message_id,
            chat_id=chat_id,
            project=project,
        )

        project_id = project.project_id if project else ""

        elements: list[dict[str, Any]] = []
        elements.append(
            {
                "tag": "markdown",
                "content": (
                    "**请在下方聊天框中直接输入你想让 Workflow 完成的任务描述**\n\n"
                    "本聊天已切换到 Workflow 编排模式，下一条消息会被当作需求描述自动进入编排流程。\n\n"
                    "示例：\n"
                    "• `审查 src/handlers 目录中的错误处理一致性`\n"
                    "• `为 feature-checkout 分支生成端到端集成测试`\n\n"
                    "也可以直接发送：\n`/wf <你的任务描述>`"
                ),
            }
        )
        elements.append({"tag": "hr"})

        cancel_value = {
            "action": WORKFLOW_CANCEL,
            "chat_id": chat_id,
            "project_id": project_id,
            "engine_session_key": "entry-cancel",
            "initiator_user_id": get_current_sender_id() or "",
        }
        cancel_btn = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": "取消"},
            "type": "default",
            "value": cancel_value,
            "behaviors": [{"type": "callback", "value": cancel_value}],
        }
        elements.extend(build_responsive_button_row([cancel_btn], mobile_force_vertical=True))

        card = CardBuilder._wrap_card(
            header_title="Workflow — 输入任务描述",
            header_template="blue",
            elements=elements,
        )
        self.reply_card(message_id, card)

    def handle_show_workflow_menu(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle entry card "开始编排任务" button — launch the inline
        requirement-entry flow which then leads into three-step orchestration.

        When no task description is attached (the usual entry-card state), we
        prompt the user to enter a description inline via a structured input
        card. When a task description is already provided, we proceed directly
        to agent selection. Users can still skip the entry card entirely by
        typing ``/wf <描述>`` directly in chat.
        """
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = self._get_root_path(chat_id, project)

        # Check Node.js availability before showing anything (fail-fast UX).
        # Use the authoritative constant-text helper so that error messages
        # here, engine.py, and bridge.py all agree on the required version.
        from ...workflow_engine.bridge import RuntimeBridge
        from ...workflow_engine.engine import _node_version_required_text

        if not RuntimeBridge.check_node_available():
            self._reply_workflow_error(
                message_id,
                "invalid_argument",
                detail=_node_version_required_text(),
            )
            return

        # Check for existing running workflow
        existing = self.ctx.workflow_engine_manager.get(chat_id, root_path)
        if existing and existing.is_running is True:
            self._reply_workflow_error(
                message_id,
                "invalid_state",
                detail="当前项目已有 Workflow 任务在执行中。发送 `/wf_status` 查看进度，或 `/stop_wf` 停止任务。",
            )
            return

        # Merely opening the requirement prompt does not own a Workflow
        # lifecycle yet.  Admission happens only once a concrete requirement
        # exists, so an abandoned entry card cannot block another user.
        has_requirement = bool(value and isinstance(value, dict) and (value.get("requirement") or "").strip())
        if not has_requirement:
            self._show_requirement_input_card(
                message_id=message_id,
                chat_id=chat_id,
                project=project,
                root_path=root_path,
            )
            return

        from ...thread import get_current_sender_id

        admission_engine = existing or self.ctx.workflow_engine_manager.get_or_create(
            chat_id,
            root_path,
            engine_name=self.get_engine_name(
                chat_id,
                project_id=(project.project_id if project else None),
            ),
        )
        (
            takeover_ok,
            takeover_error,
            admission_owner,
        ) = self._supersede_incomplete_workflow(
            admission_engine,
            root_path=root_path,
            current_user=get_current_sender_id() or "",
        )
        if not takeover_ok:
            if takeover_error == "forbidden":
                self._reply_workflow_error(
                    message_id,
                    "forbidden",
                    detail="只能替换自己发起的未完成 Workflow；管理员可代为处理。",
                )
            else:
                self._reply_workflow_error(
                    message_id,
                    "invalid_state",
                    detail=("当前项目已有 Workflow 任务在执行中。发送 `/wf_status` 查看进度，或 `/stop_wf` 停止任务。"),
                )
            return

        requirement = (value.get("requirement") or "").strip()

        self._show_agent_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            requirement=requirement,
            project=project,
            root_path=root_path,
            admission_owner=admission_owner,
        )

    def handle_workflow_list_templates(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle entry card 'templates' button — list available templates."""
        project = self._resolve_project_from_id(project_id, chat_id)
        self._handle_wf_list(message_id, chat_id, project)

    def handle_workflow_show_help(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle entry card 'help' button — show workflow help."""
        self.show_workflow_help(message_id)

    # ------------------------------------------------------------------
    # Start workflow
    # ------------------------------------------------------------------

    def start_workflow(
        self,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Start a workflow: show tool selection first, then generate script.

        Flow (tool-selection-first):
        1. User sends /wf <requirement>
        2. Show tool selection card with recommended tools
        3. User confirms tool selection → generate script based on selected tools
        4. Show confirmation card with script preview and tool distinction
        5. User can regenerate script with different tools, or confirm to execute

        If `requirement` matches a known template name, skips tool selection and
        goes directly to script generation (template tools are fixed).
        """
        project = self._ensure_project(message_id, chat_id, project)
        if not project:
            return

        root_path = project.root_path if project else self.get_working_dir(chat_id)

        # Check for existing running workflow
        existing = self.ctx.workflow_engine_manager.get(chat_id, root_path)
        if existing and existing.is_running is True:
            self._reply_workflow_error(
                message_id,
                "invalid_state",
                detail="当前项目已有 Workflow 任务在执行中。发送 `/wf_status` 查看进度，或 `/stop_wf` 停止任务。",
            )
            return

        # Check Node.js availability
        from ...workflow_engine.bridge import RuntimeBridge
        from ...workflow_engine.engine import _node_version_required_text

        if not RuntimeBridge.check_node_available():
            self._reply_workflow_error(
                message_id,
                "invalid_argument",
                detail=_node_version_required_text(),
            )
            return

        # Input validation: requirement must be non-trivial
        _req_stripped = requirement.strip()
        if len(_req_stripped) < 4:
            self._reply_workflow_error(
                message_id,
                "invalid_argument",
                detail="需求描述过短，请提供更详细的说明（至少 4 个字符）。",
            )
            return
        if len(_req_stripped) > 4000:
            self._reply_workflow_error(
                message_id,
                "invalid_argument",
                detail="需求描述过长（超过 4000 字符），请精简后重试。",
            )
            return

        # Newest-wins applies only before the runtime thread has claimed the
        # reusable engine. The same initiator (or an admin) may discard an old
        # selection/generation/queued start; another user's task is protected.
        from ...thread import get_current_sender_id

        admission_engine = existing or self.ctx.workflow_engine_manager.get_or_create(
            chat_id,
            root_path,
            engine_name=self.get_engine_name(
                chat_id,
                project_id=(project.project_id if project else None),
            ),
        )
        (
            takeover_ok,
            takeover_error,
            admission_owner,
        ) = self._supersede_incomplete_workflow(
            admission_engine,
            root_path=root_path,
            current_user=get_current_sender_id() or "",
        )
        if not takeover_ok:
            if takeover_error == "forbidden":
                self._reply_workflow_error(
                    message_id,
                    "forbidden",
                    detail="只能替换自己发起的未完成 Workflow；管理员可代为处理。",
                )
            else:
                self._reply_workflow_error(
                    message_id,
                    "invalid_state",
                    detail=("当前项目已有 Workflow 任务在执行中。发送 `/wf_status` 查看进度，或 `/stop_wf` 停止任务。"),
                )
            return

        self.add_reaction(message_id, EmojiReaction.on_multi_task_start())

        # Bind engine mode to topic for auto-routing
        self._ensure_topic_engine_context(
            mode="workflow",
            message_id=message_id,
            chat_id=chat_id,
            project=project,
        )

        # Templates now also go through the standard three-step flow:
        #   agent selection → tool selection → confirmation.
        # This keeps templates consistent with AI-generated workflows and
        # gives the user a chance to inspect / override the template's
        # default tools before execution.
        # We detect whether `requirement` is a template name here so
        # downstream handlers can record it in pending state (e.g. to
        # initialize default tool selection from template meta.tools).
        from ...thread import get_current_sender_id
        from ...workflow_engine.templates import discover_templates

        sender_id = get_current_sender_id() or ""
        parts = requirement.strip().split(None, 1)
        template_name = parts[0] if parts else ""
        try:
            templates = discover_templates(root_path, user_id=sender_id)
            is_template = template_name in {t.name for t in templates}
        except Exception:
            # Template discovery failure should not block the workflow;
            # just treat this as an AI generation path.
            is_template = False

        # Initialize pending state early so we can mark this as a
        # template launch (needed for default tool selection hint).
        pre_engine = admission_engine
        if is_template:
            with pre_engine._lock:
                if (
                    vars(pre_engine).get("_workflow_selection_owner") is not admission_owner
                    or admission_owner is None
                    or admission_owner.stop_event.is_set()
                ):
                    return
                pending = pre_engine.project.pending if pre_engine.project else None
                if pending is None:
                    from ...workflow_engine.models import PendingConfirmation

                    pending = PendingConfirmation()
                    pre_engine.project.pending = pending
                pending.is_template_hint = template_name

        # Standard three-step flow regardless of template vs AI path.
        self._show_agent_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            requirement=requirement,
            project=project,
            root_path=root_path,
            admission_owner=admission_owner,
        )

    # ------------------------------------------------------------------
    # Error surface unification
    # ------------------------------------------------------------------

    def _build_error_card(
        self,
        category: str,
        *,
        detail: str = "",
    ) -> dict[str, Any]:
        """Build a standardized error card using the unified four-category surface.

        The supplied *detail* is always passed through the shared sanitizer
        (``sanitize_for_reply``) so that file paths, tracebacks, and internal
        module names never leak to the user. Only the sanitized user-facing
        message is rendered; raw details are logged but never shown.

        Args:
            category: One of "session_expired", "invalid_state", "invalid_argument", "forbidden", "internal_error"
            detail: Optional raw detail string — sanitized before rendering.
        """
        from ...card import CardBuilder
        from ...card.ui_text import UI_TEXT
        from ...workflow_engine.errors import (
            _strip_internal_details,
        )

        title_key = f"workflow_error_{category}_title"
        body_key = f"workflow_error_{category}_body"

        title = UI_TEXT.get(title_key, "操作失败")
        # Strip anything that looks like an internal traceback / file path
        # from the raw *detail* before rendering.  We intentionally do NOT
        # use ``sanitize_for_reply`` here - that helper already wraps the
        # message in a category-specific template, which would collide with
        # the UI_TEXT body template below and produce a duplicated prefix.
        safe_detail = _strip_internal_details(detail or "")

        raw_body = UI_TEXT.get(body_key, "")
        if raw_body and "{detail}" in raw_body:
            # Don't use format() — avoid unexpected kwargs when raw_body has
            # other placeholders. Do a simple literal replace instead.
            body = raw_body.replace("{detail}", safe_detail)
        elif raw_body:
            # Template does not define a {detail} placeholder; still surface
            # a sanitized user-facing hint when one is available so the user
            # sees "脚本被篡改/验证失败"这类可操作信息 rather than a
            # generic "服务内部错误".
            if safe_detail:
                body = raw_body.rstrip() + "\n\n🔎 细节：" + safe_detail
            else:
                body = raw_body
        else:
            body = safe_detail or "操作失败，请稍后重试。"

        # Header color by category
        header_template = "red"  # default for errors

        return CardBuilder._wrap_card(
            header_title=title,
            header_template=header_template,
            elements=[
                {"tag": "markdown", "content": body},
            ],
        )

    def _reply_workflow_error(
        self,
        message_id: str,
        category: str,
        *,
        detail: str = "",
    ) -> None:
        """Reply with a standardized error card."""
        card = self._build_error_card(category, detail=detail)
        self.reply_card(message_id, card)

    def _replace_or_send_workflow_card(
        self,
        *,
        card_message_id: str | None,
        chat_id: str,
        card: dict[str, Any],
        origin_message_id: str | None = None,
    ) -> None:
        """Replace an existing workflow card, falling back to a new card.

        Generation cards are chat-sent cards, so a failed patch would otherwise
        leave the user looking at a stale "生成脚本中" card.
        """
        if card_message_id and self.update_card(card_message_id, card):
            return
        self.send_card_to_chat(
            chat_id,
            card,
            origin_message_id=origin_message_id,
        )

    @staticmethod
    def _build_workflow_card_from_renderer_data(card_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize WorkflowProgressRenderer output to a full Feishu card.

        Workflow renderer helpers return ``{"header": ..., "elements": ...}``
        because they are pure renderers. Handler delivery APIs expect a full
        CardKit 2.0 card, so keep that conversion at the handler boundary.
        """
        raw_header = card_data.get("header") if isinstance(card_data, dict) else None
        header_source = raw_header if isinstance(raw_header, dict) else {}

        raw_title = header_source.get("title")
        if isinstance(raw_title, dict):
            title_content = str(raw_title.get("content") or "Workflow")
            title_tag = str(raw_title.get("tag") or "plain_text")
        else:
            title_content = str(raw_title or "Workflow")
            title_tag = "plain_text"

        header: dict[str, Any] = {
            "title": {"tag": title_tag, "content": title_content},
            "template": str(header_source.get("template") or "blue"),
        }
        raw_subtitle = header_source.get("subtitle")
        if isinstance(raw_subtitle, dict):
            subtitle_content = str(raw_subtitle.get("content") or "")
            if subtitle_content:
                header["subtitle"] = {
                    "tag": str(raw_subtitle.get("tag") or "plain_text"),
                    "content": subtitle_content,
                }

        body = card_data.get("body") if isinstance(card_data, dict) else None
        if isinstance(body, dict) and isinstance(body.get("elements"), list):
            elements = list(body["elements"])
        else:
            root_elements = card_data.get("elements") if isinstance(card_data, dict) else None
            elements = list(root_elements) if isinstance(root_elements, list) else []

        return {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": header,
            "body": {"elements": elements},
        }

    def _replace_or_send_workflow_rendered_card(
        self,
        *,
        card_message_id: str | None,
        chat_id: str,
        card_data: dict[str, Any],
        origin_message_id: str | None = None,
        fallback_to_new: bool = True,
    ) -> str | None:
        """Replace a Workflow renderer card, falling back to a new card.

        Returns the message id that should receive future progress updates.
        """
        card = self._build_workflow_card_from_renderer_data(card_data)
        if card_message_id and self.update_card(card_message_id, card):
            return card_message_id
        if not fallback_to_new:
            return card_message_id
        return self.send_card_to_chat(
            chat_id,
            card,
            origin_message_id=origin_message_id,
        )

    def _show_initial_workflow_progress_card(
        self,
        *,
        card_message_id: str,
        chat_id: str,
        wf_project: Any,
        origin_message_id: str | None = None,
    ) -> str:
        """Switch the confirmation card to a running progress card immediately."""
        try:
            from ...workflow_engine.renderer import WorkflowProgressRenderer

            card_data = WorkflowProgressRenderer(wf_project).render_progress_card()
            return (
                self._replace_or_send_workflow_rendered_card(
                    card_message_id=card_message_id,
                    chat_id=chat_id,
                    card_data=card_data,
                    origin_message_id=origin_message_id,
                )
                or card_message_id
            )
        except Exception:
            logger.debug("Failed to show initial workflow progress card", exc_info=True)
            return card_message_id

    @staticmethod
    def _resolve_tool_lists() -> tuple[dict[str, str], list[str], list[str], list[str]]:
        """Resolve available tools, recommended order, other tools, and default selection in one call.

        Returns:
            tuple of (all_tools_dict, recommended_tools_list, other_tools_list, default_selected_list)
        """
        from ...workflow_engine.tool_registry import get_available_tools

        all_tools = get_available_tools(require_available=True)
        all_tool_names = list(all_tools.keys())

        # Simple recommendation: prefer Traex as the default replacement path.
        recommended_order = ["traex", "claude", "codex", "aiden", "gemini", "coco"]
        recommended_tools = [t for t in recommended_order if t in all_tool_names]
        other_tools = [t for t in all_tool_names if t not in recommended_tools]

        # Default selection: top 3 recommended
        default_selected = recommended_tools[:3] if recommended_tools else all_tool_names[:1]

        return all_tools, recommended_tools, other_tools, default_selected

    def _init_tool_selection_state(
        self,
        chat_id: str,
        requirement: str,
        project: Optional["ProjectContext"],
        root_path: str,
        all_tools: dict[str, str],
        recommended_tools: list[str],
        default_selected: list[str],
    ) -> tuple[Any, str, str]:
        """Initialize tool selection state (separated from card building for update_card reuse).

        Args:
            all_tools: Resolved tools dict from _resolve_tool_lists()
            recommended_tools: Resolved recommended tools list from _resolve_tool_lists()
            default_selected: Resolved default selection list from _resolve_tool_lists()

        Returns:
            Tuple of (engine, project_id, session_key)
        """
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import PendingConfirmation, WorkflowStatus

        # Store pending state
        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
        engine = self.ctx.workflow_engine_manager.get_or_create(
            chat_id,
            root_path,
            engine_name=engine_name,
        )
        if not engine.project:
            from ...workflow_engine.models import WorkflowProject

            engine._project = WorkflowProject()

        existing_pending = engine.project.pending
        existing_orchestrator = existing_pending.orchestrator_agent if existing_pending else None
        template_hint = existing_pending.is_template_hint if existing_pending else None
        template_tools: list[str] = []
        if template_hint:
            try:
                from ...workflow_engine.script_gen import extract_meta_from_script
                from ...workflow_engine.templates import load_template

                template_content = load_template(
                    root_path,
                    template_hint,
                    user_id=get_current_sender_id() or None,
                )
                if template_content:
                    template_meta = extract_meta_from_script(template_content)
                    candidate_tools = list((template_meta or {}).get("tools", []))
                    template_tools = [t for t in candidate_tools if t in all_tools]
            except Exception:
                template_tools = []
        if template_tools:
            combined: list[str] = []
            for t in list(template_tools) + list(default_selected):
                if t not in combined:
                    combined.append(t)
            effective_default = combined
        else:
            effective_default = list(default_selected)

        engine.project.status = WorkflowStatus.AWAITING_TOOL_SELECT
        engine.project.pending = PendingConfirmation(
            requirement=requirement,
            initiator_user_id=get_current_sender_id() or "",
            engine_session_key=uuid.uuid4().hex,
            selected_tools=effective_default,
            script_path=None,
            meta=None,
            orchestrator_agent=existing_orchestrator,
            is_template_hint=template_hint,
        )

        project_id = project.project_id if project else ""
        session_key = engine.project.pending.engine_session_key if engine.project and engine.project.pending else ""

        return engine, project_id, session_key

    def _build_tool_selection_card(
        self,
        engine: Any,
        requirement: str,
        chat_id: str,
        project_id: str,
        session_key: str,
        all_tools: dict[str, str],
        recommended_tools: list[str],
        other_tools: list[str],
        default_selected: list[str],
    ) -> dict:
        """Build the tool selection card (pure rendering — no state already initialized).

        Uses update_card-friendly format — ready for both initial send and in-place updates.

        Args:
            all_tools: Resolved tools dict from _resolve_tool_lists()
            recommended_tools: Resolved recommended tools list from _resolve_tool_lists()
            other_tools: Resolved other tools list from _resolve_tool_lists()
            default_selected: Resolved default selection list from _resolve_tool_lists()
        """
        from ...card import CardBuilder
        from ...card.actions.dispatch import WORKFLOW_CANCEL, WORKFLOW_CONFIRM_TOOLS, WORKFLOW_SELECT_TOOL
        from ...card.render.buttons import build_responsive_button_row
        from ...card.ui_text import UI_TEXT

        # Build card
        elements: list[dict] = []

        # --- Stepper ---
        elements.append(self._build_workflow_stepper(current=2))

        # Requirement
        elements.append(
            {
                "tag": "markdown",
                "content": f"**需求**:\n> {requirement[:200]}",
            }
        )
        elements.append(
            {
                "tag": "markdown",
                "content": "**请选择此 Workflow 允许使用的工具**（脚本生成时将优先使用选中的工具）：",
            }
        )
        elements.append({"tag": "hr"})

        # Recommended tools section
        active_tools = (
            set(engine.project.pending.selected_tools)
            if engine.project and engine.project.pending
            else set(default_selected)
        )

        if recommended_tools:
            rec_display = " | ".join(f"**[✓ `{t}`]**" if t in active_tools else f"`{t}`" for t in recommended_tools)
            elements.append(
                {
                    "tag": "markdown",
                    "content": f"⭐ **推荐工具**: {rec_display}",
                }
            )

            rec_buttons = []
            for t in recommended_tools:
                is_selected = t in active_tools
                btn_value = {
                    "action": WORKFLOW_SELECT_TOOL,
                    "tool_name": t,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": session_key,
                }
                rec_buttons.append(
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": f"{'✓ ' if is_selected else '○ '}{t}"},
                        "type": "primary" if is_selected else "default",
                        "value": btn_value,
                        "behaviors": [{"type": "callback", "value": btn_value}],
                    }
                )
            elements.extend(build_responsive_button_row(rec_buttons, mobile_force_vertical=True))

        # Other tools section
        if other_tools:
            other_display = " | ".join(f"**[✓ `{t}`]**" if t in active_tools else f"`{t}`" for t in other_tools)
            other_buttons = []
            for t in other_tools:
                is_selected = t in active_tools
                btn_value = {
                    "action": WORKFLOW_SELECT_TOOL,
                    "tool_name": t,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": session_key,
                }
                other_buttons.append(
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": f"{'✓ ' if is_selected else '○ '}{t}"},
                        "type": "primary" if is_selected else "default",
                        "value": btn_value,
                        "behaviors": [{"type": "callback", "value": btn_value}],
                    }
                )
            elements.append(
                {
                    "tag": "collapsible_panel",
                    "expanded": False,
                    "header": {
                        "title": {
                            "tag": "plain_text",
                            "content": f"🔧 更多工具 ({len(other_tools)})",
                        },
                    },
                    "elements": [
                        {"tag": "markdown", "content": other_display},
                        *build_responsive_button_row(other_buttons, mobile_force_vertical=True),
                    ],
                }
            )

        elements.append({"tag": "hr"})

        # --- Action buttons: Cancel + Confirm Tools ---
        cancel_value = {
            "action": WORKFLOW_CANCEL,
            "chat_id": chat_id,
            "project_id": project_id,
            "engine_session_key": session_key,
        }
        confirm_value = {
            "action": WORKFLOW_CONFIRM_TOOLS,
            "chat_id": chat_id,
            "project_id": project_id,
            "engine_session_key": session_key,
        }
        action_buttons = [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "取消"},
                "type": "default",
                "value": cancel_value,
                "behaviors": [{"type": "callback", "value": cancel_value}],
                "confirm": {
                    "title": {"tag": "plain_text", "content": UI_TEXT["workflow_btn_confirm_cancel_title"]},
                    "text": {"tag": "plain_text", "content": UI_TEXT["workflow_btn_confirm_cancel_body"]},
                },
            },
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "确认工具并自动执行 →"},
                "type": "primary",
                "value": confirm_value,
                "behaviors": [{"type": "callback", "value": confirm_value}],
            },
        ]
        elements.extend(build_responsive_button_row(action_buttons, mobile_force_vertical=True))

        from ...card.ui_text import UI_TEXT

        tool_select_card = CardBuilder._wrap_card(
            header_title="🔧 Workflow — 选择工具",
            header_template=UI_TEXT["workflow_header_colors"].get("tool_select", "turquoise"),
            elements=elements,
        )
        return tool_select_card

    def _show_agent_selection_card(
        self,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: Optional["ProjectContext"],
        root_path: str,
        *,
        admission_owner: _WorkflowLifecycleOwner | None = None,
    ) -> None:
        """Show orchestrator tool selection card (Step 1 of 2-step flow).

        Uses SelectionFlowController for tool+model joint selection.
        The user selects ONE orchestrator agent (tool+model combo), then
        proceeds to review agent selection.
        """
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import PendingConfirmation, WorkflowStatus
        from ...workflow_engine.selection_flow import SelectionFlowController

        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
        engine = self.ctx.workflow_engine_manager.get_or_create(
            chat_id,
            root_path,
            engine_name=engine_name,
        )

        current_sender = get_current_sender_id() or ""
        selection_owner = admission_owner or _WorkflowLifecycleOwner(
            session_key=uuid.uuid4().hex,
            initiator_user_id=current_sender,
        )
        session_key = selection_owner.session_key
        previous_owner: _WorkflowLifecycleOwner | None = None
        ctrl = SelectionFlowController()
        with engine._lock:
            current_owner = vars(engine).get("_workflow_selection_owner")
            if admission_owner is not None and (
                current_owner is not admission_owner or admission_owner.stop_event.is_set()
            ):
                return
            if (
                admission_owner is None
                and isinstance(current_owner, _WorkflowLifecycleOwner)
                and current_owner is not selection_owner
            ):
                current_owner.stop_event.set()
                current_owner.heartbeat_stop_event.set()
                previous_owner = current_owner
                self._retire_workflow_owner(engine, current_owner)
            previous_hint: Optional[str] = None
            if engine.project and engine.project.pending:
                previous_hint = getattr(
                    engine.project.pending,
                    "is_template_hint",
                    None,
                )

            if not engine.project:
                from ...workflow_engine.models import WorkflowProject

                engine._project = WorkflowProject()

            engine.project.status = WorkflowStatus.AWAITING_AGENT_SELECT
            engine.project.pending = PendingConfirmation(
                requirement=requirement,
                initiator_user_id=(selection_owner.initiator_user_id or current_sender),
                engine_session_key=session_key,
                is_template_hint=previous_hint,
            )
            engine._workflow_selection_owner = selection_owner

            if project is not None:
                # Selection state belongs to exactly one engine session.  Keep
                # this write under the engine CAS lock so an older request
                # cannot overwrite a newer request's controller snapshot.
                selection_lock = getattr(
                    project,
                    "_wf_selection_lock",
                    None,
                )
                if selection_lock is None:
                    selection_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
                    project._wf_selection_lock = selection_lock
                with selection_lock:
                    project._wf_selection_controller = ctrl
                    project._wf_selection_snapshot = ctrl.snapshot()
                    project._wf_selection_session_key = session_key

        if previous_owner is not None:
            with previous_owner.delivery_lock:
                pass
            previous_owner.done_event.set()

        project_id = project.project_id if project else ""

        # Build tool list from available workflow tools
        all_tools, recommended_tools, other_tools, _default = self._resolve_tool_lists()
        if not all_tools:
            with selection_owner.delivery_lock:
                with engine._lock:
                    owner_is_current = bool(
                        vars(engine).get("_workflow_selection_owner") is selection_owner
                        and not selection_owner.stop_event.is_set()
                    )
                if owner_is_current:
                    self.send_text_to_chat(
                        chat_id,
                        ("当前环境未检测到可用的 Workflow 编程工具，请安装 Traex/Claude/Codex 等 CLI 后重试。"),
                    )
            return

        # Format tools for controller
        available_tools = []
        for tool_name in recommended_tools + other_tools:
            available_tools.append(
                {
                    "tool_name": tool_name,
                    "display_name": tool_name,
                    "description": all_tools.get(tool_name, ""),
                    "supports_model": True,
                    "provider": "workflow",
                }
            )

        # Get available models for the pending tool
        available_models = None
        if ctrl.pending_tool_name:
            root_path = project.root_path if project else ""
            available_models = self._get_workflow_models_for_tool(ctrl.pending_tool_name, root_path)

        # Build and send card
        card = ctrl.build_orchestrator_combined_card(
            available_tools=available_tools,
            available_models=available_models,
            requirement=requirement,
            session_key=session_key,
            chat_id=chat_id,
            project_id=project_id,
        )
        with selection_owner.delivery_lock:
            with engine._lock:
                pending = engine.project.pending if engine.project else None
                owner_is_current = bool(
                    vars(engine).get("_workflow_selection_owner") is selection_owner
                    and not selection_owner.stop_event.is_set()
                    and engine.project.status == WorkflowStatus.AWAITING_AGENT_SELECT
                    and pending
                    and pending.engine_session_key == session_key
                )
            if owner_is_current:
                self.send_card_to_chat(
                    chat_id,
                    card,
                    origin_message_id=self._resolve_origin(message_id),
                )
        self._preheat_workflow_models(root_path, recommended_tools)

    def _show_tool_selection_card(
        self,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: Optional["ProjectContext"],
        root_path: str,
    ) -> None:
        """Show the initial tool selection card before script generation.

        Splits state initialization from card building so update_card can reuse
        the builder without resetting state (fixes tool selection state reset bug).

        Resolves tool lists once via _resolve_tool_lists() to eliminate duplicate
        get_available_tools() calls and race conditions between init and build.
        """
        all_tools, recommended_tools, other_tools, default_selected = self._resolve_tool_lists()
        if not all_tools:
            self.send_text_to_chat(
                chat_id, "当前环境未检测到可用的 Workflow 编程工具，请安装 Traex/Claude/Codex 等 CLI 后重试。"
            )
            return

        engine, project_id, session_key = self._init_tool_selection_state(
            chat_id=chat_id,
            requirement=requirement,
            project=project,
            root_path=root_path,
            all_tools=all_tools,
            recommended_tools=recommended_tools,
            default_selected=default_selected,
        )
        card = self._build_tool_selection_card(
            engine=engine,
            requirement=requirement,
            chat_id=chat_id,
            project_id=project_id,
            session_key=session_key,
            all_tools=all_tools,
            recommended_tools=recommended_tools,
            other_tools=other_tools,
            default_selected=default_selected,
        )
        self.send_card_to_chat(
            chat_id,
            card,
            origin_message_id=self._resolve_origin(message_id),
        )

    @staticmethod
    def _is_current_generation_session(engine: Any | None, expected_session_key: str | None) -> bool:
        """Return True if an async script-generation task still owns the session."""
        if not expected_session_key:
            return False
        if not engine or not getattr(engine, "project", None):
            return False

        from ...workflow_engine.models import WorkflowStatus

        project = engine.project
        pending = getattr(project, "pending", None)
        stored_session_key = getattr(pending, "engine_session_key", "") if pending else ""
        return (
            project.status == WorkflowStatus.GENERATING_SCRIPT
            and bool(stored_session_key)
            and stored_session_key == expected_session_key
        )

    @classmethod
    def _generation_owner_is_active(
        cls,
        engine: Any,
        owner: _WorkflowLifecycleOwner,
    ) -> bool:
        """Check generation ownership under the engine's lifecycle lock."""
        with engine._lock:
            return bool(
                vars(engine).get("_script_generation_owner") is owner
                and not owner.stop_event.is_set()
                and cls._is_current_generation_session(engine, owner.session_key)
            )

    @classmethod
    def _commit_generation_result_if_current(
        cls,
        engine: Any,
        owner: _WorkflowLifecycleOwner,
        pending: Any,
    ) -> bool:
        """CAS a prepared generation result into the still-owned session."""
        from ...workflow_engine.models import WorkflowStatus

        with engine._lock:
            if (
                vars(engine).get("_script_generation_owner") is not owner
                or owner.stop_event.is_set()
                or not cls._is_current_generation_session(engine, owner.session_key)
            ):
                return False
            engine.project.status = WorkflowStatus.AWAITING_CONFIRM
            engine.project.pending = pending
            return True

    def _generate_and_show_confirm_card(
        self,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: Optional["ProjectContext"],
        root_path: str,
        selected_tools: list[str] | None,
        *,
        expected_session_key: str | None = None,
    ) -> None:
        """Generate script (template or AI) and auto-start execution.

        Called after tool selection is confirmed, or directly for templates.
        """
        from ...card import CardBuilder
        from ...card.ui_text import UI_TEXT
        from ...thread import get_current_sender_id
        from ...workflow_engine.constants import DEFAULT_ORCHESTRATOR_AGENT
        from ...workflow_engine.models import PendingConfirmation, WorkflowStatus
        from ...workflow_engine.script_gen import extract_meta_from_script
        from ...workflow_engine.templates import discover_templates, load_template

        if not expected_session_key:
            logger.error("[workflow] Refusing script generation without a session owner")
            return

        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
        engine = self.ctx.workflow_engine_manager.get_or_create(
            chat_id,
            root_path,
            engine_name=engine_name,
        )
        owner = _WorkflowLifecycleOwner(
            session_key=expected_session_key,
            source_script_path=self._new_workflow_script_path(root_path),
        )
        previous_owner: _WorkflowLifecycleOwner | None = None
        selection_owner: _WorkflowLifecycleOwner | None = None
        generation_context: dict[str, Any] = {}
        with engine._lock:
            if not self._is_current_generation_session(engine, expected_session_key):
                logger.info(
                    "[workflow] Dropping stale script generation before delivery (expected_session=%s)",
                    expected_session_key[:8],
                )
                return
            current_owner = vars(engine).get("_script_generation_owner")
            if isinstance(current_owner, _WorkflowLifecycleOwner) and not current_owner.stop_event.is_set():
                if current_owner.session_key == expected_session_key:
                    logger.info(
                        "[workflow] Dropping duplicate script generation (session=%s)",
                        expected_session_key[:8],
                    )
                    return
                previous_owner = current_owner
            current_selection_owner = vars(engine).get("_workflow_selection_owner")
            if isinstance(
                current_selection_owner,
                _WorkflowLifecycleOwner,
            ):
                if (
                    current_selection_owner.session_key != expected_session_key
                    or current_selection_owner.stop_event.is_set()
                ):
                    logger.info(
                        "[workflow] Dropping generation with stale selection owner (session=%s)",
                        expected_session_key[:8],
                    )
                    return
                selection_owner = current_selection_owner
                self._retire_workflow_owner(
                    engine,
                    current_selection_owner,
                )
                engine._workflow_selection_owner = None
            if previous_owner is not None:
                self._retire_workflow_owner(engine, previous_owner)
            owner.claimed_event.set()
            object.__setattr__(
                owner,
                "worker_thread_id",
                threading.get_ident(),
            )
            engine._script_generation_owner = owner
            existing_pending = engine.project.pending
            generation_context = {
                "initiator_user_id": (
                    existing_pending.initiator_user_id
                    if existing_pending and getattr(existing_pending, "initiator_user_id", None)
                    else get_current_sender_id() or ""
                ),
                "orchestrator_agent": (
                    existing_pending.orchestrator_agent
                    if existing_pending and getattr(existing_pending, "orchestrator_agent", None)
                    else DEFAULT_ORCHESTRATOR_AGENT
                ),
                "is_template_hint": (
                    existing_pending.is_template_hint
                    if existing_pending and getattr(existing_pending, "is_template_hint", None)
                    else None
                ),
                "orchestrator_binding": (
                    existing_pending.orchestrator_binding
                    if existing_pending and getattr(existing_pending, "orchestrator_binding", None)
                    else None
                ),
                "review_agents": (
                    existing_pending.review_agents
                    if existing_pending is not None
                    else None
                ),
                "auto_reviewer": (
                    existing_pending.auto_reviewer
                    if existing_pending is not None
                    else None
                ),
            }

        if previous_owner is not None:
            previous_owner.stop_event.set()
            previous_owner.heartbeat_stop_event.set()
            with previous_owner.delivery_lock:
                pass
        if selection_owner is not None:
            selection_owner.stop_event.set()
            selection_owner.heartbeat_stop_event.set()
            with selection_owner.delivery_lock:
                pass
            selection_owner.done_event.set()

        heartbeat_thread: threading.Thread | None = None
        try:
            try:
                origin_message_id = self._resolve_origin(message_id)
            except Exception:
                logger.debug(
                    "Workflow origin resolution failed; using inbound message",
                    exc_info=True,
                )
                origin_message_id = message_id
        except Exception:
            logger.debug(
                "Workflow origin resolution failed; using inbound message",
                exc_info=True,
            )
            origin_message_id = message_id
        gen_msg_id: str | None = None
        heartbeat_start = time.time()

        def _owner_can_deliver() -> bool:
            return self._generation_owner_is_active(engine, owner)

        def _heartbeat_update(status_hint: str = "") -> None:
            """Update the generating card only while this owner is live."""
            with owner.delivery_lock:
                if not _owner_can_deliver():
                    return
                elapsed = int(time.time() - heartbeat_start)
                status = status_hint or "正在生成编排脚本..."
                progress_content = f"{status}\n\n**需求**: {requirement[:200]}\n\n⏱ 已等待 {elapsed} 秒"
                progress_card = CardBuilder._wrap_card(
                    header_title="🔄 Workflow — 生成脚本中...",
                    header_template=UI_TEXT["workflow_header_colors"].get("generating", "blue"),
                    elements=[{"tag": "markdown", "content": progress_content}],
                )
                if gen_msg_id:
                    self.update_card(gen_msg_id, progress_card)

        def _heartbeat_loop() -> None:
            while not owner.stop_event.is_set() and not owner.heartbeat_stop_event.is_set():
                owner.heartbeat_stop_event.wait(8.0)
                if not owner.stop_event.is_set() and not owner.heartbeat_stop_event.is_set():
                    try:
                        _heartbeat_update()
                    except Exception:
                        pass

        def _stop_heartbeat() -> None:
            owner.heartbeat_stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=2.0)

        script_path: str
        meta: dict[str, Any] | None = None
        is_fallback = False
        try:
            # Guard direct/legacy handler calls only after ownership is proven,
            # so a stale queued task cannot publish even an error card.
            if not requirement or len(requirement.strip()) < 4:
                with owner.delivery_lock:
                    if _owner_can_deliver():
                        with engine._lock:
                            engine.project.status = WorkflowStatus.IDLE
                            engine.project.pending = None
                        self._reply_workflow_error(
                            message_id,
                            "invalid_argument",
                            detail=(
                                "需求描述过短（至少 4 个字符），请重新发送 `/wf <需求描述>` 并提供更清晰的任务目标。"
                            ),
                        )
                return

            gen_card = CardBuilder._wrap_card(
                header_title="🔄 Workflow — 生成脚本中...",
                header_template=UI_TEXT["workflow_header_colors"].get("generating", "blue"),
                elements=[
                    {
                        "tag": "markdown",
                        "content": (f"正在基于选定工具生成编排脚本，请稍候...\n\n**需求**: {requirement[:200]}"),
                    }
                ],
            )
            with owner.delivery_lock:
                if not _owner_can_deliver():
                    return
                gen_msg_id = self.send_card_to_chat(
                    chat_id,
                    gen_card,
                    origin_message_id=origin_message_id,
                )
            if not _owner_can_deliver():
                return

            heartbeat_thread = threading.Thread(
                target=_heartbeat_loop,
                name="wf-gen-heartbeat",
                daemon=True,
            )
            heartbeat_thread.start()

            parts = requirement.strip().split(None, 1)
            template_name = parts[0] if parts else ""
            sender_id = get_current_sender_id() or ""
            templates = discover_templates(root_path, user_id=sender_id)
            template_names = {t.name for t in templates}

            if template_name in template_names:
                # Template path — resolve directly
                content = load_template(root_path, template_name, user_id=sender_id)
                if content:
                    # Validate template content
                    from ...workflow_engine.script_gen import validate_generated_script

                    is_valid, errors = validate_generated_script(content)
                    if not is_valid:
                        logger.warning("Template '%s' failed validation: %s", template_name, errors)
                        error_card = self._build_error_card(
                            "invalid_argument",
                            detail=(
                                f"模板 `{template_name}` 未通过安全校验:\n" + "\n".join(f"• {e}" for e in errors[:5])
                            ),
                        )
                        with owner.delivery_lock:
                            if _owner_can_deliver():
                                with engine._lock:
                                    engine.project.status = WorkflowStatus.IDLE
                                    engine.project.pending = None
                                self._replace_or_send_workflow_card(
                                    card_message_id=gen_msg_id,
                                    chat_id=chat_id,
                                    card=error_card,
                                    origin_message_id=origin_message_id,
                                )
                        return

                    script_path = owner.source_script_path
                    if not script_path:
                        raise RuntimeError("Workflow generation owner has no script path")

                    if len(parts) > 1:
                        from ...workflow_engine.templates import inject_args

                        args = self._parse_template_args(parts[1])
                        content = inject_args(content, args)

                    with owner.delivery_lock:
                        if not _owner_can_deliver():
                            raise _WorkflowGenerationCancelled("Workflow generation cancelled")
                        with open(script_path, "w", encoding="utf-8") as f:
                            f.write(content)
                    meta = extract_meta_from_script(content)
                else:
                    # Template load failed — fallback to AI
                    script_path, meta, is_fallback = self._generate_script_via_ai(
                        requirement,
                        root_path,
                        selected_tools,
                        engine,
                        progress_callback=_heartbeat_update,
                        output_path=owner.source_script_path,
                        cancel_event=owner.stop_event,
                        artifact_lock=owner.delivery_lock,
                    )
            else:
                # AI generation path with selected tools
                script_path, meta, is_fallback = self._generate_script_via_ai(
                    requirement,
                    root_path,
                    selected_tools,
                    engine,
                    progress_callback=_heartbeat_update,
                    output_path=owner.source_script_path,
                    cancel_event=owner.stop_event,
                    artifact_lock=owner.delivery_lock,
                )
            _stop_heartbeat()

            if selected_tools is not None:
                sel_tools = list(selected_tools)
            else:
                sel_tools = list((meta or {}).get("tools", selected_tools or []))
            script_tools = set((meta or {}).get("tools", []))
            allowed_tools = set(sel_tools or [])
            tools_mismatch = bool(script_tools - allowed_tools)
            script_hash = None
            if script_path:
                try:
                    with open(script_path, "rb") as f:
                        import hashlib

                        script_hash = hashlib.sha256(f.read()).hexdigest()
                except OSError:
                    script_hash = None

            prepared_pending = PendingConfirmation(
                script_path=script_path,
                requirement=requirement,
                meta=meta,
                is_fallback=is_fallback,
                initiator_user_id=generation_context["initiator_user_id"],
                engine_session_key=uuid.uuid4().hex,
                selected_tools=sel_tools,
                tools_mismatch=tools_mismatch,
                orchestrator_agent=generation_context["orchestrator_agent"],
                is_template_hint=generation_context["is_template_hint"],
                orchestrator_binding=generation_context["orchestrator_binding"],
                review_agents=generation_context["review_agents"],
                auto_reviewer=generation_context["auto_reviewer"],
                script_hash=script_hash,
            )
            if not self._commit_generation_result_if_current(
                engine,
                owner,
                prepared_pending,
            ):
                logger.info(
                    "[workflow] Dropping stale script generation result (expected_session=%s)",
                    expected_session_key[:8],
                )
                return

            project_id = project.project_id if project else ""
            self._start_pending_workflow_execution(
                message_id=gen_msg_id or message_id,
                chat_id=chat_id,
                project_id=project_id,
                project=project,
                root_path=root_path,
                engine=engine,
                allow_server_side_start=True,
                generation_owner=owner,
            )
        except Exception as exc:
            logger.error(
                "Workflow script generation failed: %s",
                exc,
                exc_info=True,
            )
            with owner.delivery_lock:
                if _owner_can_deliver():
                    with engine._lock:
                        if not self._is_current_generation_session(
                            engine,
                            owner.session_key,
                        ):
                            return
                        engine.project.status = WorkflowStatus.IDLE
                        engine.project.pending = None
                    if gen_msg_id:
                        error_card = self._build_error_card(
                            "internal_error",
                            detail=f"脚本生成失败: {exc}",
                        )
                        self._replace_or_send_workflow_card(
                            card_message_id=gen_msg_id,
                            chat_id=chat_id,
                            card=error_card,
                            origin_message_id=origin_message_id,
                        )
                    else:
                        self._reply_workflow_error(
                            message_id,
                            "internal_error",
                            detail=f"脚本生成失败: {exc}",
                        )
        finally:
            _stop_heartbeat()
            with engine._lock:
                if vars(engine).get("_script_generation_owner") is owner:
                    self._retire_workflow_owner(engine, owner)
                    engine._script_generation_owner = None
                current_project = engine.project
                current_pending = current_project.pending if current_project is not None else None
                retained_paths = {
                    getattr(current_pending, "script_path", None),
                    getattr(current_project, "script_path", None),
                }
            if owner.source_script_path not in retained_paths:
                self._remove_owned_workflow_artifact(
                    owner.source_script_path,
                    root_path=root_path,
                )
            owner.done_event.set()

    def _schedule_generate_and_show_confirm_card(
        self,
        *,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: Optional["ProjectContext"],
        root_path: str,
        selected_tools: list[str] | None,
        expected_session_key: str,
        engine: Any | None = None,
    ) -> None:
        """Submit script generation to the task scheduler.

        Script generation can spend minutes inside an ACP/CLI model call. Keep
        Feishu callback handling short so the websocket receive loop remains
        healthy while the loading card is replaced from the background task.
        """
        from ...workflow_engine.models import WorkflowStatus

        project_name = (getattr(project, "project_name", "") if project else "") or os.path.basename(root_path)
        task_id = generate_task_id(project_name or "workflow")
        if engine is not None:
            with engine._lock:
                current_session_key = (
                    engine.project.pending.engine_session_key if engine.project and engine.project.pending else ""
                )
                if not (
                    engine.project
                    and engine.project.pending
                    and engine.project.status == WorkflowStatus.GENERATING_SCRIPT
                    and current_session_key == expected_session_key
                ):
                    logger.info(
                        "[workflow] Refusing stale generation schedule (expected_session=%s, current_session=%s)",
                        (expected_session_key or "")[:8],
                        (current_session_key or "")[:8],
                    )
                    return
        if not expected_session_key or engine is None:
            logger.error("[workflow] Refusing to schedule generation without a session owner")
            return

        def report_schedule_failure(
            exc: Exception,
            *,
            next_status: WorkflowStatus,
            detail_prefix: str,
        ) -> None:
            with engine._lock:
                lifecycle_owner = vars(engine).get("_script_generation_owner") or vars(engine).get(
                    "_workflow_selection_owner"
                )
            if not isinstance(
                lifecycle_owner,
                _WorkflowLifecycleOwner,
            ):
                return
            with lifecycle_owner.delivery_lock:
                with engine._lock:
                    pending = engine.project.pending if engine.project else None
                    should_report = bool(
                        (
                            vars(engine).get("_script_generation_owner") is lifecycle_owner
                            or vars(engine).get("_workflow_selection_owner") is lifecycle_owner
                        )
                        and not lifecycle_owner.stop_event.is_set()
                        and engine.project
                        and engine.project.status == WorkflowStatus.GENERATING_SCRIPT
                        and pending
                        and pending.engine_session_key == expected_session_key
                    )
                    if should_report:
                        engine.project.status = next_status
                if should_report:
                    self._reply_workflow_error(
                        message_id,
                        "internal_error",
                        detail=f"{detail_prefix}: {exc}",
                    )

        def run_generate() -> None:
            try:
                task_engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)
                if not self._is_current_generation_session(task_engine, expected_session_key):
                    logger.info(
                        "[workflow] Skipping stale script generation task (expected_session=%s)",
                        (expected_session_key or "")[:8],
                    )
                    return
                self._generate_and_show_confirm_card(
                    message_id=message_id,
                    chat_id=chat_id,
                    requirement=requirement,
                    project=project,
                    root_path=root_path,
                    selected_tools=selected_tools,
                    expected_session_key=expected_session_key,
                )
            except Exception as exc:
                logger.error("Workflow script generation task failed: %s", exc, exc_info=True)
                report_schedule_failure(
                    exc,
                    next_status=WorkflowStatus.IDLE,
                    detail_prefix="脚本生成失败",
                )

        try:
            self._submit_engine_task(
                run_generate,
                chat_id,
                message_id,
                project,
                request_id=None,
                task_id=task_id,
                name_suffix="generate_script",
            )
        except Exception as exc:
            logger.error("Workflow script generation task submission failed: %s", exc, exc_info=True)
            report_schedule_failure(
                exc,
                next_status=WorkflowStatus.AWAITING_TOOL_SELECT,
                detail_prefix="脚本生成任务提交失败",
            )

    # ------------------------------------------------------------------
    # Stop workflow
    # ------------------------------------------------------------------

    def _get_selection_controller(
        self,
        project: Any,
        *,
        session_key: str | None = None,
    ) -> "SelectionFlowController":
        """Get or create a SelectionFlowController for the given project context.

        Stores the controller state in ``project._wf_selection_snapshot`` so it
        survives between button clicks. Callers can modify the returned
        controller and persist changes via its snapshot/restore protocol.
        A callback's session key fences that snapshot from newer workflows.
        """
        from ...workflow_engine.selection_flow import SelectionFlowController

        controller = SelectionFlowController()
        selection_lock = getattr(project, "_wf_selection_lock", None)
        if selection_lock is None:
            selection_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
            project._wf_selection_lock = selection_lock
        with selection_lock:
            current_session_key = getattr(
                project,
                "_wf_selection_session_key",
                None,
            )
            if not isinstance(current_session_key, str):
                current_session_key = None
            if current_session_key is None and session_key:
                current_session_key = session_key
                project._wf_selection_session_key = session_key
            is_current_session = bool(not session_key or not current_session_key or session_key == current_session_key)
            snapshot = getattr(project, "_wf_selection_snapshot", None) if is_current_session else None
            if not isinstance(snapshot, dict):
                snapshot = None
            if snapshot is not None:
                try:
                    controller.restore(snapshot)
                except Exception:
                    pass
            controller._workflow_session_key = session_key or current_session_key or ""
            if is_current_session:
                project._wf_selection_controller = controller
        return controller

    def _persist_selection_controller(
        self,
        project: Any,
        controller: "SelectionFlowController",
    ) -> bool:
        """Persist state only while the controller still owns this session."""
        selection_lock = getattr(project, "_wf_selection_lock", None)
        if selection_lock is None:
            selection_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
            project._wf_selection_lock = selection_lock
        with selection_lock:
            controller_session_key = getattr(
                controller,
                "_workflow_session_key",
                "",
            )
            current_session_key = getattr(
                project,
                "_wf_selection_session_key",
                "",
            )
            if not isinstance(current_session_key, str):
                current_session_key = ""
            if controller_session_key and current_session_key and controller_session_key != current_session_key:
                return False
            project._wf_selection_snapshot = controller.snapshot()
        return True

    @staticmethod
    def _retire_workflow_owner(engine: Any, owner: Any) -> None:
        """Keep a revoked owner discoverable by later cleanup retries."""
        retire = getattr(engine, "retire_lifecycle_owner", None)
        if callable(retire):
            retire(owner)

    @contextmanager
    def _selection_controller_transaction(
        self,
        project: Any,
        *,
        session_key: str,
        engine: Any | None = None,
        expected_status: Any = None,
        initiator_user_id: str = "",
    ):
        """Atomically validate, mutate, and persist one selection snapshot.

        When an engine is supplied, keep the established engine -> selection
        lock order and hold both locks through the mutation.  This makes a
        finish/status transition linearizable with concurrent button clicks:
        either the click is included in the finished snapshot, or it observes
        the new status and is discarded.
        """
        selection_lock = getattr(project, "_wf_selection_lock", None)
        if selection_lock is None:
            selection_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
            project._wf_selection_lock = selection_lock
        engine_lock = engine._lock if engine is not None else nullcontext()
        with engine_lock:
            if engine is not None:
                current_project = engine.project
                pending = current_project.pending if current_project is not None else None
                if isinstance(expected_status, (set, frozenset, tuple, list)):
                    valid_status = bool(current_project is not None and current_project.status in expected_status)
                else:
                    valid_status = bool(
                        current_project is not None
                        and (expected_status is None or current_project.status == expected_status)
                    )
                if (
                    not valid_status
                    or pending is None
                    or pending.engine_session_key != session_key
                    or (initiator_user_id and pending.initiator_user_id != initiator_user_id)
                ):
                    yield None
                    return
            with selection_lock:
                controller = self._get_selection_controller(
                    project,
                    session_key=session_key,
                )
                yield controller
                self._persist_selection_controller(project, controller)

    def _build_available_tools(self) -> list[dict]:
        """Build the available tool list for the controller (all tools, with descriptions)."""
        all_tools, recommended_tools, other_tools, _ = self._resolve_tool_lists()
        available = []
        for t in list(recommended_tools) + list(other_tools):
            available.append(
                {
                    "tool_name": t,
                    "display_name": t,
                    "description": all_tools.get(t, "") or "",
                    "supports_model": True,
                    "provider": "workflow",
                }
            )
        return available

    def _reply_pending_error_if_current(
        self,
        engine: Any,
        message_id: str,
        *,
        session_key: str,
        initiator_user_id: str,
        valid_statuses: Any,
        category: str,
        detail: str = "",
    ) -> bool:
        """Deliver a callback error only before its lifecycle is superseded."""
        statuses = valid_statuses if isinstance(valid_statuses, (tuple, list, set, frozenset)) else (valid_statuses,)

        def matches_current() -> bool:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            return bool(
                current_project
                and current_project.status in statuses
                and pending
                and pending.engine_session_key == session_key
                and pending.initiator_user_id == initiator_user_id
            )

        with engine._lock:
            owners = tuple(
                owner
                for owner in (
                    vars(engine).get("_workflow_selection_owner"),
                    vars(engine).get("_script_generation_owner"),
                    vars(engine).get("_workflow_start_owner"),
                )
                if isinstance(owner, _WorkflowLifecycleOwner) and not owner.stop_event.is_set()
            )
            owner = owners[0] if owners else None

        if owner is not None:
            with owner.delivery_lock:
                with engine._lock:
                    owner_is_current = any(
                        vars(engine).get(attr_name) is owner
                        for attr_name in (
                            "_workflow_selection_owner",
                            "_script_generation_owner",
                            "_workflow_start_owner",
                        )
                    )
                    if not owner_is_current or owner.stop_event.is_set() or not matches_current():
                        return False
                self._reply_workflow_error(
                    message_id,
                    category,
                    detail=detail,
                )
                return True

        # Legacy sessions may have no owner token. Holding the engine CAS lock
        # through this rare error delivery still prevents a stop/new request
        # from acknowledging before the reply completes.
        with engine._lock:
            if not matches_current():
                return False
            self._reply_workflow_error(
                message_id,
                category,
                detail=detail,
            )
            return True

    def _safe_execute_engine(
        self,
        executor_func: callable,
        task_id: str,
        chat_id: str,
        message_id: str,
        project: Optional["ProjectContext"],
        engine_name: str,
        reporter: Any,
        request_id: Optional[str],
        action_prefix: str = "deep",
        command_text: str = "",
        *,
        lifecycle_owner: _WorkflowLifecycleOwner | None = None,
        lifecycle_engine: Any = None,
    ):
        """Fence Workflow repo-lock/error cards with the queued-start owner."""
        if lifecycle_owner is None or lifecycle_engine is None:
            return super()._safe_execute_engine(
                executor_func=executor_func,
                task_id=task_id,
                chat_id=chat_id,
                message_id=message_id,
                project=project,
                engine_name=engine_name,
                reporter=reporter,
                request_id=request_id,
                action_prefix=action_prefix,
                command_text=command_text,
            )

        import asyncio

        from ...repo_lock import LockConflictError
        from ...utils.errors import get_error_detail

        root_path = getattr(project, "root_path", None) if project else None

        def owner_is_current() -> bool:
            with lifecycle_engine._lock:
                return bool(
                    vars(lifecycle_engine).get("_workflow_start_owner") is lifecycle_owner
                    and getattr(lifecycle_engine, "_closing", False) is not True
                    and not lifecycle_owner.stop_event.is_set()
                )

        def deliver_if_current(delivery: callable) -> None:
            with lifecycle_owner.delivery_lock:
                if owner_is_current():
                    delivery()

        def body() -> None:
            try:
                executor_func()
            except NotImplementedError:
                logger.error(
                    "%s Engine: NotImplementedError in executor_func (task_id=%s)",
                    self._get_engine_name_prefix(),
                    task_id,
                )
                deliver_if_current(
                    lambda: self.reply_text(
                        message_id,
                        "系统升级中，请重试",
                    )
                )
            except LockConflictError:
                raise
            except Exception as exc:
                if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                    logger.warning(
                        "%s Engine 执行超时 (task_id=%s): %s",
                        self._get_engine_name_prefix(),
                        task_id,
                        get_error_detail(exc),
                    )
                else:
                    logger.error(
                        "%s Engine 执行异常: %s",
                        self._get_engine_name_prefix(),
                        get_error_detail(exc),
                        exc_info=True,
                    )
                deliver_if_current(
                    lambda caught=exc: self._on_engine_error(
                        error=caught,
                        task_id=task_id,
                        chat_id=chat_id,
                        message_id=message_id,
                        project=project,
                        engine_name=engine_name,
                        reporter=reporter,
                        request_id=request_id,
                        action_prefix=action_prefix,
                    )
                )

        # Request-id resolution and scheduler latency happen before this
        # method. Recheck ownership before touching the repository lock so a
        # cleanup/supersede winner retires the stale queued closure quietly.
        if not owner_is_current():
            return None

        try:
            return self.lock_helper._with_repo_lock(
                root_path,
                chat_id,
                body,
            )
        except LockConflictError as exc:
            deliver_if_current(
                lambda caught=exc: self.lock_helper.send_lock_conflict_card(
                    caught,
                    message_id,
                    command_text,
                    chat_id=chat_id,
                )
            )
            return None

    def _send_combined_selection_card(
        self,
        message_id: str,
        chat_id: str,
        project: Any,
        requirement: str,
        session_key: str,
        *,
        is_review: bool,
    ) -> None:
        """Build and send a combined selection card using the controller.

        The controller's current ``step`` is expected to already be set to
        1 (orchestrator) or 2 (review) by the caller.
        """

        selection_lock = getattr(project, "_wf_selection_lock", None)
        if selection_lock is None:
            selection_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
            project._wf_selection_lock = selection_lock
        with selection_lock:
            ctrl = self._get_selection_controller(
                project,
                session_key=session_key,
            )
            rendered_snapshot = ctrl.snapshot()
            if getattr(project, "_wf_selection_snapshot", None) is None:
                project._wf_selection_snapshot = rendered_snapshot
        available_tools = self._build_available_tools()

        # Get available models for the pending tool
        available_models = None
        if ctrl.pending_tool_name:
            root_path = project.root_path if project else ""
            available_models = self._get_workflow_models_for_tool(ctrl.pending_tool_name, root_path)

        project_id = project.project_id if project else ""
        if is_review:
            card = ctrl.build_review_combined_card(
                available_tools=available_tools,
                available_models=available_models,
                requirement=requirement,
                session_key=session_key,
                chat_id=chat_id,
                project_id=project_id,
            )
        else:
            card = ctrl.build_orchestrator_combined_card(
                available_tools=available_tools,
                available_models=available_models,
                requirement=requirement,
                session_key=session_key,
                chat_id=chat_id,
                project_id=project_id,
            )

        from ...workflow_engine.models import WorkflowStatus

        root_path = (
            project.root_path
            if project
            else self._get_root_path(
                chat_id,
                None,
            )
        )
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)
        expected_status = WorkflowStatus.AWAITING_TOOL_SELECT if is_review else WorkflowStatus.AWAITING_AGENT_SELECT
        if not engine:
            return
        with engine._lock:
            # Delivery is the second half of the render transaction. Hold the
            # selection lock through the Feishu update and compare snapshots:
            # a slower S1 render must never overwrite an already-delivered S2.
            with selection_lock:
                pending = engine.project.pending if engine.project else None
                if (
                    not engine.project
                    or engine.project.status != expected_status
                    or not pending
                    or pending.engine_session_key != session_key
                    or getattr(
                        project,
                        "_wf_selection_session_key",
                        "",
                    )
                    != session_key
                    or getattr(
                        project,
                        "_wf_selection_snapshot",
                        None,
                    )
                    != rendered_snapshot
                ):
                    return
                if message_id and message_id != chat_id:
                    self.update_card(message_id, card)
                else:
                    self.send_card_to_chat(chat_id, card)

    def _handle_workflow_model_cascade_select(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
        *,
        is_review: bool,
        dimension: str,
    ) -> None:
        """Handle model-family/profile/effort dropdown changes."""
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        expected_status = WorkflowStatus.AWAITING_TOOL_SELECT if is_review else WorkflowStatus.AWAITING_AGENT_SELECT
        if engine.project.status != expected_status:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        button_session_key = value.get("engine_session_key", "")
        if not stored_session_key or button_session_key != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or not current_user or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        tool_name = value.get("tool_name", "")
        selected = value.get("_option") or value.get(dimension, "")
        if not tool_name or not selected:
            self._reply_workflow_error(message_id, "invalid_argument", detail="缺少模型选择参数")
            return

        _kept, rejected = self._validate_tools_against_registry([tool_name])
        if rejected:
            self._reply_pending_error_if_current(
                engine,
                message_id,
                session_key=stored_session_key,
                initiator_user_id=stored_initiator,
                valid_statuses=expected_status,
                category="invalid_argument",
                detail=f"无效的工具名称: {tool_name}",
            )
            return

        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=expected_status,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(2 if is_review else 1)
            if dimension == "model_group":
                ctrl.set_model_group(
                    tool_name,
                    selected,
                    is_review=is_review,
                )
            elif dimension == "model_profile":
                if value.get("model_group"):
                    ctrl.pending_model_group = value.get("model_group")
                ctrl.set_model_profile(
                    tool_name,
                    selected,
                    is_review=is_review,
                )
            elif dimension == "model_effort":
                if value.get("model_group"):
                    ctrl.pending_model_group = value.get("model_group")
                if value.get("model_profile"):
                    ctrl.pending_model_profile = value.get("model_profile")
                ctrl.set_model_effort(
                    tool_name,
                    selected,
                    is_review=is_review,
                )
            else:
                self._reply_workflow_error(
                    message_id,
                    "invalid_argument",
                    detail="未知模型选择维度",
                )
                return

        requirement = engine.project.pending.requirement if engine.project.pending else ""
        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=is_review,
        )

    def handle_workflow_orchestrator_select_model_group(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        self._handle_workflow_model_cascade_select(
            message_id,
            chat_id,
            project_id,
            value,
            is_review=False,
            dimension="model_group",
        )

    def handle_workflow_orchestrator_select_model_profile(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        self._handle_workflow_model_cascade_select(
            message_id,
            chat_id,
            project_id,
            value,
            is_review=False,
            dimension="model_profile",
        )

    def handle_workflow_orchestrator_select_model_effort(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        self._handle_workflow_model_cascade_select(
            message_id,
            chat_id,
            project_id,
            value,
            is_review=False,
            dimension="model_effort",
        )

    def handle_workflow_review_select_model_group(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        self._handle_workflow_model_cascade_select(
            message_id,
            chat_id,
            project_id,
            value,
            is_review=True,
            dimension="model_group",
        )

    def handle_workflow_review_select_model_profile(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        self._handle_workflow_model_cascade_select(
            message_id,
            chat_id,
            project_id,
            value,
            is_review=True,
            dimension="model_profile",
        )

    def handle_workflow_review_select_model_effort(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        self._handle_workflow_model_cascade_select(
            message_id,
            chat_id,
            project_id,
            value,
            is_review=True,
            dimension="model_effort",
        )

    def stop_workflow(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Stop the running workflow for the current project.

        Security: only the workflow initiator or a configured admin can stop it.
        """
        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        from ...workflow_engine.models import WorkflowStatus

        if not engine:
            self._reply_workflow_error(message_id, "invalid_state", detail="当前没有运行中的 Workflow 任务")
            return

        from ...thread import get_current_sender_id

        current_user = get_current_sender_id()
        admin_ids: list[str] = getattr(self.ctx.settings, "admin_user_ids", []) or []
        error: tuple[str, str] | None = None
        delivery_owners: list[_WorkflowLifecycleOwner] = []
        artifacts_to_remove: list[str] = []
        runtime_active = False
        with engine._lock:
            wf_project = engine.project
            pending = getattr(wf_project, "pending", None) if wf_project else None
            status = getattr(wf_project, "status", WorkflowStatus.IDLE)
            pending_lifecycle = status in _workflow_pending_statuses()
            stored_initiator = (
                getattr(pending, "initiator_user_id", None) if pending_lifecycle and pending else None
            ) or getattr(wf_project, "initiator_user_id", None)
            selection_owner = vars(engine).get("_workflow_selection_owner")
            generation_owner = vars(engine).get("_script_generation_owner")
            start_owner = vars(engine).get("_workflow_start_owner")
            live_selection_owner = (
                selection_owner
                if isinstance(selection_owner, _WorkflowLifecycleOwner) and not selection_owner.stop_event.is_set()
                else None
            )
            live_generation_owner = (
                generation_owner
                if isinstance(generation_owner, _WorkflowLifecycleOwner) and not generation_owner.stop_event.is_set()
                else None
            )
            live_start_owner = (
                start_owner
                if isinstance(start_owner, _WorkflowLifecycleOwner) and not start_owner.stop_event.is_set()
                else None
            )
            stored_initiator = (
                stored_initiator
                or getattr(
                    live_selection_owner,
                    "initiator_user_id",
                    None,
                )
                or getattr(
                    live_generation_owner,
                    "initiator_user_id",
                    None,
                )
                or getattr(
                    live_start_owner,
                    "initiator_user_id",
                    None,
                )
            )
            terminal_committed = status in {
                WorkflowStatus.COMPLETED,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED,
            }
            if terminal_committed:
                live_start_owner = None
            runtime_active = bool(engine.is_running) and not terminal_committed
            lifecycle_active = bool(
                runtime_active
                or live_selection_owner is not None
                or live_generation_owner is not None
                or live_start_owner is not None
                or status in _workflow_pending_statuses()
            )

            if not lifecycle_active:
                error = ("invalid_state", "当前没有运行中的 Workflow 任务")
            elif not stored_initiator or not current_user:
                error = ("forbidden", "无法验证操作者身份，停止请求被拒绝")
            elif current_user != stored_initiator and current_user not in admin_ids:
                error = ("forbidden", "只有 Workflow 发起者或管理员才能停止此任务")
            else:
                for lifecycle_owner in (
                    live_selection_owner,
                    live_generation_owner,
                    live_start_owner,
                ):
                    if lifecycle_owner is None:
                        continue
                    lifecycle_owner.stop_event.set()
                    lifecycle_owner.heartbeat_stop_event.set()
                    self._retire_workflow_owner(
                        engine,
                        lifecycle_owner,
                    )
                    if lifecycle_owner not in delivery_owners:
                        delivery_owners.append(lifecycle_owner)
                    for artifact in (
                        lifecycle_owner.source_script_path,
                        lifecycle_owner.execution_script_path,
                    ):
                        if artifact:
                            artifacts_to_remove.append(artifact)
                if live_selection_owner is not None:
                    engine._workflow_selection_owner = None
                if live_generation_owner is not None:
                    engine._script_generation_owner = None
                if live_start_owner is not None and not runtime_active:
                    engine._workflow_start_owner = None
                if pending and pending.script_path:
                    artifacts_to_remove.append(pending.script_path)
                if live_start_owner is not None and wf_project is not None and wf_project.script_path:
                    artifacts_to_remove.append(wf_project.script_path)

                if runtime_active:
                    engine.stop()
                elif wf_project is not None:
                    wf_project.status = WorkflowStatus.IDLE
                    wf_project.pending = None
                    wf_project.script_path = None

        if error is not None:
            self._reply_workflow_error(
                message_id,
                error[0],
                detail=error[1],
            )
            return

        # A delivery that had already passed its ownership check may finish,
        # but it must do so before the stop acknowledgement is sent.
        for lifecycle_owner in delivery_owners:
            with lifecycle_owner.delivery_lock:
                pass
            if (
                not runtime_active
                and not lifecycle_owner.claimed_event.is_set()
                and not lifecycle_owner.worker_started_event.is_set()
            ):
                lifecycle_owner.done_event.set()

        for artifact in dict.fromkeys(artifacts_to_remove):
            self._remove_owned_workflow_artifact(
                artifact,
                root_path=root_path,
            )

        self.reply_text(message_id, "Workflow 任务已停止。")

    def handle_workflow_stop_running(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle the "停止" button on a RUNNING workflow progress card.

        Delegates to :meth:`stop_workflow`, which owns the initiator/admin
        authorization checks and state validation. This keeps the auth logic
        in a single place and maximizes reuse.
        """
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)
        project_id = project_id or value.get("project_id", "")
        project = self._resolve_project_from_id(project_id, chat_id)
        # Resolve root_path defensively so a missing project still lets the
        # underlying stop_workflow re-derive engine state from the chat.
        self._get_root_path(chat_id, project)
        self.stop_workflow(message_id, chat_id, project)

    # ------------------------------------------------------------------
    # Confirm / Cancel actions (card button callbacks)
    # ------------------------------------------------------------------

    def handle_workflow_orchestrator_select_tool(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle orchestrator tool click — expand model panel or add default.

        Uses SelectionFlowController: toggles pending_tool_name for the
        clicked tool, persists state, and refreshes the card so the user
        sees the inline model panel (or the newly selected agent).
        """
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        if engine.project.status != WorkflowStatus.AWAITING_AGENT_SELECT:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        button_session_key = value.get("engine_session_key", "")
        if not stored_session_key or button_session_key != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or not current_user or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        tool_name = value.get("_option") or value.get("tool_name", "")
        if not tool_name:
            self._reply_workflow_error(message_id, "invalid_argument", detail="缺少 tool_name")
            return

        # Security: validate tool_name against allowed list
        from ...workflow_engine.tool_registry import get_available_tools

        available_tools = get_available_tools(require_available=True)
        if tool_name not in available_tools:
            self._reply_pending_error_if_current(
                engine,
                message_id,
                session_key=stored_session_key,
                initiator_user_id=stored_initiator,
                valid_statuses=WorkflowStatus.AWAITING_AGENT_SELECT,
                category="invalid_argument",
                detail=f"不支持的工具: {tool_name}",
            )
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        # Use SelectionFlowController
        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=WorkflowStatus.AWAITING_AGENT_SELECT,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(1)
            if "model_page" in value:
                try:
                    model_page = int(value.get("model_page", 0) or 0)
                except (TypeError, ValueError):
                    model_page = 0
                ctrl.set_model_page(
                    tool_name,
                    model_page,
                    is_review=False,
                )
            elif value.get("_option"):
                ctrl.select_tool(tool_name, is_review=False)
            else:
                ctrl.toggle_tool_expand(tool_name, is_review=False)

        requirement = engine.project.pending.requirement if engine.project.pending else ""
        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=False,
        )

    def handle_workflow_orchestrator_select_model(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle orchestrator model click — add selection and refresh card."""
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        if engine.project.status != WorkflowStatus.AWAITING_AGENT_SELECT:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        button_session_key = value.get("engine_session_key", "")
        if not stored_session_key or button_session_key != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or not current_user or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        tool_name = value.get("tool_name", "")
        display_name = value.get("display_name", "") or tool_name
        use_default = bool(value.get("use_default_model"))
        model_name = value.get("model_name")
        if not use_default and not model_name:
            model_name = value.get("name")

        # Validate tool_name against registry
        _kept, _rejected = self._validate_tools_against_registry([tool_name])
        if _rejected:
            logger.warning(
                "[workflow] Rejected unknown tool_name=%s at handle_workflow_orchestrator_select_model",
                tool_name,
            )
            self._reply_pending_error_if_current(
                engine,
                message_id,
                session_key=stored_session_key,
                initiator_user_id=stored_initiator,
                valid_statuses=WorkflowStatus.AWAITING_AGENT_SELECT,
                category="invalid_argument",
                detail=f"无效的工具名称: {tool_name}",
            )
            return

        # Validate model_name against allowed models for this tool
        if not use_default and model_name:
            available_models = self._get_workflow_models_for_tool(tool_name, root_path)
            if not workflow_model_name_is_available(model_name, available_models):
                logger.warning(
                    "[workflow] Rejected unknown model_name=%s for tool=%s at handle_workflow_orchestrator_select_model",
                    model_name,
                    tool_name,
                )
                self._reply_pending_error_if_current(
                    engine,
                    message_id,
                    session_key=stored_session_key,
                    initiator_user_id=stored_initiator,
                    valid_statuses=WorkflowStatus.AWAITING_AGENT_SELECT,
                    category="invalid_argument",
                    detail=f"无效的模型名称: {model_name}",
                )
                return

        selection = {
            "tool_name": tool_name,
            "display_name": display_name,
            "provider": value.get("provider", "workflow"),
            "supports_model": True,
        }
        if use_default or not model_name:
            selection["use_default_model"] = True
            selection["model_name"] = ""
        else:
            selection["model_name"] = model_name

        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=WorkflowStatus.AWAITING_AGENT_SELECT,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(1)
            # Orchestrator is single-select: clear before adding.
            ctrl.clear_selections(is_review=False)
            ctrl.add_or_update_selection(selection, is_review=False)

        requirement = engine.project.pending.requirement if engine.project.pending else ""
        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=False,
        )

    def handle_workflow_orchestrator_remove(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Remove a selected orchestrator item and refresh the card."""
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        if engine.project.status != WorkflowStatus.AWAITING_AGENT_SELECT:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        if value.get("engine_session_key") != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        selection_key = value.get("selection_key", "")
        if not selection_key:
            self._reply_workflow_error(message_id, "invalid_argument", detail="缺少 selection_key")
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=WorkflowStatus.AWAITING_AGENT_SELECT,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(1)
            ctrl.remove_selection(selection_key, is_review=False)

        requirement = engine.project.pending.requirement if engine.project.pending else ""
        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=False,
        )

    def handle_workflow_orchestrator_clear(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Clear all selected orchestrator items and refresh the card."""
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        if engine.project.status != WorkflowStatus.AWAITING_AGENT_SELECT:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        if value.get("engine_session_key") != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=WorkflowStatus.AWAITING_AGENT_SELECT,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(1)
            ctrl.clear_selections(is_review=False)

        requirement = engine.project.pending.requirement if engine.project.pending else ""
        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=False,
        )

    def handle_workflow_orchestrator_finish(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Finalize orchestrator selection, transition to review-agent step.

        Validates that at least one orchestrator agent has been selected
        via the controller. On success, saves the selection to the engine
        project's pending state and shows the review-agent combined card.
        """
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        if engine.project.status != WorkflowStatus.AWAITING_AGENT_SELECT:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        button_session_key = value.get("engine_session_key", "")
        if not stored_session_key or button_session_key != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or not current_user or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        from src.spec_engine.review_agents import ReviewAgentBinding

        requirement = ""
        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=WorkflowStatus.AWAITING_AGENT_SELECT,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(1)
            ok, err_msg = ctrl.validate_non_empty(is_review=False)
            current_pending = engine.project.pending if engine.project else None
            if (
                engine.project is None
                or engine.project.status != WorkflowStatus.AWAITING_AGENT_SELECT
                or current_pending is None
                or current_pending.engine_session_key != stored_session_key
                or current_pending.initiator_user_id != current_user
            ):
                return
            if not ok:
                ctrl.error_message = err_msg
            else:
                snapshot = ctrl.snapshot()
                orchestrator_items = list(
                    snapshot.get(
                        "orchestrator_selections",
                        {},
                    ).values()
                )
                first = orchestrator_items[0]
                current_pending = engine.project.pending
                current_pending.orchestrator_binding = ReviewAgentBinding.from_dict(first)
                current_pending.orchestrator_agent = first.get(
                    "tool_name",
                    "",
                )
                engine.project.status = WorkflowStatus.AWAITING_TOOL_SELECT
                ctrl.set_step(2)
            requirement = engine.project.pending.requirement or ""

        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=ok,
        )

    def handle_workflow_review_select_tool(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle review tool click — expand inline model panel."""
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        button_session_key = value.get("engine_session_key", "")
        current_user = get_current_sender_id() or ""
        validation_error: str | None = None
        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            stored_session_key = pending.engine_session_key if pending else ""
            stored_initiator = pending.initiator_user_id if pending else ""
            requirement = pending.requirement if pending else ""
            if current_project is None or current_project.status != WorkflowStatus.AWAITING_TOOL_SELECT:
                validation_error = "invalid_state"
            elif not stored_session_key or button_session_key != stored_session_key:
                validation_error = "session_expired"
            elif not stored_initiator or not current_user or current_user != stored_initiator:
                validation_error = "forbidden"

        if validation_error is not None:
            self._reply_workflow_error(message_id, validation_error)
            return

        tool_name = value.get("_option") or value.get("tool_name", "")
        if not tool_name:
            self._reply_workflow_error(message_id, "invalid_argument", detail="缺少 tool_name")
            return

        # Security: validate tool_name against allowed list
        from ...workflow_engine.tool_registry import get_available_tools

        available_tools = get_available_tools(require_available=True)
        if tool_name not in available_tools:
            self._reply_pending_error_if_current(
                engine,
                message_id,
                session_key=stored_session_key,
                initiator_user_id=stored_initiator,
                valid_statuses=WorkflowStatus.AWAITING_TOOL_SELECT,
                category="invalid_argument",
                detail=f"不支持的工具: {tool_name}",
            )
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=WorkflowStatus.AWAITING_TOOL_SELECT,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(2)
            if "model_page" in value:
                try:
                    model_page = int(value.get("model_page", 0) or 0)
                except (TypeError, ValueError):
                    model_page = 0
                ctrl.set_model_page(
                    tool_name,
                    model_page,
                    is_review=True,
                )
            elif value.get("_option"):
                ctrl.select_tool(tool_name, is_review=True)
            else:
                ctrl.toggle_tool_expand(tool_name, is_review=True)

        requirement = engine.project.pending.requirement if engine.project.pending else ""
        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=True,
        )

    def handle_workflow_review_select_model(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle review model click — add selection (multi) and refresh card."""
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        button_session_key = value.get("engine_session_key", "")
        current_user = get_current_sender_id() or ""
        validation_error: str | None = None
        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            stored_session_key = pending.engine_session_key if pending else ""
            stored_initiator = pending.initiator_user_id if pending else ""
            requirement = pending.requirement if pending else ""
            if current_project is None or current_project.status != WorkflowStatus.AWAITING_TOOL_SELECT:
                validation_error = "invalid_state"
            elif not stored_session_key or button_session_key != stored_session_key:
                validation_error = "session_expired"
            elif not stored_initiator or not current_user or current_user != stored_initiator:
                validation_error = "forbidden"

        if validation_error is not None:
            self._reply_workflow_error(message_id, validation_error)
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        tool_name = value.get("tool_name", "")
        display_name = value.get("display_name", "") or tool_name
        use_default = bool(value.get("use_default_model"))
        model_name = value.get("model_name")
        if not use_default and not model_name:
            model_name = value.get("name")

        # Validate tool_name against registry
        _kept, _rejected = self._validate_tools_against_registry([tool_name])
        if _rejected:
            logger.warning(
                "[workflow] Rejected unknown tool_name=%s at handle_workflow_review_select_model",
                tool_name,
            )
            self._reply_pending_error_if_current(
                engine,
                message_id,
                session_key=stored_session_key,
                initiator_user_id=stored_initiator,
                valid_statuses=WorkflowStatus.AWAITING_TOOL_SELECT,
                category="invalid_argument",
                detail=f"无效的工具名称: {tool_name}",
            )
            return

        # Validate model_name against allowed models for this tool
        if not use_default and model_name:
            available_models = self._get_workflow_models_for_tool(tool_name, root_path)
            if not workflow_model_name_is_available(model_name, available_models):
                logger.warning(
                    "[workflow] Rejected unknown model_name=%s for tool=%s at handle_workflow_review_select_model",
                    model_name,
                    tool_name,
                )
                self._reply_pending_error_if_current(
                    engine,
                    message_id,
                    session_key=stored_session_key,
                    initiator_user_id=stored_initiator,
                    valid_statuses=WorkflowStatus.AWAITING_TOOL_SELECT,
                    category="invalid_argument",
                    detail=f"无效的模型名称: {model_name}",
                )
                return

        selection = {
            "tool_name": tool_name,
            "display_name": display_name,
            "provider": value.get("provider", "workflow"),
            "supports_model": True,
        }
        if use_default or not model_name:
            selection["use_default_model"] = True
            selection["model_name"] = ""
        else:
            selection["model_name"] = model_name

        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=WorkflowStatus.AWAITING_TOOL_SELECT,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(2)
            ctrl.add_or_update_selection(
                selection,
                is_review=True,
                keep_panel_open=True,
            )

        requirement = engine.project.pending.requirement if engine.project.pending else ""
        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=True,
        )

    def handle_workflow_review_remove(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Remove a selected review item and refresh the card."""
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        if engine.project.status != WorkflowStatus.AWAITING_TOOL_SELECT:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        if value.get("engine_session_key") != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        selection_key = value.get("selection_key", "")
        if not selection_key:
            self._reply_workflow_error(message_id, "invalid_argument", detail="缺少 selection_key")
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=WorkflowStatus.AWAITING_TOOL_SELECT,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(2)
            ctrl.remove_selection(selection_key, is_review=True)

        requirement = engine.project.pending.requirement if engine.project.pending else ""
        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=True,
        )

    def handle_workflow_review_clear(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Clear all selected review items and refresh the card."""
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        if engine.project.status != WorkflowStatus.AWAITING_TOOL_SELECT:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        if value.get("engine_session_key") != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=WorkflowStatus.AWAITING_TOOL_SELECT,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(2)
            ctrl.clear_selections(is_review=True)

        requirement = engine.project.pending.requirement if engine.project.pending else ""
        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=True,
        )

    def handle_workflow_review_toggle_auto(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Toggle auto mode on review step (skip explicit review selection)."""
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        if engine.project.status != WorkflowStatus.AWAITING_TOOL_SELECT:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        if value.get("engine_session_key") != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=WorkflowStatus.AWAITING_TOOL_SELECT,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(2)
            ctrl.set_review_auto_mode(not ctrl.review_auto_mode)

        requirement = engine.project.pending.requirement if engine.project.pending else ""
        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=True,
        )

    def handle_workflow_review_finish(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Finalize review selection and proceed to script generation.

        Accepts the situation where the user either picked at least one
        review agent OR toggled auto mode. Both paths are considered valid
        completion of the review step. Persists the review selections into
        ``engine.project.pending.review_agents`` before triggering
        ``_generate_and_show_confirm_card``.
        """
        from ...card.events.payloads import filter_workflow_button_value
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        value = filter_workflow_button_value(value)
        project_id = value.get("project_id", "") or project_id or ""
        project = self._resolve_project_from_id(project_id, chat_id) if project_id else None
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return
        if project is None:
            self._reply_workflow_error(
                message_id,
                "invalid_state",
                detail="缺少项目上下文",
            )
            return

        button_session_key = value.get("engine_session_key", "")
        current_user = get_current_sender_id() or ""
        validation_error: str | None = None
        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            stored_session_key = pending.engine_session_key if pending else ""
            stored_initiator = pending.initiator_user_id if pending else ""
            requirement = pending.requirement if pending else ""
            if current_project is None or current_project.status != WorkflowStatus.AWAITING_TOOL_SELECT:
                validation_error = "invalid_state"
            elif not stored_session_key or button_session_key != stored_session_key:
                validation_error = "session_expired"
            elif not stored_initiator or not current_user or current_user != stored_initiator:
                validation_error = "forbidden"
        if validation_error is not None:
            self._reply_workflow_error(message_id, validation_error)
            return

        from src.spec_engine.review_agents import ReviewAgentBinding

        from ...card import CardBuilder

        all_selected: list[str] = []
        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=WorkflowStatus.AWAITING_TOOL_SELECT,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            ctrl.set_step(2)
            ok, err_msg = ctrl.validate_non_empty(is_review=True)
            current_pending = engine.project.pending if engine.project else None
            if (
                engine.project is None
                or engine.project.status != WorkflowStatus.AWAITING_TOOL_SELECT
                or current_pending is None
                or current_pending.engine_session_key != stored_session_key
                or current_pending.initiator_user_id != current_user
            ):
                return
            if not ok:
                ctrl.error_message = err_msg
                requirement = engine.project.pending.requirement or ""
            else:
                snapshot = ctrl.snapshot()
                review_items = list(snapshot.get("review_selections", {}).values())
                review_agents = [ReviewAgentBinding.from_dict(item) for item in review_items]
                current_project = engine.project
                current_pending = current_project.pending

                current_pending.review_agents = review_agents
                current_pending.auto_reviewer = bool(ctrl.review_auto_mode)
                orchestrator_tool = current_pending.orchestrator_agent or ""
                for tool_name in [
                    orchestrator_tool,
                    *(agent.tool_name for agent in review_agents),
                ]:
                    if tool_name and tool_name not in all_selected:
                        all_selected.append(tool_name)
                if all_selected:
                    current_pending.selected_tools = all_selected

                selections_summary = []
                orch_binding = current_pending.orchestrator_binding
                orch_model = getattr(orch_binding, "model_name", "") or "(默认)" if orch_binding else ""
                selections_summary.append(f"**主编排 Agent**: `{orchestrator_tool}` {orch_model}")
                if review_agents:
                    review_lines = [
                        (f"  {index + 1}. `{agent.tool_name}` {agent.model_name or '(默认)'}")
                        for index, agent in enumerate(review_agents)
                    ]
                    selections_summary.append("**评审 Agent**:\n" + "\n".join(review_lines))
                elif ctrl.review_auto_mode:
                    selections_summary.append(
                        "**评审 Agent**: Auto（无独立 Reviewer；编排器负责最终检查）"
                    )
                locked_card = CardBuilder._wrap_card(
                    header_title="✅ Workflow — 工具选择完成",
                    header_template="green",
                    elements=[
                        {
                            "tag": "markdown",
                            "content": "\n\n".join(selections_summary),
                        }
                    ],
                )
                requirement = current_pending.requirement or ""
                # Keep card order consistent with the state CAS: a newer
                # request waits for this update and therefore publishes last.
                self.update_card(message_id, locked_card)
                current_project.status = WorkflowStatus.GENERATING_SCRIPT

        if not ok:
            self._send_combined_selection_card(
                message_id=message_id,
                chat_id=chat_id,
                project=project,
                requirement=requirement,
                session_key=stored_session_key,
                is_review=True,
            )
            return

        self._schedule_generate_and_show_confirm_card(
            message_id=message_id,
            chat_id=chat_id,
            requirement=requirement,
            project=project,
            root_path=root_path,
            selected_tools=all_selected if all_selected else None,
            engine=engine,
            expected_session_key=stored_session_key,
        )

    def _get_workflow_models_for_tool(self, tool_name: str, root_path: str = "") -> list[dict]:
        """Get available models for a workflow tool.

        Resolves the tool's real provider (acp/cli) instead of passing a
        bogus ``provider="workflow"``. The previous hardcoded value always fell
        into the wrong subprocess branch of ``get_models_for_tool`` and probed
        ACP tools (traex/coco/...) through the wrong transport, returning ``[]``
        leaving the WF model panel empty.
        """
        tool_name = str(tool_name or "").strip()
        root_path = str(root_path or "")
        if not tool_name:
            return []

        def _copy_models(items: list[dict]) -> list[dict]:
            copied: list[dict] = []
            for item in items:
                clone = dict(item)
                if "reasoning_efforts" in clone:
                    clone["reasoning_efforts"] = list(clone.get("reasoning_efforts") or [])
                if "selection_variants" in clone:
                    clone["selection_variants"] = [
                        dict(variant or {}) for variant in (clone.get("selection_variants") or [])
                    ]
                copied.append(clone)
            return copied

        now = time.monotonic()
        provider = "acp" if tool_name in _WORKFLOW_ACP_MODEL_TOOLS else "ttadk"
        generation = 0
        if provider == "acp":
            from ...acp.helper import get_acp_model_cache_generation

            generation = get_acp_model_cache_generation(tool_name, root_path)
        cache: dict[
            tuple[str, str, str, int],
            tuple[float, list[dict]],
        ] = getattr(
            self,
            "_workflow_model_cache",
            {},
        )
        if not isinstance(cache, dict):
            cache = {}
        else:
            for cache_key, (cached_at, cached_models) in list(cache.items()):
                if len(cache_key) == 4:
                    (
                        cached_tool,
                        cached_root,
                        _cached_provider,
                        cached_generation,
                    ) = cache_key
                else:
                    cached_tool, cached_root, _cached_provider = cache_key
                    cached_generation = -1
                if cached_tool == tool_name and cached_root == root_path:
                    if cached_generation == generation and now - cached_at <= _WORKFLOW_MODEL_CACHE_TTL_S:
                        logger.info(
                            "[workflow] model_lookup tool=%s provider=%s count=%d duration_ms=%.1f cached=true",
                            tool_name,
                            _cached_provider,
                            len(cached_models),
                            (time.monotonic() - now) * 1000,
                        )
                        return _copy_models(cached_models)
                    cache.pop(cache_key, None)
        self._workflow_model_cache = cache

        try:
            from ...worktree_engine.tool_discovery import WorktreeToolDiscovery

            started = time.monotonic()
            discovery = WorktreeToolDiscovery()
            models = discovery.get_models_for_tool(
                tool_name,
                provider=provider,
                cwd=root_path,
                force_refresh=False,
            )
            normalized = []
            for model in models or []:
                if not model.get("name"):
                    continue
                item = {
                    "name": model.get("name", ""),
                    "display_name": model.get(
                        "display_name",
                        model.get("name", ""),
                    ),
                    "description": model.get("description", ""),
                }
                reasoning_efforts = list(model.get("reasoning_efforts") or [])
                if reasoning_efforts:
                    item["is_default"] = bool(model.get("is_default"))
                    item["reasoning_efforts"] = reasoning_efforts
                    item["adapted_reasoning_effort"] = model.get("adapted_reasoning_effort")
                selection_variants = [dict(variant or {}) for variant in (model.get("selection_variants") or [])]
                if selection_variants:
                    item["is_default"] = bool(model.get("is_default"))
                    item["selection_variants"] = selection_variants
                normalized.append(item)
            cache[(tool_name, root_path, provider, generation)] = (
                now,
                _copy_models(normalized),
            )
            logger.info(
                "[workflow] model_lookup tool=%s provider=%s count=%d duration_ms=%.1f cached=false",
                tool_name,
                provider,
                len(normalized),
                (time.monotonic() - started) * 1000,
            )
            return _copy_models(normalized)
        except Exception:
            logger.debug("[workflow] model lookup failed for tool=%s", tool_name, exc_info=True)
            return []

    def _preheat_workflow_models(self, root_path: str, tool_names: list[str]) -> None:
        """Warm likely model lists after the first /wf selection card is visible."""
        tools: list[str] = []
        seen: set[str] = set()
        for name in tool_names:
            tool = str(name or "").strip()
            if not tool or tool in seen:
                continue
            seen.add(tool)
            tools.append(tool)
            if len(tools) >= _WORKFLOW_MODEL_PREHEAT_LIMIT:
                break
        if not tools:
            return

        started: set[tuple[str, str]] = getattr(self, "_workflow_model_preheat_started", set())
        if not isinstance(started, set):
            started = set()
        pending: list[str] = []
        for tool in tools:
            key = (tool, str(root_path or ""))
            if key in started:
                continue
            started.add(key)
            pending.append(tool)
        self._workflow_model_preheat_started = started
        if not pending:
            return

        def _run() -> None:
            for tool in pending:
                try:
                    self._get_workflow_models_for_tool(tool, root_path)
                except Exception:
                    logger.debug("[workflow] model preheat failed for tool=%s", tool, exc_info=True)

        import threading as _threading

        thread = _threading.Thread(
            target=_run,
            name="workflow-model-preheat",
            daemon=True,
        )
        thread.start()

    def handle_workflow_confirm_tools(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle confirm tools button — generate script based on selected tools.

        Security: validates engine_session_key and initiator_user_id.
        Transitions from AWAITING_TOOL_SELECT to script generation.
        """
        # Strip unknown fields from button callback to prevent forged fields
        # (e.g. a client-side ``value.confirmed`` bypass).
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)

        project_id = project_id or value.get("project_id", "")
        project = self._resolve_project_from_id(project_id, chat_id)
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        button_session_key = value.get("engine_session_key", "")
        current_user = get_current_sender_id() or ""
        stored_session_key = ""
        selected_tools: list[str] = []
        requirement = ""
        error: tuple[str, str] | None = None
        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            stored_session_key = pending.engine_session_key if pending else ""
            stored_initiator = pending.initiator_user_id if pending else ""
            if current_project is None or current_project.status != WorkflowStatus.AWAITING_TOOL_SELECT:
                error = ("invalid_state", "")
            elif not stored_session_key or button_session_key != stored_session_key:
                error = ("session_expired", "")
            elif not stored_initiator or not current_user or current_user != stored_initiator:
                error = ("forbidden", "")
            elif not pending.selected_tools:
                error = ("invalid_argument", "请至少选择一个工具")
            else:
                # Snapshot and status transition share the same CAS as a
                # concurrent legacy selection click.
                selected_tools = list(pending.selected_tools)
                requirement = pending.requirement or ""
                current_project.status = WorkflowStatus.GENERATING_SCRIPT

        if error is not None:
            self._reply_workflow_error(
                message_id,
                error[0],
                detail=error[1],
            )
            return

        # Enter the same owned/scheduled generation path as the combined
        # selector. A required session key makes /stop_wf and stale-task drops
        # work for this compatibility action too.
        self._schedule_generate_and_show_confirm_card(
            message_id=message_id,
            chat_id=chat_id,
            requirement=requirement,
            project=project,
            root_path=root_path,
            selected_tools=selected_tools,
            engine=engine,
            expected_session_key=stored_session_key,
        )

    def handle_workflow_regenerate_script(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle regenerate script button — regenerate with current tool selection.

        Security: validates engine_session_key and initiator_user_id.
        Only available in AWAITING_CONFIRM state after tool changes.
        """
        # Strip unknown fields from button callback to prevent forged fields
        # (e.g. a client-side ``value.confirmed`` bypass).
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)

        project_id = project_id or value.get("project_id", "")
        project = self._resolve_project_from_id(project_id, chat_id)
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        # --- Security validation (fail-closed) ---
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        button_session_key = value.get("engine_session_key", "")
        current_user = get_current_sender_id() or ""
        error: tuple[str, str] | None = None
        selected_tools: list[str] = []
        requirement = ""
        old_script_path: str | None = None
        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            stored_session_key = pending.engine_session_key if pending else ""
            stored_initiator = pending.initiator_user_id if pending else ""
            if current_project is None or current_project.status != WorkflowStatus.AWAITING_CONFIRM:
                error = ("invalid_state", "")
            elif not stored_session_key or button_session_key != stored_session_key:
                error = ("session_expired", "")
            elif not stored_initiator or not current_user or current_user != stored_initiator:
                error = ("forbidden", "")
            elif pending is None or not pending.selected_tools:
                error = ("invalid_argument", "请至少选择一个工具")
            else:
                selected_tools = list(pending.selected_tools)
                requirement = pending.requirement or ""
                old_script_path = pending.script_path
                pending.script_path = None
                pending.script_hash = None
                current_project.status = WorkflowStatus.GENERATING_SCRIPT

        if error is not None:
            self._reply_workflow_error(
                message_id,
                error[0],
                detail=error[1],
            )
            return

        self._remove_owned_workflow_artifact(
            old_script_path,
            root_path=root_path,
        )

        # Regenerate through the cancellable background path.
        self._schedule_generate_and_show_confirm_card(
            message_id=message_id,
            chat_id=chat_id,
            requirement=requirement,
            project=project,
            root_path=root_path,
            selected_tools=selected_tools,
            engine=engine,
            expected_session_key=stored_session_key,
        )

    def _start_pending_workflow_execution(
        self,
        *,
        message_id: str,
        chat_id: str,
        project_id: str,
        project: Optional["ProjectContext"],
        root_path: str,
        engine: Any,
        allow_server_side_start: bool = False,
        generation_owner: _WorkflowLifecycleOwner | None = None,
    ) -> bool:
        """Start a generated pending workflow through the existing confirm path."""
        from ...thread import get_current_sender_id, set_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        _ = (project, root_path)
        if generation_owner is not None and generation_owner.stop_event.is_set():
            return False
        if not engine:
            if generation_owner is None:
                self._reply_workflow_error(message_id, "session_expired")
            return False
        with engine._lock:
            wf_project = engine.project
            pending = wf_project.pending if wf_project else None
            valid_owner = bool(
                generation_owner is None
                or (
                    vars(engine).get("_script_generation_owner") is generation_owner
                    and not generation_owner.stop_event.is_set()
                )
            )
            valid_status = bool(wf_project and wf_project.status == WorkflowStatus.AWAITING_CONFIRM)
        if pending is None or not valid_owner:
            if generation_owner is None:
                self._reply_workflow_error(message_id, "session_expired")
            return False
        if not valid_status:
            if generation_owner is None:
                self._reply_workflow_error(message_id, "invalid_state")
            return False
        session_key = pending.engine_session_key or ""
        if not session_key:
            if generation_owner is None:
                self._reply_workflow_error(message_id, "session_expired")
            return False

        previous_sender = get_current_sender_id()
        should_bind_sender = allow_server_side_start and bool(pending.initiator_user_id)
        if should_bind_sender:
            set_current_sender_id(pending.initiator_user_id)

        try:
            self.handle_workflow_confirm_start(
                message_id,
                chat_id,
                project_id,
                {
                    "action": "workflow_confirm_start",
                    "project_id": project_id,
                    "engine_session_key": session_key,
                },
                _generation_owner=generation_owner,
            )
        finally:
            if should_bind_sender:
                set_current_sender_id(previous_sender)

        return bool(engine.project and engine.project.status == WorkflowStatus.RUNNING)

    def handle_workflow_confirm_start(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
        *,
        _generation_owner: _WorkflowLifecycleOwner | None = None,
    ) -> None:
        """Handle confirm button click — start workflow execution.

        Security: validates initiator_user_id and engine_session_key before
        allowing execution to prevent cross-user confirmation hijacking.
        """
        # Strip unknown fields from button callback to prevent forged fields
        # (e.g. a client-side ``value.confirmed`` bypass).
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)

        # Route to correct project using project_id from button value
        project_id = project_id or value.get("project_id", "")
        project = self._resolve_project_from_id(project_id, chat_id)
        root_path = project.root_path if project else self._get_root_path(chat_id, None)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        from ...workflow_engine.models import WorkflowStatus

        def _reply_start_error(
            category: str,
            *,
            detail: str = "",
        ) -> None:
            if _generation_owner is None:
                self._reply_workflow_error(
                    message_id,
                    category,
                    detail=detail,
                )
                return
            with _generation_owner.delivery_lock:
                with engine._lock:
                    owner_is_current = bool(
                        vars(engine).get("_script_generation_owner") is _generation_owner
                        and not _generation_owner.stop_event.is_set()
                    )
                if owner_is_current:
                    self._reply_workflow_error(
                        message_id,
                        category,
                        detail=detail,
                    )

        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            if current_project is None or current_project.status != WorkflowStatus.AWAITING_CONFIRM:
                pending = None
            if _generation_owner is not None and (
                vars(engine).get("_script_generation_owner") is not _generation_owner
                or _generation_owner.stop_event.is_set()
            ):
                return

        if pending is None:
            _reply_start_error("invalid_state")
            return

        # --- Security validation ---
        from ...thread import get_current_sender_id

        current_user = get_current_sender_id() or ""
        stored_initiator = pending.initiator_user_id or ""
        stored_session_key = pending.engine_session_key or ""
        button_session_key = value.get("engine_session_key", "")

        # Fail-closed: reject if session key missing or mismatched
        if not stored_session_key or button_session_key != stored_session_key:
            logger.warning(
                "Workflow confirm rejected: session_key mismatch (button=%s, stored=%s)",
                button_session_key[:8],
                stored_session_key[:8],
            )
            _reply_start_error("session_expired")
            return

        # Fail-closed: reject if initiator unknown or user mismatch
        if not stored_initiator or not current_user or current_user != stored_initiator:
            logger.warning(
                "Workflow confirm rejected: user mismatch (operator=%s, initiator=%s)",
                current_user[:8],
                stored_initiator[:8],
            )
            _reply_start_error("forbidden")
            return

        # Retrieve pending state
        script_path = pending.script_path
        requirement = pending.requirement or ""
        selected_tools = list(pending.selected_tools or [])
        expected_script_hash = pending.script_hash
        pending_meta = pending.meta

        if not script_path:
            _reply_start_error(
                "invalid_state",
                detail="无法获取待执行脚本，请重新发送 `/wf`",
            )
            return

        # --- TOCTOU hardening (check-then-re-read-verify) ---
        # Re-read script content fresh from disk at confirm-time and
        # re-run the full validation chain. This defends against scripts
        # being tampered with after the confirmation card was shown but
        # before the user hit "confirm".
        try:
            with open(script_path, "rb") as f:
                script_bytes = f.read()
        except OSError:
            # Note: script_path is not exposed to user for security reasons
            _reply_start_error(
                "internal_error",
                detail="脚本文件读取失败，请重新发送 `/wf` 生成",
            )
            return
        script_text = script_bytes.decode("utf-8", errors="strict")
        import hashlib

        current_script_hash = hashlib.sha256(script_bytes).hexdigest()
        if expected_script_hash and current_script_hash != expected_script_hash:
            _reply_start_error(
                "internal_error",
                detail="脚本内容与生成时不一致，疑似被篡改。请重新发送 `/wf` 生成。",
            )
            return

        # Structural + security validation (mirrors the generation path).
        from ...workflow_engine.script_gen import extract_meta_from_script, validate_generated_script

        is_valid, validation_errors = validate_generated_script(script_text)
        if not is_valid:
            _reply_start_error(
                "internal_error",
                detail="脚本验证失败：" + "; ".join(validation_errors[:3]),
            )
            return
        fresh_meta = extract_meta_from_script(script_text) or {}

        # Tool consistency check — use the freshly-parsed meta, not the
        # one cached in pending. If the AI-generated script references tools
        # outside the allowed set, auto-fix by restricting meta.tools to the
        # allowed subset and rewriting tool references in the script.
        fresh_script_tools = set(fresh_meta.get("tools", []))
        allowed_tools = set(selected_tools or [])
        if fresh_script_tools and allowed_tools:
            unmatched = fresh_script_tools - allowed_tools
            if unmatched:
                logger.warning(
                    "[workflow] Auto-fixing script tools: removing %s, keeping %s",
                    sorted(unmatched),
                    sorted(fresh_script_tools & allowed_tools),
                )
                # Rewrite tool references in script to use only allowed tools
                primary_tool = (selected_tools or ["coco"])[0]
                for bad_tool in unmatched:
                    script_text = script_text.replace(f'tool: "{bad_tool}"', f'tool: "{primary_tool}"')
                    script_text = script_text.replace(f"tool: '{bad_tool}'", f"tool: '{primary_tool}'")
                # Update meta.tools in the script
                import re as _re

                kept_tools = sorted(fresh_script_tools & allowed_tools) or list(selected_tools or [])
                tools_json = str(kept_tools).replace("'", '"')
                script_text = _re.sub(
                    r"tools:\s*\[[^\]]*\]",
                    f"tools: {tools_json}",
                    script_text,
                    count=1,
                )

        # --- Workflow-refs deferred injection ---
        # Sub-workflow references live in pending.meta.workflow_refs.
        # Injecting them here (at confirm time) guarantees we only run
        # references the user actually approved on the confirmation card.
        refs_for_injection = []
        if pending_meta:
            candidate_refs = pending_meta.get("workflow_refs", []) or []
            if isinstance(candidate_refs, list):
                refs_for_injection = [r for r in candidate_refs if isinstance(r, dict)]
        if refs_for_injection:
            script_text = self._inject_workflow_refs_into_script(script_text, refs_for_injection)
            # Revalidate after injection so a broken ref payload cannot
            # silently bypass the dangerous-pattern / structural checks.
            is_valid, validation_errors = validate_generated_script(script_text)
            if not is_valid:
                joined = "; ".join(validation_errors)[:500]
                _reply_start_error(
                    "invalid_argument",
                    detail=f"workflow refs 注入后脚本未通过校验: {joined}",
                )
                return

        # --- Immutable copy for execution ---
        # Copy the verified content into a fresh /tmp file for each
        # confirmation. Using mkstemp guarantees uniqueness across
        # concurrent sessions and test suites; we additionally append the
        # script hash so accidental inspection can trace the source.
        import tempfile

        temp_fd, immutable_script_path = tempfile.mkstemp(
            prefix="ghostap-confirmed-",
            suffix=f"-{current_script_hash}.js",
        )

        def _discard_immutable_script() -> None:
            self._remove_owned_workflow_artifact(
                immutable_script_path,
                root_path=root_path,
            )

        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                f.write(script_text)
        except OSError as exc:
            try:
                os.close(temp_fd)
            except OSError:
                pass
            _discard_immutable_script()
            _reply_start_error(
                "internal_error",
                detail=f"无法创建执行用临时脚本副本: {exc}",
            )
            return

        if _generation_owner is not None:
            start_owner = _WorkflowLifecycleOwner(
                session_key=stored_session_key,
                stop_event=_generation_owner.stop_event,
                heartbeat_stop_event=_generation_owner.heartbeat_stop_event,
                delivery_lock=_generation_owner.delivery_lock,
                source_script_path=script_path,
                execution_script_path=immutable_script_path,
            )
        else:
            start_owner = _WorkflowLifecycleOwner(
                session_key=stored_session_key,
                source_script_path=script_path,
                execution_script_path=immutable_script_path,
            )

        origin_message_id = self._resolve_origin(message_id)
        run_spec = None
        binding_error: TypeError | ValueError | None = None

        def _abandon_start(*, remove_source: bool = True) -> None:
            start_owner.stop_event.set()
            _discard_immutable_script()
            if remove_source:
                self._remove_owned_workflow_artifact(
                    start_owner.source_script_path,
                    root_path=root_path,
                )
            with engine._lock:
                if vars(engine).get("_workflow_start_owner") is start_owner:
                    self._retire_workflow_owner(
                        engine,
                        start_owner,
                    )
                    engine._workflow_start_owner = None
                    if engine.project is not None:
                        engine.project.status = WorkflowStatus.IDLE
                        engine.project.script_path = None
            start_owner.done_event.set()

        # Linearize the generated-session handoff against /stop_wf. All
        # expensive validation above is speculative; this is the authoritative
        # state/session check immediately before publishing RUNNING.
        with start_owner.delivery_lock:
            with engine._lock:
                current_project = engine.project
                current_pending = current_project.pending if current_project is not None else None
                current_session_key = current_pending.engine_session_key if current_pending else ""
                generation_owner_matches = bool(
                    _generation_owner is None or vars(engine).get("_script_generation_owner") is _generation_owner
                )
                if (
                    current_project is None
                    or current_project.status != WorkflowStatus.AWAITING_CONFIRM
                    or current_session_key != stored_session_key
                    or not generation_owner_matches
                    or start_owner.stop_event.is_set()
                ):
                    _discard_immutable_script()
                    return

                try:
                    run_spec = self._build_confirmed_run_spec(
                        pending=current_pending,
                        engine=engine,
                        task=current_pending.requirement or requirement,
                        chat_id=chat_id,
                        topic_id=origin_message_id,
                    )
                except (TypeError, ValueError) as exc:
                    binding_error = exc
                else:
                    requirement = run_spec.task
                    current_project.start_execution()
                    current_project.status = WorkflowStatus.RUNNING
                    current_project.requirement = requirement
                    current_project.script_path = script_path
                    current_project.started_at = time.time()
                    current_project.selected_tools = list(run_spec.allowed_tools)
                    current_project.tool_model_map = dict(run_spec.tool_model_map)
                    current_project.run_spec = run_spec.to_dict()
                    engine._workflow_start_owner = start_owner
                    if _generation_owner is not None:
                        self._retire_workflow_owner(
                            engine,
                            _generation_owner,
                        )
                        engine._script_generation_owner = None

            if binding_error is not None:
                _discard_immutable_script()
                _reply_start_error(
                    "invalid_state",
                    detail=f"Workflow 执行绑定无效: {binding_error}",
                )
                return
            if run_spec is None:
                _discard_immutable_script()
                return

            if start_owner.stop_event.is_set():
                _abandon_start()
                return

            progress_card_message_id = self._show_initial_workflow_progress_card(
                card_message_id=message_id,
                chat_id=chat_id,
                wf_project=engine.project,
                origin_message_id=origin_message_id,
            )
            if start_owner.stop_event.is_set():
                _abandon_start()
                return

            try:
                # Use project already resolved above for engine_name.
                engine_name = self.get_engine_name(
                    chat_id,
                    project_id=project_id or None,
                )
                project_name = (project.project_name if project else "") or os.path.basename(root_path)
                task_id = generate_task_id(project_name or "workflow")
            except Exception as exc:
                error_card = self._build_error_card(
                    "internal_error",
                    detail=f"Workflow 任务准备失败: {exc}",
                )
                try:
                    self._replace_or_send_workflow_card(
                        card_message_id=progress_card_message_id,
                        chat_id=chat_id,
                        card=error_card,
                        origin_message_id=origin_message_id,
                    )
                finally:
                    _abandon_start()
                return

            def run_workflow():
                # Register the scheduler worker in the same engine critical
                # section that cleanup uses to revoke the owner. Cleanup can
                # now distinguish "never started" from "started, pre-claim"
                # without forging completion for the latter.
                with engine._lock:
                    worker_admitted = bool(
                        vars(engine).get("_workflow_start_owner") is start_owner
                        and getattr(engine, "_closing", False) is not True
                        and not start_owner.stop_event.is_set()
                    )
                    if worker_admitted:
                        start_owner.worker_started_event.set()
                        object.__setattr__(
                            start_owner,
                            "worker_thread_id",
                            threading.get_ident(),
                        )
                if not worker_admitted:
                    _abandon_start()
                    return

                def _executor():
                    callbacks = self._build_workflow_callbacks(
                        progress_card_message_id,
                        chat_id,
                        project,
                        lifecycle_owner=start_owner,
                    )
                    engine.execute_workflow(
                        script_path=immutable_script_path,
                        callbacks=callbacks,
                        run_spec=run_spec,
                        start_owner=start_owner,
                        source_script_path=script_path,
                    )

                try:
                    self._safe_execute_engine(
                        executor_func=_executor,
                        task_id=task_id,
                        chat_id=chat_id,
                        message_id=message_id,
                        project=project,
                        engine_name=engine_name,
                        reporter=self.ctx.progress_reporter,
                        request_id=self.ensure_request_id(
                            message_id,
                            chat_id=chat_id,
                            project_id=project_id or None,
                        ),
                        action_prefix="workflow",
                        command_text=f"/wf {requirement}",
                        lifecycle_owner=start_owner,
                        lifecycle_engine=engine,
                    )
                finally:
                    if not start_owner.claimed_event.is_set():
                        _abandon_start()

            try:
                self._submit_engine_task(
                    run_workflow,
                    chat_id,
                    message_id,
                    project,
                    request_id=None,
                    task_id=task_id,
                )
            except Exception as exc:
                with start_owner.delivery_lock:
                    with engine._lock:
                        should_report = bool(
                            vars(engine).get("_workflow_start_owner") is start_owner
                            and not start_owner.stop_event.is_set()
                        )
                    try:
                        if should_report:
                            error_card = self._build_error_card(
                                "internal_error",
                                detail=f"Workflow 任务提交失败: {exc}",
                            )
                            self._replace_or_send_workflow_card(
                                card_message_id=progress_card_message_id,
                                chat_id=chat_id,
                                card=error_card,
                                origin_message_id=origin_message_id,
                            )
                    finally:
                        _abandon_start()

    def handle_workflow_cancel(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle cancel button click — abort the pending workflow or
        dismiss an entry prompt.

        ``entry-cancel`` sessions represent a requirement-entry prompt:
        no engine/pending state is required to cancel them, and the
        handler simply replaces the card with a "cancelled" notice.
        For all other sessions we validate engine_session_key and
        initiator_user_id before allowing cancellation.
        """
        # Strip unknown fields from button callback to prevent forged fields
        # (e.g. a client-side ``value.confirmed`` bypass).
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)

        button_session_key = value.get("engine_session_key", "")

        # Entry-prompt cancel: validates the initiator before allowing
        # cancellation so a forged cancel button from a different user
        # cannot drop the topic workflow mode. The entry card writes the
        # original sender into ``initiator_user_id``; we match that value
        # against the current callback's sender.
        if button_session_key == "entry-cancel":
            from ...card import CardBuilder
            from ...mode import set_topic_mode
            from ...thread import get_current_sender_id

            stored_initiator = value.get("initiator_user_id", "")
            current_user = get_current_sender_id() or ""
            if not stored_initiator or current_user != stored_initiator:
                self._reply_workflow_error(message_id, "forbidden")
                return

            # Unbind the topic from ``workflow`` mode so free text after
            # cancel is not routed to the workflow orchestrator.
            try:
                set_topic_mode(chat_id, None)
            except Exception:
                logger.debug("set_topic_mode unavailable during entry-cancel; ignoring")

            cancel_card = CardBuilder._wrap_card(
                header_title="🔄 Workflow — 已取消",
                header_template="grey",
                elements=[
                    {
                        "tag": "markdown",
                        "content": (
                            "已取消输入。当前对话已退出 Workflow 模式。\n"
                            "若需开始新任务，请显式发送：\n"
                            "• `/wf <你的任务描述>` — 基于需求开始编排\n"
                            "• `/wf` — 重新查看入口卡片\n"
                            "（自由文本将走原有路由，不再被当作 Workflow 需求）"
                        ),
                    }
                ],
            )
            self.update_card(message_id, cancel_card)
            return

        project_id = project_id or value.get("project_id", "")
        project = self._resolve_project_from_id(project_id, chat_id)
        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        from ...thread import get_current_sender_id
        from ...workflow_engine.models import WorkflowStatus

        current_user = get_current_sender_id() or ""
        error_category: str | None = None
        owners_to_drain: list[_WorkflowLifecycleOwner] = []
        script_path: str | None = None
        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            stored_session_key = pending.engine_session_key if pending else ""
            stored_initiator = pending.initiator_user_id if pending else ""
            if current_project is None or current_project.status not in _workflow_pending_statuses():
                error_category = "invalid_state"
            elif not stored_session_key or button_session_key != stored_session_key:
                error_category = "session_expired"
            elif not stored_initiator or not current_user or current_user != stored_initiator:
                error_category = "forbidden"
            else:
                script_path = pending.script_path if pending else None
                for attr_name in (
                    "_workflow_selection_owner",
                    "_script_generation_owner",
                    "_workflow_start_owner",
                ):
                    candidate = vars(engine).get(attr_name)
                    if isinstance(candidate, _WorkflowLifecycleOwner):
                        candidate.stop_event.set()
                        candidate.heartbeat_stop_event.set()
                        self._retire_workflow_owner(
                            engine,
                            candidate,
                        )
                        if candidate not in owners_to_drain:
                            owners_to_drain.append(candidate)
                        setattr(engine, attr_name, None)
                current_project.status = WorkflowStatus.IDLE
                current_project.pending = None

        if error_category is not None:
            self._reply_workflow_error(message_id, error_category)
            return
        for owner_to_drain in owners_to_drain:
            with owner_to_drain.delivery_lock:
                pass
            if not owner_to_drain.claimed_event.is_set() and not owner_to_drain.worker_started_event.is_set():
                owner_to_drain.done_event.set()
        self._remove_owned_workflow_artifact(
            script_path,
            root_path=root_path,
        )

        # Update card to show cancelled
        from ...card import CardBuilder

        cancel_card = CardBuilder._wrap_card(
            header_title="🔄 Workflow — 已取消",
            header_template="grey",
            elements=[
                {
                    "tag": "markdown",
                    "content": "Workflow 已取消。如需重新开始，请发送 `/wf <需求>`。",
                }
            ],
        )
        self.update_card(message_id, cancel_card)

    @staticmethod
    def _validate_tools_against_registry(
        candidate_tools: list[str],
    ) -> tuple[list[str], list[str]]:
        """Return ``(kept, rejected)`` lists from ``candidate_tools``.

        Tool names must appear in the authoritative ``tool_registry
        .get_available_tools()`` to be kept. Unknown names are dropped and
        returned in ``rejected``.  Callers should surface a visible warning
        when ``rejected`` is non-empty — unknown tools must never reach the
        executor.
        """
        try:
            from ...workflow_engine.tool_registry import get_available_tools

            available = set(get_available_tools(require_available=True).keys())
        except Exception:
            logger.exception("tool_registry.get_available_tools() failed; falling back to empty set")
            available = set()

        kept: list[str] = []
        rejected: list[str] = []
        seen: set[str] = set()
        for name in candidate_tools:
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            if name in available:
                kept.append(name)
            else:
                rejected.append(name)
        return kept, rejected

    def handle_workflow_select_tool(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Toggle a tool in the pending workflow's selected tools list.

        Handles two states:
        - AWAITING_TOOL_SELECT: initial tool selection before script generation
        - AWAITING_CONFIRM: tool adjustment after script generation (triggers regenerate prompt)

        Security: validates engine_session_key and initiator_user_id.
        Updates the card to reflect the new selection state.
        """
        # Strip unknown fields from button callback to prevent forged fields
        # (e.g. a client-side ``value.confirmed`` bypass).
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)

        project_id = project_id or value.get("project_id", "")
        project = self._resolve_project_from_id(project_id, chat_id)
        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        from ...workflow_engine.models import WorkflowStatus

        valid_statuses = (
            WorkflowStatus.AWAITING_CONFIRM,
            WorkflowStatus.AWAITING_TOOL_SELECT,
            WorkflowStatus.AWAITING_AGENT_SELECT,
        )
        if engine.project.status not in valid_statuses:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        # --- Security validation (fail-closed) ---
        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        button_session_key = value.get("engine_session_key", "")
        if not stored_session_key or button_session_key != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        from ...thread import get_current_sender_id

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or not current_user or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        tool_name = value.get("tool_name", "")
        if not tool_name:
            self._reply_workflow_error(message_id, "invalid_argument", detail="缺少 tool_name 参数")
            return

        # --- Registry validation: unknown tool names are rejected at the boundary.
        # This prevents malicious callers from widening the tool set via forged
        # button payloads.
        _kept, _rejected = self._validate_tools_against_registry([tool_name])
        if _rejected:
            logger.warning(
                "[workflow] Rejected unknown tool_name=%s at handle_workflow_select_tool",
                tool_name,
            )
            self._reply_pending_error_if_current(
                engine,
                message_id,
                session_key=stored_session_key,
                initiator_user_id=stored_initiator,
                valid_statuses=valid_statuses,
                category="invalid_argument",
                detail=f"工具「{tool_name}」不在当前可用工具列表中，已被过滤；可用工具：{', '.join(sorted(_kept)) or '(空)'}",
            )
            return

        if project is None:
            self._reply_workflow_error(
                message_id,
                "invalid_state",
                detail="缺少项目上下文",
            )
            return

        # Pending state and the controller snapshot form one logical selection
        # record. Mutate both under engine -> selection locks so a concurrent
        # finish cannot commit a snapshot from between the two writes.
        with self._selection_controller_transaction(
            project,
            session_key=stored_session_key,
            engine=engine,
            expected_status=valid_statuses,
            initiator_user_id=current_user,
        ) as ctrl:
            if ctrl is None:
                return
            current_project = engine.project
            current_pending = current_project.pending
            selected = list(current_pending.selected_tools or [])
            if not selected and current_project.status == WorkflowStatus.AWAITING_CONFIRM:
                selected = list((current_pending.meta or {}).get("tools", []))
            if tool_name in selected:
                selected.remove(tool_name)
            else:
                selected.append(tool_name)
            if not selected:
                return
            current_pending.selected_tools = selected
            current_status = current_project.status
            requirement = current_pending.requirement or ""
            if current_status in (
                WorkflowStatus.AWAITING_TOOL_SELECT,
                WorkflowStatus.AWAITING_AGENT_SELECT,
            ):
                ctrl.add_or_update_selection(
                    {
                        "tool_name": tool_name,
                        "provider": value.get("provider", "workflow"),
                        "display_name": (value.get("display_name") or tool_name),
                        "supports_model": bool(value.get("supports_model", True)),
                        "use_default_model": True,
                    },
                    is_review=(current_status == WorkflowStatus.AWAITING_TOOL_SELECT),
                )
            else:
                script_tools = set((current_pending.meta or {}).get("tools", []))
                current_pending.tools_mismatch = bool(script_tools - set(selected))

        # Re-render based on the session state captured by the CAS.
        if current_status in (
            WorkflowStatus.AWAITING_TOOL_SELECT,
            WorkflowStatus.AWAITING_AGENT_SELECT,
        ):
            is_review = current_status == WorkflowStatus.AWAITING_TOOL_SELECT
            self._send_combined_selection_card(
                message_id=message_id,
                chat_id=chat_id,
                project=project,
                requirement=requirement,
                session_key=stored_session_key,
                is_review=is_review,
            )
        else:
            _script_content = self._read_pending_script(engine)
            with engine._lock:
                pending = engine.project.pending if engine.project else None
                if (
                    not engine.project
                    or engine.project.status != WorkflowStatus.AWAITING_CONFIRM
                    or not pending
                    or pending.engine_session_key != stored_session_key
                ):
                    return
                confirm_card = self._build_confirm_card(
                    meta=pending.meta,
                    requirement=pending.requirement or "",
                    engine_session_key=pending.engine_session_key or "",
                    chat_id=chat_id,
                    project_id=project_id,
                    is_fallback=pending.is_fallback,
                    selected_tools=pending.selected_tools,
                    script_content=_script_content,
                    orchestrator_binding=pending.orchestrator_binding,
                    review_agents=pending.review_agents,
                    auto_reviewer=pending.auto_reviewer,
                )
                self.update_card(message_id, confirm_card)

    def handle_workflow_fill_missing_tools(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """One-click fill-in of missing tools the script needs but the user
        hasn't yet selected. Updates pending.selected_tools to the union of
        existing user selection and the script-declared tool list, then
        re-renders the confirm card.
        """
        # Strip unknown fields from button callback to prevent forged fields
        # (e.g. a client-side ``value.confirmed`` bypass).
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)

        project_id = project_id or value.get("project_id", "")
        project = self._resolve_project_from_id(project_id, chat_id)
        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        from ...workflow_engine.models import WorkflowStatus

        if engine.project.status != WorkflowStatus.AWAITING_CONFIRM:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        # --- Security validation ---
        from ...thread import get_current_sender_id

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        button_session_key = value.get("engine_session_key", "")
        if not stored_session_key or button_session_key != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or not current_user or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        script_tools = list((engine.project.pending.meta or {}).get("tools", []))
        # --- Registry filter: unknown names in meta.tools must not enter
        # selected_tools. ``kept`` contains only names present in the
        # authoritative registry; ``rejected`` is surfaced to the user.
        _kept, _rejected = WorkflowHandler._validate_tools_against_registry(script_tools)
        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            if (
                current_project is None
                or current_project.status != WorkflowStatus.AWAITING_CONFIRM
                or pending is None
                or pending.engine_session_key != stored_session_key
                or pending.initiator_user_id != current_user
            ):
                return
            current = list(pending.selected_tools or [])
            # Merge existing selected tools (already validated at selection
            # time) with registry-validated script-declared tools.
            merged = sorted(set(current) | set(_kept))
            pending.selected_tools = merged

        _script_content = self._read_pending_script(engine)

        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            if (
                current_project is None
                or current_project.status != WorkflowStatus.AWAITING_CONFIRM
                or pending is None
                or pending.engine_session_key != stored_session_key
            ):
                return
            if _rejected:
                logger.warning(
                    "[workflow] fill_missing_tools dropped unknown tool names: %s",
                    _rejected,
                )
                self._reply_workflow_error(
                    message_id,
                    "invalid_argument",
                    detail=(
                        "⚠️ 补齐工具时发现脚本声明的工具中存在未知名称，"
                        "已被过滤。\n"
                        f"已过滤: {', '.join(_rejected)}\n"
                        f"已保留: {', '.join(merged) or '(空)'}"
                    ),
                )

            confirm_card = self._build_confirm_card(
                meta=pending.meta,
                requirement=pending.requirement or "",
                engine_session_key=stored_session_key,
                chat_id=chat_id,
                project_id=project_id,
                is_fallback=pending.is_fallback,
                selected_tools=merged,
                script_content=_script_content,
                orchestrator_binding=pending.orchestrator_binding,
                review_agents=pending.review_agents,
                auto_reviewer=pending.auto_reviewer,
            )
            self.update_card(message_id, confirm_card)

    def handle_workflow_back_to_tools(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Returns the user to the tool-selection card.

        This is the explicit "back" path: when the user sees a tool mismatch
        and wants to go back to toggle tools manually (rather than fill-in
        auto-magically). Keeps the existing pending state so we don't re-run
        the AI, re-generate a session key, or overwrite the user's already
        selected tools / script meta.

        The card is re-built via `_build_tool_selection_card` (pure rendering)
        and sent via `update_card`. We deliberately do NOT call
        `_show_tool_selection_card` because that helper runs
        `_init_tool_selection_state` which would reset engine_session_key and
        discard the current pending content.
        """
        # Strip unknown fields from button callback to prevent forged fields
        # (e.g. a client-side ``value.confirmed`` bypass).
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)

        project = self._resolve_project_from_id(project_id, chat_id)
        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        from ...workflow_engine.models import WorkflowStatus

        if engine.project.status != WorkflowStatus.AWAITING_CONFIRM:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        # --- Security validation (session key + sender) ---
        from ...thread import get_current_sender_id

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        button_session_key = value.get("engine_session_key", "")
        if not stored_session_key or button_session_key != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or not current_user or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        all_tools, recommended_tools, other_tools, default_selected = self._resolve_tool_lists()

        resolved_project_id = project.project_id if project is not None else (project_id or "")
        # The slow tool discovery above is speculative.  Revalidate and
        # transition under one lock so an old callback cannot move the newest
        # session backwards.
        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            if (
                current_project is None
                or current_project.status != WorkflowStatus.AWAITING_CONFIRM
                or pending is None
                or pending.engine_session_key != stored_session_key
                or pending.initiator_user_id != current_user
            ):
                return
            current_project.status = WorkflowStatus.AWAITING_TOOL_SELECT
            card = self._build_tool_selection_card(
                engine=engine,
                requirement=pending.requirement or "",
                chat_id=chat_id,
                project_id=resolved_project_id,
                session_key=stored_session_key,
                all_tools=all_tools,
                recommended_tools=recommended_tools,
                other_tools=other_tools,
                default_selected=default_selected,
            )
            self.update_card(message_id, card)

    # ------------------------------------------------------------------
    # Workflow ref (sub-workflow) handlers
    # ------------------------------------------------------------------

    def handle_workflow_view_workflow_ref(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Preview a sub-workflow reference as an inline card reply.

        Reads the referenced script (from ``meta.workflow_refs``) and shows
        its content and metadata. Does not mutate the pending workflow
        state — this is purely informational.
        """
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)

        project = self._resolve_project_from_id(project_id, chat_id)
        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        from ...workflow_engine.models import WorkflowStatus
        from ...workflow_engine.script_gen import extract_meta_from_script
        from ...workflow_engine.templates import load_template

        if engine.project.status not in (
            WorkflowStatus.AWAITING_CONFIRM,
            WorkflowStatus.AWAITING_TOOL_SELECT,
            WorkflowStatus.AWAITING_AGENT_SELECT,
        ):
            self._reply_workflow_error(message_id, "invalid_state")
            return

        # Session key + sender validation
        from ...thread import get_current_sender_id

        stored_session_key = engine.project.pending.engine_session_key if engine.project.pending else ""
        button_session_key = value.get("engine_session_key", "")
        if not stored_session_key or button_session_key != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
        if not stored_initiator or not current_user or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        raw_idx = value.get("ref_index")
        try:
            ref_index = int(raw_idx) if raw_idx is not None else -1
        except (TypeError, ValueError):
            ref_index = -1

        workflow_refs = (engine.project.pending.meta or {}).get("workflow_refs", []) if engine.project.pending else []

        if not isinstance(workflow_refs, list) or ref_index < 0 or ref_index >= len(workflow_refs):
            self._reply_workflow_error(message_id, "invalid_argument", detail="子 Workflow 引用索引无效")
            return

        ref = workflow_refs[ref_index]
        if isinstance(ref, dict):
            ref_name = ref.get("name", "unknown")
            ref_path = ref.get("path", ref.get("script_path", ""))
        else:
            ref_name = str(ref)
            ref_path = ""

        # Try to resolve a script body for display.
        # Priority: explicit ref_path → template resolution by name.
        # ``load_template`` returns the script content string, not a path;
        # use ``resolve_template_path`` when we only need to locate a file
        # and already have a name.
        script_body = ""
        resolved_path: Optional[str] = None
        sender_id = current_user or ""

        if ref_path and os.path.isfile(ref_path):
            resolved_path = ref_path
            try:
                with open(ref_path, "r", encoding="utf-8") as f:
                    script_body = f.read()
            except OSError:
                script_body = ""
        else:
            # No explicit path — attempt to resolve via the template registry.
            # Try content first; fall back to path-only resolution so preview
            # cards for refs with only a ``name`` field still surface content.
            from ...workflow_engine.templates import resolve_template_path

            try:
                script_body_from_name = load_template(root_path, ref_name, user_id=sender_id)
            except Exception:
                script_body_from_name = None

            if script_body_from_name:
                script_body = script_body_from_name
                resolved_path = resolve_template_path(root_path, ref_name, user_id=sender_id)
            else:
                resolved_path = resolve_template_path(root_path, ref_name, user_id=sender_id)
                if resolved_path:
                    try:
                        with open(resolved_path, "r", encoding="utf-8") as f:
                            script_body = f.read()
                    except OSError:
                        script_body = ""

        meta_extract: Optional[dict[str, Any]] = None
        if script_body:
            try:
                meta_extract = extract_meta_from_script(script_body)
            except Exception:
                meta_extract = None

        preview_lines: list[str] = []
        preview_lines.append(f"**{ref_name}**")
        if resolved_path:
            preview_lines.append(f"路径: `{resolved_path}`")
        else:
            preview_lines.append("路径: _（未找到对应脚本文件）_")
        if meta_extract:
            desc = (meta_extract or {}).get("description")
            if desc:
                preview_lines.append(f"描述: {desc}")
            meta_tools = (meta_extract or {}).get("tools")
            if meta_tools:
                preview_lines.append("工具: " + ", ".join(f"`{t}`" for t in meta_tools))

        # --- Preflight: sub-workflow tools vs parent allowlist ----------------
        # Compare the ref's declared tools against the pending confirmation's
        # selected_tools (the parent allowlist). Highlight any missing tools on
        # the preview card so users can decide whether to adjust tool selection
        # or remove the ref before confirming.
        parent_tools: set[str] = set()
        if engine.project.pending and engine.project.pending.selected_tools:
            parent_tools = set(engine.project.pending.selected_tools)
        ref_tools: set[str] = set((meta_extract or {}).get("tools", []) or [])
        missing = sorted(ref_tools - parent_tools)
        if missing:
            preview_lines.append(
                "\n⚠️ **工具缺失警告**：此子 Workflow 声明的以下工具不在主 Workflow 的允许工具列表中："
                + ", ".join(f"`{m}`" for m in missing)
                + "。执行时会被拒绝并报错；请在主 Workflow 工具选择中添加这些工具，或移除此引用。"
            )

        preview_text = "\n".join(preview_lines)

        # Render the script preview as a collapsible code block when present.
        elements: list[dict[str, Any]] = []
        elements.append({"tag": "markdown", "content": preview_text})

        if script_body:
            from ...workflow_engine.renderer import render_script_preview

            preview = render_script_preview(script_body) or ""
            if preview:
                elements.append(
                    {
                        "tag": "collapsible_panel",
                        "header": {"title": {"tag": "plain_text", "content": "📜 脚本内容"}},
                        "border": {"color": "grey", "corner_radius": "8px"},
                        "expanded": False,
                        "elements": [{"tag": "markdown", "content": preview}],
                    }
                )

        # Build a Feishu card directly (header + element list).
        # The builtin helpers on CardBuilder do not accept an `elements`
        # kwarg; we construct the minimal card dict by hand so this keeps
        # working when the orchestrator isn't available.
        card = {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "blue",
                "title": {
                    "tag": "plain_text",
                    "content": f"子 Workflow 预览：{ref_name}",
                },
            },
            "body": {"elements": elements},
        }
        self.update_card(message_id, card)

    def handle_workflow_remove_workflow_ref(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Remove a sub-workflow reference from ``meta.workflow_refs``.

        Only accepted in AWAITING_CONFIRM / AWAITING_TOOL_SELECT /
        AWAITING_AGENT_SELECT. After mutation, re-renders the confirm card.
        """
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)

        project = self._resolve_project_from_id(project_id, chat_id)
        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        from ...workflow_engine.models import WorkflowStatus

        valid_statuses = (
            WorkflowStatus.AWAITING_CONFIRM,
            WorkflowStatus.AWAITING_TOOL_SELECT,
            WorkflowStatus.AWAITING_AGENT_SELECT,
        )
        if engine.project.status not in valid_statuses:
            self._reply_workflow_error(message_id, "invalid_state")
            return
        if engine.project.pending is None:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        # Session key + sender validation
        from ...thread import get_current_sender_id

        stored_session_key = engine.project.pending.engine_session_key or ""
        button_session_key = value.get("engine_session_key", "")
        if not stored_session_key or button_session_key != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id or ""
        if not stored_initiator or not current_user or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        raw_idx = value.get("ref_index")
        try:
            ref_index = int(raw_idx) if raw_idx is not None else -1
        except (TypeError, ValueError):
            ref_index = -1

        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            if (
                current_project is None
                or current_project.status not in valid_statuses
                or pending is None
                or pending.engine_session_key != stored_session_key
                or pending.initiator_user_id != current_user
            ):
                return
            current_meta = pending.meta or {}
            workflow_refs = current_meta.get("workflow_refs", []) or []
            if not isinstance(workflow_refs, list) or ref_index < 0 or ref_index >= len(workflow_refs):
                self._reply_workflow_error(
                    message_id,
                    "invalid_argument",
                    detail="子 Workflow 引用索引无效",
                )
                return

            new_refs = [ref for index, ref in enumerate(workflow_refs) if index != ref_index]
            new_meta = dict(current_meta)
            new_meta["workflow_refs"] = new_refs
            pending.meta = new_meta

            # Keep the final session CAS held through delivery.
            self._rerender_confirm_or_tool_card(
                message_id,
                chat_id,
                project_id,
                engine,
            )

    def handle_workflow_add_workflow_ref(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Open a selection card to add a sub-workflow reference.

        Lists available templates as buttons. Tapping one appends it to
        ``meta.workflow_refs`` (de-duplicated by name) and re-renders the
        confirm card.
        """
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)

        project = self._resolve_project_from_id(project_id, chat_id)
        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine or not engine.project:
            self._reply_workflow_error(message_id, "session_expired")
            return

        from ...workflow_engine.models import WorkflowStatus
        from ...workflow_engine.templates import TemplateInfo, discover_templates

        valid_statuses = (
            WorkflowStatus.AWAITING_CONFIRM,
            WorkflowStatus.AWAITING_TOOL_SELECT,
            WorkflowStatus.AWAITING_AGENT_SELECT,
        )
        if engine.project.status not in valid_statuses:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        if engine.project.pending is None:
            self._reply_workflow_error(message_id, "invalid_state")
            return

        # Session key + sender validation
        from ...thread import get_current_sender_id

        stored_session_key = engine.project.pending.engine_session_key or ""
        button_session_key = value.get("engine_session_key", "")
        if not stored_session_key or button_session_key != stored_session_key:
            self._reply_workflow_error(message_id, "session_expired")
            return

        current_user = get_current_sender_id() or ""
        stored_initiator = engine.project.pending.initiator_user_id or ""
        if not stored_initiator or not current_user or current_user != stored_initiator:
            self._reply_workflow_error(message_id, "forbidden")
            return

        # A button may carry a specific template name to pick — this happens
        # when the user taps one of the template candidates in the
        # selection card. When no template name is given, show the selector.
        template_name = value.get("template_name")
        if template_name:
            self._apply_add_workflow_ref(
                message_id=message_id,
                chat_id=chat_id,
                project_id=project_id,
                engine=engine,
                template_name=str(template_name),
                root_path=root_path,
                sender_id=current_user,
                expected_session_key=stored_session_key,
                expected_initiator=current_user,
            )
            return

        # Present the template selector card.
        sender_id = current_user or ""
        templates: list[TemplateInfo] = []
        try:
            templates = discover_templates(root_path, user_id=sender_id)
        except Exception:
            templates = []

        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            if (
                current_project is None
                or current_project.status not in valid_statuses
                or pending is None
                or pending.engine_session_key != stored_session_key
                or pending.initiator_user_id != current_user
            ):
                return
            current_meta = pending.meta or {}
            existing_refs: list[Any] = current_meta.get("workflow_refs", []) or []
        existing_names = {ref.get("name") if isinstance(ref, dict) else str(ref) for ref in existing_refs}

        from ...card.actions.dispatch import WORKFLOW_ADD_WORKFLOW_REF

        elements: list[dict[str, Any]] = []
        elements.append(
            {
                "tag": "markdown",
                "content": (
                    "**选择要添加为子 Workflow 引用的模板**\n\n已引用的模板会在下方显示为「已添加」状态，避免重复。"
                ),
            }
        )

        if not templates:
            elements.append({"tag": "markdown", "content": "_当前没有可用的 Workflow 模板_。"})
        else:
            for tpl in templates:
                is_added = tpl.name in existing_names
                tpl_lines: list[str] = [f"**{tpl.name}**  _{tpl.scope}_"]
                if tpl.description:
                    tpl_lines.append(f"— {tpl.description}")
                elements.append({"tag": "markdown", "content": "\n".join(tpl_lines)})

                if is_added:
                    elements.append(
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "✓ 已添加"},
                            "type": "primary",
                            "value": {"action": "__noop__"},
                            "disabled": True,
                        }
                    )
                    continue

                pick_value = {
                    "action": WORKFLOW_ADD_WORKFLOW_REF,
                    "template_name": tpl.name,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": stored_session_key,
                }
                pick_button = [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "➕ 添加"},
                        "type": "default",
                        "value": pick_value,
                        "behaviors": [{"type": "callback", "value": pick_value}],
                    }
                ]
                elements.extend(build_responsive_button_row(pick_button, mobile_force_vertical=True))

        card = {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "blue",
                "title": {
                    "tag": "plain_text",
                    "content": "添加子 Workflow 引用",
                },
            },
            "body": {"elements": elements},
        }
        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            if (
                current_project is None
                or current_project.status not in valid_statuses
                or pending is None
                or pending.engine_session_key != stored_session_key
                or pending.initiator_user_id != current_user
            ):
                return
            self.update_card(message_id, card)

    def _apply_add_workflow_ref(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        engine: Any,
        template_name: str,
        root_path: str,
        sender_id: str,
        expected_session_key: str,
        expected_initiator: str,
    ) -> None:
        """Append a template reference to ``meta.workflow_refs`` and re-render
        the confirm card so the new reference is visible to the user.

        The script body is **not** patched here. Instead, the pending
        confirmation flow (``handle_workflow_confirm_start``) injects the
        corresponding ``workflow(template_name, {})`` call into the verified
        script content just before execution. This guarantees that only refs
        approved on the confirmation card at the moment of confirmation ever
        run, which prevents stale / tampered injections from leaking into the
        executed script (see ``_inject_workflow_refs_into_script``).

        Safety: template names are validated through
        ``validate_template_name``, restricted to those returned by
        ``discover_templates``, and resolved via ``resolve_template_path``
        so forged path-traversal names are rejected before touching disk.
        """
        from ...workflow_engine.models import WorkflowStatus
        from ...workflow_engine.script_gen import extract_meta_from_script
        from ...workflow_engine.templates import (
            discover_templates,
            resolve_template_path,
            validate_template_name,
        )

        valid_statuses = (
            WorkflowStatus.AWAITING_CONFIRM,
            WorkflowStatus.AWAITING_TOOL_SELECT,
            WorkflowStatus.AWAITING_AGENT_SELECT,
        )

        def reply_error_if_current(
            category: str,
            *,
            detail: str,
        ) -> None:
            with engine._lock:
                current_project = engine.project
                pending = current_project.pending if current_project else None
                if (
                    current_project is None
                    or current_project.status not in valid_statuses
                    or pending is None
                    or pending.engine_session_key != expected_session_key
                    or pending.initiator_user_id != expected_initiator
                ):
                    return
                self._reply_workflow_error(
                    message_id,
                    category,
                    detail=detail,
                )

        # 1) Reject invalid / path-traversal names (fail-closed).
        ok, reason = validate_template_name(template_name)
        if not ok:
            logger.warning(
                "Rejected template reference '%s' during add-workflow-ref: %s",
                template_name,
                reason,
            )
            reply_error_if_current(
                "invalid_argument",
                detail=f"非法模板名称: {template_name}",
            )
            return

        # 2) Only accept names discoverable for this user / project root.
        try:
            discoverable = {t.name for t in discover_templates(root_path, user_id=sender_id)}
        except Exception as exc:
            logger.warning("Cannot enumerate templates for add-workflow-ref: %s", repr(exc))
            reply_error_if_current(
                "invalid_argument",
                detail="无法枚举可用模板",
            )
            return
        if template_name not in discoverable:
            reply_error_if_current(
                "invalid_argument",
                detail=f"模板不在可用列表中: {template_name}",
            )
            return

        # 3) Resolve to an absolute path to confirm the template is loadable.
        # This is the final gate — a name that passes validate_template_name
        # and discover_templates but fails resolve_template_path signals a
        # broken disk state; we reject rather than attach an empty ref.
        resolved_path = resolve_template_path(root_path, template_name, user_id=sender_id)
        if not resolved_path:
            logger.warning(
                "Template '%s' could not be resolved during add-workflow-ref",
                template_name,
            )
            reply_error_if_current(
                "invalid_argument",
                detail=f"模板无法加载: {template_name}",
            )
            return

        description = ""
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                ref_script = f.read()
            extracted = extract_meta_from_script(ref_script) or {}
            description = extracted.get("description", "") or ""
        except Exception:
            description = ""

        with engine._lock:
            current_project = engine.project
            pending = current_project.pending if current_project else None
            if (
                current_project is None
                or current_project.status not in valid_statuses
                or pending is None
                or pending.engine_session_key != expected_session_key
                or pending.initiator_user_id != expected_initiator
            ):
                return
            current_meta = pending.meta or {}
            workflow_refs = list(current_meta.get("workflow_refs", []) or [])

            # De-duplicate by name so users don't see identical references.
            already_present = any(
                (ref.get("name") if isinstance(ref, dict) else str(ref)) == template_name for ref in workflow_refs
            )

            # WorkflowRefItem intentionally stores no resolved path/hash.
            if not already_present:
                ref_entry: dict[str, Any] = {"name": template_name}
                if description:
                    ref_entry["description"] = description
                ref_entry["args"] = {}
                ref_entry["failure_policy"] = "skip"
                workflow_refs.append(ref_entry)

            new_meta = dict(current_meta)
            new_meta["workflow_refs"] = workflow_refs
            pending.meta = new_meta

            self._rerender_confirm_or_tool_card(
                message_id,
                chat_id,
                project_id,
                engine,
            )

    def _rerender_confirm_or_tool_card(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        engine: Any,
    ) -> None:
        """Re-render the confirm card (or combined/tool card, depending on state)."""
        from ...workflow_engine.models import WorkflowStatus

        if not engine.project.pending:
            return

        status = engine.project.status
        if status in (WorkflowStatus.AWAITING_TOOL_SELECT, WorkflowStatus.AWAITING_AGENT_SELECT):
            # Get project context for selection controller
            project = self._resolve_project_from_id(project_id, chat_id)
            if project:
                self._send_combined_selection_card(
                    message_id=message_id,
                    chat_id=chat_id,
                    project=project,
                    requirement=engine.project.pending.requirement or "",
                    session_key=engine.project.pending.engine_session_key or "",
                    is_review=(status == WorkflowStatus.AWAITING_TOOL_SELECT),
                )
            return

        # Default path: re-render the confirm card (covers AWAITING_CONFIRM /
        # AWAITING_AGENT_SELECT since both display the same confirm-style card).
        _script_content = self._read_pending_script(engine)
        pending = engine.project.pending
        confirm_card = self._build_confirm_card(
            meta=pending.meta,
            requirement=pending.requirement or "",
            engine_session_key=pending.engine_session_key or "",
            chat_id=chat_id,
            project_id=project_id,
            is_fallback=pending.is_fallback,
            selected_tools=pending.selected_tools,
            script_content=_script_content,
            orchestrator_binding=pending.orchestrator_binding,
            review_agents=pending.review_agents,
            auto_reviewer=pending.auto_reviewer,
        )
        self.update_card(message_id, confirm_card)

    def _resolve_project_from_id(self, project_id: str, chat_id: str) -> Optional["ProjectContext"]:
        """Resolve a ProjectContext from project_id, scoped to the chat.

        Uses ``project_manager.get_project_for_chat`` so that a forged
        ``project_id`` carried in a card action payload cannot be used
        to interact with projects the chat is not allowed to see. Returns
        ``None`` for unknown or invisible projects.
        """
        if not project_id:
            return None
        try:
            return self.ctx.project_manager.get_project_for_chat(project_id, chat_id)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def show_workflow_status(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Show current workflow progress."""
        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine:
            self.reply_text(message_id, "当前没有 Workflow 任务。")
            return

        status_text = engine.get_status_text()
        card_data = engine.get_progress_card()

        if card_data:
            card_content = self._build_workflow_card_from_renderer_data(card_data)
            self.reply_card(message_id, card_content)
        else:
            self.reply_text(message_id, status_text)

    def show_workflow_help(self, message_id: str) -> None:
        """Show the Workflow usage guide as a structured Feishu card."""
        from ...card import CardBuilder
        from ...card.themes import PANEL_STYLES

        # Pull the authoritative Node-version text so the full help keeps the
        # same contract as engine.run_workflow() and the card-entry messages.
        from ...workflow_engine.engine import _node_version_required_text

        sections = [
            (
                "📚 命令列表",
                "`/wf <需求描述>` · AI 生成编排脚本并执行\n"
                "`/wf <模板名> [key=value ...]` · 使用内置或保存的模板\n"
                "`/wf_status` · 查看当前 Workflow 进度\n"
                "`/wf_save <名称> [--global]` · 保存脚本为模板\n"
                "`/wf_list` · 列出可用模板\n"
                "`/wf_delete <名称> [--global]` · 删除模板\n"
                "`/wf_history` · 查看执行历史\n"
                "`/wf_help` · 显示本帮助\n"
                "`/stop_wf` · 停止正在执行的 Workflow",
            ),
            (
                "🧭 执行流程",
                "**① 主编排 Agent** · 选择一个工具和模型\n"
                "**② 评审 Agent** · 选择一个或多个工具和模型，或使用 Auto\n"
                "**③ 自动生成并执行** · 脚本生成后立即启动，多阶段并行执行并实时更新进度卡片",
            ),
            (
                "✨ 核心能力",
                "• 多工具并行调度（coco / claude / aiden / codex / traex）\n"
                "• 工具 Agent 可继续拆分 subagent 并行工作\n"
                "• Agent 按任务动态规划角色分工\n"
                "• Journal 缓存避免重复执行\n"
                "• 子任务自动拆分与依赖编排\n"
                "• 引用并组合多个已保存 Workflow（workflow refs）",
            ),
        ]

        elements = [
            {
                "tag": "markdown",
                "content": "Workflow 通过 AI 编排脚本，将复杂需求拆解为多阶段、多 Agent 协同任务。",
            },
            {
                "tag": "markdown",
                "content": f"**运行要求**\n{_node_version_required_text()}",
            },
            {"tag": "hr"},
        ]
        for index, (title, body) in enumerate(sections):
            elements.append(
                {
                    "tag": "collapsible_panel",
                    "expanded": index == 0,
                    "header": {
                        "title": {"tag": "markdown", "content": f"**{title}**"},
                        "vertical_align": "center",
                    },
                    "border": {
                        "color": PANEL_STYLES["border_normal"],
                        "corner_radius": PANEL_STYLES["corner_radius"],
                    },
                    "vertical_spacing": PANEL_STYLES["vertical_spacing"],
                    "padding": PANEL_STYLES["padding_standard"],
                    "elements": [{"tag": "markdown", "content": body}],
                }
            )

        card = CardBuilder._wrap_card(
            header_title="⚡ Workflow 使用帮助",
            header_template="turquoise",
            elements=elements,
        )
        self.reply_card(message_id, card)

    # ------------------------------------------------------------------
    # Template management commands
    # ------------------------------------------------------------------

    def _handle_wf_save(self, message_id: str, chat_id: str, args: str, project: Optional["ProjectContext"]) -> None:
        """Save the last executed/pending workflow script as a named template.

        Usage: /wf_save <name> [--global]

        安全:
        - 统一调用 validate_template_name + resolve_safe_template_path 校验名称与目标路径
        - --global 仅限管理员使用（is_admin_user(sender_id, settings.admin_user_ids)）
        """
        root_path = self._get_root_path(chat_id, project)
        parts = args.strip().split() if args else []

        if not parts:
            self.reply_text(
                message_id,
                "用法: `/wf_save <模板名> [--global]`\n\n"
                "保存最近执行的 Workflow 脚本为可复用模板。\n"
                "• 默认保存到项目级 (`.ghostap/workflow_templates/`)\n"
                "• `--global` 保存为全局模板（仅限管理员）",
            )
            return

        name = parts[0]
        global_scope = "--global" in parts

        # 安全 1: 名称合法性校验
        from ...workflow_engine.templates import (
            is_admin_user,
            resolve_safe_template_path,
            save_template,
            validate_template_name,
        )

        ok, err_msg = validate_template_name(name)
        if not ok:
            self._reply_workflow_error(message_id, "invalid_argument", detail=err_msg)
            return

        # 安全 2: --global 仅限管理员
        if global_scope:
            from ...thread import get_current_sender_id

            sender_id = get_current_sender_id() or ""
            admin_ids = getattr(getattr(self.ctx, "settings", None), "admin_user_ids", None) or []
            if not is_admin_user(sender_id, admin_ids):
                self._reply_workflow_error(
                    message_id,
                    "forbidden",
                    detail="`--global` 仅允许管理员使用；普通用户可保存为项目级模板（不带 --global）",
                )
                return

        # 安全 3: 目标路径必须位于模板根目录内
        safe_path, path_err = resolve_safe_template_path(root_path, name, global_scope=global_scope)
        if safe_path is None:
            self._reply_workflow_error(message_id, "invalid_argument", detail=path_err or "模板路径不合法")
            return

        # 查找脚本内容
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)
        script_content = None

        # 权限：必须是 workflow 发起者或管理员才能保存。
        # 非发起者保存他人工作流脚本会被拒绝，防止越权复用。
        from ...thread import get_current_sender_id
        from ...workflow_engine.templates import is_admin_user

        caller_id = get_current_sender_id() or ""
        admin_ids_checked = getattr(getattr(self.ctx, "settings", None), "admin_user_ids", None) or []
        stored_initiator = ""
        if engine and engine.project:
            pending_initiator = engine.project.pending.initiator_user_id if engine.project.pending else ""
            stored_initiator = pending_initiator or getattr(engine.project, "initiator_user_id", "")
        if stored_initiator and caller_id != stored_initiator and not is_admin_user(caller_id, admin_ids_checked):
            self._reply_workflow_error(
                message_id,
                "forbidden",
                detail="只有 workflow 发起者或管理员可保存当前脚本；如需保存，请发起者执行保存或联系管理员",
            )
            return

        if engine and engine.project:
            script_path = (
                engine.project.pending.script_path if engine.project.pending else None
            ) or engine.project.script_path
            if script_path:
                try:
                    with open(script_path, "r", encoding="utf-8") as f:
                        script_content = f.read()
                except OSError:
                    pass

        if not script_content:
            self._reply_workflow_error(
                message_id, "invalid_state", detail="没有可保存的 Workflow 脚本，请先执行 `/wf` 生成脚本"
            )
            return

        # 权限：项目级模板若已存在，仅允许原 owner 或 admin 覆盖；
        #       --global 已在前面做过 admin 检查；新建模板则允许任何用户。
        from ...thread import get_current_sender_id
        from ...workflow_engine.templates import BUILTIN_TEMPLATES, can_delete_template

        sender_id = get_current_sender_id() or ""
        admin_ids = getattr(getattr(self.ctx, "settings", None), "admin_user_ids", None) or []

        if not global_scope:
            from pathlib import Path as _Path

            from ...workflow_engine.templates import resolve_safe_template_path as _rsp

            _target_path, _ = _rsp(root_path, name, global_scope=False)
            if _target_path and _Path(_target_path).is_file() and name not in BUILTIN_TEMPLATES:
                _allowed, _reason, _ = can_delete_template(
                    root_path,
                    name,
                    False,
                    None,
                    sender_id=sender_id,
                    admin_user_ids=admin_ids,
                )
                if not _allowed:
                    self._reply_workflow_error(
                        message_id,
                        "forbidden",
                        detail=f"{_reason}，需联系模板创建者或管理员覆盖",
                    )
                    return

        try:
            # save_template 内部也会再次执行名称 + 路径校验（defense-in-depth）
            save_template(
                root_path,
                name,
                script_content,
                global_scope=global_scope,
                owner_id=sender_id or None,
            )
            scope_label = "全局级" if global_scope else "项目级"
            self.reply_text(message_id, f"✅ 模板 `{name}` 已保存（{scope_label}）\n\n使用: `/wf {name}`")
        except (OSError, ValueError) as exc:
            from ...workflow_engine.errors import _strip_internal_details

            # Sanitize: OSError may contain local path fragments like
            # ``/data00/...`` or ``Permission denied: /home/...`` which
            # leak host-internal paths into user-facing messages. We
            # keep the sanitized message for the user and log the raw
            # exception on the server side.
            logger.error("wf_save failed for name=%s: %s", name, repr(exc))
            self._reply_workflow_error(
                message_id,
                "internal_error",
                detail=_strip_internal_details(f"保存失败: {exc}"),
            )

    def _handle_wf_list(self, message_id: str, chat_id: str, project: Optional["ProjectContext"]) -> None:
        """List available workflow templates."""
        root_path = self._get_root_path(chat_id, project)

        from ...thread import get_current_sender_id
        from ...workflow_engine.templates import discover_templates

        sender_id = get_current_sender_id() or ""
        templates = discover_templates(root_path, user_id=sender_id)

        if not templates:
            self.reply_text(
                message_id,
                "暂无可用模板。\n\n使用 `/wf_save <名称>` 保存当前脚本为模板。",
            )
            return

        lines = ["**📋 可用 Workflow 模板**:\n"]
        for t in templates:
            scope_icon = {"project": "📂", "global": "🌐", "builtin": "📦"}.get(t.scope, "")
            desc = f" — {t.description}" if t.description else ""
            lines.append(f"• `{t.name}` {scope_icon}{desc}")

        lines.append("\n使用: `/wf <模板名> [key=value ...]`")
        self.reply_text(message_id, "\n".join(lines))

    def _handle_wf_delete(self, message_id: str, chat_id: str, args: str, project: Optional["ProjectContext"]) -> None:
        """Delete a saved workflow template.

        Usage: /wf_delete <name> [--global]

        安全:
        - 统一调用 validate_template_name + resolve_safe_template_path
        - --global 仅限管理员
        """
        root_path = self._get_root_path(chat_id, project)
        parts = args.strip().split() if args else []

        if not parts:
            self.reply_text(message_id, "用法: `/wf_delete <模板名> [--global]`")
            return

        name = parts[0]
        global_scope = "--global" in parts

        # 安全 1: 名称合法性
        from ...workflow_engine.templates import (
            delete_template,
            is_admin_user,
            resolve_safe_template_path,
            validate_template_name,
        )

        ok, err_msg = validate_template_name(name)
        if not ok:
            self._reply_workflow_error(message_id, "invalid_argument", detail=err_msg)
            return

        # 安全 2: --global 仅限管理员
        if global_scope:
            from ...thread import get_current_sender_id

            sender_id = get_current_sender_id() or ""
            admin_ids = getattr(getattr(self.ctx, "settings", None), "admin_user_ids", None) or []
            if not is_admin_user(sender_id, admin_ids):
                self._reply_workflow_error(
                    message_id,
                    "forbidden",
                    detail="`--global` 仅允许管理员使用；普通用户请删除项目级模板（不带 --global）",
                )
                return

        # 安全 3: 目标路径必须位于模板根目录内
        safe_path, path_err = resolve_safe_template_path(root_path, name, global_scope=global_scope)
        if safe_path is None:
            self._reply_workflow_error(message_id, "invalid_argument", detail=path_err or "模板路径不合法")
            return

        # 权限：项目级模板需 owner 或 admin；全局级模板需 admin（--global 已在上方检查）
        from ...thread import get_current_sender_id
        from ...workflow_engine.templates import can_delete_template

        sender_id = get_current_sender_id() or ""
        admin_ids = getattr(getattr(self.ctx, "settings", None), "admin_user_ids", None) or []

        allowed, reason, _tp = can_delete_template(
            root_path,
            name,
            global_scope,
            None,
            sender_id=sender_id,
            admin_user_ids=admin_ids,
        )
        if not allowed:
            self.reply_text(message_id, f"删除失败: {reason}。请联系模板创建者或管理员。")
            return

        try:
            deleted = delete_template(root_path, name, global_scope=global_scope)
            if deleted:
                self.reply_text(message_id, f"✅ 模板 `{name}` 已删除。")
            else:
                self._reply_workflow_error(message_id, "invalid_state", detail=f"模板 `{name}` 不存在或无法删除")
        except (OSError, ValueError) as exc:
            self._reply_workflow_error(message_id, "internal_error", detail=f"删除失败: {exc}")

    def _handle_wf_history(self, message_id: str, chat_id: str, project: Optional["ProjectContext"]) -> None:
        """Show workflow execution history."""
        import time as _time

        from src.workflow_engine.history import WorkflowHistory

        root_path = self._get_root_path(chat_id, project)
        history = WorkflowHistory(root_path)
        entries = history.list_recent(limit=10)

        if not entries:
            self.reply_text(
                message_id,
                "📜 **执行历史**\n\n暂无历史记录。\n\n执行 `/wf <需求>` 启动第一个 Workflow。",
            )
            return

        lines: list[str] = ["📜 **执行历史** (最近 10 次)\n"]
        for entry in entries:
            status_icon = {"completed": "✅", "failed": "❌", "running": "🔄"}.get(entry.status, "⏸️")
            # Format timestamp
            ts = _time.strftime("%m-%d %H:%M", _time.localtime(entry.started_at))
            # Duration
            dur = ""
            if entry.finished_at and entry.started_at:
                elapsed = entry.finished_at - entry.started_at
                if elapsed < 60:
                    dur = f" {elapsed:.0f}s"
                else:
                    dur = f" {elapsed / 60:.1f}min"
            # Token count
            tokens = ""
            if entry.total_tokens > 0:
                if entry.total_tokens >= 1_000_000:
                    tokens = f" {entry.total_tokens / 1_000_000:.1f}M tok"
                elif entry.total_tokens >= 1_000:
                    tokens = f" {entry.total_tokens / 1_000:.0f}K tok"
                else:
                    tokens = f" {entry.total_tokens} tok"

            err_suffix = f" — {entry.error[:40]}" if entry.error else ""
            lines.append(f"{status_icon} `{entry.name}` {ts}{dur}{tokens} ({entry.total_agents} agents){err_suffix}")

        self.reply_text(message_id, "\n".join(lines))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_root_path(self, chat_id: str, project: Optional["ProjectContext"]) -> str:
        """Resolve root_path from project or chat."""
        if project:
            return project.root_path
        return self.get_working_dir(chat_id)

    @staticmethod
    def _new_workflow_script_path(root_path: str) -> str:
        """Allocate a generation-owned script path without sharing file state."""
        script_dir = os.path.join(root_path, ".ghostap", "workflow_scripts")
        os.makedirs(script_dir, exist_ok=True)
        return os.path.join(
            script_dir,
            f"generated-workflow-{uuid.uuid4().hex}.js",
        )

    @staticmethod
    def _remove_owned_workflow_artifact(
        path: str | None,
        *,
        root_path: str,
    ) -> None:
        """Remove only files created as disposable Workflow artifacts.

        Saved templates live in ``.ghostap/workflows`` and are deliberately
        outside this allowlist. This helper therefore cannot delete a user's
        reusable Workflow merely because stale lifecycle state was corrupted.
        """
        if not path:
            return

        import tempfile

        candidate = os.path.realpath(os.path.abspath(path))
        basename = os.path.basename(candidate)
        script_dir = os.path.realpath(os.path.join(root_path, ".ghostap", "workflow_scripts"))
        temp_root = os.path.realpath(tempfile.gettempdir())
        candidate_dir = os.path.dirname(candidate)

        generated_source = (
            candidate_dir == script_dir
            and (basename.startswith("generated-workflow-") or basename == "generated_workflow.js")
            and basename.endswith(".js")
        )
        confirmed_copy = (
            basename.startswith("ghostap-confirmed-")
            and basename.endswith(".js")
            and os.path.commonpath((candidate, temp_root)) == temp_root
        )
        if not (generated_source or confirmed_copy):
            logger.warning(
                "[workflow] Refusing to remove non-owned workflow artifact: %s",
                basename,
            )
            return
        try:
            os.remove(candidate)
        except OSError:
            pass

    def _supersede_incomplete_workflow(
        self,
        engine: Any,
        *,
        root_path: str,
        current_user: str,
    ) -> tuple[bool, str | None, _WorkflowLifecycleOwner | None]:
        """Atomically cancel the old lifecycle and admit the newest request.

        A real runtime is never replaced in-place because ``WorkflowEngine``
        resources are shared until its execution thread fully unwinds.
        """
        from ...workflow_engine.models import WorkflowProject, WorkflowStatus

        admission_owner = _WorkflowLifecycleOwner(
            session_key=uuid.uuid4().hex,
            initiator_user_id=current_user,
        )
        owners: list[_WorkflowLifecycleOwner] = []
        artifacts: list[str] = []
        with engine._lock:
            if getattr(engine, "_closing", False) is True:
                return False, "running", None
            if engine.is_running is True:
                return False, "running", None

            wf_project = engine.project
            if wf_project is None:
                wf_project = WorkflowProject()
                engine._project = wf_project
            pending = wf_project.pending
            selection_owner = vars(engine).get("_workflow_selection_owner")
            generation_owner = vars(engine).get("_script_generation_owner")
            start_owner = vars(engine).get("_workflow_start_owner")
            incomplete = bool(
                wf_project.status in _workflow_pending_statuses()
                or isinstance(selection_owner, _WorkflowLifecycleOwner)
                or isinstance(generation_owner, _WorkflowLifecycleOwner)
                or isinstance(start_owner, _WorkflowLifecycleOwner)
                or wf_project.status == WorkflowStatus.RUNNING
            )
            if incomplete:
                stored_initiator = (
                    (getattr(pending, "initiator_user_id", None) if pending is not None else None)
                    or getattr(
                        selection_owner,
                        "initiator_user_id",
                        None,
                    )
                    or getattr(wf_project, "initiator_user_id", None)
                )
                admin_ids = set(getattr(self.ctx.settings, "admin_user_ids", []) or [])
                if not current_user or not stored_initiator:
                    return False, "forbidden", None
                if current_user != stored_initiator and current_user not in admin_ids:
                    return False, "forbidden", None

                for owner in (
                    selection_owner,
                    generation_owner,
                    start_owner,
                ):
                    if not isinstance(owner, _WorkflowLifecycleOwner):
                        continue
                    owner.stop_event.set()
                    owner.heartbeat_stop_event.set()
                    self._retire_workflow_owner(engine, owner)
                    owners.append(owner)
                    for artifact in (
                        owner.source_script_path,
                        owner.execution_script_path,
                    ):
                        if artifact:
                            artifacts.append(artifact)

                if pending and pending.script_path:
                    artifacts.append(pending.script_path)
                if isinstance(start_owner, _WorkflowLifecycleOwner) and wf_project.script_path:
                    artifacts.append(wf_project.script_path)

                engine._script_generation_owner = None
                engine._workflow_start_owner = None
                engine._project = WorkflowProject()
            else:
                # Starting a new lifecycle retires the previous terminal
                # source.  It remains saveable until this exact boundary,
                # then is no longer the "latest" script for /wf_save.
                if wf_project.script_path:
                    artifacts.append(wf_project.script_path)
                engine._project = WorkflowProject()

            engine._workflow_selection_owner = admission_owner

        # An already-authorized card delivery may finish, but always before
        # the new request publishes its first card.
        for owner in owners:
            with owner.delivery_lock:
                pass
            if not owner.claimed_event.is_set() and not owner.worker_started_event.is_set():
                owner.done_event.set()
        for artifact in dict.fromkeys(artifacts):
            self._remove_owned_workflow_artifact(
                artifact,
                root_path=root_path,
            )
        return True, None, admission_owner

    def _generate_script_via_ai(
        self,
        requirement: str,
        root_path: str,
        selected_tools: list[str] | None = None,
        engine: Any = None,
        progress_callback: Any = None,
        *,
        output_path: str | None = None,
        cancel_event: threading.Event | None = None,
        artifact_lock: Any = None,
    ) -> tuple[str, dict[str, Any] | None, bool]:
        """Generate a workflow script via AI with fallback to simple generation.

        Args:
            requirement: The user's requirement text.
            root_path: Project root path.
            selected_tools: Optional list of tools selected by the user. If provided,
                the script generator will be encouraged to use these tools.
            engine: Optional workflow engine instance. If provided, the selected
                orchestrator_agent from pending state will be used for script generation.

        Returns:
            Tuple of (script_path, meta_dict_or_None, is_fallback).
        """
        from ...agent_session import close_session_safely, create_engine_session
        from ...workflow_engine.constants import (
            DEFAULT_ORCHESTRATOR_AGENT,
            SCRIPT_GEN_TIMEOUT_S,
        )
        from ...workflow_engine.script_gen import (
            build_script_gen_prompt,
            extract_meta_from_script,
            validate_generated_script,
        )

        # Resolve agent type: use pending.orchestrator_agent if available, otherwise default
        agent_type = (
            engine.project.pending.orchestrator_agent
            if engine and engine.project and engine.project.pending and engine.project.pending.orchestrator_agent
            else DEFAULT_ORCHESTRATOR_AGENT
        )

        script_dir = os.path.join(root_path, ".ghostap", "workflow_scripts")
        os.makedirs(script_dir, exist_ok=True)
        script_path = output_path or self._new_workflow_script_path(root_path)
        if os.path.realpath(os.path.dirname(script_path)) != os.path.realpath(script_dir) or not os.path.basename(
            script_path
        ).endswith(".js"):
            raise ValueError("Workflow generation output must stay in the script staging directory")

        def ensure_not_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise _WorkflowGenerationCancelled("Workflow generation cancelled")

        def write_generated_content(content: str) -> None:
            lock_context = artifact_lock if artifact_lock is not None else nullcontext()
            with lock_context:
                ensure_not_cancelled()
                with open(script_path, "w", encoding="utf-8") as file:
                    file.write(content)

        def write_fallback() -> str:
            lock_context = artifact_lock if artifact_lock is not None else nullcontext()
            with lock_context:
                ensure_not_cancelled()
                return self._write_fallback_script(
                    script_path,
                    requirement,
                    selected_tools,
                )

        # Resolve available tools via dynamic registry
        from ...workflow_engine.tool_registry import get_available_tools

        available_tools = get_available_tools(require_available=True)
        if not available_tools:
            logger.warning("No executable workflow tools detected; using fallback script")
            return write_fallback(), None, True

        # Filter available tools to selected ones if provided
        if selected_tools:
            available_tools = {k: v for k, v in available_tools.items() if k in selected_tools}
            if not available_tools:
                logger.warning("Selected workflow tools are unavailable; using fallback script")
                return write_fallback(), None, True

        # Get orchestrator binding and review agents from pending state
        orchestrator_binding = None
        review_agents = None
        auto_reviewer = None
        selected_model_name = None
        if engine and engine.project and engine.project.pending:
            orchestrator_binding = engine.project.pending.orchestrator_binding
            review_agents = engine.project.pending.review_agents
            auto_reviewer = engine.project.pending.auto_reviewer
            # Extract model_name from orchestrator_binding if not using default
            if orchestrator_binding and not orchestrator_binding.use_default_model:
                selected_model_name = orchestrator_binding.model_name

        prompt = build_script_gen_prompt(
            requirement=requirement,
            available_tools=available_tools,
            orchestrator_agent=agent_type,
            orchestrator_binding=orchestrator_binding,
            review_agents=review_agents,
            auto_reviewer=auto_reviewer,
        )

        # Attempt AI generation via one-shot ACP session
        #
        # SECURITY: Script generation runs an untrusted model against a
        # user-supplied requirement. We disable ``auto_approve`` and attach
        # a read-only tool filter so the generation session cannot mutate
        # the filesystem, execute arbitrary commands, or reach the network
        # even if the model tries to. The user confirms the *generated*
        # script later in the workflow card.
        session = None
        try:
            ensure_not_cancelled()
            session = create_engine_session(
                agent_type=agent_type,
                cwd=root_path,
                thread_id="workflow_script_gen",
                auto_approve=False,
                require_tool_filter=True,
                model_name=selected_model_name,
                cancel_event=cancel_event,
            )
            if session is None:
                logger.warning("Failed to create script-gen session; using fallback")
                return write_fallback(), None, True

            if progress_callback:
                progress_callback("已创建 AI 会话，正在发送生成请求...")

            # Apply read-only tool filter for script generation. The allowed
            # set is intentionally small: only read-side filesystem + search
            # introspection. Mutation tools (write, shell, network, code
            # execution) are rejected so a compromised model cannot bypass
            # the user confirmation step.
            _MUTATING_TOOLS: frozenset[str] = frozenset(
                [
                    "execute_command",
                    "create_terminal",
                    "run_terminal",
                    "run_shell",
                    "shell",
                    "bash",
                    "write_file",
                    "write_text_file",
                    "delete_file",
                    "remove_file",
                    "mkdir",
                    "patch_file",
                    "apply_diff",
                    "edit_file",
                    "write_to_file",
                    "http_request",
                    "http_get",
                    "http_post",
                    "fetch",
                    "download",
                    "upload",
                    "network_request",
                    "url_open",
                    "send_message",
                    "send_email",
                    "create_issue",
                ]
            )

            def _script_gen_tool_filter(tool_name: str, _params: dict | None) -> bool:
                if not isinstance(tool_name, str):
                    return False
                norm = tool_name.lower().strip()
                if norm in _MUTATING_TOOLS:
                    return False
                # Reject any tool whose name hints at a mutation.
                if any(
                    token in norm
                    for token in (
                        "write",
                        "delete",
                        "remove",
                        "exec",
                        "run",
                        "patch",
                        "post",
                        "upload",
                        "send",
                        "create",
                    )
                ):
                    return False
                return True

            try:
                session.set_tool_filter(_script_gen_tool_filter)
            except (AttributeError, TypeError, Exception) as exc:
                # FAIL-CLOSED: If we cannot enforce the read-only tool filter
                # on the script-gen session, we must not proceed to call the
                # model — otherwise a compromised or confused model could
                # mutate the filesystem or execute arbitrary commands before
                # the user has confirmed the script. Instead, fall back to a
                # static pre-vetted fallback script so the workflow path still
                # reaches the confirmation card.
                logger.warning(
                    "Applying script-gen tool filter failed (%s); fail-closed to "
                    "static fallback script; model prompt is NOT sent.",
                    type(exc).__name__,
                )
                from ...workflow_engine.script_gen import FALLBACK_SCRIPT

                write_generated_content(FALLBACK_SCRIPT)
                if session is not None:
                    close_session_safely(session)
                meta = {"name": "fallback-orchestration"}
                return script_path, meta, True

            script_gen_timeout_s = getattr(self.settings, "workflow_script_gen_timeout_s", SCRIPT_GEN_TIMEOUT_S)
            result = session.send_prompt(prompt, timeout=script_gen_timeout_s)
            ensure_not_cancelled()

            if progress_callback:
                progress_callback("收到模型响应，正在验证脚本...")

            if result and result.text:
                script_content = self._strip_markdown_fences(result.text.strip())

                is_valid, errors = validate_generated_script(script_content, review_agents=review_agents)
                if is_valid:
                    write_generated_content(script_content)
                    meta = extract_meta_from_script(script_content)
                    if meta is None:
                        meta = {}
                    return script_path, meta, False
                else:
                    logger.warning("Generated script failed validation: %s", errors)
            else:
                logger.warning("AI returned empty script content")

        except _WorkflowGenerationCancelled:
            raise
        except Exception as exc:
            logger.error("Script generation via AI failed: %s", exc, exc_info=True)
        finally:
            if session is not None:
                close_session_safely(session)

        # Fallback
        return write_fallback(), None, True

    @staticmethod
    def _strip_markdown_fences(content: str) -> str:
        """Remove markdown code fences and natural language preamble from AI output.

        AI models sometimes prefix their code output with explanatory text like
        "Let me analyze..." or "Here's the workflow script:". This method
        extracts the actual JavaScript code by:
        1. Attempting to extract code from markdown fences (even if preceded by text)
        2. Stripping any natural language preamble before the actual JS code
        """
        import re

        # Strategy 1: Find markdown code fence containing the actual code.
        # This handles cases like: "Here's the script:\n```javascript\n...code...\n```"
        fence_match = re.search(r"```\s*(?:javascript|js|)\s*\n", content, re.IGNORECASE)
        if fence_match:
            after_fence = content[fence_match.end() :]
            # Find the closing fence (last occurrence to handle nested fences in strings)
            close_idx = after_fence.rfind("```")
            if close_idx >= 0:
                content = after_fence[:close_idx].rstrip()
            else:
                content = after_fence.rstrip()
            # After extracting from fences, if it looks like valid JS, return it
            stripped = content.lstrip()
            if stripped and re.match(
                r"^(export|/[/*]|const |let |var |\"use strict\"|'use strict')",
                stripped,
            ):
                return content.strip()

        # Strategy 2: Original logic — content starts directly with a fence
        elif content.startswith("```"):
            lines = content.split("\n", 1)
            content = lines[1] if len(lines) > 1 else content
            if content.rstrip().endswith("```"):
                content = content.rstrip()[:-3].rstrip()

        # Strategy 3: Detect and strip natural language preamble.
        # If content doesn't start with valid JS syntax, find the actual code start.
        stripped = content.lstrip()
        if stripped and not re.match(
            r"^(export|/[/*]|const |let |var |\"use strict\"|'use strict'|/\*\*)",
            stripped,
        ):
            # Look for the start of the actual export statement (multiline search)
            export_match = re.search(
                r"^(export\s+const\s+meta\s*=|export\s+default\s)",
                content,
                re.MULTILINE,
            )
            if export_match:
                start_idx = export_match.start()
                # Include preceding JSDoc/comment lines that are part of the code
                preceding = content[:start_idx]
                if preceding.rstrip():
                    lines_before = preceding.rstrip().split("\n")
                    # Walk backwards to include leading comment block
                    comment_start = start_idx
                    for line in reversed(lines_before):
                        ls = line.strip()
                        if ls.startswith("//") or ls.startswith("*") or ls.startswith("/*") or ls.endswith("*/"):
                            # This line is a comment, include it
                            idx = content.rfind(line, 0, comment_start)
                            if idx >= 0:
                                comment_start = idx
                        else:
                            break
                    start_idx = comment_start
                content = content[start_idx:]

        return content.strip()

    @staticmethod
    def _find_function_close_brace(script_content: str, open_idx: int) -> int | None:
        """Return the index of the matching ``}`` for the function body starting at ``open_idx``.

        String literals (``'...'``, ``\"...\"``, ```...` ``) and comments
        (``// ...``, ``/* ... */``) are skipped so braces inside them do not
        affect the depth counter. Backslash escapes are respected inside
        string literals. Returns ``None`` if the end of file is reached
        without finding the closing brace at depth 0.
        """
        depth = 1  # caller already located the opening `{`
        i = open_idx + 1
        n = len(script_content)
        while i < n:
            ch = script_content[i]
            if ch == "/" and i + 1 < n:
                nxt = script_content[i + 1]
                if nxt == "/":
                    # Line comment: skip to end-of-line.
                    j = script_content.find("\n", i + 2)
                    i = n if j == -1 else j
                    continue
                if nxt == "*":
                    # Block comment: skip to closing */.
                    j = script_content.find("*/", i + 2)
                    i = n if j == -1 else j + 2
                    continue
            if ch in ("'", '"', "`"):
                quote = ch
                j = i + 1
                while j < n:
                    c = script_content[j]
                    if c == "\\" and j + 1 < n:
                        j += 2
                        continue
                    if c == quote:
                        break
                    j += 1
                i = j + 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return None

    @staticmethod
    def _read_pending_script(engine: Any) -> str:
        """Read script content from pending.script_path for confirm card preview."""
        path = engine.project.pending.script_path if (engine.project and engine.project.pending) else None
        if not path:
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""

    @staticmethod
    def _inject_workflow_refs_into_script(script_content: str, refs: list[dict]) -> str:
        """Inject sub-workflow references into the script body.

        Each ref dict may contain:

        - ``name`` (required, string): the template name to invoke.
        - ``args`` (optional, dict): keyword arguments forwarded to the
          sub-workflow call. Defaults to an empty object.
        - ``failure_policy`` (optional, string): ``"fail_fast"`` raises on
          error; ``"skip"`` (default) logs and continues. Any unrecognised
          value is treated as ``"skip"``.
        - ``description`` (optional, string): free-form text surfaced in the
          injected comment so editors can trace why a ref exists.

        Order of operations:

        1. If ``// <<WORKFLOW_REFS_BEGIN>>`` / ``// <<WORKFLOW_REFS_END>>`` markers
           exist in ``script_content``, the block between them is replaced with
           the generated workflow-ref calls (honoring the author's placement).
        2. Otherwise, refs are injected just before the last ``}`` closing the
           ``export default async function [NAME](...)`` body.  Both anonymous
           and named default functions are supported, matching the template
           style used by the built-in ``code-audit`` and similar templates.
        3. String/comment literal boundaries are respected when searching for
           the function's closing brace: a ``}`` inside ``'...'``, ``\"...\"``,
           ```...` ``, or a ``// ...`` / ``/* ... */`` comment does not count
           toward brace depth.
        4. If a ref name is already invoked by name anywhere in the script
           (via ``workflow('name'`` or ``workflow("name"``) we skip generating
           a duplicate call for it.

        Args:
            script_content: The raw JavaScript script content.
            refs: A list of reference dicts. Only refs with a non-empty
                ``name`` are processed.

        Returns:
            The updated script content with refs injected.
        """
        if not refs:
            return script_content

        import json as _json
        import re as _re

        # Build the list of refs to inject, de-duplicated, and skip any that
        # already have a matching ``workflow('...'`` call in the script.
        refs_to_inject: list[dict] = []
        seen: set[str] = set()
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            name = ref.get("name")
            if not name or not isinstance(name, str):
                continue
            if name in seen:
                continue
            if _re.search(
                r"\bworkflow\s*\(\s*[\"']" + _re.escape(name) + r"[\"']",
                script_content,
            ):
                # Already present — skip to avoid duplication.
                seen.add(name)
                continue
            seen.add(name)
            refs_to_inject.append(ref)

        if not refs_to_inject:
            return script_content

        generated_lines = []
        for ref in refs_to_inject:
            name = ref["name"]
            args_obj = ref.get("args") or {}
            try:
                args_json = _json.dumps(args_obj, ensure_ascii=False)
            except (TypeError, ValueError):
                args_json = "{}"
            policy = (ref.get("failure_policy") or "skip").lower()
            desc = ref.get("description") or ""
            safe_desc = desc.replace("\n", " ").replace("\r", " ")

            if policy == "fail_fast":
                call = f"  await workflow('{name}', {args_json});"
                header = f"  // ref: {name}{(' -- ' + safe_desc) if safe_desc else ''}  (failure_policy=fail_fast)"
            else:
                call = (
                    f"  try {{ await workflow('{name}', {args_json}); }} "
                    f"catch (e) {{ console.log('sub-workflow {name} skipped:', e); }}"
                )
                header = f"  // ref: {name}{(' -- ' + safe_desc) if safe_desc else ''}"
            generated_lines.append(header)
            generated_lines.append(call)

        block = "\n".join(generated_lines) + "\n"

        # Strategy 1: replace marker block if present.
        marker_start = "// <<WORKFLOW_REFS_BEGIN>>"
        marker_end = "// <<WORKFLOW_REFS_END>>"
        idx_s = script_content.find(marker_start)
        if idx_s != -1:
            idx_e = script_content.find(marker_end, idx_s + len(marker_start))
            if idx_e != -1:
                return (
                    script_content[:idx_s]
                    + marker_start
                    + "\n"
                    + block
                    + marker_end
                    + script_content[idx_e + len(marker_end) :]
                )

        # Strategy 2: find the default export function body (supporting both
        # ``export default async function (args)`` and
        # ``export default async function main(args = {})``) and inject just
        # before its final ``}``.  Brace depth is counted outside of strings
        # and comments; parameter-list braces (e.g. ``args = {}``) are
        # skipped by locating the parameter-list closing ``)`` first.
        default_match = _re.search(
            r"export\s+default\s+(?:async\s+)?function\s*(?:[A-Za-z_$][\w$]*)?\s*\(",
            script_content,
        )
        if default_match:
            # 1. Locate the matching ``)`` for the parameter list so braces
            #    inside default values (``args = {}``) are not mistaken for
            #    the function body opening brace.
            paren_open = default_match.end() - 1  # points at the opening `(`
            paren_depth = 1
            paren_i = paren_open + 1
            paren_n = len(script_content)
            close_paren = -1
            while paren_i < paren_n:
                c = script_content[paren_i]
                if c in ("'", '"', "`"):
                    q = c
                    j = paren_i + 1
                    while j < paren_n:
                        if script_content[j] == "\\" and j + 1 < paren_n:
                            j += 2
                            continue
                        if script_content[j] == q:
                            break
                        j += 1
                    paren_i = j + 1
                    continue
                if c == "(":
                    paren_depth += 1
                elif c == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        close_paren = paren_i
                        break
                paren_i += 1
            if close_paren != -1:
                # 2. Now locate the opening ``{`` of the function body,
                #    allowing for ``=>`` style or simply ``{`` after ``)``.
                body_open = script_content.find("{", close_paren + 1)
                if body_open != -1:
                    insert_at = WorkflowHandler._find_function_close_brace(
                        script_content,
                        body_open,
                    )
                    if insert_at is not None:
                        return (
                            script_content[:insert_at].rstrip()
                            + "\n"
                            + block
                            + "  "  # preserve closing-brace indentation
                            + script_content[insert_at:].lstrip("\n")
                        )

        # Fallback: append at end.
        return script_content.rstrip() + "\n\n" + block

    @staticmethod
    def _write_fallback_script(script_path: str, requirement: str, selected_tools: list[str] | None = None) -> str:
        """Write a simple fallback script and return its path."""
        from ...workflow_engine.script_gen import generate_simple_script

        script_content = generate_simple_script(requirement, selected_tools=selected_tools)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        return script_path

    @staticmethod
    def _build_confirmed_run_spec(
        *,
        pending: Any,
        engine: Any,
        task: str,
        chat_id: str,
        topic_id: str | None,
    ):
        """Freeze the confirmation-time selection into one execution contract."""
        from src.spec_engine.review_agents import ReviewAgentBinding

        from ...workflow_engine.constants import (
            MAX_TOTAL_AGENTS,
            WORKFLOW_TOTAL_TIMEOUT_S,
        )
        from ...workflow_engine.run_spec import WorkflowRunSpec

        selected_tools = list(dict.fromkeys(pending.selected_tools or []))
        orchestrator = pending.orchestrator_binding
        if orchestrator is None:
            primary_tool = (
                str(pending.orchestrator_agent or "").strip()
                or (selected_tools[0] if selected_tools else "coco")
            )
            engine_agent_type = getattr(engine, "_agent_type", None)
            if not isinstance(engine_agent_type, str) or not engine_agent_type.strip():
                engine_agent_type = primary_tool
            engine_model = getattr(engine, "_model_name", None)
            if not isinstance(engine_model, str) or not engine_model.strip():
                engine_model = None
            orchestrator = ReviewAgentBinding(
                provider="legacy",
                tool_name=primary_tool,
                display_name=primary_tool,
                agent_type=engine_agent_type,
                model_name=engine_model,
                model_display_name=engine_model,
                selection_key=f"legacy:{primary_tool}:{engine_model or 'default'}",
                use_default_model=engine_model is None,
            )

        reviewers = tuple(pending.review_agents or ())
        auto_reviewer = pending.auto_reviewer
        if auto_reviewer is None:
            # Explicit compatibility for restored pre-RunSpec confirmations;
            # all newly selected flows persist the boolean before generation.
            auto_reviewer = not reviewers

        allowed_tools = list(selected_tools)
        for tool in [orchestrator.tool_name, *(reviewer.tool_name for reviewer in reviewers)]:
            if tool and tool not in allowed_tools:
                allowed_tools.append(tool)

        tool_model_map: dict[str, str | None] = {
            tool: None for tool in allowed_tools
        }
        tool_model_map[orchestrator.tool_name] = (
            None if orchestrator.use_default_model else orchestrator.model_name
        )
        for reviewer in reviewers:
            tool_model_map.setdefault(
                reviewer.tool_name,
                None if reviewer.use_default_model else reviewer.model_name,
            )
            # When a tool is not shared with the orchestrator, the explicit
            # reviewer selection is also its ordinary runtime binding.
            if reviewer.tool_name != orchestrator.tool_name:
                tool_model_map[reviewer.tool_name] = (
                    None if reviewer.use_default_model else reviewer.model_name
                )

        settings = getattr(engine, "settings", None)
        raw_timeout = getattr(
            settings,
            "workflow_total_timeout_s",
            WORKFLOW_TOTAL_TIMEOUT_S,
        )
        try:
            timeout_s = int(raw_timeout)
        except (TypeError, ValueError):
            timeout_s = WORKFLOW_TOTAL_TIMEOUT_S
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None

        return WorkflowRunSpec(
            orchestrator=orchestrator,
            reviewers=reviewers,
            tool_model_map=tool_model_map,
            task=task,
            chat_id=chat_id,
            topic_id=topic_id,
            budget=MAX_TOTAL_AGENTS,
            deadline=deadline,
            auto_reviewer=bool(auto_reviewer),
            initiator_user_id=pending.initiator_user_id or None,
            allowed_tools=tuple(allowed_tools),
        )

    def _build_confirm_card(
        self,
        meta: dict[str, Any] | None,
        requirement: str,
        engine_session_key: str,
        chat_id: str,
        project_id: str,
        is_fallback: bool = False,
        selected_tools: list[str] | None = None,
        script_content: str = "",
        orchestrator_binding: dict | None = None,
        review_agents: list[dict] | None = None,
        auto_reviewer: bool | None = None,
    ) -> dict:
        """Build a Feishu card showing the workflow script preview for confirmation.

        Returns a Feishu card JSON dict ready for reply_card/send_card_to_chat.
        """
        from ...card import CardBuilder
        from ...card.actions.dispatch import (
            WORKFLOW_CANCEL,
            WORKFLOW_CONFIRM_START,
            WORKFLOW_SELECT_TOOL,
        )
        from ...card.render.budget import RenderBudget
        from ...card.ui_text import UI_TEXT
        from ...workflow_engine.tool_registry import get_available_tools

        # Extract meta info
        (meta or {}).get("name", "generated-workflow")
        (meta or {}).get("description", requirement[:100])
        phases = (meta or {}).get("phases", [])
        tools = (meta or {}).get("tools", selected_tools or [])
        phase_tool_mapping: dict = (meta or {}).get("phase_tool_mapping", {})
        workflow_refs = (meta or {}).get("workflow_refs", [])

        # Format orchestrator binding display
        orchestrator_display = ""
        if orchestrator_binding:
            tool_name = orchestrator_binding.tool_name
            model_name = orchestrator_binding.model_name
            use_default = getattr(orchestrator_binding, "use_default_model", True)
            orchestrator_display = f"**主编排 Agent**: `{tool_name}`"
            if use_default:
                orchestrator_display += f" (默认: {orchestrator_binding.model_display_name or '默认模型'})"
            elif model_name:
                orchestrator_display += f" · {orchestrator_binding.model_display_name or model_name}"

        # Format review agents display
        review_display = ""
        if auto_reviewer is True:
            review_display = (
                "**评审 Agent**: Auto（无独立 Reviewer；编排器负责最终检查）"
            )
        elif review_agents and len(review_agents) > 0:
            review_lines = ["**评审 Agent**:"]
            for i, agent in enumerate(review_agents):
                tool_name = agent.tool_name
                model_name = agent.model_name
                use_default = getattr(agent, "use_default_model", True)
                line = f"{i + 1}. `{tool_name}`"
                if use_default:
                    line += f" (默认: {agent.model_display_name or '默认模型'})"
                elif model_name:
                    line += f" · {agent.model_display_name or model_name}"
                review_lines.append(line)
            review_display = "\n".join(review_lines)
        elif auto_reviewer is None and review_agents is not None:
            # Restored legacy cards did not persist the explicit Auto flag.
            review_display = (
                "**评审 Agent**: Auto（无独立 Reviewer；编排器负责最终检查）"
            )

        # Pre-compute has_mismatch for action button state (used in both modes)
        allowed_tools = set(selected_tools) if selected_tools else set(tools)
        script_tools = set(tools)
        has_mismatch = bool(script_tools - allowed_tools)

        # --- Node budget pre-check ---
        # Estimate element count and apply truncation if needed
        estimated_nodes = 0
        estimated_nodes += 5  # requirement, meta, hr, phases header, workflow refs
        if phases:
            estimated_nodes += len(phases)
        if script_content:
            estimated_nodes += 2  # script preview header + content
        if selected_tools:
            estimated_nodes += len(selected_tools)
        estimated_nodes += 10  # budget buttons, action buttons, etc.

        estimated_nodes > RenderBudget.NODE_BUDGET * 0.8

        # Shared helpers
        phase_count = len(phases)
        tool_count = len(selected_tools) if selected_tools else len(tools)

        # Build elements — unified layout across normal & truncated modes.
        elements: list[dict] = []

        # --- 1. Stepper (vertical, one-line per step) ---
        elements.append(self._build_workflow_stepper(current=3, total=3))

        if is_fallback:
            elements.append(
                {
                    "tag": "markdown",
                    "content": "⚠️ AI 脚本生成失败，已使用默认模板。结果可能不完全匹配需求。",
                }
            )

        # --- 2. Requirement summary (first screen) ---
        req_trim = requirement[:300] if len(requirement) > 300 else requirement
        elements.append(
            {
                "tag": "markdown",
                "content": f"**需求**\n> {req_trim}",
            }
        )

        # --- 2b. Agent selection display ---
        agent_info_lines = []
        if orchestrator_display:
            agent_info_lines.append(orchestrator_display)
        if review_display:
            agent_info_lines.append(review_display)
        if agent_info_lines:
            elements.append({"tag": "hr"})
            elements.append(
                {
                    "tag": "markdown",
                    "content": "\n".join(agent_info_lines),
                }
            )

        # --- 3. Stats: phases / tools (one-line pair) ---
        elements.append(
            {
                "tag": "column_set",
                "flex_mode": "bisect",
                "background_style": "default",
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": [
                            {"tag": "markdown", "content": f"**{phase_count}**\n<font color='grey'>阶段数</font>"},
                        ],
                    },
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": [
                            {"tag": "markdown", "content": f"**{tool_count}**\n<font color='grey'>工具数</font>"},
                        ],
                    },
                ],
            }
        )

        # --- 4. Tool-mismatch status bar + single primary fix action ---
        if has_mismatch:
            missing = sorted(script_tools - allowed_tools)
            missing_display = ", ".join(f"`{m}`" for m in missing)
            elements.append(
                {
                    "tag": "markdown",
                    "content": (
                        f"⚠️ 脚本需要这些工具但尚未启用：{missing_display}。 点击下方『一键补齐缺失工具』即可放行执行。"
                    ),
                }
            )

        # --- 5. Primary CTA block — confirm start / cancel / mismatch fix ---
        # Visible on the first screen. Users do NOT need to open any
        # collapsible panel to unblock execution.
        from ...card.actions.dispatch import WORKFLOW_BACK_TO_TOOLS, WORKFLOW_FILL_MISSING_TOOLS

        confirm_value = {
            "action": WORKFLOW_CONFIRM_START,
            "chat_id": chat_id,
            "project_id": project_id,
            "engine_session_key": engine_session_key,
        }
        cancel_value = {
            "action": WORKFLOW_CANCEL,
            "chat_id": chat_id,
            "project_id": project_id,
            "engine_session_key": engine_session_key,
        }

        primary_buttons: list[dict] = []

        if has_mismatch:
            fill_missing_value = {
                "action": WORKFLOW_FILL_MISSING_TOOLS,
                "chat_id": chat_id,
                "project_id": project_id,
                "engine_session_key": engine_session_key,
            }
            back_tools_value = {
                "action": WORKFLOW_BACK_TO_TOOLS,
                "chat_id": chat_id,
                "project_id": project_id,
                "engine_session_key": engine_session_key,
            }
            primary_buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "➕ 一键补齐缺失工具"},
                    "type": "primary",
                    "value": fill_missing_value,
                    "behaviors": [{"type": "callback", "value": fill_missing_value}],
                }
            )
            primary_buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "↩️ 返回工具选择"},
                    "type": "default",
                    "value": back_tools_value,
                    "behaviors": [{"type": "callback", "value": back_tools_value}],
                }
            )

        confirm_disabled = has_mismatch
        confirm_disabled_tips = "脚本需要的工具尚未全部启用，请先点击『一键补齐缺失工具』" if confirm_disabled else None
        confirm_btn: dict = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": "✅ 确认执行"},
            "type": "primary" if not confirm_disabled else "default",
            "value": confirm_value,
            "behaviors": [{"type": "callback", "value": confirm_value}],
            "disabled": confirm_disabled,
        }
        if confirm_disabled_tips:
            confirm_btn["disabled_tips"] = {
                "tag": "plain_text",
                "content": confirm_disabled_tips,
            }
        primary_buttons.append(confirm_btn)

        primary_buttons.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "❌ 取消"},
                "type": "danger",
                "value": cancel_value,
                "behaviors": [{"type": "callback", "value": cancel_value}],
                "confirm": {
                    "title": {"tag": "plain_text", "content": UI_TEXT["workflow_btn_confirm_cancel_title"]},
                    "text": {"tag": "plain_text", "content": UI_TEXT["workflow_btn_confirm_cancel_body"]},
                },
            }
        )

        elements.extend(build_responsive_button_row(primary_buttons, mobile_force_vertical=True))

        # --- 5b. Sub-workflow references (first screen, independent block) ---
        # Shown on the confirm card's first screen so users can inspect and
        # edit sub-workflow refs without opening the advanced panel.
        # The actual ``workflow('name', {})`` call is injected at confirm time
        # via ``_inject_workflow_refs_into_script`` — no patch is written to
        # the script on disk here.
        from ...card.actions.dispatch import (
            WORKFLOW_ADD_WORKFLOW_REF,
            WORKFLOW_REMOVE_WORKFLOW_REF,
            WORKFLOW_VIEW_WORKFLOW_REF,
        )

        ref_count = len(workflow_refs) if isinstance(workflow_refs, list) else 0
        elements.append(
            {
                "tag": "markdown",
                "content": f"**🔗 子 Workflow 引用（{ref_count}）**",
            }
        )

        if not workflow_refs:
            # Empty state chip so the "add" entry point is still obvious.
            elements.append(
                {
                    "tag": "markdown",
                    "content": "<font color='grey'>暂无子 Workflow 引用。点击下方按钮可添加。</font>",
                }
            )
        else:
            for idx, ref in enumerate(workflow_refs):
                if isinstance(ref, dict):
                    ref_name = ref.get("name", "unknown")
                    ref_desc = ref.get("description", "")
                else:
                    ref_name = str(ref)
                    ref_desc = ""

                chip_lines = [f"`{ref_name}`"]
                if ref_desc:
                    chip_lines.append(f"  <font color='grey' size='10'>{ref_desc[:80]}</font>")
                elements.append(
                    {
                        "tag": "markdown",
                        "content": "\n".join(chip_lines),
                    }
                )

                ref_buttons: list[dict] = []
                view_value = {
                    "action": WORKFLOW_VIEW_WORKFLOW_REF,
                    "ref_index": idx,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": engine_session_key,
                }
                ref_buttons.append(
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "👁 预览"},
                        "type": "default",
                        "value": view_value,
                        "behaviors": [{"type": "callback", "value": view_value}],
                    }
                )
                remove_value = {
                    "action": WORKFLOW_REMOVE_WORKFLOW_REF,
                    "ref_index": idx,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": engine_session_key,
                }
                ref_buttons.append(
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "🗑 移除"},
                        "type": "default",
                        "value": remove_value,
                        "behaviors": [{"type": "callback", "value": remove_value}],
                        "confirm": {
                            "title": {"tag": "plain_text", "content": "移除子 Workflow 引用？"},
                            "text": {"tag": "plain_text", "content": f"确定移除「{ref_name}」？"},
                        },
                    }
                )
                elements.extend(build_responsive_button_row(ref_buttons, mobile_force_vertical=True))

        # "Add reference" main button — primary entry point.
        add_value = {
            "action": WORKFLOW_ADD_WORKFLOW_REF,
            "chat_id": chat_id,
            "project_id": project_id,
            "engine_session_key": engine_session_key,
        }
        add_button = [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "➕ 添加子 Workflow 引用"},
                "type": "default",
                "value": add_value,
                "behaviors": [{"type": "callback", "value": add_value}],
            }
        ]
        elements.extend(build_responsive_button_row(add_button, mobile_force_vertical=True))

        # --- 6. Phases panel (top-level collapsible) ---
        if phases:
            phase_elements = []
            for i, p in enumerate(phases, 1):
                title = p.get("title", p.get("name", f"Phase {i}"))
                detail = p.get("detail", "")
                line = f"**{i}. {title}**"
                if detail:
                    line += f"\n   {detail[:100]}"
                phase_tools = phase_tool_mapping.get(title) or phase_tool_mapping.get(str(i))
                if phase_tools:
                    tool_tags = ", ".join(f"`{t}`" for t in phase_tools)
                    line += f"\n   工具: {tool_tags}"
                phase_elements.append({"tag": "markdown", "content": line})
            elements.append(
                {
                    "tag": "collapsible_panel",
                    "header": {
                        "title": {"tag": "plain_text", "content": f"📋 阶段列表 ({len(phases)})"},
                    },
                    "border": {"color": "blue", "corner_radius": "8px"},
                    "expanded": False,
                    "elements": phase_elements,
                }
            )
        else:
            elements.append(
                {
                    "tag": "markdown",
                    "content": "📋 **执行阶段**: Planning → Execution",
                }
            )

        # --- 7. Script preview panel (top-level collapsible) ---
        if script_content:
            from ...workflow_engine.renderer import render_script_preview

            preview = render_script_preview(script_content)
            if preview:
                elements.append(
                    {
                        "tag": "collapsible_panel",
                        "expanded": False,
                        "header": {
                            "title": {"tag": "plain_text", "content": "📜 编排脚本预览"},
                        },
                        "border": {"color": "grey", "corner_radius": "8px"},
                        "elements": [{"tag": "markdown", "content": preview}],
                    }
                )

        # --- 8. Collapsible: Advanced options (tools / regen) ---
        # Everything below here is truly secondary; users open this only for
        # deeper inspection before confirming.
        advanced_elements: list[dict] = []

        # 8a. Tools detail + interactive toggle
        tool_descriptions = get_available_tools(require_available=True)
        recommended_order = ["traex", "claude", "codex", "aiden", "gemini", "coco"]
        tier1_tools = [t for t in recommended_order if t in allowed_tools]
        tier2_tools = [t for t in sorted(allowed_tools) if t not in recommended_order]

        tool_detail_elements: list[dict] = []
        tool_detail_elements.append(
            {
                "tag": "markdown",
                "content": "📝 **脚本计划使用**: " + (" | ".join(f"`{t}`" for t in sorted(script_tools))),
            }
        )
        tool_detail_elements.append(
            {
                "tag": "markdown",
                "content": "✅ **允许执行的工具**（点击切换，脚本只能使用勾选的工具）：",
            }
        )

        if tier1_tools:
            for tool in tier1_tools:
                desc = tool_descriptions.get(tool, tool)
                tool_detail_elements.append({"tag": "markdown", "content": f"- `{tool}`: {desc}"})

            tool_buttons = []
            for t in tier1_tools:
                is_selected = t in allowed_tools
                btn_value = {
                    "action": WORKFLOW_SELECT_TOOL,
                    "tool_name": t,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": engine_session_key,
                }
                tool_buttons.append(
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": f"{'✓ ' if is_selected else '○ '}{t}"},
                        "type": "primary" if is_selected else "default",
                        "value": btn_value,
                        "behaviors": [{"type": "callback", "value": btn_value}],
                    }
                )
            if tool_buttons:
                tool_detail_elements.extend(build_responsive_button_row(tool_buttons, mobile_force_vertical=True))

        if tier2_tools:
            tier2_elements = []
            for tool in tier2_tools:
                desc = tool_descriptions.get(tool, tool)
                tier2_elements.append({"tag": "markdown", "content": f"- `{tool}`: {desc}"})
            other_buttons = []
            for t in tier2_tools:
                is_selected = t in allowed_tools
                btn_value = {
                    "action": WORKFLOW_SELECT_TOOL,
                    "tool_name": t,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": engine_session_key,
                }
                other_buttons.append(
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": f"{'✓ ' if is_selected else '○ '}{t}"},
                        "type": "primary" if is_selected else "default",
                        "value": btn_value,
                        "behaviors": [{"type": "callback", "value": btn_value}],
                    }
                )
            tool_detail_elements.append(
                {
                    "tag": "collapsible_panel",
                    "header": {
                        "title": {"tag": "plain_text", "content": f"🔧 更多工具 ({len(tier2_tools)})"},
                    },
                    "border": {"color": "grey", "corner_radius": "8px"},
                    "expanded": False,
                    "elements": [
                        *tier2_elements,
                        *build_responsive_button_row(other_buttons, mobile_force_vertical=True),
                    ],
                }
            )

        # 6e. Regenerate script (advanced)
        from ...card.actions.dispatch import WORKFLOW_REGENERATE_SCRIPT

        regen_elements: list[dict] = []
        regen_buttons: list[dict] = []
        regen_buttons.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "🔄 重新生成编排"},
                "type": "default",
                "value": {
                    "action": WORKFLOW_REGENERATE_SCRIPT,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": engine_session_key,
                },
                "behaviors": [
                    {
                        "type": "callback",
                        "value": {
                            "action": WORKFLOW_REGENERATE_SCRIPT,
                            "chat_id": chat_id,
                            "project_id": project_id,
                            "engine_session_key": engine_session_key,
                        },
                    }
                ],
            }
        )
        regen_elements.extend(build_responsive_button_row(regen_buttons, mobile_force_vertical=True))

        # --- Combine all advanced sections into one collapsed panel ---
        # Tools/regen are non-essential for quick confirmation. Grouping
        # them under one collapsed panel keeps the first screen focused on
        # the decision (confirm / cancel / fix).
        combined_panel_elements: list[dict] = []
        combined_panel_elements.extend(advanced_elements)
        combined_panel_elements.append({"tag": "hr"})
        combined_panel_elements.extend(tool_detail_elements)
        combined_panel_elements.append({"tag": "hr"})
        combined_panel_elements.extend(regen_elements)

        elements.append(
            {
                "tag": "collapsible_panel",
                "header": {
                    "title": {"tag": "plain_text", "content": "⚙️ 查看详细信息 / 更多操作（阶段 / 工具 / 脚本）"},
                },
                "border": {"color": "grey", "corner_radius": "8px"},
                "expanded": False,
                "elements": combined_panel_elements,
            }
        )

        return CardBuilder._wrap_card(
            header_title="🔄 Workflow 确认",
            header_template=UI_TEXT["workflow_header_colors"].get("confirm", "turquoise"),
            elements=elements,
        )

    @staticmethod
    def _parse_template_args(args_text: str) -> dict[str, Any]:
        """Parse 'key=value key2=value2' into a dict."""
        args: dict[str, Any] = {}
        for token in args_text.split():
            if "=" in token:
                key, _, value = token.partition("=")
                # Try to parse as JSON literal (number, bool, null)
                import json

                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    args[key] = value
            else:
                # Positional arg → store as "target"
                args.setdefault("target", token)
        return args

    def _build_workflow_callbacks(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"],
        *,
        lifecycle_owner: _WorkflowLifecycleOwner | None = None,
    ):
        """Build WorkflowEngineCallbacks that update the Feishu card."""
        from ...workflow_engine.engine import WorkflowEngineCallbacks

        card_message_id: list[str] = [message_id]  # Mutable ref for card updates
        terminal_sent: list[bool] = [False]
        delivery_lock = (
            lifecycle_owner.delivery_lock
            if lifecycle_owner is not None
            else threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        )
        project_id = getattr(project, "project_id", "") or ""
        origin_message_id = self._resolve_origin(message_id)

        def on_progress(card_data: dict[str, Any]) -> None:
            """Update the progress card in Feishu."""
            with delivery_lock:
                if terminal_sent[0] or (lifecycle_owner is not None and lifecycle_owner.stop_event.is_set()):
                    logger.debug("Ignored workflow progress update after terminal card was sent")
                    return
                try:
                    # Keep runtime progress cards aligned with the mixin
                    # implementation: users can stop active workflows from the card,
                    # but terminal completion cards are delivered separately.
                    self._inject_workflow_stop_button(
                        card_data,
                        chat_id,
                        project_id,
                    )
                    new_id = self._replace_or_send_workflow_rendered_card(
                        card_message_id=card_message_id[0],
                        chat_id=chat_id,
                        card_data=card_data,
                        origin_message_id=origin_message_id,
                        # Heartbeats are periodic. Turning every PATCH failure
                        # into a new message creates an unbounded card stream
                        # during persistent transport or provenance failures.
                        fallback_to_new=False,
                    )
                    if new_id:
                        card_message_id[0] = new_id
                except Exception:
                    logger.debug(
                        "Failed to update workflow progress card",
                        exc_info=True,
                    )

        def on_done(wf_project) -> None:
            """Final completion — send a structured completion card."""
            # Wait for an in-flight PATCH, then fence every later progress
            # callback before report generation/upload and terminal delivery.
            with delivery_lock:
                if terminal_sent[0] or (lifecycle_owner is not None and lifecycle_owner.stop_event.is_set()):
                    return
                terminal_sent[0] = True
                report_status: dict[str, Any] | None = None
                try:
                    from ...workflow_engine.renderer import render_completion_card

                    report_status = self._send_workflow_completion_report(
                        wf_project=wf_project,
                        chat_id=chat_id,
                        message_id=message_id,
                        project=project,
                    )
                    if lifecycle_owner is not None and lifecycle_owner.stop_event.is_set():
                        return
                    card_data = render_completion_card(
                        wf_project,
                        report_status=report_status,
                    )
                    new_id = self._replace_or_send_workflow_rendered_card(
                        card_message_id=card_message_id[0],
                        chat_id=chat_id,
                        card_data=card_data,
                        origin_message_id=origin_message_id,
                    )
                    if new_id:
                        card_message_id[0] = new_id
                    else:
                        # Card delivery returned None — try text fallback
                        raise RuntimeError("Card delivery returned None")
                except Exception:
                    if lifecycle_owner is not None and lifecycle_owner.stop_event.is_set():
                        return
                    self._reply_workflow_completion_fallback(
                        message_id=message_id,
                        report_status=report_status,
                    )

        def on_error(error_msg: str) -> None:
            """Error notification — sanitize before showing to user."""
            with delivery_lock:
                if terminal_sent[0] or (lifecycle_owner is not None and lifecycle_owner.stop_event.is_set()):
                    return
                terminal_sent[0] = True
                from ...workflow_engine.errors import (
                    ErrorCategory,
                    _strip_internal_details,
                    categorize_error,
                )

                category = categorize_error(error_msg)
                if category == ErrorCategory.TOOL_NOT_ALLOWED:
                    workflow_category = "forbidden"
                elif category == ErrorCategory.SCRIPT_VALIDATION:
                    workflow_category = "invalid_argument"
                elif category == ErrorCategory.RUNTIME_TIMEOUT:
                    workflow_category = "runtime_timeout"
                elif category in (
                    ErrorCategory.AGENT_LIMIT,
                    ErrorCategory.CANCELLED,
                ):
                    workflow_category = "invalid_state"
                else:
                    workflow_category = "internal_error"

                self._reply_workflow_error(
                    message_id,
                    workflow_category,
                    detail=_strip_internal_details(error_msg or ""),
                )

        def on_log(msg: str) -> None:
            logger.debug("[WorkflowHandler] log: %s", msg)

        return WorkflowEngineCallbacks(
            on_progress=on_progress,
            on_done=on_done,
            on_error=on_error,
            on_log=on_log,
        )
