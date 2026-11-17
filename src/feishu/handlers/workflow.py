"""Workflow Engine handler — /wf, /workflow, /stop_wf, /wf_status, /wf_help commands."""

from __future__ import annotations

import json
import logging
import threading
import time  # noqa: F401
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from ...card.text_stream import append_stream_text
from ...utils.text import generate_task_id  # noqa: F401
from ...workflow_engine.storage import workflow_scripts_dir  # noqa: F401
from ..emoji import EmojiReaction
from ..slash_command_parser import SlashCommandParser
from .engine_base import BaseEngineHandler

if TYPE_CHECKING:
    from ...project import ProjectContext
    from ..handler_context import HandlerContext

logger = logging.getLogger(__name__)

_SCRIPT_GENERATION_MAX_ATTEMPTS = 3
_GENERATION_ACTIVITY_MAX_CHARS = 180
_GENERATION_ACTIVITY_BREAKS = frozenset("。！？.!?；;\n")
_WORKFLOW_STOP_QUIESCENCE_TIMEOUT_S = 5.0
_WORKFLOW_COT_REQUEST_TIMEOUT_S = 5.0
_WORKFLOW_COT_CLOSE_TIMEOUT_S = 2.5


@dataclass(slots=True)
class _WorkflowGenerationActivityBuffer:
    """Coalesce provider text tokens into readable generation activities."""

    pending: str = ""

    def feed(self, text: str) -> tuple[str, ...]:
        if not text:
            return ()
        self.pending = append_stream_text(self.pending, text)
        completed: list[str] = []
        while self.pending:
            boundary = self._first_boundary(self.pending)
            if boundary < 0 and len(self.pending) < _GENERATION_ACTIVITY_MAX_CHARS:
                break
            end = (
                boundary + 1
                if boundary >= 0
                else _GENERATION_ACTIVITY_MAX_CHARS
            )
            segment = self.pending[:end].strip()
            self.pending = self.pending[end:].lstrip()
            if segment:
                completed.append(segment)
        return tuple(completed)

    def flush(self) -> tuple[str, ...]:
        segment = self.pending.strip()
        self.pending = ""
        return (segment,) if segment else ()

    @staticmethod
    def _first_boundary(value: str) -> int:
        return next(
            (
                index
                for index, char in enumerate(value)
                if char in _GENERATION_ACTIVITY_BREAKS
            ),
            -1,
        )


@dataclass(frozen=True, slots=True)
class _WorkflowLifecycleOwner:
    """Linearization token shared by generation, card delivery, and queued start."""

    session_key: str
    initiator_user_id: str = ""
    chat_id: str = ""
    project_id: str = ""
    root_path: str = ""
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
    worker_exited_event: threading.Event = field(
        default_factory=threading.Event,
    )
    stop_delivery_fenced_event: threading.Event = field(
        default_factory=threading.Event,
    )
    worker_thread_id: int | None = field(
        default=None,
        compare=False,
    )
    active_generation_session: Any = field(
        default=None,
        compare=False,
    )
    card_message_id: str | None = field(
        default=None,
        compare=False,
    )


class _WorkflowGenerationCancelled(RuntimeError):
    """Internal control flow: a superseded generator must not write fallback."""


class _WorkflowGenerationExhausted(RuntimeError):
    """All model paths and the deterministic pool fallback were exhausted."""


def _workflow_pending_statuses():
    """States that own a pending Workflow card/session rather than a runtime run."""
    from ...workflow_engine.models import WorkflowStatus

    return {
        WorkflowStatus.GENERATING_SCRIPT,
    }


from .workflow_execution import WorkflowExecutionMixin  # noqa: E402
from .workflow_script import WorkflowScriptMixin  # noqa: E402


class WorkflowHandler(
    WorkflowExecutionMixin,
    WorkflowScriptMixin,
    BaseEngineHandler,
):
    """Manages the full lifecycle of Workflow Engine tasks.

    Commands:
        /wf <requirement>       — Start a new workflow (AI generates script)
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
        match = SlashCommandParser.parse(text)
        command = match.command if match else ""
        args = match.args if match else ""

        if command == "/stop_wf":
            self.stop_workflow(message_id, chat_id, project)
        elif command == "/wf_status":
            self.show_workflow_status(message_id, chat_id, project)
        elif command == "/wf_help":
            self.show_workflow_help(message_id)
        elif command == "/wf":
            if args:
                self.start_workflow(message_id, chat_id, args, project)
            else:
                self.show_workflow_help(message_id)
        else:
            self._reply_workflow_error(
                message_id,
                "invalid_argument",
                detail=(
                    "未知命令。可用命令列表:\n"
                    "• `/wf <需求>` — 基于需求生成编排脚本并执行\n"
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


    def _get_task_type(self) -> str:
        return "workflow_engine"

    def _show_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        self.show_workflow_status(message_id, chat_id, project)

    def _create_callbacks(
        self, message_id: str, chat_id: str, project: Optional["ProjectContext"], engine_name: str, root_path: str
    ):
        return self._build_workflow_callbacks(message_id, chat_id, project)

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
        """Start a workflow and use the default auto policy to launch generation.

        The recommended available tool and its backend-default model are used;
        generation, validation, and execution proceed without a decision gate.
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
        # admission/generation/queued start; another user's task is protected.
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

        self._launch_auto_bound_workflow(
            message_id=message_id,
            chat_id=chat_id,
            requirement=requirement,
            project=project,
            root_path=root_path,
            engine=admission_engine,
            admission_owner=admission_owner,
        )

    def handle_workflow_agent_action(
        self,
        message_id: str,
        chat_id: str,
        project_id: str | None,
        value: dict[str, Any],
    ) -> None:
        """Reject legacy Agent Pool selection card callbacks cleanly."""
        del chat_id, project_id, value
        self._reply_workflow_error(
            message_id,
            "session_expired",
            detail="旧版 Agent 选择卡已停用，请直接发送 `/wf <需求>` 自动启动。",
        )

    def _launch_auto_bound_workflow(
        self,
        *,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: "ProjectContext" | None,
        root_path: str,
        engine: Any,
        admission_owner: _WorkflowLifecycleOwner | None,
    ) -> None:
        """Auto-bind available recommended tools as the frozen pool and launch."""
        from ...spec_engine.models import ReviewAgentBinding
        from ...thread import get_current_sender_id
        from ...workflow_engine.agent_pool import (
            WorkflowAgentBinding,
            freeze_agent_pool,
            select_auto_orchestrator,
        )
        from ...workflow_engine.models import (
            PendingWorkflow,
            WorkflowProject,
            WorkflowStatus,
        )
        from ...workflow_engine.tool_registry import get_available_tools

        available_tools = get_available_tools(require_available=True)
        preferred_order = (
            "traex",
            "claude",
            "codex",
            "grok",
            "dsh",
            "agy",
            "aiden",
            "gemini",
            "coco",
        )
        selected_names = [t for t in preferred_order if t in available_tools][:3]
        if not selected_names:
            selected_names = list(available_tools)[:3]
        if not selected_names:
            if admission_owner is not None:
                admission_owner.done_event.set()
                release_owner = getattr(engine, "release_lifecycle_owner", None)
                if callable(release_owner):
                    release_owner(admission_owner)
            self._reply_workflow_error(
                message_id,
                "invalid_state",
                detail="当前环境未检测到可执行的 Workflow 工具。",
            )
            return

        agent_pool = freeze_agent_pool(
            WorkflowAgentBinding(
                agent_id=f"A{idx}",
                tool_name=name,
                model_name=None,
                display_name=str(available_tools[name] or name),
            )
            for idx, name in enumerate(selected_names, start=1)
        )
        orchestrator_binding = select_auto_orchestrator(agent_pool)
        initiator_user_id = get_current_sender_id() or ""
        workflow_project_id = uuid.uuid4().hex
        selected_tools = [binding.tool_name for binding in agent_pool]

        with engine._lock:
            wf_project = engine.project
            if not wf_project:
                wf_project = WorkflowProject()
                engine._project = wf_project
                if getattr(engine, "project", None) is None:
                    engine.project = wf_project
            if admission_owner is not None:
                admission_owner.stop_event.set()
                admission_owner.heartbeat_stop_event.set()
                admission_owner.done_event.set()
                self._retire_workflow_owner(engine, admission_owner)
            session_key = admission_owner.session_key if admission_owner is not None else uuid.uuid4().hex
            generation_owner = _WorkflowLifecycleOwner(
                session_key=session_key,
                initiator_user_id=initiator_user_id,
                chat_id=chat_id,
                project_id=workflow_project_id,
                root_path=root_path,
            )
            engine._script_generation_owner = generation_owner
            wf_project.pending = PendingWorkflow(
                requirement=requirement,
                initiator_user_id=initiator_user_id,
                engine_session_key=session_key,
                selected_tools=selected_tools,
                orchestrator_agent=orchestrator_binding.tool_name,
                orchestrator_binding=ReviewAgentBinding(
                    provider="workflow",
                    tool_name=orchestrator_binding.tool_name,
                    display_name=orchestrator_binding.display_name,
                    agent_type=orchestrator_binding.tool_name,
                    model_name=orchestrator_binding.model_name,
                    model_display_name=orchestrator_binding.model_name,
                    use_default_model=(orchestrator_binding.model_name is None),
                ),
                review_agents=[],
                auto_reviewer=True,
                agent_pool=agent_pool,
                orchestrator_agent_id=orchestrator_binding.agent_id,
                orchestrator_was_auto=True,
                project_id=workflow_project_id,
            )
            wf_project.status = WorkflowStatus.GENERATING_SCRIPT

        if admission_owner is not None:
            release_owner = getattr(engine, "release_lifecycle_owner", None)
            if callable(release_owner):
                release_owner(admission_owner)

        try:
            origin_message_id = self._resolve_origin(message_id)
        except Exception:
            origin_message_id = message_id

        generation_card_id: str | None = None
        try:
            card_data = self._workflow_generation_card_data(
                requirement=requirement,
                agent_pool=agent_pool,
                orchestrator_agent_id=orchestrator_binding.agent_id,
                orchestrator_was_auto=True,
            )
            with generation_owner.delivery_lock:
                generation_card_id = self._replace_or_send_workflow_rendered_card(
                    card_message_id=None,
                    chat_id=chat_id,
                    card_data=card_data,
                    origin_message_id=origin_message_id,
                )
                if generation_card_id:
                    object.__setattr__(generation_owner, "card_message_id", generation_card_id)
        except Exception:
            logger.debug("Failed to deliver initial workflow generation card", exc_info=True)

        self._schedule_generate_and_start_workflow(
            message_id=generation_card_id or message_id,
            chat_id=chat_id,
            requirement=requirement,
            project=project,
            root_path=root_path,
            selected_tools=selected_tools,
            expected_session_key=session_key,
            engine=engine,
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
        from ...card.builders.core import CoreBuilder
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
            if safe_detail and category != "internal_error":
                body = raw_body.rstrip() + "\n\n🔎 细节：" + safe_detail
            else:
                body = raw_body
        else:
            body = safe_detail or "操作失败，请稍后重试。"

        # Header color by category
        header_template = "red"  # default for errors

        return CoreBuilder._wrap_card(
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

        summary_content = " ".join(title_content.split()) or "Workflow 运行状态"
        if len(summary_content) < 8:
            summary_content = f"Workflow · {summary_content}"
        if len(summary_content) > 60:
            summary_content = f"{summary_content[:59]}…"
        card = {
            "schema": "2.0",
            "config": {
                "compact_width": False,
                "wide_screen_mode": True,
                "summary": {"content": summary_content},
            },
            "header": header,
            "body": {"elements": elements},
        }
        from ...card.delivery.page_mutator import guard_card_payload

        return guard_card_payload(
            json.loads(
                json.dumps(
                    card,
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8", errors="replace").decode("utf-8")
            )
        )

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
            return None
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
        """Switch the generation card to a running progress card immediately."""
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
    def _workflow_generation_card_data(
        *,
        requirement: str,
        agent_pool: Any,
        orchestrator_agent_id: str,
        orchestrator_was_auto: bool,
        current_activity: str | None = None,
        elapsed_seconds: int = 0,
        terminal_status: str | None = None,
    ) -> dict[str, Any]:
        from ...workflow_engine.renderer import WorkflowGenerationRenderer

        return WorkflowGenerationRenderer(
            requirement=requirement,
            agent_pool=agent_pool,
            orchestrator_agent_id=orchestrator_agent_id,
            orchestrator_was_auto=orchestrator_was_auto,
        ).render(
            current_activity=current_activity,
            elapsed_seconds=elapsed_seconds,
            terminal_status=terminal_status,
        )


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
        card_pages = engine.get_progress_cards()

        if card_pages:
            origin_message_id = self._resolve_origin(message_id)
            failed_pages: list[int] = []
            for page_index, card_data in enumerate(card_pages):
                delivered = self._replace_or_send_workflow_rendered_card(
                    card_message_id=None,
                    chat_id=chat_id,
                    card_data=card_data,
                    origin_message_id=origin_message_id,
                )
                if not delivered:
                    failed_pages.append(page_index)
            if failed_pages:
                pages = ", ".join(str(index + 1) for index in failed_pages)
                self.reply_text(
                    message_id,
                    f"Workflow 结果页重投仍不完整（失败页：{pages}），可再次发送 `/wf_status` 重试。",
                )
        else:
            self.reply_text(message_id, status_text)

    def show_workflow_help(self, message_id: str) -> None:
        """Show the Workflow usage guide as a structured Feishu card."""
        from ...card.builders.core import CoreBuilder
        from ...card.themes import PANEL_STYLES

        # Pull the authoritative Node-version text so the full help keeps the
        # same contract as engine.run_workflow() and the card-entry messages.
        from ...workflow_engine.engine import _node_version_required_text

        sections = [
            (
                "📚 命令列表",
                "`/wf <需求描述>`（`/workflow`） · 自动绑定推荐 Agent Pool 并启动编排\n"
                "`/wf_status`（`/workflow_status`） · 查看当前 Workflow 进度\n"
                "`/wf_help`（`/workflow_help`） · 显示本帮助\n"
                "`/stop_wf`（`/stop_workflow`） · 停止正在执行的 Workflow",
            ),
            (
                "🧭 执行流程",
                "**① 自动绑定** · 自动选取可用推荐工具构建 1–8 个 `tool+model` Agent Pool 并锁定主编排器\n"
                "**② 自动推进** · 绑定后全自动生成、验证并执行脚本，无需手动确认或批准继续",
            ),
            (
                "✨ 核心能力",
                "• 多工具并行调度（coco / claude / aiden / codex / gemini / traex / grok / dsh / agy）\n"
                "• 工具 Agent 可继续拆分 subagent 并行工作\n"
                "• Agent 按任务动态规划角色分工\n"
                "• Journal 缓存避免重复执行\n"
                "• 子任务自动拆分与依赖编排\n"
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

        card = CoreBuilder._wrap_card(
            header_title="⚡ Workflow 使用帮助",
            header_template="turquoise",
            elements=elements,
        )
        self.reply_card(message_id, card)






    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_root_path(self, chat_id: str, project: Optional["ProjectContext"]) -> str:
        """Resolve root_path from project or chat."""
        if project:
            return project.root_path
        return self.get_working_dir(chat_id)
