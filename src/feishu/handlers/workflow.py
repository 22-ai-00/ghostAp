"""Workflow Engine handler — /wf, /workflow, /stop_wf, /wf_status, /wf_help commands."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from ...utils.text import generate_task_id
from ..emoji import EmojiReaction
from .engine_base import BaseEngineHandler

if TYPE_CHECKING:
    from ...project import ProjectContext
    from ...workflow_engine.models import WorkflowProject
    from ..handler_context import HandlerContext

logger = logging.getLogger(__name__)

_SCRIPT_GENERATION_MAX_ATTEMPTS = 3


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


class _WorkflowGenerationCloseUncertain(RuntimeError):
    """The prior binding may still be live, so fallback is unsafe."""


def _workflow_pending_statuses():
    """States that own a pending Workflow card/session rather than a runtime run."""
    from ...workflow_engine.models import WorkflowStatus

    return {
        WorkflowStatus.SELECTING_AGENTS,
        WorkflowStatus.GENERATING_SCRIPT,
    }


from .workflow_script import WorkflowScriptMixin  # noqa: E402


class WorkflowHandler(WorkflowScriptMixin, BaseEngineHandler):
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
        from ...utils.command_parser import CommandParser

        cmd = CommandParser.parse_basic(text)
        command = cmd.command

        if command == "/stop_wf":
            self.stop_workflow(message_id, chat_id, project)
        elif command == "/wf_status":
            self.show_workflow_status(message_id, chat_id, project)
        elif command == "/wf_help":
            self.show_workflow_help(message_id)
        elif command == "/wf":
            arg = cmd.args
            if arg:
                self.start_workflow(message_id, chat_id, arg, project)
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

        self._start_workflow_agent_selection(
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
            if safe_detail:
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

        card = {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": header,
            "body": {"elements": elements},
        }
        from ...card.delivery.page_mutator import guard_card_payload

        return guard_card_payload(card)

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
    def _workflow_selected_option(value: dict[str, Any]) -> str | None:
        option = value.get("_option")
        if isinstance(option, dict):
            option = option.get("value")
        selected = str(option or "").strip()
        return selected or None

    @staticmethod
    def _workflow_selection_model_state(
        *,
        tool_name: str,
        root_path: str,
        pending: Any,
        selected_model: str | None = None,
        selected_profile: str | None = None,
        selected_effort: str | None = None,
    ) -> tuple[list[Any], Any]:
        from ...acp.helper import fetch_acp_models
        from ...card.render.model_cascade import resolve_model_cascade

        models = fetch_acp_models(
            tool_name,
            cwd=root_path,
            current_model=None,
        )
        state = resolve_model_cascade(
            models,
            selected_model=(
                pending.draft_model_name
                if selected_model is None
                else selected_model
            ),
            selected_profile=(
                pending.draft_profile
                if selected_profile is None
                else selected_profile
            ),
            selected_effort=(
                pending.draft_effort
                if selected_effort is None
                else selected_effort
            ),
        )
        return models, state

    @classmethod
    def _validated_workflow_draft_binding(
        cls,
        *,
        pending: Any,
        root_path: str,
        available_tools: dict[str, str],
    ) -> tuple[str | None, str | None, str | None]:
        from ...card.render.model_cascade import (
            available_model_efforts,
            available_model_names,
            available_model_profiles,
            compose_model_selection,
        )

        tool_name = str(pending.draft_tool_name or "").strip().lower()
        if tool_name not in available_tools:
            raise ValueError("所选工具当前不可用，请重新选择。")
        if not pending.draft_model_name:
            if pending.draft_profile or pending.draft_effort:
                raise ValueError("Backend default 模型不能携带 Profile 或 Effort。")
            return None, None, None

        models, _state = cls._workflow_selection_model_state(
            tool_name=tool_name,
            root_path=root_path,
            pending=pending,
        )
        model_name = str(pending.draft_model_name).strip()
        names = available_model_names(models)
        if model_name not in names:
            raise ValueError("所选模型已不可用，请重新选择。")
        profiles = available_model_profiles(models, model_name)
        profile = str(pending.draft_profile or "").strip().lower() or None
        if profiles and profile not in profiles:
            raise ValueError("所选 Profile 不属于当前模型能力。")
        if not profiles and profile is not None:
            raise ValueError("当前模型不支持 Profile。")
        efforts = available_model_efforts(models, model_name, profile)
        effort = str(pending.draft_effort or "").strip().lower() or None
        if efforts and effort not in efforts:
            raise ValueError("所选 Effort 不属于当前模型能力。")
        if not efforts and effort is not None:
            raise ValueError("当前模型不支持 Effort。")
        selection = compose_model_selection(
            models,
            model=model_name,
            profile=profile,
            effort=effort,
        )
        if selection is None:
            raise ValueError("所选模型组合不是后端声明的有效能力。")
        return selection, profile, effort

    @classmethod
    def _validate_workflow_pool_capabilities(
        cls,
        *,
        pending: Any,
        root_path: str,
        available_tools: dict[str, str],
    ) -> None:
        from ...acp.helper import fetch_acp_models
        from ...card.render.model_cascade import parse_model_selection

        for binding in pending.agent_pool:
            if binding.tool_name not in available_tools:
                raise ValueError(f"{binding.agent_id} 的工具已不可用。")
            if binding.model_name is None:
                if binding.profile or binding.effort:
                    raise ValueError(f"{binding.agent_id} 的默认模型组合无效。")
                continue
            models = fetch_acp_models(
                binding.tool_name,
                cwd=root_path,
                current_model=None,
            )
            selection = parse_model_selection(models, binding.model_name)
            if selection is None:
                raise ValueError(f"{binding.agent_id} 的模型已不可用。")
            if selection.profile != binding.profile or selection.effort != binding.effort:
                raise ValueError(f"{binding.agent_id} 的模型能力组合已变更。")

    @classmethod
    def _workflow_selection_card_data(
        cls,
        *,
        pending: Any,
        project_id: str,
        root_path: str,
        available_tools: dict[str, str],
    ) -> dict[str, Any]:
        from ...workflow_engine.renderer import WorkflowAgentSelectionRenderer

        model_state = None
        tool_name = str(pending.draft_tool_name or "").strip().lower()
        if tool_name in available_tools:
            _models, model_state = cls._workflow_selection_model_state(
                tool_name=tool_name,
                root_path=root_path,
                pending=pending,
            )
        return WorkflowAgentSelectionRenderer(
            pending,
            project_id=project_id,
            tool_options=available_tools,
            model_state=model_state,
        ).render()

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

    def _start_workflow_agent_selection(
        self,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: "ProjectContext" | None,
        root_path: str,
        admission_owner: _WorkflowLifecycleOwner | None,
    ) -> None:
        """Publish an owner-bound Agent Pool card without opening an ACP session."""
        from ...thread import get_current_sender_id
        from ...workflow_engine.models import PendingWorkflow, WorkflowProject, WorkflowStatus
        from ...workflow_engine.tool_registry import get_available_tools

        available_tools = get_available_tools(require_available=True)
        if not available_tools:
            self._reply_workflow_error(
                message_id,
                "invalid_state",
                detail="当前环境未检测到可执行的 Workflow 工具。",
            )
            return

        engine = self.ctx.workflow_engine_manager.get_or_create(
            chat_id,
            root_path,
            engine_name=self.get_engine_name(
                chat_id,
                project_id=(project.project_id if project else None),
            ),
        )
        project_id = getattr(project, "project_id", "") or ""
        recommendation_order = ("traex", "claude", "codex", "aiden", "gemini", "coco")
        recommendation_names = [
            name for name in recommendation_order if name in available_tools
        ] or list(available_tools)[:3]
        recommendations = [
            {
                "tool_name": name,
                "model_name": None,
                "display_name": str(available_tools[name] or name),
            }
            for name in recommendation_names
        ]

        with engine._lock:
            if not engine.project:
                engine._project = WorkflowProject()
            if admission_owner is None:
                admission_owner = _WorkflowLifecycleOwner(
                    session_key=uuid.uuid4().hex,
                    initiator_user_id=get_current_sender_id() or "",
                    chat_id=chat_id,
                    project_id=project_id,
                    root_path=root_path,
                )
                engine._workflow_selection_owner = admission_owner
            elif (
                vars(engine).get("_workflow_selection_owner") is not admission_owner
                or admission_owner.stop_event.is_set()
            ):
                return
            else:
                object.__setattr__(admission_owner, "chat_id", chat_id)
                object.__setattr__(admission_owner, "project_id", project_id)
                object.__setattr__(admission_owner, "root_path", root_path)

            pending = PendingWorkflow(
                requirement=requirement,
                initiator_user_id=(
                    admission_owner.initiator_user_id or get_current_sender_id() or ""
                ),
                selection_session_key=admission_owner.session_key,
                project_id=project_id,
                recommended_agents=recommendations,
                draft_tool_name=recommendation_names[0],
            )
            engine.project.pending = pending
            engine.project.status = WorkflowStatus.SELECTING_AGENTS
            engine.project.requirement = requirement
            engine.project.error = None
            engine.project.finished_at = None

        card_data = self._workflow_selection_card_data(
            pending=pending,
            project_id=project_id,
            root_path=root_path,
            available_tools=available_tools,
        )
        selection_card_id = self.send_card_to_chat(
            chat_id,
            self._build_workflow_card_from_renderer_data(card_data),
            origin_message_id=message_id,
        )
        if selection_card_id:
            with engine._lock:
                if (
                    vars(engine).get("_workflow_selection_owner") is admission_owner
                    and engine.project is not None
                    and engine.project.status == WorkflowStatus.SELECTING_AGENTS
                ):
                    object.__setattr__(
                        admission_owner,
                        "card_message_id",
                        selection_card_id,
                    )

    def handle_workflow_agent_action(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Apply one Agent Pool selection action with owner/session/project CAS."""
        from ...spec_engine.models import ReviewAgentBinding
        from ...thread import get_current_sender_id
        from ...workflow_engine.agent_pool import (
            WorkflowAgentBinding,
            select_auto_orchestrator,
        )
        from ...workflow_engine.constants import MAX_WORKFLOW_AGENT_POOL_SIZE
        from ...workflow_engine.models import WorkflowStatus
        from ...workflow_engine.tool_registry import get_available_tools

        payload_project_id = str(value.get("project_id") or project_id or "")
        if not project_id or payload_project_id != project_id:
            self._reply_workflow_error(message_id, "session_expired")
            return
        project = self._resolve_project_from_id(project_id, chat_id)
        if project is None:
            self._reply_workflow_error(message_id, "session_expired")
            return
        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)
        if engine is None:
            self._reply_workflow_error(message_id, "session_expired")
            return

        action = str(value.get("action") or "")
        selected_option = self._workflow_selected_option(value)
        session_key = str(value.get("selection_session_key") or "")
        current_user = get_current_sender_id() or ""
        available_tools = get_available_tools(require_available=True)
        error: tuple[str, str] | None = None
        generation: tuple[Any, _WorkflowLifecycleOwner, str] | None = None
        pending = None
        owner_valid = False
        authorized = False

        with engine._lock:
            owner = vars(engine).get("_workflow_selection_owner")
            wf_project = engine.project
            pending = wf_project.pending if wf_project else None
            owner_valid = bool(
                isinstance(owner, _WorkflowLifecycleOwner)
                and not owner.stop_event.is_set()
                and owner.session_key == session_key
                and owner.chat_id == chat_id
                and owner.project_id == project_id
                and owner.root_path == root_path
                and wf_project is not None
                and wf_project.status == WorkflowStatus.SELECTING_AGENTS
                and pending is not None
                and pending.selection_session_key == session_key
                and pending.project_id == project_id
            )
            if not owner_valid:
                error = ("session_expired", "该 Agent 选择卡已失效，请重新发送 `/wf`。")
            elif not current_user or current_user != (pending.initiator_user_id or ""):
                error = ("forbidden", "只有本次 Workflow 发起者可以修改 Agent Pool。")
            else:
                authorized = True
                pending.selection_error = None

            if authorized and action == "workflow_select_tool":
                tool_name = str(selected_option or "").lower()
                if tool_name not in available_tools:
                    error = ("invalid_argument", "所选工具当前不可用。")
                else:
                    pending.draft_tool_name = tool_name
                    pending.draft_model_name = None
                    pending.draft_profile = None
                    pending.draft_effort = None
            elif authorized and action == "workflow_select_model":
                model_name = str(selected_option or "").strip()
                if model_name == "default":
                    pending.draft_model_name = None
                    pending.draft_profile = None
                    pending.draft_effort = None
                else:
                    from ...card.render.model_cascade import available_model_names

                    _models, state = self._workflow_selection_model_state(
                        tool_name=pending.draft_tool_name or "",
                        root_path=root_path,
                        pending=pending,
                        selected_model=model_name,
                        selected_profile="",
                        selected_effort="",
                    )
                    if model_name not in available_model_names(_models):
                        error = ("invalid_argument", "所选模型当前不可用。")
                    else:
                        pending.draft_model_name = state.selected_model
                        pending.draft_profile = state.selected_profile
                        pending.draft_effort = state.selected_effort
            elif authorized and action == "workflow_select_profile":
                profile = str(selected_option or "").strip().lower()
                _models, state = self._workflow_selection_model_state(
                    tool_name=pending.draft_tool_name or "",
                    root_path=root_path,
                    pending=pending,
                    selected_profile=profile,
                    selected_effort="",
                )
                if not pending.draft_model_name or profile not in state.profiles:
                    error = ("invalid_argument", "所选 Profile 不属于当前模型能力。")
                else:
                    pending.draft_profile = state.selected_profile
                    pending.draft_effort = state.selected_effort
            elif authorized and action == "workflow_select_effort":
                effort = str(selected_option or "").strip().lower()
                _models, state = self._workflow_selection_model_state(
                    tool_name=pending.draft_tool_name or "",
                    root_path=root_path,
                    pending=pending,
                    selected_effort=effort,
                )
                if not pending.draft_model_name or effort not in state.efforts:
                    error = ("invalid_argument", "所选 Effort 不属于当前模型能力。")
                else:
                    pending.draft_effort = state.selected_effort
            elif authorized and action == "workflow_add_agent":
                tool_name = str(pending.draft_tool_name or "").lower()
                if tool_name not in available_tools:
                    error = ("invalid_argument", "请先选择当前可用的工具。")
                elif len(pending.agent_pool) >= MAX_WORKFLOW_AGENT_POOL_SIZE:
                    error = (
                        "invalid_argument",
                        f"并发 Agent Pool 最多允许 {MAX_WORKFLOW_AGENT_POOL_SIZE} 个 Agent。",
                    )
                else:
                    try:
                        model_name, profile, effort = self._validated_workflow_draft_binding(
                            pending=pending,
                            root_path=root_path,
                            available_tools=available_tools,
                        )
                    except ValueError as exc:
                        error = ("invalid_argument", str(exc))
                    else:
                        if any(
                            binding.tool_name == tool_name
                            and binding.model_name == model_name
                            for binding in pending.agent_pool
                        ):
                            error = (
                                "invalid_argument",
                                "相同工具和模型的 Agent 已在并发池中。",
                            )
                        else:
                            binding = WorkflowAgentBinding(
                                agent_id=f"A{pending.next_agent_sequence}",
                                tool_name=tool_name,
                                model_name=model_name,
                                display_name=str(available_tools[tool_name] or tool_name),
                                profile=profile,
                                effort=effort,
                            )
                            pending.agent_pool = (*pending.agent_pool, binding)
                            pending.next_agent_sequence += 1
            elif authorized and action == "workflow_add_recommended_pool":
                pool = list(pending.agent_pool)
                sequence = pending.next_agent_sequence
                seen = {binding.tool_model_key for binding in pool}
                for recommendation in pending.recommended_agents:
                    if len(pool) >= MAX_WORKFLOW_AGENT_POOL_SIZE:
                        break
                    tool_name = str(recommendation.get("tool_name") or "").lower()
                    model_name = recommendation.get("model_name") or None
                    if tool_name not in available_tools or (tool_name, model_name) in seen:
                        continue
                    profile = None
                    effort = None
                    if model_name is not None:
                        from ...acp.helper import fetch_acp_models
                        from ...card.render.model_cascade import parse_model_selection

                        selection = parse_model_selection(
                            fetch_acp_models(tool_name, cwd=root_path, current_model=None),
                            model_name,
                        )
                        if selection is None:
                            continue
                        profile = selection.profile
                        effort = selection.effort
                    pool.append(
                        WorkflowAgentBinding(
                            agent_id=f"A{sequence}",
                            tool_name=tool_name,
                            model_name=model_name,
                            display_name=str(
                                recommendation.get("display_name")
                                or available_tools[tool_name]
                                or tool_name
                            ),
                            profile=profile,
                            effort=effort,
                        )
                    )
                    seen.add((tool_name, model_name))
                    sequence += 1
                pending.agent_pool = tuple(pool)
                pending.next_agent_sequence = sequence
            elif authorized and action == "workflow_remove_agent":
                agent_id = str(value.get("agent_id") or "")
                if not any(binding.agent_id == agent_id for binding in pending.agent_pool):
                    error = ("invalid_argument", "目标 Agent 不在当前并发池中。")
                else:
                    pending.agent_pool = tuple(
                        binding for binding in pending.agent_pool if binding.agent_id != agent_id
                    )
                    if pending.orchestrator_agent_id == agent_id:
                        pending.orchestrator_agent_id = None
            elif authorized and action == "workflow_clear_agents":
                pending.agent_pool = ()
                pending.orchestrator_agent_id = None
            elif authorized and action == "workflow_set_orchestrator":
                agent_id = str(value.get("agent_id") or "")
                if agent_id.lower() == "auto":
                    pending.orchestrator_agent_id = None
                    pending.orchestrator_was_auto = True
                elif not any(binding.agent_id == agent_id for binding in pending.agent_pool):
                    error = ("invalid_argument", "主编排 Agent 必须来自当前并发池。")
                else:
                    pending.orchestrator_agent_id = agent_id
                    pending.orchestrator_was_auto = False
            elif authorized and action == "workflow_confirm_agents":
                if not pending.agent_pool:
                    error = ("invalid_argument", "请至少添加一个 Agent 后再执行。")
                else:
                    try:
                        self._validate_workflow_pool_capabilities(
                            pending=pending,
                            root_path=root_path,
                            available_tools=available_tools,
                        )
                    except ValueError as exc:
                        error = ("invalid_argument", str(exc))
                if error is None:
                    was_auto = pending.orchestrator_agent_id is None
                    orchestrator = (
                        select_auto_orchestrator(
                            pending.agent_pool,
                            recommendations=pending.recommended_agents or (),
                        )
                        if was_auto
                        else next(
                            binding
                            for binding in pending.agent_pool
                            if binding.agent_id == pending.orchestrator_agent_id
                        )
                    )
                    resolved_id = orchestrator.agent_id
                    generation_key = uuid.uuid4().hex
                    generation_owner = _WorkflowLifecycleOwner(
                        session_key=generation_key,
                        initiator_user_id=pending.initiator_user_id or "",
                        chat_id=chat_id,
                        project_id=project_id,
                        root_path=root_path,
                        source_script_path=self._new_workflow_script_path(root_path),
                    )
                    pending.engine_session_key = generation_key
                    pending.orchestrator_agent_id = resolved_id
                    pending.orchestrator_was_auto = was_auto
                    pending.orchestrator_agent = orchestrator.tool_name
                    pending.orchestrator_binding = ReviewAgentBinding(
                        provider="workflow",
                        tool_name=orchestrator.tool_name,
                        display_name=orchestrator.display_name,
                        agent_type=orchestrator.tool_name,
                        model_name=orchestrator.model_name,
                        model_display_name=orchestrator.model_name,
                        use_default_model=orchestrator.model_name is None,
                    )
                    pending.selected_tools = list(
                        dict.fromkeys(binding.tool_name for binding in pending.agent_pool)
                    )
                    wf_project.status = WorkflowStatus.GENERATING_SCRIPT
                    wf_project.error = None
                    wf_project.finished_at = None
                    owner.stop_event.set()
                    owner.heartbeat_stop_event.set()
                    owner.done_event.set()
                    self._retire_workflow_owner(engine, owner)
                    engine._workflow_selection_owner = None
                    engine._script_generation_owner = generation_owner
                    generation = (engine, generation_owner, generation_key)
            elif authorized:
                error = ("invalid_argument", "未知的 Workflow Agent 操作。")

        if error is not None:
            if authorized and pending is not None:
                pending.selection_error = error[1]
                card_data = self._workflow_selection_card_data(
                    pending=pending,
                    project_id=project_id,
                    root_path=root_path,
                    available_tools=available_tools,
                )
                self._replace_or_send_workflow_rendered_card(
                    card_message_id=message_id,
                    chat_id=chat_id,
                    card_data=card_data,
                    fallback_to_new=False,
                )
            else:
                self._reply_workflow_error(message_id, error[0], detail=error[1])
            return
        if generation is not None:
            _, generation_owner, generation_key = generation
            release_owner = getattr(engine, "release_lifecycle_owner", None)
            if callable(release_owner):
                release_owner(owner)
            generation_message_id = message_id
            with generation_owner.delivery_lock:
                if not self._generation_owner_is_active(engine, generation_owner):
                    return
                generation_card_data = self._workflow_generation_card_data(
                    requirement=pending.requirement or "",
                    agent_pool=pending.agent_pool,
                    orchestrator_agent_id=pending.orchestrator_agent_id or "",
                    orchestrator_was_auto=pending.orchestrator_was_auto,
                )
                generation_message_id = (
                    self._replace_or_send_workflow_rendered_card(
                        card_message_id=message_id,
                        chat_id=chat_id,
                        card_data=generation_card_data,
                        origin_message_id=self._resolve_origin(message_id),
                    )
                    or message_id
                )
                object.__setattr__(
                    generation_owner,
                    "card_message_id",
                    generation_message_id,
                )
            self._schedule_generate_and_start_workflow(
                message_id=generation_message_id,
                chat_id=chat_id,
                requirement=pending.requirement or "",
                project=project,
                root_path=root_path,
                selected_tools=list(pending.selected_tools or []),
                expected_session_key=generation_key,
                engine=engine,
            )
            return

        card_data = self._workflow_selection_card_data(
            pending=pending,
            project_id=project_id,
            root_path=root_path,
            available_tools=available_tools,
        )
        self._replace_or_send_workflow_rendered_card(
            card_message_id=message_id,
            chat_id=chat_id,
            card_data=card_data,
            fallback_to_new=False,
        )

    def _start_workflow_with_defaults(
        self,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: "ProjectContext" | None,
        root_path: str,
        admission_owner: _WorkflowLifecycleOwner | None,
    ) -> None:
        """Bind recommended defaults and immediately generate and execute."""
        from ...spec_engine.models import ReviewAgentBinding
        from ...thread import get_current_sender_id
        from ...workflow_engine.constants import DEFAULT_ORCHESTRATOR_AGENT
        from ...workflow_engine.models import PendingWorkflow, WorkflowProject, WorkflowStatus
        from ...workflow_engine.tool_registry import get_available_tools

        all_tools = get_available_tools(require_available=True)
        if not all_tools:
            self._reply_workflow_error(
                message_id,
                "invalid_state",
                detail="当前环境未检测到可执行的 Workflow 工具。",
            )
            return

        available = list(all_tools)
        recommended_order = ["traex", "claude", "codex", "aiden", "gemini", "coco"]
        recommended = [tool for tool in recommended_order if tool in all_tools]
        default_tools = recommended[:3] or available[:1]
        orchestrator_tool = DEFAULT_ORCHESTRATOR_AGENT
        if orchestrator_tool not in all_tools:
            orchestrator_tool = default_tools[0]
        if orchestrator_tool not in default_tools:
            default_tools.append(orchestrator_tool)

        engine = self.ctx.workflow_engine_manager.get_or_create(
            chat_id,
            root_path,
            engine_name=self.get_engine_name(
                chat_id,
                project_id=(project.project_id if project else None),
            ),
        )
        with engine._lock:
            if not engine.project:
                engine._project = WorkflowProject()
            if admission_owner is not None and (
                vars(engine).get("_workflow_selection_owner") is not admission_owner
                or admission_owner.stop_event.is_set()
            ):
                return
            session_key = admission_owner.session_key if admission_owner else uuid.uuid4().hex
            engine.project.pending = PendingWorkflow(
                requirement=requirement,
                initiator_user_id=get_current_sender_id() or "",
                engine_session_key=session_key,
                selected_tools=list(dict.fromkeys(default_tools)),
                orchestrator_agent=orchestrator_tool,
                orchestrator_binding=ReviewAgentBinding(
                    provider="workflow",
                    tool_name=orchestrator_tool,
                    display_name=orchestrator_tool,
                    agent_type=orchestrator_tool,
                    model_name=None,
                    model_display_name=None,
                    use_default_model=True,
                ),
                review_agents=[],
                auto_reviewer=True,
            )
            engine.project.status = WorkflowStatus.GENERATING_SCRIPT

        self._schedule_generate_and_start_workflow(
            message_id=message_id,
            chat_id=chat_id,
            requirement=requirement,
            project=project,
            root_path=root_path,
            selected_tools=list(default_tools),
            expected_session_key=session_key,
            engine=engine,
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
    def _commit_generated_workflow_if_current(
        cls,
        engine: Any,
        owner: _WorkflowLifecycleOwner,
        pending: Any,
    ) -> bool:
        """CAS a verified generated workflow into the owned generation session."""
        with engine._lock:
            current_pending = engine.project.pending if engine.project else None
            if (
                vars(engine).get("_script_generation_owner") is not owner
                or owner.stop_event.is_set()
                or not cls._is_current_generation_session(engine, owner.session_key)
                or current_pending is None
                or current_pending.project_id != owner.project_id
                or pending.project_id != owner.project_id
            ):
                return False
            engine.project.pending = pending
            return True

    def _generate_and_start_workflow(
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
            """Generate, validate, adopt registered tools, and start execution."""
            from ...thread import get_current_sender_id
            from ...workflow_engine.constants import DEFAULT_ORCHESTRATOR_AGENT
            from ...workflow_engine.models import PendingWorkflow, WorkflowStatus

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
                        owner = current_owner
                    else:
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
                    current_selection_owner.stop_event.set()
                    current_selection_owner.heartbeat_stop_event.set()
                    current_selection_owner.done_event.set()
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
                    "agent_pool": (
                        tuple(existing_pending.agent_pool)
                        if existing_pending is not None
                        else ()
                    ),
                    "orchestrator_agent_id": (
                        existing_pending.orchestrator_agent_id
                        if existing_pending is not None
                        else None
                    ),
                    "orchestrator_was_auto": (
                        existing_pending.orchestrator_was_auto
                        if existing_pending is not None
                        else False
                    ),
                    "selection_session_key": (
                        existing_pending.selection_session_key
                        if existing_pending is not None
                        else None
                    ),
                    "project_id": (
                        existing_pending.project_id
                        if existing_pending is not None
                        else None
                    ),
                    "recommended_agents": (
                        list(existing_pending.recommended_agents)
                        if existing_pending is not None
                        else []
                    ),
                }

            if previous_owner is not None:
                previous_owner.stop_event.set()
                previous_owner.heartbeat_stop_event.set()
                with previous_owner.delivery_lock:
                    pass
            if selection_owner is not None:
                with selection_owner.delivery_lock:
                    pass
                release_owner = getattr(engine, "release_lifecycle_owner", None)
                if callable(release_owner):
                    release_owner(selection_owner)

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
                    progress_card_data = self._workflow_generation_card_data(
                        requirement=requirement,
                        agent_pool=generation_context["agent_pool"],
                        orchestrator_agent_id=(
                            generation_context["orchestrator_agent_id"] or ""
                        ),
                        orchestrator_was_auto=generation_context[
                            "orchestrator_was_auto"
                        ],
                        current_activity=status,
                        elapsed_seconds=elapsed,
                    )
                    if gen_msg_id:
                        self.update_card(
                            gen_msg_id,
                            self._build_workflow_card_from_renderer_data(
                                progress_card_data
                            ),
                        )

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
            try:
                gen_card_data = self._workflow_generation_card_data(
                    requirement=requirement,
                    agent_pool=generation_context["agent_pool"],
                    orchestrator_agent_id=(
                        generation_context["orchestrator_agent_id"] or ""
                    ),
                    orchestrator_was_auto=generation_context[
                        "orchestrator_was_auto"
                    ],
                )
                with owner.delivery_lock:
                    if not _owner_can_deliver():
                        return
                    gen_msg_id = self._replace_or_send_workflow_rendered_card(
                        card_message_id=message_id,
                        chat_id=chat_id,
                        card_data=gen_card_data,
                        origin_message_id=origin_message_id,
                    ) or message_id
                    object.__setattr__(owner, "card_message_id", gen_msg_id)
                if not _owner_can_deliver():
                    return

                heartbeat_thread = threading.Thread(
                    target=_heartbeat_loop,
                    name="wf-gen-heartbeat",
                    daemon=True,
                )
                heartbeat_thread.start()

                script_path, meta = self._generate_script_via_ai(
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

                script_tools = list(dict.fromkeys((meta or {}).get("tools", [])))
                confirmed_pool = tuple(generation_context["agent_pool"])
                confirmed_tools = list(
                    dict.fromkeys(binding.tool_name for binding in confirmed_pool)
                )
                unsupported = [tool for tool in script_tools if tool not in confirmed_tools]
                if unsupported:
                    raise RuntimeError("脚本引用未确认工具: " + ", ".join(unsupported))
                selected = confirmed_tools
                if not confirmed_pool:
                    raise RuntimeError("Workflow generation lost its confirmed Agent Pool")
                script_hash = None
                if script_path:
                    try:
                        with open(script_path, "rb") as file:
                            import hashlib

                            script_hash = hashlib.sha256(file.read()).hexdigest()
                    except OSError:
                        script_hash = None

                prepared_pending = PendingWorkflow(
                    script_path=script_path,
                    requirement=requirement,
                    meta=meta,
                    initiator_user_id=generation_context["initiator_user_id"],
                    engine_session_key=expected_session_key,
                    selected_tools=selected,
                    orchestrator_agent=generation_context["orchestrator_agent"],
                    orchestrator_binding=generation_context["orchestrator_binding"],
                    review_agents=generation_context["review_agents"] or [],
                    auto_reviewer=True,
                    script_hash=script_hash,
                    agent_pool=confirmed_pool,
                    orchestrator_agent_id=generation_context["orchestrator_agent_id"],
                    orchestrator_was_auto=generation_context["orchestrator_was_auto"],
                    selection_session_key=generation_context["selection_session_key"],
                    project_id=generation_context["project_id"],
                    recommended_agents=generation_context["recommended_agents"],
                )
                if not self._commit_generated_workflow_if_current(
                    engine,
                    owner,
                    prepared_pending,
                ):
                    logger.info(
                        "[workflow] Dropping stale script generation result (expected_session=%s)",
                        expected_session_key[:8],
                    )
                    return

                self._queue_generated_workflow(
                    message_id=gen_msg_id or message_id,
                    chat_id=chat_id,
                    project=project,
                    root_path=root_path,
                    engine=engine,
                    generation_owner=owner,
                )
                started = bool(
                    engine.project
                    and engine.project.status == WorkflowStatus.RUNNING
                )
                if not started and _owner_can_deliver():
                    raise RuntimeError("Workflow 自动启动失败")
            except _WorkflowGenerationCancelled as exc:
                safe_error = self._sanitize_generation_error(exc)
                with owner.delivery_lock:
                    with engine._lock:
                        current_pending = engine.project.pending if engine.project else None
                        same_session = bool(
                            current_pending is not None
                            and current_pending.engine_session_key == owner.session_key
                        )
                        owner_is_current = bool(
                            vars(engine).get("_script_generation_owner") is owner
                        )
                        if (
                            owner_is_current
                            and same_session
                            and engine.project.status == WorkflowStatus.GENERATING_SCRIPT
                        ):
                            engine.project.status = WorkflowStatus.CANCELLED
                            engine.project.error = safe_error
                            engine.project.finished_at = time.time()
                        should_deliver = bool(
                            same_session
                            and engine.project is not None
                            and engine.project.status == WorkflowStatus.CANCELLED
                        )
                    if should_deliver and current_pending is not None:
                        cancelled_card_data = self._workflow_generation_card_data(
                            requirement=current_pending.requirement or requirement,
                            agent_pool=current_pending.agent_pool,
                            orchestrator_agent_id=(
                                current_pending.orchestrator_agent_id or ""
                            ),
                            orchestrator_was_auto=(
                                current_pending.orchestrator_was_auto
                            ),
                            current_activity=safe_error,
                            elapsed_seconds=int(time.time() - heartbeat_start),
                            terminal_status="cancelled",
                        )
                        target_message_id = (
                            gen_msg_id or owner.card_message_id or message_id
                        )
                        final_message_id = (
                            self._replace_or_send_workflow_rendered_card(
                                card_message_id=target_message_id,
                                chat_id=chat_id,
                                card_data=cancelled_card_data,
                                origin_message_id=origin_message_id,
                            )
                            or target_message_id
                        )
                        object.__setattr__(
                            owner,
                            "card_message_id",
                            final_message_id,
                        )
            except Exception as exc:
                logger.error(
                    "Workflow script generation failed: %s",
                    exc,
                    exc_info=True,
                )
                safe_error = self._sanitize_generation_error(exc)
                with owner.delivery_lock:
                    if _owner_can_deliver():
                        with engine._lock:
                            if not self._is_current_generation_session(
                                engine,
                                owner.session_key,
                            ):
                                return
                            engine.project.status = WorkflowStatus.FAILED
                            engine.project.error = safe_error
                            engine.project.finished_at = time.time()
                        if gen_msg_id:
                            error_card = self._build_error_card(
                                "internal_error",
                                detail=f"脚本生成失败: {safe_error}",
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
                                detail=f"脚本生成失败: {safe_error}",
                            )
            finally:
                _stop_heartbeat()
                owner.worker_exited_event.set()
                with engine._lock:
                    session_uncertain = owner.active_generation_session is not None
                    if vars(engine).get("_script_generation_owner") is owner:
                        self._retire_workflow_owner(engine, owner)
                        if not session_uncertain:
                            engine._script_generation_owner = None
                    elif session_uncertain:
                        self._retire_workflow_owner(engine, owner)
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
                if not session_uncertain:
                    owner.done_event.set()
                    release_owner = getattr(engine, "release_lifecycle_owner", None)
                    if callable(release_owner):
                        release_owner(owner)

    def _schedule_generate_and_start_workflow_legacy(
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
                            if next_status == WorkflowStatus.IDLE:
                                engine.project.pending = None
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
                    self._generate_and_start_workflow(
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
                    next_status=WorkflowStatus.IDLE,
                    detail_prefix="脚本生成任务提交失败",
                )

    @staticmethod
    def _sanitize_generation_error(exc: BaseException | str) -> str:
        """Return a bounded terminal reason without credentials or internals."""
        import re

        text = " ".join(str(exc or "Workflow generation failed").split())
        text = re.sub(
            r"(?i)\b(token|secret|password|api[_-]?key|access[_-]?key)\s*[=:]\s*[^\s,;]+",
            r"\1=<redacted>",
            text,
        )
        return text[:500] or "Workflow generation failed"

    def _schedule_generate_and_start_workflow(
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
    ) -> Any:
        """Submit generation and reconcile pre-callback scheduler failures."""
        from ...tasking.scheduler import TaskStatus
        from ...workflow_engine.models import WorkflowStatus

        if not expected_session_key or engine is None:
            logger.error("[workflow] Refusing to schedule generation without a session owner")
            return None
        with engine._lock:
            pending = engine.project.pending if engine.project else None
            if (
                engine.project is None
                or engine.project.status != WorkflowStatus.GENERATING_SCRIPT
                or pending is None
                or pending.engine_session_key != expected_session_key
            ):
                logger.info(
                    "[workflow] Refusing stale generation schedule (expected_session=%s)",
                    expected_session_key[:8],
                )
                return None
            lifecycle_owner = vars(engine).get("_script_generation_owner")
        if not isinstance(lifecycle_owner, _WorkflowLifecycleOwner):
            logger.error("[workflow] Refusing generation schedule without lifecycle owner")
            return None

        def fail_once(exc: BaseException | str) -> bool:
            safe_error = self._sanitize_generation_error(exc)
            with lifecycle_owner.delivery_lock:
                with engine._lock:
                    current_pending = engine.project.pending if engine.project else None
                    should_report = bool(
                        vars(engine).get("_script_generation_owner") is lifecycle_owner
                        and not lifecycle_owner.stop_event.is_set()
                        and engine.project is not None
                        and engine.project.status == WorkflowStatus.GENERATING_SCRIPT
                        and current_pending is not None
                        and current_pending.engine_session_key == expected_session_key
                    )
                    if not should_report:
                        return False
                    engine.project.status = WorkflowStatus.FAILED
                    engine.project.error = safe_error
                    engine.project.finished_at = time.time()
                    object.__setattr__(lifecycle_owner, "active_generation_session", None)
                    self._retire_workflow_owner(engine, lifecycle_owner)
                    engine._script_generation_owner = None
                error_card = self._build_error_card(
                    "internal_error",
                    detail=f"脚本生成失败: {safe_error}",
                )
                try:
                    origin_message_id = self._resolve_origin(message_id)
                except Exception:
                    origin_message_id = message_id
                self._replace_or_send_workflow_card(
                    card_message_id=message_id,
                    chat_id=chat_id,
                    card=error_card,
                    origin_message_id=origin_message_id,
                )
            lifecycle_owner.done_event.set()
            release_owner = getattr(engine, "release_lifecycle_owner", None)
            if callable(release_owner):
                release_owner(lifecycle_owner)
            return True

        def run_generate() -> None:
            try:
                task_engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)
                if not self._is_current_generation_session(
                    task_engine,
                    expected_session_key,
                ):
                    return
                self._generate_and_start_workflow(
                    message_id=message_id,
                    chat_id=chat_id,
                    requirement=requirement,
                    project=project,
                    root_path=root_path,
                    selected_tools=selected_tools,
                    expected_session_key=expected_session_key,
                )
            except _WorkflowGenerationCancelled as exc:
                cancelled_pending = None
                should_deliver = False
                with lifecycle_owner.delivery_lock:
                    with engine._lock:
                        current_pending = engine.project.pending if engine.project else None
                        if (
                            vars(engine).get("_script_generation_owner") is lifecycle_owner
                            and engine.project is not None
                            and engine.project.status == WorkflowStatus.GENERATING_SCRIPT
                            and current_pending is not None
                            and current_pending.engine_session_key == expected_session_key
                        ):
                            engine.project.status = WorkflowStatus.CANCELLED
                            engine.project.error = self._sanitize_generation_error(exc)
                            engine.project.finished_at = time.time()
                            engine._script_generation_owner = None
                            cancelled_pending = current_pending
                            should_deliver = True
                    if should_deliver and cancelled_pending is not None:
                        target_message_id = (
                            lifecycle_owner.card_message_id or message_id
                        )
                        cancelled_card_data = self._workflow_generation_card_data(
                            requirement=cancelled_pending.requirement or requirement,
                            agent_pool=cancelled_pending.agent_pool,
                            orchestrator_agent_id=(
                                cancelled_pending.orchestrator_agent_id or ""
                            ),
                            orchestrator_was_auto=(
                                cancelled_pending.orchestrator_was_auto
                            ),
                            current_activity=self._sanitize_generation_error(exc),
                            terminal_status="cancelled",
                        )
                        final_message_id = (
                            self._replace_or_send_workflow_rendered_card(
                                card_message_id=target_message_id,
                                chat_id=chat_id,
                                card_data=cancelled_card_data,
                                origin_message_id=message_id,
                            )
                            or target_message_id
                        )
                        object.__setattr__(
                            lifecycle_owner,
                            "card_message_id",
                            final_message_id,
                        )
                if should_deliver:
                    lifecycle_owner.done_event.set()
                    release_owner = getattr(engine, "release_lifecycle_owner", None)
                    if callable(release_owner):
                        release_owner(lifecycle_owner)
            except Exception as exc:
                logger.error("Workflow script generation task failed: %s", exc, exc_info=True)
                fail_once(exc)

        project_name = (
            (getattr(project, "project_name", "") if project else "")
            or os.path.basename(root_path)
        )
        task_id = generate_task_id(project_name or "workflow")
        try:
            handle = self._submit_engine_task(
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
            fail_once(exc)
            return None

        run_id = getattr(handle, "run_id", None)
        scheduler = getattr(self.ctx, "scheduler", None)
        if not run_id or scheduler is None:
            return handle

        def scheduler_state() -> Any:
            try:
                return scheduler.get_state(run_id)
            except Exception:
                logger.debug("Unable to reconcile Workflow generation scheduler state", exc_info=True)
                return None

        state = scheduler_state()
        if state is not None and getattr(state, "status", None) in {
            TaskStatus.FAILED,
            TaskStatus.CANCELED,
        }:
            fail_once(getattr(state, "error", None) or "generation scheduler rejected the task")
            return handle

        if state is not None and getattr(state, "status", None) == TaskStatus.QUEUED:
            def reconcile_queued() -> None:
                deadline = time.monotonic() + 600.0
                while time.monotonic() < deadline and not lifecycle_owner.stop_event.wait(1.0):
                    queued_state = scheduler_state()
                    if queued_state is None:
                        continue
                    status = getattr(queued_state, "status", None)
                    if status in {TaskStatus.RUNNING, TaskStatus.SUCCEEDED}:
                        return
                    if status in {TaskStatus.FAILED, TaskStatus.CANCELED}:
                        fail_once(
                            getattr(queued_state, "error", None)
                            or "generation scheduler rejected the queued task"
                        )
                        return
                if not lifecycle_owner.stop_event.is_set():
                    try:
                        handle.cancel()
                    except Exception:
                        logger.debug("Unable to cancel unreconciled generation task", exc_info=True)
                    fail_once("generation scheduler queue reconciliation timed out")

            threading.Thread(
                target=reconcile_queued,
                daemon=True,
                name=f"workflow-generation-reconcile-{expected_session_key[:8]}",
            ).start()
        return handle

    # ------------------------------------------------------------------
    # Stop workflow
    # ------------------------------------------------------------------



    @staticmethod
    def _retire_workflow_owner(engine: Any, owner: Any) -> None:
        """Keep a revoked owner discoverable by later cleanup retries."""
        retire = getattr(engine, "retire_lifecycle_owner", None)
        if callable(retire):
            retire(owner)

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
        lifecycle_owner: _WorkflowLifecycleOwner,
        lifecycle_engine: Any,
    ):
        """Fence Workflow repo-lock/error cards with the queued-start owner."""
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









    def stop_workflow(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        *,
        terminal_is_noop: bool = False,
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
        cancel_card_message_ids: list[str] = []
        sessions_to_close: list[tuple[_WorkflowLifecycleOwner, Any]] = []
        artifacts_to_remove: list[str] = []
        runtime_active = False
        terminal_notice: WorkflowStatus | None = None
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
                if terminal_is_noop and terminal_committed:
                    terminal_notice = status
                else:
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
                    if (
                        lifecycle_owner.card_message_id
                        and lifecycle_owner.card_message_id
                        not in cancel_card_message_ids
                    ):
                        cancel_card_message_ids.append(
                            lifecycle_owner.card_message_id
                        )
                    active_session = lifecycle_owner.active_generation_session
                    if active_session is not None and not any(
                        existing_session is active_session
                        for _existing_owner, existing_session in sessions_to_close
                    ):
                        sessions_to_close.append((lifecycle_owner, active_session))
                    for artifact in (
                        lifecycle_owner.source_script_path,
                        lifecycle_owner.execution_script_path,
                    ):
                        if artifact:
                            artifacts_to_remove.append(artifact)
                if live_selection_owner is not None:
                    engine._workflow_selection_owner = None
                if (
                    live_generation_owner is not None
                    and live_generation_owner.active_generation_session is None
                ):
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
                    wf_project.status = WorkflowStatus.CANCELLED
                    wf_project.error = "Workflow cancelled by user"
                    wf_project.finished_at = time.time()
                    wf_project.script_path = None

        if terminal_notice is not None:
            terminal_label = {
                WorkflowStatus.COMPLETED: "已完成",
                WorkflowStatus.FAILED: "失败",
                WorkflowStatus.CANCELLED: "已取消",
            }.get(terminal_notice, "已结束")
            self.reply_text(
                message_id,
                f"Workflow 任务已结束，当前状态：{terminal_label}。",
            )
            return

        if error is not None:
            self._reply_workflow_error(
                message_id,
                error[0],
                detail=error[1],
            )
            return

        uncertain_sessions: list[tuple[_WorkflowLifecycleOwner, Any, str]] = []
        for session_owner, session in sessions_to_close:
            cancel_confirmed = True
            close_confirmed = True
            details: list[str] = []
            try:
                cancelled = session.cancel(wait=True, timeout=2.0)
                if cancelled is False:
                    cancel_confirmed = False
                    details.append("cancel was not confirmed")
            except Exception as exc:
                cancel_confirmed = False
                details.append(f"cancel failed: {exc}")
                logger.warning("Failed to cancel active Workflow generation session", exc_info=True)
            try:
                session.close()
            except Exception as exc:
                close_confirmed = False
                details.append(f"close failed: {exc}")
                logger.warning("Failed to close active Workflow generation session", exc_info=True)
            if cancel_confirmed and close_confirmed:
                object.__setattr__(session_owner, "active_generation_session", None)
                with engine._lock:
                    if vars(engine).get("_script_generation_owner") is session_owner:
                        engine._script_generation_owner = None
            else:
                uncertain_sessions.append(
                    (session_owner, session, "; ".join(details) or "stop was not confirmed")
                )

        if uncertain_sessions:
            safe_error = "Workflow 停止未确认；为避免重叠会话，当前项目保持封锁。"
            with engine._lock:
                if engine.project is not None:
                    engine.project.status = WorkflowStatus.FAILED
                    engine.project.error = safe_error
                    engine.project.finished_at = time.time()
                for session_owner, _session, _detail in uncertain_sessions:
                    engine._script_generation_owner = session_owner
                    self._retire_workflow_owner(engine, session_owner)
            self._reply_workflow_error(
                message_id,
                "internal_error",
                detail=safe_error,
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
            release_owner = getattr(engine, "release_lifecycle_owner", None)
            if callable(release_owner):
                release_owner(lifecycle_owner)

        if not runtime_active and pending is not None and cancel_card_message_ids:
            cancelled_card_data = self._workflow_generation_card_data(
                requirement=pending.requirement or "",
                agent_pool=pending.agent_pool,
                orchestrator_agent_id=pending.orchestrator_agent_id or "",
                orchestrator_was_auto=pending.orchestrator_was_auto,
                current_activity="Workflow 任务已由用户停止",
                terminal_status="cancelled",
            )
            for card_message_id in cancel_card_message_ids:
                self._replace_or_send_workflow_rendered_card(
                    card_message_id=card_message_id,
                    chat_id=chat_id,
                    card_data=cancelled_card_data,
                    origin_message_id=message_id,
                    fallback_to_new=False,
                )

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
        self.stop_workflow(
            message_id,
            chat_id,
            project,
            terminal_is_noop=True,
        )

















    def _queue_generated_workflow(
            self,
            *,
            message_id: str,
            chat_id: str,
            project: Optional["ProjectContext"],
            root_path: str,
            engine: Any,
            generation_owner: _WorkflowLifecycleOwner,
        ) -> None:
            """Validate the owned generated script and hand it directly to runtime."""
            if not engine or not engine.project:
                return

            from ...workflow_engine.models import WorkflowStatus

            def _reply_start_error(
                category: str,
                *,
                detail: str = "",
            ) -> None:
                with generation_owner.delivery_lock:
                    with engine._lock:
                        owner_is_current = bool(
                            vars(engine).get("_script_generation_owner") is generation_owner
                            and not generation_owner.stop_event.is_set()
                        )
                    if not owner_is_current:
                        return
                    safe_error = self._sanitize_generation_error(
                        detail or f"Workflow 自动启动失败: {category}"
                    )
                    with engine._lock:
                        if (
                            vars(engine).get("_script_generation_owner")
                            is not generation_owner
                            or generation_owner.stop_event.is_set()
                        ):
                            return
                        engine.project.status = WorkflowStatus.FAILED
                        engine.project.error = safe_error
                        engine.project.finished_at = time.time()
                    error_card = self._build_error_card(
                        category,
                        detail=safe_error,
                    )
                    try:
                        origin_message_id = self._resolve_origin(message_id)
                    except Exception:
                        origin_message_id = message_id
                    self._replace_or_send_workflow_card(
                        card_message_id=(
                            generation_owner.card_message_id or message_id
                        ),
                        chat_id=chat_id,
                        card=error_card,
                        origin_message_id=origin_message_id,
                    )

            with engine._lock:
                current_project = engine.project
                pending = current_project.pending if current_project else None
                if current_project is None or current_project.status != WorkflowStatus.GENERATING_SCRIPT:
                    pending = None
                if (
                    vars(engine).get("_script_generation_owner") is not generation_owner
                    or generation_owner.stop_event.is_set()
                ):
                    return

            if pending is None:
                _reply_start_error("invalid_state")
                return

            stored_initiator = pending.initiator_user_id or ""
            stored_session_key = pending.engine_session_key or ""

            if not stored_session_key or generation_owner.session_key != stored_session_key:
                logger.warning(
                    "Workflow auto-start rejected: session_key mismatch (expected=%s, stored=%s)",
                    generation_owner.session_key[:8],
                    stored_session_key[:8],
                )
                _reply_start_error("session_expired")
                return

            if not stored_initiator:
                logger.warning(
                    "Workflow auto-start rejected: initiator is missing",
                )
                _reply_start_error("forbidden")
                return

            # Retrieve pending state
            script_path = pending.script_path
            requirement = pending.requirement or ""
            expected_script_hash = pending.script_hash

            if not script_path:
                _reply_start_error(
                    "invalid_state",
                    detail="无法获取待执行脚本，请重新发送 `/wf`",
                )
                return

            # --- TOCTOU hardening (check-then-re-read-verify) ---
            # Re-read script content fresh from disk at execution handoff time and
            # re-run the full validation chain. This defends against scripts
            # being tampered with after generation but before runtime handoff.
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

            is_valid, validation_errors = validate_generated_script(
                script_text,
                agent_pool=pending.agent_pool,
            )
            if not is_valid:
                _reply_start_error(
                    "internal_error",
                    detail="脚本验证失败：" + "; ".join(validation_errors[:3]),
                )
                return
            fresh_meta = extract_meta_from_script(script_text) or {}

            fresh_script_tools = list(dict.fromkeys(fresh_meta.get("tools", [])))
            confirmed_tools = list(
                dict.fromkeys(binding.tool_name for binding in pending.agent_pool)
            )
            unsupported = [tool for tool in fresh_script_tools if tool not in confirmed_tools]
            if unsupported:
                _reply_start_error(
                    "invalid_state",
                    detail="脚本引用未确认工具: " + ", ".join(unsupported),
                )
                return
            pending.selected_tools = confirmed_tools

            # --- Immutable copy for execution ---
            # Copy the verified content into a fresh /tmp file for each
            # verified execution. Using mkstemp guarantees uniqueness across
            # concurrent sessions and test suites; we additionally append the
            # script hash so accidental inspection can trace the source.
            import tempfile

            temp_fd, immutable_script_path = tempfile.mkstemp(
                prefix="ghostap-verified-",
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

            start_owner = _WorkflowLifecycleOwner(
                session_key=stored_session_key,
                initiator_user_id=stored_initiator,
                chat_id=chat_id,
                project_id=pending.project_id or "",
                root_path=root_path,
                card_message_id=message_id,
                stop_event=generation_owner.stop_event,
                heartbeat_stop_event=generation_owner.heartbeat_stop_event,
                delivery_lock=generation_owner.delivery_lock,
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
                release_owner = getattr(engine, "release_lifecycle_owner", None)
                if callable(release_owner):
                    release_owner(start_owner)

            # Linearize the generated-session handoff against /stop_wf. All
            # expensive validation above is speculative; this is the authoritative
            # state/session check immediately before publishing RUNNING.
            with start_owner.delivery_lock:
                with engine._lock:
                    current_project = engine.project
                    current_pending = current_project.pending if current_project is not None else None
                    current_session_key = current_pending.engine_session_key if current_pending else ""
                    generation_owner_matches = vars(engine).get("_script_generation_owner") is generation_owner
                    if (
                        current_project is None
                        or current_project.status != WorkflowStatus.GENERATING_SCRIPT
                        or current_session_key != stored_session_key
                        or not generation_owner_matches
                        or start_owner.stop_event.is_set()
                    ):
                        _discard_immutable_script()
                        return

                    try:
                        run_spec = self._build_run_spec(
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
                        self._retire_workflow_owner(
                            engine,
                            generation_owner,
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
                    project_id = getattr(project, "project_id", "") or ""
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
                "`/wf <需求描述>` · AI 生成编排脚本并执行\n"
                "`/wf_status` · 查看当前 Workflow 进度\n"
                "`/wf_help` · 显示本帮助\n"
                "`/stop_wf` · 停止正在执行的 Workflow",
            ),
            (
                "🧭 执行流程",
                "**① 自动编排** · 使用推荐可用工具和后端默认模型\n"
                "**② 自动验证** · 有界修复生成脚本并检查工具与安全约束\n"
                "**③ 自动执行** · 立即启动，多阶段并行执行并实时更新进度卡片",
            ),
            (
                "✨ 核心能力",
                "• 多工具并行调度（coco / claude / aiden / codex / traex）\n"
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

        Only disposable generated scripts and immutable execution copies are owned here.
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

        has_uncertain_session = getattr(engine, "has_uncertain_lifecycle_session", None)
        if callable(has_uncertain_session) and has_uncertain_session():
            return False, "invalid_state", None

        admission_owner = _WorkflowLifecycleOwner(
            session_key=uuid.uuid4().hex,
            initiator_user_id=current_user,
        )
        owners: list[_WorkflowLifecycleOwner] = []
        artifacts: list[str] = []
        with engine._lock:
            if callable(has_uncertain_session) and has_uncertain_session():
                return False, "invalid_state", None
            if getattr(engine, "_closing", False) is True:
                return False, "running", None
            if engine.is_running is True:
                return False, "running", None

            wf_project = engine.project
            if wf_project is None:
                wf_project = WorkflowProject()
                engine._project = wf_project
            if wf_project.status == WorkflowStatus.GENERATING_SCRIPT:
                return False, "invalid_state", None
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
                    if owner is selection_owner:
                        owner.done_event.set()
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
                # source. It is retired at this exact lifecycle boundary.
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
            release_owner = getattr(engine, "release_lifecycle_owner", None)
            if callable(release_owner):
                release_owner(owner)
        for artifact in dict.fromkeys(artifacts):
            self._remove_owned_workflow_artifact(
                artifact,
                root_path=root_path,
            )
        return True, None, admission_owner

    def _generate_script_via_ai_legacy(
        self,
        requirement: str,
        root_path: str,
        selected_tools: list[str] | None = None,
        engine: Any = None,
        progress_callback: Any = None,
        *,
        output_path: str,
        cancel_event: threading.Event | None = None,
        artifact_lock: Any = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate and validate a script with a bounded automatic retry loop."""
        from ...agent_session import close_session_safely, create_engine_session
        from ...workflow_engine.constants import DEFAULT_ORCHESTRATOR_AGENT, SCRIPT_GEN_TIMEOUT_S
        from ...workflow_engine.script_gen import (
            build_script_gen_prompt,
            extract_meta_from_script,
            validate_generated_script,
        )
        from ...workflow_engine.tool_registry import get_available_tools

        agent_type = (
            engine.project.pending.orchestrator_agent
            if engine and engine.project and engine.project.pending and engine.project.pending.orchestrator_agent
            else DEFAULT_ORCHESTRATOR_AGENT
        )
        script_dir = os.path.join(root_path, ".ghostap", "workflow_scripts")
        os.makedirs(script_dir, exist_ok=True)
        script_path = output_path
        if (
            os.path.realpath(os.path.dirname(script_path)) != os.path.realpath(script_dir)
            or not os.path.basename(script_path).endswith(".js")
        ):
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

        registered_tools = get_available_tools(require_available=True)
        if not registered_tools:
            raise RuntimeError("当前环境没有已注册且可执行的 Workflow 工具")
        preferred = [tool for tool in (selected_tools or []) if tool in registered_tools]
        ordered_names = list(dict.fromkeys([*preferred, *registered_tools]))
        prompt_tools = {name: registered_tools[name] for name in ordered_names}

        pending = engine.project.pending if engine and engine.project else None
        orchestrator_binding = pending.orchestrator_binding if pending else None
        selected_model_name = (
            orchestrator_binding.model_name
            if orchestrator_binding and not orchestrator_binding.use_default_model
            else None
        )
        base_prompt = build_script_gen_prompt(
            requirement=requirement,
            available_tools=prompt_tools,
            orchestrator_agent=agent_type,
            orchestrator_binding=orchestrator_binding,
            review_agents=[],
            auto_reviewer=True,
        )

        mutating_tools = frozenset(
            {
                "execute_command", "create_terminal", "run_terminal", "run_shell", "shell", "bash",
                "write_file", "write_text_file", "delete_file", "remove_file", "mkdir", "patch_file",
                "apply_diff", "edit_file", "write_to_file", "http_request", "http_get", "http_post",
                "fetch", "download", "upload", "network_request", "url_open", "send_message",
                "send_email", "create_issue",
            }
        )

        def script_gen_tool_filter(tool_name: str, _params: dict | None) -> bool:
            if not isinstance(tool_name, str):
                return False
            normalized = tool_name.lower().strip()
            if normalized in mutating_tools:
                return False
            return not any(
                token in normalized
                for token in ("write", "delete", "remove", "exec", "run", "patch", "post", "upload", "send", "create")
            )

        last_error = "模型未返回有效脚本"
        for attempt in range(1, _SCRIPT_GENERATION_MAX_ATTEMPTS + 1):
            ensure_not_cancelled()
            if progress_callback:
                progress_callback(f"正在生成编排脚本（尝试 {attempt}/{_SCRIPT_GENERATION_MAX_ATTEMPTS}）...")
            session = None
            try:
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
                    last_error = "无法创建脚本生成会话"
                    continue
                session.set_tool_filter(script_gen_tool_filter)
                retry_note = "" if attempt == 1 else (
                    "\n\n上一次输出未通过验证。请重新生成完整脚本，"
                    "并严格使用提示中列出的工具。"
                    f"\n上次失败原因：{last_error}"
                )
                timeout_s = getattr(self.settings, "workflow_script_gen_timeout_s", SCRIPT_GEN_TIMEOUT_S)
                result = session.send_prompt(base_prompt + retry_note, timeout=timeout_s)
                ensure_not_cancelled()
                if not result or not result.text:
                    last_error = "模型返回了空脚本"
                    continue
                script_content = self._strip_markdown_fences(result.text.strip())
                is_valid, errors = validate_generated_script(
                    script_content,
                    review_agents=[],
                )
                if not is_valid:
                    last_error = "; ".join(errors[:3]) or "脚本结构验证失败"
                    logger.warning("Generated script attempt %s failed validation: %s", attempt, errors)
                    continue
                meta = extract_meta_from_script(script_content) or {}
                unsupported = sorted(set(meta.get("tools", [])) - set(registered_tools))
                if unsupported:
                    last_error = "脚本引用未注册工具: " + ", ".join(unsupported)
                    logger.warning("Generated script attempt %s used unsupported tools: %s", attempt, unsupported)
                    continue
                write_generated_content(script_content)
                return script_path, meta
            except _WorkflowGenerationCancelled:
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Workflow script generation attempt %s failed: %s", attempt, exc)
            finally:
                if session is not None:
                    close_session_safely(session)

        raise RuntimeError(
            f"脚本生成在 {_SCRIPT_GENERATION_MAX_ATTEMPTS} 次尝试后仍未通过验证：{last_error}"
        )

    def _generate_script_via_ai(
        self,
        requirement: str,
        root_path: str,
        selected_tools: list[str] | None = None,
        engine: Any = None,
        progress_callback: Any = None,
        *,
        output_path: str,
        cancel_event: threading.Event | None = None,
        artifact_lock: Any = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate through the confirmed pool with strict attempt fencing."""
        del selected_tools
        from ...agent_session import create_engine_session
        from ...workflow_engine.models import WorkflowStatus
        from ...workflow_engine.script_gen import (
            build_script_gen_prompt,
            extract_meta_from_script,
            validate_generated_script,
        )
        from ...workflow_engine.tool_registry import get_available_tools

        if engine is None or engine.project is None or engine.project.pending is None:
            raise RuntimeError("Workflow generation session is missing")
        pending = engine.project.pending
        pool = tuple(pending.agent_pool or ())
        if not pool:
            raise RuntimeError("Workflow generation requires a confirmed Agent Pool")
        orchestrator_agent_id = pending.orchestrator_agent_id or pool[0].agent_id
        orchestrator = next(
            (item for item in pool if item.agent_id == orchestrator_agent_id),
            None,
        )
        if orchestrator is None:
            raise RuntimeError("Workflow orchestrator is outside the confirmed Agent Pool")

        registered_tools = get_available_tools(require_available=True)
        unavailable = sorted({item.tool_name for item in pool} - set(registered_tools))
        if unavailable:
            raise RuntimeError("Confirmed Workflow tool is unavailable: " + ", ".join(unavailable))
        prompt_tools = {
            item.tool_name: str(registered_tools[item.tool_name])
            for item in pool
        }
        base_prompt = build_script_gen_prompt(
            requirement=requirement,
            available_tools=prompt_tools,
            orchestrator_agent=orchestrator.tool_name,
            orchestrator_binding=orchestrator.to_dict(),
            review_agents=[],
            auto_reviewer=True,
            agent_pool=pool,
            orchestrator_agent_id=orchestrator_agent_id,
        )

        script_dir = os.path.join(root_path, ".ghostap", "workflow_scripts")
        os.makedirs(script_dir, exist_ok=True)
        if (
            os.path.realpath(os.path.dirname(output_path)) != os.path.realpath(script_dir)
            or not os.path.basename(output_path).endswith(".js")
        ):
            raise ValueError("Workflow generation output must stay in the script staging directory")

        captured_session_key = pending.engine_session_key
        started_at = time.monotonic()
        hard_deadline = started_at + 600.0
        ordered_pool = (
            orchestrator,
            *(item for item in pool if item.agent_id != orchestrator.agent_id),
        )
        mutating_tools = frozenset(
            {
                "execute_command", "create_terminal", "run_terminal", "run_shell", "shell",
                "bash", "write_file", "write_text_file", "delete_file", "remove_file",
                "mkdir", "patch_file", "apply_diff", "edit_file", "write_to_file",
                "http_request", "http_get", "http_post", "fetch", "download", "upload",
                "network_request", "url_open", "send_message", "send_email", "create_issue",
            }
        )

        def tool_filter(tool_name: str, _params: dict | None) -> bool:
            if not isinstance(tool_name, str):
                return False
            normalized = tool_name.lower().strip()
            if normalized in mutating_tools:
                return False
            return not any(
                token in normalized
                for token in (
                    "write", "delete", "remove", "exec", "run", "patch", "post",
                    "upload", "send", "create",
                )
            )

        def ensure_current() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise _WorkflowGenerationCancelled("Workflow generation cancelled")
            with engine._lock:
                current_project = engine.project
                current_pending = current_project.pending if current_project else None
                if (
                    current_project is None
                    or current_project.status != WorkflowStatus.GENERATING_SCRIPT
                    or current_pending is not pending
                    or current_pending.engine_session_key != captured_session_key
                ):
                    raise RuntimeError("stale Workflow generation session was superseded")

        def _owner_is_current_locked(owner: _WorkflowLifecycleOwner) -> bool:
            current_project = engine.project
            current_pending = current_project.pending if current_project else None
            return bool(
                vars(engine).get("_script_generation_owner") is owner
                and not owner.stop_event.is_set()
                and (cancel_event is None or not cancel_event.is_set())
                and owner.session_key == captured_session_key
                and current_project is not None
                and current_project.status == WorkflowStatus.GENERATING_SCRIPT
                and current_pending is pending
                and current_pending.engine_session_key == captured_session_key
            )

        def current_generation_owner() -> _WorkflowLifecycleOwner | None:
            with engine._lock:
                owner = vars(engine).get("_script_generation_owner")
                if (
                    isinstance(owner, _WorkflowLifecycleOwner)
                    and _owner_is_current_locked(owner)
                ):
                    return owner
                return None

        def bind_active_session(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> bool:
            with engine._lock:
                if not _owner_is_current_locked(owner):
                    return False
                active_session = owner.active_generation_session
                if active_session is not None and active_session is not session:
                    return False
                object.__setattr__(owner, "active_generation_session", session)
                return True

        def active_session_is_current(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> bool:
            with engine._lock:
                return bool(
                    _owner_is_current_locked(owner)
                    and owner.active_generation_session is session
                )

        def anchor_session_for_cleanup(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> None:
            with engine._lock:
                self._retire_workflow_owner(engine, owner)
                active_session = owner.active_generation_session
                if active_session is None or active_session is session:
                    object.__setattr__(owner, "active_generation_session", session)

        def unbind_active_session(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> None:
            with engine._lock:
                if owner.active_generation_session is session:
                    object.__setattr__(owner, "active_generation_session", None)

        session_owner: _WorkflowLifecycleOwner | None = None

        def close_strict(session: Any, *, cancel: bool) -> None:
            cancel_confirmed = True
            close_confirmed = True
            details: list[str] = []
            if cancel:
                try:
                    cancelled = session.cancel(wait=True, timeout=2.0)
                    if cancelled is False:
                        cancel_confirmed = False
                        details.append("generation session cancellation was not confirmed")
                except Exception as exc:
                    cancel_confirmed = False
                    details.append(f"generation session cancellation failed: {exc}")
            try:
                session.close()
            except Exception as exc:
                close_confirmed = False
                details.append(f"generation session close uncertain: {exc}")
            if cancel_confirmed and close_confirmed:
                if session_owner is not None:
                    unbind_active_session(session_owner, session)
                    release_owner = getattr(engine, "release_lifecycle_owner", None)
                    if callable(release_owner):
                        release_owner(session_owner)
                return
            raise _WorkflowGenerationCloseUncertain("; ".join(details))

        def transport_failure(exc: BaseException) -> bool:
            text = f"{type(exc).__name__}: {exc}".lower()
            return isinstance(exc, (TimeoutError, ConnectionError, OSError)) or any(
                token in text
                for token in (
                    "timeout", "timed out", "transport", "disconnect", "connection",
                    "rate limit", "ratelimit", "too many requests", "429",
                )
            )

        def permission_or_unsafe(exc: BaseException | str) -> bool:
            text = str(exc).lower()
            return any(
                token in text
                for token in ("permission", "forbidden", "unauthorized", "unsafe", "policy")
            )

        last_error = "模型未返回有效脚本"
        for binding in ordered_pool:
            for repair_attempt in range(1, _SCRIPT_GENERATION_MAX_ATTEMPTS + 1):
                ensure_current()
                remaining = hard_deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Workflow generation total timeout exceeded")
                if progress_callback:
                    progress_callback(
                        f"{binding.agent_id} 正在生成编排脚本（修复 {repair_attempt}/"
                        f"{_SCRIPT_GENERATION_MAX_ATTEMPTS}）"
                    )

                retry_note = ""
                if repair_attempt > 1:
                    retry_note = (
                        "\n\nBounded repair attempt "
                        f"{repair_attempt}: the previous output failed validation: {last_error}. "
                        "Return one complete corrected JavaScript module."
                    )
                session = None
                should_fallback = False
                try:
                    session_owner = current_generation_owner()
                    if session_owner is None:
                        raise RuntimeError(
                            "stale Workflow generation owner before session creation"
                        )
                    session = create_engine_session(
                        agent_type=binding.tool_name,
                        cwd=root_path,
                        thread_id=(
                            f"workflow_script_gen_{captured_session_key}_{binding.agent_id}_"
                            f"{repair_attempt}"
                        ),
                        auto_approve=False,
                        require_tool_filter=True,
                        model_name=binding.model_name,
                        cancel_event=cancel_event,
                    )
                    if session is None:
                        last_error = f"{binding.agent_id}: unable to create generation session"
                        should_fallback = True
                        break
                    if not bind_active_session(session_owner, session):
                        anchor_session_for_cleanup(session_owner, session)
                        close_strict(session, cancel=True)
                        session = None
                        raise RuntimeError(
                            "stale Workflow generation owner before session bind"
                        )
                    session.set_tool_filter(tool_filter)
                    if not active_session_is_current(session_owner, session):
                        close_strict(session, cancel=True)
                        session = None
                        raise RuntimeError(
                            "stale Workflow generation owner before prompt send"
                        )

                    def on_event(event: Any) -> None:
                        text = str(getattr(event, "text", "") or "").strip()
                        event_type = getattr(event, "event_type", None)
                        event_name = str(getattr(event_type, "value", event_type) or "").lower()
                        meaningful = self._workflow_generation_event_is_meaningful(event)
                        if meaningful and progress_callback:
                            summary = text or event_name.replace("_", " ")
                            progress_callback(
                                f"{binding.agent_id} last activity: {summary[:180]}"
                            )

                    result = session.send_prompt(
                        base_prompt + retry_note,
                        timeout=max(0.001, min(600.0, remaining)),
                        idle_timeout=max(0.001, min(120.0, remaining)),
                        on_event=on_event,
                        activity_predicate=self._workflow_generation_event_is_meaningful,
                    )
                    try:
                        ensure_current()
                    except Exception:
                        close_strict(session, cancel=True)
                        session = None
                        raise

                    stop_reason = str(getattr(result, "stop_reason", "") or "").strip().lower()
                    if stop_reason in {"cancelled", "canceled"}:
                        close_strict(session, cancel=True)
                        session = None
                        raise _WorkflowGenerationCancelled(
                            f"Workflow generation cancelled by {binding.agent_id}"
                        )
                    if stop_reason in {"timeout", "timed_out", "error", "failed", "failure"}:
                        last_error = f"{binding.agent_id} terminal stop_reason={stop_reason}"
                        close_strict(session, cancel=True)
                        session = None
                        should_fallback = True
                        break
                    if stop_reason not in {
                        "end_turn", "completed", "complete", "success", "done",
                    }:
                        last_error = f"{binding.agent_id} terminal stop_reason={stop_reason}"
                        close_strict(session, cancel=True)
                        session = None
                        should_fallback = True
                        break
                    text = str(getattr(result, "text", "") or "").strip()
                    if not text:
                        last_error = f"{binding.agent_id}: model returned an empty script"
                        close_strict(session, cancel=False)
                        session = None
                        continue
                    script_content = self._strip_markdown_fences(text)
                    is_valid, errors = validate_generated_script(
                        script_content,
                        review_agents=[],
                        agent_pool=pool,
                    )
                    if not is_valid:
                        last_error = "; ".join(errors[:3]) or "script validation failed"
                        close_strict(session, cancel=False)
                        session = None
                        if permission_or_unsafe(last_error) or "[capability]" in last_error.lower():
                            raise RuntimeError(last_error)
                        continue
                    meta = extract_meta_from_script(script_content) or {}
                    unsupported = sorted(
                        set(meta.get("tools", [])) - {item.tool_name for item in pool}
                    )
                    if unsupported:
                        last_error = "脚本引用未确认工具: " + ", ".join(unsupported)
                        close_strict(session, cancel=False)
                        session = None
                        continue

                    close_strict(session, cancel=False)
                    session = None
                    ensure_current()
                    lock_context = artifact_lock if artifact_lock is not None else nullcontext()
                    with lock_context:
                        ensure_current()
                        with open(output_path, "w", encoding="utf-8") as file:
                            file.write(script_content)
                    return output_path, meta
                except _WorkflowGenerationCloseUncertain:
                    raise
                except _WorkflowGenerationCancelled:
                    if session is not None:
                        close_strict(session, cancel=True)
                    raise
                except Exception as exc:
                    if session is not None:
                        if permission_or_unsafe(exc):
                            close_strict(session, cancel=True)
                            raise
                        if transport_failure(exc):
                            close_strict(session, cancel=True)
                            session = None
                            last_error = f"{type(exc).__name__}: {exc}"
                            should_fallback = True
                            break
                        close_strict(session, cancel=True)
                    if "stale" in str(exc).lower() or "superseded" in str(exc).lower():
                        raise
                    if permission_or_unsafe(exc):
                        raise
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "Workflow generation %s repair %s failed: %s",
                        binding.agent_id,
                        repair_attempt,
                        exc,
                    )
                    continue
                if should_fallback:
                    break
            if hard_deadline - time.monotonic() <= 0:
                raise RuntimeError("Workflow generation total timeout exceeded")

        raise RuntimeError(f"Workflow generation failed: {last_error}")

    @staticmethod
    def _workflow_generation_event_is_meaningful(event: Any) -> bool:
        text = str(getattr(event, "text", "") or "").strip()
        if text:
            return True
        event_type = getattr(event, "event_type", None)
        event_name = str(getattr(event_type, "value", event_type) or "").lower()
        return any(
            token in event_name
            for token in (
                "tool_call_start",
                "tool_call_update",
                "tool_call_done",
                "plan_update",
                "image_chunk",
            )
        )

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
    def _build_run_spec(
        *,
        pending: Any,
        engine: Any,
        task: str,
        chat_id: str,
        topic_id: str | None,
    ):
        """Freeze the automatic tool binding into one execution contract."""
        from ...spec_engine.models import ReviewAgentBinding
        from ...workflow_engine.constants import (
            MAX_TOTAL_AGENTS,
            WORKFLOW_TOTAL_TIMEOUT_S,
        )
        from ...workflow_engine.run_spec import WorkflowRunSpec

        agent_pool = tuple(pending.agent_pool or ())
        if not agent_pool:
            raise ValueError("Workflow execution is missing its confirmed Agent Pool")
        orchestrator_agent_id = pending.orchestrator_agent_id or agent_pool[0].agent_id
        orchestrator_agent = next(
            (binding for binding in agent_pool if binding.agent_id == orchestrator_agent_id),
            None,
        )
        if orchestrator_agent is None:
            raise ValueError("Workflow orchestrator is outside its confirmed Agent Pool")
        orchestrator = ReviewAgentBinding(
            provider="workflow",
            tool_name=orchestrator_agent.tool_name,
            display_name=orchestrator_agent.display_name,
            agent_type=orchestrator_agent.tool_name,
            model_name=orchestrator_agent.model_name,
            model_display_name=orchestrator_agent.model_name,
            selection_key=orchestrator_agent.agent_id,
            use_default_model=orchestrator_agent.model_name is None,
        )
        allowed_tools = list(
            dict.fromkeys(binding.tool_name for binding in agent_pool)
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
            reviewers=(),
            tool_model_map={},
            task=task,
            chat_id=chat_id,
            topic_id=topic_id,
            budget=MAX_TOTAL_AGENTS,
            deadline=deadline,
            auto_reviewer=True,
            initiator_user_id=pending.initiator_user_id or None,
            allowed_tools=tuple(allowed_tools),
            agent_pool=agent_pool,
            orchestrator_agent_id=orchestrator_agent_id,
        )

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
        from .workflow_card_pages import WorkflowCardPageDelivery

        card_message_id: list[str | None] = [message_id]  # Mutable ref for card updates
        page_delivery = WorkflowCardPageDelivery(card_message_id)
        terminal_sent: list[bool] = [False]
        delivery_lock = (
            lifecycle_owner.delivery_lock
            if lifecycle_owner is not None
            else threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        )
        project_id = getattr(project, "project_id", "") or ""
        origin_message_id = self._resolve_origin(message_id)

        def on_progress(
            card_data: dict[str, Any] | list[dict[str, Any]],
        ) -> None:
            """Update the progress card in Feishu."""
            with delivery_lock:
                if terminal_sent[0] or (lifecycle_owner is not None and lifecycle_owner.stop_event.is_set()):
                    logger.debug("Ignored workflow progress update after terminal card was sent")
                    return
                try:
                    # Keep runtime progress cards aligned with the mixin
                    # implementation: users can stop active workflows from the card,
                    # but terminal completion cards are delivered separately.
                    status_card = card_data[0] if isinstance(card_data, list) and card_data else card_data
                    if isinstance(status_card, dict):
                        self._inject_workflow_stop_button(
                            status_card,
                            chat_id,
                            project_id,
                            is_running=self._workflow_is_running_for_card(chat_id, project),
                        )
                    delivery_result = page_delivery.deliver(
                        card_data,
                        replace_or_send=self._replace_or_send_workflow_rendered_card,
                        chat_id=chat_id,
                        origin_message_id=origin_message_id,
                        # Heartbeats are periodic. Turning every PATCH failure
                        # into a new message creates an unbounded card stream
                        # during persistent transport or provenance failures.
                        status_fallback_to_new=False,
                    )
                    if delivery_result.status_message_id:
                        card_message_id[0] = delivery_result.status_message_id
                except Exception:
                    logger.debug(
                        "Failed to update workflow progress card",
                        exc_info=True,
                    )

        def _deliver_terminal_project(
            wf_project: "WorkflowProject",
            *,
            allow_stopped_owner: bool = False,
        ) -> None:
            """Deliver one authoritative terminal snapshot on stable pages."""
            report_status: dict[str, Any] | None = None
            failed_page_indexes: tuple[int, ...] = ()
            try:
                from ...workflow_engine.renderer import render_completion_cards

                card_data = render_completion_cards(wf_project)
                delivery_result = page_delivery.deliver(
                    card_data,
                    replace_or_send=self._replace_or_send_workflow_rendered_card,
                    chat_id=chat_id,
                    origin_message_id=origin_message_id,
                    terminal=True,
                )
                if delivery_result.status_message_id:
                    card_message_id[0] = delivery_result.status_message_id
                if delivery_result.fully_delivered:
                    return

                failed_page_indexes = delivery_result.failed_page_indexes
                report_status = self._send_workflow_completion_report(
                    wf_project=wf_project,
                    chat_id=chat_id,
                    message_id=message_id,
                    project=project,
                )
                if (
                    not allow_stopped_owner
                    and lifecycle_owner is not None
                    and lifecycle_owner.stop_event.is_set()
                ):
                    return

                # Retry every page once after the durable full report has
                # been generated. Existing page bindings are reused and
                # missing pages are created in order.
                retry_cards = render_completion_cards(
                    wf_project,
                    report_status=report_status,
                )
                retry_result = page_delivery.deliver(
                    retry_cards,
                    replace_or_send=self._replace_or_send_workflow_rendered_card,
                    chat_id=chat_id,
                    origin_message_id=origin_message_id,
                    terminal=True,
                )
                if retry_result.status_message_id:
                    card_message_id[0] = retry_result.status_message_id
                if retry_result.fully_delivered or report_status.get("attachment_sent"):
                    return
                failed_page_indexes = retry_result.failed_page_indexes
            except Exception:
                if (
                    not allow_stopped_owner
                    and lifecycle_owner is not None
                    and lifecycle_owner.stop_event.is_set()
                ):
                    return
                if report_status is None:
                    report_status = self._send_workflow_completion_report(
                        wf_project=wf_project,
                        chat_id=chat_id,
                        message_id=message_id,
                        project=project,
                    )
                logger.warning("Workflow terminal card delivery failed", exc_info=True)

            if report_status and report_status.get("attachment_sent"):
                self._reply_workflow_completion_fallback(
                    message_id=message_id,
                    report_status=report_status,
                    failed_page_indexes=failed_page_indexes,
                )
                return
            self._reply_workflow_completion_fallback(
                message_id=message_id,
                report_status=report_status,
                failed_page_indexes=failed_page_indexes,
            )

        def on_done(wf_project: "WorkflowProject") -> None:
            """Final completion — send a structured completion card."""
            # Wait for an in-flight PATCH, then fence every later progress
            # callback before report generation/upload and terminal delivery.
            with delivery_lock:
                if terminal_sent[0] or (lifecycle_owner is not None and lifecycle_owner.stop_event.is_set()):
                    return
                terminal_sent[0] = True
                _deliver_terminal_project(wf_project)

        def on_error(
            error_msg: str,
            wf_project: "WorkflowProject | None" = None,
        ) -> None:
            """Deliver an execution failure, or a project-less setup error."""
            with delivery_lock:
                if terminal_sent[0] or (lifecycle_owner is not None and lifecycle_owner.stop_event.is_set()):
                    return
                terminal_sent[0] = True
                if wf_project is not None:
                    _deliver_terminal_project(wf_project)
                    return

                from ...workflow_engine.errors import _strip_internal_details

                workflow_category = self._workflow_error_card_category(error_msg)
                card = self._build_error_card(
                    workflow_category,
                    detail=_strip_internal_details(error_msg or ""),
                )
                self._replace_or_send_workflow_card(
                    card_message_id=card_message_id[0],
                    chat_id=chat_id,
                    card=card,
                    origin_message_id=origin_message_id,
                )

        def on_cancelled(wf_project: "WorkflowProject") -> None:
            """Deliver the stopped run's final stream despite its stop fence."""
            with delivery_lock:
                if terminal_sent[0]:
                    return
                terminal_sent[0] = True
                _deliver_terminal_project(
                    wf_project,
                    allow_stopped_owner=True,
                )

        def on_log(msg: str) -> None:
            logger.debug("[WorkflowHandler] log: %s", msg)

        return WorkflowEngineCallbacks(
            on_progress=on_progress,
            on_done=on_done,
            on_cancelled=on_cancelled,
            on_error=on_error,
            on_log=on_log,
        )
