"""Spec Engine handler — start, status, pause, resume, stop, guide."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Optional

from ...acp.tool_discovery import AgentToolDiscovery, AgentToolOption
from ...card import CardBuilder
from ...card.actions.dispatch import (
    SPEC_REVIEW_SELECT_MODEL,
    SPEC_REVIEW_SELECT_TOOL,
)
from ...card.shared import build_responsive_layout
from ...card.themes import PANEL_STYLES, get_theme
from ...card.ui_text import UI_TEXT
from ...model_selection import DEFAULT_MODEL_OPTION_VALUE, is_default_model_option
from ...spec_engine.models import SpecProjectStatus
from ...spec_engine.review_agents import ReviewAgentBinding
from ...spec_engine.review_selection import SpecReviewSelectionController
from ...spec_engine.storage import SpecRunSummary, delete_spec_run, list_spec_runs, state_path_for_run
from ...spec_engine.task_persistence import list_pending_tasks, load_task_state
from ...tasking import TaskPriority, TaskSpec
from ...utils.errors import fmt_error, get_error_detail
from ...utils.text import generate_task_id
from ..emoji import EmojiReaction
from ..slash_command_parser import SlashCommandParser
from .base import CardActionContext
from .engine_base import BaseEngineHandler

if TYPE_CHECKING:
    from ...card.protocols import RendererProtocol
    from ...project import ProjectContext
    from ..handler_context import HandlerContext

logger = logging.getLogger(__name__)


class SpecHandler(BaseEngineHandler):
    """Manages the full lifecycle of Spec Engine tasks."""

    def __init__(self, ctx: "HandlerContext", renderer: "RendererProtocol | None" = None) -> None:
        super().__init__(ctx)
        if renderer is None:
            from ..renderers import get_renderer
            renderer = get_renderer("spec", self)
        self.renderer = renderer

    def _get_engine_manager(self):
        return self.ctx.spec_engine_manager

    def _get_engine_name_prefix(self) -> str:
        return "Spec"

    def _get_task_type(self) -> str:
        return "spec_engine"

    def _show_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        self.show_spec_status(message_id, chat_id, project)

    def _create_callbacks(
        self, message_id: str, chat_id: str, project: Optional["ProjectContext"], engine_name: str, root_path: str
    ):
        model_name = self._get_model_name(chat_id, project)
        return self.renderer.create_spec_callbacks(message_id, chat_id, project, engine_name=engine_name, model_name=model_name)

    def _get_model_name(self, chat_id: str, project: Optional["ProjectContext"]) -> str:
        """Resolve model name for subtitle display."""
        from ...utils.engine_identity import resolve_engine_identity
        project_id = project.project_id if project else None
        current_mode = self.ctx.mode_manager.get_mode(chat_id, project_id=project_id)
        identity = resolve_engine_identity(
            mode=current_mode,
            acp_tool_name=getattr(project, "acp_tool_name", None) if project else None,
            acp_model_name=getattr(project, "acp_model_name", None) if project else None,
        )
        return identity.model_name or ""

    def _get_spec_reporter(self):
        """Return the progress reporter used by _safe_execute_engine."""
        return self.ctx.progress_reporter

    def _agent_tool_discovery(self):
        discovery = getattr(self, "_spec_review_tool_discovery", None)
        if discovery is None:
            discovery = AgentToolDiscovery()
            self._spec_review_tool_discovery = discovery
        return discovery

    def _spec_review_selection_controller(self):
        ctrl = getattr(self, "_spec_review_selection_ctrl", None)
        if ctrl is None:
            ctrl = SpecReviewSelectionController()
            self._spec_review_selection_ctrl = ctrl
        return ctrl

    def _get_available_spec_review_tools(self) -> list[dict]:
        return self._agent_tool_discovery().get_available_tools()

    def _get_spec_review_models_for_tool(
        self,
        tool_name: str,
        provider: str = "acp",
        cwd: Optional[str] = None,
        current_model: Optional[str] = None,
    ) -> list[dict]:
        models = self._agent_tool_discovery().get_models_for_tool(
            tool_name, provider=provider, cwd=cwd, current_model=current_model
        )
        from ...acp.traex_selection import expand_model_option_dicts

        return expand_model_option_dicts(models)

    def _dispatch_spec_review_tool_select(
        self,
        *,
        message_id: str,
        chat_id: str,
        project: "ProjectContext",
        tools: list[dict],
        selected: list[dict] | None = None,
        message: str = "",
        select_action: str = SPEC_REVIEW_SELECT_TOOL,
        pending_tool: str = "",
        thread_root_id: str = "",
        patch_existing: bool = True,
        model_page: int = 0,
        page_tool_name: str = "",
        page_provider: str = "",
    ) -> None:
        _ = chat_id
        thread_root_id = self._spec_review_thread_root_id(thread_root_id, fallback_message_id=message_id)
        _, card_content = CardBuilder.build_spec_review_agent_select_card(
            tools=tools,
            selected=selected or [],
            project_id=project.project_id,
            message=message,
            select_action=select_action,
            pending_tool=pending_tool,
            thread_root_id=thread_root_id,
            model_page=model_page,
            page_tool_name=page_tool_name,
            page_provider=page_provider,
        )
        if patch_existing and self.update_card(message_id, card_content):
            return
        self.reply_card(message_id, card_content)

    def _patch_spec_review_starting_card(
        self,
        *,
        message_id: str,
        selected: list[dict] | None = None,
        auto: bool = False,
    ) -> None:
        _, card_content = CardBuilder.build_spec_review_starting_card(selected=selected or [], auto=auto)
        if not self.update_card(message_id, card_content):
            logger.debug("failed to patch Spec review starting card: message_id=%s", message_id)

    @staticmethod
    def _spec_review_value_thread_root(value: dict | None) -> str:
        return str((value or {}).get("thread_root_id") or "").strip()

    @staticmethod
    def _spec_review_thread_root_id(thread_root_id: str = "", *, fallback_message_id: str = "") -> str:
        if isinstance(thread_root_id, str) and thread_root_id.strip():
            return thread_root_id.strip()
        from ...thread import get_current_thread_id

        current = get_current_thread_id()
        if isinstance(current, str) and current.strip():
            return current.strip()
        return fallback_message_id

    def _start_spec_review_selection(
        self,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: "ProjectContext",
    ) -> None:
        ctrl = self._spec_review_selection_controller()
        state = ctrl.start_selection(project, goal=requirement)
        tools = self._get_available_spec_review_tools()
        if not tools:
            self._start_spec_engine_now(message_id, chat_id, requirement, project, review_agents=[])
            return
        try:
            self._dispatch_spec_review_tool_select(
                message_id=message_id,
                chat_id=chat_id,
                project=project,
                tools=tools,
                selected=[item.to_dict() for item in state.selected_items],
                message=UI_TEXT["spec_review_select_message"],
                patch_existing=False,
            )
        except Exception:
            logger.warning("Spec review selection card failed; starting without review agents", exc_info=True)
            self._start_spec_engine_now(message_id, chat_id, requirement, project, review_agents=[])

    def _on_engine_error(
        self,
        error: Exception,
        task_id: str,
        chat_id: str,
        message_id: str,
        project: Optional["ProjectContext"],
        engine_name: str,
        reporter,
        request_id: Optional[str],
        action_prefix: str = "spec",
    ) -> None:
        """Spec-specific error handling: uses renderer.build_error_card + send_card_to_chat."""
        err_msg = get_error_detail(error)
        root_path = getattr(project, "root_path", None) if project else None

        err_msg_type, err_card = self.renderer.build_error_card(
            project=project,
            engine_name=engine_name,
            error_msg=err_msg,
            project_id=project.project_id if project else None,
            engine_project_id=project.project_id if project else root_path,
            footer_note=self.format_ref_note(message_id, request_id) if request_id else None,
            error_details=f"engine={engine_name}; action=spec; request_id={request_id or 'n/a'}",
            detail_action={
                "action": "show_error_details",
                "engine_type": "spec",
                "request_id": request_id or "",
            },
            retry_action={
                "action": "spec_resume",
                "request_id": request_id or "",
            },
        )
        self.send_card_to_chat(chat_id, err_card, origin_message_id=message_id, request_id=request_id)

    def _refresh_card_view(self, message_id: str, chat_id: str, project=None):
        self.show_spec_status(message_id, chat_id, project, origin_message_id=message_id)

    def _get_ui_state(self, spec_project_id: str) -> dict:
        """Deprecated: Delegate to renderer"""
        return self.renderer.get_ui_state(spec_project_id)

    # ------------------------------------------------------------------
    # Command router
    # ------------------------------------------------------------------
    def handle_spec_command(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        match = SlashCommandParser.parse(text)
        command = match.command if match else ""
        args = match.args if match else ""

        if command == "/spec_recover" and not args:
            self.show_recoverable_tasks(message_id, chat_id)
        elif command == "/spec_recover":
            self.recover_spec_task(message_id, chat_id, args, project)
        elif command == "/spec_status":
            self.show_spec_status(message_id, chat_id, project)
        elif command == "/spec_history":
            self.show_spec_history(message_id, chat_id, text, project)
        elif command == "/spec_metrics":
            self.show_spec_metrics(message_id, chat_id, text, project)
        elif command == "/spec_config":
            self.show_spec_config(message_id, chat_id, project)
        elif command == "/spec_export":
            self.export_spec_report(message_id, chat_id, project)
        elif command == "/spec_save":
            self.save_spec_state(message_id, chat_id, project)
        elif command == "/stop_spec":
            self.stop_spec_engine(message_id, chat_id, project)
        elif command == "/spec_pause":
            self.pause_spec_engine(message_id, chat_id, project)
        elif command == "/spec_resume":
            self.resume_spec_engine(message_id, chat_id, project)
        elif command == "/spec_guide" and args:
            self.update_spec_guidance(message_id, chat_id, args, project)
        elif command == "/spec_guide":
            self.reply_text(
                message_id,
                UI_TEXT["spec_cmd_guide_usage"],
            )
        elif command == "/spec" and args:
            self.start_spec_engine(message_id, chat_id, args, project)
        elif command == "/spec":
            self.reply_text(
                message_id,
                UI_TEXT["spec_cmd_help_usage"],
            )
        else:
            self.reply_text(message_id, "❓ 未知的 Spec 命令")

    # ------------------------------------------------------------------
    # start
    # ------------------------------------------------------------------
    def start_spec_engine(
        self, message_id: str, chat_id: str, requirement: str, project: Optional["ProjectContext"] = None
    ):
        project = self._ensure_project(message_id, chat_id, project)
        if not project:
            return
        self._ensure_topic_engine_context(
            mode="spec",
            message_id=message_id,
            chat_id=chat_id,
            project=project,
        )

        root_path = project.root_path if project else self.get_working_dir(chat_id)

        existing = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if existing and existing.is_running:
            self.reply_text(
                message_id,
                "⚠️ 当前项目已有 Spec 任务在执行中\n\n发送 `/spec_status` 查看进度\n发送 `/stop_spec` 停止任务",
            )
            return

        self._start_spec_review_selection(message_id, chat_id, requirement, project)

    def _start_spec_engine_now(
        self,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: Optional["ProjectContext"] = None,
        *,
        review_agents: list[ReviewAgentBinding] | None = None,
    ):
        project = self._ensure_project(message_id, chat_id, project)
        if not project:
            return

        root_path = project.root_path if project else self.get_working_dir(chat_id)
        existing = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if existing and existing.is_running:
            self.reply_text(
                message_id,
                "⚠️ 当前项目已有 Spec 任务在执行中\n\n发送 `/spec_status` 查看进度\n发送 `/stop_spec` 停止任务",
            )
            return

        self.add_reaction(message_id, EmojiReaction.on_multi_task_start())

        request_id = self.ensure_request_id(
            message_id, chat_id=chat_id, project_id=(project.project_id if project else None)
        )
        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))

        # Spec renderer owns the first runtime card and task-card orchestration.
        # Visible plan items become first-class cards when the plan arrives;
        # later task/agent events are routed back to those cards.
        model_name = self._get_model_name(chat_id, project) or None
        engine = self.ctx.spec_engine_manager.get_or_create(
            chat_id,
            root_path,
            engine_name=engine_name,
            model_name=model_name,
        )
        if hasattr(engine, "set_review_agent_pool"):
            engine.set_review_agent_pool(review_agents or [])

        project_name = project.project_name if project else os.path.basename(root_path) or "spec"
        task_id = generate_task_id(project_name)

        _on_rate_limit = self.create_rate_limit_callback(
            chat_id, message_id, project, f"Spec({engine_name})", request_id
        )

        def run_spec_engine():
            callbacks = self.renderer.create_spec_callbacks(message_id, chat_id, project, engine_name, model_name=model_name or "")
            engine.execute(requirement, callbacks, task_id=task_id, on_rate_limit=_on_rate_limit)

        def _scheduled_run():
            self._safe_execute_engine(
                executor_func=run_spec_engine,
                task_id=task_id,
                chat_id=chat_id,
                message_id=message_id,
                project=project,
                engine_name=engine_name,
                reporter=self._get_spec_reporter(),
                request_id=request_id,
                action_prefix="spec",
                command_text=f"/spec {requirement}",
            )

        self._submit_engine_task(_scheduled_run, chat_id, message_id, project, request_id, task_id)

    def _spec_review_pending_requirement(self, project: "ProjectContext") -> str:
        state = getattr(project, "spec_review_selection_state", None)
        return str(getattr(state, "pending_goal", "") or "").strip()

    def handle_spec_review_use_auto(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str] = None,
        value: dict | None = None,
    ) -> None:
        project = self.project_manager.get_project_for_chat(project_id, chat_id) if project_id else self.project_manager.get_active_project(chat_id)
        if not project:
            self.reply_error(message_id, UI_TEXT["spec_review_project_not_found"])
            return
        requirement = self._spec_review_pending_requirement(project)
        if not requirement:
            self.reply_error(message_id, UI_TEXT["spec_cmd_help_usage"])
            return
        start_message_id = self._spec_review_thread_root_id(
            self._spec_review_value_thread_root(value),
            fallback_message_id=message_id,
        )
        self._patch_spec_review_starting_card(message_id=message_id, selected=[], auto=True)
        self._spec_review_selection_controller().reset_selection(project)
        self._start_spec_engine_now(start_message_id, chat_id, requirement, project, review_agents=[])

    def handle_spec_review_select_tool(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str] = None,
        value: dict | None = None,
    ) -> None:
        value = value or {}
        project = self.project_manager.get_project_for_chat(project_id, chat_id) if project_id else self.project_manager.get_active_project(chat_id)
        if not project:
            self.reply_error(message_id, UI_TEXT["spec_review_project_not_found"])
            return
        thread_root_id = self._spec_review_thread_root_id(
            self._spec_review_value_thread_root(value),
            fallback_message_id=message_id,
        )

        tool_name = value.get("_option") or value.get("tool_name", "")
        provider = value.get("provider", "")
        supports_model = value.get("supports_model", False)
        skip_model_selection = value.get("skip_model_selection", False)
        try:
            model_page = int(value.get("model_page", 0) or 0)
        except (TypeError, ValueError):
            model_page = 0
        if not tool_name:
            self.reply_error(message_id, UI_TEXT["spec_review_select_tool_error"])
            return

        ctrl = self._spec_review_selection_controller()
        option = AgentToolOption(
            provider=provider,
            tool_name=tool_name,
            display_name=value.get("display_name") or tool_name,
            agent_name=value.get("agent_name") or "",
            supports_model=bool(supports_model),
            model_optional=True,
            skip_model_selection=bool(skip_model_selection),
        )
        ctrl.select_tool(project, option)

        should_skip_model = not option.supports_model
        models: list[dict] = []
        if option.supports_model:
            current_model = None
            if project and getattr(project, "acp_tool_name", "") == tool_name:
                current_model = getattr(project, "acp_model_name", None)
            models = self._get_spec_review_models_for_tool(
                tool_name,
                provider=provider,
                cwd=project.root_path,
                current_model=current_model,
            )
            if not models or option.skip_model_selection:
                should_skip_model = True

        if not should_skip_model:
            model_tools = [{
                "id": DEFAULT_MODEL_OPTION_VALUE,
                "name": UI_TEXT["system_acp_default_model_option"],
                "description": UI_TEXT["system_acp_default_model_desc"],
                "use_default_model": True,
            }]
            for m in models:
                model_id = str(m.get("name") or "").strip()
                if not model_id:
                    continue
                display = str(m.get("display_name") or model_id).strip() or model_id
                blurb = str(m.get("description") or "").strip()
                if blurb and blurb != display and len(blurb) > 60:
                    blurb = blurb[:60].rstrip() + "…"
                elif blurb == display:
                    blurb = ""
                model_tools.append({"id": model_id, "name": display, "description": blurb})
            self._dispatch_spec_review_tool_select(
                message_id=message_id,
                chat_id=chat_id,
                project=project,
                tools=model_tools,
                selected=[item.to_dict() for item in ctrl._get_state(project).selected_items],
                message=UI_TEXT["spec_review_select_model_prompt"].format(tool=option.display_name),
                select_action=SPEC_REVIEW_SELECT_MODEL,
                pending_tool=option.display_name,
                thread_root_id=thread_root_id,
                model_page=model_page,
                page_tool_name=tool_name,
                page_provider=provider,
            )
            return

        model_name = None
        model_display = None
        if models:
            target = next((m for m in models if m.get("is_default")), models[0])
            model_name = target.get("name")
            model_display = target.get("display_name")
        _, _, msg = ctrl.add_pending_item(project, model_name=model_name, model_display_name=model_display)
        ctrl.back_to_tool_selection(project)
        self._dispatch_spec_review_tool_select(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            tools=self._get_available_spec_review_tools(),
            selected=[item.to_dict() for item in ctrl._get_state(project).selected_items],
            message=msg,
            thread_root_id=thread_root_id,
        )

    def handle_spec_review_select_model(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str] = None,
        value: dict | None = None,
    ) -> None:
        value = value or {}
        project = self.project_manager.get_project_for_chat(project_id, chat_id) if project_id else self.project_manager.get_active_project(chat_id)
        if not project:
            self.reply_error(message_id, UI_TEXT["spec_review_project_not_found"])
            return
        thread_root_id = self._spec_review_thread_root_id(
            self._spec_review_value_thread_root(value),
            fallback_message_id=message_id,
        )
        raw_model_name = (
            value.get("_option")
            or value.get("model_name")
            or value.get("id")
            or value.get("name")
            or value.get("tool_name")
            or None
        )
        use_default_model = bool(value.get("use_default_model")) or is_default_model_option(raw_model_name)
        model_name = None if use_default_model else raw_model_name
        model_display = None if use_default_model else (
            value.get("model_display_name") or value.get("display_name") or value.get("name") or model_name
        )
        ctrl = self._spec_review_selection_controller()
        _, _, msg = ctrl.add_pending_item(project, model_name=model_name, model_display_name=model_display)
        ctrl.back_to_tool_selection(project)
        self._dispatch_spec_review_tool_select(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            tools=self._get_available_spec_review_tools(),
            selected=[item.to_dict() for item in ctrl._get_state(project).selected_items],
            message=msg,
            thread_root_id=thread_root_id,
        )

    def handle_spec_review_remove_item(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str] = None,
        value: dict | None = None,
    ) -> None:
        value = value or {}
        project = self.project_manager.get_project_for_chat(project_id, chat_id) if project_id else self.project_manager.get_active_project(chat_id)
        if not project:
            self.reply_error(message_id, UI_TEXT["spec_review_project_not_found"])
            return
        thread_root_id = self._spec_review_thread_root_id(
            self._spec_review_value_thread_root(value),
            fallback_message_id=message_id,
        )
        ctrl = self._spec_review_selection_controller()
        _, _, msg = ctrl.remove_selected_item(project, str(value.get("selection_key") or value.get("_option") or ""))
        self._dispatch_spec_review_tool_select(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            tools=self._get_available_spec_review_tools(),
            selected=[item.to_dict() for item in ctrl._get_state(project).selected_items],
            message=msg,
            thread_root_id=thread_root_id,
        )

    def handle_spec_review_clear_items(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str] = None,
        value: dict | None = None,
    ) -> None:
        project = self.project_manager.get_project_for_chat(project_id, chat_id) if project_id else self.project_manager.get_active_project(chat_id)
        if not project:
            self.reply_error(message_id, UI_TEXT["spec_review_project_not_found"])
            return
        thread_root_id = self._spec_review_thread_root_id(
            self._spec_review_value_thread_root(value),
            fallback_message_id=message_id,
        )
        ctrl = self._spec_review_selection_controller()
        _, _, msg = ctrl.clear_selected_items(project)
        self._dispatch_spec_review_tool_select(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            tools=self._get_available_spec_review_tools(),
            selected=[item.to_dict() for item in ctrl._get_state(project).selected_items],
            message=msg,
            thread_root_id=thread_root_id,
        )

    def handle_spec_review_menu(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str] = None,
        value: dict | None = None,
    ) -> None:
        project = self.project_manager.get_project_for_chat(project_id, chat_id) if project_id else self.project_manager.get_active_project(chat_id)
        if not project:
            self.reply_error(message_id, UI_TEXT["spec_review_project_not_found"])
            return
        thread_root_id = self._spec_review_thread_root_id(
            self._spec_review_value_thread_root(value),
            fallback_message_id=message_id,
        )
        ctrl = self._spec_review_selection_controller()
        ctrl.back_to_tool_selection(project)
        self._dispatch_spec_review_tool_select(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            tools=self._get_available_spec_review_tools(),
            selected=[item.to_dict() for item in ctrl._get_state(project).selected_items],
            message=UI_TEXT["spec_review_select_message"],
            thread_root_id=thread_root_id,
        )

    def handle_spec_review_finish_selection(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str] = None,
        value: dict | None = None,
    ) -> None:
        project = self.project_manager.get_project_for_chat(project_id, chat_id) if project_id else self.project_manager.get_active_project(chat_id)
        if not project:
            self.reply_error(message_id, UI_TEXT["spec_review_project_not_found"])
            return
        start_message_id = self._spec_review_thread_root_id(
            self._spec_review_value_thread_root(value),
            fallback_message_id=message_id,
        )
        ctrl = self._spec_review_selection_controller()
        state = ctrl._get_state(project)
        requirement = self._spec_review_pending_requirement(project)
        if not requirement:
            self.reply_error(message_id, UI_TEXT["spec_cmd_help_usage"])
            return
        state = ctrl.finalize_selection(project)
        if not state.selected_items:
            self.reply_error(message_id, UI_TEXT["spec_review_no_selection_error"])
            return
        review_agents = [
            ReviewAgentBinding.from_selection_item(item)
            for item in state.selected_items
        ]
        self._patch_spec_review_starting_card(
            message_id=message_id,
            selected=[item.to_dict() for item in state.selected_items],
        )
        self._start_spec_engine_now(start_message_id, chat_id, requirement, project, review_agents=review_agents)

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------
    def show_spec_status(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        origin_message_id: Optional[str] = None,
    ):
        # User command "/spec_status" resets to status view
        if project is None:
            project = self.project_manager.get_active_project(chat_id)

        root_path = project.root_path if project else self.get_working_dir(chat_id)
        spec_project_id = project.project_id if project else root_path

        self.renderer.update_ui_state(spec_project_id, view_mode="status", view_context={})

        engine = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if not engine or not engine.project:
            runs = list_spec_runs(root_path, self.settings)
            if runs:
                self._show_spec_cache_status(message_id, chat_id, project, root_path, runs)
                return

        self.renderer.render_current_view(message_id, chat_id, project, origin_message_id)

    def _show_spec_cache_status(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"],
        root_path: str,
        runs: list[SpecRunSummary],
    ) -> None:
        latest = runs[0]
        theme_color = getattr(project, "theme_color", None) if project else None
        theme = get_theme(theme_color or "blue")
        cache_root = os.path.dirname(os.path.dirname(latest.run_dir))
        elements = [
            CardBuilder._build_directory_element(project, root_path),
            {"tag": "hr"},
            {
                "tag": "markdown",
                "content": (
                    "📋 **Spec 状态**\n\n"
                    "**Spec 缓存任务**\n"
                    f"- 发现任务: `{len(runs)}` 个\n"
                    f"- 缓存根: `{cache_root}`"
                ),
            },
        ]
        for index, run in enumerate(runs[:5], start=1):
            elements.append(self._build_spec_cache_run_panel(run, project, root_path, index=index))
        if len(runs) > 5:
            elements.append({
                "tag": "markdown",
                "content": f"... 还有 {len(runs) - 5} 个历史任务未展示",
                "text_size": "notation",
            })
        elements.append({
            "tag": "markdown",
            "content": "恢复或删除操作已放在对应任务面板内；暂停/待澄清任务恢复后会继续执行。",
            "text_size": "notation",
        })
        card = CardBuilder._wrap_card(
            CardBuilder._build_header_title(project),
            theme.header_template,
            elements,
        )
        card_content = json.dumps(card, ensure_ascii=False)
        self.reply_card(message_id, card_content)

    @staticmethod
    def _format_timestamp(ts: float) -> str:
        try:
            if not ts:
                return ""
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
        except Exception:
            return ""

    def _build_spec_cache_run_panel(
        self,
        run: SpecRunSummary,
        project: Optional["ProjectContext"],
        root_path: str,
        *,
        index: int,
    ) -> dict:
        status = run.status or "未知"
        cycle = f"{run.current_cycle}/{run.total_cycles}" if run.total_cycles else str(run.current_cycle or 0)
        updated = self._format_timestamp(run.saved_at or run.created_at) or "未知"
        requirement = (run.requirement or "").strip()
        if len(requirement) > 120:
            requirement = requirement[:120] + "..."
        background_style, border_color = self._spec_cache_panel_style(index)
        detail_lines = [
            f"**任务 ID**：`{run.run_id}`",
            f"**更新时间**：{updated}",
            f"**状态**：{status}",
            f"**Cycle**：{cycle}",
        ]
        if requirement:
            detail_lines.append(f"**目标**：{requirement}")
        if not run.state_path:
            detail_lines.append("**恢复状态**：仅发现目录，缺少可恢复 `state.json`")

        buttons: list[dict] = []
        if run.state_path:
            buttons.append(
                self._build_spec_cache_action_button(
                    text="🔄 恢复",
                    button_type="primary",
                    action="spec_restore_run",
                    run_id=run.run_id,
                    project=project,
                    root_path=root_path,
                )
            )
        buttons.append(
            self._build_spec_cache_action_button(
                text="🗑 删除",
                button_type="danger",
                action="spec_delete_run",
                run_id=run.run_id,
                project=project,
                root_path=root_path,
            )
        )

        return {
            "tag": "collapsible_panel",
            "expanded": True,
            "header": {
                "title": {"tag": "markdown", "content": f"**{run.run_id}** · {status} · cycle {cycle}"},
                "vertical_align": "center",
                "icon": {
                    "tag": "standard_icon",
                    "token": "down-small-ccm_outlined",
                    "size": "16px 16px",
                },
                "icon_position": "follow_text",
                "icon_expanded_angle": -180,
            },
            "border": {"color": border_color, "corner_radius": PANEL_STYLES["corner_radius"]},
            "vertical_spacing": PANEL_STYLES["vertical_spacing"],
            "padding": PANEL_STYLES["padding_standard"],
            "elements": [
                {
                    "tag": "column_set",
                    "flex_mode": "stretch",
                    "background_style": background_style,
                    "columns": [
                        {
                            "tag": "column",
                            "width": "weighted",
                            "weight": 1,
                            "vertical_align": "center",
                            "elements": [
                                {"tag": "markdown", "content": "\n".join(detail_lines), "text_align": "left"}
                            ],
                        }
                    ],
                },
                *build_responsive_layout(buttons, mobile_force_vertical=False),
            ],
        }

    @staticmethod
    def _spec_cache_panel_style(index: int) -> tuple[str, str]:
        styles = (
            ("wathet", "wathet"),
            ("grey", PANEL_STYLES["border_normal"]),
            ("yellow", "orange"),
            ("green", "green"),
            ("blue", PANEL_STYLES["border_history"]),
        )
        return styles[(index - 1) % len(styles)]

    @staticmethod
    def _build_spec_cache_action_button(
        *,
        text: str,
        button_type: str,
        action: str,
        run_id: str,
        project: Optional["ProjectContext"],
        root_path: str,
    ) -> dict:
        return {
            "tag": "button",
            "text": {"tag": "plain_text", "content": text},
            "type": button_type,
            "behaviors": [{
                "type": "callback",
                "value": {
                    "action": action,
                    "project_id": project.project_id if project else "",
                    "deep_project_id": root_path,
                    "run_id": run_id,
                },
            }],
        }

    def show_spec_history(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        root_path = project.root_path if project else self.get_working_dir(chat_id)
        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
        try:
            engine = self.ctx.spec_engine_manager.load_or_create_from_disk(chat_id, root_path, engine_name=engine_name)
        except Exception:
            engine = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if not engine or not engine.project:
            msg_type, card_content = CardBuilder.build_info_card(
                project=project,
                title="🗂️ Spec 历史",
                content="当前没有可查询的 Spec 历史（未运行过或未落盘）\n\n发送 `/spec <需求>` 启动后会自动生成历史。",
                engine_name=f"Spec({engine_name})",
                show_buttons=False,
            )
            self.reply_card(message_id, card_content)
            return

        tail = 20
        try:
            parts = (text or "").strip().split()
            if len(parts) >= 2 and parts[1].isdigit():
                tail = max(1, min(500, int(parts[1])))
        except Exception:
            tail = 20
        content = self.ctx.spec_reporter.format_history(engine.project, tail=tail)
        msg_type, card_content = CardBuilder.build_info_card(
            project=project,
            title="🗂️ Spec 历史",
            content=content,
            engine_name=f"Spec({engine.engine_name})",
            show_buttons=False,
        )
        self.reply_card(message_id, card_content)

    def show_spec_metrics(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        root_path = project.root_path if project else self.get_working_dir(chat_id)
        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
        try:
            engine = self.ctx.spec_engine_manager.load_or_create_from_disk(chat_id, root_path, engine_name=engine_name)
        except Exception:
            engine = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if not engine or not engine.project:
            msg_type, card_content = CardBuilder.build_info_card(
                project=project,
                title="📈 Spec 指标",
                content="当前没有可查询的 Spec 指标（未运行过或未落盘）\n\n发送 `/spec <需求>` 启动后会自动记录指标。",
                engine_name=f"Spec({engine_name})",
                show_buttons=False,
            )
            self.reply_card(message_id, card_content)
            return

        tail = 20
        try:
            parts = (text or "").strip().split()
            if len(parts) >= 2 and parts[1].isdigit():
                tail = max(1, min(500, int(parts[1])))
        except Exception:
            tail = 20
        content = self.ctx.spec_reporter.format_metrics(engine.project, tail=tail)
        msg_type, card_content = CardBuilder.build_info_card(
            project=project,
            title="📈 Spec 指标",
            content=content,
            engine_name=f"Spec({engine.engine_name})",
            show_buttons=False,
        )
        self.reply_card(message_id, card_content)

    def show_spec_config(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        s = self.settings
        content = (
            "🧩 **Spec 长程配置**\n\n"
            f"- max_cycles: `{getattr(s, 'spec_max_cycles', None)}` (limit=`{getattr(s, 'spec_max_cycles_limit', None)}`) — 最大执行轮次\n"
            f"- min_cycles: `{getattr(s, 'spec_min_cycles', None)}` — 最少执行轮次\n"
            f"- execution_timeout: `{getattr(s, 'spec_execution_timeout', None)}` (秒) — 单次执行总超时\n"
            f"- convergence_window: `{getattr(s, 'spec_convergence_window', None)}` — 收敛检测窗口轮数\n"
            f"- infinite_mode: `{getattr(s, 'spec_infinite_mode', None)}` — 无限模式\n"
            f"- disable_convergence: `{getattr(s, 'spec_disable_convergence', None)}` — 禁用收敛检测\n"
            f"- disable_early_stop: `{getattr(s, 'spec_disable_early_stop', None)}` — 禁用提前停止\n"
            f"- rebuild_session_between_cycles: `{getattr(s, 'spec_rebuild_session_between_cycles', None)}` — 轮间重建会话\n"
            "\n🔍 **发现与收敛**\n\n"
            f"- discovery_enabled: `{getattr(s, 'spec_discovery_enabled', None)}` — 启用发现阶段\n"
            f"- discovery_max_questions: `{getattr(s, 'spec_discovery_max_questions', None)}` — 发现阶段最大问题数\n"
            f"- discovery_force_nonempty: `{getattr(s, 'spec_discovery_force_nonempty', None)}` — 强制非空发现\n"
            f"- discovery_gate_on_satisfied: `{getattr(s, 'spec_discovery_gate_on_satisfied', None)}` — 满足条件时门控\n"
            f"- discovery_max_pending: `{getattr(s, 'spec_discovery_max_pending', None)}` — 最大待处理发现数\n"
            f"- discovery_cooldown_cycles: `{getattr(s, 'spec_discovery_cooldown_cycles', None)}` — 发现冷却轮次\n"
            f"- generated_specs_per_cycle: `{getattr(s, 'spec_generated_specs_per_cycle', None)}` — 每轮生成规格数\n"
            f"- generated_specs_retention: `{getattr(s, 'spec_generated_specs_retention', None)}` — 生成规格保留数\n"
            "\n💾 **持久化与压缩**\n\n"
            f"- cache_root: `{getattr(s, 'spec_cache_root', None) or '~/.cache/ghostAp'}` — Spec 过程文件根目录\n"
            f"- state_file: `{getattr(s, 'spec_state_filename', None)}` — 状态文件名\n"
            f"- state_cycles_tail: `{getattr(s, 'spec_state_cycles_tail', None)}` — 状态文件仅保留最近 N 轮\n"
            f"- state_work_items_tail: `{getattr(s, 'spec_state_work_items_tail', None)}` — 工作项保留最近 N 条\n"
            f"- state_metrics_tail: `{getattr(s, 'spec_state_metrics_tail', None)}` — 指标保留最近 N 条\n"
            f"- artifacts_dir: `{getattr(s, 'spec_artifacts_dirname', None)}` — 产物目录名\n"
            f"- cycle_artifact_retention: `{getattr(s, 'spec_cycle_artifact_retention', None)}` — 产物保留轮数\n"
            f"- cycle_output_max_chars: `{getattr(s, 'spec_cycle_output_max_chars', None)}` — 单轮输出截断字符数\n"
            f"- cycle_tasks_max: `{getattr(s, 'spec_cycle_tasks_max', None)}` — 单轮最大任务数\n"
            f"- phase_output_persist_max_chars: `{getattr(s, 'spec_phase_output_persist_max_chars', None)}` — 阶段产物持久化截断\n"
            f"- persist_phase_artifacts: `{getattr(s, 'spec_persist_phase_artifacts', None)}` — 启用阶段产物持久化\n"
            f"- persist_every_phase: `{getattr(s, 'spec_persist_every_phase', None)}` — 每阶段均持久化\n"
            f"- allow_resume_from_disk: `{getattr(s, 'spec_allow_resume_from_disk', None)}` — 允许从磁盘恢复\n"
            f"- history_log: `{getattr(s, 'spec_history_log_filename', None)}` — 历史日志文件名\n"
            "\n⛔ **终止与失败**\n\n"
            f"- backlog_stuck_window: `{getattr(s, 'spec_backlog_stuck_window', None)}` — 积压卡住窗口 (0=禁用)\n"
            f"- success_ignore_backlog: `{getattr(s, 'spec_success_ignore_backlog', None)}` — 成功时忽略积压\n"
            f"- max_retries: `{getattr(s, 'spec_max_retries', None)}` — 最大重试次数\n"
            f"- max_consecutive_failures: `{getattr(s, 'spec_max_consecutive_failures', None)}` — 最大连续失败\n"
            f"- model_switch_enabled: `{getattr(s, 'spec_model_switch_enabled', None)}` — 启用模型切换\n"
            f"- failed_task_id_override: `{getattr(s, 'spec_failed_task_id_override', None) or '(空)'}` — 失败任务覆盖\n"
            "\n🔄 **审查与重试**\n\n"
            f"- review_enabled: `{getattr(s, 'spec_review_enabled', None)}` — 启用审查\n"
            f"- review_timeout: `{getattr(s, 'spec_review_timeout', None)}` (秒) — 审查超时\n"
            f"- review_min_timeout: `{getattr(s, 'spec_review_min_timeout', None)}` (秒) — 审查最小超时\n"
            f"- review_hard_floor: `{getattr(s, 'spec_review_hard_floor', None)}` (秒) — 超时硬下限\n"
            f"- review_max_parallel: `{getattr(s, 'spec_review_max_parallel', None)}` — 最大并行审查数\n"
            f"- retry_max_attempts: `{getattr(s, 'spec_review_retry_max_attempts', None)}` (0=禁用重试) — 重试次数\n"
            f"- retry_max_delay: `{getattr(s, 'spec_review_retry_max_delay', None)}` (秒) — 重试最大延迟\n"
            "\n🛡️ **熔断器**\n\n"
            f"- circuit_enabled: `{getattr(s, 'spec_review_failure_circuit_enabled', None)}` — 启用熔断\n"
            f"- max_consecutive: `{getattr(s, 'spec_review_failure_max_consecutive', None)}` — 连续失败触发阈值\n"
            f"- cooldown_cycles: `{getattr(s, 'spec_review_failure_cooldown_cycles', None)}` — 冷却轮次\n"
            f"- max_cooldown_cycles: `{getattr(s, 'spec_review_failure_max_cooldown_cycles', None)}` — 最大冷却轮次\n"
        )
        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
        msg_type, card_content = CardBuilder.build_info_card(
            project=project,
            title="🧩 Spec 配置",
            content=content,
            engine_name=f"Spec({engine_name})",
            show_buttons=False,
        )
        self.reply_card(message_id, card_content)

    def export_spec_report(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)

        root_path = project.root_path if project else self.get_working_dir(chat_id)
        engine = self.ctx.spec_engine_manager.get(chat_id, root_path)

        # Try to load from disk if not in memory
        if not engine:
            try:
                engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
                engine = self.ctx.spec_engine_manager.load_or_create_from_disk(
                    chat_id, root_path, engine_name=engine_name
                )
            except Exception:
                logger.debug("failed to load engine from disk", exc_info=True)

        if not engine or not engine.project or not engine.project.cycles:
            self.reply_text(message_id, "❌ 当前没有可导出的 Spec 记录")
            return

        spec_project = engine.project
        lines = [f"# Spec Project Export: {spec_project.name}\n"]
        lines.append(f"**Generated at**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Status**: {spec_project.status.value}")
        lines.append(f"**Requirement**: {spec_project.requirement}\n")

        lines.append("## Acceptance Criteria")
        tracker = spec_project.criteria_tracker
        for i, c in enumerate(tracker.criteria):
            mark = "✅" if tracker.satisfied.get(i) else "🔲"
            lines.append(f"- {mark} {c}")
        lines.append("")

        # Latest successful artifacts
        latest_cycle = spec_project.current_cycle
        if latest_cycle:
            lines.append(f"## Latest Cycle (Cycle {latest_cycle.cycle_number})")

            if latest_cycle.spec_content:
                lines.append("### Functional Spec")
                lines.append(latest_cycle.spec_content)
                lines.append("")

            if latest_cycle.plan_content:
                lines.append("### Implementation Plan")
                lines.append(latest_cycle.plan_content)
                lines.append("")

            if latest_cycle.review_result:
                lines.append("### Review Result")
                lines.append(
                    self.ctx.spec_reporter.format_review_result(latest_cycle.review_result, latest_cycle.cycle_number)
                )
                lines.append("")

        # Save to file
        export_filename = f"spec_export_{spec_project.project_id}_{int(time.time())}.md"
        export_path = os.path.join(root_path, export_filename)
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.reply_text(message_id, f"✅ 导出成功: `{export_path}`")
        except Exception as e:
            self.reply_text(message_id, f"❌ 导出失败: {get_error_detail(e)}")

    def save_spec_state(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        root_path = project.root_path if project else self.get_working_dir(chat_id)
        engine = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if not engine:
            engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
            engine = self.ctx.spec_engine_manager.get_or_create(chat_id, root_path, engine_name=engine_name)
        if not engine or not engine.project:
            self.reply_text(message_id, "当前没有可保存的 Spec 状态（请先运行 /spec）")
            return
        try:
            path = engine.save_state()
            self.reply_text(message_id, f"✅ 已保存 Spec 状态到: `{path}`")
        except Exception as e:
            self.reply_text(message_id, fmt_error("保存 Spec 状态", e))

    # ------------------------------------------------------------------
    # pause / resume / stop
    # ------------------------------------------------------------------
    def pause_spec_engine(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        def _pause():
            self._pause_engine_generic(
                message_id, chat_id, project, status_paused_enum=SpecProjectStatus.PAUSED
            )
            root_path = project.root_path if project else self.get_working_dir(chat_id)
            engine = self._get_engine_manager().get(chat_id, root_path)
            if not engine:
                engine = self._get_engine_manager().get_active_engine(chat_id)
            if engine and engine.project:
                try:
                    engine.save_state()
                except Exception:
                    logger.debug("failed to save engine state on pause", exc_info=True)

        self._safe_lifecycle_action(_pause, "pause", chat_id, message_id, project)

    def resume_spec_engine(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        def _resume():
            if project is None:
                proj = self.project_manager.get_active_project(chat_id)
            else:
                proj = project

            root_path = proj.root_path if proj else self.get_working_dir(chat_id)
            manager = self._get_engine_manager()
            engine = manager.get(chat_id, root_path)

            if not engine or not engine.project:
                try:
                    engine_name = self.get_engine_name(chat_id, project_id=(proj.project_id if proj else None))
                    engine = manager.load_or_create_from_disk(
                        chat_id, root_path, engine_name=engine_name
                    )
                except Exception:
                    logger.debug("failed to load engine from disk on resume", exc_info=True)

            if not engine:
                paused = [
                    e
                    for e in manager.list_engines(chat_id)
                    if e.project and e.project.status in (SpecProjectStatus.PAUSED, SpecProjectStatus.CLARIFYING)
                ]
                if len(paused) == 1:
                    engine = paused[0]

            if (
                engine
                and engine.project
                and engine.project.status in (SpecProjectStatus.PAUSED, SpecProjectStatus.CLARIFYING)
            ):
                callbacks = self._create_callbacks(
                    message_id, chat_id, proj, engine.engine_name, engine.root_path
                )

                def run_resume():
                    engine.resume(callbacks)

                def _locked_resume():
                    run_resume()

                def _scheduled_resume():
                    self._run_with_repo_lock_or_conflict_card(
                        root_path, chat_id, _locked_resume, message_id, "/spec_resume",
                    )

                request_id = self.ensure_request_id(
                    message_id, chat_id=chat_id, project_id=(proj.project_id if proj else None)
                )
                queue_key = f"{chat_id}:{self._get_task_type()}:{proj.project_id if proj else root_path}"

                spec = TaskSpec(
                    chat_id=chat_id,
                    queue_key=queue_key,
                    name=f"{self._get_task_type()}_resume",
                    task_type=self._get_task_type(),
                    project_id=proj.project_id if proj else None,
                    message_id=message_id,
                    origin_message_id=message_id,
                    request_id=request_id,
                    priority=TaskPriority.HIGH,
                )
                handle = self.scheduler.submit(spec, lambda ctx: _scheduled_resume())
                try:
                    self.ctx.message_linker.link_task(message_id, handle.run_id)
                except Exception as e:
                    logger.debug(
                        "link_task失败(%s_resume): message_id=%s, run_id=%s, err=%s",
                        self._get_task_type(),
                        message_id,
                        handle.run_id,
                        e,
                    )
                self._show_status(message_id, chat_id, project=proj)
            else:
                self.reply_text(message_id, f"当前没有可恢复的 {self._get_engine_name_prefix()} 任务")

        self._safe_lifecycle_action(_resume, "resume", chat_id, message_id, project)

    def stop_spec_engine(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        def _stop():
            proj = project or self.project_manager.get_active_project(chat_id)
            logger.info(
                "Spec stop requested: chat_id=%s message_id=%s project_id=%s",
                chat_id,
                message_id,
                proj.project_id if proj else None,
            )
            self._stop_engine_generic(message_id, chat_id, project)
            if project is None:
                proj = self.project_manager.get_active_project(chat_id)
            else:
                proj = project
            root_path = proj.root_path if proj else self.get_working_dir(chat_id)
            engine = self._get_engine_manager().get(chat_id, root_path)
            if not engine:
                active = self._get_engine_manager().get_active_engines(chat_id)
                if len(active) == 1:
                    engine = active[0]
            if engine and engine.project:
                try:
                    engine.save_state()
                except Exception:
                    logger.debug("failed to save engine state on stop", exc_info=True)

        self._safe_lifecycle_action(_stop, "stop", chat_id, message_id, project)

    # ------------------------------------------------------------------
    # guidance
    # ------------------------------------------------------------------
    def update_spec_guidance(
        self, message_id: str, chat_id: str, guide_message: str, project: Optional["ProjectContext"] = None
    ):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)

        engine = None
        if project:
            engine = self.ctx.spec_engine_manager.get(chat_id, project.root_path)

        if not engine:
            # Fallback: if there's exactly one runnable engine in the chat
            candidates = [
                e
                for e in self.ctx.spec_engine_manager.list_engines(chat_id)
                if e.project
                and e.project.status
                in (
                    SpecProjectStatus.RUNNING,
                    SpecProjectStatus.PAUSED,
                    SpecProjectStatus.CLARIFYING,
                )
            ]
            if len(candidates) == 1:
                engine = candidates[0]

        if not engine or not engine.project:
            self.reply_text(
                message_id,
                "⚠️ 当前没有可注入引导的 Spec 任务（运行中/已暂停/待澄清）\n\n"
                "请先使用 `/spec <需求>` 启动，或发送 `/spec_status` 查看当前任务",
            )
            return

        if engine.project.status not in (
            SpecProjectStatus.RUNNING,
            SpecProjectStatus.PAUSED,
            SpecProjectStatus.CLARIFYING,
        ):
            self.reply_text(
                message_id,
                "⚠️ 当前 Spec 任务状态不支持注入引导（仅支持：运行中/已暂停/待澄清）\n\n发送 `/spec_status` 查看状态",
            )
            return

        reporter = self.ctx.spec_reporter
        engine_name = engine.engine_name

        # 尝试用 LLM 将原始需求与新引导合并为新的综合目标
        success, result = engine.refine_goal_with_guidance(guide_message)

        if success:
            content = reporter.format_goal_rewritten(guide_message, result)
            title = reporter.get_goal_rewritten_title()
        else:
            # LLM 重写失败，退化为临时注入（不改变持久目标）
            engine.inject_guidance(guide_message)
            content = reporter.format_guidance_injected(guide_message)
            title = reporter.get_guidance_injected_title()

        msg_type, card_content = CardBuilder.build_info_card(
            project=project,
            title=title,
            content=content,
            engine_name=f"Spec({engine_name})",
            show_buttons=False,
        )
        self.send_card_to_chat(chat_id, card_content)

    # ------------------------------------------------------------------
    # recover
    # ------------------------------------------------------------------
    def show_recoverable_tasks(self, message_id: str, chat_id: str):
        tasks = list_pending_tasks()
        if not tasks:
            self.reply_text(message_id, "📋 没有可恢复的任务")
            return

        lines = ["📋 **可恢复的 Spec 任务**\n"]
        for t in tasks:
            import time as _time

            created_str = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(t.created_at))
            req_summary = t.requirement[:50] + "..." if len(t.requirement) > 50 else t.requirement
            lines.append(f"**{t.task_id}**")
            lines.append(f"- 需求: {req_summary}")
            lines.append(f"- 创建时间: {created_str}")
            if t.last_error:
                error_summary = t.last_error[:80] + "..." if len(t.last_error) > 80 else t.last_error
                lines.append(f"- 最后错误: {error_summary}")
            lines.append("")

        lines.append("使用 `/spec_recover <任务ID>` 恢复指定任务")
        self.reply_text(message_id, "\n".join(lines))

    def recover_spec_task(
        self, message_id: str, chat_id: str, task_id: str, project: Optional["ProjectContext"] = None
    ):
        state = load_task_state(task_id)
        if not state:
            root_path = project.root_path if project else self.get_working_dir(chat_id)
            run_state_path = state_path_for_run(root_path, self.settings, task_id)
            if run_state_path and os.path.isfile(run_state_path):
                self.restore_spec_run(message_id, chat_id, task_id, project=project)
                return
            self.reply_text(message_id, f"❌ 未找到任务: {task_id}")
            return

        project_path = state.project_path
        if not os.path.isdir(project_path):
            self.reply_text(message_id, f"❌ 项目路径不存在: {project_path}")
            return

        if not project:
            try:
                project, _ = self.project_manager.get_or_create_project_for_path(project_path, chat_id)
            except Exception as e:
                self.reply_text(message_id, fmt_error("恢复项目上下文", e))
                return

        existing = self.ctx.spec_engine_manager.get(chat_id, project_path)
        if existing and existing.is_running:
            self.reply_text(
                message_id,
                "⚠️ 当前项目已有 Spec 任务在执行中\n\n发送 `/spec_status` 查看进度\n发送 `/stop_spec` 停止任务",
            )
            return

        self.add_reaction(message_id, EmojiReaction.on_multi_task_start())

        request_id = self.ensure_request_id(
            message_id, chat_id=chat_id, project_id=(project.project_id if project else None)
        )
        runtime = state.resolved_runtime_context()
        engine_name = state.resolved_engine_name()
        reporter = self.ctx.spec_reporter

        content = reporter.format_analyzing_start(state.requirement)
        title = f"🔄 恢复任务 {task_id}"
        msg_type, card_content = CardBuilder.build_info_card(
            project=project,
            title=title,
            content=f"{content}\n\n{self.format_ref_note(message_id, request_id)}" if request_id else content,
            engine_name=f"Spec({engine_name})",
            show_buttons=False,
        )
        self.reply_card(
            message_id, card_content
        )

        engine = self.ctx.spec_engine_manager.get_or_create(
            chat_id,
            project_path,
            engine_name=engine_name,
            agent_type=runtime.get("agent_type"),
            model_name=runtime.get("model_name") or runtime.get("current_model"),
        )

        on_rate_limit = self.create_rate_limit_callback(
            chat_id, message_id, project, f"Spec({engine_name})", request_id
        )
        try:
            engine.restore_from_task_state(state, on_rate_limit=on_rate_limit)
        except Exception as e:
            logger.warning("恢复任务上下文失败(task_id=%s): %s", task_id, get_error_detail(e), exc_info=True)
            self.reply_text(message_id, fmt_error("恢复任务上下文", e))
            return

        def run_spec_engine():
            try:
                callbacks = self.renderer.create_spec_callbacks(message_id, chat_id, project, engine_name, model_name=self._get_model_name(chat_id, project))
                # Use resume() instead of execute() to preserve state
                # The execute() method re-initializes the project, wiping previous progress.
                engine.resume(callbacks)
            except Exception as e:
                if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                    logger.warning("Spec Engine 恢复超时 (task_id=%s): %s", task_id, get_error_detail(e))
                else:
                    logger.error("Spec Engine 恢复执行异常: %s", e, exc_info=True)

                err_msg = get_error_detail(e)

                err_msg_type, err_card = self.renderer.build_error_card(
                    project=project,
                    engine_name=engine_name,
                    error_msg=err_msg,
                    project_id=project.project_id if project else None,
                    engine_project_id=project.project_id if project else project_path,
                    footer_note=self.format_ref_note(message_id, request_id) if request_id else None,
                )
                self.send_card_to_chat(chat_id, err_card, origin_message_id=message_id, request_id=request_id)

        def _locked_recover():
            run_spec_engine()

        def _scheduled_recover():
            self._run_with_repo_lock_or_conflict_card(
                project_path, chat_id, _locked_recover, message_id, f"/spec_recover {task_id}",
            )

        self._submit_engine_task(
            _scheduled_recover, chat_id, message_id, project, request_id, task_id,
            name_suffix="recover",
        )

    def restore_spec_run(
        self,
        message_id: str,
        chat_id: str,
        run_id: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        run_id = (run_id or "").strip()
        if not run_id:
            self.reply_text(message_id, "❌ 还原失败：缺少 run_id")
            return
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        root_path = project.root_path if project else self.get_working_dir(chat_id)
        state_path = state_path_for_run(root_path, self.settings, run_id)
        if not state_path or not os.path.isfile(state_path):
            self.reply_text(message_id, f"❌ 未找到 Spec 状态: {run_id}")
            return

        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
        try:
            engine = self.ctx.spec_engine_manager.load_or_create_from_state_file(
                chat_id,
                root_path,
                state_path,
                engine_name=engine_name,
            )
        except Exception as e:
            logger.warning("Spec run 还原失败(run_id=%s): %s", run_id, get_error_detail(e), exc_info=True)
            self.reply_text(message_id, fmt_error("还原 Spec 状态", e))
            return

        if not engine or not engine.project:
            self.reply_text(message_id, f"❌ Spec 状态不可用: {run_id}")
            return

        if engine.project.status == SpecProjectStatus.RUNNING and not engine.is_running:
            engine.project.status = SpecProjectStatus.PAUSED
            try:
                engine.save_state()
            except Exception:
                logger.debug("failed to save interrupted Spec state before restore", exc_info=True)

        if engine.project.status in (SpecProjectStatus.PAUSED, SpecProjectStatus.CLARIFYING):
            self.resume_spec_engine(message_id, chat_id, project)
            return

        self.renderer.update_ui_state(project.project_id if project else root_path, view_mode="status", view_context={})
        self.renderer.render_current_view(message_id, chat_id, project, origin_message_id=message_id)

    # ------------------------------------------------------------------
    # UI Interaction Handlers
    # ------------------------------------------------------------------
    def handle_card_action(self, open_message_id: str, open_chat_id: str, action_type: str, value: dict):
        """Handle spec_* card actions."""
        project_id = value.get("project_id", "")
        # Note: Spec engine uses 'deep_project_id' key for compatibility/convention with base templates,
        # but in Spec context it might be root_path or project_id.
        spec_project_id = value.get("deep_project_id", "")

        # Resolve target project (chat-scoped to prevent cross-chat leakage)
        target_project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
        if not target_project and spec_project_id:
            try:
                if os.path.isabs(spec_project_id):
                    target_project = self.project_manager.find_project_by_path(spec_project_id, chat_id=open_chat_id)
                else:
                    target_project = self.project_manager.get_project_for_chat(spec_project_id, open_chat_id)
            except Exception:
                logger.debug("failed to get target_project", exc_info=True)

        spec_actions = {
            "spec_pause": self.pause_spec_engine,
            "spec_resume": self.resume_spec_engine,
            "spec_stop": self.stop_spec_engine,
        }

        # Try dispatching standard actions first
        if self._dispatch_standard_card_action(CardActionContext(
            open_message_id=open_message_id,
            open_chat_id=open_chat_id,
            action_type=action_type,
            value=value,
            prefix="spec",
            action_map=spec_actions,
            toggle_log_method=self._toggle_log,
            switch_mode_method=self._switch_card_mode,
            toggle_ac_method=self._toggle_ac,
            project=target_project,
        )):
            return

        # Custom actions (non-standard)
        if action_type == "spec_skip_retry":
            # Skip retry wait without cancelling the entire engine
            engine = self.ctx.spec_engine_manager.get_active_engine(open_chat_id)
            if engine and hasattr(engine, 'skip_retry_event'):
                engine.skip_retry_event.set()
                self.reply_text(open_message_id, UI_TEXT["skip_retry_ack"])
            else:
                self.reply_text(open_message_id, UI_TEXT["no_active_retry"])
            return

        if action_type == "spec_retry":
            task_id = (value.get("task_id") or "").strip()
            if not task_id:
                self.reply_text(open_message_id, "❌ 重试失败：缺少 task_id")
                return
            # Reuse /spec_recover flow to resume from persisted failed-task snapshot.
            self.recover_spec_task(open_message_id, open_chat_id, task_id, project=target_project)
            return

        if action_type == "spec_restore_run":
            run_id = (value.get("run_id") or "").strip()
            self.restore_spec_run(open_message_id, open_chat_id, run_id, project=target_project)
            return

        if action_type == "spec_delete_run":
            run_id = (value.get("run_id") or "").strip()
            self.delete_spec_run_cache(open_message_id, open_chat_id, run_id, project=target_project)
            return

    def delete_spec_run_cache(
        self,
        message_id: str,
        chat_id: str,
        run_id: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        run_id = (run_id or "").strip()
        if not run_id:
            self.reply_text(message_id, "❌ 删除失败：缺少 run_id")
            return
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        root_path = project.root_path if project else self.get_working_dir(chat_id)

        engine = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if engine and engine.project and engine.project.project_id == run_id and engine.is_running:
            self.reply_text(message_id, f"⚠️ Spec 任务仍在运行，不能删除缓存: {run_id}\n\n请先发送 `/stop_spec` 停止任务。")
            return

        deleted = delete_spec_run(root_path, self.settings, run_id)
        if not deleted:
            self.reply_text(message_id, f"❌ 未找到 Spec 缓存任务: {run_id}")
            return

        self.reply_text(message_id, f"🧹 已删除 Spec 缓存任务: {run_id}")
