"""Workflow selection flow callback handlers (orchestrator + review steps).

Extracted from workflow.py to reduce handler size. These methods handle
all card callback interactions during the tool/model selection phase.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...card.render import model_cascade

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def workflow_model_name_is_available(
    model_name: str,
    available_models: list[dict[str, Any]],
) -> bool:
    """Return whether a model value is one exposed by the Workflow picker."""
    target = str(model_name or "").strip()
    if not target:
        return False
    if any(
        target == str(model.get("name") or "").strip()
        for model in available_models or []
    ):
        return True
    return any(
        target == str(variant.get("name") or "").strip()
        for group in model_cascade.build_model_groups(available_models or [])
        for variant in group.get("variants", [])
    )


class WorkflowSelectionMixin:
    """Mixin providing orchestrator/review selection callback handlers.

    NOTE: handle_workflow_{orchestrator,review}_select_{tool,model} are
    defined on WorkflowHandler directly (workflow.py) because they need
    _option handling from Feishu's select_static widget.
    """

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

        tool_name = value.get("tool_name", "")
        if not tool_name:
            self._reply_workflow_error(message_id, "invalid_argument", detail="缺少 tool_name")
            return

        # Security: validate tool_name against allowed list
        from ...workflow_engine.tool_registry import get_available_tools
        available_tools = get_available_tools(require_available=True)
        if tool_name not in available_tools:
            self._reply_workflow_error(message_id, "invalid_argument", detail=f"不支持的工具: {tool_name}")
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        # Use SelectionFlowController
        ctrl = self._get_selection_controller(
            project,
            session_key=stored_session_key,
        )
        ctrl.set_step(1)
        if "model_page" in value:
            try:
                model_page = int(value.get("model_page", 0) or 0)
            except (TypeError, ValueError):
                model_page = 0
            ctrl.set_model_page(tool_name, model_page, is_review=False)
        else:
            ctrl.toggle_tool_expand(tool_name, is_review=False)
        self._persist_selection_controller(project, ctrl)

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
            self._reply_workflow_error(message_id, "invalid_argument", detail=f"无效的工具名称: {tool_name}")
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
                self._reply_workflow_error(message_id, "invalid_argument", detail=f"无效的模型名称: {model_name}")
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

        ctrl = self._get_selection_controller(
            project,
            session_key=stored_session_key,
        )
        ctrl.set_step(1)
        # Orchestrator is single-select: clear before adding.
        ctrl.clear_selections(is_review=False)
        ctrl.add_or_update_selection(selection, is_review=False)
        self._persist_selection_controller(project, ctrl)

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

        ctrl = self._get_selection_controller(
            project,
            session_key=stored_session_key,
        )
        ctrl.set_step(1)
        ctrl.remove_selection(selection_key, is_review=False)
        self._persist_selection_controller(project, ctrl)

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

        ctrl = self._get_selection_controller(
            project,
            session_key=stored_session_key,
        )
        ctrl.set_step(1)
        ctrl.clear_selections(is_review=False)
        self._persist_selection_controller(project, ctrl)

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

        ctrl = self._get_selection_controller(
            project,
            session_key=stored_session_key,
        )
        ctrl.set_step(1)

        ok, err_msg = ctrl.validate_non_empty(is_review=False)
        if not ok:
            ctrl.error_message = err_msg
            self._persist_selection_controller(project, ctrl)
            requirement = engine.project.pending.requirement if engine.project.pending else ""
            self._send_combined_selection_card(
                message_id=message_id,
                chat_id=chat_id,
                project=project,
                requirement=requirement,
                session_key=stored_session_key,
                is_review=False,
            )
            return

        # Save orchestrator selection to pending state
        from src.spec_engine.review_agents import ReviewAgentBinding

        snapshot = ctrl.snapshot()
        orchestrator_items = list(snapshot.get("orchestrator_selections", {}).values())
        if orchestrator_items:
            first = orchestrator_items[0]
            engine.project.pending.orchestrator_binding = ReviewAgentBinding.from_dict(first)
            engine.project.pending.orchestrator_agent = first.get("tool_name", "")

        # Transition to review step
        engine.project.status = WorkflowStatus.AWAITING_TOOL_SELECT
        ctrl.set_step(2)
        self._persist_selection_controller(project, ctrl)

        requirement = engine.project.pending.requirement if engine.project.pending else ""
        self._send_combined_selection_card(
            message_id=message_id,
            chat_id=chat_id,
            project=project,
            requirement=requirement,
            session_key=stored_session_key,
            is_review=True,
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

        if engine.project.status != WorkflowStatus.AWAITING_TOOL_SELECT:
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

        tool_name = value.get("tool_name", "")
        if not tool_name:
            self._reply_workflow_error(message_id, "invalid_argument", detail="缺少 tool_name")
            return

        # Security: validate tool_name against allowed list
        from ...workflow_engine.tool_registry import get_available_tools
        available_tools = get_available_tools(require_available=True)
        if tool_name not in available_tools:
            self._reply_workflow_error(message_id, "invalid_argument", detail=f"不支持的工具: {tool_name}")
            return

        if project is None:
            self._reply_workflow_error(message_id, "invalid_state", detail="缺少项目上下文")
            return

        ctrl = self._get_selection_controller(
            project,
            session_key=stored_session_key,
        )
        ctrl.set_step(2)
        if "model_page" in value:
            try:
                model_page = int(value.get("model_page", 0) or 0)
            except (TypeError, ValueError):
                model_page = 0
            ctrl.set_model_page(tool_name, model_page, is_review=True)
        else:
            ctrl.toggle_tool_expand(tool_name, is_review=True)
        self._persist_selection_controller(project, ctrl)

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

        if engine.project.status != WorkflowStatus.AWAITING_TOOL_SELECT:
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
                "[workflow] Rejected unknown tool_name=%s at handle_workflow_review_select_model",
                tool_name,
            )
            self._reply_workflow_error(message_id, "invalid_argument", detail=f"无效的工具名称: {tool_name}")
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
                self._reply_workflow_error(message_id, "invalid_argument", detail=f"无效的模型名称: {model_name}")
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

        ctrl = self._get_selection_controller(
            project,
            session_key=stored_session_key,
        )
        ctrl.set_step(2)
        ctrl.add_or_update_selection(selection, is_review=True, keep_panel_open=True)
        self._persist_selection_controller(project, ctrl)

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

        ctrl = self._get_selection_controller(
            project,
            session_key=stored_session_key,
        )
        ctrl.set_step(2)
        ctrl.remove_selection(selection_key, is_review=True)
        self._persist_selection_controller(project, ctrl)

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

        ctrl = self._get_selection_controller(
            project,
            session_key=stored_session_key,
        )
        ctrl.set_step(2)
        ctrl.clear_selections(is_review=True)
        self._persist_selection_controller(project, ctrl)

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

        ctrl = self._get_selection_controller(
            project,
            session_key=stored_session_key,
        )
        ctrl.set_step(2)
        ctrl.set_review_auto_mode(not ctrl.review_auto_mode)
        self._persist_selection_controller(project, ctrl)

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

        if engine.project.status != WorkflowStatus.AWAITING_TOOL_SELECT:
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

        ctrl = self._get_selection_controller(
            project,
            session_key=stored_session_key,
        )
        ctrl.set_step(2)

        ok, err_msg = ctrl.validate_non_empty(is_review=True)
        if not ok:
            ctrl.error_message = err_msg
            self._persist_selection_controller(project, ctrl)
            requirement = engine.project.pending.requirement if engine.project.pending else ""
            self._send_combined_selection_card(
                message_id=message_id,
                chat_id=chat_id,
                project=project,
                requirement=requirement,
                session_key=stored_session_key,
                is_review=True,
            )
            return

        from src.spec_engine.review_agents import ReviewAgentBinding

        # Save review selections
        snapshot = ctrl.snapshot()
        review_items = list(snapshot.get("review_selections", {}).values())
        if review_items:
            engine.project.pending.review_agents = [ReviewAgentBinding.from_dict(item) for item in review_items]
        else:
            engine.project.pending.review_agents = []

        # Derive selected_tools from orchestrator + review selections
        orchestrator_tool = engine.project.pending.orchestrator_agent or ""
        review_tools = [a.tool_name for a in (engine.project.pending.review_agents or [])]
        all_selected = []
        for t in [orchestrator_tool] + review_tools:
            if t and t not in all_selected:
                all_selected.append(t)
        if all_selected:
            engine.project.pending.selected_tools = all_selected

        # Replace the selection card with a locked summary so it can't be re-clicked
        from ...card import CardBuilder
        selections_summary = []
        orch_agent = engine.project.pending.orchestrator_agent or ""
        orch_binding = engine.project.pending.orchestrator_binding
        orch_model = ""
        if orch_binding:
            orch_model = getattr(orch_binding, "model_name", "") or "(默认)"
        selections_summary.append(f"**主编排 Agent**: `{orch_agent}` {orch_model}")
        review_agents_list = engine.project.pending.review_agents or []
        if review_agents_list:
            review_lines = [f"  {i+1}. `{a.tool_name}` {a.model_name or '(默认)'}" for i, a in enumerate(review_agents_list)]
            selections_summary.append("**评审 Agent**:\n" + "\n".join(review_lines))
        elif ctrl.review_auto_mode:
            selections_summary.append("**评审 Agent**: Auto（沿用主 Agent）")
        locked_card = CardBuilder._wrap_card(
            header_title="✅ Workflow — 工具选择完成",
            header_template="green",
            elements=[{"tag": "markdown", "content": "\n\n".join(selections_summary)}],
        )
        self.update_card(message_id, locked_card)

        # Proceed to script generation without blocking the Feishu callback
        # thread on a long-running ACP/CLI model call.
        with engine._lock:
            if not engine.project or not engine.project.pending:
                return
            requirement = engine.project.pending.requirement
            engine.project.status = WorkflowStatus.GENERATING_SCRIPT
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
