"""Workflow Agent Pool selection helpers for WorkflowHandler."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...project import ProjectContext
    from .workflow import _WorkflowLifecycleOwner

logger = logging.getLogger(__name__)

_UNSET_WORKFLOW_MODEL_STATE = object()


def _workflow_module():
    """Return the workflow handler module for runtime-patchable symbols."""
    from . import workflow as _wf_mod

    return _wf_mod


class WorkflowSelectionMixin:
    """Mixin providing Agent Pool selection UI and action handling."""

    @staticmethod
    def _workflow_selected_option(value: dict[str, Any]) -> str | None:
        """Extract the trimmed option value from a Feishu card action payload."""
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
        """Fetch ACP models for a tool and resolve the cascaded selection state."""
        from ...acp.helper import fetch_acp_models

        models = fetch_acp_models(
            tool_name,
            cwd=root_path,
            current_model=None,
        )
        state = _workflow_module().WorkflowHandler._workflow_model_state_from_catalog(
            models=models,
            pending=pending,
            selected_model=selected_model,
            selected_profile=selected_profile,
            selected_effort=selected_effort,
        )
        return models, state

    @staticmethod
    def _workflow_model_state_from_catalog(
        *,
        models: list[Any],
        pending: Any,
        selected_model: str | None = None,
        selected_profile: str | None = None,
        selected_effort: str | None = None,
    ) -> Any:
        """Build the model-cascade selection state from an already-fetched catalog."""
        from ...card.render.model_cascade import resolve_model_cascade

        return resolve_model_cascade(
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

    @classmethod
    def _validated_workflow_draft_binding(
        cls,
        *,
        pending: Any,
        root_path: str,
        available_tools: dict[str, str],
        models: list[Any] | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        """Validate draft model/profile/effort against the backend's declared capabilities."""
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

        if models is None:
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
        """Verify every binding in pending.agent_pool still matches live tool capabilities."""
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
        model_state: Any = _UNSET_WORKFLOW_MODEL_STATE,
    ) -> dict[str, Any]:
        """Render the Agent Pool selection card payload for a pending workflow."""
        from ...workflow_engine.renderer import WorkflowAgentSelectionRenderer

        if model_state is _UNSET_WORKFLOW_MODEL_STATE:
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
        from ...acp.helper import kickoff_acp_model_preheat
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
        recommendation_order = ("traex", "claude", "codex", "grok", "dsh", "agy", "aiden", "gemini", "coco")
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
        kickoff_acp_model_preheat(list(available_tools), cwd=root_path)

        with engine._lock:
            if not engine.project:
                engine._project = WorkflowProject()
            if admission_owner is None:
                admission_owner = _workflow_module()._WorkflowLifecycleOwner(
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
        render_model_state: Any = _UNSET_WORKFLOW_MODEL_STATE

        with engine._lock:
            owner = vars(engine).get("_workflow_selection_owner")
            wf_project = engine.project
            pending = wf_project.pending if wf_project else None
            owner_valid = bool(
                isinstance(owner, _workflow_module()._WorkflowLifecycleOwner)
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
                tool_name = str(pending.draft_tool_name or "").strip().lower()
                if tool_name not in available_tools:
                    error = ("invalid_argument", "请先选择当前可用的工具。")
                elif model_name == "default":
                    pending.draft_model_name = None
                    pending.draft_profile = None
                    pending.draft_effort = None
                else:
                    models, state = self._workflow_selection_model_state(
                        tool_name=tool_name,
                        root_path=root_path,
                        pending=pending,
                        selected_model=model_name,
                        selected_profile="",
                        selected_effort="",
                    )
                    if model_name not in state.model_names:
                        error = ("invalid_argument", "所选模型族当前不可用。")
                        render_model_state = self._workflow_model_state_from_catalog(
                            models=models,
                            pending=pending,
                        )
                    else:
                        pending.draft_model_name = state.selected_model
                        pending.draft_profile = state.selected_profile
                        pending.draft_effort = state.selected_effort
                        render_model_state = state
            elif authorized and action == "workflow_select_profile":
                profile = str(selected_option or "").strip().lower()
                models, state = self._workflow_selection_model_state(
                    tool_name=pending.draft_tool_name or "",
                    root_path=root_path,
                    pending=pending,
                    selected_profile=profile,
                    selected_effort="",
                )
                if not pending.draft_model_name or profile not in state.profiles:
                    error = ("invalid_argument", "所选 Profile 不属于当前模型能力。")
                    render_model_state = self._workflow_model_state_from_catalog(
                        models=models,
                        pending=pending,
                    )
                else:
                    pending.draft_profile = state.selected_profile
                    pending.draft_effort = state.selected_effort
                    render_model_state = state
            elif authorized and action == "workflow_select_effort":
                effort = str(selected_option or "").strip().lower()
                models, state = self._workflow_selection_model_state(
                    tool_name=pending.draft_tool_name or "",
                    root_path=root_path,
                    pending=pending,
                    selected_effort=effort,
                )
                if not pending.draft_model_name or effort not in state.efforts:
                    error = ("invalid_argument", "所选 Effort 不属于当前模型能力。")
                    render_model_state = self._workflow_model_state_from_catalog(
                        models=models,
                        pending=pending,
                    )
                else:
                    pending.draft_effort = state.selected_effort
                    render_model_state = state
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
                    validation_models = None
                    if pending.draft_model_name:
                        (
                            validation_models,
                            render_model_state,
                        ) = self._workflow_selection_model_state(
                            tool_name=tool_name,
                            root_path=root_path,
                            pending=pending,
                        )
                    try:
                        model_name, profile, effort = self._validated_workflow_draft_binding(
                            pending=pending,
                            root_path=root_path,
                            available_tools=available_tools,
                            models=validation_models,
                        )
                    except ValueError as exc:
                        error = ("invalid_argument", str(exc))
                        if validation_models is not None:
                            pending.draft_model_name = (
                                render_model_state.selected_model or None
                            )
                            pending.draft_profile = render_model_state.selected_profile
                            pending.draft_effort = render_model_state.selected_effort
                        elif not pending.draft_model_name:
                            pending.draft_profile = None
                            pending.draft_effort = None
                    if error is None:
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
                    generation_owner = _workflow_module()._WorkflowLifecycleOwner(
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
                    model_state=render_model_state,
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
            model_state=render_model_state,
        )
        self._replace_or_send_workflow_rendered_card(
            card_message_id=message_id,
            chat_id=chat_id,
            card_data=card_data,
            fallback_to_new=False,
        )

