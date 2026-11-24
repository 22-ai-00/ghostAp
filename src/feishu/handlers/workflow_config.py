"""Interactive Workflow Teamwork configuration (/wf_config) mixin."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...project import ProjectContext

logger = logging.getLogger(__name__)

_WF_CONFIG_MAX_SESSIONS = 256
_WF_CONFIG_SESSIONS_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

_PREFERRED_TOOL_ORDER: tuple[str, ...] = (
    "traex",
    "claude",
    "codex",
    "agy",
    "dsh",
    "grok",
    "aiden",
    "gemini",
    "coco",
    "claude_w",
    "codex_w",
)


def _ui_role(role: str | None) -> str:
    cleaned = str(role or "").strip().lower()
    if cleaned in {"verifier", "reviewer", "auditor", "judge", "adversarial_verifier"}:
        return "verifier"
    if cleaned in {"scout", "explorer"}:
        return "scout"
    if cleaned in {"planner", "orchestrator", "architect", "lead"}:
        return "planner"
    if cleaned in {"synthesizer", "synth", "summarizer"}:
        return "synthesizer"
    return "worker"


def _next_agent_id(members: tuple[Any, ...]) -> str:
    used = {m.agent_id for m in members}
    idx = 1
    while f"A{idx}" in used:
        idx += 1
    return f"A{idx}"


class WorkflowConfigMixin:
    """Mixin providing `/wf_config`, interactive Card 2.0 team editor, and sync callbacks."""

    def _ensure_wf_config_store(
        self,
    ) -> tuple[
        OrderedDict[str, Any],
        OrderedDict[tuple[str, str, str], str],
    ]:
        sessions = getattr(self, "_wf_config_sessions", None)
        msg_index = getattr(self, "_wf_config_message_index", None)
        if not isinstance(sessions, OrderedDict):
            sessions = OrderedDict()
            self._wf_config_sessions = sessions
        if not isinstance(msg_index, OrderedDict):
            msg_index = OrderedDict()
            self._wf_config_message_index = msg_index
        return sessions, msg_index

    def _ensure_wf_config_locks(self) -> OrderedDict[str, Any]:
        locks = getattr(self, "_wf_config_session_locks", None)
        if not isinstance(locks, OrderedDict):
            locks = OrderedDict()
            self._wf_config_session_locks = locks
        return locks

    def _acquire_wf_config_session_lock(
        self,
        *,
        config_session_id: str,
        chat_id: str,
        project_id: str,
        message_id: str,
    ) -> Any:
        from ...utils.lock_order import LockLevel, ordered_lock

        with _WF_CONFIG_SESSIONS_LOCK:
            _sessions, msg_index = self._ensure_wf_config_store()
            locks = self._ensure_wf_config_locks()
            resolved_sid = config_session_id
            if not resolved_sid and message_id:
                resolved_sid = msg_index.get((chat_id, project_id, message_id), "") or msg_index.get(
                    (chat_id, "", message_id), ""
                )
                if not resolved_sid:
                    for (m_chat, _m_pid, m_msg), mapped_id in msg_index.items():
                        if m_chat == chat_id and m_msg == message_id and mapped_id:
                            resolved_sid = mapped_id
                            break
            if resolved_sid:
                lock_key = f"sid:{resolved_sid}"
            elif message_id:
                lock_key = f"msg:{chat_id}:{message_id}"
            else:
                lock_key = f"chat:{chat_id}:{project_id}"

            lock = locks.get(lock_key)
            if lock is None:
                lock = ordered_lock(LockLevel.ACP_ACTIVATION, name=f"wf_config:{lock_key}")
                locks[lock_key] = lock
            locks.move_to_end(lock_key)
            while len(locks) > _WF_CONFIG_MAX_SESSIONS:
                oldest_key, oldest_lock = next(iter(locks.items()))
                if oldest_key == lock_key or oldest_lock.locked():
                    break
                locks.popitem(last=False)
            return lock

    def _get_wf_config_state(
        self,
        *,
        config_session_id: str,
        chat_id: str,
        project_id: str,
        message_id: str,
    ) -> Any:
        with _WF_CONFIG_SESSIONS_LOCK:
            sessions, msg_index = self._ensure_wf_config_store()
            if config_session_id and config_session_id in sessions:
                cached = sessions[config_session_id]
                if cached.chat_id == chat_id and (
                    not project_id
                    or not cached.project_id
                    or cached.project_id == project_id
                ):
                    sessions.move_to_end(config_session_id)
                    return cached
            if message_id:
                mapped_id = msg_index.get((chat_id, project_id, message_id))
                if mapped_id and mapped_id in sessions:
                    cached = sessions[mapped_id]
                    if cached.chat_id == chat_id and (
                        not project_id
                        or not cached.project_id
                        or cached.project_id == project_id
                    ):
                        sessions.move_to_end(mapped_id)
                        return cached
        return None

    def _put_wf_config_state(
        self,
        state: Any,
        *,
        message_id: str = "",
    ) -> None:
        with _WF_CONFIG_SESSIONS_LOCK:
            sessions, msg_index = self._ensure_wf_config_store()
            sessions[state.config_session_id] = state
            sessions.move_to_end(state.config_session_id)
            while len(sessions) > _WF_CONFIG_MAX_SESSIONS:
                sessions.popitem(last=False)
            if message_id:
                key = (state.chat_id, state.project_id or "", message_id)
                msg_index[key] = state.config_session_id
                msg_index.move_to_end(key)
                while len(msg_index) > _WF_CONFIG_MAX_SESSIONS:
                    msg_index.popitem(last=False)

    @staticmethod
    def _load_workflow_capabilities(
        root_path: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Merge cached ToolCapability snapshots with authoritative available tools."""
        from ...workflow_engine.tool_registry import (
            EFFORT_CAPABLE_TOOLS,
            ToolCapability,
            get_available_tools,
            get_tool_capabilities,
        )

        raw_caps = get_tool_capabilities(cwd=root_path)
        avail_tools = get_available_tools(require_available=True)

        merged: dict[str, ToolCapability] = {}
        ordered_names = list(
            dict.fromkeys(
                [
                    *(name for name in _PREFERRED_TOOL_ORDER if name in raw_caps or name in avail_tools),
                    *raw_caps.keys(),
                    *avail_tools.keys(),
                ]
            )
        )
        for name in ordered_names:
            cap = raw_caps.get(name)
            if avail_tools:
                is_avail = name in avail_tools
                desc = str(avail_tools.get(name) or (cap.description if cap else "") or name)
                if cap is not None:
                    merged[name] = ToolCapability(
                        tool_name=cap.tool_name,
                        display_name=cap.display_name,
                        available=is_avail,
                        supports_model=cap.supports_model,
                        supports_effort=cap.supports_effort,
                        models=cap.models,
                        efforts_by_model=cap.efforts_by_model,
                        default_effort=cap.default_effort,
                        profiles_by_model=cap.profiles_by_model,
                        description=desc,
                    )
                else:
                    merged[name] = ToolCapability(
                        tool_name=name,
                        display_name=name.capitalize(),
                        available=is_avail,
                        supports_model=True,
                        supports_effort=(name in EFFORT_CAPABLE_TOOLS),
                        description=desc,
                    )
            elif cap is not None:
                merged[name] = cap

        return merged, dict(avail_tools)

    @staticmethod
    def _default_members_for_preset(
        capabilities: dict[str, Any],
        *,
        preset_id: str = "collaborative",
    ) -> tuple[Any, ...]:
        """Build an initial role-aligned team roster from available capabilities."""
        from ...workflow_engine.agent_pool import WorkflowAgentBinding, freeze_agent_pool

        avail_names = [name for name, cap in capabilities.items() if cap.available]
        if not avail_names:
            avail_names = [name for name in _PREFERRED_TOOL_ORDER if name in capabilities][:3]
        if not avail_names:
            avail_names = list(capabilities.keys())[:3]
        if not avail_names:
            return ()

        max_count = 2 if preset_id == "quick" else (4 if preset_id == "deep_collab" else 3)
        chosen = avail_names[:max_count]

        if len(chosen) == 1:
            roles = ("worker",)
        elif len(chosen) == 2:
            roles = ("worker", "verifier")
        elif preset_id == "review_team":
            roles = ("worker", "verifier", "verifier")
        elif preset_id == "deep_collab" and len(chosen) >= 4:
            roles = ("scout", "planner", "worker", "verifier")
        else:
            roles = ("scout", "worker", "verifier", "synthesizer")

        bindings: list[WorkflowAgentBinding] = []
        for idx, tool_name in enumerate(chosen):
            cap = capabilities[tool_name]
            role = roles[min(idx, len(roles) - 1)]
            model_name = cap.models[0] if getattr(cap, "models", ()) else None
            effort = cap.default_effort if getattr(cap, "supports_effort", False) else None
            bindings.append(
                WorkflowAgentBinding(
                    agent_id=f"A{idx + 1}",
                    tool_name=tool_name,
                    model_name=model_name,
                    display_name=cap.display_name,
                    effort=effort,
                    role=role,
                    roles=(role,),
                )
            )
        return freeze_agent_pool(bindings, require_nonempty=False)

    def _build_initial_config_state(
        self,
        *,
        chat_id: str,
        project: Optional["ProjectContext"],
        preset_override: str | None = None,
        fallback_preset: str | None = None,
        prefer_fallback_over_saved: bool = False,
        requirement: str | None = None,
        config_session_id: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        from ...card.builders.workflow import (
            WORKFLOW_PRESET_CATALOG,
            WorkflowConfigEditorState,
        )
        from ...project import get_project_workflow_team_config
        from ...thread import get_current_thread_id
        from ...workflow_engine.agent_pool import WorkflowAgentBinding, freeze_agent_pool

        root_path = self._get_root_path(chat_id, project)
        capabilities, _ = self._load_workflow_capabilities(root_path)
        saved_cfg = get_project_workflow_team_config(project, chat_id)

        valid_presets = {p[0] for p in WORKFLOW_PRESET_CATALOG} | {
            "recommended",
            "deep_collab",
        }
        cleaned_fallback = (
            fallback_preset
            if fallback_preset in valid_presets
            else None
        )

        members: tuple[WorkflowAgentBinding, ...] = ()
        preset_id = preset_override or cleaned_fallback or "collaborative"
        orchestrator_id = ""
        saved_as_default = False

        saved_preset = (
            str(saved_cfg.get("preset_id") or "collaborative").strip().lower()
            if saved_cfg
            else None
        )
        use_saved_cfg = bool(
            saved_cfg
            and preset_override is None
            and not (
                prefer_fallback_over_saved
                and cleaned_fallback is not None
                and cleaned_fallback != saved_preset
            )
        )
        if use_saved_cfg and saved_cfg is not None:
            raw_bindings = saved_cfg.get("bindings") or []
            parsed = [
                b
                for item in raw_bindings
                if (b := WorkflowAgentBinding.from_dict(item)) is not None
            ]
            if parsed:
                try:
                    members = freeze_agent_pool(parsed, require_nonempty=False)
                except ValueError:
                    members = ()
                if members:
                    preset_id = str(saved_cfg.get("preset_id") or "collaborative")
                    orchestrator_id = str(
                        saved_cfg.get("orchestrator_agent_id") or members[0].agent_id
                    )
                    if orchestrator_id not in {m.agent_id for m in members}:
                        orchestrator_id = members[0].agent_id
                    saved_as_default = True

        if not members:
            members = self._default_members_for_preset(capabilities, preset_id=preset_id)
            orchestrator_id = members[0].agent_id if members else ""

        first = members[0] if members else None
        avail_first_tool = next(
            (name for name, cap in capabilities.items() if cap.available),
            next(iter(capabilities), "codex"),
        )
        state = WorkflowConfigEditorState(
            config_session_id=config_session_id or uuid.uuid4().hex[:16],
            chat_id=chat_id,
            project_id=(project.project_id if project else None),
            thread_root_id=get_current_thread_id(),
            preset_id=preset_id,
            members=members,
            orchestrator_agent_id=orchestrator_id,
            selected_agent_id=(first.agent_id if first else "__new__"),
            editor_role=_ui_role(first.role if first else "worker"),
            editor_tool=(first.tool_name if first else avail_first_tool),
            editor_model=(first.model_name if first else None),
            editor_profile=(first.profile if first else None),
            editor_effort=(first.effort if first else None),
            requirement=(requirement.strip() if isinstance(requirement, str) and requirement.strip() else None),
            saved_as_default=saved_as_default,
        )
        return state, capabilities

    @staticmethod
    def _get_pending_workflow_requirement(
        project: Optional["ProjectContext"],
        chat_id: str,
        config_session_id: str,
    ) -> str | None:
        from ...project import get_project_workflow_pending_requirement

        if project is None or not chat_id or not config_session_id:
            return None
        return get_project_workflow_pending_requirement(
            project,
            config_session_id,
            chat_id=chat_id,
        )

    def _persist_pending_workflow_requirement(
        self,
        project: Optional["ProjectContext"],
        chat_id: str,
        config_session_id: str,
        requirement: str | None,
    ) -> None:
        if project is None or not chat_id or not config_session_id:
            return
        pm = getattr(getattr(self, "ctx", None), "project_manager", None)
        commit_fn = getattr(pm, "commit_workflow_pending_requirement", None) if pm is not None else None
        if (
            pm is not None
            and not type(pm).__module__.startswith("unittest.mock")
            and callable(commit_fn)
        ):
            commit_fn(
                project,
                config_session_id,
                requirement,
                chat_id=chat_id,
            )
            return
        mutator = getattr(project, "replace_workflow_pending_requirement_for_chat", None)
        if callable(mutator):
            mutator(config_session_id, requirement, chat_id=chat_id)

    def show_workflow_config(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        *,
        preset_id: str | None = None,
        requirement: str | None = None,
    ) -> None:
        """Open the interactive Card 2.0 Teamwork configuration editor (`/wf` or `/wf_config`)."""
        from ...card.builders.workflow import build_workflow_config_card

        project = self._ensure_project(message_id, chat_id, project)
        if not project:
            return

        state, capabilities = self._build_initial_config_state(
            chat_id=chat_id,
            project=project,
            preset_override=preset_id,
            requirement=requirement,
        )
        if state.requirement:
            self._persist_pending_workflow_requirement(
                project,
                chat_id,
                state.config_session_id,
                state.requirement,
            )
        card = build_workflow_config_card(state, capabilities)
        sent_id = self.reply_card(message_id, card)
        self._put_wf_config_state(state, message_id=str(sent_id or message_id))

    @staticmethod
    def _resolve_editor_cascade(
        state: Any,
        capabilities: dict[str, Any],
    ) -> tuple[Any, str | None, str | None, str | None, str]:
        from ...card.builders.workflow import (
            WORKFLOW_ROLE_CATALOG,
            capability_to_acp_options,
        )
        from ...card.render.model_cascade import resolve_model_cascade
        from ...workflow_engine.tool_registry import ToolCapability

        tool = str(state.editor_tool or "").strip().lower()
        cap = capabilities.get(tool) or ToolCapability(
            tool_name=tool or "codex",
            display_name=tool or "Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
        )
        opts = capability_to_acp_options(cap, current_model=state.editor_model)
        cascade = resolve_model_cascade(
            opts,
            current_model=state.editor_model,
            selected_model=state.editor_model,
            selected_profile=state.editor_profile,
            selected_effort=state.editor_effort,
        )
        resolved_model = cascade.selected_model or state.editor_model
        resolved_profile = cascade.selected_profile if cascade.profiles else None
        resolved_effort = (
            (cascade.selected_effort if cascade.efforts else None)
            if cap.supports_effort
            else None
        )
        valid_roles = {role_id for role_id, _ in WORKFLOW_ROLE_CATALOG}
        resolved_role = state.editor_role if state.editor_role in valid_roles else "worker"
        return cap, resolved_model, resolved_profile, resolved_effort, resolved_role

    def _build_validated_editor_binding(
        self,
        state: Any,
        capabilities: dict[str, Any],
        *,
        agent_id: str,
    ) -> tuple[Any | None, str | None]:
        from ...acp.agy_selection import is_agy_orchestrator_model
        from ...workflow_engine.agent_pool import WorkflowAgentBinding
        from ...workflow_engine.tool_registry import validate_agent_binding_spec

        cap, resolved_model, resolved_profile, resolved_effort, resolved_role = (
            self._resolve_editor_cascade(state, capabilities)
        )
        raw_effort = str(state.editor_effort or "").strip().lower() or None
        raw_profile = str(state.editor_profile or "").strip().lower() or None
        effort_to_validate = (
            (raw_effort or resolved_effort)
            if cap.supports_effort
            else None
        )
        supports_profile = bool(
            cap.supports_effort
            and (
                resolved_profile is not None
                or cap.tool_name == "traex"
                or (
                    cap.tool_name == "agy"
                    and resolved_model is not None
                    and is_agy_orchestrator_model(resolved_model)
                )
            )
        )
        profile_to_validate = (
            (raw_profile or resolved_profile)
            if supports_profile
            else None
        )
        ok, err = validate_agent_binding_spec(
            cap.tool_name,
            model_name=resolved_model,
            profile=profile_to_validate,
            effort=effort_to_validate,
            capabilities=capabilities,
        )
        if not ok:
            return None, err or "成员配置无效"
        return (
            WorkflowAgentBinding(
                agent_id=agent_id,
                tool_name=cap.tool_name,
                model_name=resolved_model,
                display_name=cap.display_name,
                profile=profile_to_validate,
                effort=effort_to_validate,
                role=resolved_role,
                roles=(resolved_role,),
            ),
            None,
        )

    def _apply_config_transition(
        self,
        message_id: str,
        chat_id: str,
        project_id: str | None,
        raw_value: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str, str, dict[str, Any] | None]:
        """Pure state-machine transition shared by sync ACK and action_registry."""
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(raw_value or {})
        action = str(value.get("action") or "")
        pid = str(project_id or value.get("project_id") or "").strip() or None
        project = self._resolve_project_from_id(pid, chat_id) if pid else None
        if project is None:
            try:
                project = self.ctx.project_manager.get_active_project(chat_id)
            except Exception:
                project = None

        root_path = self._get_root_path(chat_id, project)
        capabilities, _ = self._load_workflow_capabilities(root_path)

        cfg_sid = str(value.get("config_session_id") or "").strip()
        effective_pid = project.project_id if project else (pid or "")
        session_lock = self._acquire_wf_config_session_lock(
            config_session_id=cfg_sid,
            chat_id=chat_id,
            project_id=effective_pid,
            message_id=message_id,
        )
        with session_lock:
            return self._apply_config_transition_locked(
                message_id=message_id,
                chat_id=chat_id,
                effective_pid=effective_pid,
                project=project,
                value=value,
                action=action,
                cfg_sid=cfg_sid,
                capabilities=capabilities,
            )

    def _apply_config_transition_locked(
        self,
        *,
        message_id: str,
        chat_id: str,
        effective_pid: str,
        project: Optional["ProjectContext"],
        value: dict[str, Any],
        action: str,
        cfg_sid: str,
        capabilities: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str, str, dict[str, Any] | None]:
        from ...card.actions import dispatch as action_ids
        from ...card.builders.workflow import (
            WORKFLOW_PRESET_CATALOG,
            WORKFLOW_ROLE_CATALOG,
            WorkflowConfigEditorState,
            build_workflow_config_card,
        )
        from ...workflow_engine.agent_pool import freeze_agent_pool
        from ...workflow_engine.constants import MAX_WORKFLOW_AGENT_POOL_SIZE
        from ...workflow_engine.tool_registry import validate_agent_binding_spec

        state = self._get_wf_config_state(
            config_session_id=cfg_sid,
            chat_id=chat_id,
            project_id=effective_pid,
            message_id=message_id,
        )
        if state is None:
            raw_payload_preset = str(value.get("preset_id") or "").strip().lower() or None
            has_card_snapshot = bool(value.get("tool_name")) and action not in {
                action_ids.WF_SELECT_PRESET,
                action_ids.WF_RESET_TEAM,
                action_ids.WF_OPEN_CONFIG,
            }
            restored_requirement: str | None = None
            if cfg_sid and action != action_ids.WF_OPEN_CONFIG:
                restored_requirement = self._get_pending_workflow_requirement(
                    project,
                    chat_id,
                    cfg_sid,
                )
            state, _ = self._build_initial_config_state(
                chat_id=chat_id,
                project=project,
                preset_override=None,
                fallback_preset=raw_payload_preset,
                prefer_fallback_over_saved=has_card_snapshot,
                requirement=restored_requirement,
                config_session_id=(cfg_sid or None),
            )
            # Rehydrate editor fields from callback payload if session wasn't cached.
            # Rendered Card 2.0 callbacks omit None fields (`model_name`, `profile`,
            # `effort`) via `_clean_callback_payload()`, so when `tool_name` is present
            # we reset missing optional fields to `None` rather than leaking
            # `saved_cfg.members[0]`'s attributes across member/tool switches.
            payload_agent = str(value.get("agent_id") or "").strip()
            payload_orch = str(value.get("orchestrator_agent_id") or "").strip()
            valid_member_ids = {m.agent_id for m in state.members}
            restored_orch = (
                payload_orch
                if payload_orch in valid_member_ids
                else state.orchestrator_agent_id
            )
            if has_card_snapshot:
                state = replace(
                    state,
                    selected_agent_id=(payload_agent or state.selected_agent_id),
                    orchestrator_agent_id=restored_orch,
                    editor_tool=str(value["tool_name"]).strip().lower(),
                    editor_model=(
                        str(value["model_name"]).strip()
                        if value.get("model_name")
                        else None
                    ),
                    editor_profile=(
                        str(value["profile"]).strip().lower()
                        if value.get("profile")
                        else None
                    ),
                    editor_effort=(
                        str(value["effort"]).strip().lower()
                        if value.get("effort")
                        else None
                    ),
                    editor_role=(
                        _ui_role(str(value["role"]))
                        if value.get("role")
                        else state.editor_role
                    ),
                )
            else:
                if (
                    action
                    not in {
                        action_ids.WF_SELECT_PRESET,
                        action_ids.WF_RESET_TEAM,
                        action_ids.WF_OPEN_CONFIG,
                    }
                    and payload_orch in valid_member_ids
                ):
                    state = replace(state, orchestrator_agent_id=payload_orch)
                if payload_agent:
                    state = replace(state, selected_agent_id=payload_agent)
                    matched_member = next(
                        (m for m in state.members if m.agent_id == payload_agent),
                        None,
                    )
                    if matched_member is not None:
                        state = replace(
                            state,
                            editor_role=_ui_role(matched_member.role),
                            editor_tool=matched_member.tool_name,
                            editor_model=matched_member.model_name,
                            editor_profile=matched_member.profile,
                            editor_effort=matched_member.effort,
                        )
                if value.get("model_name"):
                    state = replace(state, editor_model=str(value["model_name"]).strip())
                if value.get("profile"):
                    state = replace(state, editor_profile=str(value["profile"]).strip().lower())
                if value.get("effort"):
                    state = replace(state, editor_effort=str(value["effort"]).strip().lower())
                if value.get("role"):
                    state = replace(state, editor_role=_ui_role(str(value["role"])))

        state = replace(state, status_notice=None)
        toast_type = "info"
        toast_content = "已更新 Workflow 配置"
        launch_spec: dict[str, Any] | None = None

        if action == action_ids.WF_OPEN_CONFIG:
            state, capabilities = self._build_initial_config_state(
                chat_id=chat_id,
                project=project,
                requirement=state.requirement,
                config_session_id=state.config_session_id,
            )
            toast_content = "已打开 Workflow 团队配置"

        elif action == action_ids.WF_SELECT_PRESET:
            chosen_preset = str(
                value.get("_option") or value.get("preset_id") or "collaborative"
            ).strip().lower()
            valid_presets = {p[0] for p in WORKFLOW_PRESET_CATALOG} | {
                "recommended",
                "deep_collab",
            }
            if chosen_preset not in valid_presets:
                toast_type = "error"
                toast_content = f"未知预设: {chosen_preset}"
                state = replace(state, status_notice=toast_content)
            else:
                if chosen_preset == "custom":
                    state = replace(
                        state,
                        preset_id="custom",
                        saved_as_default=False,
                        status_notice="已切换为自定义模式，可按成员定制 tool / model / effort / role",
                    )
                else:
                    new_members = self._default_members_for_preset(
                        capabilities,
                        preset_id=chosen_preset,
                    )
                    first = new_members[0] if new_members else None
                    state = replace(
                        state,
                        preset_id=chosen_preset,
                        members=new_members,
                        orchestrator_agent_id=(first.agent_id if first else ""),
                        selected_agent_id=(first.agent_id if first else "__new__"),
                        editor_role=_ui_role(first.role if first else "worker"),
                        editor_tool=(first.tool_name if first else state.editor_tool),
                        editor_model=(first.model_name if first else None),
                        editor_profile=(first.profile if first else None),
                        editor_effort=(first.effort if first else None),
                        saved_as_default=False,
                        status_notice=f"已切换预设：{chosen_preset}",
                    )
                toast_content = f"已选择预设：{chosen_preset}"

        elif action == action_ids.WF_SELECT_AGENT:
            target_id = str(value.get("_option") or value.get("agent_id") or "").strip()
            if target_id == "__new__":
                state = replace(
                    state,
                    selected_agent_id="__new__",
                    status_notice="新增成员模式：选择 Role / Tool / Model / Effort 后点击「添加/更新成员」",
                )
                toast_content = "已切换为新增成员"
            else:
                member = next((m for m in state.members if m.agent_id == target_id), None)
                if member is None:
                    toast_type = "warning"
                    toast_content = f"未找到成员 {target_id}"
                    state = replace(state, status_notice=toast_content)
                else:
                    state = replace(
                        state,
                        selected_agent_id=member.agent_id,
                        editor_role=_ui_role(member.role),
                        editor_tool=member.tool_name,
                        editor_model=member.model_name,
                        editor_profile=member.profile,
                        editor_effort=member.effort,
                        status_notice=f"正在编辑成员 {member.agent_id}",
                    )
                    toast_content = f"已选中成员 {member.agent_id}"

        elif action == action_ids.WF_SELECT_ROLE:
            chosen_role = _ui_role(str(value.get("_option") or value.get("role") or "worker"))
            valid_roles = {r[0] for r in WORKFLOW_ROLE_CATALOG}
            if chosen_role not in valid_roles:
                chosen_role = "worker"
            state = replace(
                state,
                editor_role=chosen_role,
                saved_as_default=False,
            )
            toast_content = f"已选择角色：{chosen_role}"

        elif action == action_ids.WF_SELECT_TOOL:
            chosen_tool = str(value.get("_option") or value.get("tool_name") or "").strip().lower()
            cap = capabilities.get(chosen_tool)
            if cap is None:
                toast_type = "error"
                toast_content = f"未知工具: {chosen_tool}"
                state = replace(state, status_notice=toast_content)
            else:
                state = replace(
                    state,
                    editor_tool=chosen_tool,
                    editor_model=None,
                    editor_profile=None,
                    editor_effort=None,
                    saved_as_default=False,
                )
                toast_content = f"已切换工具：{chosen_tool}"

        elif action == action_ids.WF_SELECT_MODEL:
            chosen_model = str(value.get("_option") or value.get("model_name") or "").strip() or None
            state = replace(
                state,
                editor_model=chosen_model,
                editor_profile=None,
                editor_effort=None,
                saved_as_default=False,
            )
            toast_content = f"已选择模型：{chosen_model or 'default'}"

        elif action == action_ids.WF_SELECT_PROFILE:
            chosen_profile = str(value.get("_option") or value.get("profile") or "").strip().lower() or None
            state = replace(
                state,
                editor_profile=chosen_profile,
                editor_effort=None,
                saved_as_default=False,
            )
            toast_content = f"已选择 Profile：{chosen_profile or 'default'}"

        elif action == action_ids.WF_SELECT_EFFORT:
            chosen_effort = str(value.get("_option") or value.get("effort") or "").strip().lower() or None
            state = replace(
                state,
                editor_effort=chosen_effort,
                saved_as_default=False,
            )
            toast_content = f"已选择 Effort：{chosen_effort or 'default'}"

        elif action == action_ids.WF_ADD_OR_UPDATE_AGENT:
            existing_ids = {m.agent_id for m in state.members}
            is_updating = state.selected_agent_id in existing_ids
            target_id = (
                state.selected_agent_id
                if is_updating
                else _next_agent_id(state.members)
            )
            new_binding, err = self._build_validated_editor_binding(
                state,
                capabilities,
                agent_id=target_id,
            )
            if new_binding is None:
                toast_type = "error"
                toast_content = err or "成员配置无效"
                state = replace(state, status_notice=toast_content)
            else:
                if not is_updating and len(state.members) >= MAX_WORKFLOW_AGENT_POOL_SIZE:
                    toast_type = "warning"
                    toast_content = f"最多支持 {MAX_WORKFLOW_AGENT_POOL_SIZE} 个 Agent 成员"
                    state = replace(state, status_notice=toast_content)
                else:
                    if is_updating:
                        candidate_members = [
                            new_binding if m.agent_id == target_id else m
                            for m in state.members
                        ]
                    else:
                        candidate_members = [*state.members, new_binding]
                    try:
                        frozen = freeze_agent_pool(candidate_members, require_nonempty=True)
                    except ValueError as exc:
                        toast_type = "error"
                        toast_content = str(exc)
                        state = replace(state, status_notice=toast_content)
                    else:
                        orch_id = state.orchestrator_agent_id or frozen[0].agent_id
                        if orch_id not in {m.agent_id for m in frozen}:
                            orch_id = frozen[0].agent_id
                        next_selected = (
                            "__new__"
                            if len(frozen) < MAX_WORKFLOW_AGENT_POOL_SIZE
                            else target_id
                        )
                        toast_type = "success"
                        toast_content = (
                            f"已更新成员 {target_id}"
                            if is_updating
                            else f"已添加成员 {target_id}"
                        )
                        state = replace(
                            state,
                            members=frozen,
                            orchestrator_agent_id=orch_id,
                            selected_agent_id=next_selected,
                            saved_as_default=False,
                            status_notice=f"{toast_content}（当前 {len(frozen)}/{MAX_WORKFLOW_AGENT_POOL_SIZE}）",
                        )

        elif action == action_ids.WF_REMOVE_AGENT:
            if len(state.members) <= 1:
                toast_type = "warning"
                toast_content = "至少需要保留 1 个成员"
                state = replace(state, status_notice=toast_content)
            else:
                existing_ids = {m.agent_id for m in state.members}
                target_id = (
                    state.selected_agent_id
                    if state.selected_agent_id in existing_ids
                    else state.members[-1].agent_id
                )
                remaining = tuple(m for m in state.members if m.agent_id != target_id)
                orch_id = state.orchestrator_agent_id
                if orch_id not in {m.agent_id for m in remaining}:
                    orch_id = remaining[0].agent_id
                first = remaining[0]
                toast_type = "info"
                toast_content = f"已移除成员 {target_id}"
                state = replace(
                    state,
                    members=remaining,
                    orchestrator_agent_id=orch_id,
                    selected_agent_id=first.agent_id,
                    editor_role=_ui_role(first.role),
                    editor_tool=first.tool_name,
                    editor_model=first.model_name,
                    editor_profile=first.profile,
                    editor_effort=first.effort,
                    saved_as_default=False,
                    status_notice=toast_content,
                )

        elif action == action_ids.WF_SET_ORCHESTRATOR:
            existing_ids = {m.agent_id for m in state.members}
            target_id = (
                state.selected_agent_id
                if state.selected_agent_id in existing_ids
                else (state.members[0].agent_id if state.members else "")
            )
            if not target_id:
                toast_type = "warning"
                toast_content = "暂无可设为主编排器的成员"
                state = replace(state, status_notice=toast_content)
            else:
                toast_type = "success"
                toast_content = f"已设置 {target_id} 为主编排器"
                state = replace(
                    state,
                    orchestrator_agent_id=target_id,
                    saved_as_default=False,
                    status_notice=toast_content,
                )

        elif action == action_ids.WF_SAVE_TEAM:
            members_to_save = list(state.members)
            validation_err: str | None = None
            current_member = next(
                (m for m in members_to_save if m.agent_id == state.selected_agent_id),
                None,
            )
            if current_member is not None:
                has_pending_edit = (
                    str(state.editor_tool or "").strip().lower() != current_member.tool_name
                    or (str(state.editor_model or "").strip() or None) != (current_member.model_name or None)
                    or (str(state.editor_profile or "").strip().lower() or None) != (current_member.profile or None)
                    or (str(state.editor_effort or "").strip().lower() or None) != (current_member.effort or None)
                    or _ui_role(state.editor_role) != _ui_role(current_member.role)
                )
                if has_pending_edit:
                    updated_binding, err_edit = self._build_validated_editor_binding(
                        state,
                        capabilities,
                        agent_id=current_member.agent_id,
                    )
                    if updated_binding is None:
                        validation_err = f"{current_member.agent_id}: {err_edit}"
                    else:
                        candidate = [
                            updated_binding if m.agent_id == current_member.agent_id else m
                            for m in members_to_save
                        ]
                        try:
                            freeze_agent_pool(candidate, require_nonempty=True)
                        except ValueError as exc:
                            validation_err = str(exc)
                        else:
                            members_to_save = candidate

            if validation_err is None:
                for member in members_to_save:
                    ok_m, err_m = validate_agent_binding_spec(
                        member.tool_name,
                        model_name=member.model_name,
                        profile=member.profile,
                        effort=member.effort,
                        capabilities=capabilities,
                    )
                    if not ok_m:
                        validation_err = f"{member.agent_id}: {err_m}"
                        break

            if validation_err:
                toast_type = "error"
                toast_content = validation_err
                state = replace(state, status_notice=toast_content)
            elif not members_to_save:
                toast_type = "error"
                toast_content = "至少需要 1 个有效成员才能保存"
                state = replace(state, status_notice=toast_content)
            else:
                try:
                    frozen = freeze_agent_pool(members_to_save, require_nonempty=True)
                except ValueError as exc:
                    toast_type = "error"
                    toast_content = str(exc)
                    state = replace(state, status_notice=toast_content)
                else:
                    orch_id = state.orchestrator_agent_id or frozen[0].agent_id
                    if orch_id not in {m.agent_id for m in frozen}:
                        orch_id = frozen[0].agent_id
                    if project is None:
                        toast_type = "error"
                        toast_content = "未找到当前项目，无法保存配置"
                        state = replace(state, status_notice=toast_content)
                    else:
                        cfg_payload = {
                            "version": 1,
                            "preset_id": state.preset_id,
                            "orchestrator_agent_id": orch_id,
                            "bindings": [m.to_dict() for m in frozen],
                            "updated_at": time.time(),
                        }
                        saved_ok = self.ctx.project_manager.commit_workflow_team_configuration(
                            project,
                            cfg_payload,
                            chat_id=chat_id,
                        )
                        if saved_ok:
                            toast_type = "success"
                            toast_content = "✅ 已保存为当前项目/会话默认 Workflow 团队"
                            state = replace(
                                state,
                                members=frozen,
                                orchestrator_agent_id=orch_id,
                                saved_as_default=True,
                                status_notice=toast_content,
                            )
                        else:
                            toast_type = "error"
                            toast_content = "保存配置失败，请稍后重试"
                            state = replace(state, status_notice=toast_content)

        elif action == action_ids.WF_RESET_TEAM:
            if project is not None:
                self.ctx.project_manager.commit_workflow_team_configuration(
                    project,
                    None,
                    chat_id=chat_id,
                )
            default_members = self._default_members_for_preset(
                capabilities,
                preset_id="collaborative",
            )
            first = default_members[0] if default_members else None
            toast_type = "info"
            toast_content = "已重置为推荐默认团队"
            state = WorkflowConfigEditorState(
                config_session_id=state.config_session_id,
                chat_id=chat_id,
                project_id=(project.project_id if project else state.project_id),
                thread_root_id=state.thread_root_id,
                preset_id="collaborative",
                members=default_members,
                orchestrator_agent_id=(first.agent_id if first else ""),
                selected_agent_id=(first.agent_id if first else "__new__"),
                editor_role=_ui_role(first.role if first else "worker"),
                editor_tool=(first.tool_name if first else state.editor_tool),
                editor_model=(first.model_name if first else None),
                editor_profile=(first.profile if first else None),
                editor_effort=(first.effort if first else None),
                requirement=state.requirement,
                saved_as_default=False,
                status_notice=toast_content,
            )

        elif action == action_ids.WF_START_WORKFLOW:
            if not state.requirement:
                toast_type = "warning"
                toast_content = "未指定任务需求，请发送 `/wf <需求>` 启动"
                state = replace(state, status_notice=toast_content)
            else:
                validation_err: str | None = None
                if not state.members:
                    validation_err = "至少需要 1 个有效成员才能启动"
                else:
                    for member in state.members:
                        ok_m, err_m = validate_agent_binding_spec(
                            member.tool_name,
                            model_name=member.model_name,
                            profile=member.profile,
                            effort=member.effort,
                            capabilities=capabilities,
                        )
                        if not ok_m:
                            validation_err = f"{member.agent_id}: {err_m}"
                            break
                if validation_err:
                    toast_type = "error"
                    toast_content = validation_err
                    state = replace(state, status_notice=toast_content)
                else:
                    self._persist_pending_workflow_requirement(
                        project,
                        chat_id,
                        state.config_session_id,
                        None,
                    )
                    launch_spec = {
                        "requirement": state.requirement,
                        "preset_id": state.preset_id,
                        "members": state.members,
                        "orchestrator_agent_id": state.orchestrator_agent_id,
                        "project": project,
                    }
                    toast_type = "info"
                    toast_content = "正在启动 Workflow…"

        cached_state = (
            replace(state, requirement=None)
            if launch_spec is not None
            else state
        )
        self._put_wf_config_state(cached_state, message_id=message_id)
        card = build_workflow_config_card(state, capabilities)
        return card, toast_type, toast_content, launch_spec

    def handle_workflow_config_callback_sync(
        self,
        message_id: str,
        chat_id: str,
        project_id: str | None,
        value: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str, str]:
        """Synchronous Card 2.0 callback handler returning ``(card, toast_type, toast_content)``."""
        card, toast_type, toast_content, _launch = self._apply_config_transition(
            message_id,
            chat_id,
            project_id,
            value,
        )
        return card, toast_type, toast_content

    def handle_workflow_config_action(
        self,
        message_id: str,
        chat_id: str,
        project_id: str | None,
        value: dict[str, Any],
    ) -> None:
        """Async/direct action_registry handler for ``WF_CONFIG_ACTIONS``."""
        card, _toast_type, _toast_content, launch_spec = self._apply_config_transition(
            message_id,
            chat_id,
            project_id,
            value,
        )
        if launch_spec is not None:
            self.start_workflow(
                message_id,
                chat_id,
                launch_spec["requirement"],
                launch_spec["project"],
                preset_id=launch_spec["preset_id"],
                override_pool=launch_spec["members"],
                override_orchestrator_id=launch_spec["orchestrator_agent_id"],
            )
            return
        if card is not None:
            if not (message_id and self.update_card(message_id, card)):
                self.reply_card(message_id, card)
