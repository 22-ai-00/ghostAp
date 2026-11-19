"""ACP session activation fence and model-selection helpers for SystemHandler."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, Optional

from ...mode.backends import BACKEND_META_BY_KEY
from ...tasking import TaskStatus
from ...utils.errors import safe_error_message

if TYPE_CHECKING:
    from ...project import ProjectContext
    from .system import _ACPActivationEffect, _ACPActivationOwner

logger = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    if name == "fetch_acp_models":
        from ...acp.helper import fetch_acp_models

        globals()["fetch_acp_models"] = fetch_acp_models
        return fetch_acp_models
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _system_module():
    """Return the system handler module for runtime-patchable symbols."""
    from . import system as _sys_mod

    return _sys_mod


class ACPActivationMixin:
    """Mixin providing ACP activation fencing and model-cascade selection."""

    @staticmethod
    def _acp_activation_key(
        chat_id: str,
        project_id: str | None,
        thread_id: str | None,
    ) -> tuple[str, str, str]:
        """Build the deduplication key for a chat/project/thread ACP activation."""
        return (chat_id, project_id or "", thread_id or "")

    @staticmethod
    def _canonical_acp_thread_id(thread_id: str | None) -> str | None:
        """Normalize the effective ACP thread identifier for the message context."""
        normalized = str(thread_id or "").strip()
        if not normalized:
            return None
        try:
            from ...thread import get_thread_manager

            context = get_thread_manager().get(normalized)
            canonical = str(
                getattr(context, "thread_root_id", "") or ""
            ).strip()
            return canonical or normalized
        except Exception:
            logger.debug(
                "failed to canonicalize ACP thread coordinate",
                exc_info=True,
            )
            return normalized

    def _claim_acp_activation(
        self,
        chat_id: str,
        project_id: str | None,
        thread_id: str | None,
        *,
        message_id: str,
        explicit_card: bool,
        tool_name: str,
        model_name: str | None,
    ) -> tuple[_ACPActivationOwner, _ACPActivationOwner | None]:
        """Register a new activation owner and supersede any prior active owner."""
        key = self._acp_activation_key(chat_id, project_id, thread_id)
        with self._acp_activation_lock:
            fence_epoch = self._acp_activation_fence_epochs.get(key, 0)
            if explicit_card:
                generation = self._acp_activation_generations.get(key, 0) + 1
                self._acp_activation_generations[key] = generation
                previous = self._acp_activation_owners.get(key)
            else:
                # User tasks are FIFO work, not configuration writes.  They
                # must never supersede one another merely because both need to
                # recover the same saved ACP session.
                generation = 0
                previous = None
            owner = _system_module()._ACPActivationOwner(
                generation=generation,
                fence_epoch=fence_epoch,
                origin_message_id=message_id,
                status_message_id=message_id,
                explicit_card=explicit_card,
                tool_name=tool_name,
                model_name=model_name,
            )
            if explicit_card:
                self._acp_activation_owners[key] = owner
            else:
                self._acp_activation_task_owners.setdefault(key, {})[
                    id(owner)
                ] = owner
            return owner, previous

    def _owns_acp_activation(
        self,
        chat_id: str,
        project_id: str | None,
        thread_id: str | None,
        owner: _ACPActivationOwner,
    ) -> bool:
        """Return True if owner still holds the current active slot and fence epoch."""
        key = self._acp_activation_key(chat_id, project_id, thread_id)
        with self._acp_activation_lock:
            if (
                owner.terminal
                or self._acp_activation_fence_epochs.get(key, 0)
                != owner.fence_epoch
            ):
                return False
            if owner.explicit_card:
                return (
                    self._acp_activation_generations.get(key)
                    == owner.generation
                    and self._acp_activation_owners.get(key) is owner
                )
            return (
                self._acp_activation_task_owners.get(key, {}).get(id(owner))
                is owner
            )

    def _invalidate_acp_activation(
        self,
        chat_id: str,
        project_id: str | None,
        thread_id: str | None,
    ) -> list[_ACPActivationOwner]:
        """Invalidate active/queued ACP owners for the coordinate and bump fence epoch."""
        key = self._acp_activation_key(chat_id, project_id, thread_id)
        with self._acp_activation_lock:
            self._acp_activation_fence_epochs[key] = (
                self._acp_activation_fence_epochs.get(key, 0) + 1
            )
            invalidated = []
            visible_owner = self._acp_activation_owners.pop(key, None)
            if visible_owner is not None:
                invalidated.append(visible_owner)
            invalidated.extend(
                self._acp_activation_task_owners.pop(key, {}).values()
            )
            self._acp_activation_generations.pop(key, None)
            self._acp_activation_fence_epochs.pop(key, None)
            return invalidated

    def _release_acp_activation(
        self,
        chat_id: str,
        project_id: str | None,
        thread_id: str | None,
        owner: _ACPActivationOwner,
        *,
        committed: bool = False,
    ) -> bool:
        """Mark exactly one still-current activation terminal."""

        key = self._acp_activation_key(chat_id, project_id, thread_id)
        with self._acp_activation_lock:
            if not self._owns_acp_activation(
                chat_id,
                project_id,
                thread_id,
                owner,
            ):
                return False
            owner.committed = committed
            owner.terminal = True
            if owner.explicit_card:
                if self._acp_activation_owners.get(key) is owner:
                    self._acp_activation_owners.pop(key, None)
            else:
                task_owners = self._acp_activation_task_owners.get(key)
                if task_owners is not None:
                    task_owners.pop(id(owner), None)
                    if not task_owners:
                        self._acp_activation_task_owners.pop(key, None)
            if (
                key not in self._acp_activation_owners
                and key not in self._acp_activation_task_owners
            ):
                self._acp_activation_generations.pop(key, None)
                self._acp_activation_fence_epochs.pop(key, None)
            return True

    def _invalidate_acp_project_activations(
        self,
        chat_id: str,
        project_id: str | None,
    ) -> list[_ACPActivationOwner]:
        """Fence every direct/topic activation for one project coordinate."""

        normalized_project_id = project_id or ""
        with self._acp_activation_lock:
            keys = {
                key
                for key in (
                    set(self._acp_activation_owners)
                    | set(self._acp_activation_task_owners)
                )
                if key[0] == chat_id and key[1] == normalized_project_id
            }
            invalidated: list[_ACPActivationOwner] = []
            for key in keys:
                invalidated.extend(
                    self._invalidate_acp_activation(
                        key[0],
                        key[1] or None,
                        key[2] or None,
                    )
                )
            return invalidated

    def _finish_invalidated_acp_cards(
        self,
        owners: list[_ACPActivationOwner],
    ) -> None:
        """Close visible selector cards invalidated by an exit or project switch."""

        for owner in owners:
            if not owner.explicit_card or owner.terminal:
                continue
            _msg_type, cancelled_content = (
                _system_module().SystemBuilder.build_acp_programming_superseded_card(
                    owner.tool_name,
                    owner.model_name,
                    _system_module().UI_TEXT["system_acp_activation_cancelled"],
                )
            )
            if not self.update_card(owner.status_message_id, cancelled_content):
                self.reply_card(owner.status_message_id, cancelled_content)
            owner.terminal = True

    def switch_active_project_with_acp_fence(
        self,
        message_id: str,
        chat_id: str,
        old_project: "ProjectContext",
        new_project_id: str,
    ) -> tuple[bool, str]:
        """Atomically fence old-project activation and move the active binding."""

        old_project_id = old_project.project_id
        with self._acp_activation_lock:
            invalidated = self._invalidate_acp_project_activations(
                chat_id,
                old_project_id,
            )
            self._finish_invalidated_acp_cards(invalidated)
            current_mode = self.mode_manager.get_mode(
                chat_id,
                project_id=old_project_id,
            )
            if current_mode in _system_module().PROGRAMMING_MODES:
                handler = self.get_handler(current_mode.value)
                if handler is not None:
                    handler.exit_mode(
                        message_id,
                        chat_id,
                        project=old_project,
                    )
            return self.project_manager.set_active_project(
                chat_id,
                new_project_id,
            )

    def _enter_mode_with_acp_model(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        model_name: Optional[str],
        project: Optional["ProjectContext"] = None,
        thread_id: Optional[str] = None,
    ) -> _ACPActivationEffect | bool | None:
        """Activate ACP mode synchronously or enqueue background activation task."""
        target_project = project or self.project_manager.get_active_project(chat_id)

        handler = (
            self.get_handler(tool_name)
            if tool_name in BACKEND_META_BY_KEY
            else None
        )
        if handler:
            # Persistent programming mode intentionally outlives an ACP transport.
            # Probe the exact direct/topic coordinate instead of inferring liveness
            # from that persistent flag.  A missing session takes the ordinary
            # startup path; an existing session is either hot-switched (only when
            # the exact direct backend supports it) or identity-guarded replaced.
            _project_id = target_project.project_id if target_project else None
            manager_getter = getattr(handler, "_get_session_manager", None)
            manager = manager_getter() if callable(manager_getter) else None
            live_session = (
                manager.get_session(
                    chat_id,
                    project_id=_project_id,
                    thread_id=thread_id,
                )
                if manager is not None
                else None
            )

            def _replace_live_session() -> _ACPActivationEffect | None:
                replace_session = getattr(manager, "replace_session", None)
                if not callable(replace_session) or live_session is None:
                    return None
                cwd = (
                    target_project.root_path
                    if target_project is not None
                    else self.get_working_dir(chat_id)
                )
                agent_override_getter = getattr(
                    handler,
                    "_get_agent_type_override",
                    None,
                )
                agent_type_override = (
                    agent_override_getter(target_project)
                    if callable(agent_override_getter)
                    else None
                )
                replacement = replace_session(
                    chat_id,
                    cwd=cwd,
                    expected_session=live_session,
                    startup_timeout=getattr(
                        self.settings,
                        "acp_startup_timeout",
                        20,
                    ),
                    project_id=_project_id,
                    agent_type_override=agent_type_override,
                    model_name=model_name,
                    thread_id=thread_id,
                )
                replacement_session = getattr(replacement, "session", None)
                if (
                    not getattr(replacement, "created", False)
                    or replacement_session is None
                ):
                    return None
                return _system_module()._ACPActivationEffect(
                    session=replacement_session,
                    changed=True,
                )

            if (
                live_session is not None
                and model_name is not None
            ):
                if tool_name == "traex":
                    backend_model_name = model_name
                else:
                    from ...acp.providers import normalize_acp_model_name

                    backend_model_name = normalize_acp_model_name(tool_name, model_name)
                active_model_name = str(
                    getattr(live_session, "_model_name", "") or ""
                ).strip()
                if (
                    backend_model_name
                    and active_model_name == backend_model_name
                ):
                    return _system_module()._ACPActivationEffect(
                        session=live_session,
                        changed=False,
                    )
            if (
                live_session is not None
                and model_name is not None
                and thread_id is None
                and callable(getattr(live_session, "set_model", None))
                and callable(getattr(handler, "switch_model", None))
            ):
                switched = handler.switch_model(
                    message_id,
                    chat_id,
                    model_name,
                    project=target_project,
                    expected_session=live_session,
                )
                return (
                    _system_module()._ACPActivationEffect(
                        session=live_session,
                        changed=True,
                    )
                    if switched
                    else None
                )
            if live_session is not None and model_name is not None:
                return _replace_live_session()
            if live_session is not None and model_name is None:
                active_model = getattr(live_session, "_model_name", None)
                if hasattr(live_session, "_model_name") and not str(
                    active_model or ""
                ).strip():
                    return _system_module()._ACPActivationEffect(
                        session=live_session,
                        changed=False,
                    )
                return _replace_live_session()
            # silent=True: the selection card owns the visible lifecycle.
            enter_kwargs = {"project": target_project, "silent": True}
            if target_project is not None:
                enter_kwargs.update(
                    model_override=model_name,
                    commit_project_state=False,
                    activate_mode=False,
                    exit_opposite_mode=False,
                    inherit_thread_context=False,
                )
            if thread_id is not None:
                enter_kwargs["thread_id"] = thread_id
            entered = bool(handler.enter_mode(message_id, chat_id, **enter_kwargs))
            if not entered:
                return None
            if target_project is None:
                return True
            if manager is None:
                return None
            session = manager.get_session(
                chat_id,
                project_id=_project_id,
                thread_id=thread_id,
            )
            if session is None:
                return None
            return _system_module()._ACPActivationEffect(session=session, changed=True)

        self.reply_error(message_id, _system_module().UI_TEXT["system_acp_unsupported_tool"].format(tool_name=tool_name))
        return None

    def _configuration_project(
        self,
        chat_id: str,
        project: Optional["ProjectContext"],
    ) -> Optional["ProjectContext"]:
        """Resolve target project or fall back to an active synthetic config project."""
        if project is not None:
            return project
        active = self.project_manager.get_active_project(chat_id)
        if active is not None:
            return active
        try:
            active, _created = self.project_manager.get_or_create_project_for_path(
                self.get_working_dir(chat_id), chat_id
            )
            return active
        except Exception:
            logger.exception("failed to resolve project for ACP configuration")
            return None

    @staticmethod
    def _model_names(models: list) -> list[str]:
        """Extract displayable model name identifiers from ACP model entries."""
        names: list[str] = []
        for item in models or []:
            variants = getattr(item, "selection_variants", ()) or ()
            for candidate in variants or (item,):
                name = str(
                    candidate.get("name", "")
                    if isinstance(candidate, dict)
                    else getattr(candidate, "name", candidate)
                ).strip()
                if name and name not in names:
                    names.append(name)
        return names

    def handle_acp_command(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Show or atomically update the saved project ACP tool."""
        project = self._configuration_project(chat_id, project)
        names = [
            str(getattr(item, "name", "") or "").strip()
            for item in _system_module().list_acp_tools()
        ]
        names = [name for name in names if name]
        current = str(getattr(project, "acp_tool_name", "") or "").strip() or "未设置"
        parts = (text or "").strip().split(maxsplit=1)
        requested = parts[1].strip().lower() if len(parts) > 1 else ""
        if not requested:
            self.reply_text(
                message_id,
                _system_module().UI_TEXT["system_acp_config_summary"].format(
                    current=current,
                    available=" · ".join(f"`{name}`" for name in names) or "无",
                ),
            )
            return
        if requested not in names:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_unsupported_tool"].format(tool_name=requested))
            return
        if project is None:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
            return
        previous_tool = str(getattr(project, "acp_tool_name", "") or "").strip()
        model_name = getattr(project, "acp_model_name", None) if previous_tool == requested else None
        active_key = self._resolve_active_programming_mode_key(chat_id, project)
        if not self.project_manager.commit_acp_configuration(
            project, tool_name=requested, model_name=model_name
        ):
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
            return
        sync_note = ""
        if active_key and active_key != requested:
            if self._sync_live_acp_tool_switch(chat_id, project, old_tool=active_key, new_tool=requested):
                sync_note = f"\n🔄 当前会话已从 `{active_key}` 切换到 `{requested}`。"
        self.reply_text(
            message_id,
            _system_module().UI_TEXT["system_acp_config_saved"].format(
                tool=requested, model=model_name or "default"
            )
            + sync_note,
        )

    def _sync_live_acp_tool_switch(
        self,
        chat_id: str,
        project: "ProjectContext",
        *,
        old_tool: str,
        new_tool: str,
    ) -> bool:
        """Retire old live session and align mode state when /acp switches active tool."""
        pid = project.project_id
        try:
            old_handler = self.get_handler(old_tool)
        except Exception:
            old_handler = None
        if old_handler is not None:
            mgr_getter = getattr(old_handler, "_get_session_manager", None)
            manager = mgr_getter() if callable(mgr_getter) else None
            if manager is not None and callable(getattr(manager, "get_session", None)):
                live_session = manager.get_session(chat_id, project_id=pid)
                if live_session is not None and isinstance(getattr(live_session, "session_id", None), str):
                    retire = getattr(manager, "retire_session", None) or getattr(manager, "end_session", None)
                    if callable(retire):
                        retire(chat_id, project_id=pid)
        try:
            new_mode = _system_module().InteractionMode(new_tool)
        except ValueError:
            return False
        project.set_programming_mode(new_tool, True)
        enter_fn = getattr(self.mode_manager, "enter_programming_mode", None)
        if callable(enter_fn):
            enter_fn(chat_id, new_mode, project_id=pid)
        return True

    def _sync_live_acp_model(
        self,
        chat_id: str,
        project: "ProjectContext",
        tool_name: str,
        model_name: Optional[str],
    ) -> str:
        """Update or retire the live session so /model takes effect immediately."""
        pid = project.project_id
        try:
            handler = self.get_handler(tool_name)
        except Exception:
            handler = None
        if handler is None:
            return ""
        if callable(getattr(handler, "_enter_mode_on_manager", None)) and hasattr(handler, "current_model"):
            handler.current_model = model_name
        mgr_getter = getattr(handler, "_get_session_manager", None)
        manager = mgr_getter() if callable(mgr_getter) else None
        if manager is None or not callable(getattr(manager, "get_session", None)):
            return ""
        live_session = manager.get_session(chat_id, project_id=pid)
        if live_session is None or not isinstance(getattr(live_session, "session_id", None), str):
            return ""
        if model_name is not None:
            set_model = getattr(live_session, "set_model", None)
            if callable(set_model):
                try:
                    if bool(set_model(model_name)):
                        return "\n🔄 已热更新当前活跃会话模型。"
                except Exception:
                    logger.warning("[ACP] live session set_model failed", exc_info=True)
        retire = getattr(manager, "retire_session", None) or getattr(manager, "end_session", None)
        if callable(retire):
            retire(chat_id, project_id=pid)
            return "\n🔄 已重置当前活跃会话，下条消息将使用新模型启动。"
        return ""

    def _fetch_acp_models(
        self,
        tool_name: str,
        *,
        cwd: Optional[str] = None,
        current_model: Optional[str] = None,
    ) -> list:
        """Fetch the compact model catalog used by the text command."""
        return getattr(sys.modules[__name__], "fetch_acp_models")(
            tool_name, cwd=cwd, current_model=current_model
        )

    @staticmethod
    def _selected_option(value: dict) -> Optional[str]:
        """Return the trusted select option injected by the Feishu callback."""
        option = value.get("_option")
        if isinstance(option, dict):
            option = option.get("value")
        selected = str(option or "").strip()
        return selected or None

    def _action_project(
        self,
        chat_id: str,
        project_id: Optional[str],
    ) -> Optional["ProjectContext"]:
        """Resolve a project scoped to chat_id from an explicit project_id string."""
        if project_id:
            return self.project_manager.get_project_for_chat(project_id, chat_id)
        return self.project_manager.get_active_project(chat_id)

    def _available_acp_tool_names(self) -> set[str]:
        """Return sorted normalized names of currently installed ACP tools."""
        return {
            name
            for raw_name in _system_module().get_providers()
            if (name := str(raw_name or "").strip().lower())
        }

    def show_explicit_acp_model_selection(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        project: Optional["ProjectContext"] = None,
        *,
        origin_message_id: Optional[str] = None,
        pending_group: Optional[str] = None,
        pending_profile: Optional[str] = None,
        pending_effort: Optional[str] = None,
        show_loading: bool = True,
    ) -> None:
        """Show the explicit model configuration surface for one backend.

        Task-bearing routes deliberately do not call this method. They keep using
        the saved project selection (or the backend default) and continue
        automatically.
        """
        tool = str(tool_name or "").strip().lower()
        if not tool or tool not in self._available_acp_tool_names():
            self.reply_error(
                message_id,
                _system_module().UI_TEXT["system_acp_unsupported_tool"].format(tool_name=tool),
            )
            return

        target_project = self._configuration_project(chat_id, project)
        if target_project is None:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
            return
        current_model = (
            getattr(target_project, "acp_model_name", None)
            if getattr(target_project, "acp_tool_name", None) == tool
            else None
        )
        card_message_id = origin_message_id
        if show_loading:
            _msg_type, loading_content = _system_module().SystemBuilder.build_acp_model_loading_card(
                tool,
                project_id=target_project.project_id,
            )
            if card_message_id:
                if not self.update_card(card_message_id, loading_content):
                    replacement_id = self.reply_card(message_id, loading_content)
                    card_message_id = (
                        replacement_id
                        if isinstance(replacement_id, str) and replacement_id
                        else None
                    )
            else:
                replacement_id = self.reply_card(message_id, loading_content)
                card_message_id = (
                    replacement_id
                    if isinstance(replacement_id, str) and replacement_id
                    else None
                )
        try:
            models = self._fetch_acp_models(
                tool,
                cwd=target_project.root_path,
                current_model=current_model,
            )
        except Exception:
            logger.exception(
                "[ACP] explicit model discovery failed tool=%s project=%s",
                tool,
                target_project.project_id,
            )
            _msg_type, error_content = _system_module().SystemBuilder.build_acp_model_error_card(
                tool,
                project_id=target_project.project_id,
            )
            if card_message_id and self.update_card(card_message_id, error_content):
                return
            self.reply_card(message_id, error_content)
            return
        _msg_type, card_content = _system_module().SystemBuilder.build_acp_model_cascade_card(
            models,
            tool,
            project_id=target_project.project_id,
            current_model=current_model,
            pending_group=pending_group,
            pending_profile=pending_profile,
            pending_effort=pending_effort,
        )
        if card_message_id and self.update_card(card_message_id, card_content):
            return
        self.reply_card(message_id, card_content)

    def handle_acp_model_cascade_select(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str],
        value: dict,
    ) -> None:
        """Redraw a selector after one server-recognized cascade dimension."""
        project = self._action_project(chat_id, project_id)
        if project is None:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
            return
        action = str(value.get("action") or "").strip()
        selected = self._selected_option(value)
        if selected is None:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_select_model_prompt"])
            return

        from ...card.actions import dispatch as action_ids

        group = str(value.get("model_group") or "").strip() or None
        profile = str(value.get("model_profile") or "").strip() or None
        effort = str(value.get("model_effort") or "").strip() or None
        if action == action_ids.SELECT_ACP_MODEL_GROUP:
            group, profile, effort = selected, None, None
        elif action == action_ids.SELECT_ACP_MODEL_PROFILE:
            profile, effort = selected, None
        elif action == action_ids.SELECT_ACP_MODEL_EFFORT:
            effort = selected
        else:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_select_model_prompt"])
            return

        self.show_explicit_acp_model_selection(
            message_id,
            chat_id,
            str(value.get("tool_name") or ""),
            project,
            origin_message_id=message_id,
            pending_group=group,
            pending_profile=profile,
            pending_effort=effort,
            show_loading=False,
        )

    def handle_open_acp_model_picker(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str],
        value: dict,
    ) -> None:
        """Open the model picker immediately using cached models when available."""
        project = self._action_project(chat_id, project_id)
        if project is None:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
            return
        tool = str(value.get("tool_name") or "").strip().lower()
        self.show_explicit_acp_model_selection(
            message_id,
            chat_id,
            tool,
            project,
            origin_message_id=message_id,
            pending_group=str(value.get("model_group") or "").strip() or None,
            pending_profile=str(value.get("model_profile") or "").strip() or None,
            pending_effort=str(value.get("model_effort") or "").strip() or None,
            show_loading=False,
        )

    def handle_refresh_acp_models(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str],
        value: dict,
    ) -> None:
        """Invalidate the backend catalog before redrawing the explicit picker."""
        project = self._action_project(chat_id, project_id)
        if project is None:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
            return
        tool = str(value.get("tool_name") or "").strip().lower()
        _system_module().invalidate_acp_model_cache(tool, project.root_path)
        self.show_explicit_acp_model_selection(
            message_id,
            chat_id,
            tool,
            project,
            origin_message_id=message_id,
            pending_group=str(value.get("model_group") or "").strip() or None,
            pending_profile=str(value.get("model_profile") or "").strip() or None,
            pending_effort=str(value.get("model_effort") or "").strip() or None,
        )

    def handle_select_acp_model(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str],
        value: dict,
    ) -> None:
        """Validate a final callback against a freshly probed capability matrix."""
        project = self._action_project(chat_id, project_id)
        if project is None:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
            return
        tool = str(value.get("tool_name") or "").strip().lower()
        if not tool or tool not in self._available_acp_tool_names():
            self.reply_error(
                message_id,
                _system_module().UI_TEXT["system_acp_unsupported_tool"].format(tool_name=tool),
            )
            return

        _system_module().invalidate_acp_model_cache(tool, project.root_path)
        models = self._fetch_acp_models(
            tool,
            cwd=project.root_path,
            current_model=(
                getattr(project, "acp_model_name", None)
                if getattr(project, "acp_tool_name", None) == tool
                else None
            ),
        )
        if value.get("use_default_model") is True:
            model_name = None
            model_group = None
            model_profile = None
            model_effort = None
        else:
            from ...card.render.model_cascade import compose_model_selection, parse_model_selection

            group = str(value.get("model_group") or "").strip()
            model_name = compose_model_selection(
                models,
                model=group,
                profile=str(value.get("model_profile") or "").strip() or None,
                effort=str(value.get("model_effort") or "").strip() or None,
            )
            if model_name is None:
                self.reply_error(
                    message_id,
                    _system_module().UI_TEXT["system_acp_unknown_model"].format(model=group),
                )
                return
            canonical = parse_model_selection(models, model_name)
            if canonical is None:
                self.reply_error(
                    message_id,
                    _system_module().UI_TEXT["system_acp_unknown_model"].format(model=group),
                )
                return
            model_group = canonical.model
            model_profile = canonical.profile
            model_effort = canonical.effort

        self._activate_acp_selection(
            message_id,
            chat_id,
            tool,
            model_name,
            project,
            explicit_card=True,
            model_group=model_group,
            model_profile=model_profile,
            model_effort=model_effort,
        )

    def handle_enter_acp_saved_selection(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        project: Optional["ProjectContext"],
        *,
        pending_prompt: Optional[str] = None,
    ) -> None:
        """Enter an ACP tool using the project's stored tool/model selection.

        Project chats use this path for free-form messages after a tool has
        already been chosen once, so normal follow-up work does not show the
        model selection card again.
        """
        tool = (tool_name or "").strip().lower()
        if not tool:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_select_tool_prompt"])
            return

        stored_model = None
        if project and getattr(project, "acp_tool_name", "") == tool:
            stored_model = getattr(project, "acp_model_name", None)
        model_name = str(stored_model).strip() if stored_model else None
        self._activate_acp_selection(
            message_id,
            chat_id,
            tool,
            model_name,
            project,
            pending_prompt=pending_prompt,
            explicit_card=not bool(pending_prompt),
        )

    def _activate_acp_selection(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        model_name: Optional[str],
        project: Optional["ProjectContext"] = None,
        *,
        pending_prompt: Optional[str] = None,
        explicit_card: bool = False,
        model_group: Optional[str] = None,
        model_profile: Optional[str] = None,
        model_effort: Optional[str] = None,
    ) -> None:
        """Validate, fence, connect/replace session, and commit ACP mode selection."""
        tool = (tool_name or "").strip().lower()
        use_default_model = model_name is None
        model = None if use_default_model else (model_name or "").strip()
        if not tool or (not use_default_model and not model):
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_select_model_prompt"])
            return
        _tool_spec = BACKEND_META_BY_KEY.get(tool)
        if _tool_spec is not None and _tool_spec.uses_claude_cli and model:
            from ...acp.claude_capabilities import (
                is_1m_variant,
                model_supports_1m,
                strip_1m_suffix,
            )
            from ...acp.claude_selection import split_claude_model_selection

            model_base, _effort = split_claude_model_selection(model)
            if (
                model_base
                and is_1m_variant(model_base)
                and not model_supports_1m(strip_1m_suffix(model_base))
            ):
                self.reply_error(
                    message_id,
                    _system_module().UI_TEXT["system_acp_unsupported_1m_model"].format(
                        model=strip_1m_suffix(model_base),
                    ),
                )
                return

        target_project = project or self.project_manager.get_active_project(chat_id)
        handler = self.get_handler(tool)

        if explicit_card:
            # Explicit direct-programming configuration is project-persistent.
            # Never inherit a topic coordinate from the card callback worker.
            thread_root_id = None
        else:
            from ...thread import get_current_thread_id

            raw_thread_id = get_current_thread_id()
            thread_root_id = self._canonical_acp_thread_id(
                raw_thread_id.strip()
                if isinstance(raw_thread_id, str) and raw_thread_id.strip()
                else None
            )
        project_id = self._project_id(target_project)
        status_message_id = message_id
        activation_owner, previous_owner = self._claim_acp_activation(
            chat_id,
            project_id,
            thread_root_id,
            message_id=message_id,
            explicit_card=explicit_card,
            tool_name=tool,
            model_name=model,
        )

        if (
            previous_owner is not None
            and previous_owner.explicit_card
            and not previous_owner.terminal
            and not previous_owner.committed
            and previous_owner.origin_message_id != message_id
        ):
            _msg_type, superseded_content = (
                _system_module().SystemBuilder.build_acp_programming_superseded_card(
                    previous_owner.tool_name,
                    previous_owner.model_name,
                    _system_module().UI_TEXT["system_acp_activation_superseded"],
                )
            )
            if not self.update_card(
                previous_owner.status_message_id,
                superseded_content,
            ):
                self.reply_card(
                    previous_owner.status_message_id,
                    superseded_content,
                )
            previous_owner.terminal = True

        def _owns_activation() -> bool:
            return self._owns_acp_activation(
                chat_id,
                project_id,
                thread_root_id,
                activation_owner,
            )

        def _replace_status_card(card_content: str) -> None:
            nonlocal status_message_id
            with self._acp_activation_lock:
                if not _owns_activation():
                    return
                if self.update_card(status_message_id, card_content):
                    return
                replacement_id = self.reply_card(status_message_id, card_content)
                if isinstance(replacement_id, str) and replacement_id:
                    status_message_id = replacement_id
                    activation_owner.status_message_id = replacement_id

        def _mark_terminal(*, committed: bool = False) -> None:
            self._release_acp_activation(
                chat_id,
                project_id,
                thread_root_id,
                activation_owner,
                committed=committed,
            )

        def _publish_failure(reason: str) -> None:
            with self._acp_activation_lock:
                if not _owns_activation():
                    return
                failure_reason = (
                    reason or _system_module().UI_TEXT["system_acp_activation_failed_safe"]
                )
                if not explicit_card:
                    self.reply_error(
                        message_id,
                        failure_reason,
                        chat_id=chat_id,
                        severity="recoverable",
                    )
                    _mark_terminal()
                    return
                _msg_type, failed_content = (
                    _system_module().SystemBuilder.build_acp_programming_failed_card(
                        tool,
                        model,
                        failure_reason,
                        project_id,
                        thread_root_id,
                        model_group=model_group,
                        model_profile=model_profile,
                        model_effort=model_effort,
                    )
                )
                _replace_status_card(failed_content)
                _mark_terminal()

        def _publish_cancelled() -> None:
            with self._acp_activation_lock:
                if not _owns_activation():
                    return
                if not explicit_card:
                    self.reply_error(
                        message_id,
                        _system_module().UI_TEXT["system_acp_activation_cancelled"],
                        chat_id=chat_id,
                        severity="recoverable",
                    )
                    _mark_terminal()
                    return
                _msg_type, cancelled_content = (
                    _system_module().SystemBuilder.build_acp_programming_superseded_card(
                        tool,
                        model,
                        _system_module().UI_TEXT["system_acp_activation_cancelled"],
                    )
                )
                _replace_status_card(cancelled_content)
                _mark_terminal()

        if explicit_card:
            _msg_type, initializing_content = (
                _system_module().SystemBuilder.build_acp_programming_initializing_card(
                    tool,
                    model,
                    project_id,
                    None,
                )
            )
            _replace_status_card(initializing_content)

        spec = _system_module().TaskSpec(
            chat_id=chat_id,
            name="activate_acp_model",
            task_type="acp_model_activation",
            project_id=project_id,
            message_id=message_id,
            origin_message_id=message_id,
            # Configuration clicks and task-bearing recovery share the project
            # queue in arrival order.  HIGH requeues with appendleft and can
            # otherwise let a later model click overtake an earlier user task.
            priority=_system_module().TaskPriority.NORMAL,
        )

        def _run_activation(_ctx) -> bool:
            with self._acp_activation_lock:
                if not _owns_activation():
                    return False
                activation_model = model
                if (
                    not explicit_card
                    and target_project is not None
                    and getattr(target_project, "acp_tool_name", None) == tool
                ):
                    # A selector queued just before this user task may have
                    # committed a newer model while the task was waiting.  A
                    # saved-selection task consumes that committed value; it
                    # must not switch the session or project back to the stale
                    # value captured at ingress.
                    activation_model = getattr(
                        target_project,
                        "acp_model_name",
                        None,
                    )
                previous_handler_model = (
                    handler.current_model
                    if (
                        thread_root_id is None
                        and handler
                        and hasattr(handler, "current_model")
                    )
                    else None
                )
                if (
                    thread_root_id is None
                    and handler
                    and hasattr(handler, "current_model")
                ):
                    # ``enter_mode`` receives the same explicit override.  This
                    # assignment keeps existing handler display/context behavior
                    # while the explicit input prevents a later selection from
                    # changing an already-running callback's startup model.
                    handler.current_model = activation_model
            try:
                activation_effect = self._enter_mode_with_acp_model(
                    message_id,
                    chat_id,
                    tool,
                    activation_model,
                    target_project,
                    thread_id=thread_root_id,
                )
            except Exception as exc:
                logger.exception(
                    "[ACP] background model activation failed chat=%s project=%s tool=%s model=%s",
                    chat_id,
                    project_id or "-",
                    tool,
                    activation_model or "<default>",
                )
                if (
                    _owns_activation()
                    and thread_root_id is None
                    and handler
                    and hasattr(handler, "current_model")
                ):
                    handler.current_model = previous_handler_model
                _publish_failure(safe_error_message(exc))
                return False

            if not activation_effect:
                if (
                    _owns_activation()
                    and thread_root_id is None
                    and handler
                    and hasattr(handler, "current_model")
                ):
                    handler.current_model = previous_handler_model
                _publish_failure(_system_module().UI_TEXT["system_acp_activation_failed_safe"])
                return False

            if isinstance(activation_effect, _system_module()._ACPActivationEffect):
                effect_session = activation_effect.session
                effect_changed = activation_effect.changed
            elif activation_effect is True:
                effect_session = None
                effect_changed = False
            else:
                # Compatibility for narrow test/custom handlers that still
                # return the historical raw session object.
                effect_session = activation_effect
                effect_changed = True

            def _retire_changed_effect() -> None:
                # Close only the exact session created/replaced by this stale
                # activation.  The manager identity guard cannot remove a
                # newer owner installed at the same coordinate.
                if effect_session is None or not effect_changed:
                    return
                effect_manager = getattr(
                    handler,
                    "_get_session_manager",
                    lambda: None,
                )()
                retire_session = getattr(
                    effect_manager,
                    "retire_session",
                    None,
                )
                if not callable(retire_session):
                    return
                try:
                    retire_session(
                        chat_id,
                        project_id=project_id,
                        thread_id=thread_root_id,
                        expected_session=effect_session,
                        timeout=getattr(
                            self.settings,
                            "acp_startup_timeout",
                            20,
                        ),
                    )
                except Exception:
                    logger.warning(
                        "[ACP] failed to retire stale activation session "
                        "chat=%s project=%s tool=%s",
                        chat_id,
                        project_id or "-",
                        tool,
                        exc_info=True,
                    )

            if not _owns_activation():
                # `/exit` or a newer explicit selection won while startup was
                # outside the lifecycle lock.
                _retire_changed_effect()
                return False

            session = None
            try:
                # Linearization point: a newer selection or /exit either wins
                # before this block (and this activation becomes stale), or waits
                # until durable project state, direct-mode state, and the visible
                # ready card all describe the same session.
                with self._acp_activation_lock:
                    if not _owns_activation():
                        return False
                    if target_project:
                        if (
                            self.project_manager.get_active_project(chat_id)
                            is not target_project
                        ):
                            _retire_changed_effect()
                            _publish_failure(
                                _system_module().UI_TEXT["system_acp_activation_failed_safe"]
                            )
                            return False
                        manager = getattr(
                            handler,
                            "_get_session_manager",
                            lambda: None,
                        )()
                        session = (
                            manager.get_session(
                                chat_id,
                                project_id=project_id,
                                thread_id=thread_root_id,
                            )
                            if manager
                            else None
                        )
                        if session is None or (
                            effect_session is not None
                            and session is not effect_session
                        ):
                            _retire_changed_effect()
                            if (
                                thread_root_id is None
                                and handler
                                and hasattr(handler, "current_model")
                            ):
                                handler.current_model = previous_handler_model
                            _publish_failure(
                                _system_module().UI_TEXT["system_acp_activation_failed_safe"]
                            )
                            return False
                        committed = True
                        if thread_root_id is None:
                            committed = self.project_manager.commit_acp_programming_activation(
                                target_project,
                                tool_name=tool,
                                model_name=activation_model,
                                session_id=session.session_id,
                                query_count=session.message_count,
                                activate_mode=True,
                            )
                        if not committed:
                            # Startup/replacement and live setModel have already
                            # changed the backend at this point.  If the durable
                            # project commit fails, retire that exact owner so
                            # runtime state cannot silently diverge from the
                            # saved tool/model selection.
                            _retire_changed_effect()
                            if (
                                thread_root_id is None
                                and handler
                                and hasattr(handler, "current_model")
                            ):
                                handler.current_model = previous_handler_model
                            _publish_failure(
                                _system_module().UI_TEXT["system_acp_config_save_failed"]
                            )
                            return False
                        if thread_root_id is None:
                            # Keep durable project selection and runtime mode on
                            # the same side of the activation/exit boundary.
                            handler._exit_opposite_mode(
                                message_id,
                                chat_id,
                                project=target_project,
                                silent=True,
                            )
                            handler._enter_mode_on_manager(
                                chat_id,
                                project_id=project_id,
                            )
                    elif effect_session is not None:
                        session = effect_session

                    if explicit_card:
                        live_cmds = None
                        cmds_getter = getattr(session, "get_available_commands", None)
                        if callable(cmds_getter):
                            raw_live_cmds = cmds_getter()
                            if isinstance(raw_live_cmds, (list, tuple)):
                                live_cmds = raw_live_cmds
                        _msg_type, ready_content = (
                            _system_module().SystemBuilder.build_acp_programming_ready_card(
                                tool,
                                activation_model,
                                project_id,
                                thread_root_id,
                                live_commands=live_cmds,
                            )
                        )
                        _replace_status_card(ready_content)
                        _mark_terminal(committed=True)
                    else:
                        # Keep a task-bearing owner live until its pending
                        # prompt has actually been handed to the exact session.
                        activation_owner.committed = True
            except Exception as exc:
                _retire_changed_effect()
                logger.exception(
                    "[ACP] model activation finalization failed chat=%s project=%s tool=%s model=%s",
                    chat_id,
                    project_id or "-",
                    tool,
                    activation_model or "<default>",
                )
                if (
                    _owns_activation()
                    and thread_root_id is None
                    and handler
                    and hasattr(handler, "current_model")
                ):
                    handler.current_model = previous_handler_model
                _publish_failure(safe_error_message(exc))
                return False

            if (
                pending_prompt
                and handler
                and hasattr(handler, "handle_message")
            ):
                if not _owns_activation():
                    return False
                try:
                    handle_kwargs = (
                        {"expected_session": session}
                        if session is not None
                        else {}
                    )
                    handler.handle_message(
                        message_id,
                        chat_id,
                        pending_prompt,
                        target_project,
                        **handle_kwargs,
                    )
                except Exception:
                    logger.exception(
                        "[ACP] pending prompt dispatch failed after activation "
                        "chat=%s project=%s tool=%s",
                        chat_id,
                        project_id or "-",
                        tool,
                    )
                    _publish_failure(
                        _system_module().UI_TEXT["system_acp_activation_failed_safe"]
                    )
                    return False
            if not explicit_card:
                _mark_terminal(committed=True)
            return True

        try:
            activation_handle = self.scheduler.submit(spec, _run_activation)
            add_done_callback = getattr(
                activation_handle,
                "add_done_callback",
                None,
            )
            if callable(add_done_callback):
                add_done_callback(
                    lambda event: (
                        _publish_cancelled()
                        if event.status is TaskStatus.CANCELED
                        else _publish_failure(
                            _system_module().UI_TEXT["system_acp_activation_failed_safe"]
                        )
                        if event.status is TaskStatus.FAILED
                        else None
                    )
                )
        except Exception as exc:
            logger.exception(
                "[ACP] failed to schedule model activation chat=%s project=%s tool=%s",
                chat_id,
                project_id or "-",
                tool,
            )
            _publish_failure(safe_error_message(exc))

    def _resolve_current_acp_tool(self, chat_id: str, project: Optional["ProjectContext"] = None) -> str:
        """Resolve the ACP tool name relevant to the current context.

        Priority:
        1. project.acp_tool_name (explicit tool set on active project)
        2. Current programming interaction mode
        3. Default: "coco"
        """
        if project and getattr(project, "acp_tool_name", ""):
            from ...project.context import ProjectContext

            tool_name = ProjectContext.normalize_acp_tool_name(
                project.acp_tool_name
            )
            if tool_name:
                project.acp_tool_name = tool_name
                return tool_name
            project.acp_tool_name = None
            project.acp_model_name = None

        project_id = project.project_id if project else None
        mode = self.mode_manager.get_mode(chat_id, project_id=project_id)
        if mode in _system_module().PROGRAMMING_MODES:
            return mode.value

        return "coco"

    def handle_model_command(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Show or atomically update the saved project model."""
        parts = (text or "").strip().split(maxsplit=1)
        requested = parts[1].strip() if len(parts) > 1 else ""
        project = self._configuration_project(chat_id, project)
        tool_name = self._resolve_current_acp_tool(chat_id, project)
        if not requested:
            self.show_explicit_acp_model_selection(
                message_id,
                chat_id,
                tool_name,
                project,
            )
            return
        cwd = (project.root_path if project else None) or self.get_working_dir(chat_id)
        current_model: Optional[str] = None
        if project and getattr(project, "acp_tool_name", "") == tool_name:
            current_model = getattr(project, "acp_model_name", None)
        model_names = self._model_names(
            self._fetch_acp_models(
                tool_name,
                cwd=cwd,
                current_model=current_model,
            )
        )
        if requested.lower() in {"list", "ls"}:
            self.reply_text(
                message_id,
                _system_module().UI_TEXT["system_model_config_summary"].format(
                    tool=tool_name,
                    current=current_model or "default",
                    available=" · ".join(f"`{name}`" for name in model_names) or "default",
                ),
            )
            return
        model_name = None if requested.lower() == "default" else requested
        if model_name and model_names and model_name not in model_names:
            self.reply_error(
                message_id,
                _system_module().UI_TEXT["system_acp_unknown_model"].format(model=model_name),
            )
            return
        _tool_spec = BACKEND_META_BY_KEY.get(tool_name)
        if _tool_spec is not None and _tool_spec.uses_claude_cli and model_name:
            from ...acp.claude_capabilities import (
                is_1m_variant,
                model_supports_1m,
                strip_1m_suffix,
            )
            from ...acp.claude_selection import split_claude_model_selection

            model_base, _effort = split_claude_model_selection(model_name)
            if (
                model_base
                and is_1m_variant(model_base)
                and not model_supports_1m(strip_1m_suffix(model_base))
            ):
                self.reply_error(
                    message_id,
                    _system_module().UI_TEXT["system_acp_unsupported_1m_model"].format(
                        model=strip_1m_suffix(model_base),
                    ),
                )
                return
        if project is None or not self.project_manager.commit_acp_configuration(
            project,
            tool_name=tool_name,
            model_name=model_name,
        ):
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
            return
        sync_note = self._sync_live_acp_model(chat_id, project, tool_name, model_name)
        self.reply_text(
            message_id,
            _system_module().UI_TEXT["system_model_config_saved"].format(
                tool=tool_name,
                model=model_name or "default",
            )
            + sync_note,
        )
