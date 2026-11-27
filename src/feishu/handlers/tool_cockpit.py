"""Tool usage (/usage) and settings (/settings) cockpit mixin for SystemHandler."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

from ...card.actions import dispatch as action_ids
from ...card.backend_meta import VISIBLE_BACKEND_METAS, get_backend_meta
from ...mode import PROGRAMMING_MODE_VALUES
from ...mode.backends import BACKEND_META_BY_KEY
from ...project import get_project_acp_model_name, get_project_acp_tool_name
from ..slash_command_parser import SlashCommandParser

if TYPE_CHECKING:
    from ...project import ProjectContext

logger = logging.getLogger(__name__)

OPEN_TOOL_USAGE = getattr(action_ids, "OPEN_TOOL_USAGE", "open_tool_usage")
REFRESH_TOOL_USAGE = getattr(action_ids, "REFRESH_TOOL_USAGE", "refresh_tool_usage")
OPEN_TOOL_SETTINGS = getattr(action_ids, "OPEN_TOOL_SETTINGS", "open_tool_settings")
REFRESH_TOOL_SETTINGS = getattr(action_ids, "REFRESH_TOOL_SETTINGS", "refresh_tool_settings")
SETTINGS_SELECT_TOOL = getattr(action_ids, "SETTINGS_SELECT_TOOL", "settings_select_tool")
SETTINGS_SELECT_MODEL_GROUP = getattr(action_ids, "SETTINGS_SELECT_MODEL_GROUP", "settings_select_model_group")
SETTINGS_SELECT_MODEL_PROFILE = getattr(action_ids, "SETTINGS_SELECT_MODEL_PROFILE", "settings_select_model_profile")
SETTINGS_SELECT_MODEL_EFFORT = getattr(action_ids, "SETTINGS_SELECT_MODEL_EFFORT", "settings_select_model_effort")
SETTINGS_SELECT_ORCH_PLANNER_MODEL = getattr(action_ids, "SETTINGS_SELECT_ORCH_PLANNER_MODEL", "settings_select_orch_planner_model")
SETTINGS_SELECT_ORCH_EXECUTOR_MODEL = getattr(action_ids, "SETTINGS_SELECT_ORCH_EXECUTOR_MODEL", "settings_select_orch_executor_model")
SETTINGS_TOGGLE_SHOW_THINKING = getattr(action_ids, "SETTINGS_TOGGLE_SHOW_THINKING", "settings_toggle_show_thinking")
SETTINGS_SELECT_VERBOSITY = getattr(action_ids, "SETTINGS_SELECT_VERBOSITY", "settings_select_verbosity")
SETTINGS_SELECT_AGENT_MODE = getattr(action_ids, "SETTINGS_SELECT_AGENT_MODE", "settings_select_agent_mode")
SETTINGS_CLEAR_SESSION = getattr(action_ids, "SETTINGS_CLEAR_SESSION", "settings_clear_session")

TOOL_COCKPIT_ACTION_TYPES: frozenset[str] = getattr(
    action_ids,
    "TOOL_COCKPIT_ACTIONS",
    frozenset(
        {
            OPEN_TOOL_USAGE,
            REFRESH_TOOL_USAGE,
            OPEN_TOOL_SETTINGS,
            REFRESH_TOOL_SETTINGS,
            SETTINGS_SELECT_TOOL,
            SETTINGS_SELECT_MODEL_GROUP,
            SETTINGS_SELECT_MODEL_PROFILE,
            SETTINGS_SELECT_MODEL_EFFORT,
            SETTINGS_TOGGLE_SHOW_THINKING,
            SETTINGS_SELECT_VERBOSITY,
            SETTINGS_SELECT_AGENT_MODE,
            SETTINGS_CLEAR_SESSION,
        }
    ),
)

_VALID_VERBOSITIES = frozenset({"low", "medium", "high"})
_VALID_AGENT_MODES = frozenset({"auto", "accept-edits", "plan"})


def _system_module():
    """Return the system handler module for runtime-patchable symbols."""
    from . import system as _sys_mod

    return _sys_mod


class ToolCockpitMixin:
    """Mixin implementing `/usage`, `/settings`, and Tool Cockpit card callbacks."""

    def _safe_active_project(self, chat_id: str) -> Optional["ProjectContext"]:
        """Return active project for *chat_id* without raising on stub contexts."""
        pm = getattr(self, "project_manager", None)
        if pm is None or not callable(getattr(pm, "get_active_project", None)):
            return None
        try:
            return pm.get_active_project(chat_id)
        except Exception:
            logger.debug("Failed to look up active project for chat=%s", chat_id, exc_info=True)
            return None

    def _safe_get_handler(self, key: str) -> Optional[Any]:
        """Look up handler *key* safely."""
        try:
            return self.get_handler(key)
        except Exception:
            return None

    def _cockpit_cwd(self, chat_id: str, project: Optional["ProjectContext"]) -> str:
        """Resolve working directory from *project* or chat working dir."""
        root_path = getattr(project, "root_path", None) if project is not None else None
        if isinstance(root_path, str) and root_path.strip():
            return root_path
        get_wd = getattr(self, "get_working_dir", None)
        if callable(get_wd):
            try:
                wd = get_wd(chat_id)
                if isinstance(wd, str) and wd.strip():
                    return wd
            except Exception:
                logger.debug("get_working_dir failed for chat=%s", chat_id, exc_info=True)
        return "."

    def _resolve_live_session_for_tool(
        self,
        chat_id: str,
        tool_name: str,
        project: Optional["ProjectContext"] = None,
        *,
        thread_root_id: Optional[str] = None,
    ) -> Optional[Any]:
        """Fetch live ACP/CLI session for *tool_name* when active."""
        handler = self._safe_get_handler(tool_name)
        manager_getter = getattr(handler, "_get_session_manager", None) if handler is not None else None
        manager = manager_getter() if callable(manager_getter) else None
        if manager is None or not hasattr(manager, "get_session"):
            return None
        try:
            return manager.get_session(
                chat_id,
                project_id=self._project_id(project),
                thread_id=thread_root_id,
            )
        except Exception:
            return None

    def _safe_active_mode_key(
        self,
        chat_id: str,
        project: Optional["ProjectContext"],
    ) -> Optional[str]:
        """Return active programming mode key or None when unavailable."""
        try:
            mode_key = self._resolve_active_programming_mode_key(chat_id, project)
            return mode_key if isinstance(mode_key, str) and mode_key else None
        except Exception:
            return None

    def _safe_current_acp_tool(
        self,
        chat_id: str,
        project: Optional["ProjectContext"],
    ) -> str:
        """Return configured or fallback ACP tool key for *chat_id*."""
        try:
            tool = self._resolve_current_acp_tool(chat_id, project)
            if isinstance(tool, str) and tool:
                return tool
        except Exception:
            pass
        return str(get_project_acp_tool_name(project, chat_id) or "coco")

    def _resolve_cockpit_tool(
        self,
        message_id: str,
        chat_id: str,
        text_or_tool: str,
        project: Optional["ProjectContext"],
    ) -> Optional[str]:
        """Parse command/card tool argument or resolve current tool key."""
        from ...acp.native_commands import normalize_native_tool_key

        raw_input = str(text_or_tool or "").strip()
        if raw_input.startswith("/"):
            parsed = SlashCommandParser.parse(raw_input)
            raw_arg = (parsed.args if parsed else "").strip()
        else:
            raw_arg = raw_input

        if raw_arg:
            token = raw_arg.split(maxsplit=1)[0]
            tool_name = normalize_native_tool_key(token)
            if tool_name not in PROGRAMMING_MODE_VALUES:
                self.reply_error(
                    message_id,
                    _system_module().UI_TEXT["system_acp_unsupported_tool"].format(tool_name=token),
                )
                return None
            return tool_name

        resolved = (
            self._safe_active_mode_key(chat_id, project)
            or self._safe_current_acp_tool(chat_id, project)
        )
        return normalize_native_tool_key(resolved) or "coco"

    def _extract_selected_option(self, value: Mapping[str, Any]) -> Optional[str]:
        """Extract selected `select_static` option from Feishu callback payload."""
        selected_helper = getattr(self, "_selected_option", None)
        if callable(selected_helper):
            picked = selected_helper(dict(value))
            if picked:
                return picked
        for key in ("_option", "selected_option", "option"):
            raw = value.get(key)
            if isinstance(raw, dict):
                raw = raw.get("value")
            text = str(raw or "").strip()
            if text:
                return text
        return None

    def _available_cockpit_tools(self, current_tool: str) -> list[str]:
        """Return selectable backend keys ensuring *current_tool* is present."""
        is_tool_available = getattr(_system_module(), "is_programming_tool_available", None)
        tools: list[str] = []
        if callable(is_tool_available):
            for meta in VISIBLE_BACKEND_METAS:
                try:
                    if is_tool_available(
                        meta.mode_key,
                        allow_sync_probe=False,
                        trigger_async_probe=True,
                    ):
                        tools.append(meta.mode_key)
                except Exception:
                    continue
        if not tools:
            tools = [meta.mode_key for meta in VISIBLE_BACKEND_METAS]
        if current_tool and current_tool not in tools:
            tools.append(current_tool)
        return tools

    def _safe_fetch_models(
        self,
        tool_name: str,
        *,
        cwd: str,
        current_model: Optional[str],
    ) -> list[Any]:
        """Fetch ACP model options without raising on unavailable backends."""
        try:
            models = self._fetch_acp_models(
                tool_name,
                cwd=cwd,
                current_model=current_model,
            )
            return list(models or ())
        except Exception:
            logger.debug("Failed to fetch ACP models for tool=%s", tool_name, exc_info=True)
            return []

    @staticmethod
    def _get_saved_tool_runtime_settings(
        project: Optional["ProjectContext"],
        chat_id: str,
        tool_name: str,
    ) -> dict[str, Any]:
        """Read chat-scoped runtime settings for *tool_name* from *project*."""
        if project is None:
            return {}
        getter = getattr(project, "get_tool_runtime_settings", None)
        if callable(getter):
            try:
                saved = getter(chat_id, tool_name)
                return dict(saved) if isinstance(saved, Mapping) else {}
            except TypeError:
                saved = getter(tool_name)
                return dict(saved) if isinstance(saved, Mapping) else {}
        fallback = getattr(project, "_tool_runtime_settings_fallback", None)
        if isinstance(fallback, Mapping):
            entry = fallback.get((str(chat_id or "").strip(), tool_name))
            return dict(entry) if isinstance(entry, Mapping) else {}
        return {}

    @staticmethod
    def _save_tool_runtime_settings(
        project: "ProjectContext",
        chat_id: str,
        tool_name: str,
        settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write chat-scoped runtime settings for *tool_name* onto *project*."""
        payload = dict(settings)
        setter = getattr(project, "set_tool_runtime_settings", None)
        if callable(setter):
            try:
                res = setter(chat_id, tool_name, payload)
                return dict(res) if isinstance(res, Mapping) else payload
            except TypeError:
                res = setter(tool_name, payload)
                return dict(res) if isinstance(res, Mapping) else payload
        fallback = getattr(project, "_tool_runtime_settings_fallback", None)
        if not isinstance(fallback, dict):
            fallback = {}
            setattr(project, "_tool_runtime_settings_fallback", fallback)
        fallback[(str(chat_id or "").strip(), tool_name)] = dict(payload)
        return payload

    def _persist_cockpit_project(self, project: Optional["ProjectContext"]) -> None:
        """Persist updated *project* state via ProjectManager when available."""
        if project is None:
            return
        pm = getattr(self, "project_manager", None)
        if pm is None:
            return
        persist = getattr(pm, "persist_project_context", None)
        if callable(persist):
            try:
                persist(project)
                return
            except Exception:
                logger.warning("Failed to persist project context in cockpit", exc_info=True)
                return
        save_all = getattr(pm, "save_projects", None)
        if callable(save_all):
            try:
                save_all()
            except Exception:
                logger.warning("Failed to save projects in cockpit", exc_info=True)

    @staticmethod
    def _apply_runtime_settings_to_session(
        session: Any,
        settings: Mapping[str, Any] | None,
    ) -> bool:
        """Forward runtime settings onto *session* when supported."""
        if session is None or not settings:
            return False
        setter = getattr(session, "set_runtime_settings", None)
        if callable(setter):
            try:
                setter(dict(settings))
                return True
            except Exception:
                logger.debug("Failed to apply runtime settings to live session", exc_info=True)
        return False

    def _collect_usage_snapshot_safe(
        self,
        tool_name: str,
        *,
        cwd: str,
        model_name: Optional[str],
        session: Any,
    ) -> Any:
        """Collect usage telemetry via Slice A or synthesize a fallback snapshot."""
        try:
            from ...acp.tool_telemetry import collect_tool_usage_snapshot

            return collect_tool_usage_snapshot(
                tool_name,
                cwd=cwd,
                model_name=model_name,
                session=session,
            )
        except ImportError:
            usage_getter = getattr(session, "get_usage_info", None) if session is not None else None
            usage = usage_getter() if callable(usage_getter) else None
            return SimpleNamespace(
                tool_name=tool_name,
                model_name=model_name or (getattr(session, "_model_name", None) if session is not None else None),
                session_id=getattr(session, "session_id", None) if session is not None else None,
                message_count=int(getattr(session, "message_count", 0) or 0) if session is not None else 0,
                created_at=getattr(session, "created_at", None) if session is not None else None,
                usage=usage,
                quota_groups=(),
                remaining_credits=None,
                capability_note="当前环境尚未加载工具配额遥测模块，仅展示会话基础统计。",
                error=None,
            )

    def _collect_settings_snapshot_safe(
        self,
        tool_name: str,
        *,
        cwd: str,
        model_name: Optional[str],
        session: Any,
        saved_runtime_settings: Mapping[str, Any],
    ) -> Any:
        """Collect settings telemetry via Slice A or synthesize a fallback snapshot."""
        try:
            from ...acp.tool_telemetry import collect_tool_settings_snapshot

            return collect_tool_settings_snapshot(
                tool_name,
                cwd=cwd,
                model_name=model_name,
                session=session,
                saved_runtime_settings=dict(saved_runtime_settings),
            )
        except ImportError:
            session_rt_getter = getattr(session, "get_runtime_settings", None) if session is not None else None
            session_rt = session_rt_getter() if callable(session_rt_getter) else {}
            merged = {**(session_rt if isinstance(session_rt, Mapping) else {}), **dict(saved_runtime_settings)}
            return SimpleNamespace(
                tool_name=tool_name,
                model_name=model_name,
                show_thinking=bool(merged.get("show_thinking", True)),
                verbosity=str(merged.get("verbosity") or "medium"),
                agent_mode=str(merged.get("agent_mode") or "auto"),
                config_options=(),
                backend_config={},
                error=None,
            )

    def _build_usage_card_payload(
        self,
        tool_name: str,
        usage_snapshot: Any,
        *,
        project_name: Optional[str],
        project_id: Optional[str],
        thread_root_id: Optional[str],
    ) -> tuple[str, str]:
        """Build `/usage` Card 2.0 payload via ToolCockpitBuilder or fallback."""
        try:
            from ...card.builders.tool_cockpit import ToolCockpitBuilder

            return ToolCockpitBuilder.build_tool_usage_card(
                tool_name,
                usage_snapshot,
                project_name=project_name,
                project_id=project_id,
                thread_root_id=thread_root_id,
            )
        except ImportError:
            from ...card.builders.core import CoreBuilder
            from ...card.shared import build_callback_button, build_responsive_layout

            meta = get_backend_meta(tool_name)
            label = meta.display_name if meta else tool_name.upper()
            sid = getattr(usage_snapshot, "session_id", None) or "无活跃会话"
            msg_count = int(getattr(usage_snapshot, "message_count", 0) or 0)
            model = getattr(usage_snapshot, "model_name", None) or "default"
            note = getattr(usage_snapshot, "capability_note", None) or ""
            lines = [
                f"**工具**：`{tool_name}` · **模型**：`{model}`",
                f"**会话**：`{sid}` · **轮数**：{msg_count}",
            ]
            if project_name:
                lines.insert(0, f"**项目**：{project_name}")
            if note:
                lines.append(note)
            base_payload = {"tool_name": tool_name, "project_id": project_id}
            if thread_root_id:
                base_payload["thread_root_id"] = thread_root_id
            buttons = [
                build_callback_button(
                    text="🔄 刷新用量",
                    action={**base_payload, "action": REFRESH_TOOL_USAGE},
                    button_type="primary",
                ),
                build_callback_button(
                    text="⚙️ 工具设置",
                    action={**base_payload, "action": OPEN_TOOL_SETTINGS},
                ),
            ]
            elements = [
                {"tag": "markdown", "content": "\n".join(lines)},
                {"tag": "hr"},
                *build_responsive_layout(buttons, layout="mobile"),
            ]
            card = CoreBuilder._wrap_card(f"📊 {label} 用量与配额", "blue", elements)
            card["config"]["update_multi"] = True
            return "interactive", json.dumps(card, ensure_ascii=False)

    def _build_settings_card_payload(
        self,
        tool_name: str,
        *,
        models: list[Any],
        current_model: Optional[str],
        available_tools: list[str],
        settings_snapshot: Any,
        project_name: Optional[str],
        project_id: Optional[str],
        thread_root_id: Optional[str],
        status_banner: str = "",
    ) -> tuple[str, str]:
        """Build `/settings` Card 2.0 payload via ToolCockpitBuilder or fallback."""
        try:
            from ...card.builders.tool_cockpit import ToolCockpitBuilder

            return ToolCockpitBuilder.build_tool_settings_card(
                tool_name,
                models=models,
                current_model=current_model,
                available_tools=available_tools,
                settings_snapshot=settings_snapshot,
                project_name=project_name,
                project_id=project_id,
                thread_root_id=thread_root_id,
                status_banner=status_banner,
            )
        except ImportError:
            from ...card.builders.core import CoreBuilder
            from ...card.shared import build_callback_button, build_responsive_layout

            meta = get_backend_meta(tool_name)
            label = meta.display_name if meta else tool_name.upper()
            base_payload = {"tool_name": tool_name, "project_id": project_id}
            if thread_root_id:
                base_payload["thread_root_id"] = thread_root_id
            elements: list[dict[str, Any]] = []
            if status_banner:
                elements.append(CoreBuilder._build_banner_element(status_banner, type="success"))
            elements.append(
                {
                    "tag": "markdown",
                    "content": (
                        f"**工具**：`{tool_name}` · **模型**：`{current_model or 'default'}`\n"
                        f"**可用工具**：{', '.join(f'`{t}`' for t in available_tools)}"
                    ),
                }
            )
            elements.append({"tag": "hr"})
            buttons = [
                build_callback_button(
                    text="📊 查看用量",
                    action={**base_payload, "action": OPEN_TOOL_USAGE},
                ),
                build_callback_button(
                    text="🧹 清空会话",
                    action={**base_payload, "action": SETTINGS_CLEAR_SESSION},
                ),
                build_callback_button(
                    text="🔄 刷新",
                    action={**base_payload, "action": REFRESH_TOOL_SETTINGS},
                    button_type="primary",
                ),
            ]
            elements.extend(build_responsive_layout(buttons, layout="mobile"))
            card = CoreBuilder._wrap_card(f"⚙️ {label} 设置驾驶舱", "blue", elements)
            card["config"]["update_multi"] = True
            return "interactive", json.dumps(card, ensure_ascii=False)

    def _deliver_cockpit_card(
        self,
        message_id: str,
        msg_type: str,
        card_content: str,
        *,
        origin_message_id: Optional[str] = None,
        in_place_card_id: Optional[str] = None,
    ) -> None:
        """Update existing cockpit card in-place or reply with a new card."""
        if in_place_card_id:
            if self.update_card(in_place_card_id, card_content):
                return
        target_mid = origin_message_id or message_id
        if msg_type == "text":
            self.reply_text(target_mid, card_content)
        else:
            self.reply_card(target_mid, card_content)

    def show_tool_usage(
        self,
        message_id: str,
        chat_id: str,
        text_or_tool: str = "",
        project: Optional["ProjectContext"] = None,
        *,
        origin_message_id: Optional[str] = None,
        in_place_card_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> None:
        """Display the `/usage` quota, credits, and session usage cockpit card."""
        target_project = project or self._safe_active_project(chat_id)
        tool_name = self._resolve_cockpit_tool(message_id, chat_id, text_or_tool, target_project)
        if tool_name is None:
            return
        cwd = self._cockpit_cwd(chat_id, target_project)
        live_session = self._resolve_live_session_for_tool(
            chat_id,
            tool_name,
            target_project,
            thread_root_id=thread_root_id,
        )
        current_model = (
            get_project_acp_model_name(target_project, chat_id, tool_name=tool_name)
            if target_project is not None
            else None
        )
        if current_model is None and live_session is not None:
            raw_live_model = getattr(live_session, "_model_name", None)
            if isinstance(raw_live_model, str) and raw_live_model.strip():
                current_model = raw_live_model.strip()

        snapshot = self._collect_usage_snapshot_safe(
            tool_name,
            cwd=cwd,
            model_name=current_model,
            session=live_session,
        )
        msg_type, card_content = self._build_usage_card_payload(
            tool_name,
            snapshot,
            project_name=getattr(target_project, "project_name", None) if target_project else None,
            project_id=self._project_id(target_project),
            thread_root_id=thread_root_id,
        )
        self._deliver_cockpit_card(
            message_id,
            msg_type,
            card_content,
            origin_message_id=origin_message_id,
            in_place_card_id=in_place_card_id,
        )

    def show_tool_settings(
        self,
        message_id: str,
        chat_id: str,
        text_or_tool: str = "",
        project: Optional["ProjectContext"] = None,
        *,
        origin_message_id: Optional[str] = None,
        in_place_card_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
        status_banner: str = "",
        refresh_models: bool = False,
    ) -> None:
        """Display the `/settings` tool, model, effort, and runtime settings cockpit card."""
        target_project = project or self._safe_active_project(chat_id)
        tool_name = self._resolve_cockpit_tool(message_id, chat_id, text_or_tool, target_project)
        if tool_name is None:
            return
        cwd = self._cockpit_cwd(chat_id, target_project)
        if refresh_models:
            invalidate = getattr(_system_module(), "invalidate_acp_model_cache", None)
            if callable(invalidate):
                try:
                    invalidate(tool_name, cwd)
                except Exception:
                    logger.debug("invalidate_acp_model_cache failed for tool=%s", tool_name, exc_info=True)

        current_model = (
            get_project_acp_model_name(target_project, chat_id, tool_name=tool_name)
            if target_project is not None
            else None
        )
        models = self._safe_fetch_models(tool_name, cwd=cwd, current_model=current_model)
        live_session = self._resolve_live_session_for_tool(
            chat_id,
            tool_name,
            target_project,
            thread_root_id=thread_root_id,
        )
        saved_runtime = self._get_saved_tool_runtime_settings(target_project, chat_id, tool_name)
        settings_snapshot = self._collect_settings_snapshot_safe(
            tool_name,
            cwd=cwd,
            model_name=current_model,
            session=live_session,
            saved_runtime_settings=saved_runtime,
        )
        available_tools = self._available_cockpit_tools(tool_name)
        msg_type, card_content = self._build_settings_card_payload(
            tool_name,
            models=models,
            current_model=current_model,
            available_tools=available_tools,
            settings_snapshot=settings_snapshot,
            project_name=getattr(target_project, "project_name", None) if target_project else None,
            project_id=self._project_id(target_project),
            thread_root_id=thread_root_id,
            status_banner=status_banner,
        )
        self._deliver_cockpit_card(
            message_id,
            msg_type,
            card_content,
            origin_message_id=origin_message_id,
            in_place_card_id=in_place_card_id,
        )

    def handle_tool_cockpit_action(
        self,
        message_id: str,
        chat_id: str,
        project_id: Optional[str],
        value: dict,
    ) -> None:
        """Dispatch all 12 `/usage` and `/settings` Tool Cockpit card callbacks."""
        from ...acp.native_commands import normalize_native_tool_key

        payload = dict(value or {})
        action = str(payload.get("action") or "").strip()
        if project_id:
            project = self._action_project(chat_id, project_id)
            if project is None:
                self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
                return
        else:
            project = self._safe_active_project(chat_id)

        thread_root_id = str(payload.get("thread_root_id") or "").strip() or None
        raw_tool = str(payload.get("tool_name") or "").strip()
        if raw_tool:
            tool_name = normalize_native_tool_key(raw_tool)
        else:
            tool_name = normalize_native_tool_key(
                self._safe_active_mode_key(chat_id, project)
                or self._safe_current_acp_tool(chat_id, project)
            )
        if not tool_name or tool_name not in PROGRAMMING_MODE_VALUES:
            self.reply_error(
                message_id,
                _system_module().UI_TEXT["system_acp_unsupported_tool"].format(tool_name=raw_tool or tool_name),
            )
            return

        if action == OPEN_TOOL_USAGE:
            self.show_tool_usage(
                message_id,
                chat_id,
                tool_name,
                project,
                origin_message_id=message_id,
                thread_root_id=thread_root_id,
            )
            return

        if action == REFRESH_TOOL_USAGE:
            self.show_tool_usage(
                message_id,
                chat_id,
                tool_name,
                project,
                origin_message_id=message_id,
                in_place_card_id=message_id,
                thread_root_id=thread_root_id,
            )
            return

        if action == OPEN_TOOL_SETTINGS:
            self.show_tool_settings(
                message_id,
                chat_id,
                tool_name,
                project,
                origin_message_id=message_id,
                thread_root_id=thread_root_id,
            )
            return

        if action == REFRESH_TOOL_SETTINGS:
            self.show_tool_settings(
                message_id,
                chat_id,
                tool_name,
                project,
                origin_message_id=message_id,
                in_place_card_id=message_id,
                thread_root_id=thread_root_id,
                status_banner="🔄 已刷新设置与模型目录",
                refresh_models=True,
            )
            return

        if action == SETTINGS_SELECT_TOOL:
            self._handle_cockpit_select_tool(
                message_id,
                chat_id,
                payload,
                project,
                thread_root_id=thread_root_id,
            )
            return

        if action in {
            SETTINGS_SELECT_MODEL_GROUP,
            SETTINGS_SELECT_MODEL_PROFILE,
            SETTINGS_SELECT_MODEL_EFFORT,
            SETTINGS_SELECT_ORCH_PLANNER_MODEL,
            SETTINGS_SELECT_ORCH_EXECUTOR_MODEL,
        }:
            self._handle_cockpit_select_model_dimension(
                message_id,
                chat_id,
                action,
                tool_name,
                payload,
                project,
                thread_root_id=thread_root_id,
            )
            return

        if action in {
            SETTINGS_TOGGLE_SHOW_THINKING,
            SETTINGS_SELECT_VERBOSITY,
            SETTINGS_SELECT_AGENT_MODE,
        }:
            self._handle_cockpit_runtime_setting(
                message_id,
                chat_id,
                action,
                tool_name,
                payload,
                project,
                thread_root_id=thread_root_id,
            )
            return

        if action == SETTINGS_CLEAR_SESSION:
            self._handle_cockpit_clear_session(
                message_id,
                chat_id,
                tool_name,
                project,
                thread_root_id=thread_root_id,
            )
            return

        logger.warning("Unhandled tool cockpit action: action=%s", action)

    def _ensure_cockpit_mutable_project(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"],
    ) -> Optional["ProjectContext"]:
        """Return *project* or resolve/create active config project for mutations."""
        target = project or self._configuration_project(chat_id, None)
        if target is None:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
            return None
        return target

    def _handle_cockpit_select_tool(
        self,
        message_id: str,
        chat_id: str,
        payload: dict[str, Any],
        project: Optional["ProjectContext"],
        *,
        thread_root_id: Optional[str],
    ) -> None:
        """Handle `SETTINGS_SELECT_TOOL` dropdown selection and sync live mode."""
        from ...acp.native_commands import normalize_native_tool_key

        selected = self._extract_selected_option(payload)
        new_tool = normalize_native_tool_key(selected)
        if not new_tool or new_tool not in PROGRAMMING_MODE_VALUES:
            self.reply_error(
                message_id,
                _system_module().UI_TEXT["system_acp_unsupported_tool"].format(tool_name=selected or ""),
            )
            return

        target_project = self._ensure_cockpit_mutable_project(message_id, chat_id, project)
        if target_project is None:
            return

        previous_tool = normalize_native_tool_key(get_project_acp_tool_name(target_project, chat_id))
        if previous_tool == new_tool:
            model_name = get_project_acp_model_name(target_project, chat_id, tool_name=new_tool)
        else:
            new_handler = self._safe_get_handler(new_tool)
            get_scoped = (
                getattr(type(new_handler), "get_scoped_model", None)
                if new_handler is not None
                else None
            )
            if callable(get_scoped):
                scoped_res = new_handler.get_scoped_model(chat_id, target_project)
                if isinstance(scoped_res, tuple) and len(scoped_res) == 2:
                    has_scoped, scoped_model = scoped_res
                    model_name = (
                        scoped_model
                        if has_scoped and isinstance(scoped_model, str) and scoped_model
                        else None
                    )
                else:
                    model_name = None
            else:
                model_name = None

        if not self._commit_acp_configuration_for_chat(
            target_project,
            tool_name=new_tool,
            model_name=model_name,
            chat_id=chat_id,
        ):
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
            return

        banner = f"✅ 已切换默认工具为 `{new_tool}`（模型：`{model_name or 'default'}`）"
        active_key = self._safe_active_mode_key(chat_id, target_project)
        if active_key and active_key != new_tool:
            switched = self._sync_live_acp_tool_switch(
                chat_id,
                target_project,
                old_tool=active_key,
                new_tool=new_tool,
            )
            try:
                target_project.set_programming_mode(active_key, False, chat_id=chat_id)
            except TypeError:
                target_project.set_programming_mode(active_key, False)
            self._persist_cockpit_project(target_project)
            if switched:
                banner += " · 活跃会话已同步切换"

        self.show_tool_settings(
            message_id,
            chat_id,
            new_tool,
            target_project,
            origin_message_id=message_id,
            in_place_card_id=message_id,
            thread_root_id=thread_root_id,
            status_banner=banner,
        )

    def _handle_cockpit_select_model_dimension(
        self,
        message_id: str,
        chat_id: str,
        action: str,
        tool_name: str,
        payload: dict[str, Any],
        project: Optional["ProjectContext"],
        *,
        thread_root_id: Optional[str],
    ) -> None:
        """Handle model group, profile, or effort selection and persist/hot-sync."""
        from ...acp.agy_selection import is_agy_orchestrator_model
        from ...card.render.model_cascade import (
            available_model_names,
            compose_model_selection,
            resolve_agy_orchestrator_cascade,
            resolve_model_cascade,
        )

        selected = self._extract_selected_option(payload)
        if selected is None:
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_select_model_prompt"])
            return

        target_project = self._ensure_cockpit_mutable_project(message_id, chat_id, project)
        if target_project is None:
            return

        cwd = self._cockpit_cwd(chat_id, target_project)
        current_model = get_project_acp_model_name(target_project, chat_id, tool_name=tool_name)
        models = self._safe_fetch_models(tool_name, cwd=cwd, current_model=current_model)
        known_names = available_model_names(models) if models else ()

        group = str(payload.get("model_group") or "").strip() or None
        profile = str(payload.get("model_profile") or "").strip() or None
        planner_model = str(payload.get("planner_model") or "").strip() or None
        executor_model = str(payload.get("executor_model") or "").strip() or None

        def _maybe_orch_cascade(
            orch_model: Optional[str],
            *,
            p_model: Optional[str] = None,
            p_eff: Optional[str] = None,
            e_model: Optional[str] = None,
            e_eff: Optional[str] = None,
        ):
            if not (
                models
                and tool_name in {"agy", "agy_b"}
                and orch_model
                and is_agy_orchestrator_model(orch_model)
            ):
                return None
            probe = resolve_agy_orchestrator_cascade(
                models,
                orchestrator_model=orch_model,
                current_model=current_model,
                pending_planner_model=p_model,
                pending_planner_effort=p_eff,
                pending_executor_model=e_model,
                pending_executor_effort=e_eff,
            )
            return probe if probe.has_role_models else None

        next_model_name: Optional[str]
        if action == SETTINGS_SELECT_MODEL_GROUP:
            if selected.lower() == "default" and "default" not in known_names:
                next_model_name = None
            elif not models:
                next_model_name = selected
            elif selected not in known_names:
                self.reply_error(
                    message_id,
                    _system_module().UI_TEXT["system_acp_unknown_model"].format(model=selected),
                )
                return
            else:
                orch_state = _maybe_orch_cascade(selected)
                if orch_state is not None:
                    next_model_name = orch_state.selection
                else:
                    state = resolve_model_cascade(
                        models,
                        current_model=current_model,
                        selected_model=selected,
                        selected_profile=None,
                        selected_effort=None,
                    )
                    next_model_name = (
                        compose_model_selection(
                            models,
                            model=state.selected_model,
                            profile=state.selected_profile,
                            effort=state.selected_effort,
                        )
                        or state.selection
                    )
        else:
            if not models:
                self.reply_error(
                    message_id,
                    _system_module().UI_TEXT["system_acp_unknown_model"].format(model=selected),
                )
                return
            orch_base = group or (
                current_model.split("/", 1)[0].strip()
                if current_model and is_agy_orchestrator_model(current_model)
                else None
            )
            if action in {
                SETTINGS_SELECT_ORCH_PLANNER_MODEL,
                SETTINGS_SELECT_ORCH_EXECUTOR_MODEL,
            }:
                orch_probe = _maybe_orch_cascade(orch_base)
                if orch_probe is None or selected not in orch_probe.role_model_names:
                    self.reply_error(
                        message_id,
                        _system_module().UI_TEXT["system_acp_unknown_model"].format(model=selected),
                    )
                    return
                if action == SETTINGS_SELECT_ORCH_PLANNER_MODEL:
                    orch_state = _maybe_orch_cascade(
                        orch_base,
                        p_model=selected,
                        p_eff=None,
                        e_model=executor_model,
                        e_eff=str(payload.get("model_effort") or "").strip() or None,
                    )
                else:
                    orch_state = _maybe_orch_cascade(
                        orch_base,
                        p_model=planner_model,
                        p_eff=profile,
                        e_model=selected,
                        e_eff=None,
                    )
                assert orch_state is not None
                next_model_name = orch_state.selection
            elif (
                tool_name in {"agy", "agy_b"}
                and orch_base
                and is_agy_orchestrator_model(orch_base)
                and (orch_probe := _maybe_orch_cascade(
                    orch_base,
                    p_model=planner_model,
                    e_model=executor_model,
                )) is not None
            ):
                effort_val = str(payload.get("model_effort") or "").strip() or None
                if action == SETTINGS_SELECT_MODEL_PROFILE:
                    if orch_probe.planner_efforts and selected.lower() not in orch_probe.planner_efforts:
                        self.reply_error(
                            message_id,
                            _system_module().UI_TEXT["system_acp_unknown_model"].format(model=selected),
                        )
                        return
                    orch_state = _maybe_orch_cascade(
                        orch_base,
                        p_model=planner_model,
                        p_eff=selected,
                        e_model=executor_model,
                        e_eff=effort_val,
                    )
                else:
                    if orch_probe.executor_efforts and selected.lower() not in orch_probe.executor_efforts:
                        self.reply_error(
                            message_id,
                            _system_module().UI_TEXT["system_acp_unknown_model"].format(model=selected),
                        )
                        return
                    orch_state = _maybe_orch_cascade(
                        orch_base,
                        p_model=planner_model,
                        p_eff=profile,
                        e_model=executor_model,
                        e_eff=selected,
                    )
                assert orch_state is not None
                next_model_name = orch_state.selection
            elif action == SETTINGS_SELECT_MODEL_PROFILE:
                state = resolve_model_cascade(
                    models,
                    current_model=current_model,
                    selected_model=group,
                    selected_profile=selected,
                    selected_effort=None,
                )
                if state.profiles and selected.lower() not in state.profiles:
                    self.reply_error(
                        message_id,
                        _system_module().UI_TEXT["system_acp_unknown_model"].format(model=selected),
                    )
                    return
                next_model_name = (
                    compose_model_selection(
                        models,
                        model=state.selected_model,
                        profile=state.selected_profile,
                        effort=state.selected_effort,
                    )
                    or state.selection
                )
            else:
                state = resolve_model_cascade(
                    models,
                    current_model=current_model,
                    selected_model=group,
                    selected_profile=profile,
                    selected_effort=selected,
                )
                if state.efforts and selected.lower() not in state.efforts:
                    self.reply_error(
                        message_id,
                        _system_module().UI_TEXT["system_acp_unknown_model"].format(model=selected),
                    )
                    return
                next_model_name = (
                    compose_model_selection(
                        models,
                        model=state.selected_model,
                        profile=state.selected_profile,
                        effort=state.selected_effort,
                    )
                    or state.selection
                )

        tool_spec = BACKEND_META_BY_KEY.get(tool_name)
        if tool_spec is not None and tool_spec.uses_claude_cli and next_model_name:
            from ...acp.claude_capabilities import (
                is_1m_variant,
                model_supports_1m,
                strip_1m_suffix,
            )
            from ...acp.claude_selection import split_claude_model_selection

            model_base, _effort = split_claude_model_selection(next_model_name)
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

        if not self._commit_acp_configuration_for_chat(
            target_project,
            tool_name=tool_name,
            model_name=next_model_name,
            chat_id=chat_id,
        ):
            self.reply_error(message_id, _system_module().UI_TEXT["system_acp_config_save_failed"])
            return

        sync_note = self._sync_live_acp_model(chat_id, target_project, tool_name, next_model_name)
        sync_clean = sync_note.strip()
        banner = f"✅ 已更新 `{tool_name}` 模型为 `{next_model_name or 'default'}`"
        if sync_clean:
            banner += f" · {sync_clean}"

        self.show_tool_settings(
            message_id,
            chat_id,
            tool_name,
            target_project,
            origin_message_id=message_id,
            in_place_card_id=message_id,
            thread_root_id=thread_root_id,
            status_banner=banner,
        )

    def _handle_cockpit_runtime_setting(
        self,
        message_id: str,
        chat_id: str,
        action: str,
        tool_name: str,
        payload: dict[str, Any],
        project: Optional["ProjectContext"],
        *,
        thread_root_id: Optional[str],
    ) -> None:
        """Handle `show_thinking`, `verbosity`, or `agent_mode` changes."""
        target_project = self._ensure_cockpit_mutable_project(message_id, chat_id, project)
        if target_project is None:
            return

        current_settings = self._get_saved_tool_runtime_settings(target_project, chat_id, tool_name)
        updated = dict(current_settings)
        live_session = self._resolve_live_session_for_tool(
            chat_id,
            tool_name,
            target_project,
            thread_root_id=thread_root_id,
        )

        if action == SETTINGS_TOGGLE_SHOW_THINKING:
            if isinstance(payload.get("next_show_thinking"), bool):
                new_val = payload["next_show_thinking"]
            else:
                cur_val = current_settings.get("show_thinking")
                if cur_val is None and live_session is not None:
                    getter = getattr(live_session, "get_runtime_settings", None)
                    live_rt = getter() if callable(getter) else {}
                    if isinstance(live_rt, Mapping) and "show_thinking" in live_rt:
                        cur_val = live_rt.get("show_thinking")
                if cur_val is None and isinstance(payload.get("current_show_thinking"), bool):
                    cur_val = payload["current_show_thinking"]
                if cur_val is not None:
                    new_val = not bool(cur_val)
                elif isinstance(payload.get("show_thinking"), bool):
                    new_val = payload["show_thinking"]
                else:
                    new_val = False
            updated["show_thinking"] = new_val
            banner = f"✅ 已{'开启' if new_val else '关闭'}思考过程显示"
        elif action == SETTINGS_SELECT_VERBOSITY:
            selected = self._extract_selected_option(payload)
            verbosity = str(selected or "").strip().lower()
            if verbosity not in _VALID_VERBOSITIES:
                self.reply_error(message_id, f"⚠️ 无效的输出详略选项: {selected or '空'}")
                return
            updated["verbosity"] = verbosity
            banner = f"✅ 已设置输出详略为 `{verbosity}`"
        else:
            selected = self._extract_selected_option(payload)
            agent_mode = str(selected or "").strip().lower()
            if agent_mode not in _VALID_AGENT_MODES:
                self.reply_error(message_id, f"⚠️ 无效的代理模式选项: {selected or '空'}")
                return
            updated["agent_mode"] = agent_mode
            banner = f"✅ 已设置代理模式为 `{agent_mode}`"

        saved = self._save_tool_runtime_settings(target_project, chat_id, tool_name, updated)
        self._persist_cockpit_project(target_project)
        self._apply_runtime_settings_to_session(live_session, saved)

        self.show_tool_settings(
            message_id,
            chat_id,
            tool_name,
            target_project,
            origin_message_id=message_id,
            in_place_card_id=message_id,
            thread_root_id=thread_root_id,
            status_banner=banner,
        )

    def _handle_cockpit_clear_session(
        self,
        message_id: str,
        chat_id: str,
        tool_name: str,
        project: Optional["ProjectContext"],
        *,
        thread_root_id: Optional[str],
    ) -> None:
        """Clear session context while keeping tool, model, and runtime settings."""
        target_project = project or self._safe_active_project(chat_id)
        target_tool = (
            tool_name
            or self._safe_active_mode_key(chat_id, target_project)
            or self._safe_current_acp_tool(chat_id, target_project)
        )
        handler = self._safe_get_handler(target_tool)
        clear_fn = getattr(handler, "clear_session_context", None) if handler is not None else None
        if callable(clear_fn):
            if thread_root_id:
                try:
                    clear_fn(chat_id, target_project, thread_id=thread_root_id)
                except TypeError:
                    clear_fn(chat_id, target_project)
            else:
                clear_fn(chat_id, target_project)
        self._persist_cockpit_project(target_project)

        self.show_tool_settings(
            message_id,
            chat_id,
            target_tool,
            target_project,
            origin_message_id=message_id,
            in_place_card_id=message_id,
            thread_root_id=thread_root_id,
            status_banner="✅ 已清空当前会话上下文",
        )
