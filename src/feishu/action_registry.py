import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ws_client import FeishuWSClient

from ..card.actions import dispatch as action_ids
from ..card.ui_text import UI_TEXT
from ..mode import PROGRAMMING_MODES, InteractionMode
from ..project import get_project_acp_tool_name
from ..utils.signing import VerifyResult, verify_command_sig
from .slash_command_parser import SlashCommandParser

logger = logging.getLogger(__name__)


def _resolve_project(client: "FeishuWSClient", pid: str | None, cid: str):
    if pid:
        return client._project_manager.get_project_for_chat(pid, cid)
    return client._project_manager.get_active_project(cid)


_REGISTERED_ACTION_TYPES: frozenset[str] = frozenset()


def is_registered_action(action_type: str) -> bool:
    """Return whether the exact callback table accepts an action."""
    return action_type in _REGISTERED_ACTION_TYPES


def init_action_registry(client: "FeishuWSClient") -> dict[str, Callable[[str, str, str | None, dict], Any]]:
    """Build the exact card-action table against authoritative handlers."""
    global _REGISTERED_ACTION_TYPES
    actions: dict[str, Callable[[str, str, str | None, dict], Any]] = {}
    handlers = client._handler_ctx.handlers
    base = handlers["coco"]
    project_handler = handlers["project"]
    system = handlers["system"]
    deep = handlers["deep"]
    spec = handlers["spec"]
    workflow = handlers["workflow"]

    for interaction_mode in (
        mode for mode in InteractionMode if mode in PROGRAMMING_MODES
    ):
        mode = interaction_mode.value

        def _show_model_selector(mid, cid, pid, val, *, _mode=mode):
            system.show_explicit_acp_model_selection(
                mid,
                cid,
                _mode,
                _resolve_project(client, pid, cid),
                origin_message_id=mid,
            )

        actions[f"enter_{mode}"] = _show_model_selector
        # Compatibility for cards rendered before the shared exit action.
        mode_handler = handlers.get(mode)
        actions[f"exit_{mode}"] = (
            mode_handler.handle_card_exit
            if mode_handler is not None and hasattr(mode_handler, "handle_card_exit")
            else (lambda mid, cid, pid, val: system.exit_current_mode(mid, cid, _resolve_project(client, pid, cid)))
        )

    # Project
    actions[action_ids.SHOW_STATUS] = (
        lambda mid, cid, pid, val: project_handler.show_project_status(
            mid, cid, _resolve_project(client, pid, cid)
        )
    )
    for action in (action_ids.SWITCH_PROJECT, action_ids.SHOW_BOARD, action_ids.REFRESH_BOARD):
        actions[action] = lambda mid, cid, pid, val: project_handler.show_project_board(
            mid, cid, origin_message_id=mid,
        )
    actions[action_ids.SWITCH_BOARD_PAGE] = (
        lambda mid, cid, pid, val: project_handler.show_project_board(
            mid, cid, origin_message_id=mid, page=val.get("page", 1)
        )
    )
    actions[action_ids.SHOW_DETAIL] = (
        lambda mid, cid, pid, val: project_handler.show_project_status(
            mid, cid, _resolve_project(client, pid, cid), origin_message_id=mid
        )
    )

    def _handle_switch_to(mid, cid, pid, val):
        project = _resolve_project(client, pid, cid)
        if project:
            project_handler.switch_project(
                mid,
                cid,
                project.project_name,
                auto_enter_coco=True,
                coco_handler=base,
                claude_handler=handlers.get("claude"),
            )
        else:
            base.reply_text(mid, UI_TEXT["lock_project_not_found_hint"])

    actions[action_ids.SWITCH_TO] = _handle_switch_to

    def _handle_list_files(mid, cid, pid, val):
        project = _resolve_project(client, pid, cid)
        if project:
            client._project_manager.set_active_project(cid, pid)
            system.submit_shell_command(mid, cid, "ls -la", project.root_path, project)
        else:
            base.reply_text(mid, UI_TEXT["lock_project_not_found_hint"])

    actions[action_ids.LIST_FILES] = _handle_list_files

    actions[action_ids.SHOW_WORKFLOW_MENU] = (
        lambda mid, cid, pid, val: workflow.show_workflow_help(mid)
    )
    actions[action_ids.WORKFLOW_STOP_RUNNING] = workflow.handle_workflow_stop_running
    for action in action_ids.WORKFLOW_AGENT_SELECTION_ACTIONS:
        actions[action] = workflow.handle_workflow_agent_action
    for action in action_ids.WF_CONFIG_ACTIONS:
        actions[action] = workflow.handle_workflow_config_action

    # System
    actions[action_ids.EXIT_PROGRAMMING_MODE] = (
        lambda mid, cid, pid, val: system.exit_current_mode(
            mid, cid, _resolve_project(client, pid, cid)
        )
    )
    actions[action_ids.SHOW_HELP_MENU] = (
        lambda mid, cid, pid, val: system.show_full_help(
            mid, cid, _resolve_project(client, pid, cid)
        )
    )
    actions[action_ids.ENTER_DEEP_PROMPT] = (
        lambda mid, cid, pid, val: system.handle_deep_prompt(mid, cid)
    )
    actions.update({
        action_ids.FORCE_RELEASE_REPO_LOCK: system.handle_force_release_repo_lock,
        action_ids.CONFIRM_FORCE_RELEASE: system.handle_confirm_force_release,
        action_ids.CONFIRM_LOCK: system.handle_confirm_lock,
        action_ids.CANCEL_LOCK: system.handle_cancel_lock,
        action_ids.CANCEL_FORCE_RELEASE: system.handle_cancel_force_release,
    })

    def _handle_show_error_details(mid, cid, pid, val):
        from src.card.error_diagnostics import render_error_diagnostic

        base.reply_text(
            mid,
            render_error_diagnostic(
                val.get("diagnostic_token"),
                chat_id=cid,
                # Diagnostic records are bound to the original triggering
                # message when the card is built.  In a real card click, ``mid``
                # is the card message being clicked, so using it here would
                # reject legitimate clicks.  Prefer the explicit payload binding
                # and fall back to ``mid`` only for older cards without it.
                origin_message_id=val.get("origin_message_id") or mid,
                request_id=val.get("request_id"),
                trace_id=val.get("trace_id"),
            ),
        )

    actions[action_ids.SHOW_ERROR_DETAILS] = _handle_show_error_details

    actions[action_ids.HELP_CATEGORY] = (
        lambda mid, cid, pid, val: system.handle_help_category(
            mid,
            cid,
            val.get("category", "main"),
            _resolve_project(client, pid, cid),
            origin_message_id=mid,
        )
    )
    actions[action_ids.SELECT_ACP_MODEL] = system.handle_select_acp_model
    for action in (
        action_ids.SELECT_ACP_MODEL_GROUP,
        action_ids.SELECT_ACP_MODEL_PROFILE,
        action_ids.SELECT_ACP_MODEL_EFFORT,
    ):
        actions[action] = system.handle_acp_model_cascade_select
    actions[action_ids.OPEN_ACP_MODEL_PICKER] = system.handle_open_acp_model_picker
    actions[action_ids.REFRESH_ACP_MODELS] = system.handle_refresh_acp_models
    actions[action_ids.SHOW_TOOL_COMMANDS] = system.handle_show_tool_commands
    actions[action_ids.RUN_NATIVE_TOOL_COMMAND] = system.handle_run_native_tool_command

    def _handle_continue_dev(mid, cid, pid, val):
        project = _resolve_project(client, pid, cid)
        if project is None:
            base.reply_text(mid, UI_TEXT["lock_project_not_found_hint"])
            return
        saved_tool = get_project_acp_tool_name(project, cid)
        if saved_tool:
            system.handle_enter_acp_saved_selection(mid, cid, saved_tool, project)
        else:
            system.show_tools_list(mid, cid, project)

    actions[action_ids.CONTINUE_DEV] = _handle_continue_dev
    actions[action_ids.NEW_PROJECT_PROMPT] = (
        lambda mid, cid, pid, val: base.reply_text(mid, UI_TEXT["project_new_usage_prompt"])
    )

    def _handle_retry_command(mid, cid, pid, val):
        payload = val or {}
        cmd = str(payload.get("_t") or "").strip()
        sig = str(payload.get("_s") or "").strip()
        if not cmd:
            base.reply_error(mid, "⛔ 缺少待重放的命令。")
            return
        verify_res = verify_command_sig(cmd, sig, chat_id=cid)
        if not verify_res:
            if verify_res is VerifyResult.EXPIRED:
                base.reply_error(mid, "⏰ 该操作按钮已过期，请直接重新发送命令。")
            elif verify_res is VerifyResult.NONCE_REUSED:
                base.reply_error(mid, "⚠️ 该操作按钮已执行过，无需重复点击。")
            else:
                base.reply_error(mid, "⛔ 命令签名校验失败，已拒绝执行。")
            return
        if payload.get("_ul"):
            raw_ue = payload.get("_ue")
            try:
                if raw_ue is not None and time.time() > float(raw_ue):
                    base.reply_error(mid, "⏰ 撤销锁定窗口已过，如需解锁请发送 `/unlock`。")
                    return
            except (TypeError, ValueError):
                base.reply_error(mid, "⛔ 无效的撤销窗口参数。")
                return
        project = _resolve_project(client, pid, cid)
        parsed = SlashCommandParser.parse(cmd)
        if parsed is not None and system.is_interceptable_command_match(parsed):
            system.handle_intercepted_command(mid, cid, cmd, project, command_match=parsed)
            return
        dispatcher = getattr(client, "_message_dispatcher", None) or getattr(client, "_dispatcher", None)
        process_fn = getattr(dispatcher, "process_with_intent", None) if dispatcher is not None else None
        if callable(process_fn):
            process_fn(mid, cid, cmd, project)
            return
        if project is not None and getattr(project, "root_path", None):
            system.submit_shell_command(mid, cid, cmd, project.root_path, project)
        else:
            base.reply_text(mid, UI_TEXT["lock_project_not_found_hint"])

    actions[action_ids.RETRY_COMMAND] = _handle_retry_command

    # Deep Engine
    actions[action_ids.SHOW_DEEP_STATUS] = (
        lambda mid, cid, pid, val: deep.show_deep_status(
            mid, cid, _resolve_project(client, pid, cid), origin_message_id=mid
        )
    )
    actions["deep_stop"] = (
        lambda mid, cid, pid, val: deep.handle_card_action(mid, cid, "deep_stop", val)
    )

    # Spec Engine
    actions[action_ids.SPEC_STOP] = (
        lambda mid, cid, pid, val: spec.handle_card_action(mid, cid, "spec_stop", val)
    )

    # Generic ENGINE_STOP — routes to correct handler based on engine_type in value
    def _handle_engine_stop(mid, cid, pid, val):
        engine_type = val.get("engine_type", "")
        # Remap to engine-specific stop action and delegate to the correct handler
        if engine_type == "deep":
            deep.handle_card_action(mid, cid, "deep_stop", val)
        elif engine_type == "spec":
            spec.handle_card_action(mid, cid, "spec_stop", val)
        else:
            logger.warning("engine_stop rejected: unknown engine_type=%s", engine_type)

    actions[action_ids.ENGINE_STOP] = _handle_engine_stop
    actions[action_ids.PROGRAMMING_STOP] = system.handle_programming_stop
    _REGISTERED_ACTION_TYPES = frozenset(actions)
    return actions
