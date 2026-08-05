import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ws_client import FeishuWSClient

from ..card import CardBuilder
from ..card.actions import dispatch as action_ids
from ..card.ui_text import UI_TEXT
from .dispatch_context import DispatchContext
from .slash_command_parser import SlashCommandParser

logger = logging.getLogger(__name__)


def _display_mode_label(mode: str) -> str:
    labels = {
        "coco": "Coco",
        "claude": "Claude",
        "claude cli": "Claude CLI",
        "aiden": "Aiden",
        "codex": "Codex",
        "gemini": "Gemini",
        "traex": "Traex",
        "ttadk": "TTADK",
    }
    raw = str(mode or "").strip()
    return labels.get(raw.lower(), raw)


class _RetryDispatchAdapter:
    """Adapter that bridges ``FeishuWSClient`` to ``RetryDispatchProtocol``.

    Each method delegates to the corresponding (often private) attribute on
    the concrete client, so ``RetryCommandHandler`` never touches them
    directly.  This is the **only** place that accesses ``ws_client``
    internals for retry dispatch.
    """

    def __init__(self, client: "FeishuWSClient") -> None:
        self._client = client

    def reply_text(self, message_id: str, text: str) -> None:
        self._client._reply_text(message_id, text)

    def try_block_with_chat_lock(
        self, chat_id: str, sender_id: str, message_id: str, *, raw_text: str = "",
    ) -> bool:
        # Best-effort: keep retry dispatch compatible with the main gate API.
        # Parse once and pass the structured CommandMatch to the gate.
        m = SlashCommandParser.parse(raw_text)
        return self._client._chat_lock_gate.check(
            chat_id, sender_id, message_id, command_match=m,
        )

    def get_project_for_chat(self, project_id: str, chat_id: str) -> Any:
        return self._client._project_manager.get_project_for_chat(project_id, chat_id)

    def get_active_project(self, chat_id: str) -> Any:
        return self._client._project_manager.get_active_project(chat_id)

    def get_repo_lock_manager(self) -> Any:
        ctx = self._client._handler_ctx
        if ctx is None:
            return None
        return getattr(ctx, "repo_lock_manager", None)

    def process_with_intent(
        self, message_id: str, chat_id: str, text: str, project: Any,
    ) -> None:
        self._client._process_with_intent(message_id, chat_id, text, project)

    def send_lock_conflict_card(
        self,
        e: Any,
        message_id: str,
        command_text: str,
        *,
        retry_count: int = 0,
        chat_id: str = "",
    ) -> None:
        self._client.send_lock_conflict_card(
            e,
            message_id,
            command_text,
            retry_count=retry_count,
            chat_id=chat_id,
        )


def _resolve_project(client: "FeishuWSClient", pid: str | None, cid: str):
    """Resolve project from pid+cid, falling back to chat's active project when pid is absent."""
    if pid:
        return DispatchContext(project_manager=client._project_manager).resolve_project(pid, cid)
    # Fallback: get the active project for this chat when no explicit project_id
    return client._project_manager.get_active_project(cid)


def init_action_registry(client: 'FeishuWSClient') -> None:
    """Initialize all card action handlers and register them to the client's action dispatcher."""
    register_programming_mode_actions(client)

    # Project
    client._register_action(
        lambda mid, cid, pid, val: client._show_project_status(
            mid, cid, _resolve_project(client, pid, cid)
        ),
        exact=action_ids.SHOW_STATUS,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._show_project_board(mid, cid, origin_message_id=mid), exact=action_ids.SWITCH_PROJECT
    )
    client._register_action(
        lambda mid, cid, pid, val: client._show_project_board(mid, cid, origin_message_id=mid), exact=action_ids.SHOW_BOARD
    )
    client._register_action(
        lambda mid, cid, pid, val: client._show_project_board(mid, cid, origin_message_id=mid), exact=action_ids.REFRESH_BOARD
    )
    client._register_action(
        lambda mid, cid, pid, val: client._show_project_board(
            mid, cid, origin_message_id=mid, page=val.get("page", 1)
        ),
        exact=action_ids.SWITCH_BOARD_PAGE,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._show_project_status(
            mid, cid, _resolve_project(client, pid, cid), origin_message_id=mid
        ),
        exact=action_ids.SHOW_DETAIL,
    )

    def _handle_switch_to(mid, cid, pid, val):
        project = _resolve_project(client, pid, cid)
        if project:
            client._switch_project(mid, cid, project.project_name)
        else:
            client._reply_text(mid, UI_TEXT["lock_project_not_found_hint"])

    client._register_action(_handle_switch_to, exact=action_ids.SWITCH_TO)

    def _handle_continue_dev(mid, cid, pid, val):
        project = _resolve_project(client, pid, cid)
        if project:
            client._project_manager.set_active_project(cid, pid)
            content = f"继续在 **{project.project_name}** 项目中开发\n\n📂 项目目录: `{project.root_path}`\n\n直接发送命令或消息即可"
            msg_type, card_content = CardBuilder.build_project_response_card(
                project, "继续开发", content, show_buttons=True
            )
            response_id = client._reply_card(mid, card_content)
            if response_id:
                client._register_message_project(response_id, project)
        else:
            client._reply_text(mid, UI_TEXT["lock_project_not_found_hint"])

    client._register_action(_handle_continue_dev, exact=action_ids.CONTINUE_DEV)

    def _handle_list_files(mid, cid, pid, val):
        project = _resolve_project(client, pid, cid)
        if project:
            client._project_manager.set_active_project(cid, pid)
            client._submit_shell_command(mid, cid, "ls -la", project.root_path, project)
        else:
            client._reply_text(mid, UI_TEXT["lock_project_not_found_hint"])

    client._register_action(_handle_list_files, exact=action_ids.LIST_FILES)

    client._register_action(
        lambda mid, cid, pid, val: client._reply_text(
            mid, "📝 创建新项目\n\n请发送: `/new 项目名 路径`\n\n例如: `/new myApp ~/workspace/myApp`"
        ),
        exact=action_ids.NEW_PROJECT_PROMPT,
    )

    client._register_action(
        lambda mid, cid, pid, val: client._handle_select_ttadk_tool(
            mid, cid, val.get("_option") or val.get("tool_name", ""), pid
        ),
        exact=action_ids.SELECT_TTADK_TOOL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_toggle_ttadk_yolo(
            mid,
            cid,
            bool(val.get("enabled")),
            val.get("view", "tool_select"),
            val.get("tool_name", ""),
            pid,
        ),
        exact=action_ids.TOGGLE_TTADK_YOLO,
    )


def register_programming_mode_actions(client: 'FeishuWSClient') -> None:
    """Register enter/exit/resume/new actions for all programming modes."""
    mode_names = ("coco", "claude", "aiden", "codex", "gemini", "traex", "ttadk", "tui2acp")
    for mode in mode_names:
        enter = getattr(client, f"_handle_card_enter_{mode}")
        exit_ = getattr(client, f"_handle_card_exit_{mode}")
        resume = getattr(client, f"_handle_card_resume_{mode}")
        new = getattr(client, f"_handle_card_new_{mode}")

        client._register_action(enter, exact=f"enter_{mode}")
        client._register_action(exit_, exact=f"exit_{mode}")
        client._register_action(
            lambda mid, cid, pid, val, _resume=resume: _resume(mid, cid, pid, val.get("session_id", "")),
            exact=f"resume_{mode}",
        )
        client._register_action(new, exact=f"new_{mode}")
    client._register_action(
        lambda mid, cid, pid, val: client._handle_select_ttadk_model(
            mid,
            cid,
            val.get("tool_name", ""),
            val.get("_option") or val.get("model_name", ""),
            _resolve_project(client, pid, cid),
        ),
        exact=action_ids.SELECT_TTADK_MODEL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_refresh_ttadk_models(mid, cid, val.get("tool_name", ""), pid),
        exact=action_ids.REFRESH_TTADK_MODELS,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_select_ttadk_combined(
            mid,
            cid,
            val.get("tool_name", ""),
            val.get("_option") or val.get("model_name", ""),
            _resolve_project(client, pid, cid),
        ),
        exact=action_ids.SELECT_TTADK_COMBINED,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_select_ttadk_combined_tool(
            mid, cid, val.get("_option", ""), _resolve_project(client, pid, cid),
        ),
        exact=action_ids.SELECT_TTADK_COMBINED_TOOL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_ttadk_command(
            mid, cid, _resolve_project(client, pid, cid), True
        ),
        exact=action_ids.SHOW_TTADK_MENU,
    )

    # Tui2ACP
    client._register_action(
        lambda mid, cid, pid, val: client._handle_select_tui2acp_adapter(
            mid, cid, val.get("adapter_name", ""), pid
        ),
        exact=action_ids.SELECT_TUI2ACP_ADAPTER,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_select_tui2acp_custom_command(
            mid, cid, val.get("_form_value", {}).get("custom_command", ""), pid
        ),
        exact=action_ids.SELECT_TUI2ACP_CUSTOM_COMMAND,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_tui2acp_command(
            mid, cid, _resolve_project(client, pid, cid), True
        ),
        exact=action_ids.SHOW_TUI2ACP_MENU,
    )

    client._register_action(
        lambda mid, cid, pid, val: client._handle_spec_review_use_auto(mid, cid, pid, val),
        exact=action_ids.SPEC_REVIEW_USE_AUTO,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_spec_review_finish_selection(mid, cid, pid, val),
        exact=action_ids.SPEC_REVIEW_FINISH_SELECTION,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_spec_review_select_tool(mid, cid, pid, val),
        exact=action_ids.SPEC_REVIEW_SELECT_TOOL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_spec_review_select_model(mid, cid, pid, val),
        exact=action_ids.SPEC_REVIEW_SELECT_MODEL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_spec_review_remove_item(mid, cid, pid, val),
        exact=action_ids.SPEC_REVIEW_REMOVE_ITEM,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_spec_review_clear_items(mid, cid, pid, val),
        exact=action_ids.SPEC_REVIEW_CLEAR_ITEMS,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_spec_review_menu(mid, cid, pid, val),
        exact=action_ids.SHOW_SPEC_REVIEW_MENU,
    )
    # Workflow
    client._register_action(
        lambda mid, cid, pid, val: client._handle_show_workflow_menu(mid, cid, pid, val),
        exact=action_ids.SHOW_WORKFLOW_MENU,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_confirm_tools(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_CONFIRM_TOOLS,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_confirm_start(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_CONFIRM_START,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_cancel(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_CANCEL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_stop_running(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_STOP_RUNNING,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_select_tool(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_SELECT_TOOL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_regenerate_script(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_REGENERATE_SCRIPT,
    )
    # Workflow two-step selection flow
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_orchestrator_select_tool(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_ORCHESTRATOR_SELECT_TOOL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_orchestrator_select_model_group(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_ORCHESTRATOR_SELECT_MODEL_GROUP,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_orchestrator_select_model_profile(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_ORCHESTRATOR_SELECT_MODEL_PROFILE,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_orchestrator_select_model_effort(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_ORCHESTRATOR_SELECT_MODEL_EFFORT,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_orchestrator_select_model(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_ORCHESTRATOR_SELECT_MODEL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_orchestrator_remove(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_ORCHESTRATOR_REMOVE,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_orchestrator_clear(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_ORCHESTRATOR_CLEAR,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_orchestrator_finish(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_ORCHESTRATOR_FINISH,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_review_select_tool(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_REVIEW_SELECT_TOOL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_review_select_model_group(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_REVIEW_SELECT_MODEL_GROUP,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_review_select_model_profile(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_REVIEW_SELECT_MODEL_PROFILE,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_review_select_model_effort(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_REVIEW_SELECT_MODEL_EFFORT,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_review_select_model(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_REVIEW_SELECT_MODEL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_review_finish(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_REVIEW_FINISH,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_review_remove(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_REVIEW_REMOVE,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_review_clear(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_REVIEW_CLEAR,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_review_toggle_auto(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_REVIEW_TOGGLE_AUTO,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_fill_missing_tools(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_FILL_MISSING_TOOLS,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_back_to_tools(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_BACK_TO_TOOLS,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_list_templates(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_LIST_TEMPLATES,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_show_help(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_SHOW_HELP,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_view_workflow_ref(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_VIEW_WORKFLOW_REF,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_remove_workflow_ref(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_REMOVE_WORKFLOW_REF,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_workflow_add_workflow_ref(mid, cid, pid, val),
        exact=action_ids.WORKFLOW_ADD_WORKFLOW_REF,
    )

    # ACP
    client._register_action(
        lambda mid, cid, pid, val: client._handle_acp_command(
            mid, cid, _resolve_project(client, pid, cid)
        ),
        exact=action_ids.SHOW_ACP_MENU,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_select_acp_tool(mid, cid, val.get("tool_name", ""), pid),
        exact=action_ids.SELECT_ACP_TOOL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_select_acp_model(
            mid,
            cid,
            val.get("tool_name", ""),
            None if val.get("use_default_model") else val.get("model_name", ""),
            _resolve_project(client, pid, cid),
        ),
        exact=action_ids.SELECT_ACP_MODEL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_refresh_acp_models(mid, cid, val.get("tool_name", ""), pid, val),
        exact=action_ids.REFRESH_ACP_MODELS,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_acp_model_cascade_select(
            mid, cid, val.get("tool_name", ""), pid, val
        ),
        exact=action_ids.SELECT_ACP_MODEL_GROUP,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_acp_model_cascade_select(
            mid, cid, val.get("tool_name", ""), pid, val
        ),
        exact=action_ids.SELECT_ACP_MODEL_PROFILE,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_acp_model_cascade_select(
            mid, cid, val.get("tool_name", ""), pid, val
        ),
        exact=action_ids.SELECT_ACP_MODEL_EFFORT,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._slock_handler.handle_new_role_select_tool(
            mid, cid, val, _resolve_project(client, pid, cid)
        ),
        exact=action_ids.SLOCK_NEW_ROLE_SELECT_TOOL,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._slock_handler.handle_new_role_tool_dropdown_change(
            mid, cid, val, _resolve_project(client, pid, cid)
        ),
        exact=action_ids.SLOCK_NEW_ROLE_SELECT_TOOL_DROPDOWN,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._slock_handler.handle_new_role_select_model(
            mid, cid, val, _resolve_project(client, pid, cid)
        ),
        exact=action_ids.SLOCK_NEW_ROLE_SELECT_MODEL,
    )
    for action in (
        action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_GROUP,
        action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_PROFILE,
        action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_EFFORT,
    ):
        client._register_action(
            lambda mid, cid, pid, val: client._slock_handler.handle_new_role_model_cascade_select(
                mid, cid, val, _resolve_project(client, pid, cid)
            ),
            exact=action,
        )

    # System
    client._register_action(
        lambda mid, cid, pid, val: client._show_full_help(
            mid, cid, _resolve_project(client, pid, cid)
        ),
        exact=action_ids.SHOW_HELP_MENU,
    )
    client._register_action(lambda mid, cid, pid, val: client._handle_deep_prompt(mid, cid), exact=action_ids.ENTER_DEEP_PROMPT)
    client._register_action(
        lambda mid, cid, pid, val: client._handle_force_release_repo_lock(mid, cid, pid, val),
        exact=action_ids.FORCE_RELEASE_REPO_LOCK,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_confirm_lock(mid, cid, pid, val),
        exact=action_ids.CONFIRM_LOCK,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_cancel_lock(mid, cid, pid, val),
        exact=action_ids.CANCEL_LOCK,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_confirm_force_release(mid, cid, pid, val),
        exact=action_ids.CONFIRM_FORCE_RELEASE,
    )
    client._register_action(
        lambda mid, cid, pid, val: client._handle_cancel_force_release(mid, cid, pid, val),
        exact=action_ids.CANCEL_FORCE_RELEASE,
    )

    from .retry_handler import RetryCommandHandler
    client._register_action(RetryCommandHandler(_RetryDispatchAdapter(client)), exact=action_ids.RETRY_COMMAND)
    def _handle_continue_degraded(mid, cid, pid, val):
        mode = str((val or {}).get("degraded_to") or "").strip()
        if mode:
            message = UI_TEXT["card_lifecycle_continue_degraded_ack"].format(mode=_display_mode_label(mode))
        else:
            message = UI_TEXT["card_lifecycle_continue_degraded_unknown_ack"]
        client._reply_text(mid, message)

    client._register_action(_handle_continue_degraded, exact=action_ids.CONTINUE_DEGRADED)

    def _handle_show_error_details(mid, cid, pid, val):
        from src.card.error_diagnostics import render_error_diagnostic

        client._reply_text(
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

    def _handle_retry_original(mid, cid, pid, val):
        from .retry_original import RetryOriginalModeUseCase

        decision = RetryOriginalModeUseCase()(mid, cid, pid, dict(val or {}))
        client._reply_text(mid, decision.message)

    client._register_action(_handle_show_error_details, exact=action_ids.SHOW_ERROR_DETAILS)
    client._register_action(_handle_retry_original, exact=action_ids.RETRY_ORIGINAL)

    # Approval — dispatches APPROVAL_RESOLVED event to update CardSession state
    def _handle_approval(mid, cid, pid, val, *, approved: bool):
        logger.info("Approval action: approved=%s, message_id=%s, chat_id=%s", approved, mid, cid)
        client._reply_text(mid, "✅ 已批准操作" if approved else "❌ 已拒绝操作")
        # Dispatch APPROVAL_RESOLVED to the engine that rendered the approval card
        try:
            engine_type = val.get("engine_type", "")
            approval_val = {**val, "approved": approved}
            if engine_type == "deep":
                client._deep_handler.handle_card_action(mid, cid, "deep_approval_resolved", approval_val)
            elif engine_type == "spec":
                client._spec_handler.handle_card_action(mid, cid, "spec_approval_resolved", approval_val)
            else:
                # Fallback: try deep handler (approval may not have engine_type in value)
                logger.debug("approval: no engine_type, trying deep")
                client._deep_handler.handle_card_action(mid, cid, "deep_approval_resolved", approval_val)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Failed to dispatch APPROVAL_RESOLVED for message_id=%s: %s", mid, repr(exc))

    client._register_action(
        lambda mid, cid, pid, val: _handle_approval(mid, cid, pid, val, approved=True),
        exact=action_ids.APPROVE_ACTION,
    )
    client._register_action(
        lambda mid, cid, pid, val: _handle_approval(mid, cid, pid, val, approved=False),
        exact=action_ids.REJECT_ACTION,
    )

    client._register_action(
        lambda mid, cid, pid, val: client._handle_help_category(
            mid,
            cid,
            val.get("category", "main"),
            _resolve_project(client, pid, cid),
            origin_message_id=mid,
        ),
        exact=action_ids.HELP_CATEGORY,
    )

    # Deep Engine
    client._register_action(
        lambda mid, cid, pid, val: client._show_deep_status(
            mid, cid, _resolve_project(client, pid, cid), origin_message_id=mid
        ),
        exact=action_ids.SHOW_DEEP_STATUS,
    )
    client._register_action(
        lambda mid, cid, pid, val, type=None: client._deep_handler.handle_card_action(mid, cid, type, val),
        prefix="deep_",
    )

    # Spec Engine
    client._register_action(
        lambda mid, cid, pid, val, type=None: client._spec_handler.handle_card_action(mid, cid, type, val),
        prefix="spec_",
    )
    client._register_action(
        lambda mid, cid, pid, val, type=None: client._slock_handler.handle_card_action(mid, cid, type, val),
        prefix="slock_",
    )
    client._register_action(
        lambda mid, cid, pid, val, type=None: client._slock_handler.handle_card_action(mid, cid, type, val),
        prefix="employee_runtime_",
    )

    # Generic ENGINE_STOP — routes to correct handler based on engine_type in value
    def _handle_engine_stop(mid, cid, pid, val):
        engine_type = val.get("engine_type", "")
        # Remap to engine-specific stop action and delegate to the correct handler
        if engine_type == "deep":
            client._deep_handler.handle_card_action(mid, cid, "deep_stop", val)
        elif engine_type == "spec":
            client._spec_handler.handle_card_action(mid, cid, "spec_stop", val)
        else:
            logger.warning("engine_stop: unknown engine_type=%s, trying deep handler", engine_type)
            client._deep_handler.handle_card_action(mid, cid, "deep_stop", val)

    client._register_action(_handle_engine_stop, exact=action_ids.ENGINE_STOP)
