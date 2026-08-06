import logging
from typing import Any

logger = logging.getLogger(__name__)

# Dispatch table: _method_name -> (registry_key, handler_method_name)
FORWARDING_MAP: dict[str, tuple[str, str]] = {
    # --- shared base handler helpers (delegate via coco) ---
    "_add_reaction": ("coco", "add_reaction"),
    "_get_working_dir": ("coco", "get_working_dir"),
    "_set_working_dir": ("coco", "set_working_dir"),
    "_ensure_request_id": ("coco", "ensure_request_id"),
    "_format_ref_note": ("coco", "format_ref_note"),
    "_register_message_project": ("coco", "register_message_project"),
    "_reply_text": ("coco", "reply_text"),
    "_reply_card": ("coco", "reply_card"),
    "_send_card_to_chat": ("coco", "send_card_to_chat"),
    "_send_text_to_chat": ("coco", "send_text_to_chat"),
    "_get_engine_name": ("coco", "get_engine_name"),
    "_record_mode_transition": ("coco", "record_mode_transition"),
    "_inject_bridge_context": ("coco", "inject_bridge_context"),
    # --- Coco mode ---
    "_enter_coco_mode": ("coco", "enter_mode"),
    "_exit_coco_mode": ("coco", "exit_mode"),
    "_handle_coco_message": ("coco", "handle_message"),
    "_handle_coco_response": ("coco", "handle_response"),
    "_show_coco_info": ("coco", "show_info"),
    "_handle_card_enter_coco": ("coco", "handle_card_enter"),
    "_handle_card_exit_coco": ("coco", "handle_card_exit"),
    "_handle_card_resume_coco": ("coco", "handle_card_resume"),
    "_handle_card_new_coco": ("coco", "handle_card_new"),
    # --- Claude mode ---
    "_enter_claude_mode": ("claude", "enter_mode"),
    "_exit_claude_mode": ("claude", "exit_mode"),
    "_handle_claude_message": ("claude", "handle_message"),
    "_handle_claude_response": ("claude", "handle_response"),
    "_show_claude_info": ("claude", "show_info"),
    "_handle_card_enter_claude": ("claude", "handle_card_enter"),
    "_handle_card_exit_claude": ("claude", "handle_card_exit"),
    "_handle_card_resume_claude": ("claude", "handle_card_resume"),
    "_handle_card_new_claude": ("claude", "handle_card_new"),
    # --- Aiden mode ---
    "_enter_aiden_mode": ("aiden", "enter_mode"),
    "_exit_aiden_mode": ("aiden", "exit_mode"),
    "_handle_aiden_message": ("aiden", "handle_message"),
    "_handle_aiden_response": ("aiden", "handle_response"),
    "_show_aiden_info": ("aiden", "show_info"),
    "_handle_card_enter_aiden": ("aiden", "handle_card_enter"),
    "_handle_card_exit_aiden": ("aiden", "handle_card_exit"),
    "_handle_card_resume_aiden": ("aiden", "handle_card_resume"),
    "_handle_card_new_aiden": ("aiden", "handle_card_new"),
    # --- Gemini mode ---
    "_enter_gemini_mode": ("gemini", "enter_mode"),
    "_exit_gemini_mode": ("gemini", "exit_mode"),
    "_handle_gemini_message": ("gemini", "handle_message"),
    "_handle_gemini_response": ("gemini", "handle_response"),
    "_show_gemini_info": ("gemini", "show_info"),
    "_handle_card_enter_gemini": ("gemini", "handle_card_enter"),
    "_handle_card_exit_gemini": ("gemini", "handle_card_exit"),
    "_handle_card_resume_gemini": ("gemini", "handle_card_resume"),
    "_handle_card_new_gemini": ("gemini", "handle_card_new"),
    # --- Codex mode ---
    "_enter_codex_mode": ("codex", "enter_mode"),
    "_exit_codex_mode": ("codex", "exit_mode"),
    "_handle_codex_message": ("codex", "handle_message"),
    "_handle_codex_response": ("codex", "handle_response"),
    "_show_codex_info": ("codex", "show_info"),
    "_handle_card_enter_codex": ("codex", "handle_card_enter"),
    "_handle_card_exit_codex": ("codex", "handle_card_exit"),
    "_handle_card_resume_codex": ("codex", "handle_card_resume"),
    "_handle_card_new_codex": ("codex", "handle_card_new"),
    # --- Traex mode ---
    "_enter_traex_mode": ("traex", "enter_mode"),
    "_exit_traex_mode": ("traex", "exit_mode"),
    "_handle_traex_message": ("traex", "handle_message"),
    "_handle_traex_response": ("traex", "handle_response"),
    "_show_traex_info": ("traex", "show_info"),
    "_handle_card_enter_traex": ("traex", "handle_card_enter"),
    "_handle_card_exit_traex": ("traex", "handle_card_exit"),
    "_handle_card_resume_traex": ("traex", "handle_card_resume"),
    "_handle_card_new_traex": ("traex", "handle_card_new"),
    # --- Tui2ACP mode ---
    "_enter_tui2acp_mode": ("tui2acp", "enter_mode"),
    "_exit_tui2acp_mode": ("tui2acp", "exit_mode"),
    "_handle_tui2acp_message": ("tui2acp", "handle_message"),
    "_handle_tui2acp_response": ("tui2acp", "handle_response"),
    "_show_tui2acp_info": ("tui2acp", "show_info"),
    "_handle_card_enter_tui2acp": ("tui2acp", "handle_card_enter"),
    "_handle_card_exit_tui2acp": ("tui2acp", "handle_card_exit"),
    "_handle_card_resume_tui2acp": ("tui2acp", "handle_card_resume"),
    "_handle_card_new_tui2acp": ("tui2acp", "handle_card_new"),
    "_handle_tui2acp_command": ("system", "handle_tui2acp_command"),
    "_handle_select_tui2acp_adapter": ("system", "handle_select_tui2acp_adapter"),
    "_handle_select_tui2acp_custom_command": ("system", "handle_select_tui2acp_custom_command"),
    "_handle_acp_command": ("system", "handle_acp_command"),
    "_handle_select_acp_tool": ("system", "handle_select_acp_tool"),
    "_handle_select_acp_model": ("system", "handle_select_acp_model"),
    "_handle_refresh_acp_models": ("system", "handle_refresh_acp_models"),
    "_handle_acp_model_cascade_select": ("system", "handle_acp_model_cascade_select"),
    "_handle_help_category": ("system", "handle_help_category"),
    "_handle_deep_prompt": ("system", "handle_deep_prompt"),
    # --- Deep Engine ---
    "_handle_deep_command": ("deep", "handle_deep_command"),
    "_start_deep_engine": ("deep", "start_deep_engine"),
    "_create_deep_callbacks": ("deep", "_create_deep_callbacks"),
    "_show_deep_status": ("deep", "show_deep_status"),
    "_show_deep_board": ("deep", "show_deep_board"),
    "_pause_deep_engine": ("deep", "pause_deep_engine"),
    "_resume_deep_engine": ("deep", "resume_deep_engine"),
    "_stop_deep_engine": ("deep", "stop_deep_engine"),
    "_stop_all_deep_engines": ("deep", "stop_all_deep_engines"),
    "_update_deep_context": ("deep", "update_deep_context"),
    "_toggle_deep_log": ("deep", "_toggle_log"),
    "_switch_deep_card_mode": ("deep", "_switch_card_mode"),
    # --- Spec Engine ---
    "_handle_spec_command": ("spec", "handle_spec_command"),
    "_start_spec_engine": ("spec", "start_spec_engine"),
    "_show_spec_status": ("spec", "show_spec_status"),
    "_pause_spec_engine": ("spec", "pause_spec_engine"),
    "_resume_spec_engine": ("spec", "resume_spec_engine"),
    "_stop_spec_engine": ("spec", "stop_spec_engine"),
    "_update_spec_guidance": ("spec", "update_spec_guidance"),
    "_toggle_spec_log": ("spec", "_toggle_log"),
    "_switch_spec_card_mode": ("spec", "_switch_card_mode"),
    "_toggle_spec_ac": ("spec", "_toggle_ac"),
    "_handle_spec_review_use_auto": ("spec", "handle_spec_review_use_auto"),
    "_handle_spec_review_finish_selection": ("spec", "handle_spec_review_finish_selection"),
    "_handle_spec_review_select_tool": ("spec", "handle_spec_review_select_tool"),
    "_handle_spec_review_select_model": ("spec", "handle_spec_review_select_model"),
    "_handle_spec_review_remove_item": ("spec", "handle_spec_review_remove_item"),
    "_handle_spec_review_clear_items": ("spec", "handle_spec_review_clear_items"),
    "_handle_spec_review_menu": ("spec", "handle_spec_review_menu"),
    # --- Project ---
    "_create_project": ("project", "create_project"),
    "_show_project_board": ("project", "show_project_board"),
    "_show_current_project": ("project", "show_current_project"),
    "_show_project_status": ("project", "show_project_status"),
    "_preserve_project_context": ("project", "preserve_project_context"),
    "_restore_project_context": ("project", "restore_project_context"),
    "_close_project": ("project", "close_project"),
    "_handle_new_chat_project": ("project", "handle_new_chat_project"),
    # --- System ---
    "_show_help": ("system", "show_help"),
    "_show_full_help": ("system", "show_full_help"),
    "_exit_current_mode": ("system", "exit_current_mode"),
    "_submit_shell_command": ("system", "submit_shell_command"),
    "_change_directory": ("system", "change_directory"),
    "_handle_intercepted_command": ("system", "handle_intercepted_command"),
    "_handle_force_release_repo_lock": ("system", "handle_force_release_repo_lock"),
    "_handle_confirm_lock": ("system", "handle_confirm_lock"),
    "_handle_cancel_lock": ("system", "handle_cancel_lock"),
    "_handle_confirm_force_release": ("system", "handle_confirm_force_release"),
    "_handle_cancel_force_release": ("system", "handle_cancel_force_release"),
    # --- Diagnostics ---
    "_show_task_board": ("diagnostics", "show_task_board"),
    "_show_context_diff": ("diagnostics", "show_context_diff"),
    "_build_context_diff_report": ("diagnostics", "_build_context_diff_report"),
    "_submit_diff_report": ("diagnostics", "_submit_diff_report"),
    "_show_message_trace": ("diagnostics", "show_message_trace"),
    # --- Slock Engine ---
    "_handle_slock_command": ("slock", "handle_slock_command"),
    "_handle_slock_message": ("slock", "handle_message"),
    # --- Workflow Engine ---
    "_handle_workflow_command": ("workflow", "handle_workflow_command"),
    "_start_workflow": ("workflow", "start_workflow"),
    "_stop_workflow": ("workflow", "stop_workflow"),
    "_show_workflow_status": ("workflow", "show_workflow_status"),
    "_handle_workflow_confirm_tools": ("workflow", "handle_workflow_confirm_tools"),
    "_handle_workflow_confirm_start": ("workflow", "handle_workflow_confirm_start"),
    "_handle_workflow_cancel": ("workflow", "handle_workflow_cancel"),
    "_handle_workflow_stop_running": ("workflow", "handle_workflow_stop_running"),
    "_handle_workflow_select_tool": ("workflow", "handle_workflow_select_tool"),
    "_handle_workflow_regenerate_script": ("workflow", "handle_workflow_regenerate_script"),
    "_handle_workflow_fill_missing_tools": ("workflow", "handle_workflow_fill_missing_tools"),
    "_handle_workflow_back_to_tools": ("workflow", "handle_workflow_back_to_tools"),
    "_handle_show_workflow_menu": ("workflow", "handle_show_workflow_menu"),
    "_handle_workflow_list_templates": ("workflow", "handle_workflow_list_templates"),
    "_handle_workflow_show_help": ("workflow", "handle_workflow_show_help"),
    "_handle_workflow_view_workflow_ref": ("workflow", "handle_workflow_view_workflow_ref"),
    "_handle_workflow_remove_workflow_ref": ("workflow", "handle_workflow_remove_workflow_ref"),
    "_handle_workflow_add_workflow_ref": ("workflow", "handle_workflow_add_workflow_ref"),
    "_handle_workflow_orchestrator_select_tool": ("workflow", "handle_workflow_orchestrator_select_tool"),
    "_handle_workflow_orchestrator_select_model_group": ("workflow", "handle_workflow_orchestrator_select_model_group"),
    "_handle_workflow_orchestrator_select_model_profile": ("workflow", "handle_workflow_orchestrator_select_model_profile"),
    "_handle_workflow_orchestrator_select_model_effort": ("workflow", "handle_workflow_orchestrator_select_model_effort"),
    "_handle_workflow_orchestrator_select_model": ("workflow", "handle_workflow_orchestrator_select_model"),
    "_handle_workflow_orchestrator_remove": ("workflow", "handle_workflow_orchestrator_remove"),
    "_handle_workflow_orchestrator_clear": ("workflow", "handle_workflow_orchestrator_clear"),
    "_handle_workflow_orchestrator_finish": ("workflow", "handle_workflow_orchestrator_finish"),
    "_handle_workflow_review_select_tool": ("workflow", "handle_workflow_review_select_tool"),
    "_handle_workflow_review_select_model_group": ("workflow", "handle_workflow_review_select_model_group"),
    "_handle_workflow_review_select_model_profile": ("workflow", "handle_workflow_review_select_model_profile"),
    "_handle_workflow_review_select_model_effort": ("workflow", "handle_workflow_review_select_model_effort"),
    "_handle_workflow_review_select_model": ("workflow", "handle_workflow_review_select_model"),
    "_handle_workflow_review_finish": ("workflow", "handle_workflow_review_finish"),
    "_handle_workflow_review_remove": ("workflow", "handle_workflow_review_remove"),
    "_handle_workflow_review_clear": ("workflow", "handle_workflow_review_clear"),
    "_handle_workflow_review_toggle_auto": ("workflow", "handle_workflow_review_toggle_auto"),
}


def _handler_classes() -> dict[str, type]:
    """Return handler classes used to validate FORWARDING_MAP method names."""
    from .handlers.deep import DeepHandler
    from .handlers.diagnostics import DiagnosticsHandler
    from .handlers.programming import (
        AidenModeHandler,
        ClaudeModeHandler,
        CocoModeHandler,
        CodexModeHandler,
        GeminiModeHandler,
        TraexModeHandler,
        Tui2acpModeHandler,
    )
    from .handlers.project import ProjectHandler
    from .handlers.slock import SlockHandler
    from .handlers.spec import SpecHandler
    from .handlers.system import SystemHandler
    from .handlers.workflow import WorkflowHandler

    return {
        "coco": CocoModeHandler,
        "claude": ClaudeModeHandler,
        "aiden": AidenModeHandler,
        "codex": CodexModeHandler,
        "gemini": GeminiModeHandler,
        "traex": TraexModeHandler,
        "tui2acp": Tui2acpModeHandler,
        "system": SystemHandler,
        "deep": DeepHandler,
        "spec": SpecHandler,
        "slock": SlockHandler,
        "workflow": WorkflowHandler,
        "project": ProjectHandler,
        "diagnostics": DiagnosticsHandler,
    }


def validate_forwarding_map() -> list[str]:
    """Validate that each forwarding target refers to an existing handler method."""
    handler_classes = _handler_classes()
    errors: list[str] = []
    for attr_name, (handler_key, method_name) in FORWARDING_MAP.items():
        handler_cls = handler_classes.get(handler_key)
        if handler_cls is None:
            errors.append(f"{attr_name}: unknown handler key {handler_key!r}")
            continue
        if not hasattr(handler_cls, method_name):
            errors.append(f"{attr_name}: {handler_key}.{method_name} is missing")
    return errors

def bind_forwarding_methods(client: Any, handler_ctx: Any) -> None:
    """
    Bind forwarding methods directly on the client instance.
    """
    for attr_name, (handler_key, method_name) in FORWARDING_MAP.items():
        handler = handler_ctx.handlers.get(handler_key)
        if handler:
            setattr(client, attr_name, getattr(handler, method_name))
