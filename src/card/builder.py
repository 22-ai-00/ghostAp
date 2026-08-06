from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.mode.manager import InteractionMode

from .actions import dispatch as action_ids
from .builders.core import CoreBuilder
from .builders.deep import DeepBuilder
from .builders.project import ProjectBuilder
from .builders.spec_review import SpecReviewBuilder
from .builders.system import SystemBuilder
from .models import EngineCardState

if TYPE_CHECKING:
    from ..project.context import ProjectContext
    from ..sandbox.executor import ExecutionResult


class CardBuilder:
    """
    Facade class that delegates card building to specialized builders.
    Maintains backward compatibility with the original CardBuilder API.
    """

    # --- Core Delegates ---

    @staticmethod
    def _build_button_grid(buttons: list[dict], columns: int = 2) -> list[dict]:
        from .shared import _build_button_grid

        return _build_button_grid(buttons, columns)

    @staticmethod
    def _build_button_row_action(buttons: list[dict]) -> list[dict]:
        from .shared import _build_button_row_action

        return _build_button_row_action(buttons)

    @staticmethod
    def _truncate_markdown(content: str, max_chars: int) -> str:
        return CoreBuilder._truncate_markdown(content, max_chars)

    @staticmethod
    def _build_content_element(content: str, with_title: Optional[str] = None, max_chars: int = 4000) -> dict:
        return CoreBuilder._build_content_element(content, with_title, max_chars)

    @staticmethod
    def _build_header_title(
        project: Optional[ProjectContext],
        mode: Optional[InteractionMode] = None,
    ) -> str:
        return CoreBuilder._build_header_title(
            project, mode=mode
        )

    @staticmethod
    def _build_directory_element(project: Optional[ProjectContext], working_dir: Optional[str] = None) -> dict:
        return CoreBuilder._build_directory_element(project, working_dir)

    @staticmethod
    def _build_footer_buttons(
        project: Optional[ProjectContext],
        mode: Optional[InteractionMode] = None,
    ) -> list[dict]:
        return CoreBuilder._build_footer_buttons(
            project, mode=mode
        )

    @staticmethod
    def _build_footer_note(project: Optional[ProjectContext], working_dir: Optional[str] = None) -> Optional[dict]:
        return CoreBuilder._build_footer_note(project, working_dir)

    @staticmethod
    def _wrap_card(header_title: str, header_template: str, elements: list[dict]) -> dict:
        return CoreBuilder._wrap_card(header_title, header_template, elements)

    @staticmethod
    def _build_image_elements(image_keys: list[str]) -> list[dict]:
        return CoreBuilder._build_image_elements(image_keys)

    @staticmethod
    def _format_time_ago(timestamp: float) -> str:
        return CoreBuilder._format_time_ago(timestamp)

    @staticmethod
    def _build_banner_element(message: str, type: str = "success") -> dict:
        return CoreBuilder._build_banner_element(message, type)

    # --- Project Delegates ---

    @staticmethod
    def _build_response_card_inner(
        project: Optional[ProjectContext],
        title: str,
        content: str,
        working_dir: Optional[str] = None,
        show_buttons: bool = True,
        mode: Optional[InteractionMode] = None,
        extra_buttons: Optional[list[dict]] = None,
        footer: Optional[str] = None,
        image_keys: Optional[list[str]] = None,
    ) -> tuple[str, str]:
        return ProjectBuilder._build_response_card_inner(
            project,
            title,
            content,
            working_dir,
            show_buttons,
            mode=mode,
            extra_buttons=extra_buttons,
            footer=footer,
            image_keys=image_keys,
        )

    @staticmethod
    def build_coco_response_card(
        project: Optional[ProjectContext],
        title: str,
        content: str,
        working_dir: Optional[str] = None,
        show_buttons: bool = True,
    ) -> tuple[str, str]:
        return ProjectBuilder.build_coco_response_card(project, title, content, working_dir, show_buttons)

    @staticmethod
    def build_smart_response_card(
        project: Optional[ProjectContext],
        title: str,
        content: str,
        working_dir: Optional[str] = None,
        show_buttons: bool = True,
    ) -> tuple[str, str]:
        return ProjectBuilder.build_smart_response_card(project, title, content, working_dir, show_buttons)

    @staticmethod
    def build_project_response_card(
        project: ProjectContext,
        title: str,
        content: str,
        show_buttons: bool = True,
        extra_buttons: Optional[list[dict]] = None,
        footer: Optional[str] = None,
        image_keys: Optional[list[str]] = None,
        banner: Optional[dict] = None,
    ) -> tuple[str, str]:
        return ProjectBuilder.build_project_response_card(
            project, title, content, show_buttons, extra_buttons, footer, image_keys, banner=banner
        )

    @staticmethod
    def build_status_board_card(
        projects: list[ProjectContext],
        current_project_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 5,
    ) -> tuple[str, str]:
        return ProjectBuilder.build_status_board_card(projects, current_project_id, page, page_size)

    @staticmethod
    def build_notification_card(
        project: ProjectContext,
        notification_type: str,
        title: str,
        content: str,
        suggestions: Optional[list[str]] = None,
        buttons: Optional[list[dict]] = None,
    ) -> tuple[str, str]:
        return ProjectBuilder.build_notification_card(project, notification_type, title, content, suggestions, buttons)

    @staticmethod
    def _build_resume_card(project: ProjectContext, mode: str) -> tuple[str, str]:
        return ProjectBuilder._build_resume_card(project, mode)

    @staticmethod
    def build_coco_resume_card(project: ProjectContext) -> tuple[str, str]:
        return ProjectBuilder.build_coco_resume_card(project)

    @staticmethod
    def build_claude_resume_card(project: ProjectContext) -> tuple[str, str]:
        return ProjectBuilder.build_claude_resume_card(project)

    @staticmethod
    def build_current_project_card(project: ProjectContext, global_working_dir: str) -> tuple[str, str]:
        return ProjectBuilder.build_current_project_card(project, global_working_dir)

    @staticmethod
    def build_project_status_report_card(project: ProjectContext, global_working_dir: str) -> tuple[str, str]:
        return ProjectBuilder.build_project_status_report_card(project, global_working_dir)

    @staticmethod
    def build_project_switch_card(project: ProjectContext, context_info: str = "") -> tuple[str, str]:
        return ProjectBuilder.build_project_switch_card(project, context_info)

    @staticmethod
    def build_project_created_card(project: ProjectContext) -> tuple[str, str]:
        return ProjectBuilder.build_project_created_card(project)

    @staticmethod
    def build_project_not_found_content(name: str, suggestions: Optional[list[ProjectContext]] = None) -> str:
        return ProjectBuilder.build_project_not_found_content(name, suggestions=suggestions)

    @staticmethod
    def build_restore_info_content(restore_info: dict) -> str:
        return ProjectBuilder.build_restore_info_content(restore_info)

    @staticmethod
    def build_project_switch_notification_card(project: ProjectContext, restore_info: dict) -> tuple[str, str]:
        return ProjectBuilder.build_project_switch_notification_card(project, restore_info)

    # --- System Delegates ---

    @staticmethod
    def build_tools_list_card(
        tools: list[dict],
        project: Optional[ProjectContext] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_tools_list_card(tools, project)

    @staticmethod
    def build_tools_status_card(
        tools: list[dict],
        active_sessions: dict[str, dict] = None,
        project: Optional[ProjectContext] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_tools_status_card(tools, active_sessions, project)

    @staticmethod
    def build_directory_change_card(
        project: Optional[ProjectContext],
        path: str,
        success: bool = True,
    ) -> Optional[tuple[str, str]]:
        return SystemBuilder.build_directory_change_card(project, path, success)

    @staticmethod
    def build_switching_status_card(tool: str, model: str) -> tuple[str, str]:
        return SystemBuilder.build_switching_status_card(tool, model)

    @staticmethod
    def build_coco_status_content(
        current_model: Optional[str],
        models: list,
    ) -> str:
        return SystemBuilder.build_coco_status_content(current_model, models)

    # --- Diagnostics Delegates ---

    @staticmethod
    def build_task_board_content(
        tasks: list,
        mode_display: str = "",
        groups: Optional[dict] = None,
        project_manager: any = None,
    ) -> str:
        from .builders.diagnostics import DiagnosticsBuilder
        return DiagnosticsBuilder.build_task_board_content(tasks, mode_display, groups, project_manager)

    @staticmethod
    def build_unified_status_content(
        entries: list,
        include_done: bool = False,
        project_name: str = ""
    ) -> str:
        from .builders.diagnostics import DiagnosticsBuilder
        return DiagnosticsBuilder.build_unified_status_content(entries, include_done, project_name)

    @staticmethod
    def format_engine_status_info(mode: str, p: any) -> str:
        from .builders.diagnostics import DiagnosticsBuilder
        return DiagnosticsBuilder.format_engine_status_info(mode, p)

    @staticmethod
    def build_message_trace_content(data: dict) -> str:
        from .builders.diagnostics import DiagnosticsBuilder
        return DiagnosticsBuilder.build_message_trace_content(data)

    @staticmethod
    def build_task_detail_content(state: any) -> str:
        from .builders.diagnostics import DiagnosticsBuilder
        return DiagnosticsBuilder.build_task_detail_content(state)

    @staticmethod
    def build_diff_report_content(
        project: any,
        from_v: any,
        to_v: any,
        entries: list,
        show_current: bool = False,
    ) -> str:
        from .builders.diagnostics import DiagnosticsBuilder
        return DiagnosticsBuilder.build_diff_report_content(project, from_v, to_v, entries, show_current)

    @staticmethod
    def build_error_card(
        exc: Exception | str,
        title: str = "操作失败",
        project: Optional[ProjectContext] = None,
        *,
        summary: Optional[str] = None,
        details: Optional[str] = None,
        detail_action: Optional[dict] = None,
        continue_action: Optional[dict] = None,
        retry_action: Optional[dict] = None,
        severity: str = "fatal",
    ) -> tuple[str, str]:
        return SystemBuilder.build_error_card(
            exc,
            title,
            project,
            summary=summary,
            details=details,
            detail_action=detail_action,
            continue_action=continue_action,
            retry_action=retry_action,
            severity=severity,
        )

    @staticmethod
    def build_shell_result_card(
        cmd: str,
        result: "ExecutionResult",
        working_dir: Optional[str] = None,
        project: Optional[ProjectContext] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_shell_result_card(cmd, result, working_dir, project)

    @staticmethod
    def build_acp_tool_select_card(tools: list, project_id: Optional[str] = None, current_tool: Optional[str] = None) -> tuple[str, str]:
        return SystemBuilder.build_acp_tool_select_card(tools, project_id=project_id, current_tool=current_tool)

    @staticmethod
    def build_tui2acp_adapter_select_card(adapters: list[dict], project_id: Optional[str] = None, current_adapter: Optional[str] = None) -> tuple[str, str]:
        return SystemBuilder.build_tui2acp_adapter_select_card(adapters, project_id=project_id, current_adapter=current_adapter)

    @staticmethod
    def build_slock_role_tool_select_card(
        role_name: str,
        tools: list[dict],
        project_id: Optional[str] = None,
        *,
        value_extra: Optional[dict] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_slock_role_tool_select_card(
            role_name,
            tools,
            project_id=project_id,
            value_extra=value_extra,
        )

    @staticmethod
    def build_slock_role_unified_select_card(
        role_name: str,
        tools: list[dict],
        models: list,
        tool_name: str,
        project_id: Optional[str] = None,
        current_model: Optional[str] = None,
        *,
        pending_group: Optional[str] = None,
        pending_profile: Optional[str] = None,
        pending_effort: Optional[str] = None,
        profile_markdown: Optional[str] = None,
        value_extra: Optional[dict] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_slock_role_unified_select_card(
            role_name,
            tools,
            models,
            tool_name,
            project_id=project_id,
            current_model=current_model,
            pending_group=pending_group,
            pending_profile=pending_profile,
            pending_effort=pending_effort,
            profile_markdown=profile_markdown,
            value_extra=value_extra,
        )

    @staticmethod
    def build_acp_model_select_card(
        models: list,
        tool_name: str,
        project_id: Optional[str] = None,
        current_model: Optional[str] = None,
        thread_root_id: Optional[str] = None,
        *,
        action_name: Optional[str] = None,
        value_extra: Optional[dict] = None,
        context_markdown: Optional[str] = None,
        refresh_action_name: Optional[str] = None,
        model_page: int = 0,
    ) -> tuple[str, str]:
        kwargs = {}
        if action_name is not None:
            kwargs["action_name"] = action_name
        if value_extra is not None:
            kwargs["value_extra"] = value_extra
        if context_markdown is not None:
            kwargs["context_markdown"] = context_markdown
        if refresh_action_name is not None:
            kwargs["refresh_action_name"] = refresh_action_name
        return SystemBuilder.build_acp_model_select_card(
            models,
            tool_name,
            project_id=project_id,
            current_model=current_model,
            thread_root_id=thread_root_id,
            model_page=model_page,
            **kwargs,
        )

    @staticmethod
    def build_acp_model_cascade_card(
        models: list,
        tool_name: str,
        project_id: Optional[str] = None,
        current_model: Optional[str] = None,
        thread_root_id: Optional[str] = None,
        *,
        pending_group: Optional[str] = None,
        pending_profile: Optional[str] = None,
        pending_effort: Optional[str] = None,
        context_markdown: Optional[str] = None,
        group_action: str = action_ids.SELECT_ACP_MODEL_GROUP,
        profile_action: str = action_ids.SELECT_ACP_MODEL_PROFILE,
        effort_action: str = action_ids.SELECT_ACP_MODEL_EFFORT,
        select_action: str = action_ids.SELECT_ACP_MODEL,
        refresh_action: str = action_ids.REFRESH_ACP_MODELS,
        value_extra: Optional[dict] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_acp_model_cascade_card(
            models,
            tool_name,
            project_id=project_id,
            current_model=current_model,
            thread_root_id=thread_root_id,
            pending_group=pending_group,
            pending_profile=pending_profile,
            pending_effort=pending_effort,
            context_markdown=context_markdown,
            group_action=group_action,
            profile_action=profile_action,
            effort_action=effort_action,
            select_action=select_action,
            refresh_action=refresh_action,
            value_extra=value_extra,
        )

    @staticmethod
    def build_acp_model_loading_card(
        tool_name: str,
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_acp_model_loading_card(
            tool_name,
            project_id=project_id,
            thread_root_id=thread_root_id,
        )

    @staticmethod
    def build_acp_model_error_card(
        tool_name: str,
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_acp_model_error_card(
            tool_name,
            project_id=project_id,
            thread_root_id=thread_root_id,
        )

    @staticmethod
    def build_acp_programming_ready_card(
        tool_name: str,
        model_name: Optional[str],
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_acp_programming_ready_card(
            tool_name,
            model_name,
            project_id=project_id,
            thread_root_id=thread_root_id,
        )

    @staticmethod
    def build_acp_programming_initializing_card(
        tool_name: str,
        model_name: Optional[str],
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_acp_programming_initializing_card(
            tool_name,
            model_name,
            project_id=project_id,
            thread_root_id=thread_root_id,
        )

    @staticmethod
    def build_acp_programming_failed_card(
        tool_name: str,
        model_name: Optional[str],
        reason: str,
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_acp_programming_failed_card(
            tool_name,
            model_name,
            reason,
            project_id=project_id,
            thread_root_id=thread_root_id,
        )

    @staticmethod
    def build_spec_review_agent_select_card(
        *,
        tools: list[dict],
        selected: list[dict] | None = None,
        project_id: Optional[str] = None,
        message: str = "",
        select_action: str = "spec_review_select_tool",
        pending_tool: str = "",
        thread_root_id: str = "",
        model_page: int = 0,
        page_tool_name: str = "",
        page_provider: str = "",
    ) -> tuple[str, str]:
        return SpecReviewBuilder.build_agent_select_card(
            tools=tools,
            selected=selected,
            project_id=project_id,
            message=message,
            select_action=select_action,
            pending_tool=pending_tool,
            thread_root_id=thread_root_id,
            model_page=model_page,
            page_tool_name=page_tool_name,
            page_provider=page_provider,
        )

    @staticmethod
    def build_spec_review_starting_card(
        *,
        selected: list[dict] | None = None,
        auto: bool = False,
    ) -> tuple[str, str]:
        return SpecReviewBuilder.build_starting_card(selected=selected, auto=auto)

    @staticmethod
    def build_command_menu_card(project: Optional[ProjectContext] = None) -> tuple[str, str]:
        return SystemBuilder.build_command_menu_card(project)

    @staticmethod
    def build_help_card(
        project: Optional[ProjectContext] = None,
        category: str = "main",
        working_dir: Optional[str] = None,
        current_mode: any = None,
        is_admin: bool = False,
        lock_enabled: bool = False,
        chat_id: str = "",
        no_admin_configured: bool = False,
        *,
        session_idle_timeout: Optional[int] = None,
        session_idle_warn_at_remaining: Optional[int] = None,
        lock_undo_window_seconds: Optional[int] = None,
    ) -> tuple[str, str]:
        return SystemBuilder.build_help_card(
            project, category, working_dir, current_mode,
            is_admin=is_admin, lock_enabled=lock_enabled, chat_id=chat_id,
            no_admin_configured=no_admin_configured,
            session_idle_timeout=session_idle_timeout,
            session_idle_warn_at_remaining=session_idle_warn_at_remaining,
            lock_undo_window_seconds=lock_undo_window_seconds,
        )

    @staticmethod
    def _build_help_card_cached(
        project_name: Optional[str],
        root_path: Optional[str],
        project_id: Optional[str],
        category: str,
        working_dir: Optional[str],
        current_mode_str: str,
        is_admin: bool = False,
    ) -> tuple[str, str]:
        return SystemBuilder._build_help_card_cached(
            project_name, root_path, project_id, category, working_dir, current_mode_str, is_admin=is_admin
        )

    # --- Deep Delegates ---

    @staticmethod
    def _build_deep_header_title(project: Optional[ProjectContext], engine_name: str = "Coco") -> str:
        return DeepBuilder._build_deep_header_title(project, engine_name)

    @staticmethod
    def _pick_deep_template(engine_name: str, status: str = "running") -> str:
        return DeepBuilder._pick_deep_template(engine_name, status)

    @staticmethod
    def _build_deep_buttons(state: EngineCardState) -> list[dict]:
        return DeepBuilder._build_deep_buttons(state)

    @staticmethod
    def build_info_card(
        project: Optional[ProjectContext],
        state: Optional[EngineCardState] = None,
        *,
        title: str = "",
        content: str = "",
        engine_name: str = "Coco",
        show_buttons: bool = True,
        working_dir: Optional[str] = None,
        progress_bar: Optional[str] = None,
        project_id: Optional[str] = None,
        engine_project_id: Optional[str] = None,
        is_executing: bool = False,
        is_paused: bool = False,
        status_line: Optional[str] = None,
        duration_line: Optional[str] = None,
        criteria_section: Optional[str] = None,
        footer_note: Optional[str] = None,
        compact: bool = False,
        expanded: bool = False,
        expand_ac: bool = False,
        action_prefix: str = "deep",
        extra_buttons: Optional[list[dict]] = None,
        warning_banner: Optional[str] = None,
    ) -> tuple[str, str]:
        if state is None:
            state = EngineCardState(
                title=title,
                content=content,
                progress_bar=progress_bar,
                project_id=project_id,
                engine_project_id=engine_project_id,
                is_executing=is_executing,
                is_paused=is_paused,
                engine_name=engine_name,
                show_buttons=show_buttons,
                working_dir=working_dir,
                status_line=status_line,
                duration_line=duration_line,
                criteria_section=criteria_section,
                footer_note=footer_note,
                compact=compact,
                expanded=expanded,
                expand_ac=expand_ac,
                action_prefix=action_prefix,
                extra_buttons=extra_buttons,
                warning_banner=warning_banner,
            )
        return DeepBuilder.build_info_card(project, state)

    @staticmethod
    def build_history_list_card(
        project: Optional[ProjectContext],
        title: str,
        content: str,
        history_buttons: list[dict],
        page: int,
        has_next: bool,
        deep_project_id: Optional[str] = None,
        engine_name: str = "Coco",
    ) -> tuple[str, str]:
        return DeepBuilder.build_history_list_card(
            project, title, content, history_buttons, page, has_next, deep_project_id, engine_name
        )
