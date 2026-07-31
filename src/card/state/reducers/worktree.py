"""Worktree sub-reducer: handles worktree engine interactive card events.

Stores structured data in ContentBlock.data (dict).
Formatting (emoji, markdown) is deferred to the render layer.
Buttons use ButtonIntent values — mapped to action_ids by the render layer.
"""
from __future__ import annotations

from dataclasses import replace

from ...events import CardEvent, CardEventType
from ...ui_text import UI_TEXT
from ..button_intent import ButtonIntent
from ..models import (
    ButtonSpec,
    CardState,
    FooterState,
    HeaderState,
    WorktreeCleanupBlock,
    WorktreeConfirmBlock,
    WorktreeMergeBlock,
    WorktreeSelectBlock,
    WorktreeUnitsBlock,
)
from ._shared import build_header
from .lifecycle import _compute_duration


def _is_model_select_action(action: str) -> bool:
    normalized = str(action or "").strip()
    return normalized == "worktree_select_model" or normalized.endswith("_select_model")


def _build_worktree_header(state: CardState, subtitle: str) -> HeaderState:
    """Reuse shared programming-style title while updating worktree step subtitle."""
    current = build_header(state.metadata, state.terminal)
    return HeaderState(
        title=current.title,
        subtitle=subtitle,
        template="wathet",
        header_source="lifecycle",
    )


def reduce_worktree(state: CardState, event: CardEvent) -> CardState:
    """Handle WORKTREE_* events — generate interactive card state."""
    match event.type:
        case CardEventType.WORKTREE_TOOL_SELECT:
            tools = event.payload.get("tools", [])
            selected = event.payload.get("selected", [])
            message = event.payload.get("message", "")
            project_id = event.payload.get("project_id", "")
            thread_root_id = event.payload.get("thread_root_id", "")
            select_action = event.payload.get("select_action", "worktree_select_tool")
            pending_tool = event.payload.get("pending_tool", "")

            data = {
                "tools": tools,
                "selected": selected,
                "message": message,
                "project_id": project_id,
                "thread_root_id": thread_root_id,
                "select_action": select_action,
                "pending_tool": pending_tool,
                "show_stepper": event.payload.get("show_stepper", True),
            }
            for key in (
                "mode_label",
                "tool_select_title",
                "model_select_title",
                "auto_action",
                "auto_text",
                "auto_description",
                "finish_action",
                "remove_action",
                "clear_action",
                "back_action",
                "page_action",
                "page_tool_name",
                "page_provider",
            ):
                value = event.payload.get(key)
                if value:
                    data[key] = value
            model_page = event.payload.get("model_page")
            if model_page:
                data["model_page"] = int(model_page)
            block = WorktreeSelectBlock(
                block_id="worktree_tool_list",
                data=data,
            )

            # 不在 reducer 层下发 footer 按钮：确认 / 移除 / 清空 等按钮统一由
            # render 层的 _render_worktree_tool_select 内嵌输出，避免出现两个 "确认选择"。
            buttons: tuple[ButtonSpec, ...] = ()

            is_model_select = _is_model_select_action(select_action)
            subtitle = (
                str(data.get("model_select_title") or UI_TEXT["worktree_step_model_select"])
                if is_model_select
                else str(data.get("tool_select_title") or UI_TEXT["worktree_step_tool_select"])
            )
            header = _build_worktree_header(state, subtitle)
            if is_model_select:
                footer_text = UI_TEXT["worktree_step_model_select_hint"]
            elif not selected:
                footer_text = UI_TEXT["wt_hint_select_at_least_one"]
            else:
                footer_text = UI_TEXT["worktree_footer_confirm_hint"]
            footer = FooterState(status="idle", status_text=footer_text)
            return replace(state, blocks=(block,), buttons=buttons, footer=footer, header=header)

        case CardEventType.WORKTREE_CONFIRM:
            selected_items = event.payload.get("selected_items", [])
            goal = event.payload.get("goal", "")
            message = event.payload.get("message", "")
            thread_root_id = event.payload.get("thread_root_id", "")
            awaiting_goal = not str(goal or "").strip()

            data = {
                "selected_items": selected_items,
                "goal": goal,
                "message": message,
                "thread_root_id": thread_root_id,
                "awaiting_goal": awaiting_goal,
            }
            block = WorktreeConfirmBlock(
                block_id="worktree_confirm",
                data=data,
            )

            buttons = (
                ButtonSpec(text=UI_TEXT["wt_btn_reselect"], action_id=ButtonIntent.WORKTREE_SHOW_MENU),
                ButtonSpec(text=UI_TEXT["wt_btn_cancel"], action_id=ButtonIntent.WORKTREE_CANCEL, type="danger"),
            )
            header_title = (
                UI_TEXT["worktree_step_awaiting_goal"]
                if awaiting_goal
                else UI_TEXT["worktree_step_confirm"]
            )
            footer_text = (
                UI_TEXT["worktree_footer_awaiting_goal"]
                if awaiting_goal
                else UI_TEXT["worktree_footer_awaiting_confirm"]
            )
            header = _build_worktree_header(state, header_title)
            footer = FooterState(status="idle", status_text=footer_text)
            return replace(state, blocks=(block,), buttons=buttons, footer=footer, header=header)

        case CardEventType.WORKTREE_PROGRESS:
            units = event.payload.get("units", [])
            message = event.payload.get("message", "")
            thread_root_id = event.payload.get("thread_root_id", "")
            iteration = event.payload.get("iteration")
            metadata = state.metadata
            if isinstance(iteration, int) and iteration > 0:
                metadata = replace(metadata, iteration_index=iteration)
                state = replace(state, metadata=metadata)

            # Compute progress stats (pure data)
            completed = sum(1 for u in units if u.get("status") == "completed")
            failed_count = sum(1 for u in units if u.get("status") == "failed")
            total = len(units)

            data = {"units": units, "message": message, "completed": completed, "total": total, "thread_root_id": thread_root_id}
            block = WorktreeUnitsBlock(
                block_id="worktree_progress",
                data=data,
            )

            # Progress in footer
            if total > 0:
                pct = int(completed / total * 100)
                progress_text = f"{completed}/{total} 完成"
            else:
                pct = None
                progress_text = None

            # Buttons: stop always available, retry when failures exist
            # Special case: all completed with no failures → show retry-all option
            if total > 0 and completed == total and failed_count == 0:
                buttons: tuple[ButtonSpec, ...] = (
                    ButtonSpec(text=UI_TEXT["wt_btn_retry_all"], action_id=ButtonIntent.WORKTREE_RETRY_ALL, type="primary", confirm=UI_TEXT["wt_btn_confirm_retry_all"]),
                    ButtonSpec(text=UI_TEXT["wt_btn_cancel"], action_id=ButtonIntent.WORKTREE_CANCEL, type="danger"),
                )
            else:
                buttons = (
                    ButtonSpec(text=UI_TEXT["wt_btn_stop"], action_id=ButtonIntent.WORKTREE_CANCEL),
                )
                if failed_count > 0:
                    buttons = (
                        ButtonSpec(text=UI_TEXT["wt_btn_retry_failed"], action_id=ButtonIntent.WORKTREE_RETRY_FAILED, type="primary", confirm=UI_TEXT["wt_btn_confirm_retry"]),
                    ) + buttons

            header = _build_worktree_header(state, UI_TEXT["worktree_step_units"])
            is_silent = event.payload.get("silent", False)
            if total > 0 and completed == total:
                footer = FooterState(status="tool_running", progress=progress_text,
                                     progress_pct=pct, status_text=UI_TEXT["worktree_footer_finishing"])
            elif is_silent:
                footer = FooterState(status="tool_running", progress=progress_text, progress_pct=pct,
                                     status_text=UI_TEXT["worktree_footer_silent"])
            else:
                footer = FooterState(status="tool_running", progress=progress_text, progress_pct=pct)
            terminal = (
                "running"
                if state.terminal == "completed_empty"
                else state.terminal
            )
            return replace(
                state,
                blocks=(block,),
                buttons=buttons,
                footer=footer,
                header=header,
                metadata=metadata,
                terminal=terminal,
            )

        case CardEventType.WORKTREE_MERGE:
            merge_notes = event.payload.get("merge_notes", [])
            base_branch = event.payload.get("base_branch", "main")
            thread_root_id = event.payload.get("thread_root_id", "")

            data = {"merge_notes": merge_notes, "base_branch": base_branch, "thread_root_id": thread_root_id}
            block = WorktreeMergeBlock(
                block_id="worktree_merge",
                data=data,
            )

            buttons = (
                ButtonSpec(
                    text=UI_TEXT["wt_btn_merge_all"],
                    action_id=ButtonIntent.WORKTREE_MERGE,
                    type="primary",
                    confirm=UI_TEXT["wt_btn_confirm_merge"].format(base_branch=base_branch),
                ),
            )
            header = _build_worktree_header(state, UI_TEXT["worktree_step_merge"])
            footer = FooterState(status="idle", status_text=UI_TEXT["worktree_footer_pending_merge"])
            return replace(state, blocks=(block,), buttons=buttons, footer=footer, header=header)

        case CardEventType.WORKTREE_CLEANUP:
            merge_notes = event.payload.get("merge_notes", [])
            base_branch = event.payload.get("base_branch", "main")
            merge_results = event.payload.get("merge_results")
            cleanup_phase = event.payload.get("cleanup_phase", "summary")
            thread_root_id = event.payload.get("thread_root_id", "")

            data = {
                "merge_notes": merge_notes, "base_branch": base_branch,
                "merge_results": merge_results, "cleanup_phase": cleanup_phase,
                "thread_root_id": thread_root_id,
            }
            block = WorktreeCleanupBlock(
                block_id="worktree_cleanup",
                data=data,
            )

            has_failed = any(
                r.get("success") is False for r in (merge_results or [])
            )

            if cleanup_phase == "completed":
                buttons_list = []
            elif cleanup_phase == "summary":
                buttons_list = [
                    ButtonSpec(
                        text=UI_TEXT["wt_btn_merge"],
                        action_id=ButtonIntent.WORKTREE_MERGE,
                        type="primary",
                        confirm=UI_TEXT["wt_btn_confirm_merge"].format(base_branch=base_branch),
                    ),
                ]
                if has_failed:
                    buttons_list.append(
                        ButtonSpec(text=UI_TEXT["wt_btn_retry_failed"], action_id=ButtonIntent.WORKTREE_RETRY_FAILED, confirm=UI_TEXT["wt_btn_confirm_retry"]),
                    )
                buttons_list.append(
                    ButtonSpec(text=UI_TEXT["wt_btn_abandon"], action_id=ButtonIntent.WORKTREE_CANCEL, type="default", confirm=UI_TEXT["wt_btn_confirm_abandon"]),
                )
            else:
                buttons_list = [
                    ButtonSpec(
                        text=UI_TEXT["wt_btn_merge"],
                        action_id=ButtonIntent.WORKTREE_MERGE,
                        type="primary",
                        confirm=UI_TEXT["wt_btn_confirm_merge"].format(base_branch=base_branch),
                    ),
                ]
                if has_failed:
                    buttons_list.append(
                        ButtonSpec(text=UI_TEXT["wt_btn_retry_failed"], action_id=ButtonIntent.WORKTREE_RETRY_FAILED, confirm=UI_TEXT["wt_btn_confirm_retry"]),
                    )
                buttons_list.append(
                    ButtonSpec(
                        text=UI_TEXT["wt_btn_cleanup"],
                        action_id=ButtonIntent.WORKTREE_CLEANUP,
                        type="danger",
                        confirm=UI_TEXT["wt_btn_confirm_cleanup"],
                    ),
                )
                buttons_list.append(
                    ButtonSpec(text=UI_TEXT["wt_btn_abandon"], action_id=ButtonIntent.WORKTREE_CANCEL, type="default", confirm=UI_TEXT["wt_btn_confirm_abandon"]),
                )
            buttons = tuple(buttons_list)

            header = _build_worktree_header(state, UI_TEXT["worktree_step_cleanup"])
            footer = FooterState(status="idle", status_text=UI_TEXT["worktree_footer_merge_cleanup"])
            return replace(state, blocks=(block,), buttons=buttons, footer=footer, header=header)

        case CardEventType.WORKTREE_COMPLETED_NO_CHANGE:
            units = event.payload.get("units", [])
            message = event.payload.get("message", "") or UI_TEXT["worktree_completed_no_change"]
            thread_root_id = event.payload.get("thread_root_id", "")
            iteration = event.payload.get("iteration")
            metadata = state.metadata
            if isinstance(iteration, int) and iteration > 0:
                metadata = replace(metadata, iteration_index=iteration)
                state = replace(state, metadata=metadata)

            data = {"units": units, "message": message, "thread_root_id": thread_root_id}
            block = WorktreeUnitsBlock(
                block_id="worktree_no_change",
                data=data,
            )

            buttons = (
                ButtonSpec(text=UI_TEXT["wt_btn_retry_all"], action_id=ButtonIntent.WORKTREE_RETRY_ALL, type="primary", confirm=UI_TEXT["wt_btn_confirm_retry_all"]),
                ButtonSpec(text=UI_TEXT["wt_btn_cancel"], action_id=ButtonIntent.WORKTREE_CANCEL, type="danger"),
            )
            header = _build_worktree_header(state, UI_TEXT["worktree_no_change_subtitle"])
            footer = FooterState(
                status="idle",
                status_text=message,
                duration_seconds=_compute_duration(
                    state,
                    event.payload.get("_now"),
                ),
            )
            return replace(state, blocks=(block,), buttons=buttons, footer=footer, header=header, terminal="completed_empty", metadata=metadata)

    return state
