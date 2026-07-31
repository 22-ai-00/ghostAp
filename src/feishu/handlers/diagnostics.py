"""Diagnostics handler — task board, context diff report, message trace, unified status."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Optional

from ...card import CardBuilder
from ...card.ui_text import UI_TEXT
from ...config import SecurityFinding, evaluate_security_posture
from ...project import context_helper
from ...tasking import TaskPriority, TaskSpec
from ...utils.errors import get_error_detail
from .base import BaseHandler
from .diagnostics_helper import DiagnosticsHelper

if TYPE_CHECKING:
    from ...project import ProjectContext

logger = logging.getLogger(__name__)


class DiagnosticsHandler(BaseHandler):
    """Task board, context diff reports, and message tracing."""

    # ------------------------------------------------------------------
    # Task board
    # ------------------------------------------------------------------

    def _build_security_posture_lines(self) -> str:
        """Render current configured and runtime security findings."""

        try:
            findings: list[SecurityFinding] = list(
                evaluate_security_posture(self.settings).findings
            )
            provider = getattr(
                self.ctx,
                "ingress_access_policy_provider",
                None,
            )
            shared = getattr(provider, "blocking_findings", ())
            if isinstance(shared, tuple):
                findings.extend(
                    finding
                    for finding in shared
                    if isinstance(finding, SecurityFinding)
                )
        except (AttributeError, TypeError, ValueError):
            logger.warning(
                "Failed to evaluate security posture for /status",
                exc_info=True,
            )
            return ""

        if not findings:
            return ""
        deduplicated = {finding.code: finding for finding in findings}
        lines = ["### 🔐 安全姿态"]
        lines.extend(
            f"- **{finding.severity.value.upper()}** `{finding.code}`："
            f"{finding.message}"
            for finding in deduplicated.values()
        )
        return "\n".join(lines)

    def show_task_board(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        arg = ""
        try:
            parts = (text or "").strip().split(None, 1)
            if len(parts) > 1:
                arg = parts[1].strip().lower()
        except Exception:
            arg = ""

        try:
            mode_display = self.mode_manager.get_mode_display_name(chat_id)
        except Exception:
            mode_display = ""

        if arg in ("all", "-a", "--all"):
            tasks = self.scheduler.list_tasks(chat_id=chat_id, include_done=False, limit=50)
            groups: dict[str, list] = {}
            for st in tasks:
                pid = st.spec.project_id or ""
                groups.setdefault(pid, []).append(st)

            content = CardBuilder.build_task_board_content(
                tasks, mode_display, groups=groups, project_manager=self.project_manager
            )

            msg_type, card_content = CardBuilder.build_smart_response_card(
                project=None,
                title=UI_TEXT["diag_task_board_title"],
                content=content,
                working_dir=self.get_working_dir(chat_id),
                show_buttons=True,
            )
            self.reply_card(message_id, card_content)
            return

        # Default: current project
        if project is None:
            project = self.project_manager.get_active_project(chat_id)

        if not project:
            self.reply_text(
                message_id,
                UI_TEXT["diag_no_active_project_tasks"]
            )
            return

        tasks = self.scheduler.list_tasks(chat_id=chat_id, project_id=project.project_id, include_done=False, limit=30)
        content = CardBuilder.build_task_board_content(tasks, mode_display)

        msg_type, card_content = CardBuilder.build_project_response_card(
            project,
            UI_TEXT["diag_task_board_title"],
            content,
            show_buttons=True,
        )
        self.reply_card(message_id, card_content)

    # ------------------------------------------------------------------
    # Unified status — /status [task_id|all]
    # ------------------------------------------------------------------

    def _build_lock_status_lines(self, chat_id: str, project: Optional["ProjectContext"] = None, is_admin: bool = False) -> str:
        """Build a Markdown snippet summarizing lock state as an independent section.

        Always shows subsystem enablement status.  Active lock details are
        appended when present.
        """
        from datetime import datetime

        from ...card.builders.lock import format_elapsed_ago, format_friendly_duration, format_lock_duration
        from ...card.ui_text import UI_TEXT

        parts: list[str] = []

        chat_lock_mgr = getattr(self.ctx, "chat_lock_manager", None)
        repo_lock_mgr_obj = getattr(self.ctx, "repo_lock_manager", None)

        # --- Chat lock detail (only when active) ---
        if chat_lock_mgr is not None:
            lock_entry = chat_lock_mgr.get_lock_info(chat_id)
            if lock_entry:
                display_name = lock_entry.locked_by_name or (
                    lock_entry.locked_by[:8] + "..." if len(lock_entry.locked_by) > 8 else lock_entry.locked_by
                )
                try:
                    lock_dt = datetime.fromtimestamp(lock_entry.locked_at_wall)
                    if lock_dt.date() == datetime.now().date():
                        lock_time_str = lock_dt.strftime("%H:%M")
                    else:
                        lock_time_str = lock_dt.strftime("%m-%d %H:%M")
                except Exception:
                    lock_time_str = UI_TEXT["lock_status_time_unknown"]
                _chat_lock_line = UI_TEXT["lock_status_chat_locked"].format(name=display_name, time=lock_time_str)
                # Append duration on a new indented sub-line for visual clarity
                _chat_lock_line += f"\n  {format_lock_duration(lock_entry.locked_at)}"
                # Append auto-unlock countdown on its own sub-line (hours/minutes mixed format)
                try:
                    _max_dur = getattr(chat_lock_mgr, "_max_duration", 86400)
                    _remaining = max(0, _max_dur - (time.monotonic() - lock_entry.locked_at))
                    if _remaining > 60:
                        _countdown_str = format_friendly_duration(_remaining)
                        _chat_lock_line += f"\n  ⏰ 预计 {_countdown_str}后自动释放"
                    else:
                        _chat_lock_line += f"\n  {UI_TEXT['lock_status_release_imminent'].strip()}"
                except Exception:
                    logger.debug("failed to get lock expiry state", exc_info=True)
                if is_admin:
                    _chat_lock_line += UI_TEXT["lock_status_admin_unlock_hint"]
                parts.append(_chat_lock_line)

        # --- Repo lock detail (only when active) ---
        root_path = getattr(project, "root_path", None) if project else None
        if root_path and repo_lock_mgr_obj is not None:
            try:
                info = repo_lock_mgr_obj.get_lock_info(root_path)
                if info:
                    from pathlib import Path
                    repo_name = Path(root_path).name or root_path
                    if info.chat_id == chat_id:
                        # Cross-check: if chat is also under /lock, warn non-admins
                        _chat_also_locked = False
                        if chat_lock_mgr is not None:
                            try:
                                _chat_also_locked = chat_lock_mgr.is_locked(chat_id)
                            except Exception:
                                logger.debug("Failed to cross-check chat lock in status", exc_info=True)
                        holder_display = UI_TEXT["lock_status_holder_self_but_chat_locked"] if _chat_also_locked else UI_TEXT["lock_status_holder_self"]
                    else:
                        holder_display = UI_TEXT["lock_status_holder_other"]
                    duration = format_elapsed_ago(time.monotonic() - info.acquired_at)
                    # Calculate remaining auto-release time
                    try:
                        from ...config import get_settings
                        idle_timeout = get_settings().repo_lock_idle_timeout
                    except Exception:
                        idle_timeout = 300
                    remaining_secs = max(0, idle_timeout - info.idle_seconds)
                    remaining_min = int(remaining_secs // 60)
                    if remaining_min > 0:
                        release_hint = UI_TEXT["lock_status_release_countdown"].format(minutes=remaining_min)
                    else:
                        release_hint = UI_TEXT["lock_status_release_imminent"]
                    _repo_lock_line = UI_TEXT["lock_status_repo_locked"].format(
                            repo_name=repo_name, holder=holder_display,
                            duration=duration, release_hint=release_hint,
                        )
                    if is_admin and info.chat_id != chat_id:
                        _repo_lock_line += UI_TEXT["lock_status_admin_force_release_hint"]
                    elif not is_admin:
                        _repo_lock_line += UI_TEXT["lock_status_nonadmin_repo_hint"]
                    parts.append(_repo_lock_line)
            except Exception:
                logger.debug("Failed to build repo lock status line", exc_info=True)

        # Build admin display names suffix
        _admin_suffix = ""
        try:
            from ...config import get_settings as _gs
            _admin_ids = _gs().admin_user_ids
            if _admin_ids:
                from ..user_cache import resolve_display_name
                _names = []
                for _uid in list(_admin_ids)[:3]:
                    _names.append(resolve_display_name(_uid) or _uid[:8])
                _admin_line = "、".join(_names)
                if len(_admin_ids) > 3:
                    _admin_line += f" 等 {len(_admin_ids)} 人"
                _admin_suffix = f"\n\n👤 Bot 管理员: {_admin_line}"
        except Exception:
            logger.debug("Failed to build admin suffix for lock status", exc_info=True)

        # When lock subsystem is enabled but no active locks, show "unlocked" status.
        _lock_enabled = chat_lock_mgr is not None or repo_lock_mgr_obj is not None
        if not parts:
            if _lock_enabled:
                return UI_TEXT["lock_status_section_header"] + UI_TEXT["lock_status_no_active_lock"] + "\n" + UI_TEXT["lock_status_no_lock_explain"] + _admin_suffix
            return ""
        return UI_TEXT["lock_status_section_header"] + "\n\n".join(parts) + _admin_suffix

    def show_unified_status(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        """Show unified status across all engine types (Deep/Spec).

        - /status          → list running/paused engine tasks for current chat
        - /status all      → include completed tasks
        - /status <task_id> → detailed status for a specific task
        """
        arg = ""
        try:
            parts = (text or "").strip().split(None, 1)
            if len(parts) > 1:
                arg = parts[1].strip()
        except Exception:
            arg = ""

        # /status <task_id> — specific task lookup
        if arg and arg.lower() not in ("all", "-a", "--all"):
            self._show_task_detail(message_id, chat_id, arg, project)
            return

        include_done = arg.lower() in ("all", "-a", "--all") if arg else False

        # Collect engines across all three types
        entries = DiagnosticsHelper.get_all_engine_statuses(self.ctx, chat_id, include_done=include_done)

        project_name = project.project_name if project else ""
        content = CardBuilder.build_unified_status_content(entries, include_done, project_name)

        # Determine admin status for lock hints
        _is_admin = False
        _chat_lock_mgr = getattr(self.ctx, "chat_lock_manager", None)
        if _chat_lock_mgr is not None:
            from ...thread import get_current_sender_id
            _sender = get_current_sender_id() or ""
            if _sender:
                _is_admin = _chat_lock_mgr.is_admin(_sender)

        # Append lock status section
        lock_lines = self._build_lock_status_lines(chat_id, project, is_admin=_is_admin)
        if lock_lines:
            content += "\n\n" + lock_lines

        security_lines = self._build_security_posture_lines()
        if security_lines:
            content += "\n\n" + security_lines

        msg_type, card_content = CardBuilder.build_smart_response_card(
            project=project,
            title=UI_TEXT["diag_unified_status_title"],
            content=content,
            working_dir=self.get_working_dir(chat_id),
            show_buttons=True,
        )
        self.reply_card(message_id, card_content)

    def _show_task_detail(
        self, message_id: str, chat_id: str, task_id: str, project: Optional["ProjectContext"] = None
    ):
        """Show detailed status for a specific task by task_id."""
        # Search across all engine managers
        for engine in self.ctx.deep_engine_manager.list_engines(chat_id):
            if (
                engine.project
                and engine.project.task_id
                and (engine.project.task_id == task_id or task_id in engine.project.task_id)
            ):
                content = self.ctx.progress_reporter.format_status(engine.project)
                title = UI_TEXT["diag_task_detail_deep_title"]
                engine_name = engine.engine_name
                msg_type, card_content = CardBuilder.build_info_card(
                    project=project,
                    title=title,
                    content=content,
                    engine_name=engine_name,
                    show_buttons=False,
                )
                self.reply_card(message_id, card_content)
                return

        for engine in self.ctx.spec_engine_manager.list_engines(chat_id):
            if (
                engine.project
                and engine.project.task_id
                and (engine.project.task_id == task_id or task_id in engine.project.task_id)
            ):
                content = self.ctx.spec_reporter.format_status(engine.project)
                title = UI_TEXT["diag_task_detail_spec_title"]
                engine_name = engine.engine_name
                msg_type, card_content = CardBuilder.build_info_card(
                    project=project,
                    title=title,
                    content=content,
                    engine_name=f"Spec({engine_name})",
                    show_buttons=False,
                )
                self.reply_card(message_id, card_content)
                return

        # Also check scheduler by task_id
        state = self.scheduler.get_state_by_task_id(task_id, chat_id=chat_id)
        if state:
            content = CardBuilder.build_task_detail_content(state)
            msg_type, card_content = CardBuilder.build_smart_response_card(
                project=project,
                title=UI_TEXT["diag_task_detail_title"],
                content=content,
                working_dir=self.get_working_dir(chat_id),
                show_buttons=False,
            )
            self.reply_card(message_id, card_content)
            return

        self.reply_text(
            message_id,
            UI_TEXT["diag_task_not_found"].format(
                id=task_id
            ),
        )

    # ------------------------------------------------------------------
    # Context diff
    # ------------------------------------------------------------------
    def show_context_diff(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        self._submit_diff_report(message_id, chat_id, text, project)

    def _build_context_diff_report(
        self, chat_id: str, text: str, project: "ProjectContext"
    ) -> tuple[bool, str, Optional[str]]:
        ctx = self.context_manager.store.get(project.project_id, chat_id=chat_id)
        if not ctx:
            return (
                False,
                "",
                UI_TEXT["diag_diff_no_record"],
            )

        versions = list(ctx.versions)
        arg = ""
        try:
            parts = (text or "").strip().split(None, 1)
            if len(parts) > 1:
                arg = parts[1].strip()
        except Exception:
            arg = ""

        ok, from_vnum, to_vnum, show_current, error_key = context_helper.resolve_diff_range(arg, versions)
        if not ok:
            return False, "", UI_TEXT[error_key]

        from_v, to_v, entries = context_helper.filter_context_entries(ctx, from_vnum, to_vnum, show_current)

        if not from_v:
            return (
                False,
                "",
                UI_TEXT["diag_diff_version_not_found"].format(
                    vnum=from_vnum, total=len(versions)
                ),
            )
        if to_vnum is not None and not to_v:
            return (
                False,
                "",
                UI_TEXT["diag_diff_version_not_found"].format(
                    vnum=to_vnum, total=len(versions)
                ),
            )

        content = CardBuilder.build_diff_report_content(project, from_v, to_v, entries, show_current)
        return True, content, None

    def _submit_diff_report(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        if not project:
            self.reply_text(
                message_id,
                UI_TEXT["diag_diff_no_active_project"]
            )
            return

        request_id = self.ensure_request_id(message_id, chat_id=chat_id, project_id=project.project_id)
        ref_note = self.format_ref_note(message_id, request_id)

        banner_tpl = UI_TEXT["diag_diff_generating_banner"]
        initial = f"{banner_tpl}\n\n{ref_note}" if ref_note else banner_tpl
        title = UI_TEXT["diag_diff_report_title"]

        _msg_type, card_content = CardBuilder.build_project_response_card(
            project, title, initial, show_buttons=False,
        )
        card_session = self.create_static_card_session(chat_id, reply_to=message_id)
        card_json = json.loads(card_content) if isinstance(card_content, str) else card_content
        card_session.send(card_json)
        card_message_id = card_session.message_id
        if card_message_id:
            try:
                self.register_message_project(card_message_id, project)
            except Exception:
                logger.debug("failed to register message project mapping", exc_info=True)
            try:
                self.ctx.message_linker.register_origin(
                    message_id, request_id=request_id, chat_id=chat_id, project_id=project.project_id
                )
                self.ctx.message_linker.link_reply(message_id, card_message_id)
            except Exception:
                logger.debug("failed to link reply message", exc_info=True)

        spec = TaskSpec(
            chat_id=chat_id,
            queue_key=f"{chat_id}:diff:{project.project_id}",
            name="diff_report",
            task_type="diff_report",
            project_id=project.project_id,
            message_id=message_id,
            origin_message_id=message_id,
            request_id=request_id,
            priority=TaskPriority.NORMAL,
        )

        def _run(task_ctx):
            try:
                try:
                    full_ref = self.format_ref_note(message_id, request_id, run_id=task_ctx.run_id)
                    if card_message_id and full_ref:
                        _msg_type, _card = CardBuilder.build_project_response_card(
                            project, title, f"{banner_tpl}\n\n{full_ref}", show_buttons=False,
                        )
                        _card_json = json.loads(_card) if isinstance(_card, str) else _card
                        card_session.send(_card_json)
                except Exception:
                    logger.debug("failed to update streaming card content", exc_info=True)

                task_ctx.progress(UI_TEXT["diag_step_parsing"], 5)
                if card_message_id:
                    try:
                        progress_tpl = UI_TEXT["diag_diff_generating_progress"]
                        progress_content = f"{progress_tpl.format(step=UI_TEXT['diag_step_parsing'], pct=5)}\n\n{self.format_ref_note(message_id, request_id, run_id=task_ctx.run_id)}"
                        _msg_type, _card = CardBuilder.build_project_response_card(
                            project, title, progress_content, show_buttons=False,
                        )
                        _card_json = json.loads(_card) if isinstance(_card, str) else _card
                        card_session.send(_card_json)
                    except Exception:
                        logger.debug("failed to update streaming card content", exc_info=True)

                ok, content, err = self._build_context_diff_report(chat_id, text, project)
                if not ok:
                    msg = err or UI_TEXT["diag_diff_failed"]
                    final_ref = self.format_ref_note(message_id, request_id, run_id=task_ctx.run_id)
                    final = f"❌ {msg}\n\n{final_ref}" if final_ref else f"❌ {msg}"
                    if card_message_id:
                        _msg_type, _card = CardBuilder.build_project_response_card(
                            project, title, final, show_buttons=False,
                        )
                        _card_json = json.loads(_card) if isinstance(_card, str) else _card
                        card_session.send(_card_json)
                    else:
                        self.reply_text(message_id, msg)
                    return

                task_ctx.progress(UI_TEXT["diag_step_generating"], 80)
                if card_message_id:
                    try:
                        progress_tpl = UI_TEXT["diag_diff_generating_progress"]
                        progress_content = f"{progress_tpl.format(step=UI_TEXT['diag_step_generating'], pct=80)}\n\n{self.format_ref_note(message_id, request_id, run_id=task_ctx.run_id)}"
                        _msg_type, _card = CardBuilder.build_project_response_card(
                            project, title, progress_content, show_buttons=False,
                        )
                        _card_json = json.loads(_card) if isinstance(_card, str) else _card
                        card_session.send(_card_json)
                    except Exception:
                        logger.debug("failed to update streaming card content", exc_info=True)

                final_ref = self.format_ref_note(message_id, request_id, run_id=task_ctx.run_id)
                final = f"{content}\n\n{final_ref}" if final_ref and final_ref not in content else content
                if card_message_id:
                    _msg_type, _card = CardBuilder.build_project_response_card(
                        project, title, final, show_buttons=False,
                        footer=UI_TEXT["diag_diff_usage_footer"],
                    )
                    _card_json = json.loads(_card) if isinstance(_card, str) else _card
                    card_session.send(_card_json)
                else:
                    _msg_type, _card = CardBuilder.build_project_response_card(
                        project, title, final, show_buttons=False,
                        footer=UI_TEXT["diag_diff_usage_footer"],
                    )
                    _card_json = json.loads(_card) if isinstance(_card, str) else _card
                    card_session.send(_card_json)
                    if card_session.message_id:
                        self.register_message_project(card_session.message_id, project)
                task_ctx.progress(UI_TEXT["diag_step_completed"], 100)
            except Exception as e:
                msg = UI_TEXT["diag_diff_exception"].format(error=get_error_detail(e))
                final_ref = self.format_ref_note(message_id, request_id, run_id=getattr(task_ctx, "run_id", None))
                final = f"❌ {msg}\n\n{final_ref}" if final_ref else f"❌ {msg}"
                try:
                    if card_message_id:
                        _msg_type, _card = CardBuilder.build_project_response_card(
                            project, title, final, show_buttons=False,
                        )
                        _card_json = json.loads(_card) if isinstance(_card, str) else _card
                        card_session.send(_card_json)
                except Exception:
                    logger.debug("failed to close streaming card", exc_info=True)
                self.reply_text(message_id, msg)
            finally:
                card_session.close()

        handle = self.scheduler.submit(spec, _run)
        try:
            self.ctx.message_linker.link_task(message_id, handle.run_id)
        except Exception:
            logger.debug("failed to link task", exc_info=True)

        if card_message_id:
            try:
                full_ref = self.format_ref_note(message_id, request_id, run_id=handle.run_id)
                msg = UI_TEXT["diag_diff_started_banner"]
                _msg_type, _card = CardBuilder.build_project_response_card(
                    project, title, f"{msg}\n\n{full_ref}", show_buttons=False,
                )
                _card_json = json.loads(_card) if isinstance(_card, str) else _card
                card_session.send(_card_json)
            except Exception:
                logger.debug("failed to update streaming card content", exc_info=True)
        return handle

    # ------------------------------------------------------------------
    # Message trace
    # ------------------------------------------------------------------
    def show_message_trace(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        arg = ""
        try:
            parts = (text or "").strip().split(None, 1)
            if len(parts) > 1:
                arg = parts[1].strip()
        except Exception:
            arg = ""

        key = arg or message_id
        data = None
        try:
            data = self.ctx.message_linker.query(key)
        except Exception:
            data = None

        if not data:
            self.reply_text(
                message_id,
                UI_TEXT["diag_trace_not_found"].format(key=key)
            )
            return

        # Chat-level isolation: reject trace data belonging to a different chat
        trace_chat_id = data.get("chat_id")
        if trace_chat_id and trace_chat_id != chat_id:
            self.reply_text(
                message_id,
                UI_TEXT["diag_trace_not_found"].format(key=key)
            )
            return

        proj_id = data.get("project_id")
        if project is None and proj_id:
            try:
                project = self.project_manager.get_project_for_chat(proj_id, chat_id)
            except Exception:
                project = None

        content = CardBuilder.build_message_trace_content(data)
        title = UI_TEXT["diag_trace_title"]
        footer = UI_TEXT["diag_trace_usage_footer"]

        if project:
            msg_type, card_content = CardBuilder.build_project_response_card(
                project,
                title,
                content,
                show_buttons=False,
                footer=footer,
            )
        else:
            msg_type, card_content = CardBuilder.build_smart_response_card(
                project=None,
                title=title,
                content=content,
                working_dir=self.get_working_dir(chat_id),
                show_buttons=False,
            )
        self.reply_card(message_id, card_content)
