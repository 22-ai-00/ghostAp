"""Diagnostics handler — task board, context diff report, message trace, unified status."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...card.ui_text import UI_TEXT
from ...mode import PROGRAMMING_MODES
from ...project import (
    context_helper,
    get_project_acp_model_name,
    get_project_acp_tool_name,
)
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

    def show_task_board(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        from ...card.builders.diagnostics import DiagnosticsBuilder
        from ...card.builders.project import ProjectBuilder

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

            content = DiagnosticsBuilder.build_task_board_content(
                tasks, mode_display, groups=groups, project_manager=self.project_manager
            )

            msg_type, card_content = ProjectBuilder.build_smart_response_card(
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
        content = DiagnosticsBuilder.build_task_board_content(tasks, mode_display)

        msg_type, card_content = ProjectBuilder.build_project_response_card(
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

    @staticmethod
    def _read_git_branch(root_path: str) -> str:
        """Read current git branch directly from .git/HEAD without subprocess overhead."""
        if not root_path:
            return ""
        try:
            root = Path(root_path)
            git_entry = root / ".git"
            if git_entry.is_file():
                raw = git_entry.read_text(encoding="utf-8", errors="ignore").strip()
                if raw.lower().startswith("gitdir:"):
                    raw_dir = raw.split(":", 1)[1].strip()
                    git_dir = Path(raw_dir)
                    if not git_dir.is_absolute():
                        git_dir = (root / git_dir).resolve()
                else:
                    return ""
            elif git_entry.is_dir():
                git_dir = git_entry
            else:
                return ""
            head_file = git_dir / "HEAD"
            if not head_file.is_file():
                return ""
            head_content = head_file.read_text(encoding="utf-8", errors="ignore").strip()
            if head_content.startswith("ref: refs/heads/"):
                return head_content[len("ref: refs/heads/"):].strip()
            if head_content.startswith("ref: "):
                return head_content[5:].strip()
            if len(head_content) >= 7:
                return f"detached@{head_content[:7]}"
        except OSError:
            return ""
        return ""

    def _collect_direct_cockpit_info(
        self,
        chat_id: str,
        project: Optional["ProjectContext"],
    ) -> dict:
        """Collect non-blocking Direct programming lane summary for /status cockpit."""
        pid = project.project_id if project else None
        mode_key = ""
        mode_display = "Smart 模式"
        try:
            mode = self.mode_manager.get_mode(chat_id, project_id=pid)
            is_prog = self.mode_manager.is_programming_mode(chat_id, project_id=pid) is True
            if mode in PROGRAMMING_MODES and is_prog:
                from ...card.backend_meta import get_backend_meta

                mode_key = mode.value
                meta = get_backend_meta(mode)
                mode_display = meta.display_name if meta else mode.value
        except Exception:
            mode_key = ""

        tool_name = mode_key or (get_project_acp_tool_name(project, chat_id) if project else "") or ""
        model_name = (
            (get_project_acp_model_name(project, chat_id, tool_name=tool_name) or "")
            if project and tool_name
            else ""
        )
        session_id = ""
        message_count = None
        usage_text = ""
        is_running = False
        elapsed_s = None

        if tool_name:
            try:
                handler = self.get_handler(tool_name)
            except Exception:
                handler = None
            if handler is not None:
                active_getter = getattr(handler, "_get_active_programming_run", None)
                active_run = active_getter(chat_id, project) if callable(active_getter) else None
                started_at = getattr(active_run, "started_at", None) if active_run is not None else None
                if isinstance(started_at, (int, float)):
                    is_running = True
                    elapsed_s = max(0, int(time.monotonic() - float(started_at)))

                mgr_getter = getattr(handler, "_get_session_manager", None)
                mgr = mgr_getter() if callable(mgr_getter) else None
                if mgr is not None and callable(getattr(mgr, "get_session", None)):
                    live = mgr.get_session(chat_id, project_id=pid)
                    live_sid = getattr(live, "session_id", None) if live is not None else None
                    if isinstance(live_sid, str) and live_sid:
                        session_id = live_sid
                        raw_count = getattr(live, "message_count", None)
                        if isinstance(raw_count, int):
                            message_count = raw_count
                        live_model = getattr(live, "_model_name", None)
                        if isinstance(live_model, str) and live_model and not model_name:
                            model_name = live_model
                        usage_getter = getattr(live, "get_usage_info", None)
                        usage = usage_getter() if callable(usage_getter) else None
                        if usage is not None and (getattr(usage, "used", 0) > 0 or getattr(usage, "size", 0) > 0):
                            used = int(getattr(usage, "used", 0) or 0)
                            size = int(getattr(usage, "size", 0) or 0)
                            if size > 0:
                                pct = int(round((used / size) * 100))
                                usage_text = f"{used:,}/{size:,} ({pct}%)"
                            else:
                                usage_text = f"{used:,} tokens"
        return {
            "mode_key": mode_key,
            "mode_display": mode_display,
            "tool_name": tool_name,
            "model_name": model_name,
            "session_id": session_id,
            "message_count": message_count,
            "usage_text": usage_text,
            "is_running": is_running,
            "elapsed_s": elapsed_s,
        }

    def show_unified_status(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        """Show the unified Workspace Cockpit (/status) or inspect a specific task id.

        - /status           → Workspace Cockpit (project, branch, direct lane, engines, locks)
        - /status all       → include completed engine tasks
        - /status <task_id> → detailed status for Deep / Spec / Workflow / scheduler task
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

        # Collect engines (Deep, Spec, Workflow)
        entries = DiagnosticsHelper.get_all_engine_statuses(self.ctx, chat_id, include_done=include_done)

        # Determine admin status for lock hints
        _is_admin = False
        _chat_lock_mgr = getattr(self.ctx, "chat_lock_manager", None)
        if _chat_lock_mgr is not None:
            from ...thread import get_current_sender_id
            _sender = get_current_sender_id() or ""
            if _sender:
                _is_admin = _chat_lock_mgr.is_admin(_sender)

        lock_lines = self._build_lock_status_lines(chat_id, project, is_admin=_is_admin)
        working_dir = (project.root_path if project and getattr(project, "root_path", None) else None) or self.get_working_dir(chat_id)
        git_branch = self._read_git_branch(working_dir)
        direct_info = self._collect_direct_cockpit_info(chat_id, project)

        from ...card.builders.diagnostics import DiagnosticsBuilder

        msg_type, card_content = DiagnosticsBuilder.build_workspace_cockpit_card(
            project=project,
            working_dir=working_dir,
            git_branch=git_branch,
            direct_info=direct_info,
            entries=entries,
            lock_lines=lock_lines,
            include_done=include_done,
        )
        if msg_type == "text":
            self.reply_text(message_id, card_content)
        else:
            self.reply_card(message_id, card_content)

    def _show_task_detail(
        self, message_id: str, chat_id: str, task_id: str, project: Optional["ProjectContext"] = None
    ):
        """Show detailed status for a specific task by task_id."""
        from ...card.builders.deep import DeepBuilder
        from ...card.builders.diagnostics import DiagnosticsBuilder
        from ...card.builders.project import ProjectBuilder
        from ...card.models import EngineCardState

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
                msg_type, card_content = DeepBuilder.build_info_card(
                    project,
                    EngineCardState(
                        title=title,
                        content=content,
                        engine_name=engine_name,
                        show_buttons=False,
                    ),
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
                msg_type, card_content = DeepBuilder.build_info_card(
                    project,
                    EngineCardState(
                        title=title,
                        content=content,
                        engine_name=f"Spec({engine_name})",
                        show_buttons=False,
                    ),
                )
                self.reply_card(message_id, card_content)
                return

        wf_mgr = getattr(self.ctx, "workflow_engine_manager", None)
        if wf_mgr is not None and callable(getattr(wf_mgr, "list_engines", None)):
            wf_engines = wf_mgr.list_engines(chat_id)
            if isinstance(wf_engines, (list, tuple)):
                for engine in wf_engines:
                    wp = getattr(engine, "workflow_project", None)
                    wf_id = getattr(wp, "workflow_id", "") if wp else ""
                    if wf_id and (wf_id == task_id or task_id in wf_id):
                        wf_status = getattr(getattr(wp, "status", None), "value", "unknown")
                        wf_name = getattr(wp, "name", "") or "Workflow"
                        metrics = getattr(wp, "metrics", None)
                        completed = int(getattr(metrics, "completed_agents", 0) or 0) if metrics else 0
                        failed = int(getattr(metrics, "failed_agents", 0) or 0) if metrics else 0
                        phases = getattr(wp, "phases", ()) or ()
                        content_lines = [
                            f"**Workflow**：`{wf_name}` (`{wf_id}`)",
                            f"• 状态：`{wf_status}`",
                            f"• 阶段数：{len(phases)} · Agent：完成 {completed} / 失败 {failed}",
                        ]
                        msg_type, card_content = DeepBuilder.build_info_card(
                            project,
                            EngineCardState(
                                title=UI_TEXT.get("diag_task_detail_workflow_title", "📊 Workflow 任务详情"),
                                content="\n".join(content_lines),
                                engine_name=getattr(engine, "engine_name", "Workflow"),
                                show_buttons=False,
                            ),
                        )
                        self.reply_card(message_id, card_content)
                        return

        # Also check scheduler by task_id
        state = self.scheduler.get_state_by_task_id(task_id, chat_id=chat_id)
        if state:
            content = DiagnosticsBuilder.build_task_detail_content(state)
            msg_type, card_content = ProjectBuilder.build_smart_response_card(
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

        from ...card.builders.diagnostics import DiagnosticsBuilder

        content = DiagnosticsBuilder.build_diff_report_content(project, from_v, to_v, entries, show_current)
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

        from ...card.builders.project import ProjectBuilder

        _msg_type, card_content = ProjectBuilder.build_project_response_card(
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
                        _msg_type, _card = ProjectBuilder.build_project_response_card(
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
                        _msg_type, _card = ProjectBuilder.build_project_response_card(
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
                        _msg_type, _card = ProjectBuilder.build_project_response_card(
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
                        _msg_type, _card = ProjectBuilder.build_project_response_card(
                            project, title, progress_content, show_buttons=False,
                        )
                        _card_json = json.loads(_card) if isinstance(_card, str) else _card
                        card_session.send(_card_json)
                    except Exception:
                        logger.debug("failed to update streaming card content", exc_info=True)

                final_ref = self.format_ref_note(message_id, request_id, run_id=task_ctx.run_id)
                final = f"{content}\n\n{final_ref}" if final_ref and final_ref not in content else content
                if card_message_id:
                    _msg_type, _card = ProjectBuilder.build_project_response_card(
                        project, title, final, show_buttons=False,
                        footer=UI_TEXT["diag_diff_usage_footer"],
                    )
                    _card_json = json.loads(_card) if isinstance(_card, str) else _card
                    card_session.send(_card_json)
                else:
                    _msg_type, _card = ProjectBuilder.build_project_response_card(
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
                        _msg_type, _card = ProjectBuilder.build_project_response_card(
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
                _msg_type, _card = ProjectBuilder.build_project_response_card(
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

        from ...card.builders.diagnostics import DiagnosticsBuilder
        from ...card.builders.project import ProjectBuilder

        content = DiagnosticsBuilder.build_message_trace_content(data)
        title = UI_TEXT["diag_trace_title"]
        footer = UI_TEXT["diag_trace_usage_footer"]

        if project:
            msg_type, card_content = ProjectBuilder.build_project_response_card(
                project,
                title,
                content,
                show_buttons=False,
                footer=footer,
            )
        else:
            msg_type, card_content = ProjectBuilder.build_smart_response_card(
                project=None,
                title=title,
                content=content,
                working_dir=self.get_working_dir(chat_id),
                show_buttons=False,
            )
        self.reply_card(message_id, card_content)
