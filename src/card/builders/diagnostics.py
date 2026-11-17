from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from src.card.actions import dispatch as action_ids
from src.card.backend_meta import get_backend_meta
from src.card.models import EngineStatusEntry
from src.card.shared import build_responsive_layout
from src.card.ui_text import UI_TEXT

from .core import CoreBuilder

if TYPE_CHECKING:
    from src.project.context import ProjectContext
    from src.tasking.scheduler import TaskState

logger = logging.getLogger(__name__)


class DiagnosticsBuilder:
    """Diagnostics-related card building utilities."""

    @staticmethod
    def _get_status_emoji(status) -> str:
        val = getattr(status, "value", status)
        return {
            "queued": "⏳",
            "running": "🔄",
            "succeeded": "✅",
            "failed": "❌",
            "canceled": "⛔",
        }.get(str(val), "❓")

    @staticmethod
    def build_task_board_content(
        tasks: list[TaskState],
        mode_display: str = "",
        groups: Optional[dict[str, list[TaskState]]] = None,
        project_manager = None,
    ) -> str:
        """Build the Markdown content for task board."""
        lines = []
        if mode_display:
            lines.append(UI_TEXT["diag_current_mode"].format(mode=mode_display))
            lines.append("")

        if not tasks:
            lines.append(UI_TEXT["diag_no_active_tasks"])
        elif groups is not None:
            # Multi-project grouped view
            for pid, items in groups.items():
                proj_name = UI_TEXT["system_no_project_display"]
                if pid and project_manager:
                    try:
                        p = project_manager.get_project_for_diagnostics(pid)
                        if p:
                            proj_name = p.project_name
                    except Exception:
                        logger.debug("failed to get project name", exc_info=True)

                id_info = UI_TEXT["diag_project_id_fmt"].format(id=pid) if pid else ""
                lines.append(UI_TEXT["diag_project_header"].format(
                    name=proj_name, id_info=id_info
                ))

                for st in items[:10]:
                    emoji = DiagnosticsBuilder._get_status_emoji(st.status)
                    pct = UI_TEXT["diag_task_pct_fmt"].format(pct=st.progress_percent) if st.progress_percent is not None else ""
                    msg = UI_TEXT["diag_task_msg_fmt"].format(msg=st.progress_message) if st.progress_message else ""
                    lines.append(UI_TEXT["diag_task_line"].format(
                        emoji=emoji,
                        run_id=st.run_id,
                        name=st.spec.name,
                        task_type=st.spec.task_type,
                        pct=pct,
                        msg=msg
                    ))
        else:
            # Single project list view
            for st in tasks:
                emoji = DiagnosticsBuilder._get_status_emoji(st.status)
                pct = UI_TEXT["diag_task_pct_fmt"].format(pct=st.progress_percent) if st.progress_percent is not None else ""
                msg = UI_TEXT["diag_task_msg_fmt"].format(msg=st.progress_message) if st.progress_message else ""
                lines.append(UI_TEXT["diag_task_line"].format(
                    emoji=emoji,
                    run_id=st.run_id,
                    name=st.spec.name,
                    task_type=st.spec.task_type,
                    pct=pct,
                    msg=msg
                ))

        return "\n".join(lines) if lines else UI_TEXT["diag_no_tasks"]

    @staticmethod
    def format_engine_status_info(mode: str, p: any) -> str:
        """Format the 'info' string for different engines (Deep/Spec)."""
        from src.utils.text import format_duration
        dur = format_duration(p.duration()) if p.duration() else ""

        deep_label = UI_TEXT["diag_engine_deep"]
        spec_label = UI_TEXT["diag_engine_spec"]

        if mode == deep_label:
            return f"{dur}" if dur else (p.status.value if hasattr(p.status, "value") else str(p.status))

        if mode == spec_label:
            criteria = f"{p.satisfied_count}/{p.total_criteria}" if p.total_criteria else ""
            phase = ""
            if p.current_cycle:
                phase = p.current_cycle.phase.display_name
            parts = [UI_TEXT["diag_status_cycle"].format(cycle=p.current_cycle_number)]
            if phase:
                parts.append(phase)
            if criteria:
                parts.append(UI_TEXT["diag_status_criteria"].format(criteria=criteria))
            if dur:
                parts.append(dur)
            return " · ".join(parts) if parts else (p.status.value if hasattr(p.status, "value") else str(p.status))

        return ""

    @staticmethod
    def build_unified_status_content(
        entries: list[EngineStatusEntry],
        include_done: bool = False,
        project_name: str = ""
    ) -> str:
        """Build the Markdown content for unified status across all engines."""
        if not entries:
            content = UI_TEXT["diag_no_engine_tasks"]
            if project_name:
                content += UI_TEXT["project_current_header"].format(name=project_name) + "\n\n"
            content += UI_TEXT["diag_engine_launch_hints"]
            return content

        status_emoji_map = {
            "idle": "⏳",
            "planning": "🧠",
            "executing": "🔄",
            "running": "🔄",
            "analyzing": "🧠",
            "paused": "⏸️",
            "completed": "✅",
            "failed": "❌",
            "aborted": "⚠️",
            "clarifying": "❓",
        }

        lines = [UI_TEXT["diag_unified_status_header"].format(count=len(entries))]
        for e in entries:
            emoji = status_emoji_map.get(e.status, "❓")
            tid_short = f" `{e.task_id[-12:]}`" if e.task_id else ""
            lines.append(UI_TEXT["diag_engine_line"].format(
                emoji=emoji,
                mode=e.mode,
                name=e.name,
                tid=tid_short,
                info=e.info
            ))

        if not include_done:
            lines.append(UI_TEXT["diag_status_all_hint"])

        return "\n".join(lines)

    @staticmethod
    def build_workspace_cockpit_card(
        project: Optional["ProjectContext"],
        working_dir: str,
        *,
        git_branch: str = "",
        direct_info: Optional[dict] = None,
        entries: Optional[list[EngineStatusEntry]] = None,
        lock_lines: str = "",
        include_done: bool = False,
    ) -> tuple[str, str]:
        """Build the unified Workspace Cockpit (/status) Card JSON 2.0."""
        d_info = direct_info or {}
        entries = list(entries or [])
        proj_name = getattr(project, "project_name", None) if project else None
        proj_id = getattr(project, "project_id", None) if project else None
        path_display = (getattr(project, "root_path", None) if project else None) or working_dir or "~"

        # 1. Workspace & git branch
        ws_lines = [
            "**📂 工作区与分支**",
            f"• 路径：`{path_display}`",
        ]
        if git_branch:
            ws_lines.append(f"• 分支：`{git_branch}`")
        elements: list[dict] = [{"tag": "markdown", "content": "\n".join(ws_lines)}]

        # 2. Direct programming lane
        mode_key = str(d_info.get("mode_key") or "").strip().lower()
        tool_name = str(d_info.get("tool_name") or "").strip().lower()
        meta = get_backend_meta(tool_name or mode_key)
        is_running = bool(d_info.get("is_running"))
        elapsed_s = d_info.get("elapsed_s")
        if meta is not None and mode_key:
            mode_label = f"{meta.emoji} **{meta.display_name}**"
        else:
            mode_label = f"🧠 **{d_info.get('mode_display') or 'Smart 模式'}**"
        state_badge = (
            f"（执行中 · 已运行 {int(elapsed_s)}s）"
            if is_running and isinstance(elapsed_s, (int, float))
            else ("（执行中）" if is_running else "（空闲）")
        )
        direct_lines = [
            "**💻 Direct 编程通道**",
            f"• 当前模式：{mode_label}{state_badge}",
        ]
        if tool_name:
            model_display = str(d_info.get("model_name") or "").strip() or "default"
            direct_lines.append(f"• 工具 / 模型：`{tool_name}` / `{model_display}`")
        sid = str(d_info.get("session_id") or "").strip()
        msg_count = d_info.get("message_count")
        usage_text = str(d_info.get("usage_text") or "").strip()
        session_bits: list[str] = []
        if sid:
            short_sid = f"{sid[:12]}…" if len(sid) > 12 else sid
            count_part = f" · {int(msg_count)} 条消息" if msg_count is not None else ""
            session_bits.append(f"会话 `{short_sid}`{count_part}")
        if usage_text:
            session_bits.append(f"上下文 `{usage_text}`")
        if session_bits:
            direct_lines.append("• " + " · ".join(session_bits))
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "\n".join(direct_lines)})

        # 3. Engines lane (Workflow / Deep / Spec)
        elements.append({"tag": "hr"})
        if entries:
            engine_md = DiagnosticsBuilder.build_unified_status_content(
                entries,
                include_done=include_done,
                project_name=proj_name or "",
            )
        else:
            engine_md = (
                "**⚙️ 长任务引擎（Workflow / Deep / Spec）**\n"
                "• 当前无活跃 Workflow / Deep / Spec 任务\n"
                "<font color='grey'>发送 `/wf <需求>`、`/deep <需求>` 或 `/spec <需求>` 可随时启动</font>"
            )
        elements.append({"tag": "markdown", "content": engine_md})

        # 4. Lock status section
        if lock_lines:
            elements.append({"tag": "hr"})
            elements.append({"tag": "markdown", "content": lock_lines})

        # 5. Context-aware action buttons
        def _cb_btn(label: str, payload: dict, *, btn_type: str = "default") -> dict:
            val = dict(payload)
            return {
                "tag": "button",
                "text": {"tag": "plain_text", "content": label},
                "type": btn_type,
                "value": val,
                "behaviors": [{"type": "callback", "value": dict(val)}],
            }

        buttons: list[dict] = []
        if tool_name:
            picker_payload = {"action": action_ids.OPEN_ACP_MODEL_PICKER, "tool_name": tool_name}
            cmds_payload = {"action": action_ids.SHOW_TOOL_COMMANDS, "tool_name": tool_name}
            if proj_id:
                picker_payload["project_id"] = proj_id
                cmds_payload["project_id"] = proj_id
            if d_info.get("model_name"):
                picker_payload["current_model"] = d_info["model_name"]
            buttons.append(_cb_btn(UI_TEXT["system_acp_switch_model_btn"], picker_payload))
            buttons.append(_cb_btn(UI_TEXT["system_acp_native_cmds_btn"], cmds_payload))
        if is_running:
            stop_payload = {"action": action_ids.PROGRAMMING_STOP}
            if tool_name:
                stop_payload["tool_name"] = tool_name
            if proj_id:
                stop_payload["project_id"] = proj_id
            buttons.append(_cb_btn("⏹️ 停止当前任务", stop_payload, btn_type="danger"))
        buttons.append(
            _cb_btn(
                UI_TEXT["project_board_title"],
                {"action": action_ids.SHOW_BOARD},
                btn_type="default" if is_running else "primary",
            )
        )
        elements.append({"tag": "hr"})
        elements.extend(build_responsive_layout(buttons, layout="mobile"))

        title_base = UI_TEXT.get("diag_cockpit_title", "🧭 工作区总览")
        header_title = f"{title_base} · {proj_name}" if proj_name else title_base
        header_color = meta.template_color if (meta is not None and mode_key) else "blue"
        card = CoreBuilder._wrap_card(header_title, header_color, elements)
        card["config"]["summary"] = {"content": header_title}
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_task_detail_content(state: TaskState) -> str:
        """Build the Markdown content for task detail."""
        emoji = DiagnosticsBuilder._get_status_emoji(state.status)
        lines = [
            UI_TEXT["diag_task_name_label"].format(name=state.spec.name),
            UI_TEXT["diag_task_type_label"].format(type=state.spec.task_type),
            UI_TEXT["diag_task_status_label"].format(
                emoji=emoji, status=state.status.value if hasattr(state.status, "value") else state.status
            ),
            UI_TEXT["diag_task_run_id_label"].format(id=state.run_id),
        ]
        if state.spec.task_id:
            lines.append(UI_TEXT["diag_task_id_label"].format(id=state.spec.task_id))
        if state.progress_message:
            lines.append(UI_TEXT["diag_task_progress_label"].format(msg=state.progress_message))
        return "\n".join(lines)

    @staticmethod
    def build_diff_report_content(
        project,
        from_v,
        to_v,
        entries: list,
        show_current: bool = False,
    ) -> str:
        """Build the Markdown content for context diff report."""

        def _short(s: str, n: int = 180) -> str:
            s = (s or "").replace("\r", " ").replace("\n", " ")
            return s if len(s) <= n else (s[: n - 1] + "…")

        lines = [UI_TEXT["diag_diff_title"]]
        lines.append(
            UI_TEXT["diag_diff_project_label"].format(
                name=project.project_name, id=project.project_id
            )
        )
        if show_current:
            lines.append(
                UI_TEXT["diag_diff_range_current_label"].format(
                    from_v=from_v.version_number
                )
            )
        elif to_v:
            lines.append(
                UI_TEXT["diag_diff_range_label"].format(
                    from_v=from_v.version_number, to_v=to_v.version_number
                )
            )

        lines.append(
            UI_TEXT["diag_diff_start_reason_label"].format(
                reason=_short(from_v.reason, 120)
            )
        )
        if to_v:
            lines.append(
                UI_TEXT["diag_diff_end_reason_label"].format(
                    reason=_short(to_v.reason, 120)
                )
            )
        lines.append(UI_TEXT["diag_diff_added_entries_label"].format(count=len(entries)))

        if not entries:
            lines.append("")
            lines.append(UI_TEXT["diag_diff_no_entries"])
            return "\n".join(lines)

        from src.project import ContextEntryType

        file_changes, mode_changes, deep_results, summaries, conversations, others = [], [], [], [], [], []
        for e in entries:
            et = getattr(e, "entry_type", None)
            if et == ContextEntryType.FILE_CHANGE:
                file_changes.append(e)
            elif et == ContextEntryType.MODE_TRANSITION:
                mode_changes.append(e)
            elif et == ContextEntryType.DEEP_ENGINE_RESULT:
                deep_results.append(e)
            elif et == ContextEntryType.AI_SUMMARY:
                summaries.append(e)
            elif et == ContextEntryType.CONVERSATION:
                conversations.append(e)
            else:
                others.append(e)

        if file_changes:
            lines.append("")
            uniq, seen = [], set()
            for e in file_changes:
                p = (e.content or "").strip()
                if not p or p in seen:
                    continue
                seen.add(p)
                uniq.append(p)
            lines.append(
                UI_TEXT["diag_diff_file_changes_header"].format(count=len(uniq))
            )
            for p in uniq[:20]:
                lines.append(f"- `{p}`")

        if deep_results:
            lines.append("")
            lines.append(
                UI_TEXT["diag_diff_deep_results_header"].format(count=len(deep_results))
            )
            for e in deep_results[:5]:
                name = (e.metadata or {}).get("name") or "unknown"
                tasks = (e.metadata or {}).get("tasks") or []
                done = sum(1 for t in tasks if isinstance(t, dict) and t.get("status") == "completed")
                lines.append(
                    UI_TEXT["diag_diff_deep_item"].format(
                        name=name, done=done, total=len(tasks)
                    )
                )

        if mode_changes:
            lines.append("")
            lines.append(
                UI_TEXT["diag_diff_mode_changes_header"].format(count=len(mode_changes))
            )
            for e in mode_changes[-10:]:
                reason = (e.metadata or {}).get("reason", "")
                lines.append(f"- {_short(e.content, 120)}{f'（{_short(reason, 80)}）' if reason else ''}")

        if summaries:
            lines.append("")
            lines.append(
                UI_TEXT["diag_diff_summaries_header"].format(count=len(summaries))
            )
            for e in summaries[-5:]:
                lines.append(f"- {_short(e.content, 200)}")

        if conversations:
            lines.append("")
            lines.append(
                UI_TEXT["diag_diff_conversations_header"].format(count=len(conversations))
            )
            for e in conversations[-8:]:
                role = (e.metadata or {}).get("role", "?")
                lines.append(f"- `{role}`: {_short(e.content, 160)}")

        if others:
            lines.append("")
            lines.append(UI_TEXT["diag_diff_others_header"].format(count=len(others)))
            for e in others[:10]:
                et = getattr(e, "entry_type", None)
                sm = getattr(e, "source_mode", None)
                et_val = getattr(et, "value", str(et))
                sm_val = getattr(sm, "value", str(sm))
                lines.append(f"- `{et_val}`/`{sm_val}`: {_short(e.content, 160)}")

        content = "\n".join(lines)
        if len(content) > 7000:
            content = content[:7000] + UI_TEXT["diag_diff_truncated"]
        return content

    @staticmethod
    def build_message_trace_content(data: dict) -> str:
        """Build the Markdown content for message trace results."""
        lines = []

        # Origin info
        origin = data.get("origin_id")
        req_id = data.get("request_id")
        pid = data.get("project_id")

        if origin:
            lines.append(f"{UI_TEXT['diag_label_origin_id']}: `{origin}`")
        if req_id:
            lines.append(f"{UI_TEXT['diag_label_request_id']}: `{req_id}`")
        if pid:
            lines.append(f"{UI_TEXT['diag_label_project_id']}: `{pid}`")

        # Replies
        replies = data.get("replies", [])
        if replies:
            lines.append("")
            lines.append(UI_TEXT["diag_trace_replies_header"].format(count=len(replies)))
            for rid in replies:
                lines.append(f"- `{rid}`")

        # Run IDs
        runs = data.get("run_ids", [])
        if runs:
            lines.append("")
            lines.append(UI_TEXT["diag_trace_runs_header"].format(count=len(runs)))
            for run_id in runs:
                lines.append(f"- `{run_id}`")

        return "\n".join(lines)
