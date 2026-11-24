"""WorkflowProgressRenderer — renders workflow progress tree for Feishu cards."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

from src.card.actions.dispatch import WORKFLOW_STOP_RUNNING
from src.card.render.buttons import build_responsive_button_row
from src.card.render.renderer import render_card as render_card
from src.card.ui_text import UI_TEXT
from src.utils.text import format_duration, format_elapsed_clock

from .models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    WorkflowProject,
    WorkflowStatus,
)
from .renderer_completion import (
    _FINDING_ICONS as _FINDING_ICONS,
)
from .renderer_completion import (
    _RESULT_JSON_PREFIX_RE as _RESULT_JSON_PREFIX_RE,
)
from .renderer_completion import (
    _VERDICT_LABELS as _VERDICT_LABELS,
)
from .renderer_completion import (
    _append_fitting_image_elements as _append_fitting_image_elements,
)
from .renderer_completion import (
    _brief_for_workflow_status as _brief_for_workflow_status,
)
from .renderer_completion import (
    _brief_section_markdown as _brief_section_markdown,
)
from .renderer_completion import (
    _card_text_for_result_brief as _card_text_for_result_brief,
)
from .renderer_completion import (
    _complete_text_or_default as _complete_text_or_default,
)
from .renderer_completion import (
    _completion_process_markdown as _completion_process_markdown,
)
from .renderer_completion import (
    _completion_report_status_markdown as _completion_report_status_markdown,
)
from .renderer_completion import (
    _decode_result_layers as _decode_result_layers,
)
from .renderer_completion import (
    _json_fence_body as _json_fence_body,
)
from .renderer_completion import (
    _render_image_block_element as _render_image_block_element,
)
from .renderer_completion import (
    _render_project_image_elements as _render_project_image_elements,
)
from .renderer_completion import (
    _render_result_brief_elements as _render_result_brief_elements,
)
from .renderer_completion import (
    _safe_decoded_result as _safe_decoded_result,
)
from .renderer_completion import (
    _safe_result_brief as _safe_result_brief,
)
from .renderer_completion import (
    _safe_terminal_error as _safe_terminal_error,
)
from .renderer_completion import (
    _sanitize_result_brief as _sanitize_result_brief,
)
from .renderer_completion import (
    _sanitize_result_display_text as _sanitize_result_display_text,
)
from .renderer_completion import (
    render_completion_card as render_completion_card,
)
from .renderer_completion import (
    render_completion_cards as render_completion_cards,
)
from .renderer_elements import (
    _CARD_MAX_BYTES as _CARD_MAX_BYTES,
)
from .renderer_elements import (
    _ELLIPSIS as _ELLIPSIS,
)
from .renderer_elements import (
    _LABEL_TRUNCATION_LIMIT as _LABEL_TRUNCATION_LIMIT,
)
from .renderer_elements import (
    _SUBAGENT_DISPLAY_LIMIT as _SUBAGENT_DISPLAY_LIMIT,
)
from .renderer_elements import (
    _SUBAGENT_STATUS_META as _SUBAGENT_STATUS_META,
)
from .renderer_elements import (
    _WORKFLOW_FRAGMENT_MAX_BYTES as _WORKFLOW_FRAGMENT_MAX_BYTES,
)
from .renderer_elements import (
    _WORKFLOW_FRAGMENT_MAX_NODES as _WORKFLOW_FRAGMENT_MAX_NODES,
)
from .renderer_elements import (
    _WORKFLOW_PAGE_KEY_FIELD as _WORKFLOW_PAGE_KEY_FIELD,
)
from .renderer_elements import (
    _WORKFLOW_WIRE_RESERVE_BYTES as _WORKFLOW_WIRE_RESERVE_BYTES,
)
from .renderer_elements import (
    _WORKFLOW_WIRE_RESERVE_NODES as _WORKFLOW_WIRE_RESERVE_NODES,
)
from .renderer_elements import (
    STATUS_ICONS as STATUS_ICONS,
)
from .renderer_elements import (
    WORKFLOW_STATUS_ICONS as WORKFLOW_STATUS_ICONS,
)
from .renderer_elements import (
    _binding_config_markdown as _binding_config_markdown,
)
from .renderer_elements import (
    _binding_model_parts as _binding_model_parts,
)
from .renderer_elements import (
    _binding_role_markdown as _binding_role_markdown,
)
from .renderer_elements import (
    _binding_roles as _binding_roles,
)
from .renderer_elements import (
    _binding_value as _binding_value,
)
from .renderer_elements import (
    _card_text_for_agent_output as _card_text_for_agent_output,
)
from .renderer_elements import (
    _collapsible_panel as _collapsible_panel,
)
from .renderer_elements import (
    _column as _column,
)
from .renderer_elements import (
    _column_set as _column_set,
)
from .renderer_elements import (
    _escape_md as _escape_md,
)
from .renderer_elements import (
    _format_tokens as _format_tokens,
)
from .renderer_elements import (
    _hr_element as _hr_element,
)
from .renderer_elements import (
    _md_element as _md_element,
)
from .renderer_elements import (
    _middle_ellipsis as _middle_ellipsis,
)
from .renderer_elements import (
    _pct as _pct,
)
from .renderer_elements import (
    _render_subagent_lines as _render_subagent_lines,
)
from .renderer_elements import (
    _safe_dynamic_markdown as _safe_dynamic_markdown,
)
from .renderer_elements import (
    _safe_dynamic_text as _safe_dynamic_text,
)
from .renderer_elements import (
    _set_workflow_page_key as _set_workflow_page_key,
)
from .renderer_elements import (
    _unicode_progress_bar as _unicode_progress_bar,
)
from .renderer_elements import (
    _utf8_safe_text as _utf8_safe_text,
)
from .renderer_execution_pages import (
    _AGENT_CARD_CACHE as _AGENT_CARD_CACHE,
)
from .renderer_execution_pages import (
    _AGENT_CARD_CACHE_LOCK as _AGENT_CARD_CACHE_LOCK,
)
from .renderer_execution_pages import (
    _AGENT_CARD_CACHE_MAX as _AGENT_CARD_CACHE_MAX,
)
from .renderer_execution_pages import (
    _EXECUTION_DISPLAY_TEXT_FIELDS as _EXECUTION_DISPLAY_TEXT_FIELDS,
)
from .renderer_execution_pages import (
    _block_cache_token as _block_cache_token,
)
from .renderer_execution_pages import (
    _execution_content_block as _execution_content_block,
)
from .renderer_execution_pages import (
    _execution_header_template as _execution_header_template,
)
from .renderer_execution_pages import (
    _execution_terminal as _execution_terminal,
)
from .renderer_execution_pages import (
    _identified_direct_agents as _identified_direct_agents,
)
from .renderer_execution_pages import (
    _ordered_direct_agents as _ordered_direct_agents,
)
from .renderer_execution_pages import (
    _paginate_progress_cards as _paginate_progress_cards,
)
from .renderer_execution_pages import (
    _prepare_execution_blocks_for_wire as _prepare_execution_blocks_for_wire,
)
from .renderer_execution_pages import (
    _render_agent_execution_cards as _render_agent_execution_cards,
)
from .renderer_execution_pages import (
    _terminalize_execution_blocks as _terminalize_execution_blocks,
)
from .renderer_execution_pages import (
    clear_agent_execution_card_cache as clear_agent_execution_card_cache,
)

# Sentinel marker list used by lint-level defensive checks.
# An empty tuple means "no markers configured → no checks applied". Tests can
# monkey-patch this to inject sentinel values and verify the defensive gate.
_AGENT_OUTPUT_FORBIDDEN_MARKERS: tuple[str, ...] = ()


class WorkflowGenerationRenderer:
    """Render the immutable Agent Pool while the workflow script is generated."""

    def __init__(
        self,
        *,
        requirement: str,
        agent_pool: Any,
        orchestrator_agent_id: str,
        orchestrator_was_auto: bool = False,
        chat_id: str = "",
        project_id: str = "",
    ) -> None:
        self.requirement = str(requirement or "")
        self.agent_pool = tuple(agent_pool or ())
        self.orchestrator_agent_id = str(orchestrator_agent_id or "")
        self.orchestrator_was_auto = bool(orchestrator_was_auto)
        self.chat_id = str(chat_id or "")
        self.project_id = str(project_id or "")

    @staticmethod
    def _one_line(value: Any, *, limit: int) -> str:
        text = _safe_dynamic_text(value, one_line=True)
        if len(text) <= limit:
            return _escape_md(text)
        return _escape_md(text[: max(0, limit - 1)].rstrip() + "…")

    def render(
        self,
        *,
        current_activity: str | None = None,
        elapsed_seconds: int = 0,
        terminal_status: str | None = None,
    ) -> dict[str, Any]:
        """Render the current generation or terminal Card JSON 2.0 payload."""
        orchestrator = next(
            (
                binding
                for binding in self.agent_pool
                if str(_binding_value(binding, "agent_id", "agentId", default=""))
                == self.orchestrator_agent_id
            ),
            None,
        )
        raw_orchestrator_label = (
            f"Auto → {self.orchestrator_agent_id}"
            if self.orchestrator_was_auto
            else self.orchestrator_agent_id
        ) or "未解析"
        orchestrator_label = _safe_dynamic_markdown(
            raw_orchestrator_label,
            one_line=True,
        )
        orchestrator_name = _safe_dynamic_markdown(
            _binding_value(
                orchestrator,
                "display_name",
                "displayName",
                default=_binding_value(orchestrator, "tool_name", "toolName", default=""),
            )
            or "",
            one_line=True,
        )
        normalized_terminal = str(terminal_status or "").strip().lower()
        is_cancelled = normalized_terminal == "cancelled"
        is_failed = normalized_terminal == "failed"
        is_completed = normalized_terminal == "completed"
        is_terminal = normalized_terminal in {"cancelled", "failed", "completed"}

        if is_cancelled:
            default_activity = "Workflow 任务已由用户停止"
        elif is_failed:
            default_activity = "Workflow 编排生成失败"
        elif is_completed:
            default_activity = "Workflow 编排脚本已生成"
        elif self.orchestrator_agent_id:
            default_activity = f"{self.orchestrator_agent_id} 正在生成并验证 Workflow 编排脚本"
        else:
            default_activity = "正在生成并验证 Workflow 编排脚本"
        activity = self._one_line(current_activity or default_activity, limit=240)
        requirement = self._one_line(self.requirement, limit=600)
        elapsed = max(0, int(elapsed_seconds or 0))
        elapsed_text = format_duration(elapsed)

        pool_lines: list[str] = []
        for binding in self.agent_pool:
            raw_agent_id = str(
                _binding_value(binding, "agent_id", "agentId", default="?")
            )
            agent_id = _safe_dynamic_markdown(raw_agent_id, one_line=True)
            display_name = _safe_dynamic_markdown(
                _binding_value(
                    binding,
                    "display_name",
                    "displayName",
                    default=_binding_value(binding, "tool_name", "toolName", default="?"),
                ),
                one_line=True,
            )
            orch_badge = " · 主编排" if raw_agent_id == self.orchestrator_agent_id else ""
            pool_lines.append(
                f"- **{agent_id}** · {display_name}{orch_badge}\n  {_binding_config_markdown(binding)}"
            )

        if is_cancelled:
            status_content = (
                "**CANCELLED** · Workflow 已取消\n"
                f"**最终状态** · {activity}\n"
                f"⏱ 结束于启动后 {elapsed_text}"
            )
        elif is_failed:
            status_content = (
                "**FAILED** · Workflow 生成失败\n"
                f"**最终状态** · {activity}\n"
                f"⏱ 结束于启动后 {elapsed_text}"
            )
        elif is_completed:
            status_content = (
                "**COMPLETED** · 编排脚本已生成\n"
                f"**最终状态** · {activity}\n"
                f"⏱ 生成耗时 {elapsed_text}\n\n"
                "① 生成脚本（已完成） → ② 验证计划 → ③ 执行节点 → ④ 汇总结果"
            )
        else:
            status_content = (
                "**当前阶段** · 生成并验证编排脚本\n"
                f"**当前操作** · {activity}\n"
                f"⏱ 已等待 {elapsed_text}\n\n"
                "① 生成脚本（进行中） → ② 验证计划 → ③ 执行节点 → ④ 汇总结果"
            )

        orchestrator_line = (
            f"**主编排 Agent** · {orchestrator_label}"
            f"{f' · {orchestrator_name}' if orchestrator_name else ''}"
        )
        if orchestrator is not None:
            orchestrator_line += f"\n{_binding_config_markdown(orchestrator)}"

        elements: list[dict[str, Any]] = [
            {
                "tag": "markdown",
                "content": status_content,
            },
            {"tag": "hr"},
            {
                "tag": "markdown",
                "content": f"**任务**\n{requirement or '未提供任务描述'}",
            },
            {"tag": "hr"},
            {
                "tag": "markdown",
                "content": orchestrator_line,
            },
            {
                "tag": "markdown",
                "content": (
                    f"**已锁定 Agent Pool** · {len(self.agent_pool)} 个\n"
                    + ("\n".join(pool_lines) if pool_lines else "暂无 Agent")
                    + "\n\n后续节点只会从此池分配；执行期间无需再次选择。"
                ),
            },
        ]

        if not is_terminal:
            stop_value = {
                "action": WORKFLOW_STOP_RUNNING,
                "chat_id": self.chat_id,
                "project_id": self.project_id,
            }
            stop_button = {
                "tag": "button",
                "text": {
                    "tag": "plain_text",
                    "content": UI_TEXT.get("workflow_btn_stop_generation", "⏹️ 停止生成"),
                },
                "type": "danger",
                "value": stop_value,
                "behaviors": [{"type": "callback", "value": dict(stop_value)}],
                "confirm": {
                    "title": {
                        "tag": "plain_text",
                        "content": UI_TEXT.get("workflow_btn_confirm_stop_title", "确认停止 Workflow？"),
                    },
                    "text": {
                        "tag": "plain_text",
                        "content": UI_TEXT.get(
                            "workflow_btn_confirm_stop_body",
                            "正在执行的步骤将中断，已完成的部分不受影响。",
                        ),
                    },
                },
            }
            elements.append({"tag": "hr"})
            elements.extend(build_responsive_button_row([stop_button]))

        if is_cancelled:
            title_content = "Workflow · 已取消"
            subtitle_content = "执行已停止"
            template = "grey"
        elif is_failed:
            title_content = "Workflow · 生成失败"
            subtitle_content = "编排已终止"
            template = "red"
        elif is_completed:
            title_content = "Workflow · 编排已生成"
            subtitle_content = "编排脚本已就绪"
            template = "green"
        else:
            title_content = "Workflow · 生成编排中"
            subtitle_content = "Agent Pool 已锁定"
            template = "blue"

        summary_content = " ".join(title_content.split()) or "Workflow 生成状态"
        if len(summary_content) < 8:
            summary_content = f"Workflow · {summary_content}"
        if len(summary_content) > 60:
            summary_content = summary_content[:59].rstrip() + "…"

        return {
            "schema": "2.0",
            "config": {
                "compact_width": False,
                "wide_screen_mode": True,
                "summary": {"content": summary_content},
            },
            "header": {
                "title": {"tag": "plain_text", "content": title_content},
                "subtitle": {"tag": "plain_text", "content": subtitle_content},
                "template": template,
            },
            "body": {"elements": elements},
        }


class WorkflowProgressRenderer:
    """Renders workflow execution state into Feishu card-compatible JSON.

    Read-only: all state mutations happen through WorkflowStateManager.
    This class only reads the WorkflowProject to produce card elements.
    """

    def __init__(self, project: WorkflowProject) -> None:
        self._project = project
        self._start_time: float = project.started_at or time.time()
        self._render_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock

    # ------------------------------------------------------------------
    # Rendering — produce Feishu card elements
    # ------------------------------------------------------------------

    def render_progress_card(self, project: WorkflowProject | None = None) -> dict[str, Any]:
        """Generate the full Feishu card JSON structure.

        Args:
            project: Optional snapshot to render; falls back to self._project.
                Callers under concurrent mutation MUST pass a snapshot() for safety.
        """
        with self._render_lock:
            if project is not None:
                saved = self._project
                self._project = project
                try:
                    return self._render_progress_card_impl()
                finally:
                    self._project = saved
            return self._render_progress_card_impl()

    def render_progress_cards(self, project: WorkflowProject | None = None) -> list[dict[str, Any]]:
        """Render status and complete direct-call streams while running.

        Result ledgers are terminal-only. Feishu messages cannot be inserted
        before an already-created ledger, so publishing one after the first
        completed call would force every later direct-call card below it.
        """
        with self._render_lock:
            target = project or self._project
            if project is not None:
                saved = self._project
                self._project = project
                try:
                    status_cards = self._render_progress_card_pages_impl()
                finally:
                    self._project = saved
            else:
                status_cards = self._render_progress_card_pages_impl()
            return [
                *[
                    _set_workflow_page_key(
                        card,
                        kind="status",
                        page_identity=-1,
                        local_page_index=local_page_index,
                    )
                    for local_page_index, card in enumerate(status_cards)
                ],
                *_render_agent_execution_cards(target),
            ]

    def _render_progress_card_impl(self) -> dict[str, Any]:
        return self._render_progress_card_pages_impl()[0]

    def _render_progress_card_pages_impl(self) -> list[dict[str, Any]]:
        elements: list[dict[str, Any]] = []

        # -- Current execution summary section (top) --
        summary = self._render_summary_section()
        if summary is not None:
            elements.append(summary)
            elements.append(_hr_element())

        agent_pool = self._render_agent_pool_section()
        if agent_pool is not None:
            elements.append(agent_pool)
            elements.append(_hr_element())

        # -- Progress bar section --
        elements.append(self._render_progress_bar_section())
        elements.append(_hr_element())

        # -- Phase tree --
        for idx, phase in enumerate(self._project.phases):
            elements.extend(self._render_phase_section(idx, phase))

        image_elements = _render_project_image_elements(self._project)
        if image_elements:
            elements.append(_hr_element())
            elements.extend(image_elements)

        # -- Token usage section (informational, no budget limit) --
        elements.append(_hr_element())
        elements.append(self._render_token_usage_section())

        # -- Metrics footer --
        elements.append(_hr_element())
        elements.append(self._render_metrics_footer())

        # Defensive check: ensure no accidental agent-output sentinel leaks
        _card_text_for_agent_output(elements, _AGENT_OUTPUT_FORBIDDEN_MARKERS)

        return _paginate_progress_cards(self._render_header(), elements)

    def _render_agent_pool_section(self) -> dict[str, Any] | None:
        """Render the immutable A-id bindings and script-declared plan."""
        pending = self._project.pending
        raw_pool: list[Any] = []
        orchestrator_agent_id: str | None = None
        orchestrator_was_auto = False
        raw_plan: list[Any] = []

        if pending is not None:
            raw_pool = list(pending.agent_pool)
            orchestrator_agent_id = pending.orchestrator_agent_id
            orchestrator_was_auto = pending.orchestrator_was_auto
            pending_meta = pending.meta or {}
            raw_plan = list(
                pending_meta.get("agentPlan") or pending_meta.get("agent_plan") or []
            )
        elif self._project.run_spec is not None:
            run_spec = self._project.run_spec
            if isinstance(run_spec, dict):
                raw_pool = list(run_spec.get("agent_pool") or [])
                orchestrator_agent_id = run_spec.get("orchestrator_agent_id")
            else:
                raw_pool = list(run_spec.agent_pool)
                orchestrator_agent_id = run_spec.orchestrator_agent_id

        if not raw_plan and self._project.meta is not None:
            raw_plan = list(self._project.meta.agent_plan)

        if not raw_pool and not raw_plan:
            return None

        lines = ["**Agent pool**"]
        if orchestrator_agent_id:
            raw_orchestrator_label = (
                f"Auto → {orchestrator_agent_id}"
                if orchestrator_was_auto
                else orchestrator_agent_id
            )
            orchestrator_label = _safe_dynamic_markdown(
                raw_orchestrator_label,
                one_line=True,
            )
            lines.append(f"- 主编排: {orchestrator_label}")
        for binding in raw_pool:
            raw_agent_id = str(
                _binding_value(binding, "agentId", "agent_id", default="?")
            )
            agent_id = _safe_dynamic_markdown(raw_agent_id, one_line=True)
            suffix = (
                " (orchestrator)"
                if raw_agent_id == orchestrator_agent_id
                else ""
            )
            lines.append(
                f"- `{agent_id}` · {_binding_config_markdown(binding)}{suffix}"
            )

        if raw_plan:
            lines.append("\n**Agent plan**")
            for raw_node in raw_plan:
                if not isinstance(raw_node, dict):
                    raw_node = raw_node.model_dump(by_alias=True)
                raw_node_id = (
                    raw_node.get("nodeId")
                    or raw_node.get("node_id")
                    or raw_node.get("node")
                    or "?"
                )
                node_id = _safe_dynamic_markdown(raw_node_id, one_line=True)
                role = _safe_dynamic_markdown(
                    raw_node.get("role") or "worker",
                    one_line=True,
                )
                static_id = raw_node.get("agentId") or raw_node.get("agent_id")
                candidates = raw_node.get("candidateAgentIds") or raw_node.get(
                    "candidate_agent_ids"
                )
                if raw_node.get("runtime") is True:
                    raw_candidates = (
                        candidates
                        if isinstance(candidates, Sequence)
                        and not isinstance(candidates, (str, bytes, bytearray))
                        else [candidates] if candidates else []
                    )
                    safe_candidates = ", ".join(
                        _safe_dynamic_markdown(item, one_line=True)
                        for item in raw_candidates
                    )
                    binding = f"运行时分配 → 候选 [{safe_candidates}]"
                else:
                    binding = (
                        "静态分配 → "
                        + _safe_dynamic_markdown(static_id, one_line=True)
                        if static_id
                        else "未绑定"
                    )
                lines.append(f"- `{node_id}` · {role} · {binding}")

        return _md_element("\n".join(lines))

    def _render_summary_section(self) -> dict[str, Any] | None:
        """Render a compact current/latest activity summary block."""
        # Find a running agent first; fall back to the most-recently changed
        # agent across all phases.
        running_agent: Any = None
        running_phase: PhaseProgress | None = None
        running_changed_at: float | None = None
        latest_agent: Any = None
        latest_agent_phase: PhaseProgress | None = None
        latest_changed_at: float | None = None
        latest_phase_only: PhaseProgress | None = None
        latest_phase_changed_at: float | None = None

        for phase in self._project.phases:
            phase_changed_at = getattr(phase, "finished_at", None) or getattr(phase, "started_at", None) or 0.0
            if latest_phase_changed_at is None or phase_changed_at > latest_phase_changed_at:
                latest_phase_only = phase
                latest_phase_changed_at = phase_changed_at
            for agent in phase.agents:
                activity_changed_at = (
                    getattr(agent, "activity_updated_at", None)
                    or getattr(agent, "started_at", None)
                    or 0.0
                )
                if agent.status == AgentStatus.RUNNING and (
                    running_changed_at is None or activity_changed_at >= running_changed_at
                ):
                    running_agent = agent
                    running_phase = phase
                    running_changed_at = activity_changed_at
                # Track most recently changed agent
                changed_at = (
                    getattr(agent, "finished_at", None)
                    or getattr(agent, "activity_updated_at", None)
                    or getattr(agent, "started_at", None)
                    or 0.0
                )
                if latest_changed_at is None or changed_at > latest_changed_at:
                    latest_agent = agent
                    latest_agent_phase = phase
                    latest_changed_at = changed_at

        active_agent = running_agent or latest_agent
        latest_phase = running_phase if running_agent is not None else latest_agent_phase
        if running_agent is not None:
            latest_changed_at = running_changed_at
        if active_agent is None and latest_phase is None:
            latest_phase = latest_phase_only
            latest_changed_at = latest_phase_changed_at

        requirement = str(self._project.requirement or "").strip()
        if active_agent is None and latest_phase is None and not requirement:
            return None

        # Compose the summary lines
        lines: list[str] = []
        if requirement:
            lines.append(
                "🎯 **任务:** "
                + _safe_dynamic_markdown(
                    requirement,
                    limit=240,
                    one_line=True,
                )
            )
        metrics = self._project.metrics
        total_phases = len(self._project.phases)
        completed_phases = min(
            total_phases,
            max(
                int(getattr(metrics, "phases_completed", 0) or 0),
                sum(1 for phase in self._project.phases if phase.finished_at is not None),
            ),
        )
        start = self._project.started_at or self._start_time
        end = self._project.finished_at or time.time()
        elapsed = max(0.0, end - start)
        lines.append(
            f"📊 **总览:** Phase {completed_phases}/{total_phases} · "
            f"Token {_format_tokens(metrics.total_tokens)} · 耗时 {format_elapsed_clock(elapsed)}"
        )
        terminal = self._project.status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        )
        if self._project.status == WorkflowStatus.COMPLETED:
            summary_title = "执行已完成"
        elif self._project.status == WorkflowStatus.FAILED:
            summary_title = "执行已失败"
        elif self._project.status == WorkflowStatus.CANCELLED:
            summary_title = "执行已取消"
        else:
            summary_title = "当前执行中"

        # Phase
        phase_title = latest_phase.title if latest_phase is not None else "(暂无阶段)"
        phase_title = _safe_dynamic_markdown(
            phase_title,
            limit=_LABEL_TRUNCATION_LIMIT,
            one_line=True,
        )
        phase_idx = self._project.phases.index(latest_phase) + 1 if latest_phase in self._project.phases else "—"
        lines.append(f"📌 **当前阶段:** 阶段 {phase_idx} · {phase_title}")

        # Active agent
        agent_label_key = "最近代理" if terminal else "当前代理"
        tool_label_key = "使用工具" if terminal else "正在使用"
        if active_agent is not None:
            agent_label = _safe_dynamic_markdown(
                active_agent.label or "agent",
                limit=_LABEL_TRUNCATION_LIMIT,
                one_line=True,
            )
            raw_aid = str(getattr(active_agent, "agent_id", "") or "").strip()
            aid_prefix = (
                f"`{_safe_dynamic_markdown(raw_aid, one_line=True)}` · "
                if raw_aid and raw_aid not in str(active_agent.label or "")
                else ""
            )
            role_md = _binding_role_markdown(active_agent)
            role_suffix = f" · {role_md}" if role_md else ""
            agent_status_icon = STATUS_ICONS.get(active_agent.status, "⏳")
            lines.append(
                f"🤖 **{agent_label_key}:** {agent_status_icon} {aid_prefix}{agent_label}{role_suffix}"
            )
            if active_agent.tool:
                binding = _binding_config_markdown(
                    active_agent,
                    include_role=False,
                    default_tool="未指定工具",
                    default_model="默认模型",
                )
                lines.append(f"🛠 **{tool_label_key}:** {binding}")
            else:
                _, _, prof, eff = _binding_model_parts(
                    active_agent,
                    default_tool="",
                    default_model="",
                )
                extras = []
                if prof:
                    extras.append(f"Profile `{_safe_dynamic_markdown(prof, one_line=True)}`")
                if eff:
                    extras.append(f"Effort `{_safe_dynamic_markdown(eff, one_line=True)}`")
                extra_text = (" · " + " · ".join(extras)) if extras else ""
                lines.append(f"🛠 **{tool_label_key}:** (未指定工具){extra_text}")
            if active_agent.task_summary:
                task = _safe_dynamic_markdown(
                    active_agent.task_summary,
                    limit=60,
                    one_line=True,
                )
                lines.append(f"📋 **当前任务:** {task}")
            attempt = max(1, int(getattr(active_agent, "attempt", 1) or 1))
            lines.append(f"🔁 **Attempt:** {attempt}")
            activity = getattr(active_agent, "current_activity", "") or ""
            if activity and not terminal and active_agent.status == AgentStatus.RUNNING:
                safe_activity = _safe_dynamic_markdown(
                    activity,
                    limit=60,
                    one_line=True,
                )
                lines.append(f"⚡ **正在:** {safe_activity}")
        else:
            lines.append(f"🤖 **{agent_label_key}:** (无代理调用)")
            lines.append(f"🛠 **{tool_label_key}:** —")

        # Last-change time
        if latest_changed_at and latest_changed_at > 0:
            from datetime import datetime  # local import — keeps module import lean

            try:
                ts = datetime.fromtimestamp(latest_changed_at).strftime("%H:%M:%S")
            except (OSError, OverflowError, ValueError):
                ts = "—"
            lines.append(f"🕒 **最近变更:** {ts}")
        else:
            lines.append("🕒 **最近变更:** —")

        # Live elapsed for a genuinely running agent — mirrors the per-agent
        # elapsed suffix so the top summary keeps advancing during a long
        # blocking agent() call (only meaningful while non-terminal).
        if (
            not terminal
            and active_agent is not None
            and active_agent.status == AgentStatus.RUNNING
            and active_agent.started_at
        ):
            elapsed = time.time() - active_agent.started_at
            if elapsed > 0:
                lines.append(f"⏱ **已运行:** {format_duration(elapsed)}")

        return _md_element(f"**{summary_title}**\n" + "\n".join(lines))

    def render_compact_status(self) -> str:
        """One-line text summary of workflow status.

        Example: "任务: code-audit | 阶段 2/3 | 7/12 代理 完成 | 450K tokens 消耗"
        """
        name = _safe_dynamic_text(self._project.name or "workflow", one_line=True)
        total_phases = len(self._project.phases)
        current_phase = self._current_phase_index() + 1

        metrics = self._project.metrics
        completed = metrics.completed_agents
        total = metrics.total_agents

        tokens = _format_tokens(metrics.total_tokens if hasattr(metrics, "total_tokens") else 0)

        status_icon = WORKFLOW_STATUS_ICONS.get(self._project.status, "\u23f3")

        return (
            f"任务: {name} | 阶段 {current_phase}/{total_phases} | "
            f"{completed}/{total} 代理 {status_icon} | {tokens} tokens 消耗"
        )

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _render_header(self) -> dict[str, Any]:
        """Render card header with workflow name + status."""
        status = self._project.status
        icon = WORKFLOW_STATUS_ICONS.get(status, "\u23f3")

        # Map status to header template color
        if status == WorkflowStatus.COMPLETED:
            template = "green"
        elif status == WorkflowStatus.FAILED:
            template = "red"
        elif status == WorkflowStatus.RUNNING:
            template = "blue"
        else:
            template = "grey"

        title = (
            f"{icon} "
            f"{_safe_dynamic_text(self._project.name or 'Workflow', one_line=True)}"
        )

        return {
            "title": {"tag": "plain_text", "content": title},
            "template": template,
        }

    def _render_progress_bar_section(self) -> dict[str, Any]:
        """Render overall progress as compact "进度 M/N · Z%" line + bar."""
        metrics = self._project.metrics
        completed = metrics.completed_agents
        total = max(metrics.total_agents, 1)
        ratio = completed / total
        pct = _pct(completed, total)
        bar = _unicode_progress_bar(ratio)
        return _md_element(f"进度 {completed}/{total} · {pct}\n{bar}")

    def _render_phase_section(self, idx: int, phase: PhaseProgress) -> list[dict[str, Any]]:
        """Render a phase with agents grouped by status into collapsible panels."""
        elements: list[dict[str, Any]] = []

        agents = phase.agents
        total_agents = len(agents)
        completed_count = sum(1 for a in agents if a.status in (AgentStatus.DONE, AgentStatus.CACHED))
        phase_tokens = sum(max(0, int(getattr(a, "token_usage", 0) or 0)) for a in agents)

        # Phase header — row 1: title (middle-ellipsis); row 2: completion count + duration
        phase_status = self._get_phase_status_icon(phase)
        phase_title = _safe_dynamic_markdown(
            phase.title,
            limit=_LABEL_TRUNCATION_LIMIT,
            one_line=True,
        )
        elements.append(
            _md_element(f"**{phase_status} 阶段 {idx + 1}: {phase_title}**")
        )

        if phase.started_at and total_agents > 0:
            elapsed = (phase.finished_at or time.time()) - phase.started_at
            duration_text = format_duration(elapsed)
            elements.append(
                _md_element(
                    f"已完成 {completed_count}/{total_agents} · "
                    f"Token {_format_tokens(phase_tokens)} · 耗时 {duration_text}"
                )
            )
        elif total_agents > 0:
            elements.append(
                _md_element(
                    f"已完成 {completed_count}/{total_agents} · Token {_format_tokens(phase_tokens)}"
                )
            )
        elif phase.finished_at:
            if phase.started_at:
                elapsed = phase.finished_at - phase.started_at
                elements.append(_md_element(f"已完成 0/0 · 耗时 {format_duration(elapsed)}"))
            else:
                elements.append(_md_element("已完成 0/0"))
        elif phase.started_at:
            elements.append(_md_element("进行中 0/0"))
        else:
            elements.append(_md_element("等待中"))

        if not agents:
            return elements

        # Group agents by status buckets
        buckets: dict[str, list[AgentProgress]] = {
            "RUNNING": [],
            "FAILED": [],
            "DONE": [],
            "CACHED": [],
            "CANCELLED": [],
            "PENDING": [],
        }
        for agent in agents:
            raw = agent.status.value if hasattr(agent.status, "value") else str(agent.status)
            key = raw.upper()
            if key in buckets:
                buckets[key].append(agent)
            else:
                buckets["PENDING"].append(agent)

        # Status → label + color mapping for collapsible_panel headers
        status_meta: dict[str, tuple[str, str]] = {
            "RUNNING": ("执行中", "blue"),
            "FAILED": ("失败", "red"),
            "DONE": ("已完成", "green"),
            "CACHED": ("缓存", "turquoise"),
            "CANCELLED": ("已取消", "grey"),
            "PENDING": ("待执行", "grey"),
        }
        terminal_operations = {
            AgentStatus.DONE: "已完成，完整输出见报告",
            AgentStatus.CACHED: "已命中缓存，完整输出见报告",
            AgentStatus.FAILED: "执行失败，错误与已有输出见报告",
            AgentStatus.CANCELLED: "已取消",
            AgentStatus.PENDING: "等待调度",
        }

        # Render status groups as collapsible panels (RUNNING/FAILED expanded, rest collapsed)
        display_order = [
            ("RUNNING", True),
            ("FAILED", True),
            ("DONE", False),
            ("CACHED", False),
            ("CANCELLED", False),
            ("PENDING", False),
        ]
        display_groups = [
            (key, expanded, offset, buckets[key][offset : offset + 8])
            for key, expanded in display_order
            for offset in range(0, len(buckets[key]), 8)
        ]
        for key, expanded, group_offset, group in display_groups:
            label, color = status_meta[key]
            lines: list[str] = []
            for agent in group:
                if agent.tool:
                    tool_badge = _binding_config_markdown(
                        agent,
                        include_role=False,
                        default_tool="未指定工具",
                        default_model="默认模型",
                    )
                else:
                    _, _, prof, eff = _binding_model_parts(
                        agent,
                        default_tool="未指定工具",
                        default_model="默认模型",
                    )
                    badge_parts = ["`未指定工具`"]
                    if prof:
                        badge_parts.append(f"Profile `{_safe_dynamic_markdown(prof, one_line=True)}`")
                    if eff:
                        badge_parts.append(f"Effort `{_safe_dynamic_markdown(eff, one_line=True)}`")
                    tool_badge = " · ".join(badge_parts)
                display_label = _safe_dynamic_markdown(
                    agent.label or "agent",
                    limit=_LABEL_TRUNCATION_LIMIT,
                    one_line=True,
                )
                raw_aid = str(getattr(agent, "agent_id", "") or "").strip()
                if raw_aid and raw_aid not in str(agent.label or ""):
                    id_and_label = f"`{_safe_dynamic_markdown(raw_aid, one_line=True)}` · {display_label}"
                else:
                    id_and_label = display_label
                role_md = _binding_role_markdown(agent)
                role_suffix = f" · {role_md}" if role_md else ""
                display_index = max(1, int(getattr(agent, "call_index", 0) or 0) + 1)
                attempt = max(1, int(getattr(agent, "attempt", 1) or 1))
                status_text = status_meta[key][0]
                task = " ".join((agent.task_summary or "未提供任务摘要").split())
                if agent.status == AgentStatus.RUNNING:
                    operation = " ".join(
                        str(getattr(agent, "current_activity", "") or "等待 Agent 返回").split()
                    )
                    elapsed_suffix = ""
                    if agent.started_at:
                        elapsed = time.time() - agent.started_at
                        if elapsed > 0:
                            elapsed_suffix = f" · 已运行 {format_duration(elapsed)}"
                else:
                    operation = terminal_operations.get(agent.status, "等待调度")
                    elapsed_suffix = (
                        f" · 耗时 {format_duration(agent.duration_s)}"
                        if agent.duration_s > 0
                        else ""
                    )
                row = (
                    f"{STATUS_ICONS.get(agent.status, '·')} #{display_index} {id_and_label}{role_suffix} · "
                    f"{status_text} · Attempt {attempt} · {tool_badge}{elapsed_suffix}\n"
                    "任务："
                    f"{_safe_dynamic_markdown(task, limit=100, one_line=True)}\n"
                    "当前操作："
                    f"{_safe_dynamic_markdown(operation, limit=120, one_line=True)}"
                )
                row += _render_subagent_lines(agent.subagents)
                lines.append(row)
            header_obj: dict[str, Any] = {
                "title": {
                    "tag": "plain_text",
                    "content": (
                        f"{label} ({group_offset + 1}-{group_offset + len(group)}"
                        f"/{len(buckets[key])})"
                        if len(buckets[key]) > len(group)
                        else f"{label} ({len(group)})"
                    ),
                },
            }
            panel = _collapsible_panel(
                header_obj,
                [_md_element("\n".join(lines))],
                expanded=expanded,
                template=color,
            )
            elements.append(panel)

        return elements

    def _render_token_usage_section(self) -> dict[str, Any]:
        """Render token consumption as compact informational line (no budget limit)."""
        metrics = self._project.metrics
        total_tokens = metrics.total_tokens if hasattr(metrics, "total_tokens") else 0
        used_str = _format_tokens(total_tokens)
        return _md_element(f"Token 消耗: {used_str}")

    def _render_metrics_footer(self) -> dict[str, Any]:
        """Render metrics footer as a 2-column stretch layout: Agents/耗时 · 缓存/失败。"""
        metrics = self._project.metrics
        start = self._project.started_at or self._start_time
        end = self._project.finished_at or time.time()
        elapsed = end - start if end >= start else time.time() - self._start_time
        elapsed_str = format_elapsed_clock(elapsed)

        # Left column: Agents + 耗时
        left_content = [
            f"**代理:** {metrics.completed_agents}/{metrics.total_agents}",
            f"**耗时:** {elapsed_str}",
        ]
        # Right column: 缓存 + 失败
        right_content = []
        if metrics.cached_agents > 0:
            right_content.append(f"**缓存:** {metrics.cached_agents}")
        if metrics.failed_agents > 0:
            right_content.append(f"**失败:** {metrics.failed_agents}")
        if not right_content:
            right_content.append("**缓存:** 0")

        return _column_set(
            [
                _column(
                    [_md_element("\n".join(left_content), text_align="center")],
                    weight=1,
                    vertical_align="center",
                ),
                _column(
                    [_md_element("\n".join(right_content), text_align="center")],
                    weight=1,
                    vertical_align="center",
                ),
            ],
            flex_mode="stretch",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_phase_index(self) -> int:
        """Return index of the current (last) phase, or 0 if none."""
        if not self._project.phases:
            return 0
        return len(self._project.phases) - 1

    def _get_phase_status_icon(self, phase: PhaseProgress) -> str:
        """Determine the overall status icon for a phase."""
        if not phase.agents:
            if phase.finished_at:
                return "\u2705"
            return "\u23f3"

        has_running = any(a.status == AgentStatus.RUNNING for a in phase.agents)
        has_failed = any(a.status == AgentStatus.FAILED for a in phase.agents)
        all_done = all(a.status in (AgentStatus.DONE, AgentStatus.CACHED, AgentStatus.CANCELLED) for a in phase.agents)

        if has_running:
            return "\U0001f504"
        if all_done:
            return "\u2705"
        if has_failed:
            return "\u274c"
        return "\u23f3"
