"""Spec Engine 进度报告器 — 格式化进度信息供 Feishu 卡片展示。"""

import re

from ..engine_base import PerspectiveReview
from ..utils.text import format_duration, make_progress_bar
from .models import (
    ReviewResult,
    SpecCycle,
    SpecPhase,
    SpecProject,
    SpecProjectStatus,
    SpecTaskStatus,
    SpecWorkItemStatus,
)


class SpecReporter:
    """格式化 Spec Engine 的进度消息，用于 Feishu 卡片展示。"""

    # ------------------------------------------------------------------
    # Content formatters
    # ------------------------------------------------------------------

    def format_analyzing_start(self, requirement: str) -> str:
        return f"""📋 **Spec Agent 启动**

📝 正在分析项目需求...
> {requirement}

⏳ 请稍候，正在提取验收标准..."""

    def format_analyzing_done(self, project: SpecProject) -> str:
        if not project.acceptance_criteria:
            return "❌ 需求分析失败"

        lines = [
            "✅ **需求分析完成**\n",
            f"📂 项目: {project.name}",
            f"🎯 需求: {project.requirement[:100]}{'...' if len(project.requirement) > 100 else ''}",
            f"📊 验收标准: {project.total_criteria} 条\n",
            "**验收标准:**",
        ]

        for i, criterion in enumerate(project.acceptance_criteria):
            lines.append(f"  🔲 {i + 1}. {criterion}")

        lines.append("\n🚀 准备开始 Spec 循环...")
        return "\n".join(lines)

    def format_phase_progress(self, current_phase: SpecPhase, completed: bool = False) -> str:
        phases = list(SpecPhase)
        parts: list[str] = []
        current_idx = phases.index(current_phase) if current_phase in phases else -1
        for i, p in enumerate(phases):
            if i < current_idx:
                parts.append(f"✅{p.display_name}")
            elif i == current_idx:
                if completed:
                    parts.append(f"✅{p.display_name}")
                else:
                    parts.append(f"▶️**{p.display_name}**")
            else:
                parts.append(f"⬜{p.display_name}")
        return " → ".join(parts)

    def format_phase_subtitle(self, cycle: int, max_cycles: int, current_phase: SpecPhase, completed: bool = False) -> str:
        """Compact phase progress for header subtitle.

        Example: "Cycle 1/3 · Spec ✅ Plan ✅ Task ✅ Build ▶️ Review ⬜"
        """
        phases = list(SpecPhase)
        current_idx = phases.index(current_phase) if current_phase in phases else -1
        parts: list[str] = []
        for i, p in enumerate(phases):
            short_name = p.value.capitalize()
            if i < current_idx:
                parts.append(f"{short_name} ✅")
            elif i == current_idx:
                parts.append(f"{short_name} {'✅' if completed else '▶️'}")
            else:
                parts.append(f"{short_name} ⬜")
        phase_text = " ".join(parts)
        if max_cycles > 1:
            return f"Cycle {cycle}/{max_cycles} · {phase_text}"
        return phase_text

    def format_phase_start_content(self, cycle: int, phase: SpecPhase, max_cycles: int) -> str:
        progress = self.format_phase_progress(phase, completed=False)
        return f"{progress}\n\n{phase.emoji} **{phase.display_name}** 执行中..."

    def format_phase_done_content(self, cycle: int, phase: SpecPhase, max_cycles: int, output: str) -> str:
        progress = self.format_phase_progress(phase, completed=True)
        summary = self._extract_phase_summary(phase, output)
        lines = [progress, ""]
        if summary:
            lines.append(f"{phase.emoji} **{phase.display_name}完成**  {summary}")
        else:
            lines.append(f"{phase.emoji} **{phase.display_name}完成**")
        return "\n".join(lines)

    def _extract_phase_summary(self, phase: SpecPhase, output: str) -> str:
        if not output:
            return ""
        if phase == SpecPhase.SPEC:
            goals_count = output.count('"goals"')
            criteria_count = output.count('"acceptance_criteria"')
            if goals_count or criteria_count:
                return "规格产物已生成"
            return ""
        elif phase == SpecPhase.PLAN:
            steps_match = re.findall(r'"steps"\s*:\s*\[', output)
            if steps_match:
                return "方案已规划"
            return ""
        elif phase == SpecPhase.TASK:
            task_lines = [line for line in output.split("\n") if re.match(r"^\s*\d+\s*[.、)]", line)]
            if task_lines:
                return f"共 {len(task_lines)} 个任务"
            return ""
        elif phase == SpecPhase.BUILD:
            line_count = len([line for line in output.split("\n") if line.strip()])
            return f"构建输出 {line_count} 行"
        return ""

    def _format_cycle_phase_details(self, cycle: SpecCycle) -> str:
        parts: list[str] = []

        if cycle.spec_artifact:
            sa = cycle.spec_artifact
            goals_count = len(sa.goals)
            ac_count = len(sa.acceptance_criteria)
            nfr_count = len(sa.non_functional_requirements)
            detail_parts = []
            if goals_count:
                detail_parts.append(f"{goals_count} 个目标")
            if ac_count:
                detail_parts.append(f"{ac_count} 条验收标准")
            if nfr_count:
                detail_parts.append(f"{nfr_count} 条非功能需求")
            parts.append(f"📋 **规格定义**: {', '.join(detail_parts) if detail_parts else '已生成'}")
        elif cycle.spec_content:
            parts.append("📋 **规格定义**: 已生成")

        if cycle.plan_artifact:
            pa = cycle.plan_artifact
            detail_parts = []
            if pa.steps:
                detail_parts.append(f"{len(pa.steps)} 个步骤")
            if pa.file_changes:
                detail_parts.append(f"{len(pa.file_changes)} 处文件变更")
            if pa.architecture:
                arch_brief = pa.architecture[:50] + "..." if len(pa.architecture) > 50 else pa.architecture
                detail_parts.append(arch_brief)
            parts.append(f"🏗️ **方案规划**: {', '.join(detail_parts) if detail_parts else '已规划'}")
        elif cycle.plan_content:
            parts.append("🏗️ **方案规划**: 已规划")

        if cycle.tasks:
            completed = sum(1 for t in cycle.tasks if t.status == SpecTaskStatus.COMPLETED)
            total = len(cycle.tasks)
            parts.append(f"📝 **任务分解**: {completed}/{total} 完成")
            for task in cycle.tasks:
                parts.append(f"  - 任务 {task.task_id}: {task.description}")
        elif cycle.tasks_total:
            parts.append(f"📝 **任务分解**: {cycle.tasks_total} 个任务")

        if cycle.build_output:
            line_count = len([line for line in cycle.build_output.split("\n") if line.strip()])
            parts.append(f"🔨 **执行构建**: 输出 {line_count} 行")

        if cycle.review_result:
            r = cycle.review_result
            passed = sum(1 for pr in r.reviews if pr.passed)
            total = len(r.reviews)
            parts.append(f"🔍 **多角色审查**: {passed}/{total} 角色通过")

        return "\n".join(parts)

    def format_cycle_start(self, cycle_number: int, max_cycles: int, criteria_status: str = "") -> str:
        progress = self.format_phase_progress(SpecPhase.SPEC, completed=False)
        base = progress
        if criteria_status:
            return f"{base}\n\n{criteria_status}"
        return base

    def format_phase_start(self, cycle: int, phase: SpecPhase) -> str:
        return f"{phase.emoji} **{phase.display_name}** [循环 {cycle}]\n\n⏳ 执行中..."

    def format_phase_done(self, cycle: int, phase: SpecPhase, content: str) -> str:
        body = content if content else "(无输出)"
        return f"""{phase.emoji} **{phase.display_name}完成** [循环 {cycle}]

```
{body}
```"""

    def format_cycle_done(self, cycle_number: int, cycle: SpecCycle) -> str:
        status_emoji = "✅" if cycle.status == "completed" else "❌"
        status_text = "完成" if cycle.status == "completed" else "失败"

        progress = self.format_phase_progress(SpecPhase.REVIEW, completed=True)

        summary = "循环执行结束"
        if cycle.review_result:
            if cycle.review_result.all_passed:
                summary = "所有视角审查通过 ✅"
            else:
                summary = f"审查发现 {cycle.review_result.total_suggestions} 条改进建议"

        duration_str = ""
        if hasattr(cycle, "duration") and cycle.duration:
            duration_str = f"\n⏱️ 耗时: {format_duration(cycle.duration)}"

        lines = [
            f"{status_emoji} **Spec 循环 [{cycle_number}] {status_text}**\n",
            progress,
            f"\n🤖 **{summary}**{duration_str}",
        ]

        phase_details = self._format_cycle_phase_details(cycle)
        if phase_details:
            lines.append("\n---\n**📦 各阶段产出：**\n")
            lines.append(phase_details)

        if cycle.review_result and not cycle.review_result.all_passed:
            lines.append("\n---\n**📋 多角色审查建议（将驱动下一轮优化）：**\n")
            for pr in cycle.review_result.reviews:
                if not pr.passed and pr.suggestions:
                    lines.append(f"\n{pr.perspective.emoji} **{pr.perspective.display_name}**:\n")
                    for s in pr.suggestions:
                        lines.append(f"- {s}")
            lines.append(f"\n💡 共 {cycle.review_result.total_suggestions} 条建议 → 驱动下一轮 Spec 循环")

        return "\n".join(lines)

    def format_review_result(self, review: ReviewResult, cycle: int) -> str:
        lines = [f"🔍 **多角色审查 [循环 {cycle}]**\n"]

        count = 0
        total_reviews = len(review.reviews)

        for pr in review.reviews:
            count += 1
            title = pr.role_display_name or pr.perspective.display_name
            agent_detail = self._format_review_agent_detail(pr)
            title_with_agent = f"{title}（{agent_detail}）" if agent_detail else title
            if pr.passed:
                lines.append(f"{pr.perspective.emoji} **{title_with_agent}**: ✅ PASS")
            else:
                status_text = pr.perspective.failure_label
                lines.append(f"{pr.perspective.emoji} **{title_with_agent}**: {status_text}")
                for s in pr.suggestions:
                    lines.append(f"- {s}")

            if count < total_reviews:
                lines.append("\n---")
            else:
                lines.append("")

        total = review.total_suggestions
        if total > 0:
            lines.append(f"💡 **改进建议: {total} 条** → 将驱动下一轮 Spec 循环")
        else:
            lines.append("✅ **所有视角均通过，无改进建议**")

        return "\n".join(lines)

    def _format_review_agent_detail(self, review: PerspectiveReview) -> str:
        """Return compact tool/model label for a role review title."""
        label = str(getattr(review, "review_agent_label", "") or "").strip()
        if label:
            return label
        agent = str(getattr(review, "review_agent_type", "") or "").strip()
        model = str(getattr(review, "review_model_name", "") or "").strip()
        if not agent and not model:
            return ""
        if not agent:
            return model
        if not model:
            return agent
        return f"{agent} / {model}"

    def format_criteria_brief(self, project: SpecProject) -> str:
        tracker = project.criteria_tracker
        if not tracker.criteria:
            return ""
        lines = [f"📋 **验收标准 ({tracker.satisfied_count}/{tracker.total_count})**"]
        for i, criterion in enumerate(tracker.criteria):
            satisfied = tracker.satisfied.get(i, False)
            marker = "✅" if satisfied else "🔲"
            lines.append(f"  {marker} {criterion}")
        return "\n".join(lines)

    def format_criteria_update(self, project: SpecProject) -> str:
        lines = ["📋 **验收标准进度**\n"]
        tracker = project.criteria_tracker

        for i, criterion in enumerate(tracker.criteria):
            satisfied = tracker.satisfied.get(i, False)
            if satisfied:
                cycle_num = tracker.satisfied_at_iteration.get(i, "?")
                lines.append(f"  ✅ {i + 1}. {criterion} (循环{cycle_num})")
            else:
                lines.append(f"  🔲 {i + 1}. {criterion}")

        progress_bar = self._make_progress_bar(tracker.satisfied_count, tracker.total_count)
        lines.append(f"\n{progress_bar}")

        return "\n".join(lines)

    def format_project_done(self, project: SpecProject) -> str:
        if project.status == SpecProjectStatus.COMPLETED:
            lines = [
                "🎉 **Spec 模式完成！**\n",
                f"📂 项目: {project.name}",
                f"🎯 需求: {project.requirement[:80]}{'...' if len(project.requirement) > 80 else ''}",
                f"🔁 总循环: {project.current_cycle_number} 轮",
                f"✅ 验收标准: {project.satisfied_count}/{project.total_criteria} 全部满足",
            ]
        elif project.status == SpecProjectStatus.FAILED:
            lines = [
                "⚠️ **Spec 模式终止**\n",
                f"📂 项目: {project.name}",
                f"📝 原因: {project.error or '未知'}",
                f"🔁 总循环: {project.current_cycle_number} 轮",
                f"📊 验收标准: {project.satisfied_count}/{project.total_criteria} 满足",
            ]

            # 固定首屏：未满足标准 + Top 建议
            tracker = project.criteria_tracker
            unsatisfied = tracker.unsatisfied_criteria
            if unsatisfied:
                lines.append("\n**未满足的验收标准（Top 8）:**")
                for c in unsatisfied[:8]:
                    lines.append(f"- [ ] {c}")

            last_cycle = project.current_cycle
            if last_cycle and last_cycle.review_result and not last_cycle.review_result.all_passed:
                pending: list[str] = []
                for pr in last_cycle.review_result.failed_perspectives:
                    for s in pr.suggestions:
                        if s:
                            pending.append(str(s))
                if pending:
                    lines.append("\n**待解决建议（Top 5）:**")
                    for s in pending[:5]:
                        lines.append(f"- {s}")

            # Actionable next steps
            tips: list[str] = []
            err = project.error or ""
            if "收敛终止" in err:
                tips.extend(
                    [
                        "使用 `/spec_guide <补充信息>` 注入更多上下文/约束，帮助继续推进",
                        "补充缺失的需求澄清点（尤其是 Spec 里出现 NEEDS CLARIFICATION 的内容）",
                        "必要时提高 `spec_max_cycles` 或暂时关闭 `spec_review_enabled` 以定位阻塞",
                    ]
                )
            elif "最大循环" in err:
                tips.extend(
                    [
                        "使用 `/spec_guide` 提供更明确的目标与边界，减少反复试错",
                        "提高 `spec_max_cycles` 并再次运行，或拆分需求缩小范围",
                    ]
                )
            if tips:
                lines.append("\n**下一步建议：**")
                for t in tips:
                    lines.append(f"- {t}")
        elif project.status == SpecProjectStatus.CANCELLED:
            lines = [
                "⏹️ **Spec 模式已停止**\n",
                f"📂 项目: {project.name}",
                f"📊 验收标准: {project.satisfied_count}/{project.total_criteria} 满足",
            ]
        else:
            lines = [
                "🔄 **Spec 模式执行中**\n",
                f"📂 项目: {project.name}",
                f"📊 验收标准: {project.satisfied_count}/{project.total_criteria} 满足",
            ]

        if project.duration():
            lines.append(f"⏱️ 总耗时: {format_duration(project.duration())}")

        # Operation summary (tool calls, modified files)
        operation_summary = self.format_operation_summary(project)
        if operation_summary:
            lines.append(f"\n{operation_summary}")

        # Full criteria list as appendix (for scanability)
        lines.append("\n**验收标准（完整列表）:**")
        tracker = project.criteria_tracker
        for i, criterion in enumerate(tracker.criteria):
            satisfied = tracker.satisfied.get(i, False)
            marker = "✅" if satisfied else "🔲"
            lines.append(f"  {marker} {i + 1}. {criterion}")

        return "\n".join(lines)

    def format_operation_summary(self, project: SpecProject) -> str:
        """格式化操作总结：工具调用统计、修改文件列表。"""
        if not project.cycles:
            return ""

        total_tool_calls = sum(c.tool_call_count for c in project.cycles)
        all_modified: set[str] = set()
        for c in project.cycles:
            all_modified.update(c.modified_files)

        if total_tool_calls == 0 and not all_modified:
            return ""

        lines: list[str] = ["**📊 操作总结**\n"]
        if total_tool_calls:
            lines.append(f"🛠️ 工具调用: {total_tool_calls} 次")
        if all_modified:
            lines.append(f"🗂️ 涉及文件: {len(all_modified)} 个")

        # Per-cycle breakdown (last 5)
        cycles_with_stats = [c for c in project.cycles if c.tool_call_count > 0]
        if len(cycles_with_stats) > 1:
            lines.append("\n**各轮次摘要（最近 5 轮）:**")
            for c in cycles_with_stats[-5:]:
                phase_parts = []
                for p_name, p_count in (c.phase_tool_stats or {}).items():
                    if p_count > 0:
                        phase_parts.append(f"{p_name}:{p_count}")
                tool_info = f" ({', '.join(phase_parts)})" if phase_parts else ""
                status_emoji = "✅" if c.status == "completed" else "❌"
                lines.append(f"  {status_emoji} 循环 {c.cycle_number}: {c.tool_call_count} 次调用{tool_info}")

        # Modified files list (max 10)
        if all_modified:
            lines.append("\n**修改的文件:**")
            sorted_files = sorted(all_modified)
            for f in sorted_files[:10]:
                lines.append(f"  - `{f}`")
            if len(sorted_files) > 10:
                lines.append(f"  - ...及其他 {len(sorted_files) - 10} 个文件")

        return "\n".join(lines)

    def format_error(self, error: str) -> str:
        err = (error or "").strip() or "未知错误"

        # Best-effort: make key fields visible/stable for users (and acceptance checks).
        # Keep the raw error in code block for debugging.
        phase_hit = re.search(r"Phase\s+([a-zA-Z0-9_\-]+)\s+失败", err)
        task_hit = re.search(r"task_id=([a-zA-Z0-9_\-]+)", err)
        internal = "Internal error" if "internal error" in err.lower() else ""

        is_timeout = "TimeoutError" in err or "操作耗时过长" in err or "timeout" in err.lower()

        summary_parts: list[str] = []
        if is_timeout:
            summary_parts.append("⏱️ 操作超时，请检查网络或稍后重试")
        if phase_hit:
            summary_parts.append(f"Phase {phase_hit.group(1)} 失败")
        if internal:
            summary_parts.append(internal)
        if task_hit:
            summary_parts.append(f"任务ID {task_hit.group(1)}")

        summary_line = ""
        if summary_parts:
            summary_line = "\n" + ("\n".join([f"- {p}" for p in summary_parts])) + "\n"

        advice = "建议您稍后点击重试。" if is_timeout else "请检查错误信息后重试。"

        return f"""❌ **Spec Agent 错误**{summary_line}
```
{err}
```

{advice}"""

    def format_status(self, project: SpecProject) -> str:
        status_text = {
            SpecProjectStatus.IDLE: "⏳ 等待开始",
            SpecProjectStatus.ANALYZING: "🧠 正在分析需求",
            SpecProjectStatus.RUNNING: "🔄 循环执行中",
            SpecProjectStatus.COMPLETED: "✅ 已完成",
            SpecProjectStatus.FAILED: "❌ 已失败",
            SpecProjectStatus.CANCELLED: "⏹️ 已停止",
        }.get(project.status, "❓ 未知状态")

        done_items = sum(1 for w in getattr(project, "work_items", []) if w.status == SpecWorkItemStatus.DONE)
        total_items = getattr(project, "work_items_total", 0)

        lines = [
            f"📊 **{project.name}** Spec 状态\n",
            f"状态: {status_text}",
            f"循环: {project.current_cycle_number} 轮",
            f"标准: {project.satisfied_count}/{project.total_criteria} 满足",
        ]
        if total_items > 0:
            lines.append(f"单元: {done_items}/{total_items} 完成")

        if project.duration():
            lines.append(f"⏱️ 已执行: {format_duration(project.duration())}")

        # Current cycle phase
        if project.current_cycle:
            cycle = project.current_cycle
            lines.append(f"\n**当前循环 [{cycle.cycle_number}]:** {cycle.phase.emoji} {cycle.phase.display_name}")

            # Pending review suggestions (top N)
            if cycle.review_result and not cycle.review_result.all_passed:
                pending: list[str] = []
                for pr in cycle.review_result.failed_perspectives:
                    for s in pr.suggestions:
                        if s:
                            pending.append(str(s))
                if pending:
                    lines.append("\n**待解决建议（Top 5）:**")
                    for s in pending[:5]:
                        lines.append(f"  - {s}")

        # Recent cycles
        if project.cycles:
            lines.append("\n**循环历史:**")
            for cycle in project.cycles[-5:]:
                status_emoji = "✅" if cycle.status == "completed" else "❌" if cycle.status == "failed" else "🔄"
                review_info = ""
                if cycle.review_result:
                    if cycle.review_result.all_passed:
                        review_info = " (审查通过)"
                    else:
                        review_info = f" ({cycle.review_result.total_suggestions}条建议)"
                lines.append(f"  {status_emoji} 循环 {cycle.cycle_number}{review_info}")

        # Backlog (generated specs)
        if getattr(project, "work_items", None):
            pending = [w for w in project.work_items if w.status == SpecWorkItemStatus.PENDING]
            in_prog = [w for w in project.work_items if w.status == SpecWorkItemStatus.IN_PROGRESS]
            lines.append("\n**待优化单元（Spec Backlog）:**")
            lines.append(f"  待办: {len(pending)}  •  执行中: {len(in_prog)}  •  总计: {len(project.work_items)}")
            for w in (in_prog + pending)[:5]:
                q = (w.question or "").strip()
                q = q if len(q) <= 80 else q[:80] + "..."
                lines.append(f"  - {w.item_id}: {q}")

        # Criteria progress
        lines.append("\n**验收标准:**")
        tracker = project.criteria_tracker
        for i, criterion in enumerate(tracker.criteria):
            satisfied = tracker.satisfied.get(i, False)
            marker = "✅" if satisfied else "🔲"
            lines.append(f"  {marker} {criterion}")

        return "\n".join(lines)

    def format_history(self, project: SpecProject, tail: int = 20) -> str:
        tail = max(1, int(tail or 20))
        lines: list[str] = [f"🗂️ **Spec 历史** — {project.name}\n"]

        # Compaction hint
        if getattr(project, "cycle_count_total", 0) and project.cycle_count_total > len(project.cycles):
            truncated = project.cycle_count_total - len(project.cycles)
            lines.append(f"⚠️ 状态文件为压缩模式：仅保留最近 {len(project.cycles)} 轮明细（前 {truncated} 轮已截断）")
            hist = getattr(project, "history_log_path", None)
            if hist:
                lines.append(f"可查完整增量历史日志：`{hist}`\n")
            else:
                root = getattr(project, "artifacts_root", None)
                if root:
                    lines.append(f"可查完整增量历史日志：见目录 `{root}`（history log 路径未加载）\n")

        if not project.cycles:
            lines.append("(暂无循环记录)")
        else:
            lines.append("**循环（最近）:**")
            for c in project.cycles[-tail:]:
                review = ""
                if c.review_result:
                    review = "PASS" if c.review_result.all_passed else f"{c.review_result.total_suggestions}条建议"
                lines.append(
                    f"- 循环 {c.cycle_number}: {c.status}"
                    f"  • spec={bool(c.spec_path)} plan={bool(c.plan_path)} tasks={bool(c.tasks_path)} build={bool(c.build_path)} review={review or '跳过'}"
                )

        if getattr(project, "work_items", None):
            lines.append("\n**生成的 Spec 文件（最近）:**")
            for w in project.work_items[-tail:]:
                status = w.status.value
                q = (w.question or "").strip()
                q = q if len(q) <= 60 else q[:60] + "..."
                deleted = " (已清理)" if getattr(w, "spec_deleted", False) else ""
                lines.append(f"- [{status}] {w.item_id}: {q}{deleted} • {w.spec_path}")
        return "\n".join(lines)

    def format_metrics(self, project: SpecProject, tail: int = 20) -> str:
        tail = max(1, int(tail or 20))
        lines: list[str] = [f"📈 **Spec 指标** — {project.name}\n"]
        if not getattr(project, "metrics_history", None):
            lines.append("(暂无指标记录)")
            return "\n".join(lines)

        last = project.metrics_history[-1]
        lines.append(
            f"当前: 循环 {last.cycle_number}  •  目标达成度={last.goal_attainment:.2f}  •  优化空间={last.improvement_space:.2f}"
        )
        lines.append(
            f"验收标准: {last.satisfied_count}/{last.total_criteria}  •  backlog 待办: {last.backlog_pending}\n"
        )

        lines.append("**最近变化:**")
        for m in project.metrics_history[-tail:]:
            lines.append(
                f"- 循环 {m.cycle_number}: +{m.new_satisfied} 达标  •  建议={m.review_suggestions}  •  达成度={m.goal_attainment:.2f}  •  空间={m.improvement_space:.2f}"
            )
        return "\n".join(lines)

    def format_guidance_injected(self, message: str) -> str:
        return f"""💬 **引导信息已注入**

> {message}

将在下一轮循环中生效。"""

    # ------------------------------------------------------------------
    # Title helpers (for card headers)
    # ------------------------------------------------------------------

    def get_analyzing_start_title(self) -> str:
        return "📋 Spec Agent 启动"

    def get_analyzing_done_title(self) -> str:
        return "✅ 需求分析完成"

    def get_cycle_start_title(self, cycle: int, max_cycles: int) -> str:
        return f"🔄 Spec 循环 [{cycle}/{max_cycles}]"

    def get_cycle_done_title(self, cycle: int, success: bool) -> str:
        if success:
            return f"✅ 循环完成 [{cycle}]"
        return f"❌ 循环失败 [{cycle}]"

    def get_phase_title(self, cycle: int, phase: SpecPhase) -> str:
        return f"{phase.emoji} {phase.display_name} [循环 {cycle}]"

    def get_review_title(self, cycle: int, all_passed: bool) -> str:
        if all_passed:
            return f"✅ 审查通过 [循环 {cycle}]"
        return f"🔍 多角色审查 [循环 {cycle}]"

    def get_project_done_title(self, project: SpecProject) -> str:
        if project.status == SpecProjectStatus.COMPLETED:
            return "🎉 Spec 模式完成！"
        elif project.status == SpecProjectStatus.FAILED:
            return "⚠️ Spec 模式终止"
        elif project.status == SpecProjectStatus.CANCELLED:
            return "⏹️ Spec 模式已停止"
        return "🔄 Spec 模式执行中"

    def format_goal_rewritten(self, guidance: str, new_requirement: str) -> str:
        # 截断过长的新目标，避免卡片内容过长
        preview = new_requirement if len(new_requirement) <= 500 else new_requirement[:500] + "..."
        return f"""🎯 **目标已更新**

**补充的约束/偏好：**
> {guidance}

**合并后的新目标：**
{preview}

后续所有迭代循环将基于此新目标执行。"""

    def get_goal_rewritten_title(self) -> str:
        return "🎯 目标已更新"

    def get_guidance_injected_title(self) -> str:
        return "💬 引导信息已注入"

    def get_error_title(self) -> str:
        return "❌ Spec Agent 错误"

    def get_status_title(self) -> str:
        return "📊 Spec 状态"

    # ------------------------------------------------------------------
    # Structured card sections (for build_info_card params)
    # ------------------------------------------------------------------

    def format_status_line(self, project: SpecProject) -> str:
        """One-line status for card metadata area."""
        status_map = {
            SpecProjectStatus.IDLE: "⏳ 等待开始",
            SpecProjectStatus.ANALYZING: "🧠 分析中",
            SpecProjectStatus.RUNNING: "🔄 循环执行中",
            SpecProjectStatus.COMPLETED: "✅ 已完成",
            SpecProjectStatus.FAILED: "❌ 已失败",
            SpecProjectStatus.CANCELLED: "⏹️ 已停止",
        }
        status_text = status_map.get(project.status, "❓ 未知")
        cycle_info = f"循环 {project.current_cycle_number}"
        criteria_info = f"标准 {project.satisfied_count}/{project.total_criteria}"

        # 增加规格单元计数提示，解决极简主义偏差导致的状态可见性缺失
        work_item_info = ""
        total_items = getattr(project, "work_items_total", 0)
        if total_items > 0:
            done_items = sum(1 for w in getattr(project, "work_items", []) if w.status == SpecWorkItemStatus.DONE)
            work_item_info = f" · 单元 {done_items}/{total_items}"

        return f"{status_text} · {cycle_info} · {criteria_info}{work_item_info}"

    def format_duration_line(self, project: SpecProject) -> str:
        """Duration line for card metadata area."""
        if not project.duration():
            return ""
        return f"⏱️ {format_duration(project.duration())}"

    def format_criteria_section(self, project: SpecProject) -> str:
        """Standalone criteria section for card layout."""
        return self.format_criteria_brief(project)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_progress_bar(completed: int, total: int) -> str:
        return make_progress_bar(completed, total)

    def get_progress_info(self, project: SpecProject) -> dict:
        done_work_items = sum(1 for w in getattr(project, "work_items", []) if w.status == SpecWorkItemStatus.DONE)
        total_work_items = getattr(project, "work_items_total", 0)

        return {
            "progress_bar": self._make_progress_bar(project.satisfied_count, project.total_criteria),
            "satisfied_count": project.satisfied_count,
            "total_criteria": project.total_criteria,
            "cycle_count": project.current_cycle_number,
            "done_work_items": done_work_items,
            "total_work_items": total_work_items,
            "status": project.status,
            "project_name": project.name,
            "project_id": project.project_id,
            "is_running": project.status == SpecProjectStatus.RUNNING,
            "is_paused": False,
        }
