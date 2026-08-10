from ..utils.text import format_duration, make_progress_bar
from .models import DeepProject, DeepProjectStatus


class ProgressReporter:
    def format_planning_start(self, requirement: str) -> str:
        return f"""🧠 **Deep Agent 启动**

📝 正在分析需求...
> {requirement}

⏳ 请稍候，正在规划任务..."""

    def format_planning_done(self, project: DeepProject) -> str:
        return f"""✅ **任务规划完成**

📂 项目: {project.name}
📍 目录: `{project.root_path}`

🚀 准备开始执行..."""

    def format_project_done(self, project: DeepProject) -> str:
        if project.status == DeepProjectStatus.COMPLETED:
            lines = [
                "🎉 **全部任务完成！**\n",
                f"📂 项目: {project.name}",
                f"⏱️ 总耗时: {format_duration(project.duration())}" if project.duration() else "",
            ]
        elif project.status == DeepProjectStatus.FAILED:
            lines = [
                "⚠️ **执行完成（有失败）**\n",
                f"📂 项目: {project.name}",
                f"⏱️ 总耗时: {format_duration(project.duration())}" if project.duration() else "",
            ]
        elif project.status == DeepProjectStatus.CANCELLED:
            lines = [
                "⏹️ **执行已取消**\n",
                f"📂 项目: {project.name}",
            ]
        else:
            lines = ["❌ **执行未进入有效终态**\n", f"📂 项目: {project.name}"]

        return "\n".join(line for line in lines if line)

    def format_error(self, error: str) -> str:
        err = (error or "").strip() or "未知错误"
        is_timeout = "TimeoutError" in err or "操作耗时过长" in err or "timeout" in err.lower()
        internal = "Internal error" if "internal error" in err.lower() else ""

        summary_parts: list[str] = []
        if is_timeout:
            summary_parts.append("⏱️ 操作超时，请检查网络或稍后重试")
        if internal:
            summary_parts.append(internal)

        summary_line = ""
        if summary_parts:
            summary_line = "\n\n" + "\n".join([f"- {p}" for p in summary_parts]) + "\n"

        advice = "自动恢复已耗尽，任务已明确失败。"

        return f"""❌ **Deep Agent 错误**{summary_line}
```
{err}
```

{advice}"""

    def format_status(self, project: DeepProject) -> str:
        status_text = {
            DeepProjectStatus.IDLE: "⏳ 等待开始",
            DeepProjectStatus.PLANNING: "🧠 正在规划",
            DeepProjectStatus.EXECUTING: "🔄 执行中",
            DeepProjectStatus.CANCELLED: "⏹️ 已取消",
            DeepProjectStatus.COMPLETED: "✅ 已完成",
            DeepProjectStatus.FAILED: "❌ 执行失败",
        }.get(project.status, "❓ 未知状态")

        lines = [
            f"📊 **{project.name}** 状态\n",
            f"状态: {status_text}",
        ]

        if project.duration():
            lines.append(f"⏱️ 已执行: {format_duration(project.duration())}")

        return "\n".join(lines)

    @staticmethod
    def _make_progress_bar(completed: int, total: int) -> str:
        return make_progress_bar(completed, total)

    def format_context_injected(self, message: str) -> str:
        return f"""💬 **上下文已注入**

> {message}

将在下一个任务执行前生效。"""

    # --- Card title helpers ---

    def get_planning_start_title(self) -> str:
        return "🧠 Deep Agent 启动"

    def get_planning_done_title(self) -> str:
        return "✅ 任务规划完成"

    def get_project_done_title(self, project: DeepProject) -> str:
        if project.status == DeepProjectStatus.COMPLETED:
            return "🎉 全部任务完成！"
        elif project.status == DeepProjectStatus.FAILED:
            return "⚠️ 执行完成（有失败）"
        if project.status == DeepProjectStatus.CANCELLED:
            return "⏹️ 执行已取消"
        return "❌ 执行未完成"

    def get_context_injected_title(self) -> str:
        return "💬 上下文已注入"

    def get_error_title(self) -> str:
        return "❌ Deep Agent 错误"

    def get_status_title(self) -> str:
        return "📊 任务状态"

    def get_progress_info(self, project: DeepProject, completed: int = 0, total: int = 0) -> dict:
        return {
            "progress_bar": self._make_progress_bar(completed, total),
            "completed_count": completed,
            "total_count": total,
            "status": project.status,
            "project_name": project.name,
            "project_id": project.project_id,
            "is_executing": project.status in (
                DeepProjectStatus.PLANNING,
                DeepProjectStatus.EXECUTING,
            ),
        }
