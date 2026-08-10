"""Terminal state markers, footer status, and status display map."""

STATUS_DISPLAY_MAP: dict[str, str] = {
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "running": "执行中",
    "planned": "已规划",
    "ready": "就绪",
    "pending": "等待中",
}

# ──────────────────────────────────────────────────────────────
# Terminal State Markers — emoji-prefixed markdown for card endings
# ──────────────────────────────────────────────────────────────
TERMINAL_MARKERS: dict[str, str] = {
    "completed": "✅ **已完成**",
    "completed_empty": "📋 **已完成（无变更）**",
    "failed": "❌ **执行失败**",
    "blocked": "⏸ **任务已阻塞**",
    "cancelled": "⏹ **已取消**",
    "archived": "📋 **已归档**",
    "ttl_expired": "⏰ **已超时关闭**",
    "denied": "🚫 **授权已拒绝**",
    "continued": "🔓 **已获得授权**",
}


def get_terminal_marker(status: str, *, reason: str | None = None) -> str | None:
    """Get terminal marker text with optional dynamic reason (for blocked state)."""
    marker = TERMINAL_MARKERS.get(status)
    if marker and status == "blocked" and reason:
        return f"⏸ **任务已阻塞** — {reason}"
    return marker


# ──────────────────────────────────────────────────────────────
# Footer Status — real-time phase indicator shown at card bottom
# ──────────────────────────────────────────────────────────────
FOOTER_STATUS: dict[str, str] = {
    "thinking": "🧠 正在思考",
    "tool_running": "🧰 正在调用工具",
}
