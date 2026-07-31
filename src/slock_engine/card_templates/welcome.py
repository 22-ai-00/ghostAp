"""Welcome card template for Slock Engine.

Provides the welcome card sent inside newly created Slock team groups.
"""

from __future__ import annotations

from .common import build_card_wrapper

__all__ = ["build_welcome_card"]


def build_welcome_card(*, team_name: str) -> dict:
    """Build a welcome card sent inside the newly created Slock team group."""
    content = (
        f"🎭 **协作团队「{team_name}」已就绪**\n\n"
        "团队已开启自主工作模式，员工 Agent 会自动响应群内消息。\n\n"
        "**快速开始:**\n"
        "• `/hire <名字>` — 雇佣新员工（选工具+模型）\n"
        "• `/role add` — 添加已有员工到本群\n"
        "• `/role list` — 查看群内员工\n"
        "• 直接发送任务描述 — 创建团队任务\n"
        "• 直接 @员工名 发消息 — 指定员工执行任务\n\n"
        "**管理命令:**\n"
        "`/role remove <名字>` · `/task status` · `/slock status` · `/team dissolve`"
    )
    elements: list[dict] = [{"tag": "markdown", "content": content}]
    return build_card_wrapper(
        header_title=f"👋 欢迎加入 {team_name}",
        header_template="indigo",
        elements=elements,
        mobile_optimize=True,
    )
