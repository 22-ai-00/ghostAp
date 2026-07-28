"""Feishu Slash Command catalog for the main GhostAP Bot.

Slash Command registration only controls the command-discovery panel. Selected
commands still arrive through Channel SDK as ordinary message events and are
executed by the existing request-scoped SlashCommandParser routing chain.
"""

from __future__ import annotations

from typing import Protocol

from lark_oapi.core.model.base_request import BaseRequest
from lark_oapi.core.model.base_response import BaseResponse

from ..autonomous.provisioning.slash_commands import (
    SlashCommand,
    SlashCommandReconciler,
    VerifiedSlashState,
)
from ..autonomous.provisioning.slash_lark import LarkSlashCommandAPI


class _AsyncLarkClient(Protocol):
    async def arequest(self, request: BaseRequest) -> BaseResponse: ...


# Register primary public spellings rather than every compatibility alias.
# This keeps the client picker useful while still covering every GhostAP
# capability group and staying comfortably below Feishu's 100-command limit.
MAIN_AGENT_COMMANDS: tuple[SlashCommand, ...] = (
    # Help and common entry points
    SlashCommand("/help", "查看 GhostAP 完整帮助"),
    SlashCommand("/menu", "打开 GhostAP 快捷操作菜单"),
    SlashCommand("/tools", "查看可用的 AI 编程工具"),
    SlashCommand("/tools_status", "查看 AI 编程工具状态"),
    # Programming modes and sessions
    SlashCommand("/coco", "进入 Coco 编程模式"),
    SlashCommand("/claude", "进入 Claude 编程模式"),
    SlashCommand("/aiden", "进入 Aiden 编程模式"),
    SlashCommand("/codex", "进入 Codex 编程模式"),
    SlashCommand("/gemini", "进入 Gemini 编程模式"),
    SlashCommand("/traex", "进入 Traex 编程模式"),
    SlashCommand("/ttadk", "进入 TTADK 多工具编程模式"),
    SlashCommand("/tui2acp", "进入 TUI2ACP 桥接模式"),
    SlashCommand("/acp", "打开 ACP 工具选择入口"),
    SlashCommand("/model", "查看或切换当前模型", "/model [list|模型名]"),
    SlashCommand("/exit", "退出当前编程模式"),
    SlashCommand("/btw", "在当前编程会话中提出旁路问题", "/btw <问题>"),
    # Programming diagnostics
    SlashCommand("/coco_status", "查看 Coco 会话状态"),
    SlashCommand("/coco_info", "查看 Coco 会话与模型信息"),
    SlashCommand("/claude_info", "查看 Claude 会话与模型信息"),
    SlashCommand("/aiden_info", "查看 Aiden 会话与模型信息"),
    SlashCommand("/codex_info", "查看 Codex 会话与模型信息"),
    SlashCommand("/gemini_info", "查看 Gemini 会话与模型信息"),
    SlashCommand("/traex_info", "查看 Traex 会话与模型信息"),
    SlashCommand("/ttadk_info", "查看 TTADK 工具与模型信息"),
    SlashCommand("/tui2acp_info", "查看 TUI2ACP 会话信息"),
    SlashCommand("/ttadk_refresh", "刷新 TTADK 模型列表"),
    # Projects, task visibility, and locks
    SlashCommand("/projects", "查看项目面板"),
    SlashCommand("/new", "创建项目", "/new <名称> [目录]"),
    SlashCommand("/new-chat", "创建或绑定项目群", "/new-chat <名称> [后缀] [目录]"),
    SlashCommand("/switch", "切换项目", "/switch <名称>"),
    SlashCommand("/close", "关闭项目", "/close <名称>"),
    SlashCommand("/status", "查看项目、模式、锁与任务状态", "/status [任务ID]"),
    SlashCommand("/tasks", "查看任务看板", "/tasks [筛选条件]"),
    SlashCommand("/diff", "查看最近版本变更", "/diff [范围]"),
    SlashCommand("/trace", "查看消息处理链路", "/trace <消息ID>"),
    SlashCommand("/lock", "锁定当前聊天的任务执行"),
    SlashCommand("/unlock", "解除当前聊天锁定"),
    SlashCommand("/setadmin", "设置或替换 GhostAP 管理员", "/setadmin [用户ID]"),
    # Deep Engine
    SlashCommand("/deep", "启动 Deep Engine 复杂任务", "/deep <需求>"),
    SlashCommand("/deep_status", "查看 Deep Engine 任务进度"),
    SlashCommand("/deep_update", "补充 Deep Engine 任务约束", "/deep_update <补充说明>"),
    SlashCommand("/stop_deep", "停止 Deep Engine 任务"),
    # Spec Engine
    SlashCommand("/spec", "启动 Spec Engine 结构化开发", "/spec <需求>"),
    SlashCommand("/spec_status", "查看 Spec Engine 任务进度"),
    SlashCommand("/spec_history", "查看 Spec Engine 历史"),
    SlashCommand("/spec_metrics", "查看 Spec Engine 目标达成度"),
    SlashCommand("/spec_config", "查看或调整 Spec Engine 配置"),
    SlashCommand("/spec_export", "导出 Spec Engine 报告"),
    SlashCommand("/spec_save", "立即保存 Spec Engine 状态"),
    SlashCommand("/spec_pause", "暂停 Spec Engine 任务"),
    SlashCommand("/spec_resume", "继续 Spec Engine 任务"),
    SlashCommand("/spec_guide", "补充 Spec Engine 引导", "/spec_guide <引导>"),
    SlashCommand("/spec_recover", "恢复失败的 Spec Engine 任务", "/spec_recover [任务ID]"),
    SlashCommand("/stop_spec", "停止 Spec Engine 任务"),
    # Worktree and Workflow
    SlashCommand("/worktree", "启动 Worktree 多工具并行执行", "/worktree [目标]"),
    SlashCommand("/wf", "启动多 Agent Workflow", "/wf <需求>"),
    SlashCommand("/wf_status", "查看当前 Workflow 进度"),
    SlashCommand("/wf_help", "查看 Workflow 使用帮助"),
    SlashCommand("/stop_wf", "停止正在运行的 Workflow"),
    SlashCommand("/wf_save", "保存 Workflow 模板", "/wf_save <名称>"),
    SlashCommand("/wf_list", "列出 Workflow 模板"),
    SlashCommand("/wf_delete", "删除 Workflow 模板", "/wf_delete <名称>"),
    SlashCommand("/wf_history", "查看 Workflow 执行历史"),
    # Slock teams and persistent employees
    SlashCommand("/slock", "激活或管理当前 Slock 团队", "/slock [status|stop|help]"),
    SlashCommand("/slocks", "查看全部 Slock 团队"),
    SlashCommand("/new-team", "创建 Slock 协作团队", "/new-team <团队名>"),
    SlashCommand("/new-role", "创建 Slock 角色", "/new-role <名称>"),
    SlashCommand("/hire", "雇佣持久数字员工", "/hire <名字> [选项]"),
    SlashCommand("/fire", "退役持久数字员工", "/fire <名字>"),
    SlashCommand("/employees", "查看在职数字员工"),
    SlashCommand("/history", "查看数字员工执行历史", "/history [员工名]"),
    SlashCommand("/employee-memory", "查看数字员工记忆摘要", "/employee-memory <员工名>"),
    SlashCommand("/role", "管理 Slock 群内角色", "/role [list|add|remove|info|move]"),
    SlashCommand("/task", "查看 Slock 任务", "/task [list|status]"),
    SlashCommand("/team", "管理 Slock 团队", "/team [list|status|dissolve]"),
    SlashCommand("/council", "发起 Slock 多 Agent 评议", "/council <问题>"),
    SlashCommand("/discuss", "发起或管理 Slock 讨论", "/discuss [主题|stop|history|list]"),
    SlashCommand("/memory", "查看 Slock 记忆", "/memory [list|group|员工名]"),
    SlashCommand("/plan", "查看 Slock 计划", "/plan [list|计划ID]"),
)


async def reconcile_main_agent_slash_commands(
    client: _AsyncLarkClient,
) -> VerifiedSlashState:
    """Converge the main Bot's server-side Slash Command panel."""

    return await SlashCommandReconciler(
        LarkSlashCommandAPI(client),
        desired=MAIN_AGENT_COMMANDS,
    ).reconcile()
