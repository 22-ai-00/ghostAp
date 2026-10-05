"""Shared unattended execution instructions for agent prompt transports."""

AUTONOMOUS_EXECUTION_PROMPT = (
    "本会话由 GhostAP 执行用户已授权的任务，没有可操作的交互终端。"
    "在原任务范围内直接执行，遇到实现选择优先采用推荐项；没有推荐项时"
    "采用符合任务目标的合理默认值并简要记录，不要等待用户确认或调用提问工具。"
    "如果本轮明确限定只做分析、生成文本或按指定格式输出，应遵守该限制。"
    "涉及文件变更或命令执行时必须实际调用工具并核实结果，不得用文字声称代替执行。"
    "权限使用本会话已配置的后端授权模式，不要要求用户手动执行命令或回复继续。"
    "完成必要验证后如实报告结果；遇到无法恢复的权限、凭据或环境错误时"
    "明确报告失败及未完成事项，不得把仅提供方案或等待授权称为完成。"
)


def append_execution_instructions(text: str) -> str:
    """Keep the original request intact when a transport accepts one text argument."""
    return f"{text}\n\n---\nGhostAP 执行约定：{AUTONOMOUS_EXECUTION_PROMPT}"
