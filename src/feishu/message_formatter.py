import json
from typing import Optional

from src.card.thresholds import THRESHOLDS


class FeishuMessageFormatter:
    @staticmethod
    def _text(text: str, styles: list = None) -> dict:
        node = {"tag": "text", "text": text}
        if styles:
            node["style"] = styles
        return node

    @staticmethod
    def _link(text: str, href: str) -> dict:
        return {"tag": "a", "text": text, "href": href}

    @staticmethod
    def _code_block(code: str, language: str = "BASH") -> list:
        return [{"tag": "code_block", "language": language, "text": code}]

    @staticmethod
    def _md(text: str) -> list:
        return [{"tag": "md", "text": text}]

    @staticmethod
    def _hr() -> list:
        return [{"tag": "hr"}]

    @staticmethod
    def format_command_result(
        command: str,
        working_dir: Optional[str],
        stdout: str,
        stderr: str,
        return_code: int,
        success: bool,
        error_message: Optional[str] = None,
    ) -> tuple[str, str]:
        title = "✅ 执行成功" if success else "❌ 执行失败"

        md_parts = []

        if working_dir:
            md_parts.append(f"📂 **工作目录**: `{working_dir}`")

        md_parts.append(f"> 🖥️ `{command}`")
        md_parts.append("")

        if error_message:
            md_parts.append(f"**错误**: {error_message}")
        else:
            if stdout:
                stdout_clean = stdout.strip()
                if len(stdout_clean) > THRESHOLDS["SHELL_STDOUT_MAX"]:
                    stdout_clean = stdout_clean[:THRESHOLDS["SHELL_STDOUT_MAX"]] + "\n... (输出已截断)"
                md_parts.append("**📤 输出**:")
                md_parts.append(f"```BASH\n{stdout_clean}\n```")

            if stderr:
                stderr_clean = stderr.strip()
                if len(stderr_clean) > THRESHOLDS["SHELL_STDERR_MAX"]:
                    stderr_clean = stderr_clean[:THRESHOLDS["SHELL_STDERR_MAX"]] + "\n... (已截断)"
                md_parts.append("**⚠️ 错误输出**:")
                md_parts.append(f"```BASH\n{stderr_clean}\n```")

            if not stdout and not stderr:
                md_parts.append("*（无输出）*")

        md_parts.append("")
        md_parts.append(f"🔢 **返回码**: {return_code}")

        content = [[{"tag": "md", "text": "\n".join(md_parts)}]]

        post_content = {"zh_cn": {"title": title, "content": content}}

        return "post", json.dumps(post_content)

    @staticmethod
    def format_safety_block(command: str, reason: str) -> tuple[str, str]:
        md_text = f"""> 🖥️ `{command}`

**❌ 拦截原因**: {reason}

*如需执行此命令，请联系管理员*"""

        content = [[{"tag": "md", "text": md_text}]]

        post_content = {"zh_cn": {"title": "🚫 命令被安全检查拦截", "content": content}}

        return "post", json.dumps(post_content)

    @staticmethod
    def format_help(current_dir: str, is_coco_mode: bool = False) -> tuple[str, str]:
        if is_coco_mode:
            md_text = """直接发送消息与 Coco 对话进行远程开发

**可用命令**:
- 说「退出」或 `/end_coco` - 退出 Coco 模式
- `/coco_info` - 查看会话信息"""
            title = "🤖 当前在 Coco 模式"
        else:
            md_text = f"""📂 **当前目录**: `{current_dir}`

**📟 Shell 模式**（默认）:
- 直接发送 shell 命令执行
- 示例: `ls -la` 或 `git status`

**🤖 Coco 模式**:
- 说「帮我写代码」进入 Coco 模式
- 说「退出」退出 Coco 模式

**📁 目录切换**:
- 说「切换到xxx目录」
- 支持自然语言描述"""
            title = "💡 使用说明"

        content = [[{"tag": "md", "text": md_text}]]

        post_content = {"zh_cn": {"title": title, "content": content}}

        return "post", json.dumps(post_content)

    @staticmethod
    def format_coco_enter() -> tuple[str, str]:
        md_text = """现在你可以直接发送消息与 Coco 对话，进行远程开发。

**💡 提示**:
- 说「退出」或 `/end_coco` - 退出 Coco 模式
- `/coco_info` - 查看会话信息"""

        content = [[{"tag": "md", "text": md_text}]]

        post_content = {"zh_cn": {"title": "🤖 已进入 Coco 模式", "content": content}}

        return "post", json.dumps(post_content)

    @staticmethod
    def format_coco_response(response: str) -> tuple[str, str]:
        content = [[{"tag": "md", "text": response}]]

        post_content = {"zh_cn": {"title": "🤖 Coco", "content": content}}

        return "post", json.dumps(post_content)

    @staticmethod
    def format_dir_change(path: str, success: bool) -> str:
        if success:
            return f"📂 **已切换到**: `{path}`"
        else:
            return f"❌ **切换失败**: {path}"

    @staticmethod
    def format_current_dir(path: str) -> str:
        return f"📂 **当前工作目录**: `{path}`"

    @staticmethod
    def format_multi_task_plan(tasks: list[dict]) -> tuple[str, str]:
        lines = ["我理解你的需求，将执行以下步骤：", ""]
        for i, task in enumerate(tasks, 1):
            desc = task.get("description", "未知操作")
            lines.append(f"{i}. {desc}")
        lines.append("")
        lines.append("🚀 **开始执行**...")

        content = [[{"tag": "md", "text": "\n".join(lines)}]]

        post_content = {"zh_cn": {"title": "📋 执行计划", "content": content}}

        return "post", json.dumps(post_content)

    @staticmethod
    def format_unknown_intent() -> tuple[str, str]:
        md_text = """**💡 你可以**:
- 直接输入 shell 命令执行
- 说「帮我写代码」进入 Coco 模式
- 说「切换到xxx目录」切换工作目录"""

        content = [[{"tag": "md", "text": md_text}]]

        post_content = {"zh_cn": {"title": "🤔 无法理解你的意图", "content": content}}

        return "post", json.dumps(post_content)

    @staticmethod
    def format_error(message: str) -> str:
        return f"❌ **错误**: {message}"

    @staticmethod
    def format_warning(message: str) -> str:
        return f"⚠️ **警告**: {message}"

    @staticmethod
    def is_post_format(result) -> bool:
        return isinstance(result, tuple) and len(result) == 2 and result[0] == "post"

    @staticmethod
    def safe_truncate_markdown(content: str, max_length: int = 25000, keep_head: bool = True) -> str:
        if not content or len(content) <= max_length:
            return content or ""
        notice = f"\n\n> ⚠️ 内容过长（超过 {max_length} 字符），已自动截断。"
        available = max_length - len(notice)
        truncated = content[:available] if keep_head else content[-available:]
        if truncated.count("```") % 2:
            truncated = truncated + "\n```" if keep_head else "```\n" + truncated
        return truncated + notice if keep_head else notice + "\n\n" + truncated
