import json


class FeishuMessageFormatter:
    @staticmethod
    def _text(text: str, styles: list = None) -> dict:
        node = {"tag": "text", "text": text}
        if styles:
            node["style"] = styles
        return node

    @staticmethod
    def _md(text: str) -> list:
        return [{"tag": "md", "text": text}]

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
    def format_dir_change(path: str, success: bool) -> str:
        if success:
            return f"📂 **已切换到**: `{path}`"
        else:
            return f"❌ **切换失败**: {path}"

    @staticmethod
    def format_current_dir(path: str) -> str:
        return f"📂 **当前工作目录**: `{path}`"

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
    def safe_truncate_markdown(content: str, max_length: int = 25000, keep_head: bool = True) -> str:
        if not content or len(content) <= max_length:
            return content or ""
        notice = f"\n\n> ⚠️ 内容过长（超过 {max_length} 字符），已自动截断。"
        available = max_length - len(notice)
        truncated = content[:available] if keep_head else content[-available:]
        if truncated.count("```") % 2:
            truncated = truncated + "\n```" if keep_head else "```\n" + truncated
        return truncated + notice if keep_head else notice + "\n\n" + truncated
