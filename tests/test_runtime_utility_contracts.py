"""Minimal contracts for the active utility owners."""


from src.card.terminal import get_terminal_marker
from src.feishu.message_formatter import FeishuMessageFormatter


def test_card_terminal_owns_dynamic_blocked_marker() -> None:
    assert get_terminal_marker("blocked", reason="需要安全输入") == (
        "⏸ **任务已阻塞** — 需要安全输入"
    )
    assert get_terminal_marker("unknown") is None


def test_message_formatter_truncates_markdown_without_open_fence() -> None:
    rendered = FeishuMessageFormatter.safe_truncate_markdown(
        "```python\nprint('hello')\n" + "x" * 100,
        max_length=64,
    )

    assert "已自动截断" in rendered
    assert rendered.count("```") % 2 == 0

