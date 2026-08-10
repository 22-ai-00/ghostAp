"""Minimal contracts for the active utility owners."""

import logging

from src.card.terminal import get_terminal_marker
from src.feishu.message_formatter import FeishuMessageFormatter
from src.utils.metrics_exporter import get_metrics_exporter


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


def test_logging_metrics_exporter_emits_unicode_json(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="src.utils.metrics_exporter"):
        get_metrics_exporter().export_metrics({"通过": 1}, prefix="workflow")

    assert 'workflow review_metrics: {"通过": 1}' in caplog.text
