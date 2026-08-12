"""契约测试：锁定核心格式化函数的输出签名。

本模块是 src/utils/text.py 和 src/card/ui_text.py 的 **契约测试**（contract test）。
目的是在重构或迁移时，通过最小化的断言集合确保公共接口的输出签名不发生无声的破坏性
变更。

范围：
- format_duration  —— 秒数 → 中文时长字符串（精确值断言）
- format_time_ago  —— 秒数 → 中文相对时间字符串（包含子串断言）
- make_progress_bar —— (completed, total) → 非空字符串（类型 + 非空断言）
- UI_TEXT          —— 必须包含指定 key，且对应值为 str
- SPEC_UI_TEXT     —— 必须存在且为 dict
"""

import pytest

from src.card.ui_text import UI_TEXT
from src.utils.text import format_duration, format_time_ago, make_progress_bar
from src.utils.ui_text import SPEC_UI_TEXT

# ---------------------------------------------------------------------------
# format_duration 契约
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "0 秒"),
        (5, "5 秒"),
        (65, "1 分钟 5 秒"),
        (3661, "1 小时 1 分钟 1 秒"),
        (-1, "0 秒"),   # 负数归零
    ],
)
def test_format_duration_contract(seconds, expected):
    """format_duration 的输出签名必须与预期精确一致。"""
    assert format_duration(seconds) == expected


# ---------------------------------------------------------------------------
# format_time_ago 契约
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds, expected_substr",
    [
        (5, "刚刚"),
        (120, "分钟前"),
        (7200, "小时前"),
        (90000, "天前"),
    ],
)
def test_format_time_ago_contract(seconds, expected_substr):
    """format_time_ago 的输出必须包含预期子串。"""
    result = format_time_ago(seconds)
    assert expected_substr in result, (
        f"format_time_ago({seconds!r}) = {result!r}，"
        f"期望包含 {expected_substr!r}"
    )


# ---------------------------------------------------------------------------
# make_progress_bar 契约
# ---------------------------------------------------------------------------

def test_make_progress_bar_returns_nonempty_string():
    """make_progress_bar 必须返回非空字符串。"""
    result = make_progress_bar(3, 10)
    assert isinstance(result, str), f"期望 str，实际得到 {type(result)}"
    assert len(result) > 0, "make_progress_bar 不应返回空字符串"


# ---------------------------------------------------------------------------
# UI_TEXT key 契约
# ---------------------------------------------------------------------------

_REQUIRED_UI_TEXT_KEYS = [
    # 时间相关
    "time_just_now",
    "time_secs_ago",
    "time_mins_ago",
    "time_hours_ago",
    "time_days_ago",
    # Deep Engine
    "deep_cmd_help_usage",
    "deep_task_exists",
    "deep_no_task_running",
    # Spec Engine
    "spec_status_empty",
    "spec_cmd_help_usage",
    # Generic Engine Lifecycle
    "engine_no_active_task",
    "engine_stop_no_active",
]


@pytest.mark.parametrize("key", _REQUIRED_UI_TEXT_KEYS)
def test_ui_text_required_keys_exist_and_are_str(key):
    """UI_TEXT 必须包含指定 key，且对应值为 str。"""
    assert key in UI_TEXT, f"UI_TEXT 缺少必需 key: {key!r}"
    value = UI_TEXT[key]
    assert isinstance(value, str), (
        f"UI_TEXT[{key!r}] 期望值类型为 str，实际为 {type(value)}"
    )


def test_employee_help_matches_current_lifecycle_command_contract():
    """Employee 帮助只宣传已实现的自动雇佣与生命周期命令。"""
    body = UI_TEXT["system_help_section_hire_body"]

    for command in (
        "`/hire <名字>`",
        "`/h <名字>`",
        "`/fire <名字或ID> [--drain]`",
        "`/history <名字或ID>`",
        "`/employee-memory <名字或ID>`",
    ):
        assert command in body

    assert "自动选择推荐可用工具和后端默认模型" in body
    assert "--tool" in body
    assert "--model" in body
    assert "弹出工具+模型选择" not in body
    assert "--prompt" not in body


def test_workflow_help_matches_single_agent_pool_confirmation_contract():
    """快捷帮助必须展示现役 Agent Pool 单次确认门。"""
    body = UI_TEXT["system_help_section_workflow_body"]

    assert "`/workflow_help`" in body
    assert "1–8" in body
    assert "Agent Pool" in body
    assert "使用此池开始编排" in body
    assert "确认后自动生成、验证并执行" in body


# ---------------------------------------------------------------------------
# SPEC_UI_TEXT 契约
# ---------------------------------------------------------------------------

def test_spec_ui_text_exists_and_is_dict():
    """SPEC_UI_TEXT 必须存在且为 dict。"""
    assert isinstance(SPEC_UI_TEXT, dict), (
        f"SPEC_UI_TEXT 期望类型为 dict，实际为 {type(SPEC_UI_TEXT)}"
    )
