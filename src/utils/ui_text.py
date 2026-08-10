"""Shared UI text constants.

This module lives in the utils (infrastructure) layer so that both
presentation layers (card/) and engine layers (spec_engine/) can import
from it without creating reverse dependencies.
"""

from __future__ import annotations

SPEC_UI_TEXT: dict[str, str] = {
    "retry_waiting": "⏳ 约 {sec} 秒后自动重试（第 {i}/{n} 次）",
    "retry_executing": "🔄 审查重试中（第 {i}/{n} 次）",
    "retry_exhausted": "⚠️ 已重试 {n} 次仍未完成（共耗时{elapsed_friendly}）。系统将自动跳过本轮审查继续执行，也可点击下方按钮或发送 /spec resume 手动恢复",
    "retry_no_retry": "⚠️ 审查超时，未进行重试。系统将在下一轮自动恢复。如频繁出现，可设置环境变量 SPEC_REVIEW_TIMEOUT 调大超时阈值，也可点击下方按钮或发送 /spec resume 继续",
    "retry_no_retry_disabled": "⚠️ 审查超时，当前配置未启用重试。可点击下方按钮或发送 /spec resume 继续",
    "retry_no_retry_budget": "⚠️ 审查超时，本轮已超出重试预算。系统将在下一轮自动恢复，也可点击下方按钮或发送 /spec resume 继续",
    "phase_retry_progress": "🔄 调用重试 {attempt}/{max_attempts}",
    "btn_stop_review": "终止整个审查",
    "btn_continue": "跳过并继续",
    "review_parse_fail_system": "审查输出解析异常（系统侧），不影响执行，将在下轮自动重试",
    "review_budget_timeout": "部分审查未在限定时间内完成，已自动跳过并继续下一轮。如持续出现，可发送 /spec resume 手动恢复",
    "circuit_breaker_skip_with_count": "审查暂停：连续{count}次异常，系统已暂时跳过本轮审查。系统将在后续轮次自动恢复，无需手动操作。如需立即恢复，发送 /spec resume 继续",
    # ── Worker layer text (perspective_worker.py) ──
    "worker_summary_passed": "通过",
    "worker_summary_exception": "异常",
    "worker_suggestion_default": "需要关注实现质量",
    "worker_summary_n_suggestions": "{n}条建议",
    "worker_summary_has_suggestions": "有建议",
    "worker_error_unknown": "未知错误",
    # ReviewPerspective display names
    "perspective_architect": "架构师",
    "perspective_product": "产品经理",
    "perspective_user": "用户",
    "perspective_tester": "测试",
    "perspective_designer": "设计师",
}
