"""Tests for _get_confirm_title — covers white-list mapping logic.

After refactoring, _get_confirm_title uses exact-match white-list mapping
(no substring matching). This test file verifies all registered mappings
and guards against false positives from substring-like action_ids.
"""

import pytest

from src.card.render.buttons import _get_confirm_title


class TestGetConfirmTitleWhiteList:
    """White-list mapping: exact action_id → confirm title."""

    # --- Stop intents ---

    def test_engine_stop(self):
        assert _get_confirm_title("intent.engine.stop") == "确定停止当前任务？"

    def test_deep_stop(self):
        assert _get_confirm_title("intent.deep.stop") == "确定停止当前任务？"

    def test_spec_stop(self):
        assert _get_confirm_title("intent.spec.stop") == "确定停止当前任务？"

    # --- Approval ---

    def test_approval_approve(self):
        assert _get_confirm_title("intent.approval.approve") == "授权执行"

    def test_approval_reject(self):
        assert _get_confirm_title("intent.approval.reject") == "拒绝请求"


class TestGetConfirmTitleNoSubstringMatching:
    """Verify that substring matching is NOT used."""

    def test_show_stopwatch_does_not_match_stop(self):
        """An action_id containing 'stop' as substring should NOT hit stop title."""
        result = _get_confirm_title("show_stopwatch")
        # Should fall to default, not "停止当前任务"
        assert result == "确认操作"

    def test_cancel_in_middle_does_not_match(self):
        """'my_cancel_handler' should NOT match cancel intent."""
        result = _get_confirm_title("my_cancel_handler")
        assert result == "确认操作"

    def test_retry_substring_does_not_match(self):
        """An unregistered retry action should not match a registered intent."""
        result = _get_confirm_title("unknown_retry_failed")
        assert result == "确认操作"

    def test_merge_substring_does_not_match(self):
        """An unregistered merge action should not match a registered intent."""
        result = _get_confirm_title("unknown_merge_branches")
        assert result == "确认操作"


class TestGetConfirmTitleFallback:
    """Fallback behavior for unregistered action_ids."""

    def test_button_text_fallback(self):
        """Unregistered action + button_text → template fallback."""
        result = _get_confirm_title("some_random_action", button_text="提交审核")
        assert result == "确认「提交审核」？"

    def test_default_fallback_no_button_text(self):
        """Unregistered action + no button_text → generic default."""
        result = _get_confirm_title("some_random_action")
        assert result == "确认操作"

    def test_empty_action_id_fallback(self):
        """Empty action_id → default fallback."""
        result = _get_confirm_title("")
        assert result == "确认操作"


class TestDeadKeyFallback:
    """Regression test: removed dead keys must fall back to default title.

    These action_ids were cleaned up during the card migration and should
    NOT be re-introduced into _CONFIRM_TITLE_MAP. They must always return
    the generic fallback '确认操作？'.
    """

    @pytest.mark.parametrize("dead_key", [
        "intent.spec.retry",
        "intent.deep.retry",
        "intent.unknown.execute",
    ])
    def test_dead_key_returns_default(self, dead_key: str):
        """Removed dead key should return generic default fallback."""
        assert _get_confirm_title(dead_key) == "确认操作"
