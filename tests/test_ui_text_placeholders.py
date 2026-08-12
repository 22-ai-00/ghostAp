"""Test that UI_TEXT does not use deprecated shorthand placeholders."""
import re

from src.card.ui_text import UI_TEXT

# Regex to find {placeholder} patterns in format strings
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

# Deprecated shorthand names that should NOT exist
_DEPRECATED_PLACEHOLDERS = {"secs", "mins"}


class TestUITextPlaceholderConsistency:
    """Verify all UI_TEXT format strings use canonical placeholder names."""

    def test_no_deprecated_shorthand_placeholders(self):
        """AC-6: No {secs} or {mins} shorthand in any UI_TEXT value."""
        violations = []
        for key, value in UI_TEXT.items():
            if not isinstance(value, str):
                continue
            found = _PLACEHOLDER_RE.findall(value)
            for placeholder in found:
                if placeholder in _DEPRECATED_PLACEHOLDERS:
                    violations.append(f"{key}: found deprecated placeholder {{{placeholder}}}")
        assert violations == [], "Deprecated placeholders found:\n" + "\n".join(violations)

class TestTTLPrewarningText:
    """AC-8: TTL prewarning contains time-related closure hint."""

    def test_ttl_prewarning_contains_closure_hint(self):
        assert "分钟后关闭" in UI_TEXT["card_session_ttl_prewarning"]
        assert "{minutes}" in UI_TEXT["card_session_ttl_prewarning"]

    def test_ttl_prewarning_references_keep_alive_btn(self):
        """Prewarning text must reference the keep-alive button name to avoid semantic gap."""
        assert "「保持连接」" in UI_TEXT["card_session_ttl_prewarning"]
        assert "保持连接" in UI_TEXT["ttl_keep_alive_btn"]


class TestErrorTextsActionable:
    """AC-9, AC-10, AC-11: Error texts contain actionable next steps."""

    def test_card_content_load_error_has_guidance(self):
        assert "联系管理员" in UI_TEXT["card_content_load_error"]

    def test_card_content_load_error_running_has_guidance(self):
        assert "自动恢复" in UI_TEXT["card_content_load_error_running"]

    def test_deep_error_no_detail_has_retry_hint(self):
        assert "{engine_cmd}" in UI_TEXT["deep_error_no_detail"]

    def test_intent_unknown_has_help_hint(self):
        assert "/help" in UI_TEXT["intent_unknown_msg"]


class TestNewPhase3Keys:
    """Verify new keys added in Phase 3 exist and are well-formed."""

    def test_toast_dedup_key_exists(self):
        assert "card_session_toast_dedup" in UI_TEXT
        assert "处理中" in UI_TEXT["card_session_toast_dedup"]
        assert "重复" not in UI_TEXT["card_session_toast_dedup"]

    def test_ttl_lock_contention_key_exists(self):
        assert "card_session_ttl_lock_contention" in UI_TEXT
        assert "{engine_cmd}" in UI_TEXT["card_session_ttl_lock_contention"]

    def test_force_close_notice_has_resource_reclaim(self):
        assert "系统回收资源" in UI_TEXT["card_session_ttl_force_close_notice"]

    def test_terminal_fallback_has_engine_cmd(self):
        assert "{engine_cmd}" in UI_TEXT["card_session_terminal_fallback_notice"]

    def test_terminal_fallback_does_not_suggest_bare_deep_command(self):
        rendered = UI_TEXT["card_session_terminal_fallback_notice"].format(engine_cmd="/deep")
        retry_rendered = UI_TEXT["card_session_terminal_retry_failed"].format(engine_cmd="/deep")

        assert "发送 /deep 开始新任务" not in rendered
        assert "重新发送 /deep" not in retry_rendered
        assert "/deep <需求描述>" in rendered
        assert "/deep <需求描述>" in retry_rendered

    def test_warning_render_fail_has_engine_cmd(self):
        assert "{engine_cmd}" in UI_TEXT["card_session_warning_render_fail"]

    def test_toasts_have_no_trailing_period(self):
        """All toast messages should not end with period for consistency."""
        toast_keys = [k for k in UI_TEXT if "toast" in k and isinstance(UI_TEXT[k], str)]
        violations = [k for k in toast_keys if UI_TEXT[k].endswith("。")]
        assert violations == [], f"Toast keys ending with period: {violations}"

    def test_system_help_tips_has_format_placeholders(self):
        """system_help_tips should contain {timeout_display} and {warn_display} for dynamic rendering."""
        text = UI_TEXT["system_help_tips"]
        assert "{timeout_display}" in text
        assert "{warn_display}" in text


# ---------------------------------------------------------------------------
# Review round 2: UX text corrections verification
# ---------------------------------------------------------------------------


class TestReviewRound2TextCorrections:
    """Verify the four UX text corrections from review round 2."""

    def test_rejected_notice_no_system_jargon(self):
        """rejected_notice must not contain '并发会话', '容量' jargon."""
        text = UI_TEXT["card_session_rejected_notice"]
        assert "并发会话" not in text
        assert "容量" not in text
        # Must contain engine_cmd placeholder
        assert "{engine_cmd}" in text
        # Verify it formats without error
        text.format(engine_cmd="/deep")

    def test_ttl_expired_concise(self):
        """ttl_expired must include recovery hint and format correctly."""
        text = UI_TEXT["card_session_ttl_expired"]
        # Generic fallback uses {expired_commands} (not engine_cmd)
        assert "{expired_commands}" in text, f"ttl_expired should include expired_commands placeholder: {text}"
        # Verify it formats without error
        text.format(expired_commands="/spec /deep")

    def test_help_tips_two_stage_close(self):
        """system_help_tips must mention advance notification before close."""
        text = UI_TEXT["system_help_tips"]
        assert "提醒" in text or "提前" in text or "通知" in text or "续期" in text
        # Verify it formats without error
        text.format(timeout_display="30 分钟", warn_display="7 分钟")

    def test_deep_error_no_detail_no_jargon(self):
        """deep_error_no_detail must not contain '无详细信息'."""
        text = UI_TEXT["deep_error_no_detail"]
        assert "无详细信息" not in text
        # Must contain engine_cmd placeholder
        assert "{engine_cmd}" in text
        # Verify it formats without error
        text.format(engine_cmd="/deep")

    def test_help_tips_includes_config_guidance(self):
        """system_help_tips should include env var hint for ops users."""
        text = UI_TEXT["system_help_tips"]
        # Should include CARD_SESSION_IDLE_TIMEOUT hint for ops adjustability
        assert "CARD_SESSION_IDLE_TIMEOUT" in text
        # Should mention close/end behavior
        assert "关闭" in text


class TestDeepErrorFallbackNoPrefix:
    """AC-20: deep_error_no_detail empty action_prefix fallback uses UI_TEXT."""

    def test_fallback_key_exists(self):
        """deep_error_fallback_no_prefix must exist in UI_TEXT."""
        assert "deep_error_fallback_no_prefix" in UI_TEXT

    def test_fallback_contains_retry_guidance(self):
        """Fallback text must contain actionable retry guidance."""
        text = UI_TEXT["deep_error_fallback_no_prefix"]
        assert "重试" in text or "重新发送" in text or "/deep" in text or "/help" in text

    def test_fallback_has_no_placeholder(self):
        """Fallback text should be a plain string with no format placeholders."""
        text = UI_TEXT["deep_error_fallback_no_prefix"]
        assert "{" not in text

class TestLockUITextPlaceholders:
    """Verify LOCK_UI_TEXT format strings use lock_undo_window_display."""

    def test_lock_help_admin_lock_cmd_format(self):
        """lock_help_admin_lock_cmd must accept lock_undo_window_display without KeyError."""
        from src.card.styles_lock import LOCK_UI_TEXT

        template = LOCK_UI_TEXT["lock_help_admin_lock_cmd"]
        result = template.format(lock_undo_window_display="5 分钟")
        assert "5 分钟" in result
        assert "/lock" in result

    def test_lock_success_lock_reply_format(self):
        """lock_success_lock_reply must accept lock_undo_window_display without KeyError."""
        from src.card.styles_lock import LOCK_UI_TEXT

        template = LOCK_UI_TEXT["lock_success_lock_reply"]
        result = template.format(lock_undo_window_display="约 2 分钟")
        assert "约 2 分钟" in result
        assert "锁定" in result
