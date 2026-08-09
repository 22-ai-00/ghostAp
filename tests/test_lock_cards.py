"""Unit tests for lock card builders (src/card/builders/lock.py)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.card.builders.lock import (
    MAX_COMMAND_TEXT_LENGTH,
    _build_p2p_multi_url,
    _compute_command_sig,
    build_chat_lock_card,
    build_lock_confirm_card,
    build_lock_success_card,
    build_repo_lock_card,
    format_elapsed_ago,
    format_lock_duration,
    format_undo_window,
    verify_command_sig,
)

# ---------------------------------------------------------------------------
# format_elapsed_ago
# ---------------------------------------------------------------------------


class TestFormatDuration:

    def test_seconds_tier(self):
        assert format_elapsed_ago(0) == "刚刚"
        assert format_elapsed_ago(4) == "刚刚"
        assert format_elapsed_ago(4.9) == "刚刚"
        assert format_elapsed_ago(5) == "5 秒前"
        assert format_elapsed_ago(30) == "30 秒前"
        assert format_elapsed_ago(59.9) == "59 秒前"

    def test_minutes_tier(self):
        assert format_elapsed_ago(60) == "1 分钟前"
        assert format_elapsed_ago(150) == "2 分钟前"
        assert format_elapsed_ago(3599) == "59 分钟前"

    def test_hours_tier(self):
        assert format_elapsed_ago(3600) == "1 小时 0 分钟前"
        assert format_elapsed_ago(3660) == "1 小时 1 分钟前"
        assert format_elapsed_ago(7200) == "2 小时 0 分钟前"
        assert format_elapsed_ago(7320) == "2 小时 2 分钟前"

    def test_days_tier(self):
        assert format_elapsed_ago(86400) == "1 天 0 小时前"
        assert format_elapsed_ago(90000) == "1 天 1 小时前"
        assert format_elapsed_ago(172800) == "2 天 0 小时前"

    def test_negative_clamped_to_zero(self):
        assert format_elapsed_ago(-10) == "刚刚"


# ---------------------------------------------------------------------------
# build_chat_lock_card
# ---------------------------------------------------------------------------


class TestBuildChatLockCard:

    def test_without_locked_by(self):
        card, _ = build_chat_lock_card()
        assert "群已锁定" in card
        assert "锁定者" not in card

    @pytest.mark.parametrize("locked_by,locked_by_name,expect_locker,expect_name,reject_raw", [
        pytest.param("ou_abcdefghijk123456", None, False, None, "ou_abcdefghijk123456", id="long_id_no_name"),
        pytest.param("ou_abc", "张三", True, "张三", "ou_abc", id="with_name"),
    ])
    def test_locked_by_variants(self, locked_by, locked_by_name, expect_locker, expect_name, reject_raw):
        kwargs = {"locked_by": locked_by}
        if locked_by_name:
            kwargs["locked_by_name"] = locked_by_name
        card, _ = build_chat_lock_card(**kwargs)
        if expect_locker:
            assert "锁定者" in card
        else:
            assert "锁定者" not in card
        if expect_name:
            assert expect_name in card
        assert reject_raw not in card
        assert "Bot 管理员" in card

    def test_readonly_commands_listed(self):
        card, _ = build_chat_lock_card()
        assert "/help" in card
        assert "/status" in card
        assert "等命令" in card

    def test_admin_name_in_contact(self):
        card, _ = build_chat_lock_card(admin_name="李四")
        assert "李四" in card
        assert "请联系 李四 执行 `/unlock` 解锁" in card
        # F-08: card should use friendly unlock wording with /unlock command
        card2, _ = build_chat_lock_card(admin_name="王五")
        assert "执行 `/unlock` 解锁" in card2

    def test_default_contact_without_admin_name(self):
        card, _ = build_chat_lock_card()
        assert "Bot 管理员执行 `/unlock` 解锁" in card
        # Empty string admin_name should also use same fallback
        card2, _ = build_chat_lock_card(admin_name="")
        assert "Bot 管理员执行 `/unlock` 解锁" in card2

    def test_unlock_command_in_card(self):
        """AC-17: card should tell non-admins to ask admin to run /unlock."""
        card, _ = build_chat_lock_card()
        assert "/unlock" in card

    def test_auto_unlock_hint_with_wall_time(self):
        """AC-22: Chat lock card shows auto-unlock countdown using format_friendly_duration."""
        import time
        # Locked 1 hour ago; default max_duration=86400s → ~23h remaining
        card, _ = build_chat_lock_card(locked_at_wall=time.time() - 3600)
        assert "自动解除" in card
        assert "小时" in card
        assert "约" in card  # format_friendly_duration output always has "约" prefix for > 60s

    def test_app_id_generates_deeplink_button(self):
        _, buttons = build_chat_lock_card(app_id="cli_test_123")
        deep_btns = [b for b in buttons if "multi_url" in b]
        assert len(deep_btns) == 1  # single "去私聊操作" button

    def test_locked_at_wall_conditional_time_display(self):
        """AC-R06: When locked_at_wall is provided, card contains '锁定时间'; otherwise not."""
        import time
        now = time.time()
        card_with, _ = build_chat_lock_card(locked_by_name="张三", locked_at_wall=now)
        assert "锁定时间" in card_with
        card_without, _ = build_chat_lock_card(locked_by_name="张三")
        assert "锁定时间" not in card_without


# ---------------------------------------------------------------------------
# build_repo_lock_card
# ---------------------------------------------------------------------------


class TestBuildRepoLockCard:

    def test_force_release_button_admin(self):
        import time
        _, buttons = build_repo_lock_card(
            "/home/user/my-repo", time.monotonic() - 10,
            is_admin=True, repo_token="abc123def456",
        )
        assert len(buttons) == 1
        btn = buttons[0]
        assert btn["tag"] == "button"
        assert btn["type"] == "danger"
        assert btn["value"]["action"] == "force_release_repo_lock"
        assert btn["value"]["_tk"] == "abc123def456"
        # Must NOT contain root_path (security: no filesystem path leakage)
        assert "root_path" not in btn["value"]

    def test_force_release_button_non_admin(self):
        import time
        _, buttons = build_repo_lock_card("/home/user/my-repo", time.monotonic() - 10)
        assert len(buttons) == 0

    def test_duration_display(self):
        import time
        markdown, _ = build_repo_lock_card("/tmp/repo", time.monotonic() - 7200)
        # Should show hours
        assert "小时" in markdown


# ---------------------------------------------------------------------------
# build_lock_success_card
# ---------------------------------------------------------------------------


class TestBuildLockSuccessCard:

    def test_lock_success(self):
        result = build_lock_success_card("lock")
        # F-19: lock+reply now returns (markdown, undo_buttons) tuple
        assert isinstance(result, tuple)
        card, buttons = result
        assert "群已锁定" in card
        assert "/unlock" in card

    def test_lock_idempotent(self):
        card = build_lock_success_card("lock", message="该群已处于锁定状态")
        assert "已处于锁定状态" in card
        assert "无需重复" in card

    def test_unlock_success(self):
        card = build_lock_success_card("unlock")
        assert "群已解锁" in card

    def test_unlock_idempotent(self):
        card = build_lock_success_card("unlock", message="该群当前未锁定")
        assert "未锁定" in card
        assert "无需解锁" in card

    def test_lock_broadcast_variant(self):
        result = build_lock_success_card("lock", variant="broadcast")
        # Now returns (markdown, buttons) tuple
        assert isinstance(result, tuple)
        md, buttons = result
        assert "锁定状态" in md
        assert "非 Bot 管理员" in md
        # broadcast should NOT contain admin-action hints
        assert "/unlock" not in md
        # Should have at least the status button
        assert any(b.get("value", {}).get("_t") == "/status" for b in buttons)

    def test_unlock_reply_broadcast_consistent(self):
        """AC-21: unlock reply and broadcast use identical wording for member ops."""
        import re
        reply_card = build_lock_success_card("unlock")
        broadcast_result = build_lock_success_card("unlock", variant="broadcast")
        broadcast_card = broadcast_result[0] if isinstance(broadcast_result, tuple) else broadcast_result
        # Both must contain the same "成员...操作" phrase
        pattern = r"所有成员现在可正常操作"
        assert re.search(pattern, reply_card), f"reply missing pattern: {reply_card}"
        assert re.search(pattern, broadcast_card), f"broadcast missing pattern: {broadcast_card}"


# ---------------------------------------------------------------------------
# build_repo_lock_card — retry button
# ---------------------------------------------------------------------------


class TestBuildRepoLockCardRetryButton:

    def test_retry_button_present_when_command_text(self):
        import time
        _, buttons = build_repo_lock_card(
            "/home/user/repo", time.monotonic() - 10,
            command_text="/deep fix bug",
        )
        retry_buttons = [b for b in buttons if b["value"]["action"] == "retry_command"]
        assert len(retry_buttons) == 1
        assert retry_buttons[0]["value"]["_t"] == "/deep fix bug"
        assert retry_buttons[0]["type"] == "primary"

    def test_retry_button_absent_when_no_command_text(self):
        import time
        _, buttons = build_repo_lock_card("/home/user/repo", time.monotonic() - 10)
        retry_buttons = [b for b in buttons if b.get("value", {}).get("action") == "retry_command"]
        assert len(retry_buttons) == 0

    def test_retry_and_force_release_both_present(self):
        import time
        _, buttons = build_repo_lock_card(
            "/home/user/repo", time.monotonic() - 10,
            is_admin=True, command_text="/deep test",
            repo_token="tok123",
        )
        actions = [b["value"]["action"] for b in buttons]
        assert actions == ["retry_command", "force_release_repo_lock"]

    def test_force_release_button_requires_token(self):
        """Admin without repo_token → no force-release button."""
        import time
        _, buttons = build_repo_lock_card(
            "/home/user/repo", time.monotonic() - 10,
            is_admin=True, repo_token="",
        )
        force_btns = [b for b in buttons if b.get("value", {}).get("action") == "force_release_repo_lock"]
        assert len(force_btns) == 0


# ---------------------------------------------------------------------------
# build_repo_lock_card — active/idle status hint (AC-20)
# ---------------------------------------------------------------------------


class TestBuildRepoLockCardActiveIdleHint:
    """AC-20: build_repo_lock_card differentiates active vs idle holder."""

    def test_active_holder_hint(self):
        """When last_active_time < 60s ago, show '对方正在操作中' with timeout info."""
        import time
        now = time.monotonic()
        markdown, _ = build_repo_lock_card(
            "/tmp/repo", now - 120,
            last_active_time_monotonic=now - 10,  # active 10s ago
        )
        assert "正在操作中" in markdown
        assert "分钟" in markdown  # timeout_min is included

    def test_idle_holder_hint(self):
        """When last_active_time > 60s ago, show idle status."""
        import time
        now = time.monotonic()
        markdown, _ = build_repo_lock_card(
            "/tmp/repo", now - 300,
            last_active_time_monotonic=now - 180,  # idle for 3 min
        )
        assert "暂无新操作" in markdown
        assert "分钟" in markdown

    def test_no_active_time_shows_general_hint(self):
        """When last_active_time is 0, show general auto-release hint."""
        import time
        markdown, _ = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 60,
            last_active_time_monotonic=0.0,
        )
        assert "无操作" in markdown
        assert "自动释放" in markdown


# ---------------------------------------------------------------------------
# build_repo_lock_card — deep link button (AC-24)
# ---------------------------------------------------------------------------


class TestBuildRepoLockCardDeepLink:
    """AC-24: build_repo_lock_card generates a private-chat deep link button."""

    def test_deep_link_button_conditional_on_app_id(self):
        import time
        _, buttons = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10,
            app_id="cli_test_app_123",
        )
        deep_btns = [b for b in buttons if "multi_url" in b]
        assert len(deep_btns) == 1
        btn = deep_btns[0]
        assert "applink" in btn["multi_url"]["url"]
        assert "cli_test_app_123" in btn["multi_url"]["url"]
        assert btn["multi_url"]["url"].startswith("https://")
        assert btn["multi_url"]["android_url"].startswith("lark://")

        _, buttons2 = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10, app_id="",
        )
        assert not [b for b in buttons2 if "multi_url" in b]

    def test_deep_link_in_markdown_text(self):
        import time
        markdown, _ = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10,
            app_id="cli_test_app_123",
        )
        assert "lark://applink/client/bot/open" in markdown
        assert "私聊" in markdown


# ---------------------------------------------------------------------------
# AC-19: P2P fallback note in repo lock card
# ---------------------------------------------------------------------------


class TestP2PFallbackNoteInRepoCard:
    """AC-19: repo lock card includes P2P fallback note when app_id is present."""

    def test_fallback_note_conditional_on_app_id(self):
        import time

        from src.card.styles_lock import LOCK_UI_TEXT

        markdown_with, _ = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10, app_id="cli_test_123",
        )
        assert LOCK_UI_TEXT["repo_lock_p2p_fallback_note"] in markdown_with

        markdown_without, _ = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10, app_id="",
        )
        assert LOCK_UI_TEXT["repo_lock_p2p_fallback_note"] not in markdown_without


# ---------------------------------------------------------------------------
# BaseHandler.send_lock_conflict_card
# ---------------------------------------------------------------------------


class TestSendLockConflictCard:
    """Tests for the centralised send_lock_conflict_card helper on BaseHandler."""

    def _make_handler(self):
        from unittest.mock import MagicMock

        ctx = MagicMock()
        ctx.settings.app_id = "test"
        ctx.settings.app_secret = "test"
        ctx.api_client_factory = MagicMock()

        from src.feishu.handlers.base import BaseHandler
        handler = BaseHandler(ctx)
        handler.im_client = MagicMock()
        handler.reply_card = MagicMock()
        handler.reply_text = MagicMock()
        return handler, ctx

    def test_sends_card_on_normal_path(self):
        import time

        from src.repo_lock import LockConflictError

        handler, ctx = self._make_handler()
        # Setup ctx managers
        ctx.chat_lock_manager.is_admin.return_value = False
        ctx.repo_lock_manager.path_to_token.return_value = "tok_abc"

        err = LockConflictError(
            "conflict",
            holder_chat_id="chat_other",
            locked_since=time.monotonic() - 60,
            root_path="/home/user/repo",
        )
        handler.send_lock_conflict_card(err, "msg_123", "/deep fix bug")

        handler.reply_card.assert_called_once()
        call_args = handler.reply_card.call_args
        assert call_args[0][0] == "msg_123"  # message_id

    def test_admin_gets_card_with_lock_info(self):
        import json
        import time
        from unittest.mock import patch

        from src.repo_lock import LockConflictError

        handler, ctx = self._make_handler()
        ctx.chat_lock_manager.is_admin.return_value = True
        ctx.repo_lock_manager.path_to_token.return_value = "tok_xyz"

        err = LockConflictError(
            "conflict",
            holder_chat_id="chat_other",
            locked_since=time.monotonic() - 30,
            root_path="/tmp/repo",
        )
        with patch("src.thread.get_current_sender_id", return_value="ou_admin"):
            handler.send_lock_conflict_card(err, "msg_456", "/deep test")

        handler.reply_card.assert_called_once()
        call_args = handler.reply_card.call_args
        assert call_args[0][0] == "msg_456"
        # Card body should contain lock info
        card_str = call_args[0][1]
        card = json.loads(card_str)
        body_text = json.dumps(card, ensure_ascii=False)
        assert "仓库锁定" in body_text

    def test_no_ctx_managers_still_works(self):
        """When chat_lock_manager / repo_lock_manager are None, card is still sent."""
        import time

        from src.repo_lock import LockConflictError

        handler, ctx = self._make_handler()
        ctx.chat_lock_manager = None
        ctx.repo_lock_manager = None

        err = LockConflictError(
            "conflict", holder_chat_id="c", locked_since=time.monotonic(), root_path="/r",
        )
        handler.send_lock_conflict_card(err, "msg_789", "test cmd")

        handler.reply_card.assert_called_once()

    def test_does_not_raise_on_internal_error(self):
        """If card building fails internally, fallback text is sent (AC-17)."""
        import time
        from unittest.mock import MagicMock, patch

        from src.repo_lock import LockConflictError

        handler, ctx = self._make_handler()
        # Make reply_message a fresh mock to track calls
        handler.reply_text = MagicMock()

        err = LockConflictError(
            "conflict", holder_chat_id="c", locked_since=time.monotonic(), root_path="/r",
        )

        # Patch at the source module so the lazy import picks up the mock
        with patch(
            "src.card.builders.lock.build_repo_lock_card",
            side_effect=RuntimeError("build failed"),
        ):
            # Should NOT raise
            handler.send_lock_conflict_card(err, "msg_err", "cmd")

        # Fallback text should have been sent
        handler.reply_text.assert_called_once()
        fallback_text = handler.reply_text.call_args[0][1]
        assert "🔒" in fallback_text
        assert "仓库被占用" in fallback_text

    def test_fallback_text_contains_lock_emoji(self):
        """AC-17: when both card and fallback fail, no exception propagates."""
        import time
        from unittest.mock import MagicMock, patch

        from src.repo_lock import LockConflictError

        handler, ctx = self._make_handler()
        # Make reply_message always fail
        handler.reply_text = MagicMock(side_effect=RuntimeError("all sends fail"))

        err = LockConflictError(
            "conflict", holder_chat_id="c", locked_since=time.monotonic(), root_path="/r",
        )

        with patch(
            "src.card.builders.lock.build_repo_lock_card",
            side_effect=RuntimeError("build failed"),
        ):
            # Should NOT raise even when fallback also fails
            handler.send_lock_conflict_card(err, "msg_fb", "cmd")


# ---------------------------------------------------------------------------
# build_repo_lock_card — command_text truncation (F-12)
# ---------------------------------------------------------------------------


class TestRepoLockCardCommandTruncation:
    """F-12: retry button omitted when command_text exceeds MAX_COMMAND_TEXT_LENGTH."""

    def test_short_command_has_retry_button(self):
        import time
        _, buttons = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10,
            command_text="/deep fix bug",
        )
        retry_btns = [b for b in buttons if b.get("value", {}).get("action") == "retry_command"]
        assert len(retry_btns) == 1

    def test_long_command_no_retry_button(self):
        import time
        long_cmd = "/deep " + "x" * (MAX_COMMAND_TEXT_LENGTH + 1)
        markdown, buttons = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10,
            command_text=long_cmd,
        )
        retry_btns = [b for b in buttons if b.get("value", {}).get("action") == "retry_command"]
        assert len(retry_btns) == 0

    def test_long_command_shows_manual_hint(self):
        """AC-24: when command_text > MAX_COMMAND_TEXT_LENGTH chars, card shows manual resend hint."""
        import time
        long_cmd = "A" * (MAX_COMMAND_TEXT_LENGTH + 1)
        markdown, _ = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10,
            command_text=long_cmd,
        )
        assert "手动重新发送" in markdown

    def test_exact_limit_has_retry_button(self):
        import time
        exact_cmd = "x" * MAX_COMMAND_TEXT_LENGTH
        _, buttons = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10,
            command_text=exact_cmd,
        )
        retry_btns = [b for b in buttons if b.get("value", {}).get("action") == "retry_command"]
        assert len(retry_btns) == 1


# ---------------------------------------------------------------------------
# build_lock_confirm_card — wording (F-09, F-10)
# ---------------------------------------------------------------------------


class TestLockConfirmCardDeprecated:
    """build_lock_confirm_card is deprecated and returns a stub."""

    def test_returns_deprecation_message_with_empty_buttons(self):
        markdown, buttons = build_lock_confirm_card("chat_test")
        assert "请重新发送 /lock" in markdown
        assert buttons == []
        _, buttons2 = build_lock_confirm_card("chat_test", confirm_timeout=60)
        assert buttons2 == []


# ---------------------------------------------------------------------------
# build_repo_lock_card — friendly idle text (F-13)
# ---------------------------------------------------------------------------


class TestRepoLockCardFriendlyIdleText:
    """F-13: idle wording uses '自动释放' for friendly auto-release text."""

    def test_idle_uses_friendly_text(self):
        import time
        now = time.monotonic()
        markdown, _ = build_repo_lock_card(
            "/tmp/repo", now - 300,
            last_active_time_monotonic=now - 180,
        )
        assert "自动释放" in markdown


# ======================================================================
# F-12: Help card lock section conditional display
# ======================================================================

class TestHelpCardLockSection:
    """F-12: Lock section in help card shown only when lock_enabled=True."""

    def test_lock_section_conditional_on_enabled(self):
        from src.card.builders.system import SystemBuilder
        SystemBuilder._build_help_card_cached.cache_clear()
        _, card_json_off = SystemBuilder.build_help_card(lock_enabled=False, session_idle_timeout=600, session_idle_warn_at_remaining=120, lock_undo_window_seconds=300)
        assert "群锁定" not in card_json_off
        SystemBuilder._build_help_card_cached.cache_clear()
        _, card_json_on = SystemBuilder.build_help_card(lock_enabled=True, session_idle_timeout=600, session_idle_warn_at_remaining=120, lock_undo_window_seconds=300)
        assert "群锁定" in card_json_on

    def test_lock_section_title_admin_vs_non_admin(self):
        """AC-16/AC-15: admin sees 'Bot 管理员专属'; non-admin does not."""
        from src.card.builders.system import SystemBuilder
        SystemBuilder._build_help_card_cached.cache_clear()
        _, card_json_admin = SystemBuilder.build_help_card(lock_enabled=True, is_admin=True, session_idle_timeout=600, session_idle_warn_at_remaining=120, lock_undo_window_seconds=300)
        assert "Bot 管理员专属" in card_json_admin
        SystemBuilder._build_help_card_cached.cache_clear()
        _, card_json_non = SystemBuilder.build_help_card(lock_enabled=True, is_admin=False, session_idle_timeout=600, session_idle_warn_at_remaining=120, lock_undo_window_seconds=300)
        assert "群锁定" in card_json_non
        assert "Bot 管理员专属" not in card_json_non

    def test_admin_help_shows_grouped_commands(self):
        """Admin help body groups commands under management and exempt headers."""
        from src.card.builders.system import SystemBuilder
        SystemBuilder._build_help_card_cached.cache_clear()
        _, card_json = SystemBuilder.build_help_card(lock_enabled=True, is_admin=True, session_idle_timeout=600, session_idle_warn_at_remaining=120, lock_undo_window_seconds=300)
        assert "管理命令" in card_json
        assert "锁定期间仍可使用的命令" in card_json
        assert "直接执行" in card_json


# ---------------------------------------------------------------------------
# build_lock_reclaim_notify_card
# ---------------------------------------------------------------------------


class TestBuildLockReclaimNotifyCard:
    """Tests for the unified lock reclaim notification builder."""

    def test_hard_timeout_reason(self):
        from src.card.builders.lock import build_lock_reclaim_notify_card
        text = build_lock_reclaim_notify_card("my-repo", reason="hard_timeout")
        assert "my-repo" in text
        assert "系统回收" in text or "超时" in text
        assert "小时" in text  # max_hours is included
        # Default reason is also hard_timeout
        text2 = build_lock_reclaim_notify_card("test-repo")
        assert "test-repo" in text2
        assert "系统回收" in text2 or "超时" in text2

    def test_force_release_reason(self):
        from src.card.builders.lock import build_lock_reclaim_notify_card
        text = build_lock_reclaim_notify_card("my-repo", reason="force_release")
        assert "my-repo" in text
        assert "管理员" in text or "释放" in text


# ---------------------------------------------------------------------------
# build_force_release_confirm_card (F-22)
# ---------------------------------------------------------------------------


class TestBuildForceReleaseConfirmCard:
    """F-22: build_force_release_confirm_card produces confirm/cancel buttons."""

    def test_basic_structure_and_token(self):
        from src.card.builders.lock import build_force_release_confirm_card
        md, buttons = build_force_release_confirm_card("tok_abc", "my-repo")
        assert "确认" in md
        assert "my-repo" in md
        assert len(buttons) == 2
        actions = [b["value"]["action"] for b in buttons]
        assert "confirm_force_release" in actions
        assert "cancel_force_release" in actions
        confirm_btn = [b for b in buttons if b["value"]["action"] == "confirm_force_release"][0]
        assert confirm_btn["value"]["_tk"] == "tok_abc"
        assert "_ts" in confirm_btn["value"]

    def test_holder_hint_included(self):
        from src.card.builders.lock import build_force_release_confirm_card
        md, _ = build_force_release_confirm_card("tok_abc", "my-repo", holder_hint="对方已空闲 3 分钟")
        assert "空闲" in md

    def test_hcid_embedded_when_provided(self):
        """F-01: _hcid is embedded in confirm button value."""
        from src.card.builders.lock import build_force_release_confirm_card
        _, buttons = build_force_release_confirm_card(
            "tok_abc", "my-repo", holder_chat_id="chat_holder_123",
        )
        confirm_btn = [b for b in buttons if b["value"]["action"] == "confirm_force_release"][0]
        assert confirm_btn["value"]["_hcid"] == "chat_holder_123"
        # Without holder_chat_id, _hcid is not present (backward compat)
        _, buttons2 = build_force_release_confirm_card("tok_abc", "my-repo")
        confirm_btn2 = [b for b in buttons2 if b["value"]["action"] == "confirm_force_release"][0]
        assert "_hcid" not in confirm_btn2["value"]


# ---------------------------------------------------------------------------
# AC-15: Card action throttled branch sends text reply
# ---------------------------------------------------------------------------


class TestAC15ThrottledTextReply:
    """Verify that the throttled card-action branch in ws_client sends a text reply."""

    def test_throttled_branch_calls_throttled_reply(self):
        """When card action is blocked AND throttled, ChatLockGate delegates to handler.send_chat_lock_throttled_reply."""
        from unittest.mock import MagicMock

        from src.feishu.chat_lock_gate import ChatLockGate
        from src.feishu.message_cache import MessageCache

        mock_clm = MagicMock()
        mock_clm.should_block_card_action.return_value = True

        mock_handler = MagicMock()
        host = MagicMock()
        host._get_handler.return_value = mock_handler

        cache = MessageCache(ttl=30, max_size=10_000, cleanup_interval=60)
        gate = ChatLockGate(chat_lock_manager=mock_clm, dedup_cache=cache, host=host)

        # First call consumes the dedup slot
        gate._should_send_intercept("chat_001", "user_001")
        # Now the throttled branch should fire
        result = gate._try_block(
            "chat_001", "user_001", "msg_001",
            is_card_action=True, action_type="some_action",
        )

        assert result is True
        mock_handler.send_chat_lock_throttled_reply.assert_called_once()


class TestThrottledReplyContainsName:
    """AC-R08: throttled reply text must include locker name and be non-empty."""

    @pytest.mark.parametrize("name", ["张三", "Bot 管理员"])
    def test_throttled_reply_contains_formatted_name(self, name):
        """Formatted throttled reply must match '群已被 .+ 锁定' regex and be non-empty."""
        import re

        from src.card.ui_text import UI_TEXT
        assert UI_TEXT["chat_locked_throttled_reply"]
        assert isinstance(UI_TEXT["chat_locked_throttled_reply"], str)
        text = UI_TEXT["chat_locked_throttled_reply"].format(name=name)
        assert re.search(r"群已被 .+ 锁定", text), f"Text did not match: {text}"


# ---------------------------------------------------------------------------
# AC-16: retry_command lock interception passes app_id
# ---------------------------------------------------------------------------


class TestAC16RetryCommandAppId:
    """Verify that retry_command's lock interception passes app_id to build_chat_lock_card."""

    def test_build_chat_lock_card_accepts_app_id(self):
        """build_chat_lock_card must accept app_id keyword argument and produce deeplink button."""
        md, btns = build_chat_lock_card(
            locked_by="user_x",
            locked_by_name="Test User",
            app_id="cli_test_app_id",
        )
        # When app_id is provided, a "去私聊" button with deeplink should exist
        p2p_btns = [b for b in btns if "multi_url" in b]
        assert len(p2p_btns) == 1
        assert "cli_test_app_id" in p2p_btns[0]["multi_url"]["url"]

    def test_build_chat_lock_card_without_or_empty_app_id_no_deeplink(self):
        """Without app_id or empty app_id, no deeplink button is generated."""
        _, btns1 = build_chat_lock_card(locked_by="user_x", locked_by_name="Test User")
        assert not [b for b in btns1 if "multi_url" in b]
        _, btns2 = build_chat_lock_card(locked_by="user_x", locked_by_name="Test User", app_id="")
        assert not [b for b in btns2 if "multi_url" in b]


# ---------------------------------------------------------------------------
# AC-18: Expiry button wording consistency
# ---------------------------------------------------------------------------


class TestAC18ExpiryButtonWording:
    """Verify that expiry button uses '再次尝试释放' not '重新释放'."""

    def test_retry_force_release_and_expired_title(self):
        from src.card.ui_text import UI_TEXT
        val = UI_TEXT["lock_btn_retry_force_release"]
        assert "再次尝试释放" in val
        assert "重新释放" not in val
        assert "lock_force_release_expired_title" in UI_TEXT
        assert "过期" in UI_TEXT["lock_force_release_expired_title"]


# ---------------------------------------------------------------------------
# AC-19: No duplicate hardcoded strings in system.py
# ---------------------------------------------------------------------------


class TestAC19NoDuplicateHardcodedStrings:
    """Verify that previously duplicated strings in system.py now use UI_TEXT."""

    def test_known_duplicates_in_ui_text_and_no_raw_in_system(self):
        """The 5 known duplicate string pairs must exist as UI_TEXT keys and not as raw literals in system.py."""
        import ast
        from pathlib import Path

        from src.card.ui_text import UI_TEXT
        expected_keys = [
            "lock_force_release_admin_only",
            "lock_repo_mgr_not_init",
            "lock_repo_already_released",
            "lock_repo_path_not_found",
            "lock_cmd_confirm_expired_msg",
        ]
        for key in expected_keys:
            assert key in UI_TEXT, f"UI_TEXT missing key: {key}"
            assert UI_TEXT[key], f"UI_TEXT['{key}'] is empty"

        source = Path("src/feishu/handlers/system.py").read_text()
        tree = ast.parse(source)
        raw_strings: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                raw_strings.append(node.value)
        banned = [
            "权限不足：仅 Bot 管理员可强制释放仓库锁",
            "仓库锁管理器未初始化",
            "该仓库锁已被释放",
            "未找到对应仓库路径",
        ]
        for s in banned:
            matches = [r for r in raw_strings if s in r]
            assert len(matches) == 0, f"Found raw hardcoded string in system.py: {s!r}"


# ---------------------------------------------------------------------------
# HMAC-SHA256 command signature (security hardening)
# ---------------------------------------------------------------------------


class TestVerifyCommandSig:
    """Verify verify_command_sig with HMAC + empty-sig rejection."""

    def test_valid_sig_passes(self):
        from unittest.mock import patch

        with patch("src.utils.signing._get_signing_key", return_value="secret"):
            sig = _compute_command_sig("test_cmd")
            assert bool(verify_command_sig("test_cmd", sig)) is True

    def test_tampered_text_rejected(self):
        from unittest.mock import patch

        with patch("src.utils.signing._get_signing_key", return_value="secret"):
            sig = _compute_command_sig("original_cmd")
            assert bool(verify_command_sig("tampered_cmd", sig)) is False

    def test_empty_sig_rejected(self):
        assert bool(verify_command_sig("any_command", "")) is False

    def test_wrong_sig_rejected(self):
        assert bool(verify_command_sig("cmd", "deadbeef" * 8)) is False

    def test_legacy_sha256_sig_accepted_within_window(self):
        """Plain SHA-256 signature is accepted when within the compat window."""
        import hashlib
        from datetime import date
        from unittest.mock import MagicMock

        cmd = "/status"
        plain_sig = hashlib.sha256(cmd.encode("utf-8")).hexdigest()

        mock_settings = MagicMock()
        mock_settings.app_secret = "real_secret"
        mock_settings.sig_compat_deploy_date = date.today().isoformat()
        mock_settings.sig_compat_window_days = 7

        with patch("src.utils.signing._get_signing_key", return_value="real_secret"), \
             patch("src.config.get_settings", return_value=mock_settings):
            # HMAC sig won't match plain sig, so fallback should kick in
            # _verify_legacy_sha256_fallback does `from src.config import get_settings`
            assert bool(verify_command_sig(cmd, plain_sig)) is True

    def test_legacy_sha256_sig_rejected_outside_window(self):
        """Plain SHA-256 signature is rejected when outside the compat window."""
        import hashlib
        from datetime import date, timedelta
        from unittest.mock import MagicMock

        cmd = "/status"
        plain_sig = hashlib.sha256(cmd.encode("utf-8")).hexdigest()

        mock_settings = MagicMock()
        mock_settings.app_secret = "real_secret"
        # Set deploy date far in the past so the window has expired
        past_date = date.today() - timedelta(days=30)
        mock_settings.sig_compat_deploy_date = past_date.isoformat()
        mock_settings.sig_compat_window_days = 7

        with patch("src.utils.signing._get_signing_key", return_value="real_secret"), \
             patch("src.config.get_settings", return_value=mock_settings):
            assert bool(verify_command_sig(cmd, plain_sig)) is False


class TestCardBuildersIncludeSig:
    """Ensure all retry_command buttons embed command_sig."""

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_build_repo_lock_card_includes_sig(self, _mock_key):
        import time
        _, buttons = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10,
            command_text="/deep fix bug",
        )
        retry_btns = [b for b in buttons if b.get("value", {}).get("action") == "retry_command"]
        assert len(retry_btns) == 1
        assert "_s" in retry_btns[0]["value"]
        assert retry_btns[0]["value"]["_s"]  # non-empty

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_build_lock_success_card_broadcast_includes_sig(self, _mock_key):
        result = build_lock_success_card("lock", variant="broadcast")
        assert isinstance(result, tuple)
        _, buttons = result
        retry_btns = [b for b in buttons if b.get("value", {}).get("action") == "retry_command"]
        assert len(retry_btns) == 1
        assert "_s" in retry_btns[0]["value"]
        assert retry_btns[0]["value"]["_s"]  # non-empty

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_repo_lock_card_sig_is_verifiable(self, _mock_key):
        """The embedded sig must pass verify_command_sig."""
        import time
        _, buttons = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10,
            command_text="/deep run tests",
        )
        retry_btn = [b for b in buttons if b.get("value", {}).get("action") == "retry_command"][0]
        cmd = retry_btn["value"]["_t"]
        sig = retry_btn["value"]["_s"]
        assert bool(verify_command_sig(cmd, sig)) is True


# ---------------------------------------------------------------------------
# _build_p2p_multi_url (Task 27: AC-28 https scheme)
# ---------------------------------------------------------------------------


class TestBuildP2PMultiUrl:
    """AC-28: _build_p2p_multi_url generates correct URL schemes."""

    def test_structure_and_schemes(self):
        result = _build_p2p_multi_url("app123")
        assert set(result.keys()) == {"url", "pc_url", "android_url", "ios_url"}
        assert result["url"].startswith("https://")
        assert result["pc_url"].startswith("https://")
        assert result["android_url"].startswith("lark://")
        assert result["ios_url"].startswith("lark://")

    def test_app_id_in_all_urls(self):
        result = _build_p2p_multi_url("my_app_id")
        for key in ("url", "pc_url", "android_url", "ios_url"):
            assert "my_app_id" in result[key], f"app_id missing in {key}"

    def test_feishu_applink_domain(self):
        result = _build_p2p_multi_url("test")
        assert "applink.feishu.cn" in result["url"]
        assert "applink.feishu.cn" in result["pc_url"]


# ---------------------------------------------------------------------------
# format_lock_duration (Task 29)
# ---------------------------------------------------------------------------


class TestFormatLockDuration:
    """format_lock_duration returns human-readable lock duration."""

    def test_seconds_and_minutes(self):
        import time as _time
        result = format_lock_duration(_time.monotonic() - 30)
        assert "30 秒" in result
        assert "已锁定" in result
        result2 = format_lock_duration(_time.monotonic() - 150)
        assert "2 分钟" in result2

    def test_hours_and_minutes(self):
        import time as _time
        result = format_lock_duration(_time.monotonic() - 7500)  # 2h 5m
        assert "2 小时" in result
        assert "5 分钟" in result

    def test_zero_and_future_clamp(self):
        import time as _time
        result = format_lock_duration(_time.monotonic())
        assert "0 秒" in result
        result2 = format_lock_duration(_time.monotonic() + 1000)
        assert "0 秒" in result2


# ---------------------------------------------------------------------------
# AC-21: Retry button always shows "🔄 重试" without count suffix
# ---------------------------------------------------------------------------


class TestRetryButtonNoCount:

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_first_retry_no_count(self, _mock_key):
        import time
        md, buttons = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10, command_text="/deep run"
        )
        retry_btns = [b for b in buttons if b.get("value", {}).get("action") == "retry_command"]
        assert len(retry_btns) == 1
        assert retry_btns[0]["text"]["content"] == "🔄 重试"
        # First conflict: no "仓库仍被占用" hint
        assert "仓库仍被占用" not in md

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_subsequent_retry_shows_still_occupied(self, _mock_key):
        import time
        md, buttons = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10, command_text="/deep run", retry_count=3
        )
        retry_btns = [b for b in buttons if b.get("value", {}).get("action") == "retry_command"]
        assert len(retry_btns) == 1
        assert retry_btns[0]["text"]["content"] == "🔄 重试"
        # Backend: retry_count still incremented in value payload
        assert retry_btns[0]["value"]["_rc"] == 4
        # AC-20: retry card shows "仓库仍被占用" but no retry count (UX cleanup)
        assert "仓库仍被占用" in md
        assert "第 3 次重试" not in md


# ---------------------------------------------------------------------------
# AC-20: Broadcast card includes /help and /status hint
# ---------------------------------------------------------------------------


class TestBroadcastCardHint:

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_broadcast_contains_help_status_hint(self, _mock_key):
        result = build_lock_success_card("lock", variant="broadcast")
        assert isinstance(result, tuple)
        md, _ = result
        assert "/help" in md
        assert "/status" in md

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_unlock_broadcast_no_hint(self, _mock_key):
        """Unlock broadcast should NOT contain the lock broadcast hint."""
        result = build_lock_success_card("unlock", variant="broadcast")
        md = result[0] if isinstance(result, tuple) else result
        assert "发送 `/help` 或 `/status` 查看详情" not in md


# ---------------------------------------------------------------------------
# AC-15: Chat lock card admin entry
# ---------------------------------------------------------------------------


class TestChatLockCardAdminEntry:

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_admin_name_shows_p2p_button(self, _mock_key):
        """With app_id, P2P button always shows (no misleading 'contact admin' button)."""
        _, buttons = build_chat_lock_card(
            locked_by="ou_abc123", locked_by_name="张三",
            admin_name="张三", app_id="cli_test"
        )
        p2p_btns = [b for b in buttons if "multi_url" in b]
        assert len(p2p_btns) == 1
        assert "去私聊" in p2p_btns[0]["text"]["content"]
        # Old misleading "联系 {admin_name}" button must NOT appear
        contact_btns = [b for b in buttons if "联系" in b.get("text", {}).get("content", "")]
        assert len(contact_btns) == 0

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_no_admin_name_shows_contact_admin(self, _mock_key):
        md, buttons = build_chat_lock_card(
            locked_by="ou_abc123", admin_name="", app_id="cli_test"
        )
        assert "Bot 管理员执行 `/unlock` 解锁" in md
        assert "请在群内询问" not in md
        # P2P button still shows (independent of admin_name)
        p2p_btns = [b for b in buttons if "multi_url" in b]
        assert len(p2p_btns) == 1

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_no_app_id_no_p2p_button(self, _mock_key):
        """Without app_id, P2P buttons should not appear even with admin_name."""
        _, buttons = build_chat_lock_card(
            locked_by="ou_abc123", locked_by_name="李四",
            admin_name="李四", app_id=""
        )
        p2p_btns = [b for b in buttons if "multi_url" in b]
        assert len(p2p_btns) == 0


# ---------------------------------------------------------------------------
# AC-17: Lock success undo button
# ---------------------------------------------------------------------------


class TestLockSuccessUndoButton:

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_undo_button_present(self, _mock_key):
        result = build_lock_success_card("lock")
        assert isinstance(result, tuple)
        md, buttons = result
        assert len(buttons) == 1
        btn = buttons[0]
        assert btn["value"]["action"] == "retry_command"
        assert btn["value"]["_t"] == "/unlock"
        assert btn["value"]["_ul"] is True
        assert "_ue" in btn["value"]
        assert "_s" in btn["value"]

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_undo_button_label(self, _mock_key):
        _, buttons = build_lock_success_card("lock")
        assert "撤销" in buttons[0]["text"]["content"]

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_undo_expires_is_future(self, _mock_key):
        import time
        _, buttons = build_lock_success_card("lock")
        expires = buttons[0]["value"]["_ue"]
        assert expires > time.time()
        assert expires <= time.time() + 310

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_idempotent_lock_no_undo(self, _mock_key):
        """Idempotent lock (message set) returns plain str, no undo button."""
        result = build_lock_success_card("lock", message="已锁定")
        assert isinstance(result, str)

    @patch("src.utils.signing._get_signing_key", return_value="test_secret_key")
    def test_unlock_reply_no_undo(self, _mock_key):
        """Unlock reply should not have undo button."""
        result = build_lock_success_card("unlock")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# AC-19: Config sanitization
# ---------------------------------------------------------------------------


class TestConfigSanitization:

    def test_no_admin_user_ids_or_dotenv_in_lock_ui_text(self):
        from src.card.styles_lock import LOCK_UI_TEXT
        _exempt_admin = {"chat_lock_no_admin_config", "chat_lock_no_admin_config_user"}
        _exempt_dotenv = {"chat_lock_no_admin_config_user"}
        for key, value in LOCK_UI_TEXT.items():
            if key not in _exempt_admin:
                assert "ADMIN_USER_IDS" not in value, (
                    f"UI text key '{key}' contains 'ADMIN_USER_IDS'"
                )
            if key not in _exempt_dotenv:
                assert ".env" not in value, (
                    f"UI text key '{key}' contains '.env'"
                )


class TestRepoLockCardConceptNote:
    """AC-15: Concept note is now included in conflict card to help users distinguish repo lock from chat lock."""

    def test_concept_note_in_card_markdown(self):
        import time

        from src.card.styles_lock import LOCK_UI_TEXT

        markdown, _ = build_repo_lock_card("/home/user/my-repo", time.monotonic() - 10)
        note = LOCK_UI_TEXT["repo_lock_concept_note"]
        assert note in markdown


# ---------------------------------------------------------------------------
# AC-16: /status no-lock educational explanation
# ---------------------------------------------------------------------------


class TestStatusNoLockExplain:
    """AC-16: _build_lock_status_lines shows educational text when no locks active."""

    def test_no_lock_shows_explain(self):
        from unittest.mock import MagicMock

        from src.card.styles_lock import LOCK_UI_TEXT

        handler = MagicMock()
        ctx = MagicMock()
        ctx.chat_lock_manager.get_lock_info.return_value = None
        ctx.repo_lock_manager = MagicMock()
        handler.ctx = ctx

        from src.feishu.handlers.diagnostics import DiagnosticsHandler
        result = DiagnosticsHandler._build_lock_status_lines(handler, "chat_001", project=None, is_admin=False)

        assert LOCK_UI_TEXT["lock_status_no_lock_explain"] in result
        assert LOCK_UI_TEXT["lock_status_no_active_lock"] in result


# ---------------------------------------------------------------------------
# AC-21: no-admin config user has .env guidance
# ---------------------------------------------------------------------------


class TestNoAdminConfigUserEnvGuidance:
    """AC-21: chat_lock_no_admin_config_user directs users to contact Bot deployer."""

    def test_has_deployer_guidance(self):
        from src.card.styles_lock import LOCK_UI_TEXT
        text = LOCK_UI_TEXT["chat_lock_no_admin_config_user"]
        assert "部署者" in text or "Bot" in text


# ---------------------------------------------------------------------------
# Task 40-41: build_chat_lock_card includes p2p guide and concept note
# ---------------------------------------------------------------------------


class TestChatLockCardContentBlocks:
    """Chat lock card includes the p2p guide text and concept note."""

    def test_p2p_guide_and_concept_note_present(self):
        from src.card.styles_lock import LOCK_UI_TEXT
        markdown, _ = build_chat_lock_card()
        assert LOCK_UI_TEXT["chat_lock_p2p_guide"] in markdown
        assert LOCK_UI_TEXT["chat_lock_concept_note"] in markdown


# ---------------------------------------------------------------------------
# Task 43: styles_lock contains only lock-related keys
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# chat_hint dead field removal: static text + parameter cleanup
# ---------------------------------------------------------------------------


class TestChatHintFieldRemoved:
    """Verify chat_hint parameter is removed and static text is used."""

    def test_same_sender_shows_static_hint(self):
        """is_same_sender=True produces card with static '另一个群' text, no placeholder."""
        import time
        markdown, _ = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10,
            is_same_sender=True,
        )
        assert "另一个群" in markdown
        assert "{chat_hint}" not in markdown

    def test_chat_hint_kwarg_raises_type_error(self):
        """Passing chat_hint= to build_repo_lock_card must raise TypeError."""
        import time
        with pytest.raises(TypeError):
            build_repo_lock_card(
                "/tmp/repo", time.monotonic() - 10,
                chat_hint="某群",
            )

    def test_same_sender_false_no_hint(self):
        """is_same_sender=False produces card without the same-sender hint line."""
        import time
        markdown, _ = build_repo_lock_card(
            "/tmp/repo", time.monotonic() - 10,
            is_same_sender=False,
        )
        assert "另一个群中持有该仓库锁" not in markdown


# ---------------------------------------------------------------------------
# format_undo_window
# ---------------------------------------------------------------------------


class TestFormatUndoWindow:
    """Test format_undo_window utility."""

    @pytest.mark.parametrize("secs,expected", [
        (0, ""),
        (60, "1 分钟"),
        (120, "2 分钟"),
        (300, "5 分钟"),
        (600, "10 分钟"),
    ])
    def test_format_undo_window(self, secs, expected):
        """format_undo_window produces friendly duration fragment (e.g. 'N 分钟')."""
        assert format_undo_window(secs) == expected

    def test_format_undo_window_non_multiple_of_60_rounds(self):
        """format_undo_window rounds non-60-multiple values (no crash)."""
        assert format_undo_window(90) == "2 分钟"
        assert format_undo_window(45) == "1 分钟"


# ---------------------------------------------------------------------------
# build_lock_help_body undo window display test
# ---------------------------------------------------------------------------


class TestLockHelpBodyUndoWindow:
    """Verify build_lock_help_body shows correct undo window time."""

    def test_undo_window_300_shows_5_minutes(self):
        from src.card.builders.lock_chat import build_lock_help_body

        body = build_lock_help_body(is_admin=True, lock_undo_window_seconds=300)
        assert "5 分钟" in body

    def test_undo_window_zero_uses_no_undo_text(self):
        """When lock_undo_window_seconds=0, help body uses lock_help_admin_lock_cmd_no_undo key."""
        from src.card.builders.lock_chat import build_lock_help_body
        from src.card.styles_lock import LOCK_UI_TEXT

        body = build_lock_help_body(is_admin=True, lock_undo_window_seconds=0)
        expected_text = LOCK_UI_TEXT["lock_help_admin_lock_cmd_no_undo"]
        assert expected_text in body
        # Should NOT contain undo window display
        assert "分钟" not in body


# ---------------------------------------------------------------------------
# Undo button creation conditional on window > 0
# ---------------------------------------------------------------------------


class TestLockChatUndoWindowZero:
    """Verify undo buttons are not created when lock_undo_window_seconds=0."""

    def test_undo_window_zero_returns_empty_buttons(self):
        """When lock_undo_window_seconds=0, no undo button should be created."""
        result = build_lock_success_card(
            "lock", lock_undo_window_seconds=0
        )
        md, buttons = result
        assert buttons == []
        assert "撤销" not in md

    def test_undo_window_positive_returns_button(self):
        """When lock_undo_window_seconds=300, undo button should be present."""
        result = build_lock_success_card(
            "lock", lock_undo_window_seconds=300
        )
        md, buttons = result
        assert len(buttons) == 1
        assert "撤销" in md

    def test_undo_window_zero_fallback_contains_unlock(self):
        """AC-9: When lock_undo_window=0, fallback text includes /unlock guidance."""
        result = build_lock_success_card(
            "lock", lock_undo_window_seconds=0, locker_name="张三"
        )
        md, buttons = result
        assert "/unlock" in md
        assert "{locker_name}" not in md  # No raw placeholder

    def test_undo_window_zero_locker_name_formatted(self):
        """AC-8: locker_name is correctly substituted in no-undo fallback."""
        result = build_lock_success_card(
            "lock", lock_undo_window_seconds=0, locker_name="Alice"
        )
        md, _ = result
        assert "Alice" in md
        # Empty locker_name should not produce malformed text
        result2 = build_lock_success_card(
            "lock", lock_undo_window_seconds=0, locker_name=""
        )
        md2, _ = result2
        assert "由 锁定" not in md2
        assert "由  锁定" not in md2
        assert "🔒 已锁定" in md2
