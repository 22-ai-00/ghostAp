"""Tests for src/card/nav_link.py — full branch coverage."""
from src.card.nav_link import format_navigation_link


class TestFormatNavigationLink:
    """format_navigation_link: session-level rotation deep-link."""

    def test_with_msg_id_returns_deep_link_and_no_fallback(self):
        nav, fallback = format_navigation_link(
            new_msg_id="om_abc123", rotation_seq=2
        )
        assert "lark://message/om_abc123" in nav
        assert fallback is None

    def test_with_msg_id_includes_seq_info(self):
        nav, _ = format_navigation_link(
            new_msg_id="om_test", rotation_seq=3
        )
        assert "3" in nav

    def test_without_msg_id_returns_fallback_notice(self):
        nav, fallback = format_navigation_link(
            new_msg_id=None, rotation_seq=1
        )
        assert "lark://message/" not in nav
        assert fallback is not None
        assert "下方" in fallback

    def test_empty_string_msg_id_treated_as_missing(self):
        nav, fallback = format_navigation_link(
            new_msg_id="", rotation_seq=1
        )
        assert "lark://message/" not in nav
        assert fallback is not None

    def test_nav_uses_pian_terminology(self):
        """Navigation text uses '第 N 篇' for continuation."""
        nav, _ = format_navigation_link(new_msg_id="om_x", rotation_seq=2)
        assert "第 2 篇" in nav

    def test_no_sid_short_in_output(self):
        """No technical session ID leaks into user-facing text."""
        nav, fallback = format_navigation_link(new_msg_id=None, rotation_seq=1)
        assert "会话" not in nav
        assert "会话" not in (fallback or "")
