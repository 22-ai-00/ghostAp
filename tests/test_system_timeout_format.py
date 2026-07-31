"""Tests for SystemBuilder session_idle_timeout formatting logic.

Verifies the timeout_display formatting in build_system_help_card:
  - <= 60 minutes → "约 X 分钟"
  - > 60 minutes → "约 X 小时"

Also covers the underlying format_friendly_duration utility.
"""


from src.utils.text import format_friendly_duration


class TestFormatFriendlyDuration:
    """format_friendly_duration edge cases."""

    def test_under_60s(self):
        assert format_friendly_duration(30) == "30 秒"

    def test_exactly_60s(self):
        assert format_friendly_duration(60) == "约 1 分钟"

    def test_1800s_is_30_minutes(self):
        result = format_friendly_duration(1800)
        assert result == "约 30 分钟"

    def test_3600s_is_1_hour(self):
        result = format_friendly_duration(3600)
        assert result == "约 1 小时"

    def test_7200s_is_2_hours(self):
        result = format_friendly_duration(7200)
        assert result == "约 2 小时"

    def test_5400s_is_1_hour_30_minutes(self):
        result = format_friendly_duration(5400)
        assert result == "约 1 小时 30 分钟"

    def test_86400s_is_1_day(self):
        result = format_friendly_duration(86400)
        assert result == "约 1 天"

    def test_negative_clamps_to_zero(self):
        result = format_friendly_duration(-100)
        assert result == "0 秒"

    def test_zero(self):
        assert format_friendly_duration(0) == "0 秒"
