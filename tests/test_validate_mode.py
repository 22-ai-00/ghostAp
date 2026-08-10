"""--validate / --check-config pre-check mode tests.

Covers:
- AC-R12: --validate exits 0 when config is valid
- AC-R12: --validate exits 1 with error message when config is invalid
- AC-R12: --check-config is an alias for --validate
- main() accepts argv parameter for testability
- AC-R28: --validate calls validate_feishu_config() and exits 1 when it returns False
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config import ConfigurationError


class TestValidateModeSuccess:
    """--validate exits 0 when configuration is valid."""

    def _make_valid_mock(self):
        mock_settings = MagicMock()
        mock_settings.validate_feishu_config.return_value = True
        mock_settings.card.session_idle_timeout = 1800
        mock_settings.card.session_idle_warn_at_remaining = 300
        mock_settings.lock_undo_window_seconds = 300
        mock_settings.card.delivery_pool_max_workers = 4
        mock_settings.card.delivery_api_timeout = 35
        mock_settings.card.max_chars = 28000
        return mock_settings

    def test_validate_exits_zero_on_valid_config(self, capsys):
        from src.main import main

        with patch("src.main.get_settings", return_value=self._make_valid_mock()):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "配置校验通过" in captured.out

    def test_check_config_alias_exits_zero(self, capsys):
        from src.main import main

        with patch("src.main.get_settings", return_value=self._make_valid_mock()):
            with pytest.raises(SystemExit) as exc_info:
                main(["--check-config"])

        assert exc_info.value.code == 0


class TestValidateModeFailure:
    """--validate exits 1 with error details when configuration is invalid."""

    def test_validate_exits_one_on_invalid_config(self, capsys):
        from src.main import main

        with patch("src.main.get_settings", side_effect=ConfigurationError("test validation error")):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "配置校验失败" in captured.err

    def test_validate_error_message_in_stderr(self, capsys):
        from src.main import main

        with patch("src.main.get_settings", side_effect=ConfigurationError("missing API key")):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "missing API key" in captured.err


class TestMainAcceptsArgv:
    """main() properly accepts argv parameter for programmatic invocation."""

    def test_no_validate_flag_proceeds_to_application(self):
        from src.main import main

        with patch("src.main.Application") as mock_app:
            mock_instance = mock_app.return_value
            mock_instance.run = lambda: None
            main([])  # no --validate flag

        mock_app.assert_called_once()

    def test_parse_known_args_ignores_unknown_flags(self, capsys):
        """Unknown flags like pytest's -x -q should not cause errors."""
        from src.main import main

        mock_settings = MagicMock()
        mock_settings.validate_feishu_config.return_value = True
        mock_settings.card.session_idle_timeout = 1800
        mock_settings.card.session_idle_warn_at_remaining = 300
        mock_settings.lock_undo_window_seconds = 300
        mock_settings.card.delivery_pool_max_workers = 4
        mock_settings.card.delivery_api_timeout = 35
        mock_settings.card.max_chars = 28000

        with patch("src.main.get_settings", return_value=mock_settings):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate", "-x", "-q"])

        assert exc_info.value.code == 0


class TestValidateFeishuCheck:
    """AC-R28: --validate calls validate_feishu_config() and exits 1 on False."""

    def test_validate_exits_one_when_feishu_config_invalid(self, capsys):
        from src.main import main

        mock_settings = MagicMock()
        mock_settings.validate_feishu_config.return_value = False

        with patch("src.main.get_settings", return_value=mock_settings):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "配置校验失败" in captured.err
        assert "APP_ID" in captured.err or "APP_SECRET" in captured.err

    def test_validate_calls_validate_feishu_config(self):
        from src.main import main

        mock_settings = MagicMock()
        mock_settings.validate_feishu_config.return_value = True
        mock_settings.card.session_idle_timeout = 1800
        mock_settings.card.session_idle_warn_at_remaining = 300
        mock_settings.lock_undo_window_seconds = 300
        mock_settings.card.delivery_pool_max_workers = 4
        mock_settings.card.delivery_api_timeout = 35
        mock_settings.card.max_chars = 28000

        with patch("src.main.get_settings", return_value=mock_settings):
            with pytest.raises(SystemExit):
                main(["--validate"])

        mock_settings.validate_feishu_config.assert_called_once()


class TestValidateParameterSummary:
    """--validate success prints structured parameter summary with groups and units."""

    def _make_mock_settings(self, **overrides):
        mock_settings = MagicMock()
        mock_settings.validate_feishu_config.return_value = True
        mock_settings.card.session_idle_timeout = overrides.get("idle_timeout", 1800)
        mock_settings.card.session_idle_warn_at_remaining = overrides.get("warn_at", 300)
        mock_settings.lock_undo_window_seconds = overrides.get("undo_window", 300)
        mock_settings.card.delivery_pool_max_workers = overrides.get("pool_workers", 4)
        mock_settings.card.delivery_api_timeout = overrides.get("api_timeout", 35)
        mock_settings.card.max_chars = overrides.get("max_chars", 28000)
        mock_settings.card.session_lock_ttl = overrides.get("lock_ttl", 600)
        mock_settings.card.session_lock_max = overrides.get("lock_max", 10000)
        return mock_settings

    def test_validate_prints_session_idle_timeout(self, capsys):
        from src.main import main

        mock_settings = self._make_mock_settings(idle_timeout=1800)

        with patch("src.main.get_settings", return_value=mock_settings):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "CARD_SESSION_IDLE_TIMEOUT" in captured.out
        assert "1800 秒" in captured.out
        assert "30 分钟" in captured.out

    def test_validate_prints_delivery_pool_max_workers(self, capsys):
        from src.main import main

        mock_settings = self._make_mock_settings(pool_workers=8)

        with patch("src.main.get_settings", return_value=mock_settings):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "CARD_DELIVERY_POOL_MAX_WORKERS" in captured.out
        assert "8" in captured.out

    def test_validate_prints_delivery_api_timeout(self, capsys):
        from src.main import main

        mock_settings = self._make_mock_settings(api_timeout=35)

        with patch("src.main.get_settings", return_value=mock_settings):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "CARD_DELIVERY_API_TIMEOUT" in captured.out
        assert "35 秒" in captured.out

    def test_validate_prints_version_header(self, capsys):
        from src.main import main

        mock_settings = self._make_mock_settings()

        with patch("src.main.get_settings", return_value=mock_settings):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "GhostAP v" in captured.out
        assert "配置校验通过" in captured.out

    def test_validate_prints_group_headers(self, capsys):
        from src.main import main

        mock_settings = self._make_mock_settings()

        with patch("src.main.get_settings", return_value=mock_settings):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "[会话超时]" in captured.out
        assert "[锁定参数]" in captured.out
        assert "[高级参数]" in captured.out

    def test_validate_prints_env_var_names(self, capsys):
        from src.main import main

        mock_settings = self._make_mock_settings()

        with patch("src.main.get_settings", return_value=mock_settings):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "CARD_SESSION_IDLE_WARN_AT_REMAINING" in captured.out
        assert "LOCK_UNDO_WINDOW_SECONDS" in captured.out
        assert "CARD_DELIVERY_API_TIMEOUT" in captured.out
        assert "CARD_MAX_CHARS" in captured.out

    def test_validate_prints_lock_pool_params(self, capsys):
        """Validate output includes CARD_SESSION_LOCK_TTL and CARD_SESSION_LOCK_MAX."""
        from src.main import main

        mock_settings = self._make_mock_settings(lock_ttl=600, lock_max=10000)

        with patch("src.main.get_settings", return_value=mock_settings):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "CARD_SESSION_LOCK_TTL" in captured.out
        assert "CARD_SESSION_LOCK_MAX" in captured.out
        assert "10000" in captured.out

    def test_validate_prints_unit_suffixes(self, capsys):
        """Validate output includes unit suffixes for params."""
        from src.main import main

        mock_settings = self._make_mock_settings(pool_workers=4, max_chars=28000)

        with patch("src.main.get_settings", return_value=mock_settings):
            with pytest.raises(SystemExit) as exc_info:
                main(["--validate"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "(threads)" in captured.out
        assert "chars" in captured.out


class TestFormatDuration:
    """Unit tests for _format_duration helper."""

    def test_exact_minutes(self):
        from src.main import _format_duration
        assert _format_duration(1800) == "1800 秒（30 分钟）"
        assert _format_duration(300) == "300 秒（5 分钟）"
        assert _format_duration(7200) == "7200 秒（120 分钟）"
        assert _format_duration(60) == "60 秒（1 分钟）"

    def test_non_exact_minutes(self):
        from src.main import _format_duration
        assert _format_duration(90) == "90 秒（~2 分钟）"
        assert _format_duration(125) == "125 秒（~3 分钟）"

    def test_zero(self):
        from src.main import _format_duration
        assert _format_duration(0) == "0 秒"




# ---------------------------------------------------------------------------
# Config cross-field interaction tests (FS-14)
# ---------------------------------------------------------------------------


class TestConfigCrossFieldValidation:
    """Cross-field and boundary validation for CardSessionConfig and lock params."""

    def test_warn_at_remaining_equal_to_idle_timeout_raises(self):
        """warn_at_remaining >= idle_timeout is invalid."""
        from pydantic import ValidationError

        from src.config import CardSessionConfig

        with pytest.raises(ValidationError):
            CardSessionConfig(
                session_idle_timeout=300,
                session_idle_warn_at_remaining=300,
            )

    def test_warn_at_remaining_exceeds_idle_timeout_raises(self):
        """warn_at_remaining > idle_timeout is invalid."""
        from pydantic import ValidationError

        from src.config import CardSessionConfig

        with pytest.raises(ValidationError):
            CardSessionConfig(
                session_idle_timeout=300,
                session_idle_warn_at_remaining=600,
            )

    def test_warn_at_remaining_less_than_idle_timeout_passes(self):
        """warn_at_remaining < idle_timeout is valid."""
        from src.config import CardSessionConfig

        cfg = CardSessionConfig(
            session_idle_timeout=1800,
            session_idle_warn_at_remaining=300,
        )
        assert cfg.session_idle_warn_at_remaining == 300

    @pytest.mark.parametrize(
        "value,should_pass",
        [
            (59, False),   # below minimum 60
            (60, True),    # exact minimum, multiple of 60
            (61, False),   # not multiple of 60
            (120, True),   # valid
            (300, True),   # default
            (600, True),   # exact maximum
            (601, False),  # above maximum
        ],
    )
    def test_lock_undo_window_seconds_boundaries(self, value, should_pass, monkeypatch):
        """Boundary values for lock_undo_window_seconds."""

        from src.config import Settings

        # We test the field_validator directly via CardSessionConfig doesn't own it;
        # it's on Settings. Use Settings validator approach.
        # The validator is on Settings class, so we test through the validator function.

        # lock_undo_window_seconds is on Settings, not CardSessionConfig.
        # Test the validator directly.

        # Simpler approach: just call the validator classmethod directly
        cls = Settings
        validator = cls._lock_undo_window_seconds_in_range

        if should_pass:
            result = validator(value, None)
            assert result == value
        else:
            with pytest.raises(ValueError):
                validator(value, None)
