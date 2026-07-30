"""Tests for slock_default_roles configuration and logging."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError


class TestSlockDefaultRolesDefaultValue:
    """Test slock_default_roles default value."""

    def test_slock_default_roles_default_is_empty_string(self, monkeypatch):
        """Settings().slock_default_roles default value is empty string."""
        # Ensure no env var override
        monkeypatch.delenv("SLOCK_DEFAULT_ROLES", raising=False)

        from src.config.settings import Settings

        s = Settings(_env_file=None)
        assert s.slock_default_roles == ""


class TestSlockDefaultRolesLogLevel:
    """Test that the supported empty configuration is not treated as an error."""

    def test_empty_slock_default_roles_is_informational(self, caplog, monkeypatch):
        """Disabling automatic role provisioning should emit INFO, not WARNING."""
        monkeypatch.delenv("SLOCK_DEFAULT_ROLES", raising=False)

        from src.config.settings import Settings

        with caplog.at_level(logging.INFO, logger="src.config.settings"):
            Settings(_env_file=None)

        matching_records = [
            r for r in caplog.records
            if "slock_default_roles is empty" in r.message
        ]
        assert len(matching_records) == 1, (
            f"Expected 1 INFO log for empty slock_default_roles. "
            f"Records: {[r.message for r in caplog.records]}"
        )
        assert matching_records[0].levelno == logging.INFO

    def test_slock_default_roles_set_via_env_emits_no_empty_notice(self, caplog, monkeypatch):
        """A configured role set must not emit the empty-setting notice."""
        monkeypatch.setenv(
            "SLOCK_DEFAULT_ROLES",
            "planner:claude,coder:codex,reviewer:claude,tester:codex",
        )

        from src.config.settings import Settings

        with caplog.at_level(logging.INFO, logger="src.config.settings"):
            Settings(_env_file=None)

        matching_records = [
            r for r in caplog.records
            if "slock_default_roles is empty" in r.message
        ]
        assert matching_records == [], (
            f"Expected no empty-setting notice when SLOCK_DEFAULT_ROLES is set. "
            f"Records: {[r.message for r in caplog.records]}"
        )


class TestSlockWakePolicyConfig:
    def test_default_wake_policy_accepts_and_normalizes_alias(self, monkeypatch):
        monkeypatch.delenv("SLOCK_DEFAULT_ROLES", raising=False)

        from src.config.settings import Settings

        s = Settings(slock_default_wake_policy=" ON-MENTION ", _env_file=None)
        assert s.slock_default_wake_policy == "on_mention"

    def test_default_wake_policy_rejects_unknown_value(self, monkeypatch):
        monkeypatch.delenv("SLOCK_DEFAULT_ROLES", raising=False)

        from src.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(slock_default_wake_policy="mention-only", _env_file=None)
