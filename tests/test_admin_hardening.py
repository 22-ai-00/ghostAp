"""Tests for setadmin hardening: p2p requirement, rate limiting, file permissions."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.admin_bootstrap import AdminBootstrapService


def _set_admin(
    service: AdminBootstrapService,
    sender_id: str,
    target_id: str = "",
    *,
    chat_type: str = "p2p",
    chat_id: str = "oc_dm",
    message_id: str = "om_request",
):
    return service.set_admin(
        sender_id,
        target_id,
        chat_type=chat_type,
        chat_id=chat_id,
        message_id=message_id,
    )


@pytest.fixture(autouse=True)
def _clear_rate_limit_state():
    """Clear class-level rate limit state between tests."""
    AdminBootstrapService._last_attempt.clear()
    yield
    AdminBootstrapService._last_attempt.clear()


class TestBootstrapRequiresP2P:
    """First-time bootstrap must be from a p2p (private) chat."""

    def test_group_chat_bootstrap_rejected(self, tmp_path):
        settings = SimpleNamespace(admin_user_ids=frozenset())
        env_path = tmp_path / ".env"

        service = AdminBootstrapService(
            env_path=env_path,
            settings_getter=lambda: settings,
        )
        result = _set_admin(service, "ou_sender", chat_type="group")

        assert result.success is False
        assert result.code == "bootstrap_requires_p2p"
        assert not env_path.exists()

    def test_p2p_chat_bootstrap_succeeds(self, tmp_path):
        settings = SimpleNamespace(admin_user_ids=frozenset())
        env_path = tmp_path / ".env"

        service = AdminBootstrapService(
            env_path=env_path,
            settings_getter=lambda: settings,
        )
        result = _set_admin(service, "ou_sender")

        assert result.success is True
        assert result.code == "bootstrap"
        assert result.target_id == "ou_sender"
        assert env_path.exists()

    def test_empty_chat_type_is_rejected_without_legacy_backdoor(self, tmp_path):
        settings = SimpleNamespace(admin_user_ids=frozenset())
        env_path = tmp_path / ".env"

        service = AdminBootstrapService(
            env_path=env_path,
            settings_getter=lambda: settings,
        )
        result = _set_admin(service, "ou_sender", chat_type="")

        assert result.success is False
        assert result.code == "invalid_request_context"
        assert not env_path.exists()

    def test_existing_admin_update_from_group_allowed(self, tmp_path):
        """Once admin exists, chat_type restriction does not apply to updates."""
        settings = SimpleNamespace(admin_user_ids=frozenset({"ou_admin"}))
        env_path = tmp_path / ".env"
        env_path.write_text("ADMIN_USER_IDS=ou_admin\n", encoding="utf-8")

        service = AdminBootstrapService(
            env_path=env_path,
            settings_getter=lambda: settings,
        )
        result = _set_admin(
            service,
            "ou_admin",
            "ou_new",
            chat_type="group",
            chat_id="oc_group",
        )

        assert result.success is True
        assert result.code == "updated"
        assert result.target_id == "ou_new"


class TestRateLimiting:
    """60-second cooldown per sender_id."""

    def test_rapid_calls_rate_limited(self, tmp_path):
        settings = SimpleNamespace(admin_user_ids=frozenset())
        env_path = tmp_path / ".env"

        svc = AdminBootstrapService(
            env_path=env_path,
            settings_getter=lambda: settings,
        )

        # First call succeeds
        result1 = _set_admin(svc, "ou_sender")
        assert result1.success is True

        # Second call within 60s is rate-limited
        result2 = _set_admin(svc, "ou_sender", message_id="om_request_2")
        assert result2.success is False
        assert result2.code == "rate_limited"

    def test_different_sender_not_rate_limited(self, tmp_path):
        settings = SimpleNamespace(admin_user_ids=frozenset())
        env_path = tmp_path / ".env"

        svc = AdminBootstrapService(
            env_path=env_path,
            settings_getter=lambda: settings,
        )

        result1 = _set_admin(svc, "ou_sender_a")
        assert result1.success is True

        # Different sender is not rate-limited
        result2 = _set_admin(
            svc,
            "ou_sender_b",
            chat_id="oc_dm_b",
            message_id="om_request_b",
        )
        # Will be "not_admin" because sender_a already became admin, but NOT rate_limited
        assert result2.code != "rate_limited"

    def test_cooldown_expires(self, tmp_path):
        settings = SimpleNamespace(admin_user_ids=frozenset({"ou_admin"}))
        env_path = tmp_path / ".env"
        env_path.write_text("ADMIN_USER_IDS=ou_admin\n", encoding="utf-8")

        svc = AdminBootstrapService(
            env_path=env_path,
            settings_getter=lambda: settings,
        )

        # First call succeeds
        result1 = _set_admin(svc, "ou_admin")
        assert result1.success is True

        # Simulate time passing beyond cooldown
        AdminBootstrapService._last_attempt["ou_admin"] = time.time() - 61

        result2 = _set_admin(svc, "ou_admin", message_id="om_request_2")
        assert result2.success is True
        assert result2.code == "updated"


class TestEnvFilePermissions:
    """The .env file should have 0o600 permissions after write."""

    def test_env_file_chmod_600(self, tmp_path):
        settings = SimpleNamespace(admin_user_ids=frozenset())
        env_path = tmp_path / ".env"

        service = AdminBootstrapService(
            env_path=env_path,
            settings_getter=lambda: settings,
        )
        _set_admin(service, "ou_sender")

        assert env_path.exists()
        mode = stat.S_IMODE(os.stat(env_path).st_mode)
        assert mode == 0o600

    def test_env_file_chmod_on_update(self, tmp_path):
        settings = SimpleNamespace(admin_user_ids=frozenset({"ou_admin"}))
        env_path = tmp_path / ".env"
        env_path.write_text("ADMIN_USER_IDS=ou_admin\n", encoding="utf-8")
        # Set permissive initial permissions
        os.chmod(env_path, 0o644)

        svc = AdminBootstrapService(
            env_path=env_path,
            settings_getter=lambda: settings,
        )
        _set_admin(svc, "ou_admin", "ou_new")

        mode = stat.S_IMODE(os.stat(env_path).st_mode)
        assert mode == 0o600


class TestAuditLogging:
    """Successful admin changes emit audit log."""

    def test_audit_log_emitted_on_success(self, tmp_path):
        settings = SimpleNamespace(admin_user_ids=frozenset())
        env_path = tmp_path / ".env"

        with patch("src.admin_bootstrap.audit_logger") as mock_audit:
            service = AdminBootstrapService(
                env_path=env_path,
                settings_getter=lambda: settings,
            )
            _set_admin(service, "ou_sender")

            mock_audit.info.assert_called_once()
            call_args = mock_audit.info.call_args
            assert "ADMIN_CHANGE" in call_args[0][0]
            rendered = " ".join(str(item) for item in call_args.args)
            assert "ou_sender" not in rendered
            expected_hash = hashlib.sha256(b"ou_sender").hexdigest()[:16]
            assert expected_hash in rendered

    def test_no_audit_log_on_failure(self, tmp_path):
        settings = SimpleNamespace(admin_user_ids=frozenset())
        env_path = tmp_path / ".env"

        with patch("src.admin_bootstrap.audit_logger") as mock_audit:
            service = AdminBootstrapService(
                env_path=env_path,
                settings_getter=lambda: settings,
            )
            _set_admin(service, "ou_sender", chat_type="group")

            mock_audit.info.assert_not_called()
