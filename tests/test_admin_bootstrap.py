from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.access_control import (
    IngressAccessPolicy,
    IngressAccessPolicyProvider,
    IngressAccessRequest,
)
from src.admin_bootstrap import AdminBootstrapService
from src.config import IngressAccessMode, SecuritySeverity
from src.config.env_file_store import EnvPreReplaceError
from src.feishu.handlers.system import SystemHandler
from src.feishu.slash_command_parser import SlashCommandParser
from src.thread import (
    set_current_is_p2p,
    set_current_sender_id,
)


@pytest.fixture(autouse=True)
def _clear_rate_limit_state():
    """Clear class-level rate limit state between tests."""
    AdminBootstrapService._last_attempt.clear()
    yield
    AdminBootstrapService._last_attempt.clear()


def _settings(
    *,
    admins: frozenset[str] = frozenset(),
    users: frozenset[str] = frozenset(),
    chats: frozenset[str] = frozenset(),
    bootstrap_scope: str = "p2p_only",
):
    return SimpleNamespace(
        admin_user_ids=admins,
        allowed_user_ids=users,
        allowed_chat_ids=chats,
        ingress_access_mode="enforced",
        admin_bootstrap_scope=bootstrap_scope,
    )


def _provider(settings) -> IngressAccessPolicyProvider:
    return IngressAccessPolicyProvider(
        IngressAccessPolicy(
            admin_ids=settings.admin_user_ids,
            allowed_user_ids=settings.allowed_user_ids,
            allowed_chat_ids=settings.allowed_chat_ids,
            mode=IngressAccessMode.ENFORCED,
            admin_bootstrap_scope=settings.admin_bootstrap_scope,
        )
    )


def test_setadmin_bootstraps_sender_as_only_admin(tmp_path):
    settings = _settings()
    env_path = tmp_path / ".env"
    provider = _provider(settings)

    result = AdminBootstrapService(
        env_path=env_path,
        settings_getter=lambda: settings,
        policy_provider=provider,
    ).set_admin(
        "ou_first",
        "ou_other",
        chat_type="p2p",
        chat_id="oc_dm",
        message_id="om_bootstrap",
    )

    assert result.success is True
    assert result.code == "bootstrap"
    assert result.target_id == "ou_first"
    assert env_path.read_text(encoding="utf-8") == (
        "ADMIN_USER_IDS=ou_first\n"
        "ALLOWED_USER_IDS=ou_first\n"
        "ALLOWED_CHAT_IDS=oc_dm\n"
    )
    assert settings.admin_user_ids == frozenset({"ou_first"})
    assert settings.allowed_user_ids == frozenset({"ou_first"})
    assert settings.allowed_chat_ids == frozenset({"oc_dm"})
    assert provider.current.decide(
        IngressAccessRequest(
            message_id="om_after_bootstrap",
            sender_id="ou_first",
            chat_id="oc_dm",
            chat_type="p2p",
            command_match=None,
        )
    ).allowed is True


def test_access_allow_chat_persists_and_immediately_updates_live_policy(tmp_path):
    settings = _settings(
        admins=frozenset({"ou_admin"}),
        users=frozenset({"ou_admin"}),
        chats=frozenset({"oc_dm"}),
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ADMIN_USER_IDS=ou_admin\n"
        "ALLOWED_USER_IDS=ou_admin\n"
        "ALLOWED_CHAT_IDS=oc_dm\n",
        encoding="utf-8",
    )
    provider = _provider(settings)

    result = AdminBootstrapService(
        env_path=env_path,
        settings_getter=lambda: settings,
        policy_provider=provider,
    ).allow_current_chat(
        "ou_admin",
        "oc_group",
        chat_type="group",
        message_id="om_allow_group",
    )

    assert result.success is True
    assert result.code == "chat_enrolled"
    assert settings.allowed_chat_ids == frozenset({"oc_dm", "oc_group"})
    assert "ALLOWED_CHAT_IDS=oc_dm,oc_group\n" in env_path.read_text(encoding="utf-8")
    assert provider.current.decide(
        IngressAccessRequest(
            message_id="om_group_message",
            sender_id="ou_admin",
            chat_id="oc_group",
            chat_type="group",
            command_match=None,
        )
    ).allowed is True


def test_any_chat_break_glass_bootstrap_enrols_current_group_atomically(
    tmp_path,
):
    settings = _settings(bootstrap_scope="any_chat")
    env_path = tmp_path / ".env"
    provider = _provider(settings)

    result = AdminBootstrapService(
        env_path=env_path,
        settings_getter=lambda: settings,
        policy_provider=provider,
    ).set_admin(
        "ou_first",
        chat_type="group",
        chat_id="oc_group",
        message_id="om_group_bootstrap",
    )

    assert result.success is True
    assert env_path.read_text(encoding="utf-8") == (
        "ADMIN_USER_IDS=ou_first\n"
        "ALLOWED_USER_IDS=ou_first\n"
        "ALLOWED_CHAT_IDS=oc_group\n"
    )
    assert provider.current.decide(
        IngressAccessRequest(
            message_id="om_after_group_bootstrap",
            sender_id="ou_first",
            chat_id="oc_group",
            chat_type="group",
            command_match=None,
        )
    ).allowed is True


def test_persistence_failure_keeps_settings_and_live_policy_unchanged(tmp_path):
    settings = _settings()
    provider = _provider(settings)
    env_store = MagicMock()
    env_store.update_with.side_effect = EnvPreReplaceError(
        "read-only filesystem"
    )

    result = AdminBootstrapService(
        env_path=tmp_path / ".env",
        settings_getter=lambda: settings,
        policy_provider=provider,
        env_store=env_store,
    ).set_admin(
        "ou_first",
        chat_type="p2p",
        chat_id="oc_dm",
        message_id="om_persist_failure",
    )

    assert result.success is False
    assert result.code == "persistence_failed"
    assert settings.admin_user_ids == frozenset()
    assert settings.allowed_user_ids == frozenset()
    assert settings.allowed_chat_ids == frozenset()
    assert provider.current.admin_ids == frozenset()
    assert provider.current.allowed_user_ids == frozenset()
    assert provider.current.allowed_chat_ids == frozenset()


def test_policy_swap_failure_keeps_old_live_snapshot_after_disk_commit(
    tmp_path,
    caplog,
):
    settings = _settings()
    provider = _provider(settings)
    original = provider.current
    provider.swap = MagicMock(side_effect=RuntimeError("swap failed"))
    env_path = tmp_path / ".env"

    result = AdminBootstrapService(
        env_path=env_path,
        settings_getter=lambda: settings,
        policy_provider=provider,
    ).set_admin(
        "ou_first",
        chat_type="p2p",
        chat_id="oc_dm",
        message_id="om_swap_failure",
    )

    assert result.success is False
    assert result.code == "policy_refresh_failed"
    assert env_path.read_text(encoding="utf-8") == (
        "ADMIN_USER_IDS=ou_first\n"
        "ALLOWED_USER_IDS=ou_first\n"
        "ALLOWED_CHAT_IDS=oc_dm\n"
    )
    assert provider.current is original
    assert settings.admin_user_ids == frozenset()
    assert settings.allowed_user_ids == frozenset()
    assert settings.allowed_chat_ids == frozenset()
    assert "ACCESS_POLICY_REFRESH_BLOCKED" in caplog.text
    assert len(provider.blocking_findings) == 1
    finding = provider.blocking_findings[0]
    assert finding.code == "ingress_policy_refresh_failed"
    assert finding.severity is SecuritySeverity.BLOCKING


def test_setadmin_non_admin_cannot_replace_existing_admin(tmp_path):
    settings = SimpleNamespace(admin_user_ids=frozenset({"ou_admin"}))
    env_path = tmp_path / ".env"
    env_path.write_text("ADMIN_USER_IDS=ou_admin\n", encoding="utf-8")

    result = AdminBootstrapService(
        env_path=env_path,
        settings_getter=lambda: settings,
    ).set_admin(
        "ou_other",
        "ou_other",
        chat_type="group",
        chat_id="oc_group",
        message_id="om_not_admin",
    )

    assert result.success is False
    assert result.code == "not_admin"
    assert env_path.read_text(encoding="utf-8") == "ADMIN_USER_IDS=ou_admin\n"
    assert settings.admin_user_ids == frozenset({"ou_admin"})


def test_setadmin_existing_admin_can_replace_single_admin(tmp_path):
    settings = SimpleNamespace(admin_user_ids=frozenset({"ou_admin"}))
    env_path = tmp_path / ".env"
    env_path.write_text("APP_ID=app\nADMIN_USER_IDS=ou_admin\n", encoding="utf-8")

    result = AdminBootstrapService(
        env_path=env_path,
        settings_getter=lambda: settings,
    ).set_admin(
        "ou_admin",
        "ou_next",
        chat_type="group",
        chat_id="oc_group",
        message_id="om_replace_admin",
    )

    assert result.success is True
    assert result.code == "updated"
    assert result.target_id == "ou_next"
    assert env_path.read_text(encoding="utf-8") == "APP_ID=app\nADMIN_USER_IDS=ou_next\n"
    assert settings.admin_user_ids == frozenset({"ou_next"})


def test_setadmin_replaces_export_style_env_line(tmp_path):
    settings = SimpleNamespace(admin_user_ids=frozenset({"ou_admin"}))
    env_path = tmp_path / ".env"
    env_path.write_text("APP_ID=app\nexport ADMIN_USER_IDS = ou_admin\n", encoding="utf-8")

    result = AdminBootstrapService(
        env_path=env_path,
        settings_getter=lambda: settings,
    ).set_admin(
        "ou_admin",
        "ou_next",
        chat_type="p2p",
        chat_id="oc_dm",
        message_id="om_export_replace",
    )

    assert result.success is True
    assert env_path.read_text(encoding="utf-8") == "APP_ID=app\nADMIN_USER_IDS=ou_next\n"


def test_setadmin_first_sender_blocks_other_service_instances(tmp_path):
    settings = SimpleNamespace(admin_user_ids=frozenset())
    env_path = tmp_path / ".env"

    first = AdminBootstrapService(env_path=env_path, settings_getter=lambda: settings)
    second = AdminBootstrapService(env_path=env_path, settings_getter=lambda: settings)

    assert first.set_admin(
        "ou_first",
        chat_type="p2p",
        chat_id="oc_dm",
        message_id="om_first",
    ).success is True
    result = second.set_admin(
        "ou_other",
        chat_type="p2p",
        chat_id="oc_other_dm",
        message_id="om_second",
    )

    assert result.success is False
    assert result.code == "not_admin"
    assert env_path.read_text(encoding="utf-8") == (
        "ADMIN_USER_IDS=ou_first\n"
        "ALLOWED_USER_IDS=ou_first\n"
        "ALLOWED_CHAT_IDS=oc_dm\n"
    )


def test_setadmin_accepts_legacy_comma_string_admins(tmp_path):
    settings = SimpleNamespace(admin_user_ids="ou_admin,ou_backup")
    env_path = tmp_path / ".env"

    result = AdminBootstrapService(
        env_path=env_path,
        settings_getter=lambda: settings,
    ).set_admin(
        "ou_backup",
        "ou_next",
        chat_type="group",
        chat_id="oc_group",
        message_id="om_legacy_admin",
    )

    assert result.success is True
    assert result.target_id == "ou_next"
    assert settings.admin_user_ids == frozenset({"ou_next"})


def test_setadmin_rejects_invalid_target_after_bootstrap(tmp_path):
    settings = SimpleNamespace(admin_user_ids=frozenset({"ou_admin"}))
    env_path = tmp_path / ".env"
    env_path.write_text("ADMIN_USER_IDS=ou_admin\n", encoding="utf-8")

    result = AdminBootstrapService(
        env_path=env_path,
        settings_getter=lambda: settings,
    ).set_admin(
        "ou_admin",
        "bad,target",
        chat_type="p2p",
        chat_id="oc_dm",
        message_id="om_invalid_target",
    )

    assert result.success is False
    assert result.code == "invalid_target"
    assert env_path.read_text(encoding="utf-8") == "ADMIN_USER_IDS=ou_admin\n"


def test_system_handler_recognizes_setadmin_command():
    m = SlashCommandParser.parse("/setadmin")
    assert SystemHandler.is_interceptable_command_match(m) is True

    m = SlashCommandParser.parse("/setadmin ou_next")
    assert SystemHandler.is_interceptable_command_match(m) is True


def test_system_handler_recognizes_access_command():
    m = SlashCommandParser.parse("/access allow-chat")
    assert SystemHandler.is_interceptable_command_match(m) is True


def test_system_handler_routes_setadmin_with_sender():
    ctx = MagicMock()
    handler = SystemHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_error = MagicMock()

    service = MagicMock()
    service.set_admin.return_value = SimpleNamespace(
        success=True,
        code="bootstrap",
        target_id="ou_first",
    )

    set_current_sender_id("ou_first")
    set_current_is_p2p(False)
    try:
        with patch("src.admin_bootstrap.AdminBootstrapService", return_value=service):
            handler.handle_intercepted_command(
                "om_1",
                "oc_1",
                "/setadmin",
                command_match=SlashCommandParser.parse("/setadmin"),
            )
    finally:
        set_current_sender_id(None)
        set_current_is_p2p(False)

    service.set_admin.assert_called_once_with(
        "ou_first",
        "",
        chat_type="group",
        chat_id="oc_1",
        message_id="om_1",
    )
    handler.reply_text.assert_called_once()
    handler.reply_error.assert_not_called()


def test_system_handler_routes_access_for_current_group_only():
    ctx = MagicMock()
    handler = SystemHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_error = MagicMock()

    service = MagicMock()
    service.allow_current_chat.return_value = SimpleNamespace(
        success=True,
        code="chat_enrolled",
        target_id="oc_1",
    )

    set_current_sender_id("ou_admin")
    set_current_is_p2p(False)
    try:
        with patch("src.admin_bootstrap.AdminBootstrapService", return_value=service):
            handler.handle_intercepted_command(
                "om_1",
                "oc_1",
                "/access allow-chat",
                command_match=SlashCommandParser.parse("/access allow-chat"),
            )
    finally:
        set_current_sender_id(None)
        set_current_is_p2p(False)

    service.allow_current_chat.assert_called_once_with(
        "ou_admin",
        "oc_1",
        chat_type="group",
        message_id="om_1",
    )
    handler.reply_text.assert_called_once()
    handler.reply_error.assert_not_called()
