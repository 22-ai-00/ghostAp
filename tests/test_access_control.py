from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.access_control import (
    AccessOperation,
    IngressAccessPolicy,
    IngressAccessPolicyProvider,
    IngressAccessRequest,
    build_ingress_access_policy,
)
from src.config import IngressAccessMode
from src.config.env_file_store import AtomicEnvFileStore
from src.feishu.slash_command_parser import SlashCommandParser


def _policy(
    *,
    admins: frozenset[str] = frozenset(),
    users: frozenset[str] = frozenset(),
    chats: frozenset[str] = frozenset(),
    mode: IngressAccessMode = IngressAccessMode.ENFORCED,
    bootstrap_scope: str = "p2p_only",
) -> IngressAccessPolicy:
    return IngressAccessPolicy(
        admin_ids=admins,
        allowed_user_ids=users,
        allowed_chat_ids=chats,
        mode=mode,
        admin_bootstrap_scope=bootstrap_scope,
    )


def _request(
    *,
    message: str = "om_test",
    sender: str = "ou_unknown",
    chat: str = "oc_unknown",
    chat_type: str = "group",
    text: str = "hello",
) -> IngressAccessRequest:
    return IngressAccessRequest(
        message_id=message,
        sender_id=sender,
        chat_id=chat,
        chat_type=chat_type,
        command_match=SlashCommandParser.parse(text),
    )


def test_empty_allowlists_reject_normal_messages() -> None:
    decision = _policy().decide(_request())

    assert decision.allowed is False
    assert decision.operation is AccessOperation.NORMAL_MESSAGE
    assert decision.reason_code == "access_not_enrolled"
    assert decision.prospective_allowed is False


def test_unconfigured_enforced_bot_allows_safe_private_bootstrap_help() -> None:
    decision = _policy().decide(_request(text="/help", chat_type="p2p"))

    assert decision.allowed is True
    assert decision.operation is AccessOperation.BOOTSTRAP_HELP
    assert decision.reason_code == "bootstrap_help"
    assert decision.prospective_allowed is False


@pytest.mark.parametrize(
    ("text", "chat_type"),
    [
        ("/help", "group"),
        ("/help extra", "p2p"),
        ("/codex", "p2p"),
    ],
)
def test_bootstrap_help_does_not_widen_unconfigured_access(
    text: str,
    chat_type: str,
) -> None:
    decision = _policy().decide(_request(text=text, chat_type=chat_type))

    assert decision.allowed is False
    assert decision.operation is AccessOperation.NORMAL_MESSAGE


def test_normal_message_requires_both_user_and_chat_dimensions() -> None:
    user_only = _policy(users=frozenset({"ou_user"})).decide(
        _request(sender="ou_user")
    )
    chat_only = _policy(chats=frozenset({"oc_group"})).decide(
        _request(chat="oc_group")
    )
    both = _policy(
        users=frozenset({"ou_user"}),
        chats=frozenset({"oc_group"}),
    ).decide(_request(sender="ou_user", chat="oc_group"))
    admin_and_chat = _policy(
        admins=frozenset({"ou_admin"}),
        chats=frozenset({"oc_group"}),
    ).decide(_request(sender="ou_admin", chat="oc_group"))

    assert user_only.allowed is False
    assert chat_only.allowed is False
    assert both.allowed is True
    assert admin_and_chat.allowed is True


@pytest.mark.parametrize(
    ("text", "chat_type", "allowed"),
    [
        ("/setadmin", "p2p", True),
        ("/SeTaDmIn ou_ignored", "p2p", True),
        ("  /setadmin\tou_ignored  ", "p2p", True),
        ("/setadmin", "group", False),
        ("/setadmin ou_ignored", "group", False),
        ("free text /setadmin", "p2p", False),
    ],
)
def test_first_setadmin_command_shape_is_p2p_only(
    text: str,
    chat_type: str,
    allowed: bool,
) -> None:
    decision = _policy().decide(_request(text=text, chat_type=chat_type))

    assert decision.allowed is allowed
    if allowed:
        assert decision.operation is AccessOperation.BOOTSTRAP_ADMIN
        assert decision.reason_code == "bootstrap_admin"


def test_any_chat_scope_is_an_explicit_group_bootstrap_break_glass() -> None:
    decision = _policy(bootstrap_scope="any_chat").decide(
        _request(text="/setadmin", chat_type="group")
    )

    assert decision.allowed is True
    assert decision.operation is AccessOperation.BOOTSTRAP_ADMIN


@pytest.mark.parametrize(
    ("text", "allowed"),
    [
        ("/access allow-chat", True),
        (" /AcCeSs\tALLOW-CHAT ", True),
        ("/access", False),
        ("/access allow-chat extra", False),
        ("/access allow-user", False),
        ("please /access allow-chat", False),
    ],
)
def test_access_allow_chat_requires_exact_normalized_command_shape(
    text: str,
    allowed: bool,
) -> None:
    decision = _policy(admins=frozenset({"ou_admin"})).decide(
        _request(sender="ou_admin", text=text)
    )

    assert decision.allowed is allowed
    if allowed:
        assert decision.operation is AccessOperation.ENROL_CURRENT_CHAT
        assert decision.reason_code == "admin_chat_enrolment"
    else:
        assert decision.operation is AccessOperation.NORMAL_MESSAGE
        assert decision.reason_code == "access_not_enrolled"


def test_non_admin_cannot_enrol_current_chat() -> None:
    decision = _policy(admins=frozenset({"ou_admin"})).decide(
        _request(sender="ou_other", text="/access allow-chat")
    )

    assert decision.allowed is False
    assert decision.operation is AccessOperation.NORMAL_MESSAGE


def test_access_allow_chat_is_group_only() -> None:
    decision = _policy(admins=frozenset({"ou_admin"})).decide(
        _request(
            sender="ou_admin",
            chat="oc_dm",
            chat_type="p2p",
            text="/access allow-chat",
        )
    )

    assert decision.allowed is False
    assert decision.reason_code == "access_not_enrolled"


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        (IngressAccessMode.SHADOW, "shadow_would_deny"),
        (IngressAccessMode.LEGACY_ALLOW_ALL, "legacy_allow_all"),
    ],
)
def test_break_glass_modes_allow_without_mutating_enrolment(
    mode: IngressAccessMode,
    reason: str,
) -> None:
    policy = _policy(mode=mode)

    decision = policy.decide(_request())

    assert decision.allowed is True
    assert decision.prospective_allowed is False
    assert decision.reason_code == reason
    assert policy.allowed_user_ids == frozenset()
    assert policy.allowed_chat_ids == frozenset()


def test_policy_provider_swaps_complete_immutable_snapshot() -> None:
    original = _policy()
    replacement = _policy(
        admins=frozenset({"ou_admin"}),
        users=frozenset({"ou_admin"}),
        chats=frozenset({"oc_dm"}),
    )
    provider = IngressAccessPolicyProvider(original)

    provider.swap(replacement)

    assert provider.current is replacement
    assert original.admin_ids == frozenset()
    assert original.allowed_user_ids == frozenset()
    assert original.allowed_chat_ids == frozenset()


def test_policy_builder_defaults_partial_non_string_settings_securely() -> None:
    settings = SimpleNamespace(
        ingress_access_mode=MagicMock(),
        admin_bootstrap_scope=MagicMock(),
        admin_user_ids=MagicMock(),
        allowed_user_ids=MagicMock(),
        allowed_chat_ids=MagicMock(),
    )

    policy = build_ingress_access_policy(settings)

    assert policy.mode is IngressAccessMode.ENFORCED
    assert policy.admin_bootstrap_scope == "p2p_only"
    assert policy.decide(_request()).allowed is False


def test_atomic_env_store_preserves_unrelated_and_commented_lines(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "APP_ID=app\n"
        "# ADMIN_USER_IDS=commented\n"
        "export ADMIN_USER_IDS = ou_old\n"
        "TAIL_WITHOUT_NEWLINE=yes",
        encoding="utf-8",
    )

    AtomicEnvFileStore(env_path).update_many(
        {
            "ADMIN_USER_IDS": "ou_new",
            "ALLOWED_USER_IDS": "ou_new",
        }
    )

    assert env_path.read_text(encoding="utf-8") == (
        "APP_ID=app\n"
        "# ADMIN_USER_IDS=commented\n"
        "ADMIN_USER_IDS=ou_new\n"
        "TAIL_WITHOUT_NEWLINE=yes\n"
        "ALLOWED_USER_IDS=ou_new\n"
    )
    assert os.stat(env_path).st_mode & 0o777 == 0o600


def test_atomic_env_store_replace_failure_preserves_original(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("APP_ID=app\nADMIN_USER_IDS=ou_old\n", encoding="utf-8")
    store = AtomicEnvFileStore(env_path)

    with patch("src.config.env_file_store.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            store.update_many({"ADMIN_USER_IDS": "ou_new"})

    assert env_path.read_text(encoding="utf-8") == (
        "APP_ID=app\nADMIN_USER_IDS=ou_old\n"
    )
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_atomic_env_store_serializes_concurrent_distinct_updates(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("BASE=kept\n", encoding="utf-8")

    def update(index: int) -> None:
        AtomicEnvFileStore(env_path).update_many({f"KEY_{index}": f"value_{index}"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(update, range(24)))

    lines = set(env_path.read_text(encoding="utf-8").splitlines())
    assert "BASE=kept" in lines
    for index in range(24):
        assert f"KEY_{index}=value_{index}" in lines
