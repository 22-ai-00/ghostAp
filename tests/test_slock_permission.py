"""Production-boundary tests for Slock admin/owner permission checks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handlers.slock import SlockHandler


def _handler(*, reply_succeeds: bool = True) -> MagicMock:
    handler = MagicMock(spec=SlockHandler)
    handler._has_slock_permission = SlockHandler._has_slock_permission.__get__(
        handler,
        SlockHandler,
    )
    handler._check_slock_permission = SlockHandler._check_slock_permission.__get__(
        handler,
        SlockHandler,
    )
    handler.reply_text.return_value = reply_succeeds
    return handler


def _engine(owner_id: str = "owner") -> MagicMock:
    engine = MagicMock()
    engine.channel.owner_id = owner_id
    return engine


@pytest.mark.parametrize(
    ("operator_id", "admin_ids", "owner_id", "expected"),
    [
        ("", frozenset(), "owner", True),
        ("admin", frozenset({"admin"}), "owner", True),
        ("owner", frozenset({"admin"}), "owner", True),
        ("member", frozenset({"admin"}), "owner", False),
        ("", frozenset({"admin"}), "owner", False),
    ],
)
def test_permission_uses_bootstrap_admin_and_owner_contract(
    operator_id: str,
    admin_ids: frozenset[str],
    owner_id: str,
    expected: bool,
) -> None:
    handler = _handler()
    settings = MagicMock(admin_user_ids=admin_ids)

    with (
        patch("src.thread.manager.get_current_sender_id", return_value=operator_id),
        patch("src.config.get_settings", return_value=settings),
    ):
        assert handler._has_slock_permission(_engine(owner_id)) is expected


@pytest.mark.parametrize("reply_succeeds", [True, False])
def test_permission_denial_replies_then_falls_back_to_chat(
    reply_succeeds: bool,
) -> None:
    handler = _handler(reply_succeeds=reply_succeeds)
    settings = MagicMock(admin_user_ids=frozenset({"admin"}))

    with (
        patch("src.thread.manager.get_current_sender_id", return_value="member"),
        patch("src.config.get_settings", return_value=settings),
    ):
        assert handler._check_slock_permission(_engine(), "msg", "chat") is False

    handler.reply_text.assert_called_once()
    assert "权限不足" in handler.reply_text.call_args.args[1]
    if reply_succeeds:
        handler.send_text_to_chat.assert_not_called()
    else:
        handler.send_text_to_chat.assert_called_once()
