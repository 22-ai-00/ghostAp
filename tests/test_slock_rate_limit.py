"""Production-boundary tests for Slock task-assignment rate limiting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handlers.slock import SlockHandler


def _handler() -> MagicMock:
    handler = MagicMock(spec=SlockHandler)
    handler._rate_limit_tracker = {}
    handler._prune_assign_rate_limit_tracker = (
        SlockHandler._prune_assign_rate_limit_tracker.__get__(handler, SlockHandler)
    )
    handler._check_assign_rate_limit = SlockHandler._check_assign_rate_limit.__get__(
        handler,
        SlockHandler,
    )
    return handler


def _engine(owner_id: str = "owner") -> MagicMock:
    engine = MagicMock()
    engine.channel.owner_id = owner_id
    return engine


def test_regular_user_is_blocked_at_limit_and_allowed_after_window() -> None:
    handler = _handler()
    settings = MagicMock(
        admin_user_ids=frozenset({"admin"}),
        slock_assign_rate_limit=2,
    )

    with (
        patch("src.thread.manager.get_current_sender_id", return_value="member"),
        patch("src.config.get_settings", return_value=settings),
        patch("time.time") as clock,
    ):
        clock.return_value = 100.0
        assert handler._check_assign_rate_limit(_engine(), "msg", "chat") is True
        assert handler._check_assign_rate_limit(_engine(), "msg", "chat") is True
        assert handler._check_assign_rate_limit(_engine(), "msg", "chat") is False

        clock.return_value = 161.0
        assert handler._check_assign_rate_limit(_engine(), "msg", "chat") is True

    handler.reply_text.assert_called_once()
    assert "每分钟最多 2 次" in handler.reply_text.call_args.args[1]


@pytest.mark.parametrize("operator_id", ["admin", "owner"])
def test_admin_and_owner_bypass_rate_limit(operator_id: str) -> None:
    handler = _handler()
    handler._rate_limit_tracker[f"chat:{operator_id}"] = [100.0, 100.0]
    settings = MagicMock(
        admin_user_ids=frozenset({"admin"}),
        slock_assign_rate_limit=2,
    )

    with (
        patch("src.thread.manager.get_current_sender_id", return_value=operator_id),
        patch("src.config.get_settings", return_value=settings),
        patch("time.time", return_value=100.0),
    ):
        assert handler._check_assign_rate_limit(_engine(), "msg", "chat") is True

    handler.reply_text.assert_not_called()
