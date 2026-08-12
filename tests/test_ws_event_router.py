from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from src.feishu.message_cache import MessageCache


class _Cache:
    def __init__(self, duplicate: bool):
        self.duplicate = duplicate
        self.seen: list[str] = []

    def is_duplicate(self, message_id: str) -> bool:
        self.seen.append(message_id)
        return self.duplicate


def test_message_ingress_guard_owns_expiry_and_duplicate_checks():
    from src.feishu.ws_event_router import MessageIngressGuard

    cache = _Cache(duplicate=True)
    guard = MessageIngressGuard(
        message_cache=cache,
        message_expire_seconds=30,
        clock_ms=lambda: 1_000_000,
    )

    assert guard.is_message_expired(1_000_000 - 31_000) is True
    assert guard.is_message_expired(1_000_000 - 29_000) is False
    assert guard.is_message_expired(0) is False
    assert guard.is_duplicate_message("msg-1") is True
    assert cache.seen == ["msg-1"]


def test_message_ingress_reservation_survives_cache_ttl_and_lru_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.feishu.ws_event_router import MessageIngressGuard

    now = [100.0]
    monkeypatch.setattr("src.feishu.message_cache.time.time", lambda: now[0])
    cache = MessageCache(ttl=1, max_size=1, cleanup_interval=60)
    guard = MessageIngressGuard(
        message_cache=cache,
        message_expire_seconds=30,
    )

    owner = guard.reserve("msg-in-flight")
    assert owner is not None
    now[0] = 102.0
    other_owner = guard.reserve("msg-other")
    assert other_owner is not None

    assert guard.reserve("msg-in-flight") is None
    assert guard.owns("msg-in-flight", owner)


def test_message_ingress_release_and_commit_are_owner_cas() -> None:
    from src.feishu.ws_event_router import MessageIngressGuard

    cache = MessageCache(ttl=300, max_size=10)
    guard = MessageIngressGuard(
        message_cache=cache,
        message_expire_seconds=30,
    )

    first_owner = guard.reserve("msg-aba")
    assert first_owner is not None
    assert guard.release("msg-aba", first_owner)
    second_owner = guard.reserve("msg-aba")
    assert second_owner is not None
    assert second_owner != first_owner

    assert not guard.release("msg-aba", first_owner)
    assert not guard.commit("msg-aba", first_owner)
    assert guard.owns("msg-aba", second_owner)
    assert guard.commit("msg-aba", second_owner)
    assert guard.reserve("msg-aba") is None


def test_ws_event_router_has_explicit_exception_contract():
    from src.feishu.ws_event_router import WSErrorAction, classify_ws_error

    assert classify_ws_error(RuntimeError("import optional"), phase="import_guard").action == WSErrorAction.USE_COMPAT_FALLBACK
    assert classify_ws_error(RuntimeError("dispatch failed"), phase="dispatch").action == WSErrorAction.REPLY_INTERNAL_ERROR
    assert classify_ws_error(RuntimeError("cleanup failed"), phase="cleanup").action == WSErrorAction.LOG_AND_CONTINUE


def test_ws_error_handler_dispatches_user_visible_diagnostic_with_log(caplog):
    from src.feishu.ws_event_router import WSErrorAction, handle_ws_error

    replies: list[str] = []

    with caplog.at_level(logging.ERROR, logger="src.feishu.ws_event_router"):
        action = handle_ws_error(
            RuntimeError("dispatch failed"),
            phase="dispatch",
            reply_internal_error=lambda exc: replies.append(str(exc)),
        )

    assert action == WSErrorAction.REPLY_INTERNAL_ERROR
    assert replies == ["dispatch failed"]
    assert "WebSocket dispatch failed" in caplog.text


def test_ws_error_handler_recoverable_import_uses_fallback_without_user_reply(caplog):
    from src.feishu.ws_event_router import WSErrorAction, handle_ws_error

    fallbacks: list[str] = []

    with caplog.at_level(logging.WARNING, logger="src.feishu.ws_event_router"):
        action = handle_ws_error(
            RuntimeError("optional import missing"),
            phase="import_guard",
            compat_fallback=lambda exc: fallbacks.append(str(exc)),
        )

    assert action == WSErrorAction.USE_COMPAT_FALLBACK
    assert fallbacks == ["optional import missing"]
    assert "compat fallback" in caplog.text


def test_ws_error_handler_fatal_error_propagates_without_success_masking():
    from src.feishu.ws_event_router import handle_ws_error

    class FatalWSBug(Exception):
        pass

    with pytest.raises(FatalWSBug):
        handle_ws_error(FatalWSBug("router invariant broken"), phase="unexpected")


def test_ws_key_paths_do_not_leave_exception_taxonomy_as_backlog_debt():
    """Final guard: key WS broad catches must be intentional, not tracked debt."""

    root = Path(__file__).resolve().parents[1]
    ws_client_source = (root / "src" / "feishu" / "ws_client.py").read_text(encoding="utf-8")
    backlog = (root / ".Memory" / "Backlog.md").read_text(encoding="utf-8")
    memory = (root / ".Memory" / "2026-05-11.md").read_text(encoding="utf-8")

    assert "classify_ws_error" in ws_client_source
    assert "refactoring-analysis #6" not in backlog
    assert "| 6 | 宽泛 except Exception 泛滥 | 存在 |" in memory


def test_ws_client_dispatch_startup_paths_have_no_uncategorized_broad_catches():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src" / "feishu" / "ws_client.py").read_text(encoding="utf-8"))
    key_functions = {"_handle_message", "start", "_handle_card_action"}
    broad_by_function: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in key_functions:
            broad_by_function[node.name] = [
                handler.lineno
                for handler in ast.walk(node)
                if isinstance(handler, ast.ExceptHandler)
                and isinstance(handler.type, ast.Name)
                and handler.type.id == "Exception"
            ]

    assert key_functions.issubset(set(broad_by_function))
    assert broad_by_function == {name: [] for name in key_functions}
