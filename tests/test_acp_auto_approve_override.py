"""Contracts for per-session ACP permission-policy overrides."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.acp.session as acp_session_module
import src.acp.sync_adapter as sync_adapter_module
import src.agent_session.factory as session_factory
from src.acp.session import ACPSession


@pytest.mark.parametrize(
    ("configured_default", "session_override", "expected"),
    [
        pytest.param(True, False, False, id="explicit-false-overrides-true"),
        pytest.param(False, True, True, id="explicit-true-overrides-false"),
        pytest.param(True, None, True, id="none-falls-back-to-settings"),
    ],
)
def test_acp_session_resolves_auto_approve_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    configured_default: bool,
    session_override: bool | None,
    expected: bool,
) -> None:
    captured: list[bool] = []

    class CapturingClient:
        def __init__(self, *args, **kwargs) -> None:
            captured.append(kwargs["auto_approve"])

    monkeypatch.setattr(acp_session_module, "GhostAPClient", CapturingClient)
    monkeypatch.setattr(
        acp_session_module,
        "get_settings",
        lambda: SimpleNamespace(
            acp_permission_auto_approve=configured_default,
            acp_stream_buffer_limit=0,
        ),
    )

    def stop_after_client_creation(*args, **kwargs):
        raise RuntimeError("test stopped after client construction")

    async def empty_stream_snippet(*args, **kwargs) -> str:
        return ""

    monkeypatch.setattr(
        acp_session_module,
        "spawn_agent_process",
        stop_after_client_creation,
    )
    monkeypatch.setattr(
        acp_session_module,
        "_read_stream_snippet",
        empty_stream_snippet,
    )

    session = ACPSession(
        "fake-agent",
        [],
        str(tmp_path),
        auto_approve=session_override,
    )
    session._bind_session_info_callback = lambda: None

    with pytest.raises(Exception, match="test stopped after client construction"):
        asyncio.run(session.start())

    assert captured == [expected]


@pytest.mark.parametrize(
    "session_override",
    [
        pytest.param(False, id="false"),
        pytest.param(True, id="true"),
        pytest.param(None, id="none"),
    ],
)
def test_engine_session_factory_preserves_auto_approve_to_sync_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    session_override: bool | None,
) -> None:
    captured: list[dict[str, object]] = []

    class CapturingSyncSession:
        def __init__(self, *args, **kwargs) -> None:
            captured.append(dict(kwargs))

        def start(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        sync_adapter_module,
        "SyncACPSession",
        CapturingSyncSession,
    )
    monkeypatch.setattr(
        session_factory,
        "RateLimitAwareSession",
        lambda session, *args, **kwargs: session,
    )
    monkeypatch.setattr(
        session_factory,
        "ModelFailureAwareSession",
        lambda session, *args, **kwargs: session,
    )
    monkeypatch.setattr(
        session_factory,
        "get_settings",
        lambda: MagicMock(rate_limit_retry_enabled=False),
    )

    session_factory.create_engine_session(
        "codex",
        str(tmp_path),
        auto_approve=session_override,
        startup_timeout=1,
        startup_retries=0,
        startup_log_failures=False,
    )

    assert len(captured) == 1
    assert captured[0]["auto_approve"] is session_override
