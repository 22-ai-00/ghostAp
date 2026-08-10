"""End-to-end ACP stdio integration test.

This validates that GhostAP's ACP client sends JSON-RPC params that match
`agent-client-protocol`'s pydantic schema (incl. aliases like `sessionId`).

It spins up a minimal ACP agent in a subprocess (python -c) using
`acp.stdio.AgentSideConnection` and exercises:
- initialize
- new_session
- list_sessions (used by GhostAP health_check)
- prompt
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from src.acp.session import ACPSession

_FAKE_AGENT_CODE = textwrap.dedent(
    r"""
    import asyncio
    import sys
    import uuid

    from acp.helpers import update_agent_message_text
    from acp.schema import InitializeResponse, ListSessionsResponse, NewSessionResponse, PromptResponse
    from acp.stdio import AgentSideConnection


    class FakeAgent:
        def __init__(self):
            self._conn = None
            self._sessions = set()

        def on_connect(self, conn):
            self._conn = conn

        async def initialize(self, protocol_version: int, client_capabilities=None, client_info=None, **kwargs):
            return InitializeResponse(protocol_version=protocol_version)

        async def new_session(self, cwd: str, mcp_servers=None, **kwargs):
            sid = "s_" + uuid.uuid4().hex[:8]
            self._sessions.add(sid)
            return NewSessionResponse(session_id=sid)


        async def list_sessions(self, cwd=None, cursor=None, **kwargs):
            return ListSessionsResponse(sessions=[])

        async def prompt(self, prompt, session_id: str, **kwargs):
            # Emit a streaming message chunk so GhostAP can aggregate text.
            if self._conn is not None:
                await self._conn.session_update(session_id=session_id, update=update_agent_message_text("hello-from-fake"))
            return PromptResponse(stop_reason="end_turn")


    async def _make_stdio_streams():
        loop = asyncio.get_running_loop()

        reader = asyncio.StreamReader()
        reader_protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: reader_protocol, sys.stdin.buffer)

        transport, protocol = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout.buffer)
        writer = asyncio.StreamWriter(transport, protocol, None, loop)
        return reader, writer


    async def main():
        reader, writer = await _make_stdio_streams()
        conn = AgentSideConnection(FakeAgent(), writer, reader, listening=False)
        await conn.listen()


    if __name__ == "__main__":
        asyncio.run(main())
    """
).strip()


@pytest.mark.asyncio
async def test_acp_stdio_prompt_and_health_check(tmp_path):
    # Use a temp cwd so the agent is sandboxed.
    cwd = str(tmp_path)

    s = ACPSession(
        agent_cmd=sys.executable,
        agent_args=["-u", "-c", _FAKE_AGENT_CODE],
        cwd=cwd,
    )

    try:
        session_id = await s.start()
        assert session_id

        # health_check uses a non-mutating `session/list` roundtrip.
        assert await s.health_check(timeout=2.0) is True

        r = await s.prompt("ping")
        assert r.stop_reason
        assert "hello-from-fake" in (r.text or "")
    finally:
        await s.close()


def test_start_session_with_retry_logs_diagnostics_on_empty_error(monkeypatch, caplog):
    """回归：启动失败但异常 message 为空时，日志仍应包含 error_type/上下文，避免线上不可定位。"""
    import logging

    from src.acp.sync_adapter import start_session_with_retry

    class _EmptyError(RuntimeError):
        def __str__(self):
            return ""

    class _FakeSyncSession:
        def __init__(self, agent_type: str, cwd: str, model_name=None):
            self._agent_type = agent_type
            self._cwd = cwd
            self._agent_cmd = "codex"
            self._agent_args = ["code", "-t", "codex"]

        def describe_agent(self) -> str:
            return f"cmd={self._agent_cmd} args={' '.join(self._agent_args)} cwd={self._cwd}"

        def start(self, startup_timeout: float = 60) -> str:
            raise _EmptyError()

        def close(self):
            return

    # Patch SyncACPSession used inside start_session_with_retry
    monkeypatch.setattr("src.acp.sync_adapter.SyncACPSession", _FakeSyncSession)
    monkeypatch.setattr("src.acp.sync_adapter.get_settings", lambda: type("S", (), {"acp_startup_retries": 1})())

    caplog.set_level(logging.WARNING)
    with pytest.raises(Exception):
        start_session_with_retry(agent_type="codex", cwd="/tmp", startup_timeout=0.1, model_name="m")

    blob = "\n".join([r.getMessage() for r in caplog.records])
    assert "Engine session start failed" in blob
    assert "error_type" in blob
    assert "codex" in blob


def test_acp_manager_ensure_session_start_failure_empty_exc_has_non_empty_detail(monkeypatch, caplog):
    """回归：ACPSessionManager 启动失败且异常 message 为空时，不应出现空原因。

    - 日志：必须包含稳定字段 fail_reason/error_text
    - 抛错：最终 RuntimeError 的 detail 不得为空（避免线上 `...: ` 空串）
    """

    import logging

    import pytest

    from src.acp.manager import ACPSessionManager
    from src.acp.startup_utils import StartupOperationalError

    class _EmptyErr(StartupOperationalError):
        def __str__(self):
            return ""

    class _FakeSyncSession:
        def __init__(self, *args, **kwargs):
            self._agent_cmd = "codex"
            self._agent_args = ["acp", "serve"]
            self._cwd = str(kwargs.get("cwd") or ".")

        def describe_agent(self) -> str:
            return f"cmd={self._agent_cmd} args={' '.join(self._agent_args)} cwd={self._cwd}"

        def start(self, startup_timeout: float = 60) -> str:
            raise _EmptyErr()

        def close(self):
            return

        def load_local_history(self, *args, **kwargs):
            return []

    # Ensure we don't touch real ACP or external binaries.
    monkeypatch.setattr("src.acp.manager.SyncACPSession", _FakeSyncSession)
    monkeypatch.setattr("src.acp.manager.SyncClaudeCLISession", _FakeSyncSession)
    monkeypatch.setattr(
        "src.acp.manager.get_settings",
        lambda: type(
            "S",
            (),
            {
                "acp_startup_retries": 1,
                "acp_startup_timeout": 0.1,
                # diagnostics config defaults
                "acp_diagnostics_redact_enabled": True,
                "acp_diagnostics_redact_patterns": [],
                "acp_diagnostics_redact_replacement": "***REDACTED***",
                "acp_diagnostics_args_limit": 200,
                "acp_diagnostics_snippet_limit": 200,
                "acp_diagnostics_total_limit": 800,
            },
        )(),
    )

    caplog.set_level(logging.WARNING)

    # 注入 fake starter：冻结“可注入启动器”接口形状，避免未来解耦重构时回归。
    def _starter(**kw):
        assert kw.get("agent_type") == "coco"
        return (_FakeSyncSession(cwd=kw.get("cwd")), "", {"attempts": []})

    mgr = ACPSessionManager("coco", session_starter=_starter)

    with pytest.raises(RuntimeError) as ei:
        mgr.ensure_session(chat_id="c", project_id="p", cwd="/tmp", startup_timeout=0.1)

    # 1) log line must have non-empty error_text
    logs = "\n".join(r.getMessage() for r in caplog.records)
    assert "Session start failed" in logs
    assert "error_text=" in logs
    # 2) raised error detail must be non-empty
    assert str(ei.value).strip()
    assert ":" in str(ei.value)
