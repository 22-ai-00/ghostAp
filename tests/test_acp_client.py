"""Tests for acp.client — GhostAPClient event handling."""

import asyncio
import getpass
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from acp.schema import AgentMessageChunk, AgentThoughtChunk, SessionInfoUpdate

from src.acp.client import GhostAPClient, _parse_plan, _parse_tool_call
from src.acp.helper import SessionKeyCodec
from src.acp.models import ACPEvent, ACPGoalInfo, ACPSessionInfo, PromptResult
from src.acp.outcome import PromptOutcome, classify_prompt_result
from src.acp.sync_adapter import resolve_agent_spec
from src.utils.env import build_clean_env


def _session_info_update(codex: dict) -> SessionInfoUpdate:
    return SessionInfoUpdate.model_validate(
        {
            "sessionUpdate": "session_info_update",
            "_meta": {"codex": codex},
        }
    )


def _text_chunk(
    chunk_type: type[AgentMessageChunk] | type[AgentThoughtChunk],
    *,
    text: str,
    message_id: str,
    phase: str,
):
    session_update = (
        "agent_message_chunk"
        if chunk_type is AgentMessageChunk
        else "agent_thought_chunk"
    )
    return chunk_type.model_validate(
        {
            "sessionUpdate": session_update,
            "content": {"type": "text", "text": text},
            "messageId": message_id,
            "_meta": {"codex": {"phase": phase}},
        }
    )


def test_build_clean_env_supplies_user_scoped_uv_cache_default() -> None:
    env = build_clean_env({"HOME": "/tmp/ghostap-test-home"})

    cache_dir = Path(env["UV_CACHE_DIR"])
    cache_dir.relative_to(Path(tempfile.gettempdir()))
    get_uid = getattr(os, "getuid", None)
    identity = str(get_uid()) if callable(get_uid) else getpass.getuser()
    assert identity in cache_dir.name


def test_build_clean_env_uses_numeric_uid_when_available(monkeypatch) -> None:
    import src.utils.env as env_module

    monkeypatch.setattr(env_module.os, "getuid", lambda: 4242, raising=False)
    monkeypatch.setattr(
        env_module.getpass,
        "getuser",
        lambda: pytest.fail("username fallback must not run when getuid exists"),
    )

    env = env_module.build_clean_env({"HOME": "/tmp/ghostap-test-home"})

    assert Path(env["UV_CACHE_DIR"]).name == "ghostap-uv-cache-4242"


def test_build_clean_env_uses_username_when_getuid_is_unavailable(
    monkeypatch,
) -> None:
    import src.utils.env as env_module

    monkeypatch.delattr(env_module.os, "getuid", raising=False)
    monkeypatch.setattr(env_module.getpass, "getuser", lambda: "windows-user")

    env = env_module.build_clean_env({"HOME": "C:/Users/windows-user"})

    assert Path(env["UV_CACHE_DIR"]).name == "ghostap-uv-cache-windows-user"


def test_build_clean_env_preserves_explicit_uv_cache_override() -> None:
    env = build_clean_env(
        {
            "HOME": "/tmp/ghostap-test-home",
            "UV_CACHE_DIR": "/operator/uv-cache",
        }
    )

    assert env["UV_CACHE_DIR"] == "/operator/uv-cache"


def test_codex_goal_session_info_is_control_plane_not_render_event(
    tmp_path: Path,
) -> None:
    events: list[ACPEvent] = []
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=events.append,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )
    update = _session_info_update(
        {
            "goal": {
                "objective": "finish the repository task",
                "status": "active",
                "tokenBudget": 12000,
                "timeUsedSeconds": 15,
                "createdAt": "2026-08-06T00:19:21Z",
                "controlMethod": "_codex/session/goal_control",
            },
            "threadStatus": {"type": "active"},
        }
    )

    asyncio.run(client.session_update("session-goal", update))

    assert events == []
    assert controls[0][0] == "session-goal"
    info = controls[0][1]
    assert info.goal_known is True
    assert info.goal is not None
    assert info.goal.status == "active"
    assert info.goal.token_budget == 12000
    assert info.goal.time_used_seconds == 15
    assert info.goal.control_method == "_codex/session/goal_control"
    assert info.thread_status_known is True
    assert info.thread_status == "active"


@pytest.mark.parametrize(
    ("phase", "expected_phase"),
    [("commentary", "commentary"), ("final_answer", "final_answer"), ("unknown", None)],
)
def test_codex_agent_message_preserves_only_allowlisted_phase(
    tmp_path: Path,
    phase: str,
    expected_phase: str | None,
) -> None:
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))

    asyncio.run(
        client.session_update(
            "session-message",
            _text_chunk(
                AgentMessageChunk,
                text="public message",
                message_id="message-final-1",
                phase=phase,
            ),
        )
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type.name == "TEXT_CHUNK"
    assert event.message_id == "message-final-1"
    assert event.codex_message_phase == expected_phase


def test_codex_thought_never_carries_final_answer_marker(tmp_path: Path) -> None:
    events: list[ACPEvent] = []
    client = GhostAPClient(on_event=events.append, root_dir=str(tmp_path))

    asyncio.run(
        client.session_update(
            "session-thought",
            _text_chunk(
                AgentThoughtChunk,
                text="结论：这仍是内部推理",
                message_id="thought-final-1",
                phase="final_answer",
            ),
        )
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type.name == "THOUGHT_CHUNK"
    assert event.message_id is None
    assert event.codex_message_phase is None


@pytest.mark.parametrize("status", ["paused", "completed"])
def test_codex_goal_session_info_accepts_terminal_status(
    tmp_path: Path,
    status: str,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update(
                {"goal": {"objective": "finish", "status": status}}
            ),
        )
    )

    assert controls[0][1].goal_known is True
    assert controls[0][1].goal is not None
    assert controls[0][1].goal.status == status


def test_codex_goal_session_info_null_is_known_clear(tmp_path: Path) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update({"goal": None}),
        )
    )

    assert controls[0][1].goal_known is True
    assert controls[0][1].goal is None


def test_codex_goal_session_info_accepts_finite_unix_created_at(
    tmp_path: Path,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update(
                {
                    "goal": {
                        "objective": "finish",
                        "status": "active",
                        "createdAt": 1785975561.25,
                    }
                }
            ),
        )
    )

    assert controls[0][1].goal is not None
    assert controls[0][1].goal.created_at == 1785975561.25


def test_codex_goal_session_info_missing_created_at_is_explicitly_absent(
    tmp_path: Path,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update(
                {"goal": {"objective": "finish", "status": "active"}}
            ),
        )
    )

    assert controls[0][1].goal is not None
    assert controls[0][1].goal.created_at is None


def test_unknown_goal_status_remains_visible_to_fail_closed_tracker(
    tmp_path: Path,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update(
                {
                    "goal": {"objective": "finish", "status": "running"},
                    "threadStatus": {"type": "idle"},
                }
            ),
        )
    )

    assert controls[0][1].goal is not None
    assert controls[0][1].goal.status == "running"
    assert controls[0][1].goal.activity_state is None


@pytest.mark.parametrize(
    ("status", "activity_state"),
    [
        ("active", True),
        ("paused", False),
        ("blocked", False),
        ("completed", False),
        ("running", None),
    ],
)
def test_goal_status_classification_is_explicitly_tristate(
    status: str,
    activity_state: bool | None,
) -> None:
    goal = ACPGoalInfo(objective="finish", status=status)

    assert goal.activity_state is activity_state
    assert goal.is_active is (activity_state is True)


def test_unknown_thread_status_does_not_emit_control_update(
    tmp_path: Path,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            _session_info_update({"threadStatus": {"type": "busy"}}),
        )
    )

    assert controls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "renamed session", "sessionUpdate": "session_info_update"},
        {
            "sessionUpdate": "session_info_update",
            "_meta": {"codex": {"goal": {"objective": "missing status"}}},
        },
        {
            "sessionUpdate": "session_info_update",
            "_meta": {
                "codex": {
                    "goal": {
                        "objective": "finish",
                        "status": "active",
                        "tokenBudget": True,
                    }
                }
            },
        },
        {
            "sessionUpdate": "session_info_update",
            "_meta": {
                "codex": {
                    "goal": {
                        "objective": "finish",
                        "status": "active",
                        "timeUsedSeconds": float("inf"),
                    }
                }
            },
        },
        {
            "sessionUpdate": "session_info_update",
            "_meta": {"vendor": {"goal": None}},
        },
    ],
)
def test_untrustworthy_goal_session_info_cannot_clear_stored_state(
    tmp_path: Path,
    payload: dict,
) -> None:
    controls: list[tuple[str, ACPSessionInfo]] = []
    client = GhostAPClient(
        on_event=lambda _event: None,
        on_session_info=lambda sid, info: controls.append((sid, info)),
        root_dir=str(tmp_path),
    )

    asyncio.run(
        client.session_update(
            "session-goal",
            SessionInfoUpdate.model_validate(payload),
        )
    )

    assert controls == [] or controls[0][1].goal_known is False


def test_prompt_preserves_empty_stop_reason_for_fail_closed_classification(
    tmp_path: Path,
):

    from src.acp.outcome import PromptOutcome, classify_prompt_result
    from src.acp.session import ACPSession

    class FakeConn:
        async def prompt(self, **_kwargs):
            return SimpleNamespace(stop_reason="")

    session = ACPSession(agent_cmd="test", agent_args=[], cwd=str(tmp_path))
    session._conn = FakeConn()
    session._session_id = "session-empty-stop"

    result = asyncio.run(session.prompt("run"))
    assessment = classify_prompt_result(result)

    assert result.stop_reason == ""
    assert assessment.outcome is PromptOutcome.INCOMPLETE
    assert assessment.stop_reason == "missing_stop_reason"


def test_acp_manager_retries_start_failure(monkeypatch, caplog):

    from src.acp import manager as mgr
    from tests.helpers import FakeSessionBase

    calls = {"start": 0}

    class FakeSession(FakeSessionBase):
        def start(self, startup_timeout: float = 60):
            calls["start"] += 1
            if calls["start"] < 3:
                raise TimeoutError("startup timeout")
            self.session_id = "s_ok"
            return self.session_id

    monkeypatch.setattr(mgr, "SyncACPSession", FakeSession)
    monkeypatch.setattr(
        mgr, "get_settings", lambda: SimpleNamespace(acp_startup_retries=3, acp_healthcheck_timeout=0.01)
    )

    caplog.set_level(logging.WARNING)
    # 默认路径仍然支持「少参数」构造
    m = mgr.ACPSessionManager("coco", session_timeout=999999)
    # The startup timeout is one absolute budget shared by all retries.  Give
    # this retry-behaviour test enough budget for its two intentional backoffs.
    s = m.ensure_session("chat1", cwd=".", startup_timeout=3.0)
    assert s.session_id == "s_ok"
    assert calls["start"] == 3

    # 启动失败日志应包含稳定字段（即便具体值为空）
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "Session start failed" in joined
    assert '"cmd"' in joined
    assert '"args"' in joined
    assert '"rc"' in joined
    assert '"stdout_snippet"' in joined
    assert '"stderr_snippet"' in joined


def test_supports_acp_serve_unsets_claudecode(monkeypatch):
    """ACP serve 探测不应继承 nested-session guard 环境变量。"""

    from src.acp import sync_adapter as sa

    # lru_cache: ensure isolation
    try:
        sa._supports_acp_serve.cache_clear()
    except Exception:
        pass

    calls = {"env": None, "queue": None}

    def fake_run(cmd, capture_output, text, timeout, env=None):
        calls["env"] = env
        return SimpleNamespace(stdout="ACP Server", stderr="")

    monkeypatch.setattr(sa.subprocess, "run", fake_run)
    with monkeypatch.context() as m:
        m.setenv("CLAUDECODE", "1")
        assert sa._supports_acp_serve("claude") is True
        assert calls["env"] is not None
        assert "CLAUDECODE" not in calls["env"]


def test_acp_session_start_passes_env_without_claudecode(monkeypatch):
    """ACPSession 启动时应主动剔除 CLAUDECODE，避免 Claude nested-session 检测。"""

    import src.acp.session as session_mod
    from src.acp.session import ACPSession

    calls = {
        "env": None,
        "capture_full_tool_content": None,
        "trust_codex_extensions": None,
    }
    real_client = session_mod.GhostAPClient

    def capturing_client(*args, **kwargs):
        calls["capture_full_tool_content"] = kwargs.get(
            "capture_full_tool_content"
        )
        calls["trust_codex_extensions"] = kwargs.get(
            "trust_codex_extensions"
        )
        return real_client(*args, **kwargs)

    class FakeConn:
        async def initialize(self, protocol_version: int = 1):
            return None

        async def new_session(self, cwd: str):
            return SimpleNamespace(session_id="s_test")

    class FakeProc:
        returncode = None

    class FakeCtx:
        async def __aenter__(self):
            return FakeConn(), FakeProc()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    def fake_spawn(to_client, command, *args, env=None, cwd=None, transport_kwargs=None, **kw):
        calls["env"] = env
        calls["queue"] = kw.get("queue")
        return FakeCtx()

    monkeypatch.setattr(session_mod, "spawn_agent_process", fake_spawn)
    monkeypatch.setattr(session_mod, "GhostAPClient", capturing_client)
    monkeypatch.setattr(
        session_mod,
        "get_settings",
        lambda: SimpleNamespace(acp_stream_buffer_limit=0),
    )

    with monkeypatch.context() as m:
        m.setenv("CLAUDECODE", "1")
        s = ACPSession(
            agent_cmd="claude",
            agent_args=["acp", "serve"],
            cwd="/tmp",
            capture_full_tool_content=True,
        )
        sid = asyncio.run(s.start())
        assert sid == "s_test"
        assert calls["capture_full_tool_content"] is True
        assert calls["trust_codex_extensions"] is False
        assert calls["env"] is not None
        assert "CLAUDECODE" not in calls["env"]

        async def allow_full_access(*_args, **_kwargs):
            return True

        codex_session = ACPSession(
            agent_cmd="codex",
            agent_args=["@agentclientprotocol/codex-acp@1.2.0"],
            cwd="/tmp",
        )
        codex_session.set_config_option = allow_full_access
        assert asyncio.run(codex_session.start()) == "s_test"
        assert calls["trust_codex_extensions"] is True

    async def publish_after_close() -> None:
        queue = calls["queue"]
        assert queue is not None
        await queue.close()
        await queue.publish(object())

    asyncio.run(publish_after_close())


def test_acp_session_start_failure_has_fail_phase(monkeypatch):
    """ACPSession.start 失败时应抛 ACPStartupError 且携带 fail_phase（spawn/initialize/new_session）。"""

    import src.acp.session as session_mod
    from src.acp.session import ACPSession, ACPStartupError

    class FakeProc:
        returncode = 7
        stdout = None
        stderr = None

    class FakeConn:
        async def initialize(self, protocol_version: int = 1):
            raise RuntimeError("init failed")

    class FakeCtx:
        async def __aenter__(self):
            return FakeConn(), FakeProc()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    def fake_spawn(to_client, command, *args, env=None, cwd=None, transport_kwargs=None, **kw):
        return FakeCtx()

    monkeypatch.setattr(session_mod, "spawn_agent_process", fake_spawn)
    monkeypatch.setattr(
        session_mod,
        "get_settings",
        lambda: SimpleNamespace(acp_stream_buffer_limit=0),
    )

    s = ACPSession(agent_cmd="claude", agent_args=["acp", "serve"], cwd="/tmp")
    with pytest.raises(ACPStartupError) as ctx:
        asyncio.run(s.start())

    e = ctx.value
    assert getattr(e, "fail_phase", "") in ("initialize", "spawn", "new_session", "unknown")


def test_acp_health_check_uses_non_mutating_session_list_probe():
    """Health checks must not reload or otherwise mutate the active session."""

    from src.acp.session import ACPSession

    calls: list[tuple[str, object]] = []

    class FakeConn:
        async def list_sessions(self, *, cwd=None):
            calls.append(("list_sessions", cwd))
            return SimpleNamespace(sessions=[])


    session = ACPSession(agent_cmd="traex", agent_args=["acp", "serve"], cwd="/repo")
    session._proc = SimpleNamespace(returncode=None)
    session._conn = FakeConn()
    session._session_id = "s_live"

    assert asyncio.run(session.health_check(timeout=0.1)) is True
    assert calls == [("list_sessions", "/repo")]

    session._session_id = ""
    assert asyncio.run(session.health_check(timeout=0.1)) is False
    assert calls == [("list_sessions", "/repo")]


def test_acp_manager_unhealthy_session_is_cleaned(monkeypatch):
    import time as _time

    from src.acp import manager as mgr

    class DeadSession:
        def __init__(self):
            self.session_id = "s_dead"
            # Idle > 30s to trigger health check path in get_session
            self.last_active = _time.time() - 60
            self.message_count = 0
            self.closed = False

        def is_server_running(self) -> bool:
            return False  # process is dead

        def to_snapshot(self):
            return {"session_id": self.session_id}

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        mgr, "get_settings", lambda: SimpleNamespace(acp_healthcheck_timeout=0.01, acp_startup_retries=1)
    )

    m = mgr.ACPSessionManager("coco", session_timeout=999999)
    dead = DeadSession()
    key = SessionKeyCodec.encode("chat1")
    m._sessions[key] = dead

    assert m.get_session("chat1") is None
    assert dead.closed is True
    assert key not in m._sessions


def test_manager_marks_only_the_active_prompt_generation_as_user_cancelled() -> None:
    from src.acp.manager import ACPSessionManager

    class _Session:
        def __init__(self) -> None:
            self.session_id = "session-user-cancel"
            self.last_active = time.time()
            self.message_count = 0
            self.marked: list[int] = []
            self.order: list[str] = []

        def active_prompt_generation(self) -> int | None:
            return 7

        def mark_user_cancel(self, generation: int) -> None:
            self.order.append("mark")
            self.marked.append(generation)

        def cancel(self, *, wait: bool, timeout: float) -> bool:
            self.order.append("cancel")
            assert wait is False
            assert timeout == 2.0
            return True

    manager = ACPSessionManager("coco")
    session = _Session()
    manager._sessions[SessionKeyCodec.encode("chat-user-cancel")] = session

    assert manager.cancel_session("chat-user-cancel", user_initiated=True) is True
    assert session.marked == [7]
    assert session.order == ["mark", "cancel"]


def test_sync_adapter_user_cancel_marker_is_bound_to_active_generation(
    monkeypatch,
) -> None:
    from src.acp.sync_adapter import SyncACPSession

    session = SyncACPSession("coco", "/tmp", agent_cmd="coco")
    generations: list[int] = []

    def cancelled_turn(*_args, **_kwargs):
        generation = session.active_prompt_generation()
        assert generation is not None
        generations.append(generation)
        session.mark_user_cancel(generation)
        return PromptResult(stop_reason="cancelled")

    monkeypatch.setattr(session, "_send_prompt_once", cancelled_turn)
    first = session.send_prompt("first")

    def stale_marker_turn(*_args, **_kwargs):
        session.mark_user_cancel(generations[0])
        return PromptResult(stop_reason="cancelled")

    monkeypatch.setattr(session, "_send_prompt_once", stale_marker_turn)
    second = session.send_prompt("second")

    assert first.cancellation_source == "user"
    assert second.cancellation_source == "provider"


def test_sync_adapter_legacy_factory_lazily_initializes_generation_state(
    monkeypatch,
) -> None:
    """Older wrapper factories only initialized the prompt serialization lock."""
    import threading

    from src.acp.sync_adapter import SyncACPSession

    session = object.__new__(SyncACPSession)
    session._prompt_lock = threading.Lock()
    monkeypatch.setattr(
        session,
        "_send_prompt_once",
        lambda *_args, **_kwargs: PromptResult(stop_reason="end_turn"),
    )

    result = session.send_prompt("legacy factory")

    assert result.stop_reason == "end_turn"
    assert session.active_prompt_generation() is None


def test_acp_manager_session_starter_success_is_not_overwritten(monkeypatch):
    """回归：session_starter 成功返回后不应被默认路径覆盖。"""

    from src.acp import manager as mgr

    class _StarterSession:
        def __init__(self):
            self.session_id = "sid_from_starter"
            self.last_active = 123.0
            self.message_count = 7

        def describe_agent(self):
            return "starter"


        def load_local_history(self, *args, **kwargs):
            return []

        def to_snapshot(self):
            return {"session_id": self.session_id}

        def close(self):
            return None

        def is_server_running(self) -> bool:
            return True

        def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
            return True

    # If fallback path is entered, this fake will explode and fail the test.
    class _ShouldNotBeUsed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("fallback SyncACPSession should not be used")

    monkeypatch.setattr(mgr, "SyncACPSession", _ShouldNotBeUsed)
    monkeypatch.setattr(
        mgr, "get_settings", lambda: SimpleNamespace(acp_healthcheck_timeout=0.01, acp_startup_retries=1)
    )

    def _starter(**kwargs):
        return (_StarterSession(), "sid_from_starter", {"attempts": []})

    m = mgr.ACPSessionManager("coco", session_starter=_starter)
    s = m.ensure_session("chat1", cwd=".", startup_timeout=0.01)
    assert s.session_id == "sid_from_starter"


class MockToolCallStart:
    """Mock ToolCallStart ACP schema object."""

    def __init__(
        self,
        tool_call_id="tc1",
        title="Read file",
        kind="read",
        status="in_progress",
        locations=None,
        raw_input=None,
        raw_output=None,
        field_meta=None,
    ):
        self.tool_call_id = tool_call_id
        self.title = title
        self.kind = kind
        self.status = status
        self.locations = locations or []
        self.raw_input = raw_input
        self.raw_output = raw_output
        self.field_meta = field_meta


class MockToolCallProgress:
    """Mock ToolCallProgress ACP schema object."""

    def __init__(
        self,
        tool_call_id="tc1",
        title="Read file",
        kind="read",
        status="completed",
        locations=None,
        raw_input=None,
        raw_output=None,
        field_meta=None,
    ):
        self.tool_call_id = tool_call_id
        self.title = title
        self.kind = kind
        self.status = status
        self.locations = locations or []
        self.raw_input = raw_input
        self.raw_output = raw_output
        self.field_meta = field_meta


class MockLocation:
    def __init__(self, path):
        self.path = path


class MockPlanEntry:
    def __init__(self, content, priority="medium", status="pending"):
        self.content = content
        self.priority = priority
        self.status = status


class TestParseToolCall:

    def test_full_tool_content_capture_is_opt_in_and_bounded_by_default(self):
        tail = "RAW_TOOL_OUTPUT_TAIL"
        raw_output = {"stdout": "x" * 13000 + tail, "token": "secret"}
        update = MockToolCallProgress(
            title="bash",
            kind="execute",
            status="completed",
            raw_input={"command": "echo test"},
            raw_output=raw_output,
        )

        compact = _parse_tool_call(update)
        captured = _parse_tool_call(update, capture_full_tool_content=True)

        assert compact.full_content is None
        assert tail not in compact.content
        assert compact.content.endswith("\n... (truncated)")
        assert captured.full_content == raw_output
        assert tail in captured.full_content["stdout"]


    def test_agent_tool_keeps_task_description_for_task_cards(self):
        update = MockToolCallStart(
            title="Agent",
            kind="other",
            status="in_progress",
            raw_input={
                "description": "实现后端接口",
                "prompt": "请实现 `/api/tasks` 接口并补测试",
                "subagent_type": "Explore",
            },
        )
        tc = _parse_tool_call(update)
        assert "实现后端接口" in tc.content

    def test_codex_context_compaction_is_nonblocking_bookkeeping(self):
        update = MockToolCallStart(
            tool_call_id="context-compaction-private",
            title="Context compacting",
            kind="other",
            status="in_progress",
            field_meta={"contextCompaction": True},
        )

        tc = _parse_tool_call(update, trust_codex_extensions=True)

        assert tc.is_context_compaction is True
        assert classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=[tc])
        ).outcome is PromptOutcome.COMPLETED

    @pytest.mark.parametrize(
        ("kind", "marker", "trust_codex_extensions"),
        [
            ("other", "true", True),
            ("other", True, False),
            ("execute", True, True),
        ],
    )
    def test_untrusted_context_compaction_marker_remains_blocking(
        self,
        kind: str,
        marker: object,
        trust_codex_extensions: bool,
    ):
        update = MockToolCallStart(
            tool_call_id="untrusted-context-compaction",
            title="Context compacting",
            kind=kind,
            status="in_progress",
            field_meta={"contextCompaction": marker},
        )

        tc = _parse_tool_call(
            update,
            trust_codex_extensions=trust_codex_extensions,
        )

        assert classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=[tc])
        ).outcome is PromptOutcome.INCOMPLETE

    def test_codex_context_compaction_unknown_status_remains_blocking(self):
        update = MockToolCallStart(
            tool_call_id="unknown-context-compaction",
            title="Context compacting",
            kind="other",
            status="unknown",
            field_meta={"contextCompaction": True},
        )

        tc = _parse_tool_call(update, trust_codex_extensions=True)

        assert tc.is_context_compaction is True
        assert classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=[tc])
        ).outcome is PromptOutcome.INCOMPLETE

    def test_codex_subagent_activity_metadata_is_normalized(self):
        update = MockToolCallStart(
            tool_call_id="activity-call-private",
            title="Start subagent card-audit",
            kind="other",
            status="in_progress",
            raw_input={
                "agentThreadId": "thread-private",
                "agentPath": "/root/card-audit",
                "activityKind": "started",
            },
            field_meta={
                "codex": {
                    "subagent": {
                        "threadId": "thread-private",
                        "path": "/root/card-audit",
                        "activity": "started",
                    }
                }
            },
        )

        tc = _parse_tool_call(update)

        assert tc.subagent_source_id == "thread-private"
        assert tc.subagent_path == "/root/card-audit"
        assert tc.subagent_activity == "started"

    @pytest.mark.parametrize(
        ("container_name", "container"),
        [
            ("subagent", "malformed-container"),
            ("collaboration", "malformed-container"),
            ("subagent", {}),
            ("collaboration", {}),
        ],
    )
    def test_codex_malformed_child_metadata_container_fails_closed(
        self,
        container_name: str,
        container: object,
    ):
        update = MockToolCallStart(
            tool_call_id="malformed-child-container",
            title="child metadata",
            kind="other",
            status="completed",
            field_meta={
                "codex": {container_name: container},
            },
        )

        tc = _parse_tool_call(update)

        assert tc.child_metadata_malformed is True
        assessment = classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=[tc])
        )
        assert assessment.outcome is PromptOutcome.INCOMPLETE
        assert "malformed" in assessment.incomplete_tool_diagnostics[0]

    def test_codex_non_string_collaboration_tool_fails_closed(self):
        update = MockToolCallStart(
            tool_call_id="malformed-collaboration-tool",
            title="child metadata",
            kind="other",
            status="completed",
            field_meta={
                "codex": {"collaboration": {"tool": 123}},
            },
        )

        tc = _parse_tool_call(update)

        assert tc.child_metadata_malformed is True
        assert classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=[tc])
        ).outcome is PromptOutcome.INCOMPLETE

    @pytest.mark.parametrize(
        "collaboration",
        [
            {"senderThreadId": "thread-parent"},
            {"receiverThreadIds": []},
            {"model": "gpt-test"},
        ],
    )
    def test_codex_collaboration_without_tool_fails_closed(
        self,
        collaboration: dict[str, object],
    ):
        update = MockToolCallStart(
            tool_call_id="missing-collaboration-tool",
            title="child metadata",
            kind="other",
            status="completed",
            field_meta={"codex": {"collaboration": collaboration}},
        )

        tc = _parse_tool_call(update)

        assert tc.child_metadata_malformed is True
        assert classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=[tc])
        ).outcome is PromptOutcome.INCOMPLETE

    def test_codex_namespaced_agents_states_are_observational(self):
        update = MockToolCallStart(
            tool_call_id="collaboration-call-private",
            title="wait_agent",
            kind="other",
            status="completed",
            raw_input={
                "prompt": "等待卡片审计",
                "receiverThreadIds": ["thread-a", "thread-b"],
                "agentsStates": {
                    "thread-a": {"status": "running", "message": "正在核查普通卡"},
                    "thread-b": {"status": "completed", "message": "已核查 Deep 卡"},
                },
                "model": "gpt-test",
                "reasoningEffort": "high",
            },
            raw_output={"result": "collaboration snapshot"},
            field_meta={
                "codex": {
                    "collaboration": {
                        "tool": "wait_agent",
                        "senderThreadId": "thread-parent",
                        "receiverThreadIds": ["thread-a", "thread-b"],
                    }
                }
            },
        )

        tc = _parse_tool_call(update)

        assert tc.collaboration_tool == "wait_agent"
        assert tc.collaboration_receivers == ("thread-a", "thread-b")
        assert tc.collaboration_model == "gpt-test"
        assert tc.content == "等待卡片审计"
        assert tc.subagent_states == (
            {"source_id": "thread-a", "status": "running", "message": "正在核查普通卡"},
            {"source_id": "thread-b", "status": "completed", "message": "已核查 Deep 卡"},
        )


    def test_unknown_tool_cannot_claim_compatible_collaboration_state(self):
        update = MockToolCallProgress(
            tool_call_id="untrusted-compatible-shape",
            title="unknown_tool",
            kind="other",
            status="completed",
            raw_input={
                "agentsStates": {
                    "thread-a": {"status": "completed", "message": "done"},
                },
            },
        )

        tc = _parse_tool_call(update)

        assert tc.collaboration_tool is None
        assert tc.subagent_states == ()

    def test_codex_collaboration_rejects_scalar_receiver_ids(self):
        update = MockToolCallStart(
            tool_call_id="collaboration-call-private",
            title="wait_agent",
            kind="other",
            status="completed",
            raw_input={
                "receiverThreadIds": "thread-a",
                "agentsStates": {
                    "thread-a": {"status": "completed", "message": "已核查"},
                },
            },
            field_meta={
                "codex": {
                    "collaboration": {
                        "tool": "wait_agent",
                        "receiverThreadIds": "thread-a",
                    }
                }
            },
        )

        tc = _parse_tool_call(update)

        assert tc.collaboration_receivers == ()
        assert tc.subagent_states == (
            {"source_id": "thread-a", "status": "completed", "message": "已核查"},
            {"source_id": "", "status": "malformed", "message": ""},
        )
        assert classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=[tc])
        ).outcome is PromptOutcome.INCOMPLETE

    def test_codex_collaboration_marks_mixed_receiver_ids_malformed(self):
        update = MockToolCallStart(
            tool_call_id="collaboration-call-private",
            title="list_agents",
            kind="other",
            status="completed",
            raw_input={
                "agentsStates": {
                    "thread-a": {"status": "completed", "message": "done"},
                },
            },
            field_meta={
                "codex": {
                    "collaboration": {
                        "tool": "list_agents",
                        "receiverThreadIds": ["thread-a", 123],
                    }
                }
            },
        )

        tc = _parse_tool_call(update)

        assert tc.collaboration_receivers == ("thread-a",)
        assert tc.subagent_states[-1] == {
            "source_id": "",
            "status": "malformed",
            "message": "",
        }

    def test_codex_numeric_subagent_identity_is_not_string_coerced(self):
        update = MockToolCallStart(
            tool_call_id="activity-call-private",
            title="Interrupt subagent",
            kind="other",
            status="completed",
            field_meta={
                "codex": {
                    "subagent": {
                        "threadId": 123,
                        "path": "/root/reviewer",
                        "activity": "interrupted",
                    }
                }
            },
        )

        tc = _parse_tool_call(update)

        assert tc.subagent_source_id is None
        assert tc.subagent_activity == "interrupted"
        assert classify_prompt_result(
            PromptResult(stop_reason="end_turn", tool_calls=[tc])
        ).outcome is PromptOutcome.INCOMPLETE

    def test_codex_collaboration_receiver_without_state_is_pending(self):
        update = MockToolCallStart(
            tool_call_id="spawn-call-private",
            title="spawn_agent",
            kind="other",
            status="completed",
            raw_input={"receiverThreadIds": ["thread-a"]},
            field_meta={
                "codex": {
                    "collaboration": {
                        "tool": "spawn_agent",
                        "receiverThreadIds": ["thread-a"],
                    }
                }
            },
        )

        tc = _parse_tool_call(update)

        assert tc.subagent_states == (
            {"source_id": "thread-a", "status": "pending", "message": ""},
        )

    def test_codex_collaboration_preserves_malformed_state_evidence(self):
        update = MockToolCallStart(
            tool_call_id="wait-call-private",
            title="wait_agent",
            kind="other",
            status="completed",
            raw_input={
                "receiverThreadIds": ["thread-a"],
                "agentsStates": "provider-malformed-secret",
            },
            field_meta={
                "codex": {
                    "collaboration": {
                        "tool": "wait_agent",
                        "receiverThreadIds": ["thread-a"],
                    }
                }
            },
        )

        tc = _parse_tool_call(update)

        assert tc.subagent_states == (
            {"source_id": "thread-a", "status": "malformed", "message": ""},
        )


class TestParsePlan:

    def test_skips_empty_entries(self):
        class MockAgentPlanUpdate:
            entries = [
                MockPlanEntry("", status="completed"),
                MockPlanEntry("   ", status="completed"),
                MockPlanEntry(None, status="completed"),
                MockPlanEntry("Real step", status="pending"),
            ]

        plan = _parse_plan(MockAgentPlanUpdate())
        assert [e.content for e in plan.entries] == ["Real step"]

    def test_empty_plan(self):
        class MockAgentPlanUpdate:
            entries = []

        plan = _parse_plan(MockAgentPlanUpdate())
        assert plan.entries == []


class TestGhostAPClient:
    def setup_method(self):
        self.events: list[ACPEvent] = []
        self.client = GhostAPClient(on_event=self.events.append)

    def _run_async(self, coro):
        """Run async coroutine in sync tests (Py3.12-safe)."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


    def test_request_permission_uses_backend_allow_always_when_needed(self):
        option = MagicMock()
        option.kind = "allow_always"
        option.option_id = "unsafe-fallback"

        result = self._run_async(
            self.client.request_permission(
                options=[option],
                session_id="s1",
                tool_call=MagicMock(),
            )
        )

        assert result.outcome.outcome == "selected"
        assert result.outcome.option_id == "unsafe-fallback"


    def test_tool_call_copies_codex_subagent_source_to_event(self):
        from acp.schema import ToolCallStart

        update = ToolCallStart.model_validate(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "activity-call-private",
                "title": "Start subagent card-audit",
                "kind": "other",
                "status": "in_progress",
                "rawInput": {
                    "agentThreadId": "thread-private",
                    "agentPath": "/root/card-audit",
                    "activityKind": "started",
                },
                "_meta": {
                    "codex": {
                        "subagent": {
                            "threadId": "thread-private",
                            "path": "/root/card-audit",
                            "activity": "started",
                        }
                    }
                },
            }
        )

        self.client._handle_tool_call_start(update)

        assert len(self.events) == 1
        assert self.events[0].source_id == "thread-private"
        assert self.events[0].tool_call is not None
        assert self.events[0].tool_call.subagent_activity == "started"



def test_read_text_file_can_resolve_outside_session_root(tmp_path: Path):
    root = str(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    client = GhostAPClient(on_event=lambda e: None, root_dir=root)
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        resp = loop.run_until_complete(
            client.read_text_file("s1", f"../{outside.name}")
        )
        assert resp.content == "outside"
    finally:
        loop.close()


def test_resolve_agent_spec_coco_has_command():
    if not shutil.which("coco"):
        pytest.skip("coco binary not available")
    cmd, args = resolve_agent_spec("coco")
    assert cmd == "coco"
    assert args == ["acp", "serve"]


def test_acp_011_permission_arguments_keep_allow_once_selection():
    client = GhostAPClient(on_event=lambda _event: None)
    option = MagicMock()
    option.kind = "allow_once"
    option.option_id = "allow-once"

    response = asyncio.run(
        client.request_permission("session-1", MagicMock(), [option])
    )

    assert response.outcome.outcome == "selected"
    assert response.outcome.option_id == "allow-once"
