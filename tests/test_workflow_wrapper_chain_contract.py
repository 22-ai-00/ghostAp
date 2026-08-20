from __future__ import annotations

import asyncio
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.acp.models import PromptResult
from src.acp.sync_adapter import SyncACPSession, start_session_with_retry
from src.agent_session.factory import create_engine_session
from src.agent_session.model_diagnostics import _apply_compaction_once
from src.agent_session.wrappers import ModelFailureAwareSession, RateLimitAwareSession


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        rate_limit_retry_enabled=True,
        rate_limit_max_retries=0,
        rate_limit_max_wait=1,
        rate_limit_base_wait=1,
        model_failure_compaction_enabled=True,
        model_failure_compaction_loop_window_s=180.0,
        model_failure_compaction_loop_max=3,
        model_failure_failover_map="",
        acp_startup_timeout=20,
    )


def _sync_acp_base(
    outcomes: list[PromptResult | BaseException],
    calls: list[dict[str, object]],
) -> SyncACPSession:
    session = object.__new__(SyncACPSession)
    session._prompt_lock = threading.Lock()
    session._acp_session = None
    pending = iter(outcomes)

    def _send_prompt_once(
        self,
        text,
        on_event=None,
        timeout=None,
        idle_timeout=None,
        activity_predicate=None,
        await_goal_quiescence=True,
        await_child_quiescence=False,
        replay_deferred_child_events=False,
    ):
        del self
        calls.append(
            {
                "text": text,
                "on_event": on_event,
                "timeout": timeout,
                "idle_timeout": idle_timeout,
                "activity_predicate": activity_predicate,
                "await_goal_quiescence": await_goal_quiescence,
                "await_child_quiescence": await_child_quiescence,
                "replay_deferred_child_events": replay_deferred_child_events,
            }
        )
        outcome = next(pending)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def _close(self):
        self._acp_session = None
        return None

    session._send_prompt_once = MethodType(_send_prompt_once, session)
    session.close = MethodType(_close, session)
    return session


def _factory_chain(tmp_path, base: SyncACPSession) -> ModelFailureAwareSession:
    settings = _settings()
    with (
        patch(
            "src.agent_session.factory._resolve_inputs",
            return_value=("codex", str(tmp_path), None),
        ),
        patch("src.agent_session.factory._start_base_session", return_value=base),
        patch("src.agent_session.factory.get_settings", return_value=settings),
        patch("src.agent_session.wrappers.get_settings", return_value=settings),
    ):
        session = create_engine_session("codex", str(tmp_path))
    assert isinstance(session, ModelFailureAwareSession)
    assert isinstance(session._inner, RateLimitAwareSession)
    assert session._inner._inner is base
    return session


def test_sync_acp_capture_policy_defaults_false(tmp_path) -> None:
    session = SyncACPSession(
        agent_type="codex",
        cwd=str(tmp_path),
        agent_cmd="codex",
    )

    assert session._capture_full_tool_content is False


def test_start_retry_forwards_explicit_capture_policy(tmp_path) -> None:
    captured: list[bool] = []

    class CapturingSession:
        def __init__(
            self,
            *,
            agent_type,
            cwd,
            capture_full_tool_content=False,
        ):
            del agent_type, cwd
            captured.append(capture_full_tool_content)

        def start(self, startup_timeout=20):
            return "started"

    start_session_with_retry(
        agent_type="codex",
        cwd=str(tmp_path),
        session_cls=CapturingSession,
        retries=1,
        capture_full_tool_content=True,
    )

    assert captured == [True]


def test_sync_session_forwards_capture_policy_to_async_session(tmp_path) -> None:
    session = SyncACPSession(
        agent_type="coco",
        cwd=str(tmp_path),
        agent_cmd="coco",
        capture_full_tool_content=True,
    )
    async_session = AsyncMock()
    async_session.start.return_value = "session-id"

    with patch(
        "src.acp.sync_adapter.ACPSession",
        return_value=async_session,
    ) as constructor:
        assert asyncio.run(session._start_session()) == "session-id"

    assert constructor.call_args.kwargs["capture_full_tool_content"] is True


def test_engine_factory_forwards_explicit_capture_policy(tmp_path) -> None:
    base = _sync_acp_base([], [])
    settings = _settings()
    with (
        patch(
            "src.agent_session.factory._resolve_inputs",
            return_value=("codex", str(tmp_path), None),
        ),
        patch(
            "src.agent_session.factory._start_base_session",
            return_value=base,
        ) as mock_start,
        patch("src.agent_session.factory.get_settings", return_value=settings),
        patch("src.agent_session.wrappers.get_settings", return_value=settings),
    ):
        create_engine_session(
            "codex",
            str(tmp_path),
            capture_full_tool_content=True,
        )

    assert mock_start.call_args.kwargs["capture_full_tool_content"] is True


@pytest.mark.parametrize("capture", [False, True])
def test_compaction_replacement_preserves_capture_policy(
    tmp_path,
    capture: bool,
) -> None:
    old = _sync_acp_base([], [])
    old._agent_type = "codex"
    old._cwd = str(tmp_path)
    old._agent_cmd = "codex"
    old._agent_args = ["acp", "serve"]
    old._capture_full_tool_content = capture
    constructor_kwargs: dict[str, object] = {}

    class Replacement:
        def start(self, startup_timeout=20):
            return "replacement"

    def build(**kwargs):
        constructor_kwargs.update(kwargs)
        return Replacement()

    assert _apply_compaction_once(
        session=old,
        session_builder=build,
        startup_timeout_s=1.0,
    ) is not None
    assert constructor_kwargs["capture_full_tool_content"] is capture


@pytest.mark.parametrize("capture", [False, True])
def test_failover_replacement_preserves_capture_policy(
    tmp_path,
    capture: bool,
) -> None:
    first = _sync_acp_base([], [])
    first._agent_cmd = "codex"
    first._agent_args = ["--model", "old"]
    first._agent_type = "codex"
    first._cwd = str(tmp_path)
    first._capture_full_tool_content = capture
    first.close = MethodType(lambda _self: None, first)
    replacement = _sync_acp_base([], [])
    replacement.start = MethodType(
        lambda _self, startup_timeout=20: "replacement",
        replacement,
    )
    constructor_kwargs: dict[str, object] = {}

    def build(**kwargs):
        constructor_kwargs.update(kwargs)
        return replacement

    session = _factory_chain(tmp_path, first)
    with (
        patch("src.agent_session.wrappers.SyncACPSession", side_effect=build),
        patch(
            "src.agent_session.wrappers._replace_model_in_agent_args",
            return_value=(["--model", "new"], True),
        ),
    ):
        assert session._do_failover(from_model="old", to_model="new") is True

    assert constructor_kwargs["capture_full_tool_content"] is capture


def test_real_factory_wrapper_chain_forwards_workflow_activity_contract(tmp_path) -> None:
    calls: list[dict[str, object]] = []
    base = _sync_acp_base([PromptResult(stop_reason="end_turn", text="ok")], calls)
    session = _factory_chain(tmp_path, base)

    def on_event(_event) -> None:
        return None

    def predicate(event) -> bool:
        return bool(event.text)

    result = session.send_prompt(
        "build",
        on_event=on_event,
        timeout=30,
        idle_timeout=5.0,
        activity_predicate=predicate,
    )

    assert result.text == "ok"
    assert calls == [
        {
            "text": "build",
            "on_event": on_event,
            "timeout": 30,
            "idle_timeout": 5.0,
            "activity_predicate": predicate,
            "await_goal_quiescence": True,
            "await_child_quiescence": False,
            "replay_deferred_child_events": False,
        }
    ]


def test_compaction_retry_preserves_workflow_activity_contract(tmp_path) -> None:
    first_calls: list[dict[str, object]] = []
    retry_calls: list[dict[str, object]] = []
    first = _sync_acp_base([RuntimeError("compaction required")], first_calls)
    replacement = _sync_acp_base(
        [PromptResult(stop_reason="end_turn", text="recovered")],
        retry_calls,
    )
    session = _factory_chain(tmp_path, first)
    session._compaction_action = lambda _session: replacement

    def predicate(event) -> bool:
        return bool(event.text)

    with patch(
        "src.agent_session.wrappers.classify_model_failure",
        return_value={"reason": "need_compaction"},
    ):
        result = session.send_prompt(
            "build",
            timeout=40,
            idle_timeout=7.0,
            activity_predicate=predicate,
        )

    assert result.text == "recovered"
    assert first_calls[0]["idle_timeout"] == 7.0
    assert retry_calls[0]["idle_timeout"] == 7.0
    assert first_calls[0]["activity_predicate"] is predicate
    assert retry_calls[0]["activity_predicate"] is predicate


def test_model_failure_wrapper_checks_cancel_before_replacement_send(tmp_path) -> None:
    first = _sync_acp_base([RuntimeError("compaction required")], [])
    replacement_calls: list[dict[str, object]] = []
    replacement = _sync_acp_base(
        [PromptResult(stop_reason="end_turn", text="must not send")],
        replacement_calls,
    )
    session = _factory_chain(tmp_path, first)

    def compact(_session):
        session._cancel_event.set()
        return replacement

    session._compaction_action = compact
    with (
        patch(
            "src.agent_session.wrappers.classify_model_failure",
            return_value={"reason": "need_compaction"},
        ),
        pytest.raises(RuntimeError, match="cancel"),
    ):
        session.send_prompt("build")

    assert replacement_calls == []


def test_compaction_and_failover_do_not_create_before_old_close_is_confirmed(
    monkeypatch,
) -> None:
    from src.agent_session.wrappers import ModelFailureAwareSession

    class OldSession:
        session_id = "old"
        _agent_type = "coco"
        _cwd = "/tmp"
        _agent_cmd = "coco"
        _agent_args = ["acp", "serve", "-c", "model.name=gpt-5.2"]

        def close(self):
            raise RuntimeError("old close is uncertain")

    class CandidateSession:
        def start(self, startup_timeout=60):
            return "candidate"

        def close(self):
            return None

    for mode in ("compaction", "failover"):
        old = OldSession()
        created = {"count": 0}

        def create_candidate(*args, **kwargs):
            created["count"] += 1
            return CandidateSession()

        monkeypatch.setattr(
            "src.agent_session.wrappers.SyncACPSession",
            create_candidate,
        )
        wrapper = ModelFailureAwareSession(
            old,
            compaction_action=lambda _session: create_candidate(),
        )

        replaced = (
            wrapper._do_compaction()
            if mode == "compaction"
            else wrapper._do_failover(
                from_model="gpt-5.2",
                to_model="gpt-5.1",
            )
        )

        assert replaced is False
        assert created["count"] == 0
        assert wrapper._inner is old


def test_cancel_after_candidate_creation_closes_candidate_without_swap() -> None:
    from src.agent_session.wrappers import ModelFailureAwareSession

    class OldSession:
        def close(self):
            return None

    class CandidateSession:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    old = OldSession()
    candidate = CandidateSession()
    wrapper = ModelFailureAwareSession(old)

    def create_after_cancel(_session):
        wrapper._cancel_event.set()
        return candidate

    wrapper._compaction_action = create_after_cancel

    assert wrapper._do_compaction() is False
    assert wrapper._inner is old
    assert candidate.close_calls == 1
    assert wrapper.uncertain_sessions == ()


def test_compaction_and_failover_replacement_peak_at_one_active_session(
    monkeypatch,
) -> None:
    from src.agent_session.wrappers import ModelFailureAwareSession

    for mode in ("compaction", "failover"):
        tracker = {"active": 1, "max_active": 1}

        class OldSession:
            session_id = "old"
            _agent_type = "coco"
            _cwd = "/tmp"
            _agent_cmd = "coco"
            _agent_args = ["acp", "serve", "-c", "model.name=gpt-5.2"]

            def __init__(self):
                self.closed = False

            def close(self):
                if not self.closed:
                    self.closed = True
                    tracker["active"] -= 1

        class CandidateSession:
            def __init__(self, **_kwargs):
                self.closed = False
                tracker["active"] += 1
                tracker["max_active"] = max(
                    tracker["max_active"],
                    tracker["active"],
                )

            def start(self, startup_timeout=60):
                return "candidate"

            def close(self):
                if not self.closed:
                    self.closed = True
                    tracker["active"] -= 1

        old = OldSession()
        monkeypatch.setattr(
            "src.agent_session.wrappers.SyncACPSession",
            CandidateSession,
        )
        wrapper = ModelFailureAwareSession(
            old,
            compaction_action=lambda _session: CandidateSession(),
        )

        replaced = (
            wrapper._do_compaction()
            if mode == "compaction"
            else wrapper._do_failover(
                from_model="gpt-5.2",
                to_model="gpt-5.1",
            )
        )

        assert replaced is True
        assert tracker == {"active": 1, "max_active": 1}
        wrapper.close()
        assert tracker["active"] == 0
