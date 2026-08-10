from __future__ import annotations

import threading
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest

from src.acp.models import PromptResult
from src.acp.sync_adapter import SyncACPSession
from src.agent_session.factory import create_engine_session
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
    session._tool_filter = None
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
        session = create_engine_session(
            "codex",
            str(tmp_path),
            require_tool_filter=True,
        )
    assert isinstance(session, ModelFailureAwareSession)
    assert isinstance(session._inner, RateLimitAwareSession)
    assert session._inner._inner is base
    return session


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


def _deny_mutation(tool_name: str, _params: dict | None) -> bool:
    normalized = tool_name.lower()
    return not any(
        token in normalized
        for token in ("shell", "write", "network")
    )


def _assert_mutation_filter_preserved(session: SyncACPSession) -> None:
    tool_filter = session.get_tool_filter()
    assert tool_filter is _deny_mutation
    assert tool_filter("shell", {}) is False
    assert tool_filter("write_file", {}) is False
    assert tool_filter("network_request", {}) is False


def test_compaction_replacement_inherits_deny_mutation_filter(tmp_path) -> None:
    first = _sync_acp_base([RuntimeError("compaction required")], [])
    replacement = _sync_acp_base(
        [PromptResult(stop_reason="end_turn", text="recovered")],
        [],
    )
    session = _factory_chain(tmp_path, first)
    session.set_tool_filter(_deny_mutation)
    session._compaction_action = lambda _session: replacement

    with patch(
        "src.agent_session.wrappers.classify_model_failure",
        return_value={"reason": "need_compaction"},
    ):
        assert session.send_prompt("build").text == "recovered"

    _assert_mutation_filter_preserved(replacement)


def test_failover_replacement_inherits_deny_mutation_filter(tmp_path) -> None:
    first = _sync_acp_base([], [])
    first._agent_cmd = "codex"
    first._agent_args = ["--model", "old"]
    first._agent_type = "codex"
    first._cwd = str(tmp_path)
    first.close = MethodType(lambda _self: None, first)
    replacement = _sync_acp_base([], [])
    replacement.start = MethodType(
        lambda _self, startup_timeout=20: "replacement",
        replacement,
    )
    session = _factory_chain(tmp_path, first)
    session.set_tool_filter(_deny_mutation)

    with (
        patch("src.agent_session.wrappers.SyncACPSession", return_value=replacement),
        patch(
            "src.agent_session.wrappers._replace_model_in_agent_args",
            return_value=(["--model", "new"], True),
        ),
    ):
        assert session._do_failover(from_model="old", to_model="new") is True

    _assert_mutation_filter_preserved(replacement)


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

        def set_tool_filter(self, tool_filter):
            self.tool_filter = tool_filter

        def get_tool_filter(self):
            return getattr(self, "tool_filter", None)

    class CandidateSession:
        def start(self, startup_timeout=60):
            return "candidate"

        def close(self):
            return None

        def set_tool_filter(self, tool_filter):
            self.tool_filter = tool_filter

        def get_tool_filter(self):
            return getattr(self, "tool_filter", None)

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


def test_failed_candidate_install_keeps_every_uncertain_reference_for_close_retry() -> None:
    import pytest

    from src.agent_session.wrappers import ModelFailureAwareSession

    class OldSession:
        def close(self):
            return None

        def set_tool_filter(self, tool_filter):
            self.tool_filter = tool_filter

        def get_tool_filter(self):
            return getattr(self, "tool_filter", None)

    class CandidateSession:
        def __init__(self):
            self.allow_close = False
            self.close_calls = 0

        def set_tool_filter(self, _tool_filter):
            raise RuntimeError("filter install failed")

        def close(self):
            self.close_calls += 1
            if not self.allow_close:
                raise RuntimeError("candidate close is uncertain")

    candidates = [CandidateSession(), CandidateSession()]
    wrapper = ModelFailureAwareSession(
        OldSession(),
        compaction_action=lambda _session: candidates.pop(0),
    )
    wrapper.set_tool_filter(lambda _tool: True)

    assert wrapper._do_compaction() is False
    first = wrapper.uncertain_sessions[0]
    assert wrapper._do_compaction() is False
    second = wrapper.uncertain_sessions[1]
    assert wrapper.uncertain_sessions == (first, second)

    second.allow_close = True
    with pytest.raises(RuntimeError):
        wrapper.close()
    assert first.close_calls == 2
    assert second.close_calls == 2
    assert wrapper.uncertain_sessions == (first,)

    first.allow_close = True
    wrapper.close()
    assert wrapper.uncertain_sessions == ()


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
