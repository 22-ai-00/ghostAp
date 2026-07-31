"""Vertical contracts for the protected Deep and Spec execution lanes."""

from __future__ import annotations

import pytest

from src.deep_engine.engine import DeepEngine
from src.engine_base import EngineRunState
from src.spec_engine.engine import SpecEngine, SpecEngineCallbacks
from src.spec_engine.models import SpecPhase, SpecProject
from tests.helpers.session_call_recorder import SessionCallRecorder


def test_deep_preserves_provider_model_and_process_local_pause_resume_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = SessionCallRecorder()
    recorder.fail_first_prompt = True
    factory = recorder.factory_for_backend("codex")
    monkeypatch.setattr("src.deep_engine.engine.create_engine_session", lambda **kwargs: factory(**kwargs))
    engine = DeepEngine("chat-protected", "/tmp/protected-lane", agent_type="codex", model_name="deep-codex-model")
    engine.plan_and_execute("implement the protected contract")
    engine.pause()
    engine.resume()
    engine.pause()

    assert [call.backend for call in recorder.factory_calls] == ["codex", "codex"]
    assert [call.model for call in recorder.factory_calls] == ["deep-codex-model", "deep-codex-model"]
    assert len(recorder.prompt_calls) == 2
    assert recorder.retry_calls == [("codex", 2), ("codex", 2)]
    assert len(recorder.cancelled_sessions) == 2


def test_spec_narrow_phase_retries_same_provider_session_without_durable_recovery(
    monkeypatch: pytest.MonkeyPatch,
):
    """Spec retries in-process; this is not a durable recovery claim."""
    recorder = SessionCallRecorder()
    recorder.fail_first_prompt = True
    engine = SpecEngine(
        "chat-protected",
        "/tmp/protected-lane",
        agent_type="gemini",
        model_name="spec-gemini-model",
        create_session_fn=recorder.factory_for_backend("gemini"),
    )
    engine._project = SpecProject.create("protected-lane", "/tmp/protected-lane")
    engine._run_state = EngineRunState.RUNNING
    engine._session = recorder.factory_for_backend("gemini")(
        cwd="/tmp/protected-lane", model_name="spec-gemini-model"
    )
    # The phase's real retry hook runs; its external factory boundary is held
    # deterministic so the retry's same-session contract is observable.
    monkeypatch.setattr("src.spec_engine.engine._recreate_session_best_effort", lambda **_kwargs: None)

    engine._run_phase(
        1,
        SpecPhase.SPEC,
        "write the protected spec",
        SpecEngineCallbacks(),
        timeout=2,
    )

    assert [call.backend for call in recorder.factory_calls] == ["gemini"]
    assert [call.model for call in recorder.factory_calls] == ["spec-gemini-model"]
    assert recorder.retry_calls == [("gemini", 2)]
    assert [event.kind for event in recorder.events] == [
        "factory", "prompt_attempt", "prompt_timeout", "prompt_attempt", "prompt", "retry_complete"
    ]
    assert len(recorder.sessions) == 1
