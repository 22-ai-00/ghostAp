"""Test that _run_phase() recursive model-switch retry is bounded."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.engine_base import EngineRunState
from src.spec_engine.engine import SpecEngine, SpecEngineCallbacks
from src.spec_engine.models import SpecPhase


def test_run_phase_recursion_limit(monkeypatch):
    """When _try_switch_model always returns True, recursion stops at depth 3."""
    monkeypatch.setattr("src.engine_base.get_settings", lambda: MagicMock(
        spec_max_cycles=1,
        spec_max_retries=0,
    ))

    eng = SpecEngine(chat_id="c", root_path="/tmp")
    eng._project = MagicMock(name="test")
    eng._session = MagicMock()
    eng._renderer = MagicMock()
    eng._run_state = EngineRunState.RUNNING

    # _send_prompt_with_retry always fails
    monkeypatch.setattr(eng, "_send_prompt_with_retry", MagicMock(side_effect=RuntimeError("model error")))
    # _try_switch_model always says "switched OK"
    monkeypatch.setattr(eng, "_try_switch_model", lambda cb: True)

    callbacks = SpecEngineCallbacks()
    callbacks.on_phase_start = None
    callbacks.on_phase_done = None
    callbacks.on_phase_event = None

    with pytest.raises(RuntimeError, match="模型切换递归超限"):
        eng._run_phase(
            cycle_num=1,
            phase=SpecPhase.BUILD,
            prompt="test prompt",
            callbacks=callbacks,
            timeout=10,
        )


def test_run_phase_succeeds_within_depth_limit(monkeypatch):
    """When switch succeeds on first retry (depth < 3), execution continues."""
    monkeypatch.setattr("src.engine_base.get_settings", lambda: MagicMock(
        spec_max_cycles=1,
        spec_max_retries=0,
    ))

    eng = SpecEngine(chat_id="c", root_path="/tmp")
    eng._project = MagicMock(name="test")
    eng._session = MagicMock()
    eng._renderer = MagicMock()
    eng._run_state = EngineRunState.RUNNING

    call_count = 0

    def flaky_send(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            raise RuntimeError("transient")
        # succeed on retry

    monkeypatch.setattr(eng, "_send_prompt_with_retry", flaky_send)
    monkeypatch.setattr(eng, "_try_switch_model", lambda cb: True)

    callbacks = SpecEngineCallbacks()
    callbacks.on_phase_start = None
    callbacks.on_phase_done = None
    callbacks.on_phase_event = None

    # Should not raise — succeeds on second attempt (depth=1)
    eng._run_phase(
        cycle_num=1,
        phase=SpecPhase.BUILD,
        prompt="test",
        callbacks=callbacks,
        timeout=10,
    )
    assert call_count == 2
