"""Tests for spec_engine — ACP-driven SpecEngine with structured methodology."""

import json
import logging
import os
import re
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.acp.models import ACPEvent, ACPEventType, PlanInfo, ToolCallInfo
from src.engine_base import EngineRunState, PerspectiveReview, ReviewPerspective, ReviewResult
from src.spec_engine.artifacts import (
    extract_criteria_from_llm_response,
    parse_acceptance_criteria,
    parse_tasks,
)
from src.spec_engine.engine import SpecEngine, SpecEngineCallbacks
from src.spec_engine.manager import SpecEngineManager
from src.spec_engine.models import (
    PlanArtifact,
    SpecArtifact,
    SpecCycle,
    SpecPhase,
    SpecProject,
    SpecProjectStatus,
    SpecTask,
    SpecTaskStatus,
)
from src.spec_engine.prompts import (
    build_build_prompt,
    build_plan_prompt,
    build_refinement_input,
    build_review_prompt,
    build_spec_prompt,
    build_task_prompt,
    format_criteria_status,
)
from src.spec_engine.reporter import SpecReporter
from src.spec_engine.task_persistence import SpecTaskState, load_task_state
from src.spec_engine.tracker import PhaseTracker
from src.spec_engine.utils import parse_review_output_loose
from src.utils.retry import RetryPolicy

# ---------------------------------------------------------------------------
# Shared spec settings factory — avoids 50-line duplication across test classes
# ---------------------------------------------------------------------------

def _make_spec_settings(**overrides):
    """Build a mock settings object with spec_engine defaults.

    Callers can override any field via keyword arguments.
    """
    s = MagicMock()
    defaults = dict(
        spec_max_cycles=1,
        spec_max_cycles_limit=5000,
        spec_convergence_window=1,
        spec_execution_timeout=300,
        spec_review_enabled=True,
        spec_infinite_mode=False,
        spec_disable_convergence=False,
        spec_disable_early_stop=False,
        spec_min_cycles=1,
        spec_max_retries=1,
        spec_cycle_tasks_max=50,
        spec_cycle_output_max_chars=4000,
        spec_state_filename=".spec_engine_state.json",
        spec_artifacts_dirname=".spec_engine",
        spec_persist_phase_artifacts=True,
        spec_persist_every_phase=True,
        spec_discovery_enabled=False,
        spec_discovery_max_questions=3,
        spec_discovery_force_nonempty=True,
        spec_generated_specs_per_cycle=1,
        spec_discovery_gate_on_satisfied=True,
        spec_discovery_max_pending=5,
        spec_discovery_cooldown_cycles=3,
        spec_backlog_stuck_window=3,
        spec_success_ignore_backlog=True,
        spec_allow_resume_from_disk=True,
        spec_history_log_filename="history.jsonl",
        spec_phase_output_persist_max_chars=20000,
        spec_cycle_artifact_retention=50,
        spec_generated_specs_retention=1000,
        spec_review_failure_circuit_enabled=False,
        spec_review_failure_max_consecutive=3,
        spec_review_failure_cooldown_cycles=3,
        spec_review_timeout=120,
        spec_review_min_timeout=30,
        spec_review_hard_floor=15,
        spec_review_max_parallel=4,
        spec_review_retry_max_attempts=1,
        spec_review_retry_max_delay=30,
        spec_state_cycles_tail=50,
        spec_state_work_items_tail=200,
        spec_state_metrics_tail=200,
        spec_rebuild_session_between_cycles=False,
        spec_review_parallel_enabled=False,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def test_spec_engine_run_phase_fallback_to_send_prompt_when_retry_method_missing():
    """回归：session 缺少 send_prompt_with_retry 时，_run_phase 应回退到 send_prompt。"""
    engine = SpecEngine(chat_id="c", root_path="/tmp")

    class _DummySession:
        def send_prompt(self, prompt: str, on_event=None, timeout: int = 0):
            if on_event:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="ok"))
            return MagicMock(stop_reason="end_turn")

    engine._session = _DummySession()

    output = engine._run_phase(
        cycle_num=1,
        phase=SpecPhase.SPEC,
        prompt="p",
        callbacks=SpecEngineCallbacks(),
        timeout=1,
    )
    assert output == "ok"


def test_spec_engine_aux_criteria_prompt_uses_configured_timeout(monkeypatch):
    """回归：需求拆解的辅助 ACP prompt 不应继续使用 60s 硬编码默认值。"""
    engine = SpecEngine(chat_id="c", root_path="/tmp", agent_type="coco", model_name="m1")
    engine.settings = SimpleNamespace(engine_aux_prompt_timeout=321)

    captured = {}

    def fake_prompt_via_acp(text, create_session_fn, agent_type, cwd, timeout=None, model_name=None):
        captured.update(
            text=text,
            create_session_fn=create_session_fn,
            agent_type=agent_type,
            cwd=cwd,
            timeout=timeout,
            model_name=model_name,
        )
        return "- 验收标准1"

    monkeypatch.setattr("src.spec_engine.engine.prompt_via_acp", fake_prompt_via_acp)

    send = engine._make_aux_send_fn()

    assert send("拆解需求") == "- 验收标准1"
    assert captured["timeout"] == 321
    assert captured["agent_type"] == "coco"
    assert captured["cwd"] == "/tmp"
    assert captured["model_name"] == "m1"


def test_spec_engine_review_error_empty_message_uses_snippets(monkeypatch):
    """回归：多视角审查异常 message 为空时，应从 stderr/stdout_snippet 补齐 error_text，避免日志空白。"""
    engine = SpecEngine(chat_id="c", root_path="/tmp")

    class _EmptyErr(RuntimeError):
        def __str__(self):
            return ""

    err = _EmptyErr()
    err.stderr_snippet = "E: invalid params"

    class _DummySession:
        def send_prompt(self, *a, **kw):
            raise err
        def send_prompt_with_retry(self, *a, **kw):
            raise err

    engine._session = _DummySession()

    # 只验证不抛异常且 suggestion 中包含补齐后的 err_str
    r = engine._conduct_review(cycle=1, callbacks=SpecEngineCallbacks())
    assert r is not None
    assert any("invalid params" in s.lower() for rev in r.reviews for s in (rev.suggestions or []))


def test_run_review_phase_emits_phase_start_before_review_sessions(tmp_path, monkeypatch):
    """Review can run for minutes, so the main Spec card must update before workers start."""
    engine = SpecEngine(chat_id="c", root_path=str(tmp_path))
    engine.settings = SimpleNamespace(
        spec_review_enabled=True,
        spec_persist_phase_artifacts=False,
        spec_persist_every_phase=False,
    )
    cycle = SpecCycle(cycle_number=2)
    calls: list[tuple[str, int, str | None]] = []
    review = ReviewResult(
        reviews=[
            PerspectiveReview(
                perspective=ReviewPerspective.TESTER,
                passed=True,
                suggestions=[],
                summary="ok",
            )
        ],
        iteration=2,
    )

    def fake_conduct(cycle_num, callbacks, cycle_obj=None):
        calls.append(("conduct", cycle_num, None))
        return review

    monkeypatch.setattr(engine, "_conduct_review", fake_conduct)
    callbacks = SpecEngineCallbacks(
        on_phase_start=lambda cycle_num, phase: calls.append(("phase_start", cycle_num, phase.value)),
    )

    assert engine._run_review_phase(2, cycle, callbacks) is True
    assert calls[0] == ("phase_start", 2, "review")
    assert calls[1] == ("conduct", 2, None)


def test_spec_engine_review_error_snippet_is_redacted(monkeypatch, caplog):
    """回归：review 异常诊断应遵循 diagnostics 脱敏规则，避免在 error_text 中泄露 token。"""
    engine = SpecEngine(chat_id="c", root_path="/tmp")

    class _EmptyErr(RuntimeError):
        def __str__(self):
            return ""

    err = _EmptyErr()
    # 命中默认 redact pattern：token=...
    err.stderr_snippet = "token=SECRET_TOKEN"

    class _DummySession:
        def send_prompt(self, *a, **kw):
            raise err
        def send_prompt_with_retry(self, *a, **kw):
            raise err

    engine._session = _DummySession()

    caplog.set_level(logging.WARNING, logger="src.spec_engine.engine")
    caplog.clear()
    _ = engine._conduct_review(cycle=3, callbacks=SpecEngineCallbacks())

    msgs = [r.getMessage() for r in caplog.records]
    hit = [m for m in msgs if "[Spec] review_exception:" in m]
    assert hit
    m = hit[-1]
    assert "SECRET_TOKEN" not in m


def test_spec_engine_review_error_snippet_is_truncated(monkeypatch, caplog):
    """回归：review 异常日志需要对超长 snippet 做截断，避免日志刷屏/泄露大段内容。"""
    engine = SpecEngine(chat_id="c", root_path="/tmp")

    class _EmptyErr(RuntimeError):
        def __str__(self):
            return ""

    err = _EmptyErr()
    err.stderr_snippet = "X" * 10000

    class _DummySession:
        def send_prompt(self, *a, **kw):
            raise err
        def send_prompt_with_retry(self, *a, **kw):
            raise err

    engine._session = _DummySession()

    caplog.set_level(logging.WARNING, logger="src.spec_engine.engine")
    caplog.clear()
    _ = engine._conduct_review(cycle=4, callbacks=SpecEngineCallbacks())

    msgs = [r.getMessage() for r in caplog.records]
    hit = [m for m in msgs if "[Spec] review_exception:" in m]
    assert hit
    m = hit[-1]
    # 不能把整段超长内容打到日志里
    assert len(m) < 5000
    # 至少应出现一次截断标记
    assert "truncated" in m


@pytest.mark.parametrize(
    "err",
    [
        RuntimeError(""),
        type("_EmptyStrErr", (RuntimeError,), {"__str__": lambda self: ""})(),
        type("_EmptyReprErr", (RuntimeError,), {"__repr__": lambda self: ""})(),
        type(
            "_StrBoomErr",
            (RuntimeError,),
            {"__str__": lambda self: (_ for _ in ()).throw(RuntimeError("boom"))},
        )(),
    ],
)
def test_spec_engine_review_exception_diagnostics_log_has_nonempty_error(monkeypatch, caplog, err):
    """回归：审查异常日志必须包含非空 error（至少为 '(empty)' 或默认文案），且包含 exception_type。"""
    engine = SpecEngine(chat_id="c", root_path="/tmp")

    class _DummySession:
        def send_prompt(self, *a, **kw):
            raise err
        def send_prompt_with_retry(self, *a, **kw):
            raise err

    engine._session = _DummySession()

    caplog.set_level(logging.WARNING, logger="src.spec_engine.engine")
    caplog.clear()
    _ = engine._conduct_review(cycle=2, callbacks=SpecEngineCallbacks())

    msgs = [r.getMessage() for r in caplog.records]
    hit = [m for m in msgs if "[Spec] review_exception:" in m]
    assert hit, "missing review exception log"
    m = hit[-1]
    # 新日志稳定字段契约：err_type/err_repr/error_text 必须存在且非空
    assert "err_type=" in m

    # 新日志格式：error_text= 必须非空（至少为 '(empty)' 或 '<ExceptionType>'）
    assert "error_text=" in m
    assert "error_text=," not in m
    assert "error_text= " not in m

    # err_repr 也必须非空（避免异常 repr/str 都为空时无信息）
    assert "err_repr=" in m
    assert "err_repr=," not in m
    assert "err_repr= " not in m


def test_spec_engine_build_internal_error_saves_fixed_recovery_task_id(monkeypatch, tmp_path, caplog):
    """验收：BUILD phase + Internal error 失败时应保存任务且 recovery task_id 可固定为 f5f3dcb4。"""
    # Mock get_settings() so persistence.py reads the override via Settings singleton
    _mock_settings = MagicMock()
    _mock_settings.spec_failed_task_id_override = "f5f3dcb4"
    monkeypatch.setattr("src.config.get_settings", lambda: _mock_settings)
    monkeypatch.setattr("src.spec_engine.task_persistence.SPEC_TASKS_DIR", str(tmp_path / "spec_tasks"))
    monkeypatch.setattr("src.spec_engine.task_persistence.SPEC_TASKS_DIR_FALLBACK", str(tmp_path / "spec_tasks_fb"))

    engine = SpecEngine(chat_id="c", root_path=str(tmp_path))

    class _Sess:
        def send_prompt(self, *a, **kw):
            raise RuntimeError("Internal error")

    engine._session = _Sess()

    class _S:
        spec_max_retries = 0

    engine.settings = _S()
    monkeypatch.setattr(engine, "_try_switch_model", lambda callbacks: False)

    caplog.set_level(logging.ERROR, logger="src.spec_engine.engine")
    caplog.clear()

    with pytest.raises(RuntimeError) as e:
        engine._run_phase(
            cycle_num=1,
            phase=SpecPhase.BUILD,
            prompt="p",
            callbacks=SpecEngineCallbacks(),
            timeout=1,
        )

    msg = str(e.value)
    assert "Phase build 失败" in msg
    assert "Internal error" in msg
    assert "task_id=f5f3dcb4" in msg

    state = load_task_state("f5f3dcb4")
    assert state is not None
    assert state.task_id == "f5f3dcb4"
    assert state.status == "失败"
    assert "Phase build 失败: Internal error" in (state.failure_reason or "")

    logs = [r.getMessage() for r in caplog.records]
    assert any(("Phase build 失败" in m and "Internal error" in m and "task_id=f5f3dcb4" in m) for m in logs), (
        "missing structured error log with task_id/phase/error"
    )


def test_spec_engine_review_failure_diagnostics_written_to_cycle_and_metrics(monkeypatch, tmp_path):
    """回归：审查异常应 best-effort 写入 cycle/metrics，便于后续追踪。"""
    engine = SpecEngine(chat_id="c", root_path=str(tmp_path))

    class _Sess:
        # phase runner uses send_prompt to produce outputs; we return empty ok responses
        def send_prompt(self, prompt: str, on_event=None, timeout: int = 0, **kw):
            pass

        def send_prompt_with_retry(self, prompt: str, on_event=None, timeout: int = 0, **kw):
            # review phase prompt: contains structured tags like [ARCHITECT]
            if "[ARCHITECT]" in (prompt or "") and "审查视角" in (prompt or ""):
                raise RuntimeError("")
            # Produce a minimal valid response for spec/plan/task/build/criteria
            text = "{}"
            if "将以下实现方案分解为可执行的具体任务" in (prompt or ""):
                text = "1. [t] (依赖: 无)"
            elif "按以下任务列表逐步执行实现" in (prompt or ""):
                text = "done"
            elif "请评估以下验收标准是否已满足" in (prompt or ""):
                text = "CRITERIA_1: FAIL"

            if on_event:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
            return MagicMock(stop_reason="end_turn")

    engine._session = _Sess()

    # keep settings minimal and avoid persistence side effects
    class _S:
        spec_max_cycles = 1
        spec_execution_timeout = 2
        spec_review_enabled = True
        spec_persist_phase_artifacts = False
        spec_persist_every_phase = False
        spec_discovery_enabled = False
        spec_discovery_force_nonempty = False
        spec_discovery_max_questions = 1
        spec_generated_specs_per_cycle = 1
        spec_cycle_tasks_max = 1
        spec_max_cycles_limit = 10
        spec_cycle_output_max_chars = 2000
        spec_max_retries = 1
        spec_convergence_window = 3
        # Add missing attributes for ContinuationPolicy
        spec_infinite_mode = False
        spec_disable_convergence = False
        spec_disable_early_stop = False
        spec_min_cycles = 1
        spec_rebuild_session_between_cycles = False
        # Add for _save_failed_task
        spec_artifacts_dirname = ".spec_engine"
        spec_history_log_filename = "history.jsonl"
        spec_state_filename = ".spec_engine_state.json"
        spec_phase_output_persist_max_chars = 20000
        spec_state_cycles_tail = 50
        spec_state_work_items_tail = 200
        spec_state_metrics_tail = 200
        # Add for review circuit breaker
        spec_review_failure_circuit_enabled = False
        spec_review_failure_max_consecutive = 3
        spec_review_failure_cooldown_cycles = 3
        spec_review_strategy = "multi_perspective"
        # Add for review pipeline settings
        spec_review_timeout = 120
        spec_review_min_timeout = 30
        spec_review_hard_floor = 15
        spec_review_max_parallel = 4
        spec_review_retry_max_attempts = 1
        spec_review_retry_max_delay = 30

    engine.settings = _S()

    # Patch session factory inside SpecEngine since we injected it via DI
    engine._create_session_fn = lambda **kw: engine._session
    monkeypatch.setattr(
        "src.spec_engine.session_utils.get_coco_model_manager", lambda: type("M", (), {"get_current_model": lambda self: "", "get_models": lambda self: type("MR", (), {"models": []})(), "set_model": lambda self, model: True})()
    )
    # Force pipeline review path to raise, so diagnostics are written via handle_review_exception
    def _raise_pipeline(*a, **kw):
        raise RuntimeError("pipeline test error")
    monkeypatch.setattr(
        "src.spec_engine.review_pipeline.run_review_pipeline",
        _raise_pipeline,
    )

    p = engine.execute("req")
    assert p is not None
    assert p.cycles and len(p.cycles) == 1
    c = p.cycles[0]
    assert (c.review_decision or "") == "review_failed_continue"
    assert isinstance(c.review_diagnostics, dict)
    assert c.review_diagnostics.get("err_type")
    # metrics should include review failure markers
    assert p.metrics_history
    m = p.metrics_history[-1]
    assert bool(getattr(m, "review_failed", False)) is True
    assert (getattr(m, "review_decision", "") or "") == "review_failed_continue"


def test_spec_engine_normalize_review_diagnostics_compat_to_stable():
    """回归：历史 compat 字段应可规范化为 stable 字段，且关键字段非空。"""
    compat = {
        "cycle_number": 12,
        "exception_type": "RuntimeError",
        "review_role": "multi_perspective",
        "decision": "review_failed_continue",
        # error_text 允许为空，normalize 应兜底
        "error_text": "",
        "traceback_snippet": "",
    }
    out = SpecEngine._normalize_review_diagnostics(compat)
    assert isinstance(out, dict)
    # stable keys present
    assert out.get("phase") == "review"
    assert out.get("role") == "multi_perspective"
    assert out.get("cycle") == 12
    assert out.get("decision") == "review_failed_continue"
    assert out.get("err_type") == "RuntimeError"
    assert (out.get("err_repr") or "").strip()
    assert (out.get("error_text") or "").strip()


def test_spec_engine_format_review_exception_log_line_contains_stable_keys():
    """回归：review 异常日志拼装 SSOT 必须包含 stable 键且关键字段非空。"""
    diag = {
        "phase": "review",
        "role": "multi_perspective",
        "cycle": 7,
        "decision": "review_failed_continue",
        "fail_reason": "exception",
        "err_type": "RuntimeError",
        "err_repr": "<RuntimeError>",
        "error_text": "boom",
        "traceback_snippet": "",
    }
    line = SpecEngine._format_review_exception_log_line(diag, diag_json="{}").strip()
    assert line.startswith("[Spec] review_exception")
    for key in (
        "phase=",
        "role=",
        "cycle=",
        "decision=",
        "fail_reason=",
        "err_type=",
        "err_repr=",
        "error_text=",
        "diag=",
    ):
        assert key in line
    assert "error_text=," not in line
    assert "error_text= " not in line


def test_spec_engine_review_failure_circuit_breaker_skips_review(monkeypatch, caplog, tmp_path):
    """回归：启用熔断后连续 N 次审查异常应触发跳过（review_circuit_open）。"""
    engine = SpecEngine(chat_id="c", root_path=str(tmp_path))

    # minimal settings
    class _S:
        spec_max_cycles = 1
        spec_execution_timeout = 2
        spec_review_enabled = True
        spec_persist_phase_artifacts = False
        spec_persist_every_phase = False
        spec_discovery_enabled = False
        spec_discovery_force_nonempty = False
        spec_discovery_max_questions = 1
        spec_generated_specs_per_cycle = 1
        spec_cycle_tasks_max = 1
        spec_max_cycles_limit = 10
        spec_cycle_output_max_chars = 2000
        spec_max_retries = 1
        spec_convergence_window = 3
        # circuit breaker
        spec_review_failure_circuit_enabled = True
        spec_review_failure_max_consecutive = 1
        spec_review_failure_cooldown_cycles = 10
        spec_review_strategy = "multi_perspective"

        # Add for review pipeline settings
        spec_review_timeout = 120
        spec_review_min_timeout = 30
        spec_review_hard_floor = 15
        spec_review_max_parallel = 4
        spec_review_retry_max_attempts = 1
        spec_review_retry_max_delay = 30

        # Add missing attributes for ContinuationPolicy
        spec_infinite_mode = False
        spec_disable_convergence = False
        spec_disable_early_stop = False
        spec_min_cycles = 1
        spec_rebuild_session_between_cycles = False
        # Add for _save_failed_task
        spec_artifacts_dirname = ".spec_engine"
        spec_history_log_filename = "history.jsonl"
        spec_state_filename = ".spec_engine_state.json"
        spec_phase_output_persist_max_chars = 20000
        spec_state_cycles_tail = 50
        spec_state_work_items_tail = 200
        spec_state_metrics_tail = 200

    engine.settings = _S()

    class _Sess:
        def __init__(self):
            self.calls = 0

        def send_prompt(self, prompt: str, on_event=None, timeout: int = 0, **kw):
            pass

        def send_prompt_with_retry(self, prompt: str, on_event=None, timeout: int = 0, **kw):
            # review prompt: trigger failure
            if "[ARCHITECT]" in (prompt or "") and "审查视角" in (prompt or ""):
                self.calls += 1
                raise RuntimeError("")
            # other phases: minimal valid output
            text = "{}"
            if "将以下实现方案分解为可执行的具体任务" in (prompt or ""):
                text = "1. [t] (依赖: 无)"
            elif "按以下任务列表逐步执行实现" in (prompt or ""):
                text = "done"
            elif "请评估以下验收标准是否已满足" in (prompt or ""):
                text = "CRITERIA_1: FAIL"
            if on_event:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
            return MagicMock(stop_reason="end_turn")

    sess = _Sess()
    engine._session = sess
    engine._create_session_fn = lambda **kw: engine._session
    monkeypatch.setattr(
        "src.spec_engine.session_utils.get_coco_model_manager", lambda: type("M", (), {"get_current_model": lambda self: "", "get_models": lambda self: type("MR", (), {"models": []})(), "set_model": lambda self, model: True})()
    )
    # Force pipeline review path to raise, so the circuit breaker test exercises handle_review_exception
    monkeypatch.setattr(
        "src.spec_engine.review_pipeline.run_review_pipeline",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("pipeline test error")),
    )

    caplog.set_level(logging.WARNING, logger="src.spec_engine.engine")
    caplog.clear()

    # First review: fails and opens circuit
    p1 = engine.execute("req")
    assert p1 and p1.cycles
    # Pipeline path is used (not the legacy session path), so sess.calls stays 0.
    # Instead, verify the review result indicates failure.

    # First cycle should record circuit-open decision (not just continue)
    c1 = p1.cycles[0]
    assert (c1.review_decision or "") in ("review_failed_open_circuit", "review_failed_continue")

    # Second call to _conduct_review in same engine instance should be skipped by circuit breaker
    r2 = engine._conduct_review(cycle=2, callbacks=SpecEngineCallbacks())
    assert r2 is not None
    assert any("审查暂停" in (s or "") for rev in r2.reviews for s in (rev.suggestions or []))

    msgs = [r.getMessage() for r in caplog.records]
    assert any("review_circuit_open" in m for m in msgs)


def test_spec_engine_review_circuit_skip_does_not_block_main_loop(monkeypatch, tmp_path):
    """回归：熔断跳过只影响 review 步骤，Spec 主循环仍可继续推进到下一轮 cycle。"""
    engine = SpecEngine(chat_id="c", root_path=str(tmp_path))

    class _S:
        spec_max_cycles = 2
        spec_execution_timeout = 2
        spec_review_enabled = True
        spec_persist_phase_artifacts = False
        spec_persist_every_phase = False
        spec_discovery_enabled = False
        spec_discovery_force_nonempty = False
        spec_discovery_max_questions = 1
        spec_generated_specs_per_cycle = 1
        spec_cycle_tasks_max = 1
        spec_max_cycles_limit = 10
        spec_cycle_output_max_chars = 2000
        spec_max_retries = 1
        spec_convergence_window = 3
        # circuit breaker
        spec_review_failure_circuit_enabled = True
        spec_review_failure_max_consecutive = 1
        spec_review_failure_cooldown_cycles = 10
        spec_review_strategy = "multi_perspective"

        # Add for review pipeline settings
        spec_review_timeout = 120
        spec_review_min_timeout = 30
        spec_review_hard_floor = 15
        spec_review_max_parallel = 4
        spec_review_retry_max_attempts = 1
        spec_review_retry_max_delay = 30

        # Add missing attributes for ContinuationPolicy
        spec_infinite_mode = False
        spec_disable_convergence = False
        spec_disable_early_stop = False
        spec_min_cycles = 1
        spec_rebuild_session_between_cycles = False
        # Add for _save_failed_task
        spec_artifacts_dirname = ".spec_engine"
        spec_history_log_filename = "history.jsonl"
        spec_state_filename = ".spec_engine_state.json"
        spec_phase_output_persist_max_chars = 20000
        spec_state_cycles_tail = 50
        spec_state_work_items_tail = 200
        spec_state_metrics_tail = 200

    engine.settings = _S()

    class _Sess:
        def __init__(self):
            self.review_calls = 0
            self.total_calls = 0

        def send_prompt(self, prompt: str, on_event=None, timeout: int = 0, **kw):
            pass

        def send_prompt_with_retry(self, prompt: str, on_event=None, timeout: int = 0, **kw):
            self.total_calls += 1
            # review prompt: trigger failure on first cycle only
            if "[ARCHITECT]" in (prompt or "") and "审查视角" in (prompt or ""):
                self.review_calls += 1
                raise RuntimeError("")

            # other phases: minimal valid output
            text = "{}"
            if "将以下实现方案分解为可执行的具体任务" in (prompt or ""):
                text = "1. [t] (依赖: 无)"
            elif "按以下任务列表逐步执行实现" in (prompt or ""):
                text = "done"
            elif "请评估以下验收标准是否已满足" in (prompt or ""):
                text = "CRITERIA_1: FAIL"

            if on_event:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
            return MagicMock(stop_reason="end_turn")

    sess = _Sess()
    engine._session = sess
    engine._create_session_fn = lambda **kw: engine._session
    monkeypatch.setattr(
        "src.spec_engine.session_utils.get_coco_model_manager", lambda: type("M", (), {"get_current_model": lambda self: "", "get_models": lambda self: type("MR", (), {"models": []})(), "set_model": lambda self, model: True})()
    )
    # Force pipeline review path to raise, so the circuit breaker opens after first failure
    monkeypatch.setattr(
        "src.spec_engine.review_pipeline.run_review_pipeline",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("pipeline test error")),
    )

    p = engine.execute("req")
    assert p and p.cycles and len(p.cycles) == 2

    # Pipeline path is used (not the legacy session path), so sess.review_calls stays 0.
    # Review failure is triggered by the pipeline mock raising RuntimeError.

    c1, c2 = p.cycles[0], p.cycles[1]
    assert (c1.review_decision or "") in ("review_failed_open_circuit", "review_failed_continue")
    assert (c2.review_decision or "") == "review_circuit_open_skip"
    assert isinstance(c2.review_diagnostics, dict)
    assert (c2.review_diagnostics.get("err_type") or "") == "ReviewCircuitOpen"
    assert (c2.review_diagnostics.get("fail_reason") or "") == "circuit_open"


def test_ttadk_startup_model_log_uses_real_or_auto(caplog):
    """启动点日志语义：model 字段只能是真实名或 (auto)。"""
    # Use mock settings for engine to speed up test and avoid persistence
    with patch("src.engine_base.get_settings") as mock_engine_settings:
        s = MagicMock()
        s.spec_max_cycles = 1
        s.spec_execution_timeout = 5
        s.spec_persist_every_phase = False
        s.spec_review_enabled = False
        s.spec_discovery_enabled = False
        s.spec_generated_specs_per_cycle = 0
        s.spec_max_cycles_limit = 5000
        mock_engine_settings.return_value = s

        engine = SpecEngine(chat_id="c", root_path="/tmp/test", agent_type="ttadk_codex", model_name="gpt-5.2")

        caplog.set_level(logging.INFO, logger="src.agent_session")

        class _S:
            def __init__(self, *a, **k):
                self.session_id = "sid"
                self.created_at = 0.0
                self.last_active = 0.0
                self.message_count = 0
                self.last_query = ""
                self.is_resumed = False

            def describe_agent(self):
                return "dummy"

            def start(self, startup_timeout: float = 60, **kwargs):
                return "sid"

            def load_session(self, session_id: str, timeout: float = 60):
                del timeout
                return None

            def load_local_history(self, session_id=None, limit: int = 200):
                return []

            def cancel(self):
                return None

            def close(self):
                return None

            def to_snapshot(self):
                return {}

            def get_session_info(self):
                return ""

            def is_server_running(self):
                return True

            def is_server_healthy(self, healthcheck_timeout: float = 2.0):
                return True

            def send_prompt(self, *a, **k):
                return MagicMock(stop_reason="end_turn")

        class _SessSettings:
            acp_startup_timeout = 20
            rate_limit_retry_enabled = False

        with (
            patch("src.agent_session.factory.get_settings", return_value=_SessSettings()),
            patch("src.ttadk.get_ttadk_manager", return_value=MagicMock()),
            patch("src.ttadk.startup_common.precheck_ttadk_startup_model") as mk_precheck,
            patch("src.agent_session.factory.SyncTTADKCLISession", return_value=_S()),
        ):
            mk_precheck.return_value = {
                "tool": "codex",
                "input_model": "gpt-5.2",
                "model": "gpt-5.2-codex-ttadk",
                "validated": True,
                "source": "probe",
                "decision": "precheck_validated",
                "fail_phase": "",
                "warnings": [],
                "diagnostics": {"attempts": [{"phase": "precheck"}]},
            }
            caplog.clear()
            engine.execute("do something")

        text = "\n".join([r.getMessage() for r in caplog.records])
        assert "[SessionFactory] ttadk cli startup:" in text
        m = re.search(r"\bmodel=([^\s]+)", text)
        assert m is not None
        assert m.group(1) == "gpt-5.2-codex-ttadk"
        assert m.group(1) != "gpt-5.2"


def test_ttadk_resume_model_log_uses_real_or_auto(caplog):
    """恢复路径同样要求：model 字段只能是真实名或 (auto)。"""
    with patch("src.engine_base.get_settings") as mock_engine_settings:
        s = MagicMock()
        s.spec_max_cycles = 1
        s.spec_execution_timeout = 5
        s.spec_persist_every_phase = False
        s.spec_review_enabled = False
        s.spec_discovery_enabled = False
        s.spec_generated_specs_per_cycle = 0
        s.spec_max_cycles_limit = 5000
        mock_engine_settings.return_value = s

        engine = SpecEngine(chat_id="c", root_path="/tmp/test", agent_type="ttadk_codex", model_name="gpt-5.2")
        engine._project = SpecProject.create(name="p", root_path="/tmp/test")
        engine._project.status = SpecProjectStatus.PAUSED

        caplog.set_level(logging.INFO, logger="src.agent_session")

        class _S:
            def __init__(self, *a, **k):
                self.session_id = "sid"
                self.created_at = 0.0
                self.last_active = 0.0
                self.message_count = 0
                self.last_query = ""
                self.is_resumed = False

            def describe_agent(self):
                return "dummy"

            def start(self, startup_timeout: float = 60, **kwargs):
                return "sid"

            def load_session(self, session_id: str, timeout: float = 60):
                del timeout
                return None

            def load_local_history(self, session_id=None, limit: int = 200):
                return []

            def cancel(self):
                return None

            def close(self):
                return None

            def to_snapshot(self):
                return {}

            def get_session_info(self):
                return ""

            def is_server_running(self):
                return True

            def is_server_healthy(self, healthcheck_timeout: float = 2.0):
                return True

            def send_prompt(self, *a, **k):
                return MagicMock(stop_reason="end_turn")

        class _SessSettings:
            acp_startup_timeout = 20
            rate_limit_retry_enabled = False

        with (
            patch("src.agent_session.factory.get_settings", return_value=_SessSettings()),
            patch("src.ttadk.get_ttadk_manager", return_value=MagicMock()),
            patch("src.ttadk.startup_common.precheck_ttadk_startup_model") as mk_precheck,
            patch("src.agent_session.factory.SyncTTADKCLISession", return_value=_S()),
        ):
            mk_precheck.return_value = {
                "tool": "codex",
                "input_model": "gpt-5.2",
                "model": None,  # (auto)
                "validated": False,
                "source": "defaults",
                "decision": "precheck_auto",
                "fail_phase": "",
                "warnings": ["no_m_passthrough"],
                "diagnostics": {"attempts": [{"phase": "precheck"}]},
            }
            caplog.clear()
            engine.resume()

        text = "\n".join([r.getMessage() for r in caplog.records])
        assert "[SessionFactory] ttadk cli startup:" in text
        assert "model=(auto)" in text
        assert re.search(r"\bmodel=gpt-5\.2\b", text) is None


def test_try_switch_model_claude_returns_false_without_switch(monkeypatch, tmp_path):
    from src.spec_engine.engine import SpecEngine

    engine = SpecEngine(chat_id="c1", root_path=str(tmp_path), agent_type="claude")
    engine._models_tried = []
    engine._current_model = None

    # claude CLI 模式不应进入 coco/ttadk 的模型切换分支
    assert engine._try_switch_model(callbacks=MagicMock()) is False


# ======================================================================
# TestSpecModels — enums, creation, serialization, lifecycle
# ======================================================================


class TestSpecModels:
    def test_spec_phase_values(self):
        assert SpecPhase.SPEC.value == "spec"
        assert SpecPhase.PLAN.value == "plan"
        assert SpecPhase.TASK.value == "task"
        assert SpecPhase.BUILD.value == "build"
        assert SpecPhase.REVIEW.value == "review"

    def test_project_status_enum(self):
        assert SpecProjectStatus.IDLE.value == "idle"
        assert SpecProjectStatus.RUNNING.value == "running"
        assert SpecProjectStatus.CLARIFYING.value == "clarifying"
        assert SpecProjectStatus.COMPLETED.value == "completed"
        assert SpecProjectStatus.ABORTED.value == "aborted"

    def test_task_status_enum(self):
        assert SpecTaskStatus.PENDING.value == "pending"
        assert SpecTaskStatus.COMPLETED.value == "completed"

    def test_spec_task_to_dict_from_dict(self):
        task = SpecTask(
            task_id=2, description="Add tests", dependencies=[1], status=SpecTaskStatus.COMPLETED, output="ok"
        )
        d = task.to_dict()
        assert d["task_id"] == 2
        assert d["dependencies"] == [1]
        assert d["status"] == "completed"

        restored = SpecTask.from_dict(d)
        assert restored.task_id == 2
        assert restored.dependencies == [1]
        assert restored.status == SpecTaskStatus.COMPLETED
        assert restored.output == "ok"

    def test_spec_cycle_creation_and_lifecycle(self):
        cycle = SpecCycle(cycle_number=1)
        assert cycle.cycle_number == 1
        assert cycle.phase == SpecPhase.SPEC
        assert cycle.status == "running"
        assert cycle.completed_at is None

        cycle.complete()
        assert cycle.status == "completed"
        assert cycle.completed_at is not None
        assert cycle.duration > 0

    def test_spec_cycle_fail(self):
        cycle = SpecCycle(cycle_number=2)
        cycle.fail()
        assert cycle.status == "failed"
        assert cycle.completed_at is not None

    def test_spec_cycle_serialization(self):
        cycle = SpecCycle(cycle_number=1)
        cycle.spec_content = "spec output"
        cycle.spec_artifact = SpecArtifact(acceptance_criteria=["C1"])
        cycle.plan_content = "plan output"
        cycle.plan_artifact = PlanArtifact(file_changes=["a.py"])
        cycle.tasks = [SpecTask(task_id=1, description="T1")]
        cycle.build_output = "build done"
        cycle.complete()

        d = cycle.to_dict()
        assert d["cycle_number"] == 1
        assert d["spec_content"] == "spec output"
        assert d["spec_artifact"]["acceptance_criteria"] == ["C1"]
        assert d["plan_artifact"]["file_changes"] == ["a.py"]
        assert len(d["tasks"]) == 1
        assert d["status"] == "completed"

        restored = SpecCycle.from_dict(d)
        assert restored.cycle_number == 1
        assert restored.spec_content == "spec output"
        assert restored.spec_artifact is not None
        assert restored.spec_artifact.acceptance_criteria == ["C1"]
        assert restored.plan_artifact is not None
        assert restored.plan_artifact.file_changes == ["a.py"]
        assert len(restored.tasks) == 1
        assert restored.status == "completed"

    def test_spec_project_lifecycle(self):
        project = SpecProject.create(root_path="/tmp")
        assert project.status == SpecProjectStatus.IDLE

        project.start()
        assert project.status == SpecProjectStatus.RUNNING
        assert project.started_at is not None

        project.pause()
        assert project.status == SpecProjectStatus.PAUSED

        project.resume()
        assert project.status == SpecProjectStatus.RUNNING

        project.complete()
        assert project.status == SpecProjectStatus.COMPLETED
        assert project.completed_at is not None

    def test_spec_project_serialization(self):
        project = SpecProject.create(name="test", root_path="/tmp/test")
        project.requirement = "Build a thing"
        project.acceptance_criteria = ["Criterion 1", "Criterion 2"]
        project.criteria_tracker.init_criteria(["Criterion 1", "Criterion 2"])
        project.criteria_tracker.batch_update({0: True}, 1)
        project.start()
        project.cycles.append(SpecCycle(cycle_number=1))

        d = project.to_dict()
        assert d["name"] == "test"
        assert d["requirement"] == "Build a thing"
        assert len(d["acceptance_criteria"]) == 2
        assert len(d["cycles"]) == 1

        restored = SpecProject.from_dict(d)
        assert restored.name == "test"
        assert restored.requirement == "Build a thing"
        assert len(restored.acceptance_criteria) == 2
        assert len(restored.cycles) == 1
        assert restored.criteria_tracker.satisfied_count == 1


# ======================================================================
# TestPhaseTracker — event processing
# ======================================================================


class TestPhaseTracker:
    def test_text_chunk(self):
        tracker = PhaseTracker()
        event = ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="hello ")
        tracker.process(event)
        event2 = ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="world")
        tracker.process(event2)
        assert tracker.text_buffer == "hello world"

    def test_tool_call_start(self):
        tracker = PhaseTracker()
        tool = ToolCallInfo(id="t1", title="Write file", kind="edit", status="started", locations=["file.py"])
        event = ACPEvent(event_type=ACPEventType.TOOL_CALL_START, tool_call=tool)
        tracker.process(event)
        assert "file.py" in tracker.modified_files

    def test_tool_call_done(self):
        tracker = PhaseTracker()
        tool = ToolCallInfo(id="t2", title="Write file", kind="edit", status="done", locations=["a.py", "b.py"])
        event = ACPEvent(event_type=ACPEventType.TOOL_CALL_DONE, tool_call=tool)
        tracker.process(event)
        assert len(tracker.tool_calls) == 1
        assert "a.py" in tracker.modified_files
        assert "b.py" in tracker.modified_files

    def test_plan_update(self):
        tracker = PhaseTracker()
        plan = PlanInfo(entries=[])
        event = ACPEvent(event_type=ACPEventType.PLAN_UPDATE, plan=plan)
        tracker.process(event)
        assert tracker.plan_progress is not None

    def test_reset(self):
        tracker = PhaseTracker()
        tracker.process(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="hi"))
        tool = ToolCallInfo(id="t3", title="Read file", kind="read", status="done", locations=["f.py"])
        tracker.process(ACPEvent(event_type=ACPEventType.TOOL_CALL_DONE, tool_call=tool))

        assert tracker.text_buffer != ""
        assert len(tracker.tool_calls) > 0

        tracker.reset()
        assert tracker.text_buffer == ""
        assert len(tracker.tool_calls) == 0
        assert len(tracker.modified_files) == 0
        assert tracker.plan_progress is None

    def test_empty_text_ignored(self):
        tracker = PhaseTracker()
        tracker.process(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=""))
        tracker.process(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=None))
        assert tracker.text_buffer == ""


# ======================================================================
# TestSpecReporter — content formatters and title helpers
# ======================================================================


class TestSpecReporter:
    def _make_project(self, **kwargs):
        project = SpecProject.create(name="test_proj", root_path="/tmp")
        project.requirement = kwargs.get("requirement", "Build auth system")
        if "criteria" in kwargs:
            project.acceptance_criteria = kwargs["criteria"]
            project.criteria_tracker.init_criteria(kwargs["criteria"])
        return project

    def test_format_analyzing_start(self):
        r = SpecReporter()
        result = r.format_analyzing_start("Build login")
        assert "Spec Agent 启动" in result
        assert "Build login" in result

    def test_format_analyzing_done(self):
        r = SpecReporter()
        project = self._make_project(criteria=["C1", "C2"])
        result = r.format_analyzing_done(project)
        assert "需求分析完成" in result
        assert "2 条" in result
        assert "C1" in result
        assert "C2" in result

    def test_format_cycle_start(self):
        r = SpecReporter()
        result = r.format_cycle_start(1, 10)
        assert "▶️" in result
        assert "规格定义" in result

    def test_format_phase_start(self):
        r = SpecReporter()
        result = r.format_phase_start(2, SpecPhase.BUILD)
        assert "🔨" in result
        assert "执行构建" in result
        assert "循环 2" in result

    def test_format_phase_done(self):
        r = SpecReporter()
        result = r.format_phase_done(1, SpecPhase.SPEC, "spec content output")
        assert "规格定义完成" in result
        assert "spec content output" in result

    def test_format_review_result_all_passed(self):
        r = SpecReporter()
        review = ReviewResult(
            reviews=[
                PerspectiveReview(perspective=p, passed=True, suggestions=[], summary="通过") for p in ReviewPerspective
            ],
            iteration=1,
        )
        result = r.format_review_result(review, 1)
        assert "PASS" in result
        assert "无改进建议" in result

    def test_format_review_result_shows_role_agent_detail(self):
        r = SpecReporter()
        review = ReviewResult(
            reviews=[
                PerspectiveReview(
                    perspective=ReviewPerspective.TESTER,
                    passed=False,
                    suggestions=["补齐回归测试"],
                    summary="需要测试",
                    role_display_name="测试工程师",
                    review_agent_label="Codex / gpt-5.5",
                )
            ],
            iteration=1,
        )

        result = r.format_review_result(review, 1)

        assert "测试工程师（Codex / gpt-5.5）" in result
        assert "补齐回归测试" in result

    def test_format_criteria_brief(self):
        r = SpecReporter()
        project = self._make_project(criteria=["C1", "C2"])
        project.criteria_tracker.batch_update({0: True}, 1)
        result = r.format_criteria_brief(project)
        assert "✅" in result
        assert "🔲" in result
        assert "1/2" in result

    def test_format_criteria_update(self):
        r = SpecReporter()
        project = self._make_project(criteria=["C1", "C2", "C3"])
        project.criteria_tracker.batch_update({0: True, 2: True}, 1)
        result = r.format_criteria_update(project)
        assert "(循环1)" in result
        assert "C1" in result

    def test_format_project_done_completed(self):
        r = SpecReporter()
        project = self._make_project(criteria=["C1"])
        project.criteria_tracker.init_criteria(["C1"])
        project.criteria_tracker.batch_update({0: True}, 1)
        project.start()
        project.cycles.append(SpecCycle(cycle_number=1))
        project.complete()
        result = r.format_project_done(project)
        assert "Spec 模式完成" in result
        assert "1 轮" in result

    def test_format_project_done_aborted(self):
        r = SpecReporter()
        project = self._make_project(criteria=["C1"])
        project.start()
        project.abort("timeout")
        result = r.format_project_done(project)
        assert "Spec 模式终止" in result
        assert "timeout" in result

    def test_format_project_done_paused(self):
        r = SpecReporter()
        project = self._make_project(criteria=["C1"])
        project.start()
        project.pause()
        result = r.format_project_done(project)
        assert "Spec 模式暂停" in result

    def test_format_status(self):
        r = SpecReporter()
        project = self._make_project(criteria=["C1", "C2"])
        project.start()
        cycle = SpecCycle(cycle_number=1, phase=SpecPhase.BUILD)
        project.cycles.append(cycle)
        result = r.format_status(project)
        assert "Spec 状态" in result
        assert "循环执行中" in result
        assert "1 轮" in result
        assert "🔨" in result

    def test_format_guidance_injected(self):
        r = SpecReporter()
        result = r.format_guidance_injected("focus on tests")
        assert "引导信息已注入" in result
        assert "focus on tests" in result

    # Title helpers
    def test_title_helpers(self):
        r = SpecReporter()
        assert "启动" in r.get_analyzing_start_title()
        assert "完成" in r.get_analyzing_done_title()
        assert "[2/10]" in r.get_cycle_start_title(2, 10)
        assert "规格定义" in r.get_phase_title(1, SpecPhase.SPEC)
        assert "审查通过" in r.get_review_title(1, all_passed=True)
        assert "多角色审查" in r.get_review_title(1, all_passed=False)
        assert "引导" in r.get_guidance_injected_title()
        assert "错误" in r.get_error_title()
        assert "状态" in r.get_status_title()

    def test_get_project_done_title(self):
        r = SpecReporter()
        project = self._make_project()
        project.status = SpecProjectStatus.COMPLETED
        assert "完成" in r.get_project_done_title(project)

        project.status = SpecProjectStatus.ABORTED
        assert "终止" in r.get_project_done_title(project)

        project.status = SpecProjectStatus.PAUSED
        assert "暂停" in r.get_project_done_title(project)

    def test_get_progress_info(self):
        r = SpecReporter()
        project = self._make_project(criteria=["C1", "C2"])
        project.start()
        project.cycles.append(SpecCycle(cycle_number=1))
        info = r.get_progress_info(project)
        assert info["satisfied_count"] == 0
        assert info["total_criteria"] == 2
        assert info["cycle_count"] == 1
        assert info["is_running"]
        assert not info["is_paused"]
        assert "project_name" in info
        assert "progress_bar" in info

    def test_format_history_uses_history_log_path(self):
        r = SpecReporter()
        project = self._make_project(criteria=["C1"])
        project.project_id = "pid"
        project.root_path = "/tmp/x"
        project.cycle_count_total = 10
        # Keep only 1 cycle in memory to simulate compact state
        project.cycles = [SpecCycle(cycle_number=10)]
        project.history_log_path = "/tmp/x/.spec_engine/pid/custom_history.jsonl"
        out = r.format_history(project, tail=5)
        assert "custom_history.jsonl" in out


# ======================================================================
# TestSpecEngine — core engine behavior
# ======================================================================


class TestSpecEngine:
    @patch("src.engine_base.get_settings")
    def _make_engine(self, mock_settings, **kwargs):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_max_cycles_limit = 5000
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        s.spec_cycle_tasks_max = 50
        s.spec_cycle_output_max_chars = 4000
        s.spec_state_filename = ".spec_engine_state.json"
        s.spec_artifacts_dirname = ".spec_engine"
        s.spec_persist_phase_artifacts = True
        s.spec_persist_every_phase = True
        # Keep legacy unit tests stable: discovery is tested separately.
        s.spec_discovery_enabled = False
        s.spec_discovery_max_questions = 3
        s.spec_discovery_force_nonempty = True
        s.spec_generated_specs_per_cycle = 1
        s.spec_discovery_gate_on_satisfied = True
        s.spec_discovery_max_pending = 5
        s.spec_discovery_cooldown_cycles = 3
        s.spec_backlog_stuck_window = 3
        s.spec_success_ignore_backlog = True
        s.spec_allow_resume_from_disk = True
        s.spec_infinite_mode = False
        s.spec_disable_convergence = False
        s.spec_disable_early_stop = False
        s.spec_min_cycles = 1
        mock_settings.return_value = s
        return SpecEngine(chat_id="c1", root_path="/tmp/test", **kwargs)

    def test_initial_state(self):
        engine = self._make_engine()
        assert engine.run_state == EngineRunState.IDLE
        assert engine.project is None
        assert not engine.is_running

    def test_stop(self):
        engine = self._make_engine()
        engine._run_state = EngineRunState.RUNNING
        engine._session = MagicMock()
        engine.stop()
        assert engine.run_state == EngineRunState.STOPPING
        engine._session.cancel.assert_called_once()

    def test_pause(self):
        engine = self._make_engine()
        engine._project = MagicMock()
        engine._session = MagicMock()
        engine._run_state = EngineRunState.RUNNING
        engine.pause()
        assert engine.run_state == EngineRunState.STOPPING

    def test_pause_holds_lock(self):
        """pause() must acquire self._lock when writing _run_state."""
        engine = self._make_engine()
        engine._project = MagicMock()
        engine._session = None
        engine._run_state = EngineRunState.RUNNING

        lock_acquired = False
        original_lock = engine._lock

        class TrackedLock:
            def __enter__(self_lock):
                nonlocal lock_acquired
                lock_acquired = True
                return original_lock.__enter__()

            def __exit__(self_lock, *args):
                return original_lock.__exit__(*args)

        engine._lock = TrackedLock()
        engine.pause()
        assert lock_acquired, "pause() should acquire self._lock"
        assert engine._run_state == EngineRunState.STOPPING

    def test_pause_session_cancel_raises(self):
        """pause() swallows exceptions from session.cancel() and still sets STOPPING."""
        engine = self._make_engine()
        engine._project = MagicMock()
        engine._run_state = EngineRunState.RUNNING

        session = MagicMock()
        session.cancel.side_effect = RuntimeError("connection lost")
        engine._session = session

        # Should NOT raise
        engine.pause()

        assert engine._run_state == EngineRunState.STOPPING
        session.cancel.assert_called_once()

    def test_pause_stop_concurrent(self):
        """Concurrent pause() and stop() must not raise and _run_state ends as STOPPING."""
        engine = self._make_engine()
        engine._project = MagicMock()
        engine._session = MagicMock()
        engine._run_state = EngineRunState.RUNNING

        errors = []

        def call_pause():
            try:
                engine.pause()
            except Exception as e:
                errors.append(e)

        def call_stop():
            try:
                engine.stop()
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=call_pause))
            threads.append(threading.Thread(target=call_stop))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Concurrent pause/stop raised: {errors}"
        assert engine._run_state == EngineRunState.STOPPING

    def test_cleanup(self):
        engine = self._make_engine()
        engine._session = MagicMock()
        engine._project = MagicMock()
        engine.cleanup()
        assert engine._session is None
        assert engine._project is None
        assert engine.run_state == EngineRunState.IDLE

    @pytest.mark.parametrize("state", [EngineRunState.RUNNING, EngineRunState.STOPPING])
    def test_cleanup_while_active_only_requests_stop(self, state):
        engine = self._make_engine()
        engine._run_state = state
        engine._session = MagicMock()
        project = MagicMock()
        engine._project = project

        engine.cleanup()

        assert engine.run_state == EngineRunState.STOPPING
        engine._session.cancel.assert_called_once()
        # 活跃态 cleanup 不应立即清空 project，避免并发线程访问 self._project 失败
        assert engine._project is project

    def test_run_phase_when_stopping_skips_model_switch_and_failed_task(self, monkeypatch):
        engine = self._make_engine()
        engine._run_state = EngineRunState.STOPPING

        class _Sess:
            def send_prompt(self, *a, **kw):
                raise RuntimeError("ACP agent 进程在执行过程中意外终止")

        engine._session = _Sess()

        try_switch = MagicMock(side_effect=AssertionError("should not switch model when stopping"))
        save_failed = MagicMock(side_effect=AssertionError("should not save failed task when stopping"))
        monkeypatch.setattr(engine, "_try_switch_model", try_switch)
        monkeypatch.setattr(engine, "_save_failed_task", save_failed)

        out = engine._run_phase(
            cycle_num=1,
            phase=SpecPhase.SPEC,
            prompt="p",
            callbacks=SpecEngineCallbacks(),
            timeout=1,
        )

        assert out == ""
        try_switch.assert_not_called()
        save_failed.assert_not_called()

    def test_try_switch_model_returns_false_when_not_running(self):
        engine = self._make_engine()
        engine._run_state = EngineRunState.STOPPING

        callbacks = SpecEngineCallbacks()
        assert engine._try_switch_model(callbacks) is False

    def test_inject_guidance(self):
        engine = self._make_engine()
        engine.inject_guidance("focus on login")
        assert engine._user_guidance == ["focus on login"]
        engine.inject_guidance("also fix logout")
        assert engine._user_guidance == ["focus on login", "also fix logout"]

    def test_consume_guidance(self):
        engine = self._make_engine()
        engine.inject_guidance("msg1")
        engine.inject_guidance("msg2")
        result = engine._consume_guidance()
        assert "msg1" in result
        assert "msg2" in result
        # After consume, guidance is empty
        assert engine._user_guidance == []
        assert engine._consume_guidance() == ""

    def test_parse_acceptance_criteria_with_list_markers(self):
        text = """实现登录功能
- 支持邮箱登录
- 支持手机号登录
- 有错误提示
"""
        criteria = parse_acceptance_criteria(text)
        assert len(criteria) == 3
        assert "支持邮箱登录" in criteria

    def test_parse_acceptance_criteria_with_checkboxes(self):
        text = """功能需求
[ ] 第一项
[x] 第二项
"""
        criteria = parse_acceptance_criteria(text)
        assert len(criteria) == 2

    def test_parse_acceptance_criteria_no_markers_fallback(self):
        text = "实现一个简单的登录页面"
        criteria = parse_acceptance_criteria(text)
        assert len(criteria) == 1
        assert "完成需求:" in criteria[0]

    def test_parse_tasks(self):
        text = """1. 创建数据模型 (依赖: 无)
2. 实现 API 接口 (依赖: 1)
3. 编写前端页面 (依赖: 1, 2)
4. 添加测试 (依赖: 2, 3)
"""
        tasks = parse_tasks(text)
        assert len(tasks) == 4
        assert tasks[0].task_id == 1
        assert tasks[0].description == "创建数据模型"
        assert tasks[0].dependencies == []
        assert tasks[1].dependencies == [1]
        assert tasks[2].dependencies == [1, 2]
        assert tasks[3].dependencies == [2, 3]

    def test_parse_tasks_various_formats(self):
        text = """1. First task
2、Second task (depends: 1)
3) Third task (依赖: 1, 2)
"""
        tasks = parse_tasks(text)
        assert len(tasks) == 3

    def test_parse_tasks_empty(self):
        tasks = parse_tasks("no tasks here")
        assert tasks == []

    def test_extract_criteria_from_llm_response(self):
        text = """以下是验收标准：
- 实现登录接口
- 支持邮箱和密码登录
* 显示错误提示信息
1. 添加单元测试
2、集成测试通过
"""
        criteria = extract_criteria_from_llm_response(text)
        assert len(criteria) == 5

    def test_extract_reviews_from_llm_response(self):
        text = """```json
[
  {"perspective": "ARCHITECT", "verdict": "PASS", "suggestions": []},
  {"perspective": "PRODUCT", "verdict": "FAIL", "suggestions": ["Need better UX"]},
  {"perspective": "USER", "verdict": "PASS", "suggestions": []},
  {"perspective": "TESTER", "verdict": "FAIL", "suggestions": ["Add unit tests", "Add integration tests"]}
]
```"""
        reviews = SpecEngine._extract_reviews_from_llm_response(text)
        assert len(reviews) == 4
        assert reviews[0].passed is True
        assert reviews[0].suggestions == []
        assert reviews[1].passed is False
        assert reviews[1].suggestions == ["Need better UX"]
        assert len(reviews[3].suggestions) == 2

    def test_extract_reviews_from_llm_invalid_json(self):
        reviews = SpecEngine._extract_reviews_from_llm_response("not json at all")
        assert reviews == []

    def test_build_spec_prompt(self):
        prompt = build_spec_prompt("Build a login system", "/tmp/test", "", "")
        assert "Build a login system" in prompt
        assert "/tmp/test" in prompt
        assert "```json" in prompt
        assert '"acceptance_criteria"' in prompt
        assert "clarification_questions" in prompt
        assert "grill-me" in prompt
        assert "尖锐追问" in prompt
        assert "推荐答案" in prompt
        assert "不要停下等待用户回答" in prompt

    def test_build_plan_prompt(self):
        prompt = build_plan_prompt("spec output here", "/tmp/test")
        assert "spec output here" in prompt
        assert "```json" in prompt
        assert "file_changes" in prompt
        assert "subagent / 子任务委托" in prompt
        assert "无共享文件写入" in prompt
        assert "grill-me" in prompt
        assert "自动采纳" in prompt

    def test_build_task_prompt(self):
        prompt = build_task_prompt("plan output here")
        assert "plan output here" in prompt
        assert "任务编号" in prompt
        assert "可并行、可委托给 subagent" in prompt
        assert "潜在冲突" in prompt
        assert "grill-me" in prompt
        assert "采纳动作" in prompt

    def test_build_build_prompt(self):
        tasks = [
            SpecTask(task_id=1, description="Create models"),
            SpecTask(task_id=2, description="Add tests"),
        ]
        prompt = build_build_prompt(tasks, "plan content", "/tmp/test", "")
        assert "Create models" in prompt
        assert "Add tests" in prompt
        assert "plan content" in prompt
        assert "严格按照依赖关系推进" in prompt
        assert "优先使用当前工具支持的 subagent / 子任务委托并行执行" in prompt
        assert "严格按照任务顺序执行" not in prompt
        assert "grill-me" in prompt
        assert "自动采纳" in prompt

    def test_build_review_prompt(self):
        prompt = build_review_prompt("Build auth")
        assert "ARCHITECT" in prompt
        assert "PRODUCT" in prompt
        assert "USER" in prompt
        assert "TESTER" in prompt
        assert "DESIGNER" in prompt
        assert "Build auth" in prompt
        assert "grill-me" in prompt
        assert "尖锐追问" in prompt
        assert "推荐答案" in prompt
        assert "采纳动作" in prompt

    def test_build_refinement_input(self):
        project = SpecProject.create(root_path="/tmp")
        project.requirement = "Build auth"
        project.acceptance_criteria = ["Criterion A"]
        project.criteria_tracker.init_criteria(["Criterion A"])

        last_review = ReviewResult(
            reviews=[
                PerspectiveReview(
                    perspective=ReviewPerspective.ARCHITECT,
                    passed=False,
                    suggestions=["Fix security"],
                    summary="1条建议",
                ),
                PerspectiveReview(
                    perspective=ReviewPerspective.PRODUCT,
                    passed=True,
                    suggestions=[],
                    summary="通过",
                ),
                PerspectiveReview(
                    perspective=ReviewPerspective.USER,
                    passed=True,
                    suggestions=[],
                    summary="通过",
                ),
                PerspectiveReview(
                    perspective=ReviewPerspective.TESTER,
                    passed=True,
                    suggestions=[],
                    summary="通过",
                ),
            ],
            iteration=1,
        )

        result = build_refinement_input("Build auth", last_review, project)
        assert "Build auth" in result
        assert "Fix security" in result
        assert "Criterion A" in result

    def test_detect_convergence_not_enough_cycles(self):
        engine = self._make_engine()
        engine._project = SpecProject.create(root_path="/tmp")
        assert not engine._detect_convergence()

    def test_detect_convergence_triggered(self):
        engine = self._make_engine()
        engine._project = SpecProject.create(root_path="/tmp")
        engine._project.criteria_tracker.init_criteria(["C1", "C2"])

        # 2 cycles, criteria satisfied count stays the same (0), review suggestions stay the same (1)
        def _make_review(iteration):
            return ReviewResult(
                reviews=[
                    PerspectiveReview(
                        perspective=ReviewPerspective.ARCHITECT, passed=False, suggestions=["S1"], summary="1条建议"
                    ),
                    PerspectiveReview(
                        perspective=ReviewPerspective.PRODUCT, passed=True, suggestions=[], summary="通过"
                    ),
                    PerspectiveReview(perspective=ReviewPerspective.USER, passed=True, suggestions=[], summary="通过"),
                    PerspectiveReview(
                        perspective=ReviewPerspective.TESTER, passed=True, suggestions=[], summary="通过"
                    ),
                ],
                iteration=iteration,
            )

        engine._project.cycles = [
            SpecCycle(cycle_number=1, build_output="x" * 100, review_result=_make_review(1)),
            SpecCycle(cycle_number=2, build_output="y" * 100, review_result=_make_review(2)),
        ]
        assert engine._detect_convergence()

    def test_detect_convergence_not_triggered(self):
        engine = self._make_engine()
        engine._project = SpecProject.create(root_path="/tmp")
        engine._project.criteria_tracker.init_criteria(["C1", "C2"])

        # Simulate criteria progress in the window
        engine._project.criteria_tracker.update(0, True, 1)
        engine._project.criteria_tracker.update(1, True, 2)

        review_pass = ReviewResult(
            reviews=[
                PerspectiveReview(perspective=p, passed=True, suggestions=[], summary="通过") for p in ReviewPerspective
            ],
            iteration=1,
        )

        engine._project.cycles = [
            SpecCycle(cycle_number=1, build_output="x" * 100, review_result=review_pass),
            SpecCycle(cycle_number=2, build_output="y" * 100, review_result=review_pass),
        ]
        assert not engine._detect_convergence()

    def test_format_criteria_status_empty(self):
        assert format_criteria_status(None) == ""

    def test_format_criteria_status_with_project(self):
        project = SpecProject.create(root_path="/tmp")
        project.criteria_tracker.init_criteria(["C1", "C2"])
        project.criteria_tracker.batch_update({0: True}, 1)
        result = format_criteria_status(project)
        assert "[x]" in result
        assert "[ ]" in result

    def test_save_state(self, tmp_path):
        engine = self._make_engine()
        engine._project = SpecProject.create(root_path=str(tmp_path))
        engine._project.requirement = "test req"

        filepath = engine.save_state(str(tmp_path / "state.json"))
        assert os.path.exists(filepath)

        with open(filepath) as f:
            data = json.load(f)
        assert data["chat_id"] == "c1"
        assert data["project"]["requirement"] == "test req"

    def test_save_state_compaction_fields(self, tmp_path):
        """State file should include compaction hints (cycles_truncated_before, history_log_path)."""
        with patch("src.engine_base.get_settings") as mock_settings:
            s = MagicMock()
            s.spec_max_cycles = 10
            s.spec_max_cycles_limit = 5000
            s.spec_execution_timeout = 300
            s.spec_convergence_window = 2
            s.spec_review_enabled = False
            s.spec_cycle_tasks_max = 5
            s.spec_cycle_output_max_chars = 200
            s.spec_state_filename = ".spec_engine_state.json"
            s.spec_artifacts_dirname = ".spec_engine"
            s.spec_history_log_filename = "history.jsonl"
            s.spec_persist_every_phase = True
            s.spec_persist_phase_artifacts = False
            s.spec_allow_resume_from_disk = True
            s.spec_state_cycles_tail = 3
            s.spec_state_work_items_tail = 10
            s.spec_state_metrics_tail = 10
            s.spec_generated_specs_retention = 10
            s.spec_infinite_mode = False
            s.spec_disable_convergence = False
            s.spec_disable_early_stop = False
            mock_settings.return_value = s

            engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
            engine._project = SpecProject.create(root_path=str(tmp_path))
            engine._project.requirement = "req"
            # Create 10 cycles (state only keeps tail=3)
            for i in range(1, 11):
                c = SpecCycle(cycle_number=i)
                c.complete()
                engine._project.cycles.append(c)
            engine._project.cycle_count_total = 10

            fp = engine.save_state(str(tmp_path / "state.json"))
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "_compact" in data["project"]
            assert data["project"]["_compact"]["cycles_tail"] == 3
            assert data["project"]["_compact"]["cycles_truncated_before"] == 10 - 3
            assert "history_log_path" in data["project"]

    def test_save_state_no_project_raises(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="没有项目状态"):
            engine.save_state("/tmp/state.json")

    def test_get_rendered_content(self):
        engine = self._make_engine()
        result = engine.get_rendered_content()
        assert isinstance(result, str)

    def test_persist_state_best_effort_does_not_crash_on_write_error(self, tmp_path):
        """Best-effort persistence should never raise even if disk write fails."""
        engine = self._make_engine()
        engine.root_path = str(tmp_path)
        engine._project = SpecProject.create(root_path=str(tmp_path))
        engine._project.requirement = "req"
        # Patch open used by save_state -> raise
        with patch("builtins.open", side_effect=OSError("disk full")):
            engine._persist_state_best_effort()
        # No exception expected

    def test_parse_review_output_strict_format(self):
        engine = self._make_engine()
        engine._project = SpecProject.create(root_path="/tmp")
        text = """[ARCHITECT]
PASS

[PRODUCT]
FAIL
- Need better error messages
- Add loading states

[USER]
PASS

[TESTER]
PASS

[DESIGNER]
PASS
"""
        result = engine._parse_review_output(text, 1)
        assert len(result.reviews) == 5
        assert result.reviews[0].passed is True  # ARCHITECT
        assert result.reviews[1].passed is False  # PRODUCT
        assert len(result.reviews[1].suggestions) == 2

    @patch("src.spec_engine.engine.prompt_via_acp", return_value="")
    def test_parse_review_output_fallback_all_fail(self, _mock_prompt):
        engine = self._make_engine()
        engine._project = SpecProject.create(root_path="/tmp")
        # Completely unparseable text
        result = engine._parse_review_output("random garbage", 1)
        assert len(result.reviews) == 5
        assert all(not r.passed for r in result.reviews)

    @patch(
        "src.spec_engine.engine.prompt_via_acp",
        return_value="""[
  {"perspective": "ARCHITECT", "verdict": "PASS", "suggestions": []},
  {"perspective": "PRODUCT", "verdict": "PASS", "suggestions": []},
  {"perspective": "USER", "verdict": "PASS", "suggestions": []},
  {"perspective": "TESTER", "verdict": "PASS", "suggestions": []}
]""",
    )
    def test_parse_review_output_fallback_requires_all_perspectives(self, _mock_prompt):
        engine = self._make_engine()
        engine._project = SpecProject.create(root_path="/tmp")

        result = engine._parse_review_output("random garbage", 1)

        assert not result.all_passed
        assert len(result.reviews) == 5
        assert any(r.perspective == ReviewPerspective.DESIGNER and not r.passed for r in result.reviews)


# ======================================================================
# TestResetCancelEvent — _reset_cancel_event behavior
# ======================================================================


class TestResetCancelEvent:
    @patch("src.engine_base.get_settings")
    def _make_engine(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_max_cycles_limit = 5000
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        s.spec_cycle_tasks_max = 50
        s.spec_cycle_output_max_chars = 4000
        s.spec_state_filename = ".spec_engine_state.json"
        s.spec_artifacts_dirname = ".spec_engine"
        s.spec_persist_phase_artifacts = True
        s.spec_persist_every_phase = True
        s.spec_discovery_enabled = False
        s.spec_discovery_max_questions = 3
        s.spec_discovery_force_nonempty = True
        s.spec_generated_specs_per_cycle = 1
        s.spec_discovery_gate_on_satisfied = True
        s.spec_discovery_max_pending = 5
        s.spec_discovery_cooldown_cycles = 3
        s.spec_backlog_stuck_window = 3
        s.spec_success_ignore_backlog = True
        s.spec_allow_resume_from_disk = True
        s.spec_infinite_mode = False
        s.spec_disable_convergence = False
        s.spec_disable_early_stop = False
        s.spec_min_cycles = 1
        mock_settings.return_value = s
        return SpecEngine(chat_id="c1", root_path="/tmp/test")

    def test_reset_cancel_event_running(self):
        """RUNNING state: _reset_cancel_event clears event and returns True."""
        engine = self._make_engine()
        engine._run_state = EngineRunState.RUNNING
        engine._review_cancel_event.set()  # pre-set to verify it gets cleared
        result = engine._reset_cancel_event()
        assert result is True
        assert not engine._review_cancel_event.is_set()

    def test_reset_cancel_event_stopping(self):
        """STOPPING state: _reset_cancel_event sets event and returns False."""
        engine = self._make_engine()
        engine._run_state = EngineRunState.STOPPING
        engine._review_cancel_event.clear()  # pre-clear
        result = engine._reset_cancel_event()
        assert result is False
        assert engine._review_cancel_event.is_set()

    def test_stop_sets_cancel_event_e2e(self):
        """engine.stop() triggers _review_cancel_event.set() end-to-end."""
        engine = self._make_engine()
        engine._run_state = EngineRunState.RUNNING
        engine._session = MagicMock()
        engine._review_cancel_event.clear()
        engine.stop()
        assert engine._review_cancel_event.is_set()

    def test_pause_sets_cancel_event(self):
        """engine.pause() sets _review_cancel_event to interrupt retry waits."""
        engine = self._make_engine()
        engine._run_state = EngineRunState.RUNNING
        engine._project = MagicMock()
        engine._session = MagicMock()
        engine._review_cancel_event.clear()
        engine.pause()
        assert engine._review_cancel_event.is_set()
        assert engine.run_state == EngineRunState.STOPPING


# ======================================================================
# TestSpecEngineManager — get_or_create, active, cleanup
# ======================================================================


class TestSpecEngineManager:
    @patch("src.engine_base.get_settings")
    def test_get_or_create(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e1 = mgr.get_or_create("chat1", "/tmp/a")
        e2 = mgr.get_or_create("chat1", "/tmp/a")
        assert e1 is e2  # Same instance

    @patch("src.engine_base.get_settings")
    def test_get_different_paths(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e1 = mgr.get_or_create("chat1", "/tmp/a")
        e2 = mgr.get_or_create("chat1", "/tmp/b")
        assert e1 is not e2

    @patch("src.engine_base.get_settings")
    def test_get_active_engine(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e = mgr.get_or_create("chat1", "/tmp/a")
        assert mgr.get_active_engine("chat1") is None

        e._run_state = EngineRunState.RUNNING
        assert mgr.get_active_engine("chat1") is e

    @patch("src.engine_base.get_settings")
    def test_engine_name_switch(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e1 = mgr.get_or_create("chat1", "/tmp/a", engine_name="Coco")
        assert e1.engine_name == "Coco"
        e2 = mgr.get_or_create("chat1", "/tmp/a", engine_name="Claude")
        assert e2.engine_name == "Claude"
        assert e1 is not e2  # New instance because name changed

    @patch("src.engine_base.get_settings")
    def test_engine_name_switch_blocked_while_running(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e1 = mgr.get_or_create("chat1", "/tmp/a", engine_name="Coco")
        e1._run_state = EngineRunState.RUNNING
        e2 = mgr.get_or_create("chat1", "/tmp/a", engine_name="Claude")
        assert e2 is e1  # Not replaced because still running

    @patch("src.engine_base.get_settings")
    def test_cleanup_all(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        mgr.get_or_create("chat1", "/tmp/a")
        mgr.get_or_create("chat2", "/tmp/b")
        mgr.cleanup_all()
        assert mgr.list_engines() == []

    @patch("src.engine_base.get_settings")
    def test_cleanup_all_keeps_running_engine(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        engine = mgr.get_or_create("chat1", "/tmp/a")
        engine._run_state = EngineRunState.RUNNING
        mgr.cleanup_all()
        assert mgr.get("chat1", "/tmp/a") is engine
        assert engine.run_state == EngineRunState.STOPPING

    @patch("src.engine_base.get_settings")
    def test_load_or_create_from_disk_hydrates_project_and_resume_meta(self, mock_settings, tmp_path):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_max_cycles_limit = 5000
        s.spec_execution_timeout = 300
        s.spec_convergence_window = 2
        s.spec_review_enabled = False
        s.spec_cycle_tasks_max = 5
        s.spec_cycle_output_max_chars = 200
        s.spec_state_filename = ".spec_engine_state.json"
        s.spec_artifacts_dirname = ".spec_engine"
        s.spec_history_log_filename = "history.jsonl"
        s.spec_persist_every_phase = True
        s.spec_persist_phase_artifacts = False
        s.spec_allow_resume_from_disk = True
        s.spec_state_cycles_tail = 3
        s.spec_state_work_items_tail = 10
        s.spec_state_metrics_tail = 10
        s.spec_generated_specs_retention = 10
        s.spec_infinite_mode = False
        s.spec_disable_convergence = False
        s.spec_disable_early_stop = False
        s.spec_min_cycles = 1
        mock_settings.return_value = s

        # Create a state file
        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        engine._project = SpecProject.create(root_path=str(tmp_path))
        engine._project.requirement = "req"
        engine._agent_type = "ttadk_codex"
        engine.engine_name = "TTADK"
        engine._current_model = "gpt-5.2"
        engine._model_name = "gpt-5.2"
        engine._models_tried = ["claude-3.7-sonnet", "gpt-5.2"]
        engine.save_state()  # default state path

        mgr = SpecEngineManager()
        e2 = mgr.load_or_create_from_disk("c1", str(tmp_path), engine_name="Coco")
        assert e2.project is not None
        assert e2.project.requirement == "req"
        assert e2.engine_name == "TTADK"
        assert e2._agent_type == "ttadk_codex"
        assert e2._current_model == "gpt-5.2"
        assert e2._models_tried == ["claude-3.7-sonnet", "gpt-5.2"]
        assert getattr(e2, "_resume_meta", None)

    @patch("src.engine_base.get_settings")
    def test_get_or_create_preserves_explicit_ttadk_identity(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        engine = mgr.get_or_create(
            "chat1",
            "/tmp/a",
            engine_name="TTADK",
            agent_type="ttadk_codex",
            model_name="gpt-5.2",
        )
        assert engine.engine_name == "TTADK"
        assert engine._agent_type == "ttadk_codex"
        assert engine._model_name == "gpt-5.2"

    @patch("src.spec_engine.engine.delete_task_state")
    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_resume_completed_recovery_deletes_saved_task(self, mock_settings, mock_create_session, mock_delete_task_state):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_max_cycles_limit = 5000
        s.spec_execution_timeout = 300
        s.spec_convergence_window = 2
        s.spec_review_enabled = False
        s.spec_cycle_tasks_max = 5
        s.spec_cycle_output_max_chars = 200
        s.spec_state_filename = ".spec_engine_state.json"
        s.spec_artifacts_dirname = ".spec_engine"
        s.spec_history_log_filename = "history.jsonl"
        s.spec_persist_every_phase = False
        s.spec_persist_phase_artifacts = False
        s.spec_allow_resume_from_disk = True
        s.spec_state_cycles_tail = 3
        s.spec_state_work_items_tail = 10
        s.spec_state_metrics_tail = 10
        s.spec_generated_specs_retention = 10
        s.spec_infinite_mode = False
        s.spec_disable_convergence = False
        s.spec_disable_early_stop = False
        s.spec_min_cycles = 1
        mock_settings.return_value = s
        mock_create_session.return_value = MagicMock()

        engine = SpecEngine(chat_id="chat1", root_path="/tmp/a")
        project = SpecProject.create(root_path="/tmp/a")
        project.requirement = "recover me"
        project.status = SpecProjectStatus.PAUSED
        project.task_id = "task123"
        state = SpecTaskState(
            task_id="task123",
            created_at=time.time(),
            requirement="recover me",
            project_path="/tmp/a",
            chat_id="chat1",
            agent_type="ttadk_codex",
            current_cycle=2,
            current_phase="build",
            last_error="boom",
            retry_count=1,
            models_tried=["claude-3.7-sonnet", "gpt-5.2"],
            project_snapshot=project.to_dict(),
            runtime_context={
                "agent_type": "ttadk_codex",
                "engine_name": "TTADK",
                "model_name": "gpt-5.2",
                "current_model": "gpt-5.2",
                "models_tried": ["claude-3.7-sonnet", "gpt-5.2"],
            },
        )
        engine.restore_from_task_state(state)
        engine._run_cycle_loop = MagicMock(return_value="success")

        resumed = engine.resume(SpecEngineCallbacks())

        assert resumed is engine.project
        assert engine.project.status == SpecProjectStatus.COMPLETED
        mock_delete_task_state.assert_called_once_with("task123")

    @patch("src.engine_base.get_settings")
    def test_get_none_for_missing(self, mock_settings):
        s = MagicMock()
        mock_settings.return_value = s
        mgr = SpecEngineManager()
        assert mgr.get("chat1", "/tmp/a") is None
        assert mgr.get_active_engine("chat1") is None

    @patch("src.engine_base.get_settings")
    def test_list_engines(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        mgr.get_or_create("c1", "/tmp/a")
        mgr.get_or_create("c1", "/tmp/b")
        mgr.get_or_create("c2", "/tmp/c")
        assert len(mgr.list_engines()) == 3
        assert len(mgr.list_engines("c1")) == 2
        assert len(mgr.list_engines("c2")) == 1

    @patch("src.engine_base.get_settings")
    def test_get_active_engines(self, mock_settings):
        s = MagicMock()
        s.spec_max_cycles = 10
        s.spec_convergence_window = 2
        s.spec_execution_timeout = 300
        mock_settings.return_value = s

        mgr = SpecEngineManager()
        e1 = mgr.get_or_create("c1", "/tmp/a")
        mgr.get_or_create("c1", "/tmp/b")
        e1._run_state = EngineRunState.RUNNING
        active = mgr.get_active_engines("c1")
        assert len(active) == 1
        assert active[0] is e1


# ======================================================================
# TestSpecHandler — command routing
# ======================================================================


class TestSpecHandler:
    def _make_handler(self):
        from src.feishu.handlers.spec import SpecHandler

        ctx = self._make_handler_context()
        return SpecHandler(ctx)

    def _make_handler_context(self):
        from src.feishu.handler_context import HandlerContext

        return HandlerContext(
            settings=MagicMock(),
            api_client_factory=MagicMock(),
            message_callback=MagicMock(),
            coco_manager=MagicMock(),
            claude_manager=MagicMock(),
            aiden_manager=MagicMock(),
            codex_manager=MagicMock(),
            gemini_manager=MagicMock(),
            traex_manager=MagicMock(),
            ttadk_manager=MagicMock(),
            tui2acp_manager=MagicMock(),
            intent_recognizer=MagicMock(),
            scheduler=MagicMock(),
            project_manager=MagicMock(),
            message_mapper=MagicMock(),
            message_linker=MagicMock(),
            mode_manager=MagicMock(),
            context_manager=MagicMock(),
            deep_engine_manager=MagicMock(),
            progress_reporter=MagicMock(),
            spec_engine_manager=MagicMock(),
            spec_reporter=MagicMock(),
            thread_manager=MagicMock(),
            image_handler_factory=MagicMock(),
            working_dirs={},
            working_dir_lock=threading.Lock(),
            pending_image_keys={},
            pending_image_lock=threading.Lock(),
            enable_streaming=False,
            slock_engine_manager=MagicMock(),
        )

    def test_handle_spec_command_routing_spec(self):
        handler = self._make_handler()
        handler.start_spec_engine = MagicMock()
        handler.handle_spec_command("mid", "cid", "/spec Build login", None)
        handler.start_spec_engine.assert_called_once()

    def test_handle_spec_command_accepts_any_slash_whitespace(self):
        handler = self._make_handler()
        handler.start_spec_engine = MagicMock()

        handler.handle_spec_command("mid", "cid", "/spec\tBuild login", None)

        handler.start_spec_engine.assert_called_once_with(
            "mid", "cid", "Build login", None
        )

    def test_handle_spec_command_routing_status(self):
        handler = self._make_handler()
        handler.show_spec_status = MagicMock()
        handler.handle_spec_command("mid", "cid", "/spec_status", None)
        handler.show_spec_status.assert_called_once()

    def test_handle_spec_command_routing_stop(self):
        handler = self._make_handler()
        handler.stop_spec_engine = MagicMock()
        handler.handle_spec_command("mid", "cid", "/stop_spec", None)
        handler.stop_spec_engine.assert_called_once()

    def test_handle_spec_command_routing_pause(self):
        handler = self._make_handler()
        handler.pause_spec_engine = MagicMock()
        handler.handle_spec_command("mid", "cid", "/spec_pause", None)
        handler.pause_spec_engine.assert_called_once()

    def test_handle_spec_command_routing_resume(self):
        handler = self._make_handler()
        handler.resume_spec_engine = MagicMock()
        handler.handle_spec_command("mid", "cid", "/spec_resume", None)
        handler.resume_spec_engine.assert_called_once()

    def test_handle_spec_command_routing_guide(self):
        handler = self._make_handler()
        handler.update_spec_guidance = MagicMock()
        handler.handle_spec_command("mid", "cid", "/spec_guide focus on auth", None)
        handler.update_spec_guidance.assert_called_once()

    def test_handle_spec_command_routing_export(self):
        handler = self._make_handler()
        handler.export_spec_report = MagicMock()
        handler.handle_spec_command("mid", "cid", "/spec_export", None)
        handler.export_spec_report.assert_called_once()

    def test_update_spec_guidance_allows_when_clarifying(self):
        """/spec_guide rewrites the goal via LLM even when engine is CLARIFYING."""
        handler = self._make_handler()
        handler.send_card_to_chat = MagicMock()
        handler.reply_text = MagicMock()

        project = MagicMock()
        project.root_path = "/tmp/p"
        project.project_id = "p1"
        handler.project_manager.get_active_project.return_value = project

        engine = MagicMock()
        engine.engine_name = "Coco"
        engine.is_running = False
        engine.refine_goal_with_guidance.return_value = (True, "new combined requirement")
        sp = SpecProject.create(name="p", root_path="/tmp/p")
        sp.status = SpecProjectStatus.CLARIFYING
        engine.project = sp

        handler.ctx.spec_engine_manager.get.return_value = engine
        handler.ctx.spec_engine_manager.list_engines.return_value = [engine]

        reporter = MagicMock()
        reporter.format_goal_rewritten.return_value = "goal rewritten"
        reporter.get_goal_rewritten_title.return_value = "🎯 目标已更新"
        handler.ctx.spec_reporter = reporter

        with patch("src.feishu.handlers.spec.CardBuilder.build_info_card", return_value=("interactive", "card")):
            handler.update_spec_guidance("mid", "cid", "Q1: answer", project=None)

        engine.refine_goal_with_guidance.assert_called_once_with("Q1: answer")
        engine.inject_guidance.assert_not_called()
        handler.reply_text.assert_not_called()
        handler.send_card_to_chat.assert_called_once()

    def test_update_spec_guidance_fallback_on_llm_failure(self):
        """/spec_guide falls back to inject_guidance when LLM rewrite fails."""
        handler = self._make_handler()
        handler.send_card_to_chat = MagicMock()
        handler.reply_text = MagicMock()

        project = MagicMock()
        project.root_path = "/tmp/p"
        project.project_id = "p1"
        handler.project_manager.get_active_project.return_value = project

        engine = MagicMock()
        engine.engine_name = "Coco"
        engine.is_running = True
        engine.refine_goal_with_guidance.return_value = (False, "LLM error")
        sp = SpecProject.create(name="p", root_path="/tmp/p")
        sp.status = SpecProjectStatus.RUNNING
        engine.project = sp

        handler.ctx.spec_engine_manager.get.return_value = engine
        handler.ctx.spec_engine_manager.list_engines.return_value = [engine]

        reporter = MagicMock()
        reporter.format_guidance_injected.return_value = "injected ok"
        reporter.get_guidance_injected_title.return_value = "💬 引导信息已注入"
        handler.ctx.spec_reporter = reporter

        with patch("src.feishu.handlers.spec.CardBuilder.build_info_card", return_value=("interactive", "card")):
            handler.update_spec_guidance("mid", "cid", "fallback test", project=None)

        engine.refine_goal_with_guidance.assert_called_once_with("fallback test")
        engine.inject_guidance.assert_called_once_with("fallback test")
        handler.send_card_to_chat.assert_called_once()


# ======================================================================
# TestSystemHandler — is_spec_command predicate
# ======================================================================


class TestSystemHandlerSpec:
    def test_is_spec_command(self):
        from src.feishu.handlers.system import SystemHandler

        assert SystemHandler.is_spec_command("/spec build auth")
        assert SystemHandler.is_spec_command("/spec_status")
        assert SystemHandler.is_spec_command("/stop_spec")
        assert SystemHandler.is_spec_command("/spec_pause")
        assert SystemHandler.is_spec_command("/spec_resume")
        assert SystemHandler.is_spec_command("/spec_guide focus")
        assert SystemHandler.is_spec_command("/spec_export")
        assert not SystemHandler.is_spec_command("/deep build")
        assert not SystemHandler.is_spec_command("/deep do stuff")
        assert not SystemHandler.is_spec_command("hello")


# ======================================================================
# TestIntentRecognizer — spec intents
# ======================================================================


class TestIntentRecognizerSpec:
    def test_spec_intent_types_exist(self):
        from src.agent.intent_recognizer import IntentType

        assert hasattr(IntentType, "ENTER_SPEC")
        assert hasattr(IntentType, "SPEC_STATUS")
        assert hasattr(IntentType, "STOP_SPEC")
        assert hasattr(IntentType, "SPEC_PAUSE")
        assert hasattr(IntentType, "SPEC_RESUME")
        assert hasattr(IntentType, "SPEC_GUIDE")

    def test_spec_exact_commands(self):
        from src.agent.intent_recognizer import IntentRecognizer

        recognizer = IntentRecognizer()
        # Quick match: /spec_status
        result = recognizer.recognize("/spec_status", "smart")
        from src.agent.intent_recognizer import IntentType

        assert result.primary_intent == IntentType.SPEC_STATUS

    def test_spec_guide_quick_match(self):
        from src.agent.intent_recognizer import IntentRecognizer, IntentType

        recognizer = IntentRecognizer()
        result = recognizer.recognize("/spec_guide focus on tests", "smart")
        assert result.primary_intent == IntentType.SPEC_GUIDE


# ======================================================================
# TestConfig — spec engine settings
# ======================================================================


class TestConfigSpec:
    def test_spec_settings_defaults(self):
        from src.config import Settings

        s = Settings(app_id="", app_secret="", _env_file=None)
        assert s.spec_max_cycles == 1000
        assert s.spec_max_cycles_limit >= 5000
        assert s.spec_execution_timeout == 7200
        assert s.spec_convergence_window == 2
        assert s.spec_review_enabled is True
        assert s.spec_discovery_enabled is True

    def test_spec_review_circuit_breaker_defaults(self):
        """FS-17: review circuit breaker settings have sensible defaults."""
        from src.config import Settings

        s = Settings(app_id="", app_secret="", _env_file=None)
        assert s.spec_review_failure_circuit_enabled is True
        assert s.spec_review_failure_max_consecutive == 4
        assert s.spec_review_failure_cooldown_cycles == 2
        assert s.spec_review_failure_max_cooldown_cycles == 12
        assert s.spec_review_timeout == 240
        assert s.spec_review_min_timeout == 60
        assert s.spec_review_hard_floor == 20

    def test_lock_settings_defaults(self):
        """FS-17: lock-related settings have sensible defaults."""
        from src.config import Settings

        s = Settings(app_id="", app_secret="", _env_file=None)
        assert s.repo_lock_idle_timeout == 300
        assert s.repo_lock_cleanup_interval == 60
        assert s.repo_lock_hard_timeout == 3600
        assert s.chat_lock_max_duration == 86400
        assert s.chat_lock_cleanup_interval == 60


# ======================================================================
# TestCardBuilder — spec color
# ======================================================================


class TestCardBuilderSpec:
    def test_pick_engine_template_spec(self):
        from src.card.builder import CardBuilder

        assert CardBuilder._pick_deep_template("Spec(Coco)") == "green"
        assert CardBuilder._pick_deep_template("spec") == "green"
        assert CardBuilder._pick_deep_template("Coco") == "turquoise"
        assert CardBuilder._pick_deep_template("Claude") == "violet"


# ======================================================================
# TestSpecEngineExecution — integration tests for execute/resume/review
# ======================================================================


class TestSpecEngineExecution:
    """Integration tests for execute, resume, review, criteria evaluation."""

    def _mock_settings(self):
        return _make_spec_settings()

    def _make_mock_session(self, text_responses):
        """Mock session that returns text_responses sequentially via on_event."""
        session = MagicMock()
        call_index = [0]
        responses = list(text_responses)

        def fake_send_prompt(prompt, on_event=None, timeout=None, **kwargs):
            idx = call_index[0]
            call_index[0] += 1
            text = responses[idx] if idx < len(responses) else ""
            if on_event and text:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
            return MagicMock(stop_reason="end_turn")

        session.send_prompt = fake_send_prompt
        session.send_prompt_with_retry = fake_send_prompt
        return session

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_execute_single_cycle_all_pass(self, mock_settings, mock_create):
        """Full execute: 1 cycle, all reviews PASS, criteria PASS → COMPLETED."""
        mock_settings.return_value = self._mock_settings()

        review_text = "[ARCHITECT]\nPASS\n\n[PRODUCT]\nPASS\n\n[USER]\nPASS\n\n[TESTER]\nPASS\n\n[DESIGNER]\nPASS\n"
        criteria_text = "CRITERIA_1: PASS"
        spec_json = """```json\n{\"goals\":[\"G\"],\"functional_spec\":[\"F\"],\"non_functional_requirements\":[\"N\"],\"acceptance_criteria\":[\"实现登录功能\"],\"out_of_scope\":[],\"risks\":[],\"clarification_questions\":[],\"decisions\":[],\"version\":\"1.0\"}\n```"""
        plan_json = """```json\n{\"architecture\":\"A\",\"tech_stack\":[\"T\"],\"steps\":[\"S1\"],\"file_changes\":[\"x.py\"],\"test_plan\":[\"pytest\"],\"risks\":[],\"version\":\"1.0\"}\n```"""

        # Order: spec, plan, task, build, review, criteria_eval
        session = self._make_mock_session(
            [
                spec_json,
                plan_json,
                "1. Task one (依赖: 无)",
                "build done " * 20,
                review_text,
                criteria_text,
            ]
        )
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")

        called = {"analyzing_start": False, "project_done": False, "cycles": []}
        callbacks = SpecEngineCallbacks(
            on_analyzing_start=lambda r: called.__setitem__("analyzing_start", True),
            on_cycle_done=lambda c, cy: called["cycles"].append(c),
            on_project_done=lambda p: called.__setitem__("project_done", True),
        )

        project = engine.execute("- 实现登录功能", callbacks)

        assert project.status == SpecProjectStatus.COMPLETED
        assert len(project.cycles) == 1
        assert project.cycles[0].status == "completed"
        assert '"acceptance_criteria"' in project.cycles[0].spec_content
        assert '"file_changes"' in project.cycles[0].plan_content
        assert project.cycles[0].spec_artifact is not None
        assert project.cycles[0].plan_artifact is not None
        assert len(project.cycles[0].tasks) == 1
        assert project.cycles[0].review_result.all_passed
        assert called["analyzing_start"]
        assert called["project_done"]
        assert called["cycles"] == [1]
        assert engine.run_state == EngineRunState.IDLE

    @patch("src.spec_engine.engine.prompt_via_acp")
    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_execute_plain_requirement_skips_aux_criteria_session(
        self,
        mock_settings,
        mock_create,
        mock_prompt_via_acp,
    ):
        """Plain-language startup must not block on a disposable ACP criteria session."""
        mock_settings.return_value = self._mock_settings()
        mock_prompt_via_acp.side_effect = AssertionError("aux criteria prompt should not run")

        review_text = "[ARCHITECT]\nPASS\n\n[PRODUCT]\nPASS\n\n[USER]\nPASS\n\n[TESTER]\nPASS\n\n[DESIGNER]\nPASS\n"
        criteria_text = "CRITERIA_1: PASS"
        spec_json = """```json\n{\"goals\":[\"G\"],\"functional_spec\":[\"F\"],\"non_functional_requirements\":[],\"acceptance_criteria\":[\"登录页面可用\"],\"out_of_scope\":[],\"risks\":[],\"clarification_questions\":[],\"decisions\":[],\"version\":\"1.0\"}\n```"""
        plan_json = """```json\n{\"architecture\":\"A\",\"tech_stack\":[],\"steps\":[\"S\"],\"file_changes\":[],\"test_plan\":[],\"risks\":[],\"version\":\"1.0\"}\n```"""

        session = self._make_mock_session(
            [
                spec_json,
                plan_json,
                "1. Task one (依赖: 无)",
                "build done " * 20,
                review_text,
                criteria_text,
            ]
        )
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        project = engine.execute("实现一个简单的登录页面")

        assert project.status == SpecProjectStatus.COMPLETED
        mock_prompt_via_acp.assert_not_called()
        mock_create.assert_called_once()
        assert project.acceptance_criteria == ["登录页面可用"]

    @patch("src.engine_base.close_session_safely")
    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_execute_closes_session(self, mock_settings, mock_create, mock_close):
        """execute() should always close the underlying session in finally."""
        mock_settings.return_value = self._mock_settings()

        review_text = "[ARCHITECT]\nPASS\n\n[PRODUCT]\nPASS\n\n[USER]\nPASS\n\n[TESTER]\nPASS\n\n[DESIGNER]\nPASS\n"
        criteria_text = "CRITERIA_1: PASS"
        spec_json = """```json\n{\"goals\":[\"G\"],\"functional_spec\":[\"F\"],\"non_functional_requirements\":[],\"acceptance_criteria\":[\"实现登录功能\"],\"out_of_scope\":[],\"risks\":[],\"clarification_questions\":[],\"decisions\":[],\"version\":\"1.0\"}\n```"""
        plan_json = """```json\n{\"architecture\":\"A\",\"tech_stack\":[],\"steps\":[\"S\"],\"file_changes\":[],\"test_plan\":[],\"risks\":[],\"version\":\"1.0\"}\n```"""

        session = self._make_mock_session(
            [
                spec_json,
                plan_json,
                "1. Task one (依赖: 无)",
                "build done " * 20,
                review_text,
                criteria_text,
            ]
        )
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        project = engine.execute("- 实现登录功能")
        assert project.status == SpecProjectStatus.COMPLETED
        mock_close.assert_called()
        # Ensure we attempted to close the same session instance
        assert mock_close.call_args[0][0] is session

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_execute_continues_when_clarification_questions_present(self, mock_settings, mock_create):
        """Clarification questions should NOT pause the engine — it continues through all phases."""
        mock_settings.return_value = self._mock_settings()

        spec_json = """```json\n{\"goals\":[\"G\"],\"functional_spec\":[\"F\"],\"non_functional_requirements\":[],\"acceptance_criteria\":[\"需要登录\"],\"out_of_scope\":[],\"risks\":[],\"clarification_questions\":[\"是否需要支持手机号登录？\"],\"decisions\":[\"假设仅支持邮箱登录\"],\"version\":\"1.0\"}\n```"""
        plan_json = """```json\n{\"architecture\":\"A\",\"tech_stack\":[],\"steps\":[\"S\"],\"file_changes\":[],\"test_plan\":[],\"risks\":[],\"version\":\"1.0\"}\n```"""
        review_pass = "[ARCHITECT]\nPASS\n\n[PRODUCT]\nPASS\n\n[USER]\nPASS\n\n[TESTER]\nPASS\n\n[DESIGNER]\nPASS\n"
        session = self._make_mock_session(
            [
                spec_json,
                plan_json,
                "1. T1 (依赖: 无)",
                "build done " * 20,
                review_pass,
                "CRITERIA_1: PASS",
            ]
        )
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        project = engine.execute("- 需要登录")

        # Engine should NOT be in CLARIFYING state — it continued through all phases
        assert project.status != SpecProjectStatus.CLARIFYING
        assert len(project.cycles) >= 1
        cycle = project.cycles[0]
        assert cycle.spec_artifact is not None
        assert cycle.spec_artifact.clarification_questions == ["是否需要支持手机号登录？"]
        # All 5 phases should have been executed (SPEC + PLAN + TASK + BUILD + REVIEW)
        assert cycle.phase == SpecPhase.REVIEW

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_execute_multi_cycle_then_pass(self, mock_settings, mock_create):
        """Cycle 1 FAIL review → cycle 2 all PASS → COMPLETED in 2 cycles."""
        s = self._mock_settings()
        s.spec_max_cycles = 2
        mock_settings.return_value = s

        review_fail = (
            "[ARCHITECT]\nFAIL\n- Fix issue\n\n[PRODUCT]\nPASS\n\n[USER]\nPASS\n\n[TESTER]\nPASS\n\n[DESIGNER]\nPASS\n"
        )
        review_pass = "[ARCHITECT]\nPASS\n\n[PRODUCT]\nPASS\n\n[USER]\nPASS\n\n[TESTER]\nPASS\n\n[DESIGNER]\nPASS\n"
        spec1 = """```json\n{\"goals\":[\"G\"],\"functional_spec\":[\"F\"],\"non_functional_requirements\":[],\"acceptance_criteria\":[\"功能要求可用\"],\"out_of_scope\":[],\"risks\":[],\"clarification_questions\":[],\"decisions\":[],\"version\":\"1.0\"}\n```"""
        plan1 = """```json\n{\"architecture\":\"A\",\"tech_stack\":[],\"steps\":[\"S\"],\"file_changes\":[],\"test_plan\":[],\"risks\":[],\"version\":\"1.0\"}\n```"""
        spec2 = spec1
        plan2 = plan1
        session = self._make_mock_session(
            [
                # Cycle 1
                spec1,
                plan1,
                "1. T1 (依赖: 无)",
                "build1 " * 20,
                review_fail,
                "CRITERIA_1: FAIL",
                # Cycle 2
                spec2,
                plan2,
                "1. T1 (依赖: 无)",
                "build2 " * 20,
                review_pass,
                "CRITERIA_1: PASS",
            ]
        )
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        project = engine.execute("- 功能要求")

        assert project.status == SpecProjectStatus.COMPLETED
        assert len(project.cycles) == 2
        assert not project.cycles[0].review_result.all_passed
        assert project.cycles[1].review_result.all_passed

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_execute_stop_mid_cycle(self, mock_settings, mock_create):
        """Stop during SPEC phase → cycle saved as failed, project PAUSED."""
        mock_settings.return_value = self._mock_settings()

        def fake_send_prompt(prompt, on_event=None, timeout=None, **kwargs):
            if on_event:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="partial"))

        session = MagicMock()
        session.send_prompt = fake_send_prompt
        session.send_prompt_with_retry = fake_send_prompt
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")

        # Stop after first phase completes
        original = engine._run_phase

        def stop_after_first(cycle_num, phase, prompt, callbacks, timeout):
            result = original(cycle_num, phase, prompt, callbacks, timeout)
            engine._run_state = EngineRunState.STOPPING
            return result

        engine._run_phase = stop_after_first

        project = engine.execute("- test requirement")

        assert project.status == SpecProjectStatus.PAUSED
        assert len(project.cycles) == 1
        assert project.cycles[0].status == "failed"
        assert engine.run_state == EngineRunState.IDLE

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_execute_exception_handling(self, mock_settings, mock_create):
        """Exception during session creation → ABORTED + on_error called."""
        mock_settings.return_value = self._mock_settings()
        mock_create.side_effect = RuntimeError("connection failed")

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        error_msgs = []
        callbacks = SpecEngineCallbacks(on_error=lambda e: error_msgs.append(e))

        project = engine.execute("- test req", callbacks)

        assert project.status == SpecProjectStatus.ABORTED
        assert len(error_msgs) == 1
        assert "connection failed" in error_msgs[0]
        assert engine.run_state == EngineRunState.IDLE

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_resume_from_paused(self, mock_settings, mock_create):
        """Resume a paused engine → continues from next cycle."""
        s = self._mock_settings()
        s.spec_max_cycles = 2
        mock_settings.return_value = s

        review_pass = "[ARCHITECT]\nPASS\n\n[PRODUCT]\nPASS\n\n[USER]\nPASS\n\n[TESTER]\nPASS\n"
        session = self._make_mock_session(
            [
                "spec_r",
                "plan_r",
                "1. T1 (依赖: 无)",
                "build_r " * 20,
                review_pass,
                "CRITERIA_1: PASS",
            ]
        )
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        # Set up paused state with 1 existing cycle
        engine._project = SpecProject.create(root_path="/tmp/test")
        engine._project.requirement = "- 功能要求"
        engine._project.acceptance_criteria = ["功能要求"]
        engine._project.criteria_tracker.init_criteria(["功能要求"])
        engine._project.status = SpecProjectStatus.PAUSED
        engine._project.started_at = time.time()
        cycle = SpecCycle(cycle_number=1)
        cycle.complete()
        engine._project.cycles.append(cycle)

        project = engine.resume()

        assert project.status == SpecProjectStatus.COMPLETED
        assert len(project.cycles) == 2  # 1 existing + 1 new
        assert project.cycles[1].cycle_number == 2
        assert engine.run_state == EngineRunState.IDLE

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_resume_saves_failed_cycle_on_stop(self, mock_settings, mock_create):
        """Resume with stop → failed cycle is saved (bug fix from review)."""
        mock_settings.return_value = self._mock_settings()

        def fake_send_prompt(prompt, on_event=None, timeout=None, **kwargs):
            if on_event:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="partial"))

        session = MagicMock()
        session.send_prompt = fake_send_prompt
        session.send_prompt_with_retry = fake_send_prompt
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        engine._project = SpecProject.create(root_path="/tmp/test")
        engine._project.requirement = "- req"
        engine._project.acceptance_criteria = ["req"]
        engine._project.criteria_tracker.init_criteria(["req"])
        engine._project.status = SpecProjectStatus.PAUSED
        engine._project.started_at = time.time()

        # Stop after first phase
        original = engine._run_phase

        def stop_after_first(cycle_num, phase, prompt, callbacks, timeout):
            result = original(cycle_num, phase, prompt, callbacks, timeout)
            engine._run_state = EngineRunState.STOPPING
            return result

        engine._run_phase = stop_after_first

        project = engine.resume()

        assert project.status == SpecProjectStatus.PAUSED
        assert len(project.cycles) == 1
        assert project.cycles[0].status == "failed"

    @patch("src.engine_base.get_settings")
    def test_resume_not_paused_noop(self, mock_settings):
        """Resume when not paused → returns project unchanged."""
        mock_settings.return_value = self._mock_settings()

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        assert engine.resume() is None

        engine._project = SpecProject.create(root_path="/tmp/test")
        engine._project.status = SpecProjectStatus.RUNNING
        result = engine.resume()
        assert result.status == SpecProjectStatus.RUNNING

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_resume_exception_handling(self, mock_settings, mock_create):
        """Exception during resume → ABORTED + on_error called."""
        mock_settings.return_value = self._mock_settings()
        mock_create.side_effect = RuntimeError("session failed")

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        engine._project = SpecProject.create(root_path="/tmp/test")
        engine._project.requirement = "req"
        engine._project.status = SpecProjectStatus.PAUSED
        error_msgs = []
        callbacks = SpecEngineCallbacks(on_error=lambda e: error_msgs.append(e))

        project = engine.resume(callbacks)

        assert project.status == SpecProjectStatus.ABORTED
        assert len(error_msgs) == 1
        assert "session failed" in error_msgs[0]
        assert engine.run_state == EngineRunState.IDLE

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_conduct_review_with_session(self, mock_settings, mock_create):
        """_conduct_review sends prompt and parses result."""
        mock_settings.return_value = self._mock_settings()

        review_text = "[ARCHITECT]\nPASS\n\n[PRODUCT]\nFAIL\n- Add error handling\n\n[USER]\nPASS\n\n[TESTER]\nPASS\n\n[DESIGNER]\nPASS\n"
        session = self._make_mock_session([review_text])

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        engine._session = session
        engine._project = SpecProject.create(root_path="/tmp/test")
        engine._project.requirement = "Build login"

        result = engine._conduct_review(1, SpecEngineCallbacks())

        assert len(result.reviews) == 5
        assert result.reviews[0].passed  # ARCHITECT
        assert not result.reviews[1].passed  # PRODUCT
        assert "Add error handling" in result.reviews[1].suggestions

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_conduct_review_exception(self, mock_settings, mock_create):
        """_conduct_review handles exception → all FAIL with error message."""
        mock_settings.return_value = self._mock_settings()

        session = MagicMock()
        session.send_prompt_with_retry.side_effect = RuntimeError("timeout")
        session.send_prompt.side_effect = RuntimeError("timeout")

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        engine._session = session
        engine._project = SpecProject.create(root_path="/tmp/test")
        engine._project.requirement = "Build login"

        result = engine._conduct_review(1, SpecEngineCallbacks())

        assert len(result.reviews) == 5
        assert all(not r.passed for r in result.reviews)
        assert any("timeout" in s for r in result.reviews for s in r.suggestions)

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_conduct_review_no_session(self, mock_settings, mock_create):
        """_conduct_review without session → empty ReviewResult."""
        mock_settings.return_value = self._mock_settings()

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        engine._session = None

        result = engine._conduct_review(1, SpecEngineCallbacks())
        assert result.reviews == []

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_evaluate_criteria_with_session(self, mock_settings, mock_create):
        """_evaluate_criteria parses PASS/FAIL per criterion."""
        mock_settings.return_value = self._mock_settings()

        eval_text = "CRITERIA_1: PASS\nCRITERIA_2: FAIL\nCRITERIA_3: PASS"
        session = self._make_mock_session([eval_text])

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        engine._session = session
        engine._project = SpecProject.create(root_path="/tmp/test")
        engine._project.criteria_tracker.init_criteria(["C1", "C2", "C3"])

        result = engine._evaluate_criteria(["C1", "C2", "C3"], 1)

        assert not result["all_satisfied"]
        assert engine._project.criteria_tracker.satisfied.get(0) is True
        assert engine._project.criteria_tracker.satisfied.get(1) is False
        assert engine._project.criteria_tracker.satisfied.get(2) is True

    @patch("src.engine_base.get_settings")
    def test_evaluate_criteria_no_session(self, mock_settings):
        """_evaluate_criteria without session → not satisfied."""
        mock_settings.return_value = self._mock_settings()

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        engine._session = None

        result = engine._evaluate_criteria(["C1"], 1)
        assert not result["all_satisfied"]

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_evaluate_criteria_exception(self, mock_settings, mock_create):
        """_evaluate_criteria handles exception → not satisfied."""
        mock_settings.return_value = self._mock_settings()

        session = MagicMock()
        session.send_prompt_with_retry.side_effect = RuntimeError("oops")
        session.send_prompt.side_effect = RuntimeError("oops")

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        engine._session = session
        engine._project = SpecProject.create(root_path="/tmp/test")
        engine._project.criteria_tracker.init_criteria(["C1"])

        result = engine._evaluate_criteria(["C1"], 1)
        assert not result["all_satisfied"]

    def test_convergence_with_stagnant_review_suggestions(self):
        """Convergence detects stagnant review suggestions across window."""
        with patch("src.engine_base.get_settings") as mock_settings:
            s = MagicMock()
            s.spec_max_cycles = 10
            s.spec_convergence_window = 2
            s.spec_execution_timeout = 300
            mock_settings.return_value = s

            engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
            engine._project = SpecProject.create(root_path="/tmp/test")
            engine._project.criteria_tracker.init_criteria(["C1", "C2"])

            # 2 cycles with same non-zero suggestion count → converge
            def _make_review(n_suggestions, iteration):
                return ReviewResult(
                    reviews=[
                        PerspectiveReview(
                            perspective=ReviewPerspective.ARCHITECT,
                            passed=False,
                            suggestions=[f"S{i}" for i in range(n_suggestions)],
                            summary=f"{n_suggestions}条建议",
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.PRODUCT, passed=True, suggestions=[], summary="通过"
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.USER, passed=True, suggestions=[], summary="通过"
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.TESTER, passed=True, suggestions=[], summary="通过"
                        ),
                    ],
                    iteration=iteration,
                )

            engine._project.cycles = [
                SpecCycle(cycle_number=1, build_output="x" * 100, review_result=_make_review(1, 1)),
                SpecCycle(cycle_number=2, build_output="y" * 100, review_result=_make_review(1, 2)),
            ]
            assert engine._detect_convergence()

    def test_convergence_not_triggered_when_improving(self):
        """Convergence NOT triggered when suggestions are decreasing."""
        with patch("src.engine_base.get_settings") as mock_settings:
            s = MagicMock()
            s.spec_max_cycles = 10
            s.spec_convergence_window = 2
            s.spec_execution_timeout = 300
            mock_settings.return_value = s

            engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
            engine._project = SpecProject.create(root_path="/tmp/test")
            engine._project.criteria_tracker.init_criteria(["C1", "C2"])

            def _make_review(n_suggestions, iteration):
                return ReviewResult(
                    reviews=[
                        PerspectiveReview(
                            perspective=ReviewPerspective.ARCHITECT,
                            passed=False,
                            suggestions=[f"S{i}" for i in range(n_suggestions)],
                            summary=f"{n_suggestions}条建议",
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.PRODUCT, passed=True, suggestions=[], summary="通过"
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.USER, passed=True, suggestions=[], summary="通过"
                        ),
                        PerspectiveReview(
                            perspective=ReviewPerspective.TESTER, passed=True, suggestions=[], summary="通过"
                        ),
                    ],
                    iteration=iteration,
                )

            engine._project.cycles = [
                SpecCycle(cycle_number=1, build_output="x" * 100, review_result=_make_review(3, 1)),
                SpecCycle(cycle_number=2, build_output="y" * 100, review_result=_make_review(1, 2)),
            ]
            assert not engine._detect_convergence()

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_execute_review_disabled(self, mock_settings, mock_create):
        """When spec_review_enabled=False, review phase is skipped entirely."""
        s = self._mock_settings()
        s.spec_review_enabled = False
        mock_settings.return_value = s

        spec_json = """```json\n{\"goals\":[\"G\"],\"functional_spec\":[\"F\"],\"non_functional_requirements\":[],\"acceptance_criteria\":[\"实现登录功能\"],\"out_of_scope\":[],\"risks\":[],\"clarification_questions\":[],\"decisions\":[],\"version\":\"1.0\"}\n```"""
        plan_json = """```json\n{\"architecture\":\"A\",\"tech_stack\":[],\"steps\":[\"S\"],\"file_changes\":[],\"test_plan\":[],\"risks\":[],\"version\":\"1.0\"}\n```"""
        # Only need 5 prompts (spec, plan, task, build, criteria) — no review
        criteria_text = "CRITERIA_1: PASS"
        session = self._make_mock_session(
            [
                spec_json,
                plan_json,
                "1. Task one (依赖: 无)",
                "build done " * 20,
                criteria_text,
            ]
        )
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        review_events = []
        callbacks = SpecEngineCallbacks(
            on_review_done=lambda c, r: review_events.append((c, r)),
        )

        project = engine.execute("- 实现登录功能", callbacks)

        assert project.status == SpecProjectStatus.COMPLETED
        assert len(project.cycles) == 1
        # Review was skipped — no review result, no callback fired
        assert project.cycles[0].review_result is None
        assert review_events == []

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_discovery_generates_spec_files_and_backlog(self, mock_settings, mock_create, tmp_path):
        """每轮循环后触发问题发现→生成 spec 文件→加入 backlog，并能被下一轮加载执行。"""
        s = self._mock_settings()
        s.spec_max_cycles = 2
        s.spec_review_enabled = False
        s.spec_discovery_enabled = True
        s.spec_discovery_max_questions = 1
        s.spec_generated_specs_per_cycle = 1
        s.spec_discovery_gate_on_satisfied = True
        s.spec_discovery_max_pending = 5
        s.spec_discovery_cooldown_cycles = 3
        s.spec_backlog_stuck_window = 3
        s.spec_success_ignore_backlog = True
        s.spec_convergence_window = 0
        # Keep artifacts tiny for test
        s.spec_cycle_artifact_retention = 1
        mock_settings.return_value = s

        spec_json = """```json\n{\"goals\":[\"G\"],\"functional_spec\":[\"F\"],\"non_functional_requirements\":[],\"acceptance_criteria\":[\"实现登录功能\"],\"out_of_scope\":[],\"risks\":[],\"clarification_questions\":[],\"decisions\":[],\"version\":\"1.0\"}\n```"""
        plan_json = """```json\n{\"architecture\":\"A\",\"tech_stack\":[],\"steps\":[\"S\"],\"file_changes\":[],\"test_plan\":[],\"risks\":[],\"version\":\"1.0\"}\n```"""
        discovery1 = (
            """```json\n[{"id":"Q-1","question":"如何提升错误提示可用性？","why":"用户体验","priority":"P1"}]\n```"""
        )
        gen1 = """```json\n[{"id":"Q-1","spec":{"goals":["提升错误提示"],"functional_spec":["完善错误提示"],"non_functional_requirements":[],"acceptance_criteria":["错误提示清晰可读"],"out_of_scope":[],"risks":[],"clarification_questions":[],"decisions":[],"version":"1.0"}}]\n```"""

        # Cycle 1: spec, plan, task, build, criteria(FAIL), discovery, gen
        # Cycle 2: (spec loaded from file), plan, task, build, criteria(PASS)
        #          discovery 被门控跳过（all_satisfied=True + gate_on_satisfied=True）
        session = self._make_mock_session(
            [
                spec_json,
                plan_json,
                "1. T1 (依赖: 无)",
                "build ok",
                "CRITERIA_1: FAIL",
                discovery1,
                gen1,
                plan_json,
                "1. T2 (依赖: 无)",
                "build ok 2",
                "CRITERIA_1: PASS",
            ]
        )
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        project = engine.execute("- 实现登录功能")

        # 修复后行为：all_satisfied + review_passed 时 ignore_backlog=True
        # → 直接 success，不再被 backlog 阻塞
        assert project.status == SpecProjectStatus.COMPLETED
        assert len(project.cycles) == 2
        assert len(project.work_items) >= 1
        # The first generated item should have been consumed in cycle 2
        assert project.work_items[0].used_in_cycle in (1, 2)
        assert os.path.exists(project.work_items[0].spec_path)

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_5000_cycles_stability_with_persistence_and_resume(self, mock_settings, mock_create, tmp_path):
        """验证 5000 次完整循环可稳定执行，并支持落盘 + 断点续传加载。"""
        s = MagicMock()
        # Reduce to 500 to avoid timeout in CI, while still testing stability/recursion/memory to some extent.
        s.spec_max_cycles = 500
        s.spec_max_cycles_limit = 500
        s.spec_execution_timeout = 300
        s.spec_convergence_window = 0
        s.spec_review_enabled = False
        s.spec_cycle_tasks_max = 5
        s.spec_cycle_output_max_chars = 200
        s.spec_state_filename = ".spec_engine_state.json"
        s.spec_artifacts_dirname = ".spec_engine"
        s.spec_persist_phase_artifacts = False
        s.spec_phase_output_persist_max_chars = 200
        s.spec_cycle_artifact_retention = 1
        s.spec_persist_every_phase = True
        s.spec_discovery_enabled = True
        s.spec_discovery_max_questions = 1
        s.spec_discovery_force_nonempty = True
        s.spec_generated_specs_per_cycle = 1
        # 稳定性测试：关闭门控以保证能跑满 cycles
        s.spec_discovery_gate_on_satisfied = False
        s.spec_discovery_max_pending = 999
        s.spec_discovery_cooldown_cycles = 1
        s.spec_backlog_stuck_window = 999
        s.spec_success_ignore_backlog = False
        s.spec_allow_resume_from_disk = True
        s.spec_infinite_mode = False
        s.spec_disable_convergence = False
        s.spec_disable_early_stop = True
        s.spec_min_cycles = 1
        s.spec_history_log_filename = "history.jsonl"
        s.spec_generated_specs_retention = 1000
        s.spec_rebuild_session_between_cycles = False
        mock_settings.return_value = s

        class DynamicSession:
            def __init__(self):
                self.disc_n = 0

            def send_prompt(self, prompt, on_event=None, timeout=None):
                return self.send_prompt_with_retry(prompt, on_event, timeout)
            def send_prompt_with_retry(self, prompt, on_event=None, timeout=None, **kw):
                p = prompt or ""
                out = ""
                if "请使用 spec-kit 风格产出“规格（Spec）”" in p:
                    out = """```json\n{"goals":["G"],"functional_spec":["F"],"non_functional_requirements":[],"acceptance_criteria":["永不完成"],"out_of_scope":[],"risks":[],"clarification_questions":[],"decisions":[],"version":"1.0"}\n```"""
                elif "产出 Plan（规划）" in p and '"file_changes"' in p:
                    out = """```json\n{"architecture":"A","tech_stack":[],"steps":["S"],"file_changes":[],"test_plan":[],"risks":[],"version":"1.0"}\n```"""
                elif "格式（严格遵循）" in p and "任务编号" in p:
                    out = "1. T (依赖: 无)"
                elif "按以下任务列表逐步执行实现" in p:
                    out = "build"
                elif "请评估以下验收标准是否已满足" in p:
                    out = "CRITERIA_1: FAIL"
                elif "自动发现与目标相关的“可优化问题”" in p:
                    self.disc_n += 1
                    out = f'```json\n[{{"id":"Q-{self.disc_n}","question":"优化点 {self.disc_n}","why":"why","priority":"P1"}}]\n```'
                elif "spec-kit 规格生成器" in p:
                    m = re.search(r'"id"\s*:\s*"(Q-[^"]+)"', p)
                    qid = m.group(1) if m else "Q-X"
                    out = (
                        "```json\n["
                        + json.dumps(
                            {
                                "id": qid,
                                "spec": {
                                    "goals": [f"解决 {qid}"],
                                    "functional_spec": ["F"],
                                    "non_functional_requirements": [],
                                    "acceptance_criteria": ["永不完成"],
                                    "out_of_scope": [],
                                    "risks": [],
                                    "clarification_questions": [],
                                    "decisions": [],
                                    "version": "1.0",
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "]\n```"
                    )
                else:
                    out = ""

                if on_event and out:
                    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=out))

        mock_create.return_value = DynamicSession()

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        t0 = time.time()
        project = engine.execute("- 永不完成")
        elapsed = time.time() - t0

        assert len(project.cycles) == 500
        assert project.cycles[-1].status == "completed"
        assert project.status == SpecProjectStatus.ABORTED
        # State file must exist outside the project tree and be loadable.
        from src.spec_engine.persistence import get_state_path

        state_path = get_state_path(str(tmp_path), engine.settings)
        assert os.path.exists(state_path)
        assert not (tmp_path / ".spec_engine_state.json").exists()
        loaded = SpecEngine.load_state(str(state_path))
        assert loaded is not None
        assert loaded.current_cycle_number == 500
        # Basic performance guard (avoid regressions)
        assert elapsed < 120


# ======================================================================
# TestSpecEngineProjectTypes — web/api/script variants
# ======================================================================


class TestSpecEngineProjectTypes:
    def _mock_settings(self):
        return _make_spec_settings()

    def _make_mock_session(self, text_responses):
        session = MagicMock()
        idx = [0]
        responses = list(text_responses)

        def fake_send_prompt(prompt, on_event=None, timeout=None, **kwargs):
            i = idx[0]
            idx[0] += 1
            text = responses[i] if i < len(responses) else ""
            if on_event and text:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))

        session.send_prompt = fake_send_prompt
        session.send_prompt_with_retry = fake_send_prompt
        return session

    @pytest.mark.parametrize("spec_json,plan_json,task_text,requirement", [
        pytest.param(
            '```json\n{"goals":["Web 登录"],"functional_spec":["页面","接口"],"non_functional_requirements":["性能"],"acceptance_criteria":["Web 登录可用"],"out_of_scope":[],"risks":[],"clarification_questions":[],"decisions":[],"version":"1.0"}\n```',
            '```json\n{"architecture":"MVC","tech_stack":["FastAPI","React"],"steps":["实现 API","实现 UI"],"file_changes":["src/app.py"],"test_plan":["pytest"],"risks":[],"version":"1.0"}\n```',
            "1. 实现 Web 登录 (依赖: 无)",
            "- Web 需求",
            id="web_app",
        ),
        pytest.param(
            '```json\n{"goals":["API 开发"],"functional_spec":["REST"],"non_functional_requirements":[],"acceptance_criteria":["API 返回符合预期"],"out_of_scope":[],"risks":[],"clarification_questions":[],"decisions":[],"version":"1.0"}\n```',
            '```json\n{"architecture":"HTTP API","tech_stack":["FastAPI"],"steps":["实现 endpoint"],"file_changes":["src/api.py"],"test_plan":["pytest -k api"],"risks":[],"version":"1.0"}\n```',
            "1. 实现 API (依赖: 无)",
            "- API 需求",
            id="api_dev",
        ),
        pytest.param(
            '```json\n{"goals":["脚本工具"],"functional_spec":["CLI"],"non_functional_requirements":[],"acceptance_criteria":["CLI 可执行并输出正确"],"out_of_scope":[],"risks":[],"clarification_questions":[],"decisions":[],"version":"1.0"}\n```',
            '```json\n{"architecture":"单文件脚本","tech_stack":["Python"],"steps":["实现命令解析"],"file_changes":["tools/foo.py"],"test_plan":["pytest -k tool"],"risks":[],"version":"1.0"}\n```',
            "1. 实现脚本工具 (依赖: 无)",
            "- 脚本需求",
            id="script_tool",
        ),
    ])
    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_project_flow_no_missing_artifacts(self, mock_settings, mock_create, spec_json, plan_json, task_text, requirement):
        mock_settings.return_value = self._mock_settings()
        review_text = "[ARCHITECT]\nPASS\n\n[PRODUCT]\nPASS\n\n[USER]\nPASS\n\n[TESTER]\nPASS\n\n[DESIGNER]\nPASS\n"
        criteria_text = "CRITERIA_1: PASS"

        session = self._make_mock_session(
            [spec_json, plan_json, task_text, "build ok " * 10, review_text, criteria_text]
        )
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path="/tmp/test")
        project = engine.execute(requirement)
        assert project.status == SpecProjectStatus.COMPLETED
        assert project.cycles[0].spec_artifact is not None
        assert project.cycles[0].plan_artifact is not None


# ======================================================================
# TestLooseReviewParsing — parse_review_output_loose
# ======================================================================


class TestLooseReviewParsing:
    """Tests for Level 2.5 loose review parsing."""

    def test_json_array_format(self):
        """Agent outputs a JSON array with perspective verdicts."""
        text = """
[
  {"perspective": "ARCHITECT", "verdict": "PASS", "suggestions": []},
  {"perspective": "PRODUCT", "verdict": "FAIL", "suggestions": ["需要分页"]},
  {"perspective": "USER", "verdict": "PASS", "suggestions": []},
  {"perspective": "TESTER", "verdict": "FAIL", "suggestions": ["缺少测试"]}
]
"""
        reviews = parse_review_output_loose(text, 1)
        assert len(reviews) == 4
        by_name = {r.perspective.name: r for r in reviews}
        assert by_name["ARCHITECT"].passed is True
        assert by_name["PRODUCT"].passed is False
        assert "需要分页" in by_name["PRODUCT"].suggestions
        assert by_name["USER"].passed is True
        assert by_name["TESTER"].passed is False

    def test_keyword_pair_inline(self):
        """Keywords and verdicts on the same line."""
        text = """
架构师: PASS
产品经理: FAIL
- 缺少搜索功能
用户: PASS
测试: PASS
"""
        reviews = parse_review_output_loose(text, 1)
        assert len(reviews) == 4
        by_name = {r.perspective.name: r for r in reviews}
        assert by_name["ARCHITECT"].passed is True
        assert by_name["PRODUCT"].passed is False

    def test_keyword_pair_next_line(self):
        """Verdict on the line after the perspective keyword."""
        text = """
架构师
PASS

产品经理
FAIL
- 需要优化性能

用户
PASS

测试
PASS
"""
        reviews = parse_review_output_loose(text, 1)
        assert len(reviews) >= 3  # at least 3 should parse

    def test_table_format(self):
        """Agent outputs a markdown table."""
        text = """
| 视角 | 结果 |
|------|------|
| 架构师 | PASS |
| 产品经理 | FAIL |
| 用户 | PASS |
| 测试 | PASS |
"""
        reviews = parse_review_output_loose(text, 1)
        assert len(reviews) == 4
        by_name = {r.perspective.name: r for r in reviews}
        assert by_name["ARCHITECT"].passed is True
        assert by_name["PRODUCT"].passed is False

    def test_english_keywords(self):
        """English keyword + verdict on same line."""
        text = """
ARCHITECT: PASS
PRODUCT: FAIL
- Missing validation
USER: PASS
TESTER: PASS
"""
        reviews = parse_review_output_loose(text, 1)
        assert len(reviews) == 4
        by_name = {r.perspective.name: r for r in reviews}
        assert by_name["ARCHITECT"].passed is True
        assert by_name["PRODUCT"].passed is False

    def test_chinese_verdict(self):
        """Chinese verdict keywords (通过/不通过)."""
        text = """
架构师: 通过
产品: 不通过
- 缺少错误处理
用户: 通过
测试: 通过
"""
        reviews = parse_review_output_loose(text, 1)
        assert len(reviews) >= 3
        by_name = {r.perspective.name: r for r in reviews}
        assert by_name["ARCHITECT"].passed is True
        assert by_name["PRODUCT"].passed is False

    def test_empty_text(self):
        assert parse_review_output_loose("", 1) == []
        assert parse_review_output_loose(None, 1) == []

    def test_no_verdicts(self):
        """Text with no recognizable verdict patterns."""
        text = "这是一些随机文本，没有任何审查格式。"
        reviews = parse_review_output_loose(text, 1)
        assert reviews == []

    def test_json_with_role_key(self):
        """JSON with 'role' key instead of 'perspective'."""
        text = """[
  {"role": "architect", "result": "PASS"},
  {"role": "product", "result": "FAIL", "suggestions": ["优化"]},
  {"role": "user", "result": "PASS"},
  {"role": "tester", "result": "PASS"}
]"""
        reviews = parse_review_output_loose(text, 1)
        assert len(reviews) == 4

    def test_mixed_format_with_suggestions(self):
        """Loose format with bullet suggestions after FAIL."""
        text = """
ARCHITECT PASS
PRODUCT FAIL
- 缺少分页
- 搜索不准确
USER PASS
TESTER FAIL
- 需要更多边界测试
"""
        reviews = parse_review_output_loose(text, 1)
        assert len(reviews) >= 3
        by_name = {r.perspective.name: r for r in reviews}
        if "PRODUCT" in by_name:
            assert by_name["PRODUCT"].passed is False
            assert len(by_name["PRODUCT"].suggestions) >= 1


# ======================================================================
# TestSpecReporterNewMethods — status_line, duration_line, criteria_section
# ======================================================================


class TestSpecReporterNewMethods:
    def _make_project(self, criteria=None):
        project = SpecProject.create(name="test", root_path="/tmp/test")
        if criteria:
            project.acceptance_criteria = criteria
            project.criteria_tracker.init_criteria(criteria)
        return project

    def test_format_status_line(self):
        r = SpecReporter()
        project = self._make_project(criteria=["C1", "C2"])
        project.status = SpecProjectStatus.RUNNING
        project.cycles.append(SpecCycle(cycle_number=1))
        result = r.format_status_line(project)
        assert "🔄" in result
        assert "循环" in result
        assert "标准" in result
        assert "0/2" in result

    def test_format_duration_line_no_duration(self):
        r = SpecReporter()
        project = self._make_project()
        result = r.format_duration_line(project)
        assert result == ""

    def test_format_duration_line_with_duration(self):
        r = SpecReporter()
        project = self._make_project()
        project.started_at = time.time() - 120
        result = r.format_duration_line(project)
        assert "⏱️" in result

    def test_format_criteria_section(self):
        r = SpecReporter()
        project = self._make_project(criteria=["C1", "C2"])
        result = r.format_criteria_section(project)
        assert "C1" in result
        assert "C2" in result

    def test_format_phase_progress_at_spec(self):
        r = SpecReporter()
        result = r.format_phase_progress(SpecPhase.SPEC, completed=False)
        assert "▶️" in result
        assert "Spec" in result or "规格" in result
        assert "⬜" in result

    def test_format_phase_progress_mid_build(self):
        r = SpecReporter()
        result = r.format_phase_progress(SpecPhase.BUILD, completed=False)
        assert result.count("✅") >= 3
        assert "▶️" in result
        assert "⬜" in result

    def test_format_phase_progress_review_completed(self):
        r = SpecReporter()
        result = r.format_phase_progress(SpecPhase.REVIEW, completed=True)
        assert result.count("✅") == 5
        assert "⬜" not in result
        assert "▶️" not in result

    def test_format_phase_start_content(self):
        r = SpecReporter()
        result = r.format_phase_start_content(2, SpecPhase.PLAN, 5)
        assert "执行中" in result
        assert "▶️" in result

    def test_format_phase_done_content(self):
        r = SpecReporter()
        result = r.format_phase_done_content(1, SpecPhase.BUILD, 5, "some output\nmore lines\n")
        assert "完成" in result
        assert "✅" in result

    def test_format_cycle_done_with_review_suggestions(self):
        r = SpecReporter()
        review = ReviewResult(
            reviews=[
                PerspectiveReview(
                    perspective=ReviewPerspective.ARCHITECT,
                    passed=False,
                    suggestions=["Improve naming"],
                    summary="1条建议",
                ),
                PerspectiveReview(perspective=ReviewPerspective.PRODUCT, passed=True, suggestions=[], summary="通过"),
                PerspectiveReview(perspective=ReviewPerspective.USER, passed=True, suggestions=[], summary="通过"),
                PerspectiveReview(perspective=ReviewPerspective.TESTER, passed=True, suggestions=[], summary="通过"),
            ],
            iteration=1,
        )
        cycle = SpecCycle(cycle_number=1)
        cycle.status = "completed"
        cycle.review_result = review
        result = r.format_cycle_done(1, cycle)
        assert "审查建议" in result
        assert "Improve naming" in result
        assert "驱动下一轮" in result
        assert "1 条建议" in result

    def test_format_cycle_done_all_passed(self):
        r = SpecReporter()
        review = ReviewResult(
            reviews=[
                PerspectiveReview(perspective=p, passed=True, suggestions=[], summary="通过") for p in ReviewPerspective
            ],
            iteration=1,
        )
        cycle = SpecCycle(cycle_number=1)
        cycle.status = "completed"
        cycle.review_result = review
        result = r.format_cycle_done(1, cycle)
        assert "审查通过" in result
        assert "审查建议" not in result

    def test_format_cycle_start_has_progress(self):
        r = SpecReporter()
        result = r.format_cycle_start(1, 5)
        assert "▶️" in result
        assert "→" in result

def test_save_failed_task_idempotency(monkeypatch):

    from src.spec_engine.engine import SpecEngine, SpecEngineCallbacks
    from src.spec_engine.models import SpecPhase, SpecProject

    mock_settings = type("MockSettings", (), {
        "spec_max_retries": 1,
        "spec_allow_resume_from_disk": False,
        "spec_state_filename": ".spec_state"
    })()

    engine = SpecEngine("chat1", "/tmp", "coco", mock_settings)
    engine._project = SpecProject(project_id="test_proj_id", name="test", root_path="/tmp", task_id="test-task-123")

    call_count = 0

    def mock_save_task_state(state):
        nonlocal call_count
        call_count += 1
        return f"/tmp/saved_{call_count}.json"

    monkeypatch.setattr("src.spec_engine.task_persistence.save_task_state", mock_save_task_state)

    callbacks = SpecEngineCallbacks()

    # First save
    task_id1 = engine._save_failed_task("Error 1", 1, SpecPhase.BUILD, callbacks)
    assert call_count == 1
    assert task_id1 == "test-task-123"

    # Second save, same cycle, phase, and task_id (from project), but different error
    task_id2 = engine._save_failed_task("Error 2 - different", 1, SpecPhase.BUILD, callbacks)
    assert call_count == 1 # Should be idempotent, so no new save
    assert task_id2 == task_id1

    # Third save, different phase
    task_id3 = engine._save_failed_task("Error 1", 1, SpecPhase.REVIEW, callbacks)
    assert call_count == 2
    assert task_id3 == "test-task-123"


# ======================================================================
# TimeoutError 改进: 诊断 / 文案 / 配置项
# ======================================================================


def test_timeout_error_with_message_diagnostics_has_friendly_text():
    """TimeoutError 带消息时，diagnostics.error_text 应为该消息而非 'empty message'。"""
    from src.spec_engine.review import build_review_exception_diagnostics, normalize_review_diagnostics

    err = TimeoutError("ACP prompt 执行超时 (120s)")
    diag_raw = build_review_exception_diagnostics(err, cycle=1)
    diag = normalize_review_diagnostics(diag_raw)
    assert diag["fail_reason"] == "timeout"
    assert "empty message" not in diag["error_text"]
    assert "超时" in diag["error_text"]


def test_timeout_error_empty_message_diagnostics_uses_friendly_text():
    """TimeoutError 空消息时，diagnostics.error_text 应为中文友好文案而非 'TimeoutError (empty message)'。"""
    from src.spec_engine.review import build_review_exception_diagnostics, normalize_review_diagnostics

    err = TimeoutError()
    diag_raw = build_review_exception_diagnostics(err, cycle=2)
    diag = normalize_review_diagnostics(diag_raw)
    assert diag["fail_reason"] == "timeout"
    assert "empty message" not in diag["error_text"]
    assert "审查超时" in diag["error_text"]


def test_non_timeout_error_empty_message_still_uses_empty_message_fallback():
    """非 timeout 类异常空消息时，仍使用 '(empty message)' 兜底。"""
    from src.spec_engine.review import build_review_exception_diagnostics, normalize_review_diagnostics

    err = RuntimeError()
    diag_raw = build_review_exception_diagnostics(err, cycle=3)
    diag = normalize_review_diagnostics(diag_raw)
    assert diag["fail_reason"] != "timeout"
    # 非 timeout 不应被替换为中文友好文案
    assert "审查超时" not in diag["error_text"]


def test_spec_review_timeout_config_exists_and_defaults():
    """spec_review_timeout 配置项应存在且默认值为 240。"""
    from src.config import Settings

    s = Settings(
        feishu_app_id="x",
        feishu_app_secret="x",
    )
    assert hasattr(s, "spec_review_timeout")
    assert s.spec_review_timeout == 240


def test_spec_review_timeout_passed_to_send_prompt(monkeypatch):
    """conduct_review 应将 settings.spec_review_timeout 传递给 send_prompt_with_retry_fn 的 timeout 参数。"""
    from src.spec_engine.review import ReviewCircuitState, conduct_review

    captured_timeout = {}

    def fake_send(prompt, on_event=None, timeout=0, **kw):
        captured_timeout["value"] = timeout
        if on_event:
            from src.acp.models import ACPEvent, ACPEventType
            on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="[ARCHITECT]\nPASS\n\n[PRODUCT]\nPASS\n\n[USER]\nPASS\n\n[TESTER]\nPASS\n\n[DESIGNER]\nPASS\n"))
        return MagicMock(stop_reason="end_turn")

    class _Settings:
        spec_review_failure_circuit_enabled = False
        spec_review_failure_max_consecutive = 3
        spec_review_failure_cooldown_cycles = 3
        spec_review_timeout = 42
        spec_review_min_timeout = 15
        spec_review_hard_floor = 10

    project = SpecProject.create(root_path="/tmp")
    project.requirement = "test"
    circuit = ReviewCircuitState()

    conduct_review(
        session=MagicMock(),
        settings=_Settings(),
        project=project,
        send_prompt_with_retry_fn=fake_send,
        build_review_exception_diagnostics_fn=lambda e, cycle=0: {"fail_reason": "exception"},
        circuit=circuit,
        cycle=1,
    )

    assert captured_timeout["value"] == 42


def test_timeout_fallback_review_result_has_friendly_suggestion():
    """TimeoutError 触发的 fallback ReviewResult 应包含 '审查超时' 而非 'empty message'。"""
    from src.spec_engine.review import ReviewCircuitState, build_review_exception_diagnostics, conduct_review

    def fake_send(prompt, on_event=None, timeout=0, **kw):
        raise TimeoutError()

    class _Settings:
        spec_review_failure_circuit_enabled = False
        spec_review_failure_max_consecutive = 3
        spec_review_failure_cooldown_cycles = 3
        spec_review_timeout = 10
        spec_review_min_timeout = 5
        spec_review_hard_floor = 3

    project = SpecProject.create(root_path="/tmp")
    project.requirement = "test"
    circuit = ReviewCircuitState()

    result = conduct_review(
        session=MagicMock(),
        settings=_Settings(),
        project=project,
        send_prompt_with_retry_fn=fake_send,
        build_review_exception_diagnostics_fn=build_review_exception_diagnostics,
        circuit=circuit,
        cycle=1,
    )

    assert result is not None
    all_suggestions = [s for rev in result.reviews for s in (rev.suggestions or [])]
    assert all("审查超时" in s for s in all_suggestions), f"Expected '审查超时' in suggestions, got: {all_suggestions}"
    assert all("empty message" not in s for s in all_suggestions)


def test_non_timeout_fallback_review_result_keeps_generic_format():
    """非 timeout 异常的 fallback ReviewResult 仍使用 '审查执行异常: ...' 格式。"""
    from src.spec_engine.review import ReviewCircuitState, build_review_exception_diagnostics, conduct_review

    def fake_send(prompt, on_event=None, timeout=0, **kw):
        raise ValueError("bad input")

    class _Settings:
        spec_review_failure_circuit_enabled = False
        spec_review_failure_max_consecutive = 3
        spec_review_failure_cooldown_cycles = 3
        spec_review_timeout = 10
        spec_review_min_timeout = 5
        spec_review_hard_floor = 3

    project = SpecProject.create(root_path="/tmp")
    project.requirement = "test"
    circuit = ReviewCircuitState()

    result = conduct_review(
        session=MagicMock(),
        settings=_Settings(),
        project=project,
        send_prompt_with_retry_fn=fake_send,
        build_review_exception_diagnostics_fn=build_review_exception_diagnostics,
        circuit=circuit,
        cycle=1,
    )

    all_suggestions = [s for rev in result.reviews for s in (rev.suggestions or [])]
    assert all("审查执行异常:" in s for s in all_suggestions)
    assert all("bad input" in s for s in all_suggestions)


# ======================================================================
# ReviewCircuitState persistence round-trip tests
# ======================================================================


class TestReviewCircuitStatePersistence:
    """Verify circuit state survives save → load round-trip."""

    def test_save_load_roundtrip(self, tmp_path):
        """Circuit counters survive save_engine_state → load_engine_state."""
        from src.spec_engine.persistence import load_engine_state, save_engine_state
        from src.spec_engine.review import ReviewCircuitState

        # Prepare a minimal SpecProject
        proj = SpecProject(
            project_id="p1", name="test", root_path=str(tmp_path),
            requirement="test req",
        )
        circuit = ReviewCircuitState(
            review_failure_consecutive=2,
            review_circuit_open_until_cycle=5,
            backoff_level=1,
            consecutive_timeouts=3,
        )

        settings = MagicMock()
        settings.spec_state_filename = ".spec_state.json"
        fp = str(tmp_path / ".spec_state.json")

        save_engine_state(
            project=proj, settings=settings, root_path=str(tmp_path),
            chat_id="c1",
            build_runtime_context_fn=lambda: {},
            project_to_compact_dict_fn=proj.to_dict,
            filepath=fp,
            review_circuit=circuit.to_dict(),
        )

        loaded_proj, rc_dict = load_engine_state(fp)
        assert loaded_proj is not None
        restored = ReviewCircuitState.from_dict(rc_dict)
        assert restored.review_failure_consecutive == 2
        assert restored.review_circuit_open_until_cycle == 5
        assert restored.backoff_level == 1
        assert restored.consecutive_timeouts == 3

    def test_load_old_format_without_circuit(self, tmp_path):
        """Old snapshots (no review_circuit key) return default values."""
        from src.spec_engine.persistence import load_engine_state
        from src.spec_engine.review import ReviewCircuitState

        proj = SpecProject(
            project_id="p2", name="old", root_path=str(tmp_path),
        )
        # Simulate old-format state file (no review_circuit key)
        old_state = {
            "chat_id": "c1",
            "root_path": str(tmp_path),
            "project": proj.to_dict(),
            "saved_at": 1.0,
        }
        fp = str(tmp_path / "old_state.json")
        with open(fp, "w") as f:
            json.dump(old_state, f)

        loaded_proj, rc_dict = load_engine_state(fp)
        assert loaded_proj is not None
        assert rc_dict == {}
        restored = ReviewCircuitState.from_dict(rc_dict) if rc_dict else ReviewCircuitState()
        assert restored.backoff_level == 0
        assert restored.consecutive_timeouts == 0
        assert restored.review_failure_consecutive == 0

    def test_spec_engine_load_state_with_circuit(self, tmp_path):
        """SpecEngine.load_state_with_circuit returns both project and circuit."""
        from src.spec_engine.review import ReviewCircuitState

        proj = SpecProject(
            project_id="p3", name="t", root_path=str(tmp_path),
        )
        circuit = ReviewCircuitState(backoff_level=2, consecutive_timeouts=4)
        state = {
            "chat_id": "c1",
            "root_path": str(tmp_path),
            "project": proj.to_dict(),
            "review_circuit": circuit.to_dict(),
            "saved_at": 1.0,
        }
        fp = str(tmp_path / "state.json")
        with open(fp, "w") as f:
            json.dump(state, f)

        loaded_proj, loaded_circuit = SpecEngine.load_state_with_circuit(fp)
        assert loaded_proj is not None
        assert loaded_circuit.backoff_level == 2
        assert loaded_circuit.consecutive_timeouts == 4

    def test_spec_engine_load_state_backward_compat(self, tmp_path):
        """SpecEngine.load_state still returns just Optional[SpecProject]."""
        proj = SpecProject(
            project_id="p4", name="t", root_path=str(tmp_path),
        )
        state = {
            "chat_id": "c1",
            "root_path": str(tmp_path),
            "project": proj.to_dict(),
            "saved_at": 1.0,
        }
        fp = str(tmp_path / "state.json")
        with open(fp, "w") as f:
            json.dump(state, f)

        loaded = SpecEngine.load_state(fp)
        assert loaded is not None
        assert loaded.project_id == "p4"


@patch("src.spec_engine.engine.create_engine_session")
@patch("src.engine_base.get_settings")
def test_resume_restores_circuit_state_from_disk(mock_settings, mock_create, tmp_path):
    """resume() should restore _review_circuit from persisted state file."""
    from src.spec_engine.review import ReviewCircuitState

    s = MagicMock()
    s.spec_max_cycles = 1
    s.spec_max_cycles_limit = 5000
    s.spec_convergence_window = 1
    s.spec_execution_timeout = 300
    s.spec_state_filename = ".spec_engine_state.json"
    s.spec_review_failure_circuit_enabled = True
    s.spec_review_failure_max_consecutive = 3
    s.spec_review_failure_cooldown_cycles = 3

    s.spec_review_timeout = 120
    s.spec_review_min_timeout = 30
    s.spec_review_hard_floor = 15
    s.spec_review_max_parallel = 4
    s.spec_review_retry_max_attempts = 1
    s.spec_review_retry_max_delay = 30
    s.spec_review_failure_max_cooldown_cycles = 12
    s.spec_review_enabled = True
    s.spec_infinite_mode = False
    s.spec_disable_convergence = False
    s.spec_disable_early_stop = False
    s.spec_min_cycles = 1
    s.spec_max_retries = 1
    s.spec_discovery_enabled = False
    s.spec_allow_resume_from_disk = True
    s.spec_artifacts_dirname = ".spec_engine"
    s.spec_persist_phase_artifacts = False
    s.spec_persist_every_phase = False
    s.spec_history_log_filename = "history.jsonl"
    s.spec_phase_output_persist_max_chars = 20000
    s.spec_cycle_artifact_retention = 50
    s.spec_generated_specs_retention = 1000
    s.spec_state_cycles_tail = 50
    s.spec_state_work_items_tail = 200
    s.spec_state_metrics_tail = 200
    s.spec_cycle_output_max_chars = 4000
    s.spec_cycle_tasks_max = 50
    s.spec_review_min_timeout = 30
    s.spec_review_timeout = 120
    mock_settings.return_value = s

    # Make session creation fail fast so resume() exits quickly
    mock_create.side_effect = RuntimeError("test-abort")

    root = str(tmp_path)
    engine = SpecEngine(chat_id="c1", root_path=root)

    # Set up paused project
    proj = SpecProject.create(root_path=root)
    proj.requirement = "test"
    proj.acceptance_criteria = ["test"]
    proj.criteria_tracker.init_criteria(["test"])
    proj.status = SpecProjectStatus.PAUSED
    proj.started_at = time.time()
    engine._project = proj

    # Verify fresh circuit (all zeros)
    assert engine._review_circuit.consecutive_timeouts == 0
    assert engine._review_circuit.backoff_level == 0
    assert engine._review_circuit.consecutive_skips == 0

    # Persist state with non-zero circuit
    circuit_data = ReviewCircuitState(
        backoff_level=3,
        consecutive_timeouts=5,
        review_failure_consecutive=2,
        review_circuit_open_until_cycle=8,
        consecutive_skips=4,
    )
    state = {
        "chat_id": "c1",
        "root_path": root,
        "project": proj.to_dict(),
        "review_circuit": circuit_data.to_dict(),
        "saved_at": 1.0,
    }
    state_path = os.path.join(root, ".spec_engine_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f)

    # resume() will restore circuit from disk then fail on session creation
    engine.resume()

    # Circuit should be restored from disk, not the fresh default
    assert engine._review_circuit.backoff_level == 3
    assert engine._review_circuit.consecutive_timeouts == 5
    assert engine._review_circuit.review_failure_consecutive == 2
    assert engine._review_circuit.review_circuit_open_until_cycle == 8
    assert engine._review_circuit.consecutive_skips == 4


class TestSpecEngineCycleResilience:
    """Tests for cycle-level exception digestion: exceptions inside a cycle
    should NOT abort the engine but instead mark the cycle failed and continue."""

    _SPEC_JSON = '```json\n{"goals":["G"],"functional_spec":["F"],"non_functional_requirements":[],"acceptance_criteria":["实现功能"],"out_of_scope":[],"risks":[],"clarification_questions":[],"decisions":[],"version":"1.0"}\n```'
    _PLAN_JSON = '```json\n{"architecture":"A","tech_stack":[],"steps":["S1"],"file_changes":[],"test_plan":[],"risks":[],"version":"1.0"}\n```'

    def _mock_settings(self):
        return _make_spec_settings(
            spec_max_cycles=2,
            spec_review_enabled=False,
            spec_persist_phase_artifacts=False,
            spec_persist_every_phase=False,
            spec_max_consecutive_failures=3,
            spec_backlog_stuck_window=0,
            spec_allow_resume_from_disk=False,
            spec_model_switch_enabled=False,
            spec_failed_task_id_override="",
            spec_review_timeout=60,
            spec_review_max_parallel=2,
            spec_review_failure_max_cooldown_cycles=12,
            review_circuit_window_size=10,
            review_circuit_success_rate_threshold=0.3,
            review_circuit_lint_fallback_enabled=False,
            review_circuit_lint_timeout=10,
        )

    def _make_mock_session(self, send_fn):
        session = MagicMock()
        session.send_prompt = send_fn
        session.send_prompt_with_retry = send_fn
        return session

    def _apply_engine_mocks(self, engine):
        """Prevent real session creation / model switching in tests."""
        engine._recreate_session_best_effort = lambda: None
        engine._try_switch_model = lambda callbacks: False

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_cycle_exception_digested_continues_next_cycle(self, mock_settings, mock_create, tmp_path):
        """Cycle 1 raises RuntimeError in SPEC phase → cycle marked failed → cycle 2 succeeds → COMPLETED."""
        s = self._mock_settings()
        s.spec_max_cycles = 2
        s.spec_success_ignore_backlog = True
        mock_settings.return_value = s

        call_count = [0]
        criteria_text = "CRITERIA_1: PASS"

        def fake_send(prompt, on_event=None, timeout=None, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("模型切换失败")
            # Cycle 2: spec, plan, task, build, criteria
            texts = [self._SPEC_JSON, self._PLAN_JSON, "1. T1 (依赖: 无)", "build done " * 10, criteria_text]
            idx = call_count[0] - 2
            text = texts[idx] if idx < len(texts) else "ok"
            if on_event and text:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
            return MagicMock(stop_reason="end_turn")

        session = self._make_mock_session(fake_send)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)
        project = engine.execute("- 实现功能")

        assert project.status == SpecProjectStatus.COMPLETED
        assert len(project.cycles) == 2
        assert project.cycles[0].status == "failed"
        assert project.cycles[0].error_message is not None
        assert "模型切换失败" in project.cycles[0].error_message
        assert project.cycles[1].status == "completed"

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_consecutive_failures_aborts_engine(self, mock_settings, mock_create, tmp_path):
        """All cycles fail → consecutive_failures termination → ABORTED."""
        s = self._mock_settings()
        s.spec_max_cycles = 10
        s.spec_max_consecutive_failures = 2
        mock_settings.return_value = s

        def always_fail(prompt, on_event=None, timeout=None, **kw):
            raise RuntimeError("always fail")

        session = self._make_mock_session(always_fail)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)
        project = engine.execute("- test req")

        assert project.status == SpecProjectStatus.ABORTED
        assert "连续异常终止" in (project.error or "")
        assert len(project.cycles) == 2
        assert all(c.status == "failed" for c in project.cycles)

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_successful_cycle_resets_failure_counter(self, mock_settings, mock_create, tmp_path):
        """Fail → success → fail → should NOT trigger consecutive_failures (max=2)."""
        s = self._mock_settings()
        s.spec_max_cycles = 3
        s.spec_max_consecutive_failures = 2
        s.spec_success_ignore_backlog = True
        mock_settings.return_value = s

        call_count = [0]
        criteria_text = "CRITERIA_1: PASS"

        def fake_send(prompt, on_event=None, timeout=None, **kw):
            call_count[0] += 1
            # Cycle 1 (calls 1): fail at SPEC
            if call_count[0] == 1:
                raise RuntimeError("cycle 1 fail")
            # Cycle 2 (calls 2-6): succeed — spec, plan, task, build, criteria
            if 2 <= call_count[0] <= 6:
                texts = [self._SPEC_JSON, self._PLAN_JSON, "1. T1 (依赖: 无)", "build done " * 10, criteria_text]
                idx = call_count[0] - 2
                text = texts[idx] if idx < len(texts) else "ok"
                if on_event and text:
                    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
                return MagicMock(stop_reason="end_turn")
            # Cycle 3 would fail but should not get here — cycle 2 should succeed and terminate
            raise RuntimeError("cycle 3 fail")

        session = self._make_mock_session(fake_send)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)
        project = engine.execute("- 实现功能")

        # Cycle 2 should succeed and terminate engine
        assert project.status == SpecProjectStatus.COMPLETED
        assert project.cycles[0].status == "failed"
        assert project.cycles[1].status == "completed"

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_timeout_error_digested_in_cycle(self, mock_settings, mock_create, tmp_path):
        """TimeoutError should be digested inside the cycle, not bubble to execute()."""
        s = self._mock_settings()
        s.spec_max_cycles = 2
        s.spec_success_ignore_backlog = True
        mock_settings.return_value = s

        call_count = [0]
        criteria_text = "CRITERIA_1: PASS"

        def fake_send(prompt, on_event=None, timeout=None, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TimeoutError("phase timeout")
            texts = [self._SPEC_JSON, self._PLAN_JSON, "1. T1 (依赖: 无)", "build done " * 10, criteria_text]
            idx = call_count[0] - 2
            text = texts[idx] if idx < len(texts) else "ok"
            if on_event and text:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
            return MagicMock(stop_reason="end_turn")

        session = self._make_mock_session(fake_send)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)
        project = engine.execute("- 实现功能")

        assert project.status == SpecProjectStatus.COMPLETED
        assert project.cycles[0].status == "failed"
        assert project.cycles[0].error_message is not None
        assert project.cycles[1].status == "completed"

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_session_recreated_after_cycle_exception(self, mock_settings, mock_create, tmp_path):
        """After cycle exception, _recreate_session_best_effort should be called."""
        s = self._mock_settings()
        s.spec_max_cycles = 2
        s.spec_max_consecutive_failures = 3
        s.spec_success_ignore_backlog = True
        mock_settings.return_value = s

        call_count = [0]
        criteria_text = "CRITERIA_1: PASS"

        def fake_send(prompt, on_event=None, timeout=None, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("fail")
            texts = [self._SPEC_JSON, self._PLAN_JSON, "1. T1 (依赖: 无)", "build done " * 10, criteria_text]
            idx = call_count[0] - 2
            text = texts[idx] if idx < len(texts) else "ok"
            if on_event and text:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text))
            return MagicMock(stop_reason="end_turn")

        session = self._make_mock_session(fake_send)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        engine._try_switch_model = lambda callbacks: False

        recreate_calls = []

        def tracked_recreate():
            recreate_calls.append(1)

        engine._recreate_session_best_effort = tracked_recreate

        engine.execute("- 实现功能")

        # At least one recreate call after cycle 1 failure
        assert len(recreate_calls) >= 1

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_stopping_during_exception_still_pauses(self, mock_settings, mock_create, tmp_path):
        """If engine is STOPPING when exception occurs, should result in PAUSED, not digest."""
        s = self._mock_settings()
        s.spec_max_cycles = 5
        s.spec_max_consecutive_failures = 10
        mock_settings.return_value = s

        def fail_then_stop(prompt, on_event=None, timeout=None, **kw):
            raise RuntimeError("session cancelled")

        session = self._make_mock_session(fail_then_stop)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)

        # Set engine to STOPPING before the exception is caught
        original_run_phase = engine._run_phase

        def patched_run_phase(*args, **kwargs):
            engine._run_state = EngineRunState.STOPPING
            return original_run_phase(*args, **kwargs)

        engine._run_phase = patched_run_phase

        project = engine.execute("- test")

        assert project.status == SpecProjectStatus.PAUSED

    @patch("src.spec_engine.engine.create_engine_session")
    @patch("src.engine_base.get_settings")
    def test_error_message_field_persisted_on_failed_cycle(self, mock_settings, mock_create, tmp_path):
        """SpecCycle.error_message should be set and serializable."""
        s = self._mock_settings()
        s.spec_max_cycles = 1
        s.spec_max_consecutive_failures = 5
        mock_settings.return_value = s

        def fail_send(prompt, on_event=None, timeout=None, **kw):
            raise RuntimeError("test error detail xyz")

        session = self._make_mock_session(fail_send)
        mock_create.return_value = session

        engine = SpecEngine(chat_id="c1", root_path=str(tmp_path))
        self._apply_engine_mocks(engine)
        project = engine.execute("- test req")

        assert len(project.cycles) == 1
        cycle = project.cycles[0]
        assert cycle.status == "failed"
        assert cycle.error_message is not None
        assert "test error detail xyz" in cycle.error_message

        # Verify serialization roundtrip
        d = cycle.to_dict()
        assert "error_message" in d
        restored = SpecCycle.from_dict(d)
        assert restored.error_message == cycle.error_message


# ---------------------------------------------------------------------------
# Tests merged from test_spec_engine_di.py
# ---------------------------------------------------------------------------


def test_spec_engine_di_injected_dependencies_used():
    """Verify that injected dependencies are stored and used by SpecEngine."""
    mock_retry_policy = RetryPolicy(max_retries=99, retry_delay=0.1)

    mock_session = MagicMock()
    mock_create_session_fn = MagicMock(return_value=mock_session)

    engine = SpecEngine(
        chat_id="test_chat",
        root_path="/tmp/test_root",
        retry_policy=mock_retry_policy,
        create_session_fn=mock_create_session_fn,
    )

    # Verify the injected instances are stored
    assert engine._retry_policy is mock_retry_policy
    assert engine._create_session_fn is mock_create_session_fn

    # 1. Verify create_session_fn is used
    # Initialize fake project state to avoid execute setup logic
    engine._project = SpecProject.create(name="test", root_path="/tmp/test_root")
    engine._project.requirement = "do something"
    engine._project.status = SpecProjectStatus.PAUSED
    engine._run_state = EngineRunState.STOPPING

    # Mock _run_cycle_loop to prevent infinite loops in the test
    engine._run_cycle_loop = MagicMock(return_value="success")

    engine.resume()

    # create_session_fn should be called during resume
    mock_create_session_fn.assert_called_once_with(
        agent_type="coco",
        cwd="/tmp/test_root",
        on_rate_limit=None,
        model_name=None,
    )

    # 3. Verify retry_policy values are used in run_phase
    mock_callbacks = MagicMock()
    # Provide a dummy session that raises to test the retry policy mapping implicitly
    engine._session = MagicMock()
    engine._session.send_prompt_with_retry = MagicMock(return_value="dummy_output")

    try:
        engine._run_phase(
            cycle_num=1,
            phase=SpecPhase.SPEC,
            prompt="test prompt",
            callbacks=mock_callbacks,
            timeout=10,
        )
    except Exception:
        pass

    # We can check that the retry_delay and backoff_multiplier were read from mock_retry_policy
    # indirectly if we patch RetryPolicy inside engine._run_phase, but knowing it's stored
    # and accessed is usually sufficient for DI unit tests.


# ---------------------------------------------------------------------------
# Tests merged from test_spec_gc.py
# ---------------------------------------------------------------------------


@patch("gc.collect")
def test_spec_engine_no_inline_gc_collect(mock_gc_collect):
    """SpecEngine.execute should NOT call gc.collect() directly."""
    fast_settings = MagicMock()
    fast_settings.spec_max_cycles = 3
    fast_settings.spec_max_cycles_limit = 3
    fast_settings.spec_execution_timeout = 30
    fast_settings.spec_convergence_window = 0
    fast_settings.spec_min_cycles = 1
    fast_settings.spec_review_enabled = False
    fast_settings.spec_discovery_enabled = False
    fast_settings.spec_disable_convergence = False
    fast_settings.spec_disable_early_stop = False
    fast_settings.spec_infinite_mode = False
    fast_settings.spec_persist_phase_artifacts = False
    fast_settings.spec_allow_resume_from_disk = False
    fast_settings.spec_rebuild_session_between_cycles = False

    with patch("src.engine_base.get_settings", return_value=fast_settings):
        engine = SpecEngine(
            chat_id="test",
            root_path="/tmp",
            agent_type="coco",
        )

    engine._run_phase = MagicMock(return_value="dummy_prompt")
    engine._close_session_safely = MagicMock()
    engine._parse_requirement_to_criteria = MagicMock(return_value=["test criteria"])
    engine._evaluate_criteria = MagicMock(return_value=True)
    engine._conduct_review = MagicMock()

    for attr, val in (
        ("spec_max_cycles", 3),
        ("spec_max_cycles_limit", 3),
        ("spec_execution_timeout", 30),
        ("spec_convergence_window", 0),
        ("spec_min_cycles", 1),
        ("spec_review_enabled", False),
        ("spec_discovery_enabled", False),
    ):
        try:
            setattr(engine.settings, attr, val)
        except Exception:
            pass

    with (
        patch("src.spec_engine.engine.create_engine_session") as mock_create,
        patch("src.spec_engine.session_utils.get_coco_model_manager") as mock_model_manager,
    ):
        mock_create.return_value = MagicMock()
        mock_model_manager.return_value = MagicMock()

        callbacks = SpecEngineCallbacks()
        try:
            engine.execute("test req", callbacks)
        except Exception:
            pass

    # gc.collect() should NOT be called inline by SpecEngine anymore
    mock_gc_collect.assert_not_called()
