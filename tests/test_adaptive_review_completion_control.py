"""Fail-closed contracts for Spec adaptive completion control."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.engine_base import EngineRunState, PerspectiveReview, ReviewPerspective
from src.spec_engine.adaptive_review import (
    AdaptiveReviewResult,
    parse_role_review_output,
    run_adaptive_role_review_pipeline,
)
from src.spec_engine.convergence import ContinuationPolicy
from src.spec_engine.engine import SpecEngine, SpecEngineCallbacks
from src.spec_engine.models import SpecCycle, SpecProject, SpecProjectStatus
from src.spec_engine.review import ReviewCircuitState
from src.spec_engine.review_agents import ReviewAgentBinding
from src.spec_engine.review_aggregation import RoleReviewOutcome
from src.spec_engine.review_artifacts import ReviewArtifacts
from src.spec_engine.review_roles import ReviewRoleSpec, completion_control_role
from src.spec_engine.review_strategy import AdaptiveRoleReviewStrategy, ReviewContext


def _run_completion_review(raw: str, *, format_retry_max_attempts: int = 0):
    role = completion_control_role()

    def factory(_role):
        def runner(prompt, on_event, timeout):
            return raw

        return runner

    return run_adaptive_role_review_pipeline(
        ReviewArtifacts(
            cycle_number=1,
            requirement="完成全部验收目标",
            cwd="/repo",
        ),
        [role],
        prompt_runner_factory=factory,
        max_parallel=1,
        timeout=5,
        completion_gate_enabled=True,
        format_retry_max_attempts=format_retry_max_attempts,
    )


def test_goal_not_met_without_suggestions_fails_closed():
    result = _run_completion_review(
        json.dumps(
            {
                "role_id": "completion_control",
                "verdict": "PASS",
                "goal_verdict": "GOAL_NOT_MET",
                "goal_confidence": "high",
                "evidence_summary": "验收目标仍未完成",
                "suggestions": [],
            }
        )
    )

    assert result.all_passed is False
    assert result.reviews[0].passed is False
    assert result.role_outcomes[0].passed is False
    assert result.role_outcomes[0].error == ""
    assert result.completion_gate_met is False


def test_explicit_completion_fail_without_suggestions_fails_closed():
    result = _run_completion_review(
        json.dumps(
            {
                "role_id": "completion_control",
                "verdict": "FAIL",
                "goal_verdict": "GOAL_MET",
                "goal_confidence": "high",
                "evidence_summary": "评审明确拒绝本轮完成",
                "suggestions": [],
            }
        )
    )

    assert result.all_passed is False
    assert result.reviews[0].passed is False
    assert result.role_outcomes[0].passed is False


def test_non_json_review_cannot_mark_completion_passed():
    result = _run_completion_review("PASS\nGOAL_MET\n所有目标已完成")

    assert result.all_passed is False
    assert result.reviews[0].passed is False
    assert result.role_outcomes[0].passed is False
    assert result.role_outcomes[0].error.startswith("format_failure:")
    assert "重试或人工确认" in result.role_outcomes[0].summary


@pytest.mark.parametrize("raw", ["[]", "null", '"PASS"'])
def test_non_object_json_review_cannot_mark_completion_passed(raw):
    result = _run_completion_review(raw)

    assert result.all_passed is False
    assert result.role_outcomes[0].passed is False
    assert result.role_outcomes[0].error.startswith("format_failure:")


@pytest.mark.parametrize(
    "payload",
    [
        {
            "role_id": "completion_control",
            "goal_verdict": "GOAL_MET",
            "evidence_summary": "测试通过",
            "suggestions": [],
        },
        {
            "role_id": "completion_control",
            "verdict": "PASS",
            "evidence_summary": "测试通过",
            "suggestions": [],
        },
    ],
)
def test_missing_completion_verdict_fails_closed(payload):
    result = _run_completion_review(json.dumps(payload))

    assert result.all_passed is False
    assert result.reviews[0].passed is False
    assert result.role_outcomes[0].passed is False


def test_completion_pass_without_evidence_fails_closed():
    outcome = parse_role_review_output(
        completion_control_role(),
        json.dumps(
            {
                "role_id": "completion_control",
                "verdict": "PASS",
                "goal_verdict": "GOAL_MET",
                "goal_confidence": "high",
                "evidence_summary": "",
                "suggestions": [],
            }
        ),
    )

    assert outcome.passed is False


def test_evidence_backed_completion_pass_remains_valid():
    result = _run_completion_review(
        json.dumps(
            {
                "role_id": "completion_control",
                "verdict": "PASS",
                "goal_verdict": "GOAL_MET",
                "goal_confidence": "high",
                "evidence_summary": "目标测试 8 passed，且目标文件存在",
                "suggestions": [],
            }
        )
    )

    assert result.all_passed is True
    assert result.reviews[0].passed is True
    assert result.role_outcomes[0].passed is True
    assert result.completion_gate_met is True


@pytest.mark.parametrize("response_role_id", [None, "", "architect"])
def test_completion_response_role_id_must_exist_and_match(response_role_id):
    payload = {
        "verdict": "PASS",
        "goal_verdict": "GOAL_MET",
        "goal_confidence": "high",
        "evidence_summary": "目标测试通过",
        "suggestions": [],
    }
    if response_role_id is not None:
        payload["role_id"] = response_role_id

    outcome = parse_role_review_output(
        completion_control_role(),
        json.dumps(payload),
    )

    assert outcome.passed is False
    assert outcome.error.startswith("format_failure:")
    assert "role_id" in outcome.error


def test_completion_gate_enabled_requires_exactly_one_completion_role():
    artifacts = ReviewArtifacts(cycle_number=1, requirement="完成目标", cwd="/repo")

    missing = run_adaptive_role_review_pipeline(
        artifacts,
        [],
        prompt_runner_factory=lambda _role: lambda *_args: "",
        completion_gate_enabled=True,
    )

    role = completion_control_role()
    valid = json.dumps(
        {
            "role_id": role.role_id,
            "verdict": "PASS",
            "goal_verdict": "GOAL_MET",
            "goal_confidence": "high",
            "evidence_summary": "目标测试通过",
            "suggestions": [],
        }
    )
    duplicate = run_adaptive_role_review_pipeline(
        artifacts,
        [role, role],
        prompt_runner_factory=lambda _role: lambda *_args: valid,
        completion_gate_enabled=True,
    )

    assert missing.all_passed is False
    assert missing.completion_gate_valid is False
    assert missing.completion_gate_error == "completion_control_missing"
    assert missing.requires_manual_confirmation is True
    assert duplicate.all_passed is False
    assert duplicate.completion_gate_valid is False
    assert duplicate.completion_gate_error == "completion_control_duplicate"
    assert duplicate.requires_manual_confirmation is True


def test_format_failure_retries_then_accepts_valid_output():
    role = completion_control_role()
    responses = iter(
        [
            "not-json",
            json.dumps(
                {
                    "role_id": role.role_id,
                    "verdict": "PASS",
                    "goal_verdict": "GOAL_MET",
                    "goal_confidence": "high",
                    "evidence_summary": "第二次返回了有效证据",
                    "suggestions": [],
                }
            ),
        ]
    )
    calls = 0

    def factory(_role):
        def runner(prompt, on_event, timeout):
            nonlocal calls
            calls += 1
            return next(responses)

        return runner

    result = run_adaptive_role_review_pipeline(
        ReviewArtifacts(cycle_number=1, requirement="完成目标", cwd="/repo"),
        [role],
        prompt_runner_factory=factory,
        completion_gate_enabled=True,
        format_retry_max_attempts=1,
    )

    assert calls == 2
    assert result.all_passed is True
    assert result.requires_manual_confirmation is False


def test_format_retry_exhaustion_requires_manual_confirmation():
    calls = 0

    def factory(_role):
        def runner(prompt, on_event, timeout):
            nonlocal calls
            calls += 1
            return "still-not-json"

        return runner

    result = run_adaptive_role_review_pipeline(
        ReviewArtifacts(cycle_number=1, requirement="完成目标", cwd="/repo"),
        [completion_control_role()],
        prompt_runner_factory=factory,
        completion_gate_enabled=True,
        format_retry_max_attempts=2,
    )

    assert calls == 3
    assert result.all_passed is False
    assert result.requires_manual_confirmation is True
    assert "人工确认" in result.manual_confirmation_reason
    assert "重试已耗尽" in result.role_outcomes[0].summary


def test_existing_spec_stage_and_retry_contract_is_unchanged(monkeypatch):
    """Format retry reuses the exact role prompt and selected tool/model."""
    normal_role = ReviewRoleSpec(
        role_id="architect",
        display_name="架构师",
        category="software",
        mission="检查实现",
        review_focus=["结构"],
        must_check=["阶段产物"],
        evidence_policy="引用证据",
        base_perspective=ReviewPerspective.ARCHITECT,
    )
    completion_role = completion_control_role()
    artifacts = ReviewArtifacts(
        cycle_number=3,
        requirement="实现登录并补测试",
        cwd="/repo",
        spec_output="SPEC-STAGE-CONTENT",
        plan_output="PLAN-STAGE-CONTENT",
        tasks_output="TASK-STAGE-CONTENT",
        build_output="BUILD-STAGE-CONTENT",
    )
    binding = ReviewAgentBinding(
        provider="acp",
        tool_name="codex",
        display_name="Codex",
        agent_type="codex",
        model_name="gpt-5.2",
    )
    calls: list[tuple[str, str, str | None, str]] = []
    role_attempts: dict[str, int] = {}

    def fake_run_with_startup_retry(
        role,
        agent_type,
        model_name,
        prompt,
        on_event,
        timeout,
        startup_timeout,
        cwd,
    ):
        calls.append((role.role_id, agent_type, model_name, prompt))
        role_attempts[role.role_id] = role_attempts.get(role.role_id, 0) + 1
        if role.role_id == normal_role.role_id:
            response_role_id = "wrong-role" if role_attempts[role.role_id] == 1 else role.role_id
            return json.dumps(
                {
                    "role_id": response_role_id,
                    "verdict": "PASS",
                    "summary": "ok",
                    "suggestions": [],
                }
            )
        return json.dumps(
            {
                "role_id": completion_role.role_id,
                "verdict": "PASS",
                "goal_verdict": "GOAL_MET",
                "goal_confidence": "high",
                "evidence_summary": "阶段产物与测试证据完整",
                "suggestions": [],
            }
        )

    monkeypatch.setattr(
        "src.spec_engine.review_strategy._run_with_startup_retry",
        fake_run_with_startup_retry,
    )
    settings = SimpleNamespace(
        spec_review_dynamic_roles_enabled=False,
        spec_review_dynamic_roles_max=1,
        spec_review_total_roles_max=2,
        spec_review_failure_circuit_enabled=False,
        spec_review_max_parallel=1,
        spec_review_timeout=30,
        spec_review_retry_max_attempts=1,
        spec_completion_gate_enabled=True,
    )

    result = AdaptiveRoleReviewStrategy().run(
        ReviewContext(
            cycle=3,
            session=None,
            settings=settings,
            project=None,
            send_prompt_with_retry_fn=lambda *args, **kwargs: "",
            build_review_exception_diagnostics_fn=lambda *args, **kwargs: {},
            circuit=ReviewCircuitState(),
            artifacts=artifacts,
            role_plan_override=[normal_role, completion_role],
            review_agents=[binding],
            agent_type="coco",
            model_name="fallback-model",
        )
    )

    architect_calls = [call for call in calls if call[0] == normal_role.role_id]
    assert result.all_passed is True
    assert len(architect_calls) == 2
    assert [(call[1], call[2]) for call in architect_calls] == [
        ("codex", "gpt-5.2"),
        ("codex", "gpt-5.2"),
    ]
    assert architect_calls[0][3] == architect_calls[1][3]
    for phase_content in (
        "SPEC-STAGE-CONTENT",
        "PLAN-STAGE-CONTENT",
        "TASK-STAGE-CONTENT",
        "BUILD-STAGE-CONTENT",
    ):
        assert phase_content in architect_calls[0][3]


def test_engine_rejects_missing_completion_verdict_from_stale_projection(
    monkeypatch,
    tmp_path,
):
    """Persisted/inconsistent review projections cannot bypass the gate."""
    engine = SpecEngine(chat_id="c", root_path=str(tmp_path))
    engine.settings = SimpleNamespace(
        spec_review_enabled=True,
        spec_review_pass_streak_required=1,
        spec_completion_gate_enabled=True,
        spec_discovery_enabled=False,
        spec_persist_phase_artifacts=False,
        spec_persist_every_phase=False,
        spec_convergence_window=0,
        spec_backlog_stuck_window=3,
        spec_success_ignore_backlog=True,
        spec_rebuild_session_between_cycles=False,
    )
    project = SpecProject.create(root_path=str(tmp_path))
    project.acceptance_criteria = ["全部完成"]
    project.criteria_tracker.init_criteria(project.acceptance_criteria)
    engine._project = project
    engine._run_state = EngineRunState.RUNNING
    engine._evaluate_criteria = MagicMock(return_value={"all_satisfied": True})
    engine._detect_convergence = MagicMock(return_value=False)
    engine._persist_state_best_effort = MagicMock()

    metrics = SimpleNamespace(
        satisfied_count=1,
        total_criteria=1,
        new_satisfied=0,
        review_suggestions=0,
        goal_attainment=1.0,
        improvement_space=0.0,
        backlog_pending=0,
        to_dict=lambda: {},
    )
    monkeypatch.setattr("src.spec_engine.engine.compute_cycle_metrics", lambda *args: metrics)
    monkeypatch.setattr("src.spec_engine.engine._persist_cycle_artifact", lambda *args: None)
    monkeypatch.setattr("src.spec_engine.engine._append_history_event", lambda *args: None)
    monkeypatch.setattr("src.spec_engine.engine._cleanup_old_cycle_artifacts", lambda *args: None)
    monkeypatch.setattr("src.spec_engine.engine._cleanup_generated_specs", lambda *args: None)

    stale_result = AdaptiveReviewResult(
        reviews=[
            PerspectiveReview(
                perspective=ReviewPerspective.PRODUCT,
                passed=True,
            )
        ],
        iteration=1,
        role_outcomes=[
            RoleReviewOutcome(
                role_id="completion_control",
                role_display_name="完成度与方向把控",
                role_category="completion_control",
                passed=True,
                blocking=True,
                goal_verdict="",
            )
        ],
        blocking_review_passed=True,
    )
    cycle = SpecCycle(cycle_number=1, review_result=stale_result)

    should_stop, reason = engine._finalize_successful_cycle(
        cycle_num=1,
        cycle=cycle,
        max_cycles=2,
        review_passed=True,
        callbacks=SpecEngineCallbacks(),
        policy=ContinuationPolicy(
            max_cycles=2,
            disable_convergence=True,
            min_cycles=1,
        ),
    )

    assert should_stop is True
    assert reason == "paused"
    assert project.status is SpecProjectStatus.PAUSED
    assert "人工确认" in (project.error or "")
