from __future__ import annotations

import json
import types

from src.spec_engine.review import ReviewCircuitState
from src.spec_engine.review_artifacts import ReviewArtifacts
from src.spec_engine.review_roles import COMPLETION_CONTROL_ROLE_ID, ReviewRoleSpec
from src.spec_engine.review_strategy import (
    AdaptiveRoleReviewStrategy,
    MultiPerspectiveStrategy,
    ReviewContext,
    select_review_strategy,
)


def _settings(**overrides):
    base = {
        "spec_review_strategy": "adaptive_roles",
        "spec_review_dynamic_roles_enabled": True,
        "spec_review_dynamic_roles_max": 3,
        "spec_review_total_roles_max": 8,
        "spec_review_max_parallel": 3,
        "spec_review_timeout": 30,
        "spec_review_failure_circuit_enabled": False,
        "spec_review_failure_max_consecutive": 3,
        "spec_review_failure_cooldown_cycles": 2,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _ctx(**overrides):
    artifacts = ReviewArtifacts(
        cycle_number=2,
        requirement="write auth code and tests",
        cwd="/tmp/project",
        build_output="implemented oauth login",
        diff_patch="diff --git a/src/auth.py b/src/auth.py\n+def login(): pass",
        touched_files=["src/auth.py"],
    )
    base = dict(
        cycle=2,
        session=object(),
        settings=_settings(),
        project=object(),
        send_prompt_with_retry_fn=lambda prompt, **kwargs: json.dumps({"verdict": "PASS", "summary": "ok", "suggestions": []}),
        build_review_exception_diagnostics_fn=lambda **kwargs: {},
        circuit=ReviewCircuitState(),
        artifacts=artifacts,
        agent_type="coco",
        model_name="gpt-test",
    )
    base.update(overrides)
    return ReviewContext(**base)


def _factory(runner):
    return lambda role: (
        lambda prompt, on_event=None, timeout=30: runner(
            prompt,
            role_id=role.role_id,
            on_event=on_event,
            timeout=timeout,
        )
    )


def test_select_default_is_adaptive_roles():
    strategy = select_review_strategy(types.SimpleNamespace())
    assert isinstance(strategy, AdaptiveRoleReviewStrategy)


def test_select_legacy_multi_perspective_explicit():
    strategy = select_review_strategy(types.SimpleNamespace(spec_review_strategy="multi_perspective"))
    assert isinstance(strategy, MultiPerspectiveStrategy)


def test_adaptive_strategy_uses_artifacts_and_records_hashes():
    seen_prompts: list[str] = []

    def runner(prompt, **kwargs):
        seen_prompts.append(prompt)
        if '"role_id": "completion_control"' in prompt:
            return json.dumps(
                {
                    "role_id": kwargs["role_id"],
                    "verdict": "PASS",
                    "goal_verdict": "GOAL_MET",
                    "goal_confidence": "high",
                    "evidence_summary": "验收证据完整",
                    "suggestions": [],
                }
            )
        return json.dumps(
            {
                "role_id": kwargs["role_id"],
                "verdict": "PASS",
                "summary": "ok",
                "suggestions": [],
            }
        )

    result = AdaptiveRoleReviewStrategy().run(_ctx(prompt_runner_factory=_factory(runner)))

    assert result.all_passed is True
    assert result.role_plan_hash
    assert result.blocking_suggestion_hash == ""
    assert any("src/auth.py" in prompt for prompt in seen_prompts)
    assert any("完成度与方向把控" in prompt for prompt in seen_prompts)
    assert any(review.role_id == COMPLETION_CONTROL_ROLE_ID for review in result.reviews)
    assert any(review.role_id for review in result.reviews)


def test_adaptive_strategy_downgrades_evidence_less_suggestion_but_keeps_fail_closed():
    def runner(prompt, **kwargs):
        return json.dumps(
            {
                "role_id": kwargs["role_id"],
                "verdict": "FAIL",
                "summary": "claim without evidence",
                "suggestions": [
                    {
                        "severity": "blocker",
                        "confidence": "high",
                        "evidence": "",
                        "recommendation": "Rewrite everything",
                    }
                ],
            }
        )

    role = ReviewRoleSpec(
        role_id="fact_checker",
        display_name="Fact Checker",
        category="research",
        mission="check facts",
        review_focus="evidence",
        must_check=["evidence"],
        evidence_policy="must cite exact evidence",
    )
    result = AdaptiveRoleReviewStrategy().run(
        _ctx(
            settings=_settings(spec_review_dynamic_roles_enabled=False),
            artifacts=ReviewArtifacts(cycle_number=1, requirement="research market", cwd="/tmp"),
            prompt_runner_factory=_factory(runner),
            role_plan_override=[role],
        )
    )

    assert result.all_passed is False
    assert result.blocking_suggestion_hash == ""
    assert "missing evidence" in result.reviews[0].suggestions[0]


def test_adaptive_strategy_falls_back_to_fixed_roles_when_planner_fails(monkeypatch):
    prompts: list[str] = []

    def runner(prompt, **kwargs):
        prompts.append(prompt)
        if '"role_id": "completion_control"' in prompt:
            return json.dumps(
                {
                    "role_id": kwargs["role_id"],
                    "verdict": "PASS",
                    "goal_verdict": "GOAL_MET",
                    "goal_confidence": "high",
                    "evidence_summary": "验收证据完整",
                    "suggestions": [],
                }
            )
        return json.dumps(
            {
                "role_id": kwargs["role_id"],
                "verdict": "PASS",
                "summary": "ok",
                "suggestions": [],
            }
        )

    monkeypatch.setattr(
        "src.spec_engine.review_strategy.build_adaptive_role_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("planner failed")),
    )

    result = AdaptiveRoleReviewStrategy().run(_ctx(prompt_runner_factory=_factory(runner)))

    assert result.all_passed is True
    assert len(result.reviews) == 6
    assert {review.role_id for review in result.reviews} == {
        "architect",
        "product",
        "user",
        "tester",
        "designer",
        COMPLETION_CONTROL_ROLE_ID,
    }
    assert prompts
