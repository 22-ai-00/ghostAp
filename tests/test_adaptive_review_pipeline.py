import json
import threading
import time

from src.engine_base import ReviewPerspective
from src.spec_engine.adaptive_review import parse_role_review_output, run_adaptive_role_review_pipeline
from src.spec_engine.review_aggregation import aggregate_role_outcomes
from src.spec_engine.review_artifacts import ReviewArtifacts
from src.spec_engine.review_roles import ReviewRoleSpec, completion_control_role


def _role(role_id: str, *, depends_on: list[str] | None = None) -> ReviewRoleSpec:
    return ReviewRoleSpec(
        role_id=role_id,
        display_name=role_id.replace("_", " ").title(),
        category="test",
        mission=f"review as {role_id}",
        review_focus=["focus"],
        must_check=["check"],
        evidence_policy="blockers require evidence",
        depends_on=depends_on or [],
        base_perspective=ReviewPerspective.TESTER if role_id == "tester" else None,
    )


def _artifacts() -> ReviewArtifacts:
    return ReviewArtifacts(cycle_number=1, requirement="ship feature", cwd="/repo")


def _json(role_id: str, *, verdict: str = "PASS", suggestions: list[dict] | None = None) -> str:
    return json.dumps(
        {
            "role_id": role_id,
            "verdict": verdict,
            "summary": f"{role_id} summary",
            "suggestions": suggestions or [],
        },
        ensure_ascii=False,
    )


def test_roles_in_same_dependency_batch_run_concurrently():
    roles = [_role("editor"), _role("fact_checker")]
    active = 0
    max_active = 0
    lock = threading.Lock()

    def factory(role):
        def runner(prompt, on_event, timeout):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.08)
            with lock:
                active -= 1
            return _json(role.role_id)
        return runner

    result = run_adaptive_role_review_pipeline(
        _artifacts(),
        roles,
        prompt_runner_factory=factory,
        max_parallel=2,
        timeout=5,
    )

    assert max_active == 2
    assert result.all_passed
    assert [review.role_id for review in result.reviews] == ["editor", "fact_checker"]


def test_dependency_layers_run_after_prerequisites_finish():
    roles = [_role("fact_checker"), _role("conclusion_editor", depends_on=["fact_checker"])]
    order: list[str] = []

    def factory(role):
        def runner(prompt, on_event, timeout):
            order.append(f"start:{role.role_id}")
            time.sleep(0.01)
            order.append(f"done:{role.role_id}")
            return _json(role.role_id)
        return runner

    run_adaptive_role_review_pipeline(
        _artifacts(),
        roles,
        prompt_runner_factory=factory,
        max_parallel=2,
        timeout=5,
    )

    assert order == [
        "start:fact_checker",
        "done:fact_checker",
        "start:conclusion_editor",
        "done:conclusion_editor",
    ]


def test_evidence_less_blocker_is_downgraded_but_explicit_fail_stays_closed():
    roles = [_role("fact_checker")]

    def factory(role):
        def runner(prompt, on_event, timeout):
            return _json(
                role.role_id,
                verdict="FAIL",
                suggestions=[
                    {
                        "severity": "blocker",
                        "confidence": "high",
                        "evidence": "",
                        "recommendation": "补充来源",
                    }
                ],
            )
        return runner

    result = run_adaptive_role_review_pipeline(
        _artifacts(),
        roles,
        prompt_runner_factory=factory,
        max_parallel=1,
        timeout=5,
    )

    assert result.all_passed is False
    assert result.reviews[0].passed is False
    assert "observation" in result.reviews[0].suggestions[0]


def test_invalid_role_json_requires_retry_or_manual_confirmation():
    outcome = parse_role_review_output(
        _role("fact_checker"),
        "I cannot provide JSON for this role review.",
    )

    assert outcome.passed is False
    assert outcome.blocking is True
    assert outcome.suggestions == []
    assert "重试或人工确认" in outcome.summary
    assert outcome.error.startswith("format_failure:")


def test_non_json_role_output_keeps_plain_text_diagnostics_but_fails_closed():
    outcome = parse_role_review_output(
        _role("tester"),
        "FAIL\n- 补充工具流式更新回归测试\n- 明确多角色审查建议和角色面板的关系",
    )

    assert outcome.passed is False
    assert outcome.blocking is True
    assert [item.recommendation for item in outcome.suggestions] == [
        "补充工具流式更新回归测试",
        "明确多角色审查建议和角色面板的关系",
    ]
    assert all("role output was not valid JSON" not in item.evidence for item in outcome.suggestions)


def test_aggregated_blocking_suggestion_keeps_role_name_and_maps_to_base_perspective():
    roles = [_role("tester")]

    def factory(role):
        def runner(prompt, on_event, timeout):
            return _json(
                role.role_id,
                verdict="FAIL",
                suggestions=[
                    {
                        "severity": "major",
                        "confidence": "high",
                        "evidence": "tests/test_feature.py lacks edge case",
                        "recommendation": "补充空输入边界测试",
                        "target": "tests/test_feature.py",
                    }
                ],
            )
        return runner

    result = run_adaptive_role_review_pipeline(
        _artifacts(),
        roles,
        prompt_runner_factory=factory,
        max_parallel=1,
        timeout=5,
    )

    assert result.all_passed is False
    assert result.reviews[0].perspective is ReviewPerspective.TESTER
    assert result.reviews[0].role_id == "tester"
    assert result.reviews[0].role_display_name == "Tester"
    assert result.reviews[0].suggestions == [
        "[Tester] 补充空输入边界测试 (evidence: tests/test_feature.py lacks edge case; target: tests/test_feature.py)"
    ]


def test_aggregator_deduplicates_same_recommendation_from_multiple_roles():
    roles = [_role("a"), _role("b")]

    def factory(role):
        def runner(prompt, on_event, timeout):
            return _json(
                role.role_id,
                verdict="FAIL",
                suggestions=[
                    {
                        "severity": "major",
                        "confidence": "high",
                        "evidence": f"{role.role_id} evidence",
                        "recommendation": "补充来源",
                    }
                ],
            )
        return runner

    result = run_adaptive_role_review_pipeline(
        _artifacts(),
        roles,
        prompt_runner_factory=factory,
        max_parallel=2,
        timeout=5,
    )
    aggregated = aggregate_role_outcomes(result.role_outcomes)

    assert len(aggregated.blocking_suggestions) == 1
    assert aggregated.blocking_suggestions[0].role_ids == ["a", "b"]


def test_completion_control_blocks_direction_drift_with_artifact_evidence():
    role = completion_control_role()

    def factory(role):
        def runner(prompt, on_event, timeout):
            # Dedicated prompt contains goal verification instructions
            assert "不择手段验证" in prompt or "GOAL_MET" in prompt
            return _json(
                role.role_id,
                verdict="FAIL",
                suggestions=[
                    {
                        "severity": "major",
                        "confidence": "high",
                        "evidence": "requirement asks Feishu card guard; diff only changes README",
                        "recommendation": "继续下一轮，回到用户要求的卡片 guard 实现和回归测试",
                        "target": "src/card/",
                    }
                ],
            )
        return runner

    result = run_adaptive_role_review_pipeline(
        ReviewArtifacts(
            cycle_number=1,
            requirement="修复 Feishu card guard 的完成度偏差",
            cwd="/repo",
            spec_output='{"acceptance_criteria":["卡片 guard 覆盖 schema 错误"]}',
            plan_output='{"steps":["修改 card guard","补回归测试"]}',
            tasks_output="1. 修改 guard\n2. 补测试",
            build_output="只更新 README，未运行测试",
            diff_patch="diff --git a/README.md b/README.md\n+docs only",
            acceptance_criteria=["卡片 guard 覆盖 schema 错误"],
            criteria_satisfied={0: False},
        ),
        [role],
        prompt_runner_factory=factory,
        max_parallel=1,
        timeout=5,
    )

    assert result.all_passed is False
    assert result.reviews[0].role_id == "completion_control"
    assert result.reviews[0].role_display_name == "完成度与方向把控"
    assert result.reviews[0].blocking is True
    assert result.blocking_suggestion_hash
    assert "继续下一轮" in result.reviews[0].suggestions[0]


def test_role_review_prompt_includes_phase_outputs_for_completion_judgment():
    role = completion_control_role()
    seen_prompts: list[str] = []

    def factory(role):
        def runner(prompt, on_event, timeout):
            seen_prompts.append(prompt)
            return _json(role.role_id)
        return runner

    run_adaptive_role_review_pipeline(
        ReviewArtifacts(
            cycle_number=1,
            requirement="修复完成度判断",
            cwd="/repo",
            spec_output="SPEC: 用户方向是修复 completion guard",
            plan_output="PLAN: 修改 review role planner",
            tasks_output="TASKS: 1. 添加独立角色",
            build_output="BUILD: tests passed",
            diff_patch="diff --git a/src/spec_engine/review_roles.py b/src/spec_engine/review_roles.py\n+role",
            acceptance_criteria=["修复 completion guard"],
            criteria_satisfied={0: True},
        ),
        [role],
        prompt_runner_factory=factory,
        max_parallel=1,
        timeout=5,
    )

    prompt = seen_prompts[0]
    # Dedicated completion_control prompt includes requirement, criteria status, and diff
    assert "修复完成度判断" in prompt
    assert "修复 completion guard" in prompt
    assert "GOAL_MET" in prompt
    assert "diff" in prompt.lower()
