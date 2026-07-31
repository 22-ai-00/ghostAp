"""Adaptive role review pipeline for Spec Engine."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from src.engine_base import PerspectiveReview, ReviewPerspective, ReviewResult
from src.grill_me import COMPLETION_CONTROL_GRILL_ME_PROTOCOL, SPEC_REVIEW_GRILL_ME_PROTOCOL
from src.spec_engine.review_aggregation import (
    AggregatedReview,
    RoleReviewOutcome,
    RoleSuggestion,
    aggregate_role_outcomes,
)
from src.spec_engine.review_artifacts import ReviewArtifacts
from src.spec_engine.review_roles import COMPLETION_CONTROL_ROLE_ID, ReviewRoleSpec, batch_roles_by_dependencies
from src.spec_engine.utils import extract_suggestions_from_body, normalize_review_verdict
from src.utils.errors import classify_timeout, get_error_detail

logger = logging.getLogger(__name__)

PromptRunner = Callable[[str, Callable, float], str]
PromptRunnerFactory = Callable[[ReviewRoleSpec], PromptRunner]


@dataclass
class AdaptiveReviewResult(ReviewResult):
    role_outcomes: list[RoleReviewOutcome] = field(default_factory=list)
    aggregated: AggregatedReview | None = None
    role_plan_hash: str = ""
    blocking_suggestion_hash: str = ""
    blocking_review_passed: bool = False
    skipped_roles_count: int = 0
    # Completion gate (Phase 3): set by completion_control role
    completion_gate_met: bool = False
    completion_gate_confidence: str = ""
    completion_gate_evidence: str = ""
    completion_gate_enabled: bool = False
    completion_gate_valid: bool = True
    completion_gate_error: str = ""
    requires_manual_confirmation: bool = False
    manual_confirmation_reason: str = ""

    @property
    def all_passed(self) -> bool:
        reviews_passed = len(self.reviews) > 0 and all(review.passed for review in self.reviews)
        gate_passed = not self.completion_gate_enabled or self.completion_gate_valid
        return reviews_passed and gate_passed


def _build_completion_control_prompt(role: ReviewRoleSpec, artifacts: ReviewArtifacts) -> str:
    """Dedicated prompt for completion_control: instructs active verification."""
    criteria_lines: list[str] = []
    for i, c in enumerate(artifacts.acceptance_criteria or []):
        status = "PASS" if artifacts.criteria_satisfied.get(i, False) else "FAIL"
        criteria_lines.append(f"  {i + 1}. [{status}] {c}")
    criteria_section = "\n".join(criteria_lines) if criteria_lines else "  (无验收标准)"

    verify_section = ""
    if artifacts.verify_command:
        v_status = "PASS" if artifacts.verify_passed else ("FAIL" if artifacts.verify_passed is False else "未执行")
        verify_section = f"""
## 客观验证命令
命令: `{artifacts.verify_command}`
结果: {v_status}
输出摘要: {artifacts.verify_output[:2000] if artifacts.verify_output else '(无输出)'}
"""

    diff = artifacts.diff_patch or ""
    if len(diff) > 15_000:
        diff = diff[:6_000] + f"\n\n...[truncated {len(diff) - 12_000} chars]...\n\n" + diff[-6_000:]

    return f"""你是独立的"完成度与方向把控"评审员。你的唯一任务：客观判定当前实现是否真正完成了用户的原始目标。

## 核心原则
- **不择手段验证**：不要只读文本——主动运行测试、检查文件存在性、验证输出、grep 代码来确认完成度。
- **用事实说话**：每条判定必须附带客观证据（命令输出、文件内容、测试结果），不接受"看起来完成了"。
- **抵制乐观偏差**：其他角色可能倾向于放过，你的职责是怀疑并验证。
- **方向对齐**：确认当前实现精确回应了用户最初提出的问题，没有偏移。

{COMPLETION_CONTROL_GRILL_ME_PROTOCOL}

## 用户原始目标
{artifacts.requirement}

## 验收标准当前状态
{criteria_section}
{verify_section}
## 当前 Diff（代码变更）
```diff
{diff}
```

## 你的验证步骤
1. 逐条检查验收标准——对每条声称 PASS 的标准，尝试找到反例或缺失证据
2. 如果有 verify_command 且为 FAIL，这是硬性证据，不允许判定完成
3. 检查是否存在方向偏移（做了用户没要的、漏了用户要的）
4. 综合判定：GOAL_MET（可以停止）或 GOAL_NOT_MET（必须继续）

## 输出格式（严格 JSON）
{{
  "role_id": "completion_control",
  "verdict": "PASS|FAIL",
  "goal_verdict": "GOAL_MET|GOAL_NOT_MET",
  "goal_confidence": "high|medium|low",
  "evidence_summary": "用 2-3 句话列举客观事实证据",
  "per_criterion": [
    {{"index": 1, "status": "VERIFIED|UNVERIFIED|FAILED", "evidence": "具体证据"}},
  ],
  "suggestions": [
    {{
      "severity": "blocker|major|minor|observation",
      "confidence": "high|medium|low",
      "evidence": "客观事实证据",
      "recommendation": "具体改进动作",
      "target": "目标文件或模块"
    }}
  ]
}}

规则：
- goal_verdict=GOAL_MET 当且仅当：所有验收标准有客观证据支撑 AND verify_command 通过（如有）AND 方向未偏移
- 如果你无法验证某条标准（缺乏证据），verdict=FAIL, goal_verdict=GOAL_NOT_MET
- verdict=PASS 当且仅当 goal_verdict=GOAL_MET 且无 blocker/major suggestions
"""


def build_role_review_prompt(role: ReviewRoleSpec, artifacts: ReviewArtifacts) -> str:
    if role.role_id == COMPLETION_CONTROL_ROLE_ID:
        return _build_completion_control_prompt(role, artifacts)
    files = "\n".join(f"- {path}" for path in (artifacts.touched_files or [])[:50])
    diff = artifacts.diff_patch or ""
    if len(diff) > 20_000:
        head = diff[:8_000]
        tail = diff[-8_000:]
        skipped = len(diff) - 16_000
        diff = f"{head}\n\n...[truncated {skipped} chars]...\n\n{tail}"
    phase_sections: list[str] = []
    if artifacts.spec_output:
        phase_sections.append(f"## Spec 输出\n{artifacts.spec_output}")
    if artifacts.plan_output:
        phase_sections.append(f"## Plan 输出\n{artifacts.plan_output}")
    if artifacts.tasks_output:
        phase_sections.append(f"## Task 输出\n{artifacts.tasks_output}")
    if artifacts.build_output:
        phase_sections.append(f"## Build 输出\n{artifacts.build_output}")
    phase_outputs = "\n\n".join(phase_sections)
    if phase_outputs:
        phase_outputs = f"\n\n{phase_outputs}"
    return f"""你是 {role.display_name}。

## 任务目标
{artifacts.requirement}

## 角色任务
{role.mission}

## 关注点
{chr(10).join(f"- {x}" for x in role.review_focus)}

## 必查项
{chr(10).join(f"- {x}" for x in role.must_check)}

## 证据规则
{role.evidence_policy}

{SPEC_REVIEW_GRILL_ME_PROTOCOL}

## 涉及文件
{files}
{phase_outputs}

## Diff
```diff
{diff}
```

请只输出 JSON:
{{
  "role_id": "{role.role_id}",
  "verdict": "PASS|FAIL",
  "summary": "short summary",
  "suggestions": [
    {{
      "severity": "blocker|major|minor|observation",
      "confidence": "high|medium|low",
      "evidence": "artifact evidence",
      "recommendation": "specific change",
      "target": "optional file or section"
    }}
  ]
}}
"""


def _extract_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def parse_role_review_output(role: ReviewRoleSpec, raw: str) -> RoleReviewOutcome:
    """Parse model output for a role, applying evidence downgrade rules."""

    try:
        data = _extract_json_object(raw)
        if not isinstance(data, dict):
            raise ValueError("review output JSON must be an object")
    except Exception as exc:
        return _parse_role_review_text_fallback(role, raw, exc)

    suggestions: list[RoleSuggestion] = []
    for item in data.get("suggestions", []) or []:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "observation").strip().lower()
        confidence = str(item.get("confidence") or "medium").strip().lower()
        evidence = str(item.get("evidence") or "").strip()
        recommendation = str(item.get("recommendation") or "").strip()
        target = str(item.get("target") or "").strip()
        if not recommendation:
            continue
        blocking = role.blocking and severity in {"blocker", "major"} and confidence == "high"
        if blocking and not evidence:
            severity = "observation"
            blocking = False
            recommendation = f"{recommendation} [downgraded: missing evidence -> observation]"
        suggestions.append(
            RoleSuggestion(
                severity=severity,
                confidence=confidence,
                evidence=evidence,
                recommendation=recommendation,
                target=target,
                blocking=blocking,
            )
        )

    verdict = str(data.get("verdict") or "").strip().upper()
    goal_verdict = str(data.get("goal_verdict") or "").strip().upper()
    goal_confidence = str(data.get("goal_confidence") or "").strip().lower()
    goal_evidence = str(data.get("evidence_summary") or "").strip()
    has_blocking = any(s.blocking for s in suggestions)

    format_error = ""
    response_role_id = str(data.get("role_id") or "").strip()
    if not response_role_id:
        format_error = "missing_role_id"
    elif response_role_id != role.role_id:
        format_error = "role_id_mismatch"
    elif verdict not in {"PASS", "FAIL"}:
        format_error = "missing_or_invalid_verdict"
    elif role.role_id == COMPLETION_CONTROL_ROLE_ID:
        if goal_verdict not in {"GOAL_MET", "GOAL_NOT_MET"}:
            format_error = "missing_or_invalid_goal_verdict"
        elif verdict == "PASS" and goal_verdict == "GOAL_MET" and not goal_evidence:
            format_error = "missing_goal_evidence"

    passed = (
        not format_error
        and verdict == "PASS"
        and not has_blocking
        and (
            role.role_id != COMPLETION_CONTROL_ROLE_ID
            or goal_verdict == "GOAL_MET"
        )
    )
    summary = str(data.get("summary") or goal_evidence or "")
    if format_error:
        format_summary = "审查结构缺少有效完成判定，需重试或人工确认"
        summary = f"{summary}；{format_summary}" if summary else format_summary

    return RoleReviewOutcome(
        role_id=role.role_id,
        role_display_name=role.display_name,
        role_category=role.category,
        passed=passed,
        summary=summary,
        suggestions=suggestions,
        raw_preview=(raw or "")[:500],
        error=f"format_failure:{format_error}" if format_error else "",
        blocking=role.blocking,
        base_perspective_value=role.base_perspective.value if role.base_perspective else "",
        goal_verdict=goal_verdict,
        goal_confidence=goal_confidence,
        goal_evidence=goal_evidence,
    )


def _parse_role_review_text_fallback(role: ReviewRoleSpec, raw: str, exc: Exception) -> RoleReviewOutcome:
    """Keep plain-text diagnostics while failing closed on invalid output."""
    text = raw or ""
    verdict = normalize_review_verdict(text)
    raw_suggestions = extract_suggestions_from_body(text, limit=max(1, int(role.max_suggestions or 5)))

    suggestions: list[RoleSuggestion] = []
    if verdict != "PASS":
        suggestions = [
            RoleSuggestion(
                severity="observation",
                confidence="low",
                evidence="",
                recommendation=suggestion,
                blocking=False,
            )
            for suggestion in raw_suggestions
            if str(suggestion or "").strip()
        ]

    if suggestions:
        summary = f"非 JSON 审查输出保留了 {len(suggestions)} 条诊断，需重试或人工确认"
    else:
        summary = "审查输出格式异常，需重试或人工确认"

    return RoleReviewOutcome(
        role_id=role.role_id,
        role_display_name=role.display_name,
        role_category=role.category,
        passed=False,
        summary=summary,
        suggestions=suggestions,
        raw_preview=text[:500],
        error=f"format_failure:{get_error_detail(exc)}",
        blocking=role.blocking,
        base_perspective_value=role.base_perspective.value if role.base_perspective else "",
    )


class RoleReviewWorker:
    """Runs one review role using a caller-provided prompt runner."""

    def __init__(
        self,
        role: ReviewRoleSpec,
        *,
        timeout: float,
        format_retry_max_attempts: int = 0,
    ):
        self.role = role
        self.timeout = float(timeout)
        self.format_retry_max_attempts = max(0, int(format_retry_max_attempts or 0))
        self._buf: list[str] = []

    def _on_event(self, event) -> None:
        text = getattr(event, "text", None)
        if text:
            self._buf.append(str(text))

    def run(self, artifacts: ReviewArtifacts, prompt_runner: PromptRunner) -> RoleReviewOutcome:
        prompt = build_role_review_prompt(self.role, artifacts)
        raw = ""
        t0 = time.monotonic()
        try:
            for attempt in range(self.format_retry_max_attempts + 1):
                self._buf.clear()
                raw = prompt_runner(prompt, self._on_event, self.timeout) or ""
                if not raw:
                    raw = "".join(self._buf)
                outcome = parse_role_review_output(self.role, raw)
                if not outcome.error.startswith("format_failure:"):
                    return outcome
                if attempt < self.format_retry_max_attempts:
                    logger.warning(
                        "[RoleReviewWorker:%s] invalid output format "
                        "(attempt %d/%d), retrying with the same role runner",
                        self.role.role_id,
                        attempt + 1,
                        self.format_retry_max_attempts + 1,
                    )
                    continue
                outcome.error = (
                    f"{outcome.error};retry_exhausted={self.format_retry_max_attempts}"
                )
                outcome.summary = (
                    f"{outcome.summary}；格式重试已耗尽，需人工确认"
                    if outcome.summary
                    else "格式重试已耗尽，需人工确认"
                )
                return outcome
        except Exception as exc:
            err = get_error_detail(exc)
            startup_elapsed = getattr(exc, "startup_elapsed_s", None)
            startup_timeout = getattr(exc, "startup_timeout_s", None)
            is_startup_failure = bool(getattr(exc, "startup_failed", False))
            is_timeout = classify_timeout(
                exc,
                elapsed_s=startup_elapsed,
                timeout_s=startup_timeout,
            )
            logger.warning(
                "[RoleReviewWorker:%s] failed: %s (startup=%s, timeout=%s)",
                self.role.role_id, err, is_startup_failure, is_timeout,
            )
            if is_timeout:
                if self.role.blocking:
                    error_str = f"timeout_blocking:{err}"
                    blocking = True
                else:
                    error_str = f"timeout_degraded:{err}"
                    blocking = False
            else:
                error_str = err
                blocking = True

            if is_timeout and is_startup_failure:
                if self.role.blocking:
                    summary = "审查启动超时（阻断，需重试或修复基础设施）"
                else:
                    summary = "审查启动超时（已降级）"
                rec_prefix = f"{self.role.display_name} 启动超时："
                evidence = f"startup timed out after {int(startup_elapsed * 1000) if startup_elapsed else int((time.monotonic() - t0) * 1000)}ms (retries exhausted)"
            elif is_timeout:
                if self.role.blocking:
                    summary = "审查超时（阻断，需重试或修复基础设施）"
                else:
                    summary = "审查超时（已降级）"
                rec_prefix = f"{self.role.display_name} 审查超时："
                evidence = f"role failed after {int((time.monotonic() - t0) * 1000)}ms"
            elif is_startup_failure:
                summary = f"审查启动失败：{err}"
                rec_prefix = f"{self.role.display_name} 启动失败："
                evidence = f"startup failed after {int(startup_elapsed * 1000) if startup_elapsed else int((time.monotonic() - t0) * 1000)}ms"
            else:
                summary = f"审查异常：{err}"
                rec_prefix = f"{self.role.display_name} 审查异常："
                evidence = f"role failed after {int((time.monotonic() - t0) * 1000)}ms"

            # Degrade non-blocking roles on startup failure — mark as skipped
            # instead of failing the whole review pipeline.
            if is_startup_failure and not self.role.blocking:
                logger.info(
                    "[RoleReviewWorker:%s] skipping non-blocking role after startup failure",
                    self.role.role_id,
                )
                return RoleReviewOutcome(
                    role_id=self.role.role_id,
                    role_display_name=self.role.display_name,
                    role_category=self.role.category,
                    passed=True,
                    summary=f"{self.role.display_name} 跳过（启动失败，角色非阻断）",
                    suggestions=[
                        RoleSuggestion(
                            severity="observation",
                            confidence="low",
                            evidence=evidence,
                            recommendation=f"{self.role.display_name} 因启动失败被跳过",
                            blocking=False,
                        )
                    ],
                    error=f"skipped_startup_failure:{err}",
                    blocking=False,
                    skipped=True,
                    base_perspective_value=self.role.base_perspective.value if self.role.base_perspective else "",
                )

            return RoleReviewOutcome(
                role_id=self.role.role_id,
                role_display_name=self.role.display_name,
                role_category=self.role.category,
                passed=False,
                summary=summary,
                suggestions=[
                    RoleSuggestion(
                        severity="major" if is_timeout else "observation",
                        confidence="low",
                        evidence=evidence,
                        recommendation=rec_prefix + err,
                        blocking=blocking,
                    )
                ],
                error=error_str,
                blocking=blocking,
                base_perspective_value=self.role.base_perspective.value if self.role.base_perspective else "",
            )


def _fallback_perspective(role: ReviewRoleSpec) -> ReviewPerspective:
    if role.base_perspective:
        return role.base_perspective
    category = role.category.lower()
    if category in {"security", "api", "performance", "software"}:
        return ReviewPerspective.ARCHITECT
    if category in {"writing", "research", "domain"}:
        return ReviewPerspective.PRODUCT
    if category in {"design", "ux"}:
        return ReviewPerspective.DESIGNER
    return ReviewPerspective.TESTER


def validate_completion_gate_outcomes(
    outcomes: list[RoleReviewOutcome],
) -> tuple[bool, str, RoleReviewOutcome | None]:
    """Validate the completion-control role contract without trusting projections."""

    completion_outcomes = [
        outcome
        for outcome in outcomes
        if outcome.role_id == COMPLETION_CONTROL_ROLE_ID
    ]
    if not completion_outcomes:
        return False, "completion_control_missing", None
    if len(completion_outcomes) != 1:
        return False, "completion_control_duplicate", None

    outcome = completion_outcomes[0]
    if outcome.error.startswith("format_failure:"):
        return False, "completion_control_format_failure", outcome
    if outcome.goal_verdict not in {"GOAL_MET", "GOAL_NOT_MET"}:
        return False, "completion_control_invalid", outcome
    return True, "", outcome


def _outcomes_to_review_result(
    outcomes: list[RoleReviewOutcome],
    iteration: int,
    *,
    completion_gate_enabled: bool = False,
) -> AdaptiveReviewResult:
    aggregated = aggregate_role_outcomes(outcomes)
    reviews: list[PerspectiveReview] = []
    blocking_by_role: dict[str, list[str]] = {}
    observation_by_role: dict[str, list[str]] = {}

    for suggestion in aggregated.blocking_suggestions:
        text = suggestion.to_repair_text(aggregated.role_names)
        for role_id in suggestion.role_ids:
            blocking_by_role.setdefault(role_id, []).append(text)
    for suggestion in aggregated.observations:
        text = f"observation: {suggestion.to_repair_text(aggregated.role_names)}"
        for role_id in suggestion.role_ids:
            observation_by_role.setdefault(role_id, []).append(text)

    for outcome in outcomes:
        role = ReviewRoleSpec(
            role_id=outcome.role_id,
            display_name=outcome.role_display_name,
            category=outcome.role_category,
            mission="",
            review_focus=[],
            must_check=[],
            evidence_policy="",
            blocking=outcome.blocking,
            base_perspective=ReviewPerspective(outcome.base_perspective_value) if outcome.base_perspective_value else None,
        )
        suggestions = blocking_by_role.get(outcome.role_id) or observation_by_role.get(outcome.role_id, [])
        reviews.append(
            PerspectiveReview(
                perspective=_fallback_perspective(role),
                passed=outcome.passed and not bool(blocking_by_role.get(outcome.role_id)),
                suggestions=suggestions,
                summary=outcome.summary,
                role_id=outcome.role_id,
                role_display_name=outcome.role_display_name,
                role_category=outcome.role_category,
                blocking=outcome.blocking,
            )
        )

    # Extract and validate completion gate from exactly one completion_control role.
    completion_gate_met = False
    completion_gate_confidence = ""
    completion_gate_evidence = ""
    completion_gate_valid = True
    completion_gate_error = ""
    completion_outcome: RoleReviewOutcome | None = None
    if completion_gate_enabled:
        (
            completion_gate_valid,
            completion_gate_error,
            completion_outcome,
        ) = validate_completion_gate_outcomes(outcomes)
    else:
        completion_outcome = next(
            (
                outcome
                for outcome in outcomes
                if outcome.role_id == COMPLETION_CONTROL_ROLE_ID
            ),
            None,
        )
    if completion_outcome is not None:
        completion_gate_met = (
            completion_gate_valid
            and completion_outcome.passed
            and completion_outcome.goal_verdict == "GOAL_MET"
        )
        completion_gate_confidence = completion_outcome.goal_confidence
        completion_gate_evidence = completion_outcome.goal_evidence

    format_failure_roles = [
        outcome.role_id
        for outcome in outcomes
        if outcome.error.startswith("format_failure:")
    ]
    manual_reasons: list[str] = []
    if format_failure_roles:
        manual_reasons.append(
            "审查输出格式重试已耗尽"
            f"（角色：{', '.join(format_failure_roles)}）"
        )
    if completion_gate_enabled and not completion_gate_valid:
        manual_reasons.append(
            f"完成度闸门契约无效（{completion_gate_error}）"
        )
    requires_manual_confirmation = bool(manual_reasons)
    manual_confirmation_reason = "；".join(manual_reasons)
    if manual_confirmation_reason:
        manual_confirmation_reason = f"{manual_confirmation_reason}，需人工确认"

    skipped_roles_count = sum(1 for o in outcomes if o.skipped)
    blocking_review_passed = all(pr.passed for pr in reviews)
    if completion_gate_enabled:
        blocking_review_passed = blocking_review_passed and completion_gate_valid

    return AdaptiveReviewResult(
        reviews=reviews,
        iteration=iteration,
        role_outcomes=outcomes,
        aggregated=aggregated,
        blocking_suggestion_hash=aggregated.blocking_hash(),
        blocking_review_passed=blocking_review_passed,
        skipped_roles_count=skipped_roles_count,
        completion_gate_met=completion_gate_met,
        completion_gate_confidence=completion_gate_confidence,
        completion_gate_evidence=completion_gate_evidence,
        completion_gate_enabled=completion_gate_enabled,
        completion_gate_valid=completion_gate_valid,
        completion_gate_error=completion_gate_error,
        requires_manual_confirmation=requires_manual_confirmation,
        manual_confirmation_reason=manual_confirmation_reason,
    )


def _run_batch(
    artifacts: ReviewArtifacts,
    roles: list[ReviewRoleSpec],
    *,
    prompt_runner_factory: PromptRunnerFactory,
    max_parallel: int,
    timeout: float,
    role_timeout_multipliers: dict[str, float] | None = None,
    format_retry_max_attempts: int = 0,
) -> list[RoleReviewOutcome]:
    outcomes: list[RoleReviewOutcome] = []
    max_parallel = max(1, int(max_parallel or 1))
    multipliers = role_timeout_multipliers or {}
    for i in range(0, len(roles), max_parallel):
        wave = roles[i:i + max_parallel]
        with ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix="role-review-") as pool:
            futures = {
                pool.submit(
                    RoleReviewWorker(
                        role,
                        timeout=timeout * multipliers.get(role.role_id, 1.0),
                        format_retry_max_attempts=format_retry_max_attempts,
                    ).run,
                    artifacts,
                    prompt_runner_factory(role),
                ): role
                for role in wave
            }
            for future in as_completed(futures):
                outcomes.append(future.result())
    order = {role.role_id: idx for idx, role in enumerate(roles)}
    outcomes.sort(key=lambda outcome: order.get(outcome.role_id, 999))
    return outcomes


def run_adaptive_role_review_pipeline(
    artifacts: ReviewArtifacts,
    roles: list[ReviewRoleSpec],
    *,
    prompt_runner_factory: PromptRunnerFactory,
    max_parallel: int = 3,
    timeout: float = 240.0,
    role_timeout_multipliers: dict[str, float] | None = None,
    iteration: int | None = None,
    completion_gate_enabled: bool = False,
    format_retry_max_attempts: int = 0,
) -> AdaptiveReviewResult:
    """Run adaptive roles with dependency batching and parallel workers."""

    outcomes: list[RoleReviewOutcome] = []
    for batch in batch_roles_by_dependencies(roles):
        outcomes.extend(
            _run_batch(
                artifacts,
                batch,
                prompt_runner_factory=prompt_runner_factory,
                max_parallel=max_parallel,
                timeout=timeout,
                role_timeout_multipliers=role_timeout_multipliers,
                format_retry_max_attempts=format_retry_max_attempts,
            )
        )
    order = {role.role_id: idx for idx, role in enumerate(roles)}
    outcomes.sort(key=lambda outcome: order.get(outcome.role_id, 999))
    return _outcomes_to_review_result(
        outcomes,
        iteration or artifacts.cycle_number,
        completion_gate_enabled=completion_gate_enabled,
    )
