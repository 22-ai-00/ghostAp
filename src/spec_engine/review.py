"""Authoritative fully automatic Spec review pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from ..agent_session import EphemeralReviewSession
from ..engine_base import PerspectiveReview, ReviewPerspective, ReviewResult
from ..grill_me import COMPLETION_CONTROL_GRILL_ME_PROTOCOL, SPEC_REVIEW_GRILL_ME_PROTOCOL
from ..utils.errors import classify_timeout, get_error_detail
from ..utils.review_diagnostics import normalize_review_diagnostics
from .models import (
    AdaptiveReviewResult,
    AggregatedReview,
    AggregatedSuggestion,
    ReviewAgentBinding,
    ReviewCircuitState,
    ReviewContext,
    ReviewRoleSpec,
    RoleReviewOutcome,
    RoleSuggestion,
)
from .retry_status import RetryEvent, RetryStatus

logger = logging.getLogger(__name__)
COMPLETION_ROLE = "completion_control"
STARTUP_RETRIES = 2


class _ReviewCancelled(RuntimeError):
    pass


def normalize_review_agents(items: Iterable[object] | None) -> list[ReviewAgentBinding]:
    result: list[ReviewAgentBinding] = []
    seen: set[str] = set()
    for item in items or ():
        binding = item if isinstance(item, ReviewAgentBinding) else ReviewAgentBinding.from_dict(item)
        if not binding:
            continue
        key = binding.selection_key or f"{binding.agent_type}:{binding.model_name or 'default'}"
        if key not in seen:
            seen.add(key)
            result.append(binding)
    return result


def review_result_to_text(review: ReviewResult) -> str:
    lines: list[str] = []
    for item in review.reviews if review else ():
        lines.append(f"[{item.role_display_name or item.perspective.name}] {'PASS' if item.passed else 'FAIL'}")
        lines.extend(f"- {suggestion}" for suggestion in item.suggestions)
        lines.append("")
    return "\n".join(lines).strip() + ("\n" if lines else "")


def _role(role_id, name, category, mission, focus, checks, *, blocking=True, perspective=None):
    return ReviewRoleSpec(
        role_id=role_id,
        display_name=name,
        category=category,
        mission=mission,
        review_focus=list(focus),
        must_check=list(checks),
        blocking=blocking,
        base_perspective=perspective,
    )


def _completion_role() -> ReviewRoleSpec:
    return _role(
        COMPLETION_ROLE,
        "完成度与方向把控",
        "completion_control",
        "独立验证产物是否真正完成用户原始目标；证据不足必须失败。",
        ("原始目标", "验收证据", "客观验证", "遗漏范围"),
        ("逐条验证验收标准", "检查验证命令", "检查方向偏移"),
        perspective=ReviewPerspective.PRODUCT,
    )


def _missing_artifacts_roles() -> list[ReviewRoleSpec]:
    return [
        _role(
            "missing_artifacts_domain",
            "领域审查员",
            "task_derived",
            "审查交付物是否符合任务领域约束；审查产物缺失时必须失败关闭。",
            ("领域约束",),
            ("审查产物是否可用",),
        ),
        _role(
            "missing_artifacts_audience",
            "目标受众代表",
            "task_derived",
            "审查交付物是否服务目标受众；审查产物缺失时必须失败关闭。",
            ("受众需要",),
            ("审查产物是否可用",),
        ),
    ]


def _expert_role_id(role: str, used: set[str]) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_") or "domain_expert"
    candidate = f"expert_{stem}"
    suffix = 2
    while candidate in used or candidate == COMPLETION_ROLE:
        candidate = f"expert_{stem}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _task_derived_roles(artifacts) -> list[ReviewRoleSpec]:
    """Translate the Spec phase's inferred experts into review roles.

    The Spec model determines expertise from the actual task before execution;
    this function deliberately does not classify keywords into canned domains.
    """
    roles: list[ReviewRoleSpec] = []
    used: set[str] = set()
    for item in getattr(artifacts, "required_experts", []) or ():
        if not isinstance(item, dict):
            continue
        name = str(item.get("role") or "").strip()
        purpose = str(item.get("purpose") or "").strip()
        if not name or not purpose:
            continue
        raw_focus = item.get("focus", [])
        raw_checks = item.get("checks", [])
        focus = [str(value).strip() for value in raw_focus if str(value).strip()] if isinstance(raw_focus, list) else []
        checks = [str(value).strip() for value in raw_checks if str(value).strip()] if isinstance(raw_checks, list) else []
        roles.append(_role(
            _expert_role_id(name, used), name, "task_derived", purpose,
            focus or ["任务目标与交付物"], checks or ["是否满足该专家定义的质量要求"],
        ))
    if roles:
        return roles

    requirement = str(getattr(artifacts, "requirement", "") or "").strip()
    target = requirement[:120] or "当前任务"
    return [
        _role("task_domain_expert", f"{target}领域专家", "task_derived", f"从任务领域专业性审查：{target}", ("领域约束",), ("结论是否符合领域事实",)),
        _role("task_audience_representative", "目标受众代表", "task_derived", f"从目标受众审查交付物是否实现：{target}", ("受众需要",), ("受众能否理解和使用交付物",)),
        _role("task_evidence_reviewer", "证据与验收审查员", "task_derived", f"从证据与验收审查：{target}", ("验收证据",), ("每项承诺是否有可检查证据",)),
    ]


def _plan_roles(ctx: ReviewContext) -> tuple[list[ReviewRoleSpec], str]:
    if ctx.role_plan_override:
        roles = list(ctx.role_plan_override)
    else:
        limit = max(2, int(getattr(ctx.settings, "spec_review_total_roles_max", 8) or 8))
        roles = _task_derived_roles(ctx.artifacts)
        roles = roles[:limit - 1] + [_completion_role()]
    if not any(role.role_id == COMPLETION_ROLE for role in roles):
        roles.append(_completion_role())
    payload = [role.to_dict() for role in roles if role.blocking]
    return roles, hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def _artifact_context(artifacts) -> str:
    criteria = "\n".join(
        f"{index + 1}. [{'PASS' if artifacts.criteria_satisfied.get(index, False) else 'FAIL'}] {criterion}"
        for index, criterion in enumerate(artifacts.acceptance_criteria or ())
    ) or "(无显式验收标准)"
    verify = "(无验证命令)"
    if artifacts.verify_command:
        state = "PASS" if artifacts.verify_passed is True else "FAIL" if artifacts.verify_passed is False else "未执行"
        verify = f"{artifacts.verify_command}\n状态: {state}\n{(artifacts.verify_output or '')[:2000]}"
    diff = artifacts.diff_patch or ""
    if len(diff) > 16_000:
        diff = diff[:7_000] + f"\n...[truncated {len(diff) - 14_000} chars]...\n" + diff[-7_000:]
    phases = "\n\n".join(
        f"## {name}\n{value}" for name, value in (
            ("Spec", artifacts.spec_output), ("Plan", artifacts.plan_output), ("Build", artifacts.build_output)
        ) if value
    )
    return f"""## 用户原始目标
{artifacts.requirement}
## 验收标准
{criteria}
## 客观验证
{verify}
## 涉及文件
{chr(10).join(f'- {path}' for path in (artifacts.touched_files or ())[:50])}
{phases}
## Diff
```diff
{diff}
```"""


def _build_prompt(role: ReviewRoleSpec, artifacts) -> str:
    context = _artifact_context(artifacts)
    if role.role_id == COMPLETION_ROLE:
        return f"""你是独立完成度审查员，证据不足必须失败，不得跟随其他角色的乐观结论。
{COMPLETION_CONTROL_GRILL_ME_PROTOCOL}
{context}
只输出严格 JSON：
{{"role_id":"completion_control","verdict":"PASS|FAIL","goal_verdict":"GOAL_MET|GOAL_NOT_MET","goal_confidence":"high|medium|low","evidence_summary":"事实证据","suggestions":[{{"severity":"blocker|major|minor|observation","confidence":"high|medium|low","evidence":"证据","recommendation":"动作","target":"位置"}}]}}
只有全部验收标准有证据、验证命令通过（如有）且方向一致，才允许 PASS/GOAL_MET。"""
    return f"""你是{role.display_name}。任务：{role.mission}
关注：{'、'.join(role.review_focus)}；必查：{'、'.join(role.must_check)}。
高置信 blocker/major 必须引用具体证据。
{SPEC_REVIEW_GRILL_ME_PROTOCOL}
{context}
只输出严格 JSON：
{{"role_id":"{role.role_id}","verdict":"PASS|FAIL","summary":"摘要","suggestions":[{{"severity":"blocker|major|minor|observation","confidence":"high|medium|low","evidence":"证据","recommendation":"动作","target":"位置"}}]}}"""


def _extract_json(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("review output must be an object")
    return value


def _failure(role: ReviewRoleSpec, message: str, error: str, *, skipped=False) -> RoleReviewOutcome:
    blocking = role.blocking and not skipped
    return RoleReviewOutcome(
        role_id=role.role_id,
        role_display_name=role.display_name,
        role_category=role.category,
        passed=skipped,
        summary=message,
        suggestions=[RoleSuggestion(
            severity="major" if blocking else "observation",
            confidence="high" if blocking else "low",
            evidence=error,
            recommendation=message,
            blocking=blocking,
        )],
        error=error,
        blocking=role.blocking,
        skipped=skipped,
        base_perspective_value=role.base_perspective.value if role.base_perspective else "",
    )


def _parse(role: ReviewRoleSpec, raw: str, artifacts) -> RoleReviewOutcome:
    try:
        data = _extract_json(raw)
        verdict = str(data.get("verdict") or "").upper()
        if str(data.get("role_id") or "") != role.role_id or verdict not in {"PASS", "FAIL"}:
            raise ValueError("invalid role_id or verdict")
        items = data.get("suggestions") or []
        if not isinstance(items, list):
            raise ValueError("suggestions must be an array")
        suggestions: list[RoleSuggestion] = []
        for item in items[:max(1, role.max_suggestions)]:
            if not isinstance(item, dict) or not str(item.get("recommendation") or "").strip():
                continue
            severity = str(item.get("severity") or "observation").lower()
            confidence = str(item.get("confidence") or "medium").lower()
            evidence = str(item.get("evidence") or "").strip()
            blocking = role.blocking and severity in {"blocker", "major"} and confidence == "high" and bool(evidence)
            suggestions.append(RoleSuggestion(
                severity=severity if evidence or severity not in {"blocker", "major"} else "observation",
                confidence=confidence,
                evidence=evidence,
                recommendation=str(item["recommendation"]).strip(),
                target=str(item.get("target") or "").strip(),
                blocking=blocking,
            ))
        if verdict == "FAIL" and not suggestions:
            raise ValueError("FAIL lacks actionable suggestions")
        goal = str(data.get("goal_verdict") or "").upper()
        confidence = str(data.get("goal_confidence") or "").lower()
        evidence = str(data.get("evidence_summary") or "").strip()
        if role.role_id == COMPLETION_ROLE:
            objective_failure = (
                (artifacts.verify_command and artifacts.verify_passed is not True)
                or any(not artifacts.criteria_satisfied.get(i, False) for i, _ in enumerate(artifacts.acceptance_criteria or ()))
            )
            if goal not in {"GOAL_MET", "GOAL_NOT_MET"} or (
                verdict == "PASS" and (goal != "GOAL_MET" or not evidence or objective_failure)
            ):
                raise ValueError("completion verdict contradicts objective evidence")
        return RoleReviewOutcome(
            role_id=role.role_id,
            role_display_name=role.display_name,
            role_category=role.category,
            passed=verdict == "PASS" and not any(item.blocking for item in suggestions),
            summary=str(data.get("summary") or evidence or ""),
            suggestions=suggestions,
            raw_preview=(raw or "")[:500],
            blocking=role.blocking,
            base_perspective_value=role.base_perspective.value if role.base_perspective else "",
            goal_verdict=goal,
            goal_confidence=confidence,
            goal_evidence=evidence,
        )
    except Exception as exc:
        return _failure(role, "审查输出结构无效，自动修复耗尽后失败关闭", f"format_failure:{get_error_detail(exc)}")


def _assign(ctx: ReviewContext, roles: list[ReviewRoleSpec]) -> dict[str, ReviewAgentBinding]:
    agents = normalize_review_agents(ctx.review_agents)
    if not agents:
        return {}
    indices, pool = list(range(len(roles))), list(agents)
    if ctx.review_agent_rng:
        ctx.review_agent_rng.shuffle(indices)
        ctx.review_agent_rng.shuffle(pool)
    return {roles[index].role_id: pool[offset % len(pool)] for offset, index in enumerate(indices)}


def _candidates(ctx: ReviewContext, primary: ReviewAgentBinding | None) -> list[tuple[str, str | None]]:
    result: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str]] = set()

    def add(agent, model):
        key = (str(agent or "coco").strip() or "coco", str(model or "").strip())
        if key not in seen:
            seen.add(key)
            result.append((key[0], key[1] or None))

    if primary:
        add(primary.agent_type, None if primary.use_default_model else primary.model_name)
    for binding in normalize_review_agents(ctx.review_agents):
        add(binding.agent_type, None if binding.use_default_model else binding.model_name)
    add(ctx.agent_type, ctx.model_name)
    return result


def _run_prompt(ctx, role, binding, prompt, on_event, timeout) -> str:
    if callable(ctx.send_prompt_with_retry_fn) and ctx.session is None and not ctx.review_agents:
        response = ctx.send_prompt_with_retry_fn(prompt, on_event=on_event, timeout=timeout)
        return str(getattr(response, "text", response) or "")
    last_error = None
    startup_timeout = float(getattr(ctx.settings, "spec_review_startup_timeout", 30) or 30)
    for agent, model in _candidates(ctx, binding):
        for attempt in range(STARTUP_RETRIES + 1):
            if ctx.cancel_event and ctx.cancel_event.is_set():
                raise _ReviewCancelled("cancelled")
            session = EphemeralReviewSession(agent, getattr(ctx.artifacts, "cwd", "") or ".", model, startup_timeout=startup_timeout)
            try:
                with session as active:
                    response = active.send_prompt(prompt, on_event=on_event, timeout=timeout)
                    return str(getattr(response, "text", response) or "")
            except Exception as exc:
                last_error = exc
                try:
                    exc.startup_failed = not session.session_started  # type: ignore[attr-defined]
                    exc.startup_elapsed_s = session.startup_elapsed_s  # type: ignore[attr-defined]
                    exc.startup_timeout_s = startup_timeout  # type: ignore[attr-defined]
                except Exception:
                    pass
                if not session.session_started and attempt < STARTUP_RETRIES:
                    delay = min(2 ** attempt, 4)
                    if ctx.cancel_event and ctx.cancel_event.wait(delay):
                        raise _ReviewCancelled("cancelled") from exc
                    if not ctx.cancel_event:
                        time.sleep(delay)
                    continue
                logger.warning("[SpecReview:%s] %s/%s failed, trying fallback", role.role_id, agent, model or "default")
                break
    raise last_error or RuntimeError("no review backend")


def _retry_event(ctx, status, attempt, maximum, delay=0):
    if ctx.on_retry_status:
        try:
            ctx.on_retry_status(RetryEvent(status=status, attempt=attempt, max_attempts=maximum, delay_sec=delay))
        except Exception:
            logger.debug("review retry callback failed", exc_info=True)


def _run_role(ctx: ReviewContext, role: ReviewRoleSpec, binding: ReviewAgentBinding | None) -> RoleReviewOutcome:
    timeout = float(getattr(ctx.settings, "spec_review_timeout", 240) or 240)
    timeout *= float((getattr(ctx.settings, "spec_review_role_timeout_multipliers", None) or {}).get(role.role_id, 1))
    retries = max(0, int(getattr(ctx.settings, "spec_review_retry_max_attempts", 2) or 0))
    prompt = _build_prompt(role, ctx.artifacts)
    last_outcome = None
    last_error = None
    for attempt in range(retries + 1):
        if ctx.cancel_event and ctx.cancel_event.is_set():
            return _failure(role, "审查已取消", "cancelled")
        chunks: list[str] = []

        def on_event(event):
            if ctx.cancel_event and ctx.cancel_event.is_set():
                raise _ReviewCancelled("cancelled")
            if getattr(event, "text", None):
                chunks.append(str(event.text))

        try:
            runner = ctx.prompt_runner_factory(role) if ctx.prompt_runner_factory else None
            raw = runner(prompt, on_event, timeout) if runner else _run_prompt(ctx, role, binding, prompt, on_event, timeout)
            last_outcome = _parse(role, str(raw or "") or "".join(chunks), ctx.artifacts)
            if not last_outcome.error.startswith("format_failure:"):
                if attempt:
                    _retry_event(ctx, RetryStatus.SUCCEEDED, attempt, retries)
                return last_outcome
            last_error = None
            prompt += "\n上次格式无效。只返回字段齐全的合法 JSON 对象。"
        except _ReviewCancelled:
            return _failure(role, "审查已取消", "cancelled")
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            delay = 0 if last_outcome else min(
                float(getattr(ctx.settings, "spec_review_retry_base_delay", 2) or 2) * (2 ** attempt),
                float(getattr(ctx.settings, "spec_review_retry_max_delay", 30) or 30),
            )
            if delay:
                _retry_event(ctx, RetryStatus.WAITING, attempt + 1, retries, delay)
                if ctx.cancel_event and ctx.cancel_event.wait(delay):
                    return _failure(role, "审查已取消", "cancelled")
                if not ctx.cancel_event:
                    time.sleep(delay)
            _retry_event(ctx, RetryStatus.EXECUTING, attempt + 1, retries)
    _retry_event(ctx, RetryStatus.EXHAUSTED, retries, retries)
    if last_outcome:
        last_outcome.error += f";retry_exhausted={retries}"
        return last_outcome
    detail = ""
    if last_error:
        try:
            diagnostics = ctx.build_review_exception_diagnostics_fn(
                last_error,
                cycle=ctx.cycle,
            )
            if isinstance(diagnostics, dict):
                detail = str(diagnostics.get("error_text") or "").strip()
        except Exception:
            logger.debug(
                "Failed to build review exception diagnostics",
                exc_info=True,
            )
        if not detail:
            detail = get_error_detail(last_error)
    else:
        detail = "unknown"
    timed_out = bool(last_error and classify_timeout(last_error, elapsed_s=getattr(last_error, "startup_elapsed_s", None), timeout_s=getattr(last_error, "startup_timeout_s", None)))
    if not role.blocking:
        return _failure(role, f"{role.display_name}暂时不可用，非阻断角色已跳过", detail, skipped=True)
    return _failure(role, f"{role.display_name}{'超时' if timed_out else '失败'}，自动重试已耗尽", f"{'timeout' if timed_out else 'worker_error'}:{detail}")


def _run_roles(ctx, roles, assignments):
    limit = min(len(roles), max(1, int(getattr(ctx.settings, "spec_review_max_parallel", 3) or 3)))
    outcomes = []
    with ThreadPoolExecutor(max_workers=limit, thread_name_prefix="spec-review-") as pool:
        futures = {pool.submit(_run_role, ctx, role, assignments.get(role.role_id)): role for role in roles}
        for future in as_completed(futures):
            role = futures[future]
            try:
                outcomes.append(future.result())
            except Exception as exc:
                outcomes.append(_failure(role, f"{role.display_name}异常失败", get_error_detail(exc)))
    order = {role.role_id: index for index, role in enumerate(roles)}
    return sorted(outcomes, key=lambda item: order.get(item.role_id, len(order)))


def _aggregate(outcomes):
    names = {item.role_id: item.role_display_name for item in outcomes}
    blocking, observations = {}, {}
    rank = {"blocker": 3, "major": 2, "minor": 1, "observation": 0}
    for outcome in outcomes:
        for suggestion in outcome.suggestions:
            key = suggestion.normalized_key()
            if not key:
                continue
            bucket = blocking if suggestion.blocking else observations
            existing = bucket.get(key)
            if not existing:
                bucket[key] = AggregatedSuggestion(
                    hashlib.sha256(key.encode()).hexdigest()[:12], suggestion.severity, suggestion.confidence,
                    [outcome.role_id], [suggestion.evidence] if suggestion.evidence else [], suggestion.recommendation,
                    suggestion.target, suggestion.blocking,
                )
            else:
                if outcome.role_id not in existing.role_ids:
                    existing.role_ids.append(outcome.role_id)
                if suggestion.evidence and suggestion.evidence not in existing.evidence:
                    existing.evidence.append(suggestion.evidence)
                if rank.get(suggestion.severity, 0) > rank.get(existing.severity, 0):
                    existing.severity = suggestion.severity
    return AggregatedReview(list(blocking.values()), list(observations.values()), names)


def _perspective(outcome):
    if outcome.base_perspective_value:
        return ReviewPerspective(outcome.base_perspective_value)
    if outcome.role_category in {"security", "api", "performance", "software"}:
        return ReviewPerspective.ARCHITECT
    if outcome.role_category in {"design", "ux"}:
        return ReviewPerspective.DESIGNER
    if outcome.role_category in {"writing", "research", "domain", "completion_control"}:
        return ReviewPerspective.PRODUCT
    return ReviewPerspective.TESTER


def validate_completion_gate_outcomes(outcomes):
    matches = [item for item in outcomes if item.role_id == COMPLETION_ROLE]
    if len(matches) != 1:
        return False, "completion_control_missing" if not matches else "completion_control_duplicate", None
    item = matches[0]
    if item.error.startswith("format_failure:") or item.goal_verdict not in {"GOAL_MET", "GOAL_NOT_MET"}:
        return False, "completion_control_invalid", item
    return True, "", item


def _build_result(outcomes, cycle, completion_enabled):
    aggregated = _aggregate(outcomes)
    blocking_by_role, observation_by_role = {}, {}
    for item in aggregated.blocking_suggestions:
        for role_id in item.role_ids:
            blocking_by_role.setdefault(role_id, []).append(item.to_repair_text(aggregated.role_names))
    for item in aggregated.observations:
        for role_id in item.role_ids:
            observation_by_role.setdefault(role_id, []).append("observation: " + item.to_repair_text(aggregated.role_names))
    reviews = [PerspectiveReview(
        perspective=_perspective(outcome),
        passed=outcome.passed and not blocking_by_role.get(outcome.role_id),
        suggestions=blocking_by_role.get(outcome.role_id) or observation_by_role.get(outcome.role_id, []),
        summary=outcome.summary,
        role_id=outcome.role_id,
        role_display_name=outcome.role_display_name,
        role_category=outcome.role_category,
        blocking=outcome.blocking,
    ) for outcome in outcomes]
    valid, error, control = validate_completion_gate_outcomes(outcomes) if completion_enabled else (True, "", next((item for item in outcomes if item.role_id == COMPLETION_ROLE), None))
    return AdaptiveReviewResult(
        reviews=reviews,
        iteration=cycle,
        role_outcomes=outcomes,
        aggregated=aggregated,
        blocking_suggestion_hash=aggregated.blocking_hash(),
        blocking_review_passed=all(review.passed for review in reviews) and valid,
        skipped_roles_count=sum(item.skipped for item in outcomes),
        completion_gate_met=bool(valid and control and control.passed and control.goal_verdict == "GOAL_MET"),
        completion_gate_confidence=control.goal_confidence if control else "",
        completion_gate_evidence=control.goal_evidence if control else "",
        completion_gate_enabled=completion_enabled,
        completion_gate_valid=valid,
        completion_gate_error=error,
    )


def _annotate(result, assignments, ctx):
    names = {"coco": "Coco", "codex": "Codex", "aiden": "Aiden", "claude": "Claude", "gemini": "Gemini", "traex": "Traex", "grok": "Grok", "dsh": "DSH"}
    agent = str(ctx.agent_type or "coco")
    default = f"{names.get(agent.lower(), agent.title())} / {ctx.model_name or '默认模型'}"
    for review in result.reviews:
        binding = assignments.get(review.role_id)
        review.review_agent_label = binding.display_label if binding else default
        review.review_agent_type = binding.agent_type if binding else agent
        review.review_model_name = (binding.model_name or "") if binding else str(ctx.model_name or "")


def _record_failure(ctx, outcomes, elapsed):
    failures = [item for item in outcomes if item.blocking and item.error and item.error != "cancelled"]
    all_timeout = bool(failures) and all("timeout" in item.error for item in failures)
    ctx.circuit.on_failure(all_timeout)
    ctx.circuit.last_review_elapsed_ms = elapsed
    threshold = max(1, int(getattr(ctx.settings, "spec_review_failure_max_consecutive", 4) or 4))
    if ctx.circuit.review_failure_consecutive >= threshold:
        ctx.circuit.backoff_level += 1
        base = max(1, int(getattr(ctx.settings, "spec_review_failure_cooldown_cycles", 2) or 2))
        cap = max(base, int(getattr(ctx.settings, "spec_review_failure_max_cooldown_cycles", 12) or 12))
        ctx.circuit.review_circuit_open_until_cycle = ctx.cycle + min(cap, base * 2 ** (ctx.circuit.backoff_level - 1))
    ctx.circuit.last_review_failure_diag = normalize_review_diagnostics({
        "phase": "review", "role": "adaptive_roles", "cycle": ctx.cycle,
        "decision": "review_failed_continue", "fail_reason": "reviewer_errors",
        "err_type": "ReviewTimeout" if all_timeout else "ReviewWorkerError",
        "err_repr": "; ".join(item.error for item in failures[:3]),
        "error_text": f"{len(failures)} blocking reviewers failed",
        "consecutive_failures": ctx.circuit.review_failure_consecutive,
    })


def conduct_review(ctx: ReviewContext) -> AdaptiveReviewResult:
    """Run role planning, parallel review, parsing, aggregation and retry automatically."""
    started = time.monotonic()
    completion = bool(getattr(ctx.settings, "spec_completion_gate_enabled", True))
    if ctx.artifacts is None:
        roles = _missing_artifacts_roles() + [_completion_role()]
        outcomes = [_failure(role, "审查产物缺失，无法验证完成度", "missing_artifacts") for role in roles]
        result = _build_result(outcomes, ctx.cycle, completion)
        _record_failure(ctx, outcomes, 0)
    else:
        roles, role_hash = _plan_roles(ctx)
        if getattr(ctx.settings, "spec_review_failure_circuit_enabled", True) and ctx.circuit.review_circuit_open_until_cycle and ctx.cycle <= ctx.circuit.review_circuit_open_until_cycle:
            ctx.circuit.consecutive_skips += 1
            outcomes = [_failure(role, "审查熔断窗口内失败关闭，后续循环自动重试", "circuit_open") for role in roles]
            result = _build_result(outcomes, ctx.cycle, completion)
            ctx.circuit.last_review_failure_diag = normalize_review_diagnostics({
                "phase": "review", "role": "adaptive_roles", "cycle": ctx.cycle,
                "decision": "review_circuit_open_skip", "fail_reason": "circuit_open",
                "err_type": "ReviewCircuitOpen", "error_text": "review circuit open",
            })
        else:
            assignments = _assign(ctx, roles)
            outcomes = _run_roles(ctx, roles, assignments)
            result = _build_result(outcomes, ctx.cycle, completion)
            result.role_plan_hash = role_hash
            result.blocking_review_passed = result.all_passed and not result.blocking_suggestion_hash
            _annotate(result, assignments, ctx)
            elapsed = int((time.monotonic() - started) * 1000)
            if any(item.blocking and item.error and item.error != "cancelled" for item in outcomes):
                _record_failure(ctx, outcomes, elapsed)
            elif not any(item.error == "cancelled" for item in outcomes):
                ctx.circuit.reset_on_success()
                ctx.circuit.last_review_elapsed_ms = elapsed
    if ctx.on_review_done:
        try:
            ctx.on_review_done(ctx.cycle, result)
        except Exception:
            logger.debug("review completion callback failed", exc_info=True)
    return result


class ReviewOrchestrator:
    """Own the persisted circuit and the only user-driven stop signal."""

    def __init__(self):
        self._circuit = ReviewCircuitState()
        self._cancel_event = threading.Event()

    @property
    def circuit(self):
        return self._circuit

    @property
    def cancel_event(self):
        return self._cancel_event

    def reset_cancel_event(self, *, is_running: bool) -> bool:
        self._cancel_event.clear()
        if not is_running:
            self._cancel_event.set()
            return False
        return True

    def signal_stop(self):
        self._cancel_event.set()

    def to_dict(self):
        return self._circuit.to_dict()

    def restore_circuit(self, data):
        self._circuit = data if isinstance(data, ReviewCircuitState) else ReviewCircuitState.from_dict(data or {})

    @classmethod
    def from_dict(cls, data):
        instance = cls()
        instance.restore_circuit(data)
        return instance
