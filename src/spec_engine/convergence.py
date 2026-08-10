"""Convergence detection and continuation policy for SpecEngine."""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from ..utils.review_diagnostics import normalize_review_diagnostics
from .models import SpecCycleMetrics, SpecProject, SpecWorkItemStatus

logger = logging.getLogger(__name__)


@dataclass
class ContinuationPolicy:
    max_cycles: int
    infinite_mode: bool = False
    disable_convergence: bool = False
    disable_early_stop: bool = False
    min_cycles: int = 1

    def should_stop(
        self,
        cycle_num: int,
        all_satisfied: bool,
        review_passed: bool,
        converged: bool,
        metrics: SpecCycleMetrics,
        backlog_stuck: bool = False,
        ignore_backlog: bool = True,
    ) -> Optional[str]:
        if cycle_num < self.min_cycles and cycle_num < self.max_cycles:
            return None

        if all_satisfied and review_passed:
            # Guard: if ALL criteria became satisfied in a single evaluation
            # (0% → 100% jump) and there are multiple criteria, require one more
            # confirming cycle. This prevents a single optimistic first-cycle
            # evaluation from causing premature termination when there are
            # dozens of goals/criteria.
            if (
                metrics.new_satisfied > 0
                and metrics.total_criteria > 1
                and metrics.new_satisfied >= metrics.total_criteria
            ):
                return None
            if not ignore_backlog and metrics.backlog_pending > 0:
                return None
            return "success"

        if self.infinite_mode:
            return None

        if (not self.disable_convergence) and converged:
            return "converged"

        if backlog_stuck and cycle_num >= 3:
            return "backlog_stuck"

        if self.disable_early_stop:
            return None

        if (
            (not all_satisfied)
            and metrics.improvement_space <= 0.2
            and metrics.backlog_pending == 0
            and metrics.new_satisfied == 0
            and cycle_num >= 3
        ):
            return "converged"
        return None


def detect_convergence(
    project: SpecProject,
    *,
    convergence_window: int,
    review_enabled: bool,
    tolerance: int = 0,
) -> bool:
    if not project:
        return False

    window = int(convergence_window or 0)
    if window < 2:
        return False
    if len(project.cycles) < window:
        return False

    if project.criteria_tracker.is_all_satisfied:
        return False

    recent = project.cycles[-window:]

    tracker = project.criteria_tracker
    counts: list[int] = []
    for c in recent:
        satisfied_now = 0
        for idx in range(len(tracker.criteria)):
            at = tracker.satisfied_at_iteration.get(idx)
            if at is not None and at <= c.cycle_number:
                satisfied_now += 1
        counts.append(satisfied_now)

    if max(counts) - min(counts) > tolerance:
        return False

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    suggestion_sets: list[frozenset[str]] = []
    if not review_enabled:
        suggestion_sets = [frozenset()] * window
    else:
        for c in recent:
            if c.review_result is None:
                return False
            # review 异常（timeout 等）产生的 fallback suggestions 是固定模板文本，
            # 连续异常会导致 suggestion 集合完全相同而误判为收敛。
            # 有异常轮次时不参与收敛判定。
            if str(c.review_decision or "").startswith("review_failed"):
                return False
            ss: set[str] = set()
            for pr in c.review_result.failed_perspectives:
                for s in pr.suggestions:
                    ns = _norm(str(s))
                    if ns:
                        ss.add(ns)
            suggestion_sets.append(frozenset(ss))

    if len(set(suggestion_sets)) != 1:
        return False

    # Suppress convergence if recent cycles show productive work (file changes
    # or tool calls) — criteria evaluation may lag behind actual progress.
    for c in recent:
        tool_count = getattr(c, "tool_call_count", 0) or 0
        modified = getattr(c, "modified_files", None) or []
        if tool_count > 0 and len(modified) > 0:
            return False

    return True


def detect_backlog_stuck(project: SpecProject, *, window: int = 3) -> bool:
    if not project or window < 1:
        return False
    history = project.metrics_history
    if len(history) < window:
        return False
    recent = history[-window:]
    if all(m.backlog_pending == 0 for m in recent):
        return False
    return all(m.backlog_pending >= recent[0].backlog_pending for m in recent[1:])


def update_review_pass_streak(
    project: SpecProject,
    review_result,
    *,
    all_satisfied: bool,
    review_passed: bool,
    required: int,
) -> bool:
    """Update project-level consecutive blocking-free review pass state."""
    required = max(1, int(required or 1))
    role_plan_hash = str(getattr(review_result, "role_plan_hash", "") or "")
    blocking_hash = str(getattr(review_result, "blocking_suggestion_hash", "") or "")
    blocking_passed = bool(getattr(review_result, "blocking_review_passed", review_passed))

    current_role_hash = str(getattr(project, "last_review_role_plan_hash", "") or "")
    if not all_satisfied or not review_passed or not blocking_passed or blocking_hash:
        project.review_pass_streak = 0
        project.last_review_role_plan_hash = role_plan_hash or current_role_hash
        project.last_review_blocking_suggestion_hash = blocking_hash
        return False

    if role_plan_hash and current_role_hash and role_plan_hash != current_role_hash:
        project.review_pass_streak = 0

    project.review_pass_streak = int(project.review_pass_streak or 0) + 1
    if role_plan_hash:
        project.last_review_role_plan_hash = role_plan_hash
    project.last_review_blocking_suggestion_hash = ""
    return project.review_pass_streak >= required


def compute_cycle_metrics(
    cycle,
    project: SpecProject,
    *,
    criteria_weight: float = 0.8,
    review_weight: float = 0.2,
) -> SpecCycleMetrics:
    if not project:
        return SpecCycleMetrics(
            cycle_number=cycle.cycle_number,
            satisfied_count=0,
            total_criteria=0,
            new_satisfied=0,
            review_suggestions=0,
            backlog_pending=0,
            goal_attainment=0.0,
            improvement_space=0.0,
        )

    tracker = project.criteria_tracker
    satisfied = tracker.satisfied_count
    total = tracker.total_count

    prev_satisfied = 0
    if project.metrics_history:
        prev_satisfied = project.metrics_history[-1].satisfied_count
    new_satisfied = max(0, satisfied - prev_satisfied)

    review_suggestions = 0
    if cycle.review_result:
        review_suggestions = int(cycle.review_result.total_suggestions)

    review_decision = str(cycle.review_decision or "")
    review_failed = bool(review_decision) and review_decision.startswith("review_failed")
    review_exception_type = ""
    review_error_text = ""
    try:
        diag = cycle.review_diagnostics
        if isinstance(diag, dict):
            d = normalize_review_diagnostics(diag)
            review_exception_type = str(d.get("err_type") or "")
            review_error_text = str(d.get("error_text") or "")
    except Exception:
        logger.debug("review diagnostics extraction failed", exc_info=True)

    backlog_pending = sum(1 for w in project.work_items if w.status == SpecWorkItemStatus.PENDING)

    criteria_ratio = (satisfied / total) if total else 0.0
    review_ratio = 1.0 if (cycle.review_result and cycle.review_result.all_passed) else 0.0
    goal_attainment = min(1.0, max(0.0, criteria_ratio * criteria_weight + review_ratio * review_weight))

    improvement_space = 0.0
    if new_satisfied > 0:
        improvement_space = 1.0
    elif review_suggestions > 0:
        improvement_space = 0.5
    elif backlog_pending > 0:
        improvement_space = 0.15
    else:
        improvement_space = 0.1

    termination_hint = ""
    if goal_attainment >= 0.999 and improvement_space <= 0.2:
        termination_hint = "可终止：目标达成度高且优化空间小"

    return SpecCycleMetrics(
        cycle_number=cycle.cycle_number,
        satisfied_count=satisfied,
        total_criteria=total,
        new_satisfied=new_satisfied,
        review_suggestions=review_suggestions,
        backlog_pending=backlog_pending,
        goal_attainment=goal_attainment,
        improvement_space=improvement_space,
        termination_hint=termination_hint,
        review_failed=review_failed,
        review_decision=review_decision,
        review_exception_type=review_exception_type,
        review_error_text=review_error_text,
    )
