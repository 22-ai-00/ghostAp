"""ReviewStrategy interface and strategy registry.

Step 2 of the review refactor. Current `conduct_review` is wrapped as
`MultiPerspectiveStrategy` without behavior changes — the engine may
continue to call `conduct_review` directly; this module offers a parallel,
forward-compatible entry point that later steps will swap in.

Strategies:
    * NoReviewStrategy          — always returns empty (fastest path).
    * MultiPerspectiveStrategy  — current 5-perspective review, unchanged.

Future steps will add: LintOnlyStrategy, PerspectiveParallelStrategy,
HeterogeneousAgentsStrategy. They all implement the same ABC so the engine
code path stays the same.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.utils.errors import classify_timeout

from ..agent_session import EphemeralReviewSession
from ..engine_base import ReviewResult
from .adaptive_review import PromptRunner, run_adaptive_role_review_pipeline
from .review import ReviewCircuitState, ReviewPipelineConfig, conduct_review
from .review_agents import ReviewAgentBinding, assign_review_agents, normalize_review_agents
from .review_artifacts import ReviewArtifacts
from .review_roles import ReviewRoleSpec, build_adaptive_role_plan, completion_control_role, fixed_programming_roles

logger = logging.getLogger(__name__)

REVIEW_STARTUP_MAX_RETRIES = 2
REVIEW_STARTUP_RETRY_BASE_DELAY = 2.0
REVIEW_STARTUP_RETRY_MAX_DELAY = 5.0


def _annotate_startup_failure(exc: Exception, elapsed_s: float, timeout_s: float) -> Exception:
    """Attach startup metadata to an exception for downstream classification."""
    try:
        exc.startup_elapsed_s = elapsed_s  # type: ignore[attr-defined]
        exc.startup_timeout_s = timeout_s  # type: ignore[attr-defined]
        exc.startup_failed = True  # type: ignore[attr-defined]
    except Exception:
        pass
    return exc


def _run_with_startup_retry(
    role: ReviewRoleSpec,
    agent_type: str,
    model_name: str | None,
    prompt: str,
    on_event: Optional[Callable],
    timeout: float,
    startup_timeout: float,
    cwd: str,
) -> str:
    """Run a prompt with startup-phase retries.

    Session creation failures (detected via the EphemeralReviewSession timing
    plus error content) are retried with exponential backoff.  Execution-phase
    failures (after the session is up) are raised immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(REVIEW_STARTUP_MAX_RETRIES + 1):
        eph = EphemeralReviewSession(
            agent_type,
            cwd,
            model_name,
            startup_timeout=startup_timeout,
        )
        try:
            with eph as session:
                res = session.send_prompt(prompt, on_event=on_event, timeout=timeout)
                return str(getattr(res, "text", res) or "")
        except Exception as exc:
            last_exc = exc
            elapsed = getattr(eph, "startup_elapsed_s", 0.0)
            is_startup_failure = not eph.session_started
            if not is_startup_failure:
                raise
            _annotate_startup_failure(exc, elapsed, startup_timeout)
            if attempt >= REVIEW_STARTUP_MAX_RETRIES:
                raise
            delay = min(
                REVIEW_STARTUP_RETRY_BASE_DELAY * (2 ** attempt),
                REVIEW_STARTUP_RETRY_MAX_DELAY,
            )
            logger.warning(
                "[ReviewStrategy:adaptive_roles] role=%s startup failed on %s/%s "
                "(attempt %d/%d, elapsed=%.1fs), retrying in %.1fs",
                role.role_id,
                agent_type,
                model_name or "default",
                attempt + 1,
                REVIEW_STARTUP_MAX_RETRIES + 1,
                elapsed,
                delay,
            )
            time.sleep(delay)
    if last_exc:
        raise last_exc
    return ""

__all__ = [
    "ReviewContext",
    "ReviewStrategy",
    "NoReviewStrategy",
    "MultiPerspectiveStrategy",
    "AdaptiveRoleReviewStrategy",
    "select_review_strategy",
]


@dataclass
class ReviewContext:
    """Inputs a strategy needs to conduct a review.

    `artifacts` is optional in this step — the current MultiPerspectiveStrategy
    still uses the live session's in-memory context. Later strategies will
    require `artifacts` to be populated.
    """

    cycle: int
    session: Any
    settings: Any
    project: Any
    send_prompt_with_retry_fn: Callable
    build_review_exception_diagnostics_fn: Callable[..., dict]
    circuit: ReviewCircuitState
    on_review_done: Optional[Callable] = None
    artifacts: Optional[ReviewArtifacts] = None
    cancel_event: Optional[threading.Event] = None
    on_retry_status: Optional[Callable[[str], None]] = None
    agent_type: str = "coco"
    model_name: Optional[str] = None
    prompt_runner_factory: Optional[Callable[[ReviewRoleSpec], PromptRunner]] = None
    role_plan_override: Optional[list[ReviewRoleSpec]] = None
    review_agents: Optional[list[ReviewAgentBinding]] = None
    review_agent_rng: Any = None


class ReviewStrategy(ABC):
    """Strategy contract — every review variant implements this."""

    name: str = "base"

    @abstractmethod
    def run(self, ctx: ReviewContext) -> ReviewResult:
        ...


class NoReviewStrategy(ReviewStrategy):
    """Skip review entirely. Useful for fast-iteration prototypes."""

    name = "none"

    def run(self, ctx: ReviewContext) -> ReviewResult:
        result = ReviewResult(iteration=ctx.cycle)
        if ctx.on_review_done:
            try:
                ctx.on_review_done(ctx.cycle, result)
            except Exception as e:
                logger.debug("[ReviewStrategy:none] on_review_done raised: %s", repr(e))
        return result


class MultiPerspectiveStrategy(ReviewStrategy):
    """Current production behavior — 5 perspectives in one ACP prompt.

    Thin wrapper over `conduct_review`. Preserves every side effect
    (circuit breaker, diagnostics, lint fallback) exactly as before.
    """

    name = "multi_perspective"

    def run(self, ctx: ReviewContext) -> ReviewResult:
        result = conduct_review(
            pipeline_cfg=ReviewPipelineConfig(
                settings=ctx.settings,
                circuit=ctx.circuit,
                cycle=ctx.cycle,
                session=ctx.session,
                project=ctx.project,
                send_prompt_with_retry_fn=ctx.send_prompt_with_retry_fn,
                build_review_exception_diagnostics_fn=ctx.build_review_exception_diagnostics_fn,
                on_review_done=ctx.on_review_done,
                cancel_event=ctx.cancel_event,
                on_retry_status=ctx.on_retry_status,
                agent_type=ctx.agent_type,
                model_name=ctx.model_name,
            ),
        )
        _annotate_review_result_agents(result, {}, ctx)
        return result


class AdaptiveRoleReviewStrategy(ReviewStrategy):
    """Adaptive task-aware review with parallel per-role workers."""

    name = "adaptive_roles"

    def run(self, ctx: ReviewContext) -> ReviewResult:
        if not ctx.artifacts:
            logger.warning("[ReviewStrategy:adaptive_roles] missing artifacts, falling back to multi_perspective")
            return MultiPerspectiveStrategy().run(ctx)

        if (
            getattr(ctx.settings, "spec_review_failure_circuit_enabled", False)
            and int(ctx.circuit.review_circuit_open_until_cycle or 0)
            and int(ctx.cycle or 0) <= int(ctx.circuit.review_circuit_open_until_cycle or 0)
        ):
            return MultiPerspectiveStrategy().run(ctx)

        try:
            role_plan = build_adaptive_role_plan(
                ctx.artifacts,
                dynamic_roles_enabled=bool(getattr(ctx.settings, "spec_review_dynamic_roles_enabled", True)),
                dynamic_roles_max=int(getattr(ctx.settings, "spec_review_dynamic_roles_max", 3) or 3),
                total_roles_max=int(getattr(ctx.settings, "spec_review_total_roles_max", 8) or 8),
            )
            roles = list(ctx.role_plan_override or role_plan.roles)
            role_plan_hash = role_plan.blocking_role_hash()
        except Exception as e:
            logger.warning("[ReviewStrategy:adaptive_roles] role planning failed, using fixed programming roles: %s", repr(e))
            fallback_roles = fixed_programming_roles()
            fallback_roles.append(completion_control_role())
            roles = list(ctx.role_plan_override or fallback_roles)
            role_plan_hash = self._hash_roles(roles)
        if ctx.role_plan_override:
            role_plan_hash = self._hash_roles(roles)

        review_agents = normalize_review_agents(ctx.review_agents)
        role_agent_map = assign_review_agents(roles, review_agents, rng=ctx.review_agent_rng)
        prompt_runner_factory = ctx.prompt_runner_factory or self._build_ephemeral_prompt_runner_factory(
            ctx,
            role_agent_map=role_agent_map,
        )
        result = run_adaptive_role_review_pipeline(
            ctx.artifacts,
            roles,
            prompt_runner_factory=prompt_runner_factory,
            iteration=ctx.cycle,
            max_parallel=int(getattr(ctx.settings, "spec_review_max_parallel", 3) or 3),
            timeout=float(getattr(ctx.settings, "spec_review_timeout", 240) or 240),
            role_timeout_multipliers=getattr(ctx.settings, "spec_review_role_timeout_multipliers", None),
            completion_gate_enabled=bool(
                getattr(ctx.settings, "spec_completion_gate_enabled", True)
            ),
            format_retry_max_attempts=max(
                0,
                int(getattr(ctx.settings, "spec_review_retry_max_attempts", 2) or 0),
            ),
        )
        result.role_plan_hash = role_plan_hash
        result.blocking_suggestion_hash = result.aggregated.blocking_hash() if result.aggregated else ""
        result.blocking_review_passed = result.all_passed and not result.blocking_suggestion_hash
        _annotate_review_result_agents(result, role_agent_map, ctx)
        ctx.circuit.reset_on_success()

        if ctx.on_review_done:
            try:
                ctx.on_review_done(ctx.cycle, result)
            except Exception as e:
                logger.debug("[ReviewStrategy:adaptive_roles] on_review_done raised: %s", repr(e))
        return result

    def _build_ephemeral_prompt_runner_factory(
        self,
        ctx: ReviewContext,
        *,
        role_agent_map: dict[str, ReviewAgentBinding] | None = None,
    ) -> Callable[[ReviewRoleSpec], PromptRunner]:
        cwd = (ctx.artifacts.cwd if ctx.artifacts else "") or "."
        role_agent_map = role_agent_map or {}
        review_startup_timeout = float(getattr(ctx.settings, "spec_review_startup_timeout", 30) or 30)

        def _factory(role: ReviewRoleSpec) -> PromptRunner:
            def _runner(prompt: str, on_event: Optional[Callable] = None, timeout: float = 240.0) -> str:
                if callable(ctx.send_prompt_with_retry_fn) and ctx.session is None and not role_agent_map:
                    res = ctx.send_prompt_with_retry_fn(prompt, on_event=on_event, timeout=timeout)
                    return str(getattr(res, "text", res) or "")
                binding = role_agent_map.get(role.role_id)
                candidates = self._review_agent_fallback_candidates(ctx, primary=binding)
                last_exc: Exception | None = None
                for index, (agent_type, model_name) in enumerate(candidates):
                    try:
                        return _run_with_startup_retry(
                            role,
                            agent_type,
                            model_name,
                            prompt,
                            on_event,
                            timeout,
                            review_startup_timeout,
                            cwd,
                        )
                    except Exception as exc:
                        last_exc = exc
                        has_fallback = index < len(candidates) - 1
                        is_timeout = classify_timeout(
                            exc,
                            elapsed_s=getattr(exc, "startup_elapsed_s", None),
                            timeout_s=getattr(exc, "startup_timeout_s", None),
                        )
                        if is_timeout and has_fallback:
                            next_agent, next_model = candidates[index + 1]
                            logger.warning(
                                "[ReviewStrategy:adaptive_roles] role=%s timed out on %s/%s, falling back to %s/%s",
                                role.role_id,
                                agent_type,
                                model_name or "default",
                                next_agent,
                                next_model or "default",
                            )
                            continue
                        raise
                if last_exc:
                    raise last_exc
                return ""

            return _runner

        return _factory

    @staticmethod
    def _review_agent_fallback_candidates(
        ctx: ReviewContext,
        *,
        primary: ReviewAgentBinding | None,
    ) -> list[tuple[str, str | None]]:
        candidates: list[tuple[str, str | None]] = []
        seen: set[tuple[str, str]] = set()

        def add(agent_type: str | None, model_name: str | None) -> None:
            agent = str(agent_type or "coco").strip() or "coco"
            model = str(model_name or "").strip()
            key = (agent, model)
            if key in seen:
                return
            seen.add(key)
            candidates.append((agent, model_name))

        if primary is not None:
            add(primary.agent_type, primary.model_name)
        else:
            add(ctx.agent_type, ctx.model_name)
        for binding in normalize_review_agents(ctx.review_agents):
            add(binding.agent_type, binding.model_name)
        add(ctx.agent_type, ctx.model_name)
        return candidates

    @staticmethod
    def _hash_roles(roles: list[ReviewRoleSpec]) -> str:
        import hashlib
        import json

        payload = [r.to_dict() for r in roles if getattr(r, "blocking", True)]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _default_agent_label(agent_type: str, model_name: str | None) -> str:
    normalized = str(agent_type or "coco").strip()
    display = normalized
    display = {
        "coco": "Coco",
        "codex": "Codex",
        "aiden": "Aiden",
        "claude": "Claude",
        "gemini": "Gemini",
    }.get(normalized.lower(), normalized.title())
    model = str(model_name or "").strip() or "默认模型"
    return f"{display} / {model}"


def _annotate_review_result_agents(
    result: ReviewResult,
    role_agent_map: dict[str, ReviewAgentBinding],
    ctx: ReviewContext,
) -> None:
    """Attach actual tool/model assignment to each review for UI rendering."""
    if not result or not getattr(result, "reviews", None):
        return
    default_label = _default_agent_label(ctx.agent_type, ctx.model_name)
    for review in result.reviews:
        binding = role_agent_map.get(getattr(review, "role_id", "") or "")
        if binding is not None:
            review.review_agent_label = binding.display_label
            review.review_agent_type = binding.agent_type
            review.review_model_name = binding.model_name or ""
            continue
        review.review_agent_label = default_label
        review.review_agent_type = str(ctx.agent_type or "coco")
        review.review_model_name = str(ctx.model_name or "")


_STRATEGY_REGISTRY: dict[str, type[ReviewStrategy]] = {
    NoReviewStrategy.name: NoReviewStrategy,
    MultiPerspectiveStrategy.name: MultiPerspectiveStrategy,
    AdaptiveRoleReviewStrategy.name: AdaptiveRoleReviewStrategy,
}


def select_review_strategy(settings) -> ReviewStrategy:
    """Pick a strategy by `settings.spec_review_strategy` (default: adaptive_roles).

    Unknown names fall back to MultiPerspectiveStrategy with a warning so
    misconfig never breaks cycles.
    """
    name = str(getattr(settings, "spec_review_strategy", "") or "adaptive_roles").strip().lower()
    cls = _STRATEGY_REGISTRY.get(name)
    if cls is None:
        logger.warning(
            "[ReviewStrategy] unknown strategy=%r, falling back to multi_perspective",
            name,
        )
        cls = MultiPerspectiveStrategy
    return cls()
