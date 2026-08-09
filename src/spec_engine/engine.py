"""ACP-driven Spec Engine — structured methodology with iterative review.

Follows spec-kit methodology: each cycle progresses through
spec → plan → task → build → review. Review suggestions feed back
as input for the next cycle. Terminates when all criteria are satisfied
and all review perspectives pass.
"""

import contextlib
import json
import logging
import os
import threading
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import ValidationError

from ..acp import ACPEvent
from ..agent_session import create_engine_session
from ..engine_base import (
    BaseEngine,
    EngineRunState,
    PerspectiveReview,
    ReviewResult,
)
from ..utils.acp_prompt import prompt_via_acp
from ..utils.errors import get_error_detail
from ..utils.retry import RetryPolicy
from ..utils.trace import TraceContext
from .artifacts import (
    merge_acceptance_criteria,
    parse_acceptance_criteria,
    parse_plan_artifact,
    parse_spec_artifact,
    parse_tasks,
)
from .convergence import (
    ContinuationPolicy,
    compute_cycle_metrics,
    detect_backlog_stuck,
    detect_convergence,
    update_review_pass_streak,
)
from .criteria import (
    decompose_criteria_with_llm as _decompose_criteria_with_llm_impl,
)
from .criteria import (
    evaluate_criteria as _evaluate_criteria_impl,
)
from .discovery import (
    build_input_from_spec_file as _build_input_from_spec_file,
)
from .discovery import (
    discover_optimization_questions as _discover_optimization_questions,
)
from .discovery import (
    generate_specs_from_discovery as _generate_specs_from_discovery,
)
from .discovery import (
    pick_next_work_item as _pick_next_work_item,
)
from .discovery import (
    should_load_spec_directly as _should_load_spec_directly,
)
from .models import (
    SpecCycle,
    SpecPhase,
    SpecProject,
    SpecProjectStatus,
    SpecWorkItem,
    SpecWorkItemStatus,
)
from .persistence import (
    append_history_event as _append_history_event,
)
from .persistence import (
    cleanup_generated_specs as _cleanup_generated_specs,
)
from .persistence import (
    cleanup_old_cycle_artifacts as _cleanup_old_cycle_artifacts,
)
from .persistence import (
    get_state_path as _get_state_path,
)
from .persistence import (
    load_engine_state as _load_engine_state,
)
from .persistence import (
    persist_cycle_artifact as _persist_cycle_artifact,
)
from .persistence import (
    persist_state_best_effort as _persist_state_best_effort,
)
from .persistence import (
    project_to_compact_dict as _project_to_compact_dict_impl,
)
from .persistence import (
    read_text_file_best_effort as _read_text_file_best_effort,
)
from .persistence import (
    save_engine_state as _save_engine_state,
)
from .persistence import (
    save_failed_task as _save_failed_task_impl,
)
from .persistence import (
    truncate_output as _truncate_output,
)
from .prompts import (
    build_build_prompt,
    build_plan_prompt,
    build_refinement_input,
    build_spec_prompt,
    build_task_prompt,
    format_criteria_status,
)
from .retry_status import RetryEvent
from .review import (
    ReviewCircuitState,
    ReviewOrchestrator,
    build_review_exception_diagnostics,
    extract_reviews_from_llm_response,
    format_review_exception_log_line,
    normalize_review_diagnostics,
    review_result_to_text,
)
from .review import (
    parse_review_output as _parse_review_output_impl,
)
from .review import (
    parse_review_with_llm as _parse_review_with_llm_impl,
)
from .review_agents import ReviewAgentBinding, normalize_review_agents
from .review_strategy import ReviewContext, select_review_strategy
from .session_utils import (
    build_runtime_context as _build_runtime_context,
)
from .session_utils import (
    initialize_model_context as _initialize_model_context,
)
from .session_utils import (
    recreate_session_best_effort as _recreate_session_best_effort,
)
from .session_utils import (
    restore_runtime_context as _restore_runtime_context,
)
from .session_utils import (
    send_prompt_with_retry as _send_prompt_with_retry,
)
from .session_utils import (
    try_switch_model as _try_switch_model,
)
from .storage import state_path_candidates as _state_path_candidates
from .task_persistence import SpecTaskState, delete_task_state
from .tracker import PhaseTracker
from .validation import SpecInput

logger = logging.getLogger(__name__)

VerifyResult = namedtuple("VerifyResult", ["passed", "output"])


@dataclass
class SpecEngineCallbacks:
    """Spec Engine event callbacks."""

    on_analyzing_start: Optional[Callable[[str], None]] = None
    on_analyzing_done: Optional[Callable[[SpecProject], None]] = None
    on_cycle_start: Optional[Callable[[int, int], None]] = None  # (current, max)
    on_phase_start: Optional[Callable[[int, SpecPhase], None]] = None
    on_phase_event: Optional[Callable[[int, SpecPhase, ACPEvent], None]] = None
    on_phase_done: Optional[Callable[[int, SpecPhase, str], None]] = None
    on_review_done: Optional[Callable[[int, ReviewResult], None]] = None
    on_cycle_done: Optional[Callable[[int, SpecCycle], None]] = None
    on_project_done: Optional[Callable[[SpecProject], None]] = None
    on_error: Optional[Callable[[str], None]] = None
    on_phase_retry: Optional[Callable[[int, int, str], None]] = None  # (attempt, max_attempts, detail)
    on_review_retry: Optional[Callable[[int, "RetryEvent"], None]] = None  # (cycle, event)
    on_model_switch: Optional[Callable[[str, str], None]] = None
    on_task_saved: Optional[Callable[[str], None]] = None


class SpecEngine(BaseEngine):
    """ACP-driven structured development engine with iterative review cycles."""

    def __init__(
        self,
        chat_id: str,
        root_path: str,
        agent_type: str = "coco",
        engine_name: str = "Coco",
        model_name: Optional[str] = None,
        *,
        retry_policy: Optional[RetryPolicy] = None,
        create_session_fn: Optional[Callable] = None,
    ):
        super().__init__(chat_id, root_path, agent_type, engine_name, model_name)
        self._project: Optional[SpecProject] = None
        self._user_guidance: list[str] = []
        self._last_review: Optional[ReviewResult] = None
        # success / paused / converged / max_cycles / stopped
        self._termination_reason: Optional[str] = None
        self._resume_meta: Optional[dict] = None
        self._retry_policy = retry_policy or RetryPolicy()
        self._create_session_fn = create_session_fn or create_engine_session
        self._models_tried: list[str] = []
        self._current_model: Optional[str] = None
        self._on_rate_limit: Optional[Callable[[int], None]] = None
        self._saved_task_id: Optional[str] = None
        # Idempotency guard for failed task persistence: avoid saving the same failure multiple times.
        # Format: (cycle_num, phase.value, task_id)
        self._saved_task_signature: Optional[tuple[int, str, str]] = None
        self._review_orchestrator = ReviewOrchestrator()
        self._last_cycle_num: int = 0
        self._last_phase: SpecPhase = SpecPhase.SPEC
        self._review_agent_pool: list[ReviewAgentBinding] = []
        self._last_verify_passed: bool | None = None
        self._last_verify_output: str = ""

    def set_review_agent_pool(self, agents: list[object] | None) -> None:
        """Set optional heterogeneous review agents for adaptive role review."""
        self._review_agent_pool = normalize_review_agents(agents)

    # ------------------------------------------------------------------
    # Compatibility properties: delegate to _review_orchestrator
    # ------------------------------------------------------------------
    @property
    def _review_circuit(self) -> ReviewCircuitState:
        return self._review_orchestrator.circuit

    @_review_circuit.setter
    def _review_circuit(self, value: ReviewCircuitState) -> None:
        self._review_orchestrator.restore_circuit(value)

    @property
    def _review_cancel_event(self) -> threading.Event:
        return self._review_orchestrator.cancel_event

    @property
    def skip_retry_event(self) -> threading.Event:
        """Event to skip retry wait without cancelling the entire retry cycle."""
        return self._review_orchestrator.skip_retry_event

    @property
    def _last_review_failure_diag(self) -> Optional[dict]:
        return self._review_orchestrator.circuit.last_review_failure_diag

    @_last_review_failure_diag.setter
    def _last_review_failure_diag(self, value: Optional[dict]):
        self._review_orchestrator.circuit.last_review_failure_diag = value

    @property
    def _review_failure_consecutive(self) -> int:
        return self._review_orchestrator.circuit.review_failure_consecutive

    @_review_failure_consecutive.setter
    def _review_failure_consecutive(self, value: int):
        self._review_orchestrator.circuit.review_failure_consecutive = value

    @property
    def _review_circuit_open_until_cycle(self) -> int:
        return self._review_orchestrator.circuit.review_circuit_open_until_cycle

    @_review_circuit_open_until_cycle.setter
    def _review_circuit_open_until_cycle(self, value: int):
        self._review_orchestrator.circuit.review_circuit_open_until_cycle = value

    def _wrap_callbacks(self, callbacks: SpecEngineCallbacks) -> SpecEngineCallbacks:
        def _wrap(fn: Optional[Callable[..., None]], name: str) -> Optional[Callable[..., None]]:
            if not fn:
                return None

            def _inner(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logger.warning("[Spec] callback %s 失败: %s", name, e, exc_info=True)
                    return None

            return _inner

        return SpecEngineCallbacks(
            on_analyzing_start=_wrap(callbacks.on_analyzing_start, "on_analyzing_start"),
            on_analyzing_done=_wrap(callbacks.on_analyzing_done, "on_analyzing_done"),
            on_cycle_start=_wrap(callbacks.on_cycle_start, "on_cycle_start"),
            on_phase_start=_wrap(callbacks.on_phase_start, "on_phase_start"),
            on_phase_event=_wrap(callbacks.on_phase_event, "on_phase_event"),
            on_phase_done=_wrap(callbacks.on_phase_done, "on_phase_done"),
            on_review_done=_wrap(callbacks.on_review_done, "on_review_done"),
            on_cycle_done=_wrap(callbacks.on_cycle_done, "on_cycle_done"),
            on_project_done=_wrap(callbacks.on_project_done, "on_project_done"),
            on_error=_wrap(callbacks.on_error, "on_error"),
            on_phase_retry=_wrap(callbacks.on_phase_retry, "on_phase_retry"),
            on_review_retry=_wrap(callbacks.on_review_retry, "on_review_retry"),
            on_model_switch=_wrap(callbacks.on_model_switch, "on_model_switch"),
            on_task_saved=_wrap(callbacks.on_task_saved, "on_task_saved"),
        )

    def _initialize_model_context(self) -> None:
        self._current_model, self._models_tried = _initialize_model_context(self._agent_type)

    @staticmethod
    def _infer_engine_name(agent_type: Optional[str]) -> str:
        normalized = str(agent_type or "").strip().lower()
        if normalized == "claude":
            return "Claude"
        if normalized == "traex":
            return "Traex"
        return "Coco"

    def _build_runtime_context(self) -> dict:
        runtime = _build_runtime_context(
            agent_type=str(self._agent_type or ""),
            engine_name=self.engine_name,
            model_name=self._model_name,
            current_model=self._current_model,
            models_tried=self._models_tried,
            infer_engine_name_fn=self._infer_engine_name,
        )
        if self._review_agent_pool:
            runtime["review_agents"] = [agent.to_dict() for agent in self._review_agent_pool]
        return runtime

    def _restore_runtime_context(
        self,
        runtime_context: Optional[dict],
        *,
        saved_task_id: Optional[str] = None,
        on_rate_limit: Optional[Callable[[int], None]] = None,
    ) -> None:
        runtime = dict(runtime_context or {})
        tentative_agent = str(runtime.get("agent_type") or self._agent_type or "coco").strip().lower() or "coco"
        result = _restore_runtime_context(
            runtime_context,
            agent_type=str(self._agent_type or ""),
            engine_name=self.engine_name,
            model_name=self._model_name,
            current_model=self._current_model,
            models_tried=self._models_tried,
            infer_engine_name_fn=self._infer_engine_name,
            initialize_model_context_fn=lambda: _initialize_model_context(tentative_agent),
            saved_task_id=saved_task_id,
            on_rate_limit=on_rate_limit,
            existing_saved_task_id=self._saved_task_id,
            project=self._project,
        )
        self._agent_type = result["agent_type"]
        self.engine_name = result["engine_name"]
        self._model_name = result["model_name"]
        self._current_model = result["current_model"]
        self._models_tried = result["models_tried"]
        self._on_rate_limit = result["on_rate_limit"]
        self._saved_task_id = result["saved_task_id"]
        self._saved_task_signature = None
        self._review_agent_pool = normalize_review_agents(runtime.get("review_agents"))

    def restore_from_task_state(
        self,
        state: SpecTaskState,
        *,
        on_rate_limit: Optional[Callable[[int], None]] = None,
    ) -> SpecProject:
        if not state.project_snapshot:
            raise ValueError("恢复失败：缺少 project_snapshot")

        project = SpecProject.from_dict(state.project_snapshot)
        project.status = SpecProjectStatus.PAUSED
        project.task_id = state.task_id
        if project.cycles:
            project.cycle_count_total = max(project.cycle_count_total, project.cycles[-1].cycle_number)

        self._project = project
        self._restore_runtime_context(
            state.resolved_runtime_context(),
            saved_task_id=state.task_id,
            on_rate_limit=on_rate_limit,
        )
        return project

    def _send_prompt_with_retry(
        self,
        prompt: str,
        *,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        before_retry: Optional[Callable[[int, Exception], None]] = None,
        total_timeout: Optional[float] = None,
    ):
        return _send_prompt_with_retry(
            self._session, prompt,
            on_event=on_event, timeout=timeout,
            retry_policy=retry_policy, before_retry=before_retry,
            total_timeout=total_timeout,
        )

    def _build_review_exception_diagnostics(self, e: Exception, *, cycle: int) -> dict:
        session_id = ""
        with contextlib.suppress(Exception):  # intentional: defensive attribute access
            session_id = str(getattr(self._session, "session_id", "") or "")
        return build_review_exception_diagnostics(
            e,
            cycle=cycle,
            project_name=(self._project.name or "").strip() if self._project else "",
            chat_id=self.chat_id or "",
            root_path=self.root_path or "",
            agent_type=self._agent_type or "",
            session_id=session_id,
        )

    @classmethod
    def _normalize_review_diagnostics(cls, diag: object) -> dict:
        return normalize_review_diagnostics(diag)

    @classmethod
    def _format_review_exception_log_line(cls, diag: dict, *, diag_json: str) -> str:
        return format_review_exception_log_line(diag, diag_json=diag_json)


    # Main execution
    # ------------------------------------------------------------------

    def _finalize_execution(
        self,
        *,
        max_cycles: int,
        callbacks: SpecEngineCallbacks,
        error: Optional[Exception] = None,
        is_timeout: bool = False,
        label: str = "Spec执行",
    ) -> None:
        """Shared termination logic for execute() and resume().

        Handles either normal termination (reason-based routing) or exception
        recovery (timeout / generic error).
        """
        if error is not None:
            # Exception path
            error_msg = self._format_engine_error(error, label, is_timeout=is_timeout, callbacks=callbacks)
            if self._project:
                self._project.status = SpecProjectStatus.ABORTED
                self._project.completed_at = time.time()
                if not self._saved_task_id:
                    try:
                        self._save_failed_task(error_msg, self._last_cycle_num, self._last_phase, callbacks)
                    except Exception as save_err:
                        logger.warning("[Spec] 异常任务保存失败: %s", save_err, exc_info=True)
            return

        # Normal termination path — route by reason
        reason = self._termination_reason or "max_cycles"
        with self._lock:
            _run_state_snapshot = self._run_state
        if _run_state_snapshot == EngineRunState.STOPPING or reason == "paused":
            self._project.status = SpecProjectStatus.PAUSED
        elif reason == "success":
            self._project.status = SpecProjectStatus.COMPLETED
            self._project.completed_at = time.time()
        else:
            if reason == "converged":
                msg = (
                    f"收敛终止：连续{self.settings.spec_convergence_window}轮无有效改进，"
                    "仍有未满足验收标准或审查未通过"
                )
                self._project.abort(msg)
            elif reason == "backlog_stuck":
                msg = "Backlog 停滞终止：连续多轮 backlog 未消减"
                self._project.abort(msg)
            elif reason == "consecutive_failures":
                n = getattr(self.settings, "spec_max_consecutive_failures", 3)
                msg = f"连续异常终止：{n} 个循环连续因异常失败"
                self._project.abort(msg)
            elif reason == "max_cycles":
                self._handle_max_cycles_termination(max_cycles)
            else:
                msg = f"终止：{reason}"
                self._project.abort(msg)

        if callbacks.on_project_done:
            callbacks.on_project_done(self._project)

    def execute(
        self,
        requirement_text: str,
        callbacks: Optional[SpecEngineCallbacks] = None,
        task_id: Optional[str] = None,
        on_rate_limit: Optional[Callable[[int], None]] = None,
    ) -> SpecProject:
        """Run the spec engine: analyze → cycle(spec→plan→task→build→review) → repeat."""
        callbacks = self._wrap_callbacks(callbacks or SpecEngineCallbacks())
        with self._lock:
            self._run_state = EngineRunState.RUNNING
            self._on_rate_limit = on_rate_limit
            self._saved_task_id = None
            self._saved_task_signature = None
            self._termination_reason = None
        max_cycles = self._resolve_max_cycles(self.settings.spec_max_cycles)

        project_name = os.path.basename(self.root_path) or "spec_project"
        self._project = SpecProject.create(name=project_name, root_path=self.root_path)

        # Initialize TraceContext
        trace_ctx = TraceContext(request_id=task_id or f"spec-{int(time.time())}")
        trace_ctx.__enter__()

        # Validation Gateway
        try:
            SpecInput(requirement_text=requirement_text, task_id=task_id)
        except ValidationError as e:
            # Flatten validation errors to a readable string
            errors = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
            error_msg = f"非法配置参数: {errors}"
            self._project.status = SpecProjectStatus.ABORTED
            self._project.error = error_msg
            self._project.completed_at = time.time()
            logger.error("[Spec:%s] %s", project_name, error_msg)
            if callbacks.on_error:
                callbacks.on_error(error_msg)
            trace_ctx.__exit__(None, None, None)
            return self._project

        self._project.task_id = task_id
        self._project.status = SpecProjectStatus.ANALYZING
        self._project.requirement = requirement_text
        self._last_cycle_num = 0
        self._last_phase = SpecPhase.SPEC

        if callbacks.on_analyzing_start:
            callbacks.on_analyzing_start(requirement_text)

        logger.info(
            "[Spec:%s] 启动, 需求长度=%d, 路径=%s, agent=%s",
            project_name,
            len(requirement_text),
            self.root_path,
            self._agent_type,
        )

        try:
            self._initialize_model_context()

            # Parse requirement from explicit user markers only. Plain-language
            # input starts with a single provisional criterion; the first Spec
            # artifact replaces it with model-derived acceptance criteria.
            criteria = parse_acceptance_criteria(requirement_text)
            self._project.acceptance_criteria = criteria
            self._project.criteria_tracker.init_criteria(criteria)
            self._project.status = SpecProjectStatus.RUNNING
            self._project.started_at = time.time()

            if callbacks.on_analyzing_done:
                callbacks.on_analyzing_done(self._project)

            # Create ACP session
            self._session = self._create_session_fn(
                agent_type=self._agent_type,
                cwd=self.root_path,
                on_rate_limit=on_rate_limit,
                model_name=self._model_name,
            )

            self._last_review = None

            self._termination_reason = self._run_cycle_loop(
                start_cycle=1,
                max_cycles=max_cycles,
                callbacks=callbacks,
                timeout=self.settings.spec_execution_timeout,
                first_raw_input=requirement_text,
            )

            self._finalize_execution(max_cycles=max_cycles, callbacks=callbacks)
            return self._project

        except TimeoutError as e:
            self._finalize_execution(max_cycles=max_cycles, callbacks=callbacks, error=e, is_timeout=True, label="Spec执行")
            return self._project

        except Exception as e:
            logger.exception("Unexpected error in Spec execution")
            self._finalize_execution(max_cycles=max_cycles, callbacks=callbacks, error=e, is_timeout=False, label="Spec执行")
            return self._project

        finally:
            trace_ctx.__exit__(None, None, None)
            self._close_session_safely()
            with self._lock:
                self._run_state = EngineRunState.IDLE

            if self._project and self._project.status == SpecProjectStatus.COMPLETED and self._saved_task_id:
                delete_task_state(self._saved_task_id)
                self._saved_task_id = None
                self._saved_task_signature = None

    # ------------------------------------------------------------------
    # Phase execution
    # ------------------------------------------------------------------
    def _run_phase(
        self,
        cycle_num: int,
        phase: SpecPhase,
        prompt: str,
        callbacks: SpecEngineCallbacks,
        timeout: int,
        _depth: int = 0,
    ) -> str:
        """Execute a single phase: send prompt, collect output, return text."""
        project_name = self._project.name if self._project else "unknown"
        logger.info("[Spec:%s] 循环 %d 阶段 %s 开始", project_name, cycle_num, phase.value)

        if not self._session:
            raise RuntimeError(f"Spec session is None before phase {phase.value} (cycle={cycle_num}), session may have failed to initialize or rebuild")

        if callbacks.on_phase_start:
            callbacks.on_phase_start(cycle_num, phase)

        def _before_retry(attempt: int, error: Exception):
            if attempt > 0:
                self._recreate_session_best_effort()
            if callbacks.on_phase_retry:
                _max = self.settings.spec_max_retries
                callbacks.on_phase_retry(attempt, _max, get_error_detail(error))

        from ..utils.retry import RetryPolicy
        retry_policy = RetryPolicy(
            max_retries=self.settings.spec_max_retries,
            retry_delay=self._retry_policy.retry_delay,
            backoff_multiplier=self._retry_policy.backoff_multiplier
        )

        try:
            tracker = PhaseTracker()

            def on_event(event: ACPEvent):
                try:
                    tracker.process(event)
                    renderer = self._renderer
                    if renderer is not None:
                        renderer.process_event(event)
                    if callbacks.on_phase_event:
                        try:
                            callbacks.on_phase_event(cycle_num, phase, event)
                        except Exception as cb_exc:
                            logger.warning("[Spec] on_phase_event callback failed: %s", get_error_detail(cb_exc), exc_info=True)
                except Exception as exc:
                    logger.warning("[Spec] on_event handler error: %s", get_error_detail(exc), exc_info=True)

            self._send_prompt_with_retry(
                prompt,
                on_event=on_event,
                timeout=timeout,
                retry_policy=retry_policy,
                before_retry=_before_retry,
            )
            output = tracker.text_buffer
            logger.info(
                "[Spec:%s] 循环 %d 阶段 %s 完成, 输出长度=%d",
                project_name, cycle_num, phase.value, len(output),
            )

            # Expose phase stats for cycle-level accumulation
            self._last_phase_stats = {
                "tool_call_count": len(tracker.tool_calls),
                "modified_files": list(tracker.modified_files),
            }

            if callbacks.on_phase_done:
                callbacks.on_phase_done(cycle_num, phase, output)

            return output

        except Exception as e:
            from ..utils.errors import get_error_detail as _ged
            last_error = _ged(e)

            try:
                override_hint = (self.settings.spec_failed_task_id_override or "").strip()
                if (
                    override_hint
                    and phase == SpecPhase.BUILD
                    and "internal error" not in (last_error or "").lower()
                ):
                    last_error = "Internal error"
            except Exception:
                logger.warning("override hint processing failed", exc_info=True)

            # 停止态下（例如服务关闭触发 cancel），phase 异常通常是 session cancel 或进程退出导致，
            # 不应继续触发模型切换或失败任务持久化。
            if self._run_state == EngineRunState.STOPPING:
                reason = last_error or type(e).__name__
                with contextlib.suppress(Exception):  # intentional: defensive string truncation
                    if len(reason) > 200:
                        reason = reason[:200] + "…(truncated)"
                logger.info("[Spec] Phase %s 中断（引擎停止中）: %s", phase.value, reason)
                return ""

            if self._try_switch_model(callbacks):
                if _depth >= 3:
                    raise RuntimeError(
                        f"Phase {phase.value} 模型切换递归超限 (depth={_depth})，停止重试"
                    ) from e
                return self._run_phase(cycle_num, phase, prompt, callbacks, timeout, _depth=_depth + 1)

            task_id = self._save_failed_task(last_error, cycle_num, phase, callbacks)
            err_preview = last_error or ""
            with contextlib.suppress(Exception):  # intentional: defensive string truncation
                if len(err_preview) > 500:
                    err_preview = err_preview[:500] + "…(truncated)"
            logger.error("[Spec] Phase %s 失败 (task_id=%s): %s", phase.value, task_id, err_preview)
            raise RuntimeError(f"Phase {phase.value} 失败，任务已保存(task_id={task_id}): {last_error}") from e

    def _verify_build_result(self) -> VerifyResult:
        if not self._project or not self._project.verify_command:
            return VerifyResult(passed=True, output="")
        from ..sandbox.executor import SandboxExecutor
        executor = SandboxExecutor()
        result = executor.execute(
            self._project.verify_command,
            cwd=self.root_path,
            interactive=False,
            chat_id=self.chat_id,
        )
        output = result.stdout
        if result.stderr:
            output = f"{output}\n{result.stderr}".strip()
        return VerifyResult(passed=result.success, output=output)

    def _try_switch_model(self, callbacks) -> bool:
        switched, new_current, _, self._models_tried, new_session = _try_switch_model(
            agent_type=self._agent_type,
            run_state=self._run_state,
            models_tried=self._models_tried,
            current_model=self._current_model,
            root_path=self.root_path,
            model_name=self._model_name,
            on_rate_limit=getattr(self, "_on_rate_limit", None),
            close_session_fn=self._close_session_safely,
            callbacks=callbacks,
        )
        if switched:
            self._current_model = new_current
            self._session = new_session
        return switched

    def _recreate_session_best_effort(self) -> None:
        new_session = _recreate_session_best_effort(
            agent_type=self._agent_type,
            root_path=self.root_path,
            on_rate_limit=getattr(self, "_on_rate_limit", None),
            current_model=self._current_model,
            model_name=self._model_name,
            close_session_fn=self._close_session_safely,
        )
        if new_session is not None:
            self._session = new_session

    def _save_failed_task(
        self,
        error: str,
        cycle_num: int,
        phase: SpecPhase,
        callbacks,
    ) -> str:
        task_id, new_saved_id, new_sig = _save_failed_task_impl(
            project=self._project,
            root_path=self.root_path,
            chat_id=self.chat_id,
            agent_type=self._agent_type,
            settings=self.settings,
            models_tried=self._models_tried,
            build_runtime_context_fn=self._build_runtime_context,
            project_to_compact_dict_fn=self._project_to_compact_dict,
            saved_task_id=self._saved_task_id,
            saved_task_signature=self._saved_task_signature,
            error=error,
            cycle_num=cycle_num,
            phase=phase,
            callbacks=callbacks,
        )
        if new_saved_id is not None:
            self._saved_task_id = new_saved_id
        if new_sig is not None:
            self._saved_task_signature = new_sig
        return task_id

    # ------------------------------------------------------------------
    # Cycle loop (shared by execute / resume)
    # ------------------------------------------------------------------
    def _accumulate_phase_stats(self, cycle: "SpecCycle", phase_name: str) -> None:
        """Accumulate _last_phase_stats into the current cycle."""
        stats = getattr(self, "_last_phase_stats", None)
        if not stats:
            return
        cycle.tool_call_count += stats.get("tool_call_count", 0)
        new_files = stats.get("modified_files", [])
        if new_files:
            cycle.modified_files = list(set(cycle.modified_files) | set(new_files))
        cycle.phase_tool_stats[phase_name] = stats.get("tool_call_count", 0)
        self._last_phase_stats = None

    def _run_cycle_loop(
        self,
        start_cycle: int,
        max_cycles: int,
        callbacks: SpecEngineCallbacks,
        timeout: int,
        first_raw_input: Optional[str] = None,
    ) -> str:
        """Execute spec cycles. Modifies ``self._project`` in-place.

        Returns a termination reason:
        - success: all criteria satisfied and (review disabled or all PASS)
        - paused: user stopped/pause requested
        - converged: no measurable progress in window
        - max_cycles: hit max_cycles without success
        - stopped: engine stopped without a cycle (edge)
        """
        termination: str = "max_cycles"

        policy = ContinuationPolicy(
            max_cycles=max_cycles,
            infinite_mode=self.settings.spec_infinite_mode,
            disable_convergence=self.settings.spec_disable_convergence,
            disable_early_stop=self.settings.spec_disable_early_stop,
            # Spec mode defaults to at least 2 cycles to ensure discovery;
            # allow overriding via settings for single-cycle tasks/tests.
            min_cycles=max(1, self.settings.spec_min_cycles),
        )

        consecutive_failures = 0
        max_consecutive = getattr(self.settings, "spec_max_consecutive_failures", 3)

        for cycle_num in range(start_cycle, max_cycles + 1):
            if self._run_state != EngineRunState.RUNNING:
                termination = "paused" if self._run_state == EngineRunState.STOPPING else "stopped"
                break

            cycle = SpecCycle(cycle_number=cycle_num)
            self._last_cycle_num = cycle_num

            if callbacks.on_cycle_start:
                callbacks.on_cycle_start(cycle_num, max_cycles)

            try:
                work_item = _pick_next_work_item(self._project, cycle_num)
                spec_input = self._prepare_cycle_input(
                    cycle_num, start_cycle, first_raw_input, work_item,
                )

                # --- SPEC → PLAN → TASK → BUILD → REVIEW ---
                # Per-phase timeout multipliers: BUILD needs more time for
                # code execution; SPEC/PLAN/TASK are fast analysis phases.
                _phase_timeout = {
                    "spec": int(timeout * 0.6),
                    "plan": int(timeout * 0.6),
                    "task": int(timeout * 0.5),
                    "build": int(timeout * 2.5),
                    "review": int(timeout * 0.8),
                }

                spec_output = self._run_spec_phase(
                    cycle_num, cycle, spec_input, work_item, callbacks, _phase_timeout["spec"],
                )
                if self._check_cycle_pause(cycle, cycle_num):
                    termination = "paused"
                    break

                plan_output = self._run_plan_phase(
                    cycle_num, cycle, spec_output, callbacks, _phase_timeout["plan"],
                )
                if self._check_cycle_pause(cycle, cycle_num):
                    termination = "paused"
                    break

                self._run_task_phase(cycle_num, cycle, plan_output, callbacks, _phase_timeout["task"])
                if self._check_cycle_pause(cycle, cycle_num):
                    termination = "paused"
                    break

                self._run_build_phase(cycle_num, cycle, plan_output, callbacks, _phase_timeout["build"])
                if self._check_cycle_pause(cycle, cycle_num):
                    termination = "paused"
                    break

                # Pre-review objective verify (results fed to completion_control)
                self._last_verify_passed = None
                self._last_verify_output = ""
                if (
                    self._project
                    and self._project.verify_command
                    and getattr(self.settings, "spec_objective_verify_enabled", True)
                ):
                    from .criteria import run_objective_verify
                    self._last_verify_passed, self._last_verify_output = run_objective_verify(
                        self._project.verify_command,
                        cwd=self.root_path,
                        timeout=int(getattr(self.settings, "spec_objective_verify_timeout", 300)),
                    )

                review_passed = self._run_review_phase(cycle_num, cycle, callbacks)
                cycle.complete()

            except Exception as cycle_exc:
                should_break, new_termination, consecutive_failures = (
                    self._handle_cycle_exception(
                        cycle, cycle_num, cycle_exc,
                        consecutive_failures, max_consecutive,
                    )
                )
                if should_break:
                    termination = new_termination
                    break
                continue

            # ---- Cycle completed successfully ----
            consecutive_failures = 0
            should_stop, stop_reason = self._finalize_successful_cycle(
                cycle_num, cycle, max_cycles, review_passed, callbacks, policy,
            )
            if should_stop:
                termination = stop_reason
                break

        return termination

    # ------------------------------------------------------------------
    # Cycle-loop helper methods (extracted from _run_cycle_loop)
    # ------------------------------------------------------------------

    def _prepare_cycle_input(
        self,
        cycle_num: int,
        start_cycle: int,
        first_raw_input: Optional[str],
        work_item: Optional[SpecWorkItem],
    ) -> str:
        """Determine the input text for the SPEC phase of a given cycle."""
        requirement = self._project.requirement
        if first_raw_input is not None and cycle_num == start_cycle:
            return first_raw_input
        if work_item and work_item.spec_path:
            return _build_input_from_spec_file(requirement, work_item)
        return build_refinement_input(requirement, self._last_review, self._project)

    def _run_spec_phase(
        self,
        cycle_num: int,
        cycle: SpecCycle,
        spec_input: str,
        work_item: Optional[SpecWorkItem],
        callbacks: SpecEngineCallbacks,
        timeout: int,
    ) -> str:
        """Execute the SPEC phase and return spec output text."""
        cycle.phase = SpecPhase.SPEC
        self._last_phase = cycle.phase
        if work_item and work_item.spec_path and _should_load_spec_directly(work_item):
            # spec 文件本身就是 spec-kit 规格产物：直接加载进入下一阶段
            spec_output = _read_text_file_best_effort(work_item.spec_path)
        else:
            spec_output = self._run_phase(
                cycle_num,
                SpecPhase.SPEC,
                build_spec_prompt(spec_input, self.root_path, self._consume_guidance(), format_criteria_status(self._project)),
                callbacks,
                timeout,
            )
        cycle.spec_content = _truncate_output(spec_output, self.settings)
        self._accumulate_phase_stats(cycle, SpecPhase.SPEC.value)
        if self.settings.spec_persist_phase_artifacts:
            cycle.spec_path = _persist_cycle_artifact(self.root_path, self.settings, self._project, cycle_num, "spec", spec_output, "json")
        cycle.spec_artifact, cycle.spec_artifact_errors = parse_spec_artifact(spec_output)

        if self.settings.spec_persist_every_phase:
            self._persist_state_best_effort()

        if work_item:
            work_item.status = SpecWorkItemStatus.DONE
            work_item.used_in_cycle = cycle_num
            _append_history_event(
                self.root_path, self.settings, self._project,
                "work_item_consumed",
                {
                    "cycle": cycle_num,
                    "item_id": work_item.item_id,
                    "question": work_item.question,
                    "spec_path": work_item.spec_path,
                },
            )

        # If the spec provides better acceptance criteria, merge into project.
        if cycle_num == 1 and cycle.spec_artifact and cycle.spec_artifact.acceptance_criteria:
            merge_acceptance_criteria(self._project, cycle.spec_artifact.acceptance_criteria)

        return spec_output

    def _run_plan_phase(
        self,
        cycle_num: int,
        cycle: SpecCycle,
        spec_output: str,
        callbacks: SpecEngineCallbacks,
        timeout: int,
    ) -> str:
        """Execute the PLAN phase and return plan output text."""
        cycle.phase = SpecPhase.PLAN
        self._last_phase = cycle.phase
        plan_output = self._run_phase(
            cycle_num,
            SpecPhase.PLAN,
            build_plan_prompt(spec_output, self.root_path, spec_artifact=cycle.spec_artifact),
            callbacks,
            timeout,
        )
        cycle.plan_content = _truncate_output(plan_output, self.settings)
        self._accumulate_phase_stats(cycle, SpecPhase.PLAN.value)
        if self.settings.spec_persist_phase_artifacts:
            cycle.plan_path = _persist_cycle_artifact(self.root_path, self.settings, self._project, cycle_num, "plan", plan_output, "json")
        cycle.plan_artifact, cycle.plan_artifact_errors = parse_plan_artifact(plan_output)

        if self.settings.spec_persist_every_phase:
            self._persist_state_best_effort()
        return plan_output

    def _run_task_phase(
        self,
        cycle_num: int,
        cycle: SpecCycle,
        plan_output: str,
        callbacks: SpecEngineCallbacks,
        timeout: int,
    ) -> None:
        """Execute the TASK phase. Populates ``cycle.tasks``."""
        cycle.phase = SpecPhase.TASK
        self._last_phase = cycle.phase
        task_output = self._run_phase(
            cycle_num,
            SpecPhase.TASK,
            build_task_prompt(plan_output, plan_artifact=cycle.plan_artifact),
            callbacks,
            timeout,
        )
        self._accumulate_phase_stats(cycle, SpecPhase.TASK.value)
        parsed_tasks = parse_tasks(task_output)
        cycle.tasks_total = len(parsed_tasks)
        cycle.tasks = parsed_tasks[: self.settings.spec_cycle_tasks_max]
        if self.settings.spec_persist_phase_artifacts:
            cycle.tasks_path = _persist_cycle_artifact(
                self.root_path, self.settings, self._project, cycle_num,
                "tasks",
                json.dumps([t.to_dict() for t in parsed_tasks], ensure_ascii=False, indent=2),
                "json",
            )

        if self.settings.spec_persist_every_phase:
            self._persist_state_best_effort()

    def _run_build_phase(
        self,
        cycle_num: int,
        cycle: SpecCycle,
        plan_output: str,
        callbacks: SpecEngineCallbacks,
        timeout: int,
    ) -> None:
        """Execute the BUILD phase."""
        cycle.phase = SpecPhase.BUILD
        self._last_phase = cycle.phase
        build_output = self._run_phase(
            cycle_num,
            SpecPhase.BUILD,
            build_build_prompt(cycle.tasks, plan_output, self.root_path, self._consume_guidance(), plan_artifact=cycle.plan_artifact),
            callbacks,
            timeout,
        )
        cycle.build_output = _truncate_output(build_output, self.settings)
        self._accumulate_phase_stats(cycle, SpecPhase.BUILD.value)
        if self.settings.spec_persist_phase_artifacts:
            cycle.build_path = _persist_cycle_artifact(self.root_path, self.settings, self._project, cycle_num, "build", build_output, "txt")

        if self.settings.spec_persist_every_phase:
            self._persist_state_best_effort()

    def _run_review_phase(
        self,
        cycle_num: int,
        cycle: SpecCycle,
        callbacks: SpecEngineCallbacks,
    ) -> bool:
        """Execute the REVIEW phase (conditional). Returns whether review passed."""
        review_passed = True
        if self.settings.spec_review_enabled:
            cycle.phase = SpecPhase.REVIEW
            self._last_phase = cycle.phase
            if callbacks.on_phase_start:
                callbacks.on_phase_start(cycle_num, SpecPhase.REVIEW)
            review_result = self._conduct_review(cycle_num, callbacks, cycle_obj=cycle)
            cycle.review_result = review_result
            # best-effort: persist review failure decision/diagnostics for traceability
            diag = self._last_review_failure_diag
            if isinstance(diag, dict) and diag:
                cycle.review_decision = str(diag.get("decision") or "review_failed_continue")
                cycle.review_diagnostics = dict(diag)
            if self.settings.spec_persist_phase_artifacts:
                cycle.review_path = _persist_cycle_artifact(
                    self.root_path, self.settings, self._project, cycle_num, "review", review_result_to_text(review_result), "txt"
                )
            self._last_review = review_result
            review_passed = review_result.all_passed

            if self.settings.spec_persist_every_phase:
                self._persist_state_best_effort()
        return review_passed

    def _check_cycle_pause(self, cycle: SpecCycle, cycle_num: int) -> bool:
        """Check if engine should pause. Marks cycle failed and persists if so."""
        if self._run_state != EngineRunState.RUNNING:
            cycle.fail()
            self._project.cycles.append(cycle)
            self._project.cycle_count_total = max(self._project.cycle_count_total, cycle_num)
            self._persist_state_best_effort()
            return True
        return False

    def _handle_cycle_exception(
        self,
        cycle: SpecCycle,
        cycle_num: int,
        exc: Exception,
        consecutive_failures: int,
        max_consecutive: int,
    ) -> tuple[bool, str, int]:
        """Handle a cycle exception.

        Returns ``(should_break, termination_reason, consecutive_failures)``.
        """
        # If engine is STOPPING (user pause/stop), preserve original pause behavior
        if self._run_state != EngineRunState.RUNNING:
            cycle.fail()
            self._project.cycles.append(cycle)
            self._project.cycle_count_total = max(self._project.cycle_count_total, cycle_num)
            self._persist_state_best_effort()
            return True, "paused", consecutive_failures

        # Digest exception: mark cycle failed, continue to next cycle
        err_detail = get_error_detail(exc)
        cycle.error_message = err_detail
        cycle.fail()

        self._project.cycles.append(cycle)
        self._project.cycle_count_total = max(self._project.cycle_count_total, cycle_num)

        logger.error(
            "[Spec:%s] 循环 %d 异常失败 (%s): %s",
            self._project.name,
            cycle_num,
            type(exc).__name__,
            (err_detail or "")[:500],
            exc_info=True,
        )

        _append_history_event(
            self.root_path, self.settings, self._project,
            "cycle_exception",
            {
                "cycle": cycle_num,
                "exception_type": type(exc).__name__,
                "error": (err_detail or "")[:500],
            },
        )

        self._persist_state_best_effort()

        # Consecutive failure protection
        consecutive_failures += 1
        if consecutive_failures >= max_consecutive:
            logger.error(
                "[Spec:%s] 连续 %d 个循环异常失败，终止引擎",
                self._project.name,
                consecutive_failures,
            )
            return True, "consecutive_failures", consecutive_failures

        # Rebuild session for next cycle
        self._recreate_session_best_effort()
        if not self._session:
            logger.error(
                "[Spec:%s] 循环 %d 异常后 Session 重建失败，下一循环将无法执行",
                self._project.name,
                cycle_num,
            )
        return False, "", consecutive_failures

    def _finalize_successful_cycle(
        self,
        cycle_num: int,
        cycle: SpecCycle,
        max_cycles: int,
        review_passed: bool,
        callbacks: SpecEngineCallbacks,
        policy: ContinuationPolicy,
    ) -> tuple[bool, str]:
        """Post-cycle processing after a successful cycle.

        Returns ``(should_stop, termination_reason)``.
        """
        self._project.cycles.append(cycle)
        self._project.cycle_count_total = max(self._project.cycle_count_total, cycle_num)

        if callbacks.on_cycle_done:
            callbacks.on_cycle_done(cycle_num, cycle)

        logger.info(
            "[Spec:%s] 循环 %d/%d 完成, 审查=%s",
            self._project.name,
            cycle_num,
            max_cycles,
            f"{cycle.review_result.total_suggestions}条建议" if cycle.review_result else "跳过",
        )

        # --- CRITERIA EVALUATION ---
        criteria_result = self._evaluate_criteria(self._project.acceptance_criteria, cycle_num)
        all_satisfied = criteria_result.get("all_satisfied", False)

        # --- POST-CYCLE PROBLEM DISCOVERY + SPEC GENERATION ---
        _backlog_pending = sum(
            1 for w in self._project.work_items
            if w.status == SpecWorkItemStatus.PENDING
        )
        if (
            self.settings.spec_discovery_enabled
            and self._run_state == EngineRunState.RUNNING
            and _backlog_pending < 5
        ):
            discovery = self._discover_optimization_questions(
                cycle_num,
                all_satisfied=all_satisfied,
                backlog_pending=_backlog_pending,
            )
            cycle.discovery_path = _persist_cycle_artifact(
                self.root_path, self.settings, self._project, cycle_num, "discovery", json.dumps(discovery, ensure_ascii=False, indent=2), "json"
            )
            new_items = self._generate_specs_from_discovery(cycle_num, discovery)
            # 防止 backlog 无限制膨胀：只保留最近 N 条（长期任务可配合外部清理）
            if new_items:
                self._project.work_items.extend(new_items)
                self._project.work_items_total = max(self._project.work_items_total, len(self._project.work_items))
                for wi in new_items:
                    _append_history_event(
                        self.root_path, self.settings, self._project,
                        "work_item_generated",
                        {
                            "cycle": cycle_num,
                            "item_id": wi.item_id,
                            "question": wi.question,
                            "spec_path": wi.spec_path,
                        },
                    )

        # --- METRICS SNAPSHOT (monitoring) ---
        metrics = compute_cycle_metrics(cycle, self._project)
        self._project.metrics_history.append(metrics)
        cycle.metrics_path = _persist_cycle_artifact(
            self.root_path, self.settings, self._project, cycle_num, "metrics", json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2), "json"
        )

        _append_history_event(
            self.root_path, self.settings, self._project,
            "cycle_done",
            {
                "cycle": cycle_num,
                "status": cycle.status,
                "satisfied": metrics.satisfied_count,
                "total": metrics.total_criteria,
                "new_satisfied": metrics.new_satisfied,
                "review_suggestions": metrics.review_suggestions,
                "goal_attainment": metrics.goal_attainment,
                "improvement_space": metrics.improvement_space,
                "backlog_pending": metrics.backlog_pending,
            },
        )

        self._persist_state_best_effort()
        _cleanup_old_cycle_artifacts(self.root_path, self.settings, self._project, cycle_num)
        _cleanup_generated_specs(self._project, self.settings)

        # --- TERMINATION CHECK (ContinuationPolicy) ---
        converged = False if policy.disable_convergence else self._detect_convergence()
        if converged:
            logger.info("[Spec:%s] 收敛检测触发, 循环 %d 轮", self._project.name, cycle_num)

        _backlog_stuck = detect_backlog_stuck(
            self._project,
            window=getattr(self.settings, "spec_backlog_stuck_window", 3),
        )
        if _backlog_stuck:
            logger.info("[Spec:%s] backlog stuck 检测触发, 循环 %d 轮", self._project.name, cycle_num)

        effective_review_passed = review_passed
        if self.settings.spec_review_enabled:
            effective_review_passed = update_review_pass_streak(
                self._project,
                cycle.review_result,
                all_satisfied=all_satisfied,
                review_passed=review_passed,
                required=getattr(self.settings, "spec_review_pass_streak_required", 2),
            )
            if review_passed and not effective_review_passed:
                logger.info(
                    "[Spec:%s] 审查通过但连续通过次数不足: %d/%d",
                    self._project.name,
                    int(self._project.review_pass_streak or 0),
                    int(getattr(self.settings, "spec_review_pass_streak_required", 2) or 2),
                )

        # --- COMPLETION GATE (Phase 3): evidence-backed early stop / veto ---
        completion_gate_met = False
        completion_gate_confidence = ""
        completion_gate_enabled = getattr(
            self.settings,
            "spec_completion_gate_enabled",
            True,
        )
        if cycle.review_result:
            from .adaptive_review import (
                AdaptiveReviewResult,
                validate_completion_gate_outcomes,
            )
            if isinstance(cycle.review_result, AdaptiveReviewResult):
                if completion_gate_enabled:
                    (
                        completion_gate_valid,
                        completion_gate_error,
                        _completion_outcome,
                    ) = validate_completion_gate_outcomes(
                        cycle.review_result.role_outcomes
                    )
                else:
                    completion_gate_valid = True
                    completion_gate_error = ""

                manual_confirmation_reason = str(
                    cycle.review_result.manual_confirmation_reason or ""
                )
                if completion_gate_enabled and not completion_gate_valid:
                    manual_confirmation_reason = manual_confirmation_reason or (
                        "完成度闸门契约无效"
                        f"（{completion_gate_error}）"
                    )
                if cycle.review_result.requires_manual_confirmation or manual_confirmation_reason:
                    base_message = manual_confirmation_reason or "审查输出无法可靠判定"
                    message = f"{base_message}；已暂停，需人工确认后再恢复执行"
                    logger.warning(
                        "[Spec:%s] %s",
                        self._project.name,
                        message,
                    )
                    self._project.pause()
                    self._project.error = message
                    self._persist_state_best_effort()
                    return True, "paused"

                completion_gate_met = (
                    completion_gate_valid
                    and cycle.review_result.completion_gate_met
                )
                completion_gate_confidence = (
                    cycle.review_result.completion_gate_confidence
                )

        # Evidence-backed early stop: bypass streak requirement when completion_control
        # confirms GOAL_MET with high confidence AND objective verify passed AND
        # no other blocking suggestions exist.
        if (
            all_satisfied
            and completion_gate_met
            and completion_gate_confidence == "high"
            and review_passed
            and not effective_review_passed
            and getattr(self, "_last_verify_passed", None) is not False
        ):
            logger.info(
                "[Spec:%s] 完成度闸门允许提前结束：completion_control=GOAL_MET(high), verify=%s",
                self._project.name,
                getattr(self, "_last_verify_passed", None),
            )
            effective_review_passed = True

        # Completion control veto: an explicit rejection, invalid/missing
        # verdict, or format failure must prevent success even when there is no
        # high-confidence blocking suggestion.
        if (
            all_satisfied
            and effective_review_passed
            and not completion_gate_met
            and cycle.review_result
            and completion_gate_enabled
        ):
            if isinstance(cycle.review_result, AdaptiveReviewResult):
                completion_control_rejected = any(
                    o.role_id == "completion_control"
                    and (not o.passed or o.goal_verdict != "GOAL_MET")
                    for o in cycle.review_result.role_outcomes
                )
                if completion_control_rejected:
                    logger.info(
                        "[Spec:%s] 完成度闸门否决：completion_control 未提供有效通过判定",
                        self._project.name,
                    )
                    effective_review_passed = False

        decision = policy.should_stop(
            cycle_num=cycle_num,
            all_satisfied=all_satisfied,
            review_passed=effective_review_passed,
            converged=converged,
            metrics=metrics,
            backlog_stuck=_backlog_stuck,
            ignore_backlog=getattr(self.settings, "spec_success_ignore_backlog", True),
        )
        if decision == "success":
            logger.info("[Spec:%s] 所有标准+审查通过, 循环 %d 轮", self._project.name, cycle_num)
            return True, "success"
        if decision == "converged":
            return True, "converged"
        if decision == "backlog_stuck":
            logger.info("[Spec:%s] backlog stuck 终止, 循环 %d 轮", self._project.name, cycle_num)
            return True, "backlog_stuck"

        if self.settings.spec_rebuild_session_between_cycles:
            logger.info(
                "[Spec:%s] 循环 %d 结束, 重建 Session 以压缩对话上下文",
                self._project.name,
                cycle_num,
            )
            self._recreate_session_best_effort()
            if not self._session:
                logger.error(
                    "[Spec:%s] 循环 %d Session 重建失败，session=None，下一循环将无法执行",
                    self._project.name,
                    cycle_num,
                )
        return False, ""

    # ------------------------------------------------------------------
    # Long-range: work items, discovery, spec generation
    # ------------------------------------------------------------------
    def _resolve_max_cycles(self, requested: int) -> int:
        with contextlib.suppress(Exception):  # intentional: defensive int coercion
            requested = int(requested)
        if not isinstance(requested, int):
            requested = 10

        limit = 5000
        with contextlib.suppress(Exception):  # intentional: defensive int coercion
            limit = int(getattr(self.settings, "spec_max_cycles_limit", 5000))
        if limit <= 0:
            limit = 5000
        if requested <= 0:
            requested = 1
        return min(requested, limit)

    def _discover_optimization_questions(
        self,
        cycle_num: int,
        all_satisfied: bool = False,
        backlog_pending: int = 0,
    ) -> list[dict]:
        return _discover_optimization_questions(
            project=self._project,
            session=self._session,
            send_prompt_fn=self._send_prompt_with_retry,
            last_review=self._last_review,
            cycle_num=cycle_num,
            settings=self.settings,
            all_satisfied=all_satisfied,
            backlog_pending=backlog_pending,
            min_cycles=max(1, self.settings.spec_min_cycles),
        )

    def _generate_specs_from_discovery(self, cycle_num: int, discovery: list[dict]) -> list:
        return _generate_specs_from_discovery(
            project=self._project,
            session=self._session,
            send_prompt_fn=self._send_prompt_with_retry,
            root_path=self.root_path,
            settings=self.settings,
            cycle_num=cycle_num,
            discovery=discovery,
        )

    # ------------------------------------------------------------------
    # Persistence helpers (state + artifacts)
    # ------------------------------------------------------------------
    def _persist_state_best_effort(self) -> None:
        _persist_state_best_effort(self._project, self.save_state, _get_state_path(self.root_path, self.settings))

    # ------------------------------------------------------------------
    # Monitoring metrics
    # ------------------------------------------------------------------

    def _make_aux_send_fn(self) -> Callable[[str], str]:
        """Create a send_fn that uses a disposable ACP sub-session."""
        def _send(text: str) -> str:
            timeout = getattr(self.settings, "engine_aux_prompt_timeout", 600)
            return prompt_via_acp(
                text,
                create_session_fn=self._create_session_fn,
                agent_type=self._agent_type,
                cwd=self.root_path,
                timeout=timeout,
                model_name=self._model_name,
            )
        return _send

    def _decompose_criteria_with_llm(self, text: str) -> list[str]:
        return _decompose_criteria_with_llm_impl(text, self.settings, send_fn=self._make_aux_send_fn())

    # ------------------------------------------------------------------
    # Criteria evaluation (reuses loop pattern)
    # ------------------------------------------------------------------
    def _evaluate_criteria(self, criteria: list[str], cycle: int) -> dict:
        cached = None
        if self._last_verify_passed is not None:
            cached = (self._last_verify_passed, self._last_verify_output)
        return _evaluate_criteria_impl(
            session=self._session,
            criteria=criteria,
            cycle=cycle,
            project=self._project,
            send_prompt_fn=self._send_prompt_with_retry,
            settings=self.settings,
            cached_verify=cached,
        )

    # ------------------------------------------------------------------
    # Review (reuses shared parsing infrastructure)
    # ------------------------------------------------------------------
    def _reset_cancel_event(self) -> bool:
        """Reset _review_cancel_event under lock, guarding against stop/pause races.

        Returns True if the event was successfully cleared (engine is RUNNING),
        False if the engine is no longer running (event is set immediately).
        """
        with self._lock:
            is_running = (self._run_state == EngineRunState.RUNNING)
        return self._review_orchestrator.reset_cancel_event(is_running=is_running)

    def _conduct_review(self, cycle: int, callbacks: SpecEngineCallbacks, cycle_obj=None) -> ReviewResult:
        # When cycle_obj is provided and parallel pipeline is enabled, collect artifacts.
        artifacts = None
        parallel_enabled = getattr(self.settings, "spec_review_parallel_enabled", True)
        if parallel_enabled and cycle_obj is not None:
            try:
                from .review_artifacts import collect_review_artifacts
                artifacts = collect_review_artifacts(
                    cycle=cycle_obj,
                    project=self._project,
                    cwd=self.root_path,
                    verify_passed=getattr(self, "_last_verify_passed", None),
                    verify_output=getattr(self, "_last_verify_output", "") or "",
                )
            except Exception as e:
                logger.warning("[Spec] collect_review_artifacts failed, falling back to legacy: %s", repr(e), exc_info=True)

        # Reset cancel_event for this review cycle; set immediately if engine is stopping.
        self._reset_cancel_event()

        # on_retry_status callback forwards to callbacks.on_review_retry for user visibility.
        def _on_retry_status(event: "RetryEvent") -> None:
            if callbacks.on_review_retry:
                callbacks.on_review_retry(cycle, event)

        strategy = select_review_strategy(self.settings)
        return strategy.run(
            ReviewContext(
                settings=self.settings,
                circuit=self._review_orchestrator.circuit,
                cycle=cycle,
                session=self._session,
                project=self._project,
                send_prompt_with_retry_fn=self._send_prompt_with_retry,
                build_review_exception_diagnostics_fn=self._build_review_exception_diagnostics,
                on_review_done=callbacks.on_review_done,
                artifacts=artifacts,
                agent_type=self._agent_type or "coco",
                model_name=self._model_name,
                on_retry_status=_on_retry_status,
                cancel_event=self._review_orchestrator.cancel_event,
                review_agents=list(self._review_agent_pool),
            ),
        )

    def _parse_review_output(self, text: str, cycle: int) -> ReviewResult:
        return _parse_review_output_impl(
            text,
            cycle,
            parse_with_llm_fn=lambda raw: _parse_review_with_llm_impl(raw, self.settings, send_fn=self._make_aux_send_fn()),
        )

    def _parse_review_with_llm(self, raw_text: str) -> list[PerspectiveReview]:
        return _parse_review_with_llm_impl(raw_text, self.settings, send_fn=self._make_aux_send_fn())

    @staticmethod
    def _extract_reviews_from_llm_response(text: str) -> list[PerspectiveReview]:
        return extract_reviews_from_llm_response(text)

    def _consume_guidance(self) -> str:
        if not self._user_guidance:
            return ""
        combined = "\n\n".join(self._user_guidance)
        self._user_guidance.clear()
        return f"\n## 用户引导\n{combined}\n"

    def _detect_convergence(self) -> bool:
        if not self._project:
            return False
        return detect_convergence(
            self._project,
            convergence_window=int(self.settings.spec_convergence_window or 0),
            review_enabled=self.settings.spec_review_enabled,
        )


    def refine_goal_with_guidance(self, guidance: str) -> tuple[bool, str]:
        """将用户引导直接追加到需求中，更新 project.requirement。

        Returns:
            (success, new_requirement_or_error_msg)
        """
        if not self._project:
            return False, "没有活跃的 Spec 项目"

        original = self._project.requirement
        if not original.strip():
            return False, "原始需求为空，无法合并"

        # 直接追加引导到需求
        self._project.requirement = f"{original}\n\n## 补充约束/偏好\n{guidance}"
        self._persist_state_best_effort()
        logger.info("[Spec] 直接追加引导到需求")
        return True, self._project.requirement

    def inject_guidance(self, message: str):
        """Inject user guidance — will be included in the next phase prompt."""
        self._user_guidance.append(message)
        logger.info("[Spec] 用户引导已注入(队列=%d): %s...", len(self._user_guidance), message[:100])

    def _on_stop(self) -> None:
        """Signal review cancel event when engine is stopped."""
        self._review_orchestrator.signal_stop()

    def pause(self):
        with self._lock:
            if self._project:
                self._project.status = SpecProjectStatus.PAUSED
            self._run_state = EngineRunState.STOPPING
            session = self._session
        self._review_orchestrator.signal_stop()
        if session:
            try:
                session.cancel()
            except Exception:
                logger.debug("[Spec] session.cancel() failed during stop", exc_info=True)

    def _handle_max_cycles_termination(self, max_cycles: int):
        is_all_satisfied = self._project.is_all_satisfied
        last_review_passed = True
        if self.settings.spec_review_enabled:
            if self._last_review:
                required = int(getattr(self.settings, "spec_review_pass_streak_required", 1) or 1)
                last_review_passed = (
                    self._last_review.all_passed
                    and int(getattr(self._project, "review_pass_streak", 0) or 0) >= required
                )
            else:
                last_review_passed = False

        if is_all_satisfied and last_review_passed:
            msg = f"达到最大循环次数({max_cycles})。核心验收标准已满足，但仍有待办优化项（Backlog）。"
            self._project.status = SpecProjectStatus.PAUSED
            self._project.error = msg + "（已暂停，可使用 /spec resume 继续执行优化项）"
            self._project.completed_at = time.time()
        else:
            msg = f"达到最大循环次数({max_cycles})仍未满足验收标准或审查未通过"
            if self.settings.spec_infinite_mode:
                self._project.status = SpecProjectStatus.PAUSED
                self._project.error = msg + "（已暂停，可继续 /spec resume 或提升 SPEC_MAX_CYCLES）"
                self._project.completed_at = time.time()
            else:
                self._project.abort(msg)

    def resume(self, callbacks: Optional[SpecEngineCallbacks] = None) -> Optional[SpecProject]:
        """Resume a paused spec execution."""
        if not self._project or self._project.status not in (SpecProjectStatus.PAUSED, SpecProjectStatus.CLARIFYING):
            return self._project

        # Restore review circuit state from persistence (survives process restart)
        try:
            for state_path in _state_path_candidates(self.root_path, self.settings):
                if os.path.isfile(state_path):
                    _, circuit = self.load_state_with_circuit(state_path)
                    self._review_orchestrator.restore_circuit(circuit)
                    break
        except Exception as e:
            logger.warning("[Spec] resume circuit restore skipped: %s", get_error_detail(e), exc_info=True)

        callbacks = self._wrap_callbacks(callbacks or SpecEngineCallbacks())
        with self._lock:
            self._run_state = EngineRunState.RUNNING
            self._project.status = SpecProjectStatus.RUNNING
            self._termination_reason = None
        additional_cycles = self._resolve_max_cycles(self.settings.spec_max_cycles)

        last_cycle_num = 0
        if self._project.cycles:
            last_cycle_num = self._project.cycles[-1].cycle_number
        start_cycle = max(last_cycle_num, self._project.cycle_count_total) + 1
        max_cycles = start_cycle + additional_cycles - 1

        try:
            self._close_session_safely()

            self._session = self._create_session_fn(
                agent_type=self._agent_type,
                cwd=self.root_path,
                on_rate_limit=getattr(self, "_on_rate_limit", None),
                model_name=self._model_name,
            )

            self._termination_reason = self._run_cycle_loop(
                start_cycle=start_cycle,
                max_cycles=max_cycles,
                callbacks=callbacks,
                timeout=self.settings.spec_execution_timeout,
            )

            self._finalize_execution(max_cycles=max_cycles, callbacks=callbacks)

        except TimeoutError as e:
            self._finalize_execution(max_cycles=max_cycles, callbacks=callbacks, error=e, is_timeout=True, label="Spec恢复")

        except Exception as e:
            logger.exception("Unexpected error in Spec resume")
            self._finalize_execution(max_cycles=max_cycles, callbacks=callbacks, error=e, is_timeout=False, label="Spec恢复")

        finally:
            self._close_session_safely()
            with self._lock:
                self._run_state = EngineRunState.IDLE

            if self._project and self._project.status == SpecProjectStatus.COMPLETED and self._saved_task_id:
                delete_task_state(self._saved_task_id)
                self._saved_task_id = None
                self._saved_task_signature = None

        return self._project

    def _project_to_compact_dict(self) -> dict:
        return _project_to_compact_dict_impl(self._project, self.settings, self.root_path)

    def save_state(self, filepath: Optional[str] = None) -> str:
        return _save_engine_state(
            self._project,
            self.settings,
            self.root_path,
            self.chat_id,
            self._build_runtime_context,
            self._project_to_compact_dict,
            filepath,
            review_circuit=self._review_orchestrator.to_dict(),
        )

    @classmethod
    def load_state(cls, filepath: str) -> Optional[SpecProject]:
        project, _rc = _load_engine_state(filepath)
        return project

    @classmethod
    def load_state_with_circuit(cls, filepath: str) -> tuple[Optional[SpecProject], ReviewCircuitState]:
        """Load project + review circuit state (backward-compatible)."""
        project, rc_dict = _load_engine_state(filepath)
        circuit = ReviewCircuitState.from_dict(rc_dict) if rc_dict else ReviewCircuitState()
        return project, circuit

    def cleanup(self):
        super().cleanup()
