"""ACP-driven Deep Engine — leverages agent's own planning capabilities.

Instead of parsing requirements → planning tasks → executing one-by-one,
the new Deep Engine sends a single comprehensive prompt to the agent and
monitors its plan/tool-call/text progress via ACP events.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..acp import ACPEvent, ACPEventType, PromptResult, run_prompt_with_continuation
from ..acp.outcome import PromptAssessment, PromptOutcome, classify_prompt_result
from ..agent_session import create_engine_session
from ..agent_session.backend_resolver import resolve_cwd
from ..engine_base import BaseEngine, BaseEngineManager
from ..grill_me import DEEP_GRILL_ME_PROTOCOL
from ..utils.debug_utils import MemorySnapshot
from ..utils.errors import get_error_detail
from ..utils.gc_monitor import get_gc_monitor
from ..utils.retry import RetryPolicy
from ..utils.trace import TraceContext
from .models import (
    DeepProject,
    DeepProjectStatus,
    EngineRunState,
    ProgressUpdate,
)
from .progress import DeepProgress

logger = logging.getLogger(__name__)

_STATUS_ICONS = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}


@dataclass
class DeepEngineCallbacks:
    """Callbacks for deep engine lifecycle events."""

    on_analyzing_start: Optional[Callable[[str], None]] = None
    on_analyzing_done: Optional[Callable[[DeepProject], None]] = None
    on_event: Optional[Callable[[ACPEvent], None]] = None
    on_text: Optional[Callable[[str], None]] = None
    on_project_done: Optional[Callable[[DeepProject], None]] = None
    on_error: Optional[Callable[[str], None]] = None

    @property
    def on_planning_start(self):
        return self.on_analyzing_start

    @on_planning_start.setter
    def on_planning_start(self, value):
        self.on_analyzing_start = value

    @property
    def on_planning_done(self):
        return self.on_analyzing_done

    @on_planning_done.setter
    def on_planning_done(self, value):
        self.on_analyzing_done = value


class _RetryingPromptSession:
    """Adapt Deep's transport retry contract to the shared auto-continuation loop."""

    def __init__(self, session, before_retry: Callable[[int, Exception], None]):
        self._session = session
        self._before_retry = before_retry

    @property
    def _force_dead(self) -> bool:
        value = getattr(self._session, "_force_dead", False)
        return value if isinstance(value, bool) else False

    @_force_dead.setter
    def _force_dead(self, value: bool) -> None:
        setattr(self._session, "_force_dead", value)

    def send_prompt(self, text, on_event=None, timeout=None):
        return self._session.send_prompt_with_retry(
            text,
            on_event=on_event,
            timeout=timeout,
            retry_policy=RetryPolicy(max_retries=2, retry_delay=2.0),
            before_retry=self._before_retry,
        )

    def __getattr__(self, name):
        return getattr(self._session, name)


class DeepEngine(BaseEngine):
    """ACP-driven Deep Engine — the agent plans and executes autonomously."""

    _state_filename = ".deep_engine_state.json"
    _gc_label = "Deep"
    _gc_threshold_default = 80.0

    def __init__(
        self,
        chat_id: str,
        root_path: str,
        agent_type: str = "coco",
        engine_name: str = "Coco",
        model_name: Optional[str] = None,
    ):
        super().__init__(chat_id, root_path, agent_type, engine_name, model_name)
        self._progress = DeepProgress()
        self._pending_context: list[str] = []
        self._planning_done_fired: bool = False
        self._last_mem_check: float = 0.0
        self._mem_snapshot = MemorySnapshot()

    @property
    def progress(self) -> DeepProgress:
        return self._progress

    def _check_memory_and_gc(self) -> None:
        """Backward-compatible memory check hook used by legacy tests/callers.

        Delegates to the global GCMonitor which handles threshold-based gc.collect().
        """
        now = time.time()
        if now - self._last_mem_check < 5.0:
            return
        self._last_mem_check = now

        gc_monitor = get_gc_monitor(memory_threshold_percent=self.settings.deep_memory_threshold)
        gc_monitor.check_and_collect(label="DeepEngine")

    def _make_on_event(self, callbacks: DeepEngineCallbacks) -> Callable[[ACPEvent], None]:
        """Create the event callback for autonomous execution."""
        gc_monitor = get_gc_monitor(memory_threshold_percent=self.settings.deep_memory_threshold)

        def on_event(event: ACPEvent):
            try:
                gc_monitor.check_and_collect(label="Deep", mem_snapshot=self._mem_snapshot)

                if self._run_state == EngineRunState.STOPPING:
                    if self._session:
                        self._session.cancel()
                    return

                # Capture local ref to avoid race with cleanup()
                renderer = self._renderer
                if not renderer:
                    return
                renderer.process_event(event)

                # Transition: planning -> executing
                # Some backends (Claude CLI) only emit TEXT_CHUNK; for ACP backend,
                # tool calls/plan updates are strong signals that execution has started.
                if self._project and self._project.status == DeepProjectStatus.PLANNING:
                    marker_hit = False
                    try:
                        if event.event_type == ACPEventType.TEXT_CHUNK:
                            txt = renderer.text_content or ""
                            # Heuristic markers for CLI backend
                            if "### 执行过程" in txt or "## 执行过程" in txt or "开始执行" in txt:
                                marker_hit = True
                    except Exception:
                        marker_hit = False

                    if marker_hit or event.event_type in (
                        ACPEventType.PLAN_UPDATE,
                        ACPEventType.TOOL_CALL_START,
                        ACPEventType.TOOL_CALL_UPDATE,
                        ACPEventType.TOOL_CALL_DONE,
                    ):
                        # Mark as executing and fire "planning done" once.
                        self._project.start()
                        if (not self._planning_done_fired) and callbacks.on_analyzing_done:
                            self._planning_done_fired = True
                            callbacks.on_analyzing_done(self._project)

                match event.event_type:
                    case ACPEventType.PLAN_UPDATE:
                        if event.plan:
                            self._progress.update_plan(event.plan)
                    case ACPEventType.TOOL_CALL_DONE:
                        if event.tool_call:
                            self._progress.record_tool(event.tool_call)
                    case ACPEventType.TEXT_CHUNK:
                        if event.text:
                            self._progress.append_text(event.text)

                if callbacks.on_event:
                    callbacks.on_event(event)
                if event.event_type == ACPEventType.TEXT_CHUNK and callbacks.on_text:
                    callbacks.on_text(event.text or "")
            except Exception as e:
                logger.warning("[Deep] on_event 回调异常(已捕获): %s", get_error_detail(e))

        return on_event

    def _fail_project_and_notify(
        self,
        error: Exception,
        label: str,
        *,
        is_timeout: bool,
        callbacks: DeepEngineCallbacks,
    ) -> str:
        """Freeze the Deep project before exposing its terminal state to UI."""
        error_msg = self._format_engine_error(
            error,
            label,
            is_timeout=is_timeout,
            callbacks=None,
        )
        if self._project is None:
            project_name = os.path.basename(self.root_path) or "deep_project"
            self._project = DeepProject.create(
                name=project_name,
                root_path=self.root_path,
            )
        self._project.fail(error_msg)
        if callbacks.on_error:
            callbacks.on_error(error_msg)
        return error_msg

    def plan_and_execute(
        self,
        requirement_text: str,
        callbacks: Optional[DeepEngineCallbacks] = None,
        task_id: Optional[str] = None,
        on_rate_limit: Optional[Callable[[int], None]] = None,
    ) -> DeepProject:
        """Single ACP prompt drives the entire deep execution."""
        callbacks = callbacks or DeepEngineCallbacks()
        with self._lock:
            self._run_state = EngineRunState.RUNNING
            self._planning_done_fired = False
            self._on_rate_limit = on_rate_limit

        project_name = os.path.basename(self.root_path) or "deep_project"
        self._project = DeepProject.create(name=project_name, root_path=self.root_path)
        self._project.task_id = task_id
        self._project.status = DeepProjectStatus.PLANNING
        self._project.started_at = time.time()

        if callbacks.on_analyzing_start:
            callbacks.on_analyzing_start(requirement_text)

        logger.info(
            "[Deep:%s] ACP执行开始, 需求长度=%d, 路径=%s, agent=%s",
            project_name,
            len(requirement_text),
            self.root_path,
            self._agent_type,
        )

        # Initialize TraceContext for the execution
        trace_ctx = TraceContext(trace_id=task_id or f"deep-{int(time.time())}")

        try:
            with trace_ctx:
                # Create session
                self._session = create_engine_session(
                    agent_type=self._agent_type,
                    cwd=resolve_cwd(self._agent_type, self.root_path),
                    on_rate_limit=on_rate_limit,
                    model_name=self._model_name,
                )

                # Build deep prompt — let agent plan and execute autonomously
                prompt = self._build_deep_prompt(requirement_text)

                on_event = self._make_on_event(callbacks)
                timeout = (
                    self.settings.coco_execution_timeout
                    if self._agent_type == "coco"
                    else self.settings.claude_execution_timeout
                )
                result, assessment = self._send_prompt_autonomously(
                    prompt,
                    on_event,
                    timeout,
                )

                # Process pending context injections as follow-up prompts
                result, assessment = self._drain_pending_context(
                    on_event,
                    timeout,
                    result,
                    assessment,
                )

                # Determine final status
                if self._run_state == EngineRunState.STOPPING:
                    self._project.cancel()
                    logger.info("[Deep:%s] 执行已取消", project_name)
                else:
                    self._apply_prompt_result(
                        result,
                        assessment=assessment,
                        project_name=project_name,
                    )

                if callbacks.on_project_done:
                    callbacks.on_project_done(self._project)

                return self._project

        except TimeoutError as e:
            if self._run_state == EngineRunState.STOPPING:
                self._project.cancel()
                if callbacks.on_project_done:
                    callbacks.on_project_done(self._project)
                return self._project
            self._fail_project_and_notify(
                e,
                "执行",
                is_timeout=True,
                callbacks=callbacks,
            )
            return self._project

        except Exception as e:
            if self._run_state == EngineRunState.STOPPING:
                self._project.cancel()
                if callbacks.on_project_done:
                    callbacks.on_project_done(self._project)
                return self._project
            self._fail_project_and_notify(
                e,
                "执行",
                is_timeout=False,
                callbacks=callbacks,
            )
            return self._project

        finally:
            with self._lock:
                self._run_state = EngineRunState.IDLE
            get_gc_monitor(memory_threshold_percent=self.settings.deep_memory_threshold).check_and_collect(label="Deep", mem_snapshot=self._mem_snapshot)

    def _apply_prompt_result(
        self,
        result: PromptResult,
        *,
        assessment: PromptAssessment,
        project_name: str,
    ) -> None:
        """Translate transport completion into the Deep project terminal state."""
        if self._project is None:
            return
        if assessment.outcome is PromptOutcome.COMPLETED:
            self._project.complete()
            logger.info(
                "[Deep:%s] 执行完成, 工具调用=%d, 修改文件=%d, 总耗时=%.1fs",
                project_name,
                len(self._progress.tool_calls),
                len(self._progress.modified_files),
                self._project.duration() or 0,
            )
        elif assessment.outcome is PromptOutcome.CANCELLED:
            self._project.cancel()
            logger.info("[Deep:%s] 执行已取消", project_name)
        else:
            self._project.fail(f"执行未完成: {assessment.detail}")
            logger.warning(
                "[Deep:%s] 执行未完成, stop_reason=%s, detail=%s",
                project_name,
                assessment.stop_reason,
                assessment.detail,
            )

    def _drain_pending_context(
        self,
        on_event,
        timeout,
        last_result: PromptResult,
        last_assessment: PromptAssessment,
    ) -> tuple[PromptResult, PromptAssessment]:
        """Send any pending context injections as follow-up prompts in the same session."""
        while self._run_state == EngineRunState.RUNNING:
            # Guard: session may have been closed by concurrent stop/cleanup
            if not self._session:
                logger.warning("[Deep] _drain_pending_context: session 已关闭, 停止处理")
                break
            with self._lock:
                batch = list(self._pending_context)
                self._pending_context.clear()
            if not batch:
                break
            ctx = "\n\n---\n\n".join(batch)
            logger.info("[Deep] 发送注入的上下文(%d条): %s...", len(batch), ctx[:100])
            follow_up = f"""用户提供了额外的上下文/指导信息，请据此继续执行：

{ctx}

请根据以上信息调整你的执行方案并继续。"""
            try:
                follow_up_result, follow_up_assessment = self._send_prompt_autonomously(
                    follow_up,
                    on_event,
                    timeout,
                )
                if follow_up_assessment.outcome is not PromptOutcome.COMPLETED:
                    with self._lock:
                        self._pending_context[0:0] = batch
                    return follow_up_result, follow_up_assessment
                last_result = follow_up_result
                last_assessment = follow_up_assessment
            except TimeoutError as e:
                with self._lock:
                    self._pending_context[0:0] = batch
                logger.warning("[Deep] _drain_pending_context 超时: %s", get_error_detail(e))
                failed = PromptResult(stop_reason="pending_context_timeout")
                return failed, classify_prompt_result(failed)
            except Exception as e:
                with self._lock:
                    self._pending_context[0:0] = batch
                logger.error("[Deep] _drain_pending_context 发送失败: %s", get_error_detail(e))
                failed = PromptResult(stop_reason="pending_context_error")
                return failed, classify_prompt_result(failed)
        return last_result, last_assessment

    def _send_prompt_autonomously(
        self,
        prompt: str,
        on_event,
        timeout: float,
    ) -> tuple[PromptResult, PromptAssessment]:
        """Run one logical Deep turn with bounded retries and safe auto-decisions."""

        def _before_retry(_attempt: int, _error: Exception) -> None:
            self._renderer.reset()
            self._planning_done_fired = False

        session = _RetryingPromptSession(self._session, _before_retry)
        execution = run_prompt_with_continuation(
            session,
            prompt,
            on_event=on_event,
            timeout_s=timeout,
            finalization_reserve_s=max(
                0,
                int(getattr(self.settings, "programming_finalization_reserve_s", 0) or 0),
            ),
            finalization_task_text=prompt,
        )
        if execution.automatic_continuations:
            logger.info(
                "[Deep] 自动续做完成: continuations=%d outcome=%s detail=%s",
                execution.automatic_continuations,
                execution.assessment.outcome.value,
                execution.assessment.detail,
            )
        return execution.result, execution.assessment

    def _build_deep_prompt(self, requirement: str) -> str:
        """Build the deep prompt — let agent autonomously plan and execute."""
        return f"""你是一个专业的软件工程师。请完成以下需求：

## 需求
{requirement}

## 工作目录
{self.root_path}

{DEEP_GRILL_ME_PROTOCOL}
## 要求
1. 必须先输出清晰的【分析】与【执行计划】，再开始调用工具执行
2. 计划需要拆成可验证的步骤（建议 3~8 步），每步一句话描述产物/验证点
3. 计划中需要识别可并行的独立工作包；对没有依赖、不会修改相同文件/接口契约/迁移配置的工作，优先使用当前工具支持的 subagent / 子任务委托并行执行
4. 对可能冲突的工作必须保持串行，并在计划中说明冲突边界
5. 执行时严格按计划与依赖关系推进；如需调整计划，先解释原因并更新计划
6. 每个关键步骤完成后做一次自检/验证（单测/运行/静态检查等，按项目能力选择）
7. 完成后输出总结：做了什么、改了哪些文件、如何验证，以及哪些任务并行/委托执行
8. 返回完成前必须回读原始需求；若仍有未完成的用户要求、未验证的核心验收路径、未结束的计划/工具调用或相关失败测试，继续执行，无法继续时明确报告阻塞/失败，不得宣称完成

## 输出格式（强制）
### 分析
- ...

### 执行计划
1. ...
2. ...

### 执行过程
（从这里开始再调用工具）
"""

    def inject_guidance(self, message: str):
        with self._lock:
            self._pending_context.append(message)
        logger.info("[Deep] 上下文已注入(待发送, 队列=%d): %s...", len(self._pending_context), message[:100])

    inject_context = inject_guidance

    # Static status messages (no f-string allocation per call)
    _STATUS_MESSAGES: dict[DeepProjectStatus, str] = {
        DeepProjectStatus.IDLE: "等待开始",
        DeepProjectStatus.PLANNING: "正在规划任务...",
        DeepProjectStatus.CANCELLED: "已取消",
        DeepProjectStatus.COMPLETED: "全部完成",
    }

    def get_progress(self) -> Optional[ProgressUpdate]:
        with self._lock:
            if not self._project:
                return None

            status = self._project.status
            # Snapshot progress data under lock
            tool_calls_count = len(self._progress.tool_calls)
            completed = self._progress.completed_steps
            total = self._progress.total_steps or 1
            project_id = self._project.project_id
            error = self._project.error

        if status == DeepProjectStatus.EXECUTING:
            message = f"执行中 (工具调用: {tool_calls_count})"
        elif status == DeepProjectStatus.FAILED:
            message = f"执行失败: {error or '未知错误'}"
        else:
            message = self._STATUS_MESSAGES.get(status, "未知状态")

        return ProgressUpdate(
            project_id=project_id,
            completed_count=completed,
            total_count=total,
            status=status,
            message=message,
        )

    def get_task_summary(self) -> str:
        with self._lock:
            if not self._project:
                return "暂无任务"

            lines = [f"📊 **{self._project.name}** 执行进度\n"]

            if self._progress.plan_entries:
                lines.append(self._progress.progress_bar)
                lines.append("")
                for entry in self._progress.plan_entries:
                    icon = _STATUS_ICONS.get(entry["status"], "⬜")
                    lines.append(f"{icon} {entry['content']}")
            else:
                lines.append(f"🔧 工具调用: {len(self._progress.tool_calls)} 次")

            if self._progress.modified_files:
                lines.append(f"\n📝 修改文件: {len(self._progress.modified_files)} 个")

            if self._project.duration():
                lines.append(f"\n⏱️ 总耗时: {self._project.duration():.1f}s")

        return "\n".join(lines)

    def load_state(self, filepath: Optional[str] = None) -> bool:
        if not filepath:
            filepath = os.path.join(self.root_path, self._state_filename)

        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._project = DeepProject.from_dict(state["project"])
            return True
        except Exception as e:
            logger.error("加载状态失败: %s", get_error_detail(e))
            return False

    def cleanup(self):
        super().cleanup()


class DeepEngineManager(BaseEngineManager["DeepEngine"]):
    """Manages DeepEngine instances per chat+project.

    Thread-safe: all dict mutations are protected by _lock.
    Uses a secondary index (_chat_keys) to avoid O(n) full-table scans.
    """

    def _create_engine(
        self,
        chat_id: str,
        root_path: str,
        agent_type: str,
        engine_name: str,
        model_name: Optional[str],
    ) -> "DeepEngine":
        return DeepEngine(
            chat_id=chat_id,
            root_path=root_path,
            agent_type=agent_type,
            engine_name=engine_name,
            model_name=model_name,
        )

    def remove(self, chat_id: str, root_path: str):
        key = f"{chat_id}:{root_path}"
        with self._lock:
            if key in self._engines:
                self._engines[key].cleanup()
                del self._engines[key]
                self._remove_index(chat_id, key)

    def find_by_deep_project_id(self, chat_id: str, deep_project_id: str) -> Optional["DeepEngine"]:
        if not deep_project_id:
            return None
        for engine in self._iter_chat_engines(chat_id):
            try:
                if engine.project and engine.project.project_id == deep_project_id:
                    return engine
            except Exception:
                continue
        return None

    def _build_snapshot(self, engine: "DeepEngine"):
        """Build EngineSnapshot with Deep-specific fields."""
        from src.card.engine_snapshot import EngineSnapshot
        project = engine.project
        progress = engine.progress
        return EngineSnapshot(
            engine_name=engine.engine_name,
            root_path=engine.root_path,
            project_id=project.project_id if project else "",
            tool_calls_count=len(progress.tool_calls) if progress else 0,
            completed_steps=progress.completed_steps if progress else 0,
            total_steps=progress.total_steps if progress else 0,
            duration_seconds=project.duration() if project else None,
            status=project.status.value if project else "",
            is_running=engine.is_running,
            ext={"project": project, "progress": progress},
        )
