from __future__ import annotations

import logging
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from ..acp.outcome import PromptOutcome, classify_prompt_result
from ..config import get_settings
from ..ttadk.models import is_invalid_model_error
from ..utils.callbacks import safe_invoke
from ..utils.errors import classify_timeout, get_error_detail
from .models import WorktreeSelectionItem, WorktreeUnit, WorktreeUnitStatus
from .reporter import REASON_DISPLAY_MAP
from .subagent_progress import WorktreeSubagentProgress

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..agent_session import SyncSession


DEFAULT_CANCEL_GRACE_SECONDS = 2.0


@dataclass
class UnitExecutionControl:
    """Thread-safe cancellation binding for one Worktree unit generation."""

    generation: int
    cancel_event: threading.Event
    cancel_done: threading.Event = field(default_factory=threading.Event)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    cancel_ack: threading.Event = field(default_factory=threading.Event)
    cancel_error: str = ""
    terminal_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _session: Any = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bind_session(self, session: Any) -> None:
        with self._lock:
            self._session = session

    def bound_session(self) -> Any:
        with self._lock:
            return self._session


def _detect_worktree_changes(worktree_path: str) -> bool:
    if not worktree_path:
        return False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree_path,
            check=False,
            capture_output=True,
            text=True,
        )
        return bool((result.stdout or "").strip())
    except Exception:
        return False


def _command_provenance(result: object) -> list[dict[str, Any]]:
    """Extract structured command facts emitted by the execution transport."""
    provenance: list[dict[str, Any]] = []
    for item in getattr(result, "tool_results", None) or ():
        if not isinstance(item, dict) or item.get("kind") != "execute":
            continue
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        command = str(data.get("command") or "").strip()
        if not command:
            continue
        raw_exit_code = data.get("exit_code")
        if isinstance(raw_exit_code, bool):
            exit_code = None
        else:
            try:
                exit_code = int(raw_exit_code)
            except (TypeError, ValueError):
                exit_code = None
        output = str(
            data.get("output")
            or data.get("stdout")
            or data.get("stderr")
            or ""
        ).strip()
        provenance.append(
            {
                "source": "worktree_dispatcher",
                "command": command,
                "exit_code": exit_code,
                "evidence": output[:4_000],
            }
        )
    return provenance


class WorktreeDispatcher:
    def __init__(
        self,
        *,
        session_factory: Optional[Callable[..., "SyncSession"]] = None,
    ) -> None:
        self._session_factory = session_factory or self._default_session_factory

    @staticmethod
    def _default_session_factory(**kwargs):
        from ..agent_session import create_sync_session_for_worktree

        return create_sync_session_for_worktree(**kwargs)

    # Deterministic strength hints for the coordinator pick. If no signal is
    # available, the first selected tool remains the coordinator.
    _MODEL_STRENGTH_HINTS: tuple[tuple[str, int], ...] = (
        ("gpt-5.5", 120),
        ("gpt-5-5", 120),
        ("gpt-5.4", 115),
        ("gpt-5-4", 115),
        ("gpt-5.2", 110),
        ("gpt-5-2", 110),
        ("opus", 105),
        ("sonnet", 95),
        ("gemini-3.1-pro", 100),
        ("gemini-3-pro", 98),
        ("gemini-2.5-pro", 92),
        ("thinking", 90),
        ("pro", 70),
    )
    _TOOL_STRENGTH_HINTS: dict[str, int] = {
        "claude": 80,
        "codex": 76,
        "coco": 74,
        "gemini": 72,
        "aiden": 68,
    }

    def plan_user_goal(self, goal: str, units: Iterable[WorktreeUnit], tools: Iterable[WorktreeSelectionItem]) -> list[WorktreeUnit]:
        normalized_goal = str(goal or "").strip()
        planned_units = list(units)
        tool_pool = list(tools)
        if not planned_units or not tool_pool:
            return planned_units

        assignments = self._assign_roles_smart(planned_units, tool_pool)
        main_tool = assignments[0][0] if assignments else None
        main_label = main_tool.display_label if main_tool else "第一个已选工具"
        for unit, (tool, role_info) in zip(planned_units, assignments):
            title, role_prompt = role_info
            is_main = tool is main_tool and unit is planned_units[0]

            # 动态绑定工具
            unit.provider = tool.provider
            unit.agent_name = tool.agent_name
            unit.tool_name = tool.tool_name
            unit.selection_key = tool.selection_key
            unit.display_name = tool.display_name
            unit.model_name = tool.model_name
            unit.metadata["worktree_main_agent"] = is_main
            unit.metadata["worktree_main_selection_key"] = main_tool.selection_key if main_tool else ""

            unit.task_title = title
            unit.task_prompt = (
                f"用户目标：{normalized_goal}\n"
                f"主控 agent：{main_label}\n"
                f"你的角色：{role_prompt}\n"
                "执行方式：主控 agent 负责统一梳理目标、拆分任务、控制节奏和最终验收；"
                "其它单元按主控目标并行承担可独立完成的具体任务。同一个工具可以被复用到多个任务，"
                "但每个单元必须只在当前 worktree 中工作，优先处理与你角色匹配且不会和其它单元争用同一文件/接口契约的任务；"
                "如发现潜在冲突，请记录冲突点和建议串行顺序，不要跨 worktree 修改。"
            )
            unit.status = WorktreeUnitStatus.PLANNED
        return planned_units

    def _fail_unit(
        self,
        unit: WorktreeUnit,
        error_msg: str,
        *,
        log_level: int = logging.ERROR,
        on_unit_update: Optional[Callable[[WorktreeUnit], None]] = None,
    ) -> None:
        unit.status = WorktreeUnitStatus.FAILED
        unit.error = error_msg
        unit.summary = error_msg
        logger.log(log_level, "[Worktree] 单元失败: unit=%s, error=%s", unit.unit_id, error_msg)
        safe_invoke(on_unit_update, unit, label="on_unit_update")

    def _cancel_unit(
        self,
        unit: WorktreeUnit,
        reason: str,
        *,
        generation: Optional[int] = None,
        on_unit_update: Optional[Callable[[WorktreeUnit], None]] = None,
    ) -> None:
        """Mark a unit as cancelled (e.g. pool-level timeout).

        Skips units that already reached COMPLETED to avoid overwriting a valid
        result.  Sets _cancel_event *before* mutating status so that the worker
        thread can observe the signal via a memory-barrier-backed Event check.
        """
        if generation is not None and unit.generation != generation:
            return
        if unit.status in {
            WorktreeUnitStatus.COMPLETED,
            WorktreeUnitStatus.FAILED,
            WorktreeUnitStatus.CANCELLED,
        }:
            return
        unit._cancel_event.set()
        unit.status = WorktreeUnitStatus.CANCELLED
        unit.error = reason
        unit.stop_reason = reason
        unit.cancellation_requested = False
        unit.cancellation_acknowledged = False
        unit.summary = REASON_DISPLAY_MAP.get(reason, reason)
        logger.warning("[Worktree] 单元取消: unit=%s, reason=%s", unit.unit_id, reason)
        safe_invoke(on_unit_update, unit, label="on_unit_update")

    @staticmethod
    def _invoke_session_cancel(
        control: UnitExecutionControl,
        *,
        timeout: float,
    ) -> None:
        """Invoke the bound session cancel API and record its acknowledgment."""
        session = control.bound_session()
        if session is None:
            control.cancel_error = "session_not_bound"
            control.cancel_done.set()
            return
        try:
            cancel = getattr(session, "cancel")
            control.cancel_requested.set()
            try:
                result = cancel(wait=True, timeout=max(0.0, timeout))
            except TypeError:
                result = cancel()
            acknowledged = result is True or (
                getattr(result, "acknowledged", None) is True
            )
            if acknowledged:
                control.cancel_ack.set()
            else:
                control.cancel_error = (
                    "cancel_not_acknowledged"
                    if result is False
                    else "cancel_ack_unknown"
                )
        except Exception as exc:
            control.cancel_error = get_error_detail(exc)
            logger.warning(
                "[Worktree] session cancel 未确认: generation=%s error=%s",
                control.generation,
                control.cancel_error,
            )
        finally:
            control.cancel_done.set()

    def _cancel_active_futures(
        self,
        future_map: dict[Future, tuple[WorktreeUnit, UnitExecutionControl]],
        active_futures: set[Future],
        *,
        cancel_grace: float,
        on_unit_update: Optional[Callable[[WorktreeUnit], None]],
    ) -> None:
        controls: list[tuple[WorktreeUnit, UnitExecutionControl]] = []
        for future in active_futures:
            unit, control = future_map[future]
            with control.terminal_lock:
                if unit.generation != control.generation or unit.status in {
                    WorktreeUnitStatus.COMPLETED,
                    WorktreeUnitStatus.FAILED,
                    WorktreeUnitStatus.CANCELLED,
                }:
                    continue
                self._cancel_unit(
                    unit,
                    "pool_timeout",
                    generation=control.generation,
                    on_unit_update=None,
                )
            future.cancel()
            controls.append((unit, control))

        grace = max(0.0, float(cancel_grace))
        deadline = time.monotonic() + grace
        for _unit, control in controls:
            cancel_thread = threading.Thread(
                target=self._invoke_session_cancel,
                kwargs={"control": control, "timeout": grace},
                name=f"wt-cancel-{control.generation}",
                daemon=True,
            )
            cancel_thread.start()

        for unit, control in controls:
            remaining = max(0.0, deadline - time.monotonic())
            control.cancel_done.wait(timeout=remaining)
            if unit.generation != control.generation:
                continue
            requested = control.cancel_requested.is_set()
            acknowledged = control.cancel_ack.is_set()
            unit.cancellation_requested = requested
            unit.cancellation_acknowledged = acknowledged
            unit.metadata["cancellation_requested"] = requested
            unit.metadata["cancellation_acknowledged"] = acknowledged
            if control.cancel_error:
                unit.metadata["cancellation_error"] = control.cancel_error
            elif not control.cancel_done.is_set():
                unit.metadata["cancellation_error"] = "cancel_ack_timeout"
            if on_unit_update is not None:
                threading.Thread(
                    target=safe_invoke,
                    args=(on_unit_update, unit),
                    kwargs={"label": "on_unit_update"},
                    name=f"wt-cancel-notify-{control.generation}",
                    daemon=True,
                ).start()

    def execute_units(
        self,
        units: Iterable[WorktreeUnit],
        *,
        timeout: Optional[int] = None,
        max_workers: Optional[int] = None,
        on_unit_update: Optional[Callable[[WorktreeUnit], None]] = None,
        pool_timeout: Optional[int] = None,
        cancel_grace: float = DEFAULT_CANCEL_GRACE_SECONDS,
    ) -> list[WorktreeUnit]:
        planned_units = list(units)
        if not planned_units:
            return []

        if pool_timeout is None:
            settings = get_settings()
            pool_timeout = getattr(settings, "worktree_pool_timeout", 600)

        workers = max(1, min(max_workers or len(planned_units), len(planned_units)))
        executor = ThreadPoolExecutor(max_workers=workers)
        future_map: dict[Future, tuple[WorktreeUnit, UnitExecutionControl]] = {}
        for unit in planned_units:
            unit.generation += 1
            unit._cancel_event.clear()
            unit.cancellation_requested = False
            unit.cancellation_acknowledged = False
            unit.stop_reason = ""
            control = UnitExecutionControl(
                generation=unit.generation,
                cancel_event=unit._cancel_event,
            )
            future = executor.submit(
                self._run_single_unit,
                unit,
                timeout=timeout,
                on_unit_update=on_unit_update,
                control=control,
            )
            future_map[future] = (unit, control)

        done, not_done = wait(
            future_map,
            timeout=max(0.0, float(pool_timeout)),
        )
        for future in done:
            unit, control = future_map[future]
            try:
                future.result()
            except Exception as exc:
                if control.cancel_event.is_set() or unit.generation != control.generation:
                    continue
                err_msg = get_error_detail(exc)
                log_level = logging.WARNING if classify_timeout(exc) else logging.ERROR
                self._fail_unit(
                    unit,
                    err_msg,
                    log_level=log_level,
                    on_unit_update=on_unit_update,
                )

        if not_done:
            self._cancel_active_futures(
                future_map,
                set(not_done),
                cancel_grace=cancel_grace,
                on_unit_update=on_unit_update,
            )
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)
        return planned_units

    def _run_single_unit(
        self,
        unit: WorktreeUnit,
        *,
        timeout: Optional[int] = None,
        on_unit_update: Optional[Callable[[WorktreeUnit], None]] = None,
        control: Optional[UnitExecutionControl] = None,
    ) -> None:
        if control is None:
            control = UnitExecutionControl(
                generation=unit.generation,
                cancel_event=unit._cancel_event,
            )
        # 防止 pool-level timeout 设置的 failed/cancelled 状态被覆盖
        if unit.status in (WorktreeUnitStatus.FAILED, WorktreeUnitStatus.CANCELLED):
            return
        unit.status = WorktreeUnitStatus.RUNNING
        unit.metadata["started_at"] = time.time()
        safe_invoke(on_unit_update, unit, label="on_unit_update")

        # 如果尚未分配具体工具（理论上 plan 阶段已完成分配，这里做安全检查）
        if not all([unit.provider, unit.tool_name]):
            self._fail_unit(unit, "工作单元未绑定执行工具", on_unit_update=on_unit_update)
            return

        try:
            session = self._start_session_with_recovery(
                unit,
                on_session_created=control.bind_session,
                is_cancelled=control.cancel_event.is_set,
            )
        except Exception as exc:
            with control.terminal_lock:
                if control.cancel_event.is_set() or unit.generation != control.generation:
                    return
                self._fail_unit(
                    unit,
                    f"启动失败: {get_error_detail(exc)}",
                    log_level=logging.ERROR,
                    on_unit_update=None,
                )
            safe_invoke(on_unit_update, unit, label="on_unit_update")
            return

        subagent_progress = None
        try:
            if control.cancel_event.is_set() or unit.generation != control.generation:
                return
            subagent_progress = WorktreeSubagentProgress(
                unit,
                on_unit_update=on_unit_update,
            )

            def _on_event(event) -> None:
                if (
                    control.cancel_event.is_set()
                    or unit.generation != control.generation
                    or unit.status != WorktreeUnitStatus.RUNNING
                ):
                    return
                subagent_progress.on_event(event)

            result = session.send_prompt(
                unit.task_prompt or unit.task_title,
                on_event=_on_event,
                timeout=timeout,
            )
            subagent_progress.close()
            # Respect cancellation set by pool-timeout while this unit was running.
            # Uses _cancel_event (threading.Event) for memory-barrier guarantee instead
            # of bare status read which relies on GIL atomicity.
            if control.cancel_event.is_set() or unit.generation != control.generation:
                return
            assessment = classify_prompt_result(result)
            has_changes = _detect_worktree_changes(unit.worktree_path)
            with control.terminal_lock:
                if (
                    control.cancel_event.is_set()
                    or unit.generation != control.generation
                    or unit.status in {
                        WorktreeUnitStatus.COMPLETED,
                        WorktreeUnitStatus.FAILED,
                        WorktreeUnitStatus.CANCELLED,
                    }
                ):
                    return
                unit.summary = (getattr(result, "text", "") or "").strip()
                unit.stop_reason = assessment.stop_reason
                unit.metadata["pending_plan_entries"] = assessment.pending_plan_entries
                unit.metadata["incomplete_tool_calls"] = assessment.incomplete_tool_calls
                unit.metadata["test_provenance"] = _command_provenance(result)
                if assessment.outcome is PromptOutcome.COMPLETED:
                    unit.status = WorktreeUnitStatus.COMPLETED
                    unit.error = ""
                elif assessment.outcome is PromptOutcome.CANCELLED:
                    unit.status = WorktreeUnitStatus.CANCELLED
                    unit.error = unit.summary or assessment.detail
                    unit.cancellation_requested = False
                    unit.cancellation_acknowledged = True
                    unit.metadata["cancellation_requested"] = False
                    unit.metadata["cancellation_acknowledged"] = True
                else:
                    unit.status = WorktreeUnitStatus.FAILED
                    unit.error = unit.summary or assessment.detail
                unit.has_changes = has_changes
            safe_invoke(on_unit_update, unit, label="on_unit_update")
        except TimeoutError as te:
            with control.terminal_lock:
                if control.cancel_event.is_set() or unit.generation != control.generation:
                    return
                self._fail_unit(
                    unit,
                    f"执行超时: {get_error_detail(te)}",
                    log_level=logging.WARNING,
                    on_unit_update=None,
                )
            safe_invoke(on_unit_update, unit, label="on_unit_update")
        except Exception as exc:
            with control.terminal_lock:
                if control.cancel_event.is_set() or unit.generation != control.generation:
                    return
                self._fail_unit(
                    unit,
                    f"执行异常: {get_error_detail(exc)}",
                    log_level=logging.ERROR,
                    on_unit_update=None,
                )
            safe_invoke(on_unit_update, unit, label="on_unit_update")
        finally:
            if subagent_progress is not None:
                subagent_progress.close()
            try:
                session.close()
            except Exception:
                logger.debug("failed to close session", exc_info=True)

    def _start_session_with_recovery(
        self,
        unit: WorktreeUnit,
        *,
        on_session_created: Optional[Callable[["SyncSession"], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> "SyncSession":
        """Start a session with TTADK-specific recovery on failure.

        Recovery flow (TTADK only):
        1. Normal session.start()
        2. On invalid-model error: retry with model_name=None (auto)
        3. On any failure: try coco fallback session
        Non-TTADK providers raise on failure (no recovery).
        """
        session = self._session_factory(
            provider=unit.provider,
            tool_name=unit.tool_name,
            working_dir=unit.worktree_path,
            model_name=unit.model_name,
        )
        safe_invoke(on_session_created, session, label="on_session_created")
        if is_cancelled and is_cancelled():
            return session
        try:
            session.start()
            return session
        except Exception as first_err:
            if unit.provider != "ttadk":
                raise
            try:
                session.close()
            except Exception:
                pass

            err_text = get_error_detail(first_err)
            logger.warning(
                "[Worktree] TTADK start failed, attempting recovery: unit=%s err=%s",
                unit.unit_id, err_text,
            )

            # Retry with model_name=None (auto) if invalid-model error
            if is_invalid_model_error(err_text):
                try:
                    session = self._session_factory(
                        provider=unit.provider,
                        tool_name=unit.tool_name,
                        working_dir=unit.worktree_path,
                        model_name=None,
                    )
                    safe_invoke(on_session_created, session, label="on_session_created")
                    if is_cancelled and is_cancelled():
                        return session
                    session.start()
                    logger.info("[Worktree] TTADK recovery succeeded with auto model: unit=%s", unit.unit_id)
                    return session
                except Exception:
                    try:
                        session.close()
                    except Exception:
                        pass

            # TTADK worktree units are isolated from ACP/coco direct mode.  A
            # generic TTADK startup failure must surface as failed/degraded
            # diagnostics instead of silently creating provider="acp", tool_name="coco".
            raise first_err

    def _assign_roles_smart(
        self,
        units: list[WorktreeUnit],
        tools: list[WorktreeSelectionItem]
    ) -> list[tuple[WorktreeSelectionItem, tuple[str, str]]]:
        """Assign a coordinator first, then deterministic parallel worker roles."""
        if not units or not tools:
            return []

        main_tool = self._select_main_agent_tool(tools)
        remaining_tools = [tool for tool in tools if tool is not main_tool]
        tool_sequence = [main_tool] + remaining_tools

        roles = self._build_role_templates(len(units))
        assignments: list[tuple[WorktreeSelectionItem, tuple[str, str]]] = []
        for index, role in enumerate(roles):
            tool = tool_sequence[index % len(tool_sequence)]
            assignments.append((tool, role))
        return assignments

    @classmethod
    def _select_main_agent_tool(cls, tools: list[WorktreeSelectionItem]) -> WorktreeSelectionItem:
        scored: list[tuple[int, int, WorktreeSelectionItem]] = []
        for index, tool in enumerate(tools):
            scored.append((cls._tool_strength_score(tool), -index, tool))
        best_score, _, best_tool = max(scored, key=lambda item: (item[0], item[1]))
        if best_score <= 0:
            return tools[0]
        return best_tool

    @classmethod
    def _tool_strength_score(cls, tool: WorktreeSelectionItem) -> int:
        text = " ".join(
            str(part or "").lower()
            for part in (
                tool.provider,
                tool.tool_name,
                tool.display_name,
                tool.agent_name,
                tool.model_name,
                tool.model_display_name,
            )
        )
        score = cls._TOOL_STRENGTH_HINTS.get(str(tool.tool_name or "").lower(), 0)
        for token, value in cls._MODEL_STRENGTH_HINTS:
            if token in text:
                score = max(score, value)
        return score

    @staticmethod
    def _build_role_templates(count: int) -> list[tuple[str, str]]:
        if count <= 1:
            return [(
                "主控规划与执行",
                "作为本次任务的主控 agent，完成目标澄清、方案拆分、实现与复核，给出最终结论",
            )]
        templates = [
            (
                "主控规划与验收",
                "作为本次任务的主控 agent，统一理解用户目标，拆出可并行任务，约束文件边界，最后给出验收和合并建议",
            ),
            ("实现与修改", "聚焦代码实现、必要修改与验证思路；如任务较多，可承担其中一个独立子任务"),
            ("测试与验证", "聚焦回归测试、执行日志、边界条件和可复现验证，补齐必要的测试或验证说明"),
            ("审查与汇总", "复核前面结果，指出风险、遗漏、冲突点与合并建议"),
        ]
        roles: list[tuple[str, str]] = []
        for index in range(count):
            if index == 0:
                roles.append(templates[0])
            elif index == count - 1:
                roles.append(templates[3])
            elif index == 2:
                roles.append(templates[2])
            else:
                label = "实现与修改" if count <= 3 else f"实现与修改 {index}"
                roles.append((label, templates[1][1]))
        return roles
