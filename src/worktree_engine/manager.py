from __future__ import annotations

import logging
from typing import Callable, Optional

from ..engine_base import BaseEngineManager
from ..project.context import ProjectContext
from .dispatcher import WorktreeDispatcher
from .git_service import WorktreeGitService
from .models import (
    DeleteWarning,
    WorktreeJourneyStatus,
    WorktreeRuntimeState,
    WorktreeUnitStatus,
    ensure_worktree_state,
    transition_journey_state,
)
from .reporter import WorktreeReporter
from .review_adapter import WorktreeReviewAdapter
from .selection import (
    WorktreeToolOption,
    format_selection_lines,
)
from .selection_controller import WorktreeSelectionController
from .session_store import WorktreeSessionKey, WorktreeSessionStore
from .tool_discovery import WorktreeToolDiscovery

logger = logging.getLogger(__name__)


class WorktreeManager(BaseEngineManager):
    """Worktree session manager.

    Inherits from :class:`BaseEngineManager` to participate in the project-wide
    lock-order hierarchy (``self._lock`` at ``LockLevel.ENGINE_MANAGER``).

    Unlike Deep/Spec managers that maintain per-instance BaseEngine objects,
    WorktreeManager manages state via a session store.  The inherited engine
    registry (``_engines``, ``_chat_keys``) is left unused; the ``_lock`` is
    the primary benefit of the base class relationship.
    """

    def __init__(self, project_manager):
        super().__init__()
        self._project_manager = project_manager
        self._git = WorktreeGitService()
        self._dispatcher = WorktreeDispatcher()
        self._reporter = WorktreeReporter()
        self._review_adapter = WorktreeReviewAdapter()
        self._discovery = WorktreeToolDiscovery()
        self._session_store = WorktreeSessionStore()
        self._selection = WorktreeSelectionController(
            state_getter=self.get_state,
            state_resetter=self.reset_state,
        )

    # ------------------------------------------------------------------
    # BaseEngineManager abstract method (not used for worktree flow)
    # ------------------------------------------------------------------

    def _create_engine(self, chat_id, root_path, agent_type, engine_name, model_name):
        """Not applicable to WorktreeManager — raises NotImplementedError.

        WorktreeManager does not manage BaseEngine instances.  The session
        store and project-level state serve as the equivalent lifecycle.
        """
        raise NotImplementedError(
            "WorktreeManager does not create engine instances; "
            "use get_state(project) for session management."
        )

    # ------------------------------------------------------------------
    # Tool discovery (delegated)
    # ------------------------------------------------------------------

    def get_available_tools(self) -> list[dict]:
        return self._discovery.get_available_tools()

    def get_ttadk_tools(self) -> list[dict]:
        return self._discovery.get_ttadk_tools()

    def get_models_for_tool(
        self,
        tool_name: str,
        provider: str = "ttadk",
        cwd: Optional[str] = None,
        current_model: Optional[str] = None,
    ) -> list[dict]:
        return self._discovery.get_models_for_tool(tool_name, provider, cwd, current_model)

    # ------------------------------------------------------------------
    # Project / state helpers
    # ------------------------------------------------------------------

    def ensure_project(self, chat_id: str, project: Optional[ProjectContext] = None) -> Optional[ProjectContext]:
        if project is not None:
            return project
        if self._project_manager is None:
            return None
        return self._project_manager.get_active_project(chat_id)

    def _session_key_for(self, project: ProjectContext) -> WorktreeSessionKey:
        from ..thread import get_current_thread_id, get_thread_manager

        project_id = str(getattr(project, "project_id", "") or "")
        thread_root_id = get_current_thread_id() or "__default__"
        chat_id = "__default__"
        if thread_root_id != "__default__":
            ctx = get_thread_manager().get(thread_root_id)
            if ctx:
                project_id = ctx.project_id or project_id
                chat_id = ctx.chat_id or chat_id
                thread_root_id = ctx.thread_root_id or thread_root_id
            else:
                thread_root_id = "__default__"
        return WorktreeSessionKey(
            project_id=project_id,
            chat_id=chat_id,
            thread_root_id=thread_root_id,
        )

    def get_session_key(self, project: ProjectContext) -> WorktreeSessionKey:
        return self._session_key_for(project)

    def get_state(self, project: ProjectContext | None = None) -> WorktreeRuntimeState:
        if not isinstance(self, WorktreeManager):
            return ensure_worktree_state(self)
        if project is None:
            raise TypeError("project is required")
        key = self._session_key_for(project)
        state = self._session_store.get(key)
        if state is None:
            legacy_state = getattr(project, "worktree_state", None)
            if key.thread_root_id == "__default__" and isinstance(legacy_state, WorktreeRuntimeState):
                state = legacy_state
                self._session_store.put(key, state)
            else:
                state = self._session_store.get_or_create(key)
        # Transitional compatibility for existing direct tests and old helper
        # paths. The session store remains the owner; the project field mirrors
        # whichever WT topic is active in the current request.
        try:
            project.worktree_state = state
        except Exception:
            pass
        return state

    def reset_state(self, project: ProjectContext, *, preserve_selection: bool = False) -> WorktreeRuntimeState:
        previous = self.get_state(project) if preserve_selection else None
        state = self._session_store.reset(self._session_key_for(project))
        if previous is not None and previous.selection.selected_items:
            state.selection.selected_items = list(previous.selection.selected_items)
            state.selection.active = True
            state.selection.stage = previous.selection.stage
            state.enabled = True
            state.summary_lines = format_selection_lines(state.selection.selected_items)
        try:
            project.worktree_state = state
        except Exception:
            pass
        return state

    # ------------------------------------------------------------------
    # Awaiting-goal helper
    # ------------------------------------------------------------------

    @staticmethod
    def is_awaiting_goal(state: Optional[WorktreeRuntimeState]) -> bool:
        """返回当前 worktree 是否处于“等待用户 goal 输入/确认”的旅程阶段。

        本函数是 Worktree 旅程状态机在“是否等待 goal”维度上的**唯一判定入口**，
        依赖 ``WorktreeJourneyStatus`` 作为真值来源，约定如下真值表（忽略异常场景）：

        - ``IDLE``          → ``False``  （尚未进入 /wt 旅程，或已完全重置）
        - ``PENDING``       → ``True``   （已记录 goal / pending_goal，等待执行或确认）
        - ``AWAITING_GOAL`` → ``True``  （工具/模型已确认，等待同话题第一条普通消息作为目标）
        - ``AUTO_EXECUTING``→ ``True``   （自动执行快速路径中，仍视为处于等待/处理该 goal 的阶段）
        - ``RUNNING``       → ``False``  （已开始实际执行各 worktree 单元，不再拦截新 goal）
        - ``COMPLETED``     → ``False``  （本次旅程已完成，后续消息按普通 SMART 流程处理）
        - ``FAILED``        → ``False``  （本次旅程失败，是否重试由上层显式操作触发）

        结合运行态中的 unit 列表，本函数的**完整判定规则**为：

        - 仅当 ``journey.status`` 处于 ``PENDING`` / ``AWAITING_GOAL`` / ``AUTO_EXECUTING``，且
          至少存在一个 ``unit.status == "ready"`` 的工作单元时返回 ``True``；
        - 其它枚举值（``IDLE``/``RUNNING``/``COMPLETED``/``FAILED``）、缺失字段、
          类型不匹配或内部异常一律返回 ``False``，以避免在 WS 层抛出错误。

        若未来新增 ``WorktreeJourneyStatus`` 枚举成员，**必须同步更新本函数上方
        的真值表契约以及对应单元测试**（遍历全部枚举值的参数化用例），否则视为
        状态机设计不完整。
        """

        if not isinstance(state, WorktreeRuntimeState):
            return False

        try:
            journey = getattr(state, "journey", None)
            if journey is None:
                return False

            status = getattr(journey, "status", None)
            # 显式基于枚举的真值表：仅在 PENDING/AUTO_EXECUTING 阶段才有可能等待 goal。
            awaiting_by_status = {
                WorktreeJourneyStatus.PENDING: True,
                WorktreeJourneyStatus.AWAITING_GOAL: True,
                WorktreeJourneyStatus.AUTO_EXECUTING: True,
                WorktreeJourneyStatus.IDLE: False,
                WorktreeJourneyStatus.RUNNING: False,
                WorktreeJourneyStatus.COMPLETED: False,
                WorktreeJourneyStatus.FAILED: False,
            }

            if awaiting_by_status.get(status) is not True:
                return False
            if status == WorktreeJourneyStatus.AWAITING_GOAL:
                return True

            units = getattr(state, "units", None) or []
            return any(getattr(u, "status", "") == WorktreeUnitStatus.READY for u in units)
        except Exception:
            # 保守兜底：出现异常时视为“未处于等待目标阶段”，交由上层走常规路径。
            return False

    # ------------------------------------------------------------------
    # Journey state helpers
    # ------------------------------------------------------------------

    @staticmethod
    def apply_journey_event(
        state: WorktreeRuntimeState,
        *,
        event: str,
        goal: Optional[str] = None,
        error: Optional[str] = None,
        silent_mode: Optional[bool] = None,
    ) -> None:
        """将语义化旅程事件应用到运行态。

        设计要点：
        - 封装对 ``transition_journey_state`` 的调用，作为单一入口；
        - 始终就地更新 ``state.journey``，避免在 handler 层直接 new/赋值；
        - 不吞掉任何错误信息，由状态机自身在 ``last_error`` 中记录非法迁移。

        该方法为副作用函数（就地更新 state），返回值通过 ``state.journey`` 读取，
        便于在调用链中自然传递完整运行态。
        """

        try:
            state.journey = transition_journey_state(
                state.journey,
                event=event,
                goal=goal,
                error=error,
                silent_mode=silent_mode,
            )
        except Exception:
            # 出现意外异常时，不破坏现有运行态，仅在 last_error 中做一个兜底标记，
            # 具体错误信息由上层日志记录负责，避免在状态机内部引入 logging 依赖。
            msg = f"旅程事件应用失败: {event or ''}"
            state.journey.last_error = state.journey.last_error or msg

    def start_selection(self, project: ProjectContext, goal: str = "") -> WorktreeRuntimeState:
        return self._selection.start_selection(project, goal)

    def reset_selection(self, project: ProjectContext) -> WorktreeRuntimeState:
        return self._selection.reset_selection(project)

    def select_tool(self, project: ProjectContext, option: WorktreeToolOption) -> WorktreeRuntimeState:
        return self._selection.select_tool(project, option)

    def add_pending_item(
        self,
        project: ProjectContext,
        *,
        model_name: Optional[str] = None,
        model_display_name: Optional[str] = None,
    ) -> tuple[WorktreeRuntimeState, bool, str]:
        return self._selection.add_pending_item(project, model_name=model_name, model_display_name=model_display_name)

    def back_to_tool_selection(self, project: ProjectContext) -> WorktreeRuntimeState:
        return self._selection.back_to_tool_selection(project)

    def remove_selected_item(
        self, project: ProjectContext, selection_key: str
    ) -> tuple[WorktreeRuntimeState, bool, str]:
        return self._selection.remove_selected_item(project, selection_key)

    def clear_selected_items(
        self, project: ProjectContext
    ) -> tuple[WorktreeRuntimeState, int, str]:
        return self._selection.clear_selected_items(project)

    def finalize_selection(self, project: ProjectContext) -> WorktreeRuntimeState:
        return self._selection.finalize_selection(project)

    def set_pending_goal(self, project: ProjectContext, goal: str) -> None:
        self._selection.set_pending_goal(project, goal)

    def mark_units_ready(self, project: ProjectContext) -> None:
        self._selection.mark_units_ready(project)

    def ensure_worktrees(
        self,
        project: ProjectContext,
        *,
        custom_base_dir: Optional[str] = None,
    ) -> WorktreeRuntimeState:
        state = self.get_state(project)
        session_key = self._session_key_for(project)
        if not state.selection.selected_items:
            state.last_error = "当前没有可创建 worktree 的工具-模型组合"
            return state
        try:
            repo_state, units = self._git.create_units(
                root_path=project.root_path,
                count=len(state.selection.selected_items),
                base_branch=state.base_branch or None,
                custom_base_dir=custom_base_dir,
                session_slug=session_key.slug if session_key.thread_root_id != "__default__" else "",
            )
        except Exception as exc:
            from ..utils.errors import get_error_detail
            state.enabled = False
            state.merge_entry_ready = False
            state.last_error = get_error_detail(exc)
            state.summary_lines = [f"- worktree 创建失败：{state.last_error}"]
            return state

        state.enabled = True
        state.git_initialized_locally = bool(repo_state.initialized)
        state.git_root = repo_state.repo_root
        state.base_branch = repo_state.base_branch
        state.units = units
        state.summary_lines = format_selection_lines(state.selection.selected_items)
        state.last_error = ""
        return state

    def plan_goal(self, project: ProjectContext, goal: str) -> WorktreeRuntimeState:
        """根据用户目标规划各 worktree 单元并更新旅程状态。

        语义：
        - 通过 ``goal_created`` 事件将目标记录到 ``journey.goal``；
        - 将旅程从 IDLE 推进到 PENDING；
        - 调用 dispatcher 生成 unit 级计划，并通过 reporter 刷新摘要信息。
        """

        state = self.get_state(project)
        normalized_goal = str(goal or "").strip()
        # 先更新旅程高层状态，再进行具体的 unit 规划。
        self.apply_journey_event(state, event="goal_created", goal=normalized_goal)
        state.units = self._dispatcher.plan_user_goal(
            normalized_goal, state.units, state.selection.selected_items
        )
        return self._reporter.refresh_state(state)

    def execute_goal(
        self,
        project: ProjectContext,
        goal: str,
        *,
        timeout: Optional[int] = None,
        on_unit_update: Optional[Callable] = None,
    ) -> WorktreeRuntimeState:
        """执行已规划的 worktree 旅程，并推进高层旅程状态机。

        状态机语义：
        - 当存在可执行单元且无致命错误时，进入 ``execution_started``（RUNNING）；
        - 执行成功后触发 ``execution_succeeded``（COMPLETED）；
        - 执行过程中出现异常时触发 ``execution_failed``（FAILED），并在 ``state.last_error`` 中保留细节。
        """

        state = self.get_state(project)
        if not state.units:
            state = self.ensure_worktrees(project)
        if state.last_error or not state.units:
            if not state.last_error:
                state.last_error = "当前没有可执行的 worktree 工作单元"
            return self._reporter.refresh_state(state)

        # 规划目标并进入 RUNNING 之前，先更新旅程事件流。
        state = self.plan_goal(project, goal)
        state.iteration_count += 1
        self.apply_journey_event(state, event="execution_started", goal=state.journey.goal)

        try:
            state.units = self._dispatcher.execute_units(
                state.units, timeout=timeout, on_unit_update=on_unit_update,
            )
            self._finalize_execution_truth(
                state,
                on_unit_update=on_unit_update,
            )
        except Exception as exc:
            from ..utils.errors import get_error_detail

            state.last_error = get_error_detail(exc)
            self.apply_journey_event(state, event="execution_failed", error=state.last_error)
        return self._reporter.refresh_state(state)

    @staticmethod
    def _incomplete_required_units(
        state: WorktreeRuntimeState,
    ) -> list:
        return [
            unit
            for unit in state.units
            if unit.status != WorktreeUnitStatus.COMPLETED
        ]

    def _finalize_execution_truth(
        self,
        state: WorktreeRuntimeState,
        *,
        on_unit_update: Optional[Callable] = None,
    ) -> None:
        incomplete = self._incomplete_required_units(state)
        if incomplete:
            detail = ", ".join(
                f"{unit.unit_id}:{getattr(unit.status, 'value', unit.status)}"
                for unit in incomplete
            )
            state.last_error = f"必要 Worktree 单元未成功完成: {detail}"
            state.review_plan = {}
            state.review_outcome = {
                "verdict": "INCONCLUSIVE",
                "passed": False,
                "error_code": "required_units_incomplete",
                "unit_outcomes": [],
            }
            self.apply_journey_event(
                state,
                event="execution_failed",
                error=state.last_error,
            )
            return

        changed_files = [
            str(unit.worktree_path or unit.branch_name or unit.unit_id)
            for unit in state.units
            if getattr(unit, "has_changes", False)
        ]
        review_plan = self._review_adapter.plan_roles(
            goal=state.journey.goal,
            changed_files=changed_files,
        )
        state.review_plan = review_plan.to_dict()
        review_outcome = self._review_adapter.review_units(
            goal=state.journey.goal,
            units=state.units,
            base_branch=state.base_branch or None,
            on_unit_update=on_unit_update,
        )
        state.review_outcome = review_outcome.to_dict()
        if not review_outcome.passed:
            verdict = getattr(review_outcome.verdict, "value", review_outcome.verdict)
            reason = review_outcome.error_code or str(verdict or "inconclusive")
            state.last_error = f"Worktree 独立评审未通过: {reason}"
            self.apply_journey_event(
                state,
                event="execution_failed",
                error=state.last_error,
            )
            return

        state.last_error = ""
        self.apply_journey_event(state, event="execution_succeeded")

    # ------------------------------------------------------------------
    # Retry failed units
    # ------------------------------------------------------------------

    def retry_failed_units(
        self,
        project: ProjectContext,
        *,
        timeout: Optional[int] = None,
        on_unit_update: Optional[Callable] = None,
    ) -> WorktreeRuntimeState:
        """Re-execute only the failed units, preserving completed ones.

        Reuses ``journey.goal`` — caller does not need to re-supply the goal.
        """
        state = self.get_state(project)

        # Guard: need a goal from the previous run
        if not state.journey.goal:
            state.last_error = "没有可重试的目标（上次执行目标为空）"
            return self._reporter.refresh_state(state)

        # Guard: must have failed/cancelled units to retry
        failed_units = [u for u in state.units if u.status in (WorktreeUnitStatus.FAILED, WorktreeUnitStatus.CANCELLED)]
        if not failed_units:
            state.last_error = "当前没有需要重试的工作单元"
            return self._reporter.refresh_state(state)

        # Guard: no units should be running (concurrent safety)
        if any(u.status == WorktreeUnitStatus.RUNNING for u in state.units):
            state.last_error = "存在正在执行的单元，请等待执行完成后再重试"
            return self._reporter.refresh_state(state)

        # Reset failed units' state fields
        state.iteration_count += 1
        for unit in failed_units:
            unit._cancel_event.clear()  # Must clear before retry — otherwise _run_single_unit skips immediately
            unit.status = WorktreeUnitStatus.PENDING
            unit.error = ""
            unit.summary = ""
            unit.has_changes = False

        # Re-plan only failed units (keep completed units unchanged)
        failed_units = self._dispatcher.plan_user_goal(
            state.journey.goal, failed_units, state.selection.selected_items
        )

        # Execute only the re-planned (previously-failed) units
        self.apply_journey_event(
            state,
            event="goal_created",
            goal=state.journey.goal,
        )
        self.apply_journey_event(
            state,
            event="execution_started",
            goal=state.journey.goal,
        )
        try:
            self._dispatcher.execute_units(
                failed_units, timeout=timeout, on_unit_update=on_unit_update,
            )
            self._finalize_execution_truth(
                state,
                on_unit_update=on_unit_update,
            )
        except Exception as exc:
            from ..utils.errors import get_error_detail
            state.last_error = get_error_detail(exc)
            self.apply_journey_event(
                state,
                event="execution_failed",
                error=state.last_error,
            )

        return self._reporter.refresh_state(state)

    # ------------------------------------------------------------------
    # Merge / cleanup
    # ------------------------------------------------------------------

    def merge_to_base(self, project: ProjectContext) -> tuple[WorktreeRuntimeState, list[dict]]:
        """Merge each completed worktree branch into *base_branch*.

        Returns ``(state, merge_results)`` where each result is a dict with
        ``display_name``, ``branch_name``, ``success``, ``detail``.
        """
        state = self.get_state(project)
        if not state.units or not state.base_branch:
            state.last_error = "没有可合并的 worktree 或基础分支未设置"
            return self._reporter.refresh_state(state), []

        review_passed = bool((state.review_outcome or {}).get("passed"))
        all_required_completed = not self._incomplete_required_units(state)
        terminal_completed = state.journey.status == WorktreeJourneyStatus.COMPLETED
        if not (
            state.merge_entry_ready
            and review_passed
            and all_required_completed
            and terminal_completed
        ):
            state.merge_entry_ready = False
            state.last_error = "Worktree 终态或独立评审未满足，已阻止自动合并"
            blocked_results = [
                {
                    "display_name": unit.display_name,
                    "branch": unit.branch_name,
                    "branch_name": unit.branch_name,
                    "success": False,
                    "detail": state.last_error,
                }
                for unit in state.units
            ]
            return self._reporter.refresh_state(state), blocked_results

        merge_results: list[dict] = []
        for unit in state.units:
            if unit.status != WorktreeUnitStatus.COMPLETED:
                merge_results.append(
                    {
                        "display_name": unit.display_name,
                        "branch": unit.branch_name,
                        "branch_name": unit.branch_name,
                        "success": False,
                        "detail": "跳过（单元未完成，需额外修复）",
                    }
                )
                continue
            if not unit.has_changes:
                merge_results.append(
                    {
                        "display_name": unit.display_name,
                        "branch": unit.branch_name,
                        "branch_name": unit.branch_name,
                        "success": True,
                        "skipped": True,
                        "detail": "跳过（无代码变更）",
                    }
                )
                continue
            try:
                if unit.worktree_path:
                    self._git.commit_worktree_changes(
                        unit.worktree_path,
                        f"Apply worktree unit {unit.unit_id}",
                    )
                ok, conflicts = self._git.merge_branch(state.git_root, unit.branch_name, state.base_branch)
                if ok:
                    merge_results.append({
                        "display_name": unit.display_name,
                        "branch": unit.branch_name,
                        "branch_name": unit.branch_name,
                        "success": True,
                        "conflict_policy": "worktree_branch_wins",
                        "detail": "合并成功；如有同文件冲突，已自动优先采用 Worktree 分支变更",
                    })
                else:
                    merge_results.append({
                        "display_name": unit.display_name,
                        "branch": unit.branch_name,
                        "branch_name": unit.branch_name,
                        "success": False,
                        "detail": f"自动冲突处理未完成，保留现场；冲突文件: {', '.join(conflicts)}",
                    })
            except Exception as exc:
                from ..utils.errors import get_error_detail
                merge_results.append({
                    "display_name": unit.display_name,
                    "branch": unit.branch_name,
                    "branch_name": unit.branch_name,
                    "success": False,
                    "detail": get_error_detail(exc),
                })

        state.last_error = ""
        state.merge_entry_ready = False
        return self._reporter.refresh_state(state), merge_results

    def cleanup_worktrees(
        self,
        project: ProjectContext,
        *,
        force: bool = False,
    ) -> tuple[WorktreeRuntimeState, list[DeleteWarning]]:
        """Remove all worktree directories and branches, reset state.

        When *force* is ``False``, checks each worktree for safety first.
        Returns ``(state, warnings)`` — warnings is non-empty if any
        worktree was skipped due to uncommitted changes / unmerged branches.
        """
        state = self.get_state(project)
        warnings: list[DeleteWarning] = []
        repo_root = state.git_root or project.root_path
        for unit in state.units:
            try:
                if unit.worktree_path:
                    warning = self._git.remove_worktree(
                        repo_root,
                        unit.worktree_path,
                        force=force,
                        base_branch=state.base_branch or None,
                    )
                    if warning is not None:
                        warnings.append(warning)
                        continue
                if unit.branch_name:
                    self._git.remove_branch(repo_root, unit.branch_name)
            except Exception:
                logger.warning(
                    "清理 worktree 失败: unit=%s path=%s branch=%s",
                    unit.unit_id, unit.worktree_path, unit.branch_name, exc_info=True,
                )
        if not warnings:
            # Only run optimize_storage and reset state when all worktrees cleaned
            try:
                self._git.optimize_storage(repo_root)
            except Exception:
                logger.warning("optimize_storage failed", exc_info=True)
            return self.reset_state(project, preserve_selection=True), []
        return self._reporter.refresh_state(state), warnings

    # ------------------------------------------------------------------
    # List / sync
    # ------------------------------------------------------------------

    def list_worktrees(self, project: ProjectContext) -> tuple[WorktreeRuntimeState, str]:
        """List all worktrees and return formatted table.

        Returns ``(state, table_string)``.
        """
        state = self.get_state(project)
        repo_root = state.git_root or project.root_path
        entries = self._git.list_worktrees(repo_root)
        table = self._reporter.format_worktree_table(entries)
        return state, table

    def sync_worktree(
        self,
        project: ProjectContext,
        worktree_path: str,
        branch: Optional[str] = None,
        *,
        force: bool = False,
    ) -> tuple[WorktreeRuntimeState, Optional[DeleteWarning]]:
        """Sync a worktree to its remote branch's latest state.

        Returns ``(state, warning)`` — warning is non-None if worktree
        has uncommitted changes and *force* is ``False``.
        """
        state = self.get_state(project)
        repo_root = state.git_root or project.root_path
        warning = self._git.sync_worktree(
            repo_root, worktree_path, branch, force=force,
        )
        return state, warning
