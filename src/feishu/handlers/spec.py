"""Spec Engine handler — start, status, stop, guide, and automatic recovery."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Optional

from ...card.builders.deep import DeepBuilder
from ...card.models import EngineCardState
from ...card.ui_text import UI_TEXT
from ...spec_engine.models import SpecProjectStatus
from ...spec_engine.persistence import load_engine_state
from ...spec_engine.recovery_lock import try_acquire_recovery_lease
from ...utils.errors import fmt_error, get_error_detail
from ...utils.text import generate_task_id
from ..emoji import EmojiReaction
from ..slash_command_parser import SlashCommandParser
from .base import CardActionContext
from .engine_base import BaseEngineHandler

if TYPE_CHECKING:
    from ...card.protocols import RendererProtocol
    from ...project import ProjectContext
    from ..handler_context import HandlerContext

logger = logging.getLogger(__name__)


class SpecHandler(BaseEngineHandler):
    """Manages the full lifecycle of Spec Engine tasks."""

    def __init__(self, ctx: "HandlerContext", renderer: "RendererProtocol | None" = None) -> None:
        super().__init__(ctx)
        if renderer is None:
            from ..renderers import get_renderer
            renderer = get_renderer("spec", self)
        self.renderer = renderer

    def _get_engine_manager(self):
        return self.ctx.spec_engine_manager

    def _get_engine_name_prefix(self) -> str:
        return "Spec"

    def _get_task_type(self) -> str:
        return "spec_engine"

    def _show_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        self.show_spec_status(message_id, chat_id, project)

    def _create_callbacks(
        self, message_id: str, chat_id: str, project: Optional["ProjectContext"], engine_name: str, root_path: str
    ):
        model_name = self._get_model_name(chat_id, project)
        return self.renderer.create_spec_callbacks(message_id, chat_id, project, engine_name=engine_name, model_name=model_name)

    def _get_model_name(self, chat_id: str, project: Optional["ProjectContext"]) -> str:
        """Resolve model name for subtitle display."""
        from ...utils.engine_identity import resolve_engine_identity
        project_id = project.project_id if project else None
        current_mode = self.ctx.mode_manager.get_mode(chat_id, project_id=project_id)
        identity = resolve_engine_identity(
            mode=current_mode,
            acp_tool_name=getattr(project, "acp_tool_name", None) if project else None,
            acp_model_name=getattr(project, "acp_model_name", None) if project else None,
        )
        return identity.model_name or ""

    def _get_spec_reporter(self):
        """Return the progress reporter used by _safe_execute_engine."""
        return self.ctx.progress_reporter

    def _on_engine_error(
        self,
        error: Exception,
        task_id: str,
        chat_id: str,
        message_id: str,
        project: Optional["ProjectContext"],
        engine_name: str,
        reporter,
        request_id: Optional[str],
        action_prefix: str = "spec",
    ) -> None:
        """Spec-specific error handling: uses renderer.build_error_card + send_card_to_chat."""
        err_msg = get_error_detail(error)
        root_path = getattr(project, "root_path", None) if project else None

        err_msg_type, err_card = self.renderer.build_error_card(
            project=project,
            engine_name=engine_name,
            error_msg=err_msg,
            project_id=project.project_id if project else None,
            engine_project_id=project.project_id if project else root_path,
            footer_note=self.format_ref_note(message_id, request_id) if request_id else None,
            error_details=f"engine={engine_name}; action=spec; request_id={request_id or 'n/a'}",
            detail_action={
                "action": "show_error_details",
                "engine_type": "spec",
                "request_id": request_id or "",
            },
        )
        self.send_card_to_chat(chat_id, err_card, origin_message_id=message_id, request_id=request_id)

    def _refresh_card_view(self, message_id: str, chat_id: str, project=None):
        self.show_spec_status(message_id, chat_id, project, origin_message_id=message_id)

    def _get_ui_state(self, spec_project_id: str) -> dict:
        """Deprecated: Delegate to renderer"""
        return self.renderer.get_ui_state(spec_project_id)

    # ------------------------------------------------------------------
    # Command router
    # ------------------------------------------------------------------
    def handle_spec_command(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        match = SlashCommandParser.parse(text)
        command = match.command if match else ""
        args = match.args if match else ""

        if command == "/spec_status":
            self.show_spec_status(message_id, chat_id, project)
        elif command == "/spec_history":
            self.show_spec_history(message_id, chat_id, text, project)
        elif command == "/spec_metrics":
            self.show_spec_metrics(message_id, chat_id, text, project)
        elif command == "/spec_config":
            self.show_spec_config(message_id, chat_id, project)
        elif command == "/spec_export":
            self.export_spec_report(message_id, chat_id, project)
        elif command == "/spec_save":
            self.save_spec_state(message_id, chat_id, project)
        elif command == "/stop_spec":
            self.stop_spec_engine(message_id, chat_id, project)
        elif command == "/spec_guide" and args:
            self.update_spec_guidance(message_id, chat_id, args, project)
        elif command == "/spec_guide":
            self.reply_text(
                message_id,
                UI_TEXT["spec_cmd_guide_usage"],
            )
        elif command == "/spec" and args:
            self.start_spec_engine(message_id, chat_id, args, project)
        elif command == "/spec":
            self.reply_text(
                message_id,
                UI_TEXT["spec_cmd_help_usage"],
            )
        else:
            self.reply_text(message_id, "❓ 未知的 Spec 命令")

    # ------------------------------------------------------------------
    # start
    # ------------------------------------------------------------------
    def start_spec_engine(
        self, message_id: str, chat_id: str, requirement: str, project: Optional["ProjectContext"] = None
    ):
        project = self._ensure_project(message_id, chat_id, project)
        if not project:
            return
        self._ensure_topic_engine_context(
            mode="spec",
            message_id=message_id,
            chat_id=chat_id,
            project=project,
        )

        root_path = project.root_path if project else self.get_working_dir(chat_id)

        existing = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if existing and existing.is_running:
            self.reply_text(
                message_id,
                "⚠️ 当前项目已有 Spec 任务在执行中\n\n发送 `/spec_status` 查看进度\n发送 `/stop_spec` 停止任务",
            )
            return

        # Spec is autonomous by default: use the active programming tool/model
        # and the configured review strategy without an intermediate picker.
        self._start_spec_engine_now(message_id, chat_id, requirement, project)

    def _start_spec_engine_now(
        self,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: Optional["ProjectContext"] = None,
    ):
        project = self._ensure_project(message_id, chat_id, project)
        if not project:
            return

        root_path = project.root_path if project else self.get_working_dir(chat_id)
        existing = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if existing and existing.is_running:
            self.reply_text(
                message_id,
                "⚠️ 当前项目已有 Spec 任务在执行中\n\n发送 `/spec_status` 查看进度\n发送 `/stop_spec` 停止任务",
            )
            return

        self.add_reaction(message_id, EmojiReaction.on_multi_task_start())

        request_id = self.ensure_request_id(
            message_id, chat_id=chat_id, project_id=(project.project_id if project else None)
        )
        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))

        # Spec renderer owns the first runtime card and task-card orchestration.
        # Visible plan items become first-class cards when the plan arrives;
        # later task/agent events are routed back to those cards.
        model_name = self._get_model_name(chat_id, project) or None
        engine = self.ctx.spec_engine_manager.get_or_create(
            chat_id,
            root_path,
            engine_name=engine_name,
            model_name=model_name,
        )
        project_name = project.project_name if project else os.path.basename(root_path) or "spec"
        task_id = generate_task_id(project_name)

        _on_rate_limit = self.create_rate_limit_callback(
            chat_id, message_id, project, f"Spec({engine_name})", request_id
        )

        def run_spec_engine():
            callbacks = self.renderer.create_spec_callbacks(message_id, chat_id, project, engine_name, model_name=model_name or "")
            engine.execute(requirement, callbacks, task_id=task_id, on_rate_limit=_on_rate_limit)

        def _scheduled_run():
            self._safe_execute_engine(
                executor_func=run_spec_engine,
                task_id=task_id,
                chat_id=chat_id,
                message_id=message_id,
                project=project,
                engine_name=engine_name,
                reporter=self._get_spec_reporter(),
                request_id=request_id,
                action_prefix="spec",
                command_text=f"/spec {requirement}",
            )

        self._submit_engine_task(_scheduled_run, chat_id, message_id, project, request_id, task_id)

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------
    def show_spec_status(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        origin_message_id: Optional[str] = None,
    ):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        root_path = project.root_path if project else self.get_working_dir(chat_id)
        spec_project_id = project.project_id if project else root_path
        self.renderer.update_ui_state(spec_project_id, view_mode="status", view_context={})

        engine = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if not engine or not engine.project:
            try:
                engine_name = self.get_engine_name(
                    chat_id, project_id=(project.project_id if project else None)
                )
                engine = self.ctx.spec_engine_manager.load_or_create_from_disk(
                    chat_id, root_path, engine_name=engine_name
                )
            except Exception:
                logger.debug("failed to load canonical Spec run-state", exc_info=True)

        if (
            engine
            and engine.project
            and engine.project.status in (SpecProjectStatus.ANALYZING, SpecProjectStatus.RUNNING)
            and not engine.is_running
            and getattr(engine, "_resume_meta", None)
        ):
            self._schedule_automatic_recovery(message_id, chat_id, project, engine)

        self.renderer.render_current_view(message_id, chat_id, project, origin_message_id)

    def _schedule_automatic_recovery(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"],
        engine,
    ) -> None:
        with engine._lock:
            if engine.is_running or getattr(engine, "_auto_recovery_claimed", False):
                return
            resume_meta = getattr(engine, "_resume_meta", None) or {}
            checkpoint_path = str(resume_meta.get("state_path") or "")
            if not checkpoint_path:
                if engine.project is not None:
                    engine.project.fail("Spec 自动恢复失败：缺少持久化检查点")
                    engine._resume_meta = None
                    try:
                        engine.save_state()
                    except Exception:
                        logger.warning("failed to persist missing recovery checkpoint", exc_info=True)
                return
            recovery_lease = try_acquire_recovery_lease(checkpoint_path)
            if recovery_lease is None:
                logger.info(
                    "Spec automatic recovery is already owned by another process: %s",
                    checkpoint_path,
                )
                return
            try:
                persisted_project, _review_circuit = load_engine_state(checkpoint_path)
                if persisted_project is None:
                    raise ValueError("Spec checkpoint has no project state")
                engine._project = persisted_project
            except Exception as exc:
                engine.project.fail(
                    f"Spec 自动恢复认领失败: {get_error_detail(exc)}"
                )
                engine._resume_meta = None
                try:
                    engine.save_state(checkpoint_path)
                except Exception:
                    logger.warning(
                        "failed to persist recovery claim failure",
                        exc_info=True,
                    )
                recovery_lease.release()
                return
            if engine.project.status not in (
                SpecProjectStatus.ANALYZING,
                SpecProjectStatus.RUNNING,
            ):
                engine._resume_meta = None
                recovery_lease.release()
                return
            attempts = max(0, int(engine.project.auto_recovery_attempts or 0))
            if attempts >= 1:
                engine.project.fail(
                    "Spec 自动恢复次数已耗尽（最多 1 次），任务明确失败"
                )
                engine._resume_meta = None
                try:
                    engine.save_state(checkpoint_path)
                except Exception:
                    logger.warning("failed to persist exhausted recovery state", exc_info=True)
                recovery_lease.release()
                return
            engine.project.auto_recovery_attempts = attempts + 1
            try:
                # Persist to the exact checkpoint that hydrated this engine
                # before publishing the in-memory claim.
                engine.save_state(checkpoint_path)
            except Exception as exc:
                engine.project.fail(f"Spec 自动恢复认领失败: {get_error_detail(exc)}")
                engine._resume_meta = None
                try:
                    engine.save_state(checkpoint_path)
                except Exception:
                    logger.warning("failed to persist recovery claim failure", exc_info=True)
                recovery_lease.release()
                return
            engine._auto_recovery_claimed = True

        try:
            request_id = self.ensure_request_id(
                message_id,
                chat_id=chat_id,
                project_id=(project.project_id if project else None),
            )
            callbacks = self._create_callbacks(
                message_id, chat_id, project, engine.engine_name, engine.root_path
            )

            def _recover() -> None:
                engine._continue_recovered_run(callbacks)

            def _scheduled() -> None:
                try:
                    self._run_with_repo_lock_or_conflict_card(
                        engine.root_path,
                        chat_id,
                        _recover,
                        message_id,
                        "spec:auto-recover",
                    )
                finally:
                    with engine._lock:
                        engine._auto_recovery_claimed = False
                    recovery_lease.release()

            self._submit_engine_task(
                _scheduled,
                chat_id,
                message_id,
                project,
                request_id,
                engine.project.task_id or engine.project.project_id,
                name_suffix="auto_recover",
            )
        except Exception as exc:
            with engine._lock:
                engine._auto_recovery_claimed = False
                engine.project.fail(f"自动恢复调度失败: {get_error_detail(exc)}")
                engine._resume_meta = None
                try:
                    engine.save_state(checkpoint_path)
                except Exception:
                    logger.debug("failed to persist automatic recovery failure", exc_info=True)
            recovery_lease.release()







    def show_spec_history(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        root_path = project.root_path if project else self.get_working_dir(chat_id)
        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
        try:
            engine = self.ctx.spec_engine_manager.load_or_create_from_disk(chat_id, root_path, engine_name=engine_name)
        except Exception:
            engine = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if not engine or not engine.project:
            msg_type, card_content = DeepBuilder.build_info_card(
                project,
                EngineCardState(
                    title="🗂️ Spec 历史",
                    content="当前没有可查询的 Spec 历史（未运行过或未落盘）\n\n发送 `/spec <需求>` 启动后会自动生成历史。",
                    engine_name=f"Spec({engine_name})",
                    show_buttons=False,
                ),
            )
            self.reply_card(message_id, card_content)
            return

        tail = 20
        try:
            parts = (text or "").strip().split()
            if len(parts) >= 2 and parts[1].isdigit():
                tail = max(1, min(500, int(parts[1])))
        except Exception:
            tail = 20
        content = self.ctx.spec_reporter.format_history(engine.project, tail=tail)
        msg_type, card_content = DeepBuilder.build_info_card(
            project,
            EngineCardState(
                title="🗂️ Spec 历史",
                content=content,
                engine_name=f"Spec({engine.engine_name})",
                show_buttons=False,
            ),
        )
        self.reply_card(message_id, card_content)

    def show_spec_metrics(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        root_path = project.root_path if project else self.get_working_dir(chat_id)
        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
        try:
            engine = self.ctx.spec_engine_manager.load_or_create_from_disk(chat_id, root_path, engine_name=engine_name)
        except Exception:
            engine = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if not engine or not engine.project:
            msg_type, card_content = DeepBuilder.build_info_card(
                project,
                EngineCardState(
                    title="📈 Spec 指标",
                    content="当前没有可查询的 Spec 指标（未运行过或未落盘）\n\n发送 `/spec <需求>` 启动后会自动记录指标。",
                    engine_name=f"Spec({engine_name})",
                    show_buttons=False,
                ),
            )
            self.reply_card(message_id, card_content)
            return

        tail = 20
        try:
            parts = (text or "").strip().split()
            if len(parts) >= 2 and parts[1].isdigit():
                tail = max(1, min(500, int(parts[1])))
        except Exception:
            tail = 20
        content = self.ctx.spec_reporter.format_metrics(engine.project, tail=tail)
        msg_type, card_content = DeepBuilder.build_info_card(
            project,
            EngineCardState(
                title="📈 Spec 指标",
                content=content,
                engine_name=f"Spec({engine.engine_name})",
                show_buttons=False,
            ),
        )
        self.reply_card(message_id, card_content)

    def show_spec_config(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        s = self.settings
        content = (
            "🧩 **Spec 长程配置**\n\n"
            f"- max_cycles: `{getattr(s, 'spec_max_cycles', None)}` (limit=`{getattr(s, 'spec_max_cycles_limit', None)}`) — 最大执行轮次\n"
            f"- min_cycles: `{getattr(s, 'spec_min_cycles', None)}` — 最少执行轮次\n"
            f"- execution_timeout: `{getattr(s, 'spec_execution_timeout', None)}` (秒) — 单次执行总超时\n"
            f"- convergence_window: `{getattr(s, 'spec_convergence_window', None)}` — 收敛检测窗口轮数\n"
            f"- infinite_mode: `{getattr(s, 'spec_infinite_mode', None)}` — 无限模式\n"
            f"- disable_convergence: `{getattr(s, 'spec_disable_convergence', None)}` — 禁用收敛检测\n"
            f"- disable_early_stop: `{getattr(s, 'spec_disable_early_stop', None)}` — 禁用提前停止\n"
            f"- rebuild_session_between_cycles: `{getattr(s, 'spec_rebuild_session_between_cycles', None)}` — 轮间重建会话\n"
            "\n🔍 **发现与收敛**\n\n"
            f"- discovery_enabled: `{getattr(s, 'spec_discovery_enabled', None)}` — 启用发现阶段\n"
            f"- discovery_max_questions: `{getattr(s, 'spec_discovery_max_questions', None)}` — 发现阶段最大问题数\n"
            f"- discovery_force_nonempty: `{getattr(s, 'spec_discovery_force_nonempty', None)}` — 强制非空发现\n"
            f"- discovery_gate_on_satisfied: `{getattr(s, 'spec_discovery_gate_on_satisfied', None)}` — 满足条件时门控\n"
            f"- discovery_max_pending: `{getattr(s, 'spec_discovery_max_pending', None)}` — 最大待处理发现数\n"
            f"- discovery_cooldown_cycles: `{getattr(s, 'spec_discovery_cooldown_cycles', None)}` — 发现冷却轮次\n"
            f"- generated_specs_per_cycle: `{getattr(s, 'spec_generated_specs_per_cycle', None)}` — 每轮生成规格数\n"
            f"- generated_specs_retention: `{getattr(s, 'spec_generated_specs_retention', None)}` — 生成规格保留数\n"
            "\n💾 **持久化与压缩**\n\n"
            f"- cache_root: `{getattr(s, 'spec_cache_root', None) or '~/.cache/ghostAp'}` — Spec 过程文件根目录\n"
            f"- state_file: `{getattr(s, 'spec_state_filename', None)}` — 状态文件名\n"
            f"- state_cycles_tail: `{getattr(s, 'spec_state_cycles_tail', None)}` — 状态文件仅保留最近 N 轮\n"
            f"- state_work_items_tail: `{getattr(s, 'spec_state_work_items_tail', None)}` — 工作项保留最近 N 条\n"
            f"- state_metrics_tail: `{getattr(s, 'spec_state_metrics_tail', None)}` — 指标保留最近 N 条\n"
            f"- artifacts_dir: `{getattr(s, 'spec_artifacts_dirname', None)}` — 产物目录名\n"
            f"- cycle_artifact_retention: `{getattr(s, 'spec_cycle_artifact_retention', None)}` — 产物保留轮数\n"
            f"- cycle_output_max_chars: `{getattr(s, 'spec_cycle_output_max_chars', None)}` — 单轮输出截断字符数\n"
            f"- cycle_tasks_max: `{getattr(s, 'spec_cycle_tasks_max', None)}` — 单轮最大任务数\n"
            f"- phase_output_persist_max_chars: `{getattr(s, 'spec_phase_output_persist_max_chars', None)}` — 阶段产物持久化截断\n"
            f"- persist_phase_artifacts: `{getattr(s, 'spec_persist_phase_artifacts', None)}` — 启用阶段产物持久化\n"
            f"- persist_every_phase: `{getattr(s, 'spec_persist_every_phase', None)}` — 每阶段均持久化\n"
            f"- allow_resume_from_disk: `{getattr(s, 'spec_allow_resume_from_disk', None)}` — 允许从磁盘恢复\n"
            f"- history_log: `{getattr(s, 'spec_history_log_filename', None)}` — 历史日志文件名\n"
            "\n⛔ **终止与失败**\n\n"
            f"- backlog_stuck_window: `{getattr(s, 'spec_backlog_stuck_window', None)}` — 积压卡住窗口 (0=禁用)\n"
            f"- success_ignore_backlog: `{getattr(s, 'spec_success_ignore_backlog', None)}` — 成功时忽略积压\n"
            f"- max_retries: `{getattr(s, 'spec_max_retries', None)}` — 最大重试次数\n"
            f"- max_consecutive_failures: `{getattr(s, 'spec_max_consecutive_failures', None)}` — 最大连续失败\n"
            f"- model_switch_enabled: `{getattr(s, 'spec_model_switch_enabled', None)}` — 启用模型切换\n"
            "\n🔄 **审查与重试**\n\n"
            f"- review_enabled: `{getattr(s, 'spec_review_enabled', None)}` — 启用审查\n"
            f"- review_timeout: `{getattr(s, 'spec_review_timeout', None)}` (秒) — 审查超时\n"
            f"- review_min_timeout: `{getattr(s, 'spec_review_min_timeout', None)}` (秒) — 审查最小超时\n"
            f"- review_hard_floor: `{getattr(s, 'spec_review_hard_floor', None)}` (秒) — 超时硬下限\n"
            f"- review_max_parallel: `{getattr(s, 'spec_review_max_parallel', None)}` — 最大并行审查数\n"
            f"- retry_max_attempts: `{getattr(s, 'spec_review_retry_max_attempts', None)}` (0=禁用重试) — 重试次数\n"
            f"- retry_max_delay: `{getattr(s, 'spec_review_retry_max_delay', None)}` (秒) — 重试最大延迟\n"
            "\n🛡️ **熔断器**\n\n"
            f"- circuit_enabled: `{getattr(s, 'spec_review_failure_circuit_enabled', None)}` — 启用熔断\n"
            f"- max_consecutive: `{getattr(s, 'spec_review_failure_max_consecutive', None)}` — 连续失败触发阈值\n"
            f"- cooldown_cycles: `{getattr(s, 'spec_review_failure_cooldown_cycles', None)}` — 冷却轮次\n"
            f"- max_cooldown_cycles: `{getattr(s, 'spec_review_failure_max_cooldown_cycles', None)}` — 最大冷却轮次\n"
        )
        engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
        msg_type, card_content = DeepBuilder.build_info_card(
            project,
            EngineCardState(
                title="🧩 Spec 配置",
                content=content,
                engine_name=f"Spec({engine_name})",
                show_buttons=False,
            ),
        )
        self.reply_card(message_id, card_content)

    def export_spec_report(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)

        root_path = project.root_path if project else self.get_working_dir(chat_id)
        engine = self.ctx.spec_engine_manager.get(chat_id, root_path)

        # Try to load from disk if not in memory
        if not engine:
            try:
                engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
                engine = self.ctx.spec_engine_manager.load_or_create_from_disk(
                    chat_id, root_path, engine_name=engine_name
                )
            except Exception:
                logger.debug("failed to load engine from disk", exc_info=True)

        if not engine or not engine.project or not engine.project.cycles:
            self.reply_text(message_id, "❌ 当前没有可导出的 Spec 记录")
            return

        spec_project = engine.project
        lines = [f"# Spec Project Export: {spec_project.name}\n"]
        lines.append(f"**Generated at**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Status**: {spec_project.status.value}")
        lines.append(f"**Requirement**: {spec_project.requirement}\n")

        lines.append("## Acceptance Criteria")
        tracker = spec_project.criteria_tracker
        for i, c in enumerate(tracker.criteria):
            mark = "✅" if tracker.satisfied.get(i) else "🔲"
            lines.append(f"- {mark} {c}")
        lines.append("")

        # Latest successful artifacts
        latest_cycle = spec_project.current_cycle
        if latest_cycle:
            lines.append(f"## Latest Cycle (Cycle {latest_cycle.cycle_number})")

            if latest_cycle.spec_content:
                lines.append("### Functional Spec")
                lines.append(latest_cycle.spec_content)
                lines.append("")

            if latest_cycle.plan_content:
                lines.append("### Implementation Plan")
                lines.append(latest_cycle.plan_content)
                lines.append("")

            if latest_cycle.review_result:
                lines.append("### Review Result")
                lines.append(
                    self.ctx.spec_reporter.format_review_result(latest_cycle.review_result, latest_cycle.cycle_number)
                )
                lines.append("")

        # Save to file
        export_filename = f"spec_export_{spec_project.project_id}_{int(time.time())}.md"
        export_path = os.path.join(root_path, export_filename)
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.reply_text(message_id, f"✅ 导出成功: `{export_path}`")
        except Exception as e:
            self.reply_text(message_id, f"❌ 导出失败: {get_error_detail(e)}")

    def save_spec_state(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)
        root_path = project.root_path if project else self.get_working_dir(chat_id)
        engine = self.ctx.spec_engine_manager.get(chat_id, root_path)
        if not engine:
            engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
            engine = self.ctx.spec_engine_manager.get_or_create(chat_id, root_path, engine_name=engine_name)
        if not engine or not engine.project:
            self.reply_text(message_id, "当前没有可保存的 Spec 状态（请先运行 /spec）")
            return
        try:
            path = engine.save_state()
            self.reply_text(message_id, f"✅ 已保存 Spec 状态到: `{path}`")
        except Exception as e:
            self.reply_text(message_id, fmt_error("保存 Spec 状态", e))

    # ------------------------------------------------------------------
    # explicit stop
    # ------------------------------------------------------------------


    def stop_spec_engine(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        def _stop():
            proj = project or self.project_manager.get_active_project(chat_id)
            logger.info(
                "Spec stop requested: chat_id=%s message_id=%s project_id=%s",
                chat_id,
                message_id,
                proj.project_id if proj else None,
            )
            self._stop_engine_generic(message_id, chat_id, project)
            if project is None:
                proj = self.project_manager.get_active_project(chat_id)
            else:
                proj = project
            root_path = proj.root_path if proj else self.get_working_dir(chat_id)
            engine = self._get_engine_manager().get(chat_id, root_path)
            if not engine:
                active = self._get_engine_manager().get_active_engines(chat_id)
                if len(active) == 1:
                    engine = active[0]
            if engine and engine.project:
                try:
                    engine.save_state()
                except Exception:
                    logger.debug("failed to save engine state on stop", exc_info=True)

        self._safe_lifecycle_action(_stop, "stop", chat_id, message_id, project)

    # ------------------------------------------------------------------
    # guidance
    # ------------------------------------------------------------------
    def update_spec_guidance(
        self, message_id: str, chat_id: str, guide_message: str, project: Optional["ProjectContext"] = None
    ):
        if project is None:
            project = self.project_manager.get_active_project(chat_id)

        engine = None
        if project:
            engine = self.ctx.spec_engine_manager.get(chat_id, project.root_path)

        if not engine:
            # Fallback: if there's exactly one runnable engine in the chat
            candidates = [
                e
                for e in self.ctx.spec_engine_manager.list_engines(chat_id)
                if e.project
                and e.project.status == SpecProjectStatus.RUNNING
            ]
            if len(candidates) == 1:
                engine = candidates[0]

        if not engine or not engine.project:
            self.reply_text(
                message_id,
                "⚠️ 当前没有可注入引导的 Spec 任务（运行中）\n\n"
                "请先使用 `/spec <需求>` 启动，或发送 `/spec_status` 查看当前任务",
            )
            return

        if engine.project.status != SpecProjectStatus.RUNNING:
            self.reply_text(
                message_id,
                "⚠️ 当前 Spec 任务状态不支持注入引导（仅支持运行中的任务）\n\n发送 `/spec_status` 查看状态",
            )
            return

        reporter = self.ctx.spec_reporter
        engine_name = engine.engine_name

        # 尝试用 LLM 将原始需求与新引导合并为新的综合目标
        success, result = engine.refine_goal_with_guidance(guide_message)

        if success:
            content = reporter.format_goal_rewritten(guide_message, result)
            title = reporter.get_goal_rewritten_title()
        else:
            # LLM 重写失败，退化为临时注入（不改变持久目标）
            engine.inject_guidance(guide_message)
            content = reporter.format_guidance_injected(guide_message)
            title = reporter.get_guidance_injected_title()

        msg_type, card_content = DeepBuilder.build_info_card(
            project,
            EngineCardState(
                title=title,
                content=content,
                engine_name=f"Spec({engine_name})",
                show_buttons=False,
            ),
        )
        self.send_card_to_chat(chat_id, card_content)

    # ------------------------------------------------------------------
    # UI Interaction Handlers
    # ------------------------------------------------------------------
    def handle_card_action(self, open_message_id: str, open_chat_id: str, action_type: str, value: dict):
        """Handle the remaining Spec card controls; only explicit stop is interactive."""
        project_id = value.get("project_id", "")
        spec_project_id = value.get("deep_project_id", "")
        target_project = (
            self.project_manager.get_project_for_chat(project_id, open_chat_id)
            if project_id else None
        )
        if not target_project and spec_project_id:
            try:
                if os.path.isabs(spec_project_id):
                    target_project = self.project_manager.find_project_by_path(
                        spec_project_id, chat_id=open_chat_id
                    )
                else:
                    target_project = self.project_manager.get_project_for_chat(
                        spec_project_id, open_chat_id
                    )
            except Exception:
                logger.debug("failed to get target_project", exc_info=True)

        self._dispatch_standard_card_action(CardActionContext(
            open_message_id=open_message_id,
            open_chat_id=open_chat_id,
            action_type=action_type,
            value=value,
            prefix="spec",
            action_map={"spec_stop": self.stop_spec_engine},
            toggle_log_method=self._toggle_log,
            switch_mode_method=self._switch_card_mode,
            toggle_ac_method=self._toggle_ac,
            project=target_project,
        ))
