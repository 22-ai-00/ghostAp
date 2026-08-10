"""Base engine handler for common engine lifecycle management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ...tasking import TaskPriority, TaskSpec
from .base import BaseHandler

if TYPE_CHECKING:
    from ...card.protocols import RendererProtocol
    from ...project import ProjectContext

logger = logging.getLogger(__name__)


class BaseEngineHandler(BaseHandler):
    """
    Abstract base class for engine handlers (DeepHandler, SpecHandler).
    Provides common lifecycle management logic.
    """

    renderer: RendererProtocol  # Subclasses MUST assign in __init__

    # ------------------------------------------------------------------
    # Repo-lock + conflict-card helper (DRY for scheduled closures)
    # ------------------------------------------------------------------

    def _run_with_repo_lock_or_conflict_card(
        self, root_path: str, chat_id: str, body_fn, message_id: str, command_text: str,
    ) -> None:
        """Execute *body_fn* under the repo lock, sending a conflict card on failure.

        Delegates to :meth:`LockHelper.handle_lock_conflict`.
        """
        self.lock_helper.handle_lock_conflict(
            body_fn, root_path, chat_id, message_id, command_text,
        )

    def _run_engine_with_conflict_card(
        self,
        body_fn,
        project: Optional["ProjectContext"],
        chat_id: str,
        message_id: str,
        command_text: str,
    ) -> None:
        """Engine-level facade for repo-lock guarded scheduled bodies."""
        root_path = project.root_path if project else self.get_working_dir(chat_id)
        self._run_with_repo_lock_or_conflict_card(root_path, chat_id, body_fn, message_id, command_text)

    def _build_engine_callbacks(
        self, message_id: str, chat_id: str, project: Optional["ProjectContext"], engine_name: str, root_path: str
    ):
        """Engine-level facade for subclass callback factories."""
        return self._create_callbacks(message_id, chat_id, project, engine_name, root_path)

    def _show_engine_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Engine-level facade for subclass status rendering."""
        return self._show_status(message_id, chat_id, project=project)

    def _get_engine_manager(self):
        """Subclasses must return their specific engine manager."""
        raise NotImplementedError

    def _get_engine_name_prefix(self) -> str:
        """Subclasses must return their engine name prefix (e.g. 'Deep', 'Spec')."""
        raise NotImplementedError

    def _get_task_type(self) -> str:
        """Subclasses must return their task type (e.g. 'deep_engine', 'spec_engine')."""
        raise NotImplementedError

    def _show_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Subclasses must implement status display."""
        raise NotImplementedError

    def _create_callbacks(
        self, message_id: str, chat_id: str, project: Optional["ProjectContext"], engine_name: str, root_path: str
    ):
        """Subclasses must implement callback creation."""
        raise NotImplementedError

    def _ensure_topic_engine_context(
        self,
        *,
        mode: str,
        message_id: str,
        chat_id: str,
        project: "ProjectContext",
    ) -> str:
        """Bind Deep/Spec style engines to the current Feishu topic.

        Engine modes are topic-scoped strategies. If a command starts outside an
        existing topic, the command message itself becomes the canonical root.
        """
        from ...thread import get_current_thread_id, set_current_thread_id

        thread_root_id = get_current_thread_id() or message_id
        ctx = self.ctx.thread_manager.get(thread_root_id)
        if not ctx:
            self.ctx.thread_manager.bind_engine(
                thread_root_id=thread_root_id,
                chat_id=chat_id,
                project_id=project.project_id,
                mode=mode,
            )
        elif ctx.mode != mode:
            self.ctx.thread_manager.bind_engine(
                thread_root_id=ctx.thread_root_id,
                chat_id=ctx.chat_id,
                project_id=ctx.project_id,
                mode=mode,
                tool_name=ctx.tool_name,
                model_name=ctx.model_name,
            )
            thread_root_id = ctx.thread_root_id
        set_current_thread_id(thread_root_id)
        return thread_root_id

    def _stop_engine_generic(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Generic stop logic."""
        if project is None:
            project = self.project_manager.get_active_project(chat_id)

        root_path = project.root_path if project else self.get_working_dir(chat_id)
        manager = self._get_engine_manager()
        engine = manager.get(chat_id, root_path)

        if not engine:
            running = manager.get_active_engines(chat_id)
            if len(running) == 1:
                engine = running[0]
            elif len(running) > 1:
                self.reply_text(
                    message_id, f"⚠️ 有多个项目正在执行 {self._get_engine_name_prefix()} 任务，请先切换项目再停止"
                )
                return

        if not engine or not engine.is_running:
            self.reply_text(message_id, f"📊 当前没有正在执行的 {self._get_engine_name_prefix()} 任务")
            return

        engine.stop()
        self._show_status(message_id, chat_id, project=project)

    def _safe_execute_engine(
        self,
        executor_func: callable,
        task_id: str,
        chat_id: str,
        message_id: str,
        project: Optional["ProjectContext"],
        engine_name: str,
        reporter: Any,
        request_id: Optional[str],
        action_prefix: str = "deep",
        command_text: str = "",
    ):
        """
        Execute engine logic with standardized error handling and reporting.

        Acquires the repo lock, runs *executor_func*, and on conflict renders
        a conflict card.  Subclasses can override ``_on_engine_error()`` to
        customize error card building and delivery.
        """
        import asyncio

        from ...repo_lock import LockConflictError
        from ...utils.errors import get_error_detail

        root_path = getattr(project, "root_path", None) if project else None

        def _body():
            try:
                executor_func()
            except NotImplementedError:
                logger.error("%s Engine: NotImplementedError in executor_func (task_id=%s)", self._get_engine_name_prefix(), task_id)
                self.reply_text(message_id, "系统升级中，请重试")
                return
            except LockConflictError:
                raise
            except Exception as e:
                if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                    logger.warning(f"{self._get_engine_name_prefix()} Engine 执行超时 (task_id={task_id}): {get_error_detail(e)}")
                else:
                    logger.error(f"{self._get_engine_name_prefix()} Engine 执行异常: {get_error_detail(e)}", exc_info=True)

                self._on_engine_error(
                    error=e,
                    task_id=task_id,
                    chat_id=chat_id,
                    message_id=message_id,
                    project=project,
                    engine_name=engine_name,
                    reporter=reporter,
                    request_id=request_id,
                    action_prefix=action_prefix,
                )

        self.lock_helper.handle_lock_conflict(
            _body, root_path, chat_id, message_id, command_text,
        )

    def _on_engine_error(
        self,
        error: Exception,
        task_id: str,
        chat_id: str,
        message_id: str,
        project: Optional["ProjectContext"],
        engine_name: str,
        reporter: Any,
        request_id: Optional[str],
        action_prefix: str = "deep",
    ) -> None:
        """Handle engine execution error — dispatch through CardSession pipeline.

        Attempts to dispatch a FAILED event through the renderer's active session
        so that error cards benefit from session lifecycle hooks (emoji, context
        persistence). Falls back to plain reply_text if no session is available.
        """
        from ...card.events import CardEvent
        from ...utils.errors import get_error_detail

        err_msg = get_error_detail(error)
        error_content = reporter.format_error(err_msg)

        ref_note = self.format_ref_note(message_id, request_id) if request_id else ""
        final_content = f"{error_content}\n\n{ref_note}" if ref_note else error_content

        # Try dispatching through the session pipeline for full hook support
        session = self.renderer.get_active_session() if hasattr(self, "renderer") else None
        if session is not None and not getattr(session, "closed", True):
            try:
                session.dispatch(
                    CardEvent.failed(
                        final_content,
                        details=f"engine={engine_name}; task_id={task_id}; request_id={request_id or 'n/a'}",
                        detail_action={
                            "action": "show_error_details",
                            "engine_type": action_prefix,
                            "task_id": task_id,
                            "request_id": request_id or "",
                        },
                    )
                )
                return
            except Exception as dispatch_err:
                logger.debug(
                    "%s Engine: session.dispatch(failed) raised, falling back to reply_text: %s",
                    self._get_engine_name_prefix(), dispatch_err,
                )

        # Fallback: plain text reply (ensures errors are never lost)
        try:
            error_title = reporter.get_error_title()
            self.reply_text(message_id, f"{error_title}\n\n{final_content}")
        except Exception as send_err:
            logger.error(
                "%s Engine 发送错误消息失败: %s", self._get_engine_name_prefix(), send_err
            )

    # ------------------------------------------------------------------
    # UI toggle helpers (DRY for Deep/Spec toggle methods)
    # ------------------------------------------------------------------

    def _refresh_card_view(
        self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None,
    ):
        """Refresh engine card after UI state change.  Subclasses may override."""
        self._show_status(message_id, chat_id, project)

    def _toggle_log(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        engine_project_id: Optional[str] = None,
        expanded: bool = False,
    ):
        """Toggle log expansion for an engine project."""
        if engine_project_id:
            self.renderer.update_ui_state(engine_project_id, expanded=expanded)
            self._refresh_card_view(message_id, chat_id, project)

    def _toggle_ac(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        engine_project_id: Optional[str] = None,
        expand_ac: bool = False,
    ):
        """Toggle acceptance-criteria expansion."""
        if engine_project_id:
            self.renderer.update_ui_state(engine_project_id, expand_ac=expand_ac)
            self._refresh_card_view(message_id, chat_id, project)

    def _switch_card_mode(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        engine_project_id: Optional[str] = None,
        compact: bool = False,
    ):
        """Switch card display mode (full / compact)."""
        if engine_project_id:
            self.renderer.update_ui_state(engine_project_id, compact=compact)
            self._refresh_card_view(message_id, chat_id, project)
        else:
            # New CardSession path: buttons rendered by CardSession don't carry
            # engine_project_id — dispatch MODE_TOGGLED directly to the active session.
            session = self.renderer.get_active_session()
            if session and not session.closed:
                from src.card.events import CardEvent
                session.dispatch(CardEvent.mode_toggled(compact=compact))

    # ------------------------------------------------------------------
    # Project auto-creation helper
    # ------------------------------------------------------------------

    def _ensure_project(
        self, message_id: str, chat_id: str, project: Optional["ProjectContext"],
    ) -> Optional["ProjectContext"]:
        """Ensure a project exists, auto-creating one if *project* is None.

        Returns the project on success, or ``None`` if creation failed (error
        reply has already been sent to the user).
        """
        if project:
            return project
        working_dir = self.get_working_dir(chat_id)
        try:
            project, is_new = self.project_manager.get_or_create_project_for_path(working_dir, chat_id)
            if is_new:
                logger.info(
                    "%s 自动创建项目: %s @ %s",
                    self._get_engine_name_prefix(), project.project_name, project.root_path,
                )
            return project
        except Exception as e:
            from ...utils.errors import get_error_detail

            self.reply_error(message_id, get_error_detail(e), title="创建项目失败")
            return None

    # ------------------------------------------------------------------
    # Task submission helper
    # ------------------------------------------------------------------

    def _submit_engine_task(
        self,
        run_fn: callable,
        chat_id: str,
        message_id: str,
        project: Optional["ProjectContext"],
        request_id: Optional[str],
        task_id: Optional[str] = None,
        *,
        name_suffix: str = "run",
        priority: TaskPriority = TaskPriority.NORMAL,
    ):
        """Build a TaskSpec, submit to the scheduler, and link the task.

        Covers the common start/recover pattern shared by Deep/Spec handlers.
        """
        task_type = self._get_task_type()
        prefix = task_type.split("_")[0]  # deep, spec
        root_path = project.root_path if project else self.get_working_dir(chat_id)

        spec = TaskSpec(
            chat_id=chat_id,
            queue_key=f"{chat_id}:{prefix}:{project.project_id if project else root_path}",
            name=f"{task_type}_{name_suffix}",
            task_type=task_type,
            project_id=project.project_id if project else None,
            message_id=message_id,
            origin_message_id=message_id,
            request_id=request_id,
            task_id=task_id or None,
            priority=priority,
        )
        handle = self.scheduler.submit(spec, lambda ctx: run_fn())
        try:
            self.ctx.message_linker.link_task(message_id, handle.run_id)
        except Exception as e:
            logger.debug(
                "link_task失败(%s_%s): message_id=%s, run_id=%s, err=%s",
                task_type, name_suffix, message_id, handle.run_id, e,
            )
        return handle

    def _safe_lifecycle_action(
        self,
        action_func: callable,
        action_name: str,
        chat_id: str,
        message_id: str,
        project: Optional["ProjectContext"] = None,
    ):
        """
        Execute explicit lifecycle actions with safe error handling.

        Args:
            action_func: Lambda or function to execute the action
            action_name: Name of action for logging (currently "stop")
            chat_id: Chat ID
            message_id: Origin message ID
            project: Project context (optional)
        """
        import asyncio

        try:
            action_func()
        except Exception as e:
            engine_prefix = self._get_engine_name_prefix()
            from ...utils.errors import get_error_detail

            if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                logger.warning(f"{engine_prefix} {action_name} 操作超时: {get_error_detail(e)}")
            else:
                logger.error(f"{engine_prefix} {action_name} 操作异常: {get_error_detail(e)}", exc_info=True)

            # Send simple error reply for lifecycle actions
            err_detail = get_error_detail(e)
            self.reply_text(message_id, f"❌ {action_name}失败: {err_detail}")
