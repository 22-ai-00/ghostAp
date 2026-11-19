"""Workflow script generation, execution, stop, and callback helpers."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import threading

    from ...project import ProjectContext
    from ...workflow_engine.models import WorkflowProject
    from .workflow import _WorkflowGenerationActivityBuffer, _WorkflowLifecycleOwner

logger = logging.getLogger(__name__)


def _workflow_module():
    """Return the workflow handler module for runtime-patchable symbols."""
    from . import workflow as _wf_mod

    return _wf_mod


class WorkflowExecutionMixin:
    """Mixin providing workflow script generation, execution, and stop flows."""

    @staticmethod
    def _is_current_generation_session(engine: Any | None, expected_session_key: str | None) -> bool:
        """Return True if an async script-generation task still owns the session."""
        if not expected_session_key:
            return False
        if not engine or not getattr(engine, "project", None):
            return False

        from ...workflow_engine.models import WorkflowStatus

        project = engine.project
        pending = getattr(project, "pending", None)
        stored_session_key = getattr(pending, "engine_session_key", "") if pending else ""
        return (
            project.status == WorkflowStatus.GENERATING_SCRIPT
            and bool(stored_session_key)
            and stored_session_key == expected_session_key
        )

    @classmethod
    def _generation_owner_is_active(
        cls,
        engine: Any,
        owner: _WorkflowLifecycleOwner,
    ) -> bool:
        """Check generation ownership under the engine's lifecycle lock."""
        with engine._lock:
            return bool(
                vars(engine).get("_script_generation_owner") is owner
                and not owner.stop_event.is_set()
                and cls._is_current_generation_session(engine, owner.session_key)
            )

    @classmethod
    def _commit_generated_workflow_if_current(
        cls,
        engine: Any,
        owner: _WorkflowLifecycleOwner,
        pending: Any,
    ) -> bool:
        """CAS a verified generated workflow into the owned generation session."""
        with engine._lock:
            current_pending = engine.project.pending if engine.project else None
            if (
                vars(engine).get("_script_generation_owner") is not owner
                or owner.stop_event.is_set()
                or not cls._is_current_generation_session(engine, owner.session_key)
                or current_pending is None
                or current_pending.project_id != owner.project_id
                or pending.project_id != owner.project_id
            ):
                return False
            engine.project.pending = pending
            return True

    def _generate_and_start_workflow(
            self,
            message_id: str,
            chat_id: str,
            requirement: str,
            project: Optional["ProjectContext"],
            root_path: str,
            selected_tools: list[str] | None,
            *,
            expected_session_key: str | None = None,
        ) -> None:
            """Generate, validate, adopt registered tools, and start execution."""
            from ...thread import get_current_sender_id
            from ...workflow_engine.constants import DEFAULT_ORCHESTRATOR_AGENT
            from ...workflow_engine.models import PendingWorkflow, WorkflowStatus

            if not expected_session_key:
                logger.error("[workflow] Refusing script generation without a session owner")
                return

            engine_name = self.get_engine_name(chat_id, project_id=(project.project_id if project else None))
            engine = self.ctx.workflow_engine_manager.get_or_create(
                chat_id,
                root_path,
                engine_name=engine_name,
            )
            owner = _workflow_module()._WorkflowLifecycleOwner(
                session_key=expected_session_key,
                source_script_path=self._new_workflow_script_path(root_path),
            )
            previous_owner: _WorkflowLifecycleOwner | None = None
            generation_context: dict[str, Any] = {}
            with engine._lock:
                if not self._is_current_generation_session(engine, expected_session_key):
                    logger.info(
                        "[workflow] Dropping stale script generation before delivery (expected_session=%s)",
                        expected_session_key[:8],
                    )
                    return
                current_owner = vars(engine).get("_script_generation_owner")
                if isinstance(current_owner, _workflow_module()._WorkflowLifecycleOwner) and not current_owner.stop_event.is_set():
                    if current_owner.session_key == expected_session_key:
                        owner = current_owner
                    else:
                        previous_owner = current_owner
                if previous_owner is not None:
                    self._retire_workflow_owner(engine, previous_owner)
                owner.claimed_event.set()
                object.__setattr__(
                    owner,
                    "worker_thread_id",
                    _workflow_module().threading.get_ident(),
                )
                engine._script_generation_owner = owner
                existing_pending = engine.project.pending
                generation_context = {
                    "initiator_user_id": (
                        existing_pending.initiator_user_id
                        if existing_pending and getattr(existing_pending, "initiator_user_id", None)
                        else get_current_sender_id() or ""
                    ),
                    "orchestrator_agent": (
                        existing_pending.orchestrator_agent
                        if existing_pending and getattr(existing_pending, "orchestrator_agent", None)
                        else DEFAULT_ORCHESTRATOR_AGENT
                    ),
                    "orchestrator_binding": (
                        existing_pending.orchestrator_binding
                        if existing_pending and getattr(existing_pending, "orchestrator_binding", None)
                        else None
                    ),
                    "review_agents": (
                        existing_pending.review_agents
                        if existing_pending is not None
                        else None
                    ),
                    "auto_reviewer": (
                        existing_pending.auto_reviewer
                        if existing_pending is not None
                        else None
                    ),
                    "agent_pool": (
                        tuple(existing_pending.agent_pool)
                        if existing_pending is not None
                        else ()
                    ),
                    "orchestrator_agent_id": (
                        existing_pending.orchestrator_agent_id
                        if existing_pending is not None
                        else None
                    ),
                    "orchestrator_was_auto": (
                        existing_pending.orchestrator_was_auto
                        if existing_pending is not None
                        else False
                    ),
                    "project_id": (
                        existing_pending.project_id
                        if existing_pending is not None
                        else None
                    ),
                    "recommended_agents": (
                        list(existing_pending.recommended_agents)
                        if existing_pending is not None
                        else []
                    ),
                }

            if previous_owner is not None:
                previous_owner.stop_event.set()
                previous_owner.heartbeat_stop_event.set()
                with previous_owner.delivery_lock:
                    pass

            heartbeat_thread: threading.Thread | None = None
            try:
                try:
                    origin_message_id = self._resolve_origin(message_id)
                except Exception:
                    logger.debug(
                        "Workflow origin resolution failed; using inbound message",
                        exc_info=True,
                    )
                    origin_message_id = message_id
            except Exception:
                logger.debug(
                    "Workflow origin resolution failed; using inbound message",
                    exc_info=True,
                )
                origin_message_id = message_id
            gen_msg_id: str | None = None
            heartbeat_start = _workflow_module().time.time()

            def _owner_can_deliver() -> bool:
                return self._generation_owner_is_active(engine, owner)

            def _heartbeat_update(status_hint: str = "") -> None:
                """Update the generating card only while this owner is live."""
                with owner.delivery_lock:
                    if not _owner_can_deliver():
                        return
                    elapsed = int(_workflow_module().time.time() - heartbeat_start)
                    status = status_hint or "正在生成编排脚本..."
                    progress_card_data = self._workflow_generation_card_data(
                        requirement=requirement,
                        agent_pool=generation_context["agent_pool"],
                        orchestrator_agent_id=(
                            generation_context["orchestrator_agent_id"] or ""
                        ),
                        orchestrator_was_auto=generation_context[
                            "orchestrator_was_auto"
                        ],
                        current_activity=status,
                        elapsed_seconds=elapsed,
                    )
                    if gen_msg_id:
                        self.update_card(
                            gen_msg_id,
                            self._build_workflow_card_from_renderer_data(
                                progress_card_data
                            ),
                        )

            def _heartbeat_loop() -> None:
                while not owner.stop_event.is_set() and not owner.heartbeat_stop_event.is_set():
                    owner.heartbeat_stop_event.wait(8.0)
                    if not owner.stop_event.is_set() and not owner.heartbeat_stop_event.is_set():
                        try:
                            _heartbeat_update()
                        except Exception:
                            pass

            def _stop_heartbeat() -> None:
                owner.heartbeat_stop_event.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=2.0)

            script_path: str
            meta: dict[str, Any] | None = None
            try:
                gen_card_data = self._workflow_generation_card_data(
                    requirement=requirement,
                    agent_pool=generation_context["agent_pool"],
                    orchestrator_agent_id=(
                        generation_context["orchestrator_agent_id"] or ""
                    ),
                    orchestrator_was_auto=generation_context[
                        "orchestrator_was_auto"
                    ],
                )
                with owner.delivery_lock:
                    if not _owner_can_deliver():
                        return
                    gen_msg_id = self._replace_or_send_workflow_rendered_card(
                        card_message_id=message_id,
                        chat_id=chat_id,
                        card_data=gen_card_data,
                        origin_message_id=origin_message_id,
                    ) or message_id
                    object.__setattr__(owner, "card_message_id", gen_msg_id)
                if not _owner_can_deliver():
                    return

                heartbeat_thread = _workflow_module().threading.Thread(
                    target=_heartbeat_loop,
                    name="wf-gen-heartbeat",
                    daemon=True,
                )
                heartbeat_thread.start()

                script_path, meta = self._generate_script_via_ai(
                    requirement,
                    root_path,
                    selected_tools,
                    engine,
                    progress_callback=_heartbeat_update,
                    output_path=owner.source_script_path,
                    cancel_event=owner.stop_event,
                    artifact_lock=owner.delivery_lock,
                )
                _stop_heartbeat()

                script_tools = list(dict.fromkeys((meta or {}).get("tools", [])))
                confirmed_pool = tuple(generation_context["agent_pool"])
                confirmed_tools = list(
                    dict.fromkeys(binding.tool_name for binding in confirmed_pool)
                )
                unsupported = [tool for tool in script_tools if tool not in confirmed_tools]
                if unsupported:
                    raise RuntimeError("脚本引用未确认工具: " + ", ".join(unsupported))
                selected = confirmed_tools
                if not confirmed_pool:
                    raise RuntimeError("Workflow generation lost its confirmed Agent Pool")
                script_hash = None
                if script_path:
                    try:
                        with open(script_path, "rb") as file:
                            import hashlib

                            script_hash = hashlib.sha256(file.read()).hexdigest()
                    except OSError:
                        script_hash = None

                prepared_pending = PendingWorkflow(
                    script_path=script_path,
                    requirement=requirement,
                    meta=meta,
                    initiator_user_id=generation_context["initiator_user_id"],
                    engine_session_key=expected_session_key,
                    selected_tools=selected,
                    orchestrator_agent=generation_context["orchestrator_agent"],
                    orchestrator_binding=generation_context["orchestrator_binding"],
                    review_agents=generation_context["review_agents"] or [],
                    auto_reviewer=True,
                    script_hash=script_hash,
                    agent_pool=confirmed_pool,
                    orchestrator_agent_id=generation_context["orchestrator_agent_id"],
                    orchestrator_was_auto=generation_context["orchestrator_was_auto"],
                    project_id=generation_context["project_id"],
                    recommended_agents=generation_context["recommended_agents"],
                )
                if not self._commit_generated_workflow_if_current(
                    engine,
                    owner,
                    prepared_pending,
                ):
                    logger.info(
                        "[workflow] Dropping stale script generation result (expected_session=%s)",
                        expected_session_key[:8],
                    )
                    return

                self._queue_generated_workflow(
                    message_id=gen_msg_id or message_id,
                    chat_id=chat_id,
                    project=project,
                    root_path=root_path,
                    engine=engine,
                    generation_owner=owner,
                )
                started = bool(
                    engine.project
                    and engine.project.status == WorkflowStatus.RUNNING
                )
                if not started and _owner_can_deliver():
                    raise RuntimeError("Workflow 自动启动失败")
            except _workflow_module()._WorkflowGenerationCancelled as exc:
                safe_error = self._sanitize_generation_error(exc)
                with owner.delivery_lock:
                    with engine._lock:
                        current_pending = engine.project.pending if engine.project else None
                        same_session = bool(
                            current_pending is not None
                            and current_pending.engine_session_key == owner.session_key
                        )
                        owner_is_current = bool(
                            vars(engine).get("_script_generation_owner") is owner
                        )
                        if (
                            owner_is_current
                            and same_session
                            and engine.project.status == WorkflowStatus.GENERATING_SCRIPT
                        ):
                            engine.project.status = WorkflowStatus.CANCELLED
                            engine.project.error = safe_error
                            engine.project.finished_at = _workflow_module().time.time()
                        should_deliver = bool(
                            same_session
                            and engine.project is not None
                            and engine.project.status == WorkflowStatus.CANCELLED
                        )
                    if should_deliver and current_pending is not None:
                        cancelled_card_data = self._workflow_generation_card_data(
                            requirement=current_pending.requirement or requirement,
                            agent_pool=current_pending.agent_pool,
                            orchestrator_agent_id=(
                                current_pending.orchestrator_agent_id or ""
                            ),
                            orchestrator_was_auto=(
                                current_pending.orchestrator_was_auto
                            ),
                            current_activity=safe_error,
                            elapsed_seconds=int(_workflow_module().time.time() - heartbeat_start),
                            terminal_status="cancelled",
                        )
                        target_message_id = (
                            gen_msg_id or owner.card_message_id or message_id
                        )
                        final_message_id = (
                            self._replace_or_send_workflow_rendered_card(
                                card_message_id=target_message_id,
                                chat_id=chat_id,
                                card_data=cancelled_card_data,
                                origin_message_id=origin_message_id,
                            )
                            or target_message_id
                        )
                        object.__setattr__(
                            owner,
                            "card_message_id",
                            final_message_id,
                        )
            except Exception as exc:
                logger.error(
                    "Workflow script generation failed: %s",
                    exc,
                    exc_info=True,
                )
                safe_error = self._sanitize_generation_error(exc)
                with owner.delivery_lock:
                    if _owner_can_deliver():
                        with engine._lock:
                            if not self._is_current_generation_session(
                                engine,
                                owner.session_key,
                            ):
                                return
                            engine.project.status = WorkflowStatus.FAILED
                            engine.project.error = safe_error
                            engine.project.finished_at = _workflow_module().time.time()
                        if gen_msg_id:
                            error_card = self._build_error_card(
                                "internal_error",
                                detail=f"脚本生成失败: {safe_error}",
                            )
                            self._replace_or_send_workflow_card(
                                card_message_id=gen_msg_id,
                                chat_id=chat_id,
                                card=error_card,
                                origin_message_id=origin_message_id,
                            )
                        else:
                            self._reply_workflow_error(
                                message_id,
                                "internal_error",
                                detail=f"脚本生成失败: {safe_error}",
                            )
            finally:
                _stop_heartbeat()
                owner.worker_exited_event.set()
                with engine._lock:
                    session_uncertain = owner.active_generation_session is not None
                    if vars(engine).get("_script_generation_owner") is owner:
                        self._retire_workflow_owner(engine, owner)
                        if not session_uncertain:
                            engine._script_generation_owner = None
                    elif session_uncertain:
                        self._retire_workflow_owner(engine, owner)
                    current_project = engine.project
                    current_pending = current_project.pending if current_project is not None else None
                    retained_paths = {
                        getattr(current_pending, "script_path", None),
                        getattr(current_project, "script_path", None),
                    }
                if owner.source_script_path not in retained_paths:
                    self._remove_owned_workflow_artifact(
                        owner.source_script_path,
                        root_path=root_path,
                    )
                if not session_uncertain:
                    owner.done_event.set()
                    release_owner = getattr(engine, "release_lifecycle_owner", None)
                    if callable(release_owner):
                        release_owner(owner)

    @staticmethod
    def _sanitize_generation_error(exc: BaseException | str) -> str:
        """Return a bounded terminal reason without credentials or internals."""
        import re

        text = " ".join(str(exc or "Workflow generation failed").split())
        text = re.sub(
            r"(?i)\b(token|secret|password|api[_-]?key|access[_-]?key)\s*[=:]\s*[^\s,;]+",
            r"\1=<redacted>",
            text,
        )
        return text[:500] or "Workflow generation failed"

    def _schedule_generate_and_start_workflow(
        self,
        *,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: Optional["ProjectContext"],
        root_path: str,
        selected_tools: list[str] | None,
        expected_session_key: str,
        engine: Any | None = None,
    ) -> Any:
        """Submit generation and reconcile pre-callback scheduler failures."""
        from ...tasking.scheduler import TaskStatus
        from ...workflow_engine.models import WorkflowStatus

        if not expected_session_key or engine is None:
            logger.error("[workflow] Refusing to schedule generation without a session owner")
            return None
        with engine._lock:
            pending = engine.project.pending if engine.project else None
            if (
                engine.project is None
                or engine.project.status != WorkflowStatus.GENERATING_SCRIPT
                or pending is None
                or pending.engine_session_key != expected_session_key
            ):
                logger.info(
                    "[workflow] Refusing stale generation schedule (expected_session=%s)",
                    expected_session_key[:8],
                )
                return None
            lifecycle_owner = vars(engine).get("_script_generation_owner")
        if not isinstance(lifecycle_owner, _workflow_module()._WorkflowLifecycleOwner):
            logger.error("[workflow] Refusing generation schedule without lifecycle owner")
            return None

        def fail_once(exc: BaseException | str) -> bool:
            safe_error = self._sanitize_generation_error(exc)
            with lifecycle_owner.delivery_lock:
                with engine._lock:
                    current_pending = engine.project.pending if engine.project else None
                    should_report = bool(
                        vars(engine).get("_script_generation_owner") is lifecycle_owner
                        and not lifecycle_owner.stop_event.is_set()
                        and engine.project is not None
                        and engine.project.status == WorkflowStatus.GENERATING_SCRIPT
                        and current_pending is not None
                        and current_pending.engine_session_key == expected_session_key
                    )
                    if not should_report:
                        return False
                    engine.project.status = WorkflowStatus.FAILED
                    engine.project.error = safe_error
                    engine.project.finished_at = _workflow_module().time.time()
                    object.__setattr__(lifecycle_owner, "active_generation_session", None)
                    self._retire_workflow_owner(engine, lifecycle_owner)
                    engine._script_generation_owner = None
                error_card = self._build_error_card(
                    "internal_error",
                    detail=f"脚本生成失败: {safe_error}",
                )
                try:
                    origin_message_id = self._resolve_origin(message_id)
                except Exception:
                    origin_message_id = message_id
                self._replace_or_send_workflow_card(
                    card_message_id=message_id,
                    chat_id=chat_id,
                    card=error_card,
                    origin_message_id=origin_message_id,
                )
            lifecycle_owner.done_event.set()
            release_owner = getattr(engine, "release_lifecycle_owner", None)
            if callable(release_owner):
                release_owner(lifecycle_owner)
            return True

        def run_generate() -> None:
            try:
                task_engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)
                if not self._is_current_generation_session(
                    task_engine,
                    expected_session_key,
                ):
                    return
                self._generate_and_start_workflow(
                    message_id=message_id,
                    chat_id=chat_id,
                    requirement=requirement,
                    project=project,
                    root_path=root_path,
                    selected_tools=selected_tools,
                    expected_session_key=expected_session_key,
                )
            except _workflow_module()._WorkflowGenerationCancelled as exc:
                cancelled_pending = None
                should_deliver = False
                with lifecycle_owner.delivery_lock:
                    with engine._lock:
                        current_pending = engine.project.pending if engine.project else None
                        if (
                            vars(engine).get("_script_generation_owner") is lifecycle_owner
                            and engine.project is not None
                            and engine.project.status == WorkflowStatus.GENERATING_SCRIPT
                            and current_pending is not None
                            and current_pending.engine_session_key == expected_session_key
                        ):
                            engine.project.status = WorkflowStatus.CANCELLED
                            engine.project.error = self._sanitize_generation_error(exc)
                            engine.project.finished_at = _workflow_module().time.time()
                            engine._script_generation_owner = None
                            cancelled_pending = current_pending
                            should_deliver = True
                    if should_deliver and cancelled_pending is not None:
                        target_message_id = (
                            lifecycle_owner.card_message_id or message_id
                        )
                        cancelled_card_data = self._workflow_generation_card_data(
                            requirement=cancelled_pending.requirement or requirement,
                            agent_pool=cancelled_pending.agent_pool,
                            orchestrator_agent_id=(
                                cancelled_pending.orchestrator_agent_id or ""
                            ),
                            orchestrator_was_auto=(
                                cancelled_pending.orchestrator_was_auto
                            ),
                            current_activity=self._sanitize_generation_error(exc),
                            terminal_status="cancelled",
                        )
                        final_message_id = (
                            self._replace_or_send_workflow_rendered_card(
                                card_message_id=target_message_id,
                                chat_id=chat_id,
                                card_data=cancelled_card_data,
                                origin_message_id=message_id,
                            )
                            or target_message_id
                        )
                        object.__setattr__(
                            lifecycle_owner,
                            "card_message_id",
                            final_message_id,
                        )
                if should_deliver:
                    lifecycle_owner.done_event.set()
                    release_owner = getattr(engine, "release_lifecycle_owner", None)
                    if callable(release_owner):
                        release_owner(lifecycle_owner)
            except Exception as exc:
                logger.error("Workflow script generation task failed: %s", exc, exc_info=True)
                fail_once(exc)

        project_name = (
            (getattr(project, "project_name", "") if project else "")
            or os.path.basename(root_path)
        )
        task_id = _workflow_module().generate_task_id(project_name or "workflow")
        try:
            handle = self._submit_engine_task(
                run_generate,
                chat_id,
                message_id,
                project,
                request_id=None,
                task_id=task_id,
                name_suffix="generate_script",
            )
        except Exception as exc:
            logger.error("Workflow script generation task submission failed: %s", exc, exc_info=True)
            fail_once(exc)
            return None

        run_id = getattr(handle, "run_id", None)
        scheduler = getattr(self.ctx, "scheduler", None)
        if not run_id or scheduler is None:
            return handle

        def scheduler_state() -> Any:
            try:
                return scheduler.get_state(run_id)
            except Exception:
                logger.debug("Unable to reconcile Workflow generation scheduler state", exc_info=True)
                return None

        state = scheduler_state()
        if state is not None and getattr(state, "status", None) in {
            TaskStatus.FAILED,
            TaskStatus.CANCELED,
        }:
            fail_once(getattr(state, "error", None) or "generation scheduler rejected the task")
            return handle

        if state is not None and getattr(state, "status", None) == TaskStatus.QUEUED:
            def reconcile_queued() -> None:
                deadline = _workflow_module().time.monotonic() + 600.0
                while _workflow_module().time.monotonic() < deadline and not lifecycle_owner.stop_event.wait(1.0):
                    queued_state = scheduler_state()
                    if queued_state is None:
                        continue
                    status = getattr(queued_state, "status", None)
                    if status in {TaskStatus.RUNNING, TaskStatus.SUCCEEDED}:
                        return
                    if status in {TaskStatus.FAILED, TaskStatus.CANCELED}:
                        fail_once(
                            getattr(queued_state, "error", None)
                            or "generation scheduler rejected the queued task"
                        )
                        return
                if not lifecycle_owner.stop_event.is_set():
                    try:
                        handle.cancel()
                    except Exception:
                        logger.debug("Unable to cancel unreconciled generation task", exc_info=True)
                    fail_once("generation scheduler queue reconciliation timed out")

            _workflow_module().threading.Thread(
                target=reconcile_queued,
                daemon=True,
                name=f"workflow-generation-reconcile-{expected_session_key[:8]}",
            ).start()
        return handle

    # ------------------------------------------------------------------
    # Stop workflow
    # ------------------------------------------------------------------



    @staticmethod
    def _retire_workflow_owner(engine: Any, owner: Any) -> None:
        """Keep a revoked owner discoverable by later cleanup retries."""
        retire = getattr(engine, "retire_lifecycle_owner", None)
        if callable(retire):
            retire(owner)

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
        *,
        lifecycle_owner: _WorkflowLifecycleOwner,
        lifecycle_engine: Any,
    ):
        """Fence Workflow repo-lock/error cards with the queued-start owner."""
        import asyncio

        from ...repo_lock import LockConflictError
        from ...utils.errors import get_error_detail

        root_path = getattr(project, "root_path", None) if project else None

        def owner_is_current() -> bool:
            with lifecycle_engine._lock:
                return bool(
                    vars(lifecycle_engine).get("_workflow_start_owner") is lifecycle_owner
                    and getattr(lifecycle_engine, "_closing", False) is not True
                    and not lifecycle_owner.stop_event.is_set()
                )

        def deliver_if_current(delivery: callable) -> None:
            with lifecycle_owner.delivery_lock:
                if owner_is_current():
                    delivery()

        def body() -> None:
            try:
                executor_func()
            except NotImplementedError:
                logger.error(
                    "%s Engine: NotImplementedError in executor_func (task_id=%s)",
                    self._get_engine_name_prefix(),
                    task_id,
                )
                deliver_if_current(
                    lambda: self.reply_text(
                        message_id,
                        "系统升级中，请重试",
                    )
                )
            except LockConflictError:
                raise
            except Exception as exc:
                if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                    logger.warning(
                        "%s Engine 执行超时 (task_id=%s): %s",
                        self._get_engine_name_prefix(),
                        task_id,
                        get_error_detail(exc),
                    )
                else:
                    logger.error(
                        "%s Engine 执行异常: %s",
                        self._get_engine_name_prefix(),
                        get_error_detail(exc),
                        exc_info=True,
                    )
                deliver_if_current(
                    lambda caught=exc: self._on_engine_error(
                        error=caught,
                        task_id=task_id,
                        chat_id=chat_id,
                        message_id=message_id,
                        project=project,
                        engine_name=engine_name,
                        reporter=reporter,
                        request_id=request_id,
                        action_prefix=action_prefix,
                    )
                )

        # Request-id resolution and scheduler latency happen before this
        # method. Recheck ownership before touching the repository lock so a
        # cleanup/supersede winner retires the stale queued closure quietly.
        if not owner_is_current():
            return None

        try:
            return self.lock_helper._with_repo_lock(
                root_path,
                chat_id,
                body,
            )
        except LockConflictError as exc:
            deliver_if_current(
                lambda caught=exc: self.lock_helper.send_lock_conflict_card(
                    caught,
                    message_id,
                    command_text,
                    chat_id=chat_id,
                )
            )
            return None









    def stop_workflow(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        *,
        terminal_is_noop: bool = False,
    ) -> None:
        """Stop the running workflow for the current project.

        Security: only the workflow initiator or a configured admin can stop it.
        """
        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        from ...workflow_engine.models import WorkflowStatus

        if not engine:
            self._reply_workflow_error(message_id, "invalid_state", detail="当前没有运行中的 Workflow 任务")
            return

        from ...thread import get_current_sender_id

        current_user = get_current_sender_id()
        admin_ids: list[str] = getattr(self.ctx.settings, "admin_user_ids", []) or []
        error: tuple[str, str] | None = None
        delivery_owners: list[_WorkflowLifecycleOwner] = []
        cancel_card_message_ids: list[str] = []
        sessions_to_close: list[tuple[_WorkflowLifecycleOwner, Any]] = []
        artifacts_to_remove: list[str] = []
        runtime_active = False
        runtime_stop_owner: _WorkflowLifecycleOwner | None = None
        terminal_notice: WorkflowStatus | None = None
        with engine._lock:
            wf_project = engine.project
            pending = getattr(wf_project, "pending", None) if wf_project else None
            status = getattr(wf_project, "status", WorkflowStatus.IDLE)
            pending_lifecycle = status in _workflow_module()._workflow_pending_statuses()
            stored_initiator = (
                getattr(pending, "initiator_user_id", None) if pending_lifecycle and pending else None
            ) or getattr(wf_project, "initiator_user_id", None)
            generation_owner = vars(engine).get("_script_generation_owner")
            start_owner = vars(engine).get("_workflow_start_owner")
            live_generation_owner = (
                generation_owner
                if isinstance(generation_owner, _workflow_module()._WorkflowLifecycleOwner) and not generation_owner.stop_event.is_set()
                else None
            )
            live_start_owner = (
                start_owner
                if isinstance(start_owner, _workflow_module()._WorkflowLifecycleOwner) and not start_owner.stop_event.is_set()
                else None
            )
            stored_initiator = (
                stored_initiator
                or getattr(
                    live_generation_owner,
                    "initiator_user_id",
                    None,
                )
                or getattr(
                    live_start_owner,
                    "initiator_user_id",
                    None,
                )
            )
            terminal_committed = status in {
                WorkflowStatus.COMPLETED,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED,
            }
            if terminal_committed:
                live_start_owner = None
            runtime_active = bool(engine.is_running) and not terminal_committed
            lifecycle_active = bool(
                runtime_active
                or live_generation_owner is not None
                or live_start_owner is not None
                or status in _workflow_module()._workflow_pending_statuses()
            )

            if not lifecycle_active:
                if terminal_is_noop and terminal_committed:
                    terminal_notice = status
                else:
                    error = ("invalid_state", "当前没有运行中的 Workflow 任务")
            elif not stored_initiator or not current_user:
                error = ("forbidden", "无法验证操作者身份，停止请求被拒绝")
            elif current_user != stored_initiator and current_user not in admin_ids:
                error = ("forbidden", "只有 Workflow 发起者或管理员才能停止此任务")
            else:
                for lifecycle_owner in (
                    live_generation_owner,
                    live_start_owner,
                ):
                    if lifecycle_owner is None:
                        continue
                    lifecycle_owner.stop_event.set()
                    lifecycle_owner.heartbeat_stop_event.set()
                    self._retire_workflow_owner(
                        engine,
                        lifecycle_owner,
                    )
                    if lifecycle_owner not in delivery_owners:
                        delivery_owners.append(lifecycle_owner)
                    if (
                        lifecycle_owner.card_message_id
                        and lifecycle_owner.card_message_id
                        not in cancel_card_message_ids
                    ):
                        cancel_card_message_ids.append(
                            lifecycle_owner.card_message_id
                        )
                    active_session = lifecycle_owner.active_generation_session
                    if active_session is not None and not any(
                        existing_session is active_session
                        for _existing_owner, existing_session in sessions_to_close
                    ):
                        sessions_to_close.append((lifecycle_owner, active_session))
                    for artifact in (
                        lifecycle_owner.source_script_path,
                        lifecycle_owner.execution_script_path,
                    ):
                        if artifact:
                            artifacts_to_remove.append(artifact)
                if (
                    live_generation_owner is not None
                    and live_generation_owner.active_generation_session is None
                ):
                    engine._script_generation_owner = None
                if live_start_owner is not None and not runtime_active:
                    engine._workflow_start_owner = None
                if pending and pending.script_path:
                    artifacts_to_remove.append(pending.script_path)
                if live_start_owner is not None and wf_project is not None and wf_project.script_path:
                    artifacts_to_remove.append(wf_project.script_path)

                if runtime_active:
                    runtime_stop_owner = live_start_owner
                    engine.stop()
                elif wf_project is not None:
                    wf_project.status = WorkflowStatus.CANCELLED
                    wf_project.error = "Workflow cancelled by user"
                    wf_project.finished_at = _workflow_module().time.time()
                    wf_project.script_path = None

        if terminal_notice is not None:
            terminal_label = {
                WorkflowStatus.COMPLETED: "已完成",
                WorkflowStatus.FAILED: "失败",
                WorkflowStatus.CANCELLED: "已取消",
            }.get(terminal_notice, "已结束")
            self.reply_text(
                message_id,
                f"Workflow 任务已结束，当前状态：{terminal_label}。",
            )
            return

        if error is not None:
            self._reply_workflow_error(
                message_id,
                error[0],
                detail=error[1],
            )
            return

        uncertain_sessions: list[tuple[_WorkflowLifecycleOwner, Any, str]] = []
        for session_owner, session in sessions_to_close:
            cancel_confirmed = True
            close_confirmed = True
            details: list[str] = []
            try:
                cancelled = session.cancel(wait=True, timeout=2.0)
                if cancelled is False:
                    cancel_confirmed = False
                    details.append("cancel was not confirmed")
            except Exception as exc:
                cancel_confirmed = False
                details.append(f"cancel failed: {exc}")
                logger.warning("Failed to cancel active Workflow generation session", exc_info=True)
            try:
                session.close()
            except Exception as exc:
                close_confirmed = False
                details.append(f"close failed: {exc}")
                logger.warning("Failed to close active Workflow generation session", exc_info=True)
            if cancel_confirmed and close_confirmed:
                object.__setattr__(session_owner, "active_generation_session", None)
                with engine._lock:
                    if vars(engine).get("_script_generation_owner") is session_owner:
                        engine._script_generation_owner = None
            else:
                uncertain_sessions.append(
                    (session_owner, session, "; ".join(details) or "stop was not confirmed")
                )

        if uncertain_sessions:
            safe_error = "Workflow 停止未确认；为避免重叠会话，当前项目保持封锁。"
            with engine._lock:
                if engine.project is not None:
                    engine.project.status = WorkflowStatus.FAILED
                    engine.project.error = safe_error
                    engine.project.finished_at = _workflow_module().time.time()
                for session_owner, _session, _detail in uncertain_sessions:
                    engine._script_generation_owner = session_owner
                    self._retire_workflow_owner(engine, session_owner)
            self._reply_workflow_error(
                message_id,
                "internal_error",
                detail=safe_error,
            )
            return

        if (
            runtime_stop_owner is not None
            and runtime_stop_owner.worker_thread_id != _workflow_module().threading.get_ident()
            and not runtime_stop_owner.done_event.wait(
                timeout=_workflow_module()._WORKFLOW_STOP_QUIESCENCE_TIMEOUT_S
            )
        ):
            # No callback may cross the user-visible stop response.  If the
            # execution owner cannot prove quiescence in time, fence its
            # cancellation card and report an explicit unconfirmed stop.
            with runtime_stop_owner.delivery_lock:
                runtime_stop_owner.stop_delivery_fenced_event.set()
            self._reply_workflow_error(
                message_id,
                "internal_error",
                detail=(
                    "Workflow 停止未在限定时间内确认；任务已收到取消信号，"
                    "当前项目保持停止中，请稍后用 /wf_status 检查。"
                ),
            )
            return

        # A delivery that had already passed its ownership check may finish,
        # but it must do so before the stop acknowledgement is sent.
        for lifecycle_owner in delivery_owners:
            with lifecycle_owner.delivery_lock:
                lifecycle_owner.stop_delivery_fenced_event.set()
            if (
                not runtime_active
                and not lifecycle_owner.claimed_event.is_set()
                and not lifecycle_owner.worker_started_event.is_set()
            ):
                lifecycle_owner.done_event.set()
            release_owner = getattr(engine, "release_lifecycle_owner", None)
            if callable(release_owner):
                release_owner(lifecycle_owner)

        if not runtime_active and pending is not None and cancel_card_message_ids:
            cancelled_card_data = self._workflow_generation_card_data(
                requirement=pending.requirement or "",
                agent_pool=pending.agent_pool,
                orchestrator_agent_id=pending.orchestrator_agent_id or "",
                orchestrator_was_auto=pending.orchestrator_was_auto,
                current_activity="Workflow 任务已由用户停止",
                terminal_status="cancelled",
            )
            for card_message_id in cancel_card_message_ids:
                self._replace_or_send_workflow_rendered_card(
                    card_message_id=card_message_id,
                    chat_id=chat_id,
                    card_data=cancelled_card_data,
                    origin_message_id=message_id,
                    fallback_to_new=False,
                )

        for artifact in dict.fromkeys(artifacts_to_remove):
            self._remove_owned_workflow_artifact(
                artifact,
                root_path=root_path,
            )

        self.reply_text(message_id, "Workflow 任务已停止。")

    def handle_workflow_stop_running(
        self,
        message_id: str,
        chat_id: str,
        project_id: str,
        value: dict[str, Any],
    ) -> None:
        """Handle the "停止" button on a RUNNING workflow progress card.

        Delegates to :meth:`stop_workflow`, which owns the initiator/admin
        authorization checks and state validation. This keeps the auth logic
        in a single place and maximizes reuse.
        """
        from ...card.events.payloads import filter_workflow_button_value

        value = filter_workflow_button_value(value)
        project_id = project_id or value.get("project_id", "")
        project = self._resolve_project_from_id(project_id, chat_id)
        # Resolve root_path defensively so a missing project still lets the
        # underlying stop_workflow re-derive engine state from the chat.
        self._get_root_path(chat_id, project)
        self.stop_workflow(
            message_id,
            chat_id,
            project,
            terminal_is_noop=True,
        )

















    def _queue_generated_workflow(
            self,
            *,
            message_id: str,
            chat_id: str,
            project: Optional["ProjectContext"],
            root_path: str,
            engine: Any,
            generation_owner: _WorkflowLifecycleOwner,
        ) -> None:
            """Validate the owned generated script and hand it directly to runtime."""
            if not engine or not engine.project:
                return

            from ...workflow_engine.models import WorkflowStatus

            def _reply_start_error(
                category: str,
                *,
                detail: str = "",
            ) -> None:
                with generation_owner.delivery_lock:
                    with engine._lock:
                        owner_is_current = bool(
                            vars(engine).get("_script_generation_owner") is generation_owner
                            and not generation_owner.stop_event.is_set()
                        )
                    if not owner_is_current:
                        return
                    safe_error = self._sanitize_generation_error(
                        detail or f"Workflow 自动启动失败: {category}"
                    )
                    with engine._lock:
                        if (
                            vars(engine).get("_script_generation_owner")
                            is not generation_owner
                            or generation_owner.stop_event.is_set()
                        ):
                            return
                        engine.project.status = WorkflowStatus.FAILED
                        engine.project.error = safe_error
                        engine.project.finished_at = _workflow_module().time.time()
                    error_card = self._build_error_card(
                        category,
                        detail=safe_error,
                    )
                    try:
                        origin_message_id = self._resolve_origin(message_id)
                    except Exception:
                        origin_message_id = message_id
                    self._replace_or_send_workflow_card(
                        card_message_id=(
                            generation_owner.card_message_id or message_id
                        ),
                        chat_id=chat_id,
                        card=error_card,
                        origin_message_id=origin_message_id,
                    )

            with engine._lock:
                current_project = engine.project
                pending = current_project.pending if current_project else None
                if current_project is None or current_project.status != WorkflowStatus.GENERATING_SCRIPT:
                    pending = None
                if (
                    vars(engine).get("_script_generation_owner") is not generation_owner
                    or generation_owner.stop_event.is_set()
                ):
                    return

            if pending is None:
                _reply_start_error("invalid_state")
                return

            stored_initiator = pending.initiator_user_id or ""
            stored_session_key = pending.engine_session_key or ""

            if not stored_session_key or generation_owner.session_key != stored_session_key:
                logger.warning(
                    "Workflow auto-start rejected: session_key mismatch (expected=%s, stored=%s)",
                    generation_owner.session_key[:8],
                    stored_session_key[:8],
                )
                _reply_start_error("session_expired")
                return

            if not stored_initiator:
                logger.warning(
                    "Workflow auto-start rejected: initiator is missing",
                )
                _reply_start_error("forbidden")
                return

            # Retrieve pending state
            script_path = pending.script_path
            requirement = pending.requirement or ""
            expected_script_hash = pending.script_hash

            if not script_path:
                _reply_start_error(
                    "invalid_state",
                    detail="无法获取待执行脚本，请重新发送 `/wf`",
                )
                return

            # --- TOCTOU hardening (check-then-re-read-verify) ---
            # Re-read script content fresh from disk at execution handoff time and
            # re-run the full validation chain. This defends against scripts
            # being tampered with after generation but before runtime handoff.
            try:
                with open(script_path, "rb") as f:
                    script_bytes = f.read()
            except OSError:
                # script_path is an internal implementation detail.
                _reply_start_error(
                    "internal_error",
                    detail="脚本文件读取失败，请重新发送 `/wf` 生成",
                )
                return
            script_text = script_bytes.decode("utf-8", errors="strict")
            import hashlib

            current_script_hash = hashlib.sha256(script_bytes).hexdigest()
            if expected_script_hash and current_script_hash != expected_script_hash:
                _reply_start_error(
                    "internal_error",
                    detail="脚本内容与生成时不一致，疑似被篡改。请重新发送 `/wf` 生成。",
                )
                return

            # Structural and frozen-pool validation mirrors generation.
            from ...workflow_engine.script_gen import extract_meta_from_script, validate_generated_script

            is_valid, validation_errors = validate_generated_script(
                script_text,
                agent_pool=pending.agent_pool,
            )
            if not is_valid:
                _reply_start_error(
                    "internal_error",
                    detail="脚本验证失败：" + "; ".join(validation_errors[:3]),
                )
                return
            fresh_meta = extract_meta_from_script(script_text) or {}

            fresh_script_tools = list(dict.fromkeys(fresh_meta.get("tools", [])))
            confirmed_tools = list(
                dict.fromkeys(binding.tool_name for binding in pending.agent_pool)
            )
            unsupported = [tool for tool in fresh_script_tools if tool not in confirmed_tools]
            if unsupported:
                _reply_start_error(
                    "invalid_state",
                    detail="脚本引用未确认工具: " + ", ".join(unsupported),
                )
                return
            pending.selected_tools = confirmed_tools

            # --- Immutable copy for execution ---
            # Copy the verified content into a fresh /tmp file for each
            # verified execution. Using mkstemp guarantees uniqueness across
            # concurrent sessions and test suites; we additionally append the
            # script hash so accidental inspection can trace the source.
            import tempfile

            temp_fd, immutable_script_path = tempfile.mkstemp(
                prefix="ghostap-verified-",
                suffix=f"-{current_script_hash}.js",
            )

            def _discard_immutable_script() -> None:
                self._remove_owned_workflow_artifact(
                    immutable_script_path,
                    root_path=root_path,
                )

            try:
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    f.write(script_text)
            except OSError as exc:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
                _discard_immutable_script()
                _reply_start_error(
                    "internal_error",
                    detail=f"无法创建执行用临时脚本副本: {exc}",
                )
                return

            start_owner = _workflow_module()._WorkflowLifecycleOwner(
                session_key=stored_session_key,
                initiator_user_id=stored_initiator,
                chat_id=chat_id,
                project_id=pending.project_id or "",
                root_path=root_path,
                card_message_id=message_id,
                stop_event=generation_owner.stop_event,
                heartbeat_stop_event=generation_owner.heartbeat_stop_event,
                delivery_lock=generation_owner.delivery_lock,
                source_script_path=script_path,
                execution_script_path=immutable_script_path,
            )

            origin_message_id = self._resolve_origin(message_id)
            run_spec = None
            binding_error: TypeError | ValueError | None = None

            def _abandon_start(*, remove_source: bool = True) -> None:
                start_owner.stop_event.set()
                _discard_immutable_script()
                if remove_source:
                    self._remove_owned_workflow_artifact(
                        start_owner.source_script_path,
                        root_path=root_path,
                    )
                with engine._lock:
                    if vars(engine).get("_workflow_start_owner") is start_owner:
                        self._retire_workflow_owner(
                            engine,
                            start_owner,
                        )
                        engine._workflow_start_owner = None
                        if engine.project is not None:
                            engine.project.status = WorkflowStatus.IDLE
                            engine.project.script_path = None
                start_owner.done_event.set()
                release_owner = getattr(engine, "release_lifecycle_owner", None)
                if callable(release_owner):
                    release_owner(start_owner)

            # Linearize the generated-session handoff against /stop_wf. All
            # expensive validation above is speculative; this is the authoritative
            # state/session check immediately before publishing RUNNING.
            with start_owner.delivery_lock:
                with engine._lock:
                    current_project = engine.project
                    current_pending = current_project.pending if current_project is not None else None
                    current_session_key = current_pending.engine_session_key if current_pending else ""
                    generation_owner_matches = vars(engine).get("_script_generation_owner") is generation_owner
                    if (
                        current_project is None
                        or current_project.status != WorkflowStatus.GENERATING_SCRIPT
                        or current_session_key != stored_session_key
                        or not generation_owner_matches
                        or start_owner.stop_event.is_set()
                    ):
                        _discard_immutable_script()
                        return

                    try:
                        run_spec = self._build_run_spec(
                            pending=current_pending,
                            engine=engine,
                            task=current_pending.requirement or requirement,
                            chat_id=chat_id,
                            topic_id=origin_message_id,
                        )
                    except (TypeError, ValueError) as exc:
                        binding_error = exc
                    else:
                        requirement = run_spec.task
                        current_project.start_execution()
                        current_project.status = WorkflowStatus.RUNNING
                        current_project.requirement = requirement
                        current_project.script_path = script_path
                        current_project.started_at = _workflow_module().time.time()
                        current_project.selected_tools = list(run_spec.allowed_tools)
                        current_project.tool_model_map = dict(run_spec.tool_model_map)
                        current_project.run_spec = run_spec.to_dict()
                        engine._workflow_start_owner = start_owner
                        self._retire_workflow_owner(
                            engine,
                            generation_owner,
                        )
                        engine._script_generation_owner = None

                if binding_error is not None:
                    _discard_immutable_script()
                    _reply_start_error(
                        "invalid_state",
                        detail=f"Workflow 执行绑定无效: {binding_error}",
                    )
                    return
                if run_spec is None:
                    _discard_immutable_script()
                    return

                if start_owner.stop_event.is_set():
                    _abandon_start()
                    return

                progress_card_message_id = self._show_initial_workflow_progress_card(
                    card_message_id=message_id,
                    chat_id=chat_id,
                    wf_project=engine.project,
                    origin_message_id=origin_message_id,
                )
                if start_owner.stop_event.is_set():
                    _abandon_start()
                    return

                try:
                    # Use project already resolved above for engine_name.
                    project_id = getattr(project, "project_id", "") or ""
                    engine_name = self.get_engine_name(
                        chat_id,
                        project_id=project_id or None,
                    )
                    project_name = (project.project_name if project else "") or os.path.basename(root_path)
                    task_id = _workflow_module().generate_task_id(project_name or "workflow")
                except Exception as exc:
                    error_card = self._build_error_card(
                        "internal_error",
                        detail=f"Workflow 任务准备失败: {exc}",
                    )
                    try:
                        self._replace_or_send_workflow_card(
                            card_message_id=progress_card_message_id,
                            chat_id=chat_id,
                            card=error_card,
                            origin_message_id=origin_message_id,
                        )
                    finally:
                        _abandon_start()
                    return

                def run_workflow():
                    # Register the scheduler worker in the same engine critical
                    # section that cleanup uses to revoke the owner. Cleanup can
                    # now distinguish "never started" from "started, pre-claim"
                    # without forging completion for the latter.
                    with engine._lock:
                        worker_admitted = bool(
                            vars(engine).get("_workflow_start_owner") is start_owner
                            and getattr(engine, "_closing", False) is not True
                            and not start_owner.stop_event.is_set()
                        )
                        if worker_admitted:
                            start_owner.worker_started_event.set()
                            object.__setattr__(
                                start_owner,
                                "worker_thread_id",
                                _workflow_module().threading.get_ident(),
                            )
                    if not worker_admitted:
                        _abandon_start()
                        return

                    def _executor():
                        callbacks = self._build_workflow_callbacks(
                            progress_card_message_id,
                            chat_id,
                            project,
                            lifecycle_owner=start_owner,
                            requirement=run_spec.task,
                        )
                        engine.execute_workflow(
                            script_path=immutable_script_path,
                            callbacks=callbacks,
                            run_spec=run_spec,
                            start_owner=start_owner,
                            source_script_path=script_path,
                        )

                    try:
                        self._safe_execute_engine(
                            executor_func=_executor,
                            task_id=task_id,
                            chat_id=chat_id,
                            message_id=message_id,
                            project=project,
                            engine_name=engine_name,
                            reporter=self.ctx.progress_reporter,
                            request_id=self.ensure_request_id(
                                message_id,
                                chat_id=chat_id,
                                project_id=project_id or None,
                            ),
                            action_prefix="workflow",
                            command_text=f"/wf {requirement}",
                            lifecycle_owner=start_owner,
                            lifecycle_engine=engine,
                        )
                    finally:
                        if not start_owner.claimed_event.is_set():
                            _abandon_start()

                try:
                    self._submit_engine_task(
                        run_workflow,
                        chat_id,
                        message_id,
                        project,
                        request_id=None,
                        task_id=task_id,
                    )
                except Exception as exc:
                    with start_owner.delivery_lock:
                        with engine._lock:
                            should_report = bool(
                                vars(engine).get("_workflow_start_owner") is start_owner
                                and not start_owner.stop_event.is_set()
                            )
                        try:
                            if should_report:
                                error_card = self._build_error_card(
                                    "internal_error",
                                    detail=f"Workflow 任务提交失败: {exc}",
                                )
                                self._replace_or_send_workflow_card(
                                    card_message_id=progress_card_message_id,
                                    chat_id=chat_id,
                                    card=error_card,
                                    origin_message_id=origin_message_id,
                                )
                        finally:
                            _abandon_start()


    @staticmethod
    def _new_workflow_script_path(root_path: str) -> str:
        """Allocate a generation-owned script path without sharing file state."""
        script_dir = _workflow_module().workflow_scripts_dir(root_path)
        os.makedirs(script_dir, exist_ok=True)
        return os.path.join(
            script_dir,
            f"generated-workflow-{uuid.uuid4().hex}.js",
        )

    @staticmethod
    def _remove_owned_workflow_artifact(
        path: str | None,
        *,
        root_path: str,
    ) -> None:
        """Remove only files created as disposable Workflow artifacts."""
        from ...workflow_engine.storage import remove_owned_workflow_artifact

        remove_owned_workflow_artifact(
            path,
            root_path=root_path,
            scripts_dir=_workflow_module().workflow_scripts_dir(root_path),
            warn_on_refusal=True,
        )

    def _supersede_incomplete_workflow(
        self,
        engine: Any,
        *,
        root_path: str,
        current_user: str,
    ) -> tuple[bool, str | None, _WorkflowLifecycleOwner | None]:
        """Atomically cancel the old lifecycle and admit the newest request.

        A real runtime is never replaced in-place because ``WorkflowEngine``
        resources are shared until its execution thread fully unwinds.
        """
        from ...workflow_engine.models import WorkflowProject, WorkflowStatus

        has_uncertain_session = getattr(engine, "has_uncertain_lifecycle_session", None)
        if callable(has_uncertain_session) and has_uncertain_session():
            return False, "invalid_state", None

        admission_owner = _workflow_module()._WorkflowLifecycleOwner(
            session_key=uuid.uuid4().hex,
            initiator_user_id=current_user,
        )
        owners: list[_WorkflowLifecycleOwner] = []
        artifacts: list[str] = []
        with engine._lock:
            if callable(has_uncertain_session) and has_uncertain_session():
                return False, "invalid_state", None
            if getattr(engine, "_closing", False) is True:
                return False, "running", None
            if engine.is_running is True:
                return False, "running", None

            wf_project = engine.project
            if wf_project is None:
                wf_project = WorkflowProject()
                engine._project = wf_project
            if wf_project.status == WorkflowStatus.GENERATING_SCRIPT:
                return False, "invalid_state", None
            pending = wf_project.pending
            generation_owner = vars(engine).get("_script_generation_owner")
            start_owner = vars(engine).get("_workflow_start_owner")
            incomplete = bool(
                wf_project.status in _workflow_module()._workflow_pending_statuses()
                or isinstance(generation_owner, _workflow_module()._WorkflowLifecycleOwner)
                or isinstance(start_owner, _workflow_module()._WorkflowLifecycleOwner)
                or wf_project.status == WorkflowStatus.RUNNING
            )
            if incomplete:
                stored_initiator = (
                    (getattr(pending, "initiator_user_id", None) if pending is not None else None)
                    or getattr(wf_project, "initiator_user_id", None)
                )
                admin_ids = set(getattr(self.ctx.settings, "admin_user_ids", []) or [])
                if not current_user or not stored_initiator:
                    return False, "forbidden", None
                if current_user != stored_initiator and current_user not in admin_ids:
                    return False, "forbidden", None

                for owner in (
                    generation_owner,
                    start_owner,
                ):
                    if not isinstance(owner, _workflow_module()._WorkflowLifecycleOwner):
                        continue
                    owner.stop_event.set()
                    owner.heartbeat_stop_event.set()
                    self._retire_workflow_owner(engine, owner)
                    owners.append(owner)
                    for artifact in (
                        owner.source_script_path,
                        owner.execution_script_path,
                    ):
                        if artifact:
                            artifacts.append(artifact)

                if pending and pending.script_path:
                    artifacts.append(pending.script_path)
                if isinstance(start_owner, _workflow_module()._WorkflowLifecycleOwner) and wf_project.script_path:
                    artifacts.append(wf_project.script_path)

                engine._script_generation_owner = None
                engine._workflow_start_owner = None
                engine._project = WorkflowProject()
            else:
                # Starting a new lifecycle retires the previous terminal
                # source. It is retired at this exact lifecycle boundary.
                if wf_project.script_path:
                    artifacts.append(wf_project.script_path)
                engine._project = WorkflowProject()

        # An already-authorized card delivery may finish, but always before
        # the new request publishes its first card.
        for owner in owners:
            with owner.delivery_lock:
                pass
            if not owner.claimed_event.is_set() and not owner.worker_started_event.is_set():
                owner.done_event.set()
            release_owner = getattr(engine, "release_lifecycle_owner", None)
            if callable(release_owner):
                release_owner(owner)
        for artifact in dict.fromkeys(artifacts):
            self._remove_owned_workflow_artifact(
                artifact,
                root_path=root_path,
            )
        return True, None, admission_owner

    def _generate_script_via_ai(
        self,
        requirement: str,
        root_path: str,
        selected_tools: list[str] | None = None,
        engine: Any = None,
        progress_callback: Any = None,
        *,
        output_path: str,
        cancel_event: threading.Event | None = None,
        artifact_lock: Any = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate through the confirmed pool with strict attempt fencing."""
        del selected_tools
        from ...agent_session import create_engine_session
        from ...workflow_engine.constants import SCRIPT_GEN_TIMEOUT_S
        from ...workflow_engine.models import WorkflowStatus
        from ...workflow_engine.script_gen import (
            build_script_gen_prompt,
            extract_meta_from_script,
            generate_simple_script,
            validate_generated_script,
        )
        from ...workflow_engine.tool_registry import get_available_tools

        if engine is None or engine.project is None or engine.project.pending is None:
            raise RuntimeError("Workflow generation session is missing")
        pending = engine.project.pending
        pool = tuple(pending.agent_pool or ())
        if not pool:
            raise RuntimeError("Workflow generation requires a confirmed Agent Pool")
        orchestrator_agent_id = pending.orchestrator_agent_id or pool[0].agent_id
        orchestrator = next(
            (item for item in pool if item.agent_id == orchestrator_agent_id),
            None,
        )
        if orchestrator is None:
            raise RuntimeError("Workflow orchestrator is outside the confirmed Agent Pool")

        registered_tools = get_available_tools(require_available=True)
        unavailable = sorted({item.tool_name for item in pool} - set(registered_tools))
        if unavailable:
            raise RuntimeError("Confirmed Workflow tool is unavailable: " + ", ".join(unavailable))
        prompt_tools = {
            item.tool_name: str(registered_tools[item.tool_name])
            for item in pool
        }
        base_prompt = build_script_gen_prompt(
            requirement=requirement,
            available_tools=prompt_tools,
            orchestrator_agent=orchestrator.tool_name,
            orchestrator_binding=orchestrator.to_dict(),
            review_agents=[],
            auto_reviewer=True,
            agent_pool=pool,
            orchestrator_agent_id=orchestrator_agent_id,
        )

        script_dir = _workflow_module().workflow_scripts_dir(root_path)
        os.makedirs(script_dir, exist_ok=True)
        if (
            os.path.realpath(os.path.dirname(output_path)) != os.path.realpath(script_dir)
            or not os.path.basename(output_path).endswith(".js")
        ):
            raise ValueError("Workflow generation output must stay in the script staging directory")

        captured_session_key = pending.engine_session_key
        started_at = _workflow_module().time.monotonic()
        hard_deadline = started_at + float(SCRIPT_GEN_TIMEOUT_S)
        configured_attempt_timeout = float(
            getattr(
                self.settings,
                "workflow_script_gen_timeout_s",
                SCRIPT_GEN_TIMEOUT_S,
            )
            or SCRIPT_GEN_TIMEOUT_S
        )
        if configured_attempt_timeout <= 0:
            configured_attempt_timeout = float(SCRIPT_GEN_TIMEOUT_S)
        ordered_pool = (
            orchestrator,
            *(item for item in pool if item.agent_id != orchestrator.agent_id),
        )
        def ensure_current() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise _workflow_module()._WorkflowGenerationCancelled("Workflow generation cancelled")
            with engine._lock:
                current_project = engine.project
                current_pending = current_project.pending if current_project else None
                if (
                    current_project is None
                    or current_project.status != WorkflowStatus.GENERATING_SCRIPT
                    or current_pending is not pending
                    or current_pending.engine_session_key != captured_session_key
                ):
                    raise RuntimeError("stale Workflow generation session was superseded")

        def _owner_is_current_locked(owner: _WorkflowLifecycleOwner) -> bool:
            current_project = engine.project
            current_pending = current_project.pending if current_project else None
            return bool(
                vars(engine).get("_script_generation_owner") is owner
                and not owner.stop_event.is_set()
                and (cancel_event is None or not cancel_event.is_set())
                and owner.session_key == captured_session_key
                and current_project is not None
                and current_project.status == WorkflowStatus.GENERATING_SCRIPT
                and current_pending is pending
                and current_pending.engine_session_key == captured_session_key
            )

        def current_generation_owner() -> _WorkflowLifecycleOwner | None:
            with engine._lock:
                owner = vars(engine).get("_script_generation_owner")
                if (
                    isinstance(owner, _workflow_module()._WorkflowLifecycleOwner)
                    and _owner_is_current_locked(owner)
                ):
                    return owner
                return None

        def bind_active_session(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> bool:
            with engine._lock:
                if not _owner_is_current_locked(owner):
                    return False
                active_session = owner.active_generation_session
                if active_session is not None and active_session is not session:
                    return False
                object.__setattr__(owner, "active_generation_session", session)
                return True

        def active_session_is_current(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> bool:
            with engine._lock:
                return bool(
                    _owner_is_current_locked(owner)
                    and owner.active_generation_session is session
                )

        def anchor_session_for_cleanup(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> None:
            with engine._lock:
                self._retire_workflow_owner(engine, owner)
                active_session = owner.active_generation_session
                if active_session is None or active_session is session:
                    object.__setattr__(owner, "active_generation_session", session)

        def unbind_active_session(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> None:
            with engine._lock:
                if owner.active_generation_session is session:
                    object.__setattr__(owner, "active_generation_session", None)

        session_owner: _WorkflowLifecycleOwner | None = None

        def close_strict(
            session: Any,
            *,
            cancel: bool,
            quarantine_event: threading.Event,
            provider_cancel_event: threading.Event,
        ) -> None:
            quarantine_event.set()
            cancel_confirmed = True
            close_confirmed = True
            details: list[str] = []
            if cancel:
                provider_cancel_event.set()
                try:
                    cancelled = session.cancel(wait=True, timeout=2.0)
                    if cancelled is False:
                        cancel_confirmed = False
                        details.append("generation session cancellation was not confirmed")
                except Exception as exc:
                    cancel_confirmed = False
                    details.append(f"generation session cancellation failed: {exc}")
            try:
                session.close()
            except Exception as exc:
                close_confirmed = False
                details.append(f"generation session close uncertain: {exc}")
            if not (cancel_confirmed and close_confirmed):
                logger.warning(
                    "Workflow generation session cleanup was not confirmed; "
                    "continuing with the next frozen-pool member: %s",
                    "; ".join(details),
                )
            if session_owner is not None:
                unbind_active_session(session_owner, session)
                release_owner = getattr(engine, "release_lifecycle_owner", None)
                if callable(release_owner):
                    release_owner(session_owner)

        def transport_failure(exc: BaseException) -> bool:
            text = f"{type(exc).__name__}: {exc}".lower()
            return isinstance(exc, (TimeoutError, ConnectionError, OSError)) or any(
                token in text
                for token in (
                    "timeout", "timed out", "transport", "disconnect", "connection",
                    "rate limit", "ratelimit", "too many requests", "429",
                )
            )

        def provider_rejection(exc: BaseException | str) -> bool:
            text = str(exc).lower()
            return any(
                token in text
                for token in ("permission", "forbidden", "unauthorized")
            )

        def validate_and_write(script_content: str) -> tuple[str, dict[str, Any]]:
            is_valid, errors = validate_generated_script(
                script_content,
                review_agents=[],
                agent_pool=pool,
            )
            if not is_valid:
                raise ValueError(
                    "; ".join(errors[:3]) or "script validation failed"
                )
            meta = extract_meta_from_script(script_content) or {}
            unsupported = sorted(
                set(meta.get("tools", [])) - {item.tool_name for item in pool}
            )
            if unsupported:
                raise ValueError(
                    "脚本引用未确认工具: " + ", ".join(unsupported)
                )
            ensure_current()
            lock_context = artifact_lock if artifact_lock is not None else nullcontext()
            with lock_context:
                ensure_current()
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(script_content)
            return output_path, meta

        last_error = "模型未返回有效脚本"
        for binding_index, binding in enumerate(ordered_pool):
            ensure_current()
            now = _workflow_module().time.monotonic()
            remaining_hard_budget = hard_deadline - now
            if remaining_hard_budget <= 0:
                last_error = "Workflow generation total timeout exceeded"
                break
            remaining_bindings = len(ordered_pool) - binding_index
            member_deadline = now + (
                remaining_hard_budget / max(1, remaining_bindings)
            )
            for repair_attempt in range(1, _workflow_module()._SCRIPT_GENERATION_MAX_ATTEMPTS + 1):
                ensure_current()
                now = _workflow_module().time.monotonic()
                remaining = hard_deadline - now
                member_remaining = member_deadline - now
                if remaining <= 0:
                    last_error = "Workflow generation total timeout exceeded"
                    break
                if member_remaining <= 0:
                    last_error = f"{binding.agent_id} generation slice exhausted"
                    break
                if progress_callback:
                    progress_callback(
                        f"{binding.agent_id} 正在生成编排脚本（修复 {repair_attempt}/"
                        f"{_workflow_module()._SCRIPT_GENERATION_MAX_ATTEMPTS}）"
                    )

                retry_note = ""
                if repair_attempt > 1:
                    retry_note = (
                        "\n\nBounded repair attempt "
                        f"{repair_attempt}: the previous output failed validation: {last_error}. "
                        "Return one complete corrected JavaScript module."
                    )
                session = None
                should_fallback = False
                activity_buffer = _workflow_module()._WorkflowGenerationActivityBuffer()
                attempt_cancel_event = _workflow_module().threading.Event()
                attempt_quarantined = _workflow_module().threading.Event()
                try:
                    session_owner = current_generation_owner()
                    if session_owner is None:
                        raise RuntimeError(
                            "stale Workflow generation owner before session creation"
                        )
                    session = create_engine_session(
                        agent_type=binding.tool_name,
                        cwd=root_path,
                        thread_id=(
                            f"workflow_script_gen_{captured_session_key}_{binding.agent_id}_"
                            f"{repair_attempt}"
                        ),
                        model_name=binding.model_name,
                        cancel_event=attempt_cancel_event,
                        # Member timeout/startup failure is an expected input
                        # to frozen-pool fallback, not an unhandled service
                        # failure. This handler reports the bounded fallback.
                        startup_log_failures=False,
                    )
                    if session is None:
                        last_error = f"{binding.agent_id}: unable to create generation session"
                        should_fallback = True
                        break
                    if not bind_active_session(session_owner, session):
                        anchor_session_for_cleanup(session_owner, session)
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        raise RuntimeError(
                            "stale Workflow generation owner before session bind"
                        )
                    if not active_session_is_current(session_owner, session):
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        raise RuntimeError(
                            "stale Workflow generation owner before prompt send"
                        )

                    def on_event(
                        event: Any,
                        *,
                        _session: Any = session,
                        _session_owner: _WorkflowLifecycleOwner = session_owner,
                        _quarantine_event: threading.Event = attempt_quarantined,
                        _activity_buffer: _WorkflowGenerationActivityBuffer = activity_buffer,
                        _binding_agent_id: str = binding.agent_id,
                        _progress_callback: Any = progress_callback,
                    ) -> None:
                        if (
                            _quarantine_event.is_set()
                            or not active_session_is_current(
                                _session_owner,
                                _session,
                            )
                        ):
                            return
                        text = str(getattr(event, "text", "") or "")
                        event_type = getattr(event, "event_type", None)
                        event_name = str(getattr(event_type, "value", event_type) or "").lower()
                        meaningful = self._workflow_generation_event_is_meaningful(event)
                        if not meaningful or not _progress_callback:
                            return
                        summaries = _activity_buffer.feed(text) if text else ()
                        if not text:
                            summaries = (
                                *_activity_buffer.flush(),
                                event_name.replace("_", " "),
                            )
                        for summary in summaries:
                            _progress_callback(
                                f"{_binding_agent_id} last activity: {summary[:180]}"
                            )

                    try:
                        attempt_timeout = max(
                            0.001,
                            min(
                                configured_attempt_timeout,
                                remaining,
                                member_remaining,
                            ),
                        )
                        result = session.send_prompt(
                            base_prompt + retry_note,
                            timeout=attempt_timeout,
                            idle_timeout=max(0.001, min(120.0, attempt_timeout)),
                            on_event=on_event,
                            activity_predicate=self._workflow_generation_event_is_meaningful,
                        )
                    finally:
                        if progress_callback:
                            for summary in activity_buffer.flush():
                                progress_callback(
                                    f"{binding.agent_id} last activity: {summary[:180]}"
                                )
                    try:
                        ensure_current()
                    except Exception:
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        raise

                    stop_reason = str(getattr(result, "stop_reason", "") or "").strip().lower()
                    if stop_reason in {"cancelled", "canceled"}:
                        last_error = (
                            f"{binding.agent_id} terminal stop_reason={stop_reason}"
                        )
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        should_fallback = True
                        break
                    if stop_reason in {"timeout", "timed_out", "error", "failed", "failure"}:
                        last_error = f"{binding.agent_id} terminal stop_reason={stop_reason}"
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        should_fallback = True
                        break
                    if stop_reason not in {
                        "end_turn", "completed", "complete", "success", "done",
                    }:
                        last_error = f"{binding.agent_id} terminal stop_reason={stop_reason}"
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        should_fallback = True
                        break
                    text = str(getattr(result, "text", "") or "").strip()
                    if not text:
                        last_error = f"{binding.agent_id}: model returned an empty script"
                        close_strict(
                            session,
                            cancel=False,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        continue
                    script_content = self._strip_markdown_fences(text)
                    is_valid, errors = validate_generated_script(
                        script_content,
                        review_agents=[],
                        agent_pool=pool,
                    )
                    if not is_valid:
                        last_error = "; ".join(errors[:3]) or "script validation failed"
                        close_strict(
                            session,
                            cancel=False,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        continue
                    meta = extract_meta_from_script(script_content) or {}
                    unsupported = sorted(
                        set(meta.get("tools", [])) - {item.tool_name for item in pool}
                    )
                    if unsupported:
                        last_error = "脚本引用未确认工具: " + ", ".join(unsupported)
                        close_strict(
                            session,
                            cancel=False,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        continue

                    close_strict(
                        session,
                        cancel=False,
                        quarantine_event=attempt_quarantined,
                        provider_cancel_event=attempt_cancel_event,
                    )
                    session = None
                    return validate_and_write(script_content)
                except _workflow_module()._WorkflowGenerationCancelled:
                    if session is not None:
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                    raise
                except Exception as exc:
                    if session is not None:
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        if transport_failure(exc) or provider_rejection(exc):
                            session = None
                            last_error = f"{type(exc).__name__}: {exc}"
                            should_fallback = True
                            break
                    if "stale" in str(exc).lower() or "superseded" in str(exc).lower():
                        raise
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "Workflow generation %s repair %s failed: %s",
                        binding.agent_id,
                        repair_attempt,
                        exc,
                    )
                    continue
                if should_fallback:
                    break

        ensure_current()
        if progress_callback:
            progress_callback(
                "模型编排未完成，正在切换池内确定性编排并继续执行"
            )
        try:
            fallback_script = generate_simple_script(
                requirement,
                agent_pool=pool,
                orchestrator_agent_id=orchestrator_agent_id,
            )
            ensure_current()
            return validate_and_write(fallback_script)
        except _workflow_module()._WorkflowGenerationCancelled:
            raise
        except Exception as exc:
            if "stale" in str(exc).lower() or "superseded" in str(exc).lower():
                raise
            logger.error(
                "Workflow deterministic pool fallback failed after model generation: %s; "
                "last model error: %s",
                exc,
                last_error,
                exc_info=True,
            )
            raise _workflow_module()._WorkflowGenerationExhausted(
                "Workflow 自动编排恢复未完成，请重新发起 `/wf`。"
            ) from exc

    @staticmethod
    def _workflow_generation_event_is_meaningful(event: Any) -> bool:
        """Return True if an ACP generation event carries user-visible activity."""
        text = str(getattr(event, "text", "") or "").strip()
        if text:
            return True
        event_type = getattr(event, "event_type", None)
        event_name = str(getattr(event_type, "value", event_type) or "").lower()
        return any(
            token in event_name
            for token in (
                "tool_call_start",
                "tool_call_update",
                "tool_call_done",
                "plan_update",
                "image_chunk",
            )
        )

    @staticmethod
    def _strip_markdown_fences(content: str) -> str:
        """Remove markdown code fences and natural language preamble from AI output.

        AI models sometimes prefix their code output with explanatory text like
        "Let me analyze..." or "Here's the workflow script:". This method
        extracts the actual JavaScript code by:
        1. Attempting to extract code from markdown fences (even if preceded by text)
        2. Stripping any natural language preamble before the actual JS code
        """
        import re

        # Strategy 1: Find markdown code fence containing the actual code.
        # This handles cases like: "Here's the script:\n```javascript\n...code...\n```"
        fence_match = re.search(r"```\s*(?:javascript|js|)\s*\n", content, re.IGNORECASE)
        if fence_match:
            after_fence = content[fence_match.end() :]
            # Find the closing fence (last occurrence to handle nested fences in strings)
            close_idx = after_fence.rfind("```")
            if close_idx >= 0:
                content = after_fence[:close_idx].rstrip()
            else:
                content = after_fence.rstrip()
            # After extracting from fences, if it looks like valid JS, return it
            stripped = content.lstrip()
            if stripped and re.match(
                r"^(export|/[/*]|const |let |var |\"use strict\"|'use strict')",
                stripped,
            ):
                return content.strip()

        # Strategy 2: Original logic — content starts directly with a fence
        elif content.startswith("```"):
            lines = content.split("\n", 1)
            content = lines[1] if len(lines) > 1 else content
            if content.rstrip().endswith("```"):
                content = content.rstrip()[:-3].rstrip()

        # Strategy 3: Detect and strip natural language preamble.
        # If content doesn't start with valid JS syntax, find the actual code start.
        stripped = content.lstrip()
        if stripped and not re.match(
            r"^(export|/[/*]|const |let |var |\"use strict\"|'use strict'|/\*\*)",
            stripped,
        ):
            # Look for the start of the actual export statement (multiline search)
            export_match = re.search(
                r"^(export\s+const\s+meta\s*=|export\s+default\s)",
                content,
                re.MULTILINE,
            )
            if export_match:
                start_idx = export_match.start()
                # Include preceding JSDoc/comment lines that are part of the code
                preceding = content[:start_idx]
                if preceding.rstrip():
                    lines_before = preceding.rstrip().split("\n")
                    # Walk backwards to include leading comment block
                    comment_start = start_idx
                    for line in reversed(lines_before):
                        ls = line.strip()
                        if ls.startswith("//") or ls.startswith("*") or ls.startswith("/*") or ls.endswith("*/"):
                            # This line is a comment, include it
                            idx = content.rfind(line, 0, comment_start)
                            if idx >= 0:
                                comment_start = idx
                        else:
                            break
                    start_idx = comment_start
                content = content[start_idx:]

        return content.strip()

    @staticmethod
    def _build_run_spec(
        *,
        pending: Any,
        engine: Any,
        task: str,
        chat_id: str,
        topic_id: str | None,
    ):
        """Freeze the automatic tool binding into one execution contract."""
        from ...spec_engine.models import ReviewAgentBinding
        from ...workflow_engine.constants import (
            MAX_TOTAL_AGENTS,
            WORKFLOW_TOTAL_TIMEOUT_S,
        )
        from ...workflow_engine.run_spec import WorkflowRunSpec

        agent_pool = tuple(pending.agent_pool or ())
        if not agent_pool:
            raise ValueError("Workflow execution is missing its confirmed Agent Pool")
        orchestrator_agent_id = pending.orchestrator_agent_id or agent_pool[0].agent_id
        orchestrator_agent = next(
            (binding for binding in agent_pool if binding.agent_id == orchestrator_agent_id),
            None,
        )
        if orchestrator_agent is None:
            raise ValueError("Workflow orchestrator is outside its confirmed Agent Pool")
        orchestrator = ReviewAgentBinding(
            provider="workflow",
            tool_name=orchestrator_agent.tool_name,
            display_name=orchestrator_agent.display_name,
            agent_type=orchestrator_agent.tool_name,
            model_name=orchestrator_agent.model_name,
            model_display_name=orchestrator_agent.model_name,
            selection_key=orchestrator_agent.agent_id,
            use_default_model=orchestrator_agent.model_name is None,
        )
        allowed_tools = list(
            dict.fromkeys(binding.tool_name for binding in agent_pool)
        )

        settings = getattr(engine, "settings", None)
        raw_timeout = getattr(
            settings,
            "workflow_total_timeout_s",
            WORKFLOW_TOTAL_TIMEOUT_S,
        )
        try:
            timeout_s = int(raw_timeout)
        except (TypeError, ValueError):
            timeout_s = WORKFLOW_TOTAL_TIMEOUT_S
        deadline = _workflow_module().time.monotonic() + timeout_s if timeout_s > 0 else None

        return WorkflowRunSpec(
            orchestrator=orchestrator,
            reviewers=(),
            tool_model_map={},
            task=task,
            chat_id=chat_id,
            topic_id=topic_id,
            budget=MAX_TOTAL_AGENTS,
            deadline=deadline,
            auto_reviewer=True,
            initiator_user_id=pending.initiator_user_id or None,
            allowed_tools=tuple(allowed_tools),
            agent_pool=agent_pool,
            orchestrator_agent_id=orchestrator_agent_id,
        )

    def _create_workflow_process_sink(
        self,
        *,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"],
        project_id: str,
        input_text: str,
    ):
        """Build the optional task-scoped COT sink without starting it."""
        if getattr(self.settings, "feishu_cot_enabled", None) is not True:
            return None
        try:
            from ..cot import FeishuCOTAPIClient, FeishuCOTStream

            linked_segments: set[str] = set()

            def link_segment(process_message_id: str) -> None:
                if process_message_id in linked_segments:
                    return
                linked_segments.add(process_message_id)
                try:
                    request_id = self.ensure_request_id(
                        message_id,
                        chat_id=chat_id,
                        project_id=project_id or None,
                    )
                    self.ctx.message_linker.register_origin(
                        message_id,
                        request_id=request_id,
                        chat_id=chat_id,
                        project_id=project_id or None,
                    )
                    self.ctx.message_linker.link_reply(
                        message_id,
                        process_message_id,
                    )
                except Exception:
                    logger.debug(
                        "Failed to link Workflow COT segment",
                        exc_info=True,
                    )
                if project is not None:
                    try:
                        self.register_message_project(process_message_id, project)
                    except Exception:
                        logger.debug(
                            "Failed to register Workflow COT project",
                            exc_info=True,
                        )

            api = FeishuCOTAPIClient(
                self.ctx.api_client_factory(),
                outbound_audit=self.ctx.main_bot_outbound_audit,
                outbound_audit_failure=self.ctx.main_bot_outbound_audit_failure,
                tenant_key_resolver=self.ctx.tenant_key_resolver,
                outbound_target_aliases=lambda target: self._reply_audit_aliases(
                    self._resolve_origin(target)
                ),
                trust_revision_provider=self._managed_card_trust_revisions,
            )
            return FeishuCOTStream(
                api,
                chat_id=chat_id,
                origin_message_id=message_id,
                reply_in_thread=self.settings.default_reply_mode == "thread",
                input_text=input_text,
                detail=self.settings.feishu_cot_detail,
                request_timeout=_workflow_module()._WORKFLOW_COT_REQUEST_TIMEOUT_S,
                close_timeout=_workflow_module()._WORKFLOW_COT_CLOSE_TIMEOUT_S,
                on_segment_started=link_segment,
            )
        except Exception:
            logger.warning(
                "Failed to initialize Workflow COT; using card pages",
                exc_info=True,
            )
            return None

    @staticmethod
    def _workflow_process_terminal_event(wf_project: "WorkflowProject"):
        """Map Workflow terminal state onto the non-authoritative COT."""
        from ...card.events import CardEvent
        from ...workflow_engine.models import WorkflowStatus

        if wf_project.status == WorkflowStatus.COMPLETED:
            return CardEvent.completed()
        if wf_project.status == WorkflowStatus.CANCELLED:
            return CardEvent.cancelled(reason=wf_project.error or "Workflow cancelled")
        return CardEvent.failed(wf_project.error or "Workflow failed")

    def _build_workflow_callbacks(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"],
        *,
        lifecycle_owner: _WorkflowLifecycleOwner | None = None,
        requirement: str = "",
    ):
        """Build one-card Workflow callbacks with optional native COT process."""
        from ...workflow_engine.engine import WorkflowEngineCallbacks
        from ...workflow_engine.process_stream import WorkflowProcessStream
        from .workflow_card_pages import WorkflowCardPageDelivery

        card_message_id: list[str | None] = [message_id]  # Mutable ref for card updates
        page_delivery = WorkflowCardPageDelivery(card_message_id)
        terminal_sent: list[bool] = [False]
        delivery_lock = (
            lifecycle_owner.delivery_lock
            if lifecycle_owner is not None
            else _workflow_module().threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        )
        project_id = getattr(project, "project_id", "") or ""
        origin_message_id = self._resolve_origin(message_id)
        process_stream = WorkflowProcessStream(
            self._create_workflow_process_sink(
                message_id=origin_message_id,
                chat_id=chat_id,
                project=project,
                project_id=project_id,
                input_text=requirement or "Workflow 执行过程",
            )
        )
        process_stream.start()

        def process_replaces_detail_cards() -> bool:
            return process_stream.active

        def _status_pages(
            card_data: dict[str, Any] | list[dict[str, Any]],
        ) -> dict[str, Any] | list[dict[str, Any]]:
            if isinstance(card_data, dict):
                return card_data
            status_pages = [
                card
                for index, card in enumerate(card_data)
                if (
                    (page_key := card.get("_workflow_page_key")) is not None
                    and isinstance(page_key, (list, tuple))
                    and page_key
                    and page_key[0] == "status"
                )
                or (page_key is None and index == 0)
            ]
            return (status_pages or card_data[:1])[:1]

        def on_progress(
            card_data: dict[str, Any] | list[dict[str, Any]],
        ) -> None:
            """Update the progress card in Feishu."""
            with delivery_lock:
                if terminal_sent[0] or (lifecycle_owner is not None and lifecycle_owner.stop_event.is_set()):
                    logger.debug("Ignored workflow progress update after terminal card was sent")
                    return
                try:
                    # Keep runtime progress cards aligned with the mixin
                    # implementation: users can stop active workflows from the card,
                    # but terminal completion cards are delivered separately.
                    status_card = card_data[0] if isinstance(card_data, list) and card_data else card_data
                    if isinstance(status_card, dict):
                        self._inject_workflow_stop_button(
                            status_card,
                            chat_id,
                            project_id,
                            is_running=self._workflow_is_running_for_card(chat_id, project),
                        )
                    delivery_result = page_delivery.deliver(
                        (
                            _status_pages(card_data)
                            if process_replaces_detail_cards()
                            else card_data
                        ),
                        replace_or_send=self._replace_or_send_workflow_rendered_card,
                        chat_id=chat_id,
                        origin_message_id=origin_message_id,
                        # Heartbeats are periodic. Turning every PATCH failure
                        # into a new message creates an unbounded card stream
                        # during persistent transport or provenance failures.
                        status_fallback_to_new=False,
                    )
                    if delivery_result.status_message_id:
                        card_message_id[0] = delivery_result.status_message_id
                except Exception:
                    logger.debug(
                        "Failed to update workflow progress card",
                        exc_info=True,
                    )

        def _deliver_terminal_project(
            wf_project: "WorkflowProject",
            *,
            allow_stopped_owner: bool = False,
        ) -> None:
            """Deliver one authoritative terminal snapshot on stable pages."""
            report_status: dict[str, Any] | None = None
            failed_page_indexes: tuple[int, ...] = ()
            try:
                from ...workflow_engine.models import WorkflowStatus
                from ...workflow_engine.renderer import render_completion_cards

                process_complete = process_stream.complete(
                    self._workflow_process_terminal_event(wf_project)
                )
                if wf_project.status != WorkflowStatus.CANCELLED:
                    report_status = self._send_workflow_completion_report(
                        wf_project=wf_project,
                        chat_id=chat_id,
                        message_id=message_id,
                        project=project,
                    ) or {}
                compact_terminal = bool(
                    process_complete
                    and (
                        wf_project.status == WorkflowStatus.CANCELLED
                        or (report_status or {}).get("attachment_sent")
                    )
                )
                card_data = render_completion_cards(
                    wf_project,
                    report_status=report_status,
                )
                if compact_terminal:
                    card_data = _status_pages(card_data)
                delivery_result = page_delivery.deliver(
                    card_data,
                    replace_or_send=self._replace_or_send_workflow_rendered_card,
                    chat_id=chat_id,
                    origin_message_id=origin_message_id,
                    terminal=True,
                )
                if delivery_result.status_message_id:
                    card_message_id[0] = delivery_result.status_message_id
                if delivery_result.fully_delivered:
                    return

                failed_page_indexes = delivery_result.failed_page_indexes
                if report_status is None:
                    report_status = self._send_workflow_completion_report(
                        wf_project=wf_project,
                        chat_id=chat_id,
                        message_id=message_id,
                        project=project,
                    )
                if (
                    not allow_stopped_owner
                    and lifecycle_owner is not None
                    and lifecycle_owner.stop_event.is_set()
                ):
                    return

                # Retry every page once after the durable full report has
                # been generated. Existing page bindings are reused and
                # missing pages are created in order.
                retry_cards = render_completion_cards(
                    wf_project,
                    report_status=report_status,
                )
                if compact_terminal:
                    retry_cards = _status_pages(retry_cards)
                retry_result = page_delivery.deliver(
                    retry_cards,
                    replace_or_send=self._replace_or_send_workflow_rendered_card,
                    chat_id=chat_id,
                    origin_message_id=origin_message_id,
                    terminal=True,
                )
                if retry_result.status_message_id:
                    card_message_id[0] = retry_result.status_message_id
                if retry_result.fully_delivered or report_status.get("attachment_sent"):
                    return
                failed_page_indexes = retry_result.failed_page_indexes
            except Exception:
                if (
                    not allow_stopped_owner
                    and lifecycle_owner is not None
                    and lifecycle_owner.stop_event.is_set()
                ):
                    return
                if report_status is None:
                    report_status = self._send_workflow_completion_report(
                        wf_project=wf_project,
                        chat_id=chat_id,
                        message_id=message_id,
                        project=project,
                    )
                logger.warning("Workflow terminal card delivery failed", exc_info=True)

            if report_status and report_status.get("attachment_sent"):
                self._reply_workflow_completion_fallback(
                    message_id=message_id,
                    report_status=report_status,
                    failed_page_indexes=failed_page_indexes,
                )
                return
            self._reply_workflow_completion_fallback(
                message_id=message_id,
                report_status=report_status,
                failed_page_indexes=failed_page_indexes,
            )

        def on_done(wf_project: "WorkflowProject") -> None:
            """Final completion — send a structured completion card."""
            # Wait for an in-flight PATCH, then fence every later progress
            # callback before report generation/upload and terminal delivery.
            with delivery_lock:
                if terminal_sent[0] or (lifecycle_owner is not None and lifecycle_owner.stop_event.is_set()):
                    process_stream.abort()
                    return
                terminal_sent[0] = True
                _deliver_terminal_project(wf_project)

        def on_error(
            error_msg: str,
            wf_project: "WorkflowProject | None" = None,
        ) -> None:
            """Deliver an execution failure, or a project-less setup error."""
            with delivery_lock:
                if terminal_sent[0] or (lifecycle_owner is not None and lifecycle_owner.stop_event.is_set()):
                    process_stream.abort()
                    return
                terminal_sent[0] = True
                if wf_project is not None:
                    _deliver_terminal_project(wf_project)
                    return
                process_stream.abort()

                from ...workflow_engine.errors import _strip_internal_details

                workflow_category = self._workflow_error_card_category(error_msg)
                card = self._build_error_card(
                    workflow_category,
                    detail=_strip_internal_details(error_msg or ""),
                )
                self._replace_or_send_workflow_card(
                    card_message_id=card_message_id[0],
                    chat_id=chat_id,
                    card=card,
                    origin_message_id=origin_message_id,
                )

        def on_cancelled(wf_project: "WorkflowProject") -> None:
            """Deliver the stopped run's final stream despite its stop fence."""
            with delivery_lock:
                stop_delivery_fence = getattr(
                    lifecycle_owner,
                    "stop_delivery_fenced_event",
                    None,
                )
                if terminal_sent[0] or (
                    isinstance(stop_delivery_fence, _workflow_module().threading.Event)
                    and stop_delivery_fence.is_set()
                ):
                    process_stream.abort()
                    return
                terminal_sent[0] = True
                _deliver_terminal_project(
                    wf_project,
                    allow_stopped_owner=True,
                )

        def on_log(msg: str) -> None:
            logger.debug("[WorkflowHandler] log: %s", msg)

        def on_agent_start(label: str, tool: str) -> None:
            process_stream.agent_started(label, tool)

        def on_agent_event(label: str, event: Any) -> None:
            process_stream.emit(label, event)

        def on_agent_done(label: str, payload: dict[str, Any]) -> None:
            process_stream.agent_done(label, payload)

        return WorkflowEngineCallbacks(
            on_progress=on_progress,
            on_done=on_done,
            on_cancelled=on_cancelled,
            on_error=on_error,
            on_log=on_log,
            on_agent_start=on_agent_start,
            on_agent_event=on_agent_event,
            on_agent_done=on_agent_done,
            image_uploader=getattr(self, "upload_acp_image", None),
        )
