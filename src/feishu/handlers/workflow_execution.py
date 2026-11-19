"""Workflow execution, stop, queueing, and lifecycle ownership helpers."""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, Optional

from .workflow_ai_codegen import WorkflowAICodegenMixin
from .workflow_callbacks import WorkflowCallbacksMixin
from .workflow_generation import WorkflowGenerationMixin

if TYPE_CHECKING:
    from ...project import ProjectContext
    from .workflow import _WorkflowLifecycleOwner

logger = logging.getLogger(__name__)


def _workflow_module():
    """Return the workflow handler module for runtime-patchable symbols."""
    from . import workflow as _wf_mod

    return _wf_mod


class WorkflowExecutionMixin(
    WorkflowGenerationMixin,
    WorkflowAICodegenMixin,
    WorkflowCallbacksMixin,
):
    """Mixin providing workflow execution, stop, queueing, and lifecycle flows."""

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
                if isinstance(generation_owner, _workflow_module()._WorkflowLifecycleOwner)
                and not generation_owner.stop_event.is_set()
                else None
            )
            live_start_owner = (
                start_owner
                if isinstance(start_owner, _workflow_module()._WorkflowLifecycleOwner)
                and not start_owner.stop_event.is_set()
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
