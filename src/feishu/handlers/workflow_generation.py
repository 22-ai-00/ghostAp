"""Workflow script generation session validation and scheduling helpers."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import threading

    from ...project import ProjectContext
    from .workflow import _WorkflowLifecycleOwner

logger = logging.getLogger(__name__)

_GENERATION_PROGRESS_DEBOUNCE_S = 8.0


def _workflow_module():
    """Return the workflow handler module for runtime-patchable symbols."""
    from . import workflow as _wf_mod

    return _wf_mod


class WorkflowGenerationMixin:
    """Mixin providing workflow script generation lifecycle and scheduling."""

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
            if (
                isinstance(current_owner, _workflow_module()._WorkflowLifecycleOwner)
                and not current_owner.stop_event.is_set()
            ):
                if current_owner.session_key == expected_session_key:
                    owner = current_owner
                    if owner.source_script_path is None:
                        object.__setattr__(
                            owner,
                            "source_script_path",
                            self._new_workflow_script_path(root_path),
                        )
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
                "preset_id": (
                    getattr(existing_pending, "preset_id", None)
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
        progress_coalescer: Any = None
        generation_terminal_fence = _workflow_module().threading.Event()
        try:
            origin_message_id = self._resolve_origin(message_id)
        except Exception:
            logger.debug(
                "Workflow origin resolution failed; using inbound message",
                exc_info=True,
            )
            origin_message_id = message_id
        gen_msg_id: str | None = None
        heartbeat_start = _workflow_module().time.time()
        card_project_id = (
            (getattr(project, "project_id", "") if project else "")
            or str(generation_context.get("project_id") or "")
        )
        last_delivered_key: tuple[str, int] | None = None

        def _owner_can_deliver() -> bool:
            return self._generation_owner_is_active(engine, owner)

        def _deliver_generation_card(status_hint: str = "") -> None:
            """Update the generating card only while this owner is live."""
            nonlocal last_delivered_key
            with owner.delivery_lock:
                if generation_terminal_fence.is_set() or not _owner_can_deliver():
                    return
                elapsed = int(_workflow_module().time.time() - heartbeat_start)
                status = status_hint or "正在生成编排脚本..."
                delivery_key = (status, elapsed // 8)
                if delivery_key == last_delivered_key:
                    return
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
                    chat_id=chat_id,
                    project_id=card_project_id,
                )
                if gen_msg_id:
                    updated = self.update_card(
                        gen_msg_id,
                        self._build_workflow_card_from_renderer_data(
                            progress_card_data
                        ),
                    )
                    if updated is not False:
                        last_delivered_key = delivery_key

        def _heartbeat_update(status_hint: str = "") -> None:
            if progress_coalescer is not None:
                progress_coalescer.enqueue(status_hint)
            else:
                _deliver_generation_card(status_hint)

        def _heartbeat_loop() -> None:
            while not owner.stop_event.is_set() and not owner.heartbeat_stop_event.is_set():
                owner.heartbeat_stop_event.wait(10.0)
                if not owner.stop_event.is_set() and not owner.heartbeat_stop_event.is_set():
                    try:
                        _heartbeat_update()
                    except Exception:
                        pass

        def _stop_heartbeat(*, discard_pending: bool = False) -> None:
            if discard_pending:
                generation_terminal_fence.set()
            owner.heartbeat_stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=2.0)
            if progress_coalescer is not None:
                progress_coalescer.stop()

        script_path: str
        meta: dict[str, Any] | None = None
        preset_id = str(generation_context.get("preset_id") or "").strip().lower()
        use_preset_fastpath = bool(preset_id and preset_id != "custom")
        try:
            if use_preset_fastpath:
                from ...card.builders.workflow import build_workflow_preset_launch_card

                gen_card_data = build_workflow_preset_launch_card(
                    requirement=requirement,
                    preset_id=preset_id,
                    members=generation_context["agent_pool"],
                    orchestrator_agent_id=(
                        generation_context["orchestrator_agent_id"] or ""
                    ),
                    chat_id=chat_id,
                    project_id=card_project_id or None,
                )
            else:
                gen_card_data = self._workflow_generation_card_data(
                    requirement=requirement,
                    agent_pool=generation_context["agent_pool"],
                    orchestrator_agent_id=(
                        generation_context["orchestrator_agent_id"] or ""
                    ),
                    orchestrator_was_auto=generation_context[
                        "orchestrator_was_auto"
                    ],
                    chat_id=chat_id,
                    project_id=card_project_id,
                )
            with owner.delivery_lock:
                if not _owner_can_deliver():
                    return
                if (
                    not use_preset_fastpath
                    and message_id
                    and owner.card_message_id == message_id
                ):
                    gen_msg_id = message_id
                else:
                    gen_msg_id = self._replace_or_send_workflow_rendered_card(
                        card_message_id=message_id,
                        chat_id=chat_id,
                        card_data=gen_card_data,
                        origin_message_id=origin_message_id,
                    ) or message_id
                    object.__setattr__(owner, "card_message_id", gen_msg_id)
            if not _owner_can_deliver():
                return

            if use_preset_fastpath:
                from ...workflow_engine.script_gen import (
                    extract_meta_from_script,
                    validate_generated_script,
                )
                from ...workflow_engine.teamwork import generate_teamwork_script

                confirmed_pool = tuple(generation_context["agent_pool"])
                if not confirmed_pool:
                    raise RuntimeError("Workflow generation lost its confirmed Agent Pool")
                script_content = generate_teamwork_script(
                    requirement,
                    agent_pool=confirmed_pool,
                    preset=preset_id,
                    orchestrator_agent_id=generation_context["orchestrator_agent_id"],
                )
                is_valid, errors = validate_generated_script(
                    script_content,
                    review_agents=[],
                    agent_pool=confirmed_pool,
                )
                if not is_valid:
                    raise ValueError("; ".join(errors[:3]) or "preset script validation failed")
                meta = extract_meta_from_script(script_content) or {}
                script_path = owner.source_script_path or self._new_workflow_script_path(root_path)
                with owner.delivery_lock:
                    if not _owner_can_deliver():
                        raise _workflow_module()._WorkflowGenerationCancelled(
                            "Workflow generation cancelled"
                        )
                    os.makedirs(os.path.dirname(script_path), exist_ok=True)
                    with open(script_path, "w", encoding="utf-8") as file:
                        file.write(script_content)
            else:
                from ...workflow_engine.progress_coalescer import ProgressCoalescer

                progress_coalescer = ProgressCoalescer(
                    on_progress=_deliver_generation_card,
                    debounce_s=_GENERATION_PROGRESS_DEBOUNCE_S,
                )
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
                preset_id=generation_context.get("preset_id"),
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
            _stop_heartbeat(discard_pending=True)
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
                        chat_id=chat_id,
                        project_id=card_project_id,
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
            _stop_heartbeat(discard_pending=True)
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
                            chat_id=chat_id,
                            project_id=(
                                (getattr(project, "project_id", "") if project else "")
                                or str(getattr(cancelled_pending, "project_id", "") or "")
                            ),
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
                while (
                    _workflow_module().time.monotonic() < deadline
                    and not lifecycle_owner.stop_event.wait(1.0)
                ):
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
