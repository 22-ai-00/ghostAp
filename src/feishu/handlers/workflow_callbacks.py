"""Workflow run-spec building, COT process sink, and card callbacks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...project import ProjectContext
    from ...workflow_engine.models import WorkflowProject
    from .workflow import _WorkflowLifecycleOwner

logger = logging.getLogger(__name__)


def _workflow_module():
    """Return the workflow handler module for runtime-patchable symbols."""
    from . import workflow as _wf_mod

    return _wf_mod


class WorkflowCallbacksMixin:
    """Mixin providing workflow run-spec binding, COT stream, and callbacks."""

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
