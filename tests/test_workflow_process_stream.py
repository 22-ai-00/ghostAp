"""Workflow process-stream and compact card-delivery contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, patch

from src.card.events import CardEvent, CardEventType
from src.card.protocols import ProcessSegmentRollover
from src.feishu.handlers.workflow import WorkflowHandler
from src.workflow_engine.models import WorkflowProject, WorkflowStatus
from src.workflow_engine.process_stream import WorkflowProcessStream


class _Sink:
    def __init__(self) -> None:
        self.started = False
        self.healthy = True
        self.message_id = "cot-message"
        self.events: list[CardEvent] = []
        self.completed: list[CardEvent] = []
        self.aborted = False
        self.reject_next = False
        self.rollover_calls = 0

    def start(self) -> None:
        self.started = True

    def emit(self, event: CardEvent) -> bool:
        if self.reject_next:
            self.reject_next = False
            self.healthy = False
            return False
        self.events.append(event)
        return True

    def rollover(self):
        self.rollover_calls += 1
        return ProcessSegmentRollover(sealed=True, started=True)

    def complete(self, event: CardEvent) -> bool:
        self.completed.append(event)
        self.started = False
        return True

    def abort(self) -> None:
        self.aborted = True
        self.started = False


def _page(label: str, kind: str, identity: int | str = -1) -> dict[str, Any]:
    return {
        "header": {
            "title": {"tag": "plain_text", "content": label},
            "template": "blue",
        },
        "elements": [{"tag": "markdown", "content": label}],
        "_workflow_page_key": (kind, identity, 0),
    }


def _handler(sink: _Sink) -> WorkflowHandler:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock(return_value="new-message")
    handler.reply_text = MagicMock()
    handler._resolve_origin = MagicMock(return_value="origin-message")
    handler._workflow_is_running_for_card = MagicMock(return_value=True)
    handler._create_workflow_process_sink = MagicMock(return_value=sink)
    handler._send_workflow_completion_report = MagicMock(
        return_value={"generated": True, "attachment_sent": True}
    )
    handler._reply_workflow_completion_fallback = MagicMock()
    return handler


def test_process_stream_namespaces_agent_events_and_closes_once() -> None:
    sink = _Sink()
    stream = WorkflowProcessStream(sink)

    assert stream.agent_started("worker", "codex")
    assert stream.emit("worker", CardEvent.text_started("answer"))
    assert stream.emit("worker", CardEvent.text_delta("answer", "hello"))
    assert stream.emit("worker", CardEvent.text_done("answer"))
    assert stream.agent_done("worker", {"error": None, "cached": False})
    assert stream.complete(CardEvent.completed())

    namespaced = [
        event.payload["block_id"]
        for event in sink.events
        if event.payload.get("block_id", "").endswith("answer")
    ]
    assert namespaced == ["workflow:worker:answer"] * 3
    assert sink.completed[0].type is CardEventType.COMPLETED
    assert not stream.active


def test_process_stream_failure_aborts_and_stays_in_card_fallback() -> None:
    sink = _Sink()
    stream = WorkflowProcessStream(sink)

    assert stream.agent_started("worker", "codex")
    sink.reject_next = True
    assert not stream.emit("worker", CardEvent.text_started("answer"))
    assert sink.aborted
    assert not stream.active
    assert not stream.agent_started("later", "claude")


def test_process_stream_rolls_over_only_at_bounded_segment_budget(
    monkeypatch,
) -> None:
    sink = _Sink()
    stream = WorkflowProcessStream(sink)
    monkeypatch.setattr(
        "src.workflow_engine.process_stream._SEGMENT_MAX_EVENTS",
        4,
    )

    assert stream.start()
    assert stream.emit("worker", CardEvent.text_started("first"))
    assert stream.emit("worker", CardEvent.text_delta("first", "one"))
    assert stream.emit("worker", CardEvent.text_done("first"))
    assert sink.rollover_calls == 0
    assert stream.emit("worker", CardEvent.text_started("second"))
    assert stream.emit("worker", CardEvent.text_delta("second", "two"))

    assert sink.rollover_calls == 1
    assert stream.active


def test_workflow_cot_replaces_detail_cards_and_preserves_report() -> None:
    sink = _Sink()
    handler = _handler(sink)
    project_context = MagicMock(project_id="project-1")
    callbacks = handler._build_workflow_callbacks(
        "status-message",
        "chat-1",
        project_context,
        requirement="implement and verify",
    )

    assert sink.started
    callbacks.on_agent_start("worker", "codex")
    callbacks.on_agent_event("worker", CardEvent.text_started("answer"))
    callbacks.on_agent_event("worker", CardEvent.text_delta("answer", "work"))
    callbacks.on_agent_event("worker", CardEvent.text_done("answer"))
    callbacks.on_progress([_page("status-running", "status"), _page("agent-running", "agent", "call-1")])

    assert [call.args[0] for call in handler.update_card.call_args_list] == [
        "status-message"
    ]
    handler.send_card_to_chat.assert_not_called()

    terminal = WorkflowProject(
        name="compact",
        status=WorkflowStatus.COMPLETED,
        result='{"final_report": "done"}',
    )
    with patch(
        "src.workflow_engine.renderer.render_completion_cards",
        return_value=[
            _page("status-terminal", "status"),
            _page("agent-terminal", "agent", "call-1"),
            _page("ledger-terminal", "ledger"),
        ],
    ):
        callbacks.on_done(terminal)

    handler._send_workflow_completion_report.assert_called_once()
    handler.send_card_to_chat.assert_not_called()
    assert "status-terminal" in str(handler.update_card.call_args_list[-1].args[1])
    assert sink.completed[0].type is CardEventType.COMPLETED


def test_workflow_cot_failure_restores_lossless_detail_cards() -> None:
    sink = _Sink()
    handler = _handler(sink)
    callbacks = handler._build_workflow_callbacks(
        "status-message",
        "chat-1",
        MagicMock(project_id="project-1"),
        requirement="implement and verify",
    )

    callbacks.on_agent_start("worker", "codex")
    sink.reject_next = True
    callbacks.on_agent_event("worker", CardEvent.text_started("answer"))
    callbacks.on_progress([_page("status", "status"), _page("agent-detail", "agent", "call-1")])

    assert sink.aborted
    assert handler.send_card_to_chat.call_count == 1
    assert "agent-detail" in str(handler.send_card_to_chat.call_args.args[1])


def test_workflow_builds_native_cot_with_shared_transport_contract() -> None:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.settings.feishu_cot_enabled = True
    handler.ctx.settings.feishu_cot_detail = "brief"
    handler.ctx.settings.default_reply_mode = "thread"
    handler._reply_audit_aliases = MagicMock(return_value=("chat-1",))
    handler._resolve_origin = MagicMock(side_effect=lambda value: value)
    handler._managed_card_trust_revisions = MagicMock(return_value=None)
    handler.ensure_request_id = MagicMock(return_value="request-1")
    handler.register_message_project = MagicMock()
    project = MagicMock(project_id="project-1")
    api = MagicMock(name="workflow_cot_api")
    sink = MagicMock(name="workflow_cot_stream")

    with (
        patch("src.feishu.cot.FeishuCOTAPIClient", return_value=api) as api_cls,
        patch("src.feishu.cot.FeishuCOTStream", return_value=sink) as stream_cls,
    ):
        result = handler._create_workflow_process_sink(
            message_id="origin-1",
            chat_id="chat-1",
            project=project,
            project_id="project-1",
            input_text="implement and verify",
        )

    assert result is sink
    api_cls.assert_called_once_with(
        handler.ctx.api_client_factory.return_value,
        outbound_audit=handler.ctx.main_bot_outbound_audit,
        outbound_audit_failure=handler.ctx.main_bot_outbound_audit_failure,
        tenant_key_resolver=handler.ctx.tenant_key_resolver,
        outbound_target_aliases=ANY,
        trust_revision_provider=handler._managed_card_trust_revisions,
    )
    stream_cls.assert_called_once_with(
        api,
        chat_id="chat-1",
        origin_message_id="origin-1",
        reply_in_thread=True,
        input_text="implement and verify",
        detail="brief",
        request_timeout=5.0,
        close_timeout=2.5,
        on_segment_started=ANY,
    )
    stream_cls.call_args.kwargs["on_segment_started"]("cot-segment-1")
    handler.ctx.message_linker.link_reply.assert_called_once_with(
        "origin-1",
        "cot-segment-1",
    )
    handler.register_message_project.assert_called_once_with(
        "cot-segment-1",
        project,
    )


def test_workflow_cot_terminal_without_report_falls_back_to_detail_cards() -> None:
    sink = _Sink()
    handler = _handler(sink)
    handler._send_workflow_completion_report.return_value = {
        "generated": True,
        "attachment_sent": False,
        "error": "upload failed",
    }
    callbacks = handler._build_workflow_callbacks(
        "status-message",
        "chat-1",
        MagicMock(project_id="project-1"),
        requirement="implement and verify",
    )
    terminal = WorkflowProject(
        name="fallback",
        status=WorkflowStatus.COMPLETED,
        result='{"final_report": "done"}',
    )

    with patch(
        "src.workflow_engine.renderer.render_completion_cards",
        return_value=[
            _page("status-terminal", "status"),
            _page("agent-terminal", "agent", "call-1"),
            _page("ledger-terminal", "ledger"),
        ],
    ):
        callbacks.on_done(terminal)

    sent = "\n".join(str(call.args[1]) for call in handler.send_card_to_chat.call_args_list)
    assert "agent-terminal" in sent
    assert "ledger-terminal" in sent
    handler._reply_workflow_completion_fallback.assert_not_called()


def test_workflow_cancelled_cot_stays_single_card_without_report_attachment() -> None:
    sink = _Sink()
    handler = _handler(sink)
    callbacks = handler._build_workflow_callbacks(
        "status-message",
        "chat-1",
        MagicMock(project_id="project-1"),
        requirement="cancel me",
    )
    terminal = WorkflowProject(
        name="cancelled",
        status=WorkflowStatus.CANCELLED,
        error="Workflow cancelled",
    )

    with patch(
        "src.workflow_engine.renderer.render_completion_cards",
        return_value=[
            _page("status-cancelled", "status"),
            _page("agent-cancelled", "agent", "call-1"),
        ],
    ):
        callbacks.on_cancelled(terminal)

    handler._send_workflow_completion_report.assert_not_called()
    handler.send_card_to_chat.assert_not_called()
    assert sink.completed[0].type is CardEventType.CANCELLED


def test_workflow_ux_preview_documents_single_card_plus_cot_contract() -> None:
    preview = (
        Path(__file__).resolve().parents[1] / "ux" / "workflow-message-card.html"
    ).read_text(encoding="utf-8")

    assert "一张 Workflow 主卡承载进度与终态" in preview
    assert "Workflow 执行过程 · COT" in preview
    assert "不会为 Agent 单独新增卡片" in preview
    assert "Workflow 完整报告.html" in preview
