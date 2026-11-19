"""End-to-end ACP image upload and rendering tests for Workflow execution."""

from __future__ import annotations

import base64
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.acp.models import ACPEvent, ACPEventType, ACPImageInfo
from src.card.events import CardEvent, CardEventType
from src.card.protocols import ProcessSegmentRollover
from src.card.schema import require_card_json_2
from src.feishu.handlers.workflow import WorkflowHandler
from src.spec_engine.models import ReviewAgentBinding
from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.engine import WorkflowEngine, WorkflowEngineCallbacks
from src.workflow_engine.executor import AgentExecutor
from src.workflow_engine.models import (
    AgentCallParams,
    WorkflowProject,
    WorkflowRunSpec,
    WorkflowStatus,
)
from src.workflow_engine.process_stream import WorkflowProcessStream
from src.workflow_engine.renderer import (
    WorkflowProgressRenderer,
    _render_agent_execution_cards,
    clear_agent_execution_card_cache,
    render_completion_card,
)
from src.workflow_engine.state_manager import WorkflowStateManager

_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRfake-png-payload"
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")


class _StrictCotSink:
    """Fake COT sink that mimics FeishuCOTSession rejecting non-COT event types."""

    _COT_TYPES = frozenset(
        {
            CardEventType.TEXT_STARTED,
            CardEventType.TEXT_DELTA,
            CardEventType.TEXT_DONE,
            CardEventType.REASONING_STARTED,
            CardEventType.REASONING_DELTA,
            CardEventType.REASONING_DONE,
            CardEventType.TOOL_STARTED,
            CardEventType.TOOL_DELTA,
            CardEventType.TOOL_DONE,
            CardEventType.TOOL_FAILED,
        }
    )

    def __init__(self) -> None:
        self.started = False
        self.healthy = True
        self.aborted = False
        self.events: list[CardEvent] = []
        self.completed: list[CardEvent] = []

    def start(self) -> None:
        self.started = True

    def emit(self, event: CardEvent) -> bool:
        if event.type not in self._COT_TYPES:
            self.healthy = False
            return False
        self.events.append(event)
        return True

    def rollover(self) -> ProcessSegmentRollover:
        return ProcessSegmentRollover(sealed=True, started=True)

    def complete(self, event: CardEvent) -> bool:
        self.completed.append(event)
        self.started = False
        return True

    def abort(self) -> None:
        self.aborted = True
        self.started = False


def _make_handler(sink: _StrictCotSink, im_client: MagicMock) -> WorkflowHandler:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.im_client = im_client
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock(return_value="sent-card-1")
    handler.reply_text = MagicMock()
    handler._resolve_origin = MagicMock(return_value="origin-msg-1")
    handler._workflow_is_running_for_card = MagicMock(return_value=True)
    handler._create_workflow_process_sink = MagicMock(return_value=sink)
    handler._send_workflow_completion_report = MagicMock(
        return_value={"generated": True, "attachment_sent": True}
    )
    handler._reply_workflow_completion_fallback = MagicMock()
    return handler


def _tagged_nodes(value: object, tag: str) -> list[dict]:
    found: list[dict] = []
    if isinstance(value, dict):
        if value.get("tag") == tag:
            found.append(value)
        for child in value.values():
            found.extend(_tagged_nodes(child, tag))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_tagged_nodes(child, tag))
    return found


def test_process_stream_ignores_non_cot_events_without_failing() -> None:
    sink = _StrictCotSink()
    stream = WorkflowProcessStream(sink)

    assert stream.agent_started("worker", "codex")
    assert stream.emit("worker", CardEvent.image_added("img-1", "img_key_1", "plot.png"))
    assert stream.emit("worker", CardEvent.image_failed("img-2", "broken.png"))
    assert stream.emit(
        "worker",
        CardEvent(
            type=CardEventType.TASK_LIST_UPDATED,
            payload={"tasks": [{"task_id": "t1", "name": "step", "status": "in_progress"}]},
        ),
    )
    assert stream.emit("worker", CardEvent.text_started("ans"))
    assert stream.emit("worker", CardEvent.text_delta("ans", "done"))
    assert stream.emit("worker", CardEvent.text_done("ans"))

    assert stream.active
    assert sink.healthy
    assert not sink.aborted
    ans_events = [
        e for e in sink.events if e.payload.get("block_id") == "workflow:worker:ans"
    ]
    assert [e.type for e in ans_events] == [
        CardEventType.TEXT_STARTED,
        CardEventType.TEXT_DELTA,
        CardEventType.TEXT_DONE,
    ]
    assert all(
        e.type not in {CardEventType.IMAGE_ADDED, CardEventType.IMAGE_FAILED, CardEventType.TASK_LIST_UPDATED}
        for e in sink.events
    )


def test_build_workflow_callbacks_wires_upload_acp_image_through_engine(tmp_path) -> None:
    sink = _StrictCotSink()
    im_client = MagicMock()
    im_client.upload_image_bytes.return_value = "img_v3_wired"
    handler = _make_handler(sink, im_client)

    callbacks = handler._build_workflow_callbacks(
        "status-msg-1",
        "chat-1",
        MagicMock(project_id="proj-1"),
        requirement="generate chart",
    )
    assert callbacks.image_uploader == handler.upload_acp_image

    captured_uploader = {}

    class _FakeExecutor:
        def __init__(self, **kwargs):
            captured_uploader["fn"] = kwargs.get("image_uploader")

        def shutdown(self, wait: bool = True) -> None:
            return None

    class _FakeBridge:
        @staticmethod
        def check_node_available() -> bool:
            return True

        def __init__(self, **_kwargs):
            return None

        def start(self) -> None:
            return None

        def run(self) -> str:
            return '{"final_report": "ok"}'

        def stop(self) -> None:
            return None

    script_file = tmp_path / "wf.js"
    script_file.write_text("// workflow\n", encoding="utf-8")
    engine = WorkflowEngine("chat-1", str(tmp_path))
    binding = WorkflowAgentBinding(
        agent_id="A1",
        tool_name="codex",
        display_name="Codex",
        model_name=None,
    )
    orch = ReviewAgentBinding(
        provider="workflow",
        tool_name="codex",
        display_name="Codex",
        agent_type="codex",
        use_default_model=True,
    )
    run_spec = WorkflowRunSpec(
        orchestrator=orch,
        reviewers=(),
        tool_model_map={"codex": None},
        task="generate chart",
        chat_id="chat-1",
        topic_id=None,
        budget=10,
        deadline=None,
        auto_reviewer=True,
        allowed_tools=("codex",),
        enforce_tool_allowlist=False,
        agent_pool=(binding,),
        orchestrator_agent_id="A1",
    )

    with (
        patch("src.workflow_engine.engine.AgentExecutor", _FakeExecutor),
        patch("src.workflow_engine.engine.RuntimeBridge", _FakeBridge),
    ):
        project = engine.execute_workflow(
            str(script_file),
            callbacks=callbacks,
            run_spec=run_spec,
        )

    assert project.status == WorkflowStatus.COMPLETED
    assert captured_uploader["fn"] == handler.upload_acp_image


def test_agent_executor_caches_image_uploads_per_run_across_retries(tmp_path) -> None:
    image = ACPImageInfo(
        image_id="sha256:retry-chart",
        mime_type="image/png",
        data=_PNG_B64,
        name="retry-chart.png",
    )
    uploader = MagicMock(return_value="img_cached_once")

    class _RetryAndRepairSession:
        def __init__(self, attempt: int) -> None:
            self._attempt = attempt
            self._turn = 0

        def send_prompt(self, _prompt: str, *, on_event=None, **_kwargs):
            self._turn += 1
            if on_event is not None:
                on_event(ACPEvent(event_type=ACPEventType.IMAGE_CHUNK, image=image))
            if self._attempt == 1:
                raise RuntimeError("503 Service Unavailable")
            text = '{"bad": 1}' if self._turn == 1 else '{"ok": true}'
            return SimpleNamespace(text=text, output_tokens=0, stop_reason="end_turn")

        def cancel(self) -> None:
            return None

        def close(self) -> None:
            return None

    captured_events: list[CardEvent] = []
    sessions = [_RetryAndRepairSession(1), _RetryAndRepairSession(2)]
    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=threading.Event(),
        on_card_event=lambda _label, event: captured_events.append(event),
        image_uploader=uploader,
    )

    try:
        with (
            patch(
                "src.agent_session.factory.create_engine_session",
                side_effect=sessions,
            ),
            patch.object(executor, "_sleep_with_backoff"),
        ):
            result = executor.execute(
                AgentCallParams(
                    prompt="plot",
                    tool="codex",
                    label="chart-call",
                    schema={"ok": "boolean"},
                )
            )
    finally:
        executor.shutdown(wait=True)

    assert result.error is None
    uploader.assert_called_once_with(image)
    added = [e for e in captured_events if e.type is CardEventType.IMAGE_ADDED]
    assert len(added) == 3
    assert {e.payload["image_key"] for e in added} == {"img_cached_once"}


@pytest.mark.parametrize(
    ("upload_return", "expect_success"),
    [
        ("img_v3_chart_ok", True),
        (None, False),
        (RuntimeError("feishu upload 500"), False),
    ],
)
def test_acp_image_chunk_end_to_end_persists_key_keeps_cot_healthy_and_renders_cards(
    tmp_path,
    upload_return: str | Exception | None,
    expect_success: bool,
) -> None:
    clear_agent_execution_card_cache()
    sink = _StrictCotSink()
    im_client = MagicMock()
    if isinstance(upload_return, Exception):
        im_client.upload_image_bytes.side_effect = upload_return
    else:
        im_client.upload_image_bytes.return_value = upload_return

    handler = _make_handler(sink, im_client)
    raw_callbacks = handler._build_workflow_callbacks(
        "status-msg-1",
        "chat-1",
        MagicMock(project_id="proj-1"),
        requirement="render workflow image",
    )

    # Exercise WorkflowEngine._handle_agent_card_event -> state_manager + callbacks + progress.
    engine = WorkflowEngine("chat-1", str(tmp_path))
    project = WorkflowProject(name="img-wf", status=WorkflowStatus.RUNNING, started_at=100.0)
    state_manager = WorkflowStateManager(project)
    engine._project = project
    engine._state_manager = state_manager
    engine._renderer_wf = WorkflowProgressRenderer(project)
    engine._callbacks = WorkflowEngineCallbacks(
        on_progress=raw_callbacks.on_progress,
        on_done=raw_callbacks.on_done,
        on_cancelled=raw_callbacks.on_cancelled,
        on_error=raw_callbacks.on_error,
        on_agent_start=raw_callbacks.on_agent_start,
        on_agent_event=raw_callbacks.on_agent_event,
        on_agent_done=raw_callbacks.on_agent_done,
        image_uploader=raw_callbacks.image_uploader,
    )

    label = state_manager.on_agent_started("plotter", "codex", "build", agent_id="A1")
    raw_callbacks.on_agent_start(label, "codex")

    image = ACPImageInfo(
        image_id="sha256:chart-1",
        mime_type="image/png",
        data=_PNG_B64,
        name="chart.png",
    )

    class _ImageSession:
        def send_prompt(self, _prompt: str, *, on_event=None, **_kwargs):
            assert on_event is not None
            on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="Rendering chart"))
            on_event(ACPEvent(event_type=ACPEventType.IMAGE_CHUNK, image=image))
            return SimpleNamespace(text="chart complete", output_tokens=10, stop_reason="end_turn")

        def cancel(self) -> None:
            return None

        def close(self) -> None:
            return None

    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=threading.Event(),
        on_card_event=engine._handle_agent_card_event,
        image_uploader=raw_callbacks.image_uploader,
    )

    try:
        with patch(
            "src.agent_session.factory.create_engine_session",
            return_value=_ImageSession(),
        ):
            call_result = executor.execute(
                AgentCallParams(prompt="draw", tool="codex", label=label)
            )
    finally:
        executor.shutdown(wait=True)

    assert call_result.error is None
    state_manager.on_agent_done(
        label,
        {"token_usage": 10, "duration_s": 0.5, "cached": False, "result": "chart complete"},
    )
    raw_callbacks.on_agent_done(label, {"error": None, "cached": False})

    # Process stream must stay active and healthy after IMAGE_ADDED / IMAGE_FAILED.
    assert sink.started
    assert sink.healthy
    assert not sink.aborted

    # Persisted state must store only image_key (never raw base64).
    persisted_dict = state_manager.snapshot().to_dict()
    persisted_json = json.dumps(persisted_dict, ensure_ascii=False)
    assert _PNG_B64 not in persisted_json
    assert "data:image/" not in persisted_json

    restored = WorkflowProject.from_dict(persisted_dict)
    restored.status = WorkflowStatus.COMPLETED
    restored.result = json.dumps(
        {
            "card_summary": {
                "verdict": "passed",
                "conclusion": "Generated chart successfully",
            }
        }
    )

    # Live progress status card renders image even when COT suppresses agent detail pages.
    progress_cards = WorkflowProgressRenderer(restored).render_progress_cards(restored)
    status_progress = next(
        c for c in progress_cards if c.get("_workflow_page_key", (None,))[0] == "status"
    )
    completion_card = render_completion_card(restored)
    agent_cards = _render_agent_execution_cards(restored)
    assert len(agent_cards) == 1

    # Deliver via compact_terminal=True path (handler.on_done suppresses "agent" pages).
    raw_callbacks.on_done(restored)
    handler.send_card_to_chat.assert_not_called()
    raw_delivered = handler.update_card.call_args_list[-1].args[1]
    delivered_terminal_card = (
        json.loads(raw_delivered) if isinstance(raw_delivered, str) else raw_delivered
    )
    assert require_card_json_2(delivered_terminal_card) is delivered_terminal_card

    for card in (status_progress, completion_card, agent_cards[0], delivered_terminal_card):
        card_json = json.dumps(card, ensure_ascii=False)
        assert _PNG_B64 not in card_json
        img_nodes = _tagged_nodes(card, "img")
        if expect_success:
            assert "img_v3_chart_ok" in persisted_json
            assert any(
                node.get("img_key") == "img_v3_chart_ok"
                and (node.get("alt") or {}).get("content") == "chart.png"
                for node in img_nodes
            )
        else:
            assert img_nodes == []
            assert "图片产物暂时无法展示" in card_json


def test_completion_card_preserves_images_even_on_minimal_fallback() -> None:
    from src.card.state.models import ContentBlock
    from src.workflow_engine.models import AgentProgress, AgentStatus, PhaseProgress

    project = WorkflowProject(
        name="oversized-brief-with-image",
        status=WorkflowStatus.COMPLETED,
        phases=[
            PhaseProgress(
                title="p1",
                agents=[
                    AgentProgress(
                        label="a1",
                        status=AgentStatus.DONE,
                        execution_blocks=[
                            ContentBlock(
                                kind="image",
                                block_id="image:img-min",
                                image_key="img_v3_minimal_fallback",
                                alt="fallback.png",
                                status="completed",
                            )
                        ],
                    )
                ],
            )
        ],
        result=json.dumps(
            {
                "card_summary": {
                    "verdict": "passed",
                    "conclusion": "Done " + ("c" * 300),
                    "findings": [
                        {"severity": "info", "text": f"finding-{i} " + ("x" * 200)}
                        for i in range(150)
                    ],
                }
            }
        ),
    )

    card = render_completion_card(project)
    img_nodes = _tagged_nodes(card, "img")
    assert any(node.get("img_key") == "img_v3_minimal_fallback" for node in img_nodes)
