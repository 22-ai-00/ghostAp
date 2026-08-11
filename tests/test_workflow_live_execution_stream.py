"""Workflow direct calls retain and render their complete ACP execution stream."""

from __future__ import annotations

import dataclasses
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
from src.card.events import CardEvent, CardEventType
from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card
from src.card.shared.truncation import count_tagged_nodes
from src.card.state.models import (
    CardMetadata,
    CardState,
    ContentBlock,
    HeaderState,
)
from src.card.state.reducer import reduce_card_state
from src.feishu.handlers.workflow import WorkflowHandler
from src.workflow_engine.executor import AgentExecutor
from src.workflow_engine.models import (
    AgentCallParams,
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.renderer import (
    WorkflowProgressRenderer,
    render_completion_cards,
)
from src.workflow_engine.state_manager import WorkflowStateManager


def _all_text(value: object) -> str:
    if dataclasses.is_dataclass(value):
        return _all_text(dataclasses.asdict(value))
    if isinstance(value, dict):
        return "\n".join(_all_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(_all_text(item) for item in value)
    return value if isinstance(value, str) else ""


def _header_title(card: dict) -> str:
    header = card.get("header") or {}
    title = header.get("title") or {}
    return str(title.get("content") or "")


def _card_size(card: dict) -> int:
    return len(json.dumps(card, ensure_ascii=False).encode("utf-8"))


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


class _EventSession:
    def __init__(self, events: list[ACPEvent]) -> None:
        self._events = events

    def send_prompt(self, _prompt: str, *, on_event=None, **_kwargs):
        for event in self._events:
            if on_event is not None:
                on_event(event)
        return SimpleNamespace(
            text="agent finished",
            output_tokens=0,
            stop_reason="end_turn",
        )

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        return None


class _SchemaRepairSession:
    def __init__(self) -> None:
        self._call_count = 0

    def send_prompt(self, _prompt: str, *, on_event=None, **_kwargs):
        self._call_count += 1
        if on_event is not None:
            on_event(
                ACPEvent(
                    event_type=ACPEventType.TEXT_CHUNK,
                    text=f"TURN{self._call_count}MARKER",
                )
            )
        text = '{"wrong": true}' if self._call_count == 1 else '{"ok": true}'
        return SimpleNamespace(text=text, output_tokens=0, stop_reason="end_turn")

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_executor_projects_every_acp_event_into_normal_card_blocks(tmp_path) -> None:
    events = [
        ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="INTRO_MARKER"),
        ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="REASON_MARKER"),
        ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="tool-1",
                title="rg",
                kind="search",
                status="in_progress",
                content="TOOL_INPUT_MARKER",
            ),
        ),
        ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id="tool-1",
                title="rg",
                kind="search",
                status="completed",
                content="TOOL_OUTPUT_MARKER",
            ),
        ),
        ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="FINAL_MARKER"),
    ]
    captured: list[CardEvent] = []
    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=threading.Event(),
        on_card_event=lambda label, event: captured.append(event) if label == "call-a" else None,
    )

    try:
        with patch(
            "src.agent_session.factory.create_engine_session",
            return_value=_EventSession(events),
        ):
            result = executor.execute(
                AgentCallParams(prompt="work", tool="codex", label="call-a")
            )
    finally:
        executor.shutdown(wait=True)

    assert result.error is None
    assert [event.type for event in captured] == [
        CardEventType.TEXT_STARTED,
        CardEventType.TEXT_DELTA,
        CardEventType.REASONING_STARTED,
        CardEventType.REASONING_DELTA,
        CardEventType.TEXT_DONE,
        CardEventType.REASONING_DONE,
        CardEventType.TOOL_STARTED,
        CardEventType.TOOL_DONE,
        CardEventType.TEXT_STARTED,
        CardEventType.TEXT_DELTA,
        CardEventType.TEXT_DONE,
    ]

    state = CardState()
    for event in captured:
        state = reduce_card_state(state, event)
    assert [block.kind for block in state.blocks] == [
        "text",
        "reasoning",
        "tool_call",
        "text",
    ]
    assert [block.status for block in state.blocks] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    projected_text = _all_text(state.blocks)
    for marker in (
        "INTRO_MARKER",
        "REASON_MARKER",
        "TOOL_INPUT_MARKER",
        "TOOL_OUTPUT_MARKER",
        "FINAL_MARKER",
    ):
        assert projected_text.count(marker) == 1


def test_executor_preserves_safe_full_structured_tool_payloads(tmp_path) -> None:
    input_marker = "STRUCTURED_INPUT_TAIL_MARKER"
    output_marker = "STRUCTURED_OUTPUT_TAIL_MARKER"
    result_marker = "STRUCTURED_RESULT_TAIL_MARKER"
    secret_marker = "SHOULD_NOT_LEAK_JSON_SECRET"
    nested_secret_marker = "SHOULD_NOT_LEAK_NESTED_JSON_SECRET"
    fenced_secret_marker = "SHOULD_NOT_LEAK_FENCED_JSON_SECRET"
    tool_input = "\ufeff\x1b[31m  \n" + json.dumps(
        {
            "path": "/tmp/example.py",
            "arguments": [f"argument-{index}" for index in range(80)],
            "tail": input_marker,
            "api_key": secret_marker,
            "nested": json.dumps({"password": nested_secret_marker}),
        },
        ensure_ascii=False,
    )
    tool_output = (
        "log prefix\n```json\n"
        + json.dumps(
            {
                "stdout": "x" * 5000,
                "tail": output_marker,
                "secret": fenced_secret_marker,
            },
            ensure_ascii=False,
        )
        + "\n```"
    )
    events = [
        ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="tool-json",
                title="bash",
                kind="execute",
                status="in_progress",
                content=tool_input,
            ),
        ),
        ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id="tool-json",
                title="bash",
                kind="execute",
                status="completed",
                content=tool_output,
                result={"artifact": result_marker},
            ),
        ),
    ]
    captured: list[CardEvent] = []
    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=threading.Event(),
        on_card_event=lambda _label, event: captured.append(event),
    )

    try:
        with patch(
            "src.agent_session.factory.create_engine_session",
            return_value=_EventSession(events),
        ):
            result = executor.execute(
                AgentCallParams(prompt="work", tool="codex", label="json-call")
            )
    finally:
        executor.shutdown(wait=True)

    assert result.error is None
    state = CardState()
    for event in captured:
        state = reduce_card_state(state, event)
    projected = _all_text(state.blocks)
    assert input_marker in projected
    assert output_marker in projected
    assert result_marker in projected
    assert secret_marker not in projected
    assert nested_secret_marker not in projected
    assert fenced_secret_marker not in projected
    assert "<redacted>" in projected
    assert "argument-79" in projected
    assert "x" * 3000 in projected


def test_executor_closes_stream_blocks_between_schema_repair_turns(tmp_path) -> None:
    captured: list[CardEvent] = []
    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=threading.Event(),
        on_card_event=lambda _label, event: captured.append(event),
    )
    try:
        with patch(
            "src.agent_session.factory.create_engine_session",
            return_value=_SchemaRepairSession(),
        ):
            result = executor.execute(
                AgentCallParams(
                    prompt="return json",
                    tool="codex",
                    label="schema-call",
                    schema={"ok": "boolean"},
                )
            )
    finally:
        executor.shutdown(wait=True)

    assert result.error is None
    state = CardState()
    for event in captured:
        state = reduce_card_state(state, event)
    text_blocks = [block for block in state.blocks if block.kind == "text"]
    assert len(text_blocks) == 2
    assert [block.content for block in text_blocks] == [
        "TURN1MARKER",
        "TURN2MARKER",
    ]
    assert all(block.status == "completed" for block in text_blocks)


def test_state_manager_isolates_calls_and_round_trips_execution_blocks() -> None:
    assert "execution_blocks" in AgentProgress.model_fields

    project = WorkflowProject(name="wf", status=WorkflowStatus.RUNNING)
    manager = WorkflowStateManager(project)
    first = manager.on_agent_started(
        "worker", "codex", "build", model="model-a", agent_id="A1"
    )
    second = manager.on_agent_started(
        "worker", "codex", "build", model="model-b", agent_id="A2"
    )

    assert manager.record_agent_card_event(first, CardEvent.text_started("first-text"))
    assert manager.record_agent_card_event(
        first, CardEvent.text_delta("first-text", "FIRST_CALL_MARKER")
    )
    assert manager.record_agent_card_event(second, CardEvent.text_started("second-text"))
    assert manager.record_agent_card_event(
        second, CardEvent.text_delta("second-text", "SECOND_CALL_MARKER")
    )
    manager.on_agent_done(first, {"result": "first result"})

    # Terminal direct calls are sticky: a late ACP frame cannot mutate history.
    assert not manager.record_agent_card_event(
        first, CardEvent.text_delta("first-text", "LATE_FRAME_MARKER")
    )

    snapshot = manager.snapshot()
    agents = [agent for phase in snapshot.phases for agent in phase.agents]
    assert len(agents) == 2
    assert "FIRST_CALL_MARKER" in _all_text(agents[0].execution_blocks)
    assert "LATE_FRAME_MARKER" not in _all_text(agents[0].execution_blocks)
    assert "SECOND_CALL_MARKER" in _all_text(agents[1].execution_blocks)
    assert "FIRST_CALL_MARKER" not in _all_text(agents[1].execution_blocks)

    restored = WorkflowProject.model_validate(snapshot.model_dump(mode="json"))
    restored_manager = WorkflowStateManager(restored)
    assert restored_manager.record_agent_card_event(
        second, CardEvent.text_delta("second-text", "_AFTER_RESTORE")
    )
    restored_snapshot = restored_manager.snapshot()
    restored_agents = [
        agent for phase in restored_snapshot.phases for agent in phase.agents
    ]
    assert "SECOND_CALL_MARKER_AFTER_RESTORE" in _all_text(
        restored_agents[1].execution_blocks
    )


def test_workflow_json_round_trip_restores_every_execution_block_kind() -> None:
    blocks = [
        ContentBlock(
            kind="reasoning",
            block_id="reasoning-active",
            content="REASONING_ROUND_TRIP",
            status="active",
        ),
        ContentBlock(
            kind="tool_call",
            block_id="tool-complete",
            tool_name="bash",
            tool_input="TOOL_INPUT_ROUND_TRIP",
            tool_output="TOOL_OUTPUT_ROUND_TRIP",
            status="completed",
        ),
        ContentBlock(
            kind="image",
            block_id="image-complete",
            image_key="img_round_trip",
            alt="IMAGE_ALT_ROUND_TRIP",
            status="completed",
        ),
        ContentBlock(
            kind="task_list",
            block_id="tasks",
            tasks=(
                {
                    "task_id": "task-1",
                    "name": "TASK_LIST_ROUND_TRIP",
                    "status": "in_progress",
                },
            ),
            current_task_id="task-1",
            status="active",
        ),
    ]
    original = WorkflowProject(
        name="round-trip",
        status=WorkflowStatus.RUNNING,
        phases=[
            PhaseProgress(
                title="build",
                agents=[
                    AgentProgress(
                        label="round-trip-agent",
                        status=AgentStatus.RUNNING,
                        execution_blocks=blocks,
                    )
                ],
            )
        ],
    )

    persisted = json.loads(json.dumps(original.to_dict(), ensure_ascii=False))
    restored = WorkflowProject.from_dict(persisted)
    manager = WorkflowStateManager(restored)
    assert manager.record_agent_card_event(
        "round-trip-agent",
        CardEvent(
            type=CardEventType.REASONING_DELTA,
            payload={"block_id": "reasoning-active", "text": "_CONTINUED"},
        ),
    )

    restored_blocks = manager.snapshot().phases[0].agents[0].execution_blocks
    assert [block.kind for block in restored_blocks] == [
        "reasoning",
        "tool_call",
        "image",
        "task_list",
    ]
    rendered = _all_text(restored_blocks)
    for marker in (
        "REASONING_ROUND_TRIP_CONTINUED",
        "TOOL_INPUT_ROUND_TRIP",
        "TOOL_OUTPUT_ROUND_TRIP",
        "IMAGE_ALT_ROUND_TRIP",
        "TASK_LIST_ROUND_TRIP",
    ):
        assert marker in rendered


def test_state_manager_does_not_drop_old_blocks_at_programming_retention_limit() -> None:
    project = WorkflowProject(name="long-wf", status=WorkflowStatus.RUNNING)
    manager = WorkflowStateManager(project)
    label = manager.on_agent_started("long", "codex", "build", agent_id="A1")

    markers = [f"BLOCK_{index:03d}" for index in range(110)]
    for index, marker in enumerate(markers):
        block_id = f"text-{index}"
        assert manager.record_agent_card_event(label, CardEvent.text_started(block_id))
        assert manager.record_agent_card_event(
            label, CardEvent.text_delta(block_id, marker)
        )
        assert manager.record_agent_card_event(label, CardEvent.text_done(block_id))

    agent = manager.snapshot().phases[0].agents[0]
    rendered = _all_text(agent.execution_blocks)
    assert len(agent.execution_blocks) == len(markers)
    assert all(rendered.count(marker) == 1 for marker in markers)


def test_full_execution_mode_uses_individual_programming_card_sections() -> None:
    state = CardState(
        metadata=CardMetadata(
            project_name="wf",
            tool_name="codex",
            model_name="model-a",
            programming_text_sections=True,
            full_execution_blocks=True,
        ),
        header=HeaderState(title="call"),
        blocks=(
            ContentBlock(
                kind="text",
                block_id="text-1",
                content="TEXT_SECTION_MARKER",
                status="completed",
            ),
            ContentBlock(
                kind="reasoning",
                block_id="reason-1",
                content="REASON_SECTION_MARKER",
                status="completed",
            ),
            ContentBlock(
                kind="tool_call",
                block_id="tool-1",
                tool_name="rg",
                tool_input="INPUT_SECTION_MARKER",
                content="UPDATE_SECTION_MARKER",
                tool_output="OUTPUT_SECTION_MARKER",
                status="completed",
            ),
            ContentBlock(
                kind="tool_call",
                block_id="tool-live",
                tool_name="bash",
                tool_input="LIVE_COMMAND_MARKER",
                content="LIVE_TOOL_DELTA_MARKER",
                status="active",
                is_latest_active=True,
            ),
            ContentBlock(
                kind="tool_call",
                block_id="tool-empty",
                tool_name="EMPTY_TOOL_MUST_NOT_RENDER",
                status="active",
            ),
            ContentBlock(
                kind="text",
                block_id="text-2",
                content="CURRENT_SECTION_MARKER",
                status="active",
            ),
        ),
    )

    cards = [page.to_feishu_json() for page in render_card(state, RenderBudget())]
    rendered = _all_text(cards)

    for marker in (
        "TEXT_SECTION_MARKER",
        "REASON_SECTION_MARKER",
        "INPUT_SECTION_MARKER",
        "UPDATE_SECTION_MARKER",
        "OUTPUT_SECTION_MARKER",
        "LIVE_TOOL_DELTA_MARKER",
        "CURRENT_SECTION_MARKER",
    ):
        assert rendered.count(marker) == 1
    assert "LIVE_COMMAND_MARKER" in rendered
    assert "EMPTY_TOOL_MUST_NOT_RENDER" not in rendered
    assert rendered.index("TEXT_SECTION_MARKER") < rendered.index("REASON_SECTION_MARKER")
    assert rendered.index("REASON_SECTION_MARKER") < rendered.index("INPUT_SECTION_MARKER")
    assert rendered.index("INPUT_SECTION_MARKER") < rendered.index("UPDATE_SECTION_MARKER")
    assert rendered.index("UPDATE_SECTION_MARKER") < rendered.index("OUTPUT_SECTION_MARKER")
    assert rendered.index("OUTPUT_SECTION_MARKER") < rendered.index("LIVE_COMMAND_MARKER")
    assert rendered.index("LIVE_TOOL_DELTA_MARKER") < rendered.index("CURRENT_SECTION_MARKER")

    panels = _tagged_nodes(cards, "collapsible_panel")
    assert len(panels) >= 5
    live_tool_panel = next(
        panel for panel in panels if "LIVE_TOOL_DELTA_MARKER" in _all_text(panel)
    )
    live_tool_text = _all_text(live_tool_panel)
    assert "**过程**" in live_tool_text
    assert "**结果**" not in live_tool_text
    for panel in panels:
        header = panel["header"]
        assert header["icon"] == {
            "tag": "standard_icon",
            "token": "down_outlined",
            "color": "grey",
        }
        assert header["icon_position"] == "right"
        assert header["icon_expanded_angle"] == -180


def test_full_execution_mode_paginates_tool_output_without_tail_truncation() -> None:
    head = "LONG_TOOL_OUTPUT_HEAD_MARKER"
    middle = "LONG_TOOL_OUTPUT_MIDDLE_MARKER"
    tail = "LONG_TOOL_OUTPUT_TAIL_MARKER"
    output = head + ("x" * 28000) + middle + ("y" * 28000) + tail
    state = CardState(
        metadata=CardMetadata(
            project_name="wf",
            tool_name="codex",
            programming_text_sections=True,
            full_execution_blocks=True,
        ),
        header=HeaderState(title="call"),
        blocks=(
            ContentBlock(
                kind="tool_call",
                block_id="long-tool",
                tool_name="bash",
                tool_input="printf long-output",
                tool_output=output,
                status="completed",
            ),
        ),
    )

    cards = [page.to_feishu_json() for page in render_card(state, RenderBudget())]
    rendered = _all_text(cards)

    assert len(cards) >= 3
    assert rendered.count(head) == 1
    assert rendered.count(middle) == 1
    assert rendered.count(tail) == 1
    assert all(_card_size(card) <= 27 * 1024 for card in cards)
    assert all(count_tagged_nodes(card) <= 180 for card in cards)


def test_segmented_text_panel_node_budget_includes_collapse_icon() -> None:
    blocks = tuple(
        ContentBlock(
            kind="text",
            block_id=f"text-{index}",
            content=f"section-{index}",
            status="completed",
        )
        for index in range(70)
    )
    state = CardState(
        metadata=CardMetadata(
            project_name="wf",
            programming_text_sections=True,
            full_execution_blocks=True,
        ),
        header=HeaderState(title="many sections"),
        blocks=blocks,
    )

    cards = [page.to_feishu_json() for page in render_card(state, RenderBudget())]

    assert len(cards) >= 2
    assert all(count_tagged_nodes(card) <= 180 for card in cards)
    assert all(_card_size(card) <= 27 * 1024 for card in cards)


def test_workflow_handler_applies_final_payload_audit_guard() -> None:
    card = WorkflowHandler._build_workflow_card_from_renderer_data(
        {
            "header": {
                "title": {"tag": "plain_text", "content": "Workflow"},
                "template": "blue",
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": (
                        "contact dev@example.com\n"
                        "![local image](file:///tmp/private.png)\n"
                        "![remote image](https://example.com/private.png)"
                    ),
                }
            ],
        }
    )

    rendered = _all_text(card)
    assert "dev@example.com" not in rendered
    assert "[redacted:email]" in rendered
    assert "file:///tmp/private.png" not in rendered
    assert "https://example.com/private.png" not in rendered
    assert rendered.count("（图片引用已移除）") == 2
    assert _card_size(card) <= 27 * 1024
    assert count_tagged_nodes(card) <= 180


def test_workflow_text_and_reasoning_streams_are_sanitized_before_persistence(
    tmp_path,
) -> None:
    answer_marker = "VISIBLE_ANSWER_CONTENT"
    reasoning_marker = "VISIBLE_REASONING_CONTENT"
    answer_secret = "supersecretvalue"
    reasoning_secret = "reasoningsecretvalue"
    events = [
        ACPEvent(
            event_type=ACPEventType.TEXT_CHUNK,
            text=(
                f"{answer_marker} API_KEY={answer_secret} "
                "\ufeff\x1b[31mcolored answer\x1b[0m\u202e"
            ),
        ),
        ACPEvent(
            event_type=ACPEventType.THOUGHT_CHUNK,
            text=(
                f"{reasoning_marker} API_\x1b[32mKEY={reasoning_secret}"
                "\x1b[0m\u2066"
            ),
        ),
    ]
    project = WorkflowProject(
        name="safe-stream",
        status=WorkflowStatus.RUNNING,
    )
    manager = WorkflowStateManager(project)
    label = manager.on_agent_started(
        "safe-call",
        "codex",
        "build",
        model="model-a",
        agent_id="A1",
    )
    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=threading.Event(),
        on_card_event=lambda event_label, event: manager.record_agent_card_event(
            event_label,
            event,
        ),
    )

    try:
        with patch(
            "src.agent_session.factory.create_engine_session",
            return_value=_EventSession(events),
        ):
            result = executor.execute(
                AgentCallParams(prompt="work", tool="codex", label=label)
            )
    finally:
        executor.shutdown(wait=True)

    assert result.error is None
    snapshot = manager.snapshot()
    persisted_text = _all_text(snapshot.phases[0].agents[0].execution_blocks)
    rendered_cards = WorkflowProgressRenderer(snapshot).render_progress_cards(
        snapshot
    )
    wire_cards = [
        WorkflowHandler._build_workflow_card_from_renderer_data(card)
        for card in rendered_cards
    ]
    wire_text = _all_text(wire_cards)

    for projected in (persisted_text, wire_text):
        assert answer_marker in projected
        assert reasoning_marker in projected
        assert "colored answer" in projected
        assert answer_secret not in projected
        assert reasoning_secret not in projected
        assert "<redacted>" in projected
        assert "\x1b" not in projected
        assert "\ufeff" not in projected
        assert "\u202e" not in projected
        assert "\u2066" not in projected


def test_workflow_text_start_metadata_is_sanitized_before_persistence() -> None:
    project = WorkflowProject(name="safe-source", status=WorkflowStatus.RUNNING)
    manager = WorkflowStateManager(project)
    label = manager.on_agent_started(
        "source-call",
        "codex",
        "build",
        agent_id="A1",
    )

    assert manager.record_agent_card_event(
        label,
        CardEvent(
            type=CardEventType.TEXT_STARTED,
            payload={
                "block_id": "source-text",
                "source_kind": "subagent",
                "source_sequence": "1\ufeff",
                "source_label": "worker API_\x1b[31mKEY=metadata-secret\x1b[0m",
                "source_ref": "peer\u202e",
            },
        ),
    )

    block = manager.snapshot().phases[0].agents[0].execution_blocks[0]
    persisted = _all_text(block)
    assert "worker" in persisted
    assert "metadata-secret" not in persisted
    assert "<redacted>" in persisted
    assert "\x1b" not in persisted
    assert "\ufeff" not in persisted
    assert "\u202e" not in persisted


def test_workflow_renderer_adds_lossless_per_call_execution_pages() -> None:
    markers = [f"STREAM_MARKER_{index:03d}" for index in range(72)]
    long_markdown = "```python\n" + "\n".join(
        f"# {marker} " + ("x" * 520) for marker in markers
    ) + "\n```"
    agent = AgentProgress(
        label="implementation",
        agent_id="A1",
        tool="codex",
        model="model-a",
        task_summary="implement the change",
        status=AgentStatus.RUNNING,
        call_index=0,
        execution_blocks=[
            ContentBlock(
                kind="text",
                block_id="long-text",
                content=long_markdown,
                status="active",
            ),
            ContentBlock(
                kind="reasoning",
                block_id="reason",
                content="WORKFLOW_REASON_MARKER",
                status="completed",
            ),
            ContentBlock(
                kind="tool_call",
                block_id="tool",
                tool_name="apply_patch",
                tool_input="WORKFLOW_TOOL_INPUT",
                tool_output="WORKFLOW_TOOL_OUTPUT",
                status="completed",
            ),
        ],
    )
    project = WorkflowProject(
        name="live-stream",
        requirement="show full execution",
        status=WorkflowStatus.RUNNING,
        phases=[PhaseProgress(title="build", agents=[agent])],
    )

    cards = WorkflowProgressRenderer(project).render_progress_cards(project)
    execution_cards = [
        card for card in cards if "A1" in _header_title(card) and "#1" in _header_title(card)
    ]

    assert len(execution_cards) >= 2
    assert all(_card_size(card) <= 27 * 1024 for card in cards)
    assert all(count_tagged_nodes(card) <= 180 for card in cards)
    execution_text = _all_text(execution_cards)
    for marker in markers:
        assert execution_text.count(marker) == 1
    positions = [execution_text.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert execution_text.count("WORKFLOW_REASON_MARKER") == 1
    assert execution_text.count("WORKFLOW_TOOL_INPUT") == 1
    assert execution_text.count("WORKFLOW_TOOL_OUTPUT") == 1


def test_terminal_cards_keep_execution_pages_before_result_ledger() -> None:
    agent = AgentProgress(
        label="review",
        agent_id="A3",
        tool="claude",
        model="model-r",
        status=AgentStatus.DONE,
        result="TERMINALLEDGERMARKER",
        call_index=2,
        execution_blocks=[
            ContentBlock(
                kind="text",
                block_id="answer",
                content="TERMINAL_STREAM_MARKER",
                status="completed",
            )
        ],
    )
    project = WorkflowProject(
        name="terminal-stream",
        status=WorkflowStatus.COMPLETED,
        phases=[PhaseProgress(title="review", agents=[agent])],
        result="done",
    )

    cards = render_completion_cards(project)
    titles = [_header_title(card) for card in cards]
    stream_page = next(
        index for index, title in enumerate(titles) if "A3" in title and "#3" in title
    )
    ledger_page = next(index for index, title in enumerate(titles) if "结果账本" in title)

    assert 0 < stream_page < ledger_page
    assert _all_text(cards[stream_page]).count("TERMINAL_STREAM_MARKER") == 1
    assert _all_text(cards[ledger_page:]).count("TERMINALLEDGERMARKER") == 1


def test_workflow_renderer_assigns_stable_semantic_page_keys() -> None:
    first = AgentProgress(
        label="first",
        agent_id="A1",
        tool="codex",
        status=AgentStatus.DONE,
        result="FIRST_RESULT",
        call_index=0,
        execution_blocks=[
            ContentBlock(
                kind="text",
                block_id="first-text",
                content="FIRST_STREAM",
                status="completed",
            )
        ],
    )
    second = AgentProgress(
        label="second",
        agent_id="A2",
        tool="claude",
        status=AgentStatus.RUNNING,
        call_index=1,
        execution_blocks=[
            ContentBlock(
                kind="text",
                block_id="second-text",
                content="SECOND_STREAM",
                status="active",
            )
        ],
    )
    project = WorkflowProject(
        name="page-keys",
        status=WorkflowStatus.RUNNING,
        phases=[PhaseProgress(title="build", agents=[first, second])],
    )

    cards = WorkflowProgressRenderer(project).render_progress_cards(project)
    keys = [card["_workflow_page_key"] for card in cards]
    agent_keys = [key for key in keys if key[0] == "agent"]

    assert keys[0] == ("status", -1, 0)
    assert len(agent_keys) == 2
    assert len(set(agent_keys)) == 2
    assert all(key[2] == 0 for key in agent_keys)
    assert all(key[0] != "ledger" for key in keys)


def test_workflow_renderer_disambiguates_default_call_indexes_across_snapshots() -> None:
    later_call = AgentProgress(
        label="repeated-call",
        agent_id="A1",
        tool="codex",
        status=AgentStatus.RUNNING,
        started_at=200.0,
        execution_blocks=[
            ContentBlock(
                kind="text",
                block_id="later-text",
                content="LATER_STREAM",
                status="active",
            )
        ],
    )
    first_snapshot = WorkflowProject(
        name="duplicate-default-index",
        status=WorkflowStatus.RUNNING,
        phases=[PhaseProgress(title="build", agents=[later_call])],
    )
    first_cards = WorkflowProgressRenderer(first_snapshot).render_progress_cards(
        first_snapshot
    )
    later_key_before = next(
        card["_workflow_page_key"]
        for card in first_cards
        if "LATER_STREAM" in _all_text(card)
    )

    earlier_call = AgentProgress(
        label="repeated-call",
        agent_id="A1",
        tool="codex",
        status=AgentStatus.RUNNING,
        started_at=100.0,
        execution_blocks=[
            ContentBlock(
                kind="text",
                block_id="earlier-text",
                content="EARLIER_STREAM",
                status="active",
            )
        ],
    )
    expanded_snapshot = WorkflowProject(
        name="duplicate-default-index",
        status=WorkflowStatus.RUNNING,
        phases=[PhaseProgress(title="build", agents=[earlier_call, later_call])],
    )
    expanded_cards = WorkflowProgressRenderer(expanded_snapshot).render_progress_cards(
        expanded_snapshot
    )
    agent_cards = [
        card
        for card in expanded_cards
        if "EARLIER_STREAM" in _all_text(card) or "LATER_STREAM" in _all_text(card)
    ]
    keys_by_stream = {
        "earlier" if "EARLIER_STREAM" in _all_text(card) else "later": card[
            "_workflow_page_key"
        ]
        for card in agent_cards
    }

    assert keys_by_stream["later"] == later_key_before
    assert keys_by_stream["earlier"] != keys_by_stream["later"]
    assert all("#1" in _header_title(card) for card in agent_cards)

    restored = WorkflowProject.model_validate_json(expanded_snapshot.model_dump_json())
    restored_cards = WorkflowProgressRenderer(restored).render_progress_cards(restored)
    restored_keys_by_stream = {
        "earlier" if "EARLIER_STREAM" in _all_text(card) else "later": card[
            "_workflow_page_key"
        ]
        for card in restored_cards
        if "EARLIER_STREAM" in _all_text(card) or "LATER_STREAM" in _all_text(card)
    }
    assert restored_keys_by_stream == keys_by_stream
