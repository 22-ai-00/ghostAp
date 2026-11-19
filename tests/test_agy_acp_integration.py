"""Regression contracts for the first-class AGY native ACP integration."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from acp.schema import (
    AgentMessageChunk,
    NewSessionResponse,
    PromptResponse,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    SetSessionConfigOptionResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)
from lark_oapi.core.model.base_response import BaseResponse
from lark_oapi.core.model.raw_response import RawResponse

import src.acp.helper as helper_mod
import src.acp.providers as providers_mod
import src.feishu.handlers.programming as programming_mod
from src.acp.agy_selection import compose_agy_model_selection, split_agy_model_selection
from src.acp.client import GhostAPClient
from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
from src.acp.session import ACPSession
from src.acp.sync_adapter import SyncACPSession
from src.agent.intent_recognizer import IntentRecognizer, IntentType
from src.agent_session.backend_resolver import is_cli_backend
from src.card.delivery.engine import CardDelivery
from src.card.programming_adapter import ProgrammingCardSession, build_programming_metadata
from src.card.render.model_cascade import resolve_model_cascade
from src.card.session import CardSession
from src.card.session.config import SessionConfig
from src.card.shared import build_mode_buttons
from src.card.state.models import TextBlock
from src.feishu.cot import FeishuCOTAPIClient, FeishuCOTStream
from src.feishu.dispatcher import MessageDispatcher
from src.feishu.emoji import EmojiReaction, EmojiType
from src.feishu.product_catalog import PUBLIC_ACTIONS
from src.mode import InteractionMode, ModeManager
from src.project.context import ProjectContext
from src.project.unified_context import ContextSourceMode
from src.utils.engine_identity import resolve_engine_identity
from src.utils.env import build_clean_env
from src.workflow_engine.constants import TOOL_DESCRIPTIONS
from src.workflow_engine.tool_registry import get_available_tools, invalidate_cache


def teardown_function() -> None:
    providers_mod._reset_providers_for_testing()


def test_agy_provider_uses_native_acp_stdio_without_cli_fallback() -> None:
    providers_mod._reset_providers_for_testing()
    providers = providers_mod.get_providers()

    assert "agy" in providers
    assert providers["agy"].get_serve_command() == ("agy", ["acp"])
    assert providers["agy"].get_serve_command("gemini-4-hydrogen/xhigh") == (
        "agy",
        ["acp"],
    )
    assert providers["agy"].get_fallback_command("gemini-4-hydrogen/xhigh") is None
    assert is_cli_backend("agy") is False


def test_agy_availability_probe_requires_acp_stdio_subcommand() -> None:
    providers_mod._reset_providers_for_testing()
    completed = Mock(
        stdout="Usage: agy acp [flags]\nStart the Agent Client Protocol (ACP) stdio server",
        stderr="",
    )
    with patch.object(providers_mod.subprocess, "run", return_value=completed) as run:
        assert providers_mod.get_providers()["agy"].check_availability() is True

    assert run.call_args.args[0] == ["agy", "acp", "--help"]

    providers_mod._reset_providers_for_testing()
    missing_stdio = Mock(stdout="Usage: agy acp [flags]\n", stderr="")
    with patch.object(providers_mod.subprocess, "run", return_value=missing_stdio):
        assert providers_mod.get_providers()["agy"].check_availability() is False


def test_build_clean_env_strips_parent_antigravity_session_guards() -> None:
    env = build_clean_env(
        {
            "HOME": "/tmp/ghostap-test-home",
            "CLAUDECODE": "1",
            "ANTIGRAVITY_AGENT": "1",
            "ANTIGRAVITY_LS_ADDRESS": "localhost:27493",
            "ANTIGRAVITY_CSRF_TOKEN": "secret-token",
            "ANTIGRAVITY_CONVERSATION_ID": "conv-1",
            "ANTIGRAVITY_TRAJECTORY_ID": "traj-1",
            "ANTIGRAVITY_SOURCE_METADATA": '{"tool":{}}',
            "ANTIGRAVITY_APP_DATA_DIR": "/tmp/custom-agy-app-data",
        }
    )

    for key in (
        "CLAUDECODE",
        "ANTIGRAVITY_AGENT",
        "ANTIGRAVITY_LS_ADDRESS",
        "ANTIGRAVITY_CSRF_TOKEN",
        "ANTIGRAVITY_CONVERSATION_ID",
        "ANTIGRAVITY_TRAJECTORY_ID",
        "ANTIGRAVITY_SOURCE_METADATA",
    ):
        assert key not in env
    assert env["ANTIGRAVITY_APP_DATA_DIR"] == "/tmp/custom-agy-app-data"


def test_agy_mode_and_project_session_round_trip() -> None:
    manager = ModeManager()
    manager.enter_agy_mode("chat", project_id="project")

    assert manager.is_agy_mode("chat", project_id="project") is True
    assert manager.is_programming_mode("chat", project_id="project") is True
    assert manager.get_mode_display_name("chat", project_id="project") == "🪐 AGY 编程模式"

    context = ProjectContext("project", "Project", "/tmp/project")
    context.set_programming_mode("agy", True, session_id="session-agy", query_count=2)
    context.update_programming_snapshot("agy", "continue", 3)

    restored = ProjectContext.from_snapshot(context.to_snapshot())

    assert restored.agy_mode is True
    assert restored.agy_session_snapshot is not None
    assert restored.agy_session_snapshot.session_id == "session-agy"
    assert restored.agy_session_snapshot.query_count == 3
    assert restored.agy_session_snapshot.last_query == "continue"


def test_agy_commands_catalog_and_shared_exit_contract() -> None:
    recognizer = IntentRecognizer()

    enter = recognizer._quick_match("/agy")
    alias = recognizer._quick_match("/enter_agy")
    info = recognizer._quick_match("/agy_info")
    keyword = recognizer._quick_match("进入agy")

    assert enter is not None and enter.primary_intent is IntentType.ENTER_AGY
    assert alias is not None and alias.primary_intent is IntentType.ENTER_AGY
    assert info is not None and info.primary_intent is IntentType.AGY_MESSAGE
    assert info.primary_data["command"] == "info"
    assert keyword is not None and keyword.primary_intent is IntentType.ENTER_AGY
    assert ContextSourceMode.AGY.value == "agy"
    assert any(
        action.command == "/agy"
        and action.programming_mode_id == "agy"
        and action.enters_programming_mode
        and action.slash_discoverable
        for action in PUBLIC_ACTIONS
    )

    exit_action = next(action for action in PUBLIC_ACTIONS if action.command == "/exit")
    integration_doc = (
        Path(__file__).resolve().parents[1] / "docs" / "agy_acp_integration.md"
    ).read_text(encoding="utf-8")

    assert not hasattr(IntentType, "EXIT_AGY")
    assert "/exit_agy" not in exit_action.aliases
    assert "/exit_agy" not in integration_doc


def test_agy_model_discovery_expands_per_model_thought_levels_and_omits_unsupported() -> None:
    discover = getattr(helper_mod, "discover_agy_model_options", None)
    assert callable(discover)

    def option(value: str, name: str) -> SimpleNamespace:
        return SimpleNamespace(value=value, name=name, description=None)

    def response(
        current_model: str,
        current_effort: str | None,
        efforts: tuple[str, ...],
    ) -> SimpleNamespace:
        config_options = [
            SimpleNamespace(
                id="model",
                category="model",
                current_value=current_model,
                options=[
                    option("gemini-4-carbon", "Gemini 4 Carbon"),
                    option("gemini-4-hydrogen", "Gemini 4 Hydrogen"),
                    option("gemini-pro-orchestrator", "Orchestrator (Carbon)"),
                ],
            )
        ]
        if efforts:
            config_options.append(
                SimpleNamespace(
                    id="thought_level",
                    category="thought_level",
                    current_value=current_effort,
                    options=[option(effort, effort.title()) for effort in efforts],
                )
            )
        return SimpleNamespace(
            session_id="session-agy",
            config_options=config_options,
        )

    initial = response("gemini-4-carbon", "medium", ("low", "medium", "high"))

    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def set_config_option(self, *, config_id: str, session_id: str, value: str):
            self.calls.append((config_id, session_id, value))
            if value == "gemini-4-hydrogen":
                return response(value, "medium", ("low", "medium", "high", "xhigh"))
            return response(value, None, ())

    connection = Connection()
    models = asyncio.run(discover(connection, initial))

    assert [model.name for model in models] == [
        "gemini-4-carbon",
        "gemini-4-hydrogen",
        "gemini-pro-orchestrator",
    ]
    assert [variant.name for variant in models[0].selection_variants] == [
        "gemini-4-carbon/low",
        "gemini-4-carbon/medium",
        "gemini-4-carbon/high",
    ]
    assert [variant.name for variant in models[1].selection_variants] == [
        "gemini-4-hydrogen/low",
        "gemini-4-hydrogen/medium",
        "gemini-4-hydrogen/high",
        "gemini-4-hydrogen/xhigh",
    ]
    assert [variant.name for variant in models[2].selection_variants] == [
        "gemini-pro-orchestrator",
    ]
    assert models[0].is_default is True
    assert connection.calls == [
        ("model", "session-agy", "gemini-4-hydrogen"),
        ("model", "session-agy", "gemini-pro-orchestrator"),
    ]

    hydrogen_state = resolve_model_cascade(models, selected_model="gemini-4-hydrogen", selected_effort="xhigh")
    assert hydrogen_state.efforts == ("low", "medium", "high", "xhigh")
    assert hydrogen_state.selection == "gemini-4-hydrogen/xhigh"

    orchestrator_state = resolve_model_cascade(models, selected_model="gemini-pro-orchestrator")
    assert orchestrator_state.efforts == ()
    assert orchestrator_state.selected_effort is None
    assert orchestrator_state.selection == "gemini-pro-orchestrator"


def test_agy_runtime_selection_applies_model_and_thought_level_and_fails_closed() -> None:
    class Session:
        def __init__(self, *, reject: bool = False) -> None:
            self.calls: list[tuple[str, str]] = []
            self.reject = reject

        async def set_config_option(self, config_id: str, value: str) -> bool:
            self.calls.append((config_id, value))
            return not self.reject

    wrapper = SyncACPSession(
        "agy",
        "/tmp",
        agent_cmd="agy",
        agent_args=["acp"],
    )
    session = Session()
    wrapper._acp_session = session  # type: ignore[assignment]

    assert asyncio.run(
        wrapper._apply_agy_selection(compose_agy_model_selection("gemini-4-hydrogen", "xhigh"))  # type: ignore[attr-defined]
    ) is True
    assert session.calls == [
        ("model", "gemini-4-hydrogen"),
        ("thought_level", "xhigh"),
    ]

    bare_session = Session()
    wrapper._acp_session = bare_session  # type: ignore[assignment]
    assert asyncio.run(wrapper._apply_agy_selection("gemini-pro-orchestrator")) is True  # type: ignore[attr-defined]
    assert bare_session.calls == [("model", "gemini-pro-orchestrator")]

    reject_session = Session(reject=True)
    wrapper._acp_session = reject_session  # type: ignore[assignment]
    assert asyncio.run(wrapper._apply_agy_selection("gemini-4-hydrogen/xhigh")) is False  # type: ignore[attr-defined]
    assert asyncio.run(wrapper._apply_agy_selection("gemini-4-hydrogen/bogus")) is False  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "value",
    ["", "   ", "gemini-4-hydrogen/", "/high", "gemini-4-hydrogen/bogus", "a/b/high"],
)
def test_agy_selection_rejects_ambiguous_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="invalid AGY model selection"):
        split_agy_model_selection(value)


def test_agy_handler_dispatch_identity_card_and_workflow_pool() -> None:
    handler_type = getattr(programming_mod, "AGYModeHandler", None)
    assert handler_type is not None
    assert handler_type.interaction_mode is InteractionMode.AGY
    assert handler_type.mode_key == "agy"
    assert "agy" in MessageDispatcher._PROGRAMMING_MODES

    identity = resolve_engine_identity(
        mode=InteractionMode.AGY,
        acp_tool_name="agy",
        acp_model_name="gemini-4-hydrogen/xhigh",
    )
    assert identity.engine_name == "AGY"
    assert identity.agent_type == "agy"
    assert identity.transport == "acp"
    assert identity.model_name == "gemini-4-hydrogen/xhigh"

    metadata = build_programming_metadata("agy")
    assert (metadata.mode_emoji, metadata.mode_name, metadata.tool_name) == (
        "🪐",
        "AGY",
        "agy",
    )
    actions = [
        button["behaviors"][0]["value"]["action"]
        for button in build_mode_buttons(InteractionMode.AGY, "project")
    ]
    assert actions == ["exit", "switch_project"]

    invalidate_cache()
    assert TOOL_DESCRIPTIONS["agy"] == "Google Antigravity 原生 ACP 编程"
    assert get_available_tools()["agy"] == "Google Antigravity 原生 ACP 编程"


def test_agy_sync_session_enables_background_tool_continuation_only_for_agy() -> None:
    agy_session = SyncACPSession("agy", "/tmp", agent_cmd="agy", agent_args=["acp"])
    codex_session = SyncACPSession("codex", "/tmp", agent_cmd="codex", agent_args=[])
    claude_session = SyncACPSession("claude", "/tmp", agent_cmd="claude", agent_args=[])
    traex_session = SyncACPSession("traex", "/tmp", agent_cmd="traex", agent_args=[])

    assert agy_session.supports_background_tool_continuation is True
    assert codex_session.supports_background_tool_continuation is False
    assert claude_session.supports_background_tool_continuation is False
    assert traex_session.supports_background_tool_continuation is False


class _AgyCardClient:
    def __init__(self) -> None:
        self.created = 0
        self.updated_cards: list[dict] = []

    def create_card(
        self,
        chat_id,
        card_json,
        *,
        reply_to=None,
        reply_in_thread=None,
        idempotency_key=None,
    ):
        del chat_id, card_json, reply_to, reply_in_thread, idempotency_key
        self.created += 1
        return f"msg-{self.created}", f"card-{self.created}"

    def update_card(self, card_id, card_json, *, sequence=0) -> None:
        del card_id, sequence
        self.updated_cards.append(card_json)

    def update_element(self, card_id, element_id, content, *, sequence=0) -> None:
        del card_id, element_id, content, sequence
        return None


class _AgyCOTRecordingClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: list[object] = []

    def request(self, request: object) -> BaseResponse:
        with self._lock:
            self.requests.append(request)
        uri = getattr(request, "uri", "")
        body = getattr(request, "body", None)
        data: dict[str, str] = {}
        if uri == "/open-apis/im/v1/message_cot" and isinstance(body, dict) and "receive_id" in body:
            data = {"cot_id": "cot_agy_bg", "message_id": "om_agy_cot_bg"}
        response = BaseResponse()
        response.code = 0
        response.raw = RawResponse()
        response.raw.status_code = 200
        response.raw.content = json.dumps({"code": 0, "msg": "success", "data": data}).encode("utf-8")
        return response


def test_agy_cot_and_card_complete_cleanly_across_continuation_turn() -> None:
    cot_client = _AgyCOTRecordingClient()
    sink = FeishuCOTStream(
        FeishuCOTAPIClient(cot_client, timeout_seconds=0.2),
        chat_id="oc_agy_bg",
        origin_message_id="om_agy_bg_origin",
        input_text="分析仓库",
        detail="brief",
        flush_interval=0.005,
        request_timeout=0.2,
    )
    card_client = _AgyCardClient()
    metadata = build_programming_metadata("agy", model_name="gemini-pro-orchestrator")
    card_session = CardSession(
        chat_id="oc_agy_bg",
        config=SessionConfig(metadata=metadata, reply_to="om_agy_bg_origin", sync_delivery=True),
        delivery=CardDelivery(card_client),
        session_id="agy-bg-continuation-test",
    )
    programming = ProgrammingCardSession(card_session, base_metadata=metadata, process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True

        # Turn 1 starts a background schedule tool and yields end_turn before TOOL_CALL_DONE arrives.
        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_START,
                tool_call=ToolCallInfo(
                    id="call_sched_bg",
                    title="Run schedule",
                    kind="other",
                    status="in_progress",
                ),
            )
        )
        programming.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="正在等待调度完成"))
        programming.begin_continuation_turn()

        # Turn 2 receives the deferred TOOL_CALL_DONE and final conclusion text.
        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_DONE,
                tool_call=ToolCallInfo(
                    id="call_sched_bg",
                    title="Run schedule",
                    kind="other",
                    status="completed",
                ),
            )
        )
        programming.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="完整分析结论"))
        programming._flush_now()
        assert sink.healthy is True
        programming.finish()

        assert sink._failed is False
        assert programming._process_cutover_announced is False
        assert card_session.state is not None
        assert card_session.state.terminal == "completed"
        assert card_session.state.footer.warning_banner is None
        assert any(
            isinstance(block, TextBlock) and block.block_id == "_final_answer" and block.content == "完整分析结论"
            for block in card_session.state.blocks
        )
        cot_contents = [
            json.loads(ev["content"])
            for req in cot_client.requests
            if isinstance(getattr(req, "body", None), dict)
            for ev in (req.body.get("events") or [])
            if isinstance(ev, dict) and isinstance(ev.get("content"), str)
        ]
        assert any(c.get("status") == "done" for c in cot_contents)
        assert not any(c.get("status") == "error" for c in cot_contents)
    finally:
        programming.abort()


def test_acp_session_holds_open_across_agy_swarm_reactive_turns(tmp_path: Path) -> None:
    session_id = "sess-agy-swarm-e2e"

    async def scenario():
        acp_session = ACPSession("agy", ["acp"], str(tmp_path))
        acp_session._swarm_settle_scale = 0.02
        acp_session._session_id = session_id
        acp_session._state.session_id = session_id
        acp_session._state.is_active = True
        client = GhostAPClient(on_event=acp_session._dispatch_event, root_dir=str(tmp_path))
        acp_session._client = client
        acp_session._bind_session_info_callback()

        turn1_finished = asyncio.Event()
        reactive_task: asyncio.Task[None] | None = None

        async def emit_reactive_turns() -> None:
            await turn1_finished.wait()
            await asyncio.sleep(0.05)
            # Turn 2: board wakeup -> swarm_wait
            await client.session_update(
                session_id,
                ToolCallStart(
                    session_update="tool_call",
                    tool_call_id="c_read",
                    title="read_board",
                    kind="other",
                    status="in_progress",
                    raw_input={"since": 0},
                ),
            )
            await client.session_update(
                session_id,
                ToolCallProgress(
                    session_update="tool_call_update",
                    tool_call_id="c_read",
                    status="completed",
                    raw_output="worker-1 started",
                ),
            )
            await client.session_update(
                session_id,
                ToolCallStart(
                    session_update="tool_call",
                    tool_call_id="c_wait",
                    title="swarm_wait",
                    kind="other",
                    status="in_progress",
                    raw_input={"since": 1},
                ),
            )
            await client.session_update(
                session_id,
                ToolCallProgress(
                    session_update="tool_call_update",
                    tool_call_id="c_wait",
                    status="completed",
                    raw_output={"status": "woken"},
                ),
            )
            await asyncio.sleep(0.05)
            # Turn 3: [done] post + final answer
            await client.session_update(
                session_id,
                ToolCallStart(
                    session_update="tool_call",
                    tool_call_id="c_done",
                    title="post_message",
                    kind="other",
                    status="in_progress",
                    raw_input={"text": "[done] All workers finished"},
                ),
            )
            await client.session_update(
                session_id,
                ToolCallProgress(
                    session_update="tool_call_update",
                    tool_call_id="c_done",
                    status="completed",
                    raw_output={"status": "posted"},
                ),
            )
            await client.session_update(
                session_id,
                AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=TextContentBlock(type="text", text="Swarm 全部重构与测试已完成。"),
                ),
            )

        class _FakeConn:
            async def prompt(self, *, session_id: str, prompt):
                nonlocal reactive_task
                del prompt
                await client.session_update(
                    session_id,
                    ToolCallStart(
                        session_update="tool_call",
                        tool_call_id="c_add",
                        title="swarm",
                        kind="other",
                        status="in_progress",
                        raw_input={"action": "add", "name": "worker-1", "brief": "work"},
                    ),
                )
                await client.session_update(
                    session_id,
                    ToolCallProgress(
                        session_update="tool_call_update",
                        tool_call_id="c_add",
                        status="completed",
                        raw_output={"status": "added", "name": "worker-1"},
                    ),
                )
                await client.session_update(
                    session_id,
                    AgentMessageChunk(
                        session_update="agent_message_chunk",
                        content=TextContentBlock(type="text", text="已启动 worker-1，等待汇报。"),
                    ),
                )
                reactive_task = asyncio.create_task(emit_reactive_turns())
                turn1_finished.set()
                return PromptResponse(stop_reason="end_turn")

        acp_session._conn = _FakeConn()
        events: list[ACPEvent] = []
        result = await acp_session.prompt("run swarm", on_event=events.append)
        assert reactive_task is not None
        await reactive_task
        return result, events

    result, events = asyncio.run(scenario())
    assert result.stop_reason == "end_turn"
    assert result.swarm_used is True
    assert result.swarm_completed is True
    assert result.swarm_final_text == "Swarm 全部重构与测试已完成。"
    assert "已启动 worker-1" in result.text
    assert "Swarm 全部重构与测试已完成。" in result.text
    assert any(ev.event_type is ACPEventType.TEXT_CHUNK and "全部重构" in (ev.text or "") for ev in events)


def test_acp_session_cancel_wakes_active_swarm_wait_and_marks_user_cancel(tmp_path: Path) -> None:
    session_id = "sess-agy-swarm-cancel"
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    try:
        acp_session = ACPSession("agy", ["acp"], str(tmp_path))
        acp_session._swarm_settle_scale = 0.05
        acp_session._session_id = session_id
        acp_session._state.session_id = session_id
        acp_session._state.is_active = True
        client = GhostAPClient(on_event=acp_session._dispatch_event, root_dir=str(tmp_path))
        acp_session._client = client
        acp_session._bind_session_info_callback()

        cancel_calls: list[str] = []

        class _FakeConn:
            async def prompt(self, *, session_id: str, prompt):
                del prompt
                await client.session_update(
                    session_id,
                    ToolCallStart(
                        session_update="tool_call",
                        tool_call_id="c_add",
                        title="swarm",
                        kind="other",
                        status="in_progress",
                        raw_input={"action": "add", "name": "worker-1", "brief": "work"},
                    ),
                )
                await client.session_update(
                    session_id,
                    ToolCallProgress(
                        session_update="tool_call_update",
                        tool_call_id="c_add",
                        status="completed",
                        raw_output={"status": "added", "name": "worker-1"},
                    ),
                )
                return PromptResponse(stop_reason="end_turn")

            async def cancel(self, *, session_id: str) -> None:
                cancel_calls.append(session_id)

        acp_session._conn = _FakeConn()
        sync_session = SyncACPSession("agy", str(tmp_path), agent_cmd="agy", agent_args=["acp"])
        sync_session._acp_session = acp_session
        sync_session._loop = loop
        sync_session._started = True

        started = threading.Event()

        def _on_event(ev: ACPEvent) -> None:
            if ev.event_type is ACPEventType.TOOL_CALL_DONE:
                started.set()

        def _cancel_when_waiting() -> None:
            assert started.wait(timeout=2.0)
            for _ in range(50):
                if acp_session.has_active_swarm_sync():
                    break
                time.sleep(0.01)
            gen = sync_session.active_prompt_generation()
            if gen is not None:
                sync_session.mark_user_cancel(gen)
            sync_session.cancel(wait=True, timeout=1.0)

        canceller = threading.Thread(target=_cancel_when_waiting, daemon=True)
        canceller.start()
        result = sync_session.send_prompt("run swarm cancel", on_event=_on_event, timeout=5)
        canceller.join(timeout=2.0)

        assert cancel_calls == [session_id]
        assert result.stop_reason == "cancelled"
        assert result.cancellation_source == "user"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2.0)
        loop.close()


def test_agy_handler_swarm_adds_working_reaction_holds_until_done_and_notifies(tmp_path: Path) -> None:
    assert EmojiReaction.on_swarm_working() == EmojiType.ROCKET
    assert EmojiReaction.on_swarm_done() == EmojiType.PARTY
    assert EmojiReaction.should_send(EmojiReaction.on_swarm_working()) is True
    assert EmojiReaction.should_send(EmojiReaction.on_swarm_done()) is True

    notice = programming_mod._format_swarm_completion_notice("Swarm 已全部完成并提交。")
    assert notice == "✅ Swarm 后台协作任务已全部完成\n\nSwarm 已全部完成并提交。"
    long_notice = programming_mod._format_swarm_completion_notice("A" * 350)
    assert "完整结果见上方任务卡片" in long_notice
    empty_notice = programming_mod._format_swarm_completion_notice("✅ 执行完成")
    assert empty_notice == "✅ Swarm 后台协作任务已全部完成"

    session_id = "sess-agy-handler-swarm"
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    try:
        acp_session = ACPSession("agy", ["acp"], str(tmp_path))
        acp_session._swarm_settle_scale = 0.02
        acp_session._session_id = session_id
        acp_session._state.session_id = session_id
        acp_session._state.is_active = True
        client = GhostAPClient(on_event=acp_session._dispatch_event, root_dir=str(tmp_path))
        acp_session._client = client
        acp_session._bind_session_info_callback()

        working_reaction_seen = threading.Event()
        reactions: list[tuple[str, str]] = []
        replies: list[tuple[str, str]] = []

        class _FakeConn:
            async def prompt(self, *, session_id: str, prompt):
                del prompt
                await client.session_update(
                    session_id,
                    ToolCallStart(
                        session_update="tool_call",
                        tool_call_id="c_add_handler",
                        title="swarm",
                        kind="other",
                        status="in_progress",
                        raw_input={"action": "add", "name": "worker-1", "brief": "do task"},
                    ),
                )
                await client.session_update(
                    session_id,
                    ToolCallProgress(
                        session_update="tool_call_update",
                        tool_call_id="c_add_handler",
                        status="completed",
                        raw_output={"status": "added", "name": "worker-1"},
                    ),
                )
                await client.session_update(
                    session_id,
                    AgentMessageChunk(
                        session_update="agent_message_chunk",
                        content=TextContentBlock(type="text", text="已派出 worker-1 后台执行。"),
                    ),
                )

                async def _finish_swarm() -> None:
                    await asyncio.sleep(0.06)
                    # Before [done] arrives, only the working reaction may exist (never PARTY).
                    assert ("msg_origin", "Fire") in reactions
                    assert ("msg_origin", "PARTY") not in reactions
                    assert not replies
                    await client.session_update(
                        session_id,
                        ToolCallStart(
                            session_update="tool_call",
                            tool_call_id="c_done_handler",
                            title="post_message",
                            kind="other",
                            status="in_progress",
                            raw_input={"text": "[done] Worker finished cleanly"},
                        ),
                    )
                    await client.session_update(
                        session_id,
                        ToolCallProgress(
                            session_update="tool_call_update",
                            tool_call_id="c_done_handler",
                            status="completed",
                            raw_output={"status": "posted"},
                        ),
                    )
                    await client.session_update(
                        session_id,
                        AgentMessageChunk(
                            session_update="agent_message_chunk",
                            content=TextContentBlock(type="text", text="Swarm 所有子任务已全部完成。"),
                        ),
                    )

                asyncio.create_task(_finish_swarm())
                return PromptResponse(stop_reason="end_turn")

        acp_session._conn = _FakeConn()
        sync_session = SyncACPSession("agy", str(tmp_path), agent_cmd="agy", agent_args=["acp"])
        sync_session._acp_session = acp_session
        sync_session._loop = loop
        sync_session._started = True

        handler = object.__new__(programming_mod.AGYModeHandler)
        handler._current_model = None
        handler._active_programming_runs = {}
        handler._active_programming_runs_lock = threading.Lock()
        handler.ctx = SimpleNamespace(
            settings=SimpleNamespace(
                coco_execution_timeout=60,
                claude_execution_timeout=60,
                programming_finalization_reserve_s=0,
                programming_max_execution_windows=1,
                programming_agent_idle_timeout_s=30,
                default_reply_mode="reply",
                card=SimpleNamespace(delivery_api_timeout=2.0),
            ),
            message_linker=MagicMock(),
            context_manager=MagicMock(),
        )

        def _add_reaction(mid: str, emoji: str) -> None:
            reactions.append((mid, emoji))
            if emoji == EmojiType.ROCKET:
                working_reaction_seen.set()

        def _reply_text(mid: str, text: str) -> None:
            replies.append((mid, text))

        handler.add_reaction = _add_reaction
        handler.reply_text = _reply_text
        handler.upload_acp_image = lambda _img: None

        card_client = _AgyCardClient()
        delivery = CardDelivery(card_client)
        metadata = build_programming_metadata("agy", model_name="gemini-4-hydrogen")

        try:
            with patch("src.thread.get_current_thread_id", return_value=None):
                fallback = handler._handle_response_with_delivery(
                    message_id="msg_origin",
                    chat_id="oc_swarm_chat",
                    text="/swarm optimize codebase",
                    session=sync_session,
                    project=None,
                    cwd=str(tmp_path),
                    global_working_dir=str(tmp_path),
                    metadata=metadata,
                    delivery=delivery,
                    delivery_timeout=2.0,
                    project_id=None,
                )
        finally:
            delivery.shutdown(timeout=2.0)

        assert fallback is False
        assert working_reaction_seen.is_set() is True
        assert reactions == [
            ("msg_origin", "Fire"),
            ("msg_origin", "PARTY"),
        ]
        assert replies == [
            ("msg_origin", "✅ Swarm 后台协作任务已全部完成\n\nSwarm 所有子任务已全部完成。"),
        ]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2.0)
        loop.close()


def test_agy_grouped_select_options_flatten_and_never_backfill_unknown_efforts() -> None:
    def make_options(*, include_efforts: bool, current_model: str) -> list:
        options: list = [
            SessionConfigOptionBoolean(
                id="model",
                name="Model bool",
                type="boolean",
                category="model",
                current_value=True,
            ),
            SessionConfigOptionSelect(
                id="model",
                name="Model",
                type="select",
                category="model",
                current_value=current_model,
                options=[
                    SessionConfigSelectGroup(group="empty", name="Empty", options=[]),
                    SessionConfigSelectGroup(
                        group="gemini",
                        name="Gemini",
                        options=[
                            SessionConfigSelectOption(
                                value="gemini-4-hydrogen",
                                name="Gemini 4 Hydrogen",
                            ),
                            SessionConfigSelectOption(
                                value="gemini-pro-orchestrator",
                                name="Gemini Pro Orchestrator",
                            ),
                        ],
                    ),
                ],
            ),
            SessionConfigOptionBoolean(
                id="thought_level",
                name="Thought bool",
                type="boolean",
                category="reasoning_effort",
                current_value=True,
            ),
            SessionConfigOptionSelect(
                id="dict_cat",
                name="Dict Cat",
                type="select",
                category={"id": "reasoning_effort"},
                current_value="xhigh",
                options=[SessionConfigSelectOption(value="xhigh", name="XHigh")],
            ),
        ]
        if include_efforts:
            options.append(
                SessionConfigOptionSelect(
                    id="thought_level",
                    name="Thought Level",
                    type="select",
                    category="thought_level",
                    current_value="medium",
                    options=[
                        SessionConfigSelectGroup(group="empty_thought", name="Empty", options=[]),
                        SessionConfigSelectGroup(
                            group="levels",
                            name="Levels",
                            options=[
                                SessionConfigSelectOption(value="low", name="Low"),
                                SessionConfigSelectOption(value="medium", name="Medium"),
                                SessionConfigSelectOption(value="high", name="High"),
                            ],
                        ),
                    ],
                )
            )
        return options

    initial = NewSessionResponse(
        session_id="sess-agy-grouped",
        config_options=make_options(include_efforts=True, current_model="gemini-4-hydrogen"),
    )

    class Connection:
        async def set_config_option(self, *, config_id: str, session_id: str, value: str):
            assert config_id == "model"
            assert session_id == "sess-agy-grouped"
            return SetSessionConfigOptionResponse(
                config_options=make_options(include_efforts=False, current_model=value),
            )

    models = asyncio.run(helper_mod.discover_agy_model_options(Connection(), initial))
    assert [m.name for m in models] == [
        "gemini-4-hydrogen",
        "gemini-pro-orchestrator",
    ]
    assert models[0].reasoning_efforts == ("low", "medium", "high")
    assert models[0].default_reasoning_effort == "medium"
    assert [v.name for v in models[0].selection_variants] == [
        "gemini-4-hydrogen/low",
        "gemini-4-hydrogen/medium",
        "gemini-4-hydrogen/high",
    ]

    assert models[1].reasoning_efforts == ()
    assert models[1].default_reasoning_effort is None
    assert [v.name for v in models[1].selection_variants] == ["gemini-pro-orchestrator"]
    assert models[1].selection_variants[0].effort is None
