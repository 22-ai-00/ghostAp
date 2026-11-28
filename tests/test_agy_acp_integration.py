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
                    option("gpt-oss-120b-medium", "GPT-OSS 120B (Medium)"),
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
            models=SimpleNamespace(
                available_models=[
                    SimpleNamespace(model_id="gemini-4-carbon-high", name="Gemini 4 Carbon (High)"),
                    SimpleNamespace(model_id="gemini-4-hydrogen-xhigh", name="Gemini 4 Hydrogen (XHigh)"),
                ]
            ),
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
        "gpt-oss-120b-medium",
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
    # Orchestrator exposes `default` (native ACP) plus Planner x Executor combinations
    assert models[2].selection_variants[0].name == "gemini-pro-orchestrator"
    assert models[2].selection_variants[0].profile == "default"
    assert models[2].selection_variants[0].effort is None
    assert len(models[2].selection_variants) == 1 + (3 * 4)
    assert "gemini-pro-orchestrator/high/xhigh" in [v.name for v in models[2].selection_variants]
    # Non-orchestrator fixed model continues to omit efforts.
    assert [variant.name for variant in models[3].selection_variants] == ["gpt-oss-120b-medium"]
    assert models[3].reasoning_efforts == ()
    assert models[2].is_default is True
    assert models[0].is_default is False
    assert connection.calls == [
        ("model", "session-agy", "gemini-4-hydrogen"),
        ("model", "session-agy", "gemini-pro-orchestrator"),
        ("model", "session-agy", "gpt-oss-120b-medium"),
    ]

    hydrogen_state = resolve_model_cascade(models, selected_model="gemini-4-hydrogen", selected_effort="xhigh")
    assert hydrogen_state.efforts == ("low", "medium", "high", "xhigh")
    assert hydrogen_state.selection == "gemini-4-hydrogen/xhigh"

    orchestrator_default = resolve_model_cascade(models, selected_model="gemini-pro-orchestrator")
    assert orchestrator_default.profiles == ("default", "low", "medium", "high")
    assert orchestrator_default.selected_profile == "default"
    assert orchestrator_default.efforts == ()
    assert orchestrator_default.selected_effort is None
    assert orchestrator_default.selection == "gemini-pro-orchestrator"

    orchestrator_custom = resolve_model_cascade(
        models,
        selected_model="gemini-pro-orchestrator",
        selected_profile="high",
        selected_effort="xhigh",
    )
    assert orchestrator_custom.selected_profile == "high"
    assert orchestrator_custom.efforts == ("low", "medium", "high", "xhigh")
    assert orchestrator_custom.selected_effort == "xhigh"
    assert orchestrator_custom.selection == "gemini-pro-orchestrator/high/xhigh"


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


def test_agy_orchestrator_discovery_respects_enterprise_and_native_acp_precedence() -> None:
    def option(value: str, name: str) -> SimpleNamespace:
        return SimpleNamespace(value=value, name=name, description=None)

    def make_response(
        current_model: str,
        efforts: tuple[str, ...],
        *,
        tier: dict | None = None,
    ) -> SimpleNamespace:
        config_options = [
            SimpleNamespace(
                id="model",
                category="model",
                current_value=current_model,
                options=[
                    option("gemini-4-neon", "Gemini 4 Neon"),
                    option("gemini-4-hydrogen", "Gemini 4 Hydrogen"),
                    option("gemini-pro-orchestrator-neon", "Orchestrator (Neon)"),
                ],
            )
        ]
        if efforts:
            config_options.append(
                SimpleNamespace(
                    id="thought_level",
                    category="thought_level",
                    current_value="medium",
                    options=[option(effort, effort.title()) for effort in efforts],
                )
            )
        return SimpleNamespace(
            session_id="sess-agy-tier",
            config_options=config_options,
            field_meta={"antigravity": {"tier": tier}} if tier else None,
        )

    class Connection:
        def __init__(self, *, orch_native_efforts: tuple[str, ...] = (), tier: dict | None = None) -> None:
            self.orch_native_efforts = orch_native_efforts
            self.tier = tier

        async def set_config_option(self, *, config_id: str, session_id: str, value: str):
            del config_id, session_id
            if value in {"gemini-4-neon", "gemini-4-hydrogen"}:
                return make_response(value, ("low", "medium", "high", "xhigh"), tier=self.tier)
            return make_response(value, self.orch_native_efforts, tier=self.tier)

    # Enterprise tier suppresses Orchestrator CLI custom-effort synthesis.
    enterprise_initial = make_response(
        "gemini-4-neon",
        ("low", "medium", "high", "xhigh"),
        tier={"id": "g1-enterprise-tier", "name": "Enterprise"},
    )
    enterprise_models = asyncio.run(
        helper_mod.discover_agy_model_options(
            Connection(tier={"id": "g1-enterprise-tier", "name": "Enterprise"}),
            enterprise_initial,
        )
    )
    assert [v.name for v in enterprise_models[2].selection_variants] == [
        "gemini-pro-orchestrator-neon"
    ]

    # Native ACP thought_level on Orchestrator (if exposed by future agy acp) takes precedence.
    native_initial = make_response("gemini-4-neon", ("low", "medium", "high", "xhigh"))
    native_models = asyncio.run(
        helper_mod.discover_agy_model_options(
            Connection(orch_native_efforts=("low", "high")),
            native_initial,
        )
    )
    assert [v.name for v in native_models[2].selection_variants] == [
        "gemini-pro-orchestrator-neon/low",
        "gemini-pro-orchestrator-neon/high",
    ]


def test_agy_orchestrator_selection_settings_and_slug_normalization() -> None:
    from src.acp.agy_selection import (
        compose_agy_orchestrator_selection,
        normalize_agy_model_input,
        parse_agy_model_selection,
        requires_agy_cli_orchestrator_session,
        resolve_agy_orchestrator_settings,
    )

    assert compose_agy_orchestrator_selection("gemini-pro-orchestrator-neon", "default") == (
        "gemini-pro-orchestrator-neon"
    )
    assert compose_agy_orchestrator_selection("gemini-pro-orchestrator-neon", "xhigh", "low") == (
        "gemini-pro-orchestrator-neon/xhigh/low"
    )
    assert compose_agy_orchestrator_selection("gemini-pro-orchestrator-neon", "high") == (
        "gemini-pro-orchestrator-neon/high/high"
    )

    parsed = parse_agy_model_selection("gemini-pro-orchestrator-neon/xhigh/low")
    assert (parsed.model_id, parsed.planner_effort, parsed.executor_effort) == (
        "gemini-pro-orchestrator-neon",
        "xhigh",
        "low",
    )

    assert requires_agy_cli_orchestrator_session("gemini-pro-orchestrator-neon/xhigh/low") is True
    assert requires_agy_cli_orchestrator_session("gemini-pro-orchestrator-neon") is False
    assert requires_agy_cli_orchestrator_session("gemini-4-neon/xhigh") is False
    assert requires_agy_cli_orchestrator_session(None) is False

    base_model, settings_payload = resolve_agy_orchestrator_settings(
        "gemini-pro-orchestrator-neon/xhigh/low"
    )
    assert base_model == "gemini-pro-orchestrator-neon"
    assert settings_payload == {
        "orchestrator": {
            "planner": "Gemini 4 Neon (XHigh)",
            "executor": "Gemini 4 Hydrogen (Low)",
        }
    }

    with pytest.raises(ValueError, match="invalid AGY model selection"):
        split_agy_model_selection("gemini-pro-orchestrator/high/low")
    with pytest.raises(ValueError, match="CLI settings"):
        split_agy_model_selection("gemini-pro-orchestrator/high")

    wrapper = SyncACPSession("agy", "/tmp", agent_cmd="agy", agent_args=["acp"])
    wrapper._acp_session = SimpleNamespace(set_config_option=MagicMock())  # type: ignore[assignment]
    assert asyncio.run(
        wrapper._apply_agy_selection("gemini-pro-orchestrator/high/low")  # type: ignore[attr-defined]
    ) is False

    known = [
        "gemini-4-neon/xhigh",
        "claude-opus-5-5/high",
        "gemini-pro-orchestrator-neon",
        "gemini-pro-orchestrator-neon/xhigh/medium",
        "gemini-pro-orchestrator-neon/xhigh/xhigh",
    ]
    assert normalize_agy_model_input("gemini-4-neon-xhigh", known) == "gemini-4-neon/xhigh"
    assert normalize_agy_model_input("claude-opus-5-5-high", known) == "claude-opus-5-5/high"
    assert (
        normalize_agy_model_input("gemini-pro-orchestrator-neon/xhigh", known)
        == "gemini-pro-orchestrator-neon/xhigh/xhigh"
    )
    assert (
        normalize_agy_model_input("gemini-pro-orchestrator-neon-xhigh", known)
        == "gemini-pro-orchestrator-neon/xhigh/xhigh"
    )


def test_sync_agy_cli_session_args_stream_json_and_set_model(tmp_path: Path) -> None:
    from src.agent_session.agy_cli import SyncAGYCLISession

    session = SyncAGYCLISession(
        cwd=str(tmp_path),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
    )
    turn1_args = session._build_args("- refactor this", resumed=False)
    assert turn1_args[0] == "agy"
    assert turn1_args[1].startswith("--prompt=- refactor this")
    assert "--output-format" in turn1_args and "stream-json" in turn1_args
    assert "--dangerously-skip-permissions" in turn1_args
    assert turn1_args[turn1_args.index("--model") + 1] == "gemini-pro-orchestrator-neon"
    settings_json = json.loads(turn1_args[turn1_args.index("--settings") + 1])
    assert settings_json == {
        "showThinking": True,
        "orchestrator": {
            "planner": "Gemini 4 Neon (XHigh)",
            "executor": "Gemini 4 Hydrogen (Low)",
        },
    }
    assert "--conversation" not in turn1_args

    # Consume realistic agy 1.5.14 stream-json events
    events: list[ACPEvent] = []
    chunks: list[str] = []
    lines = [
        json.dumps({"event": "init", "conversation_id": "conv-agy-123", "init": {}}),
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "conversation_id": "conv-agy-123",
                    "step_index": 1,
                    "state": "ACTIVE",
                    "step_type": "tool",
                    "tool_name": "run_command",
                    "tool_info": {"name": "run_command", "parameters": {"CommandLine": "ls"}},
                },
            }
        ),
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "conversation_id": "conv-agy-123",
                    "step_index": 1,
                    "state": "DONE",
                    "step_type": "tool",
                    "tool_name": "run_command",
                    "tool_info": {"name": "run_command", "output": "ok\n"},
                },
            }
        ),
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "conversation_id": "conv-agy-123",
                    "step_index": 2,
                    "state": "ACTIVE",
                    "step_type": "agent_response",
                    "text_delta": "Final ",
                },
            }
        ),
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "conversation_id": "conv-agy-123",
                    "step_index": 2,
                    "state": "DONE",
                    "step_type": "agent_response",
                    "text_delta": "Answer",
                    "usage": {"total_tokens": 321},
                },
            }
        ),
        json.dumps(
            {
                "event": "result",
                "result": {
                    "conversation_id": "conv-agy-123",
                    "status": "SUCCESS",
                    "response": "Final Answer",
                    "usage": {"total_tokens": 321},
                },
            }
        ),
    ]
    for line in lines:
        assert session._consume_stdout_line(line, chunks, events.append) == ""

    result = session._handle_process_exit(
        rc=0,
        out="".join(chunks),
        err="",
        state="ok",
        backend_error="",
        on_event=events.append,
    )
    assert session.session_id == "conv-agy-123"
    assert result.stop_reason == "end_turn"
    assert result.terminal_text == "Final Answer"
    assert result.usage is not None and result.usage.used == 321
    assert [ev.event_type for ev in events] == [
        ACPEventType.TOOL_CALL_START,
        ACPEventType.TOOL_CALL_DONE,
        ACPEventType.TEXT_CHUNK,
        ACPEventType.TEXT_CHUNK,
    ]
    assert session.supports_background_tool_continuation is False
    assert session.to_snapshot()["agent_type"] == "agy"

    # Switching between Orchestrator custom-effort presets updates in-place & resumes conversation.
    assert session.set_model("gemini-pro-orchestrator-neon/low/xhigh") is True
    turn2_args = session._build_args("next step", resumed=True)
    assert turn2_args[turn2_args.index("--conversation") + 1] == "conv-agy-123"
    turn2_settings = json.loads(turn2_args[turn2_args.index("--settings") + 1])
    assert turn2_settings["orchestrator"] == {
        "planner": "Gemini 4 Neon (Low)",
        "executor": "Gemini 4 Hydrogen (XHigh)",
    }

    # Switching back to regular ACP or bare preset asks caller to replace transport.
    assert session.set_model("gemini-4-neon/xhigh") is False
    assert session.set_model("gemini-pro-orchestrator-neon") is False


def test_agy_hybrid_transport_dispatch_card_labels_and_commands(tmp_path: Path) -> None:
    from src.acp.manager import ACPSessionManager
    from src.acp.options import ACPModelOption, ACPModelSelectionVariant
    from src.acp.startup_utils import AcpRetryStarter, StartupBackend, select_startup_backend
    from src.agent_session.agy_cli import SyncAGYCLISession
    from src.agent_session.factory import create_engine_session
    from src.card.builders.system import SystemBuilder
    from src.feishu.handlers.system import SystemHandler
    from src.feishu.product_catalog import resolve_command

    assert select_startup_backend("agy") == StartupBackend.ACP
    assert select_startup_backend("agy", model_name="gemini-4-neon/xhigh") == StartupBackend.ACP
    assert select_startup_backend("agy", model_name="gemini-pro-orchestrator-neon") == StartupBackend.ACP
    assert (
        select_startup_backend("agy", model_name="gemini-pro-orchestrator-neon/xhigh/low")
        == StartupBackend.CLI
    )

    with patch("src.agent_session.agy_cli.SyncAGYCLISession") as cli_cls:
        AcpRetryStarter._build_session(
            backend=StartupBackend.CLI,
            agent_type="agy",
            cwd=str(tmp_path),
            model_name="gemini-pro-orchestrator-neon/xhigh/low",
            acp_session_cls=MagicMock(),
            cli_session_cls=MagicMock(),
        )
    cli_cls.assert_called_once_with(
        cwd=str(tmp_path),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
    )

    settings = MagicMock(rate_limit_retry_enabled=False, acp_startup_timeout=20)
    with (
        patch("src.agent_session.factory.get_settings", return_value=settings),
        patch("src.agent_session.factory.SyncAGYCLISession") as factory_cli_cls,
    ):
        create_engine_session(
            "agy",
            str(tmp_path),
            model_name="gemini-pro-orchestrator-neon/xhigh/low",
        )
    factory_cli_cls.assert_called_once_with(
        cwd=str(tmp_path),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
    )

    # Manager ensure_session replaces existing session across ACP <-> AGY CLI transport boundary.
    mgr = ACPSessionManager("agy")
    cli_live = SyncAGYCLISession(
        cwd=str(tmp_path),
        model_name="gemini-pro-orchestrator-neon/xhigh/low",
    )
    cli_live.session_id = "conv-old"
    acp_new = MagicMock(session_id="acp-new", _model_name="gemini-4-neon/xhigh")
    mgr._sessions[mgr._session_key("chat-1", "proj-1")] = cli_live  # type: ignore[assignment]
    with patch.object(mgr, "_create_session", return_value=acp_new) as create_mock:
        returned = mgr.ensure_session(
            "chat-1",
            cwd=str(tmp_path),
            project_id="proj-1",
            model_name="gemini-4-neon/xhigh",
        )
    assert returned is acp_new
    create_mock.assert_called_once()

    # Model card labels show Planner Effort / Executor Effort only for AGY Orchestrator.
    orch_option = ACPModelOption(
        name="gemini-pro-orchestrator-neon",
        description="Orchestrator (Neon)",
        selection_variants=(
            ACPModelSelectionVariant(
                name="gemini-pro-orchestrator-neon",
                model="gemini-pro-orchestrator-neon",
                profile="default",
                is_default=True,
            ),
            ACPModelSelectionVariant(
                name="gemini-pro-orchestrator-neon/xhigh/low",
                model="gemini-pro-orchestrator-neon",
                profile="xhigh",
                effort="low",
            ),
        ),
        reasoning_efforts=("low", "medium", "high", "xhigh"),
        default_reasoning_effort="high",
    )
    reg_option = ACPModelOption(
        name="gemini-4-neon",
        description="Gemini 4 Neon",
        selection_variants=(
            ACPModelSelectionVariant(
                name="gemini-4-neon/xhigh",
                model="gemini-4-neon",
                effort="xhigh",
                is_default=True,
            ),
        ),
        reasoning_efforts=("xhigh",),
        default_reasoning_effort="xhigh",
    )
    _t, orch_card_json = SystemBuilder.build_acp_model_cascade_card(
        [orch_option, reg_option],
        "agy",
        project_id="proj-1",
        pending_group="gemini-pro-orchestrator-neon",
        pending_profile="xhigh",
        pending_effort="low",
    )
    assert "Planner Effort" in orch_card_json
    assert "Executor Effort" in orch_card_json

    _t, reg_card_json = SystemBuilder.build_acp_model_cascade_card(
        [orch_option, reg_option],
        "agy",
        project_id="proj-1",
        pending_group="gemini-4-neon",
        pending_effort="xhigh",
    )
    assert "**Planner Effort**" not in reg_card_json
    assert "**Effort**" in reg_card_json

    # /effort catalog registration + /effort and /model command handling + cross-transport activation.
    assert resolve_command("/effort") is not None

    project = ProjectContext("proj-1", "Project 1", str(tmp_path))
    project.acp_tool_name = "agy"
    project.acp_model_name = "gemini-pro-orchestrator-neon"

    committed: list[tuple[str, str | None]] = []
    project_mgr = MagicMock()
    project_mgr.get_active_project.return_value = project
    def _commit(proj, *, tool_name: str, model_name: str | None) -> bool:
        proj.acp_tool_name = tool_name
        proj.acp_model_name = model_name
        committed.append((tool_name, model_name))
        return True
    project_mgr.commit_acp_configuration.side_effect = _commit

    agy_handler = MagicMock()
    agy_handler.current_model = project.acp_model_name
    agy_session_mgr = MagicMock()
    live_acp = SimpleNamespace(session_id="acp-1", _model_name="gemini-pro-orchestrator-neon", set_model=MagicMock(return_value=False))
    replaced_cli = SimpleNamespace(session_id="cli-1", _model_name="gemini-pro-orchestrator-neon/xhigh/low")
    agy_session_mgr.get_session.return_value = live_acp
    agy_session_mgr.replace_session.return_value = SimpleNamespace(session=replaced_cli, created=True)
    agy_handler._get_session_manager.return_value = agy_session_mgr
    agy_handler._get_agent_type_override.return_value = None

    ctx = MagicMock(
        settings=SimpleNamespace(acp_startup_timeout=20),
        mode_manager=ModeManager(),
        project_manager=project_mgr,
        working_dir=str(tmp_path),
        handlers={"agy": agy_handler},
        scheduler=MagicMock(),
    )
    ctx.get_handler.side_effect = lambda name: ctx.handlers.get(name)
    system = SystemHandler(ctx)
    replies: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []
    system.reply_text = lambda mid, body, **_kw: replies.append((mid, body))  # type: ignore[method-assign]
    system.reply_error = lambda mid, body, **_kw: errors.append((mid, body))  # type: ignore[method-assign]
    system._fetch_acp_models = lambda tool, **_kw: [orch_option, reg_option]  # type: ignore[method-assign]

    from src.feishu.slash_command_parser import SlashCommandParser

    effort_match = SlashCommandParser.parse("/effort xhigh low")
    assert SystemHandler.is_ghostap_control_command(effort_match, in_programming_mode=True) is True
    assert SystemHandler.is_fast_lane_system_command("/effort xhigh low", in_programming_mode=True) is True

    # Query current /effort on Orchestrator
    system.handle_effort_command("m1", "chat-1", "/effort", project=project)
    assert "Orchestrator Effort" in replies[-1][1]

    # Set dual Planner + Executor effort
    system.handle_effort_command("m2", "chat-1", "/effort xhigh low", project=project)
    assert committed[-1] == ("agy", "gemini-pro-orchestrator-neon/xhigh/low")
    agy_session_mgr.retire_session.assert_called_once_with("chat-1", project_id="proj-1")

    # Reset back to Orchestrator default
    system.handle_effort_command("m3", "chat-1", "/effort default", project=project)
    assert committed[-1] == ("agy", "gemini-pro-orchestrator-neon")

    # Reject bogus effort
    system.handle_effort_command("m4", "chat-1", "/effort bogus", project=project)
    assert "bogus" in errors[-1][1]

    # /model accepts hyphenated slug and normalizes
    system.handle_model_command("m5", "chat-1", "/model gemini-4-neon-xhigh", project=project)
    assert committed[-1] == ("agy", "gemini-4-neon/xhigh")

    # _enter_mode_with_acp_model replaces live ACP session when targeting CLI Orchestrator
    effect = system._enter_mode_with_acp_model(
        "m6",
        "chat-1",
        "agy",
        "gemini-pro-orchestrator-neon/xhigh/low",
        project,
    )
    assert effect is not None
    assert effect.session is replaced_cli  # type: ignore[union-attr]
    agy_handler.switch_model.assert_not_called()
    agy_session_mgr.replace_session.assert_called_once()

    # Same-transport switch remains fail-closed when switch_model returns False.
    agy_handler.switch_model.return_value = False
    same_transport_effect = system._enter_mode_with_acp_model(
        "m7",
        "chat-1",
        "agy",
        "gemini-4-neon/xhigh",
        project,
    )
    assert same_transport_effect is None
    agy_handler.switch_model.assert_called_once()


def test_bound_chat_active_project_and_acp_activation_without_active_map(tmp_path: Path) -> None:
    from src.feishu.handlers.system import SystemHandler
    from src.project.manager import ProjectManager

    storage = tmp_path / "projects.json"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    pm = ProjectManager(storage_path=str(storage))
    ok, _msg, project = pm.create_project("p_bound", "BoundProj", str(repo_dir), chat_id="owner-chat")
    assert ok and project is not None

    bound_cid = "oc_bound_group_1"
    prep_ok, snap = pm.bind_managed_chat_for_saga(
        project_id=project.project_id,
        chat_id=bound_cid,
        chat_name="Bound Group",
        created_at=1000.0,
        operation_id="op-bind-1",
        expected_origin="ghostap_created",
        expected_owner_id="owner-1",
        expected_receiving_bot_ref="bot-1",
    )
    assert prep_ok and snap is not None
    assert pm.get_active_project(bound_cid) is project

    # Legacy snapshot without active_project[bound_cid] still resolves via _bound_chat_index
    pm._active_project.pop(bound_cid, None)
    assert pm.get_active_project(bound_cid) is project

    # ACP activation linearizes cleanly when active_for_chat matches bound project
    pm.complete_managed_chat_binding_saga("op-bind-1")
    reloaded = ProjectManager(storage_path=str(storage))
    reloaded._active_project.pop(bound_cid, None)
    bound_proj = reloaded.get_active_project(bound_cid)
    assert bound_proj is not None and bound_proj.project_id == "p_bound"

    # Revoke and restore clean up _active_project
    assert reloaded.revoke_managed_chat(bound_cid) is True
    assert bound_cid not in reloaded._active_project

    prep_ok2, snap2 = reloaded.bind_managed_chat_for_saga(
        project_id="p_bound",
        chat_id=bound_cid,
        chat_name="Bound Group 2",
        created_at=1001.0,
        operation_id="op-bind-2",
        expected_origin="ghostap_created",
        expected_owner_id="owner-1",
        expected_receiving_bot_ref="bot-1",
    )
    assert prep_ok2 and snap2 is not None
    assert reloaded.restore_managed_chat_binding("p_bound", snap2, operation_id="op-bind-2") is True
    assert bound_cid not in reloaded._active_project

    # Verify fallback in _activate_acp_selection via find_by_bound_chat_id even with stub manager
    live_session = SimpleNamespace(session_id="acp-bound-1", message_count=0)
    agy_mgr = MagicMock()
    agy_mgr.get_session.side_effect = [None, live_session, live_session]
    agy_handler = MagicMock()
    agy_handler._get_session_manager.return_value = agy_mgr
    agy_handler.enter_mode.return_value = True

    stub_pm = MagicMock()
    stub_pm.get_active_project.return_value = None
    stub_pm.find_by_bound_chat_id.return_value = bound_proj
    stub_pm.commit_acp_programming_activation.return_value = True

    def _run_immediately(spec, fn):
        fn(SimpleNamespace(run_id="r-bound"))
        return MagicMock()

    ctx = MagicMock(
        settings=SimpleNamespace(acp_startup_timeout=20),
        mode_manager=ModeManager(),
        project_manager=stub_pm,
        working_dir=str(repo_dir),
        handlers={"agy": agy_handler},
        scheduler=MagicMock(submit=MagicMock(side_effect=_run_immediately)),
    )
    ctx.get_handler.side_effect = lambda name: ctx.handlers.get(name)
    system = SystemHandler(ctx)
    system.update_card = MagicMock(return_value=True)  # type: ignore[method-assign]
    system.reply_card = MagicMock(return_value="om_card")  # type: ignore[method-assign]

    system._activate_acp_selection("m-bind", bound_cid, "agy", None, bound_proj, explicit_card=True)
    stub_pm.commit_acp_programming_activation.assert_called_once()


def test_agy_default_model_carbon_orchestrator_and_sync_start(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock

    for agent_name in ("agy", "agy_b"):
        sync_session = SyncACPSession(
            agent_name,
            str(tmp_path),
            agent_cmd="agy-b" if agent_name == "agy_b" else "agy",
            agent_args=["acp"],
        )
        async_sess = AsyncMock()
        async_sess.start = AsyncMock(return_value="sid-default")
        async_sess.set_config_option = AsyncMock(return_value=True)
        async_sess.close = AsyncMock()
        with patch("src.acp.sync_adapter.ACPSession", return_value=async_sess):
            assert asyncio.run(sync_session._start_session()) == "sid-default"
        async_sess.set_config_option.assert_awaited_once_with("model", "gemini-pro-orchestrator")


def test_agy_orchestrator_planner_executor_model_and_effort_cascade_and_cli(tmp_path: Path) -> None:
    from src.acp.agy_selection import (
        compose_agy_orchestrator_selection,
        is_valid_agy_catalog_selection,
        normalize_agy_model_input,
        parse_agy_model_selection,
        requires_agy_cli_orchestrator_session,
        resolve_agy_orchestrator_settings,
    )
    from src.acp.options import ACPModelOption, ACPModelSelectionVariant
    from src.agent_session.agy_cli import AGYCLIConfig, SyncAGYCLISession
    from src.card.actions import dispatch as action_ids
    from src.card.builders.system import SystemBuilder
    from src.card.builders.tool_cockpit import ToolCockpitBuilder
    from src.card.render.model_cascade import resolve_agy_orchestrator_cascade
    from src.card.shared.truncation import count_tagged_nodes
    from src.card.thresholds import THRESHOLDS
    from src.feishu.handlers.system import SystemHandler
    from src.workflow_engine.tool_registry import ToolCapability, validate_agent_binding_spec

    def _walk_nodes(obj: object):
        if isinstance(obj, dict):
            yield obj
            for child in obj.values():
                yield from _walk_nodes(child)
        elif isinstance(obj, list):
            for child in obj:
                yield from _walk_nodes(child)

    catalog = [
        ACPModelOption(
            name="gemini-4-neon",
            description="Gemini 4 Neon",
            reasoning_efforts=("low", "medium", "high", "xhigh"),
            default_reasoning_effort="high",
            selection_variants=tuple(
                ACPModelSelectionVariant(
                    name=f"gemini-4-neon/{e}", model="gemini-4-neon", effort=e, is_default=(e == "high")
                )
                for e in ("low", "medium", "high", "xhigh")
            ),
        ),
        ACPModelOption(
            name="gemini-4-carbon",
            description="Gemini 4 Carbon",
            reasoning_efforts=("low", "medium", "high"),
            default_reasoning_effort="high",
            selection_variants=tuple(
                ACPModelSelectionVariant(
                    name=f"gemini-4-carbon/{e}", model="gemini-4-carbon", effort=e, is_default=(e == "high")
                )
                for e in ("low", "medium", "high")
            ),
        ),
        ACPModelOption(
            name="gemini-4-argon",
            description="Gemini 4 Argon",
            reasoning_efforts=("low", "medium", "high"),
            default_reasoning_effort="high",
            selection_variants=tuple(
                ACPModelSelectionVariant(
                    name=f"gemini-4-argon/{e}", model="gemini-4-argon", effort=e, is_default=(e == "high")
                )
                for e in ("low", "medium", "high")
            ),
        ),
        ACPModelOption(
            name="gemini-4-hydrogen",
            description="Gemini 4 Hydrogen",
            reasoning_efforts=("low", "medium", "high", "xhigh"),
            default_reasoning_effort="high",
            selection_variants=tuple(
                ACPModelSelectionVariant(
                    name=f"gemini-4-hydrogen/{e}", model="gemini-4-hydrogen", effort=e, is_default=(e == "high")
                )
                for e in ("low", "medium", "high", "xhigh")
            ),
        ),
        ACPModelOption(
            name="gemini-pro-orchestrator-neon",
            description="Orchestrator (Neon)",
            reasoning_efforts=("low", "medium", "high", "xhigh"),
            default_reasoning_effort="high",
            selection_variants=(
                ACPModelSelectionVariant(
                    name="gemini-pro-orchestrator-neon",
                    model="gemini-pro-orchestrator-neon",
                    profile="default",
                    is_default=True,
                ),
            ),
        ),
        ACPModelOption(
            name="gemini-pro-orchestrator",
            description="Orchestrator (Carbon)",
            is_default=True,
            reasoning_efforts=("low", "medium", "high"),
            default_reasoning_effort="high",
            selection_variants=(
                ACPModelSelectionVariant(
                    name="gemini-pro-orchestrator",
                    model="gemini-pro-orchestrator",
                    profile="default",
                    is_default=True,
                ),
                ACPModelSelectionVariant(
                    name="gemini-pro-orchestrator/high/high",
                    model="gemini-pro-orchestrator",
                    profile="high",
                    effort="high",
                ),
            ),
        ),
        ACPModelOption(
            name="gemini-pro-orchestrator-argon",
            description="Orchestrator (Argon)",
            reasoning_efforts=("low", "medium", "high"),
            default_reasoning_effort="high",
            selection_variants=(
                ACPModelSelectionVariant(
                    name="gemini-pro-orchestrator-argon",
                    model="gemini-pro-orchestrator-argon",
                    profile="default",
                    is_default=True,
                ),
            ),
        ),
        ACPModelOption(
            name="gemini-3.8-flash",
            description="Gemini 3.8 Flash",
            reasoning_efforts=("low", "medium", "high"),
            default_reasoning_effort="high",
            selection_variants=tuple(
                ACPModelSelectionVariant(
                    name=f"gemini-3.8-flash/{e}", model="gemini-3.8-flash", effort=e, is_default=(e == "high")
                )
                for e in ("low", "medium", "high")
            ),
        ),
        ACPModelOption(
            name="claude-opus-5-5",
            description="Claude Opus 5.5",
            reasoning_efforts=("low", "medium", "high"),
            default_reasoning_effort="medium",
            selection_variants=tuple(
                ACPModelSelectionVariant(
                    name=f"claude-opus-5-5/{e}", model="claude-opus-5-5", effort=e, is_default=(e == "medium")
                )
                for e in ("low", "medium", "high")
            ),
        ),
        ACPModelOption(
            name="claude-sonnet-5-5",
            description="Claude Sonnet 5.5",
            reasoning_efforts=("low", "medium", "high"),
            default_reasoning_effort="medium",
            selection_variants=tuple(
                ACPModelSelectionVariant(
                    name=f"claude-sonnet-5-5/{e}", model="claude-sonnet-5-5", effort=e, is_default=(e == "medium")
                )
                for e in ("low", "medium", "high")
            ),
        ),
        ACPModelOption(
            name="gpt-oss-120b-medium",
            description="GPT-OSS 120B (Medium)",
            reasoning_efforts=(),
            selection_variants=(
                ACPModelSelectionVariant(name="gpt-oss-120b-medium", model="gpt-oss-120b-medium", is_default=True),
            ),
        ),
    ]

    # 1. Default cascade picks gemini-pro-orchestrator and Carbon+Hydrogen high, while preserving bare native default
    default_cascade = resolve_model_cascade(catalog)
    assert default_cascade.selected_model == "gemini-pro-orchestrator"
    orch_state = resolve_agy_orchestrator_cascade(catalog, orchestrator_model="gemini-pro-orchestrator")
    assert orch_state.has_role_models is True
    assert orch_state.planner_model_names == (
        "gemini-4-neon",
        "gemini-4-carbon",
        "gemini-4-argon",
        "claude-opus-5-5",
    )
    assert orch_state.executor_model_names == (
        "gemini-4-hydrogen",
        "gemini-4-neon",
        "claude-sonnet-5-5",
    )
    assert orch_state.selected_planner_model == "gemini-4-carbon"
    assert orch_state.selected_planner_effort == "high"
    assert orch_state.selected_executor_model == "gemini-4-hydrogen"
    assert orch_state.selected_executor_effort == "high"
    assert orch_state.selection == "gemini-pro-orchestrator/high/high"

    bare_preserved = resolve_agy_orchestrator_cascade(
        catalog,
        orchestrator_model="gemini-pro-orchestrator",
        current_model="gemini-pro-orchestrator",
    )
    assert bare_preserved.selected_planner_model == "gemini-4-carbon"
    assert bare_preserved.selected_planner_effort == "default"
    assert bare_preserved.selected_executor_model == "gemini-4-hydrogen"
    assert bare_preserved.selected_executor_effort == "default"
    assert bare_preserved.selection == "gemini-pro-orchestrator"

    # 2. Custom allowed Planner (Opus) + Executor (Sonnet / Hydrogen) round-trip & strict allowlist enforcement
    custom_sel = compose_agy_orchestrator_selection(
        "gemini-pro-orchestrator",
        "high",
        "xhigh",
        planner_model="claude-opus-5-5",
        executor_model="gemini-4-hydrogen",
    )
    assert custom_sel == "gemini-pro-orchestrator/claude-opus-5-5:high/gemini-4-hydrogen:xhigh"
    assert requires_agy_cli_orchestrator_session(custom_sel) is True
    assert is_valid_agy_catalog_selection(catalog, custom_sel) is True
    parsed = parse_agy_model_selection(custom_sel)
    assert (parsed.model_id, parsed.planner_model, parsed.planner_effort, parsed.executor_model, parsed.executor_effort) == (
        "gemini-pro-orchestrator",
        "claude-opus-5-5",
        "high",
        "gemini-4-hydrogen",
        "xhigh",
    )
    assert resolve_agy_orchestrator_settings(custom_sel) == (
        "gemini-pro-orchestrator",
        {"orchestrator": {"planner": "Claude Opus 5.5 (High)", "executor": "Gemini 4 Hydrogen (XHigh)"}},
    )
    assert (
        normalize_agy_model_input("gemini-pro-orchestrator/claude-opus-5-5/high/gemini-4-hydrogen/xhigh")
        == custom_sel
    )
    assert (
        normalize_agy_model_input("orchestrator/high/xhigh")
        == "gemini-pro-orchestrator/high/xhigh"
    )

    sonnet_sel = compose_agy_orchestrator_selection(
        "gemini-pro-orchestrator",
        "medium",
        "high",
        planner_model="claude-opus-5-5",
        executor_model="claude-sonnet-5-5",
    )
    assert sonnet_sel == "gemini-pro-orchestrator/claude-opus-5-5:medium/claude-sonnet-5-5:high"
    assert is_valid_agy_catalog_selection(catalog, sonnet_sel) is True
    assert resolve_agy_orchestrator_settings(sonnet_sel) == (
        "gemini-pro-orchestrator",
        {"orchestrator": {"planner": "Claude Opus 5.5 (Medium)", "executor": "Claude Sonnet 5.5 (High)"}},
    )

    # Disallowed Planner/Executor models fail-closed across selection helpers and Workflow validation
    disallowed_pairs = [
        ("gpt-oss-120b-medium", "gemini-4-hydrogen"),
        ("gemini-3.8-flash", "gemini-4-hydrogen"),
        ("gemini-4-hydrogen", "gemini-4-hydrogen"),
        ("gemini-4-carbon", "gemini-4-carbon"),
        ("gemini-4-carbon", "claude-opus-5-5"),
        ("gemini-4-carbon", "gpt-oss-120b-medium"),
    ]
    for bad_p, bad_e in disallowed_pairs:
        with pytest.raises(ValueError):
            compose_agy_orchestrator_selection(
                "gemini-pro-orchestrator",
                "high",
                "high",
                planner_model=bad_p,
                executor_model=bad_e,
            )
        bad_sel = f"gemini-pro-orchestrator/{bad_p}:high/{bad_e}:high"
        with pytest.raises(ValueError):
            parse_agy_model_selection(bad_sel)
        assert is_valid_agy_catalog_selection(catalog, bad_sel) is False

    wf_cap = ToolCapability(
        tool_name="agy",
        display_name="AGY",
        available=True,
        supports_model=True,
        supports_effort=True,
        models=tuple(opt.name for opt in catalog),
        efforts_by_model={opt.name: opt.reasoning_efforts for opt in catalog if opt.reasoning_efforts},
        authoritative_models=True,
    )
    assert validate_agent_binding_spec("agy", model_name=sonnet_sel, capabilities={"agy": wf_cap})[0] is True
    ok_bad_wf, _ = validate_agent_binding_spec(
        "agy",
        model_name="gemini-pro-orchestrator/gpt-oss-120b-medium:high/gemini-4-hydrogen:high",
        capabilities={"agy": wf_cap},
    )
    assert ok_bad_wf is False

    # 3. SyncAGYCLISession preserves custom Planner/Executor models
    cli_sess = SyncAGYCLISession(cwd=str(tmp_path), config=AGYCLIConfig(command="agy"), model_name=custom_sel)
    cli_args = cli_sess._build_args("solve it", resumed=False)
    assert json.loads(cli_args[cli_args.index("--settings") + 1])["orchestrator"] == {
        "planner": "Claude Opus 5.5 (High)",
        "executor": "Gemini 4 Hydrogen (XHigh)",
    }
    assert cli_sess.set_model(sonnet_sel) is True
    assert cli_sess._model_name == sonnet_sel
    assert "Claude Sonnet 5.5 (High)" in cli_sess.get_session_info()

    # 4. Card builder unifies Orchestrator into 1 entry, restricts 4x3 dropdowns in 2x2 column_set, omits width
    _t, card_json = SystemBuilder.build_acp_model_cascade_card(
        catalog,
        "agy",
        project_id="proj-1",
        pending_group="gemini-pro-orchestrator",
        pending_planner_model="claude-opus-5-5",
        pending_profile="high",
        pending_executor_model="gemini-4-hydrogen",
        pending_effort="xhigh",
    )
    card_obj = json.loads(card_json)
    selects_list = [node for node in _walk_nodes(card_obj) if node.get("tag") == "select_static"]
    select_names = [el["name"] for el in selects_list]
    assert select_names == ["model_group", "planner_model", "model_profile", "executor_model", "model_effort"]
    assert all("width" not in el for el in selects_list)
    selects_map = {el["name"]: el for el in selects_list}
    group_opts = selects_map["model_group"]["options"]
    assert [opt["value"] for opt in group_opts].count("gemini-pro-orchestrator") == 1
    assert "gemini-pro-orchestrator-neon" not in [opt["value"] for opt in group_opts]
    assert "gemini-pro-orchestrator-argon" not in [opt["value"] for opt in group_opts]
    assert next(o["text"]["content"] for o in group_opts if o["value"] == "gemini-pro-orchestrator") == "🪐 Orchestrator（双阶段规划+执行）"
    assert [opt["value"] for opt in selects_map["planner_model"]["options"]] == [
        "gemini-4-neon",
        "gemini-4-carbon",
        "gemini-4-argon",
        "claude-opus-5-5",
    ]
    assert [opt["value"] for opt in selects_map["executor_model"]["options"]] == [
        "gemini-4-hydrogen",
        "gemini-4-neon",
        "claude-sonnet-5-5",
    ]
    assert any(node.get("tag") == "column_set" for node in _walk_nodes(card_obj))
    assert custom_sel in card_json

    _t2, settings_json = ToolCockpitBuilder.build_tool_settings_card(
        tool_name="agy",
        models=catalog,
        current_model=custom_sel,
        available_tools=("agy", "coco"),
        project_name="Proj1",
        project_id="proj-1",
    )
    settings_card = json.loads(settings_json)
    assert count_tagged_nodes(settings_card) <= THRESHOLDS["CARD_NODE_BUDGET"]
    assert len(settings_json.encode("utf-8")) <= THRESHOLDS["CARD_BYTE_BUDGET"]
    settings_selects = {n["name"]: n for n in _walk_nodes(settings_card) if n.get("tag") == "select_static"}
    assert [opt["value"] for opt in settings_selects["planner_model"]["options"]] == [
        "gemini-4-neon",
        "gemini-4-carbon",
        "gemini-4-argon",
        "claude-opus-5-5",
    ]
    assert [opt["value"] for opt in settings_selects["executor_model"]["options"]] == [
        "gemini-4-hydrogen",
        "gemini-4-neon",
        "claude-sonnet-5-5",
    ]

    # 5. SystemHandler cascade, 1-click promotion from default, fail-closed validation, /model, /effort, /settings
    project = ProjectContext("proj-1", "Project 1", str(tmp_path))
    project.acp_tool_name = "agy"
    project.acp_model_name = "gemini-pro-orchestrator"
    committed: list[tuple[str, str | None]] = []
    pm = MagicMock()
    pm.get_project_for_chat.return_value = project
    pm.get_active_project.return_value = project

    def _commit(proj, *, tool_name: str, model_name: str | None, **_kw) -> bool:
        proj.set_acp_configuration(tool_name, model_name, chat_id="chat-1")
        committed.append((tool_name, model_name))
        return True

    pm.commit_acp_configuration.side_effect = _commit
    pm.commit_acp_programming_activation.side_effect = (
        lambda proj, *, tool_name, model_name, **_kw: _commit(proj, tool_name=tool_name, model_name=model_name)
    )
    agy_mgr = MagicMock()
    created_sess = SimpleNamespace(session_id="cli-orch-1", message_count=0, set_model=MagicMock(return_value=True))
    agy_mgr.get_session.return_value = created_sess
    agy_mgr.replace_session.return_value = SimpleNamespace(session=created_sess, created=True)
    agy_handler = MagicMock()
    agy_handler._get_session_manager.return_value = agy_mgr
    agy_handler.enter_mode.return_value = True

    ctx = MagicMock(
        settings=SimpleNamespace(acp_startup_timeout=20),
        mode_manager=ModeManager(),
        project_manager=pm,
        working_dir=str(tmp_path),
        handlers={"agy": agy_handler},
        scheduler=MagicMock(submit=MagicMock(side_effect=lambda _s, fn: fn(SimpleNamespace(run_id="r1")) or MagicMock())),
    )
    ctx.get_handler.side_effect = lambda name: ctx.handlers.get(name)
    system = SystemHandler(ctx)
    system._fetch_acp_models = lambda _t, **_kw: catalog  # type: ignore[method-assign]
    system.update_card = MagicMock(return_value=True)  # type: ignore[method-assign]
    system.reply_card = MagicMock(return_value="om_c")  # type: ignore[method-assign]
    system.reply_text = MagicMock()  # type: ignore[method-assign]
    system.reply_error = MagicMock()  # type: ignore[method-assign]

    # 1-click changing Executor Effort from "default" on /agy card promotes Planner Effort to "high"
    system.handle_acp_model_cascade_select(
        "m-card",
        "chat-1",
        "proj-1",
        {
            "action": action_ids.SELECT_ACP_MODEL_EFFORT,
            "tool_name": "agy",
            "model_group": "gemini-pro-orchestrator",
            "planner_model": "gemini-4-carbon",
            "model_profile": "default",
            "executor_model": "gemini-4-hydrogen",
            "model_effort": "default",
            "_option": "xhigh",
        },
    )
    assert "gemini-pro-orchestrator/high/xhigh" in system.update_card.call_args.args[1]

    # Pick planner model on /agy card -> redraws with claude-opus-5-5
    system.handle_acp_model_cascade_select(
        "m-card",
        "chat-1",
        "proj-1",
        {
            "action": action_ids.SELECT_ACP_ORCH_PLANNER_MODEL,
            "tool_name": "agy",
            "model_group": "gemini-pro-orchestrator",
            "planner_model": "gemini-4-carbon",
            "model_profile": "high",
            "executor_model": "gemini-4-hydrogen",
            "model_effort": "xhigh",
            "_option": "claude-opus-5-5",
        },
    )
    updated_card = system.update_card.call_args.args[1]
    assert custom_sel in updated_card

    # Reject disallowed role model on confirm
    system.handle_select_acp_model(
        "m-card",
        "chat-1",
        "proj-1",
        {
            "action": action_ids.SELECT_ACP_MODEL,
            "tool_name": "agy",
            "model_group": "gemini-pro-orchestrator",
            "planner_model": "gpt-oss-120b-medium",
            "model_profile": "high",
            "executor_model": "gemini-4-hydrogen",
            "model_effort": "xhigh",
            "use_default_model": False,
        },
    )
    system.reply_error.assert_called()

    # Confirm custom selection
    system.handle_select_acp_model(
        "m-card",
        "chat-1",
        "proj-1",
        {
            "action": action_ids.SELECT_ACP_MODEL,
            "tool_name": "agy",
            "model_group": "gemini-pro-orchestrator",
            "planner_model": "claude-opus-5-5",
            "model_profile": "high",
            "executor_model": "gemini-4-hydrogen",
            "model_effort": "xhigh",
            "use_default_model": False,
        },
    )
    assert committed[-1] == ("agy", custom_sel)

    # /effort preserves custom Planner/Executor models
    system.handle_effort_command("m-eff", "chat-1", "/effort medium high", project=project)
    assert committed[-1] == (
        "agy",
        "gemini-pro-orchestrator/claude-opus-5-5:medium/gemini-4-hydrogen:high",
    )

    # /settings cockpit changing executor model to claude-sonnet-5-5 updates and persists, while disallowed model fails closed
    system.reply_error.reset_mock()
    system.handle_tool_cockpit_action(
        "m-settings",
        "chat-1",
        "proj-1",
        {
            "action": action_ids.SETTINGS_SELECT_ORCH_EXECUTOR_MODEL,
            "tool_name": "agy",
            "model_group": "gemini-pro-orchestrator",
            "planner_model": "claude-opus-5-5",
            "model_profile": "medium",
            "executor_model": "gemini-4-hydrogen",
            "model_effort": "high",
            "_option": "gpt-oss-120b-medium",
        },
    )
    system.reply_error.assert_called_once()

    system.handle_tool_cockpit_action(
        "m-settings",
        "chat-1",
        "proj-1",
        {
            "action": action_ids.SETTINGS_SELECT_ORCH_EXECUTOR_MODEL,
            "tool_name": "agy",
            "model_group": "gemini-pro-orchestrator",
            "planner_model": "claude-opus-5-5",
            "model_profile": "medium",
            "executor_model": "gemini-4-hydrogen",
            "model_effort": "high",
            "_option": "claude-sonnet-5-5",
        },
    )
    assert committed[-1] == (
        "agy",
        "gemini-pro-orchestrator/claude-opus-5-5:medium/claude-sonnet-5-5:high",
    )

    # /settings 1-click change Executor Effort from bare default promotes Planner Effort to high
    project.set_acp_configuration("agy", "gemini-pro-orchestrator", chat_id="chat-1")
    system.handle_tool_cockpit_action(
        "m-settings-2",
        "chat-1",
        "proj-1",
        {
            "action": action_ids.SETTINGS_SELECT_MODEL_EFFORT,
            "tool_name": "agy",
            "model_group": "gemini-pro-orchestrator",
            "planner_model": "gemini-4-carbon",
            "model_profile": "default",
            "executor_model": "gemini-4-hydrogen",
            "model_effort": "default",
            "_option": "xhigh",
        },
    )
    assert committed[-1] == ("agy", "gemini-pro-orchestrator/high/xhigh")
