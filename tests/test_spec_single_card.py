from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.acp.models import ACPEvent, ACPEventType, PlanEntryInfo, PlanInfo, ToolCallInfo
from src.card.events import CardEvent
from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card
from src.card.session import CardSession
from src.card.session.config import SessionCallbacks, SessionConfig
from src.card.spec_adapter import SpecCardSession
from src.card.state.models import CardMetadata
from src.config import get_settings
from src.feishu.renderers.spec_renderer import SpecRenderer
from src.spec_engine.models import SpecPhase, SpecProject, SpecProjectStatus
from src.spec_engine.reporter import SpecReporter


@pytest.fixture
def spec_cards(monkeypatch, make_card_delivery):
    delivery, client = make_card_delivery()
    handler = MagicMock()
    handler.settings = get_settings()
    handler.ctx.spec_reporter = SpecReporter()
    handler.ctx.spec_engine_manager.snapshot.return_value = None
    renderer = SpecRenderer(handler)
    sessions = []

    def create_session(chat_id, message_id, metadata, **kwargs):
        session = CardSession(
            chat_id=chat_id,
            session_id=f"spec-test-{len(sessions)}",
            config=SessionConfig(metadata=metadata, budget=kwargs.get("budget") or RenderBudget()),
            delivery=delivery,
            callbacks=SessionCallbacks(notify_callback=lambda *_args: None),
        )
        sessions.append(session)
        return session

    monkeypatch.setattr(renderer, "create_session", create_session)
    monkeypatch.setattr(get_settings().card, "task_level_cards_enabled", True)
    callbacks = renderer.create_spec_callbacks("request", "chat", None)
    try:
        yield callbacks, renderer, sessions, client
    finally:
        callbacks.on_error("test cleanup")
        for session in sessions:
            session.close()


def test_spec_cycles_and_children_update_one_card_and_keep_history(spec_cards):
    callbacks, renderer, sessions, client = spec_cards
    callbacks.on_analyzing_start("写一篇城市观鸟指南")
    callbacks.on_cycle_start(1, 3)
    callbacks.on_phase_start(1, SpecPhase.BUILD)
    callbacks.on_phase_event(1, SpecPhase.BUILD, ACPEvent(
        event_type=ACPEventType.PLAN_UPDATE,
        plan=PlanInfo(entries=[
            PlanEntryInfo(content=f"检查项目 {index}", status="in_progress" if index == 46 else "completed")
            for index in range(47)
        ]),
    ))
    for index, role in enumerate(("思路策划", "鸟类专家")):
        callbacks.on_phase_event(1, SpecPhase.BUILD, ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(id=f"child-{index}", title="task", kind="agent", status="in_progress", content=role),
        ))
        callbacks.on_phase_event(1, SpecPhase.BUILD, ACPEvent(
            event_type=ACPEventType.TEXT_CHUNK, source_id=f"child-{index}", text=f"{role}的独立证据",
        ))
    callbacks.on_phase_event(1, SpecPhase.BUILD, ACPEvent(
        event_type=ACPEventType.TEXT_CHUNK, text="首轮结果已保留",
    ))
    callbacks.on_phase_done(1, SpecPhase.BUILD, "首轮结果已保留")
    callbacks.on_cycle_start(2, 3)
    callbacks.on_phase_start(2, SpecPhase.REVIEW)
    state = renderer.get_active_session().current.state
    rendered = json.dumps(render_card(state, RenderBudget())[0].to_feishu_json(), ensure_ascii=False)
    assert len(sessions) == len(client.created) == 1
    assert "首轮结果已保留" in rendered
    assert "独立证据" in rendered
    assert state.engine_ext.cycle_num == 2
    assert len(state.metadata.subagents) == 2
    assert not any(block.kind == "tool_call" and block.tool_name == "task" for block in state.blocks)


def test_spec_shows_inferred_experts_before_review_and_hides_raw_json(spec_cards):
    callbacks, renderer, sessions, _client = spec_cards
    callbacks.on_analyzing_start("评估观鸟活动")
    callbacks.on_cycle_start(1, 2)
    callbacks.on_phase_start(1, SpecPhase.SPEC)
    output = json.dumps({
        "goals": ["评估活动"], "acceptance_criteria": ["覆盖生态影响"],
        "required_experts": [{"role": "鸟类生态专家", "purpose": "检查栖息地干扰", "focus": ["生态"], "checks": ["核查距离"]}],
    }, ensure_ascii=False)
    callbacks.on_phase_event(1, SpecPhase.SPEC, ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=output))
    callbacks.on_phase_done(1, SpecPhase.SPEC, output)
    rendered = json.dumps(render_card(renderer.get_active_session().current.state, RenderBudget())[0].to_feishu_json(), ensure_ascii=False)
    assert "鸟类生态专家" in rendered and "检查栖息地干扰" in rendered
    assert "required_experts" not in rendered
    assert len(sessions) == 1


@pytest.mark.parametrize("status, terminal", [(SpecProjectStatus.COMPLETED, "completed"), (SpecProjectStatus.CANCELLED, "cancelled"), (SpecProjectStatus.FAILED, "failed")])
def test_spec_terminal_closes_main_card_and_rejects_late_output(spec_cards, status, terminal):
    callbacks, renderer, sessions, _client = spec_cards
    callbacks.on_analyzing_start("写作")
    callbacks.on_cycle_start(1, 1)
    callbacks.on_phase_start(1, SpecPhase.BUILD)
    project = SpecProject.create("writing", "/tmp/spec-writing")
    project.status = status
    callbacks.on_project_done(project)
    version = sessions[-1].state.version
    callbacks.on_phase_event(1, SpecPhase.BUILD, ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="late"))
    assert sessions[-1].state.terminal == terminal
    assert sessions[-1].state.version == version
    assert renderer.get_active_session() is None
    assert callbacks.on_phase_start.__self__._build_heartbeat is None


def test_spec_capacity_continuation_keeps_previous_content(make_card_delivery):
    delivery, _client = make_card_delivery()
    sessions = []
    budget = RenderBudget(byte_budget=6000, visible_chars=4000)

    def create(metadata):
        session = CardSession(
            chat_id="capacity", session_id=f"capacity-{len(sessions)}",
            config=SessionConfig(metadata=metadata, budget=budget), delivery=delivery,
            callbacks=SessionCallbacks(notify_callback=lambda *_args: None),
        )
        sessions.append(session)
        return session

    metadata = CardMetadata(engine_type="spec", mode_name="Spec", retain_full_history=True, programming_text_sections=True)
    adapter = SpecCardSession(create(metadata), budget=budget, session_factory=create)
    try:
        adapter.dispatch(CardEvent.started())
        adapter.dispatch(CardEvent.cycle_started(2, 5))
        adapter.dispatch(CardEvent.criteria_updated("核对事实来源", satisfied_count=1, total_count=2))
        adapter.dispatch(CardEvent.phase_started(2, "build", content="执行交付中"))
        for index in range(15):
            adapter.dispatch(CardEvent.text_delta(f"result-{index}", f"EVIDENCE-{index:02d} " + "详细依据 " * 25))
            adapter.dispatch(CardEvent.text_done(f"result-{index}"))
        assert adapter.current.state.engine_ext.cycle_num == 2
        assert adapter.current.state.engine_ext.phase_info == "build"
        assert adapter.current.state.engine_ext.criteria_satisfied == 1
        adapter.dispatch(CardEvent.phase_done(2, "build", "交付完成"))
        assert any(block.kind == "phase" and block.status == "completed" for block in adapter.current.state.blocks)
        adapter.finish()
        assert len(sessions) > 1
        assert all(session.closed for session in sessions)
        all_content = "\n".join(str(getattr(block, "content", "")) for session in sessions for block in session.state.blocks)
        for index in range(15):
            assert f"EVIDENCE-{index:02d}" in all_content
    finally:
        adapter.abort()
        for session in sessions:
            session.close()
