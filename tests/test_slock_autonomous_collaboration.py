"""Regression tests for autonomous Slock multi-agent collaboration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.autonomous.team import TeamAdmissionError
from src.slock_engine.intent_router import IntentResult, IntentRouter
from src.slock_engine.models import AgentIdentity, SlockTask
from src.slock_engine.slash_commands import SlockCommandAction


def _make_handler():
    from src.feishu.handlers.slock import SlockHandler

    ctx = MagicMock()
    ctx.settings = MagicMock()
    ctx.settings.slock_autonomous_task_planning_enabled = True
    ctx.settings.slock_nli_timeout = 2.5
    ctx.settings.slock_nli_confidence_threshold = 0.6
    ctx.settings.slock_reply_mode = "direct"
    ctx.slock_engine_manager = MagicMock()
    ctx.employee_team_service = None

    handler = SlockHandler(ctx)
    handler.reply_card = MagicMock(return_value="plan-card-001")
    handler.reply_text = MagicMock()
    handler.send_card_to_chat = MagicMock()
    handler._check_assign_rate_limit = MagicMock(return_value=True)
    handler._execute_routed_message = MagicMock()
    handler._intent_router = MagicMock()
    handler._intent_router.fast_classify.return_value = IntentResult(
        action=SlockCommandAction.UNKNOWN,
        confidence=0.0,
        params={},
    )
    return handler


def _make_engine_for_plan():
    engine = MagicMock()
    engine.engine_name = "Slock"
    engine.root_path = "/tmp/ghostap"
    engine.channel = MagicMock()
    engine.channel.channel_id = "chat-slock"
    engine.registry.find_by_name.return_value = None
    engine.registry.list_agents.return_value = []
    engine.add_task.return_value = SlockTask(
        task_id="task-autonomous",
        content="实现登录功能并补充测试",
        created_in="chat-slock",
    )

    chain = MagicMock()
    chain.roles = ["planner", "coder", "reviewer", "tester"]
    engine._chain_manager.find_chain_for_task.return_value = chain

    plan = MagicMock()
    plan.plan_id = "plan-autonomous"
    plan.steps = []
    engine._collaboration_orchestrator.create_plan.return_value = plan
    engine._progress_tracker.set_overview_message_id = MagicMock()
    return engine


def test_plain_task_message_starts_collaboration_plan_instead_of_single_agent_route():
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine

    with patch(
        "src.slock_engine.card_templates.progress.build_collaboration_plan_card",
        return_value={"schema": "2.0", "header": {"title": {"content": "plan"}}},
    ):
        handler.handle_message("msg-001", "chat-slock", "实现登录功能并补充测试", None)

    engine.add_task.assert_called_once_with("实现登录功能并补充测试")
    engine._collaboration_orchestrator.create_plan.assert_called_once()
    handler.reply_card.assert_called_once()
    handler._execute_routed_message.assert_not_called()


def test_plain_task_prefers_durable_visible_employee_team_service():
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine
    handler.ctx.employee_team_service = MagicMock()
    handler.ctx.employee_team_service.start_task.return_value = MagicMock(
        run_id="teamrun_1234567890"
    )

    with patch(
        "src.thread.manager.get_current_tenant_key",
        return_value="tenant_1",
    ), patch(
        "src.thread.manager.get_current_sender_id",
        return_value="ou_user",
    ):
        handler.handle_message("om_task", "oc_team", "实现登录功能并补充测试", None)

    handler.ctx.employee_team_service.start_task.assert_called_once_with(
        tenant_key="tenant_1",
        message_id="om_task",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="实现登录功能并补充测试",
    )
    engine.add_task.assert_not_called()


@pytest.mark.parametrize(
    "text",
    [
        "修复任务队列死锁并补充测试",
        "创建任务队列监控并补充告警",
    ],
)
def test_plain_team_task_does_not_wait_for_llm_intent_classification(text):
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine
    handler.ctx.employee_team_service = MagicMock()
    handler.ctx.employee_team_service.start_task.return_value = MagicMock(
        run_id="teamrun_fast_path"
    )
    handler._intent_router.fast_classify.return_value = None
    handler._classify_with_timeout = MagicMock(
        side_effect=AssertionError("Team task must not invoke LLM NLI")
    )

    with patch(
        "src.thread.manager.get_current_tenant_key",
        return_value="tenant_1",
    ), patch(
        "src.thread.manager.get_current_sender_id",
        return_value="ou_user",
    ):
        handler.handle_message(
            "om_fast",
            "oc_team",
            text,
            None,
        )

    handler.ctx.employee_team_service.start_task.assert_called_once()
    handler._classify_with_timeout.assert_not_called()
    handler._execute_routed_message.assert_not_called()


def test_non_task_conversation_still_uses_nli_before_team_admission():
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine
    handler.ctx.employee_team_service = MagicMock()
    handler._intent_router.fast_classify.return_value = None
    handler._classify_with_timeout = AsyncMock(
        return_value=IntentResult(
            action=SlockCommandAction.CHITCHAT,
            confidence=0.95,
            params={},
        )
    )

    handler.handle_message("om_weather", "oc_team", "今天天气怎么样？", None)

    handler._classify_with_timeout.assert_awaited_once()
    handler.ctx.employee_team_service.start_task.assert_not_called()


def test_natural_language_command_still_uses_nli_before_team_admission():
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine
    handler.ctx.employee_team_service = MagicMock()
    handler._intent_router.fast_classify.return_value = None
    handler._classify_with_timeout = AsyncMock(
        return_value=IntentResult(
            action=SlockCommandAction.STATUS,
            confidence=0.95,
            params={},
        )
    )
    handler._dispatch_nli_intent = MagicMock()

    handler.handle_message("om_status", "oc_team", "现在团队状态怎么样？", None)

    handler._classify_with_timeout.assert_awaited_once()
    handler._dispatch_nli_intent.assert_called_once()
    handler.ctx.employee_team_service.start_task.assert_not_called()


@pytest.mark.parametrize(
    "text",
    [
        "你们都能做什么",
        "团队能做什么",
    ],
)
def test_team_capability_question_lists_members_without_starting_task(text):
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine
    handler.ctx.employee_team_service = MagicMock()
    handler._intent_router = IntentRouter()
    handler.list_current_team_members = MagicMock()
    handler._classify_with_timeout = MagicMock(
        side_effect=AssertionError("Capability query must use deterministic NLI")
    )

    handler.handle_message("om_capabilities", "oc_team", text, None)

    handler.list_current_team_members.assert_called_once_with(
        "om_capabilities",
        "oc_team",
    )
    handler.ctx.employee_team_service.start_task.assert_not_called()
    handler._classify_with_timeout.assert_not_called()
    handler._execute_routed_message.assert_not_called()


def test_employee_list_does_not_require_legacy_slock_engine():
    handler = _make_handler()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = None
    handler.ctx.employee_team_service = MagicMock()
    handler._intent_router = IntentRouter()
    handler.list_current_team_members = MagicMock()
    handler._send_no_engine_hint = MagicMock()

    handler.handle_message(
        "om_members",
        "oc_team",
        "团队成员现在都有谁",
        None,
    )

    handler.list_current_team_members.assert_called_once_with(
        "om_members",
        "oc_team",
    )
    handler._send_no_engine_hint.assert_not_called()
    handler.ctx.employee_team_service.start_task.assert_not_called()


@pytest.mark.parametrize(
    "text",
    [
        "创建角色",
        "如何创建角色？",
        "请帮我创建角色",
        "不要创建任务",
        "不要修复登录问题",
    ],
)
def test_action_word_control_command_is_not_stolen_by_team_fast_path(text):
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine
    handler.ctx.employee_team_service = MagicMock()
    handler._intent_router.fast_classify.return_value = None
    handler._classify_with_timeout = AsyncMock(
        return_value=IntentResult(
            action=SlockCommandAction.NEW_ROLE,
            confidence=0.95,
            params={"name": ""},
        )
    )
    handler._dispatch_nli_intent = MagicMock()

    handler.handle_message("om_role", "oc_team", text, None)

    handler._classify_with_timeout.assert_awaited_once()
    handler._dispatch_nli_intent.assert_called_once()
    handler.ctx.employee_team_service.start_task.assert_not_called()


def test_plain_task_without_ready_employee_is_not_reported_as_accepted():
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine
    handler.ctx.employee_team_service = MagicMock()
    handler.ctx.employee_team_service.start_task.side_effect = TeamAdmissionError(
        "no_active_team_employee"
    )

    with patch(
        "src.thread.manager.get_current_tenant_key",
        return_value="tenant_1",
    ), patch(
        "src.thread.manager.get_current_sender_id",
        return_value="ou_user",
    ):
        handler.handle_message("om_task", "oc_team", "大家能做个自我介绍嘛？", None)

    reply = handler.reply_text.call_args.args[1]
    assert "已受理" not in reply
    assert "自动恢复" in reply
    assert "/role add" in reply
    engine.add_task.assert_not_called()


def test_replayed_terminal_team_run_is_not_reported_as_accepted():
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine
    handler.ctx.employee_team_service = MagicMock()
    handler.ctx.employee_team_service.start_task.side_effect = TeamAdmissionError(
        "team_run_action_required"
    )

    with patch(
        "src.thread.manager.get_current_tenant_key",
        return_value="tenant_1",
    ), patch(
        "src.thread.manager.get_current_sender_id",
        return_value="ou_user",
    ):
        handler.handle_message("om_task", "oc_team", "重复投递", None)

    reply = handler.reply_text.call_args.args[1]
    assert "已受理" not in reply
    assert "已经结束" in reply


def test_autonomous_collaboration_keeps_explicit_mention_on_direct_route():
    handler = _make_handler()
    engine = _make_engine_for_plan()
    agent = AgentIdentity(agent_id="agent-coder", name="Coder", role="coder")
    engine.registry.find_by_name.return_value = agent
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine

    handler.handle_message("msg-002", "chat-slock", "@Coder 实现登录功能", None)

    handler._execute_routed_message.assert_called_once_with(
        engine, "msg-002", "chat-slock", "@Coder 实现登录功能", None, agent
    )
    engine.add_task.assert_not_called()


def test_multiple_visible_employee_mentions_start_one_team_run() -> None:
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine
    handler.ctx.employee_team_service = MagicMock()
    handler.ctx.employee_team_service.start_task.return_value = MagicMock(
        run_id="teamrun_mentions"
    )
    alpha = MagicMock(
        agent_id="agt_alpha",
        name="Alpha",
        role="coder",
        capabilities=("coding",),
        member_groups=("oc_team",),
    )
    beta = MagicMock(
        agent_id="agt_beta",
        name="Beta",
        role="reviewer",
        capabilities=("review",),
        member_groups=("oc_team",),
    )
    membership = MagicMock()
    employees = {"Alpha": alpha, "Beta": beta}
    membership.find_employee_by_name.side_effect = (
        lambda _tenant_key, name: employees.get(name)
    )
    membership.is_degraded.return_value = False
    handler.ctx.employee_membership_service = membership

    with patch(
        "src.thread.manager.get_current_tenant_key",
        return_value="tenant_1",
    ), patch(
        "src.thread.manager.get_current_sender_id",
        return_value="ou_user",
    ):
        handler.handle_message(
            "om_mentions",
            "oc_team",
            "@Alpha @Beta 一起评审这个实现",
            None,
        )

    handler.ctx.employee_team_service.dispatch_direct.assert_not_called()
    handler.ctx.employee_team_service.start_task.assert_called_once_with(
        tenant_key="tenant_1",
        message_id="om_mentions",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="@Alpha @Beta 一起评审这个实现",
    )
    handler._execute_routed_message.assert_not_called()


def test_structured_visible_employee_mention_supports_spaces_in_display_name() -> None:
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine
    handler.ctx.employee_team_service = MagicMock()
    employee = MagicMock(
        agent_id="agt_alice",
        name="Alice Reviewer",
        role="reviewer",
        capabilities=("review",),
        member_groups=("oc_team",),
    )
    membership = MagicMock()
    membership.find_employee_by_name.side_effect = (
        lambda _tenant_key, name: employee if name == "Alice Reviewer" else None
    )
    membership.is_degraded.return_value = False
    handler.ctx.employee_membership_service = membership

    with patch(
        "src.thread.manager.get_current_tenant_key",
        return_value="tenant_1",
    ), patch(
        "src.thread.manager.get_current_sender_id",
        return_value="ou_user",
    ), patch(
        "src.thread.manager.get_current_mentioned_names",
        return_value=("Alice Reviewer",),
    ):
        handler.handle_message(
            "om_structured_mention",
            "oc_team",
            "_at_1 请评审这个实现",
            None,
        )

    handler.ctx.employee_team_service.dispatch_direct.assert_called_once()
    target = handler.ctx.employee_team_service.dispatch_direct.call_args.kwargs[
        "target"
    ]
    assert target.agent_id == "agt_alice"
    handler.ctx.employee_team_service.start_task.assert_not_called()


def test_autonomous_collaboration_does_not_steal_shell_like_text():
    handler = _make_handler()
    engine = _make_engine_for_plan()
    handler.ctx.slock_engine_manager.get_activated_engine.return_value = engine

    handler.handle_message("msg-003", "chat-slock", "pytest tests/ -q", None)

    engine.add_task.assert_not_called()
    handler._execute_routed_message.assert_called_once_with(
        engine, "msg-003", "chat-slock", "pytest tests/ -q", None, target_agent=None
    )


def test_discussion_partner_selection_uses_engine_router_skill_keywords():
    from src.slock_engine.discussion_manager import DiscussionManager

    reviewer = AgentIdentity(agent_id="agent-reviewer", name="Reviewer", role="reviewer")
    security = AgentIdentity(agent_id="agent-security", name="Security", role="security")
    initiator = AgentIdentity(agent_id="agent-coder", name="Coder", role="coder")

    engine = MagicMock()
    engine.registry.list_agents.return_value = [initiator, reviewer, security]
    engine._router.extract_skill_keywords.return_value = ["security"]

    manager = DiscussionManager(engine=engine)

    selected = manager._find_best_discussion_partner(
        initiator,
        "需要检查登录鉴权 security risk",
        channel_id="chat-slock",
    )

    assert selected == "agent-security"


def test_collaboration_task_dispatch_passes_visible_callbacks_to_task_execution():
    from src.slock_engine.engine import SlockEngine

    engine = SlockEngine.__new__(SlockEngine)
    engine._progress_tracker = MagicMock()
    engine._card_send_fn = MagicMock(return_value="card-msg")
    engine._card_update_fn = MagicMock(return_value=True)
    engine._mouthpiece = MagicMock()
    engine._mouthpiece.format_card.return_value = {
        "schema": "2.0",
        "header": {"title": {"content": "Coder"}},
    }
    engine.execute_task = MagicMock(return_value="done")

    task = SlockTask(
        task_id="task-step",
        content="实现登录功能",
        created_in="chat-slock",
    )
    agent = AgentIdentity(agent_id="agent-coder", name="Coder", role="coder")

    SlockEngine._dispatch_collaboration_task(engine, task, agent)

    engine.execute_task.assert_called_once()
    callbacks = engine.execute_task.call_args.args[2]
    callbacks.on_card_send({"schema": "2.0"})
    callbacks.on_agent_done(agent, "step result")

    assert engine._card_send_fn.call_count == 2
    engine._mouthpiece.format_card.assert_called_once_with(
        agent,
        "step result",
        channel_id="chat-slock",
        task_id="task-step",
    )
