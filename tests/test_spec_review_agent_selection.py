import json
import re
from random import Random
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.card import CardBuilder
from src.card.actions.dispatch import (
    SPEC_REVIEW_SELECT_MODEL,
    SPEC_REVIEW_SELECT_TOOL,
    SPEC_REVIEW_USE_AUTO,
)
from src.engine_base import ReviewPerspective
from src.feishu.handlers.spec import SpecHandler
from src.project.context import ProjectContext
from src.spec_engine.models import SpecProjectStatus
from src.spec_engine.review import ReviewCircuitState
from src.spec_engine.review_agents import ReviewAgentBinding, assign_review_agents
from src.spec_engine.review_artifacts import ReviewArtifacts
from src.spec_engine.review_roles import ReviewRoleSpec
from src.spec_engine.review_strategy import AdaptiveRoleReviewStrategy, ReviewContext
from src.spec_engine.storage import SpecRunSummary
from src.worktree_engine.models import WorktreeSelectionItem
from src.worktree_engine.selection import WorktreeToolOption


def _role(role_id: str) -> ReviewRoleSpec:
    return ReviewRoleSpec(
        role_id=role_id,
        display_name=role_id.title(),
        category="software",
        mission="review",
        review_focus=["correctness"],
        must_check=["diff"],
        evidence_policy="blockers require evidence",
        base_perspective=ReviewPerspective.ARCHITECT,
    )


def _item(tool: str, model: str | None = None) -> WorktreeSelectionItem:
    return WorktreeSelectionItem(
        provider="acp",
        tool_name=tool,
        display_name=tool.title(),
        model_name=model,
        model_display_name=model,
    )


def test_assign_review_agents_covers_selected_pool_when_roles_cover_pool():
    roles = [_role(f"role_{i}") for i in range(5)]
    agents = [
        ReviewAgentBinding.from_selection_item(_item("coco", "m1")),
        ReviewAgentBinding.from_selection_item(_item("codex", "gpt-5.2")),
        ReviewAgentBinding.from_selection_item(_item("aiden", "m3")),
    ]

    assigned = assign_review_agents(roles, agents, rng=Random(8))

    used_keys = {binding.selection_key for binding in assigned.values()}
    assert used_keys == {agent.selection_key for agent in agents}
    assert len(used_keys) > 1


def test_adaptive_review_uses_selected_agents_for_role_sessions(monkeypatch):
    roles = [_role("architect"), _role("tester"), _role("security")]
    agents = [
        ReviewAgentBinding.from_selection_item(_item("coco", "m1")),
        ReviewAgentBinding.from_selection_item(_item("codex", "gpt-5.2")),
    ]
    captured: list[tuple[str, str | None]] = []

    class FakeReviewSession:
        def __init__(
            self,
            agent_type: str,
            cwd: str,
            model_name: str | None = None,
            startup_timeout: float | None = None,
        ):
            captured.append((agent_type, model_name))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def send_prompt(self, prompt: str, on_event=None, timeout: float = 240.0):
            role_id = re.search(r'"role_id": "([^"]+)"', prompt).group(1)
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "role_id": role_id,
                        "verdict": "PASS",
                        "summary": "ok",
                        "suggestions": [],
                    }
                )
            )

    monkeypatch.setattr("src.spec_engine.review_strategy.EphemeralReviewSession", FakeReviewSession)

    settings = SimpleNamespace(
        spec_review_dynamic_roles_enabled=True,
        spec_review_dynamic_roles_max=3,
        spec_review_total_roles_max=8,
        spec_review_failure_circuit_enabled=False,
        spec_review_max_parallel=3,
        spec_review_timeout=30,
        spec_completion_gate_enabled=False,
    )
    artifacts = ReviewArtifacts(cycle_number=1, requirement="build", cwd="/tmp")
    result = AdaptiveRoleReviewStrategy().run(
        ReviewContext(
            cycle=1,
            session=None,
            settings=settings,
            project=None,
            send_prompt_with_retry_fn=lambda *args, **kwargs: "",
            build_review_exception_diagnostics_fn=lambda *args, **kwargs: {},
            circuit=ReviewCircuitState(),
            artifacts=artifacts,
            role_plan_override=roles,
            review_agents=agents,
            review_agent_rng=Random(3),
        )
    )

    assert result.all_passed is True
    assert {agent for agent, _ in captured} == {"coco", "codex"}
    assert ("codex", "gpt-5.2") in captured
    assert ("coco", "m1") in captured
    assert {review.review_agent_label for review in result.reviews} == {"Coco / m1", "Codex / gpt-5.2"}


def test_adaptive_review_falls_back_to_default_agent_when_selected_agent_times_out(monkeypatch):
    roles = [_role("architect")]
    agents = [ReviewAgentBinding.from_selection_item(_item("codex", "gpt-5.2"))]
    captured: list[tuple[str, str | None]] = []

    class FakeReviewSession:
        def __init__(
            self,
            agent_type: str,
            cwd: str,
            model_name: str | None = None,
            startup_timeout: float | None = None,
        ):
            self.agent_type = agent_type
            self.model_name = model_name
            captured.append((agent_type, model_name))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def send_prompt(self, prompt: str, on_event=None, timeout: float = 240.0):
            if self.agent_type == "codex":
                raise TimeoutError("selected review agent timed out")
            return SimpleNamespace(
                text='{"role_id":"architect","verdict":"PASS","summary":"ok","suggestions":[]}'
            )

    monkeypatch.setattr("src.spec_engine.review_strategy.EphemeralReviewSession", FakeReviewSession)

    settings = SimpleNamespace(
        spec_review_dynamic_roles_enabled=True,
        spec_review_dynamic_roles_max=1,
        spec_review_total_roles_max=1,
        spec_review_failure_circuit_enabled=False,
        spec_review_max_parallel=1,
        spec_review_timeout=30,
        spec_completion_gate_enabled=False,
    )
    artifacts = ReviewArtifacts(cycle_number=1, requirement="build", cwd="/tmp")
    result = AdaptiveRoleReviewStrategy().run(
        ReviewContext(
            cycle=1,
            session=None,
            settings=settings,
            project=None,
            send_prompt_with_retry_fn=lambda *args, **kwargs: "",
            build_review_exception_diagnostics_fn=lambda *args, **kwargs: {},
            circuit=ReviewCircuitState(),
            artifacts=artifacts,
            role_plan_override=roles,
            review_agents=agents,
            review_agent_rng=Random(3),
            agent_type="coco",
            model_name="default-model",
        )
    )

    assert result.all_passed is True
    assert captured == [("codex", "gpt-5.2"), ("coco", "default-model")]


def _make_spec_handler() -> SpecHandler:
    ctx = MagicMock()
    ctx.settings.ref_note_enabled = False
    ctx.spec_engine_manager.get.return_value = None
    ctx.mode_manager.get_mode.return_value = None
    return SpecHandler(ctx)


def _collect_markdown_content(node) -> list[str]:
    if isinstance(node, dict):
        texts = []
        if node.get("tag") == "markdown":
            texts.append(str(node.get("content") or ""))
        for value in node.values():
            texts.extend(_collect_markdown_content(value))
        return texts
    if isinstance(node, list):
        texts = []
        for item in node:
            texts.extend(_collect_markdown_content(item))
        return texts
    return []


def _collect_buttons(node) -> list[dict]:
    if isinstance(node, dict):
        buttons = []
        if node.get("tag") == "button":
            buttons.append(node)
        for value in node.values():
            buttons.extend(_collect_buttons(value))
        return buttons
    if isinstance(node, list):
        buttons = []
        for item in node:
            buttons.extend(_collect_buttons(item))
        return buttons
    return []


def test_spec_start_shows_review_agent_selection_before_submitting_task():
    handler = _make_spec_handler()
    project = ProjectContext(project_id="p-spec", project_name="Spec", root_path="/tmp/spec")
    fake_tools = [
        {
            "provider": "acp",
            "tool_name": "coco",
            "display_name": "Coco",
            "description": "ACP Coco",
            "supports_model": True,
        }
    ]

    with patch.object(handler, "_get_available_spec_review_tools", return_value=fake_tools), \
         patch.object(handler, "reply_card", return_value="spec-select-card") as reply_card, \
         patch.object(handler, "update_card") as update_card, \
         patch.object(handler, "_submit_engine_task") as submit_task:
        handler.start_spec_engine("msg-spec", "chat-spec", "implement auth", project)

    submit_task.assert_not_called()
    update_card.assert_not_called()
    reply_card.assert_called_once()
    rendered = json.loads(reply_card.call_args.args[1])
    buttons = _collect_buttons(rendered)
    actions = [button["value"].get("action") for button in buttons]
    assert SPEC_REVIEW_SELECT_TOOL in actions
    assert SPEC_REVIEW_USE_AUTO in actions
    assert "worktree_select_tool" not in actions
    assert project.spec_review_selection_state.selection.pending_goal == "implement auth"


def test_spec_status_lists_cached_runs_with_restore_and_delete_buttons(monkeypatch):
    handler = _make_spec_handler()
    project = MagicMock()
    project.project_id = "p1"
    project.project_name = "ghostAp"
    project.root_path = "/repo/ghostAp"
    run = SpecRunSummary(
        run_id="run123",
        run_dir="/cache/repo/ghostAp/.spec_engine/run123",
        state_path="/cache/repo/ghostAp/.spec_engine/run123/state.json",
        status="paused",
        requirement="restore this spec",
        current_cycle=3,
        total_cycles=10,
        saved_at=1_700_000_000,
    )
    monkeypatch.setattr("src.feishu.handlers.spec.list_spec_runs", lambda root, settings: [run])
    with patch.object(handler, "reply_card") as reply_card:
        handler.show_spec_status("msg-status", "chat-status", project)

    card = json.loads(reply_card.call_args[0][1])
    text = "\n".join(_collect_markdown_content(card))
    assert "发现任务: `1` 个" in text
    buttons = _collect_buttons(card)
    values = [
        behavior.get("value")
        for button in buttons
        for behavior in button.get("behaviors", [])
        if behavior.get("type") == "callback"
    ]
    assert {
        "action": "spec_restore_run",
        "project_id": "p1",
        "deep_project_id": "/repo/ghostAp",
        "run_id": "run123",
    } in values
    assert {
        "action": "spec_delete_run",
        "project_id": "p1",
        "deep_project_id": "/repo/ghostAp",
        "run_id": "run123",
    } in values
    button_types = {button["text"]["content"]: button.get("type") for button in buttons}
    assert button_types["🔄 恢复"] == "primary"
    assert button_types["🗑 删除"] == "danger"
    panels = [element for element in card["body"]["elements"] if element.get("tag") == "collapsible_panel"]
    assert any("run123" in panel["header"]["title"]["content"] for panel in panels)


def test_spec_status_groups_restore_and_delete_buttons_with_each_cached_run(monkeypatch):
    handler = _make_spec_handler()
    project = MagicMock()
    project.project_id = "p1"
    project.project_name = "ghostAp"
    project.root_path = "/repo/ghostAp"
    restorable = SpecRunSummary(
        run_id="run123",
        run_dir="/cache/repo/ghostAp/.spec_engine/run123",
        state_path="/cache/repo/ghostAp/.spec_engine/run123/state.json",
        status="paused",
        requirement="restore this spec",
        current_cycle=3,
        total_cycles=10,
        saved_at=1_700_000_000,
    )
    orphaned = SpecRunSummary(
        run_id="orphan456",
        run_dir="/cache/repo/ghostAp/.spec_engine/orphan456",
        state_path=None,
        status="未知",
        requirement="directory only",
        current_cycle=0,
        total_cycles=0,
        saved_at=1_700_000_100,
    )
    monkeypatch.setattr("src.feishu.handlers.spec.list_spec_runs", lambda root, settings: [restorable, orphaned])
    with patch.object(handler, "reply_card") as reply_card:
        handler.show_spec_status("msg-status", "chat-status", project)

    card = json.loads(reply_card.call_args[0][1])
    panels = [
        panel
        for panel in card["body"]["elements"]
        if panel.get("tag") == "collapsible_panel"
    ]
    assert len(panels) == 2

    panel_by_run = {
        panel["header"]["title"]["content"]: panel
        for panel in panels
    }
    run123_panel = next(panel for title, panel in panel_by_run.items() if "run123" in title)
    orphan_panel = next(panel for title, panel in panel_by_run.items() if "orphan456" in title)

    run123_buttons = _collect_buttons(run123_panel)
    run123_values = [
        behavior.get("value")
        for button in run123_buttons
        for behavior in button.get("behaviors", [])
        if behavior.get("type") == "callback"
    ]
    assert [button["text"]["content"] for button in run123_buttons] == ["🔄 恢复", "🗑 删除"]
    assert {
        "action": "spec_restore_run",
        "project_id": "p1",
        "deep_project_id": "/repo/ghostAp",
        "run_id": "run123",
    } in run123_values
    assert {
        "action": "spec_delete_run",
        "project_id": "p1",
        "deep_project_id": "/repo/ghostAp",
        "run_id": "run123",
    } in run123_values

    orphan_buttons = _collect_buttons(orphan_panel)
    orphan_values = [
        behavior.get("value")
        for button in orphan_buttons
        for behavior in button.get("behaviors", [])
        if behavior.get("type") == "callback"
    ]
    assert [button["text"]["content"] for button in orphan_buttons] == ["🗑 删除"]
    assert {
        "action": "spec_delete_run",
        "project_id": "p1",
        "deep_project_id": "/repo/ghostAp",
        "run_id": "orphan456",
    } in orphan_values

    top_level_buttons = [
        element for element in card["body"]["elements"]
        if element.get("tag") == "column_set" and _collect_buttons(element)
    ]
    assert top_level_buttons == []


def test_spec_recover_with_run_id_falls_back_to_cached_run_restore(monkeypatch):
    handler = _make_spec_handler()
    project = MagicMock()
    project.root_path = "/repo/ghostAp"
    monkeypatch.setattr("src.feishu.handlers.spec.load_task_state", lambda task_id: None)
    monkeypatch.setattr(
        "src.feishu.handlers.spec.state_path_for_run",
        lambda root, settings, run_id: "/cache/repo/ghostAp/.spec_engine/run123/state.json",
    )
    monkeypatch.setattr("src.feishu.handlers.spec.os.path.isfile", lambda path: True)
    with patch.object(handler, "restore_spec_run") as restore, patch.object(handler, "reply_text") as reply_text:
        handler.recover_spec_task("msg-recover", "chat-recover", "run123", project)

    restore.assert_called_once_with("msg-recover", "chat-recover", "run123", project=project)
    reply_text.assert_not_called()


def test_restore_spec_run_resumes_interrupted_running_state(monkeypatch):
    handler = _make_spec_handler()
    project = MagicMock()
    project.project_id = "p1"
    project.root_path = "/repo/ghostAp"
    engine = MagicMock()
    engine.project.status = SpecProjectStatus.RUNNING
    engine.is_running = False
    handler.ctx.spec_engine_manager.load_or_create_from_state_file.return_value = engine
    monkeypatch.setattr(
        "src.feishu.handlers.spec.state_path_for_run",
        lambda root, settings, run_id: "/cache/repo/ghostAp/.spec_engine/run123/state.json",
    )
    monkeypatch.setattr("src.feishu.handlers.spec.os.path.isfile", lambda path: True)

    with patch.object(handler, "get_engine_name", return_value="Coco"), \
         patch.object(handler, "resume_spec_engine") as resume:
        handler.restore_spec_run("msg-restore", "chat-restore", "run123", project)

    assert engine.project.status == SpecProjectStatus.PAUSED
    engine.save_state.assert_called_once()
    resume.assert_called_once_with("msg-restore", "chat-restore", project)


def test_delete_spec_run_cache_deletes_cached_run(monkeypatch):
    handler = _make_spec_handler()
    project = MagicMock()
    project.root_path = "/repo/ghostAp"
    deleted: list[tuple[str, str]] = []

    def fake_delete(root: str, settings, run_id: str) -> bool:
        deleted.append((root, run_id))
        return True

    monkeypatch.setattr("src.feishu.handlers.spec.delete_spec_run", fake_delete)
    with patch.object(handler, "reply_text") as reply_text:
        handler.delete_spec_run_cache("msg-delete", "chat-delete", "run123", project)

    assert deleted == [("/repo/ghostAp", "run123")]
    reply_text.assert_called_once_with("msg-delete", "🧹 已删除 Spec 缓存任务: run123")


def test_delete_spec_run_cache_blocks_running_engine(monkeypatch):
    handler = _make_spec_handler()
    project = MagicMock()
    project.root_path = "/repo/ghostAp"
    engine = MagicMock()
    engine.project.project_id = "run123"
    engine.is_running = True
    handler.ctx.spec_engine_manager.get.return_value = engine
    delete = MagicMock(return_value=True)
    monkeypatch.setattr("src.feishu.handlers.spec.delete_spec_run", delete)

    with patch.object(handler, "reply_text") as reply_text:
        handler.delete_spec_run_cache("msg-delete", "chat-delete", "run123", project)

    delete.assert_not_called()
    assert "仍在运行" in reply_text.call_args.args[1]


def test_spec_review_selection_card_does_not_render_worktree_journey_copy():
    handler = _make_spec_handler()
    project = ProjectContext(project_id="p-render", project_name="Spec", root_path="/tmp/spec-render")
    fake_tools = [
        {
            "provider": "acp",
            "tool_name": "coco",
            "display_name": "Coco",
            "description": "ACP Coco",
            "supports_model": True,
        }
    ]

    with patch.object(handler, "_get_available_spec_review_tools", return_value=fake_tools), \
         patch.object(handler, "reply_card", return_value="spec-select-card") as reply_card:
        handler.start_spec_engine("msg-render", "chat-render", "implement spec", project)

    rendered = json.loads(reply_card.call_args.args[1])
    title = rendered["header"]["title"]["content"]
    text = "\n".join(_collect_markdown_content(rendered))

    assert "步骤 1/4" not in text
    assert "(1/4)" not in text
    assert "Worktree" not in text
    assert "等待目标" not in text
    assert "cycle" not in text
    assert "Spec Review" in title


def test_spec_review_model_grid_keeps_spec_review_actions():
    _, card_content = CardBuilder.build_spec_review_agent_select_card(
        tools=[
            {"id": "gpt-5.5", "name": "gpt-5.5"},
            {"id": "gpt-5.4", "name": "gpt-5.4"},
        ],
        selected=[{"selection_key": "acp:coco:m1", "display_label": "Coco / m1"}],
        project_id="p-spec",
        thread_root_id="root-spec-msg",
        select_action=SPEC_REVIEW_SELECT_MODEL,
        pending_tool="Codex",
    )
    rendered = json.loads(card_content)
    buttons = _collect_buttons(rendered)
    actions = [button["value"].get("action") for button in buttons]

    assert SPEC_REVIEW_SELECT_MODEL in actions
    assert "worktree_select_model" not in actions
    model_buttons = [button for button in buttons if button["value"].get("model_name") == "gpt-5.5"]
    assert model_buttons[0]["value"]["thread_root_id"] == "root-spec-msg"


def test_spec_review_tool_buttons_include_selected_model_signature_for_dedupe():
    tools = [
        {
            "provider": "acp",
            "tool_name": "coco",
            "display_name": "Coco",
            "supports_model": True,
        }
    ]
    _, empty_content = CardBuilder.build_spec_review_agent_select_card(
        tools=tools,
        selected=[],
        project_id="p-spec",
        thread_root_id="root-spec-msg",
        select_action=SPEC_REVIEW_SELECT_TOOL,
    )
    _, selected_content = CardBuilder.build_spec_review_agent_select_card(
        tools=tools,
        selected=[
            {
                "provider": "acp",
                "tool_name": "coco",
                "model_name": "Test-New-Thinking",
                "display_label": "Coco / Test-New-Thinking",
                "selection_key": "acp:coco:Test-New-Thinking",
            }
        ],
        project_id="p-spec",
        thread_root_id="root-spec-msg",
        select_action=SPEC_REVIEW_SELECT_TOOL,
    )

    empty_button = next(
        button
        for button in _collect_buttons(json.loads(empty_content))
        if button["value"].get("tool_name") == "coco"
    )
    selected_button = next(
        button
        for button in _collect_buttons(json.loads(selected_content))
        if button["value"].get("tool_name") == "coco"
    )

    assert empty_button["value"]["action"] == SPEC_REVIEW_SELECT_TOOL
    assert selected_button["value"]["action"] == SPEC_REVIEW_SELECT_TOOL
    assert empty_button["value"]["_selection_sig"] == "empty"
    assert selected_button["value"]["_selection_sig"] == "acp:coco:Test-New-Thinking"
    assert empty_button["value"] != selected_button["value"]


def test_spec_review_action_patches_existing_selection_card():
    handler = _make_spec_handler()
    project = ProjectContext(project_id="p-patch", project_name="Spec", root_path="/tmp/spec-patch")

    with patch.object(handler, "update_card", return_value=True) as update_card, \
         patch.object(handler, "reply_card") as reply_card, \
         patch.object(handler.renderer, "get_or_create_session") as get_session:
        handler._dispatch_spec_review_tool_select(
            message_id="selection-card-msg",
            chat_id="chat-spec",
            project=project,
            tools=[{"provider": "acp", "tool_name": "coco", "display_name": "Coco"}],
            thread_root_id="root-spec-msg",
        )

    update_card.assert_called_once()
    assert update_card.call_args.args[0] == "selection-card-msg"
    reply_card.assert_not_called()
    get_session.assert_not_called()


def test_spec_review_auto_starts_with_empty_review_agent_pool():
    handler = _make_spec_handler()
    project = ProjectContext(project_id="p-auto", project_name="Spec", root_path="/tmp/spec-auto")
    handler.ctx.project_manager.get_project_for_chat.return_value = project
    handler._spec_review_selection_controller().start_selection(project, goal="ship it")

    with patch.object(handler, "update_card", return_value=True) as update_card, \
         patch.object(handler, "_start_spec_engine_now") as start_now:
        handler.handle_spec_review_use_auto(
            "msg-auto",
            "chat-auto",
            project_id="p-auto",
            value={"thread_root_id": "root-spec-msg"},
        )

    update_card.assert_called_once()
    assert "正在启动" in json.loads(update_card.call_args.args[1])["header"]["title"]["content"]
    start_now.assert_called_once_with(
        "root-spec-msg",
        "chat-auto",
        "ship it",
        project,
        review_agents=[],
    )


def test_spec_review_finish_starts_with_selected_review_agent_pool():
    handler = _make_spec_handler()
    project = ProjectContext(project_id="p-finish", project_name="Spec", root_path="/tmp/spec-finish")
    handler.ctx.project_manager.get_project_for_chat.return_value = project
    ctrl = handler._spec_review_selection_controller()
    ctrl.start_selection(project, goal="build review")
    ctrl.select_tool(
        project,
        WorktreeToolOption(
            provider="acp",
            tool_name="codex",
            display_name="Codex",
            supports_model=True,
            model_optional=True,
        ),
    )
    ctrl.add_pending_item(project, model_name="gpt-5.2", model_display_name="GPT 5.2")

    with patch.object(handler, "update_card", return_value=True) as update_card, \
         patch.object(handler, "_start_spec_engine_now") as start_now:
        handler.handle_spec_review_finish_selection(
            "msg-finish",
            "chat-finish",
            project_id="p-finish",
            value={"thread_root_id": "root-spec-msg"},
        )

    update_card.assert_called_once()
    assert "正在启动" in json.loads(update_card.call_args.args[1])["header"]["title"]["content"]
    start_now.assert_called_once()
    assert start_now.call_args.args[:4] == (
        "root-spec-msg",
        "chat-finish",
        "build review",
        project,
    )
    kwargs = start_now.call_args.kwargs
    agents = kwargs["review_agents"]
    assert len(agents) == 1
    assert agents[0].agent_type == "codex"
    assert agents[0].model_name == "gpt-5.2"


def test_spec_engine_start_installs_selected_review_pool_before_execution():
    handler = _make_spec_handler()
    project = ProjectContext(project_id="p-engine", project_name="Spec", root_path="/tmp/spec-engine")
    engine = MagicMock()
    agent = ReviewAgentBinding(
        provider="acp",
        tool_name="codex",
        display_name="Codex",
        agent_type="codex",
        model_name="gpt-5.5",
        model_display_name="GPT 5.5",
        selection_key="acp:codex:gpt-5.5",
    )
    handler.ctx.spec_engine_manager.get.return_value = None
    handler.ctx.spec_engine_manager.get_or_create.return_value = engine

    with patch.object(handler, "add_reaction"), \
         patch.object(handler, "ensure_request_id", return_value="req-spec"), \
         patch.object(handler, "get_engine_name", return_value="Coco"), \
         patch.object(handler, "_submit_engine_task") as submit_task:
        handler._start_spec_engine_now(
            "root-spec-msg",
            "chat-engine",
            "ship it",
            project,
            review_agents=[agent],
        )

    engine.set_review_agent_pool.assert_called_once_with([agent])
    submit_task.assert_called_once()


def test_spec_engine_start_passes_selected_coco_model_to_engine():
    handler = _make_spec_handler()
    project = ProjectContext(project_id="p-engine-model", project_name="Spec", root_path="/tmp/spec-engine-model")
    project.acp_tool_name = "coco"
    project.acp_model_name = "deepseek-v4pro"
    engine = MagicMock()
    handler.ctx.spec_engine_manager.get.return_value = None
    handler.ctx.spec_engine_manager.get_or_create.return_value = engine

    with patch.object(handler, "add_reaction"), \
         patch.object(handler, "ensure_request_id", return_value="req-spec"), \
         patch.object(handler, "get_engine_name", return_value="Coco"), \
         patch.object(handler, "_submit_engine_task"):
        handler._start_spec_engine_now(
            "root-spec-msg",
            "chat-engine",
            "ship it",
            project,
        )

    handler.ctx.spec_engine_manager.get_or_create.assert_called_once_with(
        "chat-engine",
        "/tmp/spec-engine-model",
        engine_name="Coco",
        model_name="deepseek-v4pro",
    )


def test_spec_review_select_model_patches_current_selection_card():
    handler = _make_spec_handler()
    project = ProjectContext(project_id="p-model", project_name="Spec", root_path="/tmp/spec-model")
    handler.ctx.project_manager.get_project_for_chat.return_value = project
    ctrl = handler._spec_review_selection_controller()
    ctrl.start_selection(project, goal="build review")
    ctrl.select_tool(
        project,
        WorktreeToolOption(
            provider="acp",
            tool_name="codex",
            display_name="Codex",
            supports_model=True,
            model_optional=True,
        ),
    )

    with patch.object(handler, "update_card", return_value=True) as update_card, \
         patch.object(handler, "reply_card") as reply_card, \
         patch.object(handler.renderer, "get_or_create_session") as get_session:
        handler.handle_spec_review_select_model(
            "action-card-msg",
            "chat-model",
            project_id="p-model",
            value={"model_name": "gpt-5.5", "thread_root_id": "root-spec-msg"},
        )

    update_card.assert_called_once()
    assert update_card.call_args.args[0] == "action-card-msg"
    reply_card.assert_not_called()
    get_session.assert_not_called()
    rendered = json.loads(update_card.call_args.args[1])
    buttons = _collect_buttons(rendered)
    assert all(button["value"].get("action") != "worktree_select_model" for button in buttons)
