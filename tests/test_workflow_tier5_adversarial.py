"""Tier-5 adversarial regression tests for Workflow config, persistence, journal, coalescer, and lifecycle."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.card.actions import dispatch as action_ids
from src.card.builders.workflow import WorkflowConfigEditorState
from src.feishu.handlers.workflow import WorkflowHandler, _WorkflowLifecycleOwner
from src.project import (
    ProjectManager,
    get_project_workflow_pending_requirement,
    get_project_workflow_team_config,
)
from src.utils.lock_order import _get_held, disable_lock_order_check, enable_lock_order_check
from src.workflow_engine.agent_pool import WorkflowAgentBinding, freeze_agent_pool
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.journal import WorkflowJournal
from src.workflow_engine.models import (
    AgentCallResult,
    PendingWorkflow,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.progress_coalescer import ProgressCoalescer
from src.workflow_engine.tool_registry import ToolCapability


def _make_handler(pm: ProjectManager, repo: Path) -> WorkflowHandler:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.settings = SimpleNamespace(
        admin_user_ids=[],
        default_reply_mode="reply",
        workflow_script_gen_timeout_s=10,
    )
    handler.ctx.project_manager = pm
    handler.reply_text = MagicMock(return_value="om_reply_txt")
    handler.reply_card = MagicMock(return_value="om_reply_card")
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock(return_value="om_send_card")
    handler.add_reaction = MagicMock()
    handler._resolve_origin = MagicMock(side_effect=lambda mid: mid)
    handler._ensure_topic_engine_context = MagicMock()
    handler.get_working_dir = MagicMock(return_value=str(repo))
    handler.get_engine_name = MagicMock(return_value="wf-tier5")
    return handler


def _extract_action_payload(card: dict, action: str) -> dict:
    stack: list[object] = [card]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            behaviors = node.get("behaviors")
            if isinstance(behaviors, list):
                for beh in behaviors:
                    if isinstance(beh, dict) and beh.get("type") == "callback":
                        val = beh.get("value")
                        if isinstance(val, dict) and val.get("action") == action:
                            return dict(val)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    raise AssertionError(f"Action {action!r} not found in card")


def _sample_caps() -> dict[str, ToolCapability]:
    return {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4",),
            efforts_by_model={"gpt-5.4": ("low", "high", "xhigh")},
            default_effort="high",
        ),
        "claude": ToolCapability(
            tool_name="claude",
            display_name="Claude",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("sonnet",),
            efforts_by_model={"sonnet": ("low", "high")},
            default_effort="high",
        ),
        "traex": ToolCapability(
            tool_name="traex",
            display_name="Traex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("c_o_new",),
            efforts_by_model={"c_o_new": ("low", "high")},
            profiles_by_model={"c_o_new": ("standard", "max")},
            default_effort="high",
        ),
        "coco": ToolCapability(
            tool_name="coco",
            display_name="Coco",
            available=True,
            supports_model=True,
            supports_effort=False,
            models=("coco-pro",),
        ),
    }


def test_config_cache_miss_with_card_snapshot_preserves_switched_preset(
    tmp_path: Path,
) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_preset")
    assert ok and proj is not None
    saved_cfg = {
        "version": 1,
        "preset_id": "collaborative",
        "orchestrator_agent_id": "A1",
        "bindings": [
            {
                "agent_id": "A1",
                "tool_name": "codex",
                "model_name": "gpt-5.4",
                "display_name": "Codex",
                "effort": "high",
                "role": "scout",
                "roles": ["scout"],
            },
            {
                "agent_id": "A2",
                "tool_name": "claude",
                "model_name": "sonnet",
                "display_name": "Claude",
                "effort": "high",
                "role": "worker",
                "roles": ["worker"],
            },
            {
                "agent_id": "A3",
                "tool_name": "traex",
                "model_name": "c_o_new",
                "display_name": "Traex",
                "effort": "high",
                "role": "verifier",
                "roles": ["verifier"],
            },
        ],
    }
    assert mgr.commit_workflow_team_configuration(proj, saved_cfg, chat_id="oc_preset")

    handler = _make_handler(mgr, tmp_path / "repo")
    caps = _sample_caps()
    handler._load_workflow_capabilities = MagicMock(
        return_value=(caps, {k: v.display_name for k, v in caps.items()})
    )

    handler.show_workflow_config("om_cmd", "oc_preset", proj)
    card_quick, toast_q, _ = handler.handle_workflow_config_callback_sync(
        "om_reply_card",
        "oc_preset",
        proj.project_id,
        {"action": action_ids.WF_SELECT_PRESET, "_option": "quick"},
    )
    assert toast_q == "info" and card_quick is not None
    role_payload = _extract_action_payload(card_quick, action_ids.WF_SELECT_ROLE)
    assert role_payload["preset_id"] == "quick"

    handler._wf_config_sessions.clear()
    handler._wf_config_message_index.clear()
    _card_after, toast_after, _ = handler.handle_workflow_config_callback_sync(
        "om_reply_card",
        "oc_preset",
        proj.project_id,
        {**role_payload, "_option": "verifier"},
    )
    assert toast_after == "info"
    restored = handler._get_wf_config_state(
        config_session_id=role_payload["config_session_id"],
        chat_id="oc_preset",
        project_id=proj.project_id,
        message_id="om_reply_card",
    )
    assert restored is not None
    assert restored.preset_id == "quick"
    assert len(restored.members) == 2
    assert restored.editor_role == "verifier"


def test_config_editor_clears_stale_effort_and_profile_for_non_effort_tool(
    tmp_path: Path,
) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_coco")
    assert ok and proj is not None
    handler = _make_handler(mgr, tmp_path / "repo")
    caps = _sample_caps()

    stale_state = WorkflowConfigEditorState(
        config_session_id="cfg_stale",
        chat_id="oc_coco",
        project_id=proj.project_id,
        preset_id="custom",
        members=(
            WorkflowAgentBinding(
                agent_id="A1",
                tool_name="codex",
                model_name="gpt-5.4",
                display_name="Codex",
                effort="high",
                role="worker",
            ),
        ),
        orchestrator_agent_id="A1",
        selected_agent_id="A1",
        editor_role="worker",
        editor_tool="coco",
        editor_model="coco-pro",
        editor_profile="max",
        editor_effort="high",
    )
    binding, err = handler._build_validated_editor_binding(
        stale_state,
        caps,
        agent_id="A1",
    )
    assert err is None
    assert binding is not None
    assert binding.tool_name == "coco"
    assert binding.effort is None
    assert binding.profile is None

    handler._load_workflow_capabilities = MagicMock(
        return_value=(caps, {k: v.display_name for k, v in caps.items()})
    )
    handler._put_wf_config_state(stale_state, message_id="om_coco")
    _card, toast_type, toast_msg = handler.handle_workflow_config_callback_sync(
        "om_coco",
        "oc_coco",
        proj.project_id,
        {"action": action_ids.WF_SAVE_TEAM, "config_session_id": "cfg_stale"},
    )
    assert toast_type == "success", toast_msg
    saved = get_project_workflow_team_config(proj, "oc_coco")
    assert saved is not None
    assert saved["bindings"][0]["tool_name"] == "coco"
    assert saved["bindings"][0]["effort"] is None


def test_concurrent_config_transitions_on_same_session_do_not_drop_updates(
    tmp_path: Path,
) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_race")
    assert ok and proj is not None
    handler = _make_handler(mgr, tmp_path / "repo")
    caps = _sample_caps()
    barrier = threading.Barrier(2)

    def load_with_barrier(_root: str | None = None):
        if getattr(load_with_barrier, "armed", False):
            barrier.wait(timeout=3.0)
        return caps, {k: v.display_name for k, v in caps.items()}

    handler._load_workflow_capabilities = load_with_barrier  # type: ignore[method-assign]
    handler.show_workflow_config("om_init", "oc_race", proj)
    initial = handler._get_wf_config_state(
        config_session_id="",
        chat_id="oc_race",
        project_id=proj.project_id,
        message_id="om_reply_card",
    )
    assert initial is not None
    cfg_sid = initial.config_session_id

    load_with_barrier.armed = True  # type: ignore[attr-defined]
    enable_lock_order_check(strict=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(
                handler.handle_workflow_config_callback_sync,
                "om_reply_card",
                "oc_race",
                proj.project_id,
                {"action": action_ids.WF_SELECT_EFFORT, "_option": "low", "config_session_id": cfg_sid},
            )
            f2 = pool.submit(
                handler.handle_workflow_config_callback_sync,
                "om_reply_card",
                "oc_race",
                proj.project_id,
                {"action": action_ids.WF_SELECT_ROLE, "_option": "verifier", "config_session_id": cfg_sid},
            )
            r1 = f1.result(timeout=5.0)
            r2 = f2.result(timeout=5.0)
        assert r1[1] != "error" and r2[1] != "error"
    finally:
        _get_held().clear()
        disable_lock_order_check()

    final = handler._get_wf_config_state(
        config_session_id=cfg_sid,
        chat_id="oc_race",
        project_id=proj.project_id,
        message_id="om_reply_card",
    )
    assert final is not None
    assert final.editor_effort == "low"
    assert final.editor_role == "verifier"

    # Also verify WF_SELECT_PRESET("quick") + WF_SELECT_ROLE("verifier") preserves both when preset enters lock first
    preset_entered = threading.Event()
    orig_get_state = handler._get_wf_config_state

    def get_state_with_sync(**kwargs):
        state = orig_get_state(**kwargs)
        preset_entered.set()
        return state

    handler._get_wf_config_state = get_state_with_sync  # type: ignore[method-assign]
    load_with_barrier.armed = False  # type: ignore[attr-defined]
    enable_lock_order_check(strict=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_preset = pool.submit(
                handler.handle_workflow_config_callback_sync,
                "om_reply_card",
                "oc_race",
                proj.project_id,
                {"action": action_ids.WF_SELECT_PRESET, "_option": "quick", "config_session_id": cfg_sid},
            )
            assert preset_entered.wait(timeout=3.0)
            f_role = pool.submit(
                handler.handle_workflow_config_callback_sync,
                "om_reply_card",
                "oc_race",
                proj.project_id,
                {"action": action_ids.WF_SELECT_ROLE, "_option": "scout", "config_session_id": cfg_sid},
            )
            assert f_preset.result(timeout=5.0)[1] != "error"
            assert f_role.result(timeout=5.0)[1] != "error"
    finally:
        _get_held().clear()
        disable_lock_order_check()

    after_preset = orig_get_state(
        config_session_id=cfg_sid,
        chat_id="oc_race",
        project_id=proj.project_id,
        message_id="om_reply_card",
    )
    assert after_preset is not None
    assert after_preset.preset_id == "quick"
    assert after_preset.editor_role == "scout"


def test_project_manager_rollback_on_save_failure_and_concurrent_reload(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "projects.json"
    mgr = ProjectManager(str(store_path))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_a")
    assert ok and proj is not None

    initial_cfg = {
        "version": 1,
        "preset_id": "quick",
        "orchestrator_agent_id": "A1",
        "bindings": [{"agent_id": "A1", "tool_name": "codex", "display_name": "Codex", "role": "worker"}],
    }
    assert mgr.commit_workflow_team_configuration(proj, initial_cfg, chat_id="oc_a") is True
    assert mgr.commit_workflow_pending_requirement(proj, "sid_1", "Initial req", chat_id="oc_a") is True

    # Simulate save failure -> rollback restores prior chat_state (including removing newly created chat state)
    with patch.object(mgr, "_save_projects", return_value=False):
        assert (
            mgr.commit_workflow_team_configuration(
                proj,
                {
                    "version": 1,
                    "preset_id": "deep_collab",
                    "orchestrator_agent_id": "A1",
                    "bindings": [{"agent_id": "A1", "tool_name": "codex", "display_name": "Codex", "role": "worker"}],
                },
                chat_id="oc_a",
            )
            is False
        )
        assert (
            mgr.commit_workflow_pending_requirement(proj, "sid_1", "Overwritten req", chat_id="oc_a")
            is False
        )
        assert mgr.commit_workflow_pending_requirement(proj, "sid_new", "Transient req", chat_id="oc_new") is False

    assert get_project_workflow_team_config(proj, "oc_a")["preset_id"] == "quick"
    assert get_project_workflow_pending_requirement(proj, "sid_1", "oc_a") == "Initial req"
    assert "oc_new" not in proj.chat_states

    # Concurrent team commits + ProjectManager reload stress
    errors: list[Exception] = []

    def writer(idx: int) -> None:
        try:
            payload = {
                "version": 1,
                "preset_id": "quick" if idx % 2 == 0 else "collaborative",
                "orchestrator_agent_id": "A1",
                "bindings": [
                    {
                        "agent_id": "A1",
                        "tool_name": "codex",
                        "display_name": "Codex",
                        "effort": "high" if idx % 2 == 0 else "low",
                        "role": "worker",
                    }
                ],
            }
            mgr.commit_workflow_team_configuration(proj, payload, chat_id="oc_a")
        except Exception as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            reloaded = ProjectManager(str(store_path))
            loaded_proj = reloaded.get_project_for_chat("p1", "oc_a")
            assert loaded_proj is not None
            cfg = get_project_workflow_team_config(loaded_proj, "oc_a")
            assert cfg is None or cfg["preset_id"] in {"quick", "collaborative"}
        except Exception as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(writer, idx) for idx in range(8)] + [pool.submit(reader) for _ in range(4)]
        for fut in futs:
            fut.result(timeout=5.0)
    assert errors == []


def test_journal_lru_eviction_reloads_from_disk_recovers_corrupt_and_clears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.workflow_engine.journal.workflow_journal_dir",
        lambda root_path: str(tmp_path / "journal_root"),
    )
    journal = WorkflowJournal(str(tmp_path / "repo"), "run-lru", max_entries=2)

    k1 = journal.compute_key("p1", "codex", "gpt-5.4", role="worker", effort="low")
    k2 = journal.compute_key("p1", "codex", "gpt-5.4", role="worker", effort="high")
    k3 = journal.compute_key("p1", "codex", "gpt-5.4", role="reviewer", effort="high", profile="max")
    assert len({k1, k2, k3}) == 3

    journal.store(k1, AgentCallResult(output="out-low"))
    journal.store(k2, AgentCallResult(output="out-high"))
    journal.store(k3, AgentCallResult(output="out-rev"))

    assert journal.stats()["evictions"] == 1
    assert k1 not in journal._cache
    assert journal.has(k1) is True

    # Reloads k1 from disk back into LRU memory
    reloaded = journal.get_cached(k1)
    assert reloaded is not None and reloaded.output == "out-low"
    assert k1 in journal._cache

    # Corrupt entry on disk returns None without crashing
    corrupt_path = journal._journal_dir / journal._index[k2]
    corrupt_path.write_text("{corrupt json", encoding="utf-8")
    journal._cache.pop(k2, None)
    assert journal.get_cached(k2) is None

    # clear() removes directory and resets stats
    journal.clear()
    assert not journal._journal_dir.exists()
    assert journal.stats() == {"total": 0, "hits": 0, "misses": 0, "evictions": 0}


def test_progress_coalescer_rate_limit_deferral_concurrency_and_context_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr("src.workflow_engine.progress_coalescer.time.monotonic", lambda: clock[0])

    delivered: list[str] = []
    with ProgressCoalescer(on_progress=lambda snap: delivered.append(str(snap)), debounce_s=0.05) as coalescer:
        coalescer.flush_immediate("first")
        assert delivered == ["first"]

        # Within 200ms guard -> deferred to background thread rather than immediate spam
        clock[0] = 100.05
        coalescer.flush_immediate("deferred-second")
        assert delivered == ["first"]

        # Enqueue latest snapshot concurrently and stop via context exit -> final snapshot delivered
        coalescer.enqueue("final-snapshot")
    assert "final-snapshot" in delivered


def test_wf_status_matrix_across_all_lifecycle_states_never_spams(
    tmp_path: Path,
) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_matrix")
    assert ok and proj is not None
    handler = _make_handler(mgr, tmp_path / "repo")
    handler.ctx.workflow_engine_manager = MagicMock()

    # 1. IDLE (no engine)
    handler.ctx.workflow_engine_manager.get.return_value = None
    handler.show_workflow_status("om_0", "oc_matrix", proj)
    handler.reply_text.assert_called_with("om_0", "当前没有 Workflow 任务。")

    # Real WorkflowEngine across remaining 5 states
    engine = WorkflowEngine.__new__(WorkflowEngine)
    engine._lock = threading.RLock()
    engine._state_manager = None
    engine._renderer_wf = MagicMock()
    engine._renderer_wf.render_compact_status.return_value = "compact status"
    engine._renderer_wf.render_progress_cards.return_value = [
        {"header": {"title": {"tag": "plain_text", "content": "Progress"}}, "elements": []}
    ]
    handler.ctx.workflow_engine_manager.get.return_value = engine

    # 2. GENERATING_SCRIPT (with live card -> patch, no send_card_to_chat)
    pool = freeze_agent_pool([WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex")])
    gen_owner = _WorkflowLifecycleOwner(
        session_key="gen_k",
        initiator_user_id="ou_1",
        chat_id="oc_matrix",
        project_id=proj.project_id,
        root_path=str(tmp_path / "repo"),
        card_message_id="om_live_gen",
    )
    engine._script_generation_owner = gen_owner
    engine._workflow_start_owner = None
    engine._project = WorkflowProject(
        requirement="gen req",
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingWorkflow(
            script_path=str(tmp_path / "s.js"),
            requirement="gen req",
            agent_pool=pool,
            orchestrator_agent_id="A1",
            created_at=time.time() - 3,
        ),
    )
    handler.show_workflow_status("om_1", "oc_matrix", proj)
    handler.update_card.assert_called_once()
    handler.send_card_to_chat.assert_not_called()

    # 3. RUNNING (with live card -> patch, no send_card_to_chat)
    handler.update_card.reset_mock()
    engine._script_generation_owner = None
    engine._workflow_start_owner = _WorkflowLifecycleOwner(
        session_key="run_k",
        initiator_user_id="ou_1",
        chat_id="oc_matrix",
        project_id=proj.project_id,
        root_path=str(tmp_path / "repo"),
        card_message_id="om_live_run",
    )
    engine._project.status = WorkflowStatus.RUNNING
    handler.show_workflow_status("om_2", "oc_matrix", proj)
    handler.update_card.assert_called_once()
    handler.send_card_to_chat.assert_not_called()

    # 4-6. COMPLETED / FAILED / CANCELLED with cleared live_card_id -> concise text, zero duplicate cards
    engine._workflow_start_owner = None
    for idx, terminal in enumerate(
        (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED),
        start=3,
    ):
        handler.reply_text.reset_mock()
        handler.send_card_to_chat.reset_mock()
        engine._project.status = terminal
        handler.show_workflow_status(f"om_{idx}", "oc_matrix", proj)
        handler.send_card_to_chat.assert_not_called()
        handler.reply_text.assert_called_once_with(f"om_{idx}", "compact status")


def test_ai_codegen_fallback_respects_shared_hard_deadline_and_cancellation(
    tmp_path: Path,
) -> None:
    mgr = ProjectManager(str(tmp_path / "projects.json"))
    ok, _, proj = mgr.create_project("p1", "Proj1", str(tmp_path / "repo"), chat_id="oc_codegen")
    assert ok and proj is not None
    handler = _make_handler(mgr, tmp_path / "repo")

    pool = freeze_agent_pool(
        [
            WorkflowAgentBinding(agent_id="A1", tool_name="codex", model_name="gpt-5.4", display_name="Codex"),
            WorkflowAgentBinding(agent_id="A2", tool_name="claude", model_name="sonnet", display_name="Claude"),
        ]
    )
    engine = WorkflowEngine.__new__(WorkflowEngine)
    engine._lock = threading.RLock()
    cancel_ev = threading.Event()
    owner = _WorkflowLifecycleOwner(
        session_key="codegen_k",
        initiator_user_id="ou_1",
        chat_id="oc_codegen",
        project_id=proj.project_id,
        root_path=str(tmp_path / "repo"),
        stop_event=cancel_ev,
    )
    engine._script_generation_owner = owner
    engine._project = WorkflowProject(
        requirement="build api",
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingWorkflow(
            script_path=str(tmp_path / "s.js"),
            requirement="build api",
            agent_pool=pool,
            orchestrator_agent_id="A1",
            engine_session_key="codegen_k",
        ),
    )

    from src.workflow_engine.storage import workflow_scripts_dir

    script_dir = Path(workflow_scripts_dir(str(tmp_path / "repo")))
    script_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(script_dir / "generated-workflow-t5.js")

    prompt_args: list[tuple[str, float, float]] = []

    class _TimeoutSession:
        def __init__(self, agent_type: str):
            self.agent_type = agent_type

        def send_prompt(self, _prompt, *, timeout, idle_timeout, **_kwargs):
            prompt_args.append((self.agent_type, float(timeout), float(idle_timeout)))
            if self.agent_type == "codex":
                return SimpleNamespace(stop_reason="timeout", text="")
            # Cancel during second member attempt
            cancel_ev.set()
            return SimpleNamespace(stop_reason="cancelled", text="")

        def cancel(self, **_kwargs):
            return True

        def close(self):
            return None

    with (
        patch("src.workflow_engine.tool_registry.get_available_tools", return_value={"codex": "Codex", "claude": "Claude"}),
        patch("src.agent_session.create_engine_session", side_effect=lambda **kw: _TimeoutSession(kw["agent_type"])),
    ):
        with pytest.raises(Exception) as exc_info:
            handler._generate_script_via_ai(
                "build api",
                str(tmp_path / "repo"),
                engine=engine,
                output_path=out_path,
                cancel_event=cancel_ev,
            )
    assert "cancel" in str(exc_info.value).lower()
    assert [item[0] for item in prompt_args] == ["codex", "claude"]
    assert all(0 < item[2] <= 120.0 and item[2] <= item[1] for item in prompt_args)
