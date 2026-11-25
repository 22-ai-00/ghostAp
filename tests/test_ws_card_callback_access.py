"""Regression tests for WS card callback trust resolution and Workflow stop cards."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.access_control import IngressAccessPolicy
from src.card.actions.dispatch import (
    WF_SAVE_TEAM,
    WF_SELECT_PRESET,
    WORKFLOW_STOP_RUNNING,
)
from src.card.schema import require_card_json_2
from src.config import IngressAccessMode
from src.feishu.ws_client import FeishuWSClient
from src.trust.models import ManagedGroupOrigin
from src.trust.registry import ManagedGroupRegistry, single_owner_id
from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.renderer import (
    WorkflowGenerationRenderer,
    WorkflowProgressRenderer,
)

_VALIDATE_CARD_SCRIPT = (
    Path.home() / ".agents" / "skills" / "create-lark-card" / "scripts" / "validate_card.ts"
)


def _build_card_ws_client(
    tmp_path: Path,
    *,
    mode: IngressAccessMode,
    admin_user_ids: str,
    allowed_user_ids: str = "",
    allowed_chat_ids: str = "",
) -> tuple[FeishuWSClient, ManagedGroupRegistry]:
    """Construct a FeishuWSClient wired with a real ManagedGroupRegistry."""
    scheduler = MagicMock()
    scheduler._restart_gate = MagicMock()
    registry = ManagedGroupRegistry(tmp_path / "managed-groups.json")

    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client._build_task_scheduler", return_value=scheduler),
        patch("src.feishu.ws_client.ACPSessionManager"),
        patch("src.feishu.ws_client.IntentRecognizer"),
        patch("src.feishu.ws_client.ProjectManager"),
        patch("src.feishu.ws_client.MessageProjectMapper"),
        patch("src.feishu.ws_client.DeepEngineManager"),
        patch("src.feishu.ws_client.ProgressReporter"),
        patch("src.feishu.ws_client.SpecEngineManager"),
        patch("src.feishu.ws_client.SpecReporter"),
        patch("src.mode.ModeManager"),
        patch("src.workflow_engine.manager.WorkflowEngineManager"),
    ):
        settings = MagicMock()
        settings.app_id = "test_app_id"
        settings.app_secret = "test_app_secret"
        settings.admin_user_ids = admin_user_ids
        settings.allowed_user_ids = allowed_user_ids
        settings.allowed_chat_ids = allowed_chat_ids
        settings.ingress_access_mode = mode.value
        settings.admin_bootstrap_scope = "p2p_only"
        settings.streaming_enabled = False
        settings.thread_programming_enabled = False
        settings.task_scheduler_max_concurrent = 2
        settings.task_scheduler_per_key_concurrency = 1
        settings.system_command_concurrency = 10
        settings.message_cache_ttl = 300
        settings.message_cache_max_size = 1000
        settings.message_expire_seconds = 30
        settings.card.action_dedup_ttl = 1
        settings.card.action_dedup_max_size = 5000
        settings.spec_rate_limit_capacity = 100
        settings.spec_rate_limit_fill_rate = 50.0
        settings.spec_circuit_breaker_threshold = 10
        settings.spec_circuit_breaker_recovery = 5.0
        mock_get_settings.return_value = settings

        client = FeishuWSClient(MagicMock())
        client._main_bot_open_id = "ou_main_bot"
        client._project_manager.get_active_project.return_value = None
        client._managed_group_registry = registry
        client._managed_group_owner_id = single_owner_id(admin_user_ids)
        client._chat_lock_gate.check_card_action = MagicMock(return_value=False)
        client._ingress_access_policy_provider.swap(
            IngressAccessPolicy(
                admin_ids=frozenset(
                    item.strip() for item in admin_user_ids.split(",") if item.strip()
                ),
                allowed_user_ids=frozenset(
                    item.strip() for item in allowed_user_ids.split(",") if item.strip()
                ),
                allowed_chat_ids=frozenset(
                    item.strip() for item in allowed_chat_ids.split(",") if item.strip()
                ),
                mode=mode,
                admin_bootstrap_scope="p2p_only",
            )
        )
        return client, registry


def _card_event(
    *,
    event_id: str,
    message_id: str,
    chat_id: str,
    operator_id: str,
    action_value: dict[str, object],
    option: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        schema="2.0",
        header=SimpleNamespace(
            event_id=event_id,
            event_type="card.action.trigger",
            tenant_key="tenant_card",
        ),
        event=SimpleNamespace(
            operator=SimpleNamespace(
                open_id=operator_id,
                user_id=None,
                union_id=f"on_{operator_id}",
            ),
            context=SimpleNamespace(
                open_message_id=message_id,
                open_chat_id=chat_id,
            ),
            action=SimpleNamespace(
                tag="button" if option is None else "select_static",
                name="",
                value=dict(action_value),
                behaviors=[{"type": "callback", "value": dict(action_value)}],
                option=option,
                options=None,
                form_value=None,
                input_value=None,
            ),
        ),
    )


def _register_card_origin(
    client: FeishuWSClient,
    *,
    origin_message_id: str,
    card_message_id: str,
    chat_id: str,
    chat_type: str,
    sender_id: str,
) -> None:
    client._message_linker.register_origin(
        origin_message_id,
        request_id=f"req_{origin_message_id}",
        chat_id=chat_id,
        chat_type=chat_type,
        sender_id=sender_id,
    )
    client._message_linker.link_reply(origin_message_id, card_message_id)


def _find_buttons(node: object) -> list[dict]:
    found: list[dict] = []
    stack: list[object] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if cur.get("tag") == "button":
                found.append(cur)
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return found


def _validate_with_lark_card_script(card: dict) -> dict:
    if not _VALIDATE_CARD_SCRIPT.is_file() or shutil.which("node") is None:
        return {}
    proc = subprocess.run(
        ["node", str(_VALIDATE_CARD_SCRIPT), "--json"],
        input=json.dumps(card, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert proc.returncode == 0, (
        f"validate_card.ts failed (rc={proc.returncode}): "
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    parsed = json.loads(proc.stdout)
    assert parsed["valid"] is True, f"Card schema errors: {parsed['errors']}"
    assert parsed["errors"] == []
    return parsed


@pytest.mark.parametrize(
    ("chat_type", "chat_id", "expected_is_p2p"),
    [
        ("p2p", "oc_p2p_multi", True),
        ("group", "oc_group_multi", False),
    ],
)
def test_legacy_allow_all_multi_admin_dispatches_stop_and_wf_config(
    tmp_path: Path,
    chat_type: str,
    chat_id: str,
    expected_is_p2p: bool,
) -> None:
    """LEGACY_ALLOW_ALL + multiple ADMIN_USER_IDS must allow P2P and group callbacks."""
    admins = "ou_admin_alpha,ou_admin_beta"
    assert single_owner_id(admins) == ""
    client, _registry = _build_card_ws_client(
        tmp_path,
        mode=IngressAccessMode.LEGACY_ALLOW_ALL,
        admin_user_ids=admins,
        allowed_user_ids=admins,
    )
    try:
        operator_id = "ou_admin_beta"
        stop_origin = f"om_origin_stop_{chat_type}"
        stop_card_msg = f"om_card_stop_{chat_type}"
        _register_card_origin(
            client,
            origin_message_id=stop_origin,
            card_message_id=stop_card_msg,
            chat_id=chat_id,
            chat_type=chat_type,
            sender_id=operator_id,
        )

        stop_handler = MagicMock()
        client._action_handlers[WORKFLOW_STOP_RUNNING] = stop_handler
        stop_event = _card_event(
            event_id=f"evt_stop_{chat_type}",
            message_id=stop_card_msg,
            chat_id=chat_id,
            operator_id=operator_id,
            action_value={
                "action": WORKFLOW_STOP_RUNNING,
                "chat_id": chat_id,
                "project_id": "proj_1",
            },
        )
        stop_resp = client._handle_card_action(stop_event)
        assert stop_resp.toast.type == "info"
        assert stop_resp.toast.content == "正在停止 Workflow…"
        spec, async_cb = client._scheduler.submit.call_args.args
        assert spec.is_p2p is expected_is_p2p

        async_cb(SimpleNamespace(run_id="run_stop_1", spec=spec))
        stop_handler.assert_called_once()
        assert stop_handler.call_args.args[:3] == (stop_card_msg, chat_id, "proj_1")

        # Also verify WF_CONFIG_ACTIONS on both sync _handle_card_action and async worker
        updated_card = {
            "schema": "2.0",
            "config": {"wide_screen_mode": True, "update_multi": True},
            "body": {"elements": [{"tag": "markdown", "content": "已更新预设"}]},
        }
        wf_handler = client._handler_ctx.handlers["workflow"]
        wf_handler.handle_workflow_config_callback_sync = MagicMock(
            return_value=(updated_card, "info", "已更新预设")
        )
        cfg_origin = f"om_origin_cfg_{chat_type}"
        cfg_card_msg = f"om_card_cfg_{chat_type}"
        _register_card_origin(
            client,
            origin_message_id=cfg_origin,
            card_message_id=cfg_card_msg,
            chat_id=chat_id,
            chat_type=chat_type,
            sender_id=operator_id,
        )
        cfg_event = _card_event(
            event_id=f"evt_cfg_sync_{chat_type}",
            message_id=cfg_card_msg,
            chat_id=chat_id,
            operator_id=operator_id,
            action_value={
                "action": WF_SELECT_PRESET,
                "chat_id": chat_id,
                "project_id": "proj_1",
            },
            option="collaborative",
        )
        cfg_resp = client._handle_card_action(cfg_event)
        wf_handler.handle_workflow_config_callback_sync.assert_called_once()
        assert cfg_resp.toast.content == "已更新预设"
        assert cfg_resp.card.data == updated_card

        async_cfg_handler = MagicMock()
        client._action_handlers[WF_SAVE_TEAM] = async_cfg_handler
        async_cfg_event = _card_event(
            event_id=f"evt_cfg_async_{chat_type}",
            message_id=cfg_card_msg,
            chat_id=chat_id,
            operator_id=operator_id,
            action_value={
                "action": WF_SAVE_TEAM,
                "chat_id": chat_id,
                "project_id": "proj_1",
            },
        )
        client._process_card_action_async(
            async_cfg_event,
            task_ctx=SimpleNamespace(
                run_id="run_cfg_async",
                spec=SimpleNamespace(
                    sender_id=operator_id,
                    sender_union_id=f"on_{operator_id}",
                    tenant_key="tenant_card",
                    is_p2p=expected_is_p2p,
                ),
            ),
            effective_trust=client._resolve_effective_trust(
                sender_id=operator_id,
                chat_id=chat_id,
                chat_type=chat_type,
            ),
        )
        async_cfg_handler.assert_called_once()
    finally:
        client.close()


def test_enforced_multi_admin_allows_verified_p2p_and_denies_external_and_revoked_managed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ENFORCED + multi-admin allows verified P2P and strictly denies external/revoked."""
    admins = "ou_admin_alpha,ou_admin_beta"
    assert single_owner_id(admins) == ""
    p2p_chat = "oc_p2p_admin_beta"
    external_group = "oc_external_group"
    revoked_group = "oc_revoked_managed"
    operator_id = "ou_admin_beta"

    client, registry = _build_card_ws_client(
        tmp_path,
        mode=IngressAccessMode.ENFORCED,
        admin_user_ids=admins,
        allowed_user_ids=admins,
        # Even if external_group is listed in allowed_chat_ids, non-managed groups
        # stay denied in ENFORCED mode.
        allowed_chat_ids=f"{p2p_chat},{external_group},{revoked_group}",
    )
    try:
        # Provision and revoke a managed group in the durable registry.
        registry.register(
            chat_id=revoked_group,
            owner_id="ou_admin_alpha",
            origin=ManagedGroupOrigin.GHOSTAP_CREATED,
            receiving_bot_ref="test_app_id",
            project_id="proj_revoked",
            canonical_root_ref="/tmp/proj_revoked",
            created_at=datetime.now(UTC),
        )
        registry.begin_revoke(revoked_group, requested_at=datetime.now(UTC))
        registry.complete_revoke(revoked_group)
        assert registry.active_record(revoked_group) is None
        assert registry.record(revoked_group) is not None

        # 1) Verified P2P callback succeeds for both WORKFLOW_STOP_RUNNING and WF_CONFIG_ACTIONS.
        _register_card_origin(
            client,
            origin_message_id="om_p2p_origin",
            card_message_id="om_p2p_card",
            chat_id=p2p_chat,
            chat_type="p2p",
            sender_id=operator_id,
        )
        stop_handler = MagicMock()
        client._action_handlers[WORKFLOW_STOP_RUNNING] = stop_handler
        p2p_stop = _card_event(
            event_id="evt_enforced_p2p_stop",
            message_id="om_p2p_card",
            chat_id=p2p_chat,
            operator_id=operator_id,
            action_value={
                "action": WORKFLOW_STOP_RUNNING,
                "chat_id": p2p_chat,
                "project_id": "proj_p2p",
            },
        )
        p2p_resp = client._handle_card_action(p2p_stop)
        assert p2p_resp.toast.content == "正在停止 Workflow…"
        spec, async_cb = client._scheduler.submit.call_args.args
        assert spec.is_p2p is True
        async_cb(SimpleNamespace(run_id="run_p2p_stop", spec=spec))
        stop_handler.assert_called_once()

        wf_handler = client._handler_ctx.handlers["workflow"]
        wf_handler.handle_workflow_config_callback_sync = MagicMock(
            return_value=(None, "info", "ok")
        )
        p2p_cfg = _card_event(
            event_id="evt_enforced_p2p_cfg",
            message_id="om_p2p_card",
            chat_id=p2p_chat,
            operator_id=operator_id,
            action_value={
                "action": WF_SAVE_TEAM,
                "chat_id": p2p_chat,
                "project_id": "proj_p2p",
            },
        )
        cfg_resp = client._handle_card_action(p2p_cfg)
        assert cfg_resp.toast.content == "ok"
        wf_handler.handle_workflow_config_callback_sync.assert_called_once()

        # 2) Unknown external group callback is strictly denied with redacted warning.
        client._scheduler.submit.reset_mock()
        wf_handler.handle_workflow_config_callback_sync.reset_mock()
        stop_handler.reset_mock()
        _register_card_origin(
            client,
            origin_message_id="om_ext_origin",
            card_message_id="om_ext_card",
            chat_id=external_group,
            chat_type="group",
            sender_id=operator_id,
        )
        ext_event = _card_event(
            event_id="evt_enforced_ext",
            message_id="om_ext_card",
            chat_id=external_group,
            operator_id=operator_id,
            action_value={
                "action": WORKFLOW_STOP_RUNNING,
                "chat_id": external_group,
                "project_id": "proj_ext",
            },
        )
        with caplog.at_level(logging.WARNING):
            ext_resp = client._handle_card_action(ext_event)
        assert ext_resp.toast is None and ext_resp.card is None
        client._scheduler.submit.assert_not_called()
        assert any("卡片回调被信任策略拒绝" in rec.message for rec in caplog.records)
        assert external_group not in caplog.text

        # Direct async re-check also blocks external group in ENFORCED mode.
        client._process_card_action_async(
            ext_event,
            task_ctx=SimpleNamespace(
                run_id="run_ext_denied",
                spec=SimpleNamespace(
                    sender_id=operator_id,
                    sender_union_id="",
                    tenant_key="tenant_card",
                    is_p2p=False,
                ),
            ),
        )
        stop_handler.assert_not_called()

        # 3) Revoked managed group is strictly denied even with a stale P2P linker record.
        caplog.clear()
        _register_card_origin(
            client,
            origin_message_id="om_revoked_stale_origin",
            card_message_id="om_revoked_card",
            chat_id=revoked_group,
            chat_type="p2p",
            sender_id=operator_id,
        )
        revoked_event = _card_event(
            event_id="evt_enforced_revoked",
            message_id="om_revoked_card",
            chat_id=revoked_group,
            operator_id=operator_id,
            action_value={
                "action": WORKFLOW_STOP_RUNNING,
                "chat_id": revoked_group,
                "project_id": "proj_revoked",
            },
        )
        with caplog.at_level(logging.WARNING):
            revoked_resp = client._handle_card_action(revoked_event)
        assert revoked_resp.toast is None and revoked_resp.card is None
        client._scheduler.submit.assert_not_called()
        assert any("卡片回调被信任策略拒绝" in rec.message for rec in caplog.records)
        assert revoked_group not in caplog.text

        # Direct async invocation with forged is_p2p=True also stays denied on revoked managed group.
        client._process_card_action_async(
            revoked_event,
            task_ctx=SimpleNamespace(
                run_id="run_revoked_denied",
                spec=SimpleNamespace(
                    sender_id=operator_id,
                    sender_union_id="",
                    tenant_key="tenant_card",
                    is_p2p=True,
                ),
            ),
        )
        stop_handler.assert_not_called()
    finally:
        client.close()


def test_revoked_managed_group_remains_denied_even_in_legacy_allow_all(
    tmp_path: Path,
) -> None:
    """A revoked managed group must never fall back to open-group or P2P access."""
    client, registry = _build_card_ws_client(
        tmp_path,
        mode=IngressAccessMode.LEGACY_ALLOW_ALL,
        admin_user_ids="ou_owner_only",
        allowed_user_ids="ou_owner_only",
    )
    try:
        revoked_group = "oc_revoked_legacy"
        registry.register(
            chat_id=revoked_group,
            owner_id="ou_owner_only",
            origin=ManagedGroupOrigin.GHOSTAP_CREATED,
            receiving_bot_ref="test_app_id",
            project_id="proj_revoked_legacy",
            canonical_root_ref="/tmp/proj_revoked_legacy",
            created_at=datetime.now(UTC),
        )
        registry.begin_revoke(revoked_group, requested_at=datetime.now(UTC))
        registry.complete_revoke(revoked_group)
        _register_card_origin(
            client,
            origin_message_id="om_revoked_legacy_origin",
            card_message_id="om_revoked_legacy_card",
            chat_id=revoked_group,
            chat_type="p2p",
            sender_id="ou_owner_only",
        )
        stop_handler = MagicMock()
        client._action_handlers[WORKFLOW_STOP_RUNNING] = stop_handler
        event = _card_event(
            event_id="evt_revoked_legacy",
            message_id="om_revoked_legacy_card",
            chat_id=revoked_group,
            operator_id="ou_owner_only",
            action_value={"action": WORKFLOW_STOP_RUNNING, "chat_id": revoked_group},
        )

        resp = client._handle_card_action(event)
        assert resp.toast is None and resp.card is None
        client._scheduler.submit.assert_not_called()

        client._process_card_action_async(
            event,
            task_ctx=SimpleNamespace(
                run_id="run_revoked_legacy",
                spec=SimpleNamespace(
                    sender_id="ou_owner_only",
                    sender_union_id="",
                    tenant_key="tenant_card",
                    is_p2p=True,
                ),
            ),
        )
        stop_handler.assert_not_called()
    finally:
        client.close()


def test_active_and_pending_revoke_managed_groups_deny_non_owner_even_in_legacy_allow_all(
    tmp_path: Path,
) -> None:
    """Active & pending-revoke managed groups must block non-owner / stale-P2P fallbacks."""
    client, registry = _build_card_ws_client(
        tmp_path,
        mode=IngressAccessMode.LEGACY_ALLOW_ALL,
        admin_user_ids="ou_owner_only",
        allowed_user_ids="ou_owner_only,ou_other_admin",
    )
    try:
        for chat_id, pending_revoke in (
            ("oc_active_managed_legacy", False),
            ("oc_pending_revoke_legacy", True),
        ):
            registry.register(
                chat_id=chat_id,
                owner_id="ou_owner_only",
                origin=ManagedGroupOrigin.GHOSTAP_CREATED,
                receiving_bot_ref="test_app_id",
                project_id=f"proj_{chat_id}",
                canonical_root_ref=f"/tmp/{chat_id}",
                created_at=datetime.now(UTC),
            )
            if pending_revoke:
                registry.begin_revoke(chat_id, requested_at=datetime.now(UTC))
            actor = "ou_owner_only" if pending_revoke else "ou_other_admin"
            _register_card_origin(
                client,
                origin_message_id=f"om_{chat_id}_origin",
                card_message_id=f"om_{chat_id}_card",
                chat_id=chat_id,
                chat_type="p2p",
                sender_id=actor,
            )
            stop_handler = MagicMock()
            client._action_handlers[WORKFLOW_STOP_RUNNING] = stop_handler
            event = _card_event(
                event_id=f"evt_{chat_id}",
                message_id=f"om_{chat_id}_card",
                chat_id=chat_id,
                operator_id=actor,
                action_value={"action": WORKFLOW_STOP_RUNNING, "chat_id": chat_id},
            )
            resp = client._handle_card_action(event)
            assert resp.toast is None and resp.card is None
            client._scheduler.submit.assert_not_called()
            client._process_card_action_async(
                event,
                task_ctx=SimpleNamespace(
                    run_id=f"run_{chat_id}",
                    spec=SimpleNamespace(
                        sender_id=actor,
                        sender_union_id="",
                        tenant_key="tenant_card",
                        is_p2p=True,
                    ),
                ),
            )
            stop_handler.assert_not_called()
    finally:
        client.close()


def test_generation_and_running_stop_cards_schema_and_validate_script() -> None:
    """Generation & running stop cards satisfy Card 2.0 stop-button schema without confirm."""
    from src.feishu.handlers.workflow import WorkflowHandler

    pool = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Codex",
            effort="xhigh",
            role="worker",
        ),
    )
    gen_card = WorkflowGenerationRenderer(
        requirement="Fix stop button schema",
        agent_pool=pool,
        orchestrator_agent_id="A1",
        orchestrator_was_auto=True,
        chat_id="oc_stop_test",
        project_id="proj_stop_test",
    ).render(current_activity="正在生成脚本", elapsed_seconds=5)
    require_card_json_2(gen_card)
    assert gen_card["config"]["update_multi"] is True
    gen_buttons = _find_buttons(gen_card)
    assert len(gen_buttons) == 1
    gen_btn = gen_buttons[0]
    assert gen_btn["size"] == "medium"
    assert gen_btn["behaviors"] == [{"type": "callback", "value": gen_btn["value"]}]
    assert "confirm" not in gen_btn
    _validate_with_lark_card_script(gen_card)

    wf_project = WorkflowProject(
        workflow_id="wf_stop_test",
        name="stop-card-check",
        requirement="Validate running stop button",
        status=WorkflowStatus.RUNNING,
        started_at=time.time() - 10,
        phases=[
            PhaseProgress(
                title="Execution",
                started_at=time.time() - 10,
                agents=[
                    AgentProgress(
                        agent_id="A1",
                        label="worker-1",
                        tool="codex",
                        model="gpt-5.4",
                        effort="xhigh",
                        role="worker",
                        status=AgentStatus.RUNNING,
                        started_at=time.time() - 5,
                    )
                ],
            )
        ],
    )
    wf_project.metrics.total_agents = 1
    handler = WorkflowHandler.__new__(WorkflowHandler)
    frag = WorkflowProgressRenderer(wf_project).render_progress_card()
    handler._inject_workflow_stop_button(
        frag,
        "oc_stop_test",
        "proj_stop_test",
        is_running=True,
    )
    running_card = handler._build_workflow_card_from_renderer_data(frag)
    require_card_json_2(running_card)
    assert running_card["config"]["update_multi"] is True
    running_buttons = _find_buttons(running_card)
    assert len(running_buttons) == 1
    run_btn = running_buttons[0]
    assert run_btn["size"] == "medium"
    assert run_btn["behaviors"] == [{"type": "callback", "value": run_btn["value"]}]
    assert "confirm" not in run_btn
    assert run_btn["value"]["action"] == WORKFLOW_STOP_RUNNING
    _validate_with_lark_card_script(running_card)
