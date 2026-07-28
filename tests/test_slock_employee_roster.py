"""Tests for /employees (roster) command and /role add pick→confirm flow."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.autonomous.domain.enums import EmployeeState, WorkerType
from src.feishu.handlers.slock import SlockHandler
from src.slock_engine.slash_commands import (
    SlockCommandAction,
    is_slock_command,
    parse_slock_command,
)

# ---------------------------------------------------------------------------
# Command parsing tests
# ---------------------------------------------------------------------------


class TestEmployeesCommandParsing:
    def test_parse_employees(self):
        cmd = parse_slock_command("/employees")
        assert cmd.action == SlockCommandAction.EMPLOYEE_LIST

    def test_parse_roster_alias(self):
        cmd = parse_slock_command("/roster")
        assert cmd.action == SlockCommandAction.EMPLOYEE_LIST

    def test_is_slock_command_employees_global(self):
        result = is_slock_command("/employees", chat_id=None, manager=None)
        assert result.is_command is True

    def test_is_slock_command_roster_global(self):
        result = is_slock_command("/roster", chat_id=None, manager=None)
        assert result.is_command is True

    def test_is_slock_command_employees_in_dm(self):
        result = is_slock_command("/employees")
        assert result.is_command is True


# ---------------------------------------------------------------------------
# Roster handler tests
# ---------------------------------------------------------------------------


def _projected_employee(
    *,
    agent_id: str = "agt_test1",
    name: str = "柳七月",
    emoji: str = "🤖",
    tool: str = "codex",
    model: str = "gpt-4",
    state: EmployeeState = EmployeeState.ACTIVE,
    tenant_key: str = "tenant_a",
    member_groups: tuple[str, ...] = ("oc_g1",),
    bot_principal_id: str = "bot_test1",
    role: str = "coder",
    capabilities: tuple[str, ...] = ("coding", "testing"),
) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        name=name,
        emoji=emoji,
        tool=tool,
        model=model,
        state=state,
        worker_type=WorkerType.VISIBLE,
        tenant_key=tenant_key,
        member_groups=member_groups,
        bot_principal_id=bot_principal_id,
        role=role,
        capabilities=capabilities,
    )


def _handler_for_roster(
    *,
    hire_service=None,
    fire_service=None,
    membership_service=None,
    runtime_facade=None,
) -> SlockHandler:
    handler = object.__new__(SlockHandler)
    handler.ctx = SimpleNamespace(
        employee_hire_service=hire_service,
        employee_fire_service=fire_service,
        employee_membership_service=membership_service,
        employee_runtime_facade=runtime_facade,
        project_manager=MagicMock(),
        settings=SimpleNamespace(admin_user_ids=frozenset({"ou_admin"})),
    )
    handler.reply_text = MagicMock(return_value=True)
    handler.reply_card = MagicMock(return_value=True)
    handler.send_text_to_chat = MagicMock(return_value=True)
    return handler


class TestListCurrentTeamMembers:
    def test_filters_to_current_chat_and_marks_membership_and_runtime_state(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key",
            lambda: "tenant_a",
        )
        confirmed = _projected_employee(
            agent_id="agt_atlas",
            name="Atlas",
            member_groups=("oc_team",),
            role="architect",
            capabilities=("coding", "review"),
        )
        other_chat = _projected_employee(
            agent_id="agt_nova",
            name="Nova",
            member_groups=("oc_other",),
        )
        degraded = _projected_employee(
            agent_id="agt_drift",
            name="Drift",
            member_groups=("oc_team",),
        )
        membership_service = MagicMock()
        membership_service.is_degraded.side_effect = (
            lambda agent_id, _chat_id: agent_id == degraded.agent_id
        )
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = SimpleNamespace(
            employees={
                employee.agent_id: employee
                for employee in (confirmed, other_chat, degraded)
            }
        )
        runtime_facade = MagicMock()
        runtime_facade.list_employee_runtime_statuses.return_value = (
            SimpleNamespace(
                agent_id=confirmed.agent_id,
                bot_state="ready",
                actor_state="ready_hot",
                can_accept=True,
            ),
            SimpleNamespace(
                agent_id=other_chat.agent_id,
                bot_state="ready",
                actor_state="ready_hot",
                can_accept=True,
            ),
            SimpleNamespace(
                agent_id=degraded.agent_id,
                bot_state="ready",
                actor_state="ready_hot",
                can_accept=True,
            ),
        )
        handler = _handler_for_roster(
            hire_service=hire_service,
            membership_service=membership_service,
            runtime_facade=runtime_facade,
        )

        assert hasattr(handler, "list_current_team_members"), (
            "SlockHandler must expose the current-chat employee roster"
        )
        handler.list_current_team_members("om_1", "oc_team")

        payload = "\n".join(
            [
                *(
                    str(call.args[1])
                    for call in handler.reply_text.call_args_list
                ),
                *(
                    json.dumps(call.args[1], ensure_ascii=False)
                    for call in handler.reply_card.call_args_list
                ),
            ]
        )
        assert "Atlas" in payload
        assert "architect" in payload
        assert "coding, review" in payload
        assert "Bot READY / Agent READY_HOT" in payload
        assert "可接任务" in payload
        assert "Nova" not in payload
        assert "Drift" in payload
        assert "群关系待确认" in payload
        assert "/new-role" not in payload
        hire_service.synchronize_projection.assert_called_once_with()
        membership_service.is_degraded.assert_any_call("agt_atlas", "oc_team")
        runtime_facade.list_employee_runtime_statuses.assert_called_once_with(
            "tenant_a"
        )

    def test_confirmed_unavailable_employee_remains_visible(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key",
            lambda: "tenant_a",
        )
        recovering = _projected_employee(
            agent_id="agt_recovering",
            name="Recovering",
            state=EmployeeState.ACTION_REQUIRED,
            member_groups=("oc_team",),
            role="reviewer",
            capabilities=("review",),
        )
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = SimpleNamespace(
            employees={recovering.agent_id: recovering}
        )
        membership_service = MagicMock()
        membership_service.is_degraded.return_value = False
        runtime_facade = MagicMock()
        runtime_facade.list_employee_runtime_statuses.return_value = (
            SimpleNamespace(
                agent_id=recovering.agent_id,
                bot_state="stopped",
                actor_state="stopped",
                can_accept=False,
            ),
        )
        handler = _handler_for_roster(
            hire_service=hire_service,
            membership_service=membership_service,
            runtime_facade=runtime_facade,
        )

        handler.list_current_team_members("om_1", "oc_team")

        payload = json.dumps(
            handler.reply_card.call_args.args[1],
            ensure_ascii=False,
        )
        assert "Recovering" in payload
        assert "⚠️ 待恢复" in payload
        assert "Bot STOPPED / Agent STOPPED" in payload
        assert "不可接任务" in payload

    def test_membership_health_is_loaded_in_one_batch_snapshot(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key",
            lambda: "tenant_a",
        )
        ready = _projected_employee(
            agent_id="agt_ready",
            name="Ready",
            member_groups=("oc_team",),
        )
        degraded = _projected_employee(
            agent_id="agt_degraded",
            name="Degraded",
            member_groups=("oc_team",),
        )

        class BatchMembership:
            def __init__(self) -> None:
                self.calls = []

            def degraded_for(self, agent_ids, chat_id):
                self.calls.append((tuple(agent_ids), chat_id))
                return {
                    ready.agent_id: False,
                    degraded.agent_id: True,
                }

            def is_degraded(self, *_args):
                raise AssertionError("per-employee replay must not run")

        membership_service = BatchMembership()
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = SimpleNamespace(
            employees={
                ready.agent_id: ready,
                degraded.agent_id: degraded,
            }
        )
        handler = _handler_for_roster(
            hire_service=hire_service,
            membership_service=membership_service,
            runtime_facade=None,
        )

        handler.list_current_team_members("om_1", "oc_team")

        assert membership_service.calls == [
            ((ready.agent_id, degraded.agent_id), "oc_team")
        ]
        payload = json.dumps(
            handler.reply_card.call_args.args[1],
            ensure_ascii=False,
        )
        assert "Ready" in payload
        assert "Degraded" in payload
        assert "群关系待确认" in payload

    @pytest.mark.parametrize("malformed_health", [None, 0, ""])
    def test_malformed_batch_membership_health_fails_closed(
        self,
        monkeypatch,
        malformed_health,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key",
            lambda: "tenant_a",
        )
        employee = _projected_employee(
            agent_id="agt_batch_unknown",
            name="Batch Unknown",
            member_groups=("oc_team",),
        )

        class BatchMembership:
            def degraded_for(self, agent_ids, _chat_id):
                assert tuple(agent_ids) == (employee.agent_id,)
                return {employee.agent_id: malformed_health}

        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = SimpleNamespace(
            employees={employee.agent_id: employee}
        )
        runtime_facade = MagicMock()
        runtime_facade.list_employee_runtime_statuses.return_value = (
            SimpleNamespace(
                agent_id=employee.agent_id,
                bot_state="ready",
                actor_state="ready_hot",
                can_accept=True,
            ),
        )
        handler = _handler_for_roster(
            hire_service=hire_service,
            membership_service=BatchMembership(),
            runtime_facade=runtime_facade,
        )

        handler.list_current_team_members("om_1", "oc_team")

        payload = json.dumps(
            handler.reply_card.call_args.args[1],
            ensure_ascii=False,
        )
        assert "群关系待确认" in payload
        assert "不可接任务" in payload
        assert "已确认在群" not in payload

    @pytest.mark.parametrize("malformed_health", [None, 0, ""])
    def test_malformed_legacy_membership_health_fails_closed(
        self,
        monkeypatch,
        malformed_health,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key",
            lambda: "tenant_a",
        )
        employee = _projected_employee(
            agent_id="agt_legacy_unknown",
            name="Legacy Unknown",
            member_groups=("oc_team",),
        )

        class LegacyMembership:
            def is_degraded(self, _agent_id, _chat_id):
                return malformed_health

        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = SimpleNamespace(
            employees={employee.agent_id: employee}
        )
        runtime_facade = MagicMock()
        runtime_facade.list_employee_runtime_statuses.return_value = (
            SimpleNamespace(
                agent_id=employee.agent_id,
                bot_state="ready",
                actor_state="ready_hot",
                can_accept=True,
            ),
        )
        handler = _handler_for_roster(
            hire_service=hire_service,
            membership_service=LegacyMembership(),
            runtime_facade=runtime_facade,
        )

        handler.list_current_team_members("om_1", "oc_team")

        payload = json.dumps(
            handler.reply_card.call_args.args[1],
            ensure_ascii=False,
        )
        assert "群关系待确认" in payload
        assert "不可接任务" in payload
        assert "已确认在群" not in payload

    def test_missing_runtime_view_is_explicitly_unknown(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key",
            lambda: "tenant_a",
        )
        employee = _projected_employee(
            agent_id="agt_unknown",
            name="Unknown Runtime",
            member_groups=("oc_team",),
        )
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = SimpleNamespace(
            employees={employee.agent_id: employee}
        )
        membership_service = MagicMock()
        membership_service.is_degraded.return_value = False
        runtime_facade = MagicMock()
        runtime_facade.list_employee_runtime_statuses.return_value = ()
        handler = _handler_for_roster(
            hire_service=hire_service,
            membership_service=membership_service,
            runtime_facade=runtime_facade,
        )

        handler.list_current_team_members("om_1", "oc_team")

        payload = json.dumps(
            handler.reply_card.call_args.args[1],
            ensure_ascii=False,
        )
        assert "Unknown Runtime" in payload
        assert "运行状态未知" in payload
        assert "暂不可确认接单" in payload

    def test_missing_membership_health_service_keeps_persisted_member_visible(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key",
            lambda: "tenant_a",
        )
        employee = _projected_employee(
            agent_id="agt_membership_unknown",
            name="Persisted Member",
            member_groups=("oc_team",),
        )
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = SimpleNamespace(
            employees={employee.agent_id: employee}
        )
        handler = _handler_for_roster(
            hire_service=hire_service,
            membership_service=None,
            runtime_facade=None,
        )

        handler.list_current_team_members("om_1", "oc_team")

        payload = json.dumps(
            handler.reply_card.call_args.args[1],
            ensure_ascii=False,
        )
        assert "Persisted Member" in payload
        assert "群关系待确认" in payload
        assert "暂不可确认接单" in payload

    def test_employee_name_is_rendered_as_single_line_markdown_literal(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key",
            lambda: "tenant_a",
        )
        employee = _projected_employee(
            agent_id="agt_literal",
            name="**伪标题**\n[点击](https://invalid.example)",
            member_groups=("oc_team",),
        )
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = SimpleNamespace(
            employees={employee.agent_id: employee}
        )
        membership_service = MagicMock()
        membership_service.is_degraded.return_value = False
        handler = _handler_for_roster(
            hire_service=hire_service,
            membership_service=membership_service,
            runtime_facade=MagicMock(
                list_employee_runtime_statuses=MagicMock(return_value=())
            ),
        )

        handler.list_current_team_members("om_1", "oc_team")

        content = handler.reply_card.call_args.args[1]["elements"][0]["text"][
            "content"
        ]
        assert "\n[点击]" not in content
        assert r"\*\*伪标题\*\*" in content
        assert r"\[点击\]\(https://invalid.example\)" in content


class TestListEmployeesRoster:
    def test_runtime_facade_adds_independent_bot_actor_and_admission_state(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "tenant_a")
        employee = _projected_employee(agent_id="agt_atlas", name="Atlas")
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = SimpleNamespace(
            employees={employee.agent_id: employee}
        )
        runtime_facade = MagicMock()
        runtime_facade.list_employee_runtime_statuses.return_value = (
            SimpleNamespace(
                agent_id="agt_atlas",
                bot_state="ready",
                actor_state="ready_cold",
                can_accept=True,
            ),
        )
        handler = _handler_for_roster(
            hire_service=hire_service,
            runtime_facade=runtime_facade,
        )

        handler.list_employees_roster("om_1", "oc_chat")

        card = handler.reply_card.call_args.args[1]
        payload = json.dumps(card, ensure_ascii=False)
        assert "Bot READY / Agent READY_COLD" in payload
        assert "可接任务" in payload
        assert "employee_runtime_show_status" in payload

    def test_shows_all_visible_employees_with_state(self, monkeypatch):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "tenant_a")

        emp_active = _projected_employee(name="柳七月", state=EmployeeState.ACTIVE)
        emp_stuck = _projected_employee(
            agent_id="agt_test2",
            name="南宫婉",
            state=EmployeeState.ACTION_REQUIRED,
            member_groups=(),
        )
        emp_configuring = _projected_employee(
            agent_id="agt_test3",
            name="林黛玉",
            state=EmployeeState.CONFIGURING,
            member_groups=("oc_g1", "oc_g2"),
        )

        projection = SimpleNamespace(employees={
            "agt_test1": emp_active,
            "agt_test2": emp_stuck,
            "agt_test3": emp_configuring,
        })
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = projection

        handler = _handler_for_roster(hire_service=hire_service)
        handler.list_employees_roster("om_1", "oc_chat")

        handler.reply_card.assert_called_once()
        card = handler.reply_card.call_args.args[1]
        content = card["elements"][0]["text"]["content"]
        assert "柳七月" in content
        assert "✅ 就绪" in content
        assert "南宫婉" in content
        assert "⚠️ 待处理" in content
        assert "林黛玉" in content
        assert "⏳ 配置中" in content
        assert "群×2" in content

    def test_no_employees_shows_hint(self, monkeypatch):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "tenant_a")

        projection = SimpleNamespace(employees={})
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = projection

        handler = _handler_for_roster(hire_service=hire_service)
        handler.list_employees_roster("om_1", "oc_chat")

        handler.reply_text.assert_called_once()
        assert "/hire" in handler.reply_text.call_args.args[1]

    def test_no_hire_service_shows_fallback(self, monkeypatch):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "tenant_a")

        handler = _handler_for_roster(hire_service=None)
        handler.list_employees_roster("om_1", "oc_chat")

        handler.reply_text.assert_called_once()
        assert "尚未接入" in handler.reply_text.call_args.args[1]

    def test_no_tenant_key_shows_error(self, monkeypatch):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "")

        handler = _handler_for_roster(hire_service=MagicMock())
        handler.list_employees_roster("om_1", "oc_chat")

        handler.reply_text.assert_called_once()
        assert "租户" in handler.reply_text.call_args.args[1]

    def test_filters_other_tenant(self, monkeypatch):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "tenant_a")

        emp_other = _projected_employee(tenant_key="tenant_b", name="别人")
        projection = SimpleNamespace(employees={"agt_other": emp_other})
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = projection

        handler = _handler_for_roster(hire_service=hire_service)
        handler.list_employees_roster("om_1", "oc_chat")

        handler.reply_text.assert_called_once()
        assert "/hire" in handler.reply_text.call_args.args[1]

    def test_archived_employees_are_history_not_current_roster(self, monkeypatch):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "tenant_a")

        archived = _projected_employee(state=EmployeeState.ARCHIVED, member_groups=())
        projection = SimpleNamespace(employees={"agt_test1": archived})
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = projection
        handler = _handler_for_roster(hire_service=hire_service)

        handler.list_employees_roster("om_1", "oc_chat")

        handler.reply_card.assert_not_called()
        message = handler.reply_text.call_args.args[1]
        assert "没有在职员工" in message
        assert "历史归档 1 人" in message
        assert "/hire" in message

    def test_admin_dm_shows_app_identity_and_pending_confirmation(self, monkeypatch):
        from src.autonomous.provisioning.fire_state import (
            FireCleanupMode,
            FirePhase,
        )

        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key", lambda: "tenant_a"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_sender_id", lambda: "ou_admin"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_is_p2p", lambda: True
        )
        employee = _projected_employee(
            agent_id="agt_atlas",
            name="Atlas",
            state=EmployeeState.RETIRING,
        )
        projection = SimpleNamespace(
            employees={employee.agent_id: employee},
            bot_principals={
                "bot_test1": SimpleNamespace(
                    tenant_key="tenant_a",
                    agent_id="agt_atlas",
                    app_id="cli_atlas",
                    credential_ref="secret-must-not-render",
                )
            },
        )
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = projection
        hire_service.list_states.return_value = ()
        fire_service = MagicMock()
        fire_service.list_states.return_value = (
            SimpleNamespace(
                tenant_key="tenant_a",
                agent_id="agt_atlas",
                phase=FirePhase.ACTION_REQUIRED,
                cleanup_mode=FireCleanupMode.EXTERNAL_UNKNOWN,
                error_code="external_cleanup_authority_unavailable",
                external_disposition_confirmed=False,
                app_id="cli_atlas",
            ),
        )
        handler = _handler_for_roster(
            hire_service=hire_service,
            fire_service=fire_service,
        )

        handler.list_employees_roster("om_1", "oc_dm")

        card = handler.reply_card.call_args.args[1]
        content = card["elements"][0]["text"]["content"]
        assert "agt_atlas" in content
        assert "cli_atlas" in content
        assert (
            "/fire agt_atlas --confirm-app-disposed cli_atlas" in content
        )
        assert "secret-must-not-render" not in content
        assert "bot_test1" not in content

    def test_admin_roster_marks_empty_observed_manifest_unknown(self, monkeypatch):
        from src.autonomous.provisioning import lark_app

        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key", lambda: "tenant_a"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_sender_id", lambda: "ou_admin"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_is_p2p", lambda: True
        )
        employee = _projected_employee(agent_id="agt_legacy", name="Legacy")
        manifest_hash = lark_app.current_registration_manifest().fingerprint()
        projection = SimpleNamespace(
            employees={employee.agent_id: employee},
            bot_principals={
                "bot_test1": SimpleNamespace(
                    tenant_key="tenant_a",
                    agent_id="agt_legacy",
                    app_id="cli_legacy",
                    desired_manifest_hash=manifest_hash,
                    observed_manifest_hash="",
                    manifest_evidence_source="",
                )
            },
        )
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = projection
        hire_service.list_states.return_value = ()
        handler = _handler_for_roster(hire_service=hire_service)

        handler.list_employees_roster("om_1", "oc_dm")

        content = handler.reply_card.call_args.args[1]["elements"][0]["text"][
            "content"
        ]
        assert "权限证据：❔ 未知" in content
        assert "本地期望" in content
        assert "远端未知" in content
        assert "权限证据：✅ 当前 manifest" not in content
        card = handler.reply_card.call_args.args[1]
        assert json.dumps(card).count("slock_reauthorize_employee_app") == 1

    def test_admin_roster_marks_mismatched_manifest_as_drift_with_manual_action(
        self,
        monkeypatch,
    ):
        from src.autonomous.provisioning import lark_app

        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key", lambda: "tenant_a"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_sender_id", lambda: "ou_admin"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_is_p2p", lambda: True
        )
        employee = _projected_employee(agent_id="agt_drift", name="Drift")
        desired_hash = lark_app.current_registration_manifest().fingerprint()
        observed_hash = "sha256:" + "0" * 64
        projection = SimpleNamespace(
            employees={employee.agent_id: employee},
            bot_principals={
                "bot_test1": SimpleNamespace(
                    tenant_key="tenant_a",
                    agent_id="agt_drift",
                    app_id="cli_drift",
                    desired_manifest_hash=desired_hash,
                    observed_manifest_hash=observed_hash,
                    manifest_evidence_source=(
                        "lark_oapi.aregister_app/exact_app_id"
                    ),
                )
            },
        )
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = projection
        hire_service.list_states.return_value = ()
        handler = _handler_for_roster(hire_service=hire_service)

        handler.list_employees_roster("om_1", "oc_dm")

        content = handler.reply_card.call_args.args[1]["elements"][0]["text"][
            "content"
        ]
        assert "权限证据：⚠️ 漂移" in content
        assert "sha256:0000000000…" in content
        assert f"sha256:{desired_hash.removeprefix('sha256:')[:10]}…" in content
        assert "为原 App 原地重新授权并发布" in content
        assert "无需 `/fire`" in content
        assert "未验证飞书远端状态" in content

    def test_admin_roster_marks_only_trusted_exact_app_receipt_current(
        self,
        monkeypatch,
    ):
        from src.autonomous.provisioning import lark_app

        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key", lambda: "tenant_a"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_sender_id", lambda: "ou_admin"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_is_p2p", lambda: True
        )
        employee = _projected_employee(agent_id="agt_current", name="Current")
        manifest_hash = lark_app.current_registration_manifest().fingerprint()
        projection = SimpleNamespace(
            employees={employee.agent_id: employee},
            bot_principals={
                "bot_test1": SimpleNamespace(
                    tenant_key="tenant_a",
                    agent_id="agt_current",
                    app_id="cli_current",
                    desired_manifest_hash=manifest_hash,
                    observed_manifest_hash=manifest_hash,
                    manifest_evidence_source=(
                        "lark_oapi.aregister_app/exact_app_id"
                    ),
                )
            },
        )
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = projection
        hire_service.list_states.return_value = ()
        handler = _handler_for_roster(hire_service=hire_service)

        handler.list_employees_roster("om_1", "oc_dm")

        card = handler.reply_card.call_args.args[1]
        content = card["elements"][0]["text"]["content"]
        assert "权限证据：✅ 当前 manifest" in content
        assert "飞书官方原 App 授权回执已锚定" in content
        assert "slock_reauthorize_employee_app" not in json.dumps(card)

    def test_admin_reauthorization_action_starts_durable_existing_app_flow(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key", lambda: "tenant_a"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_sender_id", lambda: "ou_admin"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_is_p2p", lambda: True
        )
        hire_service = MagicMock()
        hire_service.request_manifest_reauthorization.return_value = SimpleNamespace(
            operation_id="manifestreauth_1"
        )
        handler = _handler_for_roster(hire_service=hire_service)

        handler.handle_card_action(
            "om_roster",
            "oc_admin_dm",
            "slock_reauthorize_employee_app",
            {"agent_id": "agt_test1"},
        )

        hire_service.request_manifest_reauthorization.assert_called_once_with(
            tenant_key="tenant_a",
            agent_id="agt_test1",
            request_id="om_roster",
        )
        assert "官方授权链接" in handler.send_text_to_chat.call_args.args[1]

    def test_stale_reauthorization_action_reports_already_committed(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key", lambda: "tenant_a"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_sender_id", lambda: "ou_admin"
        )
        monkeypatch.setattr("src.thread.manager.get_current_is_p2p", lambda: True)
        hire_service = MagicMock()
        hire_service.request_manifest_reauthorization.return_value = SimpleNamespace(
            operation_id="manifestreauth_1",
            phase=SimpleNamespace(value="committed"),
        )
        handler = _handler_for_roster(hire_service=hire_service)

        handler.handle_card_action(
            "om_roster",
            "oc_admin_dm",
            "slock_reauthorize_employee_app",
            {"agent_id": "agt_test1"},
        )

        message = handler.send_text_to_chat.call_args.args[1]
        assert "无需重复授权" in message
        assert "随后发送" not in message

    @pytest.mark.parametrize("sender_id,is_p2p", [("ou_other", True), ("ou_admin", False)])
    def test_reauthorization_action_requires_admin_dm(
        self,
        monkeypatch,
        sender_id,
        is_p2p,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key", lambda: "tenant_a"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_sender_id", lambda: sender_id
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_is_p2p", lambda: is_p2p
        )
        hire_service = MagicMock()
        handler = _handler_for_roster(hire_service=hire_service)

        handler.handle_card_action(
            "om_roster",
            "oc_admin_dm",
            "slock_reauthorize_employee_app",
            {"agent_id": "agt_test1"},
        )

        hire_service.request_manifest_reauthorization.assert_not_called()
        assert "仅允许配置管理员" in handler.send_text_to_chat.call_args.args[1]

    def test_admin_dm_no_app_confirmation_requires_prior_platform_check(
        self,
        monkeypatch,
    ):
        from src.autonomous.provisioning.fire_state import (
            FireCleanupMode,
            FirePhase,
        )

        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key", lambda: "tenant_a"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_sender_id", lambda: "ou_admin"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_is_p2p", lambda: True
        )
        employee = _projected_employee(
            agent_id="agt_no_app",
            name="Atlas",
            state=EmployeeState.RETIRING,
            bot_principal_id="",
        )
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = SimpleNamespace(
            employees={employee.agent_id: employee},
            bot_principals={},
        )
        hire_service.list_states.return_value = ()
        fire_service = MagicMock()
        fire_service.list_states.return_value = (
            SimpleNamespace(
                tenant_key="tenant_a",
                agent_id="agt_no_app",
                phase=FirePhase.ACTION_REQUIRED,
                cleanup_mode=FireCleanupMode.EXTERNAL_UNKNOWN,
                error_code="external_cleanup_authority_unavailable",
                external_disposition_confirmed=False,
                app_id="",
            ),
        )
        handler = _handler_for_roster(
            hire_service=hire_service,
            fire_service=fire_service,
        )

        handler.list_employees_roster("om_1", "oc_dm")

        content = handler.reply_card.call_args.args[1]["elements"][0]["text"][
            "content"
        ]
        assert "请先确认开放平台未创建应用" in content
        assert (
            "/fire agt_no_app --confirm-app-disposed NO_APP_FOUND" in content
        )
        assert "已确认" not in content

    @pytest.mark.parametrize(
        ("sender_id", "is_p2p"),
        (("ou_other", True), ("ou_admin", False)),
    )
    def test_sensitive_roster_details_require_admin_dm(
        self,
        monkeypatch,
        sender_id,
        is_p2p,
    ):
        monkeypatch.setattr(
            "src.thread.manager.get_current_tenant_key", lambda: "tenant_a"
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_sender_id", lambda: sender_id
        )
        monkeypatch.setattr(
            "src.thread.manager.get_current_is_p2p", lambda: is_p2p
        )
        employee = _projected_employee(
            agent_id="agt_private",
            name="Atlas",
            state=EmployeeState.RETIRING,
        )
        projection = SimpleNamespace(
            employees={employee.agent_id: employee},
            bot_principals={
                "bot_test1": SimpleNamespace(
                    tenant_key="tenant_a",
                    agent_id="agt_private",
                    app_id="cli_private",
                    credential_ref="cred_private",
                )
            },
        )
        hire_service = MagicMock()
        hire_service.synchronize_projection.return_value = projection
        fire_service = MagicMock()
        fire_service.list_states.return_value = ()
        handler = _handler_for_roster(
            hire_service=hire_service,
            fire_service=fire_service,
        )

        handler.list_employees_roster("om_1", "oc_chat")

        content = handler.reply_card.call_args.args[1]["elements"][0]["text"][
            "content"
        ]
        assert "Atlas" in content
        assert "agt_private" not in content
        assert "cli_private" not in content
        assert "--confirm-app-disposed" not in content
        assert "NO_APP_FOUND" not in content
        fire_service.list_states.assert_not_called()


# ---------------------------------------------------------------------------
# /role add pick→confirm flow tests
# ---------------------------------------------------------------------------


def _employee(*, agent_id="agt_employee", member_groups=()) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        name="柳七月",
        emoji="🤖",
        tool="codex",
        member_groups=member_groups,
    )


def _handler_for_card(*, service=None) -> tuple[SlockHandler, MagicMock]:
    handler = object.__new__(SlockHandler)
    handler.ctx = SimpleNamespace(
        employee_membership_service=service,
        project_manager=MagicMock(),
    )
    handler.reply_text = MagicMock(return_value=True)
    handler.reply_card = MagicMock(return_value=True)
    handler.update_card = MagicMock(return_value=True)
    handler.send_text_to_chat = MagicMock(return_value=True)
    handler._check_slock_permission = MagicMock(return_value=True)
    handler._change_employee_membership = MagicMock()

    engine = MagicMock()
    engine.channel = SimpleNamespace(channel_id="oc_team", owner_id="ou_owner")
    manager = MagicMock()
    manager.get_activated_engine.return_value = engine
    handler._get_engine_manager = MagicMock(return_value=manager)
    return handler, service


class TestRoleAddPickConfirm:
    def test_picker_uses_schema2_static_select_without_legacy_action(self, monkeypatch):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "tenant_a")

        service = MagicMock()
        service.list_employees.return_value = [_employee()]
        handler, _ = _handler_for_card(service=service)

        handler.add_role_to_group("om_1", "oc_team")

        handler.reply_card.assert_called_once()
        card = handler.reply_card.call_args.args[1]
        blob = json.dumps(card, ensure_ascii=False)
        assert '"tag": "action"' not in blob
        assert '"tag": "select_static"' in blob

    def test_pick_shows_confirm_card_no_mutation(self, monkeypatch):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "tenant_a")

        service = MagicMock()
        service.get_employee.return_value = _employee()
        handler, _ = _handler_for_card(service=service)

        handler.handle_card_action(
            "om_1", "oc_team", "slock_role_add_pick", {"_option": "agt_employee"}
        )

        handler.update_card.assert_called_once()
        card_json = handler.update_card.call_args.args[1]
        card = json.loads(card_json)
        assert '"tag": "action"' not in card_json
        assert '"tag": "column_set"' in card_json
        assert "确认" in card["header"]["title"]["content"]
        assert "柳七月" in card["elements"][0]["text"]["content"]
        handler._change_employee_membership.assert_not_called()

    def test_legacy_select_also_shows_confirm(self, monkeypatch):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "tenant_a")

        service = MagicMock()
        service.get_employee.return_value = _employee()
        handler, _ = _handler_for_card(service=service)

        handler.handle_card_action(
            "om_1", "oc_team", "slock_role_add_select", {"_option": "agt_employee"}
        )

        handler.update_card.assert_called_once()
        handler._change_employee_membership.assert_not_called()

    def test_confirm_triggers_mutation(self, monkeypatch):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "tenant_a")

        service = MagicMock()
        service.get_employee.return_value = _employee()
        handler, _ = _handler_for_card(service=service)

        handler.handle_card_action(
            "om_1", "oc_team", "slock_role_add_confirm",
            {"agent_id": "agt_employee", "chat_id": "oc_team"},
        )

        handler._change_employee_membership.assert_called_once()
        call_kwargs = handler._change_employee_membership.call_args.kwargs
        assert call_kwargs["operation"] == "add"

    def test_confirm_with_invalid_employee_shows_error(self, monkeypatch):
        monkeypatch.setattr("src.thread.manager.get_current_tenant_key", lambda: "tenant_a")

        service = MagicMock()
        service.get_employee.return_value = None
        handler, _ = _handler_for_card(service=service)

        handler.handle_card_action(
            "om_1", "oc_team", "slock_role_add_confirm",
            {"agent_id": "agt_nonexistent", "chat_id": "oc_team"},
        )

        handler.send_text_to_chat.assert_called_once()
        assert "失效" in handler.send_text_to_chat.call_args.args[1]
        handler._change_employee_membership.assert_not_called()
