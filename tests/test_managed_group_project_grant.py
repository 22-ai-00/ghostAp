"""Task 0.9 vertical contracts for Project/Team managed-group activation."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.project.manager import ProjectCommitUncertainError, ProjectManager
from src.project_chat.lark_chat_client import CreateChatResult
from src.project_chat.service import ProjectChatService
from src.trust.models import ManagedGroupOrigin, ManagedGroupStatus
from src.trust.registry import ManagedGroupRegistry, RegistryCommitUncertainError


@pytest.fixture
def project_manager(tmp_path):
    return ProjectManager(storage_path=str(tmp_path / "projects.json"))


@pytest.fixture
def registry(tmp_path):
    return ManagedGroupRegistry(tmp_path / "managed-groups.json")


@pytest.fixture
def lark_client():
    from src.project_chat.lark_chat_client import ManagedChatValidation

    client = MagicMock()
    client.create_chat.return_value = CreateChatResult(
        chat_id="oc_project_group",
        name="project-dev",
    )
    client.delete_chat.return_value = True
    client.validate_managed_chat.return_value = ManagedChatValidation.VALID
    return client


def _project_service(project_manager, lark_client, registry, events):
    def reply(message_id, text, msg_type):
        events.append(("reply", message_id, msg_type, text))

    def send(chat_id, msg_type, text, root_id):
        active = registry.active_record(chat_id)
        assert active is not None
        events.append(("welcome", chat_id, active.revision))

    return ProjectChatService(
        project_manager=project_manager,
        lark_chat_client=lark_client,
        reply_fn=reply,
        send_to_chat_fn=send,
        managed_group_registry=registry,
        owner_id="ou_owner",
        receiving_bot_ref="cli_main_bot",
    )


def test_new_project_chat_is_active_before_welcome_without_allow_chat(
    tmp_path, project_manager, lark_client, registry
):
    root = str(tmp_path / "project")
    os.makedirs(root)
    events = []
    service = _project_service(project_manager, lark_client, registry, events)

    service.handle(
        message_id="om_new_chat",
        chat_id="oc_owner_p2p",
        sender_open_id="ou_owner",
        data={"name": "project", "path": root},
    )

    record = registry.active_record("oc_project_group")
    grant = registry.grant_for_chat("oc_project_group")
    assert record is not None and grant is not None
    assert record.status is ManagedGroupStatus.ACTIVE
    assert record.origin is ManagedGroupOrigin.GHOSTAP_CREATED
    assert record.project_id == grant.project_id
    assert record.canonical_root_ref == root
    assert events[-1][0] == "welcome"
    assert not any("allow-chat" in str(item) for item in events)
    assert lark_client.create_chat.call_args.kwargs["operation_id"].startswith(
        "new-chat:ou_owner:"
    )


@dataclass
class _FakeCreateChatResult:
    chat_id: str
    name: str


def _make_slock_handler(tmp_path, registry):
    from src.slock_engine.manager import SlockEngineManager

    ctx = MagicMock()
    settings = MagicMock()
    settings.slock_team_name_suffix = "[Slock]"
    settings.slock_memory_base_path = str(tmp_path / "slock")
    settings.slock_default_roles = ""
    settings.slock_nli_confidence_threshold = 0.7
    settings.slock_nli_timeout = 5
    settings.app_id = "cli_main_bot"
    settings.admin_user_ids = "ou_owner"
    ctx.settings = settings
    ctx.api_client_factory = MagicMock()
    ctx.project_manager = MagicMock()
    ctx.slock_engine_manager = SlockEngineManager(
        storage_base_path=settings.slock_memory_base_path
    )
    ctx.managed_group_registry = registry
    with patch("src.feishu.handlers.base.FeishuIMClient"):
        from src.feishu.handlers.slock import SlockHandler

        handler = SlockHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock(return_value="om_confirmation")
    handler.send_text_to_chat = MagicMock()
    handler.send_card_to_chat = MagicMock(return_value="om_welcome")
    handler.get_working_dir = MagicMock(return_value=str(tmp_path / "team-root"))
    return handler


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_new_team_uses_same_managed_group_registry(
    MockLarkChatClient, _sender, _session, tmp_path, registry
):
    handler = _make_slock_handler(tmp_path, registry)
    MockLarkChatClient.return_value.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_team_group",
        name="Alpha [Slock]",
    )
    observed = []

    def welcome(chat_id, *_args, **_kwargs):
        observed.append(registry.active_record(chat_id))
        return "om_welcome"

    handler.send_card_to_chat.side_effect = welcome
    handler.create_team("om_new_team", "oc_owner_p2p", "Alpha")

    record = registry.active_record("oc_team_group")
    grant = registry.grant_for_chat("oc_team_group")
    assert observed == [record]
    assert record is not None and grant is not None
    assert record.origin is ManagedGroupOrigin.GHOSTAP_CREATED
    assert record.project_id == "team:Alpha"
    assert handler.ctx.managed_group_registry is registry


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_team_registry_commit_uncertain_never_deletes_remote_chat(
    MockLarkChatClient, _sender, _session, tmp_path, registry
):
    handler = _make_slock_handler(tmp_path, registry)
    lark = MockLarkChatClient.return_value
    lark.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_team_uncertain",
        name="Uncertain [Slock]",
    )
    registry.activate = MagicMock(
        side_effect=RegistryCommitUncertainError(
            "parent fsync failed",
            committed=True,
        )
    )

    handler.create_team("om_new_team", "oc_owner_p2p", "Uncertain")

    lark.delete_chat.assert_not_called()
    handler.send_card_to_chat.assert_not_called()
    assert any(
        "结果不确定" in str(call)
        for call in handler.reply_text.call_args_list
    )


@pytest.mark.parametrize("delete_result", [False, None])
@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_team_bind_failure_persists_residual_name_block(
    MockLarkChatClient,
    _sender,
    _session,
    delete_result,
    tmp_path,
    registry,
):
    from src.slock_engine.manager import SlockEngineManager

    handler = _make_slock_handler(tmp_path, registry)
    lark = MockLarkChatClient.return_value
    lark.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_team_residual",
        name="Residual [Slock]",
    )
    lark.delete_chat.return_value = delete_result
    registry.bind_provision_chat = MagicMock(side_effect=OSError("disk full"))

    handler.create_team("om_new_team", "oc_owner_p2p", "Residual")

    lark.delete_chat.assert_not_called()
    restarted = SlockEngineManager(
        storage_base_path=handler.ctx.settings.slock_memory_base_path
    )
    assert restarted.reserve_team_name("Residual") is False


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_team_bind_commit_uncertain_never_deletes_remote_chat(
    MockLarkChatClient, _sender, _session, tmp_path, registry
):
    from src.slock_engine.manager import SlockEngineManager

    handler = _make_slock_handler(tmp_path, registry)
    lark = MockLarkChatClient.return_value
    lark.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_team_bind_uncertain",
        name="Bind Uncertain [Slock]",
    )
    registry.bind_provision_chat = MagicMock(
        side_effect=RegistryCommitUncertainError(
            "parent fsync failed",
            committed=True,
        )
    )

    handler.create_team("om_new_team", "oc_owner_p2p", "Bind Uncertain")

    lark.delete_chat.assert_not_called()
    restarted = SlockEngineManager(
        storage_base_path=handler.ctx.settings.slock_memory_base_path
    )
    assert restarted.reserve_team_name("Bind Uncertain") is True
    assert "结果不确定" in handler.reply_text.call_args.args[1]


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_team_create_outcome_unknown_retries_same_intent_inside_window(
    MockLarkChatClient, _sender, _session, tmp_path, registry
):
    from src.project_chat.errors import CreateChatError

    handler = _make_slock_handler(tmp_path, registry)
    lark = MockLarkChatClient.return_value
    lark.create_chat.side_effect = [
        CreateChatError("timeout"),
        _FakeCreateChatResult(
            chat_id="oc_team_retry",
            name="Retry [Slock]",
        ),
    ]

    handler.create_team("om_first", "oc_owner_p2p", "Retry")
    handler.create_team("om_retry", "oc_owner_p2p", "Retry")

    assert lark.create_chat.call_count == 2
    assert registry.active_record("oc_team_retry") is not None


@pytest.mark.parametrize("delete_result", [True, False, None])
def test_create_bind_registry_failure_never_reports_false_success(
    delete_result, tmp_path, project_manager, lark_client, registry
):
    root = str(tmp_path / "broken")
    os.makedirs(root)
    events = []
    service = _project_service(project_manager, lark_client, registry, events)
    lark_client.delete_chat.return_value = delete_result
    registry.activate = MagicMock(side_effect=OSError("disk full"))

    service.handle(
        message_id="om_broken",
        chat_id="oc_owner_p2p",
        sender_open_id="ou_owner",
        data={"name": "broken", "path": root},
    )

    assert not any(event[0] == "welcome" for event in events)
    assert len([event for event in events if event[0] == "reply"]) == 1
    lark_client.delete_chat.assert_not_called()
    assert registry.active_record("oc_project_group") is None
    project = project_manager.find_project_by_path(root, chat_id=None)
    assert project is None or not project.bound_chat_id
    provision_id = f"new-chat:ou_owner:broken:{os.path.normpath(root)}"
    restarted = ProjectManager(storage_path=str(tmp_path / "projects.json"))
    assert restarted.managed_group_residual(provision_id) == (
        "oc_project_group",
        "untrusted_retained",
    )


def test_project_registry_commit_uncertain_never_deletes_remote_chat(
    tmp_path, project_manager, lark_client, registry
):
    root = str(tmp_path / "uncertain")
    os.makedirs(root)
    events = []
    service = _project_service(project_manager, lark_client, registry, events)
    registry.activate = MagicMock(
        side_effect=RegistryCommitUncertainError(
            "parent fsync failed",
            committed=True,
        )
    )

    service.handle(
        message_id="om_uncertain",
        chat_id="oc_owner_p2p",
        sender_open_id="ou_owner",
        data={"name": "uncertain", "path": root},
    )

    lark_client.delete_chat.assert_not_called()
    project = project_manager.find_project_by_path(root, chat_id=None)
    assert project is not None
    assert project.bound_chat_id == "oc_project_group"
    assert not any(event[0] == "welcome" for event in events)


def test_project_bind_commit_uncertain_never_deletes_remote_chat(
    tmp_path, project_manager, lark_client, registry
):
    root = str(tmp_path / "bind-uncertain")
    os.makedirs(root)
    service = _project_service(project_manager, lark_client, registry, [])
    registry.bind_provision_chat = MagicMock(
        side_effect=RegistryCommitUncertainError(
            "parent fsync failed",
            committed=True,
        )
    )

    service.handle(
        message_id="om_bind_uncertain",
        chat_id="oc_owner_p2p",
        sender_open_id="ou_owner",
        data={"name": "bind-uncertain", "path": root},
    )

    lark_client.delete_chat.assert_not_called()
    operation_id = lark_client.create_chat.call_args.kwargs["operation_id"]
    assert project_manager.managed_group_residual(operation_id) == (
        "oc_project_group",
        "registry_bind_uncertain",
    )


def test_project_create_outcome_unknown_is_durable_and_not_abandoned(
    tmp_path, project_manager, lark_client, registry
):
    from src.project_chat.errors import (
        CreateChatError,
        CreateChatFailureDisposition,
    )

    root = str(tmp_path / "create-unknown")
    os.makedirs(root)
    service = _project_service(project_manager, lark_client, registry, [])
    lark_client.create_chat.side_effect = CreateChatError(
        "timeout",
        disposition=CreateChatFailureDisposition.OUTCOME_UNKNOWN,
    )

    service.handle(
        message_id="om_unknown",
        chat_id="oc_owner_p2p",
        sender_open_id="ou_owner",
        data={"name": "create-unknown", "path": root},
    )

    provision_id = (
        f"new-chat:ou_owner:{ProjectManager.generate_id('create-unknown')}:"
        f"{os.path.normpath(root)}"
    )
    assert registry.provision_create_state(provision_id) == "outcome_unknown"
    assert project_manager.managed_group_residual(provision_id) == (
        "outcome_unknown",
        "create_outcome_unknown",
    )


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_startup_consumes_new_chat_binding_saga_before_legacy_migration(
    client_cls, tmp_path, project_manager, lark_client, registry
):
    from src.slock_engine.manager import SlockEngineManager

    root = str(tmp_path / "new-chat-crash")
    os.makedirs(root)
    service = _project_service(project_manager, lark_client, registry, [])
    registry.activate = MagicMock(side_effect=SystemExit("simulated crash"))

    with pytest.raises(SystemExit, match="simulated crash"):
        service.handle(
            message_id="om_crash",
            chat_id="oc_owner_p2p",
            sender_open_id="ou_owner",
            data={"name": "new-chat-crash", "path": root},
        )

    restarted_projects = ProjectManager(
        storage_path=str(tmp_path / "projects.json")
    )
    assert len(restarted_projects.pending_managed_chat_binding_sagas()) == 1
    restarted_registry = ManagedGroupRegistry(
        tmp_path / "managed-groups.json"
    )
    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    client = _startup_reconciler(
        tmp_path,
        restarted_registry,
        manager,
        client_cls.return_value,
        restarted_projects,
    )

    client._reconcile_managed_groups_before_slock_restore()

    record = restarted_registry.active_record("oc_project_group")
    assert record is not None
    assert record.origin is ManagedGroupOrigin.GHOSTAP_CREATED
    assert restarted_projects.pending_managed_chat_binding_sagas() == ()
    client_cls.return_value.validate_managed_chat.assert_not_called()


def test_project_migration_requires_membership_and_receiving_bot_validation(
    tmp_path, project_manager, registry
):
    root = str(tmp_path / "legacy")
    os.makedirs(root)
    ok, _, project = project_manager.create_project(
        project_id="legacy",
        project_name="legacy",
        root_path=root,
        chat_id="oc_legacy",
    )
    assert ok and project is not None
    project.bound_chat_id = "oc_legacy"
    project.bound_chat_created_at = 123.0
    project_manager._save_projects()
    candidate = project.managed_group_migration_candidate(
        owner_id="ou_owner",
        receiving_bot_ref="cli_main_bot",
    )
    assert candidate is not None

    assert registry.import_candidate(candidate, validator=lambda _: False) is None
    assert registry.active_record("oc_legacy") is None
    imported = registry.import_candidate(candidate, validator=lambda _: True)
    assert imported is not None
    assert registry.active_record("oc_legacy") == imported[0]


def test_project_handler_injects_shared_registry_identity(registry):
    from src.feishu.handlers.project import ProjectHandler

    ctx = MagicMock()
    ctx.managed_group_registry = registry
    ctx.managed_group_owner_id = "ou_owner"
    ctx.managed_group_receiving_bot_ref = "cli_main_bot"
    ctx.api_client_factory = MagicMock()
    ctx.project_manager = MagicMock()
    ctx.handlers = {}
    ctx.managers = {}
    with (
        patch("src.feishu.handlers.base.FeishuIMClient"),
        patch("src.thread.manager.get_current_sender_id", return_value="ou_owner"),
        patch("src.project_chat.ProjectChatService") as service_cls,
    ):
        handler = ProjectHandler(ctx)
        handler.get_working_dir = MagicMock(return_value="/srv/project")
        handler.handle_new_chat_project("om_1", "oc_p2p", {"name": "project"})

    assert service_cls.call_args.kwargs["managed_group_registry"] is registry
    assert service_cls.call_args.kwargs["owner_id"] == "ou_owner"
    assert service_cls.call_args.kwargs["receiving_bot_ref"] == "cli_main_bot"


def test_registry_composition_uses_project_storage_parent(tmp_path):
    from src.feishu.ws_client import (
        _build_managed_group_registry,
        _configured_managed_group_owner_id,
    )

    project_manager = ProjectManager(storage_path=str(tmp_path / "projects.json"))
    composed = _build_managed_group_registry(project_manager)

    assert composed.storage_path == tmp_path / "managed-groups.json"
    assert _configured_managed_group_owner_id(
        MagicMock(admin_user_ids=frozenset({"ou_owner"}))
    ) == "ou_owner"


def test_project_retry_reuses_chat_bound_to_durable_provision_intent(
    tmp_path, project_manager, lark_client, registry
):
    root = str(tmp_path / "retry-project")
    os.makedirs(root)
    project_id = ProjectManager.generate_id("retry-project")
    provision_id = f"new-chat:ou_owner:{project_id}:{os.path.normpath(root)}"
    registry.begin_provision(
        provision_id=provision_id,
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id=project_id,
        canonical_root_ref=root,
        created_at=datetime.now(UTC),
    )
    registry.bind_provision_chat(provision_id, "oc_recovered")
    events = []
    service = _project_service(project_manager, lark_client, registry, events)

    service.handle(
        message_id="om_retry",
        chat_id="oc_owner_p2p",
        sender_open_id="ou_owner",
        data={"name": "retry-project", "path": root},
    )

    lark_client.create_chat.assert_not_called()
    assert registry.active_record("oc_recovered") is not None


def test_project_manager_create_rolls_back_when_initial_save_fails(
    tmp_path, project_manager
):
    root = str(tmp_path / "unsaved")
    with patch.object(project_manager, "_save_projects", return_value=False):
        success, message, project = project_manager.create_project(
            project_id="unsaved",
            project_name="unsaved",
            root_path=root,
            chat_id="oc_unsaved",
        )

    assert success is False
    assert "持久化" in message
    assert project is None
    assert project_manager.get_project_for_diagnostics("unsaved") is None
    assert project_manager.find_by_bound_chat_id("oc_unsaved") is None


def test_project_manager_close_restores_project_when_save_fails(
    tmp_path, project_manager
):
    root = str(tmp_path / "kept")
    success, _, project = project_manager.create_project(
        project_id="kept",
        project_name="kept",
        root_path=root,
        chat_id="oc_kept",
    )
    assert success and project is not None
    project.bound_chat_id = "oc_kept"
    assert project_manager._save_projects()

    with patch.object(project_manager, "_save_projects", return_value=False):
        closed, message = project_manager.close_project("kept")

    assert closed is False
    assert "持久化" in message
    assert project_manager.get_project_for_diagnostics("kept") is project
    assert project_manager.find_by_bound_chat_id("oc_kept") is project


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_team_post_active_bootstrap_failure_does_not_revoke_or_delete(
    MockLarkChatClient, _sender, _session, tmp_path, registry
):
    handler = _make_slock_handler(tmp_path, registry)
    MockLarkChatClient.return_value.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_committed", name="Committed [Slock]"
    )
    handler._bootstrap_default_roles_if_configured = MagicMock(
        side_effect=RuntimeError("bootstrap unavailable")
    )

    handler.create_team("om_create", "oc_owner_p2p", "Committed")

    assert registry.active_record("oc_committed") is not None
    MockLarkChatClient.return_value.delete_chat.assert_not_called()
    assert handler.ctx.slock_engine_manager.is_managed_chat("oc_committed") is True


def test_channel_marker_fsyncs_file_and_parent_with_random_temp_path(
    tmp_path, monkeypatch
):
    from src.slock_engine.engine import SlockEngine

    marker = tmp_path / ".slock_channel.json"
    original_fsync = os.fsync
    fsync_calls = 0
    replace_sources = []

    def counting_fsync(fd):
        nonlocal fsync_calls
        fsync_calls += 1
        original_fsync(fd)

    original_replace = os.replace

    def capture_replace(source, target):
        replace_sources.append(str(source))
        original_replace(source, target)

    monkeypatch.setattr("src.slock_engine.engine.os.fsync", counting_fsync)
    monkeypatch.setattr("src.slock_engine.engine.os.replace", capture_replace)

    SlockEngine._write_channel_marker(str(marker), {"channel_id": "oc_marker"})

    assert fsync_calls == 2
    assert replace_sources[0] != f"{marker}.tmp"


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_dissolve_rejected_delete_cancels_revoke_and_restores_active(
    MockLarkChatClient, _sender, _session, tmp_path, registry
):
    handler = _make_slock_handler(tmp_path, registry)
    lark = MockLarkChatClient.return_value
    lark.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_keep", name="Keep [Slock]"
    )
    handler.create_team("om_create", "oc_owner_p2p", "Keep")
    observed_before_delete = []

    def reject_delete(_chat_id):
        observed_before_delete.append(registry.active_record("oc_keep"))
        return False

    lark.delete_chat.side_effect = reject_delete
    handler._check_slock_permission = MagicMock(return_value=True)

    handler.dissolve_team("om_dissolve", "oc_owner_p2p", "Keep")

    assert observed_before_delete == [None]
    assert registry.pending_revokes() == ()
    assert registry.active_record("oc_keep") is not None
    assert handler.ctx.slock_engine_manager.is_managed_chat("oc_keep") is True


@pytest.mark.parametrize("marker_failure", ["raise", "missing"])
@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_dissolve_marker_prepare_failure_durably_cancels_revoke(
    MockLarkChatClient,
    _sender,
    _session,
    marker_failure,
    tmp_path,
    registry,
):
    handler = _make_slock_handler(tmp_path, registry)
    lark = MockLarkChatClient.return_value
    lark.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_marker_prepare", name="Marker Prepare [Slock]"
    )
    handler.create_team("om_create", "oc_owner_p2p", "Marker Prepare")
    handler._check_slock_permission = MagicMock(return_value=True)
    if marker_failure == "raise":
        handler.ctx.slock_engine_manager.archive_managed_chat_marker = MagicMock(
            side_effect=OSError("disk full")
        )
    else:
        handler.ctx.slock_engine_manager.archive_managed_chat_marker = MagicMock(
            return_value=None
        )

    handler.dissolve_team(
        "om_dissolve", "oc_owner_p2p", "Marker Prepare"
    )

    lark.delete_chat.assert_not_called()
    assert registry.pending_revokes() == ()
    assert registry.active_record("oc_marker_prepare") is not None
    assert handler.ctx.slock_engine_manager.is_managed_chat(
        "oc_marker_prepare"
    ) is True


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_dissolve_marker_failure_cancel_failure_stays_fail_closed(
    MockLarkChatClient, _sender, _session, tmp_path, registry
):
    handler = _make_slock_handler(tmp_path, registry)
    lark = MockLarkChatClient.return_value
    lark.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_cancel_failed", name="Cancel Failed [Slock]"
    )
    handler.create_team("om_create", "oc_owner_p2p", "Cancel Failed")
    handler._check_slock_permission = MagicMock(return_value=True)
    handler.ctx.slock_engine_manager.archive_managed_chat_marker = MagicMock(
        side_effect=OSError("disk full")
    )
    registry.cancel_revoke = MagicMock(side_effect=OSError("disk unavailable"))

    handler.dissolve_team("om_dissolve", "oc_owner_p2p", "Cancel Failed")

    lark.delete_chat.assert_not_called()
    assert registry.pending_revokes() == ("oc_cancel_failed",)
    assert registry.active_record("oc_cancel_failed") is None
    assert "取消" in handler.reply_text.call_args.args[1]


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_dissolve_tombstone_failure_leaves_durable_fail_closed_revoke(
    MockLarkChatClient, _sender, _session, tmp_path, registry
):
    handler = _make_slock_handler(tmp_path, registry)
    lark = MockLarkChatClient.return_value
    lark.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_uncertain", name="Uncertain [Slock]"
    )
    handler.create_team("om_create", "oc_owner_p2p", "Uncertain")
    lark.delete_chat.return_value = True
    registry.complete_revoke = MagicMock(side_effect=OSError("disk unavailable"))
    handler._check_slock_permission = MagicMock(return_value=True)

    handler.dissolve_team("om_dissolve", "oc_owner_p2p", "Uncertain")

    assert registry.pending_revokes() == ("oc_uncertain",)
    assert registry.active_record("oc_uncertain") is None


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_dissolve_restore_failure_keeps_pending_revoke_fail_closed(
    MockLarkChatClient, _sender, _session, tmp_path, registry
):
    handler = _make_slock_handler(tmp_path, registry)
    lark = MockLarkChatClient.return_value
    lark.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_restore_failed", name="Restore Failed [Slock]"
    )
    handler.create_team("om_create", "oc_owner_p2p", "Restore Failed")
    lark.delete_chat.return_value = False
    handler._check_slock_permission = MagicMock(return_value=True)
    handler.ctx.slock_engine_manager.restore_archived_chat_marker = MagicMock(
        side_effect=OSError("disk unavailable")
    )

    handler.dissolve_team(
        "om_dissolve", "oc_owner_p2p", "Restore Failed"
    )

    assert registry.pending_revokes() == ("oc_restore_failed",)
    assert registry.active_record("oc_restore_failed") is None


def test_lark_managed_chat_validation_requires_bot_and_owner_membership():
    from src.project_chat.lark_chat_client import (
        LarkChatClient,
        ManagedChatValidation,
    )

    api = MagicMock()
    bot_response = MagicMock()
    bot_response.success.return_value = True
    bot_response.data.is_in_chat = True
    members_response = MagicMock()
    members_response.success.return_value = True
    members_response.data.items = [MagicMock(member_id="ou_owner")]
    members_response.data.has_more = False
    members_response.data.trigger_security_conf_limit = False
    api.im.v1.chat_members.is_in_chat.return_value = bot_response
    api.im.v1.chat_members.get.return_value = members_response
    client = LarkChatClient(api_client_factory=lambda: api)

    assert (
        client.validate_managed_chat("oc_target", "ou_owner")
        is ManagedChatValidation.VALID
    )
    members_response.data.items = []
    assert (
        client.validate_managed_chat("oc_target", "ou_owner")
        is ManagedChatValidation.INVALID
    )
    api.im.v1.chat_members.is_in_chat.side_effect = RuntimeError("forbidden")
    assert (
        client.validate_managed_chat("oc_target", "ou_owner")
        is ManagedChatValidation.UNKNOWN
    )


def test_lark_managed_chat_validation_stops_repeated_page_token():
    from src.project_chat.lark_chat_client import (
        LarkChatClient,
        ManagedChatValidation,
    )

    api = MagicMock()
    bot_response = MagicMock()
    bot_response.success.return_value = True
    bot_response.data.is_in_chat = True
    members_response = MagicMock()
    members_response.success.return_value = True
    members_response.data.items = []
    members_response.data.has_more = True
    members_response.data.page_token = "same-token"
    members_response.data.trigger_security_conf_limit = False
    api.im.v1.chat_members.is_in_chat.return_value = bot_response
    api.im.v1.chat_members.get.return_value = members_response
    client = LarkChatClient(api_client_factory=lambda: api)

    assert (
        client.validate_managed_chat("oc_loop", "ou_owner")
        is ManagedChatValidation.UNKNOWN
    )
    assert api.im.v1.chat_members.get.call_count == 2


def test_owner_p2p_access_adoption_delegates_without_allow_chat(monkeypatch):
    from src.feishu.handlers.system import SystemHandler

    handler = object.__new__(SystemHandler)
    handler.ctx = MagicMock()
    handler.ctx.managed_group_owner_id = "ou_owner"
    project_handler = MagicMock()
    handler.ctx.handlers = {"project": project_handler}
    handler.reply_text = MagicMock()
    handler.reply_error = MagicMock()
    handler._admin_bootstrap_service = MagicMock()
    monkeypatch.setattr("src.thread.get_current_is_p2p", lambda: True)
    monkeypatch.setattr("src.thread.get_current_sender_id", lambda: "ou_owner")

    handler._handle_access_command(
        "om_adopt",
        "oc_owner_p2p",
        "adopt-chat oc_target project-1",
    )

    project_handler.adopt_managed_chat.assert_called_once_with(
        "om_adopt", "oc_target", "project-1"
    )
    handler._admin_bootstrap_service.assert_not_called()


def _write_slock_marker(base_path, chat_id: str) -> str:
    marker_dir = base_path / "groups" / chat_id
    marker_dir.mkdir(parents=True)
    marker = marker_dir / ".slock_channel.json"
    marker.write_text(
        json.dumps(
            {
                "channel_id": chat_id,
                "name": "Crash Team",
                "team_name": "Crash Team",
                "owner_id": "ou_owner",
                "root_path": str(base_path),
            }
        ),
        encoding="utf-8",
    )
    return str(marker)


def _activate_managed_chat(registry, chat_id: str) -> None:
    registry.register(
        chat_id=chat_id,
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id=f"project:{chat_id}",
        canonical_root_ref="/srv/project",
        created_at=datetime.now(UTC),
    )


def _startup_reconciler(tmp_path, registry, manager, remote, project_manager):
    from src.feishu.ws_client import FeishuWSClient

    client = object.__new__(FeishuWSClient)
    client._managed_group_registry = registry
    client._slock_engine_manager = manager
    client._project_manager = project_manager
    client._managed_group_owner_id = "ou_owner"
    client._managed_group_receiving_bot_ref = "cli_main_bot"
    client._get_api_client = MagicMock()
    return client


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_startup_consumes_adoption_saga_before_legacy_migration(
    client_cls, tmp_path, registry, project_manager
):
    from src.slock_engine.manager import SlockEngineManager

    root = str(tmp_path / "adopt-startup")
    success, _, project = project_manager.create_project(
        "adopt-startup", "adopt-startup", root
    )
    assert success and project is not None
    operation_id = "adopt:ou_owner:oc_adopt_startup:adopt-startup"
    registry.begin_provision(
        provision_id=operation_id,
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.OWNER_ADOPTED,
        receiving_bot_ref="cli_main_bot",
        project_id=project.project_id,
        canonical_root_ref=project.root_path,
        created_at=datetime.now(UTC),
    )
    registry.bind_provision_chat(operation_id, "oc_adopt_startup")
    bound, _ = project_manager.bind_managed_chat_for_saga(
        project.project_id,
        "oc_adopt_startup",
        created_at=datetime.now(UTC).timestamp(),
        operation_id=operation_id,
    )
    assert bound
    restarted_projects = ProjectManager(
        storage_path=str(tmp_path / "projects.json")
    )
    restarted_registry = ManagedGroupRegistry(
        tmp_path / "managed-groups.json"
    )
    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    client = _startup_reconciler(
        tmp_path,
        restarted_registry,
        manager,
        client_cls.return_value,
        restarted_projects,
    )

    client._reconcile_managed_groups_before_slock_restore()

    record = restarted_registry.active_record("oc_adopt_startup")
    assert record is not None
    assert record.origin is ManagedGroupOrigin.OWNER_ADOPTED
    assert restarted_projects.pending_managed_chat_binding_sagas() == ()
    client_cls.return_value.validate_managed_chat.assert_not_called()


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_startup_restores_true_prebind_when_saga_has_no_provision(
    client_cls, tmp_path, registry, project_manager
):
    from src.slock_engine.manager import SlockEngineManager

    root = str(tmp_path / "adopt-orphan")
    success, _, project = project_manager.create_project(
        "adopt-orphan", "adopt-orphan", root
    )
    assert success and project is not None
    operation_id = "adopt:ou_owner:oc_orphan:adopt-orphan"
    bound, _ = project_manager.bind_managed_chat_for_saga(
        project.project_id,
        "oc_orphan",
        created_at=datetime.now(UTC).timestamp(),
        operation_id=operation_id,
    )
    assert bound
    restarted_projects = ProjectManager(
        storage_path=str(tmp_path / "projects.json")
    )
    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    client = _startup_reconciler(
        tmp_path,
        registry,
        manager,
        client_cls.return_value,
        restarted_projects,
    )

    client._reconcile_managed_groups_before_slock_restore()

    restored = restarted_projects.get_project_for_diagnostics("adopt-orphan")
    assert restored is not None and restored.bound_chat_id == ""
    assert restarted_projects.pending_managed_chat_binding_sagas() == ()
    client_cls.return_value.validate_managed_chat.assert_not_called()


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_startup_reconciles_uncertain_registry_commit_before_other_recovery(
    client_cls, tmp_path, registry, project_manager
):
    from src.slock_engine.manager import SlockEngineManager

    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    registry.reconcile_uncertain_commit = MagicMock(return_value=True)
    client = _startup_reconciler(
        tmp_path, registry, manager, client_cls.return_value, project_manager
    )
    client._reconcile_managed_groups_before_slock_restore()

    registry.reconcile_uncertain_commit.assert_called_once_with()


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_startup_migration_aggregates_conflicting_chat_candidates_as_ambiguous(
    client_cls, tmp_path, registry, project_manager
):
    from src.slock_engine.manager import SlockEngineManager

    roots = (tmp_path / "a", tmp_path / "b")
    for root_path in roots:
        root_path.mkdir()
    for project_id, root in (
        ("legacy-a", str(roots[0])),
        ("legacy-b", str(roots[1])),
    ):
        success, _, project = project_manager.create_project(
            project_id, project_id, root
        )
        assert success and project is not None
        project.bound_chat_id = "oc_shared_legacy"
        project.bound_chat_created_at = 123.0
    assert project_manager._save_projects()
    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    client = _startup_reconciler(
        tmp_path, registry, manager, client_cls.return_value, project_manager
    )
    client_cls.return_value.send_text_to_open_id.return_value = True

    client._reconcile_managed_groups_before_slock_restore()

    client_cls.return_value.validate_managed_chat.assert_not_called()
    assert registry.active_record("oc_shared_legacy") is None
    assert registry.migration_dispositions() == (
        ("oc_shared_legacy", "legacy-a,legacy-b", "ambiguous"),
    )
    client_cls.return_value.send_text_to_open_id.assert_called_once()
    assert registry.unreported_migration_dispositions() == ()


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_migration_owner_notification_failure_remains_retryable(
    client_cls, tmp_path, registry, project_manager
):
    from src.slock_engine.manager import SlockEngineManager

    registry.record_migration_disposition(
        "oc_review",
        project_id="legacy-a,legacy-b",
        status="ambiguous",
    )
    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    client = _startup_reconciler(
        tmp_path, registry, manager, client_cls.return_value, project_manager
    )
    client_cls.return_value.send_text_to_open_id.return_value = False

    client._reconcile_managed_groups_before_slock_restore()

    assert registry.unreported_migration_dispositions() == (
        ("oc_review", "legacy-a,legacy-b", "ambiguous"),
    )


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_startup_pending_revoke_archives_marker_before_confirmed_delete(
    client_cls, tmp_path, registry, project_manager
):
    from src.slock_engine.manager import SlockEngineManager

    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    marker = _write_slock_marker(tmp_path / "slock", "oc_crash_before_archive")
    _activate_managed_chat(registry, "oc_crash_before_archive")
    registry.begin_revoke("oc_crash_before_archive", requested_at=datetime.now(UTC))
    client_cls.return_value.delete_chat.return_value = True
    client = _startup_reconciler(
        tmp_path, registry, manager, client_cls.return_value, project_manager
    )

    client._reconcile_managed_groups_before_slock_restore()

    assert not os.path.exists(marker)
    assert registry.record("oc_crash_before_archive").status is ManagedGroupStatus.TOMBSTONED


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_startup_rejected_delete_cancels_only_after_durable_marker_restore(
    client_cls, tmp_path, registry, project_manager
):
    from src.slock_engine.manager import SlockEngineManager

    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    marker = _write_slock_marker(tmp_path / "slock", "oc_crash_after_archive")
    _activate_managed_chat(registry, "oc_crash_after_archive")
    registry.begin_revoke("oc_crash_after_archive", requested_at=datetime.now(UTC))
    archived = manager.archive_managed_chat_marker("oc_crash_after_archive")
    assert archived and not os.path.exists(marker)
    client_cls.return_value.delete_chat.return_value = False
    client = _startup_reconciler(
        tmp_path, registry, manager, client_cls.return_value, project_manager
    )

    client._reconcile_managed_groups_before_slock_restore()

    assert os.path.isfile(marker)
    assert registry.pending_revokes() == ()
    assert registry.active_record("oc_crash_after_archive") is not None


def test_restore_from_disk_skips_and_archives_untrusted_markers(tmp_path):
    from src.slock_engine.manager import SlockEngineManager

    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    marker = _write_slock_marker(tmp_path / "slock", "oc_tombstoned")

    restored = manager.restore_from_disk(
        str(tmp_path),
        managed_group_active=lambda _chat_id: False,
    )

    assert restored == 0
    assert not os.path.exists(marker)


def _adoption_handler(project_manager, registry):
    from src.feishu.handlers.project import ProjectHandler

    handler = object.__new__(ProjectHandler)
    handler.ctx = MagicMock()
    handler.ctx.project_manager = project_manager
    handler.ctx.managed_group_registry = registry
    handler.ctx.managed_group_owner_id = "ou_owner"
    handler.ctx.managed_group_receiving_bot_ref = "cli_main_bot"
    handler.ctx.api_client_factory = MagicMock()
    handler.reply_error = MagicMock()
    handler.reply_text = MagicMock()
    return handler


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_adoption_durably_binds_project_before_registry_active(
    client_cls, tmp_path, project_manager, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    root = str(tmp_path / "adopt-order")
    success, _, project = project_manager.create_project(
        "adopt-order", "adopt-order", root
    )
    assert success and project is not None
    client_cls.return_value.validate_managed_chat.return_value = (
        ManagedChatValidation.VALID
    )
    handler = _adoption_handler(project_manager, registry)
    calls = []
    original_bind = project_manager.bind_managed_chat_for_saga
    original_activate = registry.activate

    def observe_bind(*args, **kwargs):
        calls.append("bind")
        assert registry.active_record("oc_adopt_order") is None
        return original_bind(*args, **kwargs)

    def observe_activate(*args, **kwargs):
        calls.append("activate")
        assert project_manager.find_by_bound_chat_id("oc_adopt_order") is project
        return original_activate(*args, **kwargs)

    project_manager.bind_managed_chat_for_saga = MagicMock(side_effect=observe_bind)
    registry.activate = MagicMock(side_effect=observe_activate)

    handler.adopt_managed_chat("om_adopt", "oc_adopt_order", "adopt-order")

    assert calls == ["bind", "activate"]
    assert registry.active_record("oc_adopt_order") is not None


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_adoption_activation_failure_durably_restores_project_binding(
    client_cls, tmp_path, project_manager, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    root = str(tmp_path / "adopt-rollback")
    success, _, project = project_manager.create_project(
        "adopt-rollback", "adopt-rollback", root
    )
    assert success and project is not None
    client_cls.return_value.validate_managed_chat.return_value = (
        ManagedChatValidation.VALID
    )
    handler = _adoption_handler(project_manager, registry)
    original_bind = project_manager.bind_managed_chat_for_saga
    project_manager.bind_managed_chat_for_saga = MagicMock(wraps=original_bind)
    registry.activate = MagicMock(side_effect=OSError("registry unavailable"))

    handler.adopt_managed_chat(
        "om_adopt", "oc_adopt_rollback", "adopt-rollback"
    )

    project_manager.bind_managed_chat_for_saga.assert_called_once()
    replayed = ProjectManager(storage_path=str(tmp_path / "projects.json"))
    restored = replayed.get_project_for_diagnostics("adopt-rollback")
    assert restored is not None and restored.bound_chat_id == ""
    assert registry.active_record("oc_adopt_rollback") is None


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_adoption_retry_after_crash_uses_durable_original_binding_snapshot(
    client_cls, tmp_path, project_manager, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    root = str(tmp_path / "adopt-crash")
    success, _, project = project_manager.create_project(
        "adopt-crash", "adopt-crash", root
    )
    assert success and project is not None
    operation_id = "adopt:ou_owner:oc_adopt_crash:adopt-crash"
    bound, _ = project_manager.bind_managed_chat_for_saga(
        "adopt-crash",
        "oc_adopt_crash",
        created_at=123.0,
        operation_id=operation_id,
    )
    assert bound
    restarted = ProjectManager(storage_path=str(tmp_path / "projects.json"))
    client_cls.return_value.validate_managed_chat.return_value = (
        ManagedChatValidation.VALID
    )
    handler = _adoption_handler(restarted, registry)
    registry.activate = MagicMock(side_effect=OSError("registry unavailable"))

    handler.adopt_managed_chat("om_adopt", "oc_adopt_crash", "adopt-crash")

    replayed = ProjectManager(storage_path=str(tmp_path / "projects.json"))
    restored = replayed.get_project_for_diagnostics("adopt-crash")
    assert restored is not None and restored.bound_chat_id == ""


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_adoption_registry_commit_uncertain_preserves_durable_project_binding(
    client_cls, tmp_path, project_manager, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    root = str(tmp_path / "adopt-uncertain")
    success, _, project = project_manager.create_project(
        "adopt-uncertain", "adopt-uncertain", root
    )
    assert success and project is not None
    client_cls.return_value.validate_managed_chat.return_value = (
        ManagedChatValidation.VALID
    )
    handler = _adoption_handler(project_manager, registry)
    registry.activate = MagicMock(
        side_effect=RegistryCommitUncertainError(
            "parent fsync failed",
            committed=True,
        )
    )

    handler.adopt_managed_chat(
        "om_adopt", "oc_adopt_uncertain", "adopt-uncertain"
    )

    replayed = ProjectManager(storage_path=str(tmp_path / "projects.json"))
    rebound = replayed.get_project_for_diagnostics("adopt-uncertain")
    assert rebound is not None
    assert rebound.bound_chat_id == "oc_adopt_uncertain"
    assert handler.reply_text.call_count == 0


def test_project_quarantine_survives_restart_and_blocks_bound_index(
    tmp_path, project_manager
):
    root = str(tmp_path / "quarantine")
    success, _, project = project_manager.create_project(
        "quarantine", "quarantine", root
    )
    assert success and project is not None
    assert project_manager.bind_managed_chat(
        "quarantine",
        "oc_quarantined",
        created_at=datetime.now(UTC).timestamp(),
    )

    assert project_manager.quarantine_bound_chat("oc_quarantined") is True
    restarted = ProjectManager(storage_path=str(tmp_path / "projects.json"))
    assert restarted.find_by_bound_chat_id("oc_quarantined") is None


def test_project_parent_directory_fsync_open_failure_is_committed_uncertain(
    project_manager, monkeypatch
):
    monkeypatch.setattr(
        "src.project.manager.os.open",
        MagicMock(side_effect=OSError("directory unavailable")),
    )

    with pytest.raises(ProjectCommitUncertainError) as raised:
        project_manager.quarantine_bound_chat("oc_unconfirmed")

    assert raised.value.committed is True


def test_slock_residual_requires_parent_directory_fsync(tmp_path, monkeypatch):
    from src.slock_engine.manager import SlockEngineManager

    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    original_fsync = os.fsync

    def fail_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        original_fsync(fd)

    monkeypatch.setattr("src.slock_engine.manager.os.fsync", fail_directory_fsync)

    assert manager.block_team_name_for_cleanup(
        "Residual",
        "oc_residual",
        "delete_unknown",
    ) is False


def test_project_rejects_second_pending_binding_saga_for_same_project(
    tmp_path, project_manager
):
    root = str(tmp_path / "single-project-saga")
    success, _, project = project_manager.create_project(
        "single-project-saga", "single-project-saga", root
    )
    assert success and project is not None
    first, _ = project_manager.bind_managed_chat_for_saga(
        project.project_id,
        "oc_first_saga",
        created_at=100.0,
        operation_id="op:first",
    )

    second, _ = project_manager.bind_managed_chat_for_saga(
        project.project_id,
        "oc_second_saga",
        created_at=200.0,
        operation_id="op:second",
    )

    assert first is True
    assert second is False
    assert [saga.operation_id for saga in project_manager.pending_managed_chat_binding_sagas()] == [
        "op:first"
    ]


@pytest.mark.parametrize("action", ["restore", "complete"])
def test_project_binding_saga_cas_does_not_overwrite_newer_binding(
    action, tmp_path, project_manager
):
    root = str(tmp_path / f"binding-cas-{action}")
    success, _, project = project_manager.create_project(
        f"binding-cas-{action}", f"binding-cas-{action}", root
    )
    assert success and project is not None
    bound, _ = project_manager.bind_managed_chat_for_saga(
        project.project_id,
        "oc_old_binding",
        created_at=100.0,
        operation_id=f"op:{action}",
    )
    assert bound
    project.bound_chat_id = "oc_new_binding"
    project.bound_chat_created_at = 200.0
    project.managed_binding_generation += 1
    assert project_manager._save_projects()

    if action == "restore":
        result = project_manager.restore_managed_chat_binding_saga(f"op:{action}")
    else:
        result = project_manager.complete_managed_chat_binding_saga(f"op:{action}")

    assert result is False
    restarted = ProjectManager(storage_path=str(tmp_path / "projects.json"))
    current = restarted.get_project_for_diagnostics(project.project_id)
    assert current is not None and current.bound_chat_id == "oc_new_binding"
    assert len(restarted.pending_managed_chat_binding_sagas()) == 1
    assert restarted.find_by_bound_chat_id("oc_new_binding") is None


def test_new_project_create_bind_and_saga_are_one_durable_commit(
    tmp_path, project_manager
):
    create_atomic = getattr(
        project_manager, "create_project_with_managed_chat_saga", None
    )
    assert create_atomic is not None
    root = str(tmp_path / "atomic-project")

    success, _, project = create_atomic(
        project_id="atomic-project",
        project_name="atomic-project",
        root_path=root,
        owner_chat_id="oc_owner_p2p",
        managed_chat_id="oc_atomic_group",
        managed_chat_name="atomic-dev",
        created_at=123.0,
        operation_id="new-chat:ou_owner:atomic-project",
        expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        expected_owner_id="ou_owner",
        expected_receiving_bot_ref="cli_main_bot",
    )

    assert success and project is not None
    restarted = ProjectManager(storage_path=str(tmp_path / "projects.json"))
    persisted = restarted.get_project_for_diagnostics("atomic-project")
    assert persisted is not None and persisted.bound_chat_id == "oc_atomic_group"
    assert persisted.managed_binding_generation == 1
    assert len(restarted.pending_managed_chat_binding_sagas()) == 1


def test_project_atomic_binding_reports_committed_parent_fsync_uncertainty(
    tmp_path, project_manager, monkeypatch
):
    error_type = getattr(
        __import__("src.project.manager", fromlist=["ProjectCommitUncertainError"]),
        "ProjectCommitUncertainError",
        None,
    )
    assert error_type is not None
    original_fsync = os.fsync

    def fail_parent_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("parent fsync failed")
        original_fsync(fd)

    monkeypatch.setattr("src.project.manager.os.fsync", fail_parent_fsync)
    with pytest.raises(error_type) as raised:
        project_manager.create_project_with_managed_chat_saga(
            project_id="uncertain-project",
            project_name="uncertain-project",
            root_path=str(tmp_path / "uncertain-project"),
            owner_chat_id="oc_owner_p2p",
            managed_chat_id="oc_uncertain_group",
            managed_chat_name="uncertain-dev",
            created_at=123.0,
            operation_id="new-chat:ou_owner:uncertain-project",
            expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
            expected_owner_id="ou_owner",
            expected_receiving_bot_ref="cli_main_bot",
        )

    assert raised.value.committed is True
    restarted = ProjectManager(storage_path=str(tmp_path / "projects.json"))
    assert restarted.get_project_for_diagnostics("uncertain-project") is not None
    assert len(restarted.pending_managed_chat_binding_sagas()) == 1


def test_new_chat_creation_lock_is_shared_by_root_across_source_chats(tmp_path):
    from src.project_chat.service import _get_creation_lock

    root = str(tmp_path / "shared-root")

    assert _get_creation_lock("oc_source_one", root) is _get_creation_lock(
        "oc_source_two", root
    )


def test_recovered_project_chat_is_revalidated_before_activation(
    tmp_path, project_manager, lark_client, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    root = str(tmp_path / "revalidate-recovered")
    os.makedirs(root)
    provision_id = (
        f"new-chat:ou_owner:revalidate_recovered:{os.path.normpath(root)}"
    )
    registry.begin_provision(
        provision_id=provision_id,
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id="revalidate_recovered",
        canonical_root_ref=root,
        created_at=datetime.now(UTC),
    )
    registry.bind_provision_chat(provision_id, "oc_recovered_missing")
    lark_client.validate_managed_chat.return_value = ManagedChatValidation.INVALID
    events = []
    service = _project_service(project_manager, lark_client, registry, events)

    service.handle(
        "om_retry",
        "oc_owner_p2p",
        "ou_owner",
        {"name": "revalidate-recovered", "path": root},
    )

    assert registry.active_record("oc_recovered_missing") is None
    lark_client.add_managers.assert_not_called()
    assert project_manager.managed_group_residual(provision_id) == (
        "oc_recovered_missing",
        "recovered_chat_invalid",
    )


def test_runtime_active_fast_path_finalizes_matching_pending_saga(
    tmp_path, project_manager, lark_client, registry
):
    root = str(tmp_path / "runtime-finalize")
    os.makedirs(root)
    success, _, project = project_manager.create_project(
        "runtime-finalize", "runtime-finalize", root
    )
    assert success and project is not None
    provision_id = f"new-chat:ou_owner:runtime_finalize:{os.path.normpath(root)}"
    registry.begin_provision(
        provision_id=provision_id,
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id=project.project_id,
        canonical_root_ref=root,
        created_at=datetime.now(UTC),
    )
    registry.bind_provision_chat(provision_id, "oc_runtime_finalized")
    bound, _ = project_manager.bind_managed_chat_for_saga(
        project.project_id,
        "oc_runtime_finalized",
        created_at=123.0,
        operation_id=provision_id,
        expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        expected_owner_id="ou_owner",
        expected_receiving_bot_ref="cli_main_bot",
    )
    assert bound
    registry.activate(
        provision_id=provision_id,
        chat_id="oc_runtime_finalized",
        project_id=project.project_id,
        canonical_root_ref=root,
    )
    service = _project_service(project_manager, lark_client, registry, [])

    service.handle(
        "om_fast",
        "oc_other_source",
        "ou_owner",
        {"name": "runtime-finalize", "path": root},
    )

    assert project_manager.pending_managed_chat_binding_sagas() == ()
    lark_client.create_chat.assert_not_called()


def test_startup_does_not_finalize_saga_against_wrong_active_origin(
    tmp_path, project_manager, registry
):
    from src.slock_engine.manager import SlockEngineManager

    root = str(tmp_path / "wrong-origin")
    success, _, project = project_manager.create_project(
        "wrong-origin", "wrong-origin", root
    )
    assert success and project is not None
    registry.register(
        chat_id="oc_wrong_origin",
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id=project.project_id,
        canonical_root_ref=root,
        created_at=datetime.now(UTC),
    )
    operation_id = "adopt:ou_owner:oc_wrong_origin:wrong-origin"
    registry.begin_provision(
        provision_id=operation_id,
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.OWNER_ADOPTED,
        receiving_bot_ref="cli_main_bot",
        project_id=project.project_id,
        canonical_root_ref=root,
        created_at=datetime.now(UTC),
    )
    registry.bind_provision_chat(operation_id, "oc_wrong_origin")
    bound, _ = project_manager.bind_managed_chat_for_saga(
        project.project_id,
        "oc_wrong_origin",
        created_at=123.0,
        operation_id=operation_id,
        expected_origin=ManagedGroupOrigin.OWNER_ADOPTED,
        expected_owner_id="ou_owner",
        expected_receiving_bot_ref="cli_main_bot",
    )
    assert bound
    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    remote = MagicMock()
    client = _startup_reconciler(
        tmp_path, registry, manager, remote, project_manager
    )

    client._reconcile_managed_groups_before_slock_restore()

    assert len(project_manager.pending_managed_chat_binding_sagas()) == 1
    assert project_manager.find_by_bound_chat_id("oc_wrong_origin") is None
    remote.delete_chat.assert_not_called()


def test_slock_archive_fsync_failure_restores_active_marker(tmp_path):
    from src.slock_engine.manager import SlockEngineManager

    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    marker = _write_slock_marker(tmp_path / "slock", "oc_archive_uncertain")
    original_fsync_directory = manager._fsync_directory
    calls = 0

    def fail_first_fsync(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("archive parent fsync failed")
        original_fsync_directory(path)

    manager._fsync_directory = fail_first_fsync

    with pytest.raises(OSError, match="archive parent fsync failed"):
        manager.archive_managed_chat_marker("oc_archive_uncertain")

    assert os.path.isfile(marker)


def test_slock_first_pending_cleanup_directory_fsyncs_storage_parent(tmp_path):
    from src.slock_engine.manager import SlockEngineManager

    storage = tmp_path / "slock"
    manager = SlockEngineManager(storage_base_path=str(storage))
    original = manager._fsync_directory
    fsynced = []

    def observe(path):
        fsynced.append(os.path.realpath(path))
        original(path)

    manager._fsync_directory = observe

    assert manager.block_team_name_for_cleanup(
        "First Residual", "oc_first_residual", "delete_unknown"
    )
    assert os.path.realpath(storage) in fsynced
    assert os.path.realpath(storage / "pending_cleanup") in fsynced


@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_recovered_team_chat_is_revalidated_before_activation(
    client_cls, _sender, tmp_path, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    handler = _make_slock_handler(tmp_path, registry)
    root = handler.get_working_dir("oc_owner_p2p")
    provision_id = f"new-team:ou_owner:recovered team:{root}"
    registry.begin_provision(
        provision_id=provision_id,
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id="team:Recovered Team",
        canonical_root_ref=root,
        created_at=datetime.now(UTC),
    )
    registry.bind_provision_chat(provision_id, "oc_recovered_team_missing")
    client_cls.return_value.validate_managed_chat.return_value = (
        ManagedChatValidation.INVALID
    )

    handler.create_team(
        "om_retry_team",
        "oc_owner_p2p",
        "Recovered Team",
    )

    assert registry.active_record("oc_recovered_team_missing") is None
    assert not handler.ctx.slock_engine_manager.is_managed_chat(
        "oc_recovered_team_missing"
    )
    assert "确认" in handler.reply_text.call_args.args[1]


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError])
def test_project_post_replace_failure_is_committed_uncertain(
    failure_type, tmp_path, project_manager, monkeypatch
):
    original_fsync = os.fsync

    def fail_parent_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise failure_type("parent fsync failed")
        original_fsync(fd)

    monkeypatch.setattr("src.project.manager.os.fsync", fail_parent_fsync)

    with pytest.raises(ProjectCommitUncertainError) as raised:
        project_manager.create_project_with_managed_chat_saga(
            project_id="double-uncertain",
            project_name="double-uncertain",
            root_path=str(tmp_path / "double-uncertain"),
            owner_chat_id="oc_double_uncertain",
            managed_chat_id="oc_double_uncertain",
            managed_chat_name="double-uncertain-dev",
            created_at=123.0,
            operation_id="new-chat:ou_owner:double-uncertain",
            expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
            expected_owner_id="ou_owner",
            expected_receiving_bot_ref="cli_main_bot",
        )

    assert raised.value.committed is True
    assert project_manager.get_project_for_diagnostics("double-uncertain") is not None
    assert len(project_manager.pending_managed_chat_binding_sagas()) == 1


@pytest.mark.parametrize("action", ["restore", "complete"])
def test_legacy_binding_saga_without_expected_is_non_destructive(
    action, tmp_path, project_manager
):
    operation_id = f"legacy:{action}"
    success, _, project = project_manager.create_project_with_managed_chat_saga(
        project_id=f"legacy-{action}",
        project_name=f"legacy-{action}",
        root_path=str(tmp_path / f"legacy-{action}"),
        owner_chat_id="oc_legacy",
        managed_chat_id="oc_legacy",
        managed_chat_name="legacy-dev",
        created_at=123.0,
        operation_id=operation_id,
        expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        expected_owner_id="ou_owner",
        expected_receiving_bot_ref="cli_main_bot",
    )
    assert success and project is not None
    storage = tmp_path / "projects.json"
    payload = json.loads(storage.read_text(encoding="utf-8"))
    saga = payload["managed_chat_binding_sagas"][operation_id]
    payload["managed_chat_binding_sagas"][operation_id] = {
        "chat_id": saga["chat_id"],
        "project_id": saga["project_id"],
        "remove_project_on_restore": True,
        "snapshot": saga["snapshot"],
    }
    storage.write_text(json.dumps(payload), encoding="utf-8")
    restarted = ProjectManager(storage_path=str(storage))

    result = (
        restarted.restore_managed_chat_binding_saga(operation_id)
        if action == "restore"
        else restarted.complete_managed_chat_binding_saga(operation_id)
    )

    assert result is False
    assert restarted.get_project_for_diagnostics(f"legacy-{action}") is not None
    assert len(restarted.pending_managed_chat_binding_sagas()) == 1
    assert restarted.find_by_bound_chat_id("oc_legacy") is None


def test_runtime_active_fast_path_finalizes_unique_saga_after_name_change(
    tmp_path, project_manager, lark_client, registry
):
    root = str(tmp_path / "renamed-active")
    os.makedirs(root)
    success, _, project = project_manager.create_project(
        "renamed-active", "original-name", root
    )
    assert success and project is not None
    operation_id = f"new-chat:ou_owner:original_name:{os.path.normpath(root)}"
    registry.begin_provision(
        provision_id=operation_id,
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id=project.project_id,
        canonical_root_ref=root,
        created_at=datetime.now(UTC),
    )
    registry.bind_provision_chat(operation_id, "oc_renamed_active")
    bound, _ = project_manager.bind_managed_chat_for_saga(
        project.project_id,
        "oc_renamed_active",
        created_at=123.0,
        operation_id=operation_id,
        expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        expected_owner_id="ou_owner",
        expected_receiving_bot_ref="cli_main_bot",
    )
    assert bound
    registry.activate(
        provision_id=operation_id,
        chat_id="oc_renamed_active",
        project_id=project.project_id,
        canonical_root_ref=root,
    )
    service = _project_service(project_manager, lark_client, registry, [])

    service.handle(
        "om_renamed",
        "oc_new_source",
        "ou_owner",
        {"name": "renamed-name", "path": root},
    )

    assert project_manager.pending_managed_chat_binding_sagas() == ()
    assert "oc_new_source" in project.allowed_chat_ids
    lark_client.create_chat.assert_not_called()


def test_legacy_name_fallback_refuses_root_mutation_with_pending_saga(
    tmp_path, project_manager, lark_client, registry
):
    original_root = str(tmp_path / "original-root")
    new_root = str(tmp_path / "new-root")
    os.makedirs(original_root)
    os.makedirs(new_root)
    success, _, project = project_manager.create_project(
        "root-locked", "root-locked", original_root
    )
    assert success and project is not None
    bound, _ = project_manager.bind_managed_chat_for_saga(
        project.project_id,
        "oc_root_locked",
        created_at=123.0,
        operation_id="pending-root-lock",
        expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        expected_owner_id="ou_owner",
        expected_receiving_bot_ref="cli_main_bot",
    )
    assert bound
    events = []
    service = _project_service(project_manager, lark_client, registry, events)

    service.handle(
        "om_root_change",
        "oc_owner_p2p",
        "ou_owner",
        {"name": "root-locked", "path": new_root},
    )

    assert project.root_path == original_root
    assert len(project_manager.pending_managed_chat_binding_sagas()) == 1
    lark_client.create_chat.assert_not_called()


def test_startup_quarantines_pending_saga_when_project_root_changed(
    tmp_path, project_manager, registry
):
    from src.slock_engine.manager import SlockEngineManager

    original_root = str(tmp_path / "startup-root-original")
    changed_root = str(tmp_path / "startup-root-changed")
    success, _, project = project_manager.create_project(
        "startup-root", "startup-root", original_root
    )
    assert success and project is not None
    operation_id = "startup-root-mismatch"
    registry.begin_provision(
        provision_id=operation_id,
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id=project.project_id,
        canonical_root_ref=original_root,
        created_at=datetime.now(UTC),
    )
    registry.bind_provision_chat(operation_id, "oc_startup_root")
    bound, _ = project_manager.bind_managed_chat_for_saga(
        project.project_id,
        "oc_startup_root",
        created_at=123.0,
        operation_id=operation_id,
        expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        expected_owner_id="ou_owner",
        expected_receiving_bot_ref="cli_main_bot",
    )
    assert bound
    project.root_path = changed_root
    project.working_dir = changed_root
    assert project_manager._save_projects()
    manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock"))
    remote = MagicMock()
    client = _startup_reconciler(
        tmp_path, registry, manager, remote, project_manager
    )

    client._reconcile_managed_groups_before_slock_restore()

    assert registry.active_record("oc_startup_root") is None
    assert len(project_manager.pending_managed_chat_binding_sagas()) == 1
    assert project_manager.find_by_bound_chat_id("oc_startup_root") is None
    remote.delete_chat.assert_not_called()


def test_lark_create_chat_uses_deterministic_operation_uuid():
    from src.project_chat.lark_chat_client import LarkChatClient

    api = MagicMock()
    response = MagicMock()
    response.success.return_value = True
    response.data.chat_id = "oc_uuid"
    api.im.v1.chat.create.return_value = response
    client = LarkChatClient(api_client_factory=lambda: api)

    client.create_chat(
        name="UUID Team",
        description="test",
        user_id_list=["ou_owner"],
        operation_id="new-chat:ou_owner:project:/srv/project",
    )
    first_uuid = api.im.v1.chat.create.call_args.args[0].uuid
    client.create_chat(
        name="UUID Team",
        description="test",
        user_id_list=["ou_owner"],
        operation_id="new-chat:ou_owner:project:/srv/project",
    )
    second_uuid = api.im.v1.chat.create.call_args.args[0].uuid

    assert first_uuid == second_uuid
    assert len(first_uuid) == 36


@pytest.mark.parametrize("delete_result", [False, None])
def test_project_bind_failure_persists_residual_and_blocks_blind_retry(
    delete_result, tmp_path, project_manager, lark_client, registry
):
    root = str(tmp_path / "residual-project")
    os.makedirs(root)
    events = []
    service = _project_service(project_manager, lark_client, registry, events)
    lark_client.delete_chat.return_value = delete_result
    registry.bind_provision_chat = MagicMock(
        side_effect=OSError("registry unavailable")
    )

    service.handle(
        "om_first",
        "oc_owner_p2p",
        "ou_owner",
        {"name": "residual-project", "path": root},
    )

    lark_client.delete_chat.assert_not_called()
    restarted = ProjectManager(storage_path=str(tmp_path / "projects.json"))
    provision_id = (
        f"new-chat:ou_owner:residual_project:{os.path.normpath(root)}"
    )
    assert restarted.managed_group_residual(provision_id) == (
        "oc_project_group",
        "untrusted_retained",
    )

    lark_client.create_chat.reset_mock()
    retry = _project_service(restarted, lark_client, registry, events)
    retry.handle(
        "om_retry",
        "oc_owner_p2p",
        "ou_owner",
        {"name": "residual-project", "path": root},
    )
    lark_client.create_chat.assert_not_called()


def test_project_retained_valid_retry_reuses_chat_and_clears_residual(
    tmp_path, project_manager, lark_client, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    root = str(tmp_path / "retained-project")
    os.makedirs(root)
    events = []
    service = _project_service(project_manager, lark_client, registry, events)
    original_activate = registry.activate
    attempts = 0

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("registry unavailable")
        return original_activate(**kwargs)

    registry.activate = MagicMock(side_effect=fail_once)
    request = {"name": "retained-project", "path": root}

    service.handle("om_first", "oc_owner_p2p", "ou_owner", request)
    operation_id = lark_client.create_chat.call_args.kwargs["operation_id"]
    assert project_manager.managed_group_residual(operation_id) == (
        "oc_project_group",
        "untrusted_retained",
    )
    lark_client.validate_managed_chat.return_value = ManagedChatValidation.VALID

    service.handle("om_retry", "oc_owner_p2p", "ou_owner", request)

    assert lark_client.create_chat.call_count == 1
    assert registry.active_record("oc_project_group") is not None
    assert project_manager.managed_group_residual(operation_id) is None
    assert project_manager.pending_managed_chat_binding_sagas() == ()
    assert events[-1][0] == "welcome"


def test_project_retained_unknown_retry_stays_blocked(
    tmp_path, project_manager, lark_client, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    root = str(tmp_path / "retained-project-unknown")
    os.makedirs(root)
    events = []
    service = _project_service(project_manager, lark_client, registry, events)
    registry.activate = MagicMock(side_effect=OSError("registry unavailable"))
    request = {"name": "retained-project-unknown", "path": root}

    service.handle("om_first", "oc_owner_p2p", "ou_owner", request)
    operation_id = lark_client.create_chat.call_args.kwargs["operation_id"]
    lark_client.validate_managed_chat.return_value = ManagedChatValidation.UNKNOWN
    registry.activate.reset_mock()

    service.handle("om_retry", "oc_owner_p2p", "ou_owner", request)

    assert lark_client.create_chat.call_count == 1
    registry.activate.assert_not_called()
    assert project_manager.managed_group_residual(operation_id) == (
        "oc_project_group",
        "untrusted_retained",
    )
    lark_client.validate_managed_chat.assert_called_with(
        "oc_project_group", "ou_owner"
    )
    assert "确认" in events[-1][3] or "阻止" in events[-1][3]


def test_project_retained_invalid_retry_clears_state_for_explicit_new_create(
    tmp_path, project_manager, lark_client, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    root = str(tmp_path / "retained-project-invalid")
    os.makedirs(root)
    service = _project_service(project_manager, lark_client, registry, [])
    original_activate = registry.activate
    attempts = 0

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("registry unavailable")
        return original_activate(**kwargs)

    registry.activate = MagicMock(side_effect=fail_once)
    request = {"name": "retained-project-invalid", "path": root}
    service.handle("om_first", "oc_owner_p2p", "ou_owner", request)
    operation_id = lark_client.create_chat.call_args.kwargs["operation_id"]
    original_abandon = registry.abandon_provision

    def abandon_while_residual_is_anchored(candidate_operation_id):
        assert project_manager.managed_group_residual(candidate_operation_id) == (
            "oc_project_group",
            "untrusted_retained",
        )
        return original_abandon(candidate_operation_id)

    registry.abandon_provision = MagicMock(
        side_effect=abandon_while_residual_is_anchored
    )
    lark_client.validate_managed_chat.return_value = ManagedChatValidation.INVALID

    service.handle("om_cleanup", "oc_owner_p2p", "ou_owner", request)

    assert project_manager.managed_group_residual(operation_id) is None
    assert registry.provision_chat_id(operation_id) is None
    service.handle("om_new", "oc_owner_p2p", "ou_owner", request)
    assert lark_client.create_chat.call_count == 2
    assert registry.active_record("oc_project_group") is not None


@pytest.mark.parametrize("failure", [False, "uncertain"])
def test_project_residual_write_failure_is_truthful_and_reuses_registry_intent(
    failure, tmp_path, project_manager, lark_client, registry
):
    root = str(tmp_path / f"residual-write-{failure}")
    os.makedirs(root)
    events = []
    service = _project_service(project_manager, lark_client, registry, events)
    original_record = project_manager.record_managed_group_residual
    original_activate = registry.activate
    registry.activate = MagicMock(side_effect=OSError("registry unavailable"))
    if failure is False:
        project_manager.record_managed_group_residual = MagicMock(
            return_value=False
        )
    else:
        project_manager.record_managed_group_residual = MagicMock(
            side_effect=ProjectCommitUncertainError(
                "parent durability uncertain",
                committed=True,
            )
        )
    request = {"name": f"residual-write-{failure}", "path": root}

    service.handle("om_first", "oc_owner_p2p", "ou_owner", request)

    assert "恢复记录" in events[-1][3] or "持久" in events[-1][3]
    project_manager.record_managed_group_residual = original_record
    registry.activate = original_activate
    service.handle("om_retry", "oc_owner_p2p", "ou_owner", request)

    assert lark_client.create_chat.call_count == 1
    assert registry.active_record("oc_project_group") is not None


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_team_retained_valid_retry_reuses_chat_and_consumes_block(
    client_cls, _sender, _session, tmp_path, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    handler = _make_slock_handler(tmp_path, registry)
    remote = client_cls.return_value
    remote.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_team_retained",
        name="Retained Team [Slock]",
    )
    original_bind = registry.bind_provision_chat
    attempts = 0

    def fail_once(operation_id, chat_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("registry unavailable")
        return original_bind(operation_id, chat_id)

    registry.bind_provision_chat = MagicMock(side_effect=fail_once)
    handler.create_team("om_first", "oc_owner_p2p", "Retained Team")
    remote.validate_managed_chat.return_value = ManagedChatValidation.VALID

    handler.create_team("om_retry", "oc_owner_p2p", "Retained Team")

    assert remote.create_chat.call_count == 1
    assert registry.active_record("oc_team_retained") is not None
    cleanup = tmp_path / "slock" / "pending_cleanup"
    assert not list(cleanup.glob("*.json"))


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_team_retained_unknown_retry_stays_blocked(
    client_cls, _sender, _session, tmp_path, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    handler = _make_slock_handler(tmp_path, registry)
    remote = client_cls.return_value
    remote.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_team_unknown",
        name="Unknown Team [Slock]",
    )
    registry.bind_provision_chat = MagicMock(side_effect=OSError("disk full"))
    handler.create_team("om_first", "oc_owner_p2p", "Unknown Team")
    remote.validate_managed_chat.return_value = ManagedChatValidation.UNKNOWN
    registry.bind_provision_chat.reset_mock()

    handler.create_team("om_retry", "oc_owner_p2p", "Unknown Team")

    assert remote.create_chat.call_count == 1
    assert registry.active_record("oc_team_unknown") is None
    assert "确认" in handler.reply_text.call_args.args[1]


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_team_retained_invalid_retry_clears_block_for_explicit_new_create(
    client_cls, _sender, _session, tmp_path, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    handler = _make_slock_handler(tmp_path, registry)
    remote = client_cls.return_value
    remote.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_team_invalid",
        name="Invalid Team [Slock]",
    )
    original_bind = registry.bind_provision_chat
    attempts = 0

    def fail_once(operation_id, chat_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("registry unavailable")
        return original_bind(operation_id, chat_id)

    registry.bind_provision_chat = MagicMock(side_effect=fail_once)
    handler.create_team("om_first", "oc_owner_p2p", "Invalid Team")
    original_abandon = registry.abandon_provision

    def abandon_while_block_is_anchored(candidate_operation_id):
        cleanup = tmp_path / "slock" / "pending_cleanup"
        assert list(cleanup.glob("*.json"))
        return original_abandon(candidate_operation_id)

    registry.abandon_provision = MagicMock(
        side_effect=abandon_while_block_is_anchored
    )
    remote.validate_managed_chat.return_value = ManagedChatValidation.INVALID

    handler.create_team("om_cleanup", "oc_owner_p2p", "Invalid Team")

    operation_id = (
        "new-team:ou_owner:invalid team:"
        f"{handler.get_working_dir.return_value}"
    )
    assert registry.provision_chat_id(operation_id) is None
    handler.create_team("om_new", "oc_owner_p2p", "Invalid Team")
    assert remote.create_chat.call_count == 2
    assert registry.active_record("oc_team_invalid") is not None


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_team_residual_write_failure_keeps_reservation_and_reports_truthfully(
    client_cls, _sender, _session, tmp_path, registry
):
    handler = _make_slock_handler(tmp_path, registry)
    remote = client_cls.return_value
    remote.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_team_write_failed",
        name="Write Failed [Slock]",
    )
    registry.bind_provision_chat = MagicMock(side_effect=OSError("disk full"))
    manager = handler.ctx.slock_engine_manager
    manager.block_team_name_for_cleanup = MagicMock(return_value=False)

    handler.create_team("om_first", "oc_owner_p2p", "Write Failed")

    assert manager.reserve_team_name("Write Failed") is False
    assert "持久" in handler.reply_text.call_args.args[1]


def _rewrite_binding_saga_as_legacy(storage, operation_id: str) -> None:
    payload = json.loads(storage.read_text(encoding="utf-8"))
    saga = payload["managed_chat_binding_sagas"][operation_id]
    payload["managed_chat_binding_sagas"][operation_id] = {
        "chat_id": saga["chat_id"],
        "project_id": saga["project_id"],
        "remove_project_on_restore": saga["remove_project_on_restore"],
        "snapshot": saga["snapshot"],
    }
    storage.write_text(json.dumps(payload), encoding="utf-8")


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_startup_auto_clears_exact_active_legacy_saga(
    client_cls, tmp_path, project_manager, registry
):
    from src.slock_engine.manager import SlockEngineManager

    root = str(tmp_path / "legacy-active")
    operation_id = "new-chat:ou_owner:legacy-active"
    success, _, project = project_manager.create_project_with_managed_chat_saga(
        project_id="legacy-active",
        project_name="legacy-active",
        root_path=root,
        owner_chat_id="oc_legacy_active",
        managed_chat_id="oc_legacy_active",
        managed_chat_name="legacy-active-dev",
        created_at=123.0,
        operation_id=operation_id,
        expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        expected_owner_id="ou_owner",
        expected_receiving_bot_ref="cli_main_bot",
    )
    assert success and project is not None
    registry.begin_provision(
        provision_id=operation_id,
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id=project.project_id,
        canonical_root_ref=root,
        created_at=datetime.now(UTC),
    )
    registry.bind_provision_chat(operation_id, "oc_legacy_active")
    registry.activate(
        provision_id=operation_id,
        chat_id="oc_legacy_active",
        project_id=project.project_id,
        canonical_root_ref=root,
    )
    storage = tmp_path / "projects.json"
    _rewrite_binding_saga_as_legacy(storage, operation_id)
    restarted = ProjectManager(storage_path=str(storage))
    client = _startup_reconciler(
        tmp_path,
        registry,
        SlockEngineManager(storage_base_path=str(tmp_path / "slock")),
        client_cls.return_value,
        restarted,
    )

    client._reconcile_managed_groups_before_slock_restore()

    assert restarted.pending_managed_chat_binding_sagas() == ()
    assert restarted.find_by_bound_chat_id("oc_legacy_active") is not None


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_owner_adopt_replaces_resolution_required_legacy_saga_after_validation(
    client_cls, tmp_path, project_manager, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation
    from src.slock_engine.manager import SlockEngineManager

    root = str(tmp_path / "legacy-adopt")
    legacy_operation = "new-chat:ou_owner:legacy-adopt"
    success, _, project = project_manager.create_project_with_managed_chat_saga(
        project_id="legacy-adopt",
        project_name="legacy-adopt",
        root_path=root,
        owner_chat_id="oc_legacy_adopt",
        managed_chat_id="oc_legacy_adopt",
        managed_chat_name="legacy-adopt-dev",
        created_at=123.0,
        operation_id=legacy_operation,
        expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        expected_owner_id="ou_owner",
        expected_receiving_bot_ref="cli_main_bot",
    )
    assert success and project is not None
    storage = tmp_path / "projects.json"
    _rewrite_binding_saga_as_legacy(storage, legacy_operation)
    restarted = ProjectManager(storage_path=str(storage))
    client = _startup_reconciler(
        tmp_path,
        registry,
        SlockEngineManager(storage_base_path=str(tmp_path / "slock")),
        client_cls.return_value,
        restarted,
    )
    client._reconcile_managed_groups_before_slock_restore()
    assert restarted.managed_group_residual(legacy_operation) == (
        "oc_legacy_adopt",
        "legacy_saga_resolution_required",
    )
    client_cls.return_value.validate_managed_chat.return_value = (
        ManagedChatValidation.VALID
    )
    handler = _adoption_handler(restarted, registry)

    handler.adopt_managed_chat(
        "om_adopt", "oc_legacy_adopt", "legacy-adopt"
    )

    active = registry.active_record("oc_legacy_adopt")
    assert active is not None
    assert active.origin is ManagedGroupOrigin.OWNER_ADOPTED
    assert restarted.pending_managed_chat_binding_sagas() == ()
    assert restarted.managed_group_residual(legacy_operation) is None
    assert handler.reply_error.call_count == 0


@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_owner_adopt_failure_restores_legacy_resolution_for_retry(
    client_cls, tmp_path, project_manager, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation
    from src.slock_engine.manager import SlockEngineManager

    root = str(tmp_path / "legacy-adopt-retry")
    legacy_operation = "new-chat:ou_owner:legacy-adopt-retry"
    success, _, project = project_manager.create_project_with_managed_chat_saga(
        project_id="legacy-adopt-retry",
        project_name="legacy-adopt-retry",
        root_path=root,
        owner_chat_id="oc_legacy_adopt_retry",
        managed_chat_id="oc_legacy_adopt_retry",
        managed_chat_name="legacy-adopt-retry-dev",
        created_at=123.0,
        operation_id=legacy_operation,
        expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        expected_owner_id="ou_owner",
        expected_receiving_bot_ref="cli_main_bot",
    )
    assert success and project is not None
    storage = tmp_path / "projects.json"
    _rewrite_binding_saga_as_legacy(storage, legacy_operation)
    restarted = ProjectManager(storage_path=str(storage))
    client = _startup_reconciler(
        tmp_path,
        registry,
        SlockEngineManager(storage_base_path=str(tmp_path / "slock")),
        client_cls.return_value,
        restarted,
    )
    client._reconcile_managed_groups_before_slock_restore()
    client_cls.return_value.validate_managed_chat.return_value = (
        ManagedChatValidation.VALID
    )
    handler = _adoption_handler(restarted, registry)
    original_activate = registry.activate
    attempts = 0

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            durable = ProjectManager(storage_path=str(storage))
            durable_pending = durable.pending_managed_chat_binding_sagas()
            assert len(durable_pending) == 2
            adoption = next(
                saga for saga in durable_pending if saga.expected is not None
            )
            assert adoption.displaced_legacy_operation_id == legacy_operation
            assert durable.find_by_bound_chat_id("oc_legacy_adopt_retry") is None
            raise OSError("registry unavailable")
        return original_activate(**kwargs)

    registry.activate = MagicMock(side_effect=fail_once)

    handler.adopt_managed_chat(
        "om_failed", "oc_legacy_adopt_retry", "legacy-adopt-retry"
    )

    pending = restarted.pending_managed_chat_binding_sagas()
    assert [saga.operation_id for saga in pending] == [legacy_operation]
    assert pending[0].expected is None
    assert restarted.managed_group_residual(legacy_operation) == (
        "oc_legacy_adopt_retry",
        "legacy_saga_resolution_required",
    )
    assert restarted.find_by_bound_chat_id("oc_legacy_adopt_retry") is None

    handler.adopt_managed_chat(
        "om_retry", "oc_legacy_adopt_retry", "legacy-adopt-retry"
    )

    assert registry.active_record("oc_legacy_adopt_retry") is not None
    assert restarted.pending_managed_chat_binding_sagas() == ()
    assert restarted.managed_group_residual(legacy_operation) is None
    assert restarted.find_by_bound_chat_id("oc_legacy_adopt_retry") is not None


@patch("src.slock_engine.engine.create_engine_session")
@patch("src.thread.manager.get_current_sender_id", return_value="ou_owner")
@patch("src.project_chat.lark_chat_client.LarkChatClient")
def test_team_active_cleanup_retry_only_finalizes_existing_group(
    client_cls, _sender, _session, tmp_path, registry
):
    from src.project_chat.lark_chat_client import ManagedChatValidation

    handler = _make_slock_handler(tmp_path, registry)
    remote = client_cls.return_value
    remote.create_chat.return_value = _FakeCreateChatResult(
        chat_id="oc_team_cleanup_retry",
        name="Cleanup Retry [Slock]",
    )
    original_bind = registry.bind_provision_chat
    bind_attempts = 0

    def fail_bind_once(operation_id, chat_id):
        nonlocal bind_attempts
        bind_attempts += 1
        if bind_attempts == 1:
            raise OSError("registry unavailable")
        return original_bind(operation_id, chat_id)

    registry.bind_provision_chat = MagicMock(side_effect=fail_bind_once)
    handler.create_team("om_first", "oc_owner_p2p", "Cleanup Retry")
    remote.validate_managed_chat.return_value = ManagedChatValidation.VALID
    manager = handler.ctx.slock_engine_manager
    cleanup_dir = tmp_path / "slock" / "pending_cleanup"
    original_fsync = manager._fsync_directory
    cleanup_fsync_failures = 0

    def fail_first_post_unlink_fsync(path):
        nonlocal cleanup_fsync_failures
        if Path(path) == cleanup_dir and cleanup_fsync_failures == 0:
            cleanup_fsync_failures += 1
            raise OSError("pending cleanup parent fsync failed")
        return original_fsync(path)

    manager._fsync_directory = MagicMock(
        side_effect=fail_first_post_unlink_fsync
    )
    manager.block_team_name_for_cleanup = MagicMock(return_value=False)
    registry.activate = MagicMock(wraps=registry.activate)

    handler.create_team("om_activate", "oc_owner_p2p", "Cleanup Retry")

    assert manager.retained_team_chat_id("Cleanup Retry") == (
        "oc_team_cleanup_retry"
    )
    assert not list(cleanup_dir.glob("*.json"))
    manager.block_team_name_for_cleanup.assert_called_once_with(
        "Cleanup Retry",
        "oc_team_cleanup_retry",
        "untrusted_retained",
    )
    assert "未能持久清理" in handler.reply_text.call_args.args[1]

    handler.create_team("om_finalize", "oc_owner_p2p", "Cleanup Retry")

    assert remote.create_chat.call_count == 1
    assert registry.activate.call_count == 1
    assert cleanup_fsync_failures == 1
    assert not list(cleanup_dir.glob("*.json"))
    assert manager.retained_team_chat_id("Cleanup Retry") is None
    assert "cleanup retry" not in manager._blocked_team_names
    assert "已存在" in handler.reply_text.call_args.args[1]
