"""Task 0.9 vertical contracts for Project/Team managed-group activation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.project.manager import ProjectManager
from src.project_chat.lark_chat_client import CreateChatResult
from src.project_chat.service import ProjectChatService
from src.trust.models import ManagedGroupOrigin, ManagedGroupStatus
from src.trust.registry import ManagedGroupRegistry


@pytest.fixture
def project_manager(tmp_path):
    return ProjectManager(storage_path=str(tmp_path / "projects.json"))


@pytest.fixture
def registry(tmp_path):
    return ManagedGroupRegistry(tmp_path / "managed-groups.json")


@pytest.fixture
def lark_client():
    client = MagicMock()
    client.create_chat.return_value = CreateChatResult(
        chat_id="oc_project_group",
        name="project-dev",
    )
    client.delete_chat.return_value = True
    return client


def _project_service(project_manager, lark_client, registry, events):
    def reply(message_id, text, msg_type):
        events.append(("reply", message_id, msg_type))

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
    lark_client.delete_chat.assert_called_once_with("oc_project_group")
    assert registry.active_record("oc_project_group") is None
    project = project_manager.find_project_by_path(root, chat_id=None)
    assert project is None or not project.bound_chat_id


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
