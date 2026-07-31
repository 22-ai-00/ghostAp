"""Task 0.9 vertical contracts for Project/Team managed-group activation."""

from __future__ import annotations

import os
from dataclasses import dataclass
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


def test_group_ledger_guard_rejects_unknown_before_context_write(tmp_path):
    from src.autonomous.context.group_ledger import managed_group_context_allowed

    assert managed_group_context_allowed(None) is False
    assert managed_group_context_allowed(
        registry_record=MagicMock(status=ManagedGroupStatus.TOMBSTONED)
    ) is False


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
