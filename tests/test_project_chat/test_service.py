"""Tests for ProjectChatService — the /new-chat orchestrator."""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.project.manager import ProjectManager
from src.project_chat.errors import CreateChatError
from src.project_chat.lark_chat_client import CreateChatResult
from src.project_chat.service import ProjectChatService


@pytest.fixture
def tmp_storage(tmp_path):
    return str(tmp_path / "projects.json")


@pytest.fixture
def project_manager(tmp_storage):
    return ProjectManager(storage_path=tmp_storage)


@pytest.fixture
def mock_lark_client():
    client = MagicMock()
    client.create_chat.return_value = CreateChatResult(
        chat_id="oc_new_group_123",
        name="testproj-dev",
    )
    return client


@pytest.fixture
def mock_reply_fn():
    return MagicMock()


@pytest.fixture
def service(project_manager, mock_lark_client, mock_reply_fn):
    return ProjectChatService(
        project_manager=project_manager,
        lark_chat_client=mock_lark_client,
        reply_fn=mock_reply_fn,
        send_to_chat_fn=mock_reply_fn,
    )


class TestNewProject:
    """Branch C: no existing project → create chat + create project."""

    def test_creates_project_and_chat(self, service, project_manager, mock_lark_client, tmp_path):
        path = str(tmp_path / "mycode")
        os.makedirs(path)

        service.handle(
            message_id="msg_1",
            chat_id="oc_main_chat",
            sender_open_id="ou_user_1",
            data={"name": "mycode", "path": path},
        )

        # Verify chat was created
        mock_lark_client.create_chat.assert_called_once()
        call_kwargs = mock_lark_client.create_chat.call_args[1]
        assert "mycode" in call_kwargs["name"]
        assert "ou_user_1" in call_kwargs["user_id_list"]

        # Verify sender was promoted to group manager
        mock_lark_client.add_managers.assert_called_once_with(
            "oc_new_group_123", ["ou_user_1"]
        )

        # Verify project was created with bound_chat_id
        ctx = project_manager.find_project_by_path(path)
        assert ctx is not None
        assert ctx.bound_chat_id == "oc_new_group_123"
        assert ctx.bound_chat_name == "testproj-dev"
        assert ctx.owner_chat_id == "oc_new_group_123"

    def test_idempotent_returns_existing(self, service, project_manager, mock_lark_client, tmp_path):
        """Branch A: project exists with bound_chat → no API call, return jump card."""
        path = str(tmp_path / "existing")
        os.makedirs(path)

        # Pre-create project with bound chat
        success, _, ctx = project_manager.create_project(
            project_id=None, project_name="existing", root_path=path, chat_id="oc_bound"
        )
        assert success
        ctx.bound_chat_id = "oc_bound"
        ctx.bound_chat_name = "existing-dev"
        project_manager._save_projects()

        service.handle(
            message_id="msg_2",
            chat_id="oc_main_chat",
            sender_open_id="ou_user_1",
            data={"name": "existing", "path": path},
        )

        # Should NOT create a new chat
        mock_lark_client.create_chat.assert_not_called()

        mock_reply_fn = service._reply
        message_id, card_json, msg_type = mock_reply_fn.call_args[0]
        assert message_id == "msg_2"
        assert msg_type == "interactive"
        card = json.loads(card_json)
        assert "existing-dev" in json.dumps(card, ensure_ascii=False)
        assert "openChatId=oc_bound" in json.dumps(card, ensure_ascii=False)

        # Key: originating chat must be able to see the project in the board
        projects = project_manager.get_all_projects(chat_id="oc_main_chat")
        assert any(p.project_id == ctx.project_id for p in projects), \
            "Branch A must add originating chat to allowed_chat_ids for board visibility"


class TestLegacyProjectBind:
    """Branch B: existing project without bound chat → bind it."""

    def test_bind_legacy_project_preserves_visibility(self, service, project_manager, mock_lark_client, tmp_path):
        """After binding, the project must remain visible from the originating chat."""
        path = str(tmp_path / "legacy")
        os.makedirs(path)

        # Pre-create a legacy project (empty allowed_chat_ids → visible to all)
        success, _, ctx = project_manager.create_project(
            project_id=None, project_name="legacy", root_path=path, chat_id=None
        )
        assert success
        assert not ctx.bound_chat_id

        service.handle(
            message_id="msg_bind",
            chat_id="oc_main_chat",
            sender_open_id="ou_user_1",
            data={"name": "ghostAp", "path": path},
        )

        # Verify bound
        ctx = project_manager.find_project_by_path(path)
        assert ctx.bound_chat_id == "oc_new_group_123"

        # Key assertion: user-specified name must override the old name
        assert ctx.project_name == "ghostAp"

        # Key assertion: project must still be visible from the originating chat
        projects = project_manager.get_all_projects(chat_id="oc_main_chat")
        assert any(p.project_id == ctx.project_id for p in projects), \
            "Project should remain visible from the originating chat after binding"

        # Also visible from the new group chat
        projects = project_manager.get_all_projects(chat_id="oc_new_group_123")
        assert any(p.project_id == ctx.project_id for p in projects)


class TestLegacyProjectPathMismatch:
    """Legacy project exists by name but path doesn't match current cwd."""

    def test_find_by_name_fallback_creates_chat_and_binds(
        self, service, project_manager, mock_lark_client, tmp_path
    ):
        """If find_project_by_path misses but find_project_by_name hits,
        should update root_path and bind (Branch B)."""
        old_path = str(tmp_path / "old_location")
        new_path = str(tmp_path / "cocoforclaw")
        os.makedirs(old_path)
        os.makedirs(new_path)

        # Pre-create project with OLD path (simulates legacy project)
        success, _, ctx = project_manager.create_project(
            project_id=None, project_name="cocoforclaw", root_path=old_path, chat_id=None
        )
        assert success
        assert not ctx.bound_chat_id

        # User runs /new-chat from NEW path (name matches existing project)
        service.handle(
            message_id="msg_legacy",
            chat_id="oc_main_chat",
            sender_open_id="ou_user_1",
            data={"name": "cocoforclaw", "path": new_path},
        )

        # Should create chat (Branch B) — not fail with "already exists"
        mock_lark_client.create_chat.assert_called_once()

        # Project should be updated with new path and bound
        ctx = project_manager.get_project_for_diagnostics("cocoforclaw")
        assert ctx is not None
        assert ctx.root_path == new_path
        assert ctx.working_dir == new_path
        assert ctx.bound_chat_id == "oc_new_group_123"

    def test_find_by_name_fallback_already_bound_returns_jump_card(
        self, service, project_manager, mock_lark_client, tmp_path
    ):
        """If legacy project found by name already has bound chat → Branch A."""
        old_path = str(tmp_path / "old_location")
        new_path = str(tmp_path / "myproj")
        os.makedirs(old_path)
        os.makedirs(new_path)

        # Pre-create project with OLD path and bound chat
        success, _, ctx = project_manager.create_project(
            project_id=None, project_name="myproj", root_path=old_path, chat_id="oc_old"
        )
        assert success
        ctx.bound_chat_id = "oc_existing_group"
        ctx.bound_chat_name = "myproj-dev"
        project_manager._save_projects()

        # User runs /new-chat from NEW path
        service.handle(
            message_id="msg_bound",
            chat_id="oc_main_chat",
            sender_open_id="ou_user_1",
            data={"name": "myproj", "path": new_path},
        )

        # Should NOT create a new chat (Branch A)
        mock_lark_client.create_chat.assert_not_called()

        # Root path should be updated
        ctx = project_manager.get_project_for_diagnostics("myproj")
        assert ctx.root_path == new_path


class TestRollback:
    """Verify rollback on failure after chat creation."""

    def test_rollback_on_project_create_failure(self, service, project_manager, mock_lark_client, tmp_path):
        """A known remote group is retained untrusted when local creation fails."""
        path = "/nonexistent/impossible/path/that/will/fail_create"

        # ProjectManager.create_project will fail for this path (can't mkdir)
        # Actually, let's mock it to fail
        with patch.object(
            project_manager,
            "create_project_with_managed_chat_saga",
            return_value=(False, "disk error", None),
        ):
            service.handle(
                message_id="msg_3",
                chat_id="oc_main",
                sender_open_id="ou_user_1",
                data={"name": "broken", "path": path},
            )

        mock_lark_client.delete_chat.assert_not_called()
        operation_id = f"new-chat::broken:{os.path.normpath(path)}"
        assert project_manager.managed_group_residual(operation_id) == (
            "oc_new_group_123",
            "untrusted_retained",
        )

    def test_no_rollback_on_chat_create_failure(self, service, mock_lark_client, tmp_path):
        """If create_chat fails, no rollback needed."""
        path = str(tmp_path / "newdir")
        os.makedirs(path)
        mock_lark_client.create_chat.side_effect = CreateChatError("API error")

        service.handle(
            message_id="msg_4",
            chat_id="oc_main",
            sender_open_id="ou_user_1",
            data={"name": "proj", "path": path},
        )

        mock_lark_client.delete_chat.assert_not_called()
