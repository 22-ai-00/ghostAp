"""Tests for /new-team command: create Feishu group + activate slock runtime."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

_acp_available = pytest.importorskip("acp", reason="acp package not installed")

from src.slock_engine.manager import SlockEngineManager  # noqa: E402
from src.slock_engine.models import SlockChannel  # noqa: E402

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@dataclass
class _FakeCreateChatResult:
    chat_id: str
    name: str


def _make_handler_ctx(tmp_path, *, api_client_factory=None, settings=None):
    """Build a minimal HandlerContext-like object for SlockHandler."""
    ctx = MagicMock()
    ctx.api_client_factory = api_client_factory or MagicMock()
    ctx.project_manager = MagicMock()

    # Settings
    s = MagicMock()
    s.slock_team_name_suffix = "[Slock]"
    s.slock_workspace_base = ""
    s.slock_memory_base_path = str(tmp_path / "slock_memory")
    s.slock_execution_timeout = 3600
    s.slock_max_agents_per_team = 10
    s.slock_task_claim_ttl = 3600.0
    s.slock_default_roles = ""
    # Activation guard: allow all users in tests
    s.admin_user_ids = ""
    s.slock_auto_activate_whitelist_user_ids = ""
    s.slock_auto_activate_default_policy = "allow_all"
    s.slock_passive_mode = True
    s.slock_auto_activate_rate_limit = 100
    if settings:
        for k, v in settings.items():
            setattr(s, k, v)
    ctx.settings = s

    # Slock engine manager — use real instance
    manager = SlockEngineManager(storage_base_path=s.slock_memory_base_path)
    ctx.slock_engine_manager = manager

    # Handler registry (minimal)
    ctx.handlers = {}
    ctx.mode_manager = MagicMock()
    ctx.context_manager = MagicMock()
    ctx.scheduler = MagicMock()
    return ctx


def _make_slock_handler(ctx):
    """Instantiate SlockHandler with mocked IM client."""
    with patch("src.feishu.handlers.base.FeishuIMClient"):
        from src.feishu.handlers.slock import SlockHandler

        handler = SlockHandler(ctx)
        # Mock messaging helpers
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.send_text_to_chat = MagicMock()
        handler.send_card_to_chat = MagicMock()
        handler.add_reaction = MagicMock()
        handler.get_working_dir = MagicMock(return_value="/tmp/workdir")
        handler.create_static_card_session = MagicMock()
        return handler


# ==================================================================
# Test: Happy path — group created, engine activated, messages sent
# ==================================================================


class TestCreateTeamHappyPath:
    @patch("src.slock_engine.engine.create_engine_session")
    @patch("src.thread.manager.get_current_sender_id", return_value="ou_sender123")
    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_create_team_full_flow(
        self, MockLarkChatClient, mock_sender, mock_session, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)

        # Set up root_path for workspace verification
        root_path = str(tmp_path / "project_root")
        os.makedirs(root_path)
        handler.get_working_dir.return_value = root_path

        # Mock LarkChatClient
        mock_lark = MockLarkChatClient.return_value
        mock_lark.create_chat.return_value = _FakeCreateChatResult(
            chat_id="oc_new_group_id", name="TestTeam"
        )

        # Execute
        handler.create_team("msg1", "oc_origin_chat", "TestTeam", project=None)

        # 1. Group created with correct params
        mock_lark.create_chat.assert_called_once()
        call_kwargs = mock_lark.create_chat.call_args
        assert call_kwargs.kwargs["name"] == "TestTeam [Slock]"
        assert "ou_sender123" in call_kwargs.kwargs["user_id_list"]

        # 2. Sender promoted to manager
        mock_lark.add_managers.assert_called_once_with("oc_new_group_id", ["ou_sender123"])

        # 3. Engine activated on new chat_id
        manager = ctx.slock_engine_manager
        engine = manager.get_activated_engine("oc_new_group_id")
        assert engine is not None
        assert engine.channel is not None
        assert engine.channel.channel_id == "oc_new_group_id"
        assert engine.channel.team_name == "TestTeam"

        # 4. Group marker exists in the configured app storage, not the repo
        workspace_dir = engine.memory.get_group_base_path("oc_new_group_id")
        assert os.path.isdir(workspace_dir)
        marker_path = os.path.join(workspace_dir, ".slock_channel.json")
        assert os.path.isfile(marker_path)
        with open(marker_path, "r") as f:
            marker = json.load(f)
        assert marker["channel_id"] == "oc_new_group_id"
        assert not os.path.exists(os.path.join(root_path, "slock", "oc_new_group_id"))

        # 5. Managed chat registered for event routing
        assert manager.is_managed_chat("oc_new_group_id") is True

        # 6. Welcome card sent to new group
        handler.send_card_to_chat.assert_called_once()
        new_group_args = handler.send_card_to_chat.call_args[0]
        assert new_group_args[0] == "oc_new_group_id"
        welcome_card = json.loads(new_group_args[1])
        welcome_blob = json.dumps(welcome_card, ensure_ascii=False)
        assert "TestTeam" in welcome_blob
        assert "/hire" in welcome_blob or "/role add" in welcome_blob
        assert welcome_card["schema"] == "2.0"

        # 7. Confirmation card sent to original group with a direct group jump button
        handler.reply_card.assert_called_once()
        reply_args = handler.reply_card.call_args[0]
        assert reply_args[0] == "msg1"
        card = json.loads(reply_args[1])
        card_blob = json.dumps(card, ensure_ascii=False)
        assert "TestTeam" in card_blob
        assert "事件监听" in card_blob
        assert "进入 Slock 群" in card_blob
        assert "openChatId=oc_new_group_id" in card_blob

    @patch("src.slock_engine.engine.create_engine_session")
    @patch("src.thread.manager.get_current_sender_id", return_value="ou_sender123")
    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_create_team_with_suffix(
        self, MockLarkChatClient, mock_sender, mock_session, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path, settings={"slock_team_name_suffix": "-Slock"})
        handler = _make_slock_handler(ctx)
        handler.get_working_dir.return_value = str(tmp_path / "root")

        mock_lark = MockLarkChatClient.return_value
        mock_lark.create_chat.return_value = _FakeCreateChatResult(
            chat_id="oc_prefixed", name="Alpha-Slock"
        )

        handler.create_team("msg2", "oc_origin", "Alpha")

        call_kwargs = mock_lark.create_chat.call_args.kwargs
        assert call_kwargs["name"] == "Alpha-Slock"

    @patch("src.slock_engine.engine.create_engine_session")
    @patch("src.thread.manager.get_current_sender_id", return_value="ou_sender123")
    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_create_team_does_not_duplicate_existing_slock_marker(
        self, MockLarkChatClient, mock_sender, mock_session, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        handler.get_working_dir.return_value = str(tmp_path / "root")

        mock_lark = MockLarkChatClient.return_value
        mock_lark.create_chat.return_value = _FakeCreateChatResult(
            chat_id="oc_marked", name="Alpha [Slock]"
        )

        handler.create_team("msg3", "oc_origin", "Alpha [Slock]")

        call_kwargs = mock_lark.create_chat.call_args.kwargs
        assert call_kwargs["name"] == "Alpha [Slock]"

    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_dissolve_team_deletes_feishu_group(
        self, MockLarkChatClient, tmp_path
    ):
        """`/team dissolve` stops local runtime and dissolves the Feishu group."""
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        handler._check_slock_permission = MagicMock(return_value=True)

        manager = ctx.slock_engine_manager
        engine = manager.get_or_create("oc_team", str(tmp_path / "root"), engine_name="Slock")
        engine.activate_channel(SlockChannel(channel_id="oc_team", name="Alpha [Slock]", team_name="Alpha"))
        manager.register_managed_chat("oc_team")

        handler.dissolve_team("msg-dissolve", "oc_owner", "Alpha")

        MockLarkChatClient.return_value.delete_chat.assert_called_once_with("oc_team")
        assert manager.is_managed_chat("oc_team") is False
        marker = tmp_path / "slock_memory" / "groups" / "oc_team" / ".slock_channel.json"
        assert not marker.exists()
        handler.reply_text.assert_called_once()
        assert "已解散" in handler.reply_text.call_args[0][1]

    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_dissolve_team_keeps_runtime_when_marker_archive_fails(
        self, MockLarkChatClient, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        handler._check_slock_permission = MagicMock(return_value=True)
        manager = ctx.slock_engine_manager
        engine = manager.get_or_create("oc_team", str(tmp_path / "root"), engine_name="Slock")
        engine.activate_channel(
            SlockChannel(channel_id="oc_team", name="Alpha [Slock]", team_name="Alpha")
        )
        manager.register_managed_chat("oc_team")
        manager.archive_managed_chat_marker = MagicMock(side_effect=OSError("disk full"))

        handler.dissolve_team("msg-dissolve", "oc_owner", "Alpha")

        assert manager.is_managed_chat("oc_team") is True
        assert manager.get_activated_engine("oc_team") is engine
        MockLarkChatClient.return_value.delete_chat.assert_not_called()
        assert "未解散" in handler.reply_text.call_args.args[1]

    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_dissolve_team_restores_marker_when_runtime_teardown_fails(
        self, MockLarkChatClient, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        handler._check_slock_permission = MagicMock(return_value=True)
        manager = ctx.slock_engine_manager
        engine = manager.get_or_create("oc_team", str(tmp_path / "root"), engine_name="Slock")
        engine.activate_channel(
            SlockChannel(channel_id="oc_team", name="Alpha [Slock]", team_name="Alpha")
        )
        manager.register_managed_chat("oc_team")
        real_restore = manager.restore_archived_chat_marker
        manager.restore_archived_chat_marker = MagicMock(side_effect=real_restore)
        manager.remove = MagicMock(side_effect=RuntimeError("cleanup failed"))

        handler.dissolve_team("msg-dissolve", "oc_owner", "Alpha")

        manager.restore_archived_chat_marker.assert_called_once()
        marker = tmp_path / "slock_memory" / "groups" / "oc_team" / ".slock_channel.json"
        assert marker.exists()
        assert manager.is_managed_chat("oc_team") is True
        assert manager.get_activated_engine("oc_team") is not engine
        MockLarkChatClient.return_value.delete_chat.assert_not_called()
        assert "未完整解散" in handler.reply_text.call_args.args[1]

    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_dissolve_team_restores_local_team_when_feishu_delete_fails(
        self, MockLarkChatClient, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        handler._check_slock_permission = MagicMock(return_value=True)
        manager = ctx.slock_engine_manager
        engine = manager.get_or_create("oc_team", str(tmp_path / "root"), engine_name="Slock")
        engine.activate_channel(
            SlockChannel(channel_id="oc_team", name="Alpha [Slock]", team_name="Alpha")
        )
        manager.register_managed_chat("oc_team")
        MockLarkChatClient.return_value.delete_chat.return_value = False

        handler.dissolve_team("msg-dissolve", "oc_owner", "Alpha")

        assert manager.is_managed_chat("oc_team") is True
        assert manager.get_activated_engine("oc_team") is not None
        marker = tmp_path / "slock_memory" / "groups" / "oc_team" / ".slock_channel.json"
        assert marker.exists()
        assert "未解散" in handler.reply_text.call_args.args[1]

    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_dissolve_team_does_not_revive_ghost_when_delete_result_is_unknown(
        self, MockLarkChatClient, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        handler._check_slock_permission = MagicMock(return_value=True)
        manager = ctx.slock_engine_manager
        engine = manager.get_or_create("oc_team", str(tmp_path / "root"), engine_name="Slock")
        engine.activate_channel(
            SlockChannel(channel_id="oc_team", name="Alpha [Slock]", team_name="Alpha")
        )
        manager.register_managed_chat("oc_team")
        MockLarkChatClient.return_value.delete_chat.return_value = None

        handler.dissolve_team("msg-dissolve", "oc_owner", "Alpha")

        assert manager.is_managed_chat("oc_team") is False
        marker = tmp_path / "slock_memory" / "groups" / "oc_team" / ".slock_channel.json"
        assert not marker.exists()
        assert "人工确认" in handler.reply_text.call_args.args[1]

    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_duplicate_dissolve_is_rejected_by_per_chat_claim(
        self, MockLarkChatClient, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        handler._check_slock_permission = MagicMock(return_value=True)
        manager = ctx.slock_engine_manager
        engine = manager.get_or_create("oc_team", str(tmp_path / "root"), engine_name="Slock")
        engine.activate_channel(
            SlockChannel(channel_id="oc_team", name="Alpha [Slock]", team_name="Alpha")
        )
        manager.register_managed_chat("oc_team")
        assert manager.claim_dissolve("oc_team") is True

        try:
            handler.dissolve_team("msg-dissolve", "oc_owner", "Alpha")
        finally:
            manager.release_dissolve("oc_team")

        MockLarkChatClient.return_value.delete_chat.assert_not_called()
        assert manager.is_managed_chat("oc_team") is True
        assert "重复提交" in handler.reply_text.call_args.args[1]


# ==================================================================
# Test: Failure & rollback
# ==================================================================


class TestCreateTeamRollback:
    @patch("src.slock_engine.engine.create_engine_session")
    @patch("src.thread.manager.get_current_sender_id", return_value="ou_sender123")
    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_create_chat_failure_shows_error(
        self, MockLarkChatClient, mock_sender, mock_session, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)

        mock_lark = MockLarkChatClient.return_value
        mock_lark.create_chat.side_effect = RuntimeError("API limit exceeded")

        handler.create_team("msg1", "oc_origin", "FailTeam")

        # Error reported to user
        handler.reply_text.assert_called_once()
        err_text = handler.reply_text.call_args[0][1]
        assert "创建团队群失败" in err_text

        # No group to delete (never created)
        mock_lark.delete_chat.assert_not_called()

    @patch("src.slock_engine.engine.create_engine_session")
    @patch("src.thread.manager.get_current_sender_id", return_value="ou_sender123")
    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_activation_failure_retains_untrusted_group(
        self, MockLarkChatClient, mock_sender, mock_session, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)

        mock_lark = MockLarkChatClient.return_value
        mock_lark.create_chat.return_value = _FakeCreateChatResult(
            chat_id="oc_to_delete", name="BadTeam"
        )
        mock_lark.delete_chat.return_value = True

        # Force engine activation to fail
        ctx.slock_engine_manager.get_or_create_activated = MagicMock(
            side_effect=RuntimeError("Engine init failed")
        )

        handler.create_team("msg1", "oc_origin", "BadTeam")

        # Automatic deletion is unsafe against concurrent lifecycle changes.
        mock_lark.delete_chat.assert_not_called()
        restarted = SlockEngineManager(
            storage_base_path=ctx.settings.slock_memory_base_path
        )
        assert restarted.reserve_team_name("BadTeam") is False

        # Error reported to user
        handler.reply_text.assert_called_once()
        err_text = handler.reply_text.call_args[0][1]
        assert "保留" in err_text

    @patch("src.slock_engine.engine.create_engine_session")
    @patch("src.thread.manager.get_current_sender_id", return_value="ou_sender123")
    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_welcome_card_failure_falls_back_without_rolling_back_team(
        self, MockLarkChatClient, mock_sender, mock_session, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        mock_lark = MockLarkChatClient.return_value
        mock_lark.create_chat.return_value = _FakeCreateChatResult(
            chat_id="oc_partial", name="PartialTeam"
        )
        handler.send_card_to_chat.return_value = None

        handler.create_team("msg1", "oc_origin", "PartialTeam")

        assert ctx.slock_engine_manager.is_managed_chat("oc_partial") is True
        marker = tmp_path / "slock_memory" / "groups" / "oc_partial" / ".slock_channel.json"
        assert marker.exists()
        mock_lark.delete_chat.assert_not_called()
        handler.send_text_to_chat.assert_called_once()
        assert "欢迎卡发送失败" in handler.send_text_to_chat.call_args.args[1]

    @patch("src.slock_engine.engine.create_engine_session")
    @patch("src.thread.manager.get_current_sender_id", return_value="ou_sender123")
    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_origin_confirmation_card_failure_falls_back_to_text(
        self, MockLarkChatClient, mock_sender, mock_session, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        mock_lark = MockLarkChatClient.return_value
        mock_lark.create_chat.return_value = _FakeCreateChatResult(
            chat_id="oc_created", name="CreatedTeam"
        )
        handler.send_card_to_chat.return_value = "om_welcome"
        handler.reply_card.return_value = None

        handler.create_team("msg1", "oc_origin", "CreatedTeam")

        assert ctx.slock_engine_manager.is_managed_chat("oc_created") is True
        mock_lark.delete_chat.assert_not_called()
        assert "团队已创建" in handler.reply_text.call_args.args[1]

    @pytest.mark.parametrize("delete_result", [False, None])
    @patch("src.slock_engine.engine.create_engine_session")
    @patch("src.thread.manager.get_current_sender_id", return_value="ou_sender123")
    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_failed_rollback_persistently_blocks_same_team_name(
        self, MockLarkChatClient, mock_sender, mock_session, delete_result, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        mock_lark = MockLarkChatClient.return_value
        mock_lark.create_chat.return_value = _FakeCreateChatResult(
            chat_id="oc_residual", name="ResidualTeam"
        )
        mock_lark.delete_chat.return_value = delete_result
        ctx.slock_engine_manager.get_or_create_activated = MagicMock(
            side_effect=RuntimeError("Engine init failed")
        )

        handler.create_team("msg1", "oc_origin", "ResidualTeam")

        assert ctx.slock_engine_manager.reserve_team_name("residualteam") is False
        restarted = SlockEngineManager(
            storage_base_path=str(tmp_path / "slock_memory")
        )
        assert restarted.reserve_team_name("ResidualTeam") is False
        records = list((tmp_path / "slock_memory" / "pending_cleanup").glob("*.json"))
        assert len(records) == 1

    @patch("src.thread.manager.get_current_sender_id", return_value="ou_sender123")
    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_duplicate_active_team_name_is_rejected_before_creating_group(
        self, MockLarkChatClient, mock_sender, tmp_path
    ):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        manager = ctx.slock_engine_manager
        engine = manager.get_or_create("oc_existing", str(tmp_path), engine_name="Slock")
        engine.activate_channel(
            SlockChannel(channel_id="oc_existing", name="Alpha [Slock]", team_name="Alpha")
        )
        manager.register_managed_chat("oc_existing")

        handler.create_team("msg1", "oc_origin", "alpha")

        MockLarkChatClient.return_value.create_chat.assert_not_called()
        assert "已存在" in handler.reply_text.call_args.args[1]

    def test_empty_name_shows_usage(self, tmp_path):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)

        handler.create_team("msg1", "oc_origin", "")

        handler.reply_text.assert_called_once()
        assert "用法" in handler.reply_text.call_args[0][1]

    @patch("src.thread.manager.get_current_sender_id", return_value="")
    def test_no_sender_shows_error(self, mock_sender, tmp_path):
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)

        handler.create_team("msg1", "oc_origin", "Team")

        handler.reply_text.assert_called_once()
        assert "无法获取发送者" in handler.reply_text.call_args[0][1]


# ==================================================================
# Test: Dispatcher routes managed chat messages to slock engine
# ==================================================================


class TestDispatcherSlockRouting:
    def test_managed_chat_detected(self):
        """SlockEngineManager.is_managed_chat returns True after registration."""
        manager = SlockEngineManager()
        assert manager.is_managed_chat("oc_test") is False
        manager.register_managed_chat("oc_test")
        assert manager.is_managed_chat("oc_test") is True

    def test_unregister_managed_chat(self):
        manager = SlockEngineManager()
        manager.register_managed_chat("oc_test")
        manager.unregister_managed_chat("oc_test")
        assert manager.is_managed_chat("oc_test") is False


# ==================================================================
# Test: activate_slock passes owner_id (AC-14)
# ==================================================================


class TestActivateSlockOwnerId:
    """AC-14: activate_slock() must pass sender_open_id as owner_id to SlockChannel."""

    @patch("src.slock_engine.activation_guard.get_activation_guard")
    @patch("src.slock_engine.engine.create_engine_session")
    @patch("src.thread.manager.get_current_sender_id", return_value="user-sender-123")
    def test_activate_slock_passes_owner_id(self, mock_sender, mock_session, mock_guard, tmp_path):
        """Channel created via /slock activate should have owner_id == sender_open_id."""
        mock_guard.return_value.can_auto_activate.return_value = (True, "allowed")
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        project = MagicMock()
        project.project_name = "TestProject"
        project.project_id = "proj-001"
        project.root_path = str(tmp_path / "root")
        handler._ensure_project = MagicMock(return_value=project)

        handler.activate_slock("msg-001", "oc_activate_test", project=project)

        # Verify the channel has owner_id set
        manager = ctx.slock_engine_manager
        engine = manager.get_activated_engine("oc_activate_test")
        assert engine is not None
        assert engine.channel is not None
        assert engine.channel.owner_id == "user-sender-123"

    @patch("src.slock_engine.activation_guard.get_activation_guard")
    @patch("src.slock_engine.engine.create_engine_session")
    @patch("src.thread.manager.get_current_sender_id", return_value="")
    def test_activate_slock_empty_sender_still_works(self, mock_sender, mock_session, mock_guard, tmp_path):
        """Even with empty sender_id, activation should not crash (owner_id='')."""
        mock_guard.return_value.can_auto_activate.return_value = (True, "allowed")
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        project = MagicMock()
        project.project_name = "TestProject"
        project.project_id = "proj-001"
        project.root_path = str(tmp_path / "root")
        handler._ensure_project = MagicMock(return_value=project)

        handler.activate_slock("msg-002", "oc_activate_empty", project=project)

        manager = ctx.slock_engine_manager
        engine = manager.get_activated_engine("oc_activate_empty")
        assert engine is not None
        assert engine.channel.owner_id == ""


class TestTeamAdminCommands:
    @patch("src.slock_engine.engine.create_engine_session")
    def test_list_teams_lists_all_activated_slock_teams_from_admin_chat(self, mock_session, tmp_path):
        """Admin `/team list` should show every active team, not only the current chat."""
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        manager = ctx.slock_engine_manager
        root_path = str(tmp_path / "project_root")
        os.makedirs(root_path)

        for chat_id, team_name in [("oc_alpha", "Alpha"), ("oc_beta", "Beta")]:
            engine = manager.get_or_create(chat_id, root_path, engine_name="Slock")
            engine.activate_channel(SlockChannel(channel_id=chat_id, name=team_name, team_name=team_name))
            manager.register_managed_chat(chat_id)

        handler.list_teams("msg_admin", "oc_admin")

        handler.reply_card.assert_called_once()
        card = json.loads(handler.reply_card.call_args[0][1])
        card_blob = json.dumps(card, ensure_ascii=False)
        assert "Alpha" in card_blob
        assert "Beta" in card_blob
        assert "oc_alpha" in card_blob
        assert "oc_beta" in card_blob
        assert "进入 Slock 群" in card_blob
        assert "openChatId=oc_alpha" in card_blob
        assert "openChatId=oc_beta" in card_blob

    @patch("src.slock_engine.engine.create_engine_session")
    def test_team_status_finds_team_by_name(self, mock_session, tmp_path):
        """Admin `/team status <name>` resolves a team by name and renders its status card."""
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        root_path = str(tmp_path / "project_root")
        os.makedirs(root_path)
        engine = ctx.slock_engine_manager.get_or_create_activated(
            "oc_alpha",
            root_path,
            SlockChannel(channel_id="oc_alpha", name="Alpha Chat", team_name="Alpha"),
            engine_name="Slock",
        )
        engine.get_status_card = MagicMock(return_value={"header": {"title": {"content": "Alpha"}}})

        handler.show_team_status("msg_status", "oc_admin", "Alpha")

        handler.reply_card.assert_called_once()
        card = json.loads(handler.reply_card.call_args[0][1])
        assert card["header"]["title"]["content"] == "Alpha"

    @patch("src.slock_engine.engine.create_engine_session")
    def test_team_dissolve_unregisters_and_removes_named_team(self, mock_session, tmp_path):
        """Admin `/team dissolve <name>` stops the named team runtime."""
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        manager = ctx.slock_engine_manager
        root_path = str(tmp_path / "project_root")
        engine = manager.get_or_create("oc_beta", root_path, engine_name="Slock")
        engine.activate_channel(SlockChannel(channel_id="oc_beta", name="Beta Chat", team_name="Beta", owner_id="ou_admin"))
        manager.register_managed_chat("oc_beta")

        with patch("src.thread.manager.get_current_sender_id", return_value="ou_admin"), \
             patch("src.config.get_settings", return_value=MagicMock(admin_user_ids=frozenset({"ou_admin"}))):
            handler.dissolve_team("msg_dissolve", "oc_admin", "Beta")

        assert manager.is_managed_chat("oc_beta") is False
        assert manager.get_activated_engine("oc_beta") is None
        handler.reply_text.assert_called_once()
        assert "Beta" in handler.reply_text.call_args[0][1]


class TestActivateSlockManagedChat:
    @patch("src.slock_engine.activation_guard.get_activation_guard")
    @patch("src.slock_engine.engine.create_engine_session")
    def test_activate_slock_registers_current_chat_for_team_commands(self, mock_session, mock_guard, tmp_path):
        """After `/slock`, team-local commands such as `/new-role` must be captured."""
        mock_guard.return_value.can_auto_activate.return_value = (True, "allowed")
        from src.slock_engine.slash_commands import is_slock_command

        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)
        project_root = str(tmp_path / "project_root")
        handler._ensure_project = MagicMock(return_value=MagicMock(
            root_path=project_root,
            project_name="CurrentTeam",
            project_id="p-current",
        ))
        handler.get_engine_name = MagicMock(return_value="Slock-current")
        session = MagicMock()
        handler.create_static_card_session.return_value = session

        handler.activate_slock("msg_current", "oc_current", "", project=None)

        manager = ctx.slock_engine_manager
        assert manager.is_managed_chat("oc_current") is True
        assert is_slock_command("/new-role Coder", chat_id="oc_current", manager=manager)

    def test_slock_help_mentions_templates_fork_and_auto_routing(self, tmp_path):
        """Slock's own help should use the command panel card."""
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)

        handler.show_slock_help("msg_help")

        handler.reply_card.assert_called_once()
        card_json = handler.reply_card.call_args[0][1]
        # The card should contain key command references
        assert "/role" in card_json
        assert "/task" in card_json
        assert "/team" in card_json


# ==================================================================
# Test: Restart survival — create team, simulate restart, verify restore
# ==================================================================


class TestRestartSurvival:
    @patch("src.slock_engine.engine.create_engine_session")
    @patch("src.thread.manager.get_current_sender_id", return_value="ou_sender123")
    @patch("src.project_chat.lark_chat_client.LarkChatClient")
    def test_new_team_survives_restart(
        self, MockLarkChatClient, mock_sender, mock_session, tmp_path
    ):
        """After create_team, a fresh manager.restore_from_disk should recover state."""
        ctx = _make_handler_ctx(tmp_path)
        handler = _make_slock_handler(ctx)

        root_path = str(tmp_path / "project_root")
        os.makedirs(root_path)
        handler.get_working_dir.return_value = root_path

        mock_lark = MockLarkChatClient.return_value
        mock_lark.create_chat.return_value = _FakeCreateChatResult(
            chat_id="oc_survive", name="SurviveTeam"
        )

        # Step 1: Create team (writes marker to disk)
        handler.create_team("msg1", "oc_origin", "SurviveTeam", project=None)

        # Verify original manager has the state
        original_manager = ctx.slock_engine_manager
        assert original_manager.is_managed_chat("oc_survive") is True
        assert original_manager.get_activated_engine("oc_survive") is not None

        # Step 2: Simulate restart — create a brand new manager
        new_manager = SlockEngineManager(storage_base_path=ctx.settings.slock_memory_base_path)
        assert new_manager.is_managed_chat("oc_survive") is False

        # Step 3: Restore from disk
        restored = new_manager.restore_from_disk(root_path)

        assert restored == 1
        assert new_manager.is_managed_chat("oc_survive") is True
        engine = new_manager.get_activated_engine("oc_survive")
        assert engine is not None
        assert engine.channel is not None
        assert engine.channel.channel_id == "oc_survive"
        assert engine.channel.team_name == "SurviveTeam"


class TestOwnerIdPersistence:
    """AC-14: owner_id must be persisted in marker_data and restored from disk."""

    @patch("src.slock_engine.engine.create_engine_session")
    def test_marker_data_contains_owner_id(self, mock_session, tmp_path):
        """activate_channel writes owner_id to .slock_channel.json marker."""
        manager = SlockEngineManager(storage_base_path=str(tmp_path / "slock_mem"))
        root_path = str(tmp_path / "project_root")
        engine = manager.get_or_create("oc_owner_test", root_path, engine_name="Slock")

        channel = SlockChannel(
            channel_id="oc_owner_test",
            name="Owner Team",
            team_name="OwnerTeam",
            owner_id="ou_creator_abc",
        )
        engine.activate_channel(channel)

        # Read marker from disk
        marker_path = os.path.join(
            engine.memory.get_group_base_path("oc_owner_test"),
            ".slock_channel.json",
        )
        with open(marker_path, "r") as f:
            marker = json.load(f)

        assert "owner_id" in marker
        assert marker["owner_id"] == "ou_creator_abc"

    @patch("src.slock_engine.engine.create_engine_session")
    def test_restore_from_disk_recovers_owner_id(self, mock_session, tmp_path):
        """After restart, restore_from_disk recovers owner_id from marker."""
        storage_path = str(tmp_path / "slock_mem")
        root_path = str(tmp_path / "project_root")
        os.makedirs(root_path)

        # Step 1: Create and activate with owner_id
        manager1 = SlockEngineManager(storage_base_path=storage_path)
        engine1 = manager1.get_or_create("oc_restore_owner", root_path, engine_name="Slock")
        channel = SlockChannel(
            channel_id="oc_restore_owner",
            name="Restore Team",
            team_name="RestoreTeam",
            owner_id="ou_original_owner",
        )
        engine1.activate_channel(channel)
        manager1.register_managed_chat("oc_restore_owner")

        # Step 2: Simulate restart — fresh manager
        manager2 = SlockEngineManager(storage_base_path=storage_path)
        restored = manager2.restore_from_disk(root_path)

        assert restored == 1
        engine2 = manager2.get_activated_engine("oc_restore_owner")
        assert engine2 is not None
        assert engine2.channel is not None
        assert engine2.channel.owner_id == "ou_original_owner"


class TestMarkerMerge:
    """AC-14/AC-15: _write_channel_marker merge fills missing fields without overwriting."""

    @patch("src.slock_engine.engine.create_engine_session")
    def test_merge_fills_missing_owner_id(self, mock_session, tmp_path):
        """Old marker without owner_id gets owner_id filled after activate_channel."""
        storage_path = str(tmp_path / "slock_mem")
        root_path = str(tmp_path / "project_root")

        manager = SlockEngineManager(storage_base_path=storage_path)
        engine = manager.get_or_create("oc_merge1", root_path, engine_name="Slock")

        # Pre-write an old marker WITHOUT owner_id
        group_dir = engine.memory.get_group_base_path("oc_merge1")
        os.makedirs(group_dir, exist_ok=True)
        marker_path = os.path.join(group_dir, ".slock_channel.json")
        old_marker = {
            "channel_id": "oc_merge1",
            "team_name": "OldTeam",
            "name": "Old",
            "activated_at": "2025-01-01T00:00:00Z",
        }
        with open(marker_path, "w") as f:
            json.dump(old_marker, f)

        # Now activate with owner_id set
        channel = SlockChannel(
            channel_id="oc_merge1",
            name="Old",
            team_name="OldTeam",
            owner_id="ou_new_owner",
        )
        engine.activate_channel(channel)

        # Verify owner_id was merged in
        with open(marker_path, "r") as f:
            merged = json.load(f)
        assert merged["owner_id"] == "ou_new_owner"
        # activated_at should be preserved (not overwritten)
        assert merged["activated_at"] == "2025-01-01T00:00:00Z"

    @patch("src.slock_engine.engine.create_engine_session")
    def test_merge_does_not_overwrite_with_empty(self, mock_session, tmp_path):
        """Existing owner_id is NOT overwritten when new value is empty."""
        storage_path = str(tmp_path / "slock_mem")
        root_path = str(tmp_path / "project_root")

        manager = SlockEngineManager(storage_base_path=storage_path)
        engine = manager.get_or_create("oc_merge2", root_path, engine_name="Slock")

        # Pre-write marker WITH owner_id
        group_dir = engine.memory.get_group_base_path("oc_merge2")
        os.makedirs(group_dir, exist_ok=True)
        marker_path = os.path.join(group_dir, ".slock_channel.json")
        old_marker = {
            "channel_id": "oc_merge2",
            "team_name": "Team2",
            "name": "Team2",
            "owner_id": "ou_existing_owner",
            "activated_at": "2025-01-01T00:00:00Z",
        }
        with open(marker_path, "w") as f:
            json.dump(old_marker, f)

        # Activate with empty owner_id — should NOT overwrite
        channel = SlockChannel(
            channel_id="oc_merge2",
            name="Team2",
            team_name="Team2",
            owner_id="",
        )
        engine.activate_channel(channel)

        with open(marker_path, "r") as f:
            merged = json.load(f)
        assert merged["owner_id"] == "ou_existing_owner"

    @patch("src.slock_engine.engine.create_engine_session")
    def test_restore_recovers_merged_owner_id(self, mock_session, tmp_path):
        """After merge fills owner_id, restore_from_disk correctly recovers it."""
        storage_path = str(tmp_path / "slock_mem")
        root_path = str(tmp_path / "project_root")
        os.makedirs(root_path)

        manager1 = SlockEngineManager(storage_base_path=storage_path)
        engine1 = manager1.get_or_create("oc_merge3", root_path, engine_name="Slock")

        # Pre-write old marker without owner_id
        group_dir = engine1.memory.get_group_base_path("oc_merge3")
        os.makedirs(group_dir, exist_ok=True)
        marker_path = os.path.join(group_dir, ".slock_channel.json")
        old_marker = {
            "channel_id": "oc_merge3",
            "team_name": "MergeTeam",
            "name": "MergeTeam",
            "activated_at": "2025-06-01T00:00:00Z",
        }
        with open(marker_path, "w") as f:
            json.dump(old_marker, f)

        # Activate to trigger merge — fills owner_id
        channel = SlockChannel(
            channel_id="oc_merge3",
            name="MergeTeam",
            team_name="MergeTeam",
            owner_id="ou_merged_owner",
        )
        engine1.activate_channel(channel)
        manager1.register_managed_chat("oc_merge3")

        # Simulate restart
        manager2 = SlockEngineManager(storage_base_path=storage_path)
        restored = manager2.restore_from_disk(root_path)

        assert restored == 1
        engine2 = manager2.get_activated_engine("oc_merge3")
        assert engine2 is not None
        assert engine2.channel is not None
        assert engine2.channel.owner_id == "ou_merged_owner"
