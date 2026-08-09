import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.access_control import (
    IngressAccessPolicy,
    IngressAccessPolicyProvider,
)
from src.config import IngressAccessMode
from src.feishu.slash_command_parser import SlashCommandParser
from src.feishu.ws_client import FeishuWSClient


def _make_ws_client(**extra_settings):
    """Shared helper: instantiate FeishuWSClient with mocked dependencies."""
    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client.ACPSessionManager"),
        patch("src.feishu.ws_client.IntentRecognizer"),
        patch("src.feishu.ws_client.ProjectManager"),
        patch("src.feishu.ws_client.MessageProjectMapper"),
        patch("src.feishu.ws_client.DeepEngineManager"),
        patch("src.feishu.ws_client.ProgressReporter"),
        patch("src.mode.ModeManager"),
    ):
        mock_settings = MagicMock()
        mock_settings.app_id = "test_app_id"
        mock_settings.app_secret = "test_app_secret"
        mock_settings.streaming_enabled = False
        mock_settings.task_scheduler_max_concurrent = 2
        mock_settings.task_scheduler_per_key_concurrency = 1
        mock_settings.message_cache_ttl = 300
        mock_settings.message_cache_max_size = 1000
        mock_settings.card.action_dedup_ttl = 1
        mock_settings.card.action_dedup_max_size = 5000
        mock_settings.system_command_concurrency = 10
        mock_settings.spec_rate_limit_capacity = 100
        mock_settings.spec_rate_limit_fill_rate = 50.0
        mock_settings.spec_circuit_breaker_threshold = 10
        mock_settings.spec_circuit_breaker_recovery = 5.0
        mock_settings.message_expire_seconds = 30
        for k, v in extra_settings.items():
            setattr(mock_settings, k, v)
        mock_get_settings.return_value = mock_settings
        return FeishuWSClient(MagicMock())


def _enrol_test_ingress(
    client: FeishuWSClient,
    *,
    sender_id: str = "ou_test",
    chat_id: str = "oc_1",
) -> None:
    client._ingress_access_policy_provider = IngressAccessPolicyProvider(
        IngressAccessPolicy(
            admin_ids=frozenset(),
            allowed_user_ids=frozenset({sender_id}),
            allowed_chat_ids=frozenset({chat_id}),
            mode=IngressAccessMode.ENFORCED,
            admin_bootstrap_scope="p2p_only",
        )
    )




class TestSystemCmdGateReadonlyBypass(unittest.TestCase):

    def _make_client_with_gate_active(self, chat_id="oc_1"):
        client = _make_ws_client()
        client._scheduler = MagicMock()
        client._reply_text = MagicMock()
        with client._system_cmd_gate_lock:
            client._system_cmd_inflight_by_chat[chat_id] = 1
        return client

    def _make_card_data(self, action_type, chat_id="oc_1", message_id="om_1"):
        return SimpleNamespace(
            header=SimpleNamespace(event_id="evt_1", event_type="card.action.trigger"),
            event=SimpleNamespace(
                action=SimpleNamespace(
                    value={"action": action_type, "project_id": "p1"},
                    tag="button",
                    name="btn",
                ),
                operator=SimpleNamespace(open_id="ou_x", user_id="u_x"),
                context=SimpleNamespace(open_message_id=message_id, open_chat_id=chat_id),
            ),
        )

    def test_readonly_action_bypasses_gate(self):
        from src.feishu.ws_client import _READONLY_CARD_ACTIONS

        client = self._make_client_with_gate_active()
        for action in ["deep_expand", "spec_mode_full", "deep_expand_ac"]:
            self.assertIn(action, _READONLY_CARD_ACTIONS)
            client._card_event_cache = MagicMock()
            client._card_event_cache.is_duplicate.return_value = False
            client._card_action_dedup_cache = MagicMock()
            client._card_action_dedup_cache.is_duplicate.return_value = True
            data = self._make_card_data(action)
            result = client._handle_card_action(data)
            # Dedup now returns toast instead of None
            self.assertEqual(result, {"toast": {"type": "info", "content": "操作已受理，请勿重复点击"}})
            client._reply_text.assert_not_called()

    def test_non_readonly_action_blocked_by_gate(self):
        client = self._make_client_with_gate_active()
        client._card_event_cache = MagicMock()
        client._card_event_cache.is_duplicate.return_value = False
        client._card_action_dedup_cache = MagicMock()
        client._card_action_dedup_cache.is_duplicate.return_value = True
        data = self._make_card_data("enter_coco")
        result = client._handle_card_action(data)
        self.assertIsNone(result)
        client._reply_text.assert_called_once()
        args = client._reply_text.call_args.args
        self.assertIn("系统指令处理中", args[1])
        client._scheduler.submit.assert_not_called()

    def test_gate_not_active_allows_all_actions(self):
        with (
            patch("src.feishu.ws_client.get_settings") as mock_get_settings,
            patch("src.feishu.ws_client.ACPSessionManager"),
            patch("src.feishu.ws_client.IntentRecognizer"),
            patch("src.feishu.ws_client.ProjectManager"),
            patch("src.feishu.ws_client.MessageProjectMapper"),
            patch("src.feishu.ws_client.DeepEngineManager"),
            patch("src.feishu.ws_client.ProgressReporter"),
            patch("src.mode.ModeManager"),
        ):
            mock_settings = MagicMock()
            mock_settings.app_id = "test_app_id"
            mock_settings.app_secret = "test_app_secret"
            mock_settings.streaming_enabled = False
            mock_settings.task_scheduler_max_concurrent = 2
            mock_settings.task_scheduler_per_key_concurrency = 1
            mock_settings.message_cache_ttl = 300
            mock_settings.message_cache_max_size = 1000
            mock_settings.card.action_dedup_ttl = 1
            mock_settings.card.action_dedup_max_size = 5000
            mock_settings.system_command_concurrency = 10
            mock_settings.spec_rate_limit_capacity = 100
            mock_settings.spec_rate_limit_fill_rate = 50.0
            mock_settings.spec_circuit_breaker_threshold = 10
            mock_settings.spec_circuit_breaker_recovery = 5.0
            mock_settings.message_expire_seconds = 30
            mock_get_settings.return_value = mock_settings

            client = FeishuWSClient(MagicMock())
            client._scheduler = MagicMock()
            client._reply_text = MagicMock()
            client._card_event_cache = MagicMock()
            client._card_event_cache.is_duplicate.return_value = False
            client._card_action_dedup_cache = MagicMock()
            client._card_action_dedup_cache.is_duplicate.return_value = True

            data = self._make_card_data("enter_coco")
            result = client._handle_card_action(data)
            # Dedup now returns toast instead of None
            self.assertEqual(result, {"toast": {"type": "info", "content": "操作已受理，请勿重复点击"}})
            client._reply_text.assert_not_called()


class TestIsSystemCardActionEngineControls(unittest.TestCase):

    def _make_client(self):
        return _make_ws_client()

    def _make_data(self, action_type):
        return SimpleNamespace(
            event=SimpleNamespace(
                action=SimpleNamespace(value={"action": action_type}),
            )
        )

    def test_engine_control_actions_are_system(self):
        client = self._make_client()
        engine_controls = [
            "deep_pause", "deep_stop", "deep_resume",
            "spec_pause", "spec_stop", "spec_resume",
        ]
        for action in engine_controls:
            self.assertTrue(
                client._is_system_card_action(self._make_data(action)),
                f"{action} should be a system card action",
            )

    def test_existing_system_actions_still_recognized(self):
        client = self._make_client()
        for action in ["show_status", "switch_project", "show_board", "enter_deep_prompt"]:
            self.assertTrue(
                client._is_system_card_action(self._make_data(action)),
                f"{action} should still be a system card action",
            )

    def test_non_system_actions_not_flagged(self):
        client = self._make_client()
        for action in ["enter_coco", "enter_claude", "deep_expand", "unknown_action"]:
            self.assertFalse(
                client._is_system_card_action(self._make_data(action)),
                f"{action} should NOT be a system card action",
            )


class TestOneShotDispatchToThread(unittest.TestCase):

    def _make_client(self):
        return _make_ws_client(thread_programming_enabled=True)

    def test_find_active_thread_returns_programming_context(self):
        client = self._make_client()
        client._thread_manager = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.mode = "coco"
        client._thread_manager.get_by_chat.return_value = [mock_ctx]

        result = client._find_active_thread("c1")
        self.assertEqual(result, mock_ctx)

    def test_find_active_thread_skips_smart_mode(self):
        client = self._make_client()
        client._thread_manager = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.mode = "smart"
        client._thread_manager.get_by_chat.return_value = [mock_ctx]

        result = client._find_active_thread("c1")
        self.assertIsNone(result)

    def test_find_active_thread_returns_none_when_empty(self):
        client = self._make_client()
        client._thread_manager = MagicMock()
        client._thread_manager.get_by_chat.return_value = []

        result = client._find_active_thread("c1")
        self.assertIsNone(result)

    def test_find_active_thread_disabled_returns_none(self):
        client = self._make_client()
        client._thread_manager = MagicMock()
        client.settings.thread_programming_enabled = False

        result = client._find_active_thread("c1")
        self.assertIsNone(result)
        client._thread_manager.get_by_chat.assert_not_called()

class TestActiveThreadGuidance(unittest.TestCase):

    def _make_client(self):
        return _make_ws_client(thread_programming_enabled=True)

    @patch("src.feishu.ws_client.get_current_thread_id", return_value=None)
    def test_guidance_shown_when_intent_unrecognized_with_active_thread(self, _):
        client = self._make_client()
        client._thread_manager = MagicMock()
        client._reply_text = MagicMock()
        client._add_reaction = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.mode = "coco"
        client._thread_manager.get_by_chat.return_value = [mock_ctx]

        client._execute_single_task("m1", "c1", None, "帮我写个函数", MagicMock())

        client._reply_text.assert_called_once()
        call_args = str(client._reply_text.call_args)
        assert "活跃" in call_args
        assert "话题" in call_args

    @patch("src.feishu.ws_client.get_current_thread_id", return_value=None)
    def test_no_guidance_when_no_active_thread(self, _):
        client = self._make_client()
        client._thread_manager = MagicMock()
        client._reply_text = MagicMock()
        client._thread_manager.get_by_chat.return_value = []

        client._execute_single_task("m1", "c1", None, "帮我写个函数", MagicMock())

        client._reply_text.assert_called_once()
        call_args = str(client._reply_text.call_args)
        assert "无法理解" in call_args

    @patch("src.feishu.ws_client.get_current_thread_id", return_value=None)
    def test_no_guidance_when_thread_disabled(self, _):
        client = self._make_client()
        client._thread_manager = MagicMock()
        client._reply_text = MagicMock()
        client.settings.thread_programming_enabled = False

        client._execute_single_task("m1", "c1", None, "帮我写个函数", MagicMock())

        client._reply_text.assert_called_once()
        call_args = str(client._reply_text.call_args)
        assert "无法理解" in call_args


class TestThreadPersistentProgramming(unittest.TestCase):

    def _make_client(self):
        return _make_ws_client(thread_programming_enabled=True)

    def test_dispatch_message_logic_skips_enter_mode_for_thread(self):
        client = self._make_client()
        client._add_reaction = MagicMock()
        client._process_with_intent = MagicMock()
        handler = MagicMock()
        handler.handle_message = MagicMock()
        client._get_mode_handler = MagicMock(return_value=handler)

        project = MagicMock()
        project.project_id = "p1"

        client._dispatch_message_logic(
            "m2",
            "c1",
            "继续写",
            project,
            auto_enter_mode="coco",
            command_match=SlashCommandParser.parse("继续写"),
        )

        handler.handle_message.assert_called_once_with("m2", "c1", "继续写", project)
        client._process_with_intent.assert_not_called()

    def test_dispatch_message_logic_no_enter_mode_called(self):
        client = self._make_client()
        client._add_reaction = MagicMock()
        client._enter_coco_mode = MagicMock()
        handler = MagicMock()
        client._get_mode_handler = MagicMock(return_value=handler)

        project = MagicMock()
        client._dispatch_message_logic(
            "m2",
            "c1",
            "改一下",
            project,
            auto_enter_mode="coco",
            command_match=SlashCommandParser.parse("改一下"),
        )

        client._enter_coco_mode.assert_not_called()

    def test_dispatch_message_logic_thread_exit_command(self):
        client = self._make_client()
        client._add_reaction = MagicMock()
        client._exit_current_mode = MagicMock()

        project = MagicMock()
        client._dispatch_message_logic(
            "m2",
            "c1",
            "/exit",
            project,
            auto_enter_mode="coco",
            command_match=SlashCommandParser.parse("/exit"),
        )

        client._exit_current_mode.assert_called_once()

    def test_same_mode_topic_entries_use_the_catalog_identity(self):
        """Live same-mode replies must not fall through generic entry routing."""
        cases = (
            ("coco", "/coco"),
            ("coco", "/enter_coco"),
            ("claude", "/claude"),
            ("claude", "/enter_claude"),
            ("aiden", "/aiden"),
            ("aiden", "/enter_aiden"),
            ("codex", "/codex"),
            ("codex", "/enter_codex"),
            ("gemini", "/gemini"),
            ("gemini", "/enter_gemini"),
            ("traex", "/traex"),
            ("traex", "/enter_traex"),
        )
        for mode, command in cases:
            with self.subTest(mode=mode, command=command):
                client = self._make_client()
                client._reply_text = MagicMock()
                client._process_with_intent = MagicMock()
                client._is_interceptable_command_match = MagicMock(return_value=False)
                client._is_programming_entry_command = MagicMock(return_value=True)
                client._reply_if_topic_engine_switch_blocked = MagicMock(return_value=False)

                client._dispatch_message_logic(
                    "m_same_mode",
                    "c1",
                    command,
                    MagicMock(),
                    auto_enter_mode=mode,
                    command_match=SlashCommandParser.parse(command),
                )

                client._reply_text.assert_called_once()
                client._is_programming_entry_command.assert_not_called()
                client._process_with_intent.assert_not_called()

    def test_dispatch_message_logic_all_modes_skip_enter(self):
        client = self._make_client()
        client._add_reaction = MagicMock()

        for mode in ("coco", "claude", "aiden", "codex", "gemini", "traex"):
            handler = MagicMock()
            client._get_mode_handler = MagicMock(return_value=handler)
            project = MagicMock()

            client._dispatch_message_logic(
                f"m_{mode}",
                "c1",
                "do stuff",
                project,
                auto_enter_mode=mode,
                command_match=SlashCommandParser.parse("do stuff"),
            )

            handler.handle_message.assert_called_once_with(f"m_{mode}", "c1", "do stuff", project)

    def test_dispatch_message_logic_no_auto_enter_goes_to_intent(self):
        client = self._make_client()
        client._add_reaction = MagicMock()
        client._process_with_intent = MagicMock()
        client._get_mode_handler = MagicMock()

        project = MagicMock()
        client._dispatch_message_logic(
            "m1",
            "c1",
            "你好",
            project,
            auto_enter_mode=None,
            command_match=SlashCommandParser.parse("你好"),
        )

        client._process_with_intent.assert_called_once()
        client._get_mode_handler.assert_not_called()


class TestThreadModeRetentionRobust(unittest.TestCase):
    """话题内编程模式保持的鲁棒性测试 — 覆盖第三轮修复的多层防御"""

    def _make_client(self):
        return _make_ws_client(thread_programming_enabled=True)

    def test_resolve_context_returns_mode_even_if_project_none(self):
        """_resolve_message_context 应在 thread_ctx 存在但 project 查找失败时仍返回 auto_enter_mode"""
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = True

        from src.thread.models import ThreadContext
        thread_ctx = ThreadContext(
            thread_root_id="root1", chat_id="c1", project_id="proj1", mode="coco",
        )
        client._thread_manager = MagicMock()
        client._thread_manager.get.return_value = thread_ctx

        client._project_manager = MagicMock()
        client._project_manager.get_project.return_value = None
        client._project_manager.get_project_for_chat.return_value = None
        fallback_project = MagicMock()
        fallback_project.project_id = "proj_fallback"
        client._project_manager.get_active_project.return_value = fallback_project

        message = MagicMock()
        message.message_id = "m1"
        message.chat_id = "c1"
        message.root_id = "root1"
        message.parent_id = None

        project, auto_mode = client._resolve_message_context(message)

        self.assertEqual(auto_mode, "coco")
        self.assertEqual(project, fallback_project)

    def test_engine_topic_does_not_fallback_to_unrelated_active_project(self):
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = True

        from src.thread.models import ThreadContext
        thread_ctx = ThreadContext(
            thread_root_id="root1", chat_id="c1", project_id="missing", mode="deep",
        )
        client._thread_manager = MagicMock()
        client._thread_manager.get.return_value = thread_ctx
        client._project_manager = MagicMock()
        client._project_manager.get_project_for_chat.return_value = None

        message = MagicMock(message_id="m1", chat_id="c1", root_id="root1", parent_id=None)

        project, auto_mode = client._resolve_message_context(message)

        self.assertEqual(auto_mode, "deep")
        self.assertIsNone(project)
        client._project_manager.get_active_project.assert_not_called()

    def test_resolve_context_smart_mode_returns_none_auto_enter(self):
        """thread_ctx.mode='smart' 时 auto_enter_mode 应为 None，但不再 fall through"""
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = True

        from src.thread.models import ThreadContext
        thread_ctx = ThreadContext(
            thread_root_id="root1", chat_id="c1", project_id="proj1", mode="smart",
        )
        client._thread_manager = MagicMock()
        client._thread_manager.get.return_value = thread_ctx

        project_obj = MagicMock()
        project_obj.project_id = "proj1"
        client._project_manager = MagicMock()
        client._project_manager.get_project.return_value = project_obj
        client._project_manager.get_project_for_chat.return_value = project_obj
        client._resolve_project_from_message = MagicMock(return_value=(MagicMock(), None))

        message = MagicMock()
        message.message_id = "m1"
        message.chat_id = "c1"
        message.root_id = "root1"
        message.parent_id = None

        project, auto_mode = client._resolve_message_context(message)

        self.assertIsNone(auto_mode)
        self.assertEqual(project, project_obj)
        client._resolve_project_from_message.assert_not_called()

    def test_safety_net_in_process_message_async(self):
        """安全网：即使 _resolve_message_context 返回 auto_enter_mode=None，安全网仍能从 thread_ctx 恢复"""
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = True
        client.settings.allowed_chat_ids = set()
        client.settings.allowed_user_ids = set()
        _enrol_test_ingress(client)

        from src.thread.models import ThreadContext
        thread_ctx = ThreadContext(
            thread_root_id="root1", chat_id="c1", project_id="proj1", mode="coco",
        )
        client._thread_manager = MagicMock()
        client._thread_manager.get.return_value = thread_ctx

        project = MagicMock()
        project.project_id = "proj1"
        client._project_manager = MagicMock()
        client._project_manager.get_project.return_value = project
        client._project_manager.get_project_for_chat.return_value = project
        client._project_manager.get_active_project.return_value = project

        client._validate_message = MagicMock(return_value=True)
        client._get_image_handler = MagicMock()
        parse_result = MagicMock()
        parse_result.text = "继续写"
        parse_result.image_keys = []
        client._get_image_handler.return_value.parse_message.return_value = parse_result
        client._clean_at_text = MagicMock(return_value="继续写")

        client._resolve_message_context = MagicMock(return_value=(project, None))
        client._dispatch_message_logic = MagicMock()
        client._update_task_project = MagicMock()

        data = MagicMock()
        data.event.message.message_id = "om_2"
        data.event.message.chat_id = "oc_1"
        data.event.message.chat_type = "group"
        data.event.message.root_id = "root1"
        data.event.message.create_time = None
        data.event.sender.sender_id.open_id = "ou_test"

        task_ctx = MagicMock()
        task_ctx.run_id = "run1"
        task_ctx.spec.sender_id = "ou_test"
        task_ctx.spec.is_p2p = False

        client._process_message_async(data, task_ctx=task_ctx)

        call_args = client._dispatch_message_logic.call_args
        actual_auto_mode = call_args[0][4] if len(call_args[0]) > 4 else call_args[1].get("auto_enter_mode")
        self.assertEqual(actual_auto_mode, "coco")

    def test_engine_safety_net_discards_unrelated_initial_project(self):
        client = self._make_client()
        client.settings = MagicMock(
            thread_programming_enabled=True,
            allowed_chat_ids=set(),
            allowed_user_ids=set(),
        )
        _enrol_test_ingress(client)
        from src.thread.models import ThreadContext
        client._thread_manager = MagicMock()
        client._thread_manager.get.return_value = ThreadContext(
            thread_root_id="root1",
            chat_id="c1",
            project_id="missing",
            mode="deep",
        )
        unrelated = MagicMock(project_id="unrelated")
        client._project_manager = MagicMock()
        client._project_manager.get_project_for_chat.return_value = None
        client._project_manager.get_active_project.return_value = unrelated
        client._validate_message = MagicMock(return_value=True)
        parse_result = MagicMock(text="继续执行", image_keys=[])
        client._get_image_handler = MagicMock()
        client._get_image_handler.return_value.parse_message.return_value = parse_result
        client._clean_at_text = MagicMock(return_value="继续执行")
        client._resolve_message_context = MagicMock(return_value=(unrelated, None))
        client._dispatch_message_logic = MagicMock()

        data = MagicMock()
        data.event.message.message_id = "om_2"
        data.event.message.chat_id = "oc_1"
        data.event.message.chat_type = "group"
        data.event.message.root_id = "root1"
        data.event.message.create_time = None
        data.event.sender.sender_id.open_id = "ou_test"
        client._process_message_async(data)

        args = client._dispatch_message_logic.call_args.args
        self.assertIsNone(args[3])
        self.assertEqual(args[4], "deep")
        client._project_manager.get_active_project.assert_not_called()

    def test_all_modes_resolve_from_thread_ctx(self):
        """所有编程模式都能从 thread_ctx 正确解析。"""
        for mode in ("coco", "claude", "aiden", "codex", "gemini", "traex"):
            client = self._make_client()
            client.settings = MagicMock()
            client.settings.thread_programming_enabled = True

            from src.thread.models import ThreadContext
            thread_ctx = ThreadContext(
                thread_root_id="root1", chat_id="c1", project_id="proj1", mode=mode,
            )
            client._thread_manager = MagicMock()
            client._thread_manager.get.return_value = thread_ctx

            project = MagicMock()
            project.project_id = "proj1"
            client._project_manager = MagicMock()
            client._project_manager.get_project.return_value = project
            client._project_manager.get_project_for_chat.return_value = project

            message = MagicMock()
            message.message_id = "m1"
            message.chat_id = "c1"
            message.root_id = "root1"
            message.parent_id = None

            resolved_project, auto_mode = client._resolve_message_context(message)
            self.assertEqual(auto_mode, mode, f"Mode {mode} should be returned from thread context")

    def test_resolve_context_disabled_thread_skips_thread_ctx(self):
        """thread_programming_enabled=False 时跳过 thread_ctx 查找"""
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = False

        client._thread_manager = MagicMock()
        fallback_project = MagicMock()
        client._resolve_project_from_message = MagicMock(return_value=(fallback_project, None))

        message = MagicMock()
        message.message_id = "m1"
        message.chat_id = "c1"
        message.root_id = "root1"
        message.parent_id = None

        project, auto_mode = client._resolve_message_context(message)

        client._thread_manager.get.assert_not_called()
        self.assertIsNone(auto_mode)
        client._resolve_project_from_message.assert_called_once()

    def test_safety_net_covers_image_path(self):
        """安全网在图片消息路径（_handle_image_content 返回 None mode）时也能恢复 auto_enter_mode"""
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = True
        client.settings.allowed_chat_ids = set()
        client.settings.allowed_user_ids = set()
        _enrol_test_ingress(client)

        from src.thread.models import ThreadContext
        thread_ctx = ThreadContext(
            thread_root_id="root1", chat_id="c1", project_id="proj1", mode="coco",
        )
        client._thread_manager = MagicMock()
        client._thread_manager.get.return_value = thread_ctx

        project = MagicMock()
        project.project_id = "proj1"
        client._project_manager = MagicMock()
        client._project_manager.get_project.return_value = project
        client._project_manager.get_project_for_chat.return_value = project
        client._project_manager.get_active_project.return_value = project

        client._validate_message = MagicMock(return_value=True)
        client._get_image_handler = MagicMock()
        parse_result = MagicMock()
        parse_result.text = "看这张图"
        parse_result.image_keys = ["img_key_1"]
        client._get_image_handler.return_value.parse_message.return_value = parse_result
        client._clean_at_text = MagicMock(return_value="看这张图")

        client._handle_image_content = MagicMock(return_value=(project, None, "看这张图", True))
        client._dispatch_message_logic = MagicMock()
        client._update_task_project = MagicMock()

        data = MagicMock()
        data.event.message.message_id = "om_2"
        data.event.message.chat_id = "oc_1"
        data.event.message.chat_type = "group"
        data.event.message.root_id = "root1"
        data.event.message.create_time = None
        data.event.sender.sender_id.open_id = "ou_test"

        task_ctx = MagicMock()
        task_ctx.run_id = "run1"
        task_ctx.spec.sender_id = "ou_test"
        task_ctx.spec.is_p2p = False

        client._process_message_async(data, task_ctx=task_ctx)

        call_args = client._dispatch_message_logic.call_args
        actual_auto_mode = call_args[0][4] if len(call_args[0]) > 4 else call_args[1].get("auto_enter_mode")
        self.assertEqual(actual_auto_mode, "coco")

    def test_dispatch_message_logic_handler_none_fallback(self):
        client = self._make_client()
        client._add_reaction = MagicMock()
        client._process_with_intent = MagicMock()
        client._get_mode_handler = MagicMock(return_value=None)

        project = MagicMock()
        client._dispatch_message_logic(
            "m1",
            "c1",
            "改一下",
            project,
            auto_enter_mode="coco",
            command_match=SlashCommandParser.parse("改一下"),
        )

        client._process_with_intent.assert_called_once()

    def test_dispatch_message_logic_programming_entry_intercepted(self):
        client = self._make_client()
        client._add_reaction = MagicMock()
        client._reply_text = MagicMock()
        client._process_with_intent = MagicMock()
        client._is_interceptable_command_match = MagicMock(return_value=True)
        client._get_mode_handler = MagicMock()

        project = MagicMock()
        client._dispatch_message_logic(
            "m1",
            "c1",
            "/coco",
            project,
            auto_enter_mode="coco",
            command_match=SlashCommandParser.parse("/coco"),
        )

        client._process_with_intent.assert_called_once_with(
            "m1",
            "c1",
            "/coco",
            project,
            command_match=SlashCommandParser.parse("/coco"),
            shell_fast_tracked=False,
            chat_type="group",
        )
        client._reply_text.assert_not_called()
        client._get_mode_handler.assert_not_called()

    def test_dispatch_message_logic_deep_command_forwarded_to_process_with_intent(self):
        client = self._make_client()
        client._add_reaction = MagicMock()
        client._reply_text = MagicMock()
        client._get_mode_handler = MagicMock()
        client._process_with_intent = MagicMock()

        project = MagicMock()
        client._dispatch_message_logic(
            "m1",
            "c1",
            "/deep 写一个函数",
            project,
            auto_enter_mode="coco",
            command_match=SlashCommandParser.parse("/deep 写一个函数"),
        )

        client._process_with_intent.assert_called_once_with(
            "m1",
            "c1",
            "/deep 写一个函数",
            project,
            command_match=SlashCommandParser.parse("/deep 写一个函数"),
            shell_fast_tracked=False,
            chat_type="group",
        )
        client._get_mode_handler.assert_not_called()

    def test_dispatch_message_logic_exit_with_defer(self):
        client = self._make_client()
        client._add_reaction = MagicMock()
        client._reply_text = MagicMock()
        client._control_plane.should_defer_exit = MagicMock(return_value=True)
        client._control_plane.request_deferred_exit = MagicMock()
        client._exit_current_mode = MagicMock()
        client._get_mode_handler = MagicMock()

        project = MagicMock()
        project.project_id = "p1"
        client._dispatch_message_logic(
            "m1",
            "c1",
            "/exit",
            project,
            auto_enter_mode="coco",
            command_match=SlashCommandParser.parse("/exit"),
        )

        client._control_plane.should_defer_exit.assert_called_once()
        client._control_plane.request_deferred_exit.assert_called_once()
        client._reply_text.assert_called_once()
        assert "当前任务完成后退出" in str(client._reply_text.call_args)
        client._exit_current_mode.assert_not_called()
        client._get_mode_handler.assert_not_called()


class TestDualKeyThreadContext(unittest.TestCase):

    def test_register_with_alias_and_lookup_by_alias(self):
        from src.thread.manager import ThreadContextManager
        mgr = ThreadContextManager(ttl=3600, cleanup_interval=99999)
        ctx = mgr.register("reply_id_1", "c1", "p1", mode="coco", alias_keys=["msg_id_1"])
        self.assertIs(mgr.get("reply_id_1"), ctx)
        self.assertIs(mgr.get("msg_id_1"), ctx)
        self.assertEqual(ctx.thread_root_id, "reply_id_1")
        mgr.close()

    def test_register_without_alias(self):
        from src.thread.manager import ThreadContextManager
        mgr = ThreadContextManager(ttl=3600, cleanup_interval=99999)
        ctx = mgr.register("root1", "c1", "p1", mode="claude")
        self.assertIs(mgr.get("root1"), ctx)
        self.assertIsNone(mgr.get("unknown_key"))
        mgr.close()

    def test_remove_cleans_aliases(self):
        from src.thread.manager import ThreadContextManager
        mgr = ThreadContextManager(ttl=3600, cleanup_interval=99999)
        mgr.register("reply_id_1", "c1", "p1", mode="coco", alias_keys=["msg_id_1"])
        mgr.remove("reply_id_1")
        self.assertIsNone(mgr.get("reply_id_1"))
        self.assertIsNone(mgr.get("msg_id_1"))
        mgr.close()

    def test_remove_by_alias_key_cleans_canonical(self):
        from src.thread.manager import ThreadContextManager
        mgr = ThreadContextManager(ttl=3600, cleanup_interval=99999)
        mgr.register("reply_id_1", "c1", "p1", mode="coco", alias_keys=["msg_id_1"])
        mgr.remove("msg_id_1")
        self.assertIsNone(mgr.get("msg_id_1"))
        self.assertIsNone(mgr.get("reply_id_1"))
        mgr.close()

    def test_resolve_context_with_alias_root_id(self):
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = True

        from src.thread.manager import ThreadContextManager
        real_mgr = ThreadContextManager(ttl=3600, cleanup_interval=99999)
        real_mgr.register("reply_id_1", "c1", "p1", mode="coco", alias_keys=["msg_id_1"])
        client._thread_manager = real_mgr

        project = MagicMock()
        project.project_id = "p1"
        client._project_manager.get_project.return_value = project
        client._project_manager.get_project_for_chat.return_value = project

        message = MagicMock()
        message.message_id = "m2"
        message.chat_id = "c1"
        message.root_id = "msg_id_1"
        message.parent_id = None

        from src.thread import get_current_thread_id
        resolved_project, auto_mode = client._resolve_message_context(message)
        self.assertEqual(auto_mode, "coco")
        self.assertIs(resolved_project, project)
        self.assertEqual(get_current_thread_id(), "reply_id_1")
        from src.thread import set_current_thread_id
        set_current_thread_id(None)
        real_mgr.close()

    def test_handle_message_unknown_root_does_not_fallback_to_chat_thread(self):
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = True
        _enrol_test_ingress(client)

        from src.thread.manager import ThreadContextManager
        real_mgr = ThreadContextManager(ttl=3600, cleanup_interval=99999)
        real_mgr.register("reply_id_1", "c1", "p1", mode="coco", alias_keys=["msg_id_1"])
        client._thread_manager = real_mgr
        client._project_manager.get_active_project.return_value = None

        mock_scheduler = MagicMock()
        client._scheduler = mock_scheduler
        client._extract_text_from_message = MagicMock(return_value="hello")
        client._is_system_command_message = MagicMock(return_value=False)
        client._is_likely_shell_command_message = MagicMock(return_value=False)
        client._is_spec_command = MagicMock(return_value=False)
        client._build_control_queue_key = MagicMock(return_value=None)
        client._ensure_request_id = MagicMock(return_value="req1")
        client._message_linker.link_task = MagicMock()

        event = MagicMock()
        event.message.message_id = "om_2"
        event.message.chat_id = "oc_1"
        event.message.chat_type = "group"
        event.message.content = '{"text":"hello"}'
        setattr(event.message, "root_id", "unknown_root")
        setattr(event.message, "parent_id", None)
        event.sender.sender_id.open_id = "ou_test"
        data = MagicMock()
        data.event = event
        data.schema = "im.message.p2p_v1"

        client._handle_message(data)

        call_args = mock_scheduler.submit.call_args
        self.assertIsNotNone(call_args)
        spec = call_args[0][0]
        self.assertNotIn("reply_id_1", spec.queue_key)
        self.assertIn("unknown_root", spec.queue_key)
        real_mgr.close()

    def test_process_async_safety_net_without_root_does_not_fallback_to_chat_thread(self):
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = True

        from src.thread.manager import ThreadContextManager
        real_mgr = ThreadContextManager(ttl=3600, cleanup_interval=99999)
        client._thread_manager = real_mgr

        project = MagicMock()
        project.project_id = "p1"
        client._project_manager.get_project.return_value = None
        client._project_manager.get_project_for_chat.return_value = None
        client._project_manager.get_active_project.return_value = project

        message = MagicMock()
        message.message_id = "m2"
        message.chat_id = "c1"
        setattr(message, "root_id", None)
        setattr(message, "parent_id", None)
        setattr(message, "content", "{}")

        with patch("src.feishu.ws_client.SystemHandler.is_likely_shell_command", return_value=False):
            real_mgr.register("reply_c", "c1", "p1", mode="claude")
            _project, _auto_mode, _text, _is_img = client._handle_image_content(
                message, [], "hello world", "req1", None,
            )
        self.assertIsNone(_auto_mode)
        from src.thread import get_current_thread_id, set_current_thread_id
        self.assertIsNone(get_current_thread_id())
        set_current_thread_id(None)
        real_mgr.close()

    def test_resolve_context_with_canonical_root_id(self):
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = True

        from src.thread.manager import ThreadContextManager
        real_mgr = ThreadContextManager(ttl=3600, cleanup_interval=99999)
        real_mgr.register("reply_id_1", "c1", "p1", mode="coco", alias_keys=["msg_id_1"])
        client._thread_manager = real_mgr

        project = MagicMock()
        project.project_id = "p1"
        client._project_manager.get_project.return_value = project
        client._project_manager.get_project_for_chat.return_value = project

        message = MagicMock()
        message.message_id = "m3"
        message.chat_id = "c1"
        message.root_id = "reply_id_1"
        message.parent_id = None

        resolved_project, auto_mode = client._resolve_message_context(message)
        self.assertEqual(auto_mode, "coco")
        self.assertIs(resolved_project, project)
        real_mgr.close()

    def _make_client(self):
        return _make_ws_client(thread_programming_enabled=True)

    def test_resolve_context_root_mismatch_does_not_fallback_to_chat_thread(self):
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = True

        from src.thread.manager import ThreadContextManager
        real_mgr = ThreadContextManager(ttl=3600, cleanup_interval=99999)
        real_mgr.register("reply_id_1", "c1", "p1", mode="coco", alias_keys=["msg_id_1"])
        client._thread_manager = real_mgr

        project = MagicMock()
        project.project_id = "p1"
        client._project_manager.get_project.return_value = project
        client._project_manager.get_project_for_chat.return_value = project
        client._project_manager.find_by_bound_chat_id.return_value = None
        client._project_manager.get_active_project.return_value = project

        message = MagicMock()
        message.message_id = "m2"
        message.chat_id = "c1"
        message.root_id = "some_unknown_root"
        message.parent_id = None

        from src.thread import get_current_thread_id, set_current_thread_id
        set_current_thread_id(None)
        resolved_project, auto_mode = client._resolve_message_context(message)
        self.assertIsNone(auto_mode)
        self.assertIs(resolved_project, project)
        self.assertIsNone(get_current_thread_id())
        set_current_thread_id(None)
        real_mgr.close()

    def test_safety_net_without_root_does_not_fallback_to_chat_thread(self):
        client = self._make_client()
        client.settings = MagicMock()
        client.settings.thread_programming_enabled = True

        from src.thread.manager import ThreadContextManager
        real_mgr = ThreadContextManager(ttl=3600, cleanup_interval=99999)
        real_mgr.register("reply_id_1", "c1", "p1", mode="claude", alias_keys=["msg_id_1"])
        client._thread_manager = real_mgr

        project = MagicMock()
        project.project_id = "p1"
        client._project_manager.get_project.return_value = None
        client._project_manager.get_project_for_chat.return_value = None
        client._project_manager.get_active_project.return_value = project

        message = MagicMock()
        message.message_id = "m2"
        message.chat_id = "c1"
        message.root_id = None
        setattr(message, "content", "{}")

        from src.thread import get_current_thread_id, set_current_thread_id
        set_current_thread_id(None)
        with patch("src.feishu.ws_client.SystemHandler.is_likely_shell_command", return_value=False):
            _project, _auto_mode, _text, _is_img = client._handle_image_content(
                message, [], "hello world", "req1", None,
            )
        self.assertIsNone(_auto_mode)
        self.assertIsNone(get_current_thread_id())
        set_current_thread_id(None)
        real_mgr.close()


# ======================================================================
# Chat lock interception tests
# ======================================================================


class TestChatLockInterception(unittest.TestCase):
    """Tests for chat lock interception on the card action path."""

    def test_card_action_blocked_when_locked(self):
        """Non-exempt card action is blocked when chat is locked for non-admin."""
        from src.chat_lock import ChatLockManager, _reset_chat_lock_manager_for_testing

        _reset_chat_lock_manager_for_testing()
        try:
            clm = ChatLockManager()
            with patch("src.chat_lock.get_settings") as mock_gs:
                mock_gs.return_value = MagicMock(admin_user_ids={"admin_1"})
                clm.lock_chat("chat-1", "admin_1")

                # Non-admin, non-exempt action should be blocked
                assert clm.should_block_card_action("chat-1", "user_2", "enter_coco") is True

                # Admin should not be blocked
                assert clm.should_block_card_action("chat-1", "admin_1", "enter_coco") is False
        finally:
            _reset_chat_lock_manager_for_testing()

    def test_exempt_actions_pass_through(self):
        """Exempt card actions (stop, show_, CARD_EXEMPT_ACTIONS) pass through even when locked."""
        from src.chat_lock import ChatLockManager, _reset_chat_lock_manager_for_testing

        _reset_chat_lock_manager_for_testing()
        try:
            clm = ChatLockManager()
            with patch("src.chat_lock.get_settings") as mock_gs:
                mock_gs.return_value = MagicMock(admin_user_ids={"admin_1"})
                clm.lock_chat("chat-1", "admin_1")

                # *_stop suffix → exempt
                assert clm.should_block_card_action("chat-1", "user_2", "deep_stop") is False

                # show_* prefix → exempt
                assert clm.should_block_card_action("chat-1", "user_2", "show_help_menu") is False

                # CARD_EXEMPT_ACTIONS → exempt
                assert clm.should_block_card_action("chat-1", "user_2", "force_release_repo_lock") is False
                assert clm.should_block_card_action("chat-1", "user_2", "retry_command") is False
                assert clm.should_block_card_action("chat-1", "user_2", "help_category") is False
                assert clm.should_block_card_action("chat-1", "user_2", "confirm_lock") is False
                assert clm.should_block_card_action("chat-1", "user_2", "cancel_lock") is False
        finally:
            _reset_chat_lock_manager_for_testing()

    def test_unlocked_chat_allows_all(self):
        """When chat is not locked, all actions pass through."""
        from src.chat_lock import ChatLockManager, _reset_chat_lock_manager_for_testing

        _reset_chat_lock_manager_for_testing()
        try:
            clm = ChatLockManager()
            with patch("src.chat_lock.get_settings") as mock_gs:
                mock_gs.return_value = MagicMock(admin_user_ids={"admin_1"})

                assert clm.should_block_card_action("chat-1", "user_2", "enter_coco") is False
                assert clm.should_block_card_action("chat-1", "user_2", "dangerous_action") is False
        finally:
            _reset_chat_lock_manager_for_testing()


class TestLockBlockDedupThreadSafety(unittest.TestCase):
    """AC-R04: ChatLockGate dedup (MessageCache) must be thread-safe."""

    def test_concurrent_should_send_intercept(self):
        """Multiple threads calling ChatLockGate._should_send_intercept must not lose data or raise."""
        import threading

        from src.feishu.chat_lock_gate import ChatLockGate
        from src.feishu.message_cache import MessageCache

        cache = MessageCache(ttl=30, max_size=10_000, cleanup_interval=60)
        gate = ChatLockGate(chat_lock_manager=None, dedup_cache=cache, host=MagicMock())

        errors: list[Exception] = []
        results: list[bool] = []
        lock = threading.Lock()

        def worker(thread_idx: int):
            try:
                for i in range(50):
                    r = gate._should_send_intercept(
                        f"chat_{i % 5}", f"user_{thread_idx}"
                    )
                    with lock:
                        results.append(r)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Threads raised errors: {errors}"
        # Each (chat, user) pair should get True on first call
        assert any(results), "Expected at least some True results"

    def test_dedup_cache_in_gate(self):
        """Verify ChatLockGate wraps a MessageCache for dedup."""
        from src.feishu.chat_lock_gate import ChatLockGate
        from src.feishu.message_cache import MessageCache

        cache = MessageCache(ttl=30, max_size=10_000, cleanup_interval=60)
        gate = ChatLockGate(chat_lock_manager=None, dedup_cache=cache, host=MagicMock())

        assert gate._dedup is cache
        assert isinstance(gate._dedup, MessageCache)


class TestSendLockConflictCardFacade(unittest.TestCase):
    """Tests for FeishuWSClient.send_lock_conflict_card — ensures it delegates
    via _get_handler('system') instead of getattr-based private attribute access."""

    def _make_client(self):
        return _make_ws_client()

    def test_delegates_to_system_handler_via_get_handler(self):
        """send_lock_conflict_card should use _get_handler('system'), not getattr.

        To distinguish from the old ``getattr(self, '_system_handler')`` path,
        we set ``client._system_handler`` to a *different* sentinel object.
        Only the handler registered in ``_handler_ctx.handlers["system"]``
        should be invoked.
        """
        client = self._make_client()

        mock_system_handler = MagicMock(name="handlers_dict_handler")
        sentinel_obj = MagicMock(name="sentinel_should_not_be_called")

        client._handler_ctx.handlers["system"] = mock_system_handler
        client._system_handler = sentinel_obj  # old path would use this

        err = RuntimeError("lock conflict")
        client.send_lock_conflict_card(
            err,
            "msg_1",
            "/deep fix",
            retry_count=2,
            chat_id="chat-1",
        )

        mock_system_handler.send_lock_conflict_card.assert_called_once_with(
            err,
            "msg_1",
            "/deep fix",
            retry_count=2,
            chat_id="chat-1",
        )
        sentinel_obj.send_lock_conflict_card.assert_not_called()

    def test_fallback_when_system_handler_is_none(self):
        """When handlers['system'] is None, fallback text should be sent."""
        client = self._make_client()
        client._handler_ctx.handlers["system"] = None
        client._reply_text = MagicMock()

        err = RuntimeError("lock conflict")
        client.send_lock_conflict_card(err, "msg_2", "/deep test")

        client._reply_text.assert_called_once()
        call_args = client._reply_text.call_args
        self.assertEqual(call_args[0][0], "msg_2")
        self.assertIn("🔒", call_args[0][1])

    def test_fallback_when_system_handler_key_missing(self):
        """When 'system' key is absent from handlers dict, fallback text should be sent."""
        client = self._make_client()
        client._handler_ctx.handlers.pop("system", None)
        client._reply_text = MagicMock()

        err = RuntimeError("lock conflict")
        client.send_lock_conflict_card(err, "msg_3", "/spec run")

        client._reply_text.assert_called_once()
        call_args = client._reply_text.call_args
        self.assertEqual(call_args[0][0], "msg_3")
        self.assertIn("🔒", call_args[0][1])


class TestNoGetAttrSystemHandlerPattern(unittest.TestCase):
    """Static regression guard: ensure ``getattr(self, '_system_handler' ...)``
    is never reintroduced in ws_client.py.

    This test scans source code via regex.  If ws_client.py undergoes a major
    rename/restructure, update the ``_SOURCE_PATH`` constant below.
    """

    _SOURCE_PATH = "src/feishu/ws_client.py"

    def test_no_getattr_system_handler_pattern(self):
        """ws_client.py must not contain getattr(self, '_system_handler'...)."""
        import re
        with open(self._SOURCE_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        matches = re.findall(r"getattr\(self,\s*[\"']_system_handler", source)
        self.assertEqual(
            len(matches), 0,
            f"Found {len(matches)} getattr(self, '_system_handler'...) pattern(s) in {self._SOURCE_PATH}. "
            "All handler access should use self._get_handler('system').",
        )


class TestChatLockHandlerDelegation(unittest.TestCase):
    """Verify that ChatLockGate._try_block delegates card sending to the
    system handler obtained via host._get_handler('system')."""

    def _make_gate(self, *, mock_clm=None, mock_handler=None, sentinel=None):
        """Build a ChatLockGate with mocked dependencies."""
        from src.feishu.chat_lock_gate import ChatLockGate
        from src.feishu.message_cache import MessageCache

        host = MagicMock()
        if mock_handler is not None:
            host._get_handler.return_value = mock_handler
        else:
            host._get_handler.return_value = None

        cache = MessageCache(ttl=30, max_size=10_000, cleanup_interval=60)
        gate = ChatLockGate(chat_lock_manager=mock_clm, dedup_cache=cache, host=host)
        return gate, host

    def test_try_block_uses_get_handler(self):
        """ChatLockGate._try_block should use host._get_handler('system')."""
        mock_clm = MagicMock()
        mock_clm.should_block.return_value = True
        mock_handler = MagicMock(name="dict_handler")

        gate, host = self._make_gate(mock_clm=mock_clm, mock_handler=mock_handler)

        result = gate._try_block("chat_1", "user_1", "msg_1")

        self.assertTrue(result)
        host._get_handler.assert_called_with("system")
        mock_handler.send_chat_lock_intercept_card.assert_called_once_with(
            "msg_1", "chat_1", mock_clm,
        )

    def test_try_block_fallback_when_handler_none(self):
        """ChatLockGate._try_block should fall back to host._reply_text when handler is None."""
        mock_clm = MagicMock()
        mock_clm.should_block.return_value = True

        gate, host = self._make_gate(mock_clm=mock_clm, mock_handler=None)

        result = gate._try_block("chat_1", "user_1", "msg_1")

        self.assertTrue(result)
        host._reply_text.assert_called_once()

    def test_try_block_throttled_uses_get_handler(self):
        """When dedup suppresses the full card, throttled reply should use host._get_handler."""
        mock_clm = MagicMock()
        mock_clm.should_block.return_value = True
        mock_handler = MagicMock(name="dict_handler")

        gate, host = self._make_gate(mock_clm=mock_clm, mock_handler=mock_handler)
        # First call consumes the dedup slot
        gate._should_send_intercept("chat_1", "user_1")
        # Second call should be throttled
        result = gate._try_block("chat_1", "user_1", "msg_1")

        self.assertTrue(result)
        mock_handler.send_chat_lock_throttled_reply.assert_called_once()
