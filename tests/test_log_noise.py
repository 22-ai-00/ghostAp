import asyncio
import unittest
from unittest.mock import MagicMock, patch

from src.feishu.handlers.deep import DeepHandler


class TestLogNoise(unittest.TestCase):
    def test_deep_handler_timeout_warning(self):
        """验证 DeepHandler 将 TimeoutError 记录为 warning"""
        mock_ctx = MagicMock()
        handler = DeepHandler(mock_ctx)

        # Setup project manager mock to return a tuple
        mock_project = MagicMock()
        mock_project.project_id = "test_project_id"
        mock_ctx.project_manager.get_or_create_project_for_path.return_value = (mock_project, False)

        # Mock renderer method
        handler.renderer.create_deep_callbacks = MagicMock()

        # Ensure no existing engine is running
        mock_ctx.deep_engine_manager.get.return_value = None

        # Mock other dependencies to prevent errors before scheduler.submit
        handler.get_working_dir = MagicMock(return_value="/tmp")
        handler.ensure_request_id = MagicMock(return_value="req_id")
        handler.get_engine_name = MagicMock(return_value="Deep(Coco)")
        handler.reply_text = MagicMock()
        handler.add_reaction = MagicMock()
        handler.create_rate_limit_callback = MagicMock()

        # Mock CardBuilder
        with patch("src.feishu.handlers.deep.CardBuilder") as mock_card_builder:
            mock_card_builder.build_info_card.return_value = ("interactive", "{}")

            mock_engine = MagicMock()
            mock_ctx.deep_engine_manager.get_or_create.return_value = mock_engine

            # Simulate TimeoutError in plan_and_execute
            mock_engine.plan_and_execute.side_effect = TimeoutError("Simulated timeout")

            # Patch logger in engine_base where the logging now happens
            with patch("src.feishu.handlers.engine_base.logger") as mock_logger:
                # Execute the callback immediately when submitted
                def mock_submit(spec, func):
                    func(None)
                    return MagicMock(run_id="run_id")

                handler.scheduler.submit.side_effect = mock_submit

                # Mock message linker to avoid errors
                mock_ctx.message_linker.link_task = MagicMock()

                handler.start_deep_engine("mid", "cid", "req")

                # Verify warning was logged
                mock_logger.warning.assert_called()
                call_args = mock_logger.warning.call_args
                self.assertIn("Deep Agent Engine 执行超时", call_args[0][0])

                # Verify NO error log for this exception
                mock_logger.error.assert_not_called()

    def test_lifecycle_action_timeout_warning(self):
        """验证生命周期操作(stop/pause)将 TimeoutError 记录为 warning"""
        mock_ctx = MagicMock()
        handler = DeepHandler(mock_ctx)

        # Mock dependencies
        handler.reply_text = MagicMock()
        mock_ctx.deep_engine_manager.get.return_value = MagicMock()  # engine exists
        mock_engine = mock_ctx.deep_engine_manager.get.return_value
        mock_engine.is_running = True

        # Simulate TimeoutError during stop
        mock_engine.stop.side_effect = TimeoutError("Stop timeout")

        with patch("src.feishu.handlers.engine_base.logger") as mock_logger:
            # Execute stop action which should use _safe_lifecycle_action
            handler.stop_deep_engine("mid", "cid")

            # Verify warning was logged
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            self.assertIn("Deep Agent stop 操作超时", call_args[0][0])

            # Verify NO error log for this exception
            mock_logger.error.assert_not_called()

            # Verify error message sent to user (get_error_detail formats TimeoutError)
            handler.reply_text.assert_called_with("mid", "❌ stop失败: 操作超时 (Stop timeout)")

    def test_spec_handler_timeout_warning(self):
        """验证 SpecHandler 将 TimeoutError 记录为 warning"""
        from src.feishu.handlers.spec import SpecHandler

        mock_ctx = MagicMock()
        handler = SpecHandler(mock_ctx)

        # Setup project manager mock to return a tuple
        mock_project = MagicMock()
        mock_project.project_id = "test_project_id"
        mock_ctx.project_manager.get_or_create_project_for_path.return_value = (mock_project, False)

        # Mock renderer method
        handler.renderer.create_spec_callbacks = MagicMock()

        # Ensure no existing engine is running
        mock_ctx.spec_engine_manager.get.return_value = None

        # Mock other dependencies
        handler.get_working_dir = MagicMock(return_value="/tmp")
        handler.ensure_request_id = MagicMock(return_value="req_id")
        handler.get_engine_name = MagicMock(return_value="Coco")
        handler.reply_text = MagicMock()
        handler.add_reaction = MagicMock()
        handler.create_rate_limit_callback = MagicMock()
        handler.send_card_to_chat = MagicMock()

        # Mock CardBuilder
        with patch("src.feishu.handlers.spec.CardBuilder") as mock_card_builder:
            mock_card_builder.build_info_card.return_value = ("interactive", "{}")

            mock_engine = MagicMock()
            mock_ctx.spec_engine_manager.get_or_create.return_value = mock_engine

            # Simulate TimeoutError in execute
            mock_engine.execute.side_effect = asyncio.TimeoutError("Simulated timeout")

            # Patch logger (warning now emitted from engine_base module)
            with patch("src.feishu.handlers.engine_base.logger") as mock_logger:

                def mock_submit(spec, func):
                    func(None)
                    return MagicMock(run_id="run_id")

                handler.scheduler.submit.side_effect = mock_submit

                mock_ctx.message_linker.link_task = MagicMock()

                handler.start_spec_engine("mid", "cid", "req")

                # Verify warning was logged
                mock_logger.warning.assert_called()
                args = mock_logger.warning.call_args
                # engine_base uses f-string: "Spec Engine 执行超时 (task_id=...): ..."
                log_msg = args[0][0]
                self.assertIn("Spec Engine 执行超时", log_msg)
                self.assertIn("Simulated timeout", log_msg)

                # Verify NO error log for this exception
                mock_logger.error.assert_not_called()

    def test_ws_client_message_timeout_warning(self):
        """验证 FeishuWSClient 处理消息超时时记录为 warning"""
        from contextlib import ExitStack

        from src.feishu.ws_client import FeishuWSClient

        # Create a partial mock of WSClient to test _process_message_async
        # We can't easily instantiate the full client due to dependencies,
        # so we'll mock the necessary parts

        # Mock message data
        mock_data = MagicMock()
        mock_data.event.message.message_id = "om_timeout"
        mock_data.event.message.chat_id = "oc_timeout"
        mock_data.event.message.chat_type = "group"
        mock_data.event.message.create_time = None
        mock_data.event.message.parent_id = None
        mock_data.event.message.root_id = None
        mock_data.event.message.message_type = "text"
        mock_data.event.message.content = '{"text": "hello"}'
        mock_data.event.sender.sender_id.open_id = "ou_timeout"

        with ExitStack() as stack:
            stack.enter_context(patch("src.feishu.ws_client.get_settings"))
            stack.enter_context(patch("src.feishu.ws_client.ACPSessionManager"))
            stack.enter_context(patch("src.feishu.ws_client.IntentRecognizer"))
            stack.enter_context(patch("src.feishu.ws_client.TaskScheduler"))
            stack.enter_context(patch("src.feishu.ws_client.ProjectManager"))
            stack.enter_context(patch("src.feishu.ws_client.MessageProjectMapper"))
            stack.enter_context(patch("src.feishu.ws_client.MessageLinker"))
            stack.enter_context(patch("src.mode.ModeManager"))
            stack.enter_context(patch("src.feishu.ws_client.ProjectContextManager"))
            stack.enter_context(patch("src.feishu.ws_client.DeepEngineManager"))
            stack.enter_context(patch("src.feishu.ws_client.SpecEngineManager"))
            stack.enter_context(patch("src.feishu.ws_client.HandlerContext"))
            stack.enter_context(patch("src.feishu.ws_client.ActionDispatcher"))
            stack.enter_context(patch("src.feishu.ws_client.CocoModeHandler"))
            stack.enter_context(patch("src.feishu.ws_client.ClaudeModeHandler"))
            stack.enter_context(patch("src.feishu.ws_client.DeepHandler"))
            stack.enter_context(patch("src.feishu.ws_client.SpecHandler"))
            stack.enter_context(patch("src.feishu.ws_client.ProjectHandler"))
            stack.enter_context(patch("src.feishu.ws_client.SystemHandler"))
            stack.enter_context(patch("src.feishu.ws_client.DiagnosticsHandler"))

            client = FeishuWSClient(lambda *args: None)

            from src.access_control import (
                IngressAccessPolicy,
                IngressAccessPolicyProvider,
            )
            from src.config import IngressAccessMode

            client._ingress_access_policy_provider = IngressAccessPolicyProvider(
                IngressAccessPolicy(
                    admin_ids=frozenset(),
                    allowed_user_ids=frozenset({"ou_timeout"}),
                    allowed_chat_ids=frozenset({"oc_timeout"}),
                    mode=IngressAccessMode.ENFORCED,
                    admin_bootstrap_scope="p2p_only",
                )
            )

            # Mock _ensure_request_id
            client._ensure_request_id = MagicMock(return_value="req_id")
            client._is_message_expired = MagicMock(return_value=False)
            client._is_duplicate_message = MagicMock(return_value=False)
            client._chat_lock_gate = MagicMock()
            client._chat_lock_gate.check = MagicMock(return_value=False)
            client._reply_text = MagicMock()

            # Mock _get_image_handler to return a mock that raises TimeoutError on parse_message
            mock_image_handler = MagicMock()
            mock_image_handler.parse_message.side_effect = asyncio.TimeoutError("Timeout in parsing")
            client._get_image_handler = MagicMock(return_value=mock_image_handler)

            with patch("src.feishu.ws_client.logger") as mock_logger:
                client._process_message_async(mock_data)

                # Verify warning logged
                mock_logger.warning.assert_called()
                args = mock_logger.warning.call_args
                self.assertIn("处理消息超时", args[0][0])

                # Verify NO error log for this exception
                mock_logger.error.assert_not_called()

                # Verify user was notified about timeout
                client._reply_text.assert_called_once()
                reply_text = str(client._reply_text.call_args)
                self.assertIn("超时", reply_text)

    def test_ws_client_card_action_timeout_warning(self):
        """验证 FeishuWSClient 处理卡片动作超时时记录为 warning"""
        from contextlib import ExitStack

        from src.feishu.ws_client import FeishuWSClient

        with ExitStack() as stack:
            stack.enter_context(patch("src.feishu.ws_client.get_settings"))
            stack.enter_context(patch("src.feishu.ws_client.ACPSessionManager"))
            stack.enter_context(patch("src.feishu.ws_client.IntentRecognizer"))
            stack.enter_context(patch("src.feishu.ws_client.TaskScheduler"))
            stack.enter_context(patch("src.feishu.ws_client.ProjectManager"))
            stack.enter_context(patch("src.feishu.ws_client.MessageProjectMapper"))
            stack.enter_context(patch("src.feishu.ws_client.MessageLinker"))
            stack.enter_context(patch("src.mode.ModeManager"))
            stack.enter_context(patch("src.feishu.ws_client.ProjectContextManager"))
            stack.enter_context(patch("src.feishu.ws_client.DeepEngineManager"))
            stack.enter_context(patch("src.feishu.ws_client.SpecEngineManager"))
            stack.enter_context(patch("src.feishu.ws_client.HandlerContext"))
            stack.enter_context(patch("src.feishu.ws_client.ActionDispatcher"))
            stack.enter_context(patch("src.feishu.ws_client.CocoModeHandler"))
            stack.enter_context(patch("src.feishu.ws_client.ClaudeModeHandler"))
            stack.enter_context(patch("src.feishu.ws_client.DeepHandler"))
            stack.enter_context(patch("src.feishu.ws_client.SpecHandler"))
            stack.enter_context(patch("src.feishu.ws_client.ProjectHandler"))
            stack.enter_context(patch("src.feishu.ws_client.SystemHandler"))
            stack.enter_context(patch("src.feishu.ws_client.DiagnosticsHandler"))

            client = FeishuWSClient(lambda *args: None)

            # Mock data
            mock_data = MagicMock()
            mock_data.header.event_id = "evt_id"
            mock_data.event.action.value = {"action": "test"}
            mock_data.event.context.open_message_id = "mid"
            mock_data.event.context.open_chat_id = "cid"

            client._card_event_cache.is_duplicate = MagicMock(return_value=False)
            client._chat_lock_gate = MagicMock()
            client._chat_lock_gate.check_card_action = MagicMock(return_value=False)
            client._action_dispatcher.dispatch.side_effect = asyncio.TimeoutError("Timeout in dispatch")
            client._reply_text = MagicMock()

            with patch("src.feishu.ws_client.logger") as mock_logger:
                client._process_card_action_async(mock_data)

                # Verify warning logged
                mock_logger.warning.assert_called()
                args = mock_logger.warning.call_args
                self.assertIn("处理卡片动作超时", args[0][0])

                # Verify NO error log for this exception
                mock_logger.error.assert_not_called()

                # Verify user was notified about timeout
                client._reply_text.assert_called()
                reply_text = str(client._reply_text.call_args)
                self.assertIn("超时", reply_text)


if __name__ == "__main__":
    unittest.main()
