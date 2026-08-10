import json
import unittest
from unittest.mock import MagicMock, patch

from src.card.builders.system import SystemBuilder
from src.feishu.handlers.system import SystemHandler
from src.feishu.slash_command_parser import SlashCommandParser


class TestRefactorRobustness(unittest.TestCase):
    def test_shell_output_truncation(self):
        """Test that long shell output is truncated in the card."""
        from src.sandbox.executor import ExecutionResult

        # Create a long output string (> SHELL_STDOUT_MAX=16000 chars)
        long_output = "a" * 20000
        result = ExecutionResult(return_code=0, stdout=long_output, stderr="", success=True)

        msg_type, content_json = SystemBuilder.build_shell_result_card("echo long", result)
        content = json.loads(content_json)

        # Find the code block with output
        found_output = False
        for element in content["body"]["elements"]:
            if element["tag"] == "markdown" and "```BASH" in element["content"]:
                text = element["content"]
                if "\u5df2\u622a\u65ad" in text:  # 已截断
                    found_output = True
                    # Check that it's actually shorter than the original
                    self.assertLess(len(text), 20000)
                    # Check for truncation marker
                    self.assertIn("已截断", text)

        self.assertTrue(found_output, "Did not find truncated output in card")

    def test_system_handler_dispatch(self):
        """Test that SystemHandler correctly dispatches commands using the new registry."""
        mock_ctx = MagicMock()
        handler = SystemHandler(mock_ctx)

        # Mock handlers in registry
        coco_mock = MagicMock()
        project_mock = MagicMock()
        diagnostics_mock = MagicMock()
        mock_ctx.handlers.get.side_effect = lambda k: {
            "coco": coco_mock,
            "project": project_mock,
            "diagnostics": diagnostics_mock,
        }.get(k)

        # Test exact match
        handler.handle_intercepted_command(
            "mid",
            "cid",
            "/coco_info",
            command_match=SlashCommandParser.parse("/coco_info"),
        )
        coco_mock.show_info.assert_called_with("mid", "cid", None)

        # Test prefix match
        handler.handle_intercepted_command(
            "mid",
            "cid",
            "/status detail",
            command_match=SlashCommandParser.parse("/status detail"),
        )
        diagnostics_mock.show_unified_status.assert_called_with("mid", "cid", "/status detail", None)

        # Test fallback: unknown slash commands get a concise system reply
        # instead of rendering the full help card.
        with (
            patch.object(handler, "show_full_help") as mock_help,
            patch.object(handler, "reply_text") as mock_reply,
        ):
            handler.handle_intercepted_command(
                "mid",
                "cid",
                "/unknown_cmd",
                command_match=SlashCommandParser.parse("/unknown_cmd"),
            )
            mock_reply.assert_called_once()
            self.assertIn("未知命令", mock_reply.call_args.args[1])
            mock_help.assert_not_called()

if __name__ == "__main__":
    unittest.main()
