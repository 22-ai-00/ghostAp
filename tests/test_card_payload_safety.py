import json
import unittest

from src.card.shared.truncation import check_and_truncate_payload


class TestCardPayloadSafety(unittest.TestCase):
    def test_payload_no_truncation(self):
        """Test small payload passes through"""
        small_content = json.dumps({"text": "hello"})
        result = check_and_truncate_payload(small_content, max_size=1000)
        self.assertEqual(result, small_content)

    def test_payload_truncation(self):
        """Test large string fields are truncated when they exceed the per-string cap."""
        # Threshold for content/text field truncation is 8000 chars.
        long_text = "a" * 20000
        card = {"header": {"title": "Test"}, "elements": [{"tag": "markdown", "content": long_text}]}
        content = json.dumps(card)

        # Set limit smaller than content so truncation path is taken.
        max_size = 10000
        result = check_and_truncate_payload(content, max_size=max_size)

        # Verify result is valid JSON
        truncated_card = json.loads(result)
        truncated_text = truncated_card["elements"][0]["content"]

        # Verify text was truncated (capped at 8000 + suffix)
        self.assertTrue(len(truncated_text) < 20000)
        self.assertTrue(truncated_text.endswith("…(已截断)"))
        self.assertTrue(any("内容过长" in str(element) for element in truncated_card["elements"]))

    def test_recursive_truncation(self):
        """Test nested objects are processed — uses the >10K fallback string cap."""
        long_text = "b" * 20000
        card = {"body": {"nested": {"deep": long_text}}}
        content = json.dumps(card)

        result = check_and_truncate_payload(content, max_size=12000)
        truncated_card = json.loads(result)

        deep_val = truncated_card["body"]["nested"]["deep"]
        self.assertTrue(len(deep_val) < 20000)
        self.assertIn("已截断", deep_val)

    def test_fallback_truncation(self):
        """Test drastic fallback if truncation fails to reduce size enough"""
        # Even after truncating strings to 2000 chars, if we have MANY fields, it might still be too big
        # Construct a card with many fields
        card = {"items": ["a" * 100 for _ in range(100)]}  # ~10KB
        content = json.dumps(card)

        # Set impossible limit
        max_size = 100

        result = check_and_truncate_payload(content, max_size=max_size)

        # Should return fallback card
        fallback = json.loads(result)
        self.assertEqual(fallback["header"]["title"]["content"], "⚠️ 卡片过大")


class TestDefaultBudgetTruncation(unittest.TestCase):
    """Integration test: verify truncation works with real 27KB (27*1024) budget."""

    def setUp(self):
        from src.card.shared.truncation import check_and_truncate_payload
        self.truncate = check_and_truncate_payload

    def test_oversized_payload_truncated_within_feishu_limit(self):
        """A 40KB payload should be truncated to <=28KB (Feishu card limit)."""
        # Construct a realistic card with large markdown content
        long_markdown = "x" * 35000  # 35KB of text
        card = {
            "schema": "2.0",
            "header": {"title": {"tag": "plain_text", "content": "Test Card"}},
            "body": {"elements": [
                {"tag": "markdown", "content": long_markdown, "element_id": "content_md"},
            ]},
        }
        content = json.dumps(card, ensure_ascii=False)
        assert len(content.encode("utf-8")) > 27 * 1024, "Test setup: content must exceed budget"

        # Use default max_size (27*1024 from THRESHOLDS)
        result = self.truncate(content)

        # Result must be valid JSON
        parsed = json.loads(result)
        assert "header" in parsed or "elements" in parsed or "body" in parsed

        # Result must fit within Feishu's 28KB limit
        result_size = len(result.encode("utf-8"))
        assert result_size <= 28000, f"Truncated payload {result_size} exceeds 28KB Feishu limit"

    def test_within_budget_passes_through(self):
        """A small payload under 27KB should pass through unchanged."""
        card = {
            "schema": "2.0",
            "header": {"title": {"tag": "plain_text", "content": "Small"}},
            "body": {"elements": [{"tag": "markdown", "content": "hello"}]},
        }
        content = json.dumps(card, ensure_ascii=False)
        result = self.truncate(content)
        assert result == content


class TestWorkflowPayloadWhitelistSafety(unittest.TestCase):
    def test_stop_running_payload_remains_strict_three_keys(self):
        from src.card.actions.dispatch import WORKFLOW_STOP_RUNNING
        from src.card.events.payloads import filter_workflow_button_value

        cleaned = filter_workflow_button_value(
            {
                "action": WORKFLOW_STOP_RUNNING,
                "chat_id": "oc_1",
                "project_id": "p_1",
                "effort": "high",
                "agent_id": "A1",
                "confirmed": True,
                "admin": True,
            }
        )
        self.assertEqual(
            cleaned,
            {"action": WORKFLOW_STOP_RUNNING, "chat_id": "oc_1", "project_id": "p_1"},
        )

    def test_wf_config_payload_allows_bounded_fields_and_strips_forged(self):
        from src.card.actions.dispatch import WF_SELECT_EFFORT
        from src.card.events.payloads import filter_workflow_button_value

        raw = {
            "action": WF_SELECT_EFFORT,
            "chat_id": "oc_1",
            "project_id": "p_1",
            "config_session_id": "cfg_1",
            "preset_id": "collaborative",
            "agent_id": "A2",
            "orchestrator_agent_id": "A1",
            "role": "worker",
            "tool_name": "traex",
            "model_name": "openai/gpt-4o",
            "profile": "max",
            "effort": "xhigh",
            "_option": "xhigh",
            "_options": ["low", "xhigh", "bad\nctrl"],
            "confirmed": True,
            "admin": True,
            "override_budget": 99999,
            "shell_cmd": "rm -rf /",
            "bad_ctrl": "abc\x00def",
            "tool_name_bad_type": 123,
        }
        cleaned = filter_workflow_button_value(raw)
        self.assertEqual(cleaned["action"], WF_SELECT_EFFORT)
        self.assertEqual(cleaned["effort"], "xhigh")
        self.assertEqual(cleaned["model_name"], "openai/gpt-4o")
        self.assertEqual(cleaned["_options"], ["low", "xhigh"])
        for bad in ("confirmed", "admin", "override_budget", "shell_cmd", "bad_ctrl", "tool_name_bad_type"):
            self.assertNotIn(bad, cleaned)

    def test_invalid_action_or_oversized_fields_rejected(self):
        from src.card.events.payloads import filter_workflow_button_value

        self.assertEqual(filter_workflow_button_value({"action": "bad action\n"}), {})
        cleaned = filter_workflow_button_value(
            {
                "action": "wf_select_model",
                "model_name": "x" * 300,
                "agent_id": "A1\n",
                "effort": "high",
            }
        )
        self.assertEqual(cleaned, {"action": "wf_select_model", "effort": "high"})

    def test_lone_surrogates_rejected_in_workflow_button_value(self):
        from src.card.actions.dispatch import WORKFLOW_STOP_RUNNING
        from src.card.events.payloads import filter_workflow_button_value

        cleaned = filter_workflow_button_value(
            {
                "action": "wf_select_model",
                "model_name": "bad\ud800model",
                "display_name": "bad\udfffname",
                "_option": "opt\ud800",
                "_options": ["valid", "bad\ud800opt"],
                "agent_id": "A1\ud800",
                "effort": "high",
            }
        )
        self.assertEqual(
            cleaned,
            {
                "action": "wf_select_model",
                "_options": ["valid"],
                "effort": "high",
            },
        )

        stop_cleaned = filter_workflow_button_value(
            {
                "action": WORKFLOW_STOP_RUNNING,
                "chat_id": "oc_\ud800_1",
                "project_id": "p_1",
            }
        )
        self.assertEqual(
            stop_cleaned,
            {
                "action": WORKFLOW_STOP_RUNNING,
                "project_id": "p_1",
            },
        )


if __name__ == "__main__":
    unittest.main()
