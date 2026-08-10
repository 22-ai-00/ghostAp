import json
import unittest
from unittest.mock import MagicMock

from src.feishu.handlers.base import BaseHandler
from src.feishu.renderers.base import BaseRenderer


class TestCardPayloadSafety(unittest.TestCase):
    def setUp(self):
        self.mock_handler = MagicMock(spec=BaseHandler)
        self.mock_handler.ctx = MagicMock()
        self.mock_handler.settings = MagicMock()
        self.renderer = BaseRenderer(self.mock_handler)

    def test_payload_no_truncation(self):
        """Test small payload passes through"""
        small_content = json.dumps({"text": "hello"})
        result = self.renderer._check_and_truncate_payload(small_content, max_size=1000)
        self.assertEqual(result, small_content)

    def test_payload_truncation(self):
        """Test large string fields are truncated when they exceed the per-string cap."""
        # Threshold for content/text field truncation is 8000 chars.
        long_text = "a" * 20000
        card = {"header": {"title": "Test"}, "elements": [{"tag": "markdown", "content": long_text}]}
        content = json.dumps(card)

        # Set limit smaller than content so truncation path is taken.
        max_size = 10000
        result = self.renderer._check_and_truncate_payload(content, max_size=max_size)

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

        result = self.renderer._check_and_truncate_payload(content, max_size=12000)
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

        result = self.renderer._check_and_truncate_payload(content, max_size=max_size)

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


if __name__ == "__main__":
    unittest.main()
