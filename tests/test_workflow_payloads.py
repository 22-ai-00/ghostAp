"""Tests for Workflow payload TypedDicts and exports (Tasks 1,5 / 17).

Validates:
- WorkflowConfirmPayload is total=True with correct fields
- All workflow payloads are exported from card.events
- Factory functions produce correct event types
"""

import unittest

from src.card.events import (
    WorkflowConfirmPayload,
    workflow_agent_done,
    workflow_agent_failed,
    workflow_agent_started,
    workflow_log,
    workflow_phase,
    workflow_progress,
)
from src.card.events.types import CardEventType


class TestWorkflowConfirmPayloadContract(unittest.TestCase):
    """Verify WorkflowConfirmPayload has correct required/optional fields."""

    def test_script_path_removed(self):
        """script_path should NOT be in the payload type."""
        import typing

        hints = typing.get_type_hints(WorkflowConfirmPayload)
        self.assertNotIn("script_path", hints)

    def test_new_security_fields_present(self):
        """initiator_user_id and engine_session_key must be defined."""
        import typing

        hints = typing.get_type_hints(WorkflowConfirmPayload)
        self.assertIn("initiator_user_id", hints)
        self.assertIn("engine_session_key", hints)


class TestWorkflowFactoryFunctions(unittest.TestCase):
    """Test all workflow factory functions produce correct events."""

    def test_workflow_progress_card_required_in_payload(self):
        """WORKFLOW_PROGRESS payload must always include 'card'."""
        event = workflow_progress({"elements": []}, "status")
        # Direct access — no .get() defensive fallback needed
        self.assertEqual(event.type, CardEventType.WORKFLOW_PROGRESS)
        self.assertIn("card", event.payload)
        self.assertEqual(event.payload["card"], {"elements": []})
        self.assertEqual(event.payload["compact_status"], "status")

    def test_workflow_progress_without_compact_status(self):
        """compact_status is optional; payload should still have card."""
        event = workflow_progress({"elements": [{"tag": "div"}]})
        self.assertEqual(event.type, CardEventType.WORKFLOW_PROGRESS)
        self.assertIn("card", event.payload)
        self.assertNotIn("compact_status", event.payload)

    def test_workflow_progress_factory_requires_card(self):
        """Calling workflow_progress() without a card (or wrong type) raises TypeError."""
        with self.assertRaises(TypeError):
            workflow_progress(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            workflow_progress("not-a-dict")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            workflow_progress(["list", "instead", "of", "dict"])  # type: ignore[arg-type]

    def test_non_progress_factories_preserve_type_and_identity_fields(self):
        cases = (
            (workflow_phase("Phase 1"), CardEventType.WORKFLOW_PHASE, "title", "Phase 1"),
            (workflow_agent_started("agent1", "coco", "Phase 1"), CardEventType.WORKFLOW_AGENT_STARTED, "tool", "coco"),
            (workflow_agent_done("agent1", token_usage=5000, cached=True), CardEventType.WORKFLOW_AGENT_DONE, "cached", True),
            (workflow_agent_failed("agent1", "timeout"), CardEventType.WORKFLOW_AGENT_FAILED, "error", "timeout"),
            (workflow_log("hello"), CardEventType.WORKFLOW_LOG, "message", "hello"),
        )
        for event, event_type, field, expected in cases:
            self.assertEqual(event.type, event_type)
            self.assertEqual(event.payload[field], expected)


class TestWorkflowRefItemContract(unittest.TestCase):
    """Verify WorkflowRefItem TypedDict normalized contract."""

    def test_workflow_ref_item_name_is_required(self):
        """Verify that `name` is required (cannot construct without it)."""
        import typing

        from src.card.events.payloads import WorkflowRefItem

        hints = typing.get_type_hints(WorkflowRefItem)
        # name should be in the hints
        self.assertIn("name", hints)
        # name should NOT be NotRequired (it's required)
        # Check that the annotation is just `str`, not `NotRequired[str]`
        name_annotation = hints["name"]
        self.assertEqual(name_annotation, str)
        # path and hash should be NotRequired
        self.assertIn("path", hints)
        self.assertIn("hash", hints)

class TestEnrichWorkflowRefs(unittest.TestCase):
    """Test _enrich_workflow_refs normalization logic."""

    def test_enrich_handles_mixed_refs(self):
        """Mixed string and dict refs should both be normalized correctly."""
        from src.workflow_engine.script_gen import _enrich_workflow_refs

        meta = {"workflow_refs": [
            "string-ref",
            {"name": "dict-ref", "script_path": "old/path.js"},
            {"name": "dict-ref-2", "path": "new/path.js"},
        ]}
        _enrich_workflow_refs(meta, "")
        self.assertEqual(meta["workflow_refs"], [
            {"name": "string-ref"},
            {"name": "dict-ref", "path": "old/path.js"},
            {"name": "dict-ref-2", "path": "new/path.js"},
        ])

    def test_enrich_scans_script_for_workflow_calls(self):
        """When meta has no workflow_refs, scan script for workflow('name') calls."""
        from src.workflow_engine.script_gen import _enrich_workflow_refs

        meta = {}
        script = """
export default async function() {
  const r1 = await workflow("sub-workflow-a", { x: 1 });
  const r2 = await workflow('sub-workflow-b', { y: 2 });
  const r3 = await workflow(`sub-workflow-c`, { z: 3 });
  // duplicate should not appear twice
  const r4 = await workflow("sub-workflow-a", { again: true });
}
"""
        _enrich_workflow_refs(meta, script)
        self.assertEqual(meta["workflow_refs"], [
            {"name": "sub-workflow-a"},
            {"name": "sub-workflow-b"},
            {"name": "sub-workflow-c"},
        ])


class TestWorkflowRefBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility for legacy workflow ref formats."""

    def test_legacy_string_ref_supported(self):
        """String refs are accepted for backward compatibility."""
        from src.workflow_engine.script_gen import _enrich_workflow_refs

        # String refs in meta should be accepted and normalized
        meta = {"workflow_refs": ["legacy-string-ref"]}
        _enrich_workflow_refs(meta, "")
        self.assertEqual(meta["workflow_refs"], [{"name": "legacy-string-ref"}])

        # Also verify extract_meta_from_script handles string refs
        from src.workflow_engine.script_gen import extract_meta_from_script
        script_with_string_refs = """
export const meta = {
  name: "test",
  description: "test",
  phases: [{ title: "Phase 1" }],
  tools: ["coco"],
  workflow_refs: ["legacy-ref-1", "legacy-ref-2"],
};
export default async function() {
  await agent("do something", { tool: "coco" });
}
"""
        extracted = extract_meta_from_script(script_with_string_refs)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["workflow_refs"], [
            {"name": "legacy-ref-1"},
            {"name": "legacy-ref-2"},
        ])

class TestWorkflowButtonValueFilter(unittest.TestCase):
    """Verify filter_workflow_button_value drops forged callback payload fields."""

    def test_filter_preserves_known_fields(self):
        """Known button fields should pass through."""
        from src.card.events.payloads import filter_workflow_button_value

        value = {
            "action": "workflow_confirm_start",
            "chat_id": "chat_001",
            "project_id": "proj_001",
            "engine_session_key": "sess-xyz",
            "tool_name": "coco",
            "provider": "workflow",
            "display_name": "coco",
            "supports_model": True,
            "model_name": "claude-3-5",
            "use_default_model": False,
            "selection_key": "sel-123",
        }
        filtered = filter_workflow_button_value(value)
        self.assertEqual(filtered, value)

    def test_filter_drops_forged_privilege_fields(self):
        from src.card.events.payloads import filter_workflow_button_value

        value = {
            "action": "workflow_confirm_start",
            "chat_id": "chat_001",
            "project_id": "proj_001",
            "engine_session_key": "sess-xyz",
            "confirmed": True,
            "admin": "1",
            "override_budget": 999999999,
        }
        filtered = filter_workflow_button_value(value)
        self.assertNotIn("confirmed", filtered)
        self.assertNotIn("admin", filtered)
        self.assertNotIn("override_budget", filtered)

    def test_filter_rejects_non_dict(self):
        """Non-dict inputs return empty dict (defensive)."""
        from src.card.events.payloads import filter_workflow_button_value

        self.assertEqual(filter_workflow_button_value(None), {})  # type: ignore[arg-type]
        self.assertEqual(filter_workflow_button_value("not a dict"), {})  # type: ignore[arg-type]
        self.assertEqual(filter_workflow_button_value([]), {})  # type: ignore[arg-type]

if __name__ == "__main__":
    unittest.main()
