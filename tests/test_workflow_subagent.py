"""Behavioral tests for workflow prompt construction and bridge limits.

Validates:
- Role prefix is injected when params.role is set
"""

import unittest

from src.workflow_engine.script_gen import (
    _get_agent_capability_note,
    build_script_gen_prompt,
    extract_meta_from_script,
    generate_simple_script,
    validate_generated_script,
)


class TestBuildPromptInjection(unittest.TestCase):
    """Test AgentExecutor._build_prompt injects encouragement."""

    def setUp(self):
        self._executors = []

    def tearDown(self):
        for executor in self._executors:
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass
        self._executors.clear()

    def _make_executor(self):
        import threading

        from src.workflow_engine.executor import AgentExecutor

        executor = AgentExecutor(
            cwd="/tmp",
            cancel_event=threading.Event(),
            on_token_usage=None,
        )
        self._executors.append(executor)
        return executor

    def _make_params(self, prompt="do something", role=""):
        from src.workflow_engine.models import AgentCallParams

        return AgentCallParams(
            prompt=prompt,
            tool="coco",
            role=role,
        )

    def test_role_prefix_injected(self):
        executor = self._make_executor()
        params = self._make_params(prompt="task", role="security_auditor")
        result = executor._build_prompt(params)

        self.assertTrue(result.startswith("Role: security_auditor"))

class TestBridgeArgsPassthrough(unittest.TestCase):
    """Test RuntimeBridge args parameter and passthrough."""

    def setUp(self):
        self._bridges = []

    def tearDown(self):
        for bridge in self._bridges:
            try:
                bridge.stop()
            except Exception:
                pass
        self._bridges.clear()

    def _make_bridge(self, **kwargs):
        from src.workflow_engine.bridge import RuntimeBridge

        defaults = {"script_path": "/tmp/test.js", "cwd": "/tmp"}
        defaults.update(kwargs)
        bridge = RuntimeBridge(**defaults)
        self._bridges.append(bridge)
        return bridge

    def test_bridge_stores_args(self):
        bridge = self._make_bridge(args={"key": "value", "num": 42})
        self.assertEqual(bridge._args, {"key": "value", "num": 42})

    def test_bridge_default_args_empty_dict(self):
        bridge = self._make_bridge()
        self.assertEqual(bridge._args, {})

    def test_bridge_none_args_becomes_empty_dict(self):
        bridge = self._make_bridge(args=None)
        self.assertEqual(bridge._args, {})


class TestBuildScriptGenPromptInjection(unittest.TestCase):
    """Test that build_script_gen_prompt injects SUBAGENT_ENCOURAGEMENT."""

    def test_prompt_contains_requirement(self):
        prompt = build_script_gen_prompt(
            requirement="My unique test requirement xyz123",
            available_tools=["coco"],
        )
        self.assertIn("My unique test requirement xyz123", prompt)

    def test_prompt_contains_tools_list(self):
        prompt = build_script_gen_prompt(
            requirement="Test",
            available_tools=["coco", "claude", "aiden"],
        )
        self.assertIn("`coco`", prompt)
        self.assertIn("`claude`", prompt)
        self.assertIn("`aiden`", prompt)

    def test_prompt_contains_roles_list(self):
        prompt = build_script_gen_prompt(
            requirement="Test",
            available_tools=[],
        )
        self.assertIn("architect", prompt.lower())

    def test_prompt_contains_budget(self):
        prompt = build_script_gen_prompt(
            requirement="Test",
            available_tools=[],
        )
        self.assertNotIn("预算", prompt)

class TestGenerateSimpleScriptEncouragement(unittest.TestCase):
    """Test that generate_simple_script includes SUBAGENT_ENCOURAGEMENT in agent prompts."""

    def test_generate_simple_script_returns_card_summary_envelope(self):
        script = generate_simple_script("Implement and verify a focused change")

        self.assertIn("function completionEnvelope", script)
        self.assertIn("card_summary", script)
        self.assertIn("needs_attention", script)
        self.assertIn("任务已完成，完整结果见报告。", script)

    def test_script_has_valid_structure(self):
        script = generate_simple_script("Test requirement")
        self.assertIn("export const meta", script)
        self.assertIn("export default async function", script)
        self.assertIn("agent(", script)

    def test_script_avoids_slow_static_analysis_agent(self):
        script = generate_simple_script("Fix workflow state mismatch")
        self.assertNotIn('label: "task-analysis"', script)
        self.assertNotIn("Analyze this task and determine the best execution strategy", script)

    def test_script_bounds_agent_calls_and_handles_errors(self):
        script = generate_simple_script("Fix workflow state mismatch", selected_tools=["coco", "codex"])
        self.assertIn("timeout:", script)
        self.assertIn(".error", script)
        self.assertIn("fallback", script.lower())

        is_valid, messages = validate_generated_script(script)
        self.assertTrue(is_valid, f"Expected valid fallback script, got: {messages}")

    def test_simple_script_uses_one_backend_without_race_or_llm_routing(self):
        script = generate_simple_script(
            "分析 spec 模式目标完成度如何把控，先不要动手改代码",
            selected_tools=["traex", "codex", "coco"],
        )

        self.assertNotIn("race(", script)
        self.assertNotIn("candidateTools", script)
        self.assertNotIn("await classify(", script)
        self.assertNotIn('label: "route"', script)
        self.assertNotIn("route-classify", script)

        is_valid, messages = validate_generated_script(script)
        self.assertTrue(is_valid, f"Expected valid fallback script, got: {messages}")

    def test_script_prompt_preserves_analysis_only_requests(self):
        script = generate_simple_script(
            "分析 spec 模式目标完成度如何把控，先不要动手改代码",
            selected_tools=["traex", "codex"],
        )

        self.assertIn("If the user asks for analysis only", script)
        self.assertIn("do not change code", script)

    def test_meta_parser_preserves_apostrophes_inside_double_quoted_strings(self):
        script = '''
export const meta = {
  name: "reviewer's workflow",
  description: "don't rewrite 'quoted' content",
  phases: [{ title: "It's safe", detail: "owner's evidence" }],
};
export default async function main() { return {}; }
'''

        meta = extract_meta_from_script(script)

        self.assertIsNotNone(meta)
        self.assertEqual(meta["name"], "reviewer's workflow")
        self.assertEqual(meta["description"], "don't rewrite 'quoted' content")
        self.assertEqual(meta["phases"][0]["title"], "It's safe")


class TestAgentCapabilityNotes(unittest.TestCase):
    """Test _get_agent_capability_note returns correct notes for each agent type."""

    def test_agent_capability_coco(self):
        """Coco agent should have subagent and parallel orchestration notes."""
        note = _get_agent_capability_note("coco")
        self.assertIn("全栈编程", note)
        self.assertIn("subagent", note)
        self.assertIn("并行编排", note)

    def test_agent_capability_claude(self):
        """Claude agent should have deep reasoning notes."""
        note = _get_agent_capability_note("claude")
        self.assertIn("深度推理", note)
        self.assertIn("逻辑严谨性", note)

    def test_agent_capability_aiden(self):
        """Aiden agent should have code review notes."""
        note = _get_agent_capability_note("aiden")
        self.assertIn("代码审查", note)
        self.assertIn("架构设计", note)

    def test_agent_capability_codex(self):
        """Codex agent should have fast code generation notes."""
        note = _get_agent_capability_note("codex")
        self.assertIn("快速代码生成", note)
        self.assertIn("简洁直接", note)

    def test_agent_capability_gemini(self):
        """Gemini agent should have multi-modal notes."""
        note = _get_agent_capability_note("gemini")
        self.assertIn("多模态", note)
        self.assertIn("图像", note)

    def test_agent_capability_traex(self):
        """Traex agent should have high concurrency notes."""
        note = _get_agent_capability_note("traex")
        self.assertIn("高并发", note)
        self.assertIn("轻量任务", note)

    def test_unknown_agent_defaults_to_coco(self):
        """Unknown agent type should default to coco capability notes."""
        note_unknown = _get_agent_capability_note("unknown_agent")
        note_coco = _get_agent_capability_note("coco")
        self.assertEqual(note_unknown, note_coco)

    def test_different_agents_produce_different_notes(self):
        """Different agent types should produce different capability notes."""
        note_coco = _get_agent_capability_note("coco")
        note_claude = _get_agent_capability_note("claude")
        note_aiden = _get_agent_capability_note("aiden")
        self.assertNotEqual(note_coco, note_claude)
        self.assertNotEqual(note_coco, note_aiden)
        self.assertNotEqual(note_claude, note_aiden)


if __name__ == "__main__":
    unittest.main()
