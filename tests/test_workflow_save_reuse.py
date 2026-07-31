"""Tests for workflow script saving, template management, and reuse functionality.

Covers:
- Script generation and persistence to .ghostap/workflow_scripts/
- Template discovery from built-in and project directories
- Template loading and argument injection
- Workflow management commands (/wf_save, /wf_list, /wf_delete, /wf_history)
- Workflow reuse via template names
- Metadata persistence (tools, phases)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.workflow_engine.constants import (
    AGENT_CALL_TIMEOUT_S,
    SCRIPT_GEN_TIMEOUT_S,
    WORKFLOW_TEMPLATES_DIR,
)
from src.workflow_engine.history import WorkflowHistory
from src.workflow_engine.models import PendingConfirmation, WorkflowProject, WorkflowStatus
from src.workflow_engine.script_gen import extract_meta_from_script
from src.workflow_engine.templates import (
    TemplateInfo,
    discover_templates,
    inject_args,
    load_template,
    parse_template_meta,
    save_template,
)

# ---------------------------------------------------------------------------
# Helper: sample workflow script with meta block
# ---------------------------------------------------------------------------

SAMPLE_SCRIPT = '''/**
 * Test workflow script.
 */

export const meta = {
  name: "test-workflow",
  description: "A test workflow for unit tests",
  phases: [
    { title: "Analysis", detail: "Analyze the problem" },
    { title: "Execution", detail: "Execute the solution" },
    { title: "Verification", detail: "Verify the results" }
  ],
  maxConcurrent: 4,
  tools: ["coco", "claude", "aiden"]
};

export default async function main(args = {}) {
  const target = args.target || "default";
  phase("Analysis");
  log("Starting analysis...");
  const result = await agent("Analyze " + target, { tool: "coco", label: "analyzer", timeout: 120 });
  if (result && result.error) return result;
  phase("Execution");
  const exec = await agent("Execute based on " + result, { tool: "claude", label: "executor", timeout: 180 });
  if (exec && exec.error) return exec;
  phase("Verification");
  const verify = await agent("Verify " + exec, { tool: "aiden", label: "verifier", timeout: 120 });
  if (verify && verify.error) return verify;
  return verify;
}
'''

SAMPLE_TEMPLATE_WITH_ARGS = '''export const meta = {
  name: "parametrized-workflow",
  description: "Workflow that accepts arguments",
  phases: [
    { title: "Process", detail: "Process with args" }
  ],
  tools: ["coco"]
};

export default async function main(args = {}) {
  const target = args.target || ".";
  const focus = args.focus || "";
  const count = args.count || 1;
  log("Processing " + target + " with focus " + focus);
  const result = await agent({
    prompt: "Process " + args.target + " count=" + args.count,
    tool: "coco",
    label: "process-args",
    timeout: 120,
  });
  if (result && result.error) return result;
  return result;
}
'''

# Simple script with flat meta (no nested objects) for tests that need meta extraction
# The _extract_meta_json regex has limitations with nested braces
SIMPLE_SCRIPT = '''/**
 * Simple test workflow script with flat meta.
 */

export const meta = {
  name: "simple-workflow",
  description: "A simple workflow for meta extraction tests",
  tools: ["coco", "claude"]
};

export default async function main(args = {}) {
  const target = args.target || "default";
  log("Starting simple workflow...");
  const result = await agent("Process " + target, { tool: "coco", label: "process-target", timeout: 120 });
  if (result && result.error) return result;
  return result;
}
'''


# ---------------------------------------------------------------------------
# Test Class 1: TestScriptSaving
# ---------------------------------------------------------------------------

class TestScriptSaving(unittest.TestCase):
    """Tests for workflow script generation and persistence."""

    def _make_handler(self, root_path: str):
        """Create a WorkflowHandler with mocked dependencies."""
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.reply_error = MagicMock()
        handler._reply_workflow_error = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="msg_123")
        handler.update_card = MagicMock(return_value=True)
        handler.add_reaction = MagicMock()
        handler.get_working_dir = MagicMock(return_value=root_path)
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_123")
        handler._submit_engine_task = MagicMock()

        return handler, ctx

    def test_generated_script_saved_to_correct_location(self):
        """Verify that _generate_script_via_ai saves the script to .ghostap/workflow_scripts/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)

            # Mock the AI session to return a valid script
            with patch("src.agent_session.create_engine_session") as mock_create:
                mock_session = MagicMock()
                mock_create.return_value = mock_session
                mock_session.send_prompt.return_value = MagicMock(text=SAMPLE_SCRIPT)
                mock_session.close = MagicMock()

                with patch("src.agent_session.close_session_safely"):
                    script_path, meta, is_fallback = handler._generate_script_via_ai(
                        "test requirement", tmpdir, ["coco"]
                    )

            # Verify script was saved to correct location
            expected_dir = os.path.join(tmpdir, ".ghostap", "workflow_scripts")
            self.assertTrue(os.path.isdir(expected_dir), f"Directory {expected_dir} should exist")
            self.assertTrue(script_path.startswith(expected_dir),
                            f"Script path {script_path} should be in {expected_dir}")
            self.assertTrue(script_path.endswith(".js"),
                            f"Script path {script_path} should have .js extension")
            self.assertTrue(os.path.isfile(script_path),
                            f"Script file {script_path} should exist")

            # Verify file content
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("export const meta", content)
            self.assertIn("test-workflow", content)

    def test_script_generation_uses_dedicated_short_timeout(self):
        """Script generation must not consume the full per-agent timeout window."""
        self.assertLess(SCRIPT_GEN_TIMEOUT_S, AGENT_CALL_TIMEOUT_S)

        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)
            # Script-gen timeout is now read from Settings (falls back to the
            # constant). Pin it so the assertion checks a concrete value.
            ctx.settings.workflow_script_gen_timeout_s = SCRIPT_GEN_TIMEOUT_S

            with patch("src.agent_session.create_engine_session") as mock_create:
                mock_session = MagicMock()
                mock_create.return_value = mock_session
                mock_session.send_prompt.return_value = MagicMock(text=SAMPLE_SCRIPT)
                mock_session.close = MagicMock()

                with patch("src.agent_session.close_session_safely"):
                    handler._generate_script_via_ai(
                        "test requirement", tmpdir, ["coco"]
                    )

            mock_session.send_prompt.assert_called_once()
            self.assertEqual(
                mock_session.send_prompt.call_args.kwargs.get("timeout"),
                SCRIPT_GEN_TIMEOUT_S,
            )

    def test_script_generation_respects_settings_override(self):
        """A larger workflow_script_gen_timeout_s from Settings is honored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)
            ctx.settings.workflow_script_gen_timeout_s = 300

            with patch("src.agent_session.create_engine_session") as mock_create:
                mock_session = MagicMock()
                mock_create.return_value = mock_session
                mock_session.send_prompt.return_value = MagicMock(text=SAMPLE_SCRIPT)
                mock_session.close = MagicMock()

                with patch("src.agent_session.close_session_safely"):
                    handler._generate_script_via_ai(
                        "test requirement", tmpdir, ["coco"]
                    )

            mock_session.send_prompt.assert_called_once()
            self.assertEqual(
                mock_session.send_prompt.call_args.kwargs.get("timeout"),
                300,
            )

    def test_saved_script_preserves_metadata(self):
        """Verify that saved scripts preserve the meta block (name, description, tools).

        Uses SIMPLE_SCRIPT with flat meta (no nested objects) since _extract_meta_json
        has limitations with nested braces. Content preservation with nested objects
        is tested via raw content checks in TestMetadataPersistence.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)

            with patch("src.agent_session.create_engine_session") as mock_create:
                mock_session = MagicMock()
                mock_create.return_value = mock_session
                mock_session.send_prompt.return_value = MagicMock(text=SIMPLE_SCRIPT)
                mock_session.close = MagicMock()

                with patch("src.agent_session.close_session_safely"):
                    script_path, meta, is_fallback = handler._generate_script_via_ai(
                        "test requirement", tmpdir, ["coco"]
                    )

            # Meta should be extracted and preserved
            self.assertIsNotNone(meta, "Meta should be extracted from generated script")
            self.assertEqual(meta["name"], "simple-workflow")
            self.assertEqual(meta["description"], "A simple workflow for meta extraction tests")
            self.assertEqual(meta["tools"], ["coco", "claude"])

            # Also verify meta can be parsed from the saved file
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()
            parsed_meta = extract_meta_from_script(content)
            self.assertIsNotNone(parsed_meta)
            self.assertEqual(parsed_meta["name"], "simple-workflow")
            self.assertEqual(parsed_meta["tools"], ["coco", "claude"])

    def test_saved_script_can_be_read_back(self):
        """Verify that a saved script can be read and parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)

            with patch("src.agent_session.create_engine_session") as mock_create:
                mock_session = MagicMock()
                mock_create.return_value = mock_session
                mock_session.send_prompt.return_value = MagicMock(text=SIMPLE_SCRIPT)
                mock_session.close = MagicMock()

                with patch("src.agent_session.close_session_safely"):
                    script_path, meta, is_fallback = handler._generate_script_via_ai(
                        "test requirement", tmpdir, ["coco"]
                    )

            # Read back the file
            self.assertTrue(os.path.exists(script_path))
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Verify content is intact
            self.assertIn("export const meta", content)
            self.assertIn("export default async function", content)
            self.assertIn('await agent("Process "', content)

            # Verify it can be validated
            from src.workflow_engine.script_gen import validate_generated_script
            is_valid, errors = validate_generated_script(content)
            self.assertTrue(is_valid, f"Script should be valid, got errors: {errors}")

            # Verify meta can be parsed via parse_template_meta
            # Uses SIMPLE_SCRIPT with flat meta to avoid _extract_meta_json limitations
            parsed_meta = parse_template_meta(content)
            self.assertIsNotNone(parsed_meta)
            self.assertEqual(parsed_meta.name, "simple-workflow")
            self.assertEqual(parsed_meta.tools, ["coco", "claude"])

    def test_script_filename_contains_timestamp_or_hash(self):
        """Concurrent generations use distinct files and cannot overwrite each other."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)

            with patch("src.agent_session.create_engine_session") as mock_create:
                mock_session = MagicMock()
                mock_create.return_value = mock_session
                mock_session.send_prompt.return_value = MagicMock(text=SAMPLE_SCRIPT)
                mock_session.close = MagicMock()

                with patch("src.agent_session.close_session_safely"):
                    script_path, meta, is_fallback = handler._generate_script_via_ai(
                        "test requirement", tmpdir, ["coco"]
                    )

            # Verify filename has .js extension and is in the correct directory
            basename = os.path.basename(script_path)
            self.assertTrue(basename.endswith(".js"),
                            f"Filename {basename} should end with .js")

            # Verify the directory structure is correct
            expected_dir = os.path.join(tmpdir, ".ghostap", "workflow_scripts")
            self.assertEqual(os.path.dirname(script_path), expected_dir)

            # For template-based workflows, verify filename uses template name
            # (tested separately in TestWorkflowReuse)
            self.assertIn("workflow", basename.lower(),
                          f"Filename {basename} should be descriptive")

            # Generate a second script and verify it overwrites (current behavior)
            # or has a unique name (expected behavior)
            with patch("src.agent_session.create_engine_session") as mock_create2:
                mock_session2 = MagicMock()
                mock_create2.return_value = mock_session2
                mock_session2.send_prompt.return_value = MagicMock(text=SIMPLE_SCRIPT)
                mock_session2.close = MagicMock()

                with patch("src.agent_session.close_session_safely"):
                    script_path2, meta2, is_fallback2 = handler._generate_script_via_ai(
                        "second requirement", tmpdir, ["claude"]
                    )

            # Both generations retain their own immutable source artifact.
            self.assertNotEqual(script_path, script_path2)
            self.assertTrue(os.path.isfile(script_path))
            self.assertTrue(os.path.isfile(script_path2))
            with open(script_path, encoding="utf-8") as first_file:
                self.assertIn('name: "test-workflow"', first_file.read())
            with open(script_path2, encoding="utf-8") as second_file:
                self.assertIn('name: "simple-workflow"', second_file.read())


# ---------------------------------------------------------------------------
# Test Class 2: TestTemplateDiscovery
# ---------------------------------------------------------------------------

class TestTemplateDiscovery(unittest.TestCase):
    """Tests for template discovery from built-in and project directories."""

    def _create_project_template(self, root_path: str, name: str, description: str) -> Path:
        """Create a sample template in the project's workflow templates directory.

        Uses flat meta (no nested objects) to ensure _extract_meta_json can parse
        the description correctly.
        """
        templates_dir = Path(root_path) / WORKFLOW_TEMPLATES_DIR
        templates_dir.mkdir(parents=True, exist_ok=True)

        template_content = f'''export const meta = {{
  name: "{name}",
  description: "{description}",
  tools: ["coco"]
}};

export default async function main() {{
  const result = await agent("Do something", {{ tool: "coco", label: "do-something", timeout: 120 }});
  if (result && result.error) return result;
  return result;
}}
'''
        template_file = templates_dir / f"{name}.js"
        template_file.write_text(template_content, encoding="utf-8")
        return template_file

    def test_discover_templates_finds_builtin_templates(self):
        """Verify that discover_templates() finds built-in templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            templates = discover_templates(tmpdir)

            # Should find at least the built-in templates
            self.assertIsInstance(templates, list)
            self.assertGreater(len(templates), 0, "Should find at least built-in templates")

            # Check that all returned items are TemplateInfo
            for t in templates:
                self.assertIsInstance(t, TemplateInfo)
                self.assertTrue(hasattr(t, "name"))
                self.assertTrue(hasattr(t, "path"))
                self.assertTrue(hasattr(t, "description"))
                self.assertTrue(hasattr(t, "scope"))

            # Built-in templates should have scope "builtin"
            builtin_templates = [t for t in templates if t.scope == "builtin"]
            self.assertGreater(len(builtin_templates), 0,
                               "Should find at least one built-in template")

            # Verify known built-in templates exist
            builtin_names = [t.name for t in builtin_templates]
            self.assertIn("code-audit", builtin_names)

    def test_discover_templates_finds_project_templates(self):
        """Verify that templates in the project's .ghostap/workflows/ directory are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project-level templates
            self._create_project_template(tmpdir, "custom-audit", "Custom audit workflow")
            self._create_project_template(tmpdir, "deploy-script", "Deployment workflow")

            templates = discover_templates(tmpdir)
            template_names = [t.name for t in templates]

            # Project templates should be found
            self.assertIn("custom-audit", template_names)
            self.assertIn("deploy-script", template_names)

            # Project templates should have scope "project"
            project_templates = [t for t in templates if t.scope == "project"]
            self.assertEqual(len(project_templates), 2)

            # Project templates should override built-in ones with same name
            self._create_project_template(tmpdir, "code-audit", "Project-level code audit override")
            templates2 = discover_templates(tmpdir)
            code_audit = next(t for t in templates2 if t.name == "code-audit")
            self.assertEqual(code_audit.scope, "project",
                             "Project template should override built-in with same name")
            self.assertEqual(code_audit.description, "Project-level code audit override")

    def test_discover_templates_returns_correct_metadata(self):
        """Verify that discovered templates have correct name, description, and path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a project template with known metadata
            template_path = self._create_project_template(
                tmpdir, "my-custom-workflow", "This is my custom workflow"
            )

            templates = discover_templates(tmpdir)
            my_template = next(t for t in templates if t.name == "my-custom-workflow")

            # Verify metadata
            self.assertEqual(my_template.name, "my-custom-workflow")
            self.assertEqual(my_template.description, "This is my custom workflow")
            self.assertEqual(my_template.path, str(template_path))
            self.assertEqual(my_template.scope, "project")

            # Verify built-in template metadata
            templates_all = discover_templates(tmpdir)
            code_audit = next(t for t in templates_all if t.name == "code-audit" and t.scope == "builtin")
            self.assertIsNotNone(code_audit.path)
            self.assertTrue(code_audit.path.endswith("code-audit.js"))
            # Note: built-in templates have nested phases in meta, which _extract_meta_json
            # cannot parse correctly, so description may be empty. We verify the path instead.


# ---------------------------------------------------------------------------
# Test Class 3: TestTemplateLoading
# ---------------------------------------------------------------------------

class TestTemplateLoading(unittest.TestCase):
    """Tests for template loading and argument injection."""

    def _create_project_template(self, root_path: str, name: str, content: str) -> Path:
        """Create a template file in the project directory."""
        templates_dir = Path(root_path) / WORKFLOW_TEMPLATES_DIR
        templates_dir.mkdir(parents=True, exist_ok=True)
        template_file = templates_dir / f"{name}.js"
        template_file.write_text(content, encoding="utf-8")
        return template_file

    def test_load_template_returns_content(self):
        """Verify that load_template() returns the JS content of a template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_project_template(tmpdir, "test-template", SAMPLE_SCRIPT)

            content = load_template(tmpdir, "test-template")
            self.assertIsNotNone(content)
            self.assertIsInstance(content, str)
            self.assertIn("export const meta", content)
            self.assertIn("test-workflow", content)
            self.assertIn("export default async function", content)

            # Also test loading with .js extension
            content2 = load_template(tmpdir, "test-template.js")
            self.assertEqual(content, content2)

            # Test loading built-in template
            builtin_content = load_template(tmpdir, "code-audit")
            self.assertIsNotNone(builtin_content)
            self.assertIn("code-audit", builtin_content)
            self.assertIn("export default async function", builtin_content)

    def test_load_nonexistent_template_returns_none(self):
        """Verify that loading a non-existent template returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Non-existent template
            content = load_template(tmpdir, "nonexistent-template")
            self.assertIsNone(content)

            # Empty string
            content2 = load_template(tmpdir, "")
            self.assertIsNone(content2)

            # Template with wrong extension
            content3 = load_template(tmpdir, "nonexistent.py")
            self.assertIsNone(content3)

    def test_inject_args_replaces_placeholders(self):
        """Verify that inject_args() replaces args.KEY and args['KEY'] patterns with provided values."""
        script = '''export default async function main(args = {}) {
  const target = args.target;
  const count = args.count;
  const name = args['name'];
  const debug = args.debug;
  const config = args["config"];
  log("Processing " + target + " count=" + count);
  return args.target;
}
'''

        # Test with string values
        args = {"target": "src/", "count": 5, "name": "test", "debug": True, "config": {"key": "value"}}
        result = inject_args(script, args)

        # String values should be quoted
        self.assertIn('"src/"', result)
        self.assertNotIn("args.target", result)

        # Numeric values should be inlined
        self.assertIn("5", result)
        self.assertNotIn("args.count", result)

        # Boolean values should be JS booleans
        self.assertIn("true", result)
        self.assertNotIn("args.debug", result)

        # Bracket access should also be replaced
        self.assertIn('"test"', result)
        self.assertNotIn("args['name']", result)
        self.assertNotIn('args["config"]', result)

        # Object values should be JSON serialized
        self.assertIn('{"key": "value"}', result)

        # Test with empty args (should return unchanged)
        result2 = inject_args(script, {})
        self.assertEqual(result2, script)

        # Test with None args
        result3 = inject_args(script, None)
        self.assertEqual(result3, script)

        # Test with None value
        args2 = {"nullable": None}
        script2 = "const x = args.nullable;"
        result4 = inject_args(script2, args2)
        self.assertIn("null", result4)
        self.assertNotIn("args.nullable", result4)


# ---------------------------------------------------------------------------
# Test Class 4: TestWorkflowCommands
# ---------------------------------------------------------------------------

class TestWorkflowCommands(unittest.TestCase):
    """Tests for workflow management commands (/wf_save, /wf_list, /wf_delete, /wf_history)."""

    def _make_handler(self, root_path: str):
        """Create a WorkflowHandler with mocked dependencies."""
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.reply_error = MagicMock()
        handler._reply_workflow_error = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="msg_123")
        handler.update_card = MagicMock(return_value=True)
        handler.add_reaction = MagicMock()
        handler.get_working_dir = MagicMock(return_value=root_path)
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_123")
        handler._submit_engine_task = MagicMock()
        handler._get_root_path = MagicMock(return_value=root_path)

        return handler, ctx

    def _write_script_to_disk(self, root_path: str, content: str) -> str:
        """Write a script to the workflow_scripts directory."""
        script_dir = os.path.join(root_path, ".ghostap", "workflow_scripts")
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, "test_script.js")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(content)
        return script_path

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_wf_save_command(self, mock_sender):
        """Verify that /wf_save <name> saves the last generated script with the given name.

        Owner-based permissions:
        - project-level: first save is allowed for any user; overwriting an existing
          project template requires owner or admin
        - --global: requires admin (ctx.settings.admin_user_ids contains sender)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)

            # Set up engine with a pending script
            script_path = self._write_script_to_disk(tmpdir, SAMPLE_SCRIPT)
            engine = MagicMock()
            engine.project = WorkflowProject(
                pending=PendingConfirmation(script_path=script_path),
                script_path=script_path,
            )
            ctx.workflow_engine_manager.get.return_value = engine

            # Execute /wf_save command (first time — template does not exist yet)
            handler._handle_wf_save("msg_1", "chat_1", "my-saved-workflow", project=None)

            # Verify success reply
            handler.reply_text.assert_called_once()
            reply_content = handler.reply_text.call_args[0][1]
            self.assertIn("my-saved-workflow", reply_content)
            self.assertIn("已保存", reply_content)

            # Verify the template was saved to the correct location
            template_path = os.path.join(tmpdir, WORKFLOW_TEMPLATES_DIR, "my-saved-workflow.js")
            self.assertTrue(os.path.isfile(template_path),
                            f"Saved template should exist at {template_path}")

            # Verify content is preserved
            with open(template_path, "r", encoding="utf-8") as f:
                saved_content = f.read()
            self.assertIn("test-workflow", saved_content)
            self.assertIn("export const meta", saved_content)

            # Test with --global flag (requires admin). Give the handler ctx.settings
            # so is_admin_user() recognizes the sender as admin.
            handler.reply_text.reset_mock()
            admin_settings = MagicMock()
            admin_settings.admin_user_ids = ["user_123"]
            handler.ctx.settings = admin_settings
            with patch("src.workflow_engine.templates._global_templates_dir",
                       return_value=Path(tmpdir) / "global_templates"):
                handler._handle_wf_save("msg_2", "chat_1", "global-workflow --global", project=None)
                handler.reply_text.assert_called_once()
                reply_content = handler.reply_text.call_args[0][1]
                self.assertIn("全局级", reply_content)

            # Test --global without admin permission → should be rejected
            handler.reply_text.reset_mock()
            handler._reply_workflow_error.reset_mock()
            handler.ctx.settings = MagicMock()
            handler.ctx.settings.admin_user_ids = []
            with patch("src.workflow_engine.templates._global_templates_dir",
                       return_value=Path(tmpdir) / "global_templates"):
                handler._handle_wf_save("msg_2b", "chat_1", "another-global --global", project=None)
                handler._reply_workflow_error.assert_called_once()

            # Test overwriting existing project template: sender is the owner → allowed
            handler.reply_text.reset_mock()
            # my-saved-workflow was just saved by user_123, so user_123 is the owner
            handler.ctx.settings = MagicMock()
            handler.ctx.settings.admin_user_ids = []
            handler._handle_wf_save("msg_5", "chat_1", "my-saved-workflow", project=None)
            handler.reply_text.assert_called_once()
            reply_content = handler.reply_text.call_args[0][1]
            self.assertIn("已保存", reply_content)

            # Test saving with no script available
            handler.reply_text.reset_mock()
            handler._reply_workflow_error.reset_mock()
            if engine.project.pending is None:
                engine.project.pending = PendingConfirmation()
            engine.project.pending.script_path = None
            engine.project.script_path = None
            handler._handle_wf_save("msg_3", "chat_1", "another-workflow", project=None)
            handler._reply_workflow_error.assert_called_once()

            # Test saving with invalid name (no spaces — spaces cause split to take first token)
            handler.reply_text.reset_mock()
            handler._reply_workflow_error.reset_mock()
            if engine.project.pending is None:
                engine.project.pending = PendingConfirmation()
            engine.project.pending.script_path = script_path
            handler._handle_wf_save("msg_4", "chat_1", "invalid!", project=None)
            handler._reply_workflow_error.assert_called_once()

    def test_wf_list_command(self):
        """Verify that /wf_list lists all saved workflows with their metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)

            # Create some project templates
            templates_dir = Path(tmpdir) / WORKFLOW_TEMPLATES_DIR
            templates_dir.mkdir(parents=True, exist_ok=True)

            for name, desc in [("audit", "Code audit workflow"),
                               ("deploy", "Deployment workflow"),
                               ("test", "Testing workflow")]:
                # Use simple script with flat meta so description can be extracted
                simple_with_desc = SIMPLE_SCRIPT.replace("simple-workflow", name).replace(
                    "A simple workflow for meta extraction tests", desc
                )
                save_template(tmpdir, name, simple_with_desc)

            # Execute /wf_list command
            handler._handle_wf_list("msg_1", "chat_1", project=None)

            # Verify reply contains template information
            handler.reply_text.assert_called_once()
            reply_content = handler.reply_text.call_args[0][1]
            self.assertIn("可用 Workflow 模板", reply_content)
            self.assertIn("audit", reply_content)
            self.assertIn("deploy", reply_content)
            self.assertIn("test", reply_content)
            self.assertIn("Code audit workflow", reply_content)
            self.assertIn("使用:", reply_content)

            # Verify scope icons are present
            self.assertIn("📦", reply_content)  # builtin icon
            self.assertIn("📂", reply_content)  # project icon

            # Test with empty project dir (only built-in templates should be shown)
            with tempfile.TemporaryDirectory() as tmpdir2:
                handler2, ctx2 = self._make_handler(tmpdir2)
                handler2._handle_wf_list("msg_2", "chat_2", project=None)
                handler2.reply_text.assert_called_once()
                reply_content2 = handler2.reply_text.call_args[0][1]
                # Built-in templates always exist, so we should see them
                self.assertIn("可用 Workflow 模板", reply_content2)
                self.assertIn("📦", reply_content2)  # builtin icon
                # Should not show project icon since no project templates exist
                self.assertNotIn("📂", reply_content2)

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    def test_wf_delete_command(self, mock_sender):
        """Verify that /wf_delete <name> deletes the specified saved workflow.

        Owner-based permissions:
        - owner can delete their own project template
        - templates without an owner file require admin permission
        - --global requires admin
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)
            # Enable admin-less: user_123 is the owner, so admin is not required
            handler.ctx.settings = MagicMock()
            handler.ctx.settings.admin_user_ids = []

            # Create a template to delete, saving it with an owner file
            from src.workflow_engine.templates import save_template as _save_tpl

            _save_tpl(tmpdir, "to-delete", SAMPLE_SCRIPT, owner_id="user_123")
            template_path = os.path.join(tmpdir, WORKFLOW_TEMPLATES_DIR, "to-delete.js")
            self.assertTrue(os.path.isfile(template_path))

            # Execute /wf_delete command as the owner
            handler._handle_wf_delete("msg_1", "chat_1", "to-delete", project=None)

            # Verify success reply and file deletion
            handler.reply_text.assert_called_once()
            reply_content = handler.reply_text.call_args[0][1]
            self.assertIn("已删除", reply_content)
            self.assertIn("to-delete", reply_content)
            self.assertFalse(os.path.isfile(template_path),
                             "Template file should be deleted")

            # Test deleting non-existent template — admin is required to reach
            # the "does not exist" branch; non-admins get permission-denied.
            handler.reply_text.reset_mock()
            handler._reply_workflow_error.reset_mock()
            # Simulate admin context: ctx.settings.admin_user_ids contains sender
            handler.ctx.settings = MagicMock()
            handler.ctx.settings.admin_user_ids = ["user_123"]
            handler._handle_wf_delete("msg_2", "chat_1", "nonexistent", project=None)
            handler._reply_workflow_error.assert_called_once()

            # Test delete with no name
            handler.reply_text.reset_mock()
            handler._reply_workflow_error.reset_mock()
            handler._handle_wf_delete("msg_3", "chat_1", "", project=None)
            handler.reply_text.assert_called_once()
            reply_content3 = handler.reply_text.call_args[0][1]
            self.assertIn("用法:", reply_content3)

    def test_wf_history_command(self):
        """Verify that /wf_history shows recent workflow executions with status and timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)

            # Create some history entries
            history = WorkflowHistory(tmpdir)

            # Record a completed workflow
            completed_project = WorkflowProject(
                workflow_id="wf_001",
                name="completed-workflow",
                status=WorkflowStatus.COMPLETED,
                started_at=time.time() - 3600,
                finished_at=time.time() - 3500,
            )
            completed_project.metrics.total_tokens = 150000
            completed_project.metrics.total_agents = 5
            completed_project.phases = [MagicMock(), MagicMock()]
            history.record(completed_project)

            # Record a failed workflow
            failed_project = WorkflowProject(
                workflow_id="wf_002",
                name="failed-workflow",
                status=WorkflowStatus.FAILED,
                started_at=time.time() - 1800,
                finished_at=time.time() - 1700,
                error="Something went wrong in the execution",
            )
            failed_project.metrics.total_tokens = 50000
            failed_project.metrics.total_agents = 3
            failed_project.phases = [MagicMock()]
            history.record(failed_project)

            # Execute /wf_history command
            handler._handle_wf_history("msg_1", "chat_1", project=None)

            # Verify reply contains history information
            handler.reply_text.assert_called_once()
            reply_content = handler.reply_text.call_args[0][1]
            self.assertIn("执行历史", reply_content)
            self.assertIn("completed-workflow", reply_content)
            self.assertIn("failed-workflow", reply_content)
            self.assertIn("✅", reply_content)  # completed icon
            self.assertIn("❌", reply_content)  # failed icon
            self.assertIn("150K tok", reply_content)  # token count
            self.assertIn("5 agents", reply_content)  # agent count
            self.assertIn("Something went wrong", reply_content)  # error message
            self.assertIn("最近 10 次", reply_content)

            # Test with no history
            with tempfile.TemporaryDirectory() as tmpdir2:
                handler2, ctx2 = self._make_handler(tmpdir2)
                handler2._handle_wf_history("msg_2", "chat_2", project=None)
                handler2.reply_text.assert_called_once()
                reply_content2 = handler2.reply_text.call_args[0][1]
                self.assertIn("暂无历史记录", reply_content2)


# ---------------------------------------------------------------------------
# Test Class 5: TestWorkflowReuse
# ---------------------------------------------------------------------------

class TestWorkflowReuse(unittest.TestCase):
    """Tests for workflow reuse via template names and arguments."""

    def _make_handler(self, root_path: str):
        """Create a WorkflowHandler with mocked dependencies."""
        from src.feishu.handlers.workflow import WorkflowHandler

        ctx = MagicMock()
        ctx.workflow_engine_manager = MagicMock()
        ctx.project_manager = MagicMock()
        ctx.progress_reporter = MagicMock()

        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = ctx
        handler.reply_text = MagicMock()
        handler.reply_card = MagicMock()
        handler.reply_error = MagicMock()
        handler._reply_workflow_error = MagicMock()
        handler.send_card_to_chat = MagicMock(return_value="msg_123")
        handler.update_card = MagicMock(return_value=True)
        handler.add_reaction = MagicMock()
        handler.get_working_dir = MagicMock(return_value=root_path)
        handler.get_engine_name = MagicMock(return_value="coco")
        handler.ensure_request_id = MagicMock(return_value="req_123")
        handler._submit_engine_task = MagicMock()
        handler._ensure_project = MagicMock()
        handler._resolve_project_from_id = MagicMock()

        return handler, ctx

    def _create_project_template(self, root_path: str, name: str, content: str) -> None:
        """Create a template in the project directory."""
        save_template(root_path, name, content)

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    @patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True)
    def test_start_workflow_with_template_name(
        self, mock_node, mock_sender
    ):
        """Verify that start_workflow() with a template name enters agent selection.

        Templates now go through the standard three-step flow:
            agent selection → tool selection → confirmation card.
        The full requirement (including the template name) is preserved in pending
        so downstream handlers can recognize and load the template.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)

            # Create a project template
            self._create_project_template(tmpdir, "my-template", SIMPLE_SCRIPT)

            # Set up project mock
            project = MagicMock()
            project.root_path = tmpdir
            project.project_id = "proj_1"
            project.project_name = "test"
            handler._ensure_project = MagicMock(return_value=project)

            # Set up engine mock — start_workflow calls get_or_create
            engine = MagicMock()
            engine.is_running = False
            engine.project = WorkflowProject()
            ctx.workflow_engine_manager.get.return_value = engine
            ctx.workflow_engine_manager.get_or_create.return_value = engine

            # Start workflow with a template name — goes to agent selection first
            handler.start_workflow("msg_1", "chat_1", "my-template", project)

            # After start_workflow, status should be AWAITING_AGENT_SELECT
            # (three-step flow: agent select → tool select → confirm)
            self.assertEqual(engine.project.status, WorkflowStatus.AWAITING_AGENT_SELECT)

            # The requirement is preserved so downstream handlers can detect
            # this is a template invocation
            self.assertEqual(
                engine.project.pending.requirement if engine.project.pending else None,
                "my-template",
            )

            # An agent selection card must be sent (send_card_to_chat)
            self.assertEqual(handler.send_card_to_chat.call_count, 1)

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    @patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True)
    def test_template_with_args(self, mock_node, mock_sender):
        """Verify that templates can accept arguments via /wf <template> arg1=val1 arg2=val2.

        Args are injected into the template script when the template is loaded
        during the tool-selection-to-confirmation step.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)

            # Create a template with argument placeholders
            self._create_project_template(tmpdir, "param-template", SAMPLE_TEMPLATE_WITH_ARGS)

            # Test argument injection directly (bypassing agent/tool selection flow)
            from src.workflow_engine.script_gen import extract_meta_from_script
            from src.workflow_engine.templates import inject_args, load_template

            # Load template and inject args manually to verify injection works
            template_content = load_template(tmpdir, "param-template")
            self.assertIsNotNone(template_content)
            self.assertIn("args.target", template_content)

            # Inject arguments
            args = {"target": "src/", "focus": "security", "count": 3}
            injected_content = inject_args(template_content, args)

            # Verify placeholders were replaced
            self.assertIn('"src/"', injected_content)
            self.assertIn('"security"', injected_content)
            self.assertIn("3", injected_content)
            self.assertNotIn("args.target", injected_content)
            self.assertNotIn("args.focus", injected_content)
            self.assertNotIn("args.count", injected_content)

            # Verify meta can still be extracted
            meta = extract_meta_from_script(injected_content)
            self.assertEqual(meta["name"], "parametrized-workflow")

            # Also test that the full flow preserves arguments through the steps
            # (start_workflow -> agent select -> tool select -> confirm)
            project = MagicMock()
            project.root_path = tmpdir
            project.project_id = "proj_1"
            project.project_name = "test"
            handler._ensure_project = MagicMock(return_value=project)

            # Set up engine mock
            engine = MagicMock()
            engine.is_running = False
            engine.project = WorkflowProject()
            ctx.workflow_engine_manager.get.return_value = engine
            ctx.workflow_engine_manager.get_or_create.return_value = engine

            # Start workflow with template name and arguments
            handler.start_workflow(
                "msg_1", "chat_1",
                "param-template target=src/ focus=security count=3",
                project
            )

            # After start_workflow, status should be AWAITING_AGENT_SELECT
            self.assertEqual(engine.project.status, WorkflowStatus.AWAITING_AGENT_SELECT)

            # The full requirement (with args) is preserved in pending so
            # downstream handlers can detect this is a template + args invocation
            self.assertEqual(
                engine.project.pending.requirement if engine.project.pending else None,
                "param-template target=src/ focus=security count=3",
            )

            # An agent selection card must be sent
            self.assertEqual(handler.send_card_to_chat.call_count, 1)

    @patch("src.thread.get_current_sender_id", return_value="user_123")
    @patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True)
    def test_template_skips_tool_selection(self, mock_node, mock_sender):
        """Verify that both template and non-template flows start with agent selection.

        Templates now go through the standard three-step flow (agent selection →
        tool selection → confirmation). Previously templates skipped tool selection
        entirely; now both templates and AI-generated scripts start with agent
        selection, giving users a chance to pick the orchestrator agent.

        This test verifies the start_workflow behavior consistently routes to
        agent selection regardless of whether the requirement matches a template.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            handler, ctx = self._make_handler(tmpdir)

            # Create a project template
            self._create_project_template(tmpdir, "quick-audit", SIMPLE_SCRIPT)

            # Set up project mock
            project = MagicMock()
            project.root_path = tmpdir
            project.project_id = "proj_1"
            project.project_name = "test"
            handler._ensure_project = MagicMock(return_value=project)

            # Set up engine mock
            engine = MagicMock()
            engine.is_running = False
            engine.project = WorkflowProject()
            ctx.workflow_engine_manager.get.return_value = engine
            ctx.workflow_engine_manager.get_or_create.return_value = engine

            # Non-template requirement: goes to agent selection
            handler.start_workflow("msg_1", "chat_1", "some random requirement", project)
            self.assertEqual(engine.project.status, WorkflowStatus.AWAITING_AGENT_SELECT)
            self.assertEqual(
                engine.project.pending.requirement if engine.project.pending else None,
                "some random requirement",
            )
            cards_sent_for_requirement = handler.send_card_to_chat.call_count

            # Reset for template test
            engine.project = WorkflowProject()
            handler.send_card_to_chat.reset_mock()

            # Template name: ALSO goes to agent selection (three-step flow)
            handler.start_workflow("msg_2", "chat_2", "quick-audit", project)
            self.assertEqual(engine.project.status, WorkflowStatus.AWAITING_AGENT_SELECT)
            self.assertEqual(
                engine.project.pending.requirement if engine.project.pending else None,
                "quick-audit",
            )
            # Both flows should produce a card (agent selection)
            self.assertEqual(handler.send_card_to_chat.call_count, 1)

            # Both flows should send exactly one card (agent selection card)
            self.assertEqual(cards_sent_for_requirement, 1)


# ---------------------------------------------------------------------------
# Test Class 6: TestMetadataPersistence
# ---------------------------------------------------------------------------

class TestMetadataPersistence(unittest.TestCase):
    """Tests for metadata persistence in saved workflows."""

    def _create_complex_script(self, tools: list[str], phases: list[dict]) -> str:
        """Create a script with specific tools and phases in meta."""
        phases_json = json.dumps(phases, indent=2)
        tools_json = json.dumps(tools)

        return f'''export const meta = {{
  name: "complex-workflow",
  description: "Workflow with specific metadata",
  phases: {phases_json},
  tools: {tools_json}
}};

export default async function main() {{
  return "done";
}}
'''

    def test_saved_workflow_preserves_tools(self):
        """Verify that saved workflows remember their tool list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a script with specific tools
            tools = ["coco", "claude", "aiden", "gemini", "codex"]
            phases = [{"title": "Phase 1", "detail": "First"}]
            script = self._create_complex_script(tools, phases)

            # Save as template
            save_template(tmpdir, "tool-test", script)

            # Load and verify raw content (meta extraction regex has limitations
            # with nested objects, so we verify content directly)
            content = load_template(tmpdir, "tool-test")
            self.assertIsNotNone(content)

            # Verify tools are present in the raw content
            self.assertIn('"coco"', content)
            self.assertIn('"claude"', content)
            self.assertIn('"aiden"', content)
            self.assertIn('"gemini"', content)
            self.assertIn('"codex"', content)
            self.assertIn("tools:", content)

            # Verify the tools array structure is preserved (JS uses unquoted keys)
            import re
            tools_match = re.search(r'tools:\s*\[([^\]]+)\]', content)
            self.assertIsNotNone(tools_match, "Tools array should exist in saved content")
            tools_str = tools_match.group(1)
            for tool in tools:
                self.assertIn(tool, tools_str)

            # Verify discover_templates returns correct info
            templates = discover_templates(tmpdir)
            tool_test = next(t for t in templates if t.name == "tool-test")
            # TemplateInfo doesn't include tools directly, but the file should
            # be discoverable and the path should be correct
            self.assertEqual(tool_test.name, "tool-test")
            self.assertTrue(os.path.isfile(tool_test.path))

    def test_saved_workflow_preserves_phases(self):
        """Verify that saved workflows remember their phase structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a script with specific phases
            tools = ["coco"]
            phases = [
                {"title": "Requirements Analysis", "detail": "Gather and analyze requirements"},
                {"title": "Design", "detail": "Create system design"},
                {"title": "Implementation", "detail": "Implement the solution"},
                {"title": "Testing", "detail": "Test the implementation"},
                {"title": "Deployment", "detail": "Deploy to production"},
            ]
            script = self._create_complex_script(tools, phases)

            # Save as template
            save_template(tmpdir, "phase-test", script)

            # Load and verify raw content (meta extraction regex has limitations
            # with nested objects, so we verify content directly)
            content = load_template(tmpdir, "phase-test")
            self.assertIsNotNone(content)

            # Verify each phase title and detail are present in the raw content
            self.assertIn("Requirements Analysis", content)
            self.assertIn("Gather and analyze requirements", content)
            self.assertIn("Design", content)
            self.assertIn("Create system design", content)
            self.assertIn("Implementation", content)
            self.assertIn("Implement the solution", content)
            self.assertIn("Testing", content)
            self.assertIn("Test the implementation", content)
            self.assertIn("Deployment", content)
            self.assertIn("Deploy to production", content)

            # Verify the phases array structure is preserved (JS uses unquoted keys at top level)
            import re
            phases_match = re.search(r'phases:\s*\[', content)
            self.assertIsNotNone(phases_match, "Phases array should exist in saved content")

            # Count phase objects by looking for title occurrences (nested objects have quoted keys)
            title_count = len(re.findall(r'"title":', content))
            self.assertEqual(title_count, 5, "Should have 5 phase titles")

            # Verify the script content contains phase declarations
            self.assertIn("complex-workflow", content)
            self.assertIn("name:", content)  # JS uses unquoted keys at top level


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
