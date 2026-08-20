"""Tests for dynamic roles in workflow script generation prompt.

Validates that the script generation prompt uses dynamic role guidance instead
of a static list of roles, and that the guidance appears in the correct section.
"""

import re
import unittest

from src.workflow_engine.script_gen import build_script_gen_prompt


class TestDynamicRolesGuidance(unittest.TestCase):
    """Test that the prompt contains the dynamic roles guidance text."""

    def _build_prompt(self, requirement="Test requirement"):
        return build_script_gen_prompt(
            requirement=requirement,
            available_tools=["coco", "claude", "aiden"],
        )

    def test_all_dynamic_guidance_present(self):
        """Test 1: All three key dynamic guidance phrases are present."""
        prompt = self._build_prompt()
        self.assertIn("根据任务需求自行规划角色分工", prompt)
        self.assertIn("角色不是固定列表", prompt)
        self.assertIn(
            "建议考虑：架构设计、代码实现、正确性验证、测试覆盖等维度",
            prompt,
        )


class TestNoStaticRoleList(unittest.TestCase):
    """Test that the prompt does NOT contain a static list of roles."""

    def _build_prompt(self, requirement="Test requirement"):
        return build_script_gen_prompt(
            requirement=requirement,
            available_tools=["coco", "claude", "aiden"],
        )

    def test_no_static_role_enumeration(self):
        """Test 2: No pattern of static role enumeration under Roles section."""
        prompt = self._build_prompt()
        # Find the Roles section
        roles_match = re.search(
            r"\*\*Roles \(specialized perspectives for agents\):\*\*(.*?)(?=\n## |\Z)",
            prompt,
            re.DOTALL,
        )
        self.assertIsNotNone(roles_match, "Roles section should exist")
        roles_section = roles_match.group(1)

        # Check for bullet list of role names (lowercase single words)
        # that would indicate a static list
        static_role_pattern = re.compile(
            r"^\s*-\s*(architect|reviewer|tester|coder|designer)\s*$",
            re.MULTILINE,
        )
        matches = static_role_pattern.findall(roles_section)
        self.assertEqual(
            len(matches),
            0,
            f"Roles section should not contain static role enumeration, found: {matches}",
        )


class TestRolesSectionStructure(unittest.TestCase):
    """Test that dynamic guidance appears under the correct heading."""

    def _build_prompt(self, requirement="Test requirement"):
        return build_script_gen_prompt(
            requirement=requirement,
            available_tools=["coco", "claude", "aiden"],
        )

    def test_dynamic_guidance_under_roles_heading(self):
        """Test 3: Dynamic guidance text appears under the Roles heading."""
        prompt = self._build_prompt()

        # Extract the Roles section content
        roles_match = re.search(
            r"\*\*Roles \(specialized perspectives for agents\):\*\*(.*?)(?=\n## |\Z)",
            prompt,
            re.DOTALL,
        )
        self.assertIsNotNone(roles_match, "Roles section should be found")

        roles_content = roles_match.group(1)

        # Verify all dynamic guidance phrases are within the Roles section
        self.assertIn(
            "根据任务需求自行规划角色分工",
            roles_content,
            "Dynamic guidance should be under the Roles heading",
        )
        self.assertIn(
            "角色不是固定列表",
            roles_content,
            "Dynamic guidance should be under the Roles heading",
        )
        self.assertIn(
            "建议考虑：架构设计、代码实现、正确性验证、测试覆盖等维度",
            roles_content,
            "Dynamic guidance should be under the Roles heading",
        )

class TestRoleParameterMention(unittest.TestCase):
    """Test that the guidance mentions the role parameter for agent() calls."""

    def _build_prompt(self, requirement="Test requirement"):
        return build_script_gen_prompt(
            requirement=requirement,
            available_tools=["coco", "claude", "aiden"],
        )

    def test_agent_call_with_role_parameter(self):
        """Test 4: Prompt mentions `role` parameter for `agent()` calls."""
        prompt = self._build_prompt()
        self.assertIn(
            "agent()",
            prompt,
            "Prompt should mention agent() calls",
        )
        # Check that role parameter is mentioned in context of agent() calls
        self.assertIn(
            "每个 agent() 调用可通过 `role` 参数",
            prompt,
            "Prompt should mention role parameter for agent() calls",
        )

class TestDynamicWorkflowReliabilityGuidance(unittest.TestCase):
    """Prompt guidance for Claude-style dynamic workflow behavior."""

    def _build_prompt(self, requirement="Diagnose and fix a flaky workflow bug"):
        return build_script_gen_prompt(
            requirement=requirement,
            available_tools=["traex", "claude", "aiden"],
            orchestrator_agent="traex",
        )

    def test_prompt_requires_unique_agent_labels(self):
        prompt = self._build_prompt()

        self.assertIn("每个 agent() label 必须唯一", prompt)
        self.assertIn("不要复用 task-analysis", prompt)

    def test_prompt_discourages_slow_monolithic_analysis_agent(self):
        prompt = self._build_prompt()

        self.assertIn("不要先派一个大而慢的 analysis agent", prompt)
        self.assertIn("直接基于用户需求选择 classify/fanout/verify/loop/race", prompt)

    def test_prompt_requires_timeout_and_error_fallbacks(self):
        prompt = self._build_prompt()

        self.assertIn("为每个 agent() 显式设置短超时", prompt)
        self.assertIn("检查 result.error 并提供 fallback", prompt)

    def test_role_examples_in_roles_section(self):
        """Test 4: Role examples appear within the Roles section."""
        prompt = self._build_prompt()

        roles_match = re.search(
            r"\*\*Roles \(specialized perspectives for agents\):\*\*(.*?)(?=\n## |\Z)",
            prompt,
            re.DOTALL,
        )
        self.assertIsNotNone(roles_match)
        roles_content = roles_match.group(1)

        self.assertIn(
            "architect、reviewer、tester 等",
            roles_content,
            "Role examples should be in the Roles section",
        )


class TestDifferentOrchestratorAgents(unittest.TestCase):
    """Test that different orchestrator agents all get dynamic roles guidance."""

    def _build_prompt(self, orchestrator_agent, requirement="Test requirement"):
        return build_script_gen_prompt(
            requirement=requirement,
            available_tools=["coco", "claude", "aiden"],
            orchestrator_agent=orchestrator_agent,
        )

    def test_coco_gets_dynamic_roles(self):
        """Test 5: orchestrator_agent='coco' gets dynamic roles guidance."""
        prompt = self._build_prompt("coco")
        self.assertIn("根据任务需求自行规划角色分工", prompt)
        self.assertIn("角色不是固定列表", prompt)
        self.assertNotRegex(prompt, r"-\s*architect\b")
        self.assertNotRegex(prompt, r"-\s*reviewer\b")

    def test_claude_gets_dynamic_roles(self):
        """Test 5: orchestrator_agent='claude' gets dynamic roles guidance."""
        prompt = self._build_prompt("claude")
        self.assertIn("根据任务需求自行规划角色分工", prompt)
        self.assertIn("角色不是固定列表", prompt)
        self.assertNotRegex(prompt, r"-\s*architect\b")
        self.assertNotRegex(prompt, r"-\s*reviewer\b")

    def test_aiden_gets_dynamic_roles(self):
        """Test 5: orchestrator_agent='aiden' gets dynamic roles guidance."""
        prompt = self._build_prompt("aiden")
        self.assertIn("根据任务需求自行规划角色分工", prompt)
        self.assertIn("角色不是固定列表", prompt)
        self.assertNotRegex(prompt, r"-\s*architect\b")
        self.assertNotRegex(prompt, r"-\s*reviewer\b")

    def test_all_agents_get_recommended_dimensions(self):
        """Test 5: All orchestrator agents get the recommended dimensions."""
        for agent in ["coco", "claude", "aiden"]:
            with self.subTest(orchestrator_agent=agent):
                prompt = self._build_prompt(agent)
                self.assertIn(
                    "建议考虑：架构设计、代码实现、正确性验证、测试覆盖等维度",
                    prompt,
                    f"Agent '{agent}' should get recommended dimensions",
                )


class TestDifferentRequirementTypes(unittest.TestCase):
    """Test that different requirement types all get dynamic roles guidance."""

    def _build_prompt(self, requirement):
        return build_script_gen_prompt(
            requirement=requirement,
            available_tools=["coco", "claude", "aiden"],
        )

    def test_requirement_does_not_affect_roles_section(self):
        """Test 6: Different requirements don't change the roles guidance content."""
        prompts = []
        for req in [
            "Build a web application",
            "Refactor the authentication module",
            "Write comprehensive tests",
        ]:
            prompt = self._build_prompt(req)
            # Extract the roles section
            roles_match = re.search(
                r"\*\*Roles \(specialized perspectives for agents\):\*\*(.*?)(?=\n## |\Z)",
                prompt,
                re.DOTALL,
            )
            self.assertIsNotNone(roles_match)
            prompts.append(roles_match.group(1))

        # All roles sections should be identical
        self.assertEqual(
            prompts[0],
            prompts[1],
            "Roles section should be identical for different requirements",
        )
        self.assertEqual(
            prompts[1],
            prompts[2],
            "Roles section should be identical for different requirements",
        )


class TestNoBudgetContent(unittest.TestCase):
    """Test that the prompt does not contain budget-related content."""

    def _build_prompt(self, requirement="Test requirement"):
        return build_script_gen_prompt(
            requirement=requirement,
            available_tools=["coco", "claude", "aiden"],
        )

    def test_no_budget_english(self):
        """Test 7: Prompt does not contain 'budget' (case insensitive)."""
        prompt = self._build_prompt()
        self.assertNotIn("budget", prompt.lower())

    def test_no_budget_chinese(self):
        """Test 7: Prompt does not contain '预算'."""
        prompt = self._build_prompt()
        self.assertNotIn("预算", prompt)

    def test_no_budget_with_different_agents(self):
        """Test 7: No budget content with different orchestrator agents."""
        for agent in ["coco", "claude", "aiden"]:
            with self.subTest(orchestrator_agent=agent):
                prompt = build_script_gen_prompt(
                    requirement="Test",
                    available_tools=["coco"],
                    orchestrator_agent=agent,
                )
                self.assertNotIn("budget", prompt.lower())
                self.assertNotIn("预算", prompt)


class TestPromptStructure(unittest.TestCase):
    """Test that the prompt has the correct structure with all expected sections."""

    def _build_prompt(self, requirement="Test requirement"):
        return build_script_gen_prompt(
            requirement=requirement,
            available_tools=["coco", "claude", "aiden"],
        )

    def test_script_generation_prompt_requires_card_summary_contract(self):
        prompt = self._build_prompt()

        self.assertIn("card_summary", prompt)
        self.assertIn('"verdict"', prompt)
        self.assertIn('"conclusion"', prompt)
        self.assertIn('"findings"', prompt)
        self.assertIn('"verification"', prompt)
        self.assertIn('"deliverables"', prompt)
        self.assertIn('"next_steps"', prompt)
        self.assertIn("完整语义条目", prompt)

    def test_section_order(self):
        """Test 8: Sections appear in the correct order."""
        prompt = self._build_prompt()

        positions = {
            "title": prompt.find("# Workflow Script Generation Task"),
            "user_requirement": prompt.find("## User Requirement"),
            "available_resources": prompt.find("## Available Resources"),
            "tools": prompt.find("**Tools (AI agents you can dispatch):**"),
            "roles": prompt.find("**Roles (specialized perspectives for agents):**"),
            "output_format": prompt.find("## Output Format"),
        }

        # All sections should be found
        for name, pos in positions.items():
            self.assertGreaterEqual(
                pos,
                0,
                f"Section '{name}' should be found in prompt",
            )

        # Check ordering
        self.assertLess(
            positions["title"],
            positions["user_requirement"],
            "Title should come before User Requirement",
        )
        self.assertLess(
            positions["user_requirement"],
            positions["available_resources"],
            "User Requirement should come before Available Resources",
        )
        self.assertLess(
            positions["available_resources"],
            positions["tools"],
            "Available Resources should come before Tools",
        )
        self.assertLess(
            positions["tools"],
            positions["roles"],
            "Tools should come before Roles",
        )
        self.assertLess(
            positions["roles"],
            positions["output_format"],
            "Roles should come before Output Format",
        )


if __name__ == "__main__":
    unittest.main()
