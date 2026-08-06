import pytest

from src.agent.intent_recognizer import (
    IntentMatcher,
    IntentRecognizer,
    IntentResult,
    IntentType,
    TaskStep,
)


class TestIntentRecognizerQuickMatch:
    @pytest.fixture
    def recognizer(self):
        return IntentRecognizer()

    def test_exact_command_coco(self, recognizer):
        result = recognizer._quick_match("/coco")
        assert result is not None
        assert result.primary_intent == IntentType.ENTER_COCO
        assert result.confidence == 1.0

    def test_exact_command_traex(self, recognizer):
        result = recognizer._quick_match("/traex")
        assert result is not None
        assert result.primary_intent == IntentType.ENTER_TRAEX
        assert result.confidence == 1.0

    def test_exact_command_exit(self, recognizer):
        result = recognizer._quick_match("/exit")
        assert result is not None
        assert result.primary_intent == IntentType.EXIT_MODE

    def test_coco_info_command(self, recognizer):
        result = recognizer._quick_match("/coco_info")
        assert result is not None
        assert result.primary_intent == IntentType.COCO_MESSAGE
        assert result.primary_data.get("command") == "info"

    def test_new_project_command(self, recognizer):
        result = recognizer._quick_match("/new myapp ~/workspace")
        assert result is not None
        assert result.primary_intent == IntentType.CREATE_PROJECT
        assert result.primary_data.get("name") == "myapp"
        assert result.primary_data.get("path") == "~/workspace"

    def test_new_project_command_no_path(self, recognizer):
        result = recognizer._quick_match("/new myapp")
        assert result is not None
        assert result.primary_intent == IntentType.CREATE_PROJECT
        assert result.primary_data.get("name") == "myapp"
        assert result.primary_data.get("path") == ""

    def test_switch_project_command(self, recognizer):
        result = recognizer._quick_match("/switch myapp")
        assert result is not None
        assert result.primary_intent == IntentType.SWITCH_PROJECT
        assert result.primary_data.get("name") == "myapp"

    def test_close_project_command(self, recognizer):
        result = recognizer._quick_match("/close myapp")
        assert result is not None
        assert result.primary_intent == IntentType.CLOSE_PROJECT
        assert result.primary_data.get("name") == "myapp"

    def test_shell_command_ls(self, recognizer):
        result = recognizer._quick_match("ls -la")
        assert result is not None
        assert result.primary_intent == IntentType.SHELL_COMMAND
        assert result.confidence >= 0.9

    @pytest.mark.parametrize("text", ["git status", "npm install", "python main.py"])
    def test_shell_command_other(self, recognizer, text):
        result = recognizer._quick_match(text)
        assert result is not None
        assert result.primary_intent == IntentType.SHELL_COMMAND

    @pytest.mark.parametrize(
        "text, current_mode",
        [
            ("退出", "coco"),
            ("exit", "claude"),
        ],
    )
    def test_exit_keyword_in_programming_mode(self, recognizer, text, current_mode):
        result = recognizer._quick_match(text, current_mode=current_mode)
        assert result is not None
        assert result.primary_intent == IntentType.EXIT_MODE

    def test_exit_keyword_not_in_programming_mode(self, recognizer):
        result = recognizer._quick_match("退出", current_mode="smart")
        assert result is None

    def test_project_list_keyword(self, recognizer):
        result = recognizer._quick_match("项目列表")
        assert result is not None
        assert result.primary_intent == IntentType.LIST_PROJECTS

    def test_common_word_not_matched(self, recognizer):
        result = recognizer._quick_match("hello")
        assert result is None

    def test_quick_match_delegates_to_registered_matcher(self, recognizer):
        matcher = IntentMatcher()

        def always_help(_recognizer, text, current_mode):
            assert text == "anything"
            assert current_mode == "smart"
            return IntentResult.single(
                IntentType.SHOW_HELP,
                confidence=0.42,
                original_text=text,
                reasoning="test matcher",
                description="测试注册表入口",
            )

        matcher.register("always_help", always_help)
        recognizer.intent_matcher = matcher

        result = recognizer._quick_match("anything")

        assert result is not None
        assert result.primary_intent == IntentType.SHOW_HELP
        assert result.confidence == 0.42

    def test_default_matcher_registry_exposes_rule_names(self, recognizer):
        assert recognizer.intent_matcher.rule_names[:3] == (
            "command_typo",
            "exact_command",
            "coco_info",
        )


class TestIntentRecognizerContextHint:
    @pytest.fixture
    def recognizer(self):
        return IntentRecognizer()

    @pytest.mark.parametrize(
        "current_mode, expected",
        [
            ("coco", IntentType.COCO_MESSAGE),
            ("claude", IntentType.CLAUDE_MESSAGE),
            ("traex", IntentType.TRAEX_MESSAGE),
            ("smart", IntentType.SHELL_COMMAND),
        ],
    )
    def test_fallback_intent(self, recognizer, current_mode, expected):
        fallback = recognizer._get_fallback_intent(current_mode=current_mode)
        assert fallback == expected

    def test_exact_command_tools(self, recognizer):
        result = recognizer._quick_match("/tools")
        assert result is not None
        assert result.primary_intent == IntentType.SHOW_TOOLS
        assert result.confidence == 1.0

    def test_exact_command_tools_status(self, recognizer):
        result = recognizer._quick_match("/tools_status")
        assert result is not None
        assert result.primary_intent == IntentType.TOOLS_STATUS
        assert result.confidence == 1.0



class TestIntentResult:
    def test_single_task(self):
        result = IntentResult.single(
            intent=IntentType.ENTER_COCO, confidence=0.9, original_text="帮我写代码", description="进入编程模式"
        )
        assert result.primary_intent == IntentType.ENTER_COCO
        assert result.confidence == 0.9
        assert result.is_multi_task is False
        assert len(result.tasks) == 1

    def test_multi_task(self):
        result = IntentResult(
            tasks=[
                TaskStep(intent=IntentType.CHANGE_DIR, description="切换目录", data={"path": "~/workspace"}),
                TaskStep(intent=IntentType.ENTER_COCO, description="进入编程模式", data={}),
            ],
            confidence=0.85,
            original_text="去workspace目录然后帮我写代码",
        )
        assert result.is_multi_task is True
        assert len(result.tasks) == 2
        assert result.primary_intent == IntentType.CHANGE_DIR

    def test_primary_data(self):
        result = IntentResult.single(intent=IntentType.CREATE_PROJECT, data={"name": "myapp", "path": "~/workspace"})
        assert result.primary_data.get("name") == "myapp"
        assert result.primary_data.get("path") == "~/workspace"


class TestIntentTypeMapping:
    @pytest.fixture
    def recognizer(self):
        return IntentRecognizer()

    def test_intent_map_values(self, recognizer):
        assert recognizer.INTENT_MAP["enter_coco"] == IntentType.ENTER_COCO
        assert recognizer.INTENT_MAP["exit_coco"] == IntentType.EXIT_COCO
        assert recognizer.INTENT_MAP["coco_message"] == IntentType.COCO_MESSAGE
        assert recognizer.INTENT_MAP["enter_traex"] == IntentType.ENTER_TRAEX
        assert recognizer.INTENT_MAP["traex_message"] == IntentType.TRAEX_MESSAGE
        assert recognizer.INTENT_MAP["shell"] == IntentType.SHELL_COMMAND
        assert recognizer.INTENT_MAP["unknown"] == IntentType.UNKNOWN
        assert recognizer.INTENT_MAP["show_tools"] == IntentType.SHOW_TOOLS
        assert recognizer.INTENT_MAP["tools_status"] == IntentType.TOOLS_STATUS


# ── Boundary test cases ──────────────────────────────────────────────


class TestQuickMatchBoundaryEdgeCases:
    """≥8 boundary tests covering edge cases not previously covered."""

    @pytest.fixture
    def recognizer(self):
        return IntentRecognizer()

    def test_exit_keyword_exactly_20_chars_in_programming_mode(self, recognizer):
        """Exit keyword in text exactly 20 chars long should NOT match (guard is len < 20)."""
        # "退出" is 2 chars, pad to exactly 20 chars
        text = "退出" + "x" * 18  # len == 20
        assert len(text) == 20
        result = recognizer._quick_match(text, current_mode="coco")
        assert result is None  # len < 20 guard prevents match

    def test_exit_keyword_19_chars_in_programming_mode(self, recognizer):
        """Exit keyword in text of 19 chars should match (guard is len < 20)."""
        text = "退出" + "x" * 17  # len == 19
        assert len(text) == 19
        result = recognizer._quick_match(text, current_mode="coco")
        assert result is not None
        assert result.primary_intent == IntentType.EXIT_MODE

    def test_deep_update_with_content(self, recognizer):
        """/deep_update with a message extracts the message correctly."""
        result = recognizer._quick_match("/deep_update 增加错误处理")
        assert result is not None
        assert result.primary_intent == IntentType.DEEP_UPDATE
        assert result.primary_data["message"] == "增加错误处理"

    def test_spec_command_with_requirement(self, recognizer):
        result = recognizer._quick_match("/spec 重构认证模块")
        assert result is not None
        assert result.primary_intent == IntentType.ENTER_SPEC
        assert result.primary_data["requirement"] == "重构认证模块"

    def test_heuristic_shell_single_char_word_not_matched(self, recognizer):
        """Single-char first word should NOT trigger command heuristic (2 <= len <= 15)."""
        result = recognizer._quick_match("x something")
        assert result is None

    def test_heuristic_shell_long_word_not_matched(self, recognizer):
        """First word > 15 chars should NOT trigger command heuristic."""
        result = recognizer._quick_match("abcdefghijklmnop arg")  # 16 chars
        assert result is None

    def test_info_commands_all_modes(self, recognizer):
        """All mode-specific _info commands should return correct intent."""
        info_map = {
            "/coco_info": IntentType.COCO_MESSAGE,
            "/claude_info": IntentType.CLAUDE_MESSAGE,
            "/aiden_info": IntentType.AIDEN_MESSAGE,
            "/codex_info": IntentType.CODEX_MESSAGE,
            "/gemini_info": IntentType.GEMINI_MESSAGE,
        }
        for cmd, expected_intent in info_map.items():
            result = recognizer._quick_match(cmd)
            assert result is not None, f"Expected match for {cmd}"
            assert result.primary_intent == expected_intent, f"Wrong intent for {cmd}"
            assert result.primary_data.get("command") == "info"

    def test_cd_with_tilde_path(self, recognizer):
        """cd ~/workspace should capture tilde path correctly."""
        result = recognizer._quick_match("cd ~/workspace")
        assert result is not None
        assert result.primary_intent == IntentType.CHANGE_DIR
        assert result.primary_data["path"] == "~/workspace"
