import json

import pytest

from src.card.builder import CardBuilder
from src.card.models import EngineCardState
from src.card.shared import THEMES, get_theme
from src.config import Settings
from src.project.context import ProjectContext, SessionSnapshot


class TestProjectTheme:
    def test_get_theme(self):
        theme = get_theme("green")

        assert theme.name == "green"
        assert theme.emoji == "🟢"
        assert theme.header_template == "green"

    def test_get_unknown_theme_returns_default(self):
        theme = get_theme("unknown_color")

        assert theme.name == "green"

    def test_all_themes_exist(self):
        expected_colors = ["green", "blue", "purple", "orange", "red", "turquoise"]

        for color in expected_colors:
            assert color in THEMES


class TestCardBuilder:
    @pytest.fixture
    def sample_project(self):
        return ProjectContext(
            project_id="test_project",
            project_name="Test Project",
            root_path="/tmp/test",
            working_dir="/tmp/test/src",
            theme_color="green",
            emoji_prefix="🟢",
        )

    def test_build_project_response_card(self, sample_project):
        msg_type, content = CardBuilder.build_project_response_card(
            project=sample_project,
            title="Test Title",
            content="Test content here",
            show_buttons=True,
        )

        assert msg_type == "interactive"

        card = json.loads(content)
        assert card["config"]["wide_screen_mode"] is True
        assert "Test Project" in card["header"]["title"]["content"]
        assert card["header"]["template"] == "green"

        elements = card["body"]["elements"]
        assert any("Test Title" in str(e) for e in elements)
        assert any("Test content here" in str(e) for e in elements)
        # Schema 2.0: 按钮统一使用 column_set 布局
        column_set_elements = [e for e in elements if e.get("tag") == "column_set"]
        assert len(column_set_elements) >= 1
        columns = column_set_elements[0]["columns"]
        assert len(columns) >= 1
        for col in columns:
            if col["elements"]:
                btn = col["elements"][0]
                assert "behaviors" in btn
                assert btn["behaviors"][0]["type"] == "callback"
                assert isinstance(btn["behaviors"][0]["value"], dict)
                assert btn.get("size") == "medium"

    def test_build_project_response_card_mobile_layout_forces_grid(self, sample_project):
        from src.card.shared import build_responsive_layout

        buttons = [
            {"tag": "button", "text": {"tag": "plain_text", "content": "Btn1"}, "type": "primary"},
            {"tag": "button", "text": {"tag": "plain_text", "content": "Btn2"}, "type": "default"},
        ]
        result = build_responsive_layout(buttons, layout="mobile")
        assert len(result) > 0
        assert all(e.get("tag") == "column_set" for e in result)
        assert all(e.get("flex_mode") == "none" for e in result)


    def test_build_info_card_progress_bar_not_duplicated_when_content_contains_bar(self):
        progress_bar = "[████░░░░░░] 40% (2/5)"
        content = f"任务中...\n\n{progress_bar}\n"
        _, card_content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(
                title="X",
                content=content,
                progress_bar=progress_bar,
                engine_name="Coco",
                show_buttons=False,
            ),
        )
        card = json.loads(card_content)
        progress_elems = [
            e
            for e in card.get("body", {}).get("elements", [])
            if e.get("tag") == "markdown" and str(e.get("content", "")).startswith("📊 ")
        ]
        assert len(progress_elems) == 0

    @pytest.mark.parametrize("mode,attr,emoji,label", [
        ("coco_mode", "coco_mode", "🤖", "Coco"),
        ("claude_mode", "claude_mode", "🔮", "Claude"),
    ])
    def test_build_project_response_card_ai_modes(self, sample_project, mode, attr, emoji, label):
        setattr(sample_project, attr, True)
        msg_type, content = CardBuilder.build_project_response_card(
            project=sample_project, title=f"{label} Response", content="AI response",
        )
        assert msg_type == "interactive"
        card = json.loads(content)
        assert emoji in card["header"]["title"]["content"]


    def test_build_project_response_card_no_buttons(self, sample_project):
        msg_type, content = CardBuilder.build_project_response_card(
            project=sample_project, title="Title", content="Content", show_buttons=False,
        )
        card = json.loads(content)
        column_set_elements = [e for e in card["body"]["elements"] if e.get("tag") == "column_set"]
        assert len(column_set_elements) == 0

    def test_build_status_board_card_empty(self):
        msg_type, content = CardBuilder.build_status_board_card([], None)
        assert msg_type == "interactive"
        card = json.loads(content)
        assert "项目看板" in card["header"]["title"]["content"]
        assert any("暂无项目" in str(e) for e in card["body"]["elements"])

    def test_build_status_board_card_with_projects(self, sample_project):
        project2 = ProjectContext(
            project_id="proj2", project_name="Project 2",
            root_path="/tmp/proj2", theme_color="blue", emoji_prefix="🔵",
        )
        msg_type, content = CardBuilder.build_status_board_card(
            [sample_project, project2], current_project_id="test_project",
        )
        card = json.loads(content)
        content_str = json.dumps(card, ensure_ascii=False)
        assert "Test Project" in content_str
        assert "Project 2" in content_str
        assert "(当前)" in content_str

    def test_build_notification_card(self, sample_project):
        msg_type, content = CardBuilder.build_notification_card(
            project=sample_project, notification_type="success",
            title="Task Complete", content="The task has been completed successfully.",
            suggestions=["Run tests", "Deploy to staging"],
        )
        assert msg_type == "interactive"
        card = json.loads(content)
        assert "✅" in card["header"]["title"]["content"]
        assert any("建议下一步" in str(e) for e in card["body"]["elements"])

    def test_build_resume_cards(self, sample_project):
        """Coco resume cards render session info."""
        sample_project.coco_session_snapshot = SessionSnapshot(
            session_id="session_123", query_count=10,
            last_query="帮我写一个函数", is_resumable=True,
        )
        _, content = CardBuilder.build_coco_resume_card(sample_project)
        content_str = json.dumps(json.loads(content), ensure_ascii=False)
        assert "session_123" in content_str
        assert "恢复会话" in content_str


    def test_build_project_created_card(self, sample_project):
        msg_type, content = CardBuilder.build_project_created_card(sample_project)
        card = json.loads(content)
        assert "新项目已创建" in card["header"]["title"]["content"]
        content_str = json.dumps(card)
        assert "Test Project" in content_str
        assert "test_project" in content_str

    def test_build_error_cards(self, sample_project):
        """Error cards: without project, with project, and with GhostAPError."""
        # Without project
        _, content = CardBuilder.build_error_card("Something went wrong")
        card = json.loads(content)
        assert card["header"]["template"] == "red"
        assert any("Something went wrong" in str(e) for e in card["body"]["elements"])

        # With project
        _, content2 = CardBuilder.build_error_card("Error message", project=sample_project)
        card2 = json.loads(content2)
        content_str = json.dumps(card2)
        assert "/tmp/test" in content_str
        assert "Error message" in content_str

        # With GhostAPError
        from src.utils.errors import GhostAPError
        err = GhostAPError("Business error", quick_actions=["retry", "cancel"])
        _, content3 = CardBuilder.build_error_card(err)
        content_str3 = json.dumps(json.loads(content3), ensure_ascii=False)
        assert "Business error" in content_str3
        assert "重试" in content_str3

    def test_build_project_response_card_with_images(self, sample_project):
        msg_type, content = CardBuilder.build_project_response_card(
            project=sample_project, title="Claude", content="分析结果",
            image_keys=["img_v2_abc", "img_v2_def"],
        )
        card = json.loads(content)
        elements = card["body"]["elements"]
        img_elements = [e for e in elements if e.get("tag") == "img"]
        assert len(img_elements) == 2
        assert img_elements[0]["img_key"] == "img_v2_abc"


    def test_build_image_elements(self):
        elements = CardBuilder._build_image_elements(["key1", "key2", "key3"])
        assert len(elements) == 3
        assert elements[0]["tag"] == "img"
        assert elements[0]["img_key"] == "key1"
        assert CardBuilder._build_image_elements([]) == []

    def test_format_time_ago(self):
        import time
        now = time.time()
        assert CardBuilder._format_time_ago(now) == "刚刚"
        assert "分钟前" in CardBuilder._format_time_ago(now - 120)
        assert "小时前" in CardBuilder._format_time_ago(now - 7200)


class TestDeepCard:
    @pytest.fixture
    def sample_project(self):
        return ProjectContext(
            project_id="test_project", project_name="Test Project",
            root_path="/tmp/test", working_dir="/tmp/test/src",
            theme_color="green", emoji_prefix="🟢",
        )

    def test_build_info_card_basic(self, sample_project):
        msg_type, content = CardBuilder.build_info_card(
            project=sample_project,
            state=EngineCardState(title="Test Title", content="Test content", engine_name="Coco"),
        )
        assert msg_type == "interactive"
        card = json.loads(content)
        assert card["header"]["title"]["content"] == "🧠 Test Title"
        elements = card["body"]["elements"]
        assert len(elements) >= 3

    def test_build_info_card_with_progress_bar(self, sample_project):
        msg_type, content = CardBuilder.build_info_card(
            project=sample_project,
            state=EngineCardState(
                title="Executing", content="Task in progress",
                progress_bar="[█████░░░░░] 50% (2/4)",
                is_executing=True, engine_project_id="proj123",
            ),
        )
        card = json.loads(content)
        content_str = json.dumps(card, ensure_ascii=False)
        assert "50%" in content_str
        assert "deep_pause" in content_str
        assert "deep_stop" in content_str

    def test_build_info_card_paused(self, sample_project):
        msg_type, content = CardBuilder.build_info_card(
            project=sample_project,
            state=EngineCardState(
                title="Paused", content="Execution paused",
                is_paused=True, engine_project_id="proj123",
            ),
        )
        card = json.loads(content)
        content_str = json.dumps(card, ensure_ascii=False)
        assert "deep_resume" in content_str
        assert "deep_stop" in content_str

    @pytest.mark.parametrize("engine_name,expected_template", [
        ("Coco", "turquoise"),
        ("Claude", "violet"),
    ])
    def test_build_info_card_engine_color(self, engine_name, expected_template):
        _, content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(title="Test", content="No project", engine_name=engine_name),
        )
        card = json.loads(content)
        assert card["header"]["template"] == expected_template

    def test_build_info_card_with_progress(self):
        _, content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(title="Test", content="Doing stuff", progress_bar="[===>   ]", engine_name="Coco"),
        )
        card = json.loads(content)
        found_bar = any("[===>   ]" in el.get("content", "") for el in card["body"]["elements"])
        assert found_bar

    def test_build_deep_buttons_executing(self):
        long_content = "\n".join([f"Line {i}" for i in range(55)])
        buttons = CardBuilder._build_deep_buttons(
            EngineCardState(engine_project_id="proj123", is_executing=True, content=long_content)
        )
        assert len(buttons) == 4
        texts = [b["text"]["content"] for b in buttons]
        assert "⏸️ 暂停" in texts
        assert "⏹️ 停止" in texts

    def test_build_deep_buttons_paused(self):
        long_content = "\n".join([f"Line {i}" for i in range(55)])
        buttons = CardBuilder._build_deep_buttons(
            EngineCardState(engine_project_id="proj123", is_paused=True, content=long_content)
        )
        assert len(buttons) == 4
        texts = [b["text"]["content"] for b in buttons]
        assert "▶️ 继续" in texts
        assert "⏹️ 停止" in texts

    def test_build_deep_buttons_neither(self):
        long_content = "\n".join([f"Line {i}" for i in range(55)])
        buttons = CardBuilder._build_deep_buttons(EngineCardState(engine_project_id="proj123", content=long_content))
        assert len(buttons) == 2


# ---------------------------------------------------------------------------
# 卡片结构验证 —— 所有卡片使用 v2 格式（schema 2.0，body.elements）
# ---------------------------------------------------------------------------


class TestCardSchema20Structure:
    """验证所有卡片均使用正确的 v2 卡片结构（schema 2.0，body.elements）"""

    @pytest.fixture
    def project(self):
        return ProjectContext(
            project_id="md_test", project_name="MD Test",
            root_path="/tmp/md_test", working_dir="/tmp/md_test",
            theme_color="blue", emoji_prefix="🔵",
        )

    def _parse_card(self, card_tuple: tuple[str, str]) -> dict:
        msg_type, content = card_tuple
        assert msg_type == "interactive"
        return json.loads(content)

    def _assert_v2_structure(self, card: dict, label: str):
        assert card.get("schema") == "2.0", f"{label}: schema should be '2.0'"
        assert "body" in card, f"{label}: missing body field"
        assert "elements" in card["body"], f"{label}: missing elements in body"

    def _assert_no_lark_md(self, card: dict, label: str):
        card_str = json.dumps(card, ensure_ascii=False)
        assert "lark_md" not in card_str, f"{label}: contains lark_md"

    @pytest.mark.parametrize("card_name,build_fn", [
        ("project_response", lambda p: CardBuilder.build_project_response_card(p, "Title", "Content")),
        ("coco_response", lambda p: CardBuilder.build_coco_response_card(p, "Title", "Content")),
        ("status_board", lambda p: CardBuilder.build_status_board_card([p], p.project_id)),
        ("error", lambda _: CardBuilder.build_error_card("Error msg")),
        ("deep", lambda p: CardBuilder.build_info_card(p, EngineCardState(title="T", content="C", engine_name="Coco"))),
    ])
    def test_card_v2_structure_and_no_lark_md(self, project, card_name, build_fn):
        card = self._parse_card(build_fn(project))
        self._assert_v2_structure(card, card_name)
        self._assert_no_lark_md(card, card_name)

    def test_coco_resume_card_schema(self, project):
        project.coco_session_snapshot = SessionSnapshot(
            session_id="s1", query_count=3, last_query="test", is_resumable=True,
        )
        card = self._parse_card(CardBuilder.build_coco_resume_card(project))
        self._assert_v2_structure(card, "coco_resume")
        self._assert_no_lark_md(card, "coco_resume")


# ---------------------------------------------------------------------------
# Markdown 内容元素的渲染验证 (consolidated)
# ---------------------------------------------------------------------------


class TestMarkdownContentRendering:
    """验证 _build_content_element 使用 {"tag": "markdown"} 并完整传递内容。"""

    @pytest.fixture
    def project(self):
        return ProjectContext(
            project_id="md_project", project_name="Markdown Test",
            root_path="/tmp/md", theme_color="green", emoji_prefix="🟢",
        )

    def _get_content_element(self, card_tuple: tuple[str, str]) -> dict:
        card = json.loads(card_tuple[1])
        elements = card["body"]["elements"]
        md_elements = [
            e for e in elements
            if e.get("tag") == "markdown"
            and not str(e.get("content", "")).startswith("📁")
            and not str(e.get("content", "")).startswith("📊")
            and not str(e.get("content", "")).startswith("💡")
        ]
        assert len(md_elements) >= 1, "No content markdown element found"
        return md_elements[0]

    @pytest.mark.parametrize("label,content,expected_fragments", [
        ("headings", "# 一级标题\n## 二级标题\n### 三级标题", ["# 一级标题", "## 二级标题"]),
        ("inline_code", "使用 `pip install flask` 安装", ["`pip install flask`"]),
        ("link", "参考 [飞书文档](https://open.feishu.cn/)", ["[飞书文档](https://open.feishu.cn/)"]),
        ("table", "| 方法 | 耗时 |\n|------|------|\n| GET  | 12ms |", ["| 方法 |", "| GET  |"]),
        ("code_blocks", "```python\nprint('hello')\n```\n\n```bash\necho 'hello'\n```", ["```python", "```bash"]),
    ])
    def test_markdown_syntax_passthrough(self, project, label, content, expected_fragments):
        elem = self._get_content_element(
            CardBuilder.build_project_response_card(project, "Test", content, show_buttons=False)
        )
        assert elem["tag"] == "markdown"
        for frag in expected_fragments:
            assert frag in elem["content"], f"{label}: missing '{frag}'"

    def test_mixed_markdown_ai_response(self, project):
        """模拟 AI 回复的典型复杂 Markdown"""
        content = (
            "# 分析结果\n\n## 问题描述\n"
            "1. **内存泄漏**: `session` 对象未正确关闭\n\n"
            "## 修复方案\n\n```python\nclass SessionManager:\n    pass\n```\n\n"
            "> 注意：修复后需要运行完整测试套件验证\n\n"
            "[Python线程安全](https://docs.python.org/3/library/threading.html)"
        )
        elem = self._get_content_element(
            CardBuilder.build_project_response_card(project, "代码审查", content, show_buttons=False)
        )
        assert elem["tag"] == "markdown"
        for frag in ["# 分析结果", "**内存泄漏**", "```python", "> 注意", "[Python线程安全]"]:
            assert frag in elem["content"]

    def test_content_with_title_prepended(self, project):
        elem = CardBuilder._build_content_element("这是正文内容", with_title="测试标题")
        assert elem["tag"] == "markdown"
        assert elem["content"].startswith("**测试标题**")
        assert "这是正文内容" in elem["content"]

    def test_deep_card_markdown_rendering(self, project):
        content = "## 任务进度\n- [x] 步骤一\n- [ ] 步骤二\n\n```python\nresult = run()\n```"
        card = json.loads(
            CardBuilder.build_info_card(
                project, state=EngineCardState(title="Deep任务", content=content, engine_name="Claude", show_buttons=False),
            )[1]
        )
        elements = card["body"]["elements"]
        md_content = [e for e in elements if e.get("tag") == "markdown" and "任务进度" in str(e.get("content", ""))]
        assert len(md_content) >= 1
        assert "```python" in md_content[0]["content"]

    def test_notification_card_suggestions_use_markdown(self, project):
        card = json.loads(
            CardBuilder.build_notification_card(
                project, "success", "Done", "全部完成",
                suggestions=["运行 `pytest`", "执行 **部署**"],
            )[1]
        )
        elements = card["body"]["elements"]
        suggestion_md = [
            e for e in elements if e.get("tag") == "markdown" and "建议下一步" in str(e.get("content", ""))
        ]
        assert len(suggestion_md) == 1
        assert "运行 `pytest`" in suggestion_md[0]["content"]

    def test_coco_card_passes_markdown_through(self, project):
        content = "**步骤一**: `git pull`\n\n```bash\ngit pull origin main\n```"
        card = json.loads(CardBuilder.build_coco_response_card(project, "Git操作", content, show_buttons=False)[1])
        card_text = json.dumps(card, ensure_ascii=False)
        assert "**步骤一**" in card_text
        assert "```bash" in card_text


# ---------------------------------------------------------------------------
# Markdown 渲染边界情况 (consolidated)
# ---------------------------------------------------------------------------


class TestMarkdownEdgeCases:
    """Markdown 渲染的边界情况测试"""

    @pytest.fixture
    def project(self):
        return ProjectContext(
            project_id="edge_test", project_name="Edge Test",
            root_path="/tmp/edge", theme_color="orange", emoji_prefix="🟠",
        )

    @pytest.mark.parametrize("label,content,check", [
        ("empty", "", lambda e: e["content"] == ""),
        ("html_tags", "<script>alert('xss')</script>", lambda e: "<script>" in e["content"]),
        ("unicode_emoji", "✅ 成功 | ❌ 失败", lambda e: "✅ 成功" in e["content"]),
        ("special_chars", "价格 $100 | 100% | a_b_c", lambda e: e["content"] == "价格 $100 | 100% | a_b_c"),
        ("unclosed_code", "```python\ndef foo():\n    pass", lambda e: "def foo():" in e["content"]),
        ("only_symbols", "# ## ### * ** *** - --- > >> ``` ~~", lambda e: e["content"] == "# ## ### * ** *** - --- > >> ``` ~~"),
    ])
    def test_content_element_edge_cases(self, project, label, content, check):
        elem = CardBuilder._build_content_element(content)
        assert elem["tag"] == "markdown"
        assert check(elem), f"Failed check for '{label}'"

    def test_very_long_content_truncated(self, project):
        long_line = "A" * 5000
        content = f"开始\n{long_line}\n结束"
        elem = CardBuilder._build_content_element(content)
        assert elem["tag"] == "markdown"
        assert len(elem["content"]) < len(content)
        assert "日志内容过长，已被截断" in elem["content"]
        assert elem["content"].endswith("结束")

    def test_markdown_truncation_smart_closing(self):
        # Normal truncation
        content = "a" * 5000
        truncated = CardBuilder._truncate_markdown(content, 1000)
        assert len(truncated) <= 1000
        assert "日志内容过长" in truncated

        # Code block closure
        prefix = "a" * 3000
        code_content = "code" * 1000
        full = f"{prefix}```\n{code_content}\n```"
        truncated = CardBuilder._truncate_markdown(full, 1000)
        assert "```" in truncated
        assert truncated.count("```") % 2 == 0

        # Bold closure
        prefix = "a" * 3000
        bold_content = "bold" * 1000
        full = f"{prefix}**{bold_content}**"
        truncated = CardBuilder._truncate_markdown(full, 1000)
        assert truncated.count("**") % 2 == 0

    def test_directory_element_uses_markdown(self, project):
        elem = CardBuilder._build_directory_element(project)
        assert elem["tag"] == "markdown"
        assert f"`{project.root_path}`" in elem["content"]
        assert "📁" in elem["content"]

    def test_directory_element_variants(self):
        """No project defaults to ~, custom working_dir is respected."""
        elem_none = CardBuilder._build_directory_element(None)
        assert "`~`" in elem_none["content"]
        elem_custom = CardBuilder._build_directory_element(None, working_dir="/home/user/work")
        assert "`/home/user/work`" in elem_custom["content"]

    def test_full_card_json_serialization_integrity(self, project):
        special_content = '包含 "引号" 和 \\反斜杠 以及\n换行\t制表符'
        _, card_str = CardBuilder.build_project_response_card(project, "Title", special_content, show_buttons=False)
        card = json.loads(card_str)
        md_elems = [
            e for e in card["body"]["elements"] if e.get("tag") == "markdown" and "引号" in str(e.get("content", ""))
        ]
        assert len(md_elems) == 1
        content = md_elems[0]["content"]
        assert '包含 "引号"' in content
        assert "\n" in content

    def test_notification_card_no_suggestions(self, project):
        _, content = CardBuilder.build_notification_card(project, "info", "Info", "纯信息通知")
        card = json.loads(content)
        elements = card["body"]["elements"]
        suggestion_elems = [
            e for e in elements if e.get("tag") == "markdown" and "建议下一步" in str(e.get("content", ""))
        ]
        assert len(suggestion_elems) == 0

    def test_status_board_empty_uses_markdown_for_prompt(self):
        _, content = CardBuilder.build_status_board_card([], None)
        card = json.loads(content)
        elements = card["body"]["elements"]
        prompt_md = [e for e in elements if e.get("tag") == "markdown" and "暂无项目" in str(e.get("content", ""))]
        assert len(prompt_md) == 1
        assert "/new" in prompt_md[0]["content"]

    def test_deep_card_progress_bar_uses_markdown(self):
        _, content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(
                title="Test", content="Running",
                progress_bar="[████░░░░░░] 40%",
                show_buttons=False, engine_name="Coco",
            ),
        )
        card = json.loads(content)
        elements = card["body"]["elements"]
        progress_md = [e for e in elements if e.get("tag") == "markdown" and "📊" in str(e.get("content", ""))]
        assert len(progress_md) == 1
        assert "40%" in progress_md[0]["content"]


# ---------------------------------------------------------------------------
# 回复模式配置测试
# ---------------------------------------------------------------------------


class TestReplyModeConfig:
    def test_reply_mode_defaults_and_config(self):
        """Default values and custom setting."""
        settings = Settings()
        assert settings.smart_reply_mode == "direct"
        assert settings.default_reply_mode == "thread"

        assert Settings(smart_reply_mode="thread").smart_reply_mode == "thread"
        assert Settings(default_reply_mode="direct").default_reply_mode == "direct"


class TestBuildDeepCardStructuredParams:
    """Tests for build_info_card new structured parameters."""

    def test_structured_params_render_correctly(self):
        """duration_line, status_line, footer_note all render in card."""
        _, card_content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(
                title="Full Card", content="Main content",
                progress_bar="[████░░] 60% (3/5)", engine_name="Spec(Claude)",
                show_buttons=False, status_line="🔄 循环执行中",
                duration_line="⏱️ 3分钟12秒", criteria_section="📋 标准\n✅ C1\n🔲 C2",
                footer_note="Powered by Spec",
            ),
        )
        card = json.loads(card_content)
        elements = card["body"]["elements"]
        all_content = " ".join(str(e.get("content", "")) for e in elements)

        # Status/duration in notation elements
        notation_elems = [e for e in elements if e.get("text_size") == "notation"]
        assert any("循环执行中" in e["content"] and "3分钟12秒" in e["content"] for e in notation_elems)

        # Footer note
        footer_elems = [e for e in elements if "Powered by Spec" in str(e.get("content", ""))]
        assert len(footer_elems) == 1
        assert footer_elems[0].get("text_size") == "notation"

        # All content present
        for expected in ["标准", "Main content"]:
            assert expected in all_content


class TestTerminalAndFooterMarkers:
    """terminal_state markers, footer_status, stop button, and read marker tests."""

    def test_completed_marker(self):
        _, card_content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(
                title="Done", content="ok", engine_name="Coco",
                show_buttons=False, terminal_state="completed",
            ),
        )
        card = json.loads(card_content)
        last_md = [e for e in card["body"]["elements"] if e.get("tag") == "markdown"][-1]
        assert "✅" in last_md["content"]
        assert "已完成" in last_md["content"]

    def test_failed_marker(self):
        _, card_content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(
                title="Err", content="err", engine_name="Coco",
                show_buttons=False, terminal_state="failed",
            ),
        )
        card = json.loads(card_content)
        last_md = [e for e in card["body"]["elements"] if e.get("tag") == "markdown"][-1]
        assert "❌" in last_md["content"]

    def test_no_marker_when_none(self):
        _, card_content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(
                title="Running", content="running", engine_name="Coco",
                show_buttons=False, terminal_state=None,
            ),
        )
        card = json.loads(card_content)
        from src.card.terminal import TERMINAL_MARKERS
        all_text = " ".join(str(e.get("content", "")) for e in card["body"]["elements"])
        for marker in TERMINAL_MARKERS.values():
            assert marker not in all_text

    @pytest.mark.parametrize("footer_status,emoji,text", [
        ("thinking", "🧠", "正在思考"),
        ("tool_running", "🧰", "正在调用工具"),
    ])
    def test_footer_status_renders(self, footer_status, emoji, text):
        _, card_content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(
                title="Test", content="body", engine_name="Coco",
                is_executing=True, footer_status=footer_status,
            ),
        )
        card = json.loads(card_content)
        all_text = " ".join(str(e.get("content", "")) for e in card["body"]["elements"])
        assert emoji in all_text
        assert text in all_text

    def test_no_footer_when_none(self):
        _, card_content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(
                title="Test", content="body", engine_name="Coco",
                show_buttons=False, footer_status=None,
            ),
        )
        card = json.loads(card_content)
        from src.card.terminal import FOOTER_STATUS
        all_text = " ".join(str(e.get("content", "")) for e in card["body"]["elements"])
        for fs in FOOTER_STATUS.values():
            assert fs not in all_text

    @pytest.mark.parametrize("is_executing,is_paused", [(True, False), (False, True)])
    def test_stop_danger_appears(self, is_executing, is_paused):
        _, card_content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(
                title="T", content="c", engine_name="Coco",
                is_executing=is_executing, is_paused=is_paused,
            ),
        )
        card = json.loads(card_content)
        all_text = json.dumps(card, ensure_ascii=False)
        assert "⛔ 强制停止" in all_text

    def test_unread_marker_prefix(self):
        _, card_content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(
                title="New Task", content="body", engine_name="Coco",
                show_buttons=False, is_read=False,
            ),
        )
        card = json.loads(card_content)
        header_title = card["header"]["title"]["content"]
        assert header_title.startswith("🔴 ")

    def test_read_no_marker(self):
        _, card_content = CardBuilder.build_info_card(
            project=None,
            state=EngineCardState(
                title="Task", content="body", engine_name="Coco",
                show_buttons=False, is_read=True,
            ),
        )
        card = json.loads(card_content)
        header_title = card["header"]["title"]["content"]
        assert not header_title.startswith("🔴")

    def test_default_is_read(self):
        state = EngineCardState()
        assert state.is_read is True
