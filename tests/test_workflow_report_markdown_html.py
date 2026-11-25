"""Tests for the zero-dependency GFM Markdown-to-HTML converter."""

from __future__ import annotations

import re

from src.workflow_engine.report_markdown_html import (
    looks_like_markdown,
    render_markdown_to_html,
)


class TestXssSafety:
    """Ensure untrusted raw HTML and dangerous URLs are strictly neutralized."""

    def test_raw_html_tags_and_attributes_are_escaped_everywhere(self):
        md = (
            "# Heading <script>alert('h1')</script>\n\n"
            "Paragraph with <img src=x onerror=alert(1)> and <b>raw bold</b> and <svg onload=alert(2)>.\n\n"
            "> [!WARNING]\n"
            "> Alert with <iframe src=javascript:alert(1)></iframe>\n\n"
            "| Col <script>x</script> | Safe |\n"
            "| --- | --- |\n"
            "| <img src=x onerror=alert(3)> | **<script>y</script>** |\n\n"
            "- Item <div onclick=\"alert(4)\">x</div>\n"
            "- [x] Task <script>z</script>\n\n"
            "```html<script>alert('lang')</script>\n"
            "<script>alert('code')</script>\n"
            "```\n"
        )
        out = render_markdown_to_html(md)

        for dangerous in (
            "<script",
            "<img",
            "<svg",
            "<iframe",
            "<div onclick",
            "<b>",
        ):
            assert dangerous not in out, f"Found unescaped HTML {dangerous!r} in:\n{out}"

        # Every HTML tag in output must be a safe renderer-emitted tag without event handlers.
        allowed_tags = {
            "h1", "h2", "h3", "h4", "h5", "h6",
            "div", "pre", "code", "span",
            "table", "thead", "tbody", "tr", "th", "td",
            "blockquote", "ul", "ol", "li", "p",
            "strong", "em", "del", "a", "br", "hr", "input",
        }
        for tag_match in re.finditer(r"</?([a-zA-Z0-9]+)([^>]*)>", out):
            tag_name = tag_match.group(1).lower()
            attrs = tag_match.group(2)
            assert tag_name in allowed_tags, f"Unexpected HTML tag {tag_match.group(0)!r}"
            assert not re.search(r"\bon[a-z]+\s*=", attrs, re.I), f"Event handler leaked: {tag_match.group(0)!r}"

        assert "&lt;script&gt;alert(&#x27;h1&#x27;)&lt;/script&gt;" in out
        assert "&lt;img src=x onerror=alert(1)&gt;" in out
        assert "&lt;b&gt;raw bold&lt;/b&gt;" in out
        assert "&lt;svg onload=alert(2)&gt;" in out
        assert "&lt;script&gt;alert(&#x27;lang&#x27;)&lt;/script&gt;" in out
        assert "&lt;script&gt;alert(&#x27;code&#x27;)&lt;/script&gt;" in out

    def test_unsafe_link_schemes_are_rejected_and_rendered_as_escaped_text(self):
        cases = [
            ("[click](javascript:alert(1))", "click"),
            ("[upper](JaVaScRiPt:alert(1))", "upper"),
            ("[tabbed](java\tscript:alert(1))", "tabbed"),
            ("[spaced](  javascript:alert(1))", "spaced"),
            ("[entity](&#106;avascript:alert(1))", "entity"),
            ("[hex](&#x6a;avascript:alert(1))", "hex"),
            ("[data](data:text/html,hello)", "data"),
            ("[data-tag](data:text/html,<script>alert(1)</script>)", "data-tag"),
            ("[vbs](vbscript:msgbox(1))", "vbs"),
            ("[file](file:///etc/passwd)", "file"),
            ("[proto-rel](//evil.example/phish)", "proto-rel"),
            ("[<script>bad</script>](javascript:alert(1))", "&lt;script&gt;bad&lt;/script&gt;"),
            ("![img](javascript:alert(1))", "img"),
            ("![svg](data:image/svg+xml,<svg onload=alert(1)>)", "svg"),
        ]
        for md, expected_text in cases:
            out = render_markdown_to_html(md)
            assert "<a " not in out, f"Unexpected <a> tag for {md!r}: {out}"
            assert "<img" not in out, f"Unexpected <img> tag for {md!r}: {out}"
            assert "<script" not in out, f"Unexpected <script> tag for {md!r}: {out}"
            assert "<svg" not in out, f"Unexpected <svg> tag for {md!r}: {out}"
            assert "href=" not in out, f"Unexpected href for {md!r}: {out}"
            assert expected_text in out, f"Missing escaped label for {md!r}: {out}"

    def test_safe_links_and_attribute_escaping(self):
        md = (
            "[HTTP](http://localhost:8080/healthz) "
            "[HTTPS](https://example.com/a?x=1&y=2#top) "
            "[Wiki](https://en.wikipedia.org/wiki/Function_(mathematics)) "
            "[Mail](mailto:dev@example.com) "
            "[Anchor](#section-1) "
            "[Doc](docs/foo.md) "
            '[Titled](https://example.com/p "Say \\"hi\\" & <go>")'
        )
        out = render_markdown_to_html(md)
        assert '<a href="http://localhost:8080/healthz" target="_blank" rel="noopener noreferrer">HTTP</a>' in out
        assert (
            '<a href="https://example.com/a?x=1&amp;y=2#top" target="_blank" rel="noopener noreferrer">HTTPS</a>'
            in out
        )
        assert (
            '<a href="https://en.wikipedia.org/wiki/Function_(mathematics)" '
            'target="_blank" rel="noopener noreferrer">Wiki</a>' in out
        )
        assert '<a href="mailto:dev@example.com">Mail</a>' in out
        assert '<a href="#section-1">Anchor</a>' in out
        assert '<a href="docs/foo.md">Doc</a>' in out
        assert (
            '<a href="https://example.com/p" title="Say &quot;hi&quot; &amp; &lt;go&gt;" '
            'target="_blank" rel="noopener noreferrer">Titled</a>' in out
        )

    def test_null_bytes_cannot_forge_internal_placeholders(self):
        out = render_markdown_to_html("Hello \x00C0\x00 <script>alert(1)</script>")
        assert "<script>" not in out
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out


class TestBlockElements:
    """Verify headings, fences, tables, blockquotes/alerts, lists, and HRs."""

    def test_atx_headings_h1_through_h6(self):
        md = "\n".join(f"{'#' * i} Heading {i} ##" for i in range(1, 7))
        out = render_markdown_to_html(md)
        for i in range(1, 7):
            assert f'<h{i} class="md-h{i}">Heading {i}</h{i}>' in out

        # `#1` without whitespace is not a heading.
        non_heading = render_markdown_to_html("#123 issue")
        assert "<h1" not in non_heading
        assert "<p>#123 issue</p>" == non_heading

    def test_fenced_code_blocks_arbitrary_backticks_and_tildes(self):
        md = (
            "````markdown\n"
            "# Outer\n"
            "```python\n"
            "print('**nested**')\n"
            "```\n"
            "````\n\n"
            "~~~bash\n"
            "echo \"<ok>\"\n"
            "~~~\n\n"
            "```\n"
            "plain fence\n"
            "```"
        )
        out = render_markdown_to_html(md)
        assert '<div class="md-code-block"><div class="md-code-header"><span class="md-code-lang">markdown</span></div>' in out
        assert "```python\nprint(&#x27;**nested**&#x27;)\n```" in out
        assert '<div class="md-code-block"><div class="md-code-header"><span class="md-code-lang">bash</span></div>' in out
        assert "echo &quot;&lt;ok&gt;&quot;" in out
        assert '<div class="md-code-block"><pre><code>plain fence</code></pre></div>' in out

    def test_gfm_pipe_table_alignments_escaped_pipes_backticks_and_inline(self):
        md = (
            "| Method | Path & `Pipe` | Notes |\n"
            "| :--- | :---: | ---: |\n"
            "| `POST /v1/chat` | `a | b` | a \\| b & **bold** [doc](docs/api.md) |\n"
            "| *GET* | ~~old~~ |"
        )
        out = render_markdown_to_html(md)
        assert out.startswith('<div class="md-table-wrap"><table class="md-table">')
        assert out.endswith("</table></div>")
        assert '<th style="text-align:left">Method</th>' in out
        assert '<th style="text-align:center">Path &amp; <code class="md-inline-code">Pipe</code></th>' in out
        assert '<th style="text-align:right">Notes</th>' in out
        assert '<td style="text-align:left"><code class="md-inline-code">POST /v1/chat</code></td>' in out
        assert '<td style="text-align:center"><code class="md-inline-code">a | b</code></td>' in out
        assert (
            '<td style="text-align:right">a | b &amp; <strong>bold</strong> <a href="docs/api.md">doc</a></td>'
            in out
        )
        assert '<td style="text-align:left"><em>GET</em></td>' in out
        assert '<td style="text-align:center"><del>old</del></td>' in out
        assert '<td style="text-align:right"></td>' in out

    def test_blockquotes_and_all_five_github_alerts(self):
        quote = render_markdown_to_html("> Line one\n> Line two\n>\n> - Item")
        assert '<blockquote class="md-quote">' in quote
        assert "<p>Line one\nLine two</p>" in quote
        assert '<ul class="md-list">\n<li>Item</li>\n</ul>' in quote

        for kind in ("NOTE", "TIP", "IMPORTANT", "WARNING", "CAUTION"):
            alert_md = f"> [!{kind}]\n> Alert **body** with `code`"
            alert_out = render_markdown_to_html(alert_md)
            cls = f"md-alert-{kind.lower()}"
            assert f'<div class="md-alert {cls}">' in alert_out
            assert "<strong>body</strong>" in alert_out
            assert '<code class="md-inline-code">code</code>' in alert_out

    def test_nested_lists_ordered_lists_and_task_checkboxes(self):
        md = (
            "- Parent A\n"
            "  - Child 1\n"
            "  * Child 2\n"
            "    1. Sub-step 1\n"
            "    2. Sub-step 2\n"
            "- Parent B\n\n"
            "3. Third\n"
            "4. Fourth\n"
            "  - Two-space nested bullet\n\n"
            "- [x] Completed `task`\n"
            "- [ ] Pending **task**\n"
            "- [link](docs/ok.md) not a checkbox"
        )
        out = render_markdown_to_html(md)
        assert '<ul class="md-list">' in out
        assert '<ol class="md-list">' in out
        assert '<ol class="md-list" start="3">' in out
        assert "Two-space nested bullet" in out
        assert '<ul class="md-list md-task-list">' in out
        assert '<li class="md-task-item"><input type="checkbox" checked disabled> Completed' in out
        assert '<li class="md-task-item"><input type="checkbox" disabled> Pending <strong>task</strong></li>' in out
        assert '<li><a href="docs/ok.md">link</a> not a checkbox</li>' in out

    def test_horizontal_rules_and_line_breaks(self):
        md = "Line 1  \nLine 2\\\nLine 3\n\n---\n\n***\n\n___"
        out = render_markdown_to_html(md)
        assert out.count('<hr class="md-hr">') == 3
        assert "<p>Line 1<br>\nLine 2<br>\nLine 3</p>" in out


class TestInlineFormatting:
    """Verify emphasis, strikethrough, inline code isolation, and underscores."""

    def test_inline_code_preserves_markdown_syntax_literally(self):
        md = "Use `**not bold**`, `*not italic*`, `~~not strike~~`, `` `nested` ``, and `foo_bar_baz`."
        out = render_markdown_to_html(md)
        assert '<code class="md-inline-code">**not bold**</code>' in out
        assert '<code class="md-inline-code">*not italic*</code>' in out
        assert '<code class="md-inline-code">~~not strike~~</code>' in out
        assert '<code class="md-inline-code">`nested`</code>' in out
        assert '<code class="md-inline-code">foo_bar_baz</code>' in out
        assert "<strong>" not in out
        assert "<em>" not in out
        assert "<del>" not in out

    def test_identifier_underscores_stay_intact_while_word_boundary_emphasis_works(self):
        md = (
            "Keep foo_bar_baz and oauth_scope_executor and a_b_c intact, "
            "even two snake_case_one and snake_case_two on one line, "
            "while _italic_, __bold__, *star italic*, **star bold**, "
            "***both***, **a *b***, ***a** b*, and ~~removed~~ format cleanly."
        )
        out = render_markdown_to_html(md)
        for ident in ("foo_bar_baz", "oauth_scope_executor", "a_b_c", "snake_case_one", "snake_case_two"):
            assert ident in out
        assert "<em>italic</em>" in out
        assert "<strong>bold</strong>" in out
        assert "<em>star italic</em>" in out
        assert "<strong>star bold</strong>" in out
        assert "<em><strong>both</strong></em>" in out
        assert "<strong>a <em>b</em></strong>" in out
        assert "<em><strong>a</strong> b</em>" in out
        assert "<del>removed</del>" in out

    def test_paths_with_glob_stars_do_not_falsely_italicize(self):
        md = (
            "Routes `/v1/videos*`, /v1/videos*, /v1/realtime*, "
            "/v1beta/models/*action, /v1/models/*model, and 2 * 3 * 4 stay clean."
        )
        out = render_markdown_to_html(md)
        assert "<em>" not in out
        assert '<code class="md-inline-code">/v1/videos*</code>' in out
        assert "<em>" not in render_markdown_to_html("2*(3+4)*5")
        assert looks_like_markdown("a/*.py and b/*.md") is False

    def test_links_with_inline_formatting_and_nested_brackets(self):
        md = "See [**bold** and `code` and [arr[0]]](https://example.com/path) and _hello_world_."
        out = render_markdown_to_html(md)
        assert (
            '<a href="https://example.com/path" target="_blank" rel="noopener noreferrer">'
            '<strong>bold</strong> and <code class="md-inline-code">code</code> and [arr[0]]</a>'
        ) in out
        assert "<em>hello_world</em>" in out


class TestLooksLikeMarkdownAndRealReportSamples:
    """Verify looks_like_markdown heuristics and real CLIProxyAPI report snippets."""

    def test_looks_like_markdown_distinguishes_markdown_from_plain_text(self):
        positives = [
            "# Heading",
            "## 二、上游接入方式",
            "```python\nprint(1)\n```",
            "| A | B |\n| --- | --- |\n| 1 | 2 |",
            "- bullet item",
            "1. ordered item",
            "> [!NOTE]\n> Heads up",
            "> Quoted line",
            "Use `POST /v1/chat/completions` here",
            "This is **important** news",
            "This is __bold__ text",
            "This is ~~deprecated~~ now",
            "See [docs](docs/foo.md)",
        ]
        for sample in positives:
            assert looks_like_markdown(sample) is True, f"Expected True for {sample!r}"

        negatives = [
            "",
            "   \n  ",
            "Just a normal status sentence.",
            "foo_bar_baz and snake_case_identifier",
            "Issue #42: latency > 100ms | usage|errors",
            "-10 degrees and 2 * 3 = 6",
            "version 1.2.3 released",
            "❌ Claude 空闲超时",
            '{"key": "value", "count": 3}',
        ]
        for sample in negatives:
            assert looks_like_markdown(sample) is False, f"Expected False for {sample!r}"

    def test_real_cliproxyapi_verified_output_and_conclusion_render_cleanly(self):
        conclusion_sample = (
            "当前本地仓库（分支 feat/kiro-provider，提交 40c3289e）同时提供丰富的下游协议接入：\n"
            "1. 下游协议入口：覆盖 OpenAI（`/openai/v1/videos*` 兼容适配）、"
            "Anthropic Messages（`/v1/messages`、`/v1/messages/count_tokens`）。\n"
            "2. 上游接入方式：支持 `--siwc-bind`（SIWC 配置位于 `oauth.providers.codex`），"
            "实际注册执行器为 `aistudio`、`kiro`、`oauth_scope_executor`（非独立执行器）。\n"
            "3. 动态插件扩展：启用 `plugins.enabled` 后可通过 C ABI 插件扩展。"
        )
        verified_table_sample = (
            "## 一、下游接入方式（客户端 → 本地代理）\n\n"
            "| 接入方式 | 入口 / 路径 | 代码证据 |\n"
            "|---|---|---|\n"
            "| OpenAI Chat Completions | `POST /v1/chat/completions`、`POST /v1/completions` | "
            "`internal/api/server_routes.go:74-75` |\n"
            "| Claude Code / Anthropic Messages | `POST /v1/messages`、`POST /v1/messages/count_tokens` | "
            "`server_routes.go:85-86` |\n\n"
            "---\n\n"
            "**2) 账号 / 密钥导入（无交互）** — `cmd/server/main.go:157-160`\n\n"
            "- `--vertex-import <json>` — Vertex 服务账号\n"
            "- `--kiro-import <json>` — **本分支新增**，Kiro 账号\n"
        )

        assert looks_like_markdown(conclusion_sample) is True
        assert looks_like_markdown(verified_table_sample) is True

        conclusion_html = render_markdown_to_html(conclusion_sample)
        assert '<ol class="md-list">' in conclusion_html
        assert '<code class="md-inline-code">/v1/messages</code>' in conclusion_html
        assert "oauth_scope_executor" in conclusion_html
        assert "<em>" not in conclusion_html

        table_html = render_markdown_to_html(verified_table_sample)
        assert '<h2 class="md-h2">一、下游接入方式（客户端 → 本地代理）</h2>' in table_html
        assert '<div class="md-table-wrap"><table class="md-table">' in table_html
        assert '<code class="md-inline-code">POST /v1/chat/completions</code>' in table_html
        assert '<hr class="md-hr">' in table_html
        assert "<strong>2) 账号 / 密钥导入（无交互）</strong>" in table_html
        assert "&lt;json&gt;" in table_html
        assert "<json>" not in table_html
        assert not re.search(r"\|---\|", table_html)
