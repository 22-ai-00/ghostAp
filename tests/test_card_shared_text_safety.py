from __future__ import annotations

import pytest

from src.card.delivery.page_mutator import (
    sanitize_card_text_for_audit as delivery_sanitize_card_text_for_audit,
)
from src.card.delivery.page_mutator import (
    sanitize_markdown_image_references as delivery_sanitize_markdown_images,
)
from src.card.shared.text_safety import (
    neutralize_feishu_rich_text_controls,
    sanitize_card_text_for_audit,
    sanitize_markdown_image_references,
)
from src.card.shared.truncation import (
    check_and_truncate_payload,
    count_markdown_table_blocks,
    normalize_markdown_tables_for_card,
)

_TABLE_WARNING = (
    "⚠️ 表格数量超过飞书卡片限制，已将 Markdown 表格按代码块展示，避免卡片发送失败。"
)


def _table(index: int) -> str:
    return f"| 标题 {index} |\n| --- |\n| 内容 {index} |"


def test_delivery_module_preserves_text_safety_compatibility_exports() -> None:
    assert delivery_sanitize_card_text_for_audit is sanitize_card_text_for_audit
    assert delivery_sanitize_markdown_images is sanitize_markdown_image_references


def test_audit_and_markdown_image_sanitizers_keep_existing_behavior() -> None:
    assert sanitize_card_text_for_audit("联系 alice@example.com。") == (
        "联系 [redacted:email]。"
    )
    assert sanitize_markdown_image_references(
        "前 ![不应泄漏的 alt](https://example.test/a_(b).png) 后"
    ) == "前 （图片引用已移除） 后"
    assert sanitize_markdown_image_references("损坏 ![alt") == "损坏 ！[alt"


def test_rich_text_controls_are_inert_but_inner_text_remains_readable() -> None:
    source = (
        "通知 <at id=all></at> "
        "<font color='red'>危险</font> "
        "<a href='https://example.test'>点我</a>；1 < 2 > 0"
    )

    result = neutralize_feishu_rich_text_controls(source)

    assert "＜at id=all＞＜/at＞" in result
    assert "＜font color='red'＞危险＜/font＞" in result
    assert "＜a href='https://example.test'＞点我＜/a＞" in result
    assert "1 < 2 > 0" in result
    assert neutralize_feishu_rich_text_controls(result) == result


def test_rich_text_control_attributes_may_contain_closing_brackets() -> None:
    source = '''<a title="1 > 0" data-note='x > y'>点我</a>'''

    assert neutralize_feishu_rich_text_controls(source) == (
        '''＜a title="1 ＞ 0" data-note='x ＞ y'＞点我＜/a＞'''
    )


@pytest.mark.timeout(5)
def test_rich_text_control_scan_is_bounded_for_unclosed_attribute_input() -> None:
    source = "<a " + "attribute=value " * 50_000

    assert neutralize_feishu_rich_text_controls(source) == source


def test_plain_comparisons_are_not_mistaken_for_rich_text_tags() -> None:
    source = "x<y and y>z；a < b；c > d"

    assert neutralize_feishu_rich_text_controls(source) == source


@pytest.mark.timeout(5)
def test_markdown_link_scan_is_bounded_for_many_unclosed_candidates() -> None:
    source = "[" * 50_000
    source = f"{source}]("

    assert neutralize_feishu_rich_text_controls(source) == source


@pytest.mark.timeout(5)
def test_markdown_image_scan_is_bounded_for_many_unclosed_candidates() -> None:
    source = "![x](" * 50_000

    result = sanitize_markdown_image_references(source)

    assert result.count("！[") == 50_000
    assert "![" not in result


def test_non_http_markdown_links_become_labels_while_http_links_remain() -> None:
    source = (
        "[官网](https://example.test/a_(b)) "
        "[明文](http://example.test) "
        "[脚本](JaVaScRiPt:alert(1)) "
        "[数据](data:text/html;base64,AAAA) "
        "[文件](file:///tmp/report) "
        "[邮件](mailto:user@example.test) "
        "[相对路径](/admin)"
    )

    result = neutralize_feishu_rich_text_controls(source)

    assert "[官网](https://example.test/a_(b))" in result
    assert "[明文](http://example.test)" in result
    assert "脚本" in result and "javascript:" not in result.lower()
    assert "数据" in result and "data:" not in result.lower()
    assert "文件" in result and "file:" not in result.lower()
    assert "邮件" in result and "mailto:" not in result.lower()
    assert "相对路径" in result and "](/admin)" not in result


def test_non_http_markdown_autolinks_are_neutralized() -> None:
    source = "<https://example.test> <javascript:alert(1)> <file:///tmp/a>"

    assert neutralize_feishu_rich_text_controls(source) == (
        "<https://example.test> javascript:alert(1) file:///tmp/a"
    )


def test_internal_redaction_sentinels_remain_readable() -> None:
    source = "<redacted> <redacted:api_key> <redacted:agent_path>"

    assert neutralize_feishu_rich_text_controls(source) == source


def test_markdown_table_normalization_is_noop_at_limit() -> None:
    source = "\n\n".join(_table(index) for index in range(5))

    assert count_markdown_table_blocks(source) == 5
    assert normalize_markdown_tables_for_card(source) == source


def test_markdown_table_normalization_rewrites_over_limit_once() -> None:
    source = "\n\n".join(_table(index) for index in range(6))

    result = normalize_markdown_tables_for_card(source)

    assert result.count("```text") == 6
    assert result.count(_TABLE_WARNING) == 1
    assert count_markdown_table_blocks(result) == 0
    assert normalize_markdown_tables_for_card(result) == result


def test_markdown_table_normalization_supports_cross_value_counting() -> None:
    source = _table(1)

    assert normalize_markdown_tables_for_card(
        source,
        table_limit=0,
        include_warning=False,
    ) == f"```text\n{source}\n```"


def test_fenced_table_text_does_not_contribute_to_normalization_limit() -> None:
    source = "\n\n".join(
        ["```markdown", _table(99), "```", *(_table(index) for index in range(5))]
    )

    assert count_markdown_table_blocks(source) == 5
    assert normalize_markdown_tables_for_card(source) == source


def test_table_scanner_requires_a_matching_complete_fence_closer() -> None:
    fenced = "\n\n".join(
        [
            "````markdown",
            _table(90),
            "```",  # Shorter backtick run is not a closer.
            _table(91),
            "~~~",  # A different fence character is not a closer.
            _table(92),
            "```` still code",  # A closer cannot have non-whitespace tail text.
            _table(93),
            "`````   ",  # A longer matching run with whitespace is a closer.
        ]
    )
    source = "\n\n".join([fenced, *(_table(index) for index in range(6))])

    assert count_markdown_table_blocks(source) == 6

    result = normalize_markdown_tables_for_card(source)

    assert result.count("```text") == 6
    assert result.count(_TABLE_WARNING) == 1
    assert count_markdown_table_blocks(result) == 0


def test_payload_guard_rechecks_size_after_table_rewrite() -> None:
    import json

    source = "\n\n".join(_table(index) for index in range(1_000))
    payload = json.dumps(
        {
            "schema": "2.0",
            "body": {"elements": [{"tag": "markdown", "content": source}]},
        },
        ensure_ascii=False,
    )

    guarded = check_and_truncate_payload(payload)

    assert len(guarded.encode("utf-8")) <= 27 * 1024
