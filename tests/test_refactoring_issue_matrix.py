from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / ".Memory" / "2026-05-11.md"
TODAY_MEMORY_PATH = ROOT / ".Memory" / "2026-05-12.md"
ABSTRACT_PATH = ROOT / ".Memory" / "Abstract.md"
BACKLOG_PATH = ROOT / ".Memory" / "Backlog.md"

VALID_STATUSES = {"存在", "不存在", "已被其他改动解决"}
FORBIDDEN_STATUS_VALUES = {"已修复并验证", "确认不存在", "保留兼容入口"}
OPEN_ITEM_SIGNALS = (
    "仍存在并纳入 Backlog",
    "纳入 Backlog",
    "转入 Backlog",
    "后续处理",
    "待处理",
    "未修复",
    "未闭环",
)
REQUIRED_COLUMNS = [
    "#",
    "问题摘要",
    "状态",
    "处理方式",
    "涉及范围",
    "验证依据",
    "用户可验证结果",
]


def _parse_matrix() -> list[dict[str, str]]:
    text = MATRIX_PATH.read_text(encoding="utf-8")
    marker = "### Refactoring Analysis 1–28 最终状态矩阵"
    assert marker in text, "缺少 Refactoring Analysis 1–28 最终状态矩阵"
    section = text.split(marker, 1)[1]
    rows: list[dict[str, str]] = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells == REQUIRED_COLUMNS or set(cells) == {"---"}:
            continue
        if len(cells) != len(REQUIRED_COLUMNS):
            continue
        rows.append(dict(zip(REQUIRED_COLUMNS, cells)))
    return rows


def test_refactoring_issue_matrix_covers_1_to_28_without_gaps():
    rows = _parse_matrix()

    numbers = [int(row["#"]) for row in rows]

    assert numbers == list(range(1, 29))


def test_refactoring_issue_matrix_has_valid_status_and_required_fields():
    rows = _parse_matrix()

    assert VALID_STATUSES == {"存在", "不存在", "已被其他改动解决"}
    for row in rows:
        assert row["状态"] in VALID_STATUSES
        assert row["状态"] not in FORBIDDEN_STATUS_VALUES
        for column in REQUIRED_COLUMNS:
            assert row[column], f"问题 #{row['#']} 缺少字段 {column}"


def test_refactoring_issue_matrix_is_single_user_readable_matrix():
    text = MATRIX_PATH.read_text(encoding="utf-8")

    assert "### Refactoring Analysis 1–28 用户级闭环说明" not in text
    rows = _parse_matrix()
    assert len(rows) == 28
    for row in rows:
        for column in ("状态", "处理方式", "涉及范围", "验证依据", "用户可验证结果"):
            assert row[column].strip(), f"问题 #{row['#']} 最终矩阵缺少 {column}"


def test_refactoring_issue_matrix_rejects_second_status_vocabulary():
    rows = _parse_matrix()
    invalid_rows = [
        f"#{row['#']}={row['状态']}"
        for row in rows
        if row["状态"] in FORBIDDEN_STATUS_VALUES
    ]

    assert invalid_rows == []


def test_refactoring_issue_matrix_has_no_existing_items_for_final_acceptance():
    rows = _parse_matrix()

    open_rows = []
    for row in rows:
        closure_text = "；".join(
            [
                row["状态"],
                row["处理方式"],
                row["验证依据"],
                row["用户可验证结果"],
            ]
        )
        if any(signal in closure_text for signal in OPEN_ITEM_SIGNALS):
            open_rows.append(f"#{row['#']} {row['问题摘要']}")

    assert open_rows == [], f"最终验收不允许未闭环项继续挂 Backlog 或留待后续处理: {open_rows}"


def test_refactoring_issue_matrix_does_not_leave_fixed_items_in_backlog():
    rows = _parse_matrix()
    backlog_text = BACKLOG_PATH.read_text(encoding="utf-8")

    closed_numbers = [
        row["#"]
        for row in rows
        if row["状态"] in VALID_STATUSES
    ]

    for number in closed_numbers:
        assert f"refactoring-analysis #{number}" not in backlog_text


def test_refactoring_issue_matrix_has_user_facing_closure_notes():
    rows = _parse_matrix()

    internal_only_markers = (
        " tests",
        " test",
        "tests/",
        "import guard",
        "ACP startup utils",
        "manager consistency",
        "UI text consistency",
        "card builder tests",
        "ACP model tests",
        "新增或调整命令意图时",
        "类型绕过不再集中堆积",
        "新文案有清晰归属",
        "helper",
        "facade",
        "shim",
        "import",
    )

    for row in rows:
        user_result = row["用户可验证结果"].strip()
        assert len(user_result) >= 12, f"第 {row['#']} 项用户可验证结果过短，无法供普通用户判断"
        assert not any(marker in user_result for marker in internal_only_markers), (
            f"第 {row['#']} 项用户可验证结果不应只依赖内部测试/模块术语: {user_result}"
        )


def test_refactoring_issue_matrix_treatment_starts_with_final_decision():
    rows = _parse_matrix()
    allowed_prefixes = ("已修复", "兼容闭环", "保留兼容", "无需修复")

    for row in rows:
        treatment = row["处理方式"].strip()
        assert treatment.startswith(allowed_prefixes), (
            f"问题 #{row['#']} 处理方式首句必须以明确结论开头: {treatment}"
        )


def test_refactoring_issue_memory_frontloads_final_acceptance_entry():
    text = MATRIX_PATH.read_text(encoding="utf-8")
    overview_marker = "### Refactoring Analysis 1–28 最终验收总览"
    acceptance_marker = "### Refactoring Analysis 1–28 最终状态矩阵"
    task_marker = "### 任务描述"

    assert "实际验收文件路径为 `docs/2026-05-11-refactoring-analysis.md`，以下均以该文件为准" in text
    assert "本文件内最终总览、状态矩阵与验证记录共同构成本次任务的问题矩阵入口" in text
    assert "本文件内最终矩阵与本节验证记录共同构成本次任务的问题矩阵入口" in text
    assert "唯一最终验证命令与结果" in text
    assert "uv run python -m pytest tests/ -q" in text
    assert "6333 passed, 49 warnings" in text
    assert "6325 passed, 49 warnings" not in text
    assert "6301 passed, 49 warnings" not in text
    assert "以 2026-05-12 记录" not in text
    assert text.index(overview_marker) < text.index(acceptance_marker)
    assert text.index(acceptance_marker) < text.index(task_marker)


def test_refactoring_issue_memory_has_low_density_acceptance_overview():
    text = MATRIX_PATH.read_text(encoding="utf-8")
    overview = text.split("### Refactoring Analysis 1–28 最终验收总览", 1)[1].split(
        "### Refactoring Analysis 1–28 最终状态矩阵", 1
    )[0]

    assert "| 已修复 | 21 |" in overview
    assert "| 兼容闭环 | 5 |" in overview
    assert "| 无需修复 / 已由其他改动解决 | 2 |" in overview
    assert "用户侧已闭环" in overview


def test_refactoring_issue_matrix_explains_original_status_vs_current_closure():
    text = MATRIX_PATH.read_text(encoding="utf-8")

    assert "“状态”仅表示原始审计结论是否成立" in text
    assert "“处理方式”和“用户可验证结果”才表示当前是否已闭环" in text
    assert "不表示问题仍未处理" in text
    assert "用户输入的 `doc/` 为路径口径差异" not in text


def test_compatibility_entries_have_user_level_contract():
    rows = _parse_matrix()

    compatibility_rows = [
        row
        for row in rows
        if any(token in row["处理方式"] + row["用户可验证结果"] for token in ("兼容", "shim", "旧导入", "旧调用"))
    ]
    compatibility_numbers = {row["#"] for row in compatibility_rows}
    assert {"11", "21", "23", "24", "27"}.issubset(compatibility_numbers)

    required_compat_numbers = {"11", "21", "23", "24", "27"}
    for row in rows:
        if row["#"] in required_compat_numbers:
            assert row["处理方式"].startswith("兼容闭环"), f"#{row['#']} 必须使用克制的兼容闭环表达"
            assert "用户侧已闭环" in row["用户可验证结果"]

    for row in compatibility_rows:
        text = "；".join([row["处理方式"], row["用户可验证结果"]])
        assert any(token in text for token in ("迁移", "旧", "兼容", "长期公开 API 不计划删除", "2026-06-01")), (
            f"兼容入口 #{row['#']} 必须给出普通用户可理解的迁移或保留说明"
        )


def test_refactoring_issue_memory_has_no_conflicting_final_narrative():
    final_texts = [
        MATRIX_PATH.read_text(encoding="utf-8"),
        TODAY_MEMORY_PATH.read_text(encoding="utf-8"),
        BACKLOG_PATH.read_text(encoding="utf-8"),
        ABSTRACT_PATH.read_text(encoding="utf-8"),
    ]
    forbidden_phrases = [
        "以 2026-05-12 记录为准",
        "最终矩阵已在 2026-05-12 更新",
        "2026-05-12 最终验收中完成收口",
        "27 条“已被其他改动解决”",
        "所有原“存在”项更新为闭环状态",
        "阶段性治理",
        "阶段性闭环记录",
        "首批拆解",
        "首批容错边界",
        "首批修复",
        "第一阶段",
        "长期公开兼容入口，删除前需单独公告",
        "登记到 Backlog B019-B041",
        "仍存在并纳入 Backlog",
        "仍存在但又已完全闭环",
        "全部闭环",
        "Backlog 全清理",
        "B019-B041 已在 2026-05-12 重构闭环中修复并清理",
    ]

    for path, text in zip((MATRIX_PATH, TODAY_MEMORY_PATH, BACKLOG_PATH, ABSTRACT_PATH), final_texts):
        for phrase in forbidden_phrases:
            assert phrase not in text, f"{path} 仍残留冲突叙事: {phrase}"


def test_2026_05_12_memory_is_supplement_not_acceptance_entry():
    text = TODAY_MEMORY_PATH.read_text(encoding="utf-8")

    assert "执行日志与验证补充" in text
    assert "2026-05-11 记录是问题矩阵入口，2026-05-12 记录是执行验证日志" in text
    assert "## Refactoring Analysis 最终验收闭环" not in text


def test_backlog_and_matrix_have_bidirectional_truth_for_open_items():
    rows = _parse_matrix()
    backlog_text = BACKLOG_PATH.read_text(encoding="utf-8")
    open_numbers = {
        row["#"]
        for row in rows
        if any(
            signal in "；".join([row["处理方式"], row["验证依据"], row["用户可验证结果"]])
            for signal in OPEN_ITEM_SIGNALS
        )
    }

    assert open_numbers == set()

    for number in range(1, 29):
        needle = f"refactoring-analysis #{number}"
        assert needle not in backlog_text


def test_abstract_uses_single_final_acceptance_voice():
    text = ABSTRACT_PATH.read_text(encoding="utf-8")
    forbidden_phrases = [
        "首批修复",
        "阶段收口",
        "阶段性闭环记录",
        "阶段性治理",
        "6160 passed",
        "6177 passed",
        "6234 passed",
    ]

    for phrase in forbidden_phrases:
        assert phrase not in text, f"Abstract 仍残留非最终口径: {phrase}"
    assert "2026-05-11.md](2026-05-11.md) 顶部最终矩阵" in text
    assert text.count("Refactoring Analysis 最终验收入口") == 1
    assert "最终全量验证为 `uv run python -m pytest tests/ -q` → `6333 passed, 49 warnings`" in text
    assert "最终矩阵已在 2026-05-12 更新" not in text


def test_memory_files_use_one_final_full_regression_result():
    texts = {
        "2026-05-11": MATRIX_PATH.read_text(encoding="utf-8"),
        "2026-05-12": TODAY_MEMORY_PATH.read_text(encoding="utf-8"),
        "Abstract": ABSTRACT_PATH.read_text(encoding="utf-8"),
    }
    final_result_pattern = re.compile(r"(?<!历史)(\d{4}) passed, 49 warnings")
    final_results = {
        match.group(0)
        for text in texts.values()
        for match in final_result_pattern.finditer(text)
        if "历史" not in text[max(0, match.start() - 30):match.start()]
        and "过程" not in text[max(0, match.start() - 30):match.start()]
        and "基线" not in text[max(0, match.start() - 30):match.start()]
    }

    assert final_results == {"6333 passed, 49 warnings"}
    for name, text in texts.items():
        assert "6325 passed, 49 warnings" not in text, f"{name} 仍保留旧最终全量口径"
        assert "6301 passed, 49 warnings" not in text, f"{name} 仍保留旧最终全量口径"


def test_card_preview_has_mobile_single_column_media_query():
    html = (ROOT / "ux" / "card_preview.html").read_text(encoding="utf-8")

    assert "@media (max-width: 360px)" in html
    assert ".mobile-acceptance-grid" in html
    assert '<div class="cards-grid mobile-acceptance-grid">' in html
    assert 'class="cards-grid" style="grid-template-columns' not in html
    assert 'class="cards-grid mobile-acceptance-grid" style="grid-template-columns' not in html
    mobile_section = html.split("@media (max-width: 360px)", 1)[1]
    assert "body" in mobile_section
    assert "padding: 12px" in mobile_section
    assert ".cards-grid" in mobile_section
    assert ".mobile-acceptance-grid" in mobile_section
    assert "grid-template-columns: 1fr" in mobile_section
    assert "minmax(420px, 1fr)" not in mobile_section.split("}", 2)[0]
