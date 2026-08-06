from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from src.card.builders.system import SystemBuilder


def _preview_html() -> str:
    return (Path(__file__).resolve().parents[1] / "ux" / "card_preview.html").read_text(encoding="utf-8")


def _degraded_error_card(degraded_to: str | None = "Coco") -> dict:
    continue_action = {"action": "continue_degraded", "request_id": "req-mobile"}
    retry_action = {
        "action": "retry_original",
        "original_mode": "Codex",
        "retry_mode": "Codex",
        "request_id": "req-mobile",
    }
    if degraded_to is not None:
        continue_action["degraded_to"] = degraded_to
        retry_action["degraded_to"] = degraded_to
    _, card_json = SystemBuilder.build_error_card(
        "Codex 启动失败",
        title="Codex 暂不可用",
        severity="degraded",
        detail_action={"action": "show_error_details", "trace_id": "mobile-layout"},
        continue_action=continue_action,
        retry_action=retry_action,
    )
    return json.loads(card_json)


def _button_rows_from_card(card: dict) -> list[list[str]]:
    rows: list[list[str]] = []

    def collect_buttons(node) -> list[str]:
        found: list[str] = []
        if isinstance(node, dict):
            if node.get("tag") == "button":
                found.append(node.get("text", {}).get("content", ""))
            for value in node.values():
                found.extend(collect_buttons(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(collect_buttons(item))
        return found

    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("tag") == "column_set":
                buttons = collect_buttons(node.get("columns", []))
                if buttons:
                    rows.append(buttons)
                    return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(card)
    return rows


def _compute_button_boxes_from_rows(rows: list[list[str]], viewport_width: int) -> list[tuple[int, int, int, int, str]]:
    card_padding = 16
    row_gap = 8
    content_width = viewport_width - card_padding * 2
    y = 0
    boxes: list[tuple[int, int, int, int, str]] = []
    for row in rows:
        button_width = content_width if len(row) == 1 else max(104, (content_width - row_gap) // len(row))
        x = card_padding
        for label in row:
            boxes.append((x, y, button_width, 34, label))
            x += button_width + row_gap
        y += 34 + row_gap
    return boxes


@pytest.mark.parametrize(
    ("degraded_to", "expected_rows"),
    [
        ("Coco", [["继续使用 Coco"], ["查看详情"], ["重试原模式"]]),
        ("Aiden", [["继续使用 Aiden"], ["查看详情"], ["重试原模式"]]),
        (None, [["查看详情"]]),
    ],
)
def test_degraded_error_card_mobile_buttons_use_real_card_vertical_secondary_layout(
    degraded_to: str | None,
    expected_rows: list[list[str]],
) -> None:
    rows = _button_rows_from_card(_degraded_error_card(degraded_to))

    assert rows == expected_rows
    for viewport in (360, 320):
        boxes = _compute_button_boxes_from_rows(rows, viewport)
        assert [box[4] for box in boxes] == [label for row in expected_rows for label in row]
        assert all(box[0] == 16 for box in boxes)
        assert all(box[2] == viewport - 32 for box in boxes)
        for prev, current in zip(boxes, boxes[1:]):
            assert prev[1] + prev[3] + 8 <= current[1]


class _PreviewButtonLayoutParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_degraded = False
        self.capture_depth = 0
        self.current_row: str | None = None
        self.current_button_classes: list[str] | None = None
        self.rows: dict[str, list[str]] = {"primary": [], "secondary": []}
        self.text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = (attrs_dict.get("class") or "").split()
        if tag == "div" and "card-label" in classes:
            self.capture_depth = 1
            self.text_chunks = []
            return
        if self.capture_depth:
            self.capture_depth += 1
        if self.in_degraded and tag == "div" and "button-row-primary" in classes:
            self.current_row = "primary"
        elif self.in_degraded and tag == "div" and "button-row-secondary" in classes:
            self.current_row = "secondary"
        elif self.in_degraded and tag == "button" and self.current_row:
            self.current_button_classes = classes

    def handle_data(self, data: str) -> None:
        if self.capture_depth:
            self.text_chunks.append(data)
        if self.current_button_classes is not None and self.current_row:
            text = data.strip()
            if text:
                self.rows[self.current_row].append(text)

    def handle_endtag(self, tag: str) -> None:
        if self.capture_depth:
            self.capture_depth -= 1
            if self.capture_depth == 0:
                label = "".join(self.text_chunks).strip()
                # Only the "继续 Coco · 含完整 retry 上下文" preview card is required
                # to match the production degraded card fully.
                self.in_degraded = label.startswith("错误卡 · 降级错误（继续 Coco")
        if tag == "button":
            self.current_button_classes = None
        elif tag == "div" and self.current_row:
            self.current_row = None


def test_degraded_error_card_mobile_preview_matches_real_card_labels() -> None:
    parser = _PreviewButtonLayoutParser()
    parser.feed(_preview_html())

    real_rows = _button_rows_from_card(_degraded_error_card())
    assert parser.rows["primary"][0:1] == real_rows[0]
    assert parser.rows["secondary"][0:2] == [label for row in real_rows[1:] for label in row]


def test_degraded_error_card_preview_covers_dynamic_and_unknown_modes() -> None:
    html = _preview_html()
    degraded_section = html.split("错误卡 · 降级错误（继续 Coco", 1)[1].split(
        "</div>\n</div>\n\n<!-- ==================== Footer Legend", 1
    )[0]

    assert "操作未能按原模式完成，已进入安全降级路径。" in degraded_section
    assert "可继续使用 Coco" in degraded_section
    assert "继续 Aiden" in degraded_section or "继续使用 Aiden" in degraded_section
    assert "未知目标" in degraded_section
    assert "当前暂未确定可继续模式" in degraded_section
    assert "启动失败后不再自动切换到 Coco ACP" not in degraded_section

    secondary_rows = [
        row for row in degraded_section.split('<div class="button-row button-row-secondary">')[1:]
    ]
    assert any(
        "查看详情" in row and "重试原模式" in row.split("</div>", 1)[0]
        for row in secondary_rows
    ), "至少有一张降级卡应展示 [查看详情, 重试原模式] 同行布局（含完整 retry payload）"
    assert any(
        "查看详情" in row and "重试原模式" not in row.split("</div>", 1)[0]
        for row in secondary_rows
    ), "至少有一张降级卡应只展示 查看详情 按钮（无 retry payload 或未知降级目标）"
