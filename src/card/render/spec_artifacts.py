"""Render Spec PLAN/TASK artifacts as structured card panels."""

from __future__ import annotations

from src.card.state.models import ContentBlock
from src.card.themes import PANEL_STYLES

_TASK_STYLES: tuple[tuple[str, str], ...] = (
    ("wathet", "wathet"),
    ("green", "green"),
    ("yellow", "orange"),
    ("blue", "blue"),
    ("grey", "grey"),
)
_SPEC_PLAN_BACKGROUND_STYLE = "orange"
_SPEC_PLAN_BORDER_COLOR = "orange"

_STATUS_TEXT: dict[str, str] = {
    "pending": "待执行",
    "in_progress": "执行中",
    "completed": "已完成",
    "failed": "失败",
}


def render_spec_plan_panel(block: ContentBlock) -> dict | None:
    """Render one Spec PLAN artifact as an organized collapsible panel."""
    data = getattr(block, "data", None)
    if not isinstance(data, dict):
        return None

    cycle_num = int(data.get("cycle_num") or 0)
    architecture = str(data.get("architecture") or "").strip()
    tech_stack = _string_list(data.get("tech_stack"))
    steps = _string_list(data.get("steps"))
    file_changes = _string_list(data.get("file_changes"))
    test_plan = _string_list(data.get("test_plan"))
    risks = _string_list(data.get("risks"))
    notes = _string_list(data.get("notes"))

    header_parts = ["🏗️ **方案规划**"]
    if cycle_num:
        header_parts.append(f"第 {cycle_num} 轮")
    if steps:
        header_parts.append(f"{len(steps)} 步")
    if file_changes:
        header_parts.append(f"{len(file_changes)} 项交付物")

    body_lines: list[str] = ["**方案规划**"]
    if architecture:
        body_lines.append(f"**架构/方案**：{architecture}")
    if tech_stack:
        body_lines.append(f"**工具/资料**：{'、'.join(tech_stack)}")
    _append_numbered_section(body_lines, "执行步骤", steps)
    _append_bulleted_section(body_lines, "交付物变更", file_changes)
    _append_numbered_section(body_lines, "验证方案", test_plan)
    _append_bulleted_section(body_lines, "风险", risks)
    _append_bulleted_section(body_lines, "说明", notes)

    if len(body_lines) == 1:
        return None

    return _build_panel(
        header=" · ".join(header_parts),
        body="\n".join(body_lines),
        background_style=_SPEC_PLAN_BACKGROUND_STYLE,
        border_color=_SPEC_PLAN_BORDER_COLOR,
        expanded=True,
    )


def render_spec_task_panel(block: ContentBlock) -> dict | None:
    """Render one Spec TASK item as a complete, non-truncated task panel."""
    data = getattr(block, "data", None)
    if not isinstance(data, dict):
        return None

    task_id = data.get("task_id") or data.get("task_index") or "?"
    description = str(data.get("description") or "").strip()
    if not description:
        return None
    dependencies = _dependencies(data.get("dependencies"))
    status = str(data.get("status") or "pending").strip()
    output = str(data.get("output") or "").strip()
    task_index = int(data.get("task_index") or 1)
    background_style, border_color = _TASK_STYLES[(task_index - 1) % len(_TASK_STYLES)]

    header = f"📝 **任务 {task_id}：{description}**"
    body_lines = [
        f"**任务 {task_id}**：{description}",
        f"**依赖**：{_format_dependencies(dependencies)}",
    ]
    if status:
        body_lines.append(f"**状态**：{_STATUS_TEXT.get(status, status)}")
    if output:
        body_lines.append(f"**输出**：{output}")

    return _build_panel(
        header=header,
        body="\n".join(body_lines),
        background_style=background_style,
        border_color=border_color,
        expanded=True,
    )


def render_spec_task_list_panel(blocks: list[ContentBlock] | tuple[ContentBlock, ...]) -> dict | None:
    """Render Spec TASK items as one ordered collapsible list panel."""
    rows: list[str] = []
    for index, block in enumerate(blocks, start=1):
        data = getattr(block, "data", None)
        if not isinstance(data, dict):
            continue

        task_id = data.get("task_id") or data.get("task_index") or index
        description = str(data.get("description") or "").strip()
        if not description:
            continue

        dependencies = _dependencies(data.get("dependencies"))
        status = str(data.get("status") or "pending").strip() or "pending"
        output = str(data.get("output") or "").strip()
        row = (
            f"{task_id}. {description}"
            f" · 依赖：{_format_dependencies(dependencies)}"
            f" · 状态：{_STATUS_TEXT.get(status, status)}"
        )
        if output:
            row = f"{row} · 输出：{output}"
        rows.append(row)

    if not rows:
        return None

    return _build_panel(
        header=f"📝 **任务列表-{len(rows)}**",
        body="\n".join(rows),
        background_style="default",
        border_color=PANEL_STYLES["border_normal"],
        expanded=True,
    )


def _build_panel(
    *,
    header: str,
    body: str,
    background_style: str,
    border_color: str,
    expanded: bool,
) -> dict:
    return {
        "tag": "collapsible_panel",
        "expanded": expanded,
        "header": {
            "title": {"tag": "markdown", "content": header},
            "vertical_align": "center",
            "icon": {
                "tag": "standard_icon",
                "token": "down-small-ccm_outlined",
                "size": "16px 16px",
            },
            "icon_position": "follow_text",
            "icon_expanded_angle": -180,
        },
        "border": {"color": border_color, "corner_radius": PANEL_STYLES["corner_radius"]},
        "vertical_spacing": PANEL_STYLES["vertical_spacing"],
        "padding": PANEL_STYLES["padding_standard"],
        "elements": [
            {
                "tag": "column_set",
                "flex_mode": "stretch",
                "background_style": background_style,
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "vertical_align": "center",
                        "elements": [
                            {"tag": "markdown", "content": body, "text_align": "left"}
                        ],
                    }
                ],
            }
        ],
    }


def _string_list(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dependencies(value: object) -> list[int | str]:
    if not isinstance(value, list | tuple):
        return []
    result: list[int | str] = []
    for item in value:
        if isinstance(item, int):
            result.append(item)
            continue
        text = str(item).strip()
        if not text:
            continue
        result.append(int(text) if text.isdigit() else text)
    return result


def _format_dependencies(dependencies: list[int | str]) -> str:
    if not dependencies:
        return "无"
    return "、".join(f"任务 {dep}" if isinstance(dep, int) else str(dep) for dep in dependencies)


def _append_numbered_section(lines: list[str], title: str, items: list[str]) -> None:
    if not items:
        return
    lines.append("")
    lines.append(f"**{title}**")
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item}")


def _append_bulleted_section(lines: list[str], title: str, items: list[str]) -> None:
    if not items:
        return
    lines.append("")
    lines.append(f"**{title}**")
    for item in items:
        lines.append(f"- {item}")
