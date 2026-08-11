"""Header rendering: legacy title/subtitle or programming-card v2 header."""

from __future__ import annotations

from src.card.render.banner_computer import format_elapsed
from src.card.state.models import CardState
from src.card.themes import TERMINAL_TEMPLATES
from src.card.tool_display import is_unhelpful_display_label
from src.utils.text import summarize_question_title

_TOOL_DISPLAY = {
    "coco": "Coco",
    "claude": "Claude",
    "aiden": "Aiden",
    "codex": "Codex",
    "gemini": "Gemini",
    "traex": "Traex",
    "grok": "Grok",
}

_SUBTASK_STATUS_LABELS = {
    "running": "执行中",
    "completed": "已完成",
    "completed_empty": "已完成",
    "failed": "执行失败",
    "cancelled": "已取消",
    "paused": "已暂停",
    "archived": "已封存",
    "blocked": "已阻塞",
    "denied": "已拒绝",
}
_SUBTASK_GENERIC_LABELS = frozenset(
    {"task", "agent", "subagent", "子任务", "子代理"}
)


def render_header(state: CardState, *, page_index: int = 0, total_pages: int = 1) -> dict:
    """Generate Feishu Schema 2.0 header JSON.

    Programming cards with v2 metadata keep the header compact; elapsed time
    and tool/model detail live in the footer. Plain/static cards keep the
    legacy reducer-provided title/subtitle.
    """
    if _should_render_v2_header(state):
        return _render_v2_header(state, page_index=page_index, total_pages=total_pages)

    title = _append_page_label(state.header.title, page_index=page_index, total_pages=total_pages)
    result: dict = {
        "title": {"tag": "plain_text", "content": title},
        "template": state.header.template,
    }

    if state.header.subtitle is not None:
        result["subtitle"] = {"tag": "plain_text", "content": state.header.subtitle}

    return result


def _should_render_v2_header(state: CardState) -> bool:
    metadata = state.metadata
    strong_v2_signal = any((
        metadata.working_dir,
        metadata.card_sequence != 1,
        metadata.is_subagent,
        metadata.frozen,
        metadata.bridge_phrase,
        metadata.session_started_at is not None,
    ))
    if metadata.engine_type:
        return strong_v2_signal
    return strong_v2_signal or bool(metadata.tool_name or metadata.model_name or metadata.session_started_at is not None)


def _render_v2_header(state: CardState, *, page_index: int = 0, total_pages: int = 1) -> dict:
    metadata = state.metadata
    if metadata.is_subagent:
        return _render_subtask_header(
            state,
            page_index=page_index,
            total_pages=total_pages,
        )

    tool_id = metadata.tool_name or metadata.mode_name
    tool_label = _TOOL_DISPLAY.get((tool_id or "").lower(), tool_id or metadata.mode_name or "?")
    project_name = metadata.project_name or state.header.title or "当前项目"
    seq = metadata.card_sequence
    archived = " · 已封存" if metadata.frozen else ""
    # v2 design: frozen cards hide model_name (mutual exclusion with 已封存 tag)
    model_suffix = "" if metadata.frozen else (f" · {metadata.model_name}" if metadata.model_name else "")
    elapsed_suffix = _spec_elapsed_suffix(state)

    iteration = _iteration_label(state)
    unit = _unit_label(metadata, iteration)
    page = _page_label(page_index=page_index, total_pages=total_pages)
    deep_question_title = _deep_question_title(state, iteration)
    if deep_question_title:
        title = f"{deep_question_title}{model_suffix}{elapsed_suffix}{archived}"
    elif iteration:
        title = f"{iteration}{model_suffix}{elapsed_suffix}{archived}"
    else:
        context_suffix = _title_context_suffix(state)
        title = f"📁 {project_name} · 🤖 {tool_label}{context_suffix} · #{seq}{page}{model_suffix}{elapsed_suffix}{archived}"
    if state.terminal == "failed" and not deep_question_title and "错误" not in title:
        title = f"❌ 错误 · {title}"

    subtitle = _build_v2_subtitle(
        state,
        unit=unit,
        page=page,
        include_card_position=bool(iteration),
    )

    result: dict = {
        "title": {"tag": "plain_text", "content": title},
        "template": _v2_header_template(state),
    }
    if subtitle:
        result["subtitle"] = {"tag": "plain_text", "content": subtitle}
    return result


def _render_subtask_header(
    state: CardState,
    *,
    page_index: int,
    total_pages: int,
) -> dict:
    """Render a child card around task identity rather than internal metadata."""
    metadata = state.metadata
    label = _subtask_label(metadata)
    title = "🧬 子任务"
    if label:
        title = f"{title} · {_truncate_title_part(label)}"

    display_terminal = "archived" if metadata.frozen else state.terminal
    subtitle_parts = [
        _SUBTASK_STATUS_LABELS.get(display_terminal, "执行中"),
        f"子卡 #{metadata.card_sequence}",
    ]
    if metadata.parent_card_seq:
        subtitle_parts.append(f"来自主卡 #{metadata.parent_card_seq}")
    page = _page_label(page_index=page_index, total_pages=total_pages)
    if page:
        subtitle_parts.append(page.removeprefix(" · "))

    return {
        "title": {"tag": "plain_text", "content": title},
        "subtitle": {
            "tag": "plain_text",
            "content": " · ".join(subtitle_parts),
        },
        "template": _subtask_header_template(state),
    }


def _subtask_label(metadata) -> str:
    """Return a human label while keeping child IDs out of the header."""
    label = (metadata.unit_label or "").strip()
    unit_id = (metadata.unit_id or "").strip()
    if not label or label == unit_id:
        return ""

    for prefix in ("子任务：", "子任务:", "子代理：", "子代理:"):
        if label.startswith(prefix):
            label = label[len(prefix):].strip()
            break

    if unit_id:
        for prefix in (f"子任务 {unit_id}", f"子代理 {unit_id}"):
            if label.startswith(prefix):
                label = label[len(prefix):].lstrip("：: ").strip()
                break
    if (
        label.lower() in _SUBTASK_GENERIC_LABELS
        or is_unhelpful_display_label(label)
    ):
        return ""
    return label


def _subtask_header_template(state: CardState) -> str:
    if state.metadata.frozen:
        return TERMINAL_TEMPLATES["archived"]
    if state.terminal == "running":
        return "orange"
    return TERMINAL_TEMPLATES.get(state.terminal, state.header.template)


def _title_context_suffix(state: CardState) -> str:
    labels: list[str] = []
    iteration = _iteration_label(state)
    if iteration:
        labels.append(iteration)

    unit = _unit_label(state.metadata, iteration)
    if unit:
        labels.append(unit)
    return "".join(f" · {label}" for label in labels)


def _build_v2_subtitle(
    state: CardState,
    *,
    unit: str,
    page: str,
    include_card_position: bool,
) -> str:
    metadata = state.metadata
    parts: list[str] = []
    if metadata.engine_type == "deep" and not metadata.frozen:
        parts.append(_deep_elapsed_subtitle(state))
    elif unit:
        parts.append(unit)
    if include_card_position:
        parts.append(f"#{metadata.card_sequence}")
    if page:
        parts.append(page.removeprefix(" · "))
    if metadata.is_subagent and metadata.parent_card_seq:
        parts.append(f"↳ from #{metadata.parent_card_seq}")
    return " · ".join(part for part in parts if part)


def _iteration_label(state: CardState) -> str:
    metadata = state.metadata
    index = metadata.iteration_index
    total = metadata.iteration_total
    if index is None and state.engine_ext is not None and state.engine_ext.cycle_num > 0:
        index = state.engine_ext.cycle_num
        total = state.engine_ext.max_cycles or total
    if not index:
        return ""
    if total and total > 1:
        return f"第 {index}/{total} 轮"
    return f"第 {index} 轮"


def _deep_question_title(state: CardState, iteration_label: str) -> str:
    """Return the Deep main-card question label without changing cycle state."""
    metadata = state.metadata
    if metadata.engine_type != "deep" or metadata.unit_kind:
        return ""
    if metadata.question_title is not None:
        return summarize_question_title(metadata.question_title)
    return "Deep 任务" if iteration_label else ""


def _spec_elapsed_suffix(state: CardState) -> str:
    metadata = state.metadata
    if metadata.engine_type != "spec" or metadata.frozen:
        return ""
    if not _iteration_label(state):
        return ""
    seconds = float(getattr(state.runtime_stats, "elapsed_seconds", 0.0) or 0.0)
    return f" · 总耗时 {format_elapsed(seconds)}"


def _deep_elapsed_subtitle(state: CardState) -> str:
    seconds = float(getattr(state.runtime_stats, "elapsed_seconds", 0.0) or 0.0)
    return f"总耗时 {_format_elapsed_hms(seconds)}"


def _format_elapsed_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours}时{minutes:02d}分{secs:02d}秒"


def _unit_label(metadata, iteration_label: str) -> str:
    label = (metadata.unit_label or "").strip()
    if not label or label == iteration_label:
        return ""
    unit_id = (metadata.unit_id or "").strip()
    if metadata.unit_kind == "task":
        prefix = f"任务 {unit_id}" if unit_id else "任务"
        if label.startswith(prefix):
            return _truncate_title_part(label)
        return _truncate_title_part(f"{prefix}: {label}")
    if metadata.unit_kind == "subagent":
        prefix = f"子任务 {unit_id}" if unit_id else "子任务"
        if label.startswith(prefix):
            return _truncate_title_part(label)
        return _truncate_title_part(f"{prefix}: {label}")
    return _truncate_title_part(label)


def _truncate_title_part(text: str, limit: int = 28) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _append_page_label(title: str, *, page_index: int, total_pages: int) -> str:
    page = _page_label(page_index=page_index, total_pages=total_pages)
    return f"{title}{page}" if page else title


def _page_label(*, page_index: int, total_pages: int) -> str:
    if total_pages <= 1:
        return ""
    return f" · 页 {page_index + 1}/{total_pages}"


def _v2_header_template(state: CardState) -> str:
    metadata = state.metadata
    if metadata.frozen:
        return "grey"
    if metadata.is_subagent:
        return _subtask_header_template(state)
    return state.header.template
