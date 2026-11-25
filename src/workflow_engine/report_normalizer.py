"""Payload normalization and semantic structuring for Workflow execution reports.

Converts raw, potentially double-encoded JSON, Teamwork handoff envelopes, and
Markdown/plain-text outputs into deduplicated, human-readable semantic sections
and blocks for HTML and Markdown report rendering.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from .models import WorkflowProject, WorkflowStatus

_MAX_CONTAINER_DEPTH = 64

_VERDICT_LABELS: dict[str, str] = {
    "passed": "通过",
    "needs_attention": "需处理",
    "failed": "失败",
    "cancelled": "已取消",
    "unknown": "待确认",
}

_VERDICT_BADGE_TONES: dict[str, str] = {
    "passed": "ok",
    "needs_attention": "warning",
    "failed": "error",
    "cancelled": "neutral",
    "unknown": "neutral",
}

_COLUMN_LABELS: dict[str, str] = {
    "category": "分类",
    "endpoints_or_entry": "入口与说明",
    "evidence": "源码依据",
    "type": "类型",
    "text": "内容",
    "name": "名称",
    "title": "标题",
    "path": "路径",
    "file": "文件",
    "line": "行号",
    "location": "位置",
    "status": "状态",
    "severity": "级别",
    "issue": "问题描述",
    "description": "问题描述",
    "resolution": "修正与依据",
    "suggestion": "修改建议",
    "recommendation": "修改建议",
    "fix": "修复方案",
    "round": "轮次",
    "verifier": "评审方",
    "approve": "是否通过",
    "approved": "是否通过",
    "summary": "摘要",
    "detail": "详情",
    "details": "详情",
    "role": "角色",
    "label": "标识",
    "tool": "工具",
    "model": "模型",
    "output": "输出",
    "result": "结果",
    "error": "错误",
    "stage": "阶段",
    "observation": "观察",
    "confidence": "置信度",
    "reasoning": "推导说明",
}

_KEY_LABELS: dict[str, str] = {
    **_COLUMN_LABELS,
    "preset": "预设方案",
    "rounds": "评审轮次",
    "accepted": "是否通过",
    "failures": "失败记录",
    "verifier_count": "评审方数量",
    "raw": "原始信息",
    "fallback": "降级标志",
    "scout": "Scout 阶段发现",
    "plan": "执行计划",
    "worker_findings": "执行发现",
    "parallel_results": "并行结果",
    "results": "结果集合",
    "agent_outputs": "Agent 原始输出",
    "raw_outputs": "原始输出",
    "worker_outputs": "Worker 输出",
    "reviews": "评审记录",
    "logs": "日志",
}

_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)
_MARKDOWN_FENCE_RE = re.compile(r"^\s{0,3}(?:`{3,}|~{3,})", re.MULTILINE)
_MARKDOWN_TABLE_SEP_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$",
    re.MULTILINE,
)
_MARKDOWN_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+\S", re.MULTILINE)
_MARKDOWN_QUOTE_RE = re.compile(r"^\s*>\s+\S", re.MULTILINE)
_MARKDOWN_INLINE_RE = re.compile(
    r"`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\([^)\s]+\)"
)
_SINGLE_LINE_JSON_FENCE_RE = re.compile(
    r"^\s*(?P<fence>`{3,}|~{3,})\s*json\s+(?P<body>[\{\[].*?)\s*(?P=fence)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_OPENER_RE = re.compile(
    r"\s{0,3}(?P<fence>`{3,}|~{3,})\s*json\s*",
    re.IGNORECASE,
)
_FENCE_CLOSER_RE = re.compile(r"\s{0,3}(?P<fence>`{3,}|~{3,})\s*")
_ROUND_SUMMARY_KEY_RE = re.compile(r"^round_(\d+)_summary$")

_AGENT_OUTPUT_COLLECTION_KEYS = (
    "agent_outputs",
    "raw_outputs",
    "worker_outputs",
    "parallel_results",
    "results",
)


@dataclass(frozen=True)
class StructuredTable:
    """Tabular representation with column headers and row cells."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    title: str = ""


@dataclass(frozen=True)
class NormalizedBlock:
    """Semantic content block inside a normalized Workflow report section."""

    kind: Literal["markdown", "prose", "kv", "table", "list", "callout", "json"]
    title: str = ""
    text: str = ""
    pairs: tuple[tuple[str, str], ...] = ()
    table: StructuredTable | None = None
    items: tuple[str, ...] = ()
    level: str = "info"


@dataclass(frozen=True)
class NormalizedSection:
    """Logical report section grouping related semantic blocks."""

    id: str
    title: str
    badge: str = ""
    badge_tone: str = "neutral"
    blocks: tuple[NormalizedBlock, ...] = ()
    expanded: bool = True
    is_debug_raw: bool = False


@dataclass(frozen=True)
class NormalizedAgentView:
    """Structured semantic view alongside raw output for a single agent call."""

    summary_blocks: tuple[NormalizedBlock, ...]
    raw_text: str
    has_structured_view: bool


@dataclass(frozen=True)
class NormalizedWorkflowReport:
    """Top-level normalized report structure consumed by HTML/Markdown renderers."""

    verdict: str
    verdict_label: str
    headline_conclusion: str
    sections: tuple[NormalizedSection, ...]
    agent_views: dict[tuple[int, int], NormalizedAgentView]


# ---------------------------------------------------------------------------
# Recursive JSON unwrapping
# ---------------------------------------------------------------------------


def _extract_fenced_json_body(text: str) -> str | None:
    """Return the inner JSON body if *text* is solely a fenced json block."""
    single = _SINGLE_LINE_JSON_FENCE_RE.match(text)
    if single:
        return single.group("body").strip()

    first_line, sep, remainder = text.partition("\n")
    if not sep:
        return None
    opener = _FENCE_OPENER_RE.fullmatch(first_line)
    if opener is None:
        return None
    body, closing_sep, closing_line = remainder.rstrip().rpartition("\n")
    if not closing_sep:
        return None
    closer = _FENCE_CLOSER_RE.fullmatch(closing_line)
    if closer is None:
        return None
    opening_fence = opener.group("fence")
    closing_fence = closer.group("fence")
    if opening_fence[0] != closing_fence[0] or len(closing_fence) < len(opening_fence):
        return None
    stripped_body = body.strip()
    return stripped_body if stripped_body else None


def _try_parse_json_string(text: str) -> tuple[bool, Any]:
    """Parse *text* as JSON if it starts with '{'/'[' or is a fenced json block."""
    stripped = text.strip()
    if not stripped:
        return False, text

    fenced_body = _extract_fenced_json_body(stripped)
    if fenced_body is not None:
        candidate = fenced_body
        from_fence = True
    elif stripped[0] in "{[":
        candidate = stripped
        from_fence = False
    else:
        return False, text

    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        return False, text

    if isinstance(parsed, (dict, list)):
        return True, parsed
    if from_fence and isinstance(parsed, str):
        return True, parsed
    return False, text


def _unwrap_value(value: Any, remaining_depth: int, container_depth: int) -> Any:
    """Recursively unwrap JSON-encoded strings up to *remaining_depth* decode layers."""
    if remaining_depth <= 0 or container_depth > _MAX_CONTAINER_DEPTH:
        return value

    if isinstance(value, str):
        ok, parsed = _try_parse_json_string(value)
        if not ok:
            return value
        return _unwrap_value(parsed, remaining_depth - 1, container_depth + 1)

    if isinstance(value, dict):
        return {
            key: _unwrap_value(item, remaining_depth, container_depth + 1)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _unwrap_value(item, remaining_depth, container_depth + 1)
            for item in value
        ]

    return value


def unwrap_nested_json(value: Any, *, max_depth: int = 4) -> Any:
    """Recursively unpack JSON-encoded strings inside dicts, lists, and fenced blocks.

    Parses strings whose trimmed text begins with ``{`` or ``[`` (or a complete
    ```` ```json ... ``` ```` fence) using ``json.loads``, descending into nested
    dictionaries and lists up to ``max_depth`` string-decoding layers. Non-JSON
    strings and primitive values are returned unchanged.
    """
    if max_depth <= 0:
        return value
    return _unwrap_value(value, max_depth, 0)


# ---------------------------------------------------------------------------
# Text, formatting, and deduplication helpers
# ---------------------------------------------------------------------------


def _looks_like_markdown(text: str) -> bool:
    """Return True if *text* contains common Markdown structures or inline syntax."""
    if not text or not text.strip():
        return False
    if _MARKDOWN_HEADING_RE.search(text):
        return True
    if _MARKDOWN_FENCE_RE.search(text):
        return True
    if "|" in text and _MARKDOWN_TABLE_SEP_RE.search(text):
        return True
    if _MARKDOWN_LIST_RE.search(text):
        return True
    if _MARKDOWN_QUOTE_RE.search(text):
        return True
    return bool(_MARKDOWN_INLINE_RE.search(text))


def _text_or_markdown_block(text: str, *, title: str = "") -> NormalizedBlock:
    """Build a markdown or prose block depending on text content."""
    clean = text.strip()
    kind: Literal["markdown", "prose"] = (
        "markdown" if _looks_like_markdown(clean) else "prose"
    )
    return NormalizedBlock(kind=kind, title=title, text=clean)


def _canonical_key(value: Any) -> str:
    """Build a deterministic comparison key for deduplication."""
    if isinstance(value, str):
        normalized = re.sub(r"[ \t]+", " ", value.replace("\r\n", "\n")).strip()
        return f"str:{normalized}"
    if isinstance(value, (dict, list)):
        try:
            dumped = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            dumped = str(value)
        return f"json:{dumped}"
    return f"{type(value).__name__}:{value!r}"


def _dedupe_sequence(items: Iterable[Any]) -> list[Any]:
    """Return items in first-seen order without structural duplicates."""
    seen: set[str] = set()
    unique: list[Any] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, str) and not item.strip():
            continue
        key = _canonical_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _humanize_label(key: str) -> str:
    """Map an internal field key to a localized display label."""
    if key in _KEY_LABELS:
        return _KEY_LABELS[key]
    lower = key.lower()
    if lower in _KEY_LABELS:
        return _KEY_LABELS[lower]
    return key.replace("_", " ").strip() or key


def _format_scalar(value: Any, *, key: str = "") -> str:
    """Format a scalar or compact value as display text."""
    if value is None:
        return ""
    if isinstance(value, bool):
        if key in ("accepted", "approve", "approved"):
            return "通过 (true)" if value else "未通过 (false)"
        return "是" if value else "否"
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return ""
        if all(isinstance(v, (str, int, float, bool)) for v in value):
            return "；".join(_format_scalar(v) for v in value if _format_scalar(v))
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, dict):
        if not value:
            return ""
        if all(isinstance(v, (str, int, float, bool)) for v in value.values()):
            parts = [
                f"{_humanize_label(str(k))}: {_format_scalar(v, key=str(k))}"
                for k, v in value.items()
                if v is not None and _format_scalar(v, key=str(k))
            ]
            return "；".join(parts)
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _trim_balanced_backticks(text: str) -> str:
    """Trim trailing incomplete inline backtick span so Markdown never breaks."""
    if text.count("`") % 2 == 0:
        return text
    last_tick = text.rfind("`")
    if last_tick <= 0:
        return text.replace("`", "")
    prefix = text[:last_tick].rstrip(" ,，、;；:：(")
    return prefix if prefix else text.replace("`", "")


def _extract_headline_conclusion(text: str, *, max_chars: int = 360) -> str:
    """Derive a concise, valid-Markdown headline lead from a full conclusion."""
    clean = str(text or "").strip()
    if not clean:
        return ""
    if len(clean) <= max_chars:
        return clean

    paragraphs = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    if paragraphs:
        first = paragraphs[0]
        if len(first) <= max_chars:
            selected: list[str] = [first]
            total = len(first)
            for line in paragraphs[1:]:
                if total + 1 + len(line) > max_chars:
                    break
                selected.append(line)
                total += 1 + len(line)
            joined = "\n".join(selected)
            if len(selected) < len(paragraphs) and joined.endswith((":", "：")):
                joined = joined[:-1] + "。"
            return _trim_balanced_backticks(joined)

    # First paragraph alone exceeds max_chars: break at sentence boundary if possible.
    candidate = clean[:max_chars]
    for sep in ("。", ". ", "；", "; ", "\n"):
        idx = candidate.rfind(sep)
        if idx >= max_chars // 3:
            end_pos = idx + (1 if sep in ("。", "；") else 0)
            return _trim_balanced_backticks(clean[:end_pos].strip())
    truncated = _trim_balanced_backticks(candidate.rstrip())
    return truncated if truncated.endswith(("。", ".", "…")) else f"{truncated}…"


# ---------------------------------------------------------------------------
# Generic semantic block builders
# ---------------------------------------------------------------------------


def _build_table_from_dicts(
    rows: list[dict[str, Any]],
    *,
    title: str = "",
    preferred_cols: tuple[str, ...] = (),
) -> StructuredTable | None:
    """Convert a list of dicts into a localized :class:`StructuredTable`."""
    if not rows:
        return None

    ordered_keys: list[str] = []
    seen_keys: set[str] = set()
    for key in preferred_cols:
        if any(key in row for row in rows):
            ordered_keys.append(key)
            seen_keys.add(key)
    for row in rows:
        for key in row:
            key_str = str(key)
            if key_str not in seen_keys:
                ordered_keys.append(key_str)
                seen_keys.add(key_str)

    if not ordered_keys:
        return None

    columns = tuple(_humanize_label(key) for key in ordered_keys)
    table_rows: list[tuple[str, ...]] = []
    for row in rows:
        cells = tuple(_format_scalar(row.get(key), key=key) for key in ordered_keys)
        if any(cell for cell in cells):
            table_rows.append(cells)

    if not table_rows:
        return None
    return StructuredTable(columns=columns, rows=tuple(table_rows), title=title)


def _format_list_item(item: Any) -> str:
    """Convert one list element into a readable string item."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, (int, float, bool)):
        return str(item)
    if isinstance(item, dict):
        severity = str(item.get("severity") or "").strip().lower()
        status = str(item.get("status") or "").strip()
        text = ""
        for k in ("text", "description", "issue", "claim", "summary", "title", "name", "path"):
            val = item.get(k)
            if isinstance(val, str) and val.strip():
                text = val.strip()
                break
        if text:
            prefix_bits: list[str] = []
            if severity:
                prefix_bits.append(f"[{severity}]")
            elif status and status not in ("info",):
                prefix_bits.append(f"[{status}]")
            extra_bits: list[str] = []
            for extra_k in ("evidence", "resolution", "location", "file"):
                extra_v = _format_scalar(item.get(extra_k), key=extra_k)
                if extra_v:
                    extra_bits.append(f"{_humanize_label(extra_k)}: {extra_v}")
            prefix = " ".join(prefix_bits) + " " if prefix_bits else ""
            suffix = f"（{'；'.join(extra_bits)}）" if extra_bits else ""
            return f"{prefix}{text}{suffix}".strip()
        return _format_scalar(item)
    return str(item).strip()


def _build_deliverables_blocks(
    value: Any,
    *,
    title: str = "交付成果",
) -> list[NormalizedBlock]:
    """Structure grouped or flat deliverables into tables, lists, or prose."""
    if value is None:
        return []
    if isinstance(value, str):
        return [_text_or_markdown_block(value, title=title)] if value.strip() else []
    if isinstance(value, dict):
        if "items" in value and isinstance(value["items"], list):
            group_title = str(value.get("name") or value.get("title") or title).strip()
            return _build_deliverables_group(group_title, value["items"])
        return _build_generic_blocks(value, title=title)
    if not isinstance(value, list):
        return [_text_or_markdown_block(str(value), title=title)]

    if not value:
        return []

    # Grouped deliverables: [{name: ..., items: [{category, endpoints_or_entry, evidence}]}]
    if all(isinstance(entry, dict) and isinstance(entry.get("items"), list) for entry in value):
        blocks: list[NormalizedBlock] = []
        for idx, group in enumerate(value, 1):
            group_title = str(
                group.get("name") or group.get("title") or f"{title} #{idx}"
            ).strip()
            blocks.extend(_build_deliverables_group(group_title, group["items"]))
            extra_keys = {
                k: v for k, v in group.items() if k not in ("name", "title", "items")
            }
            if extra_keys:
                blocks.extend(_build_generic_blocks(extra_keys, title=group_title))
        return blocks

    if all(isinstance(entry, dict) for entry in value):
        table = _build_table_from_dicts(
            value,
            title=title,
            preferred_cols=(
                "category",
                "type",
                "name",
                "title",
                "endpoints_or_entry",
                "text",
                "evidence",
                "path",
                "status",
            ),
        )
        if table is not None:
            return [NormalizedBlock(kind="table", title=title, table=table)]

    items = [formatted for item in value if (formatted := _format_list_item(item))]
    return [NormalizedBlock(kind="list", title=title, items=tuple(items))] if items else []


def _build_deliverables_group(group_title: str, items: list[Any]) -> list[NormalizedBlock]:
    """Build blocks for one grouped deliverable entry."""
    if not items:
        return []
    if all(isinstance(item, dict) for item in items):
        table = _build_table_from_dicts(
            items,
            title=group_title,
            preferred_cols=("category", "endpoints_or_entry", "evidence", "type", "name", "text", "path"),
        )
        if table is not None:
            return [NormalizedBlock(kind="table", title=group_title, table=table)]
    formatted = [text for item in items if (text := _format_list_item(item))]
    return [NormalizedBlock(kind="list", title=group_title, items=tuple(formatted))] if formatted else []


def _build_list_or_table_blocks(
    value: Any,
    *,
    title: str,
    preferred_cols: tuple[str, ...] = (),
    callout_if_scalar: bool = False,
    callout_level: str = "warning",
) -> list[NormalizedBlock]:
    """Structure a list, dict, or scalar into tables, lists, callouts, or prose."""
    if value is None:
        return []
    if isinstance(value, str):
        clean = value.strip()
        if not clean:
            return []
        if callout_if_scalar:
            return [NormalizedBlock(kind="callout", title=title, text=clean, level=callout_level)]
        return [_text_or_markdown_block(clean, title=title)]
    if isinstance(value, dict):
        return _build_generic_blocks(value, title=title)
    if not isinstance(value, list):
        return [_text_or_markdown_block(str(value), title=title)]
    if not value:
        return []

    if all(isinstance(item, dict) for item in value):
        table = _build_table_from_dicts(value, title=title, preferred_cols=preferred_cols)
        if table is not None:
            return [NormalizedBlock(kind="table", title=title, table=table)]

    items = [text for item in value if (text := _format_list_item(item))]
    return [NormalizedBlock(kind="list", title=title, items=tuple(items))] if items else []


def _build_generic_blocks(
    value: Any,
    *,
    title: str = "",
) -> list[NormalizedBlock]:
    """Fallback semantic structurer for arbitrary unwrapped JSON objects or arrays."""
    if value is None:
        return []
    if isinstance(value, str):
        return [_text_or_markdown_block(value, title=title)] if value.strip() else []
    if isinstance(value, (int, float, bool)):
        return [NormalizedBlock(kind="prose", title=title, text=_format_scalar(value))]
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, dict) for item in value):
            # If entries carry long narrative outputs, render each entry individually.
            has_long_text = any(
                isinstance(v, str) and (len(v) > 280 or "\n" in v)
                for row in value
                for v in row.values()
            )
            if has_long_text:
                return _build_agent_outputs_blocks(title, value)
            table = _build_table_from_dicts(value, title=title)
            if table is not None:
                return [NormalizedBlock(kind="table", title=title, table=table)]
        if all(isinstance(item, (str, int, float, bool)) for item in value):
            items = tuple(_format_scalar(item) for item in value if _format_scalar(item))
            return [NormalizedBlock(kind="list", title=title, items=items)] if items else []
        dumped = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        return [NormalizedBlock(kind="json", title=title, text=dumped)]

    if isinstance(value, dict):
        if not value:
            return []
        blocks: list[NormalizedBlock] = []
        kv_pairs: list[tuple[str, str]] = []
        complex_entries: list[tuple[str, Any]] = []

        for key, item in value.items():
            if item is None:
                continue
            key_str = str(key)
            if isinstance(item, str):
                clean = item.strip()
                if not clean:
                    continue
                if len(clean) > 160 or "\n" in clean or _looks_like_markdown(clean):
                    complex_entries.append((key_str, clean))
                else:
                    kv_pairs.append((_humanize_label(key_str), clean))
            elif isinstance(item, (int, float, bool)):
                kv_pairs.append((_humanize_label(key_str), _format_scalar(item, key=key_str)))
            elif isinstance(item, (list, dict)) and not item:
                continue
            else:
                complex_entries.append((key_str, item))

        if kv_pairs:
            blocks.append(NormalizedBlock(kind="kv", title=title, pairs=tuple(kv_pairs)))
        for key_str, item in complex_entries:
            sub_title = f"{title} · {_humanize_label(key_str)}" if title else _humanize_label(key_str)
            blocks.extend(_build_generic_blocks(item, title=sub_title))
        return blocks

    dumped = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return [NormalizedBlock(kind="json", title=title, text=dumped)]


def _build_agent_outputs_blocks(
    title: str,
    value: Any,
) -> list[NormalizedBlock]:
    """Render agent output collections (e.g. agent_outputs, parallel_results) readably."""
    if value is None:
        return []
    if isinstance(value, str):
        return [_text_or_markdown_block(value, title=title)] if value.strip() else []
    if isinstance(value, dict):
        blocks: list[NormalizedBlock] = []
        for sub_key, sub_val in value.items():
            if sub_val is None:
                continue
            sub_title = f"{title} · {sub_key}" if title else str(sub_key)
            blocks.extend(_build_generic_blocks(sub_val, title=sub_title))
        return blocks
    if isinstance(value, list):
        blocks = []
        for idx, entry in enumerate(value, 1):
            if entry is None:
                continue
            if isinstance(entry, dict):
                label = (
                    entry.get("label")
                    or entry.get("agent_id")
                    or entry.get("name")
                    or entry.get("role")
                    or f"#{idx}"
                )
                entry_title = f"{title} · {label}" if title else str(label)
                remaining = {
                    k: v
                    for k, v in entry.items()
                    if k not in ("label", "name")
                }
                # If entry only has output/result/text alongside label, render cleanly.
                output_keys = [k for k in ("output", "result", "text", "content") if k in remaining]
                if len(remaining) == 1 and output_keys:
                    blocks.extend(_build_generic_blocks(remaining[output_keys[0]], title=entry_title))
                else:
                    blocks.extend(_build_generic_blocks(remaining or entry, title=entry_title))
            else:
                blocks.extend(_build_generic_blocks(entry, title=f"{title} #{idx}" if title else f"#{idx}"))
        return blocks
    return _build_generic_blocks(value, title=title)


# ---------------------------------------------------------------------------
# Verification normalization (shared by agent and workflow levels)
# ---------------------------------------------------------------------------


def _flatten_reviews_to_evidence(
    reviews: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Extract issue rows, round summary rows, and extra notes from reviews."""
    issue_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    extra_notes: list[str] = []
    for entry in reviews:
        if not isinstance(entry, dict):
            continue
        rnd = entry.get("round")
        verifier = entry.get("verifier")
        approve = entry.get("approve")
        issues = entry.get("issues")
        prefix = f"第 {rnd} 轮" if rnd is not None else "评审"
        if verifier:
            prefix = f"{prefix} ({verifier})"
        for note_key in ("observation", "conclusion", "summary", "caveats"):
            note_val = entry.get(note_key)
            note_str = _format_scalar(note_val, key=note_key)
            if note_str:
                extra_notes.append(f"{prefix} { _humanize_label(note_key) }: {note_str}")
        if isinstance(issues, list) and issues:
            for iss in issues:
                if isinstance(iss, dict):
                    row: dict[str, Any] = {}
                    if rnd is not None:
                        row["round"] = rnd
                    if verifier is not None:
                        row["verifier"] = verifier
                    row.update(iss)
                    issue_rows.append(row)
        else:
            row = {}
            if rnd is not None:
                row["round"] = rnd
            if verifier is not None:
                row["verifier"] = verifier
            if approve is not None:
                row["approve"] = approve
            if entry.get("summary"):
                row["summary"] = entry["summary"]
            if row:
                round_rows.append(row)
    return issue_rows, round_rows, extra_notes


def _extract_uncovered_feedback(feedback: str, evidence_rows: list[dict[str, Any]]) -> str:
    """Return portions of *feedback* not already represented by *evidence_rows*."""
    if not evidence_rows:
        return feedback
    covered_tokens: list[str] = []
    for row in evidence_rows:
        for field in (
            "description",
            "issue",
            "evidence",
            "location",
            "suggestion",
            "recommendation",
            "fix",
            "observation",
        ):
            val = str(row.get(field) or "").strip()
            if val:
                covered_tokens.append(val)
    if not covered_tokens:
        return feedback

    uncovered_lines: list[str] = []
    for line in feedback.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(token in stripped for token in covered_tokens):
            continue
        # Also skip bare Location lines whose file:line is already in evidence_rows
        if stripped.startswith("Location:") and any(
            str(row.get("file") or "") in stripped
            for row in evidence_rows
            if row.get("file")
        ):
            continue
        uncovered_lines.append(stripped)
    return "\n".join(uncovered_lines)


def _build_verification_blocks(
    *,
    synth_ver: Any = None,
    runtime_ver: Any = None,
    card_ver: Any = None,
) -> tuple[list[NormalizedBlock], str, str, bool | None, bool]:
    """Merge synthesizer, runtime, and card verification views without duplication.

    Returns ``(blocks, badge, badge_tone, accepted_flag, has_reverification)``.
    """
    blocks: list[NormalizedBlock] = []

    # Separate dict sources from string/list verification entries.
    merged_dict: dict[str, Any] = {}
    text_sources: list[str] = []
    list_sources: list[Any] = []

    for src in (runtime_ver, synth_ver):
        if isinstance(src, dict):
            for k, v in src.items():
                if k == "output":
                    # runtime verification.output mirrors verified_output; handled separately.
                    continue
                if k not in merged_dict or merged_dict[k] in (None, "", []):
                    merged_dict[k] = v
                elif isinstance(merged_dict[k], list) and isinstance(v, list):
                    merged_dict[k] = _dedupe_sequence([*merged_dict[k], *v])
        elif isinstance(src, str) and src.strip():
            text_sources.append(src.strip())
        elif isinstance(src, list):
            list_sources.extend(src)

    accepted_flag: bool | None = None
    for flag_key in ("accepted", "approve", "approved"):
        if isinstance(merged_dict.get(flag_key), bool):
            accepted_flag = merged_dict[flag_key]
            break

    reverification = str(merged_dict.get("synthesizer_reverification") or "").strip()
    has_reverification = bool(reverification)

    # 1. KV overview block
    kv_pairs: list[tuple[str, str]] = []
    handled_keys: set[str] = {"output"}

    for key in ("status", "rounds", "accepted", "approve", "approved", "verifier", "confidence", "summary"):
        if key in merged_dict and merged_dict[key] is not None:
            val_str = _format_scalar(merged_dict[key], key=key)
            if val_str:
                kv_pairs.append((_humanize_label(key), val_str))
            handled_keys.add(key)

    failures = merged_dict.get("failures")
    if isinstance(failures, list):
        kv_pairs.append(("失败记录", str(len(failures))))
        handled_keys.add("failures")

    # Include compact custom scalar fields (e.g. legacy {"summary": ..., "raw": ...}).
    for key, val in list(merged_dict.items()):
        if key in handled_keys or val is None:
            continue
        if isinstance(val, (int, float, bool)) or (
            isinstance(val, str)
            and len(val.strip()) <= 160
            and "\n" not in val
            and not _ROUND_SUMMARY_KEY_RE.match(key)
            and key not in ("feedback", "synthesizer_reverification")
        ):
            val_str = _format_scalar(val, key=key)
            if val_str:
                kv_pairs.append((_humanize_label(key), val_str))
            handled_keys.add(key)

    if kv_pairs:
        blocks.append(NormalizedBlock(kind="kv", title="验证概览", pairs=tuple(kv_pairs)))

    # 2. Synthesizer reverification callout
    if reverification:
        blocks.append(
            NormalizedBlock(
                kind="callout",
                title="综合器源码复核说明",
                text=reverification,
                level="info",
            )
        )
        handled_keys.add("synthesizer_reverification")

    # 3. Round summaries (round_1_summary, round_2_summary, ...)
    round_summary_keys = sorted(
        (k for k in merged_dict if _ROUND_SUMMARY_KEY_RE.match(k)),
        key=lambda k: int(_ROUND_SUMMARY_KEY_RE.match(k).group(1)),  # type: ignore[union-attr]
    )
    for r_key in round_summary_keys:
        handled_keys.add(r_key)
        r_text = str(merged_dict.get(r_key) or "").strip()
        if not r_text:
            continue
        round_num = _ROUND_SUMMARY_KEY_RE.match(r_key).group(1)  # type: ignore[union-attr]
        blocks.append(
            _text_or_markdown_block(r_text, title=f"第 {round_num} 轮评审摘要")
        )

    # 4. Resolved issues table (synthesizer)
    resolved_issues = merged_dict.get("resolved_issues")
    if isinstance(resolved_issues, list) and resolved_issues:
        handled_keys.add("resolved_issues")
        if all(isinstance(r, dict) for r in resolved_issues):
            table = _build_table_from_dicts(
                resolved_issues,
                title="已消解评审问题",
                preferred_cols=("severity", "issue", "description", "resolution", "evidence"),
            )
            if table is not None:
                blocks.append(
                    NormalizedBlock(kind="table", title="已消解评审问题", table=table)
                )
        else:
            blocks.extend(_build_list_or_table_blocks(resolved_issues, title="已消解评审问题"))

    # 5. Evidence / reviews / issues tables
    evidence_rows: list[dict[str, Any]] = []
    if isinstance(merged_dict.get("evidence"), list):
        handled_keys.add("evidence")
        evidence_rows.extend(
            item for item in merged_dict["evidence"] if isinstance(item, dict)
        )
    if isinstance(merged_dict.get("issues"), list):
        handled_keys.add("issues")
        evidence_rows.extend(
            item for item in merged_dict["issues"] if isinstance(item, dict)
        )

    review_round_rows: list[dict[str, Any]] = []
    review_extra_notes: list[str] = []
    if isinstance(merged_dict.get("reviews"), list):
        handled_keys.add("reviews")
        extracted_issues, review_round_rows, review_extra_notes = _flatten_reviews_to_evidence(
            merged_dict["reviews"]
        )
        if not evidence_rows:
            evidence_rows = extracted_issues

    evidence_rows = _dedupe_sequence(evidence_rows)
    if evidence_rows:
        table = _build_table_from_dicts(
            evidence_rows,
            title="评审发现明细",
            preferred_cols=(
                "round",
                "verifier",
                "severity",
                "description",
                "issue",
                "evidence",
                "location",
                "suggestion",
                "resolution",
            ),
        )
        if table is not None:
            blocks.append(
                NormalizedBlock(kind="table", title="评审发现明细", table=table)
            )
        if review_extra_notes:
            blocks.append(
                NormalizedBlock(
                    kind="list",
                    title="各轮评审补充说明",
                    items=tuple(_dedupe_sequence(review_extra_notes)),
                )
            )
    elif review_round_rows:
        table = _build_table_from_dicts(
            review_round_rows,
            title="各轮评审结果",
            preferred_cols=("round", "verifier", "approve", "summary"),
        )
        if table is not None:
            blocks.append(
                NormalizedBlock(kind="table", title="各轮评审结果", table=table)
            )
        if review_extra_notes:
            blocks.append(
                NormalizedBlock(
                    kind="list",
                    title="各轮评审补充说明",
                    items=tuple(_dedupe_sequence(review_extra_notes)),
                )
            )

    # 6. Failures (if any verifier crashed/timed out)
    if isinstance(failures, list) and failures:
        blocks.extend(
            _build_list_or_table_blocks(
                failures,
                title="评审器异常记录",
                preferred_cols=("round", "verifier", "stage", "error"),
                callout_if_scalar=True,
                callout_level="error",
            )
        )

    # 7. Feedback (keep only uncovered lines when evidence_rows already capture issues)
    feedback = str(merged_dict.get("feedback") or "").strip()
    if feedback:
        handled_keys.add("feedback")
        uncovered_feedback = _extract_uncovered_feedback(feedback, evidence_rows)
        if uncovered_feedback:
            blocks.append(_text_or_markdown_block(uncovered_feedback, title="评审反馈"))

    # 8. Remaining unhandled keys inside verification dict
    for key, val in merged_dict.items():
        if key in handled_keys or val is None:
            continue
        blocks.extend(_build_generic_blocks(val, title=_humanize_label(key)))

    # 9. String verification sources (e.g. worker verification string)
    for txt in _dedupe_sequence(text_sources):
        blocks.append(_text_or_markdown_block(txt, title="验证说明"))

    # 10. Card/list verification items (skip duplicates of feedback or reverification)
    combined_list = _dedupe_sequence(
        [*list_sources, *(card_ver if isinstance(card_ver, list) else [])]
    )
    rendered_corpus = "\n".join(b.text for b in blocks if b.text)
    if feedback:
        rendered_corpus = f"{rendered_corpus}\n{feedback}"

    extra_ver_items: list[str] = []
    for item in combined_list:
        item_text = _format_list_item(item)
        if not item_text:
            continue
        if item_text in rendered_corpus:
            continue
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            raw_item_text = item["text"].strip()
            if raw_item_text and raw_item_text in rendered_corpus:
                continue
        if (merged_dict or text_sources) and item_text in ("已完成复核检查。", "独立复核已通过。"):
            continue
        extra_ver_items.append(item_text)
    if extra_ver_items:
        blocks.append(
            NormalizedBlock(
                kind="list",
                title="验证摘要条目",
                items=tuple(extra_ver_items),
            )
        )

    # Derive verification section badge
    badge = ""
    badge_tone = "neutral"
    if accepted_flag is True:
        badge = "通过"
        badge_tone = "ok"
    elif accepted_flag is False:
        if has_reverification:
            badge = "未复签 · 已源码修正"
        else:
            badge = "待修订"
        badge_tone = "warning"
    elif merged_dict.get("status"):
        badge = str(merged_dict["status"])
        badge_tone = "info"

    return blocks, badge, badge_tone, accepted_flag, has_reverification


# ---------------------------------------------------------------------------
# Agent output normalization
# ---------------------------------------------------------------------------


def _normalize_parsed_mapping(payload: dict[str, Any]) -> list[NormalizedBlock]:
    """Convert an unwrapped dict payload (handoff/review/generic) into semantic blocks."""
    blocks: list[NormalizedBlock] = []
    handled: set[str] = set()

    if payload.get("error"):
        handled.add("error")
        err_text = _format_scalar(payload["error"])
        if err_text:
            blocks.append(
                NormalizedBlock(
                    kind="callout",
                    title="执行错误",
                    text=err_text,
                    level="error",
                )
            )

    # Reviewer JSON payload shape: {approve: bool, issues: [...]}
    if isinstance(payload.get("approve"), bool):
        handled.add("approve")
        issues = payload.get("issues")
        issue_count = len(issues) if isinstance(issues, list) else 0
        if payload["approve"]:
            msg = "评审通过" + (f"（附 {issue_count} 项轻微建议）" if issue_count else "。")
            level = "success"
        else:
            msg = f"评审未通过（发现 {issue_count} 项问题）。"
            level = "warning"
        blocks.append(
            NormalizedBlock(kind="callout", title="评审判定", text=msg, level=level)
        )

    # Core narrative keys
    conclusion = str(payload.get("conclusion") or "").strip()
    summary = str(payload.get("summary") or "").strip()
    observation = str(payload.get("observation") or "").strip()

    if conclusion:
        handled.add("conclusion")
        blocks.append(_text_or_markdown_block(conclusion, title="核心结论"))
    if summary:
        handled.add("summary")
        if _canonical_key(summary) != _canonical_key(conclusion):
            blocks.append(_text_or_markdown_block(summary, title="摘要"))
    if observation:
        handled.add("observation")
        blocks.append(_text_or_markdown_block(observation, title="关键观察"))

    if payload.get("deliverables") is not None:
        handled.add("deliverables")
        blocks.extend(_build_deliverables_blocks(payload["deliverables"], title="交付成果"))

    for report_key, report_title in (("final_report", "最终报告"), ("report", "报告")):
        if payload.get(report_key) is not None:
            handled.add(report_key)
            blocks.extend(_build_generic_blocks(payload[report_key], title=report_title))

    # Nested output/handoff inside an agent result
    if payload.get("output") is not None:
        handled.add("output")
        out_val = payload["output"]
        if isinstance(out_val, dict):
            blocks.extend(_normalize_parsed_mapping(out_val))
        elif isinstance(out_val, str) and _canonical_key(out_val) != _canonical_key(conclusion):
            blocks.extend(_build_generic_blocks(out_val, title="输出内容"))
        elif not isinstance(out_val, str):
            blocks.extend(_build_generic_blocks(out_val, title="输出内容"))

    logic_val = payload.get("logic_chain") if "logic_chain" in payload else payload.get("logicChain")
    if logic_val is not None:
        handled.update(("logic_chain", "logicChain"))
        blocks.extend(_build_list_or_table_blocks(logic_val, title="分析推导"))

    for risk_key, risk_title in (
        ("caveats", "注意事项与边界"),
        ("risks", "风险提示"),
        ("findings", "关键发现"),
        ("issues", "发现问题"),
    ):
        if payload.get(risk_key) is not None:
            handled.add(risk_key)
            blocks.extend(
                _build_list_or_table_blocks(
                    payload[risk_key],
                    title=risk_title,
                    preferred_cols=("severity", "description", "issue", "evidence", "location", "suggestion"),
                    callout_if_scalar=True,
                )
            )

    if payload.get("verification") is not None:
        handled.add("verification")
        ver_blocks, _, _, _, _ = _build_verification_blocks(synth_ver=payload["verification"])
        blocks.extend(ver_blocks)

    for next_key in ("next_steps", "recommendations"):
        if payload.get(next_key) is not None:
            handled.add(next_key)
            blocks.extend(_build_list_or_table_blocks(payload[next_key], title="后续建议"))

    # Any remaining keys
    remaining = {k: v for k, v in payload.items() if k not in handled and v is not None}
    if remaining:
        blocks.extend(_build_generic_blocks(remaining))

    return blocks


def normalize_agent_output(raw_result: str | None) -> NormalizedAgentView:
    """Convert an individual ``agent.result`` payload into semantic blocks.

    Supports single- or double-encoded JSON (including fenced ```` ```json ```` blocks),
    Teamwork handoff objects, Markdown reports, error banners, and plain prose while
    preserving ``raw_text = str(raw_result or "")`` verbatim.
    """
    raw_text = str(raw_result or "")
    stripped = raw_text.strip()
    if not stripped:
        return NormalizedAgentView(summary_blocks=(), raw_text=raw_text, has_structured_view=False)

    unwrapped = unwrap_nested_json(stripped)
    if isinstance(unwrapped, dict):
        blocks = tuple(_normalize_parsed_mapping(unwrapped))
        if not blocks:
            blocks = (NormalizedBlock(kind="json", text=json.dumps(unwrapped, ensure_ascii=False, indent=2)),)
        has_structured = any(b.kind != "json" for b in blocks)
        return NormalizedAgentView(
            summary_blocks=blocks,
            raw_text=raw_text,
            has_structured_view=has_structured,
        )

    if isinstance(unwrapped, list):
        blocks = tuple(_build_generic_blocks(unwrapped))
        has_structured = any(b.kind != "json" for b in blocks)
        return NormalizedAgentView(
            summary_blocks=blocks,
            raw_text=raw_text,
            has_structured_view=has_structured,
        )

    # Non-JSON string
    if stripped.startswith("❌"):
        return NormalizedAgentView(
            summary_blocks=(NormalizedBlock(kind="callout", text=stripped, level="error"),),
            raw_text=raw_text,
            has_structured_view=False,
        )
    if stripped.startswith("⚠️"):
        return NormalizedAgentView(
            summary_blocks=(NormalizedBlock(kind="callout", text=stripped, level="warning"),),
            raw_text=raw_text,
            has_structured_view=False,
        )

    if _looks_like_markdown(stripped):
        return NormalizedAgentView(
            summary_blocks=(NormalizedBlock(kind="markdown", text=stripped),),
            raw_text=raw_text,
            has_structured_view=True,
        )

    return NormalizedAgentView(
        summary_blocks=(NormalizedBlock(kind="prose", text=stripped),),
        raw_text=raw_text,
        has_structured_view=False,
    )


# ---------------------------------------------------------------------------
# Workflow project report normalization
# ---------------------------------------------------------------------------


def _derive_verdict(
    project: WorkflowProject,
    *,
    card_summary: dict[str, Any],
    accepted_flag: bool | None,
    has_error: bool,
) -> tuple[str, str]:
    """Determine normalized verdict code and localized label."""
    if project.status == WorkflowStatus.CANCELLED:
        return "cancelled", _VERDICT_LABELS["cancelled"]

    raw_verdict = str(card_summary.get("verdict") or "").strip().lower()
    if raw_verdict in _VERDICT_LABELS:
        return raw_verdict, _VERDICT_LABELS[raw_verdict]

    if has_error or project.status == WorkflowStatus.FAILED:
        return "failed", _VERDICT_LABELS["failed"]
    if accepted_flag is False:
        return "needs_attention", _VERDICT_LABELS["needs_attention"]
    if accepted_flag is True:
        return "passed", _VERDICT_LABELS["passed"]
    if project.status == WorkflowStatus.COMPLETED and str(project.result or "").strip():
        return "passed", _VERDICT_LABELS["passed"]
    return "unknown", _VERDICT_LABELS["unknown"]


def _count_deliverable_badge(deliverables: Any) -> str:
    """Create a concise badge string summarizing grouped/flat deliverables."""
    if not isinstance(deliverables, list) or not deliverables:
        return ""
    if all(isinstance(g, dict) and isinstance(g.get("items"), list) for g in deliverables):
        group_count = len(deliverables)
        item_count = sum(len(g.get("items") or ()) for g in deliverables)
        return f"{group_count} 组 / {item_count} 项"
    return f"{len(deliverables)} 项"


def _merge_handoff_and_output(
    main_dict: dict[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    """Merge Teamwork ``handoff`` and unwrapped ``output`` into a unified mapping."""
    raw_handoff = main_dict.get("handoff")
    handoff: dict[str, Any] = raw_handoff if isinstance(raw_handoff, dict) else {}
    raw_output = main_dict.get("output")
    output_dict: dict[str, Any] = raw_output if isinstance(raw_output, dict) else {}

    merged: dict[str, Any] = {}
    consumed_main_keys: set[str] = set()
    if handoff:
        consumed_main_keys.add("handoff")

    # Start with handoff keys, then merge output_dict keys cleanly without duplicating values.
    for source in (handoff, output_dict):
        for k, v in source.items():
            if v is None:
                continue
            if k not in merged or merged[k] in (None, "", []):
                merged[k] = v
            elif isinstance(merged[k], list) and isinstance(v, list):
                merged[k] = _dedupe_sequence([*merged[k], *v])
            elif isinstance(merged[k], dict) and isinstance(v, dict):
                combined = dict(merged[k])
                for sk, sv in v.items():
                    if sk not in combined or combined[sk] in (None, "", []):
                        combined[sk] = sv
                    elif isinstance(combined[sk], list) and isinstance(sv, list):
                        combined[sk] = _dedupe_sequence([*combined[sk], *sv])
                merged[k] = combined

    if output_dict:
        consumed_main_keys.add("output")

    # Also absorb direct keys on main_dict when not inside handoff/output.
    for k, v in main_dict.items():
        if k in ("handoff", "output", "verified_output", "verification", "card_summary", "result"):
            continue
        if k in (
            "observation",
            "logic_chain",
            "logicChain",
            "caveats",
            "risks",
            "findings",
            "issues",
            "conclusion",
            "summary",
            "deliverables",
            "artifacts",
            "final_report",
            "report",
            "next_steps",
            "recommendations",
            "error",
            "stage",
            "fallback",
        ):
            consumed_main_keys.add(k)
            if k not in merged or merged[k] in (None, "", []):
                merged[k] = v
            elif isinstance(merged[k], list) and isinstance(v, list):
                merged[k] = _dedupe_sequence([*merged[k], *v])

    return merged, consumed_main_keys


def normalize_workflow_report(project: WorkflowProject) -> NormalizedWorkflowReport:
    """Normalize a :class:`WorkflowProject` into deduplicated report sections and agent views."""
    agent_views: dict[tuple[int, int], NormalizedAgentView] = {
        (phase_idx, agent_idx): normalize_agent_output(agent.result)
        for phase_idx, phase in enumerate(project.phases)
        for agent_idx, agent in enumerate(phase.agents)
    }

    raw_result = str(project.result or "").strip()
    if not raw_result:
        verdict, verdict_label = _derive_verdict(
            project, card_summary={}, accepted_flag=None, has_error=bool(project.error)
        )
        if project.error:
            headline = project.error.strip()
            blocks = (
                NormalizedBlock(
                    kind="callout",
                    title="执行错误",
                    text=headline,
                    level="error",
                ),
            )
        elif project.status == WorkflowStatus.CANCELLED:
            headline = "任务已取消，未生成最终结果。"
            blocks = (
                NormalizedBlock(
                    kind="callout",
                    title="任务状态",
                    text="本次 Workflow 已取消，未返回最终结果。",
                    level="warning",
                ),
            )
        else:
            headline = "本次 Workflow 没有返回最终结果。"
            blocks = (
                NormalizedBlock(
                    kind="prose",
                    title="执行结果",
                    text=headline,
                ),
            )
        section = NormalizedSection(
            id="summary-conclusion",
            title="核心结论与摘要",
            badge=verdict_label,
            badge_tone=_VERDICT_BADGE_TONES.get(verdict, "neutral"),
            blocks=blocks,
            expanded=True,
        )
        return NormalizedWorkflowReport(
            verdict=verdict,
            verdict_label=verdict_label,
            headline_conclusion=headline,
            sections=(section,),
            agent_views=agent_views,
        )

    unwrapped = unwrap_nested_json(raw_result)

    # Non-dict top-level results (Markdown, plain text, or JSON array)
    if not isinstance(unwrapped, dict):
        verdict, verdict_label = _derive_verdict(
            project, card_summary={}, accepted_flag=None, has_error=bool(project.error)
        )
        non_dict_sections: list[NormalizedSection] = []
        if project.error:
            non_dict_sections.append(
                NormalizedSection(
                    id="summary-conclusion",
                    title="核心结论与摘要",
                    badge=verdict_label,
                    badge_tone="error",
                    blocks=(
                        NormalizedBlock(
                            kind="callout",
                            title="执行错误",
                            text=project.error.strip(),
                            level="error",
                        ),
                    ),
                    expanded=True,
                )
            )
        if isinstance(unwrapped, list):
            headline = f"共返回 {len(unwrapped)} 条结构化结果。"
            non_dict_sections.append(
                NormalizedSection(
                    id="deliverables-details",
                    title="交付成果与详细内容",
                    badge=f"{len(unwrapped)} 项",
                    badge_tone="info",
                    blocks=tuple(_build_generic_blocks(unwrapped, title="执行结果")),
                    expanded=True,
                )
            )
        else:
            text_val = str(unwrapped).strip()
            headline = _extract_headline_conclusion(text_val)
            is_long_md = len(text_val) > 360 or _looks_like_markdown(text_val)
            non_dict_sections.append(
                NormalizedSection(
                    id="deliverables-details" if is_long_md else "summary-conclusion",
                    title="交付成果与详细内容" if is_long_md else "核心结论与摘要",
                    badge=verdict_label,
                    badge_tone=_VERDICT_BADGE_TONES.get(verdict, "neutral"),
                    blocks=(_text_or_markdown_block(text_val, title="执行结果"),),
                    expanded=True,
                )
            )
        return NormalizedWorkflowReport(
            verdict=verdict,
            verdict_label=verdict_label,
            headline_conclusion=headline,
            sections=tuple(non_dict_sections),
            agent_views=agent_views,
        )

    # Dict payload: separate envelope (card_summary, result, verification) vs direct dict
    card_summary = (
        unwrapped["card_summary"]
        if isinstance(unwrapped.get("card_summary"), dict)
        else {}
    )
    top_verification = unwrapped.get("verification")
    envelope_result = unwrapped.get("result")

    extra_top_keys: dict[str, Any] = {}
    known_cs_keys = ("verdict", "conclusion", "findings", "verification", "deliverables", "next_steps")
    for cs_key, cs_val in card_summary.items():
        if cs_key not in known_cs_keys and cs_val is not None:
            extra_top_keys[f"card_summary_{cs_key}"] = cs_val
    if isinstance(envelope_result, dict):
        main_dict = envelope_result
        for k, v in unwrapped.items():
            if k not in ("card_summary", "result", "verification") and v is not None:
                extra_top_keys[k] = v
    else:
        main_dict = unwrapped

    merged, consumed_main_keys = _merge_handoff_and_output(main_dict)
    if not isinstance(envelope_result, dict) and envelope_result is not None and main_dict is unwrapped:
        # Handle envelope {"card_summary": ..., "result": "<str or list>", "verification": ...}
        if isinstance(envelope_result, str) and envelope_result.strip():
            merged.setdefault("output_narrative", envelope_result.strip())
            consumed_main_keys.add("result")
        elif isinstance(envelope_result, list):
            merged.setdefault("deliverables", envelope_result)
            consumed_main_keys.add("result")

    # Verification merging (deduplicate top_verification vs main_dict.verification vs handoff/output.verification)
    inner_ver = merged.pop("verification", None)
    runtime_ver = main_dict.get("verification")
    if runtime_ver is None:
        runtime_ver = top_verification
    elif top_verification is not None and _canonical_key(top_verification) != _canonical_key(runtime_ver):
        # Preserve differing top_verification alongside runtime_ver if somehow distinct.
        extra_top_keys["top_verification"] = top_verification

    ver_blocks, ver_badge, ver_tone, accepted_flag, has_reverification = _build_verification_blocks(
        synth_ver=inner_ver,
        runtime_ver=runtime_ver,
        card_ver=card_summary.get("verification"),
    )

    has_error = bool(project.error or merged.get("error"))
    verdict, verdict_label = _derive_verdict(
        project,
        card_summary=card_summary,
        accepted_flag=accepted_flag,
        has_error=has_error,
    )

    # -----------------------------------------------------------------------
    # 1) 核心结论与摘要
    # -----------------------------------------------------------------------
    card_conclusion = str(card_summary.get("conclusion") or "").strip()
    merged_conclusion = str(merged.get("conclusion") or "").strip()
    summary_text = str(merged.get("summary") or "").strip()
    observation_text = str(merged.get("observation") or "").strip()

    conclusion_candidates = _dedupe_sequence(
        [card_conclusion, merged_conclusion, summary_text]
    )
    primary_conclusion = conclusion_candidates[0] if conclusion_candidates else ""
    if not primary_conclusion and has_error:
        primary_conclusion = str(project.error or merged.get("error") or "").strip()

    if primary_conclusion:
        headline_conclusion = _extract_headline_conclusion(primary_conclusion)
    elif project.status == WorkflowStatus.FAILED or has_error:
        headline_conclusion = "任务未完成，请查看错误与验证详情。"
    elif project.status == WorkflowStatus.CANCELLED:
        headline_conclusion = "任务已取消，未生成完整最终结果。"
    elif project.status == WorkflowStatus.COMPLETED:
        headline_conclusion = "任务已完成，完整结构化结果见下方报告。"
    else:
        headline_conclusion = ""

    sec1_blocks: list[NormalizedBlock] = []
    if project.error:
        sec1_blocks.append(
            NormalizedBlock(
                kind="callout",
                title="Workflow 错误",
                text=project.error.strip(),
                level="error",
            )
        )
    if merged.get("error") and _canonical_key(merged["error"]) != _canonical_key(project.error):
        sec1_blocks.append(
            NormalizedBlock(
                kind="callout",
                title="执行失败详情",
                text=_format_scalar(merged["error"]),
                level="error",
            )
        )
    if primary_conclusion:
        sec1_blocks.append(_text_or_markdown_block(primary_conclusion, title="核心结论"))
    for extra_conc in conclusion_candidates[1:]:
        sec1_blocks.append(_text_or_markdown_block(extra_conc, title="补充摘要"))
    if observation_text and _canonical_key(observation_text) != _canonical_key(primary_conclusion):
        sec1_blocks.append(_text_or_markdown_block(observation_text, title="事实观察与基线"))

    # -----------------------------------------------------------------------
    # 2) 交付成果与详细内容 (+ verified_output handling)
    # -----------------------------------------------------------------------
    sec2_blocks: list[NormalizedBlock] = []

    deliverables_val = merged.get("deliverables")
    card_deliverables = card_summary.get("deliverables")
    if not deliverables_val and card_deliverables:
        deliverables_val = card_deliverables
    elif isinstance(deliverables_val, list) and isinstance(card_deliverables, list) and card_deliverables:
        # Merge card_deliverables only if not grouped or already represented.
        if not all(isinstance(g, dict) and "items" in g for g in deliverables_val):
            deliverables_val = _dedupe_sequence([*deliverables_val, *card_deliverables])

    if deliverables_val:
        sec2_blocks.extend(_build_deliverables_blocks(deliverables_val, title="交付成果"))

    if merged.get("artifacts") is not None:
        if _canonical_key(merged["artifacts"]) != _canonical_key(deliverables_val):
            sec2_blocks.extend(_build_deliverables_blocks(merged["artifacts"], title="产物清单"))

    rendered_deliverable_texts: set[str] = {
        _canonical_key(primary_conclusion),
    }
    for rep_key, rep_title in (
        ("final_report", "最终报告"),
        ("report", "完整报告"),
        ("output_narrative", "执行输出"),
    ):
        val = merged.get(rep_key)
        if val is None:
            continue
        ck = _canonical_key(val)
        if ck in rendered_deliverable_texts:
            continue
        rendered_deliverable_texts.add(ck)
        sec2_blocks.extend(_build_generic_blocks(val, title=rep_title))

    # Non-dict string `output` alongside `handoff`
    if isinstance(main_dict.get("output"), str):
        raw_out_str = main_dict["output"].strip()
        if raw_out_str and _canonical_key(raw_out_str) not in rendered_deliverable_texts:
            rendered_deliverable_texts.add(_canonical_key(raw_out_str))
            sec2_blocks.append(_text_or_markdown_block(raw_out_str, title="输出正文"))

    # verified_output draft vs final handling
    verified_output = main_dict.get("verified_output")
    draft_section: NormalizedSection | None = None
    if verified_output is not None and (not isinstance(verified_output, str) or verified_output.strip()):
        vo_key = _canonical_key(verified_output)
        already_shown = vo_key in rendered_deliverable_texts or (
            isinstance(main_dict.get("output"), str)
            and vo_key == _canonical_key(main_dict["output"])
        )
        if not already_shown:
            is_superseded_draft = (
                accepted_flag is False
                and (has_reverification or bool(deliverables_val))
            )
            if is_superseded_draft:
                draft_blocks: list[NormalizedBlock] = [
                    NormalizedBlock(
                        kind="callout",
                        title="版本提示",
                        text=(
                            "以下内容为自动评审阶段产出的修订前初稿（评审结果 accepted=false）。"
                            "综合器已基于源码核查修正全部问题，正式结论请以上方"
                            "「核心结论与摘要」及「交付成果与详细内容」为准。"
                        ),
                        level="warning",
                    )
                ]
                if isinstance(verified_output, dict):
                    draft_blocks.extend(_normalize_parsed_mapping(verified_output))
                else:
                    draft_blocks.extend(
                        _build_generic_blocks(verified_output, title="评审阶段初稿正文")
                    )
                draft_section = NormalizedSection(
                    id="verified-output-draft",
                    title="补充参考：评审阶段初稿（部分待修订）",
                    badge="已修订初稿",
                    badge_tone="warning",
                    blocks=tuple(draft_blocks),
                    expanded=False,
                )
            else:
                if isinstance(verified_output, dict):
                    sec2_blocks.extend(_normalize_parsed_mapping(verified_output))
                else:
                    sec2_blocks.extend(
                        _build_generic_blocks(verified_output, title="已验证输出")
                    )

    deliverable_badge = _count_deliverable_badge(deliverables_val)

    # -----------------------------------------------------------------------
    # 3) 分析推导与注意事项
    # -----------------------------------------------------------------------
    sec3_blocks: list[NormalizedBlock] = []
    logic_val = merged.get("logic_chain") if "logic_chain" in merged else merged.get("logicChain")
    if logic_val is not None:
        sec3_blocks.extend(
            _build_list_or_table_blocks(logic_val, title="分析推导链路")
        )

    if merged.get("caveats") is not None:
        sec3_blocks.extend(
            _build_list_or_table_blocks(
                merged["caveats"],
                title="注意事项与限制边界",
                callout_if_scalar=True,
                callout_level="warning",
            )
        )

    combined_findings: list[Any] = []
    for f_key in ("findings", "risks", "issues"):
        f_val = merged.get(f_key)
        if isinstance(f_val, list):
            combined_findings.extend(f_val)
        elif f_val is not None:
            sec3_blocks.extend(
                _build_list_or_table_blocks(
                    f_val,
                    title=_humanize_label(f_key),
                    callout_if_scalar=True,
                    callout_level="warning",
                )
            )
    if isinstance(card_summary.get("findings"), list):
        combined_findings.extend(card_summary["findings"])
    combined_findings = _dedupe_sequence(combined_findings)
    if combined_findings:
        sec3_blocks.extend(
            _build_list_or_table_blocks(
                combined_findings,
                title="关键发现与风险",
                preferred_cols=("severity", "text", "description", "issue", "evidence", "status"),
            )
        )

    # -----------------------------------------------------------------------
    # 4) 评审与质量验证 (ver_blocks built above)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 5) 后续建议
    # -----------------------------------------------------------------------
    sec5_blocks: list[NormalizedBlock] = []
    next_items: list[Any] = []
    for n_key in ("next_steps", "recommendations"):
        n_val = merged.get(n_key)
        if isinstance(n_val, list):
            next_items.extend(n_val)
        elif n_val is not None:
            sec5_blocks.extend(_build_list_or_table_blocks(n_val, title="后续建议"))
    if isinstance(card_summary.get("next_steps"), list):
        next_items.extend(card_summary["next_steps"])
    next_items = _dedupe_sequence(next_items)
    if next_items:
        sec5_blocks.extend(_build_list_or_table_blocks(next_items, title="建议行动"))

    # -----------------------------------------------------------------------
    # 6) Agent 原始输出与扩展字段
    # -----------------------------------------------------------------------
    sec6_blocks: list[NormalizedBlock] = []
    extra_kv: list[tuple[str, str]] = []

    # Collect unhandled keys from main_dict, merged (custom fields inside output), and envelope
    known_merged_keys = {
        "observation",
        "logic_chain",
        "logicChain",
        "caveats",
        "risks",
        "findings",
        "issues",
        "conclusion",
        "summary",
        "deliverables",
        "artifacts",
        "final_report",
        "report",
        "output_narrative",
        "next_steps",
        "recommendations",
        "error",
    }
    known_main_keys = {
        *consumed_main_keys,
        "handoff",
        "output",
        "verified_output",
        "verification",
        "card_summary",
        "result",
    }

    extra_sources: dict[str, Any] = {}
    for k, v in merged.items():
        if k not in known_merged_keys and v is not None:
            extra_sources[k] = v
    for k, v in main_dict.items():
        if k not in known_main_keys and v is not None and k not in extra_sources:
            extra_sources[k] = v
    for k, v in extra_top_keys.items():
        if v is not None and k not in extra_sources:
            extra_sources[k] = v

    # Structured Agent output collection keys first
    for col_key in _AGENT_OUTPUT_COLLECTION_KEYS:
        if col_key in extra_sources:
            col_val = extra_sources.pop(col_key)
            sec6_blocks.extend(
                _build_agent_outputs_blocks(_humanize_label(col_key), col_val)
            )

    # Remaining extra keys (e.g. preset, scout, plan, stage, fallback, custom metrics)
    complex_extras: list[tuple[str, Any]] = []
    for k, v in extra_sources.items():
        if v is None:
            continue
        if isinstance(v, (int, float, bool)) or (
            isinstance(v, str)
            and len(v.strip()) <= 160
            and "\n" not in v
            and not _looks_like_markdown(v)
        ):
            formatted = _format_scalar(v, key=k)
            if formatted:
                extra_kv.append((_humanize_label(k), formatted))
        elif isinstance(v, (list, dict)) and not v:
            continue
        else:
            complex_extras.append((k, v))

    if extra_kv:
        sec6_blocks.insert(
            0,
            NormalizedBlock(kind="kv", title="运行元数据与扩展字段", pairs=tuple(extra_kv)),
        )
    for k, v in complex_extras:
        sec6_blocks.extend(_build_generic_blocks(v, title=_humanize_label(k)))

    # -----------------------------------------------------------------------
    # Assemble non-empty sections in logical reading order
    # -----------------------------------------------------------------------
    sections: list[NormalizedSection] = []
    if sec1_blocks:
        sections.append(
            NormalizedSection(
                id="summary-conclusion",
                title="核心结论与摘要",
                badge=verdict_label,
                badge_tone=_VERDICT_BADGE_TONES.get(verdict, "neutral"),
                blocks=tuple(sec1_blocks),
                expanded=True,
            )
        )
    if sec2_blocks:
        sections.append(
            NormalizedSection(
                id="deliverables-details",
                title="交付成果与详细内容",
                badge=deliverable_badge,
                badge_tone="info" if deliverable_badge else "neutral",
                blocks=tuple(sec2_blocks),
                expanded=True,
            )
        )
    if sec3_blocks:
        sections.append(
            NormalizedSection(
                id="reasoning-caveats",
                title="分析推导与注意事项",
                blocks=tuple(sec3_blocks),
                expanded=True,
            )
        )
    if ver_blocks:
        sections.append(
            NormalizedSection(
                id="verification-qa",
                title="评审与质量验证",
                badge=ver_badge,
                badge_tone=ver_tone,
                blocks=tuple(ver_blocks),
                expanded=True,
            )
        )
    if sec5_blocks:
        sections.append(
            NormalizedSection(
                id="next-steps",
                title="后续建议",
                blocks=tuple(sec5_blocks),
                expanded=True,
            )
        )
    if draft_section is not None:
        sections.append(draft_section)
    if sec6_blocks:
        sections.append(
            NormalizedSection(
                id="agent-raw-extra",
                title="Agent 原始输出与扩展字段",
                blocks=tuple(sec6_blocks),
                expanded=False,
                is_debug_raw=False,
            )
        )

    return NormalizedWorkflowReport(
        verdict=verdict,
        verdict_label=verdict_label,
        headline_conclusion=headline_conclusion,
        sections=tuple(sections),
        agent_views=agent_views,
    )
