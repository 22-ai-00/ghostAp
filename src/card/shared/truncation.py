"""Shared payload truncation utilities for Feishu card content.

Moved from src.card.render.payload_truncator to src.card.shared so that both
render and delivery layers can import without violating the uni-directional
dependency rule (handler → session → render / delivery).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.card.engine_meta import engine_type_to_cmd
from src.card.thresholds import THRESHOLDS
from src.card.ui_text import UI_TEXT
from src.utils.errors import get_error_detail

logger = logging.getLogger(__name__)

FEISHU_CARD_TABLE_LIMIT = 5
_TABLE_WARNING_CONTENT = (
    "⚠️ 表格数量超过飞书卡片限制，已将 Markdown 表格按代码块展示，避免卡片发送失败。"
)
_FENCE_LINE_RE = re.compile(
    r"^\s{0,3}(?P<fence>`{3,}|~{3,})(?P<tail>.*)$"
)
_TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_FEISHU_CARD_CONTENT_MAX_BYTES = 30 * 1024


def count_tagged_nodes(obj: Any) -> int:
    """Recursively count dicts containing a ``"tag"`` key (Feishu element nodes)."""
    count = 0
    if isinstance(obj, dict):
        if "tag" in obj:
            count += 1
        for v in obj.values():
            count += count_tagged_nodes(v)
    elif isinstance(obj, list):
        for item in obj:
            count += count_tagged_nodes(item)
    return count


def count_markdown_table_blocks(text: str) -> int:
    """Count markdown pipe-table blocks outside fenced code blocks."""
    return _rewrite_markdown_tables_as_code(text, rewrite=False)[1]


def normalize_markdown_tables_for_card(
    text: str,
    *,
    table_limit: int = FEISHU_CARD_TABLE_LIMIT,
    include_warning: bool = True,
) -> str:
    """Rewrite an over-limit Markdown table set as inert code blocks.

    Text at or below Feishu's table limit is returned unchanged. When a single
    Markdown value contains more than the limit, every recognized table block is
    rewritten and one user-facing warning is appended.
    """
    if count_markdown_table_blocks(text) <= max(0, table_limit):
        return text

    rewritten, _ = _rewrite_markdown_tables_as_code(text, rewrite=True)
    if not include_warning:
        return rewritten
    return f"{rewritten}\n\n{_TABLE_WARNING_CONTENT}"


def _count_explicit_table_nodes(obj: Any) -> int:
    if isinstance(obj, dict):
        count = 1 if obj.get("tag") == "table" else 0
        return count + sum(_count_explicit_table_nodes(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_explicit_table_nodes(item) for item in obj)
    return 0


def _count_markdown_tables_in_payload(obj: Any) -> int:
    if isinstance(obj, dict):
        count = 0
        if obj.get("tag") == "markdown" and isinstance(obj.get("content"), str):
            count += count_markdown_table_blocks(obj["content"])
        return count + sum(_count_markdown_tables_in_payload(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_markdown_tables_in_payload(item) for item in obj)
    return 0


def _guard_feishu_table_limit(card: dict) -> tuple[dict, bool]:
    """Neutralize markdown tables when Feishu would reject the card.

    Feishu reports markdown-table overflows as card table component overflows even
    when the JSON has no explicit ``tag=table`` node.
    """
    table_count = (
        _count_explicit_table_nodes(card)
        + _count_markdown_tables_in_payload(card)
    )
    if table_count <= FEISHU_CARD_TABLE_LIMIT:
        return card, False

    guarded = json.loads(json.dumps(card))
    changed = _rewrite_markdown_tables_in_payload(guarded)
    if changed:
        _append_table_limit_warning(guarded)
        logger.warning(
            "Feishu card markdown table count %d exceeds limit %d; "
            "converted markdown tables to code blocks",
            table_count,
            FEISHU_CARD_TABLE_LIMIT,
        )
    return guarded, changed


def _rewrite_markdown_tables_in_payload(obj: Any) -> bool:
    changed = False
    if isinstance(obj, dict):
        if obj.get("tag") == "markdown" and isinstance(obj.get("content"), str):
            rewritten, count = _rewrite_markdown_tables_as_code(
                obj["content"],
                rewrite=True,
            )
            if count:
                obj["content"] = rewritten
                changed = True
        for value in obj.values():
            changed = _rewrite_markdown_tables_in_payload(value) or changed
    elif isinstance(obj, list):
        for item in obj:
            changed = _rewrite_markdown_tables_in_payload(item) or changed
    return changed


def _append_table_limit_warning(card: dict) -> None:
    warning = {
        "tag": "markdown",
        "content": _TABLE_WARNING_CONTENT,
        "text_size": "notation",
    }
    body = card.get("body")
    if isinstance(body, dict) and isinstance(body.get("elements"), list):
        body["elements"].append(warning)
        return
    if isinstance(card.get("elements"), list):
        card["elements"].append(warning)


def _rewrite_markdown_tables_as_code(text: str, *, rewrite: bool) -> tuple[str, int]:
    lines = str(text or "").splitlines()
    if not lines:
        return text, 0

    out: list[str] = []
    table_count = 0
    open_fence = ""
    i = 0

    while i < len(lines):
        line = lines[i]
        fence_match = _FENCE_LINE_RE.match(line)
        if fence_match:
            fence = fence_match.group("fence")
            if not open_fence:
                open_fence = fence
            elif (
                fence[0] == open_fence[0]
                and len(fence) >= len(open_fence)
                and not fence_match.group("tail").strip()
            ):
                open_fence = ""
            out.append(line)
            i += 1
            continue

        if (
            not open_fence
            and i + 1 < len(lines)
            and _looks_like_table_row(line)
            and _is_table_separator_line(lines[i + 1])
        ):
            end = i + 2
            while end < len(lines) and _looks_like_table_row(lines[end]):
                end += 1
            table_lines = lines[i:end]
            table_count += 1
            if rewrite:
                out.extend(("```text", *table_lines, "```"))
            else:
                out.extend(table_lines)
            i = end
            continue

        out.append(line)
        i += 1

    if not rewrite or table_count == 0:
        return text, table_count
    return "\n".join(out), table_count


def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and "|" in stripped


def _is_table_separator_line(line: str) -> bool:
    stripped = line.strip().strip("|")
    if "|" not in line or not stripped:
        return False
    cells = [cell.strip().replace(" ", "") for cell in stripped.split("|")]
    non_empty = [cell for cell in cells if cell]
    return bool(non_empty) and all(_TABLE_SEPARATOR_RE.match(cell) for cell in non_empty)


def check_and_truncate_payload(
    card_content: str, max_size: int | None = None, *, engine_type: str | None = None
) -> str:
    """Check if card content exceeds size limit and truncate if necessary.

    Attempts to preserve JSON structure while truncating text fields.
    Also checks tagged-node count against CARD_NODE_BUDGET.
    """
    if max_size is None:
        max_size = THRESHOLDS["CARD_BYTE_BUDGET"]
    node_budget = THRESHOLDS["CARD_NODE_BUDGET"]

    if len(card_content.encode("utf-8")) <= max_size:
        # Still check node count even if byte size is OK
        try:
            card_check = json.loads(card_content)
            table_guarded_card, table_guarded = _guard_feishu_table_limit(card_check)
            if table_guarded:
                card_check = table_guarded_card
            if count_tagged_nodes(card_check) > node_budget:
                logger.warning(
                    "Card node count %d exceeds budget %d, will truncate",
                    count_tagged_nodes(card_check),
                    node_budget,
                )
                # Fall through to truncation logic
            else:
                if table_guarded:
                    guarded_content = json.dumps(
                        table_guarded_card,
                        ensure_ascii=False,
                    )
                    if len(guarded_content.encode("utf-8")) <= max_size:
                        return guarded_content
                    card_content = guarded_content
                else:
                    return card_content
        except (json.JSONDecodeError, Exception):
            return card_content
    else:
        pass  # Fall through to truncation logic

    logger.warning(
        "Card payload exceeds Feishu guard budget: size=%d/%d bytes nodes=%s/%d; attempting truncation",
        len(card_content.encode("utf-8")),
        max_size,
        _safe_count_nodes(card_content),
        node_budget,
    )

    try:
        card = json.loads(card_content)
        card, table_guarded = _guard_feishu_table_limit(card)
        if table_guarded:
            card_content = json.dumps(card, ensure_ascii=False)
            if (
                len(card_content.encode("utf-8")) <= max_size
                and count_tagged_nodes(card) <= THRESHOLDS["CARD_NODE_BUDGET"]
            ):
                return card_content

        # Helper to recursively truncate strings in the dict
        def truncate_recursive(obj: Any, depth: int = 0) -> Any:
            if depth > 20:  # Anti-recursion depth limit
                return obj

            if isinstance(obj, dict):
                for k, v in obj.items():
                    # Don't truncate structural keys
                    if k in ("tag", "type", "actions", "elements", "modules", "columns", "fields"):
                        obj[k] = truncate_recursive(v, depth + 1)
                    # Content fields - aggressive truncation if needed
                    elif k in ("content", "text", "value", "placeholder", "alt"):
                        # Only truncate when truly large — small strings are
                        # labels/buttons and should never be mangled.
                        if isinstance(v, str) and len(v) > 8000:
                            obj[k] = _truncate_markdown_text(v, 8000, suffix=UI_TEXT["truncation_suffix"])
                        else:
                            obj[k] = truncate_recursive(v, depth + 1)
                    else:
                        obj[k] = truncate_recursive(v, depth + 1)
            elif isinstance(obj, list):
                # If list is too long, truncate items
                if len(obj) > 60:
                    obj = obj[:60]

                for i in range(len(obj)):
                    obj[i] = truncate_recursive(obj[i], depth + 1)
            elif isinstance(obj, str):
                # Fallback for strings in other locations
                if len(obj) > 10000:
                    return _truncate_markdown_text(obj, 10000, suffix=UI_TEXT["truncation_suffix"])
            return obj

        # First pass: try smart truncation on deep content
        card_copy = json.loads(json.dumps(card))  # Deep copy
        truncated_card = truncate_recursive(card_copy)

        # Add a warning note to the card body if possible
        cmd_hint = engine_type_to_cmd(engine_type, fallback="")
        if cmd_hint:
            trunc_msg = UI_TEXT["truncation_warning_with_cmd"].format(cmd_hint=cmd_hint)
        else:
            trunc_msg = UI_TEXT["truncation_warning_generic"]
        warning_element = {
            "tag": "markdown",
            "content": trunc_msg,
            "text_size": "notation",
        }

        if "body" in truncated_card and isinstance(truncated_card.get("body", {}).get("elements"), list):
            truncated_card["body"]["elements"].append(warning_element)
        elif isinstance(truncated_card.get("elements"), list):
            truncated_card["elements"].append(warning_element)

        truncated_content = json.dumps(truncated_card, ensure_ascii=False)

        # Double check official Feishu limits after smart truncation. Element
        # overflow is not fixed by shortening text, so fallback to a compact
        # one-markdown card if nested components still exceed the budget.
        if (
            len(truncated_content.encode("utf-8")) > max_size
            or count_tagged_nodes(truncated_card) > node_budget
        ):
            fallback_card = _build_limit_fallback_card(card, engine_type=engine_type)
            fallback_content = json.dumps(fallback_card, ensure_ascii=False)
            if (
                len(fallback_content.encode("utf-8")) > max_size
                or len(fallback_content.encode("utf-8")) > _FEISHU_CARD_CONTENT_MAX_BYTES
                or count_tagged_nodes(fallback_card) > node_budget
            ):
                fallback_card["body"]["elements"][0]["content"] = UI_TEXT["truncation_fallback_prefix"]
                fallback_content = json.dumps(fallback_card, ensure_ascii=False)
            return fallback_content

        return truncated_content
    except Exception as e:
        logger.error("Failed to truncate payload: %s", get_error_detail(e))
        return card_content


def _safe_count_nodes(card_content: str) -> int | str:
    try:
        return count_tagged_nodes(json.loads(card_content))
    except Exception:
        return "unknown"


def _truncate_markdown_text(text: str, limit: int, *, suffix: str) -> str:
    """Truncate markdown without leaving an open fenced code block."""
    truncated = text[:limit].rstrip()
    closing = _open_fence_closer(truncated)
    if closing:
        truncated = f"{truncated}\n{closing}"
    separator = "" if suffix.startswith("\n") else "\n"
    return truncated + separator + suffix


def _open_fence_closer(text: str) -> str:
    """Return the closing fence needed for text, or an empty string."""
    open_fence = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fence = "```"
        elif stripped.startswith("~~~"):
            fence = "~~~"
        else:
            continue
        open_fence = "" if open_fence == fence else fence
    return open_fence


def _build_limit_fallback_card(card: dict, *, engine_type: str | None = None) -> dict:
    """Build a tiny card guaranteed to stay below Feishu element and byte caps."""
    summary_text = UI_TEXT["truncation_fallback_prefix"]
    try:
        extracted = _extract_text(card)
        if extracted:
            summary_text = _truncate_markdown_text(
                extracted,
                len(extracted),
                suffix=UI_TEXT["truncation_fallback_suffix"],
            )
    except Exception:
        logger.debug("failed to extract summary text", exc_info=True)

    cmd_hint = engine_type_to_cmd(engine_type, fallback="")
    if cmd_hint:
        trunc_msg = UI_TEXT["truncation_warning_with_cmd"].format(cmd_hint=cmd_hint)
    else:
        trunc_msg = UI_TEXT["truncation_warning_generic"]
    summary_text = f"{summary_text}\n\n{trunc_msg}"

    return {
        "schema": "2.0",
        "config": {
            **(card.get("config") if isinstance(card.get("config"), dict) else {"wide_screen_mode": True}),
            "update_multi": True,
        },
        "header": _compact_header(card),
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": summary_text,
                }
            ]
        },
    }


def _compact_header(card: dict) -> dict:
    header = card.get("header")
    title_content = UI_TEXT["truncation_card_header"]
    template = "yellow"
    if isinstance(header, dict):
        template = str(header.get("template") or template)
        title = header.get("title")
        if isinstance(title, dict) and isinstance(title.get("content"), str):
            title_content = title["content"][:80] or title_content
    return {
        "template": template,
        "title": {"tag": "plain_text", "content": title_content},
    }


def _extract_text(obj: Any, limit: int = 2000) -> str:
    """Try to extract meaningful text from card content fields."""
    if isinstance(obj, str) and len(obj) > 50:
        return obj[:limit]
    if isinstance(obj, dict):
        for k in ("content", "text"):
            v = obj.get(k)
            if isinstance(v, str) and len(v) > 50:
                return v[:limit]
        for v in obj.values():
            r = _extract_text(v, limit)
            if r:
                return r
    if isinstance(obj, list):
        for item in obj[:5]:
            r = _extract_text(item, limit)
            if r:
                return r
    return ""
