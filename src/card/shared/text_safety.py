"""Pure text-safety helpers shared by card render and delivery layers."""

from __future__ import annotations

import re

_EMAIL_ADDRESS_RE = re.compile(
    r"(?<![\w.+-])[\w.+-]+@(?:[\w-]+\.)+[A-Za-z]{2,}(?![\w.-])"
)
_MARKDOWN_IMAGE_REMOVED_TEXT = "（图片引用已移除）"
_RICH_TEXT_CONTROL_TAGS = frozenset(
    {
        "a",
        "action",
        "at",
        "button",
        "div",
        "font",
        "image",
        "img",
        "script",
        "span",
        "style",
        "text_tag",
    }
)
_MAX_RICH_TEXT_CONTROL_TAG_NAME_LENGTH = max(map(len, _RICH_TEXT_CONTROL_TAGS))
_MARKDOWN_AUTOLINK_RE = re.compile(
    r"<(?P<target>[A-Za-z][A-Za-z0-9+.-]*:[^<>\s]*)>"
)
_HTTP_SCHEMES = frozenset({"http", "https"})
_SAFE_REDACTION_SENTINEL_RE = re.compile(r"<redacted(?::[a-z0-9_]+)?>")


def sanitize_card_text_for_audit(text: str) -> str:
    """Remove text patterns known to trigger Feishu card content audit."""
    if not text:
        return text
    return _EMAIL_ADDRESS_RE.sub("[redacted:email]", text)


def _bounded_plain_image_label(_alt: str) -> str:
    return _MARKDOWN_IMAGE_REMOVED_TEXT


def sanitize_markdown_image_references(text: str) -> str:
    """Neutralize local/remote Markdown image targets before CardKit sees them."""
    if "![" not in text:
        return text

    image_starts = _substring_positions(text, "![")
    target_starts = _substring_positions(text, "](")
    parenthesis_pairs = _matching_delimiter_pairs(text, "(", ")")

    output: list[str] = []
    cursor = 0
    target_cursor = 0
    for start in image_starts:
        if start < cursor:
            continue
        output.append(text[cursor:start])

        while (
            target_cursor < len(target_starts)
            and target_starts[target_cursor] < start + 2
        ):
            target_cursor += 1
        if target_cursor >= len(target_starts):
            output.append("！[")
            cursor = start + 2
            continue

        alt_end = target_starts[target_cursor]
        target_end = parenthesis_pairs.get(alt_end + 1)
        if target_end is None:
            output.append("！[")
            cursor = start + 2
            continue

        output.append(_bounded_plain_image_label(text[start + 2 : alt_end]))
        cursor = target_end + 1
    output.append(text[cursor:])
    return "".join(output)


def neutralize_feishu_rich_text_controls(text: str) -> str:
    """Make model-origin rich-text controls inert while keeping readable text.

    Feishu/HTML-like tags are losslessly full-width escaped. Markdown links are
    preserved only for HTTP(S); all other destinations are removed while their
    labels remain visible.
    """
    if not text:
        return text

    neutralized = _neutralize_unsafe_markdown_links(text)
    neutralized = _MARKDOWN_AUTOLINK_RE.sub(
        _neutralize_unsafe_markdown_autolink,
        neutralized,
    )
    return _neutralize_rich_text_control_tags(neutralized)


def _neutralize_unsafe_markdown_links(text: str) -> str:
    if "](" not in text:
        return text

    bracket_pairs = _matching_delimiter_pairs(text, "[", "]")
    parenthesis_pairs = _matching_delimiter_pairs(text, "(", ")")
    output: list[str] = []
    cursor = 0
    scan = 0

    while scan < len(text):
        start = text.find("[", scan)
        if start < 0:
            break
        if start > 0 and text[start - 1] == "!":
            scan = start + 1
            continue

        label_end = bracket_pairs.get(start)
        if label_end is None or label_end + 1 >= len(text) or text[label_end + 1] != "(":
            scan = start + 1
            continue

        target_end = parenthesis_pairs.get(label_end + 1)
        if target_end is None:
            scan = start + 1
            continue

        target = _markdown_link_destination(text[label_end + 2 : target_end])
        if _is_http_destination(target):
            scan = target_end + 1
            continue

        output.append(text[cursor:start])
        output.append(text[start + 1 : label_end])
        cursor = target_end + 1
        scan = cursor

    if not output:
        return text
    output.append(text[cursor:])
    return "".join(output)


def _matching_delimiter_pairs(
    text: str,
    opener: str,
    closer: str,
) -> dict[int, int]:
    pairs: dict[int, int] = {}
    stack: list[int] = []
    escaped_openers: list[int] = []
    prefix_balance = [0]
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            if char == opener:
                escaped_openers.append(index)
            prefix_balance.append(prefix_balance[-1])
        elif char == "\\":
            escaped = True
            prefix_balance.append(prefix_balance[-1])
        elif char == opener:
            stack.append(index)
            prefix_balance.append(prefix_balance[-1] + 1)
        elif char == closer and stack:
            pairs[stack.pop()] = index
            prefix_balance.append(prefix_balance[-1] - 1)
        elif char == closer:
            prefix_balance.append(prefix_balance[-1] - 1)
        else:
            prefix_balance.append(prefix_balance[-1])

    next_lower = [-1] * len(prefix_balance)
    prefix_stack: list[int] = []
    for index, balance in enumerate(prefix_balance):
        while prefix_stack and balance < prefix_balance[prefix_stack[-1]]:
            next_lower[prefix_stack.pop()] = index
        prefix_stack.append(index)

    for start in escaped_openers:
        target_prefix = next_lower[start + 1]
        if target_prefix >= 0:
            pairs[start] = target_prefix - 1
    return pairs


def _substring_positions(text: str, needle: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return positions
        positions.append(index)
        start = index + 1


def _markdown_link_destination(raw_target: str) -> str:
    target = raw_target.lstrip()
    if target.startswith("<"):
        end = target.find(">", 1)
        return target[1:end] if end >= 0 else target

    escaped = False
    destination: list[str] = []
    for char in target:
        if escaped:
            destination.append(char)
            escaped = False
        elif char == "\\":
            destination.append(char)
            escaped = True
        elif char.isspace():
            break
        else:
            destination.append(char)
    return "".join(destination)


def _is_http_destination(target: str) -> bool:
    scheme_end = target.find(":")
    if scheme_end <= 0:
        return False
    return target[:scheme_end].lower() in _HTTP_SCHEMES


def _neutralize_unsafe_markdown_autolink(match: re.Match[str]) -> str:
    if _SAFE_REDACTION_SENTINEL_RE.fullmatch(match.group(0)):
        return match.group(0)
    target = match.group("target")
    return match.group(0) if _is_http_destination(target) else target


def _neutralize_rich_text_control_tags(text: str) -> str:
    """Full-width dangerous tags with a deterministic, linear-time scan."""
    if "<" not in text:
        return text

    tail_ends = _rich_text_tag_tail_ends(text)
    output: list[str] = []
    cursor = 0
    scan = 0
    while True:
        start = text.find("<", scan)
        if start < 0:
            break

        name_end = _rich_text_control_tag_name_end(text, start)
        if name_end is None:
            scan = start + 1
            continue

        next_char = text[name_end]
        if next_char == ">":
            end = name_end
        elif next_char == "/":
            end = name_end + 1 if text[name_end + 1 : name_end + 2] == ">" else -1
        else:
            end = tail_ends[name_end]

        if end < 0:
            scan = start + 1
            continue

        output.append(text[cursor:start])
        output.append(text[start : end + 1].replace("<", "＜").replace(">", "＞"))
        cursor = end + 1
        scan = cursor

    if not output:
        return text
    output.append(text[cursor:])
    return "".join(output)


def _rich_text_control_tag_name_end(text: str, start: int) -> int | None:
    index = start + 1
    if text[index : index + 1] == "/":
        index += 1
    name_start = index
    name_limit = min(
        len(text),
        name_start + _MAX_RICH_TEXT_CONTROL_TAG_NAME_LENGTH + 1,
    )
    while index < name_limit and (
        text[index].isascii() and (text[index].isalnum() or text[index] == "_")
    ):
        index += 1

    if text[name_start:index].lower() not in _RICH_TEXT_CONTROL_TAGS:
        return None
    if index >= len(text) or not (
        text[index].isspace() or text[index] in {"/", ">"}
    ):
        return None
    return index


def _rich_text_tag_tail_ends(text: str) -> list[int]:
    """Map each tag-tail offset to its first valid unquoted closing bracket."""
    tail_ends = [-1] * (len(text) + 1)
    next_quote = {"'": -1, '"': -1}
    for index in range(len(text) - 1, -1, -1):
        char = text[index]
        if char == ">":
            tail_ends[index] = index
        elif char == "<":
            tail_ends[index] = -1
        elif char in next_quote:
            closing_quote = next_quote[char]
            if closing_quote >= 0:
                tail_ends[index] = tail_ends[closing_quote + 1]
            next_quote[char] = index
        else:
            tail_ends[index] = tail_ends[index + 1]
    return tail_ends


__all__ = [
    "neutralize_feishu_rich_text_controls",
    "sanitize_card_text_for_audit",
    "sanitize_markdown_image_references",
]
