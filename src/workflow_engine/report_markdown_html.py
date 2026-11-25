"""Zero-dependency GFM Markdown-to-HTML converter for Workflow HTML reports.

Uses Python standard library modules only (``html``, ``re``, ``unicodedata``,
``urllib.parse``) and strictly escapes all untrusted raw HTML tags and
attributes so untrusted model or error output cannot inject executable HTML.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit

_ASCII_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_ALLOWED_URL_SCHEMES = frozenset({"http", "https", "mailto"})
_ALERT_KINDS = ("NOTE", "TIP", "IMPORTANT", "WARNING", "CAUTION")

_ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|())$")
_FENCE_OPEN_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
_HR_RE = re.compile(r"^ {0,3}(?:(\*[ \t]*){3,}|(-[ \t]*){3,}|(_[ \t]*){3,})$")
_BLOCKQUOTE_RE = re.compile(r"^ {0,3}> ?(.*)$")
_ALERT_MARKER_RE = re.compile(
    r"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:[ \t]+(.*))?$",
    re.IGNORECASE,
)
_TABLE_ALIGN_CELL_RE = re.compile(r"^:?-{1,}:?$")
_UL_ITEM_RE = re.compile(r"^( *)([-*+])(?:[ \t]+(.*)|())$")
_OL_ITEM_RE = re.compile(r"^( *)(\d{1,9})([.)])(?:[ \t]+(.*)|())$")
_TASK_CHECKBOX_RE = re.compile(r"^\[([ xX])\](?:[ \t]+(.*)|())$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_SPACE_RE = re.compile(r"\s+")

_HEADING_DETECT_RE = re.compile(r"(?m)^ {0,3}#{1,6}[ \t]+\S")
_FENCE_DETECT_RE = re.compile(r"(?m)^ {0,3}(`{3,}|~{3,})[^\n]*\n")
_TABLE_DELIM_DETECT_RE = re.compile(
    r"(?m)^[^\n|]*\|[^\n]*\n {0,3}\|?[ \t]*:?-{1,}:?[ \t]*(?:\|[ \t]*:?-{1,}:?[ \t]*)+\|?[ \t]*$"
)
_SINGLE_COL_TABLE_DETECT_RE = re.compile(
    r"(?m)^ {0,3}\|[^\n|]+\|[ \t]*\n {0,3}\|[ \t]*:?-{1,}:?[ \t]*\|[ \t]*$"
)
_UL_DETECT_RE = re.compile(r"(?m)^ {0,3}[-*+][ \t]+\S")
_OL_DETECT_RE = re.compile(r"(?m)^ {0,3}\d{1,3}[.)][ \t]+\S")
_ALERT_DETECT_RE = re.compile(
    r"(?im)^ {0,3}>[ \t]*\[!(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]"
)
_INLINE_CODE_DETECT_RE = re.compile(r"(`+)([^`\n]+?)\1")
_BOLD_STAR_DETECT_RE = re.compile(r"(?<!\*)\*\*(?=\S)(.+?)(?<=\S)\*\*(?!\*)")
_BOLD_UNDER_DETECT_RE = re.compile(r"(?<![\w])__(?=\S)(.+?)(?<=\S)__(?![\w])")
_STRIKE_DETECT_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_LINK_DETECT_RE = re.compile(
    r"\[[^\]\n]+\]\((?:https?://|mailto:|#|/|\.{0,2}/|[A-Za-z0-9_.-]+/[A-Za-z0-9_./#-]*|[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,10}(?:[#?][^)\s]*)?\))"
)


def _is_unicode_punct(ch: str) -> bool:
    """Return True if ``ch`` is ASCII punctuation or a Unicode punctuation char."""
    if ch in _ASCII_PUNCTUATION:
        return True
    return unicodedata.category(ch).startswith("P")


def _strip_nulls(text: str) -> str:
    """Remove null bytes so internal sentinel tokens cannot be forged."""
    return text.replace("\x00", "")


def _is_safe_url(raw_url: str) -> tuple[bool, str]:
    """Validate a link destination against the safe URL allowlist.

    Allowed schemes: ``http://``, ``https://``, ``mailto:``, ``#anchor``, and
    relative paths without schemes (rejecting protocol-relative ``//...`` and
    ``javascript:``, ``data:``, ``vbscript:``, ``file:``, etc.).
    """
    cleaned = raw_url.strip()
    if not cleaned:
        return False, ""
    if _CONTROL_CHAR_RE.search(cleaned):
        return False, ""

    # Also validate HTML-unescaped + whitespace-collapsed form to defeat
    # character-reference or control-obfuscated schemes like `&#106;avascript:`.
    decoded_compact = _SPACE_RE.sub("", html.unescape(cleaned))
    if _CONTROL_CHAR_RE.search(decoded_compact):
        return False, ""

    for candidate in (cleaned, decoded_compact):
        if candidate.startswith("#"):
            continue
        if candidate.startswith("//"):
            return False, ""
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.lower()
        if scheme:
            if scheme not in _ALLOWED_URL_SCHEMES:
                return False, ""
            if scheme in ("http", "https") and not parsed.netloc:
                return False, ""
            if scheme == "mailto" and not (parsed.path or parsed.netloc):
                return False, ""
        elif parsed.netloc:
            return False, ""

    return True, cleaned


def _unescape_destination(raw: str) -> str:
    """Unescape CommonMark backslash escapes inside a link URL or title."""
    out: list[str] = []
    idx = 0
    length = len(raw)
    while idx < length:
        ch = raw[idx]
        if ch == "\\" and idx + 1 < length and raw[idx + 1] in _ASCII_PUNCTUATION:
            out.append(raw[idx + 1])
            idx += 2
        else:
            out.append(ch)
            idx += 1
    return "".join(out)


def _match_code_span(text: str, start: int) -> tuple[int, str] | None:
    """Match a backtick inline code span starting at ``start``.

    Returns ``(end_index, inner_text)`` or ``None`` if no matching closing
    backtick run of identical length exists.
    """
    length = len(text)
    tick_end = start
    while tick_end < length and text[tick_end] == "`":
        tick_end += 1
    tick_count = tick_end - start
    if tick_count == 0:
        return None

    pos = tick_end
    while pos < length:
        if text[pos] == "`":
            close_end = pos
            while close_end < length and text[close_end] == "`":
                close_end += 1
            if close_end - pos == tick_count:
                content = text[tick_end:pos].replace("\n", " ")
                if (
                    len(content) >= 2
                    and content[0] == " "
                    and content[-1] == " "
                    and content.strip(" ")
                ):
                    content = content[1:-1]
                return close_end, content
            pos = close_end
        else:
            pos += 1
    return None


def _parse_link_destination_and_title(text: str, start: int) -> tuple[int, str, str] | None:
    """Parse ``(url)`` or ``(url "title")`` starting right after ``](``."""
    length = len(text)
    pos = start
    while pos < length and text[pos] in (" ", "\t"):
        pos += 1
    if pos < length and text[pos] == "\n":
        pos += 1
        while pos < length and text[pos] in (" ", "\t"):
            pos += 1
    if pos >= length:
        return None

    if text[pos] == "<":
        pos += 1
        url_start = pos
        while pos < length:
            ch = text[pos]
            if ch == "\\" and pos + 1 < length and text[pos + 1] in _ASCII_PUNCTUATION:
                pos += 2
                continue
            if ch in ("\n", "<"):
                return None
            if ch == ">":
                break
            pos += 1
        if pos >= length or text[pos] != ">":
            return None
        raw_url = text[url_start:pos]
        pos += 1
    else:
        url_start = pos
        paren_depth = 0
        while pos < length:
            ch = text[pos]
            if ch == "\\" and pos + 1 < length and text[pos + 1] in _ASCII_PUNCTUATION:
                pos += 2
                continue
            if ch in (" ", "\t", "\n", "<", ">"):
                break
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                if paren_depth == 0:
                    break
                paren_depth -= 1
            pos += 1
        if paren_depth != 0:
            return None
        raw_url = text[url_start:pos]
        if not raw_url and (pos >= length or text[pos] != ")"):
            return None

    ws_start = pos
    while pos < length and text[pos] in (" ", "\t"):
        pos += 1
    if pos < length and text[pos] == "\n":
        pos += 1
        while pos < length and text[pos] in (" ", "\t"):
            pos += 1

    raw_title = ""
    if pos < length and text[pos] != ")":
        if pos == ws_start:
            return None
        quote_ch = text[pos]
        if quote_ch not in ('"', "'", "("):
            return None
        close_quote = ")" if quote_ch == "(" else quote_ch
        pos += 1
        title_start = pos
        while pos < length:
            ch = text[pos]
            if ch == "\\" and pos + 1 < length and text[pos + 1] in _ASCII_PUNCTUATION:
                pos += 2
                continue
            if quote_ch == "(" and ch == "(":
                return None
            if ch == close_quote:
                break
            pos += 1
        if pos >= length or text[pos] != close_quote:
            return None
        raw_title = text[title_start:pos]
        pos += 1
        while pos < length and text[pos] in (" ", "\t"):
            pos += 1
        if pos < length and text[pos] == "\n":
            pos += 1
            while pos < length and text[pos] in (" ", "\t"):
                pos += 1

    if pos >= length or text[pos] != ")":
        return None

    return pos + 1, _unescape_destination(raw_url), _unescape_destination(raw_title)


def _find_matching_link_end(text: str, open_bracket: int) -> tuple[int, int, str, str] | None:
    """Find a valid ``[label](url)`` starting at ``open_bracket``.

    Returns ``(close_bracket, end_after_paren, url, title)`` or ``None``.
    Code spans take precedence inside the label; nested links inside a label
    prevent the outer bracket pair from becoming a link.
    """
    length = len(text)
    pos = open_bracket + 1
    bracket_depth = 1
    while pos < length:
        ch = text[pos]
        if ch == "\\" and pos + 1 < length and text[pos + 1] in _ASCII_PUNCTUATION:
            pos += 2
            continue
        if ch == "`":
            code_match = _match_code_span(text, pos)
            if code_match is not None:
                pos = code_match[0]
                continue
            while pos < length and text[pos] == "`":
                pos += 1
            continue
        if ch == "[":
            # If an inner bracket starts a valid link, outer `[` cannot be a link.
            if "](" in text[pos:]:
                inner = _find_matching_link_end(text, pos)
                if inner is not None:
                    return None
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth == 0:
                if pos + 1 < length and text[pos + 1] == "(":
                    dest = _parse_link_destination_and_title(text, pos + 2)
                    if dest is not None:
                        end_idx, url, title = dest
                        return pos, end_idx, url, title
                return None
        pos += 1
    return None


@dataclass
class _InlineSegment:
    """One atomic unit used by the emphasis/strikethrough delimiter stack."""

    kind: str  # "text", "delim", "br", "nl", "token"
    value: str
    char: str = ""
    can_open: bool = False
    can_close: bool = False
    orig_len: int = 0
    active: bool = True


def _flanking_flags(
    prev_ch: str | None,
    next_ch: str | None,
    delim_char: str,
    run_len: int,
) -> tuple[bool, bool]:
    """Compute CommonMark ``(can_open, can_close)`` flags for a delimiter run."""
    prev_ws = prev_ch is None or prev_ch in (" ", "\t", "\n")
    next_ws = next_ch is None or next_ch in (" ", "\t", "\n")
    prev_punct = prev_ch is not None and not prev_ws and prev_ch != "\x00" and _is_unicode_punct(prev_ch)
    next_punct = next_ch is not None and not next_ws and next_ch != "\x00" and _is_unicode_punct(next_ch)

    left_flanking = (not next_ws) and ((not next_punct) or prev_ws or prev_punct)
    right_flanking = (not prev_ws) and ((not prev_punct) or next_ws or next_punct)

    if delim_char == "_":
        can_open = left_flanking and ((not right_flanking) or prev_punct)
        can_close = right_flanking and ((not left_flanking) or next_punct)
        return can_open, can_close
    if delim_char == "~":
        if run_len != 2:
            return False, False
        return left_flanking, right_flanking
    return left_flanking, right_flanking


def _tokenize_inline_segments(
    text: str,
    tokens: dict[str, str],
) -> list[_InlineSegment]:
    """Split placeholder-bearing text into text, line breaks, and delimiter runs."""
    segments: list[_InlineSegment] = []
    length = len(text)
    pos = 0

    def prev_visible_char() -> str | None:
        if not segments:
            return None
        last = segments[-1]
        if last.kind in ("br", "nl"):
            return "\n"
        if last.kind == "token":
            return "\x00"
        return last.value[-1] if last.value else None

    while pos < length:
        ch = text[pos]
        if ch == "\x00":
            end_tok = text.find("\x00", pos + 1)
            if end_tok != -1:
                tok_key = text[pos : end_tok + 1]
                segments.append(_InlineSegment(kind="token", value=tokens[tok_key]))
                pos = end_tok + 1
                continue

        if ch == "\n":
            # Detect hard break via `  \n` or `\\\n` at end of preceding text segment.
            is_hard = False
            if segments and segments[-1].kind == "text":
                prev_val = segments[-1].value
                if prev_val.endswith("  "):
                    stripped = prev_val.rstrip(" ")
                    if stripped:
                        segments[-1].value = stripped
                    else:
                        segments.pop()
                    is_hard = True
                elif prev_val.endswith("\\"):
                    stripped = prev_val[:-1]
                    if stripped:
                        segments[-1].value = stripped
                    else:
                        segments.pop()
                    is_hard = True
            segments.append(_InlineSegment(kind="br" if is_hard else "nl", value=""))
            pos += 1
            continue

        if ch in ("*", "_", "~"):
            run_end = pos
            while run_end < length and text[run_end] == ch:
                run_end += 1
            run_len = run_end - pos
            prev_ch = prev_visible_char()
            next_ch: str | None
            if run_end >= length:
                next_ch = None
            elif text[run_end] == "\x00":
                next_ch = "\x00"
            else:
                next_ch = text[run_end]
            can_open, can_close = _flanking_flags(prev_ch, next_ch, ch, run_len)
            if ch == "~" and run_len != 2:
                segments.append(_InlineSegment(kind="text", value=text[pos:run_end]))
            else:
                segments.append(
                    _InlineSegment(
                        kind="delim",
                        value=text[pos:run_end],
                        char=ch,
                        can_open=can_open,
                        can_close=can_close,
                        orig_len=run_len,
                    )
                )
            pos = run_end
            continue

        text_start = pos
        while pos < length and text[pos] not in ("\x00", "\n", "*", "_", "~"):
            pos += 1
        chunk = text[text_start:pos]
        if segments and segments[-1].kind == "text":
            segments[-1].value += chunk
        else:
            segments.append(_InlineSegment(kind="text", value=chunk))

    # Trim trailing spaces before soft line breaks.
    for idx, seg in enumerate(segments):
        if seg.kind == "nl" and idx > 0 and segments[idx - 1].kind == "text":
            segments[idx - 1].value = segments[idx - 1].value.rstrip(" \t")

    return segments


def _apply_emphasis_and_escape(text: str, tokens: dict[str, str]) -> str:
    """Resolve emphasis/strikethrough delimiter pairs and escape all raw text."""
    segments = _tokenize_inline_segments(text, tokens)
    if not segments:
        return ""

    open_tags: dict[int, list[str]] = {idx: [] for idx in range(len(segments))}
    close_tags: dict[int, list[str]] = {idx: [] for idx in range(len(segments))}

    delim_indices = [idx for idx, seg in enumerate(segments) if seg.kind == "delim"]
    openers_bottom: dict[tuple[str, int, bool], int] = {}

    curr_pos = 0
    while curr_pos < len(delim_indices):
        c_idx = delim_indices[curr_pos]
        closer = segments[c_idx]
        if not closer.can_close or not closer.value:
            curr_pos += 1
            continue

        # Search backwards for a matching opener.
        o_pos = curr_pos - 1
        found_o_pos: int | None = None
        bottom_key = (closer.char, closer.orig_len % 3, closer.can_open)
        bottom = openers_bottom.get(bottom_key, -1)

        while o_pos > bottom:
            o_idx = delim_indices[o_pos]
            opener = segments[o_idx]
            if opener.can_open and opener.value and opener.char == closer.char:
                # CommonMark rule of 3 for `*` / `_` when one side can both open and close.
                odd_match = (
                    closer.char in ("*", "_")
                    and (opener.can_close or closer.can_open)
                    and ((opener.orig_len + closer.orig_len) % 3 == 0)
                    and not (opener.orig_len % 3 == 0 and closer.orig_len % 3 == 0)
                )
                if not odd_match:
                    if closer.char != "~" or (len(opener.value) == 2 and len(closer.value) == 2):
                        found_o_pos = o_pos
                        break
            o_pos -= 1

        if found_o_pos is None:
            openers_bottom[bottom_key] = curr_pos - 1
            if not closer.can_open:
                closer.active = False
            curr_pos += 1
            continue

        o_idx = delim_indices[found_o_pos]
        opener = segments[o_idx]
        if closer.char == "~":
            use_len = 2
            tag = "del"
        elif len(opener.value) >= 2 and len(closer.value) >= 2:
            use_len = 2
            tag = "strong"
        else:
            use_len = 1
            tag = "em"

        opener.value = opener.value[:-use_len]
        closer.value = closer.value[use_len:]
        # Earlier matches are strictly inner relative to later matches on the same token.
        open_tags[o_idx].insert(0, f"<{tag}>")
        close_tags[c_idx].append(f"</{tag}>")

        # Deactivate intervening delimiter runs between matched opener and closer.
        for mid_pos in range(found_o_pos + 1, curr_pos):
            mid_seg = segments[delim_indices[mid_pos]]
            mid_seg.can_open = False
            mid_seg.can_close = False

        if not closer.value:
            curr_pos += 1

    out: list[str] = []
    for idx, seg in enumerate(segments):
        if close_tags[idx]:
            out.extend(close_tags[idx])
        if seg.kind == "text":
            out.append(html.escape(seg.value, quote=True))
        elif seg.kind == "delim":
            if seg.value:
                out.append(html.escape(seg.value, quote=True))
        elif seg.kind == "token":
            out.append(seg.value)
        elif seg.kind == "br":
            out.append("<br>\n")
        elif seg.kind == "nl":
            out.append("\n")
        if open_tags[idx]:
            out.extend(open_tags[idx])

    return "".join(out)


def _render_inline(raw: str) -> str:
    """Render GFM inline Markdown (code, links, bold, italic, strike, breaks)."""
    text = _strip_nulls(raw)
    if not text:
        return ""

    tokens: dict[str, str] = {}
    token_seq = 0
    has_potential_link = "](" in text

    def add_token(kind: str, rendered_html: str) -> str:
        nonlocal token_seq
        key = f"\x00{kind}{token_seq}\x00"
        token_seq += 1
        tokens[key] = rendered_html
        return key

    parts: list[str] = []
    pos = 0
    length = len(text)

    while pos < length:
        ch = text[pos]

        # 1. Backslash escape
        if ch == "\\" and pos + 1 < length and text[pos + 1] in _ASCII_PUNCTUATION:
            parts.append(add_token("E", html.escape(text[pos + 1], quote=True)))
            pos += 2
            continue

        # 2. Inline code span (highest precedence)
        if ch == "`":
            code_match = _match_code_span(text, pos)
            if code_match is not None:
                end_pos, content = code_match
                code_html = f'<code class="md-inline-code">{html.escape(content, quote=True)}</code>'
                parts.append(add_token("C", code_html))
                pos = end_pos
                continue
            run_end = pos
            while run_end < length and text[run_end] == "`":
                run_end += 1
            parts.append(text[pos:run_end])
            pos = run_end
            continue

        # 3. Links [label](url) (and ![alt](url) rendered safely without <img>)
        is_image = ch == "!" and pos + 1 < length and text[pos + 1] == "["
        if (ch == "[" or is_image) and has_potential_link:
            bracket_pos = pos + 1 if is_image else pos
            link_match = _find_matching_link_end(text, bracket_pos)
            if link_match is not None:
                close_bracket, end_pos, raw_url, raw_title = link_match
                label_raw = text[bracket_pos + 1 : close_bracket]
                label_html = _render_inline(label_raw)
                safe, safe_url = _is_safe_url(raw_url)
                if safe:
                    href = html.escape(safe_url, quote=True)
                    title_attr = f' title="{html.escape(raw_title, quote=True)}"' if raw_title else ""
                    if safe_url.startswith(("http://", "https://")):
                        link_html = (
                            f'<a href="{href}"{title_attr} target="_blank" '
                            f'rel="noopener noreferrer">{label_html}</a>'
                        )
                    else:
                        link_html = f'<a href="{href}"{title_attr}>{label_html}</a>'
                    parts.append(add_token("L", link_html))
                else:
                    # Reject unsafe schemes (`javascript:`, `data:`, `vbscript:`, `file:`, etc.)
                    # by emitting escaped label text without any `<a>` or `<img>` element.
                    parts.append(add_token("L", label_html))
                pos = end_pos
                continue

        parts.append(ch)
        pos += 1

    return _apply_emphasis_and_escape("".join(parts), tokens)


def _parse_atx_heading(line: str) -> tuple[int, str] | None:
    """Parse an ATX heading line into ``(level, content)`` or ``None``."""
    match = _ATX_HEADING_RE.match(line)
    if not match:
        return None
    level = len(match.group(1))
    content = (match.group(2) or "").strip()
    # Strip optional closing hash sequence preceded by whitespace.
    if content:
        stripped_end = content.rstrip("#")
        if not stripped_end:
            content = ""
        elif stripped_end[-1] in (" ", "\t"):
            content = stripped_end.rstrip(" \t")
    return level, content


def _parse_fence_open(line: str) -> tuple[str, int, int, str] | None:
    """Parse an opening code fence into ``(fence_char, fence_len, indent, lang)``."""
    match = _FENCE_OPEN_RE.match(line)
    if not match:
        return None
    indent = len(match.group(1))
    fence = match.group(2)
    info = match.group(3).strip()
    fence_char = fence[0]
    if fence_char == "`" and "`" in info:
        return None
    lang = info.split()[0] if info else ""
    return fence_char, len(fence), indent, lang


def _is_fence_close(line: str, fence_char: str, min_len: int) -> bool:
    """Return True if ``line`` closes a fenced code block."""
    stripped = line.lstrip(" ")
    leading = len(line) - len(stripped)
    if leading > 3:
        return False
    core = stripped.rstrip(" \t")
    return len(core) >= min_len and set(core) == {fence_char}


def _split_table_row(line: str) -> list[str]:
    """Split a GFM pipe table row, preserving escaped pipes and code-span pipes."""
    stripped = line.strip()
    cells: list[str] = []
    buf: list[str] = []
    pos = 0
    length = len(stripped)
    in_ticks = 0

    while pos < length:
        ch = stripped[pos]
        if ch == "`":
            tick_end = pos
            while tick_end < length and stripped[tick_end] == "`":
                tick_end += 1
            run_len = tick_end - pos
            if in_ticks == 0:
                in_ticks = run_len
            elif in_ticks == run_len:
                in_ticks = 0
            buf.append(stripped[pos:tick_end])
            pos = tick_end
            continue

        if ch == "\\" and pos + 1 < length:
            nxt = stripped[pos + 1]
            if nxt == "|":
                buf.append("|")
                pos += 2
                continue
            if nxt == "\\":
                buf.append("\\\\")
                pos += 2
                continue

        if ch == "|" and in_ticks == 0:
            cells.append("".join(buf))
            buf.clear()
            pos += 1
            continue

        buf.append(ch)
        pos += 1

    cells.append("".join(buf))

    if cells and not cells[0].strip():
        cells.pop(0)
    if cells and not cells[-1].strip():
        cells.pop()

    return [cell.strip() for cell in cells]


def _parse_table_alignments(line: str, expected_cols: int) -> list[str] | None:
    """Parse a GFM table delimiter row; return alignments or ``None`` if invalid."""
    if "|" not in line:
        return None
    cells = _split_table_row(line)
    if len(cells) != expected_cols or expected_cols == 0:
        return None
    alignments: list[str] = []
    for cell in cells:
        token = cell.strip()
        if not _TABLE_ALIGN_CELL_RE.match(token):
            return None
        left = token.startswith(":")
        right = token.endswith(":")
        if left and right:
            alignments.append("center")
        elif left:
            alignments.append("left")
        elif right:
            alignments.append("right")
        else:
            alignments.append("")
    return alignments


def _parse_list_marker(line: str) -> tuple[int, str, str, int | None, str] | None:
    """Parse an unordered or ordered list marker line."""
    if _HR_RE.match(line):
        return None
    ul_match = _UL_ITEM_RE.match(line)
    if ul_match:
        indent = len(ul_match.group(1))
        delim = ul_match.group(2)
        rest = ul_match.group(3) or ""
        return indent, "ul", delim, None, rest
    ol_match = _OL_ITEM_RE.match(line)
    if ol_match:
        indent = len(ol_match.group(1))
        start_num = int(ol_match.group(2))
        delim = ol_match.group(3)
        rest = ol_match.group(4) or ""
        return indent, "ol", delim, start_num, rest
    return None


def _is_block_interrupt(lines: list[str], idx: int) -> bool:
    """Return True if ``lines[idx]`` starts a block that interrupts a paragraph."""
    line = lines[idx]
    if not line.strip():
        return True
    if _ATX_HEADING_RE.match(line):
        return True
    if _parse_fence_open(line) is not None:
        return True
    if _HR_RE.match(line):
        return True
    if _BLOCKQUOTE_RE.match(line):
        return True
    list_marker = _parse_list_marker(line)
    if list_marker is not None:
        _, m_kind, _, start_num, rest = list_marker
        if rest.strip() and (m_kind == "ul" or start_num == 1):
            return True
    if idx + 1 < len(lines) and "|" in line:
        header_cells = _split_table_row(line)
        if header_cells and _parse_table_alignments(lines[idx + 1], len(header_cells)) is not None:
            return True
    return False


def _render_table(header_cells: list[str], alignments: list[str], body_rows: list[list[str]]) -> str:
    """Render a GFM pipe table wrapped in ``<div class="md-table-wrap">``."""
    num_cols = len(header_cells)

    def align_attr(col_idx: int) -> str:
        align = alignments[col_idx] if col_idx < len(alignments) else ""
        return f' style="text-align:{align}"' if align else ""

    head_cells_html = "".join(
        f"<th{align_attr(col)}>{_render_inline(cell)}</th>" for col, cell in enumerate(header_cells)
    )
    parts = [
        '<div class="md-table-wrap"><table class="md-table">',
        f"<thead><tr>{head_cells_html}</tr></thead>",
    ]
    if body_rows:
        parts.append("<tbody>")
        for row in body_rows:
            padded = row[:num_cols] + [""] * max(0, num_cols - len(row))
            row_html = "".join(
                f"<td{align_attr(col)}>{_render_inline(cell)}</td>" for col, cell in enumerate(padded)
            )
            parts.append(f"<tr>{row_html}</tr>")
        parts.append("</tbody>")
    parts.append("</table></div>")
    return "\n".join(parts)


def _render_blockquote(quote_lines: list[str]) -> str:
    """Render a blockquote or GitHub Alert (``> [!NOTE]`` .. ``> [!CAUTION]``)."""
    if quote_lines:
        alert_match = _ALERT_MARKER_RE.match(quote_lines[0].strip())
        if alert_match:
            kind = alert_match.group(1).lower()
            title = kind.capitalize()
            inline_rest = (alert_match.group(2) or "").strip()
            body_lines = ([inline_rest] if inline_rest else []) + quote_lines[1:]
            body_html = _render_blocks(body_lines)
            if body_html:
                return (
                    f'<div class="md-alert md-alert-{kind}">'
                    f'<div class="md-alert-title">{title}</div>'
                    f'<div class="md-alert-body">\n{body_html}\n</div>'
                    f"</div>"
                )
            return (
                f'<div class="md-alert md-alert-{kind}">'
                f'<div class="md-alert-title">{title}</div>'
                f"</div>"
            )

    inner = _render_blocks(quote_lines)
    if inner:
        return f'<blockquote class="md-quote">\n{inner}\n</blockquote>'
    return '<blockquote class="md-quote"></blockquote>'


def _strip_indent(line: str, amount: int) -> str:
    """Strip up to ``amount`` leading spaces (expanding tabs to 4 spaces)."""
    expanded = line.expandtabs(4)
    count = 0
    while count < amount and count < len(expanded) and expanded[count] == " ":
        count += 1
    return expanded[count:]


def _render_list(lines: list[str], start_idx: int) -> tuple[str, int]:
    """Parse and render an unordered/ordered/task list starting at ``start_idx``."""
    first_marker = _parse_list_marker(lines[start_idx])
    assert first_marker is not None
    base_indent, list_kind, list_delim, start_num, _ = first_marker

    items_blocks: list[list[str]] = []
    idx = start_idx
    total = len(lines)
    is_loose = False

    while idx < total:
        marker = _parse_list_marker(lines[idx])
        if marker is None:
            break
        m_indent, m_kind, m_delim, _, first_rest = marker
        if m_indent != base_indent or m_kind != list_kind or m_delim != list_delim:
            break

        expanded_head = lines[idx].expandtabs(4)
        marker_prefix_len = len(expanded_head) - len(first_rest) if first_rest else len(expanded_head) + 1
        child_indent = max(base_indent + 2, min(marker_prefix_len, base_indent + 4))

        item_lines = [first_rest]
        idx += 1
        saw_blank = False

        while idx < total:
            cur = lines[idx]
            if not cur.strip():
                # Look ahead past blank lines to see if item continues or sibling follows.
                look = idx + 1
                while look < total and not lines[look].strip():
                    look += 1
                if look >= total:
                    idx = look
                    break
                nxt = lines[look]
                nxt_expanded = nxt.expandtabs(4)
                nxt_indent = len(nxt_expanded) - len(nxt_expanded.lstrip(" "))
                nxt_marker = _parse_list_marker(nxt)
                if nxt_marker is not None and nxt_marker[0] == base_indent:
                    if nxt_marker[1] == list_kind and nxt_marker[2] == list_delim:
                        is_loose = True
                    idx = look
                    break
                if nxt_indent >= child_indent or (nxt_marker is not None and nxt_indent > base_indent):
                    is_loose = True
                    item_lines.append("")
                    saw_blank = True
                    idx += 1
                    continue
                idx = look
                break

            cur_expanded = cur.expandtabs(4)
            cur_indent = len(cur_expanded) - len(cur_expanded.lstrip(" "))
            if cur_indent >= child_indent or cur_indent > base_indent:
                stripped = _strip_indent(cur, min(cur_indent, child_indent))
                item_lines.append(stripped)
                saw_blank = False
                idx += 1
                continue

            # Check if a sibling or interrupting block ends this item.
            if _is_block_interrupt(lines, idx) or saw_blank:
                break

            # Lazy paragraph continuation line inside the current list item.
            item_lines.append(cur.strip())
            idx += 1

        items_blocks.append(item_lines)

    has_task_item = False
    rendered_items: list[str] = []

    for item_lines in items_blocks:
        task_prefix = ""
        first_line = item_lines[0] if item_lines else ""
        task_match = _TASK_CHECKBOX_RE.match(first_line)
        if task_match:
            has_task_item = True
            checked = task_match.group(1).lower() == "x"
            checked_attr = " checked" if checked else ""
            task_prefix = f'<input type="checkbox"{checked_attr} disabled> '
            item_lines = [task_match.group(2) or "", *item_lines[1:]]

        # Render inner blocks cleanly; keep tight single-paragraph items compact.
        rendered_inner = _render_blocks(item_lines)
        if task_prefix:
            if rendered_inner.startswith("<p>"):
                rendered_inner = "<p>" + task_prefix + rendered_inner[3:]
            else:
                rendered_inner = task_prefix + rendered_inner
            li_open = '<li class="md-task-item">'
        else:
            li_open = "<li>"

        if not is_loose and rendered_inner.startswith("<p>"):
            close_p = rendered_inner.find("</p>")
            if close_p != -1:
                first_p = rendered_inner[3:close_p]
                rest_blocks = rendered_inner[close_p + 4 :].lstrip("\n")
                if rest_blocks:
                    rendered_inner = f"{first_p}\n{rest_blocks}"
                else:
                    rendered_inner = first_p

        rendered_items.append(f"{li_open}{rendered_inner}</li>")

    if list_kind == "ul":
        cls = "md-list md-task-list" if has_task_item else "md-list"
        return f'<ul class="{cls}">\n' + "\n".join(rendered_items) + "\n</ul>", idx

    start_attr = f' start="{start_num}"' if start_num is not None and start_num != 1 else ""
    cls = "md-list md-task-list" if has_task_item else "md-list"
    return f'<ol class="{cls}"{start_attr}>\n' + "\n".join(rendered_items) + "\n</ol>", idx


def _render_blocks(lines: list[str]) -> str:
    """Parse and render a sequence of lines into GFM HTML blocks."""
    out: list[str] = []
    idx = 0
    total = len(lines)

    while idx < total:
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue

        # 1. Fenced code block
        fence = _parse_fence_open(line)
        if fence is not None:
            fence_char, fence_len, fence_indent, lang = fence
            idx += 1
            code_lines: list[str] = []
            while idx < total:
                if _is_fence_close(lines[idx], fence_char, fence_len):
                    idx += 1
                    break
                code_lines.append(_strip_indent(lines[idx], fence_indent))
                idx += 1
            escaped_code = html.escape("\n".join(code_lines), quote=True)
            if lang:
                escaped_lang = html.escape(lang, quote=True)
                header = f'<div class="md-code-header"><span class="md-code-lang">{escaped_lang}</span></div>'
            else:
                header = ""
            out.append(f'<div class="md-code-block">{header}<pre><code>{escaped_code}</code></pre></div>')
            continue

        # 2. ATX Heading
        heading = _parse_atx_heading(line)
        if heading is not None:
            level, content = heading
            out.append(f'<h{level} class="md-h{level}">{_render_inline(content)}</h{level}>')
            idx += 1
            continue

        # 3. Horizontal rule (checked before unordered list so `---` / `***` are HRs)
        if _HR_RE.match(line):
            out.append('<hr class="md-hr">')
            idx += 1
            continue

        # 4. Blockquote / GitHub Alert
        bq_match = _BLOCKQUOTE_RE.match(line)
        if bq_match:
            quote_lines = [bq_match.group(1)]
            idx += 1
            while idx < total:
                next_bq = _BLOCKQUOTE_RE.match(lines[idx])
                if next_bq:
                    quote_lines.append(next_bq.group(1))
                    idx += 1
                elif lines[idx].strip() and not _is_block_interrupt(lines, idx) and quote_lines[-1].strip():
                    # Lazy continuation inside blockquote.
                    quote_lines.append(lines[idx].strip())
                    idx += 1
                else:
                    break
            out.append(_render_blockquote(quote_lines))
            continue

        # 5. GFM Pipe Table
        if idx + 1 < total and "|" in line:
            header_cells = _split_table_row(line)
            alignments = _parse_table_alignments(lines[idx + 1], len(header_cells)) if header_cells else None
            if header_cells and alignments is not None:
                idx += 2
                body_rows: list[list[str]] = []
                while idx < total:
                    row_line = lines[idx]
                    if not row_line.strip() or "|" not in row_line:
                        break
                    if (
                        _ATX_HEADING_RE.match(row_line)
                        or _parse_fence_open(row_line) is not None
                        or _BLOCKQUOTE_RE.match(row_line)
                    ):
                        break
                    body_rows.append(_split_table_row(row_line))
                    idx += 1
                out.append(_render_table(header_cells, alignments, body_rows))
                continue

        # 6. Lists (unordered, ordered, task)
        if _parse_list_marker(line) is not None:
            list_html, idx = _render_list(lines, idx)
            out.append(list_html)
            continue

        # 7. Paragraph
        para_lines = [line.lstrip(" ")]
        idx += 1
        while idx < total:
            if _is_block_interrupt(lines, idx):
                break
            para_lines.append(lines[idx].lstrip(" "))
            idx += 1
        para_text = "\n".join(para_lines).strip()
        if para_text:
            out.append(f"<p>{_render_inline(para_text)}</p>")

    return "\n".join(out)


def render_markdown_to_html(text: str) -> str:
    """Convert GFM Markdown into XSS-safe static HTML.

    Strictly escapes all untrusted raw HTML tags and attributes, rejects unsafe
    link schemes (``javascript:``, ``data:``, ``vbscript:``, ``file:``, etc.),
    and renders headings, fenced code blocks, pipe tables, blockquotes/alerts,
    nested/task lists, horizontal rules, paragraphs, and inline formatting.
    """
    if not text:
        return ""
    normalized = _strip_nulls(str(text)).replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return ""
    return _render_blocks(normalized.split("\n"))


def looks_like_markdown(text: str) -> bool:
    """Return True when ``text`` meaningfully uses Markdown structure or formatting."""
    if not text:
        return False
    normalized = _strip_nulls(str(text)).replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return False

    if (
        _HEADING_DETECT_RE.search(normalized)
        or _FENCE_DETECT_RE.search(normalized)
        or _TABLE_DELIM_DETECT_RE.search(normalized)
        or _SINGLE_COL_TABLE_DETECT_RE.search(normalized)
        or _UL_DETECT_RE.search(normalized)
        or _OL_DETECT_RE.search(normalized)
        or _ALERT_DETECT_RE.search(normalized)
        or _INLINE_CODE_DETECT_RE.search(normalized)
        or _BOLD_STAR_DETECT_RE.search(normalized)
        or _BOLD_UNDER_DETECT_RE.search(normalized)
        or _STRIKE_DETECT_RE.search(normalized)
        or _LINK_DETECT_RE.search(normalized)
    ):
        return True

    # Detect blockquotes only when clearly formatted (`> text` where text isn't a bare comparison).
    for line in normalized.split("\n"):
        bq = _BLOCKQUOTE_RE.match(line)
        if bq and bq.group(1).strip() and not bq.group(1).lstrip()[0].isdigit():
            return True

    return False
