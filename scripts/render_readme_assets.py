#!/usr/bin/env python3
"""Render GhostAP README showcase PNGs from the real ``ux/*.html`` card previews.

This script uses only the Python standard library and headless Google Chrome /
Chromium to wrap the existing HTML previews under ``ux/`` into framed
presentation layouts and capture high-DPI PNG assets under ``docs/assets/``.

Key implementation details:
- Standard library only (no Pillow or Playwright dependency).
- Inline vector SVG replacement for Unicode emojis inside rendered previews so
  headless Linux hosts without ``Noto Color Emoji`` never render tofu boxes.
- Lossless pure-stdlib PNG height cropping and level-9 deflate recompression so
  Chrome's outer window chrome padding is removed cleanly and PNG sizes stay
  compact.
"""

from __future__ import annotations

import argparse
import html
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_UX_DIR = ROOT_DIR / "ux"
DEFAULT_ASSETS_DIR = ROOT_DIR / "docs" / "assets"

# Chrome's non-old headless mode reserves ~87px of outer window chrome height
# from --window-size; adding extra height and cropping lossless PNG rows ensures
# the full CSS viewport renders without bottom truncation or scrollbars.
CHROME_HEIGHT_PAD_PX = 180
DEFAULT_TARGET_MAX_BYTES = 600 * 1024

# Crisp inline vector SVG icons for every emoji found across ux/*.html so
# Linux containers without color emoji fonts render sharp icons instead of tofu.
_EMOJI_SVG_PATHS: dict[str, tuple[str, str]] = {
    "🪐": ("#38bdf8", '<circle cx="10" cy="10" r="4.2" fill="#38bdf8"/><ellipse cx="10" cy="10" rx="8.5" ry="3.2" transform="rotate(-24 10 10)" fill="none" stroke="#bae6fd" stroke-width="1.5"/>'),
    "🔋": ("#34d399", '<rect x="2.5" y="5.5" width="13" height="9" rx="2" fill="none" stroke="#34d399" stroke-width="1.7"/><rect x="4.5" y="7.5" width="8.5" height="5" rx="1" fill="#34d399"/><rect x="16" y="8" width="2" height="4" rx="0.8" fill="#34d399"/>'),
    "💎": ("#60a5fa", '<polygon points="10,2.5 16.5,7.5 10,17.5 3.5,7.5" fill="#38bdf8" stroke="#bae6fd" stroke-width="1.2"/><line x1="3.5" y1="7.5" x2="16.5" y2="7.5" stroke="#e0f2fe" stroke-width="1.1"/>'),
    "📊": ("#60a5fa", '<rect x="3" y="10" width="3.2" height="6.5" rx="0.8" fill="#60a5fa"/><rect x="8.4" y="6" width="3.2" height="10.5" rx="0.8" fill="#34d399"/><rect x="13.8" y="3.5" width="3.2" height="13" rx="0.8" fill="#fbbf24"/>'),
    "🛠️": ("#94a3b8", '<path d="M13.5 3.5a3.6 3.6 0 0 0-4.7 4.7L4 13l3 3 4.8-4.8a3.6 3.6 0 0 0 4.7-4.7l-2.1 2.1-2.1-.6-.6-2.1 1.8-2.4z" fill="#cbd5e1"/>'),
    "🛠": ("#94a3b8", '<path d="M13.5 3.5a3.6 3.6 0 0 0-4.7 4.7L4 13l3 3 4.8-4.8a3.6 3.6 0 0 0 4.7-4.7l-2.1 2.1-2.1-.6-.6-2.1 1.8-2.4z" fill="#cbd5e1"/>'),
    "🗜️": ("#a78bfa", '<path d="M5 4h10v2.5H5zM5 13.5h10V16H5zM10 6.5v7M7.5 9L10 6.5 12.5 9M7.5 11L10 13.5 12.5 11" fill="none" stroke="#c4b5fd" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'),
    "🗜": ("#a78bfa", '<path d="M5 4h10v2.5H5zM5 13.5h10V16H5zM10 6.5v7M7.5 9L10 6.5 12.5 9M7.5 11L10 13.5 12.5 11" fill="none" stroke="#c4b5fd" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'),
    "🔄": ("#38bdf8", '<path d="M15.5 8.5A5.8 5.8 0 0 0 5.4 6.1M4.5 11.5a5.8 5.8 0 0 0 10.1 2.4" fill="none" stroke="#60a5fa" stroke-width="1.8" stroke-linecap="round"/><polygon points="16.5,5.5 16.5,9.5 12.7,9" fill="#60a5fa"/><polygon points="3.5,14.5 3.5,10.5 7.3,11" fill="#60a5fa"/>'),
    "🧹": ("#fbbf24", '<path d="M12.5 3.5l4 4-6.5 6.5-4-4z" fill="#94a3b8"/><path d="M6 10l-3 5.5 5.5-3z" fill="#fbbf24"/>'),
    "🧭": ("#38bdf8", '<circle cx="10" cy="10" r="7.2" fill="none" stroke="#93c5fd" stroke-width="1.6"/><polygon points="13.5,6.5 11.2,11.2 6.5,13.5 8.8,8.8" fill="#f87171"/>'),
    "📂": ("#fbbf24", '<path d="M2.5 5.5A1.5 1.5 0 0 1 4 4h3.5l1.5 1.8H16a1.5 1.5 0 0 1 1.5 1.5v7A1.5 1.5 0 0 1 16 15.8H4A1.5 1.5 0 0 1 2.5 14.3z" fill="#fbbf24"/>'),
    "💻": ("#60a5fa", '<rect x="3" y="4" width="14" height="9.5" rx="1.5" fill="#1e293b" stroke="#93c5fd" stroke-width="1.5"/><path d="M1.5 15.5h17" stroke="#cbd5e1" stroke-width="1.8" stroke-linecap="round"/>'),
    "⚙️": ("#94a3b8", '<circle cx="10" cy="10" r="3.2" fill="none" stroke="#cbd5e1" stroke-width="1.8"/><circle cx="10" cy="10" r="6.2" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="2.4 2.5"/>'),
    "⚙": ("#94a3b8", '<circle cx="10" cy="10" r="3.2" fill="none" stroke="#cbd5e1" stroke-width="1.8"/><circle cx="10" cy="10" r="6.2" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="2.4 2.5"/>'),
    "⚡": ("#fbbf24", '<polygon points="11.5,2.5 5,11 9.8,11 8.5,17.5 15,9 10.2,9" fill="#fbbf24"/>'),
    "🚀": ("#ffffff", '<path d="M10 2.5c3 2.5 4.2 6 4.2 9.5l-4.2 2.2L5.8 12C5.8 8.5 7 5 10 2.5z" fill="#f8fafc"/><circle cx="10" cy="8.5" r="1.6" fill="#2563eb"/><path d="M8 14.2l2 3.3 2-3.3z" fill="#fbbf24"/>'),
    "💾": ("#60a5fa", '<rect x="3.2" y="3.2" width="13.6" height="13.6" rx="1.8" fill="#3b82f6"/><rect x="5.8" y="3.2" width="7.2" height="4.8" fill="#e2e8f0"/><rect x="5.8" y="10.8" width="8.4" height="4.5" rx="0.8" fill="#dbeafe"/>'),
    "⏱️": ("#3b82f6", '<circle cx="10" cy="10.8" r="6.2" fill="none" stroke="#2563eb" stroke-width="1.7"/><path d="M8 2.8h4M10 10.8l2.8-2.5" stroke="#2563eb" stroke-width="1.7" stroke-linecap="round"/>'),
    "⏱": ("#3b82f6", '<circle cx="10" cy="10.8" r="6.2" fill="none" stroke="#2563eb" stroke-width="1.7"/><path d="M8 2.8h4M10 10.8l2.8-2.5" stroke="#2563eb" stroke-width="1.7" stroke-linecap="round"/>'),
    "⏹️": ("#ef4444", '<rect x="4" y="4" width="12" height="12" rx="2" fill="#ef4444"/>'),
    "⏹": ("#ef4444", '<rect x="4" y="4" width="12" height="12" rx="2" fill="#ef4444"/>'),
    "📎": ("#94a3b8", '<path d="M12.5 5.5l-5.8 5.8a2.3 2.3 0 1 0 3.2 3.2l6-6a3.6 3.6 0 0 0-5.1-5.1l-6.3 6.3a4.9 4.9 0 0 0 6.9 6.9l5.3-5.3" fill="none" stroke="#93c5fd" stroke-width="1.6" stroke-linecap="round"/>'),
    "✅": ("#22c55e", '<rect x="2.5" y="2.5" width="15" height="15" rx="3.5" fill="#16a34a"/><path d="M6.2 10.2l2.6 2.6 5-5.3" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'),
    "❌": ("#ef4444", '<circle cx="10" cy="10" r="7.5" fill="#dc2626"/><path d="M7 7l6 6M13 7l-6 6" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>'),
    "🟢": ("#22c55e", '<circle cx="10" cy="10" r="6.5" fill="#22c55e" stroke="#86efac" stroke-width="1.5"/>'),
    "⚪": ("#cbd5e1", '<circle cx="10" cy="10" r="6" fill="#cbd5e1" stroke="#94a3b8" stroke-width="1.5"/>'),
    "ℹ️": ("#3b82f6", '<rect x="2.5" y="2.5" width="15" height="15" rx="3.5" fill="#2563eb"/><circle cx="10" cy="6.4" r="1.1" fill="#ffffff"/><rect x="9.1" y="8.8" width="1.8" height="5.5" rx="0.9" fill="#ffffff"/>'),
    "ℹ": ("#3b82f6", '<rect x="2.5" y="2.5" width="15" height="15" rx="3.5" fill="#2563eb"/><circle cx="10" cy="6.4" r="1.1" fill="#ffffff"/><rect x="9.1" y="8.8" width="1.8" height="5.5" rx="0.9" fill="#ffffff"/>'),
    "💭": ("#93c5fd", '<path d="M4.5 9.5c0-3 2.5-5.2 5.7-5.2 3.3 0 5.8 2.1 5.8 5 0 2.8-2.4 4.9-5.6 4.9-.8 0-1.6-.1-2.3-.4L5.2 15.5l.6-2.4c-.8-.9-1.3-2.2-1.3-3.6z" fill="#e0f2fe"/>'),
    "📝": ("#60a5fa", '<rect x="4" y="3" width="11" height="14" rx="1.8" fill="#e2e8f0"/><path d="M6.5 7h6M6.5 10h6M6.5 13h4" stroke="#475569" stroke-width="1.4" stroke-linecap="round"/>'),
    "🛡️": ("#38bdf8", '<path d="M10 2.5l6 2.4v4.8c0 3.7-2.5 6.7-6 7.8-3.5-1.1-6-4.1-6-7.8V4.9z" fill="#38bdf8" stroke="#bae6fd" stroke-width="1.2"/>'),
    "🛡": ("#38bdf8", '<path d="M10 2.5l6 2.4v4.8c0 3.7-2.5 6.7-6 7.8-3.5-1.1-6-4.1-6-7.8V4.9z" fill="#38bdf8" stroke="#bae6fd" stroke-width="1.2"/>'),
    "🔧": ("#94a3b8", '<path d="M13.5 3.5a3.6 3.6 0 0 0-4.7 4.7L4 13l3 3 4.8-4.8a3.6 3.6 0 0 0 4.7-4.7l-2.1 2.1-2.1-.6-.6-2.1 1.8-2.4z" fill="#cbd5e1"/>'),
    "📋": ("#93c5fd", '<rect x="4" y="3.5" width="12" height="13.5" rx="1.8" fill="#cbd5e1"/><rect x="7" y="2.2" width="6" height="2.4" rx="1" fill="#64748b"/><path d="M6.5 8h7M6.5 11h7M6.5 14h4.5" stroke="#334155" stroke-width="1.4" stroke-linecap="round"/>'),
    "🧩": ("#34d399", '<path d="M4 5.5h3.5a1.6 1.6 0 1 1 3 0H14v3.5a1.6 1.6 0 1 1 0 3V15.5H4z" fill="#34d399"/>'),
    "🔮": ("#a78bfa", '<circle cx="10" cy="9" r="5.5" fill="#8b5cf6"/><rect x="6.5" y="14.5" width="7" height="2.2" rx="1" fill="#94a3b8"/>'),
}

_EMOJI_KEYS = sorted(_EMOJI_SVG_PATHS.keys(), key=len, reverse=True)
_EMOJI_RE = re.compile("|".join(re.escape(k) for k in _EMOJI_KEYS) + r"\ufe0f?")
_TOKEN_RE = re.compile(
    r"(<script\b[^>]*>.*?</script>|<style\b[^>]*>.*?</style>|<option\b[^>]*>.*?</option>|<[^>]+>)",
    re.DOTALL | re.IGNORECASE,
)


def _emoji_to_inline_svg(match: re.Match[str]) -> str:
    """Convert a matched emoji sequence into a crisp inline SVG span."""
    token = match.group(0).rstrip("\ufe0f")
    _, body = _EMOJI_SVG_PATHS.get(match.group(0)) or _EMOJI_SVG_PATHS.get(token, ("#94a3b8", ""))
    return (
        '<svg class="ghostap-emoji" viewBox="0 0 20 20" width="1.05em" height="1.05em" '
        'style="display:inline-block;vertical-align:-0.18em;margin-right:1px;flex-shrink:0;" '
        f'aria-hidden="true">{body}</svg>'
    )


def sanitize_ux_html(raw_html: str, override_css: str = "") -> str:
    """Replace emojis with inline SVGs in text nodes and inject showcase CSS.

    ``<script>``, ``<style>``, and HTML tags stay intact so interactive preview
    scripts (such as ``workflow-team-config.html``) run unchanged, while
    ``<option>`` labels strip missing-glyph emojis cleanly.
    """
    parts = _TOKEN_RE.split(raw_html)
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        lower = part.lower()
        if lower.startswith("<option"):
            out.append(_EMOJI_RE.sub("", part).replace("> ", ">"))
        elif part.startswith("<"):
            out.append(part)
        else:
            out.append(_EMOJI_RE.sub(_emoji_to_inline_svg, part))
    rendered = "".join(out)
    if override_css:
        injection = f"<style id=\"showcase-override\">\n{override_css}\n</style>\n</head>"
        rendered = rendered.replace("</head>", injection, 1)
    return rendered


@dataclass(frozen=True)
class ShowcaseSpec:
    """Configuration for a single README showcase PNG render."""

    name: str
    output_filename: str
    width: int
    height: int
    scale: float
    title_en: str
    title_zh: str
    badge: str
    pills: tuple[str, ...]
    body_html: str
    extra_css: str = ""


def find_browser_binary(explicit: str | None = None) -> str:
    """Locate a working Google Chrome or Chromium binary on Linux or macOS."""
    if explicit:
        if Path(explicit).exists() or shutil.which(explicit):
            return explicit
        raise FileNotFoundError(f"Requested browser binary not found: {explicit}")

    candidates = [
        "google-chrome",
        "chromium",
        "chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for candidate in candidates:
        resolved = shutil.which(candidate) if not candidate.startswith("/") else candidate
        if resolved and Path(resolved).exists():
            return resolved
    raise FileNotFoundError("Could not locate google-chrome or chromium binary")


def crop_and_optimize_png(png_path: Path, target_height: int) -> tuple[int, int, int]:
    """Losslessly crop bottom Chrome padding rows and recompress PNG at level 9.

    Non-interlaced PNG row filters (None/Sub/Up/Average/Paeth) depend only on
    current and prior scanlines, so keeping the top ``target_height`` scanlines
    produces a valid, pixel-exact PNG without unfiltering.
    """
    data = png_path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG signature: {png_path}")

    pos = 8
    chunks: list[tuple[bytes, bytes]] = []
    idat = bytearray()
    stride = 0
    width = 0
    kept_height = target_height

    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        cdata = data[pos + 8 : pos + 8 + length]
        pos += 12 + length

        if ctype == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(">IIBBBBB", cdata)
            if interlace != 0 or bit_depth != 8 or color_type not in (2, 6):
                raise ValueError(f"Unsupported PNG format in {png_path}: {(bit_depth, color_type, interlace)}")
            bpp = 3 if color_type == 2 else 4
            stride = 1 + width * bpp
            kept_height = min(height, target_height)
            ihdr = struct.pack(">IIBBBBB", width, kept_height, bit_depth, color_type, comp, filt, interlace)
            chunks.append((b"IHDR", ihdr))
        elif ctype == b"IDAT":
            idat.extend(cdata)
        elif ctype == b"IEND":
            raw = zlib.decompress(bytes(idat))
            keep_bytes = min(len(raw), kept_height * stride)
            compressed = zlib.compress(raw[:keep_bytes], level=9)
            chunks.append((b"IDAT", compressed))
            chunks.append((b"IEND", b""))
        else:
            chunks.append((ctype, cdata))

    out = bytearray(b"\x89PNG\r\n\x1a\n")
    for ctype, cdata in chunks:
        out.extend(struct.pack(">I", len(cdata)))
        out.extend(ctype)
        out.extend(cdata)
        out.extend(struct.pack(">I", zlib.crc32(ctype + cdata) & 0xFFFFFFFF))
    final_bytes = bytes(out)
    png_path.write_bytes(final_bytes)
    return width, kept_height, len(final_bytes)


def _frame_shell(spec: ShowcaseSpec, inner_body: str, extra_css: str = "") -> str:
    """Wrap showcase panels in a unified high-contrast framing shell."""
    pills_html = "".join(f'<span class="meta-pill">{html.escape(p)}</span>' for p in spec.pills)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(spec.title_en)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      padding: 0;
      width: {spec.width}px;
      height: {spec.height}px;
      overflow: hidden;
      font-family: 'Inter', 'SF Pro Display', 'Noto Sans CJK SC', 'PingFang SC', 'Segoe UI', system-ui, sans-serif;
      color: #f8fafc;
      background:
        radial-gradient(circle at 12% 14%, rgba(56, 189, 248, 0.16), transparent 42%),
        radial-gradient(circle at 88% 82%, rgba(99, 102, 241, 0.18), transparent 44%),
        linear-gradient(135deg, #090e1c 0%, #0f172a 55%, #162347 100%);
    }}
    .backdrop-grid {{
      position: relative;
      width: 100%;
      height: 100%;
      padding: 24px 30px 26px;
      background-image:
        linear-gradient(rgba(148, 163, 184, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.05) 1px, transparent 1px);
      background-size: 36px 36px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .showcase-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }}
    .head-left {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }}
    .eyebrow-badge {{
      padding: 5px 11px;
      border-radius: 999px;
      font: 700 11px/1 'JetBrains Mono', 'SFMono-Regular', monospace;
      letter-spacing: 0.06em;
      color: #bae6fd;
      background: rgba(14, 165, 233, 0.16);
      border: 1px solid rgba(56, 189, 248, 0.4);
      white-space: nowrap;
    }}
    .head-titles h1 {{
      margin: 0;
      font-size: 19px;
      font-weight: 800;
      letter-spacing: -0.015em;
      color: #f8fafc;
      line-height: 1.25;
    }}
    .head-titles p {{
      margin: 2px 0 0;
      font-size: 12.5px;
      color: #94a3b8;
      line-height: 1.3;
    }}
    .head-pills {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: nowrap;
    }}
    .meta-pill {{
      padding: 5px 11px;
      border-radius: 999px;
      font-size: 11.5px;
      font-weight: 600;
      color: #cbd5e1;
      background: rgba(15, 23, 42, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.24);
      white-space: nowrap;
    }}
    .stage {{
      flex: 1;
      min-height: 0;
      display: grid;
      gap: 20px;
    }}
    .window-frame {{
      display: flex;
      flex-direction: column;
      min-height: 0;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(148, 163, 184, 0.24);
      background: #0f172a;
      box-shadow: 0 22px 50px rgba(2, 6, 23, 0.55), 0 0 0 1px rgba(255, 255, 255, 0.04) inset;
    }}
    .window-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 14px;
      background: #131c31;
      border-bottom: 1px solid rgba(148, 163, 184, 0.16);
      flex: 0 0 auto;
    }}
    .dots {{
      display: flex;
      gap: 6px;
      align-items: center;
    }}
    .dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
    }}
    .dot.r {{ background: #ef4444; opacity: 0.82; }}
    .dot.y {{ background: #f59e0b; opacity: 0.82; }}
    .dot.g {{ background: #10b981; opacity: 0.82; }}
    .window-label {{
      font-size: 11.5px;
      font-weight: 600;
      color: #cbd5e1;
      white-space: nowrap;
    }}
    .window-tag {{
      padding: 2px 8px;
      border-radius: 999px;
      font: 600 10.5px/1.3 'JetBrains Mono', 'SFMono-Regular', monospace;
      color: #93c5fd;
      background: rgba(37, 99, 235, 0.18);
      border: 1px solid rgba(96, 165, 250, 0.28);
      white-space: nowrap;
    }}
    iframe.card-frame {{
      width: 100%;
      flex: 1;
      border: 0;
      display: block;
      background: transparent;
    }}
    {extra_css}
  </style>
</head>
<body>
  <div class="backdrop-grid">
    <header class="showcase-head">
      <div class="head-left">
        <span class="eyebrow-badge">{html.escape(spec.badge)}</span>
        <div class="head-titles">
          <h1>{html.escape(spec.title_en)}</h1>
          <p>{html.escape(spec.title_zh)}</p>
        </div>
      </div>
      <div class="head-pills">{pills_html}</div>
    </header>
    {inner_body}
  </div>
</body>
</html>"""


def _build_workflow_spec(ux_dir: Path) -> ShowcaseSpec:
    """Build Showcase 1: Teamwork /wf Config Card + Live Progress & COT Card."""
    cfg_raw = (ux_dir / "workflow-team-config.html").read_text(encoding="utf-8")
    msg_raw = (ux_dir / "workflow-message-card.html").read_text(encoding="utf-8")

    cfg_css = """
    html, body { background: #f6f7f9; overflow: hidden; }
    main { max-width: 100%; margin: 0; padding: 10px 12px 12px; zoom: 0.72; }
    main > p.hint { display: none; }
    .card { box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08); }
    .body { padding: 14px 16px; }
    .section { margin-top: 10px; padding-top: 10px; }
    .button-row { margin-top: 10px; }
    .payload-preview, #payload-preview { display: none; }
    .section:last-child { display: none; }
    """
    msg_css = """
    html, body { background: #f2f4f7; min-height: 0; overflow: hidden; }
    .page { width: 100%; margin: 0; padding: 10px 12px 12px; zoom: 0.73; }
    .page-head { display: none; }
    .preview-section { display: none; margin-top: 0; }
    .preview-section[aria-labelledby="running-title"] { display: block; }
    .section-head, .slot-note { display: none; }
    .card-body { padding: 12px 14px; }
    .stream { gap: 6px; padding: 9px 10px 10px; }
    .stream-block { padding: 7px 9px; }
    .message-sequence-note { margin: 8px 0; }
    """

    cfg_doc = html.escape(sanitize_ux_html(cfg_raw, cfg_css), quote=True)
    msg_doc = html.escape(sanitize_ux_html(msg_raw, msg_css), quote=True)

    body_html = f"""
    <div class="stage" style="grid-template-columns: 1fr 1fr;">
      <section class="window-frame">
        <div class="window-bar">
          <div class="dots"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span></div>
          <span class="window-label">Teamwork Pool Config · 团队预设与成员绑定</span>
          <span class="window-tag">/wf · /wf_config</span>
        </div>
        <iframe class="card-frame" scrolling="no" srcdoc="{cfg_doc}"></iframe>
      </section>
      <section class="window-frame">
        <div class="window-bar">
          <div class="dots"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span></div>
          <span class="window-label">Live Execution &amp; Merged COT · 实时主卡与过程流</span>
          <span class="window-tag">Card JSON 2.0</span>
        </div>
        <iframe class="card-frame" scrolling="no" srcdoc="{msg_doc}"></iframe>
      </section>
    </div>"""

    return ShowcaseSpec(
        name="workflow",
        output_filename="showcase-workflow.png",
        width=1440,
        height=880,
        scale=1.2,
        title_en="Teamwork /wf — Interactive Pool Config & Live Execution",
        title_zh="交互式 1–8 智能体池配置（同模型多 Effort 共存）与单卡实时进度 + 合并原生 COT 过程流",
        badge="TEAMWORK /wf",
        pills=("1–8 Heterogeneous Agents", "Freeze Pool & JS Plan", "Merged Native COT"),
        body_html=body_html,
    )


def _build_cockpit_spec(ux_dir: Path) -> ShowcaseSpec:
    """Build Showcase 2: /usage Quota & Low-Quota Alert + /settings Model/Effort Cascade."""
    cockpit_raw = (ux_dir / "tool-usage-and-settings-cockpit.html").read_text(encoding="utf-8")

    def _card_css(keep_index: int, zoom: float) -> str:
        return f"""
        html, body {{ background: #111622; overflow: hidden; height: 100%; }}
        header.page-bar {{ display: none; }}
        main {{
          max-width: 100%;
          margin: 0;
          padding: 12px;
          display: block;
          zoom: {zoom};
        }}
        main > section.card {{ display: none; }}
        main > section.card:nth-child({keep_index}) {{ display: block; }}
        .card-body {{ padding: 14px 16px; }}
        .hr {{ margin: 10px 0; }}
        .field-label {{ margin: 8px 0 4px; }}
        """

    usage_ok_doc = html.escape(sanitize_ux_html(cockpit_raw, _card_css(1, 0.88)), quote=True)
    usage_alert_doc = html.escape(sanitize_ux_html(cockpit_raw, _card_css(2, 0.88)), quote=True)
    settings_doc = html.escape(sanitize_ux_html(cockpit_raw, _card_css(3, 0.80)), quote=True)

    body_html = f"""
    <div class="stage" style="grid-template-columns: repeat(3, minmax(0, 1fr));">
      <section class="window-frame">
        <div class="window-bar">
          <div class="dots"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span></div>
          <span class="window-label">Quota &amp; Session · 配额与会话态</span>
          <span class="window-tag">/usage</span>
        </div>
        <iframe class="card-frame" scrolling="no" srcdoc="{usage_ok_doc}"></iframe>
      </section>
      <div style="display:flex; flex-direction:column; gap:14px; min-height:0;">
        <section class="window-frame" style="flex: 1;">
          <div class="window-bar">
            <div class="dots"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span></div>
            <span class="window-label">Low-Quota Alert · 自适应告警态</span>
            <span class="window-tag">/usage alert</span>
          </div>
          <iframe class="card-frame" scrolling="no" srcdoc="{usage_alert_doc}"></iframe>
        </section>
        <div class="quota-legend">
          <div class="legend-title">Adaptive Thresholds &amp; Cascade · 告警与级联策略</div>
          <div class="legend-items">
            <span><i class="swatch blue"></i>Healthy (&ge;20%)</span>
            <span><i class="swatch amber"></i>Warning (&lt;20%)</span>
            <span><i class="swatch red"></i>Critical (&lt;5%)</span>
          </div>
          <p>同组模型共享 5h 与周窗口；<code>/settings</code> 按 Tool → Model → Planner/Executor Effort 自动级联并实时持久化。</p>
        </div>
      </div>
      <section class="window-frame">
        <div class="window-bar">
          <div class="dots"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span></div>
          <span class="window-label">Runtime Settings · 模型与推理强度</span>
          <span class="window-tag">/settings</span>
        </div>
        <iframe class="card-frame" scrolling="no" srcdoc="{settings_doc}"></iframe>
      </section>
    </div>"""

    extra_css = """
    .quota-legend {
      flex: 0 0 auto;
      padding: 13px 15px;
      border-radius: 12px;
      background: #131c31;
      border: 1px solid rgba(148, 163, 184, 0.24);
      box-shadow: 0 12px 28px rgba(2, 6, 23, 0.42);
    }
    .legend-title {
      font-size: 12px;
      font-weight: 700;
      color: #e2e8f0;
      margin-bottom: 7px;
    }
    .legend-items {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 11.5px;
      color: #cbd5e1;
      margin-bottom: 7px;
    }
    .legend-items span {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .swatch {
      display: inline-block;
      width: 9px;
      height: 9px;
      border-radius: 3px;
    }
    .swatch.blue { background: #1d4ed8; }
    .swatch.amber { background: #d97706; }
    .swatch.red { background: #b91c1c; }
    .quota-legend p {
      margin: 0;
      font-size: 11.5px;
      line-height: 1.45;
      color: #94a3b8;
    }
    .quota-legend code {
      padding: 1px 5px;
      border-radius: 4px;
      background: #1e293b;
      color: #93c5fd;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
    }
    """

    return ShowcaseSpec(
        name="cockpit",
        output_filename="showcase-cockpit.png",
        width=1440,
        height=880,
        scale=1.2,
        title_en="ACP Cockpits — /usage Quota Telemetry & /settings Cascade",
        title_zh="实时模型配额窗口、G1 Credits、自适应告警态与 Tool → Model → Effort 级联设置面板",
        badge="COCKPITS",
        pills=("/usage Telemetry", "Low-Quota Alert Header", "/settings Cascade"),
        body_html=body_html,
        extra_css=extra_css,
    )


def _build_programming_spec(ux_dir: Path) -> ShowcaseSpec:
    """Build Showcase 3: Hybrid COT + /status Workspace Cockpit + Native /cmds Card."""
    cot_raw = (ux_dir / "programming-cot-hybrid.html").read_text(encoding="utf-8")
    status_raw = (ux_dir / "workspace-cockpit-status.html").read_text(encoding="utf-8")
    cmds_raw = (ux_dir / "native-tool-commands-card.html").read_text(encoding="utf-8")

    cot_css = """
    html, body { background: #f3f5f8; min-height: 0; overflow: hidden; }
    main { width: 100%; margin: 0; padding: 10px 12px; zoom: 0.64; }
    h1, .lead, .rule { display: none; }
    .states { grid-template-columns: 1fr; gap: 10px; }
    .states > article:nth-child(n+3) { display: none; }
    .chat { gap: 7px; }
    .deliverables { margin: 5px 0 0; }
    .footer { margin-top: 8px; padding-top: 7px; }
    """
    status_css = """
    html, body { background: #111318; overflow: hidden; }
    main { display: block; padding: 12px; zoom: 0.75; }
    .card { max-width: 100%; }
    .body { padding: 14px 16px; }
    .kv { padding: 9px 12px; margin: 7px 0; }
    .hr { margin: 10px 0; }
    """
    ready_css = """
    html, body { background: #111318; overflow: hidden; }
    main { display: block; padding: 12px; zoom: 0.80; }
    .cmds { display: none; }
    .card { max-width: 100%; }
    .body { padding: 14px 16px; }
    .hr { margin: 10px 0; }
    """
    cmds_css = """
    html, body { background: #111318; overflow: hidden; }
    main { display: block; padding: 12px; zoom: 0.80; }
    .ready { display: none; }
    .card { max-width: 100%; }
    """

    cot_doc = html.escape(sanitize_ux_html(cot_raw, cot_css), quote=True)
    status_doc = html.escape(sanitize_ux_html(status_raw, status_css), quote=True)
    ready_doc = html.escape(sanitize_ux_html(cmds_raw, ready_css), quote=True)
    cmds_doc = html.escape(sanitize_ux_html(cmds_raw, cmds_css), quote=True)

    body_html = f"""
    <div class="stage" style="grid-template-columns: 1.05fr 0.96fr 0.99fr;">
      <section class="window-frame">
        <div class="window-bar">
          <div class="dots"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span></div>
          <span class="window-label">Hybrid COT + Result Card · 过程流与主卡</span>
          <span class="window-tag">Direct Mode</span>
        </div>
        <iframe class="card-frame" scrolling="no" srcdoc="{cot_doc}"></iframe>
      </section>
      <div style="display:flex; flex-direction:column; gap:14px; min-height:0;">
        <section class="window-frame" style="flex: 1.48;">
          <div class="window-bar">
            <div class="dots"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span></div>
            <span class="window-label">Workspace Cockpit · 工作区总览</span>
            <span class="window-tag">/status</span>
          </div>
          <iframe class="card-frame" scrolling="no" srcdoc="{status_doc}"></iframe>
        </section>
        <section class="window-frame" style="flex: 1;">
          <div class="window-bar">
            <div class="dots"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span></div>
            <span class="window-label">Session Ready · 原生命令透传提示</span>
            <span class="window-tag">/agy</span>
          </div>
          <iframe class="card-frame" scrolling="no" srcdoc="{ready_doc}"></iframe>
        </section>
      </div>
      <section class="window-frame">
        <div class="window-bar">
          <div class="dots"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span></div>
          <span class="window-label">Native Slash &amp; Skills · 一键直达</span>
          <span class="window-tag">/cmds</span>
        </div>
        <iframe class="card-frame" scrolling="no" srcdoc="{cmds_doc}"></iframe>
      </section>
    </div>"""

    return ShowcaseSpec(
        name="programming",
        output_filename="showcase-programming.png",
        width=1440,
        height=880,
        scale=1.2,
        title_en="Direct Programming — Hybrid COT Stream, /status & Native /cmds",
        title_zh="原生 COT 推理与工具调用过程流、工作区分支/锁驾驶舱 (/status) 及后端原生命令与 Skills 面板 (/cmds)",
        badge="PROGRAMMING",
        pills=("COT Streaming", "/status Workspace Cockpit", "Clean / Pass-Through"),
        body_html=body_html,
    )


def _build_report_spec(ux_dir: Path) -> ShowcaseSpec:
    """Build Showcase 4: Structured GFM/HTML Workflow Deliverable Report."""
    report_raw = (ux_dir / "workflow-report-preview.html").read_text(encoding="utf-8")

    report_css = """
    html, body { overflow: hidden; zoom: 0.69; }
    .page-header { padding: 20px 0 18px; }
    .stats-grid { margin-top: 14px; }
    .exec-banner { margin-top: 14px; padding: 14px 18px; }
    .toolbar-wrap { position: static; }
    main { padding: 14px 24px 24px; }
    details.report-section { margin: 10px 0; }
    .section-body-rich { padding: 14px 18px; gap: 12px; }
    #section-1 .semantic-block:nth-child(n+2) { display: none; }
    #section-2, #section-3, #section-4, #section-phases, #appendix { display: none; }
    """
    report_doc = html.escape(sanitize_ux_html(report_raw, report_css), quote=True)

    body_html = f"""
    <div class="stage" style="grid-template-columns: 1fr;">
      <section class="window-frame">
        <div class="window-bar">
          <div class="dots"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span></div>
          <span class="window-label">workflow-report-4486548ffb98.html — Interactive Deliverable Report</span>
          <span class="window-tag">GFM + HTML</span>
        </div>
        <iframe class="card-frame" scrolling="no" srcdoc="{report_doc}"></iframe>
      </section>
    </div>"""

    return ShowcaseSpec(
        name="report",
        output_filename="showcase-report.png",
        width=1440,
        height=880,
        scale=1.2,
        title_en="Workflow Deliverable Report — Structured Findings & Evidence",
        title_zh="包含统计概览、执行结论、全文检索、TOC 导航、GFM 对比表及阶段溯源抽屉的可交付 HTML 报告",
        badge="DELIVERABLE",
        pills=("Above-the-Fold Verdict", "Structured GFM Tables", "Search & Audit Appendix"),
        body_html=body_html,
    )


def build_showcase_specs(ux_dir: Path) -> list[ShowcaseSpec]:
    """Construct all 4 README UI showcase definitions."""
    return [
        _build_workflow_spec(ux_dir),
        _build_cockpit_spec(ux_dir),
        _build_programming_spec(ux_dir),
        _build_report_spec(ux_dir),
    ]


def render_html_to_png(
    browser: str,
    html_content: str,
    output_png: Path,
    width: int,
    height: int,
    scale: float,
    tmp_dir: Path,
    transparent: bool = False,
) -> tuple[int, int, int]:
    """Render an HTML string to a cropped, level-9 optimized PNG via headless Chrome."""
    tmp_html = tmp_dir / f"{output_png.stem}.html"
    tmp_html.write_text(html_content, encoding="utf-8")

    # Request extra outer window height so Chrome's ~87px chrome bar does not clip viewport.
    win_h = height + CHROME_HEIGHT_PAD_PX
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        f"--force-device-scale-factor={scale}",
        f"--window-size={width},{win_h}",
        f"--screenshot={output_png}",
    ]
    if transparent:
        cmd.append("--default-background-color=00000000")
    cmd.append(tmp_html.as_uri())

    env = os.environ.copy()
    env.setdefault("LANG", "C.UTF-8")
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    target_px_height = int(math.ceil(height * scale))
    return crop_and_optimize_png(output_png, target_px_height)


def render_svg_previews(browser: str, assets_dir: Path, preview_dir: Path) -> None:
    """Render each SVG under ``assets_dir`` into ``preview_dir`` for visual inspection."""
    svg_sizes = {
        "hero-banner.svg": (1280, 380),
        "architecture-overview.svg": (1200, 620),
        "workflow-pipeline.svg": (1200, 340),
    }
    preview_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ghostap_svg_preview_") as tmp:
        tmp_dir = Path(tmp)
        for filename, (w, h) in svg_sizes.items():
            svg_path = assets_dir / filename
            if not svg_path.exists():
                continue
            svg_markup = svg_path.read_text(encoding="utf-8")
            wrapper = (
                "<!doctype html><html><body style=\"margin:0;padding:0;overflow:hidden;background:transparent;\">"
                f'<div style="width:{w}px;height:{h}px;">{svg_markup}</div></body></html>'
            )
            out_png = preview_dir / f"preview-{svg_path.stem}.png"
            pw, ph, size_b = render_html_to_png(
                browser, wrapper, out_png, w, h, scale=1.0, tmp_dir=tmp_dir, transparent=True
            )
            print(f"[svg-preview] {out_png} -> {pw}x{ph} ({size_b / 1024:.1f} KiB)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Render GhostAP README UI showcases from ux/*.html")
    parser.add_argument("--ux-dir", type=Path, default=DEFAULT_UX_DIR, help="Path to ux/*.html directory")
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR, help="Output directory for PNGs")
    parser.add_argument("--browser", type=str, default=None, help="Explicit Chrome/Chromium binary path")
    parser.add_argument("--scale", type=float, default=None, help="Override device scale factor (default per spec)")
    parser.add_argument("--only", nargs="+", default=None, help="Render only named showcases (workflow cockpit programming report)")
    parser.add_argument(
        "--preview-svgs-dir",
        type=Path,
        default=None,
        help="Optional directory to also render docs/assets/*.svg to PNGs for visual inspection",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for rendering README visual showcase PNGs."""
    args = parse_args(argv)
    browser = find_browser_binary(args.browser)
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    specs = build_showcase_specs(args.ux_dir)
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            print(f"No matching showcases for: {sorted(wanted)}", file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory(prefix="ghostap_showcase_") as tmp:
        tmp_dir = Path(tmp)
        for spec in specs:
            scale = args.scale if args.scale is not None else spec.scale
            out_png = args.assets_dir / spec.output_filename
            frame_html = _frame_shell(spec, spec.body_html, spec.extra_css)
            pw, ph, size_b = render_html_to_png(
                browser, frame_html, out_png, spec.width, spec.height, scale, tmp_dir
            )
            size_kib = size_b / 1024
            warn = " [WARN > 600 KB]" if size_b > DEFAULT_TARGET_MAX_BYTES else ""
            print(f"[showcase] {out_png} -> {pw}x{ph} ({size_kib:.1f} KiB){warn}")

    if args.preview_svgs_dir is not None:
        render_svg_previews(browser, args.assets_dir, args.preview_svgs_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
