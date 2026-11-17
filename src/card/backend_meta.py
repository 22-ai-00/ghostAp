"""Unified display and status metadata for all GhostAP programming backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.mode import InteractionMode


@dataclass(frozen=True)
class BackendMeta:
    """Single source of truth for one programming backend's display metadata."""

    mode_key: str
    cli_name: str
    display_name: str
    emoji: str
    template_color: str
    visible_in_tools_list: bool = True

    @property
    def project_mode_attr(self) -> str:
        return f"{self.mode_key}_mode"

    @property
    def project_snapshot_attr(self) -> str:
        return f"{self.mode_key}_session_snapshot"


# Explicit check order puts hidden bridges & specialized backends cleanly alongside primary tools,
# with coco last when scanning active flags so explicit non-default flags always win.
ALL_BACKEND_METAS: tuple[BackendMeta, ...] = (
    BackendMeta("coco", "coco", "Coco", "🤖", "blue", True),
    BackendMeta("claude", "claude", "Claude", "🔮", "purple", True),
    BackendMeta("claude_w", "claude-w", "Claude-W", "🔮", "purple", False),
    BackendMeta("aiden", "aiden", "Aiden", "🎯", "indigo", True),
    BackendMeta("codex", "codex", "Codex", "💻", "blue", True),
    BackendMeta("codex_w", "codex-w", "Codex-W", "💻", "blue", False),
    BackendMeta("gemini", "gemini", "Gemini", "✨", "turquoise", True),
    BackendMeta("traex", "traex", "Traex", "🚀", "orange", True),
    BackendMeta("grok", "grok", "Grok", "🌌", "violet", True),
    BackendMeta("dsh", "dsh", "DSH", "🧭", "green", True),
    BackendMeta("agy", "agy", "AGY", "🪐", "wathet", True),
)

VISIBLE_BACKEND_METAS: tuple[BackendMeta, ...] = tuple(
    meta for meta in ALL_BACKEND_METAS if meta.visible_in_tools_list
)

BACKEND_META_BY_KEY: dict[str, BackendMeta] = {
    meta.mode_key: meta for meta in ALL_BACKEND_METAS
}

_BACKEND_LOOKUP: dict[str, BackendMeta] = {}
for _meta in ALL_BACKEND_METAS:
    _BACKEND_LOOKUP[_meta.mode_key.casefold()] = _meta
    _BACKEND_LOOKUP[_meta.cli_name.casefold()] = _meta
    _BACKEND_LOOKUP[_meta.display_name.casefold()] = _meta

_SCAN_ORDER: tuple[BackendMeta, ...] = (
    *[meta for meta in ALL_BACKEND_METAS if meta.mode_key != "coco"],
    BACKEND_META_BY_KEY["coco"],
)


def get_backend_meta(key_or_name: object) -> Optional[BackendMeta]:
    """Look up backend metadata by mode_key, CLI command name, or display name."""
    if isinstance(key_or_name, InteractionMode):
        key = key_or_name.value.casefold()
    else:
        key = str(key_or_name or "").strip().casefold()
    if not key:
        return None
    if key in _BACKEND_LOOKUP:
        return _BACKEND_LOOKUP[key]
    return _BACKEND_LOOKUP.get(key.replace("-", "_"))


def resolve_project_programming_mode(project: object) -> Optional[InteractionMode]:
    """Resolve the active ``InteractionMode`` from any of the 11 project flags."""
    if project is None:
        return None
    for meta in _SCAN_ORDER:
        flag = getattr(project, meta.project_mode_attr, False)
        if isinstance(flag, bool) and flag:
            try:
                return InteractionMode(meta.mode_key)
            except ValueError:
                continue
    return None
