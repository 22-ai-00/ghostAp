"""Single source of truth (SSOT) registry for all GhostAP programming backends.

This module intentionally uses only Python standard library types so it can be
imported cleanly from ``src.mode``, ``src.project``, ``src.agent``,
``src.acp``, ``src.card``, ``src.feishu``, and ``src.spec_engine`` without
creating circular dependencies or pulling heavy UI/SDK trees into cold startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional


class InteractionMode(Enum):
    """交互模式枚举。"""

    SMART = "smart"
    COCO = "coco"
    CLAUDE = "claude"
    CLAUDE_W = "claude_w"
    AIDEN = "aiden"
    CODEX = "codex"
    CODEX_W = "codex_w"
    GEMINI = "gemini"
    TRAEX = "traex"
    GROK = "grok"
    DSH = "dsh"
    AGY = "agy"
    SHELL = "shell"


@dataclass(frozen=True)
class BackendSpec:
    """Authoritative specification for one programming tool backend."""

    mode_key: str
    interaction_mode: InteractionMode
    cli_name: str
    display_name: str
    emoji: str
    mode_emoji: str
    template_color: str
    visible_in_tools_list: bool = True
    transport: Literal["acp", "cli"] = "acp"
    uses_claude_cli: bool = False
    session_timeout_attr: str = "coco_session_timeout"
    is_coco: bool = False
    legacy_exit_mode: Literal["dedicated_intent", "shared_exit_alias", "none"] = "none"
    nl_enter_keywords: tuple[str, ...] = ()
    supports_info_command: bool = True

    @property
    def slash_command(self) -> str:
        """Primary Feishu slash command for entering this backend."""
        return f"/{self.cli_name}"

    @property
    def enter_alias_command(self) -> str:
        """Legacy ``/enter_<mode_key>`` alias for entering this backend."""
        return f"/enter_{self.mode_key}"

    @property
    def info_command(self) -> str:
        """Legacy ``/<mode_key>_info`` introspection command."""
        return f"/{self.mode_key}_info"

    @property
    def manager_attr(self) -> str:
        """Attribute name of this backend's session manager on HandlerContext."""
        return f"{self.mode_key}_manager"

    @property
    def is_mode_predicate_name(self) -> str:
        """Name of the ``is_<mode_key>_mode`` helper on ModeManager."""
        return f"is_{self.mode_key}_mode"

    @property
    def project_mode_attr(self) -> str:
        """Boolean flag attribute name on ProjectContext."""
        return f"{self.mode_key}_mode"

    @property
    def project_snapshot_attr(self) -> str:
        """Session snapshot attribute name on ProjectContext."""
        return f"{self.mode_key}_session_snapshot"

    @property
    def mode_display_name(self) -> str:
        """Formatted UI banner name for ModeManager.get_mode_display_name()."""
        return f"{self.mode_emoji} {self.display_name} 编程模式"


# Explicit check order puts hidden bridges & specialized backends cleanly alongside
# primary tools, with coco last when scanning active flags so explicit non-default
# flags always win.
ALL_BACKEND_SPECS: tuple[BackendSpec, ...] = (
    BackendSpec(
        mode_key="coco",
        interaction_mode=InteractionMode.COCO,
        cli_name="coco",
        display_name="Coco",
        emoji="🤖",
        mode_emoji="🤖",
        template_color="blue",
        visible_in_tools_list=True,
        transport="acp",
        uses_claude_cli=False,
        session_timeout_attr="coco_session_timeout",
        is_coco=True,
        legacy_exit_mode="dedicated_intent",
        nl_enter_keywords=("进入编程模式", "编程模式", "开始编程", "进入coco", "coco模式"),
        supports_info_command=True,
    ),
    BackendSpec(
        mode_key="claude",
        interaction_mode=InteractionMode.CLAUDE,
        cli_name="claude",
        display_name="Claude",
        emoji="🔮",
        mode_emoji="🔮",
        template_color="purple",
        visible_in_tools_list=True,
        transport="cli",
        uses_claude_cli=True,
        session_timeout_attr="claude_session_timeout",
        is_coco=False,
        legacy_exit_mode="dedicated_intent",
        nl_enter_keywords=("进入claude模式", "claude模式", "进入claude", "使用claude"),
        supports_info_command=True,
    ),
    BackendSpec(
        mode_key="claude_w",
        interaction_mode=InteractionMode.CLAUDE_W,
        cli_name="claude-w",
        display_name="Claude-W",
        emoji="🔮",
        mode_emoji="🪄",
        template_color="purple",
        visible_in_tools_list=False,
        transport="cli",
        uses_claude_cli=True,
        session_timeout_attr="claude_session_timeout",
        is_coco=False,
        legacy_exit_mode="shared_exit_alias",
        nl_enter_keywords=(),
        supports_info_command=False,
    ),
    BackendSpec(
        mode_key="aiden",
        interaction_mode=InteractionMode.AIDEN,
        cli_name="aiden",
        display_name="Aiden",
        emoji="🎯",
        mode_emoji="🎯",
        template_color="indigo",
        visible_in_tools_list=True,
        transport="acp",
        uses_claude_cli=False,
        session_timeout_attr="coco_session_timeout",
        is_coco=False,
        legacy_exit_mode="dedicated_intent",
        nl_enter_keywords=("进入aiden模式", "aiden模式", "进入aiden", "使用aiden"),
        supports_info_command=True,
    ),
    BackendSpec(
        mode_key="codex",
        interaction_mode=InteractionMode.CODEX,
        cli_name="codex",
        display_name="Codex",
        emoji="💻",
        mode_emoji="⚡",
        template_color="blue",
        visible_in_tools_list=True,
        transport="acp",
        uses_claude_cli=False,
        session_timeout_attr="coco_session_timeout",
        is_coco=False,
        legacy_exit_mode="dedicated_intent",
        nl_enter_keywords=("进入codex模式", "codex模式", "进入codex", "使用codex"),
        supports_info_command=True,
    ),
    BackendSpec(
        mode_key="codex_w",
        interaction_mode=InteractionMode.CODEX_W,
        cli_name="codex-w",
        display_name="Codex-W",
        emoji="💻",
        mode_emoji="🌩️",
        template_color="blue",
        visible_in_tools_list=False,
        transport="cli",
        uses_claude_cli=False,
        session_timeout_attr="coco_session_timeout",
        is_coco=False,
        legacy_exit_mode="shared_exit_alias",
        nl_enter_keywords=(),
        supports_info_command=False,
    ),
    BackendSpec(
        mode_key="gemini",
        interaction_mode=InteractionMode.GEMINI,
        cli_name="gemini",
        display_name="Gemini",
        emoji="✨",
        mode_emoji="✨",
        template_color="turquoise",
        visible_in_tools_list=True,
        transport="acp",
        uses_claude_cli=False,
        session_timeout_attr="coco_session_timeout",
        is_coco=False,
        legacy_exit_mode="dedicated_intent",
        nl_enter_keywords=("进入gemini模式", "gemini模式", "进入gemini", "使用gemini"),
        supports_info_command=True,
    ),
    BackendSpec(
        mode_key="traex",
        interaction_mode=InteractionMode.TRAEX,
        cli_name="traex",
        display_name="Traex",
        emoji="🚀",
        mode_emoji="🚀",
        template_color="orange",
        visible_in_tools_list=True,
        transport="acp",
        uses_claude_cli=False,
        session_timeout_attr="coco_session_timeout",
        is_coco=False,
        legacy_exit_mode="dedicated_intent",
        nl_enter_keywords=("进入traex模式", "traex模式", "进入traex", "使用traex"),
        supports_info_command=True,
    ),
    BackendSpec(
        mode_key="grok",
        interaction_mode=InteractionMode.GROK,
        cli_name="grok",
        display_name="Grok",
        emoji="🌌",
        mode_emoji="🌌",
        template_color="violet",
        visible_in_tools_list=True,
        transport="acp",
        uses_claude_cli=False,
        session_timeout_attr="coco_session_timeout",
        is_coco=False,
        legacy_exit_mode="dedicated_intent",
        nl_enter_keywords=("进入grok模式", "grok模式", "进入grok", "使用grok"),
        supports_info_command=True,
    ),
    BackendSpec(
        mode_key="dsh",
        interaction_mode=InteractionMode.DSH,
        cli_name="dsh",
        display_name="DSH",
        emoji="🧭",
        mode_emoji="🧭",
        template_color="green",
        visible_in_tools_list=True,
        transport="acp",
        uses_claude_cli=False,
        session_timeout_attr="coco_session_timeout",
        is_coco=False,
        legacy_exit_mode="none",
        nl_enter_keywords=("进入dsh模式", "dsh模式", "进入dsh", "使用dsh"),
        supports_info_command=True,
    ),
    BackendSpec(
        mode_key="agy",
        interaction_mode=InteractionMode.AGY,
        cli_name="agy",
        display_name="AGY",
        emoji="🪐",
        mode_emoji="🪐",
        template_color="wathet",
        visible_in_tools_list=True,
        transport="acp",
        uses_claude_cli=False,
        session_timeout_attr="coco_session_timeout",
        is_coco=False,
        legacy_exit_mode="none",
        nl_enter_keywords=("进入agy模式", "agy模式", "进入agy", "使用agy"),
        supports_info_command=True,
    ),
)

VISIBLE_BACKEND_SPECS: tuple[BackendSpec, ...] = tuple(
    spec for spec in ALL_BACKEND_SPECS if spec.visible_in_tools_list
)

BACKEND_SPEC_BY_KEY: dict[str, BackendSpec] = {
    spec.mode_key: spec for spec in ALL_BACKEND_SPECS
}

BACKEND_SPEC_BY_MODE: dict[InteractionMode, BackendSpec] = {
    spec.interaction_mode: spec for spec in ALL_BACKEND_SPECS
}

_BACKEND_LOOKUP: dict[str, BackendSpec] = {}
for _spec in ALL_BACKEND_SPECS:
    _BACKEND_LOOKUP[_spec.mode_key.casefold()] = _spec
    _BACKEND_LOOKUP[_spec.cli_name.casefold()] = _spec
    _BACKEND_LOOKUP[_spec.display_name.casefold()] = _spec

_SCAN_ORDER: tuple[BackendSpec, ...] = (
    *[spec for spec in ALL_BACKEND_SPECS if spec.mode_key != "coco"],
    BACKEND_SPEC_BY_KEY["coco"],
)

# Card-layer compatibility aliases re-exported by src/card/backend_meta.py
ALL_BACKEND_METAS = ALL_BACKEND_SPECS
VISIBLE_BACKEND_METAS = VISIBLE_BACKEND_SPECS
BACKEND_META_BY_KEY = BACKEND_SPEC_BY_KEY


def get_backend_spec(key_or_name: object) -> Optional[BackendSpec]:
    """Look up backend specification by mode_key, CLI command name, or display name."""
    if isinstance(key_or_name, InteractionMode):
        key = key_or_name.value.casefold()
    else:
        key = str(key_or_name or "").strip().casefold()
    if not key:
        return None
    if key in _BACKEND_LOOKUP:
        return _BACKEND_LOOKUP[key]
    return _BACKEND_LOOKUP.get(key.replace("-", "_"))


get_backend_meta = get_backend_spec


def resolve_project_programming_mode(
    project: object,
    chat_id: Optional[str] = None,
) -> Optional[InteractionMode]:
    """Resolve the active ``InteractionMode`` from any of the project mode flags."""
    if project is None:
        return None
    normalized_chat = str(chat_id or "").strip()
    if normalized_chat and callable(getattr(type(project), "get_programming_mode", None)):
        for spec in _SCAN_ORDER:
            if project.get_programming_mode(spec.mode_key, normalized_chat) is True:  # type: ignore[attr-defined]
                return spec.interaction_mode
        return None
    for spec in _SCAN_ORDER:
        flag = getattr(project, spec.project_mode_attr, False)
        if isinstance(flag, bool) and flag:
            return spec.interaction_mode
    return None
