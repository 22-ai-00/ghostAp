"""Project theme definitions, engine styles, and panel styles."""

from dataclasses import dataclass


@dataclass
class ProjectTheme:
    name: str
    color: str
    emoji: str
    header_template: str


# 优化后的主题配色系统，确保 WCAG AA 级对比度（至少 4.5:1）
# 选择更适合移动端显示的、对比度更好的颜色
THEMES = {
    "green": ProjectTheme("green", "green", "🟢", "green"),
    "blue": ProjectTheme("blue", "blue", "🔵", "blue"),
    "purple": ProjectTheme("purple", "purple", "🟣", "purple"),
    "orange": ProjectTheme("orange", "orange", "🟠", "orange"),
    "red": ProjectTheme("red", "red", "🔴", "red"),
    "turquoise": ProjectTheme("turquoise", "turquoise", "🩵", "turquoise"),
    "violet": ProjectTheme("violet", "violet", "🟣", "violet"),
    "indigo": ProjectTheme("indigo", "indigo", "🟣", "indigo"),
    "carmine": ProjectTheme("carmine", "carmine", "🔴", "carmine"),
    "wathet": ProjectTheme("wathet", "wathet", "🔵", "wathet"),
    "grey": ProjectTheme("grey", "grey", "⚪", "grey"),
    "yellow": ProjectTheme("yellow", "yellow", "🟡", "yellow"),
    # 深色主题变体 - 为深色模式优化的配色
    "dark_green": ProjectTheme("dark_green", "dark_green", "🌲", "dark_green"),
    "dark_blue": ProjectTheme("dark_blue", "dark_blue", "🌙", "dark_blue"),
    "dark_purple": ProjectTheme("dark_purple", "dark_purple", "🪻", "dark_purple"),
    "dark_orange": ProjectTheme("dark_orange", "dark_orange", "🍂", "dark_orange"),
    "dark_red": ProjectTheme("dark_red", "dark_red", "🍎", "dark_red"),
    "dark": ProjectTheme("dark", "dark", "⚫", "dark"),
}

# 深色主题名称列表（不参与自动分配）
DARK_THEME_NAMES = {"dark_green", "dark_blue", "dark_purple", "dark_orange", "dark_red", "dark"}


def get_available_themes(include_dark: bool = False) -> dict[str, "ProjectTheme"]:
    """获取可用的主题列表。

    Args:
        include_dark: 是否包含深色主题，默认为 False（深色主题不参与自动分配）

    Returns:
        主题字典
    """
    if include_dark:
        return THEMES.copy()
    return {name: theme for name, theme in THEMES.items() if name not in DARK_THEME_NAMES}


def get_theme(color: str) -> ProjectTheme:
    """Get a theme by color name, defaulting to green."""
    return THEMES.get(color, THEMES["green"])


# Engine Style Configuration
ENGINE_STYLES = {
    "spec": {
        "color": "green",
        "icon": "🧠",
        "label_format": "Spec Engine ({name})",
        "meta_separator": " · ",
        "features": {"history_button": False},
    },
    "claude": {
        "color": "violet",
        "icon": "🧠",
        "label_format": "Deep Agent ({name})",
        "meta_separator": " · ",
        "features": {"history_button": False},
    },
    "default": {
        "color": "turquoise",
        "icon": "🧠",
        "label_format": "Deep Agent ({name})",
        "meta_separator": " · ",
        "features": {"history_button": False},
    },
}

# ──────────────────────────────────────────────────────────────
# Collapsible Panel Styles — aligned with pokoclaw tool-calls.ts
# ──────────────────────────────────────────────────────────────
PANEL_STYLES = {
    "corner_radius": "8px",
    "padding": "8px 8px 8px 8px",
    "vertical_spacing": "8px",
    "border_normal": "grey",
    "border_failed": "red",
    "border_history": "blue",
    "border_plan": "indigo",
    "border_active": "wathet",  # Active/running state
    "border_task_list": "indigo",  # Task list panel (multi-card orchestration) — distinct from engine headers
    "padding_standard": "8px 16px",  # Standard panel content padding
    "padding_compact": "4px 12px",  # Compact padding for dense summaries
}

# ──────────────────────────────────────────────────────────────
# Card Theme: unified color mapping for card headers
# (Merged from card_theme.py — single source of truth for
#  engine_type × phase → template color)
# ──────────────────────────────────────────────────────────────

# Header template colors by terminal state
TERMINAL_TEMPLATES: dict[str, str] = {
    "completed": "green",
    "completed_empty": "orange",
    "failed": "red",
    "cancelled": "grey",
    "archived": "grey",
    "paused": "orange",
    "awaiting_approval": "indigo",
    "blocked": "red",
    "denied": "red",
    "continued": "green",
}

# Header template colors by mode name (when running)
MODE_TEMPLATES: dict[str, str] = {
    "Coco": "blue",
    "Claude": "purple",
    "Gemini": "turquoise",
    "TTADK": "orange",
    "Deep": "violet",
    "Deep Agent": "violet",
    "Spec": "green",
    "Spec Engine": "green",
    "Smart": "turquoise",
}
