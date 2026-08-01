"""Tests for card theme completeness — ensures all engine types and terminal states are covered."""


from src.card.themes import MODE_TEMPLATES, TERMINAL_TEMPLATES


class TestTerminalTemplatesComplete:
    """TERMINAL_TEMPLATES must cover all TerminalStatus values that can be terminal."""

    def test_covers_key_terminal_states(self):
        """At minimum: completed, failed, cancelled must have templates."""
        required = {"completed", "failed", "cancelled"}
        actual = set(TERMINAL_TEMPLATES.keys())
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_all_values_are_valid_colors(self):
        """All template values should be non-empty strings."""
        for key, value in TERMINAL_TEMPLATES.items():
            assert isinstance(value, str) and len(value) > 0, f"Invalid template for {key}: {value!r}"


class TestModeTemplatesComplete:
    """MODE_TEMPLATES must cover all mode names used by renderers."""

    def test_covers_known_engine_modes(self):
        """All known engine mode names used in AGENTS.md/renderers should have templates."""
        known_modes = {"Deep Agent", "Spec Engine", "Worktree"}
        actual = set(MODE_TEMPLATES.keys())
        assert known_modes.issubset(actual), f"Missing: {known_modes - actual}"

    def test_all_values_are_valid_colors(self):
        """All template values should be non-empty strings."""
        for key, value in MODE_TEMPLATES.items():
            assert isinstance(value, str) and len(value) > 0, f"Invalid template for {key}: {value!r}"

# ---------------------------------------------------------------------------
# Merged from test_card_styles_split.py
# ---------------------------------------------------------------------------


class TestThemes:
    def test_get_available_themes(self):
        from src.card.themes import DARK_THEME_NAMES, get_available_themes
        light_themes = get_available_themes(include_dark=False)
        all_themes = get_available_themes(include_dark=True)
        assert len(all_themes) >= 18
        assert len(light_themes) < len(all_themes)
        for dark_name in DARK_THEME_NAMES:
            assert dark_name not in light_themes
