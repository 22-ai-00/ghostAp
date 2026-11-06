"""Truncation limits and centralized thresholds."""

# ──────────────────────────────────────────────────────────────
# Truncation Limits — aligned with pokoclaw card-truncation.ts
# ──────────────────────────────────────────────────────────────
TRUNCATION_LIMITS: dict[str, int] = {
    "card_string_max_chars": 220,
    "card_string_max_lines": 6,
    "bash_max_chars": 240,
    "bash_max_lines": 8,
    "reasoning_tail_max": 500,
}

# ──────────────────────────────────────────────────────────────
# Centralized Thresholds — single source of truth for all
# truncation / folding / pagination limits across the project.
# ──────────────────────────────────────────────────────────────
THRESHOLDS = {
    # Card content element max characters (core.py _build_content_element)
    "CONTENT_MAX_CHARS": 25000,
    # Deep/Spec engine card folding — compact mode line threshold
    "COMPACT_LINE_THRESHOLD": 15,
    # Deep/Spec engine card folding — full mode line threshold
    "FULL_LINE_THRESHOLD": 50,
    # Acceptance criteria folding line threshold
    "AC_LINE_THRESHOLD": 10,
    # Compact mode long-line character fallback
    "COMPACT_CHAR_FALLBACK": 1500,
    # Shell command stdout max characters
    "SHELL_STDOUT_MAX": 16000,
    # Shell command stderr max characters
    "SHELL_STDERR_MAX": 8000,
    # BaseRenderer collapsible section item threshold
    "COLLAPSE_ITEM_THRESHOLD": 8,
    # BaseRenderer collapsible section long-text line threshold
    "COLLAPSE_LINE_THRESHOLD": 30,
    # BaseRenderer collapsible section display lines (when folded)
    "COLLAPSE_DISPLAY_LINES": 15,
    # Streaming card default visible characters
    "STREAMING_VISIBLE_CHARS": 25000,
    # Streaming card pagination step
    "PAGINATION_STEP": 5000,
    # Max continuation cards before giving up
    "CONTINUATION_MAX_CARDS": 10,
    # Min content length to enable collapsible panels (avoid overhead for short content)
    "COLLAPSIBLE_MIN_CHARS": 2000,
    # Max collapsible elements before falling back to flat markdown
    "COLLAPSIBLE_MAX_ELEMENTS": 20,
    # Card payload byte budget (aligned with pokoclaw: 27 * 1024)
    "CARD_BYTE_BUDGET": 27 * 1024,
    # Card node budget (max tagged nodes per card)
    "CARD_NODE_BUDGET": 180,
}
