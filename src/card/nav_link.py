"""Navigation link formatting for card continuation (续卡).

Provides a single source of truth for deep-link and fallback text
used by SessionRotator when archiving cards.
"""

from __future__ import annotations

from src.card.ui_text import UI_TEXT


def format_navigation_link(
    new_msg_id: str | None,
    rotation_seq: int,
) -> tuple[str, str | None]:
    """Build navigation summary text for an archived card.

    Args:
        new_msg_id: The message_id of the new continuation card (may be None/empty
                    if not yet delivered).
        rotation_seq: The current rotation sequence number (1-based).

    Returns:
        A tuple of (nav_summary, fallback_notice):
        - nav_summary: Markdown text to dispatch as ARCHIVED summary on the old card.
        - fallback_notice: Optional plain-text notice to send when deep-link is
                          unavailable (None when deep-link is available).
    """
    link_text = UI_TEXT["orch_nav_link_text"]

    if new_msg_id:
        nav_summary = UI_TEXT["orch_continuation_link"].format(
            seq=rotation_seq,
            link_text=link_text,
            msg_id=new_msg_id,
        )
        return nav_summary, None

    nav_summary = UI_TEXT["orch_continuation_link_fallback"].format(
        seq=rotation_seq,
    )
    fallback_notice = UI_TEXT["orch_archived_navigate_fallback"]
    return nav_summary, fallback_notice
