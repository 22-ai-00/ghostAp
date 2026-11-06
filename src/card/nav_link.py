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


def format_task_continuation_link(
    task_name: str,
    rotation_count: int,
    new_msg_id: str | None,
) -> str:
    """Build continuation text for a task-level card rotation.

    Args:
        task_name: Display name of the task.
        rotation_count: How many times this task has been rotated (1-based).
        new_msg_id: The message_id of the new continuation card (may be None/empty).

    Returns:
        Markdown text to dispatch as the final text block on the old card.
    """
    page = rotation_count + 1
    link_text = UI_TEXT["orch_nav_link_text"]

    if new_msg_id:
        return UI_TEXT["orch_task_continuation_nav"].format(
            task_name=task_name,
            page=page,
            link_text=link_text,
            msg_id=new_msg_id,
        )
    return UI_TEXT["orch_task_continuation_nav_fallback"].format(
        task_name=task_name,
        page=page,
    )


def format_back_link(old_msg_id: str | None, *, task_name: str | None = None) -> str | None:
    """Build a back-link pointing to the previous (old) card.

    Args:
        old_msg_id: The message_id of the previous card. If None/empty,
                    returns None (no back-link available).
        task_name: Optional task name for context in multi-task scenarios.

    Returns:
        Markdown text with a lark://message/ deep-link, or None if unavailable.
    """
    if not old_msg_id:
        return None
    if task_name:
        return UI_TEXT["orch_back_link_with_task"].format(
            task_name=task_name, msg_id=old_msg_id
        )
    return UI_TEXT["orch_back_link"].format(msg_id=old_msg_id)
