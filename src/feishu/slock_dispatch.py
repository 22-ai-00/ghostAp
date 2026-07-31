"""Slock passive-activation dispatch layer.

Isolates all ``src.slock_engine`` imports that were previously scattered in
``ws_client.py`` (lines 746/771/813).  Other engines never needed a direct
import path from the WS ingress layer; slock should follow the same rule.

Public API consumed by ``ws_client.py``:
    - ``should_auto_activate(...)`` — classification gate
    - ``try_passive_activation(...)`` — full activation attempt (guard → lock → bootstrap → error card)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    import threading

    from ..project import ProjectContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol for the minimal ws_client surface needed by activation logic
# ---------------------------------------------------------------------------


class _SlockHandlerLike(Protocol):
    """Minimal interface of the slock handler needed for activation."""

    def activate_slock(
        self,
        *,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: Optional[Any],
        skip_guard_check: bool,
    ) -> bool: ...


class _CardSenderLike(Protocol):
    """Minimal interface for sending cards to a chat."""

    def send_card_to_chat(self, chat_id: str, card_json: str) -> None: ...


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def should_auto_activate(
    chat_id: str,
    text: str,
    *,
    chat_type: str = "group",
    is_managed: bool = False,
) -> bool:
    """Determine whether a message should trigger slock auto-activation.

    Short-circuits for already-managed chats to avoid redundant classification
    overhead.  Only performs task classification for unmanaged group chats.

    Returns True if:
    - Chat is already managed by slock (short-circuit), or
    - Chat is an unmanaged group AND text is classified as a task
    """
    if chat_type != "group":
        return False
    if is_managed:
        return True
    if (text or "").lstrip().startswith("/"):
        return False

    from src.slock_engine.task_classifier import TaskClassifier

    return TaskClassifier.is_task(text)


def try_passive_activation(
    chat_id: str,
    text: str,
    *,
    project: Optional["ProjectContext"] = None,
    settings: Any,
    is_managed_fn: Any,
    get_chat_lock_fn: Any,
    slock_handler: _SlockHandlerLike,
    card_sender: _CardSenderLike,
    effective_trust: Any = None,
) -> tuple[bool, str]:
    """Attempt passive slock activation for an unmanaged chat.

    Encapsulates:
    1. ActivationGuard permission/rate-limit check
    2. Per-chat lock acquisition with double-check
    3. Slock handler bootstrap call
    4. Error-card notification on failure

    Parameters
    ----------
    chat_id : str
        Target chat identifier.
    text : str
        User message that triggered the activation attempt.
    project : ProjectContext | None
        Resolved project context, if any.
    settings : Settings
        Application settings (passed to ActivationGuard).
    is_managed_fn : () -> bool
        Callable returning whether the chat is already managed.
    get_chat_lock_fn : () -> threading.Lock
        Callable returning the per-chat activation lock.
    slock_handler : _SlockHandlerLike
        Handler capable of bootstrapping the slock engine.
    card_sender : _CardSenderLike
        Handler capable of sending cards to a chat (for error notifications).

    Returns
    -------
    tuple[bool, str]
        (success, reason) — same contract as the former
        ``FeishuWSClient._auto_activate_slock``.
    """
    from src.slock_engine.activation_guard import (
        ACTIVATION_ALLOWED,
        get_activation_guard,
    )
    from src.thread.manager import get_current_sender_id

    # Permission and rate-limit gate
    guard = get_activation_guard()
    sender_id = get_current_sender_id() or ""
    allowed, reason = guard.can_auto_activate(
        sender_id,
        chat_id,
        settings,
        effective_trust=effective_trust,
    )
    if not allowed:
        logger.debug(
            "Auto-activate blocked by guard for user=%s chat=%s: reason=%s",
            sender_id,
            chat_id,
            reason,
        )
        return False, reason

    chat_lock: "threading.Lock" = get_chat_lock_fn()
    with chat_lock:
        # Double-check after acquiring lock
        if is_managed_fn():
            return True, ACTIVATION_ALLOWED
        try:
            synthetic_msg_id = f"passive-activate-{chat_id}"
            success = slock_handler.activate_slock(
                message_id=synthetic_msg_id,
                chat_id=chat_id,
                requirement=text,
                project=project,
                skip_guard_check=True,
            )
            return success, ACTIVATION_ALLOWED if success else "error"
        except Exception:
            logger.warning(
                "Failed to auto-activate slock for chat %s",
                chat_id,
                exc_info=True,
            )
            # Send user-friendly card notification
            try:
                from src.slock_engine.card_templates.common import (
                    build_error_state_card,
                )

                card = build_error_state_card(
                    title="任务暂时无法自动处理",
                    error_msg="你的消息暂时无法被自动分配，正在通过其他方式处理。请稍后重试，或直接描述你的需求让系统尝试其他方式处理。",
                )
                card_sender.send_card_to_chat(
                    chat_id,
                    json.dumps(card, ensure_ascii=False),
                )
            except Exception as card_err:
                logger.warning(
                    "Failed to send activation failure card to chat %s: %s",
                    chat_id,
                    card_err,
                )
            return False, "error"
