"""Chat lock gate — ingress-level interception middleware.

Extracted from ``FeishuWSClient`` to decouple chat-lock interception logic
from the main WebSocket client class.  ``ws_client.py`` calls
``gate.check()`` / ``gate.check_card_action()`` at the message/card-action
entry points — a single-line decision replacing ~110 lines of inline code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

from ..card.ui_text import UI_TEXT
from .slash_command_parser import CommandMatch

if TYPE_CHECKING:
    from ..chat_lock import ChatLockManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol: narrow interface injected by ws_client
# ---------------------------------------------------------------------------

@runtime_checkable
class GateHandler(Protocol):
    """Minimal SystemHandler surface used by the ingress lock gate."""

    def reply_text(self, message_id: str, text: str) -> Optional[str]: ...

    def add_reaction(self, message_id: str, emoji_type: str) -> None: ...

    def send_chat_lock_intercept_card(
        self, message_id: str, chat_id: str, manager: Any,
    ) -> None: ...

    def send_chat_lock_throttled_reply(
        self, message_id: str, chat_id: str, manager: Any,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CheckResult:
    """Outcome of a gate check."""
    blocked: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# ChatLockGate
# ---------------------------------------------------------------------------

class ChatLockGate:
    """Ingress gate for chat-lock interception.

    Parameters
    ----------
    chat_lock_manager:
        The ``ChatLockManager`` singleton (may be ``None`` if not configured).
    dedup_cache:
        A ``MessageCache`` instance for deduplicating lock intercept cards
        within a 30-second window per (chat, sender).
    handler:
        System handler responsible for lock cards and fallback feedback.
    """

    def __init__(
        self,
        chat_lock_manager: Optional[ChatLockManager],
        dedup_cache: Any,  # MessageCache
        handler: GateHandler,
    ) -> None:
        self._clm = chat_lock_manager
        self._dedup = dedup_cache
        self._handler = handler

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        chat_id: str,
        sender_id: str,
        message_id: str,
        *,
        command_match: CommandMatch | None = None,
    ) -> bool:
        """Check whether a *message* should be blocked by the chat lock.

        Returns ``True`` if blocked (caller should ``return``).
        Uses fail-close semantics: exceptions → non-admin in locked chat blocked.
        """
        # Single source of truth: CommandMatch (or None for non-slash messages)
        _match = command_match

        return self._fail_close_check(
            chat_id, sender_id, message_id,
            command_match=_match,
            is_card_action=False, action_type="",
        )

    def check_card_action(
        self,
        chat_id: str,
        sender_id: str,
        message_id: str,
        *,
        action_type: str = "",
    ) -> bool:
        """Check whether a *card action* should be blocked by the chat lock.

        Returns ``True`` if blocked.
        """
        return self._fail_close_check(
            chat_id, sender_id, message_id,
            is_card_action=True, action_type=action_type,
        )

    def close(self) -> None:
        """Stop the dedup cache cleanup thread."""
        try:
            self._dedup.stop_cleanup_thread()
        except Exception:
            logger.debug("failed to stop dedup cleanup thread", exc_info=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_send_intercept(self, chat_id: str, sender_id: str) -> bool:
        """Return True if a lock intercept card should be sent (dedup)."""
        key = f"lock_block:{chat_id}:{sender_id}"
        return not self._dedup.is_duplicate(key)

    def _fail_close_check(
        self,
        chat_id: str,
        sender_id: str,
        message_id: str,
        *,
        command_match: CommandMatch | None = None,
        is_card_action: bool = False,
        action_type: str = "",
    ) -> bool:
        """Fail-close chat-lock interception (shared by message + card paths)."""
        clm = self._clm
        if clm is None:
            return False

        _is_admin = clm.is_admin(sender_id)

        try:
            if self._try_block(
                chat_id, sender_id, message_id,
                command_match=command_match,
                is_card_action=is_card_action, action_type=action_type,
            ):
                return True
        except Exception as exc:
            logger.warning("ChatLock interception error (fail-close): %s", str(exc))
            if not _is_admin and clm.is_locked(chat_id):
                if not is_card_action:
                    if self._should_send_intercept(chat_id, sender_id):
                        try:
                            self._handler.send_chat_lock_intercept_card(message_id, chat_id, clm)
                        except Exception:
                            self._handler.reply_text(message_id, UI_TEXT["chat_locked_fallback"])
                return True

        return False

    def _try_block(
        self,
        chat_id: str,
        sender_id: str,
        message_id: str,
        *,
        command_match: CommandMatch | None = None,
        is_card_action: bool = False,
        action_type: str = "",
    ) -> bool:
        """Core blocking decision + card dispatch."""
        clm = self._clm
        if clm is None:
            return False

        if is_card_action:
            blocked = clm.should_block_card_action(chat_id, sender_id, action_type)
        else:
            blocked = clm.should_block(chat_id, sender_id, command_match=command_match)

        if not blocked:
            return False

        if self._should_send_intercept(chat_id, sender_id):
            try:
                self._handler.send_chat_lock_intercept_card(message_id, chat_id, clm)
            except Exception:
                if is_card_action:
                    raise
                logger.warning(
                    "Chat lock intercept card delivery failed; using text fallback",
                    exc_info=True,
                )
                self._handler.reply_text(message_id, UI_TEXT["chat_locked_fallback"])
        else:
            self._handler.send_chat_lock_throttled_reply(message_id, chat_id, clm)
        return True
