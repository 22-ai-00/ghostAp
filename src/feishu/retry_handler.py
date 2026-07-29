"""Retry-command card action handler (extracted from action_registry.py).

``RetryCommandHandler`` encapsulates the ~100-line ``_handle_retry_command``
closure that was previously defined inline in :func:`register_programming_mode_actions`.
Each responsibility is now a separate, testable method:

- ``_verify_signature`` — HMAC signature validation
- ``_check_chat_lock`` — chat-level lock guard
- ``_resolve_project`` — project lookup + fallback + mismatch check
- ``_probe_repo_lock`` — probe-acquire-then-release with conflict card
- ``_dispatch_intent`` — forward to ``process_with_intent`` with error wrapping

``RetryDispatchProtocol`` defines a narrow interface (similar to the existing
``LockHandlerProtocol``) so that ``RetryCommandHandler`` depends only on the
methods it actually needs, rather than coupling to the full ``FeishuWSClient``
internal layout.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..project.context import ProjectContext
    from ..repo_lock import LockConflictError, RepoLockManager

logger = logging.getLogger(__name__)

# Commands that skip HMAC signature verification (read-only / safe commands)
SIGNATURE_EXEMPT_COMMANDS: frozenset[str] = frozenset({"/status"})


@runtime_checkable
class RetryDispatchProtocol(Protocol):
    """Narrow interface that RetryCommandHandler depends on.

    An adapter (``_RetryDispatchAdapter`` in ``action_registry.py``) bridges
    the concrete ``FeishuWSClient`` to this protocol.  Using a Protocol
    eliminates direct access to 5+ private attributes of ``ws_client`` and
    makes unit-test mocking straightforward.
    """

    def reply_text(self, message_id: str, text: str) -> None: ...

    def try_block_with_chat_lock(self, chat_id: str, sender_id: str, message_id: str, *, raw_text: str = "") -> bool: ...

    def get_project_for_chat(self, project_id: str, chat_id: str) -> Optional[ProjectContext]: ...

    def get_active_project(self, chat_id: str) -> Optional[ProjectContext]: ...

    def get_repo_lock_manager(self) -> Optional[RepoLockManager]: ...

    def process_with_intent(self, message_id: str, chat_id: str, text: str, project: Optional[ProjectContext]) -> None: ...

    def send_lock_conflict_card(
        self,
        e: LockConflictError,
        message_id: str,
        command_text: str,
        *,
        retry_count: int = 0,
        chat_id: str = "",
    ) -> None: ...


class RetryCommandHandler:
    """Callable action handler for the ``retry_command`` card action.

    Instantiated once during :func:`init_action_registry` and registered
    via ``client._register_action(handler, exact="retry_command")``.
    The ``__call__`` signature matches the standard action handler protocol:
    ``(message_id, chat_id, project_id, value_dict)``.
    """

    def __init__(self, dispatch: RetryDispatchProtocol) -> None:
        self._dispatch = dispatch

    @staticmethod
    def _compat_get(val: dict[str, Any], new_key: str, old_key: str, default: Any = "") -> Any:
        """Read a card action value field with backward-compatible fallback.

        During the field-rename compatibility window, buttons produced by the
        old code carry *old_key* while new code uses *new_key*.  Prefer the
        new key; fall back to the old key transparently.
        """
        v = val.get(new_key)
        if v is not None:
            return v
        return val.get(old_key, default)

    # ------------------------------------------------------------------
    # Public entry point (action handler protocol)
    # ------------------------------------------------------------------

    def __call__(self, mid: str, cid: str, pid: Optional[str], val: dict[str, Any]) -> None:
        cmd = self._compat_get(val, "_t", "command_text").strip()
        if not cmd:
            from ..card.ui_text import UI_TEXT
            self._dispatch.reply_text(mid, UI_TEXT["retry_empty_command"])
            return

        sig = self._compat_get(val, "_s", "command_sig")

        # 1. Signature verification (pass chat_id for v2 sig binding)
        if not self._verify_signature(mid, cmd, sig, cid):
            return

        # 1b. Undo-lock expiry check
        if self._compat_get(val, "_ul", "undo_lock", default=False):
            import time as _t
            undo_expires = self._compat_get(val, "_ue", "undo_expires", default=0)
            if undo_expires and _t.time() > undo_expires:
                from ..card.ui_text import UI_TEXT
                self._dispatch.reply_text(mid, UI_TEXT["lock_undo_expired"])
                return

        # 2. Chat lock check
        if self._check_chat_lock(cid, mid, cmd):
            return

        # 3. Project resolution
        project = self._resolve_project(mid, cid, pid)
        if project is _REJECTED:
            return

        # 4. Repo lock probe
        retry_count = int(self._compat_get(val, "_rc", "retry_count", default=0))
        if self._probe_repo_lock(mid, cid, cmd, project, retry_count):
            return

        # 5. Dispatch intent
        self._dispatch_intent(mid, cid, cmd, project, retry_count)

    # ------------------------------------------------------------------
    # Step methods (each independently testable)
    # ------------------------------------------------------------------

    def _verify_signature(self, mid: str, cmd: str, sig: str, chat_id: str = "") -> bool:
        """Return True when signature is valid; reply with expiry message and
        return False otherwise.  Commands in SIGNATURE_EXEMPT_COMMANDS bypass
        verification entirely."""
        if cmd.strip() in SIGNATURE_EXEMPT_COMMANDS:
            return True
        if not sig:
            from ..card.ui_text import UI_TEXT
            self._dispatch.reply_text(mid, UI_TEXT["retry_command_sig_mismatch"])
            return False
        from ..card.builders.lock import verify_command_sig
        result = verify_command_sig(cmd, sig, chat_id=chat_id)
        if result:
            return True
        from ..card.ui_text import UI_TEXT
        from ..utils.signing import VerifyResult
        if result is VerifyResult.COMPAT_EXPIRED:
            self._dispatch.reply_text(mid, UI_TEXT["retry_command_sig_upgrade_expired"])
        else:
            self._dispatch.reply_text(mid, UI_TEXT["retry_command_sig_mismatch"])
        return False

    def _check_chat_lock(self, cid: str, mid: str, cmd: str) -> bool:
        """Return True when the chat is locked and the message was blocked."""
        from ..thread import get_current_sender_id
        sender = get_current_sender_id() or ""
        return bool(self._dispatch.try_block_with_chat_lock(cid, sender, mid, raw_text=cmd))

    def _resolve_project(self, mid: str, cid: str, pid: Optional[str]):
        """Resolve the project context for this retry.

        Returns the ``ProjectContext`` on success, ``None`` when no project is
        found (which is acceptable — downstream handles it), or the sentinel
        ``_REJECTED`` when the fallback project doesn't match the requested pid.
        """
        project = self._dispatch.get_project_for_chat(pid, cid) if pid else None
        if not project:
            project = self._dispatch.get_active_project(cid)
        # F-09: Reject when fallback project differs from requested project
        if pid and project and project.project_id != pid:
            from ..card.ui_text import UI_TEXT
            self._dispatch.reply_text(mid, UI_TEXT["retry_project_unavailable"])
            return _REJECTED
        return project

    def _probe_repo_lock(
        self, mid: str, cid: str, cmd: str, project: Any, retry_count: int
    ) -> bool:
        """Probe-acquire-then-release the repo lock.

        Returns True when the lock is held by another chat (caller should
        abort).  The conflict card is sent via ``dispatch.send_lock_conflict_card``.

        Design decision (DELIBERATE TOCTOU): we probe-acquire then immediately
        release, accepting a race window before the real acquire inside
        ``process_with_intent``.  See ``docs/adr-lock-ordering.md``.
        """
        repo_lock_mgr = self._dispatch.get_repo_lock_manager()
        if not repo_lock_mgr or not project or not getattr(project, "root_path", None):
            return False

        from ..tasking import get_current_task_run_id

        owner_id = get_current_task_run_id()
        probe = repo_lock_mgr.acquire(
            project.root_path,
            cid,
            owner_id=owner_id,
        )
        if not probe.success:
            self._send_probe_conflict_card(
                mid,
                cid,
                cmd,
                project,
                probe,
                retry_count,
            )
            return True

        # Probe succeeded — release immediately; real lock acquired downstream.
        repo_lock_mgr.release(
            project.root_path,
            cid,
            owner_id=owner_id,
        )
        return False

    def _dispatch_intent(
        self, mid: str, cid: str, cmd: str, project: Any, retry_count: int
    ) -> None:
        """Forward to ``process_with_intent``, catching ``LockConflictError``."""
        try:
            self._dispatch.process_with_intent(mid, cid, cmd, project)
        except Exception as exc:
            from ..repo_lock import LockConflictError
            if isinstance(exc, LockConflictError):
                self._handle_lock_conflict(
                    exc,
                    mid,
                    cid,
                    cmd,
                    retry_count,
                )
            else:
                raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_probe_conflict_card(
        self,
        mid: str,
        cid: str,
        cmd: str,
        project: Any,
        probe: Any,
        retry_count: int,
    ) -> None:
        """Send a lock conflict card after a failed probe-acquire."""
        from ..repo_lock import LockConflictError
        err = LockConflictError(
            "Repo lock still held",
            holder_chat_id=probe.holder_chat_id or "",
            locked_since=probe.locked_since or 0.0,
            root_path=project.root_path,
            last_active_time=probe.last_active_time or 0.0,
        )
        self._dispatch.send_lock_conflict_card(
            err,
            mid,
            cmd,
            retry_count=retry_count,
            chat_id=cid,
        )

    def _handle_lock_conflict(
        self,
        exc: Any,
        mid: str,
        cid: str,
        cmd: str,
        retry_count: int,
    ) -> None:
        """Handle a ``LockConflictError`` raised during dispatch."""
        self._dispatch.send_lock_conflict_card(
            exc,
            mid,
            cmd,
            retry_count=retry_count,
            chat_id=cid,
        )


# Sentinel object indicating that _resolve_project rejected the request.
_REJECTED = object()
