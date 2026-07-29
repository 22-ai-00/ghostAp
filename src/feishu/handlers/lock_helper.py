"""Lock-related helper extracted from BaseHandler (composition pattern).

``LockHelper`` encapsulates repo-lock lifecycle management, conflict-card
rendering, and chat-lock intercept card sending.  ``BaseHandler`` delegates
to an instance of this class via thin forwarding methods, keeping the handler
focused on message routing and rendering.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass as _dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

from ...card.ui_text import UI_TEXT

if TYPE_CHECKING:
    from ...repo_lock import LockConflictError

logger = logging.getLogger(__name__)


@runtime_checkable
class LockHandlerProtocol(Protocol):
    """Narrow interface that LockHelper depends on.

    BaseHandler implicitly satisfies this protocol.  Using a Protocol
    instead of importing BaseHandler eliminates the circular dependency
    and makes unit-test mocking straightforward.
    """

    @property
    def ctx(self) -> Any: ...  # HandlerContext

    @property
    def settings(self) -> Any: ...

    def reply_text(self, message_id: str, text: str, *, reply_in_thread: Optional[bool] = None) -> Optional[str]:
        """Reply plain text; returns the sent message_id or None on failure."""
        ...

    def reply_card(self, message_id: str, card_content: str, *, reply_in_thread: Optional[bool] = None) -> Optional[str]:
        """Reply interactive card (JSON string, CardKit v2 format); returns message_id or None on failure."""
        ...

    def add_reaction(self, message_id: str, emoji_type: str) -> None: ...


@_dataclass
class _LockConflictContext:
    """Contextual info for building a repo-lock conflict card."""
    is_admin: bool = False
    is_same_sender: bool = False
    app_id: str = ""
    repo_token: str = ""
    sender_id: str = ""


class LockHelper:
    """Repo-lock and chat-lock helper — composed into every BaseHandler."""

    def __init__(self, handler: LockHandlerProtocol) -> None:
        self._h = handler

    @staticmethod
    def resolve_git_lock_root(path: str | None) -> str | None:
        """Return the nearest Git worktree root, preserving non-Git paths."""
        if not path:
            return path
        try:
            result = subprocess.run(
                ["git", "-C", path, "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            root = (result.stdout or "").strip()
            if result.returncode != 0 or not root:
                return LockHelper._find_git_marker_root(path)
            return str(Path(root).resolve(strict=True))
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return LockHelper._find_git_marker_root(path)

    @staticmethod
    def _find_git_marker_root(path: str) -> str:
        """Find the nearest on-disk ``.git`` marker without invoking Git."""
        try:
            current = Path(path).resolve(strict=True)
            if not current.is_dir():
                return path
            for candidate in (current, *current.parents):
                marker = candidate / ".git"
                if LockHelper._is_git_marker(marker, candidate):
                    return str(candidate)
        except OSError:
            return path
        return path

    @staticmethod
    def _is_git_marker(marker: Path, worktree_root: Path) -> bool:
        """Return whether *marker* names valid Git metadata for a worktree."""
        if marker.is_dir():
            return (marker / "HEAD").is_file()
        if not marker.is_file() or marker.stat().st_size > 4096:
            return False
        try:
            first_line = marker.read_text(encoding="utf-8").partition("\n")[0]
        except UnicodeError:
            return False
        prefix = "gitdir:"
        if not first_line.casefold().startswith(prefix):
            return False
        raw_git_dir = first_line[len(prefix):].strip()
        if not raw_git_dir:
            return False
        git_dir = Path(raw_git_dir)
        if not git_dir.is_absolute():
            git_dir = worktree_root / git_dir
        return (git_dir.resolve(strict=True) / "HEAD").is_file()

    # ------------------------------------------------------------------
    # Repo lock: with-lock helper (Event + daemon thread heartbeat)
    # ------------------------------------------------------------------

    def _with_repo_lock(self, root_path: str, chat_id: str, body_func):
        """Execute *body_func* while holding the repo lock for *root_path*.

        Raises ``LockConflictError`` if the lock cannot be acquired.

        A daemon heartbeat thread refreshes :meth:`touch` every 30 s so that
        long-running engine tasks are not evicted by the idle/hard timeout.

        When ``repo_lock_manager`` is not configured or *root_path* is falsy,
        *body_func* is executed without locking.
        """
        from ...thread import get_current_is_p2p

        return self._with_repo_lock_mode(
            root_path,
            chat_id,
            body_func,
            is_p2p=get_current_is_p2p(),
        )

    def _with_repo_lock_strict(self, root_path: str, chat_id: str, body_func):
        """Acquire a real repo lock even when the request came from P2P chat.

        Shell commands can restart the whole service or mutate the same working
        tree as an active programming task. They therefore must not inherit the
        general P2P lock-bypass privilege.
        """
        return self._with_repo_lock_mode(
            root_path,
            chat_id,
            body_func,
            is_p2p=False,
        )

    def _with_repo_lock_mode(
        self,
        root_path: str,
        chat_id: str,
        body_func,
        *,
        is_p2p: bool,
    ):
        """Shared lock/heartbeat implementation for bypass and strict modes."""

        repo_lock_mgr = getattr(self._h.ctx, "repo_lock_manager", None)
        if not repo_lock_mgr or not root_path:
            return body_func()

        from ...repo_lock import get_repo_lock_heartbeat_interval
        from ...tasking import get_current_task_run_id
        from ...utils.heartbeat import RepoLockHeartbeat

        _TOUCH_INTERVAL = get_repo_lock_heartbeat_interval(repo_lock_mgr)
        owner_id = get_current_task_run_id()
        stop_event = threading.Event()

        with repo_lock_mgr.hold(
            root_path,
            chat_id,
            is_p2p=is_p2p,
            owner_id=owner_id,
        ):
            hb = RepoLockHeartbeat(
                stop_event,
                lambda: repo_lock_mgr.touch(
                    root_path,
                    chat_id,
                    owner_id=owner_id,
                ),
                interval=_TOUCH_INTERVAL,
                name=f"lock-helper-{root_path}",
            )
            hb.start()
            try:
                return body_func()
            finally:
                stop_event.set()
                hb.join(timeout=2)

    # ------------------------------------------------------------------
    # Repo lock: explicit acquire / release
    # ------------------------------------------------------------------

    def _acquire_repo_lock(self, root_path: str | None, chat_id: str):
        """Explicitly acquire the repo lock (for long-held streaming scenarios).

        Returns ``(AcquireResult | None, repo_lock_mgr, needs_release)``.
        """
        from ...repo_lock import LockConflictError
        from ...tasking import get_current_task_run_id
        from ...thread import get_current_is_p2p

        repo_lock_mgr = getattr(self._h.ctx, "repo_lock_manager", None)
        if not repo_lock_mgr or not root_path:
            return None, None, False

        is_p2p = get_current_is_p2p()
        if is_p2p:
            return None, None, False

        result = repo_lock_mgr.acquire(
            root_path,
            chat_id,
            is_p2p=False,
            owner_id=get_current_task_run_id(),
        )
        if not result.success:
            raise LockConflictError(
                f"Repo lock conflict for {root_path!r} (held by another chat)",
                holder_chat_id=result.holder_chat_id or "",
                locked_since=result.locked_since or 0.0,
                root_path=root_path,
                last_active_time=result.last_active_time or 0.0,
            )
        return result, repo_lock_mgr, True

    def _release_repo_lock(self, root_path: str | None, chat_id: str, repo_lock_mgr=None) -> None:
        """Release a repo lock previously acquired via :meth:`_acquire_repo_lock`."""
        if repo_lock_mgr and root_path:
            from ...tasking import get_current_task_run_id

            repo_lock_mgr.release(
                root_path,
                chat_id,
                owner_id=get_current_task_run_id(),
            )

    # ------------------------------------------------------------------
    # Repo lock: single entry point for lock-guarded execution
    # ------------------------------------------------------------------

    def handle_lock_conflict(
        self,
        body_fn,
        root_path: str,
        chat_id: str,
        message_id: str,
        command_text: str,
        *,
        retry_count: int = 0,
    ):
        """Execute *body_fn* under the repo lock; send conflict card on failure.

        Combines :meth:`_with_repo_lock` + ``LockConflictError`` catch +
        :meth:`send_lock_conflict_card` into a single call.  Returns the
        result of *body_fn* on success, or ``None`` if a conflict occurred.
        """
        from ...repo_lock import LockConflictError

        try:
            return self._with_repo_lock(root_path, chat_id, body_fn)
        except LockConflictError as e:
            self.send_lock_conflict_card(
                e,
                message_id,
                command_text,
                retry_count=retry_count,
                chat_id=chat_id,
            )
            return None

    # ------------------------------------------------------------------
    # Lock conflict card
    # ------------------------------------------------------------------

    def send_lock_conflict_card(
        self, e: "LockConflictError", message_id: str, command_text: str, *, retry_count: int = 0, chat_id: str = "",
    ) -> None:
        """Build and send a repo-lock conflict card."""
        try:
            from ...card.builders.lock import build_repo_lock_card
            from ...card.builders.project import ProjectBuilder

            lctx = self._collect_lock_conflict_context(e, chat_id=chat_id)

            lock_content, lock_buttons = build_repo_lock_card(
                e.root_path, e.locked_since, is_admin=lctx.is_admin,
                command_text=command_text,
                repo_token=lctx.repo_token,
                last_active_time_monotonic=e.last_active_time,
                app_id=lctx.app_id,
                is_same_sender=lctx.is_same_sender,
                retry_count=retry_count,
                idle_timeout_seconds=getattr(self._h.settings, "repo_lock_idle_timeout", None),
                chat_id=chat_id,
            )
            msg_type, card_json = ProjectBuilder.build_project_response_card(
                project=None,
                title=UI_TEXT["repo_lock_card_header"],
                content=lock_content,
                show_buttons=False,
                extra_buttons=lock_buttons,
            )
            self._h.reply_card(message_id, card_json)
        except Exception as card_err:
            logger.error("Failed to send lock conflict card: %s", card_err)
            try:
                self._h.reply_text(message_id, UI_TEXT["repo_lock_conflict_fallback"])
            except Exception as fallback_err:
                logger.warning("Fallback text reply also failed: %s", fallback_err)

    def _collect_lock_conflict_context(self, e: "LockConflictError", *, chat_id: str = ""):
        """Collect contextual info needed to build a lock conflict card."""
        from ...config import get_settings
        from ...thread import get_current_sender_id

        ctx = _LockConflictContext()

        _chat_lock_mgr = getattr(self._h.ctx, "chat_lock_manager", None)
        ctx.sender_id = get_current_sender_id() or ""
        ctx.is_admin = _chat_lock_mgr.is_admin(ctx.sender_id) if _chat_lock_mgr else False

        _repo_lock_mgr = getattr(self._h.ctx, "repo_lock_manager", None)
        ctx.repo_token = _repo_lock_mgr.path_to_token(e.root_path) if _repo_lock_mgr else ""

        if ctx.sender_id and _repo_lock_mgr:
            _lock_info = _repo_lock_mgr.get_lock_info(e.root_path)
            if _lock_info and getattr(_lock_info, "last_sender_id", "") == ctx.sender_id:
                ctx.is_same_sender = True

        try:
            ctx.app_id = get_settings().app_id or ""
        except Exception:
            logger.debug("Failed to get app_id for lock conflict context", exc_info=True)

        return ctx

    # ------------------------------------------------------------------
    # Chat-lock intercept and throttled reply
    # ------------------------------------------------------------------

    def send_chat_lock_intercept_card(self, message_id: str, chat_id: str, chat_lock_manager) -> None:
        """Build and send a chat-lock intercept card with interactive buttons."""
        try:
            from ...card.builders.lock import build_chat_lock_card
            from ...card.builders.project import ProjectBuilder
            from ..emoji import EmojiReaction

            _lock_entry = chat_lock_manager.get_lock_info(chat_id)
            _locked_by = _lock_entry.locked_by if _lock_entry else None
            _locked_name = _lock_entry.locked_by_name if _lock_entry else ""

            _app_id = ""
            try:
                _app_id = self._h.settings.app_id or ""
            except Exception:
                logger.debug("Failed to get app_id for lock card", exc_info=True)

            _card_md, _card_buttons = build_chat_lock_card(
                locked_by=_locked_by, locked_by_name=_locked_name,
                admin_name=_locked_name or "",
                app_id=_app_id,
                locked_at_wall=_lock_entry.locked_at_wall if _lock_entry else None,
                max_duration_seconds=getattr(self._h.settings, "chat_lock_max_duration", None),
            )
            msg_type, card_json = ProjectBuilder.build_project_response_card(
                project=None,
                title=UI_TEXT["chat_locked_title"],
                content=_card_md,
                show_buttons=False,
                extra_buttons=_card_buttons,
            )
            self._h.reply_card(message_id, card_json)
            try:
                self._h.add_reaction(message_id, EmojiReaction.on_chat_locked())
            except Exception:
                logger.debug("Failed to add chat-lock emoji reaction", exc_info=True)
        except Exception as card_err:
            logger.error("Failed to send chat lock intercept card: %s", card_err)
            try:
                self._h.reply_text(message_id, UI_TEXT["chat_locked_fallback"])
            except Exception as fallback_err:
                logger.warning("Fallback text reply for chat lock also failed: %s", fallback_err)

    def send_chat_lock_throttled_reply(self, message_id: str, chat_id: str, chat_lock_manager) -> None:
        """Send a throttled (dedup) chat-lock reply — emoji + card with /status button."""
        from ..emoji import EmojiReaction

        _lock_entry = chat_lock_manager.get_lock_info(chat_id)
        _locked_name = _lock_entry.locked_by_name if _lock_entry else ""
        _name = _locked_name or UI_TEXT.get("ws_fallback_admin_name", "Bot 管理员")

        try:
            self._h.add_reaction(message_id, EmojiReaction.on_chat_locked())
        except Exception:
            logger.debug("Failed to add throttled chat-lock emoji reaction", exc_info=True)
        try:
            from ...card.builders.lock import _build_p2p_multi_url, _compute_command_sig
            from ...card.builders.project import ProjectBuilder
            _md = UI_TEXT["chat_locked_throttled_reply_md"].format(name=_name)
            _buttons = [{
                "tag": "button",
                "text": {"tag": "plain_text", "content": UI_TEXT["chat_lock_btn_status"]},
                "type": "default",
                "value": {"action": "retry_command", "_t": "/status", "_s": _compute_command_sig("/status")},
            }]
            _app_id = ""
            try:
                _app_id = self._h.settings.app_id or ""
            except Exception:
                logger.debug("Failed to get app_id for throttled reply", exc_info=True)
            if _app_id:
                _buttons.append({
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": UI_TEXT["chat_lock_btn_go_p2p"]},
                    "type": "default",
                    "multi_url": _build_p2p_multi_url(_app_id),
                })
            msg_type, card = ProjectBuilder.build_project_response_card(
                project=None, title=UI_TEXT["chat_locked_title"],
                content=_md, show_buttons=False, extra_buttons=_buttons,
            )
            self._h.reply_card(message_id, card)
        except Exception:
            try:
                self._h.reply_text(
                    message_id,
                    UI_TEXT["chat_locked_throttled_reply"].format(name=_name),
                )
            except Exception:
                logger.debug("Throttled reply fallback also failed", exc_info=True)
