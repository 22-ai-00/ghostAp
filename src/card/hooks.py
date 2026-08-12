"""CardSession lifecycle hooks.

Provides the SessionHook protocol and concrete implementations for
Feishu-specific side effects (emoji reactions, context persistence).

Hooks are injected at CardSession construction time and called at:
- on_dispatched(event, state): after reduce, before deliver
- on_terminal(state, reason): after terminal event delivery succeeds

All hook callbacks are wrapped in try/except by the caller (CardSession),
so a failing hook never blocks the dispatch pipeline.
Terminal hooks have a 5s timeout to prevent network hangs from blocking the pipeline.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
import weakref
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from src.card.engine_meta import engine_type_to_cmd
from src.card.events import CardEvent
from src.card.nav_link import format_task_continuation_link
from src.card.state.models import CardState, TerminalReason
from src.card.ui_text import UI_TEXT

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Timeout for terminal hooks (network-calling hooks like emoji/persistence)
HOOK_TIMEOUT_SECONDS = 5.0

# Timeout for dispatched hooks (lightweight, sync by nature)
DISPATCHED_HOOK_TIMEOUT = 3.0

# Threshold: rebuild executor after this many consecutive timeouts
_MAX_CONSECUTIVE_TIMEOUTS = 2

_dispatched_hook_context = threading.local()


class _HookExecutorManager:
    """Manages the hook thread pool with lazy rebuild on consecutive timeouts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._executor = self._create_executor()
        self._consecutive_timeouts = 0
        self._capacity = self._get_max_workers() * 4
        self._semaphore = threading.BoundedSemaphore(self._capacity)

    @staticmethod
    def _get_max_workers() -> int:
        # Default should cover short bursts of terminal hooks (e.g. many sessions
        # completing concurrently) while keeping the pool small in production.
        max_workers = 6
        try:
            from src.config import get_settings
            val = getattr(get_settings(), "hook_pool_max_workers", None)
            if isinstance(val, int) and val > 0:
                max_workers = val
        except Exception:
            pass
        return max_workers

    @staticmethod
    def _create_executor() -> concurrent.futures.ThreadPoolExecutor:
        max_workers = _HookExecutorManager._get_max_workers()
        return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hook")

    def submit(self, fn, *args):
        """Submit a callable to the executor with backpressure protection.

        If all worker slots are occupied, degrades to fire-and-forget
        (skips submission) to protect the dispatch hot path.
        """
        if not self._semaphore.acquire(blocking=False):
            logger.warning(
                "Hook executor: backpressure triggered, skipping hook submission (all %d slots occupied)",
                self._capacity,
            )
            return None
        with self._lock:
            try:
                future = self._executor.submit(self._wrap_with_semaphore_release(fn), *args)
                return future
            except Exception:
                self._semaphore.release()
                raise


    def _wrap_with_semaphore_release(self, fn):
        """Wrap fn so semaphore is released after completion."""
        def wrapper(*args):
            try:
                return fn(*args)
            finally:
                self._semaphore.release()
        return wrapper

    def record_success(self) -> None:
        """Reset consecutive timeout counter on success."""
        with self._lock:
            self._consecutive_timeouts = 0

    def record_timeout(self) -> None:
        """Record a timeout and rebuild executor if threshold reached."""
        old_executor = None
        with self._lock:
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts >= _MAX_CONSECUTIVE_TIMEOUTS:
                logger.warning("Hook executor: %d consecutive timeouts, rebuilding thread pool", self._consecutive_timeouts)
                old_executor = self._executor
                self._executor = self._create_executor()
                self._consecutive_timeouts = 0
        # Shutdown old executor outside lock (non-blocking)
        if old_executor is not None:
            try:
                old_executor.shutdown(wait=False)
            except Exception:
                pass

    def shutdown(self) -> None:
        """Gracefully shut down the executor thread pool with timeout protection.

        Uses shutdown(wait=False, cancel_futures=True) then joins threads with
        an explicit timeout to prevent hanging the process exit.

        Called during process shutdown to ensure clean thread cleanup.
        """
        _SHUTDOWN_TIMEOUT = 10.0
        with self._lock:
            executor = self._executor
        # Signal shutdown and cancel pending futures immediately
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            logger.warning("Hook executor shutdown signal failed: %s", repr(exc))
            return
        # Join worker threads with timeout to avoid indefinite blocking
        for t in getattr(executor, "_threads", set()):
            try:
                t.join(timeout=_SHUTDOWN_TIMEOUT)
            except Exception:
                pass


_hook_executor_manager = _HookExecutorManager()


def shutdown_hook_executor() -> None:
    """Shut down the global hook executor (called during graceful shutdown)."""
    _hook_executor_manager.shutdown()


def _reset_hook_executor() -> None:
    """Reset the global hook executor manager (test isolation only).

    Rebuilds the executor and resets the consecutive timeout counter.
    Should be called via a pytest fixture to prevent cross-test state leakage.
    """
    global _hook_executor_manager
    _hook_executor_manager.shutdown()
    _hook_executor_manager = _HookExecutorManager()


@runtime_checkable
class SessionHook(Protocol):
    """Protocol for CardSession lifecycle hooks.

    Implementations may define either or both methods. Missing methods
    are treated as no-ops by the CardSession.
    """

    def on_dispatched(self, event: CardEvent, state: CardState) -> None:
        """Called after reduce+render, before deliver.

        Args:
            event: The event that was just dispatched.
            state: The new state after reduce.
        """
        ...

    def on_terminal(self, state: CardState, reason: TerminalReason) -> None:
        """Called after a terminal event (COMPLETED/FAILED/CANCELLED) is delivered.

        Args:
            state: The final card state.
            reason: Terminal reason (see TerminalReason Literal type).
        """
        ...

    def on_first_delivered(self, session_id: str, msg_id: str) -> None:
        """Called once after the first successful delivery, with the message_id.

        Invoked outside the session lock to avoid lock nesting risks.
        Used by orchestrator for deep-link backfill on archived cards.

        Args:
            session_id: The session that was delivered.
            msg_id: The Feishu message_id of the first delivery.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete hook implementations
# ---------------------------------------------------------------------------


class EmojiHook:
    """Adds emoji reactions on terminal events.

    - COMPLETED → success emoji (e.g. PARTY)
    - FAILED → error emoji (e.g. SOB)
    - CANCELLED / TTL_EXPIRED → stop emoji (⏹)

    Args:
        add_reaction: Callable(message_id, emoji_type) to add a Feishu reaction.
        message_id: The message to react to.
        success_emoji: Emoji type string for success (default: PARTY).
        error_emoji: Emoji type string for failure (default: SOB).
        stop_emoji: Emoji type string for cancelled/ttl_expired (default: STOP).
    """

    SUCCESS_EMOJI_DEFAULT = "PARTY"
    ERROR_EMOJI_DEFAULT = "SOB"
    STOP_EMOJI_DEFAULT = "SKULL"

    def __init__(
        self,
        add_reaction: Callable[[str, str], Any],
        message_id: str,
        *,
        chat_id: str | None = None,
        success_emoji: str | None = None,
        error_emoji: str | None = None,
        stop_emoji: str | None = None,
    ) -> None:
        self._add_reaction = add_reaction
        self._message_id = message_id
        self._chat_id = chat_id
        self._success_emoji = self.SUCCESS_EMOJI_DEFAULT if success_emoji is None else success_emoji
        self._error_emoji = error_emoji or self.ERROR_EMOJI_DEFAULT
        self._stop_emoji = stop_emoji or self.STOP_EMOJI_DEFAULT

    def on_dispatched(self, event: CardEvent, state: CardState) -> None:
        """No-op for dispatched events."""

    def on_terminal(self, state: CardState, reason: TerminalReason) -> None:
        """Add emoji reaction based on terminal reason.

        Archived sessions (rotated out) get no emoji reaction.
        When message_id is empty, gracefully skip in degraded delivery mode.
        """
        if reason == "archived":
            return
        if not self._message_id:
            logger.debug("EmojiHook: skipping reaction (no message_id, chat_id=%s)", self._chat_id)
            return
        if reason in ("completed", "completed_empty"):
            if getattr(getattr(state, "metadata", None), "is_subagent", False) is True:
                logger.debug("EmojiHook: skipping success reaction for subagent card")
                return
            if not self._success_emoji:
                return
            self._add_reaction(self._message_id, self._success_emoji)
        elif reason in ("failed",):
            self._add_reaction(self._message_id, self._error_emoji)
        elif reason in ("cancelled", "ttl_expired"):
            self._add_reaction(self._message_id, self._stop_emoji)


class ContextPersistenceHook:
    """Persists engine results to project context on successful completion.

    On terminal(COMPLETED): calls update_context and optionally create_version.

    Args:
        update_fn: Callable to update the project context.
            Signature: update_fn(state) -> None
            The callable receives the final CardState and should extract
            whatever data it needs (this keeps the hook decoupled from
            specific context_manager APIs).
        notify_callback: Optional callable(chat_id, text) to notify user on failure.
            If provided, sends a text message when persistence fails.
        chat_id: Chat ID for notify_callback (required if notify_callback is set).
    """

    def __init__(
        self,
        update_fn: Callable[[CardState], None],
        notify_callback: Callable[[str, str], None] | None = None,
        chat_id: str | None = None,
        engine_type: str | None = None,
    ) -> None:
        self._update_fn = update_fn
        self._notify_callback = notify_callback
        self._chat_id = chat_id
        self._engine_type = engine_type

    def on_dispatched(self, event: CardEvent, state: CardState) -> None:
        """No-op for dispatched events."""

    def on_terminal(self, state: CardState, reason: TerminalReason) -> None:
        """Persist context on successful completion."""
        if reason in ("completed", "completed_empty"):
            try:
                self._update_fn(state)
            except Exception as exc:
                logger.warning("ContextPersistenceHook: persistence failed: %s", repr(exc))
                if self._notify_callback and self._chat_id:
                    try:
                        engine_cmd = engine_type_to_cmd(self._engine_type, fallback=UI_TEXT["card_session_fallback_cmd"])
                        self._notify_callback(
                            self._chat_id,
                            UI_TEXT["hook_persistence_failed_notice"].format(engine_cmd=engine_cmd),
                        )
                    except Exception as notify_exc:
                        logger.debug("ContextPersistenceHook: notify_callback failed: %s", repr(notify_exc))


# ---------------------------------------------------------------------------
# HookFirer: encapsulates hook dispatch logic extracted from CardSession
# ---------------------------------------------------------------------------


class HookFirer:
    """Fires hooks on behalf of CardSession.

    Extracted to keep CardSession focused on dispatch→reduce→render→deliver.
    Thread-safety: all methods are safe to call from any thread.
    """

    def __init__(
        self,
        hooks: tuple[SessionHook, ...],
        session_id: str,
        *,
        executor: _HookExecutorManager | None = None,
    ) -> None:
        self._hooks = hooks
        self._session_id = session_id
        self._executor = executor or _hook_executor_manager
        self._fired = threading.Event()  # ensures fire_terminal executes at most once
        self._fire_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._dispatched_condition = threading.Condition(threading.Lock())  # leaf lock: never held while acquiring a LockLevel lock
        self._dispatched_pending = 0
        self._dispatched_self_drainers = 0

    def append_hook(self, hook: SessionHook) -> None:
        """Append a hook after construction (thread-safe atomic tuple replacement).

        Used by orchestrator to inject backfill hooks on continuation sessions
        before they are exposed to concurrent dispatch.
        """
        with self._fire_lock:
            self._hooks = self._hooks + (hook,)

    @property
    def has_hooks(self) -> bool:
        return bool(self._hooks)

    def fire_dispatched(self, event: CardEvent, state: CardState | None) -> None:
        """Fire on_dispatched hooks (fire-and-forget).

        Hooks are submitted to the shared executor and return immediately —
        no synchronous wait on the dispatch hot path.  Exceptions are logged
        asynchronously via Future done-callbacks.  Hooks exceeding
        DISPATCHED_HOOK_TIMEOUT trigger executor health tracking.
        """
        if not self._hooks or state is None:
            return
        sid = self._session_id
        executor = self._executor
        submit_time = time.monotonic()
        for hook in self._hooks:
            fn = getattr(hook, "on_dispatched", None)
            if fn is None:
                continue
            with self._dispatched_condition:
                self._dispatched_pending += 1
            completion_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
            completion_done = [False]

            def _finish_once(
                _lock: threading.Lock = completion_lock,
                _done: list[bool] = completion_done,
            ) -> None:
                with _lock:
                    if _done[0]:
                        return
                    _done[0] = True
                self._finish_dispatched_hook()

            try:
                future = executor.submit(
                    self._run_dispatched_hook,
                    fn,
                    event,
                    state,
                    _finish_once,
                )
            except Exception:
                _finish_once()
                raise
            if future is None:
                _finish_once()
                continue
            hook_name = type(hook).__name__
            future.add_done_callback(
                lambda f, _finish=_finish_once: _finish()
            )
            future.add_done_callback(
                lambda f, _sid=sid, _name=hook_name, _t0=submit_time, _ex=executor: self._on_dispatched_done(f, _sid, _name, _t0, _ex)
            )

    def _run_dispatched_hook(
        self,
        fn: Callable[[CardEvent, CardState], None],
        event: CardEvent,
        state: CardState,
        finish: Callable[[], None],
    ) -> None:
        """Run one hook while marking this callback as self-drain exempt."""
        active = getattr(_dispatched_hook_context, "active_firers", None)
        if active is None:
            active = {}
            _dispatched_hook_context.active_firers = active
        firer_id = id(self)
        active[firer_id] = active.get(firer_id, 0) + 1
        try:
            fn(event, state)
        finally:
            remaining = active[firer_id] - 1
            if remaining:
                active[firer_id] = remaining
            else:
                active.pop(firer_id, None)
            finish()

    def _finish_dispatched_hook(self) -> None:
        with self._dispatched_condition:
            self._dispatched_pending -= 1
            self._dispatched_condition.notify_all()

    def drain_dispatched(self, timeout: float | None = None) -> bool:
        """Wait for pre-close dispatched hooks, excluding the calling hook.

        A hook may call ``CardSession.close()`` itself. Such a callback cannot
        wait for its own Future, but it still waits for every sibling hook from
        the same session. Multiple self-closing siblings are exempt together to
        avoid a cyclic wait.
        """
        active = getattr(_dispatched_hook_context, "active_firers", {})
        called_from_hook = active.get(id(self), 0) > 0
        timeout_s = DISPATCHED_HOOK_TIMEOUT if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._dispatched_condition:
            if called_from_hook:
                self._dispatched_self_drainers += 1
                self._dispatched_condition.notify_all()
            try:
                while True:
                    exempt = (
                        self._dispatched_self_drainers
                        if called_from_hook
                        else 0
                    )
                    if self._dispatched_pending <= exempt:
                        return True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._dispatched_condition.wait(timeout=remaining)
            finally:
                if called_from_hook:
                    self._dispatched_self_drainers -= 1
                    self._dispatched_condition.notify_all()

    @staticmethod
    def _on_dispatched_done(f: concurrent.futures.Future, sid: str, hook_name: str, submit_time: float, executor: _HookExecutorManager) -> None:
        """Done-callback for dispatched hooks: log errors and detect timeouts."""
        try:
            exc = f.exception()
        except concurrent.futures.CancelledError:
            return
        if exc is not None:
            logger.warning("HookFirer %s: on_dispatched failed (%s): %s", sid, hook_name, repr(exc))
        elapsed = time.monotonic() - submit_time
        if elapsed > DISPATCHED_HOOK_TIMEOUT:
            logger.warning(
                "HookFirer %s: on_dispatched slow (%s): %.2fs > %.1fs threshold",
                sid, hook_name, elapsed, DISPATCHED_HOOK_TIMEOUT,
            )
            executor.record_timeout()
        elif exc is None:
            executor.record_success()

    def fire_terminal(self, state: CardState | None, reason: str) -> None:
        """Fire on_terminal hooks in parallel with unified timeout.

        All hooks are submitted concurrently, then waited on with a single
        HOOK_TIMEOUT_SECONDS deadline. This bounds worst-case blocking to
        HOOK_TIMEOUT_SECONDS regardless of hook count (not N × timeout).

        Execution order is NOT guaranteed across hooks.
        Timeouts and exceptions are logged and swallowed.

        Exactly-once: subsequent calls are no-ops after the first successful invocation.
        """
        # Exactly-once guard: only the first caller proceeds
        with self._fire_lock:
            if self._fired.is_set():
                logger.debug("HookFirer %s: fire_terminal already fired, skipping", self._session_id)
                return
            self._fired.set()

        if not self._hooks or state is None:
            logger.debug("HookFirer %s: fire_terminal skipped (no hooks or state=None)", self._session_id)
            return

        # Submit all hooks in parallel
        future_to_hook: dict[concurrent.futures.Future, SessionHook] = {}
        for hook in self._hooks:
            fn = getattr(hook, "on_terminal", None)
            if fn is None:
                continue
            try:
                future = self._executor.submit(fn, state, reason)
                if future is not None:
                    future_to_hook[future] = hook
            except Exception as exc:
                logger.warning(
                    "HookFirer %s: on_terminal submit failed (%s): %s",
                    self._session_id, type(hook).__name__, repr(exc),
                )

        if not future_to_hook:
            return

        # Wait for all futures with a single timeout deadline
        done, not_done = concurrent.futures.wait(
            future_to_hook.keys(), timeout=HOOK_TIMEOUT_SECONDS
        )

        # Process completed futures
        has_timeout = False
        for future in done:
            hook = future_to_hook[future]
            try:
                future.result(timeout=0)  # Already done, just check for exceptions
            except Exception as exc:
                logger.warning(
                    "HookFirer %s: on_terminal failed (%s): %s",
                    self._session_id, type(hook).__name__, repr(exc),
                )

        # Handle timed-out futures
        for future in not_done:
            hook = future_to_hook[future]
            future.cancel()
            has_timeout = True
            logger.warning(
                "HookFirer %s: on_terminal timed out (%s) after %.1fs",
                self._session_id, type(hook).__name__, HOOK_TIMEOUT_SECONDS,
            )

        # Update executor health tracking
        if has_timeout:
            self._executor.record_timeout()
        elif done:
            self._executor.record_success()

    def fire_first_delivered(self, msg_id: str) -> None:
        """Fire on_first_delivered hooks with timeout protection.

        Hooks are submitted to the shared executor with a 3s deadline.
        Slow hooks are cancelled and logged at WARNING level.
        """
        if not self._hooks or not msg_id:
            return
        sid = self._session_id
        futures: list[concurrent.futures.Future] = []
        for hook in self._hooks:
            fn = getattr(hook, "on_first_delivered", None)
            if fn is None:
                continue
            fut = self._executor.submit(fn, sid, msg_id)
            if fut is not None:
                futures.append(fut)

        if not futures:
            return

        # Wait with timeout (same as DISPATCHED_HOOK_TIMEOUT)
        done, not_done = concurrent.futures.wait(futures, timeout=DISPATCHED_HOOK_TIMEOUT)
        for fut in not_done:
            fut.cancel()
        if not_done:
            logger.warning(
                "HookFirer %s: %d on_first_delivered hook(s) timed out (%.1fs)",
                sid, len(not_done), DISPATCHED_HOOK_TIMEOUT,
            )
            self._executor.record_timeout()
        elif done:
            self._executor.record_success()
        # Log exceptions from completed hooks
        for fut in done:
            exc = fut.exception()
            if exc:
                logger.debug(
                    "HookFirer %s: on_first_delivered hook failed: %s",
                    sid, repr(exc),
                )


# ---------------------------------------------------------------------------
# BackfillHook: patches old card with a deep-link after new card first delivers
# ---------------------------------------------------------------------------


class BackfillHook:
    """SessionHook that backfills a deep-link on the old (archived) card.

    When the new continuation session delivers its first card, this hook
    patches the old session's content with a navigation link pointing to
    the new message.

    Implements SessionHook.on_first_delivered protocol.
    """

    __slots__ = ("_old_session_ref", "_task_name", "_rotation_count")

    def __init__(
        self,
        old_session_ref: weakref.ref,
        task_name: str,
        rotation_count: int,
    ) -> None:
        self._old_session_ref = old_session_ref
        self._task_name = task_name
        self._rotation_count = rotation_count

    def on_dispatched(self, event, state) -> None:  # noqa: ARG002
        pass

    def on_terminal(self, state, reason) -> None:  # noqa: ARG002
        pass

    def on_first_delivered(self, session_id: str, msg_id: str) -> None:
        """Backfill the old card with a deep-link to the new message."""
        if not msg_id:
            return
        old_sess = self._old_session_ref()
        if old_sess is None:
            return
        if getattr(old_sess, "closed", False):
            return
        backfill_msg = format_task_continuation_link(
            task_name=self._task_name,
            rotation_count=self._rotation_count,
            new_msg_id=msg_id,
        )
        try:
            old_sess.dispatch(CardEvent.text_started("_continuation_backfill"))
            old_sess.dispatch(CardEvent.text_delta("_continuation_backfill", backfill_msg))
            old_sess.dispatch(CardEvent.text_done("_continuation_backfill"))
        except Exception:
            logger.debug("Deep-link backfill failed for task=%s", self._task_name)
