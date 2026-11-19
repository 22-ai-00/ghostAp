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
import types
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from src.card.engine_meta import engine_type_to_cmd
from src.card.events import CardEvent
from src.card.state.models import CardState, TerminalReason
from src.card.ui_text import UI_TEXT

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Timeout for terminal hooks (network-calling hooks like emoji/persistence)
HOOK_TIMEOUT_SECONDS = 5.0

# Timeout for dispatched hooks (lightweight, sync by nature)
DISPATCHED_HOOK_TIMEOUT = 3.0

# Slow hook warning threshold in milliseconds
SLOW_HOOK_THRESHOLD_MS = 1000.0

# Threshold: rebuild executor after this many consecutive timeouts
_MAX_CONSECUTIVE_TIMEOUTS = 2

_PHASE_DISPATCHED = "on_dispatched"
_PHASE_TERMINAL = "on_terminal"

_dispatched_hook_context = threading.local()


def _sanitize_telemetry_token(
    value: object,
    *,
    default: str = "unknown",
    max_len: int = 64,
) -> str:
    """Return a bounded, log-safe identifier token without user content."""
    if value is None:
        return default
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return default
    stripped = value.strip()
    if not stripped:
        return default
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-.") else "_" for ch in stripped)
    return cleaned[:max_len] or default


def _format_event_type(event: object) -> str:
    """Extract a sanitized event_type string from a CardEvent-like object."""
    raw_type = getattr(event, "type", None)
    return _sanitize_telemetry_token(raw_type, default="unknown")


def _normalize_phase(phase: str | None) -> str:
    if phase == _PHASE_TERMINAL:
        return _PHASE_TERMINAL
    return _PHASE_DISPATCHED


def _log_hook_timing(
    *,
    hook_name: str,
    phase: str,
    event_type: str,
    elapsed_ms: float,
    failed: bool = False,
) -> None:
    """Emit sanitized latency telemetry for a hook execution."""
    if elapsed_ms >= SLOW_HOOK_THRESHOLD_MS:
        logger.warning(
            "Hook slow: hook_name=%s phase=%s event_type=%s elapsed_ms=%.1f",
            hook_name,
            phase,
            event_type,
            elapsed_ms,
        )
    elif not failed:
        logger.debug(
            "Hook completed: hook_name=%s phase=%s event_type=%s elapsed_ms=%.1f",
            hook_name,
            phase,
            event_type,
            elapsed_ms,
        )


class _ExecutorGeneration:
    """Binds a ThreadPoolExecutor with its generation-scoped BoundedSemaphore."""

    def __init__(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        capacity: int,
    ) -> None:
        self.executor = executor
        self.capacity = capacity
        self.semaphore = threading.BoundedSemaphore(capacity)
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def try_acquire(self) -> tuple[bool, int, int]:
        """Attempt non-blocking permit acquisition and return (acquired, in_flight, capacity)."""
        with self._lock:
            acquired = self.semaphore.acquire(blocking=False)
            if acquired:
                self._in_flight += 1
            return acquired, self._in_flight, self.capacity

    def release(self) -> None:
        """Release one slot back to this generation's semaphore."""
        with self._lock:
            self._release_unlocked()

    def _release_unlocked(self) -> None:
        if self._in_flight > 0:
            self._in_flight -= 1
        self.semaphore.release()


@dataclass
class _ExecutorLane:
    """Isolated execution lane state for one hook phase."""

    name: str
    thread_name_prefix: str
    generation: _ExecutorGeneration | None = None
    consecutive_timeouts: int = 0


class _HookExecutorManager:
    """Manages lazy, phase-isolated hook thread pools with generational rebuilds."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._shutdown = False
        self._dispatched_lane = _ExecutorLane(
            name=_PHASE_DISPATCHED,
            thread_name_prefix="hook-dispatched",
        )
        self._terminal_lane = _ExecutorLane(
            name=_PHASE_TERMINAL,
            thread_name_prefix="hook-terminal",
        )

    @property
    def _executor(self) -> concurrent.futures.ThreadPoolExecutor | None:
        gen = self._dispatched_lane.generation
        return gen.executor if gen is not None else None

    @property
    def _terminal_executor(self) -> concurrent.futures.ThreadPoolExecutor | None:
        gen = self._terminal_lane.generation
        return gen.executor if gen is not None else None

    @property
    def _semaphore(self) -> threading.BoundedSemaphore | None:
        gen = self._dispatched_lane.generation
        return gen.semaphore if gen is not None else None

    @property
    def _generation(self) -> _ExecutorGeneration | None:
        return self._dispatched_lane.generation

    @property
    def _terminal_generation(self) -> _ExecutorGeneration | None:
        return self._terminal_lane.generation

    @property
    def _capacity(self) -> int:
        gen = self._dispatched_lane.generation
        if gen is not None:
            return gen.capacity
        return self._get_max_workers() * 4

    @property
    def _consecutive_timeouts(self) -> int:
        return self._dispatched_lane.consecutive_timeouts

    @_consecutive_timeouts.setter
    def _consecutive_timeouts(self, value: int) -> None:
        self._dispatched_lane.consecutive_timeouts = value

    def _lane_for_phase(self, phase: str | None) -> _ExecutorLane:
        if _normalize_phase(phase) == _PHASE_TERMINAL:
            return self._terminal_lane
        return self._dispatched_lane

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
    def _create_executor(thread_name_prefix: str = "hook") -> concurrent.futures.ThreadPoolExecutor:
        max_workers = _HookExecutorManager._get_max_workers()
        return concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )

    def _create_generation(self, lane: _ExecutorLane) -> _ExecutorGeneration:
        max_workers = self._get_max_workers()
        capacity = max_workers * 4
        executor = self._create_executor(thread_name_prefix=lane.thread_name_prefix)
        return _ExecutorGeneration(executor=executor, capacity=capacity)

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        phase: str = _PHASE_DISPATCHED,
        hook_name: str = "unknown",
        event_type: str = "unknown",
    ) -> concurrent.futures.Future | None:
        """Submit a callable to its phase-isolated executor with backpressure protection.

        If all worker slots for the phase are occupied or the executor has shut
        down, skips submission and returns None to protect the dispatch hot path.
        """
        safe_phase = _normalize_phase(phase)
        safe_hook_name = _sanitize_telemetry_token(hook_name, default="unknown")
        safe_event_type = _sanitize_telemetry_token(event_type, default="unknown")
        lane = self._lane_for_phase(safe_phase)

        permit_state = [False]

        def _make_release_once(target_gen: _ExecutorGeneration) -> Callable[[], None]:
            def _release_once() -> None:
                with target_gen._lock:
                    if permit_state[0]:
                        return
                    permit_state[0] = True
                    target_gen._release_unlocked()
            return _release_once

        with self._lock:
            if self._shutdown:
                logger.warning(
                    "Hook executor: submission rejected after shutdown "
                    "(hook_name=%s, phase=%s, event_type=%s)",
                    safe_hook_name,
                    safe_phase,
                    safe_event_type,
                )
                return None
            gen = lane.generation
            if gen is None:
                gen = self._create_generation(lane)
                lane.generation = gen
            acquired, in_flight, capacity = gen.try_acquire()
            if not acquired:
                logger.warning(
                    "Hook executor: backpressure triggered, skipping hook submission "
                    "(hook_name=%s, phase=%s, event_type=%s, in_flight=%d/%d, capacity=%d)",
                    safe_hook_name,
                    safe_phase,
                    safe_event_type,
                    in_flight,
                    capacity,
                    capacity,
                )
                return None

            release_once = _make_release_once(gen)
            try:
                future = gen.executor.submit(
                    self._wrap_with_release(release_once, fn),
                    *args,
                )
            except RuntimeError as exc:
                release_once()
                logger.warning(
                    "Hook executor: submission rejected after shutdown "
                    "(hook_name=%s, phase=%s, event_type=%s): %s",
                    safe_hook_name,
                    safe_phase,
                    safe_event_type,
                    repr(exc),
                )
                return None
            except Exception:
                release_once()
                raise

        future.add_done_callback(lambda _f, _rel=release_once: _rel())
        return future

    @staticmethod
    def _wrap_with_release(release_once: Callable[[], None], fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap fn so the generation-bound semaphore is released once after completion."""
        def wrapper(*args: Any) -> Any:
            try:
                return fn(*args)
            finally:
                release_once()
        return wrapper

    def record_success(self, phase: str = _PHASE_DISPATCHED) -> None:
        """Reset consecutive timeout counter on success."""
        lane = self._lane_for_phase(phase)
        with self._lock:
            lane.consecutive_timeouts = 0

    def record_timeout(self, phase: str = _PHASE_DISPATCHED) -> None:
        """Record a timeout and rebuild that lane's generation if threshold reached."""
        lane = self._lane_for_phase(phase)
        old_executor = None
        with self._lock:
            if self._shutdown:
                return
            lane.consecutive_timeouts += 1
            if lane.consecutive_timeouts >= _MAX_CONSECUTIVE_TIMEOUTS:
                logger.warning(
                    "Hook executor: %d consecutive timeouts, rebuilding thread pool (phase=%s)",
                    lane.consecutive_timeouts,
                    lane.name,
                )
                old_gen = lane.generation
                if old_gen is not None:
                    old_executor = old_gen.executor
                    lane.generation = self._create_generation(lane)
                lane.consecutive_timeouts = 0
        # Shutdown old executor outside lock (non-blocking)
        if old_executor is not None:
            try:
                old_executor.shutdown(wait=False)
            except Exception:
                pass

    def shutdown(self) -> None:
        """Gracefully shut down active hook executor pools with timeout protection.

        Uses shutdown(wait=False, cancel_futures=True) then joins threads with
        an explicit timeout to prevent hanging the process exit.

        Called during process shutdown to ensure clean thread cleanup.
        """
        _SHUTDOWN_TIMEOUT = 10.0
        executors: list[concurrent.futures.ThreadPoolExecutor] = []
        with self._lock:
            self._shutdown = True
            for lane in (self._dispatched_lane, self._terminal_lane):
                if lane.generation is not None:
                    executors.append(lane.generation.executor)
                    lane.generation = None
        # Signal shutdown and cancel pending futures immediately
        for executor in executors:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                logger.warning("Hook executor shutdown signal failed: %s", repr(exc))
                continue
            # Join worker threads with timeout to avoid indefinite blocking
            for t in list(getattr(executor, "_threads", set())):
                try:
                    t.join(timeout=_SHUTDOWN_TIMEOUT)
                except Exception:
                    pass


_hook_executor_manager = _HookExecutorManager()


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
            logger.debug("EmojiHook: skipping reaction (no message_id)")
            return
        if reason in ("completed", "completed_empty"):
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

_NOOP_DISPATCHED_METHODS = (
    EmojiHook.on_dispatched,
    ContextPersistenceHook.on_dispatched,
)


def _is_noop_dispatched(fn: object) -> bool:
    """Return True when fn is an unoverridden built-in no-op on_dispatched."""
    raw_fn = fn.__func__ if isinstance(fn, types.MethodType) else fn
    return any(raw_fn is noop for noop in _NOOP_DISPATCHED_METHODS)


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

    @property
    def has_hooks(self) -> bool:
        return bool(self._hooks)

    def fire_dispatched(self, event: CardEvent, state: CardState | None) -> None:
        """Fire on_dispatched hooks (fire-and-forget).

        Unoverridden no-op on_dispatched methods (EmojiHook, ContextPersistenceHook)
        are skipped without touching the executor. Active hooks are submitted to
        the shared executor and return immediately — no synchronous wait on the
        dispatch hot path.
        """
        if not self._hooks or state is None:
            return
        sid = self._session_id
        executor = self._executor
        event_type: str | None = None
        submit_time: float | None = None
        for hook in self._hooks:
            fn = getattr(hook, "on_dispatched", None)
            if fn is None or _is_noop_dispatched(fn):
                continue
            if event_type is None:
                event_type = _format_event_type(event)
                submit_time = time.monotonic()
            hook_name = _sanitize_telemetry_token(type(hook).__name__, default="Hook")
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
                    hook_name,
                    event_type,
                    phase=_PHASE_DISPATCHED,
                    hook_name=hook_name,
                    event_type=event_type,
                )
            except Exception:
                _finish_once()
                raise
            if future is None:
                _finish_once()
                continue
            future.add_done_callback(
                lambda f, _finish=_finish_once: _finish()
            )
            future.add_done_callback(
                lambda f, _sid=sid, _name=hook_name, _ev=event_type, _t0=submit_time, _ex=executor: (
                    self._on_dispatched_done(f, _sid, _name, _ev, _t0, _ex)
                )
            )

    def _run_dispatched_hook(
        self,
        fn: Callable[[CardEvent, CardState], None],
        event: CardEvent,
        state: CardState,
        finish: Callable[[], None],
        hook_name: str = "unknown",
        event_type: str = "unknown",
    ) -> None:
        """Run one hook while marking this callback as self-drain exempt."""
        active = getattr(_dispatched_hook_context, "active_firers", None)
        if active is None:
            active = {}
            _dispatched_hook_context.active_firers = active
        firer_id = id(self)
        active[firer_id] = active.get(firer_id, 0) + 1
        start = time.monotonic()
        failed = False
        try:
            fn(event, state)
        except Exception:
            failed = True
            raise
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            _log_hook_timing(
                hook_name=hook_name,
                phase=_PHASE_DISPATCHED,
                event_type=event_type,
                elapsed_ms=elapsed_ms,
                failed=failed,
            )
            remaining = active[firer_id] - 1
            if remaining:
                active[firer_id] = remaining
            else:
                active.pop(firer_id, None)
            finish()

    @staticmethod
    def _run_terminal_hook(
        fn: Callable[[CardState, str], None],
        state: CardState,
        reason: str,
        hook_name: str,
        event_type: str,
    ) -> None:
        """Execute one terminal hook with latency telemetry."""
        start = time.monotonic()
        failed = False
        try:
            fn(state, reason)
        except Exception:
            failed = True
            raise
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            _log_hook_timing(
                hook_name=hook_name,
                phase=_PHASE_TERMINAL,
                event_type=event_type,
                elapsed_ms=elapsed_ms,
                failed=failed,
            )

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
    def _on_dispatched_done(
        f: concurrent.futures.Future,
        sid: str,
        hook_name: str,
        event_type: str,
        submit_time: float,
        executor: _HookExecutorManager,
    ) -> None:
        """Done-callback for dispatched hooks: log errors and detect timeouts."""
        try:
            exc = f.exception()
        except concurrent.futures.CancelledError:
            return
        if exc is not None:
            logger.warning(
                "HookFirer %s: on_dispatched failed (hook_name=%s, phase=%s, event_type=%s): %s",
                sid, hook_name, _PHASE_DISPATCHED, event_type, repr(exc),
            )
        elapsed = time.monotonic() - submit_time
        if elapsed > DISPATCHED_HOOK_TIMEOUT:
            logger.warning(
                "HookFirer %s: on_dispatched exceeded timeout (hook_name=%s, phase=%s, event_type=%s): %.2fs > %.1fs",
                sid, hook_name, _PHASE_DISPATCHED, event_type, elapsed, DISPATCHED_HOOK_TIMEOUT,
            )
            executor.record_timeout(phase=_PHASE_DISPATCHED)
        elif exc is None:
            executor.record_success(phase=_PHASE_DISPATCHED)

    def fire_terminal(self, state: CardState | None, reason: str) -> None:
        """Fire on_terminal hooks in parallel with unified timeout.

        All hooks are submitted concurrently to the isolated terminal lane, then
        waited on with a single HOOK_TIMEOUT_SECONDS deadline. This bounds
        worst-case blocking to HOOK_TIMEOUT_SECONDS regardless of hook count.

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

        event_type = _sanitize_telemetry_token(reason, default="terminal")
        # Submit all hooks in parallel
        future_to_meta: dict[concurrent.futures.Future, tuple[SessionHook, str]] = {}
        for hook in self._hooks:
            fn = getattr(hook, "on_terminal", None)
            if fn is None:
                continue
            hook_name = _sanitize_telemetry_token(type(hook).__name__, default="Hook")
            try:
                future = self._executor.submit(
                    self._run_terminal_hook,
                    fn,
                    state,
                    reason,
                    hook_name,
                    event_type,
                    phase=_PHASE_TERMINAL,
                    hook_name=hook_name,
                    event_type=event_type,
                )
                if future is not None:
                    future_to_meta[future] = (hook, hook_name)
                else:
                    logger.warning(
                        "HookFirer %s: on_terminal submission rejected (hook_name=%s, phase=%s, event_type=%s)",
                        self._session_id, hook_name, _PHASE_TERMINAL, event_type,
                    )
            except Exception as exc:
                logger.warning(
                    "HookFirer %s: on_terminal submit failed (hook_name=%s, phase=%s, event_type=%s): %s",
                    self._session_id, hook_name, _PHASE_TERMINAL, event_type, repr(exc),
                )

        if not future_to_meta:
            return

        # Wait for all futures with a single timeout deadline
        done, not_done = concurrent.futures.wait(
            future_to_meta.keys(), timeout=HOOK_TIMEOUT_SECONDS
        )

        # Process completed futures
        has_timeout = False
        for future in done:
            _hook, hook_name = future_to_meta[future]
            try:
                future.result(timeout=0)  # Already done, just check for exceptions
            except Exception as exc:
                logger.warning(
                    "HookFirer %s: on_terminal failed (hook_name=%s, phase=%s, event_type=%s): %s",
                    self._session_id, hook_name, _PHASE_TERMINAL, event_type, repr(exc),
                )

        # Handle timed-out futures
        for future in not_done:
            _hook, hook_name = future_to_meta[future]
            future.cancel()
            has_timeout = True
            logger.warning(
                "HookFirer %s: on_terminal timed out (hook_name=%s, phase=%s, event_type=%s) after %.1fs",
                self._session_id, hook_name, _PHASE_TERMINAL, event_type, HOOK_TIMEOUT_SECONDS,
            )

        # Update executor health tracking
        if has_timeout:
            self._executor.record_timeout(phase=_PHASE_TERMINAL)
        elif done:
            self._executor.record_success(phase=_PHASE_TERMINAL)
