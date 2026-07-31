"""RuntimeBridge — manages the Node.js workflow runtime subprocess."""

from __future__ import annotations

import collections
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from typing import Any, Callable, Optional

from .constants import (
    AGENT_CALL_TIMEOUT_S,
    AGENT_UNLIMITED_BACKSTOP_S,
    DEFAULT_MAX_CONCURRENT,
    HARD_MAX_CONCURRENT,
    MAX_NESTING_DEPTH,
    MAX_QUEUE_SIZE,
    NODE_MIN_VERSION,
    RUNTIME_JS_PATH,
    SCHEMA_RETRY_MAX,
    STDERR_TAIL_MAX_LINES,
    WORKFLOW_TIMEOUT_HEADROOM_S,
    WORKFLOW_TOTAL_TIMEOUT_S,
)
from .errors import ErrorCategory, _strip_internal_details, sanitize_for_reply
from .models import AgentCallParams, AgentCallResult

logger = logging.getLogger(__name__)


def _settings_int(field: str, fallback: int) -> int:
    """Read an int workflow-timeout setting, falling back to the constant.

    Settings overrides (from .env) must take effect at runtime, but we must
    never let a missing/invalid config break the workflow — hence the graceful
    fallback to the import-time constant.
    """
    try:
        from src.config import get_settings

        value = getattr(get_settings(), field, fallback)
        return int(value)
    except Exception:  # pragma: no cover - defensive: config not importable
        return fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_node_version(version_str: str) -> tuple[int, ...]:
    """Parse a Node.js version string like 'v20.11.0' into (20, 11, 0)."""
    cleaned = version_str.strip().lstrip("v")
    parts = cleaned.split(".")
    return tuple(int(p) for p in parts if p.isdigit())


# ---------------------------------------------------------------------------
# RuntimeBridge
# ---------------------------------------------------------------------------


class RuntimeBridge:
    """Manages a Node.js subprocess running the workflow runtime.

    Communication uses JSON-RPC 2.0 over stdin/stdout (NDJSON — one JSON
    object per line). The bridge spawns the subprocess, dispatches incoming
    requests/notifications, and provides a thread-safe write path.
    """

    def __init__(
        self,
        script_path: str,
        cwd: str,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        on_agent_call: Optional[Callable[..., AgentCallResult]] = None,
        on_agent_aborted: Optional[Callable[..., None]] = None,  # (label, reason, request_id=None)
        on_phase: Optional[Callable[[str], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        allowed_tools: Optional[list[str]] = None,
        nesting_depth: int = 0,
        args: Optional[dict[str, Any]] = None,
        initiator_user_id: Optional[str] = None,
        workflow_deadline_monotonic: float | None = None,
    ) -> None:
        self._script_path = script_path
        self._cwd = cwd
        self._max_concurrent = min(max_concurrent, HARD_MAX_CONCURRENT)
        self._on_agent_call = on_agent_call
        self._on_agent_aborted = on_agent_aborted
        self._on_phase = on_phase
        self._on_log = on_log
        self._cancel_event = cancel_event or threading.Event()
        self._allowed_tools = allowed_tools
        self._nesting_depth = nesting_depth
        self._args = args or {}
        # Scoping identity used when resolving workflow_call templates.  Sub
        # workflows inherit this value from their parent so user-scoped
        # templates remain consistent across the tree.  May be None when the
        # bridge is constructed without a known initiator.
        self._initiator_user_id: Optional[str] = initiator_user_id
        # Confirmation-time deadline from WorkflowRunSpec. It is converted to
        # the JS wall-clock representation exactly once in
        # _ensure_workflow_deadline().
        self._confirmed_deadline_monotonic = workflow_deadline_monotonic

        # Subprocess handle
        self._process: Optional[subprocess.Popen] = None
        self._process_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock

        # Thread-safe stdin writes
        self._write_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Read loop thread
        self._reader_thread: Optional[threading.Thread] = None

        # Bounded ring buffer of the most recent Node stderr lines. The
        # dedicated stderr drain thread continuously consumes the stderr pipe
        # (to avoid a pipe-buffer deadlock) and logs each line at DEBUG. That
        # means by the time stdout closes, a one-shot _drain_stderr() would
        # find the pipe already empty — so the dying process's stderr would be
        # lost. Keeping a tail here lets the unexpected-exit diagnostic surface
        # the real cause (e.g. a V8 OOM/FATAL line) instead of a bare
        # "closed stdout unexpectedly".
        self._stderr_tail: collections.deque[str] = collections.deque(
            maxlen=STDERR_TAIL_MAX_LINES
        )
        self._stderr_tail_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Incoming message queue (for the run() loop)
        self._msg_queue: collections.deque[dict[str, Any]] = collections.deque()
        self._msg_condition = threading.Condition()

        # ThreadPoolExecutor for agent calls
        self._executor: Optional[ThreadPoolExecutor] = None

        # Separate executor for sub-workflow calls (avoids starvation)
        self._workflow_executor: Optional[ThreadPoolExecutor] = None

        # Track active futures for graceful shutdown (set for O(1) discard).
        # Once a future is submitted via ``executor.submit`` it is added to
        # ``_active_futures``; the done-callback ``_discard_future`` removes
        # it. ``in_flight_count`` reports the size of that set.
        self._active_futures: set[Future] = set()
        self._futures_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Map JSON-RPC request_id → Future for abort propagation. When the JS
        # runtime sends abort_request for a specific in-flight agent call, we
        # cancel the future here.
        self._request_futures: dict[Any, Future] = {}
        self._request_futures_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Map JSON-RPC request_id → per-call cancel_event. When abort_request
        # arrives, we set this event to interrupt the in-flight ACP session,
        # enabling race() loser agents to stop within seconds rather than
        # running to completion.
        self._request_cancel_events: dict[Any, threading.Event] = {}
        self._request_cancel_events_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Parent reference and child tracking for cascade cancellation
        self._parent: Optional["RuntimeBridge"] = None
        self._children: list["RuntimeBridge"] = []
        self._children_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Shutdown flag to make stop() / cleanup() idempotent
        self._shutdown_done = False
        self._shutdown_started = threading.Event()

        # Terminal state
        self._done = False
        self._result: Optional[str] = None
        self._error: Optional[str] = None
        # Fallback diagnostic recorded when stdout closes / the process dies
        # WITHOUT a terminal done/error frame. Kept separate from ``_error`` so
        # a real terminal frame that is still queued at EOF (a common race now
        # that the runtime flushes before exit) always wins over the synthetic
        # "process died" message. Only used if no done/error frame is applied.
        self._eof_fallback_error: Optional[str] = None

        # Workflow-level deadline shared with the JS runtime.  The monotonic
        # value drives Python-side enforcement; unix-ms values are sent to JS.
        self._workflow_started_monotonic: float | None = None
        self._workflow_deadline_monotonic: float | None = None
        self._workflow_started_unix_ms: int | None = None
        self._workflow_deadline_unix_ms: int | None = None
        # Effective total timeout (seconds) resolved from Settings once per run.
        self._workflow_total_timeout_s: int = WORKFLOW_TOTAL_TIMEOUT_S
        # Effective per-agent timeout floor (seconds) resolved from Settings
        # once per run. This is the authoritative floor sent to the JS runtime
        # so the JS-side agent_call watchdog matches the Python-side executor
        # rather than the (often much smaller) timeout baked into the script.
        # 0 means unlimited per-agent (JS relies on the total deadline / stop).
        self._workflow_agent_call_timeout_s: int = AGENT_CALL_TIMEOUT_S

    def _shutdown_requested(self) -> bool:
        """Return cancellation state, including for lightweight test doubles."""
        shutdown_event = vars(self).get("_shutdown_started")
        cancel_event = vars(self).get("_cancel_event")
        return bool(
            (
                shutdown_event is not None
                and shutdown_event.is_set() is True
            )
            or (
                cancel_event is not None
                and cancel_event.is_set() is True
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def check_node_available(cls) -> bool:
        """Check that Node.js is installed and meets the minimum version."""
        node_bin = shutil.which("node")
        if not node_bin:
            logger.warning("Node.js not found in PATH")
            return False

        try:
            result = subprocess.run(
                [node_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning("node --version returned non-zero: %s", result.stderr)
                return False

            version = _parse_node_version(result.stdout)
            if version < NODE_MIN_VERSION:
                logger.warning(
                    "Node.js version %s is below minimum %s",
                    result.stdout.strip(),
                    ".".join(str(v) for v in NODE_MIN_VERSION),
                )
                return False

            logger.debug("Node.js version OK: %s", result.stdout.strip())
            return True

        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Failed to check Node.js version: %s", repr(exc))
            return False

    def _ensure_workflow_deadline(self) -> None:
        """Initialize the per-run deadline once and keep it stable.

        A total timeout of ``0`` (or negative) means *unlimited*: no total
        deadline is enforced.  In that mode the monotonic deadline stays
        ``None`` and the unix-ms value sent to JS is ``0`` (JS treats a
        falsy deadline as ``Infinity``).  Per-agent timeouts and the
        MAX_TOTAL_AGENTS fuse still bound resource use.
        """
        if self._workflow_deadline_monotonic is not None:
            return
        # Already resolved unlimited mode on a prior call — keep it stable.
        if self._workflow_started_monotonic is not None:
            return

        # Read the effective total timeout from Settings (allows .env override)
        # exactly once per run, then cache it so init/run stay consistent.
        total_timeout_s = _settings_int(
            "workflow_total_timeout_s", WORKFLOW_TOTAL_TIMEOUT_S
        )
        self._workflow_total_timeout_s = total_timeout_s

        # Resolve the per-agent timeout floor once per run too, so the value
        # sent to JS in init() matches what the Python executor enforces.
        self._workflow_agent_call_timeout_s = _settings_int(
            "workflow_agent_call_timeout_s", AGENT_CALL_TIMEOUT_S
        )

        started_monotonic = time.monotonic()
        started_unix_ms = int(time.time() * 1000)
        self._workflow_started_monotonic = started_monotonic
        self._workflow_started_unix_ms = started_unix_ms

        confirmed_deadline = vars(self).get("_confirmed_deadline_monotonic")
        if confirmed_deadline is not None:
            remaining_s = max(
                0.0,
                confirmed_deadline - started_monotonic,
            )
            self._workflow_deadline_monotonic = confirmed_deadline
            self._workflow_deadline_unix_ms = started_unix_ms + int(remaining_s * 1000)
            self._workflow_total_timeout_s = max(0, int(remaining_s))
            return

        if total_timeout_s <= 0:
            # Unlimited: no total deadline. Leave *_deadline_* fields cleared;
            # JS receives deadline_unix_ms=0 → Infinity.
            self._workflow_deadline_monotonic = None
            self._workflow_deadline_unix_ms = 0
            return

        self._workflow_deadline_monotonic = started_monotonic + total_timeout_s
        self._workflow_deadline_unix_ms = started_unix_ms + total_timeout_s * 1000

    def start(self) -> None:
        """Spawn the Node.js runtime subprocess and wait for 'ready' signal.

        Raises RuntimeError if the process fails to start or doesn't send
        the ready notification within a reasonable time.
        """
        # Resolve paths
        runtime_path = os.path.join(self._cwd, RUNTIME_JS_PATH)
        if not os.path.isfile(runtime_path):
            # Try absolute path as fallback
            runtime_path = RUNTIME_JS_PATH

        node_bin = shutil.which("node")
        if not node_bin:
            raise RuntimeError("Node.js not found in PATH")

        cmd = [node_bin, "--experimental-vm-modules", runtime_path, self._script_path]
        logger.info("Starting Node.js runtime: %s", " ".join(cmd))

        with self._process_lock:
            if self._process is not None:
                raise RuntimeError("RuntimeBridge already started")
            if self._shutdown_requested():
                raise RuntimeError("RuntimeBridge cancelled before start")
            try:
                # Minimal environment for Node.js runtime — excludes secrets
                # and API keys from the parent process.
                safe_env = {
                    "PATH": os.environ.get("PATH", ""),
                    "HOME": os.environ.get("HOME", ""),
                    "NODE_PATH": os.environ.get("NODE_PATH", ""),
                    "LANG": os.environ.get("LANG", "en_US.UTF-8"),
                    "TERM": os.environ.get("TERM", "xterm"),
                }
                # Allow explicit NODE_OPTIONS if set (for debugging/flags)
                if os.environ.get("NODE_OPTIONS"):
                    safe_env["NODE_OPTIONS"] = os.environ["NODE_OPTIONS"]

                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self._cwd,
                    text=True,
                    bufsize=1,  # Line-buffered
                    env=safe_env,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to spawn Node.js process: {exc}"
                ) from exc

        # Serialize reader publication with stop(). If cancellation landed in
        # the narrow gap after Popen, no transport thread may appear after the
        # stop acknowledgement.
        with self._process_lock:
            if self._shutdown_requested() or self._process is None:
                self._kill_process()
                raise RuntimeError("RuntimeBridge cancelled during start")

            # Start the reader thread (daemon so it dies with the process)
            self._reader_thread = threading.Thread(
                target=self._read_loop,
                name="RuntimeBridge-reader",
                daemon=True,
            )
            self._reader_thread.start()

            # Start stderr drain thread to prevent pipe deadlock (NFR-3)
            self._stderr_thread = threading.Thread(
                target=self._stderr_reader,
                name="RuntimeBridge-stderr",
                daemon=True,
            )
            self._stderr_thread.start()

        # Wait for the 'ready' notification
        ready = self._wait_for_notification("ready", timeout=30.0)
        if ready is None:
            # Capture stderr BEFORE _kill_process() nulls out self._process,
            # otherwise the diagnostic would be empty.
            stderr_content = self._stderr_diagnostic()
            self._kill_process()
            raise RuntimeError(f"Node.js runtime did not send 'ready' within 30s. stderr: {stderr_content}")

        self._ensure_workflow_deadline()

        # Send init with workflow args and max_concurrent so the
        # JS-side parallel() primitive can bound concurrency to match the
        # Python ThreadPoolExecutor size.  Deadline fields let JS reject new
        # work before the Python hard timeout kills the process.
        self._send(
            {
                "jsonrpc": "2.0",
                "method": "init",
                "params": {
                    "args": self._args,
                    "max_concurrent": self._max_concurrent,
                    "started_unix_ms": self._workflow_started_unix_ms,
                    "deadline_unix_ms": self._workflow_deadline_unix_ms,
                    "total_timeout_s": self._workflow_total_timeout_s,
                    "agent_call_timeout_s": self._workflow_agent_call_timeout_s,
                },
            }
        )

        # Publish both pools through the same submission gate used by stop().
        # If cancellation won while Node was becoming ready, no executor may
        # appear after the stop acknowledgement.
        with self._futures_lock:
            if self._shutdown_requested():
                raise RuntimeError("RuntimeBridge cancelled during start")
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_concurrent,
                thread_name_prefix="RuntimeBridge-agent",
            )

            # Separate pool for sub-workflows (avoids agent-pool starvation).
            self._workflow_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="RuntimeBridge-subwf",
            )

        logger.info("Node.js runtime ready and initialized")

    def run(self) -> str:
        """Main event loop: process messages from the Node.js runtime.

        Blocks until the runtime sends 'done' or 'error', or until the
        total timeout expires. The total timeout is activity-aware: as long
        as the runtime is producing messages (agent calls, phase changes,
        logs), the deadline is extended. The workflow is only killed after
        a prolonged period of complete silence.

        Raises:
            RuntimeError: If the subprocess dies unexpectedly, times out,
                or sends an error notification.
        """
        if self._process is None:
            raise RuntimeError("RuntimeBridge not started — call start() first")

        self._ensure_workflow_deadline()
        # ``deadline`` is None in unlimited mode (workflow_total_timeout_s <= 0):
        # the total-timeout check is skipped and the loop only exits on
        # done/error/cancel/process-death, letting the user stop when ready.
        deadline = self._workflow_deadline_monotonic

        # Activity-based timeout: track the last time any message arrived from
        # the JS runtime. The hard deadline is used as an absolute cap, but a
        # softer idle check kills the workflow if no messages arrive for a long
        # period (meaning all agents are stuck/dead).
        last_activity = time.monotonic()
        # Idle threshold: 5 minutes of complete silence from the JS runtime
        # (no phase, no log, no agent_call, no done). This is generous because
        # during a long ACP call the JS runtime itself produces no messages —
        # only the Python executor does. The bridge-level idle is a last-resort
        # safety net; per-agent idle is handled by sync_adapter.
        bridge_idle_timeout_s = max(
            300, self._workflow_total_timeout_s // 6 if self._workflow_total_timeout_s > 0 else 300
        )

        while not self._done:
            # Check cancellation
            if self._cancel_event.is_set():
                logger.info("Cancel event set — stopping runtime")
                self._send_cancel()
                self._kill_process()
                raise RuntimeError("Workflow cancelled")

            # Check timeout: activity-aware
            if deadline is not None:
                now = time.monotonic()
                # Hard cap — absolute deadline never exceeded
                if now >= deadline:
                    # But if there are active agent calls running, extend the
                    # deadline rather than killing mid-work.
                    if self.in_flight_count > 0:
                        # Extend by the idle timeout — agents are still working
                        deadline = now + bridge_idle_timeout_s
                        self._workflow_deadline_monotonic = deadline
                        logger.info(
                            "Workflow deadline extended — %d agent(s) still in flight",
                            self.in_flight_count,
                        )
                    else:
                        # No agents running and deadline hit — truly timed out
                        idle_s = now - last_activity
                        logger.error(
                            "Workflow total timeout exceeded (%ds, idle=%.0fs, in_flight=0)",
                            self._workflow_total_timeout_s, idle_s,
                        )
                        self._kill_process()
                        raise RuntimeError(
                            f"Workflow execution exceeded total timeout of {self._workflow_total_timeout_s}s"
                        )
                remaining = deadline - time.monotonic()
            else:
                remaining = None

            # Check process health. The process may have exited cleanly right
            # after queueing its terminal done/error frame, so drain and
            # dispatch anything still queued before treating the exit as an
            # unexpected death — otherwise a valid result would be masked by a
            # synthetic "process exited" error.
            if self._process.poll() is not None and not self._done:
                self._drain_pending_messages()
                if self._done:
                    # A terminal frame was applied while draining — fall
                    # through to the normal exit path below.
                    break
                stderr_content = self._stderr_diagnostic()
                sanitized_stderr = _strip_internal_details(stderr_content)
                raise RuntimeError(
                    f"Node.js process exited unexpectedly with code "
                    f"{self._process.returncode}. stderr: {sanitized_stderr}"
                )

            # Wait for next message. Cap the block at 1s so cancellation and
            # process-death are still detected promptly in unlimited mode.
            pop_timeout = 1.0 if remaining is None else min(1.0, remaining)
            msg = self._pop_message(timeout=pop_timeout)
            if msg is None:
                continue

            # Any message from the JS runtime counts as activity
            last_activity = time.monotonic()
            self._dispatch_message(msg)

        # The loop exited because self._done is set. Drain any terminal frame
        # (done/error) that the reader thread queued just before stdout EOF so
        # a real result/error always wins over the synthetic EOF fallback.
        self._drain_pending_messages()

        # Return result or raise error. Precedence: a genuine done/error frame
        # (which sets self._result or self._error) is authoritative; the EOF
        # fallback is only used when the process vanished without one.
        if self._error:
            raise RuntimeError(f"Workflow runtime error: {self._error}")
        if self._result is not None:
            return self._result
        if self._eof_fallback_error:
            raise RuntimeError(f"Workflow runtime error: {self._eof_fallback_error}")

        return self._result or ""

    def stop(self) -> None:
        """Send cancel notification and kill the subprocess.

        Also cascades cancellation to all child sub-workflows and shuts down
        the ThreadPoolExecutors. Idempotent — calling it more than once is
        safe and does nothing after the first successful call.
        """
        # Close the submission gate before touching either executor. A request
        # already being dispatched must either publish its Future under
        # ``_futures_lock`` first or observe this event and fail closed.
        self._shutdown_started.set()
        if self._shutdown_done:
            return

        # Signal our own cancel event first
        self._cancel_event.set()

        # Cascade cancel to children first (under lock to avoid race)
        with self._children_lock:
            for child in self._children:
                try:
                    child._cancel_event.set()
                    child.stop()
                except Exception:
                    logger.exception("Failed to cascade stop to child sub-workflow")
            self._children.clear()

        # Signal all in-flight agent calls to cancel before killing the
        # process.  Each per-call cancel_event interrupts the ACP session's
        # send_prompt via the cancel-guard thread, so in-flight agents stop
        # within the guard poll interval (~200ms) rather than running to
        # their full 300s timeout.
        with self._request_cancel_events_lock:
            cancel_events_snapshot = list(self._request_cancel_events.values())
        for evt in cancel_events_snapshot:
            try:
                evt.set()
            except Exception:
                logger.debug("Failed to set per-call cancel event during stop()")

        # Kill the Node.js process FIRST before waiting on executor shutdown.
        # This ensures in-flight agent calls get stdin broken and fail-fast
        # rather than blocking stop() for up to 300s per agent.
        if self._process is not None:
            try:
                self._send_cancel()
            except (OSError, BrokenPipeError):
                pass  # Process may already be dead
            self._kill_process()

        # Wait briefly for reader/stderr threads to finish
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        if hasattr(self, "_stderr_thread") and self._stderr_thread is not None and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=2.0)

        # Cancel all pending futures and shut down executors without waiting
        # (process is already dead, in-flight calls will fail with BrokenPipe)
        with self._futures_lock:
            futures_to_cancel = list(self._active_futures)
            executor = self._executor
            workflow_executor = self._workflow_executor
            self._executor = None
            self._workflow_executor = None

        if executor is not None:
            try:
                # Snapshot active futures under the lock, then cancel outside
                # to avoid deadlock: cancel() invokes done callbacks which
                # call _discard_future, which itself acquires _futures_lock.
                for future in futures_to_cancel:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            except Exception:
                logger.debug("RuntimeBridge executor shutdown failed")
        if workflow_executor is not None:
            try:
                workflow_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                workflow_executor.shutdown(wait=False)
            except Exception:
                logger.debug("RuntimeBridge subwf executor shutdown failed")

        # Clear request futures map
        with self._request_futures_lock:
            self._request_futures.clear()
        with self._request_cancel_events_lock:
            self._request_cancel_events.clear()

        self._shutdown_done = True
        logger.info("RuntimeBridge stopped")

    def wait_for_workers(self, timeout: float | None = None) -> bool:
        """Wait until every submitted agent/sub-workflow callback is quiet.

        ``stop()`` intentionally remains non-blocking for responsive user
        cancellation. The execution owner calls this method before releasing
        its WorkflowEngine run claim, so callbacks from a retired bridge can
        never observe the component fields of a later run.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            with self._futures_lock:
                active = tuple(
                    future
                    for future in self._active_futures
                    if not future.done()
                )
            if not active:
                return True

            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
            _, not_done = wait_futures(active, timeout=remaining)
            if not_done:
                return False

    def cleanup(self) -> None:
        """Alias for stop(); ensures all resources including thread pools
        are released. Safe to call multiple times.
        """
        self.stop()
        self.wait_for_workers()

    # Context manager protocol

    def __enter__(self) -> "RuntimeBridge":
        """Enter context manager — returns self for use with `with` statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context manager — stops the bridge.

        Does not suppress exceptions (returns False).
        """
        self.cleanup()
        return False

    def __del__(self) -> None:
        """Destructor fallback — best-effort cleanup if stop() was never called.

        Defensive: guards against missing attributes and swallows exceptions
        to avoid errors during interpreter shutdown.
        """
        try:
            if hasattr(self, "_shutdown_done") and not self._shutdown_done:
                logger.warning("RuntimeBridge was not properly stopped; call stop() or use as context manager")
                try:
                    self.stop()
                except Exception:
                    logger.debug(
                        "RuntimeBridge __del__ stop() failed (ignored)",
                        exc_info=True,
                    )
        except Exception:
            # Never let __del__ raise — interpreter shutdown is fragile
            pass

    def is_alive(self) -> bool:
        """Check if the Node.js subprocess is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    # ------------------------------------------------------------------
    # Internal: Read loop
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """Reader thread — parses NDJSON lines from stdout.

        Runs as a daemon thread until stdout is closed or the process dies.
        """
        assert self._process is not None
        assert self._process.stdout is not None

        consecutive_non_json = 0
        NON_JSON_WARN_THRESHOLD = 10

        try:
            for line in self._process.stdout:
                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                    consecutive_non_json = 0
                except json.JSONDecodeError:
                    consecutive_non_json += 1
                    if consecutive_non_json >= NON_JSON_WARN_THRESHOLD:
                        logger.warning(
                            "Non-JSON output from runtime (%d consecutive): %s",
                            consecutive_non_json,
                            line[:200],
                        )
                    continue

                # Push to message queue (bounded)
                with self._msg_condition:
                    if len(self._msg_queue) >= MAX_QUEUE_SIZE:
                        error_msg = (
                            f"Runtime message queue full ({MAX_QUEUE_SIZE}); "
                            "aborting workflow to avoid a pending JS request"
                        )
                        logger.error(error_msg)
                        self._error = error_msg
                        self._done = True
                        self._msg_condition.notify_all()
                        break
                    self._msg_queue.append(msg)
                    self._msg_condition.notify()

        except (ValueError, OSError):
            # stdout closed — process is shutting down
            pass

        # stdout EOF: the process is gone. Record a fallback diagnostic but do
        # NOT set self._error directly — a valid done/error frame may still be
        # sitting in the queue (the runtime flushes it right before exit). The
        # run() loop drains the queue first and only falls back to this string
        # when no terminal frame arrived. We still set _done so the loop wakes.
        if not self._done:
            logger.debug("Reader thread: stdout closed, signalling done")
            if self._eof_fallback_error is None:
                self._eof_fallback_error = self._describe_unexpected_exit()
            # Wake the run() loop; it will drain any queued terminal frame
            # before consulting the fallback.
            with self._msg_condition:
                self._done = True
                self._msg_condition.notify_all()

        logger.debug("Reader thread exiting")

    # ------------------------------------------------------------------
    # Internal: Message dispatch
    # ------------------------------------------------------------------

    def _dispatch_message(self, msg: dict[str, Any]) -> None:
        """Route an incoming JSON-RPC message by method."""
        method = msg.get("method")
        params = msg.get("params", {})
        msg_id = msg.get("id")

        # Request from JS runtime (has id — expects a response)
        if msg_id is not None and method is not None:
            self._dispatch_request(msg)
            return

        # Notification from JS runtime (no id)
        if method == "ready":
            # Already handled during start()
            pass
        elif method == "phase":
            title = params.get("title", "")
            if self._on_phase:
                try:
                    self._on_phase(title)
                except Exception:
                    logger.exception("on_phase callback error")
        elif method == "log":
            message = params.get("message", "")
            if self._on_log:
                try:
                    self._on_log(message)
                except Exception:
                    logger.exception("on_log callback error")
        elif method == "done":
            result = params.get("result")
            self._result = json.dumps(result, ensure_ascii=False) if result is not None else ""
            self._done = True
        elif method == "error":
            error_msg = params.get("message", "Unknown error")
            stack = params.get("stack", "")
            # Store raw detail for logging; only the message (without stack)
            # propagates as the RuntimeError — handler layer does final sanitization.
            if stack:
                logger.debug("JS runtime error stack:\n%s", stack)
            self._error = error_msg
            self._done = True
        elif method == "abort_request":
            # JS runtime is asking us to abort a specific in-flight agent request
            request_id = params.get("request_id")
            self._handle_abort_request(request_id)
        elif method == "agent_aborted":
            # JS runtime notifies that an agent was aborted (e.g. race loser).
            # Used to update the progress card so aborted agents no longer
            # show as '执行中'.
            label = params.get("label", "")
            reason = params.get("reason", "Aborted")
            request_id = params.get("request_id")
            if self._on_agent_aborted:
                try:
                    self._on_agent_aborted(label, reason, request_id=request_id)
                except TypeError:
                    # Backward compat: callback only accepts (label, reason)
                    self._on_agent_aborted(label, reason)
                except Exception:
                    logger.exception("on_agent_aborted callback error")
        else:
            logger.debug("Unhandled notification method: %s", method)

    def _dispatch_request(self, msg: dict[str, Any]) -> None:
        """Handle an incoming JSON-RPC request (has id, expects response)."""
        method = msg.get("method", "")
        params = msg.get("params", {})
        request_id = msg.get("id")

        if method == "agent_call":
            self._handle_agent_call(params, request_id)
        elif method == "workflow_call":
            self._handle_workflow_call(params, request_id)
        else:
            self._send_error_response(
                request_id,
                code=-32601,
                message=f"Unknown method: {method}",
            )

    def _remaining_workflow_budget_s(self) -> float | None:
        """Return seconds remaining before the total workflow deadline."""
        if self._workflow_deadline_monotonic is None:
            return None
        return self._workflow_deadline_monotonic - time.monotonic()

    def _reject_if_workflow_budget_exhausted(self, request_id: Any) -> bool:
        """Reject a request when no useful budget remains."""
        remaining = self._remaining_workflow_budget_s()
        if remaining is None:
            return False
        if remaining > WORKFLOW_TIMEOUT_HEADROOM_S:
            return False

        self._send_error_response(
            request_id,
            code=-32002,
            message=(f"Workflow deadline exhausted before starting agent call (remaining={max(0.0, remaining):.1f}s)"),
        )
        return True

    def _cap_agent_timeout_to_remaining_budget(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve the per-agent timeout, honoring the host floor and budget.

        Two independent concerns are handled here:

        1. Host config is the authoritative *floor*. The generated JS bakes a
           small ``timeout`` (e.g. 180) into each agent() call; honoring that
           verbatim was prematurely killing long-running coding tasks. So the
           script value may only *raise* the timeout above the configured
           ``workflow_agent_call_timeout_s`` floor, never lower it. A floor of
           ``0`` means unlimited → a large finite backstop is used instead.
        2. The total-workflow deadline is authoritative: when a total deadline
           is in effect the resolved timeout is capped so one late agent()
           cannot outlive the remaining run budget.
        """
        default_timeout = _settings_int(
            "workflow_agent_call_timeout_s", AGENT_CALL_TIMEOUT_S
        )
        # Configured floor (0 => unlimited => finite backstop).
        base_s = default_timeout if default_timeout > 0 else AGENT_UNLIMITED_BACKSTOP_S

        raw_timeout = params.get("timeout")
        try:
            requested_s = int(raw_timeout) if raw_timeout is not None else 0
        except (TypeError, ValueError):
            requested_s = 0
        # Script value may only raise the effective timeout above the floor.
        effective_s = max(base_s, requested_s) if requested_s > 0 else base_s

        # When an output_schema is specified, the Python executor may perform
        # up to SCHEMA_RETRY_MAX additional prompts on the same session. The JS
        # watchdog timer must account for this total wall time, not just a single
        # prompt round-trip.
        if params.get("schema"):
            effective_s = effective_s * (1 + SCHEMA_RETRY_MAX)

        capped = dict(params)

        remaining = self._remaining_workflow_budget_s()
        if remaining is None:
            # Unlimited total deadline — no budget cap, floor logic stands.
            capped["timeout"] = max(1, effective_s)
            return capped

        budget_s = int(max(1.0, remaining - WORKFLOW_TIMEOUT_HEADROOM_S))
        capped["timeout"] = max(1, min(effective_s, budget_s))
        return capped

    def _handle_agent_call(self, params: dict[str, Any], request_id: Any) -> None:
        """Submit an agent call to the thread pool and send response on completion."""
        executor = self._executor
        if executor is None or self._shutdown_requested():
            self._send_error_response(
                request_id,
                code=-32603,
                message="Executor not available",
            )
            return

        if self._on_agent_call is None:
            self._send_error_response(
                request_id,
                code=-32603,
                message="No agent call handler configured",
            )
            return

        if self._reject_if_workflow_budget_exhausted(request_id):
            return
        params = self._cap_agent_timeout_to_remaining_budget(params)

        # Backpressure layer 1: reject if inbound JS→Python queue is full.
        # This preserves the historical MAX_QUEUE_SIZE ceiling used by
        # regression tests and prevents a runaway JS parallel() from
        # saturating the bridge loop.
        with self._msg_condition:
            inbound_depth = len(self._msg_queue)
        if inbound_depth >= MAX_QUEUE_SIZE:
            logger.warning(
                "[RuntimeBridge] backpressure rejecting agent_call: inbound queue full (%d >= %d)",
                inbound_depth,
                MAX_QUEUE_SIZE,
            )
            self._send_error_response(
                request_id,
                code=-32000,
                message=("Queue backpressure: too many pending messages, retry later"),
            )
            return

        # Backpressure layer 2: reject if the executor pool is overwhelmed.
        # ``_active_futures`` tracks submitted-but-in-flight futures. The pool's
        # own queue length is bounded by ``_max_concurrent * 2`` so transient bursts
        # still succeed while sustained floods are throttled.
        with self._futures_lock:
            active_count = len(self._active_futures)
        pending_response_pressure = active_count
        # Cap at 2x the pool size so a transient burst does not reject
        # valid work, but a sustained flood is throttled.
        pressure_cap = max(2, self._max_concurrent * 2)
        if pending_response_pressure >= pressure_cap:
            logger.warning(
                "[RuntimeBridge] backpressure rejecting agent_call (active=%d, cap=%d)",
                active_count,
                pressure_cap,
            )
            self._send_error_response(
                request_id,
                code=-32000,
                message=("Queue backpressure: too many pending agent calls, retry later"),
            )
            return

        def _execute() -> None:
            try:
                # Parse params into model
                agent_params = AgentCallParams.model_validate(params)
                # Resolve empty tool to the first allowed tool
                if not agent_params.tool and self._allowed_tools:
                    agent_params.tool = self._allowed_tools[0]
                # Get the per-call cancel event (set by _handle_abort_request)
                with self._request_cancel_events_lock:
                    call_cancel_event = self._request_cancel_events.get(request_id)
                # Execute the callback — pass per-call cancel_event if supported
                try:
                    result = self._on_agent_call(
                        agent_params,
                        cancel_event=call_cancel_event,
                        request_id=request_id,
                        deadline_monotonic=self._workflow_deadline_monotonic,
                    )
                except TypeError:
                    # Backward compat: callback only accepts (params, cancel_event) or just params
                    try:
                        result = self._on_agent_call(
                            agent_params,
                            cancel_event=call_cancel_event,
                        )
                    except TypeError:
                        result = self._on_agent_call(agent_params)
                # Build response payload
                response_data: dict[str, Any] = {}
                if result.parsed is not None:
                    response_data["data"] = result.parsed
                elif result.output is not None:
                    response_data["data"] = result.output
                if result.error:
                    response_data["error"] = result.error

                response_data["token_usage"] = result.token_usage
                response_data["duration_s"] = result.duration_s

                self._send_response(request_id, response_data)

            except Exception as exc:
                logger.exception("Agent call failed for request %s", request_id)
                self._send_error_response(
                    request_id,
                    code=-32603,
                    message=sanitize_for_reply(str(exc), ErrorCategory.INTERNAL_ERROR),
                )
            finally:
                # Unregister request future and cancel_event mappings
                with self._request_futures_lock:
                    self._request_futures.pop(request_id, None)
                with self._request_cancel_events_lock:
                    self._request_cancel_events.pop(request_id, None)

        # Create per-call cancel event (child of global cancel_event)
        # Setting the global event should also trigger per-call checks, but we
        # use a dedicated event for precise single-call cancellation (e.g. race).
        per_call_cancel = threading.Event()
        with self._request_cancel_events_lock:
            self._request_cancel_events[request_id] = per_call_cancel

        with self._futures_lock:
            if (
                self._shutdown_requested()
                or self._executor is not executor
            ):
                future = None
            else:
                future = executor.submit(_execute)
                self._active_futures.add(future)
        if future is None:
            with self._request_cancel_events_lock:
                self._request_cancel_events.pop(request_id, None)
            self._send_error_response(
                request_id,
                code=-32603,
                message="Executor not available",
            )
            return
        with self._request_futures_lock:
            self._request_futures[request_id] = future
        future.add_done_callback(lambda f: self._discard_future(f))

    # ------------------------------------------------------------------
    # Internal: Sub-workflow call
    # ------------------------------------------------------------------

    def _handle_workflow_call(self, params: dict[str, Any], request_id: Any) -> None:
        """Handle a workflow() call from the JS runtime (sub-workflow nesting).

        Max nesting depth is controlled by constants.MAX_NESTING_DEPTH.
        Sub-workflows run as a new RuntimeBridge subprocess with the same
        agent_call handler.
        Accepts either `script_path` (absolute/relative file) or `name` (template name).
        """
        if self._shutdown_requested():
            self._send_error_response(
                request_id,
                code=-32603,
                message="Workflow cancelled",
            )
            return
        if self._nesting_depth >= MAX_NESTING_DEPTH:
            self._send_error_response(
                request_id,
                code=-32603,
                message=f"Sub-workflow nesting depth exceeded (max={MAX_NESTING_DEPTH})",
            )
            return

        # SECURITY: Reject raw script_path from the runtime entirely. All
        # sub-workflow resolution MUST go through validate_template_name +
        # resolve_template_path so that user, project, global (allowlisted) global,
        # and built-in scopes are enforced uniformly. Accepting script_path would
        # bypasses those checks and could allow path traversal / scope bypasses.
        if params.get("script_path"):
            self._send_error_response(
                request_id,
                code=-32602,
                message=(
                    "Sub-workflow invocation by raw script_path is forbidden. Use workflow('<template_name') instead."
                ),
            )
            return

        name = params.get("name", "")
        if not name or not isinstance(name, str) or not name.strip():
            self._send_error_response(
                request_id,
                code=-32602,
                message="workflow() requires a 'name' argument",
            )
            return

        from .templates import (
            resolve_template_path,
            validate_template_name,
        )

        # 名称合法性校验（拒绝 '/' '\\' '..' 与绝对路径形式)
        name = name.strip()
        ok, err = validate_template_name(name)
        if not ok:
            self._send_error_response(
                request_id,
                code=-32602,
                message=f"Invalid sub-workflow name: {err}",
            )
            return

        # 统一解析：user > project > (allowlisted) global > builtin
        # resolve_template_path 在 global 查找时内部会校验 WORKFLOW_GLOBAL_TEMPLATE_ALLOWLIST。
        resolved = resolve_template_path(
            self._cwd,
            name,
            user_id=self._initiator_user_id,
        )
        if not resolved:
            self._send_error_response(
                request_id,
                code=-32602,
                message=f"Sub-workflow template not found: {name}",
            )
            return

        script_path = resolved

        # --- Sub-workflow tool-allowlist preflight --------------------------
        # Read the template's meta block and ensure every tool it declares is
        # present in the parent's allowed_tools. Failing here (before any
        # sub-process is spawned) keeps the failure cheap and gives handlers a
        # structured payload to surface in the confirmation card.
        sub_meta_tools: list[str] = []
        sub_description: str | None = None
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                sub_content = f.read()
        except OSError as exc:
            self._send_error_response(
                request_id,
                code=-32003,
                message=f"Failed to read sub-workflow template '{name}': {exc}",
            )
            return

        try:
            from .script_gen import extract_meta_from_script

            parsed = extract_meta_from_script(sub_content) or {}
            sub_meta_tools = list(parsed.get("tools", []) or [])
            sub_description = parsed.get("description")
        except Exception as exc:  # noqa: BLE001  — fail-safe, not a security issue
            logger.warning(
                "[RuntimeBridge] Failed to parse sub-workflow meta for %s: %s",
                name,
                exc,
            )
            parsed = {}
            sub_meta_tools = []

        missing_tools = sorted({t for t in sub_meta_tools if t not in set(self._allowed_tools or [])})
        if missing_tools:
            self._send_error_response(
                request_id,
                code=-32004,
                message=(
                    f"Sub-workflow '{name}' requires tools not present in "
                    f"the parent allowlist: missing={missing_tools}; "
                    f"allowed={sorted(self._allowed_tools or [])}"
                ),
                structured={
                    "kind": "missing_tools",
                    "name": name,
                    "description": sub_description,
                    "missing_tools": missing_tools,
                    "allowed_tools": sorted(self._allowed_tools or []),
                },
            )
            return

        # Extract args to pass to sub-workflow
        sub_args = params.get("args", {})

        def _execute_sub_workflow() -> None:
            try:
                if self._shutdown_requested():
                    raise RuntimeError("Workflow cancelled")
                # Independent cancel_event for sub-workflow isolation
                # Parent will cascade set() on stop()
                sub_cancel_event = threading.Event()

                sub_bridge = RuntimeBridge(
                    script_path=script_path,
                    cwd=self._cwd,
                    max_concurrent=self._max_concurrent,
                    on_agent_call=self._on_agent_call,
                    on_agent_aborted=self._on_agent_aborted,
                    on_phase=self._on_phase,
                    on_log=self._on_log,
                    cancel_event=sub_cancel_event,
                    allowed_tools=self._allowed_tools,
                    nesting_depth=self._nesting_depth + 1,
                    args=sub_args,
                    initiator_user_id=self._initiator_user_id,
                )
                sub_bridge._workflow_started_monotonic = self._workflow_started_monotonic
                sub_bridge._workflow_deadline_monotonic = self._workflow_deadline_monotonic
                sub_bridge._workflow_started_unix_ms = self._workflow_started_unix_ms
                sub_bridge._workflow_deadline_unix_ms = self._workflow_deadline_unix_ms
                # Link parent-child for cascade cancellation
                sub_bridge._parent = self
                with self._children_lock:
                    if self._shutdown_requested():
                        raise RuntimeError("Workflow cancelled")
                    self._children.append(sub_bridge)

                try:
                    sub_bridge.start()
                    result = sub_bridge.run()
                    response_data: dict[str, Any] = {"data": result}
                    self._send_response(request_id, response_data)
                finally:
                    sub_bridge.stop()
                    sub_bridge.wait_for_workers()
                    with self._children_lock:
                        if sub_bridge in self._children:
                            self._children.remove(sub_bridge)
            except Exception as exc:
                logger.exception("Sub-workflow call failed for request %s", request_id)
                self._send_error_response(
                    request_id,
                    code=-32603,
                    message=sanitize_for_reply(str(exc), ErrorCategory.INTERNAL_ERROR),
                )

        workflow_executor = self._workflow_executor
        if workflow_executor is None or self._shutdown_requested():
            self._send_error_response(
                request_id,
                code=-32603,
                message="Executor not available for sub-workflow",
            )
            return

        with self._futures_lock:
            if (
                self._shutdown_requested()
                or self._workflow_executor is not workflow_executor
            ):
                future = None
            else:
                future = workflow_executor.submit(_execute_sub_workflow)
                self._active_futures.add(future)
        if future is None:
            self._send_error_response(
                request_id,
                code=-32603,
                message="Executor not available for sub-workflow",
            )
            return
        future.add_done_callback(lambda f: self._discard_future(f))

    # ------------------------------------------------------------------
    # Internal: Transport (stdin writes — must be thread-safe)
    # ------------------------------------------------------------------

    def _send(self, msg: dict[str, Any]) -> None:
        """Write a JSON-RPC message to the subprocess stdin (thread-safe)."""
        if self._process is None or self._process.stdin is None:
            return

        line = json.dumps(msg, separators=(",", ":")) + "\n"

        with self._write_lock:
            try:
                self._process.stdin.write(line)
                self._process.stdin.flush()
            except (OSError, BrokenPipeError, ValueError) as exc:
                logger.debug("Failed to write to stdin: %s", repr(exc))
                # Process died or pipe broken — signal run() loop to exit
                if not self._done:
                    self._error = f"Runtime connection lost: {exc}"
                    self._done = True

    def _send_response(self, request_id: Any, result: Any) -> None:
        """Send a JSON-RPC success response."""
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }
        )

    def _send_error_response(
        self,
        request_id: Any,
        code: int,
        message: str,
        structured: dict[str, Any] | None = None,
    ) -> None:
        """Send a JSON-RPC error response.

        Args:
            request_id: JSON-RPC request id.
            code: Numeric error code.
            message: Human-readable message.
            structured: Optional machine-readable payload surfaced in the
                ``data`` field so callers (handlers, tests) can branch on
                specific failure kinds (e.g. missing tools in sub-workflows).
        """
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        if structured:
            payload["error"]["data"] = structured
        self._send(payload)

    def _send_cancel(self) -> None:
        """Send cancel notification to the JS runtime."""
        self._send(
            {
                "jsonrpc": "2.0",
                "method": "cancel",
                "params": {},
            }
        )

    # ------------------------------------------------------------------
    # Internal: Future tracking
    # ------------------------------------------------------------------

    def _discard_future(self, future: Future) -> None:
        """Remove a completed future from the active set (done callback)."""
        with self._futures_lock:
            self._active_futures.discard(future)

    def _handle_abort_request(self, request_id: Any) -> None:
        """Handle an abort_request notification from the JS runtime.

        Sets the per-call cancel_event to interrupt the in-flight ACP session
        (e.g. for race() loser cancellation). Also cancels the future if not
        yet started. The per-call cancel_event ensures the session's
        send_prompt loop exits within its poll interval (typically ~1s),
        allowing the session to close cleanly and quickly.

        Uses find-and-set for the cancel_event (not pop) so the worker
        thread can still retrieve it from ``_request_cancel_events``.
        This eliminates a race where an early abort would pop the event
        before the worker retrieved it, causing the worker to fall back
        to the global cancel_event and ignore the per-call abort.
        """
        # Set the per-call cancel event first — this is the fast path for
        # interrupting in-flight sessions that poll cancel_event.
        # Use get (not pop) — the worker thread may not have retrieved it yet.
        with self._request_cancel_events_lock:
            call_cancel = self._request_cancel_events.get(request_id)
        if call_cancel is not None:
            call_cancel.set()

        with self._request_futures_lock:
            future = self._request_futures.pop(request_id, None)
        if future is None:
            return
        if not future.cancel():
            # Future is already running — the per-call cancel_event above
            # should interrupt the session's send_prompt loop within its
            # poll interval. The session close in the finally block will
            # clean up resources once the call returns.
            logger.debug("Abort request %s: future already running, per-call cancel_event set", request_id)
        else:
            # Future was cancelled before it started running. The _execute
            # function will never execute, so its finally block won't clean
            # up the cancel_event entry. Do it here to prevent leaks.
            with self._request_cancel_events_lock:
                self._request_cancel_events.pop(request_id, None)

    @property
    def in_flight_count(self) -> int:
        """Count of submitted-but-not-yet-completed futures (thread-safe)."""
        with self._futures_lock:
            return len(self._active_futures)

    # ------------------------------------------------------------------
    # Internal: Message queue helpers
    # ------------------------------------------------------------------

    def _pop_message(self, timeout: float = 1.0) -> Optional[dict[str, Any]]:
        """Pop the next message from the queue, blocking up to timeout."""
        with self._msg_condition:
            if not self._msg_queue:
                self._msg_condition.wait(timeout=timeout)
            if self._msg_queue:
                return self._msg_queue.popleft()
        return None

    def _drain_pending_messages(self) -> None:
        """Dispatch every message still queued, without blocking.

        Called on the run() exit paths after the process has died / stdout has
        closed. Its purpose is to apply a terminal done/error frame that the
        reader thread queued in the instant before EOF — so a genuine result
        wins over the synthetic "process exited unexpectedly" fallback.

        Dispatching remaining ``agent_call`` requests here is harmless: the
        executor callback will fail fast because the subprocess (and thus the
        stdin write path) is gone, and _send() swallows the BrokenPipe. We stop
        as soon as a terminal frame is applied to avoid unnecessary work.
        """
        while True:
            with self._msg_condition:
                if not self._msg_queue:
                    return
                msg = self._msg_queue.popleft()
            try:
                self._dispatch_message(msg)
            except Exception:
                logger.debug("Error dispatching queued message during drain", exc_info=True)
            # A terminal frame sets _result or _error; once we have either we
            # can stop — nothing after a done/error frame is meaningful.
            if self._result is not None or self._error is not None:
                return

    def _wait_for_notification(self, method: str, timeout: float = 30.0) -> Optional[dict[str, Any]]:
        """Wait for a specific notification method, with timeout.

        Messages that don't match are re-queued.
        """
        deadline = time.monotonic() + timeout
        stash: list[dict[str, Any]] = []

        while time.monotonic() < deadline:
            if self._shutdown_requested():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            msg = self._pop_message(timeout=min(0.5, remaining))
            if msg is None:
                # Check if process died while waiting
                if self._process and self._process.poll() is not None:
                    break
                continue

            if msg.get("method") == method:
                # Put stashed messages back (prepend to front of deque)
                if stash:
                    with self._msg_condition:
                        self._msg_queue.extendleft(reversed(stash))
                        self._msg_condition.notify()
                return msg

            stash.append(msg)

        # Timed out — put stashed messages back
        if stash:
            with self._msg_condition:
                self._msg_queue.extendleft(reversed(stash))
                self._msg_condition.notify()

        return None

    # ------------------------------------------------------------------
    # Internal: Process management
    # ------------------------------------------------------------------

    def _kill_process(self) -> None:
        """Terminate and clean up the subprocess."""
        with self._process_lock:
            if self._process is None:
                return

            try:
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=5)
            except OSError:
                pass

            # Close handles
            for stream in (
                self._process.stdin,
                self._process.stdout,
                self._process.stderr,
            ):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass

            self._process = None

    def _describe_unexpected_exit(self) -> str:
        """Build a diagnostic error string for an unexpected runtime exit.

        The Node runtime now flushes stdout before exiting, so a bare stdout
        EOF here means the process died *without* emitting a terminal frame —
        e.g. SIGKILL/OOM, a native crash, or a truncated write. Include the
        exit/signal code and the buffered stderr tail so the real cause is not
        masked by a generic message. The reader thread may observe EOF a beat
        before ``poll()`` reaps the child, so briefly wait for the return code.
        """
        returncode: Optional[int] = None
        proc = self._process
        if proc is not None:
            rc = proc.poll()
            if rc is None:
                # Give the process a moment to be reaped so we can report the
                # exit/signal code rather than "still running".
                try:
                    rc = proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, OSError, ValueError):
                    rc = proc.poll()
            # Coerce defensively: a real Popen yields int|None, but tests (and
            # exotic wrappers) may hand back a non-int — never let the
            # diagnostic itself raise.
            if rc is not None:
                try:
                    returncode = int(rc)
                except (TypeError, ValueError):
                    returncode = None

        detail = "Runtime process closed stdout unexpectedly"
        if returncode is not None:
            if returncode < 0:
                detail += f" (terminated by signal {-returncode})"
            else:
                detail += f" (exit code {returncode})"

        stderr_text = _strip_internal_details(self._stderr_diagnostic())
        if stderr_text:
            detail += f". stderr: {stderr_text}"
        return detail

    def _stderr_reader(self) -> None:
        """Drain stderr continuously to prevent pipe buffer deadlock.

        Each line is logged at DEBUG and also appended to a bounded ring
        buffer (``_stderr_tail``) so the most recent stderr output survives
        for the unexpected-exit diagnostic even though the pipe itself is
        consumed here.
        """
        assert self._process is not None
        assert self._process.stderr is not None
        try:
            while True:
                line = self._process.stderr.readline()
                if not line:
                    break  # EOF — process closed stderr
                stripped = line.rstrip()
                logger.debug("[runtime stderr] %s", stripped)
                if stripped:
                    with self._stderr_tail_lock:
                        self._stderr_tail.append(stripped)
        except (OSError, ValueError):
            pass

    def _stderr_tail_text(self) -> str:
        """Return the buffered tail of Node stderr (most recent lines).

        Used by the unexpected-exit diagnostics: the continuous drain thread
        has already emptied the stderr pipe, so a one-shot :meth:`_drain_stderr`
        typically returns nothing. This tail preserves the real dying words.
        """
        with self._stderr_tail_lock:
            return "\n".join(self._stderr_tail).strip()

    def _stderr_diagnostic(self) -> str:
        """Best-effort combined stderr: buffered tail plus any pipe remainder."""
        parts: list[str] = []
        tail = self._stderr_tail_text()
        if tail:
            parts.append(tail)
        remainder = self._drain_stderr()
        # Defensive: _drain_stderr is typed -> str, but guard against test
        # mocks / exotic streams returning a non-str so the diagnostic path
        # can never itself raise.
        if isinstance(remainder, str) and remainder and remainder not in tail:
            parts.append(remainder)
        return "\n".join(parts).strip()

    def _drain_stderr(self) -> str:
        """Read remaining stderr content from the subprocess."""
        if self._process is None or self._process.stderr is None:
            return ""
        try:
            # Non-blocking read of whatever is available
            content = self._process.stderr.read()
            return content.strip() if content else ""
        except (OSError, ValueError):
            return ""
