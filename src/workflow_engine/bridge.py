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
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .constants import (
    AGENT_CALL_TIMEOUT_S,
    AGENT_UNLIMITED_BACKSTOP_S,
    DEFAULT_MAX_CONCURRENT,
    HARD_MAX_CONCURRENT,
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

STDERR_LINE_MAX_BYTES = 4 * 1024
STDERR_TOTAL_MAX_BYTES = 64 * 1024
STDERR_TRUNCATION_MARKER = "...[stderr output truncated]"


def _clip_utf8(text: str, max_bytes: int, marker: str = STDERR_TRUNCATION_MARKER) -> tuple[str, bool]:
    encoded = str(text or "").encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return str(text or ""), False
    marker_bytes = marker.encode("utf-8")
    keep = max(0, max_bytes - len(marker_bytes))
    return encoded[:keep].decode("utf-8", errors="ignore") + marker, True


def _bound_stderr_text(text: str) -> tuple[str, bool]:
    lines: list[str] = []
    truncated = False
    for raw_line in str(text or "").splitlines():
        line, line_truncated = _clip_utf8(raw_line, STDERR_LINE_MAX_BYTES)
        lines.append(line)
        truncated = truncated or line_truncated
    bounded, total_truncated = _clip_utf8(
        "\n".join(lines),
        STDERR_TOTAL_MAX_BYTES,
        "\n" + STDERR_TRUNCATION_MARKER,
    )
    return bounded, truncated or total_truncated


@dataclass(slots=True)
class _PendingRequest:
    cancel_event: threading.Event
    future: Future | None = None


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
        args: Optional[dict[str, Any]] = None,
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
        self._args = args or {}
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
        self._stderr_tail_bytes = 0
        self._stderr_tail_truncated = False
        self._stderr_tail_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

        # Incoming message queue (for the run() loop)
        self._msg_queue: collections.deque[dict[str, Any]] = collections.deque()
        self._msg_condition = threading.Condition()

        # ThreadPoolExecutor for agent calls
        self._executor: Optional[ThreadPoolExecutor] = None

        self._active_futures: set[Future] = set()
        self._requests: dict[Any, _PendingRequest] = {}
        self._request_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

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
        # The runtime is executable product code. Resolve it from the installed
        # package, never from the untrusted user project used as subprocess cwd.
        runtime_path = os.path.realpath(RUNTIME_JS_PATH)
        if not os.path.isfile(runtime_path):
            raise RuntimeError(
                "packaged Workflow runtime is missing; reinstall GhostAP"
            )

        node_bin = shutil.which("node")
        if not node_bin:
            raise RuntimeError("Node.js not found in PATH")

        cmd = [node_bin, runtime_path, self._script_path]
        logger.info("Starting Node.js runtime: %s", " ".join(cmd))

        with self._process_lock:
            if self._process is not None:
                raise RuntimeError("RuntimeBridge already started")
            if self._shutdown_requested():
                raise RuntimeError("RuntimeBridge cancelled before start")
            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self._cwd,
                    text=True,
                    bufsize=1,  # Line-buffered
                    env=os.environ.copy(),
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
        with self._request_lock:
            if self._shutdown_requested():
                raise RuntimeError("RuntimeBridge cancelled during start")
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_concurrent,
                thread_name_prefix="RuntimeBridge-agent",
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

        Also shuts down the agent-call executor. Idempotent — calling it more than once is
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

        # Signal all in-flight agent calls to cancel before killing the
        # process.  Each per-call cancel_event interrupts the ACP session's
        # send_prompt via the cancel-guard thread, so in-flight agents stop
        # within the guard poll interval (~200ms) rather than running to
        # their full 300s timeout.
        with self._request_lock:
            requests = tuple(self._requests.values())
        for request in requests:
            request.cancel_event.set()

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
        with self._request_lock:
            futures_to_cancel = list(self._active_futures)
            executor = self._executor
            self._executor = None
            self._requests.clear()

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
        self._shutdown_done = True
        logger.info("RuntimeBridge stopped")

    def wait_for_workers(self, timeout: float | None = None) -> bool:
        """Wait until every submitted agent callback is quiet.

        ``stop()`` intentionally remains non-blocking for responsive user
        cancellation. The execution owner calls this method before releasing
        its WorkflowEngine run claim, so callbacks from a retired bridge can
        never observe the component fields of a later run.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            with self._request_lock:
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
        else:
            self._reject(request_id, f"Unknown method: {method}", code=-32601)

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

        self._reject(
            request_id,
            f"Workflow deadline exhausted before starting agent call (remaining={max(0.0, remaining):.1f}s)",
            code=-32002,
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
        default_timeout = self._workflow_agent_call_timeout_s
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
            self._reject(request_id, "Executor not available")
            return

        if self._on_agent_call is None:
            self._reject(request_id, "No agent call handler configured")
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
            self._reject(
                request_id,
                "Queue backpressure: too many pending messages, retry later",
                code=-32000,
            )
            return

        # Backpressure layer 2: reject if the executor pool is overwhelmed.
        # ``_active_futures`` tracks submitted-but-in-flight futures. The pool's
        # own queue length is bounded by ``_max_concurrent * 2`` so transient bursts
        # still succeed while sustained floods are throttled.
        with self._request_lock:
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
            self._reject(
                request_id,
                "Queue backpressure: too many pending agent calls, retry later",
                code=-32000,
            )
            return

        pending = _PendingRequest(threading.Event())

        def _execute() -> None:
            try:
                # Parse params into model
                agent_params = AgentCallParams.model_validate(params)
                # Resolve empty tool to the first allowed tool
                if not agent_params.tool and self._allowed_tools:
                    agent_params.tool = self._allowed_tools[0]
                result = self._on_agent_call(
                    agent_params,
                    cancel_event=pending.cancel_event,
                    request_id=request_id,
                    deadline_monotonic=self._workflow_deadline_monotonic,
                )
                # Build response payload
                response_data: dict[str, Any] = {}
                if result.parsed is not None:
                    response_data["data"] = result.parsed
                elif result.output is not None:
                    response_data["data"] = result.output
                if result.error:
                    response_data["error"] = result.error

                response_data["agent_id"] = result.agent_id
                response_data["tool"] = result.tool
                response_data["model"] = result.model
                response_data["token_usage"] = result.token_usage
                response_data["duration_s"] = result.duration_s

                self._respond(request_id, result=response_data)

            except Exception as exc:
                logger.exception("Agent call failed for request %s", request_id)
                self._reject(
                    request_id,
                    sanitize_for_reply(str(exc), ErrorCategory.INTERNAL_ERROR),
                )
            finally:
                with self._request_lock:
                    if self._requests.get(request_id) is pending:
                        self._requests.pop(request_id, None)

        with self._request_lock:
            if self._shutdown_requested() or self._executor is not executor:
                future = None
            else:
                self._requests[request_id] = pending
                try:
                    future = executor.submit(_execute)
                except RuntimeError:
                    self._requests.pop(request_id, None)
                    future = None
                else:
                    pending.future = future
                    self._active_futures.add(future)
        if future is None:
            self._reject(request_id, "Executor not available")
            return
        future.add_done_callback(
            lambda completed: self._finish_request(request_id, pending, completed)
        )

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

    def _respond(
        self,
        request_id: Any,
        *,
        result: Any,
    ) -> None:
        self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _reject(self, request_id: Any, message: str, *, code: int = -32603) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )

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

    def _finish_request(
        self,
        request_id: Any,
        pending: _PendingRequest,
        future: Future,
    ) -> None:
        with self._request_lock:
            self._active_futures.discard(future)
            if future.cancelled() and self._requests.get(request_id) is pending:
                self._requests.pop(request_id, None)

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
        with self._request_lock:
            pending = self._requests.get(request_id)
        if pending is None:
            return
        pending.cancel_event.set()
        future = pending.future
        if future is not None and future.cancel():
            with self._request_lock:
                if self._requests.get(request_id) is pending:
                    self._requests.pop(request_id, None)

    @property
    def in_flight_count(self) -> int:
        """Count of submitted-but-not-yet-completed futures (thread-safe)."""
        with self._request_lock:
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
                stripped, line_truncated = _clip_utf8(line.rstrip(), STDERR_LINE_MAX_BYTES)
                logger.debug("[runtime stderr] %s", stripped)
                if stripped:
                    with self._stderr_tail_lock:
                        encoded_bytes = len(stripped.encode("utf-8", errors="replace"))
                        while self._stderr_tail and (
                            len(self._stderr_tail) >= (self._stderr_tail.maxlen or STDERR_TAIL_MAX_LINES)
                            or self._stderr_tail_bytes + encoded_bytes + 1 > STDERR_TOTAL_MAX_BYTES
                        ):
                            removed = self._stderr_tail.popleft()
                            self._stderr_tail_bytes -= len(
                                removed.encode("utf-8", errors="replace")
                            )
                            if self._stderr_tail:
                                self._stderr_tail_bytes = max(0, self._stderr_tail_bytes - 1)
                            self._stderr_tail_truncated = True
                        self._stderr_tail.append(stripped)
                        self._stderr_tail_bytes += encoded_bytes + (
                            1 if len(self._stderr_tail) > 1 else 0
                        )
                        self._stderr_tail_truncated = (
                            self._stderr_tail_truncated or line_truncated
                        )
        except (OSError, ValueError):
            pass

    def _stderr_tail_text(self) -> str:
        """Return the buffered tail of Node stderr (most recent lines).

        Used by the unexpected-exit diagnostics: the continuous drain thread
        has already emptied the stderr pipe, so a one-shot :meth:`_drain_stderr`
        typically returns nothing. This tail preserves the real dying words.
        """
        with self._stderr_tail_lock:
            text = "\n".join(self._stderr_tail).strip()
            truncated = self._stderr_tail_truncated
        if truncated:
            text = f"{STDERR_TRUNCATION_MARKER}\n{text}" if text else STDERR_TRUNCATION_MARKER
        bounded, _ = _bound_stderr_text(text)
        return bounded

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
        bounded, _ = _bound_stderr_text("\n".join(parts).strip())
        return bounded

    def _drain_stderr(self) -> str:
        """Read remaining stderr content from the subprocess."""
        if self._process is None or self._process.stderr is None:
            return ""
        try:
            # Non-blocking read of whatever is available
            content = self._process.stderr.read(STDERR_TOTAL_MAX_BYTES + 1)
            if not content:
                return ""
            bounded, _ = _bound_stderr_text(content.strip())
            return bounded
        except (OSError, ValueError):
            return ""
