"""AgentExecutor — executes a single agent() call via ACP/CLI session."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from collections.abc import Mapping
from typing import Any, Callable, Optional

from src.card.tool_display import sanitize_tool_failure_detail

from .constants import (
    AGENT_CALL_TIMEOUT_S,
    AGENT_IDLE_TIMEOUT_S,
    AGENT_UNLIMITED_BACKSTOP_S,
    DEFAULT_MAX_CONCURRENT,
    HARD_MAX_CONCURRENT,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE_S,
    SCHEMA_RETRY_MAX,
    SESSION_CREATE_TIMEOUT_S,
    WORKFLOW_TIMEOUT_HEADROOM_S,
)
from .errors import _strip_internal_details, is_transient_error
from .models import AgentCallParams, AgentCallResult
from .roles import get_subagent_encouragement_prompt

logger = logging.getLogger(__name__)


_SUBAGENT_STATUS_MAP = {
    "pendinginit": "running",
    "pending": "running",
    "running": "running",
    "completed": "completed",
    "shutdown": "completed",
    "errored": "failed",
    "failed": "failed",
    "notfound": "failed",
    "interrupted": "cancelled",
}
_SUBAGENT_STATUS_PROGRESS = {
    "running": "已启动",
    "completed": "已完成",
    "failed": "执行失败",
    "cancelled": "已中断",
}
_SUBAGENT_ACTIVITY = {
    "started": ("running", "已启动"),
    "interacted": ("running", "已与主 Agent 交互"),
}


def _is_subagent_tool_call(tool_call: Any) -> bool:
    if tool_call is None:
        return False
    return bool(
        getattr(tool_call, "subagent_source_id", None)
        or getattr(tool_call, "subagent_activity", None)
        or getattr(tool_call, "collaboration_tool", None)
        or getattr(tool_call, "collaboration_receivers", ())
        or getattr(tool_call, "subagent_states", ())
    )


def _safe_subagent_progress(value: object, *, opaque_ids: tuple[str, ...]) -> str:
    text = _strip_internal_details(str(value or "").strip())
    if not text:
        return ""
    return sanitize_tool_failure_detail(
        text,
        fallback="",
        max_chars=180,
        opaque_ids=opaque_ids,
    )


def _subagent_updates_from_tool_call(
    tool_call: Any,
) -> tuple[dict[str, object], ...]:
    """Normalize one ACP child-agent frame for Workflow state consumption."""
    if tool_call is None:
        return ()

    receivers = tuple(
        source_id
        for item in (getattr(tool_call, "collaboration_receivers", ()) or ())
        if (source_id := str(item or "").strip())
    )
    raw_states = getattr(tool_call, "subagent_states", ()) or ()
    states: dict[str, Mapping[str, object]] = {}
    for item in raw_states:
        if not isinstance(item, Mapping):
            continue
        source_id = str(item.get("source_id") or "").strip()
        if source_id:
            states[source_id] = item

    all_source_ids = tuple(dict.fromkeys((*receivers, *states.keys())))
    updates: dict[str, dict[str, object]] = {}

    source_id = str(getattr(tool_call, "subagent_source_id", None) or "").strip()
    activity = str(getattr(tool_call, "subagent_activity", None) or "").strip().lower()
    tool_status = str(getattr(tool_call, "status", None) or "").strip().lower()
    activity_status = (
        (
            ("cancelled", "已中断")
            if tool_status == "completed"
            else (
                ("running", "中断未完成")
                if tool_status == "failed"
                else ("running", "正在中断")
            )
        )
        if activity == "interrupted"
        else _SUBAGENT_ACTIVITY.get(activity)
    )
    if source_id and activity_status:
        status, progress = activity_status
        updates[source_id] = {
            "source_id": source_id,
            "status": status,
            "progress": progress,
            "model": None,
        }

    model_value = getattr(tool_call, "collaboration_model", None)
    model = str(model_value).strip() if model_value else None
    for child_source_id in all_source_ids:
        state = states.get(child_source_id)
        if state is None:
            if child_source_id not in updates:
                updates[child_source_id] = {
                    "source_id": child_source_id,
                    "status": "running",
                    "progress": "已启动",
                    "model": model,
                }
            continue

        raw_status = str(state.get("status") or "running").strip().lower()
        status = _SUBAGENT_STATUS_MAP.get(raw_status, "running")
        progress = _safe_subagent_progress(
            state.get("message"),
            opaque_ids=all_source_ids,
        ) or _SUBAGENT_STATUS_PROGRESS[status]
        updates[child_source_id] = {
            "source_id": child_source_id,
            "status": status,
            "progress": progress,
            "model": model,
        }

    return tuple(updates.values())


def _settings_int(field: str, fallback: int) -> int:
    """Read an int workflow-timeout setting, falling back to the constant.

    Lets .env overrides take effect at runtime while never breaking the
    executor if config is unavailable/invalid.
    """
    try:
        from src.config import get_settings

        return int(getattr(get_settings(), field, fallback))
    except Exception:  # pragma: no cover - defensive: config not importable
        return fallback


def _is_timeout_in_chain(exc: BaseException) -> bool:
    """Walk __cause__/__context__ chain looking for TimeoutError."""
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError):
            return True
        if current.__class__.__name__ == "TimeoutError":
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


def _is_startup_error(exc: BaseException) -> bool:
    """Return True if the exception is an ACP startup failure (process won't start)."""
    if exc.__class__.__name__ == "ACPStartupError":
        return True
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if current.__class__.__name__ == "ACPStartupError":
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


class AgentExecutor:
    """Executes individual agent() calls via ACP/CLI sessions.

    Each execute() call creates a short-lived, one-shot session: the session is
    opened, the prompt is sent, the result is collected, and the session is closed.
    Failures are isolated — exceptions never propagate out of execute().
    """

    def __init__(
        self,
        cwd: str,
        cancel_event: threading.Event,
        on_token_usage: Optional[Callable[[int], None]] = None,
        on_activity: Optional[Callable[[str, str], None]] = None,
        max_workers: int = DEFAULT_MAX_CONCURRENT,
        # Deprecated: kept for backwards compatibility
        budget_total: Optional[int] = None,
        on_budget_exceeded: Optional[Callable[[], None]] = None,
        on_subagent_update: Optional[
            Callable[[str, tuple[dict[str, object], ...]], None]
        ] = None,
    ) -> None:
        self.cwd = cwd
        self.cancel_event = cancel_event
        self.on_token_usage = on_token_usage
        self.on_activity = on_activity  # (label, activity_text) -> None
        self.on_subagent_update = on_subagent_update
        # Deprecated parameters - kept for backwards compatibility but ignored
        del budget_total, on_budget_exceeded
        # Shared thread pool for session creation — avoids per-call pool overhead.
        # Size is capped by HARD_MAX_CONCURRENT to prevent runaway concurrency even
        # when callers pass an explicit value.
        pool_size = max(1, min(int(max_workers), HARD_MAX_CONCURRENT))
        self._session_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=pool_size, thread_name_prefix="wf_session"
        )
        self._shutdown_done = False
        # Tracking for late-close threads so shutdown() can wait for them.
        # Without this, daemon threads may be abandoned at interpreter exit
        # and orphan ACP subprocesses.
        self._late_close_threads: list[threading.Thread] = []
        self._late_close_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        # Circuit breaker: tools whose ACP startup failed during this workflow
        # run are blacklisted so subsequent calls skip the 50s timeout penalty.
        self._startup_blacklist: dict[str, str] = {}  # tool -> error message
        self._blacklist_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        params: AgentCallParams,
        *,
        cancel_event: Optional[threading.Event] = None,
        deadline_monotonic: float | None = None,
    ) -> AgentCallResult:
        """Execute a single agent() call end-to-end with retry logic.

        1. Build the full prompt (role preamble + task + encouragement).
        2. Create a session via create_engine_session.
        3. Send prompt with timeout, collecting events.
        4. Validate output against schema if required (with retries).
        5. Retry transient errors with exponential backoff (MAX_RETRIES).
        6. Return AgentCallResult with output, token_usage, duration_s.
        7. On any exception, return AgentCallResult with error field set.

        Retry behavior:
        - Transient errors (network timeout, rate limit, etc.) are retried
          with exponential backoff: delay = RETRY_BACKOFF_BASE_S * 2^attempt
        - Permanent errors (invalid schema, permission denied) are not retried
        - Max retries: MAX_RETRIES (default 3)
        - Cancel event is checked before each retry attempt

        Args:
            params: Agent call parameters.
            cancel_event: Optional per-call cancellation event. When provided,
                it overrides the executor-level cancel_event for this specific
                call, allowing individual agent calls to be cancelled (e.g.
                race() loser abort) without affecting other concurrent calls.
        """
        start = time.monotonic()
        last_error: Optional[str] = None
        total_token_usage = 0
        # Per-call cancel event (may be None if caller doesn't provide one).
        # When provided, it is used for single-call cancellation (e.g. race
        # loser abort).  The executor-level cancel_event covers global stops.
        # Both are effective — the cancel guard and all poll loops treat them
        # with OR semantics so that either signal triggers cancellation.
        per_call_cancel_event = cancel_event
        global_cancel_event = self.cancel_event

        def _is_cancelled() -> bool:
            if global_cancel_event.is_set():
                return True
            if per_call_cancel_event is not None and per_call_cancel_event.is_set():
                return True
            return False

        def _remaining_deadline_s() -> float | None:
            if deadline_monotonic is None:
                return None
            return deadline_monotonic - time.monotonic()

        def _deadline_budget_s() -> int | None:
            remaining = _remaining_deadline_s()
            if remaining is None:
                return None
            if remaining <= WORKFLOW_TIMEOUT_HEADROOM_S:
                return 0
            return int(max(1.0, remaining - WORKFLOW_TIMEOUT_HEADROOM_S))

        def _effective_timeout_s(requested: float | int | None, fallback: float | int) -> float:
            """Resolve the effective per-call timeout in seconds.

            ``fallback`` is the authoritative host config value (from Settings /
            .env). It is treated as the *floor*, not a cap: the LLM-generated
            script frequently bakes a small ``timeout`` (e.g. 180) into each
            agent() call, and honoring that verbatim was killing legitimately
            long-running coding tasks. So the script value can only *raise* the
            timeout above the configured floor, never lower it.

            A configured value of ``<= 0`` means *unlimited*: we substitute a
            large but finite backstop (:data:`AGENT_UNLIMITED_BACKSTOP_S`) so
            the blocking call still eventually returns instead of hanging
            forever on an orphaned session. Real bounding in unlimited mode
            comes from the user's stop button, the total-workflow deadline (if
            any), and the MAX_TOTAL_AGENTS fuse.

            The result is always further capped by the remaining total-workflow
            budget when a total deadline is in effect. Sub-second values are
            preserved (fast tests rely on fractional timeouts); only ``<= 0`` is
            treated as the unlimited sentinel, never a small positive fraction.
            """
            # Configured floor (<= 0 => unlimited => finite backstop). Compare on
            # floats so a legitimate sub-second config (e.g. 0.01 in tests) is
            # NOT mistaken for the unlimited sentinel via int truncation.
            try:
                configured_s = float(fallback)
            except (TypeError, ValueError):
                configured_s = float(AGENT_CALL_TIMEOUT_S)
            base_s = float(AGENT_UNLIMITED_BACKSTOP_S) if configured_s <= 0 else configured_s

            # Script-requested value may only raise the effective timeout above
            # the configured floor.
            try:
                requested_s = float(requested) if requested is not None else 0.0
            except (TypeError, ValueError):
                requested_s = 0.0
            effective_s = max(base_s, requested_s) if requested_s > 0 else base_s

            budget_s = _deadline_budget_s()
            if budget_s is None:
                return effective_s
            return max(1.0, min(effective_s, float(budget_s)))

        # Pass a single event to session creation for backward compat; use
        # the per-call event if available, else the global one.  The OR
        # semantics are enforced by the bridge.stop() path which sets all
        # per-call events (see RuntimeBridge.stop()), and by the cancel
        # guard below which polls both events explicitly.
        call_cancel_event = per_call_cancel_event or global_cancel_event

        # Circuit breaker: skip tools whose ACP startup already failed in this run
        if params.tool:
            with self._blacklist_lock:
                blacklisted_reason = self._startup_blacklist.get(params.tool)
            if blacklisted_reason:
                logger.info(
                    "[AgentExecutor] Circuit breaker: skipping tool=%s (previous startup failure: %s)",
                    params.tool,
                    blacklisted_reason,
                )
                return AgentCallResult(
                    error=f"Circuit breaker: {params.tool} ACP startup failed earlier in this workflow ({blacklisted_reason})",
                    tool=params.tool,
                    model=params.model,
                    duration_s=0.0,
                )

        for attempt in range(MAX_RETRIES + 1):
            session = None
            cancel_guard_done: Optional[threading.Event] = None
            time.monotonic()

            try:
                # Early cancel check
                if _is_cancelled():
                    return AgentCallResult(
                        error="Cancelled before execution",
                        stop_reason="cancelled",
                        tool=params.tool,
                        model=params.model,
                        duration_s=time.monotonic() - start,
                    )
                if _deadline_budget_s() == 0:
                    return AgentCallResult(
                        error="Workflow deadline exhausted before execution",
                        tool=params.tool,
                        model=params.model,
                        duration_s=time.monotonic() - start,
                    )

                if attempt > 0:
                    logger.info(
                        "[AgentExecutor] Retry attempt %d/%d for tool=%s (previous error: %s)",
                        attempt,
                        MAX_RETRIES,
                        params.tool,
                        last_error,
                    )

                full_prompt = self._build_prompt(params)

                # Create session (with timeout protection via shared pool).
                # Poll cancel_event during creation so that race() loser aborts
                # and global /stop_wf can interrupt session startup, not just the
                # send_prompt.
                from src.agent_session.factory import create_engine_session

                future = self._session_pool.submit(
                    create_engine_session,
                    agent_type=params.tool,
                    cwd=self.cwd,
                    model_name=params.model,
                    cancel_event=call_cancel_event,
                )
                # Wait for session creation with periodic cancel checks
                session_create_timeout = _settings_int(
                    "workflow_session_create_timeout_s", SESSION_CREATE_TIMEOUT_S
                )
                create_timeout_s = _effective_timeout_s(
                    session_create_timeout,
                    session_create_timeout,
                )
                create_deadline = time.monotonic() + create_timeout_s
                session = None
                while time.monotonic() < create_deadline:
                    if _is_cancelled():
                        # Cancel during session creation — abandon the future
                        logger.debug(
                            "[AgentExecutor] cancel_event set during session creation for tool=%s",
                            params.tool,
                        )
                        self._close_late_session(future, params.tool)
                        return AgentCallResult(
                            error="Cancelled during session creation",
                            stop_reason="cancelled",
                            tool=params.tool,
                            model=params.model,
                            duration_s=time.monotonic() - start,
                        )
                    try:
                        remaining = create_deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        session = future.result(timeout=min(0.5, remaining))
                        break
                    except concurrent.futures.TimeoutError:
                        continue  # check cancel and try again
                if session is None:
                    # Timeout or didn't break out of the loop without a TimeoutError:
                    logger.error(
                        "[AgentExecutor] session creation timeout for tool=%s (>%ds) [RUNTIME_TIMEOUT]",
                        params.tool,
                        create_timeout_s,
                    )
                    self._close_late_session(future, params.tool)
                    error_msg = f"session creation timeout (>{create_timeout_s}s)"
                    return AgentCallResult(
                        error=error_msg,
                        tool=params.tool,
                        model=params.model,
                        duration_s=time.monotonic() - start,
                    )

                # Send prompt and collect result
                token_usage = 0

                # Start cancel guard: if call_cancel_event fires during send_prompt,
                # actively cancel the session so the blocking call returns quickly
                # instead of waiting for the full LLM round-trip.
                cancel_guard_done = threading.Event()
                self._start_cancel_guard(
                    session,
                    per_call_event=per_call_cancel_event,
                    global_event=global_cancel_event,
                    done_event=cancel_guard_done,
                    tool=params.tool,
                )

                prompt_timeout_s = _effective_timeout_s(
                    params.timeout,
                    _settings_int("workflow_agent_call_timeout_s", AGENT_CALL_TIMEOUT_S),
                )
                if _deadline_budget_s() == 0:
                    return AgentCallResult(
                        error="Workflow deadline exhausted before prompt execution",
                        tool=params.tool,
                        model=params.model,
                        duration_s=time.monotonic() - start,
                    )

                idle_timeout_s = _settings_int(
                    "workflow_agent_idle_timeout_s", AGENT_IDLE_TIMEOUT_S
                )

                # Build on_event callback for activity tracking
                _on_activity = self.on_activity
                _on_subagent_update = self.on_subagent_update
                _agent_label = params.label or ""

                def _event_cb(ev: Any) -> None:
                    if (not _on_activity and not _on_subagent_update) or not _agent_label:
                        return
                    try:
                        ev_type = getattr(ev, "event_type", None)
                        if ev_type is None:
                            return
                        type_val = ev_type.value if hasattr(ev_type, "value") else str(ev_type)
                        tc = getattr(ev, "tool_call", None)
                        subagent_updates = _subagent_updates_from_tool_call(tc)
                        if subagent_updates and _on_subagent_update:
                            _on_subagent_update(_agent_label, subagent_updates)
                        if _is_subagent_tool_call(tc):
                            return
                        if not _on_activity:
                            return
                        if type_val == "tool_call_start":
                            if tc:
                                title = getattr(tc, "title", "") or getattr(tc, "kind", "")
                                _on_activity(_agent_label, title[:60])
                        elif type_val == "tool_call_done":
                            if tc:
                                title = getattr(tc, "title", "") or getattr(tc, "kind", "")
                                status = getattr(tc, "status", "")
                                _on_activity(_agent_label, f"{title[:50]} ({status})")
                    except Exception:
                        pass

                # Pass idle_timeout for adaptive timeout; gracefully degrade
                # for session implementations that don't support it (e.g. test mocks).
                send_kwargs: dict[str, Any] = {
                    "on_event": (
                        _event_cb
                        if _on_activity or _on_subagent_update
                        else None
                    ),
                    "timeout": prompt_timeout_s,
                }
                if idle_timeout_s > 0:
                    send_kwargs["idle_timeout"] = float(idle_timeout_s)

                try:
                    result = session.send_prompt(full_prompt, **send_kwargs)
                except TypeError:
                    # Fallback: session doesn't accept idle_timeout
                    result = session.send_prompt(
                        full_prompt,
                        on_event=send_kwargs["on_event"],
                        timeout=prompt_timeout_s,
                    )

                # Extract text output and token usage from PromptResult
                output_text = result.text if result else ""
                token_usage = result.output_tokens or 0 if result else 0
                raw_stop_reason = getattr(result, "stop_reason", None) if result else None
                stop_reason = (
                    raw_stop_reason.strip()
                    if isinstance(raw_stop_reason, str) and raw_stop_reason.strip()
                    else None
                )

                # Report token usage via callback
                if token_usage > 0 and self.on_token_usage:
                    self.on_token_usage(token_usage)
                total_token_usage += token_usage

                # Cancel check after prompt completion
                if call_cancel_event.is_set():
                    return AgentCallResult(
                        output=output_text,
                        stop_reason=stop_reason,
                        token_usage=total_token_usage,
                        duration_s=time.monotonic() - start,
                        error="Cancelled during execution",
                        tool=params.tool,
                        model=params.model,
                    )

                # Schema validation with retry (separate from general retry)
                parsed: Optional[dict[str, Any]] = None
                schema_error: Optional[str] = None
                if params.output_schema:
                    valid, parsed = self._validate_schema(output_text, params.output_schema)

                    schema_retry_count = 0
                    while not valid and schema_retry_count < SCHEMA_RETRY_MAX:
                        if _is_cancelled():
                            break

                        schema_retry_count += 1
                        fix_prompt = self._build_schema_fix_prompt(output_text, params.output_schema)
                        logger.info(
                            "[AgentExecutor] Schema validation failed, retry %d/%d for tool=%s",
                            schema_retry_count,
                            SCHEMA_RETRY_MAX,
                            params.tool,
                        )

                        retry_timeout_s = _effective_timeout_s(
                            params.timeout,
                            _settings_int("workflow_agent_call_timeout_s", AGENT_CALL_TIMEOUT_S),
                        )
                        if _deadline_budget_s() == 0:
                            break

                        retry_kwargs: dict[str, Any] = {
                            "on_event": send_kwargs["on_event"],
                            "timeout": retry_timeout_s,
                        }
                        if idle_timeout_s > 0:
                            retry_kwargs["idle_timeout"] = float(idle_timeout_s)

                        try:
                            retry_result = session.send_prompt(fix_prompt, **retry_kwargs)
                        except TypeError:
                            retry_result = session.send_prompt(
                                fix_prompt,
                                on_event=retry_kwargs["on_event"],
                                timeout=retry_timeout_s,
                            )

                        retry_text = retry_result.text if retry_result else ""
                        retry_tokens = retry_result.output_tokens or 0 if retry_result else 0
                        raw_retry_stop_reason = (
                            getattr(retry_result, "stop_reason", None) if retry_result else None
                        )
                        stop_reason = (
                            raw_retry_stop_reason.strip()
                            if isinstance(raw_retry_stop_reason, str)
                            and raw_retry_stop_reason.strip()
                            else None
                        )

                        # Accumulate token usage from retries
                        total_token_usage += retry_tokens
                        if retry_tokens > 0 and self.on_token_usage:
                            self.on_token_usage(retry_tokens)

                        output_text = retry_text
                        valid, parsed = self._validate_schema(output_text, params.output_schema)

                    if not valid:
                        attempts = 1 + schema_retry_count
                        schema_error = (
                            "Structured output schema validation failed "
                            f"after {attempts} attempt{'s' if attempts != 1 else ''}"
                        )
                        logger.warning(
                            "[AgentExecutor] %s for tool=%s",
                            schema_error,
                            params.tool,
                        )

                duration_s = time.monotonic() - start
                return AgentCallResult(
                    output=output_text,
                    parsed=parsed,
                    stop_reason=stop_reason,
                    token_usage=total_token_usage,
                    duration_s=duration_s,
                    error=schema_error,
                    tool=params.tool,
                    model=params.model,
                )

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                last_error = error_msg
                logger.error(
                    "[AgentExecutor] execute failed (attempt %d/%d) for tool=%s: %s",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    params.tool,
                    error_msg,
                    exc_info=True,
                )

                # Circuit breaker: if ACP startup itself failed, blacklist the
                # tool for the rest of this workflow run so we don't waste 50s
                # on each subsequent call to the same broken tool.
                if _is_startup_error(e) and params.tool:
                    with self._blacklist_lock:
                        self._startup_blacklist[params.tool] = str(e)[:120]
                    logger.warning(
                        "[AgentExecutor] Circuit breaker armed: tool=%s blacklisted for this workflow run",
                        params.tool,
                    )

                # Check if we should retry.
                # TimeoutError / asyncio.TimeoutError are never retried — the
                # per-call timeout budget was already consumed and retrying the
                # same prompt would just waste another full timeout window.
                is_prompt_timeout = (
                    isinstance(e, TimeoutError) or (e.__class__.__name__ == "TimeoutError") or _is_timeout_in_chain(e)
                )
                if attempt < MAX_RETRIES and is_transient_error(error_msg) and not is_prompt_timeout:
                    # Check cancel before sleeping
                    if _is_cancelled():
                        break
                    self._sleep_with_backoff(attempt, extra_cancel=per_call_cancel_event)
                    continue

                # Permanent error or max retries exceeded — return error
                duration_s = time.monotonic() - start
                return AgentCallResult(
                    error=error_msg,
                    tool=params.tool,
                    model=params.model,
                    duration_s=duration_s,
                )

            finally:
                # Always close the session (short-lived, one-shot)
                if session is not None:
                    try:
                        session.close()
                    except Exception as close_err:
                        logger.debug("[AgentExecutor] session close failed: %s", repr(close_err))
                # Signal the cancel-guard thread to exit (if it was started)
                if cancel_guard_done is not None:
                    cancel_guard_done.set()

        # If we exited the loop due to cancel event
        duration_s = time.monotonic() - start
        return AgentCallResult(
            error="Cancelled during retry",
            stop_reason="cancelled",
            tool=params.tool,
            model=params.model,
            duration_s=duration_s,
        )

    def _close_late_session(
        self,
        future: concurrent.futures.Future[Any],
        tool: str | None,
    ) -> None:
        """Close a session that finishes after the caller already timed out.

        The close() operation runs in a separate thread (non-daemon, tracked
        in ``_late_close_threads``) so that shutdown() can wait for it to
        finish and no ACP subprocesses are orphaned at interpreter exit.
        """
        if future.cancel():
            logger.info(
                "[AgentExecutor] cancelled pending session creation for tool=%s after timeout",
                tool,
            )
            return

        def _cleanup(done: concurrent.futures.Future[Any]) -> None:
            def _close_async() -> None:
                try:
                    stale_session = done.result(timeout=30)
                except concurrent.futures.CancelledError:
                    return
                except Exception as exc:
                    logger.debug(
                        "[AgentExecutor] late session creation failed after timeout for tool=%s: %s",
                        tool,
                        repr(exc),
                    )
                    return

                if stale_session is None:
                    return
                close = getattr(stale_session, "close", None)
                if not callable(close):
                    return
                try:
                    close()
                    logger.info(
                        "[AgentExecutor] closed late session after creation timeout for tool=%s",
                        tool,
                    )
                except Exception as exc:
                    logger.debug(
                        "[AgentExecutor] late session close failed for tool=%s: %s",
                        tool,
                        repr(exc),
                    )

            t = threading.Thread(
                target=_close_async,
                name=f"wf-late-close-{tool}",
                daemon=False,
            )
            with self._late_close_lock:
                self._late_close_threads.append(t)
            t.start()

        future.add_done_callback(_cleanup)

    def _sleep_with_backoff(self, attempt: int, *, extra_cancel: Optional[threading.Event] = None) -> None:
        """Sleep for exponential backoff delay.

        Delay = RETRY_BACKOFF_BASE_S * 2^attempt

        Parameters
        ----------
        attempt:
            The zero-based retry attempt number (0 = first retry).
        extra_cancel:
            Optional additional cancel event to poll during sleep (OR semantics
            with the executor-level cancel_event).  Used for per-call cancellation
            during retry backoff (e.g. race loser abort).
        """
        delay = RETRY_BACKOFF_BASE_S * (2**attempt)
        logger.debug(
            "[AgentExecutor] Backoff delay: %.2fs before retry attempt %d",
            delay,
            attempt + 1,
        )
        # Sleep in small increments to check cancel events frequently
        sleep_start = time.monotonic()
        while time.monotonic() - sleep_start < delay:
            if self.cancel_event.is_set():
                break
            if extra_cancel is not None and extra_cancel.is_set():
                break
            time.sleep(min(0.1, delay - (time.monotonic() - sleep_start)))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = True, *, late_close_wait_s: float = 2.0) -> None:
        """Shut down the shared session pool. Safe to call multiple times.

        Parameters
        ----------
        wait:
            If ``True`` (default), block until all pending workers finish.
            If ``False``, shut down asynchronously.
        late_close_wait_s:
            Maximum seconds to wait for late-close threads (sessions that
            were still being created when the caller timed out).  These
            threads close orphaned ACP sessions; we give them a short
            window to finish to avoid leaving subprocesses running.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if self._session_pool is not None:
            try:
                self._session_pool.shutdown(wait=wait, cancel_futures=True)
            except TypeError:
                # Older Python may not accept ``cancel_futures`` keyword.
                self._session_pool.shutdown(wait=wait)
            except Exception as e:
                logger.debug("[AgentExecutor] session pool shutdown failed: %s", repr(e))
            self._session_pool = None

        # Wait for late-close threads so that no ACP subprocesses are
        # orphaned at interpreter exit.  Only wait a bounded time — if
        # sessions are taking longer than ``late_close_wait_s`` to create,
        # they'll be cleaned up by the process exit anyway.
        with self._late_close_lock:
            threads = list(self._late_close_threads)
        for t in threads:
            t.join(timeout=max(0.0, late_close_wait_s))

    def __enter__(self) -> "AgentExecutor":
        """Context manager entry — returns self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Context manager exit — shuts down the executor.

        Returns ``False`` so that any exception is propagated normally.
        """
        self.shutdown()
        return False

    def __del__(self) -> None:
        """Destructor fallback — warns and best-effort shutdown if not properly closed.

        Defensive: checks attribute existence with ``hasattr`` and wraps
        ``shutdown()`` in a broad ``try/except`` to avoid errors during
        interpreter shutdown.
        """
        if hasattr(self, "_shutdown_done") and not self._shutdown_done:
            logger.warning("AgentExecutor was not properly shut down; call shutdown() or use as context manager")
            try:
                self.shutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_cancel_guard(
        self,
        session: Any,
        *,
        per_call_event: Optional[threading.Event],
        global_event: threading.Event,
        done_event: threading.Event,
        tool: str,
    ) -> None:
        """Start a daemon thread that cancels the session when either cancel event fires.

        The guard polls both per-call and global cancel events with OR semantics
        — either signal triggers session.cancel().  The done_event is set by the
        caller in the finally block to signal the guard to exit cleanly if
        cancel never fires.

        This ensures both race() loser aborts (per-call) and global /stop_wf
        (global) actually interrupt the LLM call instead of waiting for it to
        finish.
        """

        def _guard() -> None:
            while not done_event.is_set():
                per_call_fired = per_call_event is not None and per_call_event.wait(timeout=0.1)
                global_fired = global_event.is_set()
                if per_call_fired or global_fired:
                    if done_event.is_set():
                        return
                    try:
                        session.cancel()
                        logger.debug(
                            "[AgentExecutor] cancel guard fired for tool=%s (per_call=%s, global=%s)",
                            tool,
                            per_call_fired,
                            global_fired,
                        )
                    except Exception as e:
                        logger.debug(
                            "[AgentExecutor] cancel guard session.cancel() failed: %s",
                            repr(e),
                        )
                    return

        t = threading.Thread(
            target=_guard,
            name=f"wf-cancel-guard-{tool}",
            daemon=True,
        )
        t.start()

    def _build_prompt(self, params: AgentCallParams) -> str:
        """Compose the full prompt: role prefix + task + subagent encouragement.

        Structure:
            [Role: {role}\\n\\n]  (if params.role is set)
            {prompt}
            \\n\\n{encouragement}  (if enabled via settings)
        """
        parts: list[str] = []

        # Role preamble
        if params.role:
            parts.append(f"Role: {params.role}\n\n")

        # Core task prompt
        parts.append(params.prompt)

        # Subagent encouragement suffix (may be "" when disabled via settings)
        encouragement = get_subagent_encouragement_prompt()
        if encouragement:
            parts.append(f"\n\n{encouragement}")

        return "".join(parts)

    def _validate_schema(self, output: str, schema: dict[str, Any]) -> tuple[bool, Optional[dict[str, Any]]]:
        """Try JSON parse + validate against GhostAP's compact shape schema.

        Validation strategy:
        - Parse the output as JSON.
        - Require a top-level object.
        - Recursively validate required keys and value shapes.

        Type-name descriptors and JSON exemplar values are both supported.
        ``[]`` accepts any array; ``[schema]`` validates every array item.

        Returns:
            (valid, parsed_dict_or_None)
        """
        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            # Try to extract JSON from markdown code blocks
            parsed = self._extract_json_from_text(output)
            if parsed is None:
                return False, None

        if not isinstance(parsed, dict):
            return False, None

        mismatch = self._schema_mismatch(parsed, schema)
        if mismatch is not None:
            logger.debug("[AgentExecutor] Schema validation: %s", mismatch)
            return False, None

        return True, parsed

    @classmethod
    def _schema_mismatch(
        cls,
        value: Any,
        schema: Any,
        path: str = "$",
    ) -> Optional[str]:
        """Return the first compact-schema mismatch, or ``None`` when valid."""
        if isinstance(schema, str):
            expected = schema.strip().lower()
            aliases = {
                "str": "string",
                "list": "array",
                "dict": "object",
                "float": "number",
                "int": "integer",
                "bool": "boolean",
                "none": "null",
                "*": "any",
            }
            expected = aliases.get(expected, expected)
            known_types = {
                "string",
                "array",
                "object",
                "number",
                "integer",
                "boolean",
                "null",
                "any",
            }
            # Unknown strings such as "low" are example values and therefore
            # express the string type rather than an unsupported descriptor.
            if expected not in known_types:
                expected = "string"

            type_ok = {
                "string": isinstance(value, str),
                "array": isinstance(value, list),
                "object": isinstance(value, dict),
                "number": isinstance(value, (int, float)) and not isinstance(value, bool),
                "integer": isinstance(value, int) and not isinstance(value, bool),
                "boolean": isinstance(value, bool),
                "null": value is None,
                "any": True,
            }[expected]
            return None if type_ok else f"{path} expected {expected}"

        if schema is None:
            return None if value is None else f"{path} expected null"

        if isinstance(schema, bool):
            return None if isinstance(value, bool) else f"{path} expected boolean"

        if isinstance(schema, (int, float)):
            valid_number = isinstance(value, (int, float)) and not isinstance(value, bool)
            return None if valid_number else f"{path} expected number"

        if isinstance(schema, list):
            if not isinstance(value, list):
                return f"{path} expected array"
            if not schema:
                return None
            for index, item in enumerate(value):
                mismatch = cls._schema_mismatch(item, schema[0], f"{path}[{index}]")
                if mismatch is not None:
                    return mismatch
            return None

        if isinstance(schema, dict):
            if not isinstance(value, dict):
                return f"{path} expected object"
            for key, child_schema in schema.items():
                if key not in value:
                    return f"{path}.{key} is required"
                mismatch = cls._schema_mismatch(
                    value[key],
                    child_schema,
                    f"{path}.{key}",
                )
                if mismatch is not None:
                    return mismatch
            return None

        return f"{path} has unsupported schema descriptor {type(schema).__name__}"

    def _extract_json_from_text(self, text: str) -> Optional[dict[str, Any]]:
        """Attempt to extract JSON from text that may contain markdown fences.

        Handles common cases where the agent wraps JSON in ```json ... ``` blocks.
        """
        # Try stripping markdown code fences
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove opening fence (with optional language tag)
            lines = stripped.split("\n", 1)
            if len(lines) > 1:
                body = lines[1]
                # Remove closing fence
                if body.rstrip().endswith("```"):
                    body = body.rstrip()[: -len("```")].rstrip()
                try:
                    return json.loads(body)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Try finding the first { ... } block
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _build_schema_fix_prompt(self, failed_output: str, schema: dict[str, Any]) -> str:
        """Build a prompt asking the agent to fix its output to match the schema."""
        schema_desc = json.dumps(schema, indent=2, ensure_ascii=False)
        return (
            "Your previous output did not conform to the required JSON schema.\n\n"
            f"Required schema (all keys must be present):\n```json\n{schema_desc}\n```\n\n"
            f"Your previous output was:\n```\n{failed_output[:2000]}\n```\n\n"
            "Please output ONLY valid JSON matching the schema above. "
            "Do not include any explanation, markdown fences, or extra text — "
            "just the raw JSON object."
        )
