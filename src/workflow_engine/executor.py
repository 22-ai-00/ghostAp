"""Execute one Workflow ``agent()`` call through a short-lived session."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Optional

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

if TYPE_CHECKING:
    from src.card.events import CardEvent

logger = logging.getLogger(__name__)

_SUBAGENT_ACTIVITY = {
    "started": ("running", "已启动"),
    "interacted": ("running", "已与主 Agent 交互"),
}


def _is_subagent_tool_call(tool_call: Any) -> bool:
    return tool_call is not None and any(
        getattr(tool_call, field, None)
        for field in (
            "subagent_source_id",
            "subagent_activity",
            "collaboration_tool",
            "collaboration_receivers",
            "subagent_states",
        )
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


def _subagent_updates_from_tool_call(tool_call: Any) -> tuple[dict[str, object], ...]:
    """Project ACP child-agent frames as observational, never authoritative, state."""
    if tool_call is None:
        return ()

    receivers = tuple(
        value
        for item in (getattr(tool_call, "collaboration_receivers", ()) or ())
        if (value := str(item or "").strip())
    )
    source_id = str(getattr(tool_call, "subagent_source_id", None) or "").strip()
    states = {
        state_id: item
        for item in (getattr(tool_call, "subagent_states", ()) or ())
        if isinstance(item, Mapping)
        and (state_id := str(item.get("source_id") or "").strip())
    }
    source_ids = tuple(dict.fromkeys((*receivers, *states, *((source_id,) if source_id else ()))))
    if not source_ids:
        return ()

    activity = str(getattr(tool_call, "subagent_activity", None) or "").strip().lower()
    activity_state = _SUBAGENT_ACTIVITY.get(activity)
    if activity == "interrupted":
        interrupt_status = str(getattr(tool_call, "status", None) or "").lower()
        if interrupt_status == "completed":
            activity_state = ("cancelled", "已中断")
        elif interrupt_status == "failed":
            activity_state = ("running", "中断未完成")
        else:
            activity_state = ("running", "正在中断")
    model_value = getattr(tool_call, "collaboration_model", None)
    model = str(model_value).strip() if model_value else None

    updates = []
    for child_id in source_ids:
        state = states.get(child_id)
        observed = _safe_subagent_progress(
            state.get("message") if state else "",
            opaque_ids=source_ids,
        )
        status, fallback = activity_state if child_id == source_id and activity_state else ("running", "已启动")
        updates.append(
            {
                "source_id": child_id,
                "status": status,
                "progress": observed or fallback,
                "model": model,
            }
        )
    return tuple(updates)


def _settings_int(field: str, fallback: int) -> int:
    try:
        from src.config import get_settings

        return int(getattr(get_settings(), field, fallback))
    except Exception:  # pragma: no cover - configuration must not break a run
        return fallback


def _exception_chain_has(exc: BaseException, class_name: str) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if current.__class__.__name__ == class_name:
            return True
        current = current.__cause__ or current.__context__
    return False


def _deadline_budget_s(deadline_monotonic: float | None) -> int | None:
    if deadline_monotonic is None:
        return None
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= WORKFLOW_TIMEOUT_HEADROOM_S:
        return 0
    return int(max(1.0, remaining - WORKFLOW_TIMEOUT_HEADROOM_S))


def _resolve_timeout_s(
    requested: float | int | None,
    configured: float | int,
    deadline_monotonic: float | None,
) -> float:
    """Apply the host floor, unlimited backstop, and remaining run budget."""
    try:
        configured_s = float(configured)
    except (TypeError, ValueError):
        configured_s = float(AGENT_CALL_TIMEOUT_S)
    floor = float(AGENT_UNLIMITED_BACKSTOP_S) if configured_s <= 0 else configured_s
    try:
        requested_s = float(requested) if requested is not None else 0.0
    except (TypeError, ValueError):
        requested_s = 0.0
    resolved = max(floor, requested_s) if requested_s > 0 else floor
    budget = _deadline_budget_s(deadline_monotonic)
    return resolved if budget is None else max(1.0, min(resolved, float(budget)))


class AgentExecutor:
    """Open, use, and close one isolated ACP/CLI session per call."""

    def __init__(
        self,
        cwd: str,
        cancel_event: threading.Event,
        on_token_usage: Optional[Callable[[int], None]] = None,
        on_activity: Optional[Callable[[str, str], None]] = None,
        max_workers: int = DEFAULT_MAX_CONCURRENT,
        on_subagent_update: Optional[
            Callable[[str, tuple[dict[str, object], ...]], None]
        ] = None,
        on_attempt: Optional[Callable[[str, int], None]] = None,
        on_card_event: Optional[Callable[[str, "CardEvent"], None]] = None,
    ) -> None:
        self.cwd = cwd
        self.cancel_event = cancel_event
        self.on_token_usage = on_token_usage
        self.on_activity = on_activity
        self.on_attempt = on_attempt
        self.on_subagent_update = on_subagent_update
        self.on_card_event = on_card_event
        pool_size = max(1, min(int(max_workers), HARD_MAX_CONCURRENT))
        self._session_pool: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=pool_size,
                thread_name_prefix="wf_session",
            )
        )
        self._shutdown_done = False
        self._late_close_threads: list[threading.Thread] = []
        self._late_close_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._startup_blacklist: dict[str, str] = {}
        self._blacklist_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._event_turn_seq = 0
        self._event_turn_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    def execute(
        self,
        params: AgentCallParams,
        *,
        cancel_event: Optional[threading.Event] = None,
        deadline_monotonic: float | None = None,
    ) -> AgentCallResult:
        start = time.monotonic()
        total_tokens = 0
        last_error: str | None = None

        def outcome(**values: Any) -> AgentCallResult:
            return AgentCallResult(
                tool=params.tool,
                model=params.model,
                duration_s=time.monotonic() - start,
                **values,
            )

        if params.tool:
            with self._blacklist_lock:
                blocked = self._startup_blacklist.get(params.tool)
            if blocked:
                return outcome(
                    error=(
                        f"Circuit breaker: {params.tool} ACP startup failed earlier "
                        f"in this workflow ({blocked})"
                    )
                )

        prompt = self._build_prompt(params)
        idle_timeout_s = _settings_int(
            "workflow_agent_idle_timeout_s",
            AGENT_IDLE_TIMEOUT_S,
        )

        for attempt in range(MAX_RETRIES + 1):
            session = None
            cancel_guard: threading.Event | None = None
            if self.on_attempt and params.label:
                self.on_attempt(params.label, attempt + 1)
            try:
                if self._cancelled(cancel_event):
                    return outcome(error="Cancelled before execution", stop_reason="cancelled")
                if _deadline_budget_s(deadline_monotonic) == 0:
                    return outcome(error="Workflow deadline exhausted before execution")
                if attempt:
                    logger.info(
                        "[AgentExecutor] Retry %d/%d tool=%s after %s",
                        attempt,
                        MAX_RETRIES,
                        params.tool,
                        last_error,
                    )

                session, create_error, create_cancelled = self._open_session(
                    params,
                    cancel_event,
                    deadline_monotonic,
                )
                if create_error:
                    return outcome(
                        error=create_error,
                        stop_reason="cancelled" if create_cancelled else None,
                    )
                if session is None:
                    return outcome(error="Session creation returned no session")

                cancel_guard = self._start_cancel_guard(session, cancel_event, params.tool or "agent")
                output, tokens, stop_reason = self._send_prompt(
                    session,
                    prompt,
                    params.label or "",
                    _resolve_timeout_s(
                        params.timeout,
                        _settings_int("workflow_agent_call_timeout_s", AGENT_CALL_TIMEOUT_S),
                        deadline_monotonic,
                    ),
                    idle_timeout_s,
                )
                total_tokens += self._record_tokens(tokens)
                if self._cancelled(cancel_event):
                    return outcome(
                        output=output,
                        stop_reason=stop_reason or "cancelled",
                        token_usage=total_tokens,
                        error="Cancelled during execution",
                    )

                parsed: dict[str, Any] | None = None
                schema_error: str | None = None
                if params.output_schema:
                    valid, parsed = self._validate_schema(output, params.output_schema)
                    retries = 0
                    while not valid and retries < SCHEMA_RETRY_MAX:
                        if self._cancelled(cancel_event) or _deadline_budget_s(deadline_monotonic) == 0:
                            break
                        retries += 1
                        logger.info(
                            "[AgentExecutor] Schema repair %d/%d tool=%s",
                            retries,
                            SCHEMA_RETRY_MAX,
                            params.tool,
                        )
                        output, tokens, stop_reason = self._send_prompt(
                            session,
                            self._build_schema_fix_prompt(output, params.output_schema),
                            params.label or "",
                            _resolve_timeout_s(
                                params.timeout,
                                _settings_int("workflow_agent_call_timeout_s", AGENT_CALL_TIMEOUT_S),
                                deadline_monotonic,
                            ),
                            idle_timeout_s,
                        )
                        total_tokens += self._record_tokens(tokens)
                        valid, parsed = self._validate_schema(output, params.output_schema)
                    if self._cancelled(cancel_event):
                        return outcome(
                            output=output,
                            stop_reason=stop_reason or "cancelled",
                            token_usage=total_tokens,
                            error="Cancelled during schema repair",
                        )
                    if not valid:
                        attempts = 1 + retries
                        schema_error = (
                            "Structured output schema validation failed "
                            f"after {attempts} attempt{'s' if attempts != 1 else ''}"
                        )

                return outcome(
                    output=output,
                    parsed=parsed,
                    stop_reason=stop_reason,
                    token_usage=total_tokens,
                    error=schema_error,
                )
            except Exception as exc:
                raw_error = f"{type(exc).__name__}: {exc}"
                last_error = _strip_internal_details(raw_error) or type(exc).__name__
                logger.error(
                    "[AgentExecutor] attempt %d/%d failed tool=%s: %s",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    params.tool,
                    last_error,
                    exc_info=True,
                )
                if _exception_chain_has(exc, "ACPStartupError") and params.tool:
                    with self._blacklist_lock:
                        self._startup_blacklist[params.tool] = last_error[:120]
                timed_out = _exception_chain_has(exc, "TimeoutError")
                if (
                    attempt < MAX_RETRIES
                    and is_transient_error(raw_error)
                    and not timed_out
                    and not self._cancelled(cancel_event)
                ):
                    self._sleep_with_backoff(attempt, cancel_event)
                    continue
                return outcome(error=last_error)
            finally:
                if session is not None:
                    try:
                        session.close()
                    except Exception as exc:
                        logger.debug("[AgentExecutor] session close failed: %r", exc)
                if cancel_guard is not None:
                    cancel_guard.set()

        return outcome(error="Cancelled during retry", stop_reason="cancelled")

    def _cancelled(self, per_call: threading.Event | None) -> bool:
        return self.cancel_event.is_set() or (per_call is not None and per_call.is_set())

    def _open_session(
        self,
        params: AgentCallParams,
        per_call_cancel: threading.Event | None,
        deadline_monotonic: float | None,
    ) -> tuple[Any | None, str | None, bool]:
        from src.agent_session.factory import create_engine_session

        pool = self._session_pool
        if pool is None:
            raise RuntimeError("AgentExecutor is shut down")
        call_cancel = per_call_cancel or self.cancel_event
        future = pool.submit(
            create_engine_session,
            agent_type=params.tool,
            cwd=self.cwd,
            model_name=params.model,
            cancel_event=call_cancel,
            capture_full_tool_content=True,
        )
        configured = _settings_int(
            "workflow_session_create_timeout_s",
            SESSION_CREATE_TIMEOUT_S,
        )
        timeout_s = _resolve_timeout_s(configured, configured, deadline_monotonic)
        deadline = time.monotonic() + timeout_s
        while True:
            if self._cancelled(per_call_cancel):
                self._close_late_session(future, params.tool)
                return None, "Cancelled during session creation", True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._close_late_session(future, params.tool)
                logger.error(
                    "[AgentExecutor] session creation timed out tool=%s after %.1fs",
                    params.tool,
                    timeout_s,
                )
                return None, f"session creation timeout (>{timeout_s}s)", False
            try:
                session = future.result(timeout=min(0.5, remaining))
            except concurrent.futures.TimeoutError:
                continue
            if session is None:
                return None, "Session creation returned no session", False
            return session, None, False

    def _send_prompt(
        self,
        session: Any,
        prompt: str,
        label: str,
        timeout_s: float,
        idle_timeout_s: int,
    ) -> tuple[str, int, str | None]:
        # ACP providers may retain and invoke an ``on_event`` callback after
        # ``send_prompt`` returns. Keep each semantic turn isolated so schema
        # repair and outer retries cannot append to a prior turn's blocks.
        on_event = self._event_callback(label)
        kwargs: dict[str, Any] = {"on_event": on_event, "timeout": timeout_s}
        if idle_timeout_s > 0:
            kwargs["idle_timeout"] = float(idle_timeout_s)
        try:
            result = session.send_prompt(prompt, **kwargs)
        finally:
            # One send_prompt call is one semantic ACP turn. Schema repair and
            # other bounded follow-ups must start new text/reasoning blocks.
            self._close_event_callback(on_event)
        if result is None:
            return "", 0, None
        raw_reason = getattr(result, "stop_reason", None)
        stop_reason = raw_reason.strip() if isinstance(raw_reason, str) and raw_reason.strip() else None
        return result.text or "", result.output_tokens or 0, stop_reason

    def _record_tokens(self, tokens: int) -> int:
        if tokens > 0 and self.on_token_usage:
            self.on_token_usage(tokens)
        return tokens

    def _event_callback(self, label: str) -> Callable[[Any], None] | None:
        if not label or (
            not self.on_activity
            and not self.on_subagent_update
            and not self.on_card_event
        ):
            return None

        card_bridge = None
        if self.on_card_event:
            from src.card.events import CardEvent
            from src.card.stream_bridge import ACPStreamBridge

            on_card_event = self.on_card_event
            with self._event_turn_lock:
                self._event_turn_seq += 1
                turn_seq = self._event_turn_seq

            class _DispatchTarget:
                def __init__(self) -> None:
                    self._block_ids: dict[tuple[str, str], str] = {}
                    self._tool_ids: dict[str, str] = {}
                    self._next_block_seq = 0

                def _next_id(self, family: str) -> str:
                    self._next_block_seq += 1
                    return f"_wf_turn_{turn_seq}_{family}_{self._next_block_seq}"

                def _namespaced(self, card_event: "CardEvent") -> "CardEvent":
                    raw_block_id = str(card_event.payload.get("block_id") or "")
                    if not raw_block_id:
                        return card_event
                    type_value = card_event.type.value
                    if type_value == "tool_started":
                        block_id = self._next_id("tool")
                        # A provider may recycle the same tool-call ID for a
                        # later invocation. Every start defines a new logical
                        # block; subsequent update/done frames follow it.
                        self._tool_ids[raw_block_id] = block_id
                    elif type_value.startswith("tool_"):
                        block_id = self._tool_ids.get(raw_block_id)
                        if block_id is None:
                            block_id = self._next_id("tool")
                            self._tool_ids[raw_block_id] = block_id
                    else:
                        family = (
                            "reasoning"
                            if type_value.startswith("reasoning_")
                            else "text"
                        )
                        key = (family, raw_block_id)
                        block_id = self._block_ids.get(key)
                        if block_id is None:
                            block_id = self._next_id(family)
                            self._block_ids[key] = block_id
                    return CardEvent(
                        type=card_event.type,
                        payload={**card_event.payload, "block_id": block_id},
                    )

                def dispatch(self, card_event: "CardEvent") -> None:
                    on_card_event(label, self._namespaced(card_event))

            card_bridge = ACPStreamBridge(
                _DispatchTarget(),
                preserve_tool_content=True,
            )

        callback_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        retired = False

        def callback(event: Any) -> None:
            with callback_lock:
                if retired:
                    return
                if card_bridge is not None:
                    try:
                        card_bridge.on_event(event)
                    except Exception:
                        logger.debug(
                            "[AgentExecutor] card event projection failed",
                            exc_info=True,
                        )
                try:
                    event_type = getattr(event, "event_type", None)
                    type_value = (
                        event_type.value
                        if hasattr(event_type, "value")
                        else str(event_type or "")
                    )
                    tool_call = getattr(event, "tool_call", None)
                    updates = _subagent_updates_from_tool_call(tool_call)
                    if updates and self.on_subagent_update:
                        self.on_subagent_update(label, updates)
                    if (
                        _is_subagent_tool_call(tool_call)
                        or not self.on_activity
                        or tool_call is None
                    ):
                        return
                    title = getattr(tool_call, "title", "") or getattr(
                        tool_call, "kind", ""
                    )
                    if type_value == "tool_call_start":
                        self.on_activity(label, title[:60])
                    elif type_value == "tool_call_done":
                        self.on_activity(
                            label,
                            f"{title[:50]} ({getattr(tool_call, 'status', '')})",
                        )
                except Exception:
                    logger.debug(
                        "[AgentExecutor] event callback failed",
                        exc_info=True,
                    )

        def retire() -> None:
            nonlocal retired
            with callback_lock:
                if retired:
                    return
                retired = True
                if card_bridge is not None:
                    card_bridge.close_open_blocks()

        setattr(callback, "retire", retire)
        return callback

    @staticmethod
    def _close_event_callback(callback: Callable[[Any], None] | None) -> None:
        closer = getattr(callback, "retire", None)
        if not callable(closer):
            closer = getattr(callback, "close_open_blocks", None)
        if not callable(closer):
            return
        try:
            closer()
        except Exception:
            logger.debug(
                "[AgentExecutor] card event projection close failed",
                exc_info=True,
            )

    def _start_cancel_guard(
        self,
        session: Any,
        per_call: threading.Event | None,
        tool: str,
    ) -> threading.Event:
        done = threading.Event()

        def guard() -> None:
            while not done.wait(0.1):
                if not self._cancelled(per_call):
                    continue
                if done.is_set():
                    return
                try:
                    session.cancel()
                except Exception as exc:
                    logger.debug("[AgentExecutor] cancel failed tool=%s: %r", tool, exc)
                return

        threading.Thread(
            target=guard,
            name=f"wf-cancel-guard-{tool}",
            daemon=True,
        ).start()
        return done

    def _close_late_session(
        self,
        future: concurrent.futures.Future[Any],
        tool: str | None,
    ) -> None:
        if future.cancel():
            return

        def completed(done: concurrent.futures.Future[Any]) -> None:
            def close() -> None:
                try:
                    session = done.result()
                    closer = getattr(session, "close", None)
                    if callable(closer):
                        closer()
                except concurrent.futures.CancelledError:
                    pass
                except Exception as exc:
                    logger.debug("[AgentExecutor] late session close failed tool=%s: %r", tool, exc)

            thread = threading.Thread(
                target=close,
                name=f"wf-late-close-{tool}",
                daemon=False,
            )
            with self._late_close_lock:
                self._late_close_threads.append(thread)
            thread.start()

        future.add_done_callback(completed)

    def _sleep_with_backoff(
        self,
        attempt: int,
        per_call: threading.Event | None,
    ) -> None:
        deadline = time.monotonic() + RETRY_BACKOFF_BASE_S * (2**attempt)
        while not self._cancelled(per_call):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.cancel_event.wait(min(0.1, remaining))

    def shutdown(self, wait: bool = True) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        pool, self._session_pool = self._session_pool, None
        if pool is not None:
            try:
                pool.shutdown(wait=wait, cancel_futures=True)
            except Exception as exc:
                logger.debug("[AgentExecutor] session pool shutdown failed: %r", exc)
        with self._late_close_lock:
            threads = tuple(self._late_close_threads)
        for thread in threads:
            thread.join(timeout=2.0)

    def _build_prompt(self, params: AgentCallParams) -> str:
        parts = [f"Role: {params.role}\n\n" if params.role else "", params.prompt]
        encouragement = get_subagent_encouragement_prompt()
        if encouragement:
            parts.append(f"\n\n{encouragement}")
        return "".join(parts)

    def _validate_schema(
        self,
        output: str,
        schema: dict[str, Any],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            parsed = self._extract_json_from_text(output)
        if not isinstance(parsed, dict):
            return False, None
        mismatch = self._schema_mismatch(parsed, schema)
        if mismatch:
            logger.debug("[AgentExecutor] Schema validation: %s", mismatch)
            return False, None
        return True, parsed

    @classmethod
    def _schema_mismatch(cls, value: Any, schema: Any, path: str = "$") -> str | None:
        if isinstance(schema, str):
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
            expected = aliases.get(schema.strip().lower(), schema.strip().lower())
            checks: dict[str, Callable[[Any], bool]] = {
                "string": lambda item: isinstance(item, str),
                "array": lambda item: isinstance(item, list),
                "object": lambda item: isinstance(item, dict),
                "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
                "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
                "boolean": lambda item: isinstance(item, bool),
                "null": lambda item: item is None,
                "any": lambda _item: True,
            }
            expected = expected if expected in checks else "string"
            return None if checks[expected](value) else f"{path} expected {expected}"
        if schema is None:
            return None if value is None else f"{path} expected null"
        if isinstance(schema, bool):
            return None if isinstance(value, bool) else f"{path} expected boolean"
        if isinstance(schema, (int, float)):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            return None if valid else f"{path} expected number"
        if isinstance(schema, list):
            if not isinstance(value, list):
                return f"{path} expected array"
            if schema:
                for index, item in enumerate(value):
                    if mismatch := cls._schema_mismatch(item, schema[0], f"{path}[{index}]"):
                        return mismatch
            return None
        if isinstance(schema, dict):
            if not isinstance(value, dict):
                return f"{path} expected object"
            for key, child_schema in schema.items():
                if key not in value:
                    return f"{path}.{key} is required"
                if mismatch := cls._schema_mismatch(value[key], child_schema, f"{path}.{key}"):
                    return mismatch
            return None
        return f"{path} has unsupported schema descriptor {type(schema).__name__}"

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[dict[str, Any]]:
        stripped = text.strip()
        candidates: list[str] = []
        if stripped.startswith("```") and "\n" in stripped:
            body = stripped.split("\n", 1)[1]
            candidates.append(body[:-3].rstrip() if body.rstrip().endswith("```") else body)
        first, last = text.find("{"), text.rfind("}")
        if first >= 0 and last > first:
            candidates.append(text[first : last + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _build_schema_fix_prompt(failed_output: str, schema: dict[str, Any]) -> str:
        return (
            "Your previous output did not conform to the required JSON schema.\n\n"
            "Required schema (all keys must be present):\n```json\n"
            f"{json.dumps(schema, indent=2, ensure_ascii=False)}\n```\n\n"
            f"Your previous output was:\n```\n{failed_output[:2000]}\n```\n\n"
            "Output only the valid JSON object, without explanation or markdown fences."
        )
