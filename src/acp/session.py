"""ACP session — manages a single ACP agent process lifecycle.

Wraps the ACP SDK's spawn_agent_process to provide a clean interface
for starting sessions, sending prompts, and receiving structured events.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

from acp.exceptions import RequestError
from acp.helpers import text_block
from acp.schema import PromptResponse
from acp.stdio import spawn_agent_process

from ..config import get_settings
from ..utils.async_helpers import safe_wait_for
from ..utils.errors import get_error_detail
from .client import (
    ACPHistoryStore,
    GhostAPClient,
    LocalImageEmissionBudget,
    emit_referenced_changed_local_image_events,
)
from .collaboration import merge_tool_call_sequence
from .models import (
    ACPEvent,
    ACPEventType,
    ACPGoalInfo,
    ACPSessionInfo,
    ACPSessionState,
    PromptResult,
)
from .outcome import has_transient_child_lifecycle
from .transport import LateFrameTolerantMessageQueue

logger = logging.getLogger(__name__)

_CODEX_GOAL_CONTROL_METHOD = "_codex/session/goal_control"
_MAX_DEFERRED_CHILD_EVENTS = 512


def _remember_bounded_identities(
    ledger: dict[str, None],
    values: set[str],
) -> None:
    for value in values:
        if not value:
            continue
        ledger.pop(value, None)
        ledger[value] = None
    while len(ledger) > _MAX_DEFERRED_CHILD_EVENTS:
        ledger.pop(next(iter(ledger)))


def _child_call_id(event: ACPEvent) -> str:
    tool_call = getattr(event, "tool_call", None)
    raw_id = getattr(tool_call, "id", None)
    return raw_id.strip() if isinstance(raw_id, str) else ""


def _child_source_ids(event: ACPEvent) -> set[str]:
    tool_call = getattr(event, "tool_call", None)
    if tool_call is None:
        return set()
    source_ids: set[str] = set()
    raw_source = getattr(tool_call, "subagent_source_id", None)
    if isinstance(raw_source, str) and raw_source.strip():
        source_ids.add(raw_source.strip())
    raw_receivers = getattr(tool_call, "collaboration_receivers", ())
    if isinstance(raw_receivers, str):
        if raw_receivers.strip():
            source_ids.add(raw_receivers.strip())
    elif not isinstance(
        raw_receivers,
        (bytes, bytearray, Mapping),
    ):
        for raw_receiver in raw_receivers or ():
            if isinstance(raw_receiver, str) and raw_receiver.strip():
                source_ids.add(raw_receiver.strip())
    raw_states = getattr(tool_call, "subagent_states", ())
    if isinstance(raw_states, Mapping):
        raw_state_source = raw_states.get("source_id")
        if (
            isinstance(raw_state_source, str)
            and raw_state_source.strip()
        ):
            source_ids.add(raw_state_source.strip())
    elif not isinstance(raw_states, (str, bytes, bytearray)):
        for raw_state in raw_states or ():
            if not isinstance(raw_state, Mapping):
                continue
            raw_state_source = raw_state.get("source_id")
            if (
                isinstance(raw_state_source, str)
                and raw_state_source.strip()
            ):
                source_ids.add(raw_state_source.strip())
    return source_ids


def _starts_child_generation(event: ACPEvent) -> bool:
    tool_call = getattr(event, "tool_call", None)
    if tool_call is None:
        return False
    raw_tool = getattr(tool_call, "collaboration_tool", None)
    tool = (
        raw_tool.strip().casefold()
        if isinstance(raw_tool, str)
        else ""
    )
    if tool in {"spawn_agent", "followup_task"}:
        return str(
            getattr(tool_call, "status", "") or ""
        ).strip().casefold() == "completed"
    return False


def _filter_prior_task_child_sources(
    event: ACPEvent,
    blocked_sources: set[str],
) -> ACPEvent | None:
    """Remove passive prior-task child observations from a new user turn."""
    if not blocked_sources or not _is_child_lifecycle_event(event):
        return event
    if _starts_child_generation(event):
        blocked_sources.difference_update(_child_source_ids(event))
        return event

    tool_call = event.tool_call
    if tool_call is None:
        return event
    observed_sources = _child_source_ids(event)
    blocked_observed = observed_sources & blocked_sources
    if not blocked_observed:
        return event

    raw_source = tool_call.subagent_source_id
    source_is_blocked = (
        isinstance(raw_source, str)
        and raw_source.strip() in blocked_sources
    )
    raw_receivers = tool_call.collaboration_receivers
    receivers = (
        tuple(
            receiver
            for receiver in raw_receivers
            if not (
                isinstance(receiver, str)
                and receiver.strip() in blocked_sources
            )
        )
        if not isinstance(
            raw_receivers,
            (str, bytes, bytearray, Mapping),
        )
        else raw_receivers
    )
    raw_states = tool_call.subagent_states
    states = (
        tuple(
            state
            for state in raw_states
            if not (
                isinstance(state, Mapping)
                and isinstance(state.get("source_id"), str)
                and state["source_id"].strip() in blocked_sources
            )
        )
        if not isinstance(raw_states, (str, bytes, bytearray, Mapping))
        else raw_states
    )
    remaining_sources = observed_sources - blocked_sources
    if not remaining_sources:
        return None
    return replace(
        event,
        tool_call=replace(
            tool_call,
            subagent_source_id=None if source_is_blocked else raw_source,
            subagent_path=(
                None if source_is_blocked else tool_call.subagent_path
            ),
            subagent_activity=(
                None if source_is_blocked else tool_call.subagent_activity
            ),
            collaboration_receivers=receivers,
            subagent_states=states,
        ),
    )


def _is_child_lifecycle_event(event: ACPEvent) -> bool:
    tool_call = getattr(event, "tool_call", None)
    return bool(
        tool_call is not None
        and event.event_type
        in {
            ACPEventType.TOOL_CALL_START,
            ACPEventType.TOOL_CALL_UPDATE,
            ACPEventType.TOOL_CALL_DONE,
        }
        and (
            getattr(tool_call, "collaboration_tool", None)
            or getattr(tool_call, "subagent_source_id", None)
            or getattr(tool_call, "subagent_activity", None)
            or getattr(tool_call, "subagent_states", ())
            or getattr(tool_call, "child_metadata_malformed", False)
        )
    )


class ACPStartupError(RuntimeError):
    """ACP 启动失败的统一可诊断异常（SSOT）。

    继承 RuntimeError 保持向后兼容（已有 except RuntimeError 的捕获链），
    同时标记为 GhostAP 域异常方便统一 log_exception 降级。

    字段协议（稳定）：
    - agent_cmd/agent_args/cwd: 启动命令
    - returncode/stdout_snippet/stderr_snippet: best-effort 诊断片段（应为短文本，便于日志输出/脱敏/截断）
    - fail_phase: 失败阶段（可选但强烈建议设置），用于聚合与排障
    - cause: 原始异常（保留异常链）
    """

    is_ghostap_error = True

    def __init__(
        self,
        message: str,
        *,
        agent_cmd: str,
        agent_args: list[str],
        cwd: str,
        returncode: Optional[int] = None,
        stdout_snippet: str = "",
        stderr_snippet: str = "",
        fail_phase: str = "",
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.agent_cmd = str(agent_cmd or "")
        self.agent_args = list(agent_args or [])
        self.cwd = str(cwd or "")
        self.returncode = returncode
        self.stdout_snippet = stdout_snippet or ""
        self.stderr_snippet = stderr_snippet or ""
        self.fail_phase = str(fail_phase or "")
        self.__cause__ = cause


class ACPResumeRejected(RuntimeError):
    """The provider explicitly rejected a resume target without ambiguity."""


async def _read_stream_snippet(stream: object, *, max_bytes: int = 8192, timeout: float = 0.2) -> str:
    """Best-effort read a small snippet from an asyncio stream.

    IMPORTANT: stdout is ACP JSON-RPC in success path. We only use this on startup failures.
    """
    if stream is None:
        return ""
    try:
        max_bytes = int(max_bytes or 0)
    except Exception:
        logger.debug("_read_stream_snippet: max_bytes conversion failed", exc_info=True)
        max_bytes = 8192
    max_bytes = max(0, min(max_bytes, 64 * 1024))
    if max_bytes <= 0:
        return ""
    try:
        timeout = float(timeout or 0)
    except Exception:
        logger.debug("_read_stream_snippet: timeout conversion failed", exc_info=True)
        timeout = 0.2
    timeout = max(0.05, min(timeout, 2.0))

    try:
        # asyncio.StreamReader: read(n)
        coro = getattr(stream, "read", None)
        if not callable(coro):
            return ""
        data = await safe_wait_for(coro(max_bytes), timeout=timeout, action="ACP stream read")
        if not data:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, (bytes, bytearray)):
            return bytes(data).decode("utf-8", errors="ignore")
        return str(data)
    except Exception:
        logger.debug("_read_stream_snippet: stream read failed", exc_info=True)
        return ""


async def _drain_loop_callbacks(rounds: int = 3) -> None:
    """Let subprocess pipe close callbacks run before the owning loop is closed."""
    for _ in range(max(1, int(rounds or 1))):
        await asyncio.sleep(0)


async def _call_set_session_config_option_method(
    method: Callable[..., Any],
    *,
    session_id: str,
    config_id: str,
    value: str,
) -> None:
    """Call generated ACP config-option setters across SDK spelling variants."""
    try:
        await method(session_id=session_id, config_id=config_id, value=value)
        return
    except TypeError as first_error:
        try:
            await method(session_id=session_id, option_id=config_id, value=value)
            return
        except TypeError:
            raise first_error


async def _set_session_config_option(
    conn: object,
    *,
    session_id: str,
    config_id: str,
    value: str,
) -> bool | None:
    """Set an ACP session config option when that protocol is available.

    Returns True when the request was sent successfully, None when no config-option
    path is available on this SDK/connection. Any raised exception means the new
    protocol was available but failed, and callers should fail-close instead of
    trying the old session/set_model RPC.
    """
    for name in ("set_config_option", "setConfigOption", "set_session_config_option", "setSessionConfigOption"):
        method = getattr(conn, name, None)
        if callable(method):
            await _call_set_session_config_option_method(
                method,
                session_id=session_id,
                config_id=config_id,
                value=value,
            )
            return True

    raw_conn = getattr(conn, "_conn", None)
    if raw_conn is None:
        return None

    try:
        import acp.schema as acp_schema
        from acp.client.connection import AGENT_METHODS
        from acp.schema import SetSessionConfigOptionResponse
        from acp.utils import request_model_from_dict
    except Exception:
        logger.debug("ACP SDK does not expose set_config_option low-level helpers", exc_info=True)
        return None

    await request_model_from_dict(
        raw_conn,
        AGENT_METHODS["session_set_config_option"],
        (
            getattr(acp_schema, "SetSessionConfigOptionSelectRequest", None)
            or getattr(acp_schema, "SetSessionConfigOptionRequest")
        )(
            config_id=config_id,
            session_id=session_id,
            value=value,
        ),
        SetSessionConfigOptionResponse,
    )
    return True


class ACPSession:
    """Single ACP session — manages one agent process's full lifecycle.

    This is an async class. For synchronous usage, see SyncACPSession.
    """

    def __init__(
        self,
        agent_cmd: str,
        agent_args: list[str],
        cwd: str,
        env: Optional[dict[str, str]] = None,
        auto_approve: bool | None = None,
        capture_full_tool_content: bool = False,
    ):
        self._agent_cmd = agent_cmd
        self._agent_args = agent_args
        self._cwd = cwd
        self._env_override = dict(env) if isinstance(env, dict) else None
        self._auto_approve = auto_approve
        self._capture_full_tool_content = bool(capture_full_tool_content)
        self._conn = None  # ClientSideConnection
        self._proc = None  # subprocess
        self._ctx_manager = None  # async context manager
        self._session_id: Optional[str] = None
        self._state = ACPSessionState(
            session_id="",
            agent_type=agent_cmd,
            cwd=cwd,
        )
        self._client: Optional[GhostAPClient] = None
        self._tool_filter: Optional[Callable[[str, dict | None], bool]] = None
        self._event_handler: Optional[Callable[[ACPEvent], None]] = None
        self._event_generation = 0
        self._deferred_child_events: list[ACPEvent] = []
        self._logical_task_child_call_ids: set[str] = set()
        self._logical_task_child_source_ids: set[str] = set()
        self._retired_child_call_ids: dict[str, None] = {}
        self._retired_child_source_ids: dict[str, None] = {}
        self._handler_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._prompt_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._closing = False
        self._goal_known = not self._uses_official_codex_acp()
        self._goal: ACPGoalInfo | None = None
        self._thread_status_observed = False
        self._thread_status_known = False
        self._thread_status = ""
        self._lifecycle_revision = 0
        self._lifecycle_event: asyncio.Event | None = None
        self._lifecycle_event_loop: asyncio.AbstractEventLoop | None = None
        self._loading_session_id: str | None = None
        self._load_snapshot_observed = False
        self._force_dead = False
        self._transport_epoch = 0
        self._load_epoch = 0
        self._attempted_load_targets: set[str] = set()

    def _uses_official_codex_acp(self) -> bool:
        return any(
            "@agentclientprotocol/codex-acp" in str(arg)
            for arg in self._agent_args
        )

    def _wake_lifecycle_waiter(self) -> None:
        event = self._lifecycle_event
        owner_loop = self._lifecycle_event_loop
        if event is None or owner_loop is None:
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is owner_loop:
            event.set()
            return
        if not owner_loop.is_closed():
            owner_loop.call_soon_threadsafe(event.set)

    def _reset_lifecycle(self, *, goal_known: bool) -> None:
        with self._handler_lock:
            self._goal_known = goal_known
            self._goal = None
            self._thread_status_observed = False
            self._thread_status_known = False
            self._thread_status = ""
            self._deferred_child_events.clear()
            self._logical_task_child_call_ids.clear()
            self._logical_task_child_source_ids.clear()
            self._retired_child_call_ids.clear()
            self._retired_child_source_ids.clear()
            self._lifecycle_revision += 1
        self._wake_lifecycle_waiter()

    def _on_session_info(
        self,
        session_id: str,
        info: ACPSessionInfo,
        *,
        transport_epoch: int | None = None,
        load_epoch: int | None = None,
    ) -> None:
        """Apply a trusted control-plane update for the current/load target only."""
        with self._handler_lock:
            expected_session_id = (
                self._loading_session_id
                if self._loading_session_id is not None
                else self._session_id
            )
            if (
                self._closing
                or self._force_dead
                or (
                    transport_epoch is not None
                    and transport_epoch != self._transport_epoch
                )
                or (load_epoch is not None and load_epoch != self._load_epoch)
                or session_id != expected_session_id
                or not isinstance(info, ACPSessionInfo)
            ):
                return
            changed = False
            if info.goal_known:
                self._goal_known = True
                self._goal = info.goal
                changed = True
            if info.thread_status_known:
                self._thread_status_observed = True
                self._thread_status_known = True
                self._thread_status = info.thread_status
                changed = True
            if not changed:
                return
            if self._loading_session_id is not None:
                self._load_snapshot_observed = True
            self._lifecycle_revision += 1
        self._wake_lifecycle_waiter()

    def _bind_session_info_callback(self) -> None:
        client = self._client
        if client is None:
            return
        with self._handler_lock:
            transport_epoch = self._transport_epoch
            load_epoch = self._load_epoch
        client._on_session_info = lambda session_id, info: self._on_session_info(
            session_id,
            info,
            transport_epoch=transport_epoch,
            load_epoch=load_epoch,
        )

    async def _wait_for_lifecycle_change(self, revision: int) -> None:
        loop = asyncio.get_running_loop()
        with self._handler_lock:
            if self._closing:
                raise RuntimeError("ACP session is closing")
            if revision != self._lifecycle_revision:
                return
            if self._lifecycle_event_loop is not loop:
                self._lifecycle_event = asyncio.Event()
                self._lifecycle_event_loop = loop
            event = self._lifecycle_event
            event.clear()
        await event.wait()
        with self._handler_lock:
            if self._closing:
                raise RuntimeError("ACP session is closing")

    def _goal_requires_prompt_wait_locked(self) -> bool:
        """Keep collection attached until the provider proves quiescence."""
        if self._closing:
            raise RuntimeError("ACP session is closing")
        if self._uses_official_codex_acp() and not self._goal_known:
            return True
        if not self._goal_known:
            return False
        if self._goal is None:
            return False
        activity_state = self._goal.activity_state
        if activity_state is None:
            return True
        if activity_state:
            return True
        return self._thread_status_known and self._thread_status == "active"

    async def has_active_goal(self) -> bool:
        with self._handler_lock:
            if self._uses_official_codex_acp() and not self._goal_known:
                raise RuntimeError("ACP goal state is unknown")
            if self._goal is None:
                return False
            activity_state = self._goal.activity_state
            if activity_state is None:
                raise RuntimeError(
                    f"ACP goal status is unknown: {self._goal.status}"
                )
            return activity_state

    async def _pause_active_goal(self, *, propagate_errors: bool) -> bool:
        try:
            with self._handler_lock:
                goal_known = self._goal_known
                goal = self._goal if self._goal_known else None
                session_id = self._session_id
                closing = self._closing
            if closing:
                raise RuntimeError("ACP session is closing")
            if self._uses_official_codex_acp() and not goal_known:
                raise RuntimeError("ACP goal state is unknown")
            if goal is None:
                return True
            if goal.activity_state is None:
                raise RuntimeError(f"ACP goal status is unknown: {goal.status}")
            if not goal.is_active:
                return True
            if goal.control_method != _CODEX_GOAL_CONTROL_METHOD:
                return False
            if not self._conn or not session_id:
                return False
            response = await self._conn.ext_method(
                goal.control_method[1:],
                {"sessionId": session_id, "action": "pause"},
            )
            if not isinstance(response, Mapping):
                return False
            while True:
                with self._handler_lock:
                    if self._closing:
                        raise RuntimeError("ACP session is closing")
                    if (
                        self._uses_official_codex_acp()
                        and not self._goal_known
                    ):
                        raise RuntimeError("ACP goal state is unknown")
                    current_goal = self._goal if self._goal_known else None
                    if current_goal is not None and current_goal.activity_state is None:
                        raise RuntimeError(
                            f"ACP goal status is unknown: {current_goal.status}"
                        )
                    if current_goal is None or not current_goal.is_active:
                        return True
                    revision = self._lifecycle_revision
                await self._wait_for_lifecycle_change(revision)
        except Exception:
            if propagate_errors:
                raise
            return False

    async def pause_active_goal(self) -> bool:
        return await self._pause_active_goal(propagate_errors=True)

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def state(self) -> ACPSessionState:
        return self._state

    async def start(self) -> str:
        """Start agent process and establish ACP connection. Returns session_id."""
        with self._handler_lock:
            self._closing = False
            self._transport_epoch += 1
            self._load_epoch += 1
        settings = get_settings()
        auto_approve = (
            settings.acp_permission_auto_approve
            if self._auto_approve is None
            else self._auto_approve
        )
        client = GhostAPClient(
            on_event=self._dispatch_event,
            auto_approve=auto_approve,
            root_dir=self._cwd,
            capture_full_tool_content=self._capture_full_tool_content,
        )
        self._client = client
        self._bind_session_info_callback()
        if self._tool_filter is not None:
            client.set_tool_filter(self._tool_filter)
        # Raise the stdio stream buffer limit to handle large agent responses.
        # Default asyncio limit (64KB) causes "Separator is found, but chunk is
        # longer than limit" on verbose JSON-RPC messages from long-running tasks.
        transport_kwargs = {}
        buf_limit = getattr(settings, "acp_stream_buffer_limit", 0)
        if buf_limit and buf_limit > 0:
            transport_kwargs["limit"] = buf_limit

        # Claude Code CLI refuses to launch inside another Claude Code session when
        # `CLAUDECODE` is present. Even when we spawn an ACP server (e.g. `claude acp serve`)
        # via an override, we must explicitly drop this guard env to avoid nested-session crash.
        from ..utils.env import build_clean_env
        base = dict(self._env_override) if isinstance(self._env_override, dict) else None
        env = build_clean_env(base)

        self._ctx_manager = spawn_agent_process(
            client,
            self._agent_cmd,
            *self._agent_args,
            env=env,
            cwd=self._cwd,
            transport_kwargs=transport_kwargs or None,
            queue=LateFrameTolerantMessageQueue(),
        )

        phase = ""
        try:
            phase = "spawn"
            self._conn, self._proc = await self._ctx_manager.__aenter__()
            with self._handler_lock:
                self._force_dead = False
                self._attempted_load_targets.clear()

            # Initialize protocol
            phase = "initialize"
            await self._conn.initialize(protocol_version=1)

            # Create new session
            phase = "new_session"
            session_resp = await self._conn.new_session(cwd=self._cwd)
            self._session_id = session_resp.session_id
            self._state.session_id = self._session_id
            self._state.is_active = True
            if self._uses_official_codex_acp():
                self._reset_lifecycle(goal_known=True)

            logger.info("[ACP:%s] Session started: %s", self._agent_cmd, self._session_id[:8])
            return self._session_id
        except Exception as e:
            # Best-effort capture process outputs for debugging. Only for startup failures.
            rc = None
            try:
                rc = getattr(self._proc, "returncode", None)
            except Exception:
                logger.debug("[ACP:%s] returncode extraction failed during start error", self._agent_cmd, exc_info=True)
                rc = None
            stderr_snip = ""
            stdout_snip = ""
            try:
                stderr_snip = await _read_stream_snippet(getattr(self._proc, "stderr", None))
            except Exception:
                logger.debug("[ACP:%s] stderr snippet read failed during start error", self._agent_cmd, exc_info=True)
                stderr_snip = ""
            # Only read stdout if stderr is empty; stdout may contain useful banner/error.
            if not stderr_snip:
                try:
                    stdout_snip = await _read_stream_snippet(getattr(self._proc, "stdout", None))
                except Exception:
                    logger.debug("[ACP:%s] stdout snippet read failed during start error", self._agent_cmd, exc_info=True)
                    stdout_snip = ""

            raise ACPStartupError(
                "ACP 启动失败",
                agent_cmd=self._agent_cmd,
                agent_args=list(self._agent_args or []),
                cwd=self._cwd,
                returncode=rc,
                stdout_snippet=stdout_snip,
                stderr_snippet=stderr_snip,
                fail_phase=phase or "unknown",
                cause=e,
            ) from e

    def set_tool_filter(self, filter_fn: Optional[Callable[[str, dict | None], bool]]) -> None:
        """Install a per-session tool filter on the local ACP client callbacks."""
        self._tool_filter = filter_fn
        if self._client is not None:
            self._client.set_tool_filter(filter_fn)

    async def load_session(self, session_id: str) -> None:
        """Load an existing session by ID (for resume)."""
        if not self._conn:
            raise RuntimeError("Connection not established. Call start() first.")
        target_session_id = str(session_id or "").strip()
        if not target_session_id:
            raise ValueError("ACP resume session_id is required")
        previous_session_id = self._session_id
        previous_goal_state = (
            self._goal_known,
            self._goal,
            self._thread_status_observed,
            self._thread_status_known,
            self._thread_status,
        )
        with self._handler_lock:
            if self._loading_session_id is not None:
                raise RuntimeError("ACP resume load is already in progress")
            if target_session_id in self._attempted_load_targets:
                self._force_dead = True
                raise RuntimeError(
                    "ACP resume target requires a new transport before retry"
                )
            self._attempted_load_targets.add(target_session_id)
            self._loading_session_id = target_session_id
            self._load_snapshot_observed = False
            self._load_epoch += 1
        self._bind_session_info_callback()
        self._reset_lifecycle(goal_known=not self._uses_official_codex_acp())
        try:
            try:
                await self._conn.load_session(
                    cwd=self._cwd,
                    session_id=target_session_id,
                )
            except RequestError as exc:
                with self._handler_lock:
                    clean_rejection = (
                        exc.code == -32602
                        and not self._load_snapshot_observed
                    )
                if clean_rejection:
                    raise ACPResumeRejected(
                        f"ACP resume explicitly rejected: {target_session_id}"
                    ) from exc
                raise
            self._session_id = target_session_id
            self._state.session_id = target_session_id
            if self._uses_official_codex_acp():
                while True:
                    with self._handler_lock:
                        if self._goal_known:
                            break
                        revision = self._lifecycle_revision
                    await self._wait_for_lifecycle_change(revision)
        except BaseException as exc:
            ambiguous_rejection: RequestError | None = None
            with self._handler_lock:
                clean_rejection = (
                    isinstance(exc, ACPResumeRejected)
                    and not self._load_snapshot_observed
                )
                if clean_rejection:
                    self._session_id = previous_session_id
                    self._state.session_id = previous_session_id or ""
                    (
                        self._goal_known,
                        self._goal,
                        self._thread_status_observed,
                        self._thread_status_known,
                        self._thread_status,
                    ) = previous_goal_state
                else:
                    # The load RPC may already have committed.  Keep the only
                    # identity that could now own this transport until the
                    # manager retires it; restoring the temporary new-session
                    # identity would create a split-brain session object.
                    self._session_id = target_session_id
                    self._state.session_id = target_session_id
                    self._force_dead = True
                    if (
                        isinstance(exc, ACPResumeRejected)
                        and isinstance(exc.__cause__, RequestError)
                    ):
                        ambiguous_rejection = exc.__cause__
                self._lifecycle_revision += 1
                self._loading_session_id = None
                self._load_epoch += 1
            self._bind_session_info_callback()
            self._wake_lifecycle_waiter()
            if ambiguous_rejection is not None:
                raise ambiguous_rejection from None
            raise
        with self._handler_lock:
            self._loading_session_id = None
        logger.info(
            "[ACP:%s] Session loaded: %s",
            self._agent_cmd,
            target_session_id[:8],
        )

    async def health_check(self, timeout: float = 2.0) -> bool:
        """Best-effort health check of ACP connection.

        We consider the server healthy only if:
        - underlying process is alive
        - JSON-RPC connection responds to a lightweight request within timeout
        """
        try:
            if not self._proc or self._proc.returncode is not None:
                return False
            if not self._conn:
                return False
            if not self._session_id:
                return False
            # Use a non-mutating roundtrip request. Reloading the active session
            # is not a health check: some agents reject it, while others may
            # reset session-local configuration or context.
            await safe_wait_for(
                self._conn.list_sessions(cwd=self._cwd),
                timeout=timeout,
                action="ACP 健康检查",
            )
            return True
        except Exception:
            logger.debug("[ACP:%s] health_check failed", self._agent_cmd, exc_info=True)
            return False

    async def prompt(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        *,
        await_goal_quiescence: bool = True,
        await_child_quiescence: bool = False,
        replay_deferred_child_events: bool = False,
    ) -> PromptResult:
        """Send one prompt, rejecting concurrent use of the same ACP session."""
        if not self._prompt_lock.acquire(blocking=False):
            raise RuntimeError("ACP prompt is already running for this session")
        try:
            with self._handler_lock:
                if self._closing:
                    raise RuntimeError("ACP session is closing")
            return await self._prompt_once(
                text,
                on_event=on_event,
                await_goal_quiescence=await_goal_quiescence,
                await_child_quiescence=await_child_quiescence,
                replay_deferred_child_events=replay_deferred_child_events,
            )
        finally:
            self._prompt_lock.release()

    async def _prompt_once(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        *,
        await_goal_quiescence: bool = True,
        await_child_quiescence: bool = False,
        replay_deferred_child_events: bool = False,
    ) -> PromptResult:
        """Run a prompt while the cross-thread prompt ownership gate is held."""
        if not self._conn or not self._session_id:
            raise RuntimeError("Session not started. Call start() first.")

        start_ts = time.time()

        codex_child_monitor = None
        if self._uses_official_codex_acp():
            try:
                from .codex_rollout_reconciliation import (
                    CodexChildLifecycleMonitor,
                )

                env_override = self._env_override or {}
                raw_codex_home = str(
                    env_override.get("CODEX_HOME") or ""
                ).strip()
                if raw_codex_home:
                    codex_home = raw_codex_home
                else:
                    raw_home = str(env_override.get("HOME") or "").strip()
                    codex_home = (
                        str(Path(raw_home).expanduser() / ".codex")
                        if raw_home
                        else None
                    )
                codex_child_monitor = CodexChildLifecycleMonitor(
                    parent_session_id=self._session_id,
                    cwd=self._cwd,
                    logical_task_started_at=start_ts,
                    codex_home=codex_home,
                )
            except Exception:
                logger.warning(
                    "[ACP:CODEX] child lifecycle monitor initialization failed",
                    exc_info=True,
                )

        # Collector aggregates text/tool calls/plan/modified_files.
        collected_tool_call_snapshots: list[Any] = []
        emitted_image_ids: set[str] = set()
        image_budget = LocalImageEmissionBudget()
        result = PromptResult(stop_reason="")
        last_event_monotonic = [time.monotonic()]
        image_snapshot: object = {}
        should_replay_deferred_children = (
            replay_deferred_child_events or await_child_quiescence
        )

        async def _drain_prompt_tail() -> None:
            quiet_s = 0.05
            max_drain_s = 0.15
            drain_started = time.monotonic()
            while time.monotonic() - drain_started < max_drain_s:
                quiet_for = time.monotonic() - max(
                    last_event_monotonic[0],
                    drain_started,
                )
                if quiet_for >= quiet_s:
                    break
                await asyncio.sleep(min(0.005, quiet_s - quiet_for))

        def _notify_prompt_event(ev: ACPEvent) -> None:
            if on_event:
                try:
                    on_event(ev)
                except Exception as exc:
                    logger.warning(
                        "[ACP] on_event callback error: %s",
                        get_error_detail(exc),
                    )

        def _collector(ev: ACPEvent):
            accepted = False
            lifecycle_changed = False
            with self._handler_lock:
                is_current_generation = (
                    self._event_generation == event_generation
                    and self._event_handler is _collector
                )
                if is_current_generation:
                    filtered_event = _filter_prior_task_child_sources(
                        ev,
                        prior_task_child_source_ids,
                    )
                    if filtered_event is None:
                        return
                    ev = filtered_event
                    child_lifecycle_event = _is_child_lifecycle_event(ev)
                    child_call_id = (
                        _child_call_id(ev)
                        if child_lifecycle_event
                        else ""
                    )
                    if (
                        child_call_id
                        and child_call_id in prior_task_child_call_ids
                    ):
                        if _starts_child_generation(ev):
                            prior_task_child_call_ids.discard(
                                child_call_id
                            )
                        elif _child_source_ids(ev):
                            prior_task_child_call_ids.discard(
                                child_call_id
                            )
                            self._retired_child_call_ids.pop(
                                child_call_id,
                                None,
                            )
                        else:
                            return
                    if _starts_child_generation(ev):
                        generated_sources = _child_source_ids(ev)
                        if child_call_id:
                            self._retired_child_call_ids.pop(
                                child_call_id,
                                None,
                            )
                        for generated_source in generated_sources:
                            self._retired_child_source_ids.pop(
                                generated_source,
                                None,
                            )
                    accepted = True
                    last_event_monotonic[0] = time.monotonic()
                    try:
                        if ev.event_type == ACPEventType.TEXT_CHUNK:
                            result.add_text(ev.text or "")
                        elif ev.event_type in (
                            ACPEventType.TOOL_CALL_START,
                            ACPEventType.TOOL_CALL_UPDATE,
                            ACPEventType.TOOL_CALL_DONE,
                        ):
                            if ev.tool_call:
                                collected_tool_call_snapshots.append(
                                    ev.tool_call
                                )
                                for p in ev.tool_call.locations or []:
                                    if p:
                                        result.add_modified_file(p)
                        elif ev.event_type == ACPEventType.IMAGE_CHUNK and ev.image:
                            emitted_image_ids.add(ev.image.image_id)
                        elif ev.event_type == ACPEventType.PLAN_UPDATE:
                            result.set_plan(ev.plan)
                        if codex_child_monitor is not None and ev.tool_call:
                            codex_child_monitor.observe_tool_call(ev.tool_call)
                        if child_lifecycle_event:
                            if child_call_id:
                                self._logical_task_child_call_ids.add(
                                    child_call_id
                                )
                            self._logical_task_child_source_ids.update(
                                _child_source_ids(ev)
                            )
                            self._lifecycle_revision += 1
                            lifecycle_changed = True
                    except Exception:
                        logger.debug(
                            "plan_update event processing failed",
                            exc_info=True,
                        )
            if not accepted:
                return
            if lifecycle_changed:
                self._wake_lifecycle_waiter()
            _notify_prompt_event(ev)

        with self._handler_lock:
            prior_task_child_call_ids: set[str] = set()
            prior_task_child_source_ids: set[str] = set()
            if not should_replay_deferred_children:
                _remember_bounded_identities(
                    self._retired_child_call_ids,
                    self._logical_task_child_call_ids,
                )
                _remember_bounded_identities(
                    self._retired_child_source_ids,
                    self._logical_task_child_source_ids,
                )
                prior_task_child_call_ids = set(
                    self._retired_child_call_ids
                )
                prior_task_child_source_ids = set(
                    self._retired_child_source_ids
                )
                self._logical_task_child_call_ids.clear()
                self._logical_task_child_source_ids.clear()
            deferred_child_events = self._deferred_child_events
            self._deferred_child_events = []
            if should_replay_deferred_children:
                for deferred_event in deferred_child_events:
                    if deferred_event.tool_call is not None:
                        collected_tool_call_snapshots.append(
                            deferred_event.tool_call
                        )
            if should_replay_deferred_children and deferred_child_events:
                last_event_monotonic[0] = time.monotonic()
            self._event_generation += 1
            event_generation = self._event_generation
            self._event_handler = _collector
        collector_detached = False
        child_monitor_stop = asyncio.Event()
        child_monitor_task: asyncio.Task[None] | None = None

        def _emit_codex_child_lifecycle() -> None:
            if codex_child_monitor is None:
                return
            try:
                evidence = codex_child_monitor.poll(ended_at=time.time())
            except Exception:
                logger.warning(
                    "[ACP:CODEX] child lifecycle polling failed closed",
                    exc_info=True,
                )
                return
            if evidence is not None:
                _collector(
                    ACPEvent(
                        event_type=ACPEventType.TOOL_CALL_DONE,
                        tool_call=evidence,
                    )
                )

        async def _poll_codex_child_lifecycle() -> None:
            from .codex_rollout_reconciliation import (
                CODEX_CHILD_LIFECYCLE_POLL_INTERVAL_S,
            )

            while not child_monitor_stop.is_set():
                _emit_codex_child_lifecycle()
                try:
                    await asyncio.wait_for(
                        child_monitor_stop.wait(),
                        timeout=CODEX_CHILD_LIFECYCLE_POLL_INTERVAL_S,
                    )
                except TimeoutError:
                    continue

        # Deferred child events already participate in the authoritative result
        # above. Reconciliation turns must also project the exact same evidence
        # into the active programming card. Keep ordinary user turns isolated so
        # a late event from the previous task cannot appear on a new task's card.
        if should_replay_deferred_children:
            for deferred_event in deferred_child_events:
                _notify_prompt_event(deferred_event)

        def _discover_changed_images() -> None:
            if self._client is None:
                return
            try:
                with self._handler_lock:
                    image_budget.seen_image_ids.update(emitted_image_ids)

                def _emit_new_image(event: ACPEvent) -> None:
                    if (
                        event.image is not None
                        and event.image.image_id in emitted_image_ids
                    ):
                        return
                    _collector(event)

                emit_referenced_changed_local_image_events(
                    self._cwd,
                    image_snapshot,
                    result.text,
                    _emit_new_image,
                    budget=image_budget,
                    release_snapshot=False,
                )
            except Exception:
                logger.debug(
                    "[ACP:%s] changed image discovery failed",
                    self._agent_cmd,
                    exc_info=True,
                )

        try:
            self._state.message_count += 1

            self._state.last_active = time.time()
            image_snapshot = (
                self._client.snapshot_local_images()
                if self._client is not None
                else {}
            )

            if codex_child_monitor is not None:
                child_monitor_task = asyncio.create_task(
                    _poll_codex_child_lifecycle()
                )

            response: PromptResponse = await self._conn.prompt(
                session_id=self._session_id,
                prompt=[text_block(text)],
            )

            while True:
                # A continuation can produce text and referenced images after
                # PromptResponse.  Drain and rescan on every lifecycle revision
                # before making the final atomic quiescence decision.
                await _drain_prompt_tail()
                _discover_changed_images()
                _emit_codex_child_lifecycle()
                with self._handler_lock:
                    goal_requires_wait = self._goal_requires_prompt_wait_locked()
                    child_requires_wait = (
                        await_child_quiescence
                        and has_transient_child_lifecycle(
                            collected_tool_call_snapshots
                        )
                    )
                    should_wait = (
                        (await_goal_quiescence and goal_requires_wait)
                        or child_requires_wait
                    )
                    lifecycle_revision = self._lifecycle_revision
                    if not should_wait:
                        if (
                            self._event_generation != event_generation
                            or self._event_handler is not _collector
                        ):
                            raise RuntimeError(
                                "ACP prompt collector ownership changed"
                            )
                        result.goal = self._goal if self._goal_known else None
                        self._event_handler = None
                        self._event_generation += 1
                        collector_detached = True
                        break
                await self._wait_for_lifecycle_change(lifecycle_revision)
                await _drain_prompt_tail()

        finally:
            child_monitor_stop.set()
            if child_monitor_task is not None:
                child_monitor_task.cancel()
                try:
                    await child_monitor_task
                except asyncio.CancelledError:
                    pass
            if self._client is not None:
                self._client.release_local_image_snapshot(image_snapshot)
            if not collector_detached:
                with self._handler_lock:
                    if (
                        self._event_generation == event_generation
                        and self._event_handler is _collector
                    ):
                        self._event_handler = None
                        self._event_generation += 1

        # Finalize named snapshots at their latest observation position while
        # preserving every anonymous call as an independent event.
        try:
            result.tool_calls = merge_tool_call_sequence(
                collected_tool_call_snapshots
            )
        except Exception:
            # Never turn malformed provider evidence into an empty successful
            # result.  The outcome reducer will classify raw unknown objects or
            # non-terminal snapshots fail-closed.
            result.tool_calls = list(collected_tool_call_snapshots)
            logger.debug("[ACP:%s] tool_calls finalization failed", self._agent_cmd, exc_info=True)

        result.stop_reason = str(getattr(response, "stop_reason", "") or "")
        # Best-effort: attach local tool results (execute/read/write/permission) produced during this prompt.
        try:
            store = ACPHistoryStore()
            entries = store.load(self._session_id, limit=2000)
            end_ts = time.time()
            windowed = [
                e
                for e in entries
                if isinstance(e, dict) and (e.get("ts") or 0) >= start_ts and (e.get("ts") or 0) <= end_ts
            ]
            result.ingest_history(windowed)
        except Exception:
            logger.debug("failed to ingest history", exc_info=True)

        return result

    async def set_config_option(self, config_id: str, value: str) -> bool:
        """Set one ACP session config option without legacy RPC fallback."""
        if not self._conn or not self._session_id:
            raise RuntimeError("Session not started. Call start() first.")
        option_id = str(config_id or "").strip()
        option_value = str(value or "").strip()
        if not option_id or not option_value:
            return False
        try:
            config_result = await _set_session_config_option(
                self._conn,
                session_id=self._session_id,
                config_id=option_id,
                value=option_value,
            )
            if config_result is True:
                logger.info(
                    "[ACP:%s] Config option applied: config_id=%s (session=%s)",
                    self._agent_cmd,
                    option_id,
                    self._session_id[:8],
                )
                return True
            return False
        except Exception as e:
            logger.warning(
                "[ACP:%s] set_config_option failed: config_id=%s err=%s",
                self._agent_cmd,
                option_id,
                get_error_detail(e),
            )
            return False

    async def set_model(self, model_id: str) -> bool:
        """Switch the model for this session via ACP protocol.

        Newer ACP adapters expose models as session config options and expect
        ``session/set_config_option`` with ``configId=model``. Use that protocol
        first and fail-close if it is present but rejected. Only fall back to the
        legacy ``session/set_model`` RPC when no config-option path exists.
        """
        if not self._conn or not self._session_id:
            raise RuntimeError("Session not started. Call start() first.")
        try:
            config_result = await _set_session_config_option(
                self._conn,
                session_id=self._session_id,
                config_id="model",
                value=model_id,
            )
            if config_result is True:
                logger.info(
                    "[ACP:%s] Model switched via config option: %s (session=%s)",
                    self._agent_cmd,
                    model_id,
                    self._session_id[:8],
                )
                return True
        except Exception as e:
            logger.warning(
                "[ACP:%s] set_model via config option failed: %s",
                self._agent_cmd,
                get_error_detail(e),
            )
            return False

        try:
            await self._conn.set_session_model(model_id=model_id, session_id=self._session_id)
            logger.info("[ACP:%s] Model switched to: %s (session=%s)", self._agent_cmd, model_id, self._session_id[:8])
            return True
        except Exception as e:
            logger.warning("[ACP:%s] set_model failed (agent may not support it): %s", self._agent_cmd, get_error_detail(e))
            return False

    async def cancel(self, timeout: float | None = None) -> None:
        """Cancel the current prompt execution."""
        if self._conn and self._session_id:
            pause_error: BaseException | None = None
            try:
                pause_operation = self._pause_active_goal(propagate_errors=True)
                if timeout is None:
                    paused = await pause_operation
                else:
                    pause_budget = max(0.001, float(timeout) / 2.0)
                    paused = await safe_wait_for(
                        pause_operation,
                        timeout=pause_budget,
                        action="ACP Goal 暂停",
                    )
                if await self.has_active_goal() and not paused:
                    raise RuntimeError("ACP active goal pause was not confirmed")
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                pause_error = exc

            cancel_error: BaseException | None = None
            try:
                await self._conn.cancel(session_id=self._session_id)
            except BaseException as exc:
                cancel_error = exc

            if pause_error is not None and cancel_error is not None:
                raise BaseExceptionGroup(
                    "ACP goal pause and turn cancellation both failed",
                    [pause_error, cancel_error],
                )
            if pause_error is not None:
                raise pause_error
            if cancel_error is not None:
                raise cancel_error

    async def close(self) -> None:
        """Close session and terminate agent process."""
        self._state.is_active = False
        with self._handler_lock:
            self._closing = True
            self._event_handler = None
            self._event_generation += 1
            self._lifecycle_revision += 1
            self._transport_epoch += 1
            self._load_epoch += 1
        self._wake_lifecycle_waiter()
        if self._client is not None:
            self._client.release_active_local_image_snapshot()
        if self._ctx_manager:
            try:
                await self._ctx_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(
                    "[ACP:%s] transport termination failed: %s",
                    self._agent_cmd,
                    get_error_detail(e),
                    exc_info=True,
                )
                raise
            self._ctx_manager = None
            self._conn = None
            self._proc = None
            await _drain_loop_callbacks()
        elif self._conn is not None or self._proc is not None:
            raise RuntimeError(
                "ACP transport handles exist without their process context"
            )
        logger.info("[ACP:%s] Session closed: %s", self._agent_cmd, (self._session_id or "none")[:8])

    def _dispatch_event(self, event: ACPEvent) -> None:
        """Dispatch event to the current handler."""
        deferred = False
        with self._handler_lock:
            handler = self._event_handler
            if (
                handler is None
                and not self._closing
                and _is_child_lifecycle_event(event)
            ):
                self._deferred_child_events.append(event)
                child_call_id = _child_call_id(event)
                if child_call_id:
                    self._logical_task_child_call_ids.add(
                        child_call_id
                    )
                self._logical_task_child_source_ids.update(
                    _child_source_ids(event)
                )
                if (
                    len(self._deferred_child_events)
                    > _MAX_DEFERRED_CHILD_EVENTS
                ):
                    del self._deferred_child_events[
                        :-_MAX_DEFERRED_CHILD_EVENTS
                    ]
                self._lifecycle_revision += 1
                deferred = True
        if deferred:
            self._wake_lifecycle_waiter()
        if handler:
            try:
                handler(event)
            except Exception as e:
                logger.debug("[ACP] Event handler error: %s", get_error_detail(e))
