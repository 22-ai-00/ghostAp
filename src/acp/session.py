"""ACP session — manages a single ACP agent process lifecycle.

Wraps the ACP SDK's spawn_agent_process to provide a clean interface
for starting sessions, sending prompts, and receiving structured events.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Optional

from acp.helpers import text_block
from acp.schema import PromptResponse
from acp.stdio import spawn_agent_process

from ..config import get_settings
from ..utils.async_helpers import safe_wait_for
from ..utils.errors import get_error_detail
from .client import (
    ACPHistoryStore,
    GhostAPClient,
    emit_referenced_changed_local_image_events,
)
from .models import ACPEvent, ACPEventType, ACPSessionState, PromptResult

logger = logging.getLogger(__name__)


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

    def __init__(self, agent_cmd: str, agent_args: list[str], cwd: str, env: Optional[dict[str, str]] = None):
        self._agent_cmd = agent_cmd
        self._agent_args = agent_args
        self._cwd = cwd
        self._env_override = dict(env) if isinstance(env, dict) else None
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
        self._handler_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._prompt_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._closing = False

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
        settings = get_settings()
        client = GhostAPClient(
            on_event=self._dispatch_event,
            auto_approve=settings.acp_permission_auto_approve,
            root_dir=self._cwd,
        )
        self._client = client
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
        )

        phase = ""
        try:
            phase = "spawn"
            self._conn, self._proc = await self._ctx_manager.__aenter__()

            # Initialize protocol
            phase = "initialize"
            await self._conn.initialize(protocol_version=1)

            # Create new session
            phase = "new_session"
            session_resp = await self._conn.new_session(cwd=self._cwd)
            self._session_id = session_resp.session_id
            self._state.session_id = self._session_id
            self._state.is_active = True

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

    def get_tool_filter(self) -> Optional[Callable[[str, dict | None], bool]]:
        """Return the currently installed ACP callback tool filter."""
        return self._tool_filter

    async def load_session(self, session_id: str) -> None:
        """Load an existing session by ID (for resume)."""
        if not self._conn:
            raise RuntimeError("Connection not established. Call start() first.")
        await self._conn.load_session(cwd=self._cwd, session_id=session_id)
        self._session_id = session_id
        self._state.session_id = session_id
        logger.info("[ACP:%s] Session loaded: %s", self._agent_cmd, session_id[:8])

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

    async def prompt(self, text: str, on_event: Optional[Callable[[ACPEvent], None]] = None) -> PromptResult:
        """Send one prompt, rejecting concurrent use of the same ACP session."""
        if not self._prompt_lock.acquire(blocking=False):
            raise RuntimeError("ACP prompt is already running for this session")
        try:
            with self._handler_lock:
                if self._closing:
                    raise RuntimeError("ACP session is closing")
            return await self._prompt_once(text, on_event=on_event)
        finally:
            self._prompt_lock.release()

    async def _prompt_once(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
    ) -> PromptResult:
        """Run a prompt while the cross-thread prompt ownership gate is held."""
        if not self._conn or not self._session_id:
            raise RuntimeError("Session not started. Call start() first.")

        start_ts = time.time()

        # Collector aggregates text/tool calls/plan/modified_files.
        collected_tool_calls: dict[str, Any] = {}
        emitted_image_ids: set[str] = set()
        result = PromptResult(stop_reason="")
        last_event_monotonic = [time.monotonic()]
        image_snapshot: object = {}

        def _collector(ev: ACPEvent):
            with self._handler_lock:
                is_current_generation = (
                    self._event_generation == event_generation
                    and self._event_handler is _collector
                )
            if not is_current_generation:
                return

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
                        # Keep the latest state per tool_call_id
                        collected_tool_calls[ev.tool_call.id] = ev.tool_call
                        for p in ev.tool_call.locations or []:
                            if p:
                                result.add_modified_file(p)
                elif ev.event_type == ACPEventType.IMAGE_CHUNK and ev.image:
                    emitted_image_ids.add(ev.image.image_id)
                elif ev.event_type == ACPEventType.PLAN_UPDATE:
                    result.set_plan(ev.plan)
            except Exception:
                logger.debug("plan_update event processing failed", exc_info=True)
            if on_event:
                try:
                    on_event(ev)
                except Exception as exc:
                    logger.warning("[ACP] on_event callback error: %s", get_error_detail(exc))

        with self._handler_lock:
            self._event_generation += 1
            event_generation = self._event_generation
            self._event_handler = _collector
        try:
            self._state.message_count += 1

            self._state.last_active = time.time()
            image_snapshot = (
                self._client.snapshot_local_images()
                if self._client is not None
                else {}
            )

            response: PromptResponse = await self._conn.prompt(
                session_id=self._session_id,
                prompt=[text_block(text)],
            )

            # Race guard: some ACP agents (or stdio scheduling) may deliver the final
            # PromptResponse slightly before their last `session/update`. Drain until
            # the stream has been quiet for a short window so late text *and media*
            # arrive before the handler is cleared and the card becomes terminal.
            try:
                quiet_s = 0.2 if "tui2acp" in self._agent_cmd else 0.05
                max_drain_s = 0.6 if "tui2acp" in self._agent_cmd else 0.15
                drain_started = time.monotonic()
                while time.monotonic() - drain_started < max_drain_s:
                    quiet_for = time.monotonic() - max(
                        last_event_monotonic[0],
                        drain_started,
                    )
                    if quiet_for >= quiet_s:
                        break
                    await asyncio.sleep(min(0.005, quiet_s - quiet_for))
            except Exception:
                logger.debug("grace window wait failed", exc_info=True)

            if self._client is not None:
                try:
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
                    )
                except Exception:
                    logger.debug(
                        "[ACP:%s] changed image discovery failed",
                        self._agent_cmd,
                        exc_info=True,
                    )

        finally:
            if self._client is not None:
                self._client.release_local_image_snapshot(image_snapshot)
            with self._handler_lock:
                if (
                    self._event_generation == event_generation
                    and self._event_handler is _collector
                ):
                    self._event_handler = None
                    self._event_generation += 1

        # Finalize aggregated tool call list (preserve insertion order by first-seen)
        try:
            # If we have a dict keyed by id, the values are the latest states.
            result.tool_calls = list(collected_tool_calls.values())
        except Exception:
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

    async def cancel(self) -> None:
        """Cancel the current prompt execution."""
        if self._conn and self._session_id:
            await self._conn.cancel(session_id=self._session_id)

    async def close(self) -> None:
        """Close session and terminate agent process."""
        self._state.is_active = False
        with self._handler_lock:
            self._closing = True
            self._event_handler = None
            self._event_generation += 1
        if self._client is not None:
            self._client.release_active_local_image_snapshot()
        if self._ctx_manager:
            try:
                await self._ctx_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("[ACP:%s] Error closing session: %s", self._agent_cmd, get_error_detail(e))
            self._ctx_manager = None
            self._conn = None
            self._proc = None
            await _drain_loop_callbacks()
        logger.info("[ACP:%s] Session closed: %s", self._agent_cmd, (self._session_id or "none")[:8])

    def _dispatch_event(self, event: ACPEvent) -> None:
        """Dispatch event to the current handler."""
        with self._handler_lock:
            handler = self._event_handler
        if handler:
            try:
                handler(event)
            except Exception as e:
                logger.debug("[ACP] Event handler error: %s", get_error_detail(e))
