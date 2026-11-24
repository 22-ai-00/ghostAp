"""Rate-limit and model-failure aware session wrappers."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from typing import Any, Callable, Optional

from ..acp.models import ACPEvent, PromptResult
from ..config import get_settings
from ..utils.errors import get_error_detail
from .model_diagnostics import (
    _default_compaction_action,
    _detect_rate_limit,
    _extract_model_from_agent_args,
    _replace_model_in_agent_args,
    classify_model_failure,
)
from .protocol import SyncSession, _SessionWrapper

logger = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    if name == "SyncACPSession":
        from ..acp.sync_adapter import SyncACPSession

        globals()["SyncACPSession"] = SyncACPSession
        return SyncACPSession
    if name == "SyncAGYCLISession":
        from .agy_cli import SyncAGYCLISession

        return SyncAGYCLISession
    if name == "resolve_agent_spec":
        from ..acp.sync_adapter import resolve_agent_spec

        return resolve_agent_spec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _prompt_kwargs(
    *,
    on_event: Optional[Callable[[ACPEvent], None]],
    timeout: Optional[int],
    idle_timeout: Optional[float],
    activity_predicate: Optional[Callable[[ACPEvent], bool]],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "on_event": on_event,
        "timeout": timeout,
    }
    if idle_timeout is not None:
        kwargs["idle_timeout"] = idle_timeout
    if activity_predicate is not None:
        kwargs["activity_predicate"] = activity_predicate
    return kwargs


class RateLimitAwareSession(_SessionWrapper):
    """Wraps a SyncSession with rate-limit-aware retry on send_prompt().

    Implements the full SyncSession protocol by explicit delegation (no __getattr__).
    """

    def __init__(
        self,
        inner: SyncSession,
        on_rate_limit: Optional[Callable[[int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        super().__init__(inner, cancel_event)
        self._on_rate_limit = on_rate_limit
        self._settings = get_settings()
        # Rate-limit state visible to status queries
        self.rate_limit_until: Optional[float] = None  # monotonic deadline

    def send_prompt(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
        idle_timeout: Optional[float] = None,
        activity_predicate: Optional[Callable[[ACPEvent], bool]] = None,
    ) -> PromptResult:
        prompt_kwargs = _prompt_kwargs(
            on_event=on_event,
            timeout=timeout,
            idle_timeout=idle_timeout,
            activity_predicate=activity_predicate,
        )
        if not self._settings.rate_limit_retry_enabled:
            return self._inner.send_prompt(text, **prompt_kwargs)

        max_retries = self._settings.rate_limit_max_retries
        max_wait = self._settings.rate_limit_max_wait
        base_wait = self._settings.rate_limit_base_wait
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                self.rate_limit_until = None
                return self._inner.send_prompt(text, **prompt_kwargs)
            except Exception as e:
                wait_hint = _detect_rate_limit(e)
                if wait_hint is None or attempt >= max_retries:
                    raise
                last_error = e
                wait_time = min(wait_hint or base_wait, max_wait)
                wait_time = max(wait_time, 1)

                # Notify caller (UI) — swallow callback exceptions
                try:
                    if self._on_rate_limit:
                        self._on_rate_limit(wait_time)
                except Exception:
                    logger.debug("RateLimitAwareSession: on_rate_limit callback failed", exc_info=True)

                logger.warning(
                    "[RateLimit] 限速检测，等待 %ds 后重试 (attempt=%d/%d): %s",
                    wait_time,
                    attempt + 1,
                    max_retries,
                    get_error_detail(e),
                )

                # Interruptible sleep: check cancel_event every second
                self.rate_limit_until = time.monotonic() + wait_time
                deadline = time.monotonic() + wait_time
                while time.monotonic() < deadline:
                    if self._cancel_event.is_set():
                        self.rate_limit_until = None
                        raise last_error  # re-raise original error on cancel
                    remaining = deadline - time.monotonic()
                    self._cancel_event.wait(timeout=min(remaining, 1.0))
                self.rate_limit_until = None

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        return self._inner.send_prompt(text, **prompt_kwargs)

class ModelFailureAwareSession(_SessionWrapper):
    """在 send_prompt 阶段处理模型侧错误（need compaction / loop / failover）。

    当前阶段（任务 4）：仅处理 need compaction：执行一次 compaction 动作并用同模型重试一次。
    后续任务会在此类中扩展 loop 检测与模型 failover。
    """

    def __init__(
        self,
        inner: SyncSession,
        *,
        compaction_action: Optional[Callable[[SyncSession], Optional[SyncSession]]] = None,
        on_rate_limit: Optional[Callable[[int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        super().__init__(inner, cancel_event)
        self._settings = get_settings()
        self._compaction_action = compaction_action
        self._on_rate_limit = on_rate_limit
        self._replacement_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._uncertain_sessions: list[SyncSession] = []
        # compaction loop detector (per wrapper instance)
        self._compaction_loop_events: list[float] = []

    def _loop_limits(self) -> tuple[float, int]:
        """读取 loop 检测参数（window_s, max_count）。"""
        try:
            window_s = float(getattr(self._settings, "model_failure_compaction_loop_window_s", 180.0) or 180.0)
        except Exception:
            logger.debug("ModelFailureAwareSession._loop_limits: window_s conversion failed", exc_info=True)
            window_s = 180.0
        try:
            max_count = int(getattr(self._settings, "model_failure_compaction_loop_max", 2) or 2)
        except Exception:
            logger.debug("ModelFailureAwareSession._loop_limits: max_count conversion failed", exc_info=True)
            max_count = 2
        window_s = max(0.0, window_s)
        max_count = max(1, max_count)
        return (window_s, max_count)

    def _record_compaction_event_and_check_loop(self) -> tuple[bool, int]:
        """记录一次 compaction 事件，并判断是否达到 loop 阈值。"""
        now = time.time()
        window_s, max_count = self._loop_limits()
        if window_s <= 0:
            # window<=0: 认为每次都在窗口内
            self._compaction_loop_events.append(now)
        else:
            self._compaction_loop_events = [
                t for t in self._compaction_loop_events if (now - float(t or 0.0)) <= window_s
            ]
            self._compaction_loop_events.append(now)
        n = len(self._compaction_loop_events)
        return (n >= max_count, n)

    def _unwrap_rate_limit(self) -> tuple[SyncSession, Callable[[SyncSession], SyncSession]]:
        """若 inner 是 RateLimitAwareSession，则解包到其底层 session，并提供 rewrap 函数。"""
        inner = self._inner
        if isinstance(inner, RateLimitAwareSession):
            base = getattr(inner, "_inner", None) or inner

            def _rewrap(new_base: SyncSession) -> SyncSession:
                return RateLimitAwareSession(
                    inner=new_base, on_rate_limit=self._on_rate_limit, cancel_event=self._cancel_event
                )

            return base, _rewrap

        def _id(new_base: SyncSession) -> SyncSession:
            return new_base

        return inner, _id

    def _do_compaction(self) -> bool:
        """执行一次 compaction，并在成功时替换 self._inner。"""
        action = self._compaction_action
        if action is None:
            # 默认行为：重建同 cmd/args 的 ACP session
            def action(s):
                return _default_compaction_action(session=s)

        base, rewrap = self._unwrap_rate_limit()
        return self._replace_inner_strictly(
            base=base,
            rewrap=rewrap,
            candidate_factory=lambda: action(base),
            operation="compaction",
        )

    @property
    def uncertain_sessions(self) -> tuple[SyncSession, ...]:
        """Sessions whose close could not be confirmed and must stay retryable."""
        with self._replacement_lock:
            return tuple(self._uncertain_sessions)

    @staticmethod
    def _close_confirmed(session: SyncSession, *, operation: str) -> bool:
        try:
            result = session.close()
        except Exception:
            logger.warning(
                "ModelFailureAwareSession %s close was not confirmed",
                operation,
                exc_info=True,
            )
            return False
        if result is False:
            logger.warning(
                "ModelFailureAwareSession %s close returned an unconfirmed result",
                operation,
            )
            return False
        return True

    def _remember_uncertain_session(self, session: SyncSession) -> None:
        if not any(existing is session for existing in self._uncertain_sessions):
            self._uncertain_sessions.append(session)

    def _close_rejected_candidate(
        self,
        candidate: SyncSession,
        *,
        operation: str,
    ) -> None:
        if not self._close_confirmed(candidate, operation=operation):
            self._remember_uncertain_session(candidate)

    def _replace_inner_strictly(
        self,
        *,
        base: SyncSession,
        rewrap: Callable[[SyncSession], SyncSession],
        candidate_factory: Callable[[], Optional[SyncSession]],
        operation: str,
        prepare_candidate: Optional[Callable[[SyncSession], None]] = None,
    ) -> bool:
        """Close old, prepare the candidate, then atomically swap."""
        with self._replacement_lock:
            if self._cancel_event.is_set():
                return False
            if not self._close_confirmed(base, operation=f"{operation} old session"):
                return False
            if self._cancel_event.is_set():
                return False

            new_base: Optional[SyncSession] = None
            replacement: Optional[SyncSession] = None
            try:
                new_base = candidate_factory()
                if new_base is None:
                    return False
                replacement = rewrap(new_base)
                if prepare_candidate is not None:
                    prepare_candidate(replacement)
                if self._cancel_event.is_set():
                    self._close_rejected_candidate(
                        replacement,
                        operation=f"{operation} cancelled candidate",
                    )
                    return False
            except Exception:
                logger.exception(
                    "ModelFailureAwareSession %s candidate preparation failed",
                    operation,
                )
                rejected = replacement or new_base
                if rejected is not None:
                    self._close_rejected_candidate(
                        rejected,
                        operation=f"{operation} rejected candidate",
                    )
                return False

            self._inner = replacement
            return True

    def close(self) -> None:
        """Close the current session and retry every uncertain candidate."""
        with self._replacement_lock:
            self._cancel_event.set()
            sessions = [self._inner, *self._uncertain_sessions]
            unique_sessions: list[SyncSession] = []
            for session in sessions:
                if not any(existing is session for existing in unique_sessions):
                    unique_sessions.append(session)

            uncertain: list[SyncSession] = []
            failures = 0
            for session in unique_sessions:
                if self._close_confirmed(session, operation="wrapper close"):
                    continue
                failures += 1
                if session is not self._inner:
                    uncertain.append(session)
            self._uncertain_sessions = uncertain
            if failures:
                raise RuntimeError(
                    f"unable to confirm closure of {failures} model session(s)"
                )

    def _parse_failover_map(self) -> dict[str, str]:
        """解析 failover 映射（from:to），兼容 JSON 数组模型值与 JSON 字典。"""
        raw = ""
        try:
            raw = str(getattr(self._settings, "model_failure_failover_map", "") or "")
        except Exception:
            logger.debug("ModelFailureAwareSession._parse_failover_map: settings read failed", exc_info=True)
            raw = ""
        stripped = raw.strip()
        if not stripped:
            return {}

        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                out: dict[str, str] = {}
                for raw_k, raw_v in parsed.items():
                    k = (
                        json.dumps(raw_k, ensure_ascii=False, separators=(",", ":"))
                        if isinstance(raw_k, list)
                        else str(raw_k or "").strip()
                    )
                    v = (
                        json.dumps(raw_v, ensure_ascii=False, separators=(",", ":"))
                        if isinstance(raw_v, list)
                        else str(raw_v or "").strip()
                    )
                    if k and v and k not in out:
                        out[k] = v
                if out:
                    return out

        chunks: list[str] = []
        buf: list[str] = []
        bracket_depth = 0
        brace_depth = 0
        in_quote: str | None = None
        escaped = False
        for ch in raw:
            if escaped:
                buf.append(ch)
                escaped = False
                continue
            if in_quote is not None:
                buf.append(ch)
                if ch == "\\":
                    escaped = True
                elif ch == in_quote:
                    in_quote = None
                continue
            if ch in {'"', "'"}:
                in_quote = ch
                buf.append(ch)
                continue
            if ch == "[":
                bracket_depth += 1
                buf.append(ch)
                continue
            if ch == "]":
                if bracket_depth > 0:
                    bracket_depth -= 1
                buf.append(ch)
                continue
            if ch == "{":
                brace_depth += 1
                buf.append(ch)
                continue
            if ch == "}":
                if brace_depth > 0:
                    brace_depth -= 1
                buf.append(ch)
                continue
            if (
                bracket_depth == 0
                and brace_depth == 0
                and (ch in {",", ";"} or ch.isspace())
            ):
                if buf:
                    chunks.append("".join(buf))
                    buf.clear()
                continue
            buf.append(ch)
        if buf:
            chunks.append("".join(buf))

        pairs: list[tuple[str, str]] = []
        for chunk in chunks:
            s = chunk.strip()
            if not s or ":" not in s:
                continue
            split_idx = -1
            b_depth = 0
            c_depth = 0
            q_char: str | None = None
            esc = False
            for idx, ch in enumerate(s):
                if esc:
                    esc = False
                    continue
                if q_char is not None:
                    if ch == "\\":
                        esc = True
                    elif ch == q_char:
                        q_char = None
                    continue
                if ch in {'"', "'"}:
                    q_char = ch
                    continue
                if ch == "[":
                    b_depth += 1
                    continue
                if ch == "]":
                    if b_depth > 0:
                        b_depth -= 1
                    continue
                if ch == "{":
                    c_depth += 1
                    continue
                if ch == "}":
                    if c_depth > 0:
                        c_depth -= 1
                    continue
                if ch == ":" and b_depth == 0 and c_depth == 0:
                    split_idx = idx
                    break
            if split_idx <= 0:
                continue
            a, b = s[:split_idx].strip(), s[split_idx + 1 :].strip()
            if a and b:
                pairs.append((a, b))
        out = {}
        for a, b in pairs:
            if a not in out:
                out[a] = b
        return out

    @staticmethod
    def _strip_codex_model_flags(args: list[str]) -> list[str]:
        cleaned: list[str] = []
        i = 0
        while i < len(args):
            cur = str(args[i] or "")
            if cur == "-c" and i + 1 < len(args):
                nxt = str(args[i + 1] or "")
                if nxt.startswith(("model.name=", "model=", "model_reasoning_effort=")):
                    i += 2
                    continue
            if cur in {"-m", "--model"} and i + 1 < len(args):
                i += 2
                continue
            if cur.startswith("--model="):
                i += 1
                continue
            cleaned.append(args[i])
            i += 1
        return cleaned

    @staticmethod
    def _strip_codex_effort_flags(args: list[str]) -> list[str]:
        cleaned: list[str] = []
        i = 0
        while i < len(args):
            cur = str(args[i] or "")
            if cur == "-c" and i + 1 < len(args):
                nxt = str(args[i + 1] or "")
                if nxt.startswith("model_reasoning_effort="):
                    i += 2
                    continue
            cleaned.append(args[i])
            i += 1
        return cleaned

    @staticmethod
    def _replace_codex_effort_flags(args: list[str], effort: str) -> list[str]:
        cleaned: list[str] = []
        replaced = False
        i = 0
        while i < len(args):
            cur = str(args[i] or "")
            if cur == "-c" and i + 1 < len(args):
                nxt = str(args[i + 1] or "")
                if nxt.startswith("model_reasoning_effort="):
                    if not replaced:
                        cleaned.extend(["-c", f'model_reasoning_effort="{effort}"'])
                        replaced = True
                    i += 2
                    continue
            cleaned.append(args[i])
            i += 1
        return cleaned

    @staticmethod
    def _strip_claude_model_flags(args: list[str]) -> list[str]:
        cleaned: list[str] = []
        i = 0
        while i < len(args):
            cur = str(args[i] or "")
            if cur in {"-m", "--model"} and i + 1 < len(args):
                i += 2
                continue
            if cur.startswith("--model="):
                i += 1
                continue
            cleaned.append(args[i])
            i += 1
        return cleaned

    @staticmethod
    def _strip_claude_effort_flags(args: list[str]) -> list[str]:
        cleaned: list[str] = []
        i = 0
        while i < len(args):
            cur = str(args[i] or "")
            if cur == "--effort" and i + 1 < len(args):
                i += 2
                continue
            if cur.startswith("--effort="):
                i += 1
                continue
            cleaned.append(args[i])
            i += 1
        return cleaned

    @staticmethod
    def _insert_before_double_dash(args: list[str], extra: list[str]) -> list[str]:
        cleaned = list(args)
        try:
            idx = cleaned.index("--")
        except ValueError:
            return cleaned + extra
        return cleaned[:idx] + extra + cleaned[idx:]

    @classmethod
    def _replace_claude_effort_flags(cls, args: list[str], effort: str) -> list[str]:
        cleaned: list[str] = []
        replaced = False
        i = 0
        while i < len(args):
            cur = str(args[i] or "")
            if cur == "--effort" and i + 1 < len(args):
                if not replaced:
                    cleaned.extend(["--effort", effort])
                    replaced = True
                i += 2
                continue
            if cur.startswith("--effort="):
                if not replaced:
                    cleaned.append(f"--effort={effort}")
                    replaced = True
                i += 1
                continue
            cleaned.append(args[i])
            i += 1
        if not replaced:
            return cls._insert_before_double_dash(cleaned, ["--effort", effort])
        return cleaned

    def _failover_startup_timeout(self) -> float:
        try:
            timeout_s = float(getattr(self._settings, "acp_startup_timeout", 20) or 20)
        except Exception:
            logger.debug(
                "ModelFailureAwareSession._do_failover: timeout_s conversion failed",
                exc_info=True,
            )
            timeout_s = 20.0
        return max(1.0, timeout_s)

    @staticmethod
    def _build_sync_acp_candidate(
        *,
        base: SyncSession,
        agent_type: str,
        cwd: str,
        agent_cmd: str,
        agent_args: list[str],
        model_name: str,
    ) -> SyncSession:
        sync_acp_session_cls = getattr(sys.modules[__name__], "SyncACPSession")
        explicit_env = getattr(base, "_explicit_env", None)
        candidate_kwargs: dict[str, Any] = {
            "agent_type": agent_type,
            "cwd": cwd,
            "agent_cmd": agent_cmd,
            "agent_args": list(agent_args),
            "model_name": model_name,
            "capture_full_tool_content": bool(
                getattr(base, "_capture_full_tool_content", False)
            ),
        }
        if isinstance(explicit_env, dict):
            candidate_kwargs["env"] = dict(explicit_env)
        try:
            return sync_acp_session_cls(**candidate_kwargs)
        except TypeError:
            fallback_kwargs = dict(candidate_kwargs)
            if "env" in fallback_kwargs:
                fallback_kwargs.pop("env", None)
                try:
                    return sync_acp_session_cls(**fallback_kwargs)
                except TypeError:
                    fallback_kwargs.pop("model_name", None)
                    return sync_acp_session_cls(**fallback_kwargs)
            fallback_kwargs.pop("model_name", None)
            return sync_acp_session_cls(**fallback_kwargs)

    def _do_failover(self, *, from_model: str, to_model: str) -> bool:
        """执行一次 failover：切换到 to_model，并重建 session 后替换 self._inner。"""
        from_model = str(from_model or "").strip()
        to_model = str(to_model or "").strip()
        if not to_model:
            return False

        base, rewrap = self._unwrap_rate_limit()
        agent_type = str(getattr(base, "_agent_type", "") or "").strip()
        agent_lower = agent_type.lower()
        is_cli_session = getattr(base, "_cfg", None) is not None and callable(
            getattr(base, "_build_args", None)
        )
        is_agy_cli = bool(getattr(base, "_is_agy_cli_session", False))
        is_agy = agent_lower == "agy" or is_agy_cli

        if is_agy:
            agy_cwd = str(getattr(base, "_cwd", "") or "")
            if not agy_cwd:
                return False
            timeout_s = self._failover_startup_timeout()
            from ..acp.agy_selection import requires_agy_cli_orchestrator_session

            if requires_agy_cli_orchestrator_session(to_model):
                from .agy_cli import AGYCLIConfig

                agy_cli_cls = getattr(sys.modules[__name__], "SyncAGYCLISession")
                existing_cfg = getattr(base, "_cfg", None)
                if isinstance(existing_cfg, AGYCLIConfig) or (
                    existing_cfg is not None and is_agy_cli
                ):
                    cli_cfg = existing_cfg
                else:
                    base_cmd = str(getattr(base, "_agent_cmd", "") or "").strip()
                    cli_cfg = (
                        AGYCLIConfig(command=base_cmd) if base_cmd else AGYCLIConfig()
                    )

                def create_agy_cli_candidate() -> SyncSession:
                    return agy_cli_cls(
                        cwd=agy_cwd,
                        config=cli_cfg,
                        model_name=to_model,
                    )

                return self._replace_inner_strictly(
                    base=base,
                    rewrap=rewrap,
                    candidate_factory=create_agy_cli_candidate,
                    prepare_candidate=lambda candidate: candidate.start(
                        startup_timeout=timeout_s
                    ),
                    operation="failover",
                )

            if is_cli_session or is_agy_cli:
                try:
                    resolver = getattr(sys.modules[__name__], "resolve_agent_spec")
                    resolved_cmd, resolved_args = resolver("agy", model_name=to_model)
                    agy_acp_cmd = str(resolved_cmd or "").strip()
                    agy_acp_args = list(resolved_args or [])
                except Exception:
                    logger.debug(
                        "ModelFailureAwareSession._do_failover: agy ACP provider resolution failed",
                        exc_info=True,
                    )
                    return False
                if not agy_acp_cmd and not agy_acp_args:
                    return False
                effective_agent_type = agent_type or "agy"

                def create_agy_acp_candidate() -> SyncSession:
                    return self._build_sync_acp_candidate(
                        base=base,
                        agent_type=effective_agent_type,
                        cwd=agy_cwd,
                        agent_cmd=agy_acp_cmd,
                        agent_args=agy_acp_args,
                        model_name=to_model,
                    )

                return self._replace_inner_strictly(
                    base=base,
                    rewrap=rewrap,
                    candidate_factory=create_agy_acp_candidate,
                    prepare_candidate=lambda candidate: candidate.start(
                        startup_timeout=timeout_s
                    ),
                    operation="failover",
                )

        if is_cli_session:
            cli_cwd = str(getattr(base, "_cwd", "") or "")
            if not cli_cwd:
                return False
            timeout_s = self._failover_startup_timeout()
            cli_cls = type(base)
            cli_cfg = getattr(base, "_cfg")

            def create_cli_candidate() -> SyncSession:
                return cli_cls(cwd=cli_cwd, config=cli_cfg, model_name=to_model)

            return self._replace_inner_strictly(
                base=base,
                rewrap=rewrap,
                candidate_factory=create_cli_candidate,
                prepare_candidate=lambda candidate: candidate.start(
                    startup_timeout=timeout_s
                ),
                operation="failover",
            )

        agent_cmd = str(getattr(base, "_agent_cmd", "") or "")
        agent_args = list(getattr(base, "_agent_args", []) or [])
        cwd = str(getattr(base, "_cwd", "") or "")
        if not agent_cmd and not agent_args:
            return False
        if not agent_type or not cwd:
            return False

        argv_model = to_model
        codex_effort: Optional[str] = None
        claude_effort: Optional[str] = None
        if agent_lower in {"codex", "codex_w"}:
            try:
                from ..acp.model_selection import (
                    CODEX_DEFAULT_EFFORT_TOKEN,
                    CODEX_DEFAULT_MODEL_TOKEN,
                    split_codex_model_selection,
                )

                base_model, codex_effort = split_codex_model_selection(to_model)
                if codex_effort == CODEX_DEFAULT_EFFORT_TOKEN:
                    codex_effort = None
                argv_model = (
                    ""
                    if (not base_model or base_model == CODEX_DEFAULT_MODEL_TOKEN)
                    else base_model
                )
            except Exception:
                argv_model = ""
        elif agent_lower in {"claude", "claude_w"}:
            try:
                from ..acp.claude_capabilities import strip_1m_suffix
                from ..acp.claude_selection import (
                    CLAUDE_DEFAULT_EFFORT_TOKEN,
                    CLAUDE_DEFAULT_MODEL_TOKEN,
                    split_claude_model_selection,
                )

                base_model, claude_effort = split_claude_model_selection(to_model)
                if claude_effort == CLAUDE_DEFAULT_EFFORT_TOKEN:
                    claude_effort = None
                argv_model = (
                    ""
                    if (not base_model or base_model == CLAUDE_DEFAULT_MODEL_TOKEN)
                    else strip_1m_suffix(base_model)
                )
            except Exception:
                argv_model = ""
        else:
            try:
                from ..acp.providers import normalize_acp_model_name

                argv_model = normalize_acp_model_name(agent_type, to_model) or to_model
            except Exception:
                argv_model = to_model

        if agent_lower in {"codex", "codex_w"} and not argv_model:
            had_effort_flag = any(
                str(agent_args[i] or "") == "-c"
                and i + 1 < len(agent_args)
                and str(agent_args[i + 1] or "").startswith("model_reasoning_effort=")
                for i in range(len(agent_args))
            )
            new_args = self._strip_codex_model_flags(agent_args)
            if codex_effort and had_effort_flag:
                new_args.extend(["-c", f'model_reasoning_effort="{codex_effort}"'])
            replaced = False
        elif agent_lower in {"claude", "claude_w"}:
            if not argv_model:
                new_args = self._strip_claude_model_flags(agent_args)
                replaced = True
            else:
                new_args, replaced = _replace_model_in_agent_args(agent_args, argv_model)
                if not replaced:
                    new_args = self._insert_before_double_dash(
                        new_args, ["--model", argv_model]
                    )
                    replaced = True
            if claude_effort:
                new_args = self._replace_claude_effort_flags(new_args, claude_effort)
            else:
                new_args = self._strip_claude_effort_flags(new_args)
        else:
            new_args, replaced = _replace_model_in_agent_args(agent_args, argv_model)
            if agent_lower in {"codex", "codex_w"}:
                if codex_effort:
                    new_args = self._replace_codex_effort_flags(new_args, codex_effort)
                else:
                    new_args = self._strip_codex_effort_flags(new_args)
        is_config_option_backend = agent_lower in {
            "codex",
            "codex_w",
            "traex",
            "dsh",
            "agy",
        } or bool(str(getattr(base, "_model_name", "") or "").strip())
        if not replaced and not is_config_option_backend:
            return False

        timeout_s = self._failover_startup_timeout()

        def create_candidate() -> SyncSession:
            return self._build_sync_acp_candidate(
                base=base,
                agent_type=agent_type,
                cwd=cwd,
                agent_cmd=agent_cmd,
                agent_args=list(new_args),
                model_name=to_model,
            )

        return self._replace_inner_strictly(
            base=base,
            rewrap=rewrap,
            candidate_factory=create_candidate,
            prepare_candidate=lambda candidate: candidate.start(
                startup_timeout=timeout_s
            ),
            operation="failover",
        )


    def send_prompt(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
        idle_timeout: Optional[float] = None,
        activity_predicate: Optional[Callable[[ACPEvent], bool]] = None,
    ) -> PromptResult:
        compaction_tried = False
        failover_tried = False
        prompt_kwargs = _prompt_kwargs(
            on_event=on_event,
            timeout=timeout,
            idle_timeout=idle_timeout,
            activity_predicate=activity_predicate,
        )

        def _lookup_failover(fmap: dict[str, str], key: str) -> str:
            k = str(key or "").strip()
            if not k:
                return ""
            direct = fmap.get(k)
            if direct:
                return direct
            if k.startswith("["):
                try:
                    compact_k = json.dumps(
                        json.loads(k), ensure_ascii=False, separators=(",", ":")
                    )
                except (TypeError, ValueError):
                    compact_k = ""
                if compact_k:
                    for map_k, map_v in fmap.items():
                        if map_k.startswith("["):
                            try:
                                if (
                                    json.dumps(
                                        json.loads(map_k),
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    )
                                    == compact_k
                                ):
                                    return map_v
                            except (TypeError, ValueError):
                                continue
            return ""

        def _attempt_failover(failed_hint: object, reason: str, attempt_count: int) -> bool:
            base, _ = self._unwrap_rate_limit()
            agent_args = list(getattr(base, "_agent_args", []) or [])
            base_model = str(getattr(base, "_model_name", "") or "").strip()
            failed = failed_hint or _extract_model_from_agent_args(agent_args) or base_model
            fmap = self._parse_failover_map()
            uses_config_c = any(
                arg == "-c" or str(arg).startswith("model.name=") for arg in agent_args
            )
            target = (
                _lookup_failover(fmap, str(failed or ""))
                or (_lookup_failover(fmap, base_model) if base_model else "")
                or (fmap.get("gpt-5.2", "") if uses_config_c else "")
            )
            ok = self._do_failover(from_model=str(failed or ""), to_model=str(target or ""))
            logger.warning(
                "[ModelFailure] action=failover reason=%s fail_phase=model_loop failover=%s from_model=%s to_model=%s attempt_count=%d",
                reason,
                bool(ok),
                failed or "",
                target or "",
                int(attempt_count or 0),
            )
            return ok

        while True:
            if self._cancel_event.is_set():
                raise RuntimeError("ACP prompt cancelled before send")
            try:
                return self._inner.send_prompt(text, **prompt_kwargs)
            except Exception as e:
                info = classify_model_failure(error=e)

                # 1) loop detected: attempt failover once
                if info.get("reason") == "loop_detected" and not failover_tried:
                    failover_tried = True
                    if _attempt_failover(
                        info.get("failed_model"),
                        "loop_detected",
                        int(info.get("attempt_count") or 0),
                    ):
                        continue
                    raise

                # 2) need compaction: compaction first, failover if loop or repeated
                if info.get("reason") == "need_compaction":
                    is_loop, n = self._record_compaction_event_and_check_loop()
                    try:
                        info["attempt_count"] = int(n)
                    except Exception:
                        logger.debug("ModelFailureAwareSession.send_prompt: attempt_count assignment failed", exc_info=True)

                    if not bool(getattr(self._settings, "model_failure_compaction_enabled", True)):
                        raise

                    if is_loop:
                        try:
                            window_s, max_count = self._loop_limits()
                        except Exception:
                            logger.debug("ModelFailureAwareSession.send_prompt: _loop_limits failed", exc_info=True)
                            window_s, max_count = (0.0, 1)
                        logger.warning(
                            "[ModelFailure] action=suppress reason=need_compaction fail_phase=model_loop attempt_count=%d loop_window_s=%.1f loop_max=%d",
                            int(n),
                            float(window_s or 0.0),
                            int(max_count or 1),
                        )
                        if not failover_tried:
                            failover_tried = True
                            if _attempt_failover(info.get("failed_model"), "need_compaction", n):
                                continue
                        raise

                    if compaction_tried and not failover_tried:
                        failover_tried = True
                        if _attempt_failover(info.get("failed_model"), "need_compaction", n):
                            continue

                    # First time: do compaction once
                    if not compaction_tried:
                        compaction_tried = True
                        ok = self._do_compaction()
                        logger.warning(
                            "[ModelFailure] action=compaction reason=need_compaction fail_phase=model_compaction compaction=%s model=%s failover_to=%s attempt_count=%d",
                            bool(ok),
                            info.get("failed_model") or "",
                            info.get("failover_to") or "",
                            int(info.get("attempt_count") or 0),
                        )
                        if self._cancel_event.is_set():
                            raise RuntimeError(
                                "ACP prompt cancelled during model replacement"
                            ) from e
                        if ok:
                            continue
                raise
