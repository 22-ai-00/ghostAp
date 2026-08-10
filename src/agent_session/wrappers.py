"""Rate-limit and model-failure aware session wrappers."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from ..acp.models import ACPEvent, PromptResult
from ..acp.sync_adapter import SyncACPSession
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
        """Close old, prepare candidate, install filter, then atomically swap."""
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
                tool_filter = self.get_tool_filter()
                if tool_filter is not None:
                    replacement.set_tool_filter(tool_filter)
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
        """解析 failover 映射（from:to）。"""
        raw = ""
        try:
            raw = str(getattr(self._settings, "model_failure_failover_map", "") or "")
        except Exception:
            logger.debug("ModelFailureAwareSession._parse_failover_map: settings read failed", exc_info=True)
            raw = ""
        pairs = []
        for chunk in raw.replace(",", " ").split():
            s = (chunk or "").strip()
            if not s or ":" not in s:
                continue
            a, b = s.split(":", 1)
            a, b = a.strip(), b.strip()
            if a and b:
                pairs.append((a, b))
        out: dict[str, str] = {}
        for a, b in pairs:
            if a not in out:
                out[a] = b
        return out

    def _do_failover(self, *, from_model: str, to_model: str) -> bool:
        """执行一次 failover：切换到 to_model，并重建 session 后替换 self._inner。"""
        from_model = str(from_model or "").strip()
        to_model = str(to_model or "").strip()
        if not to_model:
            return False

        base, rewrap = self._unwrap_rate_limit()
        agent_cmd = str(getattr(base, "_agent_cmd", "") or "")
        agent_args = list(getattr(base, "_agent_args", []) or [])
        agent_type = str(getattr(base, "_agent_type", "") or "")
        cwd = str(getattr(base, "_cwd", "") or "")
        if not agent_cmd and not agent_args:
            return False
        if not agent_type or not cwd:
            return False

        new_args, replaced = _replace_model_in_agent_args(agent_args, to_model)
        if not replaced:
            return False

        try:
            timeout_s = float(getattr(self._settings, "acp_startup_timeout", 20) or 20)
        except Exception:
            logger.debug("ModelFailureAwareSession._do_failover_switch: timeout_s conversion failed", exc_info=True)
            timeout_s = 20.0
        timeout_s = max(1.0, timeout_s)

        def create_candidate() -> SyncSession:
            return SyncACPSession(
                agent_type=agent_type,
                cwd=cwd,
                agent_cmd=agent_cmd,
                agent_args=list(new_args),
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
                    failed = info.get("failed_model") or _extract_model_from_agent_args(
                        list(getattr(getattr(self._inner, "_inner", self._inner), "_agent_args", []) or [])
                    )
                    fmap = self._parse_failover_map()
                    target = fmap.get(str(failed or "").strip()) or fmap.get("gpt-5.2")
                    ok = self._do_failover(from_model=str(failed or ""), to_model=str(target or ""))
                    logger.warning(
                        "[ModelFailure] action=failover reason=loop_detected fail_phase=model_loop failover=%s from_model=%s to_model=%s attempt_count=%d",
                        bool(ok),
                        failed or "",
                        target or "",
                        int(info.get("attempt_count") or 0),
                    )
                    if ok:
                        continue
                    raise

                # 2) need compaction:
                #    - 记录事件用于 loop 检测
                #    - 首次命中：先 compaction，再同模型重试一次
                #    - 若 compaction 后仍命中 need_compaction（或达到 loop 阈值）：触发 failover 一次
                if info.get("reason") == "need_compaction":
                    # loop detection (record every time)
                    is_loop, n = self._record_compaction_event_and_check_loop()
                    try:
                        info["attempt_count"] = int(n)
                    except Exception:
                        logger.debug("ModelFailureAwareSession.send_prompt: attempt_count assignment failed", exc_info=True)

                    # feature flag: disabled => no auto-repair
                    if not bool(getattr(self._settings, "model_failure_compaction_enabled", True)):
                        raise

                    # If loop detected: suppress compaction and attempt failover once.
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
                            failed = info.get("failed_model") or _extract_model_from_agent_args(
                                list(getattr(getattr(self._inner, "_inner", self._inner), "_agent_args", []) or [])
                            )
                            fmap = self._parse_failover_map()
                            target = fmap.get(str(failed or "").strip()) or fmap.get("gpt-5.2")
                            ok = self._do_failover(from_model=str(failed or ""), to_model=str(target or ""))
                            logger.warning(
                                "[ModelFailure] action=failover reason=need_compaction fail_phase=model_loop failover=%s from_model=%s to_model=%s attempt_count=%d",
                                bool(ok),
                                failed or "",
                                target or "",
                                int(n),
                            )
                            if ok:
                                continue
                        raise

                    # Not loop: if compaction already tried once, attempt failover once.
                    if compaction_tried and (not failover_tried):
                        failover_tried = True
                        failed = info.get("failed_model") or _extract_model_from_agent_args(
                            list(getattr(getattr(self._inner, "_inner", self._inner), "_agent_args", []) or [])
                        )
                        fmap = self._parse_failover_map()
                        target = fmap.get(str(failed or "").strip()) or fmap.get("gpt-5.2")
                        ok = self._do_failover(from_model=str(failed or ""), to_model=str(target or ""))
                        logger.warning(
                            "[ModelFailure] action=failover reason=need_compaction fail_phase=model_loop failover=%s from_model=%s to_model=%s attempt_count=%d",
                            bool(ok),
                            failed or "",
                            target or "",
                            int(n),
                        )
                        if ok:
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
