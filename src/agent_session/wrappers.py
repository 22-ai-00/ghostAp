"""Rate-limit and model-failure aware session wrappers."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from ..acp.models import ACPEvent, PromptResult
from ..acp.sync_adapter import SyncACPSession
from ..config import get_settings
from ..utils.errors import get_error_detail
from ..utils.retry import RetryPolicy, prompt_with_retry
from .model_diagnostics import (
    _default_compaction_action,
    _detect_rate_limit,
    _extract_model_from_agent_args,
    _remove_model_in_agent_args,
    _replace_model_in_agent_args,
    classify_model_failure,
)
from .protocol import SyncSession

logger = logging.getLogger(__name__)


class RateLimitAwareSession:
    """Wraps a SyncSession with rate-limit-aware retry on send_prompt().

    Implements the full SyncSession protocol by explicit delegation (no __getattr__).
    """

    def __init__(
        self,
        inner: SyncSession,
        on_rate_limit: Optional[Callable[[int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self._inner = inner
        self._on_rate_limit = on_rate_limit
        self._cancel_event = cancel_event or threading.Event()
        self._settings = get_settings()
        # Rate-limit state visible to status queries
        self.rate_limit_until: Optional[float] = None  # monotonic deadline

    def __getattr__(self, name: str):
        """Best-effort proxy for non-protocol attributes.

        说明：本类以"显式 delegation"实现 SyncSession 协议，避免隐藏行为。
        但仓内部分测试/诊断会读取 `_model_name` / `_agent_args` 等私有字段。
        为保持兼容，这里将未知属性透传给 inner。
        """
        return getattr(self._inner, name)

    # --- Explicit SyncSession protocol delegation ---

    @property
    def session_id(self) -> str:
        return self._inner.session_id

    @session_id.setter
    def session_id(self, value: str):
        self._inner.session_id = value

    @property
    def created_at(self) -> float:
        return self._inner.created_at

    @property
    def last_active(self) -> float:
        return self._inner.last_active

    @last_active.setter
    def last_active(self, value: float):
        self._inner.last_active = value

    @property
    def message_count(self) -> int:
        return self._inner.message_count

    @property
    def last_query(self) -> str:
        return self._inner.last_query

    @property
    def is_resumed(self) -> bool:
        return self._inner.is_resumed

    def describe_agent(self) -> str:
        return self._inner.describe_agent()

    def start(self, startup_timeout: float = 60) -> str:
        return self._inner.start(startup_timeout=startup_timeout)

    def load_session(self, session_id: str, timeout: float) -> None:
        self._inner.load_session(session_id, timeout=timeout)

    def load_local_history(self, session_id: Optional[str] = None, limit: int = 200) -> list[dict]:
        return self._inner.load_local_history(session_id=session_id, limit=limit)

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> bool | None:
        self._cancel_event.set()
        try:
            return self._inner.cancel(wait=wait, timeout=timeout)
        except TypeError:
            return self._inner.cancel()

    def close(self) -> None:
        self._inner.close()

    def to_snapshot(self) -> dict:
        return self._inner.to_snapshot()

    def get_session_info(self) -> str:
        return self._inner.get_session_info()

    def is_server_running(self) -> bool:
        return self._inner.is_server_running()

    def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
        return self._inner.is_server_healthy(healthcheck_timeout=healthcheck_timeout)

    # --- Rate-limit-aware send_prompt ---

    def send_prompt(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
    ) -> PromptResult:
        if not self._settings.rate_limit_retry_enabled:
            return self._inner.send_prompt(text, on_event=on_event, timeout=timeout)

        max_retries = self._settings.rate_limit_max_retries
        max_wait = self._settings.rate_limit_max_wait
        base_wait = self._settings.rate_limit_base_wait
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                self.rate_limit_until = None
                return self._inner.send_prompt(text, on_event=on_event, timeout=timeout)
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
        return self._inner.send_prompt(text, on_event=on_event, timeout=timeout)

    def send_prompt_with_retry(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        before_retry: Optional[Callable[[int, Exception], None]] = None,
        total_timeout: Optional[float] = None,
    ) -> PromptResult:
        return prompt_with_retry(
            lambda: self.send_prompt(text, on_event=on_event, timeout=timeout),
            self._cancel_event,
            retry_policy=retry_policy,
            before_retry=before_retry,
            total_timeout=total_timeout,
        )


class ModelFailureAwareSession:
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
        self._inner = inner
        self._settings = get_settings()
        self._compaction_action = compaction_action
        self._on_rate_limit = on_rate_limit
        self._cancel_event = cancel_event or threading.Event()

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

    def __getattr__(self, name: str):
        """Proxy unknown attributes to inner session.

        Needed so that callers (e.g. SlockEngine._apply_tool_restrictions) can
        access set_tool_filter / get_tool_filter / describe_agent etc. through
        the wrapper chain without explicit delegation for every method.
        """
        return getattr(self._inner, name)

    # --- Explicit SyncSession protocol delegation ---

    @property
    def session_id(self) -> str:
        return self._inner.session_id

    @session_id.setter
    def session_id(self, value: str):
        self._inner.session_id = value

    @property
    def created_at(self) -> float:
        return self._inner.created_at

    @property
    def last_active(self) -> float:
        return self._inner.last_active

    @last_active.setter
    def last_active(self, value: float):
        self._inner.last_active = value

    @property
    def message_count(self) -> int:
        return self._inner.message_count

    @property
    def last_query(self) -> str:
        return self._inner.last_query

    @property
    def is_resumed(self) -> bool:
        return self._inner.is_resumed

    def describe_agent(self) -> str:
        return self._inner.describe_agent()

    def start(self, startup_timeout: float = 60) -> str:
        return self._inner.start(startup_timeout=startup_timeout)

    def load_session(self, session_id: str, timeout: float) -> None:
        self._inner.load_session(session_id, timeout=timeout)

    def load_local_history(self, session_id: Optional[str] = None, limit: int = 200) -> list[dict]:
        return self._inner.load_local_history(session_id=session_id, limit=limit)

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> bool | None:
        self._cancel_event.set()
        try:
            return self._inner.cancel(wait=wait, timeout=timeout)
        except TypeError:
            return self._inner.cancel()

    def close(self) -> None:
        self._inner.close()

    def to_snapshot(self) -> dict:
        return self._inner.to_snapshot()

    def get_session_info(self) -> str:
        return self._inner.get_session_info()

    def is_server_running(self) -> bool:
        return self._inner.is_server_running()

    def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
        return self._inner.is_server_healthy(healthcheck_timeout=healthcheck_timeout)

    # --- Internal helpers ---

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
        try:
            new_base = action(base)
        except (RuntimeError, OSError, TimeoutError):
            logger.debug("ModelFailureAwareSession._apply_failover: action failed", exc_info=True)
            new_base = None

        if new_base is None:
            return False
        self._inner = rewrap(new_base)
        return True

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

        # close old
        try:
            base.close()
        except Exception:
            logger.debug("ModelFailureAwareSession._do_failover_switch: close old session failed", exc_info=True)

        # rebuild and start
        try:
            timeout_s = float(getattr(self._settings, "acp_startup_timeout", 20) or 20)
        except Exception:
            logger.debug("ModelFailureAwareSession._do_failover_switch: timeout_s conversion failed", exc_info=True)
            timeout_s = 20.0
        timeout_s = max(1.0, timeout_s)

        try:
            new_base = SyncACPSession(agent_type=agent_type, cwd=cwd, agent_cmd=agent_cmd, agent_args=list(new_args))
            new_base.start(startup_timeout=timeout_s)
        except (RuntimeError, OSError, TimeoutError):
            logger.debug("ModelFailureAwareSession._do_failover_switch: rebuild session failed", exc_info=True)
            return False

        self._inner = rewrap(new_base)
        return True

    def _is_ttadk_agent_type(self, agent_type: str) -> bool:
        try:
            return str(agent_type or "").strip().lower().startswith("ttadk_")
        except Exception:
            logger.debug("ModelFailureAwareSession._is_ttadk_agent_type: failed", exc_info=True)
            return False

    def _extract_ttadk_tool_name(self, *, agent_type: str, agent_args: list[str]) -> str:
        """best-effort 提取 ttadk tool 名称（例如 codex/claude/coco）。"""
        try:
            at = str(agent_type or "").strip().lower()
        except Exception:
            logger.debug("ModelFailureAwareSession._extract_ttadk_tool_name: agent_type conversion failed", exc_info=True)
            at = ""
        if at.startswith("ttadk_"):
            return at.replace("ttadk_", "", 1)

        # fallback: try parse "-t <tool>" from args
        try:
            xs = [str(x) for x in (agent_args or [])]
        except Exception:
            logger.debug("ModelFailureAwareSession._extract_ttadk_tool_name: args conversion failed", exc_info=True)
            xs = list(agent_args or [])
        for i, x in enumerate(xs):
            if x in ("-t", "--tool") and i + 1 < len(xs):
                v = str(xs[i + 1] or "").strip().lower()
                if v:
                    return v
        return ""

    def _runtime_invalid_model_settings(self) -> tuple[bool, bool, int, float]:
        """读取运行期 invalid-model 自愈配置（enabled, allow_autoswitch, max_retries, cooldown_s）。"""
        s = self._settings
        try:
            enabled = bool(getattr(s, "ttadk_runtime_retry_enabled", True))
        except Exception:
            logger.debug("_runtime_invalid_model_settings: enabled read failed", exc_info=True)
            enabled = True
        try:
            allow_autoswitch = bool(getattr(s, "ttadk_runtime_retry_allow_autoswitch", True))
        except Exception:
            logger.debug("_runtime_invalid_model_settings: allow_autoswitch read failed", exc_info=True)
            allow_autoswitch = True
        try:
            max_retries = int(getattr(s, "ttadk_runtime_max_retries", 1) or 1)
        except Exception:
            logger.debug("_runtime_invalid_model_settings: max_retries read failed", exc_info=True)
            max_retries = 1
        try:
            cooldown_s = max(0.0, float(getattr(s, "ttadk_runtime_retry_cooldown_s", 120) or 120))
        except Exception:
            logger.debug("_runtime_invalid_model_settings: cooldown_s read failed", exc_info=True)
            cooldown_s = 120.0
        max_retries = max(0, max_retries)
        cooldown_s = max(0.0, cooldown_s)
        return enabled, allow_autoswitch, max_retries, cooldown_s

    def _pick_best_ttadk_retry_model(
        self, *, tool_name: str, input_model: str, available_models: list[str], allow_autoswitch: bool
    ) -> str | None:
        """从 available_models 中选择候选真实 model（best-effort）。"""
        try:
            cands = [str(x).strip() for x in (available_models or []) if str(x).strip()]
        except Exception:
            logger.debug("_pick_best_ttadk_retry_model: candidates conversion failed", exc_info=True)
            cands = []
        if not cands:
            return None

        tool = str(tool_name or "").strip().lower()
        if tool:
            try:
                tool_cands = [m for m in cands if tool in m.lower()]
            except Exception:
                tool_cands = []
        else:
            tool_cands = []

        pool = tool_cands or cands
        if not allow_autoswitch:
            return pool[0]
        try:
            from ..ttadk.models import choose_best_available_model

            best = choose_best_available_model(input_model=str(input_model or ""), available_models=list(pool))
            best = str(best).strip() if best is not None else ""
            return best or pool[0]
        except Exception:
            logger.debug("_pick_best_ttadk_retry_model: choose_best_available_model failed", exc_info=True)
            return pool[0]

    def _do_ttadk_auto(self) -> bool:
        """将当前 TTADK 会话切换到 auto（移除 -m）并重建 session。"""
        base, rewrap = self._unwrap_rate_limit()
        agent_cmd = str(getattr(base, "_agent_cmd", "") or "")
        agent_args = list(getattr(base, "_agent_args", []) or [])
        agent_type = str(getattr(base, "_agent_type", "") or "")
        cwd = str(getattr(base, "_cwd", "") or "")
        if not agent_cmd and not agent_args:
            return False
        if not agent_type or not cwd:
            return False
        if not self._is_ttadk_agent_type(agent_type):
            return False

        new_args, removed = _remove_model_in_agent_args(agent_args)
        if not removed:
            # already auto
            return False

        try:
            base.close()
        except Exception:
            logger.debug("_do_ttadk_auto: close old session failed", exc_info=True)

        try:
            timeout_s = float(getattr(self._settings, "acp_startup_timeout", 20) or 20)
        except Exception:
            logger.debug("_do_ttadk_auto: timeout_s conversion failed", exc_info=True)
            timeout_s = 20.0
        timeout_s = max(1.0, timeout_s)

        try:
            new_base = SyncACPSession(agent_type=agent_type, cwd=cwd, agent_cmd=agent_cmd, agent_args=list(new_args))
            new_base.start(startup_timeout=timeout_s)
        except (RuntimeError, OSError, TimeoutError):
            logger.debug("_do_ttadk_auto: rebuild session failed", exc_info=True)
            return False

        self._inner = rewrap(new_base)
        return True

    # --- Model-failure-aware send_prompt ---

    def send_prompt(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
    ) -> PromptResult:
        compaction_tried = False
        failover_tried = False
        invalid_real_tried = False
        invalid_auto_tried = False
        degraded_tried = False
        runtime_attempts: list[dict] = []

        def _set_last_attempts():
            # 仅供诊断/单测：best-effort
            try:
                self._last_runtime_invalid_model_attempts = list(runtime_attempts)
            except Exception:
                logger.debug("ModelFailureAwareSession.send_prompt: _set_last_attempts failed", exc_info=True)

        def _record_attempt(
            step: str,
            *,
            ok: bool,
            tool: str = "",
            input_model: str = "",
            passthrough_model: str | None = None,
            error: Exception | None = None,
            extra: Optional[dict] = None,
        ):
            d: dict = {
                "phase": "runtime_invalid_model",
                "step": str(step or ""),
                "ok": bool(ok),
                "tool": str(tool or ""),
                "input_model": str(input_model or ""),
                "passthrough_model": passthrough_model,
            }
            if error is not None:
                d["error_type"] = type(error).__name__
                d["error"] = get_error_detail(error)
            if extra:
                try:
                    d.update(dict(extra))
                except Exception:
                    logger.debug("ModelFailureAwareSession.send_prompt: _record_attempt extra update failed", exc_info=True)
            runtime_attempts.append(d)
            _set_last_attempts()

        while True:
            try:
                return self._inner.send_prompt(text, on_event=on_event, timeout=timeout)
            except Exception as e:
                # ------------------------------------------------------------
                # TTADK runtime invalid-model self-healing (SSOT entry)
                # ------------------------------------------------------------
                base = getattr(self._inner, "_inner", self._inner)
                agent_type = str(getattr(base, "_agent_type", "") or "")
                if self._is_ttadk_agent_type(agent_type):
                    enabled, allow_autoswitch, max_retries, cooldown_s = self._runtime_invalid_model_settings()
                    if enabled and max_retries > 0:
                        try:
                            from ..ttadk import get_ttadk_manager
                            from ..ttadk.models import build_invalid_model_context

                            ctx = build_invalid_model_context(e, get_settings_fn=get_settings, limit=1600)
                            is_invalid = bool(ctx.get("is_invalid_model"))
                            available_models = list(ctx.get("available_models") or [])
                        except (ImportError, RuntimeError, ValueError):
                            logger.debug("ModelFailureAwareSession.send_prompt: build_invalid_model_context failed", exc_info=True)
                            ctx = {}
                            is_invalid = False
                            available_models = []

                        if is_invalid:
                            try:
                                args0 = list(getattr(base, "_agent_args", []) or [])
                            except (TypeError, AttributeError):
                                logger.debug("ModelFailureAwareSession.send_prompt: args0 extraction failed", exc_info=True)
                                args0 = []
                            tool = self._extract_ttadk_tool_name(agent_type=agent_type, agent_args=args0)
                            input_model = _extract_model_from_agent_args(args0)

                            _record_attempt(
                                "detect",
                                ok=True,
                                tool=tool,
                                input_model=input_model,
                                extra={
                                    "available_models_count": len(available_models),
                                    "retry_count": int(invalid_real_tried) + int(invalid_auto_tried),
                                },
                            )

                            # cooldown gate
                            try:
                                from ..ttadk import get_ttadk_manager
                                mgr = get_ttadk_manager()
                                allowed, last_ts = mgr.check_and_mark_runtime_invalid_model_repair(
                                    tool_name=tool,
                                    cooldown_s=float(cooldown_s or 0.0),
                                    now_ts=time.time(),
                                )
                            except (ImportError, RuntimeError, AttributeError):
                                logger.debug("ModelFailureAwareSession.send_prompt: cooldown gate check failed", exc_info=True)
                                allowed, last_ts = True, 0.0

                            if not allowed:
                                _record_attempt(
                                    "cooldown_skip",
                                    ok=False,
                                    tool=tool,
                                    input_model=input_model,
                                    extra={"cooldown_s": float(cooldown_s or 0.0), "last_ts": float(last_ts or 0.0)},
                                )
                                raise

                            # 1) retry with real model once
                            if (not invalid_real_tried) and max_retries >= 1:
                                candidate = self._pick_best_ttadk_retry_model(
                                    tool_name=tool,
                                    input_model=input_model,
                                    available_models=available_models,
                                    allow_autoswitch=allow_autoswitch,
                                )
                                candidate = str(candidate or "").strip() or None
                                if candidate and candidate != str(input_model or "").strip():
                                    invalid_real_tried = True
                                    ok = self._do_failover(
                                        from_model=str(input_model or ""), to_model=str(candidate or "")
                                    )
                                    _record_attempt(
                                        "retry_real_model",
                                        ok=bool(ok),
                                        tool=tool,
                                        input_model=input_model,
                                        passthrough_model=candidate,
                                        extra={"retry_count": int(invalid_real_tried) + int(invalid_auto_tried)},
                                    )
                                    logger.warning(
                                        "[TTADK:RuntimeInvalidModel] action=retry_real_model ok=%s tool=%s input_model=%s to_model=%s available_models=%d",
                                        bool(ok),
                                        tool,
                                        input_model,
                                        candidate,
                                        len(available_models),
                                    )
                                    if ok:
                                        continue

                            # 2) retry with auto once
                            if (not invalid_auto_tried) and max_retries >= 1:
                                invalid_auto_tried = True
                                ok = self._do_ttadk_auto()
                                _record_attempt(
                                    "retry_auto",
                                    ok=bool(ok),
                                    tool=tool,
                                    input_model=input_model,
                                    passthrough_model=None,
                                    extra={"retry_count": int(invalid_real_tried) + int(invalid_auto_tried)},
                                )
                                logger.warning(
                                    "[TTADK:RuntimeInvalidModel] action=retry_auto ok=%s tool=%s input_model=%s",
                                    bool(ok),
                                    tool,
                                    input_model,
                                )
                                if ok:
                                    continue

                            # 3) Produce explicit degraded diagnostics only.
                            # TTADK runtime self-healing must not replace the
                            # current TTADK session with a coco ACP session.
                            if not degraded_tried:
                                degraded_tried = True
                                _record_attempt(
                                    "degraded_result",
                                    ok=False,
                                    tool=tool,
                                    input_model=input_model,
                                    extra={
                                        "retry_count": int(invalid_real_tried) + int(invalid_auto_tried),
                                        "degraded": True,
                                        "session_replaced": False,
                                    },
                                )
                                logger.warning(
                                    "[TTADK:RuntimeInvalidModel] action=degraded_result session_replaced=false tool=%s input_model=%s",
                                    tool,
                                    input_model,
                                )

                            # give up: raise a diagnosable error (keep original as cause)
                            try:
                                err = RuntimeError("ttadk_runtime_invalid_model_unrecoverable")
                                try:
                                    err.tool_name = tool
                                    err.input_model = input_model
                                    err.available_models_count = len(available_models)
                                    err.attempts = list(runtime_attempts)
                                except Exception:
                                    logger.debug("ModelFailureAwareSession.send_prompt: error attribute setting failed", exc_info=True)
                                raise err from e
                            except Exception:
                                raise

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
                        if ok:
                            continue
                raise

    def send_prompt_with_retry(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        before_retry: Optional[Callable[[int, Exception], None]] = None,
        total_timeout: Optional[float] = None,
    ) -> PromptResult:
        return prompt_with_retry(
            lambda: self.send_prompt(text, on_event=on_event, timeout=timeout),
            self._cancel_event,
            retry_policy=retry_policy,
            before_retry=before_retry,
            total_timeout=total_timeout,
        )
