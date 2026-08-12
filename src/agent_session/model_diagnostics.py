"""Model failure diagnostics, compaction, and argument manipulation."""

from __future__ import annotations

import logging
import re as _re
from typing import Callable, Optional

from ..config import get_settings
from ..utils.errors import get_error_detail
from .protocol import SyncSession

logger = logging.getLogger(__name__)


_RATE_LIMIT_PATTERNS = [
    _re.compile(r"rate.?limit", _re.IGNORECASE),
    _re.compile(r"\b429\b"),
    _re.compile(r"too many requests", _re.IGNORECASE),
    _re.compile(r"overloaded", _re.IGNORECASE),
]

_RETRY_AFTER_RE = _re.compile(r"retry[_\- ]?after[:\s=]*(\d+)", _re.IGNORECASE)


# =====================================================================
# Model compaction / loop / failover detection (send_prompt-time)
# =====================================================================
#
# 这些错误通常由底层模型服务返回，表现为：
# - "Model failed: model 'gpt-5.2': receive message: need compaction"
# - "loop detected"
# - "Failing over to: gpt-5.1"
#
# 统一诊断字段（稳定契约，供日志与单测冻结）：
# - fail_phase: str         # model_compaction | model_loop | model_failover | unknown
# - reason: str             # need_compaction | loop_detected | unknown
# - failed_model: str       # 从错误文本中解析的模型名（best-effort）
# - failover_to: str        # 从错误文本中解析的 failover 目标（best-effort）
# - attempt_count: int      # loop 检测用（窗口期内计数），若未知则为 0

_NEED_COMPACTION_RE = _re.compile(r"\bneed\s+compaction\b", _re.IGNORECASE)
_LOOP_DETECTED_RE = _re.compile(r"\bloop\s+detected\b", _re.IGNORECASE)
_FAILED_MODEL_RE = _re.compile(r"\bmodel\s*['\"]([^'\"\s]+)['\"]", _re.IGNORECASE)
_FAILOVER_TO_RE = _re.compile(r"\bfailing\s+over\s+to\s*:\s*([^\s]+)", _re.IGNORECASE)


def _build_generic_error_blob(error: Exception) -> str:
    """将 error 转成可匹配的通用文本 blob（best-effort, never raises）。

    注意：该函数用于 compaction/loop/failover 等"通用模型失败"检测。
    Provider-specific parsing stays outside this generic error collector.
    """
    parts: list[str] = []
    parts.append(get_error_detail(error))
    # 兼容携带 snippet 字段的启动异常
    for k in ("stderr_snippet", "stdout_snippet", "stderr", "stdout", "message"):
        try:
            v = getattr(error, k, None)
            if v:
                parts.append(str(v))
        except Exception:
            logger.debug("_build_generic_error_blob: getattr %s failed", k, exc_info=True)
            continue
    return "\n".join([p for p in parts if p])


def _extract_failed_model(blob: str) -> str:
    """从错误文本中提取失败模型名（best-effort）。"""
    try:
        m = _FAILED_MODEL_RE.search(blob or "")
        return (m.group(1) or "").strip() if m else ""
    except Exception:
        logger.debug("_extract_failed_model: regex failed", exc_info=True)
        return ""


def _extract_failover_to(blob: str) -> str:
    """从错误文本中提取 failover 目标模型名（best-effort）。"""
    try:
        m = _FAILOVER_TO_RE.search(blob or "")
        return (m.group(1) or "").strip() if m else ""
    except Exception:
        logger.debug("_extract_failover_to: regex failed", exc_info=True)
        return ""


def classify_model_failure(*, error: Exception) -> dict:
    """分类模型失败原因（send_prompt-time）。

    返回字段遵循本文件顶部"统一诊断字段"约定。
    """
    blob = _build_generic_error_blob(error)
    failed_model = _extract_failed_model(blob)
    failover_to = _extract_failover_to(blob)

    reason = "unknown"
    fail_phase = "unknown"
    try:
        if _NEED_COMPACTION_RE.search(blob or ""):
            reason = "need_compaction"
            fail_phase = "model_compaction"
        elif _LOOP_DETECTED_RE.search(blob or ""):
            reason = "loop_detected"
            fail_phase = "model_loop"
    except Exception:
        logger.debug("classify_model_failure: classification failed", exc_info=True)
        reason = "unknown"
        fail_phase = "unknown"

    # failover 目标存在时，标记 fail_phase 为 model_failover（不覆盖更具体的 compaction/loop）
    if (fail_phase == "unknown") and bool(failover_to):
        fail_phase = "model_failover"

    # 注意：attempt_count 由上层 loop 检测器回填；这里保持 0。
    return {
        "fail_phase": fail_phase,
        "reason": reason,
        "failed_model": failed_model,
        "failover_to": failover_to,
        "attempt_count": 0,
        "error_blob": blob,
    }


def _extract_model_from_agent_args(args: list[str]) -> str:
    """从 agent_args 中 best-effort 提取当前 model 名称。"""
    try:
        xs = [str(x) for x in (args or [])]
    except (TypeError, ValueError):
        logger.debug("_extract_model_from_agent_args: args conversion failed", exc_info=True)
        return ""

    # coco: -c model.name=xxx
    for i, x in enumerate(xs):
        if not x:
            continue
        if x == "-c" and i + 1 < len(xs):
            y = str(xs[i + 1] or "")
            if y.startswith("model.name="):
                return y.split("=", 1)[1].strip()
        if "model.name=" in x:
            try:
                return x.split("model.name=", 1)[1].strip()
            except Exception:
                logger.debug("_extract_model_from_agent_args: model.name= split failed", exc_info=True)
                continue

    # generic: first -m <value>
    for i, x in enumerate(xs):
        if x == "-m" and i + 1 < len(xs):
            return str(xs[i + 1] or "").strip()
    return ""


def _replace_model_in_agent_args(args: list[str], new_model: str) -> tuple[list[str], bool]:
    """在 agent_args 中替换 model 参数（best-effort）。

    返回 (new_args, replaced)。
    """
    new_model = str(new_model or "").strip()
    if not new_model:
        return (list(args or []), False)

    try:
        xs = [str(x) for x in (args or [])]
    except (TypeError, ValueError):
        logger.debug("_replace_model_in_agent_args: args conversion failed", exc_info=True)
        xs = list(args or [])

    out = list(xs)
    replaced = False

    # coco style: -c model.name=xxx
    for i, x in enumerate(out):
        if x == "-c" and i + 1 < len(out):
            y = str(out[i + 1] or "")
            if y.startswith("model.name="):
                out[i + 1] = f"model.name={new_model}"
                replaced = True
                break
    if replaced:
        return (out, True)

    # generic: first -m <value>
    for i, x in enumerate(out):
        if x == "-m" and i + 1 < len(out):
            out[i + 1] = new_model
            return (out, True)

    return (out, False)


def _apply_compaction_once(
    *,
    session: SyncSession,
    session_builder: Optional[Callable[..., SyncSession]] = None,
    startup_timeout_s: Optional[float] = None,
) -> Optional[SyncSession]:
    """对当前 session 执行一次"轻量 compaction"处理（best-effort）。

    设计取舍：
    - 这里不尝试"压缩 LLM 上下文"（ACP 协议当前无该能力），而是通过重建会话来清空上下文。
    - 对于支持 resume 的场景，调用方应使用更高层的恢复逻辑；此处只服务于运行期自动自愈。

    返回新 session（已启动）表示已执行并认为可能有帮助；返回 None 表示无法执行。
    """
    from ..acp.sync_adapter import SyncACPSession

    # 如果没有必要的生命周期方法，直接失败（避免 AttributeError 冒泡）。
    if not hasattr(session, "close") or not hasattr(session, "start"):
        return None

    # 尽量保留 agent_type/cwd 以便重建（仅对 ACP Session 有意义）。
    agent_type = str(getattr(session, "_agent_type", "") or "")
    cwd = str(getattr(session, "_cwd", "") or "")
    if not agent_type or not cwd:
        return None

    # 继承 cmd/args 启动参数
    agent_cmd = str(getattr(session, "_agent_cmd", "") or "")
    agent_args = list(getattr(session, "_agent_args", []) or [])

    if not agent_cmd and not agent_args:
        # 不是 ACP backend 或缺少启动信息
        return None

    # 重建新会话（仅 ACP 后端），保持相同 cmd/args（即保持同模型）。
    try:
        timeout_s = float(startup_timeout_s or getattr(get_settings(), "acp_startup_timeout", 20) or 20)
    except Exception:
        logger.debug("_apply_compaction_once: timeout_s conversion failed", exc_info=True)
        timeout_s = 20.0
    timeout_s = max(1.0, timeout_s)

    builder = session_builder
    if builder is None:

        def builder(**kwargs):
            return SyncACPSession(**kwargs)

    try:
        new_sess = builder(
            agent_type=agent_type,
            cwd=cwd,
            agent_cmd=agent_cmd,
            agent_args=list(agent_args),
            capture_full_tool_content=bool(
                getattr(session, "_capture_full_tool_content", False)
            ),
        )
        new_sess.start(startup_timeout=timeout_s)
        return new_sess
    except (RuntimeError, OSError, TimeoutError):
        logger.debug("_apply_compaction_once: rebuild session failed", exc_info=True)
        return None


def _default_compaction_action(*, session: SyncSession) -> Optional[SyncSession]:
    """默认 compaction 动作（best-effort，可用于生产调用）。"""
    return _apply_compaction_once(session=session)


def _detect_rate_limit(error: Exception) -> Optional[int]:
    """Detect rate limiting from error.  Returns suggested wait seconds or 0 (detected
    but no explicit wait), or None (not a rate-limit error)."""
    msg = get_error_detail(error)
    for pat in _RATE_LIMIT_PATTERNS:
        if pat.search(msg):
            m = _RETRY_AFTER_RE.search(msg)
            if m:
                val = int(m.group(1))
                return max(1, min(val, 600))  # clamp to [1, 600]
            return 0  # detected but no explicit wait
    return None
