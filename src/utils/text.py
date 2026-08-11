"""Text utilities and shared UI copy helpers.

本模块处于 "共享文案层"：
- 仅依赖 Python 标准库；
- 可被 ACP 核心层与卡片 UI 层同时使用；
- 不直接依赖具体 UI 主题（例如颜色、布局等）。
"""

import threading
import time
import unicodedata

from src.utils.time_ago import TimeAgoBucket, compute_time_ago_bucket

_TASK_ID_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_TASK_ID_SEQ = 0


def sanitize_single_line_label(
    value: object,
    *,
    fallback: str,
    max_chars: int,
) -> str:
    """Return a bounded UTF-8-safe label without layout/control characters."""
    def _clean(raw: object) -> str:
        cleaned_chars: list[str] = []
        for character in str(raw or ""):
            category = unicodedata.category(character)
            if category == "Cs":
                cleaned_chars.append("\ufffd")
            elif category == "Cc" or character.isspace():
                cleaned_chars.append(" ")
            else:
                cleaned_chars.append(character)
        return " ".join("".join(cleaned_chars).split())

    cleaned = _clean(value)
    if not cleaned:
        cleaned = _clean(fallback)
    return cleaned[: max(0, int(max_chars))]


def utf8_replace_bytes(value: object) -> bytes:
    """Encode arbitrary text deterministically even when it contains surrogates."""
    return str(value or "").encode("utf-8", errors="replace")


def format_time_ago_from_bucket(bucket: TimeAgoBucket) -> str:
    """从已有 :class:`TimeAgoBucket` 渲染相对时间文案。

    该辅助函数主要用于调用方已经持有 bucket 的场景，避免重复计算秒数
    → bucket 的映射关系；其行为应始终与 ``render_time_ago_cn`` 一致。
    """

    return render_time_ago_cn(bucket)


def render_time_ago_cn(bucket: TimeAgoBucket) -> str:
    """将时间语义段渲染为中文文案。

    当前实现保持与原 :func:`format_time_ago` 完全一致的规则：
    - "seconds" → "刚刚"；
    - "minutes" → "{value} 分钟前"；
    - "hours"   → "{value} 小时前"；
    - "days"    → "{value} 天前"。

    未来如需支持多语言/多风格，可在调用方选择不同渲染函数，而保留 bucket 计算逻辑不变。
    """

    kind = bucket["kind"]
    value = int(bucket["value"])

    if kind == "seconds":
        return "刚刚"
    if kind == "minutes":
        return f"{value} 分钟前"
    if kind == "hours":
        return f"{value} 小时前"
    if kind == "days":
        return f"{value} 天前"

    # 理论上不会走到这里，但为了防御性编码，fallback 到「刚刚」。
    return "刚刚"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration string.

    Examples: "5 秒", "3 分钟 12 秒", "1 小时 30 分钟 5 秒"

    NOTE: 该函数只负责数值到中文描述的转换，不绑定具体 UI 主题，
    调用方自行决定前缀/图标（例如 "🔄 执行中 · "）。
    """
    if seconds < 0:
        seconds = 0
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hours > 0:
        return f"{hours} 小时 {minutes} 分钟 {secs} 秒"
    if minutes > 0:
        return f"{minutes} 分钟 {secs} 秒"
    return f"{secs} 秒"


def format_elapsed_clock(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS, with days as the largest unit."""
    total = max(0, int(seconds))
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}天 {clock}" if days else clock


def format_friendly_duration(seconds: float) -> str:
    """Format a duration into a friendly Chinese string without '前' suffix.

    - < 60s  → "X 秒"
    - < 3600s → "约 X 分钟"
    - < 86400s → "约 X 小时 Y 分钟"
    - >= 86400s → "约 X 天 Y 小时"
    """
    elapsed = max(0, seconds)
    if elapsed < 60:
        return f"{int(elapsed)} 秒"
    if elapsed < 3600:
        return f"约 {int(elapsed // 60)} 分钟"
    if elapsed < 86400:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        if minutes:
            return f"约 {hours} 小时 {minutes} 分钟"
        return f"约 {hours} 小时"
    days = int(elapsed // 86400)
    hours = int((elapsed % 86400) // 3600)
    if hours:
        return f"约 {days} 天 {hours} 小时"
    return f"约 {days} 天"


def append_duration_to_title(title: str, duration_secs: float | None) -> str:
    """Append formatted duration to title if available. E.g. '🔄 执行中 · 3分钟45秒'."""
    if duration_secs:
        return f"{title} · {format_duration(duration_secs)}"
    return title


def summarize_question_title(
    value: str | None,
    *,
    max_chars: int = 15,
    fallback: str = "Deep 任务",
) -> str:
    """Normalize a user question into a deterministic, bounded card title."""
    normalized = " ".join(str(value or "").split())
    if not normalized:
        normalized = " ".join(str(fallback or "").split())
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def format_time_ago(seconds: float) -> str:
    """统一的相对时间文案（"X 时间前" 风格）。

    该函数作为共享文案层的默认中文入口，内部实现分为两步：
    1. 使用 :func:`compute_time_ago_bucket` 将秒数转换为语义化区间；
    2. 使用 :func:`render_time_ago_cn` 将语义区间渲染为中文文案。

    语义规则（以秒数为输入）：
    - 负数或异常输入按 0 处理；
    - <60 秒：bucket.kind == "seconds" → "刚刚"；
    - <60 分钟：bucket.kind == "minutes" → "{m} 分钟前"；
    - <24 小时：bucket.kind == "hours" → "{h} 小时前"；
    - 其他：bucket.kind == "days" → "{d} 天前"。

    若调用方需要更精细的调试信息，可同时展示原始秒数/时间戳；不建议在调用方自行拼接另一套文案。
    """

    bucket = compute_time_ago_bucket(seconds)
    return render_time_ago_cn(bucket)


def generate_task_id(project_name: str) -> str:
    """Generate a human-readable task ID: {name}_{YYYYMMDD}_{HHMMSS}_{4hex}.

    Includes 4 random hex chars to avoid collisions on rapid submission.
    """
    global _TASK_ID_SEQ
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in project_name)[:30]
    # uuid4 的 16-bit 截断在高并发/同秒内仍可能碰撞；这里改为进程内自增序号保证稳定去重。
    with _TASK_ID_LOCK:
        _TASK_ID_SEQ = (_TASK_ID_SEQ + 1) & 0xFFFF
        suffix = f"{_TASK_ID_SEQ:04x}"
    return f"{safe_name}_{ts}_{suffix}"


def make_progress_bar(completed: int, total: int) -> str:
    """Render a text progress bar: ▰▰▰▰▰▱▱▱▱▱ 50% (5/10)."""
    if total == 0:
        return "▱▱▱▱▱▱▱▱▱▱ 0%"

    percent = (completed / total) * 100
    filled = int(percent / 10)
    empty = 10 - filled

    return f"{'▰' * filled}{'▱' * empty} {percent:.0f}% ({completed}/{total})"


def truncate_output(output: str, max_len: int, label: str = "输出被截断") -> str:
    if len(output) > max_len:
        return output[:max_len] + f"\n\n... ({label}，共 {len(output)} 字符)"
    return output



def get_acp_result_header_text() -> dict[str, str]:
    """Return localized ACP result section headers.

    暂时返回固定中文/英文混合文案，未来可以根据 locale 做切换。
    """

    return {
        "text": "结果文本",
        "plan": "执行计划",
        "tools": "工具调用",
        "tool_results": "工具执行记录",
        "files": "改动文件",
        # Tool descriptions used in ACP helper / system cards
        "tool_desc_coco": "字节跳动 AI",
        "tool_desc_claude": "Anthropic AI",
        "tool_desc_aiden": "Aiden CLI",
        "tool_desc_codex": "OpenAI Codex",
        "tool_desc_gemini": "Google Gemini CLI",
        "tool_desc_traex": "TRAE CLI",
        "tool_desc_grok": "xAI Grok Build",
    }
