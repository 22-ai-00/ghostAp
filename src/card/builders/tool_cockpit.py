"""Pure Card JSON 2.0 builders for Tool Usage (/usage) & Tool Settings (/settings) cockpits.

Strict import boundary: this module MUST NEVER import ``src.card.delivery``,
``src.card.session``, or ``src.feishu``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Optional

from src.acp.agy_selection import is_agy_orchestrator_model
from src.acp.options import ACPModelOption, ACPModelSelectionVariant
from src.card.actions import dispatch as action_ids
from src.card.backend_meta import get_backend_meta
from src.card.render.model_cascade import resolve_model_cascade
from src.card.schema import require_card_json_2
from src.card.shared import build_callback_button, build_responsive_layout
from src.card.shared.text_safety import neutralize_feishu_rich_text_controls
from src.card.shared.truncation import count_tagged_nodes
from src.card.thresholds import THRESHOLDS
from src.card.ui_text import UI_TEXT
from src.utils.text import format_duration

from .core import CoreBuilder

_CARD_BYTE_BUDGET = THRESHOLDS["CARD_BYTE_BUDGET"]
_CARD_NODE_BUDGET = THRESHOLDS["CARD_NODE_BUDGET"]
_MAX_SELECT_OPTIONS = 25
_MAX_QUOTA_GROUPS = 6
_MAX_BUCKETS_PER_GROUP = 6

_VERBOSITY_CHOICES: tuple[tuple[str, str], ...] = (
    ("low", "low · 精简输出"),
    ("medium", "medium · 标准输出"),
    ("high", "high · 详细输出"),
)

_AGENT_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("auto", "auto · 全自动执行"),
    ("accept-edits", "accept-edits · 自动接受编辑"),
    ("plan", "plan · 先规划后执行"),
)

_SENSITIVE_CONFIG_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "key",
    "path",
    "trustedworkspaces",
    "proxy",
    "auth",
)


def _strip_surrogates(value: Any) -> str:
    """Replace lone UTF-16 surrogates so JSON serialization stays strict UTF-8."""
    return str(value or "").encode("utf-8", errors="replace").decode("utf-8")


def _safe_one_line(value: Any, *, limit: int = 120) -> str:
    """Collapse whitespace, neutralize Feishu control tags, and bound length."""
    text = neutralize_feishu_rich_text_controls(_strip_surrogates(value))
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _escape_md(value: Any, *, limit: int = 180) -> str:
    """Escape Markdown metacharacters inside single-line untrusted values."""
    text = _safe_one_line(value, limit=limit)
    for ch in ("*", "_", "`", "|", "[", "]", "~"):
        text = text.replace(ch, "\\" + ch)
    return text


def _fit_summary(text: str, *, default: str) -> str:
    """Clamp card notification summary text into Feishu's 8..60-char envelope."""
    safe_default = _strip_surrogates(default)
    clean = " ".join(_strip_surrogates(text).split()) or safe_default
    if len(clean) < 8:
        clean = f"工具驾驶舱 · {clean}"
    if len(clean) < 8:
        clean = safe_default
    if len(clean) > 60:
        clean = clean[:59].rstrip() + "…"
    return clean


def _get_field(obj: Any, *names: str, default: Any = None) -> Any:
    """Read duck-typed attributes or mapping entries without assuming concrete types."""
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
        elif hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    return default


def _coerce_float(value: Any) -> float | None:
    """Convert numeric or percentage string values to a finite float."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if (math.isnan(value) or math.isinf(value)) else float(value)
    if isinstance(value, str):
        stripped = value.strip().rstrip("%")
        if not stripped:
            return None
        try:
            num = float(stripped)
            return None if (math.isnan(num) or math.isinf(num)) else num
        except ValueError:
            return None
    return None


def _coerce_fraction(bucket: Any) -> float | None:
    """Extract a normalized ``0.0..1.0`` remaining quota fraction from *bucket*."""
    raw = _get_field(bucket, "remaining_fraction", "remaining", "fraction")
    val = _coerce_float(raw)
    if val is not None:
        if 1.0 < val <= 100.0:
            return max(0.0, min(1.0, val / 100.0))
        return max(0.0, min(1.0, val))
    pct = _coerce_float(_get_field(bucket, "remaining_pct", "remaining_percent", "percent"))
    if pct is not None:
        return max(0.0, min(1.0, pct / 100.0))
    return None


def _coerce_int(value: Any) -> int | None:
    """Coerce an integer-like field while ignoring booleans and NaN."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _format_quota_bar(fraction: float | None) -> tuple[str, str]:
    """Render a 10-block Unicode quota bar (``█``/``░``) and percentage string."""
    if fraction is None:
        return "──────────", "未知"
    clamped = max(0.0, min(1.0, fraction))
    filled = max(0, min(10, int(round(clamped * 10))))
    bar = ("█" * filled) + ("░" * (10 - filled))
    pct = clamped * 100.0
    if 0.0 < pct < 10.0 and abs(pct - round(pct)) >= 0.05:
        pct_text = f"{pct:.1f}%"
    else:
        pct_text = f"{round(pct):.0f}%"
    return bar, pct_text


def _is_epoch_1970(reset_time: Any) -> bool:
    """Return True when *reset_time* represents the 1970-01-01 sentinel timestamp."""
    if isinstance(reset_time, (int, float)) and not isinstance(reset_time, bool):
        return reset_time <= 0
    text = str(reset_time or "").strip()
    if not text:
        return False
    if text.startswith("1970-01-01"):
        return True
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.year <= 1970 and dt.timestamp() <= 86400
    except ValueError:
        return False


def _format_reset_time(reset_time: Any, *, remaining_fraction: float | None) -> str:
    """Format quota reset timestamps, mapping 1970 full-quota sentinels cleanly."""
    if reset_time is None or str(reset_time).strip() == "":
        if remaining_fraction is not None and remaining_fraction >= 0.999:
            return UI_TEXT["system_tool_usage_full_quota"]
        return "未提供"

    if _is_epoch_1970(reset_time):
        if remaining_fraction is not None and remaining_fraction >= 0.999:
            return UI_TEXT["system_tool_usage_full_quota"]
        return "待触发"

    if isinstance(reset_time, (int, float)) and not isinstance(reset_time, bool):
        try:
            dt = datetime.fromtimestamp(float(reset_time), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return _safe_one_line(reset_time, limit=40)
    else:
        raw_str = str(reset_time).strip()
        try:
            dt = datetime.fromisoformat(raw_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
        except ValueError:
            return _safe_one_line(raw_str, limit=40)

    formatted_utc = dt.strftime("%Y-%m-%d %H:%M UTC")
    delta_s = (dt - datetime.now(timezone.utc)).total_seconds()
    if delta_s > 60:
        days = int(delta_s // 86400)
        hours = int((delta_s % 86400) // 3600)
        mins = int((delta_s % 3600) // 60)
        if days > 0:
            rel = f"{days}天{hours}小时后"
        elif hours > 0:
            rel = f"{hours}小时{mins}分后"
        else:
            rel = f"{max(1, mins)}分钟后"
        return f"{rel} ({formatted_utc})"
    return f"重置时间 {formatted_utc}"


def _format_window_label(bucket: Any) -> str:
    """Resolve human-readable window labels (e.g. ``5h`` / ``weekly``)."""
    window = str(_get_field(bucket, "window", "period", "id", default="") or "").strip()
    name = str(_get_field(bucket, "name", "label", default="") or "").strip()
    lowered = window.lower()
    if lowered == "5h" or lowered.endswith("-5h") or "five hour" in name.lower():
        return "5h 窗口"
    if lowered == "weekly" or lowered.endswith("-weekly") or "weekly" in name.lower():
        return "每周限额"
    if name:
        return _safe_one_line(name, limit=32)
    if window:
        return _safe_one_line(window, limit=32)
    return "配额"


def _format_credits(value: Any) -> str:
    """Format G1 credit counts cleanly."""
    num = _coerce_float(value)
    if num is None:
        return _safe_one_line(value, limit=32)
    if abs(num - round(num)) < 1e-6:
        return f"{int(round(num)):,}"
    return f"{num:,.2f}"


def _format_cost(amount: Any, currency: Any) -> str | None:
    """Format optional session cost with currency."""
    num = _coerce_float(amount)
    if num is None:
        return None
    cur = _safe_one_line(currency or "USD", limit=12) or "USD"
    text = f"{num:.4f}".rstrip("0").rstrip(".") if num > 0 else "0"
    return f"{text} {cur}"


def _tool_display_parts(tool_name: str) -> tuple[str, str, str]:
    """Return ``(normalized_key, emoji, display_name)`` for *tool_name*."""
    tool = str(tool_name or "").strip().lower() or "agy"
    meta = get_backend_meta(tool)
    if meta is not None:
        return tool, meta.emoji, meta.display_name
    return tool, "🤖", tool.upper()


def _cap_options_preserving(
    options: Sequence[str],
    *,
    keep: Sequence[str | None] = (),
    limit: int = _MAX_SELECT_OPTIONS,
) -> tuple[str, ...]:
    """Cap dropdown option lists while guaranteeing selected options stay present."""
    ordered = list(dict.fromkeys(opt for opt in options if opt))
    if len(ordered) <= limit:
        return tuple(ordered)
    kept = ordered[: max(0, limit - len([k for k in keep if k in ordered]))]
    for item in keep:
        if item and item in ordered and item not in kept:
            kept.append(item)
    return tuple(kept[:limit])


def _coerce_model_options(models: Sequence[Any]) -> list[ACPModelOption]:
    """Normalize duck-typed model entries into ``ACPModelOption`` instances."""
    out: list[ACPModelOption] = []
    for item in models or ():
        if isinstance(item, ACPModelOption):
            out.append(item)
            continue
        if isinstance(item, str):
            name = item.strip()
            if name:
                out.append(ACPModelOption(name=name))
            continue
        name = str(_get_field(item, "name", "model_id", "id", default="") or "").strip()
        if not name:
            continue
        raw_variants = _get_field(item, "selection_variants", default=()) or ()
        variants: list[ACPModelSelectionVariant] = []
        for var in raw_variants:
            v_name = str(_get_field(var, "name", default="") or "").strip()
            v_model = str(_get_field(var, "model", default=name) or "").strip()
            if not v_name or not v_model:
                continue
            v_profile = _get_field(var, "profile")
            v_effort = _get_field(var, "effort")
            variants.append(
                ACPModelSelectionVariant(
                    name=v_name,
                    model=v_model,
                    profile=str(v_profile).strip().lower() if v_profile else None,
                    effort=str(v_effort).strip().lower() if v_effort else None,
                    is_default=bool(_get_field(var, "is_default", default=False)),
                )
            )
        raw_efforts = _get_field(item, "reasoning_efforts", default=()) or ()
        efforts = tuple(str(e).strip().lower() for e in raw_efforts if str(e).strip())
        if not variants and efforts:
            default_eff = _get_field(item, "default_reasoning_effort")
            variants = [
                ACPModelSelectionVariant(
                    name=f"{name}/{eff}",
                    model=name,
                    effort=eff,
                    is_default=(eff == str(default_eff or "").strip().lower()),
                )
                for eff in efforts
            ]
        out.append(
            ACPModelOption(
                name=name,
                description=str(_get_field(item, "description", default="") or "").strip(),
                is_default=bool(_get_field(item, "is_default", default=False)),
                selection_variants=tuple(variants),
                reasoning_efforts=efforts,
                default_reasoning_effort=_get_field(item, "default_reasoning_effort"),
            )
        )
    return out


def _normalize_available_tools(tool: str, available_tools: Sequence[str]) -> tuple[str, ...]:
    """Deduplicate *available_tools* and ensure *tool* is selectable."""
    seen: list[str] = []
    for raw in available_tools or ():
        key = str(raw or "").strip().lower()
        if key and key not in seen:
            seen.append(key)
    if tool and tool not in seen:
        seen.insert(0, tool)
    if not seen:
        seen.append("agy")
    return tuple(seen[:_MAX_SELECT_OPTIONS])


def _extract_agy_runtime_values(settings_snapshot: Any) -> tuple[bool, str, str]:
    """Extract normalized ``(show_thinking, verbosity, agent_mode)`` for AGY."""
    runtime_map = _get_field(settings_snapshot, "runtime_settings", "saved_runtime_settings")
    raw_show = _get_field(settings_snapshot, "show_thinking")
    if raw_show is None and isinstance(runtime_map, Mapping):
        raw_show = runtime_map.get("show_thinking")
    show_thinking = bool(raw_show) if raw_show is not None else False

    raw_verbosity = _get_field(settings_snapshot, "verbosity")
    if raw_verbosity is None and isinstance(runtime_map, Mapping):
        raw_verbosity = runtime_map.get("verbosity")
    verbosity = str(raw_verbosity or "medium").strip().lower()
    if verbosity not in {"low", "medium", "high"}:
        verbosity = "medium"

    raw_mode = _get_field(settings_snapshot, "agent_mode", "agentMode")
    if raw_mode is None and isinstance(runtime_map, Mapping):
        raw_mode = runtime_map.get("agent_mode") or runtime_map.get("agentMode")
    agent_mode = str(raw_mode or "auto").strip().lower()
    if agent_mode not in {"auto", "accept-edits", "plan"}:
        agent_mode = "auto"

    return show_thinking, verbosity, agent_mode


def _extract_readonly_backend_summary(settings_snapshot: Any) -> list[tuple[str, str]]:
    """Extract safe read-only backend config summary entries, omitting secrets/paths."""
    raw = _get_field(
        settings_snapshot,
        "backend_config_summary",
        "backend_config",
        "config_summary",
        "readonly_config",
    )
    if not isinstance(raw, Mapping) or not raw:
        return []
    pairs: list[tuple[str, str]] = []
    for key, val in raw.items():
        k_str = str(key or "").strip()
        if not k_str or any(sens in k_str.lower() for sens in _SENSITIVE_CONFIG_KEY_PARTS):
            continue
        if isinstance(val, bool):
            v_str = "开启" if val else "关闭"
        elif val is None or val == "":
            continue
        elif isinstance(val, (int, float)):
            v_str = str(val)
        elif isinstance(val, str):
            if "/" in val or "\\" in val:
                continue
            v_str = _safe_one_line(val, limit=48)
        elif isinstance(val, Mapping):
            v_str = _safe_one_line(json.dumps(dict(val), ensure_ascii=False), limit=64)
        else:
            continue
        pairs.append((_safe_one_line(k_str, limit=32), v_str))
        if len(pairs) >= 8:
            break
    return pairs


def _build_card_shell(
    *,
    title: str,
    subtitle: str | None,
    template: str,
    summary_content: str,
    elements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Wrap *elements* in a validated Card JSON 2.0 shell with ``update_multi=True``."""
    header: dict[str, Any] = {
        "title": {"tag": "plain_text", "content": _safe_one_line(title, limit=60)},
        "template": template,
    }
    if subtitle:
        header["subtitle"] = {"tag": "plain_text", "content": _safe_one_line(subtitle, limit=80)}
    card: dict[str, Any] = {
        "schema": "2.0",
        "config": {
            "wide_screen_mode": True,
            "update_multi": True,
            "summary": {"content": _fit_summary(summary_content, default=title)},
        },
        "header": header,
        "body": {"elements": elements},
    }
    require_card_json_2(card)
    return card


def _build_select_static(
    *,
    name: str,
    placeholder: str,
    options: Sequence[tuple[str, str]],
    selected: Optional[str],
    callback: dict[str, Any],
) -> dict[str, Any]:
    """Construct a Card 2.0 ``select_static`` element with callback behavior."""
    option_items = [
        {
            "text": {"tag": "plain_text", "content": _safe_one_line(label, limit=64)},
            "value": _strip_surrogates(opt_val),
        }
        for opt_val, label in options
    ]
    element: dict[str, Any] = {
        "tag": "select_static",
        "name": name,
        "placeholder": {"tag": "plain_text", "content": _safe_one_line(placeholder, limit=80)},
        "options": option_items,
        "value": callback,
        "behaviors": [{"type": "callback", "value": dict(callback)}],
    }
    opt_values = [item["value"] for item in option_items]
    if selected is not None and selected in opt_values:
        element["initial_option"] = selected
    return element


def _render_quota_elements(usage_snapshot: Any) -> tuple[list[dict[str, Any]], str]:
    """Render quota bucket progress bars, G1 credits, and compute adaptive header color."""
    quota_groups = list(_get_field(usage_snapshot, "quota_groups", "groups", default=()) or ())
    min_remaining: float | None = None
    for group in quota_groups:
        for bucket in _get_field(group, "buckets", default=()) or ():
            frac = _coerce_fraction(bucket)
            if frac is not None and (min_remaining is None or frac < min_remaining):
                min_remaining = frac

    if min_remaining is not None and min_remaining < 0.05:
        header_template = "red"
    elif min_remaining is not None and min_remaining < 0.20:
        header_template = "orange"
    else:
        header_template = "blue"

    elements: list[dict[str, Any]] = []
    if quota_groups:
        quota_lines: list[str] = ["**🔋 模型配额窗口**"]
        for group in quota_groups[:_MAX_QUOTA_GROUPS]:
            group_name = _escape_md(
                _get_field(group, "name", "group_name", "label", default="模型配额"),
                limit=64,
            )
            quota_lines.append(f"\n**{group_name}**")
            buckets = list(_get_field(group, "buckets", default=()) or ())
            for bucket in buckets[:_MAX_BUCKETS_PER_GROUP]:
                frac = _coerce_fraction(bucket)
                bar, pct_text = _format_quota_bar(frac)
                win_label = _format_window_label(bucket)
                reset_text = _format_reset_time(
                    _get_field(bucket, "reset_time", "reset_at"),
                    remaining_fraction=frac,
                )
                quota_lines.append(f"• {win_label}：`{bar}` **{pct_text}** · {reset_text}")
        elements.append({"tag": "markdown", "content": "\n".join(quota_lines)})
    else:
        elements.append(
            {
                "tag": "markdown",
                "content": (
                    "**🔋 模型配额窗口**\n"
                    f"<font color='grey'>{UI_TEXT['system_tool_usage_no_quota_hint']}</font>"
                ),
            }
        )

    credits_val = _get_field(usage_snapshot, "remaining_credits", "g1_credits", "credits")
    if credits_val is not None:
        use_g1 = _get_field(usage_snapshot, "use_g1_credits")
        g1_suffix = (
            f" · {'已开启 G1 超额抵扣' if use_g1 else '未开启 G1 抵扣'}"
            if isinstance(use_g1, bool)
            else ""
        )
        elements.append(
            {
                "tag": "markdown",
                "content": f"💎 **G1 Credits 余额**：`{_format_credits(credits_val)}`{g1_suffix}",
            }
        )

    return elements, header_template


def _render_session_element(usage_snapshot: Any) -> dict[str, Any]:
    """Render live session status, model, turns, duration, context window, and cost."""
    session_id = str(_get_field(usage_snapshot, "session_id", default="") or "").strip()
    raw_active = _get_field(usage_snapshot, "session_active", "has_active_session", "is_active")
    session_active = bool(raw_active) if raw_active is not None else bool(session_id)
    session_state_str = str(_get_field(usage_snapshot, "session_state", default="") or "").strip()
    if session_state_str:
        state_badge = _escape_md(session_state_str, limit=40)
    elif session_active:
        short_sid = f"{session_id[:12]}…" if len(session_id) > 12 else session_id
        state_badge = f"🟢 活跃 (`{short_sid}`)" if short_sid else "🟢 活跃"
    else:
        state_badge = "⚪ 空闲（未启动）"

    model_name = str(
        _get_field(usage_snapshot, "model_name", "current_model", "model", default="") or ""
    ).strip()
    model_display = _escape_md(model_name or "default（后端默认）", limit=64)

    turns = _coerce_int(_get_field(usage_snapshot, "turns", "message_count", "num_turns"))
    turns_display = f"{turns} 轮" if turns is not None else "0 轮"

    duration_s = _coerce_float(
        _get_field(usage_snapshot, "duration_seconds", "session_duration_seconds")
    )
    duration_display = (
        format_duration(max(0.0, duration_s))
        if duration_s is not None and duration_s >= 0
        else "—"
    )

    nested_usage = _get_field(usage_snapshot, "usage_info", "acp_usage", "usage")
    used_tokens = _coerce_int(
        _get_field(usage_snapshot, "context_used", "used_tokens", "usage_used")
        or _get_field(nested_usage, "used")
    )
    size_tokens = _coerce_int(
        _get_field(usage_snapshot, "context_size", "context_window", "usage_size")
        or _get_field(nested_usage, "size")
    )
    if size_tokens and size_tokens > 0:
        used_val = max(0, used_tokens or 0)
        ctx_pct = round((used_val / size_tokens) * 100)
        context_display = f"`{used_val:,} / {size_tokens:,}` ({ctx_pct}%)"
    elif used_tokens and used_tokens > 0:
        context_display = f"`{used_tokens:,}` tokens"
    else:
        context_display = "暂未上报"

    cost_amount = _get_field(usage_snapshot, "cost_amount", "cost")
    if cost_amount is None:
        cost_amount = _get_field(nested_usage, "cost_amount")
    cost_currency = _get_field(usage_snapshot, "cost_currency") or _get_field(
        nested_usage, "cost_currency"
    )
    cost_display = _format_cost(cost_amount, cost_currency)

    lines = [
        "**📊 会话与上下文状态**",
        f"• 会话状态：{state_badge}",
        f"• 当前模型：`{model_display}`",
        f"• 交互轮次：`{turns_display}` · 耗时：`{duration_display}`",
        f"• 上下文窗口：{context_display}",
    ]
    if cost_display:
        lines.append(f"• 累计费用：`{cost_display}`")
    return {"tag": "markdown", "content": "\n".join(lines)}


def _shrink_usage_card_if_needed(
    *,
    header_title: str,
    subtitle: str | None,
    header_template: str,
    summary_content: str,
    elements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the usage card and drop optional notes if payload exceeds card budgets."""
    card = _build_card_shell(
        title=header_title,
        subtitle=subtitle,
        template=header_template,
        summary_content=summary_content,
        elements=elements,
    )
    if (
        count_tagged_nodes(card) <= _CARD_NODE_BUDGET
        and len(json.dumps(card, ensure_ascii=False).encode("utf-8")) <= _CARD_BYTE_BUDGET
    ):
        return card
    trimmed = [
        el
        for el in elements
        if not (isinstance(el, dict) and el.get("tag") == "markdown" and "配额说明" in str(el.get("content", "")))
    ]
    return _build_card_shell(
        title=header_title,
        subtitle=subtitle,
        template=header_template,
        summary_content=summary_content,
        elements=trimmed,
    )


def _render_model_cascade_elements(
    tool: str,
    state: Any,
    payload_fn: Any,
) -> list[dict[str, Any]]:
    """Build ``model_group``, ``model_profile``, and ``model_effort`` elements."""
    elements: list[dict[str, Any]] = []
    if not state.model_names:
        elements.append({"tag": "markdown", "content": UI_TEXT["system_acp_model_group_label"]})
        elements.append(
            _build_select_static(
                name="model_group",
                placeholder=UI_TEXT["system_tool_settings_default_model"],
                options=[("default", UI_TEXT["system_tool_settings_default_model"])],
                selected="default",
                callback=payload_fn(action_ids.SETTINGS_SELECT_MODEL_GROUP),
            )
        )
        elements.append(
            {
                "tag": "markdown",
                "text_size": "notation",
                "content": UI_TEXT["system_tool_settings_no_effort_note"],
            }
        )
        return elements

    capped_models = _cap_options_preserving(
        state.model_names,
        keep=(state.selected_model,),
        limit=_MAX_SELECT_OPTIONS,
    )
    model_options = [
        (m, UI_TEXT["system_tool_settings_default_model"] if m == "default" else m)
        for m in capped_models
    ]
    elements.append({"tag": "markdown", "content": UI_TEXT["system_acp_model_group_label"]})
    elements.append(
        _build_select_static(
            name="model_group",
            placeholder=UI_TEXT["system_acp_model_group_placeholder"],
            options=model_options,
            selected=state.selected_model,
            callback=payload_fn(action_ids.SETTINGS_SELECT_MODEL_GROUP),
        )
    )

    is_agy_orch = tool == "agy" and is_agy_orchestrator_model(state.selected_model)
    if state.profiles:
        capped_profiles = _cap_options_preserving(
            state.profiles,
            keep=(state.selected_profile,),
            limit=_MAX_SELECT_OPTIONS,
        )
        profile_label = (
            UI_TEXT["system_acp_agy_orchestrator_planner_effort_label"]
            if is_agy_orch
            else UI_TEXT["system_acp_model_profile_label"]
        )
        profile_placeholder = (
            UI_TEXT["system_acp_agy_orchestrator_planner_effort_placeholder"]
            if is_agy_orch
            else UI_TEXT["system_acp_model_profile_placeholder"]
        )
        elements.append({"tag": "markdown", "content": profile_label})
        elements.append(
            _build_select_static(
                name="model_profile",
                placeholder=profile_placeholder,
                options=[(p, p) for p in capped_profiles],
                selected=state.selected_profile,
                callback=payload_fn(action_ids.SETTINGS_SELECT_MODEL_PROFILE),
            )
        )

    if state.efforts:
        capped_efforts = _cap_options_preserving(
            state.efforts,
            keep=(state.selected_effort,),
            limit=_MAX_SELECT_OPTIONS,
        )
        effort_label = (
            UI_TEXT["system_acp_agy_orchestrator_executor_effort_label"]
            if is_agy_orch
            else UI_TEXT["system_acp_model_effort_label"]
        )
        effort_placeholder = (
            UI_TEXT["system_acp_agy_orchestrator_executor_effort_placeholder"]
            if is_agy_orch
            else UI_TEXT["system_acp_model_effort_placeholder"]
        )
        elements.append({"tag": "markdown", "content": effort_label})
        elements.append(
            _build_select_static(
                name="model_effort",
                placeholder=effort_placeholder,
                options=[(e, e) for e in capped_efforts],
                selected=state.selected_effort,
                callback=payload_fn(action_ids.SETTINGS_SELECT_MODEL_EFFORT),
            )
        )
    else:
        elements.append(
            {
                "tag": "markdown",
                "text_size": "notation",
                "content": UI_TEXT["system_tool_settings_no_effort_note"],
            }
        )
    return elements


def _render_agy_runtime_elements(
    *,
    show_thinking: bool,
    verbosity: str,
    agent_mode: str,
    payload_fn: Any,
) -> list[dict[str, Any]]:
    """Build AGY-specific runtime toggles and dropdowns."""
    thinking_label = "💭 显示思考过程：已开启" if show_thinking else "💭 显示思考过程：已关闭"
    elements: list[dict[str, Any]] = [
        {"tag": "hr"},
        {"tag": "markdown", "content": UI_TEXT["system_tool_settings_agy_section"]},
        *build_responsive_layout(
            [
                build_callback_button(
                    thinking_label,
                    payload_fn(
                        action_ids.SETTINGS_TOGGLE_SHOW_THINKING,
                        show_thinking=not show_thinking,
                    ),
                    button_type="primary" if show_thinking else "default",
                )
            ],
            layout="desktop",
        ),
        {"tag": "markdown", "content": UI_TEXT["system_tool_settings_verbosity_label"]},
        _build_select_static(
            name="verbosity",
            placeholder=UI_TEXT["system_tool_settings_verbosity_placeholder"],
            options=_VERBOSITY_CHOICES,
            selected=verbosity,
            callback=payload_fn(action_ids.SETTINGS_SELECT_VERBOSITY),
        ),
        {"tag": "markdown", "content": UI_TEXT["system_tool_settings_agent_mode_label"]},
        _build_select_static(
            name="agent_mode",
            placeholder=UI_TEXT["system_tool_settings_agent_mode_placeholder"],
            options=_AGENT_MODE_CHOICES,
            selected=agent_mode,
            callback=payload_fn(action_ids.SETTINGS_SELECT_AGENT_MODE),
        ),
    ]
    return elements


class ToolCockpitBuilder:
    """Card JSON 2.0 builders for tool usage (/usage) and settings (/settings)."""

    @staticmethod
    def build_tool_usage_card(
        tool_name: str,
        usage_snapshot: Any,
        *,
        project_name: Optional[str] = None,
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build the interactive Card JSON 2.0 cockpit for ``/usage``."""
        tool, emoji, display = _tool_display_parts(tool_name)
        elements, header_template = _render_quota_elements(usage_snapshot)
        elements.append({"tag": "hr"})
        elements.append(_render_session_element(usage_snapshot))

        error_text = _get_field(usage_snapshot, "error", "quota_error")
        note_text = _get_field(usage_snapshot, "capability_note", "note", "quota_description")
        if error_text:
            elements.append(
                {
                    "tag": "markdown",
                    "text_size": "notation",
                    "content": f"⚠️ **配额探测提示**：{_escape_md(error_text, limit=220)}",
                }
            )
        if note_text:
            elements.append(
                {
                    "tag": "markdown",
                    "text_size": "notation",
                    "content": f"<font color='grey'>ℹ️ 配额说明：{_escape_md(note_text, limit=260)}</font>",
                }
            )

        def _usage_payload(action: str, **extra: Any) -> dict[str, Any]:
            data: dict[str, Any] = {"action": action, "tool_name": tool}
            if project_id is not None:
                data["project_id"] = project_id
            if thread_root_id:
                data["thread_root_id"] = thread_root_id
            data.update(extra)
            return data

        buttons = [
            build_callback_button(
                UI_TEXT["system_tool_usage_refresh_btn"],
                _usage_payload(action_ids.REFRESH_TOOL_USAGE),
            ),
            build_callback_button(
                UI_TEXT["system_tool_usage_settings_btn"],
                _usage_payload(action_ids.OPEN_TOOL_SETTINGS),
                button_type="primary",
            ),
            build_callback_button(
                UI_TEXT["system_tool_usage_compact_btn"],
                _usage_payload(action_ids.RUN_NATIVE_TOOL_COMMAND, command="/compact"),
            ),
            build_callback_button(
                UI_TEXT["system_tool_usage_cmds_btn"],
                _usage_payload(action_ids.SHOW_TOOL_COMMANDS),
            ),
        ]
        elements.append({"tag": "hr"})
        elements.extend(
            build_responsive_layout(buttons, layout="responsive", mobile_force_vertical=False)
        )

        header_title = UI_TEXT["system_tool_usage_title"].format(emoji=emoji, tool=display)
        subtitle = f"项目：{project_name}" if project_name else None
        card = _shrink_usage_card_if_needed(
            header_title=header_title,
            subtitle=subtitle,
            header_template=header_template,
            summary_content=f"{display} 用量与会话状态 · 点击查看详情",
            elements=elements,
        )
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_tool_settings_card(
        tool_name: str,
        *,
        models: Sequence[Any],
        current_model: Optional[str],
        available_tools: Sequence[str],
        settings_snapshot: Any = None,
        project_name: Optional[str] = None,
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
        status_banner: str = "",
        pending_group: Optional[str] = None,
        pending_profile: Optional[str] = None,
        pending_effort: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build the interactive Card JSON 2.0 cockpit for ``/settings``."""
        tool, emoji, display = _tool_display_parts(tool_name)
        tools_tuple = _normalize_available_tools(tool, available_tools)
        coerced_models = _coerce_model_options(models)
        state = resolve_model_cascade(
            coerced_models,
            current_model=current_model,
            selected_model=pending_group or _get_field(settings_snapshot, "pending_group"),
            selected_profile=pending_profile or _get_field(settings_snapshot, "pending_profile"),
            selected_effort=pending_effort or _get_field(settings_snapshot, "pending_effort"),
        )
        show_thinking, verbosity, agent_mode = _extract_agy_runtime_values(settings_snapshot)

        def _settings_payload(action: str, **updates: Any) -> dict[str, Any]:
            data: dict[str, Any] = {
                "action": action,
                "tool_name": tool,
                "project_id": project_id,
                "current_model": current_model,
                "model_group": state.selected_model or None,
                "model_profile": state.selected_profile,
                "model_effort": state.selected_effort,
            }
            if thread_root_id:
                data["thread_root_id"] = thread_root_id
            if tool == "agy":
                data["show_thinking"] = show_thinking
                data["verbosity"] = verbosity
                data["agent_mode"] = agent_mode
            data.update(updates)
            return data

        elements: list[dict[str, Any]] = []
        if status_banner:
            stripped = status_banner.lstrip()
            b_type = "success" if stripped.startswith("✅") else ("warning" if stripped.startswith("⚠️") else "info")
            elements.append(
                CoreBuilder._build_banner_element(_safe_one_line(status_banner, limit=140), type=b_type)
            )

        active_model = current_model or state.selection or UI_TEXT["system_tool_settings_default_model"]
        elements.append(
            {
                "tag": "markdown",
                "content": f"**当前配置** · {emoji} `{tool}` · 模型 `{_escape_md(active_model, limit=64)}`",
            }
        )
        elements.append({"tag": "hr"})

        tool_options = [
            (t, f"{t_emoji} {t_disp} ({t})" if t_disp.lower() != t else f"{t_emoji} {t_disp}")
            for t in tools_tuple
            for _, t_emoji, t_disp in (_tool_display_parts(t),)
        ]
        elements.append({"tag": "markdown", "content": UI_TEXT["system_tool_settings_tool_label"]})
        elements.append(
            _build_select_static(
                name="tool_select",
                placeholder=UI_TEXT["system_tool_settings_tool_placeholder"],
                options=tool_options,
                selected=tool,
                callback=_settings_payload(action_ids.SETTINGS_SELECT_TOOL),
            )
        )
        elements.extend(_render_model_cascade_elements(tool, state, _settings_payload))

        if tool == "agy":
            elements.extend(
                _render_agy_runtime_elements(
                    show_thinking=show_thinking,
                    verbosity=verbosity,
                    agent_mode=agent_mode,
                    payload_fn=_settings_payload,
                )
            )

        readonly_pairs = _extract_readonly_backend_summary(settings_snapshot)
        if readonly_pairs:
            cfg_bits = " · ".join(f"`{k}`={v}" for k, v in readonly_pairs)
            elements.append(
                {
                    "tag": "markdown",
                    "text_size": "notation",
                    "content": f"<font color='grey'>后端只读配置：{cfg_bits}</font>",
                }
            )

        footer_buttons = [
            build_callback_button(
                UI_TEXT["system_tool_settings_usage_btn"],
                _settings_payload(action_ids.OPEN_TOOL_USAGE),
            ),
            build_callback_button(
                UI_TEXT["system_tool_settings_cmds_btn"],
                _settings_payload(action_ids.SHOW_TOOL_COMMANDS),
            ),
            build_callback_button(
                UI_TEXT["system_tool_settings_compact_btn"],
                _settings_payload(action_ids.RUN_NATIVE_TOOL_COMMAND, command="/compact"),
            ),
            build_callback_button(
                UI_TEXT["system_tool_settings_clear_btn"],
                _settings_payload(action_ids.SETTINGS_CLEAR_SESSION),
            ),
            build_callback_button(
                UI_TEXT["system_tool_settings_refresh_btn"],
                _settings_payload(action_ids.REFRESH_TOOL_SETTINGS),
            ),
        ]
        elements.append({"tag": "hr"})
        elements.extend(
            build_responsive_layout(footer_buttons, layout="responsive", mobile_force_vertical=False)
        )

        header_title = UI_TEXT["system_tool_settings_title"].format(tool=display)
        subtitle = f"项目：{project_name}" if project_name else None
        card = _build_card_shell(
            title=header_title,
            subtitle=subtitle,
            template="blue",
            summary_content=f"{display} 工具与模型设置 · 即时生效",
            elements=elements,
        )
        return "interactive", json.dumps(card, ensure_ascii=False)


__all__ = ["ToolCockpitBuilder"]
