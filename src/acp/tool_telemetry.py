"""Live usage, quota, credits, and runtime settings telemetry for coding tools."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..utils.env import build_clean_env
from .native_commands import normalize_native_tool_key

logger = logging.getLogger(__name__)

_DEFAULT_USAGE_TIMEOUT_S = 5.0
_DEFAULT_SETTINGS_TIMEOUT_S = 4.0

_VALID_AGY_VERBOSITY = frozenset({"low", "medium", "high"})
_VALID_AGY_AGENT_MODES = frozenset({"auto", "accept-edits", "plan"})

# Allowlist of non-secret AGY /config keys safe for Feishu display.
_SAFE_AGY_CONFIG_KEYS: tuple[str, ...] = (
    "agentMode",
    "model",
    "orchestrator",
    "showThinking",
    "verbosity",
    "toolPermission",
    "artifactReviewPolicy",
    "enableTerminalSandbox",
    "disableSlashCommands",
    "allowNonWorkspaceAccess",
    "useG1Credits",
    "pickerGrouping",
    "queuedMessages",
)

_SENSITIVE_KEY_TOKENS = (
    "token",
    "secret",
    "password",
    "passwd",
    "auth",
    "credential",
    "apikey",
    "api_key",
    "cookie",
    "proxy",
    "workspace",
    "path",
    "dir",
    "home",
    "url",
    "gcp",
)
_SECRET_VALUE_RE = re.compile(
    r"(?:bearer\s+\S+|sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuotaBucketSnapshot:
    """One rate-limit quota window (for example 5-hour or weekly limit)."""

    bucket_id: str = ""
    name: str = ""
    window: str = ""
    remaining_fraction: float | None = None
    reset_time: str = ""
    description: str = ""


@dataclass(frozen=True)
class QuotaGroupSnapshot:
    """Model family group sharing a set of rate-limit quota buckets."""

    name: str
    description: str = ""
    buckets: tuple[QuotaBucketSnapshot, ...] = ()


@dataclass(frozen=True)
class ToolUsageSnapshot:
    """Normalized usage, quota, credit, and live session telemetry snapshot."""

    tool_name: str
    cwd: str = ""
    model_name: str | None = None
    session_active: bool = False
    session_id: str = ""
    message_count: int = 0
    created_at: float | None = None
    duration_seconds: float | None = None
    context_used: int = 0
    context_size: int = 0
    cost_amount: float | None = None
    cost_currency: str | None = None
    quota_groups: tuple[QuotaGroupSnapshot, ...] = ()
    quota_description: str = ""
    remaining_credits: float | int | None = None
    credits_known: bool = False
    quota_supported: bool = False
    capability_note: str = ""
    error: str = ""


@dataclass(frozen=True)
class ToolSettingsSnapshot:
    """Normalized runtime settings, ACP config options, and backend config summary."""

    tool_name: str
    cwd: str = ""
    model_name: str | None = None
    session_active: bool = False
    session_id: str = ""
    runtime_settings: dict[str, Any] = field(default_factory=dict)
    acp_config_options: tuple[dict[str, Any], ...] = ()
    backend_config_summary: dict[str, Any] = field(default_factory=dict)
    capability_note: str = ""
    error: str = ""


def _parse_bool_flag(value: object, *, default: bool) -> bool:
    """Parse boolean runtime setting strings and scalars with a fallback default."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _parse_float_or_none(value: object) -> float | None:
    """Convert numeric or string input to float when valid, else return None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _parse_int_or_zero(value: Any) -> int:
    """Convert non-negative integer inputs safely, returning 0 on parse failure."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    try:
        return max(0, int(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _parse_fraction(value: object) -> float | None:
    """Normalize a 0..1 fraction, percentage number, or ``XX%`` string into [0.0, 1.0]."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            num = _parse_float_or_none(stripped[:-1])
            if num is None:
                return None
            return max(0.0, min(1.0, num / 100.0))
    num = _parse_float_or_none(value)
    if num is None:
        return None
    if num > 1.0 and num <= 100.0:
        return max(0.0, min(1.0, num / 100.0))
    return max(0.0, min(1.0, num))


def _is_sensitive_key(key: str) -> bool:
    """Return True when a config key name indicates paths, credentials, or tokens."""
    lowered = str(key or "").strip().casefold()
    return any(tok in lowered for tok in _SENSITIVE_KEY_TOKENS)


def _looks_like_path_or_secret(text: str) -> bool:
    """Detect filesystem paths or bearer/API secret tokens in string config values."""
    s = text.strip()
    if not s:
        return False
    if s.startswith(("/", "~/")) or re.match(r"^[A-Za-z]:[/\\]", s):
        return True
    return bool(_SECRET_VALUE_RE.search(s))


def _sanitize_config_value(value: Any, *, depth: int = 0) -> Any:
    """Recursively redact sensitive keys, paths, and token-like values from config data."""
    if depth > 3:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _looks_like_path_or_secret(value):
            return None
        return value[:160]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key_str = str(k or "").strip()
            if not key_str or _is_sensitive_key(key_str):
                continue
            sanitized = _sanitize_config_value(v, depth=depth + 1)
            if sanitized is not None and sanitized != "" and sanitized != {} and sanitized != []:
                out[key_str] = sanitized
        return out or None
    if isinstance(value, (list, tuple)):
        items = []
        for item in list(value)[:10]:
            sanitized = _sanitize_config_value(item, depth=depth + 1)
            if sanitized is not None and sanitized != "":
                items.append(sanitized)
        return items or None
    return str(value)[:160]


def _extract_safe_agy_config_summary(
    raw_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Filter AGY ``/config`` fields to an allowlisted, sanitized summary dictionary."""
    if not isinstance(raw_config, Mapping):
        return {}
    summary: dict[str, Any] = {}
    for key in _SAFE_AGY_CONFIG_KEYS:
        if key not in raw_config:
            continue
        sanitized = _sanitize_config_value(raw_config.get(key))
        if sanitized is not None and sanitized != "":
            summary[key] = sanitized
    return summary


def _parse_agy_config_tsv(response_text: str) -> dict[str, Any]:
    """Fallback parser for tab-separated AGY ``/config`` text output."""
    if not response_text:
        return {}
    raw: dict[str, Any] = {}
    allowed = set(_SAFE_AGY_CONFIG_KEYS)
    for line in response_text.splitlines():
        if "\t" not in line:
            continue
        key, val = line.split("\t", 1)
        key = key.strip()
        if key not in allowed:
            continue
        val_str = val.strip()
        if not val_str:
            continue
        if val_str in {"true", "false"}:
            raw[key] = val_str == "true"
        elif val_str.startswith("{"):
            try:
                raw[key] = json.loads(val_str)
            except (json.JSONDecodeError, TypeError):
                raw[key] = val_str
        else:
            raw[key] = val_str
    return _extract_safe_agy_config_summary(raw)


def _run_agy_slash_probe(
    prompt_cmd: str,
    *,
    cwd: str,
    timeout_s: float,
) -> tuple[dict[str, Any] | None, str, str]:
    """Run one bounded headless AGY CLI slash command and extract structured command data."""
    try:
        bounded_timeout = float(timeout_s)
    except (TypeError, ValueError):
        bounded_timeout = 0.0
    if bounded_timeout <= 0:
        return None, "", "probe timeout budget exhausted"

    agy_bin = shutil.which("agy")
    if not agy_bin:
        return None, "", "未检测到 agy CLI 可执行文件"

    safe_cwd = cwd if cwd and os.path.isdir(cwd) else None
    args = [
        agy_bin,
        f"--prompt={prompt_cmd}",
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
    ]
    try:
        proc = subprocess.run(
            args,
            cwd=safe_cwd,
            env=build_clean_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=bounded_timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "", f"agy {prompt_cmd} 探测超时 ({bounded_timeout:.1f}s)"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return None, "", str(exc) or type(exc).__name__

    command_data: dict[str, Any] | None = None
    response_text = ""
    error_text = ""

    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event") or event.get("type") or "").strip().lower()
        if event_type == "command_result":
            cmd_obj = event.get("command")
            if isinstance(cmd_obj, dict) and isinstance(cmd_obj.get("data"), dict):
                command_data = dict(cmd_obj["data"])
        elif event_type == "result":
            res_obj = event.get("result") if isinstance(event.get("result"), dict) else event
            if isinstance(res_obj, dict):
                resp = res_obj.get("response")
                if isinstance(resp, str) and resp:
                    response_text = resp
                cmd_obj = res_obj.get("command")
                if (
                    command_data is None
                    and isinstance(cmd_obj, dict)
                    and isinstance(cmd_obj.get("data"), dict)
                ):
                    command_data = dict(cmd_obj["data"])
                raw_err = res_obj.get("error")
                if isinstance(raw_err, str) and raw_err.strip():
                    error_text = raw_err.strip()
                elif isinstance(raw_err, dict) and isinstance(raw_err.get("message"), str):
                    error_text = raw_err["message"].strip()
        elif event_type == "error":
            raw_err = event.get("error")
            if isinstance(raw_err, str) and raw_err.strip():
                error_text = raw_err.strip()
            elif isinstance(raw_err, dict) and isinstance(raw_err.get("message"), str):
                error_text = raw_err["message"].strip()
            elif isinstance(event.get("message"), str) and event["message"].strip():
                error_text = event["message"].strip()

    if not error_text and proc.returncode not in (0, None):
        stderr = (proc.stderr or "").strip()
        error_text = stderr or f"agy {prompt_cmd} 退出码 {proc.returncode}"

    return command_data, response_text, error_text


def _parse_agy_usage_groups(
    command_data: Mapping[str, Any] | None,
    response_text: str,
) -> tuple[tuple[QuotaGroupSnapshot, ...], str]:
    """Parse AGY ``/usage`` command payloads or fallback TSV lines into quota groups."""
    groups: list[QuotaGroupSnapshot] = []
    description = ""
    if isinstance(command_data, Mapping):
        description = str(command_data.get("description") or "").strip()
        raw_groups = command_data.get("groups")
        if isinstance(raw_groups, Sequence) and not isinstance(raw_groups, (str, bytes)):
            for raw_group in raw_groups:
                if not isinstance(raw_group, Mapping):
                    continue
                group_name = str(raw_group.get("name") or "").strip()
                if not group_name:
                    continue
                group_desc = str(raw_group.get("description") or "").strip()
                buckets: list[QuotaBucketSnapshot] = []
                raw_buckets = raw_group.get("buckets")
                if isinstance(raw_buckets, Sequence) and not isinstance(raw_buckets, (str, bytes)):
                    for raw_bucket in raw_buckets:
                        if not isinstance(raw_bucket, Mapping):
                            continue
                        buckets.append(
                            QuotaBucketSnapshot(
                                bucket_id=str(
                                    raw_bucket.get("id") or raw_bucket.get("bucket_id") or ""
                                ).strip(),
                                name=str(raw_bucket.get("name") or "").strip(),
                                window=str(raw_bucket.get("window") or "").strip(),
                                remaining_fraction=_parse_fraction(
                                    raw_bucket.get("remaining_fraction")
                                ),
                                reset_time=str(raw_bucket.get("reset_time") or "").strip(),
                                description=str(raw_bucket.get("description") or "").strip(),
                            )
                        )
                groups.append(
                    QuotaGroupSnapshot(
                        name=group_name,
                        description=group_desc,
                        buckets=tuple(buckets),
                    )
                )

    if not groups and response_text:
        grouped_buckets: dict[str, list[QuotaBucketSnapshot]] = {}
        for line in response_text.splitlines():
            parts = [part.strip() for part in line.split("\t")]
            if len(parts) < 3:
                continue
            g_name, b_name, pct_text = parts[0], parts[1], parts[2]
            reset = parts[3] if len(parts) > 3 else ""
            if not g_name or not b_name:
                continue
            lowered_b = b_name.casefold()
            window = "weekly" if "weekly" in lowered_b else ("5h" if "5-hour" in lowered_b or "five hour" in lowered_b else "")
            grouped_buckets.setdefault(g_name, []).append(
                QuotaBucketSnapshot(
                    bucket_id="",
                    name=b_name,
                    window=window,
                    remaining_fraction=_parse_fraction(pct_text),
                    reset_time=reset,
                )
            )
        for g_name, b_list in grouped_buckets.items():
            groups.append(QuotaGroupSnapshot(name=g_name, buckets=tuple(b_list)))

    return tuple(groups), description


def _parse_agy_credits(
    command_data: Mapping[str, Any] | None,
    response_text: str,
) -> tuple[float | int | None, bool]:
    """Parse remaining G1 credits from AGY ``/credits`` JSON or fallback TSV text."""
    if isinstance(command_data, Mapping) and "remaining_credits" in command_data:
        raw_val = command_data.get("remaining_credits")
        if not isinstance(raw_val, bool):
            if isinstance(raw_val, int):
                return raw_val, True
            if isinstance(raw_val, float):
                return (int(raw_val) if raw_val.is_integer() else raw_val), True
            parsed = _parse_float_or_none(raw_val)
            if parsed is not None:
                return (int(parsed) if parsed.is_integer() else parsed), True
    for line in (response_text or "").splitlines():
        if "\t" in line:
            label, val = line.split("\t", 1)
            if "credit" in label.casefold():
                parsed = _parse_float_or_none(val)
                if parsed is not None:
                    return (int(parsed) if parsed.is_integer() else parsed), True
    return None, False


def _read_session_usage_metrics(
    session: object | None,
    explicit_model_name: str | None,
) -> dict[str, Any]:
    """Extract local session counters, model name, and ACP usage metrics if active."""
    if session is None:
        return {
            "session_active": False,
            "session_id": "",
            "message_count": 0,
            "created_at": None,
            "duration_seconds": None,
            "model_name": explicit_model_name,
            "context_used": 0,
            "context_size": 0,
            "cost_amount": None,
            "cost_currency": None,
        }

    session_id = str(getattr(session, "session_id", "") or "").strip()
    message_count = _parse_int_or_zero(getattr(session, "message_count", 0))
    created_at = _parse_float_or_none(getattr(session, "created_at", None))
    duration_seconds: float | None = None
    if created_at is not None and created_at > 0:
        duration_seconds = max(0.0, time.time() - created_at)

    raw_model = getattr(session, "_model_name", None)
    session_model = str(raw_model).strip() if isinstance(raw_model, str) and raw_model.strip() else None
    resolved_model = explicit_model_name or session_model

    context_used = 0
    context_size = 0
    cost_amount: float | None = None
    cost_currency: str | None = None
    getter = getattr(session, "get_usage_info", None)
    if callable(getter):
        try:
            usage = getter()
        except Exception:
            usage = None
        if usage is not None:
            context_used = _parse_int_or_zero(getattr(usage, "used", 0))
            context_size = _parse_int_or_zero(getattr(usage, "size", 0))
            cost_amount = _parse_float_or_none(getattr(usage, "cost_amount", None))
            raw_cur = getattr(usage, "cost_currency", None)
            cost_currency = str(raw_cur).strip() if isinstance(raw_cur, str) and raw_cur.strip() else None

    return {
        "session_active": True,
        "session_id": session_id,
        "message_count": message_count,
        "created_at": created_at,
        "duration_seconds": duration_seconds,
        "model_name": resolved_model,
        "context_used": context_used,
        "context_size": context_size,
        "cost_amount": cost_amount,
        "cost_currency": cost_currency,
    }


def _normalize_runtime_settings(
    tool_key: str,
    raw_settings: Mapping[str, Any] | None,
    *,
    session_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge and validate persisted and live runtime settings for the target tool."""
    merged: dict[str, Any] = {}
    if tool_key == "agy":
        merged["show_thinking"] = True
        merged["verbosity"] = "medium"
        merged["agent_mode"] = "auto"

    for source in (session_settings, raw_settings):
        if not isinstance(source, Mapping):
            continue
        if "show_thinking" in source or "showThinking" in source:
            raw_st = source["show_thinking"] if "show_thinking" in source else source["showThinking"]
            merged["show_thinking"] = _parse_bool_flag(raw_st, default=bool(merged.get("show_thinking", True)))
        if "verbosity" in source:
            v = str(source.get("verbosity") or "").strip().lower()
            if v in _VALID_AGY_VERBOSITY:
                merged["verbosity"] = v
        if "agent_mode" in source or "agentMode" in source:
            raw_mode = source["agent_mode"] if "agent_mode" in source else source["agentMode"]
            m = str(raw_mode or "").strip().lower()
            if m in _VALID_AGY_AGENT_MODES:
                merged["agent_mode"] = m
    return merged


def _extract_acp_config_options(session: object | None) -> tuple[dict[str, Any], ...]:
    """Read raw ACP config options from a live session or underlying ACP client."""
    if session is None:
        return ()

    candidates: list[object] = []
    for owner in (session, getattr(session, "_acp_session", None)):
        if owner is None:
            continue
        getter = getattr(owner, "get_config_options", None)
        if callable(getter):
            try:
                opts = getter()
                if opts:
                    candidates.append(opts)
            except Exception:
                pass
        for attr in ("config_options", "_config_options"):
            val = getattr(owner, attr, None)
            if val:
                candidates.append(val)

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_list in candidates:
        if isinstance(raw_list, (str, bytes, Mapping)) or not isinstance(raw_list, Sequence):
            continue
        for item in raw_list:
            root = getattr(item, "root", item)
            if isinstance(root, Mapping):
                opt_id = str(root.get("id") or "").strip()
                opt_name = str(root.get("name") or opt_id).strip()
                category = str(root.get("category") or "").strip()
                current_val = root.get("current_value") if "current_value" in root else root.get("currentValue")
                raw_choices = root.get("options") or ()
            else:
                opt_id = str(getattr(root, "id", "") or "").strip()
                opt_name = str(getattr(root, "name", "") or opt_id).strip()
                category = str(getattr(root, "category", "") or "").strip()
                current_val = getattr(root, "current_value", None)
                raw_choices = getattr(root, "options", None) or ()
            if not opt_id or opt_id in seen_ids or _is_sensitive_key(opt_id):
                continue
            seen_ids.add(opt_id)
            choices: list[str] = []
            if isinstance(raw_choices, Sequence) and not isinstance(raw_choices, (str, bytes)):
                for ch in raw_choices:
                    nested = getattr(ch, "options", None)
                    sub_items = nested if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)) else (ch,)
                    for sub in sub_items:
                        if isinstance(sub, Mapping):
                            val = str(sub.get("value") or sub.get("name") or "").strip()
                        else:
                            val = str(getattr(sub, "value", None) or getattr(sub, "name", "") or "").strip()
                        if val and val not in choices:
                            choices.append(val)
            entry: dict[str, Any] = {
                "id": opt_id,
                "name": opt_name or opt_id,
                "current_value": "" if current_val is None else str(current_val),
            }
            if category:
                entry["category"] = category
            if choices:
                entry["options"] = choices[:24]
            normalized.append(entry)
    return tuple(normalized)


def collect_tool_usage_snapshot(
    tool_name: str,
    *,
    cwd: str,
    model_name: str | None = None,
    session: object | None = None,
    timeout_s: float = _DEFAULT_USAGE_TIMEOUT_S,
) -> ToolUsageSnapshot:
    """Collect live quota, credit, and session usage metrics for *tool_name* without raising."""
    tool_key = normalize_native_tool_key(tool_name) or "tool"
    try:
        metrics = _read_session_usage_metrics(session, model_name)
    except Exception as exc:
        logger.debug("tool_telemetry session metrics read failed: %s", exc, exc_info=True)
        metrics = {
            "session_active": session is not None,
            "session_id": "",
            "message_count": 0,
            "created_at": None,
            "duration_seconds": None,
            "model_name": model_name,
            "context_used": 0,
            "context_size": 0,
            "cost_amount": None,
            "cost_currency": None,
        }

    if tool_key != "agy":
        note = (
            f"{tool_key} 后端当前未暴露远程配额查询接口；已显示当前会话的上下文窗口与费用遥测。"
            if metrics["session_active"]
            else f"{tool_key} 后端当前未暴露远程配额查询接口；开启编程会话后可实时显示上下文占用与费用遥测。"
        )
        return ToolUsageSnapshot(
            tool_name=tool_key,
            cwd=str(cwd or ""),
            model_name=metrics["model_name"],
            session_active=metrics["session_active"],
            session_id=metrics["session_id"],
            message_count=metrics["message_count"],
            created_at=metrics["created_at"],
            duration_seconds=metrics["duration_seconds"],
            context_used=metrics["context_used"],
            context_size=metrics["context_size"],
            cost_amount=metrics["cost_amount"],
            cost_currency=metrics["cost_currency"],
            quota_supported=False,
            capability_note=note,
        )

    if not shutil.which("agy"):
        return ToolUsageSnapshot(
            tool_name=tool_key,
            cwd=str(cwd or ""),
            model_name=metrics["model_name"],
            session_active=metrics["session_active"],
            session_id=metrics["session_id"],
            message_count=metrics["message_count"],
            created_at=metrics["created_at"],
            duration_seconds=metrics["duration_seconds"],
            context_used=metrics["context_used"],
            context_size=metrics["context_size"],
            cost_amount=metrics["cost_amount"],
            cost_currency=metrics["cost_currency"],
            quota_supported=False,
            capability_note="宿主机未检测到 `agy` CLI，无法查询实时配额与 Credits。",
            error="未检测到 agy CLI 可执行文件",
        )

    usage_data: dict[str, Any] | None = None
    usage_resp = ""
    usage_err = ""
    credits_data: dict[str, Any] | None = None
    credits_resp = ""
    credits_err = ""

    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="agy-usage") as pool:
            usage_future = pool.submit(_run_agy_slash_probe, "/usage", cwd=cwd, timeout_s=timeout_s)
            credits_future = pool.submit(_run_agy_slash_probe, "/credits", cwd=cwd, timeout_s=timeout_s)
            usage_data, usage_resp, usage_err = usage_future.result()
            credits_data, credits_resp, credits_err = credits_future.result()
    except Exception as exc:
        usage_err = str(exc) or type(exc).__name__

    quota_groups, quota_description = _parse_agy_usage_groups(usage_data, usage_resp)
    remaining_credits, credits_known = _parse_agy_credits(credits_data, credits_resp)
    errors = [err for err in (usage_err, credits_err) if err]
    combined_error = "; ".join(dict.fromkeys(errors))
    if not quota_groups and not credits_known and not combined_error:
        combined_error = "未获取到 AGY 配额数据"

    return ToolUsageSnapshot(
        tool_name=tool_key,
        cwd=str(cwd or ""),
        model_name=metrics["model_name"],
        session_active=metrics["session_active"],
        session_id=metrics["session_id"],
        message_count=metrics["message_count"],
        created_at=metrics["created_at"],
        duration_seconds=metrics["duration_seconds"],
        context_used=metrics["context_used"],
        context_size=metrics["context_size"],
        cost_amount=metrics["cost_amount"],
        cost_currency=metrics["cost_currency"],
        quota_groups=quota_groups,
        quota_description=quota_description,
        remaining_credits=remaining_credits,
        credits_known=credits_known,
        quota_supported=bool(quota_groups or credits_known),
        capability_note="" if (quota_groups or credits_known) else "暂未获取到实时配额，可稍后刷新重试。",
        error=combined_error,
    )


def collect_tool_settings_snapshot(
    tool_name: str,
    *,
    cwd: str,
    model_name: str | None = None,
    session: object | None = None,
    saved_runtime_settings: Mapping[str, Any] | None = None,
    timeout_s: float = _DEFAULT_SETTINGS_TIMEOUT_S,
) -> ToolSettingsSnapshot:
    """Collect runtime settings, ACP config options, and safe backend config summary."""
    tool_key = normalize_native_tool_key(tool_name) or "tool"
    session_active = session is not None
    session_id = str(getattr(session, "session_id", "") or "").strip() if session is not None else ""
    raw_session_model = getattr(session, "_model_name", None) if session is not None else None
    session_model = (
        str(raw_session_model).strip()
        if isinstance(raw_session_model, str) and raw_session_model.strip()
        else None
    )
    resolved_model = model_name or session_model

    session_runtime: Mapping[str, Any] | None = None
    if session is not None:
        getter = getattr(session, "get_runtime_settings", None)
        if callable(getter):
            try:
                candidate = getter()
                if isinstance(candidate, Mapping):
                    session_runtime = candidate
            except Exception:
                session_runtime = None

    runtime_settings = _normalize_runtime_settings(
        tool_key,
        saved_runtime_settings,
        session_settings=session_runtime,
    )
    acp_config_options = _extract_acp_config_options(session)

    backend_config_summary: dict[str, Any] = {}
    probe_error = ""
    capability_note = ""

    if tool_key == "agy":
        if shutil.which("agy"):
            cfg_data, cfg_resp, cfg_err = _run_agy_slash_probe(
                "/config",
                cwd=cwd,
                timeout_s=timeout_s,
            )
            if isinstance(cfg_data, Mapping) and isinstance(cfg_data.get("config"), Mapping):
                backend_config_summary = _extract_safe_agy_config_summary(cfg_data.get("config"))
            elif isinstance(cfg_data, Mapping):
                backend_config_summary = _extract_safe_agy_config_summary(cfg_data)
            if not backend_config_summary and cfg_resp:
                backend_config_summary = _parse_agy_config_tsv(cfg_resp)
            if cfg_err:
                probe_error = cfg_err
        else:
            capability_note = "宿主机未检测到 `agy` CLI，仅展示 GhostAP 已保存运行参数。"
    else:
        capability_note = (
            f"{tool_key} 运行设置由 GhostAP 模型与会话控制面托管。"
            if not acp_config_options
            else f"{tool_key} 已同步 ACP 运行时配置项。"
        )

    return ToolSettingsSnapshot(
        tool_name=tool_key,
        cwd=str(cwd or ""),
        model_name=resolved_model,
        session_active=session_active,
        session_id=session_id,
        runtime_settings=runtime_settings,
        acp_config_options=acp_config_options,
        backend_config_summary=backend_config_summary,
        capability_note=capability_note,
        error=probe_error,
    )
