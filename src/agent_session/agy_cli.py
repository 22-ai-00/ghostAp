"""AGY CLI stream-json session backend for Orchestrator custom Planner/Executor effort."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Optional

from ..acp.agy_selection import (
    format_agy_model_effort_label,
    get_agy_orchestrator_spec,
    parse_agy_model_selection,
    requires_agy_cli_orchestrator_session,
    resolve_agy_orchestrator_settings,
)
from ..acp.execution_instructions import AUTONOMOUS_EXECUTION_PROMPT, append_execution_instructions
from ..acp.models import (
    ACPEvent,
    ACPEventType,
    ACPUsageInfo,
    PromptResult,
    TerminalOutputProvenance,
    ToolCallInfo,
)
from .cli_base import _BaseCLISession, _CLIModuleHooks
from .process_cleanup import terminate_and_reap_process_tree

logger = logging.getLogger(__name__)

_CLI_TERMINATE_GRACE_S = 5.0
_CLI_KILL_GRACE_S = 3.0

_VALID_VERBOSITY_VALUES = frozenset({"low", "medium", "high"})
_VALID_AGENT_MODES = frozenset({"auto", "accept-edits", "plan"})
_SAFE_CONFIG_KEYS = (
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
)


def _terminate_and_reap_process(
    proc: subprocess.Popen,
    *,
    process_group_id: int | None = None,
) -> bool:
    """Bounded TERM→KILL cleanup for one AGY CLI subprocess tree."""
    return terminate_and_reap_process_tree(
        proc,
        process_group_id=process_group_id,
        terminate_grace=_CLI_TERMINATE_GRACE_S,
        kill_grace=_CLI_KILL_GRACE_S,
        label="AGYCLI",
    )


def _infer_tool_kind(tool_name: str) -> str:
    lowered = str(tool_name or "").strip().lower()
    if any(tok in lowered for tok in ("read", "view", "cat", "list", "ls")):
        return "read"
    if any(tok in lowered for tok in ("edit", "write", "replace", "patch", "modify")):
        return "edit"
    if any(tok in lowered for tok in ("delete", "remove", "rm")):
        return "delete"
    if any(tok in lowered for tok in ("grep", "search", "find", "glob")):
        return "search"
    if any(tok in lowered for tok in ("fetch", "web", "url", "http")):
        return "fetch"
    if any(tok in lowered for tok in ("run", "command", "shell", "bash", "exec")):
        return "execute"
    if "think" in lowered:
        return "think"
    return "other"


def _parse_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_bool_setting(value: object, *, default: bool) -> bool:
    """Parse boolean runtime setting values cleanly with a fallback default."""
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


def _build_usage_info(raw: object) -> ACPUsageInfo | None:
    if not isinstance(raw, dict):
        return None
    input_tokens = _parse_int_or_none(raw.get("input_tokens")) or 0
    output_tokens = _parse_int_or_none(raw.get("output_tokens")) or 0
    thinking_tokens = _parse_int_or_none(
        raw.get("thinking_tokens") if "thinking_tokens" in raw else raw.get("thought_tokens")
    ) or 0
    total_tokens = _parse_int_or_none(raw.get("total_tokens"))
    used = total_tokens if total_tokens is not None else (input_tokens + output_tokens + thinking_tokens)
    if used <= 0 and not any(k in raw for k in ("input_tokens", "output_tokens", "total_tokens")):
        return None
    return ACPUsageInfo(used=used, size=0)


def _format_reset_time(reset_time: str, remaining_fraction: float | None) -> str:
    """Format AGY quota reset timestamps, collapsing full-quota epoch stamps."""
    raw = str(reset_time or "").strip()
    if raw.startswith("1970-01-01") and (remaining_fraction is None or remaining_fraction >= 0.999):
        return "未消耗（满额）"
    return raw or "未知"


def _format_quota_bar(fraction: float | None) -> str:
    """Render a 10-block Unicode progress bar and percentage for remaining quota."""
    if fraction is None:
        return "░░░░░░░░░░ --%"
    clamped = max(0.0, min(1.0, float(fraction)))
    filled = max(0, min(10, round(clamped * 10)))
    pct = round(clamped * 100)
    return f"{'█' * filled}{'░' * (10 - filled)} {pct}%"


def _truncate_text(text: str, limit: int = 80) -> str:
    """Normalize whitespace and truncate long description text with an ellipsis."""
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _format_agy_command_result(
    command_name: str,
    data: object,
    *,
    fallback_response: str = "",
) -> str:
    """Format AGY CLI ``command_result`` JSON payloads into clean Feishu-friendly Markdown."""
    name = str(command_name or "").strip().casefold()
    payload = data if isinstance(data, Mapping) else {}

    if name in {"usage", "quota"}:
        lines = ["📊 **AGY 模型配额与用量**"]
        desc = str(payload.get("description") or "").strip()
        groups = payload.get("groups")
        if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
            for grp in groups:
                if not isinstance(grp, Mapping):
                    continue
                g_name = str(grp.get("name") or "").strip()
                if g_name:
                    lines.append(f"\n**{g_name}**")
                buckets = grp.get("buckets")
                if isinstance(buckets, Sequence) and not isinstance(buckets, (str, bytes)):
                    for bucket in buckets:
                        if not isinstance(bucket, Mapping):
                            continue
                        b_name = str(bucket.get("name") or bucket.get("window") or "Quota").strip()
                        raw_frac = bucket.get("remaining_fraction")
                        frac = float(raw_frac) if isinstance(raw_frac, (int, float)) and not isinstance(raw_frac, bool) else None
                        reset = _format_reset_time(str(bucket.get("reset_time") or ""), frac)
                        lines.append(f"- {b_name}: `{_format_quota_bar(frac)}` · 重置 `{reset}`")
        if len(lines) == 1 and fallback_response.strip():
            lines.append("")
            for row in fallback_response.strip().splitlines():
                cols = [c.strip() for c in row.split("\t") if c.strip()]
                if cols:
                    lines.append(f"- {' | '.join(cols)}")
        elif desc and len(lines) > 1:
            lines.append(f"\n> {_truncate_text(desc, 160)}")
        return "\n".join(lines).strip()

    if name == "credits":
        remaining = payload.get("remaining_credits")
        if remaining is not None and not isinstance(remaining, bool):
            return f"💳 **G1 Credits**\n\n- 剩余额度: `{remaining}`"
        if fallback_response.strip():
            return f"💳 **G1 Credits**\n\n- {fallback_response.strip().replace(chr(9), ': ')}"
        return "💳 **G1 Credits**\n\n- 暂无额度信息。"

    if name in {"config", "settings"}:
        cfg = payload.get("config") if isinstance(payload.get("config"), Mapping) else payload
        lines = ["⚙️ **AGY 当前配置摘要**"]
        if isinstance(cfg, Mapping):
            for key in _SAFE_CONFIG_KEYS:
                if key not in cfg:
                    continue
                val = cfg.get(key)
                if val is None or val == "":
                    continue
                if isinstance(val, Mapping):
                    sub_parts = [f"{k}={v}" for k, v in val.items() if v is not None and v != ""]
                    if sub_parts:
                        lines.append(f"- `{key}`: {', '.join(sub_parts)}")
                else:
                    lines.append(f"- `{key}`: `{val}`")
        if len(lines) == 1:
            lines.append("\n当前无额外可展示配置。")
        return "\n".join(lines)

    if name == "hooks":
        hooks = payload.get("hooks")
        if isinstance(hooks, Sequence) and not isinstance(hooks, (str, bytes)) and len(hooks) > 0:
            lines = [f"🪝 **AGY Hooks ({len(hooks)})**"]
            for item in hooks[:20]:
                if isinstance(item, Mapping):
                    label = str(item.get("event") or item.get("name") or item.get("matcher") or "hook").strip()
                    detail = str(item.get("command") or item.get("description") or "").strip()
                    lines.append(f"- `{label}`" + (f": {_truncate_text(detail, 72)}" if detail else ""))
                else:
                    lines.append(f"- `{item}`")
            return "\n".join(lines)
        return "🪝 **AGY Hooks**\n\n当前未配置任何 Hook。"

    if name == "permissions":
        perms = payload.get("permissions")
        if isinstance(perms, Sequence) and not isinstance(perms, (str, bytes)) and len(perms) > 0:
            lines = [f"🔐 **AGY 权限配置 ({len(perms)})**"]
            for item in perms[:20]:
                if isinstance(item, Mapping):
                    scope = str(item.get("scope") or "default").strip()
                    extras = [
                        f"{k}={v}"
                        for k, v in item.items()
                        if k != "scope" and v not in (None, "", [], {})
                    ]
                    line = f"- 作用域 `{scope}`"
                    if extras:
                        line += f" ({', '.join(extras[:4])})"
                    lines.append(line)
                else:
                    lines.append(f"- `{item}`")
            return "\n".join(lines)
        return "🔐 **AGY 权限配置**\n\n当前未配置额外权限规则。"

    if name in {"agent", "agents"}:
        agents = payload.get("agents")
        if isinstance(agents, Sequence) and not isinstance(agents, (str, bytes)) and len(agents) > 0:
            lines = [f"🤖 **AGY 可用 Agents ({len(agents)})**"]
            for item in agents[:30]:
                if isinstance(item, Mapping):
                    a_name = str(item.get("name") or item.get("id") or "").strip()
                    a_desc = str(item.get("description") or "").strip()
                    if a_name:
                        lines.append(f"- `{a_name}`" + (f" — {_truncate_text(a_desc, 72)}" if a_desc else ""))
                elif str(item or "").strip():
                    lines.append(f"- `{str(item).strip()}`")
            if len(agents) > 30:
                lines.append(f"- …及另外 {len(agents) - 30} 个 Agents")
            return "\n".join(lines)
        if fallback_response.strip():
            items = [ln.strip() for ln in fallback_response.splitlines() if ln.strip()]
            return "\n".join([f"🤖 **AGY 可用 Agents ({len(items)})**", *(f"- `{x}`" for x in items[:30])])
        return "🤖 **AGY 可用 Agents**\n\n当前未发现自定义 Agents。"

    if name == "skills":
        skills = payload.get("skills")
        if isinstance(skills, Sequence) and not isinstance(skills, (str, bytes)) and len(skills) > 0:
            lines = [f"🧩 **AGY 可用 Skills ({len(skills)})**"]
            for item in skills[:25]:
                if isinstance(item, Mapping):
                    s_name = str(item.get("name") or "").strip()
                    s_desc = str(item.get("description") or "").strip()
                    tag = " [内置]" if item.get("builtin") else ""
                    if s_name:
                        lines.append(
                            f"- `{s_name}`{tag}"
                            + (f" — {_truncate_text(s_desc, 68)}" if s_desc else "")
                        )
                elif str(item or "").strip():
                    lines.append(f"- `{str(item).strip()}`")
            if len(skills) > 25:
                lines.append(f"- …及另外 {len(skills) - 25} 个 Skills")
            return "\n".join(lines)
        if fallback_response.strip():
            lines = ["🧩 **AGY 可用 Skills**"]
            for row in fallback_response.strip().splitlines()[:25]:
                cols = [c.strip() for c in row.split("\t") if c.strip()]
                if cols:
                    lines.append(f"- `{cols[0]}`" + (f" — {_truncate_text(cols[1], 68)}" if len(cols) > 1 else ""))
            return "\n".join(lines)
        return "🧩 **AGY 可用 Skills**\n\n当前未加载任何 Skills。"

    if name == "help":
        cmds = payload.get("commands")
        if isinstance(cmds, Sequence) and not isinstance(cmds, (str, bytes)) and len(cmds) > 0:
            lines = [f"📖 **AGY 原生命令 ({len(cmds)})**"]
            for item in cmds[:30]:
                if not isinstance(item, Mapping):
                    continue
                c_name = str(item.get("name") or "").strip()
                if not c_name:
                    continue
                aliases = item.get("aliases")
                alias_str = (
                    f" ({', '.join('/' + str(a) for a in aliases if str(a).strip())})"
                    if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)) and aliases
                    else ""
                )
                c_desc = str(item.get("description") or "").strip()
                lines.append(f"- `/{c_name}`{alias_str}" + (f" — {_truncate_text(c_desc, 72)}" if c_desc else ""))
            return "\n".join(lines)
        if fallback_response.strip():
            lines = ["📖 **AGY 原生命令**"]
            for row in fallback_response.strip().splitlines()[:30]:
                cols = [c.strip() for c in row.split("\t") if c.strip()]
                if cols:
                    lines.append(f"- `{cols[0]}`" + (f" — {_truncate_text(cols[1], 72)}" if len(cols) > 1 else ""))
            return "\n".join(lines)
        return "📖 **AGY 原生命令**\n\n暂无命令列表。"

    if name == "model":
        m_id = str(payload.get("id") or payload.get("model") or "").strip()
        m_label = str(payload.get("label") or payload.get("name") or "").strip()
        is_default = payload.get("is_default")
        if m_id or m_label:
            display = f"{m_label} (`{m_id}`)" if (m_label and m_id and m_label != m_id) else f"`{m_label or m_id}`"
            suffix = "（默认）" if is_default is True else ""
            return f"🧠 **AGY 当前模型**\n\n- 模型: {display}{suffix}"
        if fallback_response.strip():
            return f"🧠 **AGY 当前模型**\n\n- {fallback_response.strip().replace(chr(9), ' · ')}"
        return "🧠 **AGY 当前模型**\n\n暂无模型状态信息。"

    if name == "effort":
        adjustable = payload.get("adjustable")
        effort = str(payload.get("effort") or payload.get("thought_level") or "").strip()
        levels = payload.get("levels")
        lines = ["⚡ **AGY 思考强度 (Effort)**"]
        if effort:
            lines.append(f"- 当前档位: `{effort}`")
        if adjustable is False:
            lines.append("- 状态: 当前模型不支持独立调节 (`not adjustable`)")
        elif adjustable is True:
            lines.append("- 状态: 支持调节")
        if isinstance(levels, Sequence) and not isinstance(levels, (str, bytes)) and levels:
            lines.append(f"- 可选档位: {', '.join(f'`{x}`' for x in levels)}")
        if len(lines) == 1 and fallback_response.strip():
            lines.append(f"- {fallback_response.strip()}")
        return "\n".join(lines)

    if name in {"plugin", "plugins"}:
        plugins = payload.get("plugins")
        if isinstance(plugins, Sequence) and not isinstance(plugins, (str, bytes)) and len(plugins) > 0:
            lines = [f"🔌 **AGY 插件 ({len(plugins)})**"]
            for item in plugins[:25]:
                if not isinstance(item, Mapping):
                    lines.append(f"- `{item}`")
                    continue
                p_name = str(item.get("name") or "").strip()
                enabled = item.get("enabled")
                status_icon = "✅ 已启用" if enabled is True else ("⏸️ 已禁用" if enabled is False else "")
                p_desc = str(item.get("description") or "").strip()
                line = f"- `{p_name}`"
                if status_icon:
                    line += f" ({status_icon})"
                if p_desc:
                    line += f" — {_truncate_text(p_desc, 72)}"
                lines.append(line)
            return "\n".join(lines)
        return "🔌 **AGY 插件**\n\n当前未安装插件。"

    if name == "changelog":
        changelog = str(payload.get("changelog") or fallback_response or "").strip()
        if not changelog:
            return "📝 **AGY 更新日志**\n\n暂无更新日志内容。"
        if len(changelog) > 2200:
            return changelog[:2200].rstrip() + "\n\n…（内容较长，已截断展示最新变更摘要）"
        return changelog

    if payload:
        try:
            rendered_json = json.dumps(payload, ensure_ascii=False, indent=2)
            return f"```json\n{rendered_json[:2000]}\n```"
        except TypeError:
            pass
    if fallback_response.strip():
        return fallback_response.strip()
    return ""


@dataclass
class AGYCLIConfig:
    """Configuration knobs for AGY CLI headless stream-json execution."""

    command: str = "agy"
    show_thinking: bool = True
    verbosity: Optional[str] = None
    agent_mode: Optional[str] = None


class SyncAGYCLISession(_BaseCLISession):
    """AGY CLI session supporting custom Orchestrator Planner/Executor effort.

    ``agy acp`` 1.5.14 exposes native ``thought_level`` only on single models and
    rejects Orchestrator thinking overrides / ``settings.json["orchestrator"]``.
    Headless ``agy --prompt=... --output-format stream-json`` with
    ``--settings '{"orchestrator": {"planner": ..., "executor": ...}}'`` applies
    both Planner and Executor effort cleanly and preserves conversation context
    across turns via ``--conversation <conversation_id>``.
    """

    _display_name = "AGY"
    _log_prefix = "AGYCLI"
    _missing_binary_prefix = "AGY CLI"
    _stdin_devnull = True
    _is_agy_cli_session = True

    def __init__(
        self,
        cwd: str,
        config: Optional[AGYCLIConfig] = None,
        *,
        model_name: Optional[str] = None,
    ) -> None:
        cfg_copy = replace(config) if config is not None else AGYCLIConfig()
        super().__init__(
            cwd,
            cfg_copy,
            model_name=model_name,
        )
        self._agent_type = "agy"
        self._last_usage: Optional[ACPUsageInfo] = None
        self._turn_segments: list[str] = []
        self._turn_current_segment: list[str] = []
        self._turn_result_status: str = ""
        self._turn_result_response: str = ""
        self._turn_result_error: str = ""
        self._turn_command_result_rendered: bool = False
        self._started_tool_steps: set[int] = set()
        self._streamed_response_steps: set[int] = set()

    @property
    def supports_background_tool_continuation(self) -> bool:
        return False

    def set_runtime_settings(self, settings: Mapping[str, Any] | None) -> None:
        """Apply normalized runtime settings (``show_thinking``, ``verbosity``, ``agent_mode``)."""
        if not isinstance(settings, Mapping):
            return
        if "show_thinking" in settings or "showThinking" in settings:
            raw_st = settings["show_thinking"] if "show_thinking" in settings else settings["showThinking"]
            self._cfg.show_thinking = _parse_bool_setting(raw_st, default=bool(self._cfg.show_thinking))
        if "verbosity" in settings:
            raw_v = settings.get("verbosity")
            if raw_v is None or str(raw_v).strip().lower() in {"", "default"}:
                self._cfg.verbosity = None
            else:
                v = str(raw_v).strip().lower()
                if v in _VALID_VERBOSITY_VALUES:
                    self._cfg.verbosity = v
        if "agent_mode" in settings or "agentMode" in settings:
            raw_m = settings["agent_mode"] if "agent_mode" in settings else settings["agentMode"]
            if raw_m is None or str(raw_m).strip().lower() in {"", "default"}:
                self._cfg.agent_mode = None
            else:
                m = str(raw_m).strip().lower()
                if m in _VALID_AGENT_MODES:
                    self._cfg.agent_mode = m

    def get_runtime_settings(self) -> dict[str, Any]:
        """Return a copy of the currently configured AGY CLI runtime settings."""
        return {
            "show_thinking": bool(self._cfg.show_thinking),
            "verbosity": self._cfg.verbosity,
            "agent_mode": self._cfg.agent_mode,
        }

    def _reset_turn_tracking(self) -> None:
        self._turn_segments.clear()
        self._turn_current_segment.clear()
        self._turn_result_status = ""
        self._turn_result_response = ""
        self._turn_result_error = ""
        self._turn_command_result_rendered = False
        self._started_tool_steps.clear()
        self._streamed_response_steps.clear()

    def _split_turn_segment(self) -> None:
        if self._turn_current_segment:
            segment = "".join(self._turn_current_segment)
            self._turn_current_segment.clear()
            if segment.strip():
                self._turn_segments.append(segment)

    def _consume_turn_terminal_text(self) -> tuple[str, str]:
        self._split_turn_segment()
        main_final_segment = self._turn_segments[-1].strip() if self._turn_segments else ""
        result_response = self._turn_result_response.strip()
        if not main_final_segment and result_response:
            main_final_segment = result_response
        self._reset_turn_tracking()
        return main_final_segment, result_response

    def _run_mcp_list_once(
        self,
        *,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[float] = None,
    ) -> PromptResult:
        """Handle bare ``/mcp`` or ``/mcp list`` via bounded ``agy mcp list`` CLI subcommand."""
        hooks = self._module_hooks()
        self._ensure_session_ready(hooks.which)

        self.last_active = time.time()
        self.message_count += 1
        self.last_query = text

        from ..utils.env import build_clean_env

        env = build_clean_env()
        self._prepare_env(env)
        bounded_timeout = float(timeout) if timeout is not None and float(timeout) > 0 else 15.0
        safe_cwd = self._cwd if self._cwd and os.path.isdir(self._cwd) else None
        try:
            proc = subprocess.run(
                [self._cfg.command, "mcp", "list"],
                cwd=safe_cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=bounded_timeout,
            )
        except subprocess.TimeoutExpired:
            msg = f"❌ AGY `mcp list` 执行超时 ({bounded_timeout:.0f}s)"
            if on_event:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=msg))
            return PromptResult(stop_reason="timeout", text=msg)
        except (subprocess.SubprocessError, OSError) as exc:
            msg = f"❌ AGY `mcp list` 执行异常: {exc}"
            if on_event:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=msg))
            return PromptResult(stop_reason="error", text=msg)

        stdout = (proc.stdout or "").strip()
        stderr = self._filter_stderr(proc.stderr or "")
        if int(proc.returncode or 0) != 0:
            detail = stderr or stdout or f"agy mcp list 退出码 {proc.returncode}"
            if on_event:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=detail))
            return PromptResult(
                stop_reason="failed",
                text=detail,
                terminal_text=detail,
                terminal_text_provenance=(
                    TerminalOutputProvenance.MAIN_FINAL_SEGMENT.value
                    if on_event
                    else TerminalOutputProvenance.UNSTREAMED_RESPONSE.value
                ),
                terminal_text_tracked=bool(on_event),
            )

        if not stdout or "no mcp servers configured" in stdout.casefold():
            formatted = "🔌 **AGY MCP 服务列表**\n\n当前未配置 MCP 服务。"
        else:
            formatted = f"🔌 **AGY MCP 服务列表**\n\n```text\n{stdout}\n```"
        if on_event:
            on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=formatted, message_id="agy_cmd_mcp"))
        return PromptResult(
            stop_reason="end_turn",
            text=formatted,
            usage=self._last_usage,
            terminal_text=formatted,
            terminal_text_provenance=(
                TerminalOutputProvenance.MAIN_FINAL_SEGMENT.value
                if on_event
                else TerminalOutputProvenance.UNSTREAMED_RESPONSE.value
            ),
            terminal_text_tracked=True,
        )

    def _send_prompt_once(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
        activity_predicate: Optional[Callable[[ACPEvent], bool]] = None,
    ) -> PromptResult:
        from ..acp.native_commands import parse_slash_command

        parsed = parse_slash_command(text)
        if parsed is not None and parsed[0] == "mcp" and parsed[1].casefold() in {"", "list"}:
            return self._run_mcp_list_once(text=text, on_event=on_event, timeout=timeout)

        self._reset_turn_tracking()
        try:
            return super()._send_prompt_once(
                text,
                on_event=on_event,
                timeout=timeout,
                idle_timeout=idle_timeout,
                activity_predicate=activity_predicate,
            )
        finally:
            self._reset_turn_tracking()

    def _module_hooks(self) -> _CLIModuleHooks:
        return _CLIModuleHooks(
            popen=subprocess.Popen,
            which=shutil.which,
            terminate_and_reap=_terminate_and_reap_process,
            terminate_grace_s=_CLI_TERMINATE_GRACE_S,
            logger=logger,
        )

    def _ensure_session_ready(self, which: Callable[[str], Optional[str]]) -> None:
        if not which(self._cfg.command):
            raise RuntimeError(f"未找到 AGY CLI 可执行文件: {self._cfg.command}")

    def _mark_resumed_after_error(self) -> None:
        self.is_resumed = bool(self.session_id)

    def _build_args(self, text: str, resumed: bool) -> list[str]:
        from ..acp.native_commands import is_non_task_native_slash, parse_slash_command

        settings_payload: dict[str, Any] = {}
        base_model: str | None = None
        if self._model_name and requires_agy_cli_orchestrator_session(self._model_name):
            base_model, settings_payload = resolve_agy_orchestrator_settings(self._model_name)
        elif self._model_name:
            base_model = self._model_name

        if self._cfg.show_thinking:
            settings_payload["showThinking"] = True
        else:
            settings_payload["showThinking"] = False
        if self._cfg.verbosity:
            settings_payload["verbosity"] = self._cfg.verbosity

        stripped = str(text or "").strip()
        parsed = parse_slash_command(stripped)
        if parsed is not None and parsed[0] in {"plugin", "plugins"} and not parsed[1]:
            stripped = "/plugin list"

        if is_non_task_native_slash(stripped):
            prompt_arg = stripped
        elif parse_slash_command(stripped) is not None:
            prompt_arg = f"{stripped}\n\n{AUTONOMOUS_EXECUTION_PROMPT}"
        else:
            prompt_arg = append_execution_instructions(text)

        args: list[str] = [
            self._cfg.command,
            f"--prompt={prompt_arg}",
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
        ]
        if base_model:
            args += ["--model", base_model]
        if settings_payload:
            args += ["--settings", json.dumps(settings_payload, ensure_ascii=False)]
        if self._cfg.agent_mode in _VALID_AGENT_MODES:
            args += ["--mode", self._cfg.agent_mode]
        if resumed and self.session_id:
            args += ["--conversation", self.session_id]
        return args

    def _filter_stderr(self, err: str) -> str:
        return "\n".join(line for line in err.splitlines() if line.strip()).strip("\n")

    def _consume_stdout_line(
        self,
        line: str,
        output_chunks: list[str],
        emit: Callable[[ACPEvent], None],
    ) -> str:
        payload = line.strip()
        if not payload:
            return ""
        try:
            event = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(event, dict):
            return ""

        event_type = str(event.get("event") or event.get("type") or "").strip().lower()
        if event_type == "init":
            raw_init = event.get("init")
            init_payload: dict[str, Any] = raw_init if isinstance(raw_init, dict) else {}
            conversation_id = str(
                event.get("conversation_id")
                or init_payload.get("conversation_id")
                or ""
            ).strip()
            if conversation_id:
                self.session_id = conversation_id
            return ""

        if event_type == "command_result":
            cmd_payload = event.get("command")
            if isinstance(cmd_payload, dict):
                return self._consume_command_result(cmd_payload, output_chunks, emit)
            return ""

        if event_type == "step_update":
            raw_step = event.get("step_update")
            step_payload: dict[str, Any] = raw_step if isinstance(raw_step, dict) else event
            return self._consume_step_update(step_payload, output_chunks, emit)

        if event_type == "result":
            raw_result = event.get("result")
            result_payload: dict[str, Any] = raw_result if isinstance(raw_result, dict) else event
            return self._consume_result_event(result_payload, output_chunks, emit)

        if event_type == "error":
            return self._extract_error_message(event)

        return ""

    def _consume_command_result(
        self,
        command_payload: dict,
        output_chunks: list[str],
        emit: Callable[[ACPEvent], None],
        *,
        fallback_response: str = "",
    ) -> str:
        """Render an AGY ``command_result`` payload once and emit it to the turn stream."""
        cmd_name = str(command_payload.get("name") or "").strip().casefold()
        data = command_payload.get("data")
        rendered = _format_agy_command_result(
            cmd_name,
            data,
            fallback_response=fallback_response,
        )
        if rendered and not self._turn_command_result_rendered:
            self._turn_command_result_rendered = True
            self._append_assistant_text(
                rendered,
                f"agy_cmd_{cmd_name or 'result'}",
                output_chunks,
                emit,
            )
        return ""

    def _consume_step_update(
        self,
        event: dict,
        output_chunks: list[str],
        emit: Callable[[ACPEvent], None],
    ) -> str:
        conversation_id = str(event.get("conversation_id") or "").strip()
        if conversation_id and not self.session_id:
            self.session_id = conversation_id
        step_type = str(event.get("step_type") or "").strip().lower()
        step_index = _parse_int_or_none(event.get("step_index"))
        step_id = f"agy_step_{step_index}" if step_index is not None else "agy_step"
        state = str(event.get("state") or "").strip().upper()

        thought_delta = (
            event.get("thought_delta")
            or event.get("thinking_delta")
            or (event.get("text_delta") if step_type in {"thinking", "thought"} else None)
            or (event.get("thinking") if step_type in {"thinking", "thought"} else None)
        )
        if isinstance(thought_delta, str) and thought_delta:
            emit(ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text=thought_delta))

        if step_type == "tool":
            self._split_turn_segment()
            tool_info = event.get("tool_info")
            info = tool_info if isinstance(tool_info, dict) else {}
            tool_name = str(
                event.get("tool_name")
                or info.get("name")
                or info.get("tool_name")
                or info.get("title")
                or "tool"
            ).strip()
            raw_content = (
                info.get("result")
                or info.get("output")
                or info.get("parameters")
                or info.get("args")
                or info.get("input")
                or info.get("arguments")
                or info.get("error")
            )
            if isinstance(raw_content, (dict, list)):
                content_str = json.dumps(raw_content, ensure_ascii=False)
            else:
                content_str = str(raw_content or "")
            is_terminal = state in {"DONE", "COMPLETED", "ERROR", "FAILED", "CANCELED", "CANCELLED"}
            is_failed = state in {"ERROR", "FAILED"} or bool(info.get("error"))
            if not is_terminal:
                if step_index is not None and step_index in self._started_tool_steps:
                    ev_type = ACPEventType.TOOL_CALL_UPDATE
                else:
                    if step_index is not None:
                        self._started_tool_steps.add(step_index)
                    ev_type = ACPEventType.TOOL_CALL_START
                status = "in_progress"
            else:
                ev_type = ACPEventType.TOOL_CALL_DONE
                status = "failed" if is_failed else "completed"
            emit(
                ACPEvent(
                    event_type=ev_type,
                    tool_call=ToolCallInfo(
                        id=step_id,
                        title=tool_name,
                        kind=_infer_tool_kind(tool_name),
                        status=status,
                        content=content_str,
                        full_content=raw_content,
                    ),
                )
            )
            return ""

        if step_type == "agent_response":
            usage_info = _build_usage_info(event.get("usage"))
            if usage_info is not None:
                self._last_usage = usage_info
            text_delta = event.get("text_delta")
            if isinstance(text_delta, str) and text_delta:
                if step_index is not None:
                    self._streamed_response_steps.add(step_index)
                self._append_assistant_text(text_delta, step_id, output_chunks, emit)
                return ""
            if state == "DONE" and (step_index is None or step_index not in self._streamed_response_steps):
                full_text = event.get("text") or event.get("response")
                if isinstance(full_text, str) and full_text:
                    self._append_assistant_text(full_text, step_id, output_chunks, emit)
            return ""

        return ""

    def _consume_result_event(
        self,
        event: dict,
        output_chunks: list[str],
        emit: Callable[[ACPEvent], None],
    ) -> str:
        conversation_id = str(event.get("conversation_id") or "").strip()
        if conversation_id and not self.session_id:
            self.session_id = conversation_id
        status = str(event.get("status") or "").strip().upper()
        self._turn_result_status = status
        usage_info = _build_usage_info(event.get("usage"))
        if usage_info is not None:
            self._last_usage = usage_info
        response_text = str(event.get("response") or "")
        cmd_payload = event.get("command")
        if isinstance(cmd_payload, dict) and not self._turn_command_result_rendered:
            self._consume_command_result(
                cmd_payload,
                output_chunks,
                emit,
                fallback_response=response_text,
            )
        if response_text and not self._turn_command_result_rendered:
            self._turn_result_response = response_text
            if not output_chunks and status in {"", "SUCCESS"}:
                self._append_assistant_text(response_text, "agy_result", output_chunks, emit)
        error_text = self._extract_error_message(event)
        if error_text:
            self._turn_result_error = error_text
        if status in {"ERROR", "FAILED"}:
            return error_text or f"agy Orchestrator 执行失败 ({status})"
        return error_text if error_text and status not in {"", "SUCCESS", "CANCELED", "CANCELLED", "INTERRUPTED"} else ""

    def _append_assistant_text(
        self,
        chunk: str,
        message_id: str,
        output_chunks: list[str],
        emit: Callable[[ACPEvent], None],
    ) -> None:
        self._turn_current_segment.append(chunk)
        output_chunks.append(chunk)
        emit(
            ACPEvent(
                event_type=ACPEventType.TEXT_CHUNK,
                text=chunk,
                message_id=message_id,
            )
        )

    @staticmethod
    def _extract_error_message(event: dict) -> str:
        raw = event.get("error")
        if isinstance(raw, dict):
            msg = raw.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        elif isinstance(raw, str) and raw.strip():
            return raw.strip()
        message = event.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        return ""

    def _handle_process_exit(
        self,
        *,
        rc: int,
        out: str,
        err: str,
        state: str,
        backend_error: str,
        on_event: Optional[Callable[[ACPEvent], None]],
    ) -> PromptResult:
        del state
        result_status = self._turn_result_status
        result_error = self._turn_result_error
        main_final_segment, result_response = self._consume_turn_terminal_tracking()
        output = out or result_response

        if result_status in {"CANCELED", "CANCELLED", "INTERRUPTED"}:
            self.is_resumed = bool(self.session_id)
            return PromptResult(
                stop_reason="cancelled",
                text=output,
                usage=self._last_usage,
                terminal_text=main_final_segment,
                terminal_text_provenance=(
                    TerminalOutputProvenance.MAIN_FINAL_SEGMENT.value
                    if main_final_segment
                    else TerminalOutputProvenance.NONE.value
                ),
                terminal_text_tracked=True,
            )

        effective_error = backend_error or result_error
        if not effective_error and rc != 0:
            effective_error = err or f"agy CLI 退出码 {rc}"
        if rc != 0 or effective_error:
            detail = effective_error or err
            if detail and detail not in output:
                output = (output + "\n" + detail).strip("\n")
                if on_event:
                    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="\n" + detail))
            if detail:
                terminal_text = (
                    f"{main_final_segment}\n{detail}".strip()
                    if main_final_segment
                    else detail.strip()
                )
                provenance = (
                    TerminalOutputProvenance.MAIN_FINAL_SEGMENT.value
                    if on_event
                    else TerminalOutputProvenance.UNSTREAMED_RESPONSE.value
                )
                tracked = bool(on_event)
            elif main_final_segment:
                terminal_text = main_final_segment
                provenance = TerminalOutputProvenance.MAIN_FINAL_SEGMENT.value
                tracked = True
            else:
                terminal_text = ""
                provenance = TerminalOutputProvenance.NONE.value
                tracked = True
            self.is_resumed = bool(self.session_id)
            return PromptResult(
                stop_reason="failed",
                text=output,
                usage=self._last_usage,
                terminal_text=terminal_text,
                terminal_text_provenance=provenance,
                terminal_text_tracked=tracked,
            )

        provenance = (
            TerminalOutputProvenance.MAIN_FINAL_SEGMENT.value
            if main_final_segment
            else TerminalOutputProvenance.NONE.value
        )
        self.is_resumed = bool(self.session_id)
        return PromptResult(
            stop_reason="end_turn",
            text=output,
            usage=self._last_usage,
            terminal_text=main_final_segment,
            terminal_text_provenance=provenance,
            terminal_text_tracked=True,
        )

    def _consume_turn_terminal_tracking(self) -> tuple[str, str]:
        return self._consume_turn_terminal_text()

    def set_model(self, model_id: str, timeout: float = 10.0) -> bool:
        """Switch model/effort in-place for compatible Orchestrator CLI selections.

        Returns False when transitioning to a native ACP model or bare
        Orchestrator preset so the caller cleanly replaces the session transport.
        """
        del timeout
        candidate = str(model_id or "").strip()
        if not requires_agy_cli_orchestrator_session(candidate):
            return False
        parsed = parse_agy_model_selection(candidate)
        resolve_agy_orchestrator_settings(candidate)
        canonical = f"{parsed.model_id}/{parsed.planner_effort}/{parsed.executor_effort}"
        self._model_name = canonical
        return True

    def get_usage_info(self) -> Optional[ACPUsageInfo]:
        return self._last_usage

    def get_available_commands(self) -> list:
        """Return static native commands registered for AGY CLI sessions."""
        from ..acp.native_commands import STATIC_TOOL_NATIVE_COMMANDS

        return list(STATIC_TOOL_NATIVE_COMMANDS["agy"])

    def describe_agent(self) -> str:
        return f"cmd={self._cfg.command} cwd={self._cwd} model={self._model_name or ''} backend=cli"

    def to_snapshot(self) -> dict:
        snapshot = super().to_snapshot()
        snapshot["agent_type"] = "agy"
        return snapshot

    def get_session_info(self) -> str:
        base = super().get_session_info()
        if not self._model_name:
            return base
        try:
            parsed = parse_agy_model_selection(self._model_name)
            spec = get_agy_orchestrator_spec(parsed.model_id)
            if spec and parsed.planner_effort and parsed.executor_effort:
                planner_label = format_agy_model_effort_label(
                    spec.planner_model_id,
                    parsed.planner_effort,
                    display_base=spec.planner_display_base,
                )
                executor_label = format_agy_model_effort_label(
                    spec.executor_model_id,
                    parsed.executor_effort,
                    display_base=spec.executor_display_base,
                )
                return (
                    f"{base}\n"
                    f"- 模型: {parsed.model_id}\n"
                    f"- Planner: {planner_label}\n"
                    f"- Executor: {executor_label}"
                )
        except ValueError:
            pass
        return f"{base}\n- 模型: {self._model_name}"
