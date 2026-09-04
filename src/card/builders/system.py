from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from src.card.error_diagnostics import register_error_diagnostic
from src.utils.errors import GhostAPError, get_error_detail

from ..actions import dispatch as action_ids
from ..shared import build_responsive_layout
from ..themes import PANEL_STYLES
from ..thresholds import THRESHOLDS
from ..ui_text import UI_TEXT
from .core import CoreBuilder
from .lock import build_lock_help_body

if TYPE_CHECKING:
    from src.command_executor import CommandExecutionResult
    from src.project.context import ProjectContext

# Sentinel injected into the lru_cache'd help card; replaced post-cache
# with live lock state so dynamic info is never frozen.
# Use a UUID-based token to avoid any collision with user-generated content.
_LOCK_BODY_PLACEHOLDER = "{{__LOCK_BODY_c0f1e2d3a4b5__}}"


class SystemBuilder:
    """System-related card building utilities."""

    _SAFE_ACTION_KEYS = {
        "action",
        "chat_id",
        "origin_message_id",
        "diagnostic_token",
        "trace_id",
        "request_id",
        "project_id",
        "degraded_to",
        "mode",
    }
    _SENSITIVE_TOKEN_RE = re.compile(
        r"(?i)\b(cmd|cwd|path|args|token|secret|password|passwd|key)\s*=\s*[^\s\n]+"
    )
    _PATH_RE = re.compile(r"(?<![\w])(?:/[\w.\-]+){2,}")

    @staticmethod
    def _callback_button(*, text: str, action: dict, button_type: str = "default") -> dict:
        """Build a Feishu callback button with value and behavior kept in sync."""
        value = dict(action)
        return {
            "tag": "button",
            "text": {"tag": "plain_text", "content": text},
            "type": button_type,
            "value": value,
            "behaviors": [{"type": "callback", "value": value}],
        }

    @staticmethod
    def _compact_help_path(path: object, *, limit: int = 42) -> str:
        text = str(path or "~").strip() or "~"
        if len(text) <= limit:
            return text
        parts = [part for part in text.split("/") if part]
        if len(parts) >= 3:
            compact = ".../" + "/".join(parts[-3:])
            if len(compact) <= limit:
                return compact
        return "..." + text[-max(1, limit - 3):]

    @staticmethod
    def build_directory_change_card(
        project: Optional[ProjectContext],
        path: str,
        success: bool = True,
    ) -> Optional[tuple[str, str]]:
        """Build a card for directory change result."""
        from .core import CoreBuilder
        from .project import ProjectBuilder

        if success:
            banner_msg = UI_TEXT["project_dir_switched_banner"].format(path=path)
            banner = CoreBuilder._build_banner_element(banner_msg, type="success")
            title = UI_TEXT["system_dir_changed_title"]
            detail = UI_TEXT["project_dir_switched_detail"].format(path=path)
        else:
            banner_msg = UI_TEXT["project_dir_switch_failed_banner"].format(path=path)
            banner = CoreBuilder._build_banner_element(banner_msg, type="error")
            title = UI_TEXT["system_error_title"]
            detail = UI_TEXT["project_dir_switch_failed_detail"].format(path=path)

        if project:
            return ProjectBuilder.build_project_response_card(
                project,
                title,
                detail,
                show_buttons=True,
                banner=banner,
            )
        return None

    @staticmethod
    def build_coco_status_content(
        current_model: Optional[str],
        models: list,
    ) -> str:
        """Build the Markdown content for Coco status info."""
        status_lines = [UI_TEXT["system_coco_status_title"]]
        status_lines.append(UI_TEXT["system_coco_current_model"].format(model=current_model or UI_TEXT["system_not_set"]))

        status_lines.append(UI_TEXT["system_coco_available_models"])
        for m in models:
            mark = "✅ " if m.name == current_model else "   "
            status_lines.append(f"{mark}`{m.name}` - {m.description}")

        return "\n".join(status_lines)

    @staticmethod
    def build_error_card(
        exc: Exception | str,
        title: str = "",
        project: Optional[ProjectContext] = None,
        *,
        summary: Optional[str] = None,
        details: Optional[str] = None,
        detail_action: Optional[dict] = None,
        continue_action: Optional[dict] = None,
        retry_action: Optional[dict] = None,
        severity: str = "fatal",
    ) -> tuple[str, str]:
        from ..shared import build_quick_actions

        if not title:
            title = UI_TEXT["system_error_title"]

        message = SystemBuilder._card_safe_summary(exc, summary=summary, severity=severity)
        severity_map = {
            "recoverable": ("orange", "🟠 可恢复错误", "可重试或自动恢复的问题"),
            "degraded": ("yellow", "🟡 降级错误", "功能已降级，核心流程会尽量继续"),
            "fatal": ("red", "🔴 致命错误", "需要停止当前操作并暴露根因"),
        }
        header_template, severity_label, severity_hint = severity_map.get(severity, severity_map["fatal"])
        quick_actions = []
        context = {}

        if isinstance(exc, GhostAPError):
            quick_actions = exc.quick_actions
            context = exc.context

        detail_binding = {
            **SystemBuilder._safe_action_payload(context),
            **SystemBuilder._safe_action_payload(detail_action),
        }
        has_trusted_detail_binding = bool(detail_binding.get("chat_id"))
        exposes_details = has_trusted_detail_binding and (
            severity == "degraded"
            or (bool(detail_action) and not quick_actions)
        )
        exposes_retry = bool(
            severity != "degraded" and not quick_actions and retry_action
        )
        exposes_quick_actions = bool(
            severity != "degraded" and quick_actions
        )
        status_text = SystemBuilder._current_status_text(
            severity,
            continue_action,
            context,
            has_details=exposes_details,
            has_retry=exposes_retry,
            has_quick_actions=exposes_quick_actions,
        )
        detail_hint = (
            f"\n\n{UI_TEXT['card_lifecycle_details_collapsed']}"
            if exposes_details
            else ""
        )
        elements = []
        if project:
            elements.append(CoreBuilder._build_directory_element(project))
            elements.append({"tag": "hr"})

        elements.append(
            CoreBuilder._build_content_element(
                f"{severity_label}\n{severity_hint}\n\n"
                f"❌ **错误摘要**\n{message}\n\n"
                f"**错误场景**\n{title}\n\n"
                f"**当前状态**\n{status_text}"
                f"{detail_hint}"
            )
        )

        # project info is handled by project_response_card if needed, but build_error_card
        # is often used for generic errors. Original code had optional project.
        # We'll stick to a simpler interactive card here or wrap it.

        if severity == "degraded" and exposes_details:
            detail_payload = SystemBuilder._build_detail_action(
                detail_action,
                title=title,
                summary=message,
                details=details,
                context=context,
            )
            elements.extend(
                build_responsive_layout(
                    [
                        SystemBuilder._callback_button(
                            text=UI_TEXT["card_lifecycle_show_details"],
                            action=detail_payload,
                            button_type="default",
                        )
                    ],
                    layout="mobile",
                )
            )

            # Safe work and recovery continue automatically; the card only
            # exposes diagnostics and never waits for a continue decision.
        elif quick_actions:
            buttons = build_quick_actions(quick_actions, context)
            elements.extend(build_responsive_layout(buttons))
        else:
            buttons = []
            if exposes_details:
                safe_detail_action = SystemBuilder._build_detail_action(
                    detail_action,
                    title=title,
                    summary=message,
                    details=details,
                    context=context,
                )
                buttons.append(
                    SystemBuilder._callback_button(
                        text=UI_TEXT["card_lifecycle_show_details"],
                        action=safe_detail_action,
                        button_type="default",
                    )
                )
            if retry_action:
                retry_button_text = UI_TEXT["card_lifecycle_restart"]
                buttons.append(
                    SystemBuilder._callback_button(
                        text=retry_button_text,
                        action=SystemBuilder._safe_action_payload(retry_action),
                        button_type="primary" if severity == "recoverable" else "default",
                    )
                )
            if buttons:
                elements.extend(build_responsive_layout(buttons))

        card = CoreBuilder._wrap_card(
            UI_TEXT["system_error_prompt_title"],
            header_template,
            elements,
        )
        notification_summary = {
            "recoverable": "操作遇到可恢复错误，打开卡片查看处理建议",
            "degraded": (
                "功能已降级，打开卡片查看当前状态与诊断"
                if exposes_details
                else "功能已降级，打开卡片查看当前状态"
            ),
            "fatal": "操作已停止，打开卡片查看错误摘要与处理建议",
        }.get(severity, "操作状态异常，打开卡片查看错误摘要")
        card["config"]["summary"] = {
            "content": notification_summary
        }
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def _safe_action_payload(payload: Optional[dict]) -> dict:
        return {key: value for key, value in dict(payload or {}).items() if key in SystemBuilder._SAFE_ACTION_KEYS}

    @staticmethod
    def _sanitize_card_text(text: object, *, fallback: str) -> str:
        value = str(text or "").strip()
        if not value:
            return fallback
        value = SystemBuilder._SENSITIVE_TOKEN_RE.sub(lambda match: f"{match.group(1)}=<redacted>", value)
        value = SystemBuilder._PATH_RE.sub("<path>", value)
        return value[:600].rstrip() or fallback

    @staticmethod
    def _card_safe_summary(exc: Exception | str, *, summary: Optional[str], severity: str) -> str:
        fallback = "操作未能按原模式完成，已进入安全降级路径。" if severity == "degraded" else UI_TEXT["system_unknown_error"]
        if severity == "degraded":
            # Degraded cards are often built from startup/runtime exceptions that
            # include commands, paths or stack traces.  The user-visible body must
            # always stay on a fixed safe boundary; raw details are disclosed only
            # through the diagnostic store after context validation.
            return fallback
        if summary is not None:
            return SystemBuilder._sanitize_card_text(summary, fallback=fallback)
        if isinstance(exc, Exception):
            return SystemBuilder._sanitize_card_text(get_error_detail(exc), fallback=fallback)
        return SystemBuilder._sanitize_card_text(exc, fallback=fallback)

    @staticmethod
    def _resolve_degraded_mode(action: Optional[dict], context: Optional[dict]) -> str:
        payload = {**dict(context or {}), **dict(action or {})}
        return str(payload.get("degraded_to") or "")

    @staticmethod
    def _current_status_text(
        severity: str,
        continue_action: Optional[dict],
        context: Optional[dict],
        *,
        has_details: bool,
        has_retry: bool,
        has_quick_actions: bool,
    ) -> str:
        if severity == "degraded":
            mode = SystemBuilder._resolve_degraded_mode(continue_action, context)
            if mode:
                return f"可继续使用 {SystemBuilder._display_mode_label(mode)}；原能力恢复由系统自动处理。"
            return "当前暂未确定可继续模式；系统将继续安全的可执行部分，或进入明确失败终态。"
        if has_quick_actions:
            return "当前操作未完成；可按卡片操作继续处理。"
        if severity == "recoverable":
            if has_details and has_retry:
                return "当前操作未完成；可查看脱敏诊断或按卡片按钮重试。"
            if has_details:
                return "当前操作未完成；可查看脱敏诊断后重新发起。"
            if has_retry:
                return "当前操作未完成；可按卡片按钮重试。"
            return "当前操作未完成；请按错误摘要中的提示重新发起。"
        if has_details and has_retry:
            return "当前操作已停止；可查看脱敏诊断或按卡片按钮重试。"
        if has_details:
            return "当前操作已停止；可查看脱敏诊断并按提示重新发起。"
        if has_retry:
            return "当前操作已停止；可按卡片按钮重试。"
        return "当前操作已停止；请按错误摘要中的提示重新发起。"

    @staticmethod
    def _display_mode_label(mode: str) -> str:
        labels = {
            "coco": "Coco",
            "claude": "Claude",
            "claude cli": "Claude CLI",
            "aiden": "Aiden",
            "codex": "Codex",
            "gemini": "Gemini",
        }
        raw = str(mode or "").strip()
        return labels.get(raw.lower(), raw)

    @staticmethod
    def _build_detail_action(
        detail_action: Optional[dict],
        *,
        title: str,
        summary: str,
        details: Optional[str],
        context: Optional[dict],
    ) -> dict:
        raw_payload = dict(detail_action or {})
        payload = {**SystemBuilder._safe_action_payload(context), **SystemBuilder._safe_action_payload(raw_payload)}
        payload["action"] = str(payload.get("action") or action_ids.SHOW_ERROR_DETAILS)
        if not payload.get("diagnostic_token"):
            raw_details = (
                raw_payload.get("details")
                or raw_payload.get("detail")
                or raw_payload.get("stderr")
                or raw_payload.get("error")
                or details
                or summary
            )
            payload["diagnostic_token"] = register_error_diagnostic(
                title=title,
                summary=summary,
                details=str(raw_details or "本次错误暂无更多诊断上下文。"),
                chat_id=payload.get("chat_id"),
                origin_message_id=payload.get("origin_message_id"),
                request_id=payload.get("request_id"),
                trace_id=payload.get("trace_id"),
            )
        return payload

    @staticmethod
    def build_shell_result_card(
        cmd: str,
        result: "CommandExecutionResult",
        working_dir: Optional[str] = None,
        project: Optional[ProjectContext] = None,
    ) -> tuple[str, str]:
        """Build an interactive card for shell command execution results."""
        if result.success:
            header_title = UI_TEXT["system_shell_success_title"]
            header_template = "turquoise"
        else:
            header_title = UI_TEXT["system_shell_failed_title"]
            header_template = "red"

        elements = [
            CoreBuilder._build_directory_element(project, working_dir),
            {"tag": "hr"},
            {"tag": "markdown", "content": f"> 🖥️ `{cmd}`"},
        ]

        if result.error_message:
            elements.append(
                {
                    "tag": "markdown",
                    "content": f"🚫 **{result.error_message}**",
                }
            )
        elif result.stdout or result.stderr:
            from ..truncation import truncate_bash_output

            _shell_notice = UI_TEXT["shell_truncated"]

            if result.stdout:
                stdout_content = truncate_bash_output(
                    result.stdout,
                    max_chars=THRESHOLDS["SHELL_STDOUT_MAX"],
                    max_lines=999999,  # no line limit for shell result cards
                    notice=_shell_notice,
                )
                elements.append(
                    {
                        "tag": "markdown",
                        "content": f"```BASH\n{stdout_content}\n```",
                    }
                )
            if result.stderr:
                stderr_content = truncate_bash_output(
                    result.stderr,
                    max_chars=THRESHOLDS["SHELL_STDERR_MAX"],
                    max_lines=999999,
                    notice=_shell_notice,
                )
                elements.append(
                    {
                        "tag": "markdown",
                        "content": f"{UI_TEXT['system_shell_stderr_label']}\n```BASH\n{stderr_content}\n```",
                    }
                )
        else:
            elements.append(
                {
                    "tag": "markdown",
                    "content": UI_TEXT["system_shell_no_output"],
                }
            )

        elements.append(
            {
                "tag": "markdown",
                "content": UI_TEXT["system_shell_return_code"].format(code=result.return_code),
                "text_size": "notation",
            }
        )

        card = CoreBuilder._wrap_card(header_title, header_template, elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_command_menu_card(project: Optional[ProjectContext] = None) -> tuple[str, str]:
        """Build a mobile-friendly command menu card."""
        # Local import preserves the card -> Feishu initialization boundary.
        from src.feishu.product_catalog import format_owner_execution_lane_summary

        project_id = project.project_id if project else None

        buttons = [
            {
                "text": UI_TEXT["system_menu_btn_new_project"],
                "type": "primary",
                "action": "new_project_prompt",
            },
            {
                "text": UI_TEXT["system_menu_btn_switch_project"],
                "type": "default",
                "action": "switch_project",
            },
            {
                "text": UI_TEXT["system_menu_btn_deep_task"],
                "type": "primary",
                "action": "enter_deep_prompt",
            },
            {
                "text": UI_TEXT["system_menu_btn_workflow"],
                "type": "primary",
                "action": "show_workflow_menu",
            },
            {
                "text": UI_TEXT["system_menu_btn_status"],
                "type": "default",
                "action": "show_status",
            },
            {
                "text": UI_TEXT["system_menu_btn_help"],
                "type": "default",
                "action": "show_help_menu",
            },
        ]

        # Convert to actual card buttons
        card_buttons = []
        for btn in buttons:
            btn_value = {"action": btn["action"], "project_id": project_id}
            card_buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": btn["text"]},
                    "type": btn["type"],
                    "value": btn_value,
                    "behaviors": [{"type": "callback", "value": btn_value}],
                }
            )

        elements = [
            CoreBuilder._build_directory_element(project),
            {"tag": "hr"},
            {"tag": "markdown", "content": UI_TEXT["system_menu_header"]},
            {
                "tag": "collapsible_panel",
                "expanded": False,
                "header": {
                    "title": {
                        "tag": "markdown",
                        "content": "**执行通道与完成度**",
                    },
                    "vertical_align": "center",
                },
                "border": {"color": "grey", "corner_radius": PANEL_STYLES["corner_radius"]},
                "vertical_spacing": PANEL_STYLES["vertical_spacing"],
                "padding": PANEL_STYLES["padding_standard"],
                "elements": [{
                    "tag": "markdown",
                    "content": format_owner_execution_lane_summary(),
                }],
            },
        ]
        elements.extend(build_responsive_layout(card_buttons))

        card = CoreBuilder._wrap_card(UI_TEXT["system_menu_title"], "blue", elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_help_card(
        project: Optional[ProjectContext] = None,
        category: str = "main",
        working_dir: Optional[str] = None,
        current_mode: any = None,
        is_admin: bool = False,
        lock_enabled: bool = False,
        chat_id: str = "",
        no_admin_configured: bool = False,
        *,
        session_idle_timeout: Optional[int] = None,
        session_idle_warn_at_remaining: Optional[int] = None,
        lock_undo_window_seconds: Optional[int] = None,
    ) -> tuple[str, str]:
        """Build a categorized help card."""
        from ...config import get_settings
        from ...mode import InteractionMode

        if session_idle_timeout is None:
            session_idle_timeout = get_settings().card.session_idle_timeout
        if session_idle_warn_at_remaining is None:
            session_idle_warn_at_remaining = get_settings().card.session_idle_warn_at_remaining
        if lock_undo_window_seconds is None:
            lock_undo_window_seconds = get_settings().lock_undo_window_seconds

        mode_emoji = {
            InteractionMode.SMART: UI_TEXT["system_mode_smart"],
            InteractionMode.COCO: UI_TEXT["system_mode_coco"],
            InteractionMode.CLAUDE: UI_TEXT["system_mode_claude"],
            InteractionMode.AIDEN: UI_TEXT["system_mode_aiden"],
            InteractionMode.CODEX: UI_TEXT["system_mode_codex"],
            InteractionMode.GEMINI: UI_TEXT["system_mode_gemini"],
            InteractionMode.TRAEX: UI_TEXT["system_mode_traex"],
            InteractionMode.GROK: UI_TEXT["system_mode_grok"],
            InteractionMode.DSH: UI_TEXT["system_mode_dsh"],
        }

        current_mode_str = mode_emoji.get(current_mode, UI_TEXT["system_mode_smart"])

        # Extract primitives for caching
        project_name = project.project_name if project else None
        root_path = project.root_path if project else None
        project_id = project.project_id if project else None

        # Bucketize timeout params to reduce lru_cache key space (ceil to nearest 60s,
        # matching the math.ceil display logic inside the cached builder)
        _bucketed_timeout = math.ceil(session_idle_timeout / 60) * 60
        _bucketed_warn = math.ceil(session_idle_warn_at_remaining / 60) * 60

        msg_type, card_json = SystemBuilder._build_help_card_cached(
            project_name=project_name,
            root_path=root_path,
            project_id=project_id,
            category=category,
            working_dir=working_dir,
            current_mode_str=current_mode_str,
            is_admin=is_admin,
            lock_enabled=lock_enabled,
            session_idle_timeout=_bucketed_timeout,
            session_idle_warn_at_remaining=_bucketed_warn,
        )

        # Post-cache injection: replace the lock-body placeholder with
        # live lock state so that lru_cache never freezes stale lock info.
        if lock_enabled and _LOCK_BODY_PLACEHOLDER in card_json:
            live_body = build_lock_help_body(is_admin=is_admin, chat_id=chat_id, lock_undo_window_seconds=lock_undo_window_seconds)
            # FS-09: Append admin guidance when ADMIN_USER_IDS is empty
            if no_admin_configured:
                live_body += "\n\n💡 如需群锁定功能，请联系 Bot 部署者完成配置"
            # The placeholder lives inside a json.dumps'd string, so we must
            # escape the replacement to keep the JSON valid (e.g. \n → \\n).
            _escaped = json.dumps(live_body, ensure_ascii=False)[1:-1]  # strip surrounding quotes
            card_json = card_json.replace(_LOCK_BODY_PLACEHOLDER, _escaped)

        return msg_type, card_json

    @staticmethod
    @lru_cache(maxsize=64)
    def _build_help_card_cached(
        project_name: Optional[str],
        root_path: Optional[str],
        project_id: Optional[str],
        category: str,
        working_dir: Optional[str],
        current_mode_str: str,
        is_admin: bool = False,
        lock_enabled: bool = False,
        session_idle_timeout: int | None = None,
        session_idle_warn_at_remaining: int | None = None,
    ) -> tuple[str, str]:
        """Build a documentation-only help card with collapsible command sections.

        The ``category`` parameter is accepted for backward compatibility but
        no longer drives tab switching — the card always renders every section
        so users see all commands at once.
        """
        del category, project_id  # kept for call-site/cache compatibility; unused

        project_info = f"**{project_name}**" if project_name else UI_TEXT["system_no_project"]
        cwd_display = SystemBuilder._compact_help_path(working_dir or root_path or "~")

        # All command sections stay in one card; only the first is expanded initially.
        sections = [
            (
                UI_TEXT["system_help_section_modes"],
                UI_TEXT["system_help_section_modes_body"]
            ),
            (
                UI_TEXT["system_help_section_deep"],
                UI_TEXT["system_help_section_deep_body"]
            ),
            (
                UI_TEXT["system_help_section_spec"],
                UI_TEXT["system_help_section_spec_body"]
            ),
            (
                UI_TEXT["system_help_section_project"],
                UI_TEXT["system_help_section_project_body"]
            ),
            (
                UI_TEXT["system_help_section_workflow"],
                UI_TEXT["system_help_section_workflow_body"]
            ),
            (
                UI_TEXT["system_help_section_hire"],
                UI_TEXT["system_help_section_hire_body"]
            ),
        ]

        # F-12: Only show lock section when lock feature is enabled
        if lock_enabled:
            _lock_title = UI_TEXT["system_help_section_lock"] if is_admin else UI_TEXT["system_help_section_lock_nonadmin"]
            sections.append((
                _lock_title,
                _LOCK_BODY_PLACEHOLDER,
            ))

        if session_idle_timeout is None or session_idle_warn_at_remaining is None:
            # Fallback defaults (should not normally be reached since build_help_card
            # always passes values, but kept for safety / direct test calls).
            session_idle_timeout = session_idle_timeout if session_idle_timeout is not None else 1800
            session_idle_warn_at_remaining = session_idle_warn_at_remaining if session_idle_warn_at_remaining is not None else 300
        timeout_seconds = session_idle_timeout
        warn_before_seconds = session_idle_warn_at_remaining
        # NOTE: config validator enforces minimum=300, sub-60s branch intentionally removed
        timeout_minutes = max(1, math.ceil(timeout_seconds / 60))
        if timeout_minutes >= 120:
            hours = timeout_minutes // 60
            timeout_display = f"{hours} 小时" if timeout_seconds % 3600 == 0 else f"约 {hours} 小时"
        else:
            timeout_display = f"{timeout_minutes} 分钟" if timeout_seconds % 60 == 0 else f"约 {timeout_minutes} 分钟"
        warn_minutes = max(1, math.ceil(warn_before_seconds / 60))
        if warn_before_seconds % 60 == 0:
            warn_display = f"{warn_minutes} 分钟"
        else:
            warn_display = f"约 {warn_minutes} 分钟"
        tips = UI_TEXT["system_help_tips"].format(
            timeout_display=timeout_display,
            warn_display=warn_display,
        )

        elements = [
            {
                "tag": "markdown",
                "text_size": "notation",
                "content": UI_TEXT["system_help_status_header"].format(
                    mode=current_mode_str,
                    cwd=cwd_display,
                    project=project_info
                ),
            },
        ]
        elements.append({"tag": "hr"})

        for idx, (title, body) in enumerate(sections):
            elements.append({
                "tag": "collapsible_panel",
                "expanded": idx == 0,
                "header": {
                    "title": {"tag": "markdown", "content": f"**{title}**"},
                    "vertical_align": "center",
                },
                "border": {"color": "grey", "corner_radius": PANEL_STYLES["corner_radius"]},
                "vertical_spacing": PANEL_STYLES["vertical_spacing"],
                "padding": PANEL_STYLES["padding_standard"],
                "elements": [{"tag": "markdown", "content": body}],
            })

        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": tips, "text_size": "notation"})

        card = CoreBuilder._wrap_card(UI_TEXT["system_help_title"], "blue", elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_model_cascade_card(
        models: list,
        tool_name: str,
        project_id: Optional[str] = None,
        current_model: Optional[str] = None,
        thread_root_id: Optional[str] = None,
        *,
        pending_group: Optional[str] = None,
        pending_profile: Optional[str] = None,
        pending_effort: Optional[str] = None,
        context_markdown: Optional[str] = None,
        group_action: str = action_ids.SELECT_ACP_MODEL_GROUP,
        profile_action: str = action_ids.SELECT_ACP_MODEL_PROFILE,
        effort_action: str = action_ids.SELECT_ACP_MODEL_EFFORT,
        select_action: str = action_ids.SELECT_ACP_MODEL,
        refresh_action: str = action_ids.REFRESH_ACP_MODELS,
        value_extra: Optional[dict] = None,
    ) -> tuple[str, str]:
        """Build the explicit model picker without activating a programming mode."""
        from src.card.render.model_cascade import resolve_model_cascade

        tool = str(tool_name or "").strip().lower()
        state = resolve_model_cascade(
            models,
            current_model=current_model,
            selected_model=pending_group,
            selected_profile=pending_profile,
            selected_effort=pending_effort,
        )

        def payload(action: str, **updates: object) -> dict:
            value = dict(value_extra or {})
            value.update(
                {
                    "action": action,
                    "tool_name": tool,
                    "project_id": project_id,
                    "current_model": current_model,
                    "model_group": state.selected_model,
                    "model_profile": state.selected_profile,
                    "model_effort": state.selected_effort,
                }
            )
            if thread_root_id:
                value["thread_root_id"] = thread_root_id
            value.update(updates)
            return value

        def select_static(
            *,
            name: str,
            placeholder: str,
            values: tuple[str, ...],
            selected: Optional[str],
            action: str,
        ) -> dict:
            callback = payload(action)
            element = {
                "tag": "select_static",
                "name": name,
                "placeholder": {"tag": "plain_text", "content": placeholder},
                "options": [
                    {
                        "text": {"tag": "plain_text", "content": value},
                        "value": value,
                    }
                    for value in values
                ],
                "value": callback,
                "behaviors": [{"type": "callback", "value": callback}],
            }
            if selected in values:
                element["initial_option"] = selected
            return element

        elements: list[dict] = [
            {
                "tag": "markdown",
                "content": UI_TEXT["system_acp_model_select_intro"],
            }
        ]
        if context_markdown:
            elements.append({"tag": "markdown", "content": str(context_markdown)})
        elements.append({"tag": "hr"})

        has_dimensions = bool(state.profiles or state.efforts)
        if has_dimensions:
            elements.extend(
                [
                    {
                        "tag": "markdown",
                        "content": UI_TEXT["system_acp_model_group_label"],
                    },
                    select_static(
                        name="model_group",
                        placeholder=UI_TEXT["system_acp_model_group_placeholder"],
                        values=state.model_names,
                        selected=state.selected_model,
                        action=group_action,
                    ),
                ]
            )
            if state.profiles:
                elements.extend(
                    [
                        {
                            "tag": "markdown",
                            "content": UI_TEXT["system_acp_model_profile_label"],
                        },
                        select_static(
                            name="model_profile",
                            placeholder=UI_TEXT["system_acp_model_profile_placeholder"],
                            values=state.profiles,
                            selected=state.selected_profile,
                            action=profile_action,
                        ),
                    ]
                )
            if state.efforts:
                elements.extend(
                    [
                        {
                            "tag": "markdown",
                            "content": UI_TEXT["system_acp_model_effort_label"],
                        },
                        select_static(
                            name="model_effort",
                            placeholder=UI_TEXT["system_acp_model_effort_placeholder"],
                            values=state.efforts,
                            selected=state.selected_effort,
                            action=effort_action,
                        ),
                    ]
                )

            default_button = SystemBuilder._callback_button(
                text=UI_TEXT["system_acp_default_model_option"],
                action=payload(
                    select_action,
                    model_name=None,
                    use_default_model=True,
                ),
            )
            confirm_button = SystemBuilder._callback_button(
                text=UI_TEXT["system_acp_confirm_model"].format(
                    model=state.selection,
                ),
                action=payload(
                    select_action,
                    model_name=state.selection,
                    use_default_model=False,
                ),
                button_type="primary",
            )
            elements.extend(build_responsive_layout([default_button, confirm_button], layout="mobile"))
        else:
            buttons = [
                SystemBuilder._callback_button(
                    text=UI_TEXT["system_acp_default_model_option"],
                    action=payload(
                        select_action,
                        model_name=None,
                        use_default_model=True,
                    ),
                    button_type="primary" if not current_model else "default",
                )
            ]
            for model_name in state.model_names:
                buttons.append(
                    SystemBuilder._callback_button(
                        text=model_name,
                        action=payload(
                            select_action,
                            model_group=model_name,
                            model_profile=None,
                            model_effort=None,
                            model_name=model_name,
                            use_default_model=False,
                        ),
                        button_type="primary" if model_name == current_model else "default",
                    )
                )
            elements.extend(build_responsive_layout(buttons, layout="mobile"))
            if not state.model_names:
                elements.append(
                    {
                        "tag": "markdown",
                        "text_size": "notation",
                        "content": UI_TEXT["system_acp_no_models_hint"],
                    }
                )

        elements.append({"tag": "hr"})
        elements.extend(
            build_responsive_layout(
                [
                    SystemBuilder._callback_button(
                        text=UI_TEXT["system_acp_refresh_models"],
                        action=payload(refresh_action),
                    )
                ],
                layout="mobile",
            )
        )
        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_model_select_title"].format(tool=tool.capitalize()),
            "blue",
            elements,
        )
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_model_loading_card(
        tool_name: str,
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build the in-place frame shown while official model data loads."""
        del project_id, thread_root_id
        tool = str(tool_name or "").strip().lower()
        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_model_loading_title"].format(tool=tool.capitalize()),
            "blue",
            [
                {
                    "tag": "markdown",
                    "content": UI_TEXT["system_acp_model_loading_body"].format(tool=tool),
                }
            ],
        )
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_model_error_card(
        tool_name: str,
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build the retryable frame shown when official model discovery fails."""
        tool = str(tool_name or "").strip().lower()
        retry_value = {
            "action": action_ids.REFRESH_ACP_MODELS,
            "tool_name": tool,
            "project_id": project_id,
        }
        if thread_root_id:
            retry_value["thread_root_id"] = thread_root_id
        elements = [
            {
                "tag": "markdown",
                "content": UI_TEXT["system_acp_model_error_body"].format(tool=tool),
            },
            {"tag": "hr"},
            *build_responsive_layout(
                [
                    SystemBuilder._callback_button(
                        text=UI_TEXT["system_acp_refresh_models"],
                        action=retry_value,
                        button_type="primary",
                    )
                ],
                layout="mobile",
            ),
        ]
        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_model_error_title"].format(tool=tool.capitalize()),
            "red",
            elements,
        )
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_programming_initializing_card(
        tool_name: str,
        model_name: Optional[str],
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build the immediate frame while the selected ACP session starts."""
        del project_id, thread_root_id
        tool = str(tool_name or "").strip().lower()
        model = str(model_name or UI_TEXT["system_acp_default_model_option"])
        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_programming_initializing_title"].format(tool=tool.capitalize()),
            "blue",
            [
                {
                    "tag": "markdown",
                    "content": UI_TEXT["system_acp_programming_initializing_body"].format(model=model),
                }
            ],
        )
        card["config"]["summary"] = {
            "content": "编程模式正在初始化，打开卡片查看进度"
        }
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_programming_ready_card(
        tool_name: str,
        model_name: Optional[str],
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build the final frame after the selected ACP session is ready."""
        tool = str(tool_name or "").strip().lower()
        model = str(model_name or UI_TEXT["system_acp_default_model_option"])
        switch_value = {
            "action": action_ids.REFRESH_ACP_MODELS,
            "tool_name": tool,
            "project_id": project_id,
            "current_model": model_name,
        }
        if thread_root_id:
            switch_value["thread_root_id"] = thread_root_id
        elements = [
            {
                "tag": "markdown",
                "content": UI_TEXT["system_acp_programming_ready_body"].format(model=model),
            },
            {"tag": "hr"},
            *build_responsive_layout(
                [
                    SystemBuilder._callback_button(
                        text=UI_TEXT["system_acp_switch_model_btn"],
                        action=switch_value,
                    )
                ],
                layout="mobile",
            ),
        ]
        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_programming_ready_title"].format(tool=tool.capitalize()),
            "green",
            elements,
        )
        card["config"]["summary"] = {
            "content": "编程模式已就绪，打开卡片查看当前模型"
        }
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_programming_failed_card(
        tool_name: str,
        model_name: Optional[str],
        reason: str,
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
        *,
        model_group: Optional[str] = None,
        model_profile: Optional[str] = None,
        model_effort: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build the retryable terminal frame after ACP startup fails."""
        tool = str(tool_name or "").strip().lower()
        model = str(model_name or UI_TEXT["system_acp_default_model_option"])
        base = {"tool_name": tool, "project_id": project_id}
        if thread_root_id:
            base["thread_root_id"] = thread_root_id
        retry_value = {
            **base,
            "action": action_ids.SELECT_ACP_MODEL,
            "model_group": model_group,
            "model_profile": model_profile,
            "model_effort": model_effort,
            "model_name": model_name,
            "use_default_model": model_name is None,
        }
        back_value = {
            **base,
            "action": action_ids.REFRESH_ACP_MODELS,
            "current_model": model_name,
        }
        safe_reason = SystemBuilder._sanitize_card_text(
            reason,
            fallback=UI_TEXT["system_acp_activation_failed_safe"],
        )
        elements = [
            {
                "tag": "markdown",
                "content": UI_TEXT["system_acp_programming_failed_body"].format(
                    model=model,
                    reason=safe_reason,
                ),
            },
            {"tag": "hr"},
            *build_responsive_layout(
                [
                    SystemBuilder._callback_button(
                        text=UI_TEXT["system_acp_retry_activation_btn"],
                        action=retry_value,
                        button_type="primary",
                    ),
                    SystemBuilder._callback_button(
                        text=UI_TEXT["system_acp_back_to_models_btn"],
                        action=back_value,
                    ),
                ],
                layout="mobile",
            ),
        ]
        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_programming_failed_title"].format(tool=tool.capitalize()),
            "orange",
            elements,
        )
        card["config"]["summary"] = {
            "content": "编程模式初始化失败，可在卡片中重试"
        }
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_programming_superseded_card(
        tool_name: str,
        model_name: Optional[str],
        reason: str,
    ) -> tuple[str, str]:
        """Build a terminal frame for an activation that cannot take effect."""
        tool = str(tool_name or "").strip().lower()
        model = str(model_name or UI_TEXT["system_acp_default_model_option"])
        safe_reason = SystemBuilder._sanitize_card_text(
            reason,
            fallback=UI_TEXT["system_acp_activation_superseded"],
        )
        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_programming_superseded_title"].format(
                tool=tool.capitalize()
            ),
            "grey",
            [
                {
                    "tag": "markdown",
                    "content": UI_TEXT[
                        "system_acp_programming_superseded_body"
                    ].format(model=model, reason=safe_reason),
                }
            ],
        )
        card["config"]["summary"] = {
            "content": "本次模型初始化未生效，打开卡片查看状态"
        }
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def _employee_roster_text(
        value: object,
        *,
        fallback: str,
        limit: int = 100,
    ) -> str:
        """Return bounded Feishu-markdown text for a roster display field."""

        text = " ".join(str(value or "").split()).strip() or fallback
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for marker in ("\\", "`", "*", "_", "[", "]"):
            text = text.replace(marker, f"\\{marker}")
        return text[:limit].rstrip("\\") or fallback

    @staticmethod
    def _employee_roster_code_value(
        value: object,
        *,
        fallback: str,
        limit: int = 100,
    ) -> str:
        """Return bounded literal text for a roster inline-code value."""

        text = " ".join(str(value or "").split()).strip() or fallback
        return text[:limit] or fallback

    @staticmethod
    def _employee_roster_inline_code(value: str) -> str:
        """Render a safe code span, falling back to prose for backtick values."""

        if "`" not in value:
            return f"`{value}`"
        value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for marker in ("\\", "`", "*", "_", "[", "]"):
            value = value.replace(marker, f"\\{marker}")
        return value

    @staticmethod
    def build_employee_roster_card(
        entries: Sequence[Mapping[str, object]],
    ) -> tuple[str, str]:
        """Build a deterministic, display-only card for visible employees."""

        state_labels = {
            "active": "🟢 在职",
            "action_required": "🟠 需要处理",
            "retiring": "🟡 退役中",
            "draft": "🔵 草稿",
            "provisioning_app": "🔵 创建应用中",
            "storing_credential": "🔵 保存凭据中",
            "configuring": "🔵 配置中",
            "validating": "🔵 验证中",
            "ready_pending_verification": "🔵 待激活",
        }
        state_order = {
            "active": 0,
            "action_required": 1,
            "draft": 2,
            "provisioning_app": 2,
            "storing_credential": 2,
            "configuring": 2,
            "validating": 2,
            "ready_pending_verification": 2,
            "retiring": 3,
        }

        normalized: list[dict[str, object]] = []
        for entry in entries:
            state = str(entry.get("state") or "").strip().lower()
            if state == "archived":
                continue
            raw_group_count = entry.get("group_count", 0)
            group_count = (
                raw_group_count
                if type(raw_group_count) is int and raw_group_count >= 0
                else 0
            )
            raw_created_at = entry.get("created_at", 0.0)
            created_at = (
                float(raw_created_at)
                if isinstance(raw_created_at, (int, float))
                and not isinstance(raw_created_at, bool)
                else 0.0
            )
            name = SystemBuilder._employee_roster_text(
                entry.get("name"),
                fallback="未命名员工",
                limit=80,
            )
            raw_role = str(entry.get("role") or "").strip()
            role = (
                SystemBuilder._employee_roster_text(
                    raw_role,
                    fallback="未设置职责",
                    limit=100,
                )
                if raw_role
                else f"未设置（发送 /employee-role {name} 职责）"
            )
            normalized.append(
                {
                    "agent_id": SystemBuilder._employee_roster_text(
                        entry.get("agent_id"),
                        fallback="unknown",
                        limit=96,
                    ),
                    "name": name,
                    "emoji": SystemBuilder._employee_roster_text(
                        entry.get("emoji"),
                        fallback="🤖",
                        limit=8,
                    ),
                    "state": state,
                    "state_label": state_labels.get(state, "🔵 状态处理中"),
                    "role": role,
                    "tool": SystemBuilder._employee_roster_code_value(
                        entry.get("tool"),
                        fallback="未设置工具",
                        limit=40,
                    ),
                    "model": SystemBuilder._employee_roster_code_value(
                        entry.get("model"),
                        fallback="默认模型",
                        limit=80,
                    ),
                    "profile": SystemBuilder._employee_roster_code_value(
                        entry.get("profile"),
                        fallback="default",
                        limit=40,
                    ),
                    "effort": SystemBuilder._employee_roster_code_value(
                        entry.get("effort"),
                        fallback="default",
                        limit=40,
                    ),
                    "group_count": group_count,
                    "created_at": created_at,
                }
            )

        normalized.sort(
            key=lambda item: (
                state_order.get(str(item["state"]), 2),
                str(item["name"]).casefold(),
                float(item["created_at"]),
                str(item["agent_id"]),
            )
        )
        active_count = sum(item["state"] == "active" for item in normalized)
        action_count = sum(
            item["state"] == "action_required" for item in normalized
        )
        other_count = len(normalized) - active_count - action_count

        elements: list[dict] = [
            {
                "tag": "markdown",
                "content": (
                    f"**共 {len(normalized)}** · 在职 {active_count} · "
                    f"需处理 {action_count} · 其他生命周期 {other_count}"
                ),
            }
        ]
        if not normalized:
            elements.append(
                CoreBuilder._build_banner_element(
                    "暂无数字员工",
                    type="info",
                )
            )
        else:
            elements.append({"tag": "hr"})
            for index, item in enumerate(normalized):
                backend = SystemBuilder._employee_roster_inline_code(
                    f"{item['tool']}/{item['model']}"
                )
                configuration = SystemBuilder._employee_roster_inline_code(
                    f"{item['profile']}/{item['effort']}"
                )
                elements.append(
                    {
                        "tag": "markdown",
                        "content": (
                            f"{item['emoji']} **{item['name']}** · {item['state_label']}\n"
                            f"职责 {item['role']} · 后端 {backend}\n"
                            f"配置 {configuration} · "
                            f"协作群 {item['group_count']}"
                        ),
                    }
                )
                if index != len(normalized) - 1:
                    elements.append({"tag": "hr"})

        elements.extend(
            [
                {"tag": "hr"},
                {
                    "tag": "markdown",
                    "text_size": "notation",
                    "content": "生命周期状态来自 Journal 权威投影，不代表 Channel 在线状态。",
                },
            ]
        )
        card = CoreBuilder._wrap_card(
            "👥 数字员工目录",
            "blue",
            elements,
            subtitle=f"当前员工 · {len(normalized)}",
        )
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_tools_list_card(
        tools: list[dict],
        project: Optional[ProjectContext] = None,
    ) -> tuple[str, str]:
        """Build a card showing all available ACP tools."""
        elements = []

        if project:
            elements.append(CoreBuilder._build_directory_element(project))
            elements.append({"tag": "hr"})

        elements.append({"tag": "markdown", "content": UI_TEXT["system_tools_list_header"]})

        # Add tool buttons
        buttons = []
        for tool in tools:
            tool_name = tool["name"]
            emoji = tool.get("emoji", "🤖")
            description = tool.get("description", "")
            is_available = tool.get("available", False)

            btn_text = f"{emoji} {tool_name.capitalize()}"
            if description:
                btn_text += f" ({description})"
            if not is_available:
                btn_text += " ⚠️"

            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": btn_text},
                    "type": "primary" if is_available else "default",
                    "disabled": not is_available,
                    "value": {"action": f"enter_{tool_name}", "project_id": project.project_id if project else None},
                }
            )

        elements.extend(build_responsive_layout(buttons))

        # Add status indicator
        available_count = sum(1 for t in tools if t.get("available", False))
        elements.append({"tag": "hr"})
        elements.append(
            {
                "tag": "markdown",
                "text_size": "notation",
                "content": UI_TEXT["system_tools_list_footer"].format(
                    available=available_count,
                    total=len(tools)
                ),
            }
        )

        card = CoreBuilder._wrap_card(UI_TEXT["system_tools_list_title"], "blue", elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_tools_status_card(
        tools: list[dict],
        active_sessions: dict[str, dict] = None,
        project: Optional[ProjectContext] = None,
    ) -> tuple[str, str]:
        """Build a card showing detailed status of all tools."""
        active_sessions = active_sessions or {}
        elements = []

        if project:
            elements.append(CoreBuilder._build_directory_element(project))
            elements.append({"tag": "hr"})

        elements.append({"tag": "markdown", "content": UI_TEXT["system_tools_status_header"]})

        for tool in tools:
            tool_name = tool["name"]
            emoji = tool.get("emoji", "🤖")
            is_available = tool.get("available", False)
            last_used = tool.get("last_used", UI_TEXT["system_never_used"])

            status_text = UI_TEXT["system_status_available"] if is_available else UI_TEXT["system_status_unavailable"]
            active_info = ""
            if tool_name in active_sessions:
                session_info = active_sessions[tool_name]
                active_info = UI_TEXT["system_tools_status_active_session"].format(
                    chat_id=session_info.get('chat_id', 'N/A')
                )

            elements.append(
                {
                    "tag": "markdown",
                    "content": UI_TEXT["system_tools_status_item"].format(
                        emoji=emoji,
                        name=tool_name.capitalize(),
                        status=status_text,
                        last_used=last_used,
                        active_info=active_info
                    ),
                }
            )

        elements.append({"tag": "hr"})

        # Quick actions
        action_buttons = []
        for tool in tools:
            if tool.get("available", False):
                action_buttons.append(
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": UI_TEXT["system_tools_status_btn_enter"].format(name=tool['name'].capitalize())},
                        "type": "default",
                        "value": {"action": f"enter_{tool['name']}", "project_id": project.project_id if project else None},
                    }
                )

        if action_buttons:
            elements.extend(build_responsive_layout(action_buttons))

        card = CoreBuilder._wrap_card(UI_TEXT["system_tools_status_title"], "blue", elements)
        return "interactive", json.dumps(card, ensure_ascii=False)
