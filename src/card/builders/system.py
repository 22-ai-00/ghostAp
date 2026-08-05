from __future__ import annotations

import json
import logging
import math
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from src.card.error_diagnostics import register_error_diagnostic
from src.model_selection import DEFAULT_MODEL_OPTION_VALUE
from src.utils.errors import GhostAPError, get_error_detail

from ..actions import dispatch as action_ids
from ..models import ModelOptionView, ToolOptionView
from ..shared import build_responsive_layout
from ..themes import PANEL_STYLES
from ..thresholds import THRESHOLDS
from ..ui_text import UI_TEXT
from .core import CoreBuilder
from .lock import build_lock_help_body

logger = logging.getLogger(__name__)

_SELECT_LABEL_MOBILE_LIMIT = 72
_BUTTON_LABEL_MOBILE_LIMIT = 40

# Cap how many concrete model buttons render on a single ACP model-select card.
# Providers like Traex expose ~90 models; rendering them all at once overflows
# Feishu's per-card element limit (ErrCode 11310). Larger lists are paginated so
# every model stays selectable across all modes that share this card.
_MAX_ACP_MODEL_BUTTONS = 20

if TYPE_CHECKING:
    from src.project.context import ProjectContext
    from src.sandbox.executor import ExecutionResult

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
        "original_mode",
        "retry_mode",
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
    def _select_option(label: str, value: str) -> dict:
        """Build a Feishu select option with consistent plain-text shape."""
        return {"text": {"tag": "plain_text", "content": SystemBuilder._mobile_safe_label(label)}, "value": value}

    @staticmethod
    def _mobile_safe_label(label: object, *, limit: int = _SELECT_LABEL_MOBILE_LIMIT) -> str:
        """Keep select labels compact enough for mobile Feishu cards."""
        text = str(label or "")
        if len(text) <= limit:
            return text
        return text[: max(1, limit - 1)].rstrip() + "…"

    @staticmethod
    def _label_with_optional_description(name: object, description: object = "") -> str:
        """Format a select/button label without duplicating empty descriptions."""
        label = str(name or "")
        desc = str(description or "").strip()
        if desc:
            label += f" ({desc})"
        return label

    @staticmethod
    def _mobile_safe_button_label(label: object) -> str:
        """Keep button labels short enough for Feishu mobile card columns."""
        return SystemBuilder._mobile_safe_label(label, limit=_BUTTON_LABEL_MOBILE_LIMIT)

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
    def _build_select_static(
        *,
        placeholder_key: str,
        action: str,
        options: list[dict],
        initial_option: Optional[str] = None,
        value_extra: Optional[dict] = None,
    ) -> dict:
        """Build a select_static element used by TTADK selection cards."""
        value = {"action": action}
        if value_extra:
            value.update(value_extra)
        return {
            "tag": "select_static",
            "placeholder": {"tag": "plain_text", "content": UI_TEXT[placeholder_key]},
            "initial_option": initial_option,
            "value": value,
            "options": options,
        }

    @staticmethod
    def _wrap_system_card(title: str, elements: list[dict], *, template: str = "blue") -> tuple[str, str]:
        """Wrap system-card elements with the standard interactive response tuple."""
        card = CoreBuilder._wrap_card(title, template, elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

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
    def build_ttadk_refresh_result_card(
        tool: str,
        result: any,
    ) -> tuple[str, str]:
        """Build a summary message for TTADK refresh result."""
        lines = [UI_TEXT["system_ttadk_refresh_success"]]
        if tool:
            lines.append(UI_TEXT["system_ttadk_refresh_label_tool"].format(tool=tool))

        source = getattr(result, "source", "")
        if source:
            lines.append(UI_TEXT["system_ttadk_refresh_label_source"].format(source=source))

        warnings = getattr(result, "warnings", None)
        if warnings:
            lines.append(
                UI_TEXT["system_ttadk_refresh_label_warning"].format(
                    warnings="; ".join(warnings)
                )
            )

        diagnostics = getattr(result, "diagnostics", None)
        if diagnostics:
            try:
                attempts = (diagnostics or {}).get("attempts")
                if attempts:
                    lines.append(
                        UI_TEXT["system_ttadk_refresh_label_diag"].format(
                            attempts=attempts
                        )
                    )
            except Exception:
                logger.debug("failed to build TTADK refresh card diagnostics", exc_info=True)

        footer = UI_TEXT["system_ttadk_refresh_footer"]
        if footer:
            lines.append(footer)

        return "text", "\n".join(lines)

    @staticmethod
    def build_switching_status_card(
        tool: str,
        model: str,
    ) -> tuple[str, str]:
        """Build a simple status message for switching tools/models."""
        msg = UI_TEXT["system_switching_to"].format(
            tool=tool, model=model
        )
        return "text", msg

    @staticmethod
    def build_ttadk_info_content(
        current_tool: Optional[str],
        current_model: Optional[str],
        tool_desc: dict[str, str],
        model_desc: dict[str, str],
    ) -> str:
        """Build the Markdown content for TTADK status info."""
        lines = [UI_TEXT["system_ttadk_info_header"]]

        if current_tool:
            tool_label = tool_desc.get(current_tool, UI_TEXT["system_ttadk_ai_tool_label"])
            lines.append(f"{UI_TEXT['system_label_current_tool']}: `{current_tool}` - {tool_label}")
        else:
            lines.append(f"{UI_TEXT['system_label_current_tool']}: " + UI_TEXT["system_not_set"])

        if current_model:
            model_label = model_desc.get(current_model, current_model)
            lines.append(f"{UI_TEXT['system_label_current_model']}: `{current_model}` - {model_label}")
        else:
            lines.append(f"{UI_TEXT['system_label_current_model']}: " + UI_TEXT["system_not_set"])

        footer = UI_TEXT["system_ttadk_info_footer"]
        if footer:
            lines.append(footer)

        return "\n".join(lines)

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

        elements = []
        if project:
            elements.append(CoreBuilder._build_directory_element(project))
            elements.append({"tag": "hr"})

        elements.append(
            CoreBuilder._build_content_element(
                f"{severity_label}\n{severity_hint}\n\n"
                f"❌ **错误摘要**\n{message}\n\n"
                f"**错误场景**\n{title}\n\n"
                f"**当前状态**\n{SystemBuilder._current_status_text(severity, continue_action, context)}\n\n"
                f"{UI_TEXT['card_lifecycle_details_collapsed']}"
            )
        )

        # project info is handled by project_response_card if needed, but build_error_card
        # is often used for generic errors. Original code had optional project.
        # We'll stick to a simpler interactive card here or wrap it.

        if severity == "degraded":
            degraded_mode = SystemBuilder._resolve_degraded_mode(continue_action, context)
            if degraded_mode:
                primary_continue = SystemBuilder._callback_button(
                    text=UI_TEXT["card_lifecycle_degraded_primary"].format(
                        mode=SystemBuilder._display_mode_label(degraded_mode)
                    ),
                    action=SystemBuilder._build_degraded_continue_action(continue_action, context),
                    button_type="primary",
                )
                elements.extend(build_responsive_layout([primary_continue]))

            secondary_buttons = SystemBuilder._build_degraded_secondary_buttons(
                detail_action=detail_action,
                continue_action=continue_action,
                retry_action=retry_action,
                context=context,
                title=title,
                summary=message,
                details=details,
            )
            elements.extend(build_responsive_layout(secondary_buttons, layout="mobile"))

            # 降级卡只保留一个主决策和两个次级决策。异常自带 quick_actions
            # 不再追加成第三组同级按钮，避免稀释“继续使用目标模式”的主决策。
        elif quick_actions:
            buttons = build_quick_actions(quick_actions, context)
            elements.extend(build_responsive_layout(buttons))
        else:
            buttons = []
            if detail_action:
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

        card = CoreBuilder._wrap_card(UI_TEXT["system_error_prompt_title"], header_template, elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def _build_degraded_continue_action(
        continue_action: Optional[dict],
        context: Optional[dict] = None,
    ) -> dict:
        action = SystemBuilder._safe_action_payload(context)
        action.update(SystemBuilder._safe_action_payload(continue_action))
        action["action"] = action_ids.CONTINUE_DEGRADED
        degraded_mode = SystemBuilder._resolve_degraded_mode(continue_action, context)
        if degraded_mode:
            action.setdefault("degraded_to", degraded_mode)
        return action

    @staticmethod
    def _build_degraded_secondary_buttons(
        *,
        detail_action: Optional[dict],
        continue_action: Optional[dict],
        retry_action: Optional[dict],
        context: Optional[dict],
        title: str,
        summary: str,
        details: Optional[str],
    ) -> list[dict]:
        safe_context = SystemBuilder._safe_action_payload(context)
        detail_payload = SystemBuilder._build_detail_action(
            detail_action,
            title=title,
            summary=summary,
            details=details,
            context=safe_context,
        )
        retry_payload = {**safe_context, **SystemBuilder._safe_action_payload(retry_action)}
        retry_payload["action"] = str(retry_payload.get("action") or action_ids.RETRY_ORIGINAL)
        retry_payload.pop("mode", None)
        buttons = [
            SystemBuilder._callback_button(
                text=UI_TEXT["card_lifecycle_show_details"],
                action=detail_payload,
                button_type="default",
            )
        ]
        if SystemBuilder._has_complete_retry_original_payload(retry_payload):
            buttons.append(
                SystemBuilder._callback_button(
                    text=UI_TEXT["card_lifecycle_retry_original"],
                    action=retry_payload,
                    button_type="default",
                )
            )
        return buttons

    @staticmethod
    def _has_complete_retry_original_payload(payload: dict) -> bool:
        if str(payload.get("action") or "") != action_ids.RETRY_ORIGINAL:
            return False
        return all(str(payload.get(field) or "").strip() for field in ("original_mode", "retry_mode", "degraded_to"))

    @staticmethod
    def _safe_degraded_context(context: Optional[dict]) -> dict:
        return SystemBuilder._safe_action_payload(context)

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
    def _current_status_text(severity: str, continue_action: Optional[dict], context: Optional[dict]) -> str:
        if severity == "degraded":
            mode = SystemBuilder._resolve_degraded_mode(continue_action, context)
            if mode:
                return f"可继续使用 {SystemBuilder._display_mode_label(mode)}，或查看脱敏诊断后再决定是否重试原模式。"
            return "当前暂未确定可继续模式；请重新发送原命令，或查看脱敏诊断后再决定是否重试。"
        if severity == "recoverable":
            return "可查看脱敏诊断，也可以按卡片按钮重试。"
        return "当前操作已停止；可查看脱敏诊断并按提示重新发起。"

    @staticmethod
    def _display_mode_label(mode: str) -> str:
        labels = {
            "coco": "Coco",
            "claude": "Claude",
            "claude cli": "Claude CLI",
            "aiden": "Aiden",
            "codex": "Codex",
            "gemini": "Gemini",
            "ttadk": "TTADK",
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
        result: "ExecutionResult",
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
    def _build_ttadk_yolo_toggle_button(
        yolo_enabled: bool, project_id: Optional[str], view: str, tool_name: str = ""
    ) -> dict:
        enabled = bool(yolo_enabled)
        label = UI_TEXT["system_ttadk_yolo_on"] if enabled else UI_TEXT["system_ttadk_yolo_off"]
        return {
            "tag": "button",
            "text": {"tag": "plain_text", "content": label},
            "type": "primary" if enabled else "default",
            "value": {
                "action": "toggle_ttadk_yolo",
                "enabled": not enabled,
                "project_id": project_id,
                "view": view,
                "tool_name": tool_name,
            },
        }

    @staticmethod
    def build_ttadk_tool_select_card(
        tools: list, project_id: Optional[str] = None, yolo_enabled: bool = False, current_tool: Optional[str] = None
    ) -> tuple[str, str]:
        elements = [{"tag": "markdown", "content": UI_TEXT["system_ttadk_select_tool_prompt"]}]

        elements.extend(
            build_responsive_layout(
                [SystemBuilder._build_ttadk_yolo_toggle_button(yolo_enabled, project_id, "tool_select")]
            )
        )
        elements.append({"tag": "hr"})

        options = [
            SystemBuilder._select_option(
                SystemBuilder._label_with_optional_description(tool.name, getattr(tool, "description", "")),
                tool.name,
            )
            for tool in tools
        ]

        elements.append(
            SystemBuilder._build_select_static(
                placeholder_key="system_ttadk_select_tool_placeholder",
                action="select_ttadk_tool",
                options=options,
                initial_option=current_tool,
                value_extra={"project_id": project_id},
            )
        )

        return SystemBuilder._wrap_system_card(UI_TEXT["system_ttadk_tool_select_title"], elements)

    @staticmethod
    def build_ttadk_model_select_card(
        models: list, tool_name: str, project_id: Optional[str] = None, yolo_enabled: bool = False, current_model: Optional[str] = None
    ) -> tuple[str, str]:
        elements = [
            {
                "tag": "markdown",
                "content": (
                    UI_TEXT["system_ttadk_select_model_prompt"].format(tool=tool_name) + "\n" +
                    UI_TEXT["system_ttadk_select_model_hint"]
                ),
            }
        ]

        elements.extend(
            build_responsive_layout(
                [SystemBuilder._build_ttadk_yolo_toggle_button(yolo_enabled, project_id, "model_select", tool_name)]
            )
        )
        elements.append({"tag": "hr"})

        options = [
            SystemBuilder._select_option(
                SystemBuilder._label_with_optional_description(model.name, getattr(model, "description", "")),
                model.name,
            )
            for model in models
        ]

        elements.append(
            SystemBuilder._build_select_static(
                placeholder_key="system_ttadk_select_model_placeholder",
                action="select_ttadk_model",
                options=options,
                initial_option=current_model,
                value_extra={"tool_name": tool_name, "project_id": project_id},
            )
        )

        # 辅助入口：强制刷新模型列表（常用于 Invalid model / 可用模型为空）
        elements.append({"tag": "hr"})
        elements.extend(
            build_responsive_layout(
                [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": UI_TEXT["system_ttadk_refresh_btn"]},
                        "type": "primary",
                        "value": {
                            "action": "refresh_ttadk_models",
                            "tool_name": tool_name,
                            "project_id": project_id,
                        },
                    }
                ]
            )
        )

        return SystemBuilder._wrap_system_card(UI_TEXT["system_ttadk_model_select_title"].format(tool=tool_name), elements)

    @staticmethod
    def build_ttadk_combined_select_card(
        tools: list,
        models_by_tool: dict,
        project_id: Optional[str] = None,
        yolo_enabled: bool = False,
        current_tool: Optional[str] = None,
        current_model: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build a combined TTADK tool + model selection card (single step)."""
        elements = [{"tag": "markdown", "content": UI_TEXT["system_ttadk_combined_select_prompt"]}]

        elements.extend(
            build_responsive_layout(
                [SystemBuilder._build_ttadk_yolo_toggle_button(yolo_enabled, project_id, "combined_select")]
            )
        )
        elements.append({"tag": "hr"})

        # Tool selection
        tool_options = [
            SystemBuilder._select_option(
                SystemBuilder._label_with_optional_description(tool.name, getattr(tool, "description", "")),
                tool.name,
            )
            for tool in tools
        ]

        elements.append({"tag": "markdown", "content": UI_TEXT["system_ttadk_label_tool"]})
        elements.append(
            SystemBuilder._build_select_static(
                placeholder_key="system_ttadk_select_tool_placeholder",
                action="select_ttadk_combined_tool",
                options=tool_options,
                initial_option=current_tool,
                value_extra={"project_id": project_id},
            )
        )

        # Model selection per tool (show current_tool's models or first tool's models by default)
        if tools and models_by_tool:
            selected_tool = current_tool
            if not selected_tool or selected_tool not in models_by_tool:
                selected_tool = tools[0].name
            models = models_by_tool.get(selected_tool, [])
            if models:
                elements.append({"tag": "hr"})
                elements.append({"tag": "markdown", "content": UI_TEXT["system_ttadk_label_model"].format(tool=selected_tool)})

                model_options = [
                    SystemBuilder._select_option(
                        SystemBuilder._label_with_optional_description(model.name, getattr(model, "description", "")),
                        model.name,
                    )
                    for model in models
                ]

                elements.append(
                    SystemBuilder._build_select_static(
                        placeholder_key="system_ttadk_select_model_placeholder",
                        action="select_ttadk_combined",
                        options=model_options,
                        initial_option=current_model,
                        value_extra={"tool_name": selected_tool, "project_id": project_id},
                    )
                )

        # Refresh button
        elements.append({"tag": "hr"})
        elements.extend(
            build_responsive_layout(
                [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": UI_TEXT["system_ttadk_refresh_btn"]},
                        "type": "primary",
                        "value": {
                            "action": "refresh_ttadk_models",
                            "project_id": project_id,
                        },
                    }
                ]
            )
        )

        return SystemBuilder._wrap_system_card(UI_TEXT["system_ttadk_combined_title"], elements)

    @staticmethod
    def build_ttadk_soft_failure_card(
        message: str,
        project_id: Optional[str] = None,
        *,
        action: str = "show_ttadk_menu",
        button_text: str = "",
    ) -> tuple[str, str]:
        # Clean the message for the banner (CoreBuilder adds its own emoji)
        banner_msg = message.replace("⚠️ ", "").strip()
        elements = [
            CoreBuilder._build_banner_element(banner_msg, type="warning")
        ]

        effective_text = button_text or UI_TEXT["system_ttadk_btn_reenter"]
        button = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": effective_text},
            "type": "primary",
            "value": {"action": action, "project_id": project_id},
        }
        elements.extend(build_responsive_layout([button]))

        card = CoreBuilder._wrap_card(UI_TEXT["system_ttadk_unavailable_title"], "blue", elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def _format_ttadk_soft_failure_message(reason: str) -> str:
        cleaned = str(reason or "").strip()
        if not cleaned:
            cleaned = UI_TEXT["system_ttadk_unavailable"]
        return UI_TEXT["system_ttadk_soft_failure_msg"].format(reason=cleaned)

    @staticmethod
    def build_ttadk_soft_failure_card_for(
        reason: str,
        project_id: Optional[str] = None,
        *,
        action: str = "show_ttadk_menu",
        button_text: str = "",
    ) -> tuple[str, str]:
        message = SystemBuilder._format_ttadk_soft_failure_message(reason)
        return SystemBuilder.build_ttadk_soft_failure_card(
            message,
            project_id,
            action=action,
            button_text=button_text or UI_TEXT["system_ttadk_btn_continue"],
        )

    @staticmethod
    def build_acp_tool_select_card(
        tools: list,
        project_id: Optional[str] = None,
        current_tool: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build an interactive card for ACP tool selection.

        ``tools`` 可以是：
        - ``ToolOptionView`` 列表（首选，卡片层通用视图模型）；
        - 旧版的字符串列表或带 ``name`` 属性的对象列表（向后兼容）。
        """

        elements = [{"tag": "markdown", "content": UI_TEXT["system_acp_select_tool_prompt"]}]

        tool_options: list[ToolOptionView] = []
        for item in tools or []:
            if isinstance(item, ToolOptionView):
                tool_options.append(item)
                continue

            name = getattr(item, "name", None) or str(item)
            name = str(name or "").strip()
            if not name:
                continue

            # 优先使用 UI_TEXT 中的描述，保持既有文案；否则回退到对象上的 description 字段
            desc_key = f"system_acp_tool_desc_{name}"
            desc = UI_TEXT[desc_key] if desc_key in UI_TEXT else getattr(item, "description", "")
            emoji = getattr(item, "emoji", "🤖")
            is_default = bool(getattr(item, "is_default", False))
            disabled = bool(getattr(item, "disabled", False))

            tool_options.append(
                ToolOptionView(
                    name=name,
                    description=str(desc or ""),
                    is_default=is_default,
                    emoji=str(emoji or "🤖"),
                    disabled=disabled,
                )
            )

        buttons = []
        for t in tool_options:
            btn_text = f"{t.emoji} {t.name}"
            if t.description:
                btn_text += f" ({t.description})"
            btn_text = SystemBuilder._mobile_safe_button_label(btn_text)

            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": btn_text},
                    "type": "primary" if t.name == current_tool else "default",
                    "disabled": bool(t.disabled),
                    "value": {
                        "action": "select_acp_tool",
                        "tool_name": t.name,
                        "project_id": project_id,
                    },
                }
            )

        elements.extend(build_responsive_layout(buttons))

        card = CoreBuilder._wrap_card(UI_TEXT["system_acp_tool_select_title"], "blue", elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_tui2acp_adapter_select_card(
        adapters: list[dict],
        project_id: Optional[str] = None,
        current_adapter: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build an interactive card for tui2acp adapter selection."""

        elements = [{"tag": "markdown", "content": "**选择要通过 tui2acp 桥接的 AI 工具：**"}]

        buttons = []
        for adapter in adapters or []:
            name = adapter.get("name", "")
            emoji = adapter.get("emoji", "🤖")
            desc = adapter.get("description", "")
            btn_text = f"{emoji} {name}"
            if desc:
                btn_text += f" ({desc})"
            btn_text = SystemBuilder._mobile_safe_button_label(btn_text)

            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": btn_text},
                    "type": "primary" if name == current_adapter else "default",
                    "value": {
                        "action": "select_tui2acp_adapter",
                        "adapter_name": name,
                        "project_id": project_id,
                    },
                }
            )

        elements.extend(build_responsive_layout(buttons))

        # Custom command input form
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**其他工具：** 输入启动命令直接进入 TUI2ACP"})
        elements.append({
            "tag": "form",
            "name": "tui2acp_custom_command_form",
            "elements": [
                {
                    "tag": "input",
                    "name": "custom_command",
                    "placeholder": {
                        "tag": "plain_text",
                        "content": "例如: aider --model gpt-4o",
                    },
                    "max_length": 200,
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "🚀 启动"},
                    "type": "primary",
                    "action_type": "form_action",
                    "name": "custom_command_submit",
                    "value": {
                        "action": "select_tui2acp_custom_command",
                        "project_id": project_id,
                    },
                },
            ],
        })

        card = CoreBuilder._wrap_card("🌉 Tui2ACP 工具选择", "purple", elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_slock_role_tool_select_card(
        role_name: str,
        tools: list[dict],
        project_id: Optional[str] = None,
        *,
        value_extra: Optional[dict] = None,
    ) -> tuple[str, str]:
        """Build the first step of `/new-role`: choose the backing tool."""

        role = str(role_name or "").strip()
        elements = [
            {
                "tag": "markdown",
                "content": f"为 Slock 角色 **{role}** 选择工具和后续模型。",
            }
        ]

        buttons = []
        for tool in tools or []:
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            label = str(tool.get("label") or name)
            emoji = str(tool.get("emoji") or "🤖")
            desc = str(tool.get("description") or "").strip()
            text = f"{emoji} {label}"
            if desc:
                text += f" ({desc})"
            buttons.append(
                SystemBuilder._callback_button(
                    text=SystemBuilder._mobile_safe_button_label(text),
                    button_type="primary" if name == "coco" else "default",
                    action={
                        **dict(value_extra or {}),
                        "action": action_ids.SLOCK_NEW_ROLE_SELECT_TOOL,
                        "role_name": role,
                        "tool_name": name,
                        "project_id": project_id,
                    },
                )
            )

        elements.extend(build_responsive_layout(buttons))
        card = CoreBuilder._wrap_card("Slock 角色工具选择", "blue", elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_slock_role_unified_select_card(
        role_name: str,
        tools: list[dict],
        models: list,
        tool_name: str,
        project_id: Optional[str] = None,
        current_model: Optional[str] = None,
        *,
        pending_group: Optional[str] = None,
        pending_profile: Optional[str] = None,
        pending_effort: Optional[str] = None,
        profile_markdown: Optional[str] = None,
        value_extra: Optional[dict] = None,
    ) -> tuple[str, str]:
        """Build a unified /hire card with tool dropdown + model cascade."""

        from ..render.model_cascade import build_model_cascade_elements, has_cascade_variants

        role = str(role_name or "").strip()
        merged_extra = dict(value_extra or {})
        merged_extra["role_name"] = role

        # Tool dropdown
        tool_options = []
        initial_tool_option = None
        for tool in tools or []:
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            emoji = str(tool.get("emoji") or "🤖")
            label = str(tool.get("label") or name)
            desc = str(tool.get("description") or "")
            display = f"{emoji} {label}" + (f" ({desc})" if desc else "")
            opt = {"text": {"tag": "plain_text", "content": display}, "value": name}
            tool_options.append(opt)
            if name == tool_name:
                initial_tool_option = opt

        tool_dropdown_value = {
            "action": action_ids.SLOCK_NEW_ROLE_SELECT_TOOL_DROPDOWN,
            "project_id": project_id,
            **merged_extra,
        }
        tool_dropdown = {
            "tag": "select_static",
            "placeholder": {"tag": "plain_text", "content": "选择工具"},
            "options": tool_options,
            "value": tool_dropdown_value,
        }
        if initial_tool_option:
            tool_dropdown["initial_option"] = initial_tool_option["value"]

        existing_app_id = str(merged_extra.get("existing_app_id") or "").strip()
        app_context = (
            f"\n\n复用已有飞书应用：`{existing_app_id}`"
            if re.fullmatch(r"cli_[A-Za-z0-9_-]{3,128}", existing_app_id)
            else ""
        )
        elements: list[dict] = [
            {
                "tag": "markdown",
                "content": f"为员工 **{role}** 选择工具和模型{app_context}",
            },
            *(
                [{"tag": "markdown", "content": profile_markdown}]
                if profile_markdown
                else []
            ),
            tool_dropdown,
            {"tag": "hr"},
        ]

        # Normalize models for cascade
        norm_models: list[dict] = []
        for model in models or []:
            if isinstance(model, dict):
                m_name = str(model.get("name") or "").strip()
                m_disp = model.get("display_name") or m_name
                m_desc = model.get("description", "")
                reasoning_efforts = tuple(model.get("reasoning_efforts") or ())
                adapted_reasoning_effort = model.get("adapted_reasoning_effort")
                is_default = bool(model.get("is_default"))
                raw_selection_variants = list(model.get("selection_variants") or [])
            else:
                m_name = str(getattr(model, "name", None) or model or "").strip()
                m_disp = getattr(model, "display_name", None) or getattr(model, "friendly_name", None) or m_name
                m_desc = getattr(model, "description", "")
                reasoning_efforts = tuple(getattr(model, "reasoning_efforts", ()) or ())
                adapted_reasoning_effort = getattr(model, "adapted_reasoning_effort", None)
                is_default = bool(getattr(model, "is_default", False))
                raw_selection_variants = list(getattr(model, "selection_variants", ()) or ())
            if not m_name:
                continue
            selection_variants = []
            for raw_variant in raw_selection_variants:
                if isinstance(raw_variant, dict):
                    selection_variants.append(dict(raw_variant))
                else:
                    selection_variants.append({
                        "name": str(getattr(raw_variant, "name", "") or ""),
                        "profile": str(getattr(raw_variant, "profile", "") or ""),
                        "effort": str(getattr(raw_variant, "effort", "default") or "default"),
                        "display_name": str(getattr(raw_variant, "display_name", "") or ""),
                        "is_variant_default": bool(getattr(raw_variant, "is_variant_default", False)),
                    })
            norm_models.append({
                "name": m_name,
                "display_name": str(m_disp or m_name),
                "description": str(m_desc or ""),
                "reasoning_efforts": reasoning_efforts,
                "adapted_reasoning_effort": adapted_reasoning_effort,
                "is_default": is_default,
                "selection_variants": selection_variants,
            })

        def _value_builder(action: str, extra: dict) -> dict:
            value = {
                "action": action,
                "tool_name": tool_name,
                "project_id": project_id,
            }
            value.update(extra)
            value.update(merged_extra)
            return value

        if has_cascade_variants(norm_models):
            elements.append({"tag": "markdown", "content": f"员工: **{role}** · 选择模型和思考深度"})
            elements.extend(
                build_model_cascade_elements(
                    models=norm_models,
                    value_builder=_value_builder,
                    group_action=action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_GROUP,
                    profile_action=action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_PROFILE,
                    effort_action=action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_EFFORT,
                    select_action=action_ids.SLOCK_NEW_ROLE_SELECT_MODEL,
                    default_action=action_ids.SLOCK_NEW_ROLE_SELECT_MODEL,
                    pending_group=pending_group,
                    pending_profile=pending_profile,
                    pending_effort=pending_effort,
                    current_model=current_model,
                    button_row_builder=build_responsive_layout,
                )
            )
        else:
            # Simple model list as buttons
            model_buttons: list[dict] = []
            for model in norm_models[:20]:
                btn_value = _value_builder(action_ids.SLOCK_NEW_ROLE_SELECT_MODEL, {"_option": model["name"]})
                btn_value["model_name"] = model["name"]
                model_buttons.append({
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": model["display_name"]},
                    "type": "default",
                    "value": btn_value,
                    "behaviors": [{"type": "callback", "value": btn_value}],
                })
            elements.extend(build_responsive_layout(model_buttons))

        card = CoreBuilder._wrap_card("Slock 员工创建", "blue", elements)
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_model_select_card(
        models: list,
        tool_name: str,
        project_id: Optional[str] = None,
        current_model: Optional[str] = None,
        thread_root_id: Optional[str] = None,
        *,
        action_name: str = action_ids.SELECT_ACP_MODEL,
        value_extra: Optional[dict] = None,
        context_markdown: Optional[str] = None,
        refresh_action_name: Optional[str] = action_ids.REFRESH_ACP_MODELS,
        model_page: int = 0,
    ) -> tuple[str, str]:
        """Build an interactive card for ACP model selection.

        ``models`` 可以是 ModelOptionView 或任意带 ``name``/``description`` 属性的对象。

        模型数量较多时（例如 Traex 约 90 个）按 ``_MAX_ACP_MODEL_BUTTONS`` 分页，
        翻页按钮复用 ``refresh_action_name`` 并携带 ``model_page``，避免一次性渲染
        所有按钮触发飞书卡片元素上限（ErrCode 11310）。"""

        elements = [
            {
                "tag": "markdown",
                "content": UI_TEXT["system_acp_select_model_prompt"].format(tool=tool_name),
            }
        ]
        if context_markdown:
            elements.append({"tag": "markdown", "content": context_markdown})

        items: list[ModelOptionView] = []
        for model in models or []:
            if isinstance(model, ModelOptionView):
                items.append(model)
                continue

            selection_variants = list(
                getattr(model, "selection_variants", ()) or ()
            )
            if selection_variants:
                model_default = bool(getattr(model, "is_default", False))
                model_description = str(
                    getattr(model, "description", "") or ""
                )
                for variant in selection_variants:
                    variant_name = str(
                        getattr(variant, "name", "") or ""
                    ).strip()
                    if not variant_name:
                        continue
                    variant_display = str(
                        getattr(variant, "display_name", "")
                        or variant_name
                    )
                    items.append(
                        ModelOptionView(
                            name=variant_name,
                            description=variant_display or model_description,
                            is_default=bool(
                                model_default
                                and getattr(
                                    variant,
                                    "is_variant_default",
                                    False,
                                )
                            ),
                            display_name=variant_display,
                        )
                    )
                continue

            m_name = getattr(model, "name", None) or str(model)
            m_name = str(m_name or "").strip()
            if not m_name:
                continue
            m_desc = getattr(model, "description", "")
            display = getattr(model, "display_name", None) or getattr(model, "friendly_name", None)

            items.append(
                ModelOptionView(
                    name=m_name,
                    description=str(m_desc or ""),
                    is_default=bool(getattr(model, "is_default", False)),
                    display_name=str(display) if display is not None else None,
                )
            )

        def _selection_value(model_name: str, *, use_default_model: bool = False) -> dict:
            value = {
                "action": action_name,
                "tool_name": tool_name,
                "model_name": model_name,
                "project_id": project_id,
                "thread_root_id": thread_root_id,
            }
            if use_default_model:
                value["use_default_model"] = True
            if value_extra:
                value.update(value_extra)
            return value

        buttons = [
            {
                "tag": "button",
                "text": {
                    "tag": "plain_text",
                    "content": SystemBuilder._mobile_safe_button_label(
                        UI_TEXT["system_acp_default_model_option"]
                    ),
                },
                "type": "primary" if not current_model else "default",
                "value": _selection_value(DEFAULT_MODEL_OPTION_VALUE, use_default_model=True),
                "behaviors": [
                    {"type": "callback", "value": _selection_value(DEFAULT_MODEL_OPTION_VALUE, use_default_model=True)}
                ],
            }
        ]

        total_models = len(items)
        total_pages = max(1, math.ceil(total_models / _MAX_ACP_MODEL_BUTTONS))
        page = min(max(0, int(model_page or 0)), total_pages - 1)
        start = page * _MAX_ACP_MODEL_BUTTONS
        end = start + _MAX_ACP_MODEL_BUTTONS
        if total_pages > 1:
            elements.append(
                {
                    "tag": "markdown",
                    "content": (
                        f"_模型 {start + 1}-{min(end, total_models)} / {total_models}"
                        f" · 第 {page + 1}/{total_pages} 页_"
                    ),
                }
            )

        for m in items[start:end]:
            label = m.display_name or m.name
            btn_text = f"{label}"
            if m.description and m.description != label:
                btn_text += f" ({m.description})"
            btn_text = SystemBuilder._mobile_safe_button_label(btn_text)

            buttons.append(
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": btn_text},
                    "type": "primary" if m.name == current_model else "default",
                    "value": _selection_value(m.name),
                    "behaviors": [{"type": "callback", "value": _selection_value(m.name)}],
                }
            )

        elements.extend(build_responsive_layout(buttons))

        # Pagination navigation (only when there is more than one page). Page
        # buttons reuse the refresh action so the handler re-renders the card
        # at the requested page; they are no-ops when refresh is disabled.
        if total_pages > 1 and refresh_action_name:
            def _page_value(target_page: int) -> dict:
                value = {
                    "action": refresh_action_name,
                    "tool_name": tool_name,
                    "project_id": project_id,
                    "thread_root_id": thread_root_id,
                    "model_page": target_page,
                }
                if value_extra:
                    value.update(value_extra)
                return value

            nav_buttons: list[dict] = []
            if page > 0:
                prev_value = _page_value(page - 1)
                nav_buttons.append(
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "上一页"},
                        "type": "default",
                        "value": prev_value,
                        "behaviors": [{"type": "callback", "value": prev_value}],
                    }
                )
            if page + 1 < total_pages:
                next_value = _page_value(page + 1)
                nav_buttons.append(
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "下一页"},
                        "type": "default",
                        "value": next_value,
                        "behaviors": [{"type": "callback", "value": next_value}],
                    }
                )
            if nav_buttons:
                elements.extend(build_responsive_layout(nav_buttons))

        if refresh_action_name:
            refresh_value = {
                "action": refresh_action_name,
                "tool_name": tool_name,
                "project_id": project_id,
                "thread_root_id": thread_root_id,
            }
            if value_extra:
                refresh_value.update(value_extra)
            elements.append({"tag": "hr"})
            elements.extend(
                build_responsive_layout(
                    [
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": UI_TEXT["system_ttadk_refresh_btn"]},
                            "type": "primary",
                            "value": refresh_value,
                            "behaviors": [{"type": "callback", "value": refresh_value}],
                        }
                    ]
                )
            )

        card = CoreBuilder._wrap_card(UI_TEXT["system_acp_model_select_title"].format(tool=tool_name), "blue", elements)
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
        """Build a cascade (family × profile × effort) ACP model-select card.

        Mirrors the Workflow ``/wf`` cascade UI: instead of paginating ~90
        Traex model buttons, the model list is compressed into
        family/profile/effort ``select_static`` dropdowns plus a "confirm"
        button (:data:`SELECT_ACP_MODEL`). Changing a dropdown re-renders the
        card via the ``SELECT_ACP_MODEL_*`` actions without entering the mode.

        Default selection remembers the last chosen model: the cascade
        reverse-solves ``current_model`` into its group/profile/effort so the
        dropdowns open pre-selected on the user's previous choice.

        Lists without splittable variants (e.g. Coco's handful of models) fall
        back to the plain paginated button card so those flows keep working.
        """
        from ..render.model_cascade import build_model_cascade_elements, has_cascade_variants

        norm_models: list[dict] = []
        for model in models or []:
            if isinstance(model, dict):
                m_name = str(model.get("name") or "").strip()
                m_disp = model.get("display_name") or m_name
                m_desc = model.get("description", "")
                reasoning_efforts = tuple(model.get("reasoning_efforts") or ())
                adapted_reasoning_effort = model.get(
                    "adapted_reasoning_effort"
                )
                is_default = bool(model.get("is_default"))
                raw_selection_variants = list(
                    model.get("selection_variants") or []
                )
            else:
                m_name = str(getattr(model, "name", None) or model or "").strip()
                m_disp = getattr(model, "display_name", None) or getattr(model, "friendly_name", None) or m_name
                m_desc = getattr(model, "description", "")
                reasoning_efforts = tuple(
                    getattr(model, "reasoning_efforts", ()) or ()
                )
                adapted_reasoning_effort = getattr(
                    model,
                    "adapted_reasoning_effort",
                    None,
                )
                is_default = bool(getattr(model, "is_default", False))
                raw_selection_variants = list(
                    getattr(model, "selection_variants", ()) or ()
                )
            if not m_name:
                continue
            selection_variants = []
            for raw_variant in raw_selection_variants:
                if isinstance(raw_variant, dict):
                    selection_variants.append(dict(raw_variant))
                else:
                    selection_variants.append({
                        "name": str(getattr(raw_variant, "name", "") or ""),
                        "profile": str(
                            getattr(raw_variant, "profile", "") or ""
                        ),
                        "effort": str(
                            getattr(raw_variant, "effort", "default")
                            or "default"
                        ),
                        "display_name": str(
                            getattr(raw_variant, "display_name", "") or ""
                        ),
                        "is_variant_default": bool(
                            getattr(raw_variant, "is_variant_default", False)
                        ),
                    })
            norm_models.append({
                "name": m_name,
                "display_name": str(m_disp or m_name),
                "description": str(m_desc or ""),
                "reasoning_efforts": reasoning_efforts,
                "adapted_reasoning_effort": adapted_reasoning_effort,
                "is_default": is_default,
                "selection_variants": selection_variants,
            })

        # No splittable variants → nothing to cascade; reuse the button card so
        # small model lists (Coco et al.) keep the original behavior.
        if not has_cascade_variants(norm_models):
            return SystemBuilder.build_acp_model_select_card(
                models,
                tool_name,
                project_id,
                current_model=current_model,
                thread_root_id=thread_root_id,
                action_name=select_action,
                value_extra=value_extra,
                context_markdown=context_markdown,
                refresh_action_name=refresh_action,
            )

        def _value_builder(action: str, extra: dict) -> dict:
            value = {
                "action": action,
                "tool_name": tool_name,
                "project_id": project_id,
                "thread_root_id": thread_root_id,
            }
            value.update(extra)
            if value_extra:
                value.update(value_extra)
            return value

        elements: list[dict] = [
            {
                "tag": "markdown",
                "content": UI_TEXT["system_acp_select_model_prompt"].format(tool=tool_name),
            }
        ]
        if context_markdown:
            elements.append({"tag": "markdown", "content": context_markdown})

        elements.extend(
            build_model_cascade_elements(
                models=norm_models,
                value_builder=_value_builder,
                group_action=group_action,
                profile_action=profile_action,
                effort_action=effort_action,
                select_action=select_action,
                default_action=select_action,
                pending_group=pending_group,
                pending_profile=pending_profile,
                pending_effort=pending_effort,
                current_model=current_model,
                button_row_builder=build_responsive_layout,
            )
        )

        # Refresh button (re-probe models), consistent with the button card.
        refresh_value = {
            "action": refresh_action,
            "tool_name": tool_name,
            "project_id": project_id,
            "thread_root_id": thread_root_id,
        }
        if value_extra:
            refresh_value.update(value_extra)
        elements.append({"tag": "hr"})
        elements.extend(
            build_responsive_layout(
                [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": UI_TEXT["system_ttadk_refresh_btn"]},
                        "type": "default",
                        "value": refresh_value,
                        "behaviors": [{"type": "callback", "value": refresh_value}],
                    }
                ]
            )
        )

        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_model_select_title"].format(tool=tool_name), "blue", elements
        )
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_model_loading_card(
        tool_name: str,
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build the first frame for ACP model discovery."""
        _ = project_id, thread_root_id
        elements = [
            {
                "tag": "markdown",
                "content": UI_TEXT["system_acp_model_loading_body"].format(tool=tool_name),
            }
        ]
        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_model_loading_title"].format(tool=tool_name),
            "blue",
            elements,
        )
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_model_error_card(
        tool_name: str,
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build an in-place error frame for ACP model discovery."""
        elements = [
            {
                "tag": "markdown",
                "content": UI_TEXT["system_acp_model_error_body"].format(tool=tool_name),
            },
            {"tag": "hr"},
        ]
        elements.extend(
            build_responsive_layout(
                [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": UI_TEXT["system_ttadk_refresh_btn"]},
                        "type": "primary",
                        "value": {
                            "action": "refresh_acp_models",
                            "tool_name": tool_name,
                            "project_id": project_id,
                            "thread_root_id": thread_root_id,
                        },
                    }
                ]
            )
        )
        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_model_error_title"].format(tool=tool_name),
            "red",
            elements,
        )
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_programming_ready_card(
        tool_name: str,
        model_name: Optional[str],
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build the final frame after an ACP model has been selected."""
        model_label = model_name or UI_TEXT["system_acp_default_model_option"]

        # Detect Anthropic 1M-context [1m] variant — its higher billing tier
        # warrants a prominent warning, especially because the choice is
        # persisted to ProjectContext and silently auto-restored on every
        # subsequent /claude entry, project open, daemon restart, or session
        # resume. Without this banner, a one-time experimental click can
        # become a hidden default that doubles base-rate billing past 200K
        # tokens of context.
        is_1m = False
        try:
            if isinstance(model_name, str) and model_name.strip():
                from src.acp.claude_capabilities import is_1m_variant
                is_1m = is_1m_variant(model_name.strip())
        except Exception:  # pragma: no cover — UI must never crash on detection
            is_1m = False

        elements: list[dict] = []
        if is_1m:
            elements.append(
                {
                    "tag": "markdown",
                    "content": UI_TEXT["system_acp_1m_warning_banner"].format(model=model_label),
                    "text_align": "left",
                }
            )
            elements.append({"tag": "hr"})

        elements.append(
            {
                "tag": "markdown",
                "content": UI_TEXT["system_acp_programming_ready_body"].format(
                    tool=tool_name,
                    model=model_label,
                ),
            }
        )
        elements.append({"tag": "hr"})

        action_buttons: list[dict] = [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": UI_TEXT["system_acp_switch_model_btn"]},
                "type": "default",
                "value": {
                    "action": "refresh_acp_models",
                    "tool_name": tool_name,
                    "project_id": project_id,
                    "thread_root_id": thread_root_id,
                },
            }
        ]
        if is_1m:
            # One-click escape hatch back to the standard (non-1M) model id.
            standard_model = (model_name or "").strip()
            if standard_model.endswith("[1m]"):
                standard_model = standard_model[: -len("[1m]")]
            action_buttons.insert(
                0,
                {
                    "tag": "button",
                    "text": {
                        "tag": "plain_text",
                        "content": UI_TEXT["system_acp_1m_switch_standard_btn"],
                    },
                    "type": "danger",
                    "value": {
                        "action": "select_acp_model",
                        "tool_name": tool_name,
                        "model_name": standard_model,
                        "project_id": project_id,
                        "thread_root_id": thread_root_id,
                    },
                },
            )
        elements.extend(build_responsive_layout(action_buttons))

        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_programming_ready_title"].format(tool=tool_name),
            "red" if is_1m else "green",
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
        """Build the immediate frame shown while the selected ACP session starts."""
        _ = project_id, thread_root_id
        model_label = model_name or UI_TEXT["system_acp_default_model_option"]
        elements = [
            {
                "tag": "markdown",
                "content": UI_TEXT["system_acp_programming_initializing_body"].format(
                    tool=tool_name,
                    model=model_label,
                ),
            }
        ]
        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_programming_initializing_title"].format(tool=tool_name),
            "blue",
            elements,
        )
        return "interactive", json.dumps(card, ensure_ascii=False)

    @staticmethod
    def build_acp_programming_failed_card(
        tool_name: str,
        model_name: Optional[str],
        reason: str,
        project_id: Optional[str] = None,
        thread_root_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build a retryable terminal frame after ACP session startup fails."""
        model_label = model_name or UI_TEXT["system_acp_default_model_option"]
        retry_value = {
            "action": action_ids.SELECT_ACP_MODEL,
            "tool_name": tool_name,
            "model_name": model_name or DEFAULT_MODEL_OPTION_VALUE,
            "project_id": project_id,
            "thread_root_id": thread_root_id,
        }
        if model_name is None:
            retry_value["use_default_model"] = True
        elements = [
            {
                "tag": "markdown",
                "content": UI_TEXT["system_acp_programming_failed_body"].format(
                    tool=tool_name,
                    model=model_label,
                    reason=reason,
                ),
            },
            {"tag": "hr"},
        ]
        elements.extend(
            build_responsive_layout(
                [
                    {
                        "tag": "button",
                        "text": {
                            "tag": "plain_text",
                            "content": UI_TEXT["system_acp_retry_activation_btn"],
                        },
                        "type": "danger",
                        "value": retry_value,
                    },
                    {
                        "tag": "button",
                        "text": {
                            "tag": "plain_text",
                            "content": UI_TEXT["system_acp_back_to_models_btn"],
                        },
                        "type": "default",
                        "value": {
                            "action": action_ids.REFRESH_ACP_MODELS,
                            "tool_name": tool_name,
                            "project_id": project_id,
                            "thread_root_id": thread_root_id,
                        },
                    },
                ]
            )
        )
        card = CoreBuilder._wrap_card(
            UI_TEXT["system_acp_programming_failed_title"].format(tool=tool_name),
            "red",
            elements,
        )
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
                "text": UI_TEXT["system_menu_btn_ttadk"],
                "type": "default",
                "action": "show_ttadk_menu",
            },
            {
                "text": UI_TEXT["system_menu_btn_acp"],
                "type": "default",
                "action": "show_acp_menu",
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
            InteractionMode.TTADK: UI_TEXT["system_mode_ttadk"],
            InteractionMode.TUI2ACP: "🌉 Tui2ACP",
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
                UI_TEXT["system_help_section_ttadk"],
                UI_TEXT["system_help_section_ttadk_body"]
            ),
            (
                UI_TEXT["system_help_section_workflow"],
                UI_TEXT["system_help_section_workflow_body"]
            ),
            (
                UI_TEXT["system_help_section_slock"],
                UI_TEXT["system_help_section_slock_body"]
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
