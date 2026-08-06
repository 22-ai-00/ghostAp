from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from src.mode.manager import InteractionMode

from ..shared import (
    build_mode_buttons,
)
from ..thresholds import THRESHOLDS
from ..ui_text import UI_TEXT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.project.context import ProjectContext


class CoreBuilder:
    """Core card building utilities."""

    @staticmethod
    def _truncate_markdown(content: str, max_chars: int) -> str:
        """Truncate markdown content safely, closing code blocks and bold tags."""
        if len(content) <= max_chars:
            return content

        warning_msg = UI_TEXT["log_truncated_warning"]
        keep_chars = max_chars - len(warning_msg) - 20  # buffer

        truncated_content = content[-keep_chars:]

        # 1. Check code blocks (```)
        cut_index = len(content) - keep_chars
        pre_cut_content = content[:cut_index]
        code_block_markers = pre_cut_content.count("```")
        is_inside_code_block = code_block_markers % 2 != 0

        # 2. Check bold tags (**)
        bold_markers = pre_cut_content.count("**")
        is_inside_bold = bold_markers % 2 != 0

        parts = [warning_msg]

        if is_inside_code_block:
            parts.append("```\n")

        if is_inside_bold:
            parts.append("**")

        parts.append(truncated_content)

        result = "".join(parts)

        # Final safety check: ensure closed
        if result.count("```") % 2 != 0:
            result += "\n```"
        if result.count("**") % 2 != 0:
            result += "**"

        return result

    @staticmethod
    def _build_content_element(content: str, with_title: Optional[str] = None, max_chars: int = THRESHOLDS["CONTENT_MAX_CHARS"]) -> dict:
        full_content = f"**{with_title}**\n\n{content}" if with_title else content

        # Smart truncation to prevent API errors and render issues
        if len(full_content) > max_chars:
            full_content = CoreBuilder._truncate_markdown(full_content, max_chars)

        return {"tag": "markdown", "content": full_content}

    @staticmethod
    def _build_banner_element(message: str, type: str = "info") -> dict:
        """Build a prominent banner element using column_set and background_style.

        语义化配色方案（Apple 风格优化）：使用更现代、更柔和的颜色，确保产品一致性和更好的用户体验。
        """
        # Apple 风格语义化配色方案：更柔和、更现代的颜色，保持一致性的同时传递语义
        style_map = {
            "success": ("green", "✅"),   # 成功使用绿色
            "warning": ("orange", "⚠️"),  # 警告使用橙色（更温和、更符合现代设计）
            "error": ("red", "❌"),      # 错误使用红色
            "info": ("wathet", "ℹ️"),    # 信息使用浅蓝色
        }
        bg_style, emoji = style_map.get(type, ("wathet", "ℹ️"))

        return {
            "tag": "column_set",
            "flex_mode": "stretch",
            "background_style": bg_style,
            "columns": [
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "vertical_align": "center",
                    "elements": [
                        {
                            "tag": "markdown",
                            "content": f"**{emoji} {message}**",
                        }
                    ],
                }
            ],
        }

    @staticmethod
    def _build_header_title(
        project: Optional[ProjectContext],
        mode: Optional[InteractionMode] = None,
    ) -> str:
        # Resolve mode from project if not provided directly
        effective_mode = mode
        if effective_mode is None and project:
            if getattr(project, "claude_mode", False):
                effective_mode = InteractionMode.CLAUDE
            elif getattr(project, "gemini_mode", False):
                effective_mode = InteractionMode.GEMINI
            elif getattr(project, "traex_mode", False):
                effective_mode = InteractionMode.TRAEX
            elif getattr(project, "coco_mode", False):
                effective_mode = InteractionMode.COCO

        if not project:
            if effective_mode == InteractionMode.CLAUDE:
                return UI_TEXT["claude_mode_title"]
            elif effective_mode == InteractionMode.GEMINI:
                return UI_TEXT["gemini_mode_title"]
            elif effective_mode == InteractionMode.TRAEX:
                return UI_TEXT["system_mode_traex"]
            mode_icon = "🤖" if effective_mode == InteractionMode.COCO else "🧠"
            mode_name = UI_TEXT["coco_mode_title"] if effective_mode == InteractionMode.COCO else UI_TEXT["smart_mode_title"]
            return f"{mode_icon} {mode_name}"

        if effective_mode == InteractionMode.CLAUDE:
            return f"🔮 {project.project_name} · Claude"
        elif effective_mode == InteractionMode.GEMINI:
            return f"✨ {project.project_name} · Gemini"
        elif effective_mode == InteractionMode.TRAEX:
            return f"🚀 {project.project_name} · Traex"
        elif effective_mode == InteractionMode.COCO:
            return f"🤖 {project.project_name} · Coco"
        else:
            return f"🧠 {project.project_name}"

    @staticmethod
    def _build_directory_element(project: Optional[ProjectContext], working_dir: Optional[str] = None) -> dict:
        if project:
            path = project.root_path
        elif working_dir:
            path = working_dir
        else:
            path = "~"

        return {"tag": "markdown", "content": f"📁 `{path}`"}

    @staticmethod
    def _build_footer_buttons(
        project: Optional[ProjectContext],
        mode: Optional[InteractionMode] = None,
        *,
        button_size: str = "medium",
    ) -> list[dict]:
        project_id_raw = getattr(project, "project_id", None) if project else None
        project_id = str(project_id_raw) if isinstance(project_id_raw, (str, int)) else None

        thread_root_id = None
        try:
            from ...thread import get_current_thread_id
            current_thread_id = get_current_thread_id()
            thread_root_id = current_thread_id if isinstance(current_thread_id, str) else None
        except Exception:
            logger.debug("failed to get thread_id", exc_info=True)

        effective_mode = mode
        if effective_mode is None and project:
            if getattr(project, "claude_mode", False):
                effective_mode = InteractionMode.CLAUDE
            elif getattr(project, "gemini_mode", False):
                effective_mode = InteractionMode.GEMINI
            elif getattr(project, "traex_mode", False):
                effective_mode = InteractionMode.TRAEX
            elif getattr(project, "coco_mode", False):
                effective_mode = InteractionMode.COCO

        return build_mode_buttons(effective_mode, project_id, thread_root_id=thread_root_id, button_size=button_size)

    @staticmethod
    def _build_footer_note(project: Optional[ProjectContext], working_dir: Optional[str] = None) -> Optional[dict]:
        if project:
            content = UI_TEXT["project_dir_label"].format(path=project.root_path)
            return {
                "tag": "markdown",
                "content": content,
                "text_size": "notation",
            }
        return None

    @staticmethod
    def _wrap_card(header_title: str, header_template: str, elements: list[dict], *, subtitle: str = "") -> dict:
        header: dict = {
            "title": {"tag": "plain_text", "content": header_title},
            "template": header_template,
        }
        if subtitle:
            header["subtitle"] = {"tag": "plain_text", "content": subtitle}
        return {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": header,
            "body": {
                "elements": elements,
            },
        }

    @staticmethod
    def _build_image_elements(image_keys: list[str]) -> list[dict]:
        elements = []
        for i, key in enumerate(image_keys):
            alt_text = UI_TEXT["image_alt_text"].format(index=i + 1)
            elements.append({"tag": "img", "img_key": key, "alt": {"tag": "plain_text", "content": alt_text}})
        return elements

    @staticmethod
    def _format_time_ago(timestamp: float) -> str:
        """[DEPRECATED] 卡片层相对时间包装函数。

        统一入口应使用 :func:`src.utils.text.format_time_ago`，本函数仅负责：
        - 接受时间戳 ``timestamp``；
        - 与当前时间做差得到秒数 ``diff``；
        - 使用 :func:`compute_time_ago_bucket` 计算语义区间；
        - 再通过共享渲染层生成最终文案。

        负数时间差会被当作 0 处理，避免出现 "负几天前" 等异常描述。
        """

        try:
            diff = float(time.time() - (timestamp or 0.0))
        except Exception:
            diff = 0.0
        if diff < 0:
            diff = 0.0

        from src.utils.text import format_time_ago_from_bucket
        from src.utils.time_ago import compute_time_ago_bucket

        bucket = compute_time_ago_bucket(diff)
        return format_time_ago_from_bucket(bucket)
