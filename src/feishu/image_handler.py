import hashlib
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..utils.errors import get_error_detail

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ImageParseResult:
    """飞书消息解析结果：文本 + 图片 key 列表"""

    text: str
    image_keys: list[str] = field(default_factory=list)


@dataclass
class ImageDownloadResult:
    """图片下载结果：成功路径 + 失败 key"""

    saved_paths: list[str] = field(default_factory=list)
    failed_keys: list[str] = field(default_factory=list)


class FeishuImageHandler:
    """处理飞书消息中的图片：解析、下载、存储"""

    CACHE_ROOT = "~/.cache/ghostAp"
    IMAGE_DIR_NAME = "picturechat"

    def __init__(self, api_client_factory: Callable[[], Any], settings: "Settings"):
        from .im_client import FeishuIMClient

        self.im_client = FeishuIMClient(api_client_factory, settings)

    def parse_message(self, message_type: str, content_str: str) -> ImageParseResult:
        """根据消息类型解析文本和图片 key

        支持三种消息类型:
        - text: 纯文本，无图片
        - image: 单张图片，无文字
        - post: 富文本，包含文字和图片
        """
        if message_type == "text":
            return self._parse_text_message(content_str)
        elif message_type == "image":
            return self._parse_image_message(content_str)
        elif message_type == "post":
            return self._parse_post_message(content_str)
        else:
            return ImageParseResult(text="", image_keys=[])

    def _parse_text_message(self, content_str: str) -> ImageParseResult:
        """解析文本消息 {"text": "..."}"""
        try:
            content_dict = json.loads(content_str)
            text = content_dict.get("text", "")
        except json.JSONDecodeError:
            text = content_str
        return ImageParseResult(text=text, image_keys=[])

    def _parse_image_message(self, content_str: str) -> ImageParseResult:
        """解析图片消息 {"image_key": "img_v2_xxx"}"""
        try:
            content_dict = json.loads(content_str)
            image_key = content_dict.get("image_key", "")
            image_keys = [image_key] if image_key else []
        except json.JSONDecodeError:
            image_keys = []
        return ImageParseResult(text="", image_keys=image_keys)

    def _parse_post_message(self, content_str: str) -> ImageParseResult:
        """解析富文本消息，提取文字和图片 key

        飞书 post 格式:
        {
          "zh_cn": {
            "title": "...",
            "content": [
              [
                {"tag": "text", "text": "some text"},
                {"tag": "img", "image_key": "img_v2_xxx"}
              ]
            ]
          }
        }
        """
        try:
            content_dict = json.loads(content_str)
        except json.JSONDecodeError:
            return ImageParseResult(text=content_str, image_keys=[])

        post_data = self._select_post_data(content_dict)
        if not post_data:
            return ImageParseResult(text="", image_keys=[])

        candidates = []
        for key in ("content", "content_v2"):
            rows = post_data.get(key)
            if isinstance(rows, list) and rows not in candidates:
                candidates.append(rows)

        parsed_candidates = []
        for content_rows in candidates:
            text_rows = []
            image_keys = []
            for row in content_rows:
                if not isinstance(row, list):
                    continue
                row_text_parts = []
                for element in row:
                    if not isinstance(element, dict):
                        continue
                    tag = element.get("tag", "")
                    if tag == "text":
                        t = element.get("text", "")
                        if t:
                            row_text_parts.append(t)
                    elif tag == "img":
                        img_key = element.get("image_key", "")
                        if img_key:
                            image_keys.append(img_key)
                if row_text_parts:
                    text_rows.append("".join(row_text_parts))
            parsed_candidates.append(("\n".join(text_rows), image_keys))

        text = self._merge_mirrored_texts([parts[0] for parts in parsed_candidates])
        image_keys = self._merge_mirrored_values([parts[1] for parts in parsed_candidates])
        return ImageParseResult(
            text=text.strip(),
            image_keys=image_keys,
        )

    @staticmethod
    def _merge_mirrored_texts(candidates: list[str]) -> str:
        """Prefer supersets and retain distinct partial text without mirror duplication."""
        selected = ""
        selected_normalized = ""
        for text in candidates:
            if not text:
                continue
            normalized = " ".join(text.split())
            if not selected:
                selected = text
                selected_normalized = normalized
            elif normalized == selected_normalized or normalized in selected_normalized:
                continue
            elif selected_normalized in normalized:
                selected = text
                selected_normalized = normalized
            else:
                selected = f"{selected}\n{text}"
                selected_normalized = " ".join(selected.split())
        return selected

    @staticmethod
    def _merge_mirrored_values(candidates: list[list[str]]) -> list[str]:
        """Merge representations while preserving repeats within either body."""
        merged: list[str] = []
        merged_counts: Counter[str] = Counter()
        for values in candidates:
            candidate_counts: Counter[str] = Counter()
            for value in values:
                candidate_counts[value] += 1
                if candidate_counts[value] > merged_counts[value]:
                    merged.append(value)
            merged_counts |= candidate_counts
        return merged

    @staticmethod
    def _select_post_data(content_dict: object) -> Optional[dict]:
        """Resolve official flat and legacy localized post bodies."""
        if not isinstance(content_dict, dict):
            return None

        if "content" in content_dict or "content_v2" in content_dict:
            return content_dict

        for lang in ("zh_cn", "en_us"):
            candidate = content_dict.get(lang)
            if isinstance(candidate, dict) and (
                "content" in candidate or "content_v2" in candidate
            ):
                return candidate

        for candidate in content_dict.values():
            if isinstance(candidate, dict) and (
                "content" in candidate or "content_v2" in candidate
            ):
                return candidate
        return None

    def download_images(
        self,
        message_id: str,
        image_keys: list[str],
        save_dir: str,
    ) -> ImageDownloadResult:
        """批量下载图片并保存到按消息隔离的子目录

        图片以序号命名（1.png, 2.png, ...），保存在
        ``{save_dir}/msg_{message_sha256}/`` 子目录下，避免不同项目、聊天和
        消息的图片冲突，也不把飞书消息 ID 暴露在本地目录名中。

        Args:
            message_id: 飞书消息 ID（下载 API 需要，同时用于子目录命名）
            image_keys: 图片 key 列表
            save_dir: 基础保存目录（不存在会自动创建）

        Returns:
            ImageDownloadResult 包含成功保存的路径和下载失败的 key
        """
        result = ImageDownloadResult()

        if not image_keys:
            return result

        message_digest = hashlib.sha256(message_id.encode("utf-8")).hexdigest()
        msg_dir = os.path.join(save_dir, f"msg_{message_digest}")
        os.makedirs(save_dir, mode=0o700, exist_ok=True)
        os.makedirs(msg_dir, mode=0o700, exist_ok=True)

        for index, image_key in enumerate(image_keys, 1):
            saved_path = self._download_single_image(
                message_id,
                image_key,
                msg_dir,
                index,
            )
            if saved_path:
                result.saved_paths.append(saved_path)
            else:
                result.failed_keys.append(image_key)

        return result

    def _download_single_image(
        self,
        message_id: str,
        image_key: str,
        save_dir: str,
        index: int = 1,
    ) -> Optional[str]:
        """下载单张图片，返回保存路径或 None"""
        try:
            response = self.im_client.get_resource(message_id=message_id, file_key=image_key, resource_type="image")

            if not response or not response.success():
                return None

            filename = f"{index}.png"
            filepath = os.path.join(save_dir, filename)
            payload = response.file.read()
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(filepath, flags, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as image_file:
                    descriptor = -1
                    image_file.write(payload)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

            logger.info("图片已保存: %s", filepath)
            return filepath

        except Exception as e:
            logger.error("下载图片异常: %s, error=%s", image_key, get_error_detail(e))
            return None

    @staticmethod
    def build_image_reference_text(saved_paths: list[str]) -> str:
        """生成追加到用户消息末尾的图片引用文本

        格式（单张）:
            \\n\\n[用户附带了 1 张参考图片，请先查看后再回答]
            图片1: /path/to/1.png
        格式（多张）:
            \\n\\n[用户附带了 N 张参考图片，请先查看后再回答]
            图片1: /path/to/1.png
            图片2: /path/to/2.png
        """
        if not saved_paths:
            return ""

        n = len(saved_paths)
        lines = [f"\n\n[用户附带了 {n} 张参考图片，请先查看后再回答]"]
        for i, path in enumerate(saved_paths, 1):
            lines.append(f"图片{i}: {path}")
        return "\n".join(lines)

    @staticmethod
    def get_image_save_dir() -> str:
        """返回所有聊天与编程模式共享的用户级图片缓存目录。"""
        cache_root = os.path.abspath(os.path.expanduser(FeishuImageHandler.CACHE_ROOT))
        return os.path.join(cache_root, FeishuImageHandler.IMAGE_DIR_NAME)
