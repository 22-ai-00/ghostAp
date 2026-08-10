"""Workflow completion reporting and progress-card helpers."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...project import ProjectContext

logger = logging.getLogger(__name__)


class WorkflowScriptMixin:
    """Mixin providing completion reporting and progress-card helpers."""


    def _reply_workflow_completion_fallback(
        self,
        *,
        message_id: str,
        report_status: dict[str, Any] | None,
        failed_page_indexes: tuple[int, ...] = (),
    ) -> None:
        """Expose a truthful, retryable terminal delivery state."""
        status = report_status or {}
        if status.get("attachment_sent"):
            lines = [
                "⚠️ Workflow 结果卡未完整投递",
                "完整 HTML 报告已回复到当前话题，请以附件为完整交付物。",
            ]
            self.reply_text(message_id, "\n\n".join(lines))
            return

        lines = [
            "❌ Workflow 完整结果交付失败（可重试）",
            "执行已进入终态，但结果分页和 HTML 附件均未完整送达；不得将当前可见内容视为完整结果。",
        ]
        if failed_page_indexes:
            pages = ", ".join(str(index + 1) for index in failed_page_indexes)
            lines.append(f"未送达页面：{pages}。")
        elif status.get("generated"):
            filename = status.get("html_filename")
            if not filename and status.get("html_path"):
                filename = os.path.basename(str(status["html_path"]))
            filename = str(filename or "Workflow HTML 报告")
            if len(filename.encode("utf-8", errors="surrogatepass")) > 800:
                filename = "Workflow HTML 报告"
            lines.append(f"完整报告已安全保存在服务端：{filename}。")
        else:
            lines.append("完整报告未生成。")
        error = str(status.get("error") or "").strip()
        if error:
            lines.append(f"附件失败原因：{error}")
        lines.append("发送 `/wf_status` 可重新投递全部结果页。")
        self.reply_text(message_id, "\n\n".join(lines))

    def _send_workflow_completion_report(
        self,
        *,
        wf_project: Any,
        chat_id: str,
        message_id: str,
        project: Optional["ProjectContext"],
    ) -> dict[str, Any]:
        """Generate a full Workflow report and reply with its HTML attachment."""
        from ...utils.errors import get_error_detail
        from ...workflow_engine.reporting import write_workflow_report_files

        try:
            root_path = self._get_root_path(chat_id, project)
        except Exception:
            logger.debug("Failed to resolve workflow report root path", exc_info=True)
            root_path = os.getcwd()

        try:
            files = write_workflow_report_files(wf_project, root_path=root_path)
        except Exception as exc:
            logger.warning("Failed to generate workflow completion report: %s", get_error_detail(exc), exc_info=True)
            return {
                "generated": False,
                "attachment_sent": False,
                "error": get_error_detail(exc),
            }

        status: dict[str, Any] = {
            "generated": True,
            "attachment_sent": False,
            "html_path": files.html_path,
            "markdown_path": files.markdown_path,
            "html_filename": files.html_filename,
            "markdown_filename": files.markdown_filename,
            "run_id": files.run_id,
        }

        im_client = getattr(self, "im_client", None)
        if im_client is None:
            status["error"] = "IM 客户端不可用，附件未发送"
            return status

        try:
            file_key = im_client.upload_file(
                files.html_path,
                file_type="stream",
                file_name=files.html_filename,
            )
            if not file_key:
                status["error"] = "上传 HTML 报告失败，已保留本地报告"
                return status

            origin = self._resolve_origin(message_id)
            response = im_client.reply_file(
                message_id,
                file_key,
                reply_in_thread=True,
                audit_aliases=self._reply_audit_aliases(origin),
            )
            if response is None:
                status["error"] = "回复 HTML 附件失败，已保留本地报告"
                return status
            if hasattr(response, "success") and not response.success():
                status["error"] = getattr(response, "msg", "回复 HTML 附件失败")
                return status

            status["attachment_sent"] = True
            status["file_key"] = file_key
            return status
        except Exception as exc:
            logger.warning("Failed to send workflow completion report attachment: %s", get_error_detail(exc), exc_info=True)
            status["error"] = get_error_detail(exc)
            return status









    @staticmethod
    def _workflow_error_card_category(error_msg: str) -> str:
        """Map an execution error to the Workflow card surface once."""
        from ...workflow_engine.errors import ErrorCategory, categorize_error

        category = categorize_error(error_msg)
        if category == ErrorCategory.TOOL_NOT_ALLOWED:
            return "forbidden"
        if category == ErrorCategory.SCRIPT_VALIDATION:
            return "invalid_argument"
        if category == ErrorCategory.RUNTIME_TIMEOUT:
            return "runtime_timeout"
        if category == ErrorCategory.REVIEW_FAILED:
            return "review_failed"
        if category in (ErrorCategory.AGENT_LIMIT, ErrorCategory.CANCELLED):
            return "invalid_state"
        return "internal_error"

    def _workflow_is_running_for_card(
        self,
        chat_id: str,
        project: Optional["ProjectContext"],
    ) -> bool:
        """Read the authoritative engine state before exposing a stop action."""
        from ...workflow_engine.models import WorkflowStatus

        try:
            root_path = self._get_root_path(chat_id, project)
            engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)
            if engine is None:
                return False
            with engine._lock:
                wf_project = engine.project
                return bool(
                    wf_project is not None
                    and wf_project.status == WorkflowStatus.RUNNING
                )
        except Exception:
            logger.debug("Failed to resolve Workflow status for progress action", exc_info=True)
            return False


    def _inject_workflow_stop_button(
        self,
        card_data: dict[str, Any],
        chat_id: str,
        project_id: str,
        *,
        is_running: bool,
    ) -> None:
        """Append a "停止" button row to a RUNNING progress card.

        ``card_data`` is the renderer output ``{"header": ..., "elements": [...]}``.
        ``on_progress`` also carries terminal flushes, so callers must pass the
        authoritative RUNNING state rather than inferring it from callback type.

        The button value carries only ``action``/``chat_id``/``project_id``.
        The handler delegates to ``stop_workflow``, which re-derives auth from
        the live engine state, so no session key is required in the payload.
        """
        from ...card.actions.dispatch import WORKFLOW_STOP_RUNNING
        from ...card.render.buttons import build_responsive_button_row
        from ...card.ui_text import UI_TEXT

        if not is_running or not isinstance(card_data, dict):
            return
        elements = card_data.get("elements")
        if not isinstance(elements, list):
            return

        confirm_title = UI_TEXT.get("workflow_btn_confirm_stop_title", "确认停止 Workflow？")
        confirm_body = UI_TEXT.get(
            "workflow_btn_confirm_stop_body",
            "正在执行的步骤将中断，已完成的部分不受影响。",
        )
        stop_value = {
            "action": WORKFLOW_STOP_RUNNING,
            "chat_id": chat_id,
            "project_id": project_id or "",
        }
        stop_button = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": "⏹️ 停止 Workflow"},
            "type": "danger",
            "value": stop_value,
            "behaviors": [{"type": "callback", "value": stop_value}],
            "confirm": {
                "title": {"tag": "plain_text", "content": confirm_title},
                "text": {"tag": "plain_text", "content": confirm_body},
            },
        }

        elements.append({"tag": "hr"})
        elements.extend(build_responsive_button_row([stop_button]))
