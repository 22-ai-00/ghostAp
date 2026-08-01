"""Unified error-card contract: error paths must route through _reply_workflow_error.

Every error / rejection / failure path in WorkflowHandler must surface to the user
as a structured red card via ``_reply_workflow_error`` — never via plain
``reply_text``.  Informational / status / help responses may still use plain text.
"""

import ast
import os
import unittest
from unittest.mock import MagicMock, patch

from src.workflow_engine.models import PendingConfirmation, WorkflowProject, WorkflowStatus


def _handler_with_mocks(project: "WorkflowProject | None" = None) -> tuple:
    """Build a WorkflowHandler with mocked reply_card / reply_text + engine manager."""
    from src.feishu.handlers.workflow import WorkflowHandler

    ctx = MagicMock()
    engine = MagicMock()
    if project is not None:
        engine.project = project
    ctx.workflow_engine_manager.get.return_value = engine
    ctx.workflow_engine_manager.get_or_create.return_value = engine

    handler = WorkflowHandler(ctx)
    handler.reply_card = MagicMock()
    handler.reply_text = MagicMock()
    handler.reply_error = MagicMock()
    handler._resolve_project_from_id = MagicMock(return_value=project)
    handler._get_root_path = MagicMock(return_value="/tmp")
    return handler, engine


def _make_project(status: WorkflowStatus, *, session_key: str = "key", initiator_user_id: str = "user1", selected_tools: list[str] | None = None, script_path: str | None = None) -> MagicMock:
    project = MagicMock(spec=[
        "project_id", "chat_id", "status", "pending", "root_path",
        "initiator_user_id",
    ])
    project.project_id = "p"
    project.chat_id = "c"
    project.status = status
    project.root_path = "/tmp"
    project.initiator_user_id = initiator_user_id
    project.pending = PendingConfirmation(
        requirement="r",
        initiator_user_id=initiator_user_id,
        engine_session_key=session_key,
        selected_tools=selected_tools or ["coco"],
        script_path=script_path,
        meta=None,
    )
    return project


# ---------------------------------------------------------------------------
# Static AST check: every `reply_text` in workflow.py must be informational
# ---------------------------------------------------------------------------

class TestStaticNoReplyTextOnErrorPaths(unittest.TestCase):
    """AST/text check — enumerate `reply_text(...)` sites and reject any
    that look like error messages (contain `失败`, `不存在`, `请`, `无法`, `过期`, `权限` etc.).
    """

    _FILE = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "src", "feishu", "handlers", "workflow.py",
    )

    ERROR_MARKERS = (
        "失败", "不存在", "无法", "过期", "权限", "拒绝", "请至少",
        "没有可保存", "脚本文件不存在", "无法获取待执行",
        "仅允许管理员", "被过滤",
    )

    def test_no_error_marked_reply_text_calls(self) -> None:
        with open(self._FILE, "r", encoding="utf-8") as fh:
            source = fh.read()

        tree = ast.parse(source, filename=self._FILE)
        violations: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # detect `self.reply_text(...)` or `reply_text(...)`
            is_reply_text = (
                isinstance(func, ast.Attribute) and func.attr == "reply_text"
                or (isinstance(func, ast.Name) and func.id == "reply_text")
            )
            if not is_reply_text:
                continue
            # Stringify the first arg string literals to look for error markers
            first_arg_literal: str = ""
            if node.args:
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        first_arg_literal += a.value + "\n"
            for kw in node.keywords:
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    first_arg_literal += kw.value.value + "\n"
            for marker in self.ERROR_MARKERS:
                if marker in first_arg_literal:
                    violations.append(
                        f"line {getattr(node, 'lineno', '?')}: `reply_text` "
                        f"looks like an error (contains '{marker}'); use `_reply_workflow_error`"
                    )
                    break
        self.assertEqual(
            violations, [],
            msg="\n".join(violations) if violations else "",
        )

    def test_reply_workflow_error_used_for_errors(self) -> None:
        """_reply_workflow_error must exist and be called at least once."""
        with open(self._FILE, "r", encoding="utf-8") as fh:
            source = fh.read()
        # must define
        self.assertIn("def _reply_workflow_error", source)
        # must invoke
        self.assertIn("self._reply_workflow_error(", source)


# ---------------------------------------------------------------------------
# Dynamic: route each handler through an invalid payload and ensure it goes
# through _reply_workflow_error (mock), not reply_text
# ---------------------------------------------------------------------------

class TestHandlerInvalidPayloadRoutesToUnifiedError(unittest.TestCase):
    """For each button-click / workflow handler, call it with an invalid payload
    (wrong session key, non-initiator user, missing fields) and confirm the
    error surfaces via ``_reply_workflow_error`` and never via plain text."""

    def _run(self, handler_method, *args, **kwargs) -> None:
        handler, *_ = _handler_with_mocks()
        handler._reply_workflow_error = MagicMock()  # type: ignore[method-assign]
        handler.reply_text = MagicMock()  # type: ignore[method-assign]

        # patch `get_current_sender_id` to return a user that differs from any default
        with patch("src.feishu.handlers.workflow.get_current_sender_id", return_value="attacker_user"):
            handler_method(handler, *args, **kwargs)
        # Error paths must use _reply_workflow_error, not reply_text.
        handler._reply_workflow_error.assert_called()  # type: ignore[attr-defined]
        handler.reply_text.assert_not_called()  # type: ignore[attr-defined]


    def _make_invalid_payload_project(self) -> MagicMock:
        return _make_project(
            WorkflowStatus.AWAITING_TOOL_SELECT,
            session_key="real_session_key_12345",
        )

    def test_handle_workflow_confirm_start_wrong_session_key(self) -> None:
        from src.feishu.handlers.workflow import WorkflowHandler

        project = _make_project(WorkflowStatus.AWAITING_CONFIRM, session_key="real",
                                initiator_user_id="attacker_user")
        handler, _ = _handler_with_mocks(project)
        handler._reply_workflow_error = MagicMock()  # type: ignore[method-assign]
        handler.reply_text = MagicMock()  # type: ignore[method-assign]
        with patch("src.thread.manager.get_current_sender_id", return_value="attacker_user"):
            WorkflowHandler.handle_workflow_confirm_start(
                handler, "msg-id", "c-1", "p-1",
                {"engine_session_key": "WRONG"},
            )
        handler._reply_workflow_error.assert_called()  # type: ignore[attr-defined]
        handler.reply_text.assert_not_called()  # type: ignore[attr-defined]

    def test_handle_workflow_cancel_wrong_session_key(self) -> None:
        from src.feishu.handlers.workflow import WorkflowHandler

        project = _make_project(WorkflowStatus.AWAITING_CONFIRM, session_key="real",
                                initiator_user_id="attacker_user")
        handler, _ = _handler_with_mocks(project)
        handler._reply_workflow_error = MagicMock()  # type: ignore[method-assign]
        handler.reply_text = MagicMock()  # type: ignore[method-assign]
        with patch("src.thread.manager.get_current_sender_id", return_value="attacker_user"):
            WorkflowHandler.handle_workflow_cancel(
                handler, "msg-id", "c-1", "p-1",
                {"engine_session_key": "WRONG"},
            )
        handler._reply_workflow_error.assert_called()  # type: ignore[attr-defined]
        handler.reply_text.assert_not_called()  # type: ignore[attr-defined]

    def test_handle_workflow_select_tool_unknown_tool(self) -> None:
        from src.feishu.handlers.workflow import WorkflowHandler

        project = _make_project(WorkflowStatus.AWAITING_TOOL_SELECT, session_key="s",
                                initiator_user_id="attacker_user")
        handler, _ = _handler_with_mocks(project)
        handler._reply_workflow_error = MagicMock()  # type: ignore[method-assign]
        handler.reply_text = MagicMock()  # type: ignore[method-assign]
        with patch("src.thread.get_current_sender_id", return_value="attacker_user"), \
             patch.object(WorkflowHandler, "_validate_tools_against_registry",
                          return_value=(["coco"], ["unknown_tool_name"])):
            WorkflowHandler.handle_workflow_select_tool(
                handler, "msg-id", "c-1", "p-1",
                {"engine_session_key": "s", "tool_name": "unknown_tool_name"},
            )
        handler._reply_workflow_error.assert_called()  # type: ignore[attr-defined]
        handler.reply_text.assert_not_called()  # type: ignore[attr-defined]

    def test_handle_workflow_fill_missing_tools_wrong_user(self) -> None:
        from src.feishu.handlers.workflow import WorkflowHandler

        project = _make_project(WorkflowStatus.AWAITING_CONFIRM, session_key="s",
                                initiator_user_id="owner_user")
        handler, _ = _handler_with_mocks(project)
        handler._reply_workflow_error = MagicMock()  # type: ignore[method-assign]
        handler.reply_text = MagicMock()  # type: ignore[method-assign]
        with patch("src.thread.get_current_sender_id", return_value="attacker_user"):
            WorkflowHandler.handle_workflow_fill_missing_tools(
                handler, "msg-id", "c-1", "p-1",
                {"engine_session_key": "s"},
            )
        handler._reply_workflow_error.assert_called()  # type: ignore[attr-defined]
        handler.reply_text.assert_not_called()  # type: ignore[attr-defined]

    def test_handle_workflow_back_to_tools_wrong_user(self) -> None:
        from src.feishu.handlers.workflow import WorkflowHandler

        project = _make_project(WorkflowStatus.AWAITING_CONFIRM, session_key="s",
                                initiator_user_id="owner_user")
        handler, _ = _handler_with_mocks(project)
        handler._reply_workflow_error = MagicMock()  # type: ignore[method-assign]
        handler.reply_text = MagicMock()  # type: ignore[method-assign]
        with patch("src.thread.get_current_sender_id", return_value="attacker_user"):
            WorkflowHandler.handle_workflow_back_to_tools(
                handler, "msg-id", "c-1", "p-1",
                {"engine_session_key": "s"},
            )
        handler._reply_workflow_error.assert_called()  # type: ignore[attr-defined]
        handler.reply_text.assert_not_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Node.js version string consistency: every user-visible Node.js version gate
# in ``src/`` must match :data:`NODE_MIN_VERSION` from constants.
# ---------------------------------------------------------------------------

import re


def _iter_source_files(root: str):
    """Yield .py paths under *root*, excluding __pycache__ and tests/."""
    for dirpath, dirnames, filenames in os.walk(root):
        # prune cache directories
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            yield os.path.join(dirpath, fname)


_NODE_VERSION_RE = re.compile(r"Node\.?js\s*[>=]+\s*(\d+)", re.IGNORECASE)


class TestNodeVersionStringConsistency(unittest.TestCase):
    """Scanning ``src/`` for any ``Node.js >= N`` string literal — *N* must
    match :data:`NODE_MIN_VERSION[0]` so that docs, error messages, and code
    never drift apart."""

    def test_node_version_messages_match_constant(self) -> None:
        from src.workflow_engine.constants import NODE_MIN_VERSION

        expected_major = NODE_MIN_VERSION[0]
        src_root = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "src",
        )

        mismatches: list[str] = []
        for path in _iter_source_files(src_root):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        for match in _NODE_VERSION_RE.finditer(line):
                            found_major = int(match.group(1))
                            if found_major != expected_major:
                                rel = os.path.relpath(
                                    path, os.path.dirname(os.path.dirname(__file__))
                                )
                                mismatches.append(
                                    f"{rel}:{lineno}: mentions Node.js >= {found_major}, "
                                    f"expected >= {expected_major} (NODE_MIN_VERSION[0])"
                                )
            except OSError:
                # Skip unreadable files
                continue

        self.assertEqual(
            mismatches, [],
            msg="\n".join(mismatches) if mismatches else "",
        )

    def test_internal_error_template_supports_detail(self) -> None:
        """The internal_error UI body must expose a {detail} placeholder so
        ``_build_error_card`` can surface sanitized operator-visible hints.
        """
        from src.card.ui_text import UI_TEXT

        body = UI_TEXT.get("workflow_error_internal_error_body", "")
        self.assertIn("{detail}", body, msg="internal_error body must expose {detail}")

    def test_build_error_card_surfaces_safe_details(self) -> None:
        """When ``_build_error_card`` receives a non-empty detail, the resulting
        markdown body must include that detail (sanitized) rather than silently
        dropping it behind a generic message.
        """
        handler, _engine = _handler_with_mocks()

        card = handler._build_error_card("internal_error", detail="脚本验证失败：非法元数据")
        # Walk the card structure and find the rendered markdown content.
        rendered_md = _extract_markdown_content(card)
        self.assertIn("脚本验证失败", rendered_md)

    def test_build_error_card_stable_without_detail(self) -> None:
        """When no detail is supplied the body must still render to valid
        markdown without leaving a bare ``{detail}`` visible to the user.
        """
        handler, _engine = _handler_with_mocks()
        card = handler._build_error_card("internal_error")
        rendered_md = _extract_markdown_content(card)
        self.assertNotIn("{detail}", rendered_md)
        self.assertTrue(len(rendered_md.strip()) > 0)


def _extract_markdown_content(card: dict) -> str:
    """Best-effort extract of the concatenated markdown strings appearing in
    the card elements.  The exact shape is an internal contract; this helper
    intentionally tolerates extra keys so renderer refactors are free to
    extend the surface.
    """
    pieces: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("tag") == "markdown" and isinstance(node.get("content"), str):
                pieces.append(node["content"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(card)
    return "\n".join(pieces)


if __name__ == "__main__":
    unittest.main()
