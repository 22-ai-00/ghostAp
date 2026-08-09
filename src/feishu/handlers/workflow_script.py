"""Workflow script generation and confirm card building.

Extracted from workflow.py to reduce handler size. Contains AI script
generation, confirm card construction, and workflow execution callbacks.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from src.card.render.buttons import build_responsive_button_row

if TYPE_CHECKING:
    from ...project import ProjectContext

logger = logging.getLogger(__name__)


class WorkflowScriptMixin:
    """Mixin providing script generation and confirm card building."""

    def _get_root_path(self, chat_id: str, project: Optional["ProjectContext"]) -> str:
        """Resolve root_path from project or chat."""
        if project:
            return project.root_path
        return self.get_working_dir(chat_id)

    def _reply_workflow_completion_fallback(
        self,
        *,
        message_id: str,
        report_status: dict[str, Any] | None,
    ) -> None:
        """Reply with terminal/report state without leaking partial raw output."""
        lines = ["✅ Workflow 已结束", "结果卡发送失败，未展示不完整结果。"]
        status = report_status or {}
        if status.get("attachment_sent"):
            lines.append("完整 HTML 报告已回复到当前话题。")
        elif status.get("generated"):
            filename = status.get("html_filename")
            if not filename and status.get("html_path"):
                filename = os.path.basename(str(status["html_path"]))
            filename = str(filename or "Workflow HTML 报告")
            if len(filename.encode("utf-8", errors="surrogatepass")) > 800:
                filename = "Workflow HTML 报告"
            lines.append(f"完整报告已保存在本地：{filename}")
        else:
            lines.append("完整报告未生成，请查看服务日志。")
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

        im_client = getattr(self, "im_client", None)
        if im_client is None:
            return {
                "generated": False,
                "attachment_sent": False,
                "error": "IM 客户端不可用，未生成附件报告",
            }

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

    def _generate_script_via_ai(
        self,
        requirement: str,
        root_path: str,
        selected_tools: list[str] | None = None,
        engine: Any = None,
        progress_callback: Any = None,
    ) -> tuple[str, dict[str, Any] | None, bool]:
        """Generate a workflow script via AI with fallback to simple generation.

        Args:
            requirement: The user's requirement text.
            root_path: Project root path.
            selected_tools: Optional list of tools selected by the user. If provided,
                the script generator will be encouraged to use these tools.
            engine: Optional workflow engine instance. If provided, the selected
                orchestrator_agent from pending state will be used for script generation.

        Returns:
            Tuple of (script_path, meta_dict_or_None, is_fallback).
        """
        from ...agent_session import close_session_safely, create_engine_session
        from ...workflow_engine.constants import (
            DEFAULT_ORCHESTRATOR_AGENT,
            SCRIPT_GEN_TIMEOUT_S,
        )
        from ...workflow_engine.script_gen import (
            build_script_gen_prompt,
            extract_meta_from_script,
            validate_generated_script,
        )

        # Resolve agent type: use pending.orchestrator_agent if available, otherwise default
        agent_type = (
            engine.project.pending.orchestrator_agent
            if engine and engine.project and engine.project.pending and engine.project.pending.orchestrator_agent
            else DEFAULT_ORCHESTRATOR_AGENT
        )

        script_dir = os.path.join(root_path, ".ghostap", "workflow_scripts")
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, "generated_workflow.js")

        # Resolve available tools via dynamic registry
        from ...workflow_engine.tool_registry import get_available_tools

        available_tools = get_available_tools(require_available=True)
        if not available_tools:
            logger.warning("No executable workflow tools detected; using fallback script")
            return self._write_fallback_script(script_path, requirement, selected_tools), None, True

        # Filter available tools to selected ones if provided
        if selected_tools:
            available_tools = {
                k: v for k, v in available_tools.items()
                if k in selected_tools
            }
            if not available_tools:
                logger.warning("Selected workflow tools are unavailable; using fallback script")
                return self._write_fallback_script(script_path, requirement, selected_tools), None, True

        # Get orchestrator binding and review agents from pending state
        orchestrator_binding = None
        review_agents = None
        selected_model_name = None
        if engine and engine.project and engine.project.pending:
            orchestrator_binding = engine.project.pending.orchestrator_binding
            review_agents = engine.project.pending.review_agents
            # Extract model_name from orchestrator_binding if not using default
            if orchestrator_binding and not orchestrator_binding.use_default_model:
                selected_model_name = orchestrator_binding.model_name

        prompt = build_script_gen_prompt(
            requirement=requirement,
            available_tools=available_tools,
            orchestrator_agent=agent_type,
            orchestrator_binding=orchestrator_binding,
            review_agents=review_agents,
        )

        # Attempt AI generation via one-shot ACP session
        #
        # SECURITY: Script generation runs an untrusted model against a
        # user-supplied requirement. We disable ``auto_approve`` and attach
        # a read-only tool filter so the generation session cannot mutate
        # the filesystem, execute arbitrary commands, or reach the network
        # even if the model tries to. The user confirms the *generated*
        # script later in the workflow card.
        session = None
        try:
            session = create_engine_session(
                agent_type=agent_type,
                cwd=root_path,
                thread_id="workflow_script_gen",
                auto_approve=False,
                require_tool_filter=True,
                model_name=selected_model_name,
            )
            if session is None:
                logger.warning("Failed to create script-gen session; using fallback")
                return self._write_fallback_script(script_path, requirement, selected_tools), None, True

            if progress_callback:
                progress_callback("已创建 AI 会话，正在发送生成请求...")

            # Apply read-only tool filter for script generation. The allowed
            # set is intentionally small: only read-side filesystem + search
            # introspection. Mutation tools (write, shell, network, code
            # execution) are rejected so a compromised model cannot bypass
            # the user confirmation step.
            _MUTATING_TOOLS: frozenset[str] = frozenset([
                "execute_command",
                "create_terminal",
                "run_terminal",
                "run_shell",
                "shell",
                "bash",
                "write_file",
                "write_text_file",
                "delete_file",
                "remove_file",
                "mkdir",
                "patch_file",
                "apply_diff",
                "edit_file",
                "write_to_file",
                "http_request",
                "http_get",
                "http_post",
                "fetch",
                "download",
                "upload",
                "network_request",
                "url_open",
                "send_message",
                "send_email",
                "create_issue",
            ])

            def _script_gen_tool_filter(tool_name: str, _params: dict | None) -> bool:
                if not isinstance(tool_name, str):
                    return False
                norm = tool_name.lower().strip()
                if norm in _MUTATING_TOOLS:
                    return False
                # Reject any tool whose name hints at a mutation.
                if any(token in norm for token in ("write", "delete", "remove", "exec", "run", "patch", "post", "upload", "send", "create")):
                    return False
                return True

            try:
                session.set_tool_filter(_script_gen_tool_filter)
            except (AttributeError, TypeError, Exception) as exc:
                # FAIL-CLOSED: If we cannot enforce the read-only tool filter
                # on the script-gen session, we must not proceed to call the
                # model — otherwise a compromised or confused model could
                # mutate the filesystem or execute arbitrary commands before
                # the user has confirmed the script. Instead, fall back to a
                # static pre-vetted fallback script so the workflow path still
                # reaches the confirmation card.
                logger.warning(
                    "Applying script-gen tool filter failed (%s); fail-closed to "
                    "static fallback script; model prompt is NOT sent.",
                    type(exc).__name__,
                )
                from ...workflow_engine.script_gen import FALLBACK_SCRIPT
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(FALLBACK_SCRIPT)
                if session is not None:
                    close_session_safely(session)
                meta = {"name": "fallback-orchestration"}
                return script_path, meta, True

            script_gen_timeout_s = getattr(
                self.settings, "workflow_script_gen_timeout_s", SCRIPT_GEN_TIMEOUT_S
            )

            # Retry loop: feed validation errors back to LLM for correction
            max_script_retries = 3
            last_errors: list[str] = []
            for gen_attempt in range(max_script_retries):
                if gen_attempt == 0:
                    current_prompt = prompt
                else:
                    # Build retry prompt with validation error feedback
                    error_summary = "\n".join(f"- {e}" for e in last_errors)
                    current_prompt = (
                        f"Your previous script had validation errors:\n{error_summary}\n\n"
                        f"Please regenerate the workflow script fixing ALL the above errors. "
                        f"Output ONLY the corrected JavaScript code, no explanations.\n\n"
                        f"Original requirement: {requirement}"
                    )
                    if progress_callback:
                        progress_callback(f"脚本验证失败，正在重试 ({gen_attempt + 1}/{max_script_retries})...")

                result = session.send_prompt(current_prompt, timeout=script_gen_timeout_s)

                if progress_callback and gen_attempt == 0:
                    progress_callback("收到模型响应，正在验证脚本...")

                if result and result.text:
                    script_content = self._strip_markdown_fences(result.text.strip())

                    is_valid, errors = validate_generated_script(script_content, review_agents=review_agents)
                    if is_valid:
                        with open(script_path, "w", encoding="utf-8") as f:
                            f.write(script_content)
                        meta = extract_meta_from_script(script_content)
                        if meta is None:
                            meta = {}
                        return script_path, meta, False
                    else:
                        last_errors = errors
                        logger.warning(
                            "Generated script failed validation (attempt %d/%d): %s",
                            gen_attempt + 1, max_script_retries, errors,
                        )
                else:
                    logger.warning("AI returned empty script content (attempt %d/%d)", gen_attempt + 1, max_script_retries)
                    break  # Empty response won't improve with retry

        except Exception as exc:
            logger.error("Script generation via AI failed: %s", exc, exc_info=True)
        finally:
            if session is not None:
                close_session_safely(session)

        # Fallback
        return self._write_fallback_script(script_path, requirement, selected_tools), None, True

    @staticmethod
    def _strip_markdown_fences(content: str) -> str:
        """Remove markdown code fences and natural language preamble from AI output.

        AI models sometimes prefix their code output with explanatory text like
        "Let me analyze..." or "Here's the workflow script:". This method
        extracts the actual JavaScript code by:
        1. Attempting to extract code from markdown fences (even if preceded by text)
        2. Stripping any natural language preamble before the actual JS code
        """
        import re

        # Strategy 1: Find markdown code fence containing the actual code.
        # This handles cases like: "Here's the script:\n```javascript\n...code...\n```"
        fence_match = re.search(r"```\s*(?:javascript|js|)\s*\n", content, re.IGNORECASE)
        if fence_match:
            after_fence = content[fence_match.end():]
            # Find the closing fence (last occurrence to handle nested fences in strings)
            close_idx = after_fence.rfind("```")
            if close_idx >= 0:
                content = after_fence[:close_idx].rstrip()
            else:
                content = after_fence.rstrip()
            # After extracting from fences, if it looks like valid JS, return it
            stripped = content.lstrip()
            if stripped and re.match(
                r"^(export|/[/*]|const |let |var |\"use strict\"|'use strict')",
                stripped,
            ):
                return content.strip()

        # Strategy 2: Original logic — content starts directly with a fence
        elif content.startswith("```"):
            lines = content.split("\n", 1)
            content = lines[1] if len(lines) > 1 else content
            if content.rstrip().endswith("```"):
                content = content.rstrip()[:-3].rstrip()

        # Strategy 3: Detect and strip natural language preamble.
        # If content doesn't start with valid JS syntax, find the actual code start.
        stripped = content.lstrip()
        if stripped and not re.match(
            r"^(export|/[/*]|const |let |var |\"use strict\"|'use strict'|/\*\*)",
            stripped,
        ):
            # Look for the start of the actual export statement (multiline search)
            export_match = re.search(
                r"^(export\s+const\s+meta\s*=|export\s+default\s)",
                content,
                re.MULTILINE,
            )
            if export_match:
                start_idx = export_match.start()
                # Include preceding JSDoc/comment lines that are part of the code
                preceding = content[:start_idx]
                if preceding.rstrip():
                    lines_before = preceding.rstrip().split("\n")
                    # Walk backwards to include leading comment block
                    comment_start = start_idx
                    for line in reversed(lines_before):
                        ls = line.strip()
                        if ls.startswith("//") or ls.startswith("*") or ls.startswith("/*") or ls.endswith("*/"):
                            # This line is a comment, include it
                            idx = content.rfind(line, 0, comment_start)
                            if idx >= 0:
                                comment_start = idx
                        else:
                            break
                    start_idx = comment_start
                content = content[start_idx:]

        return content.strip()

    @staticmethod
    def _find_function_close_brace(script_content: str, open_idx: int) -> int | None:
        """Return the index of the matching ``}`` for the function body starting at ``open_idx``.

        String literals (``'...'``, ``\"...\"``, ```...` ``) and comments
        (``// ...``, ``/* ... */``) are skipped so braces inside them do not
        affect the depth counter. Backslash escapes are respected inside
        string literals. Returns ``None`` if the end of file is reached
        without finding the closing brace at depth 0.
        """
        depth = 1  # caller already located the opening `{`
        i = open_idx + 1
        n = len(script_content)
        while i < n:
            ch = script_content[i]
            if ch == "/" and i + 1 < n:
                nxt = script_content[i + 1]
                if nxt == "/":
                    # Line comment: skip to end-of-line.
                    j = script_content.find("\n", i + 2)
                    i = n if j == -1 else j
                    continue
                if nxt == "*":
                    # Block comment: skip to closing */.
                    j = script_content.find("*/", i + 2)
                    i = n if j == -1 else j + 2
                    continue
            if ch in ("'", '"', "`"):
                quote = ch
                j = i + 1
                while j < n:
                    c = script_content[j]
                    if c == "\\" and j + 1 < n:
                        j += 2
                        continue
                    if c == quote:
                        break
                    j += 1
                i = j + 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return None

    @staticmethod
    def _read_pending_script(engine: Any) -> str:
        """Read script content from pending.script_path for confirm card preview."""
        path = engine.project.pending.script_path if (engine.project and engine.project.pending) else None
        if not path:
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""

    @staticmethod
    def _inject_workflow_refs_into_script(script_content: str, refs: list[dict]) -> str:
        """Inject sub-workflow references into the script body.

        Each ref dict may contain:

        - ``name`` (required, string): the template name to invoke.
        - ``args`` (optional, dict): keyword arguments forwarded to the
          sub-workflow call. Defaults to an empty object.
        - ``failure_policy`` (optional, string): ``"fail_fast"`` raises on
          error; ``"skip"`` (default) logs and continues. Any unrecognised
          value is treated as ``"skip"``.
        - ``description`` (optional, string): free-form text surfaced in the
          injected comment so editors can trace why a ref exists.

        Order of operations:

        1. If ``// <<WORKFLOW_REFS_BEGIN>>`` / ``// <<WORKFLOW_REFS_END>>`` markers
           exist in ``script_content``, the block between them is replaced with
           the generated workflow-ref calls (honoring the author's placement).
        2. Otherwise, refs are injected just before the last ``}`` closing the
           ``export default async function [NAME](...)`` body.  Both anonymous
           and named default functions are supported, matching the template
           style used by the built-in ``code-audit`` and similar templates.
        3. String/comment literal boundaries are respected when searching for
           the function's closing brace: a ``}`` inside ``'...'``, ``\"...\"``,
           ```...` ``, or a ``// ...`` / ``/* ... */`` comment does not count
           toward brace depth.
        4. If a ref name is already invoked by name anywhere in the script
           (via ``workflow('name'`` or ``workflow("name"``) we skip generating
           a duplicate call for it.

        Args:
            script_content: The raw JavaScript script content.
            refs: A list of reference dicts. Only refs with a non-empty
                ``name`` are processed.

        Returns:
            The updated script content with refs injected.
        """
        if not refs:
            return script_content

        import json as _json
        import re as _re

        # Build the list of refs to inject, de-duplicated, and skip any that
        # already have a matching ``workflow('...'`` call in the script.
        refs_to_inject: list[dict] = []
        seen: set[str] = set()
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            name = ref.get("name")
            if not name or not isinstance(name, str):
                continue
            if name in seen:
                continue
            if _re.search(
                r"\bworkflow\s*\(\s*[\"']" + _re.escape(name) + r"[\"']",
                script_content,
            ):
                # Already present — skip to avoid duplication.
                seen.add(name)
                continue
            seen.add(name)
            refs_to_inject.append(ref)

        if not refs_to_inject:
            return script_content

        generated_lines = []
        for ref in refs_to_inject:
            name = ref["name"]
            args_obj = ref.get("args") or {}
            try:
                args_json = _json.dumps(args_obj, ensure_ascii=False)
            except (TypeError, ValueError):
                args_json = "{}"
            policy = (ref.get("failure_policy") or "skip").lower()
            desc = ref.get("description") or ""
            safe_desc = desc.replace("\n", " ").replace("\r", " ")

            if policy == "fail_fast":
                call = f"  await workflow('{name}', {args_json});"
                header = (
                    f"  // ref: {name}{(' -- ' + safe_desc) if safe_desc else ''}"
                    f"  (failure_policy=fail_fast)"
                )
            else:
                call = (
                    f"  try {{ await workflow('{name}', {args_json}); }} "
                    f"catch (e) {{ console.log('sub-workflow {name} skipped:', e); }}"
                )
                header = (
                    f"  // ref: {name}{(' -- ' + safe_desc) if safe_desc else ''}"
                )
            generated_lines.append(header)
            generated_lines.append(call)

        block = "\n".join(generated_lines) + "\n"

        # Strategy 1: replace marker block if present.
        marker_start = "// <<WORKFLOW_REFS_BEGIN>>"
        marker_end = "// <<WORKFLOW_REFS_END>>"
        idx_s = script_content.find(marker_start)
        if idx_s != -1:
            idx_e = script_content.find(marker_end, idx_s + len(marker_start))
            if idx_e != -1:
                return (
                    script_content[:idx_s]
                    + marker_start
                    + "\n"
                    + block
                    + marker_end
                    + script_content[idx_e + len(marker_end):]
                )

        # Strategy 2: find the default export function body (supporting both
        # ``export default async function (args)`` and
        # ``export default async function main(args = {})``) and inject just
        # before its final ``}``.  Brace depth is counted outside of strings
        # and comments; parameter-list braces (e.g. ``args = {}``) are
        # skipped by locating the parameter-list closing ``)`` first.
        default_match = _re.search(
            r"export\s+default\s+(?:async\s+)?function\s*(?:[A-Za-z_$][\w$]*)?\s*\(",
            script_content,
        )
        if default_match:
            # 1. Locate the matching ``)`` for the parameter list so braces
            #    inside default values (``args = {}``) are not mistaken for
            #    the function body opening brace.
            paren_open = default_match.end() - 1  # points at the opening `(`
            paren_depth = 1
            paren_i = paren_open + 1
            paren_n = len(script_content)
            close_paren = -1
            while paren_i < paren_n:
                c = script_content[paren_i]
                if c in ("'", '"', "`"):
                    q = c
                    j = paren_i + 1
                    while j < paren_n:
                        if script_content[j] == "\\" and j + 1 < paren_n:
                            j += 2
                            continue
                        if script_content[j] == q:
                            break
                        j += 1
                    paren_i = j + 1
                    continue
                if c == "(":
                    paren_depth += 1
                elif c == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        close_paren = paren_i
                        break
                paren_i += 1
            if close_paren != -1:
                # 2. Now locate the opening ``{`` of the function body,
                #    allowing for ``=>`` style or simply ``{`` after ``)``.
                body_open = script_content.find("{", close_paren + 1)
                if body_open != -1:
                    insert_at = WorkflowScriptMixin._find_function_close_brace(
                        script_content, body_open,
                    )
                    if insert_at is not None:
                        return (
                            script_content[:insert_at].rstrip()
                            + "\n"
                            + block
                            + "  "  # preserve closing-brace indentation
                            + script_content[insert_at:].lstrip("\n")
                        )

        # Fallback: append at end.
        return script_content.rstrip() + "\n\n" + block

    @staticmethod
    @staticmethod
    def _write_fallback_script(
        script_path: str,
        requirement: str,
        selected_tools: list[str] | None = None,
        tool_model_map: dict[str, str] | None = None,
    ) -> str:
        """Write a simple fallback script and return its path."""
        from ...workflow_engine.script_gen import generate_simple_script

        script_content = generate_simple_script(
            requirement, selected_tools=selected_tools, tool_model_map=tool_model_map
        )
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        return script_path

    def _build_confirm_card(
        self,
        meta: dict[str, Any] | None,
        requirement: str,
        engine_session_key: str,
        chat_id: str,
        project_id: str,
        is_fallback: bool = False,
        selected_tools: list[str] | None = None,
        script_content: str = "",
        orchestrator_binding: dict | None = None,
        review_agents: list[dict] | None = None,
    ) -> dict:
        """Build a Feishu card showing the workflow script preview for confirmation.

        Returns a Feishu card JSON dict ready for reply_card/send_card_to_chat.
        """
        from ...card import CardBuilder
        from ...card.actions.dispatch import (
            WORKFLOW_CANCEL,
            WORKFLOW_CONFIRM_START,
            WORKFLOW_SELECT_TOOL,
        )
        from ...card.render.budget import RenderBudget
        from ...card.ui_text import UI_TEXT
        from ...workflow_engine.tool_registry import get_available_tools

        # Extract meta info
        (meta or {}).get("name", "generated-workflow")
        (meta or {}).get("description", requirement[:100])
        phases = (meta or {}).get("phases", [])
        tools = (meta or {}).get("tools", selected_tools or [])
        phase_tool_mapping: dict = (meta or {}).get("phase_tool_mapping", {})
        workflow_refs = (meta or {}).get("workflow_refs", [])

        # Format orchestrator binding display
        orchestrator_display = ""
        if orchestrator_binding:
            tool_name = orchestrator_binding.tool_name
            model_name = orchestrator_binding.model_name
            use_default = getattr(orchestrator_binding, 'use_default_model', True)
            orchestrator_display = f"**主编排 Agent**: `{tool_name}`"
            if use_default:
                orchestrator_display += f" (默认: {orchestrator_binding.model_display_name or '默认模型'})"
            elif model_name:
                orchestrator_display += f" · {orchestrator_binding.model_display_name or model_name}"

        # Format review agents display
        review_display = ""
        if review_agents and len(review_agents) > 0:
            review_lines = ["**评审 Agent**:"]
            for i, agent in enumerate(review_agents):
                tool_name = agent.tool_name
                model_name = agent.model_name
                use_default = getattr(agent, 'use_default_model', True)
                line = f"{i+1}. `{tool_name}`"
                if use_default:
                    line += f" (默认: {agent.model_display_name or '默认模型'})"
                elif model_name:
                    line += f" · {agent.model_display_name or model_name}"
                review_lines.append(line)
            review_display = "\n".join(review_lines)
        elif review_agents is not None:
            review_display = "**评审 Agent**: Auto（跳过独立评审，使用主 Agent 自评审）"

        # Pre-compute has_mismatch for action button state (used in both modes)
        allowed_tools = set(selected_tools) if selected_tools else set(tools)
        script_tools = set(tools)
        has_mismatch = bool(script_tools - allowed_tools)

        # --- Node budget pre-check ---
        # Estimate element count and apply truncation if needed
        estimated_nodes = 0
        estimated_nodes += 5  # requirement, meta, hr, phases header, workflow refs
        if phases:
            estimated_nodes += len(phases)
        if script_content:
            estimated_nodes += 2  # script preview header + content
        if selected_tools:
            estimated_nodes += len(selected_tools)
        estimated_nodes += 10  # budget buttons, action buttons, etc.

        estimated_nodes > RenderBudget.NODE_BUDGET * 0.8

        # Shared helpers
        phase_count = len(phases)
        tool_count = len(selected_tools) if selected_tools else len(tools)

        # Build elements — unified layout across normal & truncated modes.
        elements: list[dict] = []

        # --- 1. Stepper (vertical, one-line per step) ---
        elements.append(self._build_workflow_stepper(current=3, total=3))

        if is_fallback:
            elements.append({
                "tag": "markdown",
                "content": "⚠️ AI 脚本生成失败，已使用默认模板。结果可能不完全匹配需求。",
            })

        # --- 2. Requirement summary (first screen) ---
        req_trim = requirement[:300] if len(requirement) > 300 else requirement
        elements.append({
            "tag": "markdown",
            "content": f"**需求**\n> {req_trim}",
        })

        # --- 2b. Agent selection display ---
        agent_info_lines = []
        if orchestrator_display:
            agent_info_lines.append(orchestrator_display)
        if review_display:
            agent_info_lines.append(review_display)
        if agent_info_lines:
            elements.append({"tag": "hr"})
            elements.append({
                "tag": "markdown",
                "content": "\n".join(agent_info_lines),
            })

        # --- 3. Stats: phases / tools (one-line pair) ---
        elements.append({
            "tag": "column_set",
            "flex_mode": "bisect",
            "background_style": "default",
            "columns": [
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "elements": [
                        {"tag": "markdown", "content": f"**{phase_count}**\n<font color='grey'>阶段数</font>"},
                    ],
                },
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "elements": [
                        {"tag": "markdown", "content": f"**{tool_count}**\n<font color='grey'>工具数</font>"},
                    ],
                },
            ],
        })

        # --- 4. Tool-mismatch status bar + single primary fix action ---
        if has_mismatch:
            missing = sorted(script_tools - allowed_tools)
            missing_display = ", ".join(f"`{m}`" for m in missing)
            elements.append({
                "tag": "markdown",
                "content": (
                    f"⚠️ 脚本需要这些工具但尚未启用：{missing_display}。"
                    " 点击下方『一键补齐缺失工具』即可放行执行。"
                ),
            })

        # --- 5. Primary CTA block — confirm start / cancel / mismatch fix ---
        # Visible on the first screen. Users do NOT need to open any
        # collapsible panel to unblock execution.
        from ...card.actions.dispatch import WORKFLOW_BACK_TO_TOOLS, WORKFLOW_FILL_MISSING_TOOLS

        confirm_value = {
            "action": WORKFLOW_CONFIRM_START,
            "chat_id": chat_id,
            "project_id": project_id,
            "engine_session_key": engine_session_key,
        }
        cancel_value = {
            "action": WORKFLOW_CANCEL,
            "chat_id": chat_id,
            "project_id": project_id,
            "engine_session_key": engine_session_key,
        }

        primary_buttons: list[dict] = []

        if has_mismatch:
            fill_missing_value = {
                "action": WORKFLOW_FILL_MISSING_TOOLS,
                "chat_id": chat_id,
                "project_id": project_id,
                "engine_session_key": engine_session_key,
            }
            back_tools_value = {
                "action": WORKFLOW_BACK_TO_TOOLS,
                "chat_id": chat_id,
                "project_id": project_id,
                "engine_session_key": engine_session_key,
            }
            primary_buttons.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": "➕ 一键补齐缺失工具"},
                "type": "primary",
                "value": fill_missing_value,
                "behaviors": [{"type": "callback", "value": fill_missing_value}],
            })
            primary_buttons.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": "↩️ 返回工具选择"},
                "type": "default",
                "value": back_tools_value,
                "behaviors": [{"type": "callback", "value": back_tools_value}],
            })

        confirm_disabled = has_mismatch
        confirm_disabled_tips = (
            "脚本需要的工具尚未全部启用，请先点击『一键补齐缺失工具』"
            if confirm_disabled
            else None
        )
        confirm_btn: dict = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": "✅ 确认执行"},
            "type": "primary" if not confirm_disabled else "default",
            "value": confirm_value,
            "behaviors": [{"type": "callback", "value": confirm_value}],
            "disabled": confirm_disabled,
        }
        if confirm_disabled_tips:
            confirm_btn["disabled_tips"] = {
                "tag": "plain_text",
                "content": confirm_disabled_tips,
            }
        primary_buttons.append(confirm_btn)

        primary_buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": "❌ 取消"},
            "type": "danger",
            "value": cancel_value,
            "behaviors": [{"type": "callback", "value": cancel_value}],
            "confirm": {
                "title": {"tag": "plain_text", "content": UI_TEXT["workflow_btn_confirm_cancel_title"]},
                "text": {"tag": "plain_text", "content": UI_TEXT["workflow_btn_confirm_cancel_body"]},
            },
        })

        elements.extend(build_responsive_button_row(primary_buttons, mobile_force_vertical=True))

        # --- 5b. Sub-workflow references (first screen, independent block) ---
        # Shown on the confirm card's first screen so users can inspect and
        # edit sub-workflow refs without opening the advanced panel.
        # The actual ``workflow('name', {})`` call is injected at confirm time
        # via ``_inject_workflow_refs_into_script`` — no patch is written to
        # the script on disk here.
        from ...card.actions.dispatch import (
            WORKFLOW_ADD_WORKFLOW_REF,
            WORKFLOW_REMOVE_WORKFLOW_REF,
            WORKFLOW_VIEW_WORKFLOW_REF,
        )

        ref_count = len(workflow_refs) if isinstance(workflow_refs, list) else 0
        elements.append({
            "tag": "markdown",
            "content": f"**🔗 子 Workflow 引用（{ref_count}）**",
        })

        if not workflow_refs:
            # Empty state chip so the "add" entry point is still obvious.
            elements.append({
                "tag": "markdown",
                "content": "<font color='grey'>暂无子 Workflow 引用。点击下方按钮可添加。</font>",
            })
        else:
            for idx, ref in enumerate(workflow_refs):
                if isinstance(ref, dict):
                    ref_name = ref.get("name", "unknown")
                    ref_desc = ref.get("description", "")
                else:
                    ref_name = str(ref)
                    ref_desc = ""

                chip_lines = [f"`{ref_name}`"]
                if ref_desc:
                    chip_lines.append(f"  <font color='grey' size='10'>{ref_desc[:80]}</font>")
                elements.append({
                    "tag": "markdown",
                    "content": "\n".join(chip_lines),
                })

                ref_buttons: list[dict] = []
                view_value = {
                    "action": WORKFLOW_VIEW_WORKFLOW_REF,
                    "ref_index": idx,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": engine_session_key,
                }
                ref_buttons.append({
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "👁 预览"},
                    "type": "default",
                    "value": view_value,
                    "behaviors": [{"type": "callback", "value": view_value}],
                })
                remove_value = {
                    "action": WORKFLOW_REMOVE_WORKFLOW_REF,
                    "ref_index": idx,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": engine_session_key,
                }
                ref_buttons.append({
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "🗑 移除"},
                    "type": "default",
                    "value": remove_value,
                    "behaviors": [{"type": "callback", "value": remove_value}],
                    "confirm": {
                        "title": {"tag": "plain_text", "content": "移除子 Workflow 引用？"},
                        "text": {"tag": "plain_text", "content": f"确定移除「{ref_name}」？"},
                    },
                })
                elements.extend(build_responsive_button_row(ref_buttons, mobile_force_vertical=True))

        # "Add reference" main button — primary entry point.
        add_value = {
            "action": WORKFLOW_ADD_WORKFLOW_REF,
            "chat_id": chat_id,
            "project_id": project_id,
            "engine_session_key": engine_session_key,
        }
        add_button = [{
            "tag": "button",
            "text": {"tag": "plain_text", "content": "➕ 添加子 Workflow 引用"},
            "type": "default",
            "value": add_value,
            "behaviors": [{"type": "callback", "value": add_value}],
        }]
        elements.extend(build_responsive_button_row(add_button, mobile_force_vertical=True))

        # --- 6. Phases panel (top-level collapsible) ---
        if phases:
            phase_elements = []
            for i, p in enumerate(phases, 1):
                title = p.get("title", p.get("name", f"Phase {i}"))
                detail = p.get("detail", "")
                line = f"**{i}. {title}**"
                if detail:
                    line += f"\n   {detail[:100]}"
                phase_tools = phase_tool_mapping.get(title) or phase_tool_mapping.get(str(i))
                if phase_tools:
                    tool_tags = ", ".join(f"`{t}`" for t in phase_tools)
                    line += f"\n   工具: {tool_tags}"
                phase_elements.append({"tag": "markdown", "content": line})
            elements.append({
                "tag": "collapsible_panel",
                "header": {
                    "title": {"tag": "plain_text", "content": f"📋 阶段列表 ({len(phases)})"},
                },
                "border": {"color": "blue", "corner_radius": "8px"},
                "expanded": False,
                "elements": phase_elements,
            })
        else:
            elements.append({
                "tag": "markdown",
                "content": "📋 **执行阶段**: Planning → Execution",
            })

        # --- 7. Script preview panel (top-level collapsible) ---
        if script_content:
            from ...workflow_engine.renderer import render_script_preview

            preview = render_script_preview(script_content)
            if preview:
                elements.append({
                    "tag": "collapsible_panel",
                    "expanded": False,
                    "header": {
                        "title": {"tag": "plain_text", "content": "📜 编排脚本预览"},
                    },
                    "border": {"color": "grey", "corner_radius": "8px"},
                    "elements": [{"tag": "markdown", "content": preview}],
                })

        # --- 8. Collapsible: Advanced options (tools / regen) ---
        # Everything below here is truly secondary; users open this only for
        # deeper inspection before confirming.
        advanced_elements: list[dict] = []

        # 8a. Tools detail + interactive toggle
        tool_descriptions = get_available_tools(require_available=True)
        recommended_order = ["traex", "claude", "codex", "aiden", "gemini", "coco"]
        tier1_tools = [t for t in recommended_order if t in allowed_tools]
        tier2_tools = [t for t in sorted(allowed_tools) if t not in recommended_order]

        tool_detail_elements: list[dict] = []
        tool_detail_elements.append({
            "tag": "markdown",
            "content": "📝 **脚本计划使用**: " + (" | ".join(f"`{t}`" for t in sorted(script_tools))),
        })
        tool_detail_elements.append({
            "tag": "markdown",
            "content": "✅ **允许执行的工具**（点击切换，脚本只能使用勾选的工具）：",
        })

        if tier1_tools:
            for tool in tier1_tools:
                desc = tool_descriptions.get(tool, tool)
                tool_detail_elements.append({"tag": "markdown", "content": f"- `{tool}`: {desc}"})

            tool_buttons = []
            for t in tier1_tools:
                is_selected = t in allowed_tools
                btn_value = {
                    "action": WORKFLOW_SELECT_TOOL,
                    "tool_name": t,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": engine_session_key,
                }
                tool_buttons.append({
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": f"{'✓ ' if is_selected else '○ '}{t}"},
                    "type": "primary" if is_selected else "default",
                    "value": btn_value,
                    "behaviors": [{"type": "callback", "value": btn_value}],
                })
            if tool_buttons:
                tool_detail_elements.extend(build_responsive_button_row(tool_buttons, mobile_force_vertical=True))

        if tier2_tools:
            tier2_elements = []
            for tool in tier2_tools:
                desc = tool_descriptions.get(tool, tool)
                tier2_elements.append({"tag": "markdown", "content": f"- `{tool}`: {desc}"})
            other_buttons = []
            for t in tier2_tools:
                is_selected = t in allowed_tools
                btn_value = {
                    "action": WORKFLOW_SELECT_TOOL,
                    "tool_name": t,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": engine_session_key,
                }
                other_buttons.append({
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": f"{'✓ ' if is_selected else '○ '}{t}"},
                    "type": "primary" if is_selected else "default",
                    "value": btn_value,
                    "behaviors": [{"type": "callback", "value": btn_value}],
                })
            tool_detail_elements.append({
                "tag": "collapsible_panel",
                "header": {
                    "title": {"tag": "plain_text", "content": f"🔧 更多工具 ({len(tier2_tools)})"},
                },
                "border": {"color": "grey", "corner_radius": "8px"},
                "expanded": False,
                "elements": [
                    *tier2_elements,
                    *build_responsive_button_row(other_buttons, mobile_force_vertical=True),
                ],
            })

        # 6e. Regenerate script (advanced)
        from ...card.actions.dispatch import WORKFLOW_REGENERATE_SCRIPT

        regen_elements: list[dict] = []
        regen_buttons: list[dict] = []
        regen_buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": "🔄 重新生成编排"},
            "type": "default",
            "value": {
                "action": WORKFLOW_REGENERATE_SCRIPT,
                "chat_id": chat_id,
                "project_id": project_id,
                "engine_session_key": engine_session_key,
            },
            "behaviors": [{
                "type": "callback",
                "value": {
                    "action": WORKFLOW_REGENERATE_SCRIPT,
                    "chat_id": chat_id,
                    "project_id": project_id,
                    "engine_session_key": engine_session_key,
                },
            }],
        })
        regen_elements.extend(build_responsive_button_row(regen_buttons, mobile_force_vertical=True))

        # --- Combine all advanced sections into one collapsed panel ---
        # Tools/regen are non-essential for quick confirmation. Grouping
        # them under one collapsed panel keeps the first screen focused on
        # the decision (confirm / cancel / fix).
        combined_panel_elements: list[dict] = []
        combined_panel_elements.extend(advanced_elements)
        combined_panel_elements.append({"tag": "hr"})
        combined_panel_elements.extend(tool_detail_elements)
        combined_panel_elements.append({"tag": "hr"})
        combined_panel_elements.extend(regen_elements)

        elements.append({
            "tag": "collapsible_panel",
            "header": {
                "title": {"tag": "plain_text", "content": "⚙️ 查看详细信息 / 更多操作（阶段 / 工具 / 脚本）"},
            },
            "border": {"color": "grey", "corner_radius": "8px"},
            "expanded": False,
            "elements": combined_panel_elements,
        })

        return CardBuilder._wrap_card(
            header_title="🔄 Workflow 确认",
            header_template=UI_TEXT["workflow_header_colors"].get("confirm", "turquoise"),
            elements=elements,
        )

    @staticmethod
    def _parse_template_args(args_text: str) -> dict[str, Any]:
        """Parse 'key=value key2=value2' into a dict."""
        args: dict[str, Any] = {}
        for token in args_text.split():
            if "=" in token:
                key, _, value = token.partition("=")
                # Try to parse as JSON literal (number, bool, null)
                import json

                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    args[key] = value
            else:
                # Positional arg → store as "target"
                args.setdefault("target", token)
        return args

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

    def _build_workflow_callbacks(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"],
    ):
        """Build WorkflowEngineCallbacks that update the Feishu card."""
        from ...workflow_engine.engine import WorkflowEngineCallbacks
        from .workflow_card_pages import WorkflowCardPageDelivery

        card_message_id: list[str | None] = [message_id]  # Mutable ref for card updates
        page_delivery = WorkflowCardPageDelivery(card_message_id)
        terminal_sent: list[bool] = [False]
        project_id = getattr(project, "project_id", "") or ""

        def on_progress(
            card_data: dict[str, Any] | list[dict[str, Any]],
        ) -> None:
            """Update the progress card in Feishu."""
            if terminal_sent[0]:
                logger.debug("Ignored workflow progress update after terminal card was sent")
                return
            try:
                # Inject a "停止" button while the workflow is still running so
                # users can stop it directly from the progress card. Guard so
                # any failure here never breaks the progress update itself.
                status_card = card_data[0] if isinstance(card_data, list) and card_data else card_data
                if isinstance(status_card, dict):
                    self._inject_workflow_stop_button(
                        status_card,
                        chat_id,
                        project_id,
                        is_running=self._workflow_is_running_for_card(chat_id, project),
                    )
                delivery_result = page_delivery.deliver(
                    card_data,
                    replace_or_send=self._replace_or_send_workflow_rendered_card,
                    chat_id=chat_id,
                )
                if delivery_result.status_message_id:
                    card_message_id[0] = delivery_result.status_message_id
            except Exception:
                logger.debug("Failed to update workflow progress card", exc_info=True)

        def on_done(wf_project) -> None:
            """Final completion — send a structured completion card."""
            terminal_sent[0] = True
            report_status: dict[str, Any] | None = None
            try:
                from ...workflow_engine.renderer import render_completion_card

                report_status = self._send_workflow_completion_report(
                    wf_project=wf_project,
                    chat_id=chat_id,
                    message_id=message_id,
                    project=project,
                )
                card_data = render_completion_card(wf_project, report_status=report_status)
                delivery_result = page_delivery.deliver(
                    card_data,
                    replace_or_send=self._replace_or_send_workflow_rendered_card,
                    chat_id=chat_id,
                    terminal=True,
                )
                if delivery_result.status_message_id:
                    card_message_id[0] = delivery_result.status_message_id
                if not delivery_result.status_delivered:
                    raise RuntimeError("Card delivery returned None")
            except Exception:
                self._reply_workflow_completion_fallback(
                    message_id=message_id,
                    report_status=report_status,
                )

        def on_error(error_msg: str) -> None:
            """Error notification — sanitize before showing to user."""
            terminal_sent[0] = True
            from ...workflow_engine.errors import _strip_internal_details

            workflow_category = self._workflow_error_card_category(error_msg)
            card = self._build_error_card(
                workflow_category,
                detail=_strip_internal_details(error_msg or ""),
            )
            self._replace_or_send_workflow_card(
                card_message_id=card_message_id[0],
                chat_id=chat_id,
                card=card,
            )

        def on_log(msg: str) -> None:
            logger.debug("[WorkflowHandler] log: %s", msg)

        return WorkflowEngineCallbacks(
            on_progress=on_progress,
            on_done=on_done,
            on_error=on_error,
            on_log=on_log,
        )

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
