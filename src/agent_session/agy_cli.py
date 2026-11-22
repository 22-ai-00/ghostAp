"""AGY CLI stream-json session backend for Orchestrator custom Planner/Executor effort."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

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


def _parse_int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


@dataclass
class AGYCLIConfig:
    """Configuration knobs for AGY CLI headless stream-json execution."""

    command: str = "agy"
    show_thinking: bool = True


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
        super().__init__(
            cwd,
            config or AGYCLIConfig(),
            model_name=model_name,
        )
        self._agent_type = "agy"
        self._last_usage: Optional[ACPUsageInfo] = None
        self._turn_segments: list[str] = []
        self._turn_current_segment: list[str] = []
        self._turn_result_status: str = ""
        self._turn_result_response: str = ""
        self._turn_result_error: str = ""
        self._started_tool_steps: set[int] = set()
        self._streamed_response_steps: set[int] = set()

    @property
    def supports_background_tool_continuation(self) -> bool:
        return False

    def _reset_turn_tracking(self) -> None:
        self._turn_segments.clear()
        self._turn_current_segment.clear()
        self._turn_result_status = ""
        self._turn_result_response = ""
        self._turn_result_error = ""
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

    def _send_prompt_once(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
        activity_predicate: Optional[Callable[[ACPEvent], bool]] = None,
    ) -> PromptResult:
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
        from ..acp.native_commands import is_pure_zero_arg_native_slash, parse_slash_command

        base_model, settings_payload = resolve_agy_orchestrator_settings(self._model_name)
        if self._cfg.show_thinking:
            settings_payload["showThinking"] = True

        stripped = str(text or "").strip()
        if is_pure_zero_arg_native_slash(stripped):
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
            "--model",
            base_model,
            "--settings",
            json.dumps(settings_payload, ensure_ascii=False),
        ]
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
            init_payload = event.get("init") if isinstance(event.get("init"), dict) else {}
            conversation_id = str(
                event.get("conversation_id")
                or init_payload.get("conversation_id")
                or ""
            ).strip()
            if conversation_id:
                self.session_id = conversation_id
            return ""

        if event_type == "step_update":
            step_payload = (
                event.get("step_update")
                if isinstance(event.get("step_update"), dict)
                else event
            )
            return self._consume_step_update(step_payload, output_chunks, emit)

        if event_type == "result":
            result_payload = (
                event.get("result")
                if isinstance(event.get("result"), dict)
                else event
            )
            return self._consume_result_event(result_payload, output_chunks, emit)

        if event_type == "error":
            return self._extract_error_message(event)

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
        if response_text:
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
        return []

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
