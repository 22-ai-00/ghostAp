"""Codex CLI (``codex-w`` bridge) session backend."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from ..acp.execution_instructions import AUTONOMOUS_EXECUTION_PROMPT, append_execution_instructions
from ..acp.model_selection import (
    CODEX_DEFAULT_MODEL_TOKEN,
    split_codex_model_selection,
)
from ..acp.models import (
    ACPEvent,
    ACPEventType,
    PromptResult,
    TerminalOutputProvenance,
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
    """Bounded TERM→KILL cleanup for one CLI subprocess tree."""
    return terminate_and_reap_process_tree(
        proc,
        process_group_id=process_group_id,
        terminate_grace=_CLI_TERMINATE_GRACE_S,
        kill_grace=_CLI_KILL_GRACE_S,
        label="CodexCLI",
    )


@dataclass
class CodexCLIConfig:
    """Configuration knobs for the Codex CLI bridge backend."""

    command: str = "codex"
    add_dir: bool = True


class SyncCodexCLISession(_BaseCLISession):
    """Codex CLI backend bridged through the local ``codex-w`` executable.

    - Uses ``codex exec`` (print and exit) per prompt.
    - The first prompt starts a fresh thread; later prompts call
      ``codex exec resume <thread_id>``.
    - Emits TEXT_CHUNK ACP events for assistant ``agent_message`` items only.

    Two ``codex`` CLI quirks shape the argument layout (both verified against
    codex-cli 0.154.0 through the ``codex-w`` gateway wrapper):

    1. ``-c key=value`` overrides must be placed *before* the ``exec``
       subcommand.  Overrides placed after ``exec`` make codex fail to resolve
       model metadata and fall back to a ChatGPT-account route that rejects the
       LLMBox model with HTTP 400.
    2. A working directory outside a trusted git repository requires
       ``--skip-git-repo-check``; passing it unconditionally is harmless inside
       a repository.
    """

    _display_name = "Codex"
    _log_prefix = "CodexCLI"
    _missing_binary_prefix = "Codex CLI"
    _stdin_devnull = True

    def __init__(
        self,
        cwd: str,
        config: Optional[CodexCLIConfig] = None,
        *,
        model_name: Optional[str] = None,
    ):
        super().__init__(
            cwd,
            config or CodexCLIConfig(),
            model_name=model_name,
        )
        self._turn_final_answer_by_id: dict[str, list[str]] = {}
        self._turn_segments: list[str] = []
        self._turn_current_segment: list[str] = []

    def _reset_turn_terminal_tracking(self) -> None:
        self._turn_final_answer_by_id.clear()
        self._turn_segments.clear()
        self._turn_current_segment.clear()

    def _split_turn_segment(self) -> None:
        if self._turn_current_segment:
            segment = "".join(self._turn_current_segment)
            self._turn_current_segment.clear()
            if segment.strip():
                self._turn_segments.append(segment)

    def _consume_turn_terminal_tracking(self) -> tuple[str, str]:
        self._split_turn_segment()
        final_answer_text = ""
        for chunks in reversed(list(self._turn_final_answer_by_id.values())):
            candidate = "".join(chunks).strip()
            if candidate:
                final_answer_text = candidate
                break
        main_final_segment = (
            self._turn_segments[-1].strip() if self._turn_segments else ""
        )
        self._reset_turn_terminal_tracking()
        return final_answer_text, main_final_segment

    def _send_prompt_once(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
        activity_predicate: Optional[Callable[[ACPEvent], bool]] = None,
    ) -> PromptResult:
        self._reset_turn_terminal_tracking()
        try:
            return super()._send_prompt_once(
                text,
                on_event=on_event,
                timeout=timeout,
                idle_timeout=idle_timeout,
                activity_predicate=activity_predicate,
            )
        finally:
            self._reset_turn_terminal_tracking()

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
            raise RuntimeError(f"未找到 Codex CLI 可执行文件: {self._cfg.command}")

    def _mark_resumed_after_error(self) -> None:
        self.is_resumed = bool(self.session_id)

    def _build_args(self, text: str, resumed: bool) -> list[str]:
        from ..acp.native_commands import is_non_task_native_slash, parse_slash_command

        # Global ``-c`` overrides MUST precede the ``exec`` subcommand;
        # codex ignores/mis-resolves them afterwards (model metadata 400).
        args: list[str] = [self._cfg.command]
        base_model, effort = split_codex_model_selection(self._model_name)
        if base_model and base_model != CODEX_DEFAULT_MODEL_TOKEN:
            args += ["-c", f'model="{base_model}"']
        if effort:
            args += ["-c", f'model_reasoning_effort="{effort}"']

        args += [
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-C",
            self._cwd,
        ]
        if resumed and self.session_id:
            args += ["resume", self.session_id]
        stripped = str(text or "").strip()
        if is_non_task_native_slash(stripped):
            args.append(stripped)
        elif parse_slash_command(stripped) is not None:
            args.append(f"{stripped}\n\n{AUTONOMOUS_EXECUTION_PROMPT}")
        else:
            args.append(append_execution_instructions(text))
        return args

    def _filter_stderr(self, err: str) -> str:
        # Keep codex's noisy Rust tracing out of the surfaced stderr.
        return "\n".join(
            ln
            for ln in err.splitlines()
            if not ln.startswith(("20", "Reading additional input"))
        ).strip("\n")

    def _consume_stdout_line(
        self,
        line: str,
        output_chunks: list[str],
        emit: Callable[[ACPEvent], None],
    ) -> str:
        payload = line.strip()
        if not payload:
            return ""
        return self._consume_event(
            payload,
            output_chunks,
            lambda chunk: emit(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=chunk)),
            emit_event=emit,
        )

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
        final_answer_text, main_final_segment = (
            self._consume_turn_terminal_tracking()
        )
        if not backend_error and rc != 0:
            backend_error = err or f"codex exec 退出码 {rc}"
        output = out
        if rc != 0 or backend_error:
            detail = backend_error or err
            if detail:
                output = (output + "\n" + detail).strip("\n")
                if on_event:
                    on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="\n" + detail))
            if final_answer_text:
                terminal_text = final_answer_text
                provenance = TerminalOutputProvenance.FINAL_ANSWER_PHASE.value
                tracked = True
            elif detail:
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
                terminal_text=terminal_text,
                terminal_text_provenance=provenance,
                terminal_text_tracked=tracked,
            )

        if final_answer_text:
            terminal_text = final_answer_text
            provenance = TerminalOutputProvenance.FINAL_ANSWER_PHASE.value
        elif main_final_segment:
            terminal_text = main_final_segment
            provenance = TerminalOutputProvenance.MAIN_FINAL_SEGMENT.value
        else:
            terminal_text = ""
            provenance = TerminalOutputProvenance.NONE.value
        self.is_resumed = bool(self.session_id)
        return PromptResult(
            stop_reason="end_turn",
            text=output,
            terminal_text=terminal_text,
            terminal_text_provenance=provenance,
            terminal_text_tracked=True,
        )

    def _consume_event(
        self,
        payload: str,
        assistant_chunks: list[str],
        emit_text: Callable[[str], None],
        emit_event: Optional[Callable[[ACPEvent], None]] = None,
    ) -> str:
        """Parse one ``--json`` JSONL event.

        Returns a non-empty human-readable message for a fatal backend error,
        or an empty string otherwise.
        """
        try:
            event = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return ""
        event_type = str(event.get("type") or "")

        if event_type == "thread.started":
            thread_id = str(event.get("thread_id") or "").strip()
            if thread_id and not self.session_id:
                self.session_id = thread_id
            return ""

        if event_type in {"error", "turn.failed"}:
            message = self._extract_error_message(event)
            if message:
                logger.warning("[CodexCLI] backend error: %s", message)
            return message

        if event_type in {"item.started", "item.completed"}:
            item = event.get("item") or {}
            if not isinstance(item, dict):
                return ""
            item_type = str(item.get("type") or "")
            if item_type != "agent_message":
                self._split_turn_segment()
                return ""
            if event_type != "item.completed":
                return ""
        else:
            return ""

        text = str(item.get("text") or "")
        if not text:
            return ""
        raw_id = item.get("id")
        if raw_id is None:
            raw_id = item.get("message_id")
        message_id = (
            raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else None
        )
        raw_phase = item.get("phase")
        normalized_phase = (
            raw_phase.strip().casefold() if isinstance(raw_phase, str) else ""
        )
        codex_message_phase = (
            normalized_phase
            if normalized_phase in {"commentary", "final_answer"}
            else None
        )
        self._turn_current_segment.append(text)
        if codex_message_phase == "final_answer":
            self._turn_final_answer_by_id.setdefault(message_id or "", []).append(
                text
            )
        assistant_chunks.append(text)
        if emit_event is not None:
            emit_event(
                ACPEvent(
                    event_type=ACPEventType.TEXT_CHUNK,
                    text=text,
                    message_id=message_id,
                    codex_message_phase=codex_message_phase,
                )
            )
        else:
            emit_text(text)
        return ""

    @staticmethod
    def _extract_error_message(event: dict) -> str:
        """Pull a human-readable message out of error / turn.failed payloads."""
        raw = event.get("error")
        if isinstance(raw, dict):
            inner = raw.get("message")
            if isinstance(inner, str):
                # Codex sometimes double-encodes a JSON error body as a string.
                try:
                    parsed = json.loads(inner)
                    nested = (
                        parsed.get("error", {}).get("message")
                        if isinstance(parsed, dict)
                        else None
                    )
                    if isinstance(nested, str) and nested:
                        return nested
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
                return inner
        message = event.get("message")
        if isinstance(message, str):
            return message
        item = event.get("item")
        if isinstance(item, dict):
            inner_message = item.get("message")
            if isinstance(inner_message, str):
                return inner_message
        return ""
