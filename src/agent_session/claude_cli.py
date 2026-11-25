"""Claude Code CLI session backend."""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from ..acp.claude_capabilities import strip_1m_suffix
from ..acp.claude_selection import (
    CLAUDE_DEFAULT_MODEL_TOKEN,
    split_claude_model_selection,
)
from ..acp.execution_instructions import AUTONOMOUS_EXECUTION_PROMPT
from ..acp.models import ACPEvent, ACPEventType, PromptResult
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
        label="ClaudeCLI",
    )


@dataclass
class ClaudeCLIConfig:
    """Configuration knobs for Claude Code CLI backend."""

    command: str = "claude"
    add_dir: bool = True


class SyncClaudeCLISession(_BaseCLISession):
    """Claude Code CLI backend.

    - Uses `claude -p` (print and exit) per prompt.
    - Uses `--session-id` for the first prompt and `--resume <id>` afterwards.
    - Emits TEXT_CHUNK ACP events only (no plan/tool events).
    """

    _display_name = "Claude"
    _log_prefix = "ClaudeCLI"
    _missing_binary_prefix = "Claude CLI"

    def __init__(
        self,
        cwd: str,
        config: Optional[ClaudeCLIConfig] = None,
        *,
        model_name: Optional[str] = None,
    ):
        super().__init__(
            cwd,
            config or ClaudeCLIConfig(),
            model_name=model_name,
        )

    def _module_hooks(self) -> _CLIModuleHooks:
        return _CLIModuleHooks(
            popen=subprocess.Popen,
            which=shutil.which,
            terminate_and_reap=_terminate_and_reap_process,
            terminate_grace_s=_CLI_TERMINATE_GRACE_S,
            logger=logger,
        )

    def start(self, startup_timeout: float = 60) -> str:
        # No long-running server here; just validate executable and mint a session id.
        super().start(startup_timeout)
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        return self.session_id

    def _ensure_session_ready(self, which: Callable[[str], Optional[str]]) -> None:
        del which
        if not self.session_id:
            self.start()

    def _build_args(self, text: str, resumed: bool) -> list[str]:
        from ..acp.native_commands import is_non_task_native_slash

        stripped = str(text or "").strip()
        non_task_slash = is_non_task_native_slash(stripped)

        # Print mode otherwise falls back to dontAsk and denies tools that
        # would need a permission prompt. Apply the backend's own unattended
        # authorization mode on every invocation, including resumed turns.
        args: list[str] = [
            self._cfg.command,
            "-p",
            "--permission-mode",
            "bypassPermissions",
        ]
        if not non_task_slash:
            args += [
                "--append-system-prompt",
                AUTONOMOUS_EXECUTION_PROMPT,
            ]
        if self._cfg.add_dir:
            args += ["--add-dir", self._cwd]
        base_model, effort = split_claude_model_selection(self._model_name)
        if base_model and base_model != CLAUDE_DEFAULT_MODEL_TOKEN:
            args += ["--model", strip_1m_suffix(base_model)]
        if effort:
            args += ["--effort", effort]

        if resumed:
            args += ["--resume", self.session_id]
        else:
            args += ["--session-id", self.session_id]

        args.append("--")
        args.append(stripped if non_task_slash else text)
        return args

    def _prepare_env(self, env: dict[str, str]) -> None:
        # Claude Code CLI refuses to launch inside another Claude Code session.
        # Apply model-implied betas on top of the sanitized base environment.
        from ..utils.env import apply_anthropic_betas

        apply_anthropic_betas(env, self._model_name)

    def _consume_stdout_line(
        self,
        line: str,
        output_chunks: list[str],
        emit: Callable[[ACPEvent], None],
    ) -> str:
        output_chunks.append(line)
        emit(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=line))
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
        del state, backend_error
        output = out
        if rc != 0 and err:
            output = (output + "\n" + err).strip("\n")
            if on_event:
                on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="\n" + err))

        self.is_resumed = True
        stop_reason = "end_turn" if rc == 0 else "failed"
        return PromptResult(stop_reason=stop_reason, text=output)
