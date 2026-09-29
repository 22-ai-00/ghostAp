"""Shared ACP transport lifecycle helpers."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from acp.task import InMemoryMessageQueue

from ..utils.env import build_clean_env


def build_acp_process_env(
    command: str,
    args: Sequence[str],
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Prefer the installed Codex for its official ACP adapter when available.

    Explicit CODEX_PATH values (including empty ones) remain authoritative. If
    Codex is absent from the cleaned PATH, the adapter keeps its bundled fallback.
    """
    env = build_clean_env(base)
    package = "@agentclientprotocol/codex-acp"
    official_codex_acp = Path(command).name == "codex-acp" or any(
        arg.removeprefix("--package=") == package
        or arg.removeprefix("--package=").startswith(f"{package}@")
        for arg in args
    )
    if official_codex_acp and "CODEX_PATH" not in env:
        codex_path = shutil.which("codex", path=env["PATH"])
        if codex_path is not None:
            env["CODEX_PATH"] = codex_path
    return env


class LateFrameTolerantMessageQueue(InMemoryMessageQueue):
    """Drop transport frames after connection shutdown begins.

    The Python SDK stops its dispatcher queue before it stops the receive loop.
    A provider can therefore deliver a final frame after the queue has closed.
    Those frames cannot be consumed and should not turn routine connection
    teardown into a receive-loop error.
    """

    def __init__(self, *, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._accepting = True

    async def publish(self, task: Any) -> None:
        if not self._accepting:
            return
        try:
            await super().publish(task)
        except RuntimeError:
            if not self._accepting:
                return
            raise

    async def close(self) -> None:
        self._accepting = False
        await super().close()
