"""Read-only team memory adapter over existing employee data."""

from __future__ import annotations

import re
from pathlib import Path


class EmployeeGroupMemoryStore:
    """Read existing shared-memory files without owning a group execution mode."""

    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path).expanduser().resolve()

    @staticmethod
    def _safe_id(chat_id: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_:-]+", "_", chat_id or "")
        if not value or value.startswith(".") or ".." in value:
            raise ValueError("invalid team memory scope")
        return value

    def read_group_memory(self, chat_id: str) -> str:
        path = (self._base / "groups" / self._safe_id(chat_id) / "SHARED_MEMORY.md").resolve()
        if path != self._base and self._base not in path.parents:
            raise ValueError("team memory path escaped employee storage")
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""


__all__ = ["EmployeeGroupMemoryStore"]
