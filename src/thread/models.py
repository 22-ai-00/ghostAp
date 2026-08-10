from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThreadContext:
    thread_root_id: str
    chat_id: str
    project_id: str
    mode: str = "smart"
    tool_name: Optional[str] = None
    model_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    @property
    def session_key_suffix(self) -> str:
        return f"t:{self.thread_root_id}"

    def touch(self) -> None:
        self.last_active = time.time()
