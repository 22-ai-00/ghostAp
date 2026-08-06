"""Shared test doubles and helpers — importable from any test module."""
from __future__ import annotations

import time


class FakeSessionBase:
    def __init__(self, agent_type: str = "coco", cwd: str = ".", **kwargs):
        self.session_id = ""
        self.last_active = time.time()
        self.message_count = 0
        self.last_query = ""
        self.is_resumed = False
        self.created_at = time.time()
        self._agent_type = agent_type
        self._cwd = cwd

    def describe_agent(self) -> str:
        return f"cmd=fake args=acp serve cwd={self._cwd}"

    def start(self, startup_timeout: float = 60) -> str:
        self.session_id = "s_fake"
        return self.session_id

    def load_session(self, session_id: str, timeout: float = 60):
        del timeout
        self.session_id = session_id

    def load_local_history(self, *a, **kw):
        return []

    def to_snapshot(self):
        return {"session_id": self.session_id}

    def close(self):
        return None

    def get_session_info(self) -> str:
        return f"FakeSessionBase({self._agent_type})"

    def is_server_running(self) -> bool:
        return True

    def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
        return True
