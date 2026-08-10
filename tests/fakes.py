"""Shared test doubles — importable from any test module.

Extends ``FakeSessionBase`` (from ``helpers.py``) with the full session
interface used by engine tests, so individual test files no longer need
to re-define ``_S`` / ``_DummySession`` / ``FakeSession`` classes.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock


class FakeSession:
    """Universal fake session that satisfies Deep / Spec engine contracts.

    All methods are no-ops or return safe defaults.  Override ``send_prompt``
    in a subclass if you need custom behaviour (error injection, etc.).
    """

    def __init__(self, agent_type: str = "coco", cwd: str = ".") -> None:
        self.session_id: str = "sid"
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        self.message_count: int = 0
        self.last_query: str = ""
        self._agent_type = agent_type
        self._cwd = cwd

    # -- lifecycle --
    def describe_agent(self) -> str:
        return f"cmd=fake args=acp serve cwd={self._cwd}"

    def start(self, startup_timeout: float = 60) -> str:
        self.session_id = "s_fake"
        return self.session_id


    def load_local_history(self, *a, **kw):
        return []

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass

    def to_snapshot(self) -> dict:
        return {"session_id": self.session_id}

    def get_session_info(self) -> str:
        return f"FakeSession({self._agent_type})"

    def is_server_running(self) -> bool:
        return True

    def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool:
        return True

    # -- prompting --
    def send_prompt(self, *args, **kwargs):
        return MagicMock(stop_reason="end_turn")

    def send_prompt_with_retry(self, *args, **kwargs):
        return MagicMock(stop_reason="end_turn")


class TimeoutSession(FakeSession):
    """Session whose prompt methods always raise ``TimeoutError``."""

    def send_prompt(self, *args, **kwargs):
        raise TimeoutError("fake timeout")

    def send_prompt_with_retry(self, *args, **kwargs):
        raise TimeoutError("fake timeout")


class ErrorSession(FakeSession):
    """Session whose prompt methods always raise ``RuntimeError``."""

    def send_prompt(self, *args, **kwargs):
        raise RuntimeError("Internal error")


class AgentSessionSettings:
    """Minimal settings stub for ``patch("src.agent_session.get_settings", ...)``.

    Attributes match the subset accessed during session creation / tests.
    """

    acp_startup_timeout: int = 20
    rate_limit_retry_enabled: bool = False
