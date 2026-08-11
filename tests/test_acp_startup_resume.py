"""Provider session resume contracts at the shared startup boundary."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.acp.startup_utils import (
    SessionStartupCoordinator,
    SessionStartupRequest,
)


class _ResumableSession:
    def __init__(self, **_kwargs) -> None:
        self.session_id = ""
        self.loaded: list[tuple[str, float]] = []
        self.closed = False

    def start(self, startup_timeout: float = 60) -> str:
        del startup_timeout
        self.session_id = "new-transport-session"
        return self.session_id

    def load_session(self, session_id: str, timeout: float) -> None:
        self.loaded.append((session_id, timeout))
        self.session_id = session_id

    def describe_agent(self) -> str:
        return "resumable-test-session"

    def close(self) -> None:
        self.closed = True


def test_default_startup_path_loads_requested_provider_session() -> None:
    coordinator = SessionStartupCoordinator(
        manager_agent_type="codex",
        session_telemetry=MagicMock(),
        sync_acp_session_cls=_ResumableSession,
        get_settings_fn=lambda: SimpleNamespace(),
    )
    deadline = time.monotonic() + 5

    result = coordinator.start(
        SessionStartupRequest(
            key="chat\x1fproject\x1fthread",
            cwd="/tmp",
            startup_timeout=5,
            project_id="project-1",
            session_id="provider-session-1",
            effective_agent_type="codex",
            model_name=None,
            retries=1,
            deadline_monotonic=deadline,
        )
    )

    assert result.actual_id == "provider-session-1"
    assert result.session.session_id == "provider-session-1"
    assert len(result.session.loaded) == 1
    loaded_session_id, load_timeout = result.session.loaded[0]
    assert loaded_session_id == "provider-session-1"
    assert 0 < load_timeout <= 5
