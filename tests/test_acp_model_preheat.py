from __future__ import annotations

import threading

from src.acp import helper


def test_model_preheat_starts_tools_in_parallel(monkeypatch, tmp_path) -> None:
    both_started = threading.Event()
    release = threading.Event()
    started: set[str] = set()
    lock = threading.Lock()

    def fake_fetch(tool: str, _cwd: str):
        with lock:
            started.add(tool)
            if len(started) == 2:
                both_started.set()
        release.wait(timeout=2)
        return []

    monkeypatch.setattr(helper, "fetch_acp_models", fake_fetch)
    worker = helper.kickoff_acp_model_preheat(["codex", "traex"], str(tmp_path))
    assert worker is not None
    try:
        assert both_started.wait(timeout=1), "model preheat ran sequentially"
    finally:
        release.set()
        worker.join(timeout=2)

    assert started == {"codex", "traex"}
