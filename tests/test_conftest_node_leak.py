"""Regression tests for baseline-aware Node process leak detection in tests/conftest.py."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import psutil

from tests import conftest


class _FakeProc:
    def __init__(self, info=None, *, exc: BaseException | None = None) -> None:
        self._info = info
        self._exc = exc

    @property
    def info(self):
        if self._exc is not None:
            raise self._exc
        return self._info


def test_snapshot_psutil_matches_node_variants_and_handles_transient_errors(monkeypatch):
    fake_procs = [
        _FakeProc({"pid": 101, "name": "node", "create_time": 1700000000.12349}),
        _FakeProc({"pid": 102, "name": "nodejs", "create_time": 1700000001.5}),
        _FakeProc({"pid": 103, "name": "/opt/homebrew/bin/node", "create_time": 1700000002.9994}),
        _FakeProc({"pid": 104, "name": "/usr/local/bin/NODE", "create_time": 1700000003.0}),
        _FakeProc(exc=psutil.NoSuchProcess(105)),
        _FakeProc(exc=psutil.AccessDenied(106)),
        _FakeProc(exc=psutil.ZombieProcess(107)),
        _FakeProc({"pid": 108, "name": "nodemon", "create_time": 1700000004.0}),
        _FakeProc({"pid": 109, "name": "/usr/bin/node_exporter", "create_time": 1700000005.0}),
        _FakeProc({"pid": 110, "name": None, "create_time": 1700000006.0}),
        _FakeProc({"pid": 111, "name": "python3", "create_time": 1700000007.0}),
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: iter(fake_procs))

    snap = conftest._snapshot_node_processes()

    assert snap == {
        (101, 1700000000.123): "node",
        (102, 1700000001.5): "nodejs",
        (103, 1700000002.999): "/opt/homebrew/bin/node",
        (104, 1700000003.0): "/usr/local/bin/NODE",
    }


def test_snapshot_ps_fallback_uses_c_locale_and_parses_linux_and_macos(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setenv("LC_ALL", "zh_CN.UTF-8")
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")

    ps_stdout = "\n".join(
        [
            "    1 Fri Mar 13 07:05:50 2026 systemd",
            "  201 Thu Nov 19 21:00:00 2026 node",
            "  202 Thu Nov  9 08:05:01 2026 nodejs",
            "  203 Thu Nov 19 21:00:02 2026 /opt/homebrew/bin/node",
            "  204 Thu Nov 19 21:00:03 2026 /Users/dev/My Apps/node",
            "  205 Thu Nov 19 21:00:04 2026 nodemon",
            "  206 Thu Nov 19 21:00:05 2026 /usr/bin/node_exporter",
            "bad-line-without-pid",
        ]
    )

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return SimpleNamespace(returncode=0, stdout=ps_stdout, stderr="")

    monkeypatch.setattr(conftest.subprocess, "run", fake_run)
    # Exercise ImportError fallback path (not just _force_ps=True)
    monkeypatch.setitem(sys.modules, "psutil", None)

    snap = conftest._snapshot_node_processes()

    assert captured["cmd"] == ["ps", "-A", "-o", "pid=", "-o", "lstart=", "-o", "comm="]
    assert captured["env"]["LC_ALL"] == "C"
    assert captured["env"]["LANG"] == "C"
    assert snap == {
        (201, "Thu Nov 19 21:00:00 2026"): "node",
        (202, "Thu Nov 9 08:05:01 2026"): "nodejs",
        (203, "Thu Nov 19 21:00:02 2026"): "/opt/homebrew/bin/node",
        (204, "Thu Nov 19 21:00:03 2026"): "/Users/dev/My Apps/node",
    }


def test_snapshot_force_ps_handles_subprocess_error_and_real_ps(monkeypatch):
    # Real ps invocation works on host without raising
    real_snap = conftest._snapshot_node_processes(_force_ps=True)
    assert isinstance(real_snap, dict)

    def raise_os_error(*_args, **_kwargs):
        raise OSError("ps not found")

    monkeypatch.setattr(conftest.subprocess, "run", raise_os_error)
    assert conftest._snapshot_node_processes(_force_ps=True) == {}

    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="ps", timeout=5)

    monkeypatch.setattr(conftest.subprocess, "run", raise_timeout)
    assert conftest._snapshot_node_processes(_force_ps=True) == {}


def _run_leak_fixture(monkeypatch, snapshots):
    it = iter(snapshots)
    monkeypatch.setattr(conftest, "_snapshot_node_processes", lambda: next(it))
    monkeypatch.setattr(conftest.time, "sleep", lambda _s: None)
    gen = conftest._detect_resource_leaks.__wrapped__()
    next(gen)
    try:
        next(gen)
    except StopIteration:
        pass


def test_detect_resource_leaks_ignores_baseline_and_exited_node_processes(monkeypatch, capsys):
    baseline = {
        (100, 1700000000.123): "/opt/homebrew/bin/node",
        (101, 1700000001.0): "node",
    }
    # 100 still alive, 101 exited cleanly during tests
    end = {
        (100, 1700000000.123): "/opt/homebrew/bin/node",
    }
    _run_leak_fixture(monkeypatch, [baseline, end])
    err = capsys.readouterr().err
    assert "leaked node process" not in err
    assert "pid=100" not in err


def test_detect_resource_leaks_warns_on_leaked_and_pid_reused_node(monkeypatch, capsys):
    # Test both psutil float timestamps and ps lstart strings across PID reuse
    baseline_psutil = {(4242, 1700000000.1): "node"}
    end_psutil = {
        (4242, 1700000999.2): "node",  # same PID, different create_time (PID reuse)
        (5151, 1700001000.5): "/opt/homebrew/bin/node",  # newly leaked process
    }
    _run_leak_fixture(monkeypatch, [baseline_psutil, end_psutil])
    err = capsys.readouterr().err
    assert "2 leaked node process(es) still running" in err
    assert "pid=4242 (node)" in err
    assert "pid=5151 (/opt/homebrew/bin/node)" in err

    baseline_ps = {(700, "Thu Nov 19 20:00:00 2026"): "node"}
    end_ps = {(700, "Thu Nov 19 21:00:00 2026"): "node"}
    _run_leak_fixture(monkeypatch, [baseline_ps, end_ps])
    err_ps = capsys.readouterr().err
    assert "1 leaked node process(es) still running" in err_ps
    assert "pid=700 (node)" in err_ps
