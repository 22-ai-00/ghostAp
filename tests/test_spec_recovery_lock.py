from __future__ import annotations

import importlib.util
import multiprocessing
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).parents[1] / "src" / "spec_engine" / "recovery_lock.py"
)


def _load_recovery_lock_module():
    assert _MODULE_PATH.is_file(), "Spec recovery needs a cross-process lease"
    spec = importlib.util.spec_from_file_location(
        "ghostap_spec_recovery_lock_test",
        _MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _probe_recovery_lease(state_path: str, results) -> None:
    module = _load_recovery_lock_module()
    lease = module.try_acquire_recovery_lease(state_path)
    results.put(lease is not None)
    if lease is not None:
        lease.release()


def _probe_in_spawned_process(state_path: Path) -> bool:
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(
        target=_probe_recovery_lease,
        args=(str(state_path), results),
    )
    process.start()
    process.join(timeout=10)
    assert not process.is_alive()
    assert process.exitcode == 0
    acquired = results.get(timeout=1)
    results.close()
    results.join_thread()
    return acquired


def test_spec_recovery_lease_is_exclusive_across_processes(tmp_path: Path) -> None:
    module = _load_recovery_lock_module()
    state_path = tmp_path / ".spec_engine_state.json"

    lease = module.try_acquire_recovery_lease(state_path)
    assert lease is not None
    try:
        assert _probe_in_spawned_process(state_path) is False
    finally:
        lease.release()

    assert _probe_in_spawned_process(state_path) is True
