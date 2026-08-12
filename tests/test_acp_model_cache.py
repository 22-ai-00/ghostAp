from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from src.acp import helper
from src.acp.options import ACPModelOption


@pytest.fixture(autouse=True)
def _isolated_model_cache():
    with helper._cache_lock:
        helper._probe_cache.clear()
        helper._negative_cache.clear()
        helper._inflight.clear()
        helper._cache_generation.clear()
    yield
    with helper._cache_lock:
        helper._probe_cache.clear()
        helper._negative_cache.clear()
        helper._inflight.clear()
        helper._cache_generation.clear()


def test_positive_model_catalog_is_reused_for_thirty_minutes(
    monkeypatch,
    tmp_path,
) -> None:
    now = [1_000.0]
    probe = MagicMock(return_value=[ACPModelOption(name="model-a")])
    monkeypatch.setattr(helper.time, "time", lambda: now[0])
    monkeypatch.setattr(helper, "_probe_blocking", probe)
    monkeypatch.setattr(
        "src.agent_session.backend_resolver.is_cli_backend",
        lambda _tool: False,
    )

    assert helper.fetch_acp_models("traex", str(tmp_path))[0].name == "model-a"
    now[0] += 1_799.9
    assert helper.fetch_acp_models("traex", str(tmp_path))[0].name == "model-a"
    now[0] += 0.2
    assert helper.fetch_acp_models("traex", str(tmp_path))[0].name == "model-a"

    assert probe.call_count == 2


def test_failed_model_catalog_is_not_reprobed_for_five_minutes(
    monkeypatch,
    tmp_path,
) -> None:
    now = [2_000.0]
    probe = MagicMock(return_value=[])
    monkeypatch.setattr(helper.time, "time", lambda: now[0])
    monkeypatch.setattr(helper, "_probe_blocking", probe)
    monkeypatch.setattr(
        "src.agent_session.backend_resolver.is_cli_backend",
        lambda _tool: False,
    )

    assert helper.fetch_acp_models("traex", str(tmp_path)) == []
    now[0] += 299.9
    assert helper.fetch_acp_models("traex", str(tmp_path)) == []
    now[0] += 0.2
    assert helper.fetch_acp_models("traex", str(tmp_path)) == []

    assert probe.call_count == 2


def test_exact_invalidation_reprobes_only_the_requested_catalog(
    monkeypatch,
    tmp_path,
) -> None:
    probe = MagicMock(
        side_effect=lambda tool, *_args: [ACPModelOption(name=f"{tool}-model")]
    )
    monkeypatch.setattr(helper, "_probe_blocking", probe)
    monkeypatch.setattr(
        "src.agent_session.backend_resolver.is_cli_backend",
        lambda _tool: False,
    )

    first_cwd = str(tmp_path / "first")
    second_cwd = str(tmp_path / "second")
    helper.fetch_acp_models("traex", first_cwd)
    helper.fetch_acp_models("traex", second_cwd)
    helper.fetch_acp_models("codex", first_cwd)
    helper.invalidate_acp_model_cache("traex", first_cwd)
    helper.fetch_acp_models("traex", first_cwd)
    helper.fetch_acp_models("traex", second_cwd)
    helper.fetch_acp_models("codex", first_cwd)

    assert [(call.args[0], call.args[1]) for call in probe.call_args_list] == [
        ("traex", first_cwd),
        ("traex", second_cwd),
        ("codex", first_cwd),
        ("traex", first_cwd),
    ]


def test_coco_catalog_uses_the_same_long_cache_policy() -> None:
    from src.coco_model import manager

    assert manager.CACHE_TTL_SECONDS == 1_800
    assert manager.FALLBACK_CACHE_TTL_SECONDS == 300


def test_coco_real_catalog_and_fallback_obey_their_cache_windows(
    monkeypatch,
) -> None:
    from src.coco_model import manager as coco_manager
    from src.coco_model.models import CocoModel

    now = [1_000.0]
    monkeypatch.setattr(coco_manager.time, "time", lambda: now[0])
    manager = coco_manager.CocoModelManager()
    manager._initialized = True
    real_probe = MagicMock(return_value=[CocoModel(name="real-model")])
    monkeypatch.setattr(manager, "_load_models", real_probe)

    assert manager.get_models().models[0].name == "real-model"
    now[0] += 1_799.9
    assert manager.get_models().cached is True
    now[0] += 0.2
    assert manager.get_models().cached is False
    assert real_probe.call_count == 2

    manager.invalidate_cache()
    fallback_probe = MagicMock(return_value=list(coco_manager.DEFAULT_MODELS))
    monkeypatch.setattr(manager, "_load_models", fallback_probe)
    now[0] += 1.0
    assert manager.get_models().cached is False
    now[0] += 299.9
    assert manager.get_models().cached is True
    now[0] += 0.2
    assert manager.get_models().cached is False
    assert fallback_probe.call_count == 2


def test_coco_invalidation_reaches_its_dedicated_manager(
    monkeypatch,
    tmp_path,
) -> None:
    manager = MagicMock()
    monkeypatch.setattr(
        "src.coco_model.get_coco_model_manager",
        lambda: manager,
    )

    helper.invalidate_acp_model_cache("coco", str(tmp_path))

    manager.invalidate_cache.assert_called_once_with()


def test_invalidation_fences_an_older_inflight_probe_result(
    monkeypatch,
    tmp_path,
) -> None:
    old_started = threading.Event()
    release_old = threading.Event()
    probe_calls = 0
    probe_lock = threading.Lock()

    def probe(_tool, *_args):
        nonlocal probe_calls
        with probe_lock:
            probe_calls += 1
            call_number = probe_calls
        if call_number == 1:
            old_started.set()
            assert release_old.wait(timeout=2)
            return [ACPModelOption(name="old-model")]
        return [ACPModelOption(name="new-model")]

    monkeypatch.setattr(helper, "_probe_blocking", probe)
    monkeypatch.setattr(
        "src.agent_session.backend_resolver.is_cli_backend",
        lambda _tool: False,
    )
    old_result: list[list[ACPModelOption]] = []
    worker = threading.Thread(
        target=lambda: old_result.append(
            helper.fetch_acp_models("traex", str(tmp_path))
        )
    )
    worker.start()
    assert old_started.wait(timeout=1)

    helper.invalidate_acp_model_cache("traex", str(tmp_path))
    fresh = helper.fetch_acp_models("traex", str(tmp_path))
    release_old.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert fresh[0].name == "new-model"
    assert old_result[0][0].name == "old-model"
    assert helper.fetch_acp_models("traex", str(tmp_path))[0].name == "new-model"
    assert probe_calls == 2


def test_cache_fill_between_miss_and_flight_registration_does_not_reprobe(
    monkeypatch,
    tmp_path,
) -> None:
    delayed_miss = threading.Event()
    release_delayed = threading.Event()
    original_cached = helper._cached
    delayed_once = False

    def cached(key, tool):
        nonlocal delayed_once
        result = original_cached(key, tool)
        if threading.current_thread().name == "delayed-model-fetch" and not delayed_once:
            delayed_once = True
            delayed_miss.set()
            assert release_delayed.wait(timeout=2)
        return result

    probe = MagicMock(return_value=[ACPModelOption(name="model-a")])
    monkeypatch.setattr(helper, "_cached", cached)
    monkeypatch.setattr(helper, "_probe_blocking", probe)
    monkeypatch.setattr(
        "src.agent_session.backend_resolver.is_cli_backend",
        lambda _tool: False,
    )
    delayed_result: list[list[ACPModelOption]] = []
    delayed_worker = threading.Thread(
        name="delayed-model-fetch",
        target=lambda: delayed_result.append(
            helper.fetch_acp_models("traex", str(tmp_path))
        ),
    )
    delayed_worker.start()
    assert delayed_miss.wait(timeout=1)

    leading_result = helper.fetch_acp_models("traex", str(tmp_path))
    release_delayed.set()
    delayed_worker.join(timeout=2)

    assert not delayed_worker.is_alive()
    assert leading_result[0].name == "model-a"
    assert delayed_result[0][0].name == "model-a"
    assert probe.call_count == 1
