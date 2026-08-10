"""Runtime identity contracts for the pinned employee Channel SDK."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.autonomous.ingress.sdk_capability import (
    LOCKED_LARK_CHANNEL_VERSION,
    LOCKED_LARK_CHANNEL_WHEEL_SHA256,
    SDKDistributionIdentity,
    collect_sdk_distribution_identity,
    prepare_controlled_sdk_import_cache,
)


def test_installed_sdk_record_and_runtime_payload_are_verified() -> None:
    identity = collect_sdk_distribution_identity()

    assert identity.distribution_name == "lark-channel-sdk"
    assert identity.version == LOCKED_LARK_CHANNEL_VERSION
    assert identity.lock_wheel_sha256 == LOCKED_LARK_CHANNEL_WHEEL_SHA256
    assert identity.observed_wheel_archive_sha256 is None
    assert identity.record_verified is True
    assert identity.installed_identity_algorithm == "record-sha256-triples-v1"
    assert identity.runtime_identity_algorithm == "package-sha256-triples-v1"
    assert identity.path_basis == "site-packages-relative-posix"
    assert len(identity.project_lock_sha256) == 64
    assert identity.installed_record_sha256 == (
        "832add283a7ba9800978e6e94b37bab223aa266ddc0d6163ff93d564fc06ee27"
    )
    assert identity.runtime_payload_sha256 == (
        "845e6c04019aefd54ec56c37a435d45a1f5e1cff8a81b7eb5382049ad3e05c88"
    )


def test_controlled_import_cache_is_empty_and_not_source_adjacent(
    tmp_path: Path,
) -> None:
    import importlib.util
    import os
    import py_compile
    import sys

    module_name = "employee_sdk_bytecode_probe"
    source = tmp_path / f"{module_name}.py"
    source.write_text("VALUE = 'evil'\n", encoding="utf-8")
    fixed_time = 1_700_000_000
    os.utime(source, (fixed_time, fixed_time))
    source_adjacent_pyc = Path(py_compile.compile(str(source), doraise=True))
    source.write_text("VALUE = 'safe'\n", encoding="utf-8")
    os.utime(source, (fixed_time, fixed_time))
    assert source_adjacent_pyc.is_file()

    previous_prefix = sys.pycache_prefix
    previous_write = sys.dont_write_bytecode
    sys.path.insert(0, str(tmp_path))
    try:
        controlled = prepare_controlled_sdk_import_cache(tmp_path / "controlled-cache")
        cached = Path(importlib.util.cache_from_source(__file__)).resolve()
        assert cached.is_relative_to(controlled)
        assert not list(controlled.rglob("*.pyc"))
        assert sys.dont_write_bytecode is True
        module = __import__(module_name)
        assert module.VALUE == "safe"
    finally:
        sys.modules.pop(module_name, None)
        sys.path.remove(str(tmp_path))
        sys.pycache_prefix = previous_prefix
        sys.dont_write_bytecode = previous_write


def test_controlled_collector_rejects_unprepared_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setattr(sys, "pycache_prefix", None)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)

    with pytest.raises(ValueError, match="controlled SDK import cache"):
        collect_sdk_distribution_identity(require_controlled_import_cache=True)


def test_same_version_repacked_sdk_identity_is_rejected() -> None:
    value = collect_sdk_distribution_identity().to_dict()
    value["runtime_payload_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="not trusted"):
        SDKDistributionIdentity.from_dict(value)
