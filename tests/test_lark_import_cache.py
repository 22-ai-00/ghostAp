from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType


def _fixture_package(tmp_path: Path) -> Path:
    package = tmp_path / "fixture_lark"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from .value import VALUE\n",
        encoding="utf-8",
    )
    (package / "value.py").write_text("VALUE = 'cached'\n", encoding="utf-8")
    return package


def test_import_cache_builds_sourceless_archive_that_python_can_import(
    tmp_path: Path,
) -> None:
    from src.utils.lark_import_cache import ensure_package_import_cache

    package = _fixture_package(tmp_path)
    result = ensure_package_import_cache(
        package_root=package,
        cache_dir=tmp_path / "cache",
        distribution_identity="fixture-v1",
    )

    assert result.created is True
    assert importlib.util.MAGIC_NUMBER.hex() in result.path.name
    with zipfile.ZipFile(result.path) as archive:
        names = set(archive.namelist())
    assert names == {
        "fixture_lark/__init__.pyc",
        "fixture_lark/value.pyc",
    }

    runner = tmp_path / "runner"
    runner.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(result.path)
    imported = subprocess.run(
        [sys.executable, "-c", "from fixture_lark import VALUE; print(VALUE)"],
        cwd=runner,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == "cached"


def test_import_cache_reuses_matching_distribution_identity(tmp_path: Path) -> None:
    from src.utils.lark_import_cache import ensure_package_import_cache

    package = _fixture_package(tmp_path)
    first = ensure_package_import_cache(
        package_root=package,
        cache_dir=tmp_path / "cache",
        distribution_identity="fixture-v1",
    )
    first_mtime = first.path.stat().st_mtime_ns

    second = ensure_package_import_cache(
        package_root=package,
        cache_dir=tmp_path / "cache",
        distribution_identity="fixture-v1",
    )

    assert second.path == first.path
    assert second.created is False
    assert second.path.stat().st_mtime_ns == first_mtime


def test_import_cache_replaces_stale_distribution_identity(tmp_path: Path) -> None:
    from src.utils.lark_import_cache import ensure_package_import_cache

    package = _fixture_package(tmp_path)
    first = ensure_package_import_cache(
        package_root=package,
        cache_dir=tmp_path / "cache",
        distribution_identity="fixture-v1",
    )

    second = ensure_package_import_cache(
        package_root=package,
        cache_dir=tmp_path / "cache",
        distribution_identity="fixture-v2",
    )

    assert second.created is True
    assert second.path != first.path
    assert second.path.is_file()
    assert not first.path.exists()


def test_feishu_runtime_activates_import_cache_before_sdk_import(monkeypatch) -> None:
    import src.main as main_module

    events: list[str] = []
    formatter_type = object()
    emoji_type = object()
    client_type = object()

    feishu_package = ModuleType("feishu")
    feishu_package.__path__ = []  # type: ignore[attr-defined]
    formatter_module = ModuleType("feishu.message_formatter")
    formatter_module.FeishuMessageFormatter = formatter_type
    ws_module = ModuleType("feishu.ws_client")
    ws_module.EmojiReaction = emoji_type
    ws_module.FeishuWSClient = client_type

    monkeypatch.setitem(sys.modules, "feishu", feishu_package)
    monkeypatch.setitem(sys.modules, "feishu.message_formatter", formatter_module)
    monkeypatch.setitem(sys.modules, "feishu.ws_client", ws_module)
    monkeypatch.setattr(
        main_module,
        "_activate_lark_import_cache",
        lambda: events.append("cache-active"),
    )

    loaded = main_module._load_feishu_runtime()

    assert events == ["cache-active"]
    assert loaded == (formatter_type, emoji_type, client_type)
