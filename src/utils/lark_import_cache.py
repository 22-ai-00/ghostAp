"""Persistent sourceless import cache for the generated ``lark-oapi`` SDK.

The official SDK contains thousands of generated Python modules.  Loading the
source-adjacent files is disproportionately expensive on filesystems where
small-file metadata operations dominate.  This module packs the existing,
interpreter-specific bytecode into one archive owned by the active virtual
environment.  The archive is invalidated by both the wheel RECORD identity
and the Python bytecode ABI.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import py_compile
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImportCacheResult:
    """Path and creation status for one package import cache."""

    path: Path
    created: bool


def _bytecode_is_current(source: Path, bytecode: Path) -> bool:
    """Return whether a source-adjacent pyc matches this interpreter/source."""

    try:
        with bytecode.open("rb") as bytecode_file:
            header = bytecode_file.read(16)
        stat = source.stat()
    except OSError:
        return False
    if len(header) != 16 or header[:4] != importlib.util.MAGIC_NUMBER:
        return False

    flags = int.from_bytes(header[4:8], "little")
    if flags & 1:
        try:
            expected_hash = importlib.util.source_hash(source.read_bytes())
        except OSError:
            return False
        return header[8:16] == expected_hash

    recorded_mtime = int.from_bytes(header[8:12], "little")
    recorded_size = int.from_bytes(header[12:16], "little")
    return recorded_mtime == int(stat.st_mtime) & 0xFFFFFFFF and recorded_size == stat.st_size


def _usable_archive(path: Path, package_name: str) -> bool:
    try:
        if not path.is_file():
            return False
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        return f"{package_name}/__init__.pyc" in names and all(name.endswith(".pyc") for name in names)
    except (OSError, zipfile.BadZipFile):
        return False


def ensure_package_import_cache(
    *,
    package_root: Path,
    cache_dir: Path,
    distribution_identity: str,
) -> ImportCacheResult:
    """Build or reuse an atomic, Python-ABI-specific bytecode archive."""

    root = Path(package_root).resolve(strict=True)
    if not root.is_dir() or not root.name.isidentifier():
        raise ValueError("package_root must identify a Python package directory")
    if not isinstance(distribution_identity, str) or not distribution_identity:
        raise ValueError("distribution_identity must be non-empty")

    cache_tag = sys.implementation.cache_tag
    if not cache_tag:
        raise RuntimeError("Python bytecode cache tag is unavailable")
    bytecode_magic = importlib.util.MAGIC_NUMBER.hex()
    identity = hashlib.sha256(distribution_identity.encode("utf-8")).hexdigest()[:20]
    destination_dir = Path(cache_dir)
    destination_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = destination_dir / (
        f"{root.name}-{cache_tag}-{bytecode_magic}-{identity}.pyc.zip"
    )
    if _usable_archive(destination, root.name):
        return ImportCacheResult(path=destination, created=False)

    sources = sorted(root.rglob("*.py"))
    if not sources or root / "__init__.py" not in sources:
        raise ValueError("package_root does not contain an importable package")

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for source in sources:
                if source.is_symlink():
                    raise ValueError("package source contains a symlink")
                resolved_source = source.resolve(strict=True)
                if not resolved_source.is_relative_to(root):
                    raise ValueError("package source escapes package_root")

                bytecode = Path(importlib.util.cache_from_source(str(source)))
                if not _bytecode_is_current(source, bytecode):
                    compiled = py_compile.compile(str(source), doraise=True)
                    bytecode = Path(compiled)
                relative = source.relative_to(root).with_suffix(".pyc")
                archive.write(
                    bytecode,
                    f"{root.name}/{relative.as_posix()}",
                )
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    if not _usable_archive(destination, root.name):
        raise RuntimeError("generated package import cache is invalid")
    for stale in destination_dir.glob(f"{root.name}-*.pyc.zip"):
        if stale != destination and stale.is_file() and not stale.is_symlink():
            stale.unlink()
    return ImportCacheResult(path=destination, created=True)


def ensure_lark_oapi_import_cache() -> ImportCacheResult:
    """Build or reuse the cache for the uv-managed ``lark-oapi`` wheel."""

    distribution = importlib.metadata.distribution("lark-oapi")
    record = distribution.read_text("RECORD")
    if not record:
        raise RuntimeError("lark-oapi wheel RECORD is unavailable")
    package_root = Path(distribution.locate_file("lark_oapi"))
    record_sha256 = hashlib.sha256(record.encode("utf-8")).hexdigest()
    identity = f"lark-oapi\0{distribution.version}\0{record_sha256}"
    return ensure_package_import_cache(
        package_root=package_root,
        cache_dir=Path(sys.prefix) / ".ghostap-import-cache",
        distribution_identity=identity,
    )


def activate_lark_oapi_import_cache() -> ImportCacheResult:
    """Prepend the current environment's cache to this process import path."""

    result = ensure_lark_oapi_import_cache()
    cache_path = str(result.path)
    if cache_path not in sys.path:
        sys.path.insert(0, cache_path)
    return result


__all__ = [
    "ImportCacheResult",
    "activate_lark_oapi_import_cache",
    "ensure_lark_oapi_import_cache",
    "ensure_package_import_cache",
]
