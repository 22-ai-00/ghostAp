"""Runtime integrity checks for the pinned employee Channel SDK."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

LOCKED_LARK_CHANNEL_VERSION = "1.2.0"
LOCKED_LARK_CHANNEL_WHEEL_SHA256 = "c08690572a099377cdeddc3a2a1402d9645879ad137e780d80060053dc8c1570"
LOCKED_LARK_CHANNEL_INSTALLED_RECORD_SHA256 = "d539f31b6457104d5a345c0e5188deef857a6fca3a386ee7544478e735f4b4eb"
LOCKED_LARK_CHANNEL_RUNTIME_PAYLOAD_SHA256 = "1f9017d0511043c3a7dffbcc0df110619d22782cc9e957f1227a4075b1963abe"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY = Path(__file__).resolve().parents[3]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"invalid {name} fields")


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid {name}")


@dataclass(frozen=True, slots=True)
class SDKDistributionIdentity:
    distribution_name: str
    version: str
    lock_wheel_sha256: str
    observed_wheel_archive_sha256: str | None
    installed_record_sha256: str
    runtime_payload_sha256: str
    record_verified: bool
    project_lock_sha256: str
    installed_identity_algorithm: str = "record-sha256-triples-v1"
    runtime_identity_algorithm: str = "package-sha256-triples-v1"
    path_basis: str = "site-packages-relative-posix"

    def __post_init__(self) -> None:
        if self.distribution_name != "lark-channel-sdk":
            raise ValueError("unexpected Channel SDK distribution")
        if self.version != LOCKED_LARK_CHANNEL_VERSION:
            raise ValueError("unexpected Channel SDK version")
        _validate_sha256(self.lock_wheel_sha256, "lock wheel hash")
        if self.observed_wheel_archive_sha256 is not None:
            _validate_sha256(
                self.observed_wheel_archive_sha256,
                "observed wheel archive hash",
            )
        _validate_sha256(self.installed_record_sha256, "installed RECORD hash")
        _validate_sha256(self.runtime_payload_sha256, "runtime payload hash")
        _validate_sha256(self.project_lock_sha256, "project lock hash")
        if self.record_verified is not True:
            raise ValueError("SDK RECORD must be verified")
        if self.lock_wheel_sha256 != LOCKED_LARK_CHANNEL_WHEEL_SHA256:
            raise ValueError("Channel SDK wheel hash is not trusted")
        if self.installed_record_sha256 != LOCKED_LARK_CHANNEL_INSTALLED_RECORD_SHA256:
            raise ValueError("Channel SDK installed RECORD identity is not trusted")
        if self.runtime_payload_sha256 != LOCKED_LARK_CHANNEL_RUNTIME_PAYLOAD_SHA256:
            raise ValueError("Channel SDK runtime payload identity is not trusted")
        if self.installed_identity_algorithm != "record-sha256-triples-v1":
            raise ValueError("unsupported installed identity algorithm")
        if self.runtime_identity_algorithm != "package-sha256-triples-v1":
            raise ValueError("unsupported runtime identity algorithm")
        if self.path_basis != "site-packages-relative-posix":
            raise ValueError("unsupported SDK identity path basis")

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution_name": self.distribution_name,
            "installed_identity_algorithm": self.installed_identity_algorithm,
            "installed_record_sha256": self.installed_record_sha256,
            "lock_wheel_sha256": self.lock_wheel_sha256,
            "observed_wheel_archive_sha256": self.observed_wheel_archive_sha256,
            "path_basis": self.path_basis,
            "project_lock_sha256": self.project_lock_sha256,
            "record_verified": self.record_verified,
            "runtime_identity_algorithm": self.runtime_identity_algorithm,
            "runtime_payload_sha256": self.runtime_payload_sha256,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SDKDistributionIdentity:
        expected = {
            "distribution_name",
            "version",
            "lock_wheel_sha256",
            "observed_wheel_archive_sha256",
            "installed_record_sha256",
            "runtime_payload_sha256",
            "record_verified",
            "installed_identity_algorithm",
            "runtime_identity_algorithm",
            "path_basis",
            "project_lock_sha256",
        }
        _exact_fields(value, expected, "SDK identity")
        return cls(**dict(value))




def _verify_project_lock() -> str:
    pyproject_bytes = (_REPOSITORY / "pyproject.toml").read_bytes()
    lock_bytes = (_REPOSITORY / "uv.lock").read_bytes()
    pyproject = tomllib.loads(pyproject_bytes.decode("utf-8"))
    dependencies = pyproject.get("project", {}).get("dependencies", [])
    pins = [
        value
        for value in dependencies
        if isinstance(value, str)
        and re.sub(r"[-_.]+", "-", value.split("==", 1)[0].lower())
        == "lark-channel-sdk"
    ]
    if pins != [f"lark-channel-sdk=={LOCKED_LARK_CHANNEL_VERSION}"]:
        raise ValueError("pyproject must strictly pin the trusted Channel SDK")

    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    packages = [
        package
        for package in lock.get("package", [])
        if package.get("name") == "lark-channel-sdk"
    ]
    if len(packages) != 1 or packages[0].get("version") != LOCKED_LARK_CHANNEL_VERSION:
        raise ValueError("uv.lock Channel SDK package is not uniquely pinned")
    wheel_hashes = {
        wheel.get("hash")
        for wheel in packages[0].get("wheels", [])
        if isinstance(wheel, dict)
    }
    if wheel_hashes != {f"sha256:{LOCKED_LARK_CHANNEL_WHEEL_SHA256}"}:
        raise ValueError("uv.lock Channel SDK wheel hash is not trusted")
    return _sha256(
        _canonical_json(
            {
                "pyproject_sha256": _sha256(pyproject_bytes),
                "uv_lock_sha256": _sha256(lock_bytes),
            }
        )
    )


def prepare_controlled_sdk_import_cache(cache_root: Path) -> Path:
    """Force subsequent SDK imports to ignore source-adjacent bytecode caches."""
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    if any(root.iterdir()):
        raise ValueError("controlled SDK bytecode cache must be empty")
    resolved = root.resolve(strict=True)
    sys.pycache_prefix = str(resolved)
    sys.dont_write_bytecode = True
    return resolved


def collect_sdk_distribution_identity(
    *,
    require_controlled_import_cache: bool = False,
) -> SDKDistributionIdentity:
    if require_controlled_import_cache:
        prefix = sys.pycache_prefix
        if (
            not sys.dont_write_bytecode
            or not isinstance(prefix, str)
            or not prefix
            or any(
                name == "lark_channel" or name.startswith("lark_channel.")
                for name in sys.modules
            )
        ):
            raise ValueError("controlled SDK import cache is not active before import")
        cache_root = Path(prefix)
        if not cache_root.is_dir() or any(cache_root.iterdir()):
            raise ValueError("controlled SDK import cache is not empty")
    project_lock_sha256 = _verify_project_lock()
    distribution = importlib.metadata.distribution("lark-channel-sdk")
    name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"].lower())
    if name != "lark-channel-sdk":
        raise ValueError("unexpected Channel SDK distribution name")
    if distribution.version != LOCKED_LARK_CHANNEL_VERSION:
        raise ValueError("unexpected Channel SDK version")
    files = distribution.files
    if not files:
        raise ValueError("Channel SDK RECORD is unavailable")

    root = Path(distribution.locate_file("")).resolve()
    seen: set[str] = set()
    verified: list[list[Any]] = []
    record_package_files: set[str] = set()
    for package_path in files:
        relative = PurePosixPath(str(package_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe Channel SDK RECORD path")
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise ValueError("duplicate Channel SDK RECORD path")
        seen.add(relative_text)
        located = Path(distribution.locate_file(package_path))
        cursor = root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError("Channel SDK RECORD path contains symlink")
        try:
            located.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("Channel SDK RECORD path escapes distribution") from exc
        if package_path.hash is None:
            if not relative_text.endswith(".dist-info/RECORD"):
                raise ValueError("Channel SDK RECORD entry lacks hash")
            continue
        if package_path.hash.mode != "sha256":
            raise ValueError("Channel SDK RECORD entry uses non-sha256 hash")
        if not located.is_file() or located.is_symlink():
            raise ValueError("Channel SDK RECORD file is missing or unsafe")
        content = located.read_bytes()
        if package_path.size != len(content):
            raise ValueError("Channel SDK RECORD size mismatch")
        padding = "=" * (-len(package_path.hash.value) % 4)
        expected_digest = base64.urlsafe_b64decode(package_path.hash.value + padding)
        actual_digest = hashlib.sha256(content).digest()
        if actual_digest != expected_digest:
            raise ValueError("Channel SDK RECORD hash mismatch")
        digest_hex = actual_digest.hex()
        verified.append([relative_text, len(content), digest_hex])
        if relative.parts and relative.parts[0] == "lark_channel":
            record_package_files.add(relative_text)

    package_root = root / "lark_channel"
    actual_package_files: set[str] = set()
    runtime_payload: list[list[Any]] = []
    for path in sorted(package_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Channel SDK package contains symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        actual_package_files.add(relative)
        content = path.read_bytes()
        runtime_payload.append([relative, len(content), _sha256(content)])
    if actual_package_files != record_package_files:
        raise ValueError("Channel SDK package files differ from RECORD")

    import lark_channel.ws.client as sdk_client

    client_path = Path(sdk_client.__file__ or "").resolve()
    try:
        client_relative = client_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Channel SDK import is shadowed") from exc
    if client_relative != "lark_channel/ws/client.py":
        raise ValueError("Channel SDK import path is unexpected")

    return SDKDistributionIdentity(
        distribution_name=name,
        version=distribution.version,
        lock_wheel_sha256=LOCKED_LARK_CHANNEL_WHEEL_SHA256,
        observed_wheel_archive_sha256=None,
        installed_record_sha256=_sha256(_canonical_json(sorted(verified, key=lambda value: value[0]))),
        runtime_payload_sha256=_sha256(_canonical_json(sorted(runtime_payload, key=lambda value: value[0]))),
        record_verified=True,
        project_lock_sha256=project_lock_sha256,
    )
