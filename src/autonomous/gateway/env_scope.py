"""Employee backend environment construction and provider state projection."""

from __future__ import annotations

import json
import os
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from ..workspace.layout import open_child_directory, open_directory_tree


@dataclass(frozen=True, slots=True)
class EmployeeEnvironmentAuthority:
    """Secret-free identity passed to an employee credential provider."""

    tenant_key: str
    agent_id: str
    employee_version: int
    credential_ref: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.tenant_key, self.agent_id, self.credential_ref)
        ):
            raise ValueError("employee environment authority is incomplete")
        if type(self.employee_version) is not int or self.employee_version < 0:
            raise ValueError("employee environment version is invalid")


@dataclass(frozen=True, slots=True)
class EmployeeProcessEnvironmentMaterial:
    """Frozen employee-scoped runtime and provider credential material."""

    tenant_key: str
    agent_id: str
    employee_version: int
    credential_ref: str
    runtime_env: Mapping[str, str] = field(repr=False, compare=False)
    credential_env: Mapping[str, str] = field(repr=False, compare=False)
    provider_files: Mapping[str, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        EmployeeEnvironmentAuthority(
            self.tenant_key,
            self.agent_id,
            self.employee_version,
            self.credential_ref,
        )
        for name in ("runtime_env", "credential_env", "provider_files"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or any(
                not isinstance(key, str)
                or not key
                or not isinstance(item, str)
                or (name != "runtime_env" and not item)
                for key, item in value.items()
            ):
                raise ValueError(f"{name} must be a string mapping")
            object.__setattr__(self, name, MappingProxyType(dict(value)))

    @property
    def authority(self) -> EmployeeEnvironmentAuthority:
        return EmployeeEnvironmentAuthority(
            self.tenant_key,
            self.agent_id,
            self.employee_version,
            self.credential_ref,
        )


def build_employee_process_env(
    runtime_env: Mapping[str, str],
    *,
    employee_home: str,
    credential_env: Mapping[str, str] | None = None,
    codex_home: str = "",
) -> dict[str, str]:
    """Build one process env from the full host environment plus overrides."""

    if not isinstance(employee_home, str) or not employee_home.startswith("/"):
        raise ValueError("employee_home must be an absolute path")
    result = {
        key: value
        for key, value in runtime_env.items()
        if isinstance(key, str) and key and isinstance(value, str)
    }
    result["HOME"] = employee_home
    if codex_home:
        if not isinstance(codex_home, str) or not codex_home.startswith("/"):
            raise ValueError("codex_home must be an absolute path")
        result["CODEX_HOME"] = codex_home
    for key, value in dict(credential_env or {}).items():
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise ValueError("employee credential env value is invalid")
        if key == "TRAE_HOME" and not value.startswith("/"):
            raise ValueError("employee TRAE_HOME must be an absolute path")
        result[key] = value
    return dict(sorted(result.items()))


def build_employee_backend_env(
    material: EmployeeProcessEnvironmentMaterial,
    *,
    agent_type: str,
    employee_home: str,
    workspace_path: str,
) -> dict[str, str]:
    """Build one backend env and project only its backend-specific state."""

    credential_env = dict(material.credential_env)
    normalized_agent_type = str(agent_type or "").strip().casefold()
    if normalized_agent_type in {"trae", "traex"}:
        auth_file = material.provider_files.get("traex_auth_json", "")
        if not auth_file:
            raise ValueError("TraeX auth source is unavailable")
        credential_env["TRAE_HOME"] = prepare_employee_traex_home(
            employee_home=employee_home,
            auth_file=auth_file,
            constraints_file=str(Path(workspace_path) / "AGENTS.md"),
        )
    else:
        credential_env.pop("TRAE_HOME", None)
    return build_employee_process_env(
        material.runtime_env,
        employee_home=employee_home,
        credential_env=credential_env,
        codex_home=(
            str(Path(employee_home) / "runtime" / "codex-home")
            if normalized_agent_type == "codex"
            else ""
        ),
    )


def runtime_only_employee_environment(
    authority: EmployeeEnvironmentAuthority,
) -> EmployeeProcessEnvironmentMaterial:
    """Provide the current host process environment to an employee backend."""

    runtime_env = dict(os.environ)
    return EmployeeProcessEnvironmentMaterial(
        authority.tenant_key,
        authority.agent_id,
        authority.employee_version,
        authority.credential_ref,
        runtime_env,
        {},
    )


def local_employee_environment(
    authority: EmployeeEnvironmentAuthority,
    *,
    traex_auth_home: str,
) -> EmployeeProcessEnvironmentMaterial:
    """Describe the local TraeX auth source without sharing its state home."""

    material = runtime_only_employee_environment(authority)
    if not isinstance(traex_auth_home, str) or not traex_auth_home or "\x00" in traex_auth_home:
        raise ValueError("employee TraeX auth home is invalid")
    auth_file = str(Path(traex_auth_home).expanduser().absolute() / "cli" / "auth.json")
    return EmployeeProcessEnvironmentMaterial(
        authority.tenant_key,
        authority.agent_id,
        authority.employee_version,
        authority.credential_ref,
        material.runtime_env,
        {},
        {"traex_auth_json": auth_file},
    )


def prepare_employee_traex_home(
    *,
    employee_home: str,
    auth_file: str,
    constraints_file: str,
) -> str:
    """Project auth and constraints into one employee-private TraeX home."""

    if not isinstance(employee_home, str) or not employee_home.startswith("/"):
        raise ValueError("employee_home must be an absolute path")
    auth = _read_private_file(auth_file, maximum_bytes=65_536, label="TraeX auth")
    constraints = _read_private_file(
        constraints_file,
        maximum_bytes=8_192,
        label="employee constraints",
    )
    try:
        parsed = json.loads(
            auth,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        token = parsed.get("trae", {}).get("access_token")
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError("TraeX auth metadata is invalid") from None
    if not isinstance(token, str) or not token:
        raise ValueError("TraeX auth metadata is incomplete")

    employee_fd = open_directory_tree(Path(employee_home))
    try:
        runtime_fd = open_child_directory(employee_fd, "runtime")
        try:
            traex_fd = open_child_directory(runtime_fd, "trae-home")
            try:
                cli_fd = open_child_directory(traex_fd, "cli")
                try:
                    _atomic_write_leaf(cli_fd, "auth.json", auth)
                finally:
                    os.close(cli_fd)
                _atomic_write_leaf(traex_fd, "AGENTS.md", constraints)
            finally:
                os.close(traex_fd)
        finally:
            os.close(runtime_fd)
    finally:
        os.close(employee_fd)
    return str(Path(employee_home) / "runtime" / "trae-home")


def _read_private_file(path: str, *, maximum_bytes: int, label: str) -> bytes:
    if not isinstance(path, str) or not path.startswith("/") or "\x00" in path:
        raise ValueError(f"{label} path is invalid")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
            or metadata.st_size <= 0
            or metadata.st_size > maximum_bytes
        ):
            raise ValueError(f"{label} file is unsafe")
        content = b""
        while len(content) <= maximum_bytes:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - len(content)))
            if not chunk:
                break
            content += chunk
        if not content or len(content) > maximum_bytes:
            raise ValueError(f"{label} file is unsafe")
        return content
    finally:
        os.close(descriptor)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _atomic_write_leaf(parent_fd: int, filename: str, content: bytes) -> None:
    temporary = f".{filename}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short employee provider projection write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    os.close(descriptor)
    try:
        os.replace(temporary, filename, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "EmployeeEnvironmentAuthority",
    "EmployeeProcessEnvironmentMaterial",
    "build_employee_backend_env",
    "build_employee_process_env",
    "local_employee_environment",
    "prepare_employee_traex_home",
    "runtime_only_employee_environment",
]
