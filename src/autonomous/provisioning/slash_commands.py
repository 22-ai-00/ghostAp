"""Slash Command Manager: register/diff/cleanup employee bot commands."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlashCommand:
    """One registered slash command definition."""

    name: str
    description: str
    usage_hint: str = ""

    def canonical(self) -> CanonicalSlashCommand:
        """Return the exact server-visible representation of this command."""
        command = _canonical_command_name(self.name)
        description = _non_empty_text(self.description, "description")
        usage_hint = self.usage_hint.strip()
        if usage_hint:
            description = f"{description}\nUsage: {usage_hint}"
        return CanonicalSlashCommand(command=command, description=description)


@dataclass(frozen=True, order=True)
class CanonicalSlashCommand:
    """Stable Slash Command form used for diffing and request bodies."""

    command: str
    description: str
    description_i18n: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", _canonical_command_name(self.command))
        object.__setattr__(
            self,
            "description",
            _non_empty_text(self.description, "description"),
        )
        normalized_i18n = tuple(sorted(self.description_i18n))
        for locale, text in normalized_i18n:
            _non_empty_text(locale, "description locale")
            _non_empty_text(text, "localized description")
        if len({locale for locale, _ in normalized_i18n}) != len(normalized_i18n):
            raise ValueError("description locales must be unique")
        object.__setattr__(self, "description_i18n", normalized_i18n)


@dataclass(frozen=True)
class ObservedSlashCommand:
    """One strictly decoded command returned by the employee app."""

    command_id: str
    command: str
    description: str
    description_i18n: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _non_empty_text(self.command_id, "command_id"))
        raw_command = _non_empty_text(self.command, "command")
        canonical = CanonicalSlashCommand(
            command=raw_command,
            description=self.description,
            description_i18n=self.description_i18n,
        )
        if raw_command != canonical.command:
            raise ValueError("observed command name is not canonical")
        object.__setattr__(self, "command", canonical.command)
        object.__setattr__(self, "description", canonical.description)
        object.__setattr__(self, "description_i18n", canonical.description_i18n)

    @property
    def canonical(self) -> CanonicalSlashCommand:
        """Return the server-visible fields without the unstable remote ID."""
        return CanonicalSlashCommand(
            command=self.command,
            description=self.description,
            description_i18n=self.description_i18n,
        )


def _non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized or any(ord(char) < 32 and char not in "\n\t" for char in normalized):
        raise ValueError(f"{field_name} must be non-empty text")
    return normalized


def _canonical_command_name(value: str) -> str:
    name = _non_empty_text(value, "command").lstrip("/")
    if not name or "/" in name or any(char.isspace() for char in name):
        raise ValueError("command must be a non-empty name without slashes or whitespace")
    return name


EMPLOYEE_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/task", "Assign a task to this employee", "/task <description>"),
    SlashCommand("/status", "View current work status", "/status"),
    SlashCommand("/history", "View recent execution records", "/history [days]"),
    SlashCommand("/memory", "View memory summary", "/memory"),
    SlashCommand("/stop", "Stop current task execution", "/stop"),
)


class SlashCommandAPIError(RuntimeError):
    """Feishu Slash Command API call failed."""


class SlashReconciliationError(RuntimeError):
    """Exact Slash desired state could not be verified."""


class SlashCommandAPI(Protocol):
    """Port for Feishu Slash Command OpenAPI."""

    def list_commands(self, app_id: str) -> list[dict[str, Any]]: ...

    def create_command(self, app_id: str, *, name: str, description: str, usage_hint: str) -> str: ...

    def delete_command(self, app_id: str, command_id: str) -> bool: ...


class AsyncSlashCommandAPI(Protocol):
    """Async employee-owned Slash API used by production reconciliation."""

    async def list_commands(self) -> tuple[ObservedSlashCommand, ...]: ...

    async def create_command(self, command: CanonicalSlashCommand) -> str: ...

    async def update_command(
        self,
        command_id: str,
        command: CanonicalSlashCommand,
    ) -> None: ...

    async def delete_command(self, command_id: str) -> None: ...


@dataclass(frozen=True)
class VerifiedSlashState:
    """Attested result of a final exact server GET."""

    spec_hash: str
    observed_hash: str
    observed: tuple[ObservedSlashCommand, ...]
    created: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()


def _canonical_hash(commands: tuple[CanonicalSlashCommand, ...]) -> str:
    payload = [
        {
            "command": command.command,
            "description": {
                "default_value": command.description,
                "i18n": dict(command.description_i18n),
            },
        }
        for command in sorted(commands)
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _indexed_desired(
    desired: tuple[SlashCommand, ...],
) -> dict[str, CanonicalSlashCommand]:
    indexed: dict[str, CanonicalSlashCommand] = {}
    for command in desired:
        canonical = command.canonical()
        if canonical.command in indexed:
            raise SlashReconciliationError(f"duplicate desired Slash command: {canonical.command}")
        indexed[canonical.command] = canonical
    return indexed


def _indexed_observed(
    observed: tuple[ObservedSlashCommand, ...],
) -> dict[str, ObservedSlashCommand]:
    indexed: dict[str, ObservedSlashCommand] = {}
    for command in observed:
        if command.command in indexed:
            raise SlashReconciliationError(f"duplicate observed Slash command: {command.command}")
        indexed[command.command] = command
    return indexed


class SlashCommandReconciler:
    """Converge an employee app to an exact Slash Command desired set."""

    def __init__(
        self,
        api: AsyncSlashCommandAPI,
        *,
        desired: tuple[SlashCommand, ...] = EMPLOYEE_COMMANDS,
    ) -> None:
        self._api = api
        self._desired = desired

    async def reconcile(self) -> VerifiedSlashState:
        """Run GET, diff, mutations, GET and exact hash verification."""
        return await self._reconcile(self._desired)

    async def cleanup(self) -> VerifiedSlashState:
        """Remove every command and prove the final server set is empty."""

        return await self._reconcile(())

    async def observe_empty(self) -> bool:
        """Query only; never mutate while reconciling an unknown cleanup."""

        return not bool(await self._safe_list())

    async def _reconcile(
        self,
        desired_commands: tuple[SlashCommand, ...],
    ) -> VerifiedSlashState:
        desired = _indexed_desired(desired_commands)
        spec_hash = _canonical_hash(tuple(desired.values()))
        observed = _indexed_observed(await self._safe_list())
        created: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []

        for name in sorted(desired):
            target = desired[name]
            current = observed.get(name)
            if current is None:
                await self._mutate_and_resolve("POST", target)
                created.append(name)
            elif current.canonical != target:
                await self._mutate_and_resolve("PATCH", target, current.command_id)
                updated.append(name)

        for name in sorted(set(observed) - set(desired)):
            current = observed[name]
            await self._mutate_and_resolve("DELETE", current.canonical, current.command_id)
            deleted.append(name)

        for _verify_attempt in range(2):
            await asyncio.sleep(0.5)
            final_observed = await self._safe_list()
            final_index = _indexed_observed(final_observed)
            observed_hash = _canonical_hash(tuple(command.canonical for command in final_index.values()))
            if set(final_index) == set(desired) and all(
                final_index[name].canonical == target for name, target in desired.items()
            ):
                return VerifiedSlashState(
                    spec_hash=spec_hash,
                    observed_hash=observed_hash,
                    observed=tuple(sorted(final_observed, key=lambda command: command.command)),
                    created=tuple(created),
                    updated=tuple(updated),
                    deleted=tuple(deleted),
                )
        raise SlashReconciliationError("Slash exact verification failed")

    async def _safe_list(self) -> tuple[ObservedSlashCommand, ...]:
        try:
            return await self._api.list_commands()
        except Exception as exc:
            raise SlashReconciliationError(f"Slash GET failed ({type(exc).__name__})") from None

    async def _mutate_and_resolve(
        self,
        method: str,
        command: CanonicalSlashCommand,
        command_id: str = "",
    ) -> None:
        try:
            if method == "POST":
                await self._api.create_command(command)
            elif method == "PATCH":
                await self._api.update_command(command_id, command)
            else:
                await self._api.delete_command(command_id)
            return
        except Exception:
            observed = _indexed_observed(await self._safe_list())
            current = observed.get(command.command)
            applied = current is None if method == "DELETE" else (current is not None and current.canonical == command)
            if applied:
                return
            raise SlashReconciliationError(f"Slash {method} {command.command} outcome could not be verified") from None


@dataclass
class CommandSyncResult:
    """Outcome of slash command synchronization."""

    created: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
