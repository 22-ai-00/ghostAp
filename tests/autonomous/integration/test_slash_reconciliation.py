"""Integration tests for exact employee Slash Command reconciliation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.autonomous.provisioning.slash_commands import (
    CanonicalSlashCommand,
    ObservedSlashCommand,
    SlashCommand,
    SlashCommandReconciler,
    SlashReconciliationError,
)


class _InMemorySlashAPI:
    def __init__(
        self,
        commands: tuple[ObservedSlashCommand, ...] = (),
        *,
        apply_mutations: bool = True,
    ) -> None:
        self.commands = list(commands)
        self.apply_mutations = apply_mutations
        self.calls: list[tuple[str, str]] = []
        self.raise_after: set[str] = set()
        self._next_id = 1

    async def list_commands(self) -> tuple[ObservedSlashCommand, ...]:
        self.calls.append(("GET", ""))
        return tuple(self.commands)

    async def create_command(self, command: CanonicalSlashCommand) -> str:
        self.calls.append(("POST", command.command))
        command_id = f"cmd_{self._next_id}"
        self._next_id += 1
        if self.apply_mutations:
            self.commands.append(
                ObservedSlashCommand(
                    command_id=command_id,
                    command=command.command,
                    description=command.description,
                )
            )
        if "POST" in self.raise_after:
            raise RuntimeError("tenant-access-token=super-secret")
        return command_id

    async def update_command(
        self,
        command_id: str,
        command: CanonicalSlashCommand,
    ) -> None:
        self.calls.append(("PATCH", command.command))
        if self.apply_mutations:
            self.commands = [
                replace(existing, command=command.command, description=command.description)
                if existing.command_id == command_id
                else existing
                for existing in self.commands
            ]
        if "PATCH" in self.raise_after:
            raise RuntimeError("tenant-access-token=super-secret")

    async def delete_command(self, command_id: str) -> None:
        self.calls.append(("DELETE", command_id))
        if self.apply_mutations:
            self.commands = [existing for existing in self.commands if existing.command_id != command_id]
        if "DELETE" in self.raise_after:
            raise RuntimeError("tenant-access-token=super-secret")


def _desired(
    *,
    description: str = "Assign a task",
    usage_hint: str = "/task <description>",
) -> tuple[SlashCommand, ...]:
    return (SlashCommand("/task", description, usage_hint),)


def _observed(
    *,
    command_id: str = "cmd_task",
    description: str = "Assign a task\nUsage: /task <description>",
) -> ObservedSlashCommand:
    return ObservedSlashCommand(
        command_id=command_id,
        command="task",
        description=description,
    )


def test_canonical_command_removes_the_leading_slash_and_embeds_usage_in_description() -> None:
    canonical = SlashCommand("/task", "Assign a task", "/task <description>").canonical()

    assert canonical == CanonicalSlashCommand(
        command="task",
        description="Assign a task\nUsage: /task <description>",
    )


@pytest.mark.asyncio
async def test_reconcile_creates_missing_updates_drift_and_deletes_extra() -> None:
    api = _InMemorySlashAPI(
        (
            _observed(description="Outdated description\nUsage: /task old"),
            ObservedSlashCommand(
                command_id="cmd_old",
                command="old",
                description="Retired",
            ),
        )
    )
    desired = (
        *_desired(),
        SlashCommand("/status", "View current status", "/status"),
    )

    verified = await SlashCommandReconciler(api, desired=desired).reconcile()

    assert ("PATCH", "task") in api.calls
    assert ("POST", "status") in api.calls
    assert ("DELETE", "cmd_old") in api.calls
    assert verified.created == ("status",)
    assert verified.updated == ("task",)
    assert verified.deleted == ("old",)
    assert verified.spec_hash == verified.observed_hash
    assert api.calls[0] == ("GET", "")
    assert api.calls[-1] == ("GET", "")


@pytest.mark.asyncio
async def test_description_or_usage_drift_is_patched() -> None:
    api = _InMemorySlashAPI((_observed(),))

    first = await SlashCommandReconciler(
        api,
        desired=_desired(description="Assign work", usage_hint="/task <work>"),
    ).reconcile()

    assert first.updated == ("task",)
    assert api.commands[0].description == "Assign work\nUsage: /task <work>"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_second_reconcile_has_zero_mutations_and_stable_hashes() -> None:
    api = _InMemorySlashAPI()
    desired = (
        SlashCommand("/status", "View current status", "/status"),
        *_desired(),
    )
    reconciler = SlashCommandReconciler(api, desired=desired)

    first = await reconciler.reconcile()
    api.calls.clear()
    second = await reconciler.reconcile()

    assert second.spec_hash == first.spec_hash
    assert second.observed_hash == first.observed_hash
    assert second.created == ()
    assert second.updated == ()
    assert second.deleted == ()
    assert api.calls == [("GET", ""), ("GET", "")]


@pytest.mark.asyncio
async def test_cleanup_deletes_every_command_and_verifies_empty_server_state() -> None:
    api = _InMemorySlashAPI(
        (
            _observed(),
            ObservedSlashCommand("cmd_status", "status", "View status"),
        )
    )

    verified = await SlashCommandReconciler(api).cleanup()

    assert verified.observed == ()
    assert verified.spec_hash == verified.observed_hash
    assert verified.deleted == ("status", "task")
    assert api.calls[0] == ("GET", "")
    assert api.calls[-1] == ("GET", "")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_final_get_mismatch_fails_without_leaking_transport_error() -> None:
    api = _InMemorySlashAPI(apply_mutations=False)

    with pytest.raises(SlashReconciliationError) as exc_info:
        await SlashCommandReconciler(api, desired=_desired()).reconcile()

    assert "exact verification failed" in str(exc_info.value)
    assert "secret" not in str(exc_info.value).lower()
    assert api.calls == [("GET", ""), ("POST", "task"), ("GET", ""), ("GET", "")]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["POST", "PATCH", "DELETE"])
async def test_unknown_mutation_result_is_resolved_by_get_before_deciding(method: str) -> None:
    if method == "POST":
        api = _InMemorySlashAPI()
        desired = _desired()
    elif method == "PATCH":
        api = _InMemorySlashAPI((_observed(description="Old"),))
        desired = _desired()
    else:
        api = _InMemorySlashAPI((_observed(),))
        desired = ()
    api.raise_after.add(method)

    verified = await SlashCommandReconciler(api, desired=desired).reconcile()

    mutation_index = next(index for index, call in enumerate(api.calls) if call[0] == method)
    assert api.calls[mutation_index + 1] == ("GET", "")
    assert verified.spec_hash == verified.observed_hash


@pytest.mark.asyncio
async def test_unresolved_mutation_error_is_redacted() -> None:
    api = _InMemorySlashAPI(apply_mutations=False)
    api.raise_after.add("POST")

    with pytest.raises(SlashReconciliationError) as exc_info:
        await SlashCommandReconciler(api, desired=_desired()).reconcile()

    message = str(exc_info.value)
    assert "POST task outcome could not be verified" in message
    assert "super-secret" not in message
