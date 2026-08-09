"""Tests for Slash Command Manager and Channel Connection Manager."""

from __future__ import annotations

from src.autonomous.provisioning.channel import (
    ChannelConnectionManager,
    ChannelState,
)
from src.autonomous.provisioning.slash_commands import (
    SlashCommandManager,
)


class _FakeSlashAPI:
    def __init__(self, existing: list[dict] | None = None) -> None:
        self.existing = existing or []
        self.created: list[dict] = []
        self.deleted: list[str] = []

    def list_commands(self, app_id):
        return self.existing

    def create_command(self, app_id, *, name, description, usage_hint):
        self.created.append({"name": name})
        return f"cmd_{name}"

    def delete_command(self, app_id, command_id):
        self.deleted.append(command_id)
        return True


class TestSlashCommandManager:
    def test_sync_creates_missing_commands(self) -> None:
        api = _FakeSlashAPI()
        mgr = SlashCommandManager(api)
        result = mgr.sync_commands("app_1")
        assert len(result.created) == 5
        assert result.errors == []

    def test_sync_skips_existing(self) -> None:
        api = _FakeSlashAPI(existing=[
            {"name": "/task", "id": "cmd_1"},
            {"name": "/status", "id": "cmd_2"},
        ])
        mgr = SlashCommandManager(api)
        result = mgr.sync_commands("app_1")
        assert "/task" in result.unchanged
        assert "/status" in result.unchanged
        assert len(result.created) == 3

    def test_sync_deletes_extra(self) -> None:
        api = _FakeSlashAPI(existing=[
            {"name": "/old_cmd", "id": "cmd_old"},
            {"name": "/task", "id": "cmd_task"},
        ])
        mgr = SlashCommandManager(api)
        result = mgr.sync_commands("app_1")
        assert "/old_cmd" in result.deleted

    def test_cleanup_all_removes_everything(self) -> None:
        api = _FakeSlashAPI(existing=[
            {"name": "/task", "id": "cmd_1"},
            {"name": "/status", "id": "cmd_2"},
        ])
        mgr = SlashCommandManager(api)
        result = mgr.cleanup_all("app_1")
        assert len(result.deleted) == 2


class _FakeChannelSDK:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.connections: list[str] = []
        self.disconnections: list[str] = []

    def connect(self, *, app_id, app_secret, on_message, on_disconnect):
        if self._fail:
            raise RuntimeError("connection failed")
        conn = f"conn_{app_id}"
        self.connections.append(conn)
        return conn

    def disconnect(self, connection):
        self.disconnections.append(connection)


class TestChannelConnectionManager:
    def test_start_connects_employee(self) -> None:
        sdk = _FakeChannelSDK()
        mgr = ChannelConnectionManager(
            channel_sdk=sdk,
            secret_resolver=lambda aid, ref: "secret_value",
        )
        status = mgr.start(
            agent_id="agt_alpha",
            app_id="app_1",
            credential_ref="cred_1",
            on_message=lambda aid, msg: None,
        )
        assert status.state == ChannelState.CONNECTED
        assert status.agent_id == "agt_alpha"
        assert len(sdk.connections) == 1

    def test_start_failure_stays_disconnected(self) -> None:
        sdk = _FakeChannelSDK(fail=True)
        mgr = ChannelConnectionManager(
            channel_sdk=sdk,
            secret_resolver=lambda aid, ref: "secret",
        )
        status = mgr.start(
            agent_id="agt_alpha",
            app_id="app_1",
            credential_ref="cred_1",
            on_message=lambda aid, msg: None,
        )
        assert status.state == ChannelState.DISCONNECTED
        assert "connection failed" in status.error

    def test_stop_disconnects(self) -> None:
        sdk = _FakeChannelSDK()
        mgr = ChannelConnectionManager(
            channel_sdk=sdk,
            secret_resolver=lambda aid, ref: "secret",
        )
        mgr.start(
            agent_id="agt_alpha",
            app_id="app_1",
            credential_ref="cred_1",
            on_message=lambda aid, msg: None,
        )
        status = mgr.stop("agt_alpha")
        assert status.state == ChannelState.STOPPED
        assert len(sdk.disconnections) == 1

    def test_stop_all(self) -> None:
        sdk = _FakeChannelSDK()
        mgr = ChannelConnectionManager(
            channel_sdk=sdk,
            secret_resolver=lambda aid, ref: "secret",
        )
        mgr.start(agent_id="agt_a", app_id="a", credential_ref="c", on_message=lambda a, m: None)
        mgr.start(agent_id="agt_b", app_id="b", credential_ref="c", on_message=lambda a, m: None)
        count = mgr.stop_all()
        assert count == 2

    def test_list_connected(self) -> None:
        sdk = _FakeChannelSDK()
        mgr = ChannelConnectionManager(
            channel_sdk=sdk,
            secret_resolver=lambda aid, ref: "secret",
        )
        mgr.start(agent_id="agt_a", app_id="a", credential_ref="c", on_message=lambda a, m: None)
        assert len(mgr.list_connected()) == 1

    def test_duplicate_start_returns_existing(self) -> None:
        sdk = _FakeChannelSDK()
        mgr = ChannelConnectionManager(
            channel_sdk=sdk,
            secret_resolver=lambda aid, ref: "secret",
        )
        first = mgr.start(agent_id="agt_a", app_id="a", credential_ref="c", on_message=lambda a, m: None)
        second = mgr.start(agent_id="agt_a", app_id="a", credential_ref="c", on_message=lambda a, m: None)
        assert first is second
        assert len(sdk.connections) == 1
