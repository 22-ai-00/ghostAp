"""Public safety contracts for managed project-chat provisioning."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

import pytest

import src.project_chat.service as service_module
from src.project_chat.lark_chat_client import ManagedChatValidation
from src.project_chat.service import ProjectChatService
from src.trust.models import ManagedGroupOrigin
from src.trust.registry import RegistryCommitUncertainError


@dataclass
class _Context:
    project_id: str
    project_name: str
    root_path: str
    bound_chat_id: str = ""
    allowed_chat_ids: list[str] = field(default_factory=list)

    def add_chat_id(self, chat_id: str) -> None:
        if chat_id not in self.allowed_chat_ids:
            self.allowed_chat_ids.append(chat_id)


def _service(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        service_module,
        "get_settings",
        lambda: SimpleNamespace(project_chat_suffix="team"),
    )
    pm = MagicMock()
    remote = MagicMock()
    registry = MagicMock()
    replies: list[tuple[str, str, str | None]] = []
    sends: list[tuple[str, str, str, str | None]] = []
    service = ProjectChatService(
        pm,
        remote,
        lambda *args: replies.append(args),
        lambda *args: sends.append(args),
        managed_group_registry=registry,
        owner_id="ou_owner",
        receiving_bot_ref="bot_main",
    )
    monkeypatch.setattr(
        service,
        "_reply_jump_card",
        lambda message_id, ctx: replies.append(
            (message_id, f"jump:{ctx.project_id}", "interactive")
        ),
    )
    pm.find_project_by_path.return_value = None
    pm.find_project_by_name.return_value = None
    pm.managed_group_residual.return_value = None
    pm.record_managed_group_residual.return_value = True
    pm.complete_managed_chat_binding_saga.return_value = True
    registry.provision_chat_id.return_value = None
    registry.prepare_create_dispatch.return_value = True
    remote.create_chat.return_value = SimpleNamespace(chat_id="oc_created")
    return service, pm, remote, registry, replies, sends


@pytest.mark.parametrize(
    "data",
    (
        {"name": "../escape"},
        {"name": "project", "suffix": "unsafe suffix"},
    ),
)
def test_handle_rejects_unsafe_group_names_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path, data: dict[str, str]
) -> None:
    service, pm, remote, registry, replies, _ = _service(monkeypatch)

    service.handle("om_1", "oc_source", "ou_user", {"path": str(tmp_path), **data})

    assert "无效" in replies[-1][1]
    pm.find_project_by_path.assert_not_called()
    registry.begin_provision.assert_not_called()
    remote.create_chat.assert_not_called()


def test_recovered_chat_requires_owner_and_receiving_bot_membership(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    service, pm, remote, registry, replies, _ = _service(monkeypatch)
    registry.provision_chat_id.return_value = "oc_recovered"
    remote.validate_managed_chat.return_value = ManagedChatValidation.INVALID

    service.handle(
        "om_1",
        "oc_source",
        "ou_user",
        {"path": str(tmp_path), "name": "project"},
    )

    remote.validate_managed_chat.assert_called_once_with("oc_recovered", "ou_owner")
    pm.record_managed_group_residual.assert_called_once_with(
        ANY, "oc_recovered", "recovered_chat_invalid"
    )
    pm.create_project_with_managed_chat_saga.assert_not_called()
    assert "Owner/接收 Bot" in replies[-1][1]


def test_existing_binding_extends_cross_chat_scope_only_from_trusted_facts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    service, pm, remote, registry, replies, _ = _service(monkeypatch)
    ctx = _Context(
        project_id="p1",
        project_name="project",
        root_path=str(tmp_path),
        bound_chat_id="oc_project",
        allowed_chat_ids=["oc_project"],
    )
    pm.find_project_by_path.return_value = ctx
    pm.pending_managed_chat_binding_sagas_for_project.return_value = []
    pm._save_projects.return_value = True
    grant = SimpleNamespace(
        project_id="p1",
        canonical_root_ref=str(tmp_path),
        managed_group_id="oc_project",
        owner_id="ou_owner",
        backend_binding_ids=(),
        connected_target_refs=(),
    )
    active = SimpleNamespace(
        project_id="p1",
        canonical_root_ref=str(tmp_path),
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        owner_id="ou_owner",
        receiving_bot_ref="bot_main",
    )
    registry.trust_snapshot.return_value = (active, grant)

    service.handle(
        "om_valid",
        "oc_source",
        "ou_user",
        {"path": str(tmp_path), "name": "project"},
    )

    assert "oc_source" in ctx.allowed_chat_ids
    assert replies[-1][1] == "jump:p1"

    ctx.allowed_chat_ids.remove("oc_source")
    registry.trust_snapshot.return_value = (
        SimpleNamespace(**{**vars(active), "receiving_bot_ref": "other_bot"}),
        grant,
    )
    service.handle(
        "om_untrusted",
        "oc_other",
        "ou_user",
        {"path": str(tmp_path), "name": "project"},
    )

    assert "oc_other" not in ctx.allowed_chat_ids
    assert "信任事实不完整" in replies[-1][1]


def test_successful_provision_uses_stable_dispatch_and_completes_saga(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    service, pm, remote, registry, replies, sends = _service(monkeypatch)
    ctx = _Context("p1", "project", str(tmp_path), "oc_created", ["oc_created"])
    pm.create_project_with_managed_chat_saga.return_value = (True, "created", ctx)

    service.handle(
        "om_1",
        "oc_source",
        "ou_user",
        {"path": str(tmp_path), "name": "project"},
    )

    create_kwargs = remote.create_chat.call_args.kwargs
    assert create_kwargs["name"] == "project-team"
    assert create_kwargs["user_id_list"] == ["ou_user"]
    assert create_kwargs["operation_id"].startswith("new-chat:ou_owner:")
    registry.activate.assert_called_once()
    pm.complete_managed_chat_binding_saga.assert_called_once()
    assert replies[-1][1] == "jump:p1"
    assert sends[-1][0] == "oc_created"


def test_activation_failure_restores_local_saga_and_retains_remote_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    service, pm, remote, registry, replies, _ = _service(monkeypatch)
    ctx = _Context("p1", "project", str(tmp_path), "oc_created", ["oc_created"])
    pm.create_project_with_managed_chat_saga.return_value = (True, "created", ctx)
    pm.restore_managed_chat_binding_saga.return_value = True
    registry.activate.side_effect = RuntimeError("registry unavailable")

    service.handle(
        "om_1",
        "oc_source",
        "ou_user",
        {"path": str(tmp_path), "name": "project"},
    )

    operation_id = registry.activate.call_args.kwargs["provision_id"]
    pm.restore_managed_chat_binding_saga.assert_called_once_with(operation_id)
    pm.record_managed_group_residual.assert_called_once_with(
        operation_id, "oc_created", "untrusted_retained"
    )
    pm.complete_managed_chat_binding_saga.assert_not_called()
    remote.delete_chat.assert_not_called()
    assert "未获得信任" in replies[-1][1]


def test_definitely_uncommitted_activation_aborts_after_compensation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    service, pm, remote, registry, replies, sends = _service(monkeypatch)
    ctx = _Context("p1", "project", str(tmp_path), "oc_created", ["oc_created"])
    pm.create_project_with_managed_chat_saga.return_value = (True, "created", ctx)
    pm.restore_managed_chat_binding_saga.return_value = True
    registry.activate.side_effect = RegistryCommitUncertainError(
        "registry replace did not commit",
        committed=False,
    )

    service.handle(
        "om_1",
        "oc_source",
        "ou_user",
        {"path": str(tmp_path), "name": "project"},
    )

    operation_id = registry.activate.call_args.kwargs["provision_id"]
    pm.restore_managed_chat_binding_saga.assert_called_once_with(operation_id)
    pm.record_managed_group_residual.assert_called_once_with(
        operation_id, "oc_created", "untrusted_retained"
    )
    pm.complete_managed_chat_binding_saga.assert_not_called()
    assert all(not str(reply[1]).startswith("jump:") for reply in replies)
    assert sends == []
    assert "未获得信任" in replies[-1][1]
