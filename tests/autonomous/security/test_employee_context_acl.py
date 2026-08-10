"""Fail-closed ACL and integrity tests for employee L1/L2 Context reads."""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import replace
from pathlib import Path

import pytest

from src.autonomous.authorization import EmployeeAuthorizationScope
from src.autonomous.context import (
    AuthorizedContextRequest,
    AuthorizedGroupMemoryReader,
    ContextUnavailableError,
    ContextUnavailableReason,
)
from src.autonomous.context.group_memory import EmployeeGroupMemoryStore
from src.autonomous.data.facades import (
    EmployeeDocumentMaterializer,
    EmployeeMemoryFacade,
    MemoryAccessError,
    MemoryIntegrityError,
)
from src.autonomous.data.models import DataKind
from src.autonomous.data.projection import (
    DataProjectionState,
    DocumentMetadataRecord,
)
from src.autonomous.domain import (
    BotPrincipal,
    EmployeeDefinition,
    EmployeeState,
    WorkerType,
)
from src.autonomous.journal.projections import ProjectionState
from src.autonomous.workforce.registry import ProjectedAgentRegistry


def _project_l1(
    state: DataProjectionState,
    *,
    tenant_key: str = "tenant_1",
    content_hash: str,
) -> None:
    document_id = f"data_{secrets.token_hex(8)}"
    state.employee_documents[document_id] = DocumentMetadataRecord(
        document_id=document_id,
        tenant_key=tenant_key,
        agent_id="agt_1",
        owner_principal_id="ou_owner",
        kind=DataKind.L1_MEMORY,
        version=1,
        source_id="l1_memory",
        content_hash=content_hash,
        content_type="text/markdown",
        blob_ref={},
    )
    state.latest_employee_document[
        (tenant_key, "agt_1", DataKind.L1_MEMORY.value, "l1_memory")
    ] = document_id


def _facade(
    tmp_path: Path,
    *,
    content: bytes | None,
    tenant_key: str = "tenant_1",
) -> tuple[EmployeeMemoryFacade, Path, EmployeeDocumentMaterializer]:
    materializer = EmployeeDocumentMaterializer(tmp_path / "canonical")
    state = DataProjectionState()
    content_hash = hashlib.sha256(content or b"").hexdigest()
    _project_l1(state, tenant_key=tenant_key, content_hash=content_hash)
    canonical = materializer.resolve_path(
        "agt_1",
        DataKind.L1_MEMORY,
        "l1_memory",
    ).absolute
    if content is not None:
        materializer.materialize(
            "agt_1",
            DataKind.L1_MEMORY,
            "l1_memory",
            content,
            content_hash,
        )
    return (
        EmployeeMemoryFacade(
            materializer=materializer,
            state=state,
        ),
        canonical,
        materializer,
    )


def test_l1_rejects_foreign_tenant_before_file_read(tmp_path: Path) -> None:
    secret = b"foreign tenant secret"
    facade, _, _ = _facade(tmp_path, content=secret, tenant_key="tenant_1")

    with pytest.raises(MemoryAccessError) as raised:
        facade.read_l1("agt_1", "tenant_2")

    assert secret.decode() not in str(raised.value)


def test_l1_rejects_materialized_file_without_projected_owner(
    tmp_path: Path,
) -> None:
    content = b"orphan"
    materializer = EmployeeDocumentMaterializer(tmp_path / "canonical")
    materializer.materialize(
        "agt_1",
        DataKind.L1_MEMORY,
        "l1_memory",
        content,
        hashlib.sha256(content).hexdigest(),
    )
    facade = EmployeeMemoryFacade(
        materializer=materializer,
        state=DataProjectionState(),
    )

    with pytest.raises(MemoryIntegrityError, match="projected owner"):
        facade.read_l1("agt_1", "tenant_1")


def test_l1_rejects_missing_and_hash_mismatched_materialization(
    tmp_path: Path,
) -> None:
    missing, _, _ = _facade(tmp_path / "missing", content=None)
    with pytest.raises(MemoryIntegrityError, match="missing"):
        missing.read_l1("agt_1", "tenant_1")

    altered, path, _ = _facade(tmp_path / "altered", content=b"expected")
    path.write_bytes(b"tampered")
    with pytest.raises(MemoryIntegrityError, match="hash"):
        altered.read_l1("agt_1", "tenant_1")


def test_l1_rejects_symlink_without_reading_target(tmp_path: Path) -> None:
    facade, canonical, _ = _facade(tmp_path, content=b"expected")
    target = tmp_path / "outside-secret"
    target.write_text("must not be read")
    canonical.unlink()
    os.symlink(target, canonical)

    with pytest.raises(MemoryIntegrityError, match="open failed") as raised:
        facade.read_l1("agt_1", "tenant_1")

    assert "must not be read" not in str(raised.value)


@pytest.mark.parametrize("level", ["root", "agent", "memory"])
def test_l1_rejects_symlinked_canonical_parent(
    tmp_path: Path,
    level: str,
) -> None:
    content = b"outside but hash-valid"
    facade, canonical, _ = _facade(tmp_path, content=content)
    root = canonical.parents[2]
    agent = canonical.parents[1]
    memory = canonical.parent
    canonical.unlink()

    if level == "memory":
        memory.rmdir()
        outside = tmp_path / "outside-memory"
        outside.mkdir()
        (outside / "MEMORY.md").write_bytes(content)
        os.symlink(outside, memory)
    elif level == "agent":
        memory.rmdir()
        agent.rmdir()
        outside = tmp_path / "outside-agent"
        (outside / "memory").mkdir(parents=True)
        (outside / "memory" / "MEMORY.md").write_bytes(content)
        os.symlink(outside, agent)
    else:
        memory.rmdir()
        agent.rmdir()
        root.rmdir()
        outside = tmp_path / "outside-root"
        (outside / "agt_1" / "memory").mkdir(parents=True)
        (outside / "agt_1" / "memory" / "MEMORY.md").write_bytes(content)
        os.symlink(outside, root)

    with pytest.raises(MemoryIntegrityError, match="not trusted"):
        facade.read_l1("agt_1", "tenant_1")


def test_materializer_rejects_symlinked_parent_on_write(tmp_path: Path) -> None:
    materializer = EmployeeDocumentMaterializer(tmp_path / "canonical")
    agent = tmp_path / "canonical" / "agt_1"
    outside = tmp_path / "outside-agent"
    outside.mkdir()
    os.symlink(outside, agent)
    content = b"must stay inside root"

    with pytest.raises(MemoryIntegrityError, match="not trusted"):
        materializer.materialize(
            "agt_1",
            DataKind.L1_MEMORY,
            "l1_memory",
            content,
            hashlib.sha256(content).hexdigest(),
        )

    assert not (outside / "memory" / "MEMORY.md").exists()



def _request(**changes) -> AuthorizedContextRequest:
    values = {
        "tenant_key": "tenant_1",
        "agent_id": "agt_1",
        "bot_principal_id": "bot_1",
        "app_id": "cli_1",
        "channel_generation": 1,
        "chat_id": "oc_1",
        "thread_root_message_id": "om_root",
        "feishu_thread_id": "omt_1",
        "current_message_id": "om_current",
        "requester_principal_id": "ou_requester",
        "source_requester_principal_id": "ou_requester",
        "authorization_scope": EmployeeAuthorizationScope.MANAGED_GROUP,
    }
    values.update(changes)
    return AuthorizedContextRequest(**values)


def _workforce_state() -> ProjectionState:
    state = ProjectionState()
    state.employees["agt_1"] = EmployeeDefinition(
        agent_id="agt_1",
        tenant_key="tenant_1",
        owner_principal_id="ou_owner",
        worker_type=WorkerType.VISIBLE,
        state=EmployeeState.ACTIVE,
        bot_principal_id="bot_1",
        member_groups=("oc_1",),
    )
    state.bot_principals["bot_1"] = BotPrincipal(
        bot_principal_id="bot_1",
        tenant_key="tenant_1",
        agent_id="agt_1",
        app_id="cli_1",
        credential_ref="cred_1",
    )
    return state


class _Acl:
    def __init__(self, allowed: bool) -> None:
        self.allowed = allowed

    def is_authorized(self, request: AuthorizedContextRequest) -> bool:
        del request
        return self.allowed


class _Backend:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    def read_group_memory(self, chat_id: str) -> str:
        self.calls.append(chat_id)
        if self.error is not None:
            raise self.error
        return "full L2"


def _group_reader(
    state: ProjectionState,
    *,
    allowed: bool,
    backend: _Backend,
) -> AuthorizedGroupMemoryReader:
    return AuthorizedGroupMemoryReader(
        registry_provider=lambda: ProjectedAgentRegistry(state),
        requester_acl=_Acl(allowed),
        backend=backend,
    )


@pytest.mark.parametrize(
    ("context_request", "state_mutation", "allowed", "reason"),
    [
        (_request(chat_id="oc_2"), None, True, ContextUnavailableReason.SCOPE),
        (_request(), None, False, ContextUnavailableReason.PERMISSION),
        (
            _request(),
            lambda state: state.bot_principals.__setitem__(
                "bot_1",
                replace(state.bot_principals["bot_1"], credential_ref=""),
            ),
            True,
            ContextUnavailableReason.CREDENTIALS,
        ),
        (
            _request(),
            lambda state: state.bot_principals.__setitem__(
                "bot_1",
                replace(state.bot_principals["bot_1"], app_id="cli_other"),
            ),
            True,
            ContextUnavailableReason.SCOPE,
        ),
    ],
)
def test_l2_acl_failures_never_reach_backend(
    context_request: AuthorizedContextRequest,
    state_mutation,
    allowed: bool,
    reason: ContextUnavailableReason,
) -> None:
    state = _workforce_state()
    if state_mutation is not None:
        state_mutation(state)
    backend = _Backend()
    reader = _group_reader(state, allowed=allowed, backend=backend)

    with pytest.raises(ContextUnavailableError) as raised:
        reader.read(context_request)

    assert raised.value.reason is reason
    assert backend.calls == []


def test_l2_reads_only_authorized_chat_and_sanitizes_backend_failure() -> None:
    state = _workforce_state()
    backend = _Backend()
    reader = _group_reader(state, allowed=True, backend=backend)
    assert reader.read(_request()) == "full L2"
    assert backend.calls == ["oc_1"]

    failure = _Backend(RuntimeError("sensitive backend detail"))
    reader = _group_reader(state, allowed=True, backend=failure)
    with pytest.raises(ContextUnavailableError) as raised:
        reader.read(_request())
    assert raised.value.reason is ContextUnavailableReason.MEMORY
    assert "sensitive" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_owner_p2p_can_never_read_managed_group_l2() -> None:
    state = _workforce_state()
    backend = _Backend()
    reader = _group_reader(state, allowed=True, backend=backend)

    with pytest.raises(ContextUnavailableError) as raised:
        reader.read(
            _request(
                requester_principal_id="ou_owner",
                source_requester_principal_id="ou_employee_app_owner",
                authorization_scope=EmployeeAuthorizationScope.OWNER_P2P,
            )
        )

    assert raised.value.reason is ContextUnavailableReason.SCOPE
    assert backend.calls == []


def test_l2_adapter_reads_existing_employee_group_memory(tmp_path: Path) -> None:
    state = _workforce_state()
    base = tmp_path / "team"
    memory_path = base / "groups" / "oc_1" / "SHARED_MEMORY.md"
    memory_path.parent.mkdir(parents=True)
    memory_path.write_text("full shared memory", encoding="utf-8")
    reader = _group_reader(
        state,
        allowed=True,
        backend=EmployeeGroupMemoryStore(str(base)),
    )

    assert reader.read(_request()) == "full shared memory"
