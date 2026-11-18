"""Production composition: wires data keyring, BlobStore, service, and materializers."""

from __future__ import annotations

import hashlib
import secrets
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..domain import EmployeeState
from ..journal.writer import JournalWriter
from ..workspace import EmployeeWorkspaceProjector, EmployeeWorkspaceSource
from .facades import EmployeeDocumentMaterializer, EmployeeMemoryFacade
from .keyring import EmployeeDataKeyring, build_employee_data_storage
from .materializer import DailyHistoryMaterializer
from .models import (
    DataKind,
    EmployeeDataDocumentV1,
)
from .ports import PublishEmployeeDocumentCommand
from .projection import DataProjectionState
from .query import EmployeeDataRequestContextFactory, EmployeeMemoryQuery, HistoryRangeQuery
from .service import EmployeeDataService


class LegacyEmployeeDataUnsupportedError(RuntimeError):
    """Unmigrated filesystem data requires an explicit offline migration."""


def _contains_unmigrated_employee_data(agents_root: str | Path) -> bool:
    root = Path(agents_root).expanduser()
    try:
        root_mode = root.lstat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(root_mode):
        raise LegacyEmployeeDataUnsupportedError(
            "employee agents root is not a trusted directory"
        )
    leaves = (
        "execution_history.jsonl",
        "MEMORY.md",
        "memory/MEMORY.md",
    )
    for entry in root.iterdir():
        try:
            if not stat.S_ISDIR(entry.lstat().st_mode):
                continue
        except FileNotFoundError:
            continue
        for relative in leaves:
            try:
                (entry / relative).lstat()
            except FileNotFoundError:
                continue
            return True
    return False


@dataclass
class EmployeeDataComposition:
    """Complete wired data-plane ready for production use."""

    service: EmployeeDataService
    state: DataProjectionState
    history_materializer: DailyHistoryMaterializer
    document_materializer: EmployeeDocumentMaterializer
    memory_facade: EmployeeMemoryFacade
    query: HistoryRangeQuery
    memory_query: EmployeeMemoryQuery
    context_factory: EmployeeDataRequestContextFactory
    workspace_projector: EmployeeWorkspaceProjector | None = None

    def publish_document(
        self,
        command: PublishEmployeeDocumentCommand,
    ) -> None:
        """Publish one employee document from trusted orchestration."""
        content_hash = hashlib.sha256(command.content).hexdigest()
        source_id = (
            EmployeeDataDocumentV1.memory_summary_source_id(
                chat_id=command.chat_id,
                thread_root_id=command.thread_root_id,
            )
            if command.kind is DataKind.MEMORY_SUMMARY
            else command.kind.value
        )
        identity = command.idempotency_key or secrets.token_hex(16)
        doc_id = "data_" + hashlib.sha256(
            "\x00".join(
                (
                    command.tenant_key,
                    command.agent_id,
                    command.kind.value,
                    source_id,
                    identity,
                )
            ).encode()
        ).hexdigest()[:16]
        with self.service.employee_dispatch_guard():
            self.service.synchronize_projection_unlocked()
            duplicate = self.state.employee_documents.get(doc_id)
            if duplicate is not None:
                if duplicate.content_hash != content_hash:
                    raise ValueError("document idempotency conflict")
                self.document_materializer.materialize(
                    agent_id=command.agent_id,
                    kind=command.kind,
                    source_id=source_id,
                    content=command.content,
                    content_hash=content_hash,
                )
                return
            latest_key = (
                command.tenant_key,
                command.agent_id,
                command.kind.value,
                source_id,
            )
            previous_id = self.state.latest_employee_document.get(latest_key, "")
            previous = self.state.employee_documents.get(previous_id) if previous_id else None
            doc = EmployeeDataDocumentV1(
                document_id=doc_id,
                tenant_key=command.tenant_key,
                agent_id=command.agent_id,
                owner_principal_id=command.owner_principal_id,
                kind=command.kind,
                version=1 if previous is None else previous.version + 1,
                source_id=source_id,
                created_at=datetime.now(UTC).isoformat(),
                predecessor_sequence=(0 if previous is None else previous.publish_sequence),
                predecessor_hash=("" if previous is None else previous.publish_frame_hash),
                previous_document_id=previous_id,
                content_type=command.content_type,
                content_hash=content_hash,
                chat_id=(command.chat_id if command.kind == DataKind.MEMORY_SUMMARY else ""),
                thread_root_id=(
                    command.thread_root_id
                    if command.kind == DataKind.MEMORY_SUMMARY
                    else ""
                ),
            )
            self.service.publish_document(doc, command.content)
        self.document_materializer.materialize(
            agent_id=command.agent_id,
            kind=command.kind,
            source_id=source_id,
            content=command.content,
            content_hash=content_hash,
        )

    def rebuild_all(self) -> None:
        """Full projection rebuild from Journal replay."""
        snapshot = self.service.rebuild_projection()
        self.service.verify_live_blobs()
        self.history_materializer.materialize_all(snapshot)
        if self.workspace_projector is not None:
            self.workspace_projector.rebuild_all()
        agent_ids = {
            metadata.agent_id
            for metadata in snapshot.employee_documents.values()
            if not metadata.tombstoned
        }
        for agent_id in sorted(agent_ids):
            self.document_materializer.materialize_from_state(
                snapshot,
                agent_id,
                self.service.read_blob,
            )

    def close(self) -> None:
        """Release the owned encrypted BlobStore; the shared Writer is external."""
        self.service.close()


def build_employee_data_composition(
    *,
    settings: Any,
    writer: JournalWriter,
    keyring: EmployeeDataKeyring | None = None,
    admin_principal_ids: frozenset[str],
    main_bot_app_id: str,
    agents_root: str | Path,
    subject_resolver: Any = None,
) -> EmployeeDataComposition:
    """Factory that constructs the full wired data-plane from settings."""
    storage = build_employee_data_storage(settings, keyring=keyring)
    state = DataProjectionState()
    service = EmployeeDataService(
        writer=writer,
        blob_store=storage.blob_store,
        data_state=state,
        active_key_id=storage.active_key_id,
        shard_timezone=getattr(settings, "autonomous_history_timezone", "UTC"),
        authority_required=True,
    )
    service.replay_into(state)
    if state.data_authority.mode == "legacy":
        if _contains_unmigrated_employee_data(agents_root):
            service.close()
            raise LegacyEmployeeDataUnsupportedError(
                "unmigrated employee data is unsupported; run an offline migration "
                "with a prior release before starting GhostAP"
            )
        service.cutover_to_canonical()
    elif state.data_authority.mode != "canonical":
        service.close()
        raise LegacyEmployeeDataUnsupportedError(
            "unknown employee data authority is unsupported"
        )
    history_mat = DailyHistoryMaterializer(agents_root)
    doc_mat = EmployeeDocumentMaterializer(agents_root)
    memory_facade = EmployeeMemoryFacade(
        materializer=doc_mat,
        state=state,
        projection_guard=service.read_guard,
    )
    context_factory = EmployeeDataRequestContextFactory(
        admin_principal_ids=admin_principal_ids,
        main_bot_app_id=main_bot_app_id,
        subject_resolver=subject_resolver,
    )
    query = HistoryRangeQuery(
        state=state,
        context_factory=context_factory,
        max_range_days=getattr(settings, "autonomous_history_max_range_days", 31),
        page_size=getattr(settings, "autonomous_history_page_size", 50),
        audit_port=service,
    )
    memory_query = EmployeeMemoryQuery(
        memory_facade=memory_facade,
        state=state,
        context_factory=context_factory,
        audit_port=service,
    )
    from ..gateway.projection import rebuild_gateway_projection
    from ..journal.projections import ProjectionRepository

    def workspace_source_provider(
        tenant_key: str,
        agent_id: str,
    ) -> EmployeeWorkspaceSource:
        frames = tuple(writer.replay())
        workforce = ProjectionRepository().rebuild(frames)
        employee = workforce.employees.get(agent_id)
        if (
            employee is None
            or employee.tenant_key != tenant_key
            or employee.state is EmployeeState.ARCHIVED
        ):
            raise KeyError(agent_id)
        gateway = rebuild_gateway_projection(frames)
        active_attempts = tuple(
            attempt
            for attempt in gateway.attempts.values()
            if attempt.binding.tenant_key == tenant_key
            and attempt.binding.agent_id == agent_id
            and attempt.dispatch_committed
            and not attempt.terminal_status
        )
        active_assignment_id = (
            max(
                active_attempts,
                key=lambda item: item.dispatch_sequence,
            ).binding.task_id
            if active_attempts
            else ""
        )
        checkpoint_sequence = max(
            (item.dispatch_sequence for item in active_attempts),
            default=0,
        )
        return EmployeeWorkspaceSource(
            tenant_key=employee.tenant_key,
            agent_id=employee.agent_id,
            name=employee.name,
            role=employee.role,
            persona=employee.persona,
            personality_traits=employee.personality_traits,
            capabilities=employee.capabilities,
            permissions=employee.permissions,
            tool=employee.tool,
            model=employee.model,
            identity_version=employee.aggregate_version,
            projection_sequence=workforce.cursor_sequence,
            projection_hash=workforce.cursor_hash,
            knowledge_generation=0,
            active_assignment_id=active_assignment_id,
            checkpoint_ref=(
                f"journal:{checkpoint_sequence}" if checkpoint_sequence else ""
            ),
            source_refs=(),
        )

    workspace_projector = EmployeeWorkspaceProjector(
        agents_root,
        state_provider=lambda: ProjectionRepository().rebuild(writer.replay()),
        source_provider=workspace_source_provider,
    )
    return EmployeeDataComposition(
        service=service,
        state=state,
        history_materializer=history_mat,
        document_materializer=doc_mat,
        memory_facade=memory_facade,
        query=query,
        memory_query=memory_query,
        context_factory=context_factory,
        workspace_projector=workspace_projector,
    )
