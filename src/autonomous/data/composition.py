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
    ExecutionAttemptContext,
    ExecutionHistoryPayloadV1,
    ExecutionHistoryRecordV1,
    SafeExecutionSummary,
    ToolUsageV1,
)
from .ports import AuthenticatedExecutionTerminal, PublishEmployeeDocumentCommand
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
        "skill_profile.json",
        "reasoning",
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
    knowledge_service: Any = None

    def record_terminal(
        self,
        terminal: AuthenticatedExecutionTerminal,
    ) -> None:
        """Record one terminal execution outcome from trusted orchestration."""
        binding = self.state.execution_attempts.get(terminal.attempt_id)
        if binding is None:
            raise ValueError(f"no anchored attempt: {terminal.attempt_id}")
        ctx = ExecutionAttemptContext(
            tenant_key=binding.tenant_key,
            agent_id=binding.agent_id,
            owner_principal_id=binding.owner_principal_id,
            requester_principal_id=binding.requester_principal_id,
            task_id=binding.task_id,
            run_id=binding.run_id,
            attempt_id=binding.attempt_id,
            message_id=binding.message_id,
            thread_root_id=binding.thread_root_id,
            chat_id=binding.chat_id,
            tool=binding.tool,
            model=binding.model,
            effort=binding.effort,
            started_at=binding.started_at,
            terminal_epoch=binding.terminal_epoch,
        )
        ended_at = datetime.now(UTC).isoformat()
        record = ExecutionHistoryRecordV1.from_attempt(
            ctx,
            ended_at=ended_at,
            status=terminal.status,
            safe_summary=SafeExecutionSummary.build(
                status=terminal.status,
                tool_count=len(terminal.tool_usage),
                attachment_count=len(terminal.attachments),
            ),
            prompt_tokens=terminal.prompt_tokens,
            completion_tokens=terminal.completion_tokens,
            tool_usage=tuple(
                ToolUsageV1(t["name"], t["count"], t["duration_ms"], t["status"])
                for t in terminal.tool_usage
            ) if terminal.tool_usage else (),
            predecessor_sequence=0,
            predecessor_hash="",
            shard_timezone=self.service.shard_timezone,
        )
        payload = ExecutionHistoryPayloadV1(
            record_id=record.record_id,
            occurrence_key=record.occurrence_key,
            request_text=terminal.request_text[:1_000_000],
            result_text=terminal.result_text[:1_000_000],
            error_detail=terminal.error_detail[:100_000],
            attachments=terminal.attachments,
            tool_calls=(),
        )
        self.service.record_history(record, payload)

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
            else command.source_id
            if command.kind in {
                DataKind.REASONING,
                DataKind.KNOWLEDGE_PAGE,
                DataKind.KNOWLEDGE_REVIEW,
            }
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

    def enqueue_knowledge_terminal(
        self,
        terminal: AuthenticatedExecutionTerminal,
    ) -> str:
        if self.knowledge_service is None:
            raise RuntimeError("employee knowledge service is unavailable")
        return self.knowledge_service.enqueue_terminal(terminal)

    def rebuild_all(self) -> None:
        """Full projection rebuild from Journal replay."""
        snapshot = self.service.rebuild_projection()
        self.service.verify_live_blobs()
        self.history_materializer.materialize_all(snapshot)
        if self.workspace_projector is not None:
            # Rebuild the deterministic control skeleton first; committed
            # knowledge documents then overlay only managed Wiki leaves.
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
        if self.knowledge_service is not None:
            self.knowledge_service.project_all()

    def gc_unreferenced_blobs(self) -> int:
        """Quarantine blobs not referenced by any projected record or document."""
        return self.service.quarantine_unreferenced_blobs()

    def close(self) -> None:
        """Release the owned encrypted BlobStore; the shared Writer is external."""
        if self.knowledge_service is not None:
            self.knowledge_service.close()
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
    from ..team.models import TeamAssignmentStatus, TeamRunPhase
    from ..team.projection import rebuild_team_projection

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
        data_snapshot = service.rebuild_projection()
        documents = tuple(
            document
            for document in data_snapshot.employee_documents.values()
            if document.tenant_key == tenant_key
            and document.agent_id == agent_id
            and not document.tombstoned
        )
        knowledge_generation = max(
            (
                document.version
                for document in documents
                if document.kind
                in {DataKind.KNOWLEDGE_PAGE, DataKind.KNOWLEDGE_INDEX}
            ),
            default=0,
        )
        source_refs = tuple(
            (
                document.source_id,
                document.content_hash,
                document.kind.value,
                "employee",
            )
            for document in sorted(
                (
                    item
                    for item in documents
                    if item.kind is DataKind.KNOWLEDGE_PAGE
                    and data_snapshot.latest_employee_document.get(
                        (
                            item.tenant_key,
                            item.agent_id,
                            item.kind.value,
                            item.source_id,
                        )
                    )
                    == item.document_id
                ),
                key=lambda item: (item.source_id, item.version),
            )[-100:]
        )
        team = rebuild_team_projection(frames)
        active_team_assignments = tuple(
            assignment
            for assignment in team.assignments.values()
            if assignment.agent_id == agent_id
            and assignment.status
            in {
                TeamAssignmentStatus.CREATED,
                TeamAssignmentStatus.CLAIMED,
                TeamAssignmentStatus.RUNNING,
            }
            and team.runs[assignment.run_id].tenant_key == tenant_key
            and team.runs[assignment.run_id].phase
            not in {
                TeamRunPhase.COMPLETED,
                TeamRunPhase.BLOCKED,
                TeamRunPhase.CANCELED,
            }
        )
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
            sorted(
                active_team_assignments,
                key=lambda item: item.assignment_id,
            )[-1].assignment_id
            if active_team_assignments
            else max(
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
            knowledge_generation=knowledge_generation,
            active_assignment_id=active_assignment_id,
            checkpoint_ref=(
                f"journal:{checkpoint_sequence}" if checkpoint_sequence else ""
            ),
            source_refs=source_refs,
        )

    workspace_projector = EmployeeWorkspaceProjector(
        agents_root,
        state_provider=lambda: ProjectionRepository().rebuild(writer.replay()),
        source_provider=workspace_source_provider,
    )
    composition = EmployeeDataComposition(
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
    from ..knowledge.service import EmployeeKnowledgeService

    principals = frozenset(admin_principal_ids) | {
        main_bot_app_id,
        "main_bot",
    }

    def authorize_knowledge(request: Any, _source_id: str) -> bool:
        if request.requester_principal_id in principals:
            return True
        owners = {
            item.owner_principal_id
            for item in state.employee_documents.values()
            if item.tenant_key == request.tenant_key
            and item.agent_id == request.agent_id
            and not item.tombstoned
        }
        return request.requester_principal_id in owners

    composition.knowledge_service = EmployeeKnowledgeService(
        writer=writer,
        data_composition=composition,
        authorizer=authorize_knowledge,
        agents_root=agents_root,
    )
    return composition
