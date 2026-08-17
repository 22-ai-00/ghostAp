"""Journal-backed asynchronous employee knowledge ingestion service."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from datetime import UTC, datetime
from pathlib import Path

from ..data.models import DataKind
from ..data.ports import AuthenticatedExecutionTerminal, PublishEmployeeDocumentCommand
from ..journal.blob_store import BlobRef
from ..journal.frame import GENESIS_HASH, JournalEvent
from ..journal.writer import CommitState, JournalWriter
from ..workspace.layout import atomic_write_relative, open_child_directory, open_directory_tree
from .compiler import (
    DeterministicKnowledgeCompiler,
    KnowledgeCompilerPort,
    redact_knowledge_source,
)
from .lint import lint_employee_knowledge
from .models import (
    AuthorizedKnowledgeQuery,
    KnowledgeCompilation,
    KnowledgeLintReport,
    KnowledgeQueryResult,
    KnowledgeSource,
)
from .query import EmployeeKnowledgeQuery, KnowledgeAuthorizer, parse_knowledge_page
from .review import render_review_item


class KnowledgeServiceError(RuntimeError):
    pass


class EmployeeKnowledgeService:
    def __init__(
        self,
        *,
        writer: JournalWriter,
        data_composition: object,
        authorizer: KnowledgeAuthorizer,
        compiler: KnowledgeCompilerPort | None = None,
        agents_root: str | Path | None = None,
    ) -> None:
        self._writer = writer
        self._data = data_composition
        self._compiler = compiler or DeterministicKnowledgeCompiler()
        self._query = EmployeeKnowledgeQuery(
            data_composition=data_composition,
            authorizer=authorizer,
        )
        self._agents_root = Path(agents_root) if agents_root is not None else None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._known: set[str] = set()
        self._sources: dict[str, dict[str, str]] = {}
        self._terminal: set[str] = set()
        self._review_required: set[str] = set()
        self._projection_sequence = 0
        self._projection_hash = ""
        self._closed = False
        self._thread: threading.Thread | None = None

    def enqueue_terminal(self, terminal: AuthenticatedExecutionTerminal) -> str:
        if not isinstance(terminal, AuthenticatedExecutionTerminal):
            raise TypeError("terminal must be AuthenticatedExecutionTerminal")
        if terminal.status != "completed":
            raise ValueError("only completed terminals produce knowledge")
        state = self._data.service.projection_snapshot()
        metadata = next(
            (
                item
                for item in state.history_records.values()
                if item.attempt_id == terminal.attempt_id and not item.tombstoned
            ),
            None,
        )
        if metadata is None:
            raise KnowledgeServiceError("terminal history source is unavailable")
        source_hash = str(metadata.blob_ref["payload_hash"])
        ingest_id = "kng_" + hashlib.sha256(
            f"{metadata.tenant_key}\0{metadata.agent_id}\0{source_hash}".encode()
        ).hexdigest()[:24]
        with self._lock:
            self._refresh_projection()
            if ingest_id in self._known:
                return ingest_id
            self._commit(
                ingest_id,
                "knowledge.ingest.queued",
                {
                    "tenant_key": metadata.tenant_key,
                    "agent_id": metadata.agent_id,
                    "owner_principal_id": metadata.owner_principal_id,
                    "source_id": metadata.record_id,
                    "source_hash": source_hash,
                    "attempt_id": terminal.attempt_id,
                },
            )
            self._known.add(ingest_id)
            self._sources[ingest_id] = {
                "tenant_key": metadata.tenant_key,
                "agent_id": metadata.agent_id,
                "owner_principal_id": metadata.owner_principal_id,
                "source_id": metadata.record_id,
                "source_hash": source_hash,
                "attempt_id": terminal.attempt_id,
            }
            self._ensure_thread()
            self._queue.put(ingest_id)
        return ingest_id

    def query(self, request: AuthorizedKnowledgeQuery) -> KnowledgeQueryResult:
        return self._query.query(request)

    def lint(self, tenant_key: str, agent_id: str) -> KnowledgeLintReport:
        return lint_employee_knowledge(
            data_composition=self._data,
            tenant_key=tenant_key,
            agent_id=agent_id,
            agents_root=self._agents_root,
        )

    def retry_review_item(
        self,
        ingest_id: str,
        *,
        tenant_key: str,
        agent_id: str,
    ) -> str:
        """Durably reopen one review-required ingest without accepting content."""

        with self._lock:
            self._refresh_projection()
            source = self._sources.get(ingest_id)
            if source is None or ingest_id not in self._review_required:
                raise KeyError(ingest_id)
            if (
                source.get("tenant_key") != tenant_key
                or source.get("agent_id") != agent_id
            ):
                raise PermissionError("knowledge review scope mismatch")
            self._commit(
                ingest_id,
                "knowledge.ingest.retry_requested",
                {"source_id": source["source_id"]},
            )
            self._terminal.discard(ingest_id)
            self._review_required.discard(ingest_id)
            self._ensure_thread()
            self._queue.put(ingest_id)
        return ingest_id

    def list_review_items(self, tenant_key: str, agent_id: str) -> tuple[str, ...]:
        with self._lock:
            self._refresh_projection()
            return tuple(
                ingest_id
                for ingest_id in sorted(self._review_required)
                if self._sources.get(ingest_id, {}).get("tenant_key") == tenant_key
                and self._sources.get(ingest_id, {}).get("agent_id") == agent_id
            )

    def project_all(self) -> int:
        """Rebuild source manifests/logs only from committed knowledge docs."""

        if self._agents_root is None:
            return 0
        state = self._data.service.projection_snapshot()
        agent_keys = sorted(
            {
                (item.tenant_key, item.agent_id)
                for item in state.employee_documents.values()
                if item.kind is DataKind.KNOWLEDGE_PAGE and not item.tombstoned
            }
        )
        for tenant_key, agent_id in agent_keys:
            sources: dict[str, str] = {}
            generation = 0
            for item in state.employee_documents.values():
                key = (item.tenant_key, item.agent_id, item.kind.value, item.source_id)
                if (
                    item.tenant_key != tenant_key
                    or item.agent_id != agent_id
                    or item.kind is not DataKind.KNOWLEDGE_PAGE
                    or item.tombstoned
                    or state.latest_employee_document.get(key) != item.document_id
                ):
                    continue
                raw = self._data.service.read_blob(BlobRef.from_dict(item.blob_ref))
                fields, _body = parse_knowledge_page(raw.decode("utf-8"))
                generation = max(generation, int(fields["knowledge_generation"]))
                for source_id, source_hash in zip(
                    fields["source_ids"], fields["source_hashes"], strict=True
                ):
                    sources[str(source_id)] = str(source_hash)
            manifest = ["schema_version: 1", "sources:"]
            for source_id, source_hash in sorted(sources.items()):
                manifest.extend(
                    (
                        f"  - source_id: {source_id}",
                        f"    hash: {source_hash}",
                        "    type: execution_history",
                        "    visibility: employee",
                    )
                )
            root_fd = open_directory_tree(self._agents_root)
            try:
                agent_fd = open_child_directory(root_fd, agent_id)
                try:
                    atomic_write_relative(
                        agent_fd,
                        "workspace/sources/manifest.yaml",
                        ("\n".join(manifest) + "\n").encode(),
                    )
                    atomic_write_relative(
                        agent_fd,
                        "workspace/wiki/log.md",
                        f"# Knowledge log\n\nGeneration: {generation}\n".encode(),
                    )
                finally:
                    os.close(agent_fd)
            finally:
                os.close(root_fd)
        return len(agent_keys)

    def recover(self) -> int:
        with self._lock:
            self._refresh_projection()
            pending = sorted(self._known - self._terminal)
            if pending:
                self._ensure_thread()
            for ingest_id in pending:
                self._queue.put(ingest_id)
            return len(pending)

    def drain(self) -> None:
        self._queue.join()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            if thread is not None:
                self._queue.put(None)
        if thread is not None:
            thread.join(timeout=5)

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="employee-knowledge-ingest",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            ingest_id = self._queue.get()
            try:
                if ingest_id is None:
                    return
                try:
                    self._process(ingest_id)
                except Exception:
                    # The anchored queue remains pending for recover(); one bad
                    # item must never kill ingestion for every employee.
                    pass
            finally:
                self._queue.task_done()

    def _process(self, ingest_id: str) -> None:
        with self._lock:
            if ingest_id in self._terminal:
                return
            source_meta = dict(self._sources[ingest_id])
        try:
            self._commit_effect(ingest_id, "prepared")
            self._commit_effect(ingest_id, "executing")
            state = self._data.service.projection_snapshot()
            history = state.history_records.get(source_meta["source_id"])
            if history is None or history.tombstoned:
                raise KnowledgeServiceError("knowledge source is unavailable")
            payload = self._data.service.get_history_payload(source_meta["source_id"])
            clean = redact_knowledge_source(payload.result_text)
            source = KnowledgeSource(
                source_id=source_meta["source_id"],
                source_hash=source_meta["source_hash"],
                tenant_key=source_meta["tenant_key"],
                agent_id=source_meta["agent_id"],
                owner_principal_id=source_meta["owner_principal_id"],
                text=clean,
            )
            compilation = self._compiler.compile(source)
            self._validate_compilation(compilation, source)
            generation = self._publish_pages(ingest_id, source, compilation)
            self._publish_index(ingest_id, source, generation)
            self._commit(
                ingest_id,
                "knowledge.ingest.published",
                {
                    "source_id": source.source_id,
                    "source_hash": source.source_hash,
                    "generation": generation,
                    "page_ids": [page.page_id for page in compilation.pages],
                },
            )
            try:
                self.project_all()
            except Exception:
                # Canonical Blob+Journal documents remain authoritative and a
                # later rebuild retries the Markdown projection.
                pass
        except Exception as exc:
            issue = f"compile_{type(exc).__name__.lower()}"
            self._data.publish_document(
                PublishEmployeeDocumentCommand(
                    agent_id=source_meta["agent_id"],
                    tenant_key=source_meta["tenant_key"],
                    owner_principal_id=source_meta["owner_principal_id"],
                    kind=DataKind.KNOWLEDGE_REVIEW,
                    source_id=source_meta["source_id"],
                    content=render_review_item(
                        source_id=source_meta["source_id"],
                        source_hash=source_meta["source_hash"],
                        issues=(issue,),
                    ),
                    content_type="application/json",
                    idempotency_key=source_meta["source_hash"],
                )
            )
            self._commit(
                ingest_id,
                "knowledge.ingest.review_required",
                {
                    "source_id": source_meta["source_id"],
                    "source_hash": source_meta["source_hash"],
                    "issue_code": issue,
                },
            )
        with self._lock:
            self._terminal.add(ingest_id)

    @staticmethod
    def _validate_compilation(
        compilation: KnowledgeCompilation,
        source: KnowledgeSource,
    ) -> None:
        if not isinstance(compilation, KnowledgeCompilation):
            raise ValueError("compiler output schema mismatch")
        page_ids = [page.page_id for page in compilation.pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("duplicate knowledge page id")
        for page in compilation.pages:
            for link in page.links:
                if not link or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in link):
                    raise ValueError("invalid knowledge link")
            for claim in page.claims:
                if claim.source_ids != (source.source_id,):
                    raise PermissionError("compiler cited an unauthorized source")
                if redact_knowledge_source(claim.text) != claim.text:
                    raise ValueError("compiler emitted sensitive content")

    def _publish_pages(
        self,
        ingest_id: str,
        source: KnowledgeSource,
        compilation: KnowledgeCompilation,
    ) -> int:
        state = self._data.service.projection_snapshot()
        generations = [
            item.version
            for item in state.employee_documents.values()
            if item.tenant_key == source.tenant_key
            and item.agent_id == source.agent_id
            and item.kind in {DataKind.KNOWLEDGE_PAGE, DataKind.KNOWLEDGE_INDEX}
        ]
        generation = max(generations, default=0) + 1
        updated_at = datetime.now(UTC).isoformat()
        for page in compilation.pages:
            confidence = sorted({claim.confidence.value for claim in page.claims})
            frontmatter = {
                "schema_version": 1,
                "page_id": page.page_id,
                "kind": page.kind,
                "title": page.title,
                "source_ids": [source.source_id],
                "source_hashes": [source.source_hash],
                "confidence": confidence,
                "status": "published",
                "knowledge_generation": generation,
                "updated_at": updated_at,
            }
            header = "\n".join(
                f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}"
                for key, value in frontmatter.items()
            )
            claims = "\n".join(
                f"- {claim.text} `[source:{source.source_id}]`"
                for claim in page.claims
            )
            links = "\n".join(f"- [[{link}]]" for link in page.links)
            content = (
                f"---\n{header}\n---\n# {page.title}\n\n{claims}\n"
                + (f"\n## Related\n{links}\n" if links else "")
            ).encode("utf-8")
            self._data.publish_document(
                PublishEmployeeDocumentCommand(
                    agent_id=source.agent_id,
                    tenant_key=source.tenant_key,
                    owner_principal_id=source.owner_principal_id,
                    kind=DataKind.KNOWLEDGE_PAGE,
                    source_id=page.page_id,
                    content=content,
                    content_type="text/markdown",
                    idempotency_key=f"{ingest_id}:{page.page_id}:{source.source_hash}",
                )
            )
        return generation

    def _publish_index(
        self,
        ingest_id: str,
        source: KnowledgeSource,
        generation: int,
    ) -> None:
        state = self._data.service.projection_snapshot()
        page_ids = sorted(
            item.source_id
            for item in state.employee_documents.values()
            if item.tenant_key == source.tenant_key
            and item.agent_id == source.agent_id
            and item.kind is DataKind.KNOWLEDGE_PAGE
            and not item.tombstoned
            and state.latest_employee_document.get(
                (item.tenant_key, item.agent_id, item.kind.value, item.source_id)
            )
            == item.document_id
        )
        content = (
            "# Knowledge index\n\n"
            f"Generation: {generation}\n\n"
            + "\n".join(f"- [[{page_id}]]" for page_id in page_ids)
            + "\n"
        ).encode("utf-8")
        self._data.publish_document(
            PublishEmployeeDocumentCommand(
                agent_id=source.agent_id,
                tenant_key=source.tenant_key,
                owner_principal_id=source.owner_principal_id,
                kind=DataKind.KNOWLEDGE_INDEX,
                source_id=DataKind.KNOWLEDGE_INDEX.value,
                content=content,
                content_type="text/markdown",
                idempotency_key=f"{ingest_id}:index:{generation}",
            )
        )

    def _refresh_projection(self) -> None:
        anchor, frames = self._writer.committed_tail(self._projection_sequence + 1)
        anchor_hash = "" if anchor.sequence == 0 else anchor.frame_hash
        if anchor.sequence == self._projection_sequence:
            if anchor_hash != self._projection_hash:
                raise KnowledgeServiceError("knowledge projection anchor changed")
            return

        known = set(self._known)
        terminal = set(self._terminal)
        review_required = set(self._review_required)
        sources = {key: dict(value) for key, value in self._sources.items()}
        expected_sequence = self._projection_sequence + 1
        expected_previous_hash = self._projection_hash or GENESIS_HASH
        for frame in frames:
            if (
                frame.sequence != expected_sequence
                or frame.previous_hash != expected_previous_hash
            ):
                raise KnowledgeServiceError("knowledge projection tail is discontinuous")
            for event in frame.events:
                if not event.aggregate_id.startswith("knowledge-ingest:"):
                    continue
                ingest_id = event.aggregate_id.removeprefix("knowledge-ingest:")
                if event.event_type == "knowledge.ingest.queued":
                    known.add(ingest_id)
                    sources[ingest_id] = {key: str(value) for key, value in event.payload.items()}
                elif event.event_type == "knowledge.ingest.published":
                    terminal.add(ingest_id)
                    review_required.discard(ingest_id)
                elif event.event_type == "knowledge.ingest.review_required":
                    terminal.add(ingest_id)
                    review_required.add(ingest_id)
                elif event.event_type == "knowledge.ingest.retry_requested":
                    terminal.discard(ingest_id)
                    review_required.discard(ingest_id)
            expected_sequence = frame.sequence + 1
            expected_previous_hash = frame.frame_hash
        if expected_sequence - 1 != anchor.sequence or expected_previous_hash != anchor_hash:
            raise KnowledgeServiceError("knowledge projection tail does not reach anchor")
        self._known = known
        self._terminal = terminal
        self._review_required = review_required
        self._sources = sources
        self._projection_sequence = anchor.sequence
        self._projection_hash = anchor_hash

    def _commit_effect(self, ingest_id: str, state: str) -> None:
        self._commit(
            ingest_id,
            f"knowledge.compile.{state}",
            {"effect_type": "knowledge_compile"},
        )

    def _commit(self, ingest_id: str, event_type: str, payload: dict[str, object]) -> None:
        aggregate = f"knowledge-ingest:{ingest_id}"
        event = JournalEvent(event_type=event_type, aggregate_id=aggregate, payload=payload)
        with self._writer.transaction_guard():
            last = self._writer.get_last_frame()
            result = self._writer.commit(
                (event,),
                self._writer.get_aggregate_versions((aggregate,)),
                expected_head_sequence=0 if last is None else last.sequence,
                expected_head_hash="" if last is None else last.frame_hash,
            )
        if result.state is not CommitState.ANCHORED:
            raise KnowledgeServiceError("knowledge event was not anchored")


__all__ = ["EmployeeKnowledgeService", "KnowledgeServiceError"]
