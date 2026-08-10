"""Integration tests for employee memory/document projection and facade."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

import pytest

from src.autonomous.data.facades import (
    EmployeeDocumentMaterializer,
    EmployeeMemoryFacade,
)
from src.autonomous.data.models import DataKind
from src.autonomous.data.projection import DataProjectionState, DocumentMetadataRecord


def _project_document(
    state: DataProjectionState,
    *,
    tenant_key: str,
    agent_id: str,
    kind: DataKind,
    source_id: str,
    content_hash: str,
) -> None:
    document_id = f"data_{secrets.token_hex(8)}"
    state.employee_documents[document_id] = DocumentMetadataRecord(
        document_id=document_id,
        tenant_key=tenant_key,
        agent_id=agent_id,
        owner_principal_id="owner_1",
        kind=kind,
        version=1,
        source_id=source_id,
        content_hash=content_hash,
        content_type="text/markdown",
        blob_ref={},
    )
    state.latest_employee_document[
        (tenant_key, agent_id, kind.value, source_id)
    ] = document_id


class TestEmployeeDocumentMaterializer:
    def test_materialize_l1_memory(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        content = b"# Agent Memory\nI remember everything."
        content_hash = hashlib.sha256(content).hexdigest()
        path = mat.materialize("agt_alpha", DataKind.L1_MEMORY, "l1_memory", content, content_hash)
        assert path.absolute.exists()
        assert path.absolute.read_bytes() == content
        assert path.relative == "memory/MEMORY.md"

    def test_materialize_skill_profile(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        content = json.dumps({"skills": ["coding", "analysis"]}).encode()
        content_hash = hashlib.sha256(content).hexdigest()
        path = mat.materialize("agt_alpha", DataKind.SKILL_PROFILE, "skill_profile", content, content_hash)
        assert path.relative == "skill_profile.json"
        assert json.loads(path.absolute.read_bytes()) == {"skills": ["coding", "analysis"]}

    def test_materialize_reasoning(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        content = json.dumps({"reasoning": "step by step"}).encode()
        content_hash = hashlib.sha256(content).hexdigest()
        source_id = "task_123"
        path = mat.materialize("agt_alpha", DataKind.REASONING, source_id, content, content_hash)
        assert "reasoning/" in path.relative
        assert path.absolute.exists()

    def test_materialize_memory_summary(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        content = b"# Summary for chat_1"
        content_hash = hashlib.sha256(content).hexdigest()
        from src.autonomous.data.models import EmployeeDataDocumentV1
        source_id = EmployeeDataDocumentV1.memory_summary_source_id(
            chat_id="chat_1", thread_root_id=""
        )
        path = mat.materialize("agt_alpha", DataKind.MEMORY_SUMMARY, source_id, content, content_hash)
        assert "memory/summary_" in path.relative
        assert path.absolute.read_bytes() == content

    def test_content_hash_mismatch_raises(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        with pytest.raises(ValueError, match="content hash"):
            mat.materialize("agt_alpha", DataKind.L1_MEMORY, "l1_memory", b"content", "bad_hash")

    def test_verify_succeeds_for_correct_content(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        content = b"# Memory"
        content_hash = hashlib.sha256(content).hexdigest()
        mat.materialize("agt_alpha", DataKind.L1_MEMORY, "l1_memory", content, content_hash)
        assert mat.verify("agt_alpha", DataKind.L1_MEMORY, "l1_memory", content_hash)

    def test_verify_fails_for_corrupted_content(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        content = b"# Memory"
        content_hash = hashlib.sha256(content).hexdigest()
        path = mat.materialize("agt_alpha", DataKind.L1_MEMORY, "l1_memory", content, content_hash)
        path.absolute.write_bytes(b"corrupted")
        assert not mat.verify("agt_alpha", DataKind.L1_MEMORY, "l1_memory", content_hash)

    def test_verify_fails_for_missing_file(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        assert not mat.verify("agt_alpha", DataKind.L1_MEMORY, "l1_memory", "a" * 64)

    def test_materialize_overwrites_existing(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        content1 = b"# V1"
        hash1 = hashlib.sha256(content1).hexdigest()
        mat.materialize("agt_alpha", DataKind.L1_MEMORY, "l1_memory", content1, hash1)
        content2 = b"# V2 updated"
        hash2 = hashlib.sha256(content2).hexdigest()
        path = mat.materialize("agt_alpha", DataKind.L1_MEMORY, "l1_memory", content2, hash2)
        assert path.absolute.read_bytes() == content2


class TestEmployeeMemoryFacade:
    def test_reads_canonical_l1(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        content = b"# Canonical Memory"
        content_hash = hashlib.sha256(content).hexdigest()
        mat.materialize("agt_alpha", DataKind.L1_MEMORY, "l1_memory", content, content_hash)
        state = DataProjectionState()
        _project_document(
            state,
            tenant_key="tenant_1",
            agent_id="agt_alpha",
            kind=DataKind.L1_MEMORY,
            source_id="l1_memory",
            content_hash=content_hash,
        )
        facade = EmployeeMemoryFacade(
            materializer=mat,
            state=state,
        )
        result = facade.read_l1("agt_alpha", "tenant_1")
        assert result == "# Canonical Memory"


    def test_returns_none_when_no_memory_exists(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        state = DataProjectionState()
        facade = EmployeeMemoryFacade(
            materializer=mat,
            state=state,
        )
        assert facade.read_l1("agt_missing", "tenant_1") is None

    def test_reads_skill_profile_canonical(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        skill_data = {"skills": ["python", "debugging"]}
        content = json.dumps(skill_data).encode()
        content_hash = hashlib.sha256(content).hexdigest()
        mat.materialize("agt_alpha", DataKind.SKILL_PROFILE, "skill_profile", content, content_hash)
        state = DataProjectionState()
        facade = EmployeeMemoryFacade(materializer=mat, state=state)
        assert facade.read_skill_profile("agt_alpha") == skill_data

    def test_is_canonical_detects_documents(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        state = DataProjectionState()
        state.latest_employee_document[("tenant_1", "agt_alpha", "l1_memory", "l1_memory")] = "data_abc"
        facade = EmployeeMemoryFacade(materializer=mat, state=state)
        assert facade.is_canonical("agt_alpha")
        assert not facade.is_canonical("agt_beta")

    def test_memory_summary_reads_from_canonical(self, tmp_path: Path) -> None:
        mat = EmployeeDocumentMaterializer(tmp_path / "agents")
        from src.autonomous.data.models import EmployeeDataDocumentV1
        source_id = EmployeeDataDocumentV1.memory_summary_source_id(
            chat_id="chat_1", thread_root_id=""
        )
        content = b"# Summary: agent helped with task"
        content_hash = hashlib.sha256(content).hexdigest()
        mat.materialize("agt_alpha", DataKind.MEMORY_SUMMARY, source_id, content, content_hash)
        state = DataProjectionState()
        _project_document(
            state,
            tenant_key="tenant_1",
            agent_id="agt_alpha",
            kind=DataKind.MEMORY_SUMMARY,
            source_id=source_id,
            content_hash=content_hash,
        )
        facade = EmployeeMemoryFacade(materializer=mat, state=state)
        result = facade.read_memory_summary("agt_alpha", "tenant_1", "chat_1")
        assert result == "# Summary: agent helped with task"
