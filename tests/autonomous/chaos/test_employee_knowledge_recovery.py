from __future__ import annotations

import pytest

from src.autonomous.knowledge import DeterministicKnowledgeCompiler, EmployeeKnowledgeService
from tests.autonomous.knowledge_helpers import (
    build_knowledge_composition,
    close_knowledge_composition,
    seed_terminal,
)


class _CountingCompiler(DeterministicKnowledgeCompiler):
    def __init__(self) -> None:
        self.calls = 0

    def compile(self, source):
        self.calls += 1
        return super().compile(source)


def test_anchored_ingest_queue_resumes_once_after_restart(tmp_path) -> None:
    writer, composition = build_knowledge_composition(tmp_path)
    original = composition.knowledge_service
    original._ensure_thread = lambda: None  # type: ignore[method-assign]  # noqa: SLF001
    terminal = seed_terminal(composition, "restart", "Recover this durable finding.")
    original.enqueue_terminal(terminal)
    original._closed = True  # noqa: SLF001

    compiler = _CountingCompiler()
    recovered = EmployeeKnowledgeService(
        writer=writer,
        data_composition=composition,
        authorizer=lambda _request, _source: True,
        compiler=compiler,
        agents_root=tmp_path / "agents",
    )
    composition.knowledge_service = recovered
    assert recovered.recover() == 1
    recovered.drain()
    assert compiler.calls == 1
    assert recovered.recover() == 0
    close_knowledge_composition(writer, composition)


def test_enqueue_uses_verified_projection_caches_without_disk_replay(
    tmp_path,
    monkeypatch,
) -> None:
    writer, composition = build_knowledge_composition(tmp_path)
    service = composition.knowledge_service
    service._ensure_thread = lambda: None  # type: ignore[method-assign]  # noqa: SLF001
    terminal = seed_terminal(composition, "cached", "Reuse verified projections.")

    def fail_replay(*_args, **_kwargs):
        pytest.fail("hot knowledge enqueue must not replay the Journal from disk")

    monkeypatch.setattr(writer, "replay", fail_replay)
    ingest_id = service.enqueue_terminal(terminal)

    assert ingest_id.startswith("kng_")
    assert ingest_id in service._known  # noqa: SLF001
    close_knowledge_composition(writer, composition)
