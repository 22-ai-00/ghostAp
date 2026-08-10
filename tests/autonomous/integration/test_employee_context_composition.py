from __future__ import annotations

from types import SimpleNamespace

from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime


class _HireService:
    def synchronize_projection(self):
        from src.autonomous.journal.projections import ProjectionState

        return ProjectionState()


def test_context_composition_wires_acl_registry_and_injected_backend(tmp_path) -> None:
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._runtime_enabled = True
    runtime._service = _HireService()
    runtime._writer = object()
    runtime._vault = object()
    runtime._data = SimpleNamespace(memory_facade=object(), service=object())
    runtime._channels = object()
    runtime._group_ledger = None
    runtime._context_acl = None
    runtime._context_source_factory = None
    runtime._context_service = None
    runtime._group_memory_backend = None
    runtime._owns_group_memory_backend = False
    runtime._context_blockers = ("employee_context",)
    runtime._execution_blockers = ()
    source_factory = SimpleNamespace(close=lambda: None)
    backend = SimpleNamespace(read_group_memory=lambda _chat_id: "")
    settings = SimpleNamespace(
        autonomous_visible_employee_limit=1,
        autonomous_employee_storage_base=str(tmp_path),
        autonomous_manager_acl="ou_manager",
        admin_user_ids=("ou_admin",),
        autonomous_thread_context_max_messages=50,
        autonomous_group_context_max_messages=20,
        autonomous_context_max_tokens=4_000,
        autonomous_thread_context_max_chars=16_000,
        autonomous_thread_context_page_size=50,
        autonomous_group_context_page_size=20,
        autonomous_context_fetch_timeout_seconds=30.0,
        autonomous_context_max_pages=10,
    )

    runtime._compose_context(
        settings,
        context_source_factory=source_factory,
        group_memory_backend=backend,
    )

    assert runtime._context_blockers == ()
    assert runtime._context_service is not None
    assert runtime._context_source_factory is source_factory
    assert runtime._group_memory_backend is backend
    assert runtime._context_acl.is_authorized
