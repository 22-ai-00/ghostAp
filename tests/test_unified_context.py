"""Contracts for the append-only project context used by active handlers."""

from src.project.context_helper import filter_context_entries
from src.project.unified_context import (
    ContextBridgeSummary,
    ContextEntry,
    ContextEntryType,
    ContextSourceMode,
    ProjectContextManager,
    UnifiedContext,
)


def test_context_entry_exposes_domain_values() -> None:
    entry = ContextEntry(
        entry_id="entry-1",
        seq=7,
        entry_type=ContextEntryType.FILE_CHANGE,
        source_mode=ContextSourceMode.CODEX,
        content="changed src/main.py",
        metadata={"path": "src/main.py"},
        created_at=123.0,
    )

    assert entry.entry_id == "entry-1"
    assert entry.seq == 7
    assert entry.entry_type is ContextEntryType.FILE_CHANGE
    assert entry.source_mode is ContextSourceMode.CODEX
    assert entry.metadata == {"path": "src/main.py"}


def test_bridge_prompt_contains_only_available_sections() -> None:
    minimal = ContextBridgeSummary(
        from_mode=ContextSourceMode.SMART,
        to_mode=ContextSourceMode.CODEX,
    ).to_injection_prompt()
    full = ContextBridgeSummary(
        from_mode=ContextSourceMode.CODEX,
        to_mode=ContextSourceMode.DEEP_ENGINE,
        summary_text="Implemented the parser",
        key_decisions=["Keep the public API small"],
        files_modified=["src/parser.py"],
        pending_tasks=["Run integration tests"],
    ).to_injection_prompt()

    assert "[Context from previous smart session]" in minimal
    assert "[End of context]" in minimal
    assert "Implemented the parser" in full
    assert "Keep the public API small" in full
    assert "src/parser.py" in full
    assert "Run integration tests" in full


def test_append_helpers_create_typed_sequenced_entries() -> None:
    context = UnifiedContext("project")
    custom = context.add_entry(
        ContextEntry(
            entry_type=ContextEntryType.FILE_CHANGE,
            source_mode=ContextSourceMode.CODEX,
            content="changed a.py",
        )
    )
    conversation = context.add_conversation(
        "user", "fix it", ContextSourceMode.CODEX, message_id="message-1"
    )
    snapshot = context.add_session_snapshot(
        {"session_id": "session-1"}, ContextSourceMode.CLAUDE
    )
    transition = context.add_mode_transition(
        ContextSourceMode.SMART, ContextSourceMode.CODEX, "route coding task"
    )
    deep_result = context.add_deep_engine_result(
        {"name": "implementation", "tasks": [{"title": "edit", "status": "completed"}]}
    )

    assert [entry.seq for entry in context.entries] == [1, 2, 3, 4, 5]
    assert custom.entry_type is ContextEntryType.FILE_CHANGE
    assert conversation.metadata == {"role": "user", "message_id": "message-1"}
    assert snapshot.entry_type is ContextEntryType.SESSION_SNAPSHOT
    assert snapshot.source_mode is ContextSourceMode.CLAUDE
    assert transition.metadata["to_mode"] == "codex"
    assert deep_result.entry_type is ContextEntryType.DEEP_ENGINE_RESULT
    assert deep_result.source_mode is ContextSourceMode.DEEP_ENGINE


def test_get_entries_by_type_preserves_append_order() -> None:
    context = UnifiedContext("project")
    first = context.add_conversation("user", "one", ContextSourceMode.COCO)
    context.add_session_snapshot({"session_id": "session"}, ContextSourceMode.COCO)
    second = context.add_conversation("assistant", "two", ContextSourceMode.COCO)

    assert context.get_entries_by_type(ContextEntryType.CONVERSATION) == [first, second]


def test_rolling_window_evicts_old_entries_without_reusing_sequences() -> None:
    context = UnifiedContext("project", max_entries=2)
    context.add_conversation("user", "one", ContextSourceMode.CODEX)
    context.add_conversation("user", "two", ContextSourceMode.CODEX)
    context.add_conversation("user", "three", ContextSourceMode.CODEX)

    assert [entry.content for entry in context.entries] == ["two", "three"]
    assert [entry.seq for entry in context.entries] == [2, 3]
    assert context.entry_count == 2


def test_zero_entry_limit_keeps_all_entries() -> None:
    context = UnifiedContext("project", max_entries=0)
    for index in range(5):
        context.add_conversation("user", str(index), ContextSourceMode.SMART)

    assert context.entry_count == 5


def test_versions_capture_monotonic_sequence_boundaries() -> None:
    context = UnifiedContext("project")
    context.add_conversation("user", "one", ContextSourceMode.CODEX)
    first = context.create_version("checkpoint", ContextSourceMode.CODEX, "first")
    context.add_conversation("assistant", "two", ContextSourceMode.CODEX)
    second = context.create_version("complete", ContextSourceMode.CODEX)

    assert (first.version_number, second.version_number) == (1, 2)
    assert (first.entry_count, first.last_seq) == (1, 1)
    assert (second.entry_count, second.last_seq) == (2, 2)
    assert context.get_version(1) is first
    assert context.get_version(2) is second


def test_version_window_evicts_old_snapshots() -> None:
    context = UnifiedContext("project", max_versions=1)
    first = context.create_version("first", ContextSourceMode.SMART)
    second = context.create_version("second", ContextSourceMode.SMART)

    assert context.versions == [second]
    assert context.get_version(first.version_number) is None


def test_filter_context_entries_uses_version_sequence_range() -> None:
    context = UnifiedContext("project")
    context.add_conversation("user", "before", ContextSourceMode.CODEX)
    first = context.create_version("first", ContextSourceMode.CODEX)
    context.add_conversation("user", "middle-1", ContextSourceMode.CODEX)
    context.add_conversation("user", "middle-2", ContextSourceMode.CODEX)
    second = context.create_version("second", ContextSourceMode.CODEX)
    context.add_conversation("user", "after", ContextSourceMode.CODEX)

    from_version, to_version, entries = filter_context_entries(
        context, first.version_number, second.version_number
    )

    assert from_version is first
    assert to_version is second
    assert [entry.content for entry in entries] == ["middle-1", "middle-2"]


def test_filter_context_entries_survives_rolling_eviction() -> None:
    context = UnifiedContext("project", max_entries=2)
    context.add_conversation("user", "evicted", ContextSourceMode.CODEX)
    version = context.create_version("before work", ContextSourceMode.CODEX)
    context.add_conversation("user", "kept-1", ContextSourceMode.CODEX)
    context.add_conversation("user", "kept-2", ContextSourceMode.CODEX)

    _, _, entries = filter_context_entries(
        context, version.version_number, None, show_current=True
    )

    assert [entry.content for entry in entries] == ["kept-1", "kept-2"]


def test_build_bridge_summary_uses_source_conversation() -> None:
    context = UnifiedContext("project")
    context.add_conversation("user", "help me refactor auth", ContextSourceMode.COCO)
    context.add_conversation("assistant", "I'll create a plan", ContextSourceMode.COCO)

    bridge = context.build_bridge_summary(ContextSourceMode.COCO, ContextSourceMode.CLAUDE)

    assert bridge.from_mode is ContextSourceMode.COCO
    assert bridge.to_mode is ContextSourceMode.CLAUDE
    assert "refactor auth" in bridge.summary_text
    assert "create a plan" in bridge.summary_text


def test_build_bridge_summary_respects_item_limit() -> None:
    context = UnifiedContext("project")
    for index in range(5):
        context.add_conversation("user", f"message-{index}", ContextSourceMode.COCO)

    bridge = context.build_bridge_summary(
        ContextSourceMode.COCO, ContextSourceMode.CLAUDE, max_items=2
    )

    assert "message-2" not in bridge.summary_text
    assert "message-3" in bridge.summary_text
    assert "message-4" in bridge.summary_text


def test_build_bridge_summary_includes_deep_task_results() -> None:
    context = UnifiedContext("project")
    context.add_deep_engine_result(
        {
            "name": "refactor",
            "tasks": [
                {
                    "title": "Create JWT module",
                    "status": "completed",
                    "result": "Created auth/jwt.py",
                },
                {"title": "Update tests", "status": "failed", "result": None},
            ],
        }
    )

    bridge = context.build_bridge_summary(
        ContextSourceMode.DEEP_ENGINE, ContextSourceMode.COCO
    )

    assert "Create JWT module" in bridge.summary_text


def test_bridge_prompt_can_be_consumed_only_once() -> None:
    context = UnifiedContext("project")
    context.add_conversation("user", "refactor auth", ContextSourceMode.COCO)
    built = context.build_bridge_summary(ContextSourceMode.COCO, ContextSourceMode.CLAUDE)

    assert "refactor auth" in built.to_injection_prompt()
    assert context.consume_bridge_summary() is built
    assert context.last_bridge_summary is None
    assert context.consume_bridge_summary() is None


def test_bridge_skips_non_bridgeable_entries() -> None:
    context = UnifiedContext("project")
    context.add_mode_transition(ContextSourceMode.SMART, ContextSourceMode.COCO)
    context.add_session_snapshot({"session_id": "session-1"}, ContextSourceMode.COCO)
    context.add_conversation("user", "actual content", ContextSourceMode.COCO)

    bridge = context.build_bridge_summary(ContextSourceMode.COCO, ContextSourceMode.CLAUDE)

    assert "actual content" in bridge.summary_text
    assert "session-1" not in bridge.summary_text


def test_manager_get_or_create_uses_configured_limits() -> None:
    manager = ProjectContextManager(max_entries=2, max_versions=1)

    assert manager.get("project") is None
    context = manager.get_or_create("project")

    assert manager.get("project") is context
    assert manager.get_or_create("project") is context
    assert context.max_entries == 2
    assert context.max_versions == 1
    assert manager.store is manager


def test_manager_scopes_context_by_chat_and_project() -> None:
    manager = ProjectContextManager()
    first_chat = manager.get_or_create("project", chat_id="chat-1")
    second_chat = manager.get_or_create("project", chat_id="chat-2")
    other_project = manager.get_or_create("other", chat_id="chat-1")

    assert first_chat is manager.get("project", chat_id="chat-1")
    assert second_chat is manager.get("project", chat_id="chat-2")
    assert first_chat is not second_chat
    assert first_chat is not other_project
    assert manager.get("project") is None


def test_manager_update_appends_all_supported_payloads() -> None:
    manager = ProjectContextManager()

    count = manager.update_context(
        "project",
        chat_id="chat",
        conversation={
            "role": "user",
            "content": "implement it",
            "source_mode": "codex",
            "message_id": "message-1",
        },
        session_snapshot={
            "data": {"session_id": "session-1"},
            "source_mode": "claude",
        },
        mode_transition={
            "from_mode": "smart",
            "to_mode": "codex",
            "reason": "coding task",
        },
        deep_result={
            "data": {
                "name": "implementation",
                "tasks": [{"title": "edit", "status": "completed"}],
            }
        },
    )

    context = manager.get("project", chat_id="chat")
    assert count == 4
    assert context is not None
    assert [entry.entry_type for entry in context.entries] == [
        ContextEntryType.CONVERSATION,
        ContextEntryType.SESSION_SNAPSHOT,
        ContextEntryType.MODE_TRANSITION,
        ContextEntryType.DEEP_ENGINE_RESULT,
    ]
    assert context.entries[0].source_mode is ContextSourceMode.CODEX
    assert context.entries[1].source_mode is ContextSourceMode.CLAUDE


def test_manager_update_rejects_blank_project_without_creating_context() -> None:
    manager = ProjectContextManager()

    count = manager.update_context(
        "   ",
        conversation={"role": "user", "content": "ignored", "source_mode": "codex"},
    )

    assert count == 0
    assert manager.get("   ") is None
