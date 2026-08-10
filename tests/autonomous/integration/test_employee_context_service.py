"""Integration coverage for authority-bound employee Context assembly."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.autonomous.authorization import EmployeeAuthorizationScope
from src.autonomous.context import (
    AuthorizedContextRequest,
    AuthorizedGroupMemoryReader,
    ContextLayer,
    ContextPreparingExecutionPort,
    ContextUnavailableError,
    ContextUnavailableReason,
    EmployeeContextService,
    MessagePage,
    ThreadContextConfig,
)
from src.autonomous.data.projection import JournalHead
from src.autonomous.domain import (
    BotPrincipal,
    EmployeeDefinition,
    EmployeeState,
    WorkerType,
)
from src.autonomous.journal.projections import ProjectionState
from src.autonomous.workforce.registry import ProjectedAgentRegistry
from tests.autonomous.helpers import (
    FakeEmployeeMessageSource as _FakeSource,
)
from tests.autonomous.helpers import (
    make_context_message as _message,
)
from tests.autonomous.helpers import (
    message_pages as _pages,
)
from tests.autonomous.helpers import (
    stable_thread as _stable_thread,
)


def _request(**changes) -> AuthorizedContextRequest:
    values = dict(
        tenant_key="tenant_1",
        agent_id="agt_1",
        bot_principal_id="bot_1",
        app_id="cli_1",
        channel_generation=7,
        chat_id="oc_1",
        thread_root_message_id="om_root",
        feishu_thread_id="omt_1",
        current_message_id="om_current",
        requester_principal_id="ou_1",
        source_requester_principal_id="ou_user",
        authorization_scope=EmployeeAuthorizationScope.MANAGED_GROUP,
        system_prompt_token_reserve=2,
        constraints_digest="a" * 64,
    )
    values.update(changes)
    return AuthorizedContextRequest(**values)


def _state() -> ProjectionState:
    state = ProjectionState()
    state.employees["agt_1"] = EmployeeDefinition(
        agent_id="agt_1",
        tenant_key="tenant_1",
        owner_principal_id="ou_owner",
        name="Atlas",
        tool="codex",
        model="gpt",
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


class _BooleanAuthority:
    def __init__(self, values: list[bool] | None = None) -> None:
        self._values = list(values or [True])
        self.calls: list[AuthorizedContextRequest] = []

    def is_current(self, request: AuthorizedContextRequest) -> bool:
        self.calls.append(request)
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]


class _Acl:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[AuthorizedContextRequest] = []

    def is_authorized(self, request: AuthorizedContextRequest) -> bool:
        self.calls.append(request)
        return self.allowed


class _MemoryFacade:
    def __init__(self, content: str | None = "L1") -> None:
        self.content = content
        self.calls: list[tuple[str, str]] = []

    def read_l1(
        self,
        agent_id: str,
        tenant_key: str,
    ) -> str | None:
        self.calls.append((agent_id, tenant_key))
        return self.content


class _DataService:
    def __init__(self, head: JournalHead = JournalHead()) -> None:
        self.head = head

    def get_head(self) -> JournalHead:
        return self.head


class _GroupBackend:
    def __init__(self, content: str = "L2") -> None:
        self.content = content
        self.calls: list[str] = []

    def read_group_memory(self, chat_id: str) -> str:
        self.calls.append(chat_id)
        return self.content


class _SourceFactory:
    def __init__(self, *, assembly_attempts: int = 1) -> None:
        thread = _stable_thread()
        self.source = _FakeSource(
            traversals=[
                _pages(thread)
                for _ in range(assembly_attempts * 2)
            ],
        )
        self.calls = []
        self.close_calls = 0

    @contextmanager
    def open(self, *, scope, principal):
        self.calls.append((scope, principal))
        self.source.scope = scope
        try:
            yield self.source
        finally:
            self.close_calls += 1


class _Delegate:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, execution_input):
        self.calls.append(execution_input)
        return "task_1"


class _Fence:
    def run_if_current(self, request, action):
        del request
        return action()


def _composition(
    *,
    state: ProjectionState | None = None,
    generation: _BooleanAuthority | None = None,
    acl: _Acl | None = None,
    memory: _MemoryFacade | None = None,
    backend: _GroupBackend | None = None,
    source_factory: _SourceFactory | None = None,
    config: ThreadContextConfig | None = None,
    group_ledger=None,
):
    workforce_state = state or _state()

    def registry_provider() -> ProjectedAgentRegistry:
        return ProjectedAgentRegistry(workforce_state)

    generation = generation or _BooleanAuthority()
    acl = acl or _Acl()
    memory = memory or _MemoryFacade()
    backend = backend or _GroupBackend()
    source_factory = source_factory or _SourceFactory()
    data = SimpleNamespace(
        memory_facade=memory,
        service=_DataService(
            JournalHead(
                workforce_state.cursor_sequence,
                workforce_state.cursor_hash,
            )
        ),
    )
    group_reader = AuthorizedGroupMemoryReader(
        registry_provider=registry_provider,
        requester_acl=acl,
        backend=backend,
    )
    service = EmployeeContextService(
        registry_provider=registry_provider,
        generation_authority=generation,
        requester_acl=acl,
        data_composition=data,
        group_memory_reader=group_reader,
        source_factory=source_factory,
        config=config,
        group_ledger=group_ledger,
    )
    return SimpleNamespace(
        service=service,
        generation=generation,
        acl=acl,
        memory=memory,
        backend=backend,
        source_factory=source_factory,
        data=data,
    )


def test_assembles_once_from_projected_authority_and_canonical_memories() -> None:
    built = _composition()

    snapshot = built.service.assemble(_request())

    assert snapshot.l1_summary == "L1"
    assert snapshot.l2_summary == "L2"
    assert snapshot.system_prompt_tokens_reserved == 2
    assert snapshot.constraints_digest == "a" * 64
    assert built.memory.calls == [("agt_1", "tenant_1")]
    assert built.backend.calls == ["oc_1"]
    assert len(built.source_factory.calls) == 1
    scope, principal = built.source_factory.calls[0]
    assert scope == _request().to_message_scope()
    assert principal.bot_principal_id == "bot_1"
    assert built.source_factory.close_calls == 1


def test_missing_l1_and_l2_are_legal_empty_layers() -> None:
    built = _composition(
        memory=_MemoryFacade(None),
        backend=_GroupBackend(""),
    )

    snapshot = built.service.assemble(_request())

    assert snapshot.l1_summary == ""
    assert snapshot.l2_summary == ""
    assert built.source_factory.close_calls == 1


def test_owner_p2p_reads_only_thread_and_employee_l1() -> None:
    state = _state()
    state.employees["agt_1"] = replace(
        state.employees["agt_1"],
        owner_principal_id="ou_owner",
        member_groups=(),
    )
    thread = _stable_thread()
    pages = [
        MessagePage(tuple(thread[:2]), True, "1"),
        MessagePage(tuple(thread[2:]), False),
    ]
    source_factory = _SourceFactory()
    source_factory.source = _FakeSource(
        traversals=[list(pages), list(pages)],
    )

    class HostileLedger:
        calls = 0

        def assemble_partial(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("P2P must never read group ledger content")

    ledger = HostileLedger()
    built = _composition(
        state=state,
        backend=_GroupBackend("hostile group L2"),
        source_factory=source_factory,
        group_ledger=ledger,
    )
    request = _request(
        requester_principal_id="ou_owner",
        source_requester_principal_id="ou_user",
        authorization_scope=EmployeeAuthorizationScope.OWNER_P2P,
    )

    snapshot = built.service.assemble(request)

    assert [item.message_id for item in snapshot.thread_messages] == [
        "om_root",
        "om_before",
        "om_current",
    ]
    assert snapshot.l1_summary == "L1"
    assert snapshot.l2_summary == ""
    assert snapshot.group_messages == ()
    assert built.backend.calls == []
    assert built.source_factory.source.chat_calls == 0
    assert built.source_factory.source.reset_calls == 0
    assert built.source_factory.source.thread_calls == 4
    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble_canonical_partial(
            request,
            warning_reason=ContextUnavailableReason.ORDERING,
        )
    assert raised.value.reason is ContextUnavailableReason.ORDERING
    assert ledger.calls == 0
    scope, _principal = built.source_factory.calls[0]
    assert scope.authorization_scope is EmployeeAuthorizationScope.OWNER_P2P


def test_budget_preserves_full_topic_and_l1_before_recent_group() -> None:
    thread = _stable_thread()
    thread_pages = [
        MessagePage(tuple(thread[:2]), True, "1"),
        MessagePage(tuple(thread[2:]), False),
    ]
    group = _message("om_group", "G" * 10, create=1_500, position=10)

    class Source(_FakeSource):
        def list_chat_messages(self, *, page_token="", page_size=20):
            del page_token, page_size
            self.chat_calls += 1
            return MessagePage((group,), False)

    source = Source(traversals=[list(thread_pages), list(thread_pages)])
    source_factory = _SourceFactory()
    source_factory.source = source
    built = _composition(
        memory=_MemoryFacade("M" * 10),
        backend=_GroupBackend("T" * 10),
        source_factory=source_factory,
        config=ThreadContextConfig(
            max_context_chars=27,
            max_context_tokens=100,
            tokens_per_char=1.0,
        ),
    )

    snapshot = built.service.assemble(_request())

    assert [item.message_id for item in snapshot.thread_messages] == [
        "om_root",
        "om_before",
        "om_current",
    ]
    assert snapshot.l1_summary == "M" * 10
    assert snapshot.group_messages == ()
    assert snapshot.l2_summary == ""
    assert tuple(item.layer for item in snapshot.trimming_trace) == (
        ContextLayer.L2_GROUP,
        ContextLayer.GROUP_RECENT,
    )
    assert source.thread_calls == 4
    assert source.chat_calls == 2
    assert source.reset_calls == 2


def test_topic_that_alone_exceeds_budget_fails_instead_of_partial_trim() -> None:
    built = _composition(
        memory=_MemoryFacade(""),
        backend=_GroupBackend(""),
        config=ThreadContextConfig(
            max_context_chars=16,
            max_context_tokens=100,
            tokens_per_char=1.0,
        ),
    )
    request = _request(
        requester_principal_id="ou_owner",
        authorization_scope=EmployeeAuthorizationScope.OWNER_P2P,
    )

    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble(request)

    assert raised.value.reason is ContextUnavailableReason.BUDGET


def test_group_partial_binds_current_sender_to_source_principal() -> None:
    from src.autonomous.context.group_ledger import (
        GroupContextLedger,
        GroupEventPayload,
    )

    record = SimpleNamespace(
        message_id="om_current",
        chat_id="oc_1",
        thread_id="omt_1",
        payload_ref=object(),
    )
    payload = GroupEventPayload(
        sender_id="ou_user",
        sender_id_type="open_id",
        sender_type="user",
        sender_tenant_key="tenant_1",
        text="current",
        timestamp=1.0,
    )
    ledger = object.__new__(GroupContextLedger)
    ledger._config = ThreadContextConfig()  # noqa: SLF001
    ledger._blobs = SimpleNamespace(read=lambda _ref: payload.to_bytes())  # noqa: SLF001
    ledger.window = lambda **_kwargs: SimpleNamespace(records=(record,))  # type: ignore[method-assign]
    request = _request(
        requester_principal_id="ou_canonical_owner",
        source_requester_principal_id="ou_user",
    )

    snapshot = ledger.assemble_partial(
        request,
        warning_reason=ContextUnavailableReason.ORDERING,
    )

    assert snapshot.thread_messages[0].sender_id == "ou_user"


def test_group_partial_budget_discards_l2_then_group_before_l1() -> None:
    from src.autonomous.context.group_ledger import (
        GroupContextLedger,
        GroupEventPayload,
    )

    topic_ref = object()
    group_ref = object()
    current_ref = object()
    records = (
        SimpleNamespace(
            message_id="om_before",
            chat_id="oc_1",
            thread_id="omt_1",
            payload_ref=topic_ref,
        ),
        SimpleNamespace(
            message_id="om_group",
            chat_id="oc_1",
            thread_id="",
            payload_ref=group_ref,
        ),
        SimpleNamespace(
            message_id="om_current",
            chat_id="oc_1",
            thread_id="omt_1",
            payload_ref=current_ref,
        ),
    )
    payloads = {
        topic_ref: GroupEventPayload(
            sender_id="ou_topic",
            sender_id_type="open_id",
            sender_type="user",
            sender_tenant_key="tenant_1",
            text="topic",
            timestamp=0.5,
        ).to_bytes(),
        group_ref: GroupEventPayload(
            sender_id="ou_group",
            sender_id_type="open_id",
            sender_type="user",
            sender_tenant_key="tenant_1",
            text="G" * 10,
            timestamp=1.0,
        ).to_bytes(),
        current_ref: GroupEventPayload(
            sender_id="ou_user",
            sender_id_type="open_id",
            sender_type="user",
            sender_tenant_key="tenant_1",
            text="current",
            timestamp=2.0,
        ).to_bytes(),
    }
    ledger = object.__new__(GroupContextLedger)
    ledger._config = ThreadContextConfig(  # noqa: SLF001
        max_context_chars=22,
        max_context_tokens=100,
        tokens_per_char=1.0,
    )
    ledger._blobs = SimpleNamespace(read=payloads.__getitem__)  # noqa: SLF001
    ledger.window = lambda **_kwargs: SimpleNamespace(  # type: ignore[method-assign]
        records=records
    )

    snapshot = ledger.assemble_partial(
        _request(),
        warning_reason=ContextUnavailableReason.ORDERING,
        l1_summary="M" * 10,
        l2_summary="L" * 10,
    )

    assert [item.message_id for item in snapshot.thread_messages] == [
        "om_before",
        "om_current",
    ]
    assert snapshot.group_messages == ()
    assert snapshot.l1_summary == "M" * 10
    assert snapshot.l2_summary == ""
    assert tuple(item.layer for item in snapshot.trimming_trace) == (
        ContextLayer.L2_GROUP,
        ContextLayer.GROUP_RECENT,
    )


def test_group_partial_preserves_topic_beyond_group_window_limit() -> None:
    from src.autonomous.context.group_ledger import (
        GroupContextLedger,
        GroupEventPayload,
    )

    ledger = object.__new__(GroupContextLedger)
    ledger._config = ThreadContextConfig(  # noqa: SLF001
        max_group_messages=2,
        max_context_chars=10_000,
        max_context_tokens=10_000,
    )
    ledger._lock = threading.RLock()  # noqa: SLF001
    payloads: dict[object, bytes] = {}
    records = []

    def add_record(
        message_id: str,
        *,
        sequence: int,
        thread_id: str,
        sender_id: str = "ou_other",
    ) -> None:
        payload_ref = object()
        payloads[payload_ref] = GroupEventPayload(
            sender_id=sender_id,
            sender_id_type="open_id",
            sender_type="user",
            sender_tenant_key="tenant_1",
            text=message_id,
            timestamp=float(sequence),
        ).to_bytes()
        records.append(
            SimpleNamespace(
                tenant_key="tenant_1",
                chat_id="oc_1",
                message_id=message_id,
                thread_id=thread_id,
                journal_sequence=sequence,
                dedup_key=f"dedup-{sequence}",
                causal_event_id="",
                payload_ref=payload_ref,
            )
        )

    add_record("om_group_1", sequence=1, thread_id="")
    add_record("om_root", sequence=2, thread_id="omt_1")
    add_record("om_topic_1", sequence=3, thread_id="omt_1")
    add_record("om_group_2", sequence=4, thread_id="")
    add_record("om_topic_2", sequence=5, thread_id="omt_1")
    add_record("om_group_3", sequence=6, thread_id="")
    add_record(
        "om_current",
        sequence=7,
        thread_id="omt_1",
        sender_id="ou_user",
    )
    ledger._records = {record.dedup_key: record for record in records}  # noqa: SLF001
    ledger._blobs = SimpleNamespace(read=payloads.__getitem__)  # noqa: SLF001

    snapshot = ledger.assemble_partial(
        _request(),
        warning_reason=ContextUnavailableReason.ORDERING,
    )

    assert [item.message_id for item in snapshot.thread_messages] == [
        "om_root",
        "om_topic_1",
        "om_topic_2",
        "om_current",
    ]
    assert [item.message_id for item in snapshot.group_messages] == [
        "om_group_2",
        "om_group_3",
    ]
    assert snapshot.watermark is not None
    assert snapshot.watermark.message_count == 4


def test_context_close_rejects_new_work_and_drains_admitted_assembly() -> None:
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class BlockingMemory(_MemoryFacade):
        def read_l1(self, agent_id, tenant_key):
            entered.set()
            assert release.wait(2)
            return super().read_l1(
                agent_id,
                tenant_key,
            )

    built = _composition(memory=BlockingMemory())
    worker = threading.Thread(target=lambda: built.service.assemble(_request()))
    worker.start()
    assert entered.wait(2)
    closer = threading.Thread(target=lambda: (built.service.close(), closed.set()))
    closer.start()
    assert not closed.wait(0.05)
    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble(_request())
    assert raised.value.reason is ContextUnavailableReason.SOURCE

    release.set()
    assert closed.wait(2)
    worker.join()
    closer.join()


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda state: state.employees.__setitem__(
            "agt_1", replace(state.employees["agt_1"], state=EmployeeState.DRAFT)
        ), ContextUnavailableReason.SCOPE),
        (lambda state: state.employees.__setitem__(
            "agt_1", replace(state.employees["agt_1"], member_groups=())
        ), ContextUnavailableReason.SCOPE),
        (lambda state: state.bot_principals.__setitem__(
            "bot_1", replace(state.bot_principals["bot_1"], credential_ref="")
        ), ContextUnavailableReason.CREDENTIALS),
    ],
)
def test_projected_authority_failure_prevents_all_external_reads(
    mutate,
    reason: ContextUnavailableReason,
) -> None:
    state = _state()
    mutate(state)
    built = _composition(state=state)

    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble(_request())

    assert raised.value.reason is reason
    assert built.memory.calls == []
    assert built.backend.calls == []
    assert built.source_factory.calls == []


@pytest.mark.parametrize(
    ("generation", "acl", "reason"),
    [
        (_BooleanAuthority([False]), _Acl(), ContextUnavailableReason.SCOPE),
        (_BooleanAuthority(), _Acl(False), ContextUnavailableReason.PERMISSION),
    ],
)
def test_request_authority_failure_prevents_all_external_reads(
    generation: _BooleanAuthority,
    acl: _Acl,
    reason: ContextUnavailableReason,
) -> None:
    built = _composition(generation=generation, acl=acl)

    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble(_request())

    assert raised.value.reason is reason
    assert built.memory.calls == []
    assert built.backend.calls == []
    assert built.source_factory.calls == []


def test_authority_change_after_snapshot_closes_source_and_never_delegates() -> None:
    generation = _BooleanAuthority([True, True, False])
    built = _composition(generation=generation)
    delegate = _Delegate()
    port = ContextPreparingExecutionPort(
        context_service=built.service,
        authority_fence=_Fence(),
        delegate=delegate,
    )

    with pytest.raises(ContextUnavailableError) as raised:
        port.execute(_request(), tool="codex", model="gpt", effort="high")

    assert raised.value.reason is ContextUnavailableReason.SCOPE
    assert built.source_factory.close_calls == 1
    assert delegate.calls == []


def test_projection_head_mismatch_fails_before_memory_or_source_reads() -> None:
    built = _composition()
    built.data.service.head = JournalHead(1, "b" * 64)

    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble(_request())

    assert raised.value.reason is ContextUnavailableReason.MEMORY
    assert built.memory.calls == []
    assert built.backend.calls == []
    assert built.source_factory.calls == []


def test_projection_head_catchup_retries_before_external_reads() -> None:
    state = _state()
    state.cursor_sequence = 1
    state.cursor_hash = "a" * 64
    built = _composition(state=state)
    heads = iter(
        (
            JournalHead(),
            JournalHead(1, "a" * 64),
            JournalHead(1, "a" * 64),
            JournalHead(1, "a" * 64),
        )
    )
    built.data.service.get_head = lambda: next(heads)

    snapshot = built.service.assemble(_request())

    assert snapshot.l1_summary == "L1"
    assert built.memory.calls == [("agt_1", "tenant_1")]
    assert built.backend.calls == ["oc_1"]
    assert len(built.source_factory.calls) == 1


def test_semantically_unchanged_projection_advance_retries_snapshot() -> None:
    state = _state()
    state.cursor_sequence = 1
    state.cursor_hash = "a" * 64
    source_factory = _SourceFactory(assembly_attempts=2)
    built = _composition(state=state, source_factory=source_factory)
    original_read = built.memory.read_l1
    advanced = False

    def read_and_advance(
        agent_id: str,
        tenant_key: str,
    ) -> str | None:
        nonlocal advanced
        content = original_read(
            agent_id,
            tenant_key,
        )
        if not advanced:
            advanced = True
            state.cursor_sequence = 2
            state.cursor_hash = "b" * 64
            built.data.service.head = JournalHead(2, "b" * 64)
        return content

    built.memory.read_l1 = read_and_advance

    snapshot = built.service.assemble(_request())

    assert snapshot.l1_summary == "L1"
    assert built.memory.calls == [
        ("agt_1", "tenant_1"),
        ("agt_1", "tenant_1"),
    ]
    assert built.backend.calls == ["oc_1", "oc_1"]
    assert len(built.source_factory.calls) == 1


def test_projection_retry_does_not_accept_membership_revocation() -> None:
    state = _state()
    state.cursor_sequence = 1
    state.cursor_hash = "a" * 64
    built = _composition(state=state)
    original_read = built.memory.read_l1

    def read_and_revoke(
        agent_id: str,
        tenant_key: str,
    ) -> str | None:
        content = original_read(
            agent_id,
            tenant_key,
        )
        state.employees["agt_1"] = replace(
            state.employees["agt_1"],
            member_groups=(),
        )
        state.cursor_sequence = 2
        state.cursor_hash = "b" * 64
        built.data.service.head = JournalHead(2, "b" * 64)
        return content

    built.memory.read_l1 = read_and_revoke

    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble(_request())

    assert raised.value.reason is ContextUnavailableReason.SCOPE
    assert built.memory.calls == [("agt_1", "tenant_1")]
    assert built.backend.calls == []
    assert built.source_factory.calls == []


def test_partial_context_is_counted_for_drain_and_rejected_after_stop() -> None:
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    snapshot = object()

    class BlockingLedger:
        def assemble_partial(self, *_args, **_kwargs):
            entered.set()
            assert release.wait(2)
            return snapshot

    built = _composition(group_ledger=BlockingLedger())
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            built.service.assemble_canonical_partial(
                _request(),
                warning_reason=ContextUnavailableReason.REVISION,
            )
        )
    )
    worker.start()
    assert entered.wait(2)
    closer = threading.Thread(
        target=lambda: (built.service.close(), closed.set())
    )
    closer.start()
    closed_early = closed.wait(0.05)
    release.set()
    assert closed.wait(2)
    worker.join()
    closer.join()
    assert not closed_early
    assert results == [snapshot]

    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble_canonical_partial(
            _request(),
            warning_reason=ContextUnavailableReason.REVISION,
        )
    assert raised.value.reason is ContextUnavailableReason.SOURCE


def test_partial_context_rechecks_membership_after_ledger_read() -> None:
    state = _state()
    state.cursor_sequence = 1
    state.cursor_hash = "a" * 64

    class RevokingLedger:
        def assemble_partial(self, *_args, **_kwargs):
            state.employees["agt_1"] = replace(
                state.employees["agt_1"],
                member_groups=(),
            )
            state.cursor_sequence = 2
            state.cursor_hash = "b" * 64
            built.data.service.head = JournalHead(2, "b" * 64)
            return object()

    built = _composition(state=state, group_ledger=RevokingLedger())

    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble_canonical_partial(
            _request(),
            warning_reason=ContextUnavailableReason.REVISION,
        )

    assert raised.value.reason is ContextUnavailableReason.SCOPE


def test_partial_context_rejects_projection_advance_during_ledger_read() -> None:
    state = _state()
    state.cursor_sequence = 1
    state.cursor_hash = "a" * 64

    class AdvancingLedger:
        def assemble_partial(self, *_args, **_kwargs):
            state.cursor_sequence = 2
            state.cursor_hash = "b" * 64
            built.data.service.head = JournalHead(2, "b" * 64)
            return object()

    built = _composition(state=state, group_ledger=AdvancingLedger())

    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble_canonical_partial(
            _request(),
            warning_reason=ContextUnavailableReason.REVISION,
        )

    assert raised.value.reason is ContextUnavailableReason.REVISION
