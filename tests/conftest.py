"""Global test fixtures."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from src.card.delivery.engine import CardDelivery
from src.card.delivery.registry import DeliveryRegistry
from src.card.session import CardSession
from src.card.state.models import CardMetadata
from src.utils.retry import RetryPolicy


@pytest.fixture(autouse=True)
def _isolate_delivery_registry(monkeypatch, request):
    """Give each test an isolated registry and close it through production APIs."""
    real_monotonic = time.monotonic
    registry = DeliveryRegistry()
    monkeypatch.setattr(registry, "install_atexit", lambda: None)
    monkeypatch.setattr("src.card.delivery.registry.delivery_registry", registry)
    monkeypatch.setattr("src.card.delivery.engine.delivery_registry", registry)
    if hasattr(request.module, "delivery_registry"):
        monkeypatch.setattr(request.module, "delivery_registry", registry)

    yield registry

    with monkeypatch.context() as teardown_patch:
        teardown_patch.setattr(time, "monotonic", real_monotonic)
        try:
            registry.drain_in_flight(timeout=5.0)
        finally:
            registry.shutdown_all()


@pytest.fixture(autouse=True)
def _force_sync_delivery(monkeypatch):
    """Force all CardSession instances to use synchronous delivery in tests.

    The production code uses a thread pool for async delivery, but tests need
    deterministic behavior. We patch __init__ to override _sync_delivery=True
    regardless of what SessionConfig.sync_delivery was set to.
    """
    import inspect

    # Defensive assertion: ensure CardSession.__init__ still sets _sync_delivery.
    # If the constructor signature changes, this will fail loudly instead of
    # silently leaving tests with async delivery (causing flaky failures).
    _init_src = inspect.getsource(CardSession.__init__)
    assert "_sync_delivery" in _init_src, (
        "CardSession.__init__ no longer sets _sync_delivery. "
        "Update this fixture to match the new constructor pattern."
    )

    _orig_init = CardSession.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # Override: always sync in tests for determinism
        self._sync_delivery = True

    monkeypatch.setattr(CardSession, "__init__", _patched_init)
    yield


# ---------------------------------------------------------------------------
# Shared TrackingClient for adapter sequence integration tests.
# ---------------------------------------------------------------------------

class TrackingClient:
    """Mock Feishu API client that tracks all card operations.

    Uses dynamic IDs based on creation count.
    """

    def __init__(self):
        self.created: list[dict] = []
        self.updated: list[dict] = []

    def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        self.created.append(card_json)
        idx = len(self.created)
        return (f"msg_{idx}", f"card_{idx}")

    def update_card(self, card_id, card_json, *, sequence=0):
        self.updated.append(card_json)


@pytest.fixture
def tracking_client() -> TrackingClient:
    """Provide a fresh TrackingClient instance."""
    return TrackingClient()


@pytest.fixture
def make_card_delivery():
    """Factory fixture: create a CardDelivery from a TrackingClient.

    Teardown calls shutdown() on all created deliveries to stop background threads.
    """
    deliveries: list[CardDelivery] = []

    def _factory(client: TrackingClient | None = None) -> tuple[CardDelivery, TrackingClient]:
        if client is None:
            client = TrackingClient()
        d = CardDelivery(client)
        deliveries.append(d)
        return d, client

    yield _factory

    for d in deliveries:
        try:
            d._shutdown()
        except Exception:
            pass


@pytest.fixture
def make_card_session():
    """Factory fixture: create a CardSession wired to a delivery.

    Teardown calls close() on all created sessions and shutdown() on deliveries.
    """
    sessions: list[CardSession] = []
    deliveries: list[CardDelivery] = []

    def _factory(
        *,
        chat_id: str = "chat_test",
        engine_type: str = "deep",
        mode_name: str = "Test Agent",
        delivery: CardDelivery | None = None,
        client: TrackingClient | None = None,
        session_id: str | None = None,
    ) -> tuple[CardSession, TrackingClient]:
        if client is None:
            client = TrackingClient()
        if delivery is None:
            delivery = CardDelivery(client)
            deliveries.append(delivery)
        metadata = CardMetadata(engine_type=engine_type, mode_name=mode_name)
        from src.card.session.config import SessionCallbacks, SessionConfig
        config = SessionConfig(metadata=metadata, sync_delivery=True)
        session = CardSession(
            chat_id=chat_id,
            config=config,
            delivery=delivery,
            session_id=session_id,
            callbacks=SessionCallbacks(notify_callback=lambda _cid, _txt: None),
        )
        sessions.append(session)
        return session, client

    yield _factory

    for s in sessions:
        try:
            s.close()
        except Exception:
            pass
    for d in deliveries:
        try:
            d._shutdown()
        except Exception:
            pass


@pytest.fixture
def make_session_rotator(make_card_session):
    """Factory fixture: create a SessionRotator with auto-cleanup.

    Args (keyword only):
        factory: Optional callable returning a CardSession. Defaults to
                 a factory that creates sessions via make_card_session.
        engine_type: Engine type for the initial session (default "deep").
        session_id: Optional custom initial session ID.

    Returns:
        Tuple of (SessionRotator, initial_CardSession, TrackingClient).
    """
    from src.card.session.rotator import SessionRotator

    rotators: list[SessionRotator] = []

    def _factory(
        *,
        factory: object = None,
        engine_type: str = "deep",
        session_id: str | None = None,
    ):
        initial_session, client = make_card_session(
            engine_type=engine_type,
            session_id=session_id,
        )
        rotator = SessionRotator(initial_session)
        rotators.append(rotator)
        return rotator, initial_session, client

    yield _factory

    for r in rotators:
        try:
            r.close()
        except Exception:
            pass




# ---------------------------------------------------------------------------
# Common pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cancel_event():
    return threading.Event()


@pytest.fixture
def fast_retry_policy():
    return RetryPolicy(max_retries=2, retry_delay=0.01, jitter_factor=0)


@pytest.fixture
def make_settings():
    """Factory fixture returning a MagicMock settings object with sane defaults.

    Usage::

        def test_something(make_settings):
            settings = make_settings(spec_execution_timeout=60)
    """
    # Sensible defaults covering every engine subsystem
    _DEFAULTS = dict(
        # ACP / agent session
        acp_startup_timeout=20,
        acp_auto_update_timeout=120,
        acp_healthcheck_timeout=2.0,
        acp_auto_update=True,
        rate_limit_retry_enabled=False,
        # Deep engine
        coco_execution_timeout=300,
        claude_execution_timeout=600,
        # Spec engine
        spec_execution_timeout=300,
        spec_persist_every_phase=False,
        spec_review_enabled=False,
        spec_discovery_enabled=False,
        spec_cycle_tasks_max=5,
        spec_max_cycles_limit=10,
        spec_max_retries=2,
        spec_persist_phase_artifacts=False,
        # Engine shared
        engine_eval_prompt_timeout=60,
        engine_timeout_warning_threshold=600,
        # Streaming
        streaming_adaptive_interval_base=0.5,
        streaming_adaptive_interval_max=3.0,
        streaming_adaptive_rate_low=10.0,
        streaming_adaptive_rate_high=100.0,
        # Card session / lock
        lock_undo_window_seconds=300,
    )

    def _factory(**overrides):
        merged = {**_DEFAULTS, **overrides}
        s = MagicMock()
        for k, v in merged.items():
            setattr(s, k, v)
        # Provide nested card config with sane defaults (avoids MagicMock in math ops)
        if not hasattr(s, "card") or isinstance(s.card, MagicMock):
            card_mock = MagicMock()
            card_mock.session_idle_timeout = 1800
            card_mock.session_idle_warn_at_remaining = 300
            card_mock.session_lock_max = 200
            card_mock.session_lock_ttl = 7200
            card_mock.session_max_rotations = 10
            card_mock.max_chars = 28000
            card_mock.button_size = "default"
            card_mock.button_layout = "responsive"
            card_mock.mobile_force_vertical = False
            card_mock.continuation_enabled = True
            card_mock.action_dedup_ttl = 5
            card_mock.action_dedup_max_size = 200
            s.card = card_mock
        return s

    return _factory


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------






@pytest.fixture(autouse=True)
def _block_real_employee_storage(monkeypatch, tmp_path):
    """Prevent tests from accidentally writing to the legacy default employee store."""

    def _guarded_default():
        raise RuntimeError(
            "[conftest] Real default_employee_storage_base() called in tests! "
            "Pass an explicit storage_base_path=str(tmp_path / \"employee\") to "
            "ProjectedAgentRegistry/identity-backed tests."
        )
    try:
        monkeypatch.setattr(
            "src.autonomous.workforce.identity.default_employee_storage_base",
            _guarded_default,
        )
    except Exception:
        pass  # Module not yet imported — safe to skip


# ---------------------------------------------------------------------------
# lru_cache hygiene — clear module-level caches between tests.
# MAINTAIN: when adding a new @lru_cache in src/, register its cache_clear here.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_hook_executor_between_tests():
    """Reset the global hook executor to prevent cross-test state leakage."""
    yield
    from src.card.hooks import _reset_hook_executor
    _reset_hook_executor()


@pytest.fixture
def failing_handler_session():
    """Configurable fixture for simulating handler method failures.

    Returns a factory that patches handler methods to return failure values.

    Usage::

        def test_card_failure(failing_handler_session):
            handler = make_handler()
            with failing_handler_session(handler, reply_card=None, update_card=False):
                # handler.reply_card(...) returns None
                # handler.update_card(...) returns False
                ...
    """
    from contextlib import contextmanager

    @contextmanager
    def _configure(handler, *, reply_card=None, update_card=False):
        original_reply_card = handler.reply_card
        original_update_card = handler.update_card

        handler.reply_card = MagicMock(return_value=reply_card)
        handler.update_card = MagicMock(return_value=update_card)
        try:
            yield handler
        finally:
            handler.reply_card = original_reply_card
            handler.update_card = original_update_card

    return _configure


@pytest.fixture(autouse=True)
def _clear_all_lru_caches():
    """Clear all known module-level lru_caches before and after each test.

    Some tests (like test_card_buttons.py::test_no_runtime_warning_on_import)
    delete modules from sys.modules, which can cause state leakage between tests.
    We need to clear caches both before and after each test to ensure isolation.
    """
    _clear_caches()
    yield
    _clear_caches()


def _clear_caches():
    """Helper function to clear all caches.

    This is called both before and after each test to ensure proper isolation,
    especially when tests delete modules from sys.modules.
    """
    import sys

    # sync_adapter caches
    try:
        from src.acp.sync_adapter import _probe_acp_serve_help, _supports_acp_serve
        _probe_acp_serve_help.cache_clear()
        _supports_acp_serve.cache_clear()
    except Exception:
        pass
    # diagnostics caches
    try:
        from src.acp.diagnostics import _compile_redaction_patterns
        _compile_redaction_patterns.cache_clear()
    except Exception:
        pass
    # renderer signature cache
    try:
        from src.card.render.renderer import _compute_sig_cached
        _compute_sig_cached.cache_clear()
    except Exception:
        pass
    atoms_mod = sys.modules.get("src.card.render.atoms")
    if atoms_mod is not None:
        atoms_mod._block_kind_handlers = None


# ---------------------------------------------------------------------------
# Singleton hygiene — reset global singletons so tests start from clean state.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_all_singletons():
    """Reset all known global singletons after each test."""
    yield
    try:
        from src.coco_model.manager import _reset_coco_model_manager_for_testing
        _reset_coco_model_manager_for_testing()
    except Exception:
        pass
    try:
        from src.thread import set_current_tenant_key, set_current_thread_id
        from src.thread.manager import _reset_thread_manager_for_testing

        _reset_thread_manager_for_testing()
        # The manager singleton and the request-scoped ContextVar are separate.
        # Engine handler tests may bind a synthetic thread id; clear it so the
        # next test cannot serialize a stale MagicMock into an unrelated card.
        set_current_thread_id(None)
        set_current_tenant_key(None)
    except Exception:
        pass
    try:
        from src.acp.providers import _reset_providers_for_testing
        _reset_providers_for_testing()
    except Exception:
        pass
    try:
        from src.utils.env import _reset_env_for_testing
        _reset_env_for_testing()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Workflow engine resource fixtures
# ---------------------------------------------------------------------------

import tempfile


@pytest.fixture
def make_wf_bridge():
    """Factory fixture: create a RuntimeBridge with auto-teardown.

    Teardown calls .stop() on all created bridges to clean up subprocesses
    and background threads.

    Usage::

        def test_something(make_wf_bridge):
            bridge = make_wf_bridge(tmp_path=tmp_path, on_agent_call=my_handler)
    """
    from unittest.mock import MagicMock

    bridges: list = []

    def _factory(tmp_path=None, **kwargs):
        # Lazy import to avoid circular imports / startup overhead
        from src.workflow_engine.bridge import RuntimeBridge

        if tmp_path is None:
            tmp_path = tempfile.mkdtemp(prefix="wf_bridge_")

        kwargs.setdefault("script_path", "test_workflow.js")
        kwargs.setdefault(
            "on_agent_call",
            lambda params: MagicMock(output="ok"),
        )

        bridge = RuntimeBridge(
            cwd=str(tmp_path),
            **kwargs,
        )
        bridges.append(bridge)
        return bridge

    yield _factory

    for b in bridges:
        try:
            b.stop()
        except Exception:
            pass


@pytest.fixture
def make_wf_executor():
    """Factory fixture: create an AgentExecutor with auto-teardown.

    Teardown calls .shutdown(wait=True) on all created executors to stop
    their thread pools.

    Usage::

        def test_something(make_wf_executor):
            executor = make_wf_executor(tmp_path=tmp_path, cancel_event=evt)
    """
    executors: list = []

    def _factory(tmp_path=None, cancel_event=None, **kwargs):
        # Lazy import to avoid circular imports / startup overhead
        from src.workflow_engine.executor import AgentExecutor

        if tmp_path is None:
            tmp_path = tempfile.mkdtemp(prefix="wf_executor_")
        if cancel_event is None:
            cancel_event = threading.Event()

        kwargs.setdefault("max_workers", 2)

        executor = AgentExecutor(
            cwd=str(tmp_path),
            cancel_event=cancel_event,
            **kwargs,
        )
        executors.append(executor)
        return executor

    yield _factory

    for e in executors:
        try:
            e.shutdown(wait=True)
        except Exception:
            pass


@pytest.fixture
def make_wf_coalescer():
    """Factory fixture: create a ProgressCoalescer with auto-teardown.

    Teardown calls .stop() on all created coalescers to stop daemon threads
    and flush pending updates.

    Usage::

        def test_something(make_wf_coalescer):
            coalescer = make_wf_coalescer(on_progress=my_callback, debounce_s=0.01)
    """
    coalescers: list = []

    def _factory(on_progress=None, debounce_s=0.01, **kwargs):
        # Lazy import to avoid circular imports / startup overhead
        from src.workflow_engine.progress_coalescer import ProgressCoalescer

        if on_progress is None:
            on_progress = lambda snapshot: None  # noqa: E731

        coalescer = ProgressCoalescer(
            on_progress=on_progress,
            debounce_s=debounce_s,
            **kwargs,
        )
        coalescers.append(coalescer)
        return coalescer

    yield _factory

    for c in coalescers:
        try:
            c.stop()
        except Exception:
            pass


@pytest.fixture(scope="session", autouse=True)
def _detect_resource_leaks():
    """Session-scoped diagnostic fixture: detect thread/process leaks.

    Records active threads at session start and warns at session end if
    more than 5 extra threads remain. Also checks for leftover ``node``
    subprocesses.

    This is a diagnostic only — it never fails a test.
    """
    import sys

    start_count = threading.active_count()
    start_threads = {t.name for t in threading.enumerate()}

    yield

    # Give daemon threads a moment to wind down
    import time
    time.sleep(0.5)

    end_count = threading.active_count()
    end_threads = {t.name for t in threading.enumerate()}
    new_threads = end_threads - start_threads
    extra = end_count - start_count

    if extra > 5:
        print(
            f"\n[conftest] WARNING: {extra} extra threads still active at session end "
            f"(started with {start_count}, ended with {end_count}).",
            file=sys.stderr,
        )
        print(f"[conftest] New thread names: {sorted(new_threads)}", file=sys.stderr)

    # Check for leftover node subprocesses
    try:
        import psutil  # type: ignore

        node_procs = [
            p for p in psutil.process_iter(["name", "pid"])
            if p.info["name"] and "node" in p.info["name"].lower()
        ]
        if node_procs:
            print(
                f"\n[conftest] WARNING: {len(node_procs)} node process(es) still running: "
                + ", ".join(f"pid={p.info['pid']}" for p in node_procs),
                file=sys.stderr,
            )
    except ImportError:
        # Fallback: use ps aux
        try:
            import subprocess as _sp
            output = _sp.check_output(["ps", "aux"], text=True)
            node_lines = [
                line for line in output.splitlines()
                if "node" in line and "ps aux" not in line and "grep" not in line
            ]
            # Filter out entries that are clearly not node processes (e.g. "nodemon" etc is fine)
            if node_lines:
                # Only warn if there are more than a baseline — be conservative
                print(
                    f"\n[conftest] NOTE: {len(node_lines)} node-related process(es) found via ps.",
                    file=sys.stderr,
                )
        except Exception:
            pass
