"""Tests that sender_id / is_p2p / thread_id (now backed by contextvars.ContextVar)
propagate correctly through contextvars.copy_context().run() and do NOT leak
across tasks in a thread-pool scenario.
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor

from src.thread import (
    get_current_is_p2p,
    get_current_sender_id,
    get_current_sender_union_id,
    get_current_tenant_key,
    get_current_thread_id,
    set_current_is_p2p,
    set_current_sender_id,
    set_current_sender_union_id,
    set_current_tenant_key,
    set_current_thread_id,
)
from src.utils.thread_pools import submit_io


class TestContextVarPropagation:
    """Verify copy_context().run() propagates ContextVar values to child threads."""

    def test_contextvar_auto_propagation_via_copy_context(self) -> None:
        """Parent sets values → copy_context() → run() in worker sees them."""
        set_current_sender_id("user_abc")
        set_current_sender_union_id("union_abc")
        set_current_is_p2p(True)
        set_current_tenant_key("tenant_a")
        set_current_thread_id("thread_xyz")

        ctx = contextvars.copy_context()
        results: dict[str, object] = {}

        def _worker() -> None:
            results["sender_id"] = get_current_sender_id()
            results["sender_union_id"] = get_current_sender_union_id()
            results["is_p2p"] = get_current_is_p2p()
            results["tenant_key"] = get_current_tenant_key()
            results["thread_id"] = get_current_thread_id()

        submit_io(ctx.run, _worker).result(timeout=5)

        assert results["sender_id"] == "user_abc"
        assert results["sender_union_id"] == "union_abc"
        assert results["is_p2p"] is True
        assert results["tenant_key"] == "tenant_a"
        assert results["thread_id"] == "thread_xyz"

        # Cleanup parent context
        set_current_sender_id(None)
        set_current_sender_union_id(None)
        set_current_is_p2p(False)
        set_current_tenant_key(None)
        set_current_thread_id(None)

    def test_contextvar_isolation_between_tasks(self) -> None:
        """Two tasks on the same worker thread (max_workers=1): task A sets
        sender_id='A' without cleanup, task B uses a fresh context and must
        NOT see 'A'.  This is the exact bug that threading.local would expose.
        """
        observed_in_b: dict[str, object] = {}

        def _task_a() -> None:
            set_current_sender_id("A")
            set_current_is_p2p(True)
            set_current_tenant_key("tenant_a")
            # Intentionally NO cleanup — simulates a missed finally block

        def _task_b() -> None:
            observed_in_b["sender_id"] = get_current_sender_id()
            observed_in_b["is_p2p"] = get_current_is_p2p()
            observed_in_b["tenant_key"] = get_current_tenant_key()

        # Capture two separate contexts (like TaskScheduler.submit does)
        ctx_a = contextvars.copy_context()
        ctx_b = contextvars.copy_context()

        with ThreadPoolExecutor(max_workers=1) as pool:
            # Force sequential on the SAME thread
            pool.submit(ctx_a.run, _task_a).result()
            pool.submit(ctx_b.run, _task_b).result()

        # task_b must see defaults, not task_a's leaked values
        assert observed_in_b["sender_id"] is None
        assert observed_in_b["is_p2p"] is False
        assert observed_in_b["tenant_key"] is None

    def test_contextvar_no_leak_after_context_run(self) -> None:
        """After context.run() completes, the worker thread's own ContextVar
        state must not retain values that were set inside the run() call.
        """
        leaked: dict[str, object] = {}

        def _inner() -> None:
            set_current_sender_id("inside_run")
            set_current_thread_id("tid_inside")
            set_current_is_p2p(True)

        def _check_after() -> None:
            leaked["sender_id"] = get_current_sender_id()
            leaked["thread_id"] = get_current_thread_id()
            leaked["is_p2p"] = get_current_is_p2p()

        ctx = contextvars.copy_context()

        with ThreadPoolExecutor(max_workers=1) as pool:
            # Run _inner inside a context snapshot
            pool.submit(ctx.run, _inner).result()
            # Now run _check_after WITHOUT context.run — raw thread state
            pool.submit(_check_after).result()

        assert leaked["sender_id"] is None
        assert leaked["thread_id"] is None
        assert leaked["is_p2p"] is False


class TestCardActionContextVarBackfill:
    """Verify the card-action backfill pattern: prefer task_ctx.spec.sender_id,
    fall back to event operator — matching _process_card_action_async logic.
    """

    @staticmethod
    def _backfill_sender_id(spec_sender_id: str, event_operator_id: str) -> None:
        """Replicate the backfill logic from _process_card_action_async."""
        _operator_id = (
            spec_sender_id
            if spec_sender_id
            else (event_operator_id or "")
        )
        set_current_sender_id(_operator_id)

    def test_card_action_sender_id_backfill_from_task_spec(self) -> None:
        """When TaskSpec carries sender_id, ContextVar is populated from spec
        even though copy_context() captured an empty snapshot."""
        # Parent context has no sender_id set (simulates scheduler submit thread)
        set_current_sender_id(None)
        ctx = contextvars.copy_context()

        results: dict[str, object] = {}

        def _simulate_card_action() -> None:
            # Replicate _process_card_action_async backfill: spec has sender_id
            self._backfill_sender_id(
                spec_sender_id="op_123",
                event_operator_id="event_fallback_456",
            )
            results["sender_id"] = get_current_sender_id()

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(ctx.run, _simulate_card_action).result()

        assert results["sender_id"] == "op_123"

        # Cleanup
        set_current_sender_id(None)

    def test_card_action_sender_id_fallback_to_event(self) -> None:
        """When TaskSpec.sender_id is empty, fall back to event operator."""
        set_current_sender_id(None)
        ctx = contextvars.copy_context()

        results: dict[str, object] = {}

        def _simulate_card_action() -> None:
            self._backfill_sender_id(
                spec_sender_id="",
                event_operator_id="event_op_789",
            )
            results["sender_id"] = get_current_sender_id()

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(ctx.run, _simulate_card_action).result()

        assert results["sender_id"] == "event_op_789"

        # Cleanup
        set_current_sender_id(None)
