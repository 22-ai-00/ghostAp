# GhostAP Performance Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement both audited performance batches without weakening GhostAP's fail-closed trust, Journal durability, card revision, task ordering, or workflow cancellation contracts.

**Architecture:** Keep public APIs stable where practical. Bound admission and historical indexes at ingress; coalesce immutable card snapshots before expensive rendering; scope workflow cancellation to the current async lineage; make raw ACP capture opt-in; reuse authenticated Journal/trust snapshots only while their complete disk identity is unchanged; and introduce bounded concurrency only after durable effect anchoring.

**Tech Stack:** Python 3.13, `uv`, pytest, Pydantic settings, Node.js ESM/AsyncLocalStorage, lark SDK, fsync/flock Journal storage.

## Global constraints

- Preserve all pre-existing dirty worktree changes, especially Workflow model-selection/cache changes.
- Write a failing regression test before each production change and observe the expected failure.
- Never weaken permission, trust, provenance, revision, anchor-before-effect, or terminal-finalization checks.
- Use only `uv`; run the narrow test file first, then the subsystem suite.
- Do not update `.Memory/` from individual tasks. The integrator appends the final evidence once.

---

## Task 1: Bound TaskScheduler admission and reclaim idle indexes

**Files:**

- Modify: `src/tasking/scheduler.py`
- Modify: `src/tasking/__init__.py`
- Modify: `src/config/settings.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `.env.example`
- Test: `tests/test_task_scheduler.py`
- Test: `tests/test_config_validation.py`

- [x] Add RED tests for independent normal/system pending capacity, rejection without partial state, terminal-history cap, idle-key cleanup, and duplicate task-id safety.
- [x] Run `uv run python -m pytest tests/test_task_scheduler.py tests/test_config_validation.py -q` and confirm only the new contract fails.
- [x] Add `TaskQueueFullError(RateLimitExceededException)` and atomic per-lane admission counters.
- [x] Delete empty queue/running/chat/project index keys and reap terminal history during sustained activity.
- [x] Wire positive settings (`1000` normal, `100` system, `5000` terminal history) through `_build_task_scheduler()` and `.env.example`.
- [x] Re-run focused tests plus `tests/test_ws_client_reconnect.py`; run Ruff on touched files.

Safety: System retains reserved capacity; queued/running tasks are never reaped; project serialization and restart fencing remain unchanged.

## Task 2: Close card resources and coalesce before rendering

**Files:**

- Modify: `src/card/delivery/registry.py`
- Modify: `src/card/delivery/engine.py`
- Modify: `src/feishu/handlers/programming.py`
- Modify: `src/card/session/core.py`
- Modify: `src/card/timers/scheduler.py`
- Test: `tests/test_programming_response.py`
- Test: `tests/test_card_delivery_engine.py`
- Test: `tests/test_card_session.py`
- Test: `tests/test_timer_scheduler.py`
- Test: `tests/test_shutdown_multi_session.py`

- [x] Add RED lifecycle tests proving every owned programming delivery drains and shuts down exactly once on success, fallback, and exception.
- [x] Add RED async-session test proving a blocked first delivery renders only the latest pending terminal snapshot after release.
- [x] Add RED timer tests proving slow callbacks cannot stall the timer thread and shutdown cancels callbacks that have not begun.
- [x] Add public idempotent `CardDelivery.shutdown(timeout) -> bool`; make registry references weak and unregister only after a successful drain.
- [x] Ensure programming-owned delivery cleanup happens in all exit paths without shutting down shared deliveries.
- [x] Queue immutable snapshots, coalesce newest-wins before render, and protect terminal snapshots from later non-terminal events.
- [x] Move timer callbacks to a bounded callback executor while the timer thread only dequeues due work.
- [x] Run focused tests and the shutdown/card session regression group; run Ruff.

Safety: Keep `handler -> session -> render/delivery` import direction, card revision fences, terminal delivery, pagination, and observable callback errors.

## Task 3: Remove remote I/O from card ACK and redundant project fsync

**Files:**

- Modify: `src/feishu/ws_client.py`
- Modify: `src/project/manager.py`
- Test: `tests/test_ws_client_routing.py`
- Test: `tests/test_button_gate_and_dedupe.py`
- Test: `tests/test_project_isolation.py`

- [x] Add RED tests showing stale-revision refresh and system-gate notice occur only after typed ACK returns.
- [x] Add RED test proving follow-up admission failure stays fail-closed and never falls back to synchronous remote I/O.
- [x] Add RED tests proving identical active-project refresh avoids snapshot rewrite while missing membership still repairs and persists.
- [x] Enqueue refresh/notice as bounded system follow-ups and immediately ACK rejected business actions.
- [x] Skip `_save_projects()` only for already ACTIVE, already-authorized, identical bindings; update in-memory recency only.
- [x] Run focused routing/isolation tests and Ruff.

Safety: Queue pressure may drop only advisory refresh/notice work, never execute a stale/gated action. Any authorization-set change remains durable.

## Task 4: Scope Workflow request cancellation lexically

**Files:**

- Modify: `src/workflow_engine/runtime/runtime.js`
- Test: `tests/test_workflow_bridge_transport.py`
- Test: `tests/test_workflow_runtime_primitives.py`

- [x] Add RED tests for two parallel sibling races and a nested pipeline/race with an unrelated sibling.
- [x] Replace the global request-interceptor stack with an `AsyncLocalStorage` collector lineage inherited only by descendants.
- [x] Keep race dispatch fencing, request timeout, global cancellation, abort payloads, pending-request cleanup, and agent limits unchanged.
- [x] Run both Node-backed test files and verify none of the relevant tests are skipped.

## Task 5: Make raw ACP tool capture opt-in and share one prompt deadline

**Files:**

- Modify: `src/acp/client.py`
- Modify: `src/acp/session.py`
- Modify: `src/acp/sync_adapter.py`
- Modify: `src/agent_session/factory.py`
- Modify: `src/agent_session/model_diagnostics.py`
- Modify: `src/agent_session/wrappers.py`
- Modify: `src/workflow_engine/executor.py`
- Test: `tests/test_acp_client.py`
- Test: `tests/test_workflow_acp_real_stream.py`
- Test: `tests/test_workflow_wrapper_chain_contract.py`
- Test: `tests/test_workflow_executor_cancel.py`
- Test: `tests/test_rate_limit.py`
- Test: `tests/test_model_failure_failover.py`

- [x] Add RED tests proving default `full_content is None`, explicit Workflow capture preserves the payload, and replacement sessions preserve the setting.
- [x] Thread `capture_full_tool_content: bool = False` through session creation; enable it only in `AgentExecutor`.
- [ ] Add RED fake-clock tests proving rate-limit retry, compaction/failover startup, and replayed prompts consume one monotonic deadline.
- [ ] Compute the deadline once at the outer wrapper, pass decreasing remaining time, and raise `TimeoutError` before a retry when exhausted.
- [ ] Run the ACP/Workflow/fault-tolerance focused suites and Ruff.

Safety: Raw data remains absent by default and never gains a new log/persistence path. Cancellation outranks timeout; timeout is not classified as a recoverable model error.

## Task 6: Incrementalize Journal, projection clones, and workspace rebuild

**Files:**

- Modify: `src/autonomous/journal/writer.py`
- Modify: `src/autonomous/data/projection.py`
- Modify: `src/autonomous/gateway/projection.py`
- Modify: `src/autonomous/ingress/projection.py`
- Modify: `src/autonomous/ingress/router.py`
- Modify: `src/autonomous/membership/projection.py`
- Modify: `src/autonomous/workspace/layout.py`
- Modify: `src/autonomous/workspace/projector.py`
- Modify: `src/autonomous/data/composition.py`
- Test: `tests/autonomous/unit/test_journal_writer.py`
- Add: `tests/autonomous/unit/test_projection_clone_cost.py`
- Test: `tests/autonomous/security/test_employee_workspace_projection.py`
- Test: `tests/autonomous/integration/test_employee_data_composition.py`

- [ ] Add RED tests for zero-decode replay on unchanged disk, forced full verification after identity change, and always-full `verify_chain()`.
- [ ] Cache authenticated frames behind `(dev, ino, size, mtime_ns, ctime_ns)` and slice by continuous sequence; any disk-identity change forces complete prefix authentication.
- [ ] Replace `deepcopy` with explicit mutable-container copies while sharing only frozen record values.
- [ ] Add RED batch-source and unchanged-write tests; rebuild workspace sources from one Journal replay and skip atomic replace/fsync only when content, type, and modes already match.
- [ ] Run focused unit/security/integration/chaos tests and Ruff.

Safety: HMAC/hash/blob verification and commit fsync/anchor order are untouched. Any stat uncertainty or disk mutation fails closed into full validation.

## Task 7: Add bounded anchored Autonomous dispatch and indexed attachment cleanup

**Files:**

- Modify: `src/autonomous/gateway/coordinator.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Modify: `src/autonomous/ingress/attachments.py`
- Modify: `src/autonomous/ingress/router.py`
- Modify: `src/config/settings.py`
- Modify: `.env.example`
- Test: `tests/autonomous/integration/test_employee_team_gateway.py`
- Test: `tests/autonomous/integration/test_employee_runtime_startup_order.py`
- Test: `tests/autonomous/integration/test_employee_router_queues.py`
- Test: `tests/autonomous/contract/test_config_and_gate_contract.py`

- [ ] Add RED tests for two independent employees executing concurrently, anchor visibility before external execution, and close waiting for terminal finalization.
- [ ] Add bounded `dispatch_available()` (default concurrency 8) using `prepare_next()` synchronously before executor submission; keep `dispatch_next()` for compatibility.
- [ ] Reap every future and preserve incomplete durable attempts for recovery rather than manufacturing success.
- [ ] Add a replay-derived pending-cleanup acceptance-id set and sweep only those candidates, rechecking Router terminal state before I/O.
- [ ] Run focused integration, recovery/terminal chaos, config contract tests, and Ruff.

Safety: Same employee remains serial. PREPARED/EXECUTING are fsynced and anchored before any external call. Cleanup remains cleanup-started-and-anchored before unlink.

## Task 8: Cache ManagedGroupRegistry reads without extending trust

**Files:**

- Modify: `src/trust/registry.py`
- Test: `tests/test_managed_group_registry.py`
- Test: `tests/autonomous/integration/test_employee_router_queues.py`

- [ ] Add RED tests for repeated read reuse, shared read lock, cross-instance replace/revoke visibility, uncertain-marker invalidation, and corrupt replacement rejection.
- [ ] Add a `LOCK_SH` read transaction and cache only after a complete schema/relationship validation tied to target and uncertain-marker disk identities.
- [ ] Keep every mutation as `LOCK_EX + full reload`; never use cached authorization after any identity/stat/marker/corruption uncertainty.
- [ ] Run registry and Router trust regressions and Ruff.

Safety: Cache removes repeat parsing only; it does not extend authorization validity. Unknown, corrupt, uncertain, symlink, revision rollback, and missing grants remain fail-closed.

## Task 9: Integration verification and evidence

- [x] Re-read the original acceptance scope and inspect the complete diff for unrelated edits.
- [x] Run `uv run python -m pytest tests/ -q -m "not slow"`.
- [x] Run `uv run python -m pytest tests/autonomous/ -q` if not fully covered by the previous command (covered by the full `tests/` invocation).
- [x] Run `uv run python -m src.main --validate`.
- [x] Run `uv run python scripts/test_inventory.py tests/`.
- [x] Run `uv run ruff check src tests` and `git diff --check`.
- [ ] Repeat the relevant performance probes: delivery thread count, scheduler historical-key scan, Journal cached replay, and streaming accumulation.
- [x] Append implementation, reason, validation, and residual risk to `.Memory/2026-08-12.md`; add one dated summary line to `.Memory/Abstract.md` without overwriting existing changes.
