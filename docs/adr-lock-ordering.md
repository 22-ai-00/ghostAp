# ADR: Lock Ordering Convention

**Status:** Accepted
**Date:** 2026-04-26
**Context:** Multi-chat and repo-level isolation need a stable lock acquisition order.

## Problem

GhostAP uses two layers of locking for multi-chat isolation:

1. **Chat Lock** (`ChatLockManager._mu`) — message-pipeline level; gates whether a non-admin user's message is processed at all.
2. **Repo Lock** (`RepoLockManager._mu`) — handler-execution level; prevents concurrent modifications to the same repository from different chats.

Additionally, `ProjectContext._chat_lock` protects mutation of `allowed_chat_ids` / `evicted_chat_ids`, and `ProjectManager._lock` (an `RLock`) guards the project registry.
`SystemHandler._acp_activation_lock` linearizes ACP startup, durable project
selection and `/exit` so a stale activation cannot re-enter a mode.

Inconsistent acquisition order across call paths would create a deadlock risk.

## Decision

All code paths **must** acquire locks in this strict partial order:

```
BaseEngineManager._lock  (Lock, outermost)
  → BaseEngine._lock  (RLock)
    → SystemHandler._acp_activation_lock  (RLock)
      → ProjectManager._lock  (RLock)
        → ProjectContext._chat_lock  (Lock)
          → ChatLockManager._mu  (Lock)
            → RepoLockManager._mu  (Lock, innermost)
```

Rules:

1. **Never** hold an inner lock while acquiring an outer lock.
2. `ChatLockManager` checks (`should_block`, `is_locked`) execute **before** any handler code that might touch `RepoLockManager`.
3. `ProjectContext._chat_lock` is only acquired while `ProjectManager._lock` is held (via `add_chat_id` called from `ProjectManager` methods).
4. Callback closures registered on lock managers (e.g. `_on_hard_timeout_reclaim`, `on_eviction`) run **outside** the lock that triggers them, in a separate daemon thread, to avoid re-entrant acquisition.
5. ACP activation may hold its control-plane lock while committing or exiting a
   project mode. Project and lower-level lock holders must never call back into
   ACP activation ownership methods.

## TOCTOU in retry-command probe

The `_handle_retry_command` card action handler uses a **probe-acquire-then-release** pattern to fast-fail when the repo lock is still held by another chat.  This intentionally introduces a TOCTOU (Time-Of-Check-Time-Of-Use) window:

1. **Probe:** `repo_lock_mgr.acquire(root_path, cid)` — if it fails, send conflict card immediately (fast path).
2. **Release:** if probe succeeds, `repo_lock_mgr.release(root_path, cid)` — we do **not** hold the lock.
3. **Real acquire:** downstream handler (e.g. `_with_repo_lock` / `_safe_execute_engine`) acquires the lock independently.

Between step 2 and step 3, another chat **can** seize the lock.

### Why we accept this

- The card action dispatch path (`_handle_retry_command`) runs in a different execution context from the message-processing pipeline.  Propagating a held lock across `_process_with_intent` → intent routing → handler dispatch would require threading a lock handle through 4+ layers of indirection, introducing significant coupling.
- The probe exists purely as a **fast-fail optimisation** — without it, the user would wait for the full intent-routing pipeline before receiving a conflict card.
- The TOCTOU race is benign: if another chat grabs the lock between probe-release and real-acquire, the downstream handler will raise `LockConflictError`, which is caught and surfaced to the user as "lock preempted, please retry later".

### Boundary conditions

- **Extreme concurrency:** With many chats competing for the same repo, a user may see "probe OK" → "actually conflicted" more often.  The retry button with escalating `retry_count` mitigates this.
- **No data corruption:** The probe never executes any repo-mutating work; it only checks lock availability.  Actual work only happens after the real lock is held.

## Consequences

- Adding a new lock requires updating this ADR and placing it in the ordering chain.
- Test helpers that bypass `ProjectManager` and manipulate `ProjectContext` directly must still respect `_chat_lock` acquisition.
- `_lock_block_dedup_mu` was replaced by `MessageCache` (FS-02) and no longer exists.
- Runtime lock-ordering violation detection is available via `src/utils/lock_order.py`. Set `GHOSTAP_LOCK_ORDER_CHECK=1` to enable in development/CI.
