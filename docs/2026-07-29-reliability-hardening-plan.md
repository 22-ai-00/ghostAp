# GhostAP Reliability Hardening Plan

> Date: 2026-07-29
> Target branch: `dev`
> Scope: safe restart/shutdown, repository ownership, CardKit transport safety,
> ordinary-programming subtask diagnostics, and the pending durable Team recovery
> patch already present in the worktree.

## Evidence

- At 2026-07-28 11:23 an active Codex task executed `./restart.sh rr`.
  The detached worker sent `SIGTERM` to the service and its whole process group,
  closed the active ACP transport, produced `ConnectionError: Connection closed`,
  and was force-killed before the normal completion summary and lock release.
- The same remote-restart shape has repeatedly reached the force-kill path.
  `restart.sh` currently allows only 0.8 seconds and signals the process group
  during the first graceful phase.
- `FeishuWSClient.close()` destroys ordinary ACP sessions before stopping or
  draining `TaskScheduler`; running task futures are therefore broken underneath
  their handlers.
- `RepoLockManager` treats `chat_id` as the re-entrant owner. Two different
  scheduler runs in one chat can therefore mutate the same checkout concurrently.
- A final Markdown reference such as `![](/tmp/lark-auth-docs.png)` bypasses
  `ACPImagePublisher`; CardKit interprets the path as an image key and rejects the
  terminal patch.
- A failed fallback card patch is currently reported as `applied`, advances the
  visible signature, and can suppress the handler's text fallback even though no
  terminal state reached Feishu.
- Ordinary-programming agent/task failures retain only `status=failed`; the
  bounded failure detail in `ToolCallInfo.content` is discarded.
- The pending Team recovery patch passes its focused baseline, but initialization
  failures after runtime composition can leak the runtime and one callback
  compatibility shape loses durable recipient scope.

## Task 1: Harden and accept the pending Team recovery patch

**Files**

- Modify `src/autonomous/provisioning/composition.py`
- Modify `src/feishu/ws_client.py`
- Modify `tests/autonomous/integration/test_employee_hire_composition.py`
- Modify `tests/test_ws_client_reconnect.py`
- Modify `.Memory/2026-07-28.md`

**RED**

1. Add a callback compatibility test where recipient scope is accepted but
   `idempotency_key` is not. The runtime backend must retry without the optional
   idempotency key while preserving `tenant_key` and
   `requester_principal_id`.
2. Force handler construction to fail after `EmployeeDepartmentRuntime` has been
   composed. Assert the partially initialized client closes the runtime exactly
   once and re-raises the original error.
3. Add one recovery integration path that starts from anchored Team state and
   verifies the restored message origin, main-Bot reply, outbound audit, and
   terminal notify effect.

**GREEN**

- Add the missing scoped-without-idempotency invocation candidate.
- Wrap client initialization with a partial-initialization cleanup boundary.
- Correct the incident record so deployment/runtime state matches the observed
  11:23 restart evidence.

## Task 2: Add a cross-process safe restart gate

**Files**

- Add `src/utils/restart_gate.py`
- Add `tests/test_restart_gate.py`
- Modify `src/tasking/scheduler.py`
- Modify `src/feishu/ws_client.py`
- Modify `restart.sh`
- Modify `tests/test_restart_script.py`
- Modify `.gitignore`
- Modify `.env.example`
- Modify `README.md`

**RED**

1. A running scheduler task holds the drain reader lock. A detached restart
   requester must take the admission writer lock, wait for that task to finish,
   and prevent a later task from entering.
2. Lock descriptors are close-on-exec, so an ACP/tool child cannot inherit a
   reader and deadlock restart.
3. Two concurrent `rr` requests for one service generation coalesce: only the
   first generation may run stop/start; the second exits without restarting the
   newly created process.
4. Timeout is fail-closed and non-zero. It must never silently force a normal
   restart.

**GREEN**

- Give production `TaskScheduler` a run guard that takes shared admission/drain
  locks for the whole user task.
- Replace the shared `.restart_worker.sh` with a detached Python gate process.
- Capture and compare a service generation before executing stop/start.
- Keep the exclusive gate held through old-process shutdown and new-process
  readiness.

## Task 3: Correct shutdown ordering and budgets

**Files**

- Modify `src/tasking/scheduler.py`
- Modify `src/feishu/ws_client.py`
- Modify `src/main.py`
- Modify `restart.sh`
- Modify `tests/test_task_scheduler.py`
- Modify `tests/test_ws_client_patch.py`
- Modify `tests/test_restart_script.py`

**RED**

1. Once shutdown admission is fenced, queued work is canceled and new submissions
   fail, but running handlers retain ACP/card dependencies until they reach a
   terminal state.
2. `FeishuWSClient.close()` drains running scheduler work before
   `ACPSessionManager.cleanup_all()`.
3. A drain timeout requests cancellation and converges the scheduler run to
   `CANCELED`, not a false `SUCCEEDED` or generic `FAILED`.
4. Graceful stop sends `SIGTERM` only to the main service PID. Process-group
   termination is reserved for the explicitly logged forced-cleanup path.
5. Application lock managers close after the Feishu client and card delivery
   paths, not before them.

**GREEN**

- Add explicit scheduler admission fencing, bounded idle wait, and running-task
  cancellation.
- Reorder close into: stop inbound admission, drain/cancel scheduler work, close
  engines/sessions, drain card delivery, stop executors/transports, then close
  lock managers.
- Use a realistic configurable TERM grace period and report a forced stop as
  degraded rather than as a graceful success.

## Task 4: Make repository lock ownership request-scoped

**Files**

- Modify `src/tasking/scheduler.py`
- Modify `src/tasking/__init__.py`
- Modify `src/repo_lock.py`
- Modify `src/feishu/handlers/lock_helper.py`
- Modify `src/feishu/handlers/programming.py`
- Modify `src/feishu/retry_handler.py`
- Modify `tests/test_repo_lock.py`
- Modify `tests/test_shell_repo_lock_strict.py`

**RED**

1. The same `chat_id` with different scheduler `run_id` values conflicts.
2. Nested acquisition by the same `(chat_id, run_id)` remains re-entrant.
3. Release/touch by a different run cannot mutate the holder.
4. A shell-fast run in the same chat cannot overlap an active programming run.

**GREEN**

- Publish the current scheduler run through a context variable.
- Store and compare a request owner token while preserving chat-level
  attribution for cards and diagnostics.
- Thread the owner token through acquire, heartbeat, release, retry, and
  streaming paths.
- Preserve the documented P2P policy outside the already strict shell path; log
  the broader P2P integrity question as a product/security backlog item.

## Task 5: Make image/card transport fail safe

**Files**

- Modify `src/acp/session.py`
- Modify `src/card/delivery/page_mutator.py`
- Modify `src/card/delivery/types.py`
- Modify `tests/test_acp_session_event_fence.py`
- Modify `tests/test_card_delivery_page_mutator.py`

**RED**

1. A changed, explicitly referenced image inside the ACP project root emits an
   image event before the prompt snapshot is released.
2. Raw Markdown image syntax in a CardKit markdown node and in
   `element_content` is converted to bounded plain markdown unless the target is
   a valid Feishu image key. Native `tag=img` elements are unchanged.
3. `card contains invalid image keys` is classified as content-invalid even when
   the transport code is zero.
4. If the known-good content/audit fallback patch also fails, no binding
   signature or text advances and the outcome is not `applied`.

**GREEN**

- Invoke the existing changed/in-root image discovery before snapshot release.
- Sanitize only markdown transport content at the shared mutation boundary.
- Return an honest retry/failure outcome so `CardSession` remains open and the
  programming handler sends its text fallback.

## Task 6: Preserve bounded subtask failure diagnostics

**Files**

- Modify `src/card/programming_adapter.py`
- Modify `src/card/render/tools.py`
- Modify `tests/test_programming_completion_guards.py`
- Modify `tests/test_programming_card_session.py`

**RED**

1. A failed agent/task `TOOL_CALL_DONE` keeps a short sanitized failure detail in
   main-card metadata.
2. The expanded failed-subtask panel shows that detail.
3. Structured stdout, call IDs, control characters, and oversized content do not
   leak into the summary.
4. Successful subtask summaries remain compact.

**GREEN**

- Store a bounded card-safe `error` field only for failed subtask terminals.
- Render it as a subordinate line in the already-expanded failure panel.

## Task 7: Repository-wide verification and handoff

**Files**

- Modify `.Memory/2026-07-29.md`
- Modify `.Memory/Abstract.md`
- Modify `.Memory/Backlog.md`

Run, in order:

```bash
uv run python -m pytest <each new RED test file> -q
uv run python -m pytest tests/test_restart_gate.py tests/test_restart_script.py tests/test_task_scheduler.py tests/test_repo_lock.py tests/test_shell_repo_lock_strict.py -q
uv run python -m pytest tests/test_acp_session_event_fence.py tests/test_card_delivery_page_mutator.py tests/test_programming_completion_guards.py tests/test_programming_card_session.py -q
uv run python -m pytest tests/autonomous/unit/test_employee_team_service.py tests/autonomous/unit/test_team_coordinator.py tests/autonomous/integration/test_employee_hire_composition.py tests/test_project.py tests/test_ws_client_reconnect.py -q
uv run ruff check src tests
uv run python -m src.main --validate
uv run python -m pytest tests/ -q -m "not slow"
uv run python -m pytest tests/ -q
git diff --check
```

Record medium/low findings rather than expanding implementation scope without
evidence. Perform a task-level review after each GREEN task and a final whole-diff
review before committing. Commit intentionally on `dev`, push with a normal
non-force update to `origin/dev`, and do not restart the live service from inside
this task.
