# First Six Command Reliability Fixes Implementation Plan

> Implementation Tasks 1–7 completed on 2026-08-12. Final whole-repository
> suites, independent review, durable notes, and focused rollback commits are
> recorded in `.Memory/2026-08-12.md`.
>
> Historical note (2026-08-20): references below to ActionMatrix and Workflow
> execution isolation describe the implementation at that time. GhostAP's
> execution policy and process sandbox were subsequently removed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Restore the first six audited command surfaces without reviving retired Slock/Goal/Run components, while preserving durable Employee Department and Workflow trust boundaries.

**Architecture:** Main-Bot employee administration remains a thin Feishu handler over current Journal-backed Employee Department services. Workflow launches only the packaged trusted JavaScript runtime while retaining the user project as process cwd. Employee-Bot controls reply through the durable Outbox and group tasks require authoritative proof that exactly one employee was addressed.

**Tech Stack:** Python 3.13, uv, pytest, Pydantic settings, lark-oapi/lark-channel-sdk, Journal/Outbox Employee Department, Node.js Workflow runtime.

## Global Constraints

- [x] Read `AGENTS.md`, `.Memory/Abstract.md`, `docs/testing.md`, and the relevant current modules before editing.
- [x] Use `uv` for every Python dependency, test, lint, and validation command.
- [x] Start every behavior change with a focused failing regression test and record the expected failure.
- [x] Do not reconnect Slock, Goal/Run Manager, old activation verification, legacy cards, or direct SDK delivery shortcuts.
- [x] Keep Journal anchoring, lock order, ActionMatrix fail-closed checks, tenant isolation, Outbox idempotency, and Workflow Agent Pool boundaries intact.
- [x] Make one focused commit per independently fixed audit issue; do not push.
- [x] Run a task-specific review after each commit and resolve every finding before beginning the next task.

---

## Task 1: Restore Main-Bot Employee lifecycle and data commands

**Files:**

- Modify: `src/autonomous/provisioning/hire_port.py`
- Modify: `src/autonomous/provisioning/hire_service.py`
- Modify: `src/feishu/handlers/employee.py`
- Modify: `src/feishu/handlers/system.py`
- Modify: `src/feishu/product_catalog.py`
- Modify: `README.md`
- Test: `tests/autonomous/unit/test_employee_hire_admission.py`
- Test: `tests/test_employee_roster.py`
- Test: `tests/test_employee_data_handlers.py`
- Test: relevant fire/dispatcher contract suites

- [x] Add RED service tests for stable request idempotency, full-field drift rejection, service-level admin authorization, readiness/closed/capacity gates, casefold name uniqueness, existing-App uniqueness, archived-name release plus create in one Journal frame, anchor-before-submit, locks released before submit, submission failure retaining durable admission, and empty model meaning backend default.
- [x] Restore `EmployeeHireService.start_hire()` and the production implementation from the last pre-retirement contract, adapting it to current frozen domain objects and current constructor dependencies only.
- [x] Complete and validate the controlled profile before writing; derive stable IDs from tenant plus message ID; synchronize projections while holding the prescribed workforce/hire/Journal locks.
- [x] Commit workforce events and replay them before invoking the external provisioning submitter; invoke the submitter only after every admission lock is released.
- [x] Allow normalized empty model as the backend-default sentinel while rejecting non-string, whitespace-bearing, composite, or otherwise invalid model values.
- [x] Add RED handler/router tests proving `/hire`, `/h`, `/fire`, `/history`, and `/employee-memory` reach `EmployeeHandler` from every interaction mode and no longer fall through to unknown-slash handling.
- [x] Add RED authorization tests proving non-admin, non-P2P, missing-sender, missing-tenant, and hire-without-union requests fail before employee lookup or service/query invocation.
- [x] Implement strict `shlex` parsing for documented controlled hire/fire flags; reject duplicate/unknown/missing flags and explicitly reject arbitrary `--prompt`.
- [x] Select an explicitly configured strictly available ACP tool first, otherwise the repository recommendation order; store an empty model for backend default rather than inventing a `default` model.
- [x] Delegate `/fire` to current `EmployeeFireService`, keeping `SAFE_ABORT`, `EXTERNAL_UNKNOWN`, `ACTION_REQUIRED`, drain, and external-disposition confirmation truthful.
- [x] Resolve history/memory targets only within the authenticated tenant after authorization; use `AuthenticatedDataRequest`, `HistoryQuerySpec`, and `MemoryQuerySpec(full_l1=True)` so access decisions and audit remain authoritative and fail closed.
- [x] Render bounded safe metadata/text only; never return encrypted request/result blobs, secrets, filesystem paths, or data after query-audit failure.
- [x] Register exact/prefix/intercept routing in `SystemHandler`, retain catalog alias canonicalization, and align README/catalog usage with the automatic no-selection-card path.
- [x] Run focused hire, roster, role, fire, history/memory, dispatcher, and ActionMatrix tests; run Ruff on changed files.
- [x] Commit as `fix(employee): restore main bot lifecycle commands`.

## Task 2: Pin Workflow to its packaged trusted runtime

**Files:**

- Modify: `src/workflow_engine/constants.py`
- Modify: `src/workflow_engine/bridge.py`
- Test: `tests/test_workflow_bridge_transport.py`

- [x] Add a RED transport test using an external temporary project cwd containing a malicious/invalid decoy at `src/workflow_engine/runtime/runtime.js`.
- [x] Assert the Node argv uses an absolute path to GhostAP's packaged runtime, never the project decoy, while `Popen.cwd` remains the user project.
- [x] Resolve the runtime from the installed Python module directory and fail clearly before process launch if the packaged resource is missing; remove project-relative probing and fallback.
- [x] Run bridge/runtime binding tests, `node --check`, and a wheel-content check proving `runtime.js` remains packaged.
- [x] Commit as `fix(workflow): pin the packaged runtime path`.

## Task 3: Canonicalize Workflow long command aliases

**Files:**

- Modify: `src/feishu/handlers/workflow.py`
- Test: `tests/test_workflow_topic_engine.py`

- [x] Add a RED parameterized test mapping `/workflow`, `/workflow_status`, `/workflow_help`, and `/stop_workflow` to the same action and arguments as their short forms.
- [x] Replace the handler's legacy raw reparse with `SlashCommandParser`, consuming its canonical command and normalized argument contract.
- [x] Verify tabs, surrounding whitespace, and case behavior remain aligned with the public command parser, and unknown commands remain rejected.
- [x] Run Workflow topic, dispatcher routing, and robustness tests.
- [x] Commit as `fix(workflow): honor canonical command aliases`.

## Task 4: Exit Traex and Grok programming modes

**Files:**

- Modify: `src/feishu/handlers/system.py`
- Test: `tests/test_exit_deferred_behavior.py`

- [x] Add a RED table-driven test for all seven programming modes, with explicit Traex and Grok assertions that the matching handler's `exit_mode()` is called exactly once and the SMART fallback reply is not used.
- [x] Replace the incomplete hard-coded branch chain with a total programming-mode-to-handler mapping while retaining topic-engine exit precedence.
- [x] Delegate cleanup to each existing programming handler; do not manually mutate mode/session state.
- [x] Run deferred exit, footer action, Grok/Traex ACP, and direct-lane contract tests.
- [x] Commit as `fix(feishu): exit every programming backend`.

## Task 5: Make Employee-Bot `/status` return durable current status

**Files:**

- Modify: `src/autonomous/provisioning/composition.py`
- Modify: `src/autonomous/outbox/lifecycle.py`
- Modify: `src/feishu/ws_client.py`
- Test: `tests/autonomous/integration/test_employee_stop_control.py`
- Test: `tests/autonomous/integration/test_employee_outbox_delivery.py`

- [x] Add RED tests proving exact `/status` emits one employee-owned, idempotent Outbox response and drains it, without activating hires, allocating a cold actor, opening an ACP session, or entering the ordinary router.
- [x] Cover ready/busy/degraded/stopping status, no-current-task, current-chat task state, restart replay stability, malformed `/status` arguments, and owner P2P/group tenant isolation.
- [x] Inspect the current employee runtime without allocating an actor and summarize only coarse runtime state plus tasks authorized for the current chat/thread scope.
- [x] Anchor a deterministic terminal control response through `EmployeeOutboxLifecycle` before recording the completed/unavailable ingress disposition, then use the existing Outbox drain path.
- [x] Keep `/status` read-only and remove stale UI text that tells users it manually activates an already auto-activated employee.
- [x] Run status/stop, Outbox delivery, runtime startup, and autonomous authorization tests.
- [x] Commit as `fix(employee): reply to employee status controls`.

## Task 6: Route only uniquely targeted Employee-Bot group `/task`

**Files:**

- Modify: `src/autonomous/provisioning/composition.py`
- Modify: employee Channel/message normalization only if authoritative mention data is not already preserved
- Test: `tests/autonomous/integration/test_employee_runtime_startup_order.py`
- Test: `tests/autonomous/integration/test_employee_team_gateway.py`
- Test: `tests/autonomous/contract/test_employee_channel_contract.py`

- [x] Add RED tests proving an owner command that uniquely mentions the current employee and contains `/task <non-empty description>` bypasses the main-Bot slash-observation gate exactly once.
- [x] Add negative tests for bare/unaddressed group `/task`, another employee's mention, multiple employee targets, missing description, `/tasks`, `/taskfoo`, unmanaged groups, foreign tenants/unions, and stale app/generation/connection identities.
- [x] Preserve existing `/help`, `/role`, and other main-Bot slash observation suppression so one group command cannot fan out across all employee apps.
- [x] Require the current official employee Channel identity to be the unique authoritative target; route only the business description while preserving the original encrypted ingress as audit truth.
- [x] Resolve cross-App owner identity using tenant-scoped union identity rather than comparing unrelated app-scoped open IDs.
- [x] Emit a bounded usage response only when it can be delivered by the uniquely addressed employee; otherwise fail closed without a response storm.
- [x] Run startup-order, team gateway, router queue, Channel contract, slash reconciliation, and full autonomous tests.
- [x] Commit as `fix(employee): route targeted group task commands`.

## Task 7: Whole-change verification, independent review, and durable notes

**Files:**

- Modify: `.Memory/2026-08-12.md`
- Modify: `.Memory/Abstract.md`
- Modify: `.Memory/Backlog.md`
- Modify: `docs/first-six-command-fixes-plan.md`

- [x] Run `uv run python scripts/test_inventory.py tests/` on the final worktree and record its counts.
- [x] Run `uv run python -m pytest tests/ -q -m "not slow"` on the final worktree.
- [x] Run the slow partition required by `docs/testing.md` and `uv run pytest tests/autonomous/ -q` on the final worktree.
- [x] Run `uv run ruff check src tests`, `uv run python -m src.main --validate`, `node --check src/workflow_engine/runtime/runtime.js`, and `git diff --check` after the last code edit.
- [x] Finish independent whole-diff review for correctness, security, concurrency, idempotency, performance, and missing test coverage; resolve every Critical/Important finding and rerun affected suites.
- [x] Replace the provisional verification block in `.Memory/2026-08-12.md` with final-worktree evidence; keep the dated Abstract index line and reviewed backlog risks.
- [x] Remove the resolved high-priority orphan-command backlog item while retaining unrelated recovery/backlog work.
- [x] Commit documentation and evidence as `docs(audit): record command reliability repairs`.
