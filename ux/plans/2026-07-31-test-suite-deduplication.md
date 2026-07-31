# Test Suite Semantic Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce GhostAP's default pytest collection and feedback time without removing independent security, recovery, concurrency, persistence, or external-protocol contracts.

**Architecture:** Consolidate repeated parameter cases only when they call the same production boundary with the same assertion contract. Preserve every input as an in-test equivalence table, delete tests that are strict subsets of stronger contracts, and move genuinely process/wall-clock-dependent cases to the existing `slow` release layer.

**Tech Stack:** Python 3.13, pytest, `uv`, AST inventory

## Global Constraints

- Use `uv` for every Python and test command.
- Do not delete independent backend/engine contracts merely because their test bodies are identical.
- Preserve security, authorization, concurrency, Journal/effect, recovery, persistence, and Feishu schema coverage.
- Every merged loop must include the failing input in its assertion message.
- `slow` changes scheduling only; the full release command must still collect and run those tests.
- Keep production behavior unchanged in this task.

---

### Task 1: Consolidate command catalog and registry equivalents

**Files:**
- Modify: `tests/test_main_slash_commands.py`
- Modify: `tests/test_command_registry_contract.py`

**Interfaces:**
- Consumes: `MAIN_AGENT_COMMANDS`, `SystemHandler` routing predicates, `TOPIC_ENGINE_COMMANDS`, and `parse_slock_command()`
- Produces: One loop-backed contract per routing family instead of one pytest item per command

- [x] **Step 1: Preserve the before baseline**

Run:

```bash
uv run python -m pytest \
  tests/test_main_slash_commands.py \
  tests/test_command_registry_contract.py \
  --collect-only -q
```

Expected: the current parameterized command cases are collected individually.

- [x] **Step 2: Replace cheap parameter expansion with equivalence loops**

In `test_main_slash_commands.py`, retain all current command strings but iterate inside the five routing-family tests. In `test_command_registry_contract.py`, union the two same-predicate exact-command snapshots and iterate inside the prefix, Deep, Spec, and exit contracts.

Each assertion must identify the command, for example:

```python
for command in sorted(SPEC_COMMANDS):
    assert SystemHandler.is_spec_command(command), f"{command!r} was not routable"
```

- [x] **Step 3: Run the focused contracts**

Run:

```bash
uv run python -m pytest \
  tests/test_main_slash_commands.py \
  tests/test_command_registry_contract.py -q
```

Expected: PASS with every original command input still checked.

### Task 2: Remove Workflow action subset checks and reuse one registry snapshot

**Files:**
- Modify: `tests/test_workflow_api_contract.py`

**Interfaces:**
- Consumes: `_collect_workflow_constants()`, `_collect_workflow_action_ids_only()`, and `_collect_registered_workflow_handlers()`
- Produces: One complete action-registration contract and one complete handler-signature contract

- [x] **Step 1: Delete strict subset and language-mechanism checks**

Delete `test_four_new_workflow_entries_are_present()` because every listed action is already included by `test_all_workflow_constants_are_registered_in_system_actions()`. Delete `test_system_actions_contains_no_unknown_workflow_placeholders()` because it only rechecks Python string shape and is not an independent dispatch contract.

- [x] **Step 2: Build the action registry once**

Remove the per-action pytest parametrization from `test_workflow_action_has_handler_with_four_positional_args()`. Build the registered-handler mapping once, then loop over all exported Workflow action IDs while preserving the existing action-specific diagnostics.

- [x] **Step 3: Run Workflow API contracts**

Run:

```bash
uv run python -m pytest tests/test_workflow_api_contract.py -q
```

Expected: PASS with all exported action IDs still checked for registration and the four-argument handler signature.

### Task 3: Collapse repeated full-client WS routing fixtures

**Files:**
- Modify: `tests/test_ws_client_routing.py`

**Interfaces:**
- Consumes: `mock_ws_client`, `FeishuWSClient._dispatch_message_logic()`, and `SlashCommandParser.parse()`
- Produces: The same engine/mode and missing-project matrix with one client fixture per behavioral contract

- [x] **Step 1: Delete the weaker duplicate engine-routing test**

Delete `test_explicit_engine_commands_override_persistent_programming_mode()`: its four inputs and negative mode-handler assertion are a strict subset of `test_explicit_engine_command_reaches_its_final_handler_in_every_programming_mode()`, which additionally checks the concrete terminal handler across all seven programming modes.

- [x] **Step 2: Merge the three matrix tests**

Convert these parameter expansions to in-test loops:

- 4 explicit engines × 7 programming modes
- 4 topic engines × slash/plain input
- 5 safe missing-project recovery commands

Reset/rebind mocks between iterations and include `(engine, mode, command)` in assertion messages so a failure remains localizable.

- [x] **Step 3: Re-run the production-shaped routing contracts**

Run:

```bash
uv run python -m pytest \
  tests/test_ws_client_routing.py::test_explicit_engine_command_reaches_its_final_handler_in_every_programming_mode \
  tests/test_ws_client_routing.py::test_topic_engine_without_resolved_project_never_falls_back_to_smart \
  tests/test_ws_client_routing.py::test_missing_topic_project_allows_safe_recovery_and_diagnostics_commands \
  -q
```

Expected: 3 passing pytest items while checking all 41 original input combinations.

### Task 4: Delete copied-formula and hand-simulated duplicates

**Files:**
- Modify: `tests/test_system_timeout_format.py`
- Modify: `tests/test_card_builders.py`
- Modify: `tests/test_handlers.py`
- Modify: `tests/test_card_rendering.py`
- Modify: `tests/test_escalation_timeout.py`
- Modify: `tests/test_spec_engine_review_reliability.py`
- Modify: `tests/test_workflow_dynamic_roles.py`
- Modify: `tests/test_ui_text_format.py`
- Modify: `tests/test_ui_text_placeholders.py`
- Modify: `tests/test_skip_retry_feedback.py`

**Interfaces:**
- Consumes: canonical production-boundary coverage in `tests/test_action_retry.py`, `tests/test_slock_redact.py`, `tests/test_error_formatting.py`, and stronger timeout/card tests
- Produces: removal of tests that assert copied test formulas, simulate handler code locally, or duplicate the same function and boundary

- [x] **Step 1: Delete copied timeout-display formulas**

Delete the six `TestSystemBuilderTimeoutDisplay` cases and three `TestTimeoutDisplayFormatting` parameters. They reproduce formatting logic inside tests and assert only the test-owned formula; `format_friendly_duration()` and rendered help cards remain covered at the production boundary.

- [x] **Step 2: Delete hand-simulated and same-boundary duplicates**

Delete:

- two Retry tests that manually execute pseudo-handler code instead of the registered handler;
- three duplicate `redact_sensitive()` examples covered by the stronger redaction suite;
- two escalation card-message tests covered by existing timeout I/O tests;
- eight repeated `classify_timeout()` examples already covered by the canonical error-formatting suite, while retaining the exact 80% threshold and 15.9/20 lower boundary.

- [x] **Step 3: Collapse prompt/UI/retry duplication**

Delete Workflow prompt phrase/section checks that are strict subsets of the complete guidance and section-order contracts. Keep one canonical UI format-renderability scan, remove a zero-detection placeholder test and copied builder branch, and replace four skip-retry key checks with one non-empty text contract while retaining the real retry-event owner.

- [x] **Step 4: Run canonical owners and affected files**

Run:

```bash
uv run python -m pytest \
  tests/test_system_timeout_format.py \
  tests/test_card_builders.py \
  tests/test_handlers.py::TestRetryCommandHandler \
  tests/test_action_retry.py \
  tests/test_card_rendering.py \
  tests/test_slock_redact.py \
  tests/test_escalation_timeout.py \
  tests/test_spec_engine_review_reliability.py \
  tests/test_error_formatting.py -q
```

Expected: PASS.

### Task 5: Cancel TimerScheduler work before wake and join

**Files:**
- Modify: `src/card/timers/scheduler.py`
- Modify: `tests/test_timer_scheduler.py`

**Interfaces:**
- Consumes: `TimerScheduler.shutdown()`
- Produces: deterministic, idempotent shutdown that cancels queued future work and wakes the scheduler before joining its thread

- [x] **Step 1: Add a red regression**

Schedule a far-future callback, call `shutdown()`, and assert the call returns promptly, the scheduler thread is no longer alive, and the callback never runs.

- [x] **Step 2: Reorder shutdown around the scheduler lock**

Establish shutdown state, cancel queued scheduler events/clear the queue under the scheduler lock, and then signal the wake event before `join()`. Preserve repeated shutdown safety and never run a cancelled callback.

- [x] **Step 3: Run timer and resource-cleanup regressions**

Run:

```bash
uv run python -m pytest tests/test_timer_scheduler.py tests/test_resource_cleanup.py -q
uv run ruff check src/card/timers/scheduler.py tests/test_timer_scheduler.py
```

Expected: PASS and no two-second shutdown wait.

### Task 6: Put true process and wall-clock contracts in the slow layer

**Files:**
- Modify: `tests/autonomous/integration/test_employee_hire_composition.py`
- Modify: `tests/autonomous/integration/test_employee_channel_process.py`
- Modify: `tests/autonomous/integration/test_slash_reconciliation.py`
- Modify: `tests/autonomous/security/test_employee_channel_isolation.py`
- Modify: `tests/autonomous/security/test_runner_isolation.py`
- Modify: `tests/test_shell_repo_lock_strict.py`

**Interfaces:**
- Consumes: the existing `slow` marker declared by pytest configuration
- Produces: a faster `-m "not slow"` gate while retaining every process, timeout, retry, and lease contract in full/release runs

- [x] **Step 1: Mark measured >1 second real-wait/process cases**

Apply `@pytest.mark.slow` only to cases measured above one second that intentionally wait for retries, process timeout/reaping, sandbox attestation, or lease renewal. Do not mark ordinary CPU-heavy unit tests.

- [x] **Step 2: Prove both scheduling layers retain the tests**

Run:

```bash
uv run python -m pytest \
  tests/autonomous/ \
  tests/test_shell_repo_lock_strict.py \
  --collect-only -q -m slow
```

Expected: every newly marked node is present.

Run:

```bash
uv run python -m pytest \
  tests/autonomous/ \
  tests/test_shell_repo_lock_strict.py \
  -q -m "not slow"
```

Expected: PASS without the intentional wall-clock waits.

### Task 7: Verify reduction and record the decision

**Files:**
- Modify: `.Memory/2026-07-31.md`
- Modify: `.Memory/Abstract.md`

**Interfaces:**
- Consumes: focused regressions, AST inventory, pytest collection, and before/after timing
- Produces: durable evidence of exactly what was merged, retained, reclassified, and why

- [x] **Step 1: Re-run inventory and collection**

Run:

```bash
uv run python scripts/test_inventory.py tests/
uv run python -m pytest tests/ --collect-only -q
```

Expected: fewer collected pytest items, no new exact-duplicate group, and unchanged independent Deep/Spec and backend contracts.

- [x] **Step 2: Run affected suites and static checks**

Run:

```bash
uv run python -m pytest \
  tests/test_main_slash_commands.py \
  tests/test_command_registry_contract.py \
  tests/test_workflow_api_contract.py \
  tests/test_ws_client_routing.py -q
uv run ruff check \
  tests/test_main_slash_commands.py \
  tests/test_command_registry_contract.py \
  tests/test_workflow_api_contract.py \
  tests/test_ws_client_routing.py
git diff --check
```

Expected: PASS.

- [x] **Step 3: Update project memory**

Record the root-cause evidence: exact-body duplicates were already intentionally bounded, while repeated parameter fixtures and unlayered process waits inflated feedback time. Include before/after collection counts, focused runtime, retained high-risk contracts, and the exact verification commands.
