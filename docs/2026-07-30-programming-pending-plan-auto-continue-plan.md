# Ordinary Programming Pending-Plan Auto-Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `subagent-driven-development` to execute each task, with TDD evidence and task review before moving on.

**Goal:** Prevent ordinary Coco/Claude/Aiden/Codex/Gemini/Traex/TTADK programming runs from stopping at a normal `end_turn` merely because the agent left plan items pending. GhostAP must continue the same authorized task once, on the same ACP session and card, while preserving real permission and completion gates.

**Architecture:** Keep `classify_prompt_result()` fail-closed. Add a small ACP orchestration layer above the existing deadline/finalization runner that recognizes the narrow `end_turn + pending plan + no active tool` condition and sends one bounded continuation turn. Return structured execution metadata so both programming delivery paths can distinguish ordinary incompleteness from a post-continuation request for new user input. Streaming cards terminate as blocked rather than failed only in that latter case.

**Tech Stack:** Python 3.11+, ACP session APIs, existing `PromptResult`/`PromptAssessment`, `ProgrammingCardSession`, pytest, uv, ruff.

## Global Constraints

- `classify_prompt_result()` must continue to classify `end_turn` with any non-completed plan entry as `PromptOutcome.INCOMPLETE`; never turn pending work into success.
- Automatic continuation is allowed only for a natural `end_turn` with at least one pending plan entry and no non-terminal tool call.
- Use exactly one automatic continuation turn. Never loop indefinitely.
- Reuse the same ACP session, `on_event` callback, programming card, task scope, and original wall-clock deadline.
- The continuation receives only the deadline's remaining budget. It must not reset the configured timeout or bypass the existing finalization reserve and session-retirement behavior.
- Never auto-continue a timeout/finalization result, cancellation, refusal, token/turn limit, exception, or result with a non-terminal tool call.
- The continuation prompt is not authorization for new external authority. It must explicitly preserve gates for credentials, deployment/publishing, data deletion, irreversible external side effects, and ACP/sandbox/tool permissions unless the original user request already granted that exact authority. For ordinary safe, reversible design/implementation choices already inside the user's requested scope, it directs the agent to use an explicitly documented recommended option or the smallest safe default, record that choice, and continue. It defers only items that truly need new authority and continues every other in-scope item.
- Both streaming and non-streaming programming paths must use the same runner.
- If the single continuation still ends with pending plan items and no active tool, report that user input is needed. The streaming card must not emit a `FAILED` event for this case.
- Preserve one-user-task/one-main-card behavior.
- Do not modify or stage the pre-existing user changes in `.Memory/Abstract.md`, `.Memory/2026-07-30.md`, or `docs/2026-07-30-ghostap-product-convergence-plan.md`.
- Use `uv` only. All focused test output must pass; warnings already present in unchanged third-party dependencies must be identified rather than hidden.

### Task 1: Add the bounded ACP continuation runner

**Files:**

- Create: `src/acp/continuation.py`
- Modify: `src/acp/outcome.py`
- Modify: `src/acp/__init__.py`
- Create: `tests/test_acp_prompt_continuation.py`
- Modify: `tests/test_programming_completion_guards.py`

**Step 1: Write failing outcome-metadata tests**

Extend the existing completion guard tests to require structured pending-plan and incomplete-tool counts on `PromptAssessment`. Retain the existing assertion that a pending plan is incomplete.

Run:

```bash
uv run python -m pytest tests/test_programming_completion_guards.py -q
```

Expected RED: the assessment does not yet expose the required counts.

**Step 2: Add structured assessment metadata**

Add defaulted integer fields to `PromptAssessment` for pending plan entries and incomplete tool calls. Compute both sets before choosing the diagnostic so continuation eligibility does not depend on parsing localized text. Populate the counts in every classification branch where applicable without changing existing outcome semantics.

**Step 3: Write failing continuation tests**

Create focused tests covering:

1. First result is `end_turn` with pending plan, second result is complete: two sends on the same session, one continuation, completed final result.
2. Two consecutive eligible pending-plan results: exactly two sends total and `awaiting_user_input=True`.
3. First result has a pending plan and the continuation emits text but no plan update: the prior plan is carried forward and the run remains incomplete/awaiting input rather than being falsely marked complete.
4. Pending plan plus active tool: no continuation.
5. Cancellation, refusal, `max_tokens`, and timeout/finalization: no ordinary continuation.
6. Elapsed time is removed from the second turn's timeout budget; the total deadline is never reset.
7. The generated follow-up directs already-authorized work to continue, adopts documented recommended/minimum-safe defaults for ordinary reversible in-scope choices, and explicitly denies newly inferred credentials, deploy/publish, deletion, irreversible side effects, or bypassing ACP/sandbox/tool permissions while preserving authority explicitly present in the original request.

Use real `PromptResult`, `PlanInfo`, `PlanEntryInfo`, and `ToolCallInfo` values. A small fake session is acceptable; assert the actual prompts and timeout values passed to `send_prompt()`.

Run:

```bash
uv run python -m pytest tests/test_acp_prompt_continuation.py -q
```

Expected RED: the new runner/API does not exist.

**Step 4: Implement the runner**

In `src/acp/continuation.py`, add:

- An immutable result type containing the final `PromptResult`, its `PromptAssessment`, the number of automatic continuations, and `awaiting_user_input`.
- A public runner wrapping `run_prompt_with_finalization()` with the same session/event/finalization callbacks used today.
- An optional continuation-boundary callback invoked exactly once immediately before the second send, allowing streaming consumers to close the first turn's active text/reasoning blocks without creating a new card.
- A module constant limiting ordinary continuations to one.
- A narrowly scoped continuation-prompt builder with the safety language in the global constraints.
- Monotonic deadline accounting. The first turn uses the original budget; the continuation uses only `deadline - now`.
- A per-turn finalization flag that prevents a timeout finalization result from triggering ordinary continuation.
- Fail-closed per-turn result aggregation: if the continuation omits `PLAN_UPDATE`, retain the prior plan for assessment (and retain prior aggregate artifacts where the model contract requires it) so absence of an update cannot erase known pending work.

Export the new result type and runner from `src/acp/__init__.py`.

**Step 5: Verify Task 1**

Run:

```bash
uv run python -m pytest tests/test_acp_prompt_continuation.py tests/test_programming_completion_guards.py tests/test_acp_prompt_finalization.py -q
uv run ruff check src/acp/continuation.py src/acp/outcome.py tests/test_acp_prompt_continuation.py tests/test_programming_completion_guards.py
```

Expected GREEN: all tests pass and ruff reports no errors.

**Step 6: Commit Task 1 selectively**

Stage only the five Task 1 files. Do not stage any pre-existing memory or product-plan changes.

Commit subject:

```text
fix(acp): continue ordinary pending plans once
```

### Task 2: Integrate continuation into both programming delivery paths

**Files:**

- Modify: `src/feishu/handlers/programming.py`
- Modify: `src/card/programming_adapter.py`
- Modify: `src/card/state/reducers/lifecycle.py`
- Modify: `src/card/ui_text.py`
- Modify: `tests/test_card_reducer_main.py`
- Modify: `tests/test_programming_card_session.py`
- Modify: `tests/test_tui2acp_terminal_cleanup.py`

**Step 1: Write failing card-session test**

Add a `ProgrammingCardSession` test that starts a card, marks it as waiting for user input through a new adapter method, and verifies:

- The card session closes cleanly.
- The terminal state is `blocked`, not `failed`.
- The supplied reason remains available in card state.
- The reducer sets `terminal_reason="blocked"` so late async events are fenced before terminal delivery completes.

Run:

```bash
uv run python -m pytest tests/test_programming_card_session.py -q -k waiting
```

Expected RED: the adapter method does not exist.

**Step 2: Implement the card adapter terminal**

Add a clearly named `ProgrammingCardSession` method for the post-continuation waiting state. It must flush text/reasoning, close active blocks, finalize live subagent summaries without claiming success, dispatch `CardEvent.blocked(reason)`, and stop the ticker. Update the BLOCKED reducer to record `terminal_reason="blocked"`; add a reducer regression proving late state-changing events are ignored after this logical terminal transition. Do not alter global card themes or unrelated engine behavior.

**Step 3: Write failing handler regressions**

Add streaming-path tests using a fake ACP session and fake programming-card adapter:

1. Pending first turn followed by a complete continuation: exactly two prompt sends, same session/card adapter, `finish()` called, and `fail()`/waiting terminal never called.
2. Two eligible pending turns: exactly two prompt sends, waiting terminal called, and `fail()` never called.

Add or extend a non-streaming test proving the same pending-then-complete flow sends twice and returns the completed response with the success reaction.

Run:

```bash
uv run python -m pytest tests/test_tui2acp_terminal_cleanup.py tests/test_programming_card_session.py tests/test_card_reducer_main.py -q
```

Expected RED: the handlers still call the one-turn finalization runner and map pending plans directly to failure.

**Step 4: Integrate the shared runner**

Replace both direct `run_prompt_with_finalization()` calls in `ProgrammingModeHandler` with the Task 1 runner. Preserve all existing timeout finalization callbacks, replacement-session logic, retirement behavior, event rendering, heartbeat behavior, and snapshot cleanup.

In the non-streaming path, track the current active session exactly as the streaming path does: when timeout recovery creates a replacement, subsequent retirement must target that replacement rather than the original dead session.

For the streaming path:

- Completed and cancelled outcomes keep their current behavior.
- `awaiting_user_input=True` appends a localized “automatic continuation completed; confirmation is still needed” notice and invokes the new waiting terminal.
- Other incomplete outcomes keep the current fail-closed failure behavior.

For the non-streaming path:

- Use the same execution metadata.
- Render a waiting-for-confirmation title/notice instead of an error title when `awaiting_user_input=True`.
- Add success reaction only for a genuinely completed outcome.

The streaming path must pass a continuation-boundary callback that flushes and closes the first turn's active text/reasoning blocks, so a confirmation question and the resumed answer cannot concatenate into one text block.

Add the required localized strings to `src/card/ui_text.py`.

**Step 5: Verify Task 2 and adjacent behavior**

Run:

```bash
uv run python -m pytest tests/test_acp_prompt_continuation.py tests/test_programming_completion_guards.py tests/test_programming_card_session.py tests/test_card_reducer_main.py tests/test_tui2acp_terminal_cleanup.py tests/test_acp_prompt_finalization.py -q
uv run python -m pytest tests/test_handlers.py -q -k "Programming or heartbeat"
uv run ruff check src/acp/continuation.py src/acp/outcome.py src/card/programming_adapter.py src/card/state/reducers/lifecycle.py src/card/ui_text.py src/feishu/handlers/programming.py tests/test_acp_prompt_continuation.py tests/test_programming_completion_guards.py tests/test_card_reducer_main.py tests/test_programming_card_session.py tests/test_tui2acp_terminal_cleanup.py
uv run python -m src.main --validate
git diff --check
```

Every failure or warning introduced by the change must be investigated. Do not skip or mask failures.

**Step 6: Commit Task 2 selectively**

Stage only the seven Task 2 files.

Commit subject:

```text
fix(programming): auto-resume unfinished plans
```

### Task 3: Final verification, project memory, and delivery

**Files:**

- Modify: `.Memory/2026-07-30.md`
- Modify: `.Memory/Abstract.md`

**Step 1: Run the completion gate**

Run the most relevant focused set first, then the repository fast suite because the change crosses ACP session orchestration, card lifecycle, and shared programming routing:

```bash
uv run python -m pytest tests/test_acp_prompt_continuation.py tests/test_programming_completion_guards.py tests/test_programming_card_session.py tests/test_card_reducer_main.py tests/test_tui2acp_terminal_cleanup.py tests/test_acp_prompt_finalization.py -q
uv run python -m pytest tests/ -q -m "not slow"
uv run python scripts/test_inventory.py tests/
uv run python -m src.main --validate
git diff --check
```

**Step 2: Review**

Perform task-scoped review after Tasks 1 and 2, then a broad final review against this plan and the original user request. Fix every Critical/Important finding and rerun covering tests before proceeding.

**Step 3: Update project memory without swallowing user work**

Append a detailed entry to `.Memory/2026-07-30.md` describing the log evidence, root cause, implementation, tests, and residual risk. Add a short dated summary to `.Memory/Abstract.md`.

These files already contain uncommitted user work. Preserve it exactly. Do not stage or commit unrelated hunks; if a clean selective memory commit cannot be made without absorbing user changes, leave the memory additions in the worktree and report that fact.

**Step 4: Re-read acceptance scope and push**

Confirm:

- Ordinary pending plans are auto-continued once.
- The exact screenshot path no longer immediately produces a failed card.
- Real permission gates remain intact.
- Both streaming and non-streaming paths are covered.
- All required tests and validation passed.

Push the completed commits to `origin/dev`:

```bash
git push origin dev
```

Verify local `dev` and `origin/dev` resolve to the same commit and report any intentionally preserved uncommitted files.
