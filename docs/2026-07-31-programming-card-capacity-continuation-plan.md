# Ordinary Programming Card Capacity Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `test-driven-development` for the behavior change. Execute this plan in the current `dev` checkout; the user explicitly prohibited worktrees. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve every ordinary-programming card record and the complete ordered main-Agent response when a long task reaches either the 100-block state cap or the 50-completed-tool cap.

**Architecture:** Keep the reducer caps as defense in depth for every card mode, but let `ProgrammingCardSession` project an incoming event without trimming and rotate before either cap would discard data. Capacity rotation creates and seeds a continuation `CardSession`, waits until its Feishu message is visible, atomically switches the rotator, and only then archives the old card; an adapter-level reentrant lock prevents concurrent flush/ticker/ACP events from writing the old card during this handoff. Main-Agent text is recorded at ingestion in an append-only transcript independent of every bounded `CardState`. An adapter-owned active-tool registry preserves tools omitted from a bounded continuation seed, while a permanently fenced timer manager guarantees archived sessions cannot be retained or rearmed by stale callbacks.

**Tech Stack:** Python 3.11+, frozen card dataclasses, CardKit delivery, `threading.RLock`, pytest, `uv`, ruff.

## Global Constraints

- Add failing regression tests before production changes and observe the expected failures.
- Rotate before reducer trimming for both `MAX_TOTAL_BLOCKS == 100` and `MAX_COMPLETED_TOOL_BLOCKS == 50`.
- Capacity archival must not append a new content block: the frozen header/footer and visible `new_message_id` URL button provide navigation without making a 100-block old card trim itself during `ARCHIVED`.
- A new continuation card must complete its first visible delivery before the current-session pointer changes and before the old card is archived.
- After the swap, all ACP, flush, image, plan, ticker, and terminal updates target only the new session.
- Preserve the full ordered main-text transcript independently of bounded card state; subagent text and system blocks do not become the final assistant response.
- Capacity rotation bypasses the semantic `session_max_rotations` truncation limit only for this ordinary-programming safety path; existing Spec/semantic rotation behavior remains unchanged.
- If a continuation cannot become visible, do not send a capacity-crossing event through the old reducer and do not silently claim terminal card delivery success.
- Do not implement production restart recovery, restart the service, or create a worktree.
- Run targeted tests first, then the relevant card/handler suite, validation, lint, and diff checks.
- Record the completed change in `.Memory/2026-07-31.md` and `.Memory/Abstract.md`, make one dedicated commit, and push `dev`.

---

### Task 1: Lock the regression contract in tests

**Files:**

- Modify: `tests/test_programming_card_session.py`
- Modify: `tests/test_session_rotator.py`

**Interfaces:**

- Consumes: existing `ProgrammingCardSession.start/on_event/on_text/get_final_text`, `CardDelivery`, and `SessionRotator.rotate`.
- Produces: failing tests for total-block rollover, completed-tool rollover, visible-before-freeze ordering, latest-card-only updates, transcript completeness, and capacity-rotation bypass of the semantic rotation ceiling.

- [x] **Step 1: Make the programming-card fake record API ordering**

Extend `MockClient` with an ordered `operations` list. Record `create_card`, `update_card`, and `update_element` calls, including the generated `card_id`, so tests can locate the continuation create and the old-card archived update.

- [x] **Step 2: Add a total-block regression using compressed failed-tool history**

Use real ACP `TOOL_CALL_START`/failed `TOOL_CALL_DONE` events until the state would exceed `MAX_TOTAL_BLOCKS`. Assert:

```python
old_session = pcs.session
# ... fill the old card to its safe boundary, then cross it ...
assert pcs.session is not old_session
assert old_session.state.metadata.frozen is True
assert any(block.block_id == "failed-0" for block in old_session.state.blocks)
assert "early-main-text" in pcs.get_final_text()
assert "late-main-text" in pcs.get_final_text()
```

Also assert the continuation `create_card` operation precedes the archived `update_card` for the old `card_id`, and that a later text delta updates only the new card.

- [x] **Step 3: Add a completed-tool regression at the exact 50/51 boundary**

Send `MAX_COMPLETED_TOOL_BLOCKS + 1` sequential successful tool calls. Assert the old frozen card retains all first 50 completed tool blocks (including the oldest), while the 51st tool is completed on the new current card and no sliding-window deletion occurred.

- [x] **Step 4: Add a rotation-ceiling regression**

Set `session_max_rotations` to one, cross the total-block boundary twice, and assert ordinary-programming capacity rotation still creates a third session while the default `SessionRotator.rotate()` tests remain capped.

- [x] **Step 5: Run RED tests**

Run:

```bash
uv run python -m pytest \
  tests/test_programming_card_session.py \
  tests/test_session_rotator.py -q
```

Expected: the new programming regressions fail because the adapter never rotates and `get_final_text()` loses text removed from the current `CardState`; existing rotator tests remain green.

---

### Task 2: Add a pure pre-trim capacity projection

**Files:**

- Modify: `src/card/state/reducer.py`
- Modify: `tests/test_reducers.py`

**Interfaces:**

- Produces: `card_state_requires_continuation(state, event, *, total_block_limit, completed_tool_limit) -> bool`.
- Preserves: `reduce_card_state(...)` continues to enforce both existing caps for non-programming callers.

- [x] **Step 1: Factor reduction from cap enforcement**

Keep one pure internal reduction path that can return the untrimmed next state. The public reducer calls it and then applies the existing completed-tool and total-block trims exactly as before.

- [x] **Step 2: Expose a side-effect-free projection helper**

Implement the equivalent of:

```python
def card_state_requires_continuation(
    state: CardState,
    event: CardEvent,
    *,
    total_block_limit: int = MAX_TOTAL_BLOCKS,
    completed_tool_limit: int = MAX_COMPLETED_TOOL_BLOCKS,
) -> bool:
    projected = _reduce_card_state_untrimmed(state, event)
    completed = sum(
        block.kind == "tool_call" and block.status == "completed"
        for block in projected.blocks
    )
    return (
        len(projected.blocks) > total_block_limit
        or completed > completed_tool_limit
    )
```

- [x] **Step 3: Verify reducer behavior**

Add focused tests proving the helper detects each limit without mutating the input and the normal reducer still trims for callers that do not rotate.

Run:

```bash
uv run python -m pytest tests/test_reducers.py tests/test_card_reducer_main.py -q
```

Expected: PASS.

---

### Task 3: Implement visible-first capacity rotation and the independent transcript

**Files:**

- Modify: `src/card/session/rotator.py`
- Modify: `src/card/session/core.py`
- Modify: `src/card/session/ttl.py`
- Modify: `src/card/session/_ttl_mixin.py`
- Modify: `src/card/timers/manager.py`
- Modify: `src/card/delivery/engine.py`
- Modify: `src/card/protocols.py`
- Modify: `src/card/events/factories.py`
- Modify: `src/card/state/reducers/lifecycle.py`
- Modify: `src/card/programming_adapter.py`
- Modify: `src/feishu/handlers/programming.py`
- Modify: `tests/test_programming_card_session.py`
- Modify: `tests/test_session_rotator.py`

**Interfaces:**

- `ProgrammingCardSession(..., session_factory: Callable[[CardMetadata], CardSession])` receives a handler-owned continuation factory that reuses chat, reply target, delivery, callbacks, and session configuration without copying bounded state.
- `SessionRotator.rotate(factory, *, enforce_max_rotations: bool = True, archive_with_hint: bool = True) -> CardSession | None` preserves both defaults; programming capacity rollover passes `False, False`.
- `ProgrammingCardSession.get_final_text() -> str` reads only the append-only main-text transcript.

- [x] **Step 1: Provide a continuation-session factory at the handler boundary**

Build the initial and continuation `CardSession` objects through one local handler factory using the same `chat_id`, `reply_to`, `delivery`, and `card_callbacks`. Pass that factory into `ProgrammingCardSession`; continuation metadata carries the actual first session's `session_started_at`.

- [x] **Step 2: Make the semantic rotation ceiling opt-out explicit**

Add keyword-only `enforce_max_rotations` and `archive_with_hint` flags, both defaulting to `True`. Only the programming capacity path bypasses the rotation count and suppresses the archival TextBlock; all existing semantic rotation callers keep their current behavior.

- [x] **Step 3: Serialize programming dispatch and rotate before trimming**

Add an adapter `RLock` around capacity projection, continuation preparation, atomic rotate, and final event dispatch. Rotate when the projected untrimmed state would exceed 100 blocks or 50 completed tools; also rotate before a new tool starts when the current card already owns 50 completed tools.

- [x] **Step 4: Prepare and verify the continuation before returning it to the rotator**

The factory passed to `rotate()` must:

```python
new_session = self._session_factory(continuation_metadata)
new_session.dispatch(CardEvent.started())
# replay the latest TASK_LIST_UPDATED and any active ToolBlock state
if not new_session.wait_delivery_idle(timeout=visibility_timeout):
    raise RuntimeError("continuation delivery did not become idle")
if not new_session.delivered_message_id:
    raise RuntimeError("continuation card is not visible")
return new_session
```

Only after this returns may `SessionRotator` swap the pointer and dispatch `ARCHIVED` to the old session. Reset per-card text/reasoning source maps after the swap; keep global turn counters and subagent metadata.

- [x] **Step 5: Preserve main text at ingestion**

Record every main-source `TEXT_CHUNK`/`on_text()` chunk exactly once with its logical block ID before bounded state reduction. `get_final_text()` joins those logical main blocks in arrival order. When `finish(fallback_text=...)` supplies the only main answer, record `_summary` in the same transcript. Never derive the transcript from `self._rotator.current.state`.

- [x] **Step 6: Fail visibly without trimming when continuation delivery fails**

Fence further capacity-crossing card events, retain transcript ingestion, and make terminal-delivery status report failure so the handler's existing full-text reply fallback runs. Do not dispatch the offending event to the full old state.

- [x] **Step 7: Run GREEN tests**

Run:

```bash
uv run python -m pytest \
  tests/test_programming_card_session.py \
  tests/test_session_rotator.py \
  tests/test_reducers.py \
  tests/test_card_session.py \
  tests/test_card_delivery_engine.py -q
```

Expected: PASS, including both new capacity regressions.

---

### Task 4: Verify integration, record the change, and publish one commit

**Files:**

- Modify: `.Memory/2026-07-31.md`
- Modify: `.Memory/Abstract.md`

**Interfaces:**

- Consumes: all implementation and test changes from Tasks 1–3.
- Produces: validation evidence, one dedicated commit, and an updated `origin/dev`.

- [x] **Step 1: Run related and expanded verification**

```bash
uv run python -m pytest \
  tests/test_programming_card_session.py \
  tests/test_programming_completion_guards.py \
  tests/test_card_session.py \
  tests/test_card_delivery_engine.py \
  tests/test_reducers.py \
  tests/test_handlers.py -q
uv run ruff check \
  src/card/programming_adapter.py \
  src/card/events/factories.py \
  src/card/session/rotator.py \
  src/card/state/reducer.py \
  src/card/state/reducers/lifecycle.py \
  src/feishu/handlers/programming.py \
  tests/test_programming_card_session.py \
  tests/test_session_rotator.py \
  tests/test_reducers.py
uv run python -m src.main --validate
git diff --check
```

- [x] **Step 2: Request an independent code review**

Review the complete task diff against all six user requirements, with special attention to concurrency, delivery failure, the 20-rotation bypass scope, transcript source attribution, and proof that no event updates an archived session.

- [x] **Step 3: Update project memory**

Add detailed change/reason/verification/risk entries to `.Memory/2026-07-31.md` and a roughly 20-character summary link under the 2026-07-31 heading in `.Memory/Abstract.md`.

- [x] **Step 4: Commit and push**

Read `docs/commit-message-guidelines.md`, stage only this task's files, then create one dedicated commit using the repository convention and push `dev`:

```bash
git push origin dev
```

- [x] **Step 5: Verify the remote**

Confirm `git status --short --branch` is clean and `git rev-parse dev` equals `git rev-parse origin/dev`.
