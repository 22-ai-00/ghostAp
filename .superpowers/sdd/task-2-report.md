# Task 2 Implementation Report

## Status

DONE

## Scope

- Added fail-closed completion classification for provider Goal lifecycle states
  and nested child-agent snapshots.
- Preserved finalization provenance across all prompt turns without changing the
  absolute deadline or one-continuation limit.
- Preserved unresolved child snapshots across repeated outer tool updates until
  an authoritative terminal update arrives for the same `source_id`.
- Added finalization-specific programming failure copy for streaming and
  non-streaming paths, with the paused-Goal claim gated on an exact trusted
  `goal.status == "paused"` snapshot.
- Added the required desktop and 360px red error-card preview before changing
  production copy.

## RED Evidence

Outcome and continuation RED:

```bash
UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest \
  tests/test_programming_completion_guards.py \
  tests/test_acp_prompt_continuation.py -q \
  -k "provider_goal or running_child"
```

```text
21 failed, 7 passed, 58 deselected

- active/paused/blocked/unknown Goals were classified as completed or entered
  the ordinary synthesized continuation path;
- completed outer waits erased or ignored unresolved child snapshots;
- PromptContinuationResult did not expose entered_finalization.
```

Handler provenance RED:

```bash
UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest \
  tests/test_tui2acp_terminal_cleanup.py \
  tests/contracts/test_direct_programming_lane.py -q \
  -k "finalization or provider_goal"
```

```text
4 failed, 4 passed, 42 deselected

- both streaming and non-streaming finalization results rendered the ordinary
  incomplete text and omitted the execution-window exhaustion cause;
- paused/blocked post-finalization results did not carry truthful timeout copy.
```

Self-review child-merge RED:

```text
1 failed, 32 deselected

- an unknown partial update for the same source_id replaced the previously
  observed running child snapshot before an authoritative terminal update.
```

## GREEN Evidence

First Goal/child/continuation selector after implementation:

```text
28 passed, 58 deselected
```

Handler provenance selector after copy alignment:

```text
8 passed, 42 deselected
```

Final required focused suite:

```bash
UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest \
  tests/test_programming_completion_guards.py \
  tests/test_acp_prompt_continuation.py \
  tests/test_tui2acp_terminal_cleanup.py \
  tests/contracts/test_direct_programming_lane.py -q
```

```text
137 passed, 2 warnings in 9.41s
```

Changed-file quality gate:

```bash
UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run ruff check \
  src/acp/outcome.py src/acp/continuation.py \
  src/feishu/handlers/programming.py src/card/ui_text.py \
  tests/test_programming_completion_guards.py \
  tests/test_acp_prompt_continuation.py \
  tests/test_tui2acp_terminal_cleanup.py \
  tests/contracts/test_direct_programming_lane.py
```

```text
All checks passed!
git diff --check: passed
```

## Contract Coverage

- Cancellation and non-`end_turn` stop reasons retain priority over Goal state.
- `active`, `paused`, `blocked`, missing status, and unknown Goal statuses fail
  closed; only completed/no Goal reaches ordinary end-turn checks.
- Every observed Goal, including completed, disables GhostAP's synthesized
  continuation prompt.
- Paused/blocked Goals await user input only before finalization; finalization
  results never advertise continuation on a retired session.
- Nested child statuses `completed`, `failed`, and `cancelled` are terminal;
  missing, malformed, unknown, running, and pending snapshots stay unresolved.
- Multiple unresolved children count as one unresolved outer tool call.
- `entered_finalization` is accumulated with logical OR across turns.
- Finalization copy reports pending plan/tool truth and logs finalization, Goal,
  continuation, plan, and unresolved-tool fields for incident diagnosis.
- The paused copy appears only with an exact trusted paused Goal snapshot.

## Self-Review

- Re-read the Task 2 brief against the final diff and confirmed no new card
  state, renderer dependency, button, color, deadline, or continuation count.
- Found one merge subtlety during review: a non-terminal unknown child update
  could replace a previously known running snapshot. Added a failing regression
  first, then retained running/pending truth until terminal proof arrives.
- Confirmed the preview contains both exact copy contracts at desktop and 360px.
- Confirmed `.Memory/2026-08-05.md` and `docs/superpowers/` remain untouched by
  this task and are excluded from staging.

## Commit

- Baseline: `84f84c8d`
- Subject: `fix(programming): report goal timeout truthfully`
- Exact resulting SHA is reported after the report-containing commit.

## Concerns

- The focused tests emit two existing dependency deprecation warnings from
  `lark_channel`; they do not affect the selected Task 2 results.
- No Task 2 correctness or security concern remains open.
