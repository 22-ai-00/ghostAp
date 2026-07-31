# Card Footer Live Elapsed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every CardSession-backed Feishu task card keeps a live elapsed value in its footer on every card update, formatted as `HH:MM:SS` or `N天 HH:MM:SS`.

**Architecture:** Keep `CardMetadata.session_started_at` as the single monotonic start instant and `FooterState.duration_seconds` as the terminal frozen duration. Centralize the clock format in `src/utils/text.py`, with CardSession elapsed selection in `src/card/render/footer.py` and Workflow total elapsed rendering in `src/workflow_engine/renderer.py`; do not add a timer or additional delivery calls. Running renders recompute against the existing clock, while terminal and archived renders use their already-frozen duration.

**Tech Stack:** Python 3.11+, frozen dataclasses, CardSession renderer, pytest, Feishu Card JSON.

## Global Constraints

- Every CardSession-backed task footer displays elapsed time whenever the card renders.
- Workflow progress and completion total elapsed values use the same clock format.
- Running cards use `time.monotonic() - metadata.session_started_at`.
- Terminal cards use `footer.duration_seconds` and never continue increasing.
- Durations below one day use zero-padded `HH:MM:SS`.
- Durations of one day or more use `N天 HH:MM:SS`; days are the largest unit.
- The footer text is always `⏱ 用时 <duration>` in running and terminal states.
- No independent one-second heartbeat or extra Feishu update is introduced.
- Existing user changes in the working tree remain untouched.

---

### Task 1: UX preview and elapsed clock contract

**Files:**
- Create: `ux/card-footer-live-elapsed.html`
- Modify: `tests/test_footer_v2.py`
- Modify: `src/utils/text.py`
- Modify: `src/card/render/footer.py`

**Interfaces:**
- Consumes: `CardMetadata.session_started_at`, `CardMetadata.frozen_total_elapsed`, `FooterState.duration_seconds`, and `CardState.terminal`.
- Produces: `format_elapsed_clock(seconds: float) -> str`, returning `HH:MM:SS` or `N天 HH:MM:SS`.

- [x] **Step 1: Create the focused UX preview**

Create a static comparison showing the same footer position for a running card, a completed card, and a task longer than one day. Use notation-sized muted text and fixed-width numerals for `⏱ 用时 00:07:42`.

- [x] **Step 2: Write the failing formatter tests**

```python
@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00"),
        (59, "00:00:59"),
        (3661, "01:01:01"),
        (90061, "1天 01:01:01"),
    ],
)
def test_elapsed_clock_uses_zero_padded_hms_and_days(seconds, expected):
    assert format_elapsed_clock(seconds) == expected
```

- [x] **Step 3: Run the formatter test to verify RED**

Run:

```bash
uv run python -m pytest tests/test_footer_v2.py::test_elapsed_clock_uses_zero_padded_hms_and_days -q
```

Expected: collection fails because `format_elapsed_clock` is not defined.

- [x] **Step 4: Implement the minimal formatter**

```python
def format_elapsed_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}天 {clock}" if days else clock
```

- [x] **Step 5: Run the formatter test to verify GREEN**

Run:

```bash
uv run python -m pytest tests/test_footer_v2.py::test_elapsed_clock_uses_zero_padded_hms_and_days -q
```

Expected: the parametrized test passes.

### Task 2: Persistent live elapsed footer

**Files:**
- Modify: `tests/test_footer_v2.py`
- Modify: `src/card/render/footer.py`

**Interfaces:**
- Consumes: `_total_elapsed_from_session(state: CardState) -> float | None`.
- Produces: `_footer_elapsed_seconds(state: CardState, is_final_terminal: bool) -> float | None`, selecting frozen terminal duration or current live session duration.

- [x] **Step 1: Write failing running-card tests**

Add tests proving that a main programming card with no status/tool/model still renders `⏱ 用时 00:02:00`, and that a later render advances to `⏱ 用时 00:02:05` when the monotonic clock advances.

- [x] **Step 2: Write failing terminal/day tests**

Add tests proving that a completed card renders `⏱ 用时 00:00:58`, a day-scale card renders `⏱ 用时 1天 01:01:01`, and the terminal value does not depend on the current monotonic clock.

- [x] **Step 3: Run the new rendering tests to verify RED**

Run:

```bash
uv run python -m pytest tests/test_footer_v2.py -q
```

Expected: the new running main-card and normalized terminal footer assertions fail against the existing engine-specific/compact duration behavior.

- [x] **Step 4: Implement the shared elapsed selection**

Use `footer.duration_seconds` for final terminal states. Otherwise use `_total_elapsed_from_session(state)` for every CardSession card, falling back to `footer.progress_started_at` only for legacy cards without a session start. Add `⏱ 用时 <formatted>` to the existing metadata line, or render it as the sole footer metadata when no tool/model exists.

- [x] **Step 5: Remove engine-specific duration branches**

Delete the Spec-only and subagent-only running elapsed branches. Preserve subagent two-line density by keeping elapsed on the tool/model metadata line.

- [x] **Step 6: Run footer tests to verify GREEN**

Run:

```bash
uv run python -m pytest tests/test_footer_v2.py tests/test_card_render_components.py tests/test_card_renderer.py -q
```

Expected: all selected footer and renderer tests pass with the new fixed-width contract.

### Task 3: Pipeline regression and project record

**Files:**
- Modify: `src/workflow_engine/renderer.py`
- Modify: `tests/test_workflow_renderer.py`
- Modify: `.Memory/2026-07-30.md`
- Modify: `.Memory/Abstract.md`

**Interfaces:**
- Consumes: the shared footer rendering contract from Tasks 1–2.
- Produces: regression evidence and a dated project decision record.

- [x] **Step 1: Run CardSession and engine-adjacent regression**

Run:

```bash
uv run python -m pytest tests/test_card_session.py tests/test_programming_card_session.py tests/test_card_pipeline_integration.py tests/test_spec_renderer_callbacks.py -q
```

Expected: all tests pass.

- [x] **Step 2: Run touched-file lint and repository validation**

Run:

```bash
uv run ruff check src/card/render/footer.py tests/test_footer_v2.py
uv run python -m src.main --validate
git diff --check
```

Expected: all commands exit successfully.

- [x] **Step 3: Record the behavior and evidence**

Append a detailed entry to `.Memory/2026-07-30.md` describing the shared monotonic start, fixed terminal duration, day-aware format, UX preview, and exact passing test commands. Add a concise dated pointer to `.Memory/Abstract.md`.

- [x] **Step 4: Re-read the user acceptance scope**

Confirm that running and terminal footers both always display time, repeated renders advance live time, the format is `HH:MM:SS`, and days are the largest unit.
