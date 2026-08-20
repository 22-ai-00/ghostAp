# Programming Execution Window Rollover Implementation Plan

> Historical note (2026-08-20): GhostAP no longer adds an ACP permission or
> command-risk policy. Rollover preserves the provider's own permission model.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treat the configured ACP timeout as one renewable execution window so an unfinished task automatically resumes in a clean transport instead of failing after its first window.

**Architecture:** Add a transport-independent supervisor above `run_prompt_with_continuation()`. It rolls over only when a window really entered timeout finalization and the fail-closed assessment is still incomplete; normal user-input, cancellation, exception, and completed outcomes retain their current semantics. Each rollover retires the old transport, resumes its exact provider session through a compare-and-set manager operation, preserves the original task scope, and starts another bounded window. A configurable hard window ceiling prevents infinite resource consumption without confusing one window with the whole task.

**Tech Stack:** Python 3.13, dataclasses, Pydantic Settings, ACP session manager, pytest, Ruff, `uv`.

## Global Constraints

- Use `uv`; never use pip or conda.
- Preserve the handler → session → render/delivery import direction.
- Keep ACP permissions fail-closed; rollover grants no new authority.
- Never reuse a timed-out transport; resume the same provider session on a fresh transport.
- Refuse rollover if another session has taken ownership of the chat/project/thread key.
- Only timeout-finalized incomplete work may open a new window.
- User cancellation remains terminal.
- Default maximum is four execution windows; it is configurable from 1 through 24.
- All production behavior changes follow RED → GREEN TDD.

---

### Task 1: Execution-window supervisor contract

**Files:**
- Create: `src/acp/execution_windows.py`
- Modify: `src/acp/continuation.py`
- Modify: `src/acp/__init__.py`
- Test: `tests/test_acp_execution_windows.py`

**Interfaces:**
- Consumes: `PromptContinuationResult`, `PromptOutcome`, `PromptResult`, and a callback that executes one window.
- Produces: `run_prompt_across_execution_windows(...) -> PromptContinuationResult` plus the fields `execution_windows` and `window_limit_reached` on `PromptContinuationResult`.

- [ ] **Step 1: Write failing rollover tests**

```python
def test_timeout_finalized_incomplete_result_opens_fresh_window():
    first = _execution("session-1", incomplete=True, finalized=True)
    second = _execution("session-1", incomplete=False, finalized=False)
    resumed = _Session("session-1")

    result = run_prompt_across_execution_windows(
        _Session("session-1"),
        "original task",
        max_windows=4,
        execute_window=_queued_executor(first, second),
        resume_window=lambda old, session_id: resumed,
    )

    assert result.assessment.outcome is PromptOutcome.COMPLETED
    assert result.execution_windows == 2
    assert result.window_limit_reached is False
```

Also cover: a normal incomplete `end_turn` does not roll over; cancellation does not roll over; four finalized incomplete windows stop with `window_limit_reached=True`; the continuation prompt contains the original task and does not grant new authority.

- [ ] **Step 2: Run tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest tests/test_acp_execution_windows.py -q`

Expected: collection/import failure because `src.acp.execution_windows` does not exist.

- [ ] **Step 3: Implement the minimal supervisor**

```python
def run_prompt_across_execution_windows(
    session: SessionT,
    original_task: str,
    *,
    max_windows: int,
    execute_window: Callable[[SessionT, str], PromptContinuationResult],
    resume_window: Callable[[SessionT, str], SessionT],
    on_window_rollover: Callable[[int, int], None] | None = None,
) -> PromptContinuationResult:
    current = session
    prompt = original_task
    aggregate = None
    for window_index in range(1, max_windows + 1):
        execution = execute_window(current, prompt)
        aggregate = merge_window_execution(aggregate, execution, window_index)
        if execution.assessment.outcome is not PromptOutcome.INCOMPLETE:
            return aggregate
        if not execution.entered_finalization:
            return aggregate
        if window_index == max_windows:
            return replace(aggregate, window_limit_reached=True)
        session_id = require_resume_session_id(current)
        if on_window_rollover is not None:
            on_window_rollover(window_index + 1, max_windows)
        current = resume_window(current, session_id)
        prompt = build_execution_window_continuation_prompt(
            original_task, window_index + 1, max_windows, execution.assessment
        )
    raise AssertionError("unreachable")
```

Merge prior tool/plan/file evidence without changing the latest window's terminal assessment.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest tests/test_acp_execution_windows.py -q`

Expected: all tests pass.

### Task 2: Safe provider-session resume after transport retirement

**Files:**
- Modify: `src/acp/manager.py`
- Test: `tests/test_acp_manager.py`

**Interfaces:**
- Consumes: a chat/project/thread key that currently has no live session and the exact provider `session_id` retired by the preceding window.
- Produces: `resume_retired_session(..., session_id: str, startup_timeout: float) -> SyncSession`.

- [ ] **Step 1: Write failing manager tests**

```python
def test_resume_retired_session_loads_exact_provider_session(manager):
    with patch.object(manager, "_start_session_inner", return_value=resumed) as start:
        actual = manager.resume_retired_session(
            "chat-1", cwd="/repo", session_id="provider-session-1",
            startup_timeout=30, project_id="project-1"
        )
    assert actual is resumed
    assert start.call_args.args[3] == "provider-session-1"

def test_resume_retired_session_refuses_concurrent_owner(manager):
    manager._sessions[key] = concurrent_session
    with pytest.raises(RuntimeError, match="并发新会话已接管"):
        manager.resume_retired_session(...)
```

- [ ] **Step 2: Run tests and verify RED**

Run the two exact selectors in `tests/test_acp_manager.py`; expect `AttributeError` for the missing method.

- [ ] **Step 3: Implement the CAS resume operation**

Acquire the existing per-key startup lock, wait for the close tombstone, refuse any current owner, and call `_start_session_inner()` with the exact `session_id`. Do not call `ensure_session()`, because it may terminate a concurrent owner when the requested session ID differs.

- [ ] **Step 4: Run tests and verify GREEN**

Run the exact selectors again; expect both to pass.

### Task 3: Programming handler integration and user-visible semantics

**Files:**
- Modify: `src/config/settings.py`
- Modify: `src/feishu/handlers/programming.py`
- Modify: `src/card/ui_text.py`
- Test: `tests/test_programming_response.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `programming_max_execution_windows` and the supervisor from Task 1.
- Produces: automatic session rollover on timeout-finalized incomplete results, with one uninterrupted programming card and an accurate terminal message only if the multi-window ceiling is reached.

- [ ] **Step 1: Write failing handler and configuration tests**

```python
def test_programming_timeout_incomplete_resumes_and_finishes_in_second_window():
    first = _incomplete_execution(entered_finalization=True)
    second = _completed_execution("done")
    with patch("src.feishu.handlers.programming.run_prompt_with_continuation", side_effect=[first, second]):
        handler.handle_response(...)
    assert adapter.finished is True
    assert adapter.failed_text is None
    assert adapter.continuation_boundaries == 1
    manager.resume_retired_session.assert_called_once_with(..., session_id="session-1", ...)

def test_programming_window_limit_message_is_not_single_window_failure():
    execution = _incomplete_execution(
        entered_finalization=True,
        execution_windows=4,
        window_limit_reached=True,
    )
    ...
    assert "已自动续开 4 个执行窗口" in adapter.failed_text
```

Configuration tests assert the default is 4 and values outside 1..24 are rejected.

- [ ] **Step 2: Run tests and verify RED**

Run the exact new selectors; expect missing result fields/config and the first-window failure assertion.

- [ ] **Step 3: Integrate the supervisor**

Replace the handler's single `run_prompt_with_continuation()` call with `run_prompt_across_execution_windows()`. The one-window callback retains the existing finalization, continuation, event, and retirement hooks. The resume callback invokes `ACPSessionManager.resume_retired_session()`, updates `active_session`, resets per-session retirement bookkeeping, and re-registers thread context. The rollover callback calls `ProgrammingCardSession.begin_continuation_turn()` and logs only allowlisted window counts.

Add:

```python
programming_max_execution_windows: int = Field(default=4, ge=1, le=24)
```

Use a distinct terminal notice when the total window ceiling—not one ACP timeout—was exhausted.

- [ ] **Step 4: Run tests and verify GREEN**

Run the exact new selectors and then all of `tests/test_programming_response.py`; expect all to pass.

### Task 4: Regression, validation, and project memory

**Files:**
- Modify: `.Memory/2026-08-11.md`
- Modify: `.Memory/Abstract.md`
- Modify: `.Memory/Backlog.md`

**Interfaces:**
- Consumes: the completed behavior and verification outputs from Tasks 1–3.
- Produces: release evidence and removal or revision of B061 if its single-window forced-failure concern is resolved.

- [ ] **Step 1: Run focused ACP and programming tests**

Run: `UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest tests/test_acp_execution_windows.py tests/test_acp_finalization.py tests/test_acp_continuation.py tests/test_acp_manager.py tests/test_programming_response.py -q`

Expected: zero failures.

- [ ] **Step 2: Run shared regression and static validation**

Run the relevant ACP/card/session suites, Ruff on modified Python files, `uv run python -m src.main --validate`, `uv run python scripts/test_inventory.py tests/`, and `git diff --check`.

- [ ] **Step 3: Run the non-slow suite**

Run: `UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest tests/ -q -m "not slow"`

Expected: zero failures; investigate every failure rather than skipping it.

- [ ] **Step 4: Update project memory**

Record the root cause, exact architecture, RED/GREEN evidence, full regression result, configuration, and remaining operational risks. Update B061 to reflect that a single expired window now rotates safely; retain only any still-unproven real-tenant restart/long-wall-clock acceptance condition.
