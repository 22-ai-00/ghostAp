# Codex Model Selection Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/codex` model selection responsive after restart and ensure a completed system command can never leave card actions permanently blocked.

**Architecture:** Replace event-counted system-command gating with run-id ownership so repeated `RUNNING` metadata events are idempotent. Keep the official Codex ACP adapter as the only model-capability authority, but warm its exact model/Effort matrix in the background at startup and retain successful Codex probes for 30 minutes; explicit refresh still invalidates the cache.

**Tech Stack:** Python 3.13, Pydantic task events, `TaskScheduler`, threading, pytest, Ruff, official ACP adapter.

## Global Constraints

- Use `uv` only; never use pip or conda.
- The official Codex ACP adapter remains the only authority for Codex model and Effort capabilities; do not restore `~/.codex/models_cache.json` fallback.
- Preserve the final `select_acp_model` mutation gate during a real `system_exit`; do not solve the incident by globally bypassing the gate.
- A repeated `RUNNING` event for the same `run_id` must not increase gate ownership more than once.
- Distinct active system runs in one chat must each retain ownership until their own terminal event.
- Codex successful model-probe TTL is exactly 1,800 seconds; other ACP tools remain at 300 seconds.
- Startup preheat is background best-effort, never delays service startup, and logs no credentials.
- Every behavior change follows RED → GREEN TDD and preserves unrelated user changes.

---

### Task 1: Make system-command gate ownership idempotent by run ID

**Files:**
- Modify: `src/feishu/control_plane.py`
- Modify: `tests/test_control_plane.py`

**Interfaces:**
- Consumes: `TaskEvent.run_id`, `TaskEvent.chat_id`, `TaskEvent.task_type`, and `TaskEvent.status`.
- Produces: `ControlPlane.is_system_cmd_inflight(chat_id) -> bool` with idempotent lifecycle semantics while retaining `_system_cmd_inflight_by_chat` as the existing count view used by `FeishuWSClient`.

- [x] **Step 1: Write failing repeated-event and overlap tests**

Add a real `TaskEvent` factory in `tests/test_control_plane.py`:

```python
def _event(
    run_id: str,
    status: TaskStatus,
    *,
    chat_id: str = "chat1",
    task_type: str = "system_help",
) -> TaskEvent:
    return TaskEvent(
        run_id=run_id,
        chat_id=chat_id,
        status=status,
        timestamp=1.0,
        name="process_message",
        task_type=task_type,
    )
```

Add tests proving:

```python
def test_same_run_repeated_running_is_idempotent():
    cp.on_scheduler_event(_event("run-1", TaskStatus.RUNNING))
    cp.on_scheduler_event(_event("run-1", TaskStatus.RUNNING))
    cp.on_scheduler_event(_event("run-1", TaskStatus.SUCCEEDED))
    assert cp.is_system_cmd_inflight("chat1") is False

def test_distinct_runs_release_independently():
    cp.on_scheduler_event(_event("run-1", TaskStatus.RUNNING))
    cp.on_scheduler_event(_event("run-2", TaskStatus.RUNNING))
    cp.on_scheduler_event(_event("run-1", TaskStatus.SUCCEEDED))
    assert cp.is_system_cmd_inflight("chat1") is True
    cp.on_scheduler_event(_event("run-2", TaskStatus.FAILED))
    assert cp.is_system_cmd_inflight("chat1") is False
```

Parameterize `SUCCEEDED`, `FAILED`, and `CANCELED` as terminal releases, and assert a different chat is unaffected.

- [x] **Step 2: Run the gate regression tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest \
  tests/test_control_plane.py -q
```

Expected: the repeated-`RUNNING` test fails because the old counter remains at one after a single terminal event.

- [x] **Step 3: Implement run-id ownership**

In `ControlPlane.__init__`, add:

```python
self._system_cmd_runs_by_chat: dict[str, set[str]] = {}
```

In `on_scheduler_event`, for `system_help` and `system_exit`:

```python
terminal = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELED,
}
with self._system_cmd_gate_lock:
    runs = self._system_cmd_runs_by_chat.setdefault(ev.chat_id, set())
    if ev.status == TaskStatus.RUNNING:
        runs.add(ev.run_id)
    elif ev.status in terminal:
        runs.discard(ev.run_id)
    if runs:
        self._system_cmd_inflight_by_chat[ev.chat_id] = len(runs)
    else:
        self._system_cmd_runs_by_chat.pop(ev.chat_id, None)
        self._system_cmd_inflight_by_chat.pop(ev.chat_id, None)
```

Do not change the gate action whitelist and do not allow final model selection through an active exit gate.

- [x] **Step 4: Run focused and adjacent gate tests and verify GREEN**

Run:

```bash
UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest \
  tests/test_control_plane.py \
  tests/test_button_gate_and_dedupe.py \
  tests/test_ws_client_patch.py -q
```

Expected: all tests pass with no new warnings or errors.

- [x] **Step 5: Commit Task 1**

```bash
git add src/feishu/control_plane.py tests/test_control_plane.py
git commit -m "fix(feishu): release system command gate by run"
```

### Task 2: Preheat and retain the official Codex model matrix

**Files:**
- Modify: `src/acp/helper.py`
- Modify: `src/main.py`
- Modify: `tests/test_acp_model_probe_timeout.py`
- Modify: `tests/test_restart_script.py`

**Interfaces:**
- Consumes: existing `fetch_acp_models(tool_name, cwd, current_model=None, probe_timeout=None)`.
- Produces: `kickoff_acp_model_preheat(tool_names: list[str], cwd: str) -> threading.Thread | None`.
- Produces: `_positive_probe_cache_ttl(tool_name: str) -> float`, returning `1800.0` for Codex and `300.0` otherwise.

- [x] **Step 1: Write failing TTL and background-preheat tests**

In `tests/test_acp_model_probe_timeout.py`, add:

```python
def test_codex_success_cache_remains_fresh_for_thirty_minutes(monkeypatch):
    key = _helper_mod._probe_key("codex", "/repo")
    _helper_mod._acp_probe_cache[key] = (
        _helper_mod._time.time() - 600,
        [ACPModelOption(name="gpt-test", description="test", is_default=True)],
    )

    async def probe_must_not_run(*_args, **_kwargs):
        raise AssertionError("ten-minute-old Codex cache should still be fresh")

    monkeypatch.setattr(_helper_mod, "probe_acp_models", probe_must_not_run)
    assert [m.name for m in fetch_acp_models("codex", cwd="/repo")] == ["gpt-test"]
```

Add a background test that monkeypatches `fetch_acp_models`, calls `kickoff_acp_model_preheat(["codex", "codex"], "/repo")`, joins the returned thread with a bounded timeout, and asserts Codex was fetched exactly once with `cwd="/repo"`.

In `tests/test_restart_script.py`, add a startup-wiring contract:

```python
def test_application_startup_preheats_codex_model_capabilities():
    main_source = (ROOT / "src" / "main.py").read_text(encoding="utf-8")
    assert "kickoff_acp_model_preheat" in main_source
    assert 'kickoff_acp_model_preheat(["codex"], cwd=os.getcwd())' in main_source
```

- [x] **Step 2: Run the performance regression tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest \
  tests/test_acp_model_probe_timeout.py \
  -k "thirty_minutes or background_preheat" -q
```

Expected: the TTL test fails because the generic cache expires after 300 seconds, and the preheat API is absent.

- [x] **Step 3: Implement Codex TTL and best-effort background preheat**

In `src/acp/helper.py`, define:

```python
_ACP_PROBE_CACHE_TTL = 300
_CODEX_PROBE_CACHE_TTL = 1800

def _positive_probe_cache_ttl(tool_name: str) -> float:
    return float(
        _CODEX_PROBE_CACHE_TTL
        if str(tool_name or "").strip().lower() == "codex"
        else _ACP_PROBE_CACHE_TTL
    )
```

Use `_positive_probe_cache_ttl(tool_name)` in `_get_cached_probe`.

Add `kickoff_acp_model_preheat` that:

- strips and deduplicates tool names while preserving order;
- returns `None` for an empty list;
- starts one daemon thread named `acp-model-preheat`;
- sequentially calls `fetch_acp_models(tool, cwd=cwd)`;
- logs tool, count, outcome, and duration in milliseconds at INFO;
- catches each tool failure independently and never raises into startup;
- returns the started thread for bounded testing and diagnostics.

- [x] **Step 4: Wire Codex preheat into startup without blocking**

In the existing `acp_model_preheat_on_startup` block in `src/main.py`, retain Coco preheat and add:

```python
from .acp.helper import kickoff_acp_model_preheat

kickoff_acp_model_preheat(["codex"], cwd=os.getcwd())
```

The call must not join the thread and must remain inside the existing best-effort exception boundary.

- [x] **Step 5: Run focused and adjacent ACP/model tests and verify GREEN**

Run:

```bash
UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest \
  tests/test_acp_model_probe_timeout.py \
  tests/test_model_command.py \
  tests/test_model_cascade.py \
  tests/test_acp_provider_extensions.py \
  tests/test_restart_script.py -q
```

Expected: all tests pass; Codex capability extraction and adapter-only authority remain unchanged.

- [x] **Step 6: Commit Task 2**

```bash
git add \
  src/acp/helper.py src/main.py \
  tests/test_acp_model_probe_timeout.py tests/test_restart_script.py
git commit -m "perf(acp): preheat codex model capabilities"
```

### Task 3: Integration validation, memory, and delivery

**Files:**
- Modify: `.Memory/2026-07-24.md`
- Modify: `.Memory/Abstract.md`

**Interfaces:**
- Consumes: Tasks 1–2 commits and repository validation commands.
- Produces: durable incident record, validated `dev` commit range, pushed remote branch, and restarted service evidence.

- [x] **Step 1: Run expanded validation**

Run:

```bash
UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m pytest \
  tests/test_control_plane.py \
  tests/test_button_gate_and_dedupe.py \
  tests/test_ws_client_patch.py \
  tests/test_acp_model_probe_timeout.py \
  tests/test_model_command.py \
  tests/test_model_cascade.py \
  tests/test_acp_provider_extensions.py \
  tests/test_restart_script.py -q
UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run ruff check \
  src/feishu/control_plane.py src/acp/helper.py src/main.py \
  tests/test_control_plane.py tests/test_acp_model_probe_timeout.py
UV_CACHE_DIR=/tmp/ghostap-uv-cache uv run python -m src.main --validate
git diff --check
```

Expected: every command exits zero; the existing empty `slock_default_roles` notice is informational.

- [x] **Step 2: Record the incident and verification**

Append a detailed section to `.Memory/2026-07-24.md` covering:

- screenshot/log time correlation;
- repeated-`RUNNING` gate leak root cause;
- 7–17 second Codex cold-probe evidence;
- run-id ownership, 1,800-second Codex cache, and startup preheat;
- exact tests and validation results;
- remaining risk that a truly cold/failed official adapter probe still waits for the configured timeout before showing the retry card.

Add a roughly 20-character summary line under `2026-07-24` in `.Memory/Abstract.md`.

- [ ] **Step 3: Commit memory and plan**

```bash
git add \
  .Memory/2026-07-24.md \
  .Memory/Abstract.md \
  docs/superpowers/plans/2026-07-24-codex-model-selection-recovery.md
git commit -m "docs(memory): record codex selection recovery"
```

- [ ] **Step 4: Review, push, and restart**

After independent code review is clean:

```bash
git fetch origin dev
git rev-list --left-right --count origin/dev...dev
git push origin dev
./restart.sh rr
```

Require the fetched divergence to have zero commits on the remote-only side before push. Never force-push.

- [ ] **Step 5: Verify production restart**

Inspect the new `logs.log` tail and require:

- `[RESTART] remote worker begin`;
- dependency preparation success;
- `start begin`;
- `start spawned pid=... running=...`;
- `remote worker done status=0`;
- `启动飞书长连接服务`;
- no new traceback or fatal startup error after the new start marker;
- Codex background preheat completion log with model count and duration, or an explicit safe failure log followed by a healthy service.
