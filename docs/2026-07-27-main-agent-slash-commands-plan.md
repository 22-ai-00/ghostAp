# Main Agent Slash Commands Implementation Plan

> **Status:** Completed on 2026-07-27. Checked steps preserve the implementation and verification record.

**Goal:** Register GhostAP's supported primary commands in the Feishu Slash Command panel while preserving the existing Channel SDK message path and command router.

**Architecture:** Keep `lark-channel-sdk` as the inbound WebSocket transport. Add a main-agent command catalog that reuses the proven Slash v7 `lark-oapi` adapter and exact reconciler from employee provisioning, then launch one best-effort reconciliation worker before the main Channel connection loop. Slash selections continue to arrive as normal `/command args` messages and therefore use the existing `SlashCommandParser` and handlers.

**Tech Stack:** Python 3.11+, `lark-channel-sdk==1.1.0`, `lark-oapi==1.7.1`, pytest, uv

## Global Constraints

- Use only `uv`; never use pip or conda.
- Use `POST/GET/PATCH/DELETE /open-apis/application/v7/app_slash_commands` through the official `lark-oapi` client.
- A Feishu application may expose at most 100 Slash Commands.
- The main WebSocket connection must remain available when Slash scopes or the Slash API are unavailable.
- Do not register redundant compatibility aliases when a primary spelling exists.
- Existing `/command args` parsing and routing remain the execution source of truth.

---

### Task 1: Main-agent command catalog

**Files:**
- Create: `src/feishu/main_slash_commands.py`
- Test: `tests/test_main_slash_commands.py`

**Interfaces:**
- Consumes: `SlashCommand` from `src.autonomous.provisioning.slash_commands`.
- Produces: `MAIN_AGENT_COMMANDS: tuple[SlashCommand, ...]` and `reconcile_main_agent_slash_commands(client) -> VerifiedSlashState`.

- [x] **Step 1: Write failing catalog tests**

```python
def test_main_agent_catalog_is_unique_and_within_feishu_limit():
    canonical = [item.canonical() for item in MAIN_AGENT_COMMANDS]
    assert len(canonical) <= 100
    assert len({item.command for item in canonical}) == len(canonical)


def test_main_agent_catalog_covers_primary_supported_surfaces():
    names = {item.canonical().command for item in MAIN_AGENT_COMMANDS}
    assert {
        "help", "coco", "codex", "deep", "spec", "worktree", "wf",
        "projects", "slock", "hire", "role", "task", "status",
    } <= names
```

- [x] **Step 2: Run tests and verify RED**

Run: `uv run python -m pytest tests/test_main_slash_commands.py -q`

Expected: FAIL because `src.feishu.main_slash_commands` does not exist.

- [x] **Step 3: Implement the catalog and reconciliation entry**

```python
MAIN_AGENT_COMMANDS = (
    SlashCommand("/help", "查看 GhostAP 完整帮助"),
    SlashCommand("/codex", "进入 Codex 编程模式"),
    SlashCommand("/deep", "启动 Deep 深度任务"),
    SlashCommand("/spec", "启动 Spec 规格任务"),
    SlashCommand("/worktree", "启动 Worktree 隔离任务"),
    SlashCommand("/wf", "启动 Workflow 工作流"),
    SlashCommand("/projects", "查看项目列表"),
    SlashCommand("/slock", "创建或管理自主协作群"),
    SlashCommand("/hire", "雇佣主 Agent 员工"),
)


async def reconcile_main_agent_slash_commands(client):
    api = LarkSlashCommandAPI(client)
    return await SlashCommandReconciler(
        api,
        desired=MAIN_AGENT_COMMANDS,
    ).reconcile()
```

The complete tuple must include primary public commands from the existing System, Deep, Spec, Worktree, Workflow, and Slock routing surfaces, stay below 100 entries, and omit redundant aliases such as `/enter_codex`, `/end_codex`, long Workflow spellings, and one-letter Slock aliases.

- [x] **Step 4: Run tests and verify GREEN**

Run: `uv run python -m pytest tests/test_main_slash_commands.py -q`

Expected: PASS.

### Task 2: One-shot startup reconciliation

**Files:**
- Modify: `src/feishu/ws_client.py`
- Test: `tests/test_main_slash_commands.py`
- Test: `tests/test_ws_client_reconnect.py`

**Interfaces:**
- Consumes: `reconcile_main_agent_slash_commands(client)`.
- Produces: `FeishuWSClient._start_main_slash_command_sync() -> None`, which starts at most one daemon worker per client instance.

- [x] **Step 1: Write failing lifecycle tests**

```python
def test_main_slash_sync_starts_once_and_does_not_block_channel(monkeypatch):
    client = FeishuWSClient.__new__(FeishuWSClient)
    client._slash_command_sync_thread = None
    calls = []
    monkeypatch.setattr(client, "_sync_main_slash_commands", lambda: calls.append("sync"))

    client._start_main_slash_command_sync()
    first = client._slash_command_sync_thread
    first.join(timeout=1)
    client._start_main_slash_command_sync()

    assert calls == ["sync"]
    assert client._slash_command_sync_thread is first
```

Add a second test that makes reconciliation raise and verifies `_sync_main_slash_commands()` logs a scope/actionable warning without propagating.

- [x] **Step 2: Run lifecycle tests and verify RED**

Run: `uv run python -m pytest tests/test_main_slash_commands.py tests/test_ws_client_reconnect.py -q`

Expected: FAIL because the startup methods and thread state do not exist.

- [x] **Step 3: Implement best-effort startup sync**

```python
def _sync_main_slash_commands(self) -> None:
    try:
        verified = asyncio.run(
            reconcile_main_agent_slash_commands(self._get_api_client())
        )
    except Exception as exc:
        logger.warning(
            "Main Agent Slash Command sync skipped (%s); ensure "
            "application:app_slash_command:read/write are published",
            type(exc).__name__,
        )
        return
    logger.info(
        "Main Agent Slash Commands ready: total=%d created=%d updated=%d deleted=%d",
        len(verified.observed),
        len(verified.created),
        len(verified.updated),
        len(verified.deleted),
    )


def _start_main_slash_command_sync(self) -> None:
    if self._slash_command_sync_thread is not None:
        return
    thread = threading.Thread(
        target=self._sync_main_slash_commands,
        name="main-slash-command-sync",
        daemon=True,
    )
    self._slash_command_sync_thread = thread
    thread.start()
```

Initialize `_slash_command_sync_thread` in `__init__` and call the starter once before entering the reconnect loop. The worker is best-effort so a missing newly introduced scope cannot take the existing bot offline.

- [x] **Step 4: Run lifecycle tests and verify GREEN**

Run: `uv run python -m pytest tests/test_main_slash_commands.py tests/test_ws_client_reconnect.py -q`

Expected: PASS and the reconnect test still creates at least two Channel clients.

### Task 3: Operator documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `.Memory/2026-07-27.md`
- Modify: `.Memory/Abstract.md`

**Interfaces:**
- Consumes: official client version, scope, cache, and quantity limits.
- Produces: deployment instructions and durable project evidence.

- [x] **Step 1: Document required scopes and behavior**

Add `application:app_slash_command:read` and `application:app_slash_command:write` to the Feishu setup steps, state that an application version must be republished, and document PC 7.70+/mobile 7.71+, the 100-command limit, and the possible 5-minute registration plus 3-minute cache delay.

- [x] **Step 2: Run focused routing and adapter regressions**

Run:

```bash
uv run python -m pytest \
  tests/test_main_slash_commands.py \
  tests/test_ws_client_reconnect.py \
  tests/test_command_registry_contract.py \
  tests/autonomous/contract/test_slash_lark_contract.py \
  tests/autonomous/integration/test_slash_reconciliation.py -q
```

Expected: PASS.

- [x] **Step 3: Run quality and configuration gates**

Run:

```bash
uv run ruff check src/feishu/main_slash_commands.py src/feishu/ws_client.py tests/test_main_slash_commands.py
uv run python -m src.main --validate
git diff --check
```

Expected: all commands exit 0.

- [x] **Step 4: Run the repository non-slow suite**

Run: `uv run python -m pytest tests/ -q -m "not slow"`

Expected: PASS with no unexplained failures or missing terminal summary.

- [x] **Step 5: Record evidence**

Append the official contract, catalog policy, startup failure behavior, exact commands changed, test results, and remaining real-tenant cache/permission verification risk to `.Memory/2026-07-27.md`; add a dated summary line to `.Memory/Abstract.md`.
