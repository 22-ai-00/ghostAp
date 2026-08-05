# Managed Card and Slock Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore ordinary programming cards and legacy Slock teams after the managed-trust cutover without weakening callback revision checks.

**Architecture:** Card delivery will distinguish passive payloads from callback-bearing payloads, requiring the captured trust revision on every callback while allowing a passive managed card to contain no revision fields. Startup reconciliation will migrate locally anchored legacy Slock markers only after official remote Owner/Bot validation; untrusted markers remain durable and inactive instead of being destructively archived. Standalone Team grants use a stable `team:<name>` project identity and route through the restored Slock engine without requiring a legacy `ProjectContext`.

**Tech Stack:** Python 3.13, pytest, lark-oapi, existing `ManagedGroupRegistry`, `SlockEngineManager`, and `CardDelivery`.

## Global Constraints

- Use `uv` for every Python and test command.
- Registry remains the only trust source; marker data alone never authorizes ingress.
- `UNKNOWN` or `INVALID` remote validation never restores a Team.
- Tombstoned Registry records are never reactivated.
- Every behavior change starts with a failing regression test.

---

### Task 1: Passive managed-card delivery

**Files:**
- Modify: `tests/test_external_group_ingress.py`
- Modify: `src/card/delivery/engine.py`

**Interfaces:**
- Consumes: `CardDelivery.deliver(session_id, chat_id, rendered)` and `bind_managed_trust_revisions(...)`.
- Produces: callback-aware trust validation inside `CardDelivery._transform_rendered_payload(...)`.

- [ ] **Step 1: Write the failing passive-card regression**

```python
def test_managed_delivery_allows_passive_card_without_callback_stamps() -> None:
    client = MagicMock()
    client.create_card.return_value = ("om_card", "om_card")
    delivery = CardDelivery(
        client,
        registry=MagicMock(),
        payload_transform=lambda _chat_id, card: bind_managed_trust_revisions(
            card, group_revision=7, grant_revision=11
        ),
        trust_revision_provider=lambda _chat_id: (7, 11),
    )
    rendered = RenderedCard(
        _card_json={"schema": "2.0", "body": {"elements": [{"tag": "markdown", "content": "working"}]}},
        structure_signature="stable",
    )

    outcomes = delivery.deliver("session-passive", GROUP_ID, [rendered])

    assert [outcome.kind for outcome in outcomes] == ["applied"]
    client.create_card.assert_called_once()
```

- [ ] **Step 2: Run the test and verify the current mismatch rejection**

Run: `uv run pytest tests/test_external_group_ingress.py::test_managed_delivery_allows_passive_card_without_callback_stamps -q`

Expected: FAIL because the outcome is `rejected` and `create_card` is not called.

- [ ] **Step 3: Validate callback values rather than requiring a stamp on passive payloads**

Traverse every mapping, collect all revision pairs, and separately collect every `value` mapping containing `action` or `action_id`. For managed sessions require each callback value to contain exactly the captured `(group_revision, grant_revision)`; permit zero callback values only when no revision pair appears elsewhere. Preserve rejection of stale, mixed, or unmanaged stamps.

- [ ] **Step 4: Run passive, stale-revision, and external-ingress tests**

Run: `uv run pytest tests/test_external_group_ingress.py -q`

Expected: all tests PASS.

### Task 2: Legacy Slock marker migration and routing

**Files:**
- Modify: `tests/test_slock_runtime_restore.py`
- Modify: `tests/test_managed_group_ingress.py`
- Modify: `src/slock_engine/models.py`
- Modify: `src/slock_engine/engine.py`
- Modify: `src/slock_engine/manager.py`
- Modify: `src/feishu/handlers/slock.py`
- Modify: `src/feishu/ws_client.py`

**Interfaces:**
- Produces: `SlockEngineManager.managed_group_migration_candidates(...) -> tuple[dict, ...]`.
- Produces: marker field `project_id`, defaulting legacy markers to `team:<team_name>`.
- Consumes: `LarkChatClient.validate_managed_chat(chat_id, owner_id)` and `ManagedGroupRegistry.import_candidate(...)`.

- [ ] **Step 1: Add failing tests for non-destructive filtering and startup migration**

Add tests proving that `restore_from_disk(..., managed_group_active=lambda _: False)` leaves the active marker in place, that a remotely `VALID` legacy marker is imported before restore, and that `UNKNOWN`/tombstoned groups are not imported.

- [ ] **Step 2: Add the failing standalone-Team ingress test**

Construct managed trust with `project_id="team:Alpha"`, make `get_project_for_chat` return `None`, mark the Slock manager chat active, and assert `_process_message_async` reaches `_dispatch_message_logic` with `project=None`.

- [ ] **Step 3: Run the new tests and verify red failures**

Run: `uv run pytest tests/test_slock_runtime_restore.py tests/test_managed_group_ingress.py -q`

Expected: focused new tests FAIL because markers are archived, no migration candidates exist, and standalone Team ingress returns early.

- [ ] **Step 4: Implement marker metadata and safe candidate discovery**

Persist `project_id` from `SlockChannel` into `.slock_channel.json`. Candidate discovery accepts canonical active markers, plus post-cutover archived markers that have no active marker so the cutover regression can self-heal; it only returns parsed local facts and never grants trust. Change the Registry predicate failure path in `restore_from_disk` to skip without archiving.

- [ ] **Step 5: Import only remotely validated candidates**

Before Slock restore, skip any Registry-known chat, validate the receiving Bot and configured Owner through `LarkChatClient`, restore a qualifying cutover archive to the canonical marker when necessary, and call `ManagedGroupRegistry.import_candidate`. Record non-VALID migration dispositions and leave them inactive.

- [ ] **Step 6: Permit standalone managed Team routing**

When a Registry grant points at `team:<name>`, accept `project=None` only if the Slock manager confirms that exact chat is restored. Continue rejecting a missing or mismatched real project for every non-Team managed group.

- [ ] **Step 7: Run focused and adjacent tests**

Run: `uv run pytest tests/test_slock_runtime_restore.py tests/test_managed_group_ingress.py tests/test_external_group_ingress.py tests/test_slock_new_team.py tests/test_ws_client_reconnect.py -q`

Expected: all tests PASS.

### Task 3: Verification, state repair, and service recovery

**Files:**
- Modify: `.Memory/2026-08-02.md`
- Modify: `.Memory/Abstract.md`

**Interfaces:**
- Consumes: the fixed startup migration and existing `restart.sh` readiness checks.
- Produces: restored production Slock marker/Registry state and a running GhostAP process.

- [ ] **Step 1: Run lint and configuration validation**

Run: `uv run ruff check <touched Python files>`

Run: `uv run python -m src.main --validate`

Run: `git diff --check`

Expected: all commands exit 0 with no new warnings or failures.

- [ ] **Step 2: Record the incident and evidence**

Document the 08:58 card rejection, the 08-01 marker archival, root causes, exact tests, state repair, and remaining risks in `.Memory/2026-08-02.md`; add a dated one-line index entry to `.Memory/Abstract.md`.

- [ ] **Step 3: Restore and restart**

Run the fixed service startup so the remotely validated archived Team marker is restored and imported, then use `./restart.sh restart` if needed. Confirm `.ghostap.pid` is live, readiness is published, Slock restore is logged, and no `managed card payload trust revision mismatch` occurs on the first managed card.
