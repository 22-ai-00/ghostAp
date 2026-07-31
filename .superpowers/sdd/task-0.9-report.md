# Task 0.9 Report — managed-group registry and atomic lifecycle

## Status and commit

- Status: PARTIAL — lifecycle is hardened; Employee principal rotation production source is blocked
- Branch: `dev`
- Baseline: `7c34a2d9c4ec2a2d4051b9f69ea3340d3cbcbd5c`
- Commit: `073e9c32c36ad3d3d8f1c0e8062ba63aa8823623`
- Subject: `feat(trust): persist managed group lifecycle`
- Review fix: `4970193c` (`fix(trust): harden managed group lifecycle`)
- Second review fix: this commit (`fix(trust): close managed group crash windows`)
- Push: not performed

Task 0.9's Registry and Project/Team lifecycle are implemented and hardened.
The planned Employee principal-rotation production path is not complete because
the current workforce model has no trustworthy rotation event. Task 0.10
ingress/callback cutover is still missing and CP-T is not complete.

## RED evidence

The two brief-mandated files were created before `src/trust/registry.py`:

```text
uv run pytest tests/test_managed_group_registry.py \
  tests/test_managed_group_project_grant.py -q
```

Initial result: exit 2 during collection with two
`ModuleNotFoundError: No module named 'src.trust.registry'` errors.

After the registry-only GREEN, lifecycle RED was `7 failed`: the
ProjectChatService injection contract, Team registration, registry-failure
compensation, migration candidate, and GroupLedger guard did not exist.
Shared production composition RED was `2 failed`: ProjectHandler did not pass
the shared identity and no WS composition helper existed.

Self-review added a dangling-provision retry regression. It produced `1 failed`
because a later wall-clock timestamp was treated as conflicting stable intent.
The minimal fix keeps the original creation timestamp and compares only stable
provision facts. A second self-review RED produced `1 failed` because Settings
freezes `admin_user_ids` to a `frozenset`; composition now accepts exactly one
Owner from either serialized or frozen settings and fails closed otherwise. A
final `1 failed` RED proved a `None` candidate chat ID was stringified; raw
candidate types are now rejected before the external validator runs.

## GREEN and production wiring

- `src/trust/registry.py` persists a strict schema/version snapshot using a
  constructor-injected path, a leaf `RLock`, random `O_EXCL`/`O_NOFOLLOW`
  temporary file, file fsync, atomic replace, and parent fsync. Unsupported,
  malformed, duplicate-key, mismatched, or backward-revision data fails closed.
- The registry reuses Task 0.8 `ManagedGroupRecord` and `ProjectGrant`. It
  enforces one ACTIVE record and one grant per chat, monotonic revision,
  idempotent provision/register/adopt/rotation/tombstone, and tombstone
  dominance over stale candidates.
- FeishuWSClient composition creates/replays the sole Registry beside Project
  storage before handler construction. HandlerContext shares it with
  ProjectHandler/ProjectChatService and SlockHandler. No ingress or callback
  routing was changed.
- `/new-chat` and `/new-team` execute intent → Feishu create → durable bind →
  Registry ACTIVE → welcome/success. Registry or bind failure compensates local
  state, attempts deletion, sends no false success, and keeps deletion
  `False`/`None` residual groups untrusted with one clear Owner error.
- Confirmed or unknown Team dissolution writes a Registry tombstone; rejected
  deletion restores the local active team. Legacy names/descriptions,
  `allowed_chat_ids`, bound fields, and Slock markers never establish trust.
- Project migration candidates require bound chat ID, creation timestamp, and
  root. Import/adoption additionally requires an injected membership and
  receiving-bot validator. GroupLedger exposes only a fail-closed ACTIVE guard
  for Task 0.10; it is not wired into ingress in this task.

## Verification

```text
uv run pytest tests/test_managed_group_registry.py \
  tests/test_managed_group_project_grant.py \
  tests/test_project_chat/test_service.py -q
```

Result: `24 passed, 2 warnings in 3.55s`.

```text
uv run pytest tests/test_slock_*.py tests/autonomous/unit/test_group_* -q
```

Result: `2407 passed, 2 warnings in 250.87s`.

```text
uv run pytest tests/test_ws_client_routing.py tests/test_handlers.py -q
```

Result: `253 passed, 2 warnings in 9.89s`.

Touched-file Ruff: `All checks passed!`. `git diff --check`: passed with no
output. The warnings are the two pre-existing pinned Lark SDK Python 3.13
deprecations; none were suppressed.

## Not tested and risks

- Real Feishu membership/receiving-bot validation: `not_tested`.
- Real tenant/mobile delivery and residual-group manual cleanup: `not_tested`.
- Local tests use injected validators and mocked Lark deletion states; they are
  not recorded as tenant evidence.
- Task 0.10 still must resolve Registry trust before Ledger/image/project/
  classifier/session side effects and cut callbacks over to current revisions.
- CP-T remains open; this report does not claim it or CP-C-Managed complete.

## Review correction

The first commit was held locally and reviewed. The correction keeps Task 0.10
out of scope while closing the lifecycle and production-control findings.

### Additional RED evidence

- Registry batch: `6 failed` proved dual-instance lost updates, stale ACTIVE
  resurrection over a tombstone, post-replace parent-fsync memory/disk split,
  bool version acceptance, symlink following, and a timezone with no concrete
  UTC offset.
- Provision/Project batch: `4 failed` proved remote chat IDs were not durable,
  retries could create another group, and Project create/close ignored failed
  persistence.
- Team batch: `3 failed` proved a post-ACTIVE bootstrap error deleted the
  trusted group, pending revoke did not exist, and markers did not fsync file or
  parent; an additional ordering regression observed ACTIVE at delete dispatch.
- Production-control batch: `2 failed` proved no concrete Lark membership
  adapter and no Owner P2P adoption entry existed.

### Corrected behavior

- A stable lock file plus `flock` covers reload→validate→mutate→persist for
  every Registry disk transaction. Post-replace errors reread and compare the
  target before deciding whether callers may roll back. Reads also refresh, so
  pending revokes and tombstones fail closed across instances.
- Provision intents store one remote chat ID immediately after creation.
  Project and Team retries reuse it; a different chat conflicts. Project
  create/close now honor save failure, and compensation failures quarantine
  the reverse index and disclose the degraded state.
- Registry ACTIVE is the Team commit point. Later bootstrap/delivery failure
  keeps the group trusted. Dissolve persists pending revoke before deletion;
  rejection cancels it, while confirmed/unknown results complete a tombstone.
  Startup reconciles pending revokes before Slock marker restore. Marker writes
  use random exclusive temporary paths with file and parent fsync.
- `LarkChatClient.validate_managed_chat` uses the official SDK to confirm the
  current Bot and page Owner membership. API/permission/security-limit failures
  are UNKNOWN. Owner P2P `/access adopt-chat` uses exact Project resolution;
  `/access migration-status` exposes persisted ambiguous candidates; explicit
  main-Bot rotation remotely validates then CAS-updates matching records.
  Employee principal auto-rotation is intentionally not inferred.
- Bot-deleted events persist revocation before Project/Slock teardown and raise
  on failure for SDK redelivery. The duplicate GroupLedger ACTIVE predicate was
  removed; Task 0.10 must consume `EffectiveTrust` and remains unimplemented.

### Corrected verification

- Registry/Project/Slock target: `65 passed`.
- Expanded Registry/ProjectChat/ProjectManager target: `130 passed`.
- WS routing/handler compatibility: `351 passed, 17 subtests passed`.
- Touched-file Ruff and `git diff --check`: passed.
- Real Feishu tenant execution: `not_tested`; mocked SDK response tests are not
  represented as tenant evidence.

## Second review correction

The second review found crash windows that ordinary rollback tests did not
cover. The correction was developed with failing regressions first and keeps
ambiguous external outcomes fail-closed:

- Pending revoke recovery now durably archives or derives Slock marker state
  before remote deletion. Confirmed deletion retires the marker before the
  revoke completes; rejected deletion cancels only after a durable restore.
  Tombstoned/untrusted markers are never restored at startup.
- Owner adoption now validates remotely, durably binds Project state, then
  activates Registry. Its original binding snapshot is a restart-safe Project
  saga, so a crash between bind and activation can still compensate to the
  true pre-bind state. Quarantine and remote residual records are durable.
- Registry replace uncertainty is anchored before replace. A target
  parent-fsync failure raises a typed committed-but-uncertain result, hides all
  grants, and is reconciled before startup recovery. Project, adoption, and
  Slock preserve the remote group and do not report success or delete it.
- Feishu create uses a deterministic SDK `uuid`, but the implementation does
  not treat that as permanent idempotency. Provision intents persist
  prepared/dispatched/outcome-unknown/bound state and first dispatch time.
  Unknown outcomes preserve Project/Team residuals, retry the same UUID only
  inside Feishu's 10-hour window, and block blind creation after it expires.
- Migration first aggregates every legacy candidate by chat. Conflicting
  project/root bindings persist `ambiguous`; only unique consistent candidates
  are validated/imported. INVALID/UNKNOWN/AMBIGUOUS notifications go to the
  configured Owner `open_id`, are marked reported only after successful SDK
  delivery, and remain retryable after failure. Membership pagination rejects
  repeated tokens and stops at 100 pages.

### Remaining model blocker

Registry's expected-principal CAS primitive is covered, but production
Employee rotation is intentionally not wired. Workforce projection forbids
rebinding an existing principal/app identity, `EMPLOYEE_CREATED` managed groups
have no production creation path, and membership add/delete events prove only
membership—not credential rotation. A correct future path requires a
Journal-anchored principal-rebind saga with typed old/new principal, app,
agent, tenant and revision; PREPARED before remote validation, Registry CAS
limited to that employee's groups, and replayable COMMITTED/fail-closed states.
Until that source and saga exist, this acceptance item remains blocked rather
than being guessed from membership events.

### Second-correction verification

- Required Registry/ProjectGrant/ProjectChat command: `64 passed, 2 warnings`.
- Brief expanded Slock/autonomous-group command: `2407 passed, 2 warnings`.
- WS/handler/Lark/Slock compatibility command: `299 passed, 2 warnings`.
- Touched-file Ruff: `All checks passed!`; `git diff --check`: no output.
- The two warnings are the existing pinned Lark SDK Python 3.13 deprecations.
- Real Feishu tenant/mobile execution remains `not_tested`.
