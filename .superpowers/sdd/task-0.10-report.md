# Task 0.10 Report — ingress/callback trust-zone cutover

## Status

- Status: COMPLETE for the Task 0.10 local acceptance scope
- Branch: `dev`
- Push: not performed
- Real Feishu tenant/mobile execution: `not_tested`

Nested review follow-up is complete. Two Critical and three Important findings
were reproduced with RED tests and fixed before handoff.

Ingress and card callbacks now resolve durable Registry trust before business
side effects. Managed Owner requests and server-correlated Employee
continuations use the existing Direct/Deep/Spec/Worktree/Workflow/Team/Slock
routes without legacy enrollment prompts. External groups, unknown members,
stale revisions, and uncorrelated Employee Bot traffic fail closed.

## RED/GREEN evidence

The first tests-first batch produced 13 failures before the trust cutover and
then passed. The expanded P0 batch produced 10 failures / 15 passes, followed
by 7 failures / 20 passes after adding exact Outbox and membership transport
contracts. The completed focused suite plus Resolver is 35 passes.

The failures proved that the old path parsed content and scheduled work before
durable trust, accepted missing card revisions, lacked EffectiveTrust in the
Dispatcher/ActivationGuard, trusted Employee identity without causal Outbox
evidence, and restored runtime dependencies in the wrong order.

The review-fix batch initially produced 8 focused failures (with 32 existing
passes); the production-composed Owner P2P control regression independently
failed once. The completed review-focused suite is 45 passes. Fixes cover
Registry-only topic/bound-chat/image project resolution, current-trust checks
after recognition and before every multi-task step, stale-card refresh without
action dispatch, immutable CardSession revisions, and Owner P2P `/status` with
a production Registry.

## Production behavior

- `ManagedGroupRegistry.trust_snapshot()` reads the ACTIVE group and matching
  ProjectGrant in one disk transaction. Registry reconciliation precedes
  Registry-filtered Slock restore, Employee runtime recovery, membership audit,
  and the first main-Bot WebSocket connection.
- Main message ingress resolves `EffectiveTrust` before content parsing,
  GroupLedger, image handling, project lookup, scheduler submission, Slock,
  Shell, or handlers. Managed project/root facts come only from the Registry
  grant and are checked against ProjectManager before dispatch.
- Managed groups cannot invoke Host Shell or administrator routes. Explicit
  shell/admin actions use the production `ActionMatrix`; external/unknown
  traffic is rejected by WS, Dispatcher, and passive ActivationGuard fences.
- Employee continuation requires a server-reported parent/root message that
  exactly matches a terminal Outbox delivery binding, the current READY
  employee/app/generation/connection, and one anchored collaboration
  publication. SDK payload fields never establish causality. Employee-channel
  ambient Bot traffic is ignored.
- Membership events use a separate gate: ACTIVE managed Registry group plus
  exact projected employee, principal/app, Channel generation, and connection.
  Membership events never infer principal rotation.
- Interactive card production stamps both group and grant revisions in static
  BaseHandler cards and CardSession/CardDelivery payloads. Missing or stale
  revisions fail before dedup, project lookup, scheduler, chat lock, or
  `ActionDispatcher`. Trust is resolved again immediately before message/card
  dispatch, so rotation or tombstone after enqueue is fenced.
- Legacy test fixtures explicitly opt out of the real Registry. Production
  remains Registry-authoritative. Direct/Deep/Spec protected-lane call shapes
  and behavior remain unchanged when no EffectiveTrust is present.

## Verification

```text
Focused managed/external + Resolver: 35 passed
Brief Direct + protected contracts: 33 passed
Expanded Direct/ingress contracts: 163 passed
Protected lanes + focused + Card delivery: 94 passed
CardSession/CardDelivery adjacent: 64 passed
Autonomous runtime/Outbox/direct-mention/ingress: 94 passed
WS routing/reconnect/context/base-handler adjacent: 208 passed, 17 subtests passed
Touched-file Ruff: All checks passed
git diff --check: passed
```

Review-fix verification adds:

```text
Focused five-finding regressions: 45 passed
Core ingress/card/dispatcher cross-suite: 450 passed
CardSession full file: 125 passed
Employee status adjacent: 10 passed, 97 deselected
Touched-file Ruff: All checks passed
git diff --check: passed
```

The only warnings are the existing pinned Lark SDK Python 3.13 deprecations.

## Boundaries and residual risk

- Task 0.9 Employee principal rotation remains blocked by the workforce model.
  READY identity and membership events are not treated as rotation proof.
- The implementation keeps the documented personal/single-primary-process
  assumption. It adds no approval prompts, tenant policy, rollout gate, or new
  security framework.
- Real Feishu delivery, callback, membership, Channel reconnection, and mobile
  rendering are not represented as tenant evidence.
