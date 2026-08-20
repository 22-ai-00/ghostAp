# ADR: Employee runtime deployment profiles

Status: Superseded (2026-08-20)

The execution-isolation parts of this ADR are no longer active. Employee
processes now run directly on the host with the inherited environment; GhostAP
does not install Bubblewrap, Seatbelt, attestation, or an execution policy.
Journal durability, credential integrity, and Feishu identity checks remain
data/transport contracts rather than an execution sandbox.

## Context

GhostAP contains two different sets of employee-runtime capabilities:

- a built-in, bounded single-host runtime that starts from local durable state;
- optional external trust, acceptance, and isolation components intended for a
  hardened multi-replica deployment.

Treating the second set as an unfinished local code fix made the maintenance
Backlog misleading. Local tests cannot manufacture an independent rollback
witness, a real tenant QA signature, a host namespace proof, or a network
egress policy. Conversely, those missing external controls do not describe the
current built-in runtime contract.

## Decision

GhostAP documents two explicit profiles.

### Built-in single-host profile

This is the current default when `AUTONOMOUS_VISIBLE_EMPLOYEE_LIMIT` is greater
than zero (default `8`). It uses:

- locally generated runtime keys and an encrypted credential Vault;
- a local `FileAnchor`, Journal, and main-Bot send audit;
- direct host execution with permissions delegated to the selected backend;
- local automated tests plus operator-owned real-tenant acceptance.

This profile does not claim resistance to a privileged host administrator
rolling back local state. It does not claim cross-replica linearizability,
Feishu-only egress enforcement, immutable workload provenance, or completion
of real-tenant desktop/mobile/1-10-50 Bot soak acceptance. Operators that need
those properties must not infer them from a green local test suite.

### Hardened multi-replica profile

This section records an older architecture proposal and is not an available
GhostAP profile. Its former requirements included:

- an independently administered monotonic witness/ledger or KMS/HSM-backed
  anchor shared by every main-Bot replica;
- a shared activation fence and main-Bot audit provider;
- immutable build/image digest and workload identity bound to tenant, release,
  instance, expiry, and recovery authority;
- external employee isolation and egress enforcement;
- independent QA execution and signed retention of the real-tenant release
  manifest, including Provisioning, employee send/receive, desktop/mobile
  Slash, main-Bot send-count, restart/reconnect, and 1/10/50 Bot soak evidence;
- fault injection covering provider loss, stale leases, rollback, split brain,
  reconnect, and recovery.

The repository may retain reusable broker, witness, evaluator, and attestation
code. Their presence is not evidence that this profile is active. Production
composition must explicitly consume them before the hardened profile can be
claimed.

## Related local lifecycle decisions

The legacy standalone Autonomous Manager command surface is retired. Its
in-memory approvals and pre-Journal state mutations are not a second production
runtime and must never report that work was accepted. Those commands fail
closed with migration guidance. The supported production path is the
Journal-backed Team employee runtime: the main Bot manages employees through
`/hire` and `/fire`, while employee Bots expose their own `/task`, `/status`,
and `/memory` controls.

Employee App manifest evidence also distinguishes desired local configuration
from observed remote state. A new hire records only the desired manifest. An
administrator may start durable in-place reauthorization for the original App
from a private roster card; observed state is committed only after the official
Lark existing-App adapter returns the exact same App ID, desired manifest hash,
and trusted evidence source. Missing evidence, failure, timeout, mismatched
receipt fields, or interrupted recovery remains remote unknown and requires
action. Matching hashes without a trusted evidence source are not treated as
current. Reauthorization is durable single-flight per principal, App, and
manifest; recovery never starts a second remote flow behind an uncertain or
already committed result.

## Consequences

- B036-B040 are removed from the code-maintenance Backlog. Their unresolved
  external requirements live in this profile and in release runbooks.
- `AUTONOMOUS_VISIBLE_EMPLOYEE_LIMIT=0` remains the explicit kill switch for
  either profile.
- Real-tenant acceptance remains required before an operator calls a concrete
  deployment production-ready, but it is not forged or auto-signed by local
  code.
- Reintroducing mandatory external release trust is a product/architecture
  decision and requires changing production composition and its tests; it is
  not a one-line configuration change.
- Rolling out the legacy-role durable admission index requires old writers to
  be stopped (or a single-writer cutover). Older binaries do not understand the
  index authority and therefore cannot safely write concurrently during a
  mixed-version deployment.

## Persistent employee cutover status (2026-07-17)

The persistent Actor and model-led Team Coordinator are now the built-in
runtime defaults:

- `AUTONOMOUS_EMPLOYEE_RUNTIME_MODE=shadow` keeps the single legacy model call,
  derives the Actor bootstrap input without a second model call, and records
  digest-only comparison events. A mismatch is diagnostic and cannot alter the
  legacy result.
- `AUTONOMOUS_EMPLOYEE_RUNTIME_MODE=actor` and
  `AUTONOMOUS_TEAM_RUNTIME_MODE=coordinator` are explicit, non-fallback modes.
  Their automated contracts cover session reuse, restart, routing, knowledge,
  context partiality, Team recovery, selective wake, Outbox, and Fire.
- `AUTONOMOUS_EMPLOYEE_RUNTIME_MODE=actor` and
  `AUTONOMOUS_TEAM_RUNTIME_MODE=coordinator` are selected when the deployment
  does not override either setting. Production composition also treats absent
  fields on an older settings object as these persistent modes, so an upgrade
  cannot silently remain on the fixed v1 pipeline.
Emergency rollback is configuration-only: explicitly set the two legacy modes
and restart. The fixed Team pipeline and canonical employee one-shot path remain
temporarily available during migration; there is no automatic runtime fallback
from Actor/Coordinator failures.
