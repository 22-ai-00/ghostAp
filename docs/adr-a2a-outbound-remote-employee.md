# ADR: outbound A2A remote Employee pilot

## Status

Accepted for an opt-in development pilot. Production Team and Workflow pools
remain local until the recovery and local-verification gates below are wired.

## Decision

GhostAP adopts A2A specification release `1.0.1`, wire protocol `1.0`, and the
official Python SDK `1.1.2` as a new outbound remote-Agent adapter. A2A does not
replace ACP: a persistent Employee remains the Agent identity and continues to
use ACP for its internal coding-tool session.

The domain contract is SDK-free. Every attempt freezes the locally admitted
Agent ID, tenant, exact Card URL, exact endpoint, JSON-RPC binding, wire version,
Card digest, credential reference, message ID, and context ID. The instruction
is stored in the encrypted Blob store; Journal events contain only its Blob
reference.

The local Journal remains the sole source of truth. `prepared` and `executing`
must both be durable and anchored before `SendStreamingMessage`. The first
server task ID is bound to the frozen attempt before any observation is exposed.
After restart, a known task ID is reconciled with Subscribe/GetTask and is never
sent again. An executing attempt without a task ID is an unknown outcome; it is
not treated as an unsent request.

A2A `COMPLETED` maps only to local `CLAIMED_COMPLETED`. Artifact publication,
schema/evidence checks, verifier approval, and effect finalization are required
before a Team assignment or run may complete.

## Pilot boundary

- One fixed, locally trusted Card and endpoint; no open discovery.
- Outbound A2A protocol 1.0 over JSON-RPC 2.0, with streaming, GetTask,
  Subscribe, and Cancel.
- Text and bounded structured data only. Raw bytes and URL Parts are rejected.
- Credentials are resolved by the adapter after Card validation and are never
  placed in prompts, messages, artifacts, Card caches, or Journal payloads.
- No inbound A2A server, push webhook, gRPC, extension, AGP, peer mesh, spawn,
  automatic pool expansion, or Workflow remote member in this phase.

The Phase 2 two-Employee review canary remains development-only until process
restart can rebuild remote handles, reconcile known tasks without SendMessage,
and prove that remote claimed completion passes local verification. Formal
inbound A2A support remains a separate later decision and cannot reuse an SDK
TaskStore as an alternative state machine.
