# GhostAP Product Contract

GhostAP is a Feishu/Lark-native Agent Department gateway for a trusted
engineering environment.

- Main Bot: control plane for projects, employees, teams, tasks, approvals, audit.
- Employee Bot: execution identity with its own Channel, history, memory and stop.
- Provider/model/engine: replaceable execution capability, shown under Advanced.
- Host Shell: privileged host execution, disabled or explicitly authorized; it is
  not an operating-system sandbox.
- Built-in employee profile: single-host and file-backed; it does not claim
  multi-replica linearizability or privileged-host rollback resistance.

## Public command truth

- Direct programming keeps the selected Agent's existing session, cancellation,
  retry and streaming-card semantics; the control plane does not add a planning
  or routing model call.
- `/hire <name>` creates a durable Employee Bot through the controlled
  tool/model/profile flow. Arbitrary `/hire --prompt` input is rejected.
- The standalone Autonomous Manager command surface is retired. Employee and team
  actually connected to production.
- Deep and Spec remain protected execution strategies rather than being rewritten
  as a shared workflow.

This contract describes product boundaries. Security controls, process isolation
and file-backed durability must not be presented as stronger guarantees than the
running deployment can prove.
