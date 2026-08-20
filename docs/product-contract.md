# GhostAP Product Contract

GhostAP is a Feishu/Lark-native Agent Department gateway for a trusted
engineering environment.

- Main Bot: control plane for projects, employees, teams, tasks, approvals, audit.
- Employee Bot: execution identity with its own Channel, history, memory and stop.
- Provider/model/engine: replaceable execution capability, shown under Advanced.
- Host Shell: direct host execution without a GhostAP command policy or
  operating-system sandbox.
- Built-in employee profile: single-host and file-backed; it does not claim
  multi-replica linearizability or privileged-host rollback resistance.

## Public command truth

- Direct programming keeps the selected Agent's existing session, cancellation,
  retry and streaming-card semantics; the control plane does not add a planning
  or routing model call.
- Direct, Deep and Spec automatically advance each accepted task to a successful
  or explicit failed terminal state. Missing ordinary choices use the saved or
  recommended default; bounded recovery does not add confirmation or resume gates.
- Workflow accepts work through one owner-confirmed Agent Pool containing
  1-8 `tool+model` Agents. After that single confirmation, the pool and
  orchestrator are frozen and script generation, validation and execution advance
  automatically without a later script-confirmation or continuation gate.
- Users retain explicit stop and configuration controls. Tool permissions and
  execution safety are delegated to the selected Agent backend.
- `/hire <name>` creates a durable Employee Bot through the controlled
  tool/model/profile flow. Arbitrary `/hire --prompt` input is rejected.
- The standalone Autonomous Manager command surface is retired. Employee and team
  actually connected to production.
- Deep and Spec remain protected execution strategies rather than being rewritten
  as a shared workflow.
- Workflow generates task-specific JavaScript and scales from one Agent to the
  `classify`, `fanout`, `verify`, `generate`, `tournament`, `loop`, `sequence` and
  `race` primitives. Its runtime owns deterministic control flow while Agents own
  semantic work.

This contract describes product boundaries. GhostAP does not add command-risk
filtering, ACP tool filtering, Workflow isolation, or employee process isolation;
identity access and file-backed durability remain control-plane concerns.
