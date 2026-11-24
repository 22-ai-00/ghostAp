# GhostAP Product Contract

GhostAP is a Feishu/Lark-native AI engineering gateway for a trusted engineering
environment.

- Main Bot: unified control plane for projects, programming sessions, Deep/Spec/
  Workflow engines, managed groups, access control, and outbound delivery.
- Provider/model/engine: replaceable execution capability, shown under Advanced.
- Host Shell: direct host execution without a GhostAP command policy or
  operating-system sandbox.
- Persistent storage: single-host and file-backed; it does not claim
  multi-replica linearizability or privileged-host rollback resistance.

## Public command truth

- Direct programming keeps the selected Agent's existing session, cancellation,
  retry and streaming-card semantics; the control plane does not add a planning
  or routing model call.
- Direct, Deep and Spec automatically advance each accepted task to a successful
  or explicit failed terminal state. Missing ordinary choices use the saved or
  recommended default; bounded recovery does not add confirmation or resume gates.
- `/wf <requirement>` opens an interactive Teamwork launch & config card
  prefilled with the chat/project's saved team or auto-bound recommended
  Agent Pool (1-8 `tool+model` Agents with `profile`/`effort`/`role`) for
  1-click launch or per-run adjustment; `/wf --auto <requirement>` and
  `/wf --preset <id> <requirement>` bypass the config card and launch
  immediately.
- Once launched (or via `--auto` / `--preset`), Workflow
  auto-binds an available recommended Agent Pool containing 1-8 `tool+model` Agents
  (or uses the saved/selected pool) and freezes the pool and orchestrator immediately;
  script generation, validation and execution advance automatically without any
  confirmation or continuation gate.
- Users may configure per-chat/project Teamwork presets (`collaborative`,
  `quick`, `review_team`, `deep_collab`, `auto`, `custom`) and per-member
  `tool + model + profile + effort + role` bindings via `/wf` (no args) or
  `/wf_config`, or launch/override per run via `/wf <requirement>`,
  `/wf --auto <requirement>`, or `/wf [--preset <id>] <requirement>`.
- Users retain explicit stop and configuration controls. Tool permissions and
  execution safety are delegated to the selected Agent backend.
- The standalone Autonomous Manager and Employee Department surfaces are retired.
- Deep and Spec remain protected execution strategies rather than being rewritten
  as a shared workflow.
- Workflow generates task-specific JavaScript and scales from one Agent to the
  `classify`, `fanout`, `verify`, `generate`, `tournament`, `loop`, `sequence` and
  `race` primitives. Its runtime owns deterministic control flow while Agents own
  semantic work.

This contract describes product boundaries. GhostAP does not add command-risk
filtering, ACP tool filtering, or Workflow isolation; identity access and
file-backed durability remain control-plane concerns.
