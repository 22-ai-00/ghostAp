---
name: workflow-engine
description: >-
  GhostAP /wf Teamwork multi-agent workflow engine, interactive pool configuration card,
  JS runtime primitives, and report generation. Load when editing src/workflow_engine/ or /wf* handlers.
---

# Workflow Engine (`/wf`)

## Launch & Pool Freeze Contract

- `/wf <requirement>` opens the interactive Team Config & Launch Card prefilled with the chat+project's saved team or recommended Agent Pool (`1–8` `tool+model` agents with `A1`..`A8`, `profile`/`effort`/`role`; duplicate identical `tool/model/profile/effort` tuples are rejected).
- `/wf --auto <requirement>` and `/wf --preset <id> <requirement>` skip the config card, bind immediately, and start orchestration.
- `/wf` (no args) or `/wf_config` configures the chat+project default team; `/wf_status` inspects progress; `/stop_wf` cancels execution; `/wf_help` prints help.
- Once bound, pool and orchestrator freeze; script generation, validation, and execution run end-to-end without mid-run confirmation or continuation gates. Config card callbacks remain scoped to the same chat/project (legacy selection cards also remain initiator-scoped).

## Runtime, Primitives, & Reporting

- Scripts run in normal Node.js host context (>= `NODE_MIN_VERSION` in [constants.py](../../../src/workflow_engine/constants.py); all "Node too old" messages derive from it).
- Runtime primitives: `classify`, `fanout`, `verify`, `generate`, `tournament`, `loop`, `sequence`, `race` (`generate`/`loop` cap `50`, `MAX_TOTAL_AGENTS` cap `200`).
- Script, prompt, fallback, and `WorkflowRunSpec` may only use bound pool members via `agentId` (runtime scripts cannot override frozen tool/model bindings; dynamic nodes declare `runtime: true` + `candidateAgentIds`).
- Timeouts: 120s activity-idle timeout + shared 600s hard deadline across unique pool fallbacks. Only `COMPLETED`, `FAILED`, and `CANCELLED` are terminal states.
- Key SSOT modules: [commands.py](../../../src/workflow_engine/commands.py), [tool_registry.py](../../../src/workflow_engine/tool_registry.py), [script_gen.py](../../../src/workflow_engine/script_gen.py), [engine.py](../../../src/workflow_engine/engine.py), [reporting.py](../../../src/workflow_engine/reporting.py), [report_normalizer.py](../../../src/workflow_engine/report_normalizer.py), [report_markdown_html.py](../../../src/workflow_engine/report_markdown_html.py).
