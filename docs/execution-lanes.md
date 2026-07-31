# Execution lanes

GhostAP exposes every currently implemented execution lane to the single Owner.
The label beside a lane is a support expectation, never an access gate:
`mature`, `developing`, and `not_implemented` are the only completion labels.
No completion label enables a tenant rollout, allowlist, opt-in, approval, or
release-state check.

Runtime health is a separate observation: `available`, `degraded`, or
`unavailable`.  Health can explain why a current invocation cannot proceed; it
does not hide the Owner's entry from the product surface.

| Lane | Entry | Completion | Current contract |
| --- | --- | --- | --- |
| Direct | `/acp`, explicit Agent commands | mature | Keeps the selected Agent session and adds no planner/router LLM call or remote discovery hop. |
| Deep | `/deep` | mature | Existing protected Deep execution strategy. |
| Spec | `/spec` | mature | Existing protected Spec execution strategy. |
| Worktree | `/worktree`, `/wt` | developing | Existing execution stays visible; the unified adapter contract is unfinished. |
| Workflow | `/wf`, `/workflow` | developing | Existing RunSpec/reviewer execution stays visible; IR v2 and durable ports are unfinished. |
| Team | `/team`, `/new-team` | developing | Existing employee collaboration stays visible; its unified durable task graph is unfinished. |
| Slock | `/slock` | developing | Existing group collaboration stays visible; the unique execution fact source is unfinished. |

When an action has an unmet execution prerequisite, its product entry must state
that concrete blocker.  GhostAP must not substitute hidden entries for runtime
validation.

Explicit Direct, Deep, and Spec commands always take priority over SMART or
Slock automatic activation.  This is a routing-safety rule, not a maturity or
release policy.  Existing aliases and card actions remain compatible.

`/goal` is a retired standalone-Manager command.  It remains a deterministic,
non-executing migration response for compatibility, but it is not advertised as
an execution entry or a proactive-task promise.
