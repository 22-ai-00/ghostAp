---
name: acp-backends
description: >-
  Multi-backend coding tool integration (11 tools across ACP stdio and CLI bridges),
  model/profile/effort cascades, native slash pass-through, and /usage & /settings cockpits.
---

# ACP & Coding Backends

## Unified Abstraction Across 11 Backends

- Backend SSOT lives in [src/mode/backends.py](../../../src/mode/backends.py) and [src/card/backend_meta.py](../../../src/card/backend_meta.py): 9 visible modes (`coco`, `claude`, `aiden`, `codex`, `gemini`, `traex`, `grok`, `dsh`, `agy`) plus 2 hidden CLI bridges (`claude_w`, `codex_w`).
- Treat all 11 tools uniformly behind GhostAP abstractions; only branch on backend identity when transport or upstream capabilities genuinely differ.
- `ACPSessionManager` owns session-key resolution and locking—never construct or parse session keys ad hoc in handlers.
- Reference guides: [acp_provider_guide.md](../../../docs/acp_provider_guide.md), [agy_acp_integration.md](../../../docs/agy_acp_integration.md), [dsh_acp_integration.md](../../../docs/dsh_acp_integration.md), [grok_acp_integration.md](../../../docs/grok_acp_integration.md), [programming-execution-window-rollover.md](../../../docs/programming-execution-window-rollover.md).

## Native `/` Slash Hygiene & `/usage` / `/settings` Cockpits

- Non-task native commands (see `ZERO_ARG_NATIVE_COMMANDS`, `STRICT_ARG_NATIVE_COMMANDS`, and `is_non_task_native_slash()` in [native_commands.py](../../../src/acp/native_commands.py)) **must** pass cleanly without appending `AUTONOMOUS_EXECUTION_PROMPT` or bridge context across ACP and CLI sessions (`agy_cli.py`, `codex_cli.py`, `claude_cli.py`, `base.py`).
- Interactive terminal-only commands (`/vim`, `/theme`, `/login`, AGY `/context`, etc.) are intercepted via `get_tui_only_command_hint()` with actionable Feishu hints; `BLOCKED_NATIVE_COMMANDS` blocks `/logout`.
- `/usage` and `/settings` are GhostAP control commands ([tool_telemetry.py](../../../src/acp/tool_telemetry.py), [tool_cockpit.py](../../../src/feishu/handlers/tool_cockpit.py)) off the 23-item slash panel (`HIDDEN_SLASH_COMMANDS`), supporting per-chat/project `tool_runtime_settings` persisted on `ProjectContext`.
