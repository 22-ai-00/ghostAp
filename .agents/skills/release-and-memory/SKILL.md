---
name: release-and-memory
description: >-
  Git dev-to-main landing, verification gates, ./restart.sh diagnostics, .Memory/ rolling
  index maintenance, and multi-phase session handoffs. Load before commits, releases, or memory updates.
---

# Release, Verification & Memory Hygiene

## Git & Verification Flow

1. Always verify and commit focused changes on `dev` first, following [commit-message-guidelines.md](../../../docs/commit-message-guidelines.md).
2. Run validation gates matching change scope (see [testing.md](../../../docs/testing.md)):
   - Targeted pytest first; expand to full `uv run python -m pytest tests/ -q -m "not slow"` when touching shared routing, card rendering, locks, config, or session code.
   - Static & hygiene gates: `uv run python -m src.main --validate`, `uv run ruff check src/ tests/`, `uv run python scripts/test_inventory.py tests/`, `git diff --check`.
3. Push `origin/dev`, then fast-forward `main` only after `dev` is verified and pushed:
   `git push origin dev && git checkout main && git merge --ff-only dev && git push origin main` (verify all push targets succeed).
4. Live restart: run `GHOSTAP_LOG_MODE=append ./restart.sh` and confirm `[RESTART] start ready` in `logs.log` (check `[RESTART]` markers to separate script delay from Python cold start before changing app code).

## Rolling `.Memory/` & Phase Handoffs

- After meaningful tasks, add a detailed entry to `.Memory/{YYYY-MM-DD}.md` (what changed, why, verification evidence, follow-up risks) and a concise ~20-char bullet in [Abstract.md](../../../.Memory/Abstract.md).
- Keep [Abstract.md](../../../.Memory/Abstract.md) concise (`<= 60` lines); roll older months into `.Memory/Archive-*.md` instead of growing an unbounded index. Record Medium/Low findings in [Backlog.md](../../../.Memory/Backlog.md) and fix High findings immediately.
- For large multi-phase tasks, write a brief handoff note (goal, verified decisions, exact file paths, test status) between phases or branch independent work into parallel subagents/worktrees instead of relying on mid-task compaction.
- Codex `spawn_agent` rule: passing explicit `agent_type` requires `fork_turns="none"` or a positive integer; omit `agent_type` for full-history forks.
