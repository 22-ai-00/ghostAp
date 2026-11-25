## Summary / 变更说明

<!-- Briefly explain what changed and why. Reference any related issue (e.g. Closes #123). -->

## Verification Checklist / 提交自检清单

- [ ] **Linux & macOS Parity**: No Linux-only assumptions in processes, file locks, paths/temp dirs, permissions, or tooling (`AGENTS.md`); added regression coverage for platform-sensitive behavior.
- [ ] **`src/card/` Unidirectional Import Boundary**: Preserved `handler -> session -> render / delivery` (`docs/adr-card-pipeline-boundary.md`); updated `ux/*.html` previews and inspected rendered screenshots if UI/cards changed.
- [ ] **Lock Ordering & Leaf Locks**: Respected `LockLevel` partial order (`docs/adr-lock-ordering.md`); every native `threading.Lock()` / `RLock()` carries `# leaf lock: never held while acquiring a LockLevel lock`.
- [ ] **Validation & Lint**: `APP_ID=ci_dummy APP_SECRET=ci_dummy uv run python -m src.main --validate` and `uv run ruff check src/ tests/ scripts/` pass cleanly.
- [ ] **Tests**: Ran targeted tests plus `uv run python -m pytest tests/ -q -m "not slow"` (and `uv run python scripts/test_inventory.py tests/` for test changes; see `docs/testing.md`).
- [ ] **Docs & `.Memory/`**: Updated `.Memory/` rolling index / daily log if the change is significant, and verified all local Markdown links resolve.

## Test Evidence / 验证输出

```text
# Paste exact validation/test command output here (redacting any secrets)
```
