# Contributing to GhostAP

Thank you for contributing to GhostAP! Before sending a pull request, please review the repository contracts in [AGENTS.md](AGENTS.md) and [docs/product-contract.md](docs/product-contract.md).

## Development Setup (`uv` Only)

GhostAP exclusively uses **`uv`** for Python dependency and environment management (`pip` and `conda` are not supported).

- **Requirements**: Python `>=3.11` (CI tests on Python `3.12`), [`uv`](https://docs.astral.sh/uv/), and **Node.js 20+** (required for the `/wf` Workflow JS runtime).
- **Clone and sync**:

```bash
git clone https://github.com/22-ai-00/ghostAp.git
cd ghostAp
uv sync --group dev
```

- **Optional Git hooks**: Enable repository commit reminders via [.githooks/README.md](.githooks/README.md):

```bash
git config core.hooksPath .githooks
```

- **Validate configuration offline** (no real Feishu credentials required):

```bash
APP_ID=ci_dummy APP_SECRET=ci_dummy uv run python -m src.main --validate
```

## Running Tests, Linting & Builds

See [docs/testing.md](docs/testing.md) for the full tiered testing strategy. Common local gates (matching CI):

```bash
# 1. Lint
uv run ruff check src/ tests/ scripts/

# 2. Targeted tests for the area you modified
uv run python -m pytest tests/test_target.py -q

# 3. Fast suite (CI default; excludes real-sleep/fault-injection slow tests)
uv run python -m pytest tests/ -q -m "not slow"

# 4. Full suite (required before release / shared routing, card, lock, or config changes)
uv run python -m pytest tests/ -q

# 5. Test inventory & duplicate audit
uv run python scripts/test_inventory.py tests/

# 6. Wheel & sdist packaging smoke check
uv build
```

## Core Engineering Contracts

1. **Cross-Platform Parity (Linux & macOS)**
   GhostAP supports both Linux and macOS (both run in CI on every push and PR). Any code touching processes, file locks, paths/temp directories, file permissions, environment variables, or CLI binaries must use platform branches or capability probes and include regression tests—never introduce Linux-only assumptions.

2. **Orthogonal Strategies & 11-Backend Uniformity**
   Execution strategies (Direct programming, Deep, Spec, Teamwork `/wf`) are orthogonal to transport (ACP stdio / CLI bridges). New backend capabilities should support all 11 backends registered in [src/mode/backends.py](src/mode/backends.py) (`9` visible tools + `claude_w`, `codex_w`) uniformly rather than special-casing one tool.

3. **`src/card/` Unidirectional Import Boundary**
   GhostAP enforces a strict one-way dependency pipeline across `src/card/` (see [docs/adr-card-pipeline-boundary.md](docs/adr-card-pipeline-boundary.md)):

   ```text
   handler -> session -> render / delivery
   ```

   - `src/card/render/` and `src/card/builders/` must **never** import `src/card/delivery/`, `src/card/session/`, or `src/feishu/`.
   - `src/card/delivery/` must **never** import `src/card/session/`.
   - Shared cross-layer contracts belong in [src/card/protocols.py](src/card/protocols.py) or `src/card/events/`.
   - When modifying Feishu Card JSON 2.0 layouts or HTML reports, update the corresponding `ux/*.html` preview first and visually inspect rendered screenshots before finalizing.

4. **Lock Ordering & Leaf Lock Annotations**
   All multi-chat and repository locks must follow the strict `LockLevel` partial order documented in [docs/adr-lock-ordering.md](docs/adr-lock-ordering.md) and implemented in [src/utils/lock_order.py](src/utils/lock_order.py):
   - Never acquire an outer lock while holding an inner lock (run with `GHOSTAP_LOCK_ORDER_CHECK=strict` in development to catch violations).
   - Every native `threading.Lock()` or `threading.RLock()` must carry the inline comment `# leaf lock: never held while acquiring a LockLevel lock`.

5. **Documentation & Existing Comments**
   Preserve unrelated comments and docstrings, verify that all local Markdown links resolve, and keep public documentation aligned with [docs/product-contract.md](docs/product-contract.md) (no retired terms or unverified claims).

## Commit & Pull Request Guidelines

- Follow [docs/commit-message-guidelines.md](docs/commit-message-guidelines.md): `<type>(<scope>): <subject>` with accurate file/module scope in the commit body.
- Complete the checklist in [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) when opening a PR.
- For significant architectural or behavioral changes, update `.Memory/` alongside your code.
- For security issues, follow [SECURITY.md](SECURITY.md) instead of opening a public issue or PR.

## License

By contributing to GhostAP, you agree that your contributions will be licensed under the [MIT License](LICENSE).
