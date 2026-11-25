# Security Policy

## Supported Versions

Security updates are applied to the latest release on the `main` branch (`0.2.x`).

| Version | Supported |
| --- | --- |
| `0.2.x` (`main`) | Yes |
| `< 0.2.0` | No |

## Security & Trust Model

GhostAP is an **outbound-WebSocket Feishu/Lark engineering control plane designed for a trusted engineering host**. Please review [docs/product-contract.md](docs/product-contract.md) before deploying:

- **Direct host execution (no GhostAP OS sandbox)**:
  Host Shell (`/sh` or Smart-mode fallback when `DEFAULT_ACP_TOOL` is empty), ACP/CLI coding backends, and `/wf` Node.js Workflows execute **directly under the GhostAP host OS user** with that user's filesystem, process, and network permissions.
- **No secondary command or tool filtering**:
  GhostAP intentionally does **not** add command blacklists/whitelists, ACP tool filtering, Bubblewrap/Seatbelt wrappers, or a Node VM sandbox. Tool permissions and approval policies are delegated to the host OS account boundary and the selected AI agent backend.
- **Single-host file-backed durability**:
  Project state, lock state, and `.env` access configuration are local and file-backed. Do not expose a single GhostAP host instance to mutually untrusted tenants.

## Hardening Controls & Deployment Baseline

1. **Enforce Dual Allowlist Mode (`INGRESS_ACCESS_MODE=enforced`)**
   By default (`.env.example` / [src/config/settings.py](src/config/settings.py)), `INGRESS_ACCESS_MODE=legacy_allow_all` allows any private (P2P) or group chat reached by the Feishu app to invoke ordinary commands. For shared or restricted environments, explicitly enable fail-closed dual allowlisting in `.env` (see [src/access_control.py](src/access_control.py)):

   ```env
   INGRESS_ACCESS_MODE=enforced
   ADMIN_BOOTSTRAP_SCOPE=p2p_only
   ALLOWED_USER_IDS=ou_xxxxxxxx
   ALLOWED_CHAT_IDS=oc_xxxxxxxx
   ```

   With `INGRESS_ACCESS_MODE=enforced`, inbound requests require **both** an authorized user (`ADMIN_USER_IDS` or `ALLOWED_USER_IDS`) and an authorized chat (`ALLOWED_CHAT_IDS`). Admins can enroll new groups in-chat with `/access allow-chat`.

2. **Bootstrap Admin Immediately (`/setadmin`)**
   When `ADMIN_USER_IDS` is empty, GhostAP permits a **one-time** `/setadmin` bootstrap (restricted by default via `ADMIN_BOOTSTRAP_SCOPE=p2p_only` to private 1-on-1 chat with the bot; see [src/admin_bootstrap.py](src/admin_bootstrap.py)).
   - Run `/setadmin` in private chat immediately after first startup, or pre-populate `ADMIN_USER_IDS` in `.env` before launch.
   - Once `ADMIN_USER_IDS` is non-empty, anonymous bootstrap closes permanently and only existing admins can update `ADMIN_USER_IDS`, manage `/lock` / `/unlock`, or authorize chats via `/access`.

3. **Protect `.env` Credentials (`0600`) & Gate Directories**
   - Store `APP_ID`, `APP_SECRET`, and optional runtime settings only in `.env` (gitignored; template in [.env.example](.env.example)), never in code or public issues.
   - GhostAP's [AtomicEnvFileStore](src/config/env_file_store.py) creates and updates `.env` and its lock file atomically with mode `0600`. When creating `.env` manually, run `chmod 600 .env`.
   - Keep `GHOSTAP_RESTART_GATE_DIR` unset (uses a private sibling directory outside the checkout) or point it to a dedicated directory owned by the service user with no group/other permissions ([src/utils/restart_gate.py](src/utils/restart_gate.py)).

4. **Card Callback Signing & Log Redaction**
   - Feishu interactive card payloads carry HMAC-SHA256 signatures derived from `APP_SECRET` with nonce replay and chat-binding checks ([src/utils/signing.py](src/utils/signing.py)).
   - Output and diagnostics sanitizers ([src/utils/redact.py](src/utils/redact.py), [src/acp/diagnostics.py](src/acp/diagnostics.py)) redact common secret assignments, `Authorization` / `Bearer` headers, PEM private keys, and provider API key/token patterns before logging or rendering to Feishu cards (`acp_diagnostics_redact_enabled=True` by default). Redaction is defense-in-depth—still treat host logs (`logs.log`) as sensitive operational data.

## Reporting a Vulnerability

**Please do not report security vulnerabilities via public GitHub issues, pull requests, or discussions.**

If you discover a security bug (such as an access-control bypass in `enforced` mode, `/setadmin` bootstrap bypass, card signature bypass, XSS/HTML injection in Workflow reports, or credential leak):

1. Report it privately using GitHub's **Private Vulnerability Reporting** form under the repository's **Security** tab:
   `https://github.com/22-ai-00/ghostAp/security/advisories/new`
2. Include:
   - Affected GhostAP version or commit hash
   - OS (`Linux` / `macOS`) and relevant configuration (`INGRESS_ACCESS_MODE`, `ADMIN_BOOTSTRAP_SCOPE`, without secrets)
   - Clear reproduction steps or proof of concept
   - Impact assessment and any redacted log snippets

We aim to acknowledge private reports promptly, evaluate fixes against [docs/product-contract.md](docs/product-contract.md), and coordinate disclosure once a patched release is ready.
