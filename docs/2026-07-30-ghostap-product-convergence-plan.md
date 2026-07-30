# GhostAP Product Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Converge GhostAP from three overlapping product generations into a trustworthy Feishu-native Agent Department control plane while preserving existing provider, engine, main-Bot WebSocket, Slock ACP, and file-storage contracts.

**Architecture:** The main Bot becomes the department control plane, employee Bots remain the execution plane, and providers/engines move behind typed catalogs and adapters. The migration is gated: first make security and user-visible contracts truthful, then introduce one effective interaction context and one task read/control model, and only after an explicit scope decision harden engine-internal lifecycle behavior.

**Tech Stack:** Python 3.11+, Pydantic Settings, `lark-oapi==1.7.1`, `lark-channel-sdk==1.1.0`, ACP 0.11+, pytest, Ruff, uv, file-backed Journal/Blob storage, Node.js 20+ for Workflow.

## 中文执行摘要

本计划不把 GhostAP 继续做成“命令更多、模式更多的研发机器人”，而是
收敛成一个明确产品：**部署在可信工程环境中的飞书原生 Agent
Department 网关**。用户首先看到项目、员工、团队、任务、审批与审计；
Coco、Codex、Deep、Workflow 等只在高级能力层出现。

实施分四阶段、26 个任务：

1. **合同与信任（任务 1–8）**：先关闭入站鉴权、宿主 Shell、员工部门
   默认启用、群消息保留和话题状态误路由等 P0 风险，同时修正文档与真实
   行为的矛盾。
2. **产品控制面（任务 9–13）**：把飞书面板从 79 个技术命令收敛到 11
   个产品入口，建立角色化菜单、唯一有效上下文、唯一 RouteDecision 和
   跨引擎任务看板/控制接口。
3. **运行时收敛（任务 14–20）**：统一 Backend 能力目录、Provider
   选择、运行终态和检查点；涉及 Spec、Worktree、Deep、Workflow 行为
   修改的部分必须先通过 DG-2。
4. **运维与切流（任务 21–26）**：隔离退休的 Autonomous Manager，
   补齐 doctor、加密离线备份、可选 Shell OS 隔离、真实租户 Beta、
   灰度与回滚证据，再切换默认产品面。

前三个优先交付窗口是：第 1 周完成安全默认值与准入合同；第 2 周关闭
员工数据和话题误路由风险；第 3–5 周完成 11 命令产品面、路由 SSOT 与
统一任务控制。总投入约 76–123 人日；3 名工程师并行并计入灰度观察和
真实租户验收后，预计 9–14 个日历周。发布日期由门禁证据决定，不由排期
倒逼。

## Global Constraints

- Use only `uv`; never use pip or conda.
- Preserve user changes and the `dev` branch workflow; never reset, force-push, or delete unrelated files.
- Do not introduce a database. Journal, encrypted Blob, snapshots, and existing project-local files remain the persistence mechanisms.
- Do not change the externally observable main-Bot WebSocket ingress contract or employee Channel SDK transport.
- Do not bypass `ACPSessionManager` session-key ownership or Slock `_run_acp_session`.
- Preserve Journal SSOT, frozen Autonomous domain objects, PREPARED/EXECUTING anchoring, and fail-closed policy semantics.
- Preserve current Deep/Spec/Worktree/Workflow command spellings and routing as compatibility inputs. Hiding a command from the Slash panel does not remove its parser or handler.
- Behavior-changing engine work in Task 16 (Spec model selection) and Tasks
  18–20 (Worktree/Deep/Workflow lifecycle) requires Decision Gate DG-2 because
  the original product requirement says those engines' logic and routing must
  not be changed. Behavior-preserving catalog adapters and read-only lifecycle
  projections do not cross this gate.
- UI changes require an HTML preview under `ux/` before production card code changes.
- Every behavior change starts with a failing regression and ends with the most relevant focused suite. Shared route/card/config/session changes then expand to the non-slow and full suites.
- No skipped, failed, timed-out, or summary-less test run counts as passing.
- Each task is one reviewable commit using `docs/commit-message-guidelines.md`.

---

## Product Contract

GhostAP's public position after this plan is:

> A Feishu/Lark-native Agent Department gateway deployed in a team's trusted
> engineering environment. The main Bot controls projects, employees, teams,
> tasks, approvals, and audit; employee Bots execute work. Coco, Claude, Aiden,
> Codex, Gemini, Traex, TTADK, TUI2ACP, Deep, Spec, Worktree, and Workflow are
> replaceable execution capabilities, not the first-level product taxonomy.

The primary product objects are:

1. `Project`
2. `Employee`
3. `Team`
4. `Task/Run`
5. `Approval/Audit`

The primary interaction surfaces are:

- Main Bot DM: administration, security posture, hire/fire, projects, roster, audit.
- Project/team group: natural-language task submission, task board, review, stop, retry.
- Employee Bot: `/task`, `/status`, `/history`, `/memory`, `/stop`.
- Advanced drawer: provider/model selection, named engines, raw host Shell, diagnostics.

## Scope Boundaries and Decision Gates

### DG-0: Baseline

Before Task 1, record:

```bash
git status --short --branch
git rev-parse HEAD
uv run python -m src.main --validate
uv run python -m pytest tests/test_docs_references.py tests/test_auth_whitelist.py tests/test_route_decision.py -q
```

Expected: clean or explicitly understood worktree, valid configuration, and no unexplained test failure.

### DG-1: Product surface cutover

Tasks 7–12 may change what is shown in Slash/help/menu, but must retain every existing command parser and handler. Cut over the reduced surface only after a shadow reconciliation proves that every hidden command remains directly callable.

### DG-1A: First-admin bootstrap scope

The repository's durable `AGENTS.md` rule currently permits any sender's
`/setadmin` while `ADMIN_USER_IDS` is empty; it does not restrict the first call
to P2P. The recommended security contract is P2P-only, but Task 3 may adopt it
only after the owner explicitly approves this rule change and the same commit
updates `AGENTS.md`.

Without approval, preserve the any-chat first-admin behavior, emit
`admin_bootstrap_any_chat` as a release-blocking warning, and do not claim the
bootstrap race is closed. Never change the rule silently inside routing code.

### DG-2: Engine-internal lifecycle work

Task 16 changes Spec model-selection behavior, and Tasks 18–20 change
Worktree/Deep/Workflow lifecycle behavior. They therefore pause until the owner
explicitly approves relaxing the original “do not modify
Deep/Spec/Worktree/Workflow logic” constraint. Without approval:

- complete Tasks 1–15, 17, and 21–26;
- expose honest limitations through the unified task model;
- do not claim provider-local Spec retries, cross-restart recovery, or hard
  wall-clock cancellation for engines that do not provide them.

### DG-3: Production claim

Do not call Employee Department production-ready until the real-tenant matrix passes on desktop and mobile and includes restart/reconnect plus 1/10/50 employee evidence. Local tests may qualify the built-in profile as release-candidate, not as externally proven production.

## Program Sequence

| Phase | Tasks | Exit condition | Estimate |
| --- | --- | --- | --- |
| 0. Contract and trust | 1–8 | Security/config/onboarding contracts are truthful; P0 misrouting is closed | 13–20 engineer-days |
| 1. Product control plane | 9–13 | Reduced role-aware surface, effective context, production RouteDecision, unified task view | 18–28 engineer-days |
| 2. Runtime convergence | 14–20 | Backend/default parity and lifecycle projection complete; DG-2 work is complete or explicitly excluded by Track A | 27–45 engineer-days |
| 3. Operability and cutover | 21–26 | Legacy surface quarantined, doctor/backup/tenant evidence complete, selected release track enabled | 18–30 engineer-days |

With three engineers working on independent tasks, expect roughly 9–14 calendar weeks. The gates, not the dates, determine release readiness.

## Target File Map

- `src/config/security_posture.py`: pure configuration-to-posture evaluation.
- `src/sandbox/access_policy.py`: host-Shell authorization; no subprocess execution.
- `src/feishu/product_catalog.py`: public actions, compatibility aliases, roles, scopes.
- `src/feishu/effective_context.py`: one frozen effective request context.
- `src/feishu/route_executor.py`: side effects for immutable `RouteDecision` values.
- `src/tasking/control_plane.py`: cross-engine task read/control protocols and registry.
- `src/tasking/adapters/`: adapters around existing Scheduler/Deep/Spec/WT/WF/Slock/Employee APIs.
- `src/agent_session/backend_catalog.py`: provider capabilities and supported strategies.
- `src/agent_session/request.py`: typed session request replacing scattered factory parameters.
- `src/tasking/run_contract.py`: shared lifecycle, terminal reason, checkpoint, and control ports.
- `src/autonomous/context/group_retention.py`: membership-scoped retention and payload tombstones.
- `src/autonomous/legacy/`: quarantined, production-unreachable standalone Manager code.
- `ux/agent-department-control-plane.html`: reviewed role-aware menu/help/status preview.

## Audit-to-Plan Traceability

| Finding | Severity | Current evidence | Owning task / regression | Gate |
| --- | --- | --- | --- | --- |
| P0-01 Empty ingress allowlists admit ordinary traffic | P0 Security | `tests/test_auth_whitelist.py`, `src/feishu/ws_client.py` | Tasks 2–3; empty-list denial regression | DG-1A only for P2P bootstrap |
| P0-02 “Sandbox” is host Shell without an OS isolation guarantee | P0 Security | `src/sandbox/executor.py`, `.env.example` | Tasks 2, 4, 24; policy/no-fallback suites | none |
| P0-03 Employee OFF can still compose durable runtime or record text | P0 Privacy | `src/autonomous/provisioning/composition.py`, `src/feishu/ws_client.py` | Task 5; disabled-runtime regressions | none |
| P0-04 Group context lacks bounded crash-safe payload retention | P0 Privacy | `src/autonomous/context/group_ledger.py` | Task 6; tombstone/restart chaos tests | none |
| P0-05 Engine topic can be intercepted into programming mode | P0 Correctness | `src/feishu/ws_client.py`, `src/feishu/route_decision.py` | Task 7; engine/provider transition matrix | none |
| P1-01 Product entry is a 79-command technical catalog | P1 Product | `src/feishu/main_slash_commands.py` | Tasks 9–10; exact 79→11 reconciliation | DG-1 |
| P1-02 Production routing has duplicate context and imperative ladders | P1 Architecture | `src/feishu/dispatcher.py`, `src/feishu/route_decision.py` | Tasks 11–12; resolve-once/shadow parity | DG-1 |
| P1-03 `/goal` welcome contradicts retired Manager routing | P1 Product | `src/feishu/handlers/system.py`, Slock welcome | Task 8; retired migration response | none |
| P1-04 Docs advertise rejected `/hire --prompt` | P1 Product/Safety | README/help vs hire parser | Tasks 1 and 8; docs contract | none |
| P1-05 `/status` only projects Deep/Spec | P1 Product | diagnostics handler/helper | Task 13; cross-runtime `TaskSnapshot` | none |
| P1-06 Provider/default behavior drifts; Spec can cross providers | P1 Correctness | Intent/project/Spec selection code | Tasks 14–16; backend/provider matrix | Task 16 requires DG-2 |
| P1-07 Worktree timeout can wait indefinitely or report false success | P1 Correctness | Worktree dispatcher/manager | Task 18; bounded cancel and terminal tests | DG-2 |
| P1-08 Deep loses executable progress across restart | P1 Reliability | `src/deep_engine/engine.py` | Task 19; restart checkpoint tests | DG-2 |
| P1-09 Workflow Journal/snapshot claims exceed real run-local state | P1 Reliability | Workflow journal/state manager | Task 20; run recovery/current-activity tests | DG-2 |
| P1-10 Retired Manager overlaps the Employee Department concept | P1 Architecture | Autonomous exports/retired commands | Task 21; import reachability boundary | none |
| P1-11 Readiness, backup and real-tenant evidence are fragmented | P1 Operations | acceptance manifests/scripts | Tasks 22–25; doctor/restore/signed beta | DG-3 |

---

## Phase 0 — Contract and Trust

### Task 1: Freeze the public product contract

**Files:**
- Create: `docs/product-contract.md`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `docs/goals.md`
- Modify: `tests/test_docs_references.py`

**Interfaces:**
- Consumes: current production-reachable commands and Employee Department composition.
- Produces: one authoritative public positioning document and a documentation regression gate.

- [ ] **Step 1: Write the failing documentation contract**

```python
def test_public_positioning_matches_agent_department_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "Agent Department" in readme
    assert "主 Bot 是控制面" in readme
    assert "provider/transport" in readme
    assert "Shell 沙箱服务" not in pyproject


def test_public_docs_do_not_promise_rejected_hire_prompt() -> None:
    public = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "README.md", ROOT / "docs" / "product-contract.md")
    )
    assert "/hire 小明 --tool codex --model o3-pro --prompt" not in public
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run python -m pytest tests/test_docs_references.py -q
```

Expected: FAIL because positioning still describes a Shell sandbox and advertises `--prompt`.

- [ ] **Step 3: Write the exact product contract**

`docs/product-contract.md` must include:

```markdown
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
```

Update the README opening, capability table, Autonomous section, and command examples to match it. Set the package description to:

```toml
description = "飞书原生 Agent Department 控制面与研发执行网关"
```

- [ ] **Step 4: Verify the documentation contract**

Run:

```bash
uv run python -m pytest tests/test_docs_references.py -q
git diff --check
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md pyproject.toml docs/product-contract.md docs/goals.md tests/test_docs_references.py
git commit -m "docs(product): define Agent Department contract"
```

### Task 2: Add a typed security-posture evaluator

**Files:**
- Create: `src/config/security_posture.py`
- Create: `tests/test_security_posture.py`
- Modify: `src/config/settings.py`
- Modify: `src/config/singleton.py`
- Modify: `src/main.py`
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Produces: `IngressAccessMode`, `ShellAccessMode`, `SecuritySeverity`,
  `SecurityFinding`,
  `SecurityPosture`, and
  `evaluate_security_posture(settings, isolation_ready=False)`.
- Consumed by: Task 3 ingress enforcement, Task 4 Shell enforcement, Task 5 Employee runtime gate, `/status`, and `--validate`.

- [ ] **Step 1: Write the failing posture truth table**

```python
@pytest.mark.parametrize(
    ("mode", "admins", "users", "chats", "ack", "isolation_ready", "valid"),
    [
        ("disabled", "", "", "", False, False, True),
        ("admin_dm", "", "", "", False, False, False),
        ("admin_dm", "ou_admin", "", "", False, False, True),
        ("allowlisted", "", "", "", False, False, False),
        ("allowlisted", "", "ou_user", "", False, False, False),
        ("allowlisted", "", "", "oc_chat", False, False, False),
        ("allowlisted", "", "ou_user", "oc_chat", False, False, True),
        ("isolated", "", "", "", False, False, False),
        ("isolated", "", "", "", False, True, True),
        ("trusted_local", "", "", "", False, False, False),
        ("trusted_local", "", "", "", True, False, True),
    ],
)
def test_shell_access_posture_is_fail_closed(
    mode: str,
    admins: str,
    users: str,
    chats: str,
    ack: bool,
    isolation_ready: bool,
    valid: bool,
) -> None:
    settings = Settings(
        shell_access_mode=mode,
        shell_trusted_local_ack=ack,
        admin_user_ids=admins,
        allowed_user_ids=users,
        allowed_chat_ids=chats,
    )
    assert evaluate_security_posture(
        settings,
        isolation_ready=isolation_ready,
    ).is_valid is valid
```

Also assert that `shadow` and `legacy_allow_all` emit
`ingress_shadow_not_enforcing` and `ingress_legacy_allow_all`; they may support
migration or a time-bounded rollback, but must never be the default or be
visually indistinguishable from an enforced posture.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest tests/test_security_posture.py -q
```

Expected: FAIL because the types and fields do not exist.

- [ ] **Step 3: Implement the pure evaluator**

```python
class IngressAccessMode(str, Enum):
    ENFORCED = "enforced"
    SHADOW = "shadow"
    LEGACY_ALLOW_ALL = "legacy_allow_all"


class ShellAccessMode(str, Enum):
    DISABLED = "disabled"
    ADMIN_DM = "admin_dm"
    ALLOWLISTED = "allowlisted"
    ISOLATED = "isolated"
    TRUSTED_LOCAL = "trusted_local"


class SecuritySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class SecurityFinding:
    code: str
    severity: SecuritySeverity
    message: str


@dataclass(frozen=True)
class SecurityPosture:
    ingress_mode: IngressAccessMode
    shell_mode: ShellAccessMode
    employee_department_enabled: bool
    records_group_content: bool
    findings: tuple[SecurityFinding, ...]

    @property
    def is_valid(self) -> bool:
        return all(
            item.severity is not SecuritySeverity.BLOCKING
            for item in self.findings
        )


def evaluate_security_posture(
    settings: Settings,
    *,
    isolation_ready: bool = False,
) -> SecurityPosture:
    ingress_mode = IngressAccessMode(settings.ingress_access_mode)
    mode = ShellAccessMode(settings.shell_access_mode)
    findings: list[SecurityFinding] = []
    if ingress_mode is IngressAccessMode.LEGACY_ALLOW_ALL:
        findings.append(SecurityFinding(
            "ingress_legacy_allow_all",
            SecuritySeverity.WARNING,
            "legacy_allow_all is a time-bounded break-glass mode",
        ))
    if ingress_mode is IngressAccessMode.SHADOW:
        findings.append(SecurityFinding(
            "ingress_shadow_not_enforcing",
            SecuritySeverity.WARNING,
            "shadow records prospective denials but still allows traffic",
        ))
    if settings.admin_bootstrap_scope == "any_chat":
        findings.append(SecurityFinding(
            "admin_bootstrap_any_chat",
            SecuritySeverity.WARNING,
            "first-admin bootstrap remains available from any chat",
        ))
    if mode is ShellAccessMode.ADMIN_DM and not settings.admin_user_ids:
        findings.append(SecurityFinding(
            "shell_admin_missing",
            SecuritySeverity.BLOCKING,
            "admin_dm requires at least one configured administrator",
        ))
    if (
        mode is ShellAccessMode.ALLOWLISTED
        and (
            not settings.allowed_user_ids
            or not settings.allowed_chat_ids
        )
    ):
        findings.append(SecurityFinding(
            "shell_allowlist_missing",
            SecuritySeverity.BLOCKING,
            "allowlisted requires both user and chat allowlists",
        ))
    if (
        mode is ShellAccessMode.TRUSTED_LOCAL
        and not settings.shell_trusted_local_ack
    ):
        findings.append(SecurityFinding(
            "shell_trusted_local_unacknowledged",
            SecuritySeverity.BLOCKING,
            "trusted_local requires explicit risk acknowledgement",
        ))
    if mode is ShellAccessMode.ISOLATED and not isolation_ready:
        findings.append(SecurityFinding(
            "shell_isolation_unavailable",
            SecuritySeverity.BLOCKING,
            "isolated requires a successfully probed isolation backend",
        ))
    return SecurityPosture(
        ingress_mode=ingress_mode,
        shell_mode=mode,
        employee_department_enabled=settings.employee_department_enabled,
        records_group_content=settings.employee_department_enabled,
        findings=tuple(findings),
    )
```

The implementation must be pure and must emit stable codes:

- `shell_admin_missing`
- `shell_allowlist_missing`
- `shell_isolation_unavailable`
- `shell_trusted_local_unacknowledged`
- `ingress_legacy_allow_all`
- `ingress_shadow_not_enforcing`
- `admin_bootstrap_any_chat` until DG-1A is approved
- `employee_group_retention_missing`

Add these settings with secure defaults:

```python
ingress_access_mode: Literal[
    "enforced", "shadow", "legacy_allow_all"
] = "enforced"
admin_bootstrap_scope: Literal["any_chat", "p2p_only"] = "any_chat"
shell_access_mode: Literal[
    "disabled", "admin_dm", "allowlisted", "isolated", "trusted_local"
] = "disabled"
shell_trusted_local_ack: bool = False
employee_department_enabled: bool = False
employee_group_context_retention_days: int = Field(default=30, ge=1, le=3650)
```

`--validate` must print `[安全姿态]`, list stable codes, and exit non-zero for blocking findings.

- [ ] **Step 4: Run focused and configuration tests**

Run:

```bash
uv run python -m pytest tests/test_security_posture.py tests/test_config_validation.py -q
uv run python -m src.main --validate
```

Expected: tests PASS; validation passes with the checked-in `.env.example`.

- [ ] **Step 5: Commit**

```bash
git add src/config/security_posture.py src/config/settings.py src/config/singleton.py src/main.py tests/test_security_posture.py .env.example README.md
git commit -m "feat(config): add explicit security posture"
```

### Task 3: Make inbound access deny-by-default with safe onboarding

**Files:**
- Create: `src/access_control.py`
- Create: `src/config/env_file_store.py`
- Create: `tests/test_access_control.py`
- Modify: `src/admin_bootstrap.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/feishu/handlers/system.py`
- Modify: `tests/test_auth_whitelist.py`
- Modify: `tests/test_admin_bootstrap.py`
- Modify conditionally after DG-1A approval: `AGENTS.md`

**Interfaces:**
- Consumes: `IngressAccessMode` from Task 2.
- Produces: `IngressAccessRequest`, `AccessDecision`, `IngressAccessPolicy`,
  `IngressAccessPolicyProvider`, `AtomicEnvFileStore`.
- Consumed before: GroupLedger recording, image download, Shell policy, and all business routing.

- [ ] **Step 1: Write the failing onboarding and authorization tests**

```python
def test_empty_allowlists_reject_normal_messages(access_policy) -> None:
    result = access_policy.decide(
        IngressAccessRequest(
            sender_id="ou_unknown",
            chat_id="oc_unknown",
            chat_type="group",
            command_match=None,
        )
    )
    assert result.allowed is False
    assert result.reason_code == "access_not_enrolled"


def test_first_setadmin_is_p2p_only_after_dg_1a_approval(access_policy) -> None:
    allowed = access_policy.decide(
        IngressAccessRequest(
            sender_id="ou_first",
            chat_id="oc_dm",
            chat_type="p2p",
            command_match=SlashCommandParser.parse("/SeTaDmIn ou_ignored"),
        )
    )
    denied = access_policy.decide(
        IngressAccessRequest(
            sender_id="ou_first",
            chat_id="oc_group",
            chat_type="group",
            command_match=SlashCommandParser.parse("  /setadmin\tou_ignored  "),
        )
    )
    assert allowed.allowed is True
    assert denied.allowed is False
```

Add tests proving a denied message is not written to GroupLedger and an admin can
enrol the current unauthorized group with `/access allow-chat`. If DG-1A is not
approved, replace the P2P-only test with an explicit preservation test for
any-chat bootstrap plus the blocking readiness finding. Add a same-process
regression: after a successful `/setadmin` or `/access allow-chat`, the very next
message is authorized without restarting the service.

Add a command-shape matrix proving mixed-case/whitespace variants of
`/setadmin [target]` and exact `/access allow-chat` are recognized from
`CommandMatch.command`/`.args`; `/access`, `/access allow-chat extra`,
`/access allow-user`, and free text containing those words perform no
enrollment and fail closed.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest \
  tests/test_access_control.py \
  tests/test_auth_whitelist.py \
  tests/test_admin_bootstrap.py -q
```

Expected: FAIL because empty allowlists currently mean allow all.

- [ ] **Step 3: Implement pure access decisions and atomic env updates**

```python
class AccessOperation(str, Enum):
    NORMAL_MESSAGE = "normal_message"
    BOOTSTRAP_ADMIN = "bootstrap_admin"
    ENROL_CURRENT_CHAT = "enrol_current_chat"


@dataclass(frozen=True)
class IngressAccessRequest:
    sender_id: str
    chat_id: str
    chat_type: str
    command_match: CommandMatch | None


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    operation: AccessOperation
    reason_code: str
    prospective_allowed: bool


class IngressAccessPolicy:
    def __init__(
        self,
        *,
        admin_ids: frozenset[str],
        allowed_user_ids: frozenset[str],
        allowed_chat_ids: frozenset[str],
        mode: IngressAccessMode,
        admin_bootstrap_scope: Literal["any_chat", "p2p_only"],
    ) -> None:
        self._admins = admin_ids
        self._users = allowed_user_ids
        self._chats = allowed_chat_ids
        self._mode = mode
        self._admin_bootstrap_scope = admin_bootstrap_scope

    def decide(self, request: IngressAccessRequest) -> AccessDecision:
        match = request.command_match
        command = match.command if match is not None else ""
        arguments = match.args.strip().lower() if match is not None else ""
        if (
            not self._admins
            and command == "/setadmin"
            and (
                self._admin_bootstrap_scope == "any_chat"
                or request.chat_type == "p2p"
            )
        ):
            return AccessDecision(
                True,
                AccessOperation.BOOTSTRAP_ADMIN,
                "bootstrap_admin",
                True,
            )
        if (
            request.sender_id in self._admins
            and command == "/access"
            and arguments == "allow-chat"
        ):
            return AccessDecision(
                True,
                AccessOperation.ENROL_CURRENT_CHAT,
                "admin_chat_enrolment",
                True,
            )
        user_allowed = (
            request.sender_id in self._admins
            or request.sender_id in self._users
        )
        chat_allowed = request.chat_id in self._chats
        prospective_allowed = user_allowed and chat_allowed
        if self._mode is IngressAccessMode.LEGACY_ALLOW_ALL:
            return AccessDecision(
                True,
                AccessOperation.NORMAL_MESSAGE,
                "legacy_allow_all",
                prospective_allowed,
            )
        if self._mode is IngressAccessMode.SHADOW:
            return AccessDecision(
                True,
                AccessOperation.NORMAL_MESSAGE,
                (
                    "shadow_allowed"
                    if prospective_allowed
                    else "shadow_would_deny"
                ),
                prospective_allowed,
            )
        return AccessDecision(
            prospective_allowed,
            AccessOperation.NORMAL_MESSAGE,
            "allowed" if prospective_allowed else "access_not_enrolled",
            prospective_allowed,
        )


class AtomicEnvFileStore:
    def update_many(self, updates: Mapping[str, str]) -> None:
        """File lock, temp write, file fsync, replace, then parent-dir fsync."""
```

`IngressAccessPolicy` is published through an `IngressAccessPolicyProvider`
whose current value is an immutable snapshot. On successful enrollment, write
and fsync the `.env` transaction first, rebuild the complete snapshot from the
committed values, then atomically swap it into the provider. A failed disk write
does not change the live policy; a failed snapshot build keeps the old policy
and emits a blocking finding. No caller mutates the frozen ID sets in place.

After DG-1A approval, first `/setadmin` in P2P atomically writes
`ADMIN_USER_IDS`, `ALLOWED_USER_IDS`, and the current P2P
`ALLOWED_CHAT_IDS`, and the same commit updates `AGENTS.md`. Without approval,
preserve the existing any-chat bootstrap contract and its blocking readiness
finding. `/access allow-chat` accepts no arbitrary ID; a configured admin runs
it inside the target group. It is recognized only as canonical command
`/access` plus the single normalized argument `allow-chat`; extra arguments
never fall through to enrollment. `/setadmin [target]` is recognized by
canonical `/setadmin` while preserving its parsed target for
`AdminBootstrapService`: first bootstrap still assigns the sender, whereas an
existing authorized admin may replace the single admin. No policy code compares
or reparses raw/normalized full text. Ordinary messages require both the user
and chat dimensions.
`legacy_allow_all` is an explicit, audited break-glass rollback only; startup
and `/status` must display the Task 2 warning while it is active. `shadow`
executes the legacy allow path once, logs only hashed IDs and
`prospective_allowed`, and never auto-populates allowlists from observed
traffic. Require 48 hours of zero unexplained `shadow_would_deny` before
enforcement.

- [ ] **Step 4: Place authorization before all message side effects**

Run:

```bash
uv run python -m pytest \
  tests/test_access_control.py \
  tests/test_auth_whitelist.py \
  tests/test_admin_bootstrap.py \
  tests/test_ws_client_routing.py -q
```

Expected: PASS; denied messages cause zero GroupLedger, image, scheduler, Shell,
or handler calls.

- [ ] **Step 5: Commit**

```bash
git add src/access_control.py src/config/env_file_store.py src/admin_bootstrap.py src/feishu/ws_client.py src/feishu/handlers/system.py tests/test_access_control.py tests/test_auth_whitelist.py tests/test_admin_bootstrap.py
# Run the next line only when DG-1A approved the P2P-only rule:
git add AGENTS.md
git commit -m "fix(auth): deny unenrolled Feishu ingress"
```

- [ ] **Step 6: Migrate existing deployments through shadow**

Fresh installs keep the secure `enforced` default. Existing deployments that
currently rely on empty-list allow-all must first deploy the committed code with
`INGRESS_ACCESS_MODE=shadow`, configure reviewed user/chat lists, and observe 48
hours with zero unexplained `shadow_would_deny`. The observation is bound to
commit SHA; no traffic-derived auto-enrollment is permitted.

### Task 4: Enforce explicit host-Shell authorization

**Files:**
- Create: `src/sandbox/access_policy.py`
- Create: `tests/test_shell_access_policy.py`
- Modify: `src/sandbox/__init__.py`
- Modify: `src/feishu/handlers/system.py`
- Modify: `src/card/ui_text.py`
- Modify: `tests/test_shell_repo_lock_strict.py`

**Interfaces:**
- Consumes: `SecurityPosture`, sender ID, chat ID, and P2P/group origin.
- Produces: `ShellRequestContext`, `ShellAccessDecision`, `ShellAccessPolicy.decide()`.

- [ ] **Step 1: Write failing authorization tests**

```python
@pytest.mark.parametrize(
    (
        "mode",
        "is_admin",
        "is_p2p",
        "user_listed",
        "chat_listed",
        "isolation_ready",
        "trusted_ack",
        "allowed",
    ),
    [
        ("disabled", True, True, True, True, False, False, False),
        ("admin_dm", True, True, False, False, False, False, True),
        ("admin_dm", True, False, False, False, False, False, False),
        ("admin_dm", False, True, False, False, False, False, False),
        ("allowlisted", False, False, True, True, False, False, True),
        ("allowlisted", False, False, True, False, False, False, False),
        ("allowlisted", False, False, False, True, False, False, False),
        ("isolated", False, False, False, False, False, False, False),
        ("isolated", False, False, False, False, True, False, False),
        ("isolated", False, False, True, True, True, False, True),
        ("trusted_local", False, False, False, False, False, False, False),
        ("trusted_local", False, False, False, False, False, True, False),
        ("trusted_local", True, True, False, False, False, True, True),
        ("trusted_local", False, False, True, True, False, True, True),
    ],
)
def test_shell_policy_authorizes_only_the_configured_scope(
    policy_factory,
    mode: str,
    is_admin: bool,
    is_p2p: bool,
    user_listed: bool,
    chat_listed: bool,
    isolation_ready: bool,
    trusted_ack: bool,
    allowed: bool,
) -> None:
    policy = policy_factory(
        mode=mode,
        is_admin=is_admin,
        user_listed=user_listed,
        chat_listed=chat_listed,
        isolation_ready=isolation_ready,
        trusted_ack=trusted_ack,
    )
    result = policy.decide(
        ShellRequestContext(
            sender_id="ou_user",
            chat_id="oc_chat",
            is_p2p=is_p2p,
        )
    )
    assert result.allowed is allowed
```

Add a handler regression asserting `SandboxExecutor.execute()` is never called after a denied decision.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest tests/test_shell_access_policy.py tests/test_shell_repo_lock_strict.py -q
```

Expected: FAIL because access policy does not exist.

- [ ] **Step 3: Implement policy and handler gate**

```python
@dataclass(frozen=True)
class ShellRequestContext:
    sender_id: str
    chat_id: str
    is_p2p: bool


@dataclass(frozen=True)
class ShellAccessDecision:
    allowed: bool
    reason_code: str
    user_message: str


class ShellAccessPolicy:
    def __init__(
        self,
        *,
        mode: ShellAccessMode,
        admin_ids: frozenset[str],
        allowed_user_ids: frozenset[str],
        allowed_chat_ids: frozenset[str],
        trusted_local_ack: bool,
        isolation_ready: bool,
    ) -> None:
        self._mode = mode
        self._admins = admin_ids
        self._users = allowed_user_ids
        self._chats = allowed_chat_ids
        self._trusted_local_ack = trusted_local_ack
        self._isolation_ready = isolation_ready

    def decide(self, request: ShellRequestContext) -> ShellAccessDecision:
        if self._mode is ShellAccessMode.DISABLED:
            return ShellAccessDecision(False, "shell_disabled", "宿主机 Shell 已禁用")
        if self._mode is ShellAccessMode.ADMIN_DM:
            allowed = (
                request.is_p2p
                and request.sender_id in self._admins
            )
            return ShellAccessDecision(
                allowed,
                "allowed" if allowed else "admin_dm_required",
                "" if allowed else "仅管理员私聊可执行宿主机 Shell",
            )
        if self._mode is ShellAccessMode.ALLOWLISTED:
            allowed = (
                request.sender_id in self._users
                and request.chat_id in self._chats
            )
            return ShellAccessDecision(
                allowed,
                "allowed" if allowed else "shell_allowlist_denied",
                "" if allowed else "当前用户或会话未获 Shell 授权",
            )
        if self._mode is ShellAccessMode.TRUSTED_LOCAL:
            allowed = (
                self._trusted_local_ack
                and (
                    request.sender_id in self._admins
                    or (
                        request.sender_id in self._users
                        and request.chat_id in self._chats
                    )
                )
            )
            return ShellAccessDecision(
                allowed,
                "allowed" if allowed else "trusted_local_unacknowledged",
                "" if allowed else "需要显式确认 trusted_local 风险",
            )
        allowed = (
            self._mode is ShellAccessMode.ISOLATED
            and self._isolation_ready
            and (
                request.sender_id in self._admins
                or (
                    request.sender_id in self._users
                    and request.chat_id in self._chats
                )
            )
        )
        return ShellAccessDecision(
            allowed,
            (
                "allowed"
                if allowed
                else (
                    "shell_isolation_unavailable"
                    if not self._isolation_ready
                    else "shell_allowlist_denied"
                )
            ),
            (
                ""
                if allowed
                else (
                    "Shell 隔离后端不可用，已拒绝宿主执行"
                    if not self._isolation_ready
                    else "当前用户或会话未获隔离 Shell 授权"
                )
            ),
        )
```

Call the policy before repository lock acquisition and before
`SandboxExecutor.execute()`. Preserve SMART shell classification; denial is an
authorization result, not a fallback to another provider. Until Task 24 wires a
successfully probed isolation backend, `isolation_ready` is always false; setting
`shell_access_mode=isolated` must therefore deny, never invoke the current host
executor under an “isolated” label.

- [ ] **Step 4: Run focused routing and Shell tests**

Run:

```bash
uv run python -m pytest \
  tests/test_shell_access_policy.py \
  tests/test_shell_repo_lock_strict.py \
  tests/test_auth_whitelist.py \
  tests/test_ws_client_routing.py -q
```

Expected: PASS; all denied cases record zero executor calls.

- [ ] **Step 5: Commit**

```bash
git add src/sandbox/access_policy.py src/sandbox/__init__.py src/feishu/handlers/system.py src/card/ui_text.py tests/test_shell_access_policy.py tests/test_shell_repo_lock_strict.py
git commit -m "fix(shell): enforce explicit host access policy"
```

### Task 5: Make Employee Department enablement explicit

**Files:**
- Modify: `src/config/settings.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/autonomous/integration/test_employee_hire_composition.py`
- Modify: `tests/test_ws_client_routing.py`

**Interfaces:**
- Consumes: `Settings.employee_department_enabled`.
- Produces: disabled runtime blocker `employee_department_disabled`; `AUTONOMOUS_VISIBLE_EMPLOYEE_LIMIT` remains capacity only.

- [ ] **Step 1: Write the failing off-means-off tests**

```python
def test_employee_department_disabled_opens_no_durable_store(
    tmp_path: Path,
    employee_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    employee_settings.employee_department_enabled = False
    opened: list[str] = []
    monkeypatch.setattr(
        JournalWriter,
        "open",
        lambda *args, **kwargs: opened.append("journal"),
    )

    runtime = EmployeeDepartmentRuntime.from_settings(
        employee_settings,
        notification_link=lambda *args: None,
    )

    assert runtime.readiness().blockers == ("employee_department_disabled",)
    assert opened == []


def test_disabled_runtime_records_no_group_message(ws_client, group_message) -> None:
    ws_client.settings.employee_department_enabled = False
    ws_client._dispatch_message(group_message)
    ws_client._employee_department_runtime.record_group_event.assert_not_called()
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest \
  tests/autonomous/integration/test_employee_hire_composition.py \
  tests/test_ws_client_routing.py -q
```

Expected: FAIL because composition only checks visible employee limit.

- [ ] **Step 3: Gate composition before all durable resources**

At the first line of `EmployeeDepartmentRuntime.from_settings()` after argument validation:

```python
enabled = bool(getattr(settings, "employee_department_enabled", False))
if not enabled:
    if release_trust_provider is not None:
        release_trust_provider.close()
    return cls(blockers=("employee_department_disabled",))
```

Do not create Journal, Vault, audit log, GroupLedger, Channel supervisor, recovery thread, or callbacks when disabled. Update all production-like test fixtures that intend to exercise employees to set `employee_department_enabled=True`.

- [ ] **Step 4: Verify both profiles**

Run:

```bash
uv run python -m pytest \
  tests/autonomous/integration/test_employee_hire_composition.py \
  tests/test_ws_client_routing.py \
  tests/test_config_validation.py -q
uv run python -m src.main --validate
```

Expected: disabled and enabled profiles both PASS. Rollback is
`EMPLOYEE_DEPARTMENT_ENABLED=true`; capacity remains controlled separately.

- [ ] **Step 5: Commit**

```bash
git add src/config/settings.py src/autonomous/provisioning/composition.py src/feishu/ws_client.py tests/autonomous/integration/test_employee_hire_composition.py tests/test_ws_client_routing.py .env.example README.md
git commit -m "fix(autonomous): make employee runtime opt-in"
```

### Task 6: Scope and expire employee group context

**Files:**
- Create: `src/autonomous/context/group_retention.py`
- Create: `tests/autonomous/unit/test_group_retention.py`
- Create: `tests/autonomous/integration/test_group_retention_supervisor.py`
- Modify: `src/autonomous/context/group_ledger.py`
- Modify: `src/autonomous/context/__init__.py`
- Modify: `src/autonomous/journal/blob_store.py`
- Modify: `src/autonomous/membership/service.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/config/settings.py`
- Modify: `.env.example`
- Modify: `tests/test_config_validation.py`
- Modify: `tests/autonomous/integration/test_employee_hire_composition.py`
- Modify: `tests/autonomous/unit/test_group_context_ledger.py`
- Modify: `tests/autonomous/unit/test_blob_store.py`
- Create: `tests/autonomous/chaos/test_group_retention_recovery.py`

**Interfaces:**
- Produces: `GroupRecordOutcome`, `GroupLedgerDeletionReport`,
  `EmployeeDepartmentRuntime.accepts_group_context()`,
  `GroupContextLedger.tombstone_before()`,
  `GroupContextLedger.delete_chat()`,
  `GroupContextLedger.reconcile_pending_deletions()`, and
  `GroupRetentionService.run_once()`, `GroupRetentionSupervisor.start()`, and
  `GroupRetentionSupervisor.close()`.
- Consumes: canonical active employee/team membership projection and configured
  retention days, interval, and batch size.

- [ ] **Step 1: Write failing scope and tombstone tests**

```python
def test_unmanaged_group_is_not_written(
    runtime,
    journal_writer,
    blob_store,
) -> None:
    outcome = runtime.record_group_event(
        tenant_key="tenant",
        chat_id="oc_unmanaged",
        thread_id="",
        message_id="om_1",
        sender_id="ou_1",
        text="private project text",
    )
    assert outcome is GroupRecordOutcome.OUT_OF_SCOPE
    journal_writer.commit.assert_not_called()
    blob_store.stage_and_publish.assert_not_called()


def test_retention_anchors_tombstone_before_blob_release(
    ledger,
    writer,
    released: list[str],
) -> None:
    record = publish_old_record(ledger)
    report = ledger.tombstone_before(
        cutoff_timestamp=record.recorded_at + 1,
        limit=100,
    )
    assert report.tombstoned_keys == (record.dedup_key,)
    assert writer.get_last_frame().events[0].event_type == "group.event.payload_tombstoned"
    assert released == [record.payload_ref.blob_id]
```

The chaos test must crash after the tombstone anchor but before blob release,
reopen the runtime, and prove cleanup is retried without restoring readable
text. A batch-boundary test seeds `limit + 1` expired records, proves the first
call selects exactly `limit`, and proves the next call completes the remainder
without duplicate blob release.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest \
  tests/autonomous/unit/test_group_retention.py \
  tests/autonomous/chaos/test_group_retention_recovery.py -q
```

Expected: FAIL because scope and retention APIs do not exist.

- [ ] **Step 3: Implement membership scope and durable tombstones**

```python
class GroupRecordOutcome(str, Enum):
    RECORDED = "recorded"
    OUT_OF_SCOPE = "out_of_scope"
    DISABLED = "disabled"


@dataclass(frozen=True)
class GroupLedgerDeletionReport:
    selected_keys: tuple[str, ...]
    tombstoned_keys: tuple[str, ...]
    purged_blob_count: int
    pending_purge_count: int


class GroupContextLedger:
    def tombstone_before(
        self,
        *,
        cutoff_timestamp: float,
        limit: int,
    ) -> GroupLedgerDeletionReport:
        """Anchor payload tombstones, then release encrypted Blob references."""

    def delete_chat(
        self,
        *,
        tenant_key: str,
        chat_id: str,
        reason: str,
    ) -> GroupLedgerDeletionReport:
        """Tombstone one tenant/chat scope without touching another tenant."""

    def reconcile_pending_deletions(self) -> GroupLedgerDeletionReport:
        """Continue anchored tombstones whose Blob purge was interrupted."""


@dataclass(frozen=True)
class GroupRetentionPolicy:
    retention_days: int

    def cutoff(self, *, now: float) -> float:
        return now - self.retention_days * 86_400


class GroupRetentionService:
    def run_once(
        self,
        *,
        now: float,
        limit: int,
    ) -> GroupLedgerDeletionReport:
        cutoff = self._policy.cutoff(now=now)
        return self._ledger.tombstone_before(
            cutoff_timestamp=cutoff,
            limit=limit,
        )


class GroupRetentionSupervisor:
    def start(self) -> None:
        """Reconcile pending deletion, run one bounded sweep, then schedule."""

    def close(self, *, timeout: float) -> None:
        """Stop the monotonic scheduler and join it within the shutdown budget."""
```

`accepts_group_context()` must trust only canonical active membership/team state;
it must not call the network on the hot path. Journal metadata remains for audit,
while payload reads fail with a stable `group_event_payload_tombstoned` code.
Deletion order is fixed:

1. fsync and anchor the tenant/chat or retention tombstone;
2. make projections and `window()` exclude it;
3. release shared references, quarantine unreferenced Blob payloads, then purge;
4. reconcile anchored-but-unpurged entries on startup.

Anchor failure leaves the payload live; crash after anchoring never makes it
readable again. Transport replay cannot resurrect a tombstoned dedup key, and a
Blob still referenced by another live record is not purged. Membership removal,
group unbinding/archive, and bot-removed events invoke `delete_chat()`; scope
lookup failure fails closed without recording new text.

Configure secure finite defaults:

```text
EMPLOYEE_GROUP_CONTEXT_RETENTION_DAYS=30
GROUP_RETENTION_INTERVAL_SECONDS=3600
GROUP_RETENTION_BATCH_SIZE=500
```

`EmployeeDepartmentRuntime` owns exactly one supervisor when the department is
enabled. Startup first calls `reconcile_pending_deletions()`, then runs one
bounded retention batch before the periodic monotonic loop starts. Each tick
continues batches while the previous batch is full but yields between batches;
shutdown calls `close()` through the existing WS resource lifecycle. Disabled
Employee Department creates no supervisor/thread/store. Invalid, zero, or
unbounded settings fail validation.

- [ ] **Step 4: Run focused Autonomous tests**

Run:

```bash
uv run python -m pytest \
  tests/autonomous/unit/test_group_retention.py \
  tests/autonomous/unit/test_group_context_ledger.py \
  tests/autonomous/unit/test_blob_store.py \
  tests/autonomous/chaos/test_group_retention_recovery.py \
  tests/autonomous/integration/test_group_retention_supervisor.py \
  tests/autonomous/integration/test_employee_hire_composition.py \
  tests/test_config_validation.py -q
```

Expected: PASS; unmanaged groups create zero Blob/Journal writes, membership
deletion is tenant/chat-scoped, shared live Blob references survive, startup
reconciliation runs once, periodic expiry is bounded, and shutdown leaks no
retention thread.

- [ ] **Step 5: Commit**

```bash
git add src/autonomous/context/group_retention.py src/autonomous/context/group_ledger.py src/autonomous/context/__init__.py src/autonomous/journal/blob_store.py src/autonomous/membership/service.py src/autonomous/provisioning/composition.py src/feishu/ws_client.py src/config/settings.py .env.example tests/autonomous/unit/test_group_retention.py tests/autonomous/unit/test_group_context_ledger.py tests/autonomous/unit/test_blob_store.py tests/autonomous/chaos/test_group_retention_recovery.py tests/autonomous/integration/test_group_retention_supervisor.py tests/autonomous/integration/test_employee_hire_composition.py tests/test_config_validation.py
git commit -m "feat(autonomous): bound employee group context retention"
```

### Task 7: Close topic-engine to programming-mode misrouting

**Files:**
- Create: `src/feishu/mode_transition.py`
- Create: `tests/test_mode_transition.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/feishu/route_decision.py`
- Modify: `src/feishu/ws_card_action_handler.py`
- Modify: `src/feishu/action_registry.py`
- Modify: `src/feishu/handlers/system.py`
- Modify: `src/card/builders/system.py`
- Modify: `src/thread/manager.py`
- Modify: `tests/test_ws_client_routing.py`
- Modify: `tests/test_ws_client_patch.py`
- Modify: `tests/test_ws_card_action_handler.py`
- Modify: `tests/test_common_action_registry.py`
- Modify: `tests/test_handlers.py`
- Modify: `tests/test_card_builders.py`
- Modify: `tests/test_thread_manager.py`

**Interfaces:**
- Produces: `ModeTransitionRequest`, `ModeTransitionDecision`, `decide_mode_transition()`.
- Consumed by: current WS compatibility path immediately; RouteDecision cutover in Task 12.

- [ ] **Step 1: Write the failing engine/provider matrix**

```python
@pytest.mark.parametrize("engine", ["deep", "spec", "worktree", "workflow"])
@pytest.mark.parametrize("command", sorted(PROGRAMMING_ENTRY_TOKENS))
def test_programming_entry_is_rejected_inside_topic_engine(
    engine: str,
    command: str,
) -> None:
    decision = decide_mode_transition(
        ModeTransitionRequest(
            current_topic_engine=engine,
            requested_command=SlashCommandParser.parse(command),
        )
    )
    assert decision.action is ModeTransitionAction.REJECT
    assert decision.reason_code == "topic_engine_active"
```

Add an integration assertion that no ACP session is created and the next
free-text message remains routed to the original engine. Add stale
model-selection, TUI2ACP adapter-selection, and TUI2ACP custom-command card
callbacks. The cards must have been rendered before the topic engine became
active, then be clicked after activation, proving that all three callbacks
reload live context rather than trust stale card state. Add direct
`ThreadContextManager.register()`, `bind_engine()`, and `update_mode()`
regressions so none can overwrite an active engine context with a programming
mode.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest \
  tests/test_mode_transition.py \
  tests/test_ws_client_routing.py \
  tests/test_ws_client_patch.py -q
```

Expected: the integration test fails because interceptable commands bypass the
existing later guard.

- [ ] **Step 3: Implement and place the guard before interception**

```python
class ModeTransitionAction(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"


@dataclass(frozen=True)
class ModeTransitionRequest:
    current_topic_engine: str | None
    requested_command: CommandMatch


@dataclass(frozen=True)
class ModeTransitionDecision:
    action: ModeTransitionAction
    reason_code: str = ""


def decide_mode_transition(
    request: ModeTransitionRequest,
) -> ModeTransitionDecision:
    engine = (request.current_topic_engine or "").strip().lower()
    command = canonicalize_programming_entry(request.requested_command)
    if engine in {"deep", "spec", "worktree", "workflow"}:
        if command is not None:
            return ModeTransitionDecision(
                ModeTransitionAction.REJECT,
                "topic_engine_active",
            )
    return ModeTransitionDecision(ModeTransitionAction.ALLOW)
```

For this P0 fix, define one exhaustive temporary
`PROGRAMMING_ENTRY_ALIASES` mapping in `mode_transition.py`: each canonical
`/coco`, `/claude`, `/aiden`, `/codex`, `/gemini`, `/traex`, `/ttadk`,
`/tui2acp`, and `/acp` entry plus every retained `/enter_<tool>` alias.
`PROGRAMMING_ENTRY_TOKENS` is derived from its keys, and the matrix exercises
both canonical and alias forms through `SlashCommandParser.parse()`. Task 9
moves this metadata into the product catalog so the temporary table does not
survive convergence.

Call this before `_is_interceptable_command_match()` in the topic path. This
task does not stop, exit, or mutate any engine; it restores the existing “reply
in a new topic” contract. `ThreadContextManager.register()`, `bind_engine()`,
and `update_mode()` all apply the same engine→programming invariant as the final
state-write boundary; `register()` may not replace an existing engine-bound
context merely because a programming handler tries to register the topic
again.

Every provider/model card and both TUI2ACP cards carry the originating
`thread_root_id`. `src/feishu/action_registry.py` forwards it to
`SystemHandler`; the handler reloads the live `ThreadContext` before showing or
committing a provider, adapter, or custom command. A legacy card without a
thread root may proceed only when the callback can derive the root from the
current message context; otherwise it fails closed with a fresh-card
instruction. Stale cards return `topic_engine_active` and create no ACP or
TUI2ACP session. Card values never authorize the transition by themselves.

- [ ] **Step 4: Verify the complete transition matrix**

Run:

```bash
uv run python -m pytest \
  tests/test_mode_transition.py \
  tests/test_ws_client_routing.py \
  tests/test_ws_client_patch.py \
  tests/test_ws_card_action_handler.py \
  tests/test_common_action_registry.py \
  tests/test_handlers.py \
  tests/test_card_builders.py \
  tests/test_thread_manager.py \
  tests/test_workflow_topic_engine.py -q
```

Expected: PASS with zero new ACP sessions in rejected cases.

- [ ] **Step 5: Commit**

```bash
git add src/feishu/mode_transition.py src/feishu/ws_client.py src/feishu/route_decision.py src/feishu/ws_card_action_handler.py src/feishu/action_registry.py src/feishu/handlers/system.py src/card/builders/system.py src/thread/manager.py tests/test_mode_transition.py tests/test_ws_client_routing.py tests/test_ws_client_patch.py tests/test_ws_card_action_handler.py tests/test_common_action_registry.py tests/test_handlers.py tests/test_card_builders.py tests/test_thread_manager.py
git commit -m "fix(routing): reject programming entry in engine topics"
```

### Task 8: Repair onboarding and public command contradictions

**Files:**
- Create: `ux/agent-department-onboarding.html`
- Modify: `src/slock_engine/card_templates/welcome.py`
- Modify: `src/feishu/handlers/system.py`
- Modify: `src/feishu/handlers/workflow.py`
- Modify: `src/project_chat/service.py`
- Modify: `src/card/ui_text.py`
- Modify: `README.md`
- Modify: `tests/test_slock_card_templates.py`
- Modify: `tests/test_handlers.py`
- Modify: `tests/test_workflow_topic_engine.py`
- Modify: `tests/test_project_chat/test_default_intent.py`

**Interfaces:**
- Produces: one truthful migration response from retired `/goal` to
  `/task <description>` and natural-language team submission.
- Preserves: retired Manager commands never accept work; `/hire` continues to
  reject arbitrary `--prompt`.

- [ ] **Step 1: Create and review the HTML preview**

The preview must show these exact journeys:

```html
<section data-role="admin-dm">
  <button>雇佣员工</button>
  <button>查看员工</button>
  <button>创建团队</button>
</section>
<section data-role="team-group">
  <strong>直接发送任务即可</strong>
  <p>/task &lt;描述&gt; 是显式入口；/goal 已退役</p>
  <p>/task status · /slock status · /role list</p>
</section>
<section data-scope="topic">
  <strong>Workflow 在当前根话题交互，并占用当前聊天+项目运行槽</strong>
</section>
```

Open the preview through the normal `ux/` review workflow and obtain product
approval before changing card JSON.

- [ ] **Step 2: Write failing contract tests**

```python
def test_retired_goal_points_to_the_canonical_task_entry(system_handler) -> None:
    system_handler.handle_intercepted_command(
        "om_1",
        "oc_1",
        "/goal 修复登录回归",
        None,
        command_match=SlashCommandParser.parse("/goal 修复登录回归"),
    )
    reply = system_handler.reply_text.call_args.args[1]
    assert "未执行" in reply
    assert "/task 修复登录回归" in reply


def test_welcome_card_leads_with_natural_language_task() -> None:
    card = build_welcome_card(team_name="研发")
    text = json.dumps(card, ensure_ascii=False)
    assert "直接发送任务" in text
    assert "/task" in text
    assert "/goal" not in text


def test_hire_usage_does_not_advertise_prompt() -> None:
    assert "--prompt" not in UI_TEXT["system_help_section_hire_body"]
```

Also assert Workflow copy distinguishes root-topic message routing from
`chat_id + project.root_path` execution/exclusion scope, and project-chat copy
renders the effective configured provider rather than hard-coded Coco.

- [ ] **Step 3: Run and verify RED**

Run:

```bash
uv run python -m pytest \
  tests/test_slock_card_templates.py \
  tests/test_handlers.py \
  tests/test_workflow_topic_engine.py \
  tests/test_project_chat/test_default_intent.py -q
```

Expected: FAIL on `/goal` migration copy, `--prompt`, Workflow scope, or
default-provider copy.

- [ ] **Step 4: Implement the exact migration behavior**

Keep `/goal` in
`SystemHandler._RETIRED_AUTONOMOUS_MANAGER_COMMANDS`; do not route it to Slock
and do not silently reinterpret an old durable-Goal command as an immediate
team assignment. Give `/goal <text>` a deterministic, non-executing migration
response that preserves the text in a suggested `/task <text>` command.

Remove `/goal` from welcome/onboarding discovery. Remove `--prompt` from
README/help/usage, document the supported controlled role/trait options, and
leave the existing runtime rejection in place. Workflow copy must say:

> 消息交互绑定当前根话题；运行实例及同项目排他约束按
> `chat_id + project.root_path` 管理；Workflow 不替换聊天/项目持久编程模式。

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run python -m pytest \
  tests/test_slock_card_templates.py \
  tests/test_handlers.py \
  tests/test_workflow_topic_engine.py \
  tests/test_project_chat/test_default_intent.py -q
git diff --check
```

Expected: PASS.

```bash
git add ux/agent-department-onboarding.html src/slock_engine/card_templates/welcome.py src/feishu/handlers/system.py src/feishu/handlers/workflow.py src/project_chat/service.py src/card/ui_text.py README.md tests/test_slock_card_templates.py tests/test_handlers.py tests/test_workflow_topic_engine.py tests/test_project_chat/test_default_intent.py
git commit -m "fix(product): align Agent Department onboarding"
```

---

## Phase 1 — Product Control Plane

### Task 9: Create one product-action catalog and reduce the visible Slash surface

**Files:**
- Create: `src/feishu/product_catalog.py`
- Create: `tests/test_product_catalog.py`
- Modify: `src/feishu/main_slash_commands.py`
- Modify: `src/feishu/slash_command_parser.py`
- Modify: `src/feishu/mode_transition.py`
- Modify: `src/config/settings.py`
- Modify: `tests/test_main_slash_commands.py`
- Modify: `tests/test_command_registry_contract.py`
- Modify: `tests/test_mode_transition.py`
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Produces: `ProductRole`, `ProductScope`, `CompatibilityBehavior`,
  `ProductAction`, `ResolvedProductCommand`, `PUBLIC_ACTIONS`,
  `COMPATIBILITY_COMMANDS`, `get_public_actions()`, and `resolve_command()`.
- Consumed by: main Slash reconciliation, menu/help in Task 10, routing parity tests.

- [ ] **Step 1: Write the failing catalog contract**

```python
def test_department_v1_surface_is_small_and_task_oriented() -> None:
    commands = {
        action.command
        for action in get_public_actions(surface="department_v1")
    }
    assert commands == {
        "/menu",
        "/help",
        "/task",
        "/status",
        "/stop",
        "/retry",
        "/approve",
        "/hire",
        "/employees",
        "/team",
        "/project",
    }
    assert len(commands) <= 12


def test_hidden_legacy_commands_remain_compatibility_inputs() -> None:
    assert tuple(sorted(COMPATIBILITY_COMMANDS)) == LEGACY_79_COMMAND_FIXTURE
    for command in LEGACY_79_COMMAND_FIXTURE:
        assert resolve_command(command) is not None
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest tests/test_product_catalog.py tests/test_main_slash_commands.py -q
```

Expected: FAIL because the visible surface is the exact 79-command catalog.

- [ ] **Step 3: Implement typed catalog and versioned rollout**

```python
class ProductRole(str, Enum):
    ADMIN = "admin"
    LEAD = "lead"
    MEMBER = "member"
    EMPLOYEE = "employee"


class ProductScope(str, Enum):
    ADMIN_DM = "admin_dm"
    PROJECT = "project"
    TEAM = "team"
    TOPIC = "topic"
    ADVANCED = "advanced"


class CompatibilityBehavior(str, Enum):
    PASSTHROUGH = "passthrough"
    REWRITE = "rewrite"
    RETIRED_MESSAGE = "retired_message"


@dataclass(frozen=True)
class ProductAction:
    action_id: str
    command: str
    label: str
    description: str
    usage: str
    aliases: tuple[str, ...]
    roles: frozenset[ProductRole]
    scopes: frozenset[ProductScope]
    compatibility: CompatibilityBehavior
    enters_programming_mode: bool = False
    public: bool = True


@dataclass(frozen=True)
class ResolvedProductCommand:
    action: ProductAction
    invoked_as: str
    arguments: str
    rewritten_command: str | None
    rewritten_arguments: str | None


def get_public_actions(
    *,
    surface: Literal["legacy", "department_v1"],
) -> tuple[ProductAction, ...]:
    actions = LEGACY_ACTIONS if surface == "legacy" else DEPARTMENT_V1_ACTIONS
    return tuple(action for action in actions if action.public)


def resolve_command(
    token: str,
    arguments: str = "",
) -> ResolvedProductCommand | None:
    """Resolve canonical, hidden, aliased, rewritten, and retired commands."""
```

Add
`main_slash_surface: Literal["legacy", "shadow", "department_v1"] = "legacy"`.
In `shadow`, reconcile the legacy 79-command surface, calculate the desired 11
without mutating it, and emit a structured diff. `MAIN_AGENT_COMMANDS` becomes a
function of the selected surface. `SlashCommandParser` consumes
`resolve_command()` for compatibility
metadata while preserving the existing handler command token for passthrough
entries. Move Task 7's temporary programming-entry alias map into
`ProductAction` metadata in this same commit;
`decide_mode_transition()` asks the catalog whether the resolved action enters
a programming mode. No handler or guard keeps a second alias/rewrite table.
After DG-1 evidence,
change the default to `department_v1` in the final cutover task.

Compatibility rewrites are exact:

```python
COMPATIBILITY_REWRITES = {
    "/projects": ("/project", "list"),
    "/new": ("/project", "new"),
    "/switch": ("/project", "switch"),
    "/deep_status": ("/status", "--source deep"),
    "/spec_status": ("/status", "--source spec"),
    "/wf_status": ("/status", "--source workflow"),
    "/stop_deep": ("/stop", "--source deep"),
    "/stop_spec": ("/stop", "--source spec"),
    "/stop_wf": ("/stop", "--source workflow"),
}
```

`/exit` remains programming-mode exit and must never rewrite to `/stop`.

- [ ] **Step 4: Verify public and compatibility surfaces**

Run:

```bash
uv run python -m pytest \
  tests/test_product_catalog.py \
  tests/test_main_slash_commands.py \
  tests/test_command_registry_contract.py \
  tests/test_mode_transition.py -q
```

Expected: PASS for both catalog versions; every compatibility command remains
routable or produces an explicit retired response. The exact sorted 79-command
fixture is captured from the DG-0 baseline, not reconstructed from the new
catalog.

- [ ] **Step 5: Commit**

```bash
git add src/feishu/product_catalog.py src/feishu/main_slash_commands.py src/feishu/slash_command_parser.py src/feishu/mode_transition.py src/config/settings.py tests/test_product_catalog.py tests/test_main_slash_commands.py tests/test_command_registry_contract.py tests/test_mode_transition.py .env.example README.md
git commit -m "feat(product): add versioned action catalog"
```

- [ ] **Step 6: Observe the committed Slash shadow**

Deploy the exact commit with `MAIN_SLASH_SURFACE=shadow`. Require the computed
11-command set to remain stable, every baseline compatibility token to resolve,
and zero unexplained parser/handler miss for 48 hours. Bind evidence to commit
SHA and tenant hash before Task 26 reconciles `department_v1`.

### Task 10: Build role-aware menu and help cards

**Files:**
- Create: `ux/agent-department-control-plane.html`
- Create: `src/card/product_menu.py`
- Create: `tests/test_product_menu_cards.py`
- Modify: `src/card/builders/system.py`
- Modify: `src/card/ui_text.py`
- Modify: `src/feishu/handlers/system.py`

**Interfaces:**
- Consumes: `ProductAction`, effective role, project/team/topic state.
- Produces: `ProductMenuView` and `SystemBuilder.build_product_menu_card(view)`.

- [ ] **Step 1: Build the production-shape HTML preview**

The preview must contain:

```html
<nav aria-label="GhostAP Agent Department">
  <button data-action="submit_task">提交任务</button>
  <button data-action="task_board">任务看板</button>
  <button data-action="employees">数字员工</button>
  <button data-action="projects">项目</button>
  <button data-action="approvals">审批与审计</button>
  <details>
    <summary>高级工具</summary>
    <button>选择 Provider / Model</button>
    <button>选择执行策略</button>
    <button>宿主机 Shell</button>
  </details>
</nav>
```

Preview admin DM, ordinary project group, managed team group, and topic context.
The same file must also preview the unified task card in four states:
`empty`, `running`, `waiting_approval`, and `degraded_source`, including mobile
width and the exact stop/retry/approve affordances. Task 13 may not implement
production task cards until these states are reviewed.

- [ ] **Step 2: Write failing role-visibility tests**

```python
def test_member_menu_hides_admin_and_host_shell_actions() -> None:
    view = ProductMenuView(
        role=ProductRole.MEMBER,
        scope=ProductScope.PROJECT,
        project_name="ghostAp",
    )
    card = json.loads(SystemBuilder.build_product_menu_card(view)[1])
    payload = json.dumps(card, ensure_ascii=False)
    assert "提交任务" in payload
    assert "雇佣员工" not in payload
    assert "宿主机 Shell" not in payload


def test_admin_dm_menu_exposes_security_and_roster() -> None:
    view = ProductMenuView(
        role=ProductRole.ADMIN,
        scope=ProductScope.ADMIN_DM,
    )
    payload = json.dumps(
        json.loads(SystemBuilder.build_product_menu_card(view)[1]),
        ensure_ascii=False,
    )
    assert "安全姿态" in payload
    assert "数字员工" in payload


def test_team_lead_menu_adds_team_controls_without_admin_controls() -> None:
    view = ProductMenuView(
        role=ProductRole.LEAD,
        scope=ProductScope.TEAM,
        team_name="平台组",
    )
    payload = json.dumps(
        json.loads(SystemBuilder.build_product_menu_card(view)[1]),
        ensure_ascii=False,
    )
    assert "团队任务" in payload
    assert "安全姿态" not in payload
```

- [ ] **Step 3: Run and verify RED**

Run:

```bash
uv run python -m pytest tests/test_product_menu_cards.py -q
```

Expected: FAIL because menu/help ignore the product role and help always renders all sections.

- [ ] **Step 4: Implement the view model and builder**

```python
@dataclass(frozen=True)
class ProductMenuView:
    role: ProductRole
    scope: ProductScope
    project_name: str | None = None
    team_name: str | None = None
    topic_engine: str | None = None
    security_finding_codes: tuple[str, ...] = ()
```

The builder filters `ProductAction.roles`, shows scope badges (`项目持久`,
`话题交互 / 项目运行`, `团队群`, `管理员`), and keeps advanced
provider/engine actions collapsed. Replace the ignored help `category`
parameter with explicit
`role/scope`; retain an adapter for old call sites until Task 11 cutover.
`ADMIN` must come from the existing admin authorization service and `LEAD` from
durable team ownership; role changes discovery only, and every action handler
still performs its own authorization.

- [ ] **Step 5: Verify schema and commit**

Run:

```bash
uv run python -m pytest \
  tests/test_product_menu_cards.py \
  tests/test_card_schema_contract.py \
  tests/test_handlers.py -q
git diff --check
```

Expected: PASS and all cards remain within Feishu size/node limits.

```bash
git add ux/agent-department-control-plane.html src/card/product_menu.py src/card/builders/system.py src/card/ui_text.py src/feishu/handlers/system.py tests/test_product_menu_cards.py
git commit -m "feat(card): add role-aware department navigation"
```

### Task 11: Resolve one effective interaction context

**Files:**
- Create: `src/feishu/effective_context.py`
- Create: `tests/test_effective_context.py`
- Modify: `tests/test_feishu_dispatcher.py`
- Modify: `src/feishu/request_context.py`
- Modify: `src/feishu/handler_context.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/feishu/dispatcher.py`
- Modify: `src/feishu/handlers/system.py`

**Interfaces:**
- Produces: `InteractionScope`, `EffectiveInteractionContext`, `EffectiveContextResolver.resolve()`.
- Consumed by: menu/help, RouteDecision, task-control filtering, Shell policy.

- [ ] **Step 1: Write the failing precedence matrix**

```python
def test_effective_context_resolves_all_scopes(resolver) -> None:
    context = resolver.resolve(
        RequestContext(
            message_id="om_1",
            chat_id="oc_1",
            text="修复登录",
            chat_type="group",
            sender_id="ou_admin",
            project=project("p1"),
        ),
        thread_root_id="omt_1",
    )
    assert context.project_id == "p1"
    assert context.persistent_mode is InteractionMode.CODEX
    assert context.topic_engine == "workflow"
    assert context.team_id == "team_1"
    assert context.is_admin is True
    assert context.scope is InteractionScope.TOPIC
```

Add cases for admin DM, plain project group, managed team group, employee DM,
missing project, and stale thread mapping.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest tests/test_effective_context.py -q
```

Expected: FAIL because each handler currently assembles partial state independently.

- [ ] **Step 3: Implement the frozen context**

```python
class InteractionScope(str, Enum):
    ADMIN_DM = "admin_dm"
    PROJECT = "project"
    TEAM = "team"
    TOPIC = "topic"
    EMPLOYEE_DM = "employee_dm"
    UNBOUND = "unbound"


class TopicEngine(str, Enum):
    DEEP = "deep"
    SPEC = "spec"
    WORKTREE = "worktree"
    WORKFLOW = "workflow"


@dataclass(frozen=True)
class EffectiveInteractionContext:
    tenant_key: str
    sender_id: str
    chat_id: str
    chat_type: str
    project_id: str | None
    thread_root_id: str | None
    persistent_mode: InteractionMode
    topic_engine: TopicEngine | None
    team_id: str | None
    employee_id: str | None
    is_admin: bool
    scope: InteractionScope


class EffectiveContextResolver(Protocol):
    def resolve(
        self,
        request: RequestContext,
        *,
        thread_root_id: str | None,
    ) -> EffectiveInteractionContext:
        """Resolve project, mode, topic, team, employee and authority once."""
```

Resolve once at ingress and attach it to `RequestContext.effective`. Preserve the
existing project-over-chat mode precedence. Do not perform network calls.
Add `RequestContext.is_image_only: bool`, set from the existing WS image
classification before dispatch; caption/OCR enhancement may populate `text` but
must not erase this original message-kind signal.
`FeishuRequestContext` remains a one-release compatibility wrapper but must carry
the same `effective` object; it may not resolve project/thread/mode/authority a
second time.

- [ ] **Step 4: Convert dispatcher, status/help, and Shell consumers**

Replace their local mode/admin/P2P assembly with `ctx.effective`. Wire
`MessageDispatcher` to consume that object through its compatibility context,
and add an assertion that one inbound message calls
`EffectiveContextResolver.resolve()` exactly once. Then run:

```bash
uv run python -m pytest \
  tests/test_effective_context.py \
  tests/test_feishu_dispatcher.py \
  tests/test_handlers.py \
  tests/test_shell_access_policy.py \
  tests/test_ws_client_routing.py -q
```

Expected: PASS; help now displays the effective project mode.

- [ ] **Step 5: Commit**

```bash
git add src/feishu/effective_context.py src/feishu/request_context.py src/feishu/handler_context.py src/feishu/ws_client.py src/feishu/dispatcher.py src/feishu/handlers/system.py tests/test_effective_context.py tests/test_feishu_dispatcher.py
git commit -m "refactor(routing): resolve one effective context"
```

### Task 12: Shadow and cut over immutable RouteDecision

**Files:**
- Create: `src/feishu/route_executor.py`
- Create: `tests/test_route_executor.py`
- Modify: `src/feishu/route_decision.py`
- Modify: `src/feishu/dispatcher.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/config/settings.py`
- Modify: `.env.example`
- Modify: `tests/test_route_decision.py`
- Modify: `tests/test_ws_client_routing.py`
- Modify: `tests/test_config_validation.py`

**Interfaces:**
- Consumes: `RequestContext.effective`.
- Produces: `RouteDecisionMode`, `RouteReason`, `LegacyRouteTrace`, complete
  `resolve_route()`, `RouteExecutionRequest`, `RouteExecutor.execute()`, and
  structured shadow divergence without a second side-effect path.

- [ ] **Step 1: Write the failing route truth table**

```python
@pytest.mark.parametrize(
    ("scope", "text", "expected"),
    [
        ("topic:workflow", "继续", RouteTarget.TOPIC_ENGINE),
        ("topic:workflow", "/codex", RouteTarget.REPLY_TEXT),
        ("programming:codex", "/exit", RouteTarget.EXIT_MODE),
        ("team", "/task status", RouteTarget.SLOCK_COMMAND),
        ("project", "git status", RouteTarget.SHELL),
        ("project", "修复登录", RouteTarget.INTENT_RECOGNITION),
        ("project", "/unknown", RouteTarget.REPLY_TEXT),
    ],
)
def test_route_truth_table(scope, text, expected, request_factory, dispatcher):
    decision = dispatcher.resolve_route(request_factory(scope=scope, text=text))
    assert decision.target is expected


@pytest.mark.parametrize("enhanced_text", ["", "图片内容：终端报错"])
def test_image_only_signal_survives_text_enhancement(
    enhanced_text,
    request_factory,
    dispatcher,
) -> None:
    decision = dispatcher.resolve_route(
        request_factory(
            scope="project",
            text=enhanced_text,
            is_image_only=True,
        )
    )
    assert decision.target is RouteTarget.IMAGE_MESSAGE
```

Add executor tests proving exactly one handler is invoked per decision and replies
contain no raw exception or secret. Add a payload mutation regression proving
that changing the caller's source dictionary after `RouteDecision` construction
cannot change the decision and direct mutation raises `TypeError`.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest \
  tests/test_route_decision.py \
  tests/test_route_executor.py \
  tests/test_ws_client_routing.py -q
```

Expected: FAIL because `resolve_command_route()` is incomplete and production still
uses imperative `process_with_intent()`.

- [ ] **Step 3: Implement full decision and side-effect executor**

```python
class RouteDecisionMode(str, Enum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    ENFORCED = "enforced"


class RouteReason(str, Enum):
    EXPLICIT_COMMAND = "explicit_command"
    TOPIC_ENGINE = "topic_engine"
    SMART_SHELL = "smart_shell"
    DEPARTMENT_CONTEXT = "department_context"
    PROGRAMMING_MODE = "programming_mode"
    IMAGE_MESSAGE = "image_message"
    DEFAULT_INTENT = "default_intent"
    INVALID_CONTEXT = "invalid_context"


@dataclass(frozen=True)
class LegacyRouteTrace:
    target: RouteTarget
    reason: RouteReason


@dataclass(frozen=True)
class RouteDecision:
    target: RouteTarget
    reason: RouteReason
    payload: Mapping[str, object] = field(default_factory=dict)
    reactions: tuple[str, ...] = ()
    reply_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_route_payload(self.payload))


@dataclass(frozen=True)
class RouteExecutionRequest:
    message_id: str
    text: str
    is_image_only: bool
    project: ProjectContext | None
    command_match: CommandMatch | None
    effective: EffectiveInteractionContext

    @classmethod
    def from_request_context(
        cls,
        request: RequestContext,
    ) -> "RouteExecutionRequest":
        """Freeze the handler inputs resolved for this one inbound message."""


class RouteExecutor(Protocol):
    def execute(
        self,
        request: RouteExecutionRequest,
        decision: RouteDecision,
    ) -> None:
        """Execute exactly one already-authorized routing target."""
```

`MessageDispatcher.resolve_route()` must be pure. In `shadow`, execute the legacy
path once. Instrument every terminal branch of that legacy path to emit one
`LegacyRouteTrace` immediately before its existing side effect; compare that
trace with the new decision after dispatch. Do not call the legacy resolver a
second time. Log only hashed/stable IDs and reason codes. In `enforced`, execute
only `RouteExecutor`. Never run both side-effect paths.

Build `RouteExecutionRequest` exactly once from the same `RequestContext` used
for the decision. It deliberately carries the existing `ProjectContext` and
`CommandMatch` objects required by current handler APIs while all authority,
scope, mode, chat, sender, and thread decisions come from the frozen
`effective` object. `RouteDecision.payload` contains only target-specific,
JSON-safe opaque values; `freeze_route_payload()` recursively copies mappings
to read-only mappings, lists to tuples, and rejects non-JSON-safe/mutable
objects. It is not a substitute for message text, project, image kind, or
authority context. Executor tests must invoke representative Shell,
programming, topic-engine, task-control, image, and reply handlers with their
real signatures and prove that none re-resolves project or authorization.

The fixed precedence is: explicit command → topic engine → SMART Shell
classification → managed Department/Slock context → persistent programming mode
→ image-only handling → default intent. Unknown Slash input always returns a
help/error decision and never reaches NLI.

- [ ] **Step 4: Verify the shadow candidate locally**

Run shadow tests:

```bash
uv run python -m pytest \
  tests/test_route_decision.py \
  tests/test_route_executor.py \
  tests/test_ws_client_routing.py \
  tests/test_ws_client_patch.py \
  tests/test_config_validation.py \
  tests/test_workflow_topic_engine.py -q
```

Expected: PASS with zero test-fixture divergence.

- [ ] **Step 5: Commit the immutable shadow candidate**

```bash
git add src/feishu/route_executor.py src/feishu/route_decision.py src/feishu/dispatcher.py src/feishu/ws_client.py src/config/settings.py .env.example tests/test_route_decision.py tests/test_route_executor.py tests/test_ws_client_routing.py tests/test_config_validation.py
git commit -m "refactor(routing): add immutable route shadow"
```

- [ ] **Step 6: Observe the committed candidate**

Deploy that exact commit with `FEISHU_ROUTE_DECISION_MODE=shadow` to a controlled
tenant. Require zero unexplained divergence for 48 hours and bind the observation
to commit SHA, tenant hash, release version, and time window. Do not set
`enforced` in this task; Task 26 performs that cutover after all product gates
pass. Retain `legacy` for one release.

### Task 13: Add a unified task read and control plane

**Files:**
- Create: `src/tasking/control_plane.py`
- Create: `src/tasking/adapters/__init__.py`
- Create: `src/tasking/adapters/scheduler.py`
- Create: `src/tasking/adapters/engines.py`
- Create: `src/tasking/adapters/slock.py`
- Create: `src/tasking/adapters/department.py`
- Create: `src/tasking/adapters/employee.py`
- Create: `tests/test_task_control_plane.py`
- Create: `src/feishu/handlers/control_plane.py`
- Create: `tests/test_control_plane_handler.py`
- Create: `src/card/builders/control_plane.py`
- Modify: `src/feishu/handlers/diagnostics.py`
- Modify: `src/feishu/handlers/diagnostics_helper.py`
- Modify: `src/feishu/handlers/system.py`
- Modify: `src/feishu/handler_context.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/feishu/route_decision.py`
- Modify: `src/feishu/route_executor.py`
- Modify: `src/feishu/action_registry.py`
- Modify: `src/feishu/ws_card_action_handler.py`
- Modify: `src/card/actions/dispatch.py`
- Modify: `src/card/models.py`
- Modify: `src/card/builders/system.py`
- Modify: `tests/test_handlers.py`
- Modify: `tests/test_route_decision.py`
- Modify: `tests/test_route_executor.py`
- Modify: `tests/test_ws_card_action_handler.py`
- Modify: `tests/test_action_dispatcher.py`

**Interfaces:**
- Produces: `TaskKind`, `TaskRef`, `TaskLifecycleState`, `TaskCapability`,
  `TaskSummary`, `TaskQuery`, `TaskSnapshot`, `TaskControlRequest`,
  `TaskControlAdapter`, and `TaskControlPlane`.
- Consumes: existing Scheduler, Deep, Spec, Worktree, Workflow, Slock TeamRun, and Employee attempt APIs through adapters only.
- Consumes UI: the reviewed task-board states in
  `ux/agent-department-control-plane.html` from Task 10.

- [ ] **Step 1: Write the failing cross-engine projection test**

```python
def test_task_board_aggregates_every_runtime(control_plane) -> None:
    snapshot = control_plane.list_tasks(
        TaskQuery(
            scope=TaskScope(
                tenant_key="tenant",
                chat_id="oc_1",
                project_id="p1",
                thread_root_id=None,
            ),
            include_terminal=False,
        )
    )
    assert {task.ref.kind for task in snapshot.tasks} == {
        TaskKind.SCHEDULER,
        TaskKind.PROGRAMMING,
        TaskKind.DEEP,
        TaskKind.SPEC,
        TaskKind.WORKTREE,
        TaskKind.WORKFLOW,
        TaskKind.SLOCK,
        TaskKind.DEPARTMENT,
        TaskKind.TEAM,
        TaskKind.EMPLOYEE,
    }
    assert all(task.ref.task_id and task.state for task in snapshot.tasks)


def test_control_rejects_unsupported_action(control_plane) -> None:
    result = control_plane.control(
        TaskControlRequest(
            ref=TaskRef(
                kind=TaskKind.DEEP,
                task_id="deep_1",
                run_id="run_1",
            ),
            action=TaskCapability.APPROVE,
            requester=requester("ou_admin"),
            approval_id=None,
        ),
    )
    assert result.accepted is False
    assert result.reason_code == "capability_not_supported"
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest tests/test_task_control_plane.py tests/test_handlers.py -q
```

Expected: FAIL because `/status` only aggregates Deep/Spec and task control has no common interface.

- [ ] **Step 3: Implement the read/control domain**

```python
class TaskKind(str, Enum):
    SCHEDULER = "scheduler"
    PROGRAMMING = "programming"
    DEEP = "deep"
    SPEC = "spec"
    WORKTREE = "worktree"
    WORKFLOW = "workflow"
    SLOCK = "slock"
    DEPARTMENT = "department"
    TEAM = "team"
    EMPLOYEE = "employee"


class TaskLifecycleState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    CANCELLING = "cancelling"
    NEEDS_ATTENTION = "needs_attention"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELED = "canceled"
    UNKNOWN = "unknown"


class TaskCapability(str, Enum):
    STOP = "stop"
    RETRY = "retry"
    RECOVER = "recover"
    APPROVE = "approve"


@dataclass(frozen=True)
class TaskRef:
    kind: TaskKind
    task_id: str
    run_id: str | None = None


@dataclass(frozen=True)
class TaskScope:
    tenant_key: str
    chat_id: str
    project_id: str | None
    thread_root_id: str | None


@dataclass(frozen=True)
class TaskQuery:
    scope: TaskScope
    selector: str | None = None
    include_terminal: bool = False
    kinds: frozenset[TaskKind] = frozenset()
    limit: int = 50


@dataclass(frozen=True)
class TaskSummary:
    ref: TaskRef
    state: TaskLifecycleState
    title: str
    scope: TaskScope
    started_at: float | None
    updated_at: float | None
    capabilities: frozenset[TaskCapability]
    approval_ids: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class TaskSnapshot:
    tasks: tuple[TaskSummary, ...]
    degraded_kinds: tuple[TaskKind, ...] = ()


@dataclass(frozen=True)
class TaskRequester:
    actor_id: str
    scope: TaskScope
    idempotency_key: str


@dataclass(frozen=True)
class TaskControlRequest:
    ref: TaskRef
    action: TaskCapability
    requester: TaskRequester
    approval_id: str | None = None


@dataclass(frozen=True)
class TaskControlResult:
    accepted: bool
    task: TaskSummary | None
    reason_code: str


class TaskControlAdapter(Protocol):
    kind: TaskKind

    def list_tasks(self, query: TaskQuery) -> tuple[TaskSummary, ...]:
        """Return in-scope tasks without mutating their native runtime."""

    def get_task(
        self,
        ref: TaskRef,
        *,
        scope: TaskScope,
    ) -> TaskSummary | None:
        """Return one authoritative in-scope task projection or None."""

    def control(
        self,
        request: TaskControlRequest,
    ) -> TaskControlResult:
        """Reauthorize and execute one supported idempotent control action."""
```

Adapters may map native states but must not mutate engine internals. Unknown or
unrecoverable state maps to `UNKNOWN`, never to success. `TaskLifecycleState` is
the user-facing projection of Task 17's `RunStatus`; adapters own that mapping
and the two enums must not become competing write models. `TaskControlPlane`
raises typed `TaskNotFound`, `TaskAmbiguous`, and `TaskOutOfScope` errors instead
of guessing a short ID. It catches adapter read failures into
`TaskSnapshot.degraded_kinds`, so “source unavailable” cannot be rendered as “no
tasks”.

- [ ] **Step 4: Replace task/status/control entry points**

Render one card grouped by view buckets, not new persisted states:

- `ACTIVE`: `QUEUED`, `RUNNING`, `CANCELLING`;
- `NEEDS_ATTENTION`: `PAUSED`, `WAITING_APPROVAL`, `NEEDS_ATTENTION`,
  `PARTIAL`, `FAILED`, `UNKNOWN`;
- `DONE`: `SUCCEEDED`, `CANCELED`.

Buttons come only from `TaskSummary.capabilities`.

Exact command behavior:

- `/task` and `/task list` list tasks.
- `/task status [selector]` delegates to `/status`.
- `/task <other text>` submits a natural-language department task through the
  existing SMART/Team route selected by `EffectiveInteractionContext`.
- `/stop`, `/retry`, and `/approve` require an exact selector unless exactly one
  in-scope task supports that action.
- Card callbacks carry the complete `TaskRef`, action, optional `approval_id`,
  and an idempotency key in one immutable `TaskControlRequest`; the handler
  reloads authoritative scope, actor, state, approval, and capabilities.
- If a task has multiple pending approvals, `/approve` without the exact
  `approval_id` is ambiguous and performs no mutation. An approval ID missing
  from the freshly loaded `TaskSummary.approval_ids`, already resolved, or
  belonging to another task/tenant is rejected as stale or out of scope.
- Provider failure marks that source degraded; it must not make a partially
  unavailable task board look empty.

Add handler and adapter regressions for two simultaneous approvals, a stale
approval callback, and a foreign-task/foreign-tenant approval ID. Assert that
the native adapter receives the exact approved ID only after scope and
capability reauthorization, and receives no call in every rejected case.

The submission fallback is deterministic:

1. employee runtime enabled + managed team context → existing Team task path;
2. otherwise, a bound project → existing SMART single-agent path, labeled
   `单 Agent 执行` rather than “部门任务”;
3. no bound project → reject with `/project new` or `/project switch` guidance.

When `employee_department_enabled=False`, `/task` remains useful through the
single-agent path, but hire/team controls show the explicit enablement blocker;
the UI must not pretend that employee collaboration occurred.

Register control actions in all three existing gates:
`src/feishu/action_registry.py`,
`src/feishu/ws_card_action_handler.py`, and
`src/card/actions/dispatch.py`. Inject `TaskControlPlane` through
the real `HandlerContext` definition in `src/feishu/handler_context.py`, and
wire its sole production construction in `src/feishu/ws_client.py`; handlers and
card dispatchers must not construct a second control registry. Remove `/approve` from
`SystemHandler._RETIRED_AUTONOMOUS_MANAGER_COMMANDS` only in this same commit,
after the durable task-control route and its fail-closed tests exist; all other
retired Manager commands stay retired.

Add `RouteTarget.TASK_CONTROL` in the immutable route model and route canonical
`/task`, `/status`, `/stop`, `/retry`, and `/approve` decisions to the new
handler. This deliberately supersedes Task 12's legacy/Slock compatibility
target only after the control plane exists.

Run:

```bash
uv run python -m pytest \
  tests/test_task_control_plane.py \
  tests/test_control_plane_handler.py \
  tests/test_handlers.py \
  tests/test_ws_card_action_handler.py \
  tests/test_action_dispatcher.py \
  tests/test_card_schema_contract.py \
  tests/test_deep_engine.py \
  tests/test_workflow_stop_button.py -q
```

Expected: PASS; Worktree/Workflow/Team/Employee activity appears in `/status`.

- [ ] **Step 5: Commit**

```bash
git add src/tasking/control_plane.py src/tasking/adapters src/feishu/handlers/control_plane.py src/feishu/handlers/diagnostics.py src/feishu/handlers/diagnostics_helper.py src/feishu/handlers/system.py src/feishu/handler_context.py src/feishu/ws_client.py src/feishu/route_decision.py src/feishu/route_executor.py src/feishu/action_registry.py src/feishu/ws_card_action_handler.py src/card/actions/dispatch.py src/card/models.py src/card/builders/control_plane.py src/card/builders/system.py tests/test_task_control_plane.py tests/test_control_plane_handler.py tests/test_handlers.py tests/test_route_decision.py tests/test_route_executor.py tests/test_ws_card_action_handler.py tests/test_action_dispatcher.py
git commit -m "feat(tasking): add unified task control plane"
```

---

## Phase 2 — Runtime Convergence

### Task 14: Introduce one backend capability catalog and session request

**Files:**
- Create: `src/agent_session/backend_catalog.py`
- Create: `src/agent_session/legacy_backend_trace.py`
- Create: `src/agent_session/request.py`
- Create: `tests/test_backend_catalog.py`
- Create: `tests/test_backend_shadow_parity.py`
- Create: `tests/test_backend_registry_boundary.py`
- Modify: `src/agent_session/factory.py`
- Modify: `src/agent_session/backend_resolver.py`
- Modify: `src/agent_session/model_diagnostics.py`
- Modify: `src/agent_session/wrappers.py`
- Modify: `src/agent_session/__init__.py`
- Modify: `src/acp/session_factory.py`
- Modify: `src/acp/manager.py`
- Modify: `src/acp/sync_adapter.py`
- Modify: `src/acp/startup_utils.py`
- Modify: `src/utils/engine_identity.py`
- Modify: `src/engine_base.py`
- Modify: `src/deep_engine/engine.py`
- Modify: `src/spec_engine/engine.py`
- Modify: `src/spec_engine/manager.py`
- Modify: `src/spec_engine/review_agents.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/feishu/session_hub.py`
- Modify: `src/feishu/handlers/workflow.py`
- Modify: `src/worktree_engine/tool_discovery.py`
- Modify: `src/workflow_engine/tool_registry.py`
- Modify: `src/config/settings.py`
- Modify: `.env.example`
- Modify: `tests/test_session_factory.py`
- Modify: `tests/test_review_session_factory.py`
- Modify: `tests/test_worktree_session_factory.py`
- Modify: `tests/test_acp_manager_consistency.py`
- Modify: `tests/test_acp_sync_adapter.py`
- Modify: `tests/test_acp_startup_utils.py`
- Modify: `tests/test_engine_identity.py`
- Modify: `tests/test_model_failure_failover.py`
- Modify: `tests/test_rate_limit.py`
- Modify: `tests/test_diagnostics_isolation.py`
- Modify: `tests/test_tui2acp_terminal_cleanup.py`
- Modify: `tests/test_ws_client_routing.py`
- Modify: `tests/test_session_hub.py`
- Modify: `tests/test_workflow_model_selection.py`
- Modify: `tests/test_deep_engine.py`
- Modify: `tests/test_spec_engine.py`
- Modify: `tests/test_spec_review_agent_selection.py`

**Interfaces:**
- Produces: `BackendTransport`, `BackendUse`, `BackendDescriptor`, `BackendSelection`, `BackendAdapter`, `BackendCatalog`, `SessionPurpose`, `SessionRequest`, `UnifiedSessionFactory`.
- Temporarily produces: side-effect-free `LegacyBackendTrace`,
  `BackendTraceComparison`, and `resolve_legacy_backend_trace()` for one
  release's independent shadow oracle.
- Preserves: current factory functions as one-release compatibility wrappers.

- [ ] **Step 1: Write failing registry and boundary tests**

```python
def test_each_supported_tool_resolves_for_declared_uses(catalog) -> None:
    for tool in (
        "coco",
        "claude",
        "aiden",
        "codex",
        "gemini",
        "traex",
        "ttadk",
        "tui2acp",
    ):
        selection = catalog.resolve(tool, use=BackendUse.PROGRAMMING)
        assert selection.tool_name == tool
        assert selection.capability_fingerprint


def test_unknown_backend_fails_without_generic_command_fallback(catalog) -> None:
    with pytest.raises(UnknownBackendError):
        catalog.resolve("unregistered-tool", use=BackendUse.ENGINE)


def test_compound_selection_options_are_preserved(catalog) -> None:
    selection = catalog.resolve(
        "traex",
        use=BackendUse.PROGRAMMING,
        selection_options={
            "profile": "balanced",
            "effort": "high",
        },
    )
    assert selection.selection_options == (
        ("effort", "high"),
        ("profile", "balanced"),
    )


@pytest.mark.parametrize(
    ("adapters_dir", "dir_args"),
    [
        (None, ()),
        ("/fixtures/tui2acp-adapters", (
            "--adapters-dir",
            "/fixtures/tui2acp-adapters",
        )),
    ],
)
def test_tui2acp_custom_command_preserves_exact_argv(
    catalog,
    monkeypatch,
    adapters_dir,
    dir_args,
) -> None:
    monkeypatch.setattr(
        sync_adapter,
        "_resolve_tui2acp_adapters_dir",
        lambda: adapters_dir,
    )
    selection = catalog.resolve(
        "tui2acp",
        use=BackendUse.PROGRAMMING,
        selection_options={
            "custom_command": 'aider --model "gpt 4o"',
        },
    )
    assert selection.command_argv == (
        "aider",
        "--model",
        "gpt 4o",
    )
    spec = resolve_agent_spec(selection)
    assert spec.command == "tui2acp"
    assert spec.args == (
        "--agent",
        "aider",
        "--unsafe",
        "--minimal",
        *dir_args,
        "--",
        "--model",
        "gpt 4o",
    )


def test_session_transport_consumers_use_the_catalog_boundary() -> None:
    violations = scan_session_transport_allowlists(
        SESSION_TRANSPORT_MODULES,
        permitted_registration_module=(
            ROOT / "src" / "agent_session" / "backend_catalog.py"
        ),
    )
    assert violations == []


def test_no_unregistered_production_session_constructor() -> None:
    violations = scan_direct_session_construction(
        ROOT / "src",
        permitted_adapter_modules=CATALOG_ADAPTER_IMPLEMENTATIONS,
        excluded_nonruntime_modules=NONRUNTIME_LABEL_AND_DOC_MODULES,
    )
    assert violations == []


def test_shadow_detects_catalog_mapping_mutation(
    legacy_trace_resolver,
    catalog,
) -> None:
    legacy = legacy_trace_resolver.resolve(
        tool_name="codex",
        use=BackendUse.PROGRAMMING,
        model_name="o3",
        selection_options={},
    )
    mutated_catalog = catalog_with_descriptor_override(
        catalog,
        backend_id=legacy.backend_id,
        transport=BackendTransport.CLI,
    )
    current = mutated_catalog.resolve(
        "codex",
        use=BackendUse.PROGRAMMING,
        model_name="o3",
    )
    comparison = compare_backend_trace(legacy, current)
    assert comparison.matches is False
    assert comparison.different_fields == ("transport",)
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest \
  tests/test_backend_catalog.py \
  tests/test_backend_shadow_parity.py \
  tests/test_backend_registry_boundary.py \
  tests/test_session_factory.py \
  tests/test_workflow_tool_registry.py \
  tests/test_worktree_tool_discovery.py -q
```

Expected: FAIL because providers and uses are duplicated across factories and engines.

- [ ] **Step 3: Implement descriptors and typed requests**

```python
class BackendTransport(str, Enum):
    ACP = "acp"
    CLI = "cli"
    TTADK_CLI = "ttadk_cli"
    TUI2ACP_BRIDGE = "tui2acp_bridge"


class BackendUse(str, Enum):
    PROGRAMMING = "programming"
    ENGINE = "engine"
    REVIEW = "review"
    WORKTREE = "worktree"
    WORKFLOW = "workflow"
    EMPLOYEE = "employee"


@dataclass(frozen=True)
class BackendDescriptor:
    backend_id: str
    tool_name: str
    display_name: str
    transport: BackendTransport
    uses: frozenset[BackendUse]
    supports_model_selection: bool
    supports_cancel: bool
    supports_resume: bool
    requires_tool_filter: bool
    capability_version: str
    priority: int = 0


@dataclass(frozen=True)
class BackendSelection:
    backend_id: str
    tool_name: str
    transport: BackendTransport
    use: BackendUse
    model_name: str | None
    selection_options: tuple[tuple[str, str], ...]
    capability_fingerprint: str
    command_argv: tuple[str, ...] = ()


@dataclass(frozen=True)
class LegacyBackendTrace:
    backend_id: str
    tool_name: str
    transport: BackendTransport
    use: BackendUse
    model_name: str | None
    selection_options: tuple[tuple[str, str], ...]
    command_argv: tuple[str, ...]


@dataclass(frozen=True)
class BackendTraceComparison:
    matches: bool
    different_fields: tuple[str, ...]


@dataclass(frozen=True)
class SessionRequest:
    selection: BackendSelection
    cwd: str
    purpose: SessionPurpose
    thread_id: str | None = None
    auto_approve: bool = False
    require_tool_filter: bool = False
    startup_timeout: float | None = None
    startup_retries: int | None = None
    startup_log_failures: bool | None = None
    on_rate_limit: Callable[[int], None] | None = None
    cancel_event: threading.Event | None = None
```

`BackendCatalog.resolve()` rejects equal-priority ambiguity and late registration
after `freeze()`. It normalizes provider-specific dimensions such as TTADK
variant, Traex profile/effort, and Codex effort into sorted immutable
`selection_options`; unsupported keys fail closed instead of being discarded.
`UnifiedSessionFactory.create()` is the only process/session construction
implementation. Old `create_*_session()` functions translate every argument
through the migration resolver and delegate to that constructor. Wrapper parity
tests must cover rate-limit callbacks, cancellation events,
`startup_log_failures`, startup timeout/retries, tool filtering, employee
environment capture, and model normalization so convergence does not silently
remove current behavior.

TUI2ACP is represented explicitly: public tool `tui2acp` with immutable
`adapter`/custom-command selection options, an immutable normalized
`command_argv`, and
`BackendTransport.TUI2ACP_BRIDGE`. It remains a persistent programming mode and
an ACP-protocol adapter launched by the `tui2acp` CLI; it is not reclassified as
direct TTADK CLI and is not granted engine uses unless a descriptor declares
them. `tui2acp_<adapter>` legacy agent types resolve through that descriptor.

For `custom_command`, parse once with `shlex.split()`; reject empty input,
invalid quoting, NUL, or an empty executable. Never invoke a shell and never
join the tokens back into a command string. The first token becomes the
`--agent` value and every remaining token is forwarded, in order and with token
boundaries preserved, after a literal `--`:

```text
aider --model "gpt 4o"
→ tui2acp --agent aider --unsafe --minimal -- --model "gpt 4o"
```

When the bundled adapters directory resolves, preserve the current
`--adapters-dir <path>` pair before the downstream `--`; when it does not,
omit the pair. Tests inject both cases rather than depending on the checkout or
machine. Named adapters continue to use
`--adapter <name> --unsafe --minimal [--adapters-dir <path>]` and do not gain a
downstream `--` segment. `src/acp/sync_adapter.py` accepts the typed selection
(or an exact compatibility translation) instead of splitting and discarding
everything after the first word. `selection_options`,
`command_argv`, and the final process argv all contribute to the capability
fingerprint and shadow comparison, so an option-loss regression is a
divergence rather than a false match.

- [ ] **Step 4: Shadow-compare every existing consumer**

Add `backend_catalog_mode: Literal["legacy", "shadow", "enforced"] = "legacy"`.
Expose it through `src/config/settings.py` and `.env.example`.
`src/agent_session/legacy_backend_trace.py` freezes the pre-migration
provider/transport/model/options/argv mapping as a pure resolver. It performs no
tool discovery, subprocess launch, network access, or session creation, and it
must not call the new catalog. Its normalized `LegacyBackendTrace` is an
independent oracle:

- `legacy`: resolve the legacy trace, translate it to `SessionRequest`, and
  construct exactly one session through `UnifiedSessionFactory`;
- `shadow`: resolve both legacy trace and `BackendSelection`, record one
  denominator and any field-level divergence, then execute only the
  legacy-derived request;
- `enforced`: execute only the catalog-derived request after verified evidence.

This preserves one side-effecting implementation without making the shadow
self-comparison tautological. Mutation tests change a catalog transport,
model, options, and TUI2ACP argv independently and prove that the frozen legacy
trace produces a typed divergence. Task 22 attaches counters to that comparison
result without changing its decision. Keep the legacy oracle immutable through
the candidate observation window and delete it only in a later, separately
evidenced release.

The exact consumer boundary is:

- `src/agent_session/factory.py` owns all sync/engine/review/worktree adapter
  construction and compatibility wrappers;
- `src/acp/session_factory.py` and `src/acp/manager.py` delegate transport and
  model selection to the catalog;
- `src/acp/sync_adapter.py` constructs TUI2ACP process argv only from the typed
  selection and preserves downstream custom-command arguments exactly;
- `src/acp/startup_utils.py` selects retry/startup adapters from
  `SessionRequest.selection`, never by parsing `agent_type`;
- `src/agent_session/model_diagnostics.py` and `wrappers.py` preserve the
  original selection/request and perform diagnostic/model-failure rebuilds
  through the same unified factory; they never instantiate `SyncACPSession`
  directly or reconstruct transport from private `_agent_type`;
- `src/utils/engine_identity.py` becomes a compatibility projection over a
  catalog selection; `src/engine_base.py`, Deep, and Spec consume that
  selection rather than maintaining ACP/CLI/TTADK branches;
- `SpecEngineManager._resolve_engine_identity()` delegates every legacy
  engine-name/agent-type input to the catalog compatibility translator, and
  `ReviewAgentBinding.from_selection_item()` stores a typed review
  `BackendSelection`; neither retains provider-prefix transport logic;
- `src/feishu/ws_client.py` constructs one frozen catalog and unified factory,
  then injects the same instances into every `ACPSessionManager`;
- `src/feishu/session_hub.py` derives manager construction and cleanup from
  catalog descriptors (including TUI2ACP) instead of a private seven-tool list;
- `src/agent_session/backend_resolver.py` remains a one-release compatibility
  facade that delegates to the catalog;
- Worktree and Workflow discovery consume descriptor capabilities;
- Workflow model selection in `src/feishu/handlers/workflow.py` resolves
  `BackendUse.WORKFLOW` capabilities instead of
  `_WORKFLOW_ACP_MODEL_TOOLS`/`provider="acp"|"ttadk"`;
- Slock, Department/Team, and remaining handler call sites may keep calling
  compatibility wrappers but may not branch on provider names or transport.

`SESSION_TRANSPORT_MODULES` is the explicit set listed in this task:
`agent_session/factory.py`, `agent_session/backend_resolver.py`,
`agent_session/legacy_backend_trace.py`,
`agent_session/model_diagnostics.py`, `agent_session/wrappers.py`,
`acp/session_factory.py`, `acp/manager.py`, `acp/sync_adapter.py`,
`acp/startup_utils.py`, `utils/engine_identity.py`, `engine_base.py`, Deep and
Spec engine/manager/review selection, `feishu/ws_client.py`,
`feishu/session_hub.py`, Workflow handler, Worktree discovery, and Workflow
registry. The boundary
scan permits provider IDs and transport construction only in catalog adapter
registrations, the frozen legacy trace, and provider implementation modules.
It reports the file, line,
and pattern for every other hard-coded transport/creation allowlist or prefix
branch. It does not pretend that documentation, UI labels, templates, or
provider-specific adapter internals are transport decision consumers. Task 15
separately removes programming-mode/default lists. Run:

In addition to the explicit consumer list, an AST boundary scan walks all
production `src/` modules for direct session/manager constructors, `provider` or
`transport` decision literals, provider-prefix parsing, and calls to legacy
agent-spec resolution. Each hit must either delegate through the catalog or be
listed as a leaf adapter implementation with a test and a one-line reason;
display labels, serialized compatibility DTO fields, and provider-internal
process code are the only nonruntime exclusions. This full-root scan prevents a
new or previously missed transport consumer from escaping the signed shadow
gate.

```bash
uv run python -m pytest \
  tests/test_backend_catalog.py \
  tests/test_backend_shadow_parity.py \
  tests/test_backend_registry_boundary.py \
  tests/test_session_factory.py \
  tests/test_review_session_factory.py \
  tests/test_worktree_session_factory.py \
  tests/test_acp_manager_consistency.py \
  tests/test_acp_sync_adapter.py \
  tests/test_acp_startup_utils.py \
  tests/test_engine_identity.py \
  tests/test_model_failure_failover.py \
  tests/test_rate_limit.py \
  tests/test_diagnostics_isolation.py \
  tests/test_tui2acp_terminal_cleanup.py \
  tests/test_ws_client_routing.py \
  tests/test_session_hub.py \
  tests/test_workflow_model_selection.py \
  tests/test_workflow_tool_registry.py \
  tests/test_worktree_tool_discovery.py \
  tests/test_deep_engine.py \
  tests/test_spec_engine.py \
  tests/test_spec_review_agent_selection.py -q
```

Expected: PASS with identical tool/transport/model/options choices and
callback/cancel behavior for supported paths, including TUI2ACP.

- [ ] **Step 5: Commit**

```bash
git add src/agent_session/backend_catalog.py src/agent_session/legacy_backend_trace.py src/agent_session/request.py src/agent_session/factory.py src/agent_session/backend_resolver.py src/agent_session/model_diagnostics.py src/agent_session/wrappers.py src/agent_session/__init__.py src/acp/session_factory.py src/acp/manager.py src/acp/sync_adapter.py src/acp/startup_utils.py src/utils/engine_identity.py src/engine_base.py src/deep_engine/engine.py src/spec_engine/engine.py src/spec_engine/manager.py src/spec_engine/review_agents.py src/feishu/ws_client.py src/feishu/session_hub.py src/feishu/handlers/workflow.py src/worktree_engine/tool_discovery.py src/workflow_engine/tool_registry.py src/config/settings.py .env.example tests/test_backend_catalog.py tests/test_backend_shadow_parity.py tests/test_backend_registry_boundary.py tests/test_session_factory.py tests/test_review_session_factory.py tests/test_worktree_session_factory.py tests/test_acp_manager_consistency.py tests/test_acp_sync_adapter.py tests/test_acp_startup_utils.py tests/test_engine_identity.py tests/test_model_failure_failover.py tests/test_rate_limit.py tests/test_diagnostics_isolation.py tests/test_tui2acp_terminal_cleanup.py tests/test_ws_client_routing.py tests/test_session_hub.py tests/test_workflow_model_selection.py tests/test_deep_engine.py tests/test_spec_engine.py tests/test_spec_review_agent_selection.py
git commit -m "refactor(session): add backend capability catalog"
```

- [ ] **Step 6: Observe the committed catalog in shadow**

Deploy the exact commit with `BACKEND_CATALOG_MODE=shadow`. Require 48 hours of
zero unexplained backend/transport/model/options divergence and bind the report
to commit SHA and tenant hash before Task 26 may set `enforced`.

### Task 15: Remove default-provider and programming-mode drift

**Files:**
- Modify: `src/agent_session/backend_catalog.py`
- Modify: `src/feishu/handlers/system.py`
- Modify: `src/feishu/handlers/programming.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/agent/intent_recognizer.py`
- Modify: `src/project_chat/service.py`
- Modify: `tests/test_handlers.py`
- Modify: `tests/test_project_chat/test_default_intent.py`
- Modify: `tests/test_intent.py`

**Interfaces:**
- Produces: `PersistedBackendChoice`,
  `project_backend_choice_from_legacy_context()`, and module-level
  `resolve_default_programming_backend()` owned by
  `src/agent_session/backend_catalog.py`.
- Consumes: `BackendCatalog` from Task 14.

- [ ] **Step 1: Write failing provider matrix tests**

```python
@pytest.mark.parametrize("tool", SUPPORTED_PROGRAMMING_TOOLS)
@pytest.mark.parametrize("surface", ["smart", "project_chat"])
def test_project_and_smart_default_resolve_the_same_backend(
    tool: str,
    surface: str,
    default_backend_harness,
) -> None:
    choice = default_backend_harness.resolve(
        surface=surface,
        project_tool=None,
        configured_tool=tool,
    )
    assert choice.tool_name == tool
    assert choice.use is BackendUse.PROGRAMMING
```

`default_backend_harness` must invoke the real SMART and project-chat consumers,
not the shared helper directly. Add Traex `/exit` and `/btw` tests plus invalid
explicit default fail-closed. Add named-adapter and custom-command TUI2ACP cases
on both SMART and project-chat paths; `/exit` and `/btw` must preserve the exact
normalized options and `command_argv`. Add saved TTADK variant/model cases on
both real paths; they must not collapse to the global default provider or a
generic `"ttadk"` choice.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest \
  tests/test_handlers.py \
  tests/test_project_chat/test_default_intent.py \
  tests/test_intent.py -q
```

Expected: Traex and configured non-Coco cases expose current drift.

- [ ] **Step 3: Implement catalog-driven behavior**

```python
@dataclass(frozen=True)
class PersistedBackendChoice:
    tool_name: str
    model_name: str | None = None
    selection_options: tuple[tuple[str, str], ...] = ()
    command_argv: tuple[str, ...] = ()


def resolve_default_programming_backend(
    *,
    project_choice: PersistedBackendChoice | None,
    configured_choice: PersistedBackendChoice | None,
    registry: BackendCatalog,
) -> BackendSelection:
    requested = (
        project_choice
        or configured_choice
        or PersistedBackendChoice(tool_name="coco")
    )
    return registry.resolve(
        requested.tool_name,
        use=BackendUse.PROGRAMMING,
        model_name=requested.model_name,
        selection_options=dict(requested.selection_options),
        expected_command_argv=requested.command_argv,
    )
```

Programming-mode exit and `/btw` derive valid modes from
`BackendUse.PROGRAMMING`; no hard-coded enum subset remains. Project saved tool
precedes configured default, which precedes Coco. Invalid explicit tools fail
closed instead of silently falling back. The derived mode matrix includes
TUI2ACP and preserves its selected adapter/custom command; it must not flatten
that compound selection to a generic provider.

Remove `ProgrammingHandler._PROGRAMMING_MODE_KEYS` and its separate ACP-tool
set. Exit, `/btw`, and `_set_mode_on_project()` consume the catalog-derived
typed selection/compatibility translator. Real handler tests mutate a test
catalog by adding/removing one programming provider and prove mode detection,
exit, `/btw`, and persisted compound choice follow the catalog with no second
list.

The current `ProjectContext` persists `acp_tool_name` and
`tui2acp_adapter_name` separately. During the compatibility release,
`project_backend_choice_from_legacy_context()` is the only translation:
`tui2acp_adapter_name="custom:<command>"` becomes a validated
`custom_command` option and normalized argv; a named value becomes an
`adapter` option. Missing, contradictory, or malformed TUI2ACP state fails
closed with a re-selection instruction. Both SMART and project-chat callers
must pass this typed choice—not only the string `"tui2acp"`—and parity tests
assert the exact named/custom selection options, command argv, and capability
fingerprint.

The same translator owns TTADK's separate legacy fields:
`ttadk_tool_name` becomes the required immutable `variant` selection option and
`ttadk_model_name` becomes `model_name`. Because the current TTADK entry path
need not set `acp_tool_name`, a valid TTADK variant takes precedence when
deriving the saved project choice; contradictory ACP+TTADK state, an unknown
variant, or a model unsupported by that variant fails closed with re-selection
guidance. Real SMART and project-chat tests seed a saved project containing
only `ttadk_tool_name`/`ttadk_model_name` and assert the exact
`BackendSelection`, rather than invoking the translator alone.

- [ ] **Step 4: Verify provider parity**

Run:

```bash
uv run python -m pytest \
  tests/test_handlers.py \
  tests/test_project_chat/test_default_intent.py \
  tests/test_intent.py \
  tests/test_acp_model_normalization.py -q
```

Expected: PASS for every `BackendUse.PROGRAMMING` provider on SMART, project
chat, `/exit`, and `/btw`.

- [ ] **Step 5: Commit**

```bash
git add src/agent_session/backend_catalog.py src/feishu/handlers/system.py src/feishu/handlers/programming.py src/feishu/ws_client.py src/agent/intent_recognizer.py src/project_chat/service.py tests/test_handlers.py tests/test_project_chat/test_default_intent.py tests/test_intent.py
git commit -m "fix(provider): unify programming backend defaults"
```

### Task 16: Keep Spec model retry inside its selected provider

> **Decision gate:** Requires DG-2 approval because this changes Spec
> model-selection and retry behavior.

**Files:**
- Modify: `src/spec_engine/session_utils.py`
- Modify: `src/spec_engine/engine.py`
- Modify: `src/spec_engine/models.py`
- Modify: `tests/test_spec_engine.py`
- Modify: `tests/test_spec_engine_retry_integration.py`
- Modify: `tests/test_acp_model_normalization.py`

**Interfaces:**
- Produces: `ProviderModelContext`, `initialize_model_context()`,
  `choose_next_model()`.
- Consumes: `BackendCatalog` and `BackendSelection` from Task 14.

- [ ] **Step 1: Write failing provider-local retry tests**

```python
@pytest.mark.parametrize("tool", ["codex", "gemini", "aiden", "traex"])
def test_spec_retry_never_uses_coco_model_manager(
    tool,
    monkeypatch,
    catalog,
) -> None:
    coco = MagicMock()
    monkeypatch.setattr(
        session_utils,
        "get_coco_model_manager",
        coco,
    )
    selection = catalog.resolve(tool, use=BackendUse.ENGINE)
    context = initialize_model_context(
        selection,
        cwd="/repo",
        registry=catalog,
    )
    assert all(item.backend_id == selection.backend_id for item in context.candidates)
    coco.assert_not_called()
```

Add restart coverage proving backend ID, normalized model,
`selection_options`, and capability fingerprint round-trip without Coco
fallback.

- [ ] **Step 2: Run and verify RED**

```bash
uv run python -m pytest \
  tests/test_spec_engine.py \
  tests/test_spec_engine_retry_integration.py \
  tests/test_acp_model_normalization.py -q
```

Expected: FAIL because non-Claude/TTADK retries can consult the Coco model
manager.

- [ ] **Step 3: Implement provider-local model context**

```python
@dataclass(frozen=True)
class ProviderModelContext:
    selection: BackendSelection
    candidates: tuple[BackendSelection, ...]
    current_model: str | None


def initialize_model_context(
    selection: BackendSelection,
    *,
    cwd: str,
    registry: BackendCatalog,
) -> ProviderModelContext:
    """List and normalize models only through selection.backend_id."""
```

Retry may replace `model_name` inside the same backend selection. It must not
change `backend_id`, transport, or selection options. If no trustworthy model
list exists, retain the provider default or fail explicitly; never switch
provider. A recovered fingerprint mismatch maps to `NEEDS_ATTENTION`.

- [ ] **Step 4: Verify the provider matrix**

Run the RED command again. Expected: PASS for Coco, Claude, Aiden, Codex,
Gemini, Traex, and TTADK with zero cross-provider candidate.

- [ ] **Step 5: Commit**

```bash
git add src/spec_engine/session_utils.py src/spec_engine/engine.py src/spec_engine/models.py tests/test_spec_engine.py tests/test_spec_engine_retry_integration.py tests/test_acp_model_normalization.py
git commit -m "fix(spec): keep model retry provider-local"
```

### Task 17: Define a shared run lifecycle and checkpoint contract

**Files:**
- Create: `src/tasking/run_contract.py`
- Create: `src/tasking/checkpoint_store.py`
- Create: `tests/test_run_contract.py`
- Create: `tests/test_engine_lifecycle_mapping.py`
- Modify: `src/engine_base.py`
- Modify: `src/tasking/control_plane.py`
- Modify: `src/tasking/adapters/scheduler.py`
- Modify: `src/tasking/adapters/engines.py`
- Modify: `src/tasking/adapters/slock.py`
- Modify: `src/tasking/adapters/department.py`
- Modify: `src/tasking/adapters/employee.py`
- Modify: `src/config/settings.py`
- Modify: `.env.example`
- Modify: `tests/test_config_validation.py`

**Interfaces:**
- Produces: `RunStatus`, `TerminalReason`, `RunCheckpoint`, `TerminalOutcome`, `CheckpointStore`, `RunLifecycle`.
- Consumed by: engine adapters immediately; Tasks 18–20 after DG-2.

- [ ] **Step 1: Write failing lifecycle invariants**

```python
def test_terminal_state_is_sticky(lifecycle, running_checkpoint) -> None:
    completed = lifecycle.finalize(
        running_checkpoint,
        TerminalOutcome(
            status=RunStatus.SUCCEEDED,
            reason=TerminalReason.COMPLETED,
            summary="done",
        ),
    )
    with pytest.raises(InvalidRunTransition):
        lifecycle.finalize(
            completed,
            TerminalOutcome(
                status=RunStatus.FAILED,
                reason=TerminalReason.LATE_ERROR,
                summary="late",
            ),
        )


def test_cancel_is_durable_before_external_cancel(
    lifecycle,
    store,
    session,
) -> None:
    lifecycle.cancel(running_checkpoint, cancel=session.cancel)
    assert store.events[:2] == ["save:cancelling", "external:cancel"]


def test_checkpoint_identity_includes_run_id(store, first_run, second_run) -> None:
    store.save(first_run, expected_generation=None)
    store.save(second_run, expected_generation=None)
    assert store.load(first_run.engine_kind, first_run.owner_key, first_run.run_id) == first_run
    assert store.load(second_run.engine_kind, second_run.owner_key, second_run.run_id) == second_run


@pytest.mark.parametrize(
    "owner_key",
    [
        "oc_1:/repo",
        "../../outside",
        "/absolute/root/with spaces",
        "tenant:项目/../repo",
    ],
)
def test_checkpoint_identity_is_confined_and_authenticated(
    store,
    running_checkpoint,
    owner_key,
) -> None:
    checkpoint = replace(running_checkpoint, owner_key=owner_key)
    store.save(checkpoint, expected_generation=None)
    assert store.load(
        checkpoint.engine_kind,
        owner_key,
        checkpoint.run_id,
    ) == checkpoint
    assert all(path.is_relative_to(store.root) for path in store.created_paths)


@pytest.mark.parametrize("writer_mode", ["threads", "processes"])
def test_checkpoint_cas_allows_exactly_one_same_generation_writer(
    checkpoint_store_path,
    running_checkpoint,
    writer_mode,
) -> None:
    outcomes = race_checkpoint_writers(
        checkpoint_store_path,
        current=running_checkpoint,
        expected_generation=running_checkpoint.generation,
        writer_mode=writer_mode,
    )
    assert sorted(outcomes) == ["saved", "stale_generation"]


def test_checkpoint_payload_and_terminal_outcome_are_deeply_immutable(
    lifecycle,
    store,
    running_checkpoint,
) -> None:
    source = {"steps": [{"id": "s1", "done": False}]}
    saved = lifecycle.checkpoint(
        running_checkpoint,
        payload=source,
        resume_token="resume-1",
    )
    source["steps"][0]["done"] = True
    assert thaw_checkpoint_payload(saved.payload)["steps"][0]["done"] is False

    terminal = lifecycle.finalize(
        saved,
        TerminalOutcome(
            status=RunStatus.SUCCEEDED,
            reason=TerminalReason.COMPLETED,
            summary="done",
        ),
    )
    reloaded = store.load(
        terminal.engine_kind,
        terminal.owner_key,
        terminal.run_id,
    )
    assert reloaded is not None
    assert reloaded.terminal_outcome == terminal.terminal_outcome
```

Add stale generation, corrupt checkpoint, interrupted-running recovery,
terminal-outcome serialization, and two historical runs sharing one owner
tests.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest tests/test_run_contract.py tests/test_engine_lifecycle_mapping.py -q
```

Expected: FAIL because no cross-engine lifecycle contract exists.

- [ ] **Step 3: Implement immutable lifecycle primitives**

```python
class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    RECOVERING = "recovering"
    NEEDS_ATTENTION = "needs_attention"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TerminalReason(str, Enum):
    COMPLETED = "completed"
    USER_CANCELLED = "user_cancelled"
    TIMED_OUT = "timed_out"
    BACKEND_ERROR = "backend_error"
    VALIDATION_FAILED = "validation_failed"
    INTERRUPTED = "interrupted"
    LATE_ERROR = "late_error"


@dataclass(frozen=True)
class TerminalOutcome:
    status: RunStatus
    reason: TerminalReason
    summary: str
    error_code: str | None = None


@dataclass(frozen=True)
class RunCheckpoint:
    schema_version: int
    run_id: str
    engine_kind: str
    owner_key: str
    status: RunStatus
    attempt: int
    generation: int
    backend: BackendSelection | None
    payload: Mapping[str, object]
    resume_token: str | None
    updated_at: float
    terminal_outcome: TerminalOutcome | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "payload",
            freeze_checkpoint_payload(self.payload),
        )


@dataclass(frozen=True)
class RecoveryEvidence:
    owner_matches: bool
    backend_fingerprint_matches: bool
    resume_token_valid: bool
    unresolved_external_effects: bool


class CheckpointStore(Protocol):
    def load(
        self,
        engine_kind: str,
        owner_key: str,
        run_id: str,
    ) -> RunCheckpoint | None:
        """Load one exact run without repairing corruption."""

    def list_runs(
        self,
        engine_kind: str,
        owner_key: str,
        *,
        statuses: frozenset[RunStatus] = frozenset(),
        limit: int = 100,
    ) -> tuple[RunCheckpoint, ...]:
        """List bounded runs for explicit recovery discovery."""

    def save(
        self,
        checkpoint: RunCheckpoint,
        *,
        expected_generation: int | None,
    ) -> None:
        """CAS-save through temp, fsync, replace and parent-directory fsync."""


class RunLifecycle:
    def begin(
        self,
        *,
        run_id: str,
        engine_kind: str,
        owner_key: str,
        backend: BackendSelection | None,
        payload: Mapping[str, object],
    ) -> RunCheckpoint:
        """Create and durably save generation 1 in RUNNING."""

    def checkpoint(
        self,
        current: RunCheckpoint,
        *,
        payload: Mapping[str, object],
        resume_token: str | None,
    ) -> RunCheckpoint:
        """CAS-save the next generation without changing terminal state."""

    def cancel(
        self,
        current: RunCheckpoint,
        *,
        cancel: Callable[[], None],
    ) -> RunCheckpoint:
        """Anchor CANCELLING before dispatching the external cancel."""

    def finalize(
        self,
        current: RunCheckpoint,
        outcome: TerminalOutcome,
    ) -> RunCheckpoint:
        """CAS-save one sticky terminal outcome."""

    def recover(
        self,
        current: RunCheckpoint,
        evidence: RecoveryEvidence,
    ) -> RunCheckpoint:
        """Map interrupted state to a safe next generation."""
```

The file store key and on-disk path include `(engine_kind, owner_key, run_id)`;
two historical or concurrently observed runs for one owner can never overwrite
each other. “Include” means a safe identity codec, never raw path segments:
canonical UTF-8 length-prefixed `(engine_kind, owner_key, run_id)` bytes map to
a SHA-256 leaf under the configured checkpoint root. The serialized record
contains all three original identity fields plus the identity digest; load
recomputes and authenticates them. A digest leaf containing a different
identity is a collision/corruption error and is quarantined, never overwritten.
All directories are mode 0700, files/locks are 0600 regular files opened
no-follow, resolved paths must remain beneath the canonical store root, and
symlink/hard-link/traversal/special-file targets fail closed. Add permission,
`../`/absolute owner-key, symlink-swap, and forced-digest-collision regressions.

`begin()` rejects an already-existing exact run ID, while `list_runs()` is the
only bounded discovery API for restart recovery. The store writes temp →
flush/fsync → replace → parent fsync. Every exact-run write first
acquires a keyed in-process mutex and then an adjacent per-run cross-process
`flock`; inside that lock it re-reads and validates the persisted generation
before creating/replacing the temp file. Lock ordering is fixed
`in_process_mutex → flock`, locks cover terminal validation and replacement,
and release happens in `finally`. A thread and spawned-process Barrier race with
the same `expected_generation` must produce exactly one save and one
`StaleRunGeneration`; last-writer-wins is forbidden.

`freeze_checkpoint_payload()` recursively defensive-copies JSON-compatible
maps/lists into read-only maps/tuples, rejects unsupported mutable objects,
non-string map keys, NaN/Infinity, and cycles, and returns canonical data for
serialization. Neither caller mutation after `begin()`/`checkpoint()` nor a
nested mutation attempt can change saved evidence. Reload reconstructs the
same frozen shape.

Terminal is sticky and persists the complete `terminal_outcome`;
`TerminalOutcome.status` accepts only `SUCCEEDED`, `FAILED`, or `CANCELLED`.
Serialization round-trips `status`, `reason`, `summary`, and `error_code`
without relying on the opaque payload. `RunCheckpoint` has no second terminal
error field: terminal states require a non-null outcome whose status equals the
checkpoint status, and nonterminal states require `terminal_outcome=None`.
Contradictory/missing terminal outcomes fail construction and deserialization.
`cancel()` saves `CANCELLING`, then calls
the external cancellation port; only explicit native cancellation
acknowledgement may finalize `CANCELLED`. Dispatch failure or missing
acknowledgement becomes `NEEDS_ATTENTION`, never a false terminal. Resume/retry
increments generation; CAS mismatch raises `StaleRunGeneration`. Interrupted
running defaults to `NEEDS_ATTENTION` unless owner, backend fingerprint, resume
token, and external effect evidence are all trustworthy. Corrupt or
schema-unknown checkpoints are quarantined and returned as a typed recovery
failure rather than overwritten.

- [ ] **Step 4: Add read-only native-state mappings**

Map every engine's current state to `RunStatus` in adapters without changing
engine writes. Unknown maps to `NEEDS_ATTENTION`, never success. Add
`run_lifecycle_mode: Literal["read_only", "dual_write", "enforced"] =
"read_only"`; only DG-2-approved engine tasks may dual-write or enforce.
Implement the mappings in the existing Task 13 adapter files listed above:
Scheduler/Programming in `scheduler.py`, Deep/Spec/Worktree/Workflow in
`engines.py`, and Slock/Department/Team/Employee in their respective adapters.
Track A requires these mappings and their tests but performs no native-engine
write through `RunLifecycle`.

Run:

```bash
uv run python -m pytest \
  tests/test_run_contract.py \
  tests/test_engine_lifecycle_mapping.py \
  tests/test_config_validation.py \
  tests/test_task_control_plane.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tasking/run_contract.py src/tasking/checkpoint_store.py src/engine_base.py src/tasking/control_plane.py src/tasking/adapters/scheduler.py src/tasking/adapters/engines.py src/tasking/adapters/slock.py src/tasking/adapters/department.py src/tasking/adapters/employee.py src/config/settings.py .env.example tests/test_run_contract.py tests/test_engine_lifecycle_mapping.py tests/test_config_validation.py
git commit -m "feat(tasking): define durable run lifecycle contract"
```

### Task 18: Correct Worktree timeout, cancellation, and terminal aggregation

> **Decision gate:** Requires DG-2 approval.

**Files:**
- Modify: `src/worktree_engine/models.py`
- Modify: `src/worktree_engine/dispatcher.py`
- Modify: `src/worktree_engine/manager.py`
- Modify: `src/worktree_engine/reporter.py`
- Modify: `tests/test_worktree_dispatcher_timeout.py`
- Modify: `tests/test_worktree_auto_execute.py`
- Modify: `tests/test_worktree_dispatcher.py`
- Modify: `tests/test_dispatcher_cancel_unit.py`

**Interfaces:**
- Produces: `WorktreeUnitExecutionResult`, `WorktreeDispatchResult`, `UnitExecutionControl`.
- Consumes: `RunLifecycle` and `SyncSession.cancel()`.

- [ ] **Step 1: Write failing bounded-cancel tests**

```python
def test_pool_timeout_cancels_active_session_and_returns_bounded(
    dispatcher,
    blocked_session,
    fake_clock,
) -> None:
    result = dispatcher.execute_units(
        [unit("u1")],
        pool_timeout=10,
        cancel_grace=2,
    )
    assert blocked_session.cancel_calls == 1
    assert result.timed_out is True
    assert result.status is RunStatus.FAILED
    assert fake_clock.elapsed <= 12


def test_failed_unit_prevents_success_terminal(manager) -> None:
    result = dispatch_result(
        completed=("u1",),
        failed=("u2",),
    )
    manager.finish_dispatch(result)
    assert manager.state.status is not RunStatus.SUCCEEDED
```

Use Event/Barrier/fake clock, never real sleeps.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest \
  tests/test_worktree_dispatcher_timeout.py \
  tests/test_worktree_auto_execute.py \
  tests/test_worktree_dispatcher.py -q
```

Expected: timeout waits for the executor context and/or manager reports success.

- [ ] **Step 3: Implement explicit executor and generation fencing**

```python
@dataclass(frozen=True)
class WorktreeUnitExecutionResult:
    unit_id: str
    generation: int
    status: WorktreeUnitStatus
    summary: str
    error: str
    stop_reason: str
    has_changes: bool


@dataclass(frozen=True)
class WorktreeDispatchResult:
    units: tuple[WorktreeUnitExecutionResult, ...]
    status: RunStatus
    timed_out: bool = False
    error_code: str | None = None
```

On timeout: persist `CANCELLING`, set the unit Event, call bound
`session.cancel()`, call `future.cancel()`, wait only `cancel_grace`, then
`shutdown(wait=False, cancel_futures=True)`. Late results with the old generation
are discarded. Only all-required-units succeeded produces green terminal and
merge-ready notes.

- [ ] **Step 4: Verify cancellation and terminal truth**

Run:

```bash
uv run python -m pytest \
  tests/test_worktree_dispatcher_timeout.py \
  tests/test_worktree_auto_execute.py \
  tests/test_worktree_dispatcher.py \
  tests/test_dispatcher_cancel_unit.py -q
```

Expected: PASS; failed/cancelled/unknown stop reasons never report success.

- [ ] **Step 5: Commit**

```bash
git add src/worktree_engine/models.py src/worktree_engine/dispatcher.py src/worktree_engine/manager.py src/worktree_engine/reporter.py tests/test_worktree_dispatcher_timeout.py tests/test_worktree_auto_execute.py tests/test_worktree_dispatcher.py tests/test_dispatcher_cancel_unit.py
git commit -m "fix(worktree): enforce bounded truthful completion"
```

### Task 19: Persist and recover Deep progress

> **Decision gate:** Requires DG-2 approval.

**Files:**
- Modify: `src/deep_engine/models.py`
- Modify: `src/deep_engine/engine.py`
- Modify: `src/feishu/handlers/deep.py`
- Modify: `tests/test_deep_engine.py`
- Create: `tests/test_deep_restart_recovery.py`

**Interfaces:**
- Produces: `DeepCheckpointPayload`, `DeepEngineManager.get_or_load()`.
- Consumes: `RunCheckpoint` and `CheckpointStore`.

- [ ] **Step 1: Write failing restart recovery tests**

```python
def test_deep_checkpoint_round_trip_restores_execution_context(
    checkpoint_store,
    deep_engine,
) -> None:
    deep_engine.checkpoint()
    restored = DeepEngine.from_checkpoint(
        checkpoint_store.load(
            "deep",
            deep_engine.owner_key,
            deep_engine.run_id,
        )
    )
    assert restored.backend_selection == deep_engine.backend_selection
    assert restored.progress == deep_engine.progress
    assert restored.unfinished_plan == deep_engine.unfinished_plan


def test_interrupted_deep_run_recovers_paused_not_running(
    deep_manager,
    interrupted_checkpoint,
) -> None:
    restored = deep_manager.get_or_load(
        chat_id="oc_1",
        root_path="/repo",
        expected_owner_key=interrupted_checkpoint.owner_key,
    )
    assert restored.run_status is RunStatus.NEEDS_ATTENTION
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest tests/test_deep_engine.py tests/test_deep_restart_recovery.py -q
```

Expected: current load restores only project state and loses progress.

- [ ] **Step 3: Implement complete Deep checkpoint payload**

```python
@dataclass(frozen=True)
class DeepCheckpointPayload:
    requirement: str
    progress: Mapping[str, object]
    unfinished_plan: tuple[Mapping[str, object], ...]
    pending_guidance: str | None
    artifacts: tuple[str, ...]
```

Checkpoint after planning, significant step, guidance, pause/cancel, and terminal.
Persist backend selection/fingerprint and opaque resume token. Restart never
auto-runs an interrupted task; explicit resume either uses a trusted token or
starts the same provider/model with persisted completed/unfinished context.

- [ ] **Step 4: Verify fail-closed recovery**

Run:

```bash
uv run python -m pytest \
  tests/test_deep_engine.py \
  tests/test_deep_restart_recovery.py \
  tests/test_engine_timeout.py -q
```

Expected: corrupt, terminal, owner-mismatched, or backend-fingerprint-mismatched
checkpoints require attention and are not overwritten.

- [ ] **Step 5: Commit**

```bash
git add src/deep_engine/models.py src/deep_engine/engine.py src/feishu/handlers/deep.py tests/test_deep_engine.py tests/test_deep_restart_recovery.py
git commit -m "feat(deep): recover durable execution progress"
```

### Task 20: Make Workflow Journal and snapshots truthful and recoverable

> **Decision gate:** Requires DG-2 approval.

**Files:**
- Modify: `src/workflow_engine/journal.py`
- Modify: `src/workflow_engine/models.py`
- Modify: `src/workflow_engine/engine.py`
- Modify: `src/workflow_engine/state_manager.py`
- Modify: `src/workflow_engine/history.py`
- Modify: `src/workflow_engine/manager.py`
- Modify: `src/workflow_engine/runtime/runtime.js`
- Modify: `src/feishu/handlers/workflow.py`
- Modify: `src/feishu/ws_client.py`
- Create: `tests/test_workflow_run_journal.py`
- Create: `tests/test_workflow_manager_recovery.py`
- Modify: `tests/test_workflow_state_consistency.py`
- Modify: `tests/test_workflow_reliability_regression.py`
- Modify: `tests/test_workflow_runtime_primitives.py`
- Modify: `tests/test_workflow_runtime_reliability.py`

**Interfaces:**
- Produces: explicitly run-local `WorkflowRunJournal`, node checkpoints,
  `WorkflowRecoveryPayload`, `WorkflowRecoveryReport`, and manager-level
  startup reconciliation.
- Deliberately does not produce: cross-run result caching.

- [ ] **Step 1: Write failing snapshot and restart tests**

```python
def test_snapshot_preserves_current_activity_without_aliasing(state_manager) -> None:
    label = state_manager.on_agent_started(
        "agent_1",
        "codex",
        "实现",
        "修复路由",
    )
    state_manager.update_agent_activity(label, "正在读取代码")
    snapshot = state_manager.snapshot()
    copied = next(
        agent
        for phase in snapshot.phases
        for agent in phase.agents
        if agent.label == label
    )
    assert copied.current_activity == "正在读取代码"
    copied.current_activity = "mutated"
    fresh = next(
        agent
        for phase in state_manager.snapshot().phases
        for agent in phase.agents
        if agent.label == label
    )
    assert fresh.current_activity == "正在读取代码"


def test_run_journal_never_claims_cross_run_reuse(tmp_path: Path) -> None:
    first = WorkflowRunJournal(tmp_path, run_id="run_1")
    second = WorkflowRunJournal(tmp_path, run_id="run_2")
    first.commit_node(node_result("n1"))
    assert second.get_node("n1") is None
```

Add a crash-after-node-commit test proving recovery skips the already committed
node within the same run. Add a manager-level restart test: construct manager A,
persist the generated script/graph/tool selections, commit a node and simulate
process interruption, construct manager B over the same journal root, reconcile
startup, and verify the run is discoverable as `NEEDS_ATTENTION` without
auto-execution. An explicit resume must reconstruct the same requirement,
script, discovered call graph/frontier, selected tool/model bindings, and
progress, then replay the committed prefix without rerunning its agents. Add a
dynamic `classify()` branch crash test: persist the classifier result and chosen
branch, crash after the first branch node, then prove resume selects the same
branch, does not call the classifier/finished agent again, and continues at the
saved frontier. Add script/recovery-payload digest corruption, missing payload,
and non-replayable-runtime tests that fail closed without execution.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run python -m pytest \
  tests/test_workflow_run_journal.py \
  tests/test_workflow_manager_recovery.py \
  tests/test_workflow_state_consistency.py \
  tests/test_workflow_reliability_regression.py \
  tests/test_workflow_runtime_primitives.py \
  tests/test_workflow_runtime_reliability.py -q
```

Expected: snapshot loses `current_activity`, and no production
reconstruction/resume path exists.

- [ ] **Step 3: Implement honest run-local semantics**

```python
@dataclass(frozen=True)
class WorkflowNodeCheckpoint:
    run_id: str
    node_id: str
    generation: int
    input_digest: str
    backend_fingerprint: str
    state: RunStatus
    output_ref: str | None


@dataclass(frozen=True)
class WorkflowReplayEvent:
    sequence: int
    logical_call_id: str
    primitive: str
    input_digest: str
    result_ref: str
    result_digest: str
    chosen_branch: str | None = None


@dataclass(frozen=True)
class WorkflowRecoveryPayload:
    schema_version: int
    run_id: str
    generation: int
    requirement: str
    script_source: str
    script_sha256: str
    discovered_graph: tuple[tuple[str, tuple[str, ...]], ...]
    frontier: tuple[str, ...]
    backend_selections: tuple[tuple[str, BackendSelection], ...]
    progress_snapshot: WorkflowStateSnapshot
    committed_nodes: tuple[WorkflowNodeCheckpoint, ...]
    replay_log: tuple[WorkflowReplayEvent, ...]
    replayable: bool
    non_replayable_reason: str | None = None


class WorkflowRunJournal:
    def commit_recovery_payload(
        self,
        payload: WorkflowRecoveryPayload,
        *,
        expected_generation: int | None,
    ) -> None:
        """Atomically CAS-save everything needed for explicit resume."""

    def commit_node(self, checkpoint: WorkflowNodeCheckpoint) -> None:
        """Atomically commit one generation-fenced node checkpoint."""

    def get_node(self, node_id: str) -> WorkflowNodeCheckpoint | None:
        """Read one node from this run only."""

    def recover(self) -> tuple[WorkflowNodeCheckpoint, ...]:
        """Return validated same-run checkpoints in dependency order."""

    def load_recovery_payload(self) -> WorkflowRecoveryPayload:
        """Load and cross-validate the exact resumable run definition."""


class WorkflowEngineManager:
    def reconcile_startup(
        self,
        projects: Iterable[ProjectContext],
    ) -> WorkflowRecoveryReport:
        """Discover manifests under canonical, currently bound project roots."""

    def get_or_load(
        self,
        *,
        chat_id: str,
        root_path: str,
        run_id: str | None = None,
    ) -> WorkflowEngine | None:
        """Load a validated same-owner run without starting it."""
```

Rename misleading cross-run comments and user documentation. Keep Journal under
`runs/<run_id>`; do not implement cross-run cache in this convergence program.
Commit successful node checkpoints atomically, recover only the same run, reject
stale generations, and preserve `current_activity` through a deep-copy mapper.

Before the first node dispatch, atomically commit a run-local recovery payload
containing the exact bounded script source and SHA-256, requirement, immutable
backend/tool/model selections and fingerprints, empty discovered graph/replay
log, initial frontier/progress, and generation. Dynamic Workflow primitives
cannot promise a complete graph up front: `classify`, `fanout`, `generate`,
`loop`, `race`, and nested calls append stable logical call IDs, discovered
edges, chosen branches, and the next frontier as runtime RPCs occur.

For each host primitive/external Agent call, anchor the logical call and input
digest before dispatch; after completion, durably commit its result reference,
digest, branch decision, graph update, and new frontier before returning the
value to JavaScript. On every successful node commit, pause, cancel, retry
boundary, or progress transition, write the node checkpoint/replay event and
the next recovery payload through one journal transaction (or a manifest
pointer switched only after all generation files are fsynced); a crash may
expose the old complete generation, never a new result with an old
reconstruction payload.

Resume starts the same validated script in replay mode: calls with the same
ordered logical call ID and input digest receive the saved result and branch
without external dispatch; execution may leave the replay prefix only at the
saved frontier. The runtime must wrap and journal every exposed
nondeterministic input used by workflow scripts, including logical time and
randomness, or remove it from the sandbox. If it detects an unwrapped
nondeterministic host input, reordered logical call, changed input digest, or
code path that cannot be replayed, persist `replayable=False`, surface
`NEEDS_ATTENTION`, and refuse explicit resume rather than guessing.
`WorkflowEngine.save_state()` in `finally` remains a convenience snapshot, not
the durability boundary.

Each run directory also contains an atomically committed, secret-free discovery
manifest with `run_id`, keyed chat-ID digest, canonical root-path digest,
owner-key digest, generation, backend fingerprint, recovery-payload digest, and
script digest so a new process can discover it without guessing a run ID or
persisting raw tenant/chat IDs in the manifest. The separately permissioned
recovery payload may contain the requirement/script needed to resume; it uses
the state-root file mode, size limit, canonical encoding, recursive immutable
decode, and digest checks. Unknown schema, missing payload, discovered-graph
cycles, replay-log gaps, selection/fingerprint drift, script digest mismatch,
or a node/output reference outside its run directory quarantines the run and
yields `NEEDS_ATTENTION`.

`WorkflowEngineManager.reconcile_startup()` receives the canonical projects
already loaded by `ProjectManager`, scans only
`<project.root_path>/.ghostap/workflow_journal/runs/`, validates
path ownership/schema/digests, quarantines corrupt entries, and registers
interrupted runs as `NEEDS_ATTENTION`; it never scans arbitrary filesystem
roots or executes a run. `FeishuWSClient` calls reconciliation only after
project state is available. The Workflow handler uses `get_or_load()` for
`/status` and explicit resume. `get_or_load()` reconstructs only from the
validated recovery payload and current catalog bindings; scope, owner, payload,
script, discovered graph/frontier, replay log, output-ref, or fingerprint
mismatch stays fail closed. Explicit resume creates a new fenced generation,
replays only the verified same-run prefix, and skips only nodes whose logical
call ID, input digest, result digest, and backend fingerprint still match.

- [ ] **Step 4: Verify recovery and copy isolation**

Run:

```bash
uv run python -m pytest \
  tests/test_workflow_run_journal.py \
  tests/test_workflow_manager_recovery.py \
  tests/test_workflow_state_consistency.py \
  tests/test_workflow_reliability_regression.py \
  tests/test_workflow_runtime_primitives.py \
  tests/test_workflow_runtime_reliability.py \
  tests/test_workflow_executor_cancel.py -q
```

Expected: PASS; failed/cancelled/uncommitted nodes are never reused.

- [ ] **Step 5: Commit**

```bash
git add src/workflow_engine/journal.py src/workflow_engine/models.py src/workflow_engine/engine.py src/workflow_engine/state_manager.py src/workflow_engine/history.py src/workflow_engine/manager.py src/workflow_engine/runtime/runtime.js src/feishu/handlers/workflow.py src/feishu/ws_client.py tests/test_workflow_run_journal.py tests/test_workflow_manager_recovery.py tests/test_workflow_state_consistency.py tests/test_workflow_reliability_regression.py tests/test_workflow_runtime_primitives.py tests/test_workflow_runtime_reliability.py
git commit -m "fix(workflow): make run journal recoverable and truthful"
```

---

## Phase 3 — Operability and Cutover

### Task 21: Quarantine the retired standalone Autonomous runtime

**Files:**
- Create: `src/autonomous/legacy/__init__.py`
- Create: `src/autonomous/legacy/README.md`
- Create: `src/autonomous/presentation/__init__.py`
- Create: `src/autonomous/presentation/cards.py`
- Create: `tests/autonomous/contract/test_production_import_boundary.py`
- Modify: `src/autonomous/__init__.py`
- Modify: `src/autonomous/bootstrap.py`
- Modify: `src/autonomous/manager/__init__.py`
- Modify: `src/autonomous/manager/cards.py`
- Modify: `src/autonomous/manager/handler.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Modify: `src/feishu/handlers/system.py`
- Modify: `src/feishu/handlers/slock.py`
- Modify: `AGENTS.md`
- Modify: `tests/autonomous/contract/test_manager_command_surface.py`
- Modify: `tests/autonomous/unit/test_employee_runtime_cards.py`
- Modify: `tests/autonomous/integration/test_feishu_employee_creation.py`
- Modify: `tests/test_slock_status_card.py`

**Interfaces:**
- Produces: an explicit import boundary between production Employee Department and retired standalone Manager code.
- Produces: neutral `src.autonomous.presentation.cards` views/renderers for the
  production Employee Department.
- Preserves: `legacy_one_shot` and `legacy_pipeline` employee rollback modes; these are not the retired Manager.

- [ ] **Step 1: Write the failing production-reachability gate**

```python
def test_production_entrypoints_do_not_import_standalone_manager() -> None:
    violations = scan_import_graph(
        roots=(
            ROOT / "src" / "main.py",
            ROOT / "src" / "feishu" / "ws_client.py",
            ROOT / "src" / "feishu" / "handlers" / "slock.py",
            ROOT / "src" / "autonomous" / "provisioning",
            ROOT / "src" / "autonomous" / "gateway",
            ROOT / "src" / "autonomous" / "team",
            ROOT / "src" / "autonomous" / "workforce",
            ROOT / "src" / "autonomous" / "ingress",
            ROOT / "src" / "autonomous" / "supervisor",
        ),
        forbidden_prefixes=(
            "src.autonomous.manager",
            "src.autonomous.coordinator",
        ),
    )
    assert violations == []


def test_retired_manager_commands_never_accept_work(system_handler) -> None:
    for command in ("/goals", "/run", "/runs", "/decisions", "/approvals"):
        system_handler.reply_text.reset_mock()
        system_handler.handle_intercepted_command(
            "om_1",
            "oc_1",
            command,
            None,
            command_match=SlashCommandParser.parse(command),
        )
        reply = system_handler.reply_text.call_args.args[1]
        assert "已退役" in reply
        assert "未执行" in reply
```

- [ ] **Step 2: Run and verify the real current boundary**

Run:

```bash
uv run python -m pytest \
  tests/autonomous/contract/test_production_import_boundary.py \
  tests/autonomous/contract/test_manager_command_surface.py -q
```

Expected: initial failure identifies exports/tests that still imply a production surface.

- [ ] **Step 3: Quarantine without reviving mutable paths**

`src/autonomous/legacy/README.md` must state:

```markdown
This package is not a production runtime. It exists for migration evidence only.
No Feishu ingress, employee gateway, team runtime, or production composition may
import it. Its commands always fail closed.
```

Move only modules proven unreachable by the import graph. Leave old import paths
as warning/fail-closed shims for one release. Do not move shared Journal,
provisioning, workforce, gateway, team, Actor, or rollback-mode code.

Before forbidding `src.autonomous.manager`, move the shared
`EmployeeRuntimeCardView` and employee card builders from
`src/autonomous/manager/cards.py` to
`src/autonomous/presentation/cards.py`. Update production consumers in
`provisioning/composition.py`, `handlers/slock.py`, and the employee creation
path in `bootstrap.py` to import the neutral module. Keep
`manager/cards.py` as a deprecation re-export only; the import-graph test permits
that shim to be imported by compatibility tests but no production root may
reach it. Card JSON/schema output must remain byte-for-byte equivalent where
dynamic timestamps are normalized. Update the stale architecture pointer in
`AGENTS.md` in the same commit: `provisioning/composition.py` is the production
Employee Department composition; `autonomous/bootstrap.py` is the retired
standalone compatibility shell.

- [ ] **Step 4: Verify production Employee Department reachability**

Run:

```bash
uv run python -m pytest \
  tests/autonomous/contract/test_production_import_boundary.py \
  tests/autonomous/acceptance/test_persistent_employee_department.py \
  tests/autonomous/integration/test_employee_hire_composition.py \
  tests/autonomous/unit/test_employee_runtime_cards.py \
  tests/autonomous/integration/test_feishu_employee_creation.py \
  tests/test_slock_status_card.py \
  tests/test_handlers.py -q
```

Expected: PASS; only Journal-backed Slock/Employee paths accept work.

- [ ] **Step 5: Commit**

```bash
git add src/autonomous/legacy src/autonomous/presentation src/autonomous/__init__.py src/autonomous/bootstrap.py src/autonomous/manager src/autonomous/provisioning/composition.py src/feishu/handlers/system.py src/feishu/handlers/slock.py AGENTS.md tests/autonomous/contract/test_production_import_boundary.py tests/autonomous/contract/test_manager_command_surface.py tests/autonomous/unit/test_employee_runtime_cards.py tests/autonomous/integration/test_feishu_employee_creation.py tests/test_slock_status_card.py
git commit -m "refactor(autonomous): quarantine retired manager runtime"
```

### Task 22: Add a structured readiness doctor

**Files:**
- Create: `src/operations/__init__.py`
- Create: `src/operations/doctor.py`
- Create: `src/operations/metrics.py`
- Create: `src/operations/evidence.py`
- Create: `src/operations/identity.py`
- Create: `scripts/ghostap_doctor.py`
- Create: `scripts/collect_shadow_evidence.py`
- Create: `tests/test_operations_doctor.py`
- Create: `tests/test_runtime_metrics.py`
- Create: `tests/test_shadow_gate_evidence.py`
- Modify: `src/main.py`
- Modify: `src/config/settings.py`
- Modify: `.env.example`
- Modify: `src/access_control.py`
- Modify: `src/feishu/dispatcher.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/feishu/main_slash_commands.py`
- Modify: `src/feishu/product_catalog.py`
- Modify: `src/agent_session/backend_catalog.py`
- Modify: `src/agent_session/factory.py`
- Modify: `src/tasking/run_contract.py`
- Modify: `src/tasking/control_plane.py`
- Modify: `tests/test_access_control.py`
- Modify: `tests/test_route_decision.py`
- Modify: `tests/test_main_slash_commands.py`
- Modify: `tests/test_product_catalog.py`
- Modify: `tests/test_backend_catalog.py`
- Modify: `tests/test_run_contract.py`
- Modify: `tests/test_task_control_plane.py`
- Modify: `tests/test_config_validation.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `DoctorCheck`, `DoctorReport`, `RuntimeHealthEventCode`,
  `RuntimeHealthRecorder`, `RuntimeHealthSnapshot`, `ShadowGateCode`,
  `ShadowGateReport`, `SignedShadowGateReport`, `ShadowEvidenceBundle`,
  `ShadowEvidenceStore`, `ReleaseCandidateIdentity`, stable machine-readable
  readiness and evidence codes.

- [ ] **Step 1: Write the failing doctor contract**

```python
def test_doctor_reports_security_storage_and_runtime_profiles(tmp_path) -> None:
    report = run_doctor(settings_for(tmp_path))
    assert {check.code for check in report.checks} >= {
        "security_posture",
        "employee_runtime",
        "journal_anchor",
        "blob_integrity",
        "backend_catalog",
        "checkpoint_recovery",
        "feishu_transport",
    }
    assert report.ready is False


@pytest.mark.parametrize(
    ("overrides", "blocking_code"),
    [
        (
            {
                "admin_bootstrap_scope": "any_chat",
            },
            "admin_bootstrap_any_chat",
        ),
        (
            {"ingress_access_mode": "shadow"},
            "ingress_not_enforced",
        ),
        (
            {"ingress_access_mode": "legacy_allow_all"},
            "ingress_not_enforced",
        ),
    ],
)
def test_production_doctor_fails_unsafe_ingress_profile(
    tmp_path,
    overrides,
    blocking_code,
) -> None:
    report = run_doctor(
        production_settings_for(tmp_path, **overrides),
        profile="production",
    )
    check = next(item for item in report.checks if item.code == blocking_code)
    assert check.status == "fail"
    assert check.required is True
    assert report.ready is False


def test_release_identity_rejects_self_asserted_wrong_candidate() -> None:
    with pytest.raises(ReleaseIdentityMismatch):
        ReleaseCandidateIdentity(
            candidate_sha="a" * 40,
            build_attestation_sha="b" * 40,
            release_version="department-v1",
            service_instance_id="ghostap-beta-1",
            shadow_tenant_hash="1" * 64,
            employee_staging_tenant_hash="2" * 64,
            employee_production_tenant_hash="3" * 64,
        )


def test_runtime_snapshot_has_truthful_degraded_counters(runtime_probe) -> None:
    runtime_probe.record(RuntimeHealthEventCode.INGRESS_EVALUATED)
    runtime_probe.record(RuntimeHealthEventCode.SLASH_RECONCILED)
    runtime_probe.record(RuntimeHealthEventCode.SLASH_SURFACE_MISMATCH)
    runtime_probe.record(RuntimeHealthEventCode.ROUTE_COMPARED)
    runtime_probe.record(RuntimeHealthEventCode.ROUTE_SHADOW_DIVERGENCE)
    runtime_probe.record(RuntimeHealthEventCode.BACKEND_COMPARED)
    runtime_probe.record(RuntimeHealthEventCode.BACKEND_SELECTION_DIVERGENCE)
    runtime_probe.record(RuntimeHealthEventCode.TERMINAL_CONFLICT)
    runtime_probe.record(RuntimeHealthEventCode.STALE_GENERATION_DROP)
    snapshot = collect_runtime_health(runtime_probe)
    assert snapshot.ingress_evaluation_count == 1
    assert snapshot.slash_reconciliation_count == 1
    assert snapshot.route_comparison_count == 1
    assert snapshot.backend_comparison_count == 1
    assert snapshot.slash_surface_mismatch_count == 1
    assert snapshot.route_divergence_count == 1
    assert snapshot.backend_divergence_count == 1
    assert snapshot.terminal_conflict_count == 1
    assert snapshot.stale_generation_drop_count == 1
    assert "prompt" not in json.dumps(asdict(snapshot)).lower()


@pytest.mark.parametrize("gate", list(ShadowGateCode))
def test_shadow_report_is_sha_bound_signed_and_fail_closed(
    gate,
    independent_signer,
    public_keyring,
) -> None:
    report = passing_shadow_report(
        gate=gate,
        candidate_sha="a" * 40,
        window_seconds=48 * 3600,
    )
    signed = independent_signer.sign(report)
    assert verify_shadow_report(
        signed,
        keyring=public_keyring,
        expected_candidate_sha="a" * 40,
    ).gate is gate
    with pytest.raises(ShadowEvidenceError):
        verify_shadow_report(
            signed,
            keyring=public_keyring,
            expected_candidate_sha="b" * 40,
        )


def test_verified_bundle_freezes_the_exact_snapshot(
    evidence_store,
    independent_signer,
    public_keyring,
) -> None:
    snapshot = passing_runtime_snapshot(candidate_sha="a" * 40)
    reports = tuple(
        independent_signer.sign(
            passing_shadow_report(
                gate=gate,
                candidate_sha="a" * 40,
                runtime_snapshot_sha256=sha256_canonical(snapshot),
                sample_count=sample_count_for_gate(snapshot, gate),
            )
        )
        for gate in ShadowGateCode
    )
    bundle = build_shadow_evidence_bundle(snapshot, reports)
    evidence_store.write_fixture(bundle)
    assert evidence_store.load_verified_bundle(
        candidate_sha="a" * 40,
        keyring=public_keyring,
    ) == bundle

    tampered = replace(
        bundle,
        snapshot=replace(snapshot, route_comparison_count=999),
    )
    with pytest.raises(ShadowEvidenceError):
        verify_shadow_evidence_bundle(tampered, keyring=public_keyring)
```

- [ ] **Step 2: Run and verify RED**

```bash
uv run python -m pytest \
  tests/test_operations_doctor.py \
  tests/test_runtime_metrics.py \
  tests/test_shadow_gate_evidence.py \
  tests/test_config_validation.py -q
```

- [ ] **Step 3: Implement the structured report**

```python
@dataclass(frozen=True)
class DoctorCheck:
    code: str
    status: Literal["pass", "warn", "fail", "unknown", "not_applicable"]
    required: bool
    summary: str
    remediation: str


@dataclass(frozen=True)
class DoctorReport:
    profile: str
    checks: tuple[DoctorCheck, ...]

    @property
    def ready(self) -> bool:
        return all(
            not item.required or item.status not in {"fail", "unknown"}
            for item in self.checks
        )


@dataclass(frozen=True)
class ReleaseCandidateIdentity:
    candidate_sha: str
    build_attestation_sha: str
    release_version: str
    service_instance_id: str
    shadow_tenant_hash: str
    employee_staging_tenant_hash: str
    employee_production_tenant_hash: str

    def __post_init__(self) -> None:
        if self.candidate_sha != self.build_attestation_sha:
            raise ReleaseIdentityMismatch("configured SHA is not deployed SHA")


@dataclass(frozen=True)
class RuntimeHealthSnapshot:
    candidate_sha: str
    window_started_at: str
    window_ended_at: str
    backend_catalog_fingerprint: str
    degraded_backends: tuple[str, ...]
    active_runs_by_kind: tuple[tuple[str, int], ...]
    recovery_needed_by_kind: tuple[tuple[str, int], ...]
    stale_checkpoint_count: int
    ingress_evaluation_count: int
    ingress_would_deny_count: int
    slash_reconciliation_count: int
    slash_surface_mismatch_count: int
    slash_unresolved_compatibility_count: int
    route_comparison_count: int
    route_divergence_count: int
    backend_comparison_count: int
    backend_divergence_count: int
    terminal_conflict_count: int
    stale_generation_drop_count: int
    provider_drift_count: int
    task_source_degraded_count: int


class ShadowGateCode(str, Enum):
    INGRESS_ACCESS = "ingress_access"
    SLASH_SURFACE = "slash_surface"
    ROUTE_DECISION = "route_decision"
    BACKEND_CATALOG = "backend_catalog"


@dataclass(frozen=True)
class ShadowGateReport:
    schema_version: int
    gate: ShadowGateCode
    candidate_sha: str
    tenant_hash: str
    release_version: str
    service_instance: str
    window_started_at: str
    window_ended_at: str
    sample_count: int
    reviewed_expected_count: int
    unexplained_count: int
    contract_fingerprint: str
    runtime_snapshot_sha256: str


@dataclass(frozen=True)
class SignedShadowGateReport:
    report: ShadowGateReport
    key_id: str
    signature: str
    envelope_sha256: str


@dataclass(frozen=True)
class ShadowEvidenceBundle:
    schema_version: int
    snapshot: RuntimeHealthSnapshot
    reports: tuple[SignedShadowGateReport, ...]
    snapshot_sha256: str
    bundle_sha256: str


class ShadowEvidenceStore(Protocol):
    def load_verified_bundle(
        self,
        *,
        candidate_sha: str,
        keyring: Mapping[str, bytes],
    ) -> ShadowEvidenceBundle:
        """Return one immutable verified snapshot plus all four reports."""
```

JSON output contains stable codes, never prompts, credentials, raw tenant IDs,
or message content. An unavailable required check is `unknown`, not `pass`.
The selected release profile determines required checks: DG-2-only recovery is
`not_applicable` for Track A and required for Track B; this distinction is
explicit in JSON rather than silently downgraded to `warn`.

Profile mapping is stricter than raw posture severity. Candidate doctor may
render `admin_bootstrap_any_chat` as a warning to support migration analysis,
but production doctor maps it to
`DoctorCheck(status="fail", required=True)` until DG-1A authorizes and configures
P2P-only bootstrap. Production likewise fails with `ingress_not_enforced`
unless the effective `ingress_access_mode` is exactly `enforced`;
`legacy_allow_all` and `shadow` can collect evidence but can never yield
`ready=True`. These checks
run from effective validated settings, not documentation or environment text.

Add validated configuration for
`release_candidate_sha`, `release_version`,
`ghostap_service_instance_id`, `release_tenant_hash`,
`ghostap_build_commit_sha`,
`autonomous_employee_staging_tenant_hash`, and
`autonomous_employee_production_tenant_hash`,
`shadow_evidence_bundle`, `shadow_evidence_public_keyring`, and
`shadow_evidence_max_age_seconds`. Candidate profile may omit external evidence
paths; production profile requires all expected identity values and readable,
non-symlink bundle/keyring files. Public verification keys may be configured;
private signing material is never a GhostAP setting.

`src/operations/identity.py` constructs one `ReleaseCandidateIdentity` from
effective settings plus an independently supplied deployment/build attestation.
The attestation is injected by CI/deployment metadata (or derived from the
read-only deployed Git checkout), never copied from
`release_candidate_sha`. Startup, the recorder, evidence collector, production
doctor, and Task 25 Employee binding all consume this object. A missing
attestation, candidate/build SHA mismatch, release/service mismatch, malformed
tenant hash, or equal Employee staging/production tenant hashes blocks startup
in evidence/production profiles; no component may self-assert identity from the
value it is meant to verify.

`src/operations/evidence.py` recursively canonicalizes the immutable snapshot
and report JSON, computes their digests, and verifies each report's independent
Ed25519 signature against a configured public-key keyring. It rejects an
unknown key, duplicate/missing gate, non-40/64-hex SHA,
tenant/SHA/release/service mismatch, window shorter than 48 hours,
non-overlapping windows, future or expired window, zero coverage,
negative/inconsistent counts, nonzero `unexplained_count`, sample totals that
do not equal the matching snapshot denominator, bad snapshot/bundle digest, and
signature/envelope mutation. The GhostAP service never receives the private
signing key.

`scripts/collect_shadow_evidence.py` atomically freezes the bounded
`RuntimeHealthSnapshot` into one canonical `ShadowEvidenceBundle`, derives each
gate's `sample_count` from its matching total counter, and emits the four
canonical unsigned reports for independent QA signing. The signer inserts the
four signed envelopes without changing the snapshot bytes. The approved bundle
is stored outside the repository. `ShadowEvidenceStore` is read-only in
production and verifies report → snapshot digest, bundle digest, identity,
window, and denominator consistency before returning the whole bundle through
dependency injection. Production readiness never recomputes report digests
from the later, still-changing live counter file.

`RuntimeHealthRecorder` is one injected, thread-safe process collaborator backed
by a bounded atomically replaced counter snapshot under the configured state
root. It accepts only an enum code, optional engine/backend kind from a fixed
catalog, candidate SHA, and integer delta; arbitrary text, IDs, prompts, paths,
and exception bodies are rejected. `FeishuWSClient` creates it once and injects
it into the existing sources:

- Task 3 access shadow records `INGRESS_WOULD_DENY`;
- Task 9 Slash reconciliation records unresolved compatibility tokens and
  surface mismatches;
- Task 12 legacy/new comparison records `ROUTE_SHADOW_DIVERGENCE`;
- Task 14 catalog/legacy comparison records
  `BACKEND_SELECTION_DIVERGENCE` and provider drift;
- Task 17 `RunLifecycle` records rejected terminal overwrites and stale
  generation drops;
- Task 13 control-plane adapter failures record `TASK_SOURCE_DEGRADED`.

Every evaluation also records its denominator event, including successful
matches: Task 3 records `INGRESS_EVALUATED`, Task 9 records
`SLASH_RECONCILED`, Task 12 records `ROUTE_COMPARED`, and Task 14 records
`BACKEND_COMPARED`. The four corresponding snapshot totals are monotonic within
the candidate window. Divergence counters alone are never used as
`sample_count`; this makes a zero-divergence observation distinguishable from
zero traffic.

Unit tests must drive each real producer, not just call the recorder directly,
and then assert one denominator increment, the exact zero/nonzero anomaly
increment, and restart round-trip. The doctor validates the snapshot schema,
candidate SHA, observation window, and catalog fingerprint.
A missing/corrupt snapshot or a snapshot bound to another SHA is `unknown` and
blocks the associated release gate; a newly created all-zero snapshot is not a
substitute for the 48-hour shadow evidence. Candidate-profile doctor marks the
four external reports `not_applicable`; production-profile doctor requires one
verified immutable `ShadowEvidenceBundle` containing exactly one report for
every `ShadowGateCode`, and checks that all four share the bundled snapshot,
expected candidate SHA, tenant hash, release version, service instance, and
overlapping observation window.

- [ ] **Step 4: Verify**

```bash
uv run python -m pytest \
  tests/test_operations_doctor.py \
  tests/test_runtime_metrics.py \
  tests/test_shadow_gate_evidence.py \
  tests/test_main_slash_commands.py \
  tests/test_product_catalog.py -q
uv run python scripts/ghostap_doctor.py --profile candidate --json
```

- [ ] **Step 5: Commit**

```bash
git add src/operations/__init__.py src/operations/doctor.py src/operations/metrics.py src/operations/evidence.py src/operations/identity.py src/main.py src/config/settings.py .env.example src/access_control.py src/feishu/dispatcher.py src/feishu/ws_client.py src/feishu/main_slash_commands.py src/feishu/product_catalog.py src/agent_session/backend_catalog.py src/agent_session/factory.py src/tasking/run_contract.py src/tasking/control_plane.py scripts/ghostap_doctor.py scripts/collect_shadow_evidence.py tests/test_operations_doctor.py tests/test_runtime_metrics.py tests/test_shadow_gate_evidence.py tests/test_access_control.py tests/test_route_decision.py tests/test_main_slash_commands.py tests/test_product_catalog.py tests/test_backend_catalog.py tests/test_run_contract.py tests/test_task_control_plane.py tests/test_config_validation.py README.md
git commit -m "feat(operations): add structured readiness doctor"
```

### Task 23: Add verified offline encrypted backup and restore

**Files:**
- Create: `src/operations/backup.py`
- Create: `scripts/ghostap_backup.py`
- Create: `tests/test_operations_backup.py`
- Modify: `src/operations/doctor.py`
- Modify: `src/project/manager.py`
- Modify: `src/utils/restart_gate.py`
- Modify: `tests/test_operations_doctor.py`
- Create: `tests/test_project_manager.py`
- Modify: `tests/test_restart_gate.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `BackupSource`, `BackupSourceSet`, `BackupSourceResolver`,
  `BackupManifest`, `BackupVerification`, `BackupQuiescenceGate`,
  `ProjectManager.storage_path`, `RestartGate.offline_backup_guard()`,
  `create_offline_backup()`, `restore_backup()`, `verify_backup()`.

- [ ] **Step 1: Write the failing round-trip and safety tests**

```python
def test_backup_round_trip_preserves_manifest_and_integrity(tmp_path: Path) -> None:
    sources = seed_backup_source_set(
        tmp_path,
        global_state={
            "autonomy/journal/events.jsonl": b"journal",
            "autonomy/blobs/blob-1": b"blob",
            "checkpoints/deep/run-1.json": b"checkpoint",
        },
        project_state={
            "project-a": {
                "workflow_journal/runs/run-a/recovery.json": b"run-a",
            },
            "project-b": {
                "workflow_journal/runs/run-b/recovery.json": b"run-b",
            },
        },
    )
    archive = create_offline_backup(
        source_resolver=static_source_resolver(sources),
        output_path=tmp_path / "ghostap.backup",
        key_path=seed_mode_0600_key(tmp_path),
        quiescence_gate=fake_quiescence_gate(active_writer=False),
        timeout_seconds=5,
    )
    restored = restore_backup(
        archive,
        destination_roots=isolated_restore_roots(tmp_path, sources),
        key_path=tmp_path / "backup.key",
    )
    verification = verify_backup(restored)
    assert verification.valid is True
    assert verification.restored_source_ids == sources.source_ids
```

Assert the round trip restores the global Journal/Blob/checkpoint and both
project-local Workflow runs byte-for-byte. Add failures for duplicate or
overlapping roots, a project root added during resolution, active writer,
symlink, special file, wrong key, corrupt ciphertext, missing logical source,
existing destination, and manifest/hash/source-set mismatch.

- [ ] **Step 2: Run and verify RED**

```bash
uv run python -m pytest tests/test_operations_backup.py -q
```

- [ ] **Step 3: Implement offline-first backup**

```python
@dataclass(frozen=True)
class BackupSource:
    logical_id: str
    canonical_root: Path
    required: bool = True


@dataclass(frozen=True)
class BackupSourceSet:
    schema_version: int
    sources: tuple[BackupSource, ...]
    inventory_sha256: str

    @property
    def source_ids(self) -> tuple[str, ...]: ...


class BackupSourceResolver(Protocol):
    def resolve(self) -> BackupSourceSet:
        """Freeze Settings + ProjectManager roots under the held lease."""


@dataclass(frozen=True)
class BackupManifest:
    schema_version: int
    created_at: str
    source_version: str
    source_inventory_sha256: str
    files: tuple[tuple[str, str], ...]
    journal_head_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BackupLease:
    generation: str
    acquired_at: float


class BackupQuiescenceGate(Protocol):
    @contextmanager
    def offline_backup_guard(
        self,
        *,
        timeout: float,
    ) -> Iterator[BackupLease]:
        """Hold an exclusive, verified no-writer lease for the whole copy."""


def create_offline_backup(
    *,
    source_resolver: BackupSourceResolver,
    output_path: Path,
    key_path: Path,
    quiescence_gate: BackupQuiescenceGate,
    timeout_seconds: float,
) -> Path:
    """Resolve, copy, and verify every source under one supplied lease."""
```

The resolver derives a frozen inventory from effective `Settings`, the public
`ProjectManager.storage_path`, and `ProjectManager.get_all_projects()`:

- include every configured global durable root (projects, Autonomous
  Journal/snapshots/Blobs/data Blobs/credentials/audits, ACP history, run
  checkpoints, and global Workflow templates), even when a setting points
  outside `~/.ghostap`;
- include `<project.root_path>/.ghostap` for every canonical registered project,
  covering Workflow journal/history/templates and future project-local state
  under the reserved namespace;
- assign opaque stable logical IDs derived from state kind + project ID digest,
  never raw chat/user IDs;
- canonicalize roots and reject symlinks, duplicate logical IDs,
  ancestor/child overlaps, filesystem-root/home-directory breadth, and roots
  outside the approved settings/project inventory; backup output, key, staging,
  and restore destinations must all be outside every source root.

Acquire the quiescence lease *before* resolving this inventory and keep that
same lease through enumeration, copy, manifest creation, encryption, fsync, and
verification. Re-snapshot `ProjectManager` before releasing it and fail if the
canonical `(project_id, root_path)` inventory differs. Refuse while the
service/restart gate shows
active writers; reject symlinks and special files; encrypt with AES-GCM using a
mode-0600 operator key; key every manifest file as
`<logical_source_id>/<relative_path>` and record SHA-256 per file. Restore
requires an explicit empty destination mapping for every required logical
source and verifies all roots before any replacement. Never print credentials
or key material.

Promote a narrow public `RestartGate.offline_backup_guard(timeout=...)` rather
than calling its private `_exclusive_guard()`. The public guard acquires
admission and drain exclusively, verifies the gate identity, then checks the
participation/worker registry and refuses if the service or any writer process
is live; it yields a lease containing only the verified generation and
acquisition timestamp. The backup holds that lease through inventory
resolution, file enumeration, copy, manifest creation, encryption, fsync, and
verification. It never performs
“check then unlock then copy.” The CLI resolves the same checkout-scoped
RestartGate as the service and refuses when it cannot prove the service is
stopped. Tests cover a held `task_guard`, a live participation record, timeout,
and a writer appearing before exclusive acquisition.

Add a production-doctor `backup_source_coverage` check that constructs the same
resolver and fails on an omitted, overlapping, symlinked, or unreadable required
source. It validates inventory completeness, not freshness of an operator
backup; the release runbook and Task 26 record the actual encrypted
multi-source round-trip evidence.

- [ ] **Step 4: Verify**

Run:

```bash
uv run python -m pytest \
  tests/test_operations_backup.py \
  tests/test_operations_doctor.py \
  tests/test_project_manager.py \
  tests/test_restart_gate.py -q
```

Then perform a temporary-directory CLI round trip with one global and two
project-local sources.

- [ ] **Step 5: Commit**

```bash
git add src/operations/backup.py src/operations/doctor.py src/project/manager.py scripts/ghostap_backup.py src/utils/restart_gate.py tests/test_operations_backup.py tests/test_operations_doctor.py tests/test_project_manager.py tests/test_restart_gate.py README.md
git commit -m "feat(operations): add verified offline backup"
```

### Task 24: Optionally add a fail-closed OS-isolated Shell profile

**Files:**
- Create: `src/sandbox/isolation.py`
- Create: `tests/test_shell_isolation.py`
- Modify: `src/sandbox/access_policy.py`
- Modify: `src/sandbox/executor.py`
- Modify: `src/config/security_posture.py`
- Modify: `tests/test_sandbox.py`
- Modify: `tests/test_sandbox_security.py`
- Modify: `tests/test_timeout_e2e.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `IsolatedShellRequest`, `IsolatedProcessSpec`,
  `ShellIsolationBackend`, `select_shell_isolation_backend()`.

This task is required only when the selected release advertises or enables
`shell_access_mode=isolated`. A release that keeps Shell disabled records Task
24 as `not_applicable_disabled`, makes no isolation claim, and still passes the
P0 Shell gate through Tasks 2 and 4.

- [ ] **Step 1: Write failing isolation tests**

Prove no fallback, private HOME/tmp, bounded writable roots, minimal
environment, network disabled by default, no login shell, and process-tree
timeout. Also prove a ready backend without Shell caller authorization is still
denied.

- [ ] **Step 2: Run and verify RED**

```bash
uv run python -m pytest \
  tests/test_shell_isolation.py \
  tests/test_sandbox.py \
  tests/test_sandbox_security.py \
  tests/test_timeout_e2e.py -q
```

- [ ] **Step 3: Implement the backend contract**

```python
@dataclass(frozen=True)
class IsolatedShellRequest:
    command: str
    cwd: Path
    timeout_seconds: float
    writable_roots: tuple[Path, ...]
    allow_network: bool
    passthrough_env: frozenset[str]


class ShellIsolationBackend(Protocol):
    def probe(self) -> tuple[bool, str]:
        """Report availability without executing an untrusted workload."""

    def build(self, request: IsolatedShellRequest) -> IsolatedProcessSpec:
        """Build one fail-closed OS-isolated process specification."""
```

Linux uses bubblewrap; macOS uses Seatbelt. Backend unavailability or build
failure never falls back to host execution. Wire the successful probe into
`evaluate_security_posture(..., isolation_ready=...)` and
`ShellAccessPolicy(..., isolation_ready=...)`; no flag asserts readiness.

- [ ] **Step 4: Verify**

Run the RED command again. Expected: every isolated invocation proves the
backend and authorization gates were both passed.

- [ ] **Step 5: Commit**

```bash
git add src/sandbox/isolation.py src/sandbox/access_policy.py src/sandbox/executor.py src/config/security_posture.py tests/test_shell_isolation.py tests/test_sandbox.py tests/test_sandbox_security.py tests/test_timeout_e2e.py README.md
git commit -m "feat(shell): add fail-closed OS isolation"
```

### Task 25: Commit the production-owned signed real-tenant beta gate

**Files:**
- Create: `tests/autonomous/acceptance/test_real_tenant_department_beta.py`
- Modify: `src/autonomous/acceptance/manifest.py`
- Modify: `src/autonomous/acceptance/employee_release.py`
- Modify: `src/autonomous/acceptance/release_trust.py`
- Modify: `src/autonomous/acceptance/employee_release_manifest.json`
- Delete: `tests/autonomous/acceptance/employee_release_manifest.json`
- Modify: `tests/autonomous/acceptance/test_real_tenant_employee_hire.py`
- Modify: `tests/autonomous/acceptance/test_persistent_employee_department.py`
- Modify: `tests/autonomous/contract/test_employee_release_gate.py`
- Modify: `tests/autonomous/contract/test_employee_release_trust.py`
- Modify: `scripts/validate_employee_tenant.py`
- Modify: `docs/adr-employee-runtime-profiles.md`
- Modify: `README.md`

**Interfaces:**
- Produces: one production-owned acceptance manifest and signed evidence bound
  to an immutable candidate commit.
- Consumes: `ReleaseCandidateIdentity` from Task 22 to construct
  `EmployeeEnvironmentBinding`; no separately typed SHA/release/service value.

- [ ] **Step 1: Write failing manifest ownership tests**

Assert production code and both real-tenant test modules load the same path
returned by
`src.autonomous.acceptance.employee_release.default_employee_release_manifest_path()`;
reject a duplicate test-local
release manifest. In particular, replace
`Path(__file__).with_name("employee_release_manifest.json")` in
`test_real_tenant_employee_hire.py`, and replace the test-local
`MANIFEST_PATH` in `test_employee_release_gate.py`. `release_trust.py`,
persistent-department acceptance, and contract tests all consume the same
locator. Validate that all required scenario IDs and evidence fields are
present.

- [ ] **Step 2: Implement and locally verify the candidate**

The manifest must cover admin onboarding/unauthorized denial, hire and Channel
READY, five employee commands, direct/team tasks, employee-owned cards, stop,
fire, restart/reconnect/history, desktop/mobile Slash, and 1/10/50 employee
soaks. Evidence fields are tenant hash, commit SHA, release version, service
instance, timestamp, result, and signature.

Add `EmployeeEnvironmentBinding.from_release_identity(profile_id, identity)`.
It maps `candidate_sha`, release version/ID, service instance, and the two
Employee tenant hashes from the already validated identity. The CLI/env adapter
may retain its existing variable names for compatibility, but it must build and
compare the same identity object; disagreement between
`GHOSTAP_EMPLOYEE_COMMIT_SHA` and `RELEASE_CANDIDATE_SHA`, between Employee and
release service/version fields, or with the independent build attestation is a
hard failure. Contract tests deliberately mismatch each duplicated
compatibility variable and assert zero evidence ingestion.

```bash
uv run python -m pytest \
  tests/autonomous/acceptance/test_real_tenant_department_beta.py \
  tests/autonomous/acceptance/test_real_tenant_employee_hire.py \
  tests/autonomous/acceptance/test_persistent_employee_department.py \
  tests/autonomous/contract/test_employee_release_gate.py \
  tests/autonomous/contract/test_employee_release_trust.py -q
```

Without live opt-in, environment-dependent cases may skip but produce no
evidence and no “pass” claim.

- [ ] **Step 3: Commit the beta candidate before live execution**

```bash
git add src/autonomous/acceptance/manifest.py src/autonomous/acceptance/employee_release.py src/autonomous/acceptance/release_trust.py src/autonomous/acceptance/employee_release_manifest.json scripts/validate_employee_tenant.py tests/autonomous/acceptance/test_real_tenant_department_beta.py tests/autonomous/acceptance/test_real_tenant_employee_hire.py tests/autonomous/acceptance/test_persistent_employee_department.py tests/autonomous/contract/test_employee_release_gate.py tests/autonomous/contract/test_employee_release_trust.py tests/autonomous/acceptance/employee_release_manifest.json docs/adr-employee-runtime-profiles.md README.md
git commit -m "test(acceptance): define department tenant beta gate"
```

The `git add` path for the deleted test manifest intentionally stages its
deletion.

- [ ] **Step 4: Defer live execution until the final behavior candidate**

Do not bind live evidence to this intermediate gate-definition commit. Task 26
first commits all selected-track defaults and behavior into one immutable
candidate, then deploys and executes this gate against that exact SHA. This task
only proves that the manifest, validator, signing contract, and local
environment opt-in behavior are ready.

- [ ] **Step 5: Preserve the promotion rule**

Any missing scenario, skip, signature failure, tenant/SHA mismatch, or 1/10/50
soak failure blocks the production-ready claim. Local tests alone qualify only
as release-candidate evidence.

### Task 26: Cut over defaults, verify the repository, and record evidence

**Files:**
- Modify: `src/config/settings.py`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `docs/product-contract.md`
- Modify: `docs/2026-07-27-main-agent-slash-commands-plan.md`
- Create or modify: `.Memory/{UTC implementation date}.md`
- Modify: `.Memory/Abstract.md`
- Modify: `.Memory/Backlog.md` only for verified medium/low residuals

**Interfaces:**
- Consumes: every task required by the selected release track.
- Produces: Department v1 defaults, a truthful release label, release evidence,
  rollback matrix, and durable project memory.

- [ ] **Step 1: Select and record one release track**

Two tracks are valid:

- **Track A — Control-plane convergence:** Tasks 1–15, 17, 21–23, and 25 are
  complete; DG-2 is not approved. Task 24 is complete only if isolated Shell is
  enabled; otherwise it is explicitly `not_applicable_disabled`. The release
  may claim the 11-command product
  surface, secure defaults, unified task view, provider catalog, and truthful
  read-only lifecycle projection. It must not claim Spec provider-local retry,
  Worktree hard cancellation, or Deep/Workflow restart recovery.
- **Track B — Hardened runtime:** Tasks 1–23 and 25 are complete, DG-2 evidence
  exists, and Task 24 follows the same Shell-profile rule. The release may
  additionally claim the Task 16 and 18–20 engine contracts.

Write the selected track and exact completed task IDs into release evidence.
Task 26 does not treat a deliberately gated task as passed.
Both production tracks require a recorded DG-1A disposition. If the owner keeps
`admin_bootstrap_scope=any_chat`, the artifact is migration-only and
`admin_bootstrap_any_chat` remains an explicit production blocker:
production-profile doctor returns required `fail`, never a ready warning.

- [ ] **Step 2: Prepare the selected-track defaults locally**

Set:

```python
ingress_access_mode = "enforced"
main_slash_surface = "department_v1"
feishu_route_decision_mode = "enforced"
backend_catalog_mode = "enforced"
employee_department_enabled = False
shell_access_mode = "disabled"
```

Set `run_lifecycle_mode="read_only"` for Track A and
`run_lifecycle_mode="enforced"` for Track B; exactly one value is written.
Set `admin_bootstrap_scope="p2p_only"` only with DG-1A approval and the
same-commit `AGENTS.md` contract; otherwise production promotion remains blocked
as specified above.

`employee_department_enabled` and Shell remain explicit opt-ins. Keep legacy
command parsers and route rollback for one release. Mark the old 79-command plan
as superseded by this plan without rewriting its historical completion record.
These values are committed before live observation so the final behavior has an
immutable SHA. Existing deployments use explicit environment overrides for the
shadow stage; fresh-install defaults remain secure in source.

- [ ] **Step 3: Run focused phase suites**

```bash
uv run python -m pytest \
  tests/test_security_posture.py \
  tests/test_access_control.py \
  tests/test_shell_access_policy.py \
  tests/test_mode_transition.py \
  tests/test_product_catalog.py \
  tests/test_product_menu_cards.py \
  tests/test_effective_context.py \
  tests/test_route_decision.py \
  tests/test_route_executor.py \
  tests/test_task_control_plane.py \
  tests/test_control_plane_handler.py \
  tests/test_backend_catalog.py \
  tests/test_run_contract.py -q
```

Expected: PASS.

- [ ] **Step 4: Run subsystem and pre-candidate repository gates**

```bash
uv run python -m pytest tests/autonomous/ -q
uv run python scripts/test_inventory.py tests/
uv run python -m pytest tests/ -q -m "not slow"
uv run python -m pytest tests/ -q -m slow
uv run python -m pytest tests/ -q
uv run ruff check src/
uv run python -m src.main --validate
uv run python scripts/ghostap_doctor.py --profile candidate --json
git diff --check
```

Expected: every command exits 0 with a final summary. Investigate all failures,
warnings tied to changed contracts, and unexpected skips.
Backlog B052 covers the pre-existing test-only Ruff cleanup; this plan does not
misreport `tests/` Ruff as green. Candidate-profile doctor validates local
structure and reports live/shadow evidence as pending; it does not claim
production readiness.

- [ ] **Step 5: Commit and freeze the behavior candidate**

```bash
git add src/config/settings.py .env.example README.md docs/product-contract.md docs/2026-07-27-main-agent-slash-commands-plan.md
git commit -m "feat(product): prepare Agent Department v1 candidate"
git status --short
```

Require a clean worktree, record `git rev-parse HEAD` as `candidate_sha`, and do
not amend, merge, rebase, or add any source/config commit while gathering
evidence. All selected-track tasks, including Task 25's production-owned
manifest, must be ancestors of this SHA. Any behavior fix creates a new
candidate and resets every SHA-bound observation window.

Create one deployment-owned, immutable candidate identity record after this
commit. It contains the derived checkout/image `build_attestation_sha`, the
recorded candidate SHA, release version, service instance ID, shadow tenant
hash, and distinct Employee staging/production tenant hashes. The deployment
system derives the build SHA independently from Git/image provenance; it must
not populate both SHA fields by copying one operator input. Validate equality
before service startup.

- [ ] **Step 6: Observe all four shadows on that exact candidate SHA**

Deploy the exact candidate to the controlled tenant with explicit overrides:

```text
GHOSTAP_BUILD_COMMIT_SHA=<independently attested deployed SHA>
RELEASE_CANDIDATE_SHA=<candidate_sha from Step 5>
RELEASE_VERSION=<immutable release ID>
GHOSTAP_SERVICE_INSTANCE_ID=<immutable service instance ID>
RELEASE_TENANT_HASH=<controlled shadow tenant SHA-256>
AUTONOMOUS_EMPLOYEE_STAGING_TENANT_HASH=<staging tenant SHA-256>
AUTONOMOUS_EMPLOYEE_PRODUCTION_TENANT_HASH=<production tenant SHA-256>
GHOSTAP_EMPLOYEE_RELEASE_ID=<same immutable release ID>
GHOSTAP_EMPLOYEE_COMMIT_SHA=<same candidate_sha>
GHOSTAP_EMPLOYEE_SERVICE_INSTANCE_ID=<same service instance ID>
GHOSTAP_EMPLOYEE_STAGING_TENANT_HASH=<same staging tenant SHA-256>
GHOSTAP_EMPLOYEE_PRODUCTION_TENANT_HASH=<same production tenant SHA-256>
INGRESS_ACCESS_MODE=shadow
MAIN_SLASH_SURFACE=shadow
FEISHU_ROUTE_DECISION_MODE=shadow
BACKEND_CATALOG_MODE=shadow
```

Resolve those compatibility variables into one `ReleaseCandidateIdentity` at
startup and fail if any duplicate differs. Keep the exact same identity values
for Step 7; only the four mode overrides and explicit Employee enablement may
change.

Run the four observations concurrently for at least 48 hours. Require:

- Task 3: zero unexplained ingress `shadow_would_deny` after reviewed allowlists;
- Task 9: exact 79-command compatibility and stable 11-command desired surface;
- Task 12: zero unexplained route-decision divergence;
- Task 14: zero unexplained backend/transport/model/options divergence.

Freeze the exact end-of-window runtime snapshot into one
`ShadowEvidenceBundle`; each independently signed report includes and signs the
same snapshot digest, candidate SHA, tenant hash, release version, service
instance, start/end timestamps, catalog/contract fingerprint, denominator
sample count, and anomaly counts. Later Task 3/WS/dispatcher changes—or later
mutations of the live counter file—therefore cannot reuse or rewrite evidence
from an earlier commit. One shadow gate never substitutes for another.

- [ ] **Step 7: Enforce the same SHA and run product acceptance**

Remove the four shadow overrides and redeploy the *same* candidate SHA so its
committed defaults are active. First execute the disabled-profile privacy
scenario (`AC-08`) with `EMPLOYEE_DEPARTMENT_ENABLED=false`. Then enable the
Employee Department explicitly for the controlled beta tenant with
`EMPLOYEE_DEPARTMENT_ENABLED=true`, keeping the same SHA, reviewed ingress
allowlists, and Shell disabled. Record both effective configuration
fingerprints. Execute Task 25's explicitly opted-in real-tenant validator and
the following desktop/mobile scenarios:

```text
AC-01 Fresh install rejects ordinary traffic and explains /setadmin.
AC-02 Admin enrolls one group; a different group remains denied.
AC-03 Natural-language task starts without provider/engine selection.
AC-04 /status shows Programming, Deep, Spec, WT, WF, Slock, Department, Team,
      Employee and Scheduler.
AC-05 /stop, /retry and /approve fail closed on ambiguous or out-of-scope tasks.
AC-06 Hidden provider/engine commands remain directly callable.
AC-07 Engine topic rejects provider switching without any state mutation.
AC-08 Employee Department OFF records no group text and opens no employee stores.
AC-09 Retention tombstone survives restart and cannot resurrect payload.
AC-10 Host Shell is disabled, isolated, or explicitly acknowledged trusted-local.
AC-11B Spec provider-local retry, Worktree bounded cancel/terminal truth, and
       Deep/Workflow restart recovery match the DG-2-approved contracts.
AC-12 Main Slash surface contains exactly 11 public commands.
AC-13 Offline encrypted backup inventory includes every configured global root
      and all registered project `.ghostap` roots; verification plus isolated
      restore reproduces Journal/Blob/checkpoint and two Workflow runs.
```

Record evidence for each scenario. `AC-11B` is required only for Track B; Track
A records it as `not_authorized_by_DG-2`, not pass or skip. Every other
unexecuted scenario is not “passed”.

Store raw signed evidence only in the approved secure evidence store, never the
repository. Every Task 25 scenario, signature, tenant/SHA binding, restart/
reconnect check, and 1/10/50 employee soak must pass. Run:

```bash
uv run python scripts/validate_employee_tenant.py --live \
  --live-results "$APPROVED_REDACTED_CAPTURE"
uv run python scripts/ghostap_doctor.py --profile production --json
```

The validator receives `GHOSTAP_EMPLOYEE_*` values from the same immutable
candidate identity above; do not type a second SHA/release/service binding at
execution time.

Production doctor must consume the immutable signed
`ShadowEvidenceBundle`, including its frozen `RuntimeHealthSnapshot`, for the
same SHA. It must not read a newer mutable runtime snapshot as release evidence.
It also re-reads effective settings and requires `ingress_access_mode=enforced`
plus an approved P2P-only first-admin disposition; shadow/legacy ingress or
`admin_bootstrap_scope=any_chat` is a required failure even when all signed
shadow evidence passes.
Missing, skipped, expired, mismatched, or corrupt evidence blocks promotion. If
all gates pass, tag/promote the exact candidate SHA; do not create a new
behavior commit.

- [ ] **Step 8: Record evidence in a non-deployed documentation commit**

Use the UTC date on which Task 26 is executed, not the plan-authoring date:

```bash
implementation_memory=".Memory/$(date -u +%F).md"
```

That daily file must record:

- product contract and why provider/engine concepts moved under Advanced;
- exact security defaults and migration impact;
- every changed route/lifecycle/storage contract;
- exact focused, non-slow, slow, full, Ruff, validate, and inventory results;
- real-tenant evidence status, including explicit missing evidence;
- immutable candidate SHA/release tag, signed evidence digests, and observation
  windows;
- rollback flags and remaining risks.

Then:

```bash
git add "$implementation_memory" .Memory/Abstract.md .Memory/Backlog.md
git commit -m "docs(memory): record Agent Department v1 evidence"
```

This follow-up is evidence-only and is not the deployed release artifact. If it
changes source, configuration, acceptance manifests, or generated runtime
assets, the candidate SHA is invalid and Steps 3–8 restart.

---

## Rollout and Rollback Matrix

| Area | Shadow/dual mode | Enforced mode | Safe rollback |
| --- | --- | --- | --- |
| Ingress ACL | log prospective denials in controlled tenant only | deny unenrolled user/chat | explicit time-bounded legacy mode with critical warning |
| Shell | disabled by default | admin/allowlisted/isolated/trusted-local explicit | disable Shell; never auto-fallback from isolation |
| Employee runtime | disabled | explicit `EMPLOYEE_DEPARTMENT_ENABLED=true` | set false and restart |
| Slash surface | legacy 79 | department v1 11 | set legacy and reconcile; parsers remain |
| Routing | shadow compare, execute legacy once | execute RouteDecision once | set legacy; no double execution |
| Backend catalog | shadow selection | enforced selection | return to shadow; keep descriptors |
| Run lifecycle | read-only mapping/dual write | engine-specific enforced read/write | return to dual write; preserve checkpoints |
| Workflow Journal | new run-local write/read | startup recovery for same run | disable recovery read; retain journal |
| Legacy Manager | fail-closed shim | quarantined | shim only; never revive work acceptance |

## Product Success Metrics

Measure for every release:

- `% natural-language tasks started without manual provider/engine selection`
- `% tasks reaching a truthful terminal state with recorded verification`
- `p50/p95 time to first visible progress`
- `p50/p95 stop-to-terminal time`
- restart recovery success rate by task source
- route shadow divergence and misroute count
- false-success and terminal-overwrite count
- provider drift count
- ambiguous/out-of-scope control rejection count
- task retry/recover success rate
- employee Channel reconnect success
- GroupLedger bytes, tombstoned payloads, and pending purge count
- per-task token/cost when the provider reports trustworthy usage

Do not optimize command count or test count as a north-star metric. The target is
task completion without mode expertise, with truthful security and lifecycle
semantics.

## Stop-Expansion Rules

Until Task 26 passes:

- add no new top-level mode, provider-specific command, Workflow primitive, or
  independent task-state persistence format beyond the checkpoint/run-Journal
  contracts in this plan; bounded operational evidence from Tasks 22 and 25 is
  the only explicit exception;
- add no new public Slash command outside the 11-command contract;
- do not create another task/status abstraction outside `src/tasking/`;
- do not create another provider allowlist outside `BackendCatalog`;
- do not add another production Autonomous composition root;
- route all newly found high correctness/security defects into the current task,
  while medium/low findings enter `.Memory/Backlog.md`.

## Self-Review Checklist

- [ ] Every audit P0 has a task: inbound ACL, Shell trust, Employee OFF, group retention, topic transition.
- [ ] Every user-visible contradiction has a task: positioning, `/goal`, `/hire --prompt`, `/status`, Workflow scope, default provider.
- [ ] Provider/engine hiding preserves direct compatibility inputs.
- [ ] Effective context, RouteDecision, task read/control, backend catalog, and run lifecycle each have exactly one owner.
- [ ] The no-database, main WebSocket, Slock ACP, Journal, frozen-domain, and anchoring constraints are preserved.
- [ ] Engine-internal changes cannot begin without DG-2 approval.
- [ ] Every task specifies exact files, interfaces, RED/GREEN commands, acceptance, and commit.
- [ ] No task treats a skipped live test as real-tenant evidence.
- [ ] No planning placeholder remains.
