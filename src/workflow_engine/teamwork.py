"""Teamwork multi-role collaboration presets, role assignment, and script generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .agent_pool import WorkflowAgentBinding, freeze_agent_pool

TEAMWORK_ROLES: tuple[str, ...] = (
    "scout",
    "planner",
    "worker",
    "reviewer",
    "synthesizer",
)

ROLE_ALIASES: Mapping[str, str] = {
    "scout": "scout",
    "explorer": "scout",
    "planner": "planner",
    "orchestrator": "planner",
    "architect": "planner",
    "lead": "planner",
    "worker": "worker",
    "implementer": "worker",
    "coder": "worker",
    "developer": "worker",
    "executor": "worker",
    "focused-executor": "worker",
    "focused_executor": "worker",
    "reviewer": "reviewer",
    "verifier": "reviewer",
    "adversarial_verifier": "reviewer",
    "auditor": "reviewer",
    "judge": "reviewer",
    "synthesizer": "synthesizer",
    "synth": "synthesizer",
    "summarizer": "synthesizer",
}

_EFFORT_SCORES: Mapping[str | None, int] = {
    "off": 0,
    "none": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "default": 3,
    None: 3,
    "high": 4,
    "xhigh": 5,
    "max": 6,
    "ultra": 7,
}

_ROLE_TOOL_AFFINITY: Mapping[str, tuple[str, ...]] = {
    "scout": ("traex", "codex", "grok", "dsh", "gemini", "agy", "claude", "coco", "aiden"),
    "planner": ("claude", "agy", "aiden", "coco", "codex", "dsh", "traex", "gemini", "grok"),
    "worker": ("codex", "claude", "coco", "agy", "traex", "dsh", "grok", "gemini", "aiden"),
    "reviewer": ("claude", "aiden", "agy", "codex", "dsh", "grok", "traex", "gemini", "coco"),
    "synthesizer": ("claude", "agy", "codex", "dsh", "coco", "aiden", "traex", "gemini", "grok"),
}


@dataclass(frozen=True, slots=True)
class TeamworkPreset:
    """Configuration defining one Teamwork multi-role collaboration topology."""

    preset_id: str
    display_name: str
    description: str
    phases: tuple[str, ...]
    enable_scout: bool
    enable_planner: bool
    parallel_workers: bool
    max_verify_rounds: int
    multi_verifier: bool
    require_synthesizer: bool
    worker_effort_bias: str
    max_concurrent_cap: int


TEAMWORK_PRESETS: Mapping[str, TeamworkPreset] = {
    "recommended": TeamworkPreset(
        preset_id="recommended",
        display_name="推荐协作 (Standard)",
        description="Scout -> Execute -> Cross-Tool Verify -> Synthesize",
        phases=("Scout", "Execute", "Verify", "Synthesize"),
        enable_scout=True,
        enable_planner=False,
        parallel_workers=True,
        max_verify_rounds=2,
        multi_verifier=False,
        require_synthesizer=True,
        worker_effort_bias="balanced",
        max_concurrent_cap=6,
    ),
    "quick": TeamworkPreset(
        preset_id="quick",
        display_name="极速协作 (Quick)",
        description="Fast execution followed by one-round verification",
        phases=("Execute", "Verify"),
        enable_scout=False,
        enable_planner=False,
        parallel_workers=False,
        max_verify_rounds=1,
        multi_verifier=False,
        require_synthesizer=False,
        worker_effort_bias="low",
        max_concurrent_cap=2,
    ),
    "collaborative": TeamworkPreset(
        preset_id="collaborative",
        display_name="多角色协作 (Collaborative)",
        description="Scout -> Parallel Fanout Workers -> Cross-Tool Verify -> Synthesize",
        phases=("Scout", "Execute", "Verify", "Synthesize"),
        enable_scout=True,
        enable_planner=False,
        parallel_workers=True,
        max_verify_rounds=2,
        multi_verifier=False,
        require_synthesizer=True,
        worker_effort_bias="balanced",
        max_concurrent_cap=6,
    ),
    "review_team": TeamworkPreset(
        preset_id="review_team",
        display_name="对抗复核 (Review Team)",
        description="Worker execution with multi-verifier adversarial review",
        phases=("Execute", "Verify", "Synthesize"),
        enable_scout=False,
        enable_planner=False,
        parallel_workers=False,
        max_verify_rounds=3,
        multi_verifier=True,
        require_synthesizer=True,
        worker_effort_bias="high",
        max_concurrent_cap=4,
    ),
    "deep_collab": TeamworkPreset(
        preset_id="deep_collab",
        display_name="深度协作 (Deep Collab)",
        description="Scout -> Plan -> Parallel Workers -> Adversarial Verify -> Synthesize",
        phases=("Scout", "Plan", "Execute", "Verify", "Synthesize"),
        enable_scout=True,
        enable_planner=True,
        parallel_workers=True,
        max_verify_rounds=2,
        multi_verifier=True,
        require_synthesizer=True,
        worker_effort_bias="high",
        max_concurrent_cap=6,
    ),
}

PRESET_ALIASES: Mapping[str, str] = {
    "auto": "recommended",
    "standard": "recommended",
    "default": "recommended",
    "fast": "quick",
    "review": "review_team",
    "deep": "deep_collab",
}


@dataclass(frozen=True, slots=True)
class TeamworkRolePlan:
    """Resolved Teamwork role assignments for a confirmed Agent pool."""

    preset: TeamworkPreset
    pool: tuple[WorkflowAgentBinding, ...]
    orchestrator_agent_id: str
    scout_id: str
    planner_id: str
    worker_ids: tuple[str, ...]
    reviewer_ids: tuple[str, ...]
    synthesizer_id: str
    assignments: Mapping[str, str]


def normalize_teamwork_role(role: str | None) -> str | None:
    """Canonicalize role names and aliases into one of TEAMWORK_ROLES."""
    cleaned = str(role or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not cleaned:
        return None
    if cleaned in ROLE_ALIASES:
        return ROLE_ALIASES[cleaned]
    if cleaned in TEAMWORK_ROLES:
        return cleaned
    return None


def list_teamwork_presets() -> tuple[TeamworkPreset, ...]:
    """Return ordered Teamwork presets for selection and UI rendering."""
    return tuple(TEAMWORK_PRESETS.values())


def resolve_teamwork_preset(preset_id: str | None = None) -> TeamworkPreset:
    """Resolve a canonical preset or alias; raise ValueError on unknown IDs."""
    cleaned = str(preset_id or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not cleaned:
        return TEAMWORK_PRESETS["recommended"]
    canonical = PRESET_ALIASES.get(cleaned, cleaned)
    preset = TEAMWORK_PRESETS.get(canonical)
    if preset is None:
        raise ValueError(f"Unknown Teamwork preset: {preset_id}")
    return preset


def _effort_rank(binding: WorkflowAgentBinding) -> int:
    effort = str(binding.effort or "").strip().lower() or None
    base = _EFFORT_SCORES.get(effort, 3)
    if str(binding.profile or "").strip().lower() == "max":
        return base + 1
    return base


def _agent_id_tiebreak(agent_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"A-?(\d+)", agent_id, re.IGNORECASE)
    return (int(match.group(1)), "") if match else (10**9, agent_id)


def _explicit_roles_for(binding: WorkflowAgentBinding) -> tuple[str, ...]:
    roles: list[str] = []
    for raw in (binding.role, *binding.roles):
        canonical = normalize_teamwork_role(raw)
        if canonical and canonical not in roles:
            roles.append(canonical)
    return tuple(roles)


def _role_score(
    binding: WorkflowAgentBinding,
    role: str,
    *,
    preset: TeamworkPreset,
    pool_index: int,
    avoid_tool: str | None = None,
) -> tuple[int, ...]:
    explicit = _explicit_roles_for(binding)
    primary_explicit = normalize_teamwork_role(binding.role)
    explicit_penalty = 0 if primary_explicit == role else (1 if role in explicit else 2)

    affinity = _ROLE_TOOL_AFFINITY.get(role, ())
    tool_rank = affinity.index(binding.tool_name) if binding.tool_name in affinity else len(affinity)

    cross_tool_penalty = 1 if (avoid_tool and binding.tool_name == avoid_tool) else 0

    effort = _effort_rank(binding)
    if role == "scout":
        effort_penalty = effort
    elif role == "worker":
        if preset.worker_effort_bias == "low":
            effort_penalty = effort
        elif preset.worker_effort_bias == "high":
            effort_penalty = -effort
        else:
            effort_penalty = abs(effort - 3)
    else:
        # planner, reviewer, synthesizer prefer higher effort tiers.
        effort_penalty = -effort

    # Prefer bindings without conflicting primary explicit role when falling back
    other_explicit_penalty = (
        1 if (explicit_penalty == 2 and primary_explicit is not None and primary_explicit != role) else 0
    )

    return (
        explicit_penalty,
        cross_tool_penalty,
        other_explicit_penalty,
        effort_penalty,
        tool_rank,
        pool_index,
        *_agent_id_tiebreak(binding.agent_id),
    )


def _pick_best(
    pool: tuple[WorkflowAgentBinding, ...],
    role: str,
    *,
    preset: TeamworkPreset,
    exclude_ids: set[str] | None = None,
    avoid_tool: str | None = None,
) -> WorkflowAgentBinding:
    excluded = exclude_ids or set()
    candidates = [
        (idx, binding)
        for idx, binding in enumerate(pool)
        if binding.agent_id not in excluded
    ]
    if not candidates:
        candidates = list(enumerate(pool))
    _, chosen = min(
        candidates,
        key=lambda item: _role_score(
            item[1],
            role,
            preset=preset,
            pool_index=item[0],
            avoid_tool=avoid_tool,
        ),
    )
    return chosen


def assign_teamwork_roles(
    agent_pool: Iterable[WorkflowAgentBinding | dict[str, Any]],
    *,
    preset: str | TeamworkPreset = "recommended",
    orchestrator_agent_id: str | None = None,
) -> TeamworkRolePlan:
    """Assign pool bindings to Teamwork roles using explicit roles, effort tiers, and tool affinity."""
    pool = freeze_agent_pool(agent_pool, require_nonempty=True)
    resolved_preset = preset if isinstance(preset, TeamworkPreset) else resolve_teamwork_preset(preset)
    by_id = {b.agent_id: b for b in pool}

    raw_orch = str(orchestrator_agent_id or "").strip()
    if raw_orch and raw_orch not in by_id:
        raise ValueError(f"orchestrator_agent_id is outside confirmed agent_pool: {raw_orch}")

    if len(pool) == 1:
        only = pool[0]
        aid = only.agent_id
        return TeamworkRolePlan(
            preset=resolved_preset,
            pool=pool,
            orchestrator_agent_id=aid,
            scout_id=aid,
            planner_id=aid,
            worker_ids=(aid,),
            reviewer_ids=(aid,),
            synthesizer_id=aid,
            assignments={
                "scout": aid,
                "planner": aid,
                "worker": aid,
                "reviewer": aid,
                "synthesizer": aid,
            },
        )

    # Determine if an explicit reviewer exists or pick top reviewer candidate first
    # when len(pool) == 2 so that high-effort/reasoning members are preserved for
    # independent review when possible.
    if len(pool) == 2:
        # Check if either member is explicitly designated as worker or reviewer.
        explicit_workers = [b for b in pool if "worker" in _explicit_roles_for(b)]
        explicit_reviewers = [b for b in pool if "reviewer" in _explicit_roles_for(b)]
        if explicit_workers and not explicit_reviewers:
            primary_worker = _pick_best(tuple(explicit_workers), "worker", preset=resolved_preset)
            primary_reviewer = _pick_best(
                pool,
                "reviewer",
                preset=resolved_preset,
                exclude_ids={primary_worker.agent_id},
                avoid_tool=primary_worker.tool_name,
            )
        else:
            primary_reviewer = _pick_best(pool, "reviewer", preset=resolved_preset)
            primary_worker = _pick_best(
                pool,
                "worker",
                preset=resolved_preset,
                exclude_ids={primary_reviewer.agent_id},
            )
    else:
        primary_worker = _pick_best(pool, "worker", preset=resolved_preset)
        primary_reviewer = _pick_best(
            pool,
            "reviewer",
            preset=resolved_preset,
            exclude_ids={primary_worker.agent_id},
            avoid_tool=primary_worker.tool_name,
        )

    # Scout prefers fast/low-effort members (can share with worker if pool == 2,
    # or pick distinct non-reviewer when pool >= 3).
    scout_exclude = {primary_reviewer.agent_id} if len(pool) >= 3 else set()
    scout = _pick_best(pool, "scout", preset=resolved_preset, exclude_ids=scout_exclude)

    # Orchestrator / planner / synthesizer anchoring: ensure explicit orchestrator_agent_id
    # is actively bound into the plan.
    orch_id = raw_orch or _pick_best(pool, "planner", preset=resolved_preset).agent_id
    if raw_orch:
        planner = by_id[raw_orch] if resolved_preset.enable_planner else _pick_best(pool, "planner", preset=resolved_preset)
        if resolved_preset.require_synthesizer:
            synthesizer = by_id[raw_orch]
        else:
            synthesizer = _pick_best(pool, "synthesizer", preset=resolved_preset)
            # Ensure orchestrator appears in active quick roles (worker or reviewer)
            if (
                not resolved_preset.enable_planner
                and not resolved_preset.enable_scout
                and not resolved_preset.require_synthesizer
                and raw_orch not in {primary_worker.agent_id, primary_reviewer.agent_id}
            ):
                primary_reviewer = by_id[raw_orch]
    else:
        planner = _pick_best(pool, "planner", preset=resolved_preset)
        synthesizer = _pick_best(pool, "synthesizer", preset=resolved_preset)

    # Worker list for fanout: exclude primary reviewer whenever len(pool) >= 3 so
    # reviewer remains strictly independent from all workers; for len == 2 keep
    # single primary worker so primary_worker != primary_reviewer holds.
    worker_ids_list: list[str] = [primary_worker.agent_id]
    if resolved_preset.parallel_workers and len(pool) >= 3:
        explicit_extra_workers = [
            b
            for b in pool
            if b.agent_id not in {primary_worker.agent_id, primary_reviewer.agent_id}
            and "worker" in _explicit_roles_for(b)
        ]
        other_non_reviewer = [
            b
            for b in pool
            if b.agent_id not in {primary_worker.agent_id, primary_reviewer.agent_id}
            and "worker" not in _explicit_roles_for(b)
            and normalize_teamwork_role(b.role) != "reviewer"
        ]
        # Sort candidates by worker score
        extra_candidates = sorted(
            explicit_extra_workers + other_non_reviewer,
            key=lambda b: _role_score(
                b,
                "worker",
                preset=resolved_preset,
                pool_index=pool.index(b),
            ),
        )
        max_workers = max(1, min(resolved_preset.max_concurrent_cap, len(pool) - 1))
        for cand in extra_candidates:
            if len(worker_ids_list) >= max_workers:
                break
            if cand.agent_id not in worker_ids_list:
                worker_ids_list.append(cand.agent_id)

    # Reviewer list: multi-verifier presets add a second independent verifier when pool >= 3.
    reviewer_ids_list: list[str] = [primary_reviewer.agent_id]
    if resolved_preset.multi_verifier and len(pool) >= 3:
        excluded_for_extra_rev = set(worker_ids_list) | set(reviewer_ids_list)
        extra_pool = [b for b in pool if b.agent_id not in excluded_for_extra_rev]
        if not extra_pool:
            extra_pool = [
                b
                for b in pool
                if b.agent_id != primary_worker.agent_id and b.agent_id not in reviewer_ids_list
            ]
        if extra_pool:
            second_rev = _pick_best(
                tuple(extra_pool),
                "reviewer",
                preset=resolved_preset,
                avoid_tool=primary_worker.tool_name,
            )
            reviewer_ids_list.append(second_rev.agent_id)

    # Guarantee raw_orch appears in at least one active phase node if supplied.
    if raw_orch:
        active_ids: set[str] = set(worker_ids_list) | set(reviewer_ids_list)
        if resolved_preset.enable_scout:
            active_ids.add(scout.agent_id)
        if resolved_preset.enable_planner:
            active_ids.add(planner.agent_id)
        if resolved_preset.require_synthesizer:
            active_ids.add(synthesizer.agent_id)
        if raw_orch not in active_ids:
            if resolved_preset.require_synthesizer:
                synthesizer = by_id[raw_orch]
            elif resolved_preset.enable_planner:
                planner = by_id[raw_orch]
            elif resolved_preset.enable_scout:
                scout = by_id[raw_orch]
            else:
                reviewer_ids_list[0] = raw_orch

    assignments = {
        "scout": scout.agent_id,
        "planner": planner.agent_id,
        "worker": worker_ids_list[0],
        "reviewer": reviewer_ids_list[0],
        "synthesizer": synthesizer.agent_id,
    }
    return TeamworkRolePlan(
        preset=resolved_preset,
        pool=pool,
        orchestrator_agent_id=orch_id,
        scout_id=scout.agent_id,
        planner_id=planner.agent_id,
        worker_ids=tuple(worker_ids_list),
        reviewer_ids=tuple(reviewer_ids_list),
        synthesizer_id=synthesizer.agent_id,
        assignments=assignments,
    )


def _safe_meta_text(text: str) -> str:
    """Strip comment sequences from strings placed in export const meta."""
    return text.replace("//", " ").replace("/*", " ").replace("*/", " ").strip()


def _handoff_instructions(role_desc: str) -> str:
    return (
        f"Role: {role_desc}\n"
        "If the user asks for analysis only, planning only, or says not to change code, do not change code.\n"
        "On a blocker, return a clear error instead of asking the user.\n"
        "Structure your response as JSON (or clear labeled sections) with the 5-component Teamwork handoff:\n"
        "- observation: direct file paths, symbols, commands, and facts observed\n"
        "- logic_chain: step-by-step reasoning from observations to changes or findings\n"
        "- caveats: assumptions, edge cases, or areas not covered\n"
        "- conclusion: final actionable summary or implemented outcome\n"
        "- verification: commands/tests executed and exact pass/fail evidence"
    )


def generate_teamwork_script(
    requirement: str,
    *,
    agent_pool: Iterable[WorkflowAgentBinding | dict[str, Any]],
    preset: str | TeamworkPreset = "recommended",
    orchestrator_agent_id: str | None = None,
) -> str:
    """Generate a deterministic ES Module Teamwork workflow script for a confirmed pool."""
    from .script_gen import get_subagent_encouragement

    plan = assign_teamwork_roles(
        agent_pool,
        preset=preset,
        orchestrator_agent_id=orchestrator_agent_id,
    )
    by_id = {b.agent_id: b for b in plan.pool}
    encouragement = get_subagent_encouragement()
    enc_suffix = f"\n\n{encouragement}" if encouragement else ""
    req_clean = str(requirement or "").strip()

    use_fanout = plan.preset.parallel_workers and len(plan.worker_ids) >= 2

    # Build meta.agentPlan & track used tools/patterns
    agent_plan_entries: list[dict[str, Any]] = []
    used_agent_ids: list[str] = []

    def _track_agent(node: str, role: str, agent_id: str) -> None:
        binding = by_id[agent_id]
        entry: dict[str, Any] = {
            "node": node,
            "role": role,
            "agentId": agent_id,
        }
        if binding.effort:
            entry["effort"] = binding.effort
        if binding.profile:
            entry["profile"] = binding.profile
        agent_plan_entries.append(entry)
        if agent_id not in used_agent_ids:
            used_agent_ids.append(agent_id)

    phases_meta: list[dict[str, str]] = []
    if plan.preset.enable_scout:
        phases_meta.append({"title": "Scout", "detail": "Discover scope, files, and constraints"})
        _track_agent("scout", "scout", plan.scout_id)
    if plan.preset.enable_planner:
        phases_meta.append({"title": "Plan", "detail": "Design execution plan and verification criteria"})
        _track_agent("plan", "planner", plan.planner_id)

    phases_meta.append({"title": "Execute", "detail": "Execute task with bounded worker roles"})
    for idx, wid in enumerate(plan.worker_ids, 1):
        _track_agent(f"worker-{idx}", "worker", wid)

    phases_meta.append({"title": "Verify", "detail": "Independent adversarial review and revision"})
    for idx, rid in enumerate(plan.reviewer_ids, 1):
        _track_agent(f"verify-{idx}", "reviewer", rid)

    if plan.preset.require_synthesizer:
        phases_meta.append({"title": "Synthesize", "detail": "Consolidate verified handoffs and evidence"})
        _track_agent("synthesize", "synthesizer", plan.synthesizer_id)

    used_tools = list(dict.fromkeys(by_id[aid].tool_name for aid in used_agent_ids))
    patterns: list[str] = []
    if use_fanout:
        patterns.append("fanout")
    else:
        patterns.append("sequence")
    patterns.append("verify")

    max_concurrent = max(
        1,
        min(
            plan.preset.max_concurrent_cap,
            max(len(plan.worker_ids), len(plan.reviewer_ids), 2),
        ),
    )

    phases_js = ",\n    ".join(
        f'{{ title: {json.dumps(_safe_meta_text(p["title"]), ensure_ascii=False)}, '
        f'detail: {json.dumps(_safe_meta_text(p["detail"]), ensure_ascii=False)} }}'
        for p in phases_meta
    )
    agent_plan_items: list[str] = []
    for entry in agent_plan_entries:
        fields = [
            f'node: {json.dumps(_safe_meta_text(entry["node"]), ensure_ascii=False)}',
            f'role: {json.dumps(_safe_meta_text(entry["role"]), ensure_ascii=False)}',
            f'agentId: {json.dumps(entry["agentId"], ensure_ascii=False)}',
        ]
        if entry.get("effort"):
            fields.append(f'effort: {json.dumps(_safe_meta_text(entry["effort"]), ensure_ascii=False)}')
        if entry.get("profile"):
            fields.append(f'profile: {json.dumps(_safe_meta_text(entry["profile"]), ensure_ascii=False)}')
        agent_plan_items.append("{ " + ", ".join(fields) + " }")
    agent_plan_js = ",\n    ".join(agent_plan_items)

    lines: list[str] = [
        "export const meta = {",
        f'  name: {json.dumps(_safe_meta_text(f"teamwork-{plan.preset.preset_id}"), ensure_ascii=False)},',
        f'  description: {json.dumps(_safe_meta_text(plan.preset.description), ensure_ascii=False)},',
        "  phases: [",
        f"    {phases_js},",
        "  ],",
        f"  maxConcurrent: {max_concurrent},",
        f"  tools: {json.dumps(used_tools, ensure_ascii=False)},",
        f"  patterns: {json.dumps(patterns, ensure_ascii=False)},",
        "  agentPlan: [",
        f"    {agent_plan_js},",
        "  ],",
        "};",
        "",
        "export default async function main() {",
        f"  const requirement = {json.dumps(req_clean, ensure_ascii=False)};",
        "  let contextNote = requirement;",
        "  let scoutData = null;",
        "  let planData = null;",
    ]

    if plan.preset.enable_scout:
        scout_prompt = (
            "Explore the relevant files, symbols, tests, and constraints for this task.\n\n"
            "Requirement:\n"
            + req_clean
            + "\n\n"
            + _handoff_instructions("scout — gather concise factual context without modifying files")
            + enc_suffix
        ).strip()
        lines.extend(
            [
                '  phase("Scout");',
                '  log("Dispatching scout for scope discovery");',
                "  const scoutResult = await agent({",
                f"    prompt: {json.dumps(scout_prompt, ensure_ascii=False)},",
                f"    agentId: {json.dumps(plan.scout_id)},",
                '    role: "scout",',
                '    label: "teamwork-scout",',
                '    phase: "Scout",',
                "    timeout: 180,",
                "  });",
                "  if (scoutResult && !scoutResult.error) {",
                "    scoutData = scoutResult;",
                '    contextNote = contextNote + "\\n\\nScout Findings:\\n" + formatPayload(scoutResult);',
                "  }",
            ]
        )

    if plan.preset.enable_planner:
        plan_prompt_prefix = (
            "Create a concise execution and verification plan based on the requirement and scout findings.\n\n"
            "Requirement:\n"
            + req_clean
            + "\n\n"
            + _handoff_instructions("planner — decompose work items, interfaces, and test criteria")
            + enc_suffix
        ).strip()
        lines.extend(
            [
                '  phase("Plan");',
                '  log("Dispatching planner for execution design");',
                "  const planResult = await agent({",
                f"    prompt: {json.dumps(plan_prompt_prefix, ensure_ascii=False)} + (scoutData ? \"\\n\\nScout Context:\\n\" + formatPayload(scoutData) : \"\"),",
                f"    agentId: {json.dumps(plan.planner_id)},",
                '    role: "planner",',
                '    label: "teamwork-plan",',
                '    phase: "Plan",',
                "    timeout: 300,",
                "  });",
                "  if (planResult && !planResult.error) {",
                "    planData = planResult;",
                '    contextNote = contextNote + "\\n\\nExecution Plan:\\n" + formatPayload(planResult);',
                "  }",
            ]
        )

    # Execute phase
    lines.extend(
        [
            '  phase("Execute");',
            '  log("Dispatching worker execution");',
        ]
    )
    if use_fanout:
        worker_descriptors: list[str] = []
        total_workers = len(plan.worker_ids)
        for idx, wid in enumerate(plan.worker_ids, 1):
            w_prompt = (
                f"Complete workstream {idx}/{total_workers} for the task below using the shared context.\n\n"
                "Shared Context:\n${input}\n\n"
                + _handoff_instructions(f"worker {idx} of {total_workers} — execute and produce verifiable output")
                + enc_suffix
            ).strip()
            worker_descriptors.append(
                "\n".join(
                    [
                        "    {",
                        f"      prompt: {json.dumps(w_prompt, ensure_ascii=False)},",
                        f"      agentId: {json.dumps(wid)},",
                        '      role: "worker",',
                        f"      label: {json.dumps(f'teamwork-worker-{idx}')},",
                        '      phase: "Execute",',
                        "      timeout: 360,",
                        "    }",
                    ]
                )
            )
        worker_block = ",\n".join(worker_descriptors)
        lines.extend(
            [
                "  const execResult = await fanout(",
                "    contextNote,",
                "    [",
                worker_block,
                "    ],",
                "    {",
                f"      synthesizerAgentId: {json.dumps(plan.synthesizer_id)},",
                '      synthesizerRole: "synthesizer",',
                '      label: "teamwork-execute",',
                "      timeout: 360,",
                "    }",
                "  );",
                "  if (execResult && execResult.error) {",
                '    return { error: execResult.error, fallback: true, stage: "Execute", partial: null };',
                "  }",
            ]
        )
    else:
        worker_timeout = 300 if plan.preset.preset_id == "quick" else 360
        w_prompt_prefix = (
            "Fulfill the user's request thoroughly and verify your work before returning.\n\n"
            "Requirement and Context:\n"
        )
        w_prompt_suffix = (
            "\n\n"
            + _handoff_instructions("worker — implement/analyze and verify completely")
            + enc_suffix
        )
        lines.extend(
            [
                "  const execResult = await agent({",
                f"    prompt: {json.dumps(w_prompt_prefix, ensure_ascii=False)} + contextNote + {json.dumps(w_prompt_suffix, ensure_ascii=False)},",
                f"    agentId: {json.dumps(plan.worker_ids[0])},",
                '    role: "worker",',
                '    label: "teamwork-worker-1",',
                '    phase: "Execute",',
                f"    timeout: {worker_timeout},",
                "  });",
                "  if (execResult && execResult.error) {",
                '    return { error: execResult.error, fallback: true, stage: "Execute", partial: null };',
                "  }",
            ]
        )

    # Verify phase
    verify_timeout = 240 if plan.preset.preset_id == "quick" else 300
    verifier_descriptors: list[str] = []
    for idx, rid in enumerate(plan.reviewer_ids, 1):
        verifier_descriptors.append(
            "\n".join(
                [
                    "      {",
                    f"        agentId: {json.dumps(rid)},",
                    '        role: "reviewer",',
                    '        focus: "Verify correctness, completeness, edge cases, and concrete evidence against the requirement",',
                    f"        timeout: {verify_timeout},",
                    "      }",
                ]
            )
        )
    verifier_block = ",\n".join(verifier_descriptors)
    lines.extend(
        [
            '  phase("Verify");',
            '  log("Running independent adversarial verification");',
            "  const verification = await verify(execResult, {",
            f"    criteria: {json.dumps('correctness, completeness, security, and evidence for: ' + req_clean[:200], ensure_ascii=False)},",
            "    verifiers: [",
            verifier_block,
            "    ],",
            f"    reviserAgentId: {json.dumps(plan.worker_ids[0])},",
            f"    maxRounds: {plan.preset.max_verify_rounds},",
            f"    timeout: {verify_timeout},",
            "  });",
            "  const verifiedOutput = verification && verification.output !== undefined ? verification.output : execResult;",
        ]
    )

    # Synthesize phase
    if plan.preset.require_synthesizer:
        synth_prompt_prefix = (
            "Synthesize the execution output and verification evidence into a final 5-component Teamwork handoff.\n\n"
            "Original Requirement:\n"
            + req_clean
            + "\n\nVerified Output:\n"
        )
        synth_prompt_mid = "\n\nVerification Summary:\n"
        synth_prompt_end = (
            "\n\nReturn JSON with keys: observation, logic_chain, caveats, conclusion, verification, deliverables, next_steps."
            + enc_suffix
        )
        lines.extend(
            [
                '  phase("Synthesize");',
                '  log("Synthesizing final Teamwork report and card summary");',
                "  const synthResult = await agent({",
                f"    prompt: {json.dumps(synth_prompt_prefix, ensure_ascii=False)} + formatPayload(verifiedOutput) + {json.dumps(synth_prompt_mid, ensure_ascii=False)} + formatPayload(verification) + {json.dumps(synth_prompt_end, ensure_ascii=False)},",
                f"    agentId: {json.dumps(plan.synthesizer_id)},",
                '    role: "synthesizer",',
                '    label: "teamwork-synthesize",',
                '    phase: "Synthesize",',
                "    timeout: 300,",
                "  });",
                "  const finalPayload = synthResult && !synthResult.error ? synthResult : verifiedOutput;",
            ]
        )
    else:
        lines.append("  const finalPayload = verifiedOutput;")

    lines.extend(
        [
            "  return buildCardResponse(",
            f"    {json.dumps(plan.preset.preset_id)},",
            "    finalPayload,",
            "    verifiedOutput,",
            "    verification,",
            "    scoutData,",
            "    planData",
            "  );",
            "}",
            "",
            "function formatPayload(value) {",
            "  if (value == null) return \"\";",
            '  if (typeof value === "string") return value;',
            "  return JSON.stringify(value, null, 2);",
            "}",
            "",
            "function normalizeHandoff(value) {",
            "  if (value && typeof value === \"object\" && !Array.isArray(value)) {",
            "    return {",
            "      observation: value.observation || \"\",",
            "      logic_chain: value.logic_chain || value.logicChain || \"\",",
            "      caveats: value.caveats || \"\",",
            "      conclusion: value.conclusion || value.summary || \"\",",
            "      verification: value.verification || \"\",",
            "    };",
            "  }",
            "  if (typeof value === \"string\") {",
            "    try {",
            "      const parsed = JSON.parse(value);",
            "      if (parsed && typeof parsed === \"object\" && !Array.isArray(parsed)) {",
            "        return normalizeHandoff(parsed);",
            "      }",
            "    } catch (e) {}",
            "    return { observation: \"\", logic_chain: \"\", caveats: \"\", conclusion: value, verification: \"\" };",
            "  }",
            "  return { observation: \"\", logic_chain: \"\", caveats: \"\", conclusion: formatPayload(value), verification: \"\" };",
            "}",
            "",
            "function buildCardResponse(presetId, finalPayload, verifiedOutput, verification, scoutData, planData) {",
            "  const handoff = normalizeHandoff(finalPayload);",
            "  const data = finalPayload && typeof finalPayload === \"object\" ? finalPayload : {};",
            "  const accepted = Boolean(verification && verification.accepted);",
            "  const verdict = accepted ? \"passed\" : \"needs_attention\";",
            '  const conclusionText = handoff.conclusion || data.conclusion || data.summary || "任务已完成，完整结果见报告。";',
            "  const findings = Array.isArray(data.findings) ? data.findings : [];",
            "  const verItems = [",
            "    {",
            '      status: accepted ? "passed" : "warning",',
            '      text: accepted ? "独立复核已通过。" : (verification && verification.feedback ? verification.feedback : "已完成复核检查。"),',
            "    },",
            "  ];",
            "  return {",
            "    card_summary: {",
            "      verdict,",
            "      conclusion: typeof conclusionText === \"string\" ? conclusionText : formatPayload(conclusionText),",
            "      findings,",
            "      verification: verItems,",
            "      deliverables: Array.isArray(data.deliverables) ? data.deliverables : [],",
            "      next_steps: Array.isArray(data.next_steps) ? data.next_steps : [],",
            "    },",
            "    result: {",
            "      preset: presetId,",
            "      handoff,",
            "      output: finalPayload,",
            "      verified_output: verifiedOutput,",
            "      scout: scoutData,",
            "      plan: planData,",
            "      verification,",
            "    },",
            "    verification,",
            "  };",
            "}",
        ]
    )

    return "\n".join(lines) + "\n"
