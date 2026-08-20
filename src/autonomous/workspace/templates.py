"""Deterministic Markdown templates for employee workspaces."""

from __future__ import annotations

from .models import EmployeeWorkspaceSource


def render_workspace_files(source: EmployeeWorkspaceSource) -> dict[str, bytes]:
    capabilities = ", ".join(source.capabilities) or "none declared"
    traits = ", ".join(source.personality_traits) or "none declared"
    model = source.model or "provider default"
    agents = f"""# Employee: {source.name}

You are GhostAP employee `{source.agent_id}`.

## Identity
- Role: {source.role or "custom"}
- Strengths: {capabilities}
- Tool/model: {source.tool}/{model}

## Current work
Read `NOW.md`, then `tasks/active.md`. Never infer an active task from chat history.

## Knowledge
Read `wiki/index.md` before opening individual pages. Cite page paths and source IDs.
Use `sources/manifest.yaml` to locate source records.

## Boundaries
- `AGENTS.md`, `IDENTITY.md`, `NOW.md`, `tasks/`, and `wiki/` are managed projections.
- Identity, task state, and source manifests are rebuilt from the Journal.

## Project
The assigned project root and its repository instructions are supplied per assignment.
"""
    identity = f"""# Identity

- Name: {source.name}
- Agent ID: `{source.agent_id}`
- Tenant: `{source.tenant_key}`
- Role: {source.role or "custom"}
- Persona: {source.persona or "Follow the assigned role and evidence."}
- Personality traits: {traits}
- Capabilities: {capabilities}
- Tool/model: {source.tool}/{model}
- Identity version: {source.identity_version}

This file is a rebuildable projection. Authority remains in the GhostAP Journal.
"""
    active = source.active_assignment_id or "none"
    checkpoint = source.checkpoint_ref or "none"
    now = f"""# Current work

- Active assignment: `{active}`
- Checkpoint reference: `{checkpoint}`
- Projection sequence: {source.projection_sequence}
- Knowledge generation: {source.knowledge_generation}

Read `tasks/active.md` for the assignment brief. Do not infer work from chat history.
"""
    purpose = f"""# Purpose

Maintain durable, source-linked knowledge that helps {source.name} perform the `{source.role or 'custom'}` role.
Preserve conclusions, decisions, evidence references, skills, and verification results. Do not preserve hidden reasoning or sensitive source bodies.
"""
    schema = """# Knowledge schema

Wiki pages use YAML frontmatter with: schema_version, page_id, kind, title,
source_ids, source_hashes, confidence, status, knowledge_generation, updated_at.
Confidence is one of observed, inferred, or verified. Every claim cites an authorized source ID.
"""
    task = (
        "No active assignment.\n"
        if not source.active_assignment_id
        else f"Active assignment: `{source.active_assignment_id}`. See the durable coordinator checkpoint.\n"
    )
    source_lines = ["schema_version: 1", "sources:"]
    for source_id, digest, kind, visibility in source.source_refs:
        source_lines.extend(
            (
                f"  - source_id: {source_id}",
                f"    hash: {digest}",
                f"    type: {kind}",
                f"    visibility: {visibility}",
            )
        )
    if not source.source_refs:
        source_lines.append("  []")
    files = {
        "workspace/AGENTS.md": agents,
        "workspace/IDENTITY.md": identity,
        "workspace/NOW.md": now,
        "workspace/purpose.md": purpose,
        "workspace/schema.md": schema,
        "workspace/wiki/index.md": "# Knowledge index\n\nNo published pages.\n",
        "workspace/wiki/overview.md": f"# Overview\n\nRole: {source.role or 'custom'}\nCapabilities: {capabilities}\n",
        "workspace/wiki/log.md": f"# Knowledge log\n\nGeneration: {source.knowledge_generation}\n",
        "workspace/tasks/active.md": "# Active assignment\n\n" + task,
        "workspace/tasks/archive/index.md": "# Assignment archive\n\nNo archived assignments.\n",
        "workspace/sources/manifest.yaml": "\n".join(source_lines) + "\n",
        "runtime/codex-home/AGENTS.md": agents,
    }
    return {path: text.encode("utf-8") for path, text in files.items()}


__all__ = ["render_workspace_files"]
